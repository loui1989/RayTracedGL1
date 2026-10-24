// Copyright (c) 2026 RTGL1 Metal backend contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "MetalScene.h"

#include "MetalComputePipelines.h"
#include "MetalTextureManager.h"

#include "PrimitiveRouting.h"
#include "RgException.h"
#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <span>

static_assert( sizeof( RgPrimitiveVertex ) == sizeof( RTGL1::ShVertex ),
               "Primitive vertices are copied to the GPU as is" );

namespace
{

// Grows a shared buffer to fit, with headroom, so it's not reallocated every frame
void EnsureBuffer( id< MTLDevice > device, id< MTLBuffer > __strong& buffer, size_t bytes, NSString* label )
{
    bytes = std::max< size_t >( bytes, 256 );
    if( buffer && buffer.length >= bytes )
    {
        return;
    }
    buffer       = [device newBufferWithLength:bytes + bytes / 2 options:MTLResourceStorageModeShared];
    buffer.label = label;
}

template< typename T >
void Upload( id< MTLDevice > device, id< MTLBuffer > __strong& buffer, const std::vector< T >& src, NSString* label )
{
    EnsureBuffer( device, buffer, src.size() * sizeof( T ), label );
    if( !src.empty() )
    {
        memcpy( buffer.contents, src.data(), src.size() * sizeof( T ) );
    }
}

// a new buffer with 'a' then 'b'
template< typename T >
id< MTLBuffer > MakeConcatenated( id< MTLDevice >         device,
                                  const std::vector< T >& a,
                                  const std::vector< T >& b,
                                  NSString*               label )
{
    const size_t bytes = std::max< size_t >( ( a.size() + b.size() ) * sizeof( T ), 256 );

    id< MTLBuffer > buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    buffer.label           = label;

    auto* dst = static_cast< uint8_t* >( buffer.contents );
    if( !a.empty() )
    {
        memcpy( dst, a.data(), a.size() * sizeof( T ) );
    }
    if( !b.empty() )
    {
        memcpy( dst + a.size() * sizeof( T ), b.data(), b.size() * sizeof( T ) );
    }
    return buffer;
}

void EnsureAccelerationStructure( id< MTLDevice >                         device,
                                  id< MTLAccelerationStructure > __strong& as,
                                  id< MTLBuffer > __strong&                scratch,
                                  MTLAccelerationStructureDescriptor*      desc,
                                  NSString*                                label )
{
    MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];

    if( !as || as.size < sizes.accelerationStructureSize )
    {
        as       = [device newAccelerationStructureWithSize:sizes.accelerationStructureSize +
                                                       sizes.accelerationStructureSize / 2];
        as.label = label;
    }
    if( !scratch || scratch.length < sizes.buildScratchBufferSize )
    {
        scratch = [device newBufferWithLength:std::max< NSUInteger >(
                                                  256, sizes.buildScratchBufferSize + sizes.buildScratchBufferSize / 2 )
                                      options:MTLResourceStorageModePrivate];
        scratch.label = [label stringByAppendingString:@" scratch"];
    }
}

// RgTransform is a row-major 3x4, MTLPackedFloat4x3 is 4 columns of 3 floats
MTLPackedFloat4x3 ToPackedFloat4x3( const RgTransform& t )
{
    MTLPackedFloat4x3 r;
    for( int c = 0; c < 4; c++ )
    {
        r.columns[ c ] = MTLPackedFloat3Make( t.matrix[ 0 ][ c ], t.matrix[ 1 ][ c ], t.matrix[ 2 ][ c ] );
    }
    return r;
}

constexpr RgTransform IdentityTransform = { {
    { 1, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, 0, 1, 0 },
} };

uint32_t FloatToUint8( float f )
{
    return uint32_t( std::clamp( f, 0.0f, 1.0f ) * 255.0f );
}

// GeomInfoManager::AccessLayerTexCoords
const RgFloat2D* AccessLayerTexCoords( const RgMeshPrimitiveInfo& info, uint32_t layerIndex )
{
    if( auto layers = RTGL1::pnext::find< RgMeshPrimitiveTextureLayersEXT >( &info ) )
    {
        switch( layerIndex )
        {
            case 1: return layers->pLayer1 ? layers->pLayer1->pTexCoord : nullptr;
            case 2: return layers->pLayer2 ? layers->pLayer2->pTexCoord : nullptr;
            case 3: return layers->pLayer3 ? layers->pLayer3->pTexCoord : nullptr;
            default: assert( 0 ); return nullptr;
        }
    }
    return nullptr;
}

}

RTGL1::MetalScene::MetalScene( id< MTLDevice > _device, id< MTLCommandQueue > _queue )
    : device{ _device }, queue{ _queue }
{
    for( auto& f : frames )
    {
        f.matchPrev = [device newBufferWithLength:sizeof( int32_t ) * MAX_GEOM_INFO_COUNT
                                          options:MTLResourceStorageModeShared];
        f.matchPrev.label = @"Geometry indices prev to cur";
        memset( f.matchPrev.contents, 0xFF, f.matchPrev.length );
    }

    // empty static buffers
    BeginStatic( true );
    SubmitStatic();
}

RTGL1::MetalScene::~MetalScene() = default;

void RTGL1::MetalScene::VertexStorage::Clear()
{
    vertices.clear();
    indices.clear();
    for( auto& l : texCoordLayers )
    {
        l.clear();
    }
}

// VertexCollectorFilterTypeFlags_GetForGeometry
auto RTGL1::MetalScene::MakeGroup( const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& prim ) -> Group
{
    PassThrough pt;
    if( prim.flags & RG_MESH_PRIMITIVE_ALPHA_TESTED )
    {
        pt = PassThrough::AlphaTested;
    }
    else if( ( prim.flags & RG_MESH_PRIMITIVE_WATER ) || ( mesh.flags & RG_MESH_FORCE_WATER ) ||
             ( prim.flags & RG_MESH_PRIMITIVE_GLASS ) || ( mesh.flags & RG_MESH_FORCE_GLASS ) ||
             ( prim.flags & RG_MESH_PRIMITIVE_GLASS_IF_SMOOTH ) || ( prim.flags & RG_MESH_PRIMITIVE_ACID ) )
    {
        pt = PassThrough::Refract;
    }
    else
    {
        pt = PassThrough::Opaque;
    }

    Visibility pv;
    if( mesh.flags & RG_MESH_FIRST_PERSON )
    {
        pv = Visibility::FirstPerson;
    }
    else if( mesh.flags & RG_MESH_FIRST_PERSON_VIEWER )
    {
        pv = Visibility::FirstPersonViewer;
    }
    else if( prim.flags & RG_MESH_PRIMITIVE_SKY_VISIBILITY )
    {
        pv = Visibility::World2;
    }
    else if( prim.flags & RG_MESH_PRIMITIVE_NO_SHADOW )
    {
        pv = Visibility::World1;
    }
    else
    {
        pv = Visibility::World0;
    }

    return Group{ pv, pt };
}

auto RTGL1::MetalScene::Store( VertexStorage& dst, const RgMeshPrimitiveInfo& prim ) -> std::optional< VertexRange >
{
    const bool     indexed    = prim.pIndices && prim.indexCount > 0;
    const uint32_t indexCount = indexed ? prim.indexCount : prim.vertexCount;
    if( !prim.pVertices || prim.vertexCount == 0 || indexCount < 3 || indexCount % 3 != 0 )
    {
        return std::nullopt;
    }

    auto r = VertexRange{
        .firstVertex      = uint32_t( dst.vertices.size() ),
        .firstIndex       = uint32_t( dst.indices.size() ),
        .vertexCount      = prim.vertexCount,
        .indexCount       = indexCount,
        .firstVertexLayer = {},
    };

    const auto* src = reinterpret_cast< const ShVertex* >( prim.pVertices );
    dst.vertices.insert( dst.vertices.end(), src, src + prim.vertexCount );

    if( indexed )
    {
        dst.indices.insert( dst.indices.end(), prim.pIndices, prim.pIndices + prim.indexCount );
    }
    else
    {
        for( uint32_t i = 0; i < prim.vertexCount; i++ )
        {
            dst.indices.push_back( i );
        }
    }

    for( uint32_t layer = 1; layer <= 3; layer++ )
    {
        auto& texCoords                 = dst.texCoordLayers[ layer - 1 ];
        r.firstVertexLayer[ layer - 1 ] = uint32_t( texCoords.size() );

        if( LayerExists( prim, layer ) )
        {
            if( const RgFloat2D* tc = AccessLayerTexCoords( prim, layer ) )
            {
                texCoords.insert( texCoords.end(), tc, tc + prim.vertexCount );
            }
        }
    }

    return r;
}

// The same as ASManager::AddMeshPrimitive
auto RTGL1::MetalScene::MakeGeomInfo( const RgMeshInfo&          mesh,
                                      const RgMeshPrimitiveInfo& prim,
                                      const VertexRange&         range,
                                      bool                       isDynamic,
                                      const MetalTextureManager& textures ) -> ShGeometryInstance
{
    const auto pbrInfo       = pnext::find< RgMeshPrimitivePBREXT >( &prim );
    const auto layerTextures = textures.GetTexturesForLayers( prim );
    const auto layerColors   = textures.GetColorForLayers( prim );

    auto g = ShGeometryInstance{};
    for( int c = 0; c < 4; c++ )
    {
        g.model_0[ c ] = mesh.transform.matrix[ 0 ][ c ];
        g.model_1[ c ] = mesh.transform.matrix[ 1 ][ c ];
        g.model_2[ c ] = mesh.transform.matrix[ 2 ][ c ];
    }

    g.flags = GetPrimitiveFlags( &mesh, prim, isDynamic );

    g.texture_base     = layerTextures[ 0 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_base_ORM = layerTextures[ 0 ].indices[ TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_INDEX ];
    g.texture_base_N   = layerTextures[ 0 ].indices[ TEXTURE_NORMAL_INDEX ];
    g.texture_base_E   = layerTextures[ 0 ].indices[ TEXTURE_EMISSIVE_INDEX ];
    g.texture_base_D   = layerTextures[ 0 ].indices[ TEXTURE_HEIGHT_INDEX ];

    g.texture_layer1 = layerTextures[ 1 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_layer2 = layerTextures[ 2 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_layer3 = layerTextures[ 3 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];

    g.colorFactor_base   = layerColors[ 0 ];
    g.colorFactor_layer1 = layerColors[ 1 ];
    g.colorFactor_layer2 = layerColors[ 2 ];
    g.colorFactor_layer3 = layerColors[ 3 ];

    g.baseVertexIndex     = range.firstVertex;
    g.baseIndexIndex      = range.firstIndex;
    g.prevBaseVertexIndex = UINT32_MAX;
    g.prevBaseIndexIndex  = UINT32_MAX;
    g.vertexCount         = range.vertexCount;
    g.indexCount          = range.indexCount;

    g.roughnessDefault_metallicDefault = ( FloatToUint8( pbrInfo ? pbrInfo->roughnessDefault : 1.0f ) << 0 ) |
                                         ( FloatToUint8( pbrInfo ? pbrInfo->metallicDefault : 0.0f ) << 8 );

    g.emissiveMult = Utils::Saturate( prim.emissive );

    g.firstVertex_Layer1 = range.firstVertexLayer[ 0 ];
    g.firstVertex_Layer2 = range.firstVertexLayer[ 1 ];
    g.firstVertex_Layer3 = range.firstVertexLayer[ 2 ];

    return g;
}

void RTGL1::MetalScene::BeginFrame( uint32_t frameIndex )
{
    ring = ( ring + 1 ) % RingSize;

    for( auto& r : retired )
    {
        r.framesLeft = r.framesLeft > 0 ? r.framesLeft - 1 : 0;
    }
    std::erase_if( retired, []( const Retired& r ) { return r.framesLeft == 0; } );

    dynamicStorage.Clear();
    for( auto& g : dynamicGeoms )
    {
        g.clear();
    }
    replacementInstances.clear();

    // static geometry is in every frame
    frameGeomCount = 0;
    for( const auto& g : staticGeoms )
    {
        frameGeomCount += g.size();
    }
}

bool RTGL1::MetalScene::AddDynamic( const RgMeshInfo&          mesh,
                                    const RgMeshPrimitiveInfo& prim,
                                    const PrimitiveUniqueID&   uniqueID,
                                    const MetalTextureManager& textures )
{
    if( frameGeomCount >= MAX_GEOM_INFO_COUNT )
    {
        debug::Error( "Too many geometry infos: the limit is {}", MAX_GEOM_INFO_COUNT );
        return false;
    }

    const auto range = Store( dynamicStorage, prim );
    if( !range )
    {
        return false;
    }

    const Group group = MakeGroup( mesh, prim );

    dynamicGeoms[ group.ToIndex() ].push_back( Geometry{
        .info            = MakeGeomInfo( mesh, prim, *range, true, textures ),
        .uniqueID        = uniqueID,
        .noMotionVectors = !!( prim.flags & RG_MESH_PRIMITIVE_NO_MOTION_VECTORS ),
        .alphaTested     = group.passThrough == PassThrough::AlphaTested,
        .bakedTransform  = mesh.transform,
    } );
    frameGeomCount++;
    return true;
}

void RTGL1::MetalScene::BeginStatic( bool withReplacements )
{
    assert( !makingStatic );
    makingStatic = true;

    // the frames in flight might still trace them
    auto retireBlas = [ this ]( const rgl::string_map< Replacement >& rs ) {
        auto old = Retired{ .resources = {}, .framesLeft = RingSize + 1 };
        for( const auto& [ name, r ] : rs )
        {
            for( const auto& b : r.blas )
            {
                old.resources.push_back( b.blas );
                old.resources.push_back( b.scratch );
            }
        }
        std::erase( old.resources, nil );
        retired.push_back( std::move( old ) );
    };

    staticStorage.Clear();
    for( auto& g : staticGeoms )
    {
        g.clear();
    }
    retireBlas( staticInstanced );
    staticInstanced.clear();

    if( withReplacements )
    {
        retireBlas( replacements );
        replacementStorage.Clear();
        replacements.clear();
        replacementsChanged = true;
    }
}

bool RTGL1::MetalScene::AddStatic( const RgMeshInfo&          mesh,
                                   const RgMeshPrimitiveInfo& prim,
                                   const PrimitiveUniqueID&   uniqueID,
                                   const MetalTextureManager& textures )
{
    assert( makingStatic );

    const auto range = Store( staticStorage, prim );
    if( !range )
    {
        return false;
    }

    const Group group = MakeGroup( mesh, prim );

    // relative to staticBase, until SubmitStatic
    ShGeometryInstance info = MakeGeomInfo( mesh, prim, *range, false, textures );

    staticGeoms[ group.ToIndex() ].push_back( Geometry{
        .info            = info,
        .uniqueID        = uniqueID,
        .noMotionVectors = !!( prim.flags & RG_MESH_PRIMITIVE_NO_MOTION_VECTORS ),
        .alphaTested     = group.passThrough == PassThrough::AlphaTested,
        .bakedTransform  = mesh.transform,
    } );
    return true;
}

void RTGL1::MetalScene::CacheInstanced( Replacement&               r,
                                        VertexStorage&             storage,
                                        const RgMeshPrimitiveInfo& prim,
                                        uint32_t                   index )
{
    // TODO: primitiveIndex must be sequential for now, as in ASManager::CacheReplacement
    assert( r.primitives.size() == index );

    const auto range = Store( storage, prim );
    if( !range )
    {
        debug::Warning( "Failed to upload vertex data of a replacement primitive" );
        // keep the indices sequential
        r.primitives.push_back( Replacement::Primitive{
            .range         = VertexRange{},
            .group         = Group{ Visibility::World0, PassThrough::Opaque },
            .alphaTested   = false,
            .geometryIndex = UINT32_MAX,
        } );
        return;
    }

    // a replacement's visibility is from its primitive, the mesh's flags are applied to an instance
    const Group group = MakeGroup( RgMeshInfo{}, prim );

    uint32_t geometryIndex = 0;
    for( const auto& p : r.primitives )
    {
        if( p.geometryIndex != UINT32_MAX && p.group == group )
        {
            geometryIndex++;
        }
    }

    r.primitives.push_back( Replacement::Primitive{
        .range         = *range,
        .group         = group,
        .alphaTested   = group.passThrough == PassThrough::AlphaTested,
        .geometryIndex = geometryIndex,
    } );
}

void RTGL1::MetalScene::CacheReplacement( std::string_view meshName, const RgMeshPrimitiveInfo& prim, uint32_t index )
{
    assert( makingStatic );
    CacheInstanced( replacements[ std::string( meshName ) ], replacementStorage, prim, index );
    replacementsChanged = true;
}

void RTGL1::MetalScene::CacheStaticInstanced( std::string_view name, const RgMeshPrimitiveInfo& prim, uint32_t index )
{
    assert( makingStatic );
    CacheInstanced( staticInstanced[ std::string( name ) ], staticStorage, prim, index );
}

auto RTGL1::MetalScene::Offset( const VertexRange& r, const VertexRange& base ) -> VertexRange
{
    return VertexRange{
        .firstVertex      = r.firstVertex + base.firstVertex,
        .firstIndex       = r.firstIndex + base.firstIndex,
        .vertexCount      = r.vertexCount,
        .indexCount       = r.indexCount,
        .firstVertexLayer = {
            r.firstVertexLayer[ 0 ] + base.firstVertexLayer[ 0 ],
            r.firstVertexLayer[ 1 ] + base.firstVertexLayer[ 1 ],
            r.firstVertexLayer[ 2 ] + base.firstVertexLayer[ 2 ],
        },
    };
}

void RTGL1::MetalScene::BuildInstancedBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                                            Replacement&                                 r,
                                            const VertexRange&                           base )
{
    // geometries in the mesh's local space: instanced with mesh transforms
    for( uint32_t gi = 0; gi < GroupCount; gi++ )
    {
        std::vector< Geometry > geoms;
        for( const auto& p : r.primitives )
        {
            if( p.geometryIndex != UINT32_MAX && p.group.ToIndex() == gi )
            {
                assert( p.geometryIndex == geoms.size() );

                const VertexRange range = Offset( p.range, base );

                auto g            = ShGeometryInstance{};
                g.baseVertexIndex = range.firstVertex;
                g.baseIndexIndex  = range.firstIndex;
                g.indexCount      = range.indexCount;
                geoms.push_back( Geometry{
                    .info            = g,
                    .uniqueID        = PrimitiveUniqueID{ RgMeshInfo{}, RgMeshPrimitiveInfo{} },
                    .noMotionVectors = false,
                    .alphaTested     = p.alphaTested,
                    .bakedTransform  = IdentityTransform,
                } );
            }
        }
        BuildBlas( enc, r.blas[ gi ], geoms, staticVertices, staticIndices, false, false, @"Instanced BLAS" );
    }
}

void RTGL1::MetalScene::BuildBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                                   BuiltBlas&                                   dst,
                                   const std::vector< Geometry >&               geoms,
                                   id< MTLBuffer >                              vertices,
                                   id< MTLBuffer >                              indices,
                                   bool                                         bakeTransforms,
                                   bool                                         fastBuild,
                                   NSString*                                    label )
{
    if( geoms.empty() )
    {
        dst.blas = nil;
        return;
    }

    if( bakeTransforms )
    {
        std::vector< MTLPackedFloat4x3 > transforms;
        transforms.reserve( geoms.size() );
        for( const Geometry& g : geoms )
        {
            transforms.push_back( ToPackedFloat4x3( g.bakedTransform ) );
        }
        Upload( device, dst.transforms, transforms, @"BLAS transforms" );
    }

    NSMutableArray< MTLAccelerationStructureGeometryDescriptor* >* descs =
        [NSMutableArray arrayWithCapacity:geoms.size()];

    for( size_t i = 0; i < geoms.size(); i++ )
    {
        const Geometry& g = geoms[ i ];

        auto* d = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        // indices are relative to the vertex buffer offset, as on Vulkan
        d.vertexBuffer       = vertices;
        d.vertexBufferOffset = NSUInteger( g.info.baseVertexIndex ) * sizeof( ShVertex );
        d.vertexStride       = sizeof( ShVertex );
        d.vertexFormat       = MTLAttributeFormatFloat3;
        d.indexBuffer        = indices;
        d.indexBufferOffset  = NSUInteger( g.info.baseIndexIndex ) * sizeof( uint32_t );
        d.indexType          = MTLIndexTypeUInt32;
        d.triangleCount      = g.info.indexCount / 3;
        if( bakeTransforms )
        {
            d.transformationMatrixBuffer       = dst.transforms;
            d.transformationMatrixBufferOffset = i * sizeof( MTLPackedFloat4x3 );
        }
        // alpha tested ones are resolved by the shader through an intersection query
        d.opaque                                       = !g.alphaTested;
        d.allowDuplicateIntersectionFunctionInvocation = NO;
        [descs addObject:d];
    }

    auto* desc                = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = descs;
    desc.usage = fastBuild ? MTLAccelerationStructureUsagePreferFastBuild : MTLAccelerationStructureUsageNone;

    id< MTLAccelerationStructure > blas    = dst.blas;
    id< MTLBuffer >                scratch = dst.scratch;
    EnsureAccelerationStructure( device, blas, scratch, desc, label );
    dst.blas    = blas;
    dst.scratch = scratch;

    [enc buildAccelerationStructure:dst.blas descriptor:desc scratchBuffer:dst.scratch scratchBufferOffset:0];
}

void RTGL1::MetalScene::SubmitStatic()
{
    assert( makingStatic );
    makingStatic = false;

    // new resources, as the frames in flight might still read the old ones
    {
        auto old = Retired{ .resources = {}, .framesLeft = RingSize + 1 };
        old.resources.push_back( staticVertices );
        old.resources.push_back( staticIndices );
        for( auto& b : staticTexCoordLayers )
        {
            old.resources.push_back( b );
        }
        auto retire = [ &old ]( BuiltBlas& b ) {
            old.resources.push_back( b.blas );
            old.resources.push_back( b.scratch );
            old.resources.push_back( b.transforms );
            b = {};
        };
        for( auto& b : staticBlas )
        {
            retire( b );
        }
        std::erase( old.resources, nil );
        retired.push_back( std::move( old ) );
    }

    // the static scene is after the replacements
    staticBase = VertexRange{
        .firstVertex      = uint32_t( replacementStorage.vertices.size() ),
        .firstIndex       = uint32_t( replacementStorage.indices.size() ),
        .vertexCount      = 0,
        .indexCount       = 0,
        .firstVertexLayer = {
            uint32_t( replacementStorage.texCoordLayers[ 0 ].size() ),
            uint32_t( replacementStorage.texCoordLayers[ 1 ].size() ),
            uint32_t( replacementStorage.texCoordLayers[ 2 ].size() ),
        },
    };
    for( auto& group : staticGeoms )
    {
        for( Geometry& g : group )
        {
            g.info.baseVertexIndex += staticBase.firstVertex;
            g.info.baseIndexIndex += staticBase.firstIndex;
            g.info.firstVertex_Layer1 += staticBase.firstVertexLayer[ 0 ];
            g.info.firstVertex_Layer2 += staticBase.firstVertexLayer[ 1 ];
            g.info.firstVertex_Layer3 += staticBase.firstVertexLayer[ 2 ];
        }
    }

    // the static buffers: replacements, then the static scene
    staticVertices =
        MakeConcatenated( device, replacementStorage.vertices, staticStorage.vertices, @"Static vertices" );
    staticIndices = MakeConcatenated( device, replacementStorage.indices, staticStorage.indices, @"Static indices" );
    for( int l = 0; l < 3; l++ )
    {
        staticTexCoordLayers[ l ] = MakeConcatenated( device,
                                                      replacementStorage.texCoordLayers[ l ],
                                                      staticStorage.texCoordLayers[ l ],
                                                      @"Static tex coords" );
    }

    id< MTLCommandBuffer > cmd = [queue commandBuffer];
    cmd.label                  = @"Static geometry";

    id< MTLAccelerationStructureCommandEncoder > enc = [cmd accelerationStructureCommandEncoder];
    enc.label                                        = @"Build static BLAS";

    if( replacementsChanged )
    {
        for( auto& [ name, r ] : replacements )
        {
            BuildInstancedBlas( enc, r, VertexRange{} );
        }
        replacementsChanged = false;
    }
    for( auto& [ name, r ] : staticInstanced )
    {
        BuildInstancedBlas( enc, r, staticBase );
    }

    for( uint32_t gi = 0; gi < GroupCount; gi++ )
    {
        BuildBlas( enc, staticBlas[ gi ], staticGeoms[ gi ], staticVertices, staticIndices, true, false, @"Static BLAS" );
    }

    [enc endEncoding];
    [cmd commit];

    uint64_t triangles = 0;
    for( const auto& g : staticGeoms )
    {
        for( const auto& geom : g )
        {
            triangles += geom.info.indexCount / 3;
        }
    }
    if( triangles > 0 || !replacements.empty() )
    {
        debug::Info( "Metal backend: static geometry: {} triangles; {} replacements: {} triangles",
                     triangles,
                     replacements.size(),
                     replacementStorage.indices.size() / 3 );
    }
}

bool RTGL1::MetalScene::AddReplacement( const RgMeshInfo&          mesh,
                                        const RgMeshPrimitiveInfo& replacementPrim,
                                        const MetalTextureManager& textures )
{
    if( Utils::IsCstrEmpty( mesh.pMeshName ) )
    {
        return false;
    }
    const Replacement* r    = find_p( replacements, mesh.pMeshName );
    VertexRange        base = {};
    if( !r )
    {
        r    = find_p( staticInstanced, mesh.pMeshName );
        base = staticBase;
    }
    if( !r || replacementPrim.primitiveIndexInMesh >= r->primitives.size() )
    {
        assert( 0 );
        return false;
    }
    const Replacement::Primitive& p = r->primitives[ replacementPrim.primitiveIndexInMesh ];
    if( p.geometryIndex == UINT32_MAX )
    {
        return false;
    }
    if( frameGeomCount >= MAX_GEOM_INFO_COUNT )
    {
        debug::Error( "Too many geometry infos: the limit is {}", MAX_GEOM_INFO_COUNT );
        return false;
    }

    // the instance of this object and group
    auto inst = std::ranges::find_if( replacementInstances, [ & ]( const ReplacementInstance& ri ) {
        return ri.replacement == r && ri.group == p.group && ri.uniqueObjectID == mesh.uniqueObjectID;
    } );
    if( inst == replacementInstances.end() )
    {
        uint32_t count = 0;
        for( const auto& other : r->primitives )
        {
            count += ( other.geometryIndex != UINT32_MAX && other.group == p.group ) ? 1 : 0;
        }

        replacementInstances.push_back( ReplacementInstance{
            .replacement    = r,
            .base           = base,
            .group          = p.group,
            .transform      = mesh.transform,
            .meshFlags      = mesh.flags,
            .uniqueObjectID = mesh.uniqueObjectID,
            .geometries     = std::vector< std::optional< Geometry > >( count ),
        } );
        inst = std::prev( replacementInstances.end() );
    }

    auto& dst = inst->geometries[ p.geometryIndex ];
    if( dst )
    {
        debug::Warning( "Replacement primitive {} of {} was already added",
                        replacementPrim.primitiveIndexInMesh,
                        mesh.pMeshName );
        return false;
    }

    dst = Geometry{
        .info            = MakeGeomInfo( mesh, replacementPrim, Offset( p.range, base ), false, textures ),
        .uniqueID        = PrimitiveUniqueID{ mesh, replacementPrim },
        .noMotionVectors = !!( replacementPrim.flags & RG_MESH_PRIMITIVE_NO_MOTION_VECTORS ),
        .alphaTested     = p.alphaTested,
        .bakedTransform  = mesh.transform,
    };
    frameGeomCount++;
    return true;
}

bool RTGL1::MetalScene::Build( id< MTLCommandBuffer > cmd, uint32_t rayCullMaskWorld )
{
    FrameData& f = Cur();

    Upload( device, f.vertices, dynamicStorage.vertices, @"Dynamic vertices" );
    Upload( device, f.indices, dynamicStorage.indices, @"Dynamic indices" );
    for( int l = 0; l < 3; l++ )
    {
        Upload( device, f.texCoordLayers[ l ], dynamicStorage.texCoordLayers[ l ], @"Dynamic tex coords" );
    }

    // dynamic BLAS-es
    if( std::ranges::any_of( dynamicGeoms, []( const auto& g ) { return !g.empty(); } ) )
    {
        id< MTLAccelerationStructureCommandEncoder > enc = [cmd accelerationStructureCommandEncoder];
        enc.label                                        = @"Build dynamic BLAS";
        for( uint32_t gi = 0; gi < GroupCount; gi++ )
        {
            BuildBlas( enc, f.dynamicBlas[ gi ], dynamicGeoms[ gi ], f.vertices, f.indices, true, true, @"Dynamic BLAS" );
        }
        [enc endEncoding];
    }
    else
    {
        for( auto& b : f.dynamicBlas )
        {
            b.blas = nil;
        }
    }

    // TLAS instances, and ShGeometryInstance-s in the order of the instances
    std::vector< MTLAccelerationStructureUserIDInstanceDescriptor > instances;
    std::vector< ShGeometryInstance >                                geomInfos;
    std::vector< const Geometry* >                                  geomSources;
    f.usedBlas.clear();

    // ASManager::SetupTLASInstance
    auto addInstance = [ & ]( id< MTLAccelerationStructure >     blas,
                              Group                              group,
                              RgMeshInfoFlags                    meshFlags,
                              const RgTransform&                 transform,
                              std::span< const Geometry* const > geoms ) {
        if( !blas || geoms.empty() )
        {
            return;
        }

        // inherit some bits from a mesh
        Visibility pv = group.visibility;
        if( meshFlags & RG_MESH_FIRST_PERSON )
        {
            pv = Visibility::FirstPerson;
        }
        else if( meshFlags & RG_MESH_FIRST_PERSON_VIEWER )
        {
            pv = Visibility::FirstPersonViewer;
        }
        // SHIPPING_HACK
        PassThrough pt = group.passThrough;
        if( pt == PassThrough::Opaque && ( meshFlags & ( RG_MESH_FORCE_GLASS | RG_MESH_FORCE_WATER ) ) )
        {
            pt = PassThrough::Refract;
        }

        uint32_t mask = 0, customIndex = 0;
        switch( pv )
        {
            case Visibility::FirstPerson:
                mask        = INSTANCE_MASK_FIRST_PERSON;
                customIndex = INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON;
                break;
            case Visibility::FirstPersonViewer:
                mask        = INSTANCE_MASK_FIRST_PERSON_VIEWER;
                customIndex = INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER;
                break;
            case Visibility::World0:
                mask = INSTANCE_MASK_WORLD_0;
                if( !( rayCullMaskWorld & INSTANCE_MASK_WORLD_0 ) )
                {
                    return;
                }
                break;
            case Visibility::World1: mask = INSTANCE_MASK_WORLD_1; break;
            case Visibility::World2:
                mask        = INSTANCE_MASK_WORLD_2;
                customIndex = INSTANCE_CUSTOM_INDEX_FLAG_SKY;
                if( !( rayCullMaskWorld & INSTANCE_MASK_WORLD_2 ) )
                {
                    return;
                }
                break;
        }
        if( pt == PassThrough::Refract && pv != Visibility::FirstPerson && pv != Visibility::FirstPersonViewer )
        {
            // completely rewrite the mask, as in ASManager
            mask = INSTANCE_MASK_REFRACT;
        }

        const auto geometryBase = uint32_t( geomInfos.size() );
        if( geometryBase + geoms.size() > MAX_GEOM_INFO_COUNT )
        {
            debug::Error( "Too many geometry infos: the limit is {}", MAX_GEOM_INFO_COUNT );
            return;
        }

        auto inst                 = MTLAccelerationStructureUserIDInstanceDescriptor{};
        inst.transformationMatrix = ToPackedFloat4x3( transform );
        // VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR, VK_GEOMETRY_INSTANCE_FORCE_(NO_)OPAQUE_BIT_KHR
        inst.options = MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise |
                       ( pt == PassThrough::AlphaTested ? MTLAccelerationStructureInstanceOptionNonOpaque
                                                        : MTLAccelerationStructureInstanceOptionOpaque );
        inst.mask                            = mask;
        inst.intersectionFunctionTableOffset = 0;
        inst.accelerationStructureIndex      = uint32_t( f.usedBlas.size() );
        // RAY_QUERY_GROUPED_INSTANCES in RaygenCommon.h
        inst.userID = ( geometryBase << 4 ) | customIndex;

        instances.push_back( inst );
        f.usedBlas.push_back( blas );

        for( const Geometry* g : geoms )
        {
            geomInfos.push_back( g->info );
            geomSources.push_back( g );
        }
    };

    auto toPointers = []( const std::vector< Geometry >& v ) {
        std::vector< const Geometry* > r;
        r.reserve( v.size() );
        for( const auto& g : v )
        {
            r.push_back( &g );
        }
        return r;
    };

    for( uint32_t gi = 0; gi < GroupCount; gi++ )
    {
        const auto group = Group{ Visibility( gi / 3 ), PassThrough( gi % 3 ) };
        addInstance( staticBlas[ gi ].blas, group, 0, IdentityTransform, toPointers( staticGeoms[ gi ] ) );
    }
    for( const ReplacementInstance& ri : replacementInstances )
    {
        std::vector< const Geometry* > geoms;
        for( const auto& g : ri.geometries )
        {
            if( !g )
            {
                break;
            }
            geoms.push_back( &*g );
        }
        if( geoms.size() != ri.geometries.size() )
        {
            debug::Warning( "Not all primitives of a replacement were uploaded, ignoring the replacement instance" );
            continue;
        }
        addInstance( ri.replacement->blas[ ri.group.ToIndex() ].blas, ri.group, ri.meshFlags, ri.transform, geoms );
    }
    for( uint32_t gi = 0; gi < GroupCount; gi++ )
    {
        const auto group = Group{ Visibility( gi / 3 ), PassThrough( gi % 3 ) };
        addInstance( f.dynamicBlas[ gi ].blas, group, 0, IdentityTransform, toPointers( dynamicGeoms[ gi ] ) );
    }

    // GeomInfoManager: match with the previous frame, by unique IDs
    {
        auto* matchPrev = static_cast< int32_t* >( f.matchPrev.contents );
        memset( matchPrev, 0xFF, sizeof( int32_t ) * std::max( prevGeomCount, uint32_t( geomInfos.size() ) ) );

        auto curInfos = rgl::unordered_map< PrimitiveUniqueID, PrevInfo >{};
        curInfos.reserve( geomInfos.size() );

        for( uint32_t i = 0; i < geomInfos.size(); i++ )
        {
            ShGeometryInstance& dst = geomInfos[ i ];
            const Geometry&     src = *geomSources[ i ];

            auto prev = prevInfos.find( src.uniqueID );

            // GeomInfoManager::FindPrevFrameData
            if( !src.noMotionVectors && prev != prevInfos.end() && prev->second.vertexCount == dst.vertexCount &&
                prev->second.indexCount == dst.indexCount )
            {
                dst.prevBaseVertexIndex = prev->second.baseVertexIndex;
                dst.prevBaseIndexIndex  = prev->second.baseIndexIndex;
                memcpy( dst.prevModel_0, prev->second.model_0, sizeof( dst.prevModel_0 ) );
                memcpy( dst.prevModel_1, prev->second.model_1, sizeof( dst.prevModel_1 ) );
                memcpy( dst.prevModel_2, prev->second.model_2, sizeof( dst.prevModel_2 ) );
            }
            else
            {
                dst.prevBaseVertexIndex = UINT32_MAX;
            }

            // Metal: for an external temporal denoiser (CmDenoiserReactive), as such geometry has no history:
            // just appeared, changed its shape (an animation frame of a model) or its texture (of a sprite)
            if( prev == prevInfos.end() || prev->second.vertexCount != dst.vertexCount ||
                prev->second.indexCount != dst.indexCount || prev->second.texture_base != dst.texture_base ||
                prev->second.texture_base_E != dst.texture_base_E )
            {
                dst.flags |= GEOM_INST_FLAG_NO_HISTORY;
            }

            // GeomInfoManager::CopyFromStaging: the previous frame's ShGeometryInstance index -> the current one
            if( prev != prevInfos.end() )
            {
                matchPrev[ prev->second.globalIndex ] = int32_t( i );
            }

            auto& next = curInfos[ src.uniqueID ];
            next       = PrevInfo{
                      .baseVertexIndex = dst.baseVertexIndex,
                      .baseIndexIndex  = dst.baseIndexIndex,
                      .vertexCount     = dst.vertexCount,
                      .indexCount      = dst.indexCount,
                      .model_0         = {},
                      .model_1         = {},
                      .model_2         = {},
                      .globalIndex     = i,
                      .texture_base    = dst.texture_base,
                      .texture_base_E  = dst.texture_base_E,
            };
            memcpy( next.model_0, dst.model_0, sizeof( dst.model_0 ) );
            memcpy( next.model_1, dst.model_1, sizeof( dst.model_1 ) );
            memcpy( next.model_2, dst.model_2, sizeof( dst.model_2 ) );
        }

        prevInfos     = std::move( curInfos );
        prevGeomCount = uint32_t( geomInfos.size() );
    }

    Upload( device, f.geometries, geomInfos, @"Geometry instances" );

    if( instances.empty() )
    {
        f.tlas = nil;
        return false;
    }

    Upload( device, f.instances, instances, @"Instances" );

    auto* tlasDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlasDesc.instancedAccelerationStructures =
        [NSArray arrayWithObjects:f.usedBlas.data() count:f.usedBlas.size()];
    tlasDesc.instanceCount            = NSUInteger( instances.size() );
    tlasDesc.instanceDescriptorBuffer = f.instances;
    tlasDesc.instanceDescriptorType   = MTLAccelerationStructureInstanceDescriptorTypeUserID;
    tlasDesc.usage                    = MTLAccelerationStructureUsagePreferFastBuild;

    id< MTLAccelerationStructure > tlas    = f.tlas;
    id< MTLBuffer >                scratch = f.scratchTlas;
    EnsureAccelerationStructure( device, tlas, scratch, tlasDesc, @"TLAS" );
    f.tlas        = tlas;
    f.scratchTlas = scratch;

    // a separate encoder, so that the TLAS build waits for the BLAS builds
    id< MTLAccelerationStructureCommandEncoder > enc = [cmd accelerationStructureCommandEncoder];
    enc.label                                        = @"Build TLAS";
    [enc buildAccelerationStructure:f.tlas descriptor:tlasDesc scratchBuffer:f.scratchTlas scratchBufferOffset:0];
    [enc endEncoding];

    return true;
}

void RTGL1::MetalScene::PatchStaticTextures( const PrimitiveUniqueID& uniqueID, const MaterialTextures& textures )
{
    for( auto& group : staticGeoms )
    {
        for( Geometry& g : group )
        {
            if( g.uniqueID == uniqueID )
            {
                g.info.texture_base     = textures.indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
                g.info.texture_base_ORM = textures.indices[ TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_INDEX ];
                g.info.texture_base_N   = textures.indices[ TEXTURE_NORMAL_INDEX ];
                g.info.texture_base_E   = textures.indices[ TEXTURE_EMISSIVE_INDEX ];
                g.info.texture_base_D   = textures.indices[ TEXTURE_HEIGHT_INDEX ];
                return;
            }
        }
    }
}

void RTGL1::MetalScene::Bind( MetalBindingTable& b ) const
{
    const FrameData& f = Cur();
    const FrameData& p = Prev();

    b.SetAccelerationStructure( "topLevelAS", f.tlas, f.usedBlas );
    // Fluid_Particles
    b.SetAccelerationStructure( "g_topLevelAS", f.tlas, f.usedBlas );

    b.SetBuffer( "VertexBufferStatic", staticVertices );
    b.SetBuffer( "IndexBufferStatic", staticIndices );
    b.SetBuffer( "VertexBufferDynamic", f.vertices );
    b.SetBuffer( "IndexBufferDynamic", f.indices );
    // the previous frame's dynamic data, prevBaseVertexIndex-s point to them
    b.SetBuffer( "PrevPositionsBufferDynamic", p.vertices ? p.vertices : f.vertices );
    b.SetBuffer( "PrevIndexBufferDynamic", p.indices ? p.indices : f.indices );

    b.SetBuffer( "StaticTexCoordLayer1", staticTexCoordLayers[ 0 ] );
    b.SetBuffer( "StaticTexCoordLayer2", staticTexCoordLayers[ 1 ] );
    b.SetBuffer( "StaticTexCoordLayer3", staticTexCoordLayers[ 2 ] );
    b.SetBuffer( "DynamicTexCoordLayer1", f.texCoordLayers[ 0 ] );
    b.SetBuffer( "DynamicTexCoordLayer2", f.texCoordLayers[ 1 ] );
    b.SetBuffer( "DynamicTexCoordLayer3", f.texCoordLayers[ 2 ] );

    b.SetBuffer( "GeometryInstances", f.geometries );
    b.SetBuffer( "GeometryIndicesPrevToCur", f.matchPrev );
}
