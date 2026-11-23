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
#include "MetalProfiler.h"

#include "MetalComputePipelines.h"
#include "MetalTextureManager.h"

#include "PrimitiveRouting.h"
#include "RgException.h"
#include "Utils.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>

static_assert( sizeof( RgPrimitiveVertex ) == sizeof( RTGL1::ShVertex ),
               "Primitive vertices are copied to the GPU as is" );

namespace RTGL1
{
// Thief RT (0032): a test of how the scene is laid out for tracing, from the game's panel
// (rgThiefTracingTest), to find what makes long rays slow: 0 as it is; 1 see-through (alpha
// tested) triangles traced as solid; 2 the kept map in one piece (a BLAS for each group, not
// for each 24 m cell); 3 in 48 m pieces; 4 still objects taken into the map's pieces (not each
// in a BLAS of its own); 5 both 4 and 3.
std::atomic< int > g_thiefTracingTest{ 0 };
}

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

// Thief RT (0028): as EnsureBuffer, its first 'keep' bytes kept (the arena's)
void EnsureKeeping( id< MTLDevice > device, id< MTLBuffer > __strong& buffer, size_t bytes, size_t keep, NSString* label )
{
    bytes = std::max< size_t >( bytes, 256 );
    if( buffer && buffer.length >= bytes )
    {
        return;
    }
    id< MTLBuffer > grown = [device newBufferWithLength:bytes + bytes / 2 options:MTLResourceStorageModeShared];
    grown.label           = label;
    if( buffer && keep > 0 )
    {
        memcpy( grown.contents, buffer.contents, std::min< size_t >( keep, buffer.length ) );
    }
    buffer = grown;
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

// Raze RT: what a BLAS sees of a geometry -- its positions, indices and transform
// (and 'local': without the transform -- the shape alone, 0 for an untransformed geometry).
// Four independent lanes of 64-bit words: one chain of a multiply per 32-bit word took
// ~2 ms a frame over a few voxel models' 300k vertices; this, ~0.4 ms
namespace
{
inline uint64_t KeyMix( uint64_t h, uint64_t w )
{
    h ^= w;
    h *= 0xBF58476D1CE4E5B9ull;
    return h ^ ( h >> 29 );
}
}
uint64_t ContentKey( const RTGL1::ShVertex* v, uint32_t vertexCount, const uint32_t* idx, uint32_t indexCount, const RgTransform& t, uint64_t* local = nullptr )
{
    uint64_t a = 0x9E3779B97F4A7C15ull ^ ( uint64_t( vertexCount ) << 32 ) ^ indexCount;
    uint64_t b = 0xC2B2AE3D27D4EB4Full, c = 0x165667B19E3779F9ull, d = 0x27D4EB2F165667C5ull;

    uint32_t i = 0;
    for( ; i + 2 <= vertexCount; i += 2 )
    {
        uint64_t xy0, xy1;
        uint32_t z0, z1;
        memcpy( &xy0, &v[ i ].position[ 0 ], 8 );
        memcpy( &z0, &v[ i ].position[ 2 ], 4 );
        memcpy( &xy1, &v[ i + 1 ].position[ 0 ], 8 );
        memcpy( &z1, &v[ i + 1 ].position[ 2 ], 4 );
        a = KeyMix( a, xy0 );
        b = KeyMix( b, uint64_t( z0 ) | ( uint64_t( z1 ) << 32 ) );
        c = KeyMix( c, xy1 );
    }
    for( ; i < vertexCount; i++ )
    {
        uint64_t xy;
        uint32_t z;
        memcpy( &xy, &v[ i ].position[ 0 ], 8 );
        memcpy( &z, &v[ i ].position[ 2 ], 4 );
        a = KeyMix( a, xy );
        d = KeyMix( d, z );
    }
    uint32_t j = 0;
    for( ; j + 8 <= indexCount; j += 8 )
    {
        uint64_t w[ 4 ];
        memcpy( w, idx + j, sizeof( w ) );
        a = KeyMix( a, w[ 0 ] );
        b = KeyMix( b, w[ 1 ] );
        c = KeyMix( c, w[ 2 ] );
        d = KeyMix( d, w[ 3 ] );
    }
    for( ; j < indexCount; j++ )
    {
        d = KeyMix( d, idx[ j ] );
    }
    uint64_t h = KeyMix( KeyMix( KeyMix( a, b ), c ), d );

    if( local )
    {
        *local = memcmp( &t, &IdentityTransform, sizeof( t ) ) != 0 ? ( h | 1 ) : 0;
    }
    uint64_t m[ 6 ];
    memcpy( m, t.matrix, sizeof( m ) );
    for( uint64_t w : m )
    {
        h = KeyMix( h, w );
    }
    return h ? h : 1;
}

// the cell a kept geometry is put in for the test (StableCell's three 21-bit fields)
int64_t TestCell( int64_t cell, int test )
{
    if( test == 2 )
    {
        return 0;
    }
    if( test != 3 && test != 5 )
    {
        return cell;
    }
    int64_t key = 0;
    for( int r = 2; r >= 0; r-- )
    {
        int64_t q = ( cell >> ( 21 * r ) ) & 0x1FFFFF;
        if( q & 0x100000 )
        {
            q -= 0x200000;
        }
        q   = q >= 0 ? q / 2 : -( ( -q + 1 ) / 2 );
        key = ( key << 21 ) | ( q & 0x1FFFFF );
    }
    return key;
}

// Raze RT: the piece of the map a geometry is in, for the kept BLAS-es (MetalScene::StableGroup):
// a cell of this many metres, by the middle of its bounds
constexpr float StableCellSize = 24.0f;
int64_t StableCell( const RTGL1::ShVertex* v, uint32_t vertexCount, const RgTransform& t )
{
    if( vertexCount == 0 )
    {
        return 0;
    }
    float lo[ 3 ] = { v[ 0 ].position[ 0 ], v[ 0 ].position[ 1 ], v[ 0 ].position[ 2 ] };
    float hi[ 3 ] = { lo[ 0 ], lo[ 1 ], lo[ 2 ] };
    for( uint32_t i = 1; i < vertexCount; i++ )
    {
        for( int k = 0; k < 3; k++ )
        {
            lo[ k ] = std::min( lo[ k ], v[ i ].position[ k ] );
            hi[ k ] = std::max( hi[ k ], v[ i ].position[ k ] );
        }
    }
    const float c[ 3 ] = { ( lo[ 0 ] + hi[ 0 ] ) * 0.5f, ( lo[ 1 ] + hi[ 1 ] ) * 0.5f, ( lo[ 2 ] + hi[ 2 ] ) * 0.5f };
    int64_t     key    = 0;
    for( int r = 0; r < 3; r++ )
    {
        const float w = t.matrix[ r ][ 0 ] * c[ 0 ] + t.matrix[ r ][ 1 ] * c[ 1 ] + t.matrix[ r ][ 2 ] * c[ 2 ] + t.matrix[ r ][ 3 ];
        const auto  q = int64_t( std::floor( std::clamp( w / StableCellSize, -1.0e6f, 1.0e6f ) ) ) & 0x1FFFFF;
        key           = ( key << 21 ) | q;
    }
    return key;
}

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

// the texture indices of a geometry instance, from its layers' materials
void ApplyTextures( RTGL1::ShGeometryInstance& g, const std::array< RTGL1::MaterialTextures, 4 >& layerTextures )
{
    using namespace RTGL1;
    g.texture_base     = layerTextures[ 0 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_base_ORM = layerTextures[ 0 ].indices[ TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_INDEX ];
    g.texture_base_N   = layerTextures[ 0 ].indices[ TEXTURE_NORMAL_INDEX ];
    g.texture_base_E   = layerTextures[ 0 ].indices[ TEXTURE_EMISSIVE_INDEX ];
    g.texture_base_D   = layerTextures[ 0 ].indices[ TEXTURE_HEIGHT_INDEX ];

    g.texture_layer1 = layerTextures[ 1 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_layer2 = layerTextures[ 2 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
    g.texture_layer3 = layerTextures[ 3 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ];
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
    else if( prim.flags & ( 1u << 28 ) ) // (Raze RT: RT_PRIMITIVE_ONE_SIDED)
    {
        pt = PassThrough::OpaqueOneSided;
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
    // Doom 64 RT: before the others, as a shadow proxy is submitted with its sprite's flags
    else if( prim.flags & RG_MESH_PRIMITIVE_SHADOW_ONLY )
    {
        pv = Visibility::ShadowOnly;
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
        // (Thief RT: filled at once, not pushed one by one -- half a million a frame)
        const size_t at = dst.indices.size();
        dst.indices.resize( at + prim.vertexCount );
        std::iota( dst.indices.begin() + at, dst.indices.end(), 0u );
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

    // Thief RT: a few primitives with a layer 3 (the game's light), as found here
    if( auto l = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &prim ); l && l->pLayer3 )
    {
        static int told = 0;
        if( told < 4 )
        {
            ++told;
            fprintf( stderr,
                     "RTGL1 (Thief): layer 3 \"%s\" -> texture %u, colour 0x%08x, flags 0x%08x (layer 3 %s)\n",
                     l->pLayer3->pTextureName ? l->pLayer3->pTextureName : "(none)",
                     layerTextures[ 3 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ],
                     unsigned( layerColors[ 3 ] ),
                     g.flags,
                     ( g.flags & GEOM_INST_FLAG_EXISTS_LAYER3 ) ? "there" : "MISSING" );
        }
    }

    ApplyTextures( g, layerTextures );

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

    // Doom 64 RT: not saturated, and a separate GI emission (RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED)
    g.emissiveMult   = std::max( 0.0f, prim.emissive );
    g.emissiveMultGi = std::max( 0.0f, prim.emissiveGi );

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

    // (Thief RT: the kept keys of primitives not given for a while, let go)
    if( ( ++keptFrame & 255 ) == 0 )
    {
        std::erase_if( keptKeys, [ this ]( auto& e ) {
            if( e.second.frame + 256 < keptFrame )
            {
                ReleaseRetained( e.second );
                return true;
            }
            return false;
        } );
    }

    // Thief RT (0028): room in the arena for what didn't fit last frame. When most of the
    // arena is freed ranges, it's emptied instead (its primitives kept again as they come)
    {
        // (at most 128 MB of vertices in each frame's buffers)
        constexpr uint32_t ArenaLimit[ ArenaKindCount ] = { 2u << 20, 6u << 20, 2u << 20, 2u << 20, 2u << 20 };
        bool               empty                        = false;
        for( uint32_t k = 0; k < ArenaKindCount; k++ )
        {
            if( arenaShortfall[ k ] && arena[ k ].top > ( 1u << 16 ) && uint64_t( arena[ k ].used ) * 2 < arena[ k ].top )
            {
                empty = true;
            }
        }
        if( empty )
        {
            for( auto& e : keptKeys )
            {
                ReleaseRetained( e.second );
            }
            for( auto& a : arena )
            {
                a = ArenaAllocator{};
            }
        }
        for( uint32_t k = 0; k < ArenaKindCount; k++ )
        {
            if( arenaShortfall[ k ] )
            {
                const uint64_t want = uint64_t( arena[ k ].top ) + arenaShortfall[ k ];
                const uint64_t grown =
                    std::max< uint64_t >( want + want / 4, uint64_t( arenaCapacity[ k ] ) + arenaCapacity[ k ] / 2 );
                arenaCapacity[ k ] = uint32_t( std::min< uint64_t >( std::max< uint64_t >( grown, 4096 ), ArenaLimit[ k ] ) );
                arenaShortfall[ k ] = 0;
            }
        }
        EnsureArena( Cur() );
        retainedCount = retainedVertexCount = retainedIndexCount = repeatedCount = retainedCopyCount = 0;
        nextRepeatable = false;
    }

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
    const bool repeatable = nextRepeatable;
    nextRepeatable        = false;

    if( frameGeomCount >= MAX_GEOM_INFO_COUNT )
    {
        debug::Error( "Too many geometry infos: the limit is {}", MAX_GEOM_INFO_COUNT );
        return false;
    }

    // (as Store)
    const bool     indexed    = prim.pIndices && prim.indexCount > 0;
    const uint32_t indexCount = indexed ? prim.indexCount : prim.vertexCount;
    if( !prim.pVertices || prim.vertexCount == 0 || indexCount < 3 || indexCount % 3 != 0 )
    {
        return false;
    }

    const Group group     = MakeGroup( mesh, prim );
    const bool  unchanged = ( prim.flags & ( 1u << 29 ) ) != 0;

    KeptKeys& kept = keptKeys[ uniqueID ];

    // Thief RT (0028): in the arena while it comes unchanged (RTGL1_METAL_NO_ARENA=1: never)
    static const bool arenaOn = !std::getenv( "RTGL1_METAL_NO_ARENA" );
    uint8_t           layers  = 0;
    for( uint32_t l = 1; l <= 3; l++ )
    {
        if( LayerExists( prim, l ) && AccessLayerTexCoords( prim, l ) )
        {
            layers |= uint8_t( 1u << ( l - 1 ) );
        }
    }
    if( kept.retained && !( unchanged && kept.retainedVertices == prim.vertexCount &&
                            kept.retainedIndices == indexCount && kept.layers == layers ) )
    {
        ReleaseRetained( kept );
    }
    if( arenaOn && unchanged && !kept.retained )
    {
        TryRetain( kept, prim.vertexCount, indexCount, layers );
    }

    VertexRange      range;
    const ShVertex*  keyVertices;
    const uint32_t*  keyIndices;
    if( kept.retained )
    {
        FrameData& f = Cur();
        if( !( kept.written & ( 1u << ring ) ) )
        {
            WriteRetained( f, kept, prim );
            kept.written |= uint8_t( 1u << ring );
            retainedCopyCount++;
        }
        range       = RetainedRange( kept );
        keyVertices = static_cast< const ShVertex* >( f.vertices.contents ) + kept.at[ ArenaVertices ];
        keyIndices  = static_cast< const uint32_t* >( f.indices.contents ) + kept.at[ ArenaIndices ];
        retainedCount++;
        retainedVertexCount += prim.vertexCount;
        retainedIndexCount += indexCount;
    }
    else
    {
        const auto stored = Store( dynamicStorage, prim );
        if( !stored )
        {
            return false;
        }
        keyVertices = &dynamicStorage.vertices[ stored->firstVertex ];
        keyIndices  = &dynamicStorage.indices[ stored->firstIndex ];
        // after the arena in the frame's buffers
        range = *stored;
        range.firstVertex += arenaCapacity[ ArenaVertices ];
        range.firstIndex += arenaCapacity[ ArenaIndices ];
        for( uint32_t l = 0; l < 3; l++ )
        {
            range.firstVertexLayer[ l ] += arenaCapacity[ ArenaLayer1 + l ];
        }
    }

    // Thief RT: the content key and cell of a primitive marked unchanged since it was last
    // given (the same count of vertices and indices, the same transform) are as they were
    // then; others are found from the vertices, and kept for next time
    const bool reuse = unchanged && kept.contentKey != 0 &&
                       kept.vertexCount == range.vertexCount && kept.indexCount == range.indexCount &&
                       memcmp( &kept.transform, &mesh.transform, sizeof( RgTransform ) ) == 0;
    if( !reuse )
    {
        kept.contentKey  = ContentKey( keyVertices,
                                      range.vertexCount,
                                      keyIndices,
                                      range.indexCount,
                                      mesh.transform,
                                      &kept.localKey );
        kept.cell        = StableCell( keyVertices, range.vertexCount, mesh.transform );
        kept.vertexCount = range.vertexCount;
        kept.indexCount  = range.indexCount;
        kept.transform   = mesh.transform;
    }
    kept.frame = keptFrame;

    dynamicGeoms[ group.ToIndex() ].push_back( Geometry{
        .info            = MakeGeomInfo( mesh, prim, range, true, textures ),
        .uniqueID        = uniqueID,
        .noMotionVectors = !!( prim.flags & RG_MESH_PRIMITIVE_NO_MOTION_VECTORS ),
        .alphaTested     = group.passThrough == PassThrough::AlphaTested,
        .bakedTransform  = mesh.transform,
        .contentKey      = kept.contentKey,
        .triangles       = range.indexCount / 3,
        .cell            = kept.cell,
    } );
    dynamicGeoms[ group.ToIndex() ].back().localKey = kept.localKey;
    frameGeomCount++;

    // as given now, for RepeatDynamic
    if( kept.retained && repeatable )
    {
        kept.repeat              = dynamicGeoms[ group.ToIndex() ].back();
        kept.groupIndex          = group.ToIndex();
        kept.materialsGeneration = textures.MaterialsGeneration();
        const auto* tl           = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &prim );
        const char* names[ 4 ]   = {
            prim.pTextureName,
            tl && tl->pLayer1 ? tl->pLayer1->pTextureName : nullptr,
            tl && tl->pLayer2 ? tl->pLayer2->pTextureName : nullptr,
            tl && tl->pLayer3 ? tl->pLayer3->pTextureName : nullptr,
        };
        for( int i = 0; i < 4; i++ )
        {
            const std::string_view n = Utils::SafeCstr( names[ i ] );
            if( kept.textureNames[ i ] != n )
            {
                kept.textureNames[ i ] = n;
            }
        }
    }
    else
    {
        kept.repeat.reset();
    }
    return true;
}

bool RTGL1::MetalScene::RepeatDynamic( const PrimitiveUniqueID& uniqueID, const MetalTextureManager& textures )
{
    auto it = keptKeys.find( uniqueID );
    if( it == keptKeys.end() )
    {
        return false;
    }
    KeptKeys& kept = it->second;
    if( !kept.retained || !kept.repeat || frameGeomCount >= MAX_GEOM_INFO_COUNT )
    {
        return false;
    }
    if( !( kept.written & ( 1u << ring ) ) )
    {
        // from another frame's buffers that have it
        uint32_t from = RingSize;
        for( uint32_t r = 0; r < RingSize; r++ )
        {
            if( r != ring && ( kept.written & ( 1u << r ) ) )
            {
                from = r;
                break;
            }
        }
        if( from == RingSize )
        {
            return false;
        }
        CopyRetained( frames[ from ], Cur(), kept );
        kept.written |= uint8_t( 1u << ring );
        retainedCopyCount++;
    }
    // its textures found again, if materials have changed since
    if( kept.materialsGeneration != textures.MaterialsGeneration() )
    {
        std::array< MaterialTextures, 4 > lt;
        for( int i = 0; i < 4; i++ )
        {
            lt[ i ] = textures.GetMaterialTextures( kept.textureNames[ i ].empty() ? nullptr : kept.textureNames[ i ].c_str() );
        }
        ApplyTextures( kept.repeat->info, lt );
        kept.materialsGeneration = textures.MaterialsGeneration();
    }
    kept.frame = keptFrame;
    dynamicGeoms[ kept.groupIndex ].push_back( *kept.repeat );
    frameGeomCount++;
    retainedCount++;
    retainedVertexCount += kept.retainedVertices;
    retainedIndexCount += kept.retainedIndices;
    repeatedCount++;
    return true;
}

bool RTGL1::MetalScene::ArenaAllocator::Alloc( uint32_t count, uint32_t limit, uint32_t* where )
{
    // the first freed range it fits (of the first few)
    int looked = 0;
    for( auto it = freeRanges.begin(); it != freeRanges.end() && looked < 256; ++it, ++looked )
    {
        if( it->second >= count )
        {
            *where                = it->first;
            const uint32_t restAt = it->first + count;
            const uint32_t rest   = it->second - count;
            freeRanges.erase( it );
            if( rest > 0 )
            {
                freeRanges.emplace( restAt, rest );
            }
            used += count;
            return true;
        }
    }
    if( uint64_t( top ) + count > limit )
    {
        return false;
    }
    *where = top;
    top += count;
    used += count;
    return true;
}

void RTGL1::MetalScene::ArenaAllocator::Free( uint32_t where, uint32_t count )
{
    if( count == 0 )
    {
        return;
    }
    used -= std::min( used, count );
    auto it = freeRanges.emplace( where, count ).first;
    // joined with its neighbours
    if( auto next = std::next( it ); next != freeRanges.end() && it->first + it->second == next->first )
    {
        it->second += next->second;
        freeRanges.erase( next );
    }
    if( it != freeRanges.begin() )
    {
        auto prev = std::prev( it );
        if( prev->first + prev->second == it->first )
        {
            prev->second += it->second;
            freeRanges.erase( it );
            it = prev;
        }
    }
    // the last one: back to the top
    if( it->first + it->second == top )
    {
        top = it->first;
        freeRanges.erase( it );
    }
}

size_t RTGL1::MetalScene::ArenaElementSize( uint32_t kind )
{
    switch( kind )
    {
        case ArenaVertices: return sizeof( ShVertex );
        case ArenaIndices: return sizeof( uint32_t );
        default: return sizeof( RgFloat2D );
    }
}

id< MTLBuffer > __strong& RTGL1::MetalScene::ArenaBuffer( FrameData& f, uint32_t kind )
{
    switch( kind )
    {
        case ArenaVertices: return f.vertices;
        case ArenaIndices: return f.indices;
        default: return f.texCoordLayers[ kind - ArenaLayer1 ];
    }
}

void RTGL1::MetalScene::EnsureArena( FrameData& f )
{
    static NSString* const labels[ ArenaKindCount ] = {
        @"Dynamic vertices", @"Dynamic indices", @"Dynamic tex coords", @"Dynamic tex coords", @"Dynamic tex coords",
    };
    for( uint32_t k = 0; k < ArenaKindCount; k++ )
    {
        const size_t e = ArenaElementSize( k );
        EnsureKeeping( device, ArenaBuffer( f, k ), size_t( arenaCapacity[ k ] ) * e, size_t( arena[ k ].top ) * e, labels[ k ] );
    }
}

bool RTGL1::MetalScene::TryRetain( KeptKeys& k, uint32_t vertexCount, uint32_t indexCount, uint8_t layers )
{
    const uint32_t counts[ ArenaKindCount ] = {
        vertexCount,
        indexCount,
        ( layers & 1 ) ? vertexCount : 0u,
        ( layers & 2 ) ? vertexCount : 0u,
        ( layers & 4 ) ? vertexCount : 0u,
    };
    uint32_t at[ ArenaKindCount ] = {};
    uint32_t done                 = 0;
    for( ; done < ArenaKindCount; done++ )
    {
        if( counts[ done ] > 0 && !arena[ done ].Alloc( counts[ done ], arenaCapacity[ done ], &at[ done ] ) )
        {
            break;
        }
    }
    if( done < ArenaKindCount )
    {
        // no room this frame: made for it (and the rest of this primitive) at the next
        for( uint32_t i = 0; i < done; i++ )
        {
            arena[ i ].Free( at[ i ], counts[ i ] );
        }
        for( uint32_t i = done; i < ArenaKindCount; i++ )
        {
            arenaShortfall[ i ] += counts[ i ];
        }
        return false;
    }
    k.retained         = true;
    k.written          = 0;
    k.layers           = layers;
    k.retainedVertices = vertexCount;
    k.retainedIndices  = indexCount;
    for( uint32_t i = 0; i < ArenaKindCount; i++ )
    {
        k.at[ i ] = at[ i ];
    }
    return true;
}

void RTGL1::MetalScene::ReleaseRetained( KeptKeys& k )
{
    if( !k.retained )
    {
        return;
    }
    const uint32_t counts[ ArenaKindCount ] = {
        k.retainedVertices,
        k.retainedIndices,
        ( k.layers & 1 ) ? k.retainedVertices : 0u,
        ( k.layers & 2 ) ? k.retainedVertices : 0u,
        ( k.layers & 4 ) ? k.retainedVertices : 0u,
    };
    for( uint32_t i = 0; i < ArenaKindCount; i++ )
    {
        if( counts[ i ] > 0 )
        {
            arena[ i ].Free( k.at[ i ], counts[ i ] );
        }
    }
    k.retained = false;
    k.written  = 0;
    k.repeat.reset();
}

void RTGL1::MetalScene::WriteRetained( FrameData& f, const KeptKeys& k, const RgMeshPrimitiveInfo& prim )
{
    auto* v = static_cast< ShVertex* >( f.vertices.contents ) + k.at[ ArenaVertices ];
    memcpy( v, prim.pVertices, size_t( k.retainedVertices ) * sizeof( ShVertex ) );

    auto* idx = static_cast< uint32_t* >( f.indices.contents ) + k.at[ ArenaIndices ];
    if( prim.pIndices && prim.indexCount > 0 )
    {
        memcpy( idx, prim.pIndices, size_t( k.retainedIndices ) * sizeof( uint32_t ) );
    }
    else
    {
        std::iota( idx, idx + k.retainedIndices, 0u );
    }

    for( uint32_t l = 0; l < 3; l++ )
    {
        if( k.layers & ( 1u << l ) )
        {
            auto* tc = static_cast< RgFloat2D* >( f.texCoordLayers[ l ].contents ) + k.at[ ArenaLayer1 + l ];
            memcpy( tc, AccessLayerTexCoords( prim, l + 1 ), size_t( k.retainedVertices ) * sizeof( RgFloat2D ) );
        }
    }
}

void RTGL1::MetalScene::CopyRetained( const FrameData& from, FrameData& to, const KeptKeys& k )
{
    auto copy = [ & ]( id< MTLBuffer > src, id< MTLBuffer > dst, uint32_t at, uint32_t count, size_t e ) {
        memcpy( static_cast< uint8_t* >( dst.contents ) + size_t( at ) * e,
                static_cast< const uint8_t* >( src.contents ) + size_t( at ) * e,
                size_t( count ) * e );
    };
    copy( from.vertices, to.vertices, k.at[ ArenaVertices ], k.retainedVertices, sizeof( ShVertex ) );
    copy( from.indices, to.indices, k.at[ ArenaIndices ], k.retainedIndices, sizeof( uint32_t ) );
    for( uint32_t l = 0; l < 3; l++ )
    {
        if( k.layers & ( 1u << l ) )
        {
            copy( from.texCoordLayers[ l ], to.texCoordLayers[ l ], k.at[ ArenaLayer1 + l ], k.retainedVertices, sizeof( RgFloat2D ) );
        }
    }
}

auto RTGL1::MetalScene::RetainedRange( const KeptKeys& k ) const -> VertexRange
{
    return VertexRange{
        .firstVertex      = k.at[ ArenaVertices ],
        .firstIndex       = k.at[ ArenaIndices ],
        .vertexCount      = k.retainedVertices,
        .indexCount       = k.retainedIndices,
        .firstVertexLayer = { k.at[ ArenaLayer1 ], k.at[ ArenaLayer2 ], k.at[ ArenaLayer3 ] },
    };
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
    std::vector< const Geometry* > ptrs;
    ptrs.reserve( geoms.size() );
    for( const Geometry& g : geoms )
    {
        ptrs.push_back( &g );
    }
    BuildBlas( enc, dst, ptrs, vertices, indices, bakeTransforms, fastBuild, label );
}

void RTGL1::MetalScene::RetireStable( StableGroup& sg )
{
    auto old = Retired{ .resources = {}, .framesLeft = RingSize + 1 };
    old.resources.push_back( sg.blas.blas );
    old.resources.push_back( sg.blas.scratch );
    old.resources.push_back( sg.blas.transforms );
    std::erase( old.resources, nil );
    if( !old.resources.empty() )
    {
        retired.push_back( std::move( old ) );
    }
    sg.blas      = {};
    sg.keys      = {};
    sg.triangles = 0;
}

void RTGL1::MetalScene::BuildBlas( id< MTLAccelerationStructureCommandEncoder > enc,
                                   BuiltBlas&                                   dst,
                                   const std::vector< const Geometry* >&        geoms,
                                   id< MTLBuffer >                              vertices,
                                   id< MTLBuffer >                              indices,
                                   bool                                         bakeTransforms,
                                   bool                                         fastBuild,
                                   NSString*                                    label,
                                   const BuiltBlas*                             refitFrom )
{
    if( geoms.empty() )
    {
        dst.blas = nil;
        dst.refitSig = 0;
        return;
    }

    if( bakeTransforms )
    {
        std::vector< MTLPackedFloat4x3 > transforms;
        transforms.reserve( geoms.size() );
        for( const Geometry* gp : geoms )
        {
            transforms.push_back( ToPackedFloat4x3( gp->bakedTransform ) );
        }
        Upload( device, dst.transforms, transforms, @"BLAS transforms" );
    }

    NSMutableArray< MTLAccelerationStructureGeometryDescriptor* >* descs =
        [NSMutableArray arrayWithCapacity:geoms.size()];

    for( size_t i = 0; i < geoms.size(); i++ )
    {
        const Geometry& g = *geoms[ i ];

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

    // Thief RT: the frame's dynamic BLAS (the moving models: guards walking, a door
    // swinging) was built anew every frame. When it holds the same geometries as the
    // previous frame's -- the same ids, the same triangle counts, the same kind --
    // that one is refitted into this one (its tree kept, its boxes moved to the
    // vertices as they are now): a fraction of a build. Built anew every few frames
    // still (RTGL1_METAL_REFITS, default 15 refits in a row; 0 never refits), as a
    // tree refitted ever further from the shape it was built for traces slower.
    if( refitFrom )
    {
        static const uint32_t maxRefits = [] {
            const char* e = std::getenv( "RTGL1_METAL_REFITS" );
            return e ? uint32_t( std::max( 0, std::atoi( e ) ) ) : 15u;
        }();
        uint64_t sig = 0xcbf29ce484222325ull ^ uint64_t( geoms.size() );
        auto     mix = [ &sig ]( uint64_t v ) {
            sig ^= v + 0x9E3779B97F4A7C15ull + ( sig << 6 ) + ( sig >> 2 );
            sig *= 0x100000001B3ull;
        };
        for( const Geometry* g : geoms )
        {
            mix( g->uniqueID.objectId );
            mix( uint64_t( g->uniqueID.primitiveIndex ) | ( uint64_t( g->info.indexCount ) << 32 ) );
            mix( g->alphaTested ? 1 : 2 );
        }
        sig |= 1; // (0: not refittable)
        if( maxRefits > 0 )
        {
            desc.usage |= MTLAccelerationStructureUsageRefit;
        }
        if( maxRefits > 0 && refitFrom->blas && refitFrom->blas != dst.blas && refitFrom->refitSig == sig &&
            refitFrom->refits < maxRefits )
        {
            MTLAccelerationStructureSizes sizes = [device accelerationStructureSizesWithDescriptor:desc];
            if( !dst.blas || dst.blas.size < refitFrom->blas.size )
            {
                dst.blas       = [device newAccelerationStructureWithSize:refitFrom->blas.size];
                dst.blas.label = label;
            }
            const NSUInteger need = std::max< NSUInteger >( 256, sizes.refitScratchBufferSize );
            if( !dst.scratch || dst.scratch.length < need )
            {
                dst.scratch       = [device newBufferWithLength:need + need / 2 options:MTLResourceStorageModePrivate];
                dst.scratch.label = [label stringByAppendingString:@" scratch"];
            }
            [enc refitAccelerationStructure:refitFrom->blas
                                 descriptor:desc
                                destination:dst.blas
                              scratchBuffer:dst.scratch
                        scratchBufferOffset:0];
            dst.refitSig = sig;
            dst.refits   = refitFrom->refits + 1;
            dynamicRefits++;
            return;
        }
        dst.refitSig = maxRefits > 0 ? sig : 0;
        dst.refits   = 0;
        dynamicBuilds++;
    }
    else
    {
        dst.refitSig = 0;
        dst.refits   = 0;
    }

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

    id< MTLAccelerationStructureCommandEncoder > enc = MetalProfiler::AccelerationStructure( cmd );
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

    // Thief RT (0028): after the arena (kept there as it was)
    {
        auto after = [ & ]( uint32_t kind, const void* src, size_t count, NSString* label ) {
            const size_t e    = ArenaElementSize( kind );
            const size_t base = size_t( arenaCapacity[ kind ] ) * e;
            EnsureKeeping( device, ArenaBuffer( f, kind ), base + count * e, size_t( arena[ kind ].top ) * e, label );
            if( count > 0 )
            {
                memcpy( static_cast< uint8_t* >( ArenaBuffer( f, kind ).contents ) + base, src, count * e );
            }
        };
        after( ArenaVertices, dynamicStorage.vertices.data(), dynamicStorage.vertices.size(), @"Dynamic vertices" );
        after( ArenaIndices, dynamicStorage.indices.data(), dynamicStorage.indices.size(), @"Dynamic indices" );
        for( uint32_t l = 0; l < 3; l++ )
        {
            after( ArenaLayer1 + l,
                   dynamicStorage.texCoordLayers[ l ].data(),
                   dynamicStorage.texCoordLayers[ l ].size(),
                   @"Dynamic tex coords" );
        }
    }

    // Raze RT: the stable part of the dynamic geometry (see StableGroup), and the rest
    static const bool stableOn = !std::getenv( "RTGL1_METAL_NO_STABLE" );
    // a geometry joins after coming unchanged this many frames in a row (a second: an
    // animation's frames -- a voxel fire's -- come and go faster, and each one that had
    // joined made a rebuild when it went)...
    // (RTGL1_METAL_STABLE_AGE: another number of frames -- a game whose rooms stand still in
    // world space can take them in sooner, rather than tracing them slowly for a second or more)
    static const uint32_t StableMinAge = [] {
        const char* e = std::getenv( "RTGL1_METAL_STABLE_AGE" );
        const long  v = e ? std::strtol( e, nullptr, 10 ) : 0;
        return v > 0 ? uint32_t( std::min( v, 100000L ) ) : 60u;
    }();
    // ...and only when there are this many triangles to keep
    constexpr size_t StableMinTriangles = 2000;
    // a kept BLAS takes in more stable geometry at most this often, when it's 5% more
    constexpr uint32_t StableAbsorbFrames = 120;

    frameCounter++;
    {
        std::unordered_map< uint64_t, uint32_t > age;
        age.reserve( keyAge.size() + 256 );
        for( const auto& g : dynamicGeoms )
        {
            for( const Geometry& geom : g )
            {
                auto it                = keyAge.find( geom.contentKey );
                age[ geom.contentKey ] = it != keyAge.end() ? std::min( it->second + 1, 1u << 30 ) : 1u;
            }
        }
        keyAge = std::move( age );

        std::unordered_map< uint64_t, uint32_t > lage;
        lage.reserve( localAge.size() + 64 );
        for( const auto& g : dynamicGeoms )
        {
            for( const Geometry& geom : g )
            {
                if( geom.localKey )
                {
                    auto it                 = localAge.find( geom.localKey );
                    lage[ geom.localKey ] = it != localAge.end() ? std::min( it->second + 1, 1u << 30 ) : 1u;
                }
            }
        }
        localAge = std::move( lage );
    }

    // a cell's few triangles stay with the rest (a BLAS of its own is not worth it)
    constexpr size_t StableCellMinTriangles = 200;
    const int        tracingTest            = std::clamp( g_thiefTracingTest.load(), 0, 5 );
    size_t           pieceInstances = 0, objectInstances = 0, alphaInstances = 0, alphaTriangles = 0;
    (void)StableMinTriangles;

    struct CellWork
    {
        StableGroup*                   sg;
        std::vector< const Geometry* > list;
        bool                           buildNow;
        bool                           fast;
    };
    std::array< std::vector< CellWork >, GroupCount >        cellWork;
    std::array< std::vector< const Geometry* >, GroupCount > volatileLists, instancedLists;
    // models kept by their shape (see modelBlas): RTGL1_METAL_NO_MODEL_BLAS=1 turns it off
    static const bool  modelsOn          = stableOn && !std::getenv( "RTGL1_METAL_NO_MODEL_BLAS" );
    constexpr uint32_t ModelMinAge       = 2;
    constexpr uint32_t ModelMinTriangles = 32;
    for( uint32_t gi = 0; gi < GroupCount; gi++ )
    {
        const auto& geoms = dynamicGeoms[ gi ];
        auto&       cells = stable[ gi ];

        std::vector< uint8_t > used( geoms.size(), 0 );
        std::unordered_map< int64_t, std::vector< uint32_t > > byCell;
        // (test 4, 5: a still object joins its piece of the map, however small next to it)
        const bool absorbAll = tracingTest == 4 || tracingTest == 5;
        if( stableOn )
        {
            for( uint32_t i = 0; i < geoms.size(); i++ )
            {
                byCell[ TestCell( geoms[ i ].cell, tracingTest ) ].push_back( i );
            }
        }
        // a kept cell with nothing in it this frame: gone
        for( auto it = cells.begin(); it != cells.end(); )
        {
            if( !byCell.contains( it->first ) )
            {
                RetireStable( it->second );
                it = cells.erase( it );
            }
            else
            {
                ++it;
            }
        }

        for( const auto& [ cell, idxs ] : byCell )
        {
            auto         cit = cells.find( cell );
            StableGroup* sg  = cit != cells.end() ? &cit->second : nullptr;

            std::vector< const Geometry* > st;
            std::vector< uint32_t >        taken;
            bool                           valid = sg && !sg->keys.empty();
            if( valid )
            {
                std::unordered_map< uint64_t, std::vector< uint32_t > > byKey;
                byKey.reserve( idxs.size() );
                for( uint32_t i : idxs )
                {
                    byKey[ geoms[ i ].contentKey ].push_back( i );
                }
                st.reserve( sg->keys.size() );
                for( uint64_t k : sg->keys )
                {
                    auto it = byKey.find( k );
                    if( it == byKey.end() || it->second.empty() )
                    {
                        // one of them has changed or gone: the kept BLAS shows it as it was
                        valid = false;
                        break;
                    }
                    const uint32_t i = it->second.back();
                    it->second.pop_back();
                    taken.push_back( i );
                    st.push_back( &geoms[ i ] );
                }
            }
            if( valid )
            {
                for( uint32_t i : taken )
                {
                    used[ i ] = 1;
                }
            }
            else
            {
                if( sg && !sg->keys.empty() )
                {
                    RetireStable( *sg );
                }
                st.clear();
            }

            // the geometries of the cell that could join it
            size_t candidates = 0;
            for( uint32_t i : idxs )
            {
                if( !used[ i ] && keyAge[ geoms[ i ].contentKey ] >= StableMinAge )
                {
                    candidates += geoms[ i ].triangles;
                }
            }
            const bool rebuild = candidates >= ( absorbAll && sg && !sg->keys.empty() ? size_t( 1 ) : StableCellMinTriangles ) &&
                                 ( !sg || sg->keys.empty() ||
                                   ( frameCounter - sg->builtFrame >= StableAbsorbFrames &&
                                     ( absorbAll || candidates * 20 >= sg->triangles ) ) );
            bool fast = false;
            if( rebuild )
            {
                if( !sg )
                {
                    sg = &cells[ cell ];
                }
                // built for tracing; but when it keeps changing (it was built less than a second
                // ago), for building: a slow build every few frames would cost more than it saves
                fast = sg->builtFrame != 0 && frameCounter - sg->builtFrame < 60;
                RetireStable( *sg );
                for( uint32_t i : idxs )
                {
                    if( !used[ i ] && keyAge[ geoms[ i ].contentKey ] >= StableMinAge )
                    {
                        used[ i ] = 1;
                        st.push_back( &geoms[ i ] );
                    }
                }
                sg->keys.reserve( st.size() );
                for( const Geometry* g : st )
                {
                    sg->keys.push_back( g->contentKey );
                    sg->triangles += g->triangles;
                }
                sg->builtFrame = frameCounter;
                stableRebuilds++;
            }
            if( !st.empty() )
            {
                cellWork[ gi ].push_back( CellWork{ sg, std::move( st ), rebuild, fast } );
            }
        }

        auto& vol = volatileLists[ gi ];
        vol.reserve( geoms.size() );
        for( uint32_t i = 0; i < geoms.size(); i++ )
        {
            if( !used[ i ] )
            {
                // a model that moves but keeps its shape (a spinning pickup): its own BLAS, in its
                // own space, kept; instanced with its transform
                const Geometry& g = geoms[ i ];
                if( modelsOn && g.localKey && g.triangles >= ModelMinTriangles && localAge[ g.localKey ] >= ModelMinAge )
                {
                    instancedLists[ gi ].push_back( &g );
                    continue;
                }
                vol.push_back( &g );
            }
        }
    }

    size_t lastObjectsKept = 0;   // (this frame's, for the stats)

    // Thief RT: the moving objects of some size out of the shared dynamic BLAS, each into its
    // own (objectBlas); the small rest stays (RTGL1_METAL_NO_OBJECT_BLAS=1: all stays)
    std::array< std::vector< std::pair< uint64_t, std::vector< const Geometry* > > >, GroupCount > objectLists;
    {
        static const bool  objectsOn          = !std::getenv( "RTGL1_METAL_NO_OBJECT_BLAS" );
        constexpr uint32_t ObjectMinTriangles = 48;
        for( uint32_t gi = 0; objectsOn && gi < GroupCount; gi++ )
        {
            auto&                                  vol = volatileLists[ gi ];
            std::unordered_map< uint64_t, uint32_t > tris;
            for( const Geometry* g : vol )
            {
                tris[ g->uniqueID.objectId >> 12 ] += g->triangles;
            }
            std::unordered_map< uint64_t, size_t > at;
            std::vector< const Geometry* >         rest;
            for( const Geometry* g : vol )
            {
                const uint64_t key = g->uniqueID.objectId >> 12;
                if( tris[ key ] < ObjectMinTriangles )
                {
                    rest.push_back( g );
                    continue;
                }
                auto it = at.find( key );
                if( it == at.end() )
                {
                    it = at.emplace( key, objectLists[ gi ].size() ).first;
                    objectLists[ gi ].push_back( { key, {} } );
                }
                objectLists[ gi ][ it->second ].second.push_back( g );
            }
            vol.swap( rest );
        }
    }

    // dynamic BLAS-es
    if( std::ranges::any_of( dynamicGeoms, []( const auto& g ) { return !g.empty(); } ) )
    {
        id< MTLAccelerationStructureCommandEncoder > enc = MetalProfiler::AccelerationStructure( cmd );
        enc.label                                        = @"Build dynamic BLAS";
        for( uint32_t gi = 0; gi < GroupCount; gi++ )
        {
            BuildBlas( enc, f.dynamicBlas[ gi ], volatileLists[ gi ], f.vertices, f.indices, true, true, @"Dynamic BLAS",
                       &Prev().dynamicBlas[ gi ] );
            static const BuiltBlas noneBefore{};
            static const bool      keepOn = !std::getenv( "RTGL1_METAL_NO_OBJECT_KEEP" );
            // (unchanged this many frames after a refit or a fast build: built for tracing)
            constexpr uint32_t PolishAfter = 30;
            for( const auto& [ key, list ] : objectLists[ gi ] )
            {
                ObjectBlas& ob = objectBlas[ gi ][ key ];
                uint64_t    sig = 0x84222325cbf29ce4ull ^ uint64_t( list.size() );
                for( const Geometry* g : list )
                {
                    sig ^= g->contentKey + 0x9E3779B97F4A7C15ull + ( sig << 6 ) + ( sig >> 2 );
                    sig *= 0x100000001B3ull;
                    sig ^= uint64_t( g->alphaTested ? 1 : 2 ) | ( uint64_t( g->info.indexCount ) << 8 );
                }
                const bool live = ob.useSlot < RingSize && ob.lastFrame + 1 == frameCounter &&
                                  ob.slots[ ob.useSlot ].blas != nil;
                // a slot no frame in flight reads (one of RingSize is always free: the frames in
                // flight are the RingSize - 1 before this one, each using one)
                auto freeSlot = [ & ]() {
                    for( uint32_t k = 0; k < RingSize; k++ )
                    {
                        const uint32_t s = ( ring + k ) % RingSize;
                        if( s != ob.useSlot && ( ob.usedFrame[ s ] == 0 || ob.usedFrame[ s ] + RingSize <= frameCounter ) )
                        {
                            return s;
                        }
                    }
                    return ring;   // (not reached)
                };
                if( keepOn && live && sig == ob.contentSig )
                {
                    ob.sameFrames++;
                    if( !ob.polished && ob.sameFrames >= PolishAfter )
                    {
                        const uint32_t s = freeSlot();
                        BuildBlas( enc, ob.slots[ s ], list, f.vertices, f.indices, true, false, @"Object BLAS (kept)", nullptr );
                        ob.useSlot  = s;
                        ob.polished = true;
                        objectPolishes++;
                    }
                    else
                    {
                        objectKeeps++;
                        lastObjectsKept++;
                    }
                }
                else
                {
                    // refitted from the one before, if it had one (else built, refittable)
                    const uint32_t   s    = keepOn ? freeSlot() : ring;
                    const bool       had  = live && ob.useSlot != s;
                    const BuiltBlas* from = had ? &ob.slots[ ob.useSlot ] : &noneBefore;
                    BuildBlas( enc, ob.slots[ s ], list, f.vertices, f.indices, true, true, @"Object BLAS", from );
                    ob.useSlot    = s;
                    ob.contentSig = sig;
                    ob.sameFrames = 0;
                    ob.polished   = false;
                }
                ob.usedFrame[ ob.useSlot ] = frameCounter;
                ob.lastFrame               = frameCounter;
            }
            for( const Geometry* g : instancedLists[ gi ] )
            {
                ModelBlas& mb = modelBlas[ gi ][ g->localKey ];
                if( !mb.blas.blas )
                {
                    // in its own space: the instance has the transform
                    BuildBlas( enc, mb.blas, std::vector< const Geometry* >{ g }, f.vertices, f.indices, false, false, @"Model BLAS" );
                    modelBuilds++;
                }
                mb.lastFrame = frameCounter;
            }
            for( const CellWork& cw : cellWork[ gi ] )
            {
                if( cw.buildNow )
                {
                    // from this frame's vertices: the kept geometries are the same in every frame it is used
                    BuildBlas( enc, cw.sg->blas, cw.list, f.vertices, f.indices, true, cw.fast, @"Stable BLAS" );
                }
            }
        }
        [enc endEncoding];
    }
    else
    {
        for( auto& b : f.dynamicBlas )
        {
            b.blas     = nil;
            b.refitSig = 0;
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
        if( ( pt == PassThrough::Opaque || pt == PassThrough::OpaqueOneSided ) &&
            ( meshFlags & ( RG_MESH_FORCE_GLASS | RG_MESH_FORCE_WATER ) ) )
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
            case Visibility::ShadowOnly: mask = INSTANCE_MASK_RESERVED_0; break;
            case Visibility::World2:
                mask        = INSTANCE_MASK_WORLD_2;
                customIndex = INSTANCE_CUSTOM_INDEX_FLAG_SKY;
                if( !( rayCullMaskWorld & INSTANCE_MASK_WORLD_2 ) )
                {
                    return;
                }
                break;
        }
        // Doom 64 RT: alpha tested (sprites) don't see the shadow proxies, see getShadowCullMask
        if( pt == PassThrough::AlphaTested )
        {
            customIndex |= INSTANCE_CUSTOM_INDEX_FLAG_IGNORE_SHADOW_PROXY;
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
        const bool seeThrough = pt == PassThrough::AlphaTested && tracingTest != 1;
        if( pt == PassThrough::AlphaTested )
        {
            alphaInstances++;
            for( const Geometry* g : geoms )
            {
                alphaTriangles += g->triangles;
            }
        }
        inst.options = MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise |
                       ( seeThrough ? MTLAccelerationStructureInstanceOptionNonOpaque
                                    : MTLAccelerationStructureInstanceOptionOpaque ) |
                       // (Raze RT: the rays cull the triangles seen from behind, but only the
                       // one-sided group's -- see ShGlobalUniform::rayCullBackFaces)
                       ( pt == PassThrough::OpaqueOneSided ? MTLAccelerationStructureInstanceOptionNone
                                                           : MTLAccelerationStructureInstanceOptionDisableTriangleCulling );
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
        const auto group = Group{ Visibility( gi / PassThroughCount ), PassThrough( gi % PassThroughCount ) };
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
        const auto group = Group{ Visibility( gi / PassThroughCount ), PassThrough( gi % PassThroughCount ) };
        for( const CellWork& cw : cellWork[ gi ] )
        {
            addInstance( cw.sg->blas.blas, group, 0, IdentityTransform, cw.list );
            pieceInstances++;
        }
        addInstance( f.dynamicBlas[ gi ].blas, group, 0, IdentityTransform, volatileLists[ gi ] );
        for( const auto& [ key, list ] : objectLists[ gi ] )
        {
            auto it = objectBlas[ gi ].find( key );
            if( it != objectBlas[ gi ].end() )
            {
                addInstance( it->second.slots[ it->second.useSlot % RingSize ].blas, group, 0, IdentityTransform, list );
                objectInstances++;
            }
        }
        for( const Geometry* g : instancedLists[ gi ] )
        {
            auto it = modelBlas[ gi ].find( g->localKey );
            if( it != modelBlas[ gi ].end() )
            {
                addInstance( it->second.blas.blas, group, 0, g->bakedTransform, std::span< const Geometry* const >( &g, 1 ) );
            }
        }
    }
    // the moving objects not seen for a while: gone (their slots kept until the frames
    // in flight are done with them)
    for( auto& om : objectBlas )
    {
        for( auto it = om.begin(); it != om.end(); )
        {
            if( frameCounter - it->second.lastFrame > 120 )
            {
                for( BuiltBlas& b : it->second.slots )
                {
                    StableGroup tmp;
                    tmp.blas = b;
                    RetireStable( tmp );
                }
                it = om.erase( it );
            }
            else
            {
                ++it;
            }
        }
    }
    // the models not seen for a while: gone
    for( auto& mm : modelBlas )
    {
        for( auto it = mm.begin(); it != mm.end(); )
        {
            if( frameCounter - it->second.lastFrame > 300 )
            {
                StableGroup tmp;
                tmp.blas = it->second.blas;
                RetireStable( tmp );
                it = mm.erase( it );
            }
            else
            {
                ++it;
            }
        }
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
                dst.flags2 |= GEOM_INST_FLAGS2_NO_HISTORY;
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
    {
        lastStats = Stats{
            .dynamicGeometries = 0,
            .dynamicTriangles  = dynamicStorage.indices.size() / 3,
            .dynamicVertices   = dynamicStorage.vertices.size(),
            .staticGeometries  = 0,
            .tlasInstances     = instances.size(),
            .geometryInfos     = geomInfos.size(),
            .polishedGeometries = size_t( std::ranges::count_if(
                geomInfos, []( const ShGeometryInstance& g ) { return ( g.flags & GEOM_INST_FLAG_RESERVED_6 ) != 0; } ) ),
        };
        for( const auto& g : dynamicGeoms )
        {
            lastStats.dynamicGeometries += g.size();
        }
        for( const auto& g : staticGeoms )
        {
            lastStats.staticGeometries += g.size();
        }
        for( uint32_t gi = 0; gi < GroupCount; gi++ )
        {
            for( const CellWork& cw : cellWork[ gi ] )
            {
                lastStats.stableGeometries += cw.list.size();
                lastStats.stableTriangles += cw.sg->triangles;
                lastStats.stableCells++;
            }
        }
        lastStats.stableRebuilds = stableRebuilds;
        lastStats.dynamicRefits  = dynamicRefits;
        lastStats.dynamicBuilds  = dynamicBuilds;
        lastStats.objectsKept    = lastObjectsKept;
        lastStats.tracingTest     = tracingTest;
        lastStats.pieceInstances  = pieceInstances;
        lastStats.objectInstances = objectInstances;
        lastStats.alphaInstances  = alphaInstances;
        lastStats.alphaTriangles  = alphaTriangles;
        lastStats.objectKeeps    = objectKeeps;
        lastStats.objectPolishes = objectPolishes;
        for( uint32_t gi = 0; gi < GroupCount; gi++ )
        {
            lastStats.instancedModels += instancedLists[ gi ].size();
            lastStats.modelBlases += modelBlas[ gi ].size();
            lastStats.volatileGeometries += volatileLists[ gi ].size();
            lastStats.movingObjects += objectLists[ gi ].size();
            for( const Geometry* g : volatileLists[ gi ] )
            {
                lastStats.volatileTriangles += g->triangles;
            }
        }
        lastStats.modelBuilds = modelBuilds;
        lastStats.dynamicVertices += retainedVertexCount;
        lastStats.dynamicTriangles += retainedIndexCount / 3;
        lastStats.retainedGeometries = retainedCount;
        lastStats.retainedVertices   = retainedVertexCount;
        lastStats.repeatedGeometries = repeatedCount;
        lastStats.retainedCopies     = retainedCopyCount;
        lastStats.arenaVertices      = arena[ ArenaVertices ].used;
        lastStats.arenaCapacity      = arenaCapacity[ ArenaVertices ];
    }
    tlasDesc.instanceDescriptorBuffer = f.instances;
    tlasDesc.instanceDescriptorType   = MTLAccelerationStructureInstanceDescriptorTypeUserID;
    tlasDesc.usage                    = MTLAccelerationStructureUsagePreferFastBuild;

    id< MTLAccelerationStructure > tlas    = f.tlas;
    id< MTLBuffer >                scratch = f.scratchTlas;
    EnsureAccelerationStructure( device, tlas, scratch, tlasDesc, @"TLAS" );
    f.tlas        = tlas;
    f.scratchTlas = scratch;

    // a separate encoder, so that the TLAS build waits for the BLAS builds
    id< MTLAccelerationStructureCommandEncoder > enc = MetalProfiler::AccelerationStructure( cmd );
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
