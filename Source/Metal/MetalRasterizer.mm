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

#include "MetalRasterizer.h"

#include "MetalComputePipelines.h"
#include "MetalTextureManager.h"
#include "Shaders/MetalBindings.h"

#include "Const.h"
#include "Generated/ShaderCommonC.h"
#include "DrawFrameInfo.h"
#include "Matrix.h"
#include "RgException.h"
#include "Utils.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

namespace
{

id< MTLRenderPipelineState > CreateLegacyPipeline( id< MTLDevice >  device,
                                                   id< MTLLibrary > library,
                                                   MTLPixelFormat   colorFormat,
                                                   MTLPixelFormat   depthFormat,
                                                   NSString*        label,
                                                   bool             blend,
                                                   MTLBlendFactor   dst )
{
    auto* desc            = [MTLRenderPipelineDescriptor new];
    desc.label            = label;
    desc.vertexFunction   = [library newFunctionWithName:@"RasterSwapchainVertex"];
    desc.fragmentFunction = [library newFunctionWithName:@"RasterSwapchainFragment"];
    if( !desc.vertexFunction || !desc.fragmentFunction )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER, "Can't find RasterSwapchain shaders" );
    }
    desc.depthAttachmentPixelFormat = depthFormat;

    MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[ 0 ];
    att.pixelFormat                                 = colorFormat;
    att.blendingEnabled                             = blend;
    att.sourceRGBBlendFactor                        = MTLBlendFactorSourceAlpha;
    att.destinationRGBBlendFactor                   = dst;
    att.rgbBlendOperation                           = MTLBlendOperationAdd;
    att.sourceAlphaBlendFactor                      = MTLBlendFactorOne;
    att.destinationAlphaBlendFactor                 = MTLBlendFactorOne;
    att.alphaBlendOperation                         = MTLBlendOperationAdd;

    NSError*                     error = nil;
    id< MTLRenderPipelineState > pso   = [device newRenderPipelineStateWithDescriptor:desc error:&error];
    if( !pso )
    {
        throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR,
                                  std::string{ "Can't create raster pipeline: " } +
                                      error.localizedDescription.UTF8String );
    }
    return pso;
}

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

// RasterPass.cpp's RASTER_PASS_DEPTH_FORMAT
constexpr MTLPixelFormat DepthFormat = MTLPixelFormatDepth32Float;

// Rasterizer.cpp
struct RasterizedPushConst
{
    float    vp[ 16 ];
    uint32_t packedColor;
    uint32_t textureIndex;
    uint32_t emissiveTextureIndex;
    float    emissiveMult;
    uint32_t normalTextureIndex;
    uint32_t manualSrgb;
};
static_assert( offsetof( RasterizedPushConst, packedColor ) == 64 );
static_assert( offsetof( RasterizedPushConst, normalTextureIndex ) == 80 );
static_assert( sizeof( RasterizedPushConst ) == 88 );

MTLViewport ToMetal( const RgViewport& v )
{
    return MTLViewport{ v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth };
}

}

RTGL1::MetalRasterizer::MetalRasterizer( id< MTLDevice >  _device,
                                         id< MTLLibrary > library,
                                         MTLPixelFormat   swapchainFormat,
                                         bool             _vertexColorGamma,
                                         uint32_t         skyCubemapSize )
    : device{ _device }, vertexColorGamma{ _vertexColorGamma }
{
    struct
    {
        LegacyBlend    blend;
        NSString*      name;
        bool           enable;
        MTLBlendFactor dst;
    } blends[] = {
        { LegacyBlend::Opaque, @"opaque", false, MTLBlendFactorZero },
        { LegacyBlend::Alpha, @"alpha", true, MTLBlendFactorOneMinusSourceAlpha },
        { LegacyBlend::Additive, @"additive", true, MTLBlendFactorOne },
    };
    for( const auto& b : blends )
    {
        for( int f = 0; f < 2; f++ )
        {
            legacyPipelines[ f ][ size_t( b.blend ) ] =
                CreateLegacyPipeline( device,
                                      library,
                                      f == 0 ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatRGBA16Float,
                                      MTLPixelFormatInvalid,
                                      [@"Raster swapchain " stringByAppendingString:b.name],
                                      b.enable,
                                      b.dst );
        }
        skyPipelines[ size_t( b.blend ) ] = CreateLegacyPipeline( device,
                                                                  library,
                                                                  MTLPixelFormatRGBA16Float,
                                                                  DepthFormat,
                                                                  [@"Raster sky " stringByAppendingString:b.name],
                                                                  b.enable,
                                                                  b.dst );
    }

    {
        auto* d                = [MTLDepthStencilDescriptor new];
        d.depthCompareFunction = MTLCompareFunctionLessEqual; // RasterizerPipelines.cpp
        d.depthWriteEnabled    = YES;
        depthTestWrite         = [device newDepthStencilStateWithDescriptor:d];
        d.depthWriteEnabled    = NO;
        depthTestNoWrite       = [device newDepthStencilStateWithDescriptor:d];
        d.depthCompareFunction = MTLCompareFunctionAlways;
        d.depthWriteEnabled    = YES;
        depthAlwaysWrite       = [device newDepthStencilStateWithDescriptor:d];
    }

    skyCubemapSize = std::max( skyCubemapSize, 16u );
    {
        auto* d = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                        size:skyCubemapSize
                                                                   mipmapped:NO];
        d.usage          = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        d.storageMode    = MTLStorageModePrivate;
        skyCubemap       = [device newTextureWithDescriptor:d];
        skyCubemap.label = @"Sky cubemap";
    }
    {
        auto* d        = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:DepthFormat
                                                                     width:skyCubemapSize
                                                                    height:skyCubemapSize
                                                                 mipmapped:NO];
        d.usage        = MTLTextureUsageRenderTarget;
        d.storageMode  = MTLStorageModeMemoryless;
        skyDepth       = [device newTextureWithDescriptor:d];
        skyDepth.label = @"Sky cubemap depth";
    }
}

void RTGL1::MetalRasterizer::BeginFrame( uint32_t _frameIndex )
{
    frameIndex = _frameIndex;
    vertices.clear();
    indices.clear();
    for( auto& d : draws )
    {
        d.clear();
    }
    geometryUploaded = false;
}

// RasterizedDataCollector.cpp's ToPipelineState
static uint32_t ToPipelineState( RTGL1::MetalRasterizer::Type type, const RgMeshPrimitiveInfo& info );

void RTGL1::MetalRasterizer::Add( Type                       type,
                                  const RgTransform&         transform,
                                  const RgMeshPrimitiveInfo& info,
                                  const float*               pViewProjection,
                                  const RgViewport*          pViewport,
                                  const MetalTextureManager& textures )
{
    if( info.vertexCount == 0 || !info.pVertices )
    {
        return;
    }

    const bool     indexed     = info.indexCount > 0 && info.pIndices;
    const uint32_t firstVertex = uint32_t( vertices.size() );
    const uint32_t firstIndex  = indexed ? uint32_t( indices.size() ) : 0;

    vertices.insert( vertices.end(), info.pVertices, info.pVertices + info.vertexCount );
    if( indexed )
    {
        indices.insert( indices.end(), info.pIndices, info.pIndices + info.indexCount );
    }

    const auto layerTextures = textures.GetTexturesForLayers( info );
    const auto colors        = textures.GetColorForLayers( info );

    RgColor4DPacked32 baseColor = colors[ 0 ];
    if( ( info.flags & RG_MESH_PRIMITIVE_GLASS ) || ( info.flags & RG_MESH_PRIMITIVE_WATER ) )
    {
        constexpr uint8_t AlphaAboveAlphaTestThreshold = 170;

        uint8_t newAlpha =
            std::min( AlphaAboveAlphaTestThreshold, Utils::UnpackAlphaFromPacked32AsUint8( colors[ 0 ] ) );
        baseColor = Utils::ReplaceAlphaInPacked32( colors[ 0 ], newAlpha );
    }

    if( type == Type::WorldClassic )
    {
        baseColor = Utils::MultiplyColorPacked32( baseColor, info.classicLight );
    }

    auto d = DrawInfo{
        .transform        = transform,
        .texture_base     = layerTextures[ 0 ].indices[ TEXTURE_ALBEDO_ALPHA_INDEX ],
        .texture_base_N   = layerTextures[ 0 ].indices[ TEXTURE_NORMAL_INDEX ],
        .texture_base_E   = layerTextures[ 0 ].indices[ TEXTURE_EMISSIVE_INDEX ],
        .colorFactor_base = baseColor,
        .vertexCount      = info.vertexCount,
        .firstVertex      = firstVertex,
        .indexCount       = indexed ? info.indexCount : 0,
        .firstIndex       = firstIndex,
        .emissive         = Utils::Saturate( info.emissive ),
        .viewProj         = std::nullopt,
        .viewport         = pViewport ? std::optional{ *pViewport } : std::nullopt,
        .pipelineState    = ToPipelineState( type, info ),
    };
    if( pViewProjection )
    {
        d.viewProj = std::array< float, 16 >{};
        memcpy( d.viewProj->data(), pViewProjection, 16 * sizeof( float ) );
    }

    draws[ size_t( type ) ].push_back( d );
}

static uint32_t ToPipelineState( RTGL1::MetalRasterizer::Type type, const RgMeshPrimitiveInfo& info )
{
    using namespace RTGL1;

    // the same flags as PipelineStateFlagBits in RasterizedDataCollector.h
    constexpr uint32_t ALPHA_TEST = 1 << 0, TRANSLUCENT = 1 << 1, ADDITIVE = 1 << 2, DEPTH_TEST = 1 << 3,
                       DEPTH_WRITE = 1 << 4, DRAW_AS_LINES = 1 << 5, SKY_VISIBILITY = 1 << 6;

    uint32_t r = 0;

    if( info.flags & RG_MESH_PRIMITIVE_ALPHA_TESTED )
    {
        r |= ALPHA_TEST;
    }
    if( ( info.flags & RG_MESH_PRIMITIVE_TRANSLUCENT ) || ( info.flags & RG_MESH_PRIMITIVE_GLASS ) ||
        ( info.flags & RG_MESH_PRIMITIVE_WATER ) )
    {
        r |= TRANSLUCENT;
    }
    // if alpha specifies semi-transparency
    if( Utils::UnpackAlphaFromPacked32( info.color ) < MESH_TRANSLUCENT_ALPHA_THRESHOLD )
    {
        r |= TRANSLUCENT;
    }
    // if emissive and translucent, then it's additive
    if( info.emissive > std::numeric_limits< float >::epsilon() )
    {
        if( r & TRANSLUCENT )
        {
            r |= ADDITIVE;
        }
    }
    if( info.flags & RG_MESH_PRIMITIVE_SKY_VISIBILITY )
    {
        r |= SKY_VISIBILITY;
    }
    // allow depth test for world / sky geometry
    if( type != MetalRasterizer::Type::Swapchain )
    {
        r |= DEPTH_TEST;
        // depth write if not semi-transparent
        if( !( r & TRANSLUCENT ) )
        {
            r |= DEPTH_WRITE;
        }
    }
    if( auto sw = pnext::find< RgMeshPrimitiveSwapchainedEXT >( &info ) )
    {
        if( sw->flags & RG_MESH_PRIMITIVE_SWAPCHAINED_DRAW_AS_LINES )
        {
            r |= DRAW_AS_LINES;
        }
    }
    return r;
}

void RTGL1::MetalRasterizer::UploadGeometry()
{
    if( geometryUploaded )
    {
        return;
    }
    geometryUploaded = true;

    FrameData& f = frames[ frameIndex ];
    EnsureBuffer( device, f.vertices, vertices.size() * sizeof( RgPrimitiveVertex ), @"Raster vertices" );
    EnsureBuffer( device, f.indices, indices.size() * sizeof( uint32_t ), @"Raster indices" );
    if( !vertices.empty() )
    {
        memcpy( f.vertices.contents, vertices.data(), vertices.size() * sizeof( RgPrimitiveVertex ) );
    }
    if( !indices.empty() )
    {
        memcpy( f.indices.contents, indices.data(), indices.size() * sizeof( uint32_t ) );
    }
}

void RTGL1::MetalRasterizer::EnsureDepth( id< MTLTexture > __strong& depth, uint32_t width, uint32_t height )
{
    if( depth && depth.width == width && depth.height == height )
    {
        return;
    }
    auto* d       = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:DepthFormat
                                                              width:width
                                                             height:height
                                                          mipmapped:NO];
    d.usage       = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    depth         = [device newTextureWithDescriptor:d];
    depth.label   = @"Raster depth";
}

void RTGL1::MetalRasterizer::EncodeDraw( id< MTLRenderCommandEncoder > enc, const DrawInfo& d )
{
    const FrameData&       f    = frames[ frameIndex ];
    const MTLPrimitiveType type = ( d.pipelineState & DRAW_AS_LINES ) ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;

    if( d.indexCount > 0 )
    {
        // indices are relative to the primitive's first vertex
        [enc drawIndexedPrimitives:type
                        indexCount:d.indexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:f.indices
                 indexBufferOffset:d.firstIndex * sizeof( uint32_t )
                     instanceCount:1
                        baseVertex:d.firstVertex
                      baseInstance:0];
    }
    else
    {
        [enc drawPrimitives:type vertexStart:d.firstVertex vertexCount:d.vertexCount];
    }
}

void RTGL1::MetalRasterizer::DrawAll( id< MTLRenderCommandEncoder >     enc,
                                      const PassParams&                 params,
                                      Type                              type,
                                      const char*                       fragmentShader,
                                      std::span< const MTLPixelFormat > colorFormats,
                                      MTLPixelFormat                    depthFormat,
                                      const float*                      defaultViewProj,
                                      std::optional< float >            classic,
                                      bool                              coverageAlpha )
{
    if( coverageAlpha )
    {
        // MTLBlendFactorBlendAlpha: the coverage of opaque geometry is 1
        [enc setBlendColorRed:0 green:0 blue:0 alpha:1];
    }

    const auto defaultViewport = MTLViewport{ 0, 0, double( params.width ), double( params.height ), 0, 1 };

    // the render area of Rasterizer::Draw
    if( classic )
    {
        const bool  left = classic.value() > 0;
        const float c    = Utils::Saturate( std::abs( classic.value() ) );
        const float w    = float( params.width );

        auto area = MTLScissorRect{ 0, 0, params.width, params.height };
        if( left )
        {
            area.width = NSUInteger( w * c );
        }
        else
        {
            area.x     = NSUInteger( w * c );
            area.width = params.width - std::min( params.width, uint32_t( area.x ) );
        }
        area.width = std::max< NSUInteger >( area.width, 1 );
        [enc setScissorRect:area];
    }

    [enc setVertexBuffer:frames[ frameIndex ].vertices
                  offset:0
                 atIndex:MetalComputePipelines::RasterVertexBufferIndex];

    for( const DrawInfo& d : draws[ size_t( type ) ] )
    {
        const bool translucent = d.pipelineState & TRANSLUCENT;
        const bool additive    = d.pipelineState & ADDITIVE;

        // RasterizerPipelines::CreatePipeline
        auto desc = MetalComputePipelines::RenderDesc{
            .vertexShader      = "RsRasterizer_vert",
            .fragmentShader    = fragmentShader,
            .vertexConstants   = { { 0, vertexColorGamma ? 1u : 0u } },
            .fragmentConstants = { { 0, ( d.pipelineState & ALPHA_TEST ) ? 1u : 0u },
                                   { 1, ( d.pipelineState & SKY_VISIBILITY ) ? 1u : 0u } },
            .depthFormat       = depthFormat,
            .lines             = !!( d.pipelineState & DRAW_AS_LINES ),
        };
        {
            MTLBlendFactor blendSrc = MTLBlendFactorOne;
            MTLBlendFactor blendDst = MTLBlendFactorOne;
            if( translucent )
            {
                blendSrc = MTLBlendFactorSourceAlpha;
                blendDst = additive ? MTLBlendFactorOne : MTLBlendFactorOneMinusSourceAlpha;
            }
            auto blendBase = MetalComputePipelines::Blend{ additive || translucent, blendSrc, blendDst };
            if( coverageAlpha )
            {
                // alpha is the coverage: 'over' for translucent, unchanged for additive, 1 for opaque
                blendBase = MetalComputePipelines::Blend{
                    .enable        = true,
                    .src           = blendBase.enable ? blendSrc : MTLBlendFactorOne,
                    .dst           = blendBase.enable ? blendDst : MTLBlendFactorZero,
                    .separateAlpha = true,
                    .srcAlpha      = additive ? MTLBlendFactorZero
                                     : translucent ? MTLBlendFactorOne
                                                   : MTLBlendFactorBlendAlpha,
                    .dstAlpha      = additive ? MTLBlendFactorOne
                                     : translucent ? MTLBlendFactorOneMinusSourceAlpha
                                                   : MTLBlendFactorZero,
                };
            }
            const auto blendEmission   = MetalComputePipelines::Blend{ additive || translucent, MTLBlendFactorOne, MTLBlendFactorOne };
            const auto blendReactivity = MetalComputePipelines::Blend{};

            const MetalComputePipelines::Blend blends[] = { blendBase, blendEmission, blendReactivity };
            for( size_t i = 0; i < colorFormats.size() && i < 3; i++ )
            {
                desc.colorFormats[ i ] = colorFormats[ i ];
                desc.blends[ i ]       = blends[ i ];
            }
        }

        // RasterizedPushConst
        auto push = RasterizedPushConst{
            .vp                   = {},
            .packedColor          = d.colorFactor_base,
            .textureIndex         = d.texture_base,
            .emissiveTextureIndex = d.texture_base_E,
            .emissiveMult         = d.emissive,
            .normalTextureIndex   = d.texture_base_N,
            .manualSrgb           = 0,
        };
        {
            float model[ 16 ] = RG_MATRIX_TRANSPOSED( d.transform );
            Matrix::Multiply( push.vp, model, d.viewProj ? d.viewProj->data() : defaultViewProj );
        }

        params.shaders.BindRender(
            enc, desc, params.bindings, { reinterpret_cast< const uint8_t* >( &push ), sizeof( push ) } );
        if( depthFormat != MTLPixelFormatInvalid )
        {
            [enc setDepthStencilState:params.shaders.GetDepthState( d.pipelineState & DEPTH_TEST,
                                                                    d.pipelineState & DEPTH_WRITE )];
        }
        [enc setViewport:d.viewport ? ToMetal( *d.viewport ) : defaultViewport];

        EncodeDraw( enc, d );
    }
}

void RTGL1::MetalRasterizer::DrawSkyToAlbedo( id< MTLCommandBuffer > cmd,
                                              const PassParams&      params,
                                              id< MTLTexture >       albedo,
                                              const RgFloat3D&       skyViewerPosition )
{
    if( !HasDraws( Type::Sky ) )
    {
        return;
    }
    UploadGeometry();
    EnsureDepth( renderDepth, params.width, params.height );

    float skyView[ 16 ];
    Matrix::SetNewViewerPosition( skyView, params.view, skyViewerPosition.data );

    auto jitterredProj = ApplyJitter( params.projection, params.jitter, params.width, params.height );

    float defaultSkyViewProj[ 16 ];
    Matrix::Multiply( defaultSkyViewProj, skyView, jitterredProj.data() );

    auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[ 0 ].texture     = albedo;
    pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionLoad;
    pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;
    pass.depthAttachment.texture           = renderDepth;
    pass.depthAttachment.loadAction        = MTLLoadActionClear;
    pass.depthAttachment.clearDepth        = 1.0;
    pass.depthAttachment.storeAction       = MTLStoreActionDontCare;

    id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label                         = @"Rasterized sky to albedo framebuf";

    const MTLPixelFormat formats[] = { albedo.pixelFormat };
    DrawAll( enc, params, Type::Sky, "RsSky_frag", formats, DepthFormat, defaultSkyViewProj );

    [enc endEncoding];
}

void RTGL1::MetalRasterizer::DrawDecals( id< MTLCommandBuffer > cmd,
                                         const PassParams&      params,
                                         id< MTLTexture >       albedo,
                                         id< MTLTexture >       normalDecal,
                                         id< MTLTexture >       screenEmission )
{
    if( !HasDraws( Type::Decal ) )
    {
        return;
    }
    UploadGeometry();

    const uint32_t groupsX = ( params.width + COMPUTE_DECAL_APPLY_GROUP_SIZE_X - 1 ) / COMPUTE_DECAL_APPLY_GROUP_SIZE_X;
    const uint32_t groupsY = ( params.height + COMPUTE_DECAL_APPLY_GROUP_SIZE_X - 1 ) / COMPUTE_DECAL_APPLY_GROUP_SIZE_X;

    // DecalManager::CopyRtGBufferToAttachments: normals to an attachment
    {
        id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
        enc.label                          = @"Decal normals to attachment";
        params.shaders.DispatchThreadgroups( enc, "CmDecalNormalsCopy", { { 0, 0 } }, params.bindings, groupsX, groupsY );
        [enc endEncoding];
    }

    auto jitterredProj = ApplyJitter( params.projection, params.jitter, params.width, params.height );

    float defaultViewProj[ 16 ];
    Matrix::Multiply( defaultViewProj, params.view, jitterredProj.data() );

    {
        auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        {
            id< MTLTexture > colors[] = { albedo, normalDecal, screenEmission };
            for( int i = 0; i < 3; i++ )
            {
                pass.colorAttachments[ i ].texture     = colors[ i ];
                pass.colorAttachments[ i ].loadAction  = MTLLoadActionLoad;
                pass.colorAttachments[ i ].storeAction = MTLStoreActionStore;
            }
        }

        id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
        enc.label                         = @"Decals";

        // DecalManager::CreatePipelines
        auto desc = MetalComputePipelines::RenderDesc{
            .vertexShader    = "RsDecal_vert",
            .fragmentShader  = "RsDecal_frag",
            .vertexConstants = { { 0, vertexColorGamma ? 1u : 0u } },
        };
        desc.colorFormats = { albedo.pixelFormat, normalDecal.pixelFormat, screenEmission.pixelFormat, MTLPixelFormatInvalid };
        desc.blends[ 0 ]  = { true, MTLBlendFactorSourceAlpha, MTLBlendFactorOneMinusSourceAlpha };
        desc.blends[ 1 ]  = {};
        desc.blends[ 2 ]  = { true, MTLBlendFactorOne, MTLBlendFactorOne };

        [enc setVertexBuffer:frames[ frameIndex ].vertices
                      offset:0
                     atIndex:MetalComputePipelines::RasterVertexBufferIndex];
        [enc setViewport:MTLViewport{ 0, 0, double( params.width ), double( params.height ), 0, 1 }];

        for( const DrawInfo& d : draws[ size_t( Type::Decal ) ] )
        {
            auto push = RasterizedPushConst{
                .vp                   = {},
                .packedColor          = d.colorFactor_base,
                .textureIndex         = d.texture_base,
                .emissiveTextureIndex = d.texture_base_E,
                .emissiveMult         = d.emissive,
                .normalTextureIndex   = d.texture_base_N,
                .manualSrgb           = 0,
            };
            float model[ 16 ] = RG_MATRIX_TRANSPOSED( d.transform );
            Matrix::Multiply( push.vp, model, d.viewProj ? d.viewProj->data() : defaultViewProj );

            params.shaders.BindRender(
                enc, desc, params.bindings, { reinterpret_cast< const uint8_t* >( &push ), sizeof( push ) } );
            EncodeDraw( enc, d );
        }
        [enc endEncoding];
    }

    // DecalManager::CopyAttachmentsToRtGBuffer
    {
        id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
        enc.label                          = @"Decal normals to G-buffer";
        params.shaders.DispatchThreadgroups( enc, "CmDecalNormalsCopy", { { 0, 1 } }, params.bindings, groupsX, groupsY );
        [enc endEncoding];
    }
}

void RTGL1::MetalRasterizer::DrawToFinal( id< MTLCommandBuffer > cmd,
                                          const PassParams&      params,
                                          id< MTLTexture >       final,
                                          id< MTLTexture >       screenEmission,
                                          id< MTLTexture >       reactivity,
                                          float                  lightmapScreenCoverage,
                                          bool                   asOverlay )
{
    if( !HasDraws( Type::World ) )
    {
        if( asOverlay )
        {
            auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[ 0 ].texture     = final;
            pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionClear;
            pass.colorAttachments[ 0 ].clearColor  = MTLClearColorMake( 0, 0, 0, 0 );
            pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;
            id< MTLRenderCommandEncoder > enc      = [cmd renderCommandEncoderWithDescriptor:pass];
            enc.label                              = @"Clear raster overlay";
            [enc endEncoding];
        }
        return;
    }
    UploadGeometry();
    EnsureDepth( renderDepth, params.width, params.height );

    // DepthCopying: the ray traced depth to the depth buffer
    {
        auto* pass                       = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.depthAttachment.texture     = renderDepth;
        pass.depthAttachment.loadAction  = MTLLoadActionDontCare;
        pass.depthAttachment.storeAction = MTLStoreActionStore;

        id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
        enc.label                         = @"Depth copying";

        const auto desc = MetalComputePipelines::RenderDesc{
            .vertexShader       = "RsFullscreenQuad_vert",
            .fragmentShader     = "RsDepthCopying_frag",
            .depthFormat        = DepthFormat,
            .rasterVertexLayout = false,
        };
        const uint32_t push[] = { params.width, params.height };
        params.shaders.BindRender(
            enc, desc, params.bindings, { reinterpret_cast< const uint8_t* >( push ), sizeof( push ) } );
        [enc setDepthStencilState:depthAlwaysWrite];
        [enc setViewport:MTLViewport{ 0, 0, double( params.width ), double( params.height ), 0, 1 }];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [enc endEncoding];
    }

    auto jitterredProj = ApplyJitter( params.projection, params.jitter, params.width, params.height );

    float defaultViewProj[ 16 ];
    Matrix::Multiply( defaultViewProj, params.view, jitterredProj.data() );

    auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    {
        id< MTLTexture > colors[] = { final, screenEmission, reactivity };
        for( int i = 0; i < 3; i++ )
        {
            pass.colorAttachments[ i ].texture     = colors[ i ];
            pass.colorAttachments[ i ].loadAction  = MTLLoadActionLoad;
            pass.colorAttachments[ i ].storeAction = MTLStoreActionStore;
        }
        if( asOverlay )
        {
            pass.colorAttachments[ 0 ].loadAction = MTLLoadActionClear;
            pass.colorAttachments[ 0 ].clearColor = MTLClearColorMake( 0, 0, 0, 0 );
        }
    }
    pass.depthAttachment.texture     = renderDepth;
    pass.depthAttachment.loadAction  = MTLLoadActionLoad;
    pass.depthAttachment.storeAction = MTLStoreActionDontCare;

    id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label                         = @"Rasterized to final framebuf";

    const MTLPixelFormat formats[] = { final.pixelFormat, screenEmission.pixelFormat, reactivity.pixelFormat };
    DrawAll( enc,
             params,
             Type::World,
             "RsWorld_RT_frag",
             formats,
             DepthFormat,
             defaultViewProj,
             lightmapScreenCoverage > 0 ? std::optional{ -lightmapScreenCoverage } : std::nullopt,
             asOverlay );

    [enc endEncoding];
}

void RTGL1::MetalRasterizer::DrawClassic( id< MTLCommandBuffer > cmd,
                                          const PassParams&      params,
                                          id< MTLTexture >       target,
                                          const RgFloat3D&       skyViewerPosition,
                                          float                  lightmapScreenCoverage,
                                          uint32_t               renderWidth )
{
    if( lightmapScreenCoverage <= 0 || ( !HasDraws( Type::Sky ) && !HasDraws( Type::WorldClassic ) ) )
    {
        return;
    }
    UploadGeometry();
    EnsureDepth( classicDepth, params.width, params.height );

    auto l_pass = [ & ]( NSString* label ) {
        auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[ 0 ].texture     = target;
        pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionLoad;
        pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;
        pass.depthAttachment.texture           = classicDepth;
        pass.depthAttachment.loadAction        = MTLLoadActionClear;
        pass.depthAttachment.clearDepth        = 1.0;
        pass.depthAttachment.storeAction       = MTLStoreActionDontCare;

        id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
        enc.label                         = label;
        return enc;
    };

    const MTLPixelFormat formats[] = { target.pixelFormat };

    // sky
    {
        float skyView[ 16 ];
        Matrix::SetNewViewerPosition( skyView, params.view, skyViewerPosition.data );

        float defaultSkyViewProj[ 16 ];
        Matrix::Multiply( defaultSkyViewProj, skyView, params.projection );

        id< MTLRenderCommandEncoder > enc = l_pass( @"Rasterized classic sky" );
        DrawAll( enc, params, Type::Sky, "RsWorld_Classic_frag", formats, DepthFormat, defaultSkyViewProj, lightmapScreenCoverage );
        [enc endEncoding];
    }

    float defaultViewProj[ 16 ];
    Matrix::Multiply( defaultViewProj, params.view, params.projection );

    // lightmapScreenCoverage is quantized by 1/renderSize rather than 1/upscaledSize
    {
        auto l_quantize = []( float value, float by ) { return std::ceil( value / by ) * by; };

        float eps = std::max( 0.0f, 1.0f / float( renderWidth ) - 0.00001f );

        lightmapScreenCoverage = l_quantize( lightmapScreenCoverage + eps, 1.0f / float( renderWidth ) );
    }

    {
        id< MTLRenderCommandEncoder > enc = l_pass( @"Rasterized classic" );
        DrawAll( enc,
                 params,
                 Type::WorldClassic,
                 "RsWorld_Classic_frag",
                 formats,
                 DepthFormat,
                 defaultViewProj,
                 lightmapScreenCoverage );
        [enc endEncoding];
    }
}

void RTGL1::MetalRasterizer::DrawToSwapchain( id< MTLCommandBuffer > cmd,
                                              const PassParams&      params,
                                              id< MTLTexture >       target,
                                              bool                   clear )
{
    if( !HasDraws( Type::Swapchain ) && !clear )
    {
        return;
    }
    UploadGeometry();

    float defaultViewProj[ 16 ];
    Matrix::Multiply( defaultViewProj, params.view, params.projection );

    auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[ 0 ].texture     = target;
    pass.colorAttachments[ 0 ].loadAction  = clear ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.colorAttachments[ 0 ].clearColor  = MTLClearColorMake( 0, 0, 0, 0 );
    pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;

    id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
    enc.label                         = @"Rasterized to swapchain";

    const MTLPixelFormat formats[] = { target.pixelFormat };
    // 'clear': only the swapchained geometry, so its coverage is in alpha
    DrawAll( enc,
             params,
             Type::Swapchain,
             "RsSwapchain_frag",
             formats,
             MTLPixelFormatInvalid,
             defaultViewProj,
             std::nullopt,
             clear );

    [enc endEncoding];
}

auto RTGL1::MetalRasterizer::ToLegacyBlend( uint32_t pipelineState ) -> LegacyBlend
{
    return ( pipelineState & ADDITIVE )      ? LegacyBlend::Additive
           : ( pipelineState & TRANSLUCENT ) ? LegacyBlend::Alpha
                                             : LegacyBlend::Opaque;
}

void RTGL1::MetalRasterizer::EncodeLegacyDraw( id< MTLRenderCommandEncoder > enc, const DrawInfo& d, const float* mvp )
{
    auto c = MetalRasterConstants{
        .color            = d.colorFactor_base,
        .textureIndex     = d.texture_base,
        .alphaTest        = ( d.pipelineState & ALPHA_TEST ) ? 1u : 0u,
        .vertexColorGamma = vertexColorGamma,
    };
    memcpy( c.mvp, mvp, sizeof( c.mvp ) );
    [enc setVertexBytes:&c length:sizeof( c ) atIndex:MTL_RASTER_BINDING_CONSTANTS];
    [enc setFragmentBytes:&c length:sizeof( c ) atIndex:MTL_RASTER_BINDING_CONSTANTS];

    EncodeDraw( enc, d );
}

bool RTGL1::MetalRasterizer::RenderSkyCubemap( id< MTLCommandBuffer >     cmd,
                                               const MetalTextureManager& textures,
                                               const RgFloat3D&           skyViewerPosition,
                                               float                      zNear,
                                               float                      zFar )
{
    if( !HasDraws( Type::Sky ) )
    {
        return false;
    }
    UploadGeometry();

    for( uint32_t face = 0; face < 6; face++ )
    {
        // the same as viewProjCubemap in VulkanDevice.cpp
        float viewProj[ 16 ];
        Matrix::GetCubemapViewProjMat( viewProj, face, skyViewerPosition.data, zNear, zFar );

        auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[ 0 ].texture     = skyCubemap;
        pass.colorAttachments[ 0 ].slice       = face;
        pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionClear;
        pass.colorAttachments[ 0 ].clearColor  = MTLClearColorMake( 0, 0, 0, 0 );
        pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;
        pass.depthAttachment.texture           = skyDepth;
        pass.depthAttachment.loadAction        = MTLLoadActionClear;
        pass.depthAttachment.clearDepth        = 1.0;
        pass.depthAttachment.storeAction       = MTLStoreActionDontCare;

        id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
        enc.label                         = @"Sky cubemap face";
        [enc setVertexBuffer:frames[ frameIndex ].vertices offset:0 atIndex:MTL_RASTER_BINDING_VERTICES];
        [enc setFragmentBuffer:textures.TableBuffer() offset:0 atIndex:MTL_RASTER_BINDING_TEXTURES];
        textures.UseResources( enc );

        for( const DrawInfo& d : draws[ size_t( Type::Sky ) ] )
        {
            [enc setRenderPipelineState:skyPipelines[ size_t( ToLegacyBlend( d.pipelineState ) ) ]];
            [enc setDepthStencilState:( d.pipelineState & DEPTH_WRITE ) ? depthTestWrite : depthTestNoWrite];

            float model[ 16 ] = RG_MATRIX_TRANSPOSED( d.transform );
            float mvp[ 16 ];
            Matrix::Multiply( mvp, model, viewProj );
            EncodeLegacyDraw( enc, d, mvp );
        }
        [enc endEncoding];
    }
    return true;
}

void RTGL1::MetalRasterizer::DrawSwapchainLegacy( id< MTLRenderCommandEncoder > enc,
                                                  const MetalTextureManager&    textures,
                                                  uint32_t                      width,
                                                  uint32_t                      height,
                                                  MTLPixelFormat                format )
{
    if( !HasDraws( Type::Swapchain ) )
    {
        return;
    }
    UploadGeometry();

    [enc pushDebugGroup:@"Rasterized to swapchain"];
    [enc setVertexBuffer:frames[ frameIndex ].vertices offset:0 atIndex:MTL_RASTER_BINDING_VERTICES];
    [enc setFragmentBuffer:textures.TableBuffer() offset:0 atIndex:MTL_RASTER_BINDING_TEXTURES];
    textures.UseResources( enc );

    const auto fullViewport = MTLViewport{ 0, 0, double( width ), double( height ), 0, 1 };

    for( const DrawInfo& d : draws[ size_t( Type::Swapchain ) ] )
    {
        [enc setRenderPipelineState:legacyPipelines[ format == MTLPixelFormatRGBA16Float ? 1 : 0 ]
                                                   [ size_t( ToLegacyBlend( d.pipelineState ) ) ]];
        [enc setViewport:d.viewport ? ToMetal( *d.viewport ) : fullViewport];

        // swapchained geometry always has its view-projection
        float model[ 16 ] = RG_MATRIX_TRANSPOSED( d.transform );
        float mvp[ 16 ];
        Matrix::Multiply( mvp, model, d.viewProj ? d.viewProj->data() : model );
        EncodeLegacyDraw( enc, d, mvp );
    }

    [enc popDebugGroup];
}
