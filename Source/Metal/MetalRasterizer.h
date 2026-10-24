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

#pragma once

// Objective-C++ only

#import <Metal/Metal.h>

#include "Common.h"

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace RTGL1
{

class MetalBindingTable;
class MetalComputePipelines;
class MetalTextureManager;

// The same as Rasterizer + RasterizedDataCollector + RasterPass + SwapchainPass,
// with RTGL1's raster shaders (RsRasterizer.vert, RsSky.frag, RsWorld_RT.frag, RsSwapchain.frag, ...):
// - sky geometry: into a cubemap around the sky viewer (as RenderCubemap.cpp),
//   and into the albedo framebuf before the primary rays
// - world geometry (translucent, particles): into the final framebuf, with the ray traced depth
// - swapchained geometry (HUD, menus): into the upscaled framebuf, before tonemapping
// - decals: into the G-buffer, after the primary rays (DecalManager)
// TODO: lens flares
class MetalRasterizer
{
public:
    // GeometryRasterType
    enum class Type
    {
        World,
        Sky,
        Swapchain,
        Decal,
        // the same primitives as ray traced, for the classic mode (RgStartFrameInfo::lightmapScreenCoverage)
        WorldClassic,
        Count,
    };

    MetalRasterizer( id< MTLDevice >  device,
                     id< MTLLibrary > library,
                     MTLPixelFormat   swapchainFormat,
                     bool             vertexColorGamma,
                     uint32_t         skyCubemapSize );

    MetalRasterizer( const MetalRasterizer& other )            = delete;
    MetalRasterizer& operator=( const MetalRasterizer& other ) = delete;

    void BeginFrame( uint32_t frameIndex );

    // RasterizedDataCollector::AddPrimitive
    void Add( Type                       type,
              const RgTransform&         transform,
              const RgMeshPrimitiveInfo& primitive,
              const float*               pViewProjection,
              const RgViewport*          pViewport,
              const MetalTextureManager& textures );

    // Renders the sky primitives into SkyCubemap(). Returns false, if there are none.
    bool             RenderSkyCubemap( id< MTLCommandBuffer >     cmd,
                                       const MetalTextureManager& textures,
                                       const RgFloat3D&           skyViewerPosition,
                                       float                      zNear,
                                       float                      zFar );
    id< MTLTexture > SkyCubemap() const { return skyCubemap; }

    struct PassParams
    {
        MetalComputePipelines&   shaders;
        const MetalBindingTable& bindings;
        const float*             view;
        const float*             projection;
        RgFloat2D                jitter;
        uint32_t                 width;
        uint32_t                 height;
    };

    // Rasterizer::DrawSkyToAlbedo
    void DrawSkyToAlbedo( id< MTLCommandBuffer > cmd,
                          const PassParams&      params,
                          id< MTLTexture >       albedo,
                          const RgFloat3D&       skyViewerPosition );

    // Rasterizer::DrawDecals: decals modify the G-buffer after the primary rays
    void DrawDecals( id< MTLCommandBuffer > cmd,
                     const PassParams&      params,
                     id< MTLTexture >       albedo,
                     id< MTLTexture >       normalDecal,
                     id< MTLTexture >       screenEmission );

    // Rasterizer::DrawToFinalImage: DepthCopying, then the world geometry
    // 'lightmapScreenCoverage' > 0: the left part of the screen is classic, not drawn here.
    // 'asOverlay': 'final' is cleared, and gets premultiplied color with the coverage in alpha
    // (for an external denoiser, to not denoise the rasterized geometry)
    void DrawToFinal( id< MTLCommandBuffer > cmd,
                      const PassParams&      params,
                      id< MTLTexture >       final,
                      id< MTLTexture >       screenEmission,
                      id< MTLTexture >       reactivity,
                      float                  lightmapScreenCoverage,
                      bool                   asOverlay = false );

    // Rasterizer::DrawClassic: the left part of 'target' (of 'lightmapScreenCoverage' width)
    // is rasterized with the classic look; 'params' are of the target's size, renderWidth is for quantization
    void DrawClassic( id< MTLCommandBuffer > cmd,
                      const PassParams&      params,
                      id< MTLTexture >       target,
                      const RgFloat3D&       skyViewerPosition,
                      float                  lightmapScreenCoverage,
                      uint32_t               renderWidth );

    // Rasterizer::DrawToSwapchain, into an upscaled framebuf (no jitter).
    // 'clear': to get only the swapchained geometry, e.g. a UI texture for frame interpolation;
    // then alpha is its coverage
    void DrawToSwapchain( id< MTLCommandBuffer > cmd,
                          const PassParams&      params,
                          id< MTLTexture >       target,
                          bool                   clear = false );

    // When RTGL1's shaders are not used: swapchained geometry, into the current render pass
    // (that has the swapchain as a color attachment)
    void DrawSwapchainLegacy( id< MTLRenderCommandEncoder > enc,
                              const MetalTextureManager&    textures,
                              uint32_t                      width,
                              uint32_t                      height,
                              MTLPixelFormat                format );

    bool HasDraws( Type type ) const { return !draws[ size_t( type ) ].empty(); }

private:
    // RasterizedDataCollector.h
    enum PipelineStateFlagBits : uint32_t
    {
        ALPHA_TEST     = 1 << 0,
        TRANSLUCENT    = 1 << 1,
        ADDITIVE       = 1 << 2,
        DEPTH_TEST     = 1 << 3,
        DEPTH_WRITE    = 1 << 4,
        DRAW_AS_LINES  = 1 << 5,
        SKY_VISIBILITY = 1 << 6,
    };

    // RasterizedDataCollector::DrawInfo
    struct DrawInfo
    {
        RgTransform       transform;
        uint32_t          texture_base;
        uint32_t          texture_base_N;
        uint32_t          texture_base_E;
        RgColor4DPacked32 colorFactor_base;
        uint32_t          vertexCount;
        uint32_t          firstVertex;
        uint32_t          indexCount;
        uint32_t          firstIndex;
        float             emissive;

        std::optional< std::array< float, 16 > > viewProj;
        std::optional< RgViewport >              viewport;
        uint32_t                                 pipelineState;
    };

    struct FrameData
    {
        id< MTLBuffer > vertices;
        id< MTLBuffer > indices;
    };

    void UploadGeometry();
    void EnsureDepth( id< MTLTexture > __strong& depth, uint32_t width, uint32_t height );

    // Rasterizer::Draw: the draws of a type, into an already begun render pass
    // 'classic' as in RasterDrawParams: > 0 to draw only the left part, < 0 only the right part
    void DrawAll( id< MTLRenderCommandEncoder >     enc,
                  const PassParams&                 params,
                  Type                              type,
                  const char*                       fragmentShader,
                  std::span< const MTLPixelFormat > colorFormats,
                  MTLPixelFormat                    depthFormat,
                  const float*                      defaultViewProj,
                  std::optional< float >            classic = std::nullopt,
                  bool                              coverageAlpha = false );

    void EncodeDraw( id< MTLRenderCommandEncoder > enc, const DrawInfo& d );

private:
    id< MTLDevice > device;
    bool            vertexColorGamma;

    // for the sky cubemap and DrawSwapchainLegacy: hand-written shaders (RTGL1.metal)
    enum class LegacyBlend
    {
        Opaque,
        Alpha,
        Additive,
        Count,
    };
    static LegacyBlend ToLegacyBlend( uint32_t pipelineState );
    void EncodeLegacyDraw( id< MTLRenderCommandEncoder > enc, const DrawInfo& d, const float* mvp );

    // [0] for BGRA8 sRGB, [1] for RGBA16Float (EDR)
    id< MTLRenderPipelineState > legacyPipelines[ 2 ][ size_t( LegacyBlend::Count ) ];
    id< MTLRenderPipelineState > skyPipelines[ size_t( LegacyBlend::Count ) ];
    id< MTLDepthStencilState >   depthTestWrite;
    id< MTLDepthStencilState >   depthTestNoWrite;
    // DepthCopying: always write
    id< MTLDepthStencilState >   depthAlwaysWrite;

    id< MTLTexture > skyCubemap;
    id< MTLTexture > skyDepth;
    // RasterPass: a render size depth buffer, and for the classic pass of the target's size
    id< MTLTexture > renderDepth;
    id< MTLTexture > classicDepth;

    bool geometryUploaded{ false };

    FrameData frames[ MAX_FRAMES_IN_FLIGHT ];
    uint32_t  frameIndex{ 0 };

    std::vector< RgPrimitiveVertex > vertices;
    std::vector< uint32_t >          indices;
    std::vector< DrawInfo >          draws[ size_t( Type::Count ) ];
};

}
