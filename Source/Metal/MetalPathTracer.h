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

#include "RTGL1/RTGL1.h"

#include "Common.h"
#include "Generated/ShaderCommonC.h"

#include <array>
#include <functional>
#include <memory>
#include <string_view>

namespace RTGL1
{

class MetalBindingTable;
class MetalComputePipelines;
class MetalFramebuffers;

// The ray traced part of a frame, in the same order as VulkanDevice::Render:
// primary rays -> reflections / refractions -> ReSTIR direct -> indirect -> denoiser
// -> exposure -> composition -> HDR prepare (tonemapping to LDR).
// Owns the buffers that are not framebuffers: tonemapping histogram, LPM params, indirect reservoirs.
class MetalPathTracer
{
public:
    MetalPathTracer( id< MTLDevice > device, id< MTLLibrary > library, MetalComputePipelines& shaders );
    ~MetalPathTracer();

    MetalPathTracer( const MetalPathTracer& other )            = delete;
    MetalPathTracer& operator=( const MetalPathTracer& other ) = delete;

    enum class Upscaler
    {
        Nearest,
        Linear,
        // RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2
        MetalFX,
    };

    struct Params
    {
        uint32_t frameIndex;
        uint32_t width;
        uint32_t height;
        float    currentTime;

        uint32_t primaryRaysMaxAlbedoLayers;
        uint32_t indirectIlluminationMaxAlbedoLayers;
        uint32_t lightmapLayerIndex;

        // the same as ShGlobalUniform, to choose the passes on CPU
        uint32_t reflectRefractMaxDepth;
        float    volumetricMaxHistoryLength;

        id< MTLBuffer >  globalUniform;
        id< MTLTexture > skyCubemap;

        uint32_t  upscaledWidth;
        uint32_t  upscaledHeight;
        uint32_t                 frameId;
        bool                     effectWipeIsUsed;
        Upscaler                 upscaler;
        RgRenderSharpenTechnique sharpenTechnique;
        // HDR display output (EDR): HDR_DISPLAY_LINEAR
        bool                     hdr;
        RgFloat2D jitter;
        bool      resetHistory;
        float     cameraNear;
        float     cameraFar;
        float     cameraFovY;
        // RG_DENOISER_METALFX: the MetalFX temporal denoised scaler instead of SVGF and the temporal scaler,
        // only if 'upscaler' is MetalFX
        bool         metalFXDenoiser;
        const float* view;
        const float* projection;
        // also make the image without HUD (GetHudlessImage), for frame interpolation
        bool         hudlessCopy;

        // Rasterizer::DrawDecals, after the primary rays
        std::function< void( id< MTLCommandBuffer > ) >                   rasterDecals;
        // Rasterizer::DrawToFinalImage, after ImageComposition::PrepareForRaster;
        // if the texture is not null, draw into it as an overlay (see MetalRasterizer::DrawToFinal)
        std::function< void( id< MTLCommandBuffer >, id< MTLTexture > ) > rasterToFinal;
        // Rasterizer::DrawClassic into the upscaled image (or the final, if not upscaled)
        std::function< void( id< MTLCommandBuffer >, id< MTLTexture > ) > rasterClassic;
        // Rasterizer::DrawToSwapchain into the accumulation image, before the post effects on HUD
        std::function< void( id< MTLCommandBuffer >, id< MTLTexture > ) > rasterHud;
    };

    // Registers the own buffers to 'bindings', then encodes all passes into 'cmd'.
    // Returns the image to present: LDR, linear color, of the upscaled size.
    id< MTLTexture > Render( id< MTLCommandBuffer >              cmd,
                             const Params&                       params,
                             const RgDrawFrameInfo&              drawInfo,
                             MetalFramebuffers&                  framebuffers,
                             MetalBindingTable&                  bindings );

    // the image that Render returned the last time
    id< MTLTexture > GetPrevPresented() const { return prevPresented; }
    // Params::hudlessCopy: the same as Render's result, but without HUD; valid until the next Render
    id< MTLTexture > GetHudlessImage() const { return hudlessImage; }

    // MTLFXTemporalDenoisedScaler
    static bool IsMetalFXDenoiserSupported( id< MTLDevice > device );
    // the last MetalFX scaler that was used: to link a frame interpolator to it
    id GetActiveScaler() const { return activeScaler; }

    // Debug: RTGL1_METAL_SHOW for the own textures, e.g. "denoiserReactive"
    id< MTLTexture > GetDebugTexture( std::string_view name ) const;

private:
    void PrepareReservoirs( uint32_t width, uint32_t height );
    // id< MTLFXTemporalScaler >
    id   PrepareTemporalScaler( uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight );
    // id< MTLFXTemporalDenoisedScaler >, and the guide images of the input size
    id   PrepareDenoisedScaler( uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight );
    void BeginPass( id< MTLCommandBuffer > cmd, __strong id< MTLComputeCommandEncoder >& enc, NSString* name );
    void EndPass( __strong id< MTLComputeCommandEncoder >& enc );
    void PrintProfile( id< MTLCommandBuffer > cmd );

private:
    id< MTLDevice >        device;
    MetalComputePipelines& shaders;

    // EffectSimple_Instances.h: their state between frames
    struct Effects;
    std::unique_ptr< Effects > effects;

    id< MTLComputePipelineState > blitScaled;
    id< MTLSamplerState >         nearestSampler;
    id< MTLSamplerState >         linearSampler;

    // MetalFX temporal scaler, recreated on resize
    id                            temporalScaler{ nil };
    uint32_t                      scalerInputWidth{ 0 }, scalerInputHeight{ 0 };
    uint32_t                      scalerOutputWidth{ 0 }, scalerOutputHeight{ 0 };
    bool                          scalerNeedsReset{ true };

    // MetalFX temporal denoised scaler, recreated on resize; with its guides (CmComposeUnfiltered)
    id                            denoisedScaler{ nil };
    uint32_t                      denoiserInputWidth{ 0 }, denoiserInputHeight{ 0 };
    uint32_t                      denoiserOutputWidth{ 0 }, denoiserOutputHeight{ 0 };
    bool                          denoiserNeedsReset{ true };
    id< MTLTexture >              denoiserDiffuseAlbedo{ nil };
    id< MTLTexture >              denoiserSpecularAlbedo{ nil };
    id< MTLTexture >              denoiserNormal{ nil };
    id< MTLTexture >              denoiserRoughness{ nil };
    // CmDenoiserReactive: how much to ignore the history; the current and the previous frame's
    id< MTLTexture >              denoiserReactive[ 2 ]{};
    uint32_t                      denoiserReactiveCur{ 0 };
    // CmPrepareFinalDenoiser: noise-free volumetrics and emission, on top of the denoised
    id< MTLTexture >              denoiserOverlay{ nil };
    // CmDenoiserReactive: 1 where not to denoise
    id< MTLTexture >              denoiserStrength{ nil };
    // MetalRasterizer::DrawToFinal 'asOverlay': the rasterized geometry, premultiplied, with coverage
    id< MTLTexture >              denoiserRasterOverlay{ nil };
    // to reset the histories, when switching between the denoisers
    bool                          wasMetalFXDenoiser{ false };
    id                            activeScaler{ nil };

    // Volumetric: scattering volumes, one per frame index, the previous one is the history
    id< MTLTexture >      scattering[ MAX_FRAMES_IN_FLIGHT ];
    id< MTLSamplerState > volumeSampler;

    // EffectWipe
    struct WipePush
    {
        uint32_t stripWidthInPixels;
        uint32_t startFrameId;
        float    beginTime;
        float    endTime;
    };
    WipePush         wipePush{};
    id< MTLTexture > prevPresented{ nil };

    // Params::hudlessCopy: the image before HUD, and Render's result (as the effects' framebufs are reused)
    id< MTLTexture > hudlessBeforeHud{ nil };
    id< MTLTexture > withHud{ nil };
    id< MTLTexture > hudlessImage{ nil };

    id< MTLBuffer > tonemapping;
    id< MTLBuffer > lpmParams[ MAX_FRAMES_IN_FLIGHT ];
    id< MTLBuffer > indirectReservoirs[ MAX_FRAMES_IN_FLIGHT ];
    uint32_t        reservoirsWidth{ 0 };
    uint32_t        reservoirsHeight{ 0 };

    static constexpr uint32_t                   MaxProfiledPasses = 32;
    bool                                        profiling{ false };
    id< MTLCounterSampleBuffer >                counters{ nil };
    std::array< NSString*, MaxProfiledPasses > passNames{};
    uint32_t                                    passCount{ 0 };
    uint64_t                                    profiledFrames{ 0 };
};

}
