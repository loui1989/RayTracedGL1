// Copyright (c) 2020-2021 Sultim Tsyrendashiev
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

#include <cassert>
#include <cmath>

#include "DLSS2.h"
#include "DLSSRR.h"
#include "DLSS3_DX12.h"
#include "FSR2.h"
#include "FSR3_DX12.h"
#include "RgException.h"
#include "ResolutionState.h"

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( error : 4061 ) // switch must contain all cases
#endif

namespace RTGL1
{

class RenderResolutionHelper
{
public:
    RenderResolutionHelper()  = default;
    ~RenderResolutionHelper() = default;

    RenderResolutionHelper( const RenderResolutionHelper& )                = delete;
    RenderResolutionHelper( RenderResolutionHelper&& ) noexcept            = delete;
    RenderResolutionHelper& operator=( const RenderResolutionHelper& )     = delete;
    RenderResolutionHelper& operator=( RenderResolutionHelper&& ) noexcept = delete;

    void Setup( const RgStartFrameRenderResolutionParams& params,
                uint32_t                                  windowWidth,
                uint32_t                                  windowHeight,
                const FSR2*                               fsr2,
                const FSR3_DX12*                          fsr3dx12,
                const DLSS2*                              dlss2,
                const DLSS3_DX12*                         dlss3dx12,
                const DLSSRR*                             dlssRr = nullptr )
    {
        // HACKHACK: render into something when the window is minimized
        if( windowWidth == 0 || windowHeight == 0 )
        {
            windowWidth  = 32;
            windowHeight = 32;
        }

        renderWidth  = windowWidth;
        renderHeight = windowHeight;

        upscaledWidth  = windowWidth;
        upscaledHeight = windowHeight;

        upscaleTechnique   = params.upscaleTechnique;
        sharpenTechnique   = params.sharpenTechnique;
        dlssPreset         = params.dlssPreset;
        dlssRrPreset       = params.dlssRrPreset;
        resolutionMode     = params.resolutionMode;
        rayReconstruction  = params.rayReconstruction && dlssRr != nullptr;

        // Report what the caller actually asked for, next to which backends
        // exist. This separates "the game sent the wrong params" from "RTGL
        // downgraded them", which is otherwise indistinguishable from outside:
        // every rejection path here silently falls back to another upscaler and
        // drops rayReconstruction. Finding that gzdoom was sending
        // upscaleTechnique=AMD_FSR2 alongside rayReconstruction=1 (a stale
        // rt_upscale_fsr2 in its ini) took an entire investigation without it.
        // Edge-triggered: a couple of lines per session.
        {
            static bool s_have = false;
            static int  s_prev = -1;
            const int   st     = int( params.upscaleTechnique ) |
                             ( int( params.rayReconstruction != 0 ) << 8 ) |
                             ( int( dlss2 != nullptr ) << 9 ) | ( int( dlssRr != nullptr ) << 10 ) |
                             ( int( dlss3dx12 != nullptr ) << 11 );
            if( !s_have || s_prev != st )
            {
                s_have = true;
                s_prev = st;
                debug::Warning( "Setup(): params.upscaleTechnique={} params.rayReconstruction={} "
                                "| dlss2={} dlss3dx12={} dlssRr={} fsr2={}",
                                int( params.upscaleTechnique ),
                                int( params.rayReconstruction ),
                                dlss2 ? "yes" : "NULL",
                                dlss3dx12 ? "yes" : "NULL",
                                dlssRr ? "yes" : "NULL",
                                fsr2 ? "yes" : "NULL" );
            }
        }

        // check for correct values
        {
            switch( upscaleTechnique )
            {
                case RG_RENDER_UPSCALE_TECHNIQUE_NEAREST:
                case RG_RENDER_UPSCALE_TECHNIQUE_LINEAR: break;
                case RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2:
                    if( !fsr2 && !fsr3dx12 )
                    {
                        upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NEAREST;
                    }
                    break;
                case RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS:
                    if( !dlss2 && !dlss3dx12 && !dlssRr )
                    {
                        upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NEAREST;
                    }
                    break;
                default: upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_NEAREST; break;
            }
            if( rayReconstruction &&
                ( upscaleTechnique != RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS || !dlssRr ) )
            {
                rayReconstruction = false;
            }
            switch( sharpenTechnique )
            {
                case RG_RENDER_SHARPEN_TECHNIQUE_NONE:
                case RG_RENDER_SHARPEN_TECHNIQUE_NAIVE:
                case RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS: break;
                default: sharpenTechnique = RG_RENDER_SHARPEN_TECHNIQUE_NONE; break;
            }
            switch( resolutionMode )
            {
                case RG_RENDER_RESOLUTION_MODE_CUSTOM:
                case RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE:
                case RG_RENDER_RESOLUTION_MODE_PERFORMANCE:
                case RG_RENDER_RESOLUTION_MODE_BALANCED:
                case RG_RENDER_RESOLUTION_MODE_QUALITY:
                case RG_RENDER_RESOLUTION_MODE_NATIVE_AA: break;
                default: resolutionMode = RG_RENDER_RESOLUTION_MODE_BALANCED; break;
            }
        }


        if( upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 )
        {
            assert( fsr2 || fsr3dx12 );
            if( resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM )
            {
                renderWidth  = params.customRenderSize.width;
                renderHeight = params.customRenderSize.height;
            }
            else
            {
                float div = 1.0f;

                switch( resolutionMode )
                {
                    case RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE: div = 3.0f; break;
                    case RG_RENDER_RESOLUTION_MODE_PERFORMANCE: div = 2.0f; break;
                    case RG_RENDER_RESOLUTION_MODE_BALANCED: div = 1.7f; break;
                    case RG_RENDER_RESOLUTION_MODE_QUALITY: div = 1.5f; break;
                    case RG_RENDER_RESOLUTION_MODE_NATIVE_AA: div = 1.0f; break;
                    case RG_RENDER_RESOLUTION_MODE_CUSTOM:
                    default: assert( 0 ); break;
                }

                renderWidth = static_cast< uint32_t >( static_cast< float >( windowWidth ) / div );
                renderHeight =
                    static_cast< uint32_t >( static_cast< float >( windowHeight ) / div );
            }
        }
        else if( upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS )
        {
            assert( dlss2 || dlss3dx12 || dlssRr );
            if( resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM )
            {
                renderWidth  = params.customRenderSize.width;
                renderHeight = params.customRenderSize.height;
            }
            else
            {
                if( rayReconstruction && dlssRr )
                {
                    std::tie( renderWidth, renderHeight ) =
                        dlssRr->GetOptimalSettings( windowWidth, windowHeight, resolutionMode );
                }
                else if( dlss3dx12 )
                {
                    std::tie( renderWidth, renderHeight ) =
                        dlss3dx12->GetOptimalSettings( windowWidth, windowHeight, resolutionMode );
                }
                else if( dlss2 )
                {
                    std::tie( renderWidth, renderHeight ) =
                        dlss2->GetOptimalSettings( windowWidth, windowHeight, resolutionMode );
                }

                // ultra quality returns (0,0)
                if( renderWidth == 0 || renderHeight == 0 )
                {
                    renderWidth  = windowWidth;
                    renderHeight = windowHeight;
                }
            }
        }
        else
        {
            if( resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM )
            {
                renderWidth  = params.customRenderSize.width;
                renderHeight = params.customRenderSize.height;
            }
        }
    }

    float GetMipLodBias( float nativeBias = 0.0f ) const
    {
        // softer if none
        if( !IsUpscaleEnabled() )
        {
            return nativeBias;
        }

        // DLSS Programming Guide, Section 3.5
        float ratio = float( Width() ) / float( UpscaledWidth() );
        float bias  = nativeBias + std::log2( std::max( 0.01f, ratio ) ) - 1.0f;

        return bias;
    }

    // Render width always must be even for checkerboarding!
    uint32_t Width() const { return renderWidth + renderWidth % 2; }
    uint32_t Height() const { return renderHeight; }

    uint32_t UpscaledWidth() const { return upscaledWidth; }
    uint32_t UpscaledHeight() const { return upscaledHeight; }

    float Aspect() const
    {
        if( UpscaledWidth() == 0 || UpscaledHeight() == 0 )
        {
            return 1.0f;
        }
        return float( UpscaledWidth() ) / float( UpscaledHeight() );
    }

    bool IsAmdFsr2Enabled() const
    {
        return upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2;
    }
    bool IsNvDlssEnabled() const
    {
        return upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
    }
    bool IsNvDlssRayReconstructionEnabled() const
    {
        return IsNvDlssEnabled() && rayReconstruction;
    }
    bool IsUpscaleEnabled() const { return IsAmdFsr2Enabled() || IsNvDlssEnabled(); }

    float GetAmdFsrSharpness() const { return 1.0f; } // 0.0 - max, 1.0 - min

    bool IsCASInsideFSR2() const
    {
#ifdef __APPLE__
        return false; // MetalFX fills the FSR2 slot and has no built-in sharpening
#else
        return upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 &&
               sharpenTechnique == RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS;
#endif
    }

    // For the additional sharpening pass
    bool IsDedicatedSharpeningEnabled() const
    {
        return IsCASInsideFSR2() ? false : sharpenTechnique != RG_RENDER_SHARPEN_TECHNIQUE_NONE;
    }
    RgRenderSharpenTechnique GetSharpeningTechnique() const { return sharpenTechnique; }
    float                    GetSharpeningIntensity() const { return 1.0f; }

    VkFilter GetBlitFilter() const
    {
        return upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_NEAREST ? VK_FILTER_NEAREST
                                                                       : VK_FILTER_LINEAR;
    }

    RgRenderResolutionMode GetResolutionMode() const { return resolutionMode; }

    // Doom64-RT: the NVSDK_NGX_DLSS_Hint_Render_Preset to create the DLSS
    // feature with. Kept here rather than in ResolutionState because that struct
    // also drives framebuffer allocation, and a preset change must not be
    // mistaken for a resize.
    uint32_t GetDlssPreset() const { return dlssPreset; }

    // Doom64-RT: the Ray Reconstruction twin. Same reasoning -- a preset change
    // must not be mistaken for a resize, so it is not in ResolutionState.
    uint32_t GetDlssRrPreset() const { return dlssRrPreset; }

    ResolutionState GetResolutionState() const
    {
        assert( Width() % 2 == 0 );
        return ResolutionState{ Width(), Height(), UpscaledWidth(), UpscaledHeight() };
    }

private:
    uint32_t renderWidth  = 0;
    uint32_t renderHeight = 0;

    uint32_t upscaledWidth  = 0;
    uint32_t upscaledHeight = 0;

    RgRenderUpscaleTechnique upscaleTechnique = RG_RENDER_UPSCALE_TECHNIQUE_LINEAR;
    RgRenderSharpenTechnique sharpenTechnique = RG_RENDER_SHARPEN_TECHNIQUE_NONE;
    RgRenderResolutionMode   resolutionMode   = RG_RENDER_RESOLUTION_MODE_CUSTOM;
    bool                     rayReconstruction = false;
    // 5 = NVSDK_NGX_DLSS_Hint_Render_Preset_E, which is what RTGL1 hard-coded
    // before this was configurable.
    uint32_t                 dlssPreset        = 5;
    // 5 = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E, what DLSSRR.cpp
    // hard-coded before this was configurable (D was A/B'd 2026-08-07 and was
    // clearly worse). NOTE unlike SR, only 0 / 4 / 5 mean anything for RR.
    uint32_t                 dlssRrPreset      = 5;
};

}

#ifdef _MSC_VER
#pragma warning( pop )
#endif
