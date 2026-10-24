/*

Copyright (c) 2026 Doom64-RT contributors
Copyright (c) 2024 V.Shirokii
Copyright (c) 2021 Sultim Tsyrendashiev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#pragma once

#include <vector>

#include "Camera.h"
#include "Framebuffers.h"
#include "ResolutionState.h"

struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

namespace RTGL1
{

class RenderResolutionHelper;
class DLSS2;

// Native NGX Vulkan DLSS Ray Reconstruction. Requires DLSS2 to have already
// initialized NGX (shares that context; does not Init/Shutdown NGX itself).
class DLSSRR
{
public:
    static auto MakeInstance( VkDevice device, const DLSS2* dlss2 ) -> std::shared_ptr< DLSSRR >;

    DLSSRR( VkDevice device, bool ngxAlreadyInitialized );
    ~DLSSRR();

    DLSSRR( const DLSSRR& )                = delete;
    DLSSRR( DLSSRR&& ) noexcept            = delete;
    DLSSRR& operator=( const DLSSRR& )     = delete;
    DLSSRR& operator=( DLSSRR&& ) noexcept = delete;

    auto Apply( VkCommandBuffer               cmd,
                uint32_t                      frameIndex,
                Framebuffers&                 framebuffers,
                const RenderResolutionHelper& renderResolution,
                RgFloat2D                     jitterOffset,
                double                        timeDelta,
                bool                          resetAccumulation,
                bool                          specHitDistEnabled,
                bool                          disoccMaskEnabled,
                bool                          exposureTexEnabled,
                bool                          transparencyLayerEnabled,
                const float*                  worldToViewMatrix16,
                const float*                  viewToClipMatrix16 ) -> FramebufferImageIndex;

    auto GetOptimalSettings( uint32_t               userWidth,
                             uint32_t               userHeight,
                             RgRenderResolutionMode mode ) const -> std::pair< uint32_t, uint32_t >;

    bool Valid() const;

#ifdef __APPLE__
    // macOS (DLSSRR_Metal.mm, the MetalFX denoiser): if it can upscale renderSize -> outSize
    bool SupportsScale( uint32_t renderWidth, uint32_t renderHeight, uint32_t outWidth, uint32_t outHeight ) const;
    // the colour is lighting only (CmNoisyCompose, rrDemod): multiply FB_IMAGE_INDEX_RR_DEMOD_FACTOR
    // back after denoising
    void SetDemodulation( bool demod, uint32_t filter )
    {
        m_demod       = demod;
        m_demodFilter = filter;
    }

private:
    bool     m_demod{ false };
    uint32_t m_demodFilter{ 1 };

public:
#endif

private:
    void Destroy();

private:
    VkDevice m_device{};

    bool                 m_initialized{ false };
    NVSDK_NGX_Parameter* m_params{ nullptr };

    NVSDK_NGX_Handle* m_feature{ nullptr };
    ResolutionState   m_prevResolution{};
    // UINT32_MAX sentinel, so the first frame always creates the feature even
    // if the requested preset happens to equal the old hard-coded E -- the
    // same reasoning as DLSS2.h's m_prevPreset.
    uint32_t          m_prevPreset{ UINT32_MAX };
};

}
