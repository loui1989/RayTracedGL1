// Copyright (c) 2022 Sultim Tsyrendashiev
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

#include <optional>

#include "Framebuffers.h"

struct FfxFsr2Context;

namespace RTGL1
{
    class RenderResolutionHelper;

    class FSR2 : public IFramebuffersDependency
    {
    public:
        FSR2(VkDevice device, VkPhysicalDevice physDevice);
        ~FSR2() override;

        FSR2(const FSR2& other) = delete;
        FSR2(FSR2&& other) noexcept = delete;
        FSR2& operator=(const FSR2& other) = delete;
        FSR2& operator=(FSR2&& other) noexcept = delete;

        void OnFramebuffersSizeChange(const ResolutionState &resolutionState) override;

        FramebufferImageIndex Apply(
            VkCommandBuffer cmd, uint32_t frameIndex,
            const std::shared_ptr<Framebuffers> &framebuffers,
            const RenderResolutionHelper &renderResolution,
            RgFloat2D jitterOffset,
            float timeDelta,
            float nearPlane, float farPlane, float fovVerticalRad);

        static RgFloat2D GetJitter(const ResolutionState &resolutionState, uint32_t frameId);
        static bool IsFsr2Available();
        // macOS (MetalFX): the queue RTGL1 submits to
        void SetQueue(VkQueue q) { queue = q; }
#ifdef __APPLE__
        // MetalFX frame interpolation (experimental, macOS 26+), used together
        // with the MetalFX upscaler.
        static bool IsFrameGenSupported();
        // Whether frame generation may be used; call every frame before Apply().
        void SetFrameGenWanted(bool wanted);
        // Call after the frame (before the HUD) is in 'accum' and all its Vulkan work is
        // submitted. If 'engage', saves the frame and, if possible, replaces 'accum' with
        // a frame generated between the previous and this one. Returns true if it did.
        bool FrameGenInterpolate(uint32_t frameIndex, const std::shared_ptr<Framebuffers> &framebuffers,
                                 FramebufferImageIndex accum, bool engage);
        // Call after the generated frame was presented (or when FrameGenInterpolate
        // returned false): puts the real frame back into 'accum'.
        void FrameGenFinish(uint32_t frameIndex, const std::shared_ptr<Framebuffers> &framebuffers,
                            FramebufferImageIndex accum);

        // timings of the last Apply(), for profiling: CPU wait for the Vulkan queue
        // to be idle, and the MetalFX command buffer's GPU time
        static void GetLastTimings(double &waitVulkanMs, double &metalFxGpuMs);

        // Asynchronous MetalFX, without CPU waits (default; RTGL1_METALFX_SYNC=1 turns it off).
        // If it returns true, the Vulkan work before Apply() must be submitted with 'sem'
        // (a timeline semaphore) signalled to 'vulkanDoneValue', and the Vulkan work after
        // Apply() must wait for 'sem' to reach 'metalFxDoneValue'. Call right before Apply().
        bool PrepareAsync(VkSemaphore &sem, uint64_t &vulkanDoneValue, uint64_t &metalFxDoneValue);
#endif

    private:
        VkDevice device;
        VkPhysicalDevice physDevice;
        VkQueue queue = VK_NULL_HANDLE;

        std::unique_ptr<std::optional<FfxFsr2Context>> context;
    };
}