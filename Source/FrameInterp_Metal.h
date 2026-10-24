// Copyright (c) 2026 Quake RT / Doom 64 RT for macOS contributors
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

// macOS: frame generation is Apple's MetalFX frame interpolator (macOS 26). Each frame, the
// image without HUD and the HUD (drawn into FB_IMAGE_INDEX_HUD_ONLY) are given to it, and it
// generates a frame between the previous and the current one; at vkQueuePresentKHR the
// generated frame is presented, then the actual one, half a frame later.

#ifdef __APPLE__

#include <cstdint>

#include <vulkan/vulkan.h>

namespace RTGL1::MetalFI
{

// MetalFX frame interpolation can be used (macOS 26, a supported GPU, running on Metal)
bool Supported( VkDevice device );

// before drawing the HUD separately: if this frame can be interpolated (creates the interpolator)
bool Prepare( VkDevice device, uint32_t renderWidth, uint32_t renderHeight, uint32_t width, uint32_t height );

struct Params
{
    // the frame without HUD, tonemapped (linear, 0..1), and its size
    VkImage  rendered;
    uint32_t width, height;
    // the HUD (premultiplied alpha, gamma 2.2), the same size; VK_NULL_HANDLE if nothing was drawn
    VkImage  hud;
    // render resolution: NDC depth, motion vectors (in UV units)
    VkImage  depthNdc;
    VkImage  motion;
    uint32_t renderWidth, renderHeight;

    float jitterX, jitterY;
    float nearPlane, farPlane, fovYRadians;
    float deltaTime;
    bool  reset;
    // false: interpolate, but present only the actual frames (RG_FRAME_GENERATION_MODE_WITHOUT_GENERATED)
    bool  presentGenerated;
};

// Records into 'cmd' (a command buffer of the Metal implementation) the actual frame with HUD,
// and the generated one, for the next vkQueuePresentKHR. The swapchain image isn't written
// then (RTGL1's blit to it is skipped). Returns false if not done: present as usual.
bool Encode( VkDevice device, VkCommandBuffer cmd, const Params& params );

// frame generation is off: release everything
void Release();

}

#ifdef __OBJC__
#import <QuartzCore/CAMetalLayer.h>
#import <Metal/Metal.h>
namespace RTGL1::MetalFI
{
// MetalNative.mm, vkQueuePresentKHR: if a frame was encoded by Encode, presents it (generated, then
// actual) with 'cb', which waits for the rendering; returns false if there's nothing to present
bool Present( id< MTLCommandBuffer > cb, CAMetalLayer* layer );
// MetalNative.mm, vkAcquireNextImageKHR: the drawable for the swapchain image is acquired only
// if something writes into it (with frame interpolation, Present acquires its own)
bool WantsLateDrawables();
}
#endif

#endif // __APPLE__
