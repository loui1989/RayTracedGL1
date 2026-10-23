// Copyright (c) 2026 Quake RT for macOS contributors
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

// Objects of the native implementation (RTGL1_METAL_ONLY=1), shared by MetalNative.mm and
// MetalBridge.mm. Objective-C++ only. A Vulkan handle is a pointer to one of these.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "MetalNative.h"

namespace RTGL1::MetalNative
{

struct NativeMemory
{
    VkDeviceSize    size      = 0;
    uint32_t        typeIndex = 0;
    bool            hostVisible = false;
    id< MTLBuffer > buffer; // created when first needed (memory used only by images has none)
};

struct NativeBuffer
{
    VkDeviceSize       size   = 0;
    VkBufferUsageFlags usage  = 0;
    NativeMemory*      memory = nullptr;
    VkDeviceSize       offset = 0;
};

struct NativeSwapchain;

struct NativeImage
{
    VkFormat           format = VK_FORMAT_UNDEFINED;
    VkImageType        type   = VK_IMAGE_TYPE_2D;
    VkExtent3D         extent = {};
    uint32_t           mipLevels = 1, arrayLayers = 1;
    VkImageUsageFlags  usage = 0;
    VkImageCreateFlags flags = 0;
    id< MTLTexture >   texture;
    NSUInteger         memorySize = 0, memoryAlign = 256;

    // a swapchain image: its texture is the drawable acquired for it
    NativeSwapchain* swapchain = nullptr;
    uint32_t         swapchainIndex = 0;
};

struct NativeImageView
{
    NativeImage*     image = nullptr;
    VkFormat         format = VK_FORMAT_UNDEFINED;
    id< MTLTexture > texture;
};

struct NativeSemaphore
{
    id< MTLSharedEvent > event;
    uint64_t             value    = 0; // binary: the value of the latest signal
    bool                 timeline = false;
};

struct NativeFence
{
    std::mutex              lock;
    std::condition_variable cv;
    bool                    signaled = false;
    // owns this object until vkDestroyFence; completion handlers keep their own copy,
    // so a handler that runs after the fence was destroyed doesn't touch freed memory
    std::shared_ptr< NativeFence > self;

    static void Signal( const std::shared_ptr< NativeFence >& f )
    {
        std::lock_guard l( f->lock );
        f->signaled = true;
        f->cv.notify_all();
    }
};

struct NativeSurface
{
    CAMetalLayer* layer;
};

struct NativeSwapchain
{
    NativeSurface*                         surface = nullptr;
    std::vector< NativeImage* >            images;
    std::vector< id< CAMetalDrawable > >   drawables;
    uint32_t                               next   = 0;
    VkFormat                               format = VK_FORMAT_UNDEFINED;
    VkExtent2D                             extent = {};
};

struct NativeCommandPool;

struct NativeCommandBuffer
{
    NativeCommandPool*                    pool = nullptr;
    std::vector< std::function< void() > > commands; // replayed at vkQueueSubmit
};

struct NativeCommandPool
{
    std::vector< NativeCommandBuffer* > buffers;
};

struct NativeQueryPool
{
    uint32_t count = 0;
};

struct NativeAccelerationStructure
{
    VkDeviceAddress address = 0;
};

// the Metal device and the queue everything is submitted to
id< MTLDevice >       Device();
id< MTLCommandQueue > Queue();

// the drawable's texture of a swapchain image, nil for other images
id< MTLTexture > SwapchainTexture( VkImage image );
// the image's Vulkan format is sRGB but its texture isn't (swapchain images: the shaders
// that write them must encode)
bool NeedsSrgbEncode( VkImage image );
VkFormat ImageFormat( VkImage image );

struct FormatInfo
{
    MTLPixelFormat mtl          = MTLPixelFormatInvalid;
    uint32_t       blockWidth   = 1;
    uint32_t       blockHeight  = 1;
    uint32_t       bytesPerBlock = 0;
    bool           depth = false, stencil = false, srgb = false, compressed = false;
};
FormatInfo GetFormatInfo( VkFormat format );

// the queue's work submitted so far is done (for vkQueueWaitIdle / vkDeviceWaitIdle)
void WaitIdle();
// every command buffer committed to Queue() is noted, for WaitIdle()
void NoteCommitted( id< MTLCommandBuffer > cb );

// the memory a buffer is bound to and its offset in it (null if not bound)
VkDeviceMemory MemoryOf( VkBuffer buffer, VkDeviceSize& offset );
// a buffer's size (0 if unknown)
VkDeviceSize BufferSize( VkBuffer buffer );
// an image's own texture (nil for swapchain images, see SwapchainTexture)
id< MTLTexture > TextureOf( VkImage image );

}
