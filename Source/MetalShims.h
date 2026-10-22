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

// macOS: route the Vulkan calls the native Metal path needs to know about through
// MetalBridge (they are forwarded to Vulkan unchanged). Included at the end of Common.h.
// Function-like macros, so only calls are redirected: other uses of these names (e.g.
// struct members in Vma's VmaVulkanFunctions) stay untouched.

#if defined( __APPLE__ ) && !defined( RG_METAL_BRIDGE_IMPLEMENTATION )

#include "MetalBridge.h"

#define vkCreateDescriptorSetLayout(...)   RTGL1::Metal::CreateDescriptorSetLayout( __VA_ARGS__ )
#define vkDestroyDescriptorSetLayout(...)  RTGL1::Metal::DestroyDescriptorSetLayout( __VA_ARGS__ )
#define vkCreatePipelineLayout(...)        RTGL1::Metal::CreatePipelineLayout( __VA_ARGS__ )
#define vkDestroyPipelineLayout(...)       RTGL1::Metal::DestroyPipelineLayout( __VA_ARGS__ )
#define vkAllocateDescriptorSets(...)      RTGL1::Metal::AllocateDescriptorSets( __VA_ARGS__ )
#define vkFreeDescriptorSets(...)          RTGL1::Metal::FreeDescriptorSets( __VA_ARGS__ )
#define vkDestroyDescriptorPool(...)       RTGL1::Metal::DestroyDescriptorPool( __VA_ARGS__ )
#define vkResetDescriptorPool(...)         RTGL1::Metal::ResetDescriptorPool( __VA_ARGS__ )
#define vkUpdateDescriptorSets(...)        RTGL1::Metal::UpdateDescriptorSets( __VA_ARGS__ )
#define vkCreateSampler(...)               RTGL1::Metal::CreateSampler( __VA_ARGS__ )
#define vkDestroySampler(...)              RTGL1::Metal::DestroySampler( __VA_ARGS__ )
#define vkCreateBuffer(...)                RTGL1::Metal::CreateBuffer( __VA_ARGS__ )
#define vkDestroyBuffer(...)               RTGL1::Metal::DestroyBuffer( __VA_ARGS__ )
#define vkBindBufferMemory(...)            RTGL1::Metal::BindBufferMemory( __VA_ARGS__ )
#define vkGetBufferDeviceAddress(...)      RTGL1::Metal::GetBufferDeviceAddress( __VA_ARGS__ )
#define vkFreeMemory(...)                  RTGL1::Metal::FreeMemory( __VA_ARGS__ )
#define vkDestroyImageView(...)            RTGL1::Metal::DestroyImageView( __VA_ARGS__ )
#define vkCreateShaderModule(...)          RTGL1::Metal::CreateShaderModule( __VA_ARGS__ )
#define vkDestroyShaderModule(...)         RTGL1::Metal::DestroyShaderModule( __VA_ARGS__ )
#define vkCreateComputePipelines(...)      RTGL1::Metal::CreateComputePipelines( __VA_ARGS__ )
#define vkDestroyPipeline(...)             RTGL1::Metal::DestroyPipeline( __VA_ARGS__ )

#define vkCmdBindPipeline(...)             RgMtl_vkCmdBindPipeline( __VA_ARGS__ )
#define vkCmdBindDescriptorSets(...)       RgMtl_vkCmdBindDescriptorSets( __VA_ARGS__ )
#define vkCmdPushConstants(...)            RgMtl_vkCmdPushConstants( __VA_ARGS__ )
#define vkCmdDispatch(...)                 RgMtl_vkCmdDispatch( __VA_ARGS__ )
#define vkCmdPipelineBarrier(...)          RgMtl_vkCmdPipelineBarrier( __VA_ARGS__ )
#define svkCmdPipelineBarrier2KHR(...)     RgMtl_svkCmdPipelineBarrier2KHR( __VA_ARGS__ )
#define vkCmdCopyBuffer(...)               RgMtl_vkCmdCopyBuffer( __VA_ARGS__ )
#define vkCmdCopyBufferToImage(...)        RgMtl_vkCmdCopyBufferToImage( __VA_ARGS__ )
#define vkCmdBlitImage(...)                RgMtl_vkCmdBlitImage( __VA_ARGS__ )
#define vkCmdWriteTimestamp(...)           RgMtl_vkCmdWriteTimestamp( __VA_ARGS__ )
#define vkCmdResetQueryPool(...)           RgMtl_vkCmdResetQueryPool( __VA_ARGS__ )
#define vkCmdBeginRenderPass(...)          RgMtl_vkCmdBeginRenderPass( __VA_ARGS__ )

#endif
