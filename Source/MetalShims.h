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
#include "MetalNative.h"

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
#define vkDestroyImage(...)                RTGL1::Metal::DestroyImage( __VA_ARGS__ )
#define vkCreateShaderModule(...)          RTGL1::Metal::CreateShaderModule( __VA_ARGS__ )
#define vkDestroyShaderModule(...)         RTGL1::Metal::DestroyShaderModule( __VA_ARGS__ )
#define vkCreateComputePipelines(...)      RTGL1::Metal::CreateComputePipelines( __VA_ARGS__ )
#define vkDestroyPipeline(...)             RTGL1::Metal::DestroyPipeline( __VA_ARGS__ )
#define vkCreateGraphicsPipelines(...)     RTGL1::Metal::CreateGraphicsPipelines( __VA_ARGS__ )
#define vkCreateRenderPass(...)            RTGL1::Metal::CreateRenderPass( __VA_ARGS__ )
#define vkDestroyRenderPass(...)           RTGL1::Metal::DestroyRenderPass( __VA_ARGS__ )
#define vkCreateFramebuffer(...)           RTGL1::Metal::CreateFramebuffer( __VA_ARGS__ )
#define vkDestroyFramebuffer(...)          RTGL1::Metal::DestroyFramebuffer( __VA_ARGS__ )

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
#define vkCmdEndRenderPass(...)            RgMtl_vkCmdEndRenderPass( __VA_ARGS__ )
#define vkCmdBindVertexBuffers(...)        RgMtl_vkCmdBindVertexBuffers( __VA_ARGS__ )
#define vkCmdBindIndexBuffer(...)          RgMtl_vkCmdBindIndexBuffer( __VA_ARGS__ )
#define vkCmdSetViewport(...)              RgMtl_vkCmdSetViewport( __VA_ARGS__ )
#define vkCmdSetScissor(...)               RgMtl_vkCmdSetScissor( __VA_ARGS__ )
#define vkCmdDraw(...)                     RgMtl_vkCmdDraw( __VA_ARGS__ )
#define vkCmdDrawIndexed(...)              RgMtl_vkCmdDrawIndexed( __VA_ARGS__ )
#define vkCmdDrawIndexedIndirectCount(...) RgMtl_vkCmdDrawIndexedIndirectCount( __VA_ARGS__ )
#define vkCmdClearAttachments(...)         RgMtl_vkCmdClearAttachments( __VA_ARGS__ )

// RTGL1_METAL_ONLY=1: the native implementation (MetalNative.h), otherwise the Vulkan loader
#define vkCreateInstance(...) RG_VK( vkCreateInstance )( __VA_ARGS__ )
#define vkDestroyInstance(...) RG_VK( vkDestroyInstance )( __VA_ARGS__ )
#define vkEnumerateInstanceExtensionProperties(...) RG_VK( vkEnumerateInstanceExtensionProperties )( __VA_ARGS__ )
#define vkEnumerateInstanceLayerProperties(...) RG_VK( vkEnumerateInstanceLayerProperties )( __VA_ARGS__ )
#define vkEnumeratePhysicalDevices(...) RG_VK( vkEnumeratePhysicalDevices )( __VA_ARGS__ )
#define vkGetPhysicalDeviceProperties(...) RG_VK( vkGetPhysicalDeviceProperties )( __VA_ARGS__ )
#define vkGetPhysicalDeviceProperties2(...) RG_VK( vkGetPhysicalDeviceProperties2 )( __VA_ARGS__ )
#define vkGetPhysicalDeviceFeatures(...) RG_VK( vkGetPhysicalDeviceFeatures )( __VA_ARGS__ )
#define vkGetPhysicalDeviceFeatures2(...) RG_VK( vkGetPhysicalDeviceFeatures2 )( __VA_ARGS__ )
#define vkGetPhysicalDeviceMemoryProperties(...) RG_VK( vkGetPhysicalDeviceMemoryProperties )( __VA_ARGS__ )
#define vkGetPhysicalDeviceMemoryProperties2(...) RG_VK( vkGetPhysicalDeviceMemoryProperties2 )( __VA_ARGS__ )
#define vkGetPhysicalDeviceQueueFamilyProperties(...) RG_VK( vkGetPhysicalDeviceQueueFamilyProperties )( __VA_ARGS__ )
#define vkGetPhysicalDeviceFormatProperties(...) RG_VK( vkGetPhysicalDeviceFormatProperties )( __VA_ARGS__ )
#define vkEnumerateDeviceExtensionProperties(...) RG_VK( vkEnumerateDeviceExtensionProperties )( __VA_ARGS__ )
#define vkCreateDevice(...) RG_VK( vkCreateDevice )( __VA_ARGS__ )
#define vkDestroyDevice(...) RG_VK( vkDestroyDevice )( __VA_ARGS__ )
#define vkGetDeviceQueue(...) RG_VK( vkGetDeviceQueue )( __VA_ARGS__ )
#define vkGetDeviceProcAddr(...) RG_VK( vkGetDeviceProcAddr )( __VA_ARGS__ )
#define vkGetInstanceProcAddr(...) RG_VK( vkGetInstanceProcAddr )( __VA_ARGS__ )
#define vkDeviceWaitIdle(...) RG_VK( vkDeviceWaitIdle )( __VA_ARGS__ )
#define vkQueueWaitIdle(...) RG_VK( vkQueueWaitIdle )( __VA_ARGS__ )
#define vkQueueSubmit(...) RG_VK( vkQueueSubmit )( __VA_ARGS__ )
#define vkCreateMetalSurfaceEXT(...) RG_VK( vkCreateMetalSurfaceEXT )( __VA_ARGS__ )
#define vkDestroySurfaceKHR(...) RG_VK( vkDestroySurfaceKHR )( __VA_ARGS__ )
#define vkGetPhysicalDeviceSurfaceSupportKHR(...) RG_VK( vkGetPhysicalDeviceSurfaceSupportKHR )( __VA_ARGS__ )
#define vkGetPhysicalDeviceSurfaceCapabilitiesKHR(...) RG_VK( vkGetPhysicalDeviceSurfaceCapabilitiesKHR )( __VA_ARGS__ )
#define vkGetPhysicalDeviceSurfaceFormatsKHR(...) RG_VK( vkGetPhysicalDeviceSurfaceFormatsKHR )( __VA_ARGS__ )
#define vkGetPhysicalDeviceSurfacePresentModesKHR(...) RG_VK( vkGetPhysicalDeviceSurfacePresentModesKHR )( __VA_ARGS__ )
#define vkCreateSwapchainKHR(...) RG_VK( vkCreateSwapchainKHR )( __VA_ARGS__ )
#define vkDestroySwapchainKHR(...) RG_VK( vkDestroySwapchainKHR )( __VA_ARGS__ )
#define vkGetSwapchainImagesKHR(...) RG_VK( vkGetSwapchainImagesKHR )( __VA_ARGS__ )
#define vkAcquireNextImageKHR(...) RG_VK( vkAcquireNextImageKHR )( __VA_ARGS__ )
#define vkQueuePresentKHR(...) RG_VK( vkQueuePresentKHR )( __VA_ARGS__ )
#define vkAllocateMemory(...) RG_VK( vkAllocateMemory )( __VA_ARGS__ )
#define vkMapMemory(...) RG_VK( vkMapMemory )( __VA_ARGS__ )
#define vkUnmapMemory(...) RG_VK( vkUnmapMemory )( __VA_ARGS__ )
#define vkFlushMappedMemoryRanges(...) RG_VK( vkFlushMappedMemoryRanges )( __VA_ARGS__ )
#define vkInvalidateMappedMemoryRanges(...) RG_VK( vkInvalidateMappedMemoryRanges )( __VA_ARGS__ )
#define vkBindImageMemory(...) RG_VK( vkBindImageMemory )( __VA_ARGS__ )
#define vkBindBufferMemory2(...) RG_VK( vkBindBufferMemory2 )( __VA_ARGS__ )
#define vkBindImageMemory2(...) RG_VK( vkBindImageMemory2 )( __VA_ARGS__ )
#define vkGetBufferMemoryRequirements(...) RG_VK( vkGetBufferMemoryRequirements )( __VA_ARGS__ )
#define vkGetImageMemoryRequirements(...) RG_VK( vkGetImageMemoryRequirements )( __VA_ARGS__ )
#define vkGetBufferMemoryRequirements2(...) RG_VK( vkGetBufferMemoryRequirements2 )( __VA_ARGS__ )
#define vkGetImageMemoryRequirements2(...) RG_VK( vkGetImageMemoryRequirements2 )( __VA_ARGS__ )
#define vkCreateImage(...) RG_VK( vkCreateImage )( __VA_ARGS__ )
#define vkCreateImageView(...) RG_VK( vkCreateImageView )( __VA_ARGS__ )
#define vkCreateDescriptorPool(...) RG_VK( vkCreateDescriptorPool )( __VA_ARGS__ )
#define vkCreatePipelineCache(...) RG_VK( vkCreatePipelineCache )( __VA_ARGS__ )
#define vkDestroyPipelineCache(...) RG_VK( vkDestroyPipelineCache )( __VA_ARGS__ )
#define vkCreateCommandPool(...) RG_VK( vkCreateCommandPool )( __VA_ARGS__ )
#define vkDestroyCommandPool(...) RG_VK( vkDestroyCommandPool )( __VA_ARGS__ )
#define vkResetCommandPool(...) RG_VK( vkResetCommandPool )( __VA_ARGS__ )
#define vkAllocateCommandBuffers(...) RG_VK( vkAllocateCommandBuffers )( __VA_ARGS__ )
#define vkFreeCommandBuffers(...) RG_VK( vkFreeCommandBuffers )( __VA_ARGS__ )
#define vkBeginCommandBuffer(...) RG_VK( vkBeginCommandBuffer )( __VA_ARGS__ )
#define vkEndCommandBuffer(...) RG_VK( vkEndCommandBuffer )( __VA_ARGS__ )
#define vkCreateFence(...) RG_VK( vkCreateFence )( __VA_ARGS__ )
#define vkDestroyFence(...) RG_VK( vkDestroyFence )( __VA_ARGS__ )
#define vkWaitForFences(...) RG_VK( vkWaitForFences )( __VA_ARGS__ )
#define vkResetFences(...) RG_VK( vkResetFences )( __VA_ARGS__ )
#define vkGetFenceStatus(...) RG_VK( vkGetFenceStatus )( __VA_ARGS__ )
#define vkCreateSemaphore(...) RG_VK( vkCreateSemaphore )( __VA_ARGS__ )
#define vkDestroySemaphore(...) RG_VK( vkDestroySemaphore )( __VA_ARGS__ )
#define vkCreateQueryPool(...) RG_VK( vkCreateQueryPool )( __VA_ARGS__ )
#define vkDestroyQueryPool(...) RG_VK( vkDestroyQueryPool )( __VA_ARGS__ )
#define vkGetQueryPoolResults(...) RG_VK( vkGetQueryPoolResults )( __VA_ARGS__ )

#endif
