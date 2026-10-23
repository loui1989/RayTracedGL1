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

// macOS, RTGL1_METAL_ONLY=1: the Vulkan functions RTGL1 uses are implemented directly on
// Metal (MetalNative.mm, MetalBridge.mm), and MoltenVK isn't used at all. The functions
// here have the names and types of the Vulkan ones; MetalShims.h routes every call to
// them when Enabled(), to the Vulkan loader otherwise.

#ifdef __APPLE__

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

namespace RTGL1::MetalNative
{

// RTGL1_METAL_ONLY=1 was set (read once); always true when built with RG_METAL_ONLY
bool Enabled();

// a command buffer allocated by the native implementation
bool IsNativeCmd( VkCommandBuffer cmd );

#define RG_NATIVE_FUNCTIONS( X )                         \
    X( vkCreateInstance )                                \
    X( vkDestroyInstance )                               \
    X( vkEnumerateInstanceExtensionProperties )          \
    X( vkEnumerateInstanceLayerProperties )              \
    X( vkEnumeratePhysicalDevices )                      \
    X( vkGetPhysicalDeviceProperties )                   \
    X( vkGetPhysicalDeviceProperties2 )                  \
    X( vkGetPhysicalDeviceFeatures )                     \
    X( vkGetPhysicalDeviceFeatures2 )                    \
    X( vkGetPhysicalDeviceMemoryProperties )             \
    X( vkGetPhysicalDeviceMemoryProperties2 )            \
    X( vkGetPhysicalDeviceQueueFamilyProperties )        \
    X( vkGetPhysicalDeviceFormatProperties )             \
    X( vkEnumerateDeviceExtensionProperties )            \
    X( vkCreateDevice )                                  \
    X( vkDestroyDevice )                                 \
    X( vkGetDeviceQueue )                                \
    X( vkGetDeviceProcAddr )                             \
    X( vkGetInstanceProcAddr )                           \
    X( vkDeviceWaitIdle )                                \
    X( vkQueueWaitIdle )                                 \
    X( vkQueueSubmit )                                   \
    X( vkCreateMetalSurfaceEXT )                         \
    X( vkDestroySurfaceKHR )                             \
    X( vkGetPhysicalDeviceSurfaceSupportKHR )            \
    X( vkGetPhysicalDeviceSurfaceCapabilitiesKHR )       \
    X( vkGetPhysicalDeviceSurfaceFormatsKHR )            \
    X( vkGetPhysicalDeviceSurfacePresentModesKHR )       \
    X( vkCreateSwapchainKHR )                            \
    X( vkDestroySwapchainKHR )                           \
    X( vkGetSwapchainImagesKHR )                         \
    X( vkAcquireNextImageKHR )                           \
    X( vkQueuePresentKHR )                               \
    X( vkAllocateMemory )                                \
    X( vkFreeMemory )                                    \
    X( vkMapMemory )                                     \
    X( vkUnmapMemory )                                   \
    X( vkFlushMappedMemoryRanges )                       \
    X( vkInvalidateMappedMemoryRanges )                  \
    X( vkBindBufferMemory )                              \
    X( vkBindImageMemory )                               \
    X( vkBindBufferMemory2 )                             \
    X( vkBindImageMemory2 )                              \
    X( vkGetBufferMemoryRequirements )                   \
    X( vkGetImageMemoryRequirements )                    \
    X( vkGetBufferMemoryRequirements2 )                  \
    X( vkGetImageMemoryRequirements2 )                   \
    X( vkCreateBuffer )                                  \
    X( vkDestroyBuffer )                                 \
    X( vkGetBufferDeviceAddress )                        \
    X( vkCreateImage )                                   \
    X( vkDestroyImage )                                  \
    X( vkCreateImageView )                               \
    X( vkDestroyImageView )                              \
    X( vkCreateSampler )                                 \
    X( vkDestroySampler )                                \
    X( vkCreateDescriptorSetLayout )                     \
    X( vkDestroyDescriptorSetLayout )                    \
    X( vkCreateDescriptorPool )                          \
    X( vkDestroyDescriptorPool )                         \
    X( vkResetDescriptorPool )                           \
    X( vkAllocateDescriptorSets )                        \
    X( vkFreeDescriptorSets )                            \
    X( vkUpdateDescriptorSets )                          \
    X( vkCreatePipelineLayout )                          \
    X( vkDestroyPipelineLayout )                         \
    X( vkCreateShaderModule )                            \
    X( vkDestroyShaderModule )                           \
    X( vkCreatePipelineCache )                           \
    X( vkDestroyPipelineCache )                          \
    X( vkCreateComputePipelines )                        \
    X( vkCreateGraphicsPipelines )                       \
    X( vkDestroyPipeline )                               \
    X( vkCreateRenderPass )                              \
    X( vkDestroyRenderPass )                             \
    X( vkCreateFramebuffer )                             \
    X( vkDestroyFramebuffer )                            \
    X( vkCreateCommandPool )                             \
    X( vkDestroyCommandPool )                            \
    X( vkResetCommandPool )                              \
    X( vkAllocateCommandBuffers )                        \
    X( vkFreeCommandBuffers )                            \
    X( vkBeginCommandBuffer )                            \
    X( vkEndCommandBuffer )                              \
    X( vkCreateFence )                                   \
    X( vkDestroyFence )                                  \
    X( vkWaitForFences )                                 \
    X( vkResetFences )                                   \
    X( vkGetFenceStatus )                                \
    X( vkCreateSemaphore )                               \
    X( vkDestroySemaphore )                              \
    X( vkCreateQueryPool )                               \
    X( vkDestroyQueryPool )                              \
    X( vkGetQueryPoolResults )                           \
    X( vkCreateDebugUtilsMessengerEXT )                  \
    X( vkDestroyDebugUtilsMessengerEXT )                 \
    X( vkSetDebugUtilsObjectNameEXT )                    \
    X( vkCmdBeginDebugUtilsLabelEXT )                    \
    X( vkCmdEndDebugUtilsLabelEXT )                      \
    X( vkCmdPipelineBarrier2KHR )                        \
    X( vkCreateAccelerationStructureKHR )                \
    X( vkDestroyAccelerationStructureKHR )               \
    X( vkGetAccelerationStructureDeviceAddressKHR )      \
    X( vkGetAccelerationStructureBuildSizesKHR )         \
    X( vkCmdBuildAccelerationStructuresKHR )             \
    X( vkCmdTraceRaysKHR )                               \
    X( vkGetRayTracingShaderGroupHandlesKHR )            \
    X( vkCreateRayTracingPipelinesKHR )                  \
    X( vkExportMetalObjectsEXT )                         \
    X( vkCmdBindPipeline )                               \
    X( vkCmdBindDescriptorSets )                         \
    X( vkCmdPushConstants )                              \
    X( vkCmdDispatch )                                   \
    X( vkCmdPipelineBarrier )                            \
    X( vkCmdCopyBuffer )                                 \
    X( vkCmdCopyBufferToImage )                          \
    X( vkCmdBlitImage )                                  \
    X( vkCmdWriteTimestamp )                             \
    X( vkCmdResetQueryPool )                             \
    X( vkCmdBeginRenderPass )                            \
    X( vkCmdEndRenderPass )                              \
    X( vkCmdBindVertexBuffers )                          \
    X( vkCmdBindIndexBuffer )                            \
    X( vkCmdSetViewport )                                \
    X( vkCmdSetScissor )                                 \
    X( vkCmdDraw )                                       \
    X( vkCmdDrawIndexed )                                \
    X( vkCmdDrawIndexedIndirectCount )                   \
    X( vkCmdClearAttachments )                           \
    X( vkCmdFillBuffer )                                 \
    X( vkCmdClearColorImage )                            \
    X( vkCmdCopyImage )                                  \
    X( vkGetImageSubresourceLayout )

#define RG_NATIVE_DECLARE( name ) decltype( ::name ) name;
RG_NATIVE_FUNCTIONS( RG_NATIVE_DECLARE )
#undef RG_NATIVE_DECLARE

}

#ifdef RG_METAL_ONLY
// built without a Vulkan loader (no MoltenVK): every call goes to the native implementation
#define RG_VK( name ) ( &RTGL1::MetalNative::name )
#else
// a call to the native implementation or to the Vulkan loader
#define RG_VK( name ) ( RTGL1::MetalNative::Enabled() ? &RTGL1::MetalNative::name : &::name )
#endif

#endif // __APPLE__
