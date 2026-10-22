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

// macOS: native Metal execution of parts of the frame next to MoltenVK.
//
// The Vulkan calls listed below are intercepted (see MetalShims.h) to keep a shadow copy of
// what the native path needs: descriptor set layouts and contents, samplers, which memory
// backs which buffer, buffer device addresses, SPIR-V of shader modules and the setup of
// compute pipelines. With RTGL1_METAL_RT=1 the path tracer then runs natively: its compute
// shaders are translated to Metal (MetalShaderGen), the Vulkan descriptor sets become Metal
// argument buffers, the acceleration structures are built with Metal, and the work is
// ordered with the Vulkan work through a shared event / timeline semaphore.

#ifdef __APPLE__

#include <cstdint>
#include <functional>
#include <memory>

#include <vulkan/vulkan.h>

namespace RTGL1
{
class CommandBufferManager;
}

namespace RTGL1::Metal
{

// ---- Vulkan call interception (always active on macOS; only records)

VkResult CreateDescriptorSetLayout( VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* );
void     DestroyDescriptorSetLayout( VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks* );
VkResult CreatePipelineLayout( VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* );
void     DestroyPipelineLayout( VkDevice, VkPipelineLayout, const VkAllocationCallbacks* );
VkResult AllocateDescriptorSets( VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet* );
VkResult FreeDescriptorSets( VkDevice, VkDescriptorPool, uint32_t, const VkDescriptorSet* );
void     DestroyDescriptorPool( VkDevice, VkDescriptorPool, const VkAllocationCallbacks* );
VkResult ResetDescriptorPool( VkDevice, VkDescriptorPool, VkDescriptorPoolResetFlags );
void     UpdateDescriptorSets( VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const VkCopyDescriptorSet* );
VkResult CreateSampler( VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*, VkSampler* );
void     DestroySampler( VkDevice, VkSampler, const VkAllocationCallbacks* );
VkResult CreateBuffer( VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* );
void     DestroyBuffer( VkDevice, VkBuffer, const VkAllocationCallbacks* );
VkResult BindBufferMemory( VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize );
VkDeviceAddress GetBufferDeviceAddress( VkDevice, const VkBufferDeviceAddressInfo* );
void     FreeMemory( VkDevice, VkDeviceMemory, const VkAllocationCallbacks* );
void     DestroyImageView( VkDevice, VkImageView, const VkAllocationCallbacks* );
VkResult CreateShaderModule( VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* );
void     DestroyShaderModule( VkDevice, VkShaderModule, const VkAllocationCallbacks* );
VkResult CreateComputePipelines( VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* );
VkResult CreateGraphicsPipelines( VkDevice, VkPipelineCache, uint32_t, const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* );
void     DestroyPipeline( VkDevice, VkPipeline, const VkAllocationCallbacks* );
VkResult CreateRenderPass( VkDevice, const VkRenderPassCreateInfo*, const VkAllocationCallbacks*, VkRenderPass* );
void     DestroyRenderPass( VkDevice, VkRenderPass, const VkAllocationCallbacks* );
VkResult CreateFramebuffer( VkDevice, const VkFramebufferCreateInfo*, const VkAllocationCallbacks*, VkFramebuffer* );
void     DestroyFramebuffer( VkDevice, VkFramebuffer, const VkAllocationCallbacks* );

// ---- native path

// RTGL1_METAL_RT=1 was set and Init() succeeded
bool IsEnabled();
bool Init( VkDevice device, std::shared_ptr< CommandBufferManager > cmdManager );
void Shutdown();
// stops using the native path for the rest of the run (after an error)
void Disable( const char* reason );

// Translates and compiles a compute pipeline created through vkCreateComputePipelines.
bool PreparePipeline( VkPipeline pipeline, const char* debugName );

// Acceleration structures: queued here, built at the start of the next section.
void RegisterAccelerationStructureAddress( VkAccelerationStructureKHR as, VkDeviceAddress address );
// call before vkDestroyAccelerationStructureKHR
void ForgetAccelerationStructure( VkAccelerationStructureKHR as );
void QueueBottomLevelBuilds( uint32_t                                                count,
                             const VkAccelerationStructureBuildGeometryInfoKHR*      infos,
                             const VkAccelerationStructureBuildRangeInfoKHR* const* ranges );
void QueueTopLevelBuild( VkAccelerationStructureKHR                dst,
                         const VkAccelerationStructureInstanceKHR* instances,
                         uint32_t                                  instanceCount,
                         bool                                      preferFastTrace );

// Translates and compiles all compute pipelines that aren't prepared yet, in parallel.
// Pipelines created later (shader reload) are prepared when they're first dispatched.
void PrepareAllPipelines();

// A section of native Metal work in the frame: BeginSection submits 'cmd' and sets it to
// SectionCmd(), EndSection starts a new Vulkan command buffer in 'cmd' that waits for it.
// Inside a section, compute work recorded on SectionCmd() with vkCmdBindPipeline,
// vkCmdBindDescriptorSets, vkCmdPushConstants and vkCmdDispatch runs in Metal (barriers are
// not needed: the passes run in order). Other commands aren't supported there.
void BeginSection( VkCommandBuffer& cmd );
void Dispatch( VkPipeline pipeline, const VkDescriptorSet* sets, uint32_t setCount, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ );
void EndSection( VkCommandBuffer& cmd );
bool IsInSection();

inline VkCommandBuffer SectionCmd()
{
    // never given to Vulkan: the shims below route everything recorded on it to Metal
    return reinterpret_cast< VkCommandBuffer >( uintptr_t( 0x5EC7104D7A11ull ) );
}
inline bool IsSectionCmd( VkCommandBuffer cmd )
{
    return cmd == SectionCmd();
}

// The generic compute passes (denoiser, volumetrics, exposure, ...) can run in a section:
// native path is on, RTGL1_METAL_COMPUTE isn't 0, and no pass failed so far.
bool IsComputeReady();
// Render passes and draws, buffer copies can run in a section too: IsComputeReady(),
// RTGL1_METAL_RASTER isn't 0, and nothing failed so far.
bool IsRasterReady();

// RTGL1_PROFILE=1: GPU time of the work in the section since the previous mark.
// 'name' must be a string literal. Results arrive a frame or two later, see DrainTimings.
void Mark( const char* name );
void DrainTimings( const std::function< void( const char* name, double ms ) >& fn );

}

// ---- command recording shims (see MetalShims.h): forward to Vulkan, or record into Metal
// when the command buffer is RTGL1::Metal::SectionCmd()

void RgMtl_vkCmdBindPipeline( VkCommandBuffer, VkPipelineBindPoint, VkPipeline );
void RgMtl_vkCmdBindDescriptorSets( VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t* );
void RgMtl_vkCmdPushConstants( VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void* );
void RgMtl_vkCmdDispatch( VkCommandBuffer, uint32_t, uint32_t, uint32_t );
void RgMtl_vkCmdPipelineBarrier( VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier* );
void RgMtl_svkCmdPipelineBarrier2KHR( VkCommandBuffer, const VkDependencyInfo* );
void RgMtl_vkCmdCopyBuffer( VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy* );
void RgMtl_vkCmdCopyBufferToImage( VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, uint32_t, const VkBufferImageCopy* );
void RgMtl_vkCmdBlitImage( VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageBlit*, VkFilter );
void RgMtl_vkCmdWriteTimestamp( VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t );
void RgMtl_vkCmdResetQueryPool( VkCommandBuffer, VkQueryPool, uint32_t, uint32_t );
void RgMtl_vkCmdBeginRenderPass( VkCommandBuffer, const VkRenderPassBeginInfo*, VkSubpassContents );
void RgMtl_vkCmdEndRenderPass( VkCommandBuffer );
void RgMtl_vkCmdBindVertexBuffers( VkCommandBuffer, uint32_t, uint32_t, const VkBuffer*, const VkDeviceSize* );
void RgMtl_vkCmdBindIndexBuffer( VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType );
void RgMtl_vkCmdSetViewport( VkCommandBuffer, uint32_t, uint32_t, const VkViewport* );
void RgMtl_vkCmdSetScissor( VkCommandBuffer, uint32_t, uint32_t, const VkRect2D* );
void RgMtl_vkCmdDraw( VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t );
void RgMtl_vkCmdDrawIndexed( VkCommandBuffer, uint32_t, uint32_t, uint32_t, int32_t, uint32_t );
void RgMtl_vkCmdDrawIndexedIndirectCount( VkCommandBuffer, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t );
void RgMtl_vkCmdClearAttachments( VkCommandBuffer, uint32_t, const VkClearAttachment*, uint32_t, const VkClearRect* );

namespace RTGL1
{
// some code calls the extension function pointers as RTGL1::svk...
using ::RgMtl_svkCmdPipelineBarrier2KHR;
}

#endif // __APPLE__
