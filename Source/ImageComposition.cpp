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

#include "ImageComposition.h"

#include "Generated/ShaderCommonC.h"
#include "CmdLabel.h"
#include "Utils.h"

namespace
{
constexpr auto LPM_BUFFER_SIZE = sizeof( uint32_t ) * 24 * 4;
}

RTGL1::ImageComposition::ImageComposition( VkDevice                           _device,
                                           std::shared_ptr< MemoryAllocator > _allocator,
                                           std::shared_ptr< Framebuffers >    _framebuffers,
                                           const ShaderManager&               _shaderManager,
                                           const GlobalUniform&               _uniform,
                                           const Tonemapping&                 _tonemapping )
    : device( _device )
    , framebuffers( std::move( _framebuffers ) )
    , composePipelineLayout( VK_NULL_HANDLE )
    , checkerboardPipelineLayout( VK_NULL_HANDLE )
    , volumeComposePipelineLayout( VK_NULL_HANDLE )
    , composePipeline( VK_NULL_HANDLE )
    , checkerboardPipeline( VK_NULL_HANDLE )
    , volumeComposePipeline( VK_NULL_HANDLE )
    , descLayout( VK_NULL_HANDLE )
    , descPool( VK_NULL_HANDLE )
    , descSet( VK_NULL_HANDLE )
{
    lpmParams = std::make_unique< AutoBuffer >( std::move( _allocator ) );
    lpmParams->Create( LPM_BUFFER_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "LPM Params" );

    CreateDescriptors();

    {
        VkDescriptorSetLayout setLayouts[] = {
            framebuffers->GetDescSetLayout(),
            _uniform.GetDescSetLayout(),
            _tonemapping.GetDescSetLayout(),
            descLayout,
        };

        composePipelineLayout = CreatePipelineLayout(
            device, setLayouts, std::size( setLayouts ), "Composition pipeline layout" );
    }
    {
        VkDescriptorSetLayout setLayouts[] = {
            framebuffers->GetDescSetLayout(),
            _uniform.GetDescSetLayout(),
        };

        checkerboardPipelineLayout = CreatePipelineLayout(
            device, setLayouts, std::size( setLayouts ), "Checkerboard pipeline layout" );
    }
    {
        // Doom64-RT: the post-upscale volume composite. Same three sets as the
        // main composition -- it needs the framebuffers, the uniform and the
        // TONEMAPPING set, because the in-scatter has to be exposed with the
        // same ev100 the surface already was.
        VkDescriptorSetLayout setLayouts[] = {
            framebuffers->GetDescSetLayout(),
            _uniform.GetDescSetLayout(),
            _tonemapping.GetDescSetLayout(),
        };

        volumeComposePipelineLayout = CreatePipelineLayout( device,
                                                           setLayouts,
                                                           std::size( setLayouts ),
                                                           "Volume compose pipeline layout",
                                                           sizeof( uint32_t ) * 4 );
    }
    {
        // Doom64-RT: exposure + screen emissive after DLSS-RR. Same three sets
        // as the volume compose -- framebuffers, uniform, and TONEMAPPING for
        // the very ev100 CmPrepareFinal would have applied.
        VkDescriptorSetLayout setLayouts[] = {
            framebuffers->GetDescSetLayout(),
            _uniform.GetDescSetLayout(),
            _tonemapping.GetDescSetLayout(),
        };

        rrPostExposurePipelineLayout = CreatePipelineLayout( device,
                                                             setLayouts,
                                                             std::size( setLayouts ),
                                                             "RR post-exposure pipeline layout",
                                                             sizeof( uint32_t ) * 4 );
    }

    CreatePipelines( &_shaderManager );
}

RTGL1::ImageComposition::~ImageComposition()
{
    vkDestroyDescriptorPool( device, descPool, nullptr );
    vkDestroyDescriptorSetLayout( device, descLayout, nullptr );
    vkDestroyPipelineLayout( device, composePipelineLayout, nullptr );
    vkDestroyPipelineLayout( device, checkerboardPipelineLayout, nullptr );
    vkDestroyPipelineLayout( device, volumeComposePipelineLayout, nullptr );
    vkDestroyPipelineLayout( device, rrPostExposurePipelineLayout, nullptr );
    DestroyPipelines();
}

void RTGL1::ImageComposition::PrepareForRaster( VkCommandBuffer      cmd,
                                                uint32_t             frameIndex,
                                                const GlobalUniform* uniform )
{
    ProcessCheckerboard( cmd, frameIndex, uniform );
}

void RTGL1::ImageComposition::OnShaderReload( const ShaderManager* shaderManager )
{
    DestroyPipelines();
    CreatePipelines( shaderManager );
}

void RTGL1::ImageComposition::Finalize( VkCommandBuffer                     cmd,
                                        uint32_t                            frameIndex,
                                        const GlobalUniform&                uniform,
                                        const Tonemapping&                  tonemapping,
                                        const RgDrawFrameTonemappingParams& params )
{
    using FI = FramebufferImageIndex;
    auto label = CmdLabel{ cmd, "Prefinal framebuf compose" };


    // sync access
    FI fs[] = {
        FI::FB_IMAGE_INDEX_SCREEN_EMISSION, FI::FB_IMAGE_INDEX_FINAL,
        FI::FB_IMAGE_INDEX_DEPTH_WORLD,
        FI::FB_IMAGE_INDEX_SCATTERING,
        FI::FB_IMAGE_INDEX_RR_DEMOD_FACTOR,
    };
    framebuffers->BarrierMultiple( cmd, frameIndex, fs );


    // bind pipeline
    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, composePipeline );


    // bind desc sets
    VkDescriptorSet sets[] = {
        framebuffers->GetDescSet( frameIndex ),
        uniform.GetDescSet( frameIndex ),
        tonemapping.GetDescSet(),
        descSet,
    };

    vkCmdBindDescriptorSets( cmd,
                             VK_PIPELINE_BIND_POINT_COMPUTE,
                             composePipelineLayout,
                             0,
                             std::size( sets ),
                             sets,
                             0,
                             nullptr );


    vkCmdDispatch(
        cmd,
        Utils::GetWorkGroupCount( uniform.GetData()->renderWidth, COMPUTE_COMPOSE_GROUP_SIZE_X ),
        Utils::GetWorkGroupCount( uniform.GetData()->renderHeight, COMPUTE_COMPOSE_GROUP_SIZE_Y ),
        1 );
}

void RTGL1::ImageComposition::ComposeVolume( VkCommandBuffer       cmd,
                                             uint32_t              frameIndex,
                                             const GlobalUniform&  uniform,
                                             const Tonemapping&    tonemapping,
                                             FramebufferImageIndex target,
                                             uint32_t              width,
                                             uint32_t              height )
{
    using FI = FramebufferImageIndex;
    CmdLabel label( cmd, "Volume compose (post-upscale)" );

    // The target is whatever the post chain is holding. Anything else means the
    // caller changed the chain without telling this pass, and the shader would
    // silently write to the wrong image -- so it is an assert, not a fallback.
    uint32_t targetIdx;
    switch( target )
    {
        case FI::FB_IMAGE_INDEX_FINAL: targetIdx = 0; break;
        case FI::FB_IMAGE_INDEX_UPSCALED_PING: targetIdx = 1; break;
        case FI::FB_IMAGE_INDEX_UPSCALED_PONG: targetIdx = 2; break;
        default: assert( 0 ); return;
    }

    FI fs[] = { target, FI::FB_IMAGE_INDEX_SCATTERING };
    framebuffers->BarrierMultiple( cmd, frameIndex, fs );

    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volumeComposePipeline );

    VkDescriptorSet sets[] = {
        framebuffers->GetDescSet( frameIndex ),
        uniform.GetDescSet( frameIndex ),
        tonemapping.GetDescSet(),
    };

    vkCmdBindDescriptorSets( cmd,
                             VK_PIPELINE_BIND_POINT_COMPUTE,
                             volumeComposePipelineLayout,
                             0,
                             std::size( sets ),
                             sets,
                             0,
                             nullptr );

    const uint32_t push[ 4 ] = { targetIdx, width, height, 0 };
    vkCmdPushConstants( cmd,
                        volumeComposePipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        sizeof( push ),
                        push );

    // 16x16, matching the shader's local_size. Dispatched at the TARGET's size,
    // which is the upscaled size when an upscaler ran and the render size when
    // none did -- the caller knows which, this pass does not guess.
    vkCmdDispatch(
        cmd, Utils::GetWorkGroupCount( width, 16 ), Utils::GetWorkGroupCount( height, 16 ), 1 );
}

void RTGL1::ImageComposition::RrPostExposure( VkCommandBuffer      cmd,
                                              uint32_t             frameIndex,
                                              const GlobalUniform& uniform,
                                              const Tonemapping&   tonemapping,
                                              uint32_t             width,
                                              uint32_t             height )
{
    using FI = FramebufferImageIndex;
    CmdLabel label( cmd, "RR post-exposure (exposure + screen emissive)" );

    // RR's output image is fixed (DLSSRR.cpp OUTPUT_IMAGE), so unlike
    // ComposeVolume there is no target selection: this pass only ever runs
    // immediately after the RR branch, on UPSCALED_PONG, in place.
    FI fs[] = {
        FI::FB_IMAGE_INDEX_UPSCALED_PONG,
        FI::FB_IMAGE_INDEX_SCATTERING,
        FI::FB_IMAGE_INDEX_SCREEN_EMISSION,
        FI::FB_IMAGE_INDEX_RR_DISOCCLUSION,
        FI::FB_IMAGE_INDEX_RR_DEMOD_FACTOR,
    };
    framebuffers->BarrierMultiple( cmd, frameIndex, fs );

    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rrPostExposurePipeline );

    VkDescriptorSet sets[] = {
        framebuffers->GetDescSet( frameIndex ),
        uniform.GetDescSet( frameIndex ),
        tonemapping.GetDescSet(),
    };

    vkCmdBindDescriptorSets( cmd,
                             VK_PIPELINE_BIND_POINT_COMPUTE,
                             rrPostExposurePipelineLayout,
                             0,
                             std::size( sets ),
                             sets,
                             0,
                             nullptr );

    const uint32_t push[ 4 ] = { width, height, 0, 0 };
    vkCmdPushConstants( cmd,
                        rrPostExposurePipelineLayout,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        0,
                        sizeof( push ),
                        push );

    // 16x16, matching the shader's local_size; always the UPSCALED size --
    // RR ran, so the chain is at output resolution by definition.
    vkCmdDispatch(
        cmd, Utils::GetWorkGroupCount( width, 16 ), Utils::GetWorkGroupCount( height, 16 ), 1 );
}

void RTGL1::ImageComposition::ProcessCheckerboard( VkCommandBuffer      cmd,
                                                   uint32_t             frameIndex,
                                                   const GlobalUniform* uniform )
{
    using FI = FramebufferImageIndex;
    CmdLabel label( cmd, "Framebuf checkerboard interpolate" );


    // sync access
    FI fs[] = {
        FI::FB_IMAGE_INDEX_PRE_FINAL,
        FI::FB_IMAGE_INDEX_SCREEN_EMIS_R_T,
    };
    framebuffers->BarrierMultiple( cmd, frameIndex, fs );


    // bind pipeline
    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, checkerboardPipeline );


    // bind desc sets
    VkDescriptorSet sets[] = {
        framebuffers->GetDescSet( frameIndex ),
        uniform->GetDescSet( frameIndex ),
    };

    vkCmdBindDescriptorSets( cmd,
                             VK_PIPELINE_BIND_POINT_COMPUTE,
                             checkerboardPipelineLayout,
                             0,
                             std::size( sets ),
                             sets,
                             0,
                             nullptr );


    // start compute shader
    uint32_t wgCountX =
        Utils::GetWorkGroupCount( uniform->GetData()->renderWidth, COMPUTE_COMPOSE_GROUP_SIZE_X );
    uint32_t wgCountY =
        Utils::GetWorkGroupCount( uniform->GetData()->renderHeight, COMPUTE_COMPOSE_GROUP_SIZE_Y );

    vkCmdDispatch( cmd, wgCountX, wgCountY, 1 );
}

VkPipelineLayout RTGL1::ImageComposition::CreatePipelineLayout( VkDevice               device,
                                                                VkDescriptorSetLayout* pSetLayouts,
                                                                uint32_t    setLayoutCount,
                                                                const char* pDebugName,
                                                                uint32_t    pushSize )
{
    const VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = pushSize,
    };

    VkPipelineLayoutCreateInfo info = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = setLayoutCount,
        .pSetLayouts            = pSetLayouts,
        .pushConstantRangeCount = pushSize > 0 ? 1u : 0u,
        .pPushConstantRanges    = pushSize > 0 ? &push : nullptr,
    };

    VkPipelineLayout layout;
    VkResult         r = vkCreatePipelineLayout( device, &info, nullptr, &layout );
    VK_CHECKERROR( r );

    SET_DEBUG_NAME( device, layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, pDebugName );

    return layout;
}

void RTGL1::ImageComposition::CreateDescriptors()
{
    VkResult r;
    {
        VkDescriptorSetLayoutBinding binding = {
            .binding         = BINDING_LPM_PARAMS,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo = {
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &binding,
        };

        r = vkCreateDescriptorSetLayout( device, &layoutInfo, nullptr, &descLayout );
        VK_CHECKERROR( r );

        SET_DEBUG_NAME(
            device, descLayout, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "LPM Desc set layout" );
    }
    {
        VkDescriptorPoolSize poolSize = {
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
        };

        VkDescriptorPoolCreateInfo poolInfo = {
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets       = 1,
            .poolSizeCount = 1,
            .pPoolSizes    = &poolSize,
        };

        r = vkCreateDescriptorPool( device, &poolInfo, nullptr, &descPool );
        VK_CHECKERROR( r );

        SET_DEBUG_NAME( device, descPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "LPM Desc pool" );
    }
    {
        VkDescriptorSetAllocateInfo allocInfo = {
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = descPool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &descLayout,
        };

        r = vkAllocateDescriptorSets( device, &allocInfo, &descSet );
        VK_CHECKERROR( r );

        SET_DEBUG_NAME( device, descSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, "LPM Desc set" );
    }
    {
        VkDescriptorBufferInfo bfInfo = {
            .buffer = lpmParams->GetDeviceLocal(),
            .offset = 0,
            .range  = VK_WHOLE_SIZE,
        };

        VkWriteDescriptorSet wrt = {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = descSet,
            .dstBinding      = BINDING_LUM_HISTOGRAM,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &bfInfo,
        };

        vkUpdateDescriptorSets( device, 1, &wrt, 0, nullptr );
    }
}

void RTGL1::ImageComposition::CreatePipelines( const ShaderManager* shaderManager )
{
    {
        VkComputePipelineCreateInfo plInfo = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = shaderManager->GetStageInfo( "CPrepareFinal" ),
            .layout = composePipelineLayout,
        };

        VkResult r = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &plInfo, nullptr, &composePipeline );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME( device, composePipeline, VK_OBJECT_TYPE_PIPELINE, "Composition pipeline" );
    }
    {
        VkComputePipelineCreateInfo plInfo = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = shaderManager->GetStageInfo( "CCheckerboard" ),
            .layout = checkerboardPipelineLayout,
        };

        VkResult r = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &plInfo, nullptr, &checkerboardPipeline );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, checkerboardPipeline, VK_OBJECT_TYPE_PIPELINE, "Checkerboard pipeline" );
    }
    {
        VkComputePipelineCreateInfo plInfo = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = shaderManager->GetStageInfo( "CVolumeCompose" ),
            .layout = volumeComposePipelineLayout,
        };

        VkResult r = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &plInfo, nullptr, &volumeComposePipeline );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, volumeComposePipeline, VK_OBJECT_TYPE_PIPELINE, "Volume compose pipeline" );
    }
    {
        VkComputePipelineCreateInfo plInfo = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = shaderManager->GetStageInfo( "CRrPostExposure" ),
            .layout = rrPostExposurePipelineLayout,
        };

        VkResult r = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &plInfo, nullptr, &rrPostExposurePipeline );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, rrPostExposurePipeline, VK_OBJECT_TYPE_PIPELINE, "RR post-exposure pipeline" );
    }
}

void RTGL1::ImageComposition::DestroyPipelines()
{
    vkDestroyPipeline( device, composePipeline, nullptr );
    composePipeline = VK_NULL_HANDLE;

    vkDestroyPipeline( device, checkerboardPipeline, nullptr );
    checkerboardPipeline = VK_NULL_HANDLE;

    vkDestroyPipeline( device, volumeComposePipeline, nullptr );
    volumeComposePipeline = VK_NULL_HANDLE;

    vkDestroyPipeline( device, rrPostExposurePipeline, nullptr );
    rrPostExposurePipeline = VK_NULL_HANDLE;
}

#include "LpmParams.h"

auto RTGL1::ImageComposition::SetupLpmParams( VkCommandBuffer                     cmd,
                                              uint32_t                            frameIndex,
                                              const RgDrawFrameTonemappingParams& params,
                                              bool hdr )
    -> VkDescriptorSet
{
    // if no need to update device memory
    if( Utils::AreAlmostSame( lpmPrev.saturation, params.saturation ) &&
        Utils::AreAlmostSame( lpmPrev.crosstalk, params.crosstalk ) && 
        Utils::AreAlmostSameF( lpmPrev.contrast, params.contrast )  &&
        Utils::AreAlmostSame( lpmPrev.hdrSaturation, params.hdrSaturation ) &&
        Utils::AreAlmostSameF( lpmPrev.hdrContrast, params.hdrContrast ) )
    {
        return descSet;
    }

    void* pContext = lpmParams->GetMapped( frameIndex );
    MakeLpmParams( pContext, params, hdr );

    lpmParams->CopyFromStaging( cmd, frameIndex, LPM_BUFFER_SIZE );

    VkBufferMemoryBarrier2KHR b = {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
        .srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT,
        .buffer        = lpmParams->GetDeviceLocal(),
        .offset        = 0,
        .size          = LPM_BUFFER_SIZE,
    };

    VkDependencyInfoKHR info = {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &b,
    };

    svkCmdPipelineBarrier2KHR( cmd, &info );

    // cache
    lpmPrev.saturation = params.saturation;
    lpmPrev.crosstalk  = params.crosstalk;

    return descSet;
}

auto RTGL1::ImageComposition::GetLpmDescSetLayout() -> VkDescriptorSetLayout
{
    return descLayout;
}
