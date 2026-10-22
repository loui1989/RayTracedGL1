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

#include "RayTracingPipeline.h"

#include "Generated/ShaderCommonC.h"
#include "Utils.h"

using namespace RTGL1;

RayTracingPipeline::RayTracingPipeline( VkDevice                           _device,
                                        std::shared_ptr< PhysicalDevice >  _physDevice,
                                        std::shared_ptr< MemoryAllocator > _allocator,
                                        const ShaderManager*               _shaderManager,
                                        Scene*                             _scene,
                                        const GlobalUniform*               _uniform,
                                        const TextureManager*              _textureManager,
                                        const Framebuffers*                _framebuffers,
                                        const RestirBuffers*               _restirBuffers,
                                        const BlueNoise*                   _blueNoise,
                                        const CubemapManager*              _cubemapManager,
                                        const RenderCubemap*               _renderCubemap,
                                        const PortalList*                  _portalList,
                                        const Volumetric*                  _volumetric,
                                        const RgInstanceCreateInfo&        _rgInfo )
    : device( _device )
    , passes{}
    , pipelines{}
    , pipelineLayout( VK_NULL_HANDLE )
    , primaryRaysMaxAlbedoLayers( _rgInfo.primaryRaysMaxAlbedoLayers )
    , indirectIlluminationMaxAlbedoLayers( _rgInfo.indirectIlluminationMaxAlbedoLayers )
{
    // all set layouts to be used
    VkDescriptorSetLayout setLayouts[] = {
        // ray tracing acceleration structures
        _scene->GetASManager()->GetTLASDescSetLayout(),
        // storage images
        _framebuffers->GetDescSetLayout(),
        // uniform
        _uniform->GetDescSetLayout(),
        // vertex data
        _scene->GetASManager()->GetBuffersDescSetLayout(),
        // textures
        _textureManager->GetDescSetLayout(),
        // uniform random
        _blueNoise->GetDescSetLayout(),
        // light sources
        _scene->GetLightManager()->GetDescSetLayout(),
        // cubemaps, for a cubemap type of skyboxes
        _cubemapManager->GetDescSetLayout(),
        // dynamic cubemaps
        _renderCubemap->GetDescSetLayout(),
        // portals
        _portalList->GetDescSetLayout(),
        // device local buffers for restir
        _restirBuffers->GetDescSetLayout(),
        // device local buffers for volumetrics
        _volumetric->GetDescSetLayout(),
    };

    CreatePipelineLayout(setLayouts, std::size(setLayouts));

    assert(primaryRaysMaxAlbedoLayers <= MATERIALS_MAX_LAYER_COUNT);
    assert(indirectIlluminationMaxAlbedoLayers <= MATERIALS_MAX_LAYER_COUNT);

    // indexed by the former shader binding table indices
    passes[ SBT_INDEX_RAYGEN_PRIMARY ]            = { "RGenPrimary",        &primaryRaysMaxAlbedoLayers };
    passes[ SBT_INDEX_RAYGEN_REFL_REFR ]          = { "RGenReflRefr",       &primaryRaysMaxAlbedoLayers };
    passes[ SBT_INDEX_RAYGEN_DIRECT ]             = { "RGenDirect",         nullptr };
    passes[ SBT_INDEX_RAYGEN_INDIRECT_INIT ]      = { "RGenIndirectInit",   &indirectIlluminationMaxAlbedoLayers };
    passes[ SBT_INDEX_RAYGEN_INDIRECT_FINAL ]     = { "RGenIndirectFinal",  &indirectIlluminationMaxAlbedoLayers };
    passes[ SBT_INDEX_RAYGEN_GRADIENTS ]          = { "RGenGradients",      nullptr };
    passes[ SBT_INDEX_RAYGEN_INITIAL_RESERVOIRS ] = { "RInitialReservoirs", nullptr };
    passes[ SBT_INDEX_RAYGEN_VOLUMETRIC ]         = { "RVolumetric",        nullptr };
    static_assert( SBT_INDEX_RAYGEN_VOLUMETRIC < PASS_COUNT );

    CreatePipelines( _shaderManager );
}

RayTracingPipeline::~RayTracingPipeline()
{
    DestroyPipelines();
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

void RayTracingPipeline::CreatePipelineLayout(const VkDescriptorSetLayout *pSetLayouts, uint32_t setLayoutCount)
{
    VkPipelineLayoutCreateInfo plLayoutInfo = {};
    plLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plLayoutInfo.setLayoutCount = setLayoutCount;
    plLayoutInfo.pSetLayouts = pSetLayouts;

    VkResult r = vkCreatePipelineLayout(device, &plLayoutInfo, nullptr, &pipelineLayout);
    VK_CHECKERROR(r);

    SET_DEBUG_NAME(device, pipelineLayout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Path tracer pipeline layout");
}

void RayTracingPipeline::CreatePipelines(const ShaderManager *shaderManager)
{
    for( uint32_t i = 0; i < PASS_COUNT; i++ )
    {
        VkSpecializationMapEntry specEntry = {};
        VkSpecializationInfo     specInfo  = {};

        VkComputePipelineCreateInfo info = {};
        info.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage  = shaderManager->GetStageInfo( passes[ i ].pName );
        info.layout = pipelineLayout;
        assert( info.stage.stage == VK_SHADER_STAGE_COMPUTE_BIT );

        if( passes[ i ].pSpecConst != nullptr )
        {
            specEntry.constantID = 0;
            specEntry.offset     = 0;
            specEntry.size       = sizeof( uint32_t );

            specInfo.mapEntryCount = 1;
            specInfo.pMapEntries   = &specEntry;
            specInfo.dataSize      = sizeof( *passes[ i ].pSpecConst );
            specInfo.pData         = passes[ i ].pSpecConst;

            info.stage.pSpecializationInfo = &specInfo;
        }

        VkResult r = vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &info, nullptr, &pipelines[ i ] );
        VK_CHECKERROR( r );

        SET_DEBUG_NAME( device, pipelines[ i ], VK_OBJECT_TYPE_PIPELINE, passes[ i ].pName );
    }

#ifdef __APPLE__
    // native Metal path tracing: all passes must translate and compile, otherwise it stays off
    metalReady = false;
    if( Metal::IsEnabled() )
    {
        bool ok = true;
        for( uint32_t i = 0; i < PASS_COUNT; i++ )
        {
            ok = Metal::PreparePipeline( pipelines[ i ], passes[ i ].pName ) && ok;
        }
        metalReady = ok;
        if( !ok )
        {
            Metal::Disable( "a path tracing pass couldn't be prepared for Metal" );
        }
    }
#endif
}

void RayTracingPipeline::DestroyPipelines()
{
    for( VkPipeline& p : pipelines )
    {
        vkDestroyPipeline( device, p, nullptr );
        p = VK_NULL_HANDLE;
    }
}

void RayTracingPipeline::BindPass( VkCommandBuffer cmd, uint32_t passIndex ) const
{
    assert( passIndex < PASS_COUNT );
    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[ passIndex ] );
}

VkPipelineLayout RayTracingPipeline::GetLayout() const
{
    return pipelineLayout;
}

void RayTracingPipeline::OnShaderReload(const ShaderManager *shaderManager)
{
    DestroyPipelines();
    CreatePipelines(shaderManager);
}
