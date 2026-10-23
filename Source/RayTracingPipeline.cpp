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

#include <cstddef>
#include <cstring>

namespace RTGL1
{
namespace
{

    template< uint32_t Count >
    VkPipelineLayout CreatePipelineLayout( VkDevice device,
                                           const VkDescriptorSetLayout ( &setLayouts )[ Count ] )
    {
        VkPipelineLayoutCreateInfo info = {
            .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = Count,
            .pSetLayouts    = setLayouts,
        };

        VkPipelineLayout layout;
        VkResult         r = vkCreatePipelineLayout( device, &info, nullptr, &layout );

        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, layout, VK_OBJECT_TYPE_PIPELINE_LAYOUT, "Ray tracing pipeline Layout" );
        return layout;
    }

}
}


RTGL1::RayTracingPipeline::RayTracingPipeline( VkDevice                           _device,
                                               std::shared_ptr< PhysicalDevice >  _physDevice,
                                               std::shared_ptr< MemoryAllocator > _allocator,
                                               const ShaderManager&               _shaderManager,
                                               Scene&                             _scene,
                                               const GlobalUniform&               _uniform,
                                               const TextureManager&              _textureManager,
                                               const Framebuffers&                _framebuffers,
                                               const RestirBuffers&               _restirBuffers,
                                               const BlueNoise&                   _blueNoise,
                                               const LightManager&                _lightManager,
                                               const CubemapManager&              _cubemapManager,
                                               const RenderCubemap&               _renderCubemap,
                                               const PortalList&                  _portalList,
                                               const Volumetric&                  _volumetric,
                                               const RgInstanceCreateInfo&        _rgInfo )
    : device( _device ), physDevice( std::move( _physDevice ) )
{
    // all set layouts to be used
    VkDescriptorSetLayout setLayouts[] = {
        // ray tracing acceleration structures
        _scene.GetASManager()->GetTLASDescSetLayout(),
        // storage images
        _framebuffers.GetDescSetLayout(),
        // uniform
        _uniform.GetDescSetLayout(),
        // vertex data
        _scene.GetASManager()->GetBuffersDescSetLayout(),
        // textures
        _textureManager.GetDescSetLayout(),
        // uniform random
        _blueNoise.GetDescSetLayout(),
        // light sources
        _lightManager.GetDescSetLayout(),
        // cubemaps, for a cubemap type of skyboxes
        _cubemapManager.GetDescSetLayout(),
        // dynamic cubemaps
        _renderCubemap.GetDescSetLayout(),
        // portals
        _portalList.GetDescSetLayout(),
        // device local buffers for restir
        _restirBuffers.GetDescSetLayout(),
        // device local buffers for volumetrics
        _volumetric.GetDescSetLayout(),
    };

    rtPipelineLayout = CreatePipelineLayout( device, setLayouts );

    assert( _rgInfo.primaryRaysMaxAlbedoLayers <= MATERIALS_MAX_LAYER_COUNT );
    assert( _rgInfo.indirectIlluminationMaxAlbedoLayers <= MATERIALS_MAX_LAYER_COUNT );
    assert( _rgInfo.lightmapTexCoordLayerIndex < MATERIALS_MAX_LAYER_COUNT );

    const uint32_t lightmapLayerIndex =
        ( _rgInfo.lightmapTexCoordLayerIndex > 0 &&
          _rgInfo.lightmapTexCoordLayerIndex < MATERIALS_MAX_LAYER_COUNT )
            ? _rgInfo.lightmapTexCoordLayerIndex
            : 0xFF;

    // clang-format off
    // indexed by the former shader binding table indices
    passes[ SBT_INDEX_RAYGEN_PRIMARY ]            = { "RGenPrimary",        SpecConst{ _rgInfo.primaryRaysMaxAlbedoLayers, lightmapLayerIndex } };
    passes[ SBT_INDEX_RAYGEN_REFL_REFR ]          = { "RGenReflRefr",       SpecConst{ _rgInfo.primaryRaysMaxAlbedoLayers, lightmapLayerIndex } };
    passes[ SBT_INDEX_RAYGEN_DIRECT ]             = { "RGenDirect",         {} };
    passes[ SBT_INDEX_RAYGEN_INDIRECT_INIT ]      = { "RGenIndirectInit",   SpecConst{ _rgInfo.indirectIlluminationMaxAlbedoLayers, lightmapLayerIndex } };
    passes[ SBT_INDEX_RAYGEN_GRADIENTS ]          = { "RGenGradients",      {} };
    passes[ SBT_INDEX_RAYGEN_INITIAL_RESERVOIRS ] = { "RInitialReservoirs", {} };
    passes[ SBT_INDEX_RAYGEN_VOLUMETRIC ]         = { "RVolumetric",        {} };
    // clang-format on

    CreatePipelines( &_shaderManager );
}

RTGL1::RayTracingPipeline::~RayTracingPipeline()
{
    DestroyPipelines();
    vkDestroyPipelineLayout( device, rtPipelineLayout, nullptr );
}

void RTGL1::RayTracingPipeline::CreatePipelines( const ShaderManager* shaderManager )
{
    constexpr VkSpecializationMapEntry specEntries[ SpecConst::MemberCount ] = {
        {
            .constantID = 0,
            .offset     = offsetof( SpecConst, maxAlbedoLayers ),
            .size       = sizeof( SpecConst::maxAlbedoLayers ),
        },
        {
            .constantID = 1,
            .offset     = offsetof( SpecConst, lightmapLayerIndex ),
            .size       = sizeof( SpecConst::lightmapLayerIndex ),
        },
    };
    static_assert( SpecConst::MemberCount == 2 );

    for( uint32_t i = 0; i < PASS_COUNT; i++ )
    {
        assert( passes[ i ].name );

        VkComputePipelineCreateInfo info = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = shaderManager->GetStageInfo( passes[ i ].name ),
            .layout = rtPipelineLayout,
        };
        assert( info.stage.stage == VK_SHADER_STAGE_COMPUTE_BIT );

        VkSpecializationInfo specInfo = {};
        if( passes[ i ].specConst )
        {
            specInfo = {
                .mapEntryCount = std::size( specEntries ),
                .pMapEntries   = specEntries,
                .dataSize      = sizeof( SpecConst ),
                .pData         = &( passes[ i ].specConst.value() ),
            };
            info.stage.pSpecializationInfo = &specInfo;
        }

        VkResult r = vkCreateComputePipelines( device, VK_NULL_HANDLE, 1, &info, nullptr, &pipelines[ i ] );
        VK_CHECKERROR( r );
        SET_DEBUG_NAME( device, pipelines[ i ], VK_OBJECT_TYPE_PIPELINE, passes[ i ].name );
    }

    {
        VkComputePipelineCreateInfo info = {
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext  = nullptr,
            .flags  = 0,
            .stage  = shaderManager->GetStageInfo( "CmIndirectFinal" ),
            .layout = rtPipelineLayout,
        };

        VkResult r = vkCreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &info, nullptr, &compPipelineIndirectFinal );
        VK_CHECKERROR( r );
        SET_DEBUG_NAME(
            device, compPipelineIndirectFinal, VK_OBJECT_TYPE_PIPELINE, "CmIndirectFinal" );
    }
}

void RTGL1::RayTracingPipeline::DestroyPipelines()
{
    for( VkPipeline& p : pipelines )
    {
        vkDestroyPipeline( device, p, nullptr );
        p = VK_NULL_HANDLE;
    }
    vkDestroyPipeline( device, compPipelineIndirectFinal, nullptr );
    compPipelineIndirectFinal = VK_NULL_HANDLE;
}

void RTGL1::RayTracingPipeline::BindPass( VkCommandBuffer cmd, uint32_t passIndex ) const
{
    assert( passIndex < PASS_COUNT );
    vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[ passIndex ] );
}

auto RTGL1::RayTracingPipeline::GetPipeline( uint32_t passIndex ) const -> VkPipeline
{
    assert( passIndex < PASS_COUNT );
    return pipelines[ passIndex ];
}

auto RTGL1::RayTracingPipeline::GetPassName( uint32_t passIndex ) const -> const char*
{
    assert( passIndex < PASS_COUNT );
    return passes[ passIndex ].name;
}

auto RTGL1::RayTracingPipeline::GetPipelineIndirectFinal_Compute() -> VkPipeline
{
    return compPipelineIndirectFinal;
}

VkPipelineLayout RTGL1::RayTracingPipeline::GetLayout() const
{
    return rtPipelineLayout;
}

void RTGL1::RayTracingPipeline::OnShaderReload( const ShaderManager* shaderManager )
{
    DestroyPipelines();
    CreatePipelines( shaderManager );
}
