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

#pragma once

#include "BlueNoise.h"
#include "GlobalUniform.h"
#include "ShaderManager.h"
#include "Framebuffers.h"
#include "Scene.h"
#include "CubemapManager.h"
#include "RenderCubemap.h"
#include "PortalList.h"
#include "RestirBuffers.h"
#include "Volumetric.h"

namespace RTGL1
{

// The path tracer's passes. Rays are cast with ray queries (GL_EXT_ray_query) from
// ordinary compute shaders - the model Metal uses natively - instead of a Vulkan ray
// tracing pipeline with a shader binding table. One compute pipeline per pass; all
// share one pipeline layout. Indices are the former SBT_INDEX_RAYGEN_* values.
class RayTracingPipeline : public IShaderDependency
{
public:
    RayTracingPipeline( VkDevice                           device,
                        std::shared_ptr< PhysicalDevice >  physDevice,
                        std::shared_ptr< MemoryAllocator > allocator,
                        const ShaderManager*               shaderManager,
                        Scene*                             scene,
                        const GlobalUniform*               uniform,
                        const TextureManager*              textureManager,
                        const Framebuffers*                framebuffers,
                        const RestirBuffers*               restirBuffers,
                        const BlueNoise*                   blueNoise,
                        const CubemapManager*              cubemapManager,
                        const RenderCubemap*               renderCubemap,
                        const PortalList*                  portalList,
                        const Volumetric*                  volumetric,
                        const RgInstanceCreateInfo&        rgInfo );
    ~RayTracingPipeline() override;

    RayTracingPipeline(const RayTracingPipeline& other) = delete;
    RayTracingPipeline(RayTracingPipeline&& other) noexcept = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline& other) = delete;
    RayTracingPipeline& operator=(RayTracingPipeline&& other) noexcept = delete;

    // pass index is one of SBT_INDEX_RAYGEN_*
    void BindPass( VkCommandBuffer cmd, uint32_t passIndex ) const;
    VkPipelineLayout GetLayout() const;
    const char*      GetPassName( uint32_t passIndex ) const { return passes[ passIndex ].pName; }
    VkPipeline       GetPipeline( uint32_t passIndex ) const { return pipelines[ passIndex ]; }
    // macOS: the passes can run natively in Metal (RTGL1_METAL_RT=1)
    bool             IsMetalReady() const { return metalReady; }

    // compute workgroup size of the passes (must match RaygenCommon.h)
    static constexpr uint32_t GROUP_SIZE_X = 8;
    static constexpr uint32_t GROUP_SIZE_Y = 8;

    void OnShaderReload(const ShaderManager *shaderManager) override;

private:
    void CreatePipelineLayout(const VkDescriptorSetLayout *pSetLayouts, uint32_t setLayoutCount);

    void CreatePipelines(const ShaderManager *shaderManager);
    void DestroyPipelines();

private:
    struct PassInfo
    {
        const char *pName;
        // One uint specialization const to use in the shader. Can be null.
        // Otherwise, must exist throughout class lifetime.
        uint32_t *pSpecConst;
    };

    static constexpr uint32_t PASS_COUNT = 8;

private:
    VkDevice device;

    PassInfo         passes[ PASS_COUNT ];
    VkPipeline       pipelines[ PASS_COUNT ];
    VkPipelineLayout pipelineLayout;

    uint32_t primaryRaysMaxAlbedoLayers;
    uint32_t indirectIlluminationMaxAlbedoLayers;

    bool metalReady = false;
};

}
