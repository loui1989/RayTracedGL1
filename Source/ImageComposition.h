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

#include "Common.h"

#include <cfloat>
#include "ShaderManager.h"
#include "Framebuffers.h"
#include "GlobalUniform.h"
#include "Tonemapping.h"
#include "Volumetric.h"

namespace RTGL1
{

class ImageComposition : public IShaderDependency
{
public:
    ImageComposition( VkDevice                           device,
                      std::shared_ptr< MemoryAllocator > allocator,
                      std::shared_ptr< Framebuffers >    framebuffers,
                      const ShaderManager&               shaderManager,
                      const GlobalUniform&               uniform,
                      const Tonemapping&                 tonemapping );
    ~ImageComposition() override;

    ImageComposition( const ImageComposition& other )                = delete;
    ImageComposition( ImageComposition&& other ) noexcept            = delete;
    ImageComposition& operator=( const ImageComposition& other )     = delete;
    ImageComposition& operator=( ImageComposition&& other ) noexcept = delete;

    void PrepareForRaster( VkCommandBuffer cmd, uint32_t frameIndex, const GlobalUniform* uniform );

    void Finalize( VkCommandBuffer                     cmd,
                   uint32_t                            frameIndex,
                   const GlobalUniform&                uniform,
                   const Tonemapping&                  tonemapping,
                   const RgDrawFrameTonemappingParams& params );

    // Doom64-RT: apply the participating medium AFTER the upscaler, in place on
    // whichever image the post chain is currently holding. See
    // CmVolumeCompose.comp -- the outline behind volumetrics is the upscaler
    // reconstructing surface and medium added together, so the fix is to stop
    // handing it the sum.
    void ComposeVolume( VkCommandBuffer       cmd,
                        uint32_t              frameIndex,
                        const GlobalUniform&  uniform,
                        const Tonemapping&    tonemapping,
                        FramebufferImageIndex target,
                        uint32_t              width,
                        uint32_t              height );

    // Doom64-RT: apply auto exposure and the screen emissive AFTER DLSS-RR, in
    // place on RR's output image. Only dispatched when the frame's rrPreExposure
    // uniform is set (CmPrepareFinal skipped both under that flag). See
    // CmRrPostExposure.comp for why RR must denoise pre-exposure radiance.
    void RrPostExposure( VkCommandBuffer      cmd,
                         uint32_t             frameIndex,
                         const GlobalUniform& uniform,
                         const Tonemapping&   tonemapping,
                         uint32_t             width,
                         uint32_t             height );

    [[nodiscard]] auto SetupLpmParams( VkCommandBuffer                     cmd,
                                       uint32_t                            frameIndex,
                                       const RgDrawFrameTonemappingParams& params,
                                       bool hdr )
        -> VkDescriptorSet;
    [[nodiscard]] auto GetLpmDescSetLayout() -> VkDescriptorSetLayout;

    void OnShaderReload( const ShaderManager* shaderManager ) override;

private:
    void ProcessCheckerboard( VkCommandBuffer      cmd,
                              uint32_t             frameIndex,
                              const GlobalUniform* uniform );

    static VkPipelineLayout CreatePipelineLayout( VkDevice               device,
                                                  VkDescriptorSetLayout* pSetLayouts,
                                                  uint32_t               setLayoutCount,
                                                  const char*            pDebugName,
                                                  uint32_t               pushSize = 0 );

    void CreateDescriptors();

    void CreatePipelines( const ShaderManager* shaderManager );
    void DestroyPipelines();

private:
    VkDevice device;

    std::shared_ptr< Framebuffers > framebuffers;

    std::unique_ptr< AutoBuffer > lpmParams;
    struct
    {
        RgFloat3D saturation{ FLT_MAX, FLT_MAX, FLT_MAX };
        RgFloat3D crosstalk{ FLT_MAX, FLT_MAX, FLT_MAX };
        RgFloat3D hdrSaturation{ FLT_MAX, FLT_MAX, FLT_MAX };
        float     contrast{ FLT_MAX };
        float     hdrContrast{ FLT_MAX };
    } lpmPrev;

    VkPipelineLayout composePipelineLayout;
    VkPipelineLayout checkerboardPipelineLayout;
    VkPipelineLayout volumeComposePipelineLayout;
    VkPipelineLayout rrPostExposurePipelineLayout;

    VkPipeline composePipeline;
    VkPipeline checkerboardPipeline;
    VkPipeline volumeComposePipeline;
    VkPipeline rrPostExposurePipeline;

    VkDescriptorSetLayout descLayout;
    VkDescriptorPool      descPool;
    VkDescriptorSet       descSet;
};

}