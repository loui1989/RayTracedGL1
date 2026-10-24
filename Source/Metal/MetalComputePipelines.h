// Copyright (c) 2026 RTGL1 Metal backend contributors
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

// Objective-C++ only

#import <Metal/Metal.h>

#include "Common.h"

#include <filesystem>
#include <array>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace RTGL1
{

// Resources by the names that RTGL1's GLSL shaders use, e.g. "framebufAlbedo", "GlobalUniform".
// Plays the role of Vulkan descriptor sets: compute pipelines' argument buffers are filled from it.
class MetalBindingTable
{
public:
    void SetTexture( const std::string& name, id< MTLTexture > texture );
    void SetSampler( const std::string& name, id< MTLSamplerState > sampler );
    void SetBuffer( const std::string& name, id< MTLBuffer > buffer, NSUInteger offset = 0 );
    void SetAccelerationStructure( const std::string&                          name,
                                   id< MTLAccelerationStructure >              tlas,
                                   std::span< const id< MTLAccelerationStructure > > blases );
    // An argument buffer member that is a runtime sized array (bindless textures / samplers):
    // 'buffer' is used as the whole argument buffer of its descriptor set,
    // 'resources' are made resident for reading
    void SetArgumentBuffer( const std::string&                     name,
                            id< MTLBuffer >                        buffer,
                            std::span< const id< MTLResource > > resources );

    struct Entry
    {
        id< MTLResource >                    resource{ nil };
        id< MTLSamplerState >                sampler{ nil };
        NSUInteger                           offset{ 0 };
        std::vector< id< MTLResource > > extraResources{};
    };
    const Entry* Find( const std::string& name ) const;

private:
    std::unordered_map< std::string, Entry > entries;
};

// RTGL1's GLSL shaders, compiled to Metal by CompileShadersMetal.py
// (a .metallib per shader, a descriptor set is an argument buffer at buffer index = set index):
// compute pipelines, and render pipelines of vertex + fragment shaders (<name>_vert, <name>_frag)
class MetalComputePipelines
{
public:
    MetalComputePipelines( id< MTLDevice > device, std::filesystem::path folder );

    MetalComputePipelines( const MetalComputePipelines& other )            = delete;
    MetalComputePipelines& operator=( const MetalComputePipelines& other ) = delete;

    // function constants: GLSL's specialization constants, by constant_id
    using Constants = std::map< uint32_t, uint32_t >;

    // must be called once per frame, before any Dispatch
    void BeginFrame( uint32_t frameIndex );

    // exactly threadsX * threadsY * threadsZ threads (non-uniform threadgroups), as for raygen shaders;
    // the threadgroup size is the shader's local_size
    void Dispatch( id< MTLComputeCommandEncoder > enc,
                   const char*                    shaderName,
                   const Constants&               constants,
                   const MetalBindingTable&       bindings,
                   uint32_t                       threadsX,
                   uint32_t                       threadsY,
                   uint32_t                       threadsZ = 1,
                   std::span< const uint8_t >     pushConstants = {} );

    // the same as vkCmdDispatch, for compute shaders
    // push constants: GLSL's push_constant block, if the shader has one
    void DispatchThreadgroups( id< MTLComputeCommandEncoder > enc,
                               const char*                    shaderName,
                               const Constants&               constants,
                               const MetalBindingTable&       bindings,
                               uint32_t                       groupsX,
                               uint32_t                       groupsY,
                               uint32_t                       groupsZ       = 1,
                               std::span< const uint8_t >     pushConstants = {} );

    MTLSize GetThreadgroupSize( const char* shaderName, const Constants& constants );

    // as VkPipelineColorBlendAttachmentState
    struct Blend
    {
        bool           enable{ false };
        MTLBlendFactor src{ MTLBlendFactorOne };
        MTLBlendFactor dst{ MTLBlendFactorZero };
        // if set, alpha is blended with these, not with 'src' / 'dst'
        bool           separateAlpha{ false };
        MTLBlendFactor srcAlpha{ MTLBlendFactorOne };
        MTLBlendFactor dstAlpha{ MTLBlendFactorZero };

        bool operator==( const Blend& ) const = default;
    };

    struct RenderDesc
    {
        const char* vertexShader{ nullptr };
        const char* fragmentShader{ nullptr };
        Constants   vertexConstants{};
        Constants   fragmentConstants{};

        std::array< MTLPixelFormat, 4 > colorFormats{
            MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid, MTLPixelFormatInvalid };
        std::array< Blend, 4 > blends{};
        MTLPixelFormat         depthFormat{ MTLPixelFormatInvalid };

        // RasterizedDataCollector::GetVertexLayout, at RasterVertexBufferIndex; otherwise, no vertex input
        bool rasterVertexLayout{ true };
        bool lines{ false };
    };
    static constexpr NSUInteger RasterVertexBufferIndex = 30;

    // sets the render pipeline, and its resources by the names in RTGL1's shaders;
    // push constants: GLSL's push_constant block, shared by the vertex and fragment stages
    void BindRender( id< MTLRenderCommandEncoder > enc,
                     const RenderDesc&             desc,
                     const MetalBindingTable&      bindings,
                     std::span< const uint8_t >    pushConstants = {} );

    // VkPipelineDepthStencilStateCreateInfo with VK_COMPARE_OP_LESS_OR_EQUAL
    id< MTLDepthStencilState > GetDepthState( bool test, bool write );

private:
    struct Member
    {
        std::string      name;
        NSUInteger       offset;
        MTLDataType      type;
        MTLBindingAccess access;
        // for textures
        MTLDataType      textureDataType;
        MTLTextureType   textureType;
        // for arrays
        bool             isArray;
        MTLDataType      elementType;
    };

    struct ArgumentSet
    {
        NSUInteger            bufferIndex;
        NSUInteger            size;
        std::vector< Member > members;
    };

    struct Pipeline
    {
        id< MTLComputePipelineState > pso;
        std::vector< ArgumentSet >    sets;
        MTLSize                       threadgroupSize;
        // GLSL's push_constant block (SPIRV-Cross makes it a plain buffer argument)
        std::optional< NSUInteger >   pushConstantsIndex;
        NSUInteger                    pushConstantsSize{ 0 };
    };

    struct StageBindings
    {
        std::vector< ArgumentSet >  sets;
        std::optional< NSUInteger > pushConstantsIndex;
        NSUInteger                  pushConstantsSize{ 0 };
    };

    struct RenderPipeline
    {
        id< MTLRenderPipelineState > pso;
        StageBindings                vertex;
        StageBindings                fragment;
    };

    id< MTLFunction > GetFunction( const char* shaderName, const Constants& constants );
    static void       ParseBindings( NSArray< id< MTLBinding > >* bindings, StageBindings& dst );

    const Pipeline&       Get( const char* shaderName, const Constants& constants );
    const RenderPipeline& GetRender( const RenderDesc& desc );
    void                  Bind( id< MTLComputeCommandEncoder > enc,
                                const Pipeline&                p,
                                const MetalBindingTable&       bindings,
                                std::span< const uint8_t >     pushConstants );

    // fills the argument buffers of 'sets', calls setBuffer( bufferIndex, buffer, offset ) for each;
    // 'used' are the resources to make resident, by usage: read, write, read-write
    template< typename SetBufferFunc >
    void FillArgumentBuffers( const std::vector< ArgumentSet >& sets,
                              const MetalBindingTable&          bindings,
                              SetBufferFunc&&                   setBuffer,
                              std::vector< id< MTLResource > > ( &used )[ 3 ] );

    // a placeholder for a member that has no resource in the binding table
    id< MTLResource > GetDummy( const Member& m );

    // per frame memory for argument buffers
    struct Upload
    {
        id< MTLBuffer > buffer;
        NSUInteger      offset;
        uint8_t*        ptr;
    };
    Upload AllocUpload( NSUInteger size );

private:
    id< MTLDevice >       device;
    std::filesystem::path folder;

    std::map< std::string, Pipeline >                   pipelines;
    std::map< std::string, RenderPipeline >             renderPipelines;
    std::map< uint32_t, id< MTLDepthStencilState > >     depthStates;
    std::map< std::string, id< MTLLibrary > > libraries;

    std::map< std::pair< MTLTextureType, MTLDataType >, id< MTLTexture > > dummyTextures;
    id< MTLBuffer >                                                       dummyBuffer{ nil };
    id< MTLSamplerState >                                                 dummySampler{ nil };
    std::vector< std::string >                                            reportedMissing;

    struct UploadPage
    {
        id< MTLBuffer > buffer;
        NSUInteger      used;
    };
    std::vector< UploadPage > uploadPages[ MAX_FRAMES_IN_FLIGHT ];
    uint32_t                  frameIndex{ 0 };
};

}
