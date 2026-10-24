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

#include "RTGL1/RTGL1.h"

#include "Common.h"

#include <vector>

struct ShParticleSourceDef;

namespace RTGL1
{

class MetalBindingTable;
class MetalComputePipelines;

// The same as Fluid.cpp: particles are generated from sources, simulated (with collisions
// by ray queries), and rasterized as sphere impostors into framebufDepthFluid / framebufFluidNormal,
// that are smoothed and then used by the primary rays (RaygenPrimary.inl, tryAsFluid)
class MetalFluid
{
public:
    MetalFluid( id< MTLDevice > device, id< MTLLibrary > library, uint32_t fluidBudget, float particleRadius );
    ~MetalFluid();

    MetalFluid( const MetalFluid& )            = delete;
    MetalFluid& operator=( const MetalFluid& ) = delete;

    void PrepareForFrame( bool reset );
    void AddSource( const RgSpawnFluidInfo& src );
    bool Active() const;

    // Fluid::Simulate; 'bindings' must have the scene (TLAS, vertex data) and the global uniform
    void Simulate( id< MTLCommandBuffer > cmd,
                   MetalComputePipelines& shaders,
                   MetalBindingTable&     bindings,
                   uint32_t               frameIndex,
                   float                  deltaTime,
                   const RgFloat3D&       gravity );

    // Fluid::Visualize; 'bindings' must have the framebuffers
    void Visualize( id< MTLCommandBuffer > cmd,
                    MetalComputePipelines& shaders,
                    MetalBindingTable&     bindings,
                    id< MTLTexture >       depthFluid,
                    id< MTLTexture >       fluidNormal,
                    const float*           view,
                    const float*           projection,
                    uint32_t               width,
                    uint32_t               height,
                    float                  zNear,
                    float                  zFar );

    struct Ring
    {
        uint32_t begin{ 0 };
        uint32_t end{ 0 };
        bool     full{ false };
    };

private:
    void BindBuffers( MetalBindingTable& bindings, uint32_t frameIndex ) const;

private:
    id< MTLDevice >               device;
    id< MTLComputePipelineState > copyDepth{ nil };

    uint32_t maxParticles;
    float    particleRadius;

    id< MTLBuffer > particles{ nil };
    bool            particlesCleared{ false };
    // per frame in flight, as written by the CPU
    id< MTLBuffer > generateIdToSource[ MAX_FRAMES_IN_FLIGHT ]{};
    id< MTLBuffer > sources[ MAX_FRAMES_IN_FLIGHT ]{};

    // because sources can be added out-of-frame
    std::vector< ShParticleSourceDef > sourcesCached;
    std::vector< uint32_t >            sourcesCachedCount;

    Ring active{};

    // the raster pass' depth; then copied to framebufDepthFluid
    // (Vulkan aliases framebufDepthFluid's memory, Metal can't)
    id< MTLTexture > depth{ nil };
};

}
