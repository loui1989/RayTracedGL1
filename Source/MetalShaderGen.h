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

// macOS native Metal path: translation of the Vulkan compute shaders (SPIR-V) into Metal
// Shading Language with SPIRV-Cross. Each Vulkan descriptor set becomes a Metal argument
// buffer bound at buffer index = set index. Inside a set, resources get 8-byte slots in the
// same order MoltenVK uses: for each binding (by binding number), its textures, then its
// buffer or samplers. Slot n is at byte offset n * 8 and holds a resource ID (textures,
// samplers, acceleration structures) or a GPU address (buffers).

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace RTGL1::MetalGen
{

// Metal buffer index of the push constants (argument buffers use indices 0..set count-1)
constexpr uint32_t PUSH_CONSTANT_BUFFER_INDEX = 27;
// graphics: (first view, view count) for the multiview emulation, see TranslateGraphics
constexpr uint32_t VIEW_MASK_BUFFER_INDEX = 26;
// graphics: vertex buffer of Vulkan binding b is at Metal buffer index VERTEX_BUFFER_BASE + b
constexpr uint32_t VERTEX_BUFFER_BASE = 16;

struct BindingDesc
{
    uint32_t         binding;
    VkDescriptorType type;
    uint32_t         count;
};

struct SlotInfo
{
    uint32_t         binding;
    VkDescriptorType type;
    uint32_t         count;
    uint32_t         textureSlot; // first slot of the textures (if any)
    uint32_t         bufferSlot;  // slot of the buffer / acceleration structure (if any)
    uint32_t         samplerSlot; // first slot of the samplers (if any)
};

struct SetSlots
{
    std::vector< SlotInfo > slots; // sorted by binding
    uint32_t                totalSlots = 0;

    const SlotInfo* Find( uint32_t binding ) const;
};

// Sorts the bindings and assigns argument buffer slots.
SetSlots AssignSlots( std::vector< BindingDesc > bindings );

struct ComputeShader
{
    bool        ok = false;
    std::string error;
    std::string msl;
    std::string entryPoint;
    uint32_t    localSize[ 3 ] = { 1, 1, 1 };
};

// 'sets' is the pipeline layout: sets[i] is the layout of descriptor set i.
// mslVersion is e.g. 30100 for MSL 3.1.
ComputeShader TranslateCompute( const uint32_t*                spirv,
                                size_t                         wordCount,
                                const std::vector< SetSlots >& sets,
                                uint32_t                       mslVersion );

// A vertex or fragment shader (the stage is the one of the SPIR-V's entry point).
// Vertex attributes are [[attribute(location)]], read from the vertex buffers described by
// the pipeline's vertex descriptor; clip space y is flipped for Metal. multiview: the
// render pass has several views, gl_ViewIndex is emulated with instancing (multiply the
// instance count by the view count, (first view, view count) as two uints at
// VIEW_MASK_BUFFER_INDEX), and each view renders to its layer.
ComputeShader TranslateGraphics( const uint32_t*                spirv,
                                 size_t                         wordCount,
                                 const std::vector< SetSlots >& sets,
                                 uint32_t                       mslVersion,
                                 bool                           multiview );

}
