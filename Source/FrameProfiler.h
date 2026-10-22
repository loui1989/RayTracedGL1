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
#include "Buffer.h"

#include <chrono>
#include <vector>

namespace RTGL1
{

// Optional frame profiler, enabled with the environment variable RTGL1_PROFILE=1.
// Measures GPU time per render stage (timestamp queries) and CPU time per frame stage,
// and prints averages to stderr every 2 seconds.
//
// MoltenVK can only take GPU timestamps between Metal encoders, so each GPU mark ends
// the current encoder with a tiny buffer copy. This costs a little performance.
class FrameProfiler
{
public:
    static bool IsRequested();

    FrameProfiler( VkDevice                                device,
                   VkPhysicalDevice                        physDevice,
                   const std::shared_ptr< MemoryAllocator >& allocator );
    ~FrameProfiler();

    FrameProfiler( const FrameProfiler& other )                = delete;
    FrameProfiler( FrameProfiler&& other ) noexcept            = delete;
    FrameProfiler& operator=( const FrameProfiler& other )     = delete;
    FrameProfiler& operator=( FrameProfiler&& other ) noexcept = delete;

    bool IsValid() const { return pool != VK_NULL_HANDLE; }

    // Call when the frame index's fence was waited for: collects its previous results
    // and starts the GPU timeline of the new frame. Must be outside of a render pass.
    void BeginFrame( VkCommandBuffer cmd, uint32_t frameIndex );
    // Ends a GPU stage that started at the previous mark. Must be outside of a render pass.
    // 'name' must be a string literal.
    void Gpu( VkCommandBuffer cmd, const char* name );
    // Ends a CPU stage that started at the previous CPU mark. 'name' must be a string literal.
    void Cpu( const char* name );
    // Additional measured value (e.g. a Metal command buffer's GPU time).
    void Value( const char* name, double ms );

private:
    struct Acc
    {
        const char* name;
        double      sum;
    };
    static void Add( std::vector< Acc >& v, const char* name, double ms );
    void        Report( double seconds );

    static constexpr uint32_t MAX_MARKS = 40;

    VkDevice    device;
    VkQueryPool pool;
    Buffer      breakBuffer;
    double      periodNs;

    uint32_t curFrameIndex;
    struct
    {
        uint32_t    count;
        const char* names[ MAX_MARKS ];
    } marks[ MAX_FRAMES_IN_FLIGHT ];

    std::vector< Acc > gpuAcc, cpuAcc, valueAcc;
    uint32_t           gpuFrames;
    uint32_t           cpuFrames;

    std::chrono::steady_clock::time_point lastCpuMark;
    std::chrono::steady_clock::time_point lastReport;
};

}
