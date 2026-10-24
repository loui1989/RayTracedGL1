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

#include "Generated/ShaderCommonCFramebufMetal.h"

#include <array>

namespace RTGL1
{

class MetalBindingTable;

// The images of ShaderCommonGLSL.h's framebufs, the same as Framebuffers.cpp.
// A framebuf with FRAMEBUF_FLAGS_STORE_PREV and its _Prev swap their images each frame.
class MetalFramebuffers
{
public:
    explicit MetalFramebuffers( id< MTLDevice > device );

    MetalFramebuffers( const MetalFramebuffers& other )            = delete;
    MetalFramebuffers& operator=( const MetalFramebuffers& other ) = delete;

    // (re)creates the images, if the sizes have changed
    void PrepareForSize( uint32_t renderWidth, uint32_t renderHeight, uint32_t upscaledWidth, uint32_t upscaledHeight );

    // registers framebuf<Name>, framebuf<Name>_Sampler, framebuf<Name>_SamplerSmplr
    // (and the same for the _Prev ones) for the given frame
    void Bind( MetalBindingTable& table, uint32_t frameIndex ) const;

    // the image that is framebuf<Name> in the given frame
    id< MTLTexture > Get( FramebufferImageIndex index, uint32_t frameIndex ) const;

private:
    id< MTLDevice > device;

    std::array< id< MTLTexture >, ShFramebuffers_Count > images{};
    id< MTLSamplerState >                                nearestSampler;
    id< MTLSamplerState >                                bilinearSampler;

    uint32_t renderWidth{ 0 }, renderHeight{ 0 };
    uint32_t upscaledWidth{ 0 }, upscaledHeight{ 0 };
};

}
