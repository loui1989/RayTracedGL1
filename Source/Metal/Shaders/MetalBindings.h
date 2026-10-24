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

// Buffer indices of compute kernels, shared by the host and the Metal shaders.
// Metal has no descriptor sets: these play the role of RTGL1's DESC_SET_* / BINDING_*.

#define MTL_BINDING_GLOBAL_UNIFORM  0
#define MTL_BINDING_TLAS            1
#define MTL_BINDING_VERTICES        2
#define MTL_BINDING_INDICES         3
#define MTL_BINDING_GEOMETRIES      4
#define MTL_BINDING_TEXTURES        5
#define MTL_BINDING_INSTANCE_INFOS  6
#define MTL_BINDING_FRAMEBUFFERS    7

// per TLAS instance of MetalScene
struct MetalInstanceInfo
{
    // index of the first ShGeometryInstance of the instance
    uint32_t geometryBase;
    // INSTANCE_CUSTOM_INDEX_FLAG_*
    uint32_t customIndex;
};

// Rasterized swapchain (UI) draws
#define MTL_RASTER_BINDING_VERTICES  0
#define MTL_RASTER_BINDING_CONSTANTS 1
#define MTL_RASTER_BINDING_TEXTURES  2

struct MetalRasterConstants
{
    // column-major model-view-projection, as GLSL's mat4
    float    mvp[ 16 ];
    uint32_t color;
    uint32_t textureIndex;
    uint32_t alphaTest;
    uint32_t vertexColorGamma;
};
