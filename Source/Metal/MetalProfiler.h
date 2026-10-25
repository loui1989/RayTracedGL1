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

#import <Metal/Metal.h>

namespace RTGL1
{

// Debug: RTGL1_METAL_PROFILE=1, a GPU timeline of a frame every 2 seconds.
// All encoders are made through these functions: on a sampled frame, they get GPU timestamps
// at their start and end, and the frame's encoders are printed in the order of their start,
// with the time offset from the frame's first encoder. Work that isn't an RTGL1 encoder
// (MetalFX) shows up as a gap between the encoders around it.
namespace MetalProfiler
{
    void Init( id< MTLDevice > device );
    bool Enabled();

    // 'sample': if the encoders of this frame should get timestamps
    void BeginFrame( bool sample );
    // the samples are resolved and printed, when 'last' (the frame's last command buffer) completes
    void EndFrame( id< MTLCommandBuffer > last );

    id< MTLComputeCommandEncoder >               Compute( id< MTLCommandBuffer > cmd );
    id< MTLBlitCommandEncoder >                  Blit( id< MTLCommandBuffer > cmd );
    id< MTLRenderCommandEncoder >                Render( id< MTLCommandBuffer > cmd, MTLRenderPassDescriptor* pass );
    id< MTLAccelerationStructureCommandEncoder > AccelerationStructure( id< MTLCommandBuffer > cmd );
}

}
