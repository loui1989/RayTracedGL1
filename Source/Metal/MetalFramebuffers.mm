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

#include "MetalFramebuffers.h"

#include "MetalComputePipelines.h"

#include <algorithm>
#include <string>

namespace
{

id< MTLSamplerState > MakeSampler( id< MTLDevice > device, MTLSamplerMinMagFilter filter )
{
    MTLSamplerDescriptor* d   = [MTLSamplerDescriptor new];
    d.minFilter               = filter;
    d.magFilter               = filter;
    d.sAddressMode            = MTLSamplerAddressModeClampToEdge;
    d.tAddressMode            = MTLSamplerAddressModeClampToEdge;
    d.supportArgumentBuffers  = YES;
    return [device newSamplerStateWithDescriptor:d];
}

}

RTGL1::MetalFramebuffers::MetalFramebuffers( id< MTLDevice > _device )
    : device{ _device }
    , nearestSampler{ MakeSampler( _device, MTLSamplerMinMagFilterNearest ) }
    , bilinearSampler{ MakeSampler( _device, MTLSamplerMinMagFilterLinear ) }
{
}

void RTGL1::MetalFramebuffers::PrepareForSize( uint32_t _renderWidth,
                                               uint32_t _renderHeight,
                                               uint32_t _upscaledWidth,
                                               uint32_t _upscaledHeight )
{
    if( renderWidth == _renderWidth && renderHeight == _renderHeight && upscaledWidth == _upscaledWidth &&
        upscaledHeight == _upscaledHeight && images[ 0 ] )
    {
        return;
    }
    renderWidth    = _renderWidth;
    renderHeight   = _renderHeight;
    upscaledWidth  = _upscaledWidth;
    upscaledHeight = _upscaledHeight;

    for( uint32_t i = 0; i < ShFramebuffers_Count; i++ )
    {
        const ShFramebufferMetalInfo& info = ShFramebuffers_MetalInfos[ i ];

        // the same as Framebuffers::GetImageSize
        uint32_t w, h;
        if( info.flags & FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE )
        {
            w = h = 1;
        }
        else
        {
            const bool upscaled = ( info.flags & FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_UPSCALED_SIZE ) ||
                                  ( info.flags & FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM );
            w = upscaled ? upscaledWidth : renderWidth;
            h = upscaled ? upscaledHeight : renderHeight;
            w = std::max( 1u, ( w + info.sizeDivisor - 1 ) / info.sizeDivisor );
            h = std::max( 1u, ( h + info.sizeDivisor - 1 ) / info.sizeDivisor );
        }

        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.format
                                                                                        width:w
                                                                                       height:h
                                                                                    mipmapped:NO];
        desc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        desc.storageMode = MTLStorageModePrivate;
        if( info.flags & FB_IMAGE_FLAGS_FRAMEBUF_FLAGS_IS_ATTACHMENT )
        {
            desc.usage |= MTLTextureUsageRenderTarget;
        }

        images[ i ]       = [device newTextureWithDescriptor:desc];
        images[ i ].label = @( info.name );
    }
}

id< MTLTexture > RTGL1::MetalFramebuffers::Get( FramebufferImageIndex index, uint32_t frameIndex ) const
{
    // the same as Framebuffers::FrameIndexToFBIndex: _Prev of the previous frame is the current one
    const int32_t swap = ShFramebuffers_MetalInfos[ index ].swapPartnerOffset;
    return images[ ( frameIndex % 2 == 1 && swap != 0 ) ? index + swap : index ];
}

void RTGL1::MetalFramebuffers::Bind( MetalBindingTable& table, uint32_t frameIndex ) const
{
    for( uint32_t i = 0; i < ShFramebuffers_Count; i++ )
    {
        const ShFramebufferMetalInfo& info  = ShFramebuffers_MetalInfos[ i ];
        id< MTLTexture >              image = Get( FramebufferImageIndex( i ), frameIndex );

        const std::string name = info.name;
        table.SetTexture( name, image );
        if( info.hasSampler )
        {
            // SPIRV-Cross splits a combined image sampler into <name> and <name>Smplr
            table.SetTexture( name + "_Sampler", image );
            table.SetSampler( name + "_SamplerSmplr", info.bilinearSampler ? bilinearSampler : nearestSampler );
        }
    }
}
