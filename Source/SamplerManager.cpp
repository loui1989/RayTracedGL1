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

#include "SamplerManager.h"
#include "SamplerIndex.h"

#include <string>

#include "RgException.h"

namespace RTGL1
{
using namespace samplerindex;
}


RTGL1::SamplerManager::SamplerManager( VkDevice _device,
                                       uint32_t _anisotropy,
                                       bool     _forceMinificationFilterLinear )
    : device( _device )
    , mipLodBias( 0.0f )
    , anisotropy( _anisotropy )
    , forceMinificationFilterLinear( _forceMinificationFilterLinear )
{
    CreateAllSamplers( anisotropy, mipLodBias );
}

RTGL1::SamplerManager::~SamplerManager()
{
    for( auto& p : samplers )
    {
        vkDestroySampler( device, p.second, nullptr );
    }

    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        for( VkSampler s : samplersToDelete[ i ] )
        {
            vkDestroySampler( device, s, nullptr );
        }

        samplersToDelete[ i ].clear();
    };
    samplers.clear();
}

void RTGL1::SamplerManager::CreateAllSamplers( uint32_t _anisotropy, float _mipLodBias )
{
    assert( samplers.empty() );
    assert( _anisotropy == 0 || _anisotropy == 2 || _anisotropy == 4 || _anisotropy == 8 ||
            _anisotropy == 16 );

    VkSamplerCreateInfo info = {
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias              = _mipLodBias,
        .anisotropyEnable        = _anisotropy > 0 ? VK_TRUE : VK_FALSE,
        .maxAnisotropy           = float( _anisotropy ),
        .compareEnable           = VK_FALSE,
        .minLod                  = 0.0f,
        .maxLod                  = VK_LOD_CLAMP_NONE,
        .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkFilter filters[] = {
        VK_FILTER_NEAREST,
        VK_FILTER_LINEAR,
    };

    VkSamplerAddressMode modes[] = {
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };

    for( auto filter : filters )
    {
        for( auto modeU : modes )
        {
            for( auto modeV : modes )
            {
                info.minFilter    = forceMinificationFilterLinear ? VK_FILTER_LINEAR : filter;
                info.magFilter    = filter;
                info.addressModeU = modeU;
                info.addressModeV = modeV;

                uint32_t  index = ToIndex( filter, modeU, modeV );
                VkSampler sampler;

                VkResult r = vkCreateSampler( device, &info, nullptr, &sampler );
                VK_CHECKERROR( r );

                assert( samplers.find( index ) == samplers.end() );

                samplers[ index ] = sampler;
            }
        }
    }
}

void RTGL1::SamplerManager::AddAllSamplersToDestroy( uint32_t frameIndex )
{
    for( auto& p : samplers )
    {
        samplersToDelete[ frameIndex ].push_back( p.second );
    }

    samplers.clear();
}

void RTGL1::SamplerManager::PrepareForFrame( uint32_t frameIndex )
{
    for( VkSampler s : samplersToDelete[ frameIndex ] )
    {
        vkDestroySampler( device, s, nullptr );
    }

    samplersToDelete[ frameIndex ].clear();
}

VkSampler RTGL1::SamplerManager::GetSampler( RgSamplerFilter      filter,
                                             RgSamplerAddressMode addressModeU,
                                             RgSamplerAddressMode addressModeV ) const
{
    auto f = samplers.find( ToIndex( filter, addressModeU, addressModeV ) );

    if( f == samplers.end() )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT,
                           "Wrong RgSamplerFilter(" + std::to_string( filter ) +
                               ") or RgSamplerAddressMode (U: " + std::to_string( addressModeU ) +
                               ", V: " + std::to_string( addressModeV ) + ") value" );
    }

    return f->second;
}

VkSampler RTGL1::SamplerManager::GetSampler( const Handle& handle ) const
{
    assert( handle.internalIndex != 0 );
    auto f = samplers.find( handle.internalIndex );

    if( f == samplers.end() )
    {
        // pHandle->internalIndex is incorrect
        assert( 0 );
        return VK_NULL_HANDLE;
    }

    return f->second;
}

bool RTGL1::SamplerManager::TryChangeMipLodBias( uint32_t frameIndex, float newMipLodBias )
{
    constexpr float delta = 0.025f;

    if( std::abs( newMipLodBias - mipLodBias ) < delta )
    {
        return false;
    }

    AddAllSamplersToDestroy( frameIndex );
    CreateAllSamplers( anisotropy, newMipLodBias );

    mipLodBias = newMipLodBias;
    return true;
}
