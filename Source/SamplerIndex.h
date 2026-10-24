// Copyright (c) 2021 Sultim Tsyrendashiev
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

// SamplerManager::Handle's encoding of a sampler: shared by SamplerManager.cpp and SamplerHandle.cpp,
// so the Metal backend can use SamplerManager::Handle without Vulkan samplers.

#include "Common.h"

#include <utility>

namespace RTGL1
{
namespace samplerindex
{
    inline VkFilter RgFilterToVk( RgSamplerFilter r )
    {
        switch( r )
        {
            case RG_SAMPLER_FILTER_AUTO: return VK_FILTER_LINEAR;
            case RG_SAMPLER_FILTER_LINEAR: return VK_FILTER_LINEAR;
            case RG_SAMPLER_FILTER_NEAREST: return VK_FILTER_NEAREST;
            default: assert( 0 ); return VK_FILTER_NEAREST;
        }
    }

    inline VkSamplerAddressMode RgAddressModeToVk( RgSamplerAddressMode r )
    {
        switch( r )
        {
            case RG_SAMPLER_ADDRESS_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case RG_SAMPLER_ADDRESS_MODE_CLAMP: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            default: assert( 0 ); return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }

#define FILTER_LINEAR  ( 1u << 0 )
#define FILTER_NEAREST ( 2u << 0 )
#define FILTER_MASK    ( 3u << 0 )

#define ADDRESS_MODE_U_REPEAT               ( 1u << 2 )
#define ADDRESS_MODE_U_MIRRORED_REPEAT      ( 2u << 2 )
#define ADDRESS_MODE_U_CLAMP_TO_EDGE        ( 3u << 2 )
#define ADDRESS_MODE_U_CLAMP_TO_BORDER      ( 4u << 2 )
#define ADDRESS_MODE_U_MIRROR_CLAMP_TO_EDGE ( 5u << 2 )
#define ADDRESS_MODE_U_MASK                 ( 7u << 2 )

#define ADDRESS_MODE_V_REPEAT               ( 1u << 5 )
#define ADDRESS_MODE_V_MIRRORED_REPEAT      ( 2u << 5 )
#define ADDRESS_MODE_V_CLAMP_TO_EDGE        ( 3u << 5 )
#define ADDRESS_MODE_V_CLAMP_TO_BORDER      ( 4u << 5 )
#define ADDRESS_MODE_V_MIRROR_CLAMP_TO_EDGE ( 5u << 5 )
#define ADDRESS_MODE_V_MASK                 ( 7u << 5 )

    inline uint32_t ToIndex( VkFilter             filter,
                      VkSamplerAddressMode addressModeU,
                      VkSamplerAddressMode addressModeV )
    {
        uint32_t index = 0;

        switch( filter )
        {
            case VK_FILTER_NEAREST: index |= FILTER_NEAREST; break;
            case VK_FILTER_LINEAR: index |= FILTER_LINEAR; break;
            default: assert( 0 ); break;
        }

        switch( addressModeU )
        {
            case VK_SAMPLER_ADDRESS_MODE_REPEAT: index |= ADDRESS_MODE_U_REPEAT; break;
            case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
                index |= ADDRESS_MODE_U_MIRRORED_REPEAT;
                break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
                index |= ADDRESS_MODE_U_CLAMP_TO_EDGE;
                break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
                index |= ADDRESS_MODE_U_CLAMP_TO_BORDER;
                break;
            case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
                index |= ADDRESS_MODE_U_MIRROR_CLAMP_TO_EDGE;
                break;
            default: assert( 0 ); break;
        }

        switch( addressModeV )
        {
            case VK_SAMPLER_ADDRESS_MODE_REPEAT: index |= ADDRESS_MODE_V_REPEAT; break;
            case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
                index |= ADDRESS_MODE_V_MIRRORED_REPEAT;
                break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
                index |= ADDRESS_MODE_V_CLAMP_TO_EDGE;
                break;
            case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
                index |= ADDRESS_MODE_V_CLAMP_TO_BORDER;
                break;
            case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
                index |= ADDRESS_MODE_V_MIRROR_CLAMP_TO_EDGE;
                break;
            default: assert( 0 ); break;
        }

        assert( index != 0 );
        return index;
    }

    inline auto GetRgAddressModesFromIndex( uint32_t index )
    {
        RgSamplerAddressMode u = RG_SAMPLER_ADDRESS_MODE_CLAMP;
        RgSamplerAddressMode v = RG_SAMPLER_ADDRESS_MODE_CLAMP;

        if( ( index & ADDRESS_MODE_U_MASK ) == ADDRESS_MODE_U_REPEAT ||
            ( index & ADDRESS_MODE_U_MASK ) == ADDRESS_MODE_U_MIRRORED_REPEAT )
        {
            u = RG_SAMPLER_ADDRESS_MODE_REPEAT;
        }

        if( ( index & ADDRESS_MODE_V_MASK ) == ADDRESS_MODE_V_REPEAT ||
            ( index & ADDRESS_MODE_V_MASK ) == ADDRESS_MODE_V_MIRRORED_REPEAT )
        {
            v = RG_SAMPLER_ADDRESS_MODE_REPEAT;
        }

        {
            auto check =
                ToIndex( VK_FILTER_NEAREST, RgAddressModeToVk( u ), RgAddressModeToVk( v ) );

            assert( ( check & ADDRESS_MODE_U_MASK ) == ( index & ADDRESS_MODE_U_MASK ) );
            assert( ( check & ADDRESS_MODE_V_MASK ) == ( index & ADDRESS_MODE_V_MASK ) );
        }

        return std::make_pair( u, v );
    }

    inline RgSamplerFilter GetRgFilterFromIndex( uint32_t index )
    {
        if( ( index & FILTER_MASK ) == FILTER_NEAREST )
        {
            return RG_SAMPLER_FILTER_NEAREST;
        }
        if( ( index & FILTER_MASK ) == FILTER_LINEAR )
        {
            return RG_SAMPLER_FILTER_LINEAR;
        }
        assert( 0 );
        return RG_SAMPLER_FILTER_AUTO;
    }

    inline uint32_t ToIndex( RgSamplerFilter      filter,
                      RgSamplerAddressMode addressModeU,
                      RgSamplerAddressMode addressModeV )
    {
        return ToIndex( RgFilterToVk( filter ),
                        RgAddressModeToVk( addressModeU ),
                        RgAddressModeToVk( addressModeV ) );
    }

    inline uint32_t SwapFilterInIndex( uint32_t srcIndex, RgSamplerFilter newFilter )
    {
        // clear previous filter type
        uint32_t i = srcIndex & ( ~FILTER_MASK );

        switch( newFilter )
        {
            case RG_SAMPLER_FILTER_NEAREST: i |= FILTER_NEAREST; break;
            case RG_SAMPLER_FILTER_LINEAR: i |= FILTER_LINEAR; break;
            case RG_SAMPLER_FILTER_AUTO:
            default:
                assert( 0 );
                i |= FILTER_LINEAR;
                break;
        }

        assert( i != 0 );
        return i;
    }
}
}
