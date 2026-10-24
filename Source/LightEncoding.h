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

// Encoding of lights into ShLightEncoded, shared by the Vulkan (LightManager) and Metal backends

#include "RTGL1/RTGL1.h"

#include "Common.h"
#include "Generated/ShaderCommonC.h"
#include "Utils.h"

#include <optional>
#include <span>
#include <vector>

namespace RTGL1
{

constexpr uint32_t LIGHT_ARRAY_MAX_SIZE = 4096;

namespace detail
{
    uint32_t encodeE5( const RgFloat3D& c, float* normalization );
    uint32_t packHalf2x16( const float& x, const float& y );
}

ShLightEncoded EncodeAsDirectionalLight( const RgLightDirectionalEXT& info,
                                         float                        mult,
                                         const RgTransform*           transform );
ShLightEncoded EncodeAsSphereLight( const RgLightSphericalEXT& info,
                                    float                      mult,
                                    const RgTransform*         transform );
#if TRIANGLE_LIGHTS
ShLightEncoded EncodeAsTriangleLight( const RgLightPolygonalEXT& info,
                                      const RgFloat3D&           unnormalizedNormal,
                                      float                      mult,
                                      const RgTransform*         transform );
#endif
ShLightEncoded EncodeAsSpotLight( const RgLightSpotEXT& info,
                                  float                 mult,
                                  const RgTransform*    transform );

uint32_t GetLightArrayEnd( uint32_t regCount, uint32_t dirCount );

template< typename T >
inline bool IsLightColorTooDim( const T& l )
{
    if( l.intensity <= 0.00001f )
    {
        return true;
    }

    if( RTGL1::Utils::IsColor4DPacked32Zero< false >( l.color ) )
    {
        return true;
    }

    return false;
}

inline float CalculateLightStyle( const std::optional< RgLightAdditionalEXT >& extra,
                           std::span< const uint8_t >                   lightstyles )
{
    if( extra && ( extra->flags & RG_LIGHT_ADDITIONAL_LIGHTSTYLE ) )
    {
        if( extra->lightstyle >= 0 && size_t( extra->lightstyle ) < lightstyles.size() )
        {
            return float( lightstyles[ extra->lightstyle ] ) / 255.0f;
        }
        else
        {
            assert( 0 );
        }
    }
    return 1.0f;
}

// The light that volumetrics use: the closest volumetric one, or the sun
auto TryGetVolumetricLight( const RgFloat3D&                 cameraPos,
                            const std::vector< LightCopy >&  from,
                            const std::optional< uint64_t >& fallback,
                            std::span< const uint8_t >       lightstyles ) -> std::optional< uint64_t >;

}
