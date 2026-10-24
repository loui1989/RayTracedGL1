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

// Moved from LightManager.cpp, to share with the Metal backend

#include "LightEncoding.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr double RG_PI = 3.1415926535897932384626433;

constexpr float MIN_COLOR_SUM     = 0.0001f;
constexpr float MIN_SPHERE_RADIUS = 0.005f;
}

using RTGL1::detail::encodeE5;
using RTGL1::detail::packHalf2x16;

namespace RTGL1
{

static_assert( sizeof( RTGL1::ShLightEncoded ) == 24, "Change encoding" );


RTGL1::ShLightEncoded EncodeAsDirectionalLight( const RgLightDirectionalEXT& info,
                                                float                        mult,
                                                const RgTransform*           transform )
{
    assert( !transform ); // not expected

    RgFloat3D direction = RTGL1::Utils::Normalize( info.direction );

    float angularRadius = 0.5f * RTGL1::Utils::DegToRad( info.angularDiameterDegrees );
    auto  fcolor        = RTGL1::Utils::UnpackColor4DPacked32< RgFloat3D >( info.color );
    {
        fcolor.data[ 0 ] *= info.intensity* mult;
        fcolor.data[ 1 ] *= info.intensity* mult;
        fcolor.data[ 2 ] *= info.intensity* mult;
    }


    RTGL1::ShLightEncoded lt;

    lt.lightType = LIGHT_TYPE_DIRECTIONAL;

    float norm;
    lt.colorE5 = encodeE5( fcolor, &norm );
    assert( norm <= 1.05f );

    lt.ldata0 = direction.data[ 0 ];
    lt.ldata1 = direction.data[ 1 ];
    lt.ldata2 = direction.data[ 2 ];

    lt.ldata3 = angularRadius;

    return lt;
}

RTGL1::ShLightEncoded EncodeAsSphereLight( const RgLightSphericalEXT& info,
                                           float                      mult,
                                           const RgTransform*         transform )
{
    RgFloat3D pos = RTGL1::ApplyTransformToPosition( transform, info.position );

    float radius = std::max( MIN_SPHERE_RADIUS, info.radius );
    // disk is visible from the point
    float area = float( RG_PI ) * radius * radius;

    auto fcolor = RTGL1::Utils::UnpackColor4DPacked32< RgFloat3D >( info.color );
    {
        fcolor.data[ 0 ] *= info.intensity / area * mult;
        fcolor.data[ 1 ] *= info.intensity / area * mult;
        fcolor.data[ 2 ] *= info.intensity / area * mult;
    }


    RTGL1::ShLightEncoded lt;

    lt.lightType = LIGHT_TYPE_SPHERE;

    float norm;
    lt.colorE5 = encodeE5( fcolor, &norm );

    lt.ldata0 = pos.data[ 0 ];
    lt.ldata1 = pos.data[ 1 ];
    lt.ldata2 = pos.data[ 2 ];

    {
        uint32_t* rn = reinterpret_cast< uint32_t* >( &lt.ldata3 );

        *rn = packHalf2x16(
            radius, //
            norm    // additional multiplier as e5 encoding might not preserve large values
        );
    }

    return lt;
}

#if TRIANGLE_LIGHTS
RTGL1::ShLightEncoded EncodeAsTriangleLight( const RgLightPolygonalEXT& info,
                                             const RgFloat3D&           unnormalizedNormal,
                                             float                      mult,
                                             const RgTransform*         transform )
{
    assert( !transform ); // not implemented

    RgFloat3D n   = unnormalizedNormal;
    float     len = RTGL1::Utils::Length( n.data );
    n.data[ 0 ] /= len;
    n.data[ 1 ] /= len;
    n.data[ 2 ] /= len;

    float area = len * 0.5f;
    assert( area > 0.0f );

    auto fcolor = RTGL1::Utils::UnpackColor4DPacked32< RgFloat3D >( info.color );


    RTGL1::ShLightEncoded lt = {};
    lt.lightType             = LIGHT_TYPE_TRIANGLE;

    lt.color[ 0 ] = fcolor.data[ 0 ] * info.intensity / area * mult;
    lt.color[ 1 ] = fcolor.data[ 1 ] * info.intensity / area * mult;
    lt.color[ 2 ] = fcolor.data[ 2 ] * info.intensity / area * mult;

    lt.data_0[ 0 ] = info.positions[ 0 ].data[ 0 ];
    lt.data_0[ 1 ] = info.positions[ 0 ].data[ 1 ];
    lt.data_0[ 2 ] = info.positions[ 0 ].data[ 2 ];

    lt.data_1[ 0 ] = info.positions[ 1 ].data[ 0 ];
    lt.data_1[ 1 ] = info.positions[ 1 ].data[ 1 ];
    lt.data_1[ 2 ] = info.positions[ 1 ].data[ 2 ];

    lt.data_2[ 0 ] = info.positions[ 2 ].data[ 0 ];
    lt.data_2[ 1 ] = info.positions[ 2 ].data[ 1 ];
    lt.data_2[ 2 ] = info.positions[ 2 ].data[ 2 ];

    lt.data_0[ 3 ] = unnormalizedNormal.data[ 0 ];
    lt.data_1[ 3 ] = unnormalizedNormal.data[ 1 ];
    lt.data_2[ 3 ] = unnormalizedNormal.data[ 2 ];

    return lt;
}
#endif

RTGL1::ShLightEncoded EncodeAsSpotLight( const RgLightSpotEXT& info,
                                         float                 mult,
                                         const RgTransform*    transform )
{
    RgFloat3D pos = RTGL1::ApplyTransformToPosition( transform, info.position );

    RgFloat3D direction =
        RTGL1::ApplyTransformToDirection( transform, RTGL1::Utils::Normalize( info.direction ) );
    assert( std::abs( RTGL1::Utils::Length( direction.data ) - 1.0f ) < 0.001f );

    float radius = std::max( MIN_SPHERE_RADIUS, info.radius );
    float area   = float( RG_PI ) * radius * radius;

    constexpr auto clampForCos = []( float a ) {
        return std::clamp( a, 0.0f, RTGL1::Utils::DegToRad( 89 ) );
    };

    constexpr auto float01to8bit = []( float a ){
        assert( a >= 0 && a <= 1 );
        return uint8_t( std::clamp( int( a * 255 ), 0, 255 ) );
    };

    float angleInner = std::min( info.angleInner, info.angleOuter - RTGL1::Utils::DegToRad( 1 ) );
    float angleOuter = info.angleOuter;

    uint8_t cosAngleInner = float01to8bit( std::cos( clampForCos( angleInner ) ) );
    uint8_t cosAngleOuter = float01to8bit( std::cos( clampForCos( angleOuter ) ) );

    auto fcolor = RTGL1::Utils::UnpackColor4DPacked32< RgFloat3D >( info.color );
    {
        fcolor.data[ 0 ] *= info.intensity / area * mult;
        fcolor.data[ 1 ] *= info.intensity / area * mult;
        fcolor.data[ 2 ] *= info.intensity / area * mult;
    }


    RTGL1::ShLightEncoded lt;

    lt.lightType = LIGHT_TYPE_SPOT;

    float norm;
    lt.colorE5 = encodeE5( fcolor, &norm );
    
    {
        uint32_t* data0_x = reinterpret_cast< uint32_t* >( &lt.ldata0 );
        uint32_t* data0_y = reinterpret_cast< uint32_t* >( &lt.ldata1 );

        *data0_x = packHalf2x16( pos.data[ 0 ], pos.data[ 1 ] );
        *data0_y = packHalf2x16( pos.data[ 2 ], norm );
    }
    {
        uint32_t* data0_z = reinterpret_cast< uint32_t* >( &lt.ldata2 );
        uint32_t* data0_w = reinterpret_cast< uint32_t* >( &lt.ldata3 );

        *data0_z = packHalf2x16( direction.data[ 0 ], direction.data[ 1 ] );
        *data0_w = packHalf2x16( 0, direction.data[ 2 ] );

        assert( ( ( *data0_w ) & 0x0000FFFF ) == 0 );
        *data0_w = ( *data0_w & 0xFFFF0000 ) | //
                   ( cosAngleInner << 8 ) |    //
                   ( cosAngleOuter );
    }

    return lt;
}

uint32_t GetLightArrayEnd( uint32_t regCount, uint32_t dirCount )
{
    // assuming that reg lights are always after directional ones
    return LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + regCount;
}

}

#include "glm/glm.hpp"

namespace
{

using namespace glm;
#include "Shaders/Utils.h"

}

uint32_t RTGL1::detail::encodeE5( const RgFloat3D& c, float* normalization )
{
    assert( normalization );

    static_assert( sizeof( vec3 ) == sizeof( RgFloat3D ) );
    const vec3& l = *reinterpret_cast< const vec3* >( &c );

    const float norm = std::max( { l.x, l.y, l.z } ) / float{ ENCODE_E5B9G9R9_SHAREDEXP_MAX };

    uint32_t r;
    if( norm <= 1.0f )
    {
        r = encodeE5B9G9R9( l );
    }
    else
    {
        // fallback: normalize to preserve colors, to not clamp to white
        r = encodeE5B9G9R9( l / norm );
    }

#if 0
    {
        vec3 debugC = decodeE5B9G9R9( r );
        assert( length( debugC - vec3( c.data[ 0 ], c.data[ 1 ], c.data[ 2 ] ) ) < 10 );
    }
#endif

    *normalization = norm;
    return r;
}

uint32_t RTGL1::detail::packHalf2x16( const float& x, const float& y )
{
    return glm::packHalf2x16( { x, y } );
}

auto RTGL1::TryGetVolumetricLight( const RgFloat3D&                 cameraPos,
                                   const std::vector< LightCopy >&  from,
                                   const std::optional< uint64_t >& fallback,
                                   std::span< const uint8_t >       lightstyles ) -> std::optional< uint64_t >
{
    static auto l_isVolumetric = []( const LightCopy& l ) {
        return l.additional && ( l.additional->flags & RG_LIGHT_ADDITIONAL_VOLUMETRIC );
    };

    auto l_approxVolumetricIntensity = [ lightstyles ]( const LightCopy& l ) {
        assert( l_isVolumetric( l ) );

        float intensity =
            std::visit( []( const auto& lext ) { return lext.intensity; }, l.extension );

        return intensity * CalculateLightStyle( l.additional, lightstyles );
    };
    auto l_approxDistanceSq = []( const LightCopy& l, const RgFloat3D& from ) -> float {
        // clang-format off
        return std::visit( ext::overloaded{
            [ & ]( const RgLightDirectionalEXT& lext ) { return 0.f; },
            [ & ]( const RgLightSphericalEXT&   lext ) { return Utils::SqrDistanceR( lext.position, from ); },
            [ & ]( const RgLightSpotEXT&        lext ) { return Utils::SqrDistanceR( lext.position, from ); },
            [ & ]( const RgLightPolygonalEXT&   lext ) { assert( 0 ); return -1.f; },
        }, l.extension );
        // clang-format on
    };

    struct Candidate
    {
        uint64_t id;
        float    distanceSq;
    };
    auto best = std::optional< Candidate >{};
    auto any  = std::optional< Candidate >{};

    for( const auto& l : from )
    {
        if( !l_isVolumetric( l ) )
        {
            continue;
        }
        
        if( l_approxVolumetricIntensity( l ) > 0.0f )
        {
            float distSq = l_approxDistanceSq( l, cameraPos );
            if( distSq >= 0 )
            {
                // choose closest
                if( !best || distSq < best->distanceSq )
                {
                    best = Candidate{
                        .id         = l.base.uniqueID,
                        .distanceSq = distSq,
                    };
                }
            }
        }

        if( !any )
        {
            any = Candidate{
                .id         = l.base.uniqueID,
                .distanceSq = 0,
            };
        }
    }

    if( best )
    {
        return best->id;
    }

    // SHIPPING_HACK: don't fallback to sun, if at least
    // one light is marked as isVolumetric, but has 0 intensity
    if( any )
    {
        return any->id;
    }

    // if nothing, just try find the sun in the provided list
    for( const auto& l : from )
    {
        if( auto sun = std::get_if< RgLightDirectionalEXT >( &l.extension ) )
        {
            return l.base.uniqueID;
        }
    }

    return fallback;
}
