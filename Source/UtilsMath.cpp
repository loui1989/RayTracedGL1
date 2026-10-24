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
// Math helpers of Utils.h, without Vulkan dependencies:
// compiled by both the Vulkan and the Metal builds.

#include "Utils.h"

#include <cmath>

using namespace RTGL1;

constexpr float ALMOST_ZERO_THRESHOLD = 0.01f;

bool Utils::IsAlmostZero( const float v[ 3 ] )
{
    return std::abs( v[ 0 ] ) + std::abs( v[ 1 ] ) + std::abs( v[ 2 ] ) < ALMOST_ZERO_THRESHOLD;
}

bool Utils::IsAlmostZero( const RgFloat3D& v )
{
    return IsAlmostZero( v.data );
}

bool Utils::IsAlmostZero( const RgMatrix3D& m )
{
    float s = 0;

    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            s += std::abs( m.matrix[ i ][ j ] );
        }
    }

    return s < ALMOST_ZERO_THRESHOLD;
}

bool Utils::TryNormalize( float inout[ 3 ] )
{
    float len = Length( inout );

    if( len < 0.001f )
    {
        return false;
    }

    inout[ 0 ] /= len;
    inout[ 1 ] /= len;
    inout[ 2 ] /= len;
    return true;
}


void Utils::Normalize( float inout[ 3 ] )
{
    bool s = TryNormalize( inout );
    assert( s );
}

RgFloat3D Utils::Normalize( const RgFloat3D& v )
{
    RgFloat3D dst = v;
    TryNormalize( dst.data );
    return dst;
}

void Utils::SafeNormalize( float ( &v )[ 3 ], const RgFloat3D& fallback )
{
    if( TryNormalize( v ) )
    {
        return;
    }

    v[ 0 ] = fallback.data[ 0 ];
    v[ 1 ] = fallback.data[ 1 ];
    v[ 2 ] = fallback.data[ 2 ];
}

RgFloat3D Utils::SafeNormalize( const RgFloat3D& v, const RgFloat3D& fallback )
{
    RgFloat3D dst = v;
    SafeNormalize( dst.data, fallback );
    return dst;
}

void Utils::Negate( float inout[ 3 ] )
{
    inout[ 0 ] *= -1;
    inout[ 1 ] *= -1;
    inout[ 2 ] *= -1;
}

void Utils::Nullify( float inout[ 3 ] )
{
    inout[ 0 ] = 0;
    inout[ 1 ] = 0;
    inout[ 2 ] = 0;
}


void Utils::Cross( const float a[ 3 ], const float b[ 3 ], float r[ 3 ] )
{
    r[ 0 ] = a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ];
    r[ 1 ] = a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ];
    r[ 2 ] = a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ];
}

auto Utils::Cross( const RgFloat3D& a, const RgFloat3D& b ) -> RgFloat3D
{
    RgFloat3D r;
    Cross( a.data, b.data, r.data );
    return r;
}

RgFloat3D Utils::GetUnnormalizedNormal( const RgFloat3D positions[ 3 ] )
{
    const float* a = positions[ 0 ].data;
    const float* b = positions[ 1 ].data;
    const float* c = positions[ 2 ].data;

    float e1[ 3 ] = { b[ 0 ] - a[ 0 ], b[ 1 ] - a[ 1 ], b[ 2 ] - a[ 2 ] };
    float e2[ 3 ] = { c[ 0 ] - a[ 0 ], c[ 1 ] - a[ 1 ], c[ 2 ] - a[ 2 ] };

    RgFloat3D n = {};
    Cross( e1, e2, n.data );

    return n;
}

bool Utils::GetNormalAndArea( const RgFloat3D positions[ 3 ], RgFloat3D& normal, float& area )
{
    normal = GetUnnormalizedNormal( positions );

    float len = Length( normal.data );
    normal.data[ 0 ] /= len;
    normal.data[ 1 ] /= len;
    normal.data[ 2 ] /= len;

    area = len * 0.5f;
    return area > 0.01f;
}


void Utils::SetMatrix3ToGLSLMat4( float dst[ 16 ], const RgMatrix3D& src )
{
    const bool toColumnMajor = true;

    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
            float v;

            if( i < 3 && j < 3 )
            {
                v = toColumnMajor ? src.matrix[ j ][ i ] : src.matrix[ i ][ j ];
            }
            else
            {
                v = i == j ? 1 : 0;
            }

            dst[ i * 4 + j ] = v;
        }
    }
}

RgTransform Utils::MakeTransform( const RgFloat3D& up, const RgFloat3D& forward, float scale )
{
    RgFloat3D forw = Normalize( forward );

    float right[ 3 ];
    Cross( up.data, forw.data, right );

    float rot[ 3 ][ 3 ] = {
        { right[ 0 ], up.data[ 0 ], forw.data[ 0 ] },
        { right[ 1 ], up.data[ 1 ], forw.data[ 1 ] },
        { right[ 2 ], up.data[ 2 ], forw.data[ 2 ] },
    };

    float scl[ 3 ][ 3 ] = {
        { scale, 0, 0 },
        { 0, scale, 0 },
        { 0, 0, scale },
    };

    RgTransform tr = {};
    for( int i = 0; i < 3; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            for( int k = 0; k < 3; k++ )
            {
                tr.matrix[ i ][ j ] += rot[ k ][ j ] * scl[ i ][ k ];
            }
        }
    }
    return tr;
}

RgTransform Utils::MakeTransform( const RgFloat3D& position, const RgFloat3D& forward )
{
    RgFloat3D forw = Normalize( forward );

    RgFloat3D up = std::abs( forw.data[ 1 ] ) > 0.99f ? RgFloat3D{ 1, 0, 0 } : RgFloat3D{ 0, 1, 0 };

    RgFloat3D right{};
    Cross( up.data, forw.data, right.data );

    Cross( forw.data, right.data, up.data );

    float rot[ 3 ][ 3 ] = {
        { right.data[ 0 ], up.data[ 0 ], forw.data[ 0 ] },
        { right.data[ 1 ], up.data[ 1 ], forw.data[ 1 ] },
        { right.data[ 2 ], up.data[ 2 ], forw.data[ 2 ] },
    };

    return { {
        { rot[ 0 ][ 0 ], rot[ 0 ][ 1 ], rot[ 0 ][ 2 ], position.data[ 0 ] },
        { rot[ 1 ][ 0 ], rot[ 1 ][ 1 ], rot[ 1 ][ 2 ], position.data[ 1 ] },
        { rot[ 2 ][ 0 ], rot[ 2 ][ 1 ], rot[ 2 ][ 2 ], position.data[ 2 ] },
    } };
}
