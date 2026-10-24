// Copyright (c) 2023 Sultim Tsyrendashiev
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

// RgQuaternion is x,y,z,w, as Scene.cpp; must be defined before any GLM include
#ifndef GLM_FORCE_QUAT_DATA_XYZW
    #define GLM_FORCE_QUAT_DATA_XYZW 1
#endif

// Cameras and animations of an imported scene: shared by Scene.cpp and the Metal backend

#include "Camera.h"
#include "GltfImporter.h"
#include "Matrix.h"
#include "Utils.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <optional>

namespace RTGL1
{
namespace sceneanim
{
    inline Camera MakeCamera( const RgCameraInfo& info )
    {
        auto cameraInfo = Camera{
            .aspect      = info.aspect,
            .fovYRadians = info.fovYRadians,
            .cameraNear  = info.cameraNear,
            .cameraFar   = info.cameraFar,
        };
        {
            static_assert( sizeof cameraInfo.projection == 16 * sizeof( float ) );
            static_assert( sizeof cameraInfo.view == 16 * sizeof( float ) );
            static_assert( sizeof cameraInfo.projectionInverse == 16 * sizeof( float ) );
            static_assert( sizeof cameraInfo.viewInverse == 16 * sizeof( float ) );
            static_assert( sizeof info.position == 3 * sizeof( float ) );
            static_assert( sizeof info.right == 3 * sizeof( float ) );
            static_assert( sizeof info.up == 3 * sizeof( float ) );

            Matrix::MakeViewMatrix( cameraInfo.view, info.position, info.right, info.up );
            Matrix::MakeProjectionMatrix( cameraInfo.projection,
                                          cameraInfo.aspect,
                                          cameraInfo.fovYRadians,
                                          cameraInfo.cameraNear,
                                          cameraInfo.cameraFar );
        }
        Matrix::Inverse( cameraInfo.viewInverse, cameraInfo.view );
        Matrix::Inverse( cameraInfo.projectionInverse, cameraInfo.projection );
        return cameraInfo;
    }

    template< typename T >
    T linear_interp( const T& a, const T& b, float t ) = delete;

    template <>
    inline float linear_interp( const float& a, const float& b, float t )
    {
        return std::lerp( a, b, t );
    }

    template<>
    inline RgFloat3D linear_interp( const RgFloat3D& a, const RgFloat3D& b, float t )
    {
        return RgFloat3D{
            std::lerp( a.data[ 0 ], b.data[ 0 ], t ),
            std::lerp( a.data[ 1 ], b.data[ 1 ], t ),
            std::lerp( a.data[ 2 ], b.data[ 2 ], t ),
        };
    }

    template<>
    inline RgQuaternion linear_interp( const RgQuaternion& a, const RgQuaternion& b, float t )
    {
        static_assert( GLM_FORCE_QUAT_DATA_XYZW );
        static_assert( sizeof( glm::quat ) == sizeof( RgQuaternion ) );

        const auto& qa = *reinterpret_cast< const glm::quat* >( a.data );
        const auto& qb = *reinterpret_cast< const glm::quat* >( b.data );

        glm::quat result = glm::slerp( qa, qb, t );
        return *reinterpret_cast< RgQuaternion* >( &result );
    }

    inline RgFloat3D ToRgFloat3D( const glm::vec3& a )
    {
        return { a.x, a.y, a.z };
    }

    inline auto QuatToUpRightVectors( const RgQuaternion& q ) -> std::pair< RgFloat3D, RgFloat3D >
    {
        static_assert( GLM_FORCE_QUAT_DATA_XYZW );
        static_assert( sizeof( glm::quat ) == sizeof( RgQuaternion ) );

        const auto& qa = *reinterpret_cast< const glm::quat* >( q.data );

        glm::mat3 tr = mat3_cast( qa );

        return {
            ToRgFloat3D( tr[ 1 ] ), // up
            ToRgFloat3D( tr[ 0 ] ), // right
        };
    }

    inline RgCameraInfo MakeCameraFromAnim( const RgCameraInfo&                  base,
                                     const std::optional< RgFloat3D >&    pos,
                                     const std::optional< RgQuaternion >& quat,
                                     const std::optional< float >&        fovYRadians )
    {
        auto cam = RgCameraInfo{ base };
        if( pos )
        {
            cam.position = *pos;
        }
        if( quat )
        {
            const auto [ vup, vright ] = QuatToUpRightVectors( *quat );

            cam.up    = vup;
            cam.right = vright;
        }
        if( fovYRadians && ( *fovYRadians ) > 0.01f )
        {
            cam.fovYRadians = *fovYRadians;
        }
        return cam;
    }

    template< typename T >
    inline std::optional< T > SampleAnimationChannel( const AnimationChannel< T >& chan, float t )
    {
        if( chan.frames.empty() )
        {
            return std::nullopt;
        }

        for( size_t i = 0; i < chan.frames.size() - 1; i++ )
        {
            const AnimationFrame< T >& a = chan.frames[ i ];
            const AnimationFrame< T >& b = chan.frames[ i + 1 ];

            const float t0 = a.seconds;
            const float t1 = b.seconds;

            if( t0 <= t && t <= t1 )
            {
                if( a.interpolation == ANIMATION_INTERPOLATION_STEP )
                {
                    return a.value;
                }
                if( b.interpolation == ANIMATION_INTERPOLATION_STEP )
                {
                    return b.value;
                }

                // TODO: cubic
                assert( a.interpolation == ANIMATION_INTERPOLATION_LINEAR );

                const float factor = t0 < t1 ? ( t - t0 ) / ( t1 - t0 ) : 0;

                return linear_interp( a.value, b.value, factor );
            }
        }

        if( t < chan.frames.front().seconds )
        {
            return chan.frames.front().value;
        }
        return chan.frames.back().value;
    }

    inline RgCameraInfo SampleAnimation( const AnimationData& anim, const RgCameraInfo& base, float t )
    {
        auto pos  = SampleAnimationChannel( anim.position, t );
        auto quat = SampleAnimationChannel( anim.quaternion, t );
        auto fovy = SampleAnimationChannel( anim.fovYRadians, t );

        return MakeCameraFromAnim( base, pos, quat, fovy );
    }

    inline RgTransform SampleAnimationObj( const AnimationData& anim, const RgTransform& base, float t )
    {
        static auto l_column_length = []( const RgTransform& tr, int column ) {
            return Utils::Length( RgFloat3D{
                tr.matrix[ 0 ][ column ],
                tr.matrix[ 1 ][ column ],
                tr.matrix[ 2 ][ column ],
            } );
        };

        const auto pos  = SampleAnimationChannel( anim.position, t );
        const auto quat = SampleAnimationChannel( anim.quaternion, t );

        const float scale[] = {
            l_column_length( base, 0 ),
            l_column_length( base, 1 ),
            l_column_length( base, 2 ),
        };

        auto r = RgTransform{ base };

        if( pos )
        {
            r.matrix[ 0 ][ 3 ] = pos->data[ 0 ];
            r.matrix[ 1 ][ 3 ] = pos->data[ 1 ];
            r.matrix[ 2 ][ 3 ] = pos->data[ 2 ];
        }

        if( quat )
        {
            const auto [ vup, vright ] = QuatToUpRightVectors( *quat );
            const auto vforward        = Utils::Cross( vright, vup );

            // clang-format off
            r.matrix[ 0 ][ 0 ] = vright.data[ 0 ]; r.matrix[ 0 ][ 1 ] = vup.data[ 0 ]; r.matrix[ 0 ][ 2 ] = vforward.data[ 0 ];
            r.matrix[ 1 ][ 0 ] = vright.data[ 1 ]; r.matrix[ 1 ][ 1 ] = vup.data[ 1 ]; r.matrix[ 1 ][ 2 ] = vforward.data[ 1 ];
            r.matrix[ 2 ][ 0 ] = vright.data[ 2 ]; r.matrix[ 2 ][ 1 ] = vup.data[ 2 ]; r.matrix[ 2 ][ 2 ] = vforward.data[ 2 ];
            // clang-format on

            // do not lose the original scale
            // clang-format off
            r.matrix[ 0 ][ 0 ] *= scale[ 0 ]; r.matrix[ 0 ][ 1 ] *= scale[ 1 ]; r.matrix[ 0 ][ 2 ] *= scale[ 2 ];
            r.matrix[ 1 ][ 0 ] *= scale[ 0 ]; r.matrix[ 1 ][ 1 ] *= scale[ 1 ]; r.matrix[ 1 ][ 2 ] *= scale[ 2 ];
            r.matrix[ 2 ][ 0 ] *= scale[ 0 ]; r.matrix[ 2 ][ 1 ] *= scale[ 1 ]; r.matrix[ 2 ][ 2 ] *= scale[ 2 ];
            // clang-format on
        }

        return r;
    }
}
}
