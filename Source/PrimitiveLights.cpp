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

#include "PrimitiveLights.h"

#include "DrawFrameInfo.h"
#include "Utils.h"

#include <optional>

// Moved from GltfExporter.cpp, to not depend on the exporter

namespace
{

constexpr RgFloat3D operator+( const RgFloat3D& a, const RgFloat3D& b )
{
    return {
        a.data[ 0 ] + b.data[ 0 ],
        a.data[ 1 ] + b.data[ 1 ],
        a.data[ 2 ] + b.data[ 2 ],
    };
}
constexpr RgFloat3D operator*( const RgFloat3D& c, float f )
{
    return {
        f * c.data[ 0 ],
        f * c.data[ 1 ],
        f * c.data[ 2 ],
    };
}

#define POLYLIGHT_AS_SPHERE 1

RgFloat3D GetCenter( const RgFloat3D ( &positions )[ 3 ] )
{
    RgFloat3D center = { 0, 0, 0 };
    for( const auto& v : positions )
    {
        center = center + v;
    }
    return center * ( 1.0f / 3.0f );
}

std::optional< RgFloat3D > TryGetNormal( const RgFloat3D ( &positions )[ 3 ] )
{
    float     area;
    RgFloat3D normal;
    if( !RTGL1::Utils::GetNormalAndArea( positions, normal, area ) )
    {
        return std::nullopt;
    }
    return normal;
}

}

void RTGL1::MakeLightsForPrimitive( const RgMeshInfo&                      mesh,
                             const RgMeshPrimitiveInfo&             prim,
                             const RgMeshPrimitiveAttachedLightEXT& lightInfo,
                             float                                  oneGameUnitInMeters,
                             std::vector< RTGL1::PositionNormal >&  initial,
                             std::vector< RTGL1::AnyLightEXT >&     result )
{
    assert( initial.empty() );
    assert( result.empty() );

    constexpr auto toFloat3 = []( const float( &ptr )[ 3 ] ) {
        return RgFloat3D{ RG_ACCESS_VEC3( ptr ) };
    };
    using RTGL1::Utils::ApplyTransform;

    assert( prim.indexCount % 3 == 0 );
    if( prim.indexCount / 3 > 1024 )
    {
        RTGL1::debug::Warning(
            "The amount of triangles on a primitive (ID {}-{}, material name: {}) "
            "with attached light is too high ()",
            prim.indexCount / 3,
            mesh.uniqueObjectID,
            prim.primitiveIndexInMesh,
            RTGL1::Utils::SafeCstr( prim.pTextureName ) );
    }

#if POLYLIGHT_AS_SPHERE
    using PositionNormal = RTGL1::PositionNormal;

    auto merge = []( const std::optional< PositionNormal >& a,
                     const std::optional< PositionNormal >& b ) -> std::optional< PositionNormal > {
        if( a && b )
        {
            return PositionNormal{
                .position = ( a->position + b->position ) * 0.5f,
                .normal   = RTGL1::Utils::Normalize( a->normal + b->normal ),
            };
        }
        if( a && !b )
        {
            return a;
        }
        if( !a && b )
        {
            return b;
        }
        return std::nullopt;
    };

    struct Tri
    {
        RgFloat3D v[ 3 ];
    };
    auto haveCommonEdge = []( const Tri& a, const Tri& b ) {
        auto isSameEdge = []( const RgFloat3D& v0,
                              const RgFloat3D& v1,
                              const RgFloat3D& u0,
                              const RgFloat3D& u1 ) {
            return ( RTGL1::Utils::AreAlmostSame( v0, u0 ) &&
                     RTGL1::Utils::AreAlmostSame( v1, u1 ) ) ||
                   ( RTGL1::Utils::AreAlmostSame( v0, u1 ) &&
                     RTGL1::Utils::AreAlmostSame( v1, u0 ) );
        };
        constexpr int edges[][ 2 ] = {
            { 0, 1 },
            { 1, 2 },
            { 2, 0 },
        };
        for( const auto& [ av0, av1 ] : edges )
        {
            for( const auto& [ bv0, bv1 ] : edges )
            {
                if( isSameEdge( a.v[ av0 ], a.v[ av1 ], b.v[ bv0 ], b.v[ bv1 ] ) )
                {
                    return true;
                }
            }
        }
        return false;
    };
    std::optional< Tri >            prev;
    std::optional< PositionNormal > accum;

    auto flushAccum = [ &initial, &accum ]() {
        if( accum )
        {
            initial.push_back( *accum );
            accum = std::nullopt;
        }
    };

    for( uint32_t tri = 0; tri < prim.indexCount / 3; tri++ )
    {
        Tri local = {
            toFloat3( prim.pVertices[ prim.pIndices[ tri * 3 + 0 ] ].position ),
            toFloat3( prim.pVertices[ prim.pIndices[ tri * 3 + 1 ] ].position ),
            toFloat3( prim.pVertices[ prim.pIndices[ tri * 3 + 2 ] ].position ),
        };

        Tri global = {
            ApplyTransform( mesh.transform, local.v[ 0 ] ),
            ApplyTransform( mesh.transform, local.v[ 1 ] ),
            ApplyTransform( mesh.transform, local.v[ 2 ] ),
        };

        if( auto n = TryGetNormal( global.v ) )
        {
            if( prev )
            {
                if( !haveCommonEdge( global, *prev ) )
                {
                    flushAccum();
                }
            }
            prev = global;

            accum = merge( accum,
                           PositionNormal{
                               .position = GetCenter( global.v ),
                               .normal   = *n,
                           } );
        }
    }
    flushAccum();

    for( const auto& [ position, normal ] : initial )
    {
        float offset = 0.1f / oneGameUnitInMeters;

        result.emplace_back( RgLightSpotEXT{
            .sType      = RG_STRUCTURE_TYPE_LIGHT_SPOT_EXT,
            .pNext      = nullptr,
            .color      = lightInfo.color,
            .intensity  = lightInfo.intensity,
            .position   = position + normal * offset,
            .direction  = normal,
            .radius     = 0.1f, /* ignored */
            .angleOuter = RTGL1::Utils::DegToRad( 89 ),
            .angleInner = RTGL1::Utils::DegToRad( 75 ),
        } );
    }
#endif
}

void RTGL1::GltfExporter::MakeLightsForPrimitiveDynamic( const RgMeshInfo&          mesh,
                                                         const RgMeshPrimitiveInfo& primitive,
                                                         float oneGameUnitInMeters,
                                                         std::vector< PositionNormal >& tempStorage,
                                                         std::vector< AnyLightEXT >& resultStorage )
{
    assert( tempStorage.empty() );
    assert( resultStorage.empty() );

    if( auto* attachedLight = pnext::find< RgMeshPrimitiveAttachedLightEXT >( &primitive ) )
    {
        RTGL1::MakeLightsForPrimitive(
            mesh, primitive, *attachedLight, oneGameUnitInMeters, tempStorage, resultStorage );
    }
}
