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

// First light of the Metal backend: primary rays that show the albedo.
// Will be replaced by the ports of RTGL1's raygen shaders (RaygenPrimary.inl etc).

#include "ShaderCommonMSL.h"
#include "MetalBindings.h"

#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

// MetalTextureManager::TableEntry
struct TextureRef
{
    texture2d< float > tex;
    sampler            smp;
};

constexpr sampler s_blit( filter::linear, address::clamp_to_edge );
constexpr sampler s_sky( filter::linear, address::clamp_to_edge );

// getSkyAlbedo in RaygenCommon.h
// TODO: SKY_TYPE_CUBEMAP, adjustSky
static float3 getSkyAlbedo( constant ShGlobalUniform& globalUniform, texturecube< float > skyCubemap, float3 direction )
{
    if( globalUniform.skyType == SKY_TYPE_RASTERIZED_GEOMETRY )
    {
        return skyCubemap.sample( s_sky, direction ).rgb;
    }
    return globalUniform.skyColorDefault.xyz;
}

static float4 unpackUintColor( uint c )
{
    return float4( c & 255u, ( c >> 8 ) & 255u, ( c >> 16 ) & 255u, c >> 24 ) / 255.0;
}

// The same as getRayDir in ShaderCommonGLSLFunc.h
static float3 getRayDir( constant ShGlobalUniform& globalUniform, float2 inUV )
{
    inUV = inUV * 2.0 - 1.0;

    float4 target   = globalUniform.invProjection * float4( inUV.x, inUV.y, 1, 1 );
    float3 localDir = abs( target.w ) < 0.001 ? target.xyz : target.xyz / target.w;

    float4 rayDir = globalUniform.invView * float4( normalize( localDir ), 0 );

    return rayDir.xyz;
}

struct SceneRefs
{
    device const ShVertex*           vertices;
    device const uint*               indices;
    device const ShGeometryInstance* geometries;
    device const TextureRef*         textures;
};

struct Triangle
{
    float3 positions[ 3 ];
    float2 texCoords[ 3 ];
    float4 colors[ 3 ];
};

static Triangle getTriangle( thread const SceneRefs& s, uint geomIndex, uint primIndex )
{
    device const ShGeometryInstance& g = s.geometries[ geomIndex ];

    Triangle t;
    for( uint k = 0; k < 3; k++ )
    {
        uint                   i = s.indices[ g.baseIndexIndex + primIndex * 3 + k ];
        device const ShVertex& v = s.vertices[ g.baseVertexIndex + i ];

        float4 p = float4( float3( v.position ), 1.0 );
        // model rows, as in VertexData.inl
        t.positions[ k ] = float3( dot( g.model_0, p ), dot( g.model_1, p ), dot( g.model_2, p ) );
        t.texCoords[ k ] = v.texCoord;
        t.colors[ k ]    = unpackUintColor( v.color );
    }
    return t;
}

template< typename T, size_t N >
static T interpolate( thread const T ( &v )[ N ], float3 bary )
{
    return v[ 0 ] * bary.x + v[ 1 ] * bary.y + v[ 2 ] * bary.z;
}

static float3 toBary( float2 uv )
{
    return float3( 1.0 - uv.x - uv.y, uv.x, uv.y );
}

static float4 getAlbedo( thread const SceneRefs& s, uint geomIndex, uint primIndex, float2 baryUV )
{
    device const ShGeometryInstance& g = s.geometries[ geomIndex ];

    Triangle t    = getTriangle( s, geomIndex, primIndex );
    float3   bary = toBary( baryUV );

    // TODO: ray cones for the mip level
    device const TextureRef& tr  = s.textures[ g.texture_base ];
    float4                   tex = tr.tex.sample( tr.smp, interpolate( t.texCoords, bary ), level( 0 ) );

    return tex * unpackUintColor( g.colorFactor_base ) * interpolate( t.colors, bary );
}

kernel void PrimaryRays( constant ShGlobalUniform&      globalUniform [[ buffer( MTL_BINDING_GLOBAL_UNIFORM ) ]],
                         instance_acceleration_structure tlas          [[ buffer( MTL_BINDING_TLAS ) ]],
                         device const ShVertex*           vertices      [[ buffer( MTL_BINDING_VERTICES ) ]],
                         device const uint*               indices       [[ buffer( MTL_BINDING_INDICES ) ]],
                         device const ShGeometryInstance* geometries    [[ buffer( MTL_BINDING_GEOMETRIES ) ]],
                         device const TextureRef*         textures      [[ buffer( MTL_BINDING_TEXTURES ) ]],
                         device const MetalInstanceInfo*  instanceInfos [[ buffer( MTL_BINDING_INSTANCE_INFOS ) ]],
                         texture2d< float, access::write > outColor     [[ texture( 0 ) ]],
                         texturecube< float >              skyCubemap   [[ texture( 1 ) ]],
                         uint2                            pix           [[ thread_position_in_grid ]] )
{
    if( pix.x >= outColor.get_width() || pix.y >= outColor.get_height() )
    {
        return;
    }

    const float2 size = float2( globalUniform.renderWidth, globalUniform.renderHeight );
    const float2 inUV = ( float2( pix ) + 0.5 ) / size;

    const float3 origin    = globalUniform.cameraPosition.xyz;
    const float3 direction = getRayDir( globalUniform, inUV );

    float3 color = getSkyAlbedo( globalUniform, skyCubemap, direction );

    {
        const SceneRefs s = { vertices, indices, geometries, textures };

        ray r( origin, direction, max( globalUniform.primaryRayMinDist, 0.001f ), globalUniform.rayLength );

        // Non-opaque (alpha tested) candidates are tested here, opaque ones are committed by traversal.
        // TODO: as in Q2RTXM, one opaque traversal + a query limited to alpha tested geometry
        intersection_params params;
        params.assume_geometry_type( geometry_type::triangle );
        params.set_triangle_cull_mode( triangle_cull_mode::none );

        // getPrimaryVisibilityCullMask in RaygenCommon.h
        // TODO: globalUniform.rayCullMaskWorld
        const uint cullMask = INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_WORLD_1 | INSTANCE_MASK_WORLD_2 |
                              INSTANCE_MASK_REFRACT | INSTANCE_MASK_FIRST_PERSON;

        intersection_query< instancing, triangle_data > q;
        q.reset( r, tlas, cullMask, params );
        while( q.next() )
        {
            if( q.get_candidate_intersection_type() == intersection_type::triangle )
            {
                float4 a = getAlbedo( s,
                                      instanceInfos[ q.get_candidate_instance_id() ].geometryBase + q.get_candidate_geometry_id(),
                                      q.get_candidate_primitive_id(),
                                      q.get_candidate_triangle_barycentric_coord() );
                if( a.a >= 0.5 )
                {
                    q.commit_triangle_intersection();
                }
            }
        }

        // doesPayloadContainHitInfo in RaygenCommon.h: sky geometry shows the sky
        if( q.get_committed_intersection_type() == intersection_type::triangle &&
            ( instanceInfos[ q.get_committed_instance_id() ].customIndex & INSTANCE_CUSTOM_INDEX_FLAG_SKY ) == 0 )
        {
            const uint   geomIndex = instanceInfos[ q.get_committed_instance_id() ].geometryBase + q.get_committed_geometry_id();
            const uint   primIndex = q.get_committed_primitive_id();
            const float2 baryUV    = q.get_committed_triangle_barycentric_coord();

            const float4 albedo = getAlbedo( s, geomIndex, primIndex, baryUV );

            // TODO: replace with lighting, this only makes the shapes readable
            Triangle     t      = getTriangle( s, geomIndex, primIndex );
            const float3 normal = normalize( cross( t.positions[ 1 ] - t.positions[ 0 ],
                                                    t.positions[ 2 ] - t.positions[ 0 ] ) );
            const float  facing = abs( dot( normal, direction ) );

            color = albedo.rgb * ( 0.25 + 0.75 * facing );
        }
    }

    outColor.write( float4( color, 1.0 ), pix );
}

// When there's no ray traced geometry (menus, loading screens)
kernel void SkyOnly( constant ShGlobalUniform&         globalUniform [[ buffer( MTL_BINDING_GLOBAL_UNIFORM ) ]],
                     texture2d< float, access::write > outColor      [[ texture( 0 ) ]],
                     texturecube< float >              skyCubemap    [[ texture( 1 ) ]],
                     uint2                             pix           [[ thread_position_in_grid ]] )
{
    if( pix.x >= outColor.get_width() || pix.y >= outColor.get_height() )
    {
        return;
    }
    const float2 inUV = ( float2( pix ) + 0.5 ) / float2( globalUniform.renderWidth, globalUniform.renderHeight );
    outColor.write( float4( getSkyAlbedo( globalUniform, skyCubemap, getRayDir( globalUniform, inUV ) ), 1.0 ), pix );
}

// Rasterizer::DrawSkyToAlbedo: RTGL1's primary rays read the sky color from framebufAlbedo.
// The sky cubemap was rendered from the sky viewer position, so it's sampled by the view direction.
kernel void SkyToAlbedo( constant ShGlobalUniform&         globalUniform [[ buffer( MTL_BINDING_GLOBAL_UNIFORM ) ]],
                         texture2d< float, access::write > albedo        [[ texture( 0 ) ]],
                         texturecube< float >              skyCubemap    [[ texture( 1 ) ]],
                         uint2                             pix           [[ thread_position_in_grid ]] )
{
    if( pix.x >= albedo.get_width() || pix.y >= albedo.get_height() )
    {
        return;
    }
    const float2 inUV = ( float2( pix ) + 0.5 ) / float2( globalUniform.renderWidth, globalUniform.renderHeight );
    albedo.write( float4( skyCubemap.sample( s_sky, getRayDir( globalUniform, inUV ) ).rgb, 0.0 ), pix );
}

// Framebuffers::BlitForEffects: 'src' is scaled to 'dst',
// srcScale is the part of 'src' that has the image (render size / texture size)
kernel void BlitScaled( texture2d< float, access::sample > src      [[ texture( 0 ) ]],
                        texture2d< float, access::write >  dst      [[ texture( 1 ) ]],
                        sampler                            smp      [[ sampler( 0 ) ]],
                        constant float2&                   srcScale [[ buffer( 0 ) ]],
                        uint2                              pix      [[ thread_position_in_grid ]] )
{
    if( pix.x >= dst.get_width() || pix.y >= dst.get_height() )
    {
        return;
    }
    const float2 uv = ( float2( pix ) + 0.5 ) / float2( dst.get_width(), dst.get_height() );
    dst.write( src.sample( smp, uv * srcScale, level( 0 ) ), pix );
}

struct BlitVertexOut
{
    float4 position [[ position ]];
    float2 uv;
};

vertex BlitVertexOut BlitVertex( uint vid [[ vertex_id ]] )
{
    // fullscreen triangle
    const float2 uv = float2( ( vid << 1 ) & 2, vid & 2 );

    BlitVertexOut o;
    o.position = float4( uv * float2( 2, -2 ) + float2( -1, 1 ), 0, 1 );
    o.uv       = uv;
    return o;
}

// 'scale': 1, or for debug views (RTGL1_METAL_SHOW_SCALE) of the absolute values
fragment float4 BlitFragment( BlitVertexOut      in [[ stage_in ]],
                              texture2d< float > src [[ texture( 0 ) ]],
                              constant float&    scale [[ buffer( 0 ) ]] )
{
    const float4 c = src.sample( s_blit, in.uv );
    return scale == 1.0f ? c : float4( abs( c.rgb ) * scale, 1.0f );
}

// MetalFluid: the raster pass' depth to framebufDepthFluid (Vulkan aliases their memory)
kernel void FluidCopyDepth( depth2d< float >                  src [[ texture( 0 ) ]],
                            texture2d< float, access::write > dst [[ texture( 1 ) ]],
                            uint2                             id [[ thread_position_in_grid ]] )
{
    if( id.x >= dst.get_width() || id.y >= dst.get_height() )
    {
        return;
    }
    dst.write( float4( src.read( id ) ), id );
}

// Frame interpolation: the UI texture (premultiplied: rgb + ( 1 - a ) * image without UI = image with UI),
// from the images with and without UI, and the UI's coverage
kernel void FrameInterpolationUI( texture2d< float >                  withUI [[ texture( 0 ) ]],
                                  texture2d< float >                  withoutUI [[ texture( 1 ) ]],
                                  texture2d< float >                  coverage [[ texture( 2 ) ]],
                                  texture2d< float, access::write >   outUI [[ texture( 3 ) ]],
                                  uint2                               id [[ thread_position_in_grid ]] )
{
    if( id.x >= outUI.get_width() || id.y >= outUI.get_height() )
    {
        return;
    }
    const float  a   = saturate( coverage.read( id ).a );
    const float3 rgb = max( withUI.read( id ).rgb - ( 1.0f - a ) * withoutUI.read( id ).rgb, 0.0f );
    outUI.write( float4( rgb, a ), id );
}

// Rasterized to swapchain (HUD, menus, console), as RsRasterizer.vert / RsWorld.inl

struct RasterVertexOut
{
    float4 position [[ position ]];
    float4 color;
    float2 texCoord;
};

vertex RasterVertexOut RasterSwapchainVertex( uint                          vid [[ vertex_id ]],
                                              device const ShVertex*        vertices [[ buffer( MTL_RASTER_BINDING_VERTICES ) ]],
                                              constant MetalRasterConstants& c       [[ buffer( MTL_RASTER_BINDING_CONSTANTS ) ]] )
{
    device const ShVertex& v = vertices[ vid ];

    const float4x4 mvp = float4x4( float4( c.mvp[ 0 ], c.mvp[ 1 ], c.mvp[ 2 ], c.mvp[ 3 ] ),
                                   float4( c.mvp[ 4 ], c.mvp[ 5 ], c.mvp[ 6 ], c.mvp[ 7 ] ),
                                   float4( c.mvp[ 8 ], c.mvp[ 9 ], c.mvp[ 10 ], c.mvp[ 11 ] ),
                                   float4( c.mvp[ 12 ], c.mvp[ 13 ], c.mvp[ 14 ], c.mvp[ 15 ] ) );

    float4 color = unpackUintColor( v.color );
    if( c.vertexColorGamma )
    {
        color.rgb = pow( color.rgb, 2.2 );
    }

    RasterVertexOut o;
    o.position = mvp * float4( float3( v.position ), 1.0 );
    // Vulkan's clip space Y points down, Metal's up
    o.position.y = -o.position.y;
    o.color      = color;
    o.texCoord   = v.texCoord;
    return o;
}

fragment float4 RasterSwapchainFragment( RasterVertexOut                in [[ stage_in ]],
                                         constant MetalRasterConstants& c        [[ buffer( MTL_RASTER_BINDING_CONSTANTS ) ]],
                                         device const TextureRef*       textures [[ buffer( MTL_RASTER_BINDING_TEXTURES ) ]] )
{
    device const TextureRef& tr = textures[ c.textureIndex ];

    float4 color = tr.tex.sample( tr.smp, in.texCoord ) * unpackUintColor( c.color ) * in.color;

    if( c.alphaTest && color.a < 0.5 )
    {
        discard_fragment();
    }
    return color;
}
