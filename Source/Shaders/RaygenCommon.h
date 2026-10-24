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

#ifndef RAYGEN_COMMON_H_
#define RAYGEN_COMMON_H_

#extension GL_EXT_ray_query : require
#extension GL_EXT_control_flow_attributes : require

// The path tracer's ray generation stages run as compute shaders that cast rays with ray
// queries (Metal's intersector model) instead of a ray tracing pipeline with a shader binding
// table: Metal has no separate ray tracing pipeline. One invocation is one former launch ID;
// the miss, closest-hit and any-hit shaders are folded into traceRayQuery* below.
layout( local_size_x = 8, local_size_y = 8, local_size_z = 1 ) in;

#define gl_LaunchIDEXT gl_GlobalInvocationID

// the dispatch is rounded up to whole workgroups: invocations past the launch size return
#define RAYGEN_RETURN_IF_OUTSIDE( w, h, d )                                     \
    if( gl_GlobalInvocationID.x >= uint( w ) || gl_GlobalInvocationID.y >= uint( h ) || \
        gl_GlobalInvocationID.z >= uint( d ) )                                  \
    {                                                                           \
        return;                                                                 \
    }



#include "ShaderCommonGLSLFunc.h"



#if !defined(DESC_SET_TLAS) || \
    !defined(DESC_SET_GLOBAL_UNIFORM) || \
    !defined(DESC_SET_VERTEX_DATA) || \
    !defined(DESC_SET_TEXTURES) || \
    !defined(DESC_SET_RANDOM) || \
    !defined(DESC_SET_LIGHT_SOURCES)
        #error Descriptor set indices must be set!
#endif

#define LIGHT_SAMPLE_METHOD_NONE 0
#define LIGHT_SAMPLE_METHOD_DIRECT 1
#define LIGHT_SAMPLE_METHOD_INDIR 2
#define LIGHT_SAMPLE_METHOD_GRADIENTS 3
#define LIGHT_SAMPLE_METHOD_INITIAL 4
#define LIGHT_SAMPLE_METHOD_VOLUME 5
#if !defined(LIGHT_SAMPLE_METHOD)
    #error Light sampling method must be defined
#endif



#include "Surface.inl"
#include "Light.h"
#include "LightGrid.h"
#include "Media.h"
#include "RayCone.h"

#define GET_TARGET_PDF targetPdfForLightSample
#include "Reservoir.h"

#define HITINFO_INL_PRIM
    #include "HitInfo.inl"
#undef HITINFO_INL_PRIM

#define HITINFO_INL_RFL
    #include "HitInfo.inl"
#undef HITINFO_INL_RFL

#define HITINFO_INL_INDIR
    #include "HitInfo.inl"
#undef HITINFO_INL_INDIR



layout(set = DESC_SET_TLAS, binding = BINDING_ACCELERATION_STRUCTURE_MAIN)   uniform accelerationStructureEXT topLevelAS;

#ifdef DESC_SET_CUBEMAPS
#ifdef METAL
// see globalTextures_Get
#define METAL_DESC_SET_CUBEMAP_SAMPLERS 13
layout(set = DESC_SET_CUBEMAPS, binding = BINDING_CUBEMAPS) uniform textureCube globalCubemaps_Images[];
layout(set = METAL_DESC_SET_CUBEMAP_SAMPLERS, binding = 0) uniform sampler globalCubemaps_Samplers[];
#define globalCubemaps_Get( index )                                 \
    samplerCube( globalCubemaps_Images[ nonuniformEXT( index ) ], \
                 globalCubemaps_Samplers[ nonuniformEXT( index ) ] )
#else
layout(set = DESC_SET_CUBEMAPS, binding = BINDING_CUBEMAPS) uniform samplerCube globalCubemaps[];
#define globalCubemaps_Get( index ) globalCubemaps[ nonuniformEXT( index ) ]
#endif
#endif

#ifdef DESC_SET_RENDER_CUBEMAP
layout(set = DESC_SET_RENDER_CUBEMAP, binding = BINDING_RENDER_CUBEMAP) uniform samplerCube renderCubemap;
#endif

#ifdef DESC_SET_PORTALS
layout(set = DESC_SET_PORTALS, binding = BINDING_PORTAL_INSTANCES) uniform readonly PortalInstances_BT
{
    ShPortalInstance g_portals[PORTAL_MAX_COUNT];
};
#endif


// the result of the last traceRayQuery-based trace (formerly the ray payload)
ShPayload g_payload;



uint getPrimaryVisibilityCullMask()
{
    return globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT | INSTANCE_MASK_FIRST_PERSON;
}

uint getReflectionRefractionCullMask(uint surfInstCustomIndex, uint geometryInstanceFlags, bool isRefraction)
{
    uint world = globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT;

    if( ( geometryInstanceFlags & GEOM_INST_FLAG_IGNORE_REFRACT_AFTER ) != 0 )
    {
        // ignore refract geometry if requested
        world = world & ( ~INSTANCE_MASK_REFRACT );

        // if it's also a first-person geometry, then ignore everything first-person
        if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
        {
            return world;
        }
    }

    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // ignore first-person viewer -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    
    return isRefraction ? 
        // no first-person viewer in refractions
        world | INSTANCE_MASK_FIRST_PERSON :
        // no first-person in reflections
        world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
}

uint getShadowCullMask(uint surfInstCustomIndex)
{
    uint world = globalUniform.rayCullMaskWorld_Shadow;

    // Doom64-RT: a surface flagged IGNORE_SHADOW_PROXY does not see shadow-only
    // geometry at all -- set on alpha-tested instances, i.e. sprites.
    //
    // A sprite's shadow proxies are planes through its own axis, so a proxy that
    // is not edge-on to the light shadows the half of its own billboard behind
    // it; and with a light along the sprite's normal (the flashlight, which sits
    // at the camera the billboard is facing) the perpendicular proxy projects to
    // a line straight down the sprite's middle. Removing the proxies from the
    // sprite's own shadow test removes both, and costs only that one actor's
    // proxy no longer darkens another actor -- which Doom's flat-lit sprites do
    // not show anyway.
    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_IGNORE_SHADOW_PROXY) != 0)
    {
        world &= ~INSTANCE_MASK_RESERVED_0;
    }

    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // no first-person viewer shadows -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    else if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER) != 0)
    {
        // no first-person shadows -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
    else
    {
        // no first-person shadows -- on world
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
}

uint getIndirectIlluminationCullMask(uint surfInstCustomIndex)
{
    const uint world = globalUniform.rayCullMaskWorld;
    
    if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON) != 0)
    {
        // no first-person viewer indirect illumination -- on first-person
        return world | INSTANCE_MASK_FIRST_PERSON;
    }
    else if ((surfInstCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER) != 0)
    {
        // no first-person indirect illumination -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
    else
    {
        // no first-person indirect illumination -- on first-person viewer
        return world | INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }
}



uint getAdditionalRayFlags()
{
#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
    return 0;
#else
    return globalUniform.rayCullBackFaces != 0 ? gl_RayFlagsCullFrontFacingTrianglesEXT : 0;
#endif
}



bool doesPayloadContainHitInfo(const ShPayload p)
{
    if (p.instIdAndIndex == UINT32_MAX || p.geomAndPrimIndex == UINT32_MAX)
    {
        return false;
    }

    int instanceId, instanceCustomIndex;
    unpackInstanceIdAndCustomIndex(p.instIdAndIndex, instanceId, instanceCustomIndex);

    if ((instanceCustomIndex & INSTANCE_CUSTOM_INDEX_FLAG_SKY) != 0)
    {
        return false;
    }

    return true;
}

void resetPayload()
{
    g_payload.baryCoords = vec2(0.0);
    g_payload.instIdAndIndex = UINT32_MAX;
    g_payload.geomAndPrimIndex = UINT32_MAX;
}

// Former any-hit shaders (RtAlphaTest.rahit, RtAlphaTestMedia.rahit). Only alpha-tested
// instances are non-opaque (ASManager sets FORCE_NO_OPAQUE on them, FORCE_OPAQUE on the
// rest), so only they produce candidate hits, and each candidate runs the alpha test.
// 'media': the ray asks what shadows the fog (the former SBT_RAY_OFFSET_MEDIA hit groups):
// sprites don't take part.
#define RQ_ALPHA_THRESHOLD 0.5

// The alpha test of a candidate triangle. Separate from the ray query, so the macOS Metal
// path can run it as an intersection function of Metal's intersector (MetalShaderGen.cpp
// finds it by name and calls it from there).
bool rtAlphaTestAccepts( int instanceId, uint primitiveId, vec2 b, bool media )
{
    if( media && ( geometryInstances[ instanceId ].flags & GEOM_INST_FLAG_SPRITE ) != 0 )
    {
        return false;
    }

    // Only the layer 0 texture coordinates, as the former any-hit shader read them
    // (getTriangleTexInfo): this runs for every candidate along a ray, and the full
    // getTriangle -- all layers, normals, world and previous-frame positions -- made it
    // the most expensive part of tracing through alpha-tested geometry.
    const uint baseVertex  = geometryInstances[ instanceId ].baseVertexIndex;
    const uint baseIndex   = geometryInstances[ instanceId ].baseIndexIndex;

    const vec3 baryCoords = vec3( 1.0 - b.x - b.y, b.x, b.y );

    vec2 texCoord;
    if( ( geometryInstances[ instanceId ].flags & GEOM_INST_FLAG_IS_DYNAMIC ) != 0 )
    {
        const uvec3 vi = getVertIndicesDynamic( baseVertex, baseIndex, primitiveId );
        texCoord       = g_dynamicVertices[ vi[ 0 ] ].texCoord * baryCoords[ 0 ] +
                   g_dynamicVertices[ vi[ 1 ] ].texCoord * baryCoords[ 1 ] +
                   g_dynamicVertices[ vi[ 2 ] ].texCoord * baryCoords[ 2 ];
    }
    else
    {
        const uvec3 vi = getVertIndicesStatic( baseVertex, baseIndex, primitiveId );
        texCoord       = g_staticVertices[ vi[ 0 ] ].texCoord * baryCoords[ 0 ] +
                   g_staticVertices[ vi[ 1 ] ].texCoord * baryCoords[ 1 ] +
                   g_staticVertices[ vi[ 2 ] ].texCoord * baryCoords[ 2 ];
    }

    // check only first material's albedo
    const vec4 color = getTextureSample( geometryInstances[ instanceId ].texture_base, texCoord ) *
                       unpackUintColor( geometryInstances[ instanceId ].colorFactor_base );

    return ( color.r + color.g + color.b ) / 3 * color.a + color.a >= RQ_ALPHA_THRESHOLD;
}

// The same as gl_InstanceID and gl_InstanceCustomIndexEXT in hit shaders.
// RAY_QUERY_GROUPED_INSTANCES (the native Metal backend, MetalScene): a TLAS instance contains
// many geometries (one per primitive), its custom index is
// ( index of the first ShGeometryInstance << 4 ) | INSTANCE_CUSTOM_INDEX_FLAG_*
// ('committed' of rayQueryGet* must be a constant expression, hence macros)
#ifdef RAY_QUERY_GROUPED_INSTANCES
    #define rqGetInstanceId( q, committed )                                              \
        ( int( uint( rayQueryGetIntersectionInstanceCustomIndexEXT( q, committed ) ) >> 4 ) + \
          rayQueryGetIntersectionGeometryIndexEXT( q, committed ) )
    #define rqGetInstanceCustomIndex( q, committed ) \
        int( uint( rayQueryGetIntersectionInstanceCustomIndexEXT( q, committed ) ) & 0xF )
#else
    #define rqGetInstanceId( q, committed )          rayQueryGetIntersectionInstanceIdEXT( q, committed )
    #define rqGetInstanceCustomIndex( q, committed ) rayQueryGetIntersectionInstanceCustomIndexEXT( q, committed )
#endif

// One ray query object for all traces: on Metal, each one is a large intersection_query
// that lives for the whole kernel, so several of them increase register pressure
rayQueryEXT g_rayQuery;

bool rqIsCandidateAccepted( rayQueryEXT rq, bool media )
{
    return rtAlphaTestAccepts( rqGetInstanceId( rq, false ),
                               uint( rayQueryGetIntersectionPrimitiveIndexEXT( rq, false ) ),
                               rayQueryGetIntersectionBarycentricsEXT( rq, false ),
                               media );
}

// Former closest-hit (RtClsOpaque.rchit) and miss (RtMiss.rmiss) shaders: the nearest
// accepted triangle, or an empty payload.
ShPayload traceRayQuery( uint rayFlags, uint cullMask, vec3 origin, float tMin, vec3 direction, float tMax )
{
    ShPayload p;
    p.baryCoords       = vec2( 0.0 );
    p.instIdAndIndex   = UINT32_MAX;
    p.geomAndPrimIndex = UINT32_MAX;

    #define rq g_rayQuery
    rayQueryInitializeEXT( rq, topLevelAS, rayFlags, cullMask & 0xFF, origin, tMin, direction, tMax );

    while( rayQueryProceedEXT( rq ) )
    {
        if( rayQueryGetIntersectionTypeEXT( rq, false ) == gl_RayQueryCandidateIntersectionTriangleEXT )
        {
            if( rqIsCandidateAccepted( rq, false ) )
            {
                rayQueryConfirmIntersectionEXT( rq );
            }
        }
    }

    if( rayQueryGetIntersectionTypeEXT( rq, true ) == gl_RayQueryCommittedIntersectionTriangleEXT )
    {
        p.baryCoords     = rayQueryGetIntersectionBarycentricsEXT( rq, true );
        p.instIdAndIndex = packInstanceIdAndCustomIndex( rqGetInstanceId( rq, true ),
                                                         rqGetInstanceCustomIndex( rq, true ) );
        p.geomAndPrimIndex = packGeometryAndPrimitiveIndex(
            rayQueryGetIntersectionGeometryIndexEXT( rq, true ),
            rayQueryGetIntersectionPrimitiveIndexEXT( rq, true ) );
    }
    #undef rq
    return p;
}

// Former shadow rays (miss shader RtMissShadowCheck.rmiss, no closest hit): true if
// anything is hit.
bool traceRayQueryAnyHit( uint rayFlags, uint cullMask, vec3 origin, float tMin, vec3 direction, float tMax, bool media )
{
    #define rq g_rayQuery
    rayQueryInitializeEXT( rq,
                           topLevelAS,
                           rayFlags | gl_RayFlagsTerminateOnFirstHitEXT,
                           cullMask & 0xFF,
                           origin,
                           tMin,
                           direction,
                           tMax );

    while( rayQueryProceedEXT( rq ) )
    {
        if( rayQueryGetIntersectionTypeEXT( rq, false ) == gl_RayQueryCandidateIntersectionTriangleEXT )
        {
            if( rqIsCandidateAccepted( rq, media ) )
            {
                rayQueryConfirmIntersectionEXT( rq );
            }
        }
    }

    const bool hit = rayQueryGetIntersectionTypeEXT( rq, true ) != gl_RayQueryCommittedIntersectionNoneEXT;
    #undef rq
    return hit;
}

ShPayload tracePrimaryRay(vec3 origin, vec3 direction)
{
    g_payload = traceRayQuery( getAdditionalRayFlags(),
                               getPrimaryVisibilityCullMask(),
                               origin, globalUniform.primaryRayMinDist, direction, globalUniform.rayLength );
    return g_payload;
}

ShPayload traceReflectionRefractionRay(vec3 origin, vec3 direction, uint surfInstCustomIndex, uint geometryInstanceFlags, bool isRefraction)
{
    g_payload = traceRayQuery( getAdditionalRayFlags(),
                               getReflectionRefractionCullMask(surfInstCustomIndex, geometryInstanceFlags, isRefraction),
                               origin, 0.001, direction, globalUniform.rayLength );
    return g_payload;
}

ShPayload traceIndirectRay(uint surfInstCustomIndex, vec3 surfPosition, vec3 bounceDirection)
{
    g_payload = traceRayQuery( getAdditionalRayFlags(),
                               getIndirectIlluminationCullMask(surfInstCustomIndex),
                               surfPosition, 0.001, bounceDirection, globalUniform.rayLength );
    return g_payload;
}



#ifdef DESC_SET_CUBEMAPS
vec3 adjustSaturation( vec3 c, float saturation )
{
    float grey = getLuminance( c );
    return mix( vec3( grey ), c, saturation );
}

vec3 adjustSky( vec3 skyRaw )
{
    return adjustSaturation( skyRaw, globalUniform.skyColorSaturation ) *
           globalUniform.skyColorMultiplier;
}

vec3 getSkyAlbedo( vec3 direction )
{
    #ifdef DESC_SET_RENDER_CUBEMAP
    if( globalUniform.skyType == SKY_TYPE_RASTERIZED_GEOMETRY )
    {
        return texture( renderCubemap, direction ).rgb;
    }
    else
    #endif
        if( globalUniform.skyType == SKY_TYPE_CUBEMAP )
    {
        direction = mat3( globalUniform.skyCubemapRotationTransform ) * direction;

        return texture( globalCubemaps_Get( globalUniform.skyCubemapIndex ),
                        direction )
            .rgb;
    }
    else
    {
        return globalUniform.skyColorDefault.xyz;
    }
}

vec3 getSky( vec3 direction )
{
    return adjustSky( getSkyAlbedo( direction ) );
}
#endif



#if LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE

#define SHADOW_RAY_EPS       0.01
#define RAY_ORIGIN_LEAK_BIAS 0.01    // offset a bit towards a viewer to prevent light leaks from the other side of polygons

bool traceShadowRay(uint surfInstCustomIndex, vec3 start, vec3 end, bool ignoreFirstPersonViewer /* = false */)
{
    uint cullMask = getShadowCullMask(surfInstCustomIndex);

    if (ignoreFirstPersonViewer)
    {
        cullMask &= ~INSTANCE_MASK_FIRST_PERSON_VIEWER;
    }

    bool media = false;

#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
    // Doom64-RT: THIS IS A MEDIA RAY -- it asks what shadows the FOG, not what
    // shadows a surface -- and sprites must not be part of that answer.
    // Billboards are camera-facing cutouts: their "shadow" through a volume is
    // a sheet that rotates with the view, and their axis-plane shadow proxies
    // (INSTANCE_MASK_RESERVED_0) are solid rectangles -- the shell casing
    // stamped exactly that into the muzzle smoke, and on a stormy map every
    // monster strobes plane-shadows through the lightning shafts. So, unless
    // rt_volume_spriteshadow asks for the old behaviour:
    //   * the proxies are masked out entirely, and
    //   * the ray is routed through the MEDIA hit groups, whose any-hit
    //     discards GEOM_INST_FLAG_SPRITE geometry -- billboards vanish for
    //     this ray while grates and fences still alpha-cut their shafts.
    // This #if resolves per shader, so only RtVolumetric.rgen pays it.
    if( globalUniform.volumeSpriteShadow == 0 )
    {
        cullMask &= ~INSTANCE_MASK_RESERVED_0;
        media = true;
    }
#endif

    vec3 l = end - start;
    float maxDistance = length(l);
    l /= maxDistance;

    return traceRayQueryAnyHit(
        getAdditionalRayFlags(), cullMask, start, 0.001, l, maxDistance - SHADOW_RAY_EPS, media );
}

// Doom64-RT: does this point actually see the SKY along the light direction?
//
// The problem this exists for: traceShadowRay returns "lit" on a MISS, because
// RtMissShadowCheck.rmiss sets isShadowed = 0. That is correct for a sealed
// world and wrong for a Doom map, which has no geometry above a ceiling and is
// full of T-junctions at wall/ceiling seams. Rays leak out of the level, hit
// nothing, and are scored as seeing the sun -- so the sun washes rooms that
// have no opening anywhere near them. No aperture rule can catch that, because
// nothing is squeezing through anything: the ray simply left the map.
//
// In a Doom map the sky is the only legitimate way out, and GZDoom already
// hands us that geometry: sky portals arrive as RG_MESH_PRIMITIVE_SKY_VISIBILITY
// and live in INSTANCE_MASK_WORLD_2, which is excluded from the normal shadow
// mask. So probing WORLD_2 alone answers "did the ray get out through the sky,
// or through a crack?".
//
// Only called when the ordinary shadow ray already MISSED, i.e. only for points
// that are currently considered lit -- so the cost is bounded by how much of the
// screen the sun touches, not by the frame.
bool traceSunReachesSky(vec3 start, vec3 dirToLight)
{
    // a hit means the probe HIT sky geometry, which is what we want:
    // the ray reached the sky. A miss means it found no sky at all on its way
    // out of the world.
    return traceRayQueryAnyHit( getAdditionalRayFlags(),
                                INSTANCE_MASK_WORLD_2,
                                start,
                                0.001,
                                dirToLight,
                                globalUniform.sunSkyProbeMaxDist,
                                false );
}

float traceVisibility(const Surface surf, const vec3 lightPosition, uint lightIndex)
{
    const vec3 start = surf.position + surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;
    const vec3 end = lightPosition;

    const bool ignoreFirstPersonViewer = (globalUniform.lightIndexIgnoreFPVShadows == lightIndex);

    const bool isShadowed = traceShadowRay(surf.instCustomIndex, start, end, ignoreFirstPersonViewer);
    return float(!isShadowed);
}
#endif // LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE



void shade(const Surface surf, const LightSample light, float oneOverPdf, out vec3 diffuse, out vec3 specular)
{
    vec3 l = safeNormalize2(light.position - surf.position, surf.normal);
    float nl = dot(surf.normal, l);

    if( nl <= 0 )
    {
        diffuse = specular = vec3(0);
        return;
    }

    diffuse  = light.dw * nl * light.color * evalBRDFLambertian(1.0);
    specular = light.dw * nl * light.color * evalBRDFSmithGGX(surf.normal, surf.toViewerDir, l, surf.roughness, surf.specularColor);

    diffuse  *= oneOverPdf;
    specular *= oneOverPdf;

    diffuse  = max( diffuse, vec3( 0 ) );
    specular = max( specular, vec3( 0 ) );
}

float targetPdfForLightSample(const LightSample light, const Surface surf)
{
    vec3 d, s;
    shade(surf, light, 1.0, d, s);

    return getLuminance(d + s);
}

float targetPdfForLightSample(uint lightIndex, const Surface surf, const vec2 pointRnd)
{
    const LightSample light = sampleLight(lightSources[lightIndex], surf.position, pointRnd);
    return targetPdfForLightSample(light, surf);
}

Reservoir calcInitialReservoir(uint seed, uint salt, const Surface surf, const vec2 pointRnd)
{
    // RIS candidate count. Traces no rays outside the INITIAL pass (see the
    // LIGHT_SAMPLE_METHOD_INITIAL guard below), so raising it buys better light
    // importance sampling almost for free. Clamped C++-side to [1,64].
    const uint INITIAL_SAMPLES = max(globalUniform.restirInitialSamples, 1u);
    
    Reservoir regularReservoir = emptyReservoir();
#if LIGHT_GRID_ENABLED
    if (isInsideCell(surf.position))
    {
        vec3 gridWorldPos = jitterPositionForLightGrid(surf.position, rnd8_4(seed, salt++).xyz);
        int lightGridBase = cellToArrayIndex(worldToCell(gridWorldPos));

        for (uint i = 0; i < INITIAL_SAMPLES; i++)
        {
            // uniform distribution as a coarse source pdf
            float rnd = rnd16(seed, salt++);
            int lightGridArrayIndex = lightGridBase + 
                clamp(int(rnd * LIGHT_GRID_CELL_SIZE), 0, LIGHT_GRID_CELL_SIZE - 1);
            Reservoir r = unpackReservoirFromLightGrid(initialLightsGrid[lightGridArrayIndex]);

            if (!isReservoirValid(r))
            {
                continue;
            }

            uint xi = r.selected;
            float oneOverSourcePdf_xi = r.weightSum * safePositiveRcp(r.selected_targetPdf);

            LightSample lightSample = sampleLight(lightSources[xi], surf.position, pointRnd);
            float targetPdf_xi = targetPdfForLightSample(lightSample, surf);

            float rndRis = rnd16(seed, salt++);
            updateReservoir(regularReservoir, xi, targetPdf_xi, oneOverSourcePdf_xi, rndRis);
        }
    }
    else
#endif // LIGHT_GRID_ENABLED
    {      
        for (uint i = 0; i < INITIAL_SAMPLES; i++)
        {
            // uniform distribution as a coarse source pdf
            float rnd = rnd16(seed, salt++);
            uint xi = LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET + clamp(uint(rnd * globalUniform.lightCount), 0, globalUniform.lightCount - 1);
            float oneOverSourcePdf_xi = globalUniform.lightCount;

            LightSample lightSample = sampleLight(lightSources[xi], surf.position, pointRnd);
            float targetPdf_xi = targetPdfForLightSample(lightSample, surf);

            float rndRis = rnd16(seed, salt++);
            updateReservoir(regularReservoir, xi, targetPdf_xi, oneOverSourcePdf_xi, rndRis);
        }
    }
    normalizeReservoir(regularReservoir, 1);


    // Doom64-RT: sunSplit -- take the directional light OUT of the lottery.
    //
    // Below, the sun's reservoir is merged into the regular one stochastically
    // (updateCombinedReservoir with a random number), so ONE light wins per
    // pixel. That is correct importance sampling and it has a specific
    // consequence for a weak-but-huge light: with the moon at intensity 90
    // against a level's own lamps and emissives, the moon wins on a minority of
    // pixels, so its shadow is resolved on a sparse random subset of the image
    // and the denoiser flattens what is left. Symptom: sprites cast no moon
    // shadow while the same sprites shadow perfectly from a muzzle flash, which
    // wins selection nearly always because it dominates the pixels it touches
    // (screen/moon_shadow_limit.png, 2026-08-13).
    //
    // rt_shadow_samples cannot fix that -- it averages visibility for the light
    // ALREADY CHOSEN, so it only sharpens the moon where the moon was picked.
    //
    // With sunSplit on, the sun is excluded here and shaded separately and
    // deterministically in processDirectIllumination: every pixel facing it gets
    // exactly one sun shadow ray. Unbiased, because the light is removed from
    // the candidate set rather than counted twice, and cheaper than raising
    // directSamples, which multiplies rays for EVERY light to fix one.
    //
    // DIRECT and INITIAL only. INITIAL is not optional: it writes the reservoir
    // image that DIRECT's sample 0 loads, so leaving the sun in there would put
    // it back into the lottery through the stored reservoir. Indirect and
    // volumetric keep the stock behaviour, so bounce light and fog shafts are
    // bit-identical either way.
    bool includeDirectional = globalUniform.directionalLightExists != 0;
#if (LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_DIRECT) || \
    (LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_INITIAL)
    if (globalUniform.sunSplit > 0.5)
    {
        includeDirectional = false;
    }
#endif

    Reservoir dirLightReservoir = emptyReservoir();
    if (includeDirectional)
    {
        uint xi = LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET;
        float oneOverSourcePdf_xi = 1;

        LightSample lightSample = sampleLight(lightSources[xi], surf.position, pointRnd);
        float targetPdf_xi = targetPdfForLightSample(lightSample, surf);

        float rndRis = rnd16(seed, salt++);
        updateReservoir(dirLightReservoir, xi, targetPdf_xi, oneOverSourcePdf_xi, rndRis);
    }
    normalizeReservoir(dirLightReservoir, 1);

    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_INITIAL
    if (isReservoirValid(dirLightReservoir))
    {
        LightSample lightSample = sampleLight(lightSources[dirLightReservoir.selected], surf.position, pointRnd);
        float v = traceVisibility(surf, lightSample.position, dirLightReservoir.selected);

        if (v < 0.5)
        {
            dirLightReservoir = emptyReservoir();
        }
    }
    #endif // LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_INITIAL

    float rnd = rnd16(seed, salt++); 
    Reservoir combined;
    initCombinedReservoir(
        combined, 
        regularReservoir);
    updateCombinedReservoir(
        combined, 
        dirLightReservoir, rnd);
    normalizeReservoir(combined, 1);

    return combined;
}

bool testSurfaceForReuse(
    const ivec3 curChRenderArea, const ivec2 otherPix,
    float curDepth, float otherDepth,
    const vec3 curNormal, const vec3 otherNormal)
{
    const float DepthThreshold = 0.1;
    const float NormalThreshold = 0.5;

    return 
        testPixInRenderArea(otherPix, curChRenderArea) &&
        (abs(curDepth - otherDepth) / abs(curDepth) < DepthThreshold) &&
        (dot(curNormal, otherNormal) > NormalThreshold);
}

// Select light in screen-space for direct illumination.
//
// saltBase offsets every random draw so the caller can run this more than once
// per pixel and get independent selections (multi-sample-per-pixel loop in
// processDirectIllumination). initReservoir is supplied by the caller rather
// than loaded here: sample 0 passes the stored one from the initial-reservoir
// pass (so N=1 is bit-identical to stock), later samples pass fresh RIS
// candidates from calcInitialReservoir, which costs no rays.
Reservoir selectLight_Direct(const ivec2 pix, uint seed, uint saltBase,
                             const Surface surf, const vec2 pointRnd,
                             const Reservoir initReservoir)
{
    #define TEMPORAL_SAMPLES 1
    #define TEMPORAL_RADIUS 2
    // Spatial reuse: image reads only, no rays. More taps / wider radius = a
    // better-converged reservoir at the cost of bandwidth and (at large radii)
    // more rejected taps from testSurfaceForReuse. Clamped C++-side.
    const uint  SPATIAL_SAMPLES = globalUniform.restirSpatialSamples;
    const float SPATIAL_RADIUS  = globalUniform.restirSpatialRadius;

    const ivec3 chRenderArea = getCheckerboardedRenderArea(pix); // assuming that pix is checkerboarded
    const float motionZ = texelFetch(framebufMotion_Sampler, pix, 0).z;
    const float depthCur = texelFetch(framebufDepthWorld_Sampler, pix, 0).r;
    const vec2 posPrev = getPrevScreenPos(framebufMotion_Sampler, pix);
    uint salt = saltBase;

    // Blue-noise seed for reuse-tap placement (the "TODO: need low discrepancy
    // noise" below). Tiled by REGULAR pixel so adjacent pixels get adjacent
    // texels; see getBlueNoiseSeed(). With white noise the 8 spatial taps clump,
    // neighbouring pixels reuse overlapping neighbourhoods, and their estimates
    // end up correlated -- which shows as low-frequency blotching that no
    // denoiser can separate from signal. Blue noise spreads the taps and makes
    // the residual high-frequency and spatially even, which is also what
    // DLSS-RR asks for (decorrelated reservoirs, RR guide 3.5).
    const bool useBlueNoise = (globalUniform.restirBlueNoise != 0);
    // saltBase folded in so each sample of the multi-sample loop gets a
    // different blue-noise slice -- otherwise every sample would place its reuse
    // taps identically and averaging them would reduce no variance at all.
    const uint bnSeed = getBlueNoiseSeed(getRegularPixFromCheckerboardPix(pix),
                                         globalUniform.frameId + saltBase);


    Reservoir combined;
    initCombinedReservoir(
        combined, 
        initReservoir);


    // temporal
    for (int pixIndex = 0; pixIndex < TEMPORAL_SAMPLES; pixIndex++)
    {
        vec2 rndOffset = (useBlueNoise ? rndBlueNoise8(bnSeed, salt) : rnd8_4(seed, salt)).xy * 2.0 - 1.0;
        salt++;
        // Jitter radius is a uniform: at 0 this reprojects exactly. See
        // restirTemporalJitter -- on grazing surfaces the stock 2px offset moves
        // depth well past the flat 10% reuse threshold, the tap is rejected, and
        // M collapses to 1 precisely where variance is already worst.
        ivec2 pp = ivec2(floor(posPrev + rndOffset * globalUniform.restirTemporalJitter));

        {
            const float depthPrev = texelFetch(framebufDepthWorld_Prev_Sampler, pp, 0).r;
            const vec3 normalPrev = texelFetchNormal_Prev(pp);

            if (!testSurfaceForReuse(chRenderArea, pp, 
                                     depthCur, depthPrev - motionZ,
                                     surf.normal, normalPrev))
            {
                continue;
            }
        }

        Reservoir temporal = imageLoadReservoir_Prev(pp);
        // renormalize to prevent precision problems
        normalizeReservoir(temporal, initReservoir.M * max(globalUniform.restirTemporalMCap, 1u));

        float temporalTargetPdf_curSurf = 0.0;
        if (temporal.selected != LIGHT_INDEX_NONE)
        {
            uint selected_curFrame = lightSources_Index_PrevToCur[temporal.selected];

            if (selected_curFrame != UINT32_MAX && selected_curFrame != LIGHT_INDEX_NONE)
            {
                temporalTargetPdf_curSurf = targetPdfForLightSample(selected_curFrame, surf, pointRnd);
                temporal.selected = selected_curFrame;
            }
        }

        float rnd = rnd16(seed, salt++);
        updateCombinedReservoir_newSurf(
            combined, 
            temporal, temporalTargetPdf_curSurf, rnd);
    } 

    for (uint pixIndex = 0; pixIndex < SPATIAL_SAMPLES; pixIndex++)
    {
        vec2 rndOffset = (useBlueNoise ? rndBlueNoise8(bnSeed, salt) : rnd8_4(seed, salt)).xy * 2.0 - 1.0;
        salt++;
        ivec2 pp = pix + ivec2(rndOffset * SPATIAL_RADIUS);

        {
            const float depthOther = texelFetch(framebufDepthWorld_Sampler, pp, 0).r;
            const vec3 normalOther = texelFetchNormal(pp);

            if (!testSurfaceForReuse(chRenderArea, pp, 
                                     depthCur, depthOther,
                                     surf.normal, normalOther))
            {
                continue;
            }
        }

        Reservoir spatial = imageLoadReservoirInitial(pp);

        float rnd = rnd16(seed, salt++);
        updateCombinedReservoir(
            combined, 
            spatial, rnd);
    }


    if (combined.weightSum <= 0.0 || combined.selected == LIGHT_INDEX_NONE)
    {
        return emptyReservoir();
    }

    return combined;
}

Reservoir selectLight_Uniform(uint seed)
{
    float rnd = rnd16(seed, RANDOM_SALT_LIGHT_CHOOSE_DIRECT_BASE);

    uint lt = globalUniform.lightCount;
    lt += globalUniform.directionalLightExists != 0 ? 1 : 0;

    Reservoir r = emptyReservoir();

    r.selected = clamp(uint(rnd * lt), 0, lt - 1);
    r.selected_targetPdf = 1.0 / float(lt);
    r.weightSum = 1.0;
    r.M = 1;

    return r;
}

// Select light in world-space for light bounces
Reservoir selectLight_Indir(uint seed, const Surface surf, const vec2 pointRnd)
{
    return calcInitialReservoir(seed, RANDOM_SALT_LIGHT_CHOOSE_INDIRECT_BASE, surf, pointRnd);
}

vec2 getLightPointRnd(uint seed)
{
    return rnd16_2(seed, RANDOM_SALT_LIGHT_POINT) * 0.99;
}

// Point on the light for sample i of the multi-sample loop. Sample 0 must match
// getLightPointRnd() exactly so N=1 stays bit-identical to stock.
vec2 getLightPointRndForSample(uint seed, uint sampleIndex)
{
    return sampleIndex == 0u
               ? getLightPointRnd(seed)
               : rnd16_2(seed, RANDOM_SALT_LIGHT_POINT + sampleIndex) * 0.99;
}

// Last visibility term computed by traceDirectIllumination, for debugVisibility.
// A file-scope value rather than another out-parameter so the debug path adds
// nothing to the signature every caller has to thread through. Seeded to 1.0
// (fully lit) so a pixel whose direct lighting never ran -- no light chosen, or
// bounceIndex past maxBounceShadowsLights -- reads as "not shadowed" instead of
// as a false umbra, which would be exactly the wrong answer for this debug view.
float g_debugVisibility = 1.0;

#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
// Doom64-RT: the near-light fade radius in force for the froxel being shaded.
//
// It is a file-scope value rather than the uniform read directly because
// localised smoke needs a DIFFERENT fade from the fog, and needs it per cell:
// the fog wants the fade (a carried light must not white out the screen) while
// a smoke puff wants the opposite (the muzzle flash lighting the puff at the
// barrel is the whole effect). Choosing per frame instead would retune the
// shipped fog every time the player fired.
//
// GLSL requires a constant initializer on a global, so the "use the fog's
// value" state is the sentinel -1 rather than the uniform itself. A caller that
// never assigns it -- which is every caller except RtVolumetric.rgen's main()
// -- therefore behaves exactly as it did before smoke existed.
float g_volumeLightNearFade = -1.0;

// Doom64-RT: the matching FAR cutoff, and smoke-only for the same reason the
// near one is per cell. Fog wants every light in the level -- a lamp down the
// corridor IS the effect. Smoke is a small object running the all-lights
// estimate at one sample per froxel, so a saturated emissive across the room
// wins the reservoir often enough to tint the whole puff. 0 = no limit.
float g_volumeLightFarFade = 0.0;
#endif

#if LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE
bool isDirectIlluminationValid(int bounceIndex)
{
    bool v = true;
    v = v && (bounceIndex < globalUniform.maxBounceShadowsLights || globalUniform.maxBounceShadowsLights == 0);
    v = v && (globalUniform.lightCount + globalUniform.directionalLightExists > 0);
    
    return v;
}

void traceDirectIllumination( uint            seed,
                              const Surface   surf,
                              const Reservoir reservoir,
                              const vec2      pointRnd,
                              int             bounceIndex,
                              out float       out_distance,
                              out vec3        out_diffuse,
                              out vec3        out_specular
    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
                              , inout vec3 out_lightdirection
    #endif
                              )
{    
    const LightSample light = sampleLight(lightSources[reservoir.selected], surf.position, pointRnd);
    shade(surf, light, calcSelectedSampleWeight(reservoir), out_diffuse, out_specular);

#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
    // Doom64-RT: near-field fade, FOG ONLY.
    //
    // A light standing inside the medium lights the froxels around it by
    // inverse square, so a light at ~0 m -- the flashlight, a muzzle flash,
    // anything carried -- puts an enormous in-scattered term into the froxels
    // right in front of the camera and the screen whites out. That is what a
    // headlight in fog physically does, and it is unplayable: the flashlight
    // becomes a switch that blinds you.
    //
    // So scattering is faded out within g_volumeLightNearFade metres OF THE
    // LIGHT -- the uniform's value in fog, smokeLightNearFade inside a puff.
    // It is deliberately keyed off the light's distance rather than the
    // camera's, because the thing to remove is glare from a light you are
    // holding, not the fog near the camera -- the beam's shaft further down the
    // corridor is exactly the look this feature is for, and it survives.
    //
    // Directional lights are unaffected: sampleLight puts their position far
    // away, so the fade never triggers on the moon or a lightning strike.
    // 0 disables it and restores the physical behaviour.
    const float nearFade = g_volumeLightNearFade >= 0.0 ? g_volumeLightNearFade
                                                        : globalUniform.volumeLightNearFade;
    const float dToLight = length( light.position - surf.position );

    if( nearFade > 0.001 )
    {
        out_diffuse *= smoothstep( 0.0, nearFade, dToLight );
    }

    // ...and the far cutoff, faded over the last quarter of the range so a light
    // does not switch off as the puff drifts. Only smoke ever sets this.
    //
    // NOT ON A DIRECTIONAL LIGHT, and this is the bug that made smoke black in a
    // moon shaft. The near fade above is safe from it by luck -- its comment even
    // says so, "sampleLight puts their position far away, so the fade never
    // triggers on the moon" -- but the FAR fade reads the same dToLight, and a
    // position placed far away by construction is exactly what that test culls.
    // The moon was therefore multiplied to ZERO in every smoke cell while fog,
    // which never sets this value, kept its shafts. Reported as smoke staying
    // black inside a visibly lit moonbeam.
    //
    // A directional light has no position to be far from: its distance term is
    // meaningless, so it must be exempt rather than clamped.
    if( g_volumeLightFarFade > 0.001 &&
        reservoir.selected != LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET )
    {
        out_diffuse *= 1.0 - smoothstep( g_volumeLightFarFade * 0.75,
                                         g_volumeLightFarFade,
                                         dToLight );
    }
#endif
    
    if (getLuminance(out_diffuse + out_specular) <= 0.0)
    {
        out_diffuse = out_specular = vec3(0.0);
        return;
    }

    if (bounceIndex < globalUniform.maxBounceShadowsLights)
    {
        // Visibility is the dominant variance term at 1 spp: a single shadow ray
        // makes this a binary 0/1 multiply, so a pixel is either fully lit or
        // fully black regardless of how well ReSTIR chose the light. That floor
        // is what survives into the unfiltered signal, and no amount of reuse
        // decorrelation touches it (measured 2026-08-07: blue-noise reuse taps
        // changed nothing).
        //
        // Averaging visibility over N independently sampled points on the SAME
        // chosen light turns it into a fraction -> real soft shadow, variance
        // ~1/sqrt(N). Only the visibility factor is averaged; shading keeps the
        // reservoir's own sample, so the RIS weight and the light-selection
        // estimator are untouched and energy is unchanged in expectation.
        //
        // DIRECT only: secondary bounces stay at one ray, where the extra cost
        // would not pay for itself.
        float visibility;
    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_DIRECT
        const uint shadowN = clamp(globalUniform.shadowSamples, 1u, 8u);
        if (shadowN > 1u)
        {
            visibility = 0.0;
            for (uint si = 0; si < shadowN; si++)
            {
                // si == 0 reuses the reservoir's point so N=1 is bit-identical
                // to the stock path; later taps get fresh points on the light.
                vec2 prnd = (si == 0u)
                                ? pointRnd
                                : rnd16_2(seed, RANDOM_SALT_SHADOW_SAMPLES_BASE + si);

                const LightSample ls =
                    (si == 0u) ? light
                               : sampleLight(lightSources[reservoir.selected], surf.position, prnd);

                visibility += traceVisibility(surf, ls.position, reservoir.selected);
            }
            visibility /= float(shadowN);
        }
        else
        {
            visibility = traceVisibility(surf, light.position, reservoir.selected);
        }
    #else
        visibility = traceVisibility(surf, light.position, reservoir.selected);
    #endif

        // Doom64-RT: the sky-reach test, applied HERE rather than in the direct
        // pass, because this function is the one choke point every path shares --
        // surface, indirect AND volumetric. Putting it in selectLight_Direct only
        // covered surface shading, which is why the visible shafts (volumetric
        // scattering, a different LIGHT_SAMPLE_METHOD) never turned red and why
        // rt_sun_require_sky appeared to do nothing to them.
        //
        // Only for the directional light, and only where the shadow ray already
        // said "lit" -- so no extra ray is traced for anything already in shadow.
        if (visibility > 0.0 &&
            reservoir.selected == LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET &&
            (globalUniform.sunRequireSky > 0.5 || globalUniform.sunLeakDebug > 0.5))
        {
            const vec3 probeStart = surf.position + surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;
            const bool reachedSky =
                traceSunReachesSky(probeStart,
                                   safeNormalize2(light.position - surf.position, vec3(0)));

            g_sunLeakClass = reachedSky ? 1 : 2;

            // The fix runs FIRST, so the debug views show the result of it rather
            // than replacing it. With require_sky and colour mode both on, every
            // surviving shaft is red -- an all-red screen IS the confirmation the
            // fix worked. Making these mutually exclusive (the first attempt)
            // meant the fix could never be seen, only trusted.
            if (globalUniform.sunRequireSky > 0.5 && !reachedSky)
            {
                visibility = 0.0;
            }

            // mode 1: isolate the leak -- drop everything legitimate
            if (globalUniform.sunLeakDebug > 0.5 && globalUniform.sunLeakDebug < 1.5 && reachedSky)
            {
                visibility = 0.0;
            }

            // mode 2: colour whatever survived. Re-shade, because out_diffuse
            // above was computed from the moon's real colour.
            if (globalUniform.sunLeakDebug > 1.5 && visibility > 0.0)
            {
                LightSample dbg = light;
                dbg.color = (reachedSky ? vec3(1.0, 0.02, 0.02) : vec3(0.05, 1.0, 0.10))
                            * max(globalUniform.sunLeakDebugMul, 0.001);
                vec3 d2, s2;
                shade(surf, dbg, calcSelectedSampleWeight(reservoir), d2, s2);
                out_diffuse  = d2;
                out_specular = s2;
            }
        }

        g_debugVisibility = visibility;

        out_diffuse  *= visibility;
        out_specular *= visibility;

    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_DIRECT
        out_distance = length(light.position - surf.position);
    #endif
    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
        out_lightdirection = safeNormalize2( surf.position - light.position, vec3( 0 ) );
    #endif
    }
}
#endif // LIGHT_SAMPLE_METHOD != LIGHT_SAMPLE_METHOD_NONE


#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_DIRECT
// Doom64-RT: the sun's own single-light reservoir, for sunSplit.
//
// Deliberately built exactly as calcInitialReservoir builds dirLightReservoir --
// one candidate, oneOverSourcePdf 1, normalized to 1 -- so the weight
// calcSelectedSampleWeight() hands to shade() is the same one the stochastic
// path would have used had the sun won. That is what makes this a split rather
// than a second, differently-scaled copy of the light: sun-lit surfaces are the
// SAME brightness with sunSplit on or off, only less noisy.
//
// It returns a reservoir rather than shading here so the caller can pass it
// straight to traceDirectIllumination, which owns every piece of sun-specific
// logic -- sunRequireSky, the red/green leak debug, g_debugVisibility, the
// volumetric fades. Duplicating any of that here is how those get out of step.
Reservoir calcSunOnlyReservoir(const Surface surf, const vec2 pointRnd)
{
    Reservoir r = emptyReservoir();

    const uint  xi         = LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET;
    LightSample lightSample = sampleLight(lightSources[xi], surf.position, pointRnd);
    const float targetPdf   = targetPdfForLightSample(lightSample, surf);

    // rndRis 0: a single candidate is always accepted, so no random draw is
    // needed and none is spent.
    updateReservoir(r, xi, targetPdf, 1.0, 0.0);
    normalizeReservoir(r, 1);

    return r;
}

Reservoir processDirectIllumination(uint seed, const ivec2 pix, const Surface surf, out float out_distance, out vec3 out_diffuse, out vec3 out_specular)
{
    out_diffuse = out_specular = vec3(0.0);
    out_distance = MAX_RAY_LENGTH;

    if (!isDirectIlluminationValid(0))
    {
        return emptyReservoir();
    }
    // Multi-sample direct lighting.
    //
    // The path tracer is 1 spp and only converges through temporal accumulation,
    // which camera motion legitimately destroys -- so the raw signal is what
    // shows through while moving. N independent estimates, averaged, reduce that
    // variance at the SOURCE (~1/sqrt(N)), upstream of the denoiser, so A-SVGF
    // and DLSS-RR benefit equally.
    //
    // Each sample draws its own light point, its own RIS candidates and its own
    // reuse taps, so the estimates are genuinely independent rather than N
    // copies of one answer. Sample 0 reproduces the stock path exactly, which is
    // what makes N=1 a guaranteed no-op.
    const uint N = max(globalUniform.directSamples, 1u);

    Reservoir firstReservoir = emptyReservoir();
    vec3      accumDiffuse   = vec3(0.0);
    vec3      accumSpecular  = vec3(0.0);
    uint      validCount     = 0;

    for (uint si = 0; si < N; si++)
    {
        const uint saltBase = (si == 0u)
                                  ? RANDOM_SALT_LIGHT_CHOOSE_DIRECT_BASE
                                  : (RANDOM_SALT_DIRECT_SPP_BASE + si * RANDOM_SALT_SPP_STRIDE);

        const vec2 pointRnd = getLightPointRndForSample(seed, si);

        // sample 0 reuses the stored initial reservoir (stock); later samples
        // draw fresh RIS candidates in-shader, which traces no rays here
        const Reservoir initial =
            (si == 0u)
                ? imageLoadReservoirInitial(pix)
                : calcInitialReservoir(seed, saltBase + RANDOM_SALT_SPP_INITIAL_OFFSET, surf, pointRnd);

        const Reservoir reservoir =
            selectLight_Direct(pix, seed, saltBase, surf, pointRnd, initial);

        if (si == 0u)
        {
            // the temporal chain, ASVGF gradients and the specular hit distance
            // guide must stay single-valued -- always sample 0's
            firstReservoir = reservoir;
        }

        if (!isReservoirValid(reservoir))
        {
            continue;
        }

        float sampleDist;
        vec3  sampleDiffuse;
        vec3  sampleSpecular;
        traceDirectIllumination(seed, surf, reservoir, pointRnd, 0,
                                sampleDist, sampleDiffuse, sampleSpecular);

        accumDiffuse  += sampleDiffuse;
        accumSpecular += sampleSpecular;
        validCount++;

        if (si == 0u)
        {
            out_distance = sampleDist;
        }
    }

    // Divide by N, not validCount: an invalid reservoir is a legitimate zero
    // contribution for that sample, not a sample that did not happen. Dividing
    // by validCount would bias the estimate brighter wherever some samples miss.
    if (validCount > 0)
    {
        accumDiffuse  /= float(N);
        accumSpecular /= float(N);
    }
    else
    {
        accumDiffuse = accumSpecular = vec3(0.0);
    }

    // Doom64-RT: sunSplit -- the directional light, shaded deterministically.
    //
    // ONE shadow ray per pixel that faces the sun, added to the ReSTIR estimate
    // over every other light rather than competing with it. See the long note in
    // calcInitialReservoir for why the stochastic merge loses a weak sun's
    // shadows.
    //
    // This runs even when validCount == 0, and that is not an edge case: a room
    // lit only by the moon has no regular lights to build a valid reservoir
    // from, and the early-out that used to sit here would have returned black
    // for exactly the surfaces this feature exists to light. That would have
    // read as "sunSplit makes outdoor areas darker", which is the kind of
    // regression that gets a fix reverted rather than debugged.
    if (globalUniform.sunSplit > 0.5 && globalUniform.directionalLightExists != 0)
    {
        const vec2      sunRnd = getLightPointRndForSample(seed, 0);
        const Reservoir sunRes = calcSunOnlyReservoir(surf, sunRnd);

        if (isReservoirValid(sunRes))
        {
            float sunDist;
            vec3  sunDiffuse;
            vec3  sunSpecular;
            traceDirectIllumination(seed, surf, sunRes, sunRnd, 0,
                                    sunDist, sunDiffuse, sunSpecular);

            accumDiffuse  += sunDiffuse;
            accumSpecular += sunSpecular;

            // Only claim the distance if nothing else did. out_distance feeds the
            // specular hit-distance guide, and a directional light's "position"
            // is a construction far outside the map -- handing that to the
            // denoiser as a hit distance would be a lie about where the highlight
            // lives (see rt_rr_spechitdist).
            if (validCount == 0)
            {
                out_distance = MAX_RAY_LENGTH;
            }

            validCount++;
        }
    }

    if (validCount == 0)
    {
        return emptyReservoir();
    }

    out_diffuse  = accumDiffuse;
    out_specular = accumSpecular;

    return firstReservoir;
}
#endif

#if (LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_INDIR) || (LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME)
vec3 processDirectIllumination( uint          seed,
                                const Surface surf,
                                int           bounceIndex
    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
                                , inout vec3 out_lightdirection 
    #endif
                                )
{
    if (!isDirectIlluminationValid(bounceIndex))
    {
        return vec3(0.0);
    }
    const vec2 pointRnd = getLightPointRnd(seed);
    
    const Reservoir reservoir = selectLight_Indir(seed, surf, pointRnd);
    if (!isReservoirValid(reservoir))
    {
        return vec3(0.0);
    } 

    vec3 out_diffuse;
    vec3 unusedv; float unusedf;
    traceDirectIllumination(
        seed, surf, reservoir, pointRnd, bounceIndex, unusedf, out_diffuse, unusedv
    #if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_VOLUME
        , out_lightdirection
    #endif
    );

    #if SHIPPING_HACK
    if( reservoir.selected != LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET )
    {
        out_diffuse = clamp( out_diffuse, vec3( 0 ), vec3( 500 ) );
    }
    #endif

    return out_diffuse;
}
#endif

#if LIGHT_SAMPLE_METHOD == LIGHT_SAMPLE_METHOD_GRADIENTS
void processDirectIllumination(uint seed, const Surface surf, const Reservoir reservoir, out vec3 out_diffuse, out vec3 out_specular)
{
    out_diffuse = out_specular = vec3(0.0);

    if (!isDirectIlluminationValid(0))
    {
        return;
    }
    const vec2 pointRnd = getLightPointRnd(seed);

    if (!isReservoirValid(reservoir))
    {
        return;
    } 
    
    float unusedf;
    traceDirectIllumination(seed, surf, reservoir, pointRnd, 0, unusedf, out_diffuse, out_specular);
}
#endif

#endif // RAYGEN_COMMON_H_