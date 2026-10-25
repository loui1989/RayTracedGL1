// Copyright (c) 2021-2022 Sultim Tsyrendashiev
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



// This file was originally a raygen shader. But G-buffer decals are drawn
// on primary surfaces, but not in perfect reflections/refractions. Because
// 

// Must be defined:
// - either RAYGEN_PRIMARY_SHADER or RAYGEN_REFL_REFR_SHADER
// - MATERIAL_MAX_ALBEDO_LAYERS

#if defined(RAYGEN_PRIMARY_SHADER) && defined(RAYGEN_REFL_REFR_SHADER)
    #error Only one of RAYGEN_PRIMARY_SHADER and RAYGEN_REFL_REFR_SHADER must be defined
#endif
#if !defined(RAYGEN_PRIMARY_SHADER) && !defined(RAYGEN_REFL_REFR_SHADER)
    #error RAYGEN_PRIMARY_SHADER or RAYGEN_REFL_REFR_SHADER must be defined
#endif
#ifndef MATERIAL_MAX_ALBEDO_LAYERS
    #error MATERIAL_MAX_ALBEDO_LAYERS is not defined
#endif 
#ifndef MATERIAL_LIGHTMAP_LAYER_INDEX
    #error MATERIAL_LIGHTMAP_LAYER_INDEX is not defined
#endif 


#define DESC_SET_TLAS 0
#define DESC_SET_FRAMEBUFFERS 1
#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#define DESC_SET_RANDOM 5
// Doom64-RT: the volumetric set (cloud map), for cloudSunAttenuation in Light.h.
#define DESC_SET_VOLUMETRIC 11
#define DESC_SET_LIGHT_SOURCES 6
#define DESC_SET_CUBEMAPS 7
#define DESC_SET_RENDER_CUBEMAP 8
#define DESC_SET_PORTALS 9
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_NONE)
#include "RaygenCommon.h"

vec2 getMotionVectorForUpscaler(const vec2 motionCurToPrev)
{
    return motionCurToPrev;
}

#define UPSCALER_REACTIVITY_REFLREFR 0.8

vec2 getMotionForInfinitePoint(const ivec2 pix)
{
    // treat as a point with .w=0, i.e. at infinite distance
    vec3 rayDir = getRayDir(getPixelUVWithJitter(pix));

    vec3 viewSpacePosCur   = mat3(globalUniform.view)     * rayDir;
    vec3 viewSpacePosPrev  = mat3(globalUniform.viewPrev) * rayDir;

    vec3 clipSpacePosCur   = mat3(globalUniform.projection)     * viewSpacePosCur;
    vec3 clipSpacePosPrev  = mat3(globalUniform.projectionPrev) * viewSpacePosPrev;

    // don't divide by .w
    vec3 ndcCur            = clipSpacePosCur.xyz;
    vec3 ndcPrev           = clipSpacePosPrev.xyz;

    vec2 screenSpaceCur    = ndcCur.xy  * 0.5 + 0.5;
    vec2 screenSpacePrev   = ndcPrev.xy * 0.5 + 0.5;

    return screenSpacePrev - screenSpaceCur;
}

void storeSky( const ivec2 pix,
               const vec3  rayDir,
               bool        calculateSkyAndStoreToAlbedo,
               const vec3  throughput,
#ifdef RAYGEN_PRIMARY_SHADER
               float firstHitDepthNDC )
#else
               bool wasSplit )
#endif
{
    imageStore( framebufIsSky, pix, ivec4( 1 ) );

    {
        // check if was already in G-buffer after rasterization pass
        vec3 skyColor =
            calculateSkyAndStoreToAlbedo
                ? getSkyAlbedo( rayDir )
                : imageLoad( framebufAlbedo, getRegularPixFromCheckerboardPix( pix ) ).rgb;

        {
            // to hdr
            skyColor = adjustSky( skyColor ) / M_PI;
        }

        imageStore(
            framebufAlbedo, getRegularPixFromCheckerboardPix( pix ), vec4( skyColor, 0.0 ) );
    }

    vec2 m = getMotionForInfinitePoint(pix);

    imageStoreNormal(                       pix, vec3(0.0));
    imageStore(framebufMetallicRoughness,   pix, vec4(0.0));
    imageStore(framebufDepthWorld,          pix, vec4(MAX_RAY_LENGTH * 2.0));
    imageStore(framebufMotion,              pix, vec4(m, 0.0, 0.0));
    imageStore(framebufSurfacePosition,     pix, vec4(SURFACE_POSITION_INCORRECT));
    imageStore(framebufVisibilityBuffer,    pix, vec4(UINT32_MAX));
    imageStore(framebufViewDirection,       pix, vec4(rayDir, 0.0));
    imageStore(framebufScreenEmisRT,        getRegularPixFromCheckerboardPix( pix ), vec4( 0.0 ) );
#ifdef RAYGEN_PRIMARY_SHADER
    imageStore(framebufPrimaryToReflRefr,   pix, uvec4(0, 0, PORTAL_INDEX_NONE, 0));
    imageStore(framebufDepthGrad,           pix, vec4(0.0));
    imageStore(framebufDepthNdc,            getRegularPixFromCheckerboardPix(pix), vec4(clamp(firstHitDepthNDC, 0.0, 1.0)));
    imageStore(framebufMotionDlss,          getRegularPixFromCheckerboardPix(pix), vec4(getMotionVectorForUpscaler(m), 0.0, 0.0));
    imageStore(framebufThroughput,          pix, vec4(throughput, 0.0));
    imageStore(framebufReactivity,          getRegularPixFromCheckerboardPix(pix), vec4(0.0));
#else
    imageStore(framebufThroughput,          pix, vec4(throughput, wasSplit ? 1.0 : -1.0));
    imageStore(framebufReactivity,          getRegularPixFromCheckerboardPix(pix), vec4(UPSCALER_REACTIVITY_REFLREFR));
#endif
}

uint getNewRayMedia(int i, uint prevMedia, uint geometryInstanceFlags, float roughness)
{
    // if camera is not in vacuum, assume that new media is vacuum
    if (i == 0 && globalUniform.cameraMediaType != MEDIA_TYPE_VACUUM)
    {
       return MEDIA_TYPE_VACUUM;
    }

    return getMediaTypeFromFlags(geometryInstanceFlags, roughness);
}

vec3 getWaterNormal(const RayCone rayCone, const vec3 rayDir, const vec3 baseNormal, const vec3 position, bool wasPortal)
{
    const mat3 basis = getONB(baseNormal);
    const vec2 baseUV = vec2(dot(position, basis[0]), dot(position, basis[1])); 


    // how much vertical flow to apply
    float verticality = 1.0 - abs(dot(baseNormal, globalUniform.worldUpVector.xyz));

    // project basis[0] and basis[1] on up vector
    vec2 flowSpeedVertical = 10 * vec2(dot(basis[0], globalUniform.worldUpVector.xyz), 
                                       dot(basis[1], globalUniform.worldUpVector.xyz));

    vec2 flowSpeedHorizontal = vec2(1.0);


    const float uvScale = 0.05 / globalUniform.waterTextureAreaScale;
    vec2 speed0 = uvScale * mix(flowSpeedHorizontal, flowSpeedVertical, verticality) * globalUniform.waterWaveSpeed;
    vec2 speed1 = -0.9 * speed0 * mix(1.0, -0.1, verticality);


    // for texture sampling
    float derivU = globalUniform.waterTextureDerivativesMultiplier * 0.5 * uvScale * getWaterDerivU(rayCone, rayDir, baseNormal);

    // make water sharper if visible through the portal
    if (wasPortal)
    {
        derivU *= 0.1;
    }


    vec2 uv0 = uvScale * baseUV + globalUniform.time * speed0;
    vec3 n0 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv0, derivU).xyz;
    n0.xy = n0.xy * 2.0 - vec2(1.0);


    vec2 uv1 = 0.8 * uvScale * baseUV + globalUniform.time * speed1;
    vec3 n1 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv1, derivU).xyz;
    n1.xy = n1.xy * 2.0 - vec2(1.0);


    vec2 uv2 = 0.1 * (uvScale * baseUV + speed0 * sin(globalUniform.time * 0.5));
    vec3 n2 = getTextureSampleDerivU(globalUniform.waterNormalTextureIndex, uv2, derivU).xyz;
    n2.xy = n2.xy * 2.0 - vec2(1.0);


    const float strength = globalUniform.waterWaveStrength;

    const vec3 n = normalize(vec3(0, 0, 1) + strength * (0.25 * n0 + 0.2 * n1 + 0.1 * n2));
    return basis * n;   
}

// Doom64-RT: which of the four liquids this surface is.
//
// Doom 64 draws water, nukage, sludge and blood as the same 64-frame animated
// flat design in four palettes (D64W*/D64N*/D64S*/D64B*, plus the WFALL/SFALL/
// BFALL wall sheets), so they all want this one surface shader and differ only
// in body and crest colour. The engine classifies by texture name and packs the
// answer into two geometry-instance bits; index 0 is water, so a primitive that
// is merely flagged RG_MESH_PRIMITIVE_WATER behaves exactly as before.
uint getLiquidId( uint geomInstFlags )
{
    return ( ( geomInstFlags & GEOM_INST_FLAG_LIQUID_BIT0 ) != 0 ? 1u : 0u ) |
           ( ( geomInstFlags & GEOM_INST_FLAG_LIQUID_BIT1 ) != 0 ? 2u : 0u );
}

// Colour the caustic veins tend to at full mask, per liquid. Pale, never white:
// white crests read as foam/plastic. Water's is blue-white because that is what
// the D64 flat's brightest texels are; the others follow their own palette.
vec3 getLiquidCrestColor( uint liquidId )
{
    return globalUniform.stylizedLiquidCrest[ liquidId ].rgb;
}

// Doom64-RT: stylized water surface colour.
//
// The physical water path (refract + Beer-Lambert absorption + mirror
// reflection) reads far too "real" for Doom 64, and it is also wrong for these
// maps: D64W2_01 / D64W1_01 are plain FLOOR FLATS, there is no sector under
// them to refract into. So the stylized path keeps the surface opaque and
// rebuilds its look from the flat itself: a deep body colour carrying the
// texture's own pale caustic veins, shimmering with the animated wave normal.
//
// The body/crest pair comes from liquidId, so nukage, sludge and blood get the
// same treatment in their own palette rather than turning blue.
//
//   texAlbedo   the flat as sampled by the primary pass (near-black, veins peak
//               around 0.06 / 0.13 / 0.28 sRGB on the water flat)
//   waveNormal  animated water normal (getWaterNormal)
//   baseNormal  the surface normal before the waves
//   liquidId    0 water, 1 nukage, 2 sludge, 3 blood (getLiquidId)
// out caustic   0..1 vein mask, reused for the screen-space sheen
vec3 getStylizedWaterAlbedo( const vec3  texAlbedo,
                             const vec3  waveNormal,
                             const vec3  baseNormal,
                             const uint  liquidId,
                             const float flow,
                             out   float caustic )
{
    // the veins ARE the caustics in the source art: normalize the flat's
    // luminance against the brightest texel so the mask is art-independent
    float veins = clamp( getLuminance( texAlbedo ) /
                             max( 0.0001, globalUniform.stylizedWaterVeinRef ),
                         0.0,
                         1.0 );

    // wave crests: how far the animated normal tilts off the plane
    float tilt    = clamp( length( waveNormal - baseNormal ) * 3.0, 0.0, 1.0 );
    float shimmer = tilt * tilt;

    // Doom64-RT: the flow. `flow` is a detail texture ADVECTED along the vein
    // direction in the primary pass (HitInfo.inl), 0..1 centred on 0.5, and
    // exactly 0 where the texel carries no flow. It modulates the vein
    // brightness both ways, so blobs of brighter and darker liquid slide down
    // each channel -- texture moving, not brightness pulsing in place.
    const float flowAmt = globalUniform.stylizedLiquidFlow[ liquidId ];
    float       flowMod = 0.0;
    if( flowAmt > 0.0 && flow > 0.0 )
    {
        flowMod = flowAmt * ( flow * 2.0 - 1.0 );
    }

    // veins keep their shape, but breathe with the waves -- except where the
    // material relief has replaced the waves. At relief 1 the wave normal IS
    // the surface normal, so tilt and shimmer are identically zero and
    // stylizedWaterCaustic would be a silent no-op; fading it out against
    // relief says so, and hands the animation to the flow instead.
    const float relief = globalUniform.stylizedLiquidRelief[ liquidId ];
    caustic = clamp( veins * ( 1.0 +
                               globalUniform.stylizedWaterCaustic * shimmer * ( 1.0 - relief ) +
                               flowMod ),
                     0.0,
                     1.0 );

    const vec3 body  = globalUniform.stylizedLiquidTint[ liquidId ].rgb;
    const vec3 crest = mix( body, getLiquidCrestColor( liquidId ), 0.85 );

    return mix( body, crest, caustic );
}

// ---------------------------------------------------------------------------
// Doom64-RT: lava.
//
// Two things the texture pipeline cannot do for itself, which is the whole
// reason lava carries a geometry flag.
//
// 1. BLOOM. Primary emission is the raw _e sample; _e is an 8-bit texture so it
//    caps at 1.0, screen emission is that times emissionMaxScreenColor (3), and
//    rt_bloom_threshold is 16. A lava flat therefore cannot reach the bloom
//    threshold by any amount of painting -- it tops out at a fifth of it. The
//    boost is applied here, to lava only, so the rest of the game's emissives
//    keep the balance they were tuned with.
//
// 2. MOTION. The flat is a 5-frame ping-pong of a static crackle, and its
//    animation had to be averaged away (the frames light different cracks, which
//    read as blinking). So the heat has to move in the shader instead: a
//    low-frequency field drifting across the surface, plus a slow global pulse.
//
// The field is QUANTIZED to a world-space cell before it is sampled. A smooth
// gradient sliding over 64x64 pixel art reads as a modern shader bolted onto the
// wrong texture; stepping it to roughly the flat's own texel size keeps the
// motion as chunky as the thing it is moving over.
float getLavaHeat( const vec3 position )
{
    const vec3 up   = globalUniform.worldUpVector.xyz;
    const vec3 onPlane = position - up * dot( position, up );
    const mat3 onb  = getONB( up );
    vec2       xy   = vec2( dot( onPlane, onb[ 0 ] ), dot( onPlane, onb[ 1 ] ) );

    // chunky on purpose -- see above
    const float cell = max( 0.01, globalUniform.lavaFlowPixel );
    xy               = floor( xy / cell ) * cell;

    // Two layers drifting against each other, so the pattern never settles into
    // a direction the eye can follow. getTextureSampleLod, not getTextureSample:
    // a raygen shader has no quad derivatives.
    const float t  = globalUniform.time * globalUniform.lavaFlowSpeed;
    const float sc = globalUniform.lavaFlowScale;

    const float a =
        getTextureSampleLod( globalUniform.waterNormalTextureIndex, xy * sc + vec2( t, t * 0.6 ), 0.0 ).x;
    const float b = getTextureSampleLod(
                        globalUniform.waterNormalTextureIndex, xy * sc * 1.7 - vec2( t * 0.8, t * 0.3 ), 0.0 )
                        .y;

    // 0..1, centred so the mean stays put: this MULTIPLIES the emission, and a
    // field with a mean above 1 would quietly brighten the lava as well as
    // animate it, which is a different decision from the one being made here.
    const float field = clamp( ( a + b ) * 0.5, 0.0, 1.0 );
    const float drift = mix( 1.0 - globalUniform.lavaFlowStrength,
                             1.0 + globalUniform.lavaFlowStrength,
                             field );

    // and the whole surface breathing under it
    const float pulse =
        1.0 + globalUniform.lavaPulse * sin( globalUniform.time * globalUniform.lavaPulseSpeed );

    return drift * pulse;
}

// ---------------------------------------------------------------------------
// Doom64-RT: caustics projected from water onto the geometry around it.
//
// A 1-spp path tracer cannot find these. A caustic is a specular-to-diffuse
// path -- light focused by the wavy surface onto a wall -- and the chance of a
// random diffuse bounce landing on the water AND then scattering into a light
// is effectively zero, so the effect never appears no matter how good the
// denoiser is.
//
// So project them. Each shading point fires ONE probe ray straight down; if it
// lands on a surface RTGL considers water within waterCausticDist, the point is
// "over water" and gets an animated caustic field sampled from the same water
// normal texture the surface waves use, so the two agree.
//
// Applied to ALBEDO in the primary pass, NOT to the direct lighting term. Doom
// 64 interiors are lit mostly by INDIRECT light (sector emissives, sky fill),
// and the indirect term is written by RtRaygenIndirectFinal.comp -- a COMPUTE
// shader with no TLAS, which cannot fire this probe. Modulating only the direct
// term therefore computed the effect perfectly and showed nothing: multiplying
// a near-zero direct contribution changes nothing. Albedo scales direct and
// indirect alike.
//
// MULTIPLICATIVE, never additive: caustics are focused light, not a light
// source. A pitch-black room must stay black -- an additive term would make
// water glow in the dark and wash the GI, exactly the failure documented for
// world emissives in compat-patches.md.
// ---------------------------------------------------------------------------

#define WATERPROBE_MISS  0 // nothing below within range
#define WATERPROBE_OTHER 1 // hit geometry, but it is not water
#define WATERPROBE_WATER 2 // hit water

// What is directly below this point? The origin is offset along the normal so a
// point ON a wall probes the floor in FRONT of the wall rather than grazing the
// wall's own base. Returns a WATERPROBE_* value rather than a bool so the
// diagnostic can tell "hit nothing" apart from "hit something that is not
// water" -- different causes, and a bool cannot distinguish them.
//
// Closest hit only, so this cannot leak through geometry: a room above a water
// sector hits its own floor first and reports WATERPROBE_OTHER.
int probeWaterBelow( const vec3  position,
                     const vec3  normal,
                     out   vec3  waterPos,   // world-space point on the water that was hit
                     out   float waterDist,  // receiver -> water distance, metres
                     // Doom64-RT: WHICH liquid was found. A caustic is light
                     // refracted THROUGH a fluid and focused on what is under
                     // it, so an opaque one cannot make any: blood was throwing
                     // swimming-pool light onto the walls around it. The probe
                     // is receiver-side and the receiver has no idea what it is
                     // standing near, so the id has to come back out of here --
                     // and it is free, the flags are already unpacked below.
                     out   uint  liquidId )
{
    liquidId = 0;
    const vec3 up     = globalUniform.worldUpVector.xyz;
    const vec3 origin = position + normal * 0.05;

    // SLANTED, not straight down. A pool almost always has a ledge or walkway
    // around it, so a wall set back behind that ledge probes straight down onto
    // the LEDGE and never sees the water -- which is why the first version lit
    // essentially nothing: the only qualifying surfaces were the few that rise
    // directly out of the water. Tilting the probe along the surface normal
    // lets a wall look "down and out" over the ledge to the water in front of
    // it, which is also where the light physically comes from.
    //
    // A floor (normal == up) or a ceiling (normal == -up) still probes straight
    // down: the normal term only cancels or shortens the vertical component, it
    // never tips those cases sideways.
    const vec3 dir = normalize( -up + normal * globalUniform.waterCausticSlant );

    resetPayload();

    // NO face culling. RTGL culls FRONT faces everywhere else
    // (getAdditionalRayFlags -> gl_RayFlagsCullFrontFacingTrianglesEXT), i.e.
    // its winding is the opposite of the usual convention. This probe first
    // used gl_RayFlagsCullBackFacingTrianglesEXT -- exactly the faces RTGL
    // keeps -- so the ray passed straight THROUGH the water and the effect
    // never fired once. A probe only needs "is there water below", so it culls
    // nothing and cannot be wrong about winding again.
    // rayCullMaskWorld ALONE CANNOT HIT WATER. ASManager completely rewrites the
    // TLAS instance mask for refractive geometry --
    //     if( filter & FT::PT_REFRACT ) instance.mask = INSTANCE_MASK_REFRACT;
    // -- dropping every INSTANCE_MASK_WORLD_* bit, and water is refractive. So a
    // probe using rayCullMaskWorld (WORLD_0|1|2) is not merely unlikely to find
    // water, it is INCAPABLE of intersecting it: no reach, slant or winding
    // change could ever have made it return a hit. RTGL's own refl/refr path
    // says as much: getReflectionRefractionCullMask ORs in INSTANCE_MASK_REFRACT.
    g_payload = traceRayQuery( gl_RayFlagsNoneEXT,
                               globalUniform.rayCullMaskWorld | INSTANCE_MASK_REFRACT,
                               origin,
                               0.001,
                               dir,
                               globalUniform.waterCausticDist );

    waterPos  = position;
    waterDist = 0.0;

    if( !doesPayloadContainHitInfo( g_payload ) )
    {
        return WATERPROBE_MISS;
    }

    int instanceId, instanceCustomIndex;
    unpackInstanceIdAndCustomIndex( g_payload.instIdAndIndex, instanceId, instanceCustomIndex );

    if( ( geometryInstances[ instanceId ].flags & GEOM_INST_FLAG_MEDIA_TYPE_WATER ) == 0 )
    {
        return WATERPROBE_OTHER;
    }

    liquidId = getLiquidId( geometryInstances[ instanceId ].flags );

    // Resolve the exact point on the water. The caustic pattern lives on the
    // WATER surface, so it has to be sampled there and not at the receiver:
    // sampling at the receiver projected onto the horizontal plane means a
    // vertical wall gets identical UVs all the way up, i.e. infinite vertical
    // smearing. Sampling the hit point also makes the pattern foreshorten
    // correctly and gives a distance to fade by.
    int geomIndex, primIndex;
    unpackGeometryAndPrimitiveIndex( g_payload.geomAndPrimIndex, geomIndex, primIndex );

    const mat3 verts = getOnlyCurPositions( instanceId, primIndex );
    const vec3 bary  = vec3( 1.0 - g_payload.baryCoords.x - g_payload.baryCoords.y,
                             g_payload.baryCoords.x,
                             g_payload.baryCoords.y );

    waterPos  = verts * bary;
    waterDist = distance( position, waterPos );

    return WATERPROBE_WATER;
}

// Animated caustic field. Two wave layers scrolling against each other.
float getWaterCaustic( const vec3 position )
{
    const vec3 up = globalUniform.worldUpVector.xyz;

    // Project onto the horizontal plane. This is only correct because the
    // caller passes a point ON THE WATER: projecting the RECEIVER instead gave
    // a vertical wall the same UV at every height, which smeared the pattern
    // into infinite vertical stripes.
    const vec3 flattened = position - up * dot( position, up );
    const mat3 basis     = getONB( up );
    const vec2 xy        = vec2( dot( flattened, basis[ 0 ] ), dot( flattened, basis[ 1 ] ) );

    // world space here is METRES (gzdoom scales by 1/32), so this is UV per
    // METRE. At 0.09 the field repeated every ~11 m = ~350 map units: a whole
    // room inside one caustic cell, which brightens and dims as a single flat
    // wash instead of reading as caustics.
    const float s = globalUniform.waterCausticScale;
    const float t = globalUniform.time * globalUniform.waterCausticSpeed;

    vec2 uv0 = xy * s + vec2( t, t * 0.6 );
    vec2 uv1 = xy * s * 1.3 - vec2( t * 0.7, t );

    // getTextureSampleLod, NOT getTextureSample: a raygen shader has no quad
    // derivatives, so an implicit-LOD texture() fetch is undefined there.
    vec2 n0 = getTextureSampleLod( globalUniform.waterNormalTextureIndex, uv0, 0.0 ).xy * 2.0 - 1.0;
    vec2 n1 = getTextureSampleLod( globalUniform.waterNormalTextureIndex, uv1, 0.0 ).xy * 2.0 - 1.0;

    // Filaments live where the two wave fields CANCEL -- that is where a real
    // water surface acts as a converging lens. length(n0+n1) is ~0.5..1.5 over
    // most of the texture, so the first version (1 - clamp(len), then ^8) was
    // zero almost everywhere: a few sub-pixel dots, which is why it read as
    // "nothing at all" rather than "too subtle". Widen the band, soften the
    // curve, so the filaments have area.
    float converge = 1.0 - clamp( length( n0 + n1 ) * 0.55, 0.0, 1.0 );

    return pow( converge, 3.0 );
}

mat3 lookAt(const vec3 forward, const vec3 worldUp)
{
    vec3 right = cross(forward, worldUp);
    vec3 up = cross(right, forward);

    return mat3(right, up, forward);
}

vec3 getPortalNormal(const vec3 baseNormal, const vec3 inWorldOffset)
{
    if (globalUniform.twirlPortalNormal == 0)
    {
        return -baseNormal;
    }

    float phaseScale = 3;
    float timeScale = 3;
    float waveScale = 0.01;
    float tm = mod(timeScale * globalUniform.time, M_PI * 2);

    const mat3 inLookAt_Plain = lookAt(-baseNormal, globalUniform.worldUpVector.xyz);
    const vec2 localOffset_Plain = vec2(dot(inWorldOffset, inLookAt_Plain[0]), 
                                        dot(inWorldOffset, inLookAt_Plain[1]));

    float distance = length(localOffset_Plain);
    float angle = atan(localOffset_Plain.y, localOffset_Plain.x);

    float phase = sin(phaseScale * sqrt(distance) + angle + tm) + 1.0;
    phase *= waveScale;
    // less weight around center
    phase *= clamp(distance / 20, 0, 1); 

    vec3 localN = { phase, phase, 1.0 };

    return inLookAt_Plain * normalize(localN);
}

bool isBackface( const vec3 normal, const vec3 rayDir )
{
    return dot( normal, -rayDir ) < 0.0;
}

vec3 getNormal( const vec3    position,
                bool          hasNormalMap,
                vec3          normal,
                const RayCone rayCone,
                const vec3    rayDir,
                bool          isWater,
                bool          wasPortal )
{
    // TODO: water normal must be a material property, not refl/refr specific
    if( isWater && !hasNormalMap)
    {
        if( isBackface( normal, rayDir ) )
        {
            normal *= -1;
        }

        normal = getWaterNormal( rayCone, rayDir, normal, position, wasPortal );
        return sanitizeNormal( normal /* TODO: needs to be tri normal */, normal, rayDir );
    }
    else
    {
        if( isBackface( normal, rayDir ) )
        {
            normal *= -1;
        }

        return normalize( normal );
    }
}

#ifdef RAYGEN_PRIMARY_SHADER
vec4 makeNdcCoord(const ivec2 pix_regular, float ndcDepth)
{
    return vec4( ( pix_regular.x + 0.5 ) / float( globalUniform.renderWidth ) * 2.0 - 1.0,
                 ( pix_regular.y + 0.5 ) / float( globalUniform.renderHeight ) * 2.0 - 1.0,
                 ndcDepth,
                 1.0 );
}

bool tryAsFluid( const ivec2 pix, const ivec2 pix_regular, const vec3 cameraRayDir, const float hitDepth )
{
    if( globalUniform.fluidEnabled == 0 )
    {
        return false;
    }

    const float fluidDepthNDC =
        saturate( texelFetch( framebufDepthFluid_Sampler, pix_regular, 0 ).r );
    if( fluidDepthNDC > hitDepth )
    {
        return false;
    }

    // restore world position, motion vectors
    const vec4 ndcCur = makeNdcCoord( pix_regular, fluidDepthNDC );

    vec4 viewSpacePosCur = globalUniform.invProjection * ndcCur;
    viewSpacePosCur /= viewSpacePosCur.w;

          vec3 worldPos         = ( globalUniform.invView * viewSpacePosCur ).xyz;
    const vec4 viewSpacePosPrev = globalUniform.viewPrev * vec4( worldPos, 1.0 );
    const vec4 clipSpacePosPrev = globalUniform.projectionPrev * viewSpacePosPrev;
    const vec3 ndcPrev          = clipSpacePosPrev.xyz / clipSpacePosPrev.w;

    const vec2 screenSpaceCur  = ndcCur.xy * 0.5 + 0.5;
    const vec2 screenSpacePrev = ndcPrev.xy * 0.5 + 0.5;

    const vec2  motionCurToPrev            = ( screenSpacePrev - screenSpaceCur );
    const float fluidDepthLinear           = length( viewSpacePosCur.xyz );
    const float motionDepthLinearCurToPrev = length( viewSpacePosPrev.xyz ) - fluidDepthLinear;

    const vec3 normal =
        decodeNormal( texelFetch( framebufFluidNormal_Sampler, pix_regular, 0 ).r );

    // SHIPPING_HACK: apply some offset to prevent clipping with base surface
    worldPos += -0.005 * cameraRayDir;
    // SHIPPING_HACK

    // clang-format off
    // emulate primary surface
    imageStore( framebufIsSky,              pix,            ivec4( 0 ) );
    imageStore( framebufAlbedo,             pix_regular,    vec4( 1, 1, 1, 0 ) );
    imageStore( framebufScreenEmisRT,       pix_regular,    vec4( 0 ) );
    imageStoreNormal(                       pix,            normal );
    imageStore( framebufMetallicRoughness,  pix,            vec4( 1 /* metallic */, 0 /* roughness */, 0, 0 ) );
    imageStore( framebufDepthWorld,         pix,            vec4( fluidDepthLinear ) );
    imageStore( framebufDepthGrad,          pix,            vec4( 0 ) );
    imageStore( framebufMotion,             pix,            vec4( motionCurToPrev, motionDepthLinearCurToPrev, 0.0 ) );
    imageStore( framebufSurfacePosition,    pix,            vec4( worldPos, 0 /* instCustomIndex */ ) );
    imageStore( framebufVisibilityBuffer,   pix,            packVisibilityBuffer_Invalid() );
    imageStore( framebufViewDirection,      pix,            vec4( cameraRayDir, 0.0 ) );
    imageStore( framebufThroughput,         pix,            vec4( globalUniform.fluidColor.xyz, 0.0 ) );
    imageStore( framebufPrimaryToReflRefr,  pix,            uvec4( GEOM_INST_FLAG_REFRACT | GEOM_INST_FLAG_REFLECT | GEOM_INST_FLAG_MEDIA_TYPE_GLASS, 0 /* no instIdAndIndex */, 0, 0 ) );
    imageStore( framebufDepthNdc,           pix_regular,    vec4( clamp( fluidDepthNDC, 0.0, 1.0 ) ) );
    imageStore( framebufMotionDlss,         pix_regular,    vec4( getMotionVectorForUpscaler( motionCurToPrev ), 0.0, 0.0 ) );
    imageStore( framebufReactivity,         pix_regular,    vec4( 0.0 ) );
    // clang-format on

    return true;
}

void main() 
{
    RAYGEN_RETURN_IF_OUTSIDE( globalUniform.renderWidth, globalUniform.renderHeight, 1 );

    const ivec2 regularPix = ivec2(gl_LaunchIDEXT.xy);
    const ivec2 pix = getCheckerboardPix(regularPix);
    const vec2 inUV = getPixelUVWithJitter(regularPix);

    const vec3 cameraOrigin = globalUniform.cameraPosition.xyz;
    const vec3 cameraRayDir = getRayDir(inUV);
    const vec3 cameraRayDirAX = getRayDirAX(inUV);
    const vec3 cameraRayDirAY = getRayDirAY(inUV);

    // Note: offset a bit, so upscalers do not fail on the edge
    if( classicShading( ivec2( regularPix.x + 16, regularPix.y ) ) ||
        globalUniform.lightmapScreenCoverage >= 0.999 )
    {
        storeSky( pix,
                  cameraRayDir,
                  globalUniform.skyType != SKY_TYPE_RASTERIZED_GEOMETRY,
                  vec3( 1.0 ),
                  1.0 );
        return;
    }


    const ShPayload primaryPayload = tracePrimaryRay(cameraOrigin, cameraRayDir);


    const uint currentRayMedia = globalUniform.cameraMediaType;


    // was no hit
    if (!doesPayloadContainHitInfo(primaryPayload))
    {
        if( tryAsFluid( pix, regularPix, cameraRayDir, 0.999999 ) )
        {
            return;
        }

        vec3 throughput = vec3(1.0);
        // throughput *= getMediaTransmittance(currentRayMedia, pow(abs(dot(cameraRayDir, globalUniform.worldUpVector.xyz)), -3));

        // if sky is a rasterized geometry, it was already rendered to albedo framebuf 
        storeSky( pix,
                  cameraRayDir,
                  globalUniform.skyType != SKY_TYPE_RASTERIZED_GEOMETRY,
                  throughput,
                  1.0 );
        return;
    }


    vec2  motionCurToPrev;
    float motionDepthLinearCurToPrev;
    vec2  gradDepth;
    float firstHitDepthNDC;
    float firstHitDepthLinear;
    vec3  screenEmission;
    float liquidFlow;
    const ShHitInfo h = getHitInfoPrimaryRay( primaryPayload,
                                              cameraOrigin,
                                              cameraRayDir,
                                              cameraRayDirAX,
                                              cameraRayDirAY,
                                              motionCurToPrev,
                                              motionDepthLinearCurToPrev,
                                              gradDepth,
                                              firstHitDepthNDC,
                                              firstHitDepthLinear,
                                              screenEmission,
                                              liquidFlow );

    if( tryAsFluid( pix, regularPix, cameraRayDir, firstHitDepthNDC ) )
    {
        return;
    }

    vec3 throughput = vec3(1.0);
    throughput *= getMediaTransmittance(currentRayMedia, firstHitDepthLinear);


    // Doom64-RT: caustics projected from water onto this surface. Folded into
    // ALBEDO so they show under indirect light too -- see the note above
    // probeWaterBelow().
    vec3 primaryAlbedo = h.albedo; // h is const here

    if( globalUniform.waterCausticGain > 0.0 )
    {
        vec3  waterPos;
        float waterDist;
        // initialised here too: the probe only writes it when `receives` is true
        uint  probeLiquid = 0;

        // Sprites do not receive caustics. A caustic is light thrown ACROSS a
        // surface; a camera-facing billboard has no surface for it to lie on,
        // so the pattern just tints the enemy or the weapon and swims as the
        // camera turns.
        const bool receives =
            ( h.geometryInstanceFlags & GEOM_INST_FLAG_NO_WATER_CAUSTICS ) == 0;

        const int probe =
            receives ? probeWaterBelow( h.hitPosition, h.normal, waterPos, waterDist, probeLiquid )
                     : WATERPROBE_MISS;

        // Two separate falloffs, because they are not the same distance.
        //  - along the probe: light spreads, so a wall at the pool edge gets a
        //    crisp pattern and one set back gets little.
        //  - with HEIGHT above the water: real caustics climb only a little way
        //    up a wall. The probe has to reach much further sideways to clear a
        //    pool's ledge, so sharing one range ran the pattern up the full
        //    height of every wall -- which is exactly what it looked like.
        const float up_h  = dot( h.hitPosition - waterPos, globalUniform.worldUpVector.xyz );
        const float fadeD =
            1.0 - clamp( waterDist / max( 0.001, globalUniform.waterCausticDist ), 0.0, 1.0 );
        const float fadeH =
            1.0 - clamp( max( 0.0, up_h ) / max( 0.001, globalUniform.waterCausticRise ), 0.0, 1.0 );

        // Doom64-RT: per-liquid. A caustic is light refracted through a fluid
        // and focused on the surface beyond it, so how much a liquid throws is
        // a property OF that liquid -- and an opaque one throws none. Blood was
        // casting the same rippling pool-light on the surrounding walls as
        // water, which is the single loudest thing saying "this is water with
        // red paint on it".
        const float liquidCaustics = globalUniform.stylizedLiquidCaustics[ probeLiquid ];

        const float caustic = ( probe == WATERPROBE_WATER )
                                  ? getWaterCaustic( waterPos ) * fadeD * fadeH * fadeH *
                                        liquidCaustics
                                  : 0.0;

        if( globalUniform.stylizedWaterDebug > 0.0 )
        {
            //   black -- ray hit nothing below (out of range, or passed through)
            //   blue  -- hit geometry, but it is not flagged water
            //   green -- over water; brightness is the caustic field itself
            primaryAlbedo = probe == WATERPROBE_WATER ? vec3( 0.0, 0.15 + caustic, 0.0 )
                          : probe == WATERPROBE_OTHER ? vec3( 0.0, 0.0, 0.25 )
                                                      : vec3( 0.0 );
        }
        else
        {
            // Walls get their own gain. The same caustic pattern is far fainter
            // on a vertical surface -- it is seen at a grazing angle and the
            // light reaching it has already spread -- so one gain tuned on the
            // pool bottom leaves the walls invisible, and raising that gain
            // brightens the floor just as much. verticality is 0 on floors and
            // ceilings, 1 on walls.
            const float verticality =
                1.0 - abs( dot( h.normal, globalUniform.worldUpVector.xyz ) );

            const float gain = globalUniform.waterCausticGain *
                               mix( 1.0, globalUniform.waterCausticWallBoost, verticality );

            primaryAlbedo *= 1.0 + gain * caustic;
        }
    }

    // Doom64-RT lava: boost and animate the emission of lava surfaces only.
    // Emission, not albedo -- the albedo is the artist's crust and stays as
    // painted; what moves is how hot it is.
    vec3 primaryEmission = screenEmission;
    if( ( h.geometryInstanceFlags & GEOM_INST_FLAG_LAVA ) != 0 )
    {
        primaryEmission *= globalUniform.lavaEmisBoost * getLavaHeat( h.hitPosition ) *
                           globalUniform.lavaTint.rgb;

        // rt_lava_debug: does the flag survive into the shader at all? A boost
        // that produces no visible change cannot distinguish "the multiply is
        // not happening" from "the emission it multiplies is already zero", and
        // guessing between those cost a round.
        if( globalUniform.lavaDebug > 0.5 )
        {
            primaryAlbedo    = vec3( 0.0 );
            primaryEmission  = vec3( 1.0, 0.0, 1.0 );
        }
    }

    imageStore(framebufIsSky,               pix, ivec4(0));
    // Doom64-RT: the alpha channel carries the liquid flow detail across to the
    // refl/refr pass, which is where the stylized liquid surface is shaded and
    // which cannot sample a material texture for itself. framebufAlbedo.a is
    // free for this: every other store site writes 0 there and every reader --
    // Surface.inl, CmNoisyCompose, CmNrdCompose, CmNrdPack, CmSVGFAtrous, the
    // Ef* passes -- takes .rgb.
    imageStore(framebufAlbedo,              getRegularPixFromCheckerboardPix(pix), vec4(primaryAlbedo, liquidFlow));
    imageStore(framebufScreenEmisRT,        getRegularPixFromCheckerboardPix(pix), vec4(primaryEmission * throughput , 0.0));
    imageStoreNormal(                       pix, h.normal);
    imageStore(framebufMetallicRoughness,   pix, vec4(h.metallic, h.roughness, 0, 0));
    imageStore(framebufDepthWorld,          pix, vec4(firstHitDepthLinear));
    // depth gradients is not 2d, to remove vertical/horizontal artifacts
    imageStore(framebufDepthGrad,           pix, vec4(length(gradDepth)));
    imageStore(framebufMotion,              pix, vec4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0));
    imageStore(framebufSurfacePosition,     pix, vec4(h.hitPosition, uintBitsToFloat(h.instCustomIndex)));
    imageStore(framebufVisibilityBuffer,    pix, packVisibilityBuffer(primaryPayload));
    imageStore(framebufViewDirection,       pix, vec4(cameraRayDir, 0.0));
    imageStore(framebufThroughput,          pix, vec4(throughput, 0.0));

    // save some info for refl/refr shader
    // (.a: the polish mask, see ShHitInfo::polish; the other store sites write 0 there)
    imageStore(framebufPrimaryToReflRefr,   pix, uvec4(h.geometryInstanceFlags, primaryPayload.instIdAndIndex, h.portalIndex, floatBitsToUint(h.polish)));

    // save info for rasterization and upscalers (FSR/DLSS), but only about primary surface,
    // as reflections/refraction only may be losely represented via rasterization
    imageStore(framebufDepthNdc,            getRegularPixFromCheckerboardPix(pix), vec4(clamp(firstHitDepthNDC, 0.0, 1.0)));
    imageStore(framebufMotionDlss,          getRegularPixFromCheckerboardPix(pix), vec4(getMotionVectorForUpscaler(motionCurToPrev), 0.0, 0.0));
    imageStore(framebufReactivity,          getRegularPixFromCheckerboardPix(pix), vec4(0.0));
}
#endif


#ifdef RAYGEN_REFL_REFR_SHADER
void main() 
{
    RAYGEN_RETURN_IF_OUTSIDE( globalUniform.renderWidth, globalUniform.renderHeight, 1 );

    if (globalUniform.reflectRefractMaxDepth == 0)
    {
        return;
    }


    const ivec2 regularPix = ivec2(gl_LaunchIDEXT.xy);
    const ivec2 pix = getCheckerboardPix(regularPix);
    const vec2 inUV = getPixelUVWithJitter(regularPix);

    const vec3 cameraRayDir = getRayDir(inUV);
    
    if (isSkyPix(pix))
    {
        return;
    }



    // restore state from primary shader
    const uvec4 primaryToReflRefrBuf        = texelFetch(framebufPrimaryToReflRefr_Sampler, pix, 0);
    ShHitInfo h;
    const vec4  albedoAndPhase              = texelFetch(framebufAlbedo_Sampler, getRegularPixFromCheckerboardPix(pix), 0);
    h.albedo                                = albedoAndPhase.rgb;
    // Doom64-RT: the liquid flow detail the primary pass advected along the
    // height map's baked direction. See the store site for why it travels in
    // the alpha channel.
    const float liquidFlow                  = albedoAndPhase.a;
    h.hitPosition                           = texelFetch(framebufSurfacePosition_Sampler, pix, 0).xyz;
    h.geometryInstanceFlags                 = primaryToReflRefrBuf.r;
    h.portalIndex                           = primaryToReflRefrBuf.b;
    h.polish                                = uintBitsToFloat(primaryToReflRefrBuf.a);
    h.normal                                = texelFetchNormal(pix);
    {
        vec2 mr                             = texelFetch( framebufMetallicRoughness_Sampler, pix, 0 ).rg;
        h.metallic                          = mr.r;
        h.roughness                         = mr.g;
    }
    const vec3  motionBuf                   = texelFetch(framebufMotion_Sampler, pix, 0).rgb;
    vec2        motionCurToPrev             = motionBuf.rg;
    float       motionDepthLinearCurToPrev  = motionBuf.b;
    float       firstHitDepthLinear         = texelFetch(framebufDepthWorld_Sampler, pix, 0).r;
    vec3        screenEmission              = texelFetch(framebufScreenEmisRT_Sampler, getRegularPixFromCheckerboardPix(pix), 0).rgb;
    vec3        throughput                  = texelFetch(framebufThroughput_Sampler, pix, 0).rgb;
    bool        hasNormalMap                = false; // hack for water normal...
    ShPayload currentPayload;
    currentPayload.instIdAndIndex           = primaryToReflRefrBuf.g;



    RayCone rayCone;
    rayCone.width = 0;
    rayCone.spreadAngle = globalUniform.cameraRayConeSpreadAngle;

    float fullPathLength = firstHitDepthLinear;
    bool wasSplit = false;
    bool wasPortal = false;
    vec3 virtualPos = h.hitPosition;
    vec3 rayDir = cameraRayDir;
    uint currentRayMedia = globalUniform.cameraMediaType;
    // if there was no hitinfo from refl/refr, preserve primary hitinfo
    bool hitInfoWasOverwritten = false;


    propagateRayCone(rayCone, firstHitDepthLinear);



    for (int i = 0; i < globalUniform.reflectRefractMaxDepth; i++)
    {
        const uint instIndex = unpackInstanceIdAndCustomIndex(currentPayload.instIdAndIndex).y;


        bool isPixOdd = isCheckerboardPixOdd(pix) != 0;


        uint newRayMedia =
            getNewRayMedia( i, currentRayMedia, h.geometryInstanceFlags, h.roughness );

        bool isPortal =
            isPortalFromFlags( h.geometryInstanceFlags ) && h.portalIndex != PORTAL_INDEX_NONE;
        bool toRefract = isRefractFromFlags( h.geometryInstanceFlags, h.roughness );
        bool toReflect = isReflectFromFlags( h.geometryInstanceFlags, h.roughness );

        // Doom 64 RT (macOS): a polished surface seen directly (RG_MESH_PRIMITIVE_POLISHED)
        const bool polished = i == 0 && isPolishedFromFlags( h.geometryInstanceFlags ) &&
                              !toReflect && !toRefract && !isPortal;


        // Debug: RTGL1_METAL_DEBUG_POLISH (stylizedWaterDebug 3): magenta -- polished, as bright as
        // its mirror mask; green -- the polished flag, but not polished (it reflects / refracts)
        if( globalUniform.stylizedWaterDebug > 2.5 && i == 0 && isPolishedFromFlags( h.geometryInstanceFlags ) )
        {
            const ivec2 regPix = getRegularPixFromCheckerboardPix( pix );
            imageStore( framebufAlbedo, regPix, vec4( 0.0 ) );
            imageStore( framebufScreenEmisRT,
                        regPix,
                        vec4( polished ? vec3( 1, 0, 1 ) * clamp( h.polish / 3.0, 0.0, 1.0 ) + vec3( 0.02 )
                                       : vec3( 0, 1, 0 ),
                              0.0 ) );
            imageStoreNormal( pix, h.normal );
            imageStore( framebufThroughput, pix, vec4( vec3( 1.0 ), -1.0 ) );
            return;
        }

        if (!toReflect && !toRefract && !isPortal && !polished)
        {
            break;
        }


        const float curIndexOfRefraction = getIndexOfRefraction(currentRayMedia);
        const float newIndexOfRefraction = getIndexOfRefraction(newRayMedia);

        const bool isWater =
            !isPortal && ( newRayMedia == MEDIA_TYPE_WATER || currentRayMedia == MEDIA_TYPE_WATER ||
                           newRayMedia == MEDIA_TYPE_ACID || currentRayMedia == MEDIA_TYPE_ACID );

        const vec3 normal =
            getNormal( h.hitPosition, hasNormalMap, h.normal, rayCone, rayDir, isWater, wasPortal );

        // Doom64-RT: stylized water. Only the FIRST hit, only a vacuum->water
        // crossing (i.e. looking at the surface from above, camera not
        // submerged); everything else keeps the stock physical behaviour.
        const bool stylizedWater = ( globalUniform.stylizedWaterStrength > 0.0 ) &&    //
                                   isWater && i == 0 &&                               //
                                   currentRayMedia == MEDIA_TYPE_VACUUM &&             //
                                   newRayMedia == MEDIA_TYPE_WATER;

        // Doom64-RT diagnostic (rt_water_debug). Three outcomes, one glance:
        //   magenta -- the stylized branch is running
        //   green   -- RTGL sees this surface as water, but the stylized gate
        //              rejected it (media/bounce conditions)
        //   nothing -- the primitive never got RG_MESH_PRIMITIVE_WATER, so the
        //              JSON meta never reached it (tools/set_water_meta.py)
        // Written as screen emission so it is visible with no lighting at all.
        if( globalUniform.stylizedWaterDebug > 0.5 && globalUniform.stylizedWaterDebug < 1.5 && isWater )
        {
            const ivec2 regPix = getRegularPixFromCheckerboardPix( pix );
            imageStore( framebufAlbedo, regPix, vec4( 0.0 ) );
            imageStore( framebufScreenEmisRT,
                        regPix,
                        vec4( stylizedWater ? vec3( 1, 0, 1 ) : vec3( 0, 1, 0 ), 0.0 ) );
            imageStoreNormal( pix, normal );
            // alpha -1: refl/refr WITHOUT a split, so no checkerboard resolve
            imageStore( framebufThroughput, pix, vec4( vec3( 1.0 ), -1.0 ) );
            return;
        }

        vec3  rayOrigin = h.hitPosition;
        bool  doSplit   = !wasSplit;
        bool  doRefraction;
        vec3  refractionDir;
        float F;
        // Doom64-RT: the normal this surface is actually SHADED and REFLECTED
        // with. Identical to `normal` everywhere except a stylized liquid that
        // has been given its material relief back, so every other path is
        // bit-for-bit unchanged.
        vec3  shadeNormal = normal;

        if( polished )
        {
            // As the stylized water below: odd pixels keep the surface (lit as usual, with its
            // own texture and sheen), even pixels are its mirror reflection; CmCheckerboard
            // resolves them into F * reflection + (1 - F) * surface. F has the Schlick shape
            // (weak head-on, strong at grazing angles), remapped so the reflection reads.
            const float cosTheta = clamp( dot( normal, -normalize( rayDir ) ), 0.0, 1.0 );
            // per texel: the _orm roughness masks it (ShHitInfo::polish)
            F = min( mix( 0.15, 0.6, pow( 1.0 - cosTheta, 5.0 ) ) * h.polish, 0.95 );

            if( isPixOdd )
            {
                // *2 compensates the split: this half covers two pixels
                imageStore( framebufThroughput, pix, vec4( throughput * ( 1.0 - F ) * 2.0, 1.0 ) );
                imageStore( framebufReactivity,
                            getRegularPixFromCheckerboardPix( pix ),
                            vec4( UPSCALER_REACTIVITY_REFLREFR ) );
                return;
            }

            // even pixels: the reflection below, weighted by F
            doRefraction = false;
        }
        else if( stylizedWater )
        {
            // Never refract: these are opaque floor flats, there is nothing
            // below them to see. Instead spend the checkerboard split on
            //   odd  pixels: keep the water SURFACE in the G-buffer, so it is
            //                lit like any other opaque surface;
            //   even pixels: the mirror reflection, as usual.
            // CmCheckerboard resolves the two halves into
            //   F * reflection + (1 - F) * lit water surface,
            // which is what makes it read as deep blue looking down and
            // reflective at grazing angles.
            doRefraction = false;

            // Doom64-RT: give the material normal back.
            //
            // getNormal() replaced the normal-mapped normal with the animated
            // water wave, because it takes the wave branch for any water
            // surface whose hasNormalMap is false -- and at the primary hit that
            // flag is hardcoded false (it is only ever assigned for a LATER
            // bounce). So an _n map on a liquid is sampled by the primary pass,
            // written to the G-buffer, and then thrown away here. h.normal still
            // holds it, which is what makes this recoverable at all.
            //
            // relief 0 = the wave, untouched, which is what water and nukage
            // use -- nukage on purpose: poison has no authored _n to give back.
            // 1 = the authored relief alone: a still surface with real ridges,
            // which is what a coagulated blood pool is and what a ripple can
            // never be. Blood and sludge both ship at 1.
            const uint  d64_liquidId = getLiquidId( h.geometryInstanceFlags );
            const float d64_relief   = globalUniform.stylizedLiquidRelief[ d64_liquidId ];
            const vec3  d64_matN     = isBackface( h.normal, rayDir ) ? -h.normal : h.normal;
            shadeNormal              = normalize( mix( normal, d64_matN, d64_relief ) );

            // NOT physical Fresnel. Water's F0 is ~0.02, so a correct Schlick
            // term makes the reflection invisible from anywhere but a grazing
            // angle -- which is exactly why the first version looked like it
            // reflected nothing at all. Keep the SHAPE of the Schlick curve
            // (weak head-on, strong at grazing) but remap its range onto
            // [reflMin, reflMax] so the surface reads as reflective from above.
            {
                const float cosTheta = clamp( dot( shadeNormal, -normalize( rayDir ) ), 0.0, 1.0 );
                const float curve    = pow( 1.0 - cosTheta, 5.0 );
                F                    = mix( globalUniform.stylizedWaterReflMin,
                                            globalUniform.stylizedWaterReflMax,
                                            curve );
                // Doom64-RT: per-liquid reflection. A mirror is what sells
                // WATER; on an opaque mud bed it is the single loudest thing
                // saying "this is water with brown paint on it". Scaling F
                // here does both halves of the checkerboard at once -- the
                // even pixels' mirror ray is weighted by F and the odd
                // pixels' surface by (1 - F) -- so the light the surface
                // loses to the reflection comes straight back to the diffuse
                // shading instead of vanishing.
                F *= globalUniform.stylizedLiquidRefl[ d64_liquidId ];
                F                    = clamp( F, 0.0, 1.0 );
            }

            // Doom64-RT: an OPAQUE BED does not split. stylizedLiquidRefl 0
            // means "no mirror at all": the surface is shaded on EVERY pixel,
            // full resolution, no checkerboard, and its wet sheen comes from
            // the standard glossy specular off the roughness written below --
            // the lighter, dedicated reflection a mud bed wants.
            //
            // This is also a fix, not only a look. The split shades the lit
            // surface on odd screen columns only and rebuilds the even ones as
            // a 4-neighbour average, and the denoiser reprojects history in
            // that half-resolution space. Stock water has a smooth wave normal
            // and never noticed. On a high-contrast authored normal, every
            // texel alternates between "shaded" and "averaged from its
            // neighbours" as it crosses columns while the camera moves: a
            // per-texel contrast pulse that scales with slope amplitude,
            // survives the denoiser, ignores parallax and the upscaler, and
            // freezes into a stable pattern the moment the camera stops.
            // Which is exactly the flashlight bug, and every bisect arm
            // (nomaps clean, softnormal halved, flat and nodlss unchanged,
            // visible in denoised direct diffuse) agrees with it.
            // liquidNoSplit is the Options > Quality "Liquid surfaces" item: it
            // takes every liquid down this path, water's mirror included.
            const bool d64_noSplit = globalUniform.liquidNoSplit > 0.5 ||
                                     globalUniform.stylizedLiquidRefl[ d64_liquidId ] <= 0.0;

            if( isPixOdd || d64_noSplit )
            {
                float caustic;
                // the base normal must be the one getNormal() actually built
                // the waves around, or the wave-tilt term reads ~2 everywhere
                const vec3 baseNormal = d64_matN;
                const uint liquidId   = d64_liquidId;
                const vec3 surfAlbedo = getStylizedWaterAlbedo(
                    h.albedo, normal, baseNormal, liquidId, liquidFlow, caustic );

                // *2 compensates the split: this half covers two pixels.
                // No split, no compensation and nothing given to a mirror.
                throughput *= d64_noSplit ? 1.0 : ( 1.0 - F ) * 2.0;

                // a little unlit sheen so the caustic pattern still reads in
                // rooms the path tracer leaves nearly black (the original flat
                // was drawn bright); purely on-screen, casts no light
                const vec3 sheen = getLiquidCrestColor( liquidId ) * caustic *
                                   globalUniform.stylizedWaterGlow;

                const ivec2 regPix = getRegularPixFromCheckerboardPix( pix );

                // Keep position / depth / motion / visibility from the primary
                // pass -- they already describe this exact surface. Only the
                // shading inputs change.
                // rt_blood_flow_debug: paint the ADVECTED DETAIL, so "the bake
                // and the plumbing worked" is separable from "the motion is too
                // subtle to see" -- by eye those are the same picture. Green
                // blobs sliding along the veins = the flow map is live. Flat
                // blue = the direction never reached the shader, or the detail
                // never crossed framebufAlbedo.a; either way, tuning cannot help.
                if( globalUniform.liquidFlowDebug > 0.5 )
                {
                    // Instrumented after the advection measured DEAD STATIC in a
                    // burst capture while every plumbing stage checked out:
                    //   RED   = fract(time/4)  -- must visibly change second to
                    //           second, or the time uniform itself is frozen
                    //   GREEN = the advected detail (the actual flow debug)
                    //   BLUE  = the speed the shader sees, /50 -- near-black at
                    //           the shipping 0.3, saturated if a test "60" pin
                    //           arrives. Tells "speed never arrived" from "time
                    //           is dead" in one frame-pair.
                    const float timeBeat = fract( globalUniform.time * 0.25 );
                    const float speedTint =
                        clamp( globalUniform.liquidFlowSpeed * 0.02, 0.0, 1.0 );
                    imageStore( framebufAlbedo, regPix, vec4( 0.0 ) );
                    imageStore( framebufScreenEmisRT,
                                regPix,
                                liquidFlow > 0.0
                                    ? vec4( timeBeat, liquidFlow, speedTint, 0.0 )
                                    : vec4( timeBeat, 0.0, 0.15 + speedTint, 0.0 ) );
                    imageStoreNormal( pix, shadeNormal );
                    imageStore( framebufThroughput, pix, vec4( vec3( 1.0 ), -1.0 ) );
                    imageStore( framebufReactivity, regPix, vec4( UPSCALER_REACTIVITY_REFLREFR ) );
                    return;
                }

                imageStore( framebufAlbedo, regPix, vec4( surfAlbedo, 0.0 ) );
                imageStore( framebufScreenEmisRT, regPix, vec4( screenEmission + sheen, 0.0 ) );
                imageStoreNormal( pix, shadeNormal );
                // Per-liquid roughness, <= 0 meaning "keep the global". The
                // reflection RAY is a mirror off shadeNormal regardless; this
                // is what the denoiser and any later bounce see, and it is
                // what stops a rough liquid being resolved as a sharp one.
                const float d64_rough = globalUniform.stylizedLiquidRough[ liquidId ] > 0.0
                                            ? globalUniform.stylizedLiquidRough[ liquidId ]
                                            : globalUniform.stylizedWaterRoughness;
                imageStore( framebufMetallicRoughness, pix, vec4( 0.0, d64_rough, 0, 0 ) );
                // alpha == 1: was refl/refr WITH a split -> resolve checkerboard
                // alpha == -1: no split -> CmCheckerboard leaves the pixel alone
                imageStore( framebufThroughput, pix, vec4( throughput, d64_noSplit ? -1.0 : 1.0 ) );
                // Doom64-RT: THE FIX. Every other exit from this shader marks
                // its reactivity (see storeSky and the hitInfoWasOverwritten
                // path at the bottom of this file); this early return was the
                // one place that did not, and it is exactly the return this
                // liquid surface always takes. Unmarked reads as "static,
                // trust history" to DLSS's temporal upscaler, which is active
                // in every configuration this was tested under (DLSS2 Super
                // Resolution, independent of A-SVGF handling the denoise) --
                // so a flow signal that changes COLOUR ONLY, with a static
                // normal (relief pins it) and zero motion vector, gets
                // averaged toward its time-mean before it reaches the screen.
                // That average is a brighter, static crest: exactly the
                // symptom reported, on both the phase-pulse and the flow-map
                // versions, because neither ever set this.
                //
                // With no split and no flow this is an ordinary static opaque
                // surface, and telling the upscaler to distrust its history
                // would only cost it anti-aliasing. Mark it like one.
                const bool d64_plainOpaque =
                    d64_noSplit && globalUniform.stylizedLiquidFlow[ liquidId ] <= 0.0;
                imageStore( framebufReactivity,
                            regPix,
                            vec4( d64_plainOpaque ? 0.0 : UPSCALER_REACTIVITY_REFLREFR ) );
                return;
            }

            // even pixels fall through to the reflection branch below
        }
        else if (toRefract && calcRefractionDirection(curIndexOfRefraction, newIndexOfRefraction, rayDir, normal, refractionDir))
        {
            doRefraction = isPixOdd;
            F = getFresnelSchlick(curIndexOfRefraction, newIndexOfRefraction, -rayDir, normal);
        }
        else
        {
            // total internal reflection
            doRefraction = false;
            doSplit = false;
            F = 1.0;
        }
        
        if (doRefraction)
        {
            rayDir = refractionDir;
            throughput *= (1 - F);

            if( ( h.geometryInstanceFlags & GEOM_INST_FLAG_THIN_MEDIA ) != 0 )
            {
                // simulate media, knowing its width along normal
                float len = globalUniform.thinMediaWidth / max( 0.001, -dot( normal, rayDir ) );

                rayOrigin += rayDir * len;
                throughput *= getMediaTransmittance( newRayMedia, len );
                fullPathLength += len;
                
                // change media back
                if( calcRefractionDirection( newIndexOfRefraction,
                                             curIndexOfRefraction,
                                             rayDir,
                                             normal,
                                             refractionDir ) )
                {
                    rayDir = refractionDir;
                }
                newRayMedia = currentRayMedia;
            }

            // change media
            currentRayMedia = newRayMedia;
        }
        else if (isPortal)
        {
            const ShPortalInstance portal = g_portals[h.portalIndex];

            const vec3 inCenter = portal.inPosition.xyz;
            const vec3 inWorldOffset = h.hitPosition - inCenter;

            mat3 inLookAt = lookAt(getPortalNormal(normal, inWorldOffset), globalUniform.worldUpVector.xyz);

            const vec3 outCenter = portal.outPosition.xyz;
            const mat3 outLookAt = lookAt(portal.outDirection.xyz, 
                                          portal.outUp.xyz);

            // to local space; then to world space but at portal output
            rayDir = outLookAt * (transpose(inLookAt) * rayDir);

            const vec2 localOffset = vec2(dot(inWorldOffset, inLookAt[0]), 
                                          dot(inWorldOffset, inLookAt[1]));

            rayOrigin = outCenter + localOffset.x * outLookAt[0] + localOffset.y * outLookAt[1];

            wasPortal = true;
        }
        else
        {
            rayDir = reflect( normalize( rayDir ), shadeNormal );

            if( !isWater && !polished )
            {
                throughput *= getFresnelSchlick( max( 0, dot( normal, rayDir ) ),
                                                 getSpecularColor( h.albedo, h.metallic ) );
            }
            else
            {
                throughput *= F;

    #if SHIPPING_HACK
                rayOrigin += rayDir * 0.005;
    #endif
            }
        }

        if (doSplit)
        {
            throughput *= 2;
            wasSplit = true;
        }


        currentPayload = traceReflectionRefractionRay(rayOrigin, rayDir, instIndex, h.geometryInstanceFlags, doRefraction);

        
        if (!doesPayloadContainHitInfo(currentPayload))
        {
            throughput *= getMediaTransmittance(currentRayMedia, pow(abs(dot(rayDir, globalUniform.worldUpVector.xyz)), -3));

            storeSky(pix, rayDir, true, throughput, wasSplit);
            return;  
        }

        float rayLen;
        vec3 scrEmis;

        h = getHitInfoWithRayCone_ReflectionRefraction(
            currentPayload, rayCone, 
            rayOrigin, rayDir, cameraRayDir, 
            virtualPos, 
            rayLen, 
            motionCurToPrev, motionDepthLinearCurToPrev,
            scrEmis,
            hasNormalMap
        );


        hitInfoWasOverwritten = true;
        throughput *= getMediaTransmittance(currentRayMedia, rayLen);
        propagateRayCone(rayCone, rayLen);
        fullPathLength += rayLen;
        screenEmission += scrEmis * throughput;
    }


    if (!hitInfoWasOverwritten)
    {
        return;
    }


    imageStore(framebufIsSky,               pix, ivec4(0));
    imageStore(framebufAlbedo,              getRegularPixFromCheckerboardPix(pix), vec4(h.albedo, 0.0));
    imageStore(framebufScreenEmisRT,        getRegularPixFromCheckerboardPix(pix), vec4(screenEmission, 0.0));
    imageStoreNormal(                       pix, h.normal);
    imageStore(framebufMetallicRoughness,   pix, vec4(h.metallic, h.roughness, 0, 0));
    imageStore(framebufDepthWorld,          pix, vec4(fullPathLength));
    imageStore(framebufMotion,              pix, vec4(motionCurToPrev, motionDepthLinearCurToPrev, 0.0));
    imageStore(framebufSurfacePosition,     pix, vec4(h.hitPosition, uintBitsToFloat(h.instCustomIndex)));
    imageStore(framebufVisibilityBuffer,    pix, packVisibilityBuffer(currentPayload));
    imageStore(framebufViewDirection,       pix, vec4(rayDir, 0.0));
    imageStore(framebufThroughput,          pix, vec4(throughput, wasSplit ? 1.0 : -1.0));
    imageStore(framebufReactivity,          getRegularPixFromCheckerboardPix(pix), vec4(UPSCALER_REACTIVITY_REFLREFR));
}
#endif