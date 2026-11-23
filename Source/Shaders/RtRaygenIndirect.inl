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

// For indirect sample
//#extension GL_EXT_shader_16bit_storage : require
//#extension GL_EXT_shader_explicit_arithmetic_types : require

layout (constant_id = 0) const uint maxAlbedoLayerCount = 0;
layout (constant_id = 1) const uint lightmapLayerIndex = 3;
#define MATERIAL_MAX_ALBEDO_LAYERS maxAlbedoLayerCount
#define MATERIAL_LIGHTMAP_LAYER_INDEX lightmapLayerIndex

#define FORCE_EVALBRDF_GGX_LOOSE


#ifndef RT_FORCE_COMPUTE

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
#define DESC_SET_RESTIR_INDIRECT 10
#define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_INDIR)
#include "RaygenCommon.h"
#include "ReservoirIndirect.h"

#else // RT_FORCE_COMPUTE

    #define DESC_SET_FRAMEBUFFERS 1
    #define DESC_SET_GLOBAL_UNIFORM 2
    #define DESC_SET_RESTIR_INDIRECT 10
    #define LIGHT_SAMPLE_METHOD (LIGHT_SAMPLE_METHOD_INDIR)
    #include "ShaderCommonGLSLFunc.h"
    #include "Surface.inl"
    #include "Light.h"
    #include "ReservoirIndirect.h"
    layout( local_size_x = COMPUTE_INDIRECT_FINAL_GROUP_SIZE_X,
            local_size_y = COMPUTE_INDIRECT_FINAL_GROUP_SIZE_Y,
            local_size_z = 1 ) in;

#endif // !RT_FORCE_COMPUTE


#define OPTIMIZE_MEM 1

bool useDiffuse( const float roughness )
{
    return roughness >= FAKE_ROUGH_SPECULAR_THRESHOLD;
}

float getDiffuseWeight( float roughness )
{
    return smoothstep( MIN_GGX_ROUGHNESS,
                       FAKE_ROUGH_SPECULAR_THRESHOLD + FAKE_ROUGH_SPECULAR_LENGTH,
                       roughness );
}

// v -- direction to viewer
// n -- surface normal
vec3 getSpecularBounce(const uint seed, uint bounceIndex,
                       const vec3 n, const float roughness, const vec3 surfSpecularColor,
                       const vec3 v, 
                       out float oneOverSourcePdf)
{
#if OPTIMIZE_MEM
    const vec2 u = rnd8_4( seed, RANDOM_SALT_SPEC_BOUNCE( bounceIndex ) ).xy;
#else
    const vec2 u = rndBlueNoise16_2( seed, RANDOM_SALT_SPEC_BOUNCE( bounceIndex ) );
#endif
    return sampleSmithGGX( n, v, roughness, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

// n -- surface normal
vec3 getDiffuseBounce(const uint seed, uint bounceIndex, const vec3 n, out float oneOverSourcePdf)
{
#if OPTIMIZE_MEM
    const vec2 u = rnd8_4( seed, RANDOM_SALT_DIFF_BOUNCE( bounceIndex ) ).xy;
#else
    const vec2 u = rndBlueNoise16_2( seed, RANDOM_SALT_DIFF_BOUNCE( bounceIndex ) );
#endif
    return sampleLambertian( n, u[ 0 ], u[ 1 ], oneOverSourcePdf );
}

#ifndef RT_FORCE_COMPUTE

#define FIRST_BOUNCE_MIP_BIAS 0
#define SECOND_BOUNCE_MIP_BIAS 32

Surface traceBounce(const vec3 originPosition, float originRoughness, uint originInstCustomIndex,
                    const vec3 bounceDir, float bounceMipBias, out vec3 out_emission)
{
    const ShPayload p = traceIndirectRay(originInstCustomIndex, originPosition, bounceDir); 

    if (!doesPayloadContainHitInfo(p))
    {
        Surface s;
        s.isSky = true;
        return s;
    }

    return hitInfoToSurface_Indirect(
        getHitInfoBounce(p, originPosition, originRoughness, bounceMipBias, out_emission), 
        bounceDir);
}

// Doom64-RT: seed for bounce vertex b of one indirect path.
//
// Two things go wrong past depth 2 on the stock seed. RANDOM_SALT_DIFF_BOUNCE
// (Random.h) has one free index before it runs into the specular band, and
// processDirectIllumination draws its light-selection numbers from the SAME
// salts at every vertex -- so every vertex on a path picks its light with the
// same random numbers, which at depth 3+ correlates the whole path into
// structured blotches no denoiser averages out.
//
// b <= 2 returns the stock seed untouched, so depth 2 is bit-identical to the
// unrolled pair this replaced. Deeper vertices get the "virtual frame"
// treatment the multi-sample loop in main() already uses: getRandomSeed
// murmur-hashes the RAW (pix, frame) input before reducing it, so any distinct
// input is an independent seed. si*7919 + b*24593 is distinct for every
// (si, b) with si < 8, b <= 4 -- a collision needs 7919*dsi == 24593*db.
uint bounceSeed( const ivec2 pix, uint virtualFrame, uint seed, uint b )
{
    return ( b <= 2u ) ? seed : getRandomSeed( pix, virtualFrame + b * 24593u );
}

SampleIndirect processIndirect( const ivec2 pix, uint virtualFrame, const uint seed, const Surface surf, out float oneOverSourcePdf )
{
    vec3 bounceDir;

    if( useDiffuse( surf.roughness ) )
    {
        bounceDir = getDiffuseBounce( seed, 1, surf.normal, oneOverSourcePdf );
    }
    else
    {
        bounceDir = getSpecularBounce( seed,
                                    1,
                                    surf.normal,
                                    surf.roughness,
                                    surf.specularColor,
                                    surf.toViewerDir,
                                    oneOverSourcePdf );

        // swap to the common domain
        float oneOverDiffusePdf;
        {
            float z           = dot( bounceDir, surf.normal );
            oneOverDiffusePdf = z / M_PI;
        }
        oneOverSourcePdf *= oneOverDiffusePdf;
    }

    vec3 emis;
    const Surface hitSurf = traceBounce(surf.position + surf.normal * 0.01, 
                                        surf.roughness, 
                                        surf.instCustomIndex, 
                                        bounceDir, 
                                        FIRST_BOUNCE_MIP_BIAS,
                                        emis);
    emis *= globalUniform.emissionMapBoost;

    if (hitSurf.isSky)
    {
        SampleIndirect s = createSampleIndirect( //
            surf.position + bounceDir * MAX_RAY_LENGTH,
            -bounceDir,
            getSky( bounceDir ) );
        return s;
    }

    // calculate direct diffuse illumination in a hit position
    vec3 diffuse = processDirectIllumination(seed, hitSurf, 1);

    // Doom64-RT: vertices 2..N. This replaces the unrolled second bounce that
    // sat behind a commented-out gate ("TODO: investigate why uncommenting this
    // makes diffuse very red"), so it always ran and its API flag was inert.
    //
    // The stored SampleIndirect stays vertex 1 -- position, normal, and the
    // reservoir weight oneOverSourcePdf all belong to the first hit, because
    // the spatial-reuse Jacobian and shade() reconnect to THAT vertex. Deeper
    // bounces have nowhere to live but vertex 1's radiance, so they are folded
    // in here with their own throughput, exactly as the old second bounce was.
    //
    // Why the second bounce was "very red": for a cosine-sampled Lambertian
    // the per-bounce throughput is BRDF(1/pi) * cos * (pi/cos) = exactly 1,
    // albedo applied separately. The stock code multiplied by 1/pdf alone,
    // i.e. pi/cos too much -- mean ~2pi under the cosine density, with a 1/cos
    // tail -- so bounce 2 arrived ~6x too bright, ~6x too saturated (it is
    // then multiplied by albedo), and firefly-prone. And because the RIS
    // target pdf is the sample's luminance, those samples WON the reservoir
    // and were held by temporal reuse. indirectLegacyWeight reproduces that,
    // and only that, so depth 2 ships unchanged until the fix is judged.
    Surface prev       = hitSurf;
    vec3    throughput = vec3( 1.0 ); // demodulated, relative to vertex 1

    for( uint b = 2u; b <= globalUniform.indirectBounces; b++ )
    {
        const uint bseed = bounceSeed( pix, virtualFrame, seed, b );

        // min(b, 3): salt index 2 at b == 2 (stock), index 3 reused for every
        // deeper vertex under a DIFFERENT seed -- index 4 would alias the
        // specular band (RANDOM_SALT_SPEC_BOUNCE(0) == 12).
        float      oneOverPdf_b;
        const vec3 dir = getDiffuseBounce( bseed, min( b, 3u ), prev.normal, oneOverPdf_b );
        const float w  = ( globalUniform.indirectLegacyWeight != 0u ) ? oneOverPdf_b : 1.0;

        vec3 emis_b;
        const Surface hit = traceBounce( prev.position + prev.normal * 0.01,
                                         prev.roughness,
                                         prev.instCustomIndex,
                                         dir,
                                         SECOND_BOUNCE_MIP_BIAS,
                                         emis_b );

        if( hit.isSky )
        {
            diffuse += throughput * w * getSky( dir );
            break;
        }

        emis_b *= globalUniform.emissionMapBoost;

        // calculate direct diffuse illumination in the hit position
        const vec3 Lout = ( emis_b + processDirectIllumination( bseed, hit, int( b ) ) ) * hit.albedo;

        diffuse    += throughput * w * Lout;
        throughput *= w * hit.albedo;
        prev        = hit;
    }

    SampleIndirect s = createSampleIndirect( //
        hitSurf.position,
        hitSurf.normal,
        ( emis + diffuse ) * hitSurf.albedo );
    return s;
}
#endif // !RT_FORCE_COMPUTE

vec3 shade(const Surface surf, const SampleIndirect indir, float oneOverPdf)
{
    vec3  l  = safeNormalize2( unpackSampleIndirectPosition( indir ) - surf.position, vec3( 0 ) );
    float nl = dot(surf.normal, l);

    if (nl <= 0)
    {
        return vec3(0);
    }

    const vec3 radiance = decodeE5B9G9R9( indir.radianceE5 );

    if( useDiffuse( surf.roughness ) )
    {
        return oneOverPdf * nl * radiance * evalBRDFLambertian(1.0);
    }
    else
    {
        return oneOverPdf * nl * radiance * evalBRDFSmithGGX(surf.normal, surf.toViewerDir, l, surf.roughness, surf.specularColor);
    }
}

float targetPdfForIndirectSample(const SampleIndirect s)
{
    return getLuminance( decodeE5B9G9R9( s.radianceE5 ) );
}

bool testSurfaceForReuseIndirect(
    const ivec3 curChRenderArea, const ivec2 otherPix,
    float curDepth, float otherDepth,
    const vec3 curNormal, const vec3 otherNormal)
{
    const float DepthThreshold = 0.05;
    const float NormalThreshold = 0.0;

    return 
        testPixInRenderArea(otherPix, curChRenderArea) &&
        (abs(curDepth - otherDepth) / abs(curDepth) < DepthThreshold) &&
        (dot(curNormal, otherNormal) > NormalThreshold);
}



#define TEMPORAL_SAMPLES_INDIR    1
#define TEMPORAL_RADIUS_INDIR_MAX 8.0

#define SPATIAL_SAMPLES_INDIR 2
#define SPATIAL_RADIUS_INDIR  mix( 2.0, 8.0, clamp( globalUniform.renderHeight / 1080.0, 0.0, 1.0 ) ) 

#define DEBUG_TRACE_BIAS_CORRECT_RAY 0



#ifdef RT_RAYGEN_INDIRECT_INIT
void main()
{
    RAYGEN_RETURN_IF_OUTSIDE( globalUniform.renderWidth, globalUniform.renderHeight, 1 );

    const ivec2 pix = ivec2(gl_LaunchIDEXT.xy);
    const uint seed = getRandomSeed(pix, globalUniform.frameId);
    uint salt = RANDOM_SALT_RESAMPLE_INDIRECT_BASE;

    Surface surf = fetchGbufferSurface(pix);
    surf.position += surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;

    if (surf.isSky)
    {
        restirIndirect_StoreInitialSample( pix, emptySampleIndirect(), 0.0 );
        return;
    }

    // Multi-sample indirect (GI).
    //
    // One path per pixel is the dominant GI noise source, and like direct
    // lighting it is only hidden by temporal accumulation -- which motion
    // destroys. N independent paths, RIS-combined, cut that variance at the
    // source for both denoisers.
    //
    // RIS rather than a plain average because the storage format holds exactly
    // one (sample, weight) pair, and neighbours read this image during spatial
    // reuse -- so it must stay a single well-formed sample, not a blend of
    // several with mismatched positions/normals.
    const uint N = max( globalUniform.indirectSamples, 1u );

    if( N == 1u )
    {
        // stock path, kept verbatim so N=1 is bit-identical
        float          oneOverSourcePdf;
        SampleIndirect initial = processIndirect( pix, globalUniform.frameId, seed, surf, oneOverSourcePdf );

        restirIndirect_StoreInitialSample( pix, initial, oneOverSourcePdf );
        return;
    }

    ReservoirIndirect ris = emptyReservoirIndirect();

    for( uint si = 0; si < N; si++ )
    {
        // A fresh well-formed seed per sample rather than arithmetic on the
        // packed one: getRandomSeed re-hashes through murmur, so a "virtual
        // frame" index gives an independent path without corrupting the
        // blue-noise texture index/offset packing. si == 0 reuses the real seed.
        const uint virtualFrame = globalUniform.frameId + si * 7919u;
        const uint sampleSeed   = ( si == 0u ) ? seed : getRandomSeed( pix, virtualFrame );

        float          oneOverSourcePdf;
        SampleIndirect s = processIndirect( pix, virtualFrame, sampleSeed, surf, oneOverSourcePdf );

        const float targetPdf = targetPdfForIndirectSample( s );
        const float rndRis    = rnd16( seed, RANDOM_SALT_INDIRECT_SPP_BASE + si );

        updateReservoirIndirect( ris, s, targetPdf, oneOverSourcePdf, rndRis );
    }

    // Collapse the N-candidate reservoir back into the (sample, weight) pair the
    // storage format holds. The final pass rebuilds an M=1 reservoir whose
    // estimator evaluates to radiance * storedWeight, so the stored weight must
    // be the RIS weight -- (1/targetPdf_selected) * weightSum / N. Storing a raw
    // oneOverSourcePdf here instead would bias GI brightness with N.
    const float risWeight = calcSelectedSampleWeightIndirect( ris );

    restirIndirect_StoreInitialSample( pix, ris.selected, risWeight );
}
#endif // RT_RAYGEN_INDIRECT_INIT



#ifdef RT_RAYGEN_INDIRECT_FINAL
ReservoirIndirect loadInitialSampleAsReservoir( const ivec2 pix )
{
    float          oneOverSourcePdf;
    SampleIndirect s         = restirIndirect_LoadInitialSample( pix, oneOverSourcePdf );
    float          targetPdf = targetPdfForIndirectSample( s );

    ReservoirIndirect r = emptyReservoirIndirect();
    updateReservoirIndirect( r, s, targetPdf, oneOverSourcePdf, 0.5 );
    return r;
}

void main()
{
#ifndef RT_FORCE_COMPUTE
    const ivec2 pix  = ivec2( gl_LaunchIDEXT.xy );
#else
    const ivec2 pix = ivec2( gl_GlobalInvocationID.xy );
#endif
    const uint  seed = getRandomSeed( pix, globalUniform.frameId );
    uint        salt = RANDOM_SALT_RESAMPLE_INDIRECT_BASE;

    Surface surf = fetchGbufferSurface( pix );
#ifndef RT_FORCE_COMPUTE
    surf.position += surf.toViewerDir * RAY_ORIGIN_LEAK_BIAS;
#endif

    if( surf.isSky )
    {
        return;
    }


    ReservoirIndirect combined = loadInitialSampleAsReservoir( pix );


    // assuming that pix is checkerboarded
    const ivec3 chRenderArea = getCheckerboardedRenderArea( pix );
    const float motionZ           = texelFetch( framebufMotion_Sampler, pix, 0 ).z;
    const float depthCur          = texelFetch( framebufDepthWorld_Sampler, pix, 0 ).r;
    const vec2  posPrev           = getPrevScreenPos( framebufMotion_Sampler, pix );

    int spatialSamplesCount = int( SPATIAL_SAMPLES_INDIR * getDiffuseWeight( surf.roughness ) );


    for( int pixIndex = 0; pixIndex < TEMPORAL_SAMPLES_INDIR; pixIndex++ )
    {
        // TODO: need low discrepancy noise
        ivec2 pp;
        {
            vec2 rndOffset = rnd8_4( seed, salt++ ).xy * 2.0 - 1.0;
            rndOffset *= square( getDiffuseWeight( surf.roughness ) );

            pp = ivec2( floor( posPrev +
                               rndOffset * ( pixIndex == 0 ? 0.5 : TEMPORAL_RADIUS_INDIR_MAX ) ) );
        }

        {
            if( isSkyPix( pp ) )
            {
                continue;
            }
        }
        {
            const float depthPrev  = texelFetch( framebufDepthWorld_Prev_Sampler, pp, 0 ).r;
            const vec3  normalPrev = texelFetchNormal_Prev( pp );

            if( !testSurfaceForReuseIndirect(
                    chRenderArea, pp, depthCur, depthPrev - motionZ, surf.normal, normalPrev ) )
            {
                continue;
            }
        }
        {
            // framebufDISGradientHistory is written ONLY by CmASVGFGradientAtrous,
            // which runs only inside Denoiser::Denoise(). DLSS-RR skips Denoise()
            // and calls ComposeNoisy() instead, so under RR this gate reads a
            // buffer that nothing updates -- and rejecting on a stale value would
            // kill indirect temporal reuse outright, leaving GI at 1 spp with no
            // accumulation. restirIndirAntilag 0 ignores the gate; see the uniform
            // comment in GenerateShaderCommon.py.
            if( globalUniform.restirIndirAntilag != 0 )
            {
                const float antilagAlpha_Indir = texelFetch(
                    framebufDISGradientHistory_Sampler, pp / COMPUTE_ASVGF_STRATA_SIZE, 0 )[ 1 ];

                // if there's too much difference, don't use a temporal sample
                if( antilagAlpha_Indir > 0.25 )
                {
                    continue;
                }
            }
        }

        ReservoirIndirect temporal = restirIndirect_LoadReservoir_Prev( pp );
        // renormalize to prevent precision problems
        normalizeReservoirIndirect( temporal, 20 );

        float rnd = rnd16( seed, salt++ );
        updateCombinedReservoirIndirect( combined, temporal, rnd );

        break;
    }



    {
        uint nobiasM = combined.M; 

        for( int pixIndex = 0; pixIndex < spatialSamplesCount; pixIndex++ )
        {
            // TODO: need low discrepancy noise
            ivec2 pp;
            {
#if OPTIMIZE_MEM
                vec2 rndOffset = rnd16_2( seed, salt++ ) * 2.0 - 1.0;
#else
                vec2 rndOffset = rndBlueNoise16_2( seed, salt++ ) * 2.0 - 1.0;
#endif
                pp = pix + ivec2( rndOffset * SPATIAL_RADIUS_INDIR );
            }

            {
                if( isSkyPix( pp ) )
                {
                    continue;
                }

                const float depthOther  = texelFetch( framebufDepthWorld_Sampler, pp, 0 ).r;
                const vec3  normalOther = texelFetchNormal( pp );

                if( !testSurfaceForReuseIndirect(
                        chRenderArea, pp, depthCur, depthOther, surf.normal, normalOther ) )
                {
                    continue;
                }
            }

            ReservoirIndirect reservoir_q = loadInitialSampleAsReservoir( pp );

            float oneOverJacobian;
            {
                const vec3 x1_r = surf.position;
                const vec3 x1_q = texelFetch( framebufSurfacePosition_Sampler, pp, 0 ).xyz;

                const vec3 x2_q = unpackSampleIndirectPosition( reservoir_q.selected );
                const vec3 n2_q = decodeNormal( reservoir_q.selected.normalPacked );

                const DirectionAndLength phi_r = calcDirectionAndLengthSafe( x2_q, x1_r );
                const DirectionAndLength phi_q = calcDirectionAndLengthSafe( x2_q, x1_q );

                oneOverJacobian =
                    safePositiveRcp( getGeometryFactorClamped( n2_q, phi_r.dir, phi_r.len ) ) *
                    getGeometryFactorClamped( n2_q, phi_q.dir, phi_q.len );

    #if SHIPPING_HACK
                oneOverJacobian = clamp( oneOverJacobian, 0.0, 1.0 );
    #endif
            }

            float targetPdf_curSurf = 0.0;

#if DEBUG_TRACE_BIAS_CORRECT_RAY
            if( !traceShadowRay(
                    surf.instCustomIndex, surf.position, reservoir_q.selected.position, false ) )
#endif
            {
                targetPdf_curSurf =
                    targetPdfForIndirectSample( reservoir_q.selected ) * oneOverJacobian;
            }
#if OPTIMIZE_MEM
            float rnd = rnd16( seed, salt++ );
#else
            float rnd = rndBlueNoise32( seed, salt++ );
#endif
            updateCombinedReservoirIndirect_newSurf(
                combined, reservoir_q, targetPdf_curSurf, rnd );

            if( targetPdf_curSurf > 0.0 )
            {
                nobiasM += reservoir_q.M;
            }
        }

        combined.M = nobiasM;
    }

    restirIndirect_StoreReservoir( pix, combined );

    // Doom64-RT: rt_debug_restir_m 2 -- the INDIRECT reservoir's M, the same
    // green ramp RtRaygenDirect.rgen paints for the direct one at 1. Exists so
    // a change to the GI path can PROVE it left the reuse contract alone: depth
    // touches radiance only, never the stored position/normal, so M must not
    // move with it. Read it on the unfiltered-indirect debug layer.
    if( globalUniform.debugRestirM == 2u )
    {
        const float mNorm = clamp( float( combined.M ) / 32.0, 0.0, 1.0 );
        imageStoreUnfilteredIndir( pix, vec3( mNorm * 0.15, mNorm, mNorm * 0.15 ) );
        return;
    }



    const vec3 indirRadiance =
        shade( surf, combined.selected, calcSelectedSampleWeightIndirect( combined ) );

    const vec3 surfToHitPoint = unpackSampleIndirectPosition( combined.selected ) - surf.position;

    {
        const vec3 direct = texelFetchUnfilteredSpecular( pix );
        // save indirect hit distance, if brighter than the direct light
        if( getLuminance( direct ) < getLuminance( indirRadiance ) )
        {
            imageStore(
                framebufViewDirection, pix, vec4( -surf.toViewerDir, length( surfToHitPoint ) ) );
        }


        // demodulate for denoising
        imageStoreUnfilteredSpecular( pix,
                                      direct + demodulateSpecular( indirRadiance, surf.specularColor ) );
    }

    {
        imageStoreUnfilteredIndir( pix, indirRadiance );
    }
}
#endif // RT_RAYGEN_INDIRECT_FINAL
