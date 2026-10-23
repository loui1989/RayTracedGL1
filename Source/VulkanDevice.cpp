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

#include "VulkanDevice.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "HaltonSequence.h"
#include "Matrix.h"
#include "RenderResolutionHelper.h"
#include "RgException.h"
#include "Utils.h"
#include "Generated/ShaderCommonC.h"

using namespace RTGL1;

// RTGL1_PROFILE=1: GPU / CPU time per stage of a frame
#define PROFILE_GPU( cmd, name )            \
    do                                       \
    {                                        \
        if( profiler )                       \
            profiler->Gpu( ( cmd ), name );  \
    } while( 0 )
#define PROFILE_CPU( name )       \
    do                             \
    {                              \
        if( profiler )             \
            profiler->Cpu( name ); \
    } while( 0 )

VkCommandBuffer VulkanDevice::BeginFrame(const RgStartFrameInfo &startInfo)
{
    PROFILE_CPU("game (outside the renderer)");

    uint32_t frameIndex = currentFrameState.IncrementFrameIndexAndGet();

    if (!waitForOutOfFrameFence)
    {
        // wait for previous cmd with the same frame index
        Utils::WaitAndResetFence(device, frameFences[frameIndex]);
    }
    else
    {
        Utils::WaitAndResetFences(device, frameFences[frameIndex], outOfFrameFences[frameIndex]);
    }

#ifdef __APPLE__
    // frame generation presents two images per frame and relies on vsync to space them out
    swapchain->RequestVsync(startInfo.requestVSync || frameGenMode != 0);
#else
    swapchain->RequestVsync(startInfo.requestVSync);
#endif
    PROFILE_CPU("wait for GPU (frame fence)");
    swapchain->AcquireImage(imageAvailableSemaphores[frameIndex]);
    frameWorkStart = std::chrono::steady_clock::now();
    PROFILE_CPU("acquire swapchain image");

    VkSemaphore semaphoreToWaitOnSubmit = imageAvailableSemaphores[frameIndex];


    // if out-of-frame cmd exist, submit it
    {
        VkCommandBuffer preFrameCmd = currentFrameState.GetPreFrameCmdAndRemove();
        if (preFrameCmd != VK_NULL_HANDLE)
        {
            // Signal inFrameSemaphore after completion.
            // Signal outOfFrameFences, but for the next frame
            // because we can't reset cmd pool with cmds (in this case 
            // it's preFrameCmd) that are in use.
            cmdManager->Submit(preFrameCmd,
                               semaphoreToWaitOnSubmit, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               inFrameSemaphores[frameIndex],
                               outOfFrameFences[(frameIndex + 1) % MAX_FRAMES_IN_FLIGHT]);

            // should wait other semaphore in this case
            semaphoreToWaitOnSubmit = inFrameSemaphores[frameIndex];

            waitForOutOfFrameFence = true;
        }
        else
        {
            waitForOutOfFrameFence = false;
        }
    }
    currentFrameState.SetSemaphore(semaphoreToWaitOnSubmit);


    if (startInfo.requestShaderReload)
    {
        shaderManager->ReloadShaders();
    }


    // reset cmds for current frame index
    cmdManager->PrepareForFrame(frameIndex);

    // clear the data that were created MAX_FRAMES_IN_FLIGHT ago
    worldSamplerManager->PrepareForFrame(frameIndex);
    genericSamplerManager->PrepareForFrame(frameIndex);
    textureManager->PrepareForFrame(frameIndex);
    cubemapManager->PrepareForFrame(frameIndex);
    rasterizer->PrepareForFrame(frameIndex);
    decalManager->PrepareForFrame(frameIndex);

    VkCommandBuffer cmd = cmdManager->StartGraphicsCmd();

    if (profiler)
    {
        profiler->BeginFrame(cmd, frameIndex);
#ifdef __APPLE__
        // GPU time of the passes of the native Metal sections of previous frames
        Metal::DrainTimings([this](const char* name, double ms) { profiler->Value(name, ms); });
#endif
    }

#ifdef __APPLE__
    // GPU time of this frame index's previous use (its fence was waited for above)
    if (frameGenTimestamps != VK_NULL_HANDLE)
    {
        if (frameGenTimestampsWritten[frameIndex])
        {
            uint64_t ts[2] = {};
            VkResult r = vkGetQueryPoolResults(device, frameGenTimestamps, frameIndex * 2, 2,
                                               sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (r == VK_SUCCESS && ts[1] > ts[0])
            {
                frameGenGpuMs = double(ts[1] - ts[0]) * double(frameGenTimestampPeriodNs) / 1e6;
            }
            frameGenTimestampsWritten[frameIndex] = false;
        }
        vkCmdResetQueryPool(cmd, frameGenTimestamps, frameIndex * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frameGenTimestamps, frameIndex * 2);
    }
#endif

    BeginCmdLabel(cmd, "Prepare for frame");

    // start dynamic geometry recording to current frame
    scene->PrepareForFrame(cmd, frameIndex);

    return cmd;
}

#define RG_SET_VEC3( dst, x, y, z ) \
    ( dst )[ 0 ] = ( x );           \
    ( dst )[ 1 ] = ( y );           \
    ( dst )[ 2 ] = ( z )

#define RG_SET_VEC3_A( dst, xyz )  \
    ( dst )[ 0 ] = ( xyz )[ 0 ]; \
    ( dst )[ 1 ] = ( xyz )[ 1 ]; \
    ( dst )[ 2 ] = ( xyz )[ 2 ]

#define RG_MAX_VEC3( dst, m ) \
    ( dst )[ 0 ] = std::max( ( dst )[ 0 ], ( m ) );  \
    ( dst )[ 1 ] = std::max( ( dst )[ 1 ], ( m ) );  \
    ( dst )[ 2 ] = std::max( ( dst )[ 2 ], ( m ) )

void VulkanDevice::FillUniform(ShGlobalUniform *gu, const RgDrawFrameInfo &drawInfo) const
{
    const float IdentityMat4x4[16] =
    {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    const float aspect = static_cast< float >( renderResolution.Width() ) / static_cast< float >( renderResolution.Height() );

    {
        memcpy( gu->viewPrev, gu->view, 16 * sizeof( float ) );
        memcpy( gu->projectionPrev, gu->projection, 16 * sizeof( float ) );

        memcpy( gu->view, drawInfo.view, 16 * sizeof( float ) );

        Matrix::MakeProjectionMatrix( gu->projection, 
                                      aspect, 
                                      drawInfo.fovYRadians, 
                                      drawInfo.cameraNear, 
                                      drawInfo.cameraFar );

        Matrix::Inverse( gu->invView, gu->view );
        Matrix::Inverse( gu->invProjection, gu->projection );

        memcpy( gu->cameraPositionPrev, gu->cameraPosition, 3 * sizeof( float ) );
        gu->cameraPosition[ 0 ] = gu->invView[ 12 ];
        gu->cameraPosition[ 1 ] = gu->invView[ 13 ];
        gu->cameraPosition[ 2 ] = gu->invView[ 14 ];
    }

    {
        static_assert( sizeof( gu->instanceGeomInfoOffset ) == sizeof( gu->instanceGeomInfoOffsetPrev ) );
        memcpy( gu->instanceGeomInfoOffsetPrev, gu->instanceGeomInfoOffset, sizeof( gu->instanceGeomInfoOffset ) );
    }

    {
        gu->frameId   = frameId;
        gu->timeDelta = static_cast< float >( std::max< double >( currentFrameTime - previousFrameTime, 0.001 ) );
        gu->time      = static_cast< float >( currentFrameTime );
    }

    {
        gu->renderWidth  = static_cast< float >( renderResolution.Width() );
        gu->renderHeight = static_cast< float >( renderResolution.Height() );
        // render width must be always even for checkerboarding!
        assert( ( int )gu->renderWidth % 2 == 0 );

        gu->upscaledRenderWidth  = static_cast< float >( renderResolution.UpscaledWidth() );
        gu->upscaledRenderHeight = static_cast< float >( renderResolution.UpscaledHeight() );

        RgFloat2D jitter = renderResolution.IsNvDlssEnabled() ? HaltonSequence::GetJitter_Halton23( frameId ) :
                           renderResolution.IsAmdFsr2Enabled() ? FSR2::GetJitter( renderResolution.GetResolutionState(), frameId ) :
                           RgFloat2D{ 0, 0 };

        gu->jitterX = jitter.data[ 0 ];
        gu->jitterY = jitter.data[ 1 ];
    }

    {
        gu->stopEyeAdaptation = drawInfo.disableEyeAdaptation;

        if( drawInfo.pTonemappingParams != nullptr )
        {
            gu->minLogLuminance     = drawInfo.pTonemappingParams->minLogLuminance;
            gu->maxLogLuminance     = drawInfo.pTonemappingParams->maxLogLuminance;
            gu->luminanceWhitePoint = drawInfo.pTonemappingParams->luminanceWhitePoint;
        }
        else
        {
            gu->minLogLuminance     = -3.9f;
            gu->maxLogLuminance     = -2.8f;
            gu->luminanceWhitePoint = 10.0f;
        }
    }

    {
        gu->lightCount     = scene->GetLightManager()->GetLightCount();
        gu->lightCountPrev = scene->GetLightManager()->GetLightCountPrev();

        gu->directionalLightExists = scene->GetLightManager()->DoesDirectionalLightExist();
    }

    {
        static_assert( sizeof( gu->skyCubemapRotationTransform ) == sizeof( IdentityMat4x4 ) && 
                       sizeof( IdentityMat4x4 ) == 16 * sizeof( float ), "Recheck skyCubemapRotationTransform sizes" );
        memcpy( gu->skyCubemapRotationTransform, IdentityMat4x4, 16 * sizeof( float ) );

        if( drawInfo.pSkyParams != nullptr )
        {
            const auto& sp = *drawInfo.pSkyParams;

            memcpy( gu->skyColorDefault, sp.skyColorDefault.data, sizeof( float ) * 3 );
            gu->skyColorMultiplier = sp.skyColorMultiplier;
            gu->skyColorSaturation = std::max( sp.skyColorSaturation, 0.0f );

            gu->skyType = sp.skyType == RG_SKY_TYPE_CUBEMAP ? SKY_TYPE_CUBEMAP :
                          sp.skyType == RG_SKY_TYPE_RASTERIZED_GEOMETRY ? SKY_TYPE_RASTERIZED_GEOMETRY :
                          SKY_TYPE_COLOR;

            gu->skyCubemapIndex = cubemapManager->IsCubemapValid( sp.skyCubemap ) ? sp.skyCubemap : RG_EMPTY_CUBEMAP;

            if( !Utils::IsAlmostZero( drawInfo.pSkyParams->skyCubemapRotationTransform ) )
            {
                Utils::SetMatrix3ToGLSLMat4( gu->skyCubemapRotationTransform, drawInfo.pSkyParams->skyCubemapRotationTransform );
            }
        }
        else
        {
            gu->skyColorDefault[ 0 ] = gu->skyColorDefault[ 1 ] = gu->skyColorDefault[ 2 ] = gu->skyColorDefault[ 3 ] = 1.0f;
            gu->skyColorMultiplier                                                                                    = 1.0f;
            gu->skyColorSaturation                                                                                    = 1.0f;
            gu->skyType                                                                                               = SKY_TYPE_COLOR;
            gu->skyCubemapIndex                                                                                       = RG_EMPTY_CUBEMAP;
        }

        RgFloat3D skyViewerPosition = drawInfo.pSkyParams ? drawInfo.pSkyParams->skyViewerPosition : RgFloat3D{ 0, 0, 0 };

        for( uint32_t i = 0; i < 6; i++ )
        {
            float* viewProjDst = &gu->viewProjCubemap[ 16 * i ];

            Matrix::GetCubemapViewProjMat( viewProjDst, i, skyViewerPosition.data, drawInfo.cameraNear, drawInfo.cameraFar );
        }
    }

    gu->debugShowFlags = 0;

    if( drawInfo.pDebugParams != nullptr )
    {
        RgDebugDrawFlags fs = drawInfo.pDebugParams->drawFlags;

        if( fs & RG_DEBUG_DRAW_ONLY_DIFFUSE_DIRECT_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_ONLY_DIRECT_DIFFUSE;
        }
        else if( fs & RG_DEBUG_DRAW_ONLY_DIFFUSE_INDIRECT_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_ONLY_INDIRECT_DIFFUSE;
        }
        else if( fs & RG_DEBUG_DRAW_ONLY_SPECULAR_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_ONLY_SPECULAR;
        }
        else if( fs & RG_DEBUG_DRAW_UNFILTERED_DIFFUSE_DIRECT_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE;
        }
        else if( fs & RG_DEBUG_DRAW_UNFILTERED_DIFFUSE_INDIRECT_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_UNFILTERED_INDIRECT;
        }
        else if( fs & RG_DEBUG_DRAW_UNFILTERED_SPECULAR_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_UNFILTERED_SPECULAR;
        }

        if( fs & RG_DEBUG_DRAW_ALBEDO_WHITE_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_ALBEDO_WHITE;
        }
        if( fs & RG_DEBUG_DRAW_MOTION_VECTORS_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_MOTION_VECTORS;
        }
        if( fs & RG_DEBUG_DRAW_GRADIENTS_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_GRADIENTS;
        }
        if( fs & RG_DEBUG_DRAW_LIGHT_GRID_BIT )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_LIGHT_GRID;
        }
    }

    if( drawInfo.pTexturesParams != nullptr )
    {
        gu->normalMapStrength      = drawInfo.pTexturesParams->normalMapStrength;
        gu->emissionMapBoost       = std::max( drawInfo.pTexturesParams->emissionMapBoost, 0.0f );
        gu->emissionMaxScreenColor = std::max( drawInfo.pTexturesParams->emissionMaxScreenColor, 0.0f );
        gu->squareInputRoughness   = !!drawInfo.pTexturesParams->squareInputRoughness;
        gu->minRoughness           = std::clamp( drawInfo.pTexturesParams->minRoughness, 0.0f, 1.0f );
    }
    else
    {
        gu->normalMapStrength      = 1.0f;
        gu->emissionMapBoost       = 100.0f;
        gu->emissionMaxScreenColor = 1.5f;
        gu->squareInputRoughness   = 1;
        gu->minRoughness           = 0.0f;
    }

    if( drawInfo.pIlluminationParams != nullptr )
    {
        gu->maxBounceShadowsLights     = drawInfo.pIlluminationParams->maxBounceShadows;
        gu->polyLightSpotlightFactor   = std::max( 0.0f, drawInfo.pIlluminationParams->polygonalLightSpotlightFactor );
        gu->indirSecondBounce          = !!drawInfo.pIlluminationParams->enableSecondBounceForIndirect;
        gu->lightIndexIgnoreFPVShadows = scene->GetLightManager()->GetLightIndexIgnoreFPVShadows( currentFrameState.GetFrameIndex(), drawInfo.pIlluminationParams->lightUniqueIdIgnoreFirstPersonViewerShadows );
        gu->cellWorldSize              = std::max( drawInfo.pIlluminationParams->cellWorldSize, 0.001f );
        gu->gradientMultDiffuse        = std::clamp( drawInfo.pIlluminationParams->directDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu->gradientMultIndirect       = std::clamp( drawInfo.pIlluminationParams->indirectDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu->gradientMultSpecular       = std::clamp( drawInfo.pIlluminationParams->specularSensitivityToChange, 0.0f, 1.0f );
    }
    else
    {
        gu->maxBounceShadowsLights     = 2;
        gu->polyLightSpotlightFactor   = 2.0f;
        gu->indirSecondBounce          = true;
        gu->lightIndexIgnoreFPVShadows = LIGHT_INDEX_NONE;
        gu->cellWorldSize              = 1.0f;
        gu->gradientMultDiffuse        = 0.5f;
        gu->gradientMultIndirect       = 0.2f;
        gu->gradientMultSpecular       = 0.5f;
    }

    if( drawInfo.pBloomParams != nullptr )
    {
        gu->bloomThreshold              = std::max( drawInfo.pBloomParams->inputThreshold, 0.0f );
        gu->bloomIntensity              = std::max( drawInfo.pBloomParams->bloomIntensity, 0.0f );
        gu->bloomEmissionMultiplier     = std::max( drawInfo.pBloomParams->bloomEmissionMultiplier, 0.0f );
    }
    else
    {
        gu->bloomThreshold              = 15.0f;
        gu->bloomIntensity              = 1.0f;
        gu->bloomEmissionMultiplier     = 64.0f;
    }

    static_assert(
        RG_MEDIA_TYPE_VACUUM == MEDIA_TYPE_VACUUM &&
        RG_MEDIA_TYPE_WATER == MEDIA_TYPE_WATER &&
        RG_MEDIA_TYPE_GLASS == MEDIA_TYPE_GLASS && 
        RG_MEDIA_TYPE_ACID == MEDIA_TYPE_ACID,
        "Interface and GLSL constants must be identical" );

    if( drawInfo.pReflectRefractParams != nullptr )
    {
        const auto& rr = *drawInfo.pReflectRefractParams;

        if( rr.typeOfMediaAroundCamera >= 0 && rr.typeOfMediaAroundCamera < MEDIA_TYPE_COUNT )
        {
            gu->cameraMediaType = rr.typeOfMediaAroundCamera;
        }
        else
        {
            gu->cameraMediaType = MEDIA_TYPE_VACUUM;
        }

        gu->reflectRefractMaxDepth     = std::min( 4u, rr.maxReflectRefractDepth );

        gu->indexOfRefractionGlass = std::max( 0.0f, rr.indexOfRefractionGlass );
        gu->indexOfRefractionWater = std::max( 0.0f, rr.indexOfRefractionWater );

        memcpy( gu->waterColorAndDensity, rr.waterColor.data, 3 * sizeof( float ) );
        gu->waterColorAndDensity[ 3 ] = 0.0f;

        memcpy( gu->acidColorAndDensity, rr.acidColor.data, 3 * sizeof( float ) );
        gu->acidColorAndDensity[ 3 ] = rr.acidDensity;

        gu->forceNoWaterRefraction            = !!rr.forceNoWaterRefraction;
        gu->waterWaveSpeed                    = rr.waterWaveSpeed;
        gu->waterWaveStrength                 = rr.waterWaveNormalStrength;
        gu->waterTextureDerivativesMultiplier = std::max( 0.0f, rr.waterWaveTextureDerivativesMultiplier );
        if( rr.waterTextureAreaScale < 0.0001f )
        {
            gu->waterTextureAreaScale = 1.0f;
        }
        else
        {
            gu->waterTextureAreaScale = rr.waterTextureAreaScale;
        }

        gu->noBackfaceReflForNoMediaChange = !!rr.disableBackfaceReflectionsForNoMediaChange;

        gu->twirlPortalNormal = !!rr.portalNormalTwirl;

        gu->rippleCount = 0;
        if( rr.pRipples )
        {
            for( uint32_t i = 0; i < std::min< uint32_t >( rr.rippleCount, MAX_WATER_RIPPLES ); i++ )
            {
                const RgWaterRipple& w = rr.pRipples[ i ];
                if( w.strength <= 0.0f || w.wavelength <= 0.0f || w.decayTime <= 0.0f || w.age < 0.0f )
                {
                    continue;
                }
                const uint32_t k              = gu->rippleCount++;
                gu->rippleSources[ k * 4 + 0 ] = w.position[ 0 ];
                gu->rippleSources[ k * 4 + 1 ] = w.position[ 1 ];
                gu->rippleSources[ k * 4 + 2 ] = w.position[ 2 ];
                gu->rippleSources[ k * 4 + 3 ] = w.age;
                gu->rippleParams[ k * 4 + 0 ]  = w.strength;
                gu->rippleParams[ k * 4 + 1 ]  = w.wavelength;
                gu->rippleParams[ k * 4 + 2 ]  = std::max( w.speed, 0.0f );
                gu->rippleParams[ k * 4 + 3 ]  = w.decayTime;
            }
        }
    }
    else
    {
        gu->cameraMediaType        = MEDIA_TYPE_VACUUM;
        gu->reflectRefractMaxDepth = 2;

        gu->indexOfRefractionGlass = 1.52f;
        gu->indexOfRefractionWater = 1.33f;

        RG_SET_VEC3( gu->waterColorAndDensity, 0.3f, 0.73f, 0.63f );
        gu->waterColorAndDensity[ 3 ] = 0.0f;

        RG_SET_VEC3( gu->acidColorAndDensity, 0.0f, 0.66f, 0.55f );
        gu->acidColorAndDensity[ 3 ] = 10.0f;

        gu->forceNoWaterRefraction            = false;
        gu->rippleCount                       = 0;
        gu->waterWaveSpeed                    = 1.0f;
        gu->waterWaveStrength                 = 1.0f;
        gu->waterTextureDerivativesMultiplier = 1.0f;
        gu->waterTextureAreaScale             = 1.0f;

        gu->noBackfaceReflForNoMediaChange = false;

        gu->twirlPortalNormal = false;
    }

    gu->rayCullBackFaces  = rayCullBackFacingTriangles ? 1 : 0;
    gu->rayLength         = clamp( drawInfo.rayLength, 0.1f, ( float )MAX_RAY_LENGTH );
    gu->primaryRayMinDist = clamp( drawInfo.cameraNear, 0.001f, gu->rayLength );

    {
        gu->rayCullMaskWorld = 0;

        if( drawInfo.rayCullMaskWorld & RG_DRAW_FRAME_RAY_CULL_WORLD_0_BIT )
        {
            gu->rayCullMaskWorld |= INSTANCE_MASK_WORLD_0;
        }

        if( drawInfo.rayCullMaskWorld & RG_DRAW_FRAME_RAY_CULL_WORLD_1_BIT )
        {
            gu->rayCullMaskWorld |= INSTANCE_MASK_WORLD_1;
        }

        if( drawInfo.rayCullMaskWorld & RG_DRAW_FRAME_RAY_CULL_WORLD_2_BIT )
        {
            if( allowGeometryWithSkyFlag )
            {
                throw RgException( RG_WRONG_ARGUMENT, "RG_DRAW_FRAME_RAY_CULL_WORLD_2_BIT cannot be used, as RgInstanceCreateInfo::allowGeometryWithSkyFlag was true" );
            }

            gu->rayCullMaskWorld |= INSTANCE_MASK_WORLD_2;
        }

#if RAYCULLMASK_SKY_IS_WORLD2
        if( drawInfo.rayCullMaskWorld & RG_DRAW_FRAME_RAY_CULL_SKY_BIT )
        {
            if( !allowGeometryWithSkyFlag )
            {
                throw RgException( RG_WRONG_ARGUMENT, "RG_DRAW_FRAME_RAY_CULL_SKY_BIT cannot be used, as RgInstanceCreateInfo::allowGeometryWithSkyFlag was false" );
            }

            gu->rayCullMaskWorld |= INSTANCE_MASK_WORLD_2;
        }
#else
    #error Handle RG_DRAW_FRAME_RAY_CULL_SKY_BIT, if there is no WORLD_2
#endif


        if( allowGeometryWithSkyFlag )
        {
            gu->rayCullMaskWorld_Shadow = gu->rayCullMaskWorld & ( ~INSTANCE_MASK_WORLD_2 );
        }
        else
        {
            gu->rayCullMaskWorld_Shadow = gu->rayCullMaskWorld;
        }
    }

    gu->waterNormalTextureIndex = textureManager->GetWaterNormalTextureIndex();

    gu->cameraRayConeSpreadAngle = atanf( ( 2.0f * tanf( drawInfo.fovYRadians * 0.5f ) ) / ( float )renderResolution.Height() );

    if( Utils::IsAlmostZero( drawInfo.worldUpVector ) )
    {
        gu->worldUpVector[ 0 ] = 0.0f;
        gu->worldUpVector[ 1 ] = 1.0f;
        gu->worldUpVector[ 2 ] = 0.0f;
    }
    else
    {
        gu->worldUpVector[ 0 ] = drawInfo.worldUpVector.data[ 0 ];
        gu->worldUpVector[ 1 ] = drawInfo.worldUpVector.data[ 1 ];
        gu->worldUpVector[ 2 ] = drawInfo.worldUpVector.data[ 2 ];
    }

    if( drawInfo.pLightmapParams != nullptr )
    {
        gu->lightmapEnable = !!drawInfo.pLightmapParams->enableLightmaps;

        if( drawInfo.pLightmapParams->lightmapLayerIndex == 1 || drawInfo.pLightmapParams->lightmapLayerIndex == 2 )
        {
            gu->lightmapLayer = drawInfo.pLightmapParams->lightmapLayerIndex;
        }
        else
        {
            assert( 0 && "pLightMapLayerIndex must point to a value of 1 or 2. Others are invalidated" );
        }
    }
    else
    {
        gu->lightmapEnable = false;
        gu->lightmapLayer  = UINT8_MAX;
    }

    gu->lensFlareCullingInputCount = rasterizer->GetLensFlareCullingInputCount();
    gu->applyViewProjToLensFlares  = !lensFlareVerticesInScreenSpace;

    {
        gu->volumeCameraNear = std::max( drawInfo.cameraNear, 0.001f );
        gu->volumeCameraFar  = std::min(
            drawInfo.cameraFar,
            drawInfo.pVolumetricParams ? drawInfo.pVolumetricParams->volumetricFar : 100.0f );

        if( drawInfo.pVolumetricParams )
        {
            if( drawInfo.pVolumetricParams->enable )
            {
                gu->volumeEnableType = drawInfo.pVolumetricParams->useSimpleDepthBased
                                           ? VOLUME_ENABLE_SIMPLE
                                           : VOLUME_ENABLE_VOLUMETRIC;
            }
            else
            {
                gu->volumeEnableType = VOLUME_ENABLE_NONE;
            }
            gu->volumeScattering = drawInfo.pVolumetricParams->scaterring;
            gu->volumeSourceAsymmetry = std::clamp( drawInfo.pVolumetricParams->sourceAssymetry, -1.0f, 1.0f );

            RG_SET_VEC3_A( gu->volumeAmbient, drawInfo.pVolumetricParams->ambientColor.data );
            RG_MAX_VEC3( gu->volumeAmbient, 0.0f );

            RG_SET_VEC3_A( gu->volumeSourceColor, drawInfo.pVolumetricParams->sourceColor.data );
            RG_MAX_VEC3( gu->volumeSourceColor, 0.0f );

            RG_SET_VEC3_A( gu->volumeDirToSource, drawInfo.pVolumetricParams->sourceDirection.data );
            Utils::Negate( gu->volumeDirToSource );
            Utils::Normalize( gu->volumeDirToSource );

            // smoke clouds
            const RgDrawFrameVolumetricParams& vp = *drawInfo.pVolumetricParams;
            gu->smokeCount  = 0;
            gu->smokeAlbedo = vp.smokeAlbedo > 0 ? vp.smokeAlbedo : 0.1f;
            if( vp.pSmokes )
            {
                for( uint32_t i = 0; i < std::min< uint32_t >( vp.smokeCount, MAX_SMOKE_VOLUMES ); i++ )
                {
                    const RgSmokeVolume& sm = vp.pSmokes[ i ];
                    if( sm.radius <= 0.0f || ( sm.density <= 0.0f && sm.glow <= 0.0f ) )
                    {
                        continue;
                    }
                    const uint32_t k             = gu->smokeCount++;
                    gu->smokeSpheres[ k * 4 + 0 ] = sm.position[ 0 ];
                    gu->smokeSpheres[ k * 4 + 1 ] = sm.position[ 1 ];
                    gu->smokeSpheres[ k * 4 + 2 ] = sm.position[ 2 ];
                    gu->smokeSpheres[ k * 4 + 3 ] = sm.radius;
                    gu->smokeParams[ k * 4 + 0 ]  = std::max( sm.density, 0.0f );
                    gu->smokeParams[ k * 4 + 1 ]  = sm.age;
                    gu->smokeParams[ k * 4 + 2 ]  = sm.seed;
                    gu->smokeParams[ k * 4 + 3 ]  = std::max( sm.glow, 0.0f );
                }
            }
        }
        else
        {
            gu->volumeEnableType      = VOLUME_ENABLE_VOLUMETRIC;
            gu->volumeScattering      = 0.2f;
            gu->volumeSourceAsymmetry = 0.4f;
            RG_SET_VEC3( gu->volumeAmbient, 0.8f, 0.85f, 1.0f );
            RG_SET_VEC3( gu->volumeSourceColor, 0, 0, 0 );
            RG_SET_VEC3( gu->volumeDirToSource, 0, 1, 0 );
            gu->smokeCount  = 0;
            gu->smokeAlbedo = 0.1f;
        }

        if( gu->volumeEnableType != VOLUME_ENABLE_NONE )
        {
            memcpy( gu->volumeViewProj_Prev, gu->volumeViewProj, 16 * sizeof( float ) );
            memcpy( gu->volumeViewProjInv_Prev, gu->volumeViewProjInv, 16 * sizeof( float ) );

            float volumeproj[ 16 ];
            Matrix::MakeProjectionMatrix( volumeproj,
                                          aspect,
                                          drawInfo.fovYRadians,
                                          gu->volumeCameraNear,
                                          gu->volumeCameraFar );

            Matrix::Multiply( gu->volumeViewProj, gu->view, volumeproj );
            Matrix::Inverse( gu->volumeViewProjInv, gu->volumeViewProj );
        }
    }

    gu->antiFireflyEnabled = !!drawInfo.forceAntiFirefly;

    // lighting stability: sample reuse and denoiser settings
    {
        RgDrawFrameDenoisingParams dn = {};
        if( drawInfo.pDenoisingParams != nullptr )
        {
            dn = *drawInfo.pDenoisingParams;
        }
        else
        {
            dn.indirectHistory       = 8;
            dn.indirectSpatialReuse  = 4;
            dn.indirectSpatialRadius = 8.0f;
            dn.indirectFireflyClamp  = 16.0f;
            dn.temporalReuseJitter   = 0.0f;
            dn.indirectDenseFilter   = true;
            dn.indirectDeflicker     = 2.0f;
            dn.blueNoise             = true;
            dn.antilagSquared        = true;
            dn.maxHistory            = 64.0f;
            dn.reflectionMaxHistory  = 64.0f;
            dn.fogHistory            = 0.85f;
        }

        // environment overrides, e.g. to compare settings without the game's menu
        auto envFloat = []( const char* name, float& v ) {
            if( const char* e = getenv( name ) )
            {
                char*       end = nullptr;
                const float f   = strtof( e, &end );
                if( end != e )
                {
                    v = f;
                }
            }
        };
        float history = float( dn.indirectHistory );
        float spatial = float( dn.indirectSpatialReuse );
        float dense   = dn.indirectDenseFilter ? 1.f : 0.f;
        float blue    = dn.blueNoise ? 1.f : 0.f;
        float sq      = dn.antilagSquared ? 1.f : 0.f;
        envFloat( "RTGL1_GI_HISTORY", history );
        envFloat( "RTGL1_GI_SPATIAL", spatial );
        envFloat( "RTGL1_GI_SPATIAL_RADIUS", dn.indirectSpatialRadius );
        envFloat( "RTGL1_GI_FIREFLY_CLAMP", dn.indirectFireflyClamp );
        envFloat( "RTGL1_REUSE_JITTER", dn.temporalReuseJitter );
        envFloat( "RTGL1_DENOISE_DENSE", dense );
        envFloat( "RTGL1_DENOISE_DEFLICKER", dn.indirectDeflicker );
        envFloat( "RTGL1_BLUE_NOISE", blue );
        envFloat( "RTGL1_ANTILAG_SQUARED", sq );
        envFloat( "RTGL1_DENOISE_HISTORY", dn.maxHistory );
        envFloat( "RTGL1_DENOISE_SPEC_HISTORY", dn.reflectionMaxHistory );
        envFloat( "RTGL1_FOG_HISTORY", dn.fogHistory );

        gu->restirGiMaxHistory        = uint32_t( std::clamp( history, 0.0f, 64.0f ) );
        gu->restirGiSpatialSamples    = uint32_t( std::clamp( spatial, 0.0f, 8.0f ) );
        gu->restirGiSpatialRadius     = std::clamp( dn.indirectSpatialRadius, 1.0f, 32.0f );
        gu->restirGiFireflyClamp      = std::clamp( dn.indirectFireflyClamp, 0.0f, 1000.0f );
        gu->restirGiTemporalRadius    = std::clamp( dn.temporalReuseJitter, 0.0f, 8.0f );
        gu->restirTemporalRadius      = gu->restirGiTemporalRadius;
        gu->denoiserIndirectDense     = dense != 0.f ? 1 : 0;
        gu->denoiserIndirectDeflicker = std::clamp( dn.indirectDeflicker, 0.0f, 100.0f );
        gu->blueNoiseSeeding          = blue != 0.f ? 1 : 0;
        gu->antilagSquared            = sq != 0.f ? 1 : 0;
        gu->denoiserMaxHistory        = std::clamp( dn.maxHistory, 4.0f, 256.0f );
        gu->denoiserSpecMaxHistory    = std::clamp( dn.reflectionMaxHistory, 4.0f, 256.0f );
        gu->volumeHistory             = std::clamp( dn.fogHistory, 0.0f, 0.98f );

        // log when the settings change
        static float lastLogged[ 12 ] = { -1 };
        const float  now[ 12 ]        = {
            float( gu->restirGiMaxHistory ),   float( gu->restirGiSpatialSamples ), gu->restirGiSpatialRadius,
            gu->restirGiFireflyClamp,          gu->restirGiTemporalRadius,          float( gu->denoiserIndirectDense ),
            gu->denoiserIndirectDeflicker,     float( gu->blueNoiseSeeding ),       float( gu->antilagSquared ),
            gu->denoiserMaxHistory,            gu->denoiserSpecMaxHistory,          gu->volumeHistory,
        };
        if( memcmp( lastLogged, now, sizeof( now ) ) != 0 )
        {
            memcpy( lastLogged, now, sizeof( now ) );
            fprintf( stderr,
                     "[RTGL1] lighting stability: GI history %u, spatial reuse %u (radius %.1f px), "
                     "firefly clamp %.1f, reuse jitter %.1f px, dense filter %u, deflicker %.1f, "
                     "blue noise %u, squared anti-lag %u, history %.0f / reflections %.0f frames, fog history %.2f\n",
                     gu->restirGiMaxHistory,
                     gu->restirGiSpatialSamples,
                     double( gu->restirGiSpatialRadius ),
                     double( gu->restirGiFireflyClamp ),
                     double( gu->restirGiTemporalRadius ),
                     gu->denoiserIndirectDense,
                     double( gu->denoiserIndirectDeflicker ),
                     gu->blueNoiseSeeding,
                     gu->antilagSquared,
                     double( gu->denoiserMaxHistory ),
                     double( gu->denoiserSpecMaxHistory ),
                     double( gu->volumeHistory ) );
        }
    }
}

void VulkanDevice::Render(VkCommandBuffer &cmd, const RgDrawFrameInfo &drawInfo)
{
    PROFILE_CPU("game uploads geometry");
    PROFILE_GPU(cmd, "geometry uploads");
#ifdef __APPLE__
    {
        uint32_t mode = drawInfo.pRenderResolutionParams ? drawInfo.pRenderResolutionParams->frameGeneration : 0;
        const char* e = getenv("RTGL1_FRAMEGEN");
        if (e && e[0] == '1')
        {
            mode = 1;
        }
        if (mode > 2 || !renderResolution.IsAmdFsr2Enabled() || !FSR2::IsFrameGenSupported())
        {
            mode = 0;
        }
        if (mode != frameGenMode)
        {
            frameGenAutoEngaged = false;
            frameGenWorkMs      = 0;
            fprintf(stderr, "[FrameGen] mode: %s\n", mode == 0 ? "off" : mode == 1 ? "on" : "auto");
        }
        frameGenMode = mode;
        amdFsr2->SetFrameGenWanted(mode != 0);
    }
#endif
    // end of "Prepare for frame" label
    EndCmdLabel(cmd);


    const uint32_t frameIndex = currentFrameState.GetFrameIndex();

    
    bool mipLodBiasUpdated = worldSamplerManager->TryChangeMipLodBias(frameIndex, renderResolution.GetMipLodBias());
    const RgFloat2D jitter = { uniform->GetData()->jitterX, uniform->GetData()->jitterY };

    textureManager->SubmitDescriptors(frameIndex, drawInfo.pTexturesParams, mipLodBiasUpdated);
    cubemapManager->SubmitDescriptors(frameIndex);


    // submit geometry and upload uniform after getting data from a scene
    scene->SubmitForFrame(cmd, frameIndex, uniform, 
                          uniform->GetData()->rayCullMaskWorld, 
                          allowGeometryWithSkyFlag,
                          drawInfo.disableRayTracedGeometry,
                          profiler ? std::function< void( VkCommandBuffer, const char* ) >(
                                         [ this ]( VkCommandBuffer c, const char* n ) { profiler->Gpu( c, n ); } )
                                   : nullptr);
    if (profiler)
    {
        const auto st = scene->GetASManager()->GetLastDynamicBuildStats();
        profiler->Value("dynamic BLAS count", st.blasCount);
        profiler->Value("dynamic BLAS geometries", st.geometryCount);
        profiler->Value("dynamic BLAS triangles", st.primitiveCount);
    }


    framebuffers->PrepareForSize(renderResolution.GetResolutionState());

#ifdef __APPLE__
    // macOS, RTGL1_METAL_RT=1: the path tracing passes run natively in Metal, in a section
    // between Vulkan submissions (split around the decals, which are drawn with Vulkan).
    // The section continues with the denoiser, volumetrics, exposure and checkerboard
    // passes, if they could be prepared for Metal (metalCompute). With metalRaster, the
    // section also covers the rasterization and copies around them: from here to the
    // upscaling, the frame is recorded straight into Metal.
    // (with RTGL1_METAL_ONLY=1 every command buffer is Metal, no sections are needed)
    const bool metalPT      = Metal::IsEnabled() && !Metal::IsNativeMode() && rtPipeline->IsMetalReady();
    const bool metalCompute = metalPT && Metal::IsComputeReady();
    const bool metalRaster  = metalCompute && Metal::IsRasterReady();
    if (metalRaster)
    {
        Metal::BeginSection(cmd);
    }
#else
    const bool metalPT      = false;
    const bool metalCompute = false;
    const bool metalRaster  = false;
#endif
    

    if (!drawInfo.disableRasterization)
    {
        rasterizer->SubmitForFrame(cmd, frameIndex);

        // draw rasterized sky to albedo before tracing primary rays
        if (uniform->GetData()->skyType == RG_SKY_TYPE_RASTERIZED_GEOMETRY)
        {
            RgFloat3D skyViewerPosition = drawInfo.pSkyParams ? drawInfo.pSkyParams->skyViewerPosition : RgFloat3D{ 0,0,0 };

            rasterizer->DrawSkyToCubemap(cmd, frameIndex, textureManager, uniform);
            rasterizer->DrawSkyToAlbedo(cmd, frameIndex, textureManager, uniform->GetData()->view, skyViewerPosition.data, uniform->GetData()->projection, jitter, renderResolution);
        }
    }


    {
        PROFILE_GPU(cmd, "sky raster");
        lightGrid->Build(cmd, frameIndex, uniform, blueNoise, scene->GetLightManager());

        decalManager->SubmitForFrame(cmd, frameIndex);
        portalList->SubmitForFrame(cmd, frameIndex);
        PROFILE_GPU(cmd, "light grid");

        if (profiler)
        {
            pathTracer->SetPassCallback([this](VkCommandBuffer c, const char* name) { profiler->Gpu(c, name); });
        }

#ifdef __APPLE__
        // macOS (MoltenVK): after a vkCmdTraceRaysKHR dispatch, later commands in the same
        // command buffer could read wrong push constants, so each pass was continued in a
        // fresh command buffer. The path tracer now uses ray queries in ordinary compute
        // shaders, so this is off by default; RTGL1_SPLIT_CMD=1 turns it back on.
        static const bool splitCmd = [] {
            const char* e = getenv( "RTGL1_SPLIT_CMD" );
            return e && e[ 0 ] == '1';
        }();
        if( splitCmd && !metalPT )
        {
            pathTracer->SetCommandBufferSplitter( [ this, &cmd ]( VkCommandBuffer old ) {
                cmdManager->Submit( old );
                cmd = cmdManager->StartGraphicsCmd();
                return cmd;
            } );
        }
#endif
        const auto params = pathTracer->Bind( cmd,
                                              frameIndex,
                                              renderResolution.Width(),
                                              renderResolution.Height(),
                                              scene.get(),
                                              uniform.get(),
                                              textureManager.get(),
                                              framebuffers,
                                              restirBuffers,
                                              blueNoise.get(),
                                              cubemapManager.get(),
                                              rasterizer->GetRenderCubemap().get(),
                                              portalList.get(),
                                              volumetric.get() );

#ifdef __APPLE__
        if (metalPT)
        {
            pathTracer->SetMetal(true);
            if (!metalRaster)
            {
                Metal::BeginSection(cmd);
            }
            PathTracer::SetCmd(params, cmd);
        }
#endif

        pathTracer->TracePrimaryRays(params);

        // draw decals on top of primary surface
#ifdef __APPLE__
        if (metalPT && !metalRaster && decalManager->HasDecals())
        {
            Metal::EndSection(cmd);
            decalManager->Draw(cmd, frameIndex, uniform, framebuffers, textureManager);
            Metal::BeginSection(cmd);
            PathTracer::SetCmd(params, cmd);
        }
        else if (!metalPT || metalRaster)
#endif
        {
            decalManager->Draw(cmd, frameIndex, uniform, framebuffers, textureManager);
            PROFILE_GPU(cmd, "decals");
        }

        if (uniform->GetData()->reflectRefractMaxDepth > 0)
        {
            pathTracer->TraceReflectionRefractionRays(params);
        }

        if (!metalPT)
        {
            scene->GetLightManager()->BarrierLightGrid(cmd, frameIndex);
        }
        pathTracer->CalculateInitialReservoirs(params);
        pathTracer->TraceDirectllumination(params);
        pathTracer->TraceIndirectllumination(params);
        pathTracer->TraceVolumetric(params);

        pathTracer->CalculateGradientsSamples(params);

#ifdef __APPLE__
        if (metalPT)
        {
            pathTracer->SetMetal(false);
            if (!metalCompute)
            {
                Metal::EndSection(cmd);
                PROFILE_GPU(cmd, "path tracing (native Metal)");
            }
        }
#endif
        pathTracer->SetCommandBufferSplitter( nullptr );
        pathTracer->SetPassCallback( nullptr );
        denoiser->Denoise(cmd, frameIndex, uniform);
        PROFILE_GPU(cmd, "denoiser");
        volumetric->ProcessScattering( cmd, frameIndex, uniform.get(), blueNoise.get() );
        PROFILE_GPU(cmd, "volumetric process");
        tonemapping->CalculateExposure(cmd, frameIndex, uniform);
        PROFILE_GPU(cmd, "exposure");
    }

    imageComposition->PrepareForRaster( cmd, frameIndex, uniform.get() );
    volumetric->BarrierToReadScattering( cmd, frameIndex );
    volumetric->BarrierToReadIllumination( cmd );

#ifdef __APPLE__
    if (metalCompute && !metalRaster)
    {
        PROFILE_GPU(cmd, "checkerboard");
        Metal::EndSection(cmd);
        PROFILE_GPU(cmd, "native Metal section");
    }
#endif

    if (!drawInfo.disableRasterization)
    {
        // draw rasterized geometry into the final image
        rasterizer->DrawToFinalImage(
            cmd,
            frameIndex,
            textureManager,
            uniform,
            tonemapping,
            volumetric,
            uniform->GetData()->view,
            uniform->GetData()->projection,
            jitter,
            renderResolution,
            drawInfo.pLensFlareParams );
    }

    PROFILE_GPU(cmd, "raster (sprites, particles)");
    imageComposition->Finalize(
        cmd, frameIndex, uniform.get(), tonemapping.get(), volumetric.get() );
    PROFILE_GPU(cmd, "composition");


    bool enableBloom = drawInfo.pBloomParams == nullptr || (drawInfo.pBloomParams != nullptr && drawInfo.pBloomParams->bloomIntensity > 0.0f);

    if (enableBloom)
    {
        bloom->Prepare(cmd, frameIndex, uniform, tonemapping);
    }

#ifdef __APPLE__
    // MetalFX is encoded into the section too, which continues with the post effects
    // (Metal only: recorded into the frame's command buffer like everything else)
    const bool metalFxInSection = (metalRaster || Metal::IsNativeMode()) && renderResolution.IsAmdFsr2Enabled();
#endif


    FramebufferImageIndex accum = FramebufferImageIndex::FB_IMAGE_INDEX_FINAL;
    {
        // upscale finalized image
        if (renderResolution.IsNvDlssEnabled())
        {
            accum = nvDlss->Apply(cmd, frameIndex, 
                                               framebuffers, 
                                               renderResolution, 
                                               jitter);
        }
#ifdef __APPLE__
        else if (metalFxInSection)
        {
            PROFILE_GPU(cmd, "bloom prepare");
            PROFILE_CPU("record path tracing commands");
            accum = amdFsr2->Apply(cmd,
                                   frameIndex,
                                   framebuffers,
                                   renderResolution,
                                   jitter,
                                   uniform->GetData()->timeDelta,
                                   drawInfo.cameraNear,
                                   drawInfo.cameraFar,
                                   drawInfo.fovYRadians);
            PROFILE_GPU(cmd, "MetalFX");
            PROFILE_CPU("MetalFX: wait for GPU + upscale");
        }
#endif
        else if (renderResolution.IsAmdFsr2Enabled())
        {
#ifdef __APPLE__
            // MetalFX encodes its own Metal command buffer; by default ordered with
            // the Vulkan work by a timeline semaphore, without CPU waits
            PROFILE_GPU(cmd, "bloom prepare");
            PROFILE_CPU("record path tracing commands");
            VkSemaphore mfxSem = VK_NULL_HANDLE;
            uint64_t    vulkanDoneValue = 0, metalFxDoneValue = 0;
            if (amdFsr2->PrepareAsync(mfxSem, vulkanDoneValue, metalFxDoneValue))
            {
                cmdManager->SubmitTimeline(cmd, mfxSem, 0, vulkanDoneValue);
                metalFxWaitSemaphore = mfxSem;
                metalFxWaitValue     = metalFxDoneValue;
            }
            else
            {
                cmdManager->Submit(cmd);
            }
            cmd = VK_NULL_HANDLE;
            PROFILE_CPU("submit (MoltenVK)");
#endif
            accum = amdFsr2->Apply(cmd, frameIndex, 
                                                framebuffers, 
                                                renderResolution, 
                                                jitter, 
                                                uniform->GetData()->timeDelta, 
                                                drawInfo.cameraNear, 
                                                drawInfo.cameraFar, 
                                                drawInfo.fovYRadians );
#ifdef __APPLE__
            PROFILE_CPU("MetalFX: wait for GPU + upscale");
            if (profiler)
            {
                double waitMs, mfxGpuMs;
                FSR2::GetLastTimings(waitMs, mfxGpuMs);
                profiler->Value("CPU waits for path tracing GPU work", waitMs);
                profiler->Value("MetalFX GPU", mfxGpuMs);
            }
            cmd = cmdManager->StartGraphicsCmd();
            PROFILE_GPU(cmd, "MetalFX + CPU sync gap");
#endif
        }

        const RgExtent2D* pixelized = drawInfo.pRenderResolutionParams
                                          ? drawInfo.pRenderResolutionParams->pPixelizedRenderSize
                                          : nullptr;

        accum = framebuffers->BlitForEffects(
            cmd, frameIndex, accum, renderResolution.GetBlitFilter(), pixelized );
    }


    const CommonnlyUsedEffectArguments args = { cmd, frameIndex, framebuffers, uniform, renderResolution.UpscaledWidth(), renderResolution.UpscaledHeight(), (float)currentFrameTime };
    {
        if (renderResolution.IsDedicatedSharpeningEnabled())
        {
            accum = sharpening->Apply(
                cmd, frameIndex, framebuffers, renderResolution.UpscaledWidth(), renderResolution.UpscaledHeight(), accum,
                renderResolution.GetSharpeningTechnique(), renderResolution.GetSharpeningIntensity());
        }
        if (enableBloom)
        {
            accum = bloom->Apply(cmd, frameIndex, uniform, renderResolution.UpscaledWidth(), renderResolution.UpscaledHeight(), accum);
        }
        if (effectColorTint->Setup(args, drawInfo.postEffectParams.pColorTint))
        {
            accum = effectColorTint->Apply(args, accum);
        }
        if (effectInverseBW->Setup(args, drawInfo.postEffectParams.pInverseBlackAndWhite))
        {
            accum = effectInverseBW->Apply(args, accum);
        }
        if (effectHueShift->Setup(args, drawInfo.postEffectParams.pHueShift))
        {
            accum = effectHueShift->Apply(args, accum);
        }
        if (effectChromaticAberration->Setup(args, drawInfo.postEffectParams.pChromaticAberration))
        {
            accum = effectChromaticAberration->Apply(args, accum);
        }
        if (effectDistortedSides->Setup(args, drawInfo.postEffectParams.pDistortedSides))
        {
            accum = effectDistortedSides->Apply(args, accum);
        }
        if (effectWaves->Setup(args, drawInfo.postEffectParams.pWaves))
        {
            accum = effectWaves->Apply(args, accum);
        }
        if (effectRadialBlur->Setup(args, drawInfo.postEffectParams.pRadialBlur))
        {
            accum = effectRadialBlur->Apply(args, accum);
        }
    }

#ifdef __APPLE__
    if (metalRaster)
    {
        PROFILE_GPU(cmd, "upscaled post effects");
        Metal::EndSection(cmd);
        PROFILE_GPU(cmd, "native Metal section");
    }
    PROFILE_GPU(cmd, "upscaled post effects");
    // macOS (MoltenVK): draw the HUD and present from a fresh command buffer,
    // for the same reason as the path tracer's command buffer splitting
    if (frameGenTimestamps != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frameGenTimestamps, frameIndex * 2 + 1);
        frameGenTimestampsWritten[frameIndex] = true;
    }
    if (metalFxWaitSemaphore != VK_NULL_HANDLE)
    {
        // the post effects read MetalFX's output
        cmdManager->AddTimelineWaitForNextSubmit(metalFxWaitSemaphore, metalFxWaitValue);
        metalFxWaitSemaphore = VK_NULL_HANDLE;
    }
    cmdManager->Submit(cmd);

    // MetalFX frame generation (experimental): show a frame generated between the
    // previous and this one first, with the HUD drawn on top, then this frame
    bool frameGenEngage = false;
    if (frameGenMode != 0)
    {
        // GPU time of the frame's rendering if available (it leaves out waiting for
        // the display, which with vsync and frame generation is most of the frame);
        // otherwise CPU time since the swapchain image was acquired
        const double workMs =
            frameGenGpuMs > 0
                ? frameGenGpuMs
                : std::chrono::duration< double, std::milli >(std::chrono::steady_clock::now() - frameWorkStart).count();
        frameGenWorkMs = frameGenWorkMs <= 0 ? workMs : frameGenWorkMs * 0.9 + workMs * 0.1;
        if (frameGenMode == 2 && ++frameGenLogCounter % 300 == 0)
        {
            fprintf(stderr, "[FrameGen] auto: frame takes %.1f ms (%s), generating: %s\n", frameGenWorkMs,
                    frameGenGpuMs > 0 ? "GPU" : "CPU", frameGenAutoEngaged ? "yes" : "no");
        }

        if (frameGenMode == 2)
        {
            // auto: generate frames only while the game can't render at ~60 fps itself
            const bool wasEngaged = frameGenAutoEngaged;
            if (!frameGenAutoEngaged && frameGenWorkMs > 17.5)
            {
                frameGenAutoEngaged = true;
            }
            else if (frameGenAutoEngaged && frameGenWorkMs < 15.0)
            {
                frameGenAutoEngaged = false;
            }
            if (wasEngaged != frameGenAutoEngaged)
            {
                fprintf(stderr, "[FrameGen] auto: %s (frame takes %.1f ms)\n",
                        frameGenAutoEngaged ? "on" : "off", frameGenWorkMs);
            }
        }
        frameGenEngage = frameGenMode == 1 || frameGenAutoEngaged;
    }
    frameGenActive = false;

    if (renderResolution.IsAmdFsr2Enabled() &&
        amdFsr2->FrameGenInterpolate(frameIndex, framebuffers, accum, frameGenEngage))
    {
        frameGenActive = true;
        cmd = cmdManager->StartGraphicsCmd();
        if (!drawInfo.disableRasterization)
        {
            rasterizer->DrawToSwapchain( cmd,
                                         frameIndex,
                                         accum,
                                         textureManager,
                                         uniform->GetData()->view,
                                         uniform->GetData()->projection,
                                         renderResolution.UpscaledWidth(),
                                         renderResolution.UpscaledHeight() );
        }
        framebuffers->PresentToSwapchain(cmd, frameIndex, swapchain, accum, VK_FILTER_NEAREST);
        cmdManager->Submit(cmd,
                           currentFrameState.GetSemaphoreForWaitAndRemove(),
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           frameGenRenderFinishedSemaphores[frameIndex],
                           VK_NULL_HANDLE);
        swapchain->Present(queues, frameGenRenderFinishedSemaphores[frameIndex]);

        // the real frame goes to the next swapchain image
        swapchain->AcquireImage(frameGenImageAvailableSemaphores[frameIndex]);
        currentFrameState.SetSemaphore(frameGenImageAvailableSemaphores[frameIndex]);
    }
    amdFsr2->FrameGenFinish(frameIndex, framebuffers, accum);
    PROFILE_CPU("post effects + frame generation");

    cmd = cmdManager->StartGraphicsCmd();
    PROFILE_GPU(cmd, "frame generation + gap");
#endif

    // draw geometry such as HUD into an upscaled framebuf
    if (!drawInfo.disableRasterization)
    {
        rasterizer->DrawToSwapchain( cmd,
                                     frameIndex,
                                     accum,
                                     textureManager,
                                     uniform->GetData()->view,
                                     uniform->GetData()->projection,
                                     renderResolution.UpscaledWidth(),
                                     renderResolution.UpscaledHeight() );
    }

    // post-effect that work on swapchain geometry too (cmd may have been replaced above)
    {
        const CommonnlyUsedEffectArguments args = { cmd, frameIndex, framebuffers, uniform, renderResolution.UpscaledWidth(), renderResolution.UpscaledHeight(), (float)currentFrameTime };
        if (effectWipe->Setup(args, drawInfo.postEffectParams.pWipe, swapchain, frameId))
        {
            accum = effectWipe->Apply(args, blueNoise, accum);
        }
        if (drawInfo.postEffectParams.pCRT != nullptr && drawInfo.postEffectParams.pCRT->isActive)
        {
            effectCrtDemodulateEncode->Setup(args);
            accum = effectCrtDemodulateEncode->Apply(args, accum);

            effectCrtDecode->Setup(args);
            accum = effectCrtDecode->Apply(args, accum);
        }
    }

    // blit result image to present on a surface
    framebuffers->PresentToSwapchain( cmd, frameIndex, swapchain, accum, VK_FILTER_NEAREST );
}

void VulkanDevice::EndFrame(VkCommandBuffer cmd)
{
    uint32_t frameIndex = currentFrameState.GetFrameIndex();
    VkSemaphore semaphoreToWait = currentFrameState.GetSemaphoreForWaitAndRemove();

    PROFILE_GPU(cmd, "HUD + copy to screen");
    PROFILE_CPU("record HUD");

    // submit command buffer, but wait until presentation engine has completed using image
    cmdManager->Submit(
        cmd, 
        semaphoreToWait,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
        renderFinishedSemaphores[frameIndex],
        frameFences[frameIndex]);

    PROFILE_CPU("submit");

    // present on a surface when rendering will be finished
    swapchain->Present(queues, renderFinishedSemaphores[frameIndex]);
    PROFILE_CPU("present");

    frameId++;
}



#pragma region RTGL1 interface implementation

void VulkanDevice::StartFrame(const RgStartFrameInfo *startInfo)
{
    if (currentFrameState.WasFrameStarted())
    {
        throw RgException(RG_FRAME_WASNT_ENDED);
    }

    if (startInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    VkCommandBuffer newFrameCmd = BeginFrame(*startInfo);
    currentFrameState.OnBeginFrame(newFrameCmd);
}

void VulkanDevice::DrawFrame(const RgDrawFrameInfo *drawInfo)
{
    if (!currentFrameState.WasFrameStarted())
    {
        throw RgException(RG_FRAME_WASNT_STARTED);
    }

    if (drawInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    VkCommandBuffer cmd = currentFrameState.GetCmdBuffer();

    previousFrameTime = currentFrameTime;
    currentFrameTime = drawInfo->currentTime;

    renderResolution.Setup(drawInfo->pRenderResolutionParams,
                           swapchain->GetWidth(), swapchain->GetHeight(), nvDlss);

    textureManager->CheckForHotReload(cmd);

    if (renderResolution.Width() > 0 && renderResolution.Height() > 0)
    {
        FillUniform(uniform->GetData(), *drawInfo);
        Render(cmd, *drawInfo);
    }

    EndFrame(cmd);
    currentFrameState.OnEndFrame();
}

bool VulkanDevice::IsSuspended() const
{
    if (!swapchain)
    {
        return false;
    }

    return !swapchain->IsExtentOptimal();
}

bool RTGL1::VulkanDevice::IsRenderUpscaleTechniqueAvailable(RgRenderUpscaleTechnique technique) const
{
    switch (technique)
    {
        case RG_RENDER_UPSCALE_TECHNIQUE_NEAREST:
        case RG_RENDER_UPSCALE_TECHNIQUE_LINEAR:
            return true;
        case RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2:
            return FSR2::IsFsr2Available();
        case RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS:
            return nvDlss->IsDlssAvailable();
        default:
            throw RgException(RG_WRONG_ARGUMENT, "Incorrect technique was passed to rgIsRenderUpscaleTechniqueAvailable");
    }
}

void VulkanDevice::Print(const char *pMessage) const
{
    userPrint->Print(pMessage);
}


void VulkanDevice::UploadGeometry(const RgGeometryUploadInfo *uploadInfo)
{
    using namespace std::string_literals;

    if (uploadInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    if (uploadInfo->pVertices == nullptr || uploadInfo->vertexCount == 0)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Incorrect vertex data");
    }

    if ((uploadInfo->pIndices == nullptr && uploadInfo->indexCount != 0) ||
        (uploadInfo->pIndices != nullptr && uploadInfo->indexCount == 0))
    {
        throw RgException(RG_WRONG_ARGUMENT, "Incorrect index data");
    }

    if (uploadInfo->geomType != RG_GEOMETRY_TYPE_STATIC &&
        uploadInfo->geomType != RG_GEOMETRY_TYPE_STATIC_MOVABLE &&
        uploadInfo->geomType != RG_GEOMETRY_TYPE_DYNAMIC &&

        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_OPAQUE &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_ALPHA_TESTED &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_MIRROR &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_PORTAL &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_WATER_ONLY_REFLECT &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_WATER_REFLECT_REFRACT &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_GLASS_REFLECT_REFRACT &&
        uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_ACID_REFLECT_REFRACT &&

        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_WORLD_0 &&
        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_WORLD_1 &&
        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_WORLD_2 &&
        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_FIRST_PERSON &&
        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_FIRST_PERSON_VIEWER && 
        uploadInfo->visibilityType != RG_GEOMETRY_VISIBILITY_TYPE_SKY)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Incorrect type of ray traced geometry");
    }

    if (allowGeometryWithSkyFlag)
    {
        if (uploadInfo->visibilityType == RG_GEOMETRY_VISIBILITY_TYPE_WORLD_2)
        {
            throw RgException(RG_WRONG_ARGUMENT, "Geometry with RG_GEOMETRY_VISIBILITY_TYPE_WORLD_2 cannot be used, as RgInstanceCreateInfo::allowGeometryWithSkyFlag was true");
        }
    }
    else
    {
        if (uploadInfo->visibilityType == RG_GEOMETRY_VISIBILITY_TYPE_SKY)
        {
            throw RgException(RG_WRONG_ARGUMENT, "Geometry with RG_GEOMETRY_VISIBILITY_TYPE_SKY cannot be used, as RgInstanceCreateInfo::allowGeometryWithSkyFlag was false");
        }
    }

    if ((uploadInfo->flags & RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_MULTIPLY_BIT) != 0 &&
        (uploadInfo->flags & RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_ADD_BIT) != 0)
    {
        throw RgException(RG_WRONG_ARGUMENT, "RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_MULTIPLY_BIT and RG_GEOMETRY_UPLOAD_REFL_REFR_ALBEDO_ADD_BIT must be set separately");
    }

    if (scene->DoesUniqueIDExist(uploadInfo->uniqueID))
    {
        throw RgException(RG_WRONG_ARGUMENT, "Geometry with ID="s + std::to_string(uploadInfo->uniqueID) + " already exists");
    }

    if (uploadInfo->pPortalIndex != nullptr && uploadInfo->passThroughType != RG_GEOMETRY_PASS_THROUGH_TYPE_PORTAL)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Geometry's pPortalIndex is non-null, but geometry is not marked as portal");
    }

    if (uploadInfo->pPortalIndex == nullptr && uploadInfo->passThroughType == RG_GEOMETRY_PASS_THROUGH_TYPE_PORTAL)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Geometry is marked as portal, but pPortalIndex is null");
    }

    if (uploadInfo->pPortalIndex && *(uploadInfo->pPortalIndex) >= PORTAL_MAX_COUNT)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Geometry's portal index must be in [0, 62]");
    }

    scene->Upload(currentFrameState.GetFrameIndex(), *uploadInfo);
}

void VulkanDevice::UpdateGeometryTransform(const RgUpdateTransformInfo *updateInfo)
{
    if (updateInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UpdateTransform(*updateInfo);
}

void RTGL1::VulkanDevice::UpdateGeometryTexCoords(const RgUpdateTexCoordsInfo *updateInfo)
{
    if (updateInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UpdateTexCoords(*updateInfo);
}

void VulkanDevice::UploadRasterizedGeometry(const RgRasterizedGeometryUploadInfo *pUploadInfo,
                                                const float *pViewProjection, const RgViewport *pViewport)
{
    if (pUploadInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    if (pUploadInfo->renderType != RG_RASTERIZED_GEOMETRY_RENDER_TYPE_DEFAULT &&
        pUploadInfo->renderType != RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SWAPCHAIN &&
        pUploadInfo->renderType != RG_RASTERIZED_GEOMETRY_RENDER_TYPE_SKY)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Incorrect render type of rasterized geometry");
    }

    // nothing to draw (e.g. a string of spaces, an empty polygon from QC): not an error
    if (pUploadInfo->vertexCount == 0)
    {
        return;
    }

    if (pUploadInfo->pVertices == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Vertex data / count is null");
    }

    if ((pUploadInfo->pIndices == nullptr && pUploadInfo->indexCount != 0) ||
        (pUploadInfo->pIndices != nullptr && pUploadInfo->indexCount == 0))
    {
        throw RgException(RG_WRONG_ARGUMENT, "Index data / count must be both not null or null");
    }

    rasterizer->Upload(currentFrameState.GetFrameIndex(), *pUploadInfo, pViewProjection, pViewport);
}

void RTGL1::VulkanDevice::UploadLensFlare(const RgLensFlareUploadInfo *pUploadInfo)
{
    if (pUploadInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    if (pUploadInfo->indexCount == 0 || pUploadInfo->pIndices == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Lens flare index data and count must not be null");
    }

    if (pUploadInfo->vertexCount == 0 || pUploadInfo->pVertices == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Lens flare vertex data and count must not be null");
    }

    rasterizer->UploadLensFlare(currentFrameState.GetFrameIndex(), *pUploadInfo);
}

void RTGL1::VulkanDevice::UploadDecal(const RgDecalUploadInfo *pUploadInfo)
{
    if (pUploadInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    decalManager->Upload(currentFrameState.GetFrameIndex(), *pUploadInfo, textureManager);
}

void RTGL1::VulkanDevice::UploadPortal(const RgPortalUploadInfo *pUploadInfo)
{
    if (pUploadInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    portalList->Upload(currentFrameState.GetFrameIndex(), *pUploadInfo);
}

void VulkanDevice::SubmitStaticGeometries()
{
    scene->SubmitStatic();
}

void VulkanDevice::StartNewStaticScene()
{
    scene->StartNewStatic();
}

void VulkanDevice::UploadDirectionalLight(const RgDirectionalLightUploadInfo *pLightInfo)
{
    if (pLightInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UploadLight(currentFrameState.GetFrameIndex(), *pLightInfo);
}

void VulkanDevice::UploadSphericalLight(const RgSphericalLightUploadInfo *pLightInfo)
{
    if (pLightInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UploadLight(currentFrameState.GetFrameIndex(), *pLightInfo);
}

void VulkanDevice::UploadSpotlight(const RgSpotLightUploadInfo *pLightInfo)
{
    if (pLightInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UploadLight(currentFrameState.GetFrameIndex(), *pLightInfo);
}

void RTGL1::VulkanDevice::UploadPolygonalLight(const RgPolygonalLightUploadInfo *pLightInfo)
{
    if (pLightInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    scene->UploadLight(currentFrameState.GetFrameIndex(), *pLightInfo);
}

void VulkanDevice::CreateMaterial(const RgMaterialCreateInfo *createInfo, RgMaterial *result)
{
    if (createInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    *result = textureManager->CreateMaterial
    (
        currentFrameState.GetCmdBufferForMaterials(cmdManager),
        currentFrameState.GetFrameIndex(),
        *createInfo
    );
}

void VulkanDevice::CreateAnimatedMaterial(const RgAnimatedMaterialCreateInfo *createInfo, RgMaterial *result)
{
    if (createInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    if (createInfo->frameCount == 0)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Animated materials must have non-zero amount of frames");
    }

    *result = textureManager->CreateAnimatedMaterial
    (
        currentFrameState.GetCmdBufferForMaterials(cmdManager),
        currentFrameState.GetFrameIndex(),
        *createInfo
    );
}

void VulkanDevice::ChangeAnimatedMaterialFrame(RgMaterial animatedMaterial, uint32_t frameIndex)
{
    if (!currentFrameState.WasFrameStarted())
    {
        throw RgException(RG_FRAME_WASNT_STARTED);
    }

    bool wasChanged = textureManager->ChangeAnimatedMaterialFrame(animatedMaterial, frameIndex);
}

void VulkanDevice::UpdateMaterial(const RgMaterialUpdateInfo *updateInfo)
{
    if (!currentFrameState.WasFrameStarted())
    {
        throw RgException(RG_FRAME_WASNT_STARTED);
    }

    if (updateInfo == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Argument is null");
    }

    bool wasUpdated = textureManager->UpdateMaterial(currentFrameState.GetCmdBuffer(), *updateInfo);
}

void VulkanDevice::DestroyMaterial(RgMaterial material)
{
    textureManager->DestroyMaterial(currentFrameState.GetFrameIndex(), material);
}
void VulkanDevice::CreateSkyboxCubemap(const RgCubemapCreateInfo *createInfo, RgCubemap *result)
{
    *result = cubemapManager->CreateCubemap(currentFrameState.GetCmdBufferForMaterials(cmdManager), currentFrameState.GetFrameIndex(), *createInfo);
}
void VulkanDevice::DestroyCubemap(RgCubemap cubemap)
{
    cubemapManager->DestroyCubemap(currentFrameState.GetFrameIndex(), cubemap);
}
#pragma endregion 

