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

#include "HaltonSequence.h"
#include "Matrix.h"
#include "RenderResolutionHelper.h"
#include "RgException.h"
#ifdef RG_USE_DX12
#include "DX12_CopyFramebuf.h"
#endif
#include "DX12_Interop.h"
#include "Utils.h"

#include "Generated/ShaderCommonC.h"

#include <algorithm>
#include <cstring>
#ifdef RG_USE_DX12
#include <d3d12.h>
#include <d3dx12.h>
#endif

namespace RTGL1
{
namespace
{
    SwapchainType MakeSwapchainType( const RgStartFrameRenderResolutionParams& resolution )
    {
#ifdef __APPLE__
        // frame generation is MetalFX frame interpolation (FrameInterp_Metal.mm), on the usual swapchain
        return SWAPCHAIN_TYPE_VULKAN_NATIVE;
#endif
        if( resolution.frameGeneration != RG_FRAME_GENERATION_MODE_OFF )
        {
            switch( resolution.upscaleTechnique )
            {
                case RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2:
                    return SWAPCHAIN_TYPE_FRAME_GENERATION_FSR3;

                case RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS:
                    return SWAPCHAIN_TYPE_FRAME_GENERATION_DLSS3;

                default: break;
            }
        }
        return resolution.preferDxgiPresent ? SWAPCHAIN_TYPE_DXGI //
                                            : SWAPCHAIN_TYPE_VULKAN_NATIVE;
    }
}
}

VkCommandBuffer RTGL1::VulkanDevice::BeginFrame( const RgStartFrameInfo& info )
{
    uint32_t frameIndex = currentFrameState.IncrementFrameIndexAndGet();
    timelineFrame++;

    assert( timelineFrame % dxgi::MAX_FRAMES_IN_FLIGHT_DX12 == frameIndex % MAX_FRAMES_IN_FLIGHT );

    if( !waitForOutOfFrameFence )
    {
        // wait for previous cmd with the same frame index
        Utils::WaitAndResetFence( device, frameFences[ frameIndex ] );
    }
    else
    {
        Utils::WaitAndResetFences(
            device, frameFences[ frameIndex ], outOfFrameFences[ frameIndex ] );
    }

    if( swapchain->WithDXGI() )
    {
        auto present = Semaphores_GetVkDx12Shared( dxgi::SHARED_SEM_PRESENT_COPY )
                           .value_or( dxgi::SharedSemaphore{} );
        dxgi::WaitAndPrepareForFrame( present.d3d12fence, present.d3d12fenceEvent, timelineFrame );
    }


    const auto& resolution = pnext::get< RgStartFrameRenderResolutionParams >( info );
    const auto& fluidInfo  = pnext::get< RgStartFrameFluidParams >( info );

    swapchain->AcquireImage( info.vsync,
                             info.hdr,
                             MakeSwapchainType( resolution ),
                             vkswapchainAvailableSemaphores[ frameIndex ] );
    m_skipGeneratedFrame =
        ( resolution.frameGeneration == RG_FRAME_GENERATION_MODE_WITHOUT_GENERATED );
#ifdef __APPLE__
    m_frameGenerationMode = resolution.frameGeneration;
    m_fiThisFrame         = false;
#endif


    VkSemaphore semaphoreToWaitOnSubmit = VK_NULL_HANDLE;

    // if out-of-frame cmd exist, submit it
    {
        VkCommandBuffer preFrameCmd = currentFrameState.GetPreFrameCmdAndRemove();
        if( preFrameCmd != VK_NULL_HANDLE )
        {
            // Signal inFrameSemaphore after completion.
            // Signal outOfFrameFences, but for the next frame
            // because we can't reset cmd pool with cmds (in this case
            // it's preFrameCmd) that are in use.
            cmdManager->Submit_Binary( //
                preFrameCmd,
                {},
                inFrameSemaphores[ frameIndex ],
                outOfFrameFences[ ( frameIndex + 1 ) % MAX_FRAMES_IN_FLIGHT ] );

            // should wait other semaphore in this case
            semaphoreToWaitOnSubmit = inFrameSemaphores[ frameIndex ];

            waitForOutOfFrameFence = true;
        }
        else
        {
            waitForOutOfFrameFence = false;
        }
    }
    currentFrameState.SetSemaphore( semaphoreToWaitOnSubmit );


    if( devmode && devmode->reloadShaders )
    {
        shaderManager->ReloadShaders();
        devmode->reloadShaders = false;
    }
    sceneImportExport->PrepareForFrame( Utils::SafeCstr( info.pMapName ), info.allowMapAutoExport );

    {
        renderResolution.Setup( resolution,
                                swapchain->GetWidth(),
                                swapchain->GetHeight(),
                                amdFsr2.get(),
                                swapchain->WithFSR3FrameGeneration() ? amdFsr3dx12.get() : nullptr,
                                nvDlss2.get(),
                                swapchain->WithDLSS3FrameGeneration() ? nvDlss3dx12.get()
                                                                      : nullptr,
                                nvDlssRr.get() );

        framebuffers->PrepareForSize( renderResolution.GetResolutionState(),
                                      ( swapchain->WithDXGI() ) );

        m_pixelated = resolution.pixelizedRenderSizeEnable
                          ? std::optional{ resolution.pixelizedRenderSize }
                          : std::nullopt;
    }

    // reset cmds for current frame index
    cmdManager->PrepareForFrame( frameIndex );

    // clear the data that were created MAX_FRAMES_IN_FLIGHT ago
    worldSamplerManager->PrepareForFrame( frameIndex );
    genericSamplerManager->PrepareForFrame( frameIndex );
    textureManager->PrepareForFrame( frameIndex );
    cubemapManager->PrepareForFrame( frameIndex );
    rasterizer->PrepareForFrame( frameIndex );
    {
        if( m_supportsRayQueryAndPositionFetch && fluidInfo.enabled && !fluid )
        {
            fluid = std::make_shared< Fluid >( device, //
                                               cmdManager,
                                               memAllocator,
                                               framebuffers,
                                               *shaderManager,
                                               scene->GetASManager()->GetTLASDescSetLayout(),
                                               fluidInfo.particleBudget,
                                               fluidInfo.particleRadius );
            shaderManager->Subscribe( fluid );
            framebuffers->Subscribe( fluid );
            fluid->OnFramebuffersSizeChange( renderResolution.GetResolutionState() );
        }
        else if( !fluidInfo.enabled && fluid )
        {
            fluid.reset();
        }
        fluidGravity = fluidInfo.gravity;
        fluidColor   = fluidInfo.color;
    }
    if( debugWindows )
    {
        if( !debugWindows->PrepareForFrame( frameIndex, info.vsync ) )
        {
            debugWindows.reset();
            observer.reset();
        }
    }
    if( devmode )
    {
        devmode->primitivesTable.clear();
    }

    VkCommandBuffer cmd = cmdManager->StartGraphicsCmd();
    BeginCmdLabel( cmd, "Prepare for frame" );

    textureManager->TryHotReload( cmd, frameIndex );
    lightManager->PrepareForFrame( cmd, frameIndex );
    lightManager->SetLightstyles( info );
    scene->PrepareForFrame( cmd,
                            frameIndex,
                            info.ignoreExternalGeometry ||
                                ( devmode && devmode->ignoreExternalGeometry ),
                            info.staticSceneAnimationTime );

    {
        sceneImportExport->TryImportIfNew( cmd,
                                           frameIndex,
                                           *scene,
                                           *textureManager,
                                           *textureMetaManager,
                                           *lightManager,
                                           info.pResultStaticSceneStatus );

        scene->SubmitStaticLights(
            frameIndex,
            *lightManager,
            // SHIPPING_HACK
            uniform->GetData()->volumeAllowTintUnderwater &&
                uniform->GetData()->cameraMediaType == RG_MEDIA_TYPE_WATER,
            Utils::PackColorFromFloat( uniform->GetData()->volumeUnderwaterColor ) );
    }

    {
        lightmapScreenCoverage = info.lightmapScreenCoverage < 0.01f ? 0.0f
                                 : info.lightmapScreenCoverage > 0.99f
                                     ? 1.0f
                                     : info.lightmapScreenCoverage;
    }

    if( fluid )
    {
        fluid->PrepareForFrame( fluidInfo.reset );
    }

    return cmd;
}

void RTGL1::VulkanDevice::FillUniform( RTGL1::ShGlobalUniform* gu,
                                       const RgDrawFrameInfo&  drawInfo ) const
{
    const float IdentityMat4x4[ 16 ] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    const float aspect = renderResolution.Aspect();

    const auto& cameraInfo = scene->GetCamera( renderResolution.Aspect() );

    {
        memcpy( gu->viewPrev, gu->view, 16 * sizeof( float ) );
        memcpy( gu->projectionPrev, gu->projection, 16 * sizeof( float ) );

        memcpy( gu->view, cameraInfo.view, 16 * sizeof( float ) );
        memcpy( gu->projection, cameraInfo.projection, 16 * sizeof( float ) );

        memcpy( gu->invView, cameraInfo.viewInverse, 16 * sizeof( float ) );
        memcpy( gu->invProjection, cameraInfo.projectionInverse, 16 * sizeof( float ) );

        memcpy( gu->cameraPositionPrev, gu->cameraPosition, 3 * sizeof( float ) );

        auto p = MakeCameraPosition( cameraInfo );
        {
            gu->cameraPosition[ 0 ] = p.data[ 0 ];
            gu->cameraPosition[ 1 ] = p.data[ 1 ];
            gu->cameraPosition[ 2 ] = p.data[ 2 ];
        }
    }

    {
        gu->frameId   = frameId;
        gu->timeDelta = static_cast< float >(
            std::max< double >( currentFrameTime - previousFrameTime, 0.001 ) );
        gu->time = static_cast< float >( currentFrameTime );
    }

    {
        gu->renderWidth  = static_cast< float >( renderResolution.Width() );
        gu->renderHeight = static_cast< float >( renderResolution.Height() );
        // render width must be always even for checkerboarding!
        assert( ( int )gu->renderWidth % 2 == 0 );

        gu->upscaledRenderWidth  = static_cast< float >( renderResolution.UpscaledWidth() );
        gu->upscaledRenderHeight = static_cast< float >( renderResolution.UpscaledHeight() );

        RgFloat2D jitter = { 0, 0 };
        if( renderResolution.IsNvDlssEnabled() )
        {
            jitter = HaltonSequence::GetJitter_Halton23( frameId );
        }
        else if( renderResolution.IsAmdFsr2Enabled() )
        {
            if( amdFsr3dx12 )
            {
                jitter = amdFsr3dx12->GetJitter( renderResolution.GetResolutionState(), frameId );
            }
            else if( amdFsr2 )
            {
                jitter = amdFsr2->GetJitter( renderResolution.GetResolutionState(), frameId );
            }
            else
            {
                assert( 0 );
            }
        }

        gu->jitterX = jitter.data[ 0 ];
        gu->jitterY = jitter.data[ 1 ];
    }

    {
        const auto& params = pnext::get< RgDrawFrameTonemappingParams >( drawInfo );

        float luminanceMin = std::exp2( params.ev100Min ) * 12.5f / 100.0f;
        float luminanceMax = std::exp2( params.ev100Max ) * 12.5f / 100.0f;

        gu->stopEyeAdaptation   = params.disableEyeAdaptation;
        gu->minLogLuminance     = std::log2( luminanceMin );
        gu->maxLogLuminance     = std::log2( luminanceMax );
        gu->luminanceWhitePoint = params.luminanceWhitePoint;
    }

    {
        gu->lightCount     = lightManager->GetLightCount();
        gu->lightCountPrev = lightManager->GetLightCountPrev();

        gu->directionalLightExists = lightManager->DoesDirectionalLightExist();
    }

    {
        const auto& params = pnext::get< RgDrawFrameSkyParams >( drawInfo );

        static_assert( sizeof( gu->skyCubemapRotationTransform ) == sizeof( IdentityMat4x4 ) &&
                           sizeof( IdentityMat4x4 ) == 16 * sizeof( float ),
                       "Recheck skyCubemapRotationTransform sizes" );
        memcpy( gu->skyCubemapRotationTransform, IdentityMat4x4, 16 * sizeof( float ) );


        RG_SET_VEC3_A( gu->skyColorDefault, params.skyColorDefault.data );
        gu->skyColorMultiplier = std::max( 0.0f, params.skyColorMultiplier );
        gu->skyColorSaturation = std::max( 0.0f, params.skyColorSaturation );

        // Doom64-RT: sky-reach test for the directional light. See
        // traceSunReachesSky() in RaygenCommon.h.
        gu->sunRequireSky      = params.sunRequireSky > 0.5f ? 1.0f : 0.0f;
        // Pass the MODE through, do not collapse it to a flag: 0 off, 1 isolate
        // the leak, 2 colour by reason. A `> 0.5f ? 1 : 0` here silently turned
        // every request for mode 2 into mode 1, so the colour path was
        // unreachable no matter what the cvar said.
        gu->sunLeakDebug       = std::clamp( params.sunLeakDebug, 0.0f, 2.0f );
        gu->sunLeakDebugMul    = std::max( 0.0f, params.sunLeakDebugMul );
        gu->sunSkyProbeMaxDist = params.sunSkyProbeMaxDist < 0.001f
                                     ? float( MAX_RAY_LENGTH )
                                     : params.sunSkyProbeMaxDist;
        // Doom64-RT: shade the directional light outside ReSTIR's lottery.
        gu->sunSplit           = params.sunSplit > 0.5f ? 1.0f : 0.0f;

        switch( params.skyType )
        {
            case RG_SKY_TYPE_COLOR: {
                gu->skyType = SKY_TYPE_COLOR;
                break;
            }
            case RG_SKY_TYPE_CUBEMAP: {
                gu->skyType = SKY_TYPE_CUBEMAP;
                break;
            }
            case RG_SKY_TYPE_RASTERIZED_GEOMETRY: {
                gu->skyType = SKY_TYPE_RASTERIZED_GEOMETRY;
                break;
            }
            default: gu->skyType = SKY_TYPE_COLOR;
        }

        gu->skyCubemapIndex =
            cubemapManager->TryGetDescriptorIndex( params.pSkyCubemapTextureName );

        if( !Utils::IsAlmostZero( params.skyCubemapRotationTransform ) )
        {
            Utils::SetMatrix3ToGLSLMat4( gu->skyCubemapRotationTransform,
                                         params.skyCubemapRotationTransform );
        }

        RgFloat3D skyViewerPosition = params.skyViewerPosition;

        for( uint32_t i = 0; i < 6; i++ )
        {
            float* viewProjDst = &gu->viewProjCubemap[ 16 * i ];

            Matrix::GetCubemapViewProjMat( viewProjDst,
                                           i,
                                           skyViewerPosition.data,
                                           cameraInfo.cameraNear,
                                           cameraInfo.cameraFar );
        }
    }

    gu->debugShowFlags = devmode ? devmode->debugShowFlags : 0;

    {
        const auto& params = pnext::get< RgDrawFrameTexturesParams >( drawInfo );

        gu->normalMapStrength      = params.normalMapStrength;
        gu->emissionMapBoost       = std::max( params.emissionMapBoost, 0.0f );
        gu->emissionMaxScreenColor = std::max( params.emissionMaxScreenColor, 0.0f );
        gu->minRoughness           = std::clamp( params.minRoughness, 0.0f, 1.0f );
        // Doom64-RT metalness fail-safes. Defaults (1, 0, small) are inert, so a
        // caller that never sets them behaves exactly as before.
        gu->metallicMax       = std::clamp( params.metallicMax, 0.0f, 1.0f );
        gu->spritePbr            = std::clamp( params.spritePbr, 0.0f, 1.0f );
        gu->spriteMetallicMax    = std::clamp( params.spriteMetallicMax, 0.0f, 1.0f );
        gu->spriteRoughMin       = std::clamp( params.spriteRoughMin, 0.0f, 1.0f );
        gu->spriteNormalStrength = std::clamp( params.spriteNormalStrength, 0.0f, 4.0f );
        gu->worldPbr             = std::clamp( params.worldPbr, 0.0f, 1.0f );
        gu->metallicRoughCut  = std::clamp( params.metallicRoughCut, 0.0f, 1.0f );
        gu->metallicRoughBand = std::clamp( params.metallicRoughBand, 0.0f, 1.0f );
        gu->parallaxMaxDepth       = std::max( params.heightMapDepth, 0.0f );

        // The same bias TryChangeMipLodBias hands the sampler below: the
        // three-point path picks its own mip and must cancel the sampler's.
        gu->texFilterParams[ 0 ] = params.albedoFilterMode == 1 ? 1.0f : 0.0f;
        gu->texFilterParams[ 1 ] = renderResolution.GetMipLodBias( params.mipLodBiasOffset );
        gu->texFilterParams[ 2 ] = 0.0f;
        gu->texFilterParams[ 3 ] = 0.0f;
    }

    // Dev Materials A/B — also force uniforms even if Override did not rewrite draw params.
    // Flags: bit0=N, bit1=emis, bit2=metallic, bit3=H, bit4=roughness
    {
        uint32_t stripFlags = 0;
        if( devmode && devmode->materialStripNormals )
        {
            stripFlags |= 1u;
            gu->normalMapStrength = 0.0f;
        }
        if( devmode && devmode->materialStripEmissives )
        {
            stripFlags |= 2u;
            gu->emissionMapBoost       = 0.0f;
            gu->emissionMaxScreenColor = 0.0f;
        }
        if( devmode && devmode->materialStripMetallic )
        {
            stripFlags |= 4u;
        }
        // Doom64-RT: the same strip, but requested by the application rather than
        // by the Dev window, so gzdoom can put it behind a cvar (rt_metallic 0).
        if( pnext::get< RgDrawFrameTexturesParams >( drawInfo ).forceNonMetallic )
        {
            stripFlags |= 4u;
        }
        if( devmode && devmode->materialStripHeight )
        {
            stripFlags |= 8u;
            gu->parallaxMaxDepth = 0.0f;
        }
        if( devmode && devmode->materialStripRoughness )
        {
            stripFlags |= 16u;
        }
        gu->materialStripFlags = stripFlags;

        gu->materialRoughnessTowardMatte =
            ( devmode ? std::clamp( devmode->roughnessTowardMatte, 0.0f, 1.0f ) : 0.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameIlluminationParams >( drawInfo );

        gu->maxBounceShadowsLights     = params.maxBounceShadows;
        gu->polyLightSpotlightFactor   = std::max( 0.0f, params.polygonalLightSpotlightFactor );
        // [1,4], not the API's advertised 8: emissionMapBoost runs at 200 in
        // this game and emis * 200 * albedo^k compounding over depth is a
        // firefly risk. Widen after measurement, not before.
        gu->indirectBounces      = std::clamp( params.indirectBounces, 1u, 4u );
        gu->indirectLegacyWeight = !!params.indirectLegacyBounceWeight;
        gu->lightIndexIgnoreFPVShadows = lightManager->GetLightIndexForShaders(
            currentFrameState.GetFrameIndex(), params.lightUniqueIdIgnoreFirstPersonViewerShadows );
        gu->cellWorldSize       = std::max( params.cellWorldSize, 0.001f );
        gu->gradientMultDiffuse = std::clamp( params.directDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu->gradientMultIndirect =
            std::clamp( params.indirectDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu->gradientMultSpecular = std::clamp( params.specularSensitivityToChange, 0.0f, 1.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameBloomParams >( drawInfo );

        gu->bloomThreshold    = std::max( params.inputThreshold, 0.0f );
        gu->bloomIntensity    = 0.2f * std::max( params.bloomIntensity, 0.0f );
        gu->bloomEV           = std::max( params.inputEV, 0.0f );
        gu->lensDirtIntensity = std::max( params.lensDirtIntensity, 0.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameReflectRefractParams >( drawInfo );

        switch( params.typeOfMediaAroundCamera )
        {
            case RG_MEDIA_TYPE_VACUUM: gu->cameraMediaType = MEDIA_TYPE_VACUUM; break;
            case RG_MEDIA_TYPE_WATER: gu->cameraMediaType = MEDIA_TYPE_WATER; break;
            case RG_MEDIA_TYPE_GLASS: gu->cameraMediaType = MEDIA_TYPE_GLASS; break;
            case RG_MEDIA_TYPE_ACID: gu->cameraMediaType = MEDIA_TYPE_ACID; break;
            default: gu->cameraMediaType = MEDIA_TYPE_VACUUM;
        }

        gu->reflectRefractMaxDepth = std::min( 16u, params.maxReflectRefractDepth );

        gu->indexOfRefractionGlass = std::max( 0.0f, params.indexOfRefractionGlass );
        gu->indexOfRefractionWater = std::max( 0.0f, params.indexOfRefractionWater );
        gu->thinMediaWidth         = std::max( 0.0f, params.thinMediaWidth );

        memcpy( gu->waterColorAndDensity, params.waterColor.data, 3 * sizeof( float ) );
        gu->waterColorAndDensity[ 3 ] = 0.0f;

        memcpy( gu->acidColorAndDensity, params.acidColor.data, 3 * sizeof( float ) );
        gu->acidColorAndDensity[ 3 ] = std::max( 0.0f, params.acidDensity );

        gu->waterWaveSpeed    = params.waterWaveSpeed;
        gu->waterWaveStrength = params.waterWaveNormalStrength;
        gu->waterTextureDerivativesMultiplier =
            std::max( 0.0f, params.waterWaveTextureDerivativesMultiplier );
        gu->waterTextureAreaScale =
            params.waterTextureAreaScale < 0.0001f ? 1.0f : params.waterTextureAreaScale;

        gu->twirlPortalNormal = !!params.portalNormalTwirl;

        // Doom64-RT: stylized water
        gu->stylizedWaterStrength  = std::max( 0.0f, params.stylizedWaterStrength );
        gu->stylizedWaterCaustic   = std::max( 0.0f, params.stylizedWaterCaustic );
        gu->stylizedWaterReflMax   = std::clamp( params.stylizedWaterReflMax, 0.0f, 1.0f );
        gu->stylizedWaterRoughness = std::clamp( params.stylizedWaterRoughness, 0.0f, 1.0f );
        gu->stylizedWaterGlow      = std::max( 0.0f, params.stylizedWaterGlow );
        gu->stylizedWaterVeinRef   = std::max( 0.0001f, params.stylizedWaterVeinRef );
        // vec4 in std140: copy the 3 colour floats, zero the pad.
        for( int i = 0; i < 4; i++ )
        {
            memcpy( &gu->stylizedLiquidTint[ i * 4 ],
                    params.stylizedLiquidTint[ i ].data,
                    3 * sizeof( float ) );
            gu->stylizedLiquidTint[ i * 4 + 3 ] = 0.0f;

            memcpy( &gu->stylizedLiquidCrest[ i * 4 ],
                    params.stylizedLiquidCrest[ i ].data,
                    3 * sizeof( float ) );
            gu->stylizedLiquidCrest[ i * 4 + 3 ] = 0.0f;

            // These two are ONE vec4 each, holding a scalar per liquid -- not
            // four vec4s. Indexed in the shader as stylizedLiquidRelief[id].
            gu->stylizedLiquidRelief[ i ] =
                std::clamp( params.stylizedLiquidRelief[ i ], 0.0f, 1.0f );
            gu->stylizedLiquidFlow[ i ] = std::max( 0.0f, params.stylizedLiquidFlow[ i ] );
            gu->stylizedLiquidCaustics[ i ] =
                std::max( 0.0f, params.stylizedLiquidCaustics[ i ] );
            gu->stylizedLiquidRefl[ i ] = std::max( 0.0f, params.stylizedLiquidRefl[ i ] );
            // NOT clamped up from 0: <= 0 is the "use the global" sentinel.
            gu->stylizedLiquidRough[ i ] =
                std::min( params.stylizedLiquidRough[ i ], 1.0f );
        }
        gu->liquidNoSplit   = params.liquidNoSplit != 0.0f ? 1.0f : 0.0f;
        gu->liquidFlowSpeed = params.liquidFlowSpeed;
        gu->liquidFlowScale  = std::max( 0.01f, params.liquidFlowScale );
        gu->liquidFlowAspect = std::max( 0.1f, params.liquidFlowAspect );
        gu->liquidFlowDebug = params.liquidFlowDebug;
        gu->lavaEmisBoost          = std::max( 0.0f, params.lavaEmisBoost );
        gu->lavaFlowStrength       = std::clamp( params.lavaFlowStrength, 0.0f, 1.0f );
        gu->lavaFlowSpeed          = params.lavaFlowSpeed;
        gu->lavaFlowScale          = std::max( 0.0f, params.lavaFlowScale );
        gu->lavaFlowPixel          = std::max( 0.0f, params.lavaFlowPixel );
        gu->lavaPulse              = std::clamp( params.lavaPulse, 0.0f, 1.0f );
        gu->lavaPulseSpeed         = params.lavaPulseSpeed;
        gu->lavaGiBoost            = std::max( 0.0f, params.lavaGiBoost );
        gu->lavaDebug              = std::max( 0.0f, params.lavaDebug );

        memcpy( gu->lavaTint, params.lavaTint.data, 3 * sizeof( float ) );
        gu->lavaTint[ 3 ] = 0.0f;
        gu->stylizedWaterDebug     = std::max( 0.0f, params.stylizedWaterDebug );
        gu->stylizedWaterReflMin   = std::clamp( params.stylizedWaterReflMin, 0.0f, 1.0f );
        gu->waterCausticGain       = std::max( 0.0f, params.waterCausticGain );
        gu->waterCausticScale      = std::max( 0.0f, params.waterCausticScale );
        gu->waterCausticSpeed      = params.waterCausticSpeed;
        gu->waterCausticDist       = std::max( 0.0f, params.waterCausticDist );
        gu->waterCausticRise       = std::max( 0.0f, params.waterCausticRise );
        gu->waterCausticSlant      = std::max( 0.0f, params.waterCausticSlant );
        gu->waterCausticWallBoost  = std::max( 0.0f, params.waterCausticWallBoost );
    }

    gu->rayCullBackFaces  = rayCullBackFacingTriangles ? 1 : 0;
    gu->rayLength         = clamp( drawInfo.rayLength, 0.1f, float( MAX_RAY_LENGTH ) );
    gu->primaryRayMinDist = clamp( cameraInfo.cameraNear, 0.001f, gu->rayLength );

    {
        gu->rayCullMaskWorld =
            INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_WORLD_1 | INSTANCE_MASK_WORLD_2;

        // skip shadows for:
        // WORLD_1 - 'no shadows' geometry
        // WORLD_2 - 'sky' geometry
        //
        // Doom64-RT: RESERVED_0 is 'shadow only' -- geometry that exists ONLY
        // here. It is absent from rayCullMaskWorld above, so no other ray can
        // see it; this line is the single place it is switched on.
        gu->rayCullMaskWorld_Shadow = INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_RESERVED_0;
    }

    gu->waterNormalTextureIndex = textureManager->GetWaterNormalTextureIndex();
    gu->dirtMaskTextureIndex    = textureManager->GetDirtMaskTextureIndex();

    gu->cameraRayConeSpreadAngle = atanf( ( 2.0f * tanf( cameraInfo.fovYRadians * 0.5f ) ) /
                                          float( renderResolution.Height() ) );

    RG_SET_VEC3_A( gu->worldUpVector, sceneImportExport->GetWorldUp().data );

    gu->lightmapScreenCoverage = lightmapScreenCoverage;

    {
        gu->fluidEnabled = fluid && fluid->Active();
        RG_SET_VEC3_A( gu->fluidColor, fluidColor.data );
    }

    {
        const auto& params = pnext::get< RgDrawFrameVolumetricParams >( drawInfo );

        gu->volumeCameraNear = std::max( cameraInfo.cameraNear, 0.001f );
        gu->volumeCameraFar  = std::min( cameraInfo.cameraFar, params.volumetricFar );

        {
            if( params.enable )
            {
                gu->volumeEnableType =
                    params.useSimpleDepthBased ? VOLUME_ENABLE_SIMPLE : VOLUME_ENABLE_VOLUMETRIC;
            }
            else
            {
                gu->volumeEnableType = VOLUME_ENABLE_NONE;
            }
            gu->volumeScattering = params.scaterring;
            gu->volumeAsymmetry  = std::clamp( params.assymetry, -1.0f, 1.0f );

            RG_SET_VEC3_A( gu->volumeAmbient, params.ambientColor.data );
            RG_MAX_VEC3( gu->volumeAmbient, 0.0f );

#if ILLUMINATION_VOLUME
            gu->illumVolumeEnable = params.useIlluminationVolume;
#else
            gu->illumVolumeEnable = 0;
#endif

            if( auto uniqueId = scene->TryGetVolumetricLight( *lightManager,
                                                              MakeCameraPosition( cameraInfo ) ) )
            {
                gu->volumeLightSourceIndex = lightManager->GetLightIndexForShaders(
                    currentFrameState.GetFrameIndex(), &uniqueId.value() );
            }
            else
            {
                gu->volumeLightSourceIndex = LIGHT_INDEX_NONE;
            }

            RG_SET_VEC3_A( gu->volumeFallbackSrcColor, params.fallbackSourceColor.data );
            RG_MAX_VEC3( gu->volumeFallbackSrcColor, 0.0f );

            RG_SET_VEC3_A( gu->volumeFallbackSrcDirection, params.fallbackSourceDirection.data );

            gu->volumeFallbackSrcExists = Utils::TryNormalize( gu->volumeFallbackSrcDirection ) &&
                                          ( gu->volumeFallbackSrcColor[ 0 ] > 0.01f &&
                                            gu->volumeFallbackSrcColor[ 1 ] > 0.01f &&
                                            gu->volumeFallbackSrcColor[ 2 ] > 0.01f );

            gu->volumeLightMult = std::max( 0.0f, params.lightMultiplier );

            // Doom64-RT: illuminated fog. The all-lights froxel estimate lives
            // in the ILLUMINATION_VOLUME branch of RtVolumetric.rgen, so it is
            // only available when RTGL was built with that on.
#if ILLUMINATION_VOLUME
            gu->volumeAllLights = params.illuminateFromAllLights;
#else
            gu->volumeAllLights = 0;
#endif
            RG_SET_VEC3_A( gu->volumeMediaColor, params.mediaColor.data );
            RG_MAX_VEC3( gu->volumeMediaColor, 0.0f );
            RG_SET_VEC3_A( gu->volumeMediaColorFar, params.mediaColorFar.data );
            RG_MAX_VEC3( gu->volumeMediaColorFar, 0.0f );
            // Negative farScattering means "uniform": the caller did not ask for
            // a ramp, so the far end is the near end. Cheaper to answer here
            // than to make every caller restate the density twice.
            gu->volumeScatteringFar =
                params.farScattering < 0.0f ? gu->volumeScattering : params.farScattering;
            gu->volumeDensityCurve   = std::max( 0.01f, params.densityCurve );
            gu->volumeLightNearFade  = std::max( 0.0f, params.lightNearFade );
            gu->volumeSpatialBlur    = std::clamp( params.spatialBlur, 0.0f, 1.0f );
            gu->volumeDither         = std::max( 0.0f, params.ditherRadius );
            gu->volumeDitherZ        = std::max( 0.0f, params.ditherRadiusZ );
            gu->volumeOccludeEmis    = params.occludeEmission;

            // Doom64-RT: the froxel depth gate. See volume_depthGate() in
            // RtVolumetric.rgen -- this stops the volume lighting air the camera
            // cannot see, which is what the trilinear read of a prefix sum
            // otherwise smears through every wall.
            gu->volumeDepthGate     = params.depthGate > 0.5f ? 1.0f : 0.0f;
            gu->volumeDepthGateBias = params.depthGateBias;
            // Clamped ABOVE zero: a feather of exactly 0 is a hard binary cut
            // and paints the froxel grid onto every surface, which is the
            // artefact this feature is supposed to remove rather than draw.
            gu->volumeDepthGateFeather = std::max( 0.01f, params.depthGateFeather );
            // 1 or 5, nothing between: the shader's tap array is fixed-size and
            // a value it does not recognise must not silently mean "centre
            // only", which would look like the gate being too aggressive.
            gu->volumeDepthGateTaps = params.depthGateTaps >= 5 ? 5u : 1u;

            // Doom64-RT: the upscaler bias mask (CmPrepareFinal writes it into
            // FB_IMAGE_INDEX_REACTIVITY, DLSS2/FSR2 hand it to the upscaler).
            // Clamped to 0..1 because both APIs read the mask as an alpha and a
            // value outside that range is undefined rather than merely strong.
            gu->volumeUpscaleBias      = std::clamp( params.volumeUpscaleBias, 0.0f, 1.0f );
            // Clamped ABOVE zero: this is a divisor in the shader, and an edge
            // scale of 0 would mark every pixel with any transmittance gradient
            // at all -- i.e. the whole veil, which is the noisy arm.
            gu->volumeUpscaleBiasEdge  = std::max( 0.001f, params.volumeUpscaleBiasEdge );
            gu->volumeUpscaleBiasFloor = std::clamp( params.volumeUpscaleBiasFloor, 0.0f, 1.0f );
            gu->volumeUpscaleBiasDebug = params.volumeUpscaleBiasDebug ? 1u : 0u;

            // Doom64-RT: post-upscale composite, GATED HOST-SIDE.
            //
            // Two paths must keep the old ordering or they break rather than
            // improve:
            //
            //   DLSS RAY RECONSTRUCTION denoises from the composed radiance --
            //   handing it a frame with no medium in it changes what it is
            //   denoising, not just when the veil lands.
            //
            //   FRAME GENERATION interpolates frames inside its own technique,
            //   after this. Only the real frames would come through this pass,
            //   so every generated frame would be missing its fog and the whole
            //   image would strobe at half the frame rate.
            //
            // The gate is here rather than in the shader so the flag the shader
            // reads and the pass the device runs cannot disagree.
            const bool postCompOk = !renderResolution.IsNvDlssRayReconstructionEnabled() &&
                                    !swapchain->WithDLSS3FrameGeneration() &&
                                    !swapchain->WithFSR3FrameGeneration();

            gu->volumePostComp = ( params.volumePostComp && postCompOk ) ? 1u : 0u;
            gu->volumeEdgeSoft = std::max( 0.0f, params.volumeEdgeSoft );
            // Clamped above zero: this is the upper end of a smoothstep whose
            // lower end is half of it, and a threshold of 0 would mark every
            // pixel in the frame as a silhouette.
            gu->volumeEdgeSoftEdge = std::max( 0.001f, params.volumeEdgeSoftEdge );
            gu->volumeFp           = std::clamp( params.volumeFp, 0.0f, 2.0f );
            gu->volumeReproj       = params.volumeReproj ? 1u : 0u;
            gu->volumeSpriteShadow = params.volumeSpriteShadow ? 1u : 0u;
            gu->volumeGridHistory  = std::clamp( params.volumeGridHistory, 0.0f, 64.0f );
            // Doom64-RT: rrGlowPre took the volumeReserved3 spare. Only read by
            // the shaders when rrPreExposure is active, so the raw request is
            // fine here (no host gate needed). 0=post-add, 1=/exposure,
            // 2=fixed rrGlowScale ("glow as light").
            {
                const auto& illumGlow =
                    pnext::get< RgDrawFrameIlluminationParams >( drawInfo );
                gu->rrGlowPre   = std::min( illumGlow.rrGlowPre, 2u );
                gu->rrGlowScale = std::max( illumGlow.rrGlowScale, 0.0f );
            }

            gu->volumeAllowTintUnderwater = params.allowTintUnderwater;
            RG_SET_VEC3_A( gu->volumeUnderwaterColor, params.underwaterColor.data );
            RG_MAX_VEC3( gu->volumeUnderwaterColor, 0.0f );
        }

        if( gu->volumeEnableType != VOLUME_ENABLE_NONE )
        {
            memcpy( gu->volumeViewProj_Prev, gu->volumeViewProj, 16 * sizeof( float ) );
            memcpy( gu->volumeViewProjInv_Prev, gu->volumeViewProjInv, 16 * sizeof( float ) );

            float volumeproj[ 16 ];
            Matrix::MakeProjectionMatrix( volumeproj,
                                          aspect,
                                          cameraInfo.fovYRadians,
                                          gu->volumeCameraNear,
                                          gu->volumeCameraFar );

            Matrix::Multiply( gu->volumeViewProj, gu->view, volumeproj );
            Matrix::Inverse( gu->volumeViewProjInv, gu->volumeViewProj );
        }
    }

    // Doom64-RT: LOCALISED SMOKE. A separate block after the volumetric one, and
    // a separate pNext struct, so a frame that never links it -- which is every
    // frame on every map until someone fires a gun -- lands on puffCount 0 and
    // the froxel shader collapses back to the fog's arithmetic exactly.
    {
        const auto& params = pnext::get< RgDrawFrameSmokeParams >( drawInfo );

        const uint32_t count = params.pPuffs && params.pAlbedoDensity
                                   ? std::min( params.puffCount,
                                               uint32_t{ SMOKE_PUFF_MAX } )
                                   : 0u;

        gu->smokeCount = count;

        for( uint32_t i = 0; i < count; i++ )
        {
            // xyz = centre in metres, w = radius. A radius of 0 would divide by
            // zero in smoke_evalAt; the shader guards it, but drop it here too
            // so a bad puff costs nothing rather than costing a branch per cell.
            RG_SET_VEC3_A( &gu->smokePuffs[ i * 4 ], params.pPuffs[ i ].data );
            gu->smokePuffs[ i * 4 + 3 ] = std::max( 0.0f, params.pPuffs[ i ].data[ 3 ] );

            RG_SET_VEC3_A( &gu->smokeAlbedoDensity[ i * 4 ], params.pAlbedoDensity[ i ].data );
            RG_MAX_VEC3( &gu->smokeAlbedoDensity[ i * 4 ], 0.0f );
            gu->smokeAlbedoDensity[ i * 4 + 3 ] =
                std::max( 0.0f, params.pAlbedoDensity[ i ].data[ 3 ] );

            // Across-view radius. Falls back to the along-view one, which makes
            // the puff a sphere and is what every caller that does not set
            // pShape gets.
            gu->smokeShape[ i * 4 ] =
                params.pShape ? std::max( 0.001f, params.pShape[ i ].data[ 0 ] )
                              : std::max( 0.001f, params.pPuffs[ i ].data[ 3 ] );
        }
        // Doom64-RT: proves the pNext link and the uniform copy, which is the one
        // step the engine-side log cannot see. If stage C says "sent" and this
        // never prints, the struct is not reaching the library.
        //
        // AFTER the writes, deliberately. Placed before them it read the
        // PREVIOUS frame's uniform and reported puff0 as all zeros on the first
        // line -- an artifact of the instrumentation that looked exactly like
        // the bug it was meant to find.
        {
            // GATED ON params.debugMode, which is rt_smoke_debug. It was not, and
            // that made these three unstoppable: the only condition was "smoke
            // exists", so the normal launcher printed D/received and E/space every
            // 60 frames for the whole game and rt_smoke_debug 0 did nothing about
            // it -- the cvar the player would reach for lives in the engine, and
            // this instrumentation is in the library. Reported from play.
            //
            // The F/layout canary below is gated too. It is one-shot and genuinely
            // useful (it catches the stale-.obj trap), but a one-shot line nobody
            // asked for is still a line nobody asked for, and build-rtgl.cmd now
            // clears objects anyway.
            static uint32_t s_dbg = 0;
            if( params.debugMode > 0 && count > 0 && ( s_dbg++ % 60 ) == 0 )
            {
                debug::Warning( "rt_smoke D/received: count={} puff0=({:.2f},{:.2f},{:.2f}) "
                                "r={:.2f} density={:.1f} allLights={} nearFade={:.2f} "
                                "blend={:.2f} | volumeNear={:.2f} volumeFar={:.2f} "
                                "enableType={} scattering={:.3f} || UNIFORM: count={} "
                                "debug={} allLights={} puff0=({:.2f},{:.2f},{:.2f},{:.2f}) "
                                "albden0=({:.2f},{:.2f},{:.2f},{:.1f})",
                                count,
                                params.pPuffs[ 0 ].data[ 0 ],
                                params.pPuffs[ 0 ].data[ 1 ],
                                params.pPuffs[ 0 ].data[ 2 ],
                                params.pPuffs[ 0 ].data[ 3 ],
                                params.pAlbedoDensity[ 0 ].data[ 3 ],
                                params.allLights,
                                params.lightNearFade,
                                params.illumBlend,
                                gu->volumeCameraNear,
                                gu->volumeCameraFar,
                                gu->volumeEnableType,
                                gu->volumeScattering,
                                gu->smokeCount,
                                gu->smokeDebug,
                                gu->smokeAllLights,
                                gu->smokePuffs[ 0 ],
                                gu->smokePuffs[ 1 ],
                                gu->smokePuffs[ 2 ],
                                gu->smokePuffs[ 3 ],
                                gu->smokeAlbedoDensity[ 0 ],
                                gu->smokeAlbedoDensity[ 1 ],
                                gu->smokeAlbedoDensity[ 2 ],
                                gu->smokeAlbedoDensity[ 3 ] );

                // The froxel centres are built around gu->cameraPosition, which
                // comes from the INVERSE VIEW MATRIX -- not from
                // RgCameraInfo::position. If those two spaces differ, a puff in
                // engine metres can never be within a radius of any centre, and
                // the sphere test fails while every other read succeeds. That is
                // exactly the symptom: probe 3 green, probe 2 blank.
                const float dx = gu->smokePuffs[ 0 ] - gu->cameraPosition[ 0 ];
                const float dy = gu->smokePuffs[ 1 ] - gu->cameraPosition[ 1 ];
                const float dz = gu->smokePuffs[ 2 ] - gu->cameraPosition[ 2 ];
                // ONE-SHOT. This is the canary for the stale-object trap: if a
                // .obj keeps an older struct layout, these offsets stop matching
                // the SPIR-V ones quoted beside them and every field past the old
                // size silently reads zero. See tools/build-rtgl.cmd.
                static bool s_layoutLogged = false;
                if( !s_layoutLogged )
                {
                    s_layoutLogged = true;
                debug::Warning( "rt_smoke F/layout: sizeof(ShGlobalUniform)={} "
                                "offsetof smokeCount={} smokeDebug={} smokePuffs={} "
                                "smokeAlbedoDensity={}  (SPIR-V says 1520 / 1536 / 1552 / 2064)",
                                sizeof( ShGlobalUniform ),
                                offsetof( ShGlobalUniform, smokeCount ),
                                offsetof( ShGlobalUniform, smokeDebug ),
                                offsetof( ShGlobalUniform, smokePuffs ),
                                offsetof( ShGlobalUniform, smokeAlbedoDensity ) );
                }

                debug::Warning( "rt_smoke E/space: cameraPosition=({:.2f},{:.2f},{:.2f}) "
                                "puff0=({:.2f},{:.2f},{:.2f}) "
                                "|puff-camPos|={:.2f}m  (should match the engine's dist)",
                                gu->cameraPosition[ 0 ],
                                gu->cameraPosition[ 1 ],
                                gu->cameraPosition[ 2 ],
                                gu->smokePuffs[ 0 ],
                                gu->smokePuffs[ 1 ],
                                gu->smokePuffs[ 2 ],
                                std::sqrt( dx * dx + dy * dy + dz * dz ) );
            }
        }

        // Only meaningful inside a puff -- RtVolumetric.rgen picks between these
        // and the fog's values PER FROXEL, so writing them costs a fog frame
        // nothing.
        gu->smokeLightNearFade = std::max( 0.0f, params.lightNearFade );
        gu->smokeIllumBlend    = std::clamp( params.illumBlend, 0.0f, 1.0f );
#if ILLUMINATION_VOLUME
        gu->smokeAllLights = count > 0 ? params.allLights : 0;
#else
        gu->smokeAllLights = 0;
#endif
        gu->smokeDebug        = params.debugMode;
        gu->smokeLightFarFade = std::max( 0.0f, params.lightFarFade );
        gu->smokeMaxLight     = std::max( 0.0f, params.maxLight );
        gu->smokeSpp          = std::clamp( params.samplesPerCell, 1u, 32u );

        gu->smokeStylize      = std::clamp( params.stylize, 0.0f, 1.0f );
        gu->smokeStylizeSteps = std::clamp( params.stylizeSteps, 1u, 64u );
        gu->smokeStylizeGrid  = std::max( 0.0f, params.stylizeGrid );
        gu->smokeAmbient      = std::max( 0.0f, params.selfAmbient );
        gu->smokeTintBias     = std::clamp( params.tintBias, 0.0f, 1.0f );
        gu->smokeAbsorb       = std::max( 0.0f, params.absorb );
    }

    // Doom64-RT: LIGHT SHAFTS FROM ORDINARY LAMPS. See
    // RgDrawFrameLightShaftParams -- the froxel pass scatters exactly one light
    // (the sun), so beams only ever happen outdoors. This turns a short caller-
    // supplied list of uniqueIDs into shader light indices; RtVolumetric.rgen
    // adds their scattering on top of the single-light term.
    //
    // AFTER the smoke block on purpose, for the same reason smoke is after the
    // volumetric one: a frame that never links the struct lands on count 0 and
    // the volume behaves exactly as it did.
    {
        const auto& params = pnext::get< RgDrawFrameLightShaftParams >( drawInfo );

        const uint32_t wanted =
            params.pLightUniqueIds
                ? std::min( params.count, uint32_t{ VOLUME_SHAFT_LIGHT_MAX } )
                : 0u;

        // Resolved COMPACTLY, not in place. A fixture the caller listed may have
        // been culled before its light was uploaded -- the lamp walks run before
        // the frame's own budget is known -- and leaving a LIGHT_INDEX_NONE hole
        // in the middle of the array would spend the shader's nearest-first
        // budget on nothing. The list arrives sorted by distance, so dropping
        // the misses keeps that order.
        uint32_t resolved = 0;
        for( uint32_t i = 0; i < wanted; i++ )
        {
            const uint64_t id  = params.pLightUniqueIds[ i ];
            const uint32_t idx = lightManager->GetLightIndexForShaders(
                currentFrameState.GetFrameIndex(), &id );

            if( idx != LIGHT_INDEX_NONE )
            {
                gu->volumeShaftLights[ resolved++ ] = idx;
            }
        }

        for( uint32_t i = resolved; i < uint32_t{ VOLUME_SHAFT_LIGHT_MAX }; i++ )
        {
            gu->volumeShaftLights[ i ] = LIGHT_INDEX_NONE;
        }

        gu->volumeShaftCount       = resolved;
        gu->volumeShaftMult        = std::max( 0.0f, params.multiplier );
        gu->volumeShaftNearFade    = std::max( 0.0f, params.nearFade );
        gu->volumeShaftMinRadiance = std::max( 0.0f, params.minRadiance );
        // At least one, or a non-empty list would be walked and never traced --
        // a silent "the cvar does nothing" of exactly the kind this project has
        // paid for before. Clamped to the list size for the same reason.
        gu->volumeShaftMaxTraced =
            std::clamp( params.maxTraced, 1u, uint32_t{ VOLUME_SHAFT_LIGHT_MAX } );
        // Below -1 means "share the volume's". Resolved HERE rather than in the
        // shader so the two cannot drift, and so the sentinel never reaches a
        // phase function that would take k out of range.
        gu->volumeShaftAsym  = params.asymmetry < -1.0f
                                   ? gu->volumeAsymmetry
                                   : std::clamp( params.asymmetry, -1.0f, 1.0f );
        gu->volumeShaftDebug = std::min( params.debugMode, 3u );
        // Clamped at 2: that is already "no falloff at all", and past it a lamp
        // would get BRIGHTER with distance, which is not a look, it is a bug
        // waiting to be reported as one.
        gu->volumeShaftFalloff = std::clamp( params.falloffCompensation, 0.0f, 2.0f );
        gu->volumeShaftRelCull = std::clamp( params.relativeCull, 0.0f, 1.0f );
    }

    // Doom64-RT: VOLUMETRIC CLOUDS. See RgDrawFrameVolumetricCloudParams and
    // Shaders/Clouds.h for the packing. A frame that never links the struct
    // lands on enabled=0, and then nothing here is read by any shader.
    {
        const auto& p = pnext::get< RgDrawFrameVolumetricCloudParams >( drawInfo );

        const bool on = p.enabled && p.thickness > 0.0f && p.density > 0.0f;

        gu->cloudParams0[ 0 ] = on ? 1.0f : 0.0f;
        gu->cloudParams0[ 1 ] = std::max( p.altitude, 1.0f );
        gu->cloudParams0[ 2 ] = std::max( p.thickness, 1.0f );
        gu->cloudParams0[ 3 ] = std::clamp( p.coverage, 0.0f, 1.0f );

        gu->cloudParams1[ 0 ] = std::max( p.density, 0.0f );
        gu->cloudParams1[ 1 ] = 1.0f / std::max( p.featureSize, 1.0f );
        gu->cloudParams1[ 2 ] = std::clamp( p.detail, 0.0f, 1.0f );
        gu->cloudParams1[ 3 ] = p.time;

        gu->cloudParams2[ 0 ] = float( std::clamp( p.steps, 4u, 128u ) );
        gu->cloudParams2[ 1 ] = float( std::clamp( p.lightSteps, 1u, 16u ) );
        gu->cloudParams2[ 2 ] = std::max( p.horizonFade, 0.01f );
        gu->cloudParams2[ 3 ] = std::clamp( p.historyBlend, 0.0f, 0.97f );

        gu->cloudParams3[ 0 ] = std::clamp( p.transmitFloor, 0.0f, 1.0f );
        gu->cloudParams3[ 1 ] = std::clamp( p.asymmetry, -0.95f, 0.95f );
        gu->cloudParams3[ 2 ] = float( std::min( p.debugMode, 3u ) );
        gu->cloudParams3[ 3 ] = p.wind.data[ 0 ];

        gu->cloudTint[ 0 ] = std::max( p.tint.data[ 0 ], 0.0f );
        gu->cloudTint[ 1 ] = std::max( p.tint.data[ 1 ], 0.0f );
        gu->cloudTint[ 2 ] = std::max( p.tint.data[ 2 ], 0.0f );
        gu->cloudTint[ 3 ] = p.wind.data[ 1 ];

        // Normalised here, once; a zero vector becomes "straight up" rather
        // than NaN in every texel.
        {
            float d[ 3 ] = { p.lightDir.data[ 0 ], p.lightDir.data[ 1 ], p.lightDir.data[ 2 ] };
            float len    = std::sqrt( d[ 0 ] * d[ 0 ] + d[ 1 ] * d[ 1 ] + d[ 2 ] * d[ 2 ] );
            if( len < 1e-6f )
            {
                d[ 0 ] = 0.0f;
                d[ 1 ] = 1.0f;
                d[ 2 ] = 0.0f;
                len    = 1.0f;
            }
            gu->cloudLightDir[ 0 ] = d[ 0 ] / len;
            gu->cloudLightDir[ 1 ] = d[ 1 ] / len;
            gu->cloudLightDir[ 2 ] = d[ 2 ] / len;
            gu->cloudLightDir[ 3 ] = std::max( p.underStrength, 0.0f );
        }

        for( int c = 0; c < 3; c++ )
        {
            gu->cloudLightColor[ c ] = std::max( p.lightColor.data[ c ], 0.0f );
            gu->cloudUnderColor[ c ] = std::max( p.underColor.data[ c ], 0.0f );
            gu->cloudAmbient[ c ]    = std::max( p.ambient.data[ c ], 0.0f );
        }
        // The spare .w lanes carry the directional-light occlusion (Light.h):
        // lightColor.w = transmit of an opaque column, underColor.w = on/off,
        // ambient.w = how much of the occlusion to apply.
        gu->cloudLightColor[ 3 ] = std::clamp( p.lightTransmit, 0.0f, 1.0f );
        gu->cloudUnderColor[ 3 ] = ( on && p.sunOcclusion ) ? 1.0f : 0.0f;
        gu->cloudAmbient[ 3 ]    = std::clamp( p.lightOcclude, 0.0f, 1.0f );
        for( int c = 0; c < 3; c++ )
        {
            gu->cloudBackColor[ c ] = std::max( p.backColor.data[ c ], 0.0f );
        }
        gu->cloudBackColor[ 3 ] = std::max( p.backStrength, 0.0f );
        gu->cloudFireParams[ 0 ] = std::max( p.fireStrength, 0.0f );
        gu->cloudFireParams[ 1 ] = 1.0f / std::max( p.fireScale, 1.0f );
        gu->cloudFireParams[ 2 ] = 1.0f - std::clamp( p.fireCover, 0.0f, 1.0f );
        gu->cloudFireParams[ 3 ] = std::max( p.fireLit, 0.0f );
        gu->cloudLayerParams[ 0 ] = p.layers >= 2 ? 2.0f : 1.0f;
        gu->cloudLayerParams[ 1 ] = std::clamp( p.gapFraction, 0.02f, 0.8f );
        gu->cloudLayerParams[ 2 ] = std::max( p.sheetExtinction, 0.0f );
        gu->cloudLayerParams[ 3 ] = 0.0f;
        gu->cloudFireAnim[ 0 ] = std::clamp( p.firePulse, 0.0f, 1.0f );
        gu->cloudFireAnim[ 1 ] = std::max( p.firePulseSpeed, 0.0f );
        gu->cloudFireAnim[ 2 ] = std::clamp( p.fireFlicker, 0.0f, 1.0f );
        gu->cloudFireAnim[ 3 ] = 1.0f / std::max( p.cascadeWidth, 1.0f );
        gu->cloudCascade[ 0 ]  = std::max( p.cascadeStrength, 0.0f );
        gu->cloudCascade[ 1 ]  = 1.0f / std::max( p.cascadeLength, 1.0f );
        gu->cloudCascade[ 2 ]  = 1.0f - std::clamp( p.cascadeCover, 0.0f, 1.0f );
        gu->cloudCascade[ 3 ]  = p.cascadeSpeed;
    }

    gu->antiFireflyEnabled = devmode ? devmode->antiFirefly : true;

    {
        const auto& illum = pnext::get< RgDrawFrameIlluminationParams >( drawInfo );

        gu->rrDisoccEnable   = !!illum.enableRrDisocclusionMask;
        gu->rrDisoccRatio    = std::max( illum.rrDisocclusionThreshold, 1.0f );
        gu->rrDisoccMinDelta = std::max( illum.rrDisocclusionMinDelta, 0.0f );
        gu->rrDisoccShowMask = !!illum.rrDisocclusionShowMask;

        gu->rrFireflyThreshold = std::max( illum.rrFireflyThreshold, 0.0f );
        gu->rrFireflyMinLum    = std::max( illum.rrFireflyMinLum, 0.0f );
        gu->restirBlueNoise    = !!illum.restirBlueNoise;
        gu->shadowSamples      = std::clamp( illum.shadowSamples, 1u, 8u );
        // 1 = direct reservoir M, 2 = indirect reservoir M. Not a bool.
        gu->debugRestirM       = std::min( illum.debugRestirM, 2u );
        gu->debugVisibility    = std::min( illum.debugVisibility, 2u );

        // Doom64-RT: make debugVisibility 1 self-sufficient, because it was NOT, and that
        // silently invalidated two investigations.
        //
        // RtRaygenDirect.rgen writes the visibility term into the UNFILTERED DIRECT
        // buffer -- the direct DIFFUSE channel and nothing else. The normal composition
        // then adds indirect, emission and specular on top and modulates by albedo, so in
        // any room whose light is mostly emissive GI the debug view comes out looking like
        // the ordinary image with a few lights dimmed. It cannot show a shadow it did find,
        // let alone prove one absent.
        //
        // That is exactly the room this was being used in: under a Doom 64 lamp pane the
        // painted glow carries ~84% of the floor's light (rt_ceiling_bulb_emis, measured on
        // MAP94), all of it shadowless. "rt_debug_visibility says nothing casts a shadow
        // from the bulb bands" was read off this view on 2026-08-08 and treated as fact for
        // six days; it was never evidence either way.
        //
        // The fix is not a new view -- DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE already shows that
        // buffer raw, bypassing the denoiser, and the cvar descriptions already tell you to
        // pair the two by hand in the Dev window. An instrument whose answer depends on
        // remembering to tick a box elsewhere is a trap, so mode 1 now turns it on itself.
        // Mode 2 deliberately does not: it is *meant* to be read against normal shading, to
        // locate an umbra against the geometry casting it.
        if( gu->debugVisibility == 1 )
        {
            gu->debugShowFlags |= DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE;
        }
        gu->debugShowFlags |= illum.debugShowFlags;
        gu->restirTemporalJitter = std::clamp( illum.restirTemporalJitter, 0.0f, 8.0f );
        gu->rrSpecHitDist      = !!illum.rrSpecularHitDistance;

        // Doom64-RT: DLSS-RR pre-exposure reorder. HOST-GATED to frames where
        // the RR branch will actually run -- the exact condition guarding
        // nvDlssRr->Apply. This gate is load-bearing: if the flag were set on a
        // frame that silently fell back to DLSS2/FSR2 (null RR object, RR
        // rejected), CmPrepareFinal would skip the EV100 multiply and NOTHING
        // downstream would apply it -- a zero-exposure frame. The call site of
        // the post pass reads this same uniform, so shader and pass can never
        // disagree (the volumePostComp pattern).
        gu->rrPreExposure = ( illum.rrPreExposure && //
                              renderResolution.IsNvDlssRayReconstructionEnabled() &&
                              nvDlssRr != nullptr )
                                ? 1u
                                : 0u;
#ifdef __APPLE__
        // the MetalFX denoiser (DLSSRR_Metal.mm) gets the exposed colour, as the upscaler does
        gu->rrPreExposure = 0u;
#endif
        gu->rrPreExpDebug = !!illum.rrPreExposureDebug;

        // NRD lane: validation overlay switch. (nrdReserved0 became
        // rrGlowScale, assigned with rrGlowPre above; nrdReserved1/2 became
        // the demodulation pair below.)
        gu->nrdValidation = !!illum.nrdValidation;
        gu->rrDemod       = !!illum.rrDemod;
        gu->rrDemodFilter = std::min( illum.rrDemodFilter, 2u );
#ifdef __APPLE__
        // the MetalFX denoiser: with demodulation it denoises the lighting only, and the
        // albedo / texel detail is multiplied back afterwards, at output resolution, by
        // DLSSRR_Metal.mm (not by the post-RR exposure pass: rrPreExposure is 0 above);
        // 2 marks this for CmPrepareFinal. Only on the frames the denoiser runs.
        gu->rrDemod = ( illum.rrDemod && renderResolution.IsNvDlssRayReconstructionEnabled() && nvDlssRr )
                          ? 2u
                          : 0u;
#endif
        gu->svgfFp        = std::min( illum.svgfFp, 2u );
        gu->svgfFpGrad    = !!illum.svgfFpGrad;
        gu->svgfIndirMaxHist = std::clamp( illum.svgfIndirMaxHist, 0.0f, 256.0f );
        gu->svgfIndirAntilag = !!illum.svgfIndirAntilag;

        gu->directSamples         = std::clamp( illum.directSamples, 1u, 8u );
        gu->indirectSamples       = std::clamp( illum.indirectSamples, 1u, 8u );
        // 32 -> 64 (2026-08-17): pure loop bound in calcInitialReservoir, no
        // rays traced, no fixed-size arrays -- and scenes dense with small
        // emitters (the stripe-bulb rooms) genuinely need more candidates to
        // sample their light set without per-tile luminance noise.
        gu->restirInitialSamples  = std::clamp( illum.restirInitialSamples, 1u, 64u );
        gu->restirSpatialSamples  = std::clamp( illum.restirSpatialSamples, 0u, 16u );
        gu->restirSpatialRadius   = std::clamp( illum.restirSpatialRadius, 1.0f, 64.0f );
        gu->restirTemporalMCap    = std::clamp( illum.restirTemporalMCap, 1u, 64u );
        gu->rrGuideMin            = std::clamp( illum.rrGuideMin, 0.0f, 1.0f );
        gu->rrGuideMode           = std::clamp( illum.rrGuideMode, 0u, 2u );
        // The indirect antilag gate reads framebufDISGradientHistory, written
        // ONLY inside Denoiser::Denoise() (A-SVGF). On RR and NRD frames that
        // pass never runs, so the gate samples an UNINITIALIZED or stale
        // buffer -- and any garbage alpha > 0.25 rejects the indirect
        // temporal tap, silently pinning GI at raw 1 spp with no ReSTIR
        // accumulation. That was the long-suspected "either a dead no-op or
        // it rejects GI reuse every frame" (GenerateShaderCommon.py), settled
        // 2026-08-17 by the user's report of indirect-only noise under RR.
        // HOST-GATED to frames where A-SVGF actually runs. (If an NRD
        // bring-up fails and the frame falls back to A-SVGF, the gate is off
        // for that session -- antilag off means slight GI ghosting risk,
        // strictly better than the alternative.)
        {
            const bool asvgfRunsThisFrame =
                !( renderResolution.IsNvDlssRayReconstructionEnabled() && nvDlssRr ) &&
                !illum.nrdDenoiser;
            gu->restirIndirAntilag = !!illum.restirIndirAntilag && asvgfRunsThisFrame;
        }

        const bool fromGame = !!illum.enableRrTemporalPrefilter;
        if( devmode && devmode->rrTemporalPrefilterSticky )
        {
            gu->rrTemporalPrefilterEnabled = uint32_t( devmode->rrTemporalPrefilter );
        }
        else
        {
            gu->rrTemporalPrefilterEnabled = uint32_t( fromGame );
            if( devmode )
            {
                // Keep Dev checkbox visually synced with the active game value.
                devmode->rrTemporalPrefilter = fromGame;
            }
        }

        if( devmode && devmode->illumSensSticky )
        {
            gu->gradientMultDiffuse =
                std::clamp( devmode->illumSensDirect, 0.0f, 1.0f );
            gu->gradientMultIndirect =
                std::clamp( devmode->illumSensIndirect, 0.0f, 1.0f );
            gu->gradientMultSpecular =
                std::clamp( devmode->illumSensSpec, 0.0f, 1.0f );
        }
        else if( devmode )
        {
            devmode->illumSensDirect   = gu->gradientMultDiffuse;
            devmode->illumSensIndirect = gu->gradientMultIndirect;
            devmode->illumSensSpec     = gu->gradientMultSpecular;
        }
    }

    if( swapchain->IsHDREnabled() )
    {
        gu->hdrDisplay = swapchain->IsST2084ColorSpace() ? HDR_DISPLAY_ST2084 : HDR_DISPLAY_LINEAR;
    }
    else
    {
        gu->hdrDisplay = HDR_DISPLAY_NONE;
    }
}

auto RTGL1::VulkanDevice::Render( VkCommandBuffer& cmd, const RgDrawFrameInfo& drawInfo )
    -> FramebufferImageIndex
{
    // end of "Prepare for frame" label
    EndCmdLabel( cmd );


    const uint32_t frameIndex = currentFrameState.GetFrameIndex();
    const double   timeDelta  = std::max< double >( currentFrameTime - previousFrameTime, 0.0001 );
    const bool     resetHistory = drawInfo.resetHistory;


    const auto& cameraInfo = scene->GetCamera( renderResolution.Aspect() );

    bool mipLodBiasUpdated = worldSamplerManager->TryChangeMipLodBias(
        frameIndex,
        renderResolution.GetMipLodBias(
            pnext::get< RgDrawFrameTexturesParams >( drawInfo ).mipLodBiasOffset ) );
    const RgFloat2D jitter = { uniform->GetData()->jitterX, uniform->GetData()->jitterY };

    textureManager->SubmitDescriptors(
        frameIndex, pnext::get< RgDrawFrameTexturesParams >( drawInfo ), mipLodBiasUpdated );
    cubemapManager->SubmitDescriptors( frameIndex );

    lightManager->SubmitForFrame( cmd, frameIndex );

    uniform->Upload( cmd, frameIndex );

    // submit geometry and upload uniform after getting data from a scene
    scene->SubmitForFrame( cmd,
                           frameIndex,
                           uniform,
                           uniform->GetData()->rayCullMaskWorld,
                           drawInfo.disableRayTracedGeometry );

    if( drawInfo.presentPrevFrame )
    {
        return m_prevAccum;
    }

    if( auto w = pnext::get< RgDrawFramePostEffectsParams >( drawInfo ).pWipe )
    {
        effectWipe->CopyToWipeEffectSourceIfNeeded( cmd, //
                                                    frameIndex,
                                                    *framebuffers,
                                                    m_prevAccum,
                                                    renderResolution.GetResolutionState(),
                                                    w );
    }

    if( !drawInfo.disableRasterization )
    {
        rasterizer->SubmitForFrame( cmd, frameIndex );

        // draw rasterized sky to albedo before tracing primary rays
        if( uniform->GetData()->skyType == RG_SKY_TYPE_RASTERIZED_GEOMETRY )
        {
            // Doom64-RT: the volumetric cloud map, marched once, read by both
            // sky passes below.
            volumetric->ProcessClouds( cmd, frameIndex, *uniform, *blueNoise );

            rasterizer->DrawSkyToCubemap(
                cmd, frameIndex, *textureManager, *uniform, *tonemapping, *volumetric );
            rasterizer->DrawSkyToAlbedo(
                cmd,
                frameIndex,
                *textureManager,
                *uniform,
                *tonemapping,
                *volumetric,
                cameraInfo.view,
                pnext::get< RgDrawFrameSkyParams >( drawInfo ).skyViewerPosition,
                cameraInfo.projection,
                jitter,
                renderResolution );
        }

        if( fluid && !( devmode && devmode->fluidStopVisualize ) )
        {
            fluid->Visualize( cmd,
                              frameIndex,
                              cameraInfo.view,
                              cameraInfo.projection,
                              renderResolution,
                              cameraInfo.cameraNear,
                              cameraInfo.cameraFar );
        }
    }


    {
        lightGrid->Build( cmd, frameIndex, uniform, blueNoise, lightManager );

        portalList->SubmitForFrame( cmd, frameIndex );

        float volumetricMaxHistoryLen =
            resetHistory ? 0
                         : pnext::get< RgDrawFrameVolumetricParams >( drawInfo ).maxHistoryLength;

        const auto params = pathTracer->BindRayTracing( cmd,
                                                        frameIndex,
                                                        renderResolution.Width(),
                                                        renderResolution.Height(),
                                                        *scene,
                                                        *uniform,
                                                        *textureManager,
                                                        framebuffers,
                                                        restirBuffers,
                                                        *blueNoise,
                                                        *lightManager,
                                                        *cubemapManager,
                                                        *rasterizer->GetRenderCubemap(),
                                                        *portalList,
                                                        *volumetric );

        pathTracer->TracePrimaryRays( params );

        // draw decals on top of primary surface
        rasterizer->DrawDecals( cmd,
                                frameIndex,
                                *uniform,
                                *textureManager,
                                cameraInfo.view,
                                cameraInfo.projection,
                                jitter,
                                renderResolution );

        if( uniform->GetData()->reflectRefractMaxDepth > 0 )
        {
            pathTracer->TraceReflectionRefractionRays( params );
        }

        lightManager->BarrierLightGrid( cmd, frameIndex );
        pathTracer->CalculateInitialReservoirs( params );
        pathTracer->TraceDirectllumination( params );
        pathTracer->TraceIndirectllumination( params );
        pathTracer->TraceVolumetric( params );

        if( fluid )
        {
            fluid->Simulate( cmd,
                             frameIndex,
                             scene->GetASManager()->GetTLASDescSet( frameIndex ),
                             float( timeDelta ),
                             fluidGravity );
        }

        pathTracer->CalculateGradientsSamples( params );
        pathTracer->FinalizeIndirectIllumination_Compute( cmd,
                                                          frameIndex,
                                                          renderResolution.Width(),
                                                          renderResolution.Height(),
                                                          *scene,
                                                          *uniform,
                                                          *textureManager,
                                                          *framebuffers,
                                                          *restirBuffers,
                                                          *blueNoise,
                                                          *lightManager,
                                                          *cubemapManager,
                                                          *rasterizer->GetRenderCubemap(),
                                                          *portalList,
                                                          *volumetric );
        // Ground truth for "is DLSS-RR actually running?". Every input to this
        // decision fails silently: a null nvDlssRr (NGX create failed, or the
        // whole DLSSRR.cpp compiled to its stub), a non-DLSS upscaler, or a Dev
        // override all just quietly select A-SVGF. Nothing downstream reports
        // which path ran, so a wrong assumption here is unfalsifiable from the
        // game side -- which is exactly what happened for several sessions.
        // Edge-triggered, WARNING severity so it is visible without -rtdebug.
        {
            const bool haveRrObject = ( nvDlssRr != nullptr );
            const bool dlssOn       = renderResolution.IsNvDlssEnabled();
            const bool rrEnabled    = renderResolution.IsNvDlssRayReconstructionEnabled();
            const bool rrActive     = rrEnabled && haveRrObject;
            // Doom64-RT: reordered exposure is part of the path identity -- an
            // A/B of rt_rr_preexposure must be able to read its arm back here.
            const bool rrPreExp     = rrActive && uniform->GetData()->rrPreExposure != 0;

            static bool s_have = false;
            static bool s_prev = false;
            static bool s_prevPreExp = false;
            if( !s_have || s_prev != rrActive || s_prevPreExp != rrPreExp )
            {
                s_have = true;
                s_prev = rrActive;
                s_prevPreExp = rrPreExp;
                debug::Warning( "Denoiser path: {} (DLSS-RR object={}, DLSS upscaler={}, "
                                "RR flag={}, preExposure={})",
                                rrActive ? "DLSS-RR (ComposeNoisy -> nvDlssRr->Apply)"
                                         : "A-SVGF (Denoise)",
                                haveRrObject ? "present" : "NULL",
                                dlssOn ? "on" : "off",
                                rrEnabled ? "on" : "off",
                                rrPreExp ? "on (post-RR pass)" : "off (baked in CmPrepareFinal)" );
            }

            // ReSTIR feeds BOTH denoisers, so report it regardless of which one
            // runs -- the launcher now defaults to A-SVGF, and gating this on RR
            // would leave the default path with no way to verify its own uniforms.
            {
                static bool     s_rHave = false;
                static uint32_t s_rPrev[ 4 ] = {};
                const uint32_t  init = uniform->GetData()->restirInitialSamples;
                const uint32_t  spat = uniform->GetData()->restirSpatialSamples;
                // GI depth and the shadow depth it needs are part of the
                // trigger: an arm that moves only rt_gi_bounces must still
                // get its one printed proof that the value reached the shader.
                const uint32_t  gib  = uniform->GetData()->indirectBounces;
                const uint32_t  shd  = uniform->GetData()->maxBounceShadowsLights;
                if( !s_rHave || s_rPrev[ 0 ] != init || s_rPrev[ 1 ] != spat ||
                    s_rPrev[ 2 ] != gib || s_rPrev[ 3 ] != shd )
                {
                    s_rHave      = true;
                    s_rPrev[ 0 ] = init;
                    s_rPrev[ 1 ] = spat;
                    s_rPrev[ 2 ] = gib;
                    s_rPrev[ 3 ] = shd;
                    debug::Warning( "ReSTIR: initialSamples={} (stock 8), "
                                    "spatialSamples={} (stock 8), spatialRadius={} "
                                    "(stock 30), temporalMCap={} (stock 20), "
                                    "temporalJitter={} (stock 2), shadowSamples={} "
                                    "(stock 1), sppDirect={}, sppIndirect={}, "
                                    "giBounces={} (stock 2), giLegacyWeight={} (stock 1), "
                                    "maxBounceShadows={} (vertex i lit iff i < this)",
                                    init,
                                    spat,
                                    uniform->GetData()->restirSpatialRadius,
                                    uniform->GetData()->restirTemporalMCap,
                                    uniform->GetData()->restirTemporalJitter,
                                    uniform->GetData()->shadowSamples,
                                    uniform->GetData()->directSamples,
                                    uniform->GetData()->indirectSamples,
                                    gib,
                                    uniform->GetData()->indirectLegacyWeight,
                                    shd );
                }
            }

            // Report which OPTIONAL RR guides are actually bound. rt_rr_disocc 0
            // used to leave pInDisocclusionMask bound (writing zeros) instead of
            // passing nullptr, so the "off" arm never tested the configuration it
            // claimed to. An unreported binding is one nobody can falsify.
            if( rrActive )
            {
                const uint32_t disocc  = uniform->GetData()->rrDisoccEnable;
                const uint32_t spechit = uniform->GetData()->rrSpecHitDist;
                const uint32_t guide   = uniform->GetData()->rrGuideMode;

                const auto& illumRr = pnext::get< RgDrawFrameIlluminationParams >( drawInfo );
                const uint32_t exptex =
                    ( uniform->GetData()->rrPreExposure != 0 && illumRr.rrExposureTexture ) ? 1u
                                                                                            : 0u;
                const uint32_t translayer =
                    ( illumRr.rrTransparencyLayer && !drawInfo.disableRasterization ) ? 1u : 0u;

                static bool     s_gHave = false;
                static uint32_t s_gPrev[ 5 ] = {};
                if( !s_gHave || s_gPrev[ 0 ] != disocc || s_gPrev[ 1 ] != spechit ||
                    s_gPrev[ 2 ] != guide || s_gPrev[ 3 ] != exptex || s_gPrev[ 4 ] != translayer )
                {
                    s_gHave    = true;
                    s_gPrev[ 0 ] = disocc;
                    s_gPrev[ 1 ] = spechit;
                    s_gPrev[ 2 ] = guide;
                    s_gPrev[ 3 ] = exptex;
                    s_gPrev[ 4 ] = translayer;
                    debug::Warning( "RR guides: pInDisocclusionMask={}, "
                                    "pInSpecularHitDistance={}, albedo guide mode={}, "
                                    "pInExposureTexture={}, pInTransparencyLayer={}",
                                    disocc ? "BOUND" : "nullptr",
                                    spechit ? "BOUND" : "nullptr",
                                    guide,
                                    exptex ? "BOUND (1x1, CmPrepareFinal)" : "nullptr",
                                    translayer ? "BOUND (raster redirected)" : "nullptr" );

                    // b031a21 replaced the hardcoded TEMPORAL_RADIUS 2 with this
                    // uniform, and bisect blames that commit for the RR worm
                    // regression even though 2.0 should reproduce the constant
                    // exactly. Report the value actually reaching the shader:
                    // either the cvar is not 2, or the uniform is not landing
                    // where the shader reads it. Print the neighbouring members
                    // too -- if those are also wrong, it is the std140 layout.
                    debug::Warning( "ReSTIR uniforms: temporalJitter={} (stock 2), "
                                    "shadowSamples={} (stock 1), debugRestirM={}, "
                                    "spatialSamples={} (stock 8), spatialRadius={} "
                                    "(stock 30), temporalMCap={} (stock 20), "
                                    "initialSamples={} (stock 8), indirAntilagGate={}",
                                    uniform->GetData()->restirTemporalJitter,
                                    uniform->GetData()->shadowSamples,
                                    uniform->GetData()->debugRestirM,
                                    uniform->GetData()->restirSpatialSamples,
                                    uniform->GetData()->restirSpatialRadius,
                                    uniform->GetData()->restirTemporalMCap,
                                    uniform->GetData()->restirInitialSamples,
                                    uniform->GetData()->restirIndirAntilag
                                        ? "on (reads a buffer RR never writes)"
                                        : "off" );
                }
            }
        }

        // Doom64-RT: the NRD lane, stage 2 of docs/plan-nrd-denoiser.md. With
        // rt_nrd set (and RR inactive -- precedence RR > NRD > A-SVGF), ReLAX
        // replaces A-SVGF for the frame: CmNrdPack stages the same raw
        // unfiltered signals into NRD's input layouts, NRDIntegration records
        // ReLAX's dispatches, CmNrdCompose remodulates the denoised lighting
        // into PreFinal with CmNoisyCompose's exact arithmetic. EVERYTHING
        // downstream is the untouched A-SVGF frame shape: exposure baked in
        // CmPrepareFinal, glow before the upscaler, DLSS-SR reconstructing --
        // the shape the user reports as stable. Any failure at any step falls
        // back to A-SVGF for the frame, loudly.
        bool nrdRanThisFrame = false;
        {
            const uint32_t nrdRequested =
                pnext::get< RgDrawFrameIlluminationParams >( drawInfo ).nrdDenoiser;
            const bool rrBlocks =
                renderResolution.IsNvDlssRayReconstructionEnabled() && nvDlssRr;
            const bool wantNrd = nrdRequested != 0 && !rrBlocks;

            // Edge-triggered arrival print -- "no NRD line at all" must be
            // distinguishable from "the request never reached RTGL" without a
            // debugger (2026-08-17: rt_nrd 1 produced total silence, and this
            // is how it was bisected).
            {
                static bool     s_nHave = false;
                static uint32_t s_nPrev[ 2 ] = {};
                if( !s_nHave || s_nPrev[ 0 ] != nrdRequested || s_nPrev[ 1 ] != uint32_t( rrBlocks ) )
                {
                    s_nHave      = true;
                    s_nPrev[ 0 ] = nrdRequested;
                    s_nPrev[ 1 ] = uint32_t( rrBlocks );
                    debug::Warning( "NRD request: illum.nrdDenoiser={}, blockedByRR={}",
                                    nrdRequested,
                                    rrBlocks ? "yes" : "no" );
                }
            }

            if( wantNrd )
            {
                if( !nrdDenoiser )
                {
                    nrdDenoiser = std::make_shared< NrdDenoiser >( instance,
                                                                   physDevice->Get(),
                                                                   device,
                                                                   vkEnabledInstanceExtensions,
                                                                   vkEnabledDeviceExtensions,
                                                                   queues->GetIndexGraphics(),
                                                                   1 );
                }

                if( nrdDenoiser->EnsureReady( renderResolution.Width(),
                                              renderResolution.Height() ) )
                {
                    denoiser->PackForNrd( cmd, frameIndex, uniform );

                    const auto& illumNrd =
                        pnext::get< RgDrawFrameIlluminationParams >( drawInfo );
                    const auto tuning = NrdDenoiser::Tuning{
                        .maxAccumFrames   = illumNrd.nrdMaxAccumFrames,
                        .fastAccumFrames  = illumNrd.nrdFastAccumFrames,
                        .atrousIterations = illumNrd.nrdAtrousIterations,
                        .prepassDiffuse   = illumNrd.nrdPrepassDiffuse,
                        .prepassSpecular  = illumNrd.nrdPrepassSpecular,
                        .phiLuminance     = illumNrd.nrdPhiLuminance,
                        .minHitDistWeight = illumNrd.nrdMinHitDistWeight,
                        .antiFirefly      = !!illumNrd.nrdAntiFirefly,
                    };

                    nrdRanThisFrame =
                        nrdDenoiser->Denoise( cmd,
                                              frameIndex,
                                              *framebuffers,
                                              uniform->GetData(),
                                              timeDelta,
                                              resetHistory,
                                              uniform->GetData()->nrdValidation != 0,
                                              tuning );

                    if( nrdRanThisFrame )
                    {
                        denoiser->ComposeAfterNrd( cmd, frameIndex, uniform );
                    }
                }
            }
        }

        if( renderResolution.IsNvDlssRayReconstructionEnabled() && nvDlssRr )
        {
            // Do NOT call AccumulateForRR here — feeding A-SVGF temporal into RR
            // produced a faded duplicate/ghost depth view (2026-08-05). Keep raw
            // ComposeNoisy → DLSS-RR. Soft analytic-light fades instead.
            denoiser->ComposeNoisy( cmd, frameIndex, uniform );
        }
        else if( !nrdRanThisFrame )
        {
            denoiser->Denoise( cmd, frameIndex, uniform );
        }
        volumetric->ProcessScattering(
            cmd, frameIndex, *uniform, *blueNoise, *framebuffers, volumetricMaxHistoryLen );
        tonemapping->CalculateExposure( cmd, frameIndex, uniform );
    }

    imageComposition->PrepareForRaster( cmd, frameIndex, uniform.get() );
    volumetric->BarrierToReadIllumination( cmd, frameIndex );

    // Doom64-RT: with DLSS-RR active, rasterized content (every translucent
    // sprite in the game, particles, lens flares) is redirected into the NGX
    // transparency layer instead of the final image. Baked into FINAL it
    // becomes part of RR's colour input while every guide (albedo, normal,
    // depth, MV) describes the opaque surface BEHIND it -- content the network
    // is explicitly told is not there, i.e. noise to remove. The layer is
    // NGX-composited after denoise+upscale, which is NVIDIA's sanctioned
    // route for exactly this content. Gated per-frame on the same condition
    // the RR branch below uses, so the layer can never be bound on a frame
    // whose raster did not fill it (disableRasterization included).
    const bool rrTransLayer = renderResolution.IsNvDlssRayReconstructionEnabled() &&
                              nvDlssRr != nullptr &&
                              !!pnext::get< RgDrawFrameIlluminationParams >( drawInfo )
                                    .rrTransparencyLayer &&
                              !drawInfo.disableRasterization;

    if( !drawInfo.disableRasterization )
    {
        // draw rasterized geometry into the final image (or the RR layer)
        auto label = CmdLabel{ cmd, "Rasterized geometry" };
        rasterizer->DrawToFinalImage( cmd,
                                      frameIndex,
                                      *textureManager,
                                      *uniform,
                                      *tonemapping,
                                      *volumetric,
                                      cameraInfo.view,
                                      cameraInfo.projection,
                                      jitter,
                                      renderResolution,
                                      lightmapScreenCoverage,
                                      rrTransLayer );
    }

    imageComposition->Finalize( cmd,
                                frameIndex,
                                *uniform,
                                *tonemapping,
                                pnext::get< RgDrawFrameTonemappingParams >( drawInfo ) );

    FramebufferImageIndex accum       = FB_IMAGE_INDEX_FINAL;
    bool                  needHudOnly = false;
    {
        auto l_todx12 = [ this, frameIndex ]( VkCommandBuffer vkcmd,
                                              auto& technique ) -> ID3D12GraphicsCommandList* {
            if( !dxgi::HasDX12Instance() )
            {
                return nullptr;
            }
#ifdef RG_USE_DX12
            technique.CopyVkInputsToDX12( vkcmd, //
                                          frameIndex,
                                          *framebuffers,
                                          renderResolution.GetResolutionState() );

            const VkSemaphore initFrameFinished = currentFrameState.GetSemaphoreForWaitAndRemove();

            auto vktodx12 = Semaphores_GetVkDx12Shared( dxgi::SHARED_SEM_FSR3_IN );
            if( !vktodx12 )
            {
                return nullptr;
            }

            cmdManager->Submit_Timeline( //
                vkcmd,
                nullptr,
                ToWait{ initFrameFinished, SEMAPHORE_IS_BINARY },
                ToSignal{ vktodx12->vksemaphore, timelineFrame } );

            ID3D12CommandQueue* dx12queue = dxgi::GetD3D12CommandQueue();
            if( !dx12queue )
            {
                return nullptr;
            }
            HRESULT hr = dx12queue->Wait( vktodx12->d3d12fence, timelineFrame );
            assert( SUCCEEDED( hr ) );

            return dxgi::CreateD3D12CommandList( frameIndex );
#else
            return nullptr;
#endif
        };
        auto l_tovk = [ this, frameIndex ]( ID3D12GraphicsCommandList* dx12cmd,
                                            auto& technique ) -> VkCommandBuffer {
            if( !dx12cmd || !dxgi::HasDX12Instance() )
            {
                currentFrameState.SetSemaphore( nullptr );
                return cmdManager->StartGraphicsCmd();
            }
#ifdef RG_USE_DX12
            HRESULT hr = dx12cmd->Close();
            assert( SUCCEEDED( hr ) );

            auto dx12tovk = Semaphores_GetVkDx12Shared( dxgi::SHARED_SEM_FSR3_OUT );
            if( !dx12tovk )
            {
                return nullptr;
            }

            ID3D12CommandQueue* dx12queue = dxgi::GetD3D12CommandQueue();
            if( !dx12queue )
            {
                return nullptr;
            }
            ID3D12CommandList* p = dx12cmd;
            dx12queue->ExecuteCommandLists( 1, &p );
            hr = dx12queue->Signal( dx12tovk->d3d12fence, timelineFrame );

            // next cmd should wait for DX12
            currentFrameState.SetSemaphore( SUCCEEDED( hr ) ? dx12tovk->vksemaphore : nullptr );
            auto vkcmd = cmdManager->StartGraphicsCmd();

            technique.CopyDX12OutputToVk( vkcmd, //
                                          frameIndex,
                                          *framebuffers,
                                          renderResolution.GetResolutionState() );
            return vkcmd;
#else
            return VK_NULL_HANDLE;
#endif
        };

        // upscale finalized image
        // (Ray Reconstruction first: on macOS it's the MetalFX denoiser, with the FSR2 / MetalFX slot)
        if( renderResolution.IsNvDlssRayReconstructionEnabled() && nvDlssRr )
        {
            {
                // Exposure texture only with the pre-exposure reorder: against
                // an already-exposed input it would declare the scale twice.
                const bool rrExpTex =
                    uniform->GetData()->rrPreExposure != 0 &&
                    !!pnext::get< RgDrawFrameIlluminationParams >( drawInfo ).rrExposureTexture;

#ifdef __APPLE__
                nvDlssRr->SetDemodulation( uniform->GetData()->rrDemod != 0 );
#endif
                accum = nvDlssRr->Apply( cmd,
                                         frameIndex,
                                         *framebuffers,
                                         renderResolution,
                                         jitter,
                                         timeDelta,
                                         resetHistory,
                                         uniform->GetData()->rrSpecHitDist != 0,
                                         uniform->GetData()->rrDisoccEnable != 0,
                                         rrExpTex,
                                         rrTransLayer,
                                         uniform->GetData()->view,
                                         uniform->GetData()->projection );
            }
        }
        else if( renderResolution.IsNvDlssEnabled() )
        {
            if( nvDlss3dx12 && swapchain->WithDLSS3FrameGeneration() )
            {
                ID3D12GraphicsCommandList* dx12cmd = l_todx12( cmd, *nvDlss3dx12 );

                if( auto u = nvDlss3dx12->Apply( dx12cmd,
                                                 frameIndex,
                                                 *framebuffers,
                                                 renderResolution,
                                                 jitter,
                                                 timeDelta,
                                                 resetHistory,
                                                 cameraInfo,
                                                 frameId,
                                                 m_skipGeneratedFrame ) )
                {
                    accum = *u;
                    needHudOnly = false; // providing FB_IMAGE_INDEX_HUD_ONLY to DLSS3 doesn't work
                }
                else
                {
                    swapchain->MarkAsFailed( SWAPCHAIN_TYPE_FRAME_GENERATION_DLSS3 );
                }

                cmd = l_tovk( dx12cmd, *nvDlss3dx12 );
            }
            else if( nvDlss2 )
            {
                accum = nvDlss2->Apply( cmd,
                                        frameIndex,
                                        *framebuffers,
                                        renderResolution,
                                        jitter,
                                        timeDelta,
                                        resetHistory,
                                        // Doom64-RT: bind the volumetric's
                                        // silhouette mask only when the feature
                                        // is on, so 0 is the untouched path.
                                        uniform->GetData()->volumeUpscaleBias > 0.0f );
            }
            else
            {
                assert( 0 );
            }
        }
        else if( renderResolution.IsAmdFsr2Enabled() )
        {
            if( amdFsr3dx12 && swapchain->WithFSR3FrameGeneration() )
            {
                ID3D12GraphicsCommandList* dx12cmd = l_todx12( cmd, *amdFsr3dx12 );

                if( auto u = amdFsr3dx12->Apply( dx12cmd,
                                                 frameIndex,
                                                 *framebuffers,
                                                 renderResolution,
                                                 jitter,
                                                 timeDelta,
                                                 cameraInfo.cameraNear,
                                                 cameraInfo.cameraFar,
                                                 cameraInfo.fovYRadians,
                                                 resetHistory,
                                                 sceneImportExport->GetWorldScale(),
                                                 m_skipGeneratedFrame ) )
                {
                    accum = *u;
                    needHudOnly = true;
                }
                else
                {
                    swapchain->MarkAsFailed( SWAPCHAIN_TYPE_FRAME_GENERATION_FSR3 );
                }

                cmd = l_tovk( dx12cmd, *amdFsr3dx12 );
            }
            else if( amdFsr2 )
            {
                auto label = CmdLabel{ cmd, "Upscale (FSR2 / MetalFX)" };
                accum = amdFsr2->Apply( cmd,
                                        frameIndex,
                                        *framebuffers,
                                        renderResolution,
                                        jitter,
                                        timeDelta,
                                        cameraInfo.cameraNear,
                                        cameraInfo.cameraFar,
                                        cameraInfo.fovYRadians,
                                        resetHistory,
                                        sceneImportExport->GetWorldScale() );
            }
            else
            {
                assert( 0 );
            }
        }

        // Doom64-RT: THE EXPOSURE GOES BACK ON HERE under the DLSS-RR
        // pre-exposure reorder. CmPrepareFinal skipped the EV100 multiply and
        // the screen-emissive add so RR could denoise linear pre-exposure
        // radiance (its guide declares exposure unsupported, S3.7); this pass
        // reapplies both on RR's output, at output resolution, BEFORE
        // DrawClassic / BlitForEffects / sharpening / bloom -- everything after
        // this line still sees a normally-exposed image, as it always has.
        //
        // Gated on the UNIFORM, not the cvar chain: it is the exact value
        // CmPrepareFinal read this frame, so skip-side and apply-side cannot
        // disagree. gu->rrPreExposure is host-gated to the RR branch above, so
        // accum is UPSCALED_PONG by construction here.
        if( uniform->GetData()->rrPreExposure != 0 )
        {
            assert( accum == FB_IMAGE_INDEX_UPSCALED_PONG );
            imageComposition->RrPostExposure( cmd,
                                              frameIndex,
                                              *uniform,
                                              *tonemapping,
                                              renderResolution.UpscaledWidth(),
                                              renderResolution.UpscaledHeight() );
        }

        // Doom64-RT: THE MEDIUM GOES ON HERE, after the upscaler and before
        // anything else touches the image.
        //
        // CmPrepareFinal left the surface alone when volumePostComp is set, so
        // the upscaler has just reconstructed an ordinary game image instead of
        // one with a stepped, jittering veil baked into it -- which is what it
        // was drawing dark lines along. See CmVolumeCompose.comp.
        //
        // The position is load-bearing in both directions. AFTER the upscaler,
        // obviously. But BEFORE DrawClassic below, because classic-shaded pixels
        // are painted over the top and are meant to skip volumetrics entirely --
        // exactly what CmPrepareFinal's own classicShading() guard does at
        // render resolution. And before BlitForEffects / sharpening / bloom, so
        // those still see a veiled image, as they always have.
        if( uniform->GetData()->volumePostComp != 0 )
        {
            const bool upscaled = ( accum == FB_IMAGE_INDEX_UPSCALED_PING ||
                                    accum == FB_IMAGE_INDEX_UPSCALED_PONG );

            imageComposition->ComposeVolume(
                cmd,
                frameIndex,
                *uniform,
                *tonemapping,
                accum,
                upscaled ? renderResolution.UpscaledWidth() : renderResolution.Width(),
                upscaled ? renderResolution.UpscaledHeight() : renderResolution.Height() );
        }

        if( lightmapScreenCoverage > 0 && !drawInfo.disableRasterization )
        {
            rasterizer->DrawClassic(
                cmd,
                frameIndex,
                accum,
                *textureManager,
                *uniform,
                *tonemapping,
                *volumetric,
                cameraInfo.view,
                cameraInfo.projection,
                renderResolution,
                lightmapScreenCoverage,
                pnext::get< RgDrawFrameSkyParams >( drawInfo ).skyViewerPosition );
        }

        accum = framebuffers->BlitForEffects( cmd,
                                              frameIndex,
                                              accum,
                                              renderResolution.GetBlitFilter(),
                                              m_pixelated ? &m_pixelated.value() : nullptr );
    }


    const auto args = CommonnlyUsedEffectArguments{
        .cmd          = cmd,
        .frameIndex   = frameIndex,
        .framebuffers = framebuffers,
        .uniform      = uniform,
        .width        = renderResolution.UpscaledWidth(),
        .height       = renderResolution.UpscaledHeight(),
        .currentTime  = float( currentFrameTime ),
    };

    {
        if( renderResolution.IsDedicatedSharpeningEnabled() )
        {
            accum = sharpening->Apply( cmd,
                                       frameIndex,
                                       framebuffers,
                                       renderResolution.UpscaledWidth(),
                                       renderResolution.UpscaledHeight(),
                                       accum,
                                       renderResolution.GetSharpeningTechnique(),
                                       renderResolution.GetSharpeningIntensity() );
        }

        if( pnext::get< RgDrawFrameBloomParams >( drawInfo ).bloomIntensity > 0.0f )
        {
            accum = bloom->Apply( cmd,
                                  frameIndex,
                                  *uniform,
                                  *tonemapping,
                                  *textureManager,
                                  renderResolution.UpscaledWidth(),
                                  renderResolution.UpscaledHeight(),
                                  accum );
        }

        auto l_applyIf = [ &args ]( auto&                 effect,
                                    auto&                 setupArg,
                                    FramebufferImageIndex input ) -> FramebufferImageIndex {
            return effect->Setup( args, setupArg ) ? effect->Apply( args, input ) : input;
        };

        const auto& postef = pnext::get< RgDrawFramePostEffectsParams >( drawInfo );

        accum = l_applyIf( effectTeleport, postef.pTeleport, accum );
        accum = l_applyIf( effectColorTint, postef.pColorTint, accum );
        accum = l_applyIf( effectInverseBW, postef.pInverseBlackAndWhite, accum );
        accum = l_applyIf( effectHueShift, postef.pHueShift, accum );
        accum = l_applyIf( effectNightVision, postef.pNightVision, accum );
        accum = l_applyIf( effectChromaticAberration, postef.pChromaticAberration, accum );
        accum = l_applyIf( effectDistortedSides, postef.pDistortedSides, accum );
        accum = l_applyIf( effectWaves, postef.pWaves, accum );
        accum = l_applyIf( effectRadialBlur, postef.pRadialBlur, accum );
        accum = l_applyIf( effectVHS, postef.pVHS, accum );
    }

#ifdef __APPLE__
    // macOS frame generation: MetalFX frame interpolation, with the MetalFX upscaler / denoiser;
    // it gets the image without HUD, and the HUD separately (FB_IMAGE_INDEX_HUD_ONLY)
    {
        const bool fiWanted = m_frameGenerationMode != RG_FRAME_GENERATION_MODE_OFF &&
                              renderResolution.IsAmdFsr2Enabled() && amdFsr2 &&
                              MetalFI::Prepare( device,
                                                renderResolution.Width(),
                                                renderResolution.Height(),
                                                renderResolution.UpscaledWidth(),
                                                renderResolution.UpscaledHeight() );
        if( !fiWanted && m_fiWasActive )
        {
            MetalFI::Release();
        }
        m_fiWasActive = fiWanted;
        m_fiThisFrame = fiWanted;
        if( fiWanted )
        {
            const bool hasHud = !drawInfo.disableRasterization && rasterizer->HasSwapchainGeometry();

            needHudOnly = true;
            m_fiParams  = MetalFI::Params{
                 .rendered         = VK_NULL_HANDLE, // at the end of Render
                 .width            = renderResolution.UpscaledWidth(),
                 .height           = renderResolution.UpscaledHeight(),
                 .hud              = hasHud ? framebuffers->GetImage( FB_IMAGE_INDEX_HUD_ONLY, frameIndex )
                                            : VK_NULL_HANDLE,
                 .depthNdc         = framebuffers->GetImage( FB_IMAGE_INDEX_DEPTH_NDC, frameIndex ),
                 .motion           = framebuffers->GetImage( FB_IMAGE_INDEX_MOTION_DLSS, frameIndex ),
                 .renderWidth      = renderResolution.Width(),
                 .renderHeight     = renderResolution.Height(),
                 .jitterX          = jitter.data[ 0 ],
                 .jitterY          = jitter.data[ 1 ],
                 .nearPlane        = cameraInfo.cameraNear,
                 .farPlane         = cameraInfo.cameraFar,
                 .fovYRadians      = cameraInfo.fovYRadians,
                 .deltaTime        = float( timeDelta ),
                 .reset            = resetHistory,
                 .presentGenerated = m_frameGenerationMode == RG_FRAME_GENERATION_MODE_ON,
            };
        }
    }
#endif

    // draw geometry such as HUD into an upscaled framebuf
    if( !drawInfo.disableRasterization )
    {
        auto label = CmdLabel{ cmd, "Rasterized HUD to swapchain" };
        if( !needHudOnly )
        {
            framebuffers->BarrierOne(
                cmd, frameIndex, accum, RTGL1::Framebuffers::BarrierType::Storage );

            rasterizer->DrawToSwapchain( cmd,
                                         frameIndex,
                                         accum,
                                         *textureManager,
                                         uniform->GetData()->view,
                                         uniform->GetData()->projection,
                                         renderResolution.UpscaledWidth(),
                                         renderResolution.UpscaledHeight(),
                                         swapchain->IsHDREnabled() );
        }
        else
        {
            rasterizer->DrawToSwapchain( cmd,
                                         frameIndex,
                                         FB_IMAGE_INDEX_HUD_ONLY,
                                         *textureManager,
                                         uniform->GetData()->view,
                                         uniform->GetData()->projection,
                                         renderResolution.UpscaledWidth(),
                                         renderResolution.UpscaledHeight(),
                                         swapchain->IsHDREnabled() );

#ifdef RG_USE_DX12
            FramebufferImageIndex todx12[] = { FB_IMAGE_INDEX_HUD_ONLY };
            Framebuf_CopyVkToDX12( cmd,
                                   frameIndex,
                                   *framebuffers,
                                   renderResolution.UpscaledWidth(),
                                   renderResolution.UpscaledHeight(),
                                   todx12 );
#endif
        }
    }

    // post-effect that work on swapchain geometry too
    {
        const auto& postef = pnext::get< RgDrawFramePostEffectsParams >( drawInfo );

        if( effectWipe->Setup( args, postef.pWipe, frameId ) )
        {
            accum = effectWipe->Apply( args, *blueNoise, accum );
        }

        if( effectDither->Setup( args, postef.pDither ) )
        {
            accum = effectDither->Apply( args, accum );
        }

        if( postef.pCRT != nullptr && postef.pCRT->isActive )
        {
            effectCrtDemodulateEncode->Setup( args );
            accum = effectCrtDemodulateEncode->Apply( args, accum );

            effectCrtDecode->Setup( args );
            accum = effectCrtDecode->Apply( args, accum );
        }
    }

    // convert scene HDR to a present HDR compatible space,
    // or apply a tonemapping to fit into LDR
    {
        const auto& tnmp = pnext::get< RgDrawFrameTonemappingParams >( drawInfo );

        VkDescriptorSet lpmDescSet =
            imageComposition->SetupLpmParams( cmd, frameIndex, tnmp, swapchain->IsHDREnabled() );
        effectHDRPrepare->Setup( args, tnmp );

        VkDescriptorSet descSets[] = {
            args.framebuffers->GetDescSet( args.frameIndex ),
            args.uniform->GetDescSet( args.frameIndex ),
            lpmDescSet,
        };
        accum = effectHDRPrepare->Apply( descSets, args, accum );
    }

#ifdef __APPLE__
    if( m_fiThisFrame )
    {
        framebuffers->BarrierOne( cmd, frameIndex, accum );
        m_fiParams.rendered = framebuffers->GetImage( accum, frameIndex );
    }
#endif

    m_prevAccum = accum;
    return accum;
}

#if 0
namespace
{
void WaitTimelineAndSignalBinary( VkQueue     q,
                                  VkSemaphore towait_timeline,
                                  uint64_t    towait_value,
                                  VkSemaphore tosignal_binary )
{
    VkPipelineStageFlags s = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    auto timeline = VkTimelineSemaphoreSubmitInfo{
        .sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext                     = nullptr,
        .waitSemaphoreValueCount   = 1,
        .pWaitSemaphoreValues      = &towait_value,
        .signalSemaphoreValueCount = 0,
        .pSignalSemaphoreValues    = nullptr,
    };

    auto info = VkSubmitInfo{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = &timeline,
        .waitSemaphoreCount   = 1,
        .pWaitSemaphores      = &towait_timeline,
        .pWaitDstStageMask    = &s,
        .commandBufferCount   = 0,
        .pCommandBuffers      = nullptr,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = &tosignal_binary,
    };

    VkResult r = vkQueueSubmit( q, 1, &info, VK_NULL_HANDLE );
    RTGL1::VK_CHECKERROR( r );
}
}
#endif

void RTGL1::VulkanDevice::EndFrame( VkCommandBuffer cmd, FramebufferImageIndex rendered )
{
    auto label = CmdLabel{ cmd, "Blit to swapchain" };

    const uint32_t    frameIndex        = currentFrameState.GetFrameIndex();
    const VkSemaphore initFrameFinished = currentFrameState.GetSemaphoreForWaitAndRemove();

    // present debug window
    if( debugWindows && !debugWindows->IsMinimized() )
    {
        VkCommandBuffer debugCmd = cmdManager->StartGraphicsCmd();
        debugWindows->SubmitForFrame( debugCmd, frameIndex );

        VkSemaphore towait[] = {
            debugWindows->GetSwapchainImageAvailableSemaphore_Binary( frameIndex ),
        };
        cmdManager->Submit_Binary( //
            debugCmd,
            towait,
            debugFinishedSemaphores[ frameIndex ], // signal
            VK_NULL_HANDLE );

        VkResult       r       = VK_SUCCESS;
        VkSwapchainKHR sw      = debugWindows->GetSwapchainHandle();
        uint32_t       swIndex = debugWindows->GetSwapchainCurrentImageIndex();

        auto presentInfo = VkPresentInfoKHR{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &debugFinishedSemaphores[ frameIndex ],
            .swapchainCount     = 1,
            .pSwapchains        = &sw,
            .pImageIndices      = &swIndex,
            .pResults           = &r,
        };
        vkQueuePresentKHR( queues->GetGraphics(), &presentInfo );
        debugWindows->OnQueuePresent( r );
    }

    if( nvDlss3dx12 )
    {
        nvDlss3dx12->Reflex_RenderEnd();
        nvDlss3dx12->Reflex_PresentStart();
    }


    const auto rendered_size =
        framebuffers->GetFramebufSize( renderResolution.GetResolutionState(), rendered );


    // present
    if( swapchain->WithDXGI() )
    {
        [ & ] {
#ifdef RG_USE_DX12
            ID3D12CommandQueue* dx12queue = dxgi::GetD3D12CommandQueue();
            if( !dx12queue )
            {
                return;
            }

            // copy vk to dx12 buffer
            {
                FramebufferImageIndex fs[] = {
                    rendered,
                };
                Framebuf_CopyVkToDX12( cmd, //
                                       frameIndex,
                                       *framebuffers,
                                       rendered_size.width,
                                       rendered_size.height,
                                       fs );
            }
            // submit vk, and wait for vk in dx12
            {
                auto renderFin = Semaphores_GetVkDx12Shared( dxgi::SHARED_SEM_RENDER_FINISHED );
                if( !renderFin )
                {
                    debug::Warning( "Skipping DXGI present, as Semaphores_GetVkDx12Shared failed" );
                    return;
                }

                cmdManager->Submit_Timeline( //
                    cmd,
                    frameFences[ frameIndex ],
                    ToWait{ initFrameFinished, timelineFrame },
                    ToSignal{ renderFin->vksemaphore, timelineFrame } );

                HRESULT hr = dx12queue->Wait( renderFin->d3d12fence, timelineFrame );
                assert( SUCCEEDED( hr ) );
            }

            ID3D12GraphicsCommandList* dx12cmd = dxgi::CreateD3D12CommandList( frameIndex );
            // blit to the swapchain's shadow buffer (copysrc)
            {
                uint32_t dst_w      = 0;
                uint32_t dst_h      = 0;
                bool     dst_tosrgb = false;

                ID3D12Resource* src = dxgi::Framebuf_GetVkDx12Shared( rendered ).d3d12resource;
                ID3D12Resource* dst = dxgi::GetSwapchainCopySrc( &dst_w, &dst_h, &dst_tosrgb );

                dxgi::DispatchBlit( dx12cmd, src, dst, dst_w, dst_h, dst_tosrgb );
            }
            // copy from the shadow buffer to the actual swapchain image
            {
                ID3D12Resource* src = dxgi::GetSwapchainCopySrc();
                ID3D12Resource* dst = dxgi::GetSwapchainBack( swapchain->GetCurrentImageIndex() );

                dx12cmd->CopyResource( dst, src );
                {
                    D3D12_RESOURCE_BARRIER bs[] = {
                        CD3DX12_RESOURCE_BARRIER::Transition( dst, //
                                                              D3D12_RESOURCE_STATE_COPY_DEST,
                                                              D3D12_RESOURCE_STATE_PRESENT ),
                    };
                    dx12cmd->ResourceBarrier( std::size( bs ), bs );
                }
            }
            HRESULT hr = dx12cmd->Close();
            assert( SUCCEEDED( hr ) );

            // submit dx12, wait for execution, and present
            {
                auto present = Semaphores_GetVkDx12Shared( dxgi::SHARED_SEM_PRESENT_COPY );
                if( !present )
                {
                    debug::Warning( "Skipping DXGI present, as Semaphores_GetVkDx12Shared failed" );
                    return;
                }

                ID3D12CommandList* p = dx12cmd;
                dx12queue->ExecuteCommandLists( 1, &p );
                hr = dx12queue->Signal( present->d3d12fence, timelineFrame );
                assert( SUCCEEDED( hr ) );

                dxgi::Present( present->d3d12fence, timelineFrame );
            }
#endif // RG_USE_DX12
        }();
    }
    else
    {
        // copy to swapchain's back buffer
        bool presentedByFrameInterpolation = false;
#ifdef __APPLE__
        // macOS frame generation: the actual frame (HUD on top) and the generated one are made
        // here, and presented by vkQueuePresentKHR (FrameInterp_Metal.mm)
        if( m_fiThisFrame && m_fiParams.rendered != VK_NULL_HANDLE )
        {
            presentedByFrameInterpolation = MetalFI::Encode( device, cmd, m_fiParams );
        }
        m_fiThisFrame = false;
#endif
        if( !presentedByFrameInterpolation )
        {
            framebuffers->BarrierOne( cmd, frameIndex, rendered );

            swapchain->BlitForPresent( cmd,
                                       framebuffers->GetImage( rendered, frameIndex ),
                                       rendered_size,
                                       VK_FILTER_NEAREST,
                                       VK_IMAGE_LAYOUT_GENERAL );
        }

        uint32_t    towait_count = 0;
        VkSemaphore towait[ 2 ]  = {};
        if( swapchain->Valid() )
        {
            towait[ towait_count++ ] = vkswapchainAvailableSemaphores[ frameIndex ];
        }
        if( initFrameFinished )
        {
            towait[ towait_count++ ] = initFrameFinished;
        }

        cmdManager->Submit_Binary( //
            cmd,
            std::span{ towait, towait_count },
            emulatedSemaphores[ frameIndex ], // signal
            frameFences[ frameIndex ] );

        if( swapchain->Valid() )
        {
            VkResult       r       = VK_SUCCESS;
            VkSwapchainKHR sw      = swapchain->GetHandle();
            uint32_t       swIndex = swapchain->GetCurrentImageIndex();

            // present to surfaces after finishing the rendering
            auto presentInfo = VkPresentInfoKHR{
                .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores    = &emulatedSemaphores[ frameIndex ],
                .swapchainCount     = 1,
                .pSwapchains        = &sw,
                .pImageIndices      = &swIndex,
                .pResults           = &r,
            };
            vkQueuePresentKHR( queues->GetGraphics(), &presentInfo );
            swapchain->OnQueuePresent( r );
        }
    }

    if( nvDlss3dx12 )
    {
        nvDlss3dx12->Reflex_PresentEnd();
    }

    frameId++;

    if( nvDlss3dx12 )
    {
        nvDlss3dx12->Reflex_SimStart( frameId );
    }
}



// Interface implementation



void RTGL1::VulkanDevice::StartFrame( const RgStartFrameInfo* pOriginalInfo )
{
    if( currentFrameState.WasFrameStarted() )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_ENDED );
    }

    if( pOriginalInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }

    if( pOriginalInfo->sType != RG_STRUCTURE_TYPE_START_FRAME_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }

    auto startFrame_Core = [ this ]( const RgStartFrameInfo& info ) {
        VkCommandBuffer newFrameCmd = BeginFrame( info );
        currentFrameState.OnBeginFrame( newFrameCmd );
    };

    auto startFrame_WithDevmode = [ this, startFrame_Core ]( const RgStartFrameInfo& original ) {
        auto modified            = RgStartFrameInfo{ original };
        auto modified_Resolution = pnext::get< RgStartFrameRenderResolutionParams >( original );
        auto modified_Fluid      = pnext::get< RgStartFrameFluidParams >( original );

        // clang-format off
        modified_Resolution .pNext = modified.pNext;
        modified_Fluid      .pNext = &modified_Resolution;
        modified            .pNext = &modified_Fluid;
        // clang-format on

        Dev_Override( modified, modified_Resolution, modified_Fluid );

        startFrame_Core( modified );
    };

    if( Dev_IsDevmodeInitialized() )
    {
        startFrame_WithDevmode( *pOriginalInfo );
    }
    else
    {
        startFrame_Core( *pOriginalInfo );
    }
}

void RTGL1::VulkanDevice::DrawFrame( const RgDrawFrameInfo* pOriginalInfo )
{
    if( !currentFrameState.WasFrameStarted() )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_STARTED );
    }
    if( pOriginalInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pOriginalInfo->sType != RG_STRUCTURE_TYPE_DRAW_FRAME_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }

    DrawEndUserWarnings();

    auto drawFrame_Core = [ this ]( const RgDrawFrameInfo& info ) {
        VkCommandBuffer cmd = currentFrameState.GetCmdBuffer();

        previousFrameTime = currentFrameTime;
        currentFrameTime  = info.currentTime;

        if( observer )
        {
            observer->RecheckFiles();
        }

        if( nvDlss3dx12 )
        {
            nvDlss3dx12->Reflex_SimEnd();
            nvDlss3dx12->Reflex_RenderStart();
        }

        FramebufferImageIndex rendered;

        if( renderResolution.Width() > 0 && renderResolution.Height() > 0 )
        {
            FillUniform( uniform->GetData(), info );
            Dev_Draw();
            rendered = Render( cmd, info );
        }
        else
        {
            rendered = m_prevAccum;
        }

        EndFrame( cmd, rendered );
        currentFrameState.OnEndFrame();


        sceneImportExport->TryExport( *textureManager, ovrdFolder );
    };

    auto drawFrame_WithScene = [ this, &drawFrame_Core ]( const RgDrawFrameInfo& original ) {
        auto modified            = RgDrawFrameInfo{ original };
        auto modified_Volumetric = pnext::get< RgDrawFrameVolumetricParams >( original );
        auto modified_Sky        = pnext::get< RgDrawFrameSkyParams >( original );

        sceneMetaManager->Modify(
            sceneImportExport->GetImportMapName(), modified_Volumetric, modified_Sky );

        // clang-format off
        modified_Volumetric     .pNext = modified.pNext;
        modified_Sky            .pNext = &modified_Volumetric;
        modified                .pNext = &modified_Sky;
        // clang-format on

        drawFrame_Core( modified );
    };

    auto drawFrame_WithDevmode = [ this, &drawFrame_WithScene ]( const RgDrawFrameInfo& original ) {
        auto modified              = RgDrawFrameInfo{ original };
        auto modified_Illumination = pnext::get< RgDrawFrameIlluminationParams >( original );
        auto modified_Tonemapping  = pnext::get< RgDrawFrameTonemappingParams >( original );
        auto modified_Textures     = pnext::get< RgDrawFrameTexturesParams >( original );

        // clang-format off
        modified_Illumination   .pNext = modified.pNext;
        modified_Tonemapping    .pNext = &modified_Illumination;
        modified_Textures       .pNext = &modified_Tonemapping;
        modified                .pNext = &modified_Textures;
        // clang-format on

        Dev_Override( modified_Illumination, modified_Tonemapping, modified_Textures );

        drawFrame_WithScene( modified );
    };

    if( Dev_IsDevmodeInitialized() )
    {
        drawFrame_WithDevmode( *pOriginalInfo );
    }
    else
    {
        drawFrame_WithScene( *pOriginalInfo );
    }
}

namespace RTGL1
{
namespace
{
    bool IsRasterized( const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& primitive )
    {
        if( primitive.flags & RG_MESH_PRIMITIVE_DECAL )
        {
            return true;
        }

        if( primitive.flags & RG_MESH_PRIMITIVE_SKY )
        {
            return true;
        }

        if( !( primitive.flags & RG_MESH_PRIMITIVE_GLASS ) &&
            !( mesh.flags & RG_MESH_FORCE_GLASS ) &&
            !( primitive.flags & RG_MESH_PRIMITIVE_WATER ) &&
            !( mesh.flags & RG_MESH_FORCE_WATER ) && !( primitive.flags & RG_MESH_PRIMITIVE_ACID ) )
        {
            if( primitive.flags & RG_MESH_PRIMITIVE_TRANSLUCENT )
            {
                return true;
            }

            if( Utils::UnpackAlphaFromPacked32( primitive.color ) <
                MESH_TRANSLUCENT_ALPHA_THRESHOLD )
            {
                return true;
            }
        }

        return false;
    }
}
}

void RTGL1::VulkanDevice::UploadMeshPrimitive( const RgMeshInfo*          pMesh,
                                               const RgMeshPrimitiveInfo* pPrimitive )
{
    if( pPrimitive == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pPrimitive->sType != RG_STRUCTURE_TYPE_MESH_PRIMITIVE_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( pPrimitive->vertexCount == 0 || pPrimitive->pVertices == nullptr )
    {
        return;
    }
    Dev_TryBreak( pPrimitive->pTextureName, false );


    auto logDebugStat = [ this ]( Devmode::DebugPrimMode     mode,
                                  const RgMeshInfo*          mesh,
                                  const RgMeshPrimitiveInfo& prim,
                                  UploadResult               rtResult = UploadResult::Fail ) {
        if( !devmode || devmode->primitivesTableMode != mode )
        {
            return;
        }

        switch( mode )
        {
            case Devmode::DebugPrimMode::RayTraced:
                devmode->primitivesTable.push_back( Devmode::DebugPrim{
                    .result         = rtResult,
                    .callIndex      = uint32_t( devmode->primitivesTable.size() ),
                    .objectId       = mesh->uniqueObjectID,
                    .meshName       = Utils::SafeCstr( mesh->pMeshName ),
                    .primitiveIndex = prim.primitiveIndexInMesh,
                    .textureName    = Utils::SafeCstr( prim.pTextureName ),
                } );
                break;
            case Devmode::DebugPrimMode::Rasterized:
                devmode->primitivesTable.push_back( Devmode::DebugPrim{
                    .result         = UploadResult::Dynamic,
                    .callIndex      = uint32_t( devmode->primitivesTable.size() ),
                    .objectId       = mesh->uniqueObjectID,
                    .meshName       = Utils::SafeCstr( mesh->pMeshName ),
                    .primitiveIndex = prim.primitiveIndexInMesh,
                    .textureName    = Utils::SafeCstr( prim.pTextureName ),
                } );
                break;
            case Devmode::DebugPrimMode::NonWorld:
                devmode->primitivesTable.push_back( Devmode::DebugPrim{
                    .result         = UploadResult::Dynamic,
                    .callIndex      = uint32_t( devmode->primitivesTable.size() ),
                    .objectId       = 0,
                    .meshName       = {},
                    .primitiveIndex = prim.primitiveIndexInMesh,
                    .textureName    = Utils::SafeCstr( prim.pTextureName ),
                } );
                break;
            case Devmode::DebugPrimMode::Decal:
                devmode->primitivesTable.push_back( Devmode::DebugPrim{
                    .result         = UploadResult::Dynamic,
                    .callIndex      = uint32_t( devmode->primitivesTable.size() ),
                    .objectId       = 0,
                    .meshName       = {},
                    .primitiveIndex = 0,
                    .primitiveName  = {},
                    .textureName    = Utils::SafeCstr( prim.pTextureName ),
                } );
                break;
            case Devmode::DebugPrimMode::None:
            default: break;
        }
    };

    // --- //

    auto uploadPrimitive_Core = [ this, &logDebugStat ]( const RgMeshInfo&          mesh,
                                                         const RgMeshPrimitiveInfo& prim ) {
        assert( !pnext::find< RgMeshPrimitiveSwapchainedEXT >( &prim ) );

        if( IsRasterized( mesh, prim ) )
        {
            rasterizer->Upload( currentFrameState.GetFrameIndex(),
                                prim.flags & RG_MESH_PRIMITIVE_SKY     ? GeometryRasterType::SKY
                                : prim.flags & RG_MESH_PRIMITIVE_DECAL ? GeometryRasterType::DECAL
                                                                       : GeometryRasterType::WORLD,
                                mesh.transform,
                                prim,
                                nullptr,
                                nullptr );

            logDebugStat( prim.flags & RG_MESH_PRIMITIVE_DECAL ? Devmode::DebugPrimMode::Decal
                                                               : Devmode::DebugPrimMode::Rasterized,
                          &mesh,
                          prim );
        }
        else
        {
            // upload a primitive, potentially loading replacements
            UploadResult r = scene->UploadPrimitive( currentFrameState.GetFrameIndex(),
                                                     mesh,
                                                     prim,
                                                     *textureManager,
                                                     *lightManager,
                                                     false );

            if( lightmapScreenCoverage > 0 )
            {
                if( !( mesh.flags & RG_MESH_FIRST_PERSON_VIEWER ) )
                {
                    rasterizer->Upload( currentFrameState.GetFrameIndex(),
                                        GeometryRasterType::WORLD_CLASSIC,
                                        mesh.transform,
                                        prim,
                                        nullptr,
                                        nullptr );
                }
            }

            logDebugStat( Devmode::DebugPrimMode::RayTraced, &mesh, prim, r );


            if( auto e = sceneImportExport->TryGetExporter( mesh.flags &
                                                            RG_MESH_EXPORT_AS_SEPARATE_FILE ) )
            {
                auto allowMeshExport = [ & ]( const RgMeshInfo& m ) {
                    if( r != UploadResult::ExportableDynamic &&
                        r != UploadResult::ExportableStatic )
                    {
                        return false;
                    }
                    if( scene->ReplacementExists( m ) )
                    {
                        if( devmode && devmode->allowExportOfExistingReplacements )
                        {
                            return true;
                        }
                        return false;
                    }
                    return true;
                };

                if( allowMeshExport( mesh ) )
                {
                    e->AddPrimitive( mesh, prim );
                }

                // SHIPPING_HACK: add lights to the scene gltf even for non-exportable geometry
                if( !( mesh.flags & RG_MESH_EXPORT_AS_SEPARATE_FILE ) )
                {
                    e->AddPrimitiveLights( mesh, prim );
                }
            }


            // TODO: remove legacy way to attach lights
            if( auto attachedLight = pnext::find< RgMeshPrimitiveAttachedLightEXT >( &prim ) )
            {
                bool quad = ( prim.indexCount == 6 && prim.vertexCount == 4 ) ||
                            ( prim.indexCount == 0 && prim.vertexCount == 6 );

                if( attachedLight->evenOnDynamic || quad )
                {
                    assert( tempStorageLights.empty() );

                    if( quad )
                    {
                        auto center = RgFloat3D{ 0, 0, 0 };
                        {
                            for( uint32_t v = 0; v < prim.vertexCount; v++ )
                            {
                                center.data[ 0 ] += prim.pVertices[ v ].position[ 0 ];
                                center.data[ 1 ] += prim.pVertices[ v ].position[ 1 ];
                                center.data[ 2 ] += prim.pVertices[ v ].position[ 2 ];
                            }
                            center.data[ 0 ] /= float( prim.vertexCount );
                            center.data[ 1 ] /= float( prim.vertexCount );
                            center.data[ 2 ] /= float( prim.vertexCount );
                        }

                        center.data[ 0 ] += mesh.transform.matrix[ 0 ][ 3 ];
                        center.data[ 1 ] += mesh.transform.matrix[ 1 ][ 3 ];
                        center.data[ 2 ] += mesh.transform.matrix[ 2 ][ 3 ];

                        tempStorageLights.emplace_back( RgLightSphericalEXT{
                            .sType     = RG_STRUCTURE_TYPE_LIGHT_SPHERICAL_EXT,
                            .pNext     = nullptr,
                            .color     = attachedLight->color,
                            .intensity = attachedLight->intensity,
                            .position  = center,
                            .radius    = 0.1f,
                        } );
                    }
                    else
                    {
                        GltfExporter::MakeLightsForPrimitiveDynamic(
                            mesh,
                            prim,
                            sceneImportExport->GetWorldScale(),
                            tempStorageInit,
                            tempStorageLights );
                    }

                    auto hashCombine = []< typename T >( uint64_t seed, const T& v ) {
                        seed ^= std::hash< T >{}( v ) + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
                        return seed;
                    };

                    static const uint64_t attchSalt =
                        hashCombine( 0, std::string_view{ "attachedlight" } );

                    // NOTE: can't use texture / mesh name, as texture can be
                    // just 1 frame of animation sequence.. so this is more stable
                    uint64_t hashBase{ attchSalt };
                    hashBase = hashCombine( hashBase, mesh.uniqueObjectID );
                    hashBase = hashCombine( hashBase, prim.primitiveIndexInMesh );

                    uint64_t counter = 0;

                    for( AnyLightEXT& lext : tempStorageLights )
                    {
                        std::visit(
                            [ & ]< typename T >( T& specific ) {
                                static_assert( detail::AreLinkable< T, RgLightInfo > );
                                
                                RgLightInfo linfo = {
                                    .sType        = RG_STRUCTURE_TYPE_LIGHT_INFO,
                                    .pNext        = &specific,
                                    .uniqueID     = hashCombine( hashBase, counter ),
                                    .isExportable = false,
                                };

                                UploadLight( &linfo );
                            },
                            lext );

                        counter++;
                    }

                    tempStorageInit.clear();
                    tempStorageLights.clear();
                }
            }
        }
    };

    // --- //

    auto uploadPrimitive_WithMeta = [ this, &uploadPrimitive_Core ](
                                        const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& prim ) {
        // ignore replacement, if the scene requires
        if( mesh.isExportable && ( mesh.flags & RG_MESH_EXPORT_AS_SEPARATE_FILE ) &&
            !Utils::IsCstrEmpty( mesh.pMeshName ) )
        {
            if( sceneMetaManager->IsReplacementIgnored( sceneImportExport->GetImportMapName(),
                                                        mesh.pMeshName ) )
            {
                return;
            }
        }

        auto modified = RgMeshPrimitiveInfo{ prim };

        auto modified_attachedLight = std::optional< RgMeshPrimitiveAttachedLightEXT >{};
        auto modified_pbr           = std::optional< RgMeshPrimitivePBREXT >{};

        if( auto original = pnext::find< RgMeshPrimitiveAttachedLightEXT >( &prim ) )
        {
            modified_attachedLight = *original;
        }

        if( auto original = pnext::find< RgMeshPrimitivePBREXT >( &prim ) )
        {
            modified_pbr = *original;
        }

        if( mesh.flags & RG_MESH_FORCE_MIRROR )
        {
            modified.flags |= RG_MESH_PRIMITIVE_MIRROR;
        }
        if( mesh.flags & RG_MESH_FORCE_GLASS )
        {
            modified.flags |= RG_MESH_PRIMITIVE_GLASS;
        }
        if( mesh.flags & RG_MESH_FORCE_WATER )
        {
            modified.flags |= RG_MESH_PRIMITIVE_WATER;
        }

        if( !textureMetaManager->Modify( modified, modified_attachedLight, modified_pbr, false ) )
        {
            return;
        }

        // Dev Materials A/B: drop texture-meta emissives / attached lights on upload.
        if( devmode && devmode->materialStripEmissives )
        {
            modified.emissive = 0.0f;
            modified_attachedLight.reset();
        }
        if( devmode && modified_pbr )
        {
            if( devmode->materialStripMetallic )
            {
                modified_pbr->metallicDefault = 0.0f;
            }
            if( devmode->materialStripRoughness )
            {
                modified_pbr->roughnessDefault = 1.0f;
            }
        }

        if( modified_attachedLight )
        {
            // insert
            modified_attachedLight.value().pNext = modified.pNext;
            modified.pNext                       = &modified_attachedLight.value();
        }

        if( modified_pbr )
        {
            // insert
            modified_pbr.value().pNext = modified.pNext;
            modified.pNext             = &modified_pbr.value();
        }

        uploadPrimitive_Core( mesh, modified );
    };

    // --- //

    auto uploadPrimitive_FilterSwapchained = [ this, &uploadPrimitive_WithMeta, &logDebugStat ](
                                                 const RgMeshInfo*          mesh,
                                                 const RgMeshPrimitiveInfo& prim ) {
        if( mesh )
        {
            if( mesh->sType != RG_STRUCTURE_TYPE_MESH_INFO )
            {
                throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
            }
        }

        if( auto raster = pnext::find< RgMeshPrimitiveSwapchainedEXT >( &prim ) )
        {
            float vp[ 16 ];
            if( raster->pViewProjection )
            {
                memcpy( vp, raster->pViewProjection, sizeof( vp ) );
            }
            else
            {
                const auto& cameraInfo = scene->GetCamera( renderResolution.Aspect() );

                const float* v = raster->pView ? raster->pView : cameraInfo.view;
                const float* p = raster->pProjection ? raster->pProjection : cameraInfo.projection;
                Matrix::Multiply( vp, p, v );
            }

            rasterizer->Upload( currentFrameState.GetFrameIndex(),
                                GeometryRasterType::SWAPCHAIN,
                                mesh ? mesh->transform : RG_TRANSFORM_IDENTITY,
                                prim,
                                vp,
                                raster->pViewport );

            logDebugStat( Devmode::DebugPrimMode::NonWorld, nullptr, prim );
        }
        else
        {
            if( mesh == nullptr )
            {
                throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
            }
            if( mesh->flags & RG_MESH_EXPORT_AS_SEPARATE_FILE )
            {
                if( !mesh->isExportable )
                {
                    throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT,
                                       "RG_MESH_INFO_EXPORT_AS_SEPARATE_FILE is set, "
                                       "expected isExportable to be true" );
                }
            }

            uploadPrimitive_WithMeta( *mesh, prim );
        }
    };

    // --- //

    uploadPrimitive_FilterSwapchained( pMesh, *pPrimitive );
}

void RTGL1::VulkanDevice::UploadLensFlare( const RgLensFlareInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_LENS_FLARE_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }

    float emisMult = 0.0f;

    if( auto meta = textureMetaManager->Access( pInfo->pTextureName ) )
    {
        emisMult = meta->emissiveMult;

        if( meta->forceIgnore || meta->forceIgnoreIfRasterized )
        {
            return;
        }
    }

    rasterizer->UploadLensFlare(
        currentFrameState.GetFrameIndex(), *pInfo, emisMult, *textureManager );

    if( devmode && devmode->primitivesTableMode == Devmode::DebugPrimMode::Rasterized )
    {
        devmode->primitivesTable.push_back( Devmode::DebugPrim{
            .result         = UploadResult::Dynamic,
            .callIndex      = uint32_t( devmode->primitivesTable.size() ),
            .objectId       = 0,
            .meshName       = {},
            .primitiveIndex = 0,
            .primitiveName  = {},
            .textureName    = Utils::SafeCstr( pInfo->pTextureName ),
        } );
    }
}

void RTGL1::VulkanDevice::SpawnFluid( const RgSpawnFluidInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_SPAWN_FLUID_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( !fluid )
    {
        return;
    }
    fluid->AddSource( *pInfo );
}

void RTGL1::VulkanDevice::UploadCamera( const RgCameraInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_CAMERA_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( Utils::SqrLength( pInfo->right.data ) < 0.01f )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Null RgCameraInfo::right" );
    }
    if( Utils::SqrLength( pInfo->up.data ) < 0.01f )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Null RgCameraInfo::up" );
    }

    auto base = [ this ]( const RgCameraInfo& info ) {
        scene->AddDefaultCamera( info );

        if( auto readback =
                pnext::find< RgCameraInfoReadbackEXT >( const_cast< RgCameraInfo* >( &info ) ) )
        {
            const Camera& cameraInfo = scene->GetCamera( renderResolution.Aspect() );

            static_assert( sizeof readback->view == sizeof cameraInfo.view );
            static_assert( sizeof readback->projection == sizeof cameraInfo.projection );
            static_assert( sizeof readback->viewInverse == sizeof cameraInfo.viewInverse );
            static_assert( sizeof readback->projectionInverse ==
                           sizeof cameraInfo.projectionInverse );

            memcpy( readback->view, cameraInfo.view, sizeof cameraInfo.view );
            memcpy( readback->projection, cameraInfo.projection, sizeof cameraInfo.projection );
            memcpy( readback->viewInverse, cameraInfo.viewInverse, sizeof cameraInfo.viewInverse );
            memcpy( readback->projectionInverse,
                    cameraInfo.projectionInverse,
                    sizeof cameraInfo.projectionInverse );
        }
    };

    if( Dev_IsDevmodeInitialized() )
    {
        auto modified = RgCameraInfo{ *pInfo };
        Dev_Override( modified );

        base( modified );
    }
    else
    {
        base( *pInfo );
    }
}

void RTGL1::VulkanDevice::UploadLight( const RgLightInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_LIGHT_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }

    auto findExt =
        []( const RgLightInfo& info ) -> std::optional< std::variant< RgLightDirectionalEXT,
                                                                      RgLightSphericalEXT,
                                                                      RgLightSpotEXT,
                                                                      RgLightPolygonalEXT > > {
        if( auto l = pnext::find< RgLightDirectionalEXT >( &info ) )
        {
            return *l;
        }
        if( auto l = pnext::find< RgLightSphericalEXT >( &info ) )
        {
            return *l;
        }
        if( auto l = pnext::find< RgLightSpotEXT >( &info ) )
        {
            return *l;
        }
        if( auto l = pnext::find< RgLightPolygonalEXT >( &info ) )
        {
            return *l;
        }
        return {};
    };

    auto findAdditional = []( const RgLightInfo& info ) -> std::optional< RgLightAdditionalEXT > {
        if( auto l = pnext::find< RgLightAdditionalEXT >( &info ) )
        {
            return *l;
        }
        return {};
    };

    auto ext = findExt( *pInfo );
    if( !ext )
    {
        debug::Warning( "Couldn't find RgLightDirectionalEXT, RgLightSphericalEXT, RgLightSpotEXT "
                        "or RgLightPolygonalEXT on RgLightInfo (uniqueID={})",
                        pInfo->uniqueID );
        return;
    }

    auto light = LightCopy{
        .base       = *pInfo,
        .extension  = *ext,
        .additional = findAdditional( *pInfo ),
    };

    // reset pNext, as using in-place members
    {
        light.base.pNext = nullptr;
        std::visit( []( auto& e ) { e.pNext = nullptr; }, light.extension );
        if( light.additional )
        {
            light.additional->pNext = nullptr;
        }
    }

    UploadResult r =
        scene->UploadLight( currentFrameState.GetFrameIndex(), light, *lightManager, false );

    if( auto e = sceneImportExport->TryGetExporter( false ) )
    {
        if( r == UploadResult::ExportableDynamic || r == UploadResult::ExportableStatic )
        {
            e->AddLight( light );
        }
    }
}

void RTGL1::VulkanDevice::ProvideOriginalTexture( const RgOriginalTextureInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_ORIGINAL_TEXTURE_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    Dev_TryBreak( pInfo->pTextureName, true );

    textureManager->TryCreateMaterial( currentFrameState.GetCmdBufferForMaterials( cmdManager ),
                                       currentFrameState.GetFrameIndex(),
                                       *pInfo,
                                       ovrdFolder );

    // SHIPPING_HACK begin
    if( !Utils::IsCstrEmpty( pInfo->pTextureName ) )
    {
        auto texturesToUpdateOnStaticGeom = scene->m_primitivesToUpdateTextures.find( pInfo->pTextureName );
        if( texturesToUpdateOnStaticGeom != scene->m_primitivesToUpdateTextures.end() )
        {
            for( const PrimitiveUniqueID& geomUniqueId : texturesToUpdateOnStaticGeom->second )
            {
                scene->GetASManager()->Hack_PatchTexturesForStaticPrimitive(
                    geomUniqueId, pInfo->pTextureName, *textureManager );
            }
        }
    }
    // SHIPPING_HACK end
}

void RTGL1::VulkanDevice::MarkOriginalTextureAsDeleted( const char* pTextureName )
{
    textureManager->TryDestroyMaterial( currentFrameState.GetFrameIndex(), pTextureName );
    cubemapManager->TryDestroyCubemap( currentFrameState.GetFrameIndex(), pTextureName );
}

bool RTGL1::VulkanDevice::IsUpscaleTechniqueAvailable( RgRenderUpscaleTechnique technique,
                                                       RgFrameGenerationMode    frameGeneration,
                                                       const char** ppFailureReason ) const
{
    if( ppFailureReason )
    {
        *ppFailureReason = nullptr;
    }

    switch( technique )
    {
        case RG_RENDER_UPSCALE_TECHNIQUE_NEAREST:
        case RG_RENDER_UPSCALE_TECHNIQUE_LINEAR:
            if( frameGeneration != RG_FRAME_GENERATION_MODE_OFF )
            {
                return false;
            }
            return true;


        case RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2:
#ifdef __APPLE__
            if( frameGeneration != RG_FRAME_GENERATION_MODE_OFF )
            {
                // MetalFX frame interpolation, with the MetalFX upscaler
                if( !amdFsr2 || !MetalFI::Supported( device ) )
                {
                    if( ppFailureReason )
                    {
                        *ppFailureReason = "MetalFX frame interpolation needs macOS 26 and a supported GPU";
                    }
                    return false;
                }
                return true;
            }
#endif
            if( frameGeneration != RG_FRAME_GENERATION_MODE_OFF )
            {
                const char* error = swapchain->FailReason( SWAPCHAIN_TYPE_FRAME_GENERATION_FSR3 );
                assert( error == nullptr || error[ 0 ] != '\0' );

                if( ppFailureReason )
                {
                    *ppFailureReason = error;
                }
                return !error;
            }
            return bool( amdFsr2 );


        case RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS: {
            if( frameGeneration != RG_FRAME_GENERATION_MODE_OFF )
            {
                const char* error = swapchain->FailReason( SWAPCHAIN_TYPE_FRAME_GENERATION_DLSS3 );
                assert( error == nullptr || error[ 0 ] != '\0' );

                if( ppFailureReason )
                {
                    *ppFailureReason = error;
                }
                return !error;
            }
            return bool( nvDlss2 );
        }

        default: {
            throw RgException(
                RG_RESULT_WRONG_FUNCTION_ARGUMENT,
                "Incorrect technique was passed to rgIsRenderUpscaleTechniqueAvailable" );
        }
    }
}

bool RTGL1::VulkanDevice::IsDXGIAvailable( const char** ppFailureReason ) const
{
    const char* dxgiError = swapchain->FailReason( SWAPCHAIN_TYPE_DXGI );
    assert( dxgiError == nullptr || dxgiError[ 0 ] != '\0' );

    if( ppFailureReason )
    {
        *ppFailureReason = dxgiError;
    }
    return swapchain && !dxgiError;
}

RgFeatureFlags RTGL1::VulkanDevice::GetSupportedFeatures() const
{
    RgFeatureFlags f = 0;

    if( swapchain && swapchain->SupportsHDR() )
    {
        f |= RG_FEATURE_HDR;
    }

    if( m_supportsRayQueryAndPositionFetch )
    {
        f |= RG_FEATURE_FLUID;
    }

    return f;
}

RgUtilMemoryUsage RTGL1::VulkanDevice::RequestMemoryUsage() const
{
    auto& [ r_lastTime, r_usage ] = cachedMemoryUsage;

    constexpr auto CheckEachSeconds = 0.5;
    if( std::abs( currentFrameTime - r_lastTime ) > CheckEachSeconds )
    {
        r_lastTime = currentFrameTime;
        r_usage    = RTGL1::RequestMemoryUsage( physDevice->Get() );
    }

    return r_usage;
}

RgPrimitiveVertex* RTGL1::VulkanDevice::ScratchAllocForVertices( uint32_t vertexCount )
{
    // TODO: scratch allocator
    return new RgPrimitiveVertex[ vertexCount ];
}

void RTGL1::VulkanDevice::ScratchFree( const RgPrimitiveVertex* pPointer )
{
    // TODO: scratch allocator
    delete[] pPointer;
}

void RTGL1::VulkanDevice::Print( std::string_view msg, RgMessageSeverityFlags severity ) const
{
    static auto printMutex = std::mutex{};

    auto l = std::lock_guard{ printMutex };

    if( devmode )
    {
        auto getCountIfSameAsLast = [ & ]() -> uint32_t* {
            if( !devmode->logs.empty() )
            {
                auto& [ severityLast, count, msgLast ] = devmode->logs.back();
                if( severityLast == severity && msgLast == msg )
                {
                    return &count;
                }
            }
            return nullptr;
        };

        if( uint32_t* same = getCountIfSameAsLast() )
        {
            *same = *same + 1;
        }
        else
        {
            if( devmode->logs.size() > 2048 )
            {
                devmode->logs.pop_front();
            }

            devmode->logs.emplace_back( severity, 1, msg );
        }
    }

    if( userPrint )
    {
        userPrint->Print( msg.data(), severity );
    }
}
