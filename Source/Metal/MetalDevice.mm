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

#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>

#include "MetalDevice.h"

#include "MetalComputePipelines.h"
#include "MetalFluid.h"
#include "MetalFramebuffers.h"
#include "MetalLightManager.h"
#include "MetalPathTracer.h"
#include "MetalRasterizer.h"
#include "MetalScene.h"
#include "MetalSceneManager.h"
#include "MetalTextureManager.h"
#include "Shaders/MetalBindings.h"

#include "DrawFrameInfo.h"
#include "JsonParser.h"
#include "LibraryConfig.h"
#include "HaltonSequence.h"
#include "ImageLoader.h"
#include "Matrix.h"
#include "PrimitiveLights.h"
#include "PrimitiveRouting.h"
#include "SceneMeta.h"
#include "TextureMeta.h"
#include "RgException.h"

#include "Stb/stb_image_write.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <optional>

// the same as in VulkanDevice_Init.cpp
namespace RTGL1
{

auto g_libConfig = std::optional< LibraryConfig >{};

auto LibConfig() -> const LibraryConfig&
{
    if( !g_libConfig )
    {
        assert( 0 ); // reading before init
        static auto def = LibraryConfig{};
        return def;
    }
    return *g_libConfig;
}

auto g_ovrdFolder = std::filesystem::path{}; // hack

}

namespace
{

// Warn once per entry point that the Metal backend doesn't handle it yet,
// so the application still runs while features are being ported.
void NotImplementedYet( const char* what )
{
    static std::mutex                      m;
    static std::vector< const char* >      reported;

    auto l = std::lock_guard{ m };
    if( std::ranges::find( reported, what ) == reported.end() )
    {
        reported.push_back( what );
        RTGL1::debug::Warning( "Metal backend: {} is not implemented yet", what );
    }
}

// the folder of libRTGL1.dylib
std::filesystem::path GetLibraryFolder()
{
    Dl_info info{};
    if( !dladdr( reinterpret_cast< void* >( &GetLibraryFolder ), &info ) || !info.dli_fname )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER, "Can't find the RTGL1 library path" );
    }
    return std::filesystem::path{ info.dli_fname }.parent_path();
}

// RTGL1.metallib is built next to the library
id< MTLLibrary > LoadShaderLibrary( id< MTLDevice > device )
{
    const auto path = GetLibraryFolder() / "RTGL1.metallib";

    NSError*         error = nil;
    id< MTLLibrary > lib   = [device newLibraryWithURL:[NSURL fileURLWithPath:@( path.c_str() )]
                                               error:&error];
    if( !lib )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                                  std::string{ "Can't load " } + path.string() + ": " +
                                      error.localizedDescription.UTF8String );
    }
    return lib;
}

id< MTLFunction > GetFunction( id< MTLLibrary > lib, NSString* name )
{
    id< MTLFunction > f = [lib newFunctionWithName:name];
    if( !f )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                                  std::string{ "Can't find shader function " } + name.UTF8String );
    }
    return f;
}

// Debug: RTGL1_METAL_CAPTURE=<file.png> saves the presented image of frame RTGL1_METAL_CAPTURE_FRAME
// (default: 60), to check the output without a screen, e.g. for image comparisons.
struct CaptureRequest
{
    std::string path;
    uint64_t    frame;
    // RTGL1_METAL_CAPTURE_COUNT: consecutive frames, saved as <path>_<i>.png
    uint64_t    count;

    bool Contains( uint64_t f ) const { return f >= frame && f < frame + count; }

    std::string PathFor( uint64_t f ) const
    {
        if( count <= 1 )
        {
            return path;
        }
        const auto dot = path.rfind( '.' );
        return path.substr( 0, dot ) + "_" + std::to_string( f - frame ) +
               ( dot == std::string::npos ? "" : path.substr( dot ) );
    }
};

std::optional< CaptureRequest > GetCaptureRequest()
{
    const char* path = std::getenv( "RTGL1_METAL_CAPTURE" );
    if( !path || path[ 0 ] == '\0' )
    {
        return std::nullopt;
    }
    const char* frame = std::getenv( "RTGL1_METAL_CAPTURE_FRAME" );
    const char* count = std::getenv( "RTGL1_METAL_CAPTURE_COUNT" );
    return CaptureRequest{
        .path  = path,
        .frame = frame ? std::strtoull( frame, nullptr, 10 ) : 60,
        .count = std::max< uint64_t >( 1, count ? std::strtoull( count, nullptr, 10 ) : 1 ),
    };
}

void OnFrameCompleted( id< MTLCommandBuffer > c, std::atomic< double >& gpuFrameTime )
{
    if( c.GPUEndTime > c.GPUStartTime )
    {
        const double t = std::min( c.GPUEndTime - c.GPUStartTime, 0.25 );
        gpuFrameTime.store( gpuFrameTime.load() * 0.9 + t * 0.1 );
    }
    if( c.status == MTLCommandBufferStatusError )
    {
        static uint32_t reported = 0;
        if( reported++ < 8 )
        {
            RTGL1::debug::Error( "Metal backend: command buffer failed: {}",
                                 c.error ? c.error.localizedDescription.UTF8String : "(no info)" );
        }
    }
}

// bgra8 is the swapchain image, sRGB-encoded
void SaveCapture( id< MTLBuffer > bgra8, uint32_t width, uint32_t height, const std::string& path )
{
    const auto* src = static_cast< const uint8_t* >( bgra8.contents );
    auto        dst = std::vector< uint8_t >( size_t( width ) * height * 4 );
    for( size_t i = 0; i < size_t( width ) * height; i++ )
    {
        dst[ i * 4 + 0 ] = src[ i * 4 + 2 ];
        dst[ i * 4 + 1 ] = src[ i * 4 + 1 ];
        dst[ i * 4 + 2 ] = src[ i * 4 + 0 ];
        dst[ i * 4 + 3 ] = 255;
    }
    if( stbi_write_png( path.c_str(), int( width ), int( height ), 4, dst.data(), int( width * 4 ) ) )
    {
        RTGL1::debug::Info( "Metal backend: captured a frame to {}", path );
    }
    else
    {
        RTGL1::debug::Warning( "Metal backend: failed to write {}", path );
    }
}

void CheckPipeline( id pso, NSError* error, const char* name )
{
    if( !pso )
    {
        throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR,
                                  std::string{ "Can't create pipeline " } + name + ": " +
                                      error.localizedDescription.UTF8String );
    }
}

}

namespace
{

// view and projection of a camera, the same as MakeCamera in Scene.cpp
// Debug: RTGL1_METAL_DEBUG_SPIN=<radians> sways the camera, to test temporal techniques without input
float g_debugSpinAngle = 0.0f;

void MakeCameraMatrices( const RgCameraInfo& c, float ( &view )[ 16 ], float ( &projection )[ 16 ] )
{
    if( c.pView )
    {
        memcpy( view, c.pView, sizeof( view ) );
    }
    else
    {
        RTGL1::Matrix::MakeViewMatrix( view, c.position, c.right, c.up );
    }
    if( g_debugSpinAngle != 0.0f )
    {
        // view space rotation around Y, applied after the view matrix (column-major)
        const float cs = std::cos( g_debugSpinAngle ), sn = std::sin( g_debugSpinAngle );
        const float r[ 16 ] = { cs, 0, -sn, 0, 0, 1, 0, 0, sn, 0, cs, 0, 0, 0, 0, 1 };
        float       v[ 16 ];
        memcpy( v, view, sizeof( v ) );
        for( int col = 0; col < 4; col++ )
        {
            for( int row = 0; row < 4; row++ )
            {
                float sum = 0;
                for( int k = 0; k < 4; k++ )
                {
                    sum += r[ k * 4 + row ] * v[ col * 4 + k ];
                }
                view[ col * 4 + row ] = sum;
            }
        }
    }
    RTGL1::Matrix::MakeProjectionMatrix( projection, c.aspect, c.fovYRadians, c.cameraNear, c.cameraFar );
}

struct UniformFrameInfo
{
    uint32_t  renderWidth;
    uint32_t  renderHeight;
    uint32_t  upscaledWidth;
    uint32_t  upscaledHeight;
    RgFloat2D jitter;
    uint32_t  frameId;
    double    currentTime;
    double    previousTime;
    RgFloat3D worldUp;
    bool      rayCullBackFaces;
    uint32_t  frameIndex;
    uint32_t  waterNormalTextureIndex;
    uint32_t  dirtMaskTextureIndex;
    float     lightmapScreenCoverage;
    uint32_t  volumeLightSourceIndex;
    bool      hdr;
};

// The same as VulkanDevice::FillUniform.
// 'gu' persists between frames: the previous frame's values are used for the *Prev ones.
// TODO: cubemaps, volumetrics
void FillUniform( RTGL1::ShGlobalUniform&              gu,
                  const RgDrawFrameInfo&               drawInfo,
                  const std::optional< RgCameraInfo >& camera,
                  const RTGL1::MetalLightManager&      lights,
                  const UniformFrameInfo&              fi )
{
    using namespace RTGL1;

    const float IdentityMat4x4[ 16 ] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    const float cameraNear = camera ? camera->cameraNear : 0.1f;
    const float cameraFar  = camera ? camera->cameraFar : 10000.0f;

    {
        memcpy( gu.viewPrev, gu.view, 16 * sizeof( float ) );
        memcpy( gu.projectionPrev, gu.projection, 16 * sizeof( float ) );
        memcpy( gu.cameraPositionPrev, gu.cameraPosition, 3 * sizeof( float ) );

        if( camera )
        {
            MakeCameraMatrices( *camera, gu.view, gu.projection );
        }
        else
        {
            memcpy( gu.view, IdentityMat4x4, sizeof( gu.view ) );
            memcpy( gu.projection, IdentityMat4x4, sizeof( gu.projection ) );
        }
        Matrix::Inverse( gu.invView, gu.view );
        Matrix::Inverse( gu.invProjection, gu.projection );

        gu.cameraPosition[ 0 ] = gu.invView[ 12 ];
        gu.cameraPosition[ 1 ] = gu.invView[ 13 ];
        gu.cameraPosition[ 2 ] = gu.invView[ 14 ];
    }

    {
        gu.frameId   = fi.frameId;
        gu.timeDelta = float( std::max< double >( fi.currentTime - fi.previousTime, 0.001 ) );
        gu.time      = float( fi.currentTime );
    }

    {
        gu.renderWidth          = float( fi.renderWidth );
        gu.renderHeight         = float( fi.renderHeight );
        gu.upscaledRenderWidth  = float( fi.upscaledWidth );
        gu.upscaledRenderHeight = float( fi.upscaledHeight );
        gu.jitterX              = fi.jitter.data[ 0 ];
        gu.jitterY              = fi.jitter.data[ 1 ];
    }

    {
        const auto& params = pnext::get< RgDrawFrameTonemappingParams >( drawInfo );

        float luminanceMin = std::exp2( params.ev100Min ) * 12.5f / 100.0f;
        float luminanceMax = std::exp2( params.ev100Max ) * 12.5f / 100.0f;

        gu.stopEyeAdaptation   = params.disableEyeAdaptation;
        gu.minLogLuminance     = std::log2( luminanceMin );
        gu.maxLogLuminance     = std::log2( luminanceMax );
        gu.luminanceWhitePoint = params.luminanceWhitePoint;
    }

    {
        gu.lightCount             = lights.GetLightCount();
        gu.lightCountPrev         = lights.GetLightCountPrev();
        gu.directionalLightExists = lights.DoesDirectionalLightExist();
    }

    {
        const auto& params = pnext::get< RgDrawFrameSkyParams >( drawInfo );

        memcpy( gu.skyCubemapRotationTransform, IdentityMat4x4, 16 * sizeof( float ) );

        RG_SET_VEC3_A( gu.skyColorDefault, params.skyColorDefault.data );
        gu.skyColorMultiplier = std::max( 0.0f, params.skyColorMultiplier );
        gu.skyColorSaturation = std::max( 0.0f, params.skyColorSaturation );

        switch( params.skyType )
        {
            case RG_SKY_TYPE_RASTERIZED_GEOMETRY: gu.skyType = SKY_TYPE_RASTERIZED_GEOMETRY; break;
            case RG_SKY_TYPE_CUBEMAP: // TODO: cubemaps
            case RG_SKY_TYPE_COLOR:
            default: gu.skyType = SKY_TYPE_COLOR; break;
        }
        gu.skyCubemapIndex = 0;

        if( !Utils::IsAlmostZero( params.skyCubemapRotationTransform ) )
        {
            Utils::SetMatrix3ToGLSLMat4( gu.skyCubemapRotationTransform, params.skyCubemapRotationTransform );
        }

        for( uint32_t i = 0; i < 6; i++ )
        {
            Matrix::GetCubemapViewProjMat(
                &gu.viewProjCubemap[ 16 * i ], i, params.skyViewerPosition.data, cameraNear, cameraFar );
        }
    }

    gu.debugShowFlags = 0;

    {
        const auto& params = pnext::get< RgDrawFrameTexturesParams >( drawInfo );

        gu.normalMapStrength      = params.normalMapStrength;
        gu.emissionMapBoost       = std::max( params.emissionMapBoost, 0.0f );
        gu.emissionMaxScreenColor = std::max( params.emissionMaxScreenColor, 0.0f );
        gu.minRoughness           = std::clamp( params.minRoughness, 0.0f, 1.0f );
        gu.parallaxMaxDepth       = std::max( params.heightMapDepth, 0.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameIlluminationParams >( drawInfo );

        gu.maxBounceShadowsLights     = params.maxBounceShadows;
        gu.polyLightSpotlightFactor   = std::max( 0.0f, params.polygonalLightSpotlightFactor );
        gu.indirSecondBounce          = !!params.enableSecondBounceForIndirect;
        gu.lightIndexIgnoreFPVShadows = lights.GetLightIndexForShaders(
            fi.frameIndex, params.lightUniqueIdIgnoreFirstPersonViewerShadows );
        gu.cellWorldSize              = std::max( params.cellWorldSize, 0.001f );
        gu.gradientMultDiffuse  = std::clamp( params.directDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu.gradientMultIndirect = std::clamp( params.indirectDiffuseSensitivityToChange, 0.0f, 1.0f );
        gu.gradientMultSpecular = std::clamp( params.specularSensitivityToChange, 0.0f, 1.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameBloomParams >( drawInfo );

        gu.bloomThreshold    = std::max( params.inputThreshold, 0.0f );
        gu.bloomIntensity    = 0.2f * std::max( params.bloomIntensity, 0.0f );
        gu.bloomEV           = std::max( params.inputEV, 0.0f );
        gu.lensDirtIntensity = std::max( params.lensDirtIntensity, 0.0f );
    }

    {
        const auto& params = pnext::get< RgDrawFrameReflectRefractParams >( drawInfo );

        switch( params.typeOfMediaAroundCamera )
        {
            case RG_MEDIA_TYPE_WATER: gu.cameraMediaType = MEDIA_TYPE_WATER; break;
            case RG_MEDIA_TYPE_GLASS: gu.cameraMediaType = MEDIA_TYPE_GLASS; break;
            case RG_MEDIA_TYPE_ACID: gu.cameraMediaType = MEDIA_TYPE_ACID; break;
            case RG_MEDIA_TYPE_VACUUM:
            default: gu.cameraMediaType = MEDIA_TYPE_VACUUM; break;
        }

        gu.reflectRefractMaxDepth = std::min( 16u, params.maxReflectRefractDepth );

        gu.indexOfRefractionGlass = std::max( 0.0f, params.indexOfRefractionGlass );
        gu.indexOfRefractionWater = std::max( 0.0f, params.indexOfRefractionWater );
        gu.thinMediaWidth         = std::max( 0.0f, params.thinMediaWidth );

        memcpy( gu.waterColorAndDensity, params.waterColor.data, 3 * sizeof( float ) );
        gu.waterColorAndDensity[ 3 ] = 0.0f;

        memcpy( gu.acidColorAndDensity, params.acidColor.data, 3 * sizeof( float ) );
        gu.acidColorAndDensity[ 3 ] = std::max( 0.0f, params.acidDensity );

        gu.waterWaveSpeed    = params.waterWaveSpeed;
        gu.waterWaveStrength = params.waterWaveNormalStrength;
        gu.waterTextureDerivativesMultiplier =
            std::max( 0.0f, params.waterWaveTextureDerivativesMultiplier );
        gu.waterTextureAreaScale =
            params.waterTextureAreaScale < 0.0001f ? 1.0f : params.waterTextureAreaScale;

        gu.twirlPortalNormal = !!params.portalNormalTwirl;
    }

    gu.rayCullBackFaces  = fi.rayCullBackFaces ? 1 : 0;
    gu.rayLength         = std::clamp( drawInfo.rayLength, 0.1f, float( MAX_RAY_LENGTH ) );
    gu.primaryRayMinDist = std::clamp( cameraNear, 0.001f, gu.rayLength );

    gu.rayCullMaskWorld        = INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_WORLD_1 | INSTANCE_MASK_WORLD_2;
    gu.rayCullMaskWorld_Shadow = INSTANCE_MASK_WORLD_0;

    gu.waterNormalTextureIndex = fi.waterNormalTextureIndex;
    gu.dirtMaskTextureIndex    = fi.dirtMaskTextureIndex;

    gu.cameraRayConeSpreadAngle =
        atanf( ( 2.0f * tanf( ( camera ? camera->fovYRadians : 1.0f ) * 0.5f ) ) / float( fi.renderHeight ) );

    RG_SET_VEC3_A( gu.worldUpVector, fi.worldUp.data );

    gu.lightmapScreenCoverage = fi.lightmapScreenCoverage;
    gu.fluidEnabled           = 0;

    {
        const auto& params = pnext::get< RgDrawFrameVolumetricParams >( drawInfo );

        gu.volumeCameraNear = std::max( cameraNear, 0.001f );
        gu.volumeCameraFar  = std::min( cameraFar, params.volumetricFar );

        if( params.enable )
        {
            gu.volumeEnableType = params.useSimpleDepthBased ? VOLUME_ENABLE_SIMPLE : VOLUME_ENABLE_VOLUMETRIC;
        }
        else
        {
            gu.volumeEnableType = VOLUME_ENABLE_NONE;
        }
        gu.volumeScattering = params.scaterring;
        gu.volumeAsymmetry  = std::clamp( params.assymetry, -1.0f, 1.0f );

        RG_SET_VEC3_A( gu.volumeAmbient, params.ambientColor.data );
        RG_MAX_VEC3( gu.volumeAmbient, 0.0f );

        // ILLUMINATION_VOLUME is 0
        gu.illumVolumeEnable = 0;

        gu.volumeLightSourceIndex = fi.volumeLightSourceIndex;

        RG_SET_VEC3_A( gu.volumeFallbackSrcColor, params.fallbackSourceColor.data );
        RG_MAX_VEC3( gu.volumeFallbackSrcColor, 0.0f );

        RG_SET_VEC3_A( gu.volumeFallbackSrcDirection, params.fallbackSourceDirection.data );

        gu.volumeFallbackSrcExists = Utils::TryNormalize( gu.volumeFallbackSrcDirection ) &&
                                     ( gu.volumeFallbackSrcColor[ 0 ] > 0.01f &&
                                       gu.volumeFallbackSrcColor[ 1 ] > 0.01f &&
                                       gu.volumeFallbackSrcColor[ 2 ] > 0.01f );

        gu.volumeLightMult = std::max( 0.0f, params.lightMultiplier );

        gu.volumeAllowTintUnderwater = params.allowTintUnderwater;
        RG_SET_VEC3_A( gu.volumeUnderwaterColor, params.underwaterColor.data );
        RG_MAX_VEC3( gu.volumeUnderwaterColor, 0.0f );

        if( gu.volumeEnableType != VOLUME_ENABLE_NONE )
        {
            memcpy( gu.volumeViewProj_Prev, gu.volumeViewProj, 16 * sizeof( float ) );
            memcpy( gu.volumeViewProjInv_Prev, gu.volumeViewProjInv, 16 * sizeof( float ) );

            float volumeproj[ 16 ];
            Matrix::MakeProjectionMatrix( volumeproj,
                                          camera ? camera->aspect : 1.0f,
                                          camera ? camera->fovYRadians : 1.0f,
                                          gu.volumeCameraNear,
                                          gu.volumeCameraFar );

            Matrix::Multiply( gu.volumeViewProj, gu.view, volumeproj );
            Matrix::Inverse( gu.volumeViewProjInv, gu.volumeViewProj );
        }
    }

    gu.antiFireflyEnabled = true;

    gu.hdrDisplay = fi.hdr ? HDR_DISPLAY_LINEAR : HDR_DISPLAY_NONE;
}

}

namespace
{

struct RenderResolution
{
    uint32_t width;
    uint32_t height;
    bool     temporalUpscaling;
};

// The same as RenderResolutionHelper::Setup.
// RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 is done by the MetalFX temporal scaler.
RenderResolution CalculateRenderResolution( const RgStartFrameRenderResolutionParams& params,
                                            uint32_t                                  windowWidth,
                                            uint32_t                                  windowHeight,
                                            bool                                      metalFXAvailable )
{
    uint32_t w = windowWidth, h = windowHeight;

    const bool temporal = params.upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 && metalFXAvailable;

    if( params.resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM )
    {
        if( params.customRenderSize.width > 0 && params.customRenderSize.height > 0 )
        {
            w = std::min( params.customRenderSize.width, windowWidth );
            h = std::min( params.customRenderSize.height, windowHeight );
        }
    }
    else if( temporal )
    {
        float div = 1.7f;
        switch( params.resolutionMode )
        {
            case RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE: div = 3.0f; break;
            case RG_RENDER_RESOLUTION_MODE_PERFORMANCE: div = 2.0f; break;
            case RG_RENDER_RESOLUTION_MODE_BALANCED: div = 1.7f; break;
            case RG_RENDER_RESOLUTION_MODE_QUALITY: div = 1.5f; break;
            case RG_RENDER_RESOLUTION_MODE_NATIVE_AA: div = 1.0f; break;
            default: break;
        }
        w = uint32_t( float( windowWidth ) / div );
        h = uint32_t( float( windowHeight ) / div );
    }

    // RTGL1 needs an even render width, for checkerboarding
    w = std::max( 2u, w + w % 2 );
    h = std::max( 1u, h );
    return RenderResolution{ w, h, temporal };
}

}

struct RTGL1::MetalDevice::Impl
{
    id< MTLDevice >       device{ nil };
    id< MTLCommandQueue > queue{ nil };
    CAMetalLayer*         layer{ nil };

    // limits the CPU to MAX_FRAMES_IN_FLIGHT frames ahead of the GPU
    dispatch_semaphore_t frameSemaphore{ nil };
    bool                 frameStarted{ false };
    uint32_t             frameIndex{ 0 };

    bool hardwareRayTracing{ false };
    bool metalFXAvailable{ false };
    // MTLFXTemporalDenoisedScaler: RG_DENOISER_METALFX
    bool metalFXDenoiserAvailable{ false };
    float metalFXDenoiserMinScale{ 1.0f }, metalFXDenoiserMaxScale{ 1.0f };
    // MTLFXFrameInterpolator: RgFrameGenerationMode, with the MetalFX temporal upscaling
    bool frameInterpolationAvailable{ false };
    bool effectWipeIsUsed{ false };
    // RgStartFrameInfo::lightmapScreenCoverage, for the classic mode
    float lightmapScreenCoverage{ 0 };

    PFN_rgPrint pfnPrint{ nullptr };
    void*       pUserPrintData{ nullptr };

    id< MTLLibrary >               library{ nil };
    id< MTLComputePipelineState >  skyOnly{ nil };
    // [0] SDR: BGRA8 sRGB, [1] EDR: RGBA16Float extended linear sRGB
    id< MTLRenderPipelineState >   blit[ 2 ]{};
    bool                           hdr{ false };

    std::unique_ptr< MetalTextureManager > textures;
    std::unique_ptr< MetalScene >          scene;
    std::unique_ptr< TextureMetaManager >  textureMeta;
    std::unique_ptr< SceneMetaManager >    sceneMeta;
    std::unique_ptr< MetalSceneManager >   sceneManager;
    // GltfExporter::MakeLightsForPrimitiveDynamic
    std::vector< PositionNormal >          tempStorageInit;
    std::vector< AnyLightEXT >             tempStorageLights;
    std::unique_ptr< MetalRasterizer >     rasterizer;
    std::unique_ptr< MetalLightManager >   lights;

    id< MTLBuffer >  globalUniform[ MAX_FRAMES_IN_FLIGHT ]{};
    id< MTLTexture > outColor{ nil };

    std::optional< RgCameraInfo > camera;

    std::filesystem::path ovrdFolder;

    RgStartFrameRenderResolutionParams resolution{};

    // RTGL1's GLSL shaders (bin/shaders/), if present
    std::unique_ptr< MetalComputePipelines > rtShaders;
    std::unique_ptr< MetalFramebuffers >     framebuffers;
    std::unique_ptr< MetalPathTracer >       pathTracer;
    // RgStartFrameFluidParams::enabled
    std::unique_ptr< MetalFluid >            fluid;
    RgFloat3D                                fluidGravity{ 0, -9.8f, 0 };
    RgFloat3D                                fluidColor{ 1, 1, 1 };
    MetalBindingTable                        bindings;
    id< MTLTexture >                         blueNoise{ nil };
    id< MTLSamplerState >                    linearClampSampler{ nil };

    ShGlobalUniform uniform{};
    uint32_t        frameId{ 0 };
    double          previousFrameTime{ 0 };
    RgFloat3D       worldUp{ 0, 1, 0 };
    bool            rayCullBackFaces{ false };
    uint32_t        primaryRaysMaxAlbedoLayers{ 1 };
    uint32_t        indirectIlluminationMaxAlbedoLayers{ 1 };
    uint32_t        lightmapLayerIndex{ 0 };

    uint64_t                        frameCounter{ 0 };
    std::optional< CaptureRequest > capture{ GetCaptureRequest() };
    // RgCameraInfo::pView is valid only during rgUploadCamera, so it's copied here
    float cameraView[ 16 ]{};

    // Frame generation: MetalFX frame interpolator (id< MTLFXFrameInterpolator >),
    // generates a frame between the previous and the current ones, that is presented before the current one
    struct FrameInterpolation
    {
        id               interpolator{ nil };
        uint32_t         inputWidth{ 0 }, inputHeight{ 0 };
        uint32_t         outputWidth{ 0 }, outputHeight{ 0 };
        MTLPixelFormat   format{ MTLPixelFormatInvalid };
        // the current and the previous color, without HUD
        id< MTLTexture > color[ 2 ]{};
        // the actual frames to present, with HUD
        id< MTLTexture > present[ 2 ]{};
        uint32_t         current{ 0 };
        bool             hasHistory{ false };
        // HUD only, the interpolator puts it on top of the generated frame:
        // 'uiCoverage' is rasterized, 'ui' is made by FrameInterpolationUI
        id< MTLTexture > uiCoverage{ nil };
        id< MTLTexture > ui{ nil };
        id< MTLComputePipelineState > uiPipeline{ nil };
        // per 'current', as they are presented after the next frame might have been started
        id< MTLTexture > output[ 2 ]{};

        // presentation is asynchronous to rendering: as each frame needs 2 drawables,
        // acquiring them while rendering would serialize the CPU and the GPU
        id< MTLCommandQueue > presentQueue{ nil };
        dispatch_queue_t      presentDispatch{ nil };

        bool wasActive{ false };
    } fg;
    id PrepareFrameInterpolator( uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, MTLPixelFormat format );
    // Commits 'cmd'; after it's completed, presents 'generated' (if not null), then 'real'.
    // 'capturePath': to save 'real', as presented
    void PresentAsync( id< MTLCommandBuffer > cmd,
                       id< MTLTexture >       real,
                       id< MTLTexture >       generated,
                       std::optional< std::string > capturePath,
                       bool                         captureGenerated = false );
    // GPU time of a frame (smoothed), to pace the generated and the actual frames:
    // not the CPU frame time, as it includes waiting for drawables, which depends on the pacing itself
    std::shared_ptr< std::atomic< double > > gpuFrameTime{ std::make_shared< std::atomic< double > >( 1.0 / 60.0 ) };

    // RG_VSYNC_HALF_RATE: present each frame for 2 refresh intervals, for even frame times
    // if the frame rate is lower than the display's (instead of alternating, e.g. 16.7 / 33.3 ms at 40 FPS on 60 Hz).
    // Not chosen automatically by the measured GPU time, as it depends on the GPU's clock, which depends on the pacing.
    bool   vsyncHalfRate{ false };
    double RefreshInterval() const
    {
        double hz = 0;
        if( CGDisplayModeRef mode = CGDisplayCopyDisplayMode( CGMainDisplayID() ) )
        {
            hz = CGDisplayModeGetRefreshRate( mode );
            CGDisplayModeRelease( mode );
        }
        return 1.0 / ( hz > 1 ? hz : 60.0 );
    }
    double PacedInterval() const
    {
        // a bit less than 2 intervals, so the next refresh after it is the second one
        return vsyncHalfRate && layer.displaySyncEnabled ? RefreshInterval() * 2.0 - 0.001 : 0.0;
    }
    bool frameInterpolationWarned{ false };
};

void RTGL1::MetalDevice::Impl::PresentAsync( id< MTLCommandBuffer >       cmd,
                                             id< MTLTexture >             real,
                                             id< MTLTexture >             generated,
                                             std::optional< std::string > capturePath,
                                             bool                         captureGenerated )
{
    id< MTLRenderPipelineState > pipeline     = blit[ hdr ? 1 : 0 ];
    CAMetalLayer*                metalLayer   = layer;
    id< MTLCommandQueue >        q            = fg.presentQueue;
    dispatch_queue_t             dq           = fg.presentDispatch;
    dispatch_semaphore_t         sem          = frameSemaphore;
    auto                         frameTime    = gpuFrameTime;
    id< MTLDevice >              dev          = device;

    [cmd addCompletedHandler:^( id< MTLCommandBuffer > c ) {
      OnFrameCompleted( c, *frameTime );

      dispatch_async( dq, ^{
        id< MTLCommandBuffer > pc = [q commandBuffer];
        pc.label                  = @"RTGL1 present";

        auto l_blit = [ & ]( id< CAMetalDrawable > drawable, id< MTLTexture > src, NSString* label ) {
            MTLRenderPassDescriptor* pass          = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[ 0 ].texture     = drawable.texture;
            pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionDontCare;
            pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;

            id< MTLRenderCommandEncoder > enc = [pc renderCommandEncoderWithDescriptor:pass];
            enc.label                         = label;
            [enc setRenderPipelineState:pipeline];
            [enc setFragmentTexture:src atIndex:0];
            const float scale = 1.0f;
            [enc setFragmentBytes:&scale length:sizeof( scale ) atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [enc endEncoding];
        };

        id< MTLBuffer > captureBuffer = nil;
        uint32_t        cw = 0, ch = 0;

        auto l_capture = [ & ]( id< CAMetalDrawable > d ) {
            cw            = uint32_t( d.texture.width );
            ch            = uint32_t( d.texture.height );
            captureBuffer = [dev newBufferWithLength:size_t( cw ) * ch * 4 options:MTLResourceStorageModeShared];

            id< MTLBlitCommandEncoder > b = [pc blitCommandEncoder];
            b.label                       = @"Capture";
            [b copyFromTexture:d.texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake( 0, 0, 0 )
                          sourceSize:MTLSizeMake( cw, ch, 1 )
                            toBuffer:captureBuffer
                   destinationOffset:0
              destinationBytesPerRow:cw * 4
            destinationBytesPerImage:size_t( cw ) * ch * 4];
            [b endEncoding];
        };

        bool generatedPresented = false;
        if( generated )
        {
            if( id< CAMetalDrawable > d = [metalLayer nextDrawable] )
            {
                l_blit( d, generated, @"Blit generated frame to swapchain" );
                if( capturePath && captureGenerated )
                {
                    l_capture( d );
                }
                [pc presentDrawable:d];
                generatedPresented = true;
            }
        }

        if( id< CAMetalDrawable > d = [metalLayer nextDrawable] )
        {
            l_blit( d, real, @"Blit to swapchain" );
            if( capturePath && !captureBuffer )
            {
                l_capture( d );
            }

            if( generatedPresented )
            {
                // pace: the generated frame is shown for a half of the frame time
                [pc presentDrawable:d afterMinimumDuration:std::clamp( frameTime->load() * 0.5, 0.0, 0.05 )];
            }
            else
            {
                [pc presentDrawable:d];
            }
        }

        // the textures of this frame can be reused, only after they were presented
        [pc addCompletedHandler:^( id< MTLCommandBuffer > ) {
          dispatch_semaphore_signal( sem );
        }];
        [pc commit];

        if( captureBuffer )
        {
            [pc waitUntilCompleted];
            SaveCapture( captureBuffer, cw, ch, *capturePath );
        }
      } );
    }];
    [cmd commit];
}

id RTGL1::MetalDevice::Impl::PrepareFrameInterpolator( uint32_t       inputWidth,
                                                       uint32_t       inputHeight,
                                                       uint32_t       outputWidth,
                                                       uint32_t       outputHeight,
                                                       MTLPixelFormat format )
{
    if( fg.interpolator && fg.inputWidth == inputWidth && fg.inputHeight == inputHeight &&
        fg.outputWidth == outputWidth && fg.outputHeight == outputHeight && fg.format == format )
    {
        return fg.interpolator;
    }

    if( @available( macOS 26.0, * ) )
    {
        MTLFXFrameInterpolatorDescriptor* d = [MTLFXFrameInterpolatorDescriptor new];
        d.colorTextureFormat                = format;
        d.outputTextureFormat               = format;
        d.uiTextureFormat                   = format;
        d.depthTextureFormat                = MTLPixelFormatR32Float;
        d.motionTextureFormat               = MTLPixelFormatRG16Float;
        d.inputWidth                        = inputWidth;
        d.inputHeight                       = inputHeight;
        d.outputWidth                       = outputWidth;
        d.outputHeight                      = outputHeight;

        id< MTLFXFrameInterpolator > fi = [d newFrameInterpolatorWithDevice:device];
        if( !fi )
        {
            debug::Warning( "Metal backend: can't create MetalFX frame interpolator" );
            return nil;
        }

        auto l_make = [ & ]( MTLTextureUsage usage, NSString* label ) {
            MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                         width:outputWidth
                                                                                        height:outputHeight
                                                                                     mipmapped:NO];
            t.usage                 = usage | MTLTextureUsageShaderRead;
            t.storageMode           = MTLStorageModePrivate;
            id< MTLTexture > tex    = [device newTextureWithDescriptor:t];
            tex.label               = label;
            return tex;
        };
        fg.color[ 0 ] = l_make( fi.colorTextureUsage, @"Frame interpolation - Color A" );
        fg.color[ 1 ] = l_make( fi.colorTextureUsage, @"Frame interpolation - Color B" );
        fg.present[ 0 ] = l_make( 0, @"Frame interpolation - Present A" );
        fg.present[ 1 ] = l_make( 0, @"Frame interpolation - Present B" );
        fg.ui          = l_make( fi.uiTextureUsage | MTLTextureUsageShaderWrite, @"Frame interpolation - UI" );
        fg.uiCoverage  = l_make( MTLTextureUsageRenderTarget, @"Frame interpolation - UI coverage" );
        if( !fg.uiPipeline )
        {
            NSError* error = nil;
            fg.uiPipeline  = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"FrameInterpolationUI"]
                                                                  error:&error];
            if( !fg.uiPipeline )
            {
                debug::Warning( "Metal backend: FrameInterpolationUI: {}",
                                error ? error.localizedDescription.UTF8String : "" );
                return nil;
            }
        }
        fg.output[ 0 ] = l_make( fi.outputTextureUsage, @"Frame interpolation - Output A" );
        fg.output[ 1 ] = l_make( fi.outputTextureUsage, @"Frame interpolation - Output B" );
        if( !fg.presentQueue )
        {
            fg.presentQueue       = [device newCommandQueue];
            fg.presentQueue.label = @"RTGL1 present";
            fg.presentDispatch    = dispatch_queue_create( "RTGL1 present", DISPATCH_QUEUE_SERIAL );
        }

        fg.interpolator = fi;
        fg.inputWidth   = inputWidth;
        fg.inputHeight  = inputHeight;
        fg.outputWidth  = outputWidth;
        fg.outputHeight = outputHeight;
        fg.format       = format;
        fg.hasHistory   = false;

        debug::Info( "Metal backend: MetalFX frame interpolation {}x{} -> {}x{}",
                     inputWidth,
                     inputHeight,
                     outputWidth,
                     outputHeight );
        return fg.interpolator;
    }
    return nil;
}

RTGL1::MetalDevice::MetalDevice( const RgInstanceCreateInfo* pInfo )
    : impl{ std::make_unique< Impl >() }
{
    if( !pInfo->pMetalSurfaceCreateInfo || !pInfo->pMetalSurfaceCreateInfo->pLayer )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT,
                           "Metal backend requires pMetalSurfaceCreateInfo with a CAMetalLayer" );
    }

    impl->pfnPrint       = pInfo->pfnPrint;
    impl->pUserPrintData = pInfo->pUserPrintData;

    impl->ovrdFolder = Utils::SafeCstr( pInfo->pOverrideFolderPath );
    g_ovrdFolder     = impl->ovrdFolder;
    g_libConfig      = json_parser::ReadFileAs< LibraryConfig >( impl->ovrdFolder / "RTGL1.json" ).value_or( LibraryConfig{} );

    impl->device = MTLCreateSystemDefaultDevice();
    if( !impl->device )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR, "MTLCreateSystemDefaultDevice failed" );
    }

    if( !impl->device.supportsRaytracing )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR,
                           "Metal device doesn't support ray tracing" );
    }
    // Apple9 (M3 and later) traverses acceleration structures in hardware,
    // older Apple Silicon runs the same Metal ray tracing API in software
    impl->hardwareRayTracing = [impl->device supportsFamily:MTLGPUFamilyApple9];
    impl->metalFXAvailable   = [MTLFXTemporalScalerDescriptor supportsDevice:impl->device];
    impl->metalFXDenoiserAvailable = impl->metalFXAvailable && MetalPathTracer::IsMetalFXDenoiserSupported( impl->device );
    if( @available( macOS 26.0, * ) )
    {
        if( impl->metalFXDenoiserAvailable )
        {
            impl->metalFXDenoiserMinScale =
                [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMinScaleForDevice:impl->device];
            impl->metalFXDenoiserMaxScale =
                [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMaxScaleForDevice:impl->device];
        }
        impl->frameInterpolationAvailable =
            impl->metalFXAvailable && [MTLFXFrameInterpolatorDescriptor supportsDevice:impl->device];
    }

    impl->queue       = [impl->device newCommandQueue];
    impl->queue.label = @"RTGL1";

    impl->layer = const_cast< CAMetalLayer* >( pInfo->pMetalSurfaceCreateInfo->pLayer );
    impl->layer.device             = impl->device;
    impl->layer.pixelFormat        = MTLPixelFormatBGRA8Unorm_sRGB;
    // capture reads the swapchain image back
    impl->layer.framebufferOnly    = !impl->capture;
    impl->layer.maximumDrawableCount = 3;

    impl->frameSemaphore = dispatch_semaphore_create( MAX_FRAMES_IN_FLIGHT );

    // Debug: RTGL1_METAL_HUD=1 shows the Metal Performance HUD (the same as MTL_HUD_ENABLED=1),
    // and logs its statistics to the unified log (/usr/bin/log stream --predicate 'process == "<app>"')
    if( const char* e = std::getenv( "RTGL1_METAL_HUD" ); e && e[ 0 ] == '1' )
    {
        impl->layer.developerHUDProperties = @{ @"mode" : @"default", @"logging" : @"default" };
    }

    impl->library = LoadShaderLibrary( impl->device );
    {
        NSError* error    = nil;
        impl->skyOnly = [impl->device newComputePipelineStateWithFunction:GetFunction( impl->library, @"SkyOnly" )
                                                                    error:&error];
        CheckPipeline( impl->skyOnly, error, "SkyOnly" );
    }
    for( int i = 0; i < 2; i++ )
    {
        auto* desc                             = [MTLRenderPipelineDescriptor new];
        desc.label                             = @"Blit";
        desc.vertexFunction                    = GetFunction( impl->library, @"BlitVertex" );
        desc.fragmentFunction                  = GetFunction( impl->library, @"BlitFragment" );
        desc.colorAttachments[ 0 ].pixelFormat = i == 0 ? MTLPixelFormatBGRA8Unorm_sRGB : MTLPixelFormatRGBA16Float;

        NSError* error  = nil;
        impl->blit[ i ] = [impl->device newRenderPipelineStateWithDescriptor:desc error:&error];
        CheckPipeline( impl->blit[ i ], error, "Blit" );
    }

    impl->textures   = std::make_unique< MetalTextureManager >( impl->device,
                                                          impl->queue,
                                                          impl->ovrdFolder,
                                                          pInfo->pbrTextureSwizzling,
                                                          !!pInfo->textureSamplerForceMinificationFilterLinear,
                                                          !!pInfo->textureSamplerForceNormalMapFilterLinear );
    impl->scene       = std::make_unique< MetalScene >( impl->device, impl->queue );
    impl->lights      = std::make_unique< MetalLightManager >( impl->device );
    impl->textureMeta = std::make_unique< TextureMetaManager >( impl->ovrdFolder / DATABASE_FOLDER );
    impl->sceneMeta   = std::make_unique< SceneMetaManager >( impl->ovrdFolder / DATABASE_FOLDER / "scenes.json" );
    impl->sceneManager = std::make_unique< MetalSceneManager >(
        *pInfo, impl->ovrdFolder, *impl->scene, *impl->textures, *impl->lights, *impl->textureMeta );
    impl->rasterizer = std::make_unique< MetalRasterizer >( impl->device,
                                                            impl->library,
                                                            impl->layer.pixelFormat,
                                                            !!pInfo->rasterizedVertexColorGamma,
                                                            pInfo->rasterizedSkyCubemapSize );

    for( auto& b : impl->globalUniform )
    {
        b       = [impl->device newBufferWithLength:sizeof( ShGlobalUniform )
                                      options:MTLResourceStorageModeShared];
        b.label = @"Global uniform";
    }

    impl->effectWipeIsUsed           = !!pInfo->effectWipeIsUsed;
    impl->worldUp                    = pInfo->worldUp;
    impl->rayCullBackFaces           = !!pInfo->rayCullBackFacingTriangles;
    impl->primaryRaysMaxAlbedoLayers          = pInfo->primaryRaysMaxAlbedoLayers;
    impl->indirectIlluminationMaxAlbedoLayers = pInfo->indirectIlluminationMaxAlbedoLayers;
    // the same as in RayTracingPipeline.cpp
    impl->lightmapLayerIndex = pInfo->lightmapTexCoordLayerIndex > 0 &&
                                       pInfo->lightmapTexCoordLayerIndex < MATERIALS_MAX_LAYER_COUNT
                                   ? pInfo->lightmapTexCoordLayerIndex
                                   : 0xFF;

    if( const auto rtShadersFolder = GetLibraryFolder() / "shaders"; std::filesystem::exists( rtShadersFolder ) )
    {
        impl->rtShaders    = std::make_unique< MetalComputePipelines >( impl->device, rtShadersFolder );
        impl->framebuffers = std::make_unique< MetalFramebuffers >( impl->device );
        impl->pathTracer   = std::make_unique< MetalPathTracer >( impl->device, impl->library, *impl->rtShaders );

        {
            MTLSamplerDescriptor* d  = [MTLSamplerDescriptor new];
            d.minFilter              = MTLSamplerMinMagFilterLinear;
            d.magFilter              = MTLSamplerMinMagFilterLinear;
            d.sAddressMode           = MTLSamplerAddressModeClampToEdge;
            d.tAddressMode           = MTLSamplerAddressModeClampToEdge;
            d.rAddressMode           = MTLSamplerAddressModeClampToEdge;
            d.supportArgumentBuffers = YES;
            impl->linearClampSampler = [impl->device newSamplerStateWithDescriptor:d];
        }

        // the same as BlueNoise.cpp
        {
            MTLTextureDescriptor* d = [MTLTextureDescriptor new];
            d.textureType           = MTLTextureType2DArray;
            d.pixelFormat           = MTLPixelFormatRGBA8Unorm;
            d.width                 = BLUE_NOISE_TEXTURE_SIZE;
            d.height                = BLUE_NOISE_TEXTURE_SIZE;
            d.arrayLength           = BLUE_NOISE_TEXTURE_COUNT;
            d.usage                 = MTLTextureUsageShaderRead;
            d.storageMode           = MTLStorageModeShared;
            impl->blueNoise         = [impl->device newTextureWithDescriptor:d];
            impl->blueNoise.label   = @"Blue noise";

            const auto path   = impl->ovrdFolder / "BlueNoise_LDR_RGBA_128.ktx2";
            auto       loader = ImageLoader{};
            auto       loaded = loader.LoadLayered( path );

            if( !loaded || loaded->baseSize.width != BLUE_NOISE_TEXTURE_SIZE ||
                loaded->baseSize.height != BLUE_NOISE_TEXTURE_SIZE ||
                loaded->layerData.size() != BLUE_NOISE_TEXTURE_COUNT ||
                loaded->format != VK_FORMAT_R8G8B8A8_UNORM )
            {
                loader.FreeLoaded();
                throw RgException( RG_RESULT_ERROR_CANT_FIND_HARDCODED_RESOURCES,
                                   "Can't load blue noise: " + path.string() +
                                       ", it must be R8G8B8A8_UNORM with " +
                                       std::to_string( BLUE_NOISE_TEXTURE_COUNT ) + " layers of " +
                                       std::to_string( BLUE_NOISE_TEXTURE_SIZE ) + "x" +
                                       std::to_string( BLUE_NOISE_TEXTURE_SIZE ) );
            }

            for( uint32_t layer = 0; layer < BLUE_NOISE_TEXTURE_COUNT; layer++ )
            {
                [impl->blueNoise replaceRegion:MTLRegionMake2D( 0, 0, BLUE_NOISE_TEXTURE_SIZE, BLUE_NOISE_TEXTURE_SIZE )
                                   mipmapLevel:0
                                         slice:layer
                                     withBytes:loaded->layerData[ layer ]
                                   bytesPerRow:BLUE_NOISE_TEXTURE_SIZE * 4
                                 bytesPerImage:0];
            }
            loader.FreeLoaded();
        }


        debug::Info( "Metal backend: using RTGL1 shaders from {}", rtShadersFolder.string() );
    }

    debug::Info( "Metal device: {}", impl->device.name.UTF8String );
    debug::Info( "Metal ray tracing: {}",
                 impl->hardwareRayTracing ? "hardware" : "software (no RT hardware)" );
}

RTGL1::MetalDevice::~MetalDevice()
{
    // wait for the frames in flight, so their completion handlers don't outlive the device
    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        dispatch_semaphore_wait( impl->frameSemaphore, DISPATCH_TIME_FOREVER );
    }
    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        dispatch_semaphore_signal( impl->frameSemaphore );
    }
}

void RTGL1::MetalDevice::StartFrame( const RgStartFrameInfo* pInfo )
{
    if( !pInfo )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( impl->frameStarted )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_ENDED );
    }

    dispatch_semaphore_wait( impl->frameSemaphore, DISPATCH_TIME_FOREVER );
    impl->frameStarted = true;

    // follow the window size, so the application doesn't need to resize the drawable
    CGSize size  = impl->layer.bounds.size;
    CGFloat scale = impl->layer.contentsScale;
    CGSize drawableSize = CGSizeMake( std::max( 1.0, size.width * scale ),
                                      std::max( 1.0, size.height * scale ) );
    if( !CGSizeEqualToSize( impl->layer.drawableSize, drawableSize ) )
    {
        impl->layer.drawableSize = drawableSize;
    }

    impl->layer.displaySyncEnabled = !!pInfo->vsync;
    impl->vsyncHalfRate            = pInfo->vsync == RG_VSYNC_HALF_RATE;

    // HDR output as Vulkan's VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT swapchain: macOS EDR
    if( impl->hdr != !!pInfo->hdr )
    {
        impl->hdr = !!pInfo->hdr;
        if( impl->hdr )
        {
            CGColorSpaceRef space                       = CGColorSpaceCreateWithName( kCGColorSpaceExtendedLinearSRGB );
            impl->layer.pixelFormat                     = MTLPixelFormatRGBA16Float;
            impl->layer.colorspace                      = space;
            impl->layer.wantsExtendedDynamicRangeContent = YES;
            CGColorSpaceRelease( space );
        }
        else
        {
            impl->layer.pixelFormat                     = MTLPixelFormatBGRA8Unorm_sRGB;
            impl->layer.colorspace                      = nil;
            impl->layer.wantsExtendedDynamicRangeContent = NO;
        }
    }

    impl->lightmapScreenCoverage = pInfo->lightmapScreenCoverage < 0.01f   ? 0.0f
                                   : pInfo->lightmapScreenCoverage > 0.99f ? 1.0f
                                                                           : pInfo->lightmapScreenCoverage;

    impl->resolution       = pnext::get< RgStartFrameRenderResolutionParams >( *pInfo );

    // the same as VulkanDevice::StartFrame
    {
        const auto& fluidInfo = pnext::get< RgStartFrameFluidParams >( *pInfo );
        if( fluidInfo.enabled && !impl->fluid && impl->rtShaders )
        {
            impl->fluid = std::make_unique< MetalFluid >(
                impl->device, impl->library, fluidInfo.particleBudget, fluidInfo.particleRadius );
        }
        else if( !fluidInfo.enabled && impl->fluid )
        {
            impl->fluid.reset();
        }
        impl->fluidGravity = fluidInfo.gravity;
        impl->fluidColor   = fluidInfo.color;
        if( impl->fluid )
        {
            impl->fluid->PrepareForFrame( fluidInfo.reset );
        }
    }
    impl->resolution.pNext = nullptr;

    impl->textures->OnFrameStart();
    impl->scene->BeginFrame( impl->frameIndex );
    impl->rasterizer->BeginFrame( impl->frameIndex );
    impl->lights->PrepareForFrame( impl->frameIndex );
    impl->lights->SetLightstyles( *pInfo );

    impl->sceneManager->PrepareForFrame( impl->frameIndex, *pInfo );
    impl->sceneManager->SubmitStaticLights(
        impl->frameIndex,
        // SHIPPING_HACK
        impl->uniform.volumeAllowTintUnderwater && impl->uniform.cameraMediaType == RG_MEDIA_TYPE_WATER,
        Utils::PackColorFromFloat( impl->uniform.volumeUnderwaterColor ) );
}

void RTGL1::MetalDevice::DrawFrame( const RgDrawFrameInfo* pInfo )
{
    if( !pInfo )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( !impl->frameStarted )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_STARTED );
    }

    // the same as VulkanDevice::DrawFrame: a scene can modify some parameters
    auto modified            = RgDrawFrameInfo{ *pInfo };
    auto modified_Volumetric = pnext::get< RgDrawFrameVolumetricParams >( *pInfo );
    auto modified_Sky        = pnext::get< RgDrawFrameSkyParams >( *pInfo );
    {
        impl->sceneMeta->Modify( impl->sceneManager->GetImportMapName(), modified_Volumetric, modified_Sky );

        modified_Volumetric.pNext = modified.pNext;
        modified_Sky.pNext        = &modified_Volumetric;
        modified.pNext            = &modified_Sky;
        pInfo                     = &modified;
    }

    @autoreleasepool
    {
        id< MTLCommandBuffer > cmd = [impl->queue commandBuffer];
        cmd.label                  = @"RTGL1 frame";

        const CGSize   drawableSize = impl->layer.drawableSize;
        const uint32_t width        = uint32_t( drawableSize.width );
        const uint32_t height       = uint32_t( drawableSize.height );

        if( !impl->outColor || impl->outColor.width != width || impl->outColor.height != height )
        {
            MTLTextureDescriptor* desc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                   width:width
                                                                  height:height
                                                               mipmapped:NO];
            desc.usage         = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            desc.storageMode   = MTLStorageModePrivate;
            impl->outColor       = [impl->device newTextureWithDescriptor:desc];
            impl->outColor.label = @"Out color";
        }

        impl->textures->SetDynamicSamplerFilter(
            pnext::get< RgDrawFrameTexturesParams >( *pInfo ).dynamicSamplerFilter );

        if( impl->rtShaders )
        {
            impl->rtShaders->BeginFrame( impl->frameIndex );
        }

        // Debug: with RTGL1_METAL_PROFILE=1, the acceleration structures are built
        // in a separate command buffer, to print its GPU time
        static const bool profile = [] {
            const char* e = std::getenv( "RTGL1_METAL_PROFILE" );
            return e && e[ 0 ] == '1';
        }();
        id< MTLCommandBuffer > cmdScene = profile ? [impl->queue commandBuffer] : cmd;

        const CGSize   drawableSizeForAspect = impl->layer.drawableSize;
        const auto     camera                = impl->sceneManager->GetCamera(
            impl->camera, float( drawableSizeForAspect.width / std::max( 1.0, drawableSizeForAspect.height ) ) );

        const bool hasScene = camera && !pInfo->disableRayTracedGeometry &&
                              impl->scene->Build( cmdScene,
                                                  INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_WORLD_1 | INSTANCE_MASK_WORLD_2 );

        if( profile )
        {
            const uint64_t frame = impl->frameCounter;
            [cmdScene addCompletedHandler:^( id< MTLCommandBuffer > c ) {
              if( frame % 120 == 0 )
              {
                  debug::Info( "Metal profile: {:>8.3f} ms  Acceleration structures",
                               ( c.GPUEndTime - c.GPUStartTime ) * 1000.0 );
              }
            }];
            [cmdScene commit];
            [cmd addCompletedHandler:^( id< MTLCommandBuffer > c ) {
              // frames in flight overlap on the GPU, so the time between the frames' ends is the frame time
              static CFTimeInterval prevEnd = 0, sum = 0;
              static uint32_t       count   = 0;
              if( prevEnd > 0 )
              {
                  sum += c.GPUEndTime - prevEnd;
                  count++;
              }
              prevEnd = c.GPUEndTime;
              if( frame % 120 == 0 && count > 0 )
              {
                  debug::Info( "Metal profile: {:>8.3f} ms  GPU frame time (average)", sum / count * 1000.0 );
                  sum   = 0;
                  count = 0;
              }
            }];
        }
        const bool rtPath = impl->rtShaders && hasScene;

        const auto res = rtPath ? CalculateRenderResolution( impl->resolution, width, height, impl->metalFXAvailable )
                                : RenderResolution{ width, height, false };
        const uint32_t renderWidth  = res.width;
        const uint32_t renderHeight = res.height;

        // RG_DENOISER_METALFX: denoises and upscales, so only with the MetalFX upscaling
        bool metalFXDenoiser = res.temporalUpscaling && impl->metalFXDenoiserAvailable &&
                               impl->resolution.denoiser == RG_DENOISER_METALFX;
        if( metalFXDenoiser )
        {
            const float scale = float( width ) / float( renderWidth );
            if( scale < impl->metalFXDenoiserMinScale - 0.01f || scale > impl->metalFXDenoiserMaxScale + 0.01f )
            {
                static bool warned = false;
                if( !warned )
                {
                    warned = true;
                    debug::Warning( "Metal backend: MetalFX denoiser supports scale in [{}, {}], but requested {}. "
                                    "Using the default denoiser",
                                    impl->metalFXDenoiserMinScale,
                                    impl->metalFXDenoiserMaxScale,
                                    scale );
                }
                metalFXDenoiser = false;
            }
        }

        // RgFrameGenerationMode: MetalFX frame interpolation, only with the MetalFX upscaling
        // Debug: RTGL1_METAL_DEBUG_FG_OFF_AT=<frame>, to test switching it off
        const char* fgOffAt          = std::getenv( "RTGL1_METAL_DEBUG_FG_OFF_AT" );
        const bool  debugFgOff       = fgOffAt && impl->frameCounter >= std::strtoull( fgOffAt, nullptr, 10 );
        const bool frameInterpolation = res.temporalUpscaling && impl->frameInterpolationAvailable &&
                                        impl->resolution.frameGeneration != RG_FRAME_GENERATION_MODE_OFF &&
                                        !pInfo->presentPrevFrame && !debugFgOff;
        if( !frameInterpolation )
        {
            impl->fg.hasHistory = false;
        }
        if( frameInterpolation != impl->fg.wasActive )
        {
            impl->fg.wasActive = frameInterpolation;
            debug::Info( "Metal backend: MetalFX frame interpolation {}", frameInterpolation ? "on" : "off" );
            // (not when only paused, e.g. presentPrevFrame while loading)
            if( !frameInterpolation && ( impl->resolution.frameGeneration == RG_FRAME_GENERATION_MODE_OFF ||
                                         !res.temporalUpscaling || debugFgOff ) )
            {
                // release, so it doesn't stay active (e.g. the Metal HUD reports it while it exists);
                // the pending presentation retains its own textures
                impl->fg.interpolator = nil;
                impl->fg.color[ 0 ] = impl->fg.color[ 1 ] = nil;
                impl->fg.present[ 0 ] = impl->fg.present[ 1 ] = nil;
                impl->fg.output[ 0 ] = impl->fg.output[ 1 ] = nil;
                impl->fg.ui = nil;
                impl->fg.uiCoverage = nil;
                impl->fg.inputWidth = impl->fg.inputHeight = impl->fg.outputWidth = impl->fg.outputHeight = 0;
            }
        }
        // the generated frame to present before the current one
        id< MTLTexture > generated = nil;
        // if not null, present asynchronously: this (a copy of the current frame), after 'generated'
        id< MTLTexture > asyncPresentReal = nil;

        // RenderResolutionHelper::GetMipLodBias: sharper textures, if upscaling
        impl->textures->SetMipLodBias(
            res.temporalUpscaling ? std::log2( std::max( 0.01f, float( renderWidth ) / float( width ) ) ) - 1.0f
                                  : 0.0f );

        // global uniform
        {
            ShGlobalUniform& gu = impl->uniform;
            FillUniform( gu,
                         *pInfo,
                         camera,
                         *impl->lights,
                         UniformFrameInfo{
                             .renderWidth      = renderWidth,
                             .renderHeight     = renderHeight,
                             .upscaledWidth    = width,
                             .upscaledHeight   = height,
                             // the same as VulkanDevice for DLSS
                             .jitter = res.temporalUpscaling ? HaltonSequence::GetJitter_Halton23( impl->frameId )
                                                             : RgFloat2D{ 0, 0 },
                             .frameId          = impl->frameId,
                             .currentTime      = pInfo->currentTime,
                             .previousTime     = impl->previousFrameTime,
                             .worldUp          = impl->worldUp,
                             .rayCullBackFaces = impl->rayCullBackFaces,
                             .frameIndex       = impl->frameIndex,
                             .waterNormalTextureIndex = impl->textures->GetWaterNormalTextureIndex(),
                             .dirtMaskTextureIndex    = impl->textures->GetDirtMaskTextureIndex(),
                             .lightmapScreenCoverage  = impl->lightmapScreenCoverage,
                             .volumeLightSourceIndex  = [ & ]() -> uint32_t {
                                 if( !camera )
                                 {
                                     return LIGHT_INDEX_NONE;
                                 }
                                 float v[ 16 ], proj[ 16 ], inv[ 16 ];
                                 MakeCameraMatrices( *camera, v, proj );
                                 Matrix::Inverse( inv, v );
                                 const auto uniqueId = impl->sceneManager->TryGetVolumetricLight(
                                     RgFloat3D{ inv[ 12 ], inv[ 13 ], inv[ 14 ] } );
                                 return uniqueId ? impl->lights->GetLightIndexForShaders( impl->frameIndex, &*uniqueId )
                                                 : LIGHT_INDEX_NONE;
                             }(),
                             .hdr                     = impl->hdr,
                         } );

            if( gu.skyType == SKY_TYPE_RASTERIZED_GEOMETRY )
            {
                const auto& sky   = pnext::get< RgDrawFrameSkyParams >( *pInfo );
                const bool  drawn = impl->rasterizer->RenderSkyCubemap( cmd,
                                                                       *impl->textures,
                                                                       sky.skyViewerPosition,
                                                                       camera ? camera->cameraNear : 0.1f,
                                                                       camera ? camera->cameraFar : 10000.0f );
                if( !drawn )
                {
                    gu.skyType = SKY_TYPE_COLOR;
                }
            }

            gu.fluidEnabled = impl->fluid && impl->fluid->Active();
            RG_SET_VEC3_A( gu.fluidColor, impl->fluidColor.data );

            memcpy( impl->globalUniform[ impl->frameIndex ].contents, &gu, sizeof( gu ) );
        }

        id< MTLTexture > toPresent = impl->outColor;

        if( rtPath && pInfo->presentPrevFrame && impl->pathTracer->GetPrevPresented() )
        {
            toPresent = impl->pathTracer->GetPrevPresented();
        }
        else if( rtPath )
        {
            impl->framebuffers->PrepareForSize( renderWidth, renderHeight, width, height );

            // resources by the names in RTGL1's GLSL shaders
            MetalBindingTable& b = impl->bindings;
            impl->framebuffers->Bind( b, impl->frameIndex );
            b.SetBuffer( "GlobalUniform", impl->globalUniform[ impl->frameIndex ] );
            impl->scene->Bind( b );
            b.SetArgumentBuffer(
                "globalTextures_Images", impl->textures->ImageTableBuffer(), impl->textures->LiveTextures() );
            b.SetArgumentBuffer( "globalTextures_Samplers", impl->textures->SamplerTableBuffer(), {} );
            b.SetTexture( "blueNoiseTextures", impl->blueNoise );
            b.SetTexture( "renderCubemap", impl->rasterizer->SkyCubemap() );
            b.SetSampler( "renderCubemapSmplr", impl->linearClampSampler );
            impl->lights->Bind( b, impl->frameIndex );

            const auto rasterParams = MetalRasterizer::PassParams{
                .shaders    = *impl->rtShaders,
                .bindings   = b,
                .view       = impl->uniform.view,
                .projection = impl->uniform.projection,
                .jitter     = { impl->uniform.jitterX, impl->uniform.jitterY },
                .width      = renderWidth,
                .height     = renderHeight,
            };

            // draw rasterized sky to albedo before tracing primary rays
            if( impl->uniform.skyType == SKY_TYPE_RASTERIZED_GEOMETRY && !pInfo->disableRasterization )
            {
                impl->rasterizer->DrawSkyToAlbedo(
                    cmd,
                    rasterParams,
                    impl->framebuffers->Get( FB_IMAGE_INDEX_ALBEDO, impl->frameIndex ),
                    pnext::get< RgDrawFrameSkyParams >( *pInfo ).skyViewerPosition );
            }

            // Debug: RTGL1_METAL_DEBUG_FLUID_SPAWN=<count>, spawn fluid in front of the camera each frame
            if( impl->fluid && camera )
            {
                const char* spawn = std::getenv( "RTGL1_METAL_DEBUG_FLUID_SPAWN" );
                const char* until = std::getenv( "RTGL1_METAL_DEBUG_FLUID_SPAWN_UNTIL" );
                if( spawn && ( !until || impl->frameCounter < std::strtoull( until, nullptr, 10 ) ) )
                {
                    float v[ 16 ], proj[ 16 ], inv[ 16 ];
                    MakeCameraMatrices( *camera, v, proj );
                    Matrix::Inverse( inv, v );
                    // view space -Z is forward
                    const RgFloat3D pos     = { inv[ 12 ], inv[ 13 ], inv[ 14 ] };
                    const RgFloat3D forward = { -inv[ 8 ], -inv[ 9 ], -inv[ 10 ] };
                    // RTGL1_METAL_DEBUG_FLUID_SPAWN_DROP: how much lower than the camera, in meters
                    const char*     dropStr = std::getenv( "RTGL1_METAL_DEBUG_FLUID_SPAWN_DROP" );
                    const float     drop    = dropStr ? float( std::atof( dropStr ) ) : 2.3f;
                    // RTGL1_METAL_DEBUG_FLUID_SPAWN_DIST: how far ahead, in meters
                    const char*     distStr = std::getenv( "RTGL1_METAL_DEBUG_FLUID_SPAWN_DIST" );
                    const float     dist    = distStr ? float( std::atof( distStr ) ) : 4.0f;
                    impl->fluid->AddSource( RgSpawnFluidInfo{
                        .sType                  = RG_STRUCTURE_TYPE_SPAWN_FLUID_INFO,
                        .pNext                  = nullptr,
                        // ahead, and to the right (not behind the first-person weapon), near the floor
                        // (a high fall is too fast for Fluid_Particles' collisions, as on Vulkan)
                        .position               = { pos.data[ 0 ] + forward.data[ 0 ] * dist + inv[ 0 ] * 1.5f - impl->worldUp.data[ 0 ] * drop,
                                                    pos.data[ 1 ] + forward.data[ 1 ] * dist + inv[ 1 ] * 1.5f - impl->worldUp.data[ 1 ] * drop,
                                                    pos.data[ 2 ] + forward.data[ 2 ] * dist + inv[ 2 ] * 1.5f - impl->worldUp.data[ 2 ] * drop },
                        .radius                 = 0.05f,
                        .velocity               = { forward.data[ 0 ], forward.data[ 1 ], forward.data[ 2 ] },
                        .dispersionVelocity     = 0.9f,
                        .dispersionAngleDegrees = 30,
                        .count                  = uint32_t( std::max( 1, std::atoi( spawn ) ) ),
                    } );
                }
            }

            // Fluid::Visualize: the particles of the previous simulation, before the primary rays
            if( impl->fluid && !pInfo->disableRasterization )
            {
                impl->fluid->Visualize( cmd,
                                        *impl->rtShaders,
                                        b,
                                        impl->framebuffers->Get( FB_IMAGE_INDEX_DEPTH_FLUID, impl->frameIndex ),
                                        impl->framebuffers->Get( FB_IMAGE_INDEX_FLUID_NORMAL, impl->frameIndex ),
                                        impl->uniform.view,
                                        impl->uniform.projection,
                                        renderWidth,
                                        renderHeight,
                                        camera ? camera->cameraNear : 0.1f,
                                        camera ? camera->cameraFar : 10000.0f );
            }

            toPresent = impl->pathTracer->Render(
                cmd,
                MetalPathTracer::Params{
                    .frameIndex                          = impl->frameIndex,
                    .width                               = renderWidth,
                    .height                              = renderHeight,
                    .currentTime                         = float( pInfo->currentTime ),
                    .primaryRaysMaxAlbedoLayers          = impl->primaryRaysMaxAlbedoLayers,
                    .indirectIlluminationMaxAlbedoLayers = impl->indirectIlluminationMaxAlbedoLayers,
                    .lightmapLayerIndex                  = impl->lightmapLayerIndex,
                    .reflectRefractMaxDepth              = impl->uniform.reflectRefractMaxDepth,
                    .volumetricMaxHistoryLength =
                        pInfo->resetHistory ? 0 : pnext::get< RgDrawFrameVolumetricParams >( *pInfo ).maxHistoryLength,
                    .globalUniform = impl->globalUniform[ impl->frameIndex ],
                    .skyCubemap    = impl->rasterizer->SkyCubemap(),
                    .upscaledWidth     = width,
                    .upscaledHeight    = height,
                    .upscaler          = res.temporalUpscaling ? MetalPathTracer::Upscaler::MetalFX
                                         : impl->resolution.upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_LINEAR
                                             ? MetalPathTracer::Upscaler::Linear
                                             : MetalPathTracer::Upscaler::Nearest,
                    .frameId           = impl->frameId,
                    .effectWipeIsUsed  = impl->effectWipeIsUsed,
                    .sharpenTechnique  = impl->resolution.sharpenTechnique,
                    .hdr               = impl->hdr,
                    .jitter            = { impl->uniform.jitterX, impl->uniform.jitterY },
                    .resetHistory      = !!pInfo->resetHistory,
                    .cameraNear        = camera ? camera->cameraNear : 0.1f,
                    .cameraFar         = camera ? camera->cameraFar : 10000.0f,
                    .cameraFovY        = camera ? camera->fovYRadians : 1.0f,
                    .metalFXDenoiser   = metalFXDenoiser,
                    .view              = impl->uniform.view,
                    .projection        = impl->uniform.projection,
                    .hudlessCopy       = frameInterpolation,
                    .rasterDecals =
                        [ & ]( id< MTLCommandBuffer > c ) {
                            if( !pInfo->disableRasterization )
                            {
                                impl->rasterizer->DrawDecals(
                                    c,
                                    rasterParams,
                                    impl->framebuffers->Get( FB_IMAGE_INDEX_ALBEDO, impl->frameIndex ),
                                    impl->framebuffers->Get( FB_IMAGE_INDEX_NORMAL_DECAL, impl->frameIndex ),
                                    impl->framebuffers->Get( FB_IMAGE_INDEX_SCREEN_EMISSION, impl->frameIndex ) );
                            }
                        },
                    .rasterToFinal =
                        [ & ]( id< MTLCommandBuffer > c, id< MTLTexture > overlay ) {
                            if( !pInfo->disableRasterization )
                            {
                                impl->rasterizer->DrawToFinal(
                                    c,
                                    rasterParams,
                                    overlay ? overlay : impl->framebuffers->Get( FB_IMAGE_INDEX_FINAL, impl->frameIndex ),
                                    impl->framebuffers->Get( FB_IMAGE_INDEX_SCREEN_EMISSION, impl->frameIndex ),
                                    impl->framebuffers->Get( FB_IMAGE_INDEX_REACTIVITY, impl->frameIndex ),
                                    impl->lightmapScreenCoverage,
                                    overlay != nil );
                            }
                        },
                    .rasterClassic =
                        [ & ]( id< MTLCommandBuffer > c, id< MTLTexture > target ) {
                            if( !pInfo->disableRasterization && impl->lightmapScreenCoverage > 0 )
                            {
                                auto classicParams   = rasterParams;
                                classicParams.width  = uint32_t( target.width );
                                classicParams.height = uint32_t( target.height );
                                impl->rasterizer->DrawClassic( c,
                                                               classicParams,
                                                               target,
                                                               pnext::get< RgDrawFrameSkyParams >( *pInfo ).skyViewerPosition,
                                                               impl->lightmapScreenCoverage,
                                                               renderWidth );
                            }
                        },
                    .rasterHud =
                        [ & ]( id< MTLCommandBuffer > c, id< MTLTexture > target ) {
                            if( !pInfo->disableRasterization )
                            {
                                auto hudParams   = rasterParams;
                                hudParams.width  = uint32_t( target.width );
                                hudParams.height = uint32_t( target.height );
                                impl->rasterizer->DrawToSwapchain( c, hudParams, target );
                            }
                        },
                },
                *pInfo,
                *impl->framebuffers,
                b );

            // Fluid::Simulate, with this frame's TLAS
            if( impl->fluid )
            {
                impl->fluid->Simulate( cmd,
                                       *impl->rtShaders,
                                       b,
                                       impl->frameIndex,
                                       float( std::max( pInfo->currentTime - impl->previousFrameTime, 0.0 ) ),
                                       impl->fluidGravity );
            }

            // FrameGeneration: MetalFX frame interpolation between the previous and the current images
            if( frameInterpolation )
            {
                if( @available( macOS 26.0, * ) )
                {
                    id< MTLFXFrameInterpolator > fi =
                        impl->PrepareFrameInterpolator( renderWidth, renderHeight, width, height, toPresent.pixelFormat );
                    if( fi && toPresent.width == width && toPresent.height == height )
                    {
                        auto&            fg   = impl->fg;
                        id< MTLTexture > cur  = fg.color[ fg.current ];
                        id< MTLTexture > prev = fg.color[ fg.current ^ 1 ];

                        // the interpolator gets the image without HUD, and puts the HUD on top of the generated one
                        // (with 'uiTextureComposited', i.e. given the image with HUD, the generated one is black);
                        // HUD is drawn as usual (before HDR prepare), the UI texture is the difference
                        id< MTLTexture > hudless = impl->pathTracer->GetHudlessImage();
                        if( !hudless )
                        {
                            hudless = toPresent;
                        }
                        {
                            id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
                            blit.label                       = @"Frame interpolation - Copy color";
                            [blit copyFromTexture:hudless toTexture:cur];
                            // the actual frame with HUD; a copy, as presented after the next frame might have started
                            [blit copyFromTexture:toPresent toTexture:fg.present[ fg.current ]];
                            [blit endEncoding];
                        }

                        auto hudParams   = rasterParams;
                        hudParams.width  = width;
                        hudParams.height = height;
                        impl->rasterizer->DrawToSwapchain( cmd, hudParams, fg.uiCoverage, true );
                        {
                            id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
                            enc.label                          = @"Frame interpolation - UI";
                            [enc setComputePipelineState:fg.uiPipeline];
                            [enc setTexture:toPresent atIndex:0];
                            [enc setTexture:hudless atIndex:1];
                            [enc setTexture:fg.uiCoverage atIndex:2];
                            [enc setTexture:fg.ui atIndex:3];
                            [enc dispatchThreads:MTLSizeMake( width, height, 1 ) threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
                            [enc endEncoding];
                        }

                        if( fg.hasHistory && !pInfo->resetHistory )
                        {
                            const float delta = float( pInfo->currentTime - impl->previousFrameTime );

                            fi.colorTexture         = cur;
                            fi.prevColorTexture     = prev;
                            fi.depthTexture         = impl->framebuffers->Get( FB_IMAGE_INDEX_DEPTH_NDC, impl->frameIndex );
                            fi.motionTexture        = impl->framebuffers->Get( FB_IMAGE_INDEX_MOTION_DLSS, impl->frameIndex );
                            fi.uiTexture            = fg.ui;
                            fi.uiTextureComposited  = NO;
                            fi.outputTexture        = fg.output[ fg.current ];
                            // the same as for the temporal scaler: motion vectors are in UV space
                            fi.motionVectorScaleX   = float( renderWidth );
                            fi.motionVectorScaleY   = float( renderHeight );
                            fi.jitterOffsetX        = -impl->uniform.jitterX;
                            fi.jitterOffsetY        = -impl->uniform.jitterY;
                            fi.deltaTime            = std::clamp( delta, 0.001f, 0.25f );
                            fi.nearPlane            = camera ? camera->cameraNear : 0.1f;
                            fi.farPlane             = camera ? camera->cameraFar : 10000.0f;
                            fi.fieldOfView          = ( camera ? camera->fovYRadians : 1.0f ) * 180.0f / float( M_PI );
                            fi.aspectRatio          = float( width ) / float( height );
                            fi.depthReversed        = NO;
                            fi.shouldResetHistory   = NO;
                            [fi encodeToCommandBuffer:cmd];

                            if( impl->resolution.frameGeneration == RG_FRAME_GENERATION_MODE_ON )
                            {
                                generated = fg.output[ fg.current ];
                            }
                        }
                        asyncPresentReal = fg.present[ fg.current ];
                        fg.hasHistory    = true;
                        fg.current ^= 1;
                    }
                    else
                    {
                        impl->fg.hasHistory = false;
                    }
                }
            }

            // Debug: RTGL1_METAL_SHOW=framebuf<Name> presents that framebuffer instead
            if( const char* show = std::getenv( "RTGL1_METAL_SHOW" ) )
            {
                if( id< MTLTexture > t = impl->pathTracer->GetDebugTexture( show ) )
                {
                    toPresent = t;
                }
                for( uint32_t i = 0; i < ShFramebuffers_Count; i++ )
                {
                    if( std::strcmp( ShFramebuffers_MetalInfos[ i ].name, show ) == 0 )
                    {
                        toPresent = impl->framebuffers->Get( FramebufferImageIndex( i ), impl->frameIndex );
                    }
                }
            }
        }
        else
        {
            // nothing to trace, or no RTGL1 shaders
            id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
            enc.label                          = @"Sky only";
            [enc setComputePipelineState:impl->skyOnly];
            [enc setBuffer:impl->globalUniform[ impl->frameIndex ] offset:0 atIndex:MTL_BINDING_GLOBAL_UNIFORM];
            [enc setTexture:impl->outColor atIndex:0];
            [enc setTexture:impl->rasterizer->SkyCubemap() atIndex:1];

            const MTLSize group = MTLSizeMake( 8, 8, 1 );
            [enc dispatchThreads:MTLSizeMake( width, height, 1 ) threadsPerThreadgroup:group];
            [enc endEncoding];
        }

        id< MTLBuffer > captureBuffer = nil;
        uint32_t        captureWidth = 0, captureHeight = 0;

        const double paced = impl->PacedInterval();

        if( asyncPresentReal )
        {
            const bool captureNow = impl->capture && impl->capture->Contains( impl->frameCounter );
            if( captureNow && impl->hdr )
            {
                debug::Warning( "Metal backend: capture is not supported with HDR output" );
            }
            impl->PresentAsync( cmd,
                                asyncPresentReal,
                                generated,
                                captureNow && !impl->hdr ? std::optional{ impl->capture->PathFor( impl->frameCounter ) }
                                                         : std::nullopt,
                                // Debug: RTGL1_METAL_CAPTURE_GENERATED=1 captures the generated frame instead
                                std::getenv( "RTGL1_METAL_CAPTURE_GENERATED" ) != nullptr );
        }
        else
        {
            // the rendering is committed separately, so its GPU time doesn't include waiting for a drawable
            auto gpuFrameTime = impl->gpuFrameTime;
            [cmd addCompletedHandler:^( id< MTLCommandBuffer > c ) {
              OnFrameCompleted( c, *gpuFrameTime );
            }];
            [cmd commit];

            id< MTLCommandBuffer > pc = [impl->queue commandBuffer];
            pc.label                  = @"RTGL1 present";

            // acquire the drawable as late as possible
            if( id< CAMetalDrawable > drawable = [impl->layer nextDrawable] )
            {
                MTLRenderPassDescriptor* pass          = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[ 0 ].texture     = drawable.texture;
                pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionDontCare;
                pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;

                id< MTLRenderCommandEncoder > enc = [pc renderCommandEncoderWithDescriptor:pass];
                enc.label                         = @"Blit to swapchain";
                [enc setRenderPipelineState:impl->blit[ impl->hdr ? 1 : 0 ]];
                [enc setFragmentTexture:toPresent atIndex:0];
                {
                    // Debug: RTGL1_METAL_SHOW_SCALE, for RTGL1_METAL_SHOW
                    const char* showScale = std::getenv( "RTGL1_METAL_SHOW_SCALE" );
                    const float scale     = showScale ? float( std::atof( showScale ) ) : 1.0f;
                    [enc setFragmentBytes:&scale length:sizeof( scale ) atIndex:0];
                }
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

                // with RTGL1's shaders, it was drawn into the upscaled framebuf before the tonemapping
                if( !rtPath )
                {
                    impl->rasterizer->DrawSwapchainLegacy( enc,
                                                           *impl->textures,
                                                           uint32_t( drawable.texture.width ),
                                                           uint32_t( drawable.texture.height ),
                                                           drawable.texture.pixelFormat );
                }
                [enc endEncoding];

                if( impl->capture && impl->capture->Contains( impl->frameCounter ) && impl->hdr )
                {
                    debug::Warning( "Metal backend: capture is not supported with HDR output" );
                }
                else if( impl->capture && impl->capture->Contains( impl->frameCounter ) )
                {
                    captureWidth  = uint32_t( drawable.texture.width );
                    captureHeight = uint32_t( drawable.texture.height );
                    captureBuffer = [impl->device newBufferWithLength:size_t( captureWidth ) * captureHeight * 4
                                                              options:MTLResourceStorageModeShared];

                    id< MTLBlitCommandEncoder > blit = [pc blitCommandEncoder];
                    blit.label                       = @"Capture";
                    [blit copyFromTexture:drawable.texture
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake( 0, 0, 0 )
                                  sourceSize:MTLSizeMake( captureWidth, captureHeight, 1 )
                                    toBuffer:captureBuffer
                           destinationOffset:0
                      destinationBytesPerRow:captureWidth * 4
                    destinationBytesPerImage:size_t( captureWidth ) * captureHeight * 4];
                    [blit endEncoding];
                }

                if( paced > 0 )
                {
                    [pc presentDrawable:drawable afterMinimumDuration:paced];
                }
                else
                {
                    [pc presentDrawable:drawable];
                }
            }

            dispatch_semaphore_t sem = impl->frameSemaphore;
            [pc addCompletedHandler:^( id< MTLCommandBuffer > c ) {
              if( c.status == MTLCommandBufferStatusError )
              {
                  OnFrameCompleted( c, *gpuFrameTime );
              }
              dispatch_semaphore_signal( sem );
            }];
            [pc commit];

            if( captureBuffer )
            {
                [pc waitUntilCompleted];
                [cmd waitUntilCompleted];
                debug::Info( "Metal backend: GPU time of the captured frame: {:.2f} ms",
                             ( cmd.GPUEndTime - cmd.GPUStartTime ) * 1000.0 );
                SaveCapture( captureBuffer, captureWidth, captureHeight, impl->capture->PathFor( impl->frameCounter ) );
            }
        }
    }

    impl->frameCounter++;
    impl->frameId++;
    if( const char* spin = std::getenv( "RTGL1_METAL_DEBUG_SPIN" ) )
    {
        // sway, to keep looking at the same place
        g_debugSpinAngle = float( std::atof( spin ) ) * std::sin( float( impl->frameCounter ) * 0.1f );
    }
    impl->previousFrameTime = pInfo->currentTime;

    impl->frameStarted = false;
    impl->frameIndex   = ( impl->frameIndex + 1 ) % MAX_FRAMES_IN_FLIGHT;
}

void RTGL1::MetalDevice::UploadMeshPrimitive( const RgMeshInfo*          pMesh,
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
    if( !impl->frameStarted )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_STARTED );
    }

    if( auto raster = pnext::find< RgMeshPrimitiveSwapchainedEXT >( pPrimitive ) )
    {
        // the same as in VulkanDevice::UploadMeshPrimitive
        float vp[ 16 ];
        if( raster->pViewProjection )
        {
            memcpy( vp, raster->pViewProjection, sizeof( vp ) );
        }
        else
        {
            float view[ 16 ]{}, projection[ 16 ]{};
            const CGSize drawableSize = impl->layer.drawableSize;
            if( auto c = impl->sceneManager->GetCamera(
                    impl->camera, float( drawableSize.width / std::max( 1.0, drawableSize.height ) ) ) )
            {
                MakeCameraMatrices( *c, view, projection );
            }
            const float* v = raster->pView ? raster->pView : view;
            const float* p = raster->pProjection ? raster->pProjection : projection;
            Matrix::Multiply( vp, p, v );
        }

        impl->rasterizer->Add( MetalRasterizer::Type::Swapchain,
                               pMesh ? pMesh->transform : RG_TRANSFORM_IDENTITY,
                               *pPrimitive,
                               vp,
                               raster->pViewport,
                               *impl->textures );
        return;
    }

    if( !pMesh )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pMesh->sType != RG_STRUCTURE_TYPE_MESH_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( ( pMesh->flags & RG_MESH_EXPORT_AS_SEPARATE_FILE ) && !pMesh->isExportable )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT,
                           "RG_MESH_INFO_EXPORT_AS_SEPARATE_FILE is set, expected isExportable to be true" );
    }

    const RgMeshInfo& mesh = *pMesh;

    // the same as uploadPrimitive_WithMeta in VulkanDevice::UploadMeshPrimitive

    // ignore replacement, if the scene requires
    if( mesh.isExportable && ( mesh.flags & RG_MESH_EXPORT_AS_SEPARATE_FILE ) && !Utils::IsCstrEmpty( mesh.pMeshName ) )
    {
        if( impl->sceneMeta->IsReplacementIgnored( impl->sceneManager->GetImportMapName(), mesh.pMeshName ) )
        {
            return;
        }
    }

    auto modified               = RgMeshPrimitiveInfo{ *pPrimitive };
    auto modified_attachedLight = std::optional< RgMeshPrimitiveAttachedLightEXT >{};
    auto modified_pbr           = std::optional< RgMeshPrimitivePBREXT >{};

    if( auto original = pnext::find< RgMeshPrimitiveAttachedLightEXT >( pPrimitive ) )
    {
        modified_attachedLight = *original;
    }
    if( auto original = pnext::find< RgMeshPrimitivePBREXT >( pPrimitive ) )
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

    if( !impl->textureMeta->Modify( modified, modified_attachedLight, modified_pbr, false ) )
    {
        return;
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

    const RgMeshPrimitiveInfo& prim = modified;

    // uploadPrimitive_Core
    if( IsRasterized( mesh, prim ) )
    {
        impl->rasterizer->Add( prim.flags & RG_MESH_PRIMITIVE_SKY     ? MetalRasterizer::Type::Sky
                               : prim.flags & RG_MESH_PRIMITIVE_DECAL ? MetalRasterizer::Type::Decal
                                                                      : MetalRasterizer::Type::World,
                               mesh.transform,
                               prim,
                               nullptr,
                               nullptr,
                               *impl->textures );
        return;
    }

    // upload a primitive, potentially loading replacements
    impl->sceneManager->UploadPrimitive( impl->frameIndex, mesh, prim );

    if( impl->lightmapScreenCoverage > 0 )
    {
        if( !( mesh.flags & RG_MESH_FIRST_PERSON_VIEWER ) )
        {
            impl->rasterizer->Add(
                MetalRasterizer::Type::WorldClassic, mesh.transform, prim, nullptr, nullptr, *impl->textures );
        }
    }

    // TODO: remove legacy way to attach lights
    if( auto attachedLight = pnext::find< RgMeshPrimitiveAttachedLightEXT >( &prim ) )
    {
        bool quad = ( prim.indexCount == 6 && prim.vertexCount == 4 ) ||
                    ( prim.indexCount == 0 && prim.vertexCount == 6 );

        if( attachedLight->evenOnDynamic || quad )
        {
            auto& tempStorageInit   = impl->tempStorageInit;
            auto& tempStorageLights = impl->tempStorageLights;
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
                    mesh, prim, impl->sceneManager->GetWorldScale(), tempStorageInit, tempStorageLights );
            }

            auto hashCombine = []< typename T >( uint64_t seed, const T& v ) {
                seed ^= std::hash< T >{}( v ) + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 );
                return seed;
            };

            static const uint64_t attchSalt = hashCombine( 0, std::string_view{ "attachedlight" } );

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

void RTGL1::MetalDevice::UploadLensFlare( const RgLensFlareInfo* pInfo )
{
    NotImplementedYet( "rgUploadLensFlare" );
}

void RTGL1::MetalDevice::SpawnFluid( const RgSpawnFluidInfo* pInfo )
{
    if( !pInfo )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    // the same as VulkanDevice: ignored, if fluid is disabled
    if( impl->fluid )
    {
        impl->fluid->AddSource( *pInfo );
    }
}

void RTGL1::MetalDevice::UploadCamera( const RgCameraInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_CAMERA_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }

    // TODO: other cameras than the main one
    impl->camera        = *pInfo;
    impl->camera->pNext = nullptr;
    if( pInfo->pView )
    {
        memcpy( impl->cameraView, pInfo->pView, sizeof( impl->cameraView ) );
        impl->camera->pView = impl->cameraView;
    }

    // the same as in VulkanDevice::UploadCamera
    if( auto readback = pnext::find< RgCameraInfoReadbackEXT >( const_cast< RgCameraInfo* >( pInfo ) ) )
    {
        static_assert( sizeof readback->view == 16 * sizeof( float ) );
        static_assert( sizeof readback->projection == 16 * sizeof( float ) );

        MakeCameraMatrices( *impl->camera, readback->view, readback->projection );
        Matrix::Inverse( readback->viewInverse, readback->view );
        Matrix::Inverse( readback->projectionInverse, readback->projection );
    }
}

// The same as VulkanDevice::UploadLight, without static lights and export
void RTGL1::MetalDevice::UploadLight( const RgLightInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    if( pInfo->sType != RG_STRUCTURE_TYPE_LIGHT_INFO )
    {
        throw RgException( RG_RESULT_WRONG_STRUCTURE_TYPE );
    }
    if( !impl->frameStarted )
    {
        throw RgException( RG_RESULT_FRAME_WASNT_STARTED );
    }

    auto light = LightCopy{ .base = *pInfo };
    if( auto l = pnext::find< RgLightDirectionalEXT >( pInfo ) )
    {
        light.extension = *l;
    }
    else if( auto l = pnext::find< RgLightSphericalEXT >( pInfo ) )
    {
        light.extension = *l;
    }
    else if( auto l = pnext::find< RgLightSpotEXT >( pInfo ) )
    {
        light.extension = *l;
    }
    else if( auto l = pnext::find< RgLightPolygonalEXT >( pInfo ) )
    {
        light.extension = *l;
    }
    else
    {
        debug::Warning( "Couldn't find RgLightDirectionalEXT, RgLightSphericalEXT, RgLightSpotEXT "
                        "or RgLightPolygonalEXT on RgLightInfo (uniqueID={})",
                        pInfo->uniqueID );
        return;
    }
    if( auto l = pnext::find< RgLightAdditionalEXT >( pInfo ) )
    {
        light.additional = *l;
    }

    // reset pNext, as using in-place members
    light.base.pNext = nullptr;
    std::visit( []( auto& e ) { e.pNext = nullptr; }, light.extension );
    if( light.additional )
    {
        light.additional->pNext = nullptr;
    }

    impl->sceneManager->UploadLight( impl->frameIndex, light, false );
}

void RTGL1::MetalDevice::ProvideOriginalTexture( const RgOriginalTextureInfo* pInfo )
{
    if( pInfo == nullptr )
    {
        throw RgException( RG_RESULT_WRONG_FUNCTION_ARGUMENT, "Argument is null" );
    }
    impl->textures->Provide( *pInfo );
    // SHIPPING_HACK
    impl->sceneManager->OnOriginalTextureProvided( pInfo->pTextureName );
}

void RTGL1::MetalDevice::ProvideOriginalCubemapTexture( const RgOriginalCubemapInfo* pInfo )
{
    NotImplementedYet( "ProvideOriginalCubemapTexture" );
}

void RTGL1::MetalDevice::MarkOriginalTextureAsDeleted( const char* pTextureName )
{
    impl->textures->MarkAsDeleted( pTextureName );
}

bool RTGL1::MetalDevice::IsUpscaleTechniqueAvailable( RgRenderUpscaleTechnique technique,
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
            return frameGeneration == RG_FRAME_GENERATION_MODE_OFF;

        case RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2:
            // done by the MetalFX temporal scaler; frame generation by the MetalFX frame interpolator
            if( !impl->metalFXAvailable )
            {
                if( ppFailureReason )
                {
                    *ppFailureReason = "MetalFX temporal scaling is not supported by this GPU";
                }
                return false;
            }
            if( frameGeneration != RG_FRAME_GENERATION_MODE_OFF && !impl->frameInterpolationAvailable )
            {
                if( ppFailureReason )
                {
                    *ppFailureReason = "MetalFX frame interpolation requires macOS 26 and a supported GPU";
                }
                return false;
            }
            return true;

        case RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS:
            if( ppFailureReason )
            {
                *ppFailureReason = "DLSS is not available on macOS";
            }
            return false;

        default: return false;
    }
}

bool RTGL1::MetalDevice::IsDXGIAvailable( const char** ppFailureReason ) const
{
    if( ppFailureReason )
    {
        *ppFailureReason = "DXGI is not available on macOS";
    }
    return false;
}

RgFeatureFlags RTGL1::MetalDevice::GetSupportedFeatures() const
{
    // HDR: macOS EDR (on an SDR display, values above 1.0 are clipped)
    return RG_FEATURE_HDR | ( impl->rtShaders ? RG_FEATURE_FLUID : 0 ) |
           ( impl->metalFXDenoiserAvailable ? RG_FEATURE_METALFX_DENOISER : 0 );
}

RgUtilMemoryUsage RTGL1::MetalDevice::RequestMemoryUsage() const
{
    return RgUtilMemoryUsage{
        .vramUsed  = impl->device.currentAllocatedSize,
        .vramTotal = impl->device.recommendedMaxWorkingSetSize,
    };
}

RgPrimitiveVertex* RTGL1::MetalDevice::ScratchAllocForVertices( uint32_t count )
{
    return new RgPrimitiveVertex[ count ];
}

void RTGL1::MetalDevice::ScratchFree( const RgPrimitiveVertex* pPointer )
{
    delete[] pPointer;
}

void RTGL1::MetalDevice::Print( std::string_view msg, RgMessageSeverityFlags severity ) const
{
    static auto printMutex = std::mutex{};

    auto l = std::lock_guard{ printMutex };

    if( impl->pfnPrint )
    {
        // msg is a view into a null-terminated string, the same as in VulkanDevice
        impl->pfnPrint( msg.data(), severity, impl->pUserPrintData );
    }
}
