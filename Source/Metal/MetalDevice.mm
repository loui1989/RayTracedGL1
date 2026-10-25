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
#include "MetalProfiler.h"

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
#include "Camera.h"
#include "PrimitiveLights.h"
#include "PrimitiveRouting.h"
#include "SceneMeta.h"
#include "TextureMeta.h"
#include "RgException.h"

#include "Stb/stb_image_write.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    id< MTLLibrary > lib   = nil;
    if( std::filesystem::exists( path ) )
    {
        lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@( path.c_str() )] error:&error];
    }
    else
    {
        // built without Xcode's Metal toolchain: RTGL1.metal, with its includes inlined
        lib = RTGL1::MetalComputePipelines::CompileLibrarySource( device, GetLibraryFolder() / "RTGL1.metal", error );
    }
    if( !lib )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                                  std::string{ "Can't load " } + path.string() + ": " +
                                      ( error ? error.localizedDescription.UTF8String : "no RTGL1.metallib or RTGL1.metal" ) );
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

// Scene.h
RgFloat3D MakeCameraPosition( const RTGL1::Camera& c )
{
    return RgFloat3D{ c.viewInverse[ 12 ], c.viewInverse[ 13 ], c.viewInverse[ 14 ] };
}

// What VulkanDevice::FillUniform reads from VulkanDevice's members, from the Metal backend:
// the same names and methods, so that Doom 64 RT's FillUniform is used as is.
struct FillUniformCtx
{
    struct Res
    {
        uint32_t w, h, uw, uh;
        // the MetalFX temporal scaling, in place of FSR2
        bool     temporal;

        uint32_t Width() const { return w; }
        uint32_t Height() const { return h; }
        uint32_t UpscaledWidth() const { return uw; }
        uint32_t UpscaledHeight() const { return uh; }
        float    Aspect() const { return uw == 0 || uh == 0 ? 1.0f : float( uw ) / float( uh ); }
        bool     IsNvDlssEnabled() const { return false; }
        bool     IsAmdFsr2Enabled() const { return temporal; }
        bool     IsNvDlssRayReconstructionEnabled() const { return false; }
        // RenderResolutionHelper::GetMipLodBias
        float    GetMipLodBias( float nativeBias = 0.0f ) const
        {
            if( !temporal || uw == 0 )
            {
                return nativeBias;
            }
            return nativeBias + std::log2( std::max( 0.01f, float( w ) / float( uw ) ) ) - 1.0f;
        }
    } res;

    const RTGL1::MetalLightManager*   lights;
    const RTGL1::MetalTextureManager* textures;

    struct FrameState
    {
        uint32_t frameIndex;
        uint32_t GetFrameIndex() const { return frameIndex; }
    } frameState;

    struct Swapchain
    {
        bool hdr;
        bool IsHDREnabled() const { return hdr; }
        // HDR on macOS is EDR, extended linear sRGB
        bool IsST2084ColorSpace() const { return false; }
        bool WithDLSS3FrameGeneration() const { return false; }
        bool WithFSR3FrameGeneration() const { return false; }
    } swapchain;

    struct Scene
    {
        RTGL1::Camera              camera;
        std::optional< uint64_t > volumetricLight;
        RgFloat3D                  worldUp;

        const RTGL1::Camera& GetCamera( float ) const { return camera; }
        template< typename L >
        std::optional< uint64_t > TryGetVolumetricLight( const L&, const RgFloat3D& ) const
        {
            return volumetricLight;
        }
        RgFloat3D GetWorldUp() const { return worldUp; }
    } scene;

    // no cubemap textures on Metal yet
    struct Cubemaps
    {
        uint32_t TryGetDescriptorIndex( const char* ) const { return 0; }
    } cubemaps;

    // VulkanDevice_Dev's fields that FillUniform reads; there's no Dev window on Metal
    struct Devmode
    {
        uint32_t debugShowFlags{ 0 };
        bool     materialStripNormals{ false };
        bool     materialStripEmissives{ false };
        bool     materialStripMetallic{ false };
        bool     materialStripHeight{ false };
        bool     materialStripRoughness{ false };
        float    roughnessTowardMatte{ 0 };
        bool     antiFirefly{ true };
        bool     rrTemporalPrefilterSticky{ false };
        bool     rrTemporalPrefilter{ false };
        bool     illumSensSticky{ false };
        float    illumSensDirect{ 0 };
        float    illumSensIndirect{ 0 };
        float    illumSensSpec{ 0 };
    };

    uint32_t  frameId;
    double    currentTime;
    double    previousTime;
    bool      rayCullBackFaces;
    float     lightmapScreenCoverage;
    bool      fluidActive;
    RgFloat3D fluidColor;
    RgFloat2D jitter;

    // as Fluid::Active
    bool Active() const { return fluidActive; }
};

// Doom 64 RT's VulkanDevice::FillUniform.
// 'gu' persists between frames: the previous frame's values are used for the *Prev ones.
void FillUniform( RTGL1::ShGlobalUniform* gu, const RgDrawFrameInfo& drawInfo, const FillUniformCtx& ctx )
{
    using namespace RTGL1;
    using std::clamp;

    // the members that VulkanDevice::FillUniform reads, from the Metal backend
    const auto& renderResolution          = ctx.res;
    const auto* lightManager              = ctx.lights;
    const auto* textureManager            = ctx.textures;
    const auto& currentFrameState         = ctx.frameState;
    const auto* swapchain                 = &ctx.swapchain;
    const auto* scene                     = &ctx.scene;
    const auto* sceneImportExport         = &ctx.scene;
    const auto* cubemapManager            = &ctx.cubemaps;
    FillUniformCtx::Devmode* devmode      = nullptr;
    const void*              nvDlssRr     = nullptr;
    const uint32_t           frameId      = ctx.frameId;
    const double             currentFrameTime  = ctx.currentTime;
    const double             previousFrameTime = ctx.previousTime;
    const bool               rayCullBackFacingTriangles = ctx.rayCullBackFaces;
    const float              lightmapScreenCoverage     = ctx.lightmapScreenCoverage;
    const auto*              fluid                      = ctx.fluidActive ? &ctx : nullptr;
    const RgFloat3D          fluidColor                 = ctx.fluidColor;

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

        // HaltonSequence::GetJitter_Halton23, if the MetalFX temporal scaling is on
        RgFloat2D jitter = ctx.jitter;

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

    // Debug: RTGL1_METAL_PROFILE=1, the CPU time of the frame's parts, printed every 2 seconds
    struct CpuProfile
    {
        bool     enabled{ false };
        using Clock = std::chrono::steady_clock;
        Clock::time_point startFrameEnd{};
        Clock::time_point lastPrint{};
        bool              printNow{ false };
        double            sincePrint{ 0 }; // ms, between the last two prints
        double   waitFrame{ 0 }, startFrame{ 0 }, betweenFrames{ 0 }, uploads{ 0 }, sceneBuild{ 0 },
                 render{ 0 }, drawFrame{ 0 }, clouds{ 0 }, skyCubemap{ 0 }, skyAlbedo{ 0 };
        uint64_t skyDraws{ 0 };
        uint64_t primitives{ 0 }, frames{ 0 };

        static double Ms( Clock::time_point a, Clock::time_point b )
        {
            return std::chrono::duration< double, std::milli >( b - a ).count();
        }
    } cpu;
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

            id< MTLRenderCommandEncoder > enc = MetalProfiler::Render( pc, pass );
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

            id< MTLBlitCommandEncoder > b = MetalProfiler::Blit( pc );
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

    impl->cpu.enabled = []() {
        const char* e = std::getenv( "RTGL1_METAL_PROFILE" );
        return e && e[ 0 ] == '1';
    }();
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
    MetalProfiler::Init( impl->device );
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

    const auto tWait = Impl::CpuProfile::Clock::now();
    dispatch_semaphore_wait( impl->frameSemaphore, DISPATCH_TIME_FOREVER );
    impl->frameStarted = true;
    const auto tStart = Impl::CpuProfile::Clock::now();
    impl->cpu.waitFrame += Impl::CpuProfile::Ms( tWait, tStart );

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

    impl->cpu.startFrameEnd = Impl::CpuProfile::Clock::now();
    impl->cpu.startFrame += Impl::CpuProfile::Ms( tStart, impl->cpu.startFrameEnd );
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

    const auto tDraw = Impl::CpuProfile::Clock::now();
    impl->cpu.betweenFrames += Impl::CpuProfile::Ms( impl->cpu.startFrameEnd, tDraw );
    // print the profile every 2 seconds, whatever the frame rate
    impl->cpu.printNow = impl->cpu.enabled && tDraw - impl->cpu.lastPrint >= std::chrono::seconds( 2 );
    if( impl->cpu.printNow )
    {
        impl->cpu.sincePrint = Impl::CpuProfile::Ms( impl->cpu.lastPrint, tDraw );
        impl->cpu.lastPrint  = tDraw;
    }
    MetalProfiler::BeginFrame( impl->cpu.printNow );

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

        static const bool profile = [] {
            const char* e = std::getenv( "RTGL1_METAL_PROFILE" );
            return e && e[ 0 ] == '1';
        }();

        const CGSize   drawableSizeForAspect = impl->layer.drawableSize;
        const auto     camera                = impl->sceneManager->GetCamera(
            impl->camera, float( drawableSizeForAspect.width / std::max( 1.0, drawableSizeForAspect.height ) ) );

        const auto tBuild   = Impl::CpuProfile::Clock::now();
        const bool hasScene = camera && !pInfo->disableRayTracedGeometry &&
                              impl->scene->Build( cmd,
                                                  INSTANCE_MASK_WORLD_0 | INSTANCE_MASK_WORLD_1 | INSTANCE_MASK_WORLD_2 );
        impl->cpu.sceneBuild += Impl::CpuProfile::Ms( tBuild, Impl::CpuProfile::Clock::now() );

        if( profile )
        {
            const bool printNow = impl->cpu.printNow;
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
              if( printNow && count > 0 )
              {
                  debug::Warning( "Metal profile: {:>8.3f} ms  GPU frame time (average)", sum / count * 1000.0 );
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
        // (Doom 64 RT's macOS menu asks for it as Ray Reconstruction with the MetalFX upscaler)
        bool metalFXDenoiser = res.temporalUpscaling && impl->metalFXDenoiserAvailable &&
                               ( impl->resolution.denoiser == RG_DENOISER_METALFX ||
                                 impl->resolution.rayReconstruction );
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
        // (with Doom 64 RT's offset, RgDrawFrameTexturesParams::mipLodBiasOffset; the same bias is in the
        // uniform, texFilterParams, for the three-point filtering)
        impl->textures->SetMipLodBias(
            FillUniformCtx::Res{ renderWidth, renderHeight, width, height, res.temporalUpscaling }.GetMipLodBias(
                pnext::get< RgDrawFrameTexturesParams >( *pInfo ).mipLodBiasOffset ) );

        // global uniform
        {
            ShGlobalUniform& gu = impl->uniform;

            auto ctx = FillUniformCtx{
                .res        = { renderWidth, renderHeight, width, height, res.temporalUpscaling },
                .lights     = impl->lights.get(),
                .textures   = impl->textures.get(),
                .frameState = { impl->frameIndex },
                .swapchain  = { impl->hdr },
                .scene      = {},
                .cubemaps   = {},
                .frameId    = impl->frameId,
                .currentTime            = pInfo->currentTime,
                .previousTime           = impl->previousFrameTime,
                .rayCullBackFaces       = impl->rayCullBackFaces,
                .lightmapScreenCoverage = impl->lightmapScreenCoverage,
                .fluidActive            = impl->fluid && impl->fluid->Active(),
                .fluidColor             = impl->fluidColor,
                // the same as VulkanDevice for DLSS
                .jitter = res.temporalUpscaling ? HaltonSequence::GetJitter_Halton23( impl->frameId )
                                                : RgFloat2D{ 0, 0 },
            };
            {
                Camera& c = ctx.scene.camera;
                if( camera )
                {
                    MakeCameraMatrices( *camera, c.view, c.projection );
                    c.aspect      = camera->aspect;
                    c.fovYRadians = camera->fovYRadians;
                    c.cameraNear  = camera->cameraNear;
                    c.cameraFar   = camera->cameraFar;
                }
                else
                {
                    Matrix::MakeViewMatrix( c.view, RgFloat3D{ 0, 0, 0 }, RgFloat3D{ 1, 0, 0 }, RgFloat3D{ 0, 1, 0 } );
                    Matrix::MakeProjectionMatrix( c.projection, 1.0f, 1.0f, 0.1f, 10000.0f );
                    c.cameraNear = 0.1f;
                    c.cameraFar  = 10000.0f;
                }
                Matrix::Inverse( c.viewInverse, c.view );
                Matrix::Inverse( c.projectionInverse, c.projection );
                ctx.scene.worldUp = impl->worldUp;
                if( camera )
                {
                    ctx.scene.volumetricLight = impl->sceneManager->TryGetVolumetricLight( MakeCameraPosition( c ) );
                }
            }
            FillUniform( &gu, *pInfo, ctx );

            // the sky cubemap is rendered below, with RTGL1's shaders (RenderCubemap::Draw)
            if( gu.skyType == SKY_TYPE_RASTERIZED_GEOMETRY &&
                ( !impl->rtShaders || pInfo->disableRasterization ||
                  !impl->rasterizer->HasDraws( MetalRasterizer::Type::Sky ) ) )
            {
                gu.skyType = SKY_TYPE_COLOR;
            }

            gu.fluidEnabled = impl->fluid && impl->fluid->Active();
            RG_SET_VEC3_A( gu.fluidColor, impl->fluidColor.data );

            memcpy( impl->globalUniform[ impl->frameIndex ].contents, &gu, sizeof( gu ) );
        }

        id< MTLTexture > toPresent = impl->outColor;

        const bool presentPrev = rtPath && pInfo->presentPrevFrame && impl->pathTracer->GetPrevPresented();

        // resources by the names in RTGL1's GLSL shaders
        if( impl->rtShaders && !presentPrev )
        {
            MetalBindingTable& b = impl->bindings;
            b.SetBuffer( "GlobalUniform", impl->globalUniform[ impl->frameIndex ] );
            b.SetArgumentBuffer(
                "globalTextures_Images", impl->textures->ImageTableBuffer(), impl->textures->LiveTextures() );
            b.SetArgumentBuffer( "globalTextures_Samplers", impl->textures->SamplerTableBuffer(), {} );
            b.SetTexture( "blueNoiseTextures", impl->blueNoise );
            b.SetTexture( "renderCubemap", impl->rasterizer->SkyCubemap() );
            b.SetSampler( "renderCubemapSmplr", impl->linearClampSampler );
            impl->lights->Bind( b, impl->frameIndex );
            impl->pathTracer->BindVolumetric( b, impl->frameIndex );

            if( impl->uniform.skyType == SKY_TYPE_RASTERIZED_GEOMETRY )
            {
                const auto             tSky   = Impl::CpuProfile::Clock::now();

                // Doom 64 RT: the volumetric cloud map, marched once, read by both sky passes
                impl->pathTracer->ProcessClouds( cmd, b, impl->frameIndex, impl->uniform.cloudParams0[ 0 ] > 0.5f );
                const auto tClouds = Impl::CpuProfile::Clock::now();
                impl->rasterizer->RenderSkyCubemap( cmd, *impl->rtShaders, b );

                impl->cpu.skyCubemap += Impl::CpuProfile::Ms( tClouds, Impl::CpuProfile::Clock::now() );
                impl->cpu.clouds += Impl::CpuProfile::Ms( tSky, tClouds );
                impl->cpu.skyDraws += impl->rasterizer->DrawCount( MetalRasterizer::Type::Sky );
            }
        }

        if( presentPrev )
        {
            toPresent = impl->pathTracer->GetPrevPresented();
        }
        else if( rtPath )
        {
            impl->framebuffers->PrepareForSize( renderWidth, renderHeight, width, height );

            MetalBindingTable& b = impl->bindings;
            impl->framebuffers->Bind( b, impl->frameIndex );
            impl->scene->Bind( b );

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
                const auto tSky = Impl::CpuProfile::Clock::now();
                impl->rasterizer->DrawSkyToAlbedo(
                    cmd,
                    rasterParams,
                    impl->framebuffers->Get( FB_IMAGE_INDEX_ALBEDO, impl->frameIndex ),
                    pnext::get< RgDrawFrameSkyParams >( *pInfo ).skyViewerPosition );
                impl->cpu.skyAlbedo += Impl::CpuProfile::Ms( tSky, Impl::CpuProfile::Clock::now() );
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

            const auto tRender = Impl::CpuProfile::Clock::now();
            struct RenderTimer
            {
                Impl&                              impl;
                Impl::CpuProfile::Clock::time_point t;
                ~RenderTimer() { impl.cpu.render += Impl::CpuProfile::Ms( t, Impl::CpuProfile::Clock::now() ); }
            } renderTimer{ *impl, tRender };
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
                    .volumePostComp    = impl->uniform.volumePostComp != 0,
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
                            id< MTLBlitCommandEncoder > blit = MetalProfiler::Blit( cmd );
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
                            id< MTLComputeCommandEncoder > enc = MetalProfiler::Compute( cmd );
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
                            fi.fieldOfView          = ( camera ? camera->fovYRadians : 1.0f ) * 180.0f / float( Utils::Pi );
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
            id< MTLComputeCommandEncoder > enc = MetalProfiler::Compute( cmd );
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
            MetalProfiler::EndFrame( cmd );
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

                id< MTLRenderCommandEncoder > enc = MetalProfiler::Render( pc, pass );
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

                    id< MTLBlitCommandEncoder > blit = MetalProfiler::Blit( pc );
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

            MetalProfiler::EndFrame( pc );
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

    impl->cpu.drawFrame += Impl::CpuProfile::Ms( tDraw, Impl::CpuProfile::Clock::now() );
    if( impl->cpu.enabled && ( ++impl->cpu.frames, impl->cpu.printNow ) )
    {
        auto& c = impl->cpu;
        const double n = double( c.frames );
        debug::Warning( "Metal profile: {:.1f} fps over the last {:.1f} s",
                        n * 1000.0 / std::max( 1.0, c.sincePrint ), c.sincePrint / 1000.0 );
        debug::Warning( "Metal profile (CPU, ms per frame): waiting for a frame slot {:.2f}, StartFrame {:.2f}, "
                        "game + uploads {:.2f} (uploads {:.2f}, {:.0f} primitives), DrawFrame {:.2f} "
                        "(acceleration structures {:.2f}, path tracer encoding {:.2f}, clouds {:.2f}, "
                        "sky cubemap {:.2f}, sky to albedo {:.2f}; {:.0f} sky draws)",
                        c.waitFrame / n, c.startFrame / n, c.betweenFrames / n, c.uploads / n,
                        double( c.primitives ) / n, c.drawFrame / n, c.sceneBuild / n, c.render / n,
                        c.clouds / n, c.skyCubemap / n, c.skyAlbedo / n, double( c.skyDraws ) / n );
        if( impl->scene )
        {
            const auto& st = impl->scene->lastStats;
            debug::Warning( "Metal profile: scene: {} dynamic geometries ({} triangles, {} vertices), "
                            "{} static geometries, {} TLAS instances, {} geometry infos",
                            st.dynamicGeometries, st.dynamicTriangles, st.dynamicVertices,
                            st.staticGeometries, st.tlasInstances, st.geometryInfos );
        }
        c.waitFrame = c.startFrame = c.betweenFrames = c.uploads = c.sceneBuild = c.render = c.drawFrame = 0;
        c.clouds = c.skyCubemap = c.skyAlbedo = 0;
        c.skyDraws = 0;
        c.primitives = 0;
        c.frames     = 0;
    }
    else if( !impl->cpu.enabled )
    {
        impl->cpu = {};
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
    struct UploadTimer
    {
        Impl&                               impl;
        Impl::CpuProfile::Clock::time_point t{ Impl::CpuProfile::Clock::now() };
        ~UploadTimer()
        {
            impl.cpu.uploads += Impl::CpuProfile::Ms( t, Impl::CpuProfile::Clock::now() );
            impl.cpu.primitives++;
        }
    } uploadTimer{ *impl };

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
