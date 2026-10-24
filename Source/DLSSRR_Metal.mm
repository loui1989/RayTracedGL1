// Copyright (c) 2026 Quake RT / Doom 64 RT for macOS contributors
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

// macOS: the "Ray Reconstruction" slot is Apple's MetalFX temporal denoised scaler (macOS 26):
// it denoises the noisy path traced image and upscales it, in place of RTGL1's own denoiser
// (A-SVGF) + the MetalFX temporal scaler. It gets the same inputs as DLSS-RR does
// (DLSSRR.cpp): the noisy colour composed by CmNoisyCompose and its guides (diffuse / specular
// albedo, world space normal + roughness), depth, motion, and the translucent raster layer.
// Used with the MetalFX upscaler (the FSR2 slot), see RenderResolutionHelper.

#include "DLSSRR.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <simd/simd.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "Common.h"
#include "Framebuffers.h"
#include "MetalBridge.h"
#include "MetalNative.h"
#include "RenderResolutionHelper.h"

namespace
{
using FI = RTGL1::FramebufferImageIndex;

struct RrState
{
    VkDevice                    vkDevice = VK_NULL_HANDLE;
    PFN_vkExportMetalObjectsEXT exportFn = nullptr;

    id< MTLDevice >               device   = nil;
    id< MTLComputePipelineState > prepPipe = nil;
    id< MTLComputePipelineState > copyPipe = nil;

    float minScale = 1.0f, maxScale = 1.0f;

    id        scaler = nil; // id< MTLFXTemporalDenoisedScaler >
    id< MTLTexture > color = nil, depth = nil, motion = nil;
    id< MTLTexture > diffAlbedo = nil, specAlbedo = nil, normal = nil, roughness = nil;
    id< MTLTexture > overlay = nil, out = nil, reactive = nil;

    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    bool     reset          = true;
    bool     initOk         = false;
    uint32_t loggedFailures = 0;
};

RrState g;

const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct RrPrepParams
{
    uint useOverlay;
    uint useDisocc;
};

// RTGL1's framebuffers -> the formats the denoiser was created with (sanitized)
kernel void rr_prep( texture2d< float, access::read >  srcColor      [[texture(0)]],
                     texture2d< float, access::read >  srcDepth      [[texture(1)]],
                     texture2d< float, access::read >  srcMotion     [[texture(2)]],
                     texture2d< float, access::read >  srcDiffAlbedo [[texture(3)]],
                     texture2d< float, access::read >  srcNormalRough[[texture(4)]],
                     texture2d< float, access::read >  srcSpecAlbedo [[texture(5)]],
                     texture2d< float, access::read >  srcOverlay    [[texture(6)]],
                     texture2d< float, access::write > color         [[texture(7)]],
                     texture2d< float, access::write > depth         [[texture(8)]],
                     texture2d< float, access::write > motion        [[texture(9)]],
                     texture2d< float, access::write > diffAlbedo    [[texture(10)]],
                     texture2d< float, access::write > specAlbedo    [[texture(11)]],
                     texture2d< float, access::write > normal        [[texture(12)]],
                     texture2d< float, access::write > roughness     [[texture(13)]],
                     texture2d< float, access::write > overlay       [[texture(14)]],
                     texture2d< float, access::read >  srcDisocc     [[texture(15)]],
                     texture2d< float, access::write > reactive      [[texture(16)]],
                     constant RrPrepParams& prm                       [[buffer(0)]],
                     uint2 p [[thread_position_in_grid]] )
{
    if( p.x >= color.get_width() || p.y >= color.get_height() )
        return;

    float3 c = srcColor.read( p ).rgb;
    c        = select( c, float3( 0.0 ), !isfinite( c ) );
    color.write( float4( clamp( c, 0.0f, 65000.0f ), 1.0 ), p );

    float d = srcDepth.read( p ).r;
    depth.write( float4( isfinite( d ) ? clamp( d, 0.0f, 1.0f ) : 1.0f ), p );

    float2 m = srcMotion.read( p ).rg;
    m        = select( m, float2( 0.0 ), !isfinite( m ) );
    motion.write( float4( clamp( m, -2.0f, 2.0f ), 0.0, 0.0 ), p );

    float3 da = srcDiffAlbedo.read( p ).rgb;
    da        = select( da, float3( 0.5 ), !isfinite( da ) );
    diffAlbedo.write( float4( saturate( da ), 1.0 ), p );

    float3 sa = srcSpecAlbedo.read( p ).rgb;
    sa        = select( sa, float3( 0.04 ), !isfinite( sa ) );
    specAlbedo.write( float4( saturate( sa ), 1.0 ), p );

    float4 nr = srcNormalRough.read( p );
    float3 n  = nr.xyz;
    float  l  = length( n );
    n         = ( isfinite( l ) && l > 1e-4 ) ? n / l : float3( 0.0, 0.0, 1.0 );
    normal.write( float4( n, 0.0 ), p );
    roughness.write( float4( isfinite( nr.w ) ? saturate( nr.w ) : 1.0 ), p );

    float4 o = prm.useOverlay != 0 ? srcOverlay.read( p ) : float4( 0.0 );
    o        = select( o, float4( 0.0 ), !isfinite( o ) );
    overlay.write( float4( max( o.rgb, 0.0f ), saturate( o.a ) ), p );

    // CmNoisyCompose's mask of sharp lighting changes (muzzle flashes, explosions, flickering
    // lights): no history there, as DLSS-RR's disocclusion mask does
    float r = prm.useDisocc != 0 ? srcDisocc.read( p ).r : 0.0;
    reactive.write( float4( ( isfinite( r ) && r > 0.0 ) ? 1.0 : 0.0 ), p );
}

struct RrCopyParams
{
    uint   demod;
    uint   filter; // as rt_rr_demod_filter: 0 bilinear, 1 Catmull-Rom, 2 nearest
    float2 renderSize;
    float2 jitter;
};

float3 SampleCatmullRom( texture2d< float > t, sampler s, float2 uv, float2 size )
{
    float2 samplePos = uv * size;
    float2 texPos1   = floor( samplePos - 0.5 ) + 0.5;
    float2 f         = samplePos - texPos1;
    float2 w0        = f * ( -0.5 + f * ( 1.0 - 0.5 * f ) );
    float2 w1        = 1.0 + f * f * ( -2.5 + 1.5 * f );
    float2 w2        = f * ( 0.5 + f * ( 2.0 - 1.5 * f ) );
    float2 w3        = f * f * ( -0.5 + 0.5 * f );
    float2 w12       = w1 + w2;
    float2 p0        = ( texPos1 - 1.0 ) / size;
    float2 p3        = ( texPos1 + 2.0 ) / size;
    float2 p12       = ( texPos1 + w2 / w12 ) / size;
    float3 acc       = float3( 0.0 );
    acc += t.sample( s, float2( p0.x, p0.y ), level( 0 ) ).rgb * w0.x * w0.y;
    acc += t.sample( s, float2( p12.x, p0.y ), level( 0 ) ).rgb * w12.x * w0.y;
    acc += t.sample( s, float2( p3.x, p0.y ), level( 0 ) ).rgb * w3.x * w0.y;
    acc += t.sample( s, float2( p0.x, p12.y ), level( 0 ) ).rgb * w0.x * w12.y;
    acc += t.sample( s, float2( p12.x, p12.y ), level( 0 ) ).rgb * w12.x * w12.y;
    acc += t.sample( s, float2( p3.x, p12.y ), level( 0 ) ).rgb * w3.x * w12.y;
    acc += t.sample( s, float2( p0.x, p3.y ), level( 0 ) ).rgb * w0.x * w3.y;
    acc += t.sample( s, float2( p12.x, p3.y ), level( 0 ) ).rgb * w12.x * w3.y;
    acc += t.sample( s, float2( p3.x, p3.y ), level( 0 ) ).rgb * w3.x * w3.y;
    return max( acc, float3( 0.0 ) );
}

// into RTGL1's upscaled framebuffer; with demodulation, the denoised lighting is multiplied by the
// albedo / texel factor CmNoisyCompose divided out (nearest texel at the unjittered position, as
// CmRrPostExposure's rrDemodFilter 2: the pixel art stays crisp, it never goes through the network)
kernel void rr_copy_out( texture2d< float, access::read >  src         [[texture(0)]],
                         texture2d< float, access::write > dst         [[texture(1)]],
                         texture2d< float >                demodFactor [[texture(2)]],
                         constant RrCopyParams& prm                    [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]] )
{
    if( gid.x >= src.get_width() || gid.y >= src.get_height() )
        return;
    float3 c = src.read( gid ).rgb;
    c        = select( c, float3( 0.0 ), !isfinite( c ) );
    c        = max( c, 0.0f );
    if( prm.demod != 0 )
    {
        constexpr sampler lin( filter::linear, address::clamp_to_edge, coord::normalized );
        // in texture UV (the framebuffer can be larger than the render size)
        float2 texSize = float2( demodFactor.get_width(), demodFactor.get_height() );
        float2 uv      = ( ( float2( gid ) + 0.5 ) / float2( src.get_width(), src.get_height() ) -
                      prm.jitter / prm.renderSize ) *
                    prm.renderSize / texSize;
        float3 m;
        if( prm.filter == 2 )
        {
            int2 nn = clamp( int2( floor( uv * texSize ) ), int2( 0 ), int2( prm.renderSize ) - 1 );
            m       = demodFactor.read( uint2( nn ) ).rgb;
        }
        else if( prm.filter == 1 )
        {
            m = SampleCatmullRom( demodFactor, lin, uv, texSize );
        }
        else
        {
            m = demodFactor.sample( lin, uv, level( 0 ) ).rgb;
        }
        m         = select( m, float3( 1.0 ), !isfinite( m ) );
        c        *= max( m, 0.0f );
    }
    dst.write( float4( c, 1.0 ), gid );
}
)";

void Log( const char* msg )
{
    fprintf( stderr, "[MetalFX denoiser] %s\n", msg );
}

bool InitDevice( VkDevice vkDevice )
{
    g.vkDevice = vkDevice;
    g.exportFn = reinterpret_cast< PFN_vkExportMetalObjectsEXT >(
        vkGetDeviceProcAddr( vkDevice, "vkExportMetalObjectsEXT" ) );
    if( !g.exportFn )
    {
        return false;
    }

    VkExportMetalDeviceInfoEXT  dInfo = { VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT };
    VkExportMetalObjectsInfoEXT info  = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                        = &dInfo;
    g.exportFn( vkDevice, &info );
    g.device = dInfo.mtlDevice;
    if( !g.device )
    {
        return false;
    }

    if( @available( macOS 26.0, * ) )
    {
        if( ![MTLFXTemporalDenoisedScalerDescriptor supportsDevice:g.device] )
        {
            Log( "not supported by this GPU" );
            return false;
        }
        g.minScale = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMinScaleForDevice:g.device];
        g.maxScale = [MTLFXTemporalDenoisedScalerDescriptor supportedInputContentMaxScaleForDevice:g.device];
    }
    else
    {
        Log( "needs macOS 26" );
        return false;
    }

    NSError*         err = nil;
    id< MTLLibrary > lib = [g.device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                                                  options:nil
                                                    error:&err];
    if( !lib )
    {
        fprintf( stderr,
                 "[MetalFX denoiser] shader compile failed: %s\n",
                 err ? err.localizedDescription.UTF8String : "?" );
        return false;
    }
    g.prepPipe = [g.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rr_prep"] error:&err];
    g.copyPipe = [g.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rr_copy_out"]
                                                         error:&err];
    if( !g.prepPipe || !g.copyPipe )
    {
        fprintf( stderr,
                 "[MetalFX denoiser] pipeline creation failed: %s\n",
                 err ? err.localizedDescription.UTF8String : "?" );
        return false;
    }

    fprintf( stderr,
             "[MetalFX denoiser] available, upscaling %.2fx .. %.2fx\n",
             double( g.minScale ),
             double( g.maxScale ) );
    return true;
}

id< MTLTexture > TextureOf( VkImage image )
{
    VkExportMetalTextureInfoEXT t    = { VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT };
    t.image                          = image;
    t.plane                          = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT info = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                       = &t;
    g.exportFn( g.vkDevice, &info );
    return t.mtlTexture;
}

void ReleaseScaler()
{
    g.scaler     = nil;
    g.color      = nil;
    g.depth      = nil;
    g.motion     = nil;
    g.diffAlbedo = nil;
    g.specAlbedo = nil;
    g.normal     = nil;
    g.roughness  = nil;
    g.overlay    = nil;
    g.out        = nil;
    g.reactive   = nil;
    g.renderW = g.renderH = g.outW = g.outH = 0;
}

bool EnsureScaler( uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh )
{
    if( g.scaler && g.renderW == rw && g.renderH == rh && g.outW == ow && g.outH == oh )
    {
        return true;
    }
    ReleaseScaler();

    if( @available( macOS 26.0, * ) )
    {
        constexpr MTLPixelFormat albedoFormat    = MTLPixelFormatRGBA8Unorm;
        constexpr MTLPixelFormat normalFormat    = MTLPixelFormatRGBA16Float;
        constexpr MTLPixelFormat roughnessFormat = MTLPixelFormatR16Float;
        constexpr MTLPixelFormat colorFormat     = MTLPixelFormatRGBA16Float;

        MTLFXTemporalDenoisedScalerDescriptor* d = [MTLFXTemporalDenoisedScalerDescriptor new];
        d.colorTextureFormat                     = colorFormat;
        d.depthTextureFormat                     = MTLPixelFormatR32Float;
        d.motionTextureFormat                    = MTLPixelFormatRG16Float;
        d.diffuseAlbedoTextureFormat             = albedoFormat;
        d.specularAlbedoTextureFormat            = albedoFormat;
        d.normalTextureFormat                    = normalFormat;
        d.roughnessTextureFormat                 = roughnessFormat;
        d.outputTextureFormat                    = colorFormat;
        d.inputWidth                             = rw;
        d.inputHeight                            = rh;
        d.outputWidth                            = ow;
        d.outputHeight                           = oh;
        // the colour is already exposed (CmPrepareFinal)
        d.autoExposureEnabled                    = NO;
        // translucent sprites / particles: drawn into a layer, composited over the denoised image
        d.transparencyOverlayTextureEnabled      = YES;
        d.transparencyOverlayTextureFormat       = colorFormat;
        // where the lighting changed sharply: CmNoisyCompose's disocclusion mask
        d.reactiveMaskTextureEnabled             = YES;
        d.reactiveMaskTextureFormat              = MTLPixelFormatR8Unorm;

        id< MTLFXTemporalDenoisedScaler > scaler = [d newTemporalDenoisedScalerWithDevice:g.device];
        if( !scaler )
        {
            fprintf( stderr, "[MetalFX denoiser] can't create a %ux%u -> %ux%u denoiser\n", rw, rh, ow, oh );
            return false;
        }

        auto makeTex = [ & ]( MTLPixelFormat fmt, uint32_t w, uint32_t h, MTLTextureUsage usage ) {
            MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                                          width:w
                                                                                         height:h
                                                                                      mipmapped:NO];
            td.storageMode           = MTLStorageModePrivate;
            td.usage                 = usage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            return [g.device newTextureWithDescriptor:td];
        };
        g.color      = makeTex( colorFormat, rw, rh, scaler.colorTextureUsage );
        g.depth      = makeTex( MTLPixelFormatR32Float, rw, rh, scaler.depthTextureUsage );
        g.motion     = makeTex( MTLPixelFormatRG16Float, rw, rh, scaler.motionTextureUsage );
        g.diffAlbedo = makeTex( albedoFormat, rw, rh, scaler.diffuseAlbedoTextureUsage );
        g.specAlbedo = makeTex( albedoFormat, rw, rh, scaler.specularAlbedoTextureUsage );
        g.normal     = makeTex( normalFormat, rw, rh, scaler.normalTextureUsage );
        g.roughness  = makeTex( roughnessFormat, rw, rh, scaler.roughnessTextureUsage );
        g.overlay    = makeTex( colorFormat, rw, rh, scaler.transparencyOverlayTextureUsage );
        g.out        = makeTex( colorFormat, ow, oh, scaler.outputTextureUsage );
        g.reactive   = makeTex( MTLPixelFormatR8Unorm, rw, rh, scaler.reactiveMaskTextureUsage );
        if( !g.color || !g.depth || !g.motion || !g.diffAlbedo || !g.specAlbedo || !g.normal || !g.roughness ||
            !g.overlay || !g.out || !g.reactive )
        {
            Log( "texture allocation failed" );
            ReleaseScaler();
            return false;
        }

        g.scaler  = scaler;
        g.renderW = rw;
        g.renderH = rh;
        g.outW    = ow;
        g.outH    = oh;
        g.reset   = true;
        fprintf( stderr, "[MetalFX denoiser] %ux%u -> %ux%u\n", rw, rh, ow, oh );
        return true;
    }
    return false;
}
}

auto RTGL1::DLSSRR::MakeInstance( VkDevice device, const DLSS2* ) -> std::shared_ptr< DLSSRR >
{
    auto inst = std::make_shared< DLSSRR >( device, false );
    if( !inst->Valid() )
    {
        return {};
    }
    return inst;
}

RTGL1::DLSSRR::DLSSRR( VkDevice device, bool ) : m_device{ device }
{
    @autoreleasepool
    {
        if( RTGL1::MetalNative::Enabled() )
        {
            m_initialized = InitDevice( device );
            g.initOk      = m_initialized;
        }
    }
}

RTGL1::DLSSRR::~DLSSRR()
{
    Destroy();
}

void RTGL1::DLSSRR::Destroy()
{
    @autoreleasepool
    {
        ReleaseScaler();
        g.prepPipe = nil;
        g.copyPipe = nil;
        g.device   = nil;
        g.initOk   = false;
    }
    m_initialized = false;
}

bool RTGL1::DLSSRR::Valid() const
{
    return m_initialized;
}

bool RTGL1::DLSSRR::SupportsScale( uint32_t renderWidth, uint32_t renderHeight, uint32_t outWidth, uint32_t outHeight ) const
{
    if( !m_initialized || renderWidth == 0 || renderHeight == 0 )
    {
        return false;
    }
    const float sx = float( outWidth ) / float( renderWidth );
    const float sy = float( outHeight ) / float( renderHeight );
    return std::min( sx, sy ) >= g.minScale - 0.01f && std::max( sx, sy ) <= g.maxScale + 0.01f;
}

auto RTGL1::DLSSRR::GetOptimalSettings( uint32_t userWidth, uint32_t userHeight, RgRenderResolutionMode mode ) const
    -> std::pair< uint32_t, uint32_t >
{
    float div = 1.7f;
    switch( mode )
    {
        case RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE: div = 3.0f; break;
        case RG_RENDER_RESOLUTION_MODE_PERFORMANCE: div = 2.0f; break;
        case RG_RENDER_RESOLUTION_MODE_BALANCED: div = 1.7f; break;
        case RG_RENDER_RESOLUTION_MODE_QUALITY: div = 1.5f; break;
        case RG_RENDER_RESOLUTION_MODE_NATIVE_AA: div = 1.0f; break;
        default: break;
    }
    div = std::clamp( div, std::max( 1.0f, g.minScale ), std::max( 1.0f, g.maxScale ) );
    return { uint32_t( float( userWidth ) / div ), uint32_t( float( userHeight ) / div ) };
}

auto RTGL1::DLSSRR::Apply( VkCommandBuffer               cmd,
                           uint32_t                      frameIndex,
                           Framebuffers&                 framebuffers,
                           const RenderResolutionHelper& renderResolution,
                           RgFloat2D                     jitterOffset,
                           double                        timeDelta,
                           bool                          resetAccumulation,
                           bool, // specHitDistEnabled: MetalFX derives it itself
                           bool                          disoccMaskEnabled,
                           bool, // exposureTexEnabled: the colour is always exposed on macOS
                           bool                          transparencyLayerEnabled,
                           const float*                  worldToViewMatrix16,
                           const float*                  viewToClipMatrix16 ) -> FramebufferImageIndex
{
    @autoreleasepool
    {
        if( !m_initialized || !RTGL1::MetalNative::IsNativeCmd( cmd ) )
        {
            if( g.loggedFailures++ < 5 )
            {
                Log( "not running natively on Metal, the frame isn't denoised" );
            }
            return FI::FB_IMAGE_INDEX_FINAL;
        }

        const ResolutionState rs = renderResolution.GetResolutionState();
        if( !EnsureScaler( rs.renderWidth, rs.renderHeight, rs.upscaledWidth, rs.upscaledHeight ) )
        {
            return FI::FB_IMAGE_INDEX_FINAL;
        }

        auto tex = [ & ]( FI i ) { return TextureOf( framebuffers.GetImage( i, frameIndex ) ); };

        id< MTLTexture > srcColor   = tex( FI::FB_IMAGE_INDEX_FINAL );
        id< MTLTexture > srcDepth   = tex( FI::FB_IMAGE_INDEX_DEPTH_NDC );
        id< MTLTexture > srcMotion  = tex( FI::FB_IMAGE_INDEX_MOTION_DLSS );
        // the guides staged by CmNoisyCompose, as for DLSS-RR
        id< MTLTexture > srcDiff    = tex( FI::FB_IMAGE_INDEX_DIFF_COLOR_HISTORY );
        id< MTLTexture > srcNormal  = tex( FI::FB_IMAGE_INDEX_DIFF_PING_COLOR_AND_VARIANCE );
        id< MTLTexture > srcSpec    = tex( FI::FB_IMAGE_INDEX_DIFF_PONG_COLOR_AND_VARIANCE );
        id< MTLTexture > srcOverlay = tex( FI::FB_IMAGE_INDEX_RR_TRANSPARENCY );
        id< MTLTexture > outputVk   = tex( FI::FB_IMAGE_INDEX_UPSCALED_PONG );
        id< MTLTexture > demodTex   = tex( FI::FB_IMAGE_INDEX_RR_DEMOD_FACTOR );
        id< MTLTexture > disoccTex  = tex( FI::FB_IMAGE_INDEX_RR_DISOCCLUSION );
        if( !srcColor || !srcDepth || !srcMotion || !srcDiff || !srcNormal || !srcSpec || !srcOverlay || !outputVk ||
            !demodTex || !disoccTex )
        {
            if( g.loggedFailures++ < 5 )
            {
                Log( "couldn't get Metal textures for the denoiser inputs" );
            }
            return FI::FB_IMAGE_INDEX_FINAL;
        }

        if( resetAccumulation )
        {
            g.reset = true;
        }

        simd_float4x4 view = matrix_identity_float4x4, proj = matrix_identity_float4x4;
        const bool    haveMatrices = worldToViewMatrix16 && viewToClipMatrix16;
        if( haveMatrices )
        {
            static_assert( sizeof( view ) == sizeof( float ) * 16 );
            memcpy( &view, worldToViewMatrix16, sizeof( view ) );
            memcpy( &proj, viewToClipMatrix16, sizeof( proj ) );
            // RTGL1's projection is for Vulkan's clip space (Y down), Metal's is Y up
            for( int c = 0; c < 4; c++ )
            {
                proj.columns[ c ][ 1 ] = -proj.columns[ c ][ 1 ];
            }
        }

        const uint32_t prepPrm[ 2 ] = { transparencyLayerEnabled ? 1u : 0u, disoccMaskEnabled ? 1u : 0u };

        struct
        {
            uint32_t demod;
            uint32_t filter;
            float    renderSize[ 2 ];
            float    jitter[ 2 ];
        } copyPrm = {
            m_demod ? 1u : 0u,
            m_demodFilter,
            { float( rs.renderWidth ), float( rs.renderHeight ) },
            { jitterOffset.data[ 0 ], jitterOffset.data[ 1 ] },
        };
        const RgFloat2D jitter    = jitterOffset;
        (void)timeDelta;

        RTGL1::Metal::RecordExternal( cmd, [ = ]( id< MTLCommandBuffer > cb, id< MTLFence > fence ) {
            if( @available( macOS 26.0, * ) )
            {
                // 1. inputs into the formats the denoiser was created with
                {
                    id< MTLComputeCommandEncoder > enc = [cb computeCommandEncoder];
                    enc.label                          = @"MetalFX denoiser input prep";
                    if( fence )
                    {
                        [enc waitForFence:fence];
                    }
                    [enc setComputePipelineState:g.prepPipe];
                    id< MTLTexture > textures[] = {
                        srcColor,  srcDepth,  srcMotion,    srcDiff,      srcNormal,    srcSpec,
                        srcOverlay, g.color, g.depth,       g.motion,     g.diffAlbedo, g.specAlbedo,
                        g.normal,   g.roughness, g.overlay,  disoccTex,    g.reactive,
                    };
                    for( NSUInteger i = 0; i < std::size( textures ); i++ )
                    {
                        [enc setTexture:textures[ i ] atIndex:i];
                    }
                    [enc setBytes:&prepPrm length:sizeof( prepPrm ) atIndex:0];
                    [enc dispatchThreads:MTLSizeMake( rs.renderWidth, rs.renderHeight, 1 )
                        threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
                    [enc endEncoding];
                }

                // 2. denoise + upscale; the same conventions as the temporal scaler (FSR2_Metal.mm)
                {
                    id< MTLFXTemporalDenoisedScaler > ds = g.scaler;
                    ds.colorTexture                      = g.color;
                    ds.depthTexture                      = g.depth;
                    ds.motionTexture                     = g.motion;
                    ds.diffuseAlbedoTexture              = g.diffAlbedo;
                    ds.specularAlbedoTexture             = g.specAlbedo;
                    ds.normalTexture                     = g.normal;
                    ds.roughnessTexture                  = g.roughness;
                    ds.transparencyOverlayTexture        = g.overlay;
                    ds.reactiveMaskTexture               = g.reactive;
                    ds.outputTexture                     = g.out;
                    ds.motionVectorScaleX                = float( rs.renderWidth );
                    ds.motionVectorScaleY                = float( rs.renderHeight );
                    ds.jitterOffsetX                     = -jitter.data[ 0 ];
                    ds.jitterOffsetY                     = -jitter.data[ 1 ];
                    ds.preExposure                       = 1.0f;
                    ds.depthReversed                     = NO;
                    if( haveMatrices )
                    {
                        ds.worldToViewMatrix = view;
                        ds.viewToClipMatrix  = proj;
                    }
                    ds.shouldResetHistory = g.reset;
                    [ds encodeToCommandBuffer:cb];
                    g.reset = false;
                }

                // 3. into RTGL1's upscaled framebuffer
                {
                    id< MTLComputeCommandEncoder > enc = [cb computeCommandEncoder];
                    enc.label                          = @"MetalFX denoiser output copy";
                    [enc setComputePipelineState:g.copyPipe];
                    [enc setTexture:g.out atIndex:0];
                    [enc setTexture:outputVk atIndex:1];
                    [enc setTexture:demodTex atIndex:2];
                    [enc setBytes:&copyPrm length:sizeof( copyPrm ) atIndex:0];
                    [enc dispatchThreads:MTLSizeMake( rs.upscaledWidth, rs.upscaledHeight, 1 )
                        threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
                    if( fence )
                    {
                        [enc updateFence:fence];
                    }
                    [enc endEncoding];
                }
            }
        } );
    }

    return FI::FB_IMAGE_INDEX_UPSCALED_PONG;
}

#endif // __APPLE__
