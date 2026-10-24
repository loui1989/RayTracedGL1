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

// macOS frame generation: MetalFX frame interpolation (see FrameInterp_Metal.h)

#include "FrameInterp_Metal.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "MetalBridge.h"
#include "MetalNative.h"

namespace
{
constexpr MTLPixelFormat COLOR_FORMAT = MTLPixelFormatRGBA16Float;

struct FiState
{
    VkDevice                    vkDevice = VK_NULL_HANDLE;
    PFN_vkExportMetalObjectsEXT exportFn = nullptr;

    bool initTried = false;
    bool initOk    = false;

    id< MTLDevice >               device      = nil;
    id< MTLComputePipelineState > prepPipe    = nil;
    id< MTLComputePipelineState > lowresPipe  = nil;
    id< MTLLibrary >              library     = nil;
    id< MTLRenderPipelineState >  presentPipe = nil;
    MTLPixelFormat                presentFormat = MTLPixelFormatInvalid;
    id< MTLSamplerState >         sampler     = nil;

    id               interpolator = nil; // id< MTLFXFrameInterpolator >
    uint32_t         renderW = 0, renderH = 0, outW = 0, outH = 0;
    // the frame without HUD, current and previous
    id< MTLTexture > color[ 2 ] = {};
    // the actual frame (with HUD) and the generated one, per 'cur': presented after the frame
    id< MTLTexture > real[ 2 ] = {};
    id< MTLTexture > generated[ 2 ] = {};
    id< MTLTexture > ui     = nil;
    id< MTLTexture > depth  = nil;
    id< MTLTexture > motion = nil;
    uint32_t         cur        = 0;
    bool             hasHistory = false;

    // for the next vkQueuePresentKHR
    id< MTLTexture > pendingReal      = nil;
    id< MTLTexture > pendingGenerated = nil;
    bool             lateDrawables    = false;

    // pacing: the generated frame is shown for a half of the frame time
    double lastPresent    = 0;
    double frameInterval  = 1.0 / 60.0;
    bool   loggedFirst    = false;
};

FiState g;

double Now()
{
    using namespace std::chrono;
    return duration< double >( steady_clock::now().time_since_epoch() ).count();
}

const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct FiPrepParams
{
    uint hasHud;
};

float3 SrgbEncode( float3 c )
{
    c = saturate( c );
    return select( 1.055 * pow( c, float3( 1.0 / 2.4 ) ) - 0.055, c * 12.92, c <= 0.0031308 );
}

// the frame without HUD (linear) -> sRGB encoded, as RTGL1 writes the swapchain;
// the HUD (premultiplied, gamma) -> the UI texture, and on top of the frame -> the actual frame
kernel void fi_prep( texture2d< float, access::read >  rendered [[texture(0)]],
                     texture2d< float, access::read >  hud      [[texture(1)]],
                     texture2d< float, access::write > color    [[texture(2)]],
                     texture2d< float, access::write > ui       [[texture(3)]],
                     texture2d< float, access::write > real     [[texture(4)]],
                     constant FiPrepParams& prm                 [[buffer(0)]],
                     uint2 p [[thread_position_in_grid]] )
{
    if( p.x >= color.get_width() || p.y >= color.get_height() )
        return;

    float3 c = ( p.x < rendered.get_width() && p.y < rendered.get_height() ) ? rendered.read( p ).rgb : float3( 0.0 );
    c        = select( c, float3( 0.0 ), !isfinite( c ) );
    float3 s = SrgbEncode( c );

    float4 h = float4( 0.0 );
    if( prm.hasHud != 0 && p.x < hud.get_width() && p.y < hud.get_height() )
    {
        h   = hud.read( p );
        h   = select( h, float4( 0.0 ), !isfinite( h ) );
        h.a = saturate( h.a );
    }

    color.write( float4( s, 1.0 ), p );
    ui.write( h, p );
    real.write( float4( saturate( h.rgb + ( 1.0 - h.a ) * s ), 1.0 ), p );
}

// render resolution inputs, sanitized
kernel void fi_lowres( texture2d< float, access::read >  srcDepth  [[texture(0)]],
                       texture2d< float, access::read >  srcMotion [[texture(1)]],
                       texture2d< float, access::write > depth     [[texture(2)]],
                       texture2d< float, access::write > motion    [[texture(3)]],
                       uint2 p [[thread_position_in_grid]] )
{
    if( p.x >= depth.get_width() || p.y >= depth.get_height() )
        return;
    float d = srcDepth.read( p ).r;
    depth.write( float4( isfinite( d ) ? clamp( d, 0.0f, 1.0f ) : 1.0f ), p );
    float2 m = srcMotion.read( p ).rg;
    m        = select( m, float2( 0.0 ), !isfinite( m ) );
    motion.write( float4( clamp( m, -2.0f, 2.0f ), 0.0, 0.0 ), p );
}

struct VOut
{
    float4 pos [[position]];
    float2 uv;
};

vertex VOut fi_present_vs( uint vid [[vertex_id]] )
{
    float2 p = float2( float( ( vid << 1 ) & 2 ), float( vid & 2 ) );
    VOut   o;
    o.pos = float4( p * 2.0 - 1.0, 0.0, 1.0 );
    o.uv  = float2( p.x, 1.0 - p.y );
    return o;
}

// already encoded: into the (non-sRGB) drawable as is
fragment float4 fi_present_fs( VOut in [[stage_in]],
                               texture2d< float > src [[texture(0)]],
                               sampler smp [[sampler(0)]] )
{
    return float4( src.sample( smp, in.uv ).rgb, 1.0 );
}
)";

void Log( const char* msg )
{
    fprintf( stderr, "[MetalFX frame interpolation] %s\n", msg );
}

bool Init( VkDevice vkDevice )
{
    if( g.initTried && g.vkDevice == vkDevice )
    {
        return g.initOk;
    }
    g          = FiState{};
    g.initTried = true;
    g.vkDevice  = vkDevice;

    if( !RTGL1::MetalNative::Enabled() )
    {
        return false;
    }
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
        if( ![MTLFXFrameInterpolatorDescriptor supportsDevice:g.device] )
        {
            Log( "not supported by this GPU" );
            return false;
        }
    }
    else
    {
        Log( "needs macOS 26" );
        return false;
    }

    NSError* err = nil;
    g.library    = [g.device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&err];
    if( !g.library )
    {
        fprintf( stderr,
                 "[MetalFX frame interpolation] shader compile failed: %s\n",
                 err ? err.localizedDescription.UTF8String : "?" );
        return false;
    }
    g.prepPipe   = [g.device newComputePipelineStateWithFunction:[g.library newFunctionWithName:@"fi_prep"] error:&err];
    g.lowresPipe = [g.device newComputePipelineStateWithFunction:[g.library newFunctionWithName:@"fi_lowres"]
                                                            error:&err];
    if( !g.prepPipe || !g.lowresPipe )
    {
        fprintf( stderr,
                 "[MetalFX frame interpolation] pipeline creation failed: %s\n",
                 err ? err.localizedDescription.UTF8String : "?" );
        return false;
    }

    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter             = MTLSamplerMinMagFilterLinear;
    sd.magFilter             = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode          = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode          = MTLSamplerAddressModeClampToEdge;
    g.sampler                = [g.device newSamplerStateWithDescriptor:sd];

    Log( "available" );
    g.initOk = true;
    return true;
}

id< MTLTexture > TextureOf( VkImage image )
{
    if( image == VK_NULL_HANDLE )
    {
        return nil;
    }
    VkExportMetalTextureInfoEXT t    = { VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT };
    t.image                          = image;
    t.plane                          = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT info = { VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT };
    info.pNext                       = &t;
    g.exportFn( g.vkDevice, &info );
    return t.mtlTexture;
}

void ReleaseInterpolator()
{
    g.interpolator = nil;
    g.color[ 0 ] = g.color[ 1 ] = nil;
    g.real[ 0 ] = g.real[ 1 ] = nil;
    g.generated[ 0 ] = g.generated[ 1 ] = nil;
    g.ui         = nil;
    g.depth      = nil;
    g.motion     = nil;
    g.hasHistory = false;
    g.renderW = g.renderH = g.outW = g.outH = 0;
}

bool EnsureInterpolator( uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh )
{
    if( g.interpolator && g.renderW == rw && g.renderH == rh && g.outW == ow && g.outH == oh )
    {
        return true;
    }
    ReleaseInterpolator();

    if( @available( macOS 26.0, * ) )
    {
        MTLFXFrameInterpolatorDescriptor* d = [MTLFXFrameInterpolatorDescriptor new];
        d.colorTextureFormat                = COLOR_FORMAT;
        d.outputTextureFormat               = COLOR_FORMAT;
        d.uiTextureFormat                   = COLOR_FORMAT;
        d.depthTextureFormat                = MTLPixelFormatR32Float;
        d.motionTextureFormat               = MTLPixelFormatRG16Float;
        d.inputWidth                        = rw;
        d.inputHeight                       = rh;
        d.outputWidth                       = ow;
        d.outputHeight                      = oh;

        id< MTLFXFrameInterpolator > fi = [d newFrameInterpolatorWithDevice:g.device];
        if( !fi )
        {
            fprintf( stderr, "[MetalFX frame interpolation] can't create a %ux%u -> %ux%u interpolator\n", rw, rh, ow, oh );
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
        for( int i = 0; i < 2; i++ )
        {
            g.color[ i ]     = makeTex( COLOR_FORMAT, ow, oh, fi.colorTextureUsage );
            g.real[ i ]      = makeTex( COLOR_FORMAT, ow, oh, 0 );
            g.generated[ i ] = makeTex( COLOR_FORMAT, ow, oh, fi.outputTextureUsage );
        }
        g.ui     = makeTex( COLOR_FORMAT, ow, oh, fi.uiTextureUsage );
        g.depth  = makeTex( MTLPixelFormatR32Float, rw, rh, 0 );
        g.motion = makeTex( MTLPixelFormatRG16Float, rw, rh, 0 );
        if( !g.color[ 0 ] || !g.color[ 1 ] || !g.real[ 0 ] || !g.real[ 1 ] || !g.generated[ 0 ] ||
            !g.generated[ 1 ] || !g.ui || !g.depth || !g.motion )
        {
            Log( "texture allocation failed" );
            ReleaseInterpolator();
            return false;
        }

        g.interpolator = fi;
        g.renderW      = rw;
        g.renderH      = rh;
        g.outW         = ow;
        g.outH         = oh;
        g.hasHistory   = false;
        fprintf( stderr, "[MetalFX frame interpolation] %ux%u -> %ux%u\n", rw, rh, ow, oh );
        return true;
    }
    return false;
}

bool EnsurePresentPipeline( MTLPixelFormat format )
{
    if( g.presentPipe && g.presentFormat == format )
    {
        return true;
    }
    MTLRenderPipelineDescriptor* rpd      = [MTLRenderPipelineDescriptor new];
    rpd.vertexFunction                    = [g.library newFunctionWithName:@"fi_present_vs"];
    rpd.fragmentFunction                  = [g.library newFunctionWithName:@"fi_present_fs"];
    rpd.colorAttachments[ 0 ].pixelFormat = format;
    NSError* err                          = nil;
    g.presentPipe                         = [g.device newRenderPipelineStateWithDescriptor:rpd error:&err];
    g.presentFormat                       = format;
    if( !g.presentPipe )
    {
        fprintf( stderr,
                 "[MetalFX frame interpolation] present pipeline failed: %s\n",
                 err ? err.localizedDescription.UTF8String : "?" );
        return false;
    }
    return true;
}

void BlitToDrawable( id< MTLCommandBuffer > cb, id< CAMetalDrawable > drawable, id< MTLTexture > src, NSString* label )
{
    MTLRenderPassDescriptor* pass          = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[ 0 ].texture     = drawable.texture;
    pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionDontCare;
    pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;

    id< MTLRenderCommandEncoder > enc = [cb renderCommandEncoderWithDescriptor:pass];
    enc.label                         = label;
    [enc setRenderPipelineState:g.presentPipe];
    [enc setFragmentTexture:src atIndex:0];
    [enc setFragmentSamplerState:g.sampler atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}
}

bool RTGL1::MetalFI::Supported( VkDevice device )
{
    @autoreleasepool
    {
        return Init( device );
    }
}

bool RTGL1::MetalFI::Prepare( VkDevice device, uint32_t renderWidth, uint32_t renderHeight, uint32_t width, uint32_t height )
{
    @autoreleasepool
    {
        if( !Init( device ) || width == 0 || height == 0 || renderWidth == 0 || renderHeight == 0 )
        {
            return false;
        }
        return EnsureInterpolator( renderWidth, renderHeight, width, height );
    }
}

void RTGL1::MetalFI::Release()
{
    @autoreleasepool
    {
        ReleaseInterpolator();
        g.pendingReal      = nil;
        g.pendingGenerated = nil;
    }
}

bool RTGL1::MetalFI::Encode( VkDevice device, VkCommandBuffer cmd, const Params& p )
{
    @autoreleasepool
    {
        if( !Init( device ) || !RTGL1::MetalNative::IsNativeCmd( cmd ) || p.width == 0 || p.height == 0 ||
            p.renderWidth == 0 || p.renderHeight == 0 )
        {
            return false;
        }
        if( !EnsureInterpolator( p.renderWidth, p.renderHeight, p.width, p.height ) )
        {
            return false;
        }

        id< MTLTexture > rendered = TextureOf( p.rendered );
        id< MTLTexture > hud      = TextureOf( p.hud );
        id< MTLTexture > srcDepth = TextureOf( p.depthNdc );
        id< MTLTexture > srcMot   = TextureOf( p.motion );
        if( !rendered || !srcDepth || !srcMot )
        {
            return false;
        }

        const uint32_t   slot        = g.cur;
        const bool       interpolate = g.hasHistory && !p.reset;
        id< MTLTexture > cur         = g.color[ slot ];
        id< MTLTexture > prev        = g.color[ slot ^ 1 ];
        id< MTLTexture > real        = g.real[ slot ];
        id< MTLTexture > generated   = g.generated[ slot ];
        id< MTLTexture > ui          = g.ui;
        id< MTLTexture > depth       = g.depth;
        id< MTLTexture > motion      = g.motion;
        id               fiObj       = g.interpolator;
        const uint32_t   hasHud      = hud ? 1u : 0u;
        const Params     prm         = p;

        RTGL1::Metal::RecordExternal( cmd, [ = ]( id< MTLCommandBuffer > cb, id< MTLFence > fence ) {
            {
                id< MTLComputeCommandEncoder > enc = [cb computeCommandEncoder];
                enc.label                          = @"Frame interpolation - Prepare";
                if( fence )
                {
                    [enc waitForFence:fence];
                }
                [enc setComputePipelineState:g.prepPipe];
                [enc setTexture:rendered atIndex:0];
                [enc setTexture:( hud ? hud : rendered ) atIndex:1];
                [enc setTexture:cur atIndex:2];
                [enc setTexture:ui atIndex:3];
                [enc setTexture:real atIndex:4];
                [enc setBytes:&hasHud length:sizeof( hasHud ) atIndex:0];
                [enc dispatchThreads:MTLSizeMake( prm.width, prm.height, 1 ) threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];

                [enc setComputePipelineState:g.lowresPipe];
                [enc setTexture:srcDepth atIndex:0];
                [enc setTexture:srcMot atIndex:1];
                [enc setTexture:depth atIndex:2];
                [enc setTexture:motion atIndex:3];
                [enc dispatchThreads:MTLSizeMake( prm.renderWidth, prm.renderHeight, 1 )
                    threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
                if( fence )
                {
                    [enc updateFence:fence];
                }
                [enc endEncoding];
            }

            if( interpolate )
            {
                if( @available( macOS 26.0, * ) )
                {
                    id< MTLFXFrameInterpolator > fi = fiObj;
                    fi.colorTexture                 = cur;
                    fi.prevColorTexture             = prev;
                    fi.depthTexture                 = depth;
                    fi.motionTexture                = motion;
                    fi.uiTexture                    = ui;
                    fi.uiTextureComposited          = NO;
                    fi.outputTexture                = generated;
                    // the same conventions as the temporal scaler (FSR2_Metal.mm)
                    fi.motionVectorScaleX           = float( prm.renderWidth );
                    fi.motionVectorScaleY           = float( prm.renderHeight );
                    fi.jitterOffsetX                = -prm.jitterX;
                    fi.jitterOffsetY                = -prm.jitterY;
                    fi.deltaTime                    = std::clamp( prm.deltaTime, 0.001f, 0.25f );
                    fi.nearPlane                    = prm.nearPlane;
                    fi.farPlane                     = prm.farPlane;
                    fi.fieldOfView                  = prm.fovYRadians * 180.0f / float( M_PI );
                    fi.aspectRatio                  = float( prm.width ) / float( prm.height );
                    fi.depthReversed                = NO;
                    fi.shouldResetHistory           = NO;
                    [fi encodeToCommandBuffer:cb];
                }
            }
        } );

        g.pendingReal      = real;
        g.pendingGenerated = ( interpolate && p.presentGenerated ) ? generated : nil;
        g.lateDrawables    = true;
        g.hasHistory       = true;
        g.cur ^= 1;

        if( !g.loggedFirst )
        {
            g.loggedFirst = true;
            Log( "on" );
        }
        return true;
    }
}

bool RTGL1::MetalFI::WantsLateDrawables()
{
    return g.lateDrawables;
}

bool RTGL1::MetalFI::Present( id< MTLCommandBuffer > cb, CAMetalLayer* layer )
{
    if( !g.pendingReal )
    {
        g.lateDrawables = false;
        return false;
    }

    id< MTLTexture > real      = g.pendingReal;
    id< MTLTexture > generated = g.pendingGenerated;
    g.pendingReal              = nil;
    g.pendingGenerated         = nil;

    {
        const double now = Now();
        const double dt  = now - g.lastPresent;
        if( g.lastPresent > 0 && dt > 0 && dt < 0.25 )
        {
            g.frameInterval = g.frameInterval * 0.9 + dt * 0.1;
        }
        g.lastPresent = now;
    }

    if( !layer || !EnsurePresentPipeline( layer.pixelFormat ) )
    {
        return true; // nothing presented this frame
    }

    bool generatedPresented = false;
    if( generated )
    {
        if( id< CAMetalDrawable > d = [layer nextDrawable] )
        {
            BlitToDrawable( cb, d, generated, @"Blit generated frame to swapchain" );
            [cb presentDrawable:d];
            generatedPresented = true;
        }
    }
    if( id< CAMetalDrawable > d = [layer nextDrawable] )
    {
        BlitToDrawable( cb, d, real, @"Blit to swapchain" );
        if( generatedPresented )
        {
            // the generated frame is shown for a half of the frame time
            [cb presentDrawable:d afterMinimumDuration:std::clamp( g.frameInterval * 0.5, 0.0, 0.05 )];
        }
        else
        {
            [cb presentDrawable:d];
        }
    }
    return true;
}

#endif // __APPLE__
