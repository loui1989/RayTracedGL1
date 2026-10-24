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

#include "MetalPathTracer.h"

#include "EffectSimple_Instances.h"

#import <MetalFX/MetalFX.h>

#include "MetalCas.h"
#include "MetalComputePipelines.h"
#include "MetalFramebuffers.h"
#include "Shaders/MetalBindings.h"

#include "DrawFrameInfo.h"
#include "LpmParams.h"
#include "RgException.h"
#include "Utils.h"

#include <cstdlib>
#include <cstring>

namespace
{

id< MTLComputePipelineState > MakePipeline( id< MTLDevice > device, id< MTLLibrary > library, NSString* name )
{
    id< MTLFunction > f = [library newFunctionWithName:name];
    if( !f )
    {
        throw RTGL1::RgException( RG_RESULT_ERROR_CANT_FIND_SHADER,
                                  std::string{ "Can't find shader function " } + name.UTF8String );
    }
    NSError* error = nil;
    auto     pso   = [device newComputePipelineStateWithFunction:f error:&error];
    if( !pso )
    {
        throw RTGL1::RgException( RG_RESULT_GRAPHICS_API_ERROR,
                                  std::string{ "Can't create pipeline " } + name.UTF8String + ": " +
                                      error.localizedDescription.UTF8String );
    }
    return pso;
}

uint32_t GroupCount( uint32_t size, uint32_t groupSize )
{
    return ( size + groupSize - 1 ) / groupSize;
}


template< typename T >
std::span< const uint8_t > AsBytes( const T& t )
{
    return { reinterpret_cast< const uint8_t* >( &t ), sizeof( T ) };
}

}

struct RTGL1::MetalPathTracer::Effects
{
    EffectTeleport            teleport;
    EffectColorTint           colorTint;
    EffectInverseBW           inverseBW;
    EffectHueShift            hueShift;
    EffectNightVision         nightVision;
    EffectChromaticAberration chromaticAberration;
    EffectDistortedSides      distortedSides;
    EffectWaves               waves;
    EffectRadialBlur          radialBlur;
    EffectVHS                 vhs;
    EffectDither              dither;
    EffectCrtDemodulateEncode crtDemodulateEncode;
    EffectCrtDecode           crtDecode;
    EffectHDRPrepare          hdrPrepare;
};

RTGL1::MetalPathTracer::~MetalPathTracer() = default;

RTGL1::MetalPathTracer::MetalPathTracer( id< MTLDevice >        _device,
                                         id< MTLLibrary >       library,
                                         MetalComputePipelines& _shaders )
    : device{ _device }, shaders{ _shaders }, effects{ std::make_unique< Effects >() }
{
    blitScaled  = MakePipeline( device, library, @"BlitScaled" );

    for( bool linear : { false, true } )
    {
        MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
        d.minFilter             = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        d.magFilter             = d.minFilter;
        d.sAddressMode          = MTLSamplerAddressModeClampToEdge;
        d.tAddressMode          = MTLSamplerAddressModeClampToEdge;
        ( linear ? linearSampler : nearestSampler ) = [device newSamplerStateWithDescriptor:d];
    }

    // Volumetric::CreateImages, Volumetric::CreateSampler
    for( auto& v : scattering )
    {
        auto* d        = [MTLTextureDescriptor new];
        d.textureType  = MTLTextureType3D;
        d.pixelFormat  = MTLPixelFormatRGBA16Float;
        d.width        = VOLUMETRIC_SIZE_X;
        d.height       = VOLUMETRIC_SIZE_Y;
        d.depth        = VOLUMETRIC_SIZE_Z;
        d.usage        = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode  = MTLStorageModePrivate;
        v              = [device newTextureWithDescriptor:d];
        v.label        = @"Scattering Volume";
    }
    {
        MTLSamplerDescriptor* d  = [MTLSamplerDescriptor new];
        d.minFilter              = MTLSamplerMinMagFilterLinear;
        d.magFilter              = MTLSamplerMinMagFilterLinear;
        d.mipFilter              = MTLSamplerMipFilterLinear;
        d.sAddressMode           = MTLSamplerAddressModeClampToEdge;
        d.tAddressMode           = MTLSamplerAddressModeClampToEdge;
        d.rAddressMode           = MTLSamplerAddressModeClampToEdge;
        d.supportArgumentBuffers = YES;
        volumeSampler            = [device newSamplerStateWithDescriptor:d];
    }

    tonemapping = [device newBufferWithLength:sizeof( ShTonemapping ) options:MTLResourceStorageModePrivate];
    tonemapping.label = @"Tonemapping";

    if( const char* e = std::getenv( "RTGL1_METAL_PROFILE" ); e && e[ 0 ] == '1' )
    {
        id< MTLCounterSet > timestampSet = nil;
        for( id< MTLCounterSet > set in device.counterSets )
        {
            if( [set.name isEqualToString:MTLCommonCounterSetTimestamp] )
            {
                timestampSet = set;
            }
        }
        if( timestampSet && [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] )
        {
            MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
            d.counterSet                        = timestampSet;
            d.storageMode                       = MTLStorageModeShared;
            d.sampleCount                       = MaxProfiledPasses * 2;

            NSError* error = nil;
            counters       = [device newCounterSampleBufferWithDescriptor:d error:&error];
            profiling      = counters != nil;
        }
        if( !profiling )
        {
            debug::Warning( "Metal backend: GPU timestamps are not supported, RTGL1_METAL_PROFILE is ignored" );
        }
    }

    for( auto& b : lpmParams )
    {
        b       = [device newBufferWithLength:LPM_PARAMS_SIZE options:MTLResourceStorageModeShared];
        b.label = @"LPM params";
    }
}

// Debug: RTGL1_METAL_PROFILE=1 makes a compute encoder per pass, with GPU timestamps
// at the encoder boundaries, and prints the times of the passes every few seconds.
void RTGL1::MetalPathTracer::BeginPass( id< MTLCommandBuffer > cmd, __strong id< MTLComputeCommandEncoder >& enc, NSString* name )
{
    if( !profiling )
    {
        [enc pushDebugGroup:name];
        return;
    }

    assert( enc == nil );
    MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
    if( passCount < MaxProfiledPasses )
    {
        desc.sampleBufferAttachments[ 0 ].sampleBuffer              = counters;
        desc.sampleBufferAttachments[ 0 ].startOfEncoderSampleIndex = passCount * 2;
        desc.sampleBufferAttachments[ 0 ].endOfEncoderSampleIndex   = passCount * 2 + 1;
        passNames[ passCount ]                                      = name;
        passCount++;
    }
    enc       = [cmd computeCommandEncoderWithDescriptor:desc];
    enc.label = name;
}

void RTGL1::MetalPathTracer::EndPass( __strong id< MTLComputeCommandEncoder >& enc )
{
    if( !profiling )
    {
        [enc popDebugGroup];
        return;
    }
    [enc endEncoding];
    enc = nil;
}

void RTGL1::MetalPathTracer::PrintProfile( id< MTLCommandBuffer > cmd )
{
    if( ++profiledFrames % 120 != 0 )
    {
        return;
    }

    id< MTLCounterSampleBuffer > sampleBuffer = counters;
    const uint32_t               count        = passCount;
    auto                         names        = passNames;

    [cmd addCompletedHandler:^( id< MTLCommandBuffer > ) {
      NSData* data = [sampleBuffer resolveCounterRange:NSMakeRange( 0, count * 2 )];
      if( !data )
      {
          return;
      }
      const auto* ts    = static_cast< const MTLCounterResultTimestamp* >( data.bytes );
      double      total = 0;
      for( uint32_t i = 0; i < count; i++ )
      {
          // Apple GPUs' timestamps are in nanoseconds
          const double ms = double( ts[ i * 2 + 1 ].timestamp - ts[ i * 2 ].timestamp ) / 1e6;
          total += ms;
          debug::Info( "Metal profile: {:>8.3f} ms  {}", ms, names[ i ].UTF8String );
      }
      debug::Info( "Metal profile: {:>8.3f} ms  total", total );
    }];
}

id RTGL1::MetalPathTracer::PrepareTemporalScaler( uint32_t inputWidth,
                                                 uint32_t inputHeight,
                                                 uint32_t outputWidth,
                                                 uint32_t outputHeight )
{
    if( temporalScaler && scalerInputWidth == inputWidth && scalerInputHeight == inputHeight &&
        scalerOutputWidth == outputWidth && scalerOutputHeight == outputHeight )
    {
        return temporalScaler;
    }

    MTLFXTemporalScalerDescriptor* d = [MTLFXTemporalScalerDescriptor new];
    d.colorTextureFormat             = MTLPixelFormatRGBA16Float;
    d.depthTextureFormat             = MTLPixelFormatR32Float;
    d.motionTextureFormat            = MTLPixelFormatRG16Float;
    d.outputTextureFormat            = MTLPixelFormatRGBA16Float;
    d.inputWidth                     = inputWidth;
    d.inputHeight                    = inputHeight;
    d.outputWidth                    = outputWidth;
    d.outputHeight                   = outputHeight;
    // exposure is already applied in CmPrepareFinal
    d.autoExposureEnabled            = NO;
    // FB_IMAGE_INDEX_REACTIVITY, as for FSR2
    d.reactiveMaskTextureEnabled     = YES;
    d.reactiveMaskTextureFormat      = MTLPixelFormatR8Unorm;

    temporalScaler = [d newTemporalScalerWithDevice:device];
    if( !temporalScaler )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR, "Can't create MetalFX temporal scaler" );
    }
    scalerInputWidth   = inputWidth;
    scalerInputHeight  = inputHeight;
    scalerOutputWidth  = outputWidth;
    scalerOutputHeight = outputHeight;
    scalerNeedsReset   = true;

    debug::Info( "Metal backend: MetalFX temporal scaling {}x{} -> {}x{}", inputWidth, inputHeight, outputWidth, outputHeight );
    return temporalScaler;
}

id< MTLTexture > RTGL1::MetalPathTracer::GetDebugTexture( std::string_view name ) const
{
    if( name == "denoiserReactive" ) return denoiserReactive[ denoiserReactiveCur ];
    if( name == "denoiserDiffuseAlbedo" ) return denoiserDiffuseAlbedo;
    if( name == "denoiserSpecularAlbedo" ) return denoiserSpecularAlbedo;
    if( name == "denoiserNormal" ) return denoiserNormal;
    if( name == "denoiserRoughness" ) return denoiserRoughness;
    if( name == "denoiserOverlay" ) return denoiserOverlay;
    if( name == "denoiserStrength" ) return denoiserStrength;
    if( name == "denoiserRasterOverlay" ) return denoiserRasterOverlay;
    return nil;
}

bool RTGL1::MetalPathTracer::IsMetalFXDenoiserSupported( id< MTLDevice > device )
{
    if( @available( macOS 26.0, * ) )
    {
        return [MTLFXTemporalDenoisedScalerDescriptor supportsDevice:device];
    }
    return false;
}

id RTGL1::MetalPathTracer::PrepareDenoisedScaler( uint32_t inputWidth,
                                                  uint32_t inputHeight,
                                                  uint32_t outputWidth,
                                                  uint32_t outputHeight )
{
    if( denoisedScaler && denoiserInputWidth == inputWidth && denoiserInputHeight == inputHeight &&
        denoiserOutputWidth == outputWidth && denoiserOutputHeight == outputHeight )
    {
        return denoisedScaler;
    }

    if( @available( macOS 26.0, * ) )
    {
        constexpr MTLPixelFormat albedoFormat    = MTLPixelFormatRGBA8Unorm;
        constexpr MTLPixelFormat normalFormat    = MTLPixelFormatRGBA16Float;
        constexpr MTLPixelFormat roughnessFormat = MTLPixelFormatR16Float;

        MTLFXTemporalDenoisedScalerDescriptor* d = [MTLFXTemporalDenoisedScalerDescriptor new];
        d.colorTextureFormat                     = MTLPixelFormatRGBA16Float;
        d.depthTextureFormat                     = MTLPixelFormatR32Float;
        d.motionTextureFormat                    = MTLPixelFormatRG16Float;
        d.diffuseAlbedoTextureFormat             = albedoFormat;
        d.specularAlbedoTextureFormat            = albedoFormat;
        d.normalTextureFormat                    = normalFormat;
        d.roughnessTextureFormat                 = roughnessFormat;
        d.outputTextureFormat                    = MTLPixelFormatRGBA16Float;
        d.inputWidth                             = inputWidth;
        d.inputHeight                            = inputHeight;
        d.outputWidth                            = outputWidth;
        d.outputHeight                           = outputHeight;
        // exposure is already applied in CmPrepareFinal
        d.autoExposureEnabled                    = NO;
        // CmDenoiserReactive
        d.reactiveMaskTextureEnabled             = YES;
        d.reactiveMaskTextureFormat              = MTLPixelFormatR8Unorm;
        d.denoiseStrengthMaskTextureEnabled      = YES;
        d.denoiseStrengthMaskTextureFormat       = MTLPixelFormatR8Unorm;
        // CmPrepareFinalDenoiser
        d.transparencyOverlayTextureEnabled      = YES;
        d.transparencyOverlayTextureFormat       = MTLPixelFormatRGBA16Float;

        id< MTLFXTemporalDenoisedScaler > scaler = [d newTemporalDenoisedScalerWithDevice:device];
        if( !scaler )
        {
            throw RgException( RG_RESULT_GRAPHICS_API_ERROR, "Can't create MetalFX temporal denoised scaler" );
        }

        auto l_make = [ this, inputWidth, inputHeight ]( MTLPixelFormat format, MTLTextureUsage usage, NSString* label ) {
            MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                         width:inputWidth
                                                                                        height:inputHeight
                                                                                     mipmapped:NO];
            t.usage                 = usage | MTLTextureUsageShaderWrite;
            t.storageMode           = MTLStorageModePrivate;
            id< MTLTexture > tex    = [device newTextureWithDescriptor:t];
            tex.label               = label;
            return tex;
        };
        denoiserDiffuseAlbedo  = l_make( albedoFormat, scaler.diffuseAlbedoTextureUsage, @"MetalFX denoiser - Diffuse albedo" );
        denoiserSpecularAlbedo = l_make( albedoFormat, scaler.specularAlbedoTextureUsage, @"MetalFX denoiser - Specular albedo" );
        denoiserNormal         = l_make( normalFormat, scaler.normalTextureUsage, @"MetalFX denoiser - Normal" );
        denoiserRoughness      = l_make( roughnessFormat, scaler.roughnessTextureUsage, @"MetalFX denoiser - Roughness" );
        denoiserOverlay  = l_make( MTLPixelFormatRGBA16Float, scaler.transparencyOverlayTextureUsage, @"MetalFX denoiser - Overlay" );
        denoiserStrength = l_make( MTLPixelFormatR8Unorm, scaler.denoiseStrengthMaskTextureUsage, @"MetalFX denoiser - Strength" );
        denoiserRasterOverlay =
            l_make( MTLPixelFormatRGBA16Float, MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead, @"MetalFX denoiser - Raster overlay" );
        for( auto& r : denoiserReactive )
        {
            // shared, to clear on the CPU: its history is read
            MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                         width:inputWidth
                                                                                        height:inputHeight
                                                                                     mipmapped:NO];
            t.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            t.storageMode = MTLStorageModeShared;
            r             = [device newTextureWithDescriptor:t];
            r.label       = @"MetalFX denoiser - Reactive";

            const auto zeros = std::vector< uint8_t >( size_t( inputWidth ) * inputHeight, 0 );
            [r replaceRegion:MTLRegionMake2D( 0, 0, inputWidth, inputHeight )
                 mipmapLevel:0
                   withBytes:zeros.data()
                 bytesPerRow:inputWidth];
        }

        denoisedScaler       = scaler;
        denoiserInputWidth   = inputWidth;
        denoiserInputHeight  = inputHeight;
        denoiserOutputWidth  = outputWidth;
        denoiserOutputHeight = outputHeight;
        denoiserNeedsReset   = true;

        debug::Info( "Metal backend: MetalFX temporal denoised scaling {}x{} -> {}x{}",
                     inputWidth,
                     inputHeight,
                     outputWidth,
                     outputHeight );
        return denoisedScaler;
    }
    throw RgException( RG_RESULT_GRAPHICS_API_ERROR, "MetalFX temporal denoised scaler requires macOS 26" );
}

void RTGL1::MetalPathTracer::PrepareReservoirs( uint32_t width, uint32_t height )
{
    if( reservoirsWidth == width && reservoirsHeight == height )
    {
        return;
    }
    for( auto& b : indirectReservoirs )
    {
        b = [device newBufferWithLength:sizeof( uint32_t ) * PACKED_INDIRECT_RESERVOIR_SIZE_IN_WORDS * width * height
                                options:MTLResourceStorageModePrivate];
        b.label = @"Restir Indirect - Reservoirs";
    }
    reservoirsWidth  = width;
    reservoirsHeight = height;
}

id< MTLTexture > RTGL1::MetalPathTracer::Render( id< MTLCommandBuffer >              cmd,
                                                 const Params&                       p,
                                                 const RgDrawFrameInfo&              drawInfo,
                                                 MetalFramebuffers&                  framebuffers,
                                                 MetalBindingTable&                  b )
{
    const auto& tnmp   = pnext::get< RgDrawFrameTonemappingParams >( drawInfo );
    const auto& bloom  = pnext::get< RgDrawFrameBloomParams >( drawInfo );
    const auto& postef = pnext::get< RgDrawFramePostEffectsParams >( drawInfo );

    const uint32_t frameIndex = p.frameIndex;
    const uint32_t prevFrame  = ( frameIndex + MAX_FRAMES_IN_FLIGHT - 1 ) % MAX_FRAMES_IN_FLIGHT;

    PrepareReservoirs( p.width, p.height );

    const bool metalFXDenoiser = p.metalFXDenoiser && p.upscaler == Upscaler::MetalFX;
    if( metalFXDenoiser != wasMetalFXDenoiser )
    {
        // the other denoiser's history is outdated
        scalerNeedsReset   = true;
        denoiserNeedsReset = true;
        wasMetalFXDenoiser = metalFXDenoiser;
    }
    if( metalFXDenoiser )
    {
        PrepareDenoisedScaler( p.width, p.height, p.upscaledWidth, p.upscaledHeight );
        b.SetTexture( "denoiserDiffuseAlbedo", denoiserDiffuseAlbedo );
        b.SetTexture( "denoiserSpecularAlbedo", denoiserSpecularAlbedo );
        b.SetTexture( "denoiserNormal", denoiserNormal );
        b.SetTexture( "denoiserRoughness", denoiserRoughness );
        denoiserReactiveCur ^= 1;
        b.SetTexture( "denoiserReactive", denoiserReactive[ denoiserReactiveCur ] );
        b.SetTexture( "denoiserReactive_Prev", denoiserReactive[ denoiserReactiveCur ^ 1 ] );
        b.SetTexture( "denoiserOverlay", denoiserOverlay );
        b.SetTexture( "denoiserStrength", denoiserStrength );
        b.SetTexture( "denoiserRasterOverlay", denoiserRasterOverlay );
    }

    MakeLpmParams( lpmParams[ frameIndex ].contents, tnmp, p.hdr );

    b.SetBuffer( "Historam", tonemapping );
    b.SetBuffer( "LpmParams", lpmParams[ frameIndex ] );
    b.SetBuffer( "RestirIndirectReservoirs", indirectReservoirs[ frameIndex ] );
    b.SetBuffer( "RestirIndirectReservoirs_Prev", indirectReservoirs[ prevFrame ] );
    // Volumetric::UpdateDescriptors
    b.SetTexture( "g_volumetric", scattering[ frameIndex ] );
    b.SetTexture( "g_volumetric_Sampler", scattering[ frameIndex ] );
    b.SetSampler( "g_volumetric_SamplerSmplr", volumeSampler );
    b.SetTexture( "g_volumetric_Sampler_Prev", scattering[ prevFrame ] );
    b.SetSampler( "g_volumetric_Sampler_PrevSmplr", volumeSampler );

    const MetalComputePipelines::Constants primaryConstants = {
        { 0, p.primaryRaysMaxAlbedoLayers },
        { 1, p.lightmapLayerIndex },
    };
    const MetalComputePipelines::Constants indirectConstants = {
        { 0, p.indirectIlluminationMaxAlbedoLayers },
        { 1, p.lightmapLayerIndex },
    };

    passCount = 0;

    // EffectWipe::CopyToWipeEffectSourceIfNeeded: the previously presented image melts away
    if( postef.pWipe && postef.pWipe->beginNow && prevPresented )
    {
        id< MTLTexture > dst = framebuffers.Get( FB_IMAGE_INDEX_WIPE_EFFECT_SOURCE, frameIndex );
        if( prevPresented.width == dst.width && prevPresented.height == dst.height &&
            prevPresented.pixelFormat == dst.pixelFormat )
        {
            id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
            blit.label                       = @"Wipe effect source";
            [blit copyFromTexture:prevPresented toTexture:dst];
            [blit endEncoding];
        }
        else
        {
            id< MTLComputeCommandEncoder > c = [cmd computeCommandEncoder];
            c.label                          = @"Wipe effect source";
            [c setComputePipelineState:blitScaled];
            [c setTexture:prevPresented atIndex:0];
            [c setTexture:dst atIndex:1];
            [c setSamplerState:nearestSampler atIndex:0];
            const float srcScale[] = { 1.0f, 1.0f };
            [c setBytes:srcScale length:sizeof( srcScale ) atIndex:0];
            [c dispatchThreads:MTLSizeMake( dst.width, dst.height, 1 ) threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
            [c endEncoding];
        }
    }

    id< MTLComputeCommandEncoder > enc = nil;
    if( !profiling )
    {
        enc       = [cmd computeCommandEncoder];
        enc.label = @"RTGL1 path tracing";
    }

    BeginPass( cmd, enc, @"Primary rays" );
    shaders.Dispatch( enc, "RtRaygenPrimary", primaryConstants, b, p.width, p.height );
    EndPass( enc );

    // draw decals on top of primary surface
    if( p.rasterDecals )
    {
        if( !profiling )
        {
            [enc endEncoding];
            enc = nil;
        }
        p.rasterDecals( cmd );
        if( !profiling )
        {
            enc       = [cmd computeCommandEncoder];
            enc.label = @"RTGL1 path tracing";
        }
    }

    if( p.reflectRefractMaxDepth > 0 )
    {
        BeginPass( cmd, enc, @"Reflections and refractions" );
        shaders.Dispatch( enc, "RtRaygenReflRefr", primaryConstants, b, p.width, p.height );
        EndPass( enc );
    }

    BeginPass( cmd, enc, @"Initial reservoirs" );
    shaders.Dispatch( enc, "RtInitialReservoirs", {}, b, p.width, p.height );
    EndPass( enc );

    BeginPass( cmd, enc, @"Direct illumination" );
    shaders.Dispatch( enc, "RtRaygenDirect", {}, b, p.width, p.height );
    EndPass( enc );

    BeginPass( cmd, enc, @"Indirect illumination - Init" );
    shaders.Dispatch( enc, "RtRaygenIndirectInit", indirectConstants, b, p.width, p.height );
    EndPass( enc );

    // PathTracer::TraceVolumetric
    BeginPass( cmd, enc, @"Volumetric illumination" );
    shaders.Dispatch( enc, "RtVolumetric", {}, b, VOLUMETRIC_SIZE_X, VOLUMETRIC_SIZE_Y, VOLUMETRIC_SIZE_Z );
    EndPass( enc );

    BeginPass( cmd, enc, @"Gradient samples" );
    shaders.Dispatch( enc,
                      "RtGradients",
                      {},
                      b,
                      GroupCount( p.width, COMPUTE_ASVGF_STRATA_SIZE ),
                      GroupCount( p.height, COMPUTE_ASVGF_STRATA_SIZE ) );
    EndPass( enc );

    BeginPass( cmd, enc, @"Indirect illumination - Final" );
    shaders.DispatchThreadgroups( enc,
                                  "RtRaygenIndirectFinal",
                                  indirectConstants,
                                  b,
                                  GroupCount( p.width, COMPUTE_INDIRECT_FINAL_GROUP_SIZE_X ),
                                  GroupCount( p.height, COMPUTE_INDIRECT_FINAL_GROUP_SIZE_Y ) );
    EndPass( enc );

    // Denoiser::Denoise
    if( metalFXDenoiser )
    {
        // MetalFX denoises the composed image later, only compose the unfiltered illumination
        BeginPass( cmd, enc, @"Compose unfiltered (MetalFX denoiser)" );
        shaders.DispatchThreadgroups( enc,
                                      "CmComposeUnfiltered",
                                      {},
                                      b,
                                      GroupCount( p.width, COMPUTE_COMPOSE_GROUP_SIZE_X ),
                                      GroupCount( p.height, COMPUTE_COMPOSE_GROUP_SIZE_Y ) );

        // the gradients, as for SVGF: where the lighting changed, for CmDenoiserReactive
        const uint32_t gx = GroupCount( p.width / COMPUTE_ASVGF_STRATA_SIZE, COMPUTE_GRADIENT_ATROUS_GROUP_SIZE_X );
        const uint32_t gy = GroupCount( p.height / COMPUTE_ASVGF_STRATA_SIZE, COMPUTE_GRADIENT_ATROUS_GROUP_SIZE_X );
        for( uint32_t i = 0; i < COMPUTE_ASVGF_GRADIENT_ATROUS_ITERATION_COUNT; i++ )
        {
            shaders.DispatchThreadgroups( enc, "CmASVGFGradientAtrous", { { 0, i } }, b, gx, gy );
        }
        EndPass( enc );
    }
    else
    {
        BeginPass( cmd, enc, @"Denoiser" );

        const uint32_t gx = GroupCount( p.width / COMPUTE_ASVGF_STRATA_SIZE, COMPUTE_GRADIENT_ATROUS_GROUP_SIZE_X );
        const uint32_t gy = GroupCount( p.height / COMPUTE_ASVGF_STRATA_SIZE, COMPUTE_GRADIENT_ATROUS_GROUP_SIZE_X );
        for( uint32_t i = 0; i < COMPUTE_ASVGF_GRADIENT_ATROUS_ITERATION_COUNT; i++ )
        {
            shaders.DispatchThreadgroups( enc, "CmASVGFGradientAtrous", { { 0, i } }, b, gx, gy );
        }

        const uint32_t x = GroupCount( p.width, COMPUTE_SVGF_TEMPORAL_GROUP_SIZE_X );
        const uint32_t y = GroupCount( p.height, COMPUTE_SVGF_TEMPORAL_GROUP_SIZE_X );
        shaders.DispatchThreadgroups( enc, "CmSVGFTemporalAccumulation", {}, b, x, y );
        shaders.DispatchThreadgroups( enc, "CmAntiFirefly", {}, b, x, y );
        shaders.DispatchThreadgroups( enc, "CmSVGFEstimateVariance", {}, b, x, y );

        shaders.DispatchThreadgroups( enc, "CmSVGFAtrous_Iter0", {}, b, x, y );
        for( uint32_t i = 1; i < COMPUTE_SVGF_ATROUS_ITERATION_COUNT; i++ )
        {
            shaders.DispatchThreadgroups( enc, "CmSVGFAtrous", { { 0, i } }, b, x, y );
        }

        EndPass( enc );
    }

    // Volumetric::ProcessScattering
    {
        BeginPass( cmd, enc, @"Volumetric process" );
        shaders.DispatchThreadgroups( enc,
                                      "CmVolumetricProcess",
                                      {},
                                      b,
                                      GroupCount( VOLUMETRIC_SIZE_X, COMPUTE_VOLUMETRIC_GROUP_SIZE_X ),
                                      GroupCount( VOLUMETRIC_SIZE_Y, COMPUTE_VOLUMETRIC_GROUP_SIZE_Y ) );
        EndPass( enc );
    }
    // accumulate to screen space
    {
        BeginPass( cmd, enc, @"Volumetric accumulation" );
        const float maxHistoryLength = p.volumetricMaxHistoryLength;
        shaders.DispatchThreadgroups( enc,
                                      "CmScatterAccum",
                                      {},
                                      b,
                                      GroupCount( p.width, COMPUTE_SCATTER_ACCUM_GROUP_SIZE_X ),
                                      GroupCount( p.height, COMPUTE_SCATTER_ACCUM_GROUP_SIZE_X ),
                                      1,
                                      AsBytes( maxHistoryLength ) );
        EndPass( enc );
    }

    // Tonemapping::CalculateExposure
    {
        BeginPass( cmd, enc, @"Exposure" );
        shaders.DispatchThreadgroups( enc,
                                      "CmLuminanceHistogram",
                                      {},
                                      b,
                                      GroupCount( p.width, COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_X ),
                                      GroupCount( p.height, COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_Y ) );
        shaders.DispatchThreadgroups( enc, "CmLuminanceAvg", {}, b, 1, 1 );
        EndPass( enc );
    }

    const uint32_t cx = GroupCount( p.width, COMPUTE_COMPOSE_GROUP_SIZE_X );
    const uint32_t cy = GroupCount( p.height, COMPUTE_COMPOSE_GROUP_SIZE_Y );

    // ImageComposition::PrepareForRaster
    BeginPass( cmd, enc, @"Checkerboard" );
    shaders.DispatchThreadgroups( enc, "CmCheckerboard", {}, b, cx, cy );
    EndPass( enc );

    // Rasterizer::DrawToFinalImage
    if( p.rasterToFinal )
    {
        if( !profiling )
        {
            [enc endEncoding];
            enc = nil;
        }
        p.rasterToFinal( cmd, metalFXDenoiser ? denoiserRasterOverlay : nil );
        if( !profiling )
        {
            enc       = [cmd computeCommandEncoder];
            enc.label = @"RTGL1 composition";
        }
    }

    // ImageComposition::Finalize
    BeginPass( cmd, enc, @"Prefinal framebuf compose" );
    // the MetalFX denoiser gets the noise-free volumetrics and emission separately
    shaders.DispatchThreadgroups( enc, metalFXDenoiser ? "CmPrepareFinalDenoiser" : "CmPrepareFinal", {}, b, cx, cy );
    EndPass( enc );

    // after the rasterization, as it writes the reactivity
    if( metalFXDenoiser )
    {
        BeginPass( cmd, enc, @"MetalFX denoiser reactive mask" );
        shaders.DispatchThreadgroups( enc, "CmDenoiserReactive", {}, b, cx, cy );
        EndPass( enc );
    }

    // upscale Final into one of the upscaled framebufs:
    // the MetalFX temporal scaler (as FSR2.cpp) or Framebuffers::BlitForEffects
    id< MTLTexture > ping     = framebuffers.Get( FB_IMAGE_INDEX_UPSCALED_PING, frameIndex );
    id< MTLTexture > pong     = framebuffers.Get( FB_IMAGE_INDEX_UPSCALED_PONG, frameIndex );
    id< MTLTexture > final    = framebuffers.Get( FB_IMAGE_INDEX_FINAL, frameIndex );
    bool             accumIsPing = true;

    auto l_rasterClassic = [ & ]( id< MTLTexture > target ) {
        if( !p.rasterClassic )
        {
            return;
        }
        if( !profiling && enc )
        {
            [enc endEncoding];
            enc = nil;
        }
        p.rasterClassic( cmd, target );
        if( !profiling )
        {
            enc       = [cmd computeCommandEncoder];
            enc.label = @"RTGL1 post processing";
        }
    };

    if( p.upscaler == Upscaler::MetalFX )
    {
        if( !profiling )
        {
            [enc endEncoding];
            enc = nil;
        }

        if( metalFXDenoiser )
        {
            if( @available( macOS 26.0, * ) )
            {
                id< MTLFXTemporalDenoisedScaler > ds =
                    PrepareDenoisedScaler( p.width, p.height, p.upscaledWidth, p.upscaledHeight );
                ds.colorTexture          = final;
                ds.depthTexture          = framebuffers.Get( FB_IMAGE_INDEX_DEPTH_NDC, frameIndex );
                ds.motionTexture         = framebuffers.Get( FB_IMAGE_INDEX_MOTION_DLSS, frameIndex );
                ds.diffuseAlbedoTexture  = denoiserDiffuseAlbedo;
                ds.specularAlbedoTexture = denoiserSpecularAlbedo;
                ds.normalTexture         = denoiserNormal;
                ds.roughnessTexture      = denoiserRoughness;
                ds.reactiveMaskTexture   = denoiserReactive[ denoiserReactiveCur ];
                ds.denoiseStrengthMaskTexture = denoiserStrength;
                ds.transparencyOverlayTexture = denoiserOverlay;
                ds.outputTexture         = pong;
                // the same as for the temporal scaler
                ds.motionVectorScaleX    = float( p.width );
                ds.motionVectorScaleY    = float( p.height );
                ds.jitterOffsetX         = -p.jitter.data[ 0 ];
                ds.jitterOffsetY         = -p.jitter.data[ 1 ];
                ds.preExposure           = 1.0f;
                ds.depthReversed         = NO;
                if( p.view && p.projection )
                {
                    simd_float4x4 v, pr;
                    static_assert( sizeof( v ) == sizeof( float ) * 16 );
                    memcpy( &v, p.view, sizeof( v ) );
                    memcpy( &pr, p.projection, sizeof( pr ) );
                    // RTGL1's projection is for Vulkan's clip space (Y down), Metal's is Y up
                    pr.columns[ 0 ][ 1 ] = -pr.columns[ 0 ][ 1 ];
                    pr.columns[ 1 ][ 1 ] = -pr.columns[ 1 ][ 1 ];
                    pr.columns[ 2 ][ 1 ] = -pr.columns[ 2 ][ 1 ];
                    pr.columns[ 3 ][ 1 ] = -pr.columns[ 3 ][ 1 ];
                    ds.worldToViewMatrix = v;
                    ds.viewToClipMatrix  = pr;
                }
                ds.shouldResetHistory = p.resetHistory || denoiserNeedsReset;
                denoiserNeedsReset    = false;
                [ds encodeToCommandBuffer:cmd];
                activeScaler = ds;
            }
        }
        else
        {
            id< MTLFXTemporalScaler > scaler = PrepareTemporalScaler( p.width, p.height, p.upscaledWidth, p.upscaledHeight );
            scaler.colorTexture       = final;
            scaler.depthTexture       = framebuffers.Get( FB_IMAGE_INDEX_DEPTH_NDC, frameIndex );
            scaler.motionTexture      = framebuffers.Get( FB_IMAGE_INDEX_MOTION_DLSS, frameIndex );
            scaler.reactiveMaskTexture = framebuffers.Get( FB_IMAGE_INDEX_REACTIVITY, frameIndex );
            scaler.outputTexture      = pong;
            scaler.inputContentWidth  = p.width;
            scaler.inputContentHeight = p.height;
            // the same as FSR2.cpp: motion vectors are in UV space
            scaler.motionVectorScaleX = float( p.width );
            scaler.motionVectorScaleY = float( p.height );
            // the same sign as for FSR2 / DLSS (checked: the opposite one is noticeably blurrier)
            scaler.jitterOffsetX      = -p.jitter.data[ 0 ];
            scaler.jitterOffsetY      = -p.jitter.data[ 1 ];
            scaler.preExposure        = 1.0f;
            scaler.depthReversed      = NO;
            scaler.reset              = p.resetHistory || scalerNeedsReset;
            scalerNeedsReset          = false;
            [scaler encodeToCommandBuffer:cmd];
            activeScaler = scaler;
        }

        accumIsPing = false;

        if( !profiling )
        {
            enc       = [cmd computeCommandEncoder];
            enc.label = @"RTGL1 post processing";
        }

        // Rasterizer::DrawClassic, into the upscaled image
        l_rasterClassic( pong );
    }
    else
    {
        // Rasterizer::DrawClassic, into the final image before the blit
        l_rasterClassic( final );

        BeginPass( cmd, enc, @"Blit for effects" );
        [enc setComputePipelineState:blitScaled];
        [enc setTexture:final atIndex:0];
        [enc setTexture:ping atIndex:1];
        [enc setSamplerState:p.upscaler == Upscaler::Linear ? linearSampler : nearestSampler atIndex:0];
        const float srcSize[] = { float( p.width ) / float( final.width ), float( p.height ) / float( final.height ) };
        [enc setBytes:srcSize length:sizeof( srcSize ) atIndex:0];
        [enc dispatchThreads:MTLSizeMake( ping.width, ping.height, 1 ) threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];
        EndPass( enc );
    }

    // Sharpening::Apply.
    // On Vulkan, FSR2 does CAS itself (RenderResolutionHelper::IsCASInsideFSR2),
    // MetalFX doesn't sharpen: so CAS is a dedicated pass also with MetalFX.
    if( p.sharpenTechnique == RG_RENDER_SHARPEN_TECHNIQUE_NAIVE ||
        p.sharpenTechnique == RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS )
    {
        struct
        {
            uint32_t con0[ 4 ];
            uint32_t con1[ 4 ];
        } casPush;
        MakeCasPush( casPush.con0, casPush.con1, 1.0f, uint32_t( ping.width ), uint32_t( ping.height ) );

        BeginPass( cmd, enc, @"Sharpening" );
        shaders.DispatchThreadgroups( enc,
                                      "CmCas",
                                      { { 0, accumIsPing ? 1u : 0u },
                                        { 1, p.sharpenTechnique == RG_RENDER_SHARPEN_TECHNIQUE_NAIVE ? 1u : 0u } },
                                      b,
                                      GroupCount( uint32_t( ping.width ), 16 ),
                                      GroupCount( uint32_t( ping.height ), 16 ),
                                      1,
                                      AsBytes( casPush ) );
        EndPass( enc );
        accumIsPing = !accumIsPing;
    }

    const auto effectGroupsX = GroupCount( uint32_t( ping.width ), COMPUTE_EFFECT_GROUP_SIZE_X );
    const auto effectGroupsY = GroupCount( uint32_t( ping.height ), COMPUTE_EFFECT_GROUP_SIZE_Y );

    const auto args = CommonnlyUsedEffectArguments{
        .width       = uint32_t( ping.width ),
        .height      = uint32_t( ping.height ),
        .currentTime = p.currentTime,
    };

    // EffectBase::Dispatch: accum -> the other one of ping / pong
    auto applyEffect = [ & ]< typename T >( T& effect, NSString* label ) {
        BeginPass( cmd, enc, label );
        shaders.DispatchThreadgroups( enc,
                                      T::GetShaderName().c_str(),
                                      { { 0, accumIsPing ? 1u : 0u } },
                                      b,
                                      effectGroupsX,
                                      effectGroupsY,
                                      1,
                                      effect.GetPushBytes() );
        EndPass( enc );
        accumIsPing = !accumIsPing;
    };
    auto applyIf = [ & ]< typename T, typename Arg >( T& effect, const Arg& setupArg, NSString* label ) {
        if( effect.Setup( args, setupArg ) )
        {
            applyEffect( effect, label );
        }
    };

    // Bloom::Apply
    if( bloom.bloomIntensity > 0.0f )
    {
        BeginPass( cmd, enc, @"Bloom" );

        auto sizeOf = [ & ]( FramebufferImageIndex fb ) {
            id< MTLTexture > t = framebuffers.Get( fb, frameIndex );
            return std::pair{ uint32_t( t.width ), uint32_t( t.height ) };
        };

        {
            auto [ w, h ] = sizeOf( FB_IMAGE_INDEX_BLOOM );
            shaders.DispatchThreadgroups( enc,
                                          "CmBloomPreload",
                                          { { 0, accumIsPing ? 1u : 0u } },
                                          b,
                                          GroupCount( w, COMPUTE_BLOOM_APPLY_GROUP_SIZE_X ),
                                          GroupCount( h, COMPUTE_BLOOM_APPLY_GROUP_SIZE_Y ) );
        }

        constexpr FramebufferImageIndex mips[] = {
            FB_IMAGE_INDEX_BLOOM,      FB_IMAGE_INDEX_BLOOM_MIP1, FB_IMAGE_INDEX_BLOOM_MIP2, FB_IMAGE_INDEX_BLOOM_MIP3,
            FB_IMAGE_INDEX_BLOOM_MIP4, FB_IMAGE_INDEX_BLOOM_MIP5, FB_IMAGE_INDEX_BLOOM_MIP6, FB_IMAGE_INDEX_BLOOM_MIP7,
        };
        static_assert( std::size( mips ) == COMPUTE_BLOOM_STEP_COUNT + 1 );

        for( uint32_t i = 0; i < COMPUTE_BLOOM_STEP_COUNT; i++ )
        {
            auto [ w, h ] = sizeOf( mips[ i + 1 ] );
            shaders.DispatchThreadgroups( enc,
                                          "CmBloomDownsample",
                                          { { 0, i } },
                                          b,
                                          GroupCount( w, COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_X ),
                                          GroupCount( h, COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_Y ) );
        }
        // start from the other side
        for( int i = COMPUTE_BLOOM_STEP_COUNT - 1; i >= 0; i-- )
        {
            auto [ w, h ] = sizeOf( mips[ i ] );
            shaders.DispatchThreadgroups( enc,
                                          "CmBloomUpsample",
                                          { { 0, uint32_t( i ) } },
                                          b,
                                          GroupCount( w, COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_X ),
                                          GroupCount( h, COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_Y ) );
        }

        shaders.DispatchThreadgroups( enc,
                                      "CmBloomApply",
                                      { { 0, accumIsPing ? 1u : 0u } },
                                      b,
                                      GroupCount( uint32_t( ping.width ), COMPUTE_BLOOM_APPLY_GROUP_SIZE_X ),
                                      GroupCount( uint32_t( ping.height ), COMPUTE_BLOOM_APPLY_GROUP_SIZE_Y ) );
        accumIsPing = !accumIsPing;

        EndPass( enc );
    }

    // the same order as VulkanDevice::Render
    {
        Effects& e = *effects;
        applyIf( e.teleport, postef.pTeleport, @"Teleport" );
        applyIf( e.colorTint, postef.pColorTint, @"Color tint" );
        applyIf( e.inverseBW, postef.pInverseBlackAndWhite, @"Inverse black and white" );
        applyIf( e.hueShift, postef.pHueShift, @"Hue shift" );
        applyIf( e.nightVision, postef.pNightVision, @"Night vision" );
        applyIf( e.chromaticAberration, postef.pChromaticAberration, @"Chromatic aberration" );
        applyIf( e.distortedSides, postef.pDistortedSides, @"Distorted sides" );
        applyIf( e.waves, postef.pWaves, @"Waves" );
        applyIf( e.radialBlur, postef.pRadialBlur, @"Radial blur" );
        applyIf( e.vhs, postef.pVHS, @"VHS" );

        // draw geometry such as HUD into an upscaled framebuf
        if( p.rasterHud )
        {
            if( !profiling )
            {
                [enc endEncoding];
                enc = nil;
            }
            if( p.hudlessCopy )
            {
                id< MTLTexture > cur = accumIsPing ? ping : pong;
                if( !hudlessBeforeHud || hudlessBeforeHud.width != cur.width || hudlessBeforeHud.height != cur.height )
                {
                    MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:cur.pixelFormat
                                                                                                 width:cur.width
                                                                                                height:cur.height
                                                                                             mipmapped:NO];
                    t.usage          = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                    t.storageMode    = MTLStorageModePrivate;
                    hudlessBeforeHud = [device newTextureWithDescriptor:t];
                    withHud          = [device newTextureWithDescriptor:t];
                    hudlessBeforeHud.label = @"Hudless - before HUD";
                    withHud.label          = @"Hudless - with HUD";
                }
                id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
                blit.label                       = @"Hudless - copy before HUD";
                [blit copyFromTexture:cur toTexture:hudlessBeforeHud];
                [blit endEncoding];
            }
            p.rasterHud( cmd, accumIsPing ? ping : pong );
            if( !profiling )
            {
                enc       = [cmd computeCommandEncoder];
                enc.label = @"RTGL1 post processing";
            }
        }

        // EffectWipe::Setup
        if( postef.pWipe && p.effectWipeIsUsed )
        {
            wipePush.stripWidthInPixels =
                uint32_t( float( args.width ) * std::clamp( postef.pWipe->stripWidth, 0.0f, 1.0f ) );
            if( postef.pWipe->beginNow )
            {
                wipePush.startFrameId = p.frameId;
                wipePush.beginTime    = args.currentTime;
                wipePush.endTime      = args.currentTime + postef.pWipe->duration;
            }
            if( !( wipePush.stripWidthInPixels == 0 || wipePush.beginTime >= wipePush.endTime ||
                   args.currentTime >= wipePush.endTime ) )
            {
                BeginPass( cmd, enc, @"Wipe" );
                shaders.DispatchThreadgroups( enc,
                                              "EfWipe",
                                              { { 0, accumIsPing ? 1u : 0u } },
                                              b,
                                              effectGroupsX,
                                              effectGroupsY,
                                              1,
                                              AsBytes( wipePush ) );
                EndPass( enc );
                accumIsPing = !accumIsPing;
            }
        }

        applyIf( e.dither, postef.pDither, @"Dither" );

        if( postef.pCRT != nullptr && postef.pCRT->isActive )
        {
            e.crtDemodulateEncode.Setup( args );
            applyEffect( e.crtDemodulateEncode, @"CRT demodulate" );

            e.crtDecode.Setup( args );
            applyEffect( e.crtDecode, @"CRT decode" );
        }

        // convert scene HDR to a present HDR compatible space, or apply a tonemapping to fit into LDR
        e.hdrPrepare.Setup( args, tnmp );
        applyEffect( e.hdrPrepare, @"HDR prepare" );

        // the same, but without HUD: HDR prepare of the image before HUD
        // (wipe, dither and CRT are not for the MetalFX upscaling, that is required for frame interpolation)
        hudlessImage = nil;
        if( p.rasterHud && p.hudlessCopy && hudlessBeforeHud )
        {
            id< MTLTexture > result = accumIsPing ? ping : pong;
            id< MTLTexture > other  = accumIsPing ? pong : ping;

            if( !profiling )
            {
                [enc endEncoding];
                enc = nil;
            }
            id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
            blit.label                       = @"Hudless";
            [blit copyFromTexture:result toTexture:withHud];
            [blit copyFromTexture:hudlessBeforeHud toTexture:other];
            [blit endEncoding];
            if( !profiling )
            {
                enc       = [cmd computeCommandEncoder];
                enc.label = @"RTGL1 post processing";
            }

            accumIsPing = !accumIsPing;
            applyEffect( e.hdrPrepare, @"HDR prepare - without HUD" );

            hudlessImage = accumIsPing ? ping : pong;
            // the result is the one with HUD
            if( !profiling )
            {
                [enc endEncoding];
                enc = nil;
            }
            else
            {
                PrintProfile( cmd );
            }
            prevPresented = withHud;
            return prevPresented;
        }
    }

    if( !profiling )
    {
        [enc endEncoding];
    }
    else
    {
        PrintProfile( cmd );
    }

    // the latest target: each pass flips 'accumIsPing'
    prevPresented = accumIsPing ? ping : pong;
    return prevPresented;
}
