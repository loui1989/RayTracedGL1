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

#include "MetalFluid.h"

#include "MetalComputePipelines.h"

#include "Generated/ShaderCommonC.h"
#include "Shaders/Fluid_Def.h"
#include "DebugPrint.h"
#include "RgException.h"
#include "Utils.h"

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>

namespace
{

// the same as Fluid.cpp
constexpr uint32_t MAX_PARTICLES_DEFAULT = 1024 * 1024;

using IdToSource               = uint8_t;
constexpr uint32_t MAX_SOURCES = std::numeric_limits< IdToSource >::max();

constexpr uint32_t QUAD_VERTEX_COUNT = 4;

struct ParticlesPush_T
{
    RgFloat3D gravity;
    float     deltaTime;
    uint32_t  activeRingBegin;
    uint32_t  activeRingLength;
    uint32_t  generateRingBegin;
    uint32_t  generateRingLength;
};
static_assert( sizeof( ParticlesPush_T ) == 32 );

struct VisualizePush_T
{
    float proj[ 16 ];
    float view[ 16 ];
};

struct FluidSmoothPush_T
{
    uint32_t renderWidth;
    uint32_t renderHeight;
    float    zNear;
    float    zFar;
};

// Fluid_DepthSmooth: must be even, so the final write is into framebufDepthFluid
constexpr uint32_t SMOOTH_ITERATIONS = 2;

template< typename T >
std::span< const uint8_t > AsBytes( const T& t )
{
    return { reinterpret_cast< const uint8_t* >( &t ), sizeof( T ) };
}

uint32_t GroupCount( uint32_t size, uint32_t groupSize )
{
    return ( size + groupSize - 1 ) / groupSize;
}

// RingBuf of Fluid.cpp
using Ring = RTGL1::MetalFluid::Ring;

struct Range
{
    uint32_t first{ 0 };
    uint32_t count{ 0 };
};

uint32_t Length( const Ring& r, uint32_t max )
{
    if( r.full )
    {
        return max;
    }
    return r.end < r.begin ? max + r.end - r.begin : r.end - r.begin;
}

std::array< Range, 2 > AsRanges( const Ring& r, uint32_t max )
{
    if( r.full )
    {
        return { Range{ 0, max }, Range{} };
    }
    if( r.end < r.begin )
    {
        return { Range{ r.begin, max - r.begin }, Range{ 0, r.end + 1 } };
    }
    return { Range{ r.begin, r.end - r.begin }, Range{} };
}

Ring MakeRing( uint32_t first, uint32_t count, uint32_t max )
{
    count = std::min( count, max );
    return Ring{
        .begin = first % max,
        .end   = ( first + count ) % max,
        .full  = ( count == max ),
    };
}

Ring AppendRing( const Ring& base, const Ring& increment, uint32_t max )
{
    if( Length( base, max ) + Length( increment, max ) >= max )
    {
        return MakeRing( base.begin + Length( increment, max ), max, max );
    }
    return MakeRing( base.begin, std::min( max, Length( base, max ) + Length( increment, max ) ), max );
}

Ring MakeEmptyWithBeginningAtEndOf( const Ring& after )
{
    return Ring{ .begin = after.end, .end = after.end, .full = false };
}

}

RTGL1::MetalFluid::MetalFluid( id< MTLDevice >  _device,
                               id< MTLLibrary > library,
                               uint32_t         fluidBudget,
                               float            _particleRadius )
    : device{ _device }
{
    fluidBudget    = std::clamp( fluidBudget, 4096u, MAX_PARTICLES_DEFAULT );
    maxParticles   = Utils::Align( fluidBudget, 4096u );
    particleRadius = std::clamp( _particleRadius, 0.01f, 1.0f );

    particles       = [device newBufferWithLength:maxParticles * sizeof( ShParticleDef )
                                    options:MTLResourceStorageModePrivate];
    particles.label = @"Fluid Particles";

    for( uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++ )
    {
        generateIdToSource[ i ] = [device newBufferWithLength:maxParticles * sizeof( IdToSource )
                                                      options:MTLResourceStorageModeShared];
        generateIdToSource[ i ].label = @"Fluid Generate: Particle ID to Source";
        memset( generateIdToSource[ i ].contents, 0xFF, generateIdToSource[ i ].length );

        sources[ i ]       = [device newBufferWithLength:MAX_SOURCES * sizeof( ShParticleSourceDef )
                                           options:MTLResourceStorageModeShared];
        sources[ i ].label = @"Fluid Sources";
    }
    sourcesCached.reserve( MAX_SOURCES );
    sourcesCachedCount.reserve( MAX_SOURCES );

    NSError* error = nil;
    copyDepth      = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"FluidCopyDepth"]
                                                          error:&error];
    if( !copyDepth )
    {
        throw RgException( RG_RESULT_GRAPHICS_API_ERROR,
                           std::string{ "Metal backend: FluidCopyDepth: " } +
                               ( error ? error.localizedDescription.UTF8String : "" ) );
    }
}

RTGL1::MetalFluid::~MetalFluid() = default;

void RTGL1::MetalFluid::PrepareForFrame( bool reset )
{
    if( reset )
    {
        sourcesCached.clear();
        sourcesCachedCount.clear();
        active = MakeEmptyWithBeginningAtEndOf( active );
    }
}

void RTGL1::MetalFluid::AddSource( const RgSpawnFluidInfo& src )
{
    if( src.count == 0 )
    {
        return;
    }
    if( src.count >= maxParticles )
    {
        debug::Error( "Too many particles in a fluid source. Max={}", maxParticles );
        return;
    }
    if( sourcesCached.size() >= MAX_SOURCES )
    {
        debug::Error( "Too many fluid sources in a frame, ignoring" );
        return;
    }

    sourcesCached.push_back( ShParticleSourceDef{
        .position_dispersionAngle = glm::packHalf4x16( {
            src.position.data[ 0 ],
            src.position.data[ 1 ],
            src.position.data[ 2 ],
            std::clamp( src.dispersionAngleDegrees / 180.f, 0.f, 1.f ),
        } ),
        .velocity_dispersion      = glm::packHalf4x16( {
            src.velocity.data[ 0 ],
            src.velocity.data[ 1 ],
            src.velocity.data[ 2 ],
            std::clamp( src.dispersionVelocity, 0.f, 1.f ),
        } ),
    } );
    sourcesCachedCount.push_back( src.count );
}

bool RTGL1::MetalFluid::Active() const
{
    return Length( active, maxParticles ) > 0 || !sourcesCached.empty();
}

void RTGL1::MetalFluid::BindBuffers( MetalBindingTable& b, uint32_t frameIndex ) const
{
    b.SetBuffer( "ParticlesArray_T", particles );
    b.SetBuffer( "GenerateIdToSource_T", generateIdToSource[ frameIndex ] );
    b.SetBuffer( "Sources_T", sources[ frameIndex ] );
}

void RTGL1::MetalFluid::Simulate( id< MTLCommandBuffer > cmd,
                                  MetalComputePipelines& shaders,
                                  MetalBindingTable&     b,
                                  uint32_t               frameIndex,
                                  float                  deltaTime,
                                  const RgFloat3D&       gravity )
{
    if( !Active() )
    {
        return;
    }

    if( !particlesCleared )
    {
        id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
        blit.label                       = @"Fluid Particles - Clear";
        [blit fillBuffer:particles range:NSMakeRange( 0, particles.length ) value:0xFF];
        [blit endEncoding];
        particlesCleared = true;
    }

    // append generated at the end of active
    Ring generateRing = MakeEmptyWithBeginningAtEndOf( active );

    const auto sourceCount = uint32_t( sourcesCached.size() );
    if( sourceCount > 0 )
    {
        memcpy( sources[ frameIndex ].contents, sourcesCached.data(), sizeof( ShParticleSourceDef ) * sourceCount );
        auto* idToSource = static_cast< IdToSource* >( generateIdToSource[ frameIndex ].contents );

        for( uint32_t sourceId = 0; sourceId < sourceCount; sourceId++ )
        {
            const Ring added = MakeRing( generateRing.end, sourcesCachedCount[ sourceId ], maxParticles );
            for( const Range& r : AsRanges( added, maxParticles ) )
            {
                if( r.count > 0 )
                {
                    memset( &idToSource[ r.first ], int( sourceId ), sizeof( IdToSource ) * r.count );
                }
            }
            generateRing = AppendRing( generateRing, added, maxParticles );
            active       = AppendRing( active, added, maxParticles );
        }
    }
    sourcesCached.clear();
    sourcesCachedCount.clear();

    const bool generate = Length( generateRing, maxParticles ) > 0 && sourceCount > 0;

    BindBuffers( b, frameIndex );

    const auto push = ParticlesPush_T{
        .gravity            = gravity,
        .deltaTime          = deltaTime,
        .activeRingBegin    = active.begin,
        .activeRingLength   = Length( active, maxParticles ),
        .generateRingBegin  = generateRing.begin,
        .generateRingLength = Length( generateRing, maxParticles ),
    };

    const MetalComputePipelines::Constants constants = {
        { 0, maxParticles },
        { 1, std::bit_cast< uint32_t >( particleRadius ) },
    };

    id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
    enc.label                          = @"Fluid Particles Simulate";
    if( generate )
    {
        shaders.DispatchThreadgroups( enc,
                                      "Fluid_Generate",
                                      constants,
                                      b,
                                      GroupCount( push.generateRingLength, COMPUTE_FLUID_PARTICLES_GENERATE_GROUP_SIZE_X ),
                                      1,
                                      1,
                                      AsBytes( push ) );
    }
    shaders.DispatchThreadgroups( enc,
                                  "Fluid_Particles",
                                  constants,
                                  b,
                                  GroupCount( push.activeRingLength, COMPUTE_FLUID_PARTICLES_GROUP_SIZE_X ),
                                  1,
                                  1,
                                  AsBytes( push ) );
    [enc endEncoding];

    // Debug: RTGL1_METAL_DEBUG_FLUID_DUMP, print some particles
    static uint32_t dumpCounter = 0;
    if( std::getenv( "RTGL1_METAL_DEBUG_FLUID_DUMP" ) && ( dumpCounter++ % 30 ) == 0 && push.activeRingLength > 0 )
    {
        const uint32_t  n    = std::min( 4096u, push.activeRingLength );
        id< MTLBuffer > read = [device newBufferWithLength:n * sizeof( ShParticleDef )
                                                   options:MTLResourceStorageModeShared];
        id< MTLBlitCommandEncoder > blit = [cmd blitCommandEncoder];
        const uint32_t first = active.begin;
        const uint32_t cnt   = std::min( n, maxParticles - first );
        [blit copyFromBuffer:particles
                sourceOffset:first * sizeof( ShParticleDef )
                    toBuffer:read
           destinationOffset:0
                        size:cnt * sizeof( ShParticleDef )];
        [blit endEncoding];
        const uint32_t activeLen = push.activeRingLength;
        [cmd addCompletedHandler:^( id< MTLCommandBuffer > ) {
          const auto* p = static_cast< const ShParticleDef* >( read.contents );
          float    minZ = 1e9f, maxZ = -1e9f;
          uint32_t fallen = 0;
          for( uint32_t i = 0; i < cnt; i++ )
          {
              minZ = std::min( minZ, p[ i ].position[ 2 ] );
              maxZ = std::max( maxZ, p[ i ].position[ 2 ] );
              fallen += p[ i ].position[ 2 ] < -1.0f ? 1 : 0;
          }
          debug::Info( "FLUIDDUMP active {} p0 ({:.2f} {:.2f} {:.2f}) v0 ({:.2f} {:.2f} {:.2f}) z [{:.2f}, {:.2f}] fallen {} of {}",
                       activeLen, p[ 0 ].position[ 0 ], p[ 0 ].position[ 1 ], p[ 0 ].position[ 2 ],
                       p[ 0 ].velocity[ 0 ], p[ 0 ].velocity[ 1 ], p[ 0 ].velocity[ 2 ], minZ, maxZ, fallen, cnt );
        }];
    }
}

void RTGL1::MetalFluid::Visualize( id< MTLCommandBuffer > cmd,
                                   MetalComputePipelines& shaders,
                                   MetalBindingTable&     b,
                                   id< MTLTexture >       depthFluid,
                                   id< MTLTexture >       fluidNormal,
                                   const float*           view,
                                   const float*           projection,
                                   uint32_t               width,
                                   uint32_t               height,
                                   float                  zNear,
                                   float                  zFar )
{
    if( !Active() )
    {
        return;
    }

    if( !depth || depth.width != width || depth.height != height )
    {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
        d.usage       = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;
        depth         = [device newTextureWithDescriptor:d];
        depth.label   = @"Fluid - Raster depth";
    }

    BindBuffers( b, 0 );

    const MetalComputePipelines::Constants constants = {
        { 0, maxParticles },
        { 1, std::bit_cast< uint32_t >( particleRadius ) },
    };

    // the sphere impostors
    {
        auto* pass                             = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[ 0 ].texture     = fluidNormal;
        pass.colorAttachments[ 0 ].loadAction  = MTLLoadActionClear;
        pass.colorAttachments[ 0 ].clearColor  = MTLClearColorMake( 4294967295.0, 4294967295.0, 4294967295.0, 4294967295.0 );
        pass.colorAttachments[ 0 ].storeAction = MTLStoreActionStore;
        pass.depthAttachment.texture           = depth;
        pass.depthAttachment.loadAction        = MTLLoadActionClear;
        pass.depthAttachment.clearDepth        = 1.0;
        pass.depthAttachment.storeAction       = MTLStoreActionStore;

        id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
        enc.label                         = @"Fluid Particles Visualize";

        auto desc = MetalComputePipelines::RenderDesc{
            .vertexShader       = "Fluid_Visualize_vert",
            .fragmentShader     = "Fluid_Visualize_frag",
            .vertexConstants    = constants,
            .fragmentConstants  = constants,
            .depthFormat        = MTLPixelFormatDepth32Float,
            .rasterVertexLayout = false,
        };
        desc.colorFormats[ 0 ] = fluidNormal.pixelFormat;

        auto push = VisualizePush_T{};
        // no jittering
        memcpy( push.proj, projection, sizeof( push.proj ) );
        memcpy( push.view, view, sizeof( push.view ) );

        shaders.BindRender( enc, desc, b, AsBytes( push ) );
        [enc setDepthStencilState:shaders.GetDepthState( true, true )];
        [enc setViewport:MTLViewport{ 0, 0, double( width ), double( height ), 0, 1 }];
        [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip
                vertexStart:0
                vertexCount:QUAD_VERTEX_COUNT
              instanceCount:Length( active, maxParticles )
               baseInstance:active.begin];
        [enc endEncoding];
    }

    id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoder];
    enc.label                          = @"Fluid Smoothing";

    // framebufDepthFluid <- the raster depth
    [enc setComputePipelineState:copyDepth];
    [enc setTexture:depth atIndex:0];
    [enc setTexture:depthFluid atIndex:1];
    [enc dispatchThreads:MTLSizeMake( width, height, 1 ) threadsPerThreadgroup:MTLSizeMake( 8, 8, 1 )];

    const auto push = FluidSmoothPush_T{
        .renderWidth  = width,
        .renderHeight = height,
        .zNear        = zNear,
        .zFar         = zFar,
    };
    for( uint32_t iter = 0; iter < SMOOTH_ITERATIONS; iter++ )
    {
        shaders.DispatchThreadgroups( enc,
                                      "Fluid_DepthSmooth",
                                      { { 0, iter } },
                                      b,
                                      GroupCount( width, COMPUTE_EFFECT_GROUP_SIZE_X ),
                                      GroupCount( height, COMPUTE_EFFECT_GROUP_SIZE_Y ),
                                      1,
                                      AsBytes( push ) );
    }
    [enc endEncoding];
}
