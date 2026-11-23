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

#include "MetalProfiler.h"

#include "DebugPrint.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace
{

constexpr uint32_t MaxEncoders = 384;
// Thief RT: every frame is sampled (to catch the slow ones, below), so a buffer comes
// round again a few frames on: enough of them that a frame's is read before it's reused
constexpr uint32_t RingSize = 8;

struct Frame
{
    id< MTLCounterSampleBuffer > samples{ nil };
    // to read their labels when resolving: the labels are set after the encoders are made
    std::vector< id >            encoders;
    // printed whatever it took (the frame every 2 seconds); else only if it's a spike
    bool                         always{ false };
};

// Thief RT: a frame that takes the GPU far longer than the frames before it (a hitch of a
// second or so) is printed whole -- "Metal GPU spike" -- with what it took and the
// average: the timeline every 2 seconds rarely caught one.
struct Spikes
{
    std::mutex                            m;
    double                                avgMs{ 0.0 };
    uint64_t                              frames{ 0 };
    uint32_t                              told{ 0 };
    std::chrono::steady_clock::time_point start{ std::chrono::steady_clock::now() };
    std::chrono::steady_clock::time_point lastTold{};
    uint32_t                              toldThisSecond{ 0 };
};
Spikes g_spikes;

struct State
{
    bool                         enabled{ false };
    id< MTLCounterSampleBuffer > rings[ RingSize ]{};
    uint32_t                     ring{ 0 };
    // the current frame, if it's sampled
    std::shared_ptr< Frame >     frame;
};

State g_state;

// a free pair of sample indices in the current frame, or nullopt
bool NextSlot( NSUInteger& start, NSUInteger& end )
{
    if( !g_state.frame || g_state.frame->encoders.size() >= MaxEncoders )
    {
        return false;
    }
    start = g_state.frame->encoders.size() * 2;
    end   = start + 1;
    return true;
}

void Track( id enc )
{
    if( g_state.frame && enc )
    {
        g_state.frame->encoders.push_back( enc );
    }
}

std::string LabelOf( id enc )
{
    NSString* l = nil;
    if( [enc respondsToSelector:@selector( label )] )
    {
        l = [enc label];
    }
    return l ? std::string{ l.UTF8String } : std::string{ "(no label)" };
}

void Print( const Frame& f )
{
    const NSUInteger count = f.encoders.size();
    const bool       always = f.always;
    if( count == 0 )
    {
        return;
    }

    NSData* data = [f.samples resolveCounterRange:NSMakeRange( 0, count * 2 )];
    if( !data )
    {
        return;
    }
    const auto* ts = static_cast< const MTLCounterResultTimestamp* >( data.bytes );

    struct Entry
    {
        uint64_t    start, end;
        std::string name;
    };
    std::vector< Entry > entries;
    entries.reserve( count );
    for( NSUInteger i = 0; i < count; i++ )
    {
        const uint64_t a = ts[ i * 2 ].timestamp;
        const uint64_t b = ts[ i * 2 + 1 ].timestamp;
        if( a == MTLCounterErrorValue || b == MTLCounterErrorValue || a == 0 || b < a )
        {
            continue;
        }
        entries.push_back( { a, b, LabelOf( f.encoders[ i ] ) } );
    }
    if( entries.empty() )
    {
        return;
    }
    std::ranges::sort( entries, []( const Entry& x, const Entry& y ) { return x.start < y.start; } );

    // Apple GPUs' timestamps are in nanoseconds
    auto ms = []( uint64_t ns ) { return double( ns ) / 1e6; };

    const uint64_t first = entries.front().start;
    uint64_t       last  = 0;
    uint64_t       busy  = 0;
    {
        uint64_t curStart = entries.front().start, curEnd = entries.front().end;
        for( const Entry& e : entries )
        {
            last = std::max( last, e.end );
            if( e.start > curEnd )
            {
                busy += curEnd - curStart;
                curStart = e.start;
                curEnd   = e.end;
            }
            else
            {
                curEnd = std::max( curEnd, e.end );
            }
        }
        busy += curEnd - curStart;
    }

    // a spike: far longer than the average so far (once there is one to go by)
    const double span = ms( last - first );
    bool         spike = false;
    double       avg   = 0.0;
    double       at    = 0.0;
    {
        std::lock_guard l( g_spikes.m );
        avg = g_spikes.avgMs;
        const auto now = std::chrono::steady_clock::now();
        at             = std::chrono::duration< double >( now - g_spikes.start ).count();
        if( g_spikes.frames >= 120 && span > std::max( avg * 1.8, avg + 12.0 ) && g_spikes.told < 120 )
        {
            // (a few of a hitch's frames, not all of them)
            if( now - g_spikes.lastTold > std::chrono::seconds( 1 ) )
            {
                g_spikes.lastTold       = now;
                g_spikes.toldThisSecond = 0;
            }
            if( g_spikes.toldThisSecond < 3 )
            {
                g_spikes.toldThisSecond++;
                g_spikes.told++;
                spike = true;
            }
        }
        // (a spike counts for little in the average: it's to tell the next one by)
        const double w     = g_spikes.frames < 60 ? 0.2 : 0.03;
        const double taken = std::min( span, std::max( avg, 1.0 ) * 3.0 );
        g_spikes.avgMs     = g_spikes.frames == 0 ? span : g_spikes.avgMs + ( taken - g_spikes.avgMs ) * w;
        g_spikes.frames++;
    }
    if( !always && !spike )
    {
        return;
    }
    // (a spike under its own name: the timeline every 2 seconds is averaged into the game's log)
    const char* kind = always ? "Metal GPU timeline" : "Metal GPU spike";
    if( !always )
    {
        RTGL1::debug::Warning( "{}: a frame took {:.2f} ms, the frames before it {:.2f} on average ({:.1f} s since start)",
                               kind, span, avg, at );
    }

    RTGL1::debug::Warning( "{}: {:.2f} ms from the first encoder's start to the last one's end, "
                           "{:.2f} ms with any encoder running, {} encoders",
                           kind,
                           ms( last - first ),
                           ms( busy ),
                           entries.size() );

    uint64_t prevEnd = first;
    for( const Entry& e : entries )
    {
        if( e.start > prevEnd && e.start - prevEnd >= 200'000 )
        {
            RTGL1::debug::Warning( "{}: +{:>7.2f}  {:>6.2f} ms  (no encoder: MetalFX, or waiting)",
                                   kind,
                                   ms( prevEnd - first ),
                                   ms( e.start - prevEnd ) );
        }
        RTGL1::debug::Warning(
            "{}: +{:>7.2f}  {:>6.2f} ms  {}", kind, ms( e.start - first ), ms( e.end - e.start ), e.name );
        prevEnd = std::max( prevEnd, e.end );
    }
}

}

void RTGL1::MetalProfiler::Init( id< MTLDevice > device )
{
    const char* e = std::getenv( "RTGL1_METAL_PROFILE" );
    if( !e || e[ 0 ] != '1' )
    {
        return;
    }

    id< MTLCounterSet > timestampSet = nil;
    for( id< MTLCounterSet > set in device.counterSets )
    {
        if( [set.name isEqualToString:MTLCommonCounterSetTimestamp] )
        {
            timestampSet = set;
        }
    }
    if( !timestampSet || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] )
    {
        debug::Warning( "Metal backend: GPU timestamps are not supported, there will be no GPU timeline" );
        return;
    }

    for( auto& r : g_state.rings )
    {
        MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet                        = timestampSet;
        d.storageMode                       = MTLStorageModeShared;
        d.sampleCount                       = MaxEncoders * 2;
        d.label                             = @"RTGL1 profiler";

        NSError* error = nil;
        r              = [device newCounterSampleBufferWithDescriptor:d error:&error];
        if( !r )
        {
            debug::Warning( "Metal backend: can't make a counter sample buffer: {}",
                            error ? error.localizedDescription.UTF8String : "(no info)" );
            return;
        }
    }
    g_state.enabled = true;
}

bool RTGL1::MetalProfiler::Enabled()
{
    return g_state.enabled;
}

void RTGL1::MetalProfiler::BeginFrame( bool sample )
{
    g_state.frame = nullptr;
    // (every frame: a spike is told when it comes; RTGL1_METAL_PROFILE_SPIKES=0, only the sampled ones)
    static const bool every = [] {
        const char* e = std::getenv( "RTGL1_METAL_PROFILE_SPIKES" );
        return !( e && e[ 0 ] == '0' );
    }();
    if( g_state.enabled && ( sample || every ) )
    {
        g_state.ring  = ( g_state.ring + 1 ) % RingSize;
        g_state.frame = std::make_shared< Frame >();
        g_state.frame->samples = g_state.rings[ g_state.ring ];
        g_state.frame->encoders.reserve( 128 );
        g_state.frame->always = sample;
    }
}

void RTGL1::MetalProfiler::EndFrame( id< MTLCommandBuffer > last )
{
    if( auto f = std::move( g_state.frame ) )
    {
        [last addCompletedHandler:^( id< MTLCommandBuffer > ) {
          Print( *f );
        }];
    }
    g_state.frame = nullptr;
}

id< MTLComputeCommandEncoder > RTGL1::MetalProfiler::Compute( id< MTLCommandBuffer > cmd )
{
    NSUInteger a, b;
    if( !NextSlot( a, b ) )
    {
        return [cmd computeCommandEncoder];
    }
    MTLComputePassDescriptor* desc                              = [MTLComputePassDescriptor computePassDescriptor];
    desc.sampleBufferAttachments[ 0 ].sampleBuffer              = g_state.frame->samples;
    desc.sampleBufferAttachments[ 0 ].startOfEncoderSampleIndex = a;
    desc.sampleBufferAttachments[ 0 ].endOfEncoderSampleIndex   = b;

    id< MTLComputeCommandEncoder > enc = [cmd computeCommandEncoderWithDescriptor:desc];
    Track( enc );
    return enc;
}

id< MTLBlitCommandEncoder > RTGL1::MetalProfiler::Blit( id< MTLCommandBuffer > cmd )
{
    NSUInteger a, b;
    if( !NextSlot( a, b ) )
    {
        return [cmd blitCommandEncoder];
    }
    MTLBlitPassDescriptor* desc                                 = [MTLBlitPassDescriptor blitPassDescriptor];
    desc.sampleBufferAttachments[ 0 ].sampleBuffer              = g_state.frame->samples;
    desc.sampleBufferAttachments[ 0 ].startOfEncoderSampleIndex = a;
    desc.sampleBufferAttachments[ 0 ].endOfEncoderSampleIndex   = b;

    id< MTLBlitCommandEncoder > enc = [cmd blitCommandEncoderWithDescriptor:desc];
    Track( enc );
    return enc;
}

id< MTLRenderCommandEncoder > RTGL1::MetalProfiler::Render( id< MTLCommandBuffer > cmd, MTLRenderPassDescriptor* pass )
{
    NSUInteger a, b;
    if( !NextSlot( a, b ) )
    {
        pass.sampleBufferAttachments[ 0 ].sampleBuffer = nil;
        return [cmd renderCommandEncoderWithDescriptor:pass];
    }
    // Thief RT: the fragment stage's start to its end -- a pass's vertex stage starts as soon
    // as it can, and its fragments then wait for what the pass reads (the swapchain blit waited
    // for the frame's post-processing: 2 ms of waiting, shown as its time)
    auto* s                       = pass.sampleBufferAttachments[ 0 ];
    s.sampleBuffer                = g_state.frame->samples;
    s.startOfVertexSampleIndex    = MTLCounterDontSample;
    s.endOfVertexSampleIndex      = MTLCounterDontSample;
    s.startOfFragmentSampleIndex  = a;
    s.endOfFragmentSampleIndex    = b;

    id< MTLRenderCommandEncoder > enc = [cmd renderCommandEncoderWithDescriptor:pass];
    Track( enc );
    return enc;
}

id< MTLAccelerationStructureCommandEncoder > RTGL1::MetalProfiler::AccelerationStructure( id< MTLCommandBuffer > cmd )
{
    NSUInteger a, b;
    if( !NextSlot( a, b ) )
    {
        return [cmd accelerationStructureCommandEncoder];
    }
    MTLAccelerationStructurePassDescriptor* desc =
        [MTLAccelerationStructurePassDescriptor accelerationStructurePassDescriptor];
    desc.sampleBufferAttachments[ 0 ].sampleBuffer              = g_state.frame->samples;
    desc.sampleBufferAttachments[ 0 ].startOfEncoderSampleIndex = a;
    desc.sampleBufferAttachments[ 0 ].endOfEncoderSampleIndex   = b;

    id< MTLAccelerationStructureCommandEncoder > enc = [cmd accelerationStructureCommandEncoderWithDescriptor:desc];
    Track( enc );
    return enc;
}
