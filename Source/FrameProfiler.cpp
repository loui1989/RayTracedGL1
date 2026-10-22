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

#include "FrameProfiler.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace RTGL1;

namespace
{
constexpr VkDeviceSize BREAK_COPY_SIZE = 256;
}

bool FrameProfiler::IsRequested()
{
    const char* e = getenv( "RTGL1_PROFILE" );
    return e && e[ 0 ] == '1';
}

FrameProfiler::FrameProfiler( VkDevice                                  _device,
                              VkPhysicalDevice                          _physDevice,
                              const std::shared_ptr< MemoryAllocator >& _allocator )
    : device( _device )
    , pool( VK_NULL_HANDLE )
    , periodNs( 0 )
    , curFrameIndex( 0 )
    , marks{}
    , gpuFrames( 0 )
    , cpuFrames( 0 )
    , lastCpuMark( std::chrono::steady_clock::now() )
    , lastReport( std::chrono::steady_clock::now() )
{
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties( _physDevice, &props );
    if( props.limits.timestampPeriod <= 0 )
    {
        fprintf( stderr, "[Profile] GPU timestamps are not supported\n" );
        return;
    }
    periodNs = props.limits.timestampPeriod;

    VkQueryPoolCreateInfo qi = {};
    qi.sType                 = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType             = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount            = MAX_MARKS * MAX_FRAMES_IN_FLIGHT;
    if( vkCreateQueryPool( device, &qi, nullptr, &pool ) != VK_SUCCESS )
    {
        pool = VK_NULL_HANDLE;
        fprintf( stderr, "[Profile] couldn't create a timestamp query pool\n" );
        return;
    }

    breakBuffer.Init( _allocator,
                      BREAK_COPY_SIZE * 2,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      "Profiler encoder break" );

    fprintf( stderr, "[Profile] enabled, averages every 2 seconds\n" );
}

FrameProfiler::~FrameProfiler()
{
    breakBuffer.Destroy();
    if( pool != VK_NULL_HANDLE )
    {
        vkDestroyQueryPool( device, pool, nullptr );
    }
}

void FrameProfiler::Add( std::vector< Acc >& v, const char* name, double ms )
{
    for( Acc& a : v )
    {
        if( a.name == name || std::string( a.name ) == name )
        {
            a.sum += ms;
            return;
        }
    }
    v.push_back( { name, ms } );
}

void FrameProfiler::BeginFrame( VkCommandBuffer cmd, uint32_t frameIndex )
{
    if( !IsValid() )
    {
        return;
    }
    assert( frameIndex < MAX_FRAMES_IN_FLIGHT );

    // results of this frame index's previous use: its fence was waited for
    auto& m = marks[ frameIndex ];
    if( m.count > 1 )
    {
        uint64_t ts[ MAX_MARKS ] = {};
        VkResult r               = vkGetQueryPoolResults( device,
                                            pool,
                                            frameIndex * MAX_MARKS,
                                            m.count,
                                            sizeof( ts ),
                                            ts,
                                            sizeof( uint64_t ),
                                            VK_QUERY_RESULT_64_BIT );
        if( r == VK_SUCCESS )
        {
            for( uint32_t i = 1; i < m.count; i++ )
            {
                const double ms = ts[ i ] > ts[ i - 1 ]
                                      ? double( ts[ i ] - ts[ i - 1 ] ) * periodNs / 1e6
                                      : 0.0;
                Add( gpuAcc, m.names[ i ], ms );
            }
            Add( gpuAcc,
                 "= total",
                 ts[ m.count - 1 ] > ts[ 0 ] ? double( ts[ m.count - 1 ] - ts[ 0 ] ) * periodNs / 1e6
                                             : 0.0 );
            gpuFrames++;
        }
    }

    curFrameIndex = frameIndex;
    m.count       = 0;
    cpuFrames++;
    vkCmdResetQueryPool( cmd, pool, frameIndex * MAX_MARKS, MAX_MARKS );
    Gpu( cmd, "start" );
}

void FrameProfiler::Gpu( VkCommandBuffer cmd, const char* name )
{
    if( !IsValid() )
    {
        return;
    }
    auto& m = marks[ curFrameIndex ];
    if( m.count >= MAX_MARKS )
    {
        return;
    }

    vkCmdWriteTimestamp(
        cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, curFrameIndex * MAX_MARKS + m.count );
    m.names[ m.count ] = name;
    m.count++;

    // end the current Metal encoder, so MoltenVK samples the timestamp here
    // instead of at the end of a long batch of dispatches
    VkBufferCopy region = { 0, BREAK_COPY_SIZE, BREAK_COPY_SIZE };
    vkCmdCopyBuffer( cmd, breakBuffer.GetBuffer(), breakBuffer.GetBuffer(), 1, &region );
}

void FrameProfiler::Cpu( const char* name )
{
    if( !IsValid() )
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    Add( cpuAcc, name, std::chrono::duration< double, std::milli >( now - lastCpuMark ).count() );
    lastCpuMark = now;

    if( now - lastReport > std::chrono::seconds( 2 ) )
    {
        Report( std::chrono::duration< double >( now - lastReport ).count() );
        lastReport = now;
    }
}

void FrameProfiler::Value( const char* name, double ms )
{
    if( IsValid() )
    {
        Add( valueAcc, name, ms );
    }
}

void FrameProfiler::Report( double seconds )
{
    std::string  s;
    char         buf[ 128 ];

    snprintf( buf, sizeof( buf ), "[Profile] %.1f fps\n", double( cpuFrames ) / seconds );
    s += buf;

    s += "[Profile]   GPU ms:";
    for( const Acc& a : gpuAcc )
    {
        snprintf( buf, sizeof( buf ), " %s %.2f |", a.name, gpuFrames ? a.sum / gpuFrames : 0.0 );
        s += buf;
    }
    s += "\n[Profile]   CPU ms:";
    double total = 0;
    for( const Acc& a : cpuAcc )
    {
        snprintf( buf, sizeof( buf ), " %s %.2f |", a.name, cpuFrames ? a.sum / cpuFrames : 0.0 );
        s += buf;
        total += a.sum;
    }
    snprintf( buf, sizeof( buf ), " = total %.2f", cpuFrames ? total / cpuFrames : 0.0 );
    s += buf;
    if( !valueAcc.empty() )
    {
        s += "\n[Profile]   other ms:";
        for( const Acc& a : valueAcc )
        {
            snprintf( buf, sizeof( buf ), " %s %.2f |", a.name, cpuFrames ? a.sum / cpuFrames : 0.0 );
            s += buf;
        }
    }
    s += "\n";
    fputs( s.c_str(), stderr );

    gpuAcc.clear();
    cpuAcc.clear();
    valueAcc.clear();
    gpuFrames = 0;
    cpuFrames = 0;
}
