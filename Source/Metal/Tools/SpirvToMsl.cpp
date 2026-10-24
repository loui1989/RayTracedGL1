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

// Build-time tool: SPIR-V compute / vertex / fragment shader -> MSL, for CompileShadersMetal.py.
//
// The same as 'spirv-cross --msl --msl-argument-buffers ...', but buffer blocks without
// an instance name are named after their block type (e.g. "GeometryInstances_BT" -> "GeometryInstances"),
// instead of "m_<id>". So every member of a descriptor set's argument buffer has a stable name,
// and the host fills argument buffers by member names (from Metal pipeline reflection).
//
// Also writes a compute shader's local_size as "x y z" (Metal pipelines don't expose it).
// Vertex shaders flip Y: Vulkan's clip space Y is down, Metal's is up.
//
// Usage: SpirvToMsl <input.spv> <output.metal> <entry point name> <output local size .txt, or - for none>

// from the spirv-cross-msl CMake target, not to pick up another installation's headers
#include <spirv_msl.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{

std::vector< uint32_t > ReadSpirv( const char* path )
{
    std::ifstream f( path, std::ios::binary );
    if( !f )
    {
        throw std::runtime_error( std::string{ "Can't open " } + path );
    }
    std::vector< char > bytes{ std::istreambuf_iterator< char >( f ), {} };
    if( bytes.size() % sizeof( uint32_t ) != 0 )
    {
        throw std::runtime_error( std::string{ "Not a SPIR-V file: " } + path );
    }
    std::vector< uint32_t > words( bytes.size() / sizeof( uint32_t ) );
    memcpy( words.data(), bytes.data(), bytes.size() );
    return words;
}

std::string StripSuffix( std::string s, std::string_view suffix )
{
    if( s.size() > suffix.size() && s.ends_with( suffix ) )
    {
        s.resize( s.size() - suffix.size() );
    }
    return s;
}

}

int main( int argc, char* argv[] )
{
    if( argc != 5 )
    {
        fprintf( stderr,
                 "Usage: %s <input.spv> <output.metal> <entry point name> <output local size .txt>\n",
                 argv[ 0 ] );
        return 1;
    }

    try
    {
        auto msl = spirv_cross::CompilerMSL{ ReadSpirv( argv[ 1 ] ) };

        auto opts                                 = msl.get_msl_options();
        opts.set_msl_version( 3, 1 );
        opts.argument_buffers                     = true;
        opts.argument_buffers_tier                = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
        // all resources of a set, even if unused by a shader:
        // a set's argument buffer has the same members for all shaders
        opts.force_active_argument_buffer_resources = true;
        // Vulkan allows a fragment output with fewer components than its attachment has (vec3 to RGBA16F),
        // Metal doesn't
        opts.pad_fragment_output_components = true;
        msl.set_msl_options( opts );

        // device address space, to allow runtime sized arrays and large argument buffers
        for( uint32_t set = 0; set < 16; set++ )
        {
            msl.set_argument_buffer_device_address_space( set, true );
        }

        // name anonymous buffer blocks after their type
        const auto resources = msl.get_shader_resources();
        for( const auto* list : { &resources.uniform_buffers, &resources.storage_buffers } )
        {
            for( const auto& r : *list )
            {
                if( msl.get_name( r.id ).empty() )
                {
                    msl.set_name( r.id, StripSuffix( msl.get_name( r.base_type_id ), "_BT" ) );
                }
            }
        }

        const spv::ExecutionModel model = msl.get_entry_points_and_stages()[ 0 ].execution_model;
        if( model == spv::ExecutionModelVertex )
        {
            auto common               = msl.get_common_options();
            common.vertex.flip_vert_y = true;
            msl.set_common_options( common );
        }

        msl.rename_entry_point( "main", argv[ 3 ], model );

        std::ofstream out( argv[ 2 ] );
        std::string source = msl.compile();
        // RTGL1's acceleration structures have only triangles:
        // let Metal skip the bounding box intersection path of the ray queries
        {
            const std::string what = "    intersection_params ip;\n";
            if( auto i = source.find( what ); i != std::string::npos )
            {
                source.insert( i + what.size(), "    ip.assume_geometry_type(geometry_type::triangle);\n" );
            }
        }
        out << source;
        if( !out )
        {
            throw std::runtime_error( std::string{ "Can't write " } + argv[ 2 ] );
        }

        if( model != spv::ExecutionModelGLCompute || std::string_view{ argv[ 4 ] } == "-" )
        {
            return 0;
        }

        // local_size, possibly from specialization constants (with their default values)
        uint32_t size[ 3 ];
        {
            spirv_cross::SpecializationConstant spec[ 3 ];
            msl.get_work_group_size_specialization_constants( spec[ 0 ], spec[ 1 ], spec[ 2 ] );
            for( uint32_t i = 0; i < 3; i++ )
            {
                size[ i ] = spec[ i ].id != spirv_cross::ID{ 0 }
                                ? msl.get_constant( spec[ i ].id ).scalar()
                                : msl.get_execution_mode_argument( spv::ExecutionModeLocalSize, i );
            }
        }
        std::ofstream threads( argv[ 4 ] );
        threads << size[ 0 ] << " " << size[ 1 ] << " " << size[ 2 ] << "\n";
        if( !threads )
        {
            throw std::runtime_error( std::string{ "Can't write " } + argv[ 4 ] );
        }
    }
    catch( const std::exception& e )
    {
        fprintf( stderr, "%s: %s\n", argv[ 1 ], e.what() );
        return 1;
    }
    return 0;
}
