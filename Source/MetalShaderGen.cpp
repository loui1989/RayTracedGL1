// Copyright (c) 2026 Quake RT for macOS contributors
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

#include "MetalShaderGen.h"

#include <algorithm>
#include <cctype>
#include <exception>

#ifdef RG_HAS_SPIRV_CROSS
#include <spirv_msl.hpp>
#endif

namespace RTGL1::MetalGen
{

namespace
{
struct ResourceCounts
{
    uint32_t textures = 0;
    uint32_t buffers  = 0;
    uint32_t samplers = 0;
};

ResourceCounts CountsOf( VkDescriptorType type )
{
    switch( type )
    {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return { 1, 0, 0 };
        case VK_DESCRIPTOR_TYPE_SAMPLER: return { 0, 0, 1 };
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return { 1, 0, 1 };
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return { 0, 1, 0 };
        default: return {};
    }
}

#ifdef RG_HAS_SPIRV_CROSS
spirv_cross::SPIRType::BaseType BaseTypeOf( VkDescriptorType type )
{
    using T = spirv_cross::SPIRType;
    switch( type )
    {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return T::Image;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return T::Sampler;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return T::SampledImage;
        case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: return T::AccelerationStructure;
        default: return T::Void;
    }
}
#endif
}

const SlotInfo* SetSlots::Find( uint32_t binding ) const
{
    for( const SlotInfo& s : slots )
    {
        if( s.binding == binding )
        {
            return &s;
        }
    }
    return nullptr;
}

SetSlots AssignSlots( std::vector< BindingDesc > bindings )
{
    std::sort( bindings.begin(), bindings.end(), []( const BindingDesc& a, const BindingDesc& b ) {
        return a.binding < b.binding;
    } );

    SetSlots r;
    uint32_t cursor = 0;
    for( const BindingDesc& b : bindings )
    {
        const ResourceCounts c = CountsOf( b.type );

        SlotInfo s    = {};
        s.binding     = b.binding;
        s.type        = b.type;
        s.count       = b.count;
        s.textureSlot = cursor;
        s.bufferSlot  = cursor + c.textures * b.count;
        s.samplerSlot = cursor + c.textures * b.count;
        r.slots.push_back( s );

        cursor += ( c.textures + c.buffers + c.samplers ) * b.count;
    }
    r.totalSlots = cursor;
    return r;
}

namespace
{
ComputeShader Translate( const uint32_t*                spirv,
                         size_t                         wordCount,
                         const std::vector< SetSlots >& sets,
                         uint32_t                       mslVersion,
                         bool                           graphics,
                         bool                           multiview )
{
    ComputeShader result;
#ifndef RG_HAS_SPIRV_CROSS
    result.error = "built without SPIRV-Cross (RG_SPIRV_CROSS_DIR)";
    return result;
#else

    try
    {
        spirv_cross::CompilerMSL msl( spirv, wordCount );

        spirv_cross::CompilerMSL::Options o = msl.get_msl_options();
        o.platform                          = spirv_cross::CompilerMSL::Options::macOS;
        o.msl_version                       = mslVersion;
        o.argument_buffers                  = true;
        o.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
        o.pad_argument_buffer_resources          = true;
        o.force_active_argument_buffer_resources = false;
        o.texture_buffer_native                  = true;
        o.emulate_subgroups                      = false;
        o.fixed_subgroup_size                    = 32;
        o.swizzle_texture_samples                = false;
        // implicit buffers, in case a shader needs one (then it's rejected below)
        o.swizzle_buffer_index         = 30;
        o.buffer_size_buffer_index     = 29;
        o.dynamic_offsets_buffer_index = 28;
        if( graphics )
        {
            // fragment outputs get as many components as the attachments have
            o.pad_fragment_output_components = true;
            // gl_ViewIndex: the draws are instanced once per view, rendering to the layer of
            // the view (the instance count is multiplied when drawing)
            o.multiview                    = multiview;
            o.multiview_layered_rendering  = multiview;
            o.view_index_from_device_index = false;
            o.view_mask_buffer_index       = VIEW_MASK_BUFFER_INDEX;
        }
        msl.set_msl_options( o );

        if( graphics )
        {
            // Vulkan's clip space has y pointing down, Metal's up
            auto common              = msl.get_common_options();
            common.vertex.flip_vert_y = true;
            msl.set_common_options( common );
        }

        spv::ExecutionModel model = spv::ExecutionModelGLCompute;
        {
            const auto eps = msl.get_entry_points_and_stages();
            if( !eps.empty() )
            {
                model = eps[ 0 ].execution_model;
            }
        }

        for( uint32_t set = 0; set < sets.size(); set++ )
        {
            for( const SlotInfo& s : sets[ set ].slots )
            {
                spirv_cross::MSLResourceBinding rb;
                rb.stage       = model;
                rb.basetype    = BaseTypeOf( s.type );
                rb.desc_set    = set;
                rb.binding     = s.binding;
                rb.count       = s.count;
                rb.msl_texture = s.textureSlot;
                rb.msl_buffer  = s.bufferSlot;
                rb.msl_sampler = s.samplerSlot;
                msl.add_msl_resource_binding( rb );
            }
        }

        // push constants: a separate buffer, set with setBytes
        {
            spirv_cross::MSLResourceBinding rb;
            rb.stage      = model;
            rb.basetype   = spirv_cross::SPIRType::Void;
            rb.desc_set   = spirv_cross::kPushConstDescSet;
            rb.binding    = spirv_cross::kPushConstBinding;
            rb.count      = 1;
            rb.msl_buffer = PUSH_CONSTANT_BUFFER_INDEX;
            msl.add_msl_resource_binding( rb );
        }

        result.msl = msl.compile();

        // Ray queries: the acceleration structures contain only triangles, so tell the
        // intersector it doesn't need to handle bounding box geometry
        {
            const std::string from = "    intersection_params ip;\n";
            const size_t      at   = result.msl.find( from );
            if( at != std::string::npos )
            {
                result.msl.insert( at + from.size(),
                                   "    ip.assume_geometry_type(geometry_type::triangle);\n" );
            }
        }

        if( msl.needs_buffer_size_buffer() || msl.needs_swizzle_buffer() ||
            msl.needs_acceleration_structure_address_table() )
        {
            result.error = "the shader needs an implicit buffer (buffer sizes, swizzles or an "
                           "acceleration structure table), which the native Metal path doesn't provide";
            return result;
        }

        const auto entryPoints = msl.get_entry_points_and_stages();
        if( entryPoints.empty() )
        {
            result.error = "no entry point";
            return result;
        }
        result.entryPoint =
            msl.get_cleansed_entry_point_name( entryPoints[ 0 ].name, entryPoints[ 0 ].execution_model );

        const auto& ep = msl.get_entry_point( entryPoints[ 0 ].name, entryPoints[ 0 ].execution_model );
        result.localSize[ 0 ] = std::max( 1u, ep.workgroup_size.x );
        result.localSize[ 1 ] = std::max( 1u, ep.workgroup_size.y );
        result.localSize[ 2 ] = std::max( 1u, ep.workgroup_size.z );

        result.ok = true;
    }
    catch( const std::exception& e )
    {
        result.error = e.what();
    }
    return result;
#endif
}
}

namespace
{

// ---- Metal's intersector instead of intersection queries
//
// SPIRV-Cross turns GL_EXT_ray_query into Metal intersection queries, which the shader
// drives candidate by candidate (RaygenCommon.h traceRayQuery / traceRayQueryAnyHit,
// with the alpha test rtAlphaTestAccepts on every candidate of non-opaque geometry).
// Apple's GPUs trace faster with the intersector, which traverses in one call and runs
// the alpha test as an intersection function. This rewrites the translated shader: the
// two trace functions get intersector bodies, the alpha test is called from an
// intersection function, and the resources it reads reach it as pointers in the ray
// payload. Anything unexpected -> no rewrite (the caller keeps the query version).

struct FuncPos
{
    size_t lineStart = std::string::npos; // start of the line with the name
    size_t paramsBegin = 0, paramsEnd = 0; // inside the parentheses
    size_t bodyBegin = 0, bodyEnd = 0;     // '{' .. one past '}'
};

bool FindFunction( const std::string& msl, const std::string& name, FuncPos& f )
{
    const std::string key = " " + name + "(";
    size_t            at  = 0;
    while( ( at = msl.find( key, at ) ) != std::string::npos )
    {
        const size_t ls = msl.rfind( '\n', at );
        const size_t lineStart = ls == std::string::npos ? 0 : ls + 1;
        // a definition: the line starts with its return type (no indentation)
        if( lineStart < msl.size() && msl[ lineStart ] != ' ' && msl[ lineStart ] != '\t' )
        {
            size_t p     = at + key.size();
            int    depth = 1;
            size_t q     = p;
            for( ; q < msl.size() && depth > 0; q++ )
            {
                depth += msl[ q ] == '(' ? 1 : msl[ q ] == ')' ? -1 : 0;
            }
            if( depth != 0 )
            {
                return false;
            }
            const size_t close = q - 1;
            size_t       b     = close + 1;
            while( b < msl.size() && ( msl[ b ] == ' ' || msl[ b ] == '\n' || msl[ b ] == '\r' ) )
            {
                b++;
            }
            if( b < msl.size() && msl[ b ] == '{' )
            {
                int    d = 0;
                size_t e = b;
                for( ; e < msl.size(); e++ )
                {
                    d += msl[ e ] == '{' ? 1 : msl[ e ] == '}' ? -1 : 0;
                    if( d == 0 )
                    {
                        break;
                    }
                }
                if( d != 0 )
                {
                    return false;
                }
                f.lineStart   = lineStart;
                f.paramsBegin = p;
                f.paramsEnd   = close;
                f.bodyBegin   = b;
                f.bodyEnd     = e + 1;
                return true;
            }
        }
        at += key.size();
    }
    return false;
}

std::vector< std::string > SplitParams( const std::string& params )
{
    std::vector< std::string > out;
    int                        depth = 0;
    std::string                cur;
    for( char c : params )
    {
        if( c == '<' || c == '(' || c == '[' )
        {
            depth++;
        }
        else if( c == '>' || c == ')' || c == ']' )
        {
            depth--;
        }
        if( c == ',' && depth == 0 )
        {
            out.push_back( cur );
            cur.clear();
            continue;
        }
        cur += c;
    }
    if( !cur.empty() )
    {
        out.push_back( cur );
    }
    for( auto& p : out )
    {
        const size_t a = p.find_first_not_of( " \t\n" );
        const size_t b = p.find_last_not_of( " \t\n" );
        p              = a == std::string::npos ? "" : p.substr( a, b - a + 1 );
    }
    return out;
}

// "thread const uint& rayFlags" -> "rayFlags"
std::string ParamName( const std::string& param )
{
    size_t e = param.size();
    while( e > 0 && ( isalnum( uint8_t( param[ e - 1 ] ) ) || param[ e - 1 ] == '_' ) )
    {
        e--;
    }
    return param.substr( e );
}

void ReplaceAll( std::string& s, const std::string& from, const std::string& to )
{
    for( size_t at = 0; ( at = s.find( from, at ) ) != std::string::npos; at += to.size() )
    {
        s.replace( at, from.size(), to );
    }
}

std::string RewriteForIntersector( std::string msl )
{
    const char* queryType = "raytracing::intersection_query<raytracing::instancing, raytracing::triangle_data>";
    const char* tags      = "raytracing::triangle_data, raytracing::instancing";

    FuncPos alpha, cand, closest, anyhit;
    if( !FindFunction( msl, "rtAlphaTestAccepts", alpha ) || !FindFunction( msl, "rqIsCandidateAccepted", cand ) )
    {
        return {};
    }
    const bool hasClosest = FindFunction( msl, "traceRayQuery", closest );
    const bool hasAnyHit  = FindFunction( msl, "traceRayQueryAnyHit", anyhit );
    if( !hasClosest && !hasAnyHit )
    {
        return {};
    }

    // the resources of the alpha test (after instanceId, primitiveId, b, media)
    const auto alphaParams = SplitParams( msl.substr( alpha.paramsBegin, alpha.paramsEnd - alpha.paramsBegin ) );
    if( alphaParams.size() < 4 )
    {
        return {};
    }
    struct Res
    {
        std::string type, name;
    };
    std::vector< Res > res;
    for( size_t i = 4; i < alphaParams.size(); i++ )
    {
        const std::string& p   = alphaParams[ i ];
        const size_t       amp = p.rfind( '&' );
        if( amp == std::string::npos || p.find( "thread" ) == 0 )
        {
            return {}; // a resource passed by value (e.g. a texture): not supported
        }
        res.push_back( { p.substr( 0, amp ), ParamName( p ) } );
    }

    // the bodies, built against each function's own parameter names
    auto makeTraceBody = [ & ]( const FuncPos& f, bool isAnyHit, std::string& body ) -> bool {
        const auto ps = SplitParams( msl.substr( f.paramsBegin, f.paramsEnd - f.paramsBegin ) );
        if( ps.size() < ( isAnyHit ? 7u : 6u ) )
        {
            return false;
        }
        std::string as, rq;
        for( const auto& p : ps )
        {
            if( p.find( "acceleration_structure" ) != std::string::npos )
            {
                as = ParamName( p );
            }
            if( p.find( queryType ) != std::string::npos )
            {
                rq = ParamName( p );
            }
        }
        if( as.empty() || rq.empty() )
        {
            return false;
        }
        for( const Res& r : res )
        {
            bool found = false;
            for( const auto& p : ps )
            {
                found = found || ParamName( p ) == r.name;
            }
            if( !found )
            {
                return false;
            }
        }
        const std::string flags = ParamName( ps[ 0 ] ), mask = ParamName( ps[ 1 ] ), origin = ParamName( ps[ 2 ] ),
                          tmin = ParamName( ps[ 3 ] ), dir = ParamName( ps[ 4 ] ), tmax = ParamName( ps[ 5 ] );

        body = "{\n    RtglAlphaPayload rtglPayload;\n";
        body += std::string( "    rtglPayload.media = " ) + ( isAnyHit ? ParamName( ps[ 6 ] ) : std::string( "false" ) ) + ";\n";
        for( const Res& r : res )
        {
            body += "    rtglPayload." + r.name + " = &" + r.name + ";\n";
        }
        body += std::string( "    raytracing::intersector<" ) + tags + "> rtglIt;\n";
        body += "    rtglConfigure( rtglIt, " + flags + ( isAnyHit ? " | 4u" : "" ) + " );\n";
        body += "    auto rtglHit = rtglIt.intersect( raytracing::ray( " + origin + ", " + dir + ", " + tmin + ", " + tmax +
                " ), " + as + ", " + mask + " & 255u, " + rq + ".ft, rtglPayload );\n";
        if( isAnyHit )
        {
            body += "    return rtglHit.type != raytracing::intersection_type::none;\n}";
        }
        else
        {
            body += "    ShPayload p;\n"
                    "    p.baryCoords = float2( 0.0 );\n"
                    "    p.instIdAndIndex = 4294967295u;\n"
                    "    p.geomAndPrimIndex = 4294967295u;\n"
                    "    if( rtglHit.type == raytracing::intersection_type::triangle )\n"
                    "    {\n"
                    "        p.baryCoords = rtglHit.triangle_barycentric_coord;\n"
                    "        p.instIdAndIndex = ( uint( rtglHit.instance_id ) << 4u ) | ( uint( rtglHit.user_instance_id ) & 15u );\n"
                    "        p.geomAndPrimIndex = uint( rtglHit.primitive_id );\n"
                    "    }\n"
                    "    return p;\n}";
        }
        return true;
    };

    // replace from the end, so the earlier positions stay valid
    struct Edit
    {
        size_t      begin, end;
        std::string text;
    };
    std::vector< Edit > edits;
    edits.push_back( { cand.bodyBegin, cand.bodyEnd, "{\n    return true;\n}" } );
    if( hasClosest )
    {
        std::string b;
        if( !makeTraceBody( closest, false, b ) )
        {
            return {};
        }
        edits.push_back( { closest.bodyBegin, closest.bodyEnd, b } );
    }
    if( hasAnyHit )
    {
        std::string b;
        if( !makeTraceBody( anyhit, true, b ) )
        {
            return {};
        }
        edits.push_back( { anyhit.bodyBegin, anyhit.bodyEnd, b } );
    }
    std::sort( edits.begin(), edits.end(), []( const Edit& a, const Edit& b ) { return a.begin > b.begin; } );
    for( const Edit& e : edits )
    {
        msl.replace( e.begin, e.end - e.begin, e.text );
    }

    // nothing else may use the query
    if( msl.find( ".next()" ) != std::string::npos || msl.find( "get_candidate" ) != std::string::npos ||
        msl.find( "get_committed" ) != std::string::npos || msl.find( ".reset(" ) != std::string::npos )
    {
        return {};
    }

    ReplaceAll( msl, queryType, "RtglTracer" );

    // the kernel: the function table as an argument, copied into the tracer
    const size_t kernel = msl.find( "kernel void main0(" );
    if( kernel == std::string::npos )
    {
        return {};
    }
    {
        int    depth = 0;
        size_t q     = msl.find( '(', kernel );
        for( ; q < msl.size(); q++ )
        {
            depth += msl[ q ] == '(' ? 1 : msl[ q ] == ')' ? -1 : 0;
            if( depth == 0 )
            {
                break;
            }
        }
        if( q >= msl.size() )
        {
            return {};
        }
        msl.insert( q,
                    std::string( ", raytracing::intersection_function_table<" ) + tags + "> rtglFunctionTable [[buffer(" +
                        std::to_string( INTERSECTION_FUNCTION_TABLE_BUFFER_INDEX ) + ")]]" );
    }
    {
        const size_t body = msl.find( '{', kernel );
        const size_t decl = msl.find( "    RtglTracer ", body );
        if( decl == std::string::npos )
        {
            return {};
        }
        const size_t semi = msl.find( ';', decl );
        const std::string name = msl.substr( decl + 15, semi - ( decl + 15 ) );
        msl.insert( semi + 1, "\n    " + name + ".ft = rtglFunctionTable;" );
    }

    // the types and the helper before the first function, the intersection function last
    std::string payload = "struct RtglAlphaPayload\n{\n    bool media;\n";
    for( const Res& r : res )
    {
        payload += "    " + r.type + "* " + r.name + ";\n";
    }
    payload += "};\n\n";
    const std::string types =
        payload + std::string( "struct RtglTracer\n{\n    raytracing::intersection_function_table<" ) + tags +
        "> ft;\n};\n\n"
        "// Vulkan ray flags -> intersector options (as spvMakeIntersectionParams)\n"
        "template< typename I >\n"
        "static inline __attribute__((always_inline)) void rtglConfigure( thread I& it, uint flags )\n"
        "{\n"
        "    if( ( flags & 1u ) != 0u ) it.force_opacity( raytracing::forced_opacity::opaque );\n"
        "    if( ( flags & 2u ) != 0u ) it.force_opacity( raytracing::forced_opacity::non_opaque );\n"
        "    if( ( flags & 4u ) != 0u ) it.accept_any_intersection( true );\n"
        "    if( ( flags & 16u ) != 0u ) it.set_triangle_cull_mode( raytracing::triangle_cull_mode::back );\n"
        "    if( ( flags & 32u ) != 0u ) it.set_triangle_cull_mode( raytracing::triangle_cull_mode::front );\n"
        "    if( ( flags & 64u ) != 0u ) it.set_opacity_cull_mode( raytracing::opacity_cull_mode::opaque );\n"
        "    if( ( flags & 128u ) != 0u ) it.set_opacity_cull_mode( raytracing::opacity_cull_mode::non_opaque );\n"
        "}\n\n";
    const size_t firstFn = msl.find( "static inline __attribute__((always_inline))" );
    if( firstFn == std::string::npos )
    {
        return {};
    }
    msl.insert( firstFn, types );

    // the tags in the attribute unqualified, as in Apple's samples
    std::string fn = std::string( "\nusing namespace metal::raytracing;\n\n[[intersection(triangle, triangle_data, instancing)]]\n" ) + "bool " + INTERSECTOR_FUNCTION +
                     "( uint rtglInstance [[instance_id]], uint rtglPrimitive [[primitive_id]],\n"
                     "    float2 rtglBary [[barycentric_coord]], ray_data RtglAlphaPayload& rtglPayload [[payload]] )\n"
                     "{\n"
                     "    const int  instanceId  = int( rtglInstance );\n"
                     "    const uint primitiveId = rtglPrimitive;\n"
                     "    const float2 b         = rtglBary;\n"
                     "    const bool media       = rtglPayload.media;\n"
                     "    return rtAlphaTestAccepts( instanceId, primitiveId, b, media";
    for( const Res& r : res )
    {
        fn += ", *rtglPayload." + r.name;
    }
    fn += " );\n}\n";
    msl += fn;
    return msl;
}

}

ComputeShader TranslateCompute( const uint32_t*                spirv,
                                size_t                         wordCount,
                                const std::vector< SetSlots >& sets,
                                uint32_t                       mslVersion )
{
    ComputeShader sh = Translate( spirv, wordCount, sets, mslVersion, false, false );
    if( sh.ok )
    {
        sh.mslIntersector = RewriteForIntersector( sh.msl );
        if( !sh.mslIntersector.empty() )
        {
            // the same with the function table itself as the tracer (in case the compiler
            // doesn't take a function table as a struct member)
            std::string alt = sh.mslIntersector;
            const std::string tableType =
                "raytracing::intersection_function_table<raytracing::triangle_data, raytracing::instancing>";
            const std::string structDef = "struct RtglTracer\n{\n    " + tableType + " ft;\n};\n";
            const size_t      sd        = alt.find( structDef );
            const size_t      decl      = alt.find( ".ft = rtglFunctionTable;" );
            if( sd != std::string::npos && decl != std::string::npos )
            {
                alt.replace( sd, structDef.size(), "typedef " + tableType + " RtglTracer;\n" );
                // "    RtglTracer rq;\n    rq.ft = rtglFunctionTable;" -> "    RtglTracer rq = rtglFunctionTable;"
                const size_t d2   = alt.find( ".ft = rtglFunctionTable;" );
                const size_t semi = alt.rfind( ';', d2 );
                alt.replace( semi, d2 + std::string( ".ft = rtglFunctionTable;" ).size() - semi,
                             " = rtglFunctionTable;" );
                // strip the member access everywhere else
                for( size_t at = 0; ( at = alt.find( ".ft, rtglPayload", at ) ) != std::string::npos; )
                {
                    alt.erase( at, 3 );
                }
                sh.mslIntersectorAlt = alt;
            }
        }
    }
    return sh;
}

ComputeShader TranslateGraphics( const uint32_t*                spirv,
                                 size_t                         wordCount,
                                 const std::vector< SetSlots >& sets,
                                 uint32_t                       mslVersion,
                                 bool                           multiview )
{
    return Translate( spirv, wordCount, sets, mslVersion, true, multiview );
}

}
