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

ComputeShader TranslateCompute( const uint32_t*                spirv,
                                size_t                         wordCount,
                                const std::vector< SetSlots >& sets,
                                uint32_t                       mslVersion )
{
    return Translate( spirv, wordCount, sets, mslVersion, false, false );
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
