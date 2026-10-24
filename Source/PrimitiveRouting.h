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

#pragma once

#include "RTGL1/RTGL1.h"

#include "Common.h"
#include "Const.h"
#include "DrawFrameInfo.h"
#include "Generated/ShaderCommonC.h"
#include "Utils.h"

namespace RTGL1
{

// Whether a primitive is drawn by the rasterizer instead of being ray traced.
// Shared by the Vulkan and Metal backends.
inline bool IsRasterized( const RgMeshInfo& mesh, const RgMeshPrimitiveInfo& primitive )
{
    if( primitive.flags & RG_MESH_PRIMITIVE_DECAL )
    {
        return true;
    }

    if( primitive.flags & RG_MESH_PRIMITIVE_SKY )
    {
        return true;
    }

    if( !( primitive.flags & RG_MESH_PRIMITIVE_GLASS ) &&
        !( mesh.flags & RG_MESH_FORCE_GLASS ) &&
        !( primitive.flags & RG_MESH_PRIMITIVE_WATER ) &&
        !( mesh.flags & RG_MESH_FORCE_WATER ) && !( primitive.flags & RG_MESH_PRIMITIVE_ACID ) )
    {
        if( primitive.flags & RG_MESH_PRIMITIVE_TRANSLUCENT )
        {
            return true;
        }

        if( Utils::UnpackAlphaFromPacked32( primitive.color ) <
            MESH_TRANSLUCENT_ALPHA_THRESHOLD )
        {
            return true;
        }
    }

    return false;
}

namespace detail
{
#if !SUPPRESS_TEXLAYERS
inline uint32_t GetMaterialBlendFlags( const RgTextureLayerBlendType* blend, uint32_t layerIndex )
{
    if( blend == nullptr )
    {
        return 0;
    }
    assert( layerIndex < GEOM_INST_FLAG_BLENDING_LAYER_COUNT );

    uint32_t bitOffset = MATERIAL_BLENDING_TYPE_BIT_COUNT * layerIndex;

    switch( *blend )
    {
        case RG_TEXTURE_LAYER_BLEND_TYPE_OPAQUE: return MATERIAL_BLENDING_TYPE_OPAQUE << bitOffset;
        case RG_TEXTURE_LAYER_BLEND_TYPE_ALPHA: return MATERIAL_BLENDING_TYPE_ALPHA << bitOffset;
        case RG_TEXTURE_LAYER_BLEND_TYPE_ADD: return MATERIAL_BLENDING_TYPE_ADD << bitOffset;
        case RG_TEXTURE_LAYER_BLEND_TYPE_SHADE: return MATERIAL_BLENDING_TYPE_SHADE << bitOffset;
        default: assert( 0 ); return 0;
    }
}

inline uint32_t GetMaterialBlendFlags( const RgMeshPrimitiveTextureLayersEXT& info, uint32_t layerIndex )
{
    assert( layerIndex <= 3 );

    switch( layerIndex )
    {
        case 0: return RG_TEXTURE_LAYER_BLEND_TYPE_OPAQUE;
        case 1: return GetMaterialBlendFlags( info.pLayer1 ? &info.pLayer1->blend : nullptr, 1 );
        case 2: return GetMaterialBlendFlags( info.pLayer2 ? &info.pLayer1->blend : nullptr, 2 );
        case 3: return GetMaterialBlendFlags( info.pLayer3 ? &info.pLayer1->blend : nullptr, 3 );
        default: assert( 0 ); return 0;
    }
}
#endif
}

inline bool LayerExists( const RgMeshPrimitiveInfo& info, uint32_t layerIndex )
{
#if !SUPPRESS_TEXLAYERS
    if( auto layers = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &info ) )
    {
        switch( layerIndex )
        {
            case 1: return layers->pLayer1 && layers->pLayer1->pTexCoord;
            case 2: return layers->pLayer2 && layers->pLayer2->pTexCoord;
            case 3: return layers->pLayer3 && layers->pLayer3->pTexCoord;
            default: assert( 0 ); return false;
        }
    }
#endif

    return false;
}

// ShGeometryInstance::flags of a primitive.
// Shared by the Vulkan (GeomInfoManager) and Metal backends.
inline uint32_t GetPrimitiveFlags( const RgMeshInfo*          mesh,
                                   const RgMeshPrimitiveInfo& info,
                                   bool                       isDynamicVertexData )
{
    uint32_t f = 0;


#if !SUPPRESS_TEXLAYERS
    if( auto layers = pnext::find< RgMeshPrimitiveTextureLayersEXT >( &info ) )
    {
        f |= LayerExists( info, 1 ) ? GEOM_INST_FLAG_EXISTS_LAYER1 : 0;
        f |= LayerExists( info, 2 ) ? GEOM_INST_FLAG_EXISTS_LAYER2 : 0;
        f |= LayerExists( info, 3 ) ? GEOM_INST_FLAG_EXISTS_LAYER3 : 0;

        f |= detail::GetMaterialBlendFlags( *layers, 0 );
        f |= detail::GetMaterialBlendFlags( *layers, 1 );
        f |= detail::GetMaterialBlendFlags( *layers, 2 );
        f |= detail::GetMaterialBlendFlags( *layers, 3 );
    }
#endif

    if( ( info.flags & RG_MESH_PRIMITIVE_MIRROR ) ||
        ( mesh && ( mesh->flags & RG_MESH_FORCE_MIRROR ) ) )
    {
        f |= GEOM_INST_FLAG_REFLECT;
    }

    if( ( info.flags & RG_MESH_PRIMITIVE_WATER ) ||
        ( mesh && ( mesh->flags & RG_MESH_FORCE_WATER ) ) )
    {
        f |= GEOM_INST_FLAG_MEDIA_TYPE_WATER;
        f |= GEOM_INST_FLAG_REFLECT;
        f |= GEOM_INST_FLAG_REFRACT;

        // Doom64-RT: which of the four liquids this is. Carried as two bits
        // rather than a per-primitive colour because the geometry instance has
        // no spare colour channel, and four palettes is all Doom 64 needs.
        // Neither bit set == index 0 == water, so an upstream caller that only
        // sets RG_MESH_PRIMITIVE_WATER lands on the old behaviour.
        f |= ( info.flags & RG_MESH_PRIMITIVE_LIQUID_BIT0 ) ? GEOM_INST_FLAG_LIQUID_BIT0 : 0;
        f |= ( info.flags & RG_MESH_PRIMITIVE_LIQUID_BIT1 ) ? GEOM_INST_FLAG_LIQUID_BIT1 : 0;

    }

    if( info.flags & RG_MESH_PRIMITIVE_NO_WATER_CAUSTICS )
    {
        f |= GEOM_INST_FLAG_NO_WATER_CAUSTICS;
    }

    if( info.flags & RG_MESH_PRIMITIVE_LAVA )
    {
        f |= GEOM_INST_FLAG_LAVA;
    }

    if( info.flags & RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED )
    {
        f |= GEOM_INST_FLAG_EMIS_SCREEN_SCALED;
    }

    if( info.flags & RG_MESH_PRIMITIVE_ACID )
    {
        f |= GEOM_INST_FLAG_MEDIA_TYPE_ACID;
        f |= GEOM_INST_FLAG_REFLECT;
        f |= GEOM_INST_FLAG_REFRACT;
    }

    if( ( info.flags & RG_MESH_PRIMITIVE_GLASS ) ||
        ( mesh && ( mesh->flags & RG_MESH_FORCE_GLASS ) ) )
    {
        f |= GEOM_INST_FLAG_MEDIA_TYPE_GLASS;
        f |= GEOM_INST_FLAG_REFLECT;
        f |= GEOM_INST_FLAG_REFRACT;
    }

    if( info.flags & RG_MESH_PRIMITIVE_SPRITE )
    {
        f |= GEOM_INST_FLAG_SPRITE;
    }

    if( info.flags & RG_MESH_PRIMITIVE_GLASS_IF_SMOOTH )
    {
        f |= GEOM_INST_FLAG_GLASS_IF_SMOOTH;
    }

    if( info.flags & RG_MESH_PRIMITIVE_MIRROR_IF_SMOOTH )
    {
        f |= GEOM_INST_FLAG_MIRROR_IF_SMOOTH;
    }

    // the spare bit, see GEOM_INST_FLAG_POLISHED in Media.h
    if( info.flags & RG_MESH_PRIMITIVE_POLISHED )
    {
        f |= GEOM_INST_FLAG_RESERVED_6;
    }

    if( ( info.flags & RG_MESH_PRIMITIVE_IGNORE_REFRACT_AFTER ) ||
        ( mesh && ( mesh->flags & RG_MESH_FORCE_IGNORE_REFRACT_AFTER ) ) )
    {
        f |= GEOM_INST_FLAG_IGNORE_REFRACT_AFTER;
    }

    if( !( info.flags & RG_MESH_PRIMITIVE_DONT_GENERATE_NORMALS ) )
    {
        f |= GEOM_INST_FLAG_GENERATE_NORMALS;
    }

    if( info.flags & RG_MESH_PRIMITIVE_FORCE_EXACT_NORMALS )
    {
        f |= GEOM_INST_FLAG_EXACT_NORMALS;
    }

    if( info.flags & RG_MESH_PRIMITIVE_THIN_MEDIA )
    {
        f |= GEOM_INST_FLAG_THIN_MEDIA;
    }

    if( isDynamicVertexData )
    {
        f |= GEOM_INST_FLAG_IS_DYNAMIC;
    }

    return f;
}

}
