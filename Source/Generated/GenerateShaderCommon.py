# Copyright (c) 2020-2021 Sultim Tsyrendashiev
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This script generates two separate header files for C and GLSL but with identical data

import sys
import os
import re
from math import log2


TYPE_FLOAT32    = 0
TYPE_INT32      = 1
TYPE_UINT32     = 2


C_TYPE_NAMES = {
    TYPE_FLOAT32:       "float",
    TYPE_INT32:         "int32_t",
    TYPE_UINT32:        "uint32_t",
}

GLSL_TYPE_NAMES = {
    TYPE_FLOAT32:       "float",
    TYPE_INT32:         "int",
    TYPE_UINT32:        "uint",
    (TYPE_FLOAT32,  2): "vec2",
    (TYPE_FLOAT32,  3): "vec3",
    (TYPE_FLOAT32,  4): "vec4",
    (TYPE_INT32,    2): "ivec2",
    (TYPE_INT32,    3): "ivec3",
    (TYPE_INT32,    4): "ivec4",
    (TYPE_UINT32,   2): "uvec2",
    (TYPE_UINT32,   3): "uvec3",
    (TYPE_UINT32,   4): "uvec4",
    (TYPE_FLOAT32, 22): "mat2",
    (TYPE_FLOAT32, 23): "mat2x3",
    (TYPE_FLOAT32, 32): "mat3x2",
    (TYPE_FLOAT32, 33): "mat3",
    (TYPE_FLOAT32, 34): "mat3x4",
    (TYPE_FLOAT32, 43): "mat4x3",
    (TYPE_FLOAT32, 44): "mat4",
}


TYPE_ACTUAL_SIZES = {
    TYPE_FLOAT32:   4,
    TYPE_INT32:     4,
    TYPE_UINT32:    4,
    (TYPE_FLOAT32,  2): 8,
    (TYPE_FLOAT32,  3): 12,
    (TYPE_FLOAT32,  4): 16,
    (TYPE_INT32,    2): 8,
    (TYPE_INT32,    3): 12,
    (TYPE_INT32,    4): 16,
    (TYPE_UINT32,   2): 8,
    (TYPE_UINT32,   3): 12,
    (TYPE_UINT32,   4): 16,
    (TYPE_FLOAT32, 22): 16,
    (TYPE_FLOAT32, 23): 24,
    (TYPE_FLOAT32, 32): 24,
    (TYPE_FLOAT32, 33): 36,
    (TYPE_FLOAT32, 34): 48,
    (TYPE_FLOAT32, 43): 48,
    (TYPE_FLOAT32, 44): 64,
}

GLSL_TYPE_SIZES_STD_430 = {
    TYPE_FLOAT32:   4,
    TYPE_INT32:     4,
    TYPE_UINT32:    4,
    (TYPE_FLOAT32,  2): 8,
    (TYPE_FLOAT32,  3): 16,
    (TYPE_FLOAT32,  4): 16,
    (TYPE_INT32,    2): 8,
    (TYPE_INT32,    3): 16,
    (TYPE_INT32,    4): 16,
    (TYPE_UINT32,   2): 8,
    (TYPE_UINT32,   3): 16,
    (TYPE_UINT32,   4): 16,
    (TYPE_FLOAT32, 22): 16,
    (TYPE_FLOAT32, 23): 24,
    (TYPE_FLOAT32, 32): 24,
    (TYPE_FLOAT32, 33): 36,
    (TYPE_FLOAT32, 34): 48,
    (TYPE_FLOAT32, 43): 48,
    (TYPE_FLOAT32, 44): 64,
}


# These types are only for image format use!
TYPE_UNORM8     = 3
TYPE_UINT8      = 4
TYPE_UINT16     = 5
TYPE_FLOAT16    = 6
TYPE_PACK_11    = 128   # R11G11B10
TYPE_PACK_E5    = 129   # shared exponent, E5B9G9R9

COMPONENT_R     = 0
COMPONENT_RG    = 1
COMPONENT_RGB   = 2
COMPONENT_RGBA  = 3


VULKAN_IMAGE_FORMATS = {
    (TYPE_UNORM8,   COMPONENT_R):       "VK_FORMAT_R8_UNORM",
    (TYPE_UNORM8,   COMPONENT_RG):      "VK_FORMAT_R8G8_UNORM",
    (TYPE_UNORM8,   COMPONENT_RGBA):    "VK_FORMAT_R8G8B8A8_UNORM",

    (TYPE_UINT8,    COMPONENT_R):       "VK_FORMAT_R8_UINT",
    (TYPE_UINT8,    COMPONENT_RG):      "VK_FORMAT_R8G8_UINT",
    (TYPE_UINT8,    COMPONENT_RGBA):    "VK_FORMAT_R8G8B8A8_UINT",

    (TYPE_UINT16,   COMPONENT_R):       "VK_FORMAT_R16_UINT",
    (TYPE_UINT16,   COMPONENT_RG):      "VK_FORMAT_R16G16_UINT",
    (TYPE_UINT16,   COMPONENT_RGBA):    "VK_FORMAT_R16G16B16A16_UINT",

    (TYPE_UINT32,   COMPONENT_R):       "VK_FORMAT_R32_UINT",
    (TYPE_UINT32,   COMPONENT_RG):      "VK_FORMAT_R32G32_UINT",
    (TYPE_UINT32,   COMPONENT_RGBA):    "VK_FORMAT_R32G32B32A32_UINT",

    (TYPE_FLOAT16,  COMPONENT_R):       "VK_FORMAT_R16_SFLOAT",
    (TYPE_FLOAT16,  COMPONENT_RG):      "VK_FORMAT_R16G16_SFLOAT",
    (TYPE_FLOAT16,  COMPONENT_RGBA):    "VK_FORMAT_R16G16B16A16_SFLOAT",

    (TYPE_FLOAT32,  COMPONENT_R):       "VK_FORMAT_R32_SFLOAT",
    (TYPE_FLOAT32,  COMPONENT_RG):      "VK_FORMAT_R32G32_SFLOAT",
    (TYPE_FLOAT32,  COMPONENT_RGBA):    "VK_FORMAT_R32G32B32A32_SFLOAT",

    (TYPE_PACK_11,  COMPONENT_RGB):    "VK_FORMAT_B10G11R11_UFLOAT_PACK32",
    # Should've been VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 
    # but not enough devices support storage images with such format.
    # So pack/unpack is done manually.
    (TYPE_PACK_E5,  COMPONENT_RGB):    "VK_FORMAT_R32_UINT",
}

GLSL_IMAGE_FORMATS = {
    (TYPE_UNORM8,   COMPONENT_R):       "r8",
    (TYPE_UNORM8,   COMPONENT_RG):      "rg8",
    (TYPE_UNORM8,   COMPONENT_RGBA):    "rgba8",

    (TYPE_UINT8,    COMPONENT_R):       "r8ui",
    (TYPE_UINT8,    COMPONENT_RG):      "rg8ui",
    (TYPE_UINT8,    COMPONENT_RGBA):    "rgba8ui",

    (TYPE_UINT16,   COMPONENT_R):       "r16ui",
    (TYPE_UINT16,   COMPONENT_RG):      "rg16ui",
    (TYPE_UINT16,   COMPONENT_RGBA):    "rgba16ui",

    (TYPE_UINT32,   COMPONENT_R):       "r32ui",
    (TYPE_UINT32,   COMPONENT_RG):      "rg32ui",
    (TYPE_UINT32,   COMPONENT_RGBA):    "rgba32ui",

    (TYPE_FLOAT16,  COMPONENT_R):       "r16f",
    (TYPE_FLOAT16,  COMPONENT_RG):      "rg16f",
    (TYPE_FLOAT16,  COMPONENT_RGBA):    "rgba16f",

    (TYPE_FLOAT32,  COMPONENT_R):       "r32f",
    (TYPE_FLOAT32,  COMPONENT_RG):      "rg32f",
    (TYPE_FLOAT32,  COMPONENT_RGBA):    "rgba32f",

    (TYPE_PACK_11,  COMPONENT_RGB):    "r11f_g11f_b10f",
    (TYPE_PACK_E5,  COMPONENT_RGB):    "r32ui",
}

GLSL_IMAGE_2D_TYPE = { 
    TYPE_FLOAT32    : "image2D",
    TYPE_INT32      : "iimage2D",
    TYPE_UINT32     : "uimage2D",
    TYPE_UNORM8     : "image2D",
    TYPE_UINT8      : "uimage2D",
    TYPE_UINT16     : "uimage2D",
    TYPE_FLOAT16    : "image2D",
    TYPE_PACK_11    : "image2D",
    TYPE_PACK_E5    : "uimage2D",
}

GLSL_SAMPLER_2D_TYPE = { 
    TYPE_FLOAT32    : "sampler2D",
    TYPE_INT32      : "isampler2D",
    TYPE_UINT32     : "usampler2D",
    TYPE_UNORM8     : "sampler2D",
    TYPE_UINT8      : "usampler2D",
    TYPE_UINT16     : "usampler2D",
    TYPE_FLOAT16    : "sampler2D",
    TYPE_PACK_11    : "sampler2D",
    TYPE_PACK_E5    : "usampler2D",
}


USE_BASE_STRUCT_NAME_IN_VARIABLE_STRIDE = False
USE_MULTIDIMENSIONAL_ARRAYS_IN_C = False
CONST_TO_EVALUATE = "CONST VALUE MUST BE EVALUATED"







# --------------------------------------------------------------------------------------------- #
# User defined constants
# --------------------------------------------------------------------------------------------- #

GRADIENT_ESTIMATION_ENABLED = True
FRAMEBUF_IGNORE_ATTACHMENTS_DEFINE = "FRAMEBUF_IGNORE_ATTACHMENTS" # define this, to not specify framebufs that are used as attachments
def BIT( i ):
    return "1 << " + str( i )

CONST = {
    "MAX_INSTANCE_COUNT"                    : 1 << 16,
    "MAX_GEOM_INFO_COUNT"                   : 1 << 17,
    
    "BINDING_VERTEX_BUFFER_STATIC"              : 0,
    "BINDING_VERTEX_BUFFER_DYNAMIC"             : 1,
    "BINDING_INDEX_BUFFER_STATIC"               : 2,
    "BINDING_INDEX_BUFFER_DYNAMIC"              : 3,
    "BINDING_GEOMETRY_INSTANCES"                : 4,
    "BINDING_GEOMETRY_INSTANCES_MATCH_PREV"     : 5,
    "BINDING_PREV_POSITIONS_BUFFER_DYNAMIC"     : 6,
    "BINDING_PREV_INDEX_BUFFER_DYNAMIC"         : 7,
    "BINDING_STATIC_TEXCOORD_LAYER_1"           : 8,
    "BINDING_STATIC_TEXCOORD_LAYER_2"           : 9,
    "BINDING_STATIC_TEXCOORD_LAYER_3"           : 10,
    "BINDING_DYNAMIC_TEXCOORD_LAYER_1"          : 11,
    "BINDING_DYNAMIC_TEXCOORD_LAYER_2"          : 12,
    "BINDING_DYNAMIC_TEXCOORD_LAYER_3"          : 13,
    "BINDING_GLOBAL_UNIFORM"                    : 0,
    "BINDING_ACCELERATION_STRUCTURE_MAIN"       : 0,
    "BINDING_TEXTURES"                          : 0,
    "BINDING_CUBEMAPS"                          : 0,
    "BINDING_RENDER_CUBEMAP"                    : 0,
    "BINDING_BLUE_NOISE"                        : 0,
    "BINDING_LUM_HISTOGRAM"                     : 0,
    "BINDING_LIGHT_SOURCES"                     : 0,
    "BINDING_LIGHT_SOURCES_PREV"                : 1,
    "BINDING_LIGHT_SOURCES_INDEX_PREV_TO_CUR"   : 2,
    "BINDING_LIGHT_SOURCES_INDEX_CUR_TO_PREV"   : 3,
    "BINDING_INITIAL_LIGHTS_GRID"               : 4,
    "BINDING_INITIAL_LIGHTS_GRID_PREV"          : 5,
    "BINDING_LENS_FLARES_CULLING_INPUT"         : 0,
    "BINDING_LENS_FLARES_DRAW_CMDS"             : 1,
    "BINDING_DRAW_LENS_FLARES_INSTANCES"        : 0,
    "BINDING_PORTAL_INSTANCES"                  : 0,
    "BINDING_LPM_PARAMS"                        : 0,
    "BINDING_RESTIR_INDIRECT_RESERVOIRS"        : 2,
    "BINDING_RESTIR_INDIRECT_RESERVOIRS_PREV"   : 1,
    "BINDING_VOLUMETRIC_STORAGE"                : 0,
    "BINDING_VOLUMETRIC_SAMPLER"                : 1,
    "BINDING_VOLUMETRIC_SAMPLER_PREV"           : 2,
    # Doom64-RT: uncommented alongside ILLUMINATION_VOLUME=1 above. Volumetric.cpp's
    # #if ILLUMINATION_VOLUME descriptor-set blocks (image + sampler for the
    # illumination volume) reference these two names; with the define on and these
    # commented out, the shader/C++ compile fails on "undeclared identifier" -- this
    # dict is a single source shared by GLSL and C++ headers, so both need it live
    # together, not just the top-level ILLUMINATION_VOLUME flag (2026-08-08).
    "BINDING_VOLUMETRIC_ILLUMINATION"           : 3,
    "BINDING_VOLUMETRIC_ILLUMINATION_SAMPLER"   : 4,
    # Doom64-RT: the volumetric CLOUD MAP -- a lat-long image of world
    # directions, marched by CmCloudMap.comp and sampled by the sky fragment
    # shaders. Lives in the volumetric set so the sky pipelines, which already
    # carry that set in their layout, need no new descriptor plumbing.
    "BINDING_VOLUMETRIC_CLOUDMAP_STORAGE"       : 5,
    "BINDING_VOLUMETRIC_CLOUDMAP_SAMPLER"       : 6,
    "BINDING_VOLUMETRIC_CLOUDMAP_SAMPLER_PREV"  : 7,
    "BINDING_FLUID_PARTICLES_ARRAY"             : 0,
    "BINDING_FLUID_GENERATE_ID_TO_SOURCE"       : 1,
    "BINDING_FLUID_SOURCES"                     : 2,

    "INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON"           : BIT( 0 ),
    "INSTANCE_CUSTOM_INDEX_FLAG_FIRST_PERSON_VIEWER"    : BIT( 1 ),
    "INSTANCE_CUSTOM_INDEX_FLAG_SKY"                    : BIT( 2 ),
    # Doom64-RT: this surface must IGNORE shadow-only geometry when it shadow
    # tests (INSTANCE_MASK_RESERVED_0). Set on every alpha-tested instance,
    # which is what sprites are.
    #
    # Why it has to exist: a sprite's shadow proxies are planes through the
    # actor's own axis, so any proxy that is not edge-on to the light shadows
    # the half of its OWN billboard that lies behind it. With the flashlight --
    # a light at the camera, i.e. along the sprite's normal -- the perpendicular
    # proxy is edge-on to it and projects to a line straight down the middle of
    # the sprite it belongs to. That is a black stripe on every enemy in the
    # beam, and no amount of plane count or spacing removes it: it is what
    # centred proxy geometry does.
    "INSTANCE_CUSTOM_INDEX_FLAG_IGNORE_SHADOW_PROXY"    : BIT( 3 ),

    "INSTANCE_MASK_WORLD_0"                 : BIT( 0 ),
    "INSTANCE_MASK_WORLD_1"                 : BIT( 1 ),
    "INSTANCE_MASK_WORLD_2"                 : BIT( 2 ),
    "INSTANCE_MASK_RESERVED_0"              : BIT( 3 ),
    "INSTANCE_MASK_RESERVED_1"              : BIT( 4 ),
    "INSTANCE_MASK_REFRACT"                 : BIT( 5 ),
    "INSTANCE_MASK_FIRST_PERSON"            : BIT( 6 ),
    "INSTANCE_MASK_FIRST_PERSON_VIEWER"     : BIT( 7 ),
    
    "PAYLOAD_INDEX_DEFAULT"                 : 0,
    "PAYLOAD_INDEX_SHADOW"                  : 1,
    
    "SBT_INDEX_RAYGEN_PRIMARY"              : 0,
    "SBT_INDEX_RAYGEN_REFL_REFR"            : 1,
    "SBT_INDEX_RAYGEN_DIRECT"               : 2,
    "SBT_INDEX_RAYGEN_INDIRECT_INIT"        : 3,
#   "SBT_INDEX_RAYGEN_INDIRECT_FINAL"       :  ,
    "SBT_INDEX_RAYGEN_GRADIENTS"            : 4,
    "SBT_INDEX_RAYGEN_INITIAL_RESERVOIRS"   : 5,
    "SBT_INDEX_RAYGEN_VOLUMETRIC"           : 6,
    "SBT_INDEX_MISS_DEFAULT"                : 0,
    "SBT_INDEX_MISS_SHADOW"                 : 1,
    "SBT_INDEX_HITGROUP_FULLY_OPAQUE"       : 0,
    "SBT_INDEX_HITGROUP_ALPHA_TESTED"       : 1,
    # Doom64-RT: MEDIA hit groups, reached by adding SBT_RAY_OFFSET_MEDIA to the
    # ray's sbtRecordOffset. Same closest-hit as the pair above; the alpha-tested
    # group's any-hit additionally discards GEOM_INST_FLAG_SPRITE geometry, so a
    # volumetric shadow ray ignores billboards while grates and fences still cut
    # their shafts. Instances keep offsets 0/1; the RAY chooses the table half.
    "SBT_INDEX_HITGROUP_MEDIA_FULLY_OPAQUE" : 2,
    "SBT_INDEX_HITGROUP_MEDIA_ALPHA_TESTED" : 3,
    "SBT_RAY_OFFSET_MEDIA"                  : 2,
    
    "MATERIAL_NO_TEXTURE"                   : 0,

    "MATERIAL_BLENDING_TYPE_OPAQUE"         : 0,
    "MATERIAL_BLENDING_TYPE_ALPHA"          : 1,
    "MATERIAL_BLENDING_TYPE_ADD"            : 2,
    "MATERIAL_BLENDING_TYPE_SHADE"          : 3,
    "MATERIAL_BLENDING_TYPE_BIT_COUNT"      : 2,
    "MATERIAL_BLENDING_TYPE_BIT_MASK"       : CONST_TO_EVALUATE,

    "GEOM_INST_FLAG_BLENDING_LAYER_COUNT"   : 4,         
    # first 8 bits (MATERIAL_BLENDING_TYPE_BIT_COUNT * GEOM_INST_FLAG_BLENDING_LAYER_COUNT)
    # are for the blending flags per each layer, others can be used
    "GEOM_INST_FLAG_NO_WATER_CAUSTICS"      : BIT( 8 ),
    # Doom64-RT: 2-bit liquid index (water / nukage / sludge / blood), read by
    # the stylized water surface to pick its body and crest colour. 0 = water,
    # so anything that only sets GEOM_INST_FLAG_MEDIA_TYPE_WATER is unchanged.
    "GEOM_INST_FLAG_LIQUID_BIT0"            : BIT( 9 ),
    "GEOM_INST_FLAG_LIQUID_BIT1"            : BIT( 10 ),
    # Doom64-RT: lava surface. See RG_MESH_PRIMITIVE_LAVA.
    "GEOM_INST_FLAG_LAVA"                   : BIT( 11 ),
    # Doom64-RT: scale on-screen emission by emissiveMult, as the indirect path
    # already does. See RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED.
    "GEOM_INST_FLAG_EMIS_SCREEN_SCALED"     : BIT( 12 ),
    "GEOM_INST_FLAG_GLASS_IF_SMOOTH"        : BIT( 13 ),
    "GEOM_INST_FLAG_MIRROR_IF_SMOOTH"       : BIT( 14 ),
    "GEOM_INST_FLAG_EXISTS_LAYER1"          : BIT( 15 ),
    "GEOM_INST_FLAG_EXISTS_LAYER2"          : BIT( 16 ),
    "GEOM_INST_FLAG_EXISTS_LAYER3"          : BIT( 17 ),
    "GEOM_INST_FLAG_MEDIA_TYPE_ACID"        : BIT( 18 ),
    "GEOM_INST_FLAG_EXACT_NORMALS"          : BIT( 19 ),
    "GEOM_INST_FLAG_IGNORE_REFRACT_AFTER"   : BIT( 20 ),
    "GEOM_INST_FLAG_SPRITE"                 : BIT( 21 ),
    "GEOM_INST_FLAG_RESERVED_6"             : BIT( 22 ),
    "GEOM_INST_FLAG_THIN_MEDIA"             : BIT( 23 ),
    "GEOM_INST_FLAG_REFRACT"                : BIT( 24 ),
    "GEOM_INST_FLAG_REFLECT"                : BIT( 25 ),
    "GEOM_INST_FLAG_PORTAL"                 : BIT( 26 ),
    "GEOM_INST_FLAG_MEDIA_TYPE_WATER"       : BIT( 27 ),
    "GEOM_INST_FLAG_MEDIA_TYPE_GLASS"       : BIT( 28 ),
    "GEOM_INST_FLAG_GENERATE_NORMALS"       : BIT( 29 ),
    "GEOM_INST_FLAG_INVERTED_NORMALS"       : BIT( 30 ),
    "GEOM_INST_FLAG_IS_DYNAMIC"             : BIT( 31 ),

    "SKY_TYPE_COLOR"                        : 0,
    "SKY_TYPE_CUBEMAP"                      : 1,
    "SKY_TYPE_RASTERIZED_GEOMETRY"          : 2,

    # to reduce shader opearations
    "SUPPRESS_TEXLAYERS"                    : 1,
    
    "BLUE_NOISE_TEXTURE_COUNT"              : 128,
    "BLUE_NOISE_TEXTURE_SIZE"               : 128,
    "BLUE_NOISE_TEXTURE_SIZE_POW"           : CONST_TO_EVALUATE,

    "COMPUTE_COMPOSE_GROUP_SIZE_X"          : 16,
    "COMPUTE_COMPOSE_GROUP_SIZE_Y"          : 16,

    "COMPUTE_DECAL_APPLY_GROUP_SIZE_X"      : 16,
    
    "COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_X"   : 16,
    "COMPUTE_BLOOM_UPSAMPLE_GROUP_SIZE_Y"   : 16,
    "COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_X" : 16,
    "COMPUTE_BLOOM_DOWNSAMPLE_GROUP_SIZE_Y" : 16,
    "COMPUTE_BLOOM_APPLY_GROUP_SIZE_X"      : 16,
    "COMPUTE_BLOOM_APPLY_GROUP_SIZE_Y"      : 16,
    "COMPUTE_BLOOM_STEP_COUNT"              : 7,

    "COMPUTE_EFFECT_GROUP_SIZE_X"           : 16,
    "COMPUTE_EFFECT_GROUP_SIZE_Y"           : 16,

    "COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_X"    : 16,
    "COMPUTE_LUM_HISTOGRAM_GROUP_SIZE_Y"    : 16,
    "COMPUTE_LUM_HISTOGRAM_BIN_COUNT"       : 256,

    "COMPUTE_VERT_PREPROC_GROUP_SIZE_X"     : 256,
    "VERT_PREPROC_MODE_ONLY_DYNAMIC"        : 0,
    "VERT_PREPROC_MODE_DYNAMIC_AND_MOVABLE" : 1,
    "VERT_PREPROC_MODE_ALL"                 : 2,

    "GRADIENT_ESTIMATION_ENABLED"           : int(GRADIENT_ESTIMATION_ENABLED),
    "COMPUTE_GRADIENT_ATROUS_GROUP_SIZE_X"  : 16,
    "COMPUTE_ANTIFIREFLY_GROUP_SIZE_X"      : 16,
    "COMPUTE_SVGF_TEMPORAL_GROUP_SIZE_X"    : 16,
    "COMPUTE_SVGF_VARIANCE_GROUP_SIZE_X"    : 16,
    "COMPUTE_SVGF_ATROUS_GROUP_SIZE_X"      : 16,
    "COMPUTE_SVGF_ATROUS_ITERATION_COUNT"   : 4,

    "COMPUTE_ASVGF_STRATA_SIZE"                         : 3,
    "COMPUTE_ASVGF_GRADIENT_ATROUS_ITERATION_COUNT"     : 4,  

    "COMPUTE_INDIRECT_DRAW_FLARES_GROUP_SIZE_X"         : 256,
    "LENS_FLARES_MAX_DRAW_CMD_COUNT"                    : 512,

    "COMPUTE_FLUID_PARTICLES_GROUP_SIZE_X"              : 256,
    "COMPUTE_FLUID_PARTICLES_GENERATE_GROUP_SIZE_X"     : 256,

    "DEBUG_SHOW_FLAG_MOTION_VECTORS"        : BIT( 0 ),
    "DEBUG_SHOW_FLAG_GRADIENTS"             : BIT( 1 ),
    "DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE"    : BIT( 2 ),
    "DEBUG_SHOW_FLAG_UNFILTERED_SPECULAR"   : BIT( 3 ),
    "DEBUG_SHOW_FLAG_UNFILTERED_INDIRECT"   : BIT( 4 ),
    "DEBUG_SHOW_FLAG_ONLY_DIRECT_DIFFUSE"   : BIT( 5 ),
    "DEBUG_SHOW_FLAG_ONLY_SPECULAR"         : BIT( 6 ),
    "DEBUG_SHOW_FLAG_ONLY_INDIRECT_DIFFUSE" : BIT( 7 ),
    "DEBUG_SHOW_FLAG_LIGHT_GRID"            : BIT( 8 ),
    "DEBUG_SHOW_FLAG_ALBEDO_WHITE"          : BIT( 9 ),
    "DEBUG_SHOW_FLAG_NORMALS"               : BIT( 10 ),
    "DEBUG_SHOW_FLAG_BLOOM"                 : BIT( 11 ),
    
    "MAX_RAY_LENGTH"                        : "10000.0",

    "MEDIA_TYPE_VACUUM"                     : 0,
    "MEDIA_TYPE_WATER"                      : 1,
    "MEDIA_TYPE_GLASS"                      : 2,
    "MEDIA_TYPE_ACID"                       : 3,
    "MEDIA_TYPE_COUNT"                      : 4,

    "GEOM_INST_NO_TRIANGLE_INFO"            : "UINT32_MAX",

    "LIGHT_TYPE_NONE"                       : 0,
    "LIGHT_TYPE_DIRECTIONAL"                : 1,
    "LIGHT_TYPE_SPHERE"                     : 2,
    "LIGHT_TYPE_SPOT"                       : 3,
    "LIGHT_TYPE_TRIANGLE"                   : 4,

    "TRIANGLE_LIGHTS"                       : 0,

    "LIGHT_ARRAY_DIRECTIONAL_LIGHT_OFFSET"  : 0,
    "LIGHT_ARRAY_REGULAR_LIGHTS_OFFSET"     : 1,

    "LIGHT_INDEX_NONE"                      : ((1 << 15) - 1),

    "LIGHT_GRID_ENABLED"                    : 0, # no effect on enabling?
#   "LIGHT_GRID_SIZE_X"                     : 16,
#   "LIGHT_GRID_SIZE_Y"                     : 16,
#   "LIGHT_GRID_SIZE_Z"                     : 16,
#   "LIGHT_GRID_CELL_SIZE"                  : 128,
#   "COMPUTE_LIGHT_GRID_GROUP_SIZE_X"       : 256,

    "PORTAL_INDEX_NONE"                     : 63,
    "PORTAL_MAX_COUNT"                      : 63,

    "PACKED_INDIRECT_RESERVOIR_SIZE_IN_WORDS" : 5,

    "VOLUMETRIC_SIZE_X"                     : 160,
    "VOLUMETRIC_SIZE_Y"                     : 88,
    "VOLUMETRIC_SIZE_Z"                     : 64,
    "COMPUTE_VOLUMETRIC_GROUP_SIZE_X"       : 16,
    "COMPUTE_VOLUMETRIC_GROUP_SIZE_Y"       : 16,
    "COMPUTE_SCATTER_ACCUM_GROUP_SIZE_X"    : 16,

    # Doom64-RT: the cloud map. Lat-long, u = azimuth over 360 degrees, v =
    # altitude over the UPPER hemisphere only (clouds are never below the
    # horizon), so 1024x256 is 2.8 texels per degree of azimuth and the same
    # of altitude. The march cost is this many texels, whatever the screen is.
    "CLOUDMAP_WIDTH"                        : 1024,
    "CLOUDMAP_HEIGHT"                       : 256,
    "COMPUTE_CLOUDMAP_GROUP_SIZE_X"         : 16,
    "COMPUTE_CLOUDMAP_GROUP_SIZE_Y"         : 16,

    # Doom64-RT: capacity of the localised-smoke puff list. The puffs ride in
    # the global uniform rather than a storage buffer, so this is a hard limit
    # and must match RG_MAX_SMOKE_PUFFS in Include/RTGL1/RTGL1.h.
    #
    # 32 -> 128 when ambient sources arrived. Muzzle smoke alone fits in 32; a
    # room full of torches does not, and the pool's overflow rule is oldest-out,
    # so an undersized buffer does not degrade -- it deletes whichever smoke is
    # oldest, which is the player's.
    #
    # THE COST IS 48 BYTES A PUFF (three vec4 arrays), so this takes the whole
    # ShGlobalUniform from 3552 to 8160 bytes. That is still under the 16384 the
    # Vulkan spec GUARANTEES for maxUniformBufferRange, which is the number that
    # matters -- a bigger cap would need the storage-buffer rewrite instead.
    "SMOKE_PUFF_MAX"                        : 128,

    # Doom64-RT: capacity of the light-shaft list, and must match
    # RG_MAX_SHAFT_LIGHTS in Include/RTGL1/RTGL1.h. Same storage argument as the
    # puffs -- these ride in the global uniform -- but far cheaper: one uint32
    # each, packed four to a uvec4, so 32 lights cost 128 bytes.
    #
    # The number is a BUDGET, not a capacity problem. Cost here is per FROXEL,
    # and the shader stops after volumeShaftMaxTraced shadow rays, so this only
    # bounds how many candidates the per-cell radiance cull gets to choose from.
    # A MULTIPLE OF FOUR, because the shader unpacks by [ i >> 2 ][ i & 3 ].
    "VOLUME_SHAFT_LIGHT_MAX"                : 64,

    "VOLUME_ENABLE_NONE"                    : 0,
    "VOLUME_ENABLE_SIMPLE"                  : 1,
    "VOLUME_ENABLE_VOLUMETRIC"              : 2,

    "HDR_DISPLAY_NONE"                      : 0,
    "HDR_DISPLAY_LINEAR"                    : 1,
    "HDR_DISPLAY_ST2084"                    : 2,

    # Doom64-RT: turned on so RsWorld.inl's rasterized-primitive branch can sample the
    # illumination volume (globalUniform.illumVolumeEnable) instead of the
    # max(1, avgLuminance) fallback. That branch is what lets a see-through sprite
    # (spectre, nightmare imp) darken with the room while the sprite stays translucent
    # and its _e emissive stays untouched -- ldrEmis is computed from baseColor() AFTER
    # this multiply, so the eyes never see the dimming. Was 0 upstream (2026-08-08).
    "ILLUMINATION_VOLUME"                   : 1,

    "COMPUTE_INDIRECT_FINAL_GROUP_SIZE_X"   : 16,
    "COMPUTE_INDIRECT_FINAL_GROUP_SIZE_Y"   : 16,
}

CONST_GLSL_ONLY = {
    "SURFACE_POSITION_INCORRECT"            : 10000000.0,  
}


def align(c, alignment):
    return ((c + alignment - 1) // alignment) * alignment


def align4(a):
    return ((a + 3) >> 2) << 2


def evalConst():
    # flags for each layer
    assert CONST[ "MATERIAL_BLENDING_TYPE_BIT_COUNT" ] * CONST[ "GEOM_INST_FLAG_BLENDING_LAYER_COUNT" ] <= 8
    CONST[ "MATERIAL_BLENDING_TYPE_BIT_MASK" ]          = ( 1 << int( CONST[ "MATERIAL_BLENDING_TYPE_BIT_COUNT" ] ) ) - 1

    CONST[ "BLUE_NOISE_TEXTURE_SIZE_POW" ]              = int( log2( CONST[ "BLUE_NOISE_TEXTURE_SIZE" ] ) )

    assert len( [ None for _, v in CONST.items() if v == CONST_TO_EVALUATE ] ) == 0, "All CONST_TO_EVALUATE values must be calculated"


# --------------------------------------------------------------------------------------------- #
# User defined structs
# --------------------------------------------------------------------------------------------- #
# Each member is defined as a tuple (base type, dimensions, name, count).
# Dimensions:   1 - scalar, 
#               2 - gvec2,
#               3 - gvec3, 
#               4 - gvec4, 
#               [xy] - gmat[xy] (i.e. 32 - gmat32)
# If count > 1 and dimensions is 2, 3 or 4 (matrices are not supported)
# then it'll be represented as an array with size (count*dimensions).

VERTEX_STRUCT = [
    (TYPE_FLOAT32,      3,     "position",              1),
    (TYPE_UINT32 ,      1,     "normalPacked",          1),
    (TYPE_FLOAT32,      2,     "texCoord",              1),
    (TYPE_UINT32,       1,     "color",                 1),
    (TYPE_UINT32,       1,     "_pad0",                 1),
]

VERTEX_COMPACT_STRUCT = [
    (TYPE_FLOAT32,      3,     "position",              1),
    (TYPE_UINT32 ,      1,     "normalPacked",          1),
]

# Must be careful with std140 offsets! They are set manually.
# Other structs are using std430 and padding is done automatically.
GLOBAL_UNIFORM_STRUCT = [
    (TYPE_FLOAT32,     44,      "view",                         1),
    (TYPE_FLOAT32,     44,      "invView",                      1),
    (TYPE_FLOAT32,     44,      "viewPrev",                     1),
    (TYPE_FLOAT32,     44,      "projection",                   1),
    (TYPE_FLOAT32,     44,      "invProjection",                1),
    (TYPE_FLOAT32,     44,      "projectionPrev",               1),

    (TYPE_FLOAT32,     44,      "volumeViewProj",               1),
    (TYPE_FLOAT32,     44,      "volumeViewProjInv",            1),
    (TYPE_FLOAT32,     44,      "volumeViewProj_Prev",          1),
    (TYPE_FLOAT32,     44,      "volumeViewProjInv_Prev",       1),

    (TYPE_FLOAT32,      1,      "cellWorldSize",                1),
    (TYPE_FLOAT32,      1,      "lightmapScreenCoverage",       1),
    (TYPE_UINT32,       1,      "illumVolumeEnable",            1),
    (TYPE_FLOAT32,      1,      "renderWidth",                  1),

    (TYPE_FLOAT32,      1,      "renderHeight",                 1),
    (TYPE_UINT32,       1,      "frameId",                      1),
    (TYPE_FLOAT32,      1,      "timeDelta",                    1),
    (TYPE_FLOAT32,      1,      "minLogLuminance",              1),

    (TYPE_FLOAT32,      1,      "maxLogLuminance",              1),
    (TYPE_FLOAT32,      1,      "luminanceWhitePoint",          1),
    (TYPE_UINT32,       1,      "stopEyeAdaptation",            1),
    (TYPE_UINT32,       1,      "directionalLightExists",       1),
    
    (TYPE_FLOAT32,      1,      "polyLightSpotlightFactor",     1),
    (TYPE_UINT32,       1,      "skyType",                      1),
    (TYPE_FLOAT32,      1,      "skyColorMultiplier",           1),
    (TYPE_UINT32,       1,      "skyCubemapIndex",              1),
    
    (TYPE_FLOAT32,      4,      "skyColorDefault",              1),

    (TYPE_FLOAT32,      4,      "cameraPosition",               1),
    (TYPE_FLOAT32,      4,      "cameraPositionPrev",           1),

    (TYPE_UINT32,       1,      "debugShowFlags",               1),
    # Doom64-RT: GI path depth -- vertices after the primary hit, [1..4].
    # Was indirSecondBounce, which nothing read: the shader gate meant to
    # consume it sat commented out in RtRaygenIndirect.inl ("diffuse very
    # red"), so the second bounce always ran. Same slot, same type, so std140
    # is untouched. rt_gi_bounces.
    (TYPE_UINT32,       1,      "indirectBounces",              1),
    (TYPE_UINT32,       1,      "lightCount",                   1),
    (TYPE_UINT32,       1,      "lightCountPrev",               1),

    (TYPE_FLOAT32,      1,      "emissionMapBoost",             1),
    (TYPE_FLOAT32,      1,      "emissionMaxScreenColor",       1),
    (TYPE_FLOAT32,      1,      "normalMapStrength",            1),
    (TYPE_FLOAT32,      1,      "skyColorSaturation",           1),

    (TYPE_UINT32,       1,      "maxBounceShadowsLights",           1),
    (TYPE_FLOAT32,      1,      "rayLength",                        1),
    (TYPE_UINT32,       1,      "rayCullBackFaces",                 1),
    (TYPE_UINT32,       1,      "rayCullMaskWorld",                 1),

    (TYPE_FLOAT32,      1,      "bloomIntensity",                   1),
    (TYPE_FLOAT32,      1,      "bloomThreshold",                   1),
    (TYPE_FLOAT32,      1,      "bloomEV",                          1),
    (TYPE_UINT32,       1,      "reflectRefractMaxDepth",           1),

    (TYPE_UINT32,       1,      "cameraMediaType",                  1),
    (TYPE_FLOAT32,      1,      "indexOfRefractionWater",           1),
    (TYPE_FLOAT32,      1,      "indexOfRefractionGlass",           1),
    (TYPE_FLOAT32,      1,      "waterTextureDerivativesMultiplier",1),

    (TYPE_UINT32,       1,      "volumeEnableType",                 1),
    (TYPE_FLOAT32,      1,      "volumeScattering",                 1),
    (TYPE_FLOAT32,      1,      "lensDirtIntensity",                1),
    (TYPE_UINT32,       1,      "waterNormalTextureIndex",          1),

    (TYPE_FLOAT32,      1,      "thinMediaWidth",                   1),
    (TYPE_FLOAT32,      1,      "time",                             1),
    (TYPE_FLOAT32,      1,      "waterWaveSpeed",                   1),
    (TYPE_FLOAT32,      1,      "waterWaveStrength",                1),

    (TYPE_FLOAT32,      4,      "waterColorAndDensity",             1),
    (TYPE_FLOAT32,      4,      "acidColorAndDensity",              1),

    (TYPE_FLOAT32,      1,      "cameraRayConeSpreadAngle",         1),
    (TYPE_FLOAT32,      1,      "waterTextureAreaScale",            1),
    (TYPE_UINT32,       1,      "dirtMaskTextureIndex",             1),
    (TYPE_FLOAT32,      1,      "upscaledRenderWidth",              1),

    (TYPE_FLOAT32,      4,      "worldUpVector",                    1),

    (TYPE_FLOAT32,      1,      "upscaledRenderHeight",             1),
    (TYPE_FLOAT32,      1,      "jitterX",                          1),
    (TYPE_FLOAT32,      1,      "jitterY",                          1),
    (TYPE_FLOAT32,      1,      "primaryRayMinDist",                1),

    (TYPE_UINT32,       1,      "rayCullMaskWorld_Shadow",          1),
    (TYPE_UINT32,       1,      "volumeAllowTintUnderwater",        1),
    # When set, ComposeNoisy reads A-SVGF temporal outputs instead of raw unfiltered
    (TYPE_UINT32,       1,      "rrTemporalPrefilterEnabled",       1),
    (TYPE_UINT32,       1,      "twirlPortalNormal",                1),

    (TYPE_UINT32,       1,      "lightIndexIgnoreFPVShadows",       1),
    (TYPE_FLOAT32,      1,      "gradientMultDiffuse",              1),
    (TYPE_FLOAT32,      1,      "gradientMultIndirect",             1),
    (TYPE_FLOAT32,      1,      "gradientMultSpecular",             1),

    (TYPE_FLOAT32,      1,      "minRoughness",                     1),
    # Doom64-RT metalness fail-safes. Four scalars on purpose: the C and GLSL
    # halves of this struct pack by different rules, so a scalar run that is not
    # a multiple of four silently moves every vec4 after it in GLSL only.
    (TYPE_FLOAT32,      1,      "metallicMax",                      1),
    (TYPE_FLOAT32,      1,      "metallicRoughCut",                 1),
    (TYPE_FLOAT32,      1,      "metallicRoughBand",                1),
    # Doom64-RT: the SPRITE set. A billboard is one quad with one normal, so its
    # indirect specular reflection vector is identical for every texel -- the
    # whole sprite samples the room in one direction and takes the same colour
    # everywhere. That is a failure walls cannot have, so it needs its own dials.
    # Exactly FOUR scalars: the run has to stay a multiple of 4 or the C and GLSL
    # packings diverge (tools/check_uniform_layout.py guards this).
    (TYPE_FLOAT32,      1,      "spritePbr",                        1),
    (TYPE_FLOAT32,      1,      "spriteMetallicMax",                1),
    (TYPE_FLOAT32,      1,      "spriteRoughMin",                   1),
    (TYPE_FLOAT32,      1,      "spriteNormalStrength",             1),
    # Doom64-RT: the same mix for WALLS AND FLATS. Taken from the metallicPad0
    # slot -- a float where a float was -- so std140 is untouched and the scalar
    # run keeps its length.
    (TYPE_FLOAT32,      1,      "worldPbr",                         1),
    (TYPE_FLOAT32,      1,      "volumeCameraNear",                 1),
    (TYPE_FLOAT32,      1,      "volumeCameraFar",                  1),
    (TYPE_UINT32,       1,      "antiFireflyEnabled",               1),

    (TYPE_FLOAT32,      4,      "volumeAmbient",                    1),
    (TYPE_FLOAT32,      4,      "volumeUnderwaterColor",            1),
    (TYPE_FLOAT32,      4,      "volumeFallbackSrcColor",           1),
    (TYPE_FLOAT32,      4,      "volumeFallbackSrcDirection",       1),

    (TYPE_FLOAT32,      1,      "volumeAsymmetry",                  1),
    (TYPE_UINT32,       1,      "volumeLightSourceIndex",           1),
    (TYPE_FLOAT32,      1,      "volumeFallbackSrcExists",          1),
    (TYPE_FLOAT32,      1,      "volumeLightMult",                  1),

    (TYPE_UINT32,       1,      "hdrDisplay",                       1),
    (TYPE_FLOAT32,      1,      "parallaxMaxDepth",                 1),
    (TYPE_UINT32,       1,      "fluidEnabled",                     1),
    # bit0=N, bit1=emis, bit2=metallic, bit3=height, bit4=roughness (Dev Materials A/B)
    (TYPE_UINT32,       1,      "materialStripFlags",               1),
    # Dev: mix authored roughness toward 1.0 (0=authored, 1=fully matte). Live A/B for RR shimmer.
    (TYPE_FLOAT32,      1,      "materialRoughnessTowardMatte",     1),
    # DLSS-RR disocclusion mask (transient-light history discard). Keep these 3
    # together with materialRoughnessTowardMatte so vec4 members stay 16-aligned.
    (TYPE_UINT32,       1,      "rrDisoccEnable",                   1),
    (TYPE_FLOAT32,      1,      "rrDisoccRatio",                    1),
    (TYPE_FLOAT32,      1,      "rrDisoccMinDelta",                 1),

    (TYPE_UINT32,       1,      "rrDisoccShowMask",                 1),
    # DLSS-RR firefly clamp, applied to the noisy lighting in CmNoisyCompose
    # before it reaches RR. RTGL's RR path has no spatial or temporal prefilter
    # at all (A-SVGF gets anti-firefly + variance-driven atrous), so single-
    # sample spikes go to NGX raw. Remix runs an equivalent clamp before RR.
    # 0 = off. Taken from the _pad slots so std140 layout is unchanged.
    (TYPE_FLOAT32,      1,      "rrFireflyThreshold",               1),
    (TYPE_FLOAT32,      1,      "rrFireflyMinLum",                  1),
    # ReSTIR reuse taps sampled with tiled blue noise instead of hash white
    # noise (the old "TODO: need low discrepancy noise" in selectLight_Direct).
    (TYPE_UINT32,       1,      "restirBlueNoise",                  1),

    (TYPE_FLOAT32,      4,      "fluidColor",                       1),

    # --- Samples per pixel + ReSTIR quality (Doom64-RT) -----------------------
    # The path tracer is 1 spp; convergence comes from temporal accumulation
    # (ReSTIR M + denoiser history), both of which motion legitimately destroys.
    # These trade GPU time for a quieter RAW signal, which helps A-SVGF and
    # DLSS-RR equally because it is upstream of both. All default to stock.
    # Two full vec4 groups so std140 alignment of what follows is unchanged.
    (TYPE_UINT32,       1,      "directSamples",                    1),
    (TYPE_UINT32,       1,      "indirectSamples",                  1),
    (TYPE_UINT32,       1,      "restirInitialSamples",             1),
    (TYPE_UINT32,       1,      "restirSpatialSamples",             1),

    (TYPE_FLOAT32,      1,      "restirSpatialRadius",              1),
    (TYPE_UINT32,       1,      "restirTemporalMCap",               1),
    # DLSS-RR albedo-guide floor. RR demodulates colour by these guides, so a
    # guide approaching zero makes that division explode -- correlated dark
    # filaments in dim, dark-albedo areas. The sky branch of CmNoisyCompose
    # already floors its guides ("bounded and non-zero", RR guide 3.4.2); the
    # world branch never did. 0 = no floor (old behaviour).
    (TYPE_FLOAT32,      1,      "rrGuideMin",                       1),
    # What the DLSS-RR albedo guides actually contain. RR does not merely divide
    # colour by these and multiply back -- it also uses them as edge-detection
    # and reprojection weights, so they must be SMOOTH material properties.
    #   0 = raw material: albedo / F0. What shipped before 2026-08-05, i.e. the
    #       iterations where the worm artifact was reportedly absent.
    #   1 = full demodulation: ro_d * throughput * ambient (2026-08-06 change).
    #       Correct in the "divide then remodulate" sense, but folds two
    #       non-smooth terms in: throughput is fetched in CHECKERBOARD space, so
    #       adjacent output pixels read non-adjacent texels; and
    #       getMaterialAmbient() is a thresholded quadratic that reaches exactly
    #       0 at black albedo, putting a derivative kink along dark-texture
    #       contours -- which no guide FLOOR can remove.
    #   2 = material demodulation only: ro_d / envBRDFApprox2, no throughput,
    #       no ambient. Separates "which factors" from "which reflectivity model".
    (TYPE_UINT32,       1,      "rrGuideMode",                      1),

    # Indirect ReSTIR temporal reuse is gated on framebufDISGradientHistory
    # (the A-SVGF antilag gradient): a tap is rejected when alpha > 0.25.
    # That buffer is written ONLY by CmASVGFGradientAtrous, which runs only
    # inside Denoiser::Denoise() -- and DLSS-RR skips Denoise() entirely in
    # favour of ComposeNoisy(). So under RR the gate reads a buffer nothing
    # updates. Either it is a dead no-op, or it rejects GI temporal reuse every
    # frame, which would leave indirect lighting effectively 1-spp with no
    # accumulation: surfaces fizzle, and only the denoiser's own history hides
    # it -- exactly what motion takes away.
    # 1 = apply the gate (stock), 0 = ignore it.
    (TYPE_UINT32,       1,      "restirIndirAntilag",               1),
    # Debug: write the shadow-ray visibility term into the unfiltered direct
    # buffer instead of radiance. Visibility is the ONLY thing a shadow ray
    # produces, so this separates "the occluder never blocks the ray" from
    # "the shadow is cast but drowned in fill light or smeared by the
    # denoiser" -- two causes that look identical in the final image, and that
    # four inconclusive A/B ladders failed to tell apart.
    # 1 = greyscale visibility (black = shadowed), 2 = red tint on shadowed
    # pixels over normal shading, so an umbra can be located in context.
    # Taken from a _pad slot so the std140 layout is unchanged: the group
    # ending at rrSpecHitDist must stay a whole vec4 or viewProjCubemap below
    # loses its 16-byte alignment and the C and GLSL layouts diverge.
    (TYPE_UINT32,       1,      "debugVisibility",                  1),
    # DLSS-RR pre-exposure reorder (both taken from _pad slots, layout unchanged).
    # 1 = CmPrepareFinal SKIPS the EV100 multiply and the screen-emissive add, so
    # RR denoises linear pre-exposure radiance (NVIDIA's RR guide S3.7 declares
    # exposure unsupported for RR -- the input is supposed to be pre-exposure),
    # and CmRrPostExposure reapplies both on the UPSCALED output. Host-gated to
    # frames where the RR branch actually runs: if this were 1 on a DLSS2/FSR
    # fallback frame, nothing downstream would apply exposure at all.
    (TYPE_UINT32,       1,      "rrPreExposure",                    1),
    # Debug: CmRrPostExposure tints its output magenta. The absurd arm -- "no
    # visible difference" and "the pass never ran" are otherwise the same image.
    (TYPE_UINT32,       1,      "rrPreExpDebug",                    1),

    # Shadow rays per pixel for DIRECT illumination. 1 = stock. The direct
    # estimate multiplies by a single binary traceVisibility(), so that 0/1 term
    # dominates the 1-spp variance and is untouched by anything ReSTIR does
    # (measured: blue-noise reuse taps changed nothing). Averaging N points on
    # the chosen light turns it into a soft fraction; variance falls ~1/sqrt(N).
    # Kept as a full vec4 group so viewProjCubemap below stays 16-byte aligned.
    (TYPE_UINT32,       1,      "shadowSamples",                    1),
    # Debug: write ReSTIR reservoir M (accumulated sample count) into the
    # unfiltered direct buffer instead of radiance. M is what makes 1-spp
    # ReSTIR converge; if it collapses under camera motion, that is the noise.
    (TYPE_UINT32,       1,      "debugRestirM",                     1),
    # ReSTIR temporal tap jitter radius, in pixels. Stock is 2. The jitter buys
    # decorrelation but costs history exactly where it is hardest to keep: on
    # grazing surfaces a 2px offset changes depth by far more than the flat 10%
    # reuse threshold, so the tap is rejected and M collapses. 0 = reproject
    # exactly (what RTXDI does).
    (TYPE_FLOAT32,      1,      "restirTemporalJitter",             1),
    # DLSS-RR: feed pInSpecularHitDistance (see the SpecularHitDistance framebuf).
    (TYPE_UINT32,       1,      "rrSpecHitDist",                    1),

    # --- Stylized water (Doom64-RT) ------------------------------------------
    # Doom 64 water flats (D64W2_01 / D64W1_01) are opaque FLOOR flats: the
    # physical refract+absorb path has nothing to refract into and reads far too
    # real for the art. See getStylizedWaterAlbedo() in RaygenPrimary.inl.
    # Two full scalar groups + one vec4 so viewProjCubemap stays 16-aligned.
    # 0 = off, stock physical water.
    (TYPE_FLOAT32,      1,      "stylizedWaterStrength",            1),
    # how hard the wave crests brighten the texture's own caustic veins
    (TYPE_FLOAT32,      1,      "stylizedWaterCaustic",             1),
    # Fresnel clamp: 1.0 = a true mirror at grazing angles
    (TYPE_FLOAT32,      1,      "stylizedWaterReflMax",             1),
    (TYPE_FLOAT32,      1,      "stylizedWaterRoughness",           1),

    # unlit on-screen sheen on the veins (casts no light)
    (TYPE_FLOAT32,      1,      "stylizedWaterGlow",                1),
    # luminance of the flat's brightest texel, used to normalize the vein mask
    (TYPE_FLOAT32,      1,      "stylizedWaterVeinRef",             1),
    # Diagnostic. 1 = paint every surface the shader sees as water:
    # MAGENTA if the stylized branch is taken, GREEN if RTGL flagged it water
    # but the stylized gate rejected it. No colour at all => the primitive never
    # got RG_MESH_PRIMITIVE_WATER, i.e. the JSON meta never reached it.
    (TYPE_FLOAT32,      1,      "stylizedWaterDebug",               1),
    # Reflection strength at NORMAL incidence. Real water is F0=0.02, i.e.
    # looking straight down the reflection is 2% and effectively invisible.
    # This is the artistic floor that makes it read as reflective from above.
    (TYPE_FLOAT32,      1,      "stylizedWaterReflMin",             1),

    # Body and crest colour per LIQUID, indexed by the 2-bit liquid id in the
    # geometry instance flags (GEOM_INST_FLAG_LIQUID_BIT*):
    #   0 water   1 nukage   2 sludge   3 blood
    # Doom 64's four liquids are the same animated 64-frame flat design in four
    # palettes, so they share this whole shader and differ only here. [0] is
    # water, which is the index a primitive with neither bit set gets.
    (TYPE_FLOAT32,      4,      "stylizedLiquidTint",               4),
    (TYPE_FLOAT32,      4,      "stylizedLiquidCrest",              4),

    # ONE vec4 each, holding a SCALAR per liquid -- not four vec4s like the two
    # colours above. Indexed the same way: stylizedLiquidRelief[liquidId].
    #
    # relief: how much of the authored _n survives against the animated water
    # wave. getNormal() overwrites the normal-mapped normal with the wave for
    # any water surface, so without this a liquid can never show an _n at all.
    # 0 keeps the old behaviour exactly, which is what water/nukage/sludge use.
    #
    # flow: depth of the detail-texture advection along the veins -- a flow
    # map. The vein DIRECTION is baked into the height map's .g/.b as a vector
    # (a raw angle would tear at a junction under bilinear filtering; a vector
    # merely shrinks toward zero and fades). HitInfo.inl advects a detail
    # texture along it with a two-phase ping-pong and hands the result across
    # in framebufAlbedo.a.
    (TYPE_FLOAT32,      4,      "stylizedLiquidRelief",             1),
    (TYPE_FLOAT32,      4,      "stylizedLiquidFlow",               1),

    # refl: scales the remapped-Schlick reflection F, per liquid. 1 is the
    # stylizedWaterRefl{Min,Max} curve untouched, which is what water gets and
    # what every liquid used to get. Mud is not a mirror: the same mirror
    # reflection that sells water is the single loudest thing saying "this is
    # water with brown paint on it".
    #
    # rough: surface roughness override. <= 0 means "use stylizedWaterRoughness"
    # (0.1, a near-mirror), so a liquid that does not set it is unchanged.
    # Note this is the roughness the DENOISER and the G-buffer see -- the
    # reflection RAY is a pure mirror off shadeNormal either way, so what
    # actually scatters a rough liquid's reflection is its authored relief
    # normal. The two are meant to be used together.
    (TYPE_FLOAT32,      4,      "stylizedLiquidRefl",               1),
    (TYPE_FLOAT32,      4,      "stylizedLiquidRough",              1),
    # Per-liquid scale on the caustics that liquid PROJECTS onto the geometry
    # around it. A caustic is light refracted through a fluid and focused on
    # what lies beyond it, so an opaque one makes none -- blood was throwing
    # swimming-pool light on its own walls. The probe is receiver-side, so
    # probeWaterBelow has to hand the liquid id back out for this.
    (TYPE_FLOAT32,      4,      "stylizedLiquidCaustics",           1),
    # The detail is sampled in a vein-aligned frame (u along the channel and
    # scrolling, v across it): speed = detail tiles scrolled per second, scale =
    # detail tiles per liquid tile, aspect = across-vein frequency multiplier
    # that stretches the noise into lengthwise streaks.
    (TYPE_FLOAT32,      1,      "liquidFlowSpeed",                  1),
    (TYPE_FLOAT32,      1,      "liquidFlowScale",                  1),
    (TYPE_FLOAT32,      1,      "liquidFlowAspect",                 1),
    # 1 = paint the advected detail. Flat blue means it never crossed
    # framebufAlbedo.a, which is indistinguishable from "too subtle" by eye and
    # cost nothing to make separable.
    (TYPE_FLOAT32,      1,      "liquidFlowDebug",                  1),

    # --- Lava (Doom64-RT) ----------------------------------------------------
    # The lava's emission cannot be baked bright enough to bloom: _e is 8-bit so
    # it caps at 1.0, screen emission is _e * emissionMaxScreenColor (3), and
    # rt_bloom_threshold is 16. lavaEmisBoost is applied to lava surfaces only,
    # which is the whole reason the geometry carries a LAVA flag.
    (TYPE_FLOAT32,      1,      "lavaEmisBoost",                    1),
    # Heat that moves. A low-frequency field drifting across the surface,
    # QUANTIZED to lavaFlowPixel world units so it stays as chunky as the flat
    # it sits on -- a smooth gradient over pixel art reads as a modern shader
    # bolted onto the wrong texture.
    (TYPE_FLOAT32,      1,      "lavaFlowStrength",                 1),
    (TYPE_FLOAT32,      1,      "lavaFlowSpeed",                    1),
    (TYPE_FLOAT32,      1,      "lavaFlowScale",                    1),
    (TYPE_FLOAT32,      1,      "lavaFlowPixel",                    1),
    # Whole-surface breathing, on top of the drift.
    (TYPE_FLOAT32,      1,      "lavaPulse",                        1),
    (TYPE_FLOAT32,      1,      "lavaPulseSpeed",                   1),
    # Indirect (GI) multiplier for lava emission, separate from the screen one.
    # This is the knob that lets the LAVA ITSELF light the room, as an area
    # source, instead of the grid of analytic point lights -- which is what a
    # lake actually is, and which does not leave circles of illumination on the
    # walls as the player walks past each one.
    (TYPE_FLOAT32,      1,      "lavaGiBoost",                      1),
    # Debug: 1 = paint every surface the shader sees as lava magenta. Answers
    # "does the LAVA flag survive into the shader" on its own, which no amount
    # of staring at brightness can.
    (TYPE_FLOAT32,      1,      "lavaDebug",                        1),
    # THE PAD COUNT IS LOAD-BEARING. This block is scalars in a std140 struct, so
    # it must be a multiple of FOUR floats or every field after it -- including
    # the mat4s at the end -- reads shifted, and the frame comes out black with
    # only the HUD on top. Eleven floats here did exactly that. Count them.
    # Taken from the _padlava0 slot so std140 is unchanged. > 0.5: NO stylized
    # liquid splits, whatever its refl says -- every liquid shaded on every
    # pixel at full resolution, no mirror ray, sheen from its roughness. The
    # Options > Quality "Liquid surfaces" item; see d64_noSplit in
    # RaygenPrimary.inl for why the split is unstable on an authored normal.
    (TYPE_FLOAT32,      1,      "liquidNoSplit",                    1),
    (TYPE_FLOAT32,      1,      "_padlava1",                        1),
    (TYPE_FLOAT32,      1,      "_padlava",                         1),
    # Hue of the heat, multiplying the lava's emission on BOTH the screen and
    # the GI path. The flat's own cracks photograph yellow-orange once they are
    # boosted; pulling green and blue down is what makes it read as molten rock
    # rather than as a light bulb. A vec4, so it does not disturb the scalar
    # count above -- which is load-bearing, see the note there.
    (TYPE_FLOAT32,      4,      "lavaTint",                         1),

    # --- Directional light: sky-reach test (Doom64-RT) ------------------------
    # A shadow ray that hits NOTHING is scored as lit (RtMissShadowCheck.rmiss
    # sets isShadowed = 0). Doom maps are not watertight and have no geometry
    # above a ceiling, so rays leave the world through T-junctions at wall/
    # ceiling seams and come back "lit" -- the moon then washes rooms that have
    # no opening at all. Widening the light or gating apertures cannot fix that,
    # because nothing is being squeezed through anything.
    #
    # The test: in a Doom map the sky is the ONLY legitimate way out, so a ray
    # that escapes without hitting sky geometry escaped through a modelling gap.
    # 1 = require the ray to reach sky (WORLD_2) before counting as lit.
    (TYPE_FLOAT32,      1,      "sunRequireSky",                    1),
    # Diagnostic. 1 = ignore the light's colour and paint by WHY a surface is
    # lit: RED  = the ray reached sky, i.e. genuine moonlight through a real
    # opening; GREEN = the ray escaped into the void, i.e. a leak. Surfaces the
    # moon does not reach are untouched. Works whether or not sunRequireSky is
    # on, so the leak can be seen before it is fixed.
    (TYPE_FLOAT32,      1,      "sunLeakDebug",                     1),
    # Brightness of those debug colours. These rooms are near-black, so the
    # classification is invisible at the moon's real intensity.
    (TYPE_FLOAT32,      1,      "sunLeakDebugMul",                  1),
    # Max length of the extra sky probe ray, in meters. Only traced when the
    # normal shadow ray already missed, i.e. only for pixels that are currently
    # lit, so this is bounded by how much of the screen the moon touches.
    (TYPE_FLOAT32,      1,      "sunSkyProbeMaxDist",               1),

    # --- Projected water caustics (Doom64-RT) --------------------------------
    # Caustics cast BY the water ONTO the geometry around it. A path tracer at
    # 1 spp will never find these by itself (they are a focused specular-to-
    # diffuse path), so they are projected: each shading point fires one probe
    # ray straight down, and if it lands on a water-flagged surface within
    # waterCausticDist, its lighting is modulated by an animated caustic field
    # sampled from the same water normal texture the surface waves use.
    # Multiplicative, so a dark room stays dark. 0 = off, no probe ray traced.
    (TYPE_FLOAT32,      1,      "waterCausticGain",                 1),
    # caustic field frequency, in UV per world unit
    (TYPE_FLOAT32,      1,      "waterCausticScale",                1),
    (TYPE_FLOAT32,      1,      "waterCausticSpeed",                1),
    # how far below a surface the water may be and still light it (world units)
    (TYPE_FLOAT32,      1,      "waterCausticDist",                 1),

    # How far ABOVE the water the caustics still reach, metres. Separate from
    # waterCausticDist on purpose: real caustics climb only a little way up a
    # wall, while the probe itself has to reach much further sideways to clear
    # a pool's ledge. One combined range made the pattern run up the full
    # height of every wall.
    (TYPE_FLOAT32,      1,      "waterCausticRise",                 1),
    # how far the probe tilts along the surface normal (0 = straight down)
    (TYPE_FLOAT32,      1,      "waterCausticSlant",                1),
    # Extra gain for VERTICAL receivers. Caustics on a wall are seen at a
    # grazing angle and are physically fainter than the same pattern on a floor,
    # so a single gain that reads well on the pool bottom leaves the walls
    # barely visible -- and raising it brightens everything instead.
    (TYPE_FLOAT32,      1,      "waterCausticWallBoost",            1),
    # --- Illuminated fog (Doom64-RT) -----------------------------------------
    # The froxel pass scatters ONE light: whatever TryGetVolumetricLight picked,
    # which on a map with no sun and no RG_LIGHT_ADDITIONAL_VOLUMETRIC light is
    # nothing at all -- so its fog is flat ambient with no source in it. 1 makes
    # each froxel run the full direct-lighting estimate instead, so every light
    # in the map scatters. Taken from the _padc1 slot, so std140 is unchanged.
    (TYPE_UINT32,       1,      "volumeAllLights",                  1),

    # Scattering albedo of the medium. Multiplies the whole in-scattered term --
    # ambient AND lit -- so a coloured fog colours what glows inside it too.
    # Extinction stays monochrome (the transmittance channel is a single float
    # all the way to CmPrepareFinal), so distance fades TOWARD this colour
    # rather than filtering by it. { 1, 1, 1 } is the no-op.
    #
    # NEAR value of a near->far ramp: the froxel grid's slices are uniform in
    # DISTANCE (VOLUMETRIC_DISTANCE_POW 1), so cell.z is a straight depth
    # fraction and near/far can be separated for free. Setting the _Far pair
    # equal to these is the uniform medium.
    (TYPE_FLOAT32,      4,      "volumeMediaColor",                 1),
    (TYPE_FLOAT32,      4,      "volumeMediaColorFar",              1),

    # Density and tint at the FAR plane of the volume, and the shape of the
    # interpolation between them (1 = linear, >1 holds the near value longer and
    # thickens late, <1 thickens immediately).
    (TYPE_FLOAT32,      1,      "volumeScatteringFar",              1),
    (TYPE_FLOAT32,      1,      "volumeDensityCurve",               1),
    # Fog only: in-scattering is faded out within this many metres OF A LIGHT,
    # so a light carried at the camera (flashlight, muzzle flash) does not white
    # out the froxels in front of it by inverse square. Keyed off the LIGHT's
    # distance, not the camera's, so the beam further down the corridor -- the
    # look this is all for -- survives. 0 = physical behaviour.
    (TYPE_FLOAT32,      1,      "volumeLightNearFade",              1),
    # Doom64-RT: 1 = reproduce the stock bounce>=2 weighting, which multiplied
    # by 1/pdf (= pi/cos) and nothing else -- pi/cos too much for a cosine-
    # sampled Lambertian, mean ~2pi. 0 = the correct throughput of exactly 1.
    # Taken from the _padf2 slot: a uint is the same 4 bytes, so std140 is
    # unchanged and check_uniform_layout.py stays silent. rt_gi_bounce_legacy.
    (TYPE_UINT32,       1,      "indirectLegacyWeight",             1),

    # --- Localised smoke (Doom64-RT) -----------------------------------------
    # A separate group, appended AFTER the whole volume block rather than
    # interleaved with it, so a mistake here cannot silently shift
    # volumeMediaColor's offset and retune the shipped fog.
    #
    # One full scalar group of four, then two vec4 arrays, so viewProjCubemap
    # below keeps its 16-byte alignment.
    #
    # smokeCount 0 is the no-smoke state: the loop in Smoke.h does not execute
    # and the medium arithmetic collapses back to the fog's exactly.
    (TYPE_UINT32,       1,      "smokeCount",                       1),
    # Near-light fade, and all-lights temporal blend, used ONLY in froxels that
    # contain smoke -- chosen per cell, so fog cells keep volumeLightNearFade
    # and the stock 0.05. A muzzle flash lighting the smoke at the barrel is the
    # whole effect, and the fog's 2 m fade plus a 0.05 blend would erase it and
    # then smear what was left over ~0.7 s.
    (TYPE_FLOAT32,      1,      "smokeLightNearFade",               1),
    (TYPE_FLOAT32,      1,      "smokeIllumBlend",                  1),
    # Whether a froxel CONTAINING smoke runs the all-lights estimate. Separate
    # from volumeAllLights on purpose: that one switches the WHOLE volume off the
    # single-light path, and the single-light path is the only place the sun's
    # sky-probe test lives -- so flipping it per frame deletes a map's light
    # shafts for as long as a puff exists. This is read per cell instead.
    (TYPE_UINT32,       1,      "smokeAllLights",                   1),

    # Shader-side probe (rt_smoke_debug 2/3). 2 paints the froxels a puff
    # actually covers; 3 paints EVERY froxel whenever the puff list is non-empty.
    # The pair separates "the shader cannot read the uniform" from "it reads it
    # and no cell passes the sphere test", which no amount of engine-side logging
    # can distinguish.
    (TYPE_UINT32,       1,      "smokeDebug",                       1),
    # Metres beyond which a light stops lighting SMOKE. Only smoke cells consult
    # it, so fog and the global medium keep receiving every light as before.
    (TYPE_FLOAT32,      1,      "smokeLightFarFade",                1),
    # Spatial blur applied to the froxel volume before it is integrated, 0..1 as
    # a blend against the raw grid. The volume is lit at ONE sample per cell and
    # is the only buffer in the renderer with no spatial filter of any kind, so
    # its variance reaches the screen untouched. Volumetric lighting is
    # low-frequency by nature -- blurring it costs almost nothing visually.
    (TYPE_FLOAT32,      1,      "volumeSpatialBlur",                1),
    # ONE pad, not two. The scalar run from smokeCount to here must be a
    # MULTIPLE OF FOUR: C packs scalars contiguously, std140 aligns the vec4
    # arrays that follow to 16 bytes, and any other count makes the two layouts
    # disagree from that point on. Adding volumeSpatialBlur without removing a
    # pad took the run to nine, shifted every array by 12 bytes in GLSL only,
    # and produced smoke that vanished and black bands on screen.
    # Ceiling on the in-scattered radiance inside SMOKE. A carried light -- the
    # flashlight, the plasma rifle's glow -- sits at ~0 m, so it lights the
    # froxels around it by inverse square and a dense puff in front of the camera
    # goes pure white. That is the same physics rt_fog_light_near exists for, and
    # smoke deliberately disables that fade so a muzzle flash can light its own
    # puff. This is the backstop: it bounds the result instead of the cause.
    (TYPE_FLOAT32,      1,      "smokeMaxLight",                    1),

    # Samples per froxel INSIDE SMOKE. The volume is estimated at one NEE sample
    # and one shadow ray per cell, and that is the source of the variance every
    # filter here is trying to hide. Raising it globally would be unaffordable --
    # 900k cells -- but a puff occupies a few per cent of them, so paying for
    # more samples only where smoke is costs almost nothing and attacks the noise
    # where it is made rather than smoothing it afterwards.
    #
    # A full group of four: the scalar run before the vec4 arrays must stay
    # divisible by four or C and std140 disagree. See tools/check_uniform_layout.py.
    (TYPE_UINT32,       1,      "smokeSpp",                         1),
    # How far, in FROXELS, the per-pixel volume sample is jittered. Stock is 2,
    # which hides the grid in a low-contrast medium -- but with dense smoke the
    # jitter reaches across geometry silhouettes into columns belonging to other
    # surfaces, and that draws dark outlines around everything seen through the
    # smoke.
    (TYPE_FLOAT32,      1,      "volumeDither",                     1),
    # Whether SCREEN EMISSION is attenuated by the volumetric medium. It was
    # added after the volumetric composite and outside its guard, so an emissive
    # panel shone through fog and smoke at full strength as though nothing were
    # in front of it.
    (TYPE_UINT32,       1,      "volumeOccludeEmis",                1),

    # PIXEL-ART STYLIZATION, smoke only (Doom64-RT). The froxel volume produces
    # a smooth exponential falloff, which is physically right and reads as an
    # airbrushed blob against art that is entirely hard-edged pixels. These
    # posterize the puff's falloff into bands and optionally snap its evaluation
    # to a world grid, so smoke gains the stepped edge the sprites have.
    #
    # It lives INSIDE smoke_evalAt rather than as a screen-space filter for the
    # reason every other smoke change does: with smokeCount 0 that function
    # returns zero whatever these say, so the fog's arithmetic still collapses
    # bit for bit and ab.cmd smoke-fogsafe still holds.
    #
    # 0 = the smooth falloff, 1 = fully banded.
    (TYPE_FLOAT32,      1,      "smokeStylize",                     1),
    # How many bands. Few is chunky and poster-like, many approaches smooth.
    (TYPE_UINT32,       1,      "smokeStylizeSteps",                1),
    # Metres of world-space voxel snapping, 0 = off. WORLD space, not screen:
    # screen-space blocks crawl as the camera turns, which reads as noise rather
    # than as style. World voxels stay put in the level.
    (TYPE_FLOAT32,      1,      "smokeStylizeGrid",                 1),

    # TWO pads now, not one, and the count is arithmetic rather than taste. The
    # scalar run from smokeCount to here must stay a MULTIPLE OF FOUR, so what
    # is ADDED has to be a multiple of four too: three new fields above + one
    # new pad = four. The first attempt added three fields and three pads, which
    # is six, and check_uniform_layout.py caught every vec4 array after it
    # sitting 8 bytes further along in GLSL than in C -- exactly the failure that
    # note below describes, found by the build instead of by a day of debugging.
    # SMOKE'S OWN UNLIT FLOOR, and it needs one because the volume's ambient is
    # per FRAME and global. rt_smoke_ambient was only ever applied when smoke
    # OWNED the volume (no fog and rt_volume_type 0), which the shipping config
    # never is -- so smoke had no self-visibility at all and was visible only
    # while a light was actually on it. A muzzle flash lasts 2-3 frames; the
    # trail it leaves lasts two seconds, and all of that was invisible.
    #
    # Per froxel, like smokeLightNearFade and smokeIllumBlend, so a cell with no
    # smoke in it is untouched and the fog collapse still holds.
    (TYPE_FLOAT32,      1,      "smokeAmbient",                     1),
    # How hard a smoke cell asserts its OWN albedo against the medium it is
    # mixed into. smoke_blendTint is a density-weighted average, which is
    # correct and has a bad consequence: a thin puff standing in a lit room is
    # mostly the ROOM's medium colour, so powder smoke turns beige in a beige
    # room and vanishes. 0 = the honest weighted average, 1 = the puff keeps its
    # own colour regardless of how thin it is.
    (TYPE_FLOAT32,      1,      "smokeTintBias",                    1),

    # ABSORPTION, smoke only, and the reason smoke is invisible in a bright room.
    #
    # The froxel stores rgb = in-scattered light and a = EXTINCTION, which is
    # scattering + absorbtion -- and absorbtion has been hardcoded 0 since the
    # volume was written, so the medium is purely scattering. A purely
    # scattering medium ADDS as much light as it takes away: in a dark room the
    # addition is what you see, and in a room lit evenly from above the two
    # cancel and the puff has no contrast against the wall in either direction.
    # That is exactly the report -- a barrel's fat burst still reads because its
    # density is 8x higher, while the gun's thin wisp disappears entirely.
    #
    # Real powder smoke is sooty: its single-scattering albedo is well under 1.
    # This is that missing fraction, and it is what lets smoke read DARK against
    # a bright background instead of only bright against a dark one.
    (TYPE_FLOAT32,      1,      "smokeAbsorb",                      1),
    # Doom64-RT: take the DIRECTIONAL light out of ReSTIR's per-pixel lottery and
    # shade it deterministically instead (RaygenCommon.h, calcSunOnlyReservoir).
    # A weak-but-huge light loses that lottery on most pixels, so its shadows are
    # resolved on a sparse random subset and the denoiser flattens them -- the
    # moon casting no sprite shadows while a muzzle flash casts perfect ones.
    # Taken from the _pads8 slot, a float where a uint was, so the scalar run
    # length and therefore std140 are unchanged.
    (TYPE_FLOAT32,      1,      "sunSplit",                         1),
    # Doom64-RT: the DEPTH half of the per-pixel volume jitter, in FROXELS,
    # split out from volumeDither because the two axes are not the same problem.
    #
    # The jitter is drawn from a HEMISPHERE and negated (CmScatterAccum.comp), so
    # its z component is always <= 0: the volume is never sampled deeper than the
    # surface, only shallower. That is deliberate -- it is what stops the jitter
    # reading a froxel column belonging to geometry BEHIND the surface, which is
    # the dark-outline-through-smoke artefact -- but it makes the depth term a
    # BIAS rather than a dither. g_volumetric is a prefix sum from the camera
    # outward, so a consistently shallower sample returns consistently less
    # accumulated in-scattering, and the far end of every column is simply
    # missing.
    #
    # Mean shortfall is 0.33 * radius * (volumeCameraFar / VOLUMETRIC_SIZE_Z)
    # metres -- E[rnd01] = 0.5 times E[sqrt(1-u)] = 2/3. At the shipping pins
    # (dither 5, far 60) that was 1.56 m of deleted light shaft, five times the
    # stock 0.31 m, because rt_volume_dither 5 and rt_volume_far 60 multiply.
    #
    # x/y keep volumeDither: those components are r*cos(phi), r*sin(phi),
    # symmetric about zero, so they blur the shaft's edge without moving it.
    # Depth only has to break SLICE BANDING, a half-froxel artefact, so it wants
    # a sub-froxel radius and nothing more.
    #
    # Taken from the _pads9 slot -- a float where a uint was, the same trick
    # sunSplit above used -- so the scalar run length and therefore std140 are
    # unchanged and check_uniform_layout.py stays quiet.
    (TYPE_FLOAT32,      1,      "volumeDitherZ",                    1),

    # --- Light shafts from ordinary lamps (Doom64-RT) -------------------------
    # NINE fields where there was ONE pad, and the count is arithmetic, not
    # taste: the scalar run from smokeCount down to here must stay a MULTIPLE OF
    # FOUR or C and std140 disagree from the first vec4 array onward. Removing
    # _pads10 and adding nine is a net +8. tools/check_uniform_layout.py is the
    # gate, and build-rtgl.cmd refuses to build when it complains.
    #
    # How many entries of volumeShaftLights below are live. 0 is the stock
    # volume: the loop in RtVolumetric.rgen does not execute and nothing about
    # the single-light path changes.
    (TYPE_UINT32,       1,      "volumeShaftCount",                 1),
    # Multiplier on what these lights scatter, on top of volumeLightMult. Its
    # own knob so a lamp's shaft can be tuned without moving the sun's.
    (TYPE_FLOAT32,      1,      "volumeShaftMult",                  1),
    # Near-light fade in metres, the same mechanism as volumeLightNearFade and
    # needed more here: these are point lights sitting INSIDE the medium, so the
    # froxels touching a bulb saturate by inverse square and the shaft reads as
    # a white ball instead of a beam.
    (TYPE_FLOAT32,      1,      "volumeShaftNearFade",              1),
    # Radiance below which a light is skipped BEFORE its shadow ray is traced.
    # This is the cull that makes the loop affordable -- a few ALU against one
    # ray per cell -- and in a typical froxel only one or two lamps survive it.
    (TYPE_FLOAT32,      1,      "volumeShaftMinRadiance",           1),
    # Hard cap on shadow rays per froxel, independent of volumeShaftCount. The
    # list arrives nearest-first, so the budget is spent on the lights that
    # actually reach the cell.
    (TYPE_UINT32,       1,      "volumeShaftMaxTraced",             1),
    # The phase function's asymmetry for THESE lights only. The moon's shafts
    # are tuned at volumeAsymmetry, and its ~11x forward bias is what carries
    # them; a lamp overhead is seen from every angle at once and wants a
    # different one. Sharing the knob would mean retuning nine fogged maps and
    # the moon every time a corridor lamp looked wrong -- the same coupling
    # rt_solo_small_intensity and rt_ceiling_bulb_gain exist to avoid on the
    # engine side. Below -1 means "use volumeAsymmetry".
    (TYPE_FLOAT32,      1,      "volumeShaftAsym",                  1),
    # Shader-side probe, and it exists because this project has twice concluded
    # a knob was wrong when the value was never reaching the shader at all.
    #   1 -- paint a froxel RED wherever any shaft light survives the radiance
    #        cull, before any shadow ray. Says the list arrived and the indices
    #        resolve, with no dependence on visibility or on magnitude.
    #   2 -- as 1, but only where the light was also UNSHADOWED. The pair
    #        separates "no light reaches this cell" from "the occluder blocks
    #        it", which the final image cannot.
    #   3 -- paint every froxel GREEN whenever the list is non-empty, with no
    #        position or visibility test at all. Blank here means the uniform is
    #        not being read.
    (TYPE_UINT32,       1,      "volumeShaftDebug",                 1),
    # HOW FAST A SHAFT DIES WITH DISTANCE FROM ITS LAMP, and the reason lamp
    # shafts read as a puddle of light around the bulb rather than as a beam.
    #
    # sampleSphereLight sets dw = calcSolidAngleForSphere(radius, d), i.e. the
    # scattering is INVERSE SQUARE -- 36x dimmer at 6 m than at 1 m. That is
    # correct for a bare point light and it is not what a light shaft looks
    # like: the shafts this renderer already had come from the SUN, which is
    # directional and does not fall off at all, which is exactly why they read
    # across a whole level and a lamp's does not.
    #
    # This is the exponent given back: radiance *= pow( max( d, 1 ), k ).
    #   0 = physical inverse square (what a bare bulb does)
    #   1 = 1/d
    #   2 = no falloff beyond a metre, i.e. sun-like
    # The max(d,1) means nothing inside a metre is ever boosted, so this cannot
    # fight volumeShaftNearFade, which owns that region.
    (TYPE_FLOAT32,      1,      "volumeShaftFalloff",               1),
    # RELATIVE cull, as a fraction of the brightest candidate AT THIS FROXEL.
    #
    # volumeShaftMinRadiance is an absolute floor and cannot answer the question
    # that matters here, which is not "is this light bright" but "is this light
    # worth a ray COMPARED TO the others reaching this cell". Without it the ray
    # budget is spent in list order -- and the list is sorted by distance to the
    # CAMERA, so froxels far down a corridor were tested against the lamps behind
    # the player and never against their own.
    #
    # 0 disables it and restores pure list order.
    (TYPE_FLOAT32,      1,      "volumeShaftRelCull",               1),

    # --- The froxel depth gate (Doom64-RT) ------------------------------------
    # EXACTLY FOUR FIELDS, and the count is arithmetic rather than taste: there
    # are no _pads left, and the scalar run from smokeCount down to here must
    # stay a MULTIPLE OF FOUR or C and std140 disagree from the first vec4 array
    # onward. tools/check_uniform_layout.py is the gate and build-rtgl.cmd
    # refuses to build when it complains.
    #
    # WHAT IT FIXES. g_volumetric is a prefix sum from the camera outward
    # (CmVolumetricProcess.comp) stored at froxel CENTRES, and it is read
    # TRILINEARLY at the surface's distance. A wall therefore collects
    # sum(up to slice k) + frac * (own contribution of slice k+1) -- and slice
    # k+1 is BEHIND the wall, in the next room, where the froxel legitimately
    # sees a lamp and is legitimately bright. Up to a whole slice of the lit air
    # behind a wall lands on it, and volume_toSamplePosition_T CLAMPS z to
    # [0,1], so a surface nearer than slice 0's centre (volumeCameraNear +
    # 0.0078 * reach, i.e. ~0.47 m at a 60 m reach) collects slice 0 WHOLESALE.
    # That is why the leak is worst with your face against the wall.
    #
    # No per-light visibility test can fix it -- every shadow ray involved is
    # correct, and the wall IS shadowing the froxels in front of it. The wall
    # simply falls inside a cell that was shaded for the far side of it. So the
    # gate is at the CELL: weight a froxel by how much of it lies in front of
    # the visible surface for its own screen column.
    #
    # Measured negative before this was written: the leak survives
    # rt_cpu_cullmode 2 (whole map in the acceleration structure), so it is not
    # a missing occluder. See docs/plan-light-shafts.md 4d.
    #
    # 0 = off, stock behaviour.
    (TYPE_FLOAT32,      1,      "volumeDepthGate",                  1),
    # METRES of slack beyond the surface before a cell starts being weighted
    # down. 0 centres the ramp on the surface itself, which is the physical
    # reading: a cell straddling the surface contributes the fraction of itself
    # that is in front of it.
    (TYPE_FLOAT32,      1,      "volumeDepthGateBias",              1),
    # Width of that ramp in FROXEL SLICES, centred on surface + bias. 1 = the
    # cell containing the surface contributes about half, which is what a
    # straddling cell physically should. Larger is softer and leaks more;
    # 0 is a hard cut and shows the grid.
    (TYPE_FLOAT32,      1,      "volumeDepthGateFeather",           1),
    # Depth taps across the froxel column's screen footprint, and the MAXIMUM is
    # used. 1 = centre only; 5 = centre + four corners.
    #
    # The max matters: one column covers renderWidth/160 x renderHeight/88
    # pixels -- about 12x12 at 1080p -- and those pixels can see very different
    # depths. Taking the max means a cell is only killed when it is behind
    # EVERYTHING in the footprint, so air genuinely visible past a thin
    # foreground edge survives. Under-culling is the safe direction here; the
    # centre tap alone draws a hard edge along every silhouette.
    (TYPE_UINT32,       1,      "volumeDepthGateTaps",              1),

    # Doom64-RT: TELL THE UPSCALER THE MEDIUM IS THERE.
    #
    # The dark line at every edge seen through smoke or fog is drawn by the
    # TEMPORAL UPSCALER, not by this grid -- measured on MAP93, where turning
    # DLSS and FSR both off removes it completely while the depth gate, the
    # dither, the blur, the depth bias, the scattering history and the slice
    # thickness each move it by 0.3 luminance levels or less.
    # docs/rt-volumetric-edge-outlines.md has the ladder.
    #
    # The reason is ordering: CmPrepareFinal composites hdr*a + rgb at RENDER
    # resolution and the upscaler runs after it, so what DLSS receives at a
    # silhouette is a colour with two very different media states already baked
    # into either side of the edge -- and nothing in its inputs says that
    # discontinuity belongs to the medium rather than to the surface. Both
    # upscalers have an input for exactly this (DLSS pInBiasCurrentColorMask,
    # FSR2 transparencyAndComposition) and RTGL passed neither.
    #
    # These four build that mask. Master strength; 0 = off, and the mask is
    # written as zero so the upscalers behave exactly as before.
    (TYPE_FLOAT32,      1,      "volumeUpscaleBias",                1),
    # The transmittance DIFFERENCE across a pixel that counts as a full-strength
    # edge. This is what keeps the mask tight: biasing toward the current frame
    # trades the outline for noise, so it must land on silhouettes and not on
    # the whole veil -- the veil is where the smoke's own noise lives, and that
    # noise is exactly what the temporal history is there to average away.
    (TYPE_FLOAT32,      1,      "volumeUpscaleBiasEdge",            1),
    # A constant floor applied wherever the medium is present at all, before the
    # edge term. 0 = silhouettes only, which is the intent; raise it only to
    # answer "is the mask reaching the upscaler", because a floor over the whole
    # veil is the noisy arm by construction.
    (TYPE_FLOAT32,      1,      "volumeUpscaleBiasFloor",           1),
    # 1 = paint the mask on screen instead of the image. A mask handed to a
    # black box is otherwise unobservable: without this, "no change" cannot be
    # told apart from "the mask is all zeros", which is this project's most
    # expensive recurring failure.
    (TYPE_UINT32,       1,      "volumeUpscaleBiasDebug",           1),

    # Doom64-RT: APPLY THE MEDIUM AFTER THE UPSCALER (CmVolumeCompose.comp).
    #
    # The bias mask above treats the symptom and was measured doing it badly:
    # it buys part of the outline by discarding temporal history, and inside a
    # veil that history is what averages out the volume's noise, so the smoke
    # goes noisy in motion. This removes the cause instead. CmPrepareFinal stops
    # compositing, the upscaler reconstructs an ordinary surface image, and the
    # veil is applied at output resolution afterwards. Algebra preserved exactly;
    # see the header of CmVolumeCompose.comp.
    #
    # 1 also FORCES the emissive occlusion (the T factor moves into the post
    # pass), so rt_volume_occlude_emis 0 is not expressible while this is on.
    (TYPE_UINT32,       1,      "volumePostComp",                   1),
    # Doom64-RT: soften the medium's step across a silhouette by this many
    # PIXELS before compositing. A mitigation that works on any path, including
    # the ones postcomp has to leave alone (DLSS Ray Reconstruction, frame
    # generation): it attacks the step's contrast rather than the ordering, and
    # unlike the bias mask it never touches temporal history. 0 = off.
    (TYPE_FLOAT32,      1,      "volumeEdgeSoft",                   1),
    # How big a relative depth break counts as a silhouette for the feather
    # above, as the second difference of 1/depth. Lower marks more edges. This
    # is the reach knob: at the shipping 0.15 the strong silhouettes (a monster,
    # a pillar corner) are caught and the fainter ones -- a floor seam, a shallow
    # step -- are not, and those are what is left of the artefact.
    (TYPE_FLOAT32,      1,      "volumeEdgeSoftEdge",               1),
    # Doom64-RT: THE FIRST-PERSON WEAPON MUST NOT DAMAGE THE MEDIUM.
    # 0 = old path, 1 = fix, 2 = debug (paint the affected pixels).
    #
    # Two halves, both keyed off the FIRST_PERSON flag in
    # framebufSurfacePosition.w:
    #   * CmScatterAccum, under the weapon, integrates the froxel volume TO FAR
    #     instead of to the weapon's 0.3 m -- so the buffer keeps a live,
    #     accumulated estimate of the WORLD's fog flowing under the sprite, and
    #     uncovering is seamless. (A freeze was tried first and dragged the fog
    #     around with the gun -- the motion vectors under the weapon are the
    #     weapon's own.)
    #   * CmPrepareFinal composites NO fog on first-person pixels -- there is
    #     ~0.3 m of medium between the eye and the gun, so "no fog on the gun"
    #     is the physically correct image, and it is also what stops the stored
    #     world-fog values from being painted over the viewmodel.
    (TYPE_FLOAT32,      1,      "volumeFp",                         1),
    # Doom64-RT: how CmScatterAccum validates reprojected medium history.
    # 0 = the surface rules (strict depth + normal), 1 = MEDIA rules: relative
    # path-length within 25%, no normal test at all. The medium is an integral
    # along the ray -- it does not care what surface ends the ray, only how far
    # away it is. The surface rules reject history at every billboard silhouette
    # under camera motion for a difference (fog-to-monster at 20 m vs
    # fog-to-wall at 24 m) that is a few percent of veil.
    (TYPE_UINT32,       1,      "volumeReproj",                     1),
    # Doom64-RT: do sprites shadow the froxel medium? 0 = no (billboards and
    # their axis-plane shadow proxies are invisible to the volumetric pass's
    # shadow rays), 1 = the old behaviour. A camera-facing cutout casting a
    # volumetric shadow sheet is wrong in every frame it appears: the proxies
    # are solid planes (the shell casing stamped a RECTANGLE of shadow into the
    # muzzle smoke), and billboards rotate with the camera, so their sheets
    # sweep the fog as the view turns -- the MAP12 thunder smear.
    (TYPE_UINT32,       1,      "volumeSpriteShadow",               1),
    # Doom64-RT: IN-GRID temporal accumulation length, in frames. 0 = off (the
    # legacy screen-space accumulation in CmScatterAccum runs instead). When
    # > 0, CmVolumetricProcess EMA-blends each froxel cell against its world
    # position reprojected into the previous frame's grid -- no surfaces
    # involved, so no history rejection and no restart trails at silhouettes --
    # and CmScatterAccum stops accumulating in screen space entirely.
    (TYPE_FLOAT32,      1,      "volumeGridHistory",                1),
    # Doom64-RT: under the DLSS-RR pre-exposure reorder, put the screen-emission
    # GLOW back into RR's INPUT (in pre-exposure units: divided by the EV100
    # factor pre-RR, multiplied back post-RR -- algebraically exact). The
    # post-RR add sampled the JITTERED render-res glow buffer with one bilinear
    # tap per frame at a shifting sub-texel phase, which shimmered on every
    # sharp emissive edge -- reported from play as flickery/unstable lamp
    # bulbs. Pre-reorder the glow always sat in RR's input and was dejittered
    # and temporally stabilized with everything else; this restores that while
    # keeping exposure post-RR. Read only when rrPreExposure is on. Took the
    # volumeReserved3 spare (a uint where a float was -- same 4 bytes, scalar
    # run unchanged, tools/check_uniform_layout.py is the gate).
    (TYPE_UINT32,       1,      "rrGlowPre",                        1),

    # NRD lane stage 2. nrdValidation: CmNrdCompose paints NRD's own
    # OUT_VALIDATION overlay instead of the image -- the vendor-supplied
    # "did the guides arrive correctly" view. Three spares ride along because
    # THE SCALAR RUN MUST STAY A MULTIPLE OF FOUR (see volumeDepthGate's
    # comment above; tools/check_uniform_layout.py is the gate).
    (TYPE_UINT32,       1,      "nrdValidation",                    1),
    # rrGlowPre mode 2: artistic multiplier on the glow overlay (default 1).
    # Mode 2 divides by the SMOOTHED exposure (the 1x1 RrExposure EMA), so
    # the glow is steady-state exact at its old calibration in every room,
    # while the slowly-drifting divisor keeps RR's input pulse-free through
    # adaptation. (The first cut used a fixed mid-exposure constant here and
    # ran the glow ~7x hot in dark rooms.) Took the nrdReserved0 spare.
    (TYPE_FLOAT32,      1,      "rrGlowScale",                      1),
    # DLSS-RR albedo demodulation (the Remix approach, demodulate.comp.slang):
    # RR denoises LIGHTING ONLY -- CmNoisyCompose divides the combined
    # modulation factor M out of its input and stores M in RrDemodFactor;
    # CmRrPostExposure re-multiplies M at OUTPUT resolution with the filter
    # below. The pixel-art texel edges live in M and never enter the network,
    # which is what was blurring them (soft even at DLAA = the model's floor).
    # rrDemodFilter: 0 = bilinear, 1 = Catmull-Rom, 2 = nearest (full chunky).
    # Took the two nrdReserved spares.
    (TYPE_UINT32,       1,      "rrDemod",                          1),
    (TYPE_UINT32,       1,      "rrDemodFilter",                    1),
    # Doom64-RT: the first-person weapon must not damage the SURFACE
    # denoiser's history either (docs/rt-volumetric-weapon-trails.md, the
    # same class one buffer over). svgfFp: 0 = stock, 1 = a pixel whose
    # history was fully rejected borrows validated neighbour history instead
    # of restarting from one sample, 2 = debug (tint borrowed pixels).
    # svgfFpGrad: 0 = stock, 1 = the A-SVGF gradient never samples the
    # weapon and treats a vanished light as a change, not as "no change".
    (TYPE_UINT32,       1,      "svgfFp",                           1),
    (TYPE_UINT32,       1,      "svgfFpGrad",                       1),
    # Doom64-RT: the indirect (bounce) ghost. A-SVGF accumulated indirect over
    # up to 256 frames, so a rocket flash's bounce lingered for seconds after
    # the light died -- read in play as "the light lingers". svgfIndirMaxHist
    # caps the indirect history length in frames (0 = stock 256).
    # svgfIndirAntilag 1 drops the "if it's bright enough, don't drop history"
    # suppression on the indirect antilag, which protected the brightest ghosts.
    (TYPE_FLOAT32,      1,      "svgfIndirMaxHist",                 1),
    (TYPE_UINT32,       1,      "svgfIndirAntilag",                 1),

    # Doom64-RT: VOLUMETRIC CLOUDS (RgDrawFrameVolumetricCloudParams). All
    # vec4s, so they sit after the scalar run and cannot disturb it.
    #   cloudParams0: x enabled, y altitude (m), z thickness (m), w coverage
    #   cloudParams1: x density /m, y 1/featureSize, z detail, w time (s)
    #   cloudParams2: x steps, y lightSteps, z horizonFade (deg), w historyBlend
    #   cloudParams3: x transmitFloor, y asymmetry, z debugMode, w wind.x (m/s)
    #   cloudTint:    rgb albedo, w wind.y (m/s)
    #   cloudLightDir: xyz toward the light, w underStrength
    #   cloudLightColor / cloudUnderColor / cloudAmbient: rgb, w spare
    (TYPE_FLOAT32,      4,      "cloudParams0",                     1),
    (TYPE_FLOAT32,      4,      "cloudParams1",                     1),
    (TYPE_FLOAT32,      4,      "cloudParams2",                     1),
    (TYPE_FLOAT32,      4,      "cloudParams3",                     1),
    (TYPE_FLOAT32,      4,      "cloudTint",                        1),
    (TYPE_FLOAT32,      4,      "cloudLightDir",                    1),
    (TYPE_FLOAT32,      4,      "cloudLightColor",                  1),
    (TYPE_FLOAT32,      4,      "cloudUnderColor",                  1),
    (TYPE_FLOAT32,      4,      "cloudAmbient",                     1),
    # THE FIRE SKY. cloudBackColor: rgb = fire colour, w = strength of the
    # glow BEHIND the slab (added as glow * T in the composite: thin cloud
    # burns, dense cloud is a silhouette, clear sky is the glow).
    # cloudFireParams: x = emission strength of FIRE POCKETS -- a second,
    # sparse noise field inside the slab that glows along the ray and lights
    # the cloud around it (the "fire between the clouds"); y = 1/pocket
    # feature size; z = pocket threshold (1 - cover); w = how strongly a
    # pocket lights the cloud near it.
    (TYPE_FLOAT32,      4,      "cloudBackColor",                   1),
    (TYPE_FLOAT32,      4,      "cloudFireParams",                  1),
    # cloudLayerParams: x = layers (1 or 2), y = gap fraction of the slab
    # between the two decks (the fire sheet lives in it), z = sheet
    # extinction relative to the cloud's, w spare.
    (TYPE_FLOAT32,      4,      "cloudLayerParams",                 1),
    # cloudFireAnim: x = slow pulse amplitude (0..1), y = pulse speed, z =
    # fast flicker amplitude, w = 1/streak width of the cascades (m).
    # cloudCascade: FLAME CASCADES -- fire raining from the cloud base as
    # vertical streaks. x = emission, y = 1/length below the base (m), z =
    # streak threshold (1 - cover), w = fall speed (m/s). Time is cloudParams1.w.
    (TYPE_FLOAT32,      4,      "cloudFireAnim",                    1),
    (TYPE_FLOAT32,      4,      "cloudCascade",                     1),

    # xyz = centre in world space (metres, the same space as a light's position
    # and as volume_getCenter's output), w = radius in metres.
    (TYPE_FLOAT32,      4,      "smokePuffs",           CONST[ "SMOKE_PUFF_MAX" ]),
    # rgb = scattering albedo, a = density at the core, already scaled by the
    # volume's slice thickness engine-side so it reads as optical depth per
    # METRE and does not change meaning when the volume's reach does.
    (TYPE_FLOAT32,      4,      "smokeAlbedoDensity",   CONST[ "SMOKE_PUFF_MAX" ]),
    # x = the puff's radius ACROSS THE VIEW, in metres, which is the only radius
    # you actually see. smokePuffs.w is its radius ALONG the view, held at half a
    # froxel slice so the grid can resolve it at all. yzw spare.
    (TYPE_FLOAT32,      4,      "smokeShape",           CONST[ "SMOKE_PUFF_MAX" ]),

    # Doom64-RT: shader indices of the lights that get air around them, packed
    # four to a uvec4 -- std140 pads a bare uint array to 16 bytes an element,
    # so an array of scalars would cost four times this for nothing. Unpacked in
    # the shader as volumeShaftLights[ i >> 2 ][ i & 3 ]. LIGHT_INDEX_NONE marks
    # an entry the engine listed but whose light was not uploaded this frame.
    (TYPE_UINT32,       4,      "volumeShaftLights",    CONST[ "VOLUME_SHAFT_LIGHT_MAX" ] // 4),

    # Doom64-RT: albedo filtering (RgDrawFrameTexturesParams::albedoFilterMode).
    # x = mode (0 hardware sampler, 1 N64 three-point), y = the sampler's
    # mipLodBias, which the shader must add to its own LOD and cancel from the
    # explicit LOD it samples at. zw spare. A vec4, so it sits after the scalar
    # run and cannot disturb it.
    (TYPE_FLOAT32,      4,      "texFilterParams",      1),

    # for std140
    (TYPE_FLOAT32,     44,      "viewProjCubemap",              6),
    (TYPE_FLOAT32,     44,      "skyCubemapRotationTransform",  1),
]

GEOM_INSTANCE_STRUCT = [
    (TYPE_FLOAT32,      4,      "model_0",              1),
    (TYPE_FLOAT32,      4,      "model_1",              1),
    (TYPE_FLOAT32,      4,      "model_2",              1),
    (TYPE_FLOAT32,      4,      "prevModel_0",          1),
    (TYPE_FLOAT32,      4,      "prevModel_1",          1),
    (TYPE_FLOAT32,      4,      "prevModel_2",          1),

    (TYPE_UINT32,       1,      "flags",                1),
    (TYPE_UINT32,       1,      "texture_base",         1),
    (TYPE_UINT32,       1,      "texture_base_ORM",     1),
    (TYPE_UINT32,       1,      "texture_base_N",       1),
    (TYPE_UINT32,       1,      "texture_base_E",       1),
    (TYPE_UINT32,       1,      "texture_layer1",       1),
    (TYPE_UINT32,       1,      "texture_layer2",       1),
    (TYPE_UINT32,       1,      "texture_layer3",       1),

    (TYPE_UINT32,       1,      "colorFactor_base",     1),
    (TYPE_UINT32,       1,      "colorFactor_layer1",   1),
    (TYPE_UINT32,       1,      "colorFactor_layer2",   1),
    (TYPE_UINT32,       1,      "colorFactor_layer3",   1),

    (TYPE_UINT32,       1,      "baseVertexIndex",      1),
    (TYPE_UINT32,       1,      "baseIndexIndex",       1),
    (TYPE_UINT32,       1,      "prevBaseVertexIndex",  1),
    (TYPE_UINT32,       1,      "prevBaseIndexIndex",   1),

    (TYPE_UINT32,       1,      "vertexCount",          1),
    (TYPE_UINT32,       1,      "indexCount",           1),
    (TYPE_UINT32,       1,      "texture_base_D",       1),
    (TYPE_UINT32,       1,      "roughnessDefault_metallicDefault", 1),

    (TYPE_FLOAT32,      1,      "emissiveMult",         1),
    (TYPE_UINT32,       1,      "firstVertex_Layer1",   1),
    (TYPE_UINT32,       1,      "firstVertex_Layer2",   1),
    (TYPE_UINT32,       1,      "firstVertex_Layer3",   1),

    # Doom64-RT: GI emission for EMIS_SCREEN_SCALED instances, in the material's
    # units, animated separately from emissiveMult (which is then screen-only).
    # See RgMeshPrimitiveInfo::emissiveGi. Padded to a whole 16-byte row.
    (TYPE_FLOAT32,      1,      "emissiveMultGi",       1),
    (TYPE_UINT32,       1,      "_padGi0",              1),
    (TYPE_UINT32,       1,      "_padGi1",              1),
    (TYPE_UINT32,       1,      "_padGi2",              1),
]

# TODO: make more compact
LIGHT_ENCODED_STRUCT = [
    (TYPE_UINT32,       1,      "lightType",            1),
    (TYPE_UINT32,       1,      "colorE5",              1),
    (TYPE_FLOAT32,      1,      "ldata0",               1),
    (TYPE_FLOAT32,      1,      "ldata1",               1),
    (TYPE_FLOAT32,      1,      "ldata2",               1),
    (TYPE_FLOAT32,      1,      "ldata3",               1),
]

# TODO: light index / target pdf - 16 bits
LIGHT_IN_CELL = [
    (TYPE_UINT32,       1,      "selected_lightIndex",  1),
    (TYPE_FLOAT32,      1,      "selected_targetPdf",   1),
    (TYPE_FLOAT32,      1,      "weightSum",            1),
]

TONEMAPPING_STRUCT = [
    (TYPE_UINT32,       1,      "histogram",            CONST["COMPUTE_LUM_HISTOGRAM_BIN_COUNT"]),
    (TYPE_FLOAT32,      1,      "avgLuminance",         1),
]

INDIRECT_DRAW_CMD_STRUCT = [
    (TYPE_UINT32,       1,      "indexCount",           1),
    (TYPE_UINT32,       1,      "instanceCount",        1),
    (TYPE_UINT32,       1,      "firstIndex",           1),
    (TYPE_INT32,        1,      "vertexOffset",         1),
    (TYPE_UINT32,       1,      "firstInstance",        1),
    (TYPE_FLOAT32,      1,      "positionToCheck_X",    1),
    (TYPE_FLOAT32,      1,      "positionToCheck_Y",    1),
    (TYPE_FLOAT32,      1,      "positionToCheck_Z",    1),
]

LENS_FLARES_INSTANCE_STRUCT = [
    (TYPE_UINT32,       1,      "packedColor",          1),
    (TYPE_UINT32,       1,      "textureIndex",         1),
    (TYPE_UINT32,       1,      "emissiveTextureIndex", 1),
    (TYPE_FLOAT32,      1,      "emissiveMult",         1),
]

PORTAL_INSTANCE_STRUCT = [
    (TYPE_FLOAT32,      4,      "inPosition",               1),
    (TYPE_FLOAT32,      4,      "outPosition",              1),
    (TYPE_FLOAT32,      4,      "outDirection",             1),
    (TYPE_FLOAT32,      4,      "outUp",                    1),
]

STRUCT_ALIGNMENT_NONE       = 0
STRUCT_ALIGNMENT_STD430     = 1
STRUCT_ALIGNMENT_STD140     = 2

STRUCT_BREAK_TYPE_NONE      = 0
STRUCT_BREAK_TYPE_COMPLEX   = 1
STRUCT_BREAK_TYPE_ONLY_C    = 2

# (structTypeName): (structDefinition, onlyForGLSL, alignmentFlags, breakComplex)
# alignmentFlags    -- if using a struct in dynamic array, it must be aligned with 16 bytes
# breakType         -- if member's type is not primitive and its count>0 then
#                      it'll be represented as an array of primitive types
STRUCTS = {
    "ShVertex":                 (VERTEX_STRUCT,                 False,  STRUCT_ALIGNMENT_STD140,    0),
    "ShVertexCompact":          (VERTEX_COMPACT_STRUCT,         False,  STRUCT_ALIGNMENT_STD140,    0),
    "ShGlobalUniform":          (GLOBAL_UNIFORM_STRUCT,         False,  STRUCT_ALIGNMENT_STD140,    STRUCT_BREAK_TYPE_ONLY_C),
    "ShGeometryInstance":       (GEOM_INSTANCE_STRUCT,          False,  STRUCT_ALIGNMENT_STD430,    0),
    "ShTonemapping":            (TONEMAPPING_STRUCT,            False,  0,                          0),
    "ShLightEncoded":           (LIGHT_ENCODED_STRUCT,          False,  0,                          0),
    "ShLightInCell":            (LIGHT_IN_CELL,                 False,  STRUCT_ALIGNMENT_STD430,    0),
    "ShIndirectDrawCommand":    (INDIRECT_DRAW_CMD_STRUCT,      False,  STRUCT_ALIGNMENT_STD430,    0),
    # TODO: should be STRUCT_ALIGNMENT_STD430, but current generator is not great as it just adds pads at the end, so it's 0
    "ShLensFlareInstance":      (LENS_FLARES_INSTANCE_STRUCT,   False,  0,                          0),
    "ShPortalInstance":         (PORTAL_INSTANCE_STRUCT,        False,  STRUCT_ALIGNMENT_STD140,    0),
}

# --------------------------------------------------------------------------------------------- #
# User defined buffers: uniform, storage buffer
# --------------------------------------------------------------------------------------------- #





# --------------------------------------------------------------------------------------------- #
# User defined framebuffers
# --------------------------------------------------------------------------------------------- #

FRAMEBUF_DESC_SET_NAME              = "DESC_SET_FRAMEBUFFERS"
FRAMEBUF_BASE_BINDING               = 0
FRAMEBUF_PREFIX                     = "framebuf"
FRAMEBUF_SAMPLER_POSTFIX            = "_Sampler"
FRAMEBUF_DEBUG_NAME_PREFIX          = "Framebuf "
FRAMEBUF_STORE_PREV_POSTFIX         = "_Prev"
FRAMEBUF_SAMPLER_INVALID_BINDING    = "FB_SAMPLER_INVALID_BINDING"

# only info for 2 frames are used: current and previous
FRAMEBUF_FLAGS_STORE_PREV           = 1 << 0
FRAMEBUF_FLAGS_NO_SAMPLER           = 1 << 1
FRAMEBUF_FLAGS_IS_ATTACHMENT        = 1 << 2
FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM     = 1 << 3
FRAMEBUF_FLAGS_FORCE_SIZE_1_3       = 1 << 4
FRAMEBUF_FLAGS_BILINEAR_SAMPLER     = 1 << 9
FRAMEBUF_FLAGS_UPSCALED_SIZE        = 1 << 10
FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE    = 1 << 11
FRAMEBUF_FLAGS_USAGE_TRANSFER       = 1 << 12

# only these flags are shown for C++ side
FRAMEBUF_FLAGS_ENUM = {
    "FRAMEBUF_FLAGS_IS_ATTACHMENT"      : FRAMEBUF_FLAGS_IS_ATTACHMENT,
    "FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM"   : FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM,
    "FRAMEBUF_FLAGS_FORCE_SIZE_1_3"     : FRAMEBUF_FLAGS_FORCE_SIZE_1_3,
    "FRAMEBUF_FLAGS_BILINEAR_SAMPLER"   : FRAMEBUF_FLAGS_BILINEAR_SAMPLER,
    "FRAMEBUF_FLAGS_UPSCALED_SIZE"      : FRAMEBUF_FLAGS_UPSCALED_SIZE,
    "FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE"  : FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE,
    "FRAMEBUF_FLAGS_USAGE_TRANSFER"     : FRAMEBUF_FLAGS_USAGE_TRANSFER,
}

FRAMEBUFFERS = {
    # (image name)                      : (base format type, components,    flags)
    "Albedo"                            : (TYPE_PACK_11,    COMPONENT_RGB,  FRAMEBUF_FLAGS_IS_ATTACHMENT),
    "IsSky"                             : (TYPE_UINT8,      COMPONENT_R,    0),
    "Normal"                            : (TYPE_UINT32,     COMPONENT_R,    FRAMEBUF_FLAGS_STORE_PREV),
    "MetallicRoughness"                 : (TYPE_UNORM8,     COMPONENT_RG,   FRAMEBUF_FLAGS_STORE_PREV),
    "DepthWorld"                        : (TYPE_FLOAT16,    COMPONENT_R,    FRAMEBUF_FLAGS_STORE_PREV),
    "DepthGrad"                         : (TYPE_FLOAT16,    COMPONENT_R,    0),
    "DepthNdc"                          : (TYPE_FLOAT32,    COMPONENT_R,    0),
    "DepthFluid"                        : (TYPE_FLOAT32,    COMPONENT_R,    0),
    "DepthFluidTemp"                    : (TYPE_FLOAT32,    COMPONENT_R,    0),
    "FluidNormal"                       : (TYPE_UINT32,     COMPONENT_R,    FRAMEBUF_FLAGS_IS_ATTACHMENT),
    "FluidNormalTemp"                   : (TYPE_UINT32,     COMPONENT_R,    0),
    "Motion"                            : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "UnfilteredDirect"                  : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "UnfilteredSpecular"                : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "UnfilteredIndir"                   : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "SurfacePosition"                   : (TYPE_FLOAT32,    COMPONENT_RGBA, FRAMEBUF_FLAGS_STORE_PREV),
    "VisibilityBuffer"                  : (TYPE_FLOAT32,    COMPONENT_RGBA, FRAMEBUF_FLAGS_STORE_PREV),
    "ViewDirection"                     : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_STORE_PREV),
    "PrimaryToReflRefr"                 : (TYPE_UINT32,     COMPONENT_RGBA, 0),
    "Throughput"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "PreFinal"                          : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "Final"                             : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_IS_ATTACHMENT),

    "UpscaledPing"                      : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_IS_ATTACHMENT | FRAMEBUF_FLAGS_UPSCALED_SIZE | FRAMEBUF_FLAGS_USAGE_TRANSFER),  # dst for DLSS and blitting in,
    "UpscaledPong"                      : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_IS_ATTACHMENT | FRAMEBUF_FLAGS_UPSCALED_SIZE | FRAMEBUF_FLAGS_USAGE_TRANSFER),  #       src for WipeEffectSource 

    # for upscalers
    "MotionDlss"                        : (TYPE_FLOAT16,    COMPONENT_RG,   0),
    "Reactivity"                        : (TYPE_UNORM8,     COMPONENT_R,    FRAMEBUF_FLAGS_IS_ATTACHMENT),
    "HudOnly"                           : (TYPE_UNORM8,     COMPONENT_RGBA, FRAMEBUF_FLAGS_IS_ATTACHMENT | FRAMEBUF_FLAGS_UPSCALED_SIZE | FRAMEBUF_FLAGS_USAGE_TRANSFER),  # src for framegen

    "AccumHistoryLength"                : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_STORE_PREV),
    
    # TODO: pack float16 to e5
    "DiffTemporary"                     : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "DiffAccumColor"                    : (TYPE_PACK_E5,    COMPONENT_RGB,  FRAMEBUF_FLAGS_STORE_PREV),
    "DiffAccumMoments"                  : (TYPE_FLOAT16,    COMPONENT_RG,   FRAMEBUF_FLAGS_STORE_PREV),
    "DiffColorHistory"                  : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "DiffPingColorAndVariance"          : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "DiffPongColorAndVariance"          : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    
    "SpecAccumColor"                    : (TYPE_PACK_E5,    COMPONENT_RGB,  FRAMEBUF_FLAGS_STORE_PREV),
    "SpecPingColor"                     : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "SpecPongColor"                     : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    
    "IndirAccum"                        : (TYPE_PACK_E5,    COMPONENT_RGB,  FRAMEBUF_FLAGS_STORE_PREV),
    "IndirPing"                         : (TYPE_PACK_E5,    COMPONENT_RGB,  0),
    "IndirPong"                         : (TYPE_PACK_E5,    COMPONENT_RGB,  0),

    "AtrousFilteredVariance"            : (TYPE_FLOAT16,    COMPONENT_R,    0),
    
    "NormalDecal"                       : (TYPE_UINT32,     COMPONENT_R,    FRAMEBUF_FLAGS_IS_ATTACHMENT),
    
    # Doom64-RT: BILINEAR, for CmVolumeCompose.comp -- the post-upscale pass
    # samples this by normalised coordinate at OUTPUT resolution, so it needs
    # interpolation between render-res texels. Safe for everything else: every
    # other consumer reads it with texelFetch, which ignores the sampler filter.
    "Scattering"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_STORE_PREV | FRAMEBUF_FLAGS_BILINEAR_SAMPLER),
    "ScatteringHistory"                 : (TYPE_FLOAT16,    COMPONENT_R,    FRAMEBUF_FLAGS_STORE_PREV),
    
    # need separate one for RT, to resolve checkerboarded across multiple pixels
    "ScreenEmisRT"                      : (TYPE_PACK_11,    COMPONENT_RGB,  0),
    # Doom64-RT: BILINEAR for CmRrPostExposure.comp, which re-adds the screen
    # emissive at OUTPUT resolution under the DLSS-RR pre-exposure reorder and
    # samples this by normalised coordinate. Same argument as Scattering above:
    # every other consumer reads it with texelFetch, which ignores the sampler
    # filter (CmPrepareFinal.comp, CmDecalNormalsCopy.comp), or writes it as an
    # image/attachment, which has no sampler at all.
    "ScreenEmission"                    : (TYPE_PACK_11,    COMPONENT_RGB,  FRAMEBUF_FLAGS_IS_ATTACHMENT | FRAMEBUF_FLAGS_BILINEAR_SAMPLER),

    "Bloom"                             : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip1"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip2"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip3"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip4"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip5"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip6"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
    "Bloom_Mip7"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_BLOOM | FRAMEBUF_FLAGS_BILINEAR_SAMPLER | FRAMEBUF_FLAGS_UPSCALED_SIZE),
  
    "WipeEffectSource"                  : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_UPSCALED_SIZE | FRAMEBUF_FLAGS_USAGE_TRANSFER), # dst to copy in
    
    "Reservoirs"                        : (TYPE_UINT32,     COMPONENT_RG,   FRAMEBUF_FLAGS_STORE_PREV),
    "ReservoirsInitial"                 : (TYPE_UINT32,     COMPONENT_RG,   0),

    "IndirectReservoirsInitial"         : (TYPE_UINT32,     COMPONENT_RGBA, 0),

    # DLSS-RR: pInDisocclusionMask (sentinel 10000.0 forces RR history discard)
    # and the tile-luminance history it is computed from. Written by CmNoisyCompose
    # in regular (non-checkerboard) pixel space, RR path only.
    "RrDisocclusion"                    : (TYPE_FLOAT16,    COMPONENT_R,    0),
    "RrLumHistory"                      : (TYPE_FLOAT16,    COMPONENT_R,    FRAMEBUF_FLAGS_STORE_PREV),

    # DLSS-RR: pInSpecularHitDistance -- world distance from the shading point to
    # whatever produced the highlight. RR needs this to reproject specular, which
    # does NOT live on the surface; without it specular is reprojected as if it
    # did, so glossy surfaces smear and fizzle under camera motion. Was bound to
    # FB_DEPTH_WORLD once (primary-hit camera distance -- the wrong signal) and
    # then set to nullptr; this is the correct value. Written by RtRaygenDirect
    # in checkerboard space, resolved like the other direct outputs.
    "SpecularHitDistance"               : (TYPE_FLOAT16,    COMPONENT_R,    0),

    # DLSS-RR: pInExposureTexture -- the "1x1 texture containing the final
    # exposure scale" (nvsdk_ngx_defs.h). Under the pre-exposure reorder the
    # network's colour input is raw radiance whose absolute scale swings ~52x
    # with auto-exposure (EV100 2.0..7.7); the network is not scale-invariant,
    # so it must be told the scale. CmPrepareFinal thread (0,0) writes
    # ev100ToLuminousExposure(getCurrentEV100()) here -- the exact factor the
    # post-RR pass multiplies by, from the same tonemapping buffer in the same
    # frame. GPU-side because the value lives in tonemapping.avgLuminance and a
    # readback would cost a frame of latency.
    "RrExposure"                        : (TYPE_FLOAT32,    COMPONENT_R,    FRAMEBUF_FLAGS_SINGLE_PIXEL_SIZE),

    # NRD lane (docs/plan-nrd-denoiser.md stage 2): the ReLAX data path.
    # All render-res, regular (non-checkerboard) pixel space, written by
    # CmNrdPack from the same unfiltered buffers A-SVGF consumes, denoised by
    # NRDIntegration (NrdDenoiser.cpp), remodulated into PreFinal by
    # CmNrdCompose. Formats follow NRD's contracts:
    #  - radiance+hitDist: RGBA16F (RELAX_FrontEnd_PackRadianceAndHitDist)
    #  - normal+roughness: encoding 2 (_NRD_EncodeNormalRoughness101010)
    #    produces [0,1] values; FLOAT16 stores them losslessly enough and
    #    needs no new format plumbing (A2 formats would).
    #  - viewZ: true view-space Z (not radial distance), FP32 -- ReLAX bases
    #    its disocclusion tests on it.
    #  - motion: our 2.5D contract verbatim (xy = UV delta cur->prev,
    #    z = distance delta), motionVectorScale = {1,1,1}.
    "NrdDiffuse"                        : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdSpecular"                       : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdDiffuseOut"                     : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdSpecularOut"                    : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdNormalRoughness"                : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdViewZ"                          : (TYPE_FLOAT32,    COMPONENT_R,    0),
    "NrdMotion"                         : (TYPE_FLOAT16,    COMPONENT_RGBA, 0),
    "NrdBaseColorMetalness"             : (TYPE_UNORM8,     COMPONENT_RGBA, 0),
    "NrdValidation"                     : (TYPE_UNORM8,     COMPONENT_RGBA, 0),

    # DLSS-RR demodulation: the combined modulation factor M (ro_d+ro_s times
    # guideMod, floored) that CmNoisyCompose divided out of RR's input.
    # Re-multiplied at output res by CmRrPostExposure -- the crisp albedo/
    # texel content rides HERE instead of through the network. Bilinear
    # sampler flag so filter modes 0/1 can textureLod it.
    "RrDemodFactor"                     : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_BILINEAR_SAMPLER),

    # DLSS-RR: pInTransparencyLayer -- premultiplied RGBA16F, render res,
    # composited by NGX AFTER denoise+upscale. Every translucent sprite in the
    # game is RASTERIZED (rt_draw.cpp: "A translucent sprite is RASTERIZED by
    # RTGL1 -- never in the BLAS"); without this layer they are baked into RR's
    # colour input while every guide (albedo, normal, depth, MV) describes the
    # opaque wall BEHIND them, so the network treats them as noise to remove.
    # RGBA16F because alpha carries the NGX occlusion of the denoised
    # background, which RGB-only packed formats cannot. Same size/aspect as
    # Final so the world raster pass can render into it unchanged.
    "RrTransparency"                    : (TYPE_FLOAT16,    COMPONENT_RGBA, FRAMEBUF_FLAGS_IS_ATTACHMENT),
}

if GRADIENT_ESTIMATION_ENABLED:
    FRAMEBUFFERS.update({
        "GradientInputs"                : (TYPE_FLOAT16,    COMPONENT_RG,   FRAMEBUF_FLAGS_STORE_PREV),
        "DISPingGradient"               : (TYPE_UNORM8,     COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_1_3),
        "DISPongGradient"               : (TYPE_UNORM8,     COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_1_3),
        "DISGradientHistory"            : (TYPE_UNORM8,     COMPONENT_RGBA, FRAMEBUF_FLAGS_FORCE_SIZE_1_3),
        "GradientPrevPix"               : (TYPE_UINT8,      COMPONENT_R,    FRAMEBUF_FLAGS_FORCE_SIZE_1_3),
    })


# ---
# User defined structs END
# ---








def getAllConstDefs(constDict):
    return "\n".join([
        "#define %s (%s)" % (name, str(value))
        for name, value in constDict.items()
    ]) + "\n\n"


def getMemberSizeStd430(baseType, dim, count):
    if dim == 1:
        return GLSL_TYPE_SIZES_STD_430[baseType] * count
    elif count == 1:
        return GLSL_TYPE_SIZES_STD_430[(baseType, dim)]
    else:
        return GLSL_TYPE_SIZES_STD_430[baseType] * dim * count


def getMemberActualSize(baseType, dim, count):
    if dim == 1:
        return TYPE_ACTUAL_SIZES[baseType] * count
    else:
        return TYPE_ACTUAL_SIZES[(baseType, dim)] * count

CURRENT_PAD_INDEX = 0

def getPadsForStruct(typeNames, uint32ToAdd):
    global CURRENT_PAD_INDEX
    r = ""
    padStr = "    " + typeNames[TYPE_UINT32] + " __pad%d;\n"
    for i in range(uint32ToAdd):
        r += padStr % (CURRENT_PAD_INDEX + i)
    CURRENT_PAD_INDEX += uint32ToAdd
    return r


# useVecMatTypes:
def getStruct(name, definition, typeNames, alignmentType, breakType):
    r = "struct " + name + "\n{\n"

    if alignmentType == STRUCT_ALIGNMENT_STD140 and typeNames == C_TYPE_NAMES:
        print("Struct \"" + name + "\" is using std140, alignment must be set manually.")

    global CURRENT_PAD_INDEX
    CURRENT_PAD_INDEX = 0

    curSize = 0
    curOffset = 0

    for baseType, dim, mname, count in definition:
        assert(count > 0)
        r += "    "

        if count == 1:
            if dim == 1:
                r +="%s %s" % (typeNames[baseType], mname)
            elif (baseType, dim) in typeNames:
                r += "%s %s" % (typeNames[(baseType, dim)], mname)
            elif dim <= 4:
                r += "%s %s[%d]" % (typeNames[baseType], mname, dim)
            else:
                if USE_MULTIDIMENSIONAL_ARRAYS_IN_C:
                    r += "%s %s[%d][%d]" % (typeNames[baseType], mname, dim // 10, dim % 10)
                else:
                    r += "%s %s[%d]" % (typeNames[baseType], mname, (dim // 10) * (dim % 10))
        else:
            #if dim > 4 and typeNames == C_TYPE_NAMES:
            #    raise Exception("If count > 1, dimensions must be in [1..4]")
            if breakType == STRUCT_BREAK_TYPE_COMPLEX or (typeNames == C_TYPE_NAMES and breakType == STRUCT_BREAK_TYPE_ONLY_C):
                if dim <= 4:
                    r += "%s %s[%d]" % (typeNames[baseType], mname, align4(count * dim))
                else:
                    r += "%s %s[%d]" % (typeNames[baseType], mname, align4(count * int(TYPE_ACTUAL_SIZES[(baseType, dim)] / 4)))
            else:
                if dim == 1:
                    r += "%s %s[%d]" % (typeNames[baseType], mname, count)
                elif (baseType, dim) in typeNames:
                    r += "%s %s[%d]" % (typeNames[(baseType, dim)], mname, count)
                else:
                    r += "%s %s[%d][%d]" % (typeNames[baseType], mname, count, dim)

        r += ";\n"

        if (alignmentType == STRUCT_ALIGNMENT_STD430):
            for i in range(count):
                memberSize = getMemberActualSize(baseType, dim, 1)
                memberAlignment = getMemberSizeStd430(baseType, dim, 1)

                alignedOffset = align(curOffset, memberAlignment)
                diff = alignedOffset - curOffset

                if diff > 0:
                    assert (diff) % 4 == 0
                    r += getPadsForStruct(typeNames, diff // 4)

                curSize += curOffset + memberSize

    if (alignmentType == STRUCT_ALIGNMENT_STD430) and curSize % 16 != 0:
        if (curSize % 16) % 4 != 0:
            raise Exception("Size of struct %s is not 4-byte aligned!" % name)
        uint32ToAdd = (align(curSize, 16) - curSize) // 4
        r += getPadsForStruct(typeNames, uint32ToAdd)

    r += "};\n"
    return r


def getAllStructDefs(typeNames):
    return "\n".join(
        getStruct(name, structDef, typeNames, alignmentType, breakType)
        for name, (structDef, onlyForGLSL, alignmentType, breakType) in STRUCTS.items()
        if not (onlyForGLSL and (typeNames == C_TYPE_NAMES))
    ) + "\n"


def capitalizeFirstLetter(s):
    return s[:1].upper() + s[1:]



CURRENT_FRAMEBUF_BINDING_COUNT = 0

def getGLSLFramebufDeclaration(name, baseFormat, components, flags):
    global CURRENT_FRAMEBUF_BINDING_COUNT

    binding = FRAMEBUF_BASE_BINDING + CURRENT_FRAMEBUF_BINDING_COUNT
    bindingSampler = binding + 1
    CURRENT_FRAMEBUF_BINDING_COUNT += 1

    r = ""
    if flags & FRAMEBUF_FLAGS_IS_ATTACHMENT:
        r += "#ifndef " + FRAMEBUF_IGNORE_ATTACHMENTS_DEFINE + "\n"

    template = ("layout(set = %s, binding = %d, %s) uniform %s %s;")

    r += template % (FRAMEBUF_DESC_SET_NAME, binding, 
        GLSL_IMAGE_FORMATS[(baseFormat, components)], 
        GLSL_IMAGE_2D_TYPE[baseFormat], name)

    if flags & FRAMEBUF_FLAGS_STORE_PREV:
        r += "\n"
        r += getGLSLFramebufDeclaration(
            name + FRAMEBUF_STORE_PREV_POSTFIX, baseFormat, components, 
            flags & ~FRAMEBUF_FLAGS_STORE_PREV)

    if flags & FRAMEBUF_FLAGS_IS_ATTACHMENT:
        r += "\n#endif"

    return r


def getGLSLFramebufSamplerDeclaration(name, baseFormat, components, flags):
    global CURRENT_FRAMEBUF_BINDING_COUNT

    binding = FRAMEBUF_BASE_BINDING + CURRENT_FRAMEBUF_BINDING_COUNT - 1
    bindingSampler = binding + 1
    CURRENT_FRAMEBUF_BINDING_COUNT += 1

    r = ""
    if flags & FRAMEBUF_FLAGS_IS_ATTACHMENT:
        r += "#ifndef " + FRAMEBUF_IGNORE_ATTACHMENTS_DEFINE + "\n"

    templateSampler = ("layout(set = %s, binding = %d) uniform %s %s;")

    r += templateSampler % (FRAMEBUF_DESC_SET_NAME, bindingSampler,
        GLSL_SAMPLER_2D_TYPE[baseFormat], name + FRAMEBUF_SAMPLER_POSTFIX)

    if flags & FRAMEBUF_FLAGS_STORE_PREV:
        r += "\n"
        r += getGLSLFramebufSamplerDeclaration(
            name + FRAMEBUF_STORE_PREV_POSTFIX, baseFormat, components, 
            flags & ~FRAMEBUF_FLAGS_STORE_PREV)

    if flags & FRAMEBUF_FLAGS_IS_ATTACHMENT:
        r += "\n#endif"

    return r


def getGLSLFramebufPackUnpackE5(name, withPrev):
    templateImgStore = ("void imageStore%s(const ivec2 pix, const vec3 unpacked) "
                        "{ imageStore(%s, pix, uvec4(encodeE5B9G9R9(unpacked))); }")
    templateTxlFetch = ("vec3 texelFetch%s(const ivec2 pix)"
                        "{ return decodeE5B9G9R9(texelFetch(%s, pix, 0).r); }")
    r  = templateImgStore % (name, FRAMEBUF_PREFIX + name) + "\n"
    r += templateTxlFetch % (name, FRAMEBUF_PREFIX + name + FRAMEBUF_SAMPLER_POSTFIX) + "\n"
    if withPrev:
        r += templateTxlFetch % (name + FRAMEBUF_STORE_PREV_POSTFIX, FRAMEBUF_PREFIX + name + FRAMEBUF_STORE_PREV_POSTFIX + FRAMEBUF_SAMPLER_POSTFIX) + "\n"
    return r
    

def getAllGLSLFramebufDeclarations():
    global CURRENT_FRAMEBUF_BINDING_COUNT
    CURRENT_FRAMEBUF_BINDING_COUNT = 0
    return "#ifdef " + FRAMEBUF_DESC_SET_NAME \
        \
        + "\n\n// framebuffer indices\n" \
        \
        + "\n".join(
        "#define FB_IMAGE_INDEX_%s %d" % (s, d) for (s, d) in getAllFramebufEnumTuples()
        ) \
        \
        + "\n\n// framebuffers\n" \
        \
        + "\n".join(
            getGLSLFramebufDeclaration(FRAMEBUF_PREFIX + name, baseFormat, components, flags)
            for name, (baseFormat, components, flags) in FRAMEBUFFERS.items()
        ) \
        \
        + "\n\n// samplers\n" \
        + "\n".join(
            getGLSLFramebufSamplerDeclaration(FRAMEBUF_PREFIX + name, baseFormat, components, flags)
            for name, (baseFormat, components, flags) in FRAMEBUFFERS.items()
            if not (flags & FRAMEBUF_FLAGS_NO_SAMPLER)
        ) \
        \
        + "\n\n// pack/unpack formats\n" \
        + "\n".join(
            getGLSLFramebufPackUnpackE5(name, flags & FRAMEBUF_FLAGS_STORE_PREV)
            for name, (baseFormat, components, flags) in FRAMEBUFFERS.items()
            if baseFormat == TYPE_PACK_E5 and not (flags & FRAMEBUF_FLAGS_NO_SAMPLER)
        ) \
        \
        + "\n\n#endif\n"


def removeCoupledDuplicateChars(str, charToRemove = '_'):
    r = ""
    for i in range(0, len(str)):
        if i == 0 or str[i] != str[i - 1] or str[i] != charToRemove:
            r += str[i]
    return r


# make all letters capital and insert "_" before 
# capital letters in the original string
def capitalizeForEnum(s):
    return removeCoupledDuplicateChars("_".join(filter(None, re.split("([A-Z][^A-Z]*)", s))).upper())


# returns (name, index) tuples for framebuf-s
def getAllFramebufEnumTuples():
    names = []
    for name, (_, _, flags) in FRAMEBUFFERS.items():
        names.append(name)
        if flags & FRAMEBUF_FLAGS_STORE_PREV:
            names.append(name + FRAMEBUF_STORE_PREV_POSTFIX)

    return [(capitalizeForEnum(names[i]), i) for i in range(len(names))]


def getAllFramebufConstants():
    fbConst = "#define " + FRAMEBUF_SAMPLER_INVALID_BINDING + " 0xFFFFFFFF\n\n"

    fbEnum = "enum FramebufferImageIndex\n{\n" + "\n".join(
        "    FB_IMAGE_INDEX_%s = %d," % (s, d) for (s, d) in getAllFramebufEnumTuples()
    ) + "\n};\n\n"

    fbFlags = "enum FramebufferImageFlagBits\n{\n" + "\n".join(
        "    FB_IMAGE_FLAGS_%s = %d," % (flName, flValue)
        for (flName, flValue) in FRAMEBUF_FLAGS_ENUM.items()
    ) + "\n};\ntypedef uint32_t FramebufferImageFlags;\n\n"

    return fbConst + fbEnum + fbFlags


def getPublicFlags(flags):
    r = " | ".join(
        "RTGL1::FB_IMAGE_FLAGS_" + flName
        for (flName, flValue) in FRAMEBUF_FLAGS_ENUM.items()
        if flValue & flags
    )
    if r == "":
        return "0"
    else:
        return r


_ShFramebuffers_Count = 0


def getAllVulkanFramebufDeclarations():
    return ("constexpr uint32_t ShFramebuffers_Count = %s;\n"
            "extern const VkFormat ShFramebuffers_Formats[];\n"
            "extern const FramebufferImageFlags ShFramebuffers_Flags[];\n"
            "extern const uint32_t ShFramebuffers_Bindings[];\n"
            "extern const uint32_t ShFramebuffers_BindingsSwapped[];\n"
            "extern const uint32_t ShFramebuffers_Sampler_Bindings[];\n"
            "extern const uint32_t ShFramebuffers_Sampler_BindingsSwapped[];\n"
            "extern const char *const ShFramebuffers_DebugNames[];\n"
            "extern const wchar_t *const ShFramebuffers_DebugNamesW[];\n\n") % str(_ShFramebuffers_Count)


def getAllVulkanFramebufDefinitions():
    template = ("const VkFormat RTGL1::ShFramebuffers_Formats[] = \n{\n%s};\n\n"
                "const RTGL1::FramebufferImageFlags RTGL1::ShFramebuffers_Flags[] = \n{\n%s};\n\n"
                "const uint32_t RTGL1::ShFramebuffers_Bindings[] = \n{\n%s};\n\n"
                "const uint32_t RTGL1::ShFramebuffers_BindingsSwapped[] = \n{\n%s};\n\n"
                "const uint32_t RTGL1::ShFramebuffers_Sampler_Bindings[] = \n{\n%s};\n\n"
                "const uint32_t RTGL1::ShFramebuffers_Sampler_BindingsSwapped[] = \n{\n%s};\n\n"
                "const char *const RTGL1::ShFramebuffers_DebugNames[] = \n{\n%s};\n\n"
                "const wchar_t *const RTGL1::ShFramebuffers_DebugNamesW[] = \n{\n%s};\n\n")
    TAB_STR = "    "
    formats = ""
    count = 0
    publicFlags = ""
    samplerCount = 0
    bindings = ""    
    bindingsSwapped = ""
    samplerBindings = ""
    samplerBindingsSwapped = ""
    names = ""
    for name, (baseFormat, components, flags) in FRAMEBUFFERS.items():
        formats += TAB_STR + VULKAN_IMAGE_FORMATS[(baseFormat, components)] + ", // " + name + "\n"
        names += TAB_STR + "\"" + FRAMEBUF_DEBUG_NAME_PREFIX + name + "\",\n"
        publicFlags += TAB_STR + getPublicFlags(flags) + ", // " + name + "\n"

        if not flags & FRAMEBUF_FLAGS_STORE_PREV:
            bindings                += TAB_STR + str(count)         + ",\n"
            bindingsSwapped         += TAB_STR + str(count)         + ",\n"
        else:
            bindings                += TAB_STR + str(count)         + ",\n"
            bindings                += TAB_STR + str(count + 1)     + ",\n"
            bindingsSwapped         += TAB_STR + str(count + 1)     + ",\n"
            bindingsSwapped         += TAB_STR + str(count)         + ",\n"
            
            formats += TAB_STR + VULKAN_IMAGE_FORMATS[(baseFormat, components)] + ", // " + name + FRAMEBUF_STORE_PREV_POSTFIX + "\n"
            names += TAB_STR + "\"" + FRAMEBUF_DEBUG_NAME_PREFIX + name + FRAMEBUF_STORE_PREV_POSTFIX + "\",\n"
            publicFlags += TAB_STR + getPublicFlags(flags) + ", // " + name + FRAMEBUF_STORE_PREV_POSTFIX + "\n"
            count += 1

        count += 1

    for name, (baseFormat, components, flags) in FRAMEBUFFERS.items():
        bindingIndex     = str(count + samplerCount)
        bindingIndexNext = str(count + samplerCount + 1)
        
        if flags & FRAMEBUF_FLAGS_NO_SAMPLER:
            bindingIndex = bindingIndexNext = FRAMEBUF_SAMPLER_INVALID_BINDING

        if not flags & FRAMEBUF_FLAGS_STORE_PREV:
            samplerBindings         += TAB_STR + bindingIndex       + ",\n"
            samplerBindingsSwapped  += TAB_STR + bindingIndex       + ",\n"
        else:
            samplerBindings         += TAB_STR + bindingIndex       + ",\n"
            samplerBindings         += TAB_STR + bindingIndexNext   + ",\n"
            samplerBindingsSwapped  += TAB_STR + bindingIndexNext   + ",\n"
            samplerBindingsSwapped  += TAB_STR + bindingIndex       + ",\n"
            samplerCount += 1

        samplerCount += 1

    wnames = names.replace(f"\"{FRAMEBUF_DEBUG_NAME_PREFIX}", f"L\"{FRAMEBUF_DEBUG_NAME_PREFIX}")

    global _ShFramebuffers_Count
    _ShFramebuffers_Count = count

    return template % (formats, publicFlags, bindings, bindingsSwapped, samplerBindings, samplerBindingsSwapped, names, wnames)


FILE_HEADER = "// This file was generated by GenerateShaderCommon.py\n\n"


def writeToC(commonHeaderFile, fbHeaderFile, fbSourceFile):
    commonHeaderFile.write(FILE_HEADER)
    commonHeaderFile.write("#pragma once\n\n")
    commonHeaderFile.write("namespace RTGL1\n{\n\n")
    commonHeaderFile.write("#include <stdint.h>\n\n")
    commonHeaderFile.write(getAllConstDefs(CONST))
    commonHeaderFile.write(getAllStructDefs(C_TYPE_NAMES))
    commonHeaderFile.write("}")

    fbSourceFile.write(FILE_HEADER)
    fbSourceFile.write("#include \"%s\"\n\n" % os.path.basename(fbHeaderFile.name))
    fbSourceFile.write(getAllVulkanFramebufDefinitions())

    fbHeaderFile.write(FILE_HEADER)
    fbHeaderFile.write("#pragma once\n\n")
    fbHeaderFile.write("#include \"../Common.h\"\n\n")
    fbHeaderFile.write("namespace RTGL1\n{\n\n")
    fbHeaderFile.write(getAllFramebufConstants())
    fbHeaderFile.write(getAllVulkanFramebufDeclarations())
    fbHeaderFile.write("}")


def writeToGLSL(f):
    f.write(FILE_HEADER)
    f.write(getAllConstDefs(CONST))
    f.write(getAllConstDefs(CONST_GLSL_ONLY))
    f.write(getAllStructDefs(GLSL_TYPE_NAMES))
    f.write(getAllGLSLFramebufDeclarations())


def main():
    basePath = ""

    for i in range(len(sys.argv)):
        if "--help" == sys.argv[i] or "-help" == sys.argv[i]:
            print("--path     : specify path to target folder in the next argument")
            return
        if "--path" == sys.argv[i]:
            if i + 1 < len(sys.argv):
                basePath = sys.argv[i + 1]
                if not os.path.exists(basePath):
                    print("Folder with path \"" + basePath + "\" doesn't exist.")
                    return
            else:
                print("--path expects folder path in the next argument.")
                return

    evalConst()
    # with open('ShaderConfig.csv', newline='') as csvfile:
    with open(basePath + "ShaderCommonC.h", "w") as commonHeaderFile:
        with open(basePath + "ShaderCommonCFramebuf.h", "w") as fbHeaderFile:
            with open(basePath + "ShaderCommonCFramebuf.cpp", "w") as fbSourceFile:
                writeToC(commonHeaderFile, fbHeaderFile, fbSourceFile)
    with open(basePath + "ShaderCommonGLSL.h", "w") as f:
        writeToGLSL(f)

# main
if __name__ == "__main__":
    main()
