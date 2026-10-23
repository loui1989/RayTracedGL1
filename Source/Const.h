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

#include <string_view>
#include <cstdint>

namespace RTGL1
{

constexpr uint32_t ALLOCATOR_BLOCK_SIZE_STAGING_TEXTURES = 64 * 512 * 512 * 4;
constexpr uint32_t ALLOCATOR_BLOCK_SIZE_TEXTURES         = 64 * 512 * 512 * 4;

constexpr uint32_t TEXTURE_FILE_PATH_MAX_LENGTH      = 512;
constexpr uint32_t TEXTURE_FILE_NAME_MAX_LENGTH      = 256;
constexpr uint32_t TEXTURE_FILE_EXTENSION_MAX_LENGTH = 16;

// Doom64-RT: raised from 4096. This is the DESIRED ceiling -- the value actually
// used is clamped to what the GPU reports, see InitTextureCountMax() in
// TextureManager.cpp.
//
// 4096 was not enough once gzdoom-rt started precaching a level's actors: one
// material claims up to TEXTURES_PER_MATERIAL_COUNT (5) entries, so ~2100
// materials exhausted the array, 487 textures were dropped, and because the
// overflow is a Warning rather than an error the game carried on and drew its
// HUD as untextured blocks.
//
// It is a SESSION budget, not a per-level one: nothing is ever freed while a
// level changes, so a long session could reach the old ceiling with no
// precaching at all. Nothing needed regenerating for this -- the shaders declare
// `uniform sampler2D globalTextures[]` unbounded and index it with nonuniformEXT,
// and every index in ShaderCommonC.h is a full uint32_t with no bit packing.
constexpr uint32_t TEXTURE_COUNT_MAX           = 16384;
constexpr uint32_t EMPTY_TEXTURE_INDEX         = 0;
constexpr uint32_t MATERIALS_MAX_LAYER_COUNT   = 4;
constexpr uint32_t TEXTURES_PER_MATERIAL_COUNT = 5;

constexpr const char* TEXTURE_ALBEDO_ALPHA_POSTFIX                 = "";
constexpr const char* TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_POSTFIX = "_orm";
constexpr const char* TEXTURE_NORMAL_POSTFIX                       = "_n";
constexpr const char* TEXTURE_EMISSIVE_POSTFIX                     = "_e";
constexpr const char* TEXTURE_HEIGHT_POSTFIX                       = "_h";

constexpr uint32_t TEXTURE_ALBEDO_ALPHA_INDEX                 = 0;
constexpr uint32_t TEXTURE_OCCLUSION_ROUGHNESS_METALLIC_INDEX = 1;
constexpr uint32_t TEXTURE_NORMAL_INDEX                       = 2;
constexpr uint32_t TEXTURE_EMISSIVE_INDEX                     = 3;
constexpr uint32_t TEXTURE_HEIGHT_INDEX                       = 4;

constexpr uint32_t MAX_PREGENERATED_MIPMAP_LEVELS = 20;

constexpr float MESH_TRANSLUCENT_ALPHA_THRESHOLD = 0.98f;

#define RTGL1_MAIN_ROOT_NODE "rtgl1_main_root"

constexpr std::string_view TEXTURES_FOLDER           = "mat";
constexpr std::string_view TEXTURES_FOLDER_DEV       = "mat_dev";
constexpr std::string_view TEXTURES_FOLDER_ORIGINALS = "mat_src";
constexpr std::string_view SCENES_FOLDER             = "scenes";
constexpr std::string_view REPLACEMENTS_FOLDER       = "replace";
constexpr std::string_view SHADERS_FOLDER            = "shaders";
constexpr std::string_view DATABASE_FOLDER           = "data";

constexpr std::string_view TEXTURES_FOLDER_JUNCTION        = "mat_junction";
constexpr std::wstring_view TEXTURES_FOLDER_JUNCTION_W     = L"mat_junction";
constexpr std::string_view TEXTURES_FOLDER_JUNCTION_PREFIX = "mat_junction/";
constexpr std::string_view TEXTURES_FOLDER_EXTERNAL        = "ext";

constexpr std::wstring_view SCENE_PATCH_SUFFIX = L"_patch";

constexpr const char* MATERIAL_NAME_SCENEBUILDINGWARNING = "_rgscenewarn";

}