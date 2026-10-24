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

#include "LibraryConfig.h"

#include <RTGL1/RTGL1.h>

#include <array>
#include <filesystem>
#include <string>
#include <optional>
#include <set>
#include <vector>

namespace RTGL1
{

struct TextureMeta
{
    constexpr static int Version{ 0 };
    constexpr static int RequiredVersion{ 0 };

    std::string textureName = {};

    bool forceIgnore             = false;
    bool forceIgnoreIfRasterized = false;

    bool forceAlphaTest   = false;
    bool forceTranslucent = false;
    bool forceOpaque      = false;

    bool forceGenerateNormals = false;
    bool forceExactNormals    = false;

    bool isMirror             = false;
    bool isWater              = false;
    bool isWaterIfTranslucent = false;
    bool isGlass              = false;
    bool isGlassIfTranslucent = false;
    bool isAcid               = false;

    bool isGlassIfSmooth  = false;
    bool isMirrorIfSmooth = false;
    // RG_MESH_PRIMITIVE_POLISHED
    bool isPolished       = false;

    bool isThinMedia = false;

    float metallicDefault  = 0.0f;
    float roughnessDefault = 1.0f;
    float emissiveMult     = 0.0f;

    float                    attachedLightIntensity     = 0.0f;
    std::array< uint8_t, 3 > attachedLightColor         = { { 255, 255, 255 } };
    char                     attachedLightColorHEX[ 7 ] = "FFFFFF";
    bool                     attachedLightEvenOnDynamic = false;

    bool noShadow = false;
};

struct TextureMetaArray
{
    constexpr static int Version{ 0 };
    constexpr static int RequiredVersion{ 0 };

    std::vector< TextureMeta > array;
};



struct SceneMeta
{
    constexpr static int Version{ 0 };
    constexpr static int RequiredVersion{ 0 };

    std::string sceneName = {};

    std::optional< float > sky;
    std::optional< std::array< float, 3 > > forceSkyPlainColor;

    std::optional< float > scatter;
    std::optional< float > volumeFar;
    std::optional< float > volumeAssymetry;
    std::optional< float > volumeLightMultiplier;

    std::optional< std::array< float, 3 > > volumeAmbient;
    std::optional< std::array< float, 3 > > volumeUnderwaterColor;

    std::set< std::string > ignoredReplacements;
};

struct SceneMetaArray
{
    constexpr static int Version{ 0 };
    constexpr static int RequiredVersion{ 0 };

    std::vector< SceneMeta > array;
};



struct PrimitiveExtraInfo
{
    int isGlass         = 0;
    int isMirror        = 0;
    int isWater         = 0;
    int isSkyVisibility = 0;
    int isAcid          = 0;
    int isThinMedia     = 0;
    int noShadow        = 0;
};

// Persisted RTGL Dev window state (rt/devmode_settings.json). POD mirror of Devmode.
struct DevmodeSettings
{
    constexpr static int Version{ 1 };
    constexpr static int RequiredVersion{ 1 };

    int version = Version;

    float fontGlobalScale = 1.f;

    bool debugWindowOnTop = false;
    bool antiFirefly      = true;

    bool  rrTemporalPrefilter       = false;
    bool  rrTemporalPrefilterSticky = false;
    bool  illumSensSticky          = false;
    float illumSensDirect          = 1.f;
    float illumSensIndirect        = 0.75f;
    float illumSensSpec            = 1.f;
    bool  rayReconstruction        = false;
    bool  rayReconstructionSticky  = false;

    bool materialStripNormals    = false;
    bool materialStripMetallic   = false;
    bool materialStripRoughness  = false;
    bool materialStripHeight     = false;
    bool materialStripEmissives  = false;
    float roughnessTowardMatte   = 0.f;
    // Legacy combined toggles (migrate on load).
    bool materialStripPbrMaps    = false;
    bool materialStripOrm        = false;

    // drawInfoOvrd
    bool  ovrd_enable                          = false;
    int   ovrd_maxBounceShadows                = 0;
    int   ovrd_indirectBounces                 = 2;
    bool  ovrd_indirectLegacyBounceWeight      = true;
    float ovrd_directDiffuseSensitivityToChange   = 1.f;
    float ovrd_indirectDiffuseSensitivityToChange = 1.f;
    float ovrd_specularSensitivityToChange        = 1.f;
    bool  ovrd_disableEyeAdaptation            = false;
    float ovrd_ev100Min                        = 0.f;
    float ovrd_ev100Max                        = 0.f;
    std::array< float, 3 > ovrd_saturation     = { { 1.f, 1.f, 1.f } };
    std::array< float, 3 > ovrd_crosstalk      = { { 0.f, 0.f, 0.f } };
    bool  ovrd_vsync                           = false;
    int   ovrd_frameGeneration                 = 0; // RgFrameGenerationMode
    bool  ovrd_preferDxgiPresent               = true;
    bool  ovrd_hdr                             = false;
    int   ovrd_upscaleTechnique                = 3; // NVIDIA_DLSS typical
    int   ovrd_sharpenTechnique                = 0;
    int   ovrd_resolutionMode                  = 2; // BALANCED
    float ovrd_customRenderSizeScale           = 1.f;
    bool  ovrd_pixelizedEnable                 = false;
    int   ovrd_pixelizedHeight                 = 480;
    bool  ovrd_rayReconstruction               = false;
    float ovrd_normalMapStrength               = 1.f;
    float ovrd_heightMapDepth                  = 1.f;
    float ovrd_emissionMapBoost                = 1.f;
    float ovrd_emissionMaxScreenColor          = 1.f;
    float ovrd_lightmapScreenCoverage          = 0.f;
    bool  ovrd_fluidEnabled                    = false;
    std::array< float, 3 > ovrd_fluidGravity   = { { 0.f, 0.f, -14.f } };
    bool  ovrd_allowMapAutoExport              = false;

    // cameraOvrd (non-transient)
    bool  cam_fovEnable     = false;
    float cam_fovDeg        = 90.f;
    bool  cam_customEnable  = false;
    std::array< float, 3 > cam_customPos    = { { 0.f, 0.f, 0.f } };
    std::array< float, 2 > cam_customAngles = { { 0.f, 0.f } };

    bool ignoreExternalGeometry           = false;
    bool allowExportOfExistingReplacements = false;
    bool materialsTableEnable             = false;
    int  primitivesTableMode              = 0;
    bool breakOnTexturePrimitive          = false;
    bool breakOnTextureImage              = false;
    std::string breakOnTexture            = {};
    uint32_t logFlags =
        RG_MESSAGE_SEVERITY_VERBOSE | RG_MESSAGE_SEVERITY_INFO | RG_MESSAGE_SEVERITY_WARNING |
        RG_MESSAGE_SEVERITY_ERROR;
    bool logAutoScroll = true;
};



struct CameraExtraInfo
{
    struct FovAnimFrame
    {
        int   frame24{ 0 };
        float fovDegrees{ 0 };
    };

    static constexpr uint32_t LatestVersion = 0;

    uint32_t                    version{ LatestVersion };
    std::vector< int >          anim_cuts_24fps{};
    std::vector< FovAnimFrame > anim_fov_24fps{};
};



namespace json_parser
{
    namespace detail
    {
        auto ReadTextureMetaArray( const std::filesystem::path& path )
            -> std::optional< TextureMetaArray >;

        auto ReadSceneMetaArray( const std::filesystem::path& path )
            -> std::optional< SceneMetaArray >;

        auto ReadLibraryConfig( const std::filesystem::path& path )
            -> std::optional< LibraryConfig >;

        auto ReadDevmodeSettings( const std::filesystem::path& path )
            -> std::optional< DevmodeSettings >;

        bool WriteDevmodeSettings( const std::filesystem::path& path,
                                   const DevmodeSettings&       settings );

        auto ReadLightExtraInfo( const std::string_view& data )
            -> std::optional< RgLightAdditionalEXT >;

        auto ReadPrimitiveExtraInfo( const std::string_view& data ) -> PrimitiveExtraInfo;

        auto ReadCameraExtraInfo( const std::string_view& data ) -> CameraExtraInfo;
    }

    // clang-format off
    template< typename T > auto ReadFileAs( const std::filesystem::path& path ) = delete;
    template<> inline auto ReadFileAs< TextureMetaArray >( const std::filesystem::path& path ) { return detail::ReadTextureMetaArray( path ); }
    template<> inline auto ReadFileAs< SceneMetaArray   >( const std::filesystem::path& path ) { return detail::ReadSceneMetaArray( path ); }
    template<> inline auto ReadFileAs< LibraryConfig    >( const std::filesystem::path& path ) { return detail::ReadLibraryConfig( path ); }
    template<> inline auto ReadFileAs< DevmodeSettings  >( const std::filesystem::path& path ) { return detail::ReadDevmodeSettings( path ); }

    template< typename T > auto ReadStringAs( const std::string_view& str ) = delete;
    template<> inline auto ReadStringAs< RgLightAdditionalEXT >( const std::string_view& data ) { return detail::ReadLightExtraInfo( data ); }
    template<> inline auto ReadStringAs< PrimitiveExtraInfo   >( const std::string_view& data ) { return detail::ReadPrimitiveExtraInfo( data ); }
    template<> inline auto ReadStringAs< CameraExtraInfo   >( const std::string_view& data ) { return detail::ReadCameraExtraInfo( data ); }
    // clang-format on

    std::string MakeJsonString( const RgLightAdditionalEXT& info );
    std::string MakeJsonString( const PrimitiveExtraInfo& info );

    inline bool WriteFileAs( const std::filesystem::path& path, const DevmodeSettings& settings )
    {
        return detail::WriteDevmodeSettings( path, settings );
    }
}

}
