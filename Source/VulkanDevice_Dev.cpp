// Copyright (c) 2022 Sultim Tsyrendashiev
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

#include "VulkanDevice.h"

#include "Matrix.h"
#include "JsonParser.h"

#include "Generated/ShaderCommonC.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <chrono>
#include <cstring>
#include <ranges>

namespace
{

template< typename To, typename From >
To ClampPix( From v )
{
    return std::clamp( To( v ), To( 96 ), To( 3840 ) );
}

constexpr const char* kDevmodeSettingsFile = "devmode_settings.json";
constexpr double      kDevmodeSaveDebounceSec = 2.0;

auto DevmodeSettingsPath( const std::filesystem::path& ovrdFolder ) -> std::filesystem::path
{
    return ovrdFolder / kDevmodeSettingsFile;
}

auto CaptureDevmodeSettings( const RTGL1::Devmode& d ) -> RTGL1::DevmodeSettings
{
    using DS = RTGL1::DevmodeSettings;
    DS s{};
    s.version                      = DS::Version;
    s.fontGlobalScale              = d.fontGlobalScale;
    s.debugWindowOnTop             = d.debugWindowOnTop;
    s.antiFirefly                  = d.antiFirefly;
    s.rrTemporalPrefilter           = d.rrTemporalPrefilter;
    s.rrTemporalPrefilterSticky     = d.rrTemporalPrefilterSticky;
    s.illumSensSticky              = d.illumSensSticky;
    s.illumSensDirect              = d.illumSensDirect;
    s.illumSensIndirect            = d.illumSensIndirect;
    s.illumSensSpec                = d.illumSensSpec;
    s.rayReconstruction            = d.rayReconstruction;
    s.rayReconstructionSticky      = d.rayReconstructionSticky;
    s.materialStripNormals         = d.materialStripNormals;
    s.materialStripMetallic        = d.materialStripMetallic;
    s.materialStripRoughness       = d.materialStripRoughness;
    s.materialStripHeight          = d.materialStripHeight;
    s.materialStripEmissives       = d.materialStripEmissives;
    s.roughnessTowardMatte         = d.roughnessTowardMatte;
    s.materialStripPbrMaps         = false;
    s.materialStripOrm             = false;

    const auto& m = d.drawInfoOvrd;
    s.ovrd_enable                        = m.enable;
    s.ovrd_maxBounceShadows              = m.maxBounceShadows;
    s.ovrd_indirectBounces               = m.indirectBounces;
    s.ovrd_indirectLegacyBounceWeight    = m.indirectLegacyBounceWeight;
    s.ovrd_directDiffuseSensitivityToChange   = m.directDiffuseSensitivityToChange;
    s.ovrd_indirectDiffuseSensitivityToChange = m.indirectDiffuseSensitivityToChange;
    s.ovrd_specularSensitivityToChange        = m.specularSensitivityToChange;
    s.ovrd_disableEyeAdaptation          = m.disableEyeAdaptation;
    s.ovrd_ev100Min                      = m.ev100Min;
    s.ovrd_ev100Max                      = m.ev100Max;
    s.ovrd_saturation = { { m.saturation[ 0 ], m.saturation[ 1 ], m.saturation[ 2 ] } };
    s.ovrd_crosstalk  = { { m.crosstalk[ 0 ], m.crosstalk[ 1 ], m.crosstalk[ 2 ] } };
    s.ovrd_vsync                         = m.vsync;
    s.ovrd_frameGeneration               = int( m.frameGeneration );
    s.ovrd_preferDxgiPresent             = m.preferDxgiPresent;
    s.ovrd_hdr                           = m.hdr;
    s.ovrd_upscaleTechnique              = int( m.upscaleTechnique );
    s.ovrd_sharpenTechnique              = int( m.sharpenTechnique );
    s.ovrd_resolutionMode                = int( m.resolutionMode );
    s.ovrd_customRenderSizeScale         = m.customRenderSizeScale;
    s.ovrd_pixelizedEnable               = m.pixelizedEnable;
    s.ovrd_pixelizedHeight               = m.pixelizedHeight;
    s.ovrd_rayReconstruction             = m.rayReconstruction;
    s.ovrd_normalMapStrength             = m.normalMapStrength;
    s.ovrd_heightMapDepth                = m.heightMapDepth;
    s.ovrd_emissionMapBoost              = m.emissionMapBoost;
    s.ovrd_emissionMaxScreenColor        = m.emissionMaxScreenColor;
    s.ovrd_lightmapScreenCoverage        = m.lightmapScreenCoverage;
    s.ovrd_fluidEnabled                  = m.fluidEnabled;
    s.ovrd_fluidGravity = { { m.fluidGravity.data[ 0 ],
                              m.fluidGravity.data[ 1 ],
                              m.fluidGravity.data[ 2 ] } };
    s.ovrd_allowMapAutoExport            = m.allowMapAutoExport;

    s.cam_fovEnable    = d.cameraOvrd.fovEnable;
    s.cam_fovDeg       = d.cameraOvrd.fovDeg;
    s.cam_customEnable = d.cameraOvrd.customEnable;
    s.cam_customPos    = { { d.cameraOvrd.customPos.data[ 0 ],
                          d.cameraOvrd.customPos.data[ 1 ],
                          d.cameraOvrd.customPos.data[ 2 ] } };
    s.cam_customAngles = { { d.cameraOvrd.customAngles.data[ 0 ],
                             d.cameraOvrd.customAngles.data[ 1 ] } };

    s.ignoreExternalGeometry            = d.ignoreExternalGeometry;
    s.allowExportOfExistingReplacements = d.allowExportOfExistingReplacements;
    s.materialsTableEnable              = d.materialsTableEnable;
    s.primitivesTableMode               = int( d.primitivesTableMode );
    s.breakOnTexturePrimitive           = d.breakOnTexturePrimitive;
    s.breakOnTextureImage               = d.breakOnTextureImage;
    s.breakOnTexture                    = d.breakOnTexture;
    s.logFlags                          = d.logFlags;
    s.logAutoScroll                     = d.logAutoScroll;
    return s;
}

void ApplyDevmodeSettings( RTGL1::Devmode& d, const RTGL1::DevmodeSettings& s )
{
    d.fontGlobalScale =
        std::clamp( s.fontGlobalScale, 0.75f, 2.5f );
    d.debugWindowOnTop             = s.debugWindowOnTop;
    d.antiFirefly                  = s.antiFirefly;
    d.rrTemporalPrefilter           = s.rrTemporalPrefilter;
    d.illumSensDirect              = std::clamp( s.illumSensDirect, 0.f, 1.f );
    d.illumSensIndirect            = std::clamp( s.illumSensIndirect, 0.f, 1.f );
    d.illumSensSpec                = std::clamp( s.illumSensSpec, 0.f, 1.f );
    d.rayReconstruction            = s.rayReconstruction;

    // Sticky flags are deliberately NOT restored from disk.
    //
    // These make a Dev-UI knob replace the game's per-frame value, and
    // rayReconstructionSticky does so even when the Override master switch is
    // OFF (Dev_Override, the `else if( devmode->rayReconstructionSticky )`
    // branch). Persisting them meant that touching the RR checkbox once, in any
    // session, silently killed `rt_rayreconstr` in every later launch -- while
    // gzdoom's `rt_rr_status` still reported "RR REQUESTED = YES", because that
    // reads the request *before* this override is applied. That cost several
    // sessions of A/B tests run against A-SVGF while believing they measured
    // DLSS-RR (2026-08-07).
    //
    // The values above still persist, so Dev tuning survives a relaunch; only
    // the switches that make them override the game reset. Same reasoning as
    // forcing rt_rr_reset_hold/_now/_debug to 0 in the launcher: a diagnostic
    // must never outlive the session that enabled it.
    d.rrTemporalPrefilterSticky    = false;
    d.illumSensSticky              = false;
    d.rayReconstructionSticky      = false;
    d.materialStripNormals         = s.materialStripNormals;
    d.materialStripMetallic        = s.materialStripMetallic;
    d.materialStripRoughness       = s.materialStripRoughness;
    d.materialStripHeight          = s.materialStripHeight;
    d.materialStripEmissives       = s.materialStripEmissives;
    d.roughnessTowardMatte         = std::clamp( s.roughnessTowardMatte, 0.f, 1.f );
    // Migrate old combined toggles.
    if( s.materialStripPbrMaps && !s.materialStripNormals && !s.materialStripMetallic &&
        !s.materialStripRoughness && !s.materialStripHeight )
    {
        d.materialStripNormals   = true;
        d.materialStripMetallic  = true;
        d.materialStripRoughness = true;
        d.materialStripHeight    = true;
    }
    if( s.materialStripOrm && !s.materialStripMetallic && !s.materialStripRoughness )
    {
        d.materialStripMetallic  = true;
        d.materialStripRoughness = true;
    }

    auto& m = d.drawInfoOvrd;
    // NOT restored from disk on purpose -- see the sticky-flag note below.
    // The override *values* persist (so tuning survives a relaunch), but the
    // master switch that makes them replace the game's values does not.
    m.enable                        = false;
    m.maxBounceShadows              = s.ovrd_maxBounceShadows;
    m.indirectBounces               = s.ovrd_indirectBounces;
    m.indirectLegacyBounceWeight    = s.ovrd_indirectLegacyBounceWeight;
    m.directDiffuseSensitivityToChange   = s.ovrd_directDiffuseSensitivityToChange;
    m.indirectDiffuseSensitivityToChange = s.ovrd_indirectDiffuseSensitivityToChange;
    m.specularSensitivityToChange        = s.ovrd_specularSensitivityToChange;
    m.disableEyeAdaptation          = s.ovrd_disableEyeAdaptation;
    m.ev100Min                      = s.ovrd_ev100Min;
    m.ev100Max                      = s.ovrd_ev100Max;
    m.saturation[ 0 ] = s.ovrd_saturation[ 0 ];
    m.saturation[ 1 ] = s.ovrd_saturation[ 1 ];
    m.saturation[ 2 ] = s.ovrd_saturation[ 2 ];
    m.crosstalk[ 0 ]  = s.ovrd_crosstalk[ 0 ];
    m.crosstalk[ 1 ]  = s.ovrd_crosstalk[ 1 ];
    m.crosstalk[ 2 ]  = s.ovrd_crosstalk[ 2 ];
    m.vsync                         = s.ovrd_vsync;
    m.frameGeneration = static_cast< RgFrameGenerationMode >( s.ovrd_frameGeneration );
    m.preferDxgiPresent             = s.ovrd_preferDxgiPresent;
    m.hdr                           = s.ovrd_hdr;
    m.upscaleTechnique =
        static_cast< RgRenderUpscaleTechnique >( s.ovrd_upscaleTechnique );
    m.sharpenTechnique =
        static_cast< RgRenderSharpenTechnique >( s.ovrd_sharpenTechnique );
    m.resolutionMode = static_cast< RgRenderResolutionMode >( s.ovrd_resolutionMode );
    m.customRenderSizeScale         = s.ovrd_customRenderSizeScale;
    m.pixelizedEnable               = s.ovrd_pixelizedEnable;
    m.pixelizedHeight               = s.ovrd_pixelizedHeight;
    m.rayReconstruction             = s.ovrd_rayReconstruction;
    m.normalMapStrength             = s.ovrd_normalMapStrength;
    m.heightMapDepth                = s.ovrd_heightMapDepth;
    m.emissionMapBoost              = s.ovrd_emissionMapBoost;
    m.emissionMaxScreenColor        = s.ovrd_emissionMaxScreenColor;
    m.lightmapScreenCoverage        = s.ovrd_lightmapScreenCoverage;
    m.fluidEnabled                  = s.ovrd_fluidEnabled;
    m.fluidGravity = { s.ovrd_fluidGravity[ 0 ],
                       s.ovrd_fluidGravity[ 1 ],
                       s.ovrd_fluidGravity[ 2 ] };
    m.allowMapAutoExport            = s.ovrd_allowMapAutoExport;

    d.cameraOvrd.fovEnable    = s.cam_fovEnable;
    d.cameraOvrd.fovDeg       = s.cam_fovDeg;
    d.cameraOvrd.customEnable = s.cam_customEnable;
    d.cameraOvrd.customPos    = { s.cam_customPos[ 0 ],
                               s.cam_customPos[ 1 ],
                               s.cam_customPos[ 2 ] };
    d.cameraOvrd.customAngles = { s.cam_customAngles[ 0 ], s.cam_customAngles[ 1 ] };

    d.ignoreExternalGeometry            = s.ignoreExternalGeometry;
    d.allowExportOfExistingReplacements = s.allowExportOfExistingReplacements;
    d.materialsTableEnable              = s.materialsTableEnable;
    d.primitivesTableMode =
        static_cast< RTGL1::Devmode::DebugPrimMode >( s.primitivesTableMode );
    d.breakOnTexturePrimitive = s.breakOnTexturePrimitive;
    d.breakOnTextureImage     = s.breakOnTextureImage;
    std::memset( d.breakOnTexture, 0, sizeof( d.breakOnTexture ) );
    std::strncpy( d.breakOnTexture,
                  s.breakOnTexture.c_str(),
                  sizeof( d.breakOnTexture ) - 1 );
    d.logFlags      = s.logFlags;
    d.logAutoScroll = s.logAutoScroll;
}

void MarkDevmodeDirty( RTGL1::Devmode& d )
{
    d.settingsDirty   = true;
    d.settingsDirtyAt = ImGui::GetTime();
}

void MaybeSaveDevmodeSettings( RTGL1::Devmode&                   d,
                               const std::filesystem::path&      ovrdFolder,
                               bool                              force )
{
    if( !d.settingsDirty )
    {
        return;
    }
    if( !force && ( ImGui::GetTime() - d.settingsDirtyAt ) < kDevmodeSaveDebounceSec )
    {
        return;
    }
    if( RTGL1::json_parser::WriteFileAs( DevmodeSettingsPath( ovrdFolder ),
                                         CaptureDevmodeSettings( d ) ) )
    {
        d.settingsDirty = false;
    }
}

void ResetDevmodeToDefaults( RTGL1::Devmode& d )
{
    // Re-value-init by assignment from a fresh instance.
    auto fresh            = RTGL1::Devmode{};
    // Preserve breakOnTexture buffer zeroing via fresh.
    d                     = std::move( fresh );
    d.settingsDirty       = true;
    d.settingsDirtyAt     = ImGui::GetTime();
}

struct WholeWindow
{
    explicit WholeWindow( std::string_view name )
    {
#ifdef IMGUI_HAS_VIEWPORT
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos( viewport->WorkPos );
        ImGui::SetNextWindowSize( viewport->WorkSize );
        ImGui::SetNextWindowViewport( viewport->ID );
#else
        ImGui::SetNextWindowPos( ImVec2( 0.0f, 0.0f ) );
        ImGui::SetNextWindowSize( ImGui::GetIO().DisplaySize );
#endif
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );

        if( ImGui::Begin( name.data(),
                          nullptr,
                          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                              ImGuiWindowFlags_NoBackground ) )
        {
            beginSuccess = ImGui::BeginTabBar( "##TabBar", ImGuiTabBarFlags_Reorderable );
        }
    }

    WholeWindow( const WholeWindow& other )                = delete;
    WholeWindow( WholeWindow&& other ) noexcept            = delete;
    WholeWindow& operator=( const WholeWindow& other )     = delete;
    WholeWindow& operator=( WholeWindow&& other ) noexcept = delete;

    explicit operator bool() const { return beginSuccess; }

    ~WholeWindow()
    {
        if( beginSuccess )
        {
            ImGui::EndTabBar();
        }
        ImGui::End();
        ImGui::PopStyleVar( 1 );
    }

private:
    bool beginSuccess{ false };
};

}

bool RTGL1::VulkanDevice::Dev_IsDevmodeInitialized() const
{
    return debugWindows && devmode;
}

void RTGL1::VulkanDevice::Dev_LoadSettings( const DevmodeSettings& settings )
{
    if( !devmode )
    {
        return;
    }
    ApplyDevmodeSettings( *devmode, settings );
    ImGui::GetIO().FontGlobalScale = std::clamp( devmode->fontGlobalScale, 0.75f, 2.5f );
}

void RTGL1::VulkanDevice::Dev_SaveSettings( bool force ) const
{
    if( !devmode )
    {
        return;
    }
    MaybeSaveDevmodeSettings( *devmode, ovrdFolder, force );
}

namespace
{

template< size_t N >
struct StringLiteral
{
    consteval StringLiteral( const char ( &str )[ N ] ) { std::copy_n( str, N, value ); }
    consteval auto c_str() const { return value; }
    char value[ N ];
};

template< StringLiteral Name >
void imgui_ShowAlwaysOnCheckbox()
{
    ImGui::BeginDisabled( true );
    static bool alwaysOn;
    alwaysOn = true;
    ImGui::Checkbox( Name.c_str(), &alwaysOn );
    ImGui::EndDisabled();
}

}

void RTGL1::VulkanDevice::Dev_Draw() const
{
    if( !Dev_IsDevmodeInitialized() )
    {
        return;
    }

    if( debugWindows->IsMinimized() )
    {
        return;
    }

    // Live font scale (base TTF = 15px).
    ImGui::GetIO().FontGlobalScale = std::clamp( devmode->fontGlobalScale, 0.75f, 2.5f );

    auto w = WholeWindow( "Main window" );
    if( !w )
    {
        return;
    }

    if( ImGui::BeginTabItem( "General" ) )
    {
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.59f, 0.98f, 0.26f, 0.40f ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.59f, 0.98f, 0.26f, 1.00f ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4( 0.53f, 0.98f, 0.06f, 1.00f ) );
        devmode->reloadShaders = ImGui::Button( "Reload shaders", { -1, 96 } );
        ImGui::PopStyleColor( 3 );

        if( ImGui::SliderFloat( "UI font scale", &devmode->fontGlobalScale, 0.75f, 2.5f, "%.2f" ) )
        {
            MarkDevmodeDirty( *devmode );
        }
        ImGui::TextDisabled( "Persisted to rt/devmode_settings.json (with other Dev knobs)." );

        if( ImGui::Button( "Reset Dev settings to defaults", { -1, 0 } ) )
        {
            ResetDevmodeToDefaults( *devmode );
            // Force immediate save so a crash mid-session does not restore bad Override.
            MaybeSaveDevmodeSettings( *devmode, ovrdFolder, true );
        }
        ImGui::TextDisabled(
            "Clears Override / sticky RR / material kills. Or delete rt/devmode_settings.json." );

        auto& modifiers = devmode->drawInfoOvrd;

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        ImGui::Checkbox( "Override", &modifiers.enable );
        ImGui::BeginDisabled( !modifiers.enable );
        if( ImGui::TreeNodeEx( "Present", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            ImGui::Checkbox( "HDR", &modifiers.hdr );
            
            if( modifiers.frameGeneration != RG_FRAME_GENERATION_MODE_OFF &&
                modifiers.upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS)
            {
                imgui_ShowAlwaysOnCheckbox< "Vsync" >();
            }
            else
            {
                ImGui::Checkbox( "Vsync", &modifiers.vsync );
            }
            
            if( modifiers.frameGeneration == RG_FRAME_GENERATION_MODE_OFF )
            {
                ImGui::Checkbox( "Prefer DXGI for Present", &modifiers.preferDxgiPresent );
            }
            else
            {
                imgui_ShowAlwaysOnCheckbox< "Prefer DXGI for Present" >();
            }

            static_assert( sizeof( RgRenderUpscaleTechnique ) == sizeof( int ) ); // edited as int
            static_assert( sizeof( RgRenderSharpenTechnique ) == sizeof( int ) ); // edited as int
            static_assert( sizeof( RgRenderResolutionMode ) == sizeof( int ) ); // edited as int
            static_assert( sizeof( RgFrameGenerationMode ) == sizeof( int ) ); // edited as int

            {
                ImGui::Spacing();
                ImGui::TextUnformatted( "Frame Generation:" );
                ImGui::RadioButton( "Off##FG",
                                    reinterpret_cast< int* >( &modifiers.frameGeneration ),
                                    RG_FRAME_GENERATION_MODE_OFF );
                ImGui::SameLine();
                ImGui::BeginDisabled( !IsUpscaleTechniqueAvailable( modifiers.upscaleTechnique, //
                                                                    RG_FRAME_GENERATION_MODE_ON,
                                                                    nullptr ) );
                ImGui::RadioButton( "On##FG",
                                    reinterpret_cast< int* >( &modifiers.frameGeneration ),
                                    RG_FRAME_GENERATION_MODE_ON );
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled(
                    !IsUpscaleTechniqueAvailable( modifiers.upscaleTechnique, //
                                                  RG_FRAME_GENERATION_MODE_WITHOUT_GENERATED,
                                                  nullptr ) );
                ImGui::RadioButton( "On, but skip generated frame##FG",
                                    reinterpret_cast< int* >( &modifiers.frameGeneration ),
                                    RG_FRAME_GENERATION_MODE_WITHOUT_GENERATED );
                ImGui::EndDisabled();
            }

            const char* dlssError{};
            const char* fsrError{};

            bool dlssOk = IsUpscaleTechniqueAvailable( RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS, //
                                                       modifiers.frameGeneration,
                                                       &dlssError );
            bool fsrOk  = IsUpscaleTechniqueAvailable( RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2, //
                                                      modifiers.frameGeneration,
                                                      &fsrError );
            {
                ImGui::Spacing();
                ImGui::TextUnformatted( "Upscaler:" );
                ImGui::RadioButton( "Linear##Upscale",
                                    reinterpret_cast< int* >( &modifiers.upscaleTechnique ),
                                    RG_RENDER_UPSCALE_TECHNIQUE_LINEAR );
                ImGui::SameLine();
                ImGui::RadioButton( "Nearest##Upscale",
                                    reinterpret_cast< int* >( &modifiers.upscaleTechnique ),
                                    RG_RENDER_UPSCALE_TECHNIQUE_NEAREST );
                ImGui::SameLine();
                ImGui::BeginDisabled( !fsrOk );
                ImGui::RadioButton( "AMD FSR##Upscale",
                                    reinterpret_cast< int* >( &modifiers.upscaleTechnique ),
                                    RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 );
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::BeginDisabled( !dlssOk );
                ImGui::RadioButton( "NVIDIA DLSS##Upscale",
                                    reinterpret_cast< int* >( &modifiers.upscaleTechnique ),
                                    RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS );
                ImGui::EndDisabled();
            }
            if( !Utils::IsCstrEmpty( dlssError ) )
            {
                ImGui::TextUnformatted( dlssError );
            }
            if( !Utils::IsCstrEmpty( fsrError ) )
            {
                ImGui::TextUnformatted( fsrError );
            }

            bool forceCustom =
                modifiers.upscaleTechnique != RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 &&
                modifiers.upscaleTechnique != RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS;
            if( forceCustom )
            {
                modifiers.resolutionMode = RG_RENDER_RESOLUTION_MODE_CUSTOM;
            }

            {
                ImGui::RadioButton( "Custom##Resolution",
                                    reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                    RG_RENDER_RESOLUTION_MODE_CUSTOM );
                ImGui::SameLine();
                ImGui::BeginDisabled( forceCustom );
                ImGui::RadioButton( "Ultra Performance##Resolution",
                                    reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                    RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE );
                ImGui::SameLine();
                ImGui::RadioButton( "Performance##Resolution",
                                    reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                    RG_RENDER_RESOLUTION_MODE_PERFORMANCE );
                ImGui::SameLine();
                ImGui::RadioButton( "Balanced##Resolution",
                                    reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                    RG_RENDER_RESOLUTION_MODE_BALANCED );
                ImGui::SameLine();
                ImGui::RadioButton( "Quality##Resolution",
                                    reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                    RG_RENDER_RESOLUTION_MODE_QUALITY );
                if( modifiers.upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS ||
                    modifiers.upscaleTechnique == RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2 )
                {
                    ImGui::SameLine();
                    ImGui::RadioButton( "Native AA##Resolution",
                                        reinterpret_cast< int* >( &modifiers.resolutionMode ),
                                        RG_RENDER_RESOLUTION_MODE_NATIVE_AA );
                }
                ImGui::EndDisabled();
            }
            {
                ImGui::BeginDisabled(
                    !( modifiers.resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM ) );

                ImGui::SliderFloat(
                    "Custom render size", &modifiers.customRenderSizeScale, 0.1f, 1.5f );

                ImGui::EndDisabled();
            }
            {
                ImGui::Checkbox( "Downscale to pixelized", &modifiers.pixelizedEnable );
                if( modifiers.pixelizedEnable )
                {
                    ImGui::SliderInt( "Pixelization size", &modifiers.pixelizedHeight, 100, 600 );
                }
            }
            {
                if( ImGui::Checkbox( "DLSS Ray Reconstruction##Present",
                                     &modifiers.rayReconstruction ) )
                {
                    devmode->rayReconstruction       = modifiers.rayReconstruction;
                    devmode->rayReconstructionSticky = true;
                }
                ImGui::TextDisabled( "Also under RR / Denoise live (works without Override)." );
            }

            {
                ImGui::Spacing();
                ImGui::TextUnformatted( "Sharpening:" );
                ImGui::RadioButton( "None##Sharp",
                                    reinterpret_cast< int* >( &modifiers.sharpenTechnique ),
                                    RG_RENDER_SHARPEN_TECHNIQUE_NONE );
                ImGui::SameLine();
                ImGui::RadioButton( "Naive##Sharp",
                                    reinterpret_cast< int* >( &modifiers.sharpenTechnique ),
                                    RG_RENDER_SHARPEN_TECHNIQUE_NAIVE );
                ImGui::SameLine();
                ImGui::RadioButton( "AMD CAS##Sharp",
                                    reinterpret_cast< int* >( &modifiers.sharpenTechnique ),
                                    RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS );
            }

            ImGui::TreePop();
        }
        if( ImGui::TreeNode( "Tonemapping" ) )
        {
            ImGui::Checkbox( "Disable eye adaptation", &modifiers.disableEyeAdaptation );
            ImGui::SliderFloat( "EV100 min", &modifiers.ev100Min, -3, 16, "%.1f" );
            ImGui::SliderFloat( "EV100 max", &modifiers.ev100Max, -3, 16, "%.1f" );
            ImGui::SliderFloat3( "Saturation", modifiers.saturation, -1, 1, "%.1f" );
            ImGui::SliderFloat3( "Crosstalk", modifiers.crosstalk, 0.0f, 1.0f, "%.2f" );
            ImGui::TreePop();
        }
        if( ImGui::TreeNode( "Illumination" ) )
        {
            ImGui::Checkbox( "Anti-firefly", &devmode->antiFirefly );
            ImGui::TextDisabled( "A-SVGF Denoise path only (skipped when DLSS-RR is on)." );
            // 0..8, not the old 0..2: that clamp encoded "only three bounce
            // indices exist", which stopped being true with indirectBounces.
            ImGui::SliderInt( "Shadow rays max depth",
                              &modifiers.maxBounceShadows,
                              0,
                              8,
                              "%d",
                              ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput );
            ImGui::TextDisabled( "A vertex at index >= this samples no analytic lights; depth N needs N+1." );
            ImGui::SliderInt( "Indirect bounces",
                              &modifiers.indirectBounces,
                              1,
                              4,
                              "%d",
                              ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput );
            ImGui::Checkbox( "Legacy bounce weight (stock pi/cos overweight on bounce >= 2)",
                             &modifiers.indirectLegacyBounceWeight );
            ImGui::SliderFloat( "Sensitivity to change: Diffuse Direct",
                                &modifiers.directDiffuseSensitivityToChange,
                                0.0f,
                                1.0f,
                                "%.2f" );
            ImGui::SliderFloat( "Sensitivity to change: Diffuse Indirect",
                                &modifiers.indirectDiffuseSensitivityToChange,
                                0.0f,
                                1.0f,
                                "%.2f" );
            ImGui::SliderFloat( "Sensitivity to change: Specular",
                                &modifiers.specularSensitivityToChange,
                                0.0f,
                                1.0f,
                                "%.2f" );
            ImGui::TreePop();
        }
        if( ImGui::TreeNode( "Texturing" ) )
        {
            ImGui::SliderFloat( "Normal map Scale", &modifiers.normalMapStrength, 0.f, 1.f );
            ImGui::SliderFloat( "Height map Depth", &modifiers.heightMapDepth, 0.f, 0.05f );
            ImGui::SliderFloat( "Emission map GI Boost", &modifiers.emissionMapBoost, 0.f, 100.f );
            ImGui::SliderFloat( "Emission map Screen Scale", &modifiers.emissionMaxScreenColor, 0.f, 100.f );
            ImGui::TreePop();
        }
        if( ImGui::TreeNode( "Lightmap" ) )
        {
            ImGui::SliderFloat( "Screen coverage", &modifiers.lightmapScreenCoverage, 0.0f, 1.0f );
            ImGui::TreePop();
        }
        if( ImGui::TreeNodeEx( "Fluid", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            ImGui::Checkbox( "Enable", &modifiers.fluidEnabled );
            ImGui::DragFloat3(
                "Gravity##fluid", modifiers.fluidGravity.data, 0.1f, -100, 100, "%.1f" );
            modifiers.fluidReset = ImGui::Button( "Reset", { -1, 48 } );
            ImGui::Checkbox( "Suppress Fluid Raster", &devmode->fluidStopVisualize );
            if( ImGui::TreeNodeEx( "Debug Spawn##fluidspw", ImGuiTreeNodeFlags_DefaultOpen ) )
            {
                static int       spawnCount                   = 1000;
                static RgFloat3D spawnPosition                = { 0, 3, 0 };
                static RgFloat3D spawnVelocity                = { 0, 2, 0 };
                static float     spawnVelocityDispersion      = 1.0f;
                static float     spawnVelocityDispersionAngle = 180;
                {
                    ImGui::InputInt( "Count##fluidspw", &spawnCount, 1000, 10'000 );
                    spawnCount = std::clamp( spawnCount, 0, 1'000'000 );
                }
                ImGui::DragFloat3( "Position##fluidspw", spawnPosition.data, 0.5f );
                ImGui::DragFloat3( "Velocity##fluidspw", spawnVelocity.data, 0.5f );
                ImGui::DragFloat( "Dispersion##fluidspw", &spawnVelocityDispersion, 0.1f, 0, 1 );
                ImGui::DragFloat(
                    "Dispersion Angle##fluidspw", &spawnVelocityDispersionAngle, 5, 0, 180 );
                if( ImGui::Button( "Spawn", { -1, 48 } ) )
                {
                    auto info = RgSpawnFluidInfo{
                        .sType                  = RG_STRUCTURE_TYPE_SPAWN_FLUID_INFO,
                        .pNext                  = nullptr,
                        .position               = spawnPosition,
                        .radius                 = 0,
                        .velocity               = spawnVelocity,
                        .dispersionVelocity     = spawnVelocityDispersion,
                        .dispersionAngleDegrees = spawnVelocityDispersionAngle,
                        .count                  = static_cast< uint32_t >( spawnCount ),
                    };
                    static_assert( sizeof( RgSpawnFluidInfo ) == 56, "Change here" );
                    // dev const hack
                    const_cast< VulkanDevice* >( this )->SpawnFluid( &info );
                }
                ImGui::TreePop();
            }
            ImGui::TreePop();
        }
        ImGui::EndDisabled();

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        // Always editable (not gated by Override) for quick before/after denoise / RR tweaks.
        if( ImGui::TreeNodeEx( "RR / Denoise live", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            ImGui::TextUnformatted( "These knobs apply immediately (no Override required)." );

            if( ImGui::Checkbox( "DLSS Ray Reconstruction", &devmode->rayReconstruction ) )
            {
                devmode->rayReconstructionSticky = true;
            }
            ImGui::TextDisabled( "On = ComposeNoisy + DLSS-RR; Off = A-SVGF Denoise path." );
            if( ImGui::Button( "Follow game (rt_rayreconstr)", { -1, 0 } ) )
            {
                devmode->rayReconstructionSticky = false;
            }

            ImGui::Separator();
            ImGui::Checkbox( "Anti-firefly (A-SVGF Denoise)", &devmode->antiFirefly );
            ImGui::TextDisabled( "Only runs when RR is off (Denoise path)." );

            if( ImGui::Checkbox( "RR temporal prefilter (A-SVGF)", &devmode->rrTemporalPrefilter ) )
            {
                devmode->rrTemporalPrefilterSticky = true;
            }
            ImGui::TextDisabled(
                "EXPERIMENTAL — default OFF. Caused faded duplicate/ghost depth view\n"
                "(ASVGF temporal + RR both reproject; checkerboard vs regular coords).\n"
                "Prefer soft lamp fades (rt_ceiling_lamp_fade / off)." );
            if( ImGui::Button( "A: RR temporal OFF", { -1, 0 } ) )
            {
                devmode->rrTemporalPrefilter       = false;
                devmode->rrTemporalPrefilterSticky = true;
            }
            if( ImGui::Button( "B: RR temporal ON", { -1, 0 } ) )
            {
                devmode->rrTemporalPrefilter       = true;
                devmode->rrTemporalPrefilterSticky = true;
            }
            if( ImGui::Button( "RR temporal: use game cvar", { -1, 0 } ) )
            {
                devmode->rrTemporalPrefilterSticky = false;
            }

            ImGui::Separator();
            ImGui::TextUnformatted( "Lighting-change sensitivity" );
            ImGui::TextDisabled( "Higher = drop history faster when lights change." );
            auto sensEdited = false;
            sensEdited |= ImGui::SliderFloat( "Direct##rrsens",
                                              &devmode->illumSensDirect,
                                              0.0f,
                                              1.0f,
                                              "%.2f" );
            sensEdited |= ImGui::SliderFloat( "Indirect##rrsens",
                                              &devmode->illumSensIndirect,
                                              0.0f,
                                              1.0f,
                                              "%.2f" );
            sensEdited |= ImGui::SliderFloat( "Specular##rrsens",
                                              &devmode->illumSensSpec,
                                              0.0f,
                                              1.0f,
                                              "%.2f" );
            if( sensEdited )
            {
                devmode->illumSensSticky = true;
            }
            if( ImGui::Button( "Sens: follow game cvars", { -1, 0 } ) )
            {
                devmode->illumSensSticky = false;
            }
            if( ImGui::Button( "Sens presets: stable (0.5/0.2/0.5)", { -1, 0 } ) )
            {
                devmode->illumSensDirect   = 0.5f;
                devmode->illumSensIndirect = 0.2f;
                devmode->illumSensSpec     = 0.5f;
                devmode->illumSensSticky   = true;
            }
            if( ImGui::Button( "Sens presets: responsive (1/1/1)", { -1, 0 } ) )
            {
                devmode->illumSensDirect   = 1.0f;
                devmode->illumSensIndirect = 1.0f;
                devmode->illumSensSpec     = 1.0f;
                devmode->illumSensSticky   = true;
            }
            if( ImGui::Button( "Sens presets: play default (1/0.75/1)", { -1, 0 } ) )
            {
                devmode->illumSensDirect   = 1.0f;
                devmode->illumSensIndirect = 0.75f;
                devmode->illumSensSpec     = 1.0f;
                devmode->illumSensSticky   = true;
            }

            ImGui::Separator();
            if( devmode->rayReconstructionSticky || devmode->rrTemporalPrefilterSticky ||
                devmode->illumSensSticky )
            {
                ImGui::TextColored( ImVec4( 1.f, 0.85f, 0.2f, 1.f ), "Sticky Dev override(s) active" );
            }
            else
            {
                ImGui::TextDisabled( "Following game / draw params" );
            }
            ImGui::TextWrapped(
                "Upscaler quality / sharpen / pixelize still live under Override → Present." );
            ImGui::TreePop();
        }

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        if( ImGui::TreeNodeEx( "Materials A/B", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            ImGui::TextUnformatted(
                "Live kill-switches for authored RT overlays (no Override needed)." );
            ImGui::TextDisabled(
                "Does not disable dynlights / flashlight / muzzle / ceiling lamps." );
            ImGui::TextDisabled(
                "Split N / ORM / H to isolate RR walk noise (all ON = previous 'Strip PBR maps')." );

            if( ImGui::Checkbox( "Strip normals (_n)", &devmode->materialStripNormals ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled( "Zeros normalMapStrength; HitInfo skips normal sampling." );

            if( ImGui::Checkbox( "Strip metallic (_orm B)", &devmode->materialStripMetallic ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled( "Force metallic=0 (dielectric). Roughness unchanged." );

            if( ImGui::Checkbox( "Strip roughness (_orm G)", &devmode->materialStripRoughness ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled( "Force roughness=1 (fully matte). Metallic unchanged." );

            if( ImGui::SliderFloat( "Roughness toward matte",
                                    &devmode->roughnessTowardMatte,
                                    0.f,
                                    1.f,
                                    "%.2f" ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled(
                "mix(authored, 1). 0=maps as-is, 1=same as Strip roughness. Use this for residual RR shimmer." );
            if( ImGui::Button( "Matte 0", { 0, 0 } ) )
            {
                devmode->roughnessTowardMatte = 0.f;
                MarkDevmodeDirty( *devmode );
            }
            ImGui::SameLine();
            if( ImGui::Button( "Matte 0.5", { 0, 0 } ) )
            {
                devmode->roughnessTowardMatte = 0.5f;
                MarkDevmodeDirty( *devmode );
            }
            ImGui::SameLine();
            if( ImGui::Button( "Matte 1", { 0, 0 } ) )
            {
                devmode->roughnessTowardMatte = 1.f;
                MarkDevmodeDirty( *devmode );
            }

            if( ImGui::Checkbox( "Strip height (_h parallax)", &devmode->materialStripHeight ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled( "Zeros parallaxMaxDepth; HitInfo skips height sampling." );

            if( ImGui::Checkbox( "Strip emissives / attached lights",
                                 &devmode->materialStripEmissives ) )
            {
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TextDisabled(
                "Zeros emission boosts; shaders force emission=0; drops texture-meta attached lights "
                "on upload." );

            if( ImGui::Button( "ORM strips ON (metal+rough)", { -1, 0 } ) )
            {
                devmode->materialStripMetallic  = true;
                devmode->materialStripRoughness = true;
                MarkDevmodeDirty( *devmode );
            }
            if( ImGui::Button( "ORM strips OFF", { -1, 0 } ) )
            {
                devmode->materialStripMetallic  = false;
                devmode->materialStripRoughness = false;
                MarkDevmodeDirty( *devmode );
            }
            if( ImGui::Button( "PBR strips ON (N+metal+rough+H)", { -1, 0 } ) )
            {
                devmode->materialStripNormals   = true;
                devmode->materialStripMetallic  = true;
                devmode->materialStripRoughness = true;
                devmode->materialStripHeight    = true;
                MarkDevmodeDirty( *devmode );
            }
            if( ImGui::Button( "PBR strips OFF", { -1, 0 } ) )
            {
                devmode->materialStripNormals   = false;
                devmode->materialStripMetallic  = false;
                devmode->materialStripRoughness = false;
                devmode->materialStripHeight    = false;
                MarkDevmodeDirty( *devmode );
            }
            if( ImGui::Button( "All materials strips ON", { -1, 0 } ) )
            {
                devmode->materialStripNormals   = true;
                devmode->materialStripMetallic  = true;
                devmode->materialStripRoughness = true;
                devmode->materialStripHeight    = true;
                devmode->materialStripEmissives = true;
                MarkDevmodeDirty( *devmode );
            }
            if( ImGui::Button( "All materials strips OFF", { -1, 0 } ) )
            {
                devmode->materialStripNormals   = false;
                devmode->materialStripMetallic  = false;
                devmode->materialStripRoughness = false;
                devmode->materialStripHeight    = false;
                devmode->materialStripEmissives = false;
                MarkDevmodeDirty( *devmode );
            }
            ImGui::TreePop();
        }

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        if( ImGui::TreeNode( "Debug show" ) )
        {
            std::pair< const char*, uint32_t > fs[] = {
                { "Unfiltered diffuse direct", DEBUG_SHOW_FLAG_UNFILTERED_DIFFUSE },
                { "Unfiltered diffuse indirect", DEBUG_SHOW_FLAG_UNFILTERED_INDIRECT },
                { "Unfiltered specular", DEBUG_SHOW_FLAG_UNFILTERED_SPECULAR },
                { "Diffuse direct", DEBUG_SHOW_FLAG_ONLY_DIRECT_DIFFUSE },
                { "Diffuse indirect", DEBUG_SHOW_FLAG_ONLY_INDIRECT_DIFFUSE },
                { "Specular", DEBUG_SHOW_FLAG_ONLY_SPECULAR },
                { "Albedo white", DEBUG_SHOW_FLAG_ALBEDO_WHITE },
                { "Normals", DEBUG_SHOW_FLAG_NORMALS },
                { "Motion vectors", DEBUG_SHOW_FLAG_MOTION_VECTORS },
                { "Gradients", DEBUG_SHOW_FLAG_GRADIENTS },
                { "Light grid", DEBUG_SHOW_FLAG_LIGHT_GRID },
                { "Bloom", DEBUG_SHOW_FLAG_BLOOM },
            };
            for( const auto [ name, f ] : fs )
            {
                ImGui::CheckboxFlags( name, &devmode->debugShowFlags, f );
            }
            ImGui::TreePop();
        }

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        if( ImGui::TreeNode( "Camera" ) )
        {
            auto& modifiers = devmode->cameraOvrd;

            ImGui::Checkbox( "FOV Override", &modifiers.fovEnable );
            {
                ImGui::BeginDisabled( !modifiers.fovEnable );
                ImGui::SliderFloat( "Vertical FOV", &modifiers.fovDeg, 10, 120, "%.0f degrees" );
                ImGui::EndDisabled();
            }

            ImGui::Checkbox( "Freelook", &modifiers.customEnable );
            ImGui::TextUnformatted(
                "Freelook:\n"
                "    * WASD - to move\n"
                "    * Alt - hold to rotate\n"
                "NOTE: inputs are read only from this window, and not from the game's one" );
            if( modifiers.customEnable )
            {
                if( ImGui::IsKeyPressed( ImGuiKey_LeftAlt ) )
                {
                    if( ImGui::IsMousePosValid() )
                    {
                        modifiers.intr_lastMouse  = { ImGui::GetMousePos().x,
                                                      ImGui::GetMousePos().y };
                        modifiers.intr_lastAngles = modifiers.customAngles;
                    }
                }
                if( ImGui::IsKeyReleased( ImGuiKey_LeftAlt ) )
                {
                    modifiers.intr_lastMouse  = {};
                    modifiers.intr_lastAngles = modifiers.customAngles;
                }

                if( modifiers.intr_lastMouse && ImGui::IsMousePosValid() )
                {
                    modifiers.customAngles = {
                        modifiers.intr_lastAngles.data[ 0 ] -
                            ( ImGui::GetMousePos().x - modifiers.intr_lastMouse->data[ 0 ] ),
                        modifiers.intr_lastAngles.data[ 1 ] -
                            ( ImGui::GetMousePos().y - modifiers.intr_lastMouse->data[ 1 ] ),
                    };
                }
                else
                {
                    modifiers.intr_lastMouse  = {};
                    modifiers.intr_lastAngles = modifiers.customAngles;
                }

                {
                    float speed = 0.1f * sceneImportExport->GetWorldScale();

                    RgFloat3D up, right;
                    Matrix::MakeUpRightFrom( up,
                                             right,
                                             Utils::DegToRad( modifiers.customAngles.data[ 0 ] ),
                                             Utils::DegToRad( modifiers.customAngles.data[ 1 ] ),
                                             sceneImportExport->GetWorldUp(),
                                             sceneImportExport->GetWorldRight() );
                    RgFloat3D fwd = Utils::Cross( up, right );

                    auto fma = []( const RgFloat3D& a, float mult, const RgFloat3D& b ) {
                        return RgFloat3D{ a.data[ 0 ] + mult * b.data[ 0 ],
                                          a.data[ 1 ] + mult * b.data[ 1 ],
                                          a.data[ 2 ] + mult * b.data[ 2 ] };
                    };

                    modifiers.customPos = fma(
                        modifiers.customPos, ImGui::IsKeyDown( ImGuiKey_A ) ? -speed : 0, right );
                    modifiers.customPos = fma(
                        modifiers.customPos, ImGui::IsKeyDown( ImGuiKey_D ) ? +speed : 0, right );
                    modifiers.customPos = fma(
                        modifiers.customPos, ImGui::IsKeyDown( ImGuiKey_W ) ? +speed : 0, fwd );
                    modifiers.customPos = fma(
                        modifiers.customPos, ImGui::IsKeyDown( ImGuiKey_S ) ? -speed : 0, fwd );
                }
            }
            ImGui::TreePop();
        }

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );
        devmode->breakOnTexture[ std::size( devmode->breakOnTexture ) - 1 ] = '\0';
        ImGui::TextUnformatted( "Debug break on texture: " );
        ImGui::Checkbox( "Image upload", &devmode->breakOnTextureImage );
        ImGui::Checkbox( "Primitive upload", &devmode->breakOnTexturePrimitive );
        ImGui::InputText( "##Debug break on texture text",
                          devmode->breakOnTexture,
                          std::size( devmode->breakOnTexture ) );

        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        ImGui::Checkbox( "Always on top", &devmode->debugWindowOnTop );
        debugWindows->SetAlwaysOnTop( devmode->debugWindowOnTop );

        ImGui::Text( "%.3f ms/frame (%.1f FPS)",
                     1000.0f / ImGui::GetIO().Framerate,
                     ImGui::GetIO().Framerate );

        {
            static bool wasActive = false;
            const bool  active    = ImGui::IsAnyItemActive();
            if( wasActive && !active )
            {
                MarkDevmodeDirty( *devmode );
            }
            wasActive = active;
        }
        MaybeSaveDevmodeSettings( *devmode, ovrdFolder, false );

        ImGui::EndTabItem();

        ImGui::Text( "Chosen volumetric light: %d",
                     uniform->GetData()->volumeLightSourceIndex == LIGHT_INDEX_NONE
                         ? -1
                         : uniform->GetData()->volumeLightSourceIndex );
    }

    if( ImGui::BeginTabItem( "Primitives" ) )
    {
        ImGui::Checkbox( "Ignore external geometry", &devmode->ignoreExternalGeometry );
        ImGui::Dummy( ImVec2( 0, 4 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 4 ) );

        using PrimMode = Devmode::DebugPrimMode;

        int*     modePtr = reinterpret_cast< int* >( &devmode->primitivesTableMode );
        PrimMode mode    = devmode->primitivesTableMode;

        ImGui::TextUnformatted( "Record: " );
        ImGui::SameLine();
        ImGui::RadioButton( "None", modePtr, static_cast< int >( PrimMode::None ) );
        ImGui::SameLine();
        ImGui::RadioButton( "Ray-traced", modePtr, static_cast< int >( PrimMode::RayTraced ) );
        ImGui::SameLine();
        ImGui::RadioButton( "Rasterized", modePtr, static_cast< int >( PrimMode::Rasterized ) );
        ImGui::SameLine();
        ImGui::RadioButton( "Non-world", modePtr, static_cast< int >( PrimMode::NonWorld ) );
        ImGui::SameLine();
        ImGui::RadioButton( "Decals", modePtr, static_cast< int >( PrimMode::Decal ) );

        ImGui::TextUnformatted(
            "Red    - if exportable, but not found in GLTF, so uploading as dynamic" );
        ImGui::TextUnformatted( "Green  - if exportable was found in GLTF" );

        if( ImGui::BeginTable( "Primitives table",
                               6,
                               ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti |
                                   ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY ) )
        {
            {
                ImGui::TableSetupColumn( "Call",
                                         ImGuiTableColumnFlags_NoHeaderWidth |
                                             ImGuiTableColumnFlags_DefaultSort );
                ImGui::TableSetupColumn( "Object ID", ImGuiTableColumnFlags_NoHeaderWidth );
                ImGui::TableSetupColumn( "Mesh name", ImGuiTableColumnFlags_NoHeaderWidth );
                ImGui::TableSetupColumn( "Primitive index", ImGuiTableColumnFlags_NoHeaderWidth );
                ImGui::TableSetupColumn( "Primitive name", ImGuiTableColumnFlags_NoHeaderWidth );
                ImGui::TableSetupColumn( "Texture",
                                         ImGuiTableColumnFlags_NoHeaderWidth |
                                             ImGuiTableColumnFlags_WidthStretch );
                ImGui::TableHeadersRow();
                if( ImGui::IsItemHovered() )
                {
                    ImGui::SetTooltip(
                        "Right-click to open menu\nMiddle-click to copy texture name" );
                }
            }

            if( ImGuiTableSortSpecs* sortspecs = ImGui::TableGetSortSpecs() )
            {
                sortspecs->SpecsDirty = true;

                std::ranges::sort(
                    devmode->primitivesTable,
                    [ sortspecs ]( const Devmode::DebugPrim& a,
                                   const Devmode::DebugPrim& b ) -> bool {
                        for( int n = 0; n < sortspecs->SpecsCount; n++ )
                        {
                            const ImGuiTableColumnSortSpecs* srt = &sortspecs->Specs[ n ];

                            std::strong_ordering ord = std::strong_ordering::equal;
                            switch( srt->ColumnIndex )
                            {
                                case 0: ord = ( a.callIndex <=> b.callIndex ); break;
                                case 1: ord = ( a.objectId <=> b.objectId ); break;
                                case 2: ord = ( a.meshName <=> b.meshName ); break;
                                case 3: ord = ( a.primitiveIndex <=> b.primitiveIndex ); break;
                                case 4: ord = ( a.primitiveName <=> b.primitiveName ); break;
                                case 5: ord = ( a.textureName <=> b.textureName ); break;
                                default: assert( 0 ); return false;
                            }

                            if( std::is_gt( ord ) )
                            {
                                return srt->SortDirection != ImGuiSortDirection_Ascending;
                            }

                            if( std::is_lt( ord ) )
                            {
                                return srt->SortDirection == ImGuiSortDirection_Ascending;
                            }
                        }

                        return a.callIndex < b.callIndex;
                    } );
            }

            ImGuiListClipper clipper;
            clipper.Begin( int( devmode->primitivesTable.size() ) );
            while( clipper.Step() )
            {
                for( int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++ )
                {
                    const auto& prim = devmode->primitivesTable[ i ];
                    ImGui::TableNextRow();

                    if( prim.result == UploadResult::ExportableStatic )
                    {
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0,
                                                IM_COL32( 0, 128, 0, 64 ) );
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg1,
                                                IM_COL32( 0, 128, 0, 128 ) );
                    }
                    else if( prim.result == UploadResult::ExportableDynamic )
                    {
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0,
                                                IM_COL32( 128, 0, 0, 64 ) );
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg1,
                                                IM_COL32( 128, 0, 0, 128 ) );
                    }
                    else
                    {
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, IM_COL32( 0, 0, 0, 1 ) );
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg1, IM_COL32( 0, 0, 0, 1 ) );
                    }


                    ImGui::TableNextColumn();
                    if( prim.result != UploadResult::Fail )
                    {
                        ImGui::Text( "%u", prim.callIndex );
                    }
                    else
                    {
                        ImGui::TextUnformatted( "fail" );
                    }

                    ImGui::TableNextColumn();
                    if( mode != PrimMode::Decal && mode != PrimMode::NonWorld )
                    {
                        ImGui::Text( "%llu", prim.objectId );
                    }

                    ImGui::TableNextColumn();
                    if( mode != PrimMode::Decal && mode != PrimMode::NonWorld )
                    {
                        ImGui::TextUnformatted( prim.meshName.c_str() );
                    }

                    ImGui::TableNextColumn();
                    if( mode != PrimMode::Decal )
                    {
                        ImGui::Text( "%u", prim.primitiveIndex );
                    }

                    ImGui::TableNextColumn();
                    if( mode != PrimMode::Decal )
                    {
                        ImGui::TextUnformatted( prim.primitiveName.c_str() );
                    }

                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted( prim.textureName.c_str() );
                    if( ImGui::IsMouseReleased( ImGuiMouseButton_Middle ) &&
                        ImGui::IsItemHovered() )
                    {
                        ImGui::SetClipboardText( prim.textureName.c_str() );
                    }
                    else
                    {
                        if( ImGui::BeginPopupContextItem( std::format( "##popup{}", i ).c_str() ) )
                        {
                            if( ImGui::MenuItem( "Copy texture name" ) )
                            {
                                ImGui::SetClipboardText( prim.textureName.c_str() );
                                ImGui::CloseCurrentPopup();
                            }
                            ImGui::EndPopup();
                        }
                    }
                }
            }

            ImGui::EndTable();
        }
        ImGui::EndTabItem();
    }

    if( ImGui::BeginTabItem( "Log" ) )
    {
        ImGui::Checkbox( "Auto-scroll", &devmode->logAutoScroll );
        ImGui::SameLine();
        if( ImGui::Button( "Clear" ) )
        {
            devmode->logs.clear();
        }
        ImGui::Separator();

        ImGui::CheckboxFlags( "Errors", &devmode->logFlags, RG_MESSAGE_SEVERITY_ERROR );
        ImGui::SameLine();
        ImGui::CheckboxFlags( "Warnings", &devmode->logFlags, RG_MESSAGE_SEVERITY_WARNING );
        ImGui::SameLine();
        ImGui::CheckboxFlags( "Info", &devmode->logFlags, RG_MESSAGE_SEVERITY_INFO );
        ImGui::SameLine();
        ImGui::CheckboxFlags( "Verbose", &devmode->logFlags, RG_MESSAGE_SEVERITY_VERBOSE );
        ImGui::Separator();

        if( ImGui::BeginChild( "##LogScrollingRegion",
                               ImVec2( 0, 0 ),
                               false,
                               ImGuiWindowFlags_HorizontalScrollbar ) )
        {
            for( const auto& [ severity, count, text ] : devmode->logs )
            {
                RgMessageSeverityFlags filtered = severity & devmode->logFlags;

                if( filtered == 0 )
                {
                    continue;
                }

                std::optional< ImU32 > color;
                if( filtered & RG_MESSAGE_SEVERITY_ERROR )
                {
                    color = IM_COL32( 255, 0, 0, 255 );
                }
                else if( filtered & RG_MESSAGE_SEVERITY_WARNING )
                {
                    color = IM_COL32( 255, 255, 0, 255 );
                }

                if( color )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, *color );
                }

                if( count == 1 )
                {
                    ImGui::TextUnformatted( text.data() );
                }
                else
                {
                    ImGui::Text( "[%u] %s", count, text.data() );
                }

                if( color )
                {
                    ImGui::PopStyleColor();
                }
            }

            if( devmode->logAutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() )
            {
                ImGui::SetScrollHereY( 1.0f );
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }

    if( ImGui::BeginTabItem( "Import/Export" ) )
    {
        auto& dev = sceneImportExport->dev;
        if( !dev.exportName.enable )
        {
            dev.exportName.SetDefaults( *sceneImportExport );
        }
        if( !dev.importName.enable )
        {
            dev.importName.SetDefaults( *sceneImportExport );
        }
        if( !dev.worldTransform.enable )
        {
            dev.worldTransform.SetDefaults( *sceneImportExport );
        }

        {
            ImGui::Text( "Resource folder: %s",
                         std::filesystem::absolute( ovrdFolder ).string().c_str() );
        }
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 16 ) );
        {
            ImGui::BeginDisabled( dev.buttonRecording );
            if( ImGui::Button( "Reimport replacements GLTF", { -1, 80 } ) )
            {
                sceneImportExport->RequestReplacementsReimport();
            }
            ImGui::Dummy( ImVec2( 0, 8 ) );
            if( ImGui::Button( "Reimport map GLTF", { -1, 80 } ) )
            {
                sceneImportExport->RequestReimport();
            }

            ImGui::Text( "Map import path: %s",
                         sceneImportExport->dev_GetSceneImportGltfPath().c_str() );
            ImGui::BeginDisabled( !dev.importName.enable );
            {
                ImGui::InputText(
                    "Import map name", dev.importName.value, std::size( dev.importName.value ) );
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox( "Custom##import", &dev.importName.enable );
            ImGui::EndDisabled();
        }
        ImGui::Dummy( ImVec2( 0, 16 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 16 ) );
        {
            ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0.98f, 0.59f, 0.26f, 0.40f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImVec4( 0.98f, 0.59f, 0.26f, 1.00f ) );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImVec4( 0.98f, 0.53f, 0.06f, 1.00f ) );
            {
                const auto halfWidth = ( ImGui::GetContentRegionAvail().x -
                                         ImGui::GetCurrentWindow()->WindowPadding.x ) *
                                       0.5f;
                if( dev.buttonRecording )
                {
                    ImGui::BeginDisabled( true );
                    ImGui::Button( "Replacements are being recorded...", { halfWidth, 80 } );
                    ImGui::EndDisabled();
                }
                else
                {
                    if( ImGui::Button( "Export replacements GLTF\n from this frame",
                                       { halfWidth, 80 } ) )
                    {
                        sceneImportExport->RequestReplacementsExport_OneFrame();
                    }
                }
                ImGui::SameLine();
                if( dev.buttonRecording )
                {
                    if( ImGui::Button( "Stop recording\nand Export into GLTF", { halfWidth, 80 } ) )
                    {
                        sceneImportExport->RequestReplacementsExport_RecordEnd();
                        dev.buttonRecording = false;
                    }
                }
                else
                {
                    if( ImGui::Button( "Start recording\nreplacements into GLTF",
                                       { halfWidth, 80 } ) )
                    {
                        sceneImportExport->RequestReplacementsExport_RecordBegin();
                        dev.buttonRecording = true;
                    }
                }
            }
            ImGui::BeginDisabled( dev.buttonRecording );
            ImGui::Checkbox( "Allow export of existing replacements",
                             &devmode->allowExportOfExistingReplacements );
            ImGui::Dummy( ImVec2( 0, 16 ) );
            if( ImGui::Button( "Export map GLTF", { -1, 80 } ) )
            {
                sceneImportExport->RequestExport();
            }
            ImGui::PopStyleColor( 3 );
            ImGui::Checkbox( "Allow auto-export, if scene's GLTF doesn't exist",
                             &devmode->drawInfoOvrd.allowMapAutoExport );
            ImGui::Dummy( ImVec2( 0, 8 ) );
            ImGui::Text( "Export path: %s",
                         sceneImportExport->dev_GetSceneExportGltfPath().c_str() );
            ImGui::BeginDisabled( !dev.exportName.enable );
            {
                ImGui::InputText(
                    "Export map name", dev.exportName.value, std::size( dev.exportName.value ) );
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::Checkbox( "Custom##export", &dev.exportName.enable );
            ImGui::EndDisabled();
        }
        ImGui::Dummy( ImVec2( 0, 16 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 16 ) );
        {
            ImGui::BeginDisabled( dev.buttonRecording );
            ImGui::Checkbox( "Custom import/export world space", &dev.worldTransform.enable );
            ImGui::BeginDisabled( !dev.worldTransform.enable );
            {
                ImGui::SliderFloat3( "World Up vector", dev.worldTransform.up.data, -1.0f, 1.0f );
                ImGui::SliderFloat3(
                    "World Forward vector", dev.worldTransform.forward.data, -1.0f, 1.0f );
                ImGui::InputFloat(
                    std::format( "1 unit = {} meters", dev.worldTransform.scale ).c_str(),
                    &dev.worldTransform.scale );
            }
            ImGui::EndDisabled();
            ImGui::EndDisabled();
        }
        ImGui::EndTabItem();
    }

    if( ImGui::BeginTabItem( "Textures" ) )
    {
        if( ImGui::Button( "Export original textures", { -1, 80 } ) )
        {
            textureManager->ExportOriginalMaterialTextures( ovrdFolder /
                                                            TEXTURES_FOLDER_ORIGINALS );
        }
        ImGui::Text( "Export path: %s",
                     ( ovrdFolder / TEXTURES_FOLDER_ORIGINALS ).string().c_str() );
        ImGui::Dummy( ImVec2( 0, 16 ) );
        ImGui::Separator();
        ImGui::Dummy( ImVec2( 0, 16 ) );

        enum
        {
            ColumnTextureIndex0,
            ColumnTextureIndex1,
            ColumnTextureIndex2,
            ColumnTextureIndex3,
            ColumnTextureIndex4,
            ColumnMaterialName,
            Column_Count,
        };
        static_assert( std::size( TextureManager::Debug_MaterialInfo{}.textures.indices ) == 5 );

        ImGui::Checkbox( "Record", &devmode->materialsTableEnable );
        ImGui::TextUnformatted( "Blue - if material is non-original (i.e. was loaded from GLTF)" );
        if( ImGui::BeginTable( "Materials table",
                               Column_Count,
                               ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_Sortable | ImGuiTableFlags_SortMulti |
                                   ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY ) )
        {
            auto materialInfos = devmode->materialsTableEnable
                                     ? textureManager->Debug_GetMaterials()
                                     : std::vector< TextureManager::Debug_MaterialInfo >{};
            {
                ImGui::TableSetupColumn( "A", 0, 8 );
                ImGui::TableSetupColumn( "P", 0, 8 );
                ImGui::TableSetupColumn( "N", 0, 8 );
                ImGui::TableSetupColumn( "E", 0, 8 );
                ImGui::TableSetupColumn( "H", 0, 8 );
                ImGui::TableSetupColumn( "Material name",
                                         ImGuiTableColumnFlags_WidthStretch |
                                             ImGuiTableColumnFlags_DefaultSort,
                                         -1 );
                ImGui::TableHeadersRow();
                if( ImGui::IsItemHovered() )
                {
                    ImGui::SetTooltip(
                        "Right-click to open menu\nMiddle-click to copy texture name" );
                }
            }

            if( ImGuiTableSortSpecs* sortspecs = ImGui::TableGetSortSpecs() )
            {
                sortspecs->SpecsDirty = true;

                std::ranges::sort(
                    materialInfos,
                    [ sortspecs ]( const TextureManager::Debug_MaterialInfo& a,
                                   const TextureManager::Debug_MaterialInfo& b ) -> bool {
                        for( int n = 0; n < sortspecs->SpecsCount; n++ )
                        {
                            const ImGuiTableColumnSortSpecs* srt = &sortspecs->Specs[ n ];

                            std::strong_ordering ord = std::strong_ordering::equal;
                            switch( srt->ColumnIndex )
                            {
                                case ColumnTextureIndex0:
                                    ord = ( a.textures.indices[ 0 ] <=> b.textures.indices[ 0 ] );
                                    break;
                                case ColumnTextureIndex1:
                                    ord = ( a.textures.indices[ 1 ] <=> b.textures.indices[ 1 ] );
                                    break;
                                case ColumnTextureIndex2:
                                    ord = ( a.textures.indices[ 2 ] <=> b.textures.indices[ 2 ] );
                                    break;
                                case ColumnTextureIndex3:
                                    ord = ( a.textures.indices[ 3 ] <=> b.textures.indices[ 3 ] );
                                    break;
                                case ColumnTextureIndex4:
                                    ord = ( a.textures.indices[ 4 ] <=> b.textures.indices[ 4 ] );
                                    break;
                                case ColumnMaterialName:
                                    ord = ( a.materialName <=> b.materialName );
                                    break;
                                default: continue;
                            }

                            if( std::is_gt( ord ) )
                            {
                                return srt->SortDirection != ImGuiSortDirection_Ascending;
                            }

                            if( std::is_lt( ord ) )
                            {
                                return srt->SortDirection == ImGuiSortDirection_Ascending;
                            }
                        }

                        return a.materialName < b.materialName;
                    } );
            }

            ImGuiListClipper clipper;
            clipper.Begin( int( materialInfos.size() ) );
            while( clipper.Step() )
            {
                for( int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++ )
                {
                    const auto& mat = materialInfos[ i ];
                    ImGui::TableNextRow();
                    ImGui::PushID( i );

                    if( mat.isOriginal )
                    {
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0,
                                                IM_COL32( 0, 0, 128, 64 ) );
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg1,
                                                IM_COL32( 0, 0, 128, 128 ) );
                    }
                    else
                    {
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg0, IM_COL32( 0, 0, 0, 1 ) );
                        ImGui::TableSetBgColor( ImGuiTableBgTarget_RowBg1, IM_COL32( 0, 0, 0, 1 ) );
                    }

                    auto writeTexIndex = [ &mat ]( int channel ) {
                        assert( channel >= 0 && channel < std::size( mat.textures.indices ) );
                        if( mat.textures.indices[ channel ] != EMPTY_TEXTURE_INDEX )
                        {
                            ImGui::Text( "%u", mat.textures.indices[ channel ] );
                        }
                    };

                    for( auto col = 0; col < Column_Count; col++ )
                    {
                        ImGui::TableNextColumn();

                        switch( col )
                        {
                            case ColumnTextureIndex0:
                                writeTexIndex( 0 );
                                if( ImGui::TableGetColumnFlags( col ) &
                                    ImGuiTableColumnFlags_IsHovered )
                                {
                                    ImGui::SetTooltip( "Image\n[RGB]Albedo\n[A] "
                                                       "Alpha (0.0 - fully transparent)" );
                                }
                                break;

                            case ColumnTextureIndex1:
                                writeTexIndex( 1 );
                                if( ImGui::TableGetColumnFlags( col ) &
                                    ImGuiTableColumnFlags_IsHovered )
                                {
                                    ImGui::SetTooltip(
                                        "Image\n[R]Occlusion (disabled by default)\n[G] "
                                        "Roughness\n[B] Metallic" );
                                }
                                break;

                            case ColumnTextureIndex2:
                                writeTexIndex( 2 );
                                if( ImGui::TableGetColumnFlags( col ) &
                                    ImGuiTableColumnFlags_IsHovered )
                                {
                                    ImGui::SetTooltip(
                                        "Image\n[R] Normal X offset\n[G] Normal Y offset" );
                                }
                                break;

                            case ColumnTextureIndex3:
                                writeTexIndex( 3 );
                                if( ImGui::TableGetColumnFlags( col ) &
                                    ImGuiTableColumnFlags_IsHovered )
                                {
                                    ImGui::SetTooltip( "Image\n[RGB] Emission color" );
                                }
                                break;

                            case ColumnTextureIndex4:
                                writeTexIndex( 4 );
                                if( ImGui::TableGetColumnFlags( col ) &
                                    ImGuiTableColumnFlags_IsHovered )
                                {
                                    ImGui::SetTooltip( "Image\n[R] Height map\n"
                                                       "    0.0 - deepest point\n"
                                                       "    1.0 - surface level" );
                                }
                                break;

                            case ColumnMaterialName:
                                ImGui::TextUnformatted( mat.materialName.c_str() );

                                if( ImGui::IsMouseReleased( ImGuiMouseButton_Middle ) &&
                                    ImGui::IsItemHovered() )
                                {
                                    ImGui::SetClipboardText( mat.materialName.c_str() );
                                }
                                else
                                {
                                    if( ImGui::BeginPopupContextItem(
                                            std::format( "##popup{}", i ).c_str() ) )
                                    {
                                        if( ImGui::MenuItem( "Copy texture name" ) )
                                        {
                                            ImGui::SetClipboardText( mat.materialName.c_str() );
                                            ImGui::CloseCurrentPopup();
                                        }
                                        ImGui::EndPopup();
                                    }
                                }
                                break;

                            default: break;
                        }
                    }

                    ImGui::PopID();
                }
            }

            ImGui::EndTable();
        }
        ImGui::EndTabItem();
    }

    // Persist edits from any tab (debounce-save after widget deactivation).
    {
        static bool wasActive = false;
        const bool  active    = ImGui::IsAnyItemActive();
        if( wasActive && !active )
        {
            MarkDevmodeDirty( *devmode );
        }
        wasActive = active;
    }
    MaybeSaveDevmodeSettings( *devmode, ovrdFolder, false );
}

// DLSS-RR is the one Dev knob that silently contradicts the game and is
// invisible from the game side: gzdoom's rt_rr_status reads its own request,
// which is what we are about to replace here. Warn whenever the applied value
// disagrees with what the game asked for, edge-triggered so it does not spam
// every frame. Requires -rtdebug to be visible (rt_main.cpp mutes RTGL
// messages otherwise).
static void Dev_WarnIfRrOverridden( bool gameWants, bool applied )
{
    static bool s_haveprev = false;
    static bool s_prevgame = false;
    static bool s_prevappl = false;

    if( s_haveprev && gameWants == s_prevgame && applied == s_prevappl )
    {
        return;
    }
    s_haveprev = true;
    s_prevgame = gameWants;
    s_prevappl = applied;

    if( gameWants != applied )
    {
        RTGL1::debug::Warning( "Dev override: DLSS Ray Reconstruction forced {} "
                               "(game requested {} via rt_rayreconstr). "
                               "Use \"Follow game (rt_rayreconstr)\" in the Dev UI, or delete "
                               "rt/devmode_settings.json, to hand control back.",
                               applied ? "ON" : "OFF",
                               gameWants ? "ON" : "OFF" );
    }
}

void RTGL1::VulkanDevice::Dev_Override( RgStartFrameInfo&                   info,
                                        RgStartFrameRenderResolutionParams& resolution,
                                        RgStartFrameFluidParams&            fluid ) const
{
    if( !Dev_IsDevmodeInitialized() )
    {
        return;
    }

    auto& modifiers = devmode->drawInfoOvrd;

    if( modifiers.enable )
    {
        RgStartFrameInfo&                   dst       = info;
        RgStartFrameRenderResolutionParams& dst_resol = resolution;
        RgStartFrameFluidParams&            dst_fluid = fluid;

        // apply modifiers
        {
            dst.vsync                  = modifiers.vsync;
            dst.hdr                    = modifiers.hdr;
            dst.allowMapAutoExport     = modifiers.allowMapAutoExport;
            dst.lightmapScreenCoverage = modifiers.lightmapScreenCoverage;
        }
        {
            dst_fluid.enabled = modifiers.fluidEnabled;
            dst_fluid.reset   = modifiers.fluidReset;
            dst_fluid.gravity = modifiers.fluidGravity;
        }
        {
            float aspect = float( renderResolution.UpscaledWidth() ) /
                           float( renderResolution.UpscaledHeight() );

            dst_resol.upscaleTechnique  = modifiers.upscaleTechnique;
            dst_resol.resolutionMode    = modifiers.resolutionMode;
            dst_resol.frameGeneration   = modifiers.frameGeneration;
            dst_resol.preferDxgiPresent = modifiers.preferDxgiPresent;
            dst_resol.sharpenTechnique  = modifiers.sharpenTechnique;
            dst_resol.customRenderSize  = {
                ClampPix< uint32_t >( modifiers.customRenderSizeScale *
                                      float( renderResolution.UpscaledWidth() ) ),
                ClampPix< uint32_t >( modifiers.customRenderSizeScale *
                                      float( renderResolution.UpscaledHeight() ) ),
            };
            dst_resol.pixelizedRenderSizeEnable = modifiers.pixelizedEnable;
            dst_resol.pixelizedRenderSize       = {
                ClampPix< uint32_t >(
                    static_cast< uint32_t >( aspect * float( modifiers.pixelizedHeight ) ) ),
                ClampPix< uint32_t >( modifiers.pixelizedHeight ),
            };
            if( devmode->rayReconstructionSticky )
            {
                Dev_WarnIfRrOverridden( !!dst_resol.rayReconstruction,
                                        devmode->rayReconstruction );
                dst_resol.rayReconstruction   = devmode->rayReconstruction;
                modifiers.rayReconstruction   = devmode->rayReconstruction;
            }
            else
            {
                Dev_WarnIfRrOverridden( !!dst_resol.rayReconstruction,
                                        modifiers.rayReconstruction );
                dst_resol.rayReconstruction = modifiers.rayReconstruction;
                devmode->rayReconstruction  = modifiers.rayReconstruction;
            }
        }
    }
    else if( devmode->rayReconstructionSticky )
    {
        // Reached with the Override master switch OFF -- a sticky Dev-UI RR
        // toggle still replaces the game's rt_rayreconstr here. Intentional
        // (the UI advertises "works without Override"), but silent, so warn.
        Dev_WarnIfRrOverridden( !!resolution.rayReconstruction,
                                devmode->rayReconstruction );
        resolution.rayReconstruction = devmode->rayReconstruction;
    }
    else
    {
        const RgStartFrameInfo&                   src       = info;
        const RgStartFrameRenderResolutionParams& src_resol = resolution;
        const RgStartFrameFluidParams&            src_fluid = fluid;

        // reset modifiers
        {
            modifiers.vsync                  = src.vsync;
            modifiers.hdr                    = src.hdr;
            modifiers.allowMapAutoExport     = src.allowMapAutoExport;
            modifiers.lightmapScreenCoverage = src.lightmapScreenCoverage;
        }
        {
            modifiers.fluidEnabled = src_fluid.enabled;
            modifiers.fluidReset   = src_fluid.reset;
            modifiers.fluidGravity = src_fluid.gravity;
        }
        {
            modifiers.upscaleTechnique  = src_resol.upscaleTechnique;
            modifiers.resolutionMode    = src_resol.resolutionMode;
            modifiers.frameGeneration   = src_resol.frameGeneration;
            modifiers.preferDxgiPresent = src_resol.preferDxgiPresent;
            modifiers.sharpenTechnique  = src_resol.sharpenTechnique;
            modifiers.rayReconstruction = !!src_resol.rayReconstruction;
            if( !devmode->rayReconstructionSticky )
            {
                devmode->rayReconstruction = modifiers.rayReconstruction;
            }

            if( modifiers.resolutionMode == RG_RENDER_RESOLUTION_MODE_CUSTOM )
            {
                modifiers.customRenderSizeScale = float( src_resol.customRenderSize.height ) /
                                                  float( renderResolution.UpscaledHeight() );
            }
            else
            {
                modifiers.customRenderSizeScale = 1.0f;
            }

            modifiers.pixelizedEnable = src_resol.pixelizedRenderSizeEnable;
            modifiers.pixelizedHeight =
                src_resol.pixelizedRenderSizeEnable
                    ? ClampPix< int >( src_resol.pixelizedRenderSize.height )
                    : 0;
        }
    }
}

void RTGL1::VulkanDevice::Dev_Override( RgCameraInfo& info ) const
{
    if( !Dev_IsDevmodeInitialized() )
    {
        assert( 0 );
        return;
    }

    auto& modifiers = devmode->cameraOvrd;

    if( modifiers.fovEnable )
    {
        info.fovYRadians = Utils::DegToRad( modifiers.fovDeg );
    }
    else
    {
        modifiers.fovDeg = Utils::RadToDeg( info.fovYRadians );
    }

    if( modifiers.customEnable )
    {
        RgCameraInfo& dst_camera = info;

        dst_camera.position = modifiers.customPos;
        Matrix::MakeUpRightFrom( dst_camera.up,
                                 dst_camera.right,
                                 Utils::DegToRad( modifiers.customAngles.data[ 0 ] ),
                                 Utils::DegToRad( modifiers.customAngles.data[ 1 ] ),
                                 sceneImportExport->GetWorldUp(),
                                 sceneImportExport->GetWorldRight() );
    }
    else
    {
        const RgCameraInfo& src_camera = info;

        modifiers.customPos    = src_camera.position;
        modifiers.customAngles = { 0, 0 };
    }
}

void RTGL1::VulkanDevice::Dev_Override( RgDrawFrameIlluminationParams& illumination,
                                        RgDrawFrameTonemappingParams&  tonemappingp,
                                        RgDrawFrameTexturesParams&     textures ) const
{
    if( !Dev_IsDevmodeInitialized() )
    {
        return;
    }

    auto& modifiers = devmode->drawInfoOvrd;

    if( modifiers.enable )
    {
        RgDrawFrameIlluminationParams& dst_illum = illumination;
        RgDrawFrameTonemappingParams&  dst_tnmp  = tonemappingp;
        RgDrawFrameTexturesParams&     dst_tex   = textures;

        // apply modifiers
        {
            dst_illum.maxBounceShadows                 = modifiers.maxBounceShadows;
            dst_illum.indirectBounces                  = uint32_t( modifiers.indirectBounces );
            dst_illum.indirectLegacyBounceWeight       = modifiers.indirectLegacyBounceWeight;
            dst_illum.directDiffuseSensitivityToChange = modifiers.directDiffuseSensitivityToChange;
            dst_illum.indirectDiffuseSensitivityToChange =
                modifiers.indirectDiffuseSensitivityToChange;
            dst_illum.specularSensitivityToChange = modifiers.specularSensitivityToChange;
        }
        {
            dst_tnmp.disableEyeAdaptation = modifiers.disableEyeAdaptation;
            dst_tnmp.ev100Min             = modifiers.ev100Min;
            dst_tnmp.ev100Max             = modifiers.ev100Max;
            dst_tnmp.saturation           = { RG_ACCESS_VEC3( modifiers.saturation ) };
            dst_tnmp.crosstalk            = { RG_ACCESS_VEC3( modifiers.crosstalk ) };
        }
        {
            dst_tex.normalMapStrength      = modifiers.normalMapStrength;
            dst_tex.heightMapDepth         = modifiers.heightMapDepth;
            dst_tex.emissionMapBoost       = modifiers.emissionMapBoost;
            dst_tex.emissionMaxScreenColor = modifiers.emissionMaxScreenColor;
        }
    }
    else
    {
        const RgDrawFrameIlluminationParams& src_illum = illumination;
        const RgDrawFrameTonemappingParams&  src_tnmp  = tonemappingp;
        const RgDrawFrameTexturesParams&     src_tex   = textures;

        // reset modifiers from game — do not clobber live RR/Denoise sticky knobs
        {
            modifiers.maxBounceShadows                 = int( src_illum.maxBounceShadows );
            modifiers.indirectBounces                  = int( src_illum.indirectBounces );
            modifiers.indirectLegacyBounceWeight       = !!src_illum.indirectLegacyBounceWeight;
            modifiers.directDiffuseSensitivityToChange = src_illum.directDiffuseSensitivityToChange;
            modifiers.indirectDiffuseSensitivityToChange =
                src_illum.indirectDiffuseSensitivityToChange;
            modifiers.specularSensitivityToChange = src_illum.specularSensitivityToChange;
        }
        {
            modifiers.disableEyeAdaptation = src_tnmp.disableEyeAdaptation;
            modifiers.ev100Min             = src_tnmp.ev100Min;
            modifiers.ev100Max             = src_tnmp.ev100Max;
            RG_SET_VEC3_A( modifiers.saturation, src_tnmp.saturation.data );
            RG_SET_VEC3_A( modifiers.crosstalk, src_tnmp.crosstalk.data );
        }
        {
            modifiers.normalMapStrength      = src_tex.normalMapStrength;
            modifiers.heightMapDepth         = src_tex.heightMapDepth;
            modifiers.emissionMapBoost       = src_tex.emissionMapBoost;
            modifiers.emissionMaxScreenColor = src_tex.emissionMaxScreenColor;
        }
    }

    // Materials A/B kill-switches (apply even when Override is off).
    if( devmode->materialStripNormals )
    {
        textures.normalMapStrength = 0.f;
    }
    if( devmode->materialStripHeight )
    {
        textures.heightMapDepth = 0.f;
    }
    if( devmode->materialStripEmissives )
    {
        textures.emissionMapBoost       = 0.f;
        textures.emissionMaxScreenColor = 0.f;
    }
}

void RTGL1::VulkanDevice::Dev_TryBreak( const char* pTextureName, bool isImageUpload )
{
#ifdef _MSC_VER
    if( !devmode )
    {
        return;
    }

    if( isImageUpload )
    {
        if( !devmode->breakOnTextureImage )
        {
            return;
        }
    }
    else
    {
        if( !devmode->breakOnTexturePrimitive )
        {
            return;
        }
    }

    if( Utils::IsCstrEmpty( devmode->breakOnTexture ) || Utils::IsCstrEmpty( pTextureName ) )
    {
        return;
    }

    devmode->breakOnTexture[ std::size( devmode->breakOnTexture ) - 1 ] = '\0';
    if( std::strcmp( devmode->breakOnTexture, Utils::SafeCstr( pTextureName ) ) == 0 )
    {
        __debugbreak();
        devmode->breakOnTextureImage     = false;
        devmode->breakOnTexturePrimitive = false;
    }
#endif
}

namespace RTGL1
{
extern bool g_showAutoExportPlaque;
}

void RTGL1::VulkanDevice::DrawEndUserWarnings()
{
    constexpr int   OverallDurationInSeconds = 7;
    constexpr float FadingInSeconds          = 3;

    using clock        = std::chrono::high_resolution_clock;
    static auto stopAt = clock::time_point{};
    static auto ratio  = 0.0f;

    if( g_showAutoExportPlaque )
    {
        stopAt                 = clock::now() + std::chrono::seconds{ OverallDurationInSeconds };
        ratio                  = 1.0f;
        g_showAutoExportPlaque = false;
    }

    if( ratio <= 0 )
    {
        return;
    }

    {
        float diffStop = std::chrono::duration< float >{ stopAt - clock::now() }.count();
        ratio          = std::clamp( diffStop / FadingInSeconds, 0.f, 1.f );
        if( ratio <= 0 )
        {
            return;
        }
    }

    static constexpr RgColor4DPacked32 white       = Utils::PackColor( 255, 255, 255, 255 );
    static constexpr RgPrimitiveVertex quadVerts[] = {
        RgPrimitiveVertex{ .position = { -1, -1, 0 }, .texCoord = { 0, 0 }, .color = white },
        RgPrimitiveVertex{ .position = { -1, +1, 0 }, .texCoord = { 0, 1 }, .color = white },
        RgPrimitiveVertex{ .position = { +1, -1, 0 }, .texCoord = { 1, 0 }, .color = white },
        RgPrimitiveVertex{ .position = { +1, -1, 0 }, .texCoord = { 1, 0 }, .color = white },
        RgPrimitiveVertex{ .position = { -1, +1, 0 }, .texCoord = { 0, 1 }, .color = white },
        RgPrimitiveVertex{ .position = { +1, +1, 0 }, .texCoord = { 1, 1 }, .color = white },
    };

    const float screen[] = {
        static_cast< float >( renderResolution.GetResolutionState().upscaledWidth ),
        static_cast< float >( renderResolution.GetResolutionState().upscaledHeight ),
    };
    // size of MATERIAL_NAME_SCENEBUILDINGWARNING texture
    constexpr float plaque[] = {
        1024,
        256,
    };
    if( screen[ 0 ] < 1 || screen[ 1 ] < 1 )
    {
        return;
    }

    constexpr float safeZoneAt1080 = 96;
    constexpr float heightAt1080   = 128;

    const float safeZone  = safeZoneAt1080 / 1080 * screen[ 1 ];
    const float pixHeight = heightAt1080 / 1080 * screen[ 1 ];
    const float pixWidth  = pixHeight / plaque[ 1 ] * plaque[ 0 ];

    auto vp = RgViewport{
        .x        = screen[ 0 ] / 2 - pixWidth / 2, // at center
        .y        = safeZone,
        .width    = pixWidth,
        .height   = pixHeight,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    static constexpr float identity[] = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    };
    auto sw = RgMeshPrimitiveSwapchainedEXT{
        .sType           = RG_STRUCTURE_TYPE_MESH_PRIMITIVE_SWAPCHAINED_EXT,
        .pViewport       = &vp,
        .pViewProjection = identity,
    };
    auto prim = RgMeshPrimitiveInfo{
        .sType                = RG_STRUCTURE_TYPE_MESH_PRIMITIVE_INFO,
        .pNext                = &sw,
        .flags                = RG_MESH_PRIMITIVE_TRANSLUCENT,
        .primitiveIndexInMesh = 0,
        .pVertices            = quadVerts,
        .vertexCount          = std::size( quadVerts ),
        .pTextureName         = MATERIAL_NAME_SCENEBUILDINGWARNING,
        .color                = Utils::PackColorFromFloat( 1, 1, 1, ratio ),
    };

    auto warnPlaque = RgMeshInfo{
        .sType          = RG_STRUCTURE_TYPE_MESH_INFO,
        .uniqueObjectID = 0,
        .pMeshName      = nullptr,
        .transform      = RG_TRANSFORM_IDENTITY,
    };

    UploadMeshPrimitive( &warnPlaque, &prim );
}
