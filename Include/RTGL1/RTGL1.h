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

#ifndef RTGL1_H_
#define RTGL1_H_

#include <stdint.h>

#if defined( _WIN32 )
    #define RGAPI_CALL __stdcall
    #define RGAPI_PTR  RGAPI_CALL
    #ifdef RG_LIBRARY_EXPORTS
        #define RGAPI __declspec( dllexport )
    #else
        #define RGAPI __declspec( dllimport )
    #endif // RTGL1_EXPORTS
    #define RGCONV __cdecl
#else
    #define RGAPI_CALL
    #define RGAPI_PTR
    #define RGAPI
    #define RGCONV
#endif // defined(_WIN32)

#define RG_RTGL_VERSION_API "001.006.000"

#ifdef RG_USE_SURFACE_WIN32
    #include <windows.h>
#endif // RG_USE_SURFACE_WIN32
#ifdef RG_USE_SURFACE_METAL
    #ifdef __OBJC__
@class CAMetalLayer;
    #else
typedef void CAMetalLayer;
    #endif
#endif // RG_USE_SURFACE_METAL
#ifdef RG_USE_SURFACE_WAYLAND
    #include <wayland-client.h>
#endif // RG_USE_SURFACE_WAYLAND
#ifdef RG_USE_SURFACE_XCB
    #include <xcb/xcb.h>
#endif // RG_USE_SURFACE_XCB
#ifdef RG_USE_SURFACE_XLIB
    #include <X11/Xlib.h>
#endif // RG_USE_SURFACE_XLIB

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t RgBool32;
#define RG_FALSE        0
#define RG_TRUE         1

// Doom64-RT: capacity of RgDrawFrameSmokeParams. The puffs ride in the global
// uniform rather than a storage buffer, so this is a hard limit and must match
// SMOKE_PUFF_MAX in Source/Generated/GenerateShaderCommon.py.
#define RG_MAX_SMOKE_PUFFS 128

// Doom64-RT: capacity of RgDrawFrameLightShaftParams. Same reasoning as the
// puffs above -- the indices ride in the global uniform, so this is a hard
// limit and must match VOLUME_SHAFT_LIGHT_MAX in
// Source/Generated/GenerateShaderCommon.py. Kept a multiple of four: the
// uniform packs them as uvec4, and the shader unpacks by [i >> 2][i & 3].
//
// 32 -> 64 when it turned out that a list this short could not REACH: a room
// full of ceiling fixtures fills every slot within a few metres of the camera
// and nothing further away is sent at all. Cost is one uint32 each in the
// uniform, and n evaluations per froxel in the shader's first pass -- ALU only,
// no rays, which is not what this feature costs.
#define RG_MAX_SHAFT_LIGHTS 64

typedef enum RgResult
{
    RG_RESULT_SUCCESS,
    RG_RESULT_SUCCESS_FOUND_MESH,
    RG_RESULT_SUCCESS_FOUND_TEXTURE,
    RG_RESULT_CANT_FIND_DYNAMIC_LIBRARY,
    RG_RESULT_CANT_FIND_ENTRY_FUNCTION_IN_DYNAMIC_LIBRARY,
    RG_RESULT_NOT_INITIALIZED,
    RG_RESULT_ALREADY_INITIALIZED,
    RG_RESULT_GRAPHICS_API_ERROR,
    RG_RESULT_INTERNAL_ERROR,
    RG_RESULT_CANT_FIND_SUPPORTED_PHYSICAL_DEVICE,
    RG_RESULT_FRAME_WASNT_STARTED,
    RG_RESULT_FRAME_WASNT_ENDED,
    RG_RESULT_WRONG_FUNCTION_CALL,
    RG_RESULT_WRONG_FUNCTION_ARGUMENT,
    RG_RESULT_WRONG_STRUCTURE_TYPE,
    RG_RESULT_ERROR_CANT_FIND_HARDCODED_RESOURCES,
    RG_RESULT_ERROR_CANT_FIND_SHADER,
    RG_RESULT_ERROR_MEMORY_ALIGNMENT,
    RG_RESULT_ERROR_NO_VULKAN_EXTENSION,
} RgResult;

typedef enum RgMessageSeverityFlagBits
{
    RG_MESSAGE_SEVERITY_VERBOSE = 1,
    RG_MESSAGE_SEVERITY_INFO    = 2,
    RG_MESSAGE_SEVERITY_WARNING = 4,
    RG_MESSAGE_SEVERITY_ERROR   = 8,
} RgMessageSeverityFlagBits;
typedef uint32_t RgMessageSeverityFlags;

typedef void ( *PFN_rgPrint )( const char*            pMessage,
                               RgMessageSeverityFlags flags,
                               void*                  pUserData );

typedef struct RgWin32SurfaceCreateInfo   RgWin32SurfaceCreateInfo;
typedef struct RgMetalSurfaceCreateInfo   RgMetalSurfaceCreateInfo;
typedef struct RgWaylandSurfaceCreateInfo RgWaylandSurfaceCreateInfo;
typedef struct RgXcbSurfaceCreateInfo     RgXcbSurfaceCreateInfo;
typedef struct RgXlibSurfaceCreateInfo    RgXlibSurfaceCreateInfo;

#ifdef RG_USE_SURFACE_WIN32
typedef struct RgWin32SurfaceCreateInfo
{
    HINSTANCE           hinstance;
    HWND                hwnd;
} RgWin32SurfaceCreateInfo;
#endif // RG_USE_SURFACE_WIN32

#ifdef RG_USE_SURFACE_METAL
typedef struct RgMetalSurfaceCreateInfo
{
    const CAMetalLayer* pLayer;
} RgMetalSurfaceCreateInfo;
#endif // RG_USE_SURFACE_METAL

#ifdef RG_USE_SURFACE_WAYLAND
typedef struct RgWaylandSurfaceCreateInfo
{
    struct wl_display*  display;
    struct wl_surface*  surface;
} RgWaylandSurfaceCreateInfo;
#endif // RG_USE_SURFACE_WAYLAND

#ifdef RG_USE_SURFACE_XCB
typedef struct RgXcbSurfaceCreateInfo
{
    xcb_connection_t*   connection;
    xcb_window_t        window;
} RgXcbSurfaceCreateInfo;
#endif // RG_USE_SURFACE_XCB

#ifdef RG_USE_SURFACE_XLIB
typedef struct RgXlibSurfaceCreateInfo
{
    Display*            dpy;
    Window              window;
} RgXlibSurfaceCreateInfo;
#endif // RG_USE_SURFACE_XLIB

typedef enum RgStructureType
{
    RG_STRUCTURE_TYPE_NONE                                = 0,
    RG_STRUCTURE_TYPE_INSTANCE_CREATE_INFO                = 1,
    RG_STRUCTURE_TYPE_MESH_INFO                           = 2,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_INFO                 = 3,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_PORTAL_EXT           = 4,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_TEXTURE_LAYERS_EXT   = 5,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_PBR_EXT              = 6,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_ATTACHED_LIGHT_EXT   = 7,
    RG_STRUCTURE_TYPE_MESH_PRIMITIVE_SWAPCHAINED_EXT      = 8,
    RG_STRUCTURE_TYPE_LIGHT_INFO                          = 9,
    RG_STRUCTURE_TYPE_LIGHT_DIRECTIONAL_EXT               = 10,
    RG_STRUCTURE_TYPE_LIGHT_SPHERICAL_EXT                 = 11,
    RG_STRUCTURE_TYPE_LIGHT_POLYGONAL_EXT                 = 12,
    RG_STRUCTURE_TYPE_LIGHT_SPOT_EXT                      = 13,
    RG_STRUCTURE_TYPE_LIGHT_ADDITIONAL_EXT                = 14,
    RG_STRUCTURE_TYPE_ORIGINAL_TEXTURE_INFO               = 15,
    RG_STRUCTURE_TYPE_START_FRAME_INFO                    = 16,
    RG_STRUCTURE_TYPE_DRAW_FRAME_INFO                     = 17,
    RG_STRUCTURE_TYPE_DRAW_FRAME_ILLUMINATION_PARAMS      = 19,
    RG_STRUCTURE_TYPE_DRAW_FRAME_VOLUMETRIC_PARAMS        = 20,
    RG_STRUCTURE_TYPE_DRAW_FRAME_TONEMAPPING_PARAMS       = 21,
    RG_STRUCTURE_TYPE_DRAW_FRAME_BLOOM_PARAMS             = 22,
    RG_STRUCTURE_TYPE_DRAW_FRAME_REFLECT_REFRACT_PARAMS   = 23,
    RG_STRUCTURE_TYPE_DRAW_FRAME_SKY_PARAMS               = 24,
    RG_STRUCTURE_TYPE_DRAW_FRAME_TEXTURES_PARAMS          = 25,
    RG_STRUCTURE_TYPE_DRAW_FRAME_POST_EFFECTS_PARAMS      = 27,
    RG_STRUCTURE_TYPE_LENS_FLARE_INFO                     = 28,
    RG_STRUCTURE_TYPE_CAMERA_INFO                         = 30,
    RG_STRUCTURE_TYPE_ORIGINAL_TEXTURE_DETAILS_EXT        = 31,
    RG_STRUCTURE_TYPE_CAMERA_INFO_READ_BACK_EXT           = 32,
    RG_STRUCTURE_TYPE_START_FRAME_RENDER_RESOLUTION_PARAMS  = 33,
    RG_STRUCTURE_TYPE_SPAWN_FLUID_INFO                      = 34,
    RG_STRUCTURE_TYPE_START_FRAME_FLUID_PARAMS              = 35,
    RG_STRUCTURE_TYPE_DRAW_FRAME_SMOKE_PARAMS               = 36,
    RG_STRUCTURE_TYPE_DRAW_FRAME_LIGHT_SHAFT_PARAMS         = 37,
    RG_STRUCTURE_TYPE_DRAW_FRAME_VOLUMETRIC_CLOUD_PARAMS    = 38,
} RgStructureType;

typedef enum RgTextureSwizzling
{
    RG_TEXTURE_SWIZZLING_NULL_ROUGHNESS_METALLIC,
    RG_TEXTURE_SWIZZLING_NULL_METALLIC_ROUGHNESS,
    RG_TEXTURE_SWIZZLING_OCCLUSION_ROUGHNESS_METALLIC,
    RG_TEXTURE_SWIZZLING_OCCLUSION_METALLIC_ROUGHNESS,
    RG_TEXTURE_SWIZZLING_ROUGHNESS_METALLIC,
    RG_TEXTURE_SWIZZLING_METALLIC_ROUGHNESS,
} RgTextureSwizzling;

typedef struct RgFloat2D
{
    float data[ 2 ];
} RgFloat2D;

typedef struct RgFloat3D
{
    float data[ 3 ];
} RgFloat3D;

typedef struct RgFloat4D
{
    float data[ 4 ];
} RgFloat4D;

typedef struct RgQuaternion
{
    float data[ 4 ];
} RgQuaternion;

typedef struct RgInstanceCreateInfo
{
    RgStructureType             sType;
    void*                       pNext;

    // Set to RG_RTGL_VERSION_API, for compatibility checks.
    const char*                 version;
    // Set to 'sizeof( RgInterface )', for compatibility checks.
    uint64_t                    sizeOfRgInterface;

    // Application name.
    const char*                 pAppName;
    // Application GUID. Generate it for your application and specify it here.
    const char*                 pAppGUID;

    // Exactly one of these surface create infos must be not null.
    RgWin32SurfaceCreateInfo*   pWin32SurfaceInfo;
    RgMetalSurfaceCreateInfo*   pMetalSurfaceCreateInfo;
    RgWaylandSurfaceCreateInfo* pWaylandSurfaceCreateInfo;
    RgXcbSurfaceCreateInfo*     pXcbSurfaceCreateInfo;
    RgXlibSurfaceCreateInfo*    pXlibSurfaceCreateInfo;

    // Folder for all resources.
    const char*                 pOverrideFolderPath;

    // Optional function to print messages from the library.
    // Requires "VulkanValidation" in the configuration file.
    PFN_rgPrint                 pfnPrint;
    // Custom user data that is passed to pfnUserPrint.
    void*                       pUserPrintData;
    RgMessageSeverityFlags      allowedMessages;

    // How many texture layers should be used to get albedo color for primary rays / indrect illumination.
    uint32_t                    primaryRaysMaxAlbedoLayers;
    uint32_t                    indirectIlluminationMaxAlbedoLayers;

    // How many vertices to allocate for static and replacements (load once) geometry.
    // Bytes allocated in VRAM: 2 * replacementsMaxVertexCount * sizeof(RgPrimitiveVertex)
    uint64_t                    replacementsMaxVertexCount;
    // How many vertices to allocate for dynamic (load each frame) geometry.
    // Bytes allocated in VRAM: 3 * dynamicMaxVertexCount * sizeof(RgPrimitiveVertex)
    uint64_t                    dynamicMaxVertexCount;

    RgBool32                    rayCullBackFacingTriangles;
    RgBool32                    allowTexCoordLayer1;
    RgBool32                    allowTexCoordLayer2;
    RgBool32                    allowTexCoordLayer3;
    // Which layer to interpret as a lightmap. Can be 1, 2 or 3.
    // Set to 0, if no lightmaps.
    uint32_t                    lightmapTexCoordLayerIndex;

    // Memory that must be allocated for vertex and index buffers of rasterized geometry.
    // It can't be changed after rgCreateInstance.
    // If buffer is full, rasterized data will be ignored
    uint32_t                    rasterizedMaxVertexCount;
    uint32_t                    rasterizedMaxIndexCount;
    // Apply gamma correction to packed rasterized vertex colors.
    RgBool32                    rasterizedVertexColorGamma;

    // Size of a cubemap side to render rasterized sky in.
    uint32_t                    rasterizedSkyCubemapSize;
    
    // If true, 'filter' in RgMaterialCreateInfo, RgCubemapCreateInfo
    // will set only magnification filter.
    RgBool32                    textureSamplerForceMinificationFilterLinear;
    RgBool32                    textureSamplerForceNormalMapFilterLinear;

    RgTextureSwizzling          pbrTextureSwizzling;

    RgBool32                    effectWipeIsUsed;

    // Used for exporting.
    // Up is also used for additional water flow calculations.
    RgFloat3D                   worldUp;
    RgFloat3D                   worldForward;
    // Used for exporting.
    // 1 game unit should correspond to (worldScale) meters.
    float                       worldScale;

    float                       importedLightIntensityScaleDirectional;
    float                       importedLightIntensityScaleSphere;
    float                       importedLightIntensityScaleSpot;
} RgInstanceCreateInfo;

typedef struct RgInterface RgInterface;

typedef RgResult( RGAPI_PTR* PFN_rgCreateInstance )( const RgInstanceCreateInfo* pInfo,
                                                     RgInterface*                pInterface );
typedef RgResult( RGAPI_PTR* PFN_rgDestroyInstance )();

// .exe should export these symbols to find 'D3D12Core.dll', otherwise D3D12 init might fail,
// because user's Windows might have an older D3D12Core which is incompatible with RTGL.
// 'pOverrideFolderPath' here should be the same as 'RgInstanceCreateInfo::pOverrideFolderPath'
#define RG_D3D12CORE_HELPER( pOverrideFolderPath )                                  \
    extern "C" __declspec( dllexport ) extern const uint32_t D3D12SDKVersion = 611; \
    extern "C" __declspec( dllexport ) extern const char*    D3D12SDKPath =         \
        ( pOverrideFolderPath "\\bin\\" ); 



// Row-major transformation matrix.
typedef struct RgTransform
{
    float       matrix[ 3 ][ 4 ];
} RgTransform;

typedef struct RgMatrix3D
{
    float       matrix[ 3 ][ 3 ];
} RgMatrix3D;

typedef struct RgExtent2D
{
    uint32_t    width;
    uint32_t    height;
} RgExtent2D;

// Struct is used to transform from NDC to window coordinates.
// x, y, width, height are specified in pixels. (x,y) defines top-left corner.
typedef struct RgViewport
{
    float       x;
    float       y;
    float       width;
    float       height;
    float       minDepth;
    float       maxDepth;
} RgViewport;

typedef uint32_t RgColor4DPacked32;
typedef uint32_t RgNormalPacked32;



typedef enum RgMeshPrimitiveFlagBits
{
    RG_MESH_PRIMITIVE_ALPHA_TESTED          = 1 << 0,
    RG_MESH_PRIMITIVE_TRANSLUCENT           = 1 << 1,
    RG_MESH_PRIMITIVE_SKY                   = 1 << 2,
    RG_MESH_PRIMITIVE_MIRROR                = 1 << 3,
    RG_MESH_PRIMITIVE_GLASS                 = 1 << 4,
    RG_MESH_PRIMITIVE_WATER                 = 1 << 5,
    RG_MESH_PRIMITIVE_DONT_GENERATE_NORMALS = 1 << 6,
    RG_MESH_PRIMITIVE_FORCE_EXACT_NORMALS   = 1 << 7,
    // If roughness is too small, act as a mirror (perfect reflection).
    RG_MESH_PRIMITIVE_MIRROR_IF_SMOOTH      = 1 << 8,
    // If roughness is too small, act as a glass (perfect reflection/refraction).
    RG_MESH_PRIMITIVE_GLASS_IF_SMOOTH       = 1 << 9,
    // Ignore refracting geometry behind this primitive.
    RG_MESH_PRIMITIVE_IGNORE_REFRACT_AFTER  = 1 << 10,
    RG_MESH_PRIMITIVE_ACID                  = 1 << 11,
    RG_MESH_PRIMITIVE_THIN_MEDIA            = 1 << 12,
    RG_MESH_PRIMITIVE_SKY_VISIBILITY        = 1 << 13,
    // If set, the first triangle is analyzed to make a decal.
    // Requires vertexCount >= 3.
    RG_MESH_PRIMITIVE_DECAL                 = 1 << 14,
    RG_MESH_PRIMITIVE_EXPORT_INVERT_NORMALS = 1 << 15,
    RG_MESH_PRIMITIVE_NO_SHADOW             = 1 << 16,
    RG_MESH_PRIMITIVE_NO_MOTION_VECTORS     = 1 << 17,
    // Doom64-RT: this primitive does not RECEIVE projected water caustics.
    // Set on sprites (enemies, items, the first-person weapon): a caustic is
    // light thrown across a surface, and a camera-facing billboard has no
    // surface for it to lie on -- it just tints the sprite.
    RG_MESH_PRIMITIVE_NO_WATER_CAUSTICS     = 1 << 18,
    // Doom64-RT: this primitive's own `emissive` WINS over the emissiveMult its
    // material carries in textures.json, instead of being overwritten by it.
    // Passing 0 makes the surface non-emissive; passing a smaller number than the
    // material's leaves it dimly emissive, which is how a lamp pane keeps enough
    // glow to bloom without also lighting its room a second time.
    //
    // Needed because emissiveMult is a property of the TEXTURE and Doom 64 uses
    // one texture for two different fixtures: SFLATAQ is a lamp ceiling in one
    // sector and a wall light strip in the next. When the lamp ceilings were
    // switched to real point lights, the glow had to come off them or the room
    // would be lit twice -- but zeroing it in textures.json also put out every
    // wall strip in the game. TextureMeta applies the material AFTER the caller
    // fills in RgMeshPrimitiveInfo::emissive, so the caller cannot simply pass 0;
    // this flag is what survives that override.
    RG_MESH_PRIMITIVE_EMISSIVE_OVERRIDE     = 1 << 19,
    // Doom64-RT: which LIQUID this water-flagged primitive is, as a 2-bit index
    // into stylizedLiquidTint[] / stylizedLiquidCrest[]. Doom 64 has four
    // liquids sharing one surface shader and differing only in colour:
    //   0 water (D64W1/W2, WFALL)  1 nukage (D64N1/N2, SFALL)
    //   2 sludge (D64S1/S2)        3 blood  (D64B1/B2, BFALL)
    // Only meaningful together with RG_MESH_PRIMITIVE_WATER; 0 (neither bit)
    // is water, so every existing caller keeps its behaviour.
    RG_MESH_PRIMITIVE_LIQUID_BIT0           = 1 << 19,
    RG_MESH_PRIMITIVE_LIQUID_BIT1           = 1 << 20,
    // Doom64-RT: lava. Not a liquid id -- lava does not use the water surface
    // path at all. This exists because the lava's emission has to be handled by
    // the renderer rather than baked: an 8-bit _e map caps at 1.0, screen
    // emission is _e * emissionMaxScreenColor (3), and the bloom threshold is
    // 16, so a lava flat physically cannot bloom no matter what is painted in
    // it. It also gives the flow animation somewhere to live.
    RG_MESH_PRIMITIVE_LAVA                  = 1 << 21,
    // Doom64-RT: SHADOW-ONLY geometry. In the acceleration structure and blocks
    // shadow rays, but is invisible to primary, reflection, refraction and
    // indirect rays -- the exact complement of RG_MESH_PRIMITIVE_NO_SHADOW.
    //
    // What it is for: a sprite is a camera-facing billboard with no thickness,
    // so when a light lies in the billboard's plane its shadow collapses to a
    // line, and because the quad turns to face the viewer the shadow's SHAPE
    // changes as the camera rotates -- which nothing physical does. The caller
    // fixes that by submitting invisible proxy quads at world-fixed angles and
    // letting them do the occluding.
    //
    // Lands on INSTANCE_MASK_RESERVED_0, which is absent from rayCullMaskWorld
    // (WORLD_0|1|2) and therefore from every other ray's cull mask by
    // construction; only rayCullMaskWorld_Shadow adds it back.
    RG_MESH_PRIMITIVE_SHADOW_ONLY           = 1 << 22,
    // Doom64-RT: this primitive is a camera-facing BILLBOARD. Set it and the
    // sprite material dials below apply instead of the world ones.
    RG_MESH_PRIMITIVE_SPRITE                = 1 << 23,
    // Doom64-RT: VOLUMETRIC CLOUDS, for RG_MESH_PRIMITIVE_SKY geometry only.
    // The clouds are marched once per frame into a world-direction map
    // (RgDrawFrameVolumetricCloudParams); these say what each sky draw does
    // with it. Neither flag does anything unless the clouds are enabled.
    //
    // SKY_CLOUDS: this is the BACKDROP (the dome). The fragment becomes
    // backdrop * transmittance + cloud radiance -- the clouds are composited
    // over it.
    RG_MESH_PRIMITIVE_SKY_CLOUDS            = 1 << 24,
    // SKY_BEHIND_CLOUDS: this sits behind the clouds but is drawn AFTER the
    // backdrop (a moon disc). Multiplied by transmittance only, so it is
    // occluded by cloud without adding the cloud a second time.
    //
    // A sky draw with neither flag (a lightning bolt) is in front of the
    // clouds and is left untouched.
    RG_MESH_PRIMITIVE_SKY_BEHIND_CLOUDS     = 1 << 25,
    // Doom64-RT: scale this primitive's ON-SCREEN emission by `emissive`, the way
    // the indirect path already does.
    //
    // The asymmetry this exists for is deliberate and documented in HitInfo.inl:
    // with an _e map, PRIMARY and REFLECTION emission is the RAW _e sample, and
    // emissiveMult is applied on the INDIRECT path only. So a caller can change how
    // much a painted fixture contributes to bounce light, and cannot change how
    // bright it LOOKS, at all.
    //
    // That is correct for nearly everything -- a painted lamp should look the same
    // brightness whatever the renderer is doing with its GI. It is wrong for a
    // fixture that is supposed to switch: MAP01's pre-exit pillar sweeps a light
    // around four bulb panels, and with the raw-_e rule the painted bulbs stay lit
    // at full strength through the whole cycle no matter what `emissive` says --
    // four permanently-on lamps with a light moving behind them.
    //
    // OPT-IN, one bit per primitive, so the rest of the game keeps the balance it
    // was tuned with. Combine with RG_MESH_PRIMITIVE_EMISSIVE_OVERRIDE, or
    // TextureMeta will replace `emissive` with the material's constant first.
    RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED = 1 << 26,
} RgMeshPrimitiveFlagBits;
typedef uint32_t RgMeshPrimitiveFlags;

typedef enum RgMeshInfoFlagBits
{
    RG_MESH_EXPORT_AS_SEPARATE_FILE = 1 << 0,
    RG_MESH_FIRST_PERSON            = 1 << 1,
    RG_MESH_FIRST_PERSON_VIEWER     = 1 << 2,
    // Force all primitives of this mesh to be a mirror. Useful for overriding replacement RgMeshPrimitiveFlags.
    RG_MESH_FORCE_MIRROR            = 1 << 3,
    // Force all primitives of this mesh to be a glass. Useful for overriding replacement RgMeshPrimitiveFlags.
    RG_MESH_FORCE_GLASS             = 1 << 4,
    // Force all primitives of this mesh to be a water. Useful for overriding replacement RgMeshPrimitiveFlags.
    RG_MESH_FORCE_WATER             = 1 << 5,
    // Force all primitives of this mesh to ignore refracting geometry behind this primitive. Useful for overriding replacement RgMeshPrimitiveFlags.
    RG_MESH_FORCE_IGNORE_REFRACT_AFTER      = 1 << 6,
} RgMeshInfoFlagBits;
typedef uint32_t RgMeshInfoFlags;

typedef struct RgPrimitiveVertex
{
    float                   position[ 3 ];
    RgNormalPacked32        normalPacked;
    float                   texCoord[ 2 ];
    RgColor4DPacked32       color;
    uint32_t                _pad0;
} RgPrimitiveVertex;

// Can be linked after RgMeshPrimitiveInfo.
typedef struct RgMeshPrimitivePortalEXT
{
    RgStructureType         sType;
    void*                   pNext;
    RgFloat3D               inPosition;
    RgFloat3D               inDirection;
    RgFloat3D               outPosition;
    RgFloat3D               outDirection;
} RgMeshPrimitivePortalEXT;

typedef enum RgTextureLayerBlendType
{
    RG_TEXTURE_LAYER_BLEND_TYPE_OPAQUE,
    RG_TEXTURE_LAYER_BLEND_TYPE_ALPHA,
    RG_TEXTURE_LAYER_BLEND_TYPE_ADD,
    RG_TEXTURE_LAYER_BLEND_TYPE_SHADE,
} RgTextureLayerBlendType;

typedef struct RgTextureLayer
{
    const RgFloat2D*        pTexCoord;
    const char*             pTextureName;
    RgTextureLayerBlendType blend;
    RgColor4DPacked32       color;
} RgTextureLayer;

// Can be linked after RgMeshPrimitiveInfo.
typedef struct RgMeshPrimitiveTextureLayersEXT
{
    RgStructureType         sType;
    void*                   pNext;
    RgTextureLayerBlendType baseLayerBlend;
    RgTextureLayer*         pLayer1;
    RgTextureLayer*         pLayer2;
    RgTextureLayer*         pLayer3;
} RgMeshPrimitiveTextureLayersEXT;

// Can be linked after RgMeshPrimitiveInfo.
typedef struct RgMeshPrimitivePBREXT
{
    RgStructureType         sType;
    void*                   pNext;
    // Multipliers for Roughness-Metallic texture.
    // If no texture present, multipliers are used directly as plain values.
    // Clamped to [0.0, 1.0]
    // Default: 1.0, if Roughness-Metallic texture exists
    //          0.0, otherwise
    float                   metallicDefault;
    // Default: 1.0
    float                   roughnessDefault;
} RgMeshPrimitivePBREXT;

// Can be linked after RgMeshPrimitiveInfo.
typedef struct RgMeshPrimitiveAttachedLightEXT
{
    RgStructureType         sType;
    void*                   pNext;
    float                   intensity;
    RgColor4DPacked32       color;
    RgBool32                evenOnDynamic;
} RgMeshPrimitiveAttachedLightEXT;

typedef enum RgMeshPrimitiveSwapchainedFlagBits
{
    RG_MESH_PRIMITIVE_SWAPCHAINED_DRAW_AS_LINES = 1,
} RgMeshPrimitiveSwapchainedFlagBits;
typedef uint32_t RgMeshPrimitiveSwapchainedFlags;

// To draw directly into a swapchain image, at full resolution.
// Can be linked after RgMeshPrimitiveInfo.
typedef struct RgMeshPrimitiveSwapchainedEXT
{
    RgStructureType                 sType;
    void*                           pNext;
    RgMeshPrimitiveSwapchainedFlags flags;
    const RgViewport*               pViewport;
    const float*                    pView;
    const float*                    pProjection;
    const float*                    pViewProjection;
} RgMeshPrimitiveSwapchainedEXT;

// Primitive is an indexed or non-indexed geometry with a material.
typedef struct RgMeshPrimitiveInfo
{
    RgStructureType             sType;
    void*                       pNext;
    RgMeshPrimitiveFlags        flags;
    uint32_t                    primitiveIndexInMesh;
    const RgPrimitiveVertex*    pVertices;
    uint32_t                    vertexCount;
    const uint32_t*             pIndices;
    uint32_t                    indexCount;
    const char*                 pTextureName;
    uint32_t                    textureFrame;
    // If alpha < 1.0, then RG_MESH_PRIMITIVE_TRANSLUCENT is assumed.
    RgColor4DPacked32           color;
    float                       emissive;
    // Default: 1.0
    float                       classicLight;
    // Doom64-RT: read ONLY when RG_MESH_PRIMITIVE_EMISSIVE_SCREEN_SCALED is set.
    // What this primitive feeds the GI (indirect path), in the material's units,
    // while `emissive` is then what it LOOKS like on the primary ray. Two values
    // because they are animated on different clocks: a fixture's painted bulbs
    // are delayed to meet the light they cast, which arrives late through the
    // denoiser history -- and the light itself must not be delayed with them, or
    // the two chase each other forever. Ignored otherwise; leave 0.
    float                       emissiveGi;
} RgMeshPrimitiveInfo;

// Mesh is a set of primitives.
typedef struct RgMeshInfo
{
    RgStructureType             sType;
    void*                       pNext;
    RgMeshInfoFlags             flags;
    // Object is an instance of a mesh.
    uint64_t                    uniqueObjectID;
    // Name and primitive index is used to override meshes.
    const char*                 pMeshName;
    RgTransform                 transform;
    // Set to true, if an object can be exported.
    // If RG_MESH_INFO_EXPORT_AS_SEPARATE_FILE is set, a separate 3D model file
    // will be found. If not set, a scene name is used.
    RgBool32                    isExportable;
    float                       animationTime;
    // Default: 1.0
    float                       localLightsIntensity;
} RgMeshInfo;

typedef RgResult( RGAPI_PTR* PFN_rgUploadMeshPrimitive )( const RgMeshInfo*          pMesh,
                                                          const RgMeshPrimitiveInfo* pPrimitive );



// Render specified vertex geometry, if 'pointToCheck' is not hidden.
typedef struct RgLensFlareInfo
{
    RgStructureType             sType;
    void*                       pNext;
    // Must be in world space.
    uint32_t                    vertexCount;
    const RgPrimitiveVertex*    pVertices;
    // Must not be null.
    uint32_t                    indexCount;
    const uint32_t*             pIndices;
    const char*                 pTextureName;
    // Point in the world space.
    RgFloat3D                   pointToCheck;
} RgLensFlareInfo;

typedef RgResult( RGAPI_PTR* PFN_rgUploadLensFlare )( const RgLensFlareInfo* pInfo );


typedef struct RgSpawnFluidInfo
{
    RgStructureType sType;
    void*           pNext;
    RgFloat3D       position;
    float           radius;
    RgFloat3D       velocity;
    // Each particle would have a random velocity in range
    // [ ( 1.0 - dispersionVelocity ) * velocity, velocity ]
    float           dispersionVelocity;
    // [0, 180]
    float           dispersionAngleDegrees;
    uint32_t        count;
} RgSpawnFluidInfo;
typedef RgResult( RGAPI_PTR* PFN_rgSpawnFluid )( const RgSpawnFluidInfo* pInfo );



// If provided, members are initialized in rgUploadCamera().
// Can be linked after RgCameraInfo.
typedef struct RgCameraInfoReadbackEXT
{
    RgStructureType sType;
    void*           pNext;
    float           view[ 16 ];
    float           projection[ 16 ];
    float           viewInverse[ 16 ];
    float           projectionInverse[ 16 ];
} RgCameraInfoReadbackEXT;

typedef uint32_t RgCameraFlags;

typedef struct RgCameraInfo
{
    RgStructureType sType;
    void*           pNext;
    RgCameraFlags   flags;
    RgFloat3D       position;
    RgFloat3D       up;
    RgFloat3D       right;
    float           fovYRadians;
    float           aspect;
    // Near and far planes for a projection matrix.
    float           cameraNear;
    float           cameraFar;
    // Optional view matrix ({ pView[0],pView[1],pView[2],pView[3] } is a column).
    const float     *pView;
} RgCameraInfo;

typedef RgResult( RGAPI_PTR* PFN_rgUploadCamera )( const RgCameraInfo* pInfo );



typedef enum RgLightAdditionalFlagBits
{
    // Use the light source for scattering.
    // Only one per scene is allowed.
    // If GLTF is used, this can be overwritten by a GLTF's sun.
    RG_LIGHT_ADDITIONAL_VOLUMETRIC                  = 1,
    // Multiply the light intensity by a parent mesh's intensity ''.
    RG_LIGHT_ADDITIONAL_APPLY_PARENT_MESH_INTENSITY = 2,
    RG_LIGHT_ADDITIONAL_LIGHTSTYLE                  = 4,
} RgLightAdditionalFlagBits;
typedef uint32_t RgLightAdditionalFlags;

// Can be linked after RgLightDirectionalEXT / RgLightSphericalEXT / RgLightPolygonalEXT /
// RgLightSpotEXT.
typedef struct RgLightAdditionalEXT
{
    RgStructureType             sType;
    void*                       pNext;
    RgLightAdditionalFlags      flags;
    // If flags contain RG_LIGHT_ADDITIONAL_LIGHTSTYLE,
    // light intensity is multiplied by ( pLightstyleValues8[ lightstyle ] / 255.0 )
    int                         lightstyle;
    // If non-empty, 'hashName' is used to calculate 'uniqueId' when imported from GLTF.
    // Use case: there are 2 animation frames represented by 2 gltf nodes:
    // if 'hashName' are not set or not the same, then when anim.frame is changed,
    // 'uniqueId' won't be identical, so path tracer won't be able to match these lights,
    // causing noise.
    char                        hashName[ 40 ];
} RgLightAdditionalEXT;

// Can be linked after RgLightInfo.
typedef struct RgLightDirectionalEXT
{
    RgStructureType     sType;
    void*               pNext;
    RgColor4DPacked32   color;
    // Luminous flux received by a surface, in lumen / m^2
    // (i.e. illuminance, in lux)
    float               intensity;
    RgFloat3D           direction;
    float               angularDiameterDegrees;
} RgLightDirectionalEXT;

// Can be linked after RgLightInfo.
typedef struct RgLightSphericalEXT
{
    RgStructureType     sType;
    void*               pNext;
    RgColor4DPacked32   color;
    // Luminous flux in lumen
    float               intensity;
    RgFloat3D           position;
    float               radius;
} RgLightSphericalEXT;

// Can be linked after RgLightInfo.
typedef struct RgLightPolygonalEXT
{
    RgStructureType     sType;
    void*               pNext;
    RgColor4DPacked32   color;
    // Luminous flux in lumen
    float               intensity;
    RgFloat3D           positions[ 3 ];
} RgLightPolygonalEXT;

// Can be linked after RgLightInfo.
typedef struct RgLightSpotEXT
{
    RgStructureType     sType;
    void*               pNext;
    RgColor4DPacked32   color;
    // Luminous flux in lumen
    float               intensity;
    RgFloat3D           position;
    RgFloat3D           direction;
    float               radius;
    // Inner cone angle. In radians.
    float               angleOuter;
    // Outer cone angle. In radians.
    float               angleInner;
} RgLightSpotEXT;

typedef struct RgLightInfo
{
    RgStructureType sType;
    void*           pNext;
    // Used to match the same light source from the previous frame.
    uint64_t        uniqueID;
    RgBool32        isExportable;
} RgLightInfo;

typedef RgResult( RGAPI_PTR* PFN_rgUploadLight)(const RgLightInfo* pInfo );



typedef enum RgSamplerFilter
{
    RG_SAMPLER_FILTER_AUTO,
    RG_SAMPLER_FILTER_LINEAR,
    RG_SAMPLER_FILTER_NEAREST,
} RgSamplerFilter;

typedef enum RgSamplerAddressMode
{
    RG_SAMPLER_ADDRESS_MODE_REPEAT,
    RG_SAMPLER_ADDRESS_MODE_CLAMP,
} RgSamplerAddressMode;

typedef enum RgFormat
{
    // TODO: expand
    RG_FORMAT_UNDEFINED        = 0,
    RG_FORMAT_R8_UNORM         = 9,
    RG_FORMAT_R8_SRGB          = 15,
    RG_FORMAT_R8G8B8A8_UNORM   = 37,
    RG_FORMAT_R8G8B8A8_SRGB    = 43,
    RG_FORMAT_B8G8R8A8_UNORM   = 44,
    RG_FORMAT_B8G8R8A8_SRGB    = 50,
} RgFormat;

typedef enum RgOriginalTextureInfoFlagBits
{
    RG_ORIGINAL_TEXTURE_INFO_FORCE_EXPORT_AS_EXTERNAL = 1,
} RgOriginalTextureInfoFlagBits;
typedef uint32_t RgOriginalTextureInfoFlags;

// Can be linked after RgOriginalTextureInfo.
typedef struct RgOriginalTextureDetailsEXT
{
    RgStructureType            sType;
    void*                      pNext;
    RgOriginalTextureInfoFlags flags;
    RgFormat                   format;
} RgOriginalTextureDetailsEXT;

typedef struct RgOriginalTextureInfo
{
    RgStructureType         sType;
    void*                   pNext;
    const char*             pTextureName;
    // R8G8B8A8 pixel data. Must be (size.width * size.height * 4) bytes.
    const void*             pPixels;
    RgExtent2D              size;
    RgSamplerFilter         filter;
    RgSamplerAddressMode    addressModeU;
    RgSamplerAddressMode    addressModeV;
} RgOriginalTextureInfo;

typedef RgResult( RGAPI_PTR* PFN_rgProvideOriginalTexture )( const RgOriginalTextureInfo* pInfo );
typedef RgResult( RGAPI_PTR* PFN_rgMarkOriginalTextureAsDeleted )( const char* pTextureName );



typedef enum RgRenderUpscaleTechnique
{
    RG_RENDER_UPSCALE_TECHNIQUE_LINEAR,
    RG_RENDER_UPSCALE_TECHNIQUE_NEAREST,
    RG_RENDER_UPSCALE_TECHNIQUE_AMD_FSR2,
    RG_RENDER_UPSCALE_TECHNIQUE_NVIDIA_DLSS,
} RgRenderUpscaleTechnique;

typedef enum RgFrameGenerationMode
{
    RG_FRAME_GENERATION_MODE_OFF,               // Completely unload frame generation logic
    RG_FRAME_GENERATION_MODE_WITHOUT_GENERATED, // Run, but don't present generated frames
    RG_FRAME_GENERATION_MODE_ON,                // Enable
} RgFrameGenerationMode;

typedef enum RgRenderSharpenTechnique
{
    RG_RENDER_SHARPEN_TECHNIQUE_NONE,
    RG_RENDER_SHARPEN_TECHNIQUE_NAIVE,
    RG_RENDER_SHARPEN_TECHNIQUE_AMD_CAS,
} RgRenderSharpenTechnique;

typedef enum RgRenderResolutionMode
{
    RG_RENDER_RESOLUTION_MODE_CUSTOM,
    RG_RENDER_RESOLUTION_MODE_ULTRA_PERFORMANCE,
    RG_RENDER_RESOLUTION_MODE_PERFORMANCE,
    RG_RENDER_RESOLUTION_MODE_BALANCED,
    RG_RENDER_RESOLUTION_MODE_QUALITY,
    RG_RENDER_RESOLUTION_MODE_NATIVE_AA,
} RgRenderResolutionMode;

// Can be linked after RgStartFrameInfo.
typedef struct RgStartFrameRenderResolutionParams
{
    RgStructureType          sType;
    void*                    pNext;
    RgRenderUpscaleTechnique upscaleTechnique;
    RgRenderResolutionMode   resolutionMode;
    RgFrameGenerationMode    frameGeneration;
    RgBool32                 preferDxgiPresent;
    RgRenderSharpenTechnique sharpenTechnique;
    // Used, if resolutionMode is RG_RENDER_RESOLUTION_MODE_CUSTOM
    RgExtent2D               customRenderSize;
    // If true, final image will be downscaled to 'pixelizedRenderSize' at the very end.
    // Needed, if pixelized look is needed, but the actual rendering should
    // be done in higher resolution.
    RgBool32                 pixelizedRenderSizeEnable;
    RgExtent2D               pixelizedRenderSize;
    // When true and upscaleTechnique is NVIDIA_DLSS: skip A-SVGF and run DLSS Ray
    // Reconstruction (denoise + upscale). Requires nvngx_dlssd.dll. Frame generation
    // should be OFF. Ignores this flag if Ray Reconstruction is unavailable.
    RgBool32                 rayReconstruction;
    // Doom64-RT: which DLSS render preset to ask NGX for, as the raw
    // NVSDK_NGX_DLSS_Hint_Render_Preset value -- 0 = Default (let the DLL pick),
    // 5 = E, 10 = J, 11 = K. RTGL1 hard-coded E, which on a DLSS 4 runtime
    // (310.x) forces the legacy CNN model; the SDK header marks E deprecated and
    // K "best image quality". Changing this re-creates the DLSS feature.
    uint32_t                 dlssPreset;
    // Doom64-RT: the DLSS-RR (Ray Reconstruction) twin of dlssPreset, as the
    // raw NVSDK_NGX_RayReconstruction_Hint_Render_Preset value. Only three are
    // meaningful on current runtimes: 0 = Default (the DLL picks, NVIDIA's
    // recommendation), 4 = D (default transformer), 5 = E (latest transformer,
    // what RTGL1 hard-coded before this was configurable). A-C were removed in
    // SDK 310.4.0 and 6..15 revert to default. Changing this re-creates the
    // RR feature, so it takes effect live.
    uint32_t                 dlssRrPreset;
} RgStartFrameRenderResolutionParams;

// Can be linked after RgStartFrameInfo.
typedef struct RgStartFrameFluidParams
{
    RgStructureType sType;
    void*           pNext;
    RgBool32        enabled;
    RgBool32        reset;
    RgFloat3D       gravity;
    RgFloat3D       color;
    uint32_t        particleBudget;
    float           particleRadius;
} RgStartFrameFluidParams;

typedef enum RgStaticSceneStatusFlagBits
{
    RG_STATIC_SCENE_STATUS_LOADED            = 1,
    RG_STATIC_SCENE_STATUS_NEW_SCENE_STARTED = 2,
    RG_STATIC_SCENE_STATUS_EXPORT_STARTED    = 4,
} RgStaticSceneStatusFlagBits;
typedef uint32_t RgStaticSceneStatusFlags;

typedef struct RgStartFrameInfo
{
    RgStructureType sType;
    void*           pNext;
    const char*     pMapName;
    RgBool32        ignoreExternalGeometry;
    RgBool32        vsync;
    RgBool32        hdr;
    RgBool32        allowMapAutoExport;
    // How much of the screen should be rendered in a lightmap mode.
    // In [0.0, 1.0]
    float           lightmapScreenCoverage;
    uint32_t        lightstyleValuesCount;
    const uint8_t*  pLightstyleValues8;
    RgStaticSceneStatusFlags* pResultStaticSceneStatus;
    float           staticSceneAnimationTime;
} RgStartFrameInfo;

typedef RgResult( RGAPI_PTR* PFN_rgStartFrame )( const RgStartFrameInfo* pInfo );



typedef enum RgSkyType
{
    RG_SKY_TYPE_COLOR,
    RG_SKY_TYPE_CUBEMAP,
    RG_SKY_TYPE_RASTERIZED_GEOMETRY,
} RgSkyType;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameTonemappingParams
{
    RgStructureType sType;
    void*           pNext;
    RgBool32        disableEyeAdaptation;
    float           ev100Min;
    float           ev100Max;
    float           luminanceWhitePoint;
    // A per channel adjustment, use <0 decrease, 0=no change, >0 increase.
    // Default: 0 0 0
    RgFloat3D       saturation;
    // One channel must be 1.0, the rest can be <= 1.0 but not zero.
    // Default: 1.0 1.0 1.0
    RgFloat3D       crosstalk;
    // Default: 0.1
    float           contrast;
    // Default: 1
    float           hdrBrightness;
    // Default: 0.1
    float           hdrContrast;
    // Default: 0.25 0.25 0.25
    RgFloat3D       hdrSaturation;
} RgDrawFrameTonemappingParams;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameSkyParams
{
    RgStructureType sType;
    void*           pNext;
    RgSkyType       skyType;
    // Used as a main color for RG_SKY_TYPE_COLOR.
    RgFloat3D       skyColorDefault;
    // The result sky color is multiplied by this value.
    float           skyColorMultiplier;
    float           skyColorSaturation;
    // A point from which rays are traced while using RG_SKY_TYPE_RASTERIZED_GEOMETRY.
    RgFloat3D       skyViewerPosition;
    // If sky type is RG_SKY_TYPE_CUBEMAP, this cubemap is used.
    const char*     pSkyCubemapTextureName;
    // Apply this transform to the direction when sampling a sky cubemap (RG_SKY_TYPE_CUBEMAP).
    // If equals to zero, then default value is used.
    // Default: identity matrix.
    RgMatrix3D      skyCubemapRotationTransform;
    // Doom64-RT: sky-reach test for the directional light.
    // A shadow ray that hits nothing is scored as LIT, and Doom maps are not
    // watertight, so rays escape at wall/ceiling seams and the sun washes rooms
    // with no opening. 1 = a ray must reach sky geometry to count as lit.
    float           sunRequireSky;
    // Diagnostic: 1 = paint by WHY a surface is lit. RED = ray reached sky
    // (real light through a real opening), GREEN = ray escaped into the void
    // (a leak). Independent of sunRequireSky, so leaks can be seen unfixed.
    float           sunLeakDebug;
    // Brightness of those debug colours; these rooms are near-black.
    float           sunLeakDebugMul;
    // Max length of the sky probe ray, in meters. Only traced when the normal
    // shadow ray already missed.
    float           sunSkyProbeMaxDist;
    // Doom64-RT: 1 = shade the directional light SEPARATELY from ReSTIR.
    // Normally the sun is merged into the per-pixel light reservoir
    // stochastically, so one light wins per pixel; a weak-but-huge sun loses
    // that draw on most pixels and its shadows come back sparse and get
    // denoised away. With this, every pixel facing the sun gets its own shadow
    // ray. Unbiased -- the light is removed from the candidate set, not counted
    // twice -- and costs one ray, unlike raising directSamples.
    float           sunSplit;
} RgDrawFrameSkyParams;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameTexturesParams
{
    RgStructureType sType;
    void*           pNext;
    // What sampler filter to use for materials with RG_MATERIAL_CREATE_DYNAMIC_SAMPLER_FILTER_BIT.
    // Should be changed infrequently, as it reloads all texture descriptors.
    RgSamplerFilter dynamicSamplerFilter;
    // Added to the texture mip LOD bias. RTGL follows the DLSS Programming
    // Guide §3.5 and uses log2(renderRes/outputRes) - 1.0, which at Balanced is
    // about -1.77 mips: textures are sampled far sharper than the render
    // resolution can represent. That is right for DLSS-SR, which turns the
    // resulting aliasing plus jitter into detail from a CLEAN image. DLSS-RR
    // gets the same over-sharp texture inside a NOISY albedo guide, and aliased
    // high-frequency detail is theoretically hard for a denoiser to tell from
    // noise. MEASURED 2026-08-07 at +1.0 and +1.8: does NOT fix the worm
    // artifact on distant detailed textures -- render resolution does (see
    // mipLodBiasOffset notes in rr-noise-investigation.md). Kept as a knob.
    // Positive values soften (e.g. +1.0 cancels the guide's -1.0 term).
    // Default: 0 (inert)
    float           mipLodBiasOffset;
    float           normalMapStrength;
    // Multiplier for emission map values for indirect lighting.
    float           emissionMapBoost;
    // Upper bound for emissive materials in primary albedo channel (i.e. on screen).
    float           emissionMaxScreenColor;
    // Default: 0.0
    float           minRoughness;
    // The deepest point that the 0.0 value of height map defines.
    // Default 0.02
    float           heightMapDepth;
    // Doom64-RT: force metallic = 0 on every surface, ignoring both the ORM map's
    // blue channel and metallicDefault. Same effect as the devmode "strip
    // metallic" toggle, but reachable by the application so it can sit behind a
    // cvar. Metals have no diffuse lobe, so a rough dark metal is the noisiest
    // and darkest thing in a 1-spp interior -- this is the escape hatch.
    // Default: 0 (metals as authored)
    RgBool32        forceNonMetallic;
    // Doom64-RT: ceiling on metalness, so every metal keeps at least
    // (1 - metallicMax) of its diffuse response and can never go fully black in
    // an unlit room. Default: 1 (inert)
    float           metallicMax;
    // Roughness at which metalness fades out, over a metallicRoughBand-wide band.
    // Very rough "metal" is usually oxide, paint or scale, all of which are
    // dielectrics -- so this is both physical and a safety net against a wrong
    // label. Default: 0 (disabled)
    float           metallicRoughCut;
    float           metallicRoughBand;
    // Doom64-RT: THE SPRITE SET, applied only where RG_MESH_PRIMITIVE_SPRITE is
    // set. A billboard is one quad carrying one normal, so its indirect specular
    // reflection vector is the SAME for every texel: the whole sprite samples the
    // room in a single direction and gains one flat colour across its body -- a
    // warm wall off to the side turns a soldier beige. Walls never do this, so
    // clamping both with one set of numbers cannot work.
    // spritePbr: 0 = ignore the authored _orm and _n on sprites entirely
    //            (metallic 0, roughness 1, geometric normal). The master OFF
    //            switch for the whole sprite material pass. Default: 1
    float           spritePbr;
    // Ceiling on a sprite's metalness. Default: 1 (inert)
    float           spriteMetallicMax;
    // FLOOR on a sprite's roughness -- the direct lever on the wash above, since
    // a broader lobe spreads that single reflection instead of mirroring it.
    // Default: 0 (inert)
    float           spriteRoughMin;
    // How much of a sprite's normal map to apply. Sprite normals are DERIVED
    // from the silhouette rather than authored, so this exists to dial back a
    // guess. Default: 1
    float           spriteNormalStrength;
    // Doom64-RT: the same mix for WALLS AND FLATS. 1 = materials exactly as
    // authored, 0 = plain dielectric. Sprites have their own dial because their
    // failure mode is their own; this one exists so the world can be dialled
    // back for the same reason -- less specular means less for the denoiser to
    // resolve. Default: 1 (inert)
    float           worldPbr;
    // Doom64-RT: how the ALBEDO of ray-traced surfaces is filtered.
    // 0 = the hardware sampler (i.e. dynamicSamplerFilter: nearest or bilinear).
    // 1 = N64-style three-point filtering: the texel quad is split along its
    //     diagonal and the three nearest texels are blended, the look of the
    //     original console. Done in the shader at a whole mip level, so pair it
    //     with RG_SAMPLER_FILTER_NEAREST. Normal/ORM/emissive maps are unaffected.
    // Default: 0
    uint32_t        albedoFilterMode;
} RgDrawFrameTexturesParams;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameIlluminationParams
{
    RgStructureType sType;
    void*           pNext;
    // Shadow rays are cast, if illumination bounce index is in [0, maxBounceShadows).
    uint32_t        maxBounceShadows;
    // Doom64-RT: number of path vertices after the primary hit for indirect
    // diffuse, clamped to [1..4] inside. 1 = a single bounce, 2 = the stock
    // two. A vertex at index >= maxBounceShadows samples NO analytic lights
    // (RaygenCommon.h isDirectIlluminationValid), so to light depth N pass
    // maxBounceShadows >= N + 1 -- the caller owns that coupling.
    // Replaces enableSecondBounceForIndirect, which nothing ever read.
    // Default: 2
    uint32_t        indirectBounces;
    // Doom64-RT: if true, bounces >= 2 keep the stock weighting -- radiance
    // multiplied by 1/pdf alone, which for a cosine-sampled Lambertian is
    // pi/cos too much (mean ~2pi): the second bounce came out bright,
    // saturated and firefly-prone. False applies the correct throughput of
    // exactly 1. Default: true, so the shipped image does not move until the
    // fix has been judged on its own.
    RgBool32        indirectLegacyBounceWeight;
    // Size of the side of a cell for the light grid. Use RG_DEBUG_DRAW_LIGHT_GRID_BIT for the debug view.
    // Each cell is used to store a fixed amount of light samples that are important for the cell's center and radius.
    // Default: 1.0
    float           cellWorldSize;
    // If 0.0, then the change of illumination won't be checked, i.e. if a light source suddenly disappeared,
    // its lighting still will be visible. But if it's 1.0, then lighting will be dropped at the given screen region
    // and the accumulation will start from scratch.
    // Default: 0.5
    float           directDiffuseSensitivityToChange;
    // Default: 0.2
    float           indirectDiffuseSensitivityToChange;
    // Default: 0.5
    float           specularSensitivityToChange;
    // The higher the value, the more polygonal lights act like a spotlight. 
    // Default: 2.0
    float           polygonalLightSpotlightFactor;
    // For which light first-person viewer shadows should be ignored.
    // E.g. first-person flashlight.
    // Null, if none.
    const uint64_t* lightUniqueIdIgnoreFirstPersonViewerShadows;
    // DLSS-RR: run A-SVGF temporal accumulation before ComposeNoisy and feed those
    // buffers to RR (spatial atrous still skipped). Stabilizes flickering lights.
    // NOTE: currently INERT -- AccumulateForRR() is never called, VulkanDevice
    // always takes ComposeNoisy() on the RR path. Kept for API compatibility.
    RgBool32        enableRrTemporalPrefilter;
    // DLSS-RR: write a disocclusion mask that forces RR to discard its temporal
    // history where scene lighting changed sharply vs the previous frame
    // (motion-reprojected, per 16x16 tile). Fixes transient lights (barrel
    // explosions, muzzle flashes, occluded glows) lingering for seconds.
    // Default: true
    RgBool32        enableRrDisocclusionMask;
    // Tile-mean luminance ratio (current vs previous, symmetric) above which
    // RR history is discarded. Lower = more responsive, noisier. Default: 3.0
    float           rrDisocclusionThreshold;
    // Additional absolute tile-mean luminance delta required to fire, to avoid
    // false positives in near-black areas. Default: 0.01
    float           rrDisocclusionMinDelta;
    // Debug: tint pixels red where the disocclusion mask fired. Default: false
    RgBool32        rrDisocclusionShowMask;
    // DLSS-RR: neighbourhood firefly clamp on the noisy lighting in
    // ComposeNoisy, before it reaches NGX. The RR path has no spatial or
    // temporal prefilter of any kind (A-SVGF gets anti-firefly + a
    // variance-driven atrous), so isolated 1-spp spikes go to RR raw; Remix
    // runs an equivalent clamp. A pixel is scaled down only when its luminance
    // exceeds this multiple of the BRIGHTEST of its 4 spatial neighbours, so
    // genuinely bright regions (whose neighbours are also bright) are left
    // alone and only isolated spikes are touched.
    // Lower = more aggressive. 0 disables. Default: 4.0
    float           rrFireflyThreshold;
    // Absolute luminance floor below which the clamp is skipped, so ratios are
    // not evaluated in near-black areas where they are meaningless.
    // Default: 0.01
    float           rrFireflyMinLum;
    // Place ReSTIR's temporal/spatial reuse taps with tiled blue noise instead
    // of hash white noise. With white noise the taps clump and neighbouring
    // pixels reuse overlapping neighbourhoods, correlating their estimates into
    // low-frequency blotching that no denoiser can separate from signal. Blue
    // noise spreads them, leaving a high-frequency, spatially even residual --
    // which is also what DLSS-RR requires (decorrelated reservoirs, guide 3.5).
    // Reduces variance at the source, so it helps A-SVGF and RR alike.
    // Default: true
    RgBool32        restirBlueNoise;
    // Shadow rays per pixel for DIRECT illumination (clamped to [1,8]).
    // The direct estimate multiplies by a single binary traceVisibility(), so
    // that 0/1 term dominates 1-spp variance and is untouched by anything
    // ReSTIR does. Averaging N independent points on the chosen light turns it
    // into a soft fraction; variance falls roughly as 1/sqrt(N). Only the
    // visibility factor is averaged -- shading keeps the reservoir's own
    // sample, so the RIS estimator and expected energy are unchanged.
    // Costs N-1 extra rays per pixel on the direct pass. 1 = stock.
    // Default: 1
    uint32_t        shadowSamples;
    // Debug: write the ReSTIR reservoir's accumulated sample count M into the
    // unfiltered-direct buffer instead of radiance (green ramp, M/32). ReSTIR
    // at 1 spp only converges because temporal reuse grows M; if M collapses
    // under camera motion the raw signal is genuinely noisier while moving,
    // upstream of any denoiser. 1 = the DIRECT reservoir (RtRaygenDirect.rgen),
    // 2 = the INDIRECT reservoir, painted into the unfiltered-indirect buffer
    // (RtRaygenIndirect.inl) -- view it with the unfiltered-indirect debug
    // layer. Default: 0
    uint32_t        debugRestirM;
    // Debug: write the shadow-ray visibility term into the unfiltered-direct
    // buffer. The final image cannot separate "the occluder never blocked the
    // ray" from "the shadow is cast but drowned in fill light or smeared by the
    // denoiser" -- both read as no shadow. Visibility is the only thing a shadow
    // ray produces, so this shows that and nothing else.
    // 0 = off, 1 = greyscale (black where shadowed), 2 = normal shading with
    // shadowed pixels tinted red. Default: 0
    uint32_t        debugVisibility;
    // Doom64-RT: raw DEBUG_SHOW_FLAG_* bits OR'd into the debug view, so a
    // layer can be isolated from a cvar instead of the dev window. 1 motion,
    // 2 gradients, 4 unfiltered direct, 8 unfiltered spec, 16 unfiltered
    // indirect, 32 only direct diffuse, 64 only specular, 128 only indirect,
    // 1024 normals. Default 0.
    uint32_t        debugShowFlags;
    // ReSTIR temporal reuse tap jitter radius in pixels. Stock 2.0. The jitter
    // decorrelates the temporal tap, but on grazing surfaces a 2px offset moves
    // depth far past the flat 10% reuse threshold, so the tap is rejected and M
    // (and with it 1-spp convergence) collapses exactly where variance is worst.
    // 0 reprojects exactly, as RTXDI does. Default: 2.0
    float           restirTemporalJitter;
    // Feed DLSS-RR pInSpecularHitDistance: the world distance from the shading
    // point to whatever produced the highlight. Specular does not live ON the
    // surface, so without this RR reprojects highlights as if it did and glossy
    // surfaces smear/fizzle under camera motion. Shipping RR integrations all
    // provide it. Default: true
    RgBool32        rrSpecularHitDistance;

    // --- Samples per pixel (Doom64-RT) --------------------------------------
    // The path tracer is 1 spp: convergence is supplied entirely by temporal
    // accumulation (ReSTIR reservoir M + the denoiser's history), and camera
    // motion legitimately destroys both, leaving the raw 1-spp signal exposed.
    // These trade GPU time for a quieter signal at the SOURCE, which is upstream
    // of the denoiser choice and therefore helps A-SVGF and DLSS-RR equally.
    // Both clamped to [1,8]. 1 = stock behaviour.

    // Independent direct-lighting estimates per pixel, averaged. Each costs one
    // extra shadow ray plus RIS/reuse math (no extra rays for the candidates --
    // calcInitialReservoir traces none outside the initial pass). Default: 1
    uint32_t        directSamples;
    // Independent indirect (GI) paths per pixel, RIS-combined into the single
    // initial reservoir. Each costs ~2 bounce rays + 2 shadow rays, so this is
    // the expensive one. Default: 1
    uint32_t        indirectSamples;

    // --- ReSTIR quality (previously hardcoded) ------------------------------
    // RIS candidate lights considered when building the initial reservoir.
    // Traces no rays -- pure importance-sampling quality. Clamped [1,32].
    // Default: 8
    uint32_t        restirInitialSamples;
    // Spatial reuse taps in the direct pass. Image reads, no rays.
    // Clamped [0,16]. Default: 8
    uint32_t        restirSpatialSamples;
    // Radius in pixels of those spatial taps. Clamped [1,64]. Default: 30
    float           restirSpatialRadius;
    // Cap on accumulated temporal M, as a multiple of the initial reservoir's M.
    // Higher = longer history = smoother when still, slower to react.
    // Clamped [1,64]. Default: 20
    uint32_t        restirTemporalMCap;
    // DLSS-RR albedo-guide floor. RR demodulates colour by the diffuse/specular
    // albedo guides; as a guide approaches zero that division explodes, which
    // shows as correlated dark filaments in dim, dark-albedo regions (and is
    // undefined on metallic surfaces, where ro_d is exactly 0). Flooring is
    // near-free because RR remodulates with the same guide. The sky path has
    // always done this (RR guide 3.4.2); the world path did not.
    // 0 disables the floor (pre-fix behaviour). Default: 0.01
    float           rrGuideMin;
    // What the DLSS-RR albedo guides contain. RR uses them as edge-detection and
    // reprojection weights as well as for demodulation, so they want to be
    // smooth material properties.
    //   0 = raw material (albedo / F0)  - behaviour before 2026-08-05
    //   1 = ro_d/envBRDF * throughput * ambient - full demodulation (default)
    //   2 = ro_d/envBRDF only, no throughput or ambient
    // Default: 1
    uint32_t        rrGuideMode;
    // Apply the A-SVGF antilag gate to INDIRECT ReSTIR temporal reuse.
    // The gradient buffer it reads is written only by Denoiser::Denoise(),
    // which DLSS-RR skips -- so under RR the gate reads stale data and may be
    // rejecting GI temporal reuse every frame. 0 ignores the gate. Default: 1
    uint32_t        restirIndirAntilag;
    // Doom64-RT: feed DLSS-RR PRE-exposure radiance. CmPrepareFinal skips its
    // EV100 multiply and screen-emissive add; CmRrPostExposure reapplies both
    // on the upscaled output. NVIDIA's RR guide (S3.7) declares exposure
    // unsupported for RR -- with exposure baked in, RR's temporal history
    // mixed frames captured at different exposures whenever auto-exposure
    // adapted. Only takes effect on frames where the RR branch actually runs;
    // A-SVGF / DLSS-SR / FSR2 paths are untouched. Default: true
    RgBool32        rrPreExposure;
    // Debug: CmRrPostExposure tints its output magenta, so "no visible
    // difference" and "the pass never ran" stop being the same image.
    // Default: false
    RgBool32        rrPreExposureDebug;
    // Doom64-RT: bind the 1x1 GPU-written exposure texture to DLSS-RR
    // (pInExposureTexture). Only meaningful with rrPreExposure: the network's
    // input is then raw radiance whose absolute scale swings ~52x with
    // auto-exposure, and the network is not scale-invariant -- this is the
    // SDK's mechanism for telling it the scale. Ignored when rrPreExposure is
    // off. Default: true
    RgBool32        rrExposureTexture;
    // Doom64-RT: route rasterized content (every translucent sprite in the
    // game, particles, lens flares) into DLSS-RR's transparency layer
    // (pInTransparencyLayer, NGX-composited after denoise+upscale) instead of
    // baking it into the denoiser's colour input, where every guide describes
    // the opaque surface BEHIND it. RR frames only; all other paths keep
    // rasterizing into the final image. Default: true
    RgBool32        rrTransparencyLayer;
    // Doom64-RT: the NVIDIA NRD lane (ReBLUR/ReLAX/SIGMA via NRI wrapping the
    // existing VkDevice). Stage 1: setting this brings the full NRD stack up
    // and reports liveness in the log; the actual denoising data path is
    // stage 2 and A-SVGF keeps running until it lands. Ignored while DLSS-RR
    // is active (precedence: RR > NRD > A-SVGF). Default: false
    RgBool32        nrdDenoiser;
    // Doom64-RT: where the screen-emission glow goes under rrPreExposure.
    //   0 = added after RR (jitter-corrected Catmull-Rom; residual shimmer
    //       on sharp emissive edges is irreducible in this mode)
    //   1 = in RR's input, divided by live exposure (exact brightness, but
    //       the input pulses while auto-exposure adapts)
    //   2 = in RR's input at the FIXED rrGlowScale ("glow as light": stable
    //       input, no shimmer, glow rides the exposure like the emissive
    //       surface it halos)
    // Only read while rrPreExposure is active. Default: 0
    uint32_t        rrGlowPre;
    // Doom64-RT: mode 2's artistic glow multiplier; the mode divides by a
    // smoothed exposure, so 1.0 = the old calibrated brightness everywhere.
    // Default: 1
    float           rrGlowScale;
    // Doom64-RT: DLSS-RR albedo demodulation (the Remix approach) -- RR
    // denoises lighting only; the modulation factor carrying the crisp
    // albedo/texel content is re-applied after RR at output resolution.
    // Default: true. rrDemodFilter: 0 bilinear, 1 Catmull-Rom, 2 nearest.
    RgBool32        rrDemod;
    uint32_t        rrDemodFilter;
    // Doom64-RT: NRD lane -- paint NRD's OUT_VALIDATION overlay instead of
    // the image (vendor-supplied guide/reprojection sanity view). Only
    // meaningful while nrdDenoiser is active. Default: false
    RgBool32        nrdValidation;
    // Doom64-RT: ReLAX tuning, pushed to NRD every frame (live-tunable).
    // Zeros mean "use ReLAX defaults". See nrd::RelaxSettings for semantics.
    uint32_t        nrdMaxAccumFrames;      // default 30
    uint32_t        nrdFastAccumFrames;     // default 6
    uint32_t        nrdAtrousIterations;    // [2..8], default 5
    float           nrdPrepassDiffuse;      // pixels, default 30
    float           nrdPrepassSpecular;     // pixels, default 50
    float           nrdPhiLuminance;        // diffuse a-trous luminance sensitivity, default 2
    float           nrdMinHitDistWeight;    // (0;0.2], default 0.1; smaller for ReSTIR
    RgBool32        nrdAntiFirefly;         // default true
    // Doom64-RT: the first-person weapon vs the SURFACE denoiser's history.
    // svgfFp: 0 = stock A-SVGF (a pixel the gun just uncovered restarts from
    // one sample and reads as a dark silhouette beside lagging neighbours
    // under a decaying light), 1 = fully rejected pixels borrow validated
    // neighbour history, 2 = debug tint. svgfFpGrad: the gradient never
    // samples the weapon and a vanished light counts as a change. Default 0.
    uint32_t        svgfFp;
    RgBool32        svgfFpGrad;
    // Doom64-RT: indirect history cap in frames (0 = stock 256) and the
    // indirect antilag without its brightness suppression. The bounce of a
    // transient light otherwise ghosts for seconds after the light is gone.
    float           svgfIndirMaxHist;
    RgBool32        svgfIndirAntilag;
} RgDrawFrameIlluminationParams;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameVolumetricParams
{
    RgStructureType sType;
    void*           pNext;
    RgBool32        enable;
    // Default: 8.0
    float           maxHistoryLength;
    // If true, volumetric illumination is not calculated, just
    // using simple depth-based fog with ambient color.
    RgBool32        useSimpleDepthBased;
    // Farthest distance for volumetric illumination calculation.
    // Should be minimal to have better precision around camera.
    // Default: 100.0
    float           volumetricFar;
    RgFloat3D       ambientColor;
    // Default: 0.2
    float           scaterring;
    // g parameter [-1..1] for the Henyey-Greenstein phase function.
    // Default: 0.0 (isotropic)
    float           assymetry;
    // If true, maintain a world-space grid, each cell of which contains
    // illumination, that is used for scattering.
    RgBool32        useIlluminationVolume;
    // If light source is not provided (RgLightExtraInfo::isVolumetric), use this fallback info.
    // If color is too dim or direction is invalid,
    // then no light sources for scattering.
    RgFloat3D       fallbackSourceColor;
    RgFloat3D       fallbackSourceDirection;
    // Multiplier for light for scattering.
    float           lightMultiplier;
    RgBool32        allowTintUnderwater;
    RgFloat3D       underwaterColor;
    // Doom64-RT: ILLUMINATED FOG.
    //
    // The stock froxel pass scatters exactly ONE light -- the one
    // LightManager::TryGetVolumetricLight picks (a RG_LIGHT_ADDITIONAL_VOLUMETRIC
    // light if any, otherwise the sun). On a map with no sun and no volumetric
    // light, the fog therefore receives nothing at all and is flat ambient.
    // When this is set, each froxel runs the full NEE/ReSTIR direct estimate
    // instead, so every torch, lava pool and monitor lights the fog it sits in.
    // Requires RTGL to be built with ILLUMINATION_VOLUME.
    RgBool32        illuminateFromAllLights;
    // Scattering albedo of the medium: multiplies the in-scattered radiance,
    // ambient and lit alike, so the fog and everything glowing inside it take
    // this hue. Extinction stays monochrome, so distant geometry fades TOWARD
    // this colour rather than being colour-filtered by it.
    // Default (and the no-op value): { 1, 1, 1 }.
    //
    // This is the NEAR end of a near->far ramp; the three fields below are the
    // far end. The froxel grid's slices are uniform in distance, so separating
    // the two costs one mix() per cell.
    RgFloat3D       mediaColor;
    // Tint and density at the far plane of the volume. Set equal to
    // mediaColor / scaterring for a uniform medium.
    RgFloat3D       mediaColorFar;
    float           farScattering;
    // Shape of the near->far interpolation. 1 = linear, > 1 holds the near
    // value longer and thickens late, < 1 thickens immediately. Default: 1.
    float           densityCurve;
    // Attenuate SCREEN EMISSION by the medium's transmittance. Without it an
    // emissive surface shines through fog and smoke at full strength. Default:
    // false (stock).
    RgBool32        occludeEmission;
    // Per-pixel volume sample jitter ACROSS THE SCREEN, in FROXELS. Stock 2.
    // Lower values reduce the dark outlines dense media draw around geometry
    // silhouettes, at the cost of showing more of the grid. Default: 2.
    float           ditherRadius;
    // The same jitter ALONG THE VIEW, in FROXELS, and a separate number because
    // the depth axis is not the same problem. The jitter is drawn from a
    // hemisphere and negated, so its depth component is always toward the
    // camera -- never past the surface, which is what keeps it from reading a
    // froxel column belonging to geometry behind it. That makes it a BIAS, not
    // a dither: the volume is a prefix sum from the camera outward, so a
    // consistently shallower sample loses the far end of every column, at a
    // mean of 0.33 * radius * (volumetricFar / VOLUMETRIC_SIZE_Z) metres.
    // Depth only has to break slice banding, so keep this sub-froxel.
    // Default: 1.
    float           ditherRadiusZ;
    // 0..1 blend of a 3x3 spatial blur applied to the froxel volume before it
    // is integrated along Z. The volume is estimated at one sample per cell and
    // has no spatial filter at all, so at 0 its variance goes to the screen raw.
    // Default: 0 (stock behaviour).
    float           spatialBlur;
    // Fade volumetric in-scattering out within this many metres OF A LIGHT.
    // A light at the camera lights the froxels in front of it by inverse
    // square and whites out the screen -- physically what a headlight in fog
    // does, and unplayable. 0 = no fade (physical). Default: 0.
    float           lightNearFade;
    // Weight each froxel by how much of it lies IN FRONT OF the geometry the
    // camera can see, so the volume stops lighting air behind a wall.
    //
    // The volume is a prefix sum stored at cell CENTRES and read TRILINEARLY at
    // the surface's distance, so a wall collects part of the cell BEHIND it --
    // where the froxel legitimately sees the next room's lamp. And the sample's
    // z is CLAMPED to [0,1], so a surface nearer than slice 0's centre
    // (cameraNear + 0.0078 * volumetricFar) collects slice 0 whole, which is why
    // the artefact peaks with the camera against a wall. Every shadow ray
    // involved is already correct; the wall just falls inside a cell shaded for
    // its far side, so no per-light test can reach it.
    // 0 = off (stock). Default: 0.
    float           depthGate;
    // Metres of slack beyond the surface before a cell is weighted down.
    // 0 centres the ramp on the surface itself. Default: 0.
    float           depthGateBias;
    // Width of that ramp in FROXEL SLICES. 1 makes a cell the surface bisects
    // contribute about half, which is what a straddling cell physically
    // deserves; larger is softer and leaks more, 0 is a hard cut that shows the
    // grid. Default: 1.
    float           depthGateFeather;
    // Depth taps across the froxel column's screen footprint; the MAXIMUM wins.
    // 1 = centre only, 5 = centre + corners. A column spans many pixels which
    // can see very different depths, so the max is what keeps air visible past a
    // thin foreground edge alive -- the centre tap alone cuts a hard edge along
    // every silhouette. Default: 5.
    uint32_t        depthGateTaps;
    // Doom64-RT: THE UPSCALER BIAS MASK. The thin dark line at every edge seen
    // through a medium is drawn by the temporal upscaler, not by the froxel
    // grid: the composite happens at render resolution and the upscaler runs
    // after it, so a silhouette arrives with two different media states already
    // baked into either side of it and nothing in the upscaler's inputs says
    // the discontinuity is the medium's. Both DLSS and FSR2 take a mask meaning
    // "favour the current frame over history here"; these fill it in.
    // 0 = off, the mask is written as zero and both upscalers behave as before.
    // Default: 0.
    float           volumeUpscaleBias;
    // Transmittance difference across a pixel that counts as a full-strength
    // edge. Keeps the mask on silhouettes: biasing toward the current frame
    // buys the outline back in noise, and the veil is where the smoke's noise
    // lives. Default: 0.05.
    float           volumeUpscaleBiasEdge;
    // Constant bias wherever the medium is present at all. 0 = silhouettes
    // only, which is the intent. Default: 0.
    float           volumeUpscaleBiasFloor;
    // 1 = draw the mask on screen instead of the image, because a mask handed
    // to a black box is otherwise unobservable. Default: 0.
    uint32_t        volumeUpscaleBiasDebug;
    // Doom64-RT: apply the medium AFTER the upscaler instead of before it. The
    // outline is the upscaler reconstructing surface and medium added together;
    // this stops handing it the sum. Algebra is preserved exactly, but it
    // FORCES emissive occlusion (that factor moves into the post pass), and it
    // is ignored on paths that would break it -- DLSS Ray Reconstruction, and
    // frame generation, which interpolates frames inside its own technique
    // where a Vulkan-side post pass cannot reach. Default: 0.
    uint32_t        volumePostComp;
    // Doom64-RT: soften the medium's step across a silhouette by this many
    // pixels before compositing. Mitigation for the paths postcomp cannot take;
    // unlike the bias mask it never touches temporal history. Default: 0.
    float           volumeEdgeSoft;
    // Doom64-RT: how big a relative depth break counts as a silhouette for
    // volumeEdgeSoft, as the second difference of 1/depth. Lower marks more
    // edges. Default: 0.15.
    float           volumeEdgeSoftEdge;
    // Doom64-RT: the first-person weapon must not damage the medium. 0 = old
    // path, 1 = fix (the accumulator keeps the WORLD's fog alive under the
    // sprite and the composite paints no fog on the gun), 2 = debug. Default 0.
    float           volumeFp;
    // Doom64-RT: how the medium's temporal history is validated. 0 = surface
    // rules (strict depth + normal), 1 = media rules (relative path length
    // within 25%, no normal test). Default 0.
    uint32_t        volumeReproj;
    // Doom64-RT: do sprites shadow the froxel medium? 0 = no (billboards and
    // their shadow proxies are invisible to volumetric shadow rays), 1 = the
    // old behaviour. Default 1 to stay bit-identical for old callers.
    uint32_t        volumeSpriteShadow;
    // Doom64-RT: in-grid temporal accumulation length, in frames. 0 = off
    // (legacy screen-space accumulation). > 0 = EMA the froxel grid in world
    // space and recompute the ray integral fresh every frame -- the structural
    // fix for every history-restart artefact at silhouettes. Default 0.
    float           volumeGridHistory;
} RgDrawFrameVolumetricParams;

// Doom64-RT: LOCALISED SMOKE.
//
// The volumetric medium above is one density for the whole level, and the
// froxel grid is camera-fitted, so there is no way to say "there is smoke
// HERE". This struct adds a small list of world-space spheres whose density is
// ADDED to that medium inside RtVolumetric.rgen; everything downstream is
// untouched, because CmVolumetricProcess.comp is a straight front-to-back
// prefix sum over whatever the raygen wrote and gives the puffs correct
// occlusion and transmittance for free.
//
// It is deliberately a SEPARATE struct rather than more fields on
// RgDrawFrameVolumetricParams: the fog is shipped and tuned, and a struct that
// does not change size cannot break a caller that does not know about smoke.
// A frame that never links this gets puffCount 0, which collapses the shader
// arithmetic back to exactly the fog's.
//
// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameSmokeParams
{
    RgStructureType  sType;
    void*            pNext;
    // Number of entries in the two arrays below. Clamped to
    // RG_MAX_SMOKE_PUFFS. Default: 0 (no smoke, no cost).
    uint32_t         puffCount;
    // xyz = centre in world space (the same space as RgLightInfo positions and
    // as the froxel centres, i.e. metres), w = radius in metres.
    const RgFloat4D* pPuffs;
    // rgb = scattering albedo of this puff, a = density at its core, already
    // scaled by the volume's slice thickness so it reads as optical depth per
    // metre rather than per cell. Parallel to pPuffs.
    const RgFloat4D* pAlbedoDensity;
    // x = the puff's radius ACROSS the view in metres; yzw unused. Parallel to
    // pPuffs, whose .w is the radius ALONG the view.
    //
    // The two differ because the froxel grid's two axes differ by a factor of
    // forty: at 1.5 m one cell is 1.7 cm across the screen but 47 cm deep. A
    // SPHERE therefore cannot be thin -- to be resolved in depth at all it needs
    // a radius of half a slice, and that same radius is what you then see. Making
    // the puff an ellipsoid stretched along the view axis buys a filament that is
    // genuinely centimetres wide on screen while still landing on a cell centre,
    // and the stretch is invisible because it points down the axis you are
    // looking along. Null = spherical, using pPuffs.w for both.
    const RgFloat4D* pShape;
    // The near-light fade (RgDrawFrameVolumetricParams::lightNearFade) applied
    // to froxels that CONTAIN smoke, instead of the fog's value. A muzzle flash
    // lighting the smoke at the barrel is the entire point of the effect, and
    // the fog's 2 m fade would erase it. Chosen per froxel, so fog cells keep
    // the fog's value exactly. Default: 0 (no fade inside smoke).
    float            lightNearFade;
    // Temporal blend factor for the all-lights estimate in froxels that contain
    // smoke, instead of the stock 0.05. A muzzle flash lasts 2-3 frames; at
    // 0.05 the volume needs ~0.7 s to respond and as long to let go, so the
    // smoke would light up after the flash and then linger. Higher is more
    // responsive and noisier. Default: 0.05 (the stock value).
    float            illumBlend;
    // Run the full all-lights direct estimate in froxels that CONTAIN smoke, so
    // a puff is lit by the muzzle flash that made it.
    //
    // Deliberately NOT RgDrawFrameVolumetricParams::illuminateFromAllLights:
    // that one switches the whole volume off the single-light path, and the
    // single-light path is the only place the sun's sky-probe test lives -- so
    // setting it because a puff exists deletes every light shaft in the level
    // for as long as the player is shooting. This is read per froxel.
    // Default: false.
    RgBool32         allLights;
    // Metres beyond which a light stops lighting SMOKE. Smoke runs the all-lights
    // estimate while the rest of the volume runs the single-light path, so a puff
    // picks up every emissive in the room; at one sample per froxel a saturated
    // one far away arrives as a colour cast over the whole puff rather than as a
    // tint. Faded smoothly over the last quarter of the range. 0 = no limit.
    float            lightFarFade;
    // Direct-lighting samples per froxel INSIDE smoke; 1 = stock. Each costs one
    // NEE sample and one shadow ray, but only in cells a puff covers, which is a
    // few per cent of the grid. Default: 1.
    uint32_t         samplesPerCell;
    // Ceiling on in-scattered radiance inside smoke. A light carried at ~0 m
    // floods the froxels around it by inverse square and a dense puff in front
    // of the camera turns pure white. 0 = no clamp. Default: 0.
    float            maxLight;
    // 0 off. 2 paints the froxels a puff covers; 3 paints every froxel whenever
    // the list is non-empty. Diagnostic only. Default: 0.
    uint32_t         debugMode;
    // PIXEL-ART STYLIZATION, 0..1. The volume gives a puff a smooth exponential
    // falloff, which is physically right and reads as an airbrushed blob against
    // art made entirely of hard-edged pixels. This posterizes that falloff into
    // `stylizeSteps` bands. Applied inside the puff evaluation, so with no puffs
    // it changes nothing at all. Default: 0.
    float            stylize;
    // Band count for `stylize`. Few is chunky; many approaches smooth. Default: 4.
    uint32_t         stylizeSteps;
    // Metres of WORLD-space voxel snapping for the puff field, 0 = off. World
    // rather than screen space on purpose: screen-space blocks crawl when the
    // camera turns, which reads as noise instead of style. Default: 0.
    float            stylizeGrid;
    // Smoke's OWN unlit floor, applied per froxel inside puffs only. The
    // volume's ambientColor is one per-frame value for the whole grid, so it
    // cannot be raised for smoke without raising it for fog; this is the
    // smoke-only equivalent, and it is what makes a puff visible after the
    // muzzle flash that lit it has gone. Scaled by puff density. Default: 0.
    float            selfAmbient;
    // How hard a smoke cell asserts its own albedo against the medium it is
    // blended into, 0..1. The blend is density-weighted, so a thin puff comes
    // out mostly the room's colour and stops reading as smoke; this biases it
    // back. 0 = the plain weighted average. Default: 0.
    float            tintBias;
    // Absorption as a fraction of the puff's density, smoke only. The medium is
    // otherwise purely scattering, which adds as much light as it removes and
    // leaves a thin puff with no contrast under even lighting. Absorption
    // darkens what is behind the puff without adding light, so smoke can read
    // against a background brighter than itself. Default: 0.
    float            absorb;
} RgDrawFrameSmokeParams;

// Doom64-RT: LIGHT SHAFTS FROM ORDINARY LAMPS.
//
// The froxel pass scatters exactly ONE light -- whatever
// LightManager::TryGetVolumetricLight picked, which in practice is the sun.
// That is why this renderer can only produce visible beams outdoors: every
// ceiling lamp, grate and doorway inside a level is a light with no air around
// it. RgDrawFrameVolumetricParams::illuminateFromAllLights exists and does the
// opposite of what is wanted here -- it replaces the single-light path with a
// stochastic all-lights estimate, which drops the sun's sky probe (and so every
// shaft the map already has), shades the medium with a fake normal that scores
// dot ~ 0 for a light directly overhead, and costs a temporal history to
// denoise.
//
// So this is a small EXPLICIT list instead: the caller decides which fixtures
// deserve air around them and hands over their uniqueIDs, and the shader adds
// their scattering on top of the single-light term. Selection is a policy
// question about a particular game's art, and it belongs on the caller's side.
//
// The list is per frame and is expected to be culled and sorted by the caller
// (nearest first): everything past `count` is simply not scattered.
//
// Cost is per FROXEL, not per pixel -- one shadow ray per qualifying light per
// cell, over a 64-slice grid. `maxTraced` and `minRadiance` are the budget, and
// they are what keeps this affordable: in a typical cell only one or two lamps
// are close enough to contribute at all.
//
// Separate struct rather than more fields on RgDrawFrameVolumetricParams, for
// the same reason RgDrawFrameSmokeParams is: the fog is shipped and tuned, and
// a struct that does not change size cannot break a caller that does not know
// about shafts. A frame that never links this gets count 0, and the volume
// behaves exactly as it did.
//
// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameLightShaftParams
{
    RgStructureType sType;
    void*           pNext;
    // Number of entries in pLightUniqueIds. Clamped to RG_MAX_SHAFT_LIGHTS.
    // Default: 0 -- no shafts, no cost, the stock single-light volume.
    uint32_t        count;
    // uniqueIDs of lights uploaded this frame (RgLightInfo::uniqueID). An ID
    // that is not found among this frame's lights is skipped silently: a
    // fixture may legitimately have been culled after the caller listed it.
    const uint64_t* pLightUniqueIds;
    // Multiplier on the scattering these lights contribute, applied on top of
    // RgDrawFrameVolumetricParams::lightMultiplier. Its own knob because a
    // lamp's shaft wants to be tunable without moving the sun's.
    // Default: 1.
    float           multiplier;
    // Fade in-scattering out within this many metres OF A LIGHT, exactly as
    // RgDrawFrameVolumetricParams::lightNearFade does for the fog's own pass.
    // Point lights sitting IN the medium are the whole feature here, so this
    // matters more than it does there: without it the froxels touching a bulb
    // saturate and the shaft reads as a white ball. 0 = physical. Default: 0.
    float           nearFade;
    // Skip a light whose unshadowed in-scattered radiance at this froxel is
    // below this, BEFORE tracing its shadow ray. This is the cull that makes
    // the loop cheap -- it is a few ALU per light against one ray -- and it is
    // in the same units as the radiance the shader accumulates, so it is best
    // found by A/B rather than derived. Default: 0.
    float           minRadiance;
    // Hard cap on shadow rays per froxel, independent of `count`. The list is
    // walked nearest-first, so this spends the budget on the lights that
    // actually reach the cell. Default: 4.
    uint32_t        maxTraced;
    // Henyey-Greenstein asymmetry [-1..1] for THESE lights only. The sun's
    // shafts are tuned at RgDrawFrameVolumetricParams::assymetry and its strong
    // forward bias is what carries them; a lamp overhead is seen from every
    // angle at once and generally wants less. Below -1 means "use the
    // volumetric params' value". Default: -2 (share it).
    float           asymmetry;
    // 0 off. 1 paints a froxel red wherever a shaft light survives the radiance
    // cull, before any shadow ray; 2 the same but only where it is also
    // unshadowed; 3 paints every froxel green whenever the list is non-empty,
    // with no position or visibility test at all. 1-vs-2 separates "no light
    // reaches here" from "an occluder blocks it"; 3 answers "is the uniform
    // being read", which no amount of caller-side logging can. Default: 0.
    uint32_t        debugMode;
    // How much of the inverse-square falloff to hand back, as an exponent:
    // radiance *= pow( max( distanceToLight, 1 ), this ). 0 = physical, 1 = 1/d,
    // 2 = no falloff beyond a metre.
    //
    // It exists because a sphere light's weight is a solid angle, so a lamp's
    // scattering is inverse square and dies 36x over from 1 m to 6 m -- it reads
    // as a puddle of light around the bulb rather than as a beam. The shafts a
    // renderer like this already has come from the SUN, which is directional and
    // does not fall off at all; that difference, not the brightness, is why one
    // reads across a level and the other does not. Default: 0 (physical).
    float           falloffCompensation;
    // Skip a light whose radiance at this froxel is below this FRACTION of the
    // brightest candidate at the same froxel, in addition to minRadiance.
    //
    // `maxTraced` is a per-froxel budget and something has to decide who spends
    // it. An absolute floor cannot: the question is not "is this light bright"
    // but "is it worth a ray compared to the others reaching this cell". Without
    // a relative bar the budget goes in list order, and a caller that sorts by
    // distance to the CAMERA will starve every froxel that is not next to it.
    // 0 disables it. Default: 0.05.
    float           relativeCull;
} RgDrawFrameLightShaftParams;

// Doom64-RT: VOLUMETRIC CLOUDS. Can be linked after RgDrawFrameInfo.
//
// A cloud slab between `altitude` and `altitude + thickness` metres above the
// sky viewer, ray-marched ONCE per frame into a lat-long map of world
// directions (not per screen pixel -- the sky is rasterised twice, once at
// render resolution into the albedo and once into the 256^2 GI cubemap, and
// the same map feeds both). RsSky.frag composites the map over sky draws
// flagged RG_MESH_PRIMITIVE_SKY_CLOUDS and occludes draws flagged
// RG_MESH_PRIMITIVE_SKY_BEHIND_CLOUDS.
//
// The map is a WORLD-DIRECTION image, so its temporal history (historyBlend)
// is keyed by direction alone: no motion vectors, no depth, no reprojection,
// and therefore nothing a sprite in front of the sky can invalidate. That is
// deliberate -- the froxel volume's screen-space history was what sprites
// smeared (see the caller's rt-volumetric-weapon-trails.md).
//
// Output radiance is in the same units as the sky texture (0..1, scaled by
// RgDrawFrameSkyParams::skyColorMultiplier downstream), so a tint of 1,1,1 on
// a white-lit cloud lands at the brightness the sky art would.
typedef struct RgDrawFrameVolumetricCloudParams
{
    RgStructureType sType;
    void*           pNext;
    // 0 = off: the map is not marched and the sky draws are untouched.
    RgBool32        enabled;
    // Bottom of the slab above the sky viewer, metres. Default 600.
    float           altitude;
    // Slab thickness, metres. Default 500.
    float           thickness;
    // 0..1, fraction of the sky covered. Default 0.55.
    float           coverage;
    // Extinction per metre inside a fully dense cell. Default 0.01.
    float           density;
    // Size of the largest noise feature, metres. Default 1500.
    float           featureSize;
    // How much the high-frequency erosion eats into the shapes, 0..1. Default 0.5.
    float           detail;
    // Wind vector, metres per second, in the sky viewer's horizontal plane.
    RgFloat2D       wind;
    // Seconds, for the wind. Pass the caller's playsim time so weather stops
    // when the level does.
    float           time;
    // View-ray steps through the slab and light steps toward the light.
    // Default 32 and 6.
    uint32_t        steps;
    uint32_t        lightSteps;
    // The cloud's own colour (scattering albedo). Default 1,1,1.
    RgFloat3D       tint;
    // Direction TOWARD the light (the moon) and its colour at the cloud,
    // in sky-texture units. A light below the horizon is ignored.
    RgFloat3D       lightDir;
    RgFloat3D       lightColor;
    // A second light from BELOW -- a burning ground. Colour in sky-texture
    // units, strength 0 = off. Default 0.
    RgFloat3D       underColor;
    float           underStrength;
    // Ambient added to every cloud sample, sky-texture units. Default 0.
    RgFloat3D       ambient;
    // Henyey-Greenstein asymmetry for the light. Default 0.3.
    float           asymmetry;
    // What a FULLY opaque column still transmits, 0..1 (the same meaning as
    // the shell deck's transmit). Default 0.
    float           transmitFloor;
    // Altitude (degrees) below which the clouds fade to nothing, so the slab's
    // far edge does not end as a hard line. Default 4.
    float           horizonFade;
    // Temporal blend of the previous map, 0..0.95. Default 0.8.
    float           historyBlend;
    // 0 normal; 1 paint transmittance; 2 paint density only; 3 bypass the
    // composite (map still marched).
    uint32_t        debugMode;
    // PER-RAY OCCLUSION OF THE DIRECTIONAL LIGHT. Every shadow ray toward the
    // directional light samples the cloud map where it would cross the slab
    // (parallax-corrected at the slab's middle) and attenuates the light by
    // T + (1 - T) * lightTransmit * tint, blended toward 1 by (1 -
    // lightOcclude). Surfaces, indirect bounces and the froxel volume all go
    // through the same sample, so shafts get real gaps between clouds and
    // the ground gets cloud shadows that drift with the wind.
    // While this is on the caller should NOT also dim the light globally.
    RgBool32        sunOcclusion;
    // What a fully opaque column still passes of the light, 0..1.
    float           lightTransmit;
    // 0..1, how much of the occlusion to apply. Default 1.
    float           lightOcclude;
    // A glow BEHIND the slab, sky-texture units. Composited as glow *
    // transmittance over backdrop draws: thin cloud burns with it, dense
    // cloud is a silhouette against it, clear sky IS it. This is what a
    // burning sky looks like (dark cloud against fire), and it is not the
    // same thing as underColor, which lights the underside evenly.
    // Default 0,0,0.
    RgFloat3D       backColor;
    // Strength of the backColor glow behind the slab. Default 0.
    float           backStrength;
    // FIRE POCKETS: a second, sparse noise field inside the slab. Where it
    // is above its threshold the cloud gas itself GLOWS in backColor along
    // the ray (fire in the gaps and thin edges) and lights the cloud around
    // it -- the "fire between the clouds". fireStrength is the emission
    // (sky units, 0 = off), fireScale the pocket feature size in metres,
    // fireCover the fraction of the slab that burns (0..1), fireLit how
    // strongly a pocket lights the cloud near it.
    float           fireStrength;
    float           fireScale;
    float           fireCover;
    float           fireLit;
    // TWO DECKS. layers = 2 splits the slab into a lower deck, a gap of
    // gapFraction of the thickness, and an upper deck; the fire field
    // becomes a SHEET in the gap -- an emissive burning layer between the
    // clouds that lights the upper deck's underside and the lower deck's
    // tops, each through its own optical depth toward it. layers = 1 is the
    // single slab with fire pockets. Default 1, 0.2.
    uint32_t        layers;
    float           gapFraction;
    // Extinction of the sheet itself relative to the cloud's density, so
    // looking through fire is a haze rather than glass. Default 0.3.
    float           sheetExtinction;
    // The fire PULSES: each burning patch breathes slowly (firePulse, 0..1
    // amplitude, at firePulseSpeed) and flickers fast on top (fireFlicker).
    // Applied to the emission and to the light the fire casts on the decks.
    // Default 0, 1, 0.
    float           firePulse;
    float           firePulseSpeed;
    float           fireFlicker;
    // FLAME CASCADES: the fire raining DOWN from the cloud base as thin
    // vertical streaks, seeded under the burning parts of the sheet and
    // falling at cascadeSpeed m/s through a band cascadeLength metres below
    // the base. cascadeStrength is the emission (0 = off), cascadeCover the
    // fraction of the base that rains (0..1), cascadeWidth the streak width
    // in metres. Default 0, 400, 0.3, 80, 40.
    float           cascadeStrength;
    float           cascadeLength;
    float           cascadeCover;
    float           cascadeSpeed;
    float           cascadeWidth;
} RgDrawFrameVolumetricCloudParams;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameBloomParams
{
    RgStructureType sType;
    void*           pNext;
    // EV value to adjust bloom inputs
    float           inputEV;
    float           inputThreshold;
    // Scale to apply to a calculated bloom
    // Negative value disables bloom pass
    float           bloomIntensity;
    float           lensDirtIntensity;
} RgDrawFrameBloomParams;

typedef struct RgPostEffectWipe
{
    // [0..1] where 1 is whole screen width.
    float       stripWidth;
    RgBool32    beginNow;
    float       duration;
} RgPostEffectWipe;

typedef struct RgPostEffectRadialBlur
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectRadialBlur;

typedef struct RgPostEffectChromaticAberration
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
} RgPostEffectChromaticAberration;

typedef struct RgPostEffectInverseBlackAndWhite
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectInverseBlackAndWhite;

typedef struct RgPostEffectHueShift
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectHueShift;

typedef struct RgPostEffectNightVision
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectNightVision;

typedef struct RgPostEffectDistortedSides
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectDistortedSides;

typedef struct RgPostEffectWaves
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       amplitude;
    float       speed;
    float       xMultiplier;
} RgPostEffectWaves;

typedef struct RgPostEffectColorTint
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
    RgFloat3D   color;
} RgPostEffectColorTint;

typedef struct RgPostEffectCRT
{
    RgBool32    isActive;
} RgPostEffectCRT;

typedef struct RgPostEffectVHS
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
} RgPostEffectVHS;

typedef struct RgPostEffectDither
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
    float       intensity;
} RgPostEffectDither;

typedef struct RgPostEffectTeleport
{
    RgBool32    isActive;
    float       transitionDurationIn;
    float       transitionDurationOut;
} RgPostEffectTeleport;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFramePostEffectsParams
{
    RgStructureType                         sType;
    void*                                   pNext;
    // Must be null, if effectWipeIsUsed was false.
    const RgPostEffectWipe*                 pWipe;
    const RgPostEffectRadialBlur*           pRadialBlur;
    const RgPostEffectChromaticAberration*  pChromaticAberration;
    const RgPostEffectInverseBlackAndWhite* pInverseBlackAndWhite;
    const RgPostEffectHueShift*             pHueShift;
    const RgPostEffectNightVision*          pNightVision;
    const RgPostEffectDistortedSides*       pDistortedSides;
    const RgPostEffectWaves*                pWaves;
    const RgPostEffectColorTint*            pColorTint;
    const RgPostEffectTeleport*             pTeleport;
    const RgPostEffectCRT*                  pCRT;
    const RgPostEffectVHS*                  pVHS;
    const RgPostEffectDither*               pDither;
} RgDrawFramePostEffectsParams;

typedef enum RgMediaType
{
    RG_MEDIA_TYPE_VACUUM,
    RG_MEDIA_TYPE_WATER,
    RG_MEDIA_TYPE_GLASS,
    RG_MEDIA_TYPE_ACID,
} RgMediaType;

// Can be linked after RgDrawFrameInfo.
typedef struct RgDrawFrameReflectRefractParams
{
    RgStructureType sType;
    void*           pNext;
    uint32_t        maxReflectRefractDepth;
    // Media type, in which camera currently is.
    RgMediaType     typeOfMediaAroundCamera;
    // Default: 1.52
    float           indexOfRefractionGlass;
    // Default: 1.33
    float           indexOfRefractionWater;
    float           thinMediaWidth;
    float           waterWaveSpeed;
    float           waterWaveNormalStrength;
    // Color at 1 meter depth.
    RgFloat3D       waterColor;
    // Color at 1 meter depth.
    RgFloat3D       acidColor;
    float           acidDensity;
    // The lower this value, the sharper water normal textures.
    // Default: 1.0
    float           waterWaveTextureDerivativesMultiplier;
    // The larger this value, the larger the area one water texture covers.
    // If equals to 0.0, then default value is used.
    // Default: 1.0
    float           waterTextureAreaScale;
    // If true, portal normal will be twirled around its 'inPosition'.
    RgBool32        portalNormalTwirl;
    // Doom64-RT: stylized water. Water surfaces stop refracting and instead
    // stay opaque, shaded as a deep blue body carrying the flat's own caustic
    // veins, with a Fresnel-weighted mirror reflection on top.
    // 0 = off (stock physical water).
    float           stylizedWaterStrength;
    // How hard the animated wave crests brighten the caustic veins.
    float           stylizedWaterCaustic;
    // Fresnel clamp for the reflection half. 1.0 = a true mirror at grazing.
    float           stylizedWaterReflMax;
    // Roughness written for the water surface (specular sheen from lights).
    float           stylizedWaterRoughness;
    // Unlit on-screen sheen on the veins. Casts no light.
    float           stylizedWaterGlow;
    // Luminance of the flat's brightest texel; normalizes the vein mask.
    float           stylizedWaterVeinRef;
    // Doom64-RT: body / crest colour per LIQUID, indexed by the 2-bit
    // RG_MESH_PRIMITIVE_LIQUID_BIT* value. [0] is water and is what a caller
    // that never sets those bits gets, so this is the old stylizedWaterTint.
    //   body  -- the colour the surface tends to where the flat is black
    //   crest -- the colour the caustic veins tend to at full mask (pale, but
    //            NOT white: white crests read as foam/plastic)
    RgFloat3D       stylizedLiquidTint[ 4 ];
    RgFloat3D       stylizedLiquidCrest[ 4 ];
    // Doom64-RT: how much of the MATERIAL normal survives on a liquid surface,
    // per liquid id. 0 = the animated water wave, exactly as before. 1 = the
    // authored _n only, i.e. a still surface with real relief -- which is what
    // a coagulated blood pool is, and what the ripple cannot be.
    //
    // The wave is not merely cosmetic here: getNormal() OVERWRITES the
    // normal-mapped normal with it for any water surface, so without this a
    // liquid can never show an _n map at all.
    float           stylizedLiquidRelief[ 4 ];
    // Doom64-RT: depth of the FLOW MAP on each liquid -- a detail texture
    // advected along the vein direction baked into the height map's .g/.b, so
    // texture visibly travels down each channel. 0 = still.
    float           stylizedLiquidFlow[ 4 ];
    // Doom64-RT: per-liquid scale on the stylized reflection F. 1 = the
    // stylizedWaterRefl{Min,Max} curve untouched. Below 1 for an opaque bed
    // like mud, where a mirror reflection is what makes it read as water.
    // EXACTLY 0 = no mirror ray and no checkerboard split at all: the surface
    // is shaded on every pixel and its sheen is the standard glossy specular.
    float           stylizedLiquidRefl[ 4 ];
    // Doom64-RT: per-liquid surface roughness. <= 0 = use
    // stylizedWaterRoughness, so a liquid that does not set it is unchanged.
    float           stylizedLiquidRough[ 4 ];
    // Doom64-RT: != 0 forces the no-split path for EVERY stylized liquid,
    // regardless of stylizedLiquidRefl. Full-res surface, no mirror ray.
    float           liquidNoSplit;
    // Doom64-RT: per-liquid scale on the caustics this liquid PROJECTS onto the
    // geometry around it. A caustic is light refracted through a fluid and
    // focused beyond it, so an opaque liquid casts none. 1 = as before.
    float           stylizedLiquidCaustics[ 4 ];
    // The detail is sampled in a vein-aligned frame (u along the channel and
    // scrolling, v across it): speed = detail tiles scrolled per second,
    // scale = detail tiles per liquid tile, aspect = across-vein frequency
    // multiplier that stretches the noise into lengthwise streaks.
    float           liquidFlowSpeed;
    float           liquidFlowScale;
    float           liquidFlowAspect;
    // 1 = paint the advected detail on liquid surfaces. Flat blue = it never
    // arrived, which by eye is identical to "too subtle".
    float           liquidFlowDebug;
    // Doom64-RT lava. Screen-emission multiplier for lava surfaces, on top of
    // emissionMaxScreenColor -- this is what lets the cracks reach the bloom
    // threshold. The rest animate the heat: a slowly drifting field that is
    // quantized to a world-space cell so it stays as chunky as the art.
    float           lavaEmisBoost;
    float           lavaFlowStrength;
    float           lavaFlowSpeed;
    float           lavaFlowScale;
    float           lavaFlowPixel;
    float           lavaPulse;
    float           lavaPulseSpeed;
    // Indirect multiplier: the lava lighting the room as an AREA source, which
    // is what it is. Analytic point lights over a lake give each one its own
    // pool and a visible circle on the wall as the player walks past.
    float           lavaGiBoost;
    // 1 = paint lava surfaces magenta.
    float           lavaDebug;

    // Hue of the heat, applied to both the screen emission and the GI.
    RgFloat3D       lavaTint;
    // Diagnostic: paint water surfaces magenta (stylized branch running) or
    // green (RTGL sees water, stylized gate rejected). 0 = off.
    float           stylizedWaterDebug;
    // Reflection strength at normal incidence (looking straight down).
    // Physical water is 0.02; raise it to make the surface read as reflective.
    float           stylizedWaterReflMin;
    // Caustics projected from water onto the geometry around it. 0 = off
    // (and no probe ray is traced).
    float           waterCausticGain;
    float           waterCausticScale;
    float           waterCausticSpeed;
    // How far below a surface the water may be and still light it, world units.
    float           waterCausticDist;
    // How far ABOVE the water the caustics still reach, world units.
    float           waterCausticRise;
    // How far the probe tilts along the surface normal. 0 = straight down.
    float           waterCausticSlant;
    // Extra gain applied to VERTICAL receivers only (walls). 1 = no boost.
    float           waterCausticWallBoost;
} RgDrawFrameReflectRefractParams;

typedef struct RgDrawFrameInfo
{
    RgStructureType             sType;
    void*                       pNext;
    // Max value: 10000.0
    float                       rayLength;

    RgBool32                    disableRayTracedGeometry;
    RgBool32                    disableRasterization;
    RgBool32                    presentPrevFrame;
    RgBool32                    resetHistory;

    double                      currentTime;
} RgDrawFrameInfo;

typedef RgResult( RGAPI_PTR* PFN_rgDrawFrame)(const RgDrawFrameInfo* pInfo );



typedef enum RgUtilImScratchTopology
{
    RG_UTIL_IM_SCRATCH_TOPOLOGY_TRIANGLES,
    RG_UTIL_IM_SCRATCH_TOPOLOGY_TRIANGLE_STRIP,
    RG_UTIL_IM_SCRATCH_TOPOLOGY_TRIANGLE_FAN,
    RG_UTIL_IM_SCRATCH_TOPOLOGY_QUADS,
} RgUtilImScratchTopology;

typedef struct RgUtilMemoryUsage
{
    size_t vramUsed;
    size_t vramTotal;
} RgUtilMemoryUsage;

typedef enum RgFeatureFlagBits
{
    RG_FEATURE_HDR      = 1,
    RG_FEATURE_FLUID    = 2,
} RgFeatureFlagBits;
typedef uint32_t RgFeatureFlags;

typedef RgPrimitiveVertex*  ( RGAPI_PTR* PFN_rgUtilScratchAllocForVertices      )( uint32_t vertexCount );
typedef void                ( RGAPI_PTR* PFN_rgUtilScratchFree                  )( const RgPrimitiveVertex* pPointer );
typedef void                ( RGAPI_PTR* PFN_rgUtilScratchGetIndices            )( RgUtilImScratchTopology topology, uint32_t vertexCount, const uint32_t** ppOutIndices, uint32_t* pOutIndexCount );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchClear               )();
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchStart               )( RgUtilImScratchTopology topology );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchVertex              )( float x, float y, float z ); // Push vertex to a list
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchNormal              )( float x, float y, float z );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchTexCoord            )( float u, float v );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchTexCoord_Layer1     )( float u, float v );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchTexCoord_Layer2     )( float u, float v );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchTexCoord_Layer3     )( float u, float v );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchColor               )( RgColor4DPacked32 color );
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchEnd                 )();
typedef void                ( RGAPI_PTR* PFN_rgUtilImScratchSetToPrimitive      )( RgMeshPrimitiveInfo* pTarget ); // Set accumulated vertices to pTarget
typedef RgBool32            ( RGAPI_PTR* PFN_rgUtilIsUpscaleTechniqueAvailable  )( RgRenderUpscaleTechnique technique, RgFrameGenerationMode frameGeneration, const char **ppFailureReason );
typedef RgBool32            ( RGAPI_PTR* PFN_rgUtilIsDXGIAvailable              )( const char **ppFailureReason );
typedef RgUtilMemoryUsage   ( RGAPI_PTR* PFN_rgUtilRequestMemoryUsage           )();
typedef const char*         ( RGAPI_PTR* PFN_rgUtilGetResultDescription         )( RgResult result );
typedef RgColor4DPacked32   ( RGAPI_PTR* PFN_rgUtilPackColorByte4D              )( uint8_t r, uint8_t g, uint8_t b, uint8_t a );
typedef RgColor4DPacked32   ( RGAPI_PTR* PFN_rgUtilPackColorFloat4D             )( float r, float g, float b, float a );
typedef RgNormalPacked32    ( RGAPI_PTR* PFN_rgUtilPackNormal                   )( float x, float y, float z );
typedef void                ( RGAPI_PTR* PFN_rgUtilExportAsTGA                  )( const void* pPixels, uint32_t width, uint32_t height, const char* pPath );
typedef RgFeatureFlags      ( RGAPI_PTR* PFN_rgUtilGetSupportedFeatures         )();



typedef struct RgInterface
{
    PFN_rgCreateInstance                  rgCreateInstance;
    PFN_rgDestroyInstance                 rgDestroyInstance;
    // Main
    PFN_rgStartFrame                      rgStartFrame;
    PFN_rgUploadCamera                    rgUploadCamera;
    PFN_rgUploadMeshPrimitive             rgUploadMeshPrimitive;
    PFN_rgUploadLensFlare                 rgUploadLensFlare;
    PFN_rgUploadLight                     rgUploadLight;
    PFN_rgProvideOriginalTexture          rgProvideOriginalTexture;
    PFN_rgMarkOriginalTextureAsDeleted    rgMarkOriginalTextureAsDeleted;
    PFN_rgDrawFrame                       rgDrawFrame;
    // Utils
    PFN_rgUtilScratchAllocForVertices     rgUtilScratchAllocForVertices;
    PFN_rgUtilScratchFree                 rgUtilScratchFree;
    PFN_rgUtilScratchGetIndices           rgUtilScratchGetIndices;
    PFN_rgUtilImScratchClear              rgUtilImScratchClear;
    PFN_rgUtilImScratchStart              rgUtilImScratchStart;
    PFN_rgUtilImScratchVertex             rgUtilImScratchVertex;
    PFN_rgUtilImScratchNormal             rgUtilImScratchNormal;
    PFN_rgUtilImScratchTexCoord           rgUtilImScratchTexCoord;
    PFN_rgUtilImScratchTexCoord_Layer1    rgUtilImScratchTexCoord_Layer1;
    PFN_rgUtilImScratchTexCoord_Layer2    rgUtilImScratchTexCoord_Layer2;
    PFN_rgUtilImScratchTexCoord_Layer3    rgUtilImScratchTexCoord_Layer3;
    PFN_rgUtilImScratchColor              rgUtilImScratchColor;
    PFN_rgUtilImScratchEnd                rgUtilImScratchEnd;
    PFN_rgUtilImScratchSetToPrimitive     rgUtilImScratchSetToPrimitive;
    PFN_rgUtilIsUpscaleTechniqueAvailable rgUtilIsUpscaleTechniqueAvailable;
    PFN_rgUtilIsDXGIAvailable             rgUtilDXGIAvailable;
    PFN_rgUtilRequestMemoryUsage          rgUtilRequestMemoryUsage;
    PFN_rgUtilGetResultDescription        rgUtilGetResultDescription;
    PFN_rgUtilPackColorByte4D             rgUtilPackColorByte4D;
    PFN_rgUtilPackColorFloat4D            rgUtilPackColorFloat4D;
    PFN_rgUtilPackNormal                  rgUtilPackNormal;
    PFN_rgUtilExportAsTGA                 rgUtilExportAsTGA;
    PFN_rgUtilGetSupportedFeatures        rgUtilGetSupportedFeatures;
    // Additional
    PFN_rgSpawnFluid                      rgSpawnFluid;
} RgInterface;

#if defined( _WIN32 )

inline RgResult rgLoadLibraryAndCreate( const RgInstanceCreateInfo* pInfo,
                                        RgBool32                    useDebugBinary,
                                        const char*                 customBinaryPath,
                                        RgInterface*                pOutInterface,
                                        HMODULE*                    pOutDll )
{
    if( pOutDll )
    {
        *pOutDll = NULL;
    }

    if( !pInfo || !pInfo->pOverrideFolderPath )
    {
        return RG_RESULT_WRONG_FUNCTION_ARGUMENT;
    }

    if( pInfo->version == nullptr || strcmp( pInfo->version, RG_RTGL_VERSION_API ) != 0 )
    {
        return RG_RESULT_WRONG_FUNCTION_ARGUMENT;
    }

    if( pInfo->sizeOfRgInterface != sizeof( RgInterface ) )
    {
        return RG_RESULT_WRONG_FUNCTION_ARGUMENT;
    }

    char rtglDllPath[ MAX_PATH ];
    {
        const char* dllPath = customBinaryPath ? customBinaryPath
                              : useDebugBinary ? "\\bin\\debug\\RTGL1.dll"
                                               : "\\bin\\RTGL1.dll";

        if( strlen( pInfo->pOverrideFolderPath ) + strlen( dllPath ) >= MAX_PATH - 1 )
        {
            return RG_RESULT_WRONG_FUNCTION_ARGUMENT;
        }

        rtglDllPath[ 0 ] = '\0';
        strcpy( rtglDllPath, pInfo->pOverrideFolderPath );
        strcat( rtglDllPath, dllPath );
    }

    HMODULE dll = LoadLibraryA( rtglDllPath );
    if( !dll )
    {
        return RG_RESULT_CANT_FIND_DYNAMIC_LIBRARY;
    }

    PFN_rgCreateInstance createFunc =
        ( PFN_rgCreateInstance )GetProcAddress( dll, "rgCreateInstance" );

    if( !createFunc )
    {
        FreeLibrary( dll );
        return RG_RESULT_CANT_FIND_ENTRY_FUNCTION_IN_DYNAMIC_LIBRARY;
    }

    if( pOutDll )
    {
        *pOutDll = dll;
    }

    return createFunc( pInfo, pOutInterface );
}

inline RgResult rgDestroyAndUnloadLibrary( RgInterface* pInterface, HMODULE dll )
{
    if( !pInterface || !dll )
    {
        return RG_RESULT_WRONG_FUNCTION_ARGUMENT;
    }

    pInterface->rgDestroyInstance();
    memset( pInterface, 0, sizeof( *pInterface ) );

    FreeLibrary( dll );

    return RG_RESULT_SUCCESS;
}

#endif // defined( _WIN32 ) 

#ifdef __cplusplus
}
#endif

#endif // RTGL1_H_
