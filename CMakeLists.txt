cmake_minimum_required(VERSION 3.20)
project(RayTracedGL1 CXX C)

set(CMAKE_CXX_STANDARD 23)
if (NOT WIN32)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE )

include(CheckIPOSupported)
check_ipo_supported()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)

set(PublicHeaders 
    Include/RTGL1/RTGL1.h
)

set(Sources
    "Source/RTGL1.cpp"
    "Source/RTGL1A.cpp"
    "Source/VulkanDevice.cpp"
    "Source/VulkanDevice_Init.cpp"
    "Source/Buffer.cpp"
    "Source/Scene.cpp"
    "Source/PhysicalDevice.cpp"
    "Source/Queues.cpp"
    "Source/Swapchain.cpp"
    "Source/GlobalUniform.cpp"
    "Source/CommandBufferManager.cpp"
    "Source/ShaderManager.cpp"
    "Source/RayTracingPipeline.cpp"
    "Source/VertexCollector.cpp"
    "Source/ASManager.cpp"
    "Source/VertexCollectorFilter.cpp"
    "Source/ASBuilder.cpp"
    "Source/ScratchBuffer.cpp"
    "Source/Utils.cpp"
    "Source/PathTracer.cpp"
    "Source/Common.cpp"
    "Source/Matrix.cpp"
    "Source/Rasterizer.cpp"
    "Source/RasterizedDataCollector.cpp"
    "Source/Vma/vk_mem_alloc_imp.cpp"
    "Source/ImageLoader.cpp" 
    "Source/TextureManager.cpp" 
    "Source/MemoryAllocator.cpp" 
    "Source/SamplerManager.cpp" 
    "Source/TextureOverrides.cpp"
    "Source/TextureDescriptors.cpp" 
    "Source/TextureUploader.cpp"
    "Source/VertexCollectorFilterType.cpp"
    "Source/Generated/ShaderCommonCFramebuf.cpp" 
    "Source/Framebuffers.cpp"
    "Source/BlueNoise.cpp"
    "Source/ImageComposition.cpp"
    "Source/Tonemapping.cpp"
    "Source/LightManager.cpp"
    "Source/AutoBuffer.cpp"
    "Source/ASComponent.cpp"
    "Source/CubemapManager.cpp"
    "Source/CubemapUploader.cpp"
    "Source/GeomInfoManager.cpp"
    "Source/VertexPreprocessing.cpp"
    "Source/Denoiser.cpp"
    "Source/RasterizerPipelines.cpp"
    "Source/RenderCubemap.cpp"
    "Source/DepthCopying.cpp"
    "Source/RasterPass.cpp"
    "Source/SwapchainPass.cpp"
    "Source/RgException.cpp"
    "Source/Bloom.cpp"
    "Source/Sharpening.cpp"
    "Source/DLSS2.cpp"
    "Source/DLSSRR.cpp"
    "Source/NrdDenoiser.cpp"
    "Source/DLSS3_DX12.cpp"
    "Source/HaltonSequence.cpp"
    "Source/LensFlares.cpp"
    "Source/DecalManager.cpp"
    "Source/EffectBase.cpp"
    "Source/LightGrid.cpp"
    "Source/FSR2.cpp"
    "Source/FSR3_DX12.cpp"
    "Source/Stb/stb_image.cpp"
    "Source/Stb/stb_image_write.cpp"
    "Source/ImageLoaderDev.cpp"
    "Source/PortalList.cpp"
    "Source/RestirBuffers.cpp"
    "Source/Volumetric.cpp"
    "Source/DebugWindows.cpp"
    "Source/ScratchImmediate.cpp"
    "Source/GltfExporter.cpp"
    "Source/GltfImporter.cpp"
    "Source/FolderObserver.cpp"
    "Source/TextureExporter.cpp"
    "Source/TextureMeta.cpp"
    "Source/VulkanDevice_Dev.cpp"
    "Source/SceneMeta.cpp"
    "Source/Fluid.cpp"
    "Source/DX12_Swapchain.cpp"
    "Source/DX12_Vulkan.cpp"
)



set(KTXSourceFolder Source/KTX/lib)

set(KTXSources
    "${KTXSourceFolder}/texture.c"
    "${KTXSourceFolder}/texture1.c"
    "${KTXSourceFolder}/texture2.c"
    "${KTXSourceFolder}/hashlist.c"
    "${KTXSourceFolder}/checkheader.c"
    "${KTXSourceFolder}/swap.c"
    "${KTXSourceFolder}/memstream.c"
    "${KTXSourceFolder}/filestream.c"
    "${KTXSourceFolder}/vkloader.c"
    "${KTXSourceFolder}/vk_funcs.c"
    "${KTXSourceFolder}/basisu/zstd/zstd.c"
    "${KTXSourceFolder}/dfdutils/createdfd.c"
    "${KTXSourceFolder}/dfdutils/interpretdfd.c"
    "${KTXSourceFolder}/dfdutils/queries.c"
    "${KTXSourceFolder}/dfdutils/vk2dfd.c"
)

option(RG_WITH_EXPORTS          "Compile library for export"                ON)

option(RG_WITH_SURFACE_WIN32    "Build with Win32 VkSurfaceKHR support"     OFF)
option(RG_WITH_SURFACE_METAL    "Build with Metal VkSurfaceKHR support"     OFF)
option(RG_WITH_SURFACE_WAYLAND  "Build with Wayland VkSurfaceKHR support"   OFF)
option(RG_WITH_SURFACE_XCB      "Build with Xcb VkSurfaceKHR support"       OFF)
option(RG_WITH_SURFACE_XLIB     "Build with Xlib VkSurfaceKHR support"      OFF)

option(RG_WITH_IMGUI            "Build RTGL1 with ImGui debug windows"      ON)
option(RG_WITH_DX12             "Build RTGL1 with DX12 features"            ON)
option(RG_WITH_NATIVE_DLSS      "Build RTGL1 with native DLSS2"             ON)
# OFF by default: A-SVGF is the shipping baseline denoiser, and NRD needs an
# SDK that is not in the repo (deps/NRD + deps/NRI, built by
# tools/build-nrd-deps.cmd). Defaulting ON made a clean checkout -- the CI
# release runner included -- fail configure with the FATAL_ERROR below rather
# than build the baseline. Turn it ON to work on the NRD lane.
option(RG_WITH_NRD              "Build RTGL1 with the NVIDIA NRD denoiser lane (NRI-wrapped)" OFF)

option(RG_WITH_EXAMPLES         "Build with examples executable"            ON)

if (APPLE)
    # macOS: Vulkan is implemented on Metal inside the library (Source/MetalNative.mm,
    # MetalBridge.mm), without a Vulkan loader or MoltenVK. No DX12 / DLSS / NRD; the
    # MetalFX temporal upscaler fills the FSR2 slot (Source/FSR2_Metal.mm).
    message(STATUS "macOS: Vulkan on Metal (RG_METAL_ONLY)")
    set(RG_WITH_SURFACE_METAL ON  CACHE BOOL "" FORCE)
    set(RG_WITH_SURFACE_WIN32 OFF CACHE BOOL "" FORCE)
    set(RG_WITH_DX12          OFF CACHE BOOL "" FORCE)
    set(RG_WITH_IMGUI         OFF CACHE BOOL "" FORCE)
    set(RG_WITH_NATIVE_DLSS   OFF CACHE BOOL "" FORCE)
    set(RG_WITH_NRD           OFF CACHE BOOL "" FORCE)
    set(RG_WITH_EXAMPLES      OFF CACHE BOOL "" FORCE)
    enable_language(OBJCXX)
endif()

if (NOT RG_WITH_DX12)
    # stubs that report DXGI / FSR3 / DLSS3 frame generation unavailable
    list(REMOVE_ITEM Sources
        "Source/FSR3_DX12.cpp"
        "Source/DLSS3_DX12.cpp"
        "Source/DX12_Swapchain.cpp"
        "Source/DX12_Vulkan.cpp")
    list(APPEND Sources "Source/NoDX12.cpp")
endif()
if (APPLE)
    list(REMOVE_ITEM Sources "Source/FSR2.cpp")
    list(APPEND Sources
        "Source/FSR2_Metal.mm"
        "Source/MetalBridge.mm"
        "Source/MetalNative.mm"
        "Source/MetalShaderGen.cpp")
    set_source_files_properties(
        "Source/FSR2_Metal.mm" "Source/MetalBridge.mm" "Source/MetalNative.mm"
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()


# for KTX-Software
add_definitions(-DKHRONOS_STATIC -DLIBKTX)


# options to definitions
if (RG_WITH_EXPORTS)
    message(STATUS "RG_WITH_EXPORTS enabled. RTGL1 functions are exported.")
    add_definitions(-DRG_LIBRARY_EXPORTS)
endif()

if (RG_WITH_SURFACE_WIN32)
    message(STATUS "RG_WITH_SURFACE_WIN32 enabled")
    add_definitions(-DNOMINMAX)
    add_definitions(-DRG_USE_SURFACE_WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
endif()
if (RG_WITH_SURFACE_METAL)
    message(STATUS "RG_WITH_SURFACE_METAL enabled")
    add_definitions(-DRG_USE_SURFACE_METAL)
    add_definitions(-DVK_USE_PLATFORM_METAL_EXT)
endif()
if (RG_WITH_SURFACE_WAYLAND)
    message(STATUS "RG_WITH_SURFACE_WAYLAND enabled")
    add_definitions(-DRG_USE_SURFACE_WAYLAND)
    add_definitions(-DVK_USE_PLATFORM_WAYLAND_KHR)
endif()
if (RG_WITH_SURFACE_XCB)
    message(STATUS "RG_WITH_SURFACE_XCB enabled")
    add_definitions(-DRG_USE_SURFACE_XCB)
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
endif()
if (RG_WITH_SURFACE_XLIB)
    message(STATUS "RG_WITH_SURFACE_XLIB enabled")
    add_definitions(-DRG_USE_SURFACE_XLIB)
    add_definitions(-DVK_USE_PLATFORM_XLIB_KHR)
endif()


# Vulkan
set(VulkanSdkDir "Source/VulkanSDK/1.3.280.0")
message(STATUS "Adding Vulkan from ${VulkanSdkDir}")
add_library(Vulkan INTERFACE)
target_include_directories(Vulkan INTERFACE "${VulkanSdkDir}/Include")
if (APPLE)
    # no loader: the library implements the Vulkan functions it uses (MetalNative.mm)
    target_compile_definitions(Vulkan INTERFACE RG_METAL_ONLY)
elseif (WIN32)
    target_link_libraries(Vulkan INTERFACE "${CMAKE_SOURCE_DIR}/${VulkanSdkDir}/Lib/vulkan-1.lib")
else()
    target_link_libraries(Vulkan INTERFACE vulkan)
endif()


# DirectX 12
if (RG_WITH_DX12)
    message(STATUS "RG_WITH_DX12 enabled")
    add_definitions(-DRG_USE_DX12)
    set(D3D12_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    find_package(D3D12 REQUIRED)
    add_library(D3D12 INTERFACE)
    target_include_directories(D3D12 INTERFACE "Source/DX12/include/directx")
    target_link_libraries(D3D12 INTERFACE ${D3D12_LIBRARIES})
endif()


# Imgui
if (RG_WITH_IMGUI)
    message(STATUS "RG_WITH_IMGUI enabled")
    add_definitions(-DRG_USE_IMGUI)
    set(ImGuiSourceFolder Source/imgui)
    add_library(ImGui STATIC)
    target_sources(ImGui PRIVATE 
        ${ImGuiSourceFolder}/imgui_demo.cpp
        ${ImGuiSourceFolder}/imgui_draw.cpp
        ${ImGuiSourceFolder}/imgui_tables.cpp
        ${ImGuiSourceFolder}/imgui_widgets.cpp
        ${ImGuiSourceFolder}/imgui.cpp
        ${ImGuiSourceFolder}/backends/imgui_impl_vulkan.cpp
        ${ImGuiSourceFolder}/backends/imgui_impl_glfw.cpp
    )
    target_include_directories(ImGui PUBLIC ${ImGuiSourceFolder} ${ImGuiSourceFolder}/backends)
    # Use Vulkan and GLFW as ImGui backend
    target_link_libraries(ImGui PUBLIC Vulkan)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(Source/glfw)
    target_link_libraries(ImGui PRIVATE glfw)
    target_include_directories(ImGui PRIVATE glfw)
else()
    # the developer UI code (VulkanDevice_Dev.cpp) still compiles against ImGui's core;
    # without the debug windows it never runs
    set(ImGuiSourceFolder Source/imgui)
    add_library(ImGui STATIC
        ${ImGuiSourceFolder}/imgui_draw.cpp
        ${ImGuiSourceFolder}/imgui_tables.cpp
        ${ImGuiSourceFolder}/imgui_widgets.cpp
        ${ImGuiSourceFolder}/imgui.cpp
    )
    target_include_directories(ImGui PUBLIC ${ImGuiSourceFolder})
    set_target_properties(ImGui PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


add_library(RayTracedGL1 SHARED  
    ${Sources}
    ${KTXSources}
    ${PublicHeaders}
)
set(LibraryName "RTGL1")
add_definitions(-DRG_LIBRARY_NAME="${LibraryName}")
set_target_properties(RayTracedGL1 PROPERTIES
    OUTPUT_NAME ${LibraryName}
)
set_target_properties(RayTracedGL1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY $<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/Build/bin/debug,${CMAKE_SOURCE_DIR}/Build/bin>
)
target_include_directories(RayTracedGL1 PUBLIC "Include")


# Vulkan
target_link_libraries(RayTracedGL1 PRIVATE Vulkan)

# DirectX
if (RG_WITH_DX12)
    target_link_libraries(RayTracedGL1 PRIVATE D3D12)
endif()

# GLM
target_include_directories(RayTracedGL1 PRIVATE Source/glm)

# KTX
target_include_directories(RayTracedGL1 PRIVATE "Source/KTX/include")
target_include_directories(RayTracedGL1 PRIVATE "Source/KTX/include" "Source/KTX/other_include" "Source/KTX/lib/basisu/zstd")

# DLSS
target_include_directories(RayTracedGL1 PRIVATE "Source/Streamline/include")

# Accept the SDK location from EITHER the environment or -DDLSS_SDK_PATH=...
# Previously this tested `DEFINED ENV{DLSS_SDK_PATH}` only, so configuring with
# -DDLSS_SDK_PATH silently skipped this whole block: native DLSS2 AND DLSS Ray
# Reconstruction were compiled out (DLSSRR.cpp falls through to its empty stub,
# nvDlssRr is always null), with no error and no runtime log line -- because the
# log strings themselves were compiled out. Worse, CMake cached the -D value, so
# every later rebuild reused the poisoned cache. Cost a full debugging session:
# every RR cvar and guide fix appeared to do nothing because there was no RR.
if (NOT DEFINED DLSS_SDK_PATH AND DEFINED ENV{DLSS_SDK_PATH})
    set(DLSS_SDK_PATH "$ENV{DLSS_SDK_PATH}")
endif()

if (RG_WITH_NATIVE_DLSS AND DLSS_SDK_PATH)
    if (NOT EXISTS "${DLSS_SDK_PATH}/include/nvsdk_ngx.h")
        message(FATAL_ERROR
            "RG_WITH_NATIVE_DLSS=ON and DLSS_SDK_PATH='${DLSS_SDK_PATH}', but "
            "${DLSS_SDK_PATH}/include/nvsdk_ngx.h is missing. Fix the path or set "
            "RG_WITH_NATIVE_DLSS=OFF; do not build silently without Ray Reconstruction.")
    endif()
    message(STATUS "Found Native DLSS SDK: ${DLSS_SDK_PATH}")
    target_include_directories(RayTracedGL1 PRIVATE "${DLSS_SDK_PATH}/include")
    add_definitions(-DRG_USE_NATIVE_DLSS2)
    if (WIN32)
        # NVIDIA/DLSS layout uses lib/Windows_x86_64/x64 (older trees used x86_64)
        set(_NGX_LIB_DIR "${DLSS_SDK_PATH}/lib/Windows_x86_64/x64")
        if (NOT EXISTS "${_NGX_LIB_DIR}/nvsdk_ngx_d.lib")
            set(_NGX_LIB_DIR "${DLSS_SDK_PATH}/lib/Windows_x86_64/x86_64")
        endif()
        if (BUILD_TYPE STREQUAL "DEBUG")
            target_link_libraries(RayTracedGL1 PRIVATE "${_NGX_LIB_DIR}/nvsdk_ngx_d_dbg.lib")
        else()
            target_link_libraries(RayTracedGL1 PRIVATE "${_NGX_LIB_DIR}/nvsdk_ngx_d.lib")
        endif()
    else()
        target_link_libraries(RayTracedGL1 PRIVATE "${DLSS_SDK_PATH}/lib/Linux_x86_64/libnvsdk_ngx.a")
        target_link_libraries(RayTracedGL1 PRIVATE ${CMAKE_DL_LIBS})
    endif()
elseif (RG_WITH_NATIVE_DLSS)
    # Loud, because the resulting DLL is indistinguishable at a glance from a
    # working one -- it just has no DLSS-RR in it at all.
    message(WARNING
        "RG_WITH_NATIVE_DLSS=ON but DLSS_SDK_PATH is not set (neither as an "
        "environment variable nor -DDLSS_SDK_PATH=...). Native DLSS2 and DLSS "
        "Ray Reconstruction WILL BE COMPILED OUT of RTGL1.dll: rt_rayreconstr "
        "will appear to work but nvDlssRr is always null and no DLSSRR: log "
        "lines will ever be printed.")
endif()

# NRD + NRI (Doom64-RT: the third denoiser lane, docs/plan-nrd-denoiser.md).
# Duke-RT's pattern: NRD's seven sources are compiled straight into this
# target with the embedded pre-generated SPIR-V headers (deps/NRD/_Shaders,
# produced once by tools/build-nrd-deps.cmd); NRI is a VK-only static build
# from its own CMake (same script). The NRD_* definitions here MUST match
# deps/NRD/Shaders/NRDConfig.hlsli or the C++ side and the shaders disagree
# about the data layout silently.
if (RG_WITH_NRD)
    if (NOT DEFINED NRD_DEPS_PATH)
        # deps/RTGL's siblings in the Doom64-RT layout
        get_filename_component(NRD_DEPS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    endif()

    set(_NRD_DIR  "${NRD_DEPS_PATH}/NRD")
    set(_NRI_DIR  "${NRD_DEPS_PATH}/NRI")
    set(_ML_DIR   "${NRD_DEPS_PATH}/MathLib")
    set(_SM_DIR   "${NRD_DEPS_PATH}/ShaderMake")
    # NRI is a SHARED library on purpose: its VK backend embeds its own
    # VulkanMemoryAllocator implementation, and RTGL1.dll already embeds one --
    # two static VMA implementation TUs in one binary are LNK2005 across every
    # Vma* class. Duke-RT ships NRI as runtime DLLs for the same reason.
    # NRI.dll must be staged next to RTGL1.dll (tools/build-rtgl.cmd does).
    set(_NRI_LIBS "${_NRI_DIR}/_Bin/Release/NRI.lib")

    if (NOT EXISTS "${_NRD_DIR}/Include/NRD.h" OR NOT EXISTS "${_NRD_DIR}/_Shaders")
        message(FATAL_ERROR
            "RG_WITH_NRD=ON but '${_NRD_DIR}' is missing NRD or its pre-generated "
            "_Shaders headers. Run tools/build-nrd-deps.cmd first, or configure with "
            "-DRG_WITH_NRD=OFF; do not build silently without the NRD lane.")
    endif()
    if (NOT EXISTS "${_NRI_DIR}/_Bin/Release/NRI.lib")
        message(FATAL_ERROR
            "RG_WITH_NRD=ON but NRI is not built ('${_NRI_DIR}/_Bin/Release/NRI.lib' "
            "missing). Run tools/build-nrd-deps.cmd first.")
    endif()

    message(STATUS "RG_WITH_NRD enabled: NRD at ${_NRD_DIR}, NRI at ${_NRI_DIR}")

    target_sources(RayTracedGL1 PRIVATE
        "${_SM_DIR}/ShaderMake/ShaderBlob.cpp"
        "${_NRD_DIR}/Source/Wrapper.cpp"
        "${_NRD_DIR}/Source/InstanceImpl.cpp"
        "${_NRD_DIR}/Source/Reblur.cpp"
        "${_NRD_DIR}/Source/Relax.cpp"
        "${_NRD_DIR}/Source/Sigma.cpp"
        "${_NRD_DIR}/Source/Reference.cpp"
        "${_NRD_DIR}/Source/Timer.cpp"
    )
    target_include_directories(RayTracedGL1 PRIVATE
        "${_NRD_DIR}/Include"
        "${_NRD_DIR}/Integration"
        "${_NRD_DIR}/_Shaders"
        "${_ML_DIR}"
        "${_NRI_DIR}/Include"
        "${_SM_DIR}"
    )
    target_link_libraries(RayTracedGL1 PRIVATE ${_NRI_LIBS})
    # DELAY-LOADED, and this is load-bearing: the app loads rt/bin/RTGL1.dll
    # with plain LoadLibraryA (rgLoadLibraryAndCreate), so a STATIC import of
    # NRI.dll would be resolved against the EXE's directory and PATH -- not
    # rt/bin -- and RTGL1.dll itself would fail to load ("RTGL1.dll not
    # found", with NRI.dll never mentioned). Delay-loading defers resolution
    # until the first NRI call; NrdDenoiser::EnsureReady explicitly loads
    # NRI.dll from RTGL1.dll's own directory before that call. Bonus: NRI.dll
    # is not even mapped unless rt_nrd is actually enabled.
    target_link_options(RayTracedGL1 PRIVATE "/DELAYLOAD:NRI.dll")
    target_link_libraries(RayTracedGL1 PRIVATE delayimp)
    # Must mirror deps/NRD/Shaders/NRDConfig.hlsli exactly (one deliberate
    # divergence from Duke-RT there: ANTIFIREFLY=1, documented in the file).
    target_compile_definitions(RayTracedGL1 PRIVATE
        RG_USE_NRD
        NRD_STATIC_LIBRARY=1
        NRD_NORMAL_ENCODING=2
        NRD_ROUGHNESS_ENCODING=1
        NRD_SUPPORTS_VIEWPORT_OFFSET=0
        NRD_SUPPORTS_CHECKERBOARD=0
        NRD_SUPPORTS_HISTORY_CONFIDENCE=0
        NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX=0
        NRD_SUPPORTS_BASECOLOR_METALNESS=1
        NRD_SUPPORTS_ANTIFIREFLY=1
        REBLUR_PERFORMANCE_MODE=0
        NRD_EMBEDS_SPIRV_SHADERS=1
        NRD_EMBEDS_DXIL_SHADERS=0
        NRD_EMBEDS_DXBC_SHADERS=0
        SPIRV_SREG_OFFSET=0
        SPIRV_BREG_OFFSET=2
        SPIRV_UREG_OFFSET=3
        SPIRV_TREG_OFFSET=20
    )
endif()

# FSR2
target_include_directories(RayTracedGL1 PRIVATE "Source/FSR2/include" )

# Debug windows - ImGui
if (RG_WITH_IMGUI)
    target_link_libraries(RayTracedGL1 PRIVATE ImGui)
    target_include_directories(RayTracedGL1 PRIVATE ImGui)
    target_link_libraries(RayTracedGL1 PRIVATE glfw)
    target_include_directories(RayTracedGL1 PRIVATE glfw)
else()
    target_link_libraries(RayTracedGL1 PRIVATE ImGui)
endif()

# macOS: Metal
if (APPLE)
    target_link_libraries(RayTracedGL1 PRIVATE
        "-framework Foundation" "-framework Metal" "-framework MetalFX"
        "-framework QuartzCore" "-framework CoreGraphics")
    set_target_properties(RayTracedGL1 PROPERTIES
        INSTALL_NAME_DIR "@rpath"
        OBJCXX_STANDARD 23
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Build/bin")

    # shaders are translated from SPIR-V to Metal at startup with SPIRV-Cross
    set(RG_SPIRV_CROSS_DIR "" CACHE PATH "SPIRV-Cross source directory (required on macOS)")
    if (NOT RG_SPIRV_CROSS_DIR)
        message(FATAL_ERROR "macOS: set RG_SPIRV_CROSS_DIR to a SPIRV-Cross source checkout")
    endif()
    set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_SHARED OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_STATIC ON CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_FORCE_PIC ON CACHE BOOL "" FORCE)
    add_subdirectory("${RG_SPIRV_CROSS_DIR}" spirv-cross EXCLUDE_FROM_ALL)
    target_link_libraries(RayTracedGL1 PRIVATE spirv-cross-msl spirv-cross-glsl spirv-cross-core)
    target_compile_definitions(RayTracedGL1 PRIVATE RG_HAS_SPIRV_CROSS)

    # libc++ doesn't include standard headers transitively like MSVC's STL does
    target_compile_options(RayTracedGL1 PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includealgorithm>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includecstring>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includecstdio>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includecfloat>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includelimits>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includememory>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includestring>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-includefunctional>"
        "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wno-nullability-completeness>")
endif()

# Json parser - glaze
add_library(GlazeWrapper STATIC "Source/JsonParser.cpp")
add_subdirectory(Source/glaze)
target_include_directories(GlazeWrapper PRIVATE "Include")
target_link_libraries(GlazeWrapper PRIVATE glaze::glaze)
target_link_libraries(RayTracedGL1 PRIVATE GlazeWrapper)

if (RG_WITH_EXAMPLES)
    message(STATUS "RG_WITH_EXAMPLES enabled")
    add_definitions(-DASSET_DIRECTORY="../")
    add_executable(RtglExample Tests/RtglTest.cpp)
    set_property(TARGET RtglExample PROPERTY CXX_STANDARD 20)
    target_link_libraries(RtglExample glfw)
    target_include_directories(ImGui PRIVATE glfw)
    target_include_directories(RtglExample PRIVATE "Include" )
    target_include_directories(RtglExample PUBLIC Source/glm)
    # not needed, but let RtglExample be built after RayTracedGL1
    add_dependencies(RtglExample RayTracedGL1)
endif()

# VS hot-reload - disabled because of glaze
if (MSVC AND WIN32 AND NOT MSVC_VERSION VERSION_LESS 142)
    target_link_options(RayTracedGL1 PRIVATE $<$<CONFIG:Debug>:/INCREMENTAL>)
    target_compile_options(RayTracedGL1 PRIVATE $<$<CONFIG:Debug>:/ZI>)

    if (TARGET RtglExample)
        target_link_options(RtglExample PRIVATE $<$<CONFIG:Debug>:/INCREMENTAL>)
        target_compile_options(RtglExample PRIVATE $<$<CONFIG:Debug>:/ZI>)
    endif()
endif()
