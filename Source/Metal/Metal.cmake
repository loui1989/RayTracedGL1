# Native Metal backend (macOS, Apple Silicon).
# Included from the root CMakeLists.txt when RG_WITH_METAL is ON;
# builds libRTGL1.dylib without Vulkan, DX12, DLSS or FSR.

if (NOT APPLE)
    message(FATAL_ERROR "RG_WITH_METAL is only supported on macOS")
endif()

enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 23)
# for GLFW's .m files
enable_language(OBJC)

set(MetalSources
    "Source/RTGL1.cpp"
    "Source/RgException.cpp"
    "Source/ScratchImmediate.cpp"
    "Source/UtilsMath.cpp"
    "Source/UtilsFormat.cpp"
    "Source/Matrix.cpp"
    "Source/HaltonSequence.cpp"
    "Source/SamplerHandle.cpp"
    "Source/LightEncoding.cpp"
    "Source/LpmParams.cpp"
    "Source/Stb/stb_image_write.cpp"
    "Source/Stb/stb_image.cpp"
    "Source/ImageLoader.cpp"
    "Source/ImageLoaderDev.cpp"
    "Source/TextureOverrides.cpp"
    "Source/GltfImporter.cpp"
    "Source/TextureMeta.cpp"
    "Source/SceneMeta.cpp"
    "Source/PrimitiveLights.cpp"
    "Source/Metal/MetalCgltf.cpp"
    "Source/Metal/MetalCas.cpp"
    "Source/Metal/MetalSceneManager.mm"
    "Source/Metal/MetalComputePipelines.mm"
    "Source/Metal/MetalDevice.mm"
    "Source/Metal/MetalFramebuffers.mm"
    "Source/Metal/MetalRasterizer.mm"
    "Source/Metal/MetalLightManager.mm"
    "Source/Metal/MetalPathTracer.mm"
    "Source/Metal/MetalFluid.mm"
    "Source/Metal/MetalScene.mm"
    "Source/Metal/MetalTextureManager.mm"
    "Source/Metal/MetalTextureExporter.cpp"
)

# Shaders: compiled into RTGL1.metallib, next to libRTGL1.dylib
set(MetalShaderSources
    "${CMAKE_SOURCE_DIR}/Source/Metal/Shaders/RTGL1.metal"
)
set(MetalShaderDependencies
    ${MetalShaderSources}
    "${CMAKE_SOURCE_DIR}/Source/Metal/Shaders/MetalBindings.h"
    "${CMAKE_SOURCE_DIR}/Source/Generated/ShaderCommonMSL.h"
)
set(MetalLibOutputDir $<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/Build/bin/debug,${CMAKE_SOURCE_DIR}/Build/bin>)
set(MetalShaderFlags -I "${CMAKE_SOURCE_DIR}/Source/Generated" -I "${CMAKE_SOURCE_DIR}/Source/Metal/Shaders")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND MetalShaderFlags -gline-tables-only -frecord-sources)
endif()
set(MetalAirFiles "")
foreach(src ${MetalShaderSources})
    get_filename_component(name ${src} NAME_WE)
    set(air "${CMAKE_CURRENT_BINARY_DIR}/shaders/${name}.air")
    add_custom_command(
        OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND xcrun -sdk macosx metal ${MetalShaderFlags} -c ${src} -o ${air}
        DEPENDS ${MetalShaderDependencies}
        COMMENT "Compiling Metal shader ${name}.metal"
        VERBATIM
    )
    list(APPEND MetalAirFiles ${air})
endforeach()
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/RTGL1.metallib"
    COMMAND xcrun -sdk macosx metallib ${MetalAirFiles} -o "${CMAKE_CURRENT_BINARY_DIR}/shaders/RTGL1.metallib"
    DEPENDS ${MetalAirFiles}
    COMMENT "Linking RTGL1.metallib"
    VERBATIM
)

# RTGL1's GLSL shaders (Source/Shaders), compiled for Metal into bin/shaders/<name>.metallib
find_package(spirv_cross_core CONFIG REQUIRED)
find_package(spirv_cross_glsl CONFIG REQUIRED)
find_package(spirv_cross_msl CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(GLSLANG_VALIDATOR glslangValidator HINTS /opt/homebrew/bin REQUIRED)

add_executable(SpirvToMsl "Source/Metal/Tools/SpirvToMsl.cpp")
set_property(TARGET SpirvToMsl PROPERTY CXX_STANDARD 20)
# the static SPIRV-Cross libraries aren't built with LTO
set_property(TARGET SpirvToMsl PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(SpirvToMsl PRIVATE spirv-cross-msl spirv-cross-glsl spirv-cross-core)

file(GLOB RtglGlslShaderDependencies
    "${CMAKE_SOURCE_DIR}/Source/Shaders/*"
    "${CMAKE_SOURCE_DIR}/Source/Shaders/*/*"
    "${CMAKE_SOURCE_DIR}/Source/Generated/ShaderCommonGLSL.h"
)
set(RtglGlslShadersStamp "${CMAKE_CURRENT_BINARY_DIR}/shaders/rtgl_glsl.stamp")
add_custom_command(
    OUTPUT ${RtglGlslShadersStamp}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/Source/Metal/CompileShadersMetal.py"
            --shaders "${CMAKE_SOURCE_DIR}/Source/Shaders"
            --generated "${CMAKE_SOURCE_DIR}/Source/Generated"
            --build "${CMAKE_CURRENT_BINARY_DIR}/shaders/glsl"
            --output "${MetalLibOutputDir}/shaders"
            --spirv-to-msl $<TARGET_FILE:SpirvToMsl>
            --glslang ${GLSLANG_VALIDATOR}
    COMMAND ${CMAKE_COMMAND} -E touch ${RtglGlslShadersStamp}
    DEPENDS ${RtglGlslShaderDependencies} "${CMAKE_SOURCE_DIR}/Source/Metal/CompileShadersMetal.py" SpirvToMsl
    COMMENT "Compiling RTGL1 GLSL shaders for Metal"
    VERBATIM
)

add_custom_target(RTGL1MetalShaders
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/shaders/RTGL1.metallib" ${RtglGlslShadersStamp}
)

# KTX-Software, for .ktx2 textures (the same sources as the Vulkan build)
set(MetalKTXSources
    "Source/KTX/lib/texture.c"
    "Source/KTX/lib/texture1.c"
    "Source/KTX/lib/texture2.c"
    "Source/KTX/lib/hashlist.c"
    "Source/KTX/lib/checkheader.c"
    "Source/KTX/lib/swap.c"
    "Source/KTX/lib/memstream.c"
    "Source/KTX/lib/filestream.c"
    # for ktxTexture_GetVkFormat; Vulkan functions are loaded at runtime only if used, and they aren't
    "Source/KTX/lib/vkloader.c"
    "Source/KTX/lib/vk_funcs.c"
    "Source/KTX/lib/basisu/zstd/zstd.c"
    "Source/KTX/lib/dfdutils/createdfd.c"
    "Source/KTX/lib/dfdutils/interpretdfd.c"
    "Source/KTX/lib/dfdutils/queries.c"
    "Source/KTX/lib/dfdutils/vk2dfd.c"
)

# Json parser - glaze, the same as the Vulkan build
add_library(GlazeWrapper STATIC "Source/JsonParser.cpp")
add_subdirectory(Source/glaze)
target_include_directories(GlazeWrapper PRIVATE "Include")
target_link_libraries(GlazeWrapper PRIVATE glaze::glaze)

add_library(RayTracedGL1 SHARED
    ${MetalSources}
    ${MetalKTXSources}
    ${PublicHeaders}
)
set_target_properties(RayTracedGL1 PROPERTIES
    OUTPUT_NAME "RTGL1"
    LIBRARY_OUTPUT_DIRECTORY $<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/Build/bin/debug,${CMAKE_SOURCE_DIR}/Build/bin>
)
target_compile_definitions(RayTracedGL1 PRIVATE
    RG_USE_METAL
    RG_USE_SURFACE_METAL
    RG_LIBRARY_NAME="RTGL1"
)
target_compile_options(RayTracedGL1 PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_include_directories(RayTracedGL1 PUBLIC "Include")
target_include_directories(RayTracedGL1 PRIVATE
    "Source"
    # shared code still uses Vulkan types in its headers; nothing links to Vulkan
    "Source/VulkanSDK/1.3.280.0/Include"
    "Source/glm"
    "Source/KTX/include"
    "Source/KTX/other_include"
    "Source/KTX/lib/basisu/zstd"
)
target_compile_definitions(RayTracedGL1 PRIVATE KHRONOS_STATIC LIBKTX)
target_link_libraries(RayTracedGL1 PRIVATE GlazeWrapper)
target_link_libraries(RayTracedGL1 PRIVATE
    "-framework Metal"
    "-framework QuartzCore"
    "-framework Foundation"
    "-framework MetalFX"
    "-framework CoreGraphics"
)
add_dependencies(RayTracedGL1 RTGL1MetalShaders)
add_custom_command(TARGET RayTracedGL1 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/shaders/RTGL1.metallib" "${MetalLibOutputDir}/RTGL1.metallib"
    VERBATIM
)

if (RG_WITH_EXAMPLES)
    message(STATUS "RG_WITH_EXAMPLES enabled")
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(Source/glfw)

    add_executable(RtglExample Tests/RtglTest.cpp Tests/RtglTestMetal.mm)
    set_property(TARGET RtglExample PROPERTY CXX_STANDARD 20)
    target_compile_definitions(RtglExample PRIVATE ASSET_DIRECTORY="${CMAKE_SOURCE_DIR}/Build/")
    target_compile_options(RtglExample PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_link_libraries(RtglExample glfw "-framework QuartzCore" "-framework AppKit")
    target_include_directories(RtglExample PRIVATE "Include")
    target_include_directories(RtglExample PUBLIC Source/glm)
    # not needed, but let RtglExample be built after RayTracedGL1
    add_dependencies(RtglExample RayTracedGL1)
endif()
