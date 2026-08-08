cmake_minimum_required(VERSION 3.24)

project(
    HybridSplatRasterizer
    VERSION 2.0.0
    LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(HS_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
set(
    HS_ORIGINAL_RASTERIZER_DIR
    "${HS_VENDOR_DIR}/standard_rasterizer")
if(NOT EXISTS "${HS_ORIGINAL_RASTERIZER_DIR}/forward.cu")
    message(FATAL_ERROR "HybridSplat forward rasterizer source is missing")
endif()

set(
    HS_ORIGINAL_TRACER_SOURCE_DIR
    "${HS_VENDOR_DIR}/optix_tracer")
if(NOT EXISTS "${HS_ORIGINAL_TRACER_SOURCE_DIR}/forward.cu")
    message(FATAL_ERROR "HybridSplat OptiX forward source is missing")
endif()

set(
    HS_DETERMINISTIC_FLASHGS_DIR
    "${HS_VENDOR_DIR}/flashgs")
if(NOT EXISTS "${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/preprocess.cu")
    message(FATAL_ERROR "Deterministic FlashGS source is missing")
endif()

set(
    HS_OPTIX_INCLUDE_DIR
    ""
    CACHE PATH
    "NVIDIA OptiX SDK include directory")
if(NOT EXISTS "${HS_OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "OptiX SDK headers are missing; pass -DHS_OPTIX_INCLUDE_DIR=/path/to/optix/include")
endif()

set(
    HS_RASTERIZER_BUILD_ID
    ""
    CACHE STRING
    "Versioned HybridSplat native backend build identifier")
if(HS_RASTERIZER_BUILD_ID STREQUAL "")
    message(FATAL_ERROR "HS_RASTERIZER_BUILD_ID must be provided by ops/build.py")
endif()

if(CMAKE_CUDA_FLAGS MATCHES "(^|[ ;])(--use_fast_math|-use_fast_math|-ffast-math)([ ;]|$)")
    message(FATAL_ERROR "HybridSplat numerical CUDA library forbids fast-math")
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89-real)
endif()

set(HS_CUDA_ARCH "89" CACHE STRING "CUDA compute capability without a dot")

set(HS_OPTIX_PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/optix")
set(HS_OPTIX_PTX_FILE "${HS_OPTIX_PTX_DIR}/forward.ptx")
add_custom_command(
    OUTPUT "${HS_OPTIX_PTX_FILE}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${HS_OPTIX_PTX_DIR}"
    COMMAND
        "${CMAKE_CUDA_COMPILER}"
        --ptx
        --std=c++17
        --gpu-architecture=compute_${HS_CUDA_ARCH}
        --ftz=false
        --prec-div=true
        --prec-sqrt=true
        -I"${HS_ORIGINAL_TRACER_SOURCE_DIR}"
        -I"${HS_OPTIX_INCLUDE_DIR}"
        "${HS_ORIGINAL_TRACER_SOURCE_DIR}/forward.cu"
        -o "${HS_OPTIX_PTX_FILE}"
    DEPENDS
        "${HS_ORIGINAL_TRACER_SOURCE_DIR}/forward.cu"
        "${HS_ORIGINAL_TRACER_SOURCE_DIR}/params.h"
        "${HS_ORIGINAL_TRACER_SOURCE_DIR}/config.h"
        "${HS_ORIGINAL_TRACER_SOURCE_DIR}/auxiliary.h"
    COMMENT "Compiling HybridSplat OptiX forward PTX for compute_${HS_CUDA_ARCH}"
    VERBATIM)
add_custom_target(hybridsplat_optix_ptx DEPENDS "${HS_OPTIX_PTX_FILE}")

add_library(
    hybridsplat_rasterizer
    SHARED
    rasterizer/c_api.cpp
    optix/optix_backend.cpp
    rasterizer/flashgs_backend.cu
    rasterizer/flashgs_prepare.cpp
    rasterizer/full_rasterizer.cu
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/preprocess.cu
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/render.cu
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/sort.cu
    ${HS_ORIGINAL_RASTERIZER_DIR}/forward.cu
    rasterizer/precise_camera_center.cu
    rasterizer/precise_direction.cu
    rasterizer/precise_disk_rotation.cu
    rasterizer/precise_exp.cu
    rasterizer/precise_matrix.cu
    rasterizer/precise_activation.cu
    rasterizer/precise_reflection_trace.cu
    rasterizer/precise_sh.cu)

add_dependencies(hybridsplat_rasterizer hybridsplat_optix_ptx)

set_source_files_properties(
    rasterizer/precise_direction.cu
    rasterizer/precise_disk_rotation.cu
    rasterizer/precise_reflection_trace.cu
    rasterizer/precise_sh.cu
    PROPERTIES
        COMPILE_OPTIONS "--fmad=false")

set_source_files_properties(
    rasterizer/precise_activation.cu
    PROPERTIES
        COMPILE_OPTIONS "--fmad=false;--ftz=false")

set_source_files_properties(
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/preprocess.cu
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/render.cu
    ${HS_DETERMINISTIC_FLASHGS_DIR}/cuda_rasterizer/sort.cu
    PROPERTIES
        COMPILE_OPTIONS "-O1;-Xptxas=-O1")

target_include_directories(
    hybridsplat_rasterizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/rasterizer
        ${CMAKE_CURRENT_SOURCE_DIR}/optix
        ${HS_DETERMINISTIC_FLASHGS_DIR}
        ${HS_DETERMINISTIC_FLASHGS_DIR}/glm
        ${HS_ORIGINAL_RASTERIZER_DIR}
        ${HS_VENDOR_DIR}/standard_glm
        ${HS_ORIGINAL_TRACER_SOURCE_DIR}
        ${HS_OPTIX_INCLUDE_DIR})

target_compile_features(
    hybridsplat_rasterizer
    PRIVATE
        cxx_std_17
        cuda_std_17)

target_compile_definitions(
    hybridsplat_rasterizer
    PRIVATE
        HS_RASTERIZER_BUILDING_LIBRARY=1
        HS_RASTERIZER_BUILD_ID="${HS_RASTERIZER_BUILD_ID}")

target_link_libraries(
    hybridsplat_rasterizer
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cusolver
        ${CMAKE_DL_LIBS})

target_compile_options(
    hybridsplat_rasterizer
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--ftz=false;--prec-div=true;--prec-sqrt=true>)

set_target_properties(
    hybridsplat_rasterizer
    PROPERTIES
        CXX_EXTENSIONS OFF
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION 2
        OUTPUT_NAME hybridsplat_rasterizer)

include(GNUInstallDirs)

install(
    TARGETS hybridsplat_rasterizer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    FILES include/hybridsplat_rasterizer_c_api.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    FILES "${HS_OPTIX_PTX_FILE}"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hybridsplat)
