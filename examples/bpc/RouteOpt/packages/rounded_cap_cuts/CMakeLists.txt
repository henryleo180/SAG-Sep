cmake_minimum_required(VERSION 3.10)
project(rounded_cap_cuts)

add_library(ROUNDED_CAP_CUTS_INCLUDES INTERFACE)
target_include_directories(ROUNDED_CAP_CUTS_INCLUDES INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/chg_rc_getter/include
        ${CMAKE_CURRENT_SOURCE_DIR}/coefficient_getter/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/separation/include
)

file(GLOB_RECURSE ROUNDED_CAP_CUTS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/chg_rc_getter/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/coefficient_getter/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/separation/src/*.cpp
)

target_sources(ROUNDED_CAP_CUTS_INCLUDES INTERFACE ${ROUNDED_CAP_CUTS_SOURCES})

# ---------------------------------------------------------------------------
# SAG-Sep engine (added by the SAG-Sep branch-price-and-cut case study)
# ---------------------------------------------------------------------------
set(SAGSEP_C "" CACHE PATH
        "SAG-Sep engine tree; must contain include/sep_api.h")
set(SAGSEP_LIB "" CACHE FILEPATH
        "libsagsep.a, as produced by `make -C \${SAGSEP_C} lib`")
set(ONNXRUNTIME_HOME "" CACHE PATH
        "ONNX Runtime root; must contain include/onnxruntime_c_api.h and lib/libonnxruntime.so")

if (NOT SAGSEP_C OR NOT EXISTS "${SAGSEP_C}/include/sep_api.h")
    message(FATAL_ERROR
            "SAGSEP_C must point at the SAG-Sep engine tree containing include/sep_api.h "
            "(got \"${SAGSEP_C}\").")
endif ()
if (NOT SAGSEP_LIB OR NOT EXISTS "${SAGSEP_LIB}")
    message(FATAL_ERROR
            "SAGSEP_LIB must point at libsagsep.a; build it with "
            "`make -C ${SAGSEP_C} lib GUROBI_HOME=... ONNXRUNTIME_HOME=...` (got \"${SAGSEP_LIB}\").")
endif ()
if (NOT ONNXRUNTIME_HOME OR NOT EXISTS "${ONNXRUNTIME_HOME}/lib/libonnxruntime.so")
    message(FATAL_ERROR
            "ONNXRUNTIME_HOME must point at an ONNX Runtime root with lib/libonnxruntime.so "
            "(got \"${ONNXRUNTIME_HOME}\").")
endif ()

add_library(SAGSEP_ENGINE STATIC IMPORTED GLOBAL)
set_target_properties(SAGSEP_ENGINE PROPERTIES
        IMPORTED_LOCATION "${SAGSEP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${SAGSEP_C}/include"
        INTERFACE_LINK_LIBRARIES "${ONNXRUNTIME_HOME}/lib/libonnxruntime.so;m"
        INTERFACE_LINK_OPTIONS "-Wl,-rpath,${ONNXRUNTIME_HOME}/lib"
)

target_link_libraries(ROUNDED_CAP_CUTS_INCLUDES INTERFACE SAGSEP_ENGINE)
