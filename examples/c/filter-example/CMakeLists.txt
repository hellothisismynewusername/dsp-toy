cmake_minimum_required(VERSION 3.10)
project(filter-example C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# header file
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dsp_toy_lib SHARED IMPORTED)

if(WIN32)
    set_target_properties(dsp_toy_lib PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/dsp_toy_lib.dll"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/dsp_toy_lib.dll.lib"
    )
endif()

add_executable(example_app main.c)

target_link_libraries(example_app dsp_toy_lib)

if(NOT WIN32)
    set_target_properties(example_app PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    )
endif()
