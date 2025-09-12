# CMakeLists.txt

# TT-Metalium : Adding spatter source directory to CMAKE_SOURCE_DIR
set(CMAKE_SOURCE_DIR "/home/user/tt-metal/tt_metal/programming_examples/spatter")

cmake_minimum_required(VERSION 3.25)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(DisableIntree)

project(Spatter VERSION 2.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

include(BuildType)
include(CompilerType)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# TT-Metalium : Added flag to enable or disable support TT-METAL
set(USE_TT_METAL on)

set(USE_OPENMP off)
set(USE_MPI off)
include(pkgs/JSONSupport)
include(pkgs/MPISupport)
include(pkgs/OpenMPSupport)
include(pkgs/CUDASupport)
include(pkgs/MetaliumSupport)

# Create gz_read executable
add_executable(gz_read standard-suite/binary-traces/gz_read.cc)

# TT-Metalium : Added tt-metal build dependencies
if (USE_TT_METAL)
        target_link_libraries(
                gz_read
                PUBLIC
                        tt_metal
                        m
                        pthread
                        z
                )


        target_include_directories(
                gz_read
                PRIVATE
                        ${UMD_HOME}
                        ${PROJECT_SOURCE_DIR}
                        ${PROJECT_SOURCE_DIR}/tt_metal
                        ${PROJECT_SOURCE_DIR}/tt_metal/common
                        ${CMAKE_CURRENT_SOURCE_DIR}
                )

        #message("${UMD_HOME}")
        #message("${PROJECT_SOURCE_DIR}")
        #message("${CMAKE_CURRENT_SOURCE_DIR}")
        #message("${PROJECT_BINARY_DIR}")

        set_target_properties(
                gz_read
                PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                        ${PROJECT_BINARY_DIR}
                )

        list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS  gz_read)

else()
        target_link_libraries(gz_read z)
endif()

add_subdirectory(src)

#enable_testing()
#add_subdirectory(tests)