# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter1 main.cc)
#target_compile_options(spatter1 PUBLIC "-fnew-alignment 32")

target_include_directories(
            spatter1
            PRIVATE
                ${UMD_HOME}
                ${PROJECT_SOURCE_DIR}
                ${PROJECT_SOURCE_DIR}/tt_metal
                ${PROJECT_SOURCE_DIR}/tt_metal/common
                ${CMAKE_CURRENT_SOURCE_DIR}
                /home/user/tt-metal/tt_metal/programming_examples/spatter/src/Spatter
                /home/user/tt-metal/tt_metal/programming_examples/spatter/build_serial/_deps/nlohmann_json-src/include
        )

target_link_libraries(
            spatter1
            PUBLIC
                tt_metal
                m
                pthread
                Spatter
        )
set_target_properties(
            spatter1
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${PROJECT_BINARY_DIR}/programming_examples/spatter
        )

list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS spatter1)

#set(VECADD_SRCS2 /home/user/tt-metal/tt_metal/programming_examples/spatter/src/main.cc)

#CREATE_PGM_EXAMPLES_EXE("${VECADD_SRCS2}" "spatter") # output binaries to build/programming_examples/contributed

add_custom_target(spatter DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})