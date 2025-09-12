# Spatter/src/CMakeLists.txt

add_subdirectory(Spatter)

add_executable(spatter main.cc)
#target_compile_options(spatter PUBLIC "-fnew-alignment 32")

if (USE_TT_METAL)
        target_include_directories(
                spatter
                PRIVATE
                        ${UMD_HOME}
                        ${PROJECT_SOURCE_DIR}
                        ${PROJECT_SOURCE_DIR}/tt_metal
                        ${PROJECT_SOURCE_DIR}/tt_metal/common
                        ${CMAKE_CURRENT_SOURCE_DIR}
                        ${PROJECT_SOURCE_DIR}/tt_metal/programming_examples/spatter/src/Spatter
                )

        target_link_libraries(
                spatter
                PUBLIC
                        tt_metal
                        m
                        pthread
                        Spatter
                )

        set_target_properties(
                spatter
                PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                        ${PROJECT_BINARY_DIR}
                )

        list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS spatter)
        add_custom_target(spatter_tt_metal DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
else()

        target_link_libraries(spatter ${COMMON_LINK_LIBRARIES} Spatter)
        
        set_target_properties(spatter PROPERTIES
                COMPILE_DEFINITIONS "${COMMON_COMPILE_DEFINITIONS}"
                COMPILE_OPTIONS "${WARNING_FLAGS}"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
endif()
