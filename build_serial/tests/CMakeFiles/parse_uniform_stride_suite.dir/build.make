# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/user/cmake_3_30_6_version/cmake-3.30.6-build/bin/cmake

# The command to remove a file.
RM = /home/user/cmake_3_30_6_version/cmake-3.30.6-build/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/user/cmake_3_30_6_version/spatter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/user/cmake_3_30_6_version/spatter/build_serial

# Include any dependencies generated for this target.
include tests/CMakeFiles/parse_uniform_stride_suite.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/parse_uniform_stride_suite.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/parse_uniform_stride_suite.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/parse_uniform_stride_suite.dir/flags.make

tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o: tests/CMakeFiles/parse_uniform_stride_suite.dir/flags.make
tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o: /home/user/cmake_3_30_6_version/spatter/tests/parse_uniform_stride_suite.cc
tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o: tests/CMakeFiles/parse_uniform_stride_suite.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o -MF CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o.d -o CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o -c /home/user/cmake_3_30_6_version/spatter/tests/parse_uniform_stride_suite.cc

tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.i"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/tests/parse_uniform_stride_suite.cc > CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.i

tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.s"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/tests/parse_uniform_stride_suite.cc -o CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.s

# Object files for target parse_uniform_stride_suite
parse_uniform_stride_suite_OBJECTS = \
"CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o"

# External object files for target parse_uniform_stride_suite
parse_uniform_stride_suite_EXTERNAL_OBJECTS =

tests/parse_uniform_stride_suite: tests/CMakeFiles/parse_uniform_stride_suite.dir/parse_uniform_stride_suite.cc.o
tests/parse_uniform_stride_suite: tests/CMakeFiles/parse_uniform_stride_suite.dir/build.make
tests/parse_uniform_stride_suite: src/Spatter/libSpatter.a
tests/parse_uniform_stride_suite: tests/CMakeFiles/parse_uniform_stride_suite.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable parse_uniform_stride_suite"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parse_uniform_stride_suite.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/parse_uniform_stride_suite.dir/build: tests/parse_uniform_stride_suite
.PHONY : tests/CMakeFiles/parse_uniform_stride_suite.dir/build

tests/CMakeFiles/parse_uniform_stride_suite.dir/clean:
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/tests && $(CMAKE_COMMAND) -P CMakeFiles/parse_uniform_stride_suite.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/parse_uniform_stride_suite.dir/clean

tests/CMakeFiles/parse_uniform_stride_suite.dir/depend:
	cd /home/user/cmake_3_30_6_version/spatter/build_serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/cmake_3_30_6_version/spatter /home/user/cmake_3_30_6_version/spatter/tests /home/user/cmake_3_30_6_version/spatter/build_serial /home/user/cmake_3_30_6_version/spatter/build_serial/tests /home/user/cmake_3_30_6_version/spatter/build_serial/tests/CMakeFiles/parse_uniform_stride_suite.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : tests/CMakeFiles/parse_uniform_stride_suite.dir/depend

