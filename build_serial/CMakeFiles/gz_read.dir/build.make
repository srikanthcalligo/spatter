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
include CMakeFiles/gz_read.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/gz_read.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/gz_read.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gz_read.dir/flags.make

CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o: CMakeFiles/gz_read.dir/flags.make
CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o: /home/user/cmake_3_30_6_version/spatter/standard-suite/binary-traces/gz_read.cc
CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o: CMakeFiles/gz_read.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o -MF CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o.d -o CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o -c /home/user/cmake_3_30_6_version/spatter/standard-suite/binary-traces/gz_read.cc

CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/standard-suite/binary-traces/gz_read.cc > CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.i

CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/standard-suite/binary-traces/gz_read.cc -o CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.s

# Object files for target gz_read
gz_read_OBJECTS = \
"CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o"

# External object files for target gz_read
gz_read_EXTERNAL_OBJECTS =

gz_read: CMakeFiles/gz_read.dir/standard-suite/binary-traces/gz_read.cc.o
gz_read: CMakeFiles/gz_read.dir/build.make
gz_read: CMakeFiles/gz_read.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gz_read"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gz_read.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gz_read.dir/build: gz_read
.PHONY : CMakeFiles/gz_read.dir/build

CMakeFiles/gz_read.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gz_read.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gz_read.dir/clean

CMakeFiles/gz_read.dir/depend:
	cd /home/user/cmake_3_30_6_version/spatter/build_serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/cmake_3_30_6_version/spatter /home/user/cmake_3_30_6_version/spatter /home/user/cmake_3_30_6_version/spatter/build_serial /home/user/cmake_3_30_6_version/spatter/build_serial /home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles/gz_read.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/gz_read.dir/depend

