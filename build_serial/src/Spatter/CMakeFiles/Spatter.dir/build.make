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
include src/Spatter/CMakeFiles/Spatter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/Spatter/CMakeFiles/Spatter.dir/compiler_depend.make

# Include the progress variables for this target.
include src/Spatter/CMakeFiles/Spatter.dir/progress.make

# Include the compile flags for this target's objects.
include src/Spatter/CMakeFiles/Spatter.dir/flags.make

src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o: src/Spatter/CMakeFiles/Spatter.dir/flags.make
src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o: /home/user/cmake_3_30_6_version/spatter/src/Spatter/Configuration.cc
src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o: src/Spatter/CMakeFiles/Spatter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o -MF CMakeFiles/Spatter.dir/Configuration.cc.o.d -o CMakeFiles/Spatter.dir/Configuration.cc.o -c /home/user/cmake_3_30_6_version/spatter/src/Spatter/Configuration.cc

src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Spatter.dir/Configuration.cc.i"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/src/Spatter/Configuration.cc > CMakeFiles/Spatter.dir/Configuration.cc.i

src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Spatter.dir/Configuration.cc.s"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/src/Spatter/Configuration.cc -o CMakeFiles/Spatter.dir/Configuration.cc.s

src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o: src/Spatter/CMakeFiles/Spatter.dir/flags.make
src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o: /home/user/cmake_3_30_6_version/spatter/src/Spatter/JSONParser.cc
src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o: src/Spatter/CMakeFiles/Spatter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o -MF CMakeFiles/Spatter.dir/JSONParser.cc.o.d -o CMakeFiles/Spatter.dir/JSONParser.cc.o -c /home/user/cmake_3_30_6_version/spatter/src/Spatter/JSONParser.cc

src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Spatter.dir/JSONParser.cc.i"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/src/Spatter/JSONParser.cc > CMakeFiles/Spatter.dir/JSONParser.cc.i

src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Spatter.dir/JSONParser.cc.s"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/src/Spatter/JSONParser.cc -o CMakeFiles/Spatter.dir/JSONParser.cc.s

src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o: src/Spatter/CMakeFiles/Spatter.dir/flags.make
src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o: /home/user/cmake_3_30_6_version/spatter/src/Spatter/PatternParser.cc
src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o: src/Spatter/CMakeFiles/Spatter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o -MF CMakeFiles/Spatter.dir/PatternParser.cc.o.d -o CMakeFiles/Spatter.dir/PatternParser.cc.o -c /home/user/cmake_3_30_6_version/spatter/src/Spatter/PatternParser.cc

src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Spatter.dir/PatternParser.cc.i"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/src/Spatter/PatternParser.cc > CMakeFiles/Spatter.dir/PatternParser.cc.i

src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Spatter.dir/PatternParser.cc.s"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/src/Spatter/PatternParser.cc -o CMakeFiles/Spatter.dir/PatternParser.cc.s

src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o: src/Spatter/CMakeFiles/Spatter.dir/flags.make
src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o: /home/user/cmake_3_30_6_version/spatter/src/Spatter/Timer.cc
src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o: src/Spatter/CMakeFiles/Spatter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o -MF CMakeFiles/Spatter.dir/Timer.cc.o.d -o CMakeFiles/Spatter.dir/Timer.cc.o -c /home/user/cmake_3_30_6_version/spatter/src/Spatter/Timer.cc

src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Spatter.dir/Timer.cc.i"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/user/cmake_3_30_6_version/spatter/src/Spatter/Timer.cc > CMakeFiles/Spatter.dir/Timer.cc.i

src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Spatter.dir/Timer.cc.s"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/user/cmake_3_30_6_version/spatter/src/Spatter/Timer.cc -o CMakeFiles/Spatter.dir/Timer.cc.s

# Object files for target Spatter
Spatter_OBJECTS = \
"CMakeFiles/Spatter.dir/Configuration.cc.o" \
"CMakeFiles/Spatter.dir/JSONParser.cc.o" \
"CMakeFiles/Spatter.dir/PatternParser.cc.o" \
"CMakeFiles/Spatter.dir/Timer.cc.o"

# External object files for target Spatter
Spatter_EXTERNAL_OBJECTS =

src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/Configuration.cc.o
src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/JSONParser.cc.o
src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/PatternParser.cc.o
src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/Timer.cc.o
src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/build.make
src/Spatter/libSpatter.a: src/Spatter/CMakeFiles/Spatter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/user/cmake_3_30_6_version/spatter/build_serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libSpatter.a"
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && $(CMAKE_COMMAND) -P CMakeFiles/Spatter.dir/cmake_clean_target.cmake
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Spatter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/Spatter/CMakeFiles/Spatter.dir/build: src/Spatter/libSpatter.a
.PHONY : src/Spatter/CMakeFiles/Spatter.dir/build

src/Spatter/CMakeFiles/Spatter.dir/clean:
	cd /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter && $(CMAKE_COMMAND) -P CMakeFiles/Spatter.dir/cmake_clean.cmake
.PHONY : src/Spatter/CMakeFiles/Spatter.dir/clean

src/Spatter/CMakeFiles/Spatter.dir/depend:
	cd /home/user/cmake_3_30_6_version/spatter/build_serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/user/cmake_3_30_6_version/spatter /home/user/cmake_3_30_6_version/spatter/src/Spatter /home/user/cmake_3_30_6_version/spatter/build_serial /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter /home/user/cmake_3_30_6_version/spatter/build_serial/src/Spatter/CMakeFiles/Spatter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/Spatter/CMakeFiles/Spatter.dir/depend

