# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alunos/tei/2024/tei27386/ProjetoCED

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alunos/tei/2024/tei27386/ProjetoCED/build

# Include any dependencies generated for this target.
include CMakeFiles/canny_sequential.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/canny_sequential.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/canny_sequential.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/canny_sequential.dir/flags.make

CMakeFiles/canny_sequential.dir/canny_sequential.c.o: CMakeFiles/canny_sequential.dir/flags.make
CMakeFiles/canny_sequential.dir/canny_sequential.c.o: ../canny_sequential.c
CMakeFiles/canny_sequential.dir/canny_sequential.c.o: CMakeFiles/canny_sequential.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/canny_sequential.dir/canny_sequential.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/canny_sequential.dir/canny_sequential.c.o -MF CMakeFiles/canny_sequential.dir/canny_sequential.c.o.d -o CMakeFiles/canny_sequential.dir/canny_sequential.c.o -c /home/alunos/tei/2024/tei27386/ProjetoCED/canny_sequential.c

CMakeFiles/canny_sequential.dir/canny_sequential.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/canny_sequential.dir/canny_sequential.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/alunos/tei/2024/tei27386/ProjetoCED/canny_sequential.c > CMakeFiles/canny_sequential.dir/canny_sequential.c.i

CMakeFiles/canny_sequential.dir/canny_sequential.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/canny_sequential.dir/canny_sequential.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/alunos/tei/2024/tei27386/ProjetoCED/canny_sequential.c -o CMakeFiles/canny_sequential.dir/canny_sequential.c.s

# Object files for target canny_sequential
canny_sequential_OBJECTS = \
"CMakeFiles/canny_sequential.dir/canny_sequential.c.o"

# External object files for target canny_sequential
canny_sequential_EXTERNAL_OBJECTS =

canny_sequential: CMakeFiles/canny_sequential.dir/canny_sequential.c.o
canny_sequential: CMakeFiles/canny_sequential.dir/build.make
canny_sequential: CMakeFiles/canny_sequential.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable canny_sequential"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/canny_sequential.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/canny_sequential.dir/build: canny_sequential
.PHONY : CMakeFiles/canny_sequential.dir/build

CMakeFiles/canny_sequential.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/canny_sequential.dir/cmake_clean.cmake
.PHONY : CMakeFiles/canny_sequential.dir/clean

CMakeFiles/canny_sequential.dir/depend:
	cd /home/alunos/tei/2024/tei27386/ProjetoCED/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alunos/tei/2024/tei27386/ProjetoCED /home/alunos/tei/2024/tei27386/ProjetoCED /home/alunos/tei/2024/tei27386/ProjetoCED/build /home/alunos/tei/2024/tei27386/ProjetoCED/build /home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles/canny_sequential.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/canny_sequential.dir/depend

