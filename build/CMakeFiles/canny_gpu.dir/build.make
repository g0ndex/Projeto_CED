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
include CMakeFiles/canny_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/canny_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/canny_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/canny_gpu.dir/flags.make

CMakeFiles/canny_gpu.dir/canny_gpu.cu.o: CMakeFiles/canny_gpu.dir/flags.make
CMakeFiles/canny_gpu.dir/canny_gpu.cu.o: ../canny_gpu.cu
CMakeFiles/canny_gpu.dir/canny_gpu.cu.o: CMakeFiles/canny_gpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/canny_gpu.dir/canny_gpu.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/canny_gpu.dir/canny_gpu.cu.o -MF CMakeFiles/canny_gpu.dir/canny_gpu.cu.o.d -x cu -c /home/alunos/tei/2024/tei27386/ProjetoCED/canny_gpu.cu -o CMakeFiles/canny_gpu.dir/canny_gpu.cu.o

CMakeFiles/canny_gpu.dir/canny_gpu.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/canny_gpu.dir/canny_gpu.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/canny_gpu.dir/canny_gpu.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/canny_gpu.dir/canny_gpu.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target canny_gpu
canny_gpu_OBJECTS = \
"CMakeFiles/canny_gpu.dir/canny_gpu.cu.o"

# External object files for target canny_gpu
canny_gpu_EXTERNAL_OBJECTS =

canny_gpu: CMakeFiles/canny_gpu.dir/canny_gpu.cu.o
canny_gpu: CMakeFiles/canny_gpu.dir/build.make
canny_gpu: CMakeFiles/canny_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable canny_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/canny_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/canny_gpu.dir/build: canny_gpu
.PHONY : CMakeFiles/canny_gpu.dir/build

CMakeFiles/canny_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/canny_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/canny_gpu.dir/clean

CMakeFiles/canny_gpu.dir/depend:
	cd /home/alunos/tei/2024/tei27386/ProjetoCED/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alunos/tei/2024/tei27386/ProjetoCED /home/alunos/tei/2024/tei27386/ProjetoCED /home/alunos/tei/2024/tei27386/ProjetoCED/build /home/alunos/tei/2024/tei27386/ProjetoCED/build /home/alunos/tei/2024/tei27386/ProjetoCED/build/CMakeFiles/canny_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/canny_gpu.dir/depend

