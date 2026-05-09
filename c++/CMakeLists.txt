# Require a relatively modern version of CMake
cmake_minimum_required(VERSION 3.28)

# Set the name of this project
project(ElectrodynamicsSimulations CXX)

# Force-enable compiler color diagnostics
# (when Ninja it used, it pipes compiler output which hinders automatic tty detection)
set(GCC_COLOR_DIAGNOSTICS_FLAGS "-fdiagnostics-color")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COLOR_DIAGNOSTICS_FLAGS}")

# Add an executable for the Laguerre-Gauss beam angular momentum transfer experiment
add_executable(LaguerreGaussBeamAngularMomentumTransfer)

# Add source files
target_sources(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        src/main.c++
        src/vector.c++
        src/beam.c++
        src/polarization.c++
        src/integrate.c++
        src/npy_io.c++
        src/angular_momentum.c++
)

# Use a modern C++ standard
set_property(TARGET LaguerreGaussBeamAngularMomentumTransfer PROPERTY CXX_STANDARD 20)

set(USE_OPENMP TRUE CACHE BOOL "Enable OpenMP support")
set(USE_OPENACC FALSE CACHE BOOL "Enable OpenACC support")

message(STATUS "OpenMP support: ${USE_OPENMP}")
message(STATUS "OpenACC support: ${USE_OPENACC}")

# Compiler flags
target_compile_options(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        # Enable as many warnings as possible
        -Wall -Wextra -Wpedantic
        # Treat warnings as errors
        # -Werror
        # Always include debug symbols
        -g
        # Enable OpenMP pragmas support
        $<$<BOOL:${USE_OPENMP}>:
            -fopenmp
        >
        # Enable OpenACC pragmas support
        $<$<BOOL:${USE_OPENACC}>:
            -fopenacc
            -foffload=nvptx-none
        >
        # Ubuntu enables these by default for all C++ compilers;
        # disable them, since they make no sense for HPC code.
        -fcf-protection=none
        -fno-stack-protector
        # Compile and optimize for native (host) architecture
        -march=native
        # Enable some optimizations for debug build
        $<$<CONFIG:Debug>:
            -O1
        >
        # Enable fast math for release build
        $<$<CONFIG:Release>:
            -ffast-math
        >
        # Enable LTO for release build
        $<$<CONFIG:Release>:
            -flto
        >
)

target_link_options(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        # Enable OpenMP runtime library support
        $<$<BOOL:${USE_OPENMP}>:
            -fopenmp
        >
        # Enable OpenACC runtime library support
        $<$<BOOL:${USE_OPENACC}>:
            -fopenacc
        >
        # Enable LTO for release build
        $<$<CONFIG:Release>:
            -flto
        >
)
