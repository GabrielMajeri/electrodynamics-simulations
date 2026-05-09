# Require a relatively modern version of CMake
cmake_minimum_required(VERSION 3.28)

# Set the name of this project
project(ElectrodynamicsSimulations LANGUAGES Fortran)

# Force-enable compiler color diagnostics
# (when Ninja it used, it pipes compiler output which hinders automatic tty detection)
set(GCC_COLOR_DIAGNOSTICS_FLAGS "-fdiagnostics-color")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${GCC_COLOR_DIAGNOSTICS_FLAGS}")

# Add an executable for the Laguerre-Gauss beam angular momentum transfer experiment
add_executable(LaguerreGaussBeamAngularMomentumTransfer)

# Add source files
target_sources(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        src/main.f90
        src/constants.f90
        src/types.f90
        src/vector.f90
        src/initial_conditions_generation.f90
        src/numpy.f90
        src/beam.f90
        src/integrate.f90
        src/angular_momentum.f90
)

# Use a modern Fortran standard
set_property(TARGET LaguerreGaussBeamAngularMomentumTransfer PROPERTY FORTRAN_STANDARD 2008)

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
        # Enable backtraces
        -fbacktrace
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
