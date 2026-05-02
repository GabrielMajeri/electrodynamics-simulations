# Require a relatively modern version of CMake
cmake_minimum_required(VERSION 4.2)

# Set the name of this project
project(ElectrodynamicsSimulations)

# Force-enable compiler color diagnostics
# (when Ninja it used, it pipes compiler output which hinders automatic tty detection)
set(GCC_COLOR_DIAGNOSTICS_FLAGS "-fdiagnostics-color")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COLOR_DIAGNOSTICS_FLAGS}")

# Add an executable for the Laguerre-Gauss beam angular momentum transfer experiment
add_executable(LaguerreGaussBeamAngularMomentumTransfer)

# Add source files
target_sources(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        c++/main.c++
)

# Use a modern C++ standard
set_property(TARGET LaguerreGaussBeamAngularMomentumTransfer PROPERTY CXX_STANDARD 20)

# Compiler flags
target_compile_options(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        # Enable as many warnings as possible
        -Wall -Wextra -Wpedantic
        # Treat warnings as errors
        # -Werror
        # Enable OpenMP pragmas support
        -fopenmp
        # Enable some optimizations for debug build
        $<$<CONFIG:Debug>:
            -O1
        >
        # Enable fast math for release build
        $<$<CONFIG:Release>:
            -ffast-math
        >
)

target_link_options(LaguerreGaussBeamAngularMomentumTransfer
    PRIVATE
        # Enable OpenMP runtime library support
        -fopenmp
)
