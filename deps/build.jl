# How to modify build.jl:
# 1) Declare build functions for each Fortran codes as 'build_*_fortran()'
# 2) Add these functions to the results array in build_fortran()
# 3) Export the functions you want to be accessible from outside


const parent_dir = joinpath(@__DIR__, "..", "src")

include(joinpath(@__DIR__, "build_helpers.jl"))

export build_fortran
export build_spline_fortran, build_vacuum_fortran

build_fortran() # Call the build function to execute the builds
