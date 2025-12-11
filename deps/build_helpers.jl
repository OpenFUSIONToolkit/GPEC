# How to modify build_helpers.jl:
# 1) Declare build functions for each Fortran codes as 'build_*_fortran()'
# 2) Add these functions to the results array in build_fortran()
# 3) Export the functions you want to be accessible from outside


const parent_dir = joinpath(@__DIR__, "..", "src")

export build_spline_fortran, build_vacuum_fortran

function build_fortran()
    ENV["FC"] = get(ENV, "FC", "gfortran")
    # Set OS-dependent flags
    println("Setting up environment for ", Sys.KERNEL)
    if Sys.isapple()
        ENV["LIBS"] = "-framework Accelerate"
        ENV["LIBSUFFIX"] = ".dylib"
    elseif Sys.islinux()
        ENV["LIBS"] = "-lopenblas"
        ENV["LIBSUFFIX"] = ".so"
    elseif Sys.iswindows()
        error("Unsupported OS for Fortran build- try using WSL")
        #ENV["LIBS"] = "-lopenblas"
        #ENV["LIBSUFFIX"] = ".dll"
    else
        error("Unsupported OS for Fortran build")
    end
    ENV["RECURSFLAG"] = "-frecursive"
    ENV["LEGACYFLAG"] = "-std=legacy"
    ENV["ZEROFLAG"] = "-finit-local-zero"
    ENV["FFLAGS"] = "-fPIC"
    ENV["LDFLAGS"] = "-shared"

    results = [
        # build_jpec_fortran() add here
        build_spline_fortran(),
        build_vacuum_fortran()
    ]

    if all(results)
        @info "All Fortran builds succeeded!"
    else
        @error "Some Fortran builds failed."
    end

end

function build_spline_fortran()
    dir = joinpath(parent_dir, "Splines", "fortran")
    try
        run(pipeline(`make -C $dir`))
        @info "Splines-fortran compiled well"
        return true
    catch e
        @error "Failed to build Splines-fortran: $e"
        return false
    end
end

function build_vacuum_fortran()
    dir = joinpath(parent_dir, "Vacuum", "fortran")
    try
        run(pipeline(`make -C $dir`))
        @info "Vacuum-fortran compiled well"
        return true
    catch e
        @error "Failed to build Vacuum-fortran: $e"
        return false
    end

end

# Example for including more fortran:
#
# function build_jpec_fortran()
#     dir = joinpath(parent_dir, "Gpec", "fortran")
#     try
#         run(pipeline(`make -C $dir`, stdout=DevNull, stderr=DevNull))
#         println("Gpec-fortran compiled well")
#         return true
#     catch e
#         println("Failed to build Gpec-fortran: $e")
#         return false
#     end
# end

