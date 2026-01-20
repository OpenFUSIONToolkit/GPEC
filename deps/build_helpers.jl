# How to modify build_helpers.jl:
# 1) Declare build functions for each Fortran codes as 'build_*_fortran()'
# 2) Add these functions to the results array in build_fortran()
# 3) Export the functions you want to be accessible from outside


const parent_dir = joinpath(@__DIR__, "..", "src")

export build_spline_fortran, build_vacuum_fortran, build_biest

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
        build_vacuum_fortran(),
        build_biest()
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

function build_biest()
    dir = joinpath(parent_dir, "BIEST")
    try
        # Build BIEST library
        run(pipeline(`make -C $dir`))
        @info "BIEST compiled successfully"

        # Copy the compiled library to the lib directory
        lib_dir = joinpath(parent_dir, "BIEST", "lib")
        if !isdir(lib_dir)
            mkdir(lib_dir)
        end

        # Look for the compiled shared library and copy it
        lib_file = joinpath(dir, "libbiest.so")
        if Sys.isapple()
            lib_file = joinpath(dir, "libbiest.dylib")
        end

        if isfile(lib_file)
            cp(lib_file, joinpath(lib_dir, basename(lib_file)); force=true)
            @info "BIEST library copied to $lib_dir"
        end

        return true
    catch e
        @warn "BIEST build may have issues (this is not critical): $e"
        @warn "BIEST functionality from Julia will not be available unless the library is compiled separately"
        @warn "To manually compile BIEST, run: cd src/BIEST && make"
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
