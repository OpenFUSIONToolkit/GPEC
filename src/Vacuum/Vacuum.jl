module VacuumMod

export mscvac, set_dcon_params


const libdir = joinpath(@__DIR__, "..", "..", "deps")
const libvac = joinpath(libdir, "libvac")

"""
    set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)

Initialize DCON parameters for vacuum field calculations.

# Arguments

  - `mthin`: Number of theta grid points (Integer)
  - `lmin`: Minimum poloidal mode number (Integer)
  - `lmax`: Maximum poloidal mode number (Integer)
  - `nnin`: Toroidal mode number (Integer)
  - `qa1in`: Safety factor parameter (Float64)
  - `xin`: Vector of radial coordinates at plasma boundary (Vector{Float64})
  - `zin`: Vector of vertical coordinates at plasma boundary (Vector{Float64})
  - `deltain`: Vector of displacement values (Vector{Float64})

# Note

This function must be called before using `mscvac` to perform vacuum calculations.
The coordinate and displacement vectors should have length `lmax - lmin + 1`.

# Examples

```julia
mthin, lmin, lmax, nnin = Int32(4), Int32(1), Int32(4), Int32(2)
qa1in = 1.23
n_modes = lmax - lmin + 1
xin = rand(Float64, n_modes)
zin = rand(Float64, n_modes)
deltain = rand(Float64, n_modes)

set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)
```
"""
function set_dcon_params(mthin::Integer, lmin::Integer, lmax::Integer, nnin::Integer,
    qa1in::Float64,
    xin::Vector{Float64}, zin::Vector{Float64}, deltain::Vector{Float64})

    return ccall((:set_dcon_params_, libvac),
        Nothing,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint},
            Ref{Cdouble},
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        Ref(Int32(mthin)), Ref(Int32(lmin)), Ref(Int32(lmax)), Ref(Int32(nnin)),
        Ref(qa1in),
        pointer(xin), pointer(zin), pointer(deltain))
end

"""
    unset_dcon_params()

Unset DCON parameters previously set by `set_dcon_params`.

This subroutine deallocates in-memory arrays (`x_dcon`, `z_dcon`, and `delta_dcon`)
and resets the internal DCON state for future vacuum calculations.

# Notes

  - Must be called after `set_dcon_params` if you want to reset the DCON memory.
  - No arguments are required.

# Example

```julia
# Set parameters
set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Reset DCON parameters
unset_dcon_params()
```
"""
function unset_dcon_params()
    ccall((:unset_dcon_params_, libvac), Nothing, ())
end

"""
    mscvac(wv, mpert, mtheta, mthvac, complex_flag, kernelsignin, wall_flag, farwal_flag, grrio, xzptso, op_ahgfile=nothing)

Compute the vacuum response matrix for magnetostatic perturbations.

# Arguments

  - `wv`: Pre-allocated complex matrix (mpert × mpert) to store vacuum response (Array{ComplexF64,2})
  - `mpert`: Number of perturbation modes (Integer)
  - `mtheta`: Number of theta grid points for plasma (Integer)
  - `mthvac`: Number of theta grid points for vacuum region (Integer)
  - `complex_flag`: Whether to use complex arithmetic (Bool)
  - `kernelsignin`: Sign convention for vacuum kernels (Float64, typically -1.0)
  - `wall_flag`: Whether to include an externally defined wall shape (Bool)
  - `farwal_flag`: Whether to use far-wall approximation (Bool)
  - `grrio`: Green's function data (Array{Float64,2})
  - `xzptso`: Source point coordinates (Array{Float64,2})
  - `op_ahgfile`: Optional communication file for when set_dcon_params is not called (String or Nothing)

# Returns

  - Modifies `wv` in-place with the computed vacuum response matrix
  - Returns the modified `wv` matrix

# Note

Requires prior initialization with `set_dcon_params()` before calling this function.

# Examples

```julia
# Initialize parameters first
set_dcon_params(mthin, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Set up vacuum calculation
mpert = 5
mtheta = 256
mthvac = 256
wv = zeros(ComplexF64, mpert, mpert)
complex_flag = true
kernelsignin = -1.0
wall_flag = false
farwal_flag = true
grrio = rand(Float64, 2 * (mthvac + 5), mpert * 2)
xzptso = rand(Float64, mthvac + 5, 4)

# Perform calculation
mscvac(wv, mpert, mtheta, mthvac, complex_flag, kernelsignin,
    wall_flag, farwal_flag, grrio, xzptso)
```
"""
function mscvac(
    wv::Array{ComplexF64,2},
    mpert::Integer,
    mtheta::Integer,
    mthvac::Integer,
    complex_flag::Bool,
    kernelsignin::Float64,
    wall_flag::Bool,
    farwal_flag::Bool,
    grrio::Array{Float64,2},
    xzptso::Array{Float64,2},
    op_ahgfile::Union{Nothing,String}=nothing,
    folder::String="."
)

    ahgfile_ptr = if op_ahgfile === nothing
        Ptr{UInt8}(C_NULL)
    else
        Base.cconvert(Ptr{UInt8}, op_ahgfile)
    end

    # TODO: this allows VACUUM to be called from a specified folder, like the rest of the Julia DCON
    # vac.in is hardcoded in the fortran, so this just changes the working directory temporarily for VACUUM
    # Eventually, find a better way to do this
    cd(folder) do
        return ccall((:__vacuum_mod_MOD_mscvac, libvac),
            Nothing,
            (Ptr{ComplexF64},       # wv(mpert,mpert)
                Ref{Cint},            # mpert
                Ref{Cint},            # mtheta
                Ref{Cint},            # mthvac
                Ref{Cint},            # complex_flag (logical)
                Ref{Cdouble},         # kernelsignin
                Ref{Cint},            # wall_flag (logical)
                Ref{Cint},            # farwal_flag (logical)
                Ptr{Cdouble},         # grrio(:,:)
                Ptr{Cdouble},         # xzptso(:,:)
                Ptr{UInt8}),          # op_ahgfile (optional)
            pointer(wv),
            Ref(Int32(mpert)),
            Ref(Int32(mtheta)),
            Ref(Int32(mthvac)),
            Ref(Int32(complex_flag)),
            Ref(kernelsignin),
            Ref(Int32(wall_flag)),
            Ref(Int32(farwal_flag)),
            pointer(grrio),
            pointer(xzptso),
            ahgfile_ptr
        )
    end

    return wv
end

end