module Vacuum

using TOML, Interpolations, SpecialFunctions, LinearAlgebra, Printf

include("VacuumStructs.jl")
include("VacuumInternals.jl")

export mscvac, set_dcon_params, VacuumInput, compute_vacuum_response
export compute_vacuum_field
export kernel!
export WallShapeSettings

# ======================================================================
# Legacy fortran vacuum module interface
# ======================================================================

const libdir = joinpath(@__DIR__, "..", "..", "deps")
const libvac = joinpath(libdir, "libvac")

"""
    set_dcon_params(mtheta, lmin, lmax, nnin, qa1in, xin, zin, deltain)

Initialize DCON parameters for vacuum field calculations.

# Arguments

- `mtheta`: Number of theta grid points (Integer)
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
mtheta, lmin, lmax, nnin = Int32(4), Int32(1), Int32(4), Int32(2)
qa1in = 1.23
n_modes = lmax - lmin + 1
xin = rand(Float64, n_modes)
zin = rand(Float64, n_modes)
deltain = rand(Float64, n_modes)

set_dcon_params(mtheta, lmin, lmax, nnin, qa1in, xin, zin, deltain)
```
"""
function set_dcon_params(mtheta::Integer, lmin::Integer, lmax::Integer, nnin::Integer,
    qa1in::Float64,
    xin::Vector{Float64}, zin::Vector{Float64}, deltain::Vector{Float64})

    return ccall((:set_dcon_params_, libvac),
        Nothing,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cint},
            Ref{Cdouble},
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
        Ref(Int32(mtheta)), Ref(Int32(lmin)), Ref(Int32(lmax)), Ref(Int32(nnin)),
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
set_dcon_params(mtheta, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Reset DCON parameters
unset_dcon_params()
```
"""
function unset_dcon_params()
    ccall((:unset_dcon_params_, libvac), Nothing, ())
end

"""
    mscvac(wv, mpert, mtheta, mtheta_vacuum, complex_flag, kernelsignin, wall_flag, farwall_flag, grrio, xzptso, op_ahgfile=nothing)

Compute the vacuum response matrix for magnetostatic perturbations.

# Arguments

- `wv`: Pre-allocated complex matrix (mpert × mpert) to store vacuum response (Array{ComplexF64,2})
- `mpert`: Number of perturbation modes (Integer)
- `mtheta`: Number of theta grid points for plasma (Integer)
- `mtheta_vacuum`: Number of theta grid points for vacuum region (Integer)
- `complex_flag`: Whether to use complex arithmetic (Bool)
- `kernelsignin`: Sign convention for vacuum kernels (Float64, typically -1.0)
- `wall_flag`: Whether to include an externally defined wall shape (Bool)
- `farwall_flag`: Whether to use far-wall approximation (Bool)
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
set_dcon_params(mtheta, lmin, lmax, nnin, qa1in, xin, zin, deltain)

# Set up vacuum calculation
mpert = 5
mtheta = 256
mtheta_vacuum = 256
wv = zeros(ComplexF64, mpert, mpert)
complex_flag = true
kernelsignin = -1.0
wall_flag = false
farwall_flag = true
grrio = rand(Float64, 2 * (mtheta_vacuum + 5), mpert * 2)
xzptso = rand(Float64, mtheta_vacuum + 5, 4)

# Perform calculation
mscvac(wv, mpert, mtheta, mtheta_vacuum, complex_flag, kernelsignin,
    wall_flag, farwall_flag, grrio, xzptso)
```
"""
function mscvac(
    wv::Array{ComplexF64,2},
    mpert::Integer,
    mtheta::Integer,
    mtheta_vacuum::Integer,
    complex_flag::Bool,
    kernelsignin::Float64,
    wall_flag::Bool,
    farwall_flag::Bool,
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
                Ref{Cint},            # mtheta_vacuum
                Ref{Cint},            # complex_flag (logical)
                Ref{Cdouble},         # kernelsignin
                Ref{Cint},            # wall_flag (logical)
                Ref{Cint},            # farwall_flag (logical)
                Ptr{Cdouble},         # grrio(:,:)
                Ptr{Cdouble},         # xzptso(:,:)
                Ptr{UInt8}),          # op_ahgfile (optional)
            pointer(wv),
            Ref(Int32(mpert)),
            Ref(Int32(mtheta)),
            Ref(Int32(mtheta_vacuum)),
            Ref(Int32(complex_flag)),
            Ref(kernelsignin),
            Ref(Int32(wall_flag)),
            Ref(Int32(farwall_flag)),
            pointer(grrio),
            pointer(xzptso),
            ahgfile_ptr
        )
    end

    return wv
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Compute the vacuum response matrix using provided vacuum inputs. 

This is the pure Julia implementation that replaces the Fortran `mscvac` function. 
It returns the relevant arrays: `wv`, `grri`, and `xzpts`.

# Arguments

- `inputs::VacuumInput`: Struct containing vacuum calculation parameters including mode numbers, 
  grid resolution, toroidal mode number, and plasma boundary information.
- `wall_settings::WallShapeSettings`: Struct specifying the wall geometry configuration.

# Returns

- `wv`: Complex vacuum response matrix (mpert × mpert) relating plasma perturbations to vacuum response
- `grri`: Green's function response matrix (2*mtheta × 2*mpert) in Fourier space
- `xzpts`: Coordinate array (mtheta × 4) containing [R_plasma, Z_plasma, R_wall, Z_wall]

# Notes

- This function initializes the plasma surface and wall geometry internally
- The vacuum response includes plasma-plasma and plasma-wall coupling effects
- For n=0 modes with closed walls, a regularization factor is added to prevent singularities
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

    # Initialization and allocations
    (; mtheta, mpert, n, kernelsign, force_wv_symmetry) = inputs
    plasma_surf = initialize_plasma_surface(inputs)
    wall = initialize_wall(inputs, plasma_surf, wall_settings)
    grri = zeros(2 * mtheta, 2 * mpert)
    grad_greenfunction_mat = zeros(2 * mtheta, 2 * mtheta)
    greenfunction_temp = zeros(mtheta, mtheta)

    # Plasma–Plasma block
    j1, j2 = 1, 1
    kernel!(grad_greenfunction_mat, greenfunction_temp, plasma_surf.x, plasma_surf.z, plasma_surf.x, plasma_surf.z, j1, j2, 1, n)

    # Fourier transform plasma-plasma block
    fourier_transform!(grri, greenfunction_temp, plasma_surf.cslth, 0, 0)
    fourier_transform!(grri, greenfunction_temp, plasma_surf.snlth, 0, mpert)

    !wall.nowall && begin
        @warn "Vacuum response calculations with wall are open-beta and still being benchmarked for accuracy."
        # Plasma–Wall block
        j1, j2 = 1, 2
        kernel!(grad_greenfunction_mat, greenfunction_temp, plasma_surf.x, plasma_surf.z, wall.x, wall.z, j1, j2, 0, n)

        # Wall–Wall block
        j1, j2 = 2, 2
        kernel!(grad_greenfunction_mat, greenfunction_temp, wall.x, wall.z, wall.x, wall.z, j1, j2, 0, n)

        # Wall–Plasma block
        j1, j2 = 2, 1
        kernel!(grad_greenfunction_mat, greenfunction_temp, wall.x, wall.z, plasma_surf.x, plasma_surf.z, j1, j2, 1, n)

        # Fourier transform wall blocks into grri
        fourier_transform!(grri, greenfunction_temp, plasma_surf.cslth, mtheta, 0)
        fourier_transform!(grri, greenfunction_temp, plasma_surf.snlth, mtheta, mpert)
    end

    # Add cn0 to make grdgre nonsingular for n=0 modes
    cn0 = 1.0 # expose to user if anyone ever actually tries to use this
    (abs(n) <= 1e-5 && !wall.nowall && wall.is_closed_toroidal) && begin
        @warn "Adding $cn0 to diagonal of grdgre to regularize n=0 mode; this may affect accuracy of results."
        mth12 = wall.nowall ? mtheta : 2 * mtheta
        for i in 1:mth12, j in 1:mth12
            grad_greenfunction_mat[i, j] += cn0
        end
    end

    # Only needed for mutual inductance with the wall calculations
    (kernelsign < 0) && begin
        grad_greenfunction_mat .*= kernelsign
        # Account for factor of 2 in diagonal terms in eq. 90 of Chance
        for i in 1:2 * mtheta
            grad_greenfunction_mat[i, i] += 2.0
        end
    end

    # Invert the vacuum response system of equations, eqs. 92-94ish of Chance 1997 (gelimb in Fortran)
    # If plasma only, lower blocks will be empty
    if wall.nowall
        @views grri[1:mtheta, :] .= grad_greenfunction_mat[1:mtheta, 1:mtheta] \ grri[1:mtheta, :]
    else
        grri .= grad_greenfunction_mat \ grri
    end

    # There's some logic that computes xpass/zpass and chiwc/chiws here, might eventually be needed?

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr = zeros(mpert, mpert)
    aii = zeros(mpert, mpert)
    ari = zeros(mpert, mpert)
    air = zeros(mpert, mpert)
    fourier_inverse_transform!(arr, grri, plasma_surf.cslth, 0, 0)
    fourier_inverse_transform!(aii, grri, plasma_surf.snlth, 0, mpert)
    fourier_inverse_transform!(ari, grri, plasma_surf.snlth, 0, 0)
    fourier_inverse_transform!(air, grri, plasma_surf.cslth, 0, mpert)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    vacmat = arr .+ aii
    vacmti = air .- ari
    # Force symmetry of response matrix if desired
    force_wv_symmetry && begin
        for l1 in 1:mpert
            for l2 in l1:mpert
                vacmat[l1, l2] = 0.5 * (vacmat[l1, l2] + vacmat[l2, l1])
                vacmti[l1, l2] = 0.5 * (vacmti[l1, l2] - vacmti[l2, l1])
            end
        end
    end
    wv = complex.(vacmat, vacmti)

    # Create xzpts array
    xzpts = zeros(Float64, inputs.mtheta, 4)
    @views xzpts[:, 1] .= plasma_surf.x
    @views xzpts[:, 2] .= plasma_surf.z
    @views xzpts[:, 3] .= wall.x
    @views xzpts[:, 4] .= wall.z
    return wv, grri, xzpts
end

"""
    compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry, 
           Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

Calculate the perturbed magnetic field in the vacuum region resulting from a normal
magnetic field perturbation (`Bn`) at the plasma surface. Replaces `mscfld` from Fortran.

This function orchestrates the vacuum field calculation by:
1. Calling `vaccal!` to compute the vacuum response kernel (`grri`)
2. Defining a grid of points (`R_grid`, `Z_grid`) where the field is to be calculated
3. Calling `_pickup_field` to compute the magnetic field components on that grid using the kernel
   and the source perturbation `Bn`

# Arguments

- `inputs::VacuumInput`: Struct containing vacuum calculation parameters (n, mpert, mtheta, etc.)
- `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions
- `wall::WallGeometry`: Struct with wall geometry
- `Bn::Vector{<:Number}`: Complex vector of Fourier harmonics of the normal magnetic field
  perturbation at the plasma surface, `B_n = B_n_real + i*B_n_imag`. Length must be `mpert`.
- `R_grid::AbstractVector`: Vector of R coordinates for the output field grid
- `Z_grid::AbstractVector`: Vector of Z coordinates for the output field grid

# Returns

- `B_R::Matrix{ComplexF64}`: The R-component of the magnetic field on the grid
- `B_Z::Matrix{ComplexF64}`: The Z-component of the magnetic field on the grid
- `B_phi::Matrix{ComplexF64}`: The toroidal component of the magnetic field on the grid
- `grid_info::Matrix{Int}`: Information about the grid points (1=inside plasma, 0=outside)
"""
function compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry, 
                Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

    # 1. Call vaccal! to get the inverted Green's function matrix
    # The Fortran version calls the whole chain (ent33 -> vaccal), 
    # here we assume vaccal! provides what we need.
    wv, grri = vaccal!(inputs, plasma_surf, wall)

    # Separate real and imaginary parts of the source perturbation
    Bn_real = real.(Bn)
    Bn_imag = imag.(Bn)

    # 2. Define grid and parameters for pickup routine
    nx = length(R_grid)
    nz = length(Z_grid)
    
    # 3. Call the field pickup routine
    B_R, B_Z, B_phi, grid_info = _pickup_field(
        inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid
    )

    return B_R, B_Z, B_phi, grid_info
end
end