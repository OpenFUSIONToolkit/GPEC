module Vacuum

using TOML, Interpolations, SpecialFunctions, LinearAlgebra, Printf
using StaticArrays
using FastGaussQuadrature
using ..BIEST

include("VacuumStructs.jl")
include("VacuumInternals.jl")
include("Vacuum3D.jl")

export mscvac, set_dcon_params, VacuumInput, compute_vacuum_response, compute_vacuum_response_3D
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
    grad_green = zeros(2 * mtheta, 2 * mtheta)
    green_temp = zeros(mtheta, mtheta)

    # 𝒢ₗ(θⱼ) from Chance eq. 106-108. first mtheta rows are plasma as observer, second are wall
    # First mpert columns are real (cosine), second mpert are imaginary (sine)
    green_fourier = zeros(2 * mtheta, 2 * mpert)
    PLASMA_ROW_OFFSET = 0
    WALL_ROW_OFFSET = mtheta
    COS_COL_OFFSET = 0
    SIN_COL_OFFSET = mpert

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, n)

    # Fourier transform plasma-plasma block
    fourier_transform!(green_fourier, green_temp, plasma_surf.cos_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(green_fourier, green_temp, plasma_surf.sin_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)

    !wall.nowall && begin
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, n)

        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, n)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, n)

        # Fourier transform wall blocks into green_fourier
        fourier_transform!(green_fourier, green_temp, plasma_surf.cos_mn_basis, WALL_ROW_OFFSET, COS_COL_OFFSET)
        fourier_transform!(green_fourier, green_temp, plasma_surf.sin_mn_basis, WALL_ROW_OFFSET, SIN_COL_OFFSET)
    end

    # Add cn0 to make grdgre nonsingular for n=0 modes
    cn0 = 1.0 # expose to user if anyone ever actually tries to use this
    (abs(n) <= 1e-5 && !wall.nowall && wall.is_closed_toroidal) && begin
        @warn "Adding $cn0 to diagonal of grdgre to regularize n=0 mode; this may affect accuracy of results."
        mth12 = wall.nowall ? mtheta : 2 * mtheta
        for i in 1:mth12, j in 1:mth12
            grad_green[i, j] += cn0
        end
    end

    # Only needed for mutual inductance with the wall calculations
    (kernelsign < 0) && begin
        grad_green .*= kernelsign
        # Account for factor of 2 in diagonal terms in eq. 90 of Chance
        for i in 1:(2*mtheta)
            grad_green[i, i] += 2.0
        end
    end

    # Invert the vacuum response system of equations, eqs. 92-94ish of Chance 1997 (gelimb in Fortran)
    # If plasma only, lower blocks will be empty
    if wall.nowall
        @views green_fourier[1:mtheta, :] .= grad_green[1:mtheta, 1:mtheta] \ green_fourier[1:mtheta, :]
    else
        green_fourier .= grad_green \ green_fourier
    end

    # There's some logic that computes xpass/zpass and chiwc/chiws here, might eventually be needed?

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr, aii, ari, air = ntuple(_ -> zeros(mpert, mpert), 4)
    fourier_inverse_transform!(arr, green_fourier, plasma_surf.cos_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET, 4π^2 / mtheta)
    fourier_inverse_transform!(aii, green_fourier, plasma_surf.sin_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET, 4π^2 / mtheta)
    fourier_inverse_transform!(ari, green_fourier, plasma_surf.sin_mn_basis, PLASMA_ROW_OFFSET, COS_COL_OFFSET, 4π^2 / mtheta)
    fourier_inverse_transform!(air, green_fourier, plasma_surf.cos_mn_basis, PLASMA_ROW_OFFSET, SIN_COL_OFFSET, 4π^2 / mtheta)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    wv = complex.(arr .+ aii, air .- ari)
    # Force symmetry of response matrix if desired
    force_wv_symmetry && hermitianpart!(wv)

    # Create xzpts array
    xzpts = zeros(inputs.mtheta, 4)
    @views xzpts[:, 1] .= plasma_surf.x
    @views xzpts[:, 2] .= plasma_surf.z
    @views xzpts[:, 3] .= wall.x
    @views xzpts[:, 4] .= wall.z
    return wv, green_fourier, xzpts
end

function compute_vacuum_response_3D(inputs::VacuumInput, wall_settings::WallShapeSettings)

    # Initialization and allocations
    (; mtheta, mpert, n, kernelsign, force_wv_symmetry, nzeta, npert) = inputs
    num_gridpoints = nzeta * mtheta
    num_modes = npert * mpert

    plasma_surf = initialize_plasma_surface(inputs)
    wall = initialize_wall(inputs, plasma_surf, wall_settings)
    grad_green = zeros(num_gridpoints, num_gridpoints) # for walls, this is 2*mtheta x 2*mtheta
    green_temp = zeros(num_gridpoints, num_gridpoints)

    plasma_surf3D = PlasmaGeometry3D(plasma_surf, inputs)

    if false # dA debugging
        a = 0.1
        R0 = 10
        ϕ_grid = range(; start=0, length=nzeta, step=2π/nzeta)
        θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
        analytic_dA = [a * (R0 + a * cos(θ)) * (4π^2 / mtheta / nzeta) for θ in θ_grid, ϕ in ϕ_grid]
        println("Computed 3D plasma surface differential area elements dA: $(sum(plasma_surf3D.dA))")
        println("Sum of analytic dA: $(sum(analytic_dA))")  # DEBUG
        println("Analytic = $(4π^2 * a * R0)")  # DEBUG
        println("Difference in analytic and computed dA:")  # DEBUG
        display(analytic_dA - reshape(plasma_surf3D.dA, size(analytic_dA)))  # DEBUG
        println("Max difference in dA: $(maximum(abs.(analytic_dA - reshape(plasma_surf3D.dA, size(analytic_dA)))))")  # DEBUG
        println("Min difference in dA: $(minimum(abs.(analytic_dA - reshape(plasma_surf3D.dA, size(analytic_dA)))))")  # DEBUG
        error("Debugging dA")  # DEBUG
    end

    # 𝒢ₗ(θⱼ) from Chance eq. 106-108. first num_gridpoints rows are plasma as observer, second are wall
    # First num_modes columns are real (cosine), second num_modes are imaginary (sine)
    green_fourier = zeros(num_gridpoints, 2 * num_modes)
    PLASMA_ROW_OFFSET = 0
    WALL_ROW_OFFSET = num_gridpoints
    COS_COL_OFFSET = 0
    SIN_COL_OFFSET = num_modes

    !wall.nowall && error("No walls yet!") # DEBUG

    # Plasma–Plasma block
    # println("Calling BIEST with Nt=$nzeta, Np=$mtheta (total 3D points: $(num_gridpoints))...")
    # compute_green_matrices!(green_temp, grad_green, plasma_surf.x, plasma_surf.z, plasma_surf.ν, nzeta)
    # compute_green_matrices!(green_temp, grad_green, plasma_surf3D)
    # # display(green_temp)
    # # display(grad_green)
    # green_BIEST = copy(green_temp)
    # grad_green_BIEST = copy(grad_green)

    # G = single-layer kernel, K = double-layer kernel
    compute_3D_kernel_matrix!(grad_green, green_temp, plasma_surf3D, plasma_surf3D; INTERP_ORDER=6)
    # display(green_temp)
    # display(grad_green)

    # Compare BIEST and regular kernel matrices
    # println("\n=== Comparing BIEST vs Regular Kernel Matrices ===")
    # println("\nRelative difference single layer (regular - BIEST):")
    # display((green_temp .- green_BIEST) ./ green_BIEST)
    # println("Max relative difference in green_temp: $(maximum(abs.((green_temp - green_BIEST) ./ green_BIEST)))")

    # println("\nRelative difference double layer (regular - BIEST):")
    # display((grad_green .- grad_green_BIEST) ./ grad_green_BIEST)
    # println("Max relative difference in grad_green: $(maximum(abs.((grad_green - grad_green_BIEST) ./ grad_green_BIEST)))")

    # Sum Green's function matrices over toroidal direction to recover 2D poloidal slice
    # green_2D = zeros(ComplexF64, mtheta, mtheta)
    # gradgreen_2D = zeros(ComplexF64, mtheta, mtheta)
    # dζ = 2π / nzeta
    # # Perform Fourier integral over zeta to get 2D kernels G_2D = 1/2π ∫ G_3D e^{i n (ζ - ζ')} dζ'
    # for ipol_obs in 1:mtheta
    #     itor_obs = 1
    #     idx = (itor_obs - 1) * mtheta + ipol_obs
    #     for j in 1:mtheta
    #         green_2D[idx, j] = 1/2π * sum(green_temp[idx, (l-1)*mtheta + j] * exp(im * n * (itor_obs - l) * dζ) for l in 1:nzeta)
    #         gradgreen_2D[idx, j] = 1/2π * sum(grad_green[idx, (l-1)*mtheta + j] * exp(im * n * (itor_obs - l) * dζ) for l in 1:nzeta)
    #     end
    # end
    # Test Green's function by applying to unit source: green_test[i] = ∫ G B dθ' ≈ Σⱼ Gᵢⱼ Bⱼ Δθ'
    # B = ones(mtheta)
    # green_test = ((4π .* green_2D)) * B # account for 2D green's function being 1/r (not 1/4πr)
    # println("3D Green's integral with a unit source:")
    # display(real.(green_test))
    # green_test = ((4π .* gradgreen_2D) + I) * B # account for 2D green's function being 1/r (not 1/4πr)
    # println("3D Grad Green's integral with a unit source:")
    # display(real.(green_test))
    # println("Sum over zeta entries of green_3D (single-layer):")
    # display(real.(green_2D))
    # println("Sum over zeta entries of gradgreen_3D (double-layer):")
    # display(real.(gradgreen_2D[1:mtheta, 1:mtheta]))
    # display(real.(gradgreen_2D[mtheta+1:2*mtheta, 1:mtheta])) # confirmed that this is the same as the first mtheta rows
    # identity = Matrix{ComplexF64}(I, mtheta, mtheta)
    # gradgreen_2D[1:mtheta, 1:mtheta] .+= identity .* 0.5  # Add identity*0.5 to double-layer kernel for jump condition

    grad_green += I * 0.5  # Add 0.5I to double-layer kernel for jump condition

    # Fourier transform plasma-plasma block
    fourier_transform!(green_fourier, green_temp, plasma_surf3D.cos_mn_basis3D, PLASMA_ROW_OFFSET, COS_COL_OFFSET)
    fourier_transform!(green_fourier, green_temp, plasma_surf3D.sin_mn_basis3D, PLASMA_ROW_OFFSET, SIN_COL_OFFSET)

    !wall.nowall && error("No walls yet!")

    # Add cn0 to make grdgre nonsingular for n=0 modes
    (abs(n) <= 1e-5 && !wall.nowall && wall.is_closed_toroidal) && error("No walls yet!")

    # Only needed for mutual inductance with the wall calculations
    (kernelsign < 0) && error("No walls yet!")

    # Invert the vacuum response system of equations, eqs. 92-94ish of Chance 1997 (gelimb in Fortran)
    # If plasma only, lower blocks will be empty
    if wall.nowall
        @views green_fourier[1:num_gridpoints, :] .= grad_green[1:num_gridpoints, 1:num_gridpoints] \ green_fourier[1:num_gridpoints, :]
    else
        error("No walls yet!")
        green_fourier .= grad_green \ green_fourier
    end

    # Perform inverse Fourier transforms to get response matrix components (eq. 115-118 of Chance 2007)
    arr, aii, ari, air = ntuple(_ -> zeros(num_modes, num_modes), 4)
    fourier_inverse_transform!(arr, green_fourier, plasma_surf3D.cos_mn_basis3D, PLASMA_ROW_OFFSET, COS_COL_OFFSET, 4π^2 / num_gridpoints)
    fourier_inverse_transform!(aii, green_fourier, plasma_surf3D.sin_mn_basis3D, PLASMA_ROW_OFFSET, SIN_COL_OFFSET, 4π^2 / num_gridpoints)
    fourier_inverse_transform!(ari, green_fourier, plasma_surf3D.sin_mn_basis3D, PLASMA_ROW_OFFSET, COS_COL_OFFSET, 4π^2 / num_gridpoints)
    fourier_inverse_transform!(air, green_fourier, plasma_surf3D.cos_mn_basis3D, PLASMA_ROW_OFFSET, SIN_COL_OFFSET, 4π^2 / num_gridpoints)

    # Final form of vacuum response matrix (eq. 114 of Chance 2007)
    wv = complex.(arr .+ aii, air .- ari)
    # Force symmetry of response matrix if desired
    force_wv_symmetry && hermitianpart!(wv)

    # Create xzpts array
    xzpts = zeros(inputs.mtheta, 4)
    @views xzpts[:, 1] .= plasma_surf.x
    @views xzpts[:, 2] .= plasma_surf.z
    @views xzpts[:, 3] .= wall.x
    @views xzpts[:, 4] .= wall.z
    return wv, green_fourier, xzpts
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
