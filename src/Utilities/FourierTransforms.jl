"""
    FourierTransforms

Utility module for efficient Fourier transforms using pre-computed basis functions.

Provides a functor-based interface for Fourier transforms between theta-space and mode-space
representations. Supports both simple harmonic transforms (cos(m*θ), sin(m*θ)) and phase-shifted
transforms (cos(m*θ + n*qa*δ), sin(m*θ + n*qa*δ)) used in vacuum field calculations.

# Features

- Pre-computes trigonometric basis functions for efficiency
- Direct complex number support (no manual real/imaginary splitting)
- Type-stable functor pattern for high performance
- Supports different grids (mthvac, mtheta) with different mode ranges

# Example

```julia
using JPEC.Utilities.FourierTransforms

# For PerturbedEquilibrium (no phase shift)
ft = FourierTransform(mtheta, mpert, mlow)

# For Vacuum (with n*qa*delta phase)
ft_vac = FourierTransform(mthvac, mpert, mlow; n=1, qa=2.5, delta=delta_array)

# Forward transform: theta-space → Fourier modes
theta_data = randn(mtheta, 10)  # 10 different theta-space functions
modes = ft(theta_data)          # Complex modes [mpert, 10]

# Inverse transform: Fourier modes → theta-space
reconstructed = inverse(ft, modes)
```
"""
module FourierTransforms

using LinearAlgebra

export FourierTransform, inverse
export compute_fourier_coefficients
export transform!, inverse_transform!
export fourier_transform!, fourier_inverse_transform!

"""
    compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow; n=nothing, ν=zeros(mtheta))

Compute Fourier basis function coefficients for transforms between physical-space and mode-space.
Supports both 2D and 3D geometries. In 2D, we only use one toroidal mode at a time, so we can
just use the n argument. In 3D, we need to compute the basis for all modes and grid points.

# Arguments

- `mtheta::Int`: Number of poloidal grid points (theta resolution)
- `mpert::Int`: Number of Fourier modes (spectral resolution)
- `mlow::Int`: Lowest mode number (mode numbering starts here)
- `nzeta::Int`: Number of toroidal grid points
- `npert::Int`: Number of toroidal modes
- `nlow::Int`: Lowest toroidal mode number

# Keyword Arguments

- `n::Union{Nothing, Int}=nothing`: Toroidal mode number for 2D (default: nothing)
- `ν::Vector{Float64}=zeros(mtheta)`: Toroidal angle offset array (default: no offset, n*ν = 0)

# Returns

- 2D
  - `cos_mn_basis::Matrix{Float64}`: Cosine coefficients `cos(m*θ - n*ν)` [mtheta, mpert]
  - `sin_mn_basis::Matrix{Float64}`: Sine coefficients `sin(m*θ - n*ν)` [mtheta, mpert]
- 3D
  - `cos_mn_basis::Matrix{Float64}`: Cosine coefficients `cos(m*θ - n*ν - n*ϕ)` [mtheta * nzeta, mpert * npert]
  - `sin_mn_basis::Matrix{Float64}`: Sine coefficients `sin(m*θ - n*ν - n*ϕ)` [mtheta * nzeta, mpert * npert]

# Notes

The theta and phi grids are uniform: `θᵢ = 2π*i/mtheta` for `i = 0:mtheta-1` and `ϕⱼ = 2π*j/nzeta` for `j = 0:nzeta-1`

When `n=0, ν=0` (default), this reduces to simple harmonic basis:
- `cos_mn_basis[i,l] = cos(m*θᵢ)`
- `sin_mn_basis[i,l] = sin(m*θᵢ)`
"""
function compute_fourier_coefficients(
    mtheta::Int,
    mpert::Int,
    mlow::Int,
    nzeta::Int,
    npert::Int,
    nlow::Int;
    n::Union{Nothing, Int}=nothing,
    ν::Vector{Float64}=zeros(Float64, mtheta)
)
    # Validate inputs
    @assert length(ν) == mtheta "ν must have length mtheta"
    @assert !(nzeta > 1 && n !== nothing) "If nzeta > 1, don't set n since we use the full nlow - nhigh range for the toroidal modes"
    @assert !(n === nothing && nzeta == 1) "If nzeta == 1 (2D), set n to the toroidal mode number"

    # Uniform theta grid: [0, 2π)
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)

    if !isnothing(n)
        # In 2D, we only use one toroidal mode at a time
        # Compute sin(mθ - nν) and cos(mθ - nν)
        sin_mn_basis = sin.((mlow .+ (0:(mpert-1))') .* θ_grid .- n .* ν)
        cos_mn_basis = cos.((mlow .+ (0:(mpert-1))') .* θ_grid .- n .* ν)
    else # nzeta > 1
        # In 3D, we need to compute the basis for all modes and grid points
        # Compute sin(mθ - nν - nϕ) and cos(mθ - nν - nϕ)
        ϕ_grid = range(; start=0, length=nzeta, step=2π/nzeta)
        sin_mn_basis = zeros(mtheta * nzeta, mpert * npert)
        cos_mn_basis = zeros(mtheta * nzeta, mpert * npert)
        for idx_n in 1:npert, idx_m in 1:mpert
            n = nlow + idx_n - 1
            m = mlow + idx_m - 1
            for (j, ϕ) in enumerate(ϕ_grid), (i, θ) in enumerate(θ_grid)
                cos_mn_basis[i+(j-1)*mtheta, idx_m+(idx_n-1)*mpert] = cos(m * θ - n * (ν[i] + ϕ))
                sin_mn_basis[i+(j-1)*mtheta, idx_m+(idx_n-1)*mpert] = sin(m * θ - n * (ν[i] + ϕ))
            end
        end
    end

    return cos_mn_basis, sin_mn_basis
end

"""
    FourierTransform

Callable struct for efficient Fourier transforms with pre-computed basis functions.

# Fields

- `mtheta::Int`: Number of theta grid points
- `mpert::Int`: Number of Fourier modes
- `mlow::Int`: Lowest mode number
- `cslth::Matrix{Float64}`: Cosine basis functions [mtheta, mpert]
- `snlth::Matrix{Float64}`: Sine basis functions [mtheta, mpert]

# Usage

Create once with appropriate grid parameters, then use repeatedly:

```julia
# Create transform (coefficients computed once)
ft = FourierTransform(mtheta, mpert, mlow)

# Forward transform (callable): theta → modes
modes = ft(theta_data)

# Inverse transform: modes → theta
theta_reconstructed = inverse(ft, modes)
```

See also: [`compute_fourier_coefficients`](@ref), [`inverse`](@ref)
"""
struct FourierTransform
    mtheta::Int
    mpert::Int
    mlow::Int
    cslth::Matrix{Float64}
    snlth::Matrix{Float64}
end

"""
    FourierTransform(mtheta, mpert, mlow; n=0, qa=0.0, delta=zeros(mtheta))

Construct a FourierTransform object with pre-computed basis functions.

# Arguments

- `mtheta::Int`: Number of poloidal grid points
- `mpert::Int`: Number of Fourier modes
- `mlow::Int`: Lowest mode number

# Keyword Arguments

- `n::Int=0`: Toroidal mode number
- `qa::Float64=0.0`: Safety factor at boundary
- `delta::Vector{Float64}=zeros(mtheta)`: Toroidal phase shift array

# Returns

`FourierTransform` object ready for forward and inverse transforms.

# Examples

```julia
# Simple case: no phase shift (for PerturbedEquilibrium)
ft = FourierTransform(480, 40, -20)

# With toroidal phase (for Vacuum calculations)
ft_vac = FourierTransform(480, 40, -20; n=1, ν=ν_array)
```
"""
function FourierTransform(
    mtheta::Int,
    mpert::Int,
    mlow::Int;
    n::Int=0,
    ν::Vector{Float64}=zeros(Float64, mtheta)
)
    cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, 1, 1, 1; n, ν)
    return FourierTransform(mtheta, mpert, mlow, cos_mn_basis, sin_mn_basis)
end

"""
    (ft::FourierTransform)(data::AbstractVecOrMat{<:Real})

Forward Fourier transform: theta-space → mode-space (callable functor).

Transforms real-valued data at theta grid points into complex Fourier mode coefficients.

# Arguments

- `data::AbstractVecOrMat{Float64}`: Data at theta points
  - If `Vector{Float64}` with length `mtheta`: single function to transform
  - If `Matrix{Float64}` with size `(mtheta, n)`: n functions to transform simultaneously

# Returns

- Complex Fourier coefficients
  - If input is `Vector`: returns `Vector{ComplexF64}` of length `mpert`
  - If input is `Matrix`: returns `Matrix{ComplexF64}` of size `(mpert, n)`

# Formula

For each mode `l` (corresponding to mode number `m = mlow + l - 1`):

```
mode[l] = Σᵢ data[i] * (cos(m*θᵢ + phase) + im*sin(m*θᵢ + phase))
```

This is equivalent to computing:
```
real_part[l] = Σᵢ data[i] * cos(m*θᵢ + phase)
imag_part[l] = Σᵢ data[i] * sin(m*θᵢ + phase)
mode[l] = complex(real_part[l], imag_part[l])
```

# Examples

```julia
ft = FourierTransform(480, 40, -20)

# Transform a single function
f_theta = sin.(theta_grid)
f_modes = ft(f_theta)  # Vector{ComplexF64} of length 40

# Transform multiple functions at once
data = randn(480, 10)  # 10 different functions
modes = ft(data)       # Matrix{ComplexF64} of size (40, 10)
```
"""
function (ft::FourierTransform)(data::AbstractVecOrMat{<:Real})
    # Forward transform using matrix multiplication
    # real_part = data^T * cslth  (or cslth^T * data for vectors)
    # imag_part = data^T * snlth  (or snlth^T * data for vectors)
    # Return as complex

    if data isa AbstractVector
        # For vector input: [mtheta] → [mpert]
        @assert length(data) == ft.mtheta "Input vector must have length mtheta=$(ft.mtheta)"
        real_part = ft.cslth' * data
        imag_part = ft.snlth' * data
        return complex.(real_part, imag_part)
    else
        # For matrix input: [mtheta, n] → [mpert, n]
        @assert size(data, 1) == ft.mtheta "Input matrix first dimension must be mtheta=$(ft.mtheta)"
        real_part = ft.cslth' * data
        imag_part = ft.snlth' * data
        return complex.(real_part, imag_part)
    end
end

"""
    (ft::FourierTransform)(data::AbstractVecOrMat{<:Complex})

Forward Fourier transform for complex-valued theta-space data.

Transforms complex data at theta grid points into complex Fourier mode coefficients.

# Arguments

- `data::AbstractVecOrMat{ComplexF64}`: Complex data at theta points

# Returns

- Complex Fourier coefficients with same shape as real input case

# Formula

For complex input `data = data_real + im*data_imag`, computes:

```
Re{mode[l]} = Σᵢ (data_real[i]*cos - data_imag[i]*sin)
Im{mode[l]} = Σᵢ (data_real[i]*sin + data_imag[i]*cos)
```

where cos and sin are the basis functions at theta point i for mode l.
"""
function (ft::FourierTransform)(data::AbstractVecOrMat{<:Complex})
    # For complex input, need to handle real and imaginary parts carefully
    # Forward transform: exp(-i*m*θ) = cos(m*θ) - i*sin(m*θ)

    if data isa AbstractVector
        @assert length(data) == ft.mtheta "Input vector must have length mtheta=$(ft.mtheta)"
        real_part = ft.cslth' * real.(data) .- ft.snlth' * imag.(data)
        imag_part = ft.cslth' * imag.(data) .+ ft.snlth' * real.(data)
        return complex.(real_part, imag_part)
    else
        @assert size(data, 1) == ft.mtheta "Input matrix first dimension must be mtheta=$(ft.mtheta)"
        real_part = ft.cslth' * real.(data) .- ft.snlth' * imag.(data)
        imag_part = ft.cslth' * imag.(data) .+ ft.snlth' * real.(data)
        return complex.(real_part, imag_part)
    end
end

"""
    inverse(ft::FourierTransform, modes::AbstractVecOrMat{<:Complex})

Inverse Fourier transform: mode-space → theta-space.

Reconstructs theta-space data from complex Fourier mode coefficients.

# Arguments

- `modes::AbstractVecOrMat{ComplexF64}`: Fourier mode coefficients
  - If `Vector{ComplexF64}` with length `mpert`: single mode expansion to reconstruct
  - If `Matrix{ComplexF64}` with size `(mpert, n)`: n mode expansions to reconstruct

# Returns

- Reconstructed data at theta points
  - If input is `Vector`: returns `Vector{ComplexF64}` of length `mtheta`
  - If input is `Matrix`: returns `Matrix{ComplexF64}` of size `(mtheta, n)`

# Formula

For each theta point `i`:

```
data[i] = (2π/mtheta) * Σₗ modes[l] * (cos(m*θᵢ + phase) + im*sin(m*θᵢ + phase))
```

This is equivalent to:
```
data[i] = (2π/mtheta) * Σₗ [Re{modes[l]}*cos - Im{modes[l]}*sin +
                             im*(Re{modes[l]}*sin + Im{modes[l]}*cos)]
```

# Notes

The normalization factor `2π/mtheta` ensures proper Fourier series reconstruction.

For real-valued theta data, the result will have negligible imaginary parts (machine precision).
Use `real.(inverse(ft, modes))` to extract the real part if needed.

# Examples

```julia
ft = FourierTransform(480, 40, -20)

# Reconstruct from modes
modes = randn(ComplexF64, 40)
theta_data = inverse(ft, modes)  # Vector{ComplexF64} of length 480

# For real reconstruction
theta_real = real.(inverse(ft, modes))
```
"""
function inverse(ft::FourierTransform, modes::AbstractVecOrMat{<:Complex})
    # Inverse transform with normalization
    # For mode expansion: f(θ) = Σₗ cₗ * exp(i*m*θ) = Σₗ [Re{cₗ}*cos - Im{cₗ}*sin + i*(Re{cₗ}*sin + Im{cₗ}*cos)]

    dth = 2π / ft.mtheta

    if modes isa AbstractVector
        @assert length(modes) == ft.mpert "Input vector must have length mpert=$(ft.mpert)"
        # Real and imaginary parts of reconstructed function
        real_part = ft.cslth * real.(modes) .- ft.snlth * imag.(modes)
        imag_part = ft.cslth * imag.(modes) .+ ft.snlth * real.(modes)
        return complex.(real_part, imag_part) .* dth
    else
        @assert size(modes, 1) == ft.mpert "Input matrix first dimension must be mpert=$(ft.mpert)"
        real_part = ft.cslth * real.(modes) .- ft.snlth * imag.(modes)
        imag_part = ft.cslth * imag.(modes) .+ ft.snlth * real.(modes)
        return complex.(real_part, imag_part) .* dth
    end
end

"""
    inverse(ft::FourierTransform, modes::AbstractVecOrMat{<:Real})

Inverse Fourier transform for real mode coefficients (pure cosine expansion).

Special case where all modes are real-valued, corresponding to a cosine-only Fourier series.

# Arguments

- `modes::AbstractVecOrMat{Float64}`: Real Fourier coefficients

# Returns

- Real-valued data at theta points with normalization factor `2π/mtheta`

# Formula

```
data[i] = (2π/mtheta) * Σₗ modes[l] * cos(m*θᵢ + phase)
```

# Notes

This is a special case and less commonly used. Most applications use complex modes.
"""
function inverse(ft::FourierTransform, modes::AbstractVecOrMat{<:Real})
    dth = 2π / ft.mtheta

    if modes isa AbstractVector
        @assert length(modes) == ft.mpert "Input vector must have length mpert=$(ft.mpert)"
        return (ft.cslth * modes) .* dth
    else
        @assert size(modes, 1) == ft.mpert "Input matrix first dimension must be mpert=$(ft.mpert)"
        return (ft.cslth * modes) .* dth
    end
end

# ==============================================================================
# In-place transform functions for performance-critical applications
# ==============================================================================

"""
    transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, data::AbstractVecOrMat{<:Real})

In-place forward Fourier transform for real-valued theta-space data.

More efficient than the allocating version `ft(data)` when called repeatedly,
as it reuses the pre-allocated output array.

# Arguments

- `output::AbstractVecOrMat{ComplexF64}`: Pre-allocated output array for Fourier modes
  - For vector input: must have length `mpert`
  - For matrix input: must have size `(mpert, n)` where `n = size(data, 2)`
- `ft::FourierTransform`: Fourier transform object with pre-computed coefficients
- `data::AbstractVecOrMat{Float64}`: Real-valued data at theta points
  - For vector: length must be `mtheta`
  - For matrix: first dimension must be `mtheta`

# Returns

- `output`: The modified output array (for chaining)

# Performance

This function uses in-place matrix multiplication (`mul!`) to avoid allocations,
making it suitable for tight loops and performance-critical code.

# Example

```julia
ft = FourierTransform(480, 40, -20)

# Pre-allocate output buffer
modes = zeros(ComplexF64, 40)

# Reuse buffer in loop
for i in 1:1000
    data = get_theta_data(i)
    transform!(modes, ft, data)  # No allocations
    process_modes(modes)
end
```
"""
function transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, data::AbstractVecOrMat{<:Real})
    if data isa AbstractVector
        @assert length(data) == ft.mtheta "Input vector must have length mtheta=$(ft.mtheta)"
        @assert length(output) == ft.mpert "Output vector must have length mpert=$(ft.mpert)"

        # Extract real and imaginary views
        real_part = reinterpret(Float64, output)
        real_view = @view real_part[1:2:end]    # Real components
        imag_view = @view real_part[2:2:end]    # Imaginary components

        # In-place computation: real = cslth' * data, imag = snlth' * data
        mul!(real_view, ft.cslth', data)
        mul!(imag_view, ft.snlth', data)
    else
        @assert size(data, 1) == ft.mtheta "Input matrix first dimension must be mtheta=$(ft.mtheta)"
        @assert size(output, 1) == ft.mpert "Output matrix first dimension must be mpert=$(ft.mpert)"
        @assert size(output, 2) == size(data, 2) "Output and input must have same number of columns"

        # For matrices, we need temporary storage for real/imag parts
        n_cols = size(data, 2)
        real_part = similar(output, Float64, ft.mpert, n_cols)
        imag_part = similar(output, Float64, ft.mpert, n_cols)

        # In-place computation
        mul!(real_part, ft.cslth', data)
        mul!(imag_part, ft.snlth', data)

        # Combine into complex output
        output .= complex.(real_part, imag_part)
    end

    return output
end

"""
    transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, data::AbstractVecOrMat{<:Complex})

In-place forward Fourier transform for complex-valued theta-space data.

# Arguments

- `output::AbstractVecOrMat{ComplexF64}`: Pre-allocated output array for Fourier modes
- `ft::FourierTransform`: Fourier transform object
- `data::AbstractVecOrMat{ComplexF64}`: Complex-valued data at theta points

# Returns

- `output`: The modified output array

# Formula

For complex input `data = data_real + im*data_imag`:
```
Re{mode[l]} = Σᵢ (data_real[i]*cos - data_imag[i]*sin)
Im{mode[l]} = Σᵢ (data_real[i]*sin + data_imag[i]*cos)
```
"""
function transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, data::AbstractVecOrMat{<:Complex})
    if data isa AbstractVector
        @assert length(data) == ft.mtheta "Input vector must have length mtheta=$(ft.mtheta)"
        @assert length(output) == ft.mpert "Output vector must have length mpert=$(ft.mpert)"

        # Temporary storage for intermediate results
        real_part = similar(output, Float64)
        imag_part = similar(output, Float64)
        temp1 = similar(output, Float64)
        temp2 = similar(output, Float64)

        # Re{mode} = cslth' * real(data) - snlth' * imag(data)
        mul!(temp1, ft.cslth', real.(data))
        mul!(temp2, ft.snlth', imag.(data))
        real_part .= temp1 .- temp2

        # Im{mode} = cslth' * imag(data) + snlth' * real(data)
        mul!(temp1, ft.cslth', imag.(data))
        mul!(temp2, ft.snlth', real.(data))
        imag_part .= temp1 .+ temp2

        output .= complex.(real_part, imag_part)
    else
        @assert size(data, 1) == ft.mtheta "Input matrix first dimension must be mtheta=$(ft.mtheta)"
        @assert size(output, 1) == ft.mpert "Output matrix first dimension must be mpert=$(ft.mpert)"
        @assert size(output, 2) == size(data, 2) "Output and input must have same number of columns"

        n_cols = size(data, 2)
        real_part = similar(output, Float64, ft.mpert, n_cols)
        imag_part = similar(output, Float64, ft.mpert, n_cols)
        temp1 = similar(real_part)
        temp2 = similar(real_part)

        # Re{mode} = cslth' * real(data) - snlth' * imag(data)
        mul!(temp1, ft.cslth', real.(data))
        mul!(temp2, ft.snlth', imag.(data))
        real_part .= temp1 .- temp2

        # Im{mode} = cslth' * imag(data) + snlth' * real(data)
        mul!(temp1, ft.cslth', imag.(data))
        mul!(temp2, ft.snlth', real.(data))
        imag_part .= temp1 .+ temp2

        output .= complex.(real_part, imag_part)
    end

    return output
end

"""
    inverse_transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, modes::AbstractVecOrMat{<:Complex})

In-place inverse Fourier transform: mode-space → theta-space.

More efficient than the allocating version `inverse(ft, modes)` when called repeatedly.

# Arguments

- `output::AbstractVecOrMat{ComplexF64}`: Pre-allocated output array for theta-space data
  - For vector: must have length `mtheta`
  - For matrix: must have size `(mtheta, n)` where `n = size(modes, 2)`
- `ft::FourierTransform`: Fourier transform object
- `modes::AbstractVecOrMat{ComplexF64}`: Complex Fourier mode coefficients
  - For vector: length must be `mpert`
  - For matrix: first dimension must be `mpert`

# Returns

- `output`: The modified output array (for chaining)

# Formula

```
data[i] = (2π/mtheta) * Σₗ [Re{modes[l]}*cos - Im{modes[l]}*sin +
                             im*(Re{modes[l]}*sin + Im{modes[l]}*cos)]
```

# Example

```julia
ft = FourierTransform(480, 40, -20)

# Pre-allocate output buffer
theta_data = zeros(ComplexF64, 480)

# Reuse buffer in loop
for i in 1:1000
    modes = get_modes(i)
    inverse_transform!(theta_data, ft, modes)  # No allocations
    process_theta_data(theta_data)
end
```
"""
function inverse_transform!(output::AbstractVecOrMat{ComplexF64}, ft::FourierTransform, modes::AbstractVecOrMat{<:Complex})
    dth = 2π / ft.mtheta

    if modes isa AbstractVector
        @assert length(modes) == ft.mpert "Input vector must have length mpert=$(ft.mpert)"
        @assert length(output) == ft.mtheta "Output vector must have length mtheta=$(ft.mtheta)"

        # Temporary storage
        real_part = similar(output, Float64)
        imag_part = similar(output, Float64)
        temp1 = similar(output, Float64)
        temp2 = similar(output, Float64)

        # Re{data} = cslth * real(modes) - snlth * imag(modes)
        mul!(temp1, ft.cslth, real.(modes))
        mul!(temp2, ft.snlth, imag.(modes))
        real_part .= (temp1 .- temp2) .* dth

        # Im{data} = cslth * imag(modes) + snlth * real(modes)
        mul!(temp1, ft.cslth, imag.(modes))
        mul!(temp2, ft.snlth, real.(modes))
        imag_part .= (temp1 .+ temp2) .* dth

        output .= complex.(real_part, imag_part)
    else
        @assert size(modes, 1) == ft.mpert "Input matrix first dimension must be mpert=$(ft.mpert)"
        @assert size(output, 1) == ft.mtheta "Output matrix first dimension must be mtheta=$(ft.mtheta)"
        @assert size(output, 2) == size(modes, 2) "Output and input must have same number of columns"

        n_cols = size(modes, 2)
        real_part = similar(output, Float64, ft.mtheta, n_cols)
        imag_part = similar(output, Float64, ft.mtheta, n_cols)
        temp1 = similar(real_part)
        temp2 = similar(real_part)

        # Re{data} = cslth * real(modes) - snlth * imag(modes)
        mul!(temp1, ft.cslth, real.(modes))
        mul!(temp2, ft.snlth, imag.(modes))
        real_part .= (temp1 .- temp2) .* dth

        # Im{data} = cslth * imag(modes) + snlth * real(modes)
        mul!(temp1, ft.cslth, imag.(modes))
        mul!(temp2, ft.snlth, real.(modes))
        imag_part .= (temp1 .+ temp2) .* dth

        output .= complex.(real_part, imag_part)
    end

    return output
end

# ==============================================================================
# Low-level matrix transforms with offsets (for Vacuum module compatibility)
# ==============================================================================

"""
    fourier_transform!(gil::Matrix{Float64}, gij::Matrix{Float64}, cs::Matrix{Float64}; row_offset::Int=0, col_offset::Int=0)

Low-level Fourier transform with offset support for Vacuum module.

Performs a truncated Fourier transform of `gij` onto `gil` using pre-computed
Fourier coefficients `cs`, with support for block offsets in the output matrix.

This is used by the Vacuum module for transforming Green's function matrices that
have specific block structure (e.g., plasma and wall contributions packed together).

# Arguments

- `gil::Matrix{Float64}`: Output matrix, updated in-place at offset block
- `gij::Matrix{Float64}`: Input matrix (mtheta × mtheta) containing theta-space data
- `cs::Matrix{Float64}`: Fourier coefficient matrix (mtheta × mpert)
- `row_offset::Int`: Row offset in `gil` matrix
- `col_offset::Int`: Column offset in `gil` matrix

# Operation

Computes: `gil[row_offset+i, col_offset+l] = Σⱼ gij[i, j] * cs[j, l]` for i ∈ 1:size(cs, 1), l ∈ 1:size(cs, 2)

# Notes

- This function uses 0-based offset convention (add 1 for Julia indexing)
- Used for packing real and imaginary parts in separate blocks of `gil`

# Example

```julia
ft = FourierTransform(mtheta, mpert, mlow; n=n, qa=qa, delta=delta)
gil = zeros(2*mtheta, 2*mpert)  # Packed real/imag structure
gij = ... # Some theta-space data

# Transform using cosine coefficients, real part block (offset 0, 0)
fourier_transform!(gil, gij, ft.cslth; row_offset=0, col_offset=0)

# Transform using sine coefficients, imaginary part block (offset 0, mpert)
fourier_transform!(gil, gij, ft.snlth; row_offset=0, col_offset=mpert)
```
"""
function fourier_transform!(gil::AbstractMatrix{Float64}, gij::AbstractMatrix{Float64}, cs::Matrix{Float64}; row_offset::Int=0, col_offset::Int=0)
    mul!(view(gil, row_offset+1:row_offset+size(cs, 1), col_offset+1:col_offset+size(cs, 2)), gij, cs)
end

"""
    fourier_inverse_transform!(gll::Matrix{Float64}, gil::Matrix{Float64}, cs::Matrix{Float64}, m00::Int, l00::Int)

Low-level inverse Fourier transform with offset support for Vacuum module.

Performs the inverse Fourier transform of `gil` onto `gll` using pre-computed
Fourier coefficients `cs`, with support for block offsets in the input matrix.

This is used by the Vacuum module for inverse transforming Green's function matrices
back to mode space from theta space.

# Arguments

- `gll::Matrix{Float64}`: Output matrix (mpert × mpert), updated in-place
- `gil::Matrix{Float64}`: Input matrix containing Fourier-transformed data
- `cs::Matrix{Float64}`: Fourier coefficient matrix (mtheta × mpert), either `cslth` or `snlth`
- `m00::Int`: Row offset in `gil` matrix (0-based indexing convention)
- `l00::Int`: Column offset in `gil` matrix (0-based indexing convention)

# Operation

Computes: `gll[l2, l1] = dθdζ * Σᵢ cs[i, l2] * gil[m00+i, l00+l1]`

# Notes

- The normalization factor `4π^2 / size(cs, 1)` is 2π * mtheta * 2π * nzeta (nzeta = 1 for 2D)
- This function uses 0-based offset convention (add 1 for Julia indexing)
- The `cs` matrix should be either `cslth` or `snlth` from a `FourierTransform` object
# Example

```julia
ft = FourierTransform(mtheta, mpert, mlow; n=n, qa=qa, delta=delta)
gil = ... # Transformed Green's function data
arr = zeros(mpert, mpert)

# Inverse transform from real part block using cosine coefficients
fourier_inverse_transform!(arr, gil, ft.cslth)
```
"""
function fourier_inverse_transform!(gll::AbstractMatrix{Float64}, gil::AbstractMatrix{Float64}, cs::Matrix{Float64}; row_offset::Int=0, col_offset::Int=0)
    mul!(gll, cs', view(gil, row_offset+1:row_offset+size(cs, 1), col_offset+1:col_offset+size(cs, 2)), 4π^2 / size(cs, 1), 0.0)
end

end # module FourierTransforms
