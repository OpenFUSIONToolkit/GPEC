"""
    FourierTransforms

Pre-computed complex Fourier basis and functor interface for θ ↔ mode transforms.

`basis[ℓ, i] = exp(-i(m_ℓ θ_i - n ν_i))` with shape `(mpert, mtheta)`. Forward: `basis * data / mtheta`;
inverse: `adjoint(basis) * modes` (Fortran `iscdftf`/`iscdftb`; see `docs/src/conventions.md`).
"""
module FourierTransforms

using LinearAlgebra

export FourierTransform, inverse, inverse_transform!, transform!
export compute_fourier_coefficients

"""
    compute_fourier_coefficients(mtheta, m_modes, n, ν)

Build complex basis ``\\exp(-i(m\\theta - n\\nu))`` on the uniform poloidal grid.

## Arguments

  - `mtheta`: number of poloidal grid points
  - `m_modes`: poloidal mode numbers (one row per mode)
  - `n`: toroidal mode number
  - `ν`: toroidal angle offset on the poloidal grid, length `mtheta`

## Returns

  - Basis matrix, size `(length(m_modes), mtheta)`
"""
function compute_fourier_coefficients(mtheta::Int, m_modes::AbstractVector{<:Integer}, n::Integer, ν::Vector{Float64})

    @assert length(ν) == mtheta "ν must have length mtheta"

    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    arg = m_modes' .* θ_grid .- n .* ν
    return transpose(exp.(-im .* arg))
end

"""
    compute_fourier_coefficients(mtheta, m_modes, nzeta, n_modes; nfp=1)

Build 3D basis for every `(m, n)` in `m_modes × n_modes` (rows m-fast, n-slow).

## Arguments

  - `mtheta`: poloidal grid points per toroidal plane
  - `m_modes`: poloidal mode numbers
  - `nzeta`: full-torus toroidal grid points (must be divisible by `nfp`)
  - `n_modes`: toroidal mode numbers

## Keyword Arguments

  - `nfp`: number of field periods; when `> 1`, emits one period only (`nzeta ÷ nfp` planes)

## Returns

  - Basis matrix, size `(length(m_modes) * length(n_modes), mtheta * (nzeta ÷ nfp))`
"""
function compute_fourier_coefficients(
    mtheta::Int,
    m_modes::AbstractVector{<:Integer},
    nzeta::Int,
    n_modes::AbstractVector{<:Integer};
    nfp::Int=1
)
    @assert nzeta % nfp == 0 "nzeta ($nzeta) must be divisible by nfp ($nfp)"

    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    ζ_grid = range(; start=0, length=nzeta, step=2π/nzeta)
    nzeta_out = nzeta ÷ nfp

    mpert = length(m_modes)
    basis = zeros(ComplexF64, mpert * length(n_modes), mtheta * nzeta_out)
    for (idx_n, n) in enumerate(n_modes)
        n_row_offset = (idx_n - 1) * mpert
        for (idx_m, m) in enumerate(m_modes)
            row = idx_m + n_row_offset
            for j in 1:nzeta_out, (i, θ) in enumerate(θ_grid)
                col = i + (j - 1) * mtheta
                basis[row, col] = cis(-(m * θ - n * ζ_grid[j]))
            end
        end
    end

    return basis
end

"""
    FourierTransform

Struct with precomputed complex Fourier basis for repeated θ ↔ mode transforms.

# Fields

  - `mtheta`: poloidal grid size
  - `mpert`: number of poloidal modes
  - `mlow`: lowest poloidal mode number
  - `basis`: ``\\exp(-i(m\\theta - n\\nu))``, size `(mpert, mtheta)`
"""
struct FourierTransform
    mtheta::Int
    mpert::Int
    mlow::Int
    basis::Matrix{ComplexF64}
end

"""
    FourierTransform(mtheta, mpert, mlow; n=0, ν=zeros(mtheta))

Construct a transform with precomputed basis for contiguous modes `mlow:(mlow+mpert-1)`.

## Arguments

  - `mtheta`: number of poloidal grid points
  - `mpert`: number of poloidal modes
  - `mlow`: lowest poloidal mode number

## Keyword Arguments

  - `n`: toroidal mode number (default 0)
  - `ν`: toroidal angle offset on the poloidal grid, length `mtheta`

## Returns

  - `FourierTransform` ready for forward and inverse transforms
"""
function FourierTransform(
    mtheta::Int,
    mpert::Int,
    mlow::Int;
    n::Int=0,
    ν::Vector{Float64}=zeros(Float64, mtheta)
)
    basis = compute_fourier_coefficients(mtheta, mlow:(mlow+mpert-1), n, ν)
    return FourierTransform(mtheta, mpert, mlow, basis)
end

"""
    (ft::FourierTransform)(data)

Forward transform from θ-space to mode space.

## Arguments

  - `data`: real or complex samples on the poloidal grid — `Vector{mtheta}` or `Matrix{mtheta, :}`

## Returns

  - Mode coefficients — `Vector{mpert}` or `Matrix{mpert, :}`
"""
function (ft::FourierTransform)(data::AbstractVecOrMat{<:Number})
    if data isa AbstractVector
        @assert length(data) == ft.mtheta "Input vector must have length mtheta=$(ft.mtheta)"
    else
        @assert size(data, 1) == ft.mtheta "Input matrix first dimension must be mtheta=$(ft.mtheta)"
    end

    return (ft.basis * data) ./ ft.mtheta
end

"""
    inverse(ft, modes)

Inverse transform from mode space to θ-space.

## Arguments

  - `ft`: precomputed transform
  - `modes`: complex mode coefficients — `Vector{mpert}` or `Matrix{mpert, :}`

## Returns

  - θ-space data — `Vector{mtheta}` or `Matrix{mtheta, :}`
"""
function inverse(ft::FourierTransform, modes::AbstractVecOrMat{<:Complex})
    if modes isa AbstractVector
        @assert length(modes) == ft.mpert "Input vector must have length mpert=$(ft.mpert)"
    else
        @assert size(modes, 1) == ft.mpert "Input matrix first dimension must be mpert=$(ft.mpert)"
    end

    return adjoint(ft.basis) * modes
end

"""
    inverse_transform!(output, ft, modes)

In-place inverse transform (mode → θ-space) writing into `output`: `output .= adjoint(basis) * modes`.

Allocation-free wrapper over [`inverse`](@ref) for reuse in hot loops. `output` must have first
dimension `mtheta` and match the column count of `modes` (whose first dimension is `mpert`).
"""
inverse_transform!(output::AbstractVecOrMat{<:Complex}, ft::FourierTransform, modes::AbstractVecOrMat{<:Complex}) = mul!(output, adjoint(ft.basis), modes)

"""
    transform!(output, ft, data)

In-place forward transform (θ-space → mode) writing into `output`: `output .= (basis * data) / mtheta`.

Allocation-free wrapper over the [`FourierTransform`](@ref) functor for reuse in hot loops. `output`
must have first dimension `mpert` and match the column count of `data` (whose first dimension is `mtheta`).
"""
function transform!(output::AbstractVecOrMat{<:Complex}, ft::FourierTransform, data::AbstractVecOrMat{<:Number})
    mul!(output, ft.basis, data)
    output ./= ft.mtheta
    return output
end

end # module FourierTransforms
