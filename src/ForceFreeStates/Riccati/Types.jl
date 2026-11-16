# Integration-chunk and chunk-propagator types for the fundamental-matrix (Riccati/STRIDE) driver.

"""
    IntegrationChunk

A struct representing a region of integration in the Euler-Lagrange solver.

## Fields

  - `psi_start::Float64` - Starting ψ coordinate for this integration region
  - `psi_end::Float64` - Ending ψ coordinate for this integration region
  - `needs_crossing::Bool` - Whether a rational surface crossing is needed after this chunk
  - `ising::Int` - Index of the singular surface associated with this chunk (0 if none)
  - `direction::Int` - +1 forward, -1 backward. A backward chunk still has `psi_start < psi_end` but
    its propagator maps the state at `psi_end` to `psi_start`; used for well-conditioned crossing chunks.
"""
@kwdef struct IntegrationChunk
    psi_start::Float64
    psi_end::Float64
    needs_crossing::Bool
    ising::Int = 0
    direction::Int = 1   # +1 forward, -1 backward
end

"""
    ChunkPropagator

Fundamental matrix of one integration chunk: column `j` is the solution started from column `j` of the chunk's
start basis (`propagator_start`).

## Fields

  - `u::Array{ComplexF64,3}` - The matrix at the end of the integration, `(N, 2N, 2)`: U₁ in slice 1, U₂ in slice 2.
  - `psi::Vector{Float64}` - ψ of every solver step, in integration order.
  - `steps::Vector{Array{ComplexF64,3}}` - The matrix at each `psi`.
"""
struct ChunkPropagator
    u::Array{ComplexF64,3}
    psi::Vector{Float64}
    steps::Vector{Array{ComplexF64,3}}
end
