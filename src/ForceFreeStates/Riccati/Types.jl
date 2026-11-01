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

Fundamental matrix for one integration chunk, stored as two N×N×2 solution blocks.
Represents the propagator Φ(ψ₂,ψ₁) computed by integrating the EL ODE from two
identity-block initial conditions:

  - `block_upper_ic`: result of integrating with IC = (I_N, 0_N)  (U₁ = I, U₂ = 0)
  - `block_lower_ic`: result of integrating with IC = (0_N, I_N)  (U₁ = 0, U₂ = I)

Applying the propagator to the current state `u_prev`:

u₁_new = block_upper_ic[:,:,1] · u₁_prev + block_lower_ic[:,:,1] · u₂_prev
u₂_new = block_upper_ic[:,:,2] · u₁_prev + block_lower_ic[:,:,2] · u₂_prev

Each chunk starts from an identity IC rather than the accumulated state, so chunks integrate in parallel.
"""
struct ChunkPropagator
    block_upper_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (I, 0)
    block_lower_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (0, I)
end
ChunkPropagator(N::Int) = ChunkPropagator(zeros(ComplexF64, N, N, 2), zeros(ComplexF64, N, N, 2))
