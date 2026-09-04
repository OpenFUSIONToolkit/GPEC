# Integration-chunk and chunk-propagator types for the fundamental-matrix (Riccati/STRIDE) driver.

"""
    IntegrationChunk

A struct representing a region of integration in the Euler-Lagrange solver.

## Fields

  - `psi_start::Float64` - Starting ψ coordinate for this integration region
  - `psi_end::Float64` - Ending ψ coordinate for this integration region
  - `needs_crossing::Bool` - Whether a rational surface crossing is needed after this chunk
  - `ising::Int` - Index of the singular surface associated with this chunk (0 if none)
  - `direction::Int` - Integration direction: +1 forward (axis→edge), -1 backward (edge→axis).
    For `direction=-1` chunks, `psi_start` < `psi_end` but integration proceeds from `psi_end`
    toward `psi_start`. The resulting propagator maps state at `psi_end` → state at `psi_start`.
    Used in bidirectional parallel FM to produce well-conditioned crossing-chunk propagators:
    solutions that grow exponentially forward (toward a singularity) decay when integrated
    backward, so the backward propagator is well-conditioned.
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

Since each chunk starts from a bounded identity IC (rather than the accumulated state),
exponential growth within a chunk does not affect the conditioning of the overall
assembly. This enables `Threads.@threads` parallel integration across all chunks.
"""
struct ChunkPropagator
    block_upper_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (I, 0)
    block_lower_ic::Array{ComplexF64,3}   # shape (N, N, 2) — result from IC = (0, I)
end
ChunkPropagator(N::Int) = ChunkPropagator(zeros(ComplexF64, N, N, 2), zeros(ComplexF64, N, N, 2))
