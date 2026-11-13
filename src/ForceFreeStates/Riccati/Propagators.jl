# Chunk propagators: their integration and application, and the reduction of a state to (S, I).

"""
    renormalize_riccati_inplace!(u, N)

Reduce the N × N × 2 state `u` = (U₁, U₂) in place to (S, I), S = U₁U₂⁻¹.
"""
function renormalize_riccati_inplace!(u::Array{ComplexF64,3}, N::Int)
    U2_copy = copy(@view u[:, :, 2])
    rdiv!(@view(u[:, :, 1]), lu!(U2_copy))
    fill!(@view(u[:, :, 2]), 0)
    for i in 1:N
        u[i, i, 2] = 1
    end
end

"""
    propagator_start(chunks, i, bases, N) -> Array{ComplexF64,3}

Start basis (N × 2N × 2) of chunk `i`'s propagator: the left basis of its surface for a crossing chunk, the
right basis for the chunk after one [Glasser & Kolemen 2018, Phys. Plasmas 25, 082502, Eq. 33], else the identity.
"""
function propagator_start(chunks::Vector{IntegrationChunk}, i::Int, bases::AbstractDict, N::Int)
    chunks[i].needs_crossing && return bases[chunks[i].ising].left
    i > 1 && chunks[i-1].needs_crossing && return bases[chunks[i-1].ising].right
    u0 = zeros(ComplexF64, N, 2N, 2)
    for j in 1:N
        u0[j, j, 1] = 1
        u0[j, N+j, 2] = 1
    end
    return u0
end

"""
    chunk_propagator(chunk, u0, rtol, equil, mats, modes) -> ChunkPropagator

Integrate the ideal EL ODE over `chunk` from the start basis `u0` at relative tolerance `rtol`, keeping every
solver step.
"""
function chunk_propagator(chunk::IntegrationChunk, u0::Array{ComplexF64,3}, rtol::Float64,
    equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, modes::ModeIndices)
    tspan = chunk.direction == 1 ? (chunk.psi_start, chunk.psi_end) : (chunk.psi_end, chunk.psi_start)
    spline_hint, mats_hint = Ref(1), Ref(1)
    function derivative!(du, u, _, psi)
        el_derivatives!(du, u, false, equil, mats, modes, psi, spline_hint, mats_hint)
        return nothing
    end
    sol = solve_el(ODEProblem(derivative!, copy(u0), tspan), rtol; save_everystep=true)
    return ChunkPropagator(sol.u[end], sol.t, sol.u)
end

"""
    apply_propagator!(odet, prop)

Multiply `odet.u` by the chunk propagator `prop` [Glasser, Kolemen & Glasser 2018, Phys. Plasmas 25, 032507, Eq. 29].
"""
function apply_propagator!(odet::OdeState, prop::ChunkPropagator)
    x = [odet.u[:, :, 1]; odet.u[:, :, 2]]
    @views for k in 1:2
        mul!(odet.u[:, :, k], prop.u[:, :, k], x)
    end
end

"""
    apply_propagator_inverse!(odet, prop)

Solve the chunk propagator `prop` against `odet.u`, which carries a state across a backward chunk
[Glasser, Kolemen & Glasser 2018, Phys. Plasmas 25, 032507, Eq. 33].
"""
function apply_propagator_inverse!(odet::OdeState, prop::ChunkPropagator)
    N = size(odet.u, 1)
    x = [prop.u[:, :, 1]; prop.u[:, :, 2]] \ [odet.u[:, :, 1]; odet.u[:, :, 2]]
    odet.u[:, :, 1] .= x[1:N, :]
    odet.u[:, :, 2] .= x[(N+1):(2N), :]
end
