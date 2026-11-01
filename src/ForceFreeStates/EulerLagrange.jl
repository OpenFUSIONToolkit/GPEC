"""
EdgeScanState

Holds the state and results for the edge dW stability scan over ψ ∈ [dW_edge_scan_start, psilim].
Initialized and populated by `findmax_dW_edge!`; results written to HDF5 under `EdgeScan/`.
The energies are generalized (W, N) pencil values: power-normalized and invariant to the
working (Jacobian) coordinate (see `power_norm_matrix!`).

## Fields

  - `wvmat` - Precomputed wv matrix spline (raw, no singfac); singfac applied analytically in `free_compute_total`.
  - `wv_hint::Base.RefValue{Int}` - Search hint for wvmat spline (different grid from equilibrium profiles).
  - `psi, q` - ψ and q values at each edge scan step.
  - `total_eigenvalue, plasma_energy, vacuum_energy, vacuum_eigenvalue` - Power-normalized energy components at each step (NaN for steps where the wp solve was singular).
    These drive the truncation choice and are written to `EdgeScan/`.
"""
@kwdef mutable struct EdgeScanState
    numpert_total::Int
    N_edge::Int

    # Vacuum matrix spline and evaluation infrastructure
    wvmat::CubicSeriesInterpolant{Float64,ComplexF64} = _empty_series_interp_complex(numpert_total^2)
    wv_hint::Base.RefValue{Int} = Ref(1)

    # Scan results; NaN where the wp solve was singular
    psi::Vector{Float64} = Vector{Float64}(undef, N_edge)
    q::Vector{Float64} = Vector{Float64}(undef, N_edge)
    total_eigenvalue::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    plasma_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_energy::Vector{ComplexF64} = fill(complex(NaN), N_edge)
    vacuum_eigenvalue::Vector{Float64} = fill(NaN, N_edge)
end

EdgeScanState(numpert_total::Int, N_edge::Int) = EdgeScanState(; numpert_total, N_edge)

"""
OdeState

Mutable state of a Forward Euler-Lagrange integration: the stored solution, Gaussian-reduction
records needed to undo it, and singular-surface asymptotic coefficients.

## Fields

  - `numpert_total::Int` - Total number of (m, n) modes.
  - `numunorms_init::Int` - Allocated number of Gaussian-reduction records.
  - `msing::Int` - Number of singular surfaces (sizes `ca_l`/`ca_r`).
  - `numsteps_init::Int` - Initial allocation of stored steps.
  - `step::Int` - Index of the next stored step (1-based).
  - `total_steps::Int` - Total ODE steps taken, saved or not.
  - `psi_store::Vector{Float64}` - ψ at each stored step.
  - `q_store::Vector{Float64}` - q at each stored step.
  - `u_store::Array{ComplexF64,4}` - Solution `(u₁, u₂)` at each stored step, shape `(numpert_total, numpert_total, 2, nstep)`.
  - `du_store::Array{ComplexF64,3}` - du₁/dψ at each stored step, shape `(numpert_total, numpert_total, step)`; empty until `materialize_derivative_stores!` fills it (the
    galerkin-matched path supplies it at construction).
  - `xi_s_store::Array{ComplexF64,3}` - Clebsch displacement Ξ_s (Glasser 2016 Eq. 18) at each stored step; empty until materialized, like `du_store`.
  - `u_store_el_basis::Bool` - True when `u_store` holds the Euler-Lagrange state; false on the sparse parallel path (Riccati-basis chunk endpoints), where derivative
    stores cannot be materialized.
  - `du_store_populated::Bool` - True once `du_store`/`xi_s_store` hold valid data in the final basis.
  - `crit_store::Vector{Float64}` - Critical eigenvalue (smallest eigenvalue of W⁻¹) at each stored step.
  - `ca_r::Array{ComplexF64,4}` - Asymptotic coefficients right of each singular surface, shape `(numpert_total, numpert_total, 2, msing)`.
  - `ca_l::Array{ComplexF64,4}` - Asymptotic coefficients left of each singular surface, same shape.
  - `ca_populated::Bool` - True once an ideal crossing has filled `ca_l`/`ca_r`; false on kinetic and galerkin-matched runs, where the writer emits zero-extent datasets.
  - `edge_scan::EdgeScanState` - Edge dW scan results; a disabled sentinel (`N_edge = 0`) unless `findmax_dW_edge!` ran.
  - `psifac::Float64` - Current ψ of the integrator.
  - `q::Float64` - q at `psifac`.
  - `u::Array{ComplexF64,3}` - Current solution, shape `(numpert_total, numpert_total, 2)`.
  - `ising_start::Int` - Index of the last singular surface inside the starting ψ.
  - `needs_crossing::Bool` - Whether a rational surface must be crossed after the current region.
  - `nzero::Int` - Number of detected zero crossings of the critical eigenvalue.
  - `new::Bool` - Whether the next norm call records fresh reference norms `unorm0`.
  - `unorm::Vector{Float64}` - Current solution-column norms.
  - `unorm0::Vector{Float64}` - Reference norms since the last reduction.
  - `ifix::Int` - Number of Gaussian reductions performed.
  - `index::Array{Int,2}` - Pivot column order for each reduction, shape `(numpert_total, numunorms_init)`.
  - `sing_flag::Vector{Bool}` - Whether each reduction was at a singular surface.
  - `zeroed_idx::Vector{Vector{Int}}` - For each ideal crossing, the positions in `index` of the zeroed resonant solutions.
  - `fixfac::Array{ComplexF64,3}` - Gaussian-reduction factors, shape `(numpert_total, numpert_total, numunorms_init)`.
  - `fixstep::Vector{Int64}` - Last stored step before each reduction.
  - `spline_hint::Base.RefValue{Int}` - Shared interval hint for profile splines evaluated at the same ψ.
  - `mats_hint::Base.RefValue{Int}` - Interval hint for the matrix splines; per `OdeState`, so per-thread copies do not race.
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int
    numunorms_init::Int
    msing::Int
    numsteps_init::Int

    # Saved data throughout integration
    step::Int = 1
    total_steps::Int = 0
    psi_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    q_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    u_store::Array{ComplexF64,4} = Array{ComplexF64}(undef, numpert_total, numpert_total, 2, numsteps_init)
    du_store::Array{ComplexF64,3} = Array{ComplexF64}(undef, numpert_total, numpert_total, 0)
    xi_s_store::Array{ComplexF64,3} = Array{ComplexF64}(undef, numpert_total, numpert_total, 0)
    u_store_el_basis::Bool = true
    du_store_populated::Bool = false
    crit_store::Vector{Float64} = Vector{Float64}(undef, numsteps_init)
    ca_r::Array{ComplexF64,4} = zeros(ComplexF64, numpert_total, numpert_total, 2, msing)
    ca_l::Array{ComplexF64,4} = zeros(ComplexF64, numpert_total, numpert_total, 2, msing)
    ca_populated::Bool = false

    edge_scan::EdgeScanState = EdgeScanState(numpert_total, 0)

    # Data for integrator
    psifac::Float64 = 0.0
    q::Float64 = 0.0
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ising_start::Int = 0
    needs_crossing::Bool = false
    nzero::Int = 0

    # Used for Gaussian reduction
    new::Bool = true
    unorm::Vector{Float64} = zeros(Float64, numpert_total)
    unorm0::Vector{Float64} = zeros(Float64, numpert_total)
    ifix::Int = 0
    index::Array{Int,2} = zeros(Int, numpert_total, numunorms_init)
    sing_flag::Vector{Bool} = falses(numunorms_init)
    zeroed_idx::Vector{Vector{Int}} = [Int[] for _ in 1:numunorms_init]
    fixfac::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, numunorms_init)
    fixstep::Vector{Int64} = zeros(Int64, numunorms_init)

    # Spline interval hints
    spline_hint::Base.RefValue{Int} = Ref(1)
    mats_hint::Base.RefValue{Int} = Ref(1)
end

OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) =
    OdeState(; numpert_total, numsteps_init, numunorms_init, msing)

"""
    compute_delta_prime_from_ca!(odet, intr, equil)

**Stub, not physically valid.** Per-surface Δ' estimate `(ca_r[i,i,2,s] - ca_l[i,i,2,s]) / (4π²·psio)`
from the asymptotic coefficients. It ignores inter-surface coupling and the vacuum, so it does not
agree with the physical `ForceFreeStatesInternal.delta_prime_matrix`. Used only by tests.
"""
function compute_delta_prime_from_ca!(odet::OdeState, intr::ForceFreeStatesInternal, equil::Equilibrium.PlasmaEquilibrium)
    denom = (2π)^2 * equil.psio  # = twopi * chi1 in SingularCoupling.jl
    for s in 1:intr.msing
        sing = intr.sing[s]
        n_modes = length(sing.m)
        resize!(intr.sing[s].delta_prime, n_modes)
        for i in 1:n_modes
            ipert_res = 1 + sing.m[i] - intr.mlow + (sing.n[i] - intr.nlow) * intr.mpert
            if 1 <= ipert_res <= intr.numpert_total
                Δca = odet.ca_r[ipert_res, ipert_res, 2, s] - odet.ca_l[ipert_res, ipert_res, 2, s]
                intr.sing[s].delta_prime[i] = Δca / denom
            else
                intr.sing[s].delta_prime[i] = 0.0 + 0.0im
            end
        end
    end
end

# Empirical (a, b) of STRIDE's log-divergent ODE cost model at the axis, rational surfaces and edge.
# Changing them changes the chunk split, and hence every Riccati output.
const ODE_COST_AXIS = (a=39695.0, b=212830.0)
const ODE_COST_RAT = (a=17147.0, b=470710.0)
const ODE_COST_EDGE = (a=1646.0, b=4683.0)

"""
    ode_itime_cost(psi1, psi2, intr) -> Float64

Relative ODE integration cost of [ψ₁, ψ₂] from STRIDE's empirical log-divergent model (Glasser 2018),
with coefficients `ODE_COST_AXIS`, `ODE_COST_RAT`, `ODE_COST_EDGE`. Additive over sub-intervals,
so `balance_integration_chunks` can bisect for equal cost.
"""
function ode_itime_cost(psi1::Float64, psi2::Float64, intr::ForceFreeStatesInternal)
    _logdiv(a, b, x1, x2) = (a / b) * abs(log(1.0 + b * abs(x2)) - log(1.0 + b * abs(x1)))

    cost = _logdiv(ODE_COST_AXIS.a, ODE_COST_AXIS.b, psi1, psi2)
    for sing in intr.sing
        cost += _logdiv(ODE_COST_RAT.a, ODE_COST_RAT.b, psi1 - sing.psifac, psi2 - sing.psifac)
    end
    cost += _logdiv(ODE_COST_EDGE.a, ODE_COST_EDGE.b, psi1 - intr.psilim, psi2 - intr.psilim)
    return cost
end

"""
    balance_integration_chunks(chunks, ctrl, intr) -> Vector{IntegrationChunk}

Split the highest-`ode_itime_cost` chunk at its equal-cost midpoint until the count reaches
`ctrl.algorithm.nchunks` (`0` = auto). The target never depends on the thread count, so Riccati
output is thread-count independent. Only the last sub-chunk of a split keeps its crossing.
"""
function balance_integration_chunks(chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)
    min_chunks = 2 * intr.msing + 3
    # Enough sub-chunks per segment that no single long-span propagator is too ill-conditioned for the BVP
    min_bvp_intervals = 8 * (intr.msing + 1) + intr.msing
    if ctrl.algorithm.nchunks > 0
        if ctrl.algorithm.nchunks < min_chunks
            @warn "nchunks = $(ctrl.algorithm.nchunks) is below the $min_chunks chunks required by $(intr.msing) singular surfaces; clamping up."
        end
        target_n = max(ctrl.algorithm.nchunks, min_chunks)
    else
        target_n = max(min_chunks, min_bvp_intervals)
    end

    result = collect(chunks)

    while length(result) < target_n
        # Find the highest-cost splittable chunk
        best_idx = 0
        best_cost = -Inf
        for (i, chunk) in enumerate(result)
            width = chunk.psi_end - chunk.psi_start
            if width > 1e-8
                c = ode_itime_cost(chunk.psi_start, chunk.psi_end, intr)
                if c > best_cost
                    best_cost = c
                    best_idx = i
                end
            end
        end

        best_idx == 0 && break  # No more splittable chunks

        chunk = result[best_idx]
        total_cost = best_cost
        target_cost = total_cost / 2.0

        # Bisect for ψ_mid with cost(psi_start, ψ_mid) ≈ target_cost
        lo, hi = chunk.psi_start, chunk.psi_end
        for _ in 1:50
            mid = (lo + hi) / 2.0
            if ode_itime_cost(chunk.psi_start, mid, intr) < target_cost
                lo = mid
            else
                hi = mid
            end
        end
        psi_mid = (lo + hi) / 2.0

        left = IntegrationChunk(; psi_start=chunk.psi_start, psi_end=psi_mid,
            needs_crossing=false, ising=0, direction=1)
        right = IntegrationChunk(; psi_start=psi_mid, psi_end=chunk.psi_end,
            needs_crossing=chunk.needs_crossing, ising=chunk.ising,
            direction=chunk.direction)
        splice!(result, best_idx, [left, right])
    end

    return result
end

"""
    eulerlagrange_integration(ctrl, equil, mats, intr) -> (odet, propagators, chunks, S_left)

Integrate the Euler-Lagrange equations from the axis to `intr.psilim`, crossing each singular
surface on the way. Dispatches on `ctrl.algorithm` to
[`riccati_eulerlagrange_integration`](@ref) (the chunked propagator BVP) or
[`forward_eulerlagrange_integration`](@ref).

Only the Riccati branch populates `propagators` / `chunks` / `S_left`, which
`compute_delta_prime_matrix!` consumes for the Δ' BVP; the forward branch returns `nothing`
for all three.
"""
function eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

    if ctrl.algorithm isa Riccati
        ctrl.kinetic !== nothing && error("Riccati() does not support kinetic runs; use Forward().")
        return riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
    elseif ctrl.algorithm isa Forward
        return forward_eulerlagrange_integration(ctrl, equil, mats, intr)
    elseif ctrl.algorithm isa Galerkin
        error("Galerkin() solves the Euler-Lagrange system variationally, not by ODE integration; " *
              "it is dispatched to galerkin_solve.")
    end
    error("Unknown integrator: $(typeof(ctrl.algorithm)). Expected Forward, Riccati or Galerkin.")
end

"""
    forward_eulerlagrange_integration(ctrl, equil, mats, intr; verbose=intr.verbose) -> (odet, nothing, nothing, nothing)

Forward branch of [`eulerlagrange_integration`](@ref): integrates chunk by chunk from the axis,
applying Gaussian reduction whenever a solution norm ratio exceeds `ctrl.algorithm.renorm_threshold` and undoing it
via `transform_u!` at the end, so `odet.u_store` comes back dense in the axis basis. Call
directly to force this branch regardless of `ctrl.algorithm`; `verbose` overrides
`intr.verbose` for progress logging.
"""
function forward_eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal;
    verbose::Bool=intr.verbose)

    odet = OdeState(intr.numpert_total, ctrl.algorithm.numsteps_init, ctrl.algorithm.numunorms_init, intr.msing)
    initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)

    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    if verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    for chunk in chunks
        integrate_el_region!(odet, ctrl, equil, mats, intr, chunk)
        if verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" odet.q)),  steps = $(odet.total_steps)"
        end

        # Cross a singular surface after integration if this chunk requires it
        if chunk.needs_crossing
            if ctrl.kinetic !== nothing
                cross_kinetic_singular_surf!(odet, ctrl, equil, mats, intr, chunk.ising)
            else
                cross_ideal_singular_surf!(odet, ctrl, equil, mats, intr, chunk.ising)
            end
        end
    end

    # `odet.step` points one past the last stored step; trim unused storage
    odet.step -= 1
    trim_storage!(odet)

    scan_edge_dW!(odet, ctrl, equil, mats, intr; verbose)

    if verbose
        @info "Evaluating fixed-boundary stability criterion"
    end
    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    # Undo Gaussian reduction to get true solution vectors
    transform_u!(odet, intr)

    return (odet, nothing, nothing, nothing)
end

"""
    compute_axis_init(mats, profiles, intr, psi_low) -> (U1_init, U2_init)

Opt-in diagonal approximation to the axis Frobenius problem [Glasser Phys. Plasmas 2016 112506 Eq. 51],
evaluated at finite `psi_low` rather than in the limit ψ → 0. Used only when `frobenius_psi_max` is
positive and the integration starts at or below it.

For each mode, forms the 2×2 diagonal block of ψ_low · M from the diagonal entries of F̄⁻¹, K̄, and Ḡ,
discards mode coupling, and returns that block's eigenvector normalized so that U₂ = I (or U₁ = 1, U₂ = 0
when the momentum component underflows). The component ratio does not scale as a positive power of
`psi_low`. A negative Ḡ diagonal, including the m = 0 block, gives an imaginary conjugate pair, so the
returned U₁ is non-real and which branch is stored follows the eigen-solver's column order.
"""
function compute_axis_init(mats::MatrixSplines, profiles::Equilibrium.ProfileSplines,
    intr::ForceFreeStatesInternal, psi_low::Float64)
    N = intr.numpert_total
    hint = Ref(1)

    F_lower = zeros(ComplexF64, N, N)
    kmat = zeros(ComplexF64, N, N)
    gmat = zeros(ComplexF64, N, N)
    mats.ideal.F_spline_lower(vec(F_lower), psi_low; hint=hint)
    mats.ideal.K_spline(vec(kmat), psi_low; hint=hint)
    mats.ideal.G_spline(vec(gmat), psi_low; hint=hint)

    # singfac[j] = 1 / (m_j − n_j · q) for each mode j
    q0 = profiles.q_spline(psi_low; hint=hint)
    singfac = vec(1.0 ./ ((intr.mlow:intr.mhigh) .- q0 .* (intr.nlow:intr.nhigh)'))

    # F̄⁻¹ = (F_lower · F_lower')⁻¹ via the Cholesky factor
    Finv = Matrix{ComplexF64}(I, N, N)
    ldiv!(LowerTriangular(F_lower), Finv)
    ldiv!(UpperTriangular(F_lower'), Finv)

    U1_init = zeros(ComplexF64, N, N)
    U2_init = Matrix{ComplexF64}(I, N, N)

    for j in 1:N
        sf = singfac[j]
        fi = Finv[j, j]
        k = kmat[j, j]
        kd = conj(k)          # K̄†[j,j]
        g = gmat[j, j]

        # 2×2 ODE matrix block for mode j [Glasser 2016 Eq. 22-24, diagonal approximation]
        m11 = -sf * fi * k
        m12 = sf^2 * fi
        m21 = g - kd * fi * k
        m22 = sf * kd * fi

        # Frobenius matrix A₀_j = ψ_low · M_j [Glasser 2016 Eq. 51]
        #! format: off
        F_eig = eigen([psi_low*m11  psi_low*m12;
                       psi_low*m21  psi_low*m22])
        #! format: on
        eig_vals = F_eig.values
        eig_vecs = F_eig.vectors

        # Regular eigenvector: larger Re(a); if degenerate (m = 0), the one dominated by U₁
        r1 = real(eig_vals[1])
        r2 = real(eig_vals[2])
        i_reg = if abs(r1 - r2) > Base.sqrt(Base.eps(Float64))
            r1 > r2 ? 1 : 2
        else
            abs(eig_vecs[1, 1]) >= abs(eig_vecs[2, 1]) ? 1 : 2
        end

        v1, v2 = eig_vecs[1, i_reg], eig_vecs[2, i_reg]

        # Normalize to U₂ = 1, or U₁ = 1, U₂ = 0 if v₂ ≈ 0
        if abs(v2) > Base.sqrt(Base.eps(Float64)) * abs(v1)
            U1_init[j, j] = v1 / v2
        else
            U1_init[j, j] = one(ComplexF64)
            U2_init[j, j] = zero(ComplexF64)
        end
    end

    return U1_init, U2_init
end

"""
    initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, mats::MatrixSplines, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

Initialize the OdeState at the axis start: `psifac`, `ising_start` and `u`.
The default `ctrl.frobenius_psi_max = 0` uses the fixed start, U₁ = 0, U₂ = I, with no warning. The
Frobenius start is used only when `frobenius_psi_max` is positive and the starting surface lies at or
below it.
"""
function initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, mats::MatrixSplines,
    profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

    # Start at the qlow bound set by sing_min!, or the first grid node
    odet.psifac = max(intr.psilow, profiles.xs[1])

    # Last singular (kinsing on kinetic runs) surface inside the start
    if ctrl.kinetic !== nothing
        odet.ising_start = searchsortedfirst(getfield.(intr.kinsing, :psifac), odet.psifac) - 1
    else
        odet.ising_start = searchsortedfirst(getfield.(intr.sing, :psifac), odet.psifac) - 1
    end

    fixed_start = odet.psifac > ctrl.frobenius_psi_max
    if fixed_start
        # U₁ = 0, U₂ = I: ξ^ψ = 0 at the starting surface [Glasser 2016 §VI]
        for ipert in 1:intr.numpert_total
            odet.u[ipert, ipert, 2] = 1
        end
    else
        @warn "Frobenius axis start at ψ_N = $(round(odet.psifac; sigdigits=4)). " *
              "This start is currently unreliable - use only if explicitly requested." *
              "Set frobenius_psi_max = 0 for the fixed start (U₁ = 0, U₂ = I)."
        U1_init, U2_init = compute_axis_init(mats, profiles, intr, odet.psifac)
        odet.u[:, :, 1] .= U1_init
        odet.u[:, :, 2] .= U2_init
    end
end

"""
    chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)

Split the domain from `odet.psifac` to the edge into `IntegrationChunk`s, each ending
`singfac_min` short of the next singular surface to cross (kinetic runs: kinetic singular surfaces).
"""
function chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal; bidirectional::Bool=false)
    chunks = IntegrationChunk[]

    psi_current = odet.psifac
    ising_current = odet.ising_start

    # Next singular surface that is resonant in [mlow, mhigh] or beyond psilim
    function find_next_resonant_surface!(ising::Int, intr::ForceFreeStatesInternal)
        ising += 1
        while ising <= intr.msing
            if intr.psilim < intr.sing[ising].psifac ||
               any(m -> intr.mlow <= m <= intr.mhigh, intr.sing[ising].m)
                break
            end
            ising += 1
        end
        return ising
    end

    # Next kinetic singular surface that is resonant in [mlow, mhigh] or beyond psilim
    function find_next_kinsing!(ising::Int, intr::ForceFreeStatesInternal)
        ising += 1
        while ising <= intr.kmsing
            if intr.psilim < intr.kinsing[ising].psifac
                break
            end
            nq = intr.kinsing[ising].q * minimum(intr.kinsing[ising].n)
            if intr.mlow <= nq && nq <= intr.mhigh
                break
            end
            ising += 1
        end
        return ising
    end

    if ctrl.kinetic !== nothing && intr.kmsing > 0 && ctrl.singfac_min > 0
        # Kinetic: chunk around each kinetic singular surface, where F̄⁻¹ blows up
        ising_current = find_next_kinsing!(ising_current, intr)
        while ising_current <= intr.kmsing && intr.psilim >= intr.kinsing[ising_current].psifac && ctrl.singfac_min != 0
            # Stop singfac_min/|n q'| short of the surface
            psi_end = intr.kinsing[ising_current].psifac -
                      ctrl.singfac_min / abs(minimum(intr.kinsing[ising_current].n) * intr.kinsing[ising_current].q1)

            if psi_current >= psi_end
                # The crossing standoff reaches back past the current chunk start: skip this surface, loudly.
                ks = intr.kinsing[ising_current]
                @warn "Skipping kinetic singular surface m/n=$(join(string.(ks.m, "/", ks.n), ", ")) at ψ=$(ks.psifac) (q'=$(ks.q1)): " *
                      "its crossing standoff singfac_min/|n q'| = $(ks.psifac - psi_end) reaches back past the current chunk start ψ=$psi_current."
                ising_current = find_next_kinsing!(ising_current, intr)
                continue
            end

            push!(chunks, IntegrationChunk(;
                psi_start=psi_current,
                psi_end=psi_end,
                needs_crossing=true,
                ising=ising_current
            ))

            # After crossing, jump to the other side of the singular surface
            dpsi = intr.kinsing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            ising_current = find_next_kinsing!(ising_current, intr)
        end

        # Final chunk to the edge
        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    elseif ctrl.kinetic !== nothing
        # Kinetic with no kinsing surfaces (or singfac_min = 0): a single chunk
        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    else
        ising_current = find_next_resonant_surface!(ising_current, intr)
        while ising_current <= intr.msing && intr.psilim >= intr.sing[ising_current].psifac && ctrl.singfac_min != 0
            psi_end = intr.sing[ising_current].psifac - ctrl.singfac_min /
                                                        abs(minimum(intr.sing[ising_current].n) * intr.sing[ising_current].q1)

            # The crossing standoff singfac_min/|n q'| must not reach back past the current chunk start.
            if psi_end <= psi_current
                sg = intr.sing[ising_current]
                error(
                    "Singular surface q=$(sg.q) at ψ=$(sg.psifac) (q'=$(sg.q1)): its crossing standoff " *
                    "singfac_min/|n q'| = $(sg.psifac - psi_end) reaches back past the current chunk start ψ=$psi_current. " *
                    "Move the domain start (psilow, or a qlow that lands on this surface) off it, or lower singfac_min if a neighboring crossing is too close."
                )
            end
            @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Overlapping chunks detected"

            push!(chunks, IntegrationChunk(;
                psi_start=psi_current,
                psi_end=psi_end,
                needs_crossing=true,
                ising=ising_current,
                direction=bidirectional ? -1 : 1
            ))

            # After crossing, jump to the other side of the singular surface
            dpsi = intr.sing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            ising_current = find_next_resonant_surface!(ising_current, intr)
        end

        @assert psi_current < intr.psilim * (1 - eps) "Final chunk has invalid bounds"
        @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Final chunk overlaps with previous chunk"

        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    end

    return chunks
end

"""
    cross_ideal_singular_surf!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

Cross ideal singular surface `ising`: reduce on the resonant rows, record the left asymptotic
coefficients, replace the resonant solutions by the small asymptotic solutions on the far side,
and record the right coefficients. Port of Fortran `ode_ideal_cross`.
"""
function cross_ideal_singular_surf!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    ising::Int
)

    singp = intr.sing[ising]
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    nres = length(ipert_res)

    # Reduce on the resonant rows so the solutions removed below are the resonant ones
    compute_solution_norms!(odet.u, odet, ctrl, intr, true; resonant_rows=ipert_res)

    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0)
    sing_asymp_left = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0)
    dpsi = singp.psifac - odet.psifac # ψ_res - ψ (positive)

    ua = sing_get_ua(sing_asymp_left, dpsi)
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # The reduction put the column pivoted on ipert_res[i] at index position i; zeroed_idx records it for transform_u!
    odet.zeroed_idx[odet.ifix] = collect(1:nres)
    for i in 1:nres
        odet.u[:, odet.index[i, odet.ifix], :] .= 0
    end

    # Trapezoidal step across the surface
    params = (ctrl, equil, mats, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Swap in the small resonant asymptotic solution on the right side
    ua = sing_get_ua(sing_asymp_right, dpsi)
    for i in 1:nres
        odet.u[ipert_res[i], :, :] .= 0
        odet.u[:, odet.index[i, odet.ifix], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
    end
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
    odet.ca_populated = true

    # No Δ' here: it needs the Riccati gauge (U₂ = I); see compute_delta_prime_matrix!
    odet.q = equil.profiles.q_spline(odet.psifac; hint=odet.spline_hint)
    store_ode_data!(odet, odet.psifac, odet.u)
end

"""
    cross_kinetic_singular_surf!(odet, ctrl, equil, mats, intr, ising)

Cross kinetic singular surface `ising` with a trapezoidal step between the two sides, with no
asymptotics. Port of Fortran `ode_kin_cross` with `con_flag=true`.
"""
function cross_kinetic_singular_surf!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    ising::Int
)
    compute_solution_norms!(odet.u, odet, ctrl, intr, true)

    # Trapezoidal step across the surface
    ksurf = intr.kinsing[ising]
    dpsi = ksurf.psifac - odet.psifac

    params = (ctrl, equil, mats, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)

    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac = ksurf.psifac + dpsi
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    odet.q = equil.profiles.q_spline(odet.psifac; hint=odet.spline_hint)
    store_ode_data!(odet, odet.psifac, odet.u)
end

"""
    column_abstol!(abstol, u, rtol) -> abstol

Per-column absolute tolerance `max|u[:, j, k]|·rtol` for each column `j` of U₁ (`k = 1`) and U₂ (`k = 2`),
so each solution column's error is controlled relative to its own magnitude. An all-zero block (e.g. U₁ = 0
at a fixed start) takes the scale of its column's other block; the blocks differ in units, so this is a first-step safeguard.
"""
function column_abstol!(abstol::AbstractArray{Float64,3}, u::AbstractArray{<:Number,3}, rtol::Real)
    for j in axes(u, 2)
        a1 = maximum(abs, @view u[:, j, 1])
        a2 = maximum(abs, @view u[:, j, 2])
        afill = max(a1, a2, floatmin(Float64))  # keeps an all-zero column's abstol nonzero
        abstol[:, j, 1] .= (a1 > 0 ? a1 : afill) * rtol
        abstol[:, j, 2] .= (a2 > 0 ? a2 : afill) * rtol
    end
    return abstol
end

column_abstol(u::AbstractArray{<:Number,3}, rtol::Real) = column_abstol!(similar(u, Float64), u, rtol)

function refresh_column_abstol!(integrator)
    column_abstol!(integrator.opts.abstol, integrator.u, integrator.opts.reltol)
    u_modified!(integrator, false)
end

# Refresh-only callback for solves with no reduction, renormalization or storage of their own.
const COLUMN_ABSTOL_CALLBACK = DiscreteCallback((u, t, integrator) -> true, refresh_column_abstol!; save_positions=(false, false))

"""
    solve_el(prob, rtol; callback=COLUMN_ABSTOL_CALLBACK)

Vern9 solve of an Euler-Lagrange problem returning the end state, with the per-column abstol of `column_abstol!`.
A custom `callback` must refresh that abstol itself, after any reduction or renormalization of the state.
Errors when the solver stops short of the interval's end (e.g. the step size underflows), naming the ψ reached.
"""
function solve_el(prob, rtol; callback=COLUMN_ABSTOL_CALLBACK)
    sol = solve(prob, Vern9(); reltol=rtol, abstol=column_abstol(prob.u0, rtol), callback, save_everystep=false, save_end=true)
    SciMLBase.successful_retcode(sol) ||
        error("Euler-Lagrange integration aborted at ψ = $(@sprintf("%.6f", sol.t[end])) before ψ = $(@sprintf("%.6f", prob.tspan[2])) " *
              "(retcode $(sol.retcode)). Loosen eulerlagrange_tolerance or refine the ψ grid; for kinetic runs, lower kinetic_factor.")
    return sol
end

"""
    integrate_el_region!(odet, ctrl, equil, mats, intr, chunk)

Integrate the Euler-Lagrange equations over `chunk`, applying Gaussian reduction and storing steps
in `odet` from a per-step callback. Steps are stored densely near both chunk ends (in q), in the
edge-scan band, and every `save_interval` steps otherwise.
"""
function integrate_el_region!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    chunk::IntegrationChunk
)

    # Fraction of the chunk's q-range stored densely at each end
    near_q_frac = 0.05

    q_start = equil.profiles.q_spline(chunk.psi_start)
    q_end = equil.profiles.q_spline(chunk.psi_end)
    q_range = abs(q_end - q_start)

    steps_in_segment = Ref(0)

    function segment_callback!(integrator)
        ctrl, _, _, intr, odet, chunk = integrator.p

        odet.total_steps += 1
        steps_in_segment[] += 1

        compute_solution_norms!(integrator.u, odet, ctrl, intr, false)
        column_abstol!(integrator.opts.abstol, integrator.u, integrator.opts.reltol)

        # The first step is always saved, even when q_range ≈ 0
        near_start = abs(odet.q - q_start) < near_q_frac * q_range || steps_in_segment[] == 1
        near_end = abs(odet.q - q_end) < near_q_frac * q_range
        # Always save in the edge scan region so findmax_dW_edge! has dense q coverage.
        in_edge_scan = ctrl.dW_edge_scan_start < intr.psilim && integrator.t >= ctrl.dW_edge_scan_start

        if near_start || near_end || (odet.total_steps % ctrl.algorithm.save_interval == 0) || in_edge_scan
            # q at the accepted point, not the last internal Runge-Kutta stage
            odet.q = equil.profiles.q_spline(integrator.t; hint=odet.spline_hint)
            store_ode_data!(odet, integrator.t, integrator.u)
        end
    end

    cb = DiscreteCallback((u, t, integrator) -> true, segment_callback!)
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end), (ctrl, equil, mats, intr, odet, chunk))
    rtol = ctrl.algorithm.eulerlagrange_tolerance
    sol = solve_el(prob, rtol; callback=cb)

    # Always store the final (pre-crossing or edge) state
    if odet.step == 1 || odet.psi_store[odet.step-1] != sol.t[end]
        odet.q = equil.profiles.q_spline(sol.t[end]; hint=odet.spline_hint)
        store_ode_data!(odet, sol.t[end], sol.u[end])
    end

    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool; resonant_rows=Int[])

Compute the column norms of u₁ and apply `apply_gaussian_reduction!` to `u` in place when their
growth since the last reduction exceeds `ctrl.algorithm.renorm_threshold`, or when `sing_flag` is set.
The first call after a reduction only records reference norms. Errors if any norm is zero.
Port of Fortran `ode_unorm`.

### Arguments

  - `u` - Solution array, updated in place (the integrator state, to avoid copies)
  - `sing_flag` - Whether this call is at a singular surface
  - `resonant_rows` - Resonant rows at an ideal crossing; when given, the reduction always runs and pivots on them
"""
function compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool;
    resonant_rows=Int[])

    odet.unorm .= norm.(eachcol(u[:, :, 1]))
    if minimum(odet.unorm) == 0
        jmax = argmin(odet.unorm)
        error("One of the first solution vector norms unorm(1,$jmax) = 0")
    end

    # The first call after a reduction records the reference norms; an ideal crossing reduces regardless
    at_ideal_crossing = !isempty(resonant_rows)
    if odet.new && !at_ideal_crossing
        odet.new = false
        odet.unorm0 .= odet.unorm
        return
    end

    # Growth since the last reduction (raw norms if a crossing comes right after one)
    if !odet.new
        odet.unorm ./= odet.unorm0
    end
    uratio = maximum(odet.unorm) / minimum(odet.unorm)

    if uratio > ctrl.algorithm.renorm_threshold || sing_flag
        if odet.ifix < ctrl.algorithm.numunorms_init
            odet.ifix += 1
        else
            @warn "unorm storage reached, no longer saving fixfac data. Stability outputs and unorming will be correct, but cannot reconstruct `u`. \n
            Increase `numunorms_init` if needed. Automatic resizing will be added in a future version."
        end
        apply_gaussian_reduction!(u, odet, intr, sing_flag; resonant_rows)
        odet.new = true
    end
end

"""
    apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool; resonant_rows=Int[])

Triangularize the solution columns of `u` in place by Gaussian reduction, recording the factors in
`odet.fixfac`/`odet.index` for `transform_u!`. Port of Fortran `ode_fixup`.

At an ideal crossing, the first pivots are the `resonant_rows`, each on the remaining column largest
there. Growth order cannot pick the resonant solution if a `renorm_threshold` reduction fired just before the
crossing, since every column has then grown alike.
"""
function apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool; resonant_rows=Int[])

    ifix = odet.ifix
    odet.sing_flag[ifix] = sing_flag
    # The previous region ends at the step before this fixup
    odet.fixstep[ifix] = odet.step - 1

    for isol in 1:intr.numpert_total
        odet.fixfac[isol, isol, ifix] = 1
    end

    # Triangularize primary solutions, resonant rows first, then columns from largest to smallest growth
    growth_order = sortperm(odet.unorm; rev=true)
    row_free = trues(intr.numpert_total)
    col_free = trues(intr.numpert_total)
    for isol in 1:intr.numpert_total
        if isol <= length(resonant_rows)
            kpert = resonant_rows[isol]
            ksol = argmax(j -> col_free[j] ? abs(u[kpert, j, 1]) : -Inf, 1:intr.numpert_total)
        else
            ksol = growth_order[findfirst(j -> col_free[j], growth_order)]
            @views kpert = argmax(abs.(u[:, ksol, 1]) .* row_free)
        end
        odet.index[isol, ifix] = ksol
        col_free[ksol] = false
        row_free[kpert] = false
        # Eliminate other solution vectors below the pivot
        for jsol in 1:intr.numpert_total
            if col_free[jsol]
                odet.fixfac[ksol, jsol, ifix] = -u[kpert, jsol, 1] / u[kpert, ksol, 1]
                @. @views u[:, jsol, :] .= u[:, jsol, :] .+ u[:, ksol, :] .* odet.fixfac[ksol, jsol, ifix]
                u[kpert, jsol, 1] = 0
            end
        end
    end
end

"""
    findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

Fill `odet.edge_scan` with the total dW at every stored step in [`ctrl.dW_edge_scan_start`, `intr.psilim`]
and return the stored-step index of the peak total eigenvalue (NaN steps ignored).
"""
function findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

    # Stored steps from dW_edge_scan_start on are contiguous
    edge_start = findfirst(i -> odet.psi_store[i] >= ctrl.dW_edge_scan_start, 1:odet.step)
    N_edge = odet.step - edge_start + 1

    odet.edge_scan = EdgeScanState(intr.numpert_total, N_edge)
    es = odet.edge_scan

    es.psi .= odet.psi_store[edge_start:odet.step]
    es.q .= odet.q_store[edge_start:odet.step]

    es.wvmat = free_compute_wv_spline(ctrl, equil, intr)

    # Steps with a singular wp solve stay NaN
    for j in 1:N_edge
        istep = edge_start + j - 1
        odet.psifac = odet.psi_store[istep]
        odet.u .= odet.u_store[:, :, :, istep]
        try
            result = free_compute_total(equil, mats, intr, odet)
            es.total_eigenvalue[j] = result.total_eigenvalue
            es.plasma_energy[j] = result.plasma_energy
            es.vacuum_energy[j] = result.vacuum_energy
            es.vacuum_eigenvalue[j] = result.vacuum_eigenvalue
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
        end
    end

    peak_j = argmax(j -> isnan(real(es.total_eigenvalue[j])) ? typemin(Float64) : real(es.total_eigenvalue[j]), 1:N_edge)
    return edge_start + peak_j - 1
end

"""
    scan_edge_dW!(odet, ctrl, equil, mats, intr; verbose=intr.verbose) -> Bool

Record the edge-dW scan over [dW_edge_scan_start, psilim] on `odet.edge_scan`. With `truncate_at_dW_peak`, the
scan's peak becomes the plasma edge via `truncate_integration!`; otherwise the integration is left
untouched. Returns whether it truncated.
"""
function scan_edge_dW!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines, intr::ForceFreeStatesInternal; verbose::Bool=intr.verbose)
    ctrl.dW_edge_scan_start < intr.psilim || return false
    saved_psifac, saved_u = odet.psifac, copy(odet.u)
    peak_step = findmax_dW_edge!(odet, ctrl, equil, mats, intr)
    psi_peak, q_peak = odet.psi_store[peak_step], odet.q_store[peak_step]
    if ctrl.truncate_at_dW_peak
        truncate_integration!(odet, intr, equil, peak_step)
    else
        odet.psifac = saved_psifac
        odet.u .= saved_u
    end
    verbose && @info "Edge-dW peak at ψ = $(@sprintf("%.4f", psi_peak)), q = $(@sprintf("%.3f", q_peak)); " *
          (ctrl.truncate_at_dW_peak ? "adopted as the plasma edge" : "integration domain unchanged")
    return ctrl.truncate_at_dW_peak
end

"""
    transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

Undo the Gaussian reductions applied during integration, transforming the stored solutions in
`odet` in place into the true solution vectors.
"""
function transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

    gauss = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix + 1)
    gauss_buffer = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    identity = Matrix{ComplexF64}(I, intr.numpert_total, intr.numpert_total)
    temp = Matrix{ComplexF64}(undef, intr.numpert_total, intr.numpert_total)
    mask = trues(intr.numpert_total)

    # Construct gaussian reduction matrices for each fixup
    for ifix in 1:odet.ifix
        gauss[:, :, ifix] .= identity
        mask .= true
        for isol in 1:intr.numpert_total
            ksol = odet.index[isol, ifix]
            mask[ksol] = false
            temp .= identity
            for jsol in 1:intr.numpert_total
                if mask[jsol]
                    temp[ksol, jsol] = odet.fixfac[ksol, jsol, ifix]
                end
            end
            mul!(gauss_buffer, view(gauss,:,:,ifix), temp)
            gauss[:, :, ifix] .= gauss_buffer
        end
        # Drop the resonant solutions zeroed at an ideal crossing
        if odet.sing_flag[ifix]
            for zeroed_idx in odet.zeroed_idx[ifix]
                gauss[:, odet.index[zeroed_idx, ifix], ifix] .= 0.0
            end
        end
    end

    # Region i lies between fixups i-1 and i; its transform is the product of all later reductions
    transforms[:, :, end] .= identity
    for ifix in odet.ifix:-1:1
        mul!(view(transforms,:,:,ifix), view(gauss,:,:,ifix), view(transforms,:,:,(ifix+1)))
    end

    jfix = 1
    for ifix in 1:(odet.ifix+1)
        kfix = ifix != odet.ifix + 1 ? odet.fixstep[ifix] : odet.step
        @views for istep in jfix:kfix
            mul!(gauss_buffer, odet.u_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.u_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 2, istep] .= gauss_buffer
            # Only analytically supplied (galerkin) derivative stores exist yet; materialized ones come after
            if !isempty(odet.du_store)
                mul!(gauss_buffer, odet.du_store[:, :, istep], transforms[:, :, ifix])
                odet.du_store[:, :, istep] .= gauss_buffer
                mul!(gauss_buffer, odet.xi_s_store[:, :, istep], transforms[:, :, ifix])
                odet.xi_s_store[:, :, istep] .= gauss_buffer
            end
        end
        jfix = kfix + 1
    end
end

"""
    sing_der!(
        du::Array{ComplexF64,3},
        u::Array{ComplexF64,3},
        params::Tuple{ForceFreeStatesControl, Equilibrium.PlasmaEquilibrium, MatrixSplines, ForceFreeStatesInternal, OdeState, IntegrationChunk},
        psieval::Float64
    )

In-place Euler-Lagrange derivative du/dψ at `psieval` [Glasser Phys. Plasmas 2016 112506 Eq. 24],
ideal or kinetic, in the `(du, u, p, t)` form the ODE solver calls; `du` and `u` have shape
`(numpert_total, numpert_total, 2)`. Ξ_s is not computed here; see [`compute_node_xi_s!`](@ref).
"""
function sing_der!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    params::Tuple{ForceFreeStatesControl,Equilibrium.PlasmaEquilibrium,
        MatrixSplines,ForceFreeStatesInternal,OdeState,IntegrationChunk},
    psieval::Float64)
    ctrl, equil, mats, intr, odet, _ = params
    return sing_der!(du, u, ctrl, equil, mats, intr, odet, psieval)
end

"""
    sing_der!(du, u, ctrl, equil, mats, intr, odet, psieval)

Unpacked-argument form of the Euler-Lagrange derivative, using `odet`'s spline hints and
recording q at `psieval` in `odet.q`. Not safe to call concurrently on a shared `odet`;
multi-threaded callers should use [`el_derivatives!`](@ref) with their own hints.
"""
function sing_der!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ForceFreeStatesInternal, odet::OdeState, psieval::Float64)
    odet.q = el_derivatives!(du, u, ctrl.kinetic !== nothing, equil, mats, intr, psieval, odet.spline_hint, odet.mats_hint)
    return nothing
end

"""
    el_derivatives!(du, u, kinetic, equil, mats, intr, psieval, spline_hint, mats_hint) -> q

Euler-Lagrange (or, when `kinetic` is true, FKG) derivative kernel: writes du₁/dψ and du₂/dψ at
`psieval` into `du` and returns q there. Holds no state of its own — the two hints are the
caller's interval-search accelerators, so concurrent callers just pass their own.
"""
@with_pool pool function el_derivatives!(du::Array{ComplexF64,3}, u::Array{ComplexF64,3},
    kinetic::Bool, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines,
    intr::ModeSpace, psieval::Float64, spline_hint::Base.RefValue{Int}, mats_hint::Base.RefValue{Int})

    Npert = intr.numpert_total

    singfac_vec = acquire!(pool, Float64, Npert)
    singfac_mat = reshape(singfac_vec, intr.mpert, intr.npert)

    fmat_lower = acquire!(pool, ComplexF64, Npert, Npert)
    kmat = similar!(pool, fmat_lower)
    gmat = similar!(pool, fmat_lower)
    tmp_mat = similar!(pool, fmat_lower)

    fill!(tmp_mat, zero(ComplexF64))
    u1 = @view(u[:, :, 1])
    u2 = @view(u[:, :, 2])
    du1 = @view(du[:, :, 1])
    du2 = @view(du[:, :, 2])

    # singfac = 1 / (m - nq)
    q = equil.profiles.q_spline(psieval; hint=spline_hint)
    singfac_mat .= 1.0 ./ ((intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)')

    kin = mats.kinetic
    if kinetic
        kin === nothing && error("el_derivatives! called with kinetic=true but mats carries no kinetic matrices")
        # Kinetic path: FKG matrices, evaluated with the caller's hint (never a shared one)
        f0mat = similar!(pool, fmat_lower)
        pmat_kin = similar!(pool, fmat_lower)
        paat_kin = similar!(pool, fmat_lower)
        kkmat_kin = similar!(pool, fmat_lower)
        kkaat_kin = similar!(pool, fmat_lower)
        r1mat_kin = similar!(pool, fmat_lower)
        r2mat_kin = similar!(pool, fmat_lower)
        r3mat_kin = similar!(pool, fmat_lower)
        gaat_kin = similar!(pool, fmat_lower)

        kin.F0_spline(vec(f0mat), psieval; hint=mats_hint)
        kin.P_spline(vec(pmat_kin), psieval; hint=mats_hint)
        kin.P_spline_adj(vec(paat_kin), psieval; hint=mats_hint)
        kin.Kk_spline(vec(kkmat_kin), psieval; hint=mats_hint)
        kin.Kk_spline_adj(vec(kkaat_kin), psieval; hint=mats_hint)
        kin.R1_spline(vec(r1mat_kin), psieval; hint=mats_hint)
        kin.R2_spline(vec(r2mat_kin), psieval; hint=mats_hint)
        kin.R3_spline(vec(r3mat_kin), psieval; hint=mats_hint)
        kin.G_spline_adj(vec(gaat_kin), psieval; hint=mats_hint)

        # Build singfac-dependent F̄, K̄, K̄†, Ḡ† matrices (Logan 2015 Appendix C, Eqs C.5-C.11):
        # F̄(i,j) = q1*f0*q2 - q1*P - P†'*q2 + R1
        # K̄(i,j) = q1*KK + R2
        # K̄†(i,j) = KK†*q2 + R3
        # where q1 = (m₁ - n*q), q2 = (m₂ - n*q) — direct singfac, NOT 1/(m-nq) as in ideal path
        singfac_direct = acquire!(pool, Float64, Npert)
        singfac_direct_mat = reshape(singfac_direct, intr.mpert, intr.npert)
        singfac_direct_mat .= (intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)'

        # fmat_lower, kmat, kaat_kin hold F̄, K̄, K̄†
        kaat_kin = similar!(pool, fmat_lower)
        for j in 1:Npert
            q2 = singfac_direct[j]
            for i in 1:Npert
                q1 = singfac_direct[i]
                fmat_lower[i, j] = q1 * f0mat[i, j] * q2 - q1 * pmat_kin[i, j] -
                                   conj(paat_kin[j, i]) * q2 + r1mat_kin[i, j]
                kmat[i, j] = q1 * kkmat_kin[i, j] + r2mat_kin[i, j]
                kaat_kin[i, j] = kkaat_kin[i, j] * q2 + r3mat_kin[i, j]
            end
        end
        gmat .= gaat_kin

        # Kinetic ODE (Logan 2015 Eq 7.46): singfac absorbed into F̄/K̄/K̄†, no explicit Q⁻¹
        # du₁ = F̄⁻¹(u₂ - K̄·u₁)
        du1 .= u2
        mul!(tmp_mat, kmat, u1)
        du1 .-= tmp_mat
        # Kinetic F̄ is non-Hermitian: LU rather than Cholesky
        _, ipiv2, _ = LAPACK.getrf!(fmat_lower)
        LAPACK.getrs!('N', fmat_lower, ipiv2, du1)

        # du₂ = Ḡ†·u₁ + K̄†·du₁  (Logan 2015 Eq C.10-C.11)
        mul!(tmp_mat, gmat, u1)
        du2 .= tmp_mat
        mul!(tmp_mat, kaat_kin, du1)
        du2 .+= tmp_mat

    else
        # Ideal path [Glasser 2016 Eqs. 22-24]
        mats.ideal.F_spline_lower(vec(fmat_lower), psieval; hint=mats_hint)
        mats.ideal.K_spline(vec(kmat), psieval; hint=mats_hint)
        mats.ideal.G_spline(vec(gmat), psieval; hint=mats_hint)

        # du[1] = - F̄⁻¹ * K̄ * u[1] + F̄⁻¹ * Q⁻¹ * u[2]
        du1 .= u2 .* singfac_vec
        mul!(tmp_mat, kmat, u1)
        du1 .-= tmp_mat
        ldiv!(LowerTriangular(fmat_lower), du1)
        ldiv!(UpperTriangular(fmat_lower'), du1)
        # du[2] = G * u[1] + K̄^† * du[1] = G * u[1] - K̄^† * F̄⁻¹ * K̄ * u[1] + K̄^† * F̄⁻¹ * Q⁻¹ * u[2]
        mul!(tmp_mat, gmat, u1)
        du2 .= tmp_mat
        mul!(tmp_mat, adjoint(kmat), du1)
        du2 .+= tmp_mat
        # du[1] = - Q⁻¹ * F̄⁻¹ * K̄ * u[1] + Q⁻¹ * F̄⁻¹ * Q⁻¹ * u[2]
        du1 .*= singfac_vec
    end
    return q
end

"""
    compute_node_xi_s!(xi_s, du1, u1, mats, psieval; kinetic=false, hint=Ref(1))

Evaluate Ξ_s = -A⁻¹(B·Ξ′_ψ + C·Ξ_ψ) [Glasser Phys. Plasmas 2016 112506 Eq. 18] at `psieval` into
`xi_s`. `du1` and `u1` are the Ξ′_ψ and Ξ_ψ blocks at that ψ (an `el_derivatives!` result and its input).
With `kinetic=true` the non-Hermitian kinetic A is used.
"""
@with_pool pool function compute_node_xi_s!(xi_s::AbstractMatrix{ComplexF64}, du1::AbstractMatrix{ComplexF64},
    u1::AbstractMatrix{ComplexF64}, mats::MatrixSplines, psieval::Float64; kinetic::Bool=false, hint::Base.RefValue{Int}=Ref(1))

    Npert = size(u1, 1)
    amat = acquire!(pool, ComplexF64, Npert, Npert)
    bmat = similar!(pool, amat)
    cmat = similar!(pool, amat)
    tmp_mat = similar!(pool, amat)

    # The kinetic A is non-Hermitian, hence LU below instead of Cholesky
    active_mats = kinetic ? mats.kinetic : mats.ideal
    active_mats === nothing && error("compute_node_xi_s! called with kinetic=true but mats carries no kinetic matrices")
    active_mats.A_spline(vec(amat), psieval; hint=hint)
    active_mats.B_spline(vec(bmat), psieval; hint=hint)
    active_mats.C_spline(vec(cmat), psieval; hint=hint)

    # bmat = A⁻¹ bmat, cmat = A⁻¹ cmat in place
    if kinetic
        _, ipiv, _ = LAPACK.getrf!(amat)
        LAPACK.getrs!('N', amat, ipiv, bmat)
        LAPACK.getrs!('N', amat, ipiv, cmat)
    else
        LAPACK.potrf!('U', amat)
        LAPACK.potrs!('U', amat, bmat)
        LAPACK.potrs!('U', amat, cmat)
    end

    mul!(tmp_mat, bmat, du1)
    xi_s .= .-tmp_mat
    mul!(tmp_mat, cmat, u1)
    xi_s .-= tmp_mat
    return xi_s
end
