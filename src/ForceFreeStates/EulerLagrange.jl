"""
EdgeScanState

Holds the state and results for the edge dW stability scan over ψ ∈ [psiedge, psilim].
Initialized and populated by `findmax_dW_edge!`; results written to HDF5 under `EdgeScan/`.
The energies are generalized (W, N) pencil values: power-normalized and invariant to the
working (Jacobian) coordinate (see `power_norm_matrix!`).

## Fields

  - `wvmat` - Precomputed wv matrix spline (raw, no singfac); singfac applied analytically in `free_compute_total`.
  - `wv_hint::Base.RefValue{Int}` - Search hint for wvmat spline (different grid from equilibrium profiles).
  - `psi, q` - ψ and q values at each edge scan step.
  - `total_eigenvalue, plasma_energy, vacuum_energy, vacuum_eigenvalue` - Power-normalized energy components at each step (NaN for steps where the wp solve was singular). These drive the truncation choice and are written to `EdgeScan/`.
"""
@kwdef mutable struct EdgeScanState
    numpert_total::Int
    N_edge::Int

    # Vacuum matrix spline and evaluation infrastructure
    wvmat::CubicSeriesInterpolant{Float64,ComplexF64} = _empty_series_interp_complex(numpert_total^2)
    wv_hint::Base.RefValue{Int} = Ref(1)

    # Scan results (written to HDF5 under EdgeScan/; NaN where free_compute_total raised SingularException)
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

A mutable struct to hold the state of the ODE solver used by the ForceFreeStates integration routines.
This struct stores configuration parameters used to allocate arrays, the evolving stored
solution during integration, diagnostic arrays used for normalization / Gaussian reduction,
and a small set of temporary matrices and factors used to compute singular-layer corrections.

## Fields

  - `numpert_total::Int` - Total number of Fourier mode combinations (m × n) used in the calculation.

  - `numunorms_init::Int` - Initial allocation size for the number of normalization operations recorded.

  - `msing::Int` - Number of singular surfaces in the equilibrium (used to size asymptotic coefficient arrays).

  - `numsteps_init::Int` - Initial allocation size for the number of integration steps to store.

  - `step::Int` - Current integration step index (1-based, like `istep` in the original Fortran).

  - `psi_store::Vector{Float64}` - Stored psi values at each saved integration step (length `numsteps_init`).

  - `q_store::Vector{Float64}` - Stored q values at each saved integration step (length `numsteps_init`).

  - `u_store::Array{ComplexF64,4}` - Stored solution arrays at each saved step with shape
    `(numpert_total, numpert_total, 2, numsteps_init)` (complex solution state used by the solver).

  - `du_store::Array{ComplexF64,3}` - dΞ_ψ/dψ (the u₁ block only) at each saved step, shape
    `(numpert_total, numpert_total, step)`. Empty until `materialize_derivative_stores!` fills it,
    except on the galerkin-matched path which supplies the analytic derivative at construction.
    du₂/dψ is never stored densely — its only consumer evaluates it on demand at bracket nodes.

  - `xi_s_store::Array{ComplexF64,3}` - Clebsch displacement Ξ_s at each saved step, eq. 18 of Glasser 2016,
    shape `(numpert_total, numpert_total, step)`. Empty until materialized, same as `du_store`.

  - `u_store_el_basis::Bool` - True when `u_store` holds the Euler-Lagrange state `(u₁, u₂)`, so the
    derivative kernel can be re-applied to it. False on the sparse parallel path, whose stored columns
    are chunk-endpoint Riccati matrices; `materialize_derivative_stores!` refuses to run there.

  - `du_store_populated::Bool` - True once `du_store`/`xi_s_store` hold valid data in the final
    (post-transform, post-normalization) basis. Set by `materialize_derivative_stores!` or by the
    galerkin-matched constructor; stays false where the stores cannot be materialized, e.g. the
    sparse parallel path whose solution is in the Riccati basis.

  - `crit_store::Vector{Float64}` - Stored crit parameter values (smallest eigenvalue of W⁻ꜝ) (length `numsteps_init`).

  - `ca_r::Array{ComplexF64,4}` - Asymptotic coefficients just to the right of each singular surface
    with shape `(numpert_total, numpert_total, 2, msing)`.

  - `ca_l::Array{ComplexF64,4}` - Asymptotic coefficients just to the left of each singular surface
    with shape `(numpert_total, numpert_total, 2, msing)`.

  - `ca_populated::Bool` - True once an ideal singular-surface crossing has filled `ca_l`/`ca_r`; kinetic and
    galerkin-matched runs never populate them and leave this false, and the HDF5 writer then emits zero-extent
    `ca_left`/`ca_right` datasets instead of unpopulated arrays.

  - `edge_scan::EdgeScanState` - Edge dW scan state and results. Initialized as a disabled sentinel (N_edge=0) and replaced by `findmax_dW_edge!` when a scan runs.

  - `psifac::Float64` - Current normalized flux coordinate for the integrator.

  - `q::Float64` - Safety factor value at `psifac` (current q during integration).

  - `u::Array{ComplexF64,3}` - Current working solution arrays with shape `(numpert_total, numpert_total, 2)`.

  - `ising_start::Int` - Index of the starting singular surface to be crossed during integration.

  - `psimax::Float64` - Maximum psi value for which the integrator is allowed to run in next integration region.

  - `needs_crossing::Bool` - Flag indicating whether a rational surface needs to be crossed after the current integration region.

  - `nzero::Int` - Count of detected zero crossings (used for diagnostics).

  - `new::Bool` - Flag indicating whether a new `unorm0` should be computed after a fixup.

    # Initialization parameters

  - `unorm::Vector{Float64}` - Current norms of the solution vectors (length `numpert_total`).

  - `unorm0::Vector{Float64}` - Reference/initial norms of the solution vectors (length `numpert_total`).

    # Saved data throughout integration

  - `ifix::Int` - Number of normalization operations performed (index into normalization arrays).

# Total ODE solver steps taken (all steps, not just saved ones)

  - `index::Array{Int,2}` - Index matrix used for sorting solution norms with shape `(numpert_total, numunorms_init)`.

  - `sing_flag::Vector{Bool}` - Boolean flags indicating which stored normalizations correspond to singular solutions    # Edge dW scan state and results (disabled sentinel when psiedge >= psilim, i.e. no edge scan)
    (length `numunorms_init`).

  - `zeroed_idx::Vector{Vector{Int}}` - For each ideal rational surface jump, a vector of indices of solutions that were zeroed.    # Data for integrator

  - `fixfac::Array{ComplexF64,3}` - Fix-up factors for Gaussian reduction with shape `(numpert_total, numpert_total, numunorms_init)`.

  - `fixstep::Vector{Int64}` - Step indices (psi step positions) at which normalization/fixups were performed (length `numunorms_init`).
"""
@kwdef mutable struct OdeState
    # Initialization parameters
    numpert_total::Int
    numunorms_init::Int
    msing::Int
    numsteps_init::Int

    # Saved data throughout integration
    step::Int = 1
    total_steps::Int = 0  # Total ODE solver steps taken (all steps, not just saved ones)
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

    # Edge dW scan state and results (disabled sentinel when psiedge >= psilim, i.e. no edge scan)
    edge_scan::EdgeScanState = EdgeScanState(numpert_total, 0)

    # Data for integrator
    psifac::Float64 = 0.0
    q::Float64 = 0.0
    u::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 2)
    ising_start::Int = 0
    psimax::Float64 = 0.0
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

    # Kinetic workspace arrays: evaluated from Kw_spline/Kt_spline splines at current psi
    kwmat::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 6)
    ktmat::Array{ComplexF64,3} = zeros(ComplexF64, numpert_total, numpert_total, 6)

    # Shared hint for CubicInterpolant interval search optimization during ODE integration
    # All splines evaluated at the same psi can share this hint for O(1) interval lookups
    spline_hint::Base.RefValue{Int} = Ref(1)
    # Shared 2D hint for CubicInterpolantND (rzphi splines) during ODE integration
    # Tuple of (psi_hint, theta_hint) for O(1) interval lookups in 2D bicubic splines
    rzphi_hint::Tuple{Base.RefValue{Int},Base.RefValue{Int}} = (Ref(1), Ref(1))
    # Per-thread hint for MatrixSplines matrix splines (A_spline/B_spline/C_spline/F_spline_lower/K_spline/G_spline
    # and kinetic equivalents). Lives on OdeState — which is already cloned per thread in the
    # parallel BVP path — so concurrent sing_der! invocations don't race on a shared Ref.
    mats_hint::Base.RefValue{Int} = Ref(1)
end

OdeState(numpert_total::Int, numsteps_init::Int, numunorms_init::Int, msing::Int) =
    OdeState(; numpert_total, numsteps_init, numunorms_init, msing)

"""
    compute_delta_prime_from_ca!(odet, intr, equil)

**STUB — not physically valid.** Compute a per-surface Δ' estimate from the asymptotic
coefficients `ca_l`/`ca_r` using `Δ'[i] = (ca_r[i,i,2,s] - ca_l[i,i,2,s]) / (4π²·psio)`.

The physically valid tearing-stability Δ' is `ForceFreeStatesInternal.delta_prime_matrix`,
computed via the STRIDE global BVP in `compute_delta_prime_matrix!`. The per-surface
ca-based formula here ignores inter-surface coupling and the vacuum BC, and should
**not** be expected to agree with `delta_prime_matrix`. Retained for reference / future
work on intra-surface coupling diagnostics.

Not called from any integration driver. Used only by tests / benchmarks that exercise
the stub formula directly.
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

# Empirical log-divergent ODE-cost coefficients (a, b) for each reference point:
# axis (ψ=0, steep), rational surfaces (ψ=ψ_s, moderate), edge (ψ=ψ_lim, mild).
# Per reference, the contribution to the cost is (a/b) · |log(1 + b·|ψ-ref|)| evaluated
# at the interval endpoints. Coefficients are ported from STRIDE's ode_itime cost model
# (Fortran reference) and unchanged here. Tune only after re-fitting against a per-chunk
# step-count sweep; touching these affects parallel-chunk load balancing.
const ODE_COST_AXIS  = (a = 39695.0, b = 212830.0)
const ODE_COST_RAT   = (a = 17147.0, b = 470710.0)
const ODE_COST_EDGE  = (a =  1646.0, b =   4683.0)

"""
    ode_itime_cost(psi1, psi2, intr) -> Float64

Estimate the relative ODE integration cost for the interval [ψ₁, ψ₂] using the empirical
log-divergent cost model from STRIDE (Glasser 2018). Coefficients are the module constants
`ODE_COST_AXIS`, `ODE_COST_RAT`, `ODE_COST_EDGE`. The cost is additive for sub-intervals
not containing rational surfaces, which makes it suitable for equal-cost splitting via
bisection in `balance_integration_chunks`.
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
    min_crossing_chunks(msing) -> Int

Floor on the chunk count set by the `msing` rational-surface crossings; explicit `nchunks`
requests below it are clamped up (with a warning) by `balance_integration_chunks`.
"""
min_crossing_chunks(msing::Integer) = 2 * msing + 3

"""
    auto_chunk_target(msing) -> Int

The chunk-count target `balance_integration_chunks` resolves `nchunks = 0` to. Beyond the
crossing floor, BVP propagator conditioning needs at least 8 non-crossing sub-chunks per
segment (axis→surf₁, surfᵢ→surfᵢ₊₁, surfₙ→edge) plus the crossing chunks — without them,
`assemble_fm_matrix(condition=true)` cannot keep accumulated products well-conditioned
because single long-span propagators may already have cond ~ 10²⁴ (STRIDE uses 33 intervals
for comparable problems). Derived from `msing` alone — never from the thread count — which
is what keeps the chunk list, and hence every Riccati output, identical whatever `julia -t`
provides. Tests that steer decompositions relative to the auto target call this instead of
mirroring the formula.
"""
auto_chunk_target(msing::Integer) = max(min_crossing_chunks(msing), 8 * (msing + 1) + msing)

"""
    balance_integration_chunks(chunks, ctrl, intr) -> Vector{IntegrationChunk}

Sub-divide integration chunks to produce a load-balanced set for the Riccati BVP.
Starts from the output of `chunk_el_integration_bounds` and iteratively splits the
highest-cost chunk (by `ode_itime_cost`) until the total chunk count reaches the target
set by `ctrl.nchunks` (`0` = auto). The target is derived from problem structure only —
never from `Threads.nthreads()` — so the chunk list, and hence every Riccati output, is
identical whatever thread count `julia -t` provides.

Each split finds the equal-cost midpoint ψ_mid via bisection:
  ode_itime_cost(psi_start, psi_mid) ≈ ode_itime_cost(psi_start, psi_end) / 2

Sub-chunks inherit `needs_crossing=false` and `ising=0`. Only the LAST sub-chunk of
each original chunk retains `needs_crossing=true` and the original `ising`, so the
rational surface crossing still fires at the correct ψ in the serial assembly phase.
"""
function balance_integration_chunks(chunks::Vector{IntegrationChunk}, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)
    min_chunks = min_crossing_chunks(intr.msing)
    if ctrl.nchunks > 0
        if ctrl.nchunks < min_chunks
            @warn "nchunks = $(ctrl.nchunks) is below the $min_chunks chunks required by $(intr.msing) singular surfaces; clamping up."
        end
        target_n = max(ctrl.nchunks, min_chunks)
    else
        target_n = auto_chunk_target(intr.msing)
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

        # Bisect to find ψ_mid where cost(psi_start, ψ_mid) ≈ target_cost
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
surface on the way (Fortran `ode_run`). Dispatches on `ctrl.integrator` to
[`riccati_eulerlagrange_integration`](@ref) (the chunked propagator BVP) or
[`forward_eulerlagrange_integration`](@ref).

Only the Riccati branch populates `propagators` / `chunks` / `S_left`, which
`compute_delta_prime_matrix!` consumes for the Δ' BVP; the forward branch returns `nothing`
for all three.
"""
function eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

    if ctrl.integrator == "riccati"
        ctrl.kinetic_factor > 0 && error("kinetic runs require integrator=\"forward\"; the Riccati integrator has no kinetic crossing.")
        return riccati_eulerlagrange_integration(ctrl, equil, mats, intr)
    elseif ctrl.integrator == "forward"
        return forward_eulerlagrange_integration(ctrl, equil, mats, intr)
    elseif ctrl.integrator == "galerkin"
        error("integrator = \"galerkin\" solves the Euler-Lagrange system variationally, not by ODE integration; " *
              "it is dispatched to galerkin_solve.")
    end
    error("Unknown integrator: $(ctrl.integrator). Expected \"forward\", \"riccati\", or \"galerkin\".")
end

"""
    forward_eulerlagrange_integration(ctrl, equil, mats, intr; verbose=ctrl.verbose) -> (odet, nothing, nothing, nothing)

Forward branch of [`eulerlagrange_integration`](@ref): integrates chunk by chunk from the axis,
applying Gaussian reduction whenever a solution norm ratio exceeds `ctrl.ucrit` and undoing it
via `transform_u!` at the end, so `odet.u_store` comes back dense in the axis basis. Call
directly to force this branch regardless of `ctrl.integrator`; `verbose` overrides
`ctrl.verbose` for progress logging.
"""
function forward_eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal;
    verbose::Bool=ctrl.verbose)

    # Initialization
    odet = OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)
    if ctrl.sing_start <= 0
        initialize_el_at_axis!(odet, ctrl, mats, equil.profiles, intr)
    elseif ctrl.sing_start <= intr.msing
        error("sing_start > 0 not implemented yet!")
        # initialize_el_at_singular_surf!(ctrl, equil, intr, odet)
    else
        error("Invalid value for sing_start: $(ctrl.sing_start) > msing = $(intr.msing)")
    end

    # Pre-compute all integration chunks
    chunks = chunk_el_integration_bounds(odet, ctrl, intr)

    # Print initial integration condition
    if verbose
        @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" equil.profiles.q_spline(odet.psifac)))"
    end

    # Iterate through each integration chunk
    for chunk in chunks
        # Integrate this region and display progress
        integrate_el_region!(odet, ctrl, equil, mats, intr, chunk)
        if verbose
            @info "   ψ = $((@sprintf "%.3f" odet.psifac)),  q = $((@sprintf "%.3f" odet.q)),  steps = $(odet.total_steps)"
        end

        # Cross a singular surface after integration if this chunk requires it
        if chunk.needs_crossing
            if ctrl.kinetic_factor > 0
                cross_kinetic_singular_surf!(odet, ctrl, equil, mats, intr, chunk.ising)
            else
                cross_ideal_singular_surf!(odet, ctrl, equil, mats, intr, chunk.ising)
            end
        end
    end

    # Deallocate unused storage of integration data.
    # `odet.step` was incremented one past the last filled index in integrate_el_region!.
    odet.step -= 1
    trim_storage!(odet)

    # Edge-dW scan over [psiedge, psilim] — populates odet.edge_scan for HDF5 output.
    # The scan mutates odet.psifac and odet.u internally; save/restore them around the call.
    # findmax_dW_edge! also (re)allocates odet.edge_scan; that field is the diagnostic
    # product and is intentionally NOT restored.
    #
    # Default (ctrl.truncate_at_dW_peak = false): diagnostic-only. Integration domain is
    # determined solely by qhigh / psihigh / dmlim so Δ' and δW are independent of peak
    # location. Legacy path (true) reproduces the ode_record_edge heuristic from Fortran
    # STRIDE — psilim/qlim/u are pulled back to the dW peak. Preserved for experimental
    # work; see the ForceFreeStatesControl docstring for the reliability caveats.
    if ctrl.psiedge < intr.psilim
        saved_psifac, saved_u = odet.psifac, copy(odet.u)
        peak_step = findmax_dW_edge!(odet, ctrl, equil, mats, intr)
        if ctrl.truncate_at_dW_peak
            # Legacy: truncate integration data to dW peak (corrupts Δ' and δW).
            odet.step = peak_step
            trim_storage!(odet)
            intr.psilim = odet.psi_store[end]
            intr.qlim = odet.q_store[end]
            odet.u .= odet.u_store[:, :, :, end]
            if verbose
                @info "Truncating integration at peak edge dW (LEGACY — Δ'/δW unreliable): ψ = $((@sprintf "%.3f" odet.psi_store[odet.step])),  q = $((@sprintf "%.3f" odet.q_store[odet.step]))"
            end
        else
            odet.psifac = saved_psifac
            odet.u .= saved_u
            if verbose
                @info "Edge-dW peak (diagnostic): ψ = $((@sprintf "%.3f" odet.psi_store[peak_step])),  q = $((@sprintf "%.3f" odet.q_store[peak_step])); integration domain unchanged"
            end
        end
    end

    # Evaluate stability criterion (critical determinant) of saved solutions
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

Compute axis initial conditions for the Euler-Lagrange ODE via the Frobenius
leading-coefficient eigenvalue problem [Glasser Phys. Plasmas 2016 112506 Eq. 51]:

    lim_{ψ→0} [ψ M(ψ) − a I] v = 0

For each mode j, solves the 2×2 Frobenius eigenvalue problem for the diagonal block of
A₀ = ψ_low · M(ψ_low). The eigenvector with Re(a) ≥ 0 is the regular (non-singular)
Frobenius solution. Returns U₁_init and U₂_init normalized so that U₂_init = I (consistent
with the N independent solutions convention).

For m≠0 the regular eigenvector has a negligible U₁ component (~ψ_low^(|m|/2)), recovering
the Glasser [0, I] limit as ψ_low → 0. For m=0 (degenerate a≈0), the regular eigenvector
is identified by dominant |U₁| component, giving the physically correct constant-displacement
Frobenius solution and avoiding the spurious logarithmic irregularity.
"""
function compute_axis_init(mats::MatrixSplines, profiles::Equilibrium.ProfileSplines,
        intr::ForceFreeStatesInternal, psi_low::Float64)
    N    = intr.numpert_total
    hint = Ref(1)

    # Evaluate stability matrices at psi_low
    F_lower = zeros(ComplexF64, N, N)
    kmat    = zeros(ComplexF64, N, N)
    gmat    = zeros(ComplexF64, N, N)
    mats.ideal.F_spline_lower(vec(F_lower), psi_low; hint=hint)
    mats.ideal.K_spline(vec(kmat),          psi_low; hint=hint)
    mats.ideal.G_spline(vec(gmat),          psi_low; hint=hint)

    # singfac[j] = 1 / (m_j − n_j · q) for each mode j
    q0      = profiles.q_spline(psi_low; hint=hint)
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
        k  = kmat[j, j]
        kd = conj(k)          # K̄†[j,j]
        g  = gmat[j, j]

        # 2×2 ODE matrix block for mode j [Glasser 2016 Eq. 22-24, diagonal approximation]
        m11 = -sf * fi * k
        m12 =  sf^2 * fi
        m21 =  g - kd * fi * k
        m22 =  sf * kd * fi

        # Frobenius matrix A₀_j = ψ_low · M_j [Glasser 2016 Eq. 51]
        #! format: off
        F_eig = eigen([psi_low*m11  psi_low*m12;
                       psi_low*m21  psi_low*m22])
        #! format: on
        eig_vals = F_eig.values
        eig_vecs = F_eig.vectors

        # Select the regular eigenvector: larger Re(a) for m≠0.
        # For degenerate a≈0 (m=0): prefer dominant |U₁| component (regular = constant solution).
        r1 = real(eig_vals[1])
        r2 = real(eig_vals[2])
        i_reg = if abs(r1 - r2) > Base.sqrt(Base.eps(Float64))
            r1 > r2 ? 1 : 2
        else
            abs(eig_vecs[1, 1]) >= abs(eig_vecs[2, 1]) ? 1 : 2
        end

        v1, v2 = eig_vecs[1, i_reg], eig_vecs[2, i_reg]

        # Normalize so that U₂_init[j,j] = 1. If v₂ ≈ 0 (purely displacement solution),
        # set U₁=1, U₂=0 instead.
        if abs(v2) > Base.sqrt(Base.eps(Float64)) * abs(v1)
            U1_init[j, j] = v1 / v2
        else
            U1_init[j, j] =  one(ComplexF64)
            U2_init[j, j] = zero(ComplexF64)
        end
    end

    return U1_init, U2_init
end

"""
    initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, mats::MatrixSplines, profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

Initialize the OdeState struct for the case of sing_start = 0 (axis initialization).
Formerly `ode_axis_init!`. This now only initializes `psifac`, `ising_start`, and `u`.

### TODOs

Move ising_start logic to chunk_el_integration_bounds?
"""
function initialize_el_at_axis!(odet::OdeState, ctrl::ForceFreeStatesControl, mats::MatrixSplines,
        profiles::Equilibrium.ProfileSplines, intr::ForceFreeStatesInternal)

    # Default psifac to minimum equilibrium psi value
    odet.psifac = profiles.xs[1]

    # Use Newton iteration to find starting psi if qlow is above q0
    if ctrl.qlow > profiles.q_spline.y[1]
        # Find last index where q < qlow
        idx = findlast(jpsi -> profiles.q_spline.y[jpsi-1] < ctrl.qlow, 2:profiles.npts)
        if idx !== nothing
            odet.psifac = profiles.xs[idx]
        end
        odet.psifac = find_zero(
            (psi -> profiles.q_spline(psi) - ctrl.qlow,
                psi -> profiles.q_deriv(psi)),
            odet.psifac, Roots.Newton()
        )
    end

    # Find starting singular surface (where sing.psifac > psi(qlow/q0))
    # Note: This logic is kept in initialize_el_at_axis! rather than chunk_el_integration_bounds
    # because it depends on the starting psifac which is set here. The logic for sing_start != 0
    # and kinetic mode would also live here when implemented.
    if ctrl.kinetic_factor > 0
        # Use kinetic singular surfaces (kinsing) for crossing points
        odet.ising_start = searchsortedfirst(getfield.(intr.kinsing, :psifac), odet.psifac) - 1
    else
        odet.ising_start = searchsortedfirst(getfield.(intr.sing, :psifac), odet.psifac) - 1
    end

    if ctrl.fixed_axis
        # Original Glasser initialization: U₁=0, U₂=I [Glasser 2016 §VI].
        # Constrains the axis displacement ξ^ψ=0 for all modes (fixed magnetic axis).
        # Retained as a reference/comparison option; the default (fixed_axis=false) is Frobenius.
        for ipert in 1:intr.numpert_total
            odet.u[ipert, ipert, 2] = 1
        end
    else
        # Frobenius initialization [Glasser 2016 §VI Eq. 51]: selects the regular
        # (non-logarithmic) solution for each mode, including the correct constant
        # displacement solution for the degenerate m=0 case (free magnetic axis).
        U1_init, U2_init = compute_axis_init(mats, profiles, intr, odet.psifac)
        odet.u[:, :, 1] .= U1_init
        odet.u[:, :, 2] .= U2_init
    end
end

# TODO: NOT IMPLEMENTED YET! (low priority, just make sure sing_start = 0 in toml)
function initialize_el_at_singular_surf()
    return
end

"""
    chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal)

Pre-compute all integration chunks from the current position to the edge.
Returns a vector of `IntegrationChunk` objects, each representing a region to integrate
and whether it needs a rational surface crossing beforehand.

This function replaces the iterative while-loop logic with a single upfront computation,
making the integration flow more predictable and easier to parallelize (e.g., for STRIDE).

### Arguments

  - `odet::OdeState` - ODE state struct (starting position and singular surface index)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `intr::ForceFreeStatesInternal` - Internal data (singular surfaces, limits)

### Returns

  - `Vector{IntegrationChunk}` - Array of integration chunks to process
"""
function chunk_el_integration_bounds(odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal; bidirectional::Bool=false)
    chunks = IntegrationChunk[]

    # Start from current position
    psi_current = odet.psifac
    ising_current = odet.ising_start

    # Wrapper to find next singular surface to integrate toward that is resonant within integration limits
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

    # Wrapper to find next kinetic singular surface within integration limits
    # Mirrors Fortran ode.f:185-191 filter: skip kinsing surfaces beyond psilim
    # or whose resonant mode falls outside the truncation range [mlow, mhigh]
    function find_next_kinsing!(ising::Int, intr::ForceFreeStatesInternal)
        ising += 1
        while ising <= intr.kmsing
            if intr.psilim < intr.kinsing[ising].psifac
                break
            end
            # Check resonance: n*q should fall within [mlow, mhigh]
            nq = intr.kinsing[ising].q * minimum(intr.kinsing[ising].n)
            if intr.mlow <= nq && nq <= intr.mhigh
                break
            end
            ising += 1
        end
        return ising
    end

    # -------------------- Create chunks ------------------------
    if ctrl.kinetic_factor > 0 && intr.kmsing > 0 && ctrl.singfac_min > 0
        # Kinetic mode with kinsing surfaces: chunk around each kinetically-displaced
        # singular surface, mirroring Fortran ode.f:184-201 (kin_flag path).
        # The ODE's F̄⁻¹ blows up at these locations; the trapezoidal crossing in
        # cross_kinetic_singular_surf! steps over each singularity.
        ising_current = find_next_kinsing!(ising_current, intr)
        while ising_current <= intr.kmsing && intr.psilim >= intr.kinsing[ising_current].psifac && ctrl.singfac_min != 0
            # Set integration limit to just before the next kinsing surface
            # Fortran: psimax = kinsing(ising)%psifac - singfac_min / |nn * kinsing(ising)%q1|
            psi_end = intr.kinsing[ising_current].psifac -
                      ctrl.singfac_min / abs(minimum(intr.kinsing[ising_current].n) * intr.kinsing[ising_current].q1)

            if psi_current >= psi_end
                # Surface too close to current position — skip it
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
    elseif ctrl.kinetic_factor > 0
        # Kinetic mode with no kinsing surfaces (or singfac_min==0): single chunk.
        # Kinetic contributions are weak enough that F̄ stays well-conditioned.
        push!(chunks, IntegrationChunk(;
            psi_start=psi_current,
            psi_end=(intr.psilim * (1 - eps)),
            needs_crossing=false,
            ising=0
        ))
    else
        # Loop through singular surfaces to cross until edge is reached
        ising_current = find_next_resonant_surface!(ising_current, intr)
        while ising_current <= intr.msing && intr.psilim >= intr.sing[ising_current].psifac && ctrl.singfac_min != 0
            # Set integration limit to just before the next singular surface
            psi_end = intr.sing[ising_current].psifac - ctrl.singfac_min /
                                                        abs(minimum(intr.sing[ising_current].n) * intr.sing[ising_current].q1)

            # Validate chunk bounds
            @assert psi_current < psi_end "Invalid chunk bounds: psi_start=$psi_current >= psi_end=$psi_end"
            @assert isempty(chunks) || psi_current >= chunks[end].psi_end "Overlapping chunks detected"

            push!(chunks, IntegrationChunk(;
                psi_start=psi_current,
                psi_end=psi_end,
                needs_crossing=true,
                ising=ising_current,
                direction = bidirectional ? -1 : 1
            ))

            # After crossing, we jump to the other side of the singular surface
            dpsi = intr.sing[ising_current].psifac - psi_end
            psi_current = psi_end + 2 * dpsi

            # Move to next singular surface that is either resonant or beyond integration limits
            ising_current = find_next_resonant_surface!(ising_current, intr)
        end

        # No more singular surfaces to cross, set integration limit to edge
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

Handle the crossing of a rational surface during integration if kinetic mode is disabled.
Formerly `ode_ideal_cross!`. Performs the same function as `ode_ideal_cross` in the Fortran code.
Differences mainly in integration data storage logic, but otherwise identical. It normalizes and
reinitializes the solution vector at the singularity, and updates relevant state variables.

Asymptotics are now computed on-demand here instead of being pre-computed, making it clear
that asymptotic calculations are specific to ideal ForceFreeStates and not inherent to the singular surface.

### Arguments

  - `ising::Int` - Index of the singular surface being crossed
"""
function cross_ideal_singular_surf!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    ising::Int
)

    # Fixup solution at singular surface
    compute_solution_norms!(odet.u, odet, ctrl, intr, true)

    # Compute direction-specific asymptotic power series for this singular surface
    singp = intr.sing[ising]
    sing_asymp_right = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=1.0)
    sing_asymp_left = compute_sing_asymptotics(singp, ctrl, equil, mats, intr; sig=-1.0, alpha_override=sing_asymp_right.alpha)
    dpsi = singp.psifac - odet.psifac # ψ_res - ψ (positive)

    # Get asymptotic coefficients before crossing (left side)
    ua = sing_get_ua(sing_asymp_left, dpsi)
    odet.ca_l[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)

    # Single n: remove largest solution and sub in asymptotics on the other side
    # Multi-n: if we remove the N largest modes in arbitrary order, we can mess up the
    # diagonal structure of the matrix and later calculations. zeroed_idx let's us make sure
    # the solution vector we're zeroing corresponds to the same block as the resonant mode we
    # introduce. It is also needed when transforming u back to the full solution after integration.
    ipert_res = 1 .+ singp.m .- intr.mlow .+ (singp.n .- intr.nlow) .* intr.mpert
    if ctrl.kinetic_factor == 0
        # Eliminate the solution with the largest norm (in the same block) for each resonance
        odet.zeroed_idx[odet.ifix] = Int[]
        for i in eachindex(sing_asymp_right.r1)
            push!(odet.zeroed_idx[odet.ifix], findfirst(j -> (ipert_res[i] - 1) ÷ intr.mpert == (odet.index[j, odet.ifix] - 1) ÷ intr.mpert, 1:intr.numpert_total))
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= 0
        end
    end

    # Re-initialize on opposite side of rational surface by approximating solution
    params = (ctrl, equil, mats, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac += 2 * dpsi # jump to other side of singular surface
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Apply asymptotic solution on other side of singular surface (right side)
    ua = sing_get_ua(sing_asymp_right, dpsi)
    if ctrl.kinetic_factor == 0
        for i in eachindex(sing_asymp_right.r1)
            # Zero out the resonant components
            odet.u[ipert_res[i], :, :] .= 0
            # Introduce the small asymptotic resonant solution on the other side of the singular surface
            odet.u[:, odet.index[odet.zeroed_idx[odet.ifix][i], odet.ifix], :] .= ua[:, ipert_res[i]+intr.numpert_total, :]
        end
    end
    # Get asymptotic coefficients after crossing rational surface
    odet.ca_r[:, :, :, ising] .= sing_get_ca(odet.u, ua, intr)
    odet.ca_populated = true

    # Δ' is NOT computed for the standard path. The physical Δ' requires the solution
    # columns to be in the Riccati gauge (U₂=I), maintained only by Riccati renormalization.
    # The standard path's solution columns grow from the axis with an arbitrary complex
    # phase; dividing by the outer asymptotic coefficient normalizes magnitude but not phase,
    # so the result is in a different convention. The canonical Δ' is the STRIDE BVP matrix
    # (compute_delta_prime_matrix!) populated by the parallel FM path.

    # Store values after crossing step and advance
    odet.q = equil.profiles.q_spline(odet.psifac; hint=odet.spline_hint)
    store_ode_data!(odet, odet.psifac, odet.u)
end

"""
    cross_kinetic_singular_surf!(odet, ctrl, equil, mats, intr, ising)

Cross a kinetically-displaced singular surface using a simple trapezoidal step.
Matches Fortran `ode_kin_cross` with `con_flag=true` (`ode.f:615-619`): evaluate
the ODE RHS on both sides of the singularity and take a trapezoidal Euler step
across. No asymptotic analysis, no Gaussian elimination — the kinetic FKG
formulation absorbs the ideal singularity and the trapezoidal step handles
the residual near-singularity.

Much simpler than `cross_ideal_singular_surf!` which requires asymptotic
power series and solution vector surgery.
"""
function cross_kinetic_singular_surf!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    ising::Int
)
    # Normalize solution at singular surface [Fortran: ode_unorm(.TRUE.)]
    compute_solution_norms!(odet.u, odet, ctrl, intr, true)

    # Trapezoidal step across the kinsing surface [Fortran ode.f:616-619, con_flag=true]
    ksurf = intr.kinsing[ising]
    dpsi = ksurf.psifac - odet.psifac

    params = (ctrl, equil, mats, intr, odet, IntegrationChunk(0.0, 0.0, false, ising, 1))
    du1 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
    du2 = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)

    sing_der!(du1, odet.u, params, odet.psifac)
    odet.psifac = ksurf.psifac + dpsi  # symmetric jump to other side
    sing_der!(du2, odet.u, params, odet.psifac)
    odet.u .+= (du1 .+ du2) .* dpsi

    # Store crossing step
    odet.q = equil.profiles.q_spline(odet.psifac; hint=odet.spline_hint)
    store_ode_data!(odet, odet.psifac, odet.u)
end


"""
    integrate_el_region!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal, chunk::IntegrationChunk)

Integrate the Euler-Lagrange equations from `psi_start` to `psi_end`.
Formerly `ode_step!`. Performs the same function as `ode_step` in the Fortran code, with the addition of
a callback function to handle tolerances, normalization, and storage at each
step of the integration. In Fortran, this was performed by running LSODE in one-step
mode (so ode_step was called hundreds of times) and calling the relevant functions in
a DO loop. Here, we use the DifferentialEquations.jl interface to achieve the same
functionality in a more Julian way. The integration bounds are now explicit arguments,
making it clear what region is being integrated.

### Arguments

  - `odet::OdeState` - ODE state struct (modified in-place)
  - `ctrl::ForceFreeStatesControl` - Control parameters
  - `equil::Equilibrium.PlasmaEquilibrium` - Plasma equilibrium
  - `mats::MatrixSplines` - Fourier fit variables
  - `intr::ForceFreeStatesInternal` - Internal data
  - `chunk::IntegrationChunk` - Integration chunk containing start and end ψ for integration

### TODOs

Check sensitivity of results to tolerances, currently using same logic as Fortran
Check absolute tolerances, currently only relative tolerances are updated
"""
function integrate_el_region!(
    odet::OdeState,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium,
    mats::MatrixSplines,
    intr::ForceFreeStatesInternal,
    chunk::IntegrationChunk
)

    # Fraction of the q-range defining "near boundary" dense-save zones at each end of
    # a segment. TODO: expose as a ctrl field when a good default is validated.
    near_q_frac = 0.05

    # q at segment boundaries — used for symmetric near-boundary heuristic.
    # odet.q is updated at every step inside sing_der!, so we compare against these
    # fixed endpoints in the callback rather than using psi-based distances.
    q_start = equil.profiles.q_spline(chunk.psi_start)
    q_end = equil.profiles.q_spline(chunk.psi_end)
    q_range = abs(q_end - q_start)

    steps_in_segment = Ref(0)

    function segment_callback!(integrator)
        ctrl, _, _, intr, odet, chunk = integrator.p

        odet.total_steps += 1
        steps_in_segment[] += 1

        compute_solution_norms!(integrator.u, odet, ctrl, intr, false)

        # Save near segment boundaries (symmetric, in q not psi) and every Nth step.
        # The step-count fallback (== 1) guarantees the first step is always saved
        # even for near-degenerate segments where q_range ≈ 0.
        near_start = abs(odet.q - q_start) < near_q_frac * q_range || steps_in_segment[] == 1
        near_end = abs(odet.q - q_end) < near_q_frac * q_range
        # Always save in the edge scan region so findmax_dW_edge! has dense q coverage.
        in_edge_scan = ctrl.psiedge < intr.psilim && integrator.t >= ctrl.psiedge

        if near_start || near_end || (odet.total_steps % ctrl.save_interval == 0) || in_edge_scan
            # q at the accepted point, not the last internal Runge-Kutta stage
            odet.q = equil.profiles.q_spline(integrator.t; hint=odet.spline_hint)
            store_ode_data!(odet, integrator.t, integrator.u)
        end
    end

    cb = DiscreteCallback((u, t, integrator) -> true, segment_callback!)
    prob = ODEProblem(sing_der!, odet.u, (chunk.psi_start, chunk.psi_end), (ctrl, equil, mats, intr, odet, chunk))
    sol = solve(prob, Vern9(); reltol=ctrl.eulerlagrange_tolerance, callback=cb, save_everystep=false, save_end=true)

    # Unconditionally save the final step if the callback did not already capture it.
    # Guarantees the pre-crossing (or pre-edge) state is always stored in u_store,
    # regardless of where the last accepted step landed relative to the near_end band.
    if odet.step == 1 || odet.psi_store[odet.step-1] != sol.t[end]
        odet.q = equil.profiles.q_spline(sol.t[end]; hint=odet.spline_hint)
        store_ode_data!(odet, sol.t[end], sol.u[end])
    end

    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
    compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

Computes norms of the solution vectors of the array `u` and normalizes them
if this is not the first call after a fixup. Formerly `ode_unorm!`.
Throws an error if any vector norm is zero. It then compares the variation in norms
relative to initial values after a fixup, and applies the Gaussian reduction via
`apply_gaussian_reduction!` if the variation exceeds `ctrl.ucrit` or if `sing_flag` is true.
Performs the same function as `ode_unorm` in the Fortran code, with minor differences in indexing
and array handling.

Note that we pass `u` as an argument to reduce the number of copies that are
necessary in the integration callback, since otherwise we'd need to copy from
the integrator state to `odet.u` and then back again. This way, we can just
operate on `u` directly without extra copies.

### Arguments

  - u: Current solution vector array, updated in-place if fixfac is called
  - sing_flag: Indicates if normalization is occuring at a singular surface or not

### TODOs

Add resizing logic for unorm arrays when ifix exceeds allocated size
"""
function compute_solution_norms!(u::Array{ComplexF64,3}, odet::OdeState, ctrl::ForceFreeStatesControl, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Compute norms of first solution vectors, abort if any are zero
    odet.unorm .= norm.(eachcol(u[:, :, 1]))
    if minimum(odet.unorm) == 0
        jmax = argmin(odet.unorm)
        error("One of the first solution vector norms unorm(1,$jmax) = 0")
    end

    # Normalize unorm and perform Gaussian reduction if required
    if odet.new
        odet.new = false
        odet.unorm0 .= odet.unorm
    else
        odet.unorm ./= odet.unorm0
        uratio = maximum(odet.unorm) / minimum(odet.unorm)
        if uratio > ctrl.ucrit || sing_flag
            # TODO: add resizing logic here as well
            if odet.ifix < ctrl.numunorms_init
                odet.ifix += 1
            else
                @warn "unorm storage reached, no longer saving fixfac data. Stability outputs and unorming will be correct, but cannot reconstruct `u`. \n
                Increase `numunorms_init` if needed. Automatic resizing will be added in a future version."
            end
            apply_gaussian_reduction!(u, odet, intr, sing_flag)
            odet.new = true
        end
    end
end

"""
    apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

Applies Gaussian reduction to orthogonalize solution vectors in `u`.
Formerly `ode_fixup!`. Performs the same function as `ode_fixup` in the Fortran code,
except now relevant `fixfac` data are stored in memory instead of dumped to `euler.bin`.
Used when the spread in norms exceeds a threshold or when a rational surface is reached.
This will update both `u` and relevant fields in `odet` in-place. See the
description of `compute_solution_norms!` for more details on the benefits of in-place `u`
updates.
"""
function apply_gaussian_reduction!(u::Array{ComplexF64,3}, odet::OdeState, intr::ForceFreeStatesInternal, sing_flag::Bool)

    # Store data for the current fixup
    ifix = odet.ifix
    odet.sing_flag[ifix] = sing_flag
    # Since the current step has been fixed-up, we denote the end of the previous
    # fixup region as the previous step (this just avoids a -1 index later)
    odet.fixstep[ifix] = odet.step - 1

    # Initialize fixfac
    for isol in 1:intr.numpert_total
        odet.fixfac[isol, isol, ifix] = 1
    end
    # Sort unorm in descending order (since we triangularize from largest to smallest)
    odet.index[:, ifix] = sortperm(odet.unorm; rev=true)

    # Triangularize primary solutions
    mask = trues(2, intr.numpert_total)
    for isol in 1:intr.numpert_total
        ksol = odet.index[isol, ifix]
        mask[2, ksol] = false
        # Set pivot row based on max location
        @views kpert = argmax(abs.(u[:, ksol, 1]) .* mask[1, :])
        mask[1, kpert] = false
        # Eliminate other solution vectors below the pivot
        for jsol in 1:intr.numpert_total
            if mask[2, jsol]
                odet.fixfac[ksol, jsol, ifix] = -u[kpert, jsol, 1] / u[kpert, ksol, 1]
                @. @views u[:, jsol, :] .= u[:, jsol, :] .+ u[:, ksol, :] .* odet.fixfac[ksol, jsol, ifix]
                u[kpert, jsol, 1] = 0
            end
        end
    end
end

"""
    findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

Records the total dW in the integration region between `ctrl.psiedge` and
`ctrl.psilim`. This performs the same function as `ode_record_edge` in the
Fortran, but everything is now done post-integration which cleans up the logic,
i.e. no "_edge" arrays.

The dW is stored at that step index in `odet.dW_edge`; because we initialize
dW_edge to -Inf, we can just take the max value after integration to get the
total dW at the edge and avoid unphysical kink modes that might occur just
inside rational surfaces.

We have also separated the computation of the wv matrix spline and the total dW
calculation into `free_compute_wv_spline` and `free_compute_total` respectively
for clarity. We create the wv matrix spline once prior to the loop.
"""
function findmax_dW_edge!(odet::OdeState, ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium, mats::MatrixSplines, intr::ForceFreeStatesInternal)

    # Find the first ODE step at or past psiedge; all subsequent steps are contiguous edge steps
    edge_start = findfirst(i -> odet.psi_store[i] >= ctrl.psiedge, 1:odet.step)
    N_edge = odet.step - edge_start + 1

    # Initialize EdgeScanState sized exactly to the number of edge steps
    odet.edge_scan = EdgeScanState(intr.numpert_total, N_edge)
    es = odet.edge_scan

    es.psi .= odet.psi_store[edge_start:odet.step]
    es.q .= odet.q_store[edge_start:odet.step]

    # Create a rough spline for wv matrix between psiedge -> psilim so we can approximate dW
    es.wvmat = free_compute_wv_spline(ctrl, equil, intr)

    # Loop with compact index j into EdgeScanState; ODE index is edge_start + j - 1.
    # Steps where free_compute_total hits a singular wp solve are left as NaN per the EdgeScanState contract.
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

    # Return the ODE step index at peak total_eigenvalue (NaN-safe; failed steps ignored)
    peak_j = argmax(j -> isnan(real(es.total_eigenvalue[j])) ? typemin(Float64) : real(es.total_eigenvalue[j]), 1:N_edge)
    return edge_start + peak_j - 1
end

"""
    transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

Constructs the transformation matrices to form the true solution vectors. Effectively
"undoes" the Gaussian reduction applied during fixups throughout the integration, such that
we have the true solution vectors for use in GPEC. Modifies the store arrays in `odet` in-place.
Performs a similar function as `idcon_transform` + `idcon_build` in the Fortran code,
except we separate the building of the transformation matrices and determining the coefficients
for a chosen force-free solution, which can be done in postprocessing.
"""
function transform_u!(odet::OdeState, intr::ForceFreeStatesInternal)

    # Gaussian reduction matrices for each fixup
    gauss = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix)
    # Transformation matrices for each region between fixups (ifix + 1 regions)
    transforms = Array{ComplexF64,3}(undef, intr.numpert_total, intr.numpert_total, odet.ifix + 1)
    # Temporary workspace matrices
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
        # Account for zeroed indices at singular surfaces in `ode_ideal_cross`
        if odet.sing_flag[ifix]
            for zeroed_idx in odet.zeroed_idx[ifix]
                gauss[:, odet.index[zeroed_idx, ifix], ifix] .= 0.0
            end
        end
    end

    # Concatenate gaussian reduction matrix to form transform matrix for each region
    # Here, the i'th region is between the (i-1)'th and i'th fixup e.g. transforms[:, :, 1]
    # is the transform matrix for the region between init and first fixup
    # and mfix + 1 is the for the region after the last fixup and before the edge
    transforms[:, :, end] .= identity
    for ifix in odet.ifix:-1:1
        mul!(view(transforms,:,:,ifix), view(gauss,:,:,ifix), view(transforms,:,:,(ifix+1)))
    end

    # Now that we have the transform matrices, we can apply them to the solution vectors
    # "undoing" the Gaussian reductions to get the true solution vectors
    jfix = 1
    for ifix in 1:(odet.ifix+1)
        # If after the last fixup, go to the end of integration.
        # Cap kfix at odet.step: fixstep entries from fixups AFTER the peak (set during integration
        # before trim_storage!) can exceed the trimmed storage size and must be clamped.
        kfix = ifix != odet.ifix + 1 ? min(odet.fixstep[ifix], odet.step) : odet.step
        jfix > odet.step && break
        @views for istep in jfix:kfix
            # This is u1->u4 in Fortran
            mul!(gauss_buffer, odet.u_store[:, :, 1, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 1, istep] .= gauss_buffer
            mul!(gauss_buffer, odet.u_store[:, :, 2, istep], transforms[:, :, ifix])
            odet.u_store[:, :, 2, istep] .= gauss_buffer
            # Derivative stores are empty unless a path filled them analytically (galerkin);
            # materialized ones are computed after this transform and need no fixup.
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

Evaluate the derivative of the Euler-Lagrange equations [Glasser Phys. Plasmas 2016 112506 eq. 24].
This implements du/dψ for both the ideal and kinetic MHD eigenvalue problems.

This function performs the same role as `sing_der` in the Fortran code, with main differences
coming from hiding LAPACK operations under the hood via Julia's LinearAlgebra package,
so the code is much more straightforward.

This follows the Julia DifferentialEquations package format for in place updating.

    ode_function!(du, u, p, t)

From DifferentialEquations.jl docs: Defining your ODE function to be in-place updating
can have performance benefits. What this means is that, instead of writing a function
which outputs its solution, you write a function which updates a vector that is designated
to hold the solution. By doing this, DifferentialEquations.jl's solver packages are able
to reduce the amount of array allocations and achieve better performance.

Wherever possible, in-place operations on pre-allocated arrays are used to minimize memory allocations.
All LAPACK operations are handled under the hood by Julia's LinearAlgebra package, so we can obtain a much
more simplistic code with similar performance.

### Arguments

  - `du::Array{ComplexF64,3}`: Pre-allocated array to hold the derivative result, shape (mpert, mpert, 2), updated in-place
  - `u::Array{ComplexF64,3}`: Current state array, shape (mpert, mpert, 2)
  - `params::Tuple{ForceFreeStatesControl, PlasmaEquilibrium, MatrixSplines, ForceFreeStatesInternal, OdeState, IntegrationChunk}`: Tuple of relevant structs
  - `psieval::Float64`: Current psi value at which to evaluate the derivative

The unpacked-argument method carries the arithmetic; this tuple method is the thin adapter the
integrator calls. Ξ_s is *not* computed here — it is a save-point quantity, obtained from
[`compute_node_xi_s!`](@ref) only where it is actually consumed.
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
    odet.q = el_derivatives!(du, u, ctrl.kinetic_factor > 0, equil, mats, intr, psieval, odet.spline_hint, odet.mats_hint)
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

    # Allocate temporary arrays from the pool
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

    # Compute singfac = 1 / (m - nq)
    # Use caller-supplied hint for O(1) interval lookup during sequential ODE integration
    q = equil.profiles.q_spline(psieval; hint=spline_hint)
    singfac_mat .= 1.0 ./ ((intr.mlow:intr.mhigh) .- q .* (intr.nlow:intr.nhigh)')

    kin = mats.kinetic
    if kinetic
        kin === nothing && error("el_derivatives! called with kinetic=true but mats carries no kinetic matrices")
        # ---- Kinetic path with pre-computed FKG matrices ----
        # Use the caller's hint, not mats._hint (shared, racy in the parallel BVP)
        # Load FKG sub-matrices (note: reusing fmat_lower/kmat/gmat as workspace)
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

        # Build F, K, K† with singfac (using fmat_lower, kmat, gmat as workspace for F, K, K†)
        kaat_kin = similar!(pool, fmat_lower)  # K† matrix
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
        # gmat = gaat (already loaded)
        gmat .= gaat_kin

        # Kinetic ODE (Logan 2015 Eq 7.46): singfac absorbed into F̄/K̄/K̄†, no explicit Q⁻¹
        # du₁ = F̄⁻¹(u₂ - K̄·u₁)
        du1 .= u2
        mul!(tmp_mat, kmat, u1)
        du1 .-= tmp_mat
        # LU factorize F (non-Hermitian, non-symmetric); direct LAPACK for the same hot-loop reason
        _, ipiv2, _ = LAPACK.getrf!(fmat_lower)
        LAPACK.getrs!('N', fmat_lower, ipiv2, du1)

        # du₂ = Ḡ†·u₁ + K̄†·du₁  (Logan 2015 Eq C.10-C.11)
        mul!(tmp_mat, gmat, u1)
        du2 .= tmp_mat
        mul!(tmp_mat, kaat_kin, du1)
        du2 .+= tmp_mat

    else
        # ---- Ideal path ----
        # Evaluate matrix splines at the current psi (hint is the caller's, never shared)
        mats.ideal.F_spline_lower(vec(fmat_lower), psieval; hint=mats_hint)
        mats.ideal.K_spline(vec(kmat), psieval; hint=mats_hint)
        mats.ideal.G_spline(vec(gmat), psieval; hint=mats_hint)

        # See equations 22-24 in Glasser 2016 DCON paper for derivation
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

Evaluate Ξ_s = -A⁻¹(B·Ξ′_ψ + C·Ξ_ψ) [Glasser Phys. Plasmas 2016 112506 eq. 18] at `psieval`,
writing into `xi_s`. `du1` and `u1` are the Ξ′_ψ and Ξ_ψ blocks at the same ψ, i.e. slices of a
`el_derivatives!` result and its input state.

Split out of the derivative kernel because Ξ_s is needed only at saved nodes, not at every
Runge-Kutta stage. Ideal runs factor the Hermitian A by Cholesky; with `kinetic=true` A picks up
non-Hermitian contributions and needs an LU.
"""
@with_pool pool function compute_node_xi_s!(xi_s::AbstractMatrix{ComplexF64}, du1::AbstractMatrix{ComplexF64},
    u1::AbstractMatrix{ComplexF64}, mats::MatrixSplines, psieval::Float64; kinetic::Bool=false, hint::Base.RefValue{Int}=Ref(1))

    Npert = size(u1, 1)
    amat = acquire!(pool, ComplexF64, Npert, Npert)
    bmat = similar!(pool, amat)
    cmat = similar!(pool, amat)
    tmp_mat = similar!(pool, amat)

    # A/B/C of the active model: the kinetic A is non-Hermitian, hence the factorization split below
    active_mats = kinetic ? mats.kinetic : mats.ideal
    active_mats === nothing && error("compute_node_xi_s! called with kinetic=true but mats carries no kinetic matrices")
    active_mats.A_spline(vec(amat), psieval; hint=hint)
    active_mats.B_spline(vec(bmat), psieval; hint=hint)
    active_mats.C_spline(vec(cmat), psieval; hint=hint)

    # Solve bmat = A⁻¹ * bmat, cmat = A⁻¹ * cmat in-place
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

