"""
    Compute

High-level computation functions for KineticForces.
Orchestrates torque/energy calculations across multiple methods
using adaptive Gauss-Kronrod quadrature at all levels (ψ, λ, energy).
"""

# ============================================================================
# Adaptive Gauss-Kronrod ψ integration via QuadGK.BatchIntegrand
# ============================================================================

# Minimum ψ separation for quadrature panel boundaries: interior points closer than this to
# each other or to a domain bound are merged/dropped, so coincident rational and kinetic-
# resonance surfaces do not create degenerate (near-zero-width) Gauss-Kronrod panels.
const PANEL_MERGE_ATOL = 1e-8

"""
    psi_panel_points(interior, x0, xout) → Vector{Float64}

Build the ψ-quadrature node list `[x0, interior points strictly inside (x0, xout), xout]`.
`interior` is the raw union of resonant-surface locations (rational surfaces ∪ kinetic
resonances); this function owns the ordering: sort, drop near-duplicates (closer than
`PANEL_MERGE_ATOL`, e.g. a kinetic resonance coinciding with a rational), and drop points
within `PANEL_MERGE_ATOL` of a bound to avoid degenerate panels. Paneling the integral at
these surfaces puts the resonant torque-density peaks (reg_spot/collisionally broadened, but
narrow in ψ) on Gauss-Kronrod interval endpoints, which the rule handles natively instead of
hunting them by adaptive bisection.
"""
function psi_panel_points(interior::Vector{Float64}, x0::Float64, xout::Float64)
    pts = sort(filter(p -> x0 + PANEL_MERGE_ATOL < p < xout - PANEL_MERGE_ATOL, interior))
    deduped = [p for (i, p) in enumerate(pts) if i == 1 || p - pts[i-1] > PANEL_MERGE_ATOL]
    return [x0; deduped; xout]
end

"""
    check_psi_quadrature_convergence(total, quad_err, ctrl, method)

Warn when the ψ torque quadrature terminated without satisfying the requested tolerances
(hit `maxevals_psi`), or when a nonzero user-set `atol_psi` dominated termination — the
silent-garbage scenario for weak applied fields, since NTV scales as δB².
"""
function check_psi_quadrature_convergence(total::ComplexF64, quad_err::Float64,
                                          ctrl::KineticForcesControl, method::String)
    if quad_err > max(ctrl.atol_psi, ctrl.rtol_psi * abs(total))
        @warn "ψ torque quadrature ($method) did not converge within maxevals_psi=$(ctrl.maxevals_psi): " *
              "error estimate $quad_err vs |T|=$(abs(total)) N·m. Raise maxevals_psi or loosen rtol_psi."
    elseif ctrl.atol_psi > ctrl.rtol_psi * abs(total)
        @warn "atol_psi=$(ctrl.atol_psi) N·m dominated ψ quadrature termination ($method): " *
              "|T|=$(abs(total)) N·m is small relative to atol_psi, so the relative error is uncontrolled. " *
              "NTV scales as δB², so weak applied fields shrink the torque quadratically — recommend atol_psi=0 (rtol-only)."
    end
    return nothing
end

"""
    integrate_psi_quadgk(n, nl, zi, mi, wdfac, divxfac, electron, method,
                          equil, intr, ctrl, kinetic_profiles; psi_min, psi_max) → NamedTuple

Integrate torque over ψ using adaptive Gauss-Kronrod quadrature with
`QuadGK.BatchIntegrand`. Every integrand evaluation is logged, giving a
diagnostic T(ψ) profile at no extra cost (the values are computed anyway
— we just keep them).

# Returns
NamedTuple with:
- `total::ComplexF64`: Total integrated torque
- `torque_profile`: NamedTuple of (psi, dtdpsi, t_cumulative) from evaluation points
- `matrix_integrated`: Trapezoidal-integrated mpert×mpert×6 matrix (if matrix method)
- `psi_nsteps::Int`: Number of integrand evaluations
- `psi_quad_error::Float64`: Quadrature error estimate for the total torque
- `panel_psis::Vector{Float64}`: Quadrature panel boundaries actually used (bounds + interior resonant surfaces)
- `resonance_psis::Vector{Float64}`: Located kinetic-resonance ψ surfaces (Ω_ℓ(x=1)=0), for diagnostics/plotting

The integral is paneled at the rational-surface ψ locations (`intr.sing_psis`) and capped
at `ctrl.maxevals_psi` evaluations; a warning is emitted if the quadrature fails to reach
`ctrl.rtol_psi`/`ctrl.atol_psi` or if a nonzero `atol_psi` dominates termination.
"""
function integrate_psi_quadgk(
    n::Int, nl::Int, zi::Int, mi::Int,
    wdfac::Float64, divxfac::Float64, electron::Bool,
    method::String, equil, intr::KineticForcesInternal, ctrl::KineticForcesControl,
    kinetic_profiles::Equilibrium.KineticProfileSplines;
    psi_min::Float64=0.0, psi_max::Float64=1.0
)
    is_matrix_method = occursin("mm", method)
    mpert = intr.mpert

    # Equilibrium splines are unreliable near the magnetic axis (epsr→0, J→0);
    # start at psilow to avoid degenerate bounce/drift frequencies.
    # Cap at intr.psilim (DCON's integration limit): perturbation interpolants only
    # have data below this, and extrapolation near q=qlim blows up.
    x0 = max(psi_min, equil.config.psilow)
    xout = min(psi_max, intr.psilim, 1.0 - 1e-6)

    if x0 >= xout
        return (total=ComplexF64(0.0), torque_profile=nothing, matrix_integrated=nothing, psi_nsteps=0, psi_quad_error=0.0,
                panel_psis=Float64[], resonance_psis=Float64[])
    end

    # The outer ψ-integral (the QuadGK batch / ψ-node loop) stays serial: QuadGK's refine
    # loop invokes the callback with small batches (~15 nodes), so threading it is
    # fork-join-bound. Instead thread the inner bounce-harmonic loop (2·nl+1 harmonics),
    # which carries the ~40 ms of per-ψ work. `tpsi!` writes per-`intr` scratch buffers and
    # spline hints, so each thread gets its own `deepcopy(intr)` (the same per-thread-scratch
    # pattern the self-consistent kinetic matrix build uses). Per-harmonic results are stored
    # by index and reduced in fixed ℓ order, so the sum is identical to the serial reduction
    # and independent of thread count. The threadid()-indexed buffers require this to be a
    # top-level @threads (the caller loops n serially); never nest it inside another @threads.
    # The deepcopy set is sized by thread count and rebuilt once per call (per n), so multi-n
    # runs with many threads scale memory as nthreads×sizeof(intr).
    nharm = 1 + 2 * nl
    nth = Threads.maxthreadid()
    thread_intrs = [deepcopy(intr) for _ in 1:nth]
    thread_tpsi = [Ref{ComplexF64}(0.0im) for _ in 1:nth]
    thread_wtw = is_matrix_method ? [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nth] : nothing
    harm_vals = Vector{ComplexF64}(undef, nharm)
    harm_elems = is_matrix_method ? [zeros(ComplexF64, mpert, mpert, 6) for _ in 1:nharm] : nothing

    logged_psi = Float64[]
    logged_dtdpsi = ComplexF64[]
    logged_elems = is_matrix_method ? Vector{Array{ComplexF64,3}}() : nothing

    function psi_batch!(y::AbstractVector{ComplexF64}, x::AbstractVector)
        for k in eachindex(x)
            psi = Float64(x[k])

            # Each bounce harmonic is independent; thread over them, each thread using its own
            # `intr` copy and writing its own harm_vals[ell_idx] / harm_elems[ell_idx] slot.
            Threads.@threads :static for ell_idx in 1:nharm
                tid = Threads.threadid()
                l = ell_idx - 1 - nl
                w = is_matrix_method ? thread_wtw[tid] : nothing
                is_matrix_method && fill!(w, 0)
                tpsi!(thread_tpsi[tid], psi, n, l, zi, mi, wdfac, divxfac,
                      electron, method, equil, thread_intrs[tid], kinetic_profiles;
                      op_wmats=w,
                      atol_xlmda=ctrl.atol_xlmda, rtol_xlmda=ctrl.rtol_xlmda)
                harm_vals[ell_idx] = thread_tpsi[tid][]
                is_matrix_method && (harm_elems[ell_idx] .= w)
            end

            # Reduce in fixed ℓ order (bit-identical to the serial sum).
            total = ComplexF64(0.0)
            for ell_idx in 1:nharm
                total += harm_vals[ell_idx]
            end
            y[k] = total

            push!(logged_psi, psi)
            push!(logged_dtdpsi, total)
            if is_matrix_method
                elems_accum = zeros(ComplexF64, mpert, mpert, 6)
                for ell_idx in 1:nharm
                    elems_accum .+= harm_elems[ell_idx]
                end
                push!(logged_elems, elems_accum)
            end
        end
    end

    # Panels at the rational surfaces the run resolved plus the kinetic-resonance
    # surfaces (thermal-energy Ω_ℓ = 0 for ℓ ∈ -nl:nl) — both are torque-density peaks.
    resonance_psis = kinetic_resonance_psi_nodes(kinetic_profiles, equil; n, nl, zi, mi, electron, wdfac)
    pts = psi_panel_points(vcat(intr.sing_psis, resonance_psis), x0, xout)

    bi = QuadGK.BatchIntegrand(psi_batch!, ComplexF64[], Float64[])
    total, quad_err = quadgk(bi, pts...; atol=ctrl.atol_psi, rtol=ctrl.rtol_psi, maxevals=ctrl.maxevals_psi)

    check_psi_quadrature_convergence(total, quad_err, ctrl, method)

    # Sort logs by ψ for the diagnostic torque profile.
    perm = sortperm(logged_psi)
    sorted_psi = logged_psi[perm]
    sorted_dtdpsi = logged_dtdpsi[perm]

    # Cumulative trapezoidal integration for T(ψ) profile
    npts = length(sorted_psi)
    t_cumulative = Vector{ComplexF64}(undef, npts)
    if npts >= 1
        t_cumulative[1] = ComplexF64(0.0)
        for i in 2:npts
            dpsi = sorted_psi[i] - sorted_psi[i-1]
            t_cumulative[i] = t_cumulative[i-1] + 0.5 * (sorted_dtdpsi[i-1] + sorted_dtdpsi[i]) * dpsi
        end
    end

    torque_profile = npts > 1 ? (psi=sorted_psi, dtdpsi=sorted_dtdpsi, t_cumulative=t_cumulative) : nothing

    # Trapezoidal integration of kinetic matrices over ψ (matrix methods only)
    matrix_integrated = nothing
    if is_matrix_method && !isnothing(logged_elems) && length(logged_elems) > 1
        sorted_elems = logged_elems[perm]
        matrix_integrated = zeros(ComplexF64, mpert, mpert, 6)
        for i in 2:npts
            dpsi = sorted_psi[i] - sorted_psi[i-1]
            matrix_integrated .+= 0.5 .* (sorted_elems[i-1] .+ sorted_elems[i]) .* dpsi
        end
    end

    in_span(p) = x0 + PANEL_MERGE_ATOL < p < xout - PANEL_MERGE_ATOL
    n_rational = count(in_span, intr.sing_psis)
    n_resonance = count(in_span, resonance_psis)
    @info "ψ torque quadrature ($method): T=$total N·m, error estimate $quad_err, $npts integrand evaluations over " *
          "$(length(pts) - 1) panels ($n_rational rational + $n_resonance kinetic resonance surfaces)"

    return (total=total, torque_profile=torque_profile, matrix_integrated=matrix_integrated, psi_nsteps=npts, psi_quad_error=quad_err,
            panel_psis=pts, resonance_psis=sort(resonance_psis))
end


# ============================================================================
# High-level orchestration
# ============================================================================

"""
    combine_species_states(states) -> KineticForcesState

Sum per-species [`KineticForcesState`](@ref) results into a single total (τ = Σ_s τ_s). Per
method: the scalar `total_torque`/`total_energy` are summed exactly; the dT/dψ profile is summed
by evaluating each species' `torque_profile` interpolant on the sorted union of the species ψ
grids, and the cumulative T(ψ) is re-integrated (trapezoid) from that summed profile. All species
share the same ψ-integration range (`ctrl.psilims`), so the grids differ only in adaptive nodes.
"""
function combine_species_states(states::AbstractVector{KineticForcesState})
    combined = KineticForcesState()
    isempty(states) && return combined
    method_names = collect(keys(first(states).method_results))
    for mname in method_names
        results = [s.method_results[mname] for s in states if haskey(s.method_results, mname)]
        isempty(results) && continue
        grid = sort(unique(reduce(vcat, (r.psi_grid for r in results); init=Float64[])))
        # Sum each species' dT/dψ onto the union grid by linear interpolation of its own
        # (psi_grid, dtdpsi) arrays (torque_profile interpolants are not populated here); zero
        # outside a species' ψ range.
        dtdpsi = zeros(ComplexF64, length(grid))
        for r in results
            length(r.psi_grid) >= 2 || continue
            o = sortperm(r.psi_grid); xs = r.psi_grid[o]; ys = r.dtdpsi[o]
            for (k, ψ) in enumerate(grid)
                (ψ < xs[1] || ψ > xs[end]) && continue
                j = searchsortedlast(xs, ψ)
                if j == length(xs)
                    dtdpsi[k] += ys[end]
                else
                    t = (ψ - xs[j]) / (xs[j+1] - xs[j])
                    dtdpsi[k] += (1 - t) * ys[j] + t * ys[j+1]
                end
            end
        end
        tcum = zeros(ComplexF64, length(grid))
        for j in 2:length(grid)
            tcum[j] = tcum[j-1] + 0.5 * (dtdpsi[j] + dtdpsi[j-1]) * (grid[j] - grid[j-1])
        end
        combined.method_results[mname] = MethodResult(;
            method=mname, nn=first(results).nn,
            total_torque=sum(r.total_torque for r in results),
            total_energy=sum(r.total_energy for r in results),
            psi_grid=grid, dtdpsi=dtdpsi, t_cumulative=tcum,
            psi_nsteps=sum(r.psi_nsteps for r in results))
    end
    combined.completed = true
    return combined
end

"""
    compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                ctrl::KineticForcesControl, equil, kinetic_profiles)

Calculate torque/energy for all enabled methods.
For each method, integrates over flux surfaces using adaptive QuadGK
quadrature via `integrate_psi_quadgk`.
For multi-n calculations, loops over toroidal mode numbers and assembles
block-diagonal kinetic matrices.

# Arguments
- `state::KineticForcesState`: Accumulates results for all methods
- `intr::KineticForcesInternal`: Internal state with equilibrium data
- `ctrl::KineticForcesControl`: Control parameters specifying which methods to run
- `equil`: PlasmaEquilibrium with 2D interpolants
- `kinetic_profiles::Equilibrium.KineticProfileSplines`: Named kinetic-profile splines
"""
function compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                     ctrl::KineticForcesControl, equil,
                                     kinetic_profiles::Equilibrium.KineticProfileSplines)

    for entry in METHOD_REGISTRY
        getfield(ctrl, entry.flag) || continue

        method = entry.name
        intr.method = method
        is_matrix_method = occursin("mm", method)

        if ctrl.verbose
            println("---------------------------------------------")
            println("$method - $(entry.doc)")
        end

        # Allocate full block-diagonal matrix if needed
        op_wmats_full = nothing
        if is_matrix_method && intr.numpert_total > 0
            op_wmats_full = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 6)
        end

        total_torque = ComplexF64(0.0)
        npert = max(intr.npert, 1)

        # Capture per-ψ profile and step count from the single-n case (or the
        # first n when npert > 1) for output diagnostics.
        psi_grid_out = Float64[]
        dtdpsi_out = ComplexF64[]
        t_cum_out = ComplexF64[]
        psi_nsteps_total = 0
        panel_psis_out = Float64[]
        resonance_psis_out = Float64[]

        for n_idx in 1:npert
            n = intr.nlow + n_idx - 1
            if n == 0
                n = ctrl.nn  # fallback to control parameter for single-n
            end

            # Adaptive QuadGK integration over ψ (serial BatchIntegrand)
            result = integrate_psi_quadgk(
                n, ctrl.nl, ctrl.zi, ctrl.mi,
                ctrl.wdfac, ctrl.divxfac, ctrl.electron,
                method, equil, intr, ctrl, kinetic_profiles;
                psi_min=ctrl.psilims[1], psi_max=ctrl.psilims[2])

            total_torque += result.total
            psi_nsteps_total += result.psi_nsteps

            if n_idx == 1
                panel_psis_out = result.panel_psis
                resonance_psis_out = result.resonance_psis
                if !isnothing(result.torque_profile)
                    psi_grid_out = result.torque_profile.psi
                    dtdpsi_out = result.torque_profile.dtdpsi
                    t_cum_out = result.torque_profile.t_cumulative
                end
            end

            # Insert n-block into full matrix
            if is_matrix_method && !isnothing(result.matrix_integrated) && !isnothing(op_wmats_full)
                i_start = (n_idx - 1) * intr.mpert + 1
                i_end = n_idx * intr.mpert
                op_wmats_full[i_start:i_end, i_start:i_end, :] .= result.matrix_integrated
            end
        end

        # Eq. (19) Logan et al. PoP 20, 122507 (2013): Im(T) = 2n·δW_k. δW_k is real.
        total_energy = imag(total_torque) / (2 * ctrl.nn)

        result_entry = MethodResult(;
            method=method,
            nn=ctrl.nn,
            total_torque=total_torque,
            total_energy=total_energy,
            psi_grid=psi_grid_out,
            dtdpsi=dtdpsi_out,
            t_cumulative=t_cum_out,
            psi_nsteps=psi_nsteps_total,
            panel_psis=panel_psis_out,
            resonance_psis=resonance_psis_out,
        )
        state.method_results[method] = result_entry

        if is_matrix_method && !isnothing(op_wmats_full)
            state.kinetic_matrices[method] = op_wmats_full
        end

        if ctrl.verbose
            @printf("%-24s%11.3e\n", "Total torque = ", real(total_torque))
            @printf("%-24s%11.3e\n", "Total Kinetic Energy = ", imag(total_torque) / (2 * ctrl.nn))
            if imag(total_torque) != 0.0
                @printf("%-24s%11.3e\n", "alpha/s  = ", real(total_torque) / (-1 * imag(total_torque)))
            end
            println("$method - Finished")
            println("---------------------------------------------")
        end
    end

    state.completed = true
end
