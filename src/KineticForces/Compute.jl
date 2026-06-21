"""
    Compute

High-level computation functions for KineticForces.
Orchestrates torque/energy calculations across multiple methods
using adaptive Gauss-Kronrod quadrature at all levels (ψ, λ, energy).
"""

# ============================================================================
# Adaptive Gauss-Kronrod ψ integration via QuadGK.BatchIntegrand
# ============================================================================

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
        return (total=ComplexF64(0.0), torque_profile=nothing, matrix_integrated=nothing, psi_nsteps=0)
    end

    # Buffers for the batch callback. The outer ψ-integral is intentionally
    # serial: QuadGK.BatchIntegrand's refine loop invokes the callback many
    # times with small batches (~15 nodes per Kronrod rule), and Threads.@threads
    # fork-join overhead at this granularity dominates the ~40 ms per-ψ work,
    # producing catastrophic slowdowns at ≥2 threads. Serial `for` inside the
    # batch matches 1-thread wall time and is the fastest correct option found.
    tpsi_val = Ref{ComplexF64}(0.0im)
    wtw_l = is_matrix_method ? zeros(ComplexF64, mpert, mpert, 6) : nothing

    logged_psi = Float64[]
    logged_dtdpsi = ComplexF64[]
    logged_elems = is_matrix_method ? Vector{Array{ComplexF64,3}}() : nothing

    function psi_batch!(y::AbstractVector{ComplexF64}, x::AbstractVector)
        for k in eachindex(x)
            psi = Float64(x[k])

            if is_matrix_method && !isnothing(wtw_l)
                wtw_l .= 0
            end
            elems_accum = is_matrix_method ? zeros(ComplexF64, mpert, mpert, 6) : nothing

            total = ComplexF64(0.0)
            for ell_idx in 1:(1 + 2 * nl)
                l = ell_idx - 1 - nl
                if is_matrix_method && !isnothing(wtw_l)
                    wtw_l .= 0
                end

                tpsi!(tpsi_val, psi, n, l, zi, mi, wdfac, divxfac,
                      electron, method, equil, intr, kinetic_profiles;
                      op_wmats=wtw_l,
                      atol_xlmda=ctrl.atol_xlmda, rtol_xlmda=ctrl.rtol_xlmda)
                total += tpsi_val[]

                if is_matrix_method && !isnothing(wtw_l) && !isnothing(elems_accum)
                    elems_accum .+= wtw_l
                end
            end

            y[k] = total

            push!(logged_psi, psi)
            push!(logged_dtdpsi, total)
            if is_matrix_method && !isnothing(elems_accum)
                push!(logged_elems, copy(elems_accum))
            end
        end
    end

    bi = QuadGK.BatchIntegrand(psi_batch!, ComplexF64[], Float64[])
    total, _ = quadgk(bi, x0, xout; atol=ctrl.atol_psi, rtol=ctrl.rtol_psi)

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

    return (total=total, torque_profile=torque_profile, matrix_integrated=matrix_integrated, psi_nsteps=npts)
end


# ============================================================================
# High-level orchestration
# ============================================================================

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

            if n_idx == 1 && !isnothing(result.torque_profile)
                psi_grid_out = result.torque_profile.psi
                dtdpsi_out = result.torque_profile.dtdpsi
                t_cum_out = result.torque_profile.t_cumulative
            end

            # Insert n-block into full matrix
            if is_matrix_method && !isnothing(result.matrix_integrated) && !isnothing(op_wmats_full)
                i_start = (n_idx - 1) * intr.mpert + 1
                i_end = n_idx * intr.mpert
                op_wmats_full[i_start:i_end, i_start:i_end, :] .= result.matrix_integrated
            end
        end

        # Eq. (19) Logan et al. PoP 20, 122507 (2013): Im(T) = 2n·δW_k, both real quantities.
        # Store δW in Re slot so downstream code uses real(total_energy).
        total_energy = complex(imag(total_torque) / (2 * ctrl.nn), 0.0)

        result_entry = MethodResult(;
            method=method,
            nn=ctrl.nn,
            total_torque=total_torque,
            total_energy=total_energy,
            psi_grid=psi_grid_out,
            dtdpsi=dtdpsi_out,
            t_cumulative=t_cum_out,
            psi_nsteps=psi_nsteps_total,
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


