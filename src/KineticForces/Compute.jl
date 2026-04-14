"""
    Compute

High-level computation functions for KineticForces.
Orchestrates torque/energy calculations across multiple methods
using dynamic ODE integration at all levels (ψ, λ, energy).
"""

# ============================================================================
# Psi integration params
# ============================================================================

"""
    PsiIntegrationParams

Parameters for the outer ψ ODE integration.
Passed to `psi_integrand!` via the ODE solver's `p` field.
Ports Fortran `tintgrnd` common block variables.
"""
mutable struct PsiIntegrationParams
    n::Int
    nl::Int
    zi::Int
    mi::Int
    wdfac::Float64
    divxfac::Float64
    electron::Bool
    method::String
    equil::Any
    intr::KineticForcesInternal
    ctrl::KineticForcesControl
    kinetic_profiles::Equilibrium.KineticProfileSplines
    # Mutable storage for kinetic matrix accumulation across ℓ harmonics
    elems::Union{Nothing, Array{ComplexF64,3}}
    # Growing storage for (ψ, dT/dψ, elems) at each ODE step
    psi_history::Vector{Float64}
    torque_history::Vector{ComplexF64}
    matrix_history::Union{Nothing, Vector{Array{ComplexF64,3}}}
end


# ============================================================================
# Core ODE-based ψ integration
# ============================================================================

"""
    integrate_psi_ode(n, nl, zi, mi, wdfac, divxfac, electron, method,
                      equil, intr, ctrl, kinetic_profiles; psi_min, psi_max) → NamedTuple

Integrate torque over ψ using adaptive ODE solver.
Ports Fortran `tintgrl_lsode` from torque.F90 lines 1163-1348.

The ODE state has neq = 2*(1 + 2*nl) real equations: real and imaginary
parts for each bounce harmonic ℓ ∈ {-nl, ..., nl}.

# Returns
NamedTuple with:
- `total::ComplexF64`: Total integrated torque
- `torque_profile`: Spline of dT/dψ vs ψ (from ODE trajectory)
- `matrix_integrated`: Trapezoidal-integrated mpert×mpert×6 matrix (if matrix method)
"""
function integrate_psi_ode(
    n::Int, nl::Int, zi::Int, mi::Int,
    wdfac::Float64, divxfac::Float64, electron::Bool,
    method::String, equil, intr::KineticForcesInternal, ctrl::KineticForcesControl,
    kinetic_profiles::Equilibrium.KineticProfileSplines;
    psi_min::Float64=0.0, psi_max::Float64=1.0
)
    is_matrix_method = occursin("mm", method)
    mpert = intr.mpert

    neq = 2 * (1 + 2 * nl)

    # Initialize params
    elems = is_matrix_method ? zeros(ComplexF64, mpert, mpert, 6) : nothing
    matrix_history = is_matrix_method ? Vector{Array{ComplexF64,3}}() : nothing

    params = PsiIntegrationParams(
        n, nl, zi, mi, wdfac, divxfac, electron, method,
        equil, intr, ctrl, kinetic_profiles,
        elems,
        Float64[], ComplexF64[], matrix_history)

    # Clip integration bounds to equilibrium data range
    # (Fortran line 1239-1241: clip to sq and xs_m ranges)
    # Equilibrium splines are unreliable near the magnetic axis (epsr→0, J→0);
    # start at psilow to avoid degenerate bounce/drift frequencies.
    # Cap upper bound at intr.psilim (DCON's integration limit): the perturbation
    # interpolants only have data below this, and extrapolation near the q=qlim
    # boundary blows up — matches Fortran PENTRC behavior.
    x0 = max(psi_min, equil.config.psilow)
    xout = min(psi_max, intr.psilim, 1.0 - 1e-6)

    if x0 >= xout
        return (total=ComplexF64(0.0), torque_profile=nothing, matrix_integrated=nothing)
    end

    y0 = zeros(neq)
    prob = ODEProblem(psi_integrand!, y0, (x0, xout), params)

    # Use Tsit5 (non-stiff) matching Fortran mf=10
    sol = solve(prob, Tsit5();
        reltol=ctrl.rtol_psi, abstol=ctrl.atol_psi,
        maxiters=10000,
        # Save at each internal step to capture torque profile
        save_everystep=true)

    if sol.retcode != :Success
        @warn "integrate_psi_ode may have issues for method=$method, n=$n" maxlog=3
    end

    # Extract total: sum over all bounce harmonics
    yf = sol.u[end]
    total = ComplexF64(0.0)
    for ell_idx in 1:(1 + 2*nl)
        total += complex(yf[2*(ell_idx-1)+1], yf[2*(ell_idx-1)+2])
    end

    # Torque profile from saved trajectory (available for diagnostics)
    torque_profile = nothing
    if length(params.psi_history) > 2
        # Store as (psi, dT/dpsi) pairs for later spline fitting if needed
        torque_profile = (psi=copy(params.psi_history), dtdpsi=copy(params.torque_history))
    end

    # Trapezoidal integration of kinetic matrices over ψ
    matrix_integrated = nothing
    if is_matrix_method && length(params.psi_history) > 1
        npsi = length(params.psi_history)
        matrix_integrated = zeros(ComplexF64, mpert, mpert, 6)
        for i in 2:npsi
            dpsi = params.psi_history[i] - params.psi_history[i-1]
            matrix_integrated .+= 0.5 .* (params.matrix_history[i-1] .+ params.matrix_history[i]) .* dpsi
        end
    end

    return (total=total, torque_profile=torque_profile, matrix_integrated=matrix_integrated)
end


"""
    psi_integrand!(dy, y, p::PsiIntegrationParams, psi)

Outer ψ ODE integrand. At each ψ, loops over bounce harmonics ℓ,
calls tpsi!() for each, and accumulates results.

Ports Fortran `tintgrnd` subroutine (torque.F90 lines 1356-1459).
"""
function psi_integrand!(dy, _, p::PsiIntegrationParams, psi)
    is_matrix_method = !isnothing(p.elems)

    if is_matrix_method
        p.elems .= 0
    end

    # Pre-allocate tpsi output
    tpsi_val = Ref{ComplexF64}(0.0 + 0.0im)
    wtw_l = is_matrix_method ? zeros(ComplexF64, p.intr.mpert, p.intr.mpert, 6) : nothing

    total_torque = ComplexF64(0.0)

    for ell_idx in 1:(1 + 2*p.nl)
        l = ell_idx - 1 - p.nl  # ℓ from -nl to nl

        tpsi!(tpsi_val, psi, p.n, l, p.zi, p.mi, p.wdfac, p.divxfac,
              p.electron, p.method, p.equil, p.intr, p.kinetic_profiles;
              op_wmats=wtw_l,
              atol_xlmda=p.ctrl.atol_xlmda, rtol_xlmda=p.ctrl.rtol_xlmda)

        # Pack real/imag into ODE state
        dy[2*(ell_idx-1)+1] = real(tpsi_val[])
        dy[2*(ell_idx-1)+2] = imag(tpsi_val[])

        total_torque += tpsi_val[]

        # Accumulate matrices across ℓ harmonics
        if is_matrix_method && !isnothing(wtw_l)
            p.elems .+= wtw_l
        end
    end

    # Record trajectory for profile reconstruction
    push!(p.psi_history, psi)
    push!(p.torque_history, total_torque)
    if is_matrix_method
        push!(p.matrix_history, copy(p.elems))
    end

    return nothing
end


# ============================================================================
# High-level orchestration
# ============================================================================

"""
    compute_torque_all_methods!(state::KineticForcesState, intr::KineticForcesInternal,
                                ctrl::KineticForcesControl, equil, kinetic_profiles)

Calculate torque/energy for all enabled methods using ODE integration.
For each method, integrates over flux surfaces using adaptive ODE solver
(replacing former trapezoidal integration on fixed grid).
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

    flags = [
        ctrl.fgar_flag, ctrl.tgar_flag, ctrl.pgar_flag,
        ctrl.rlar_flag, ctrl.clar_flag, ctrl.fcgl_flag,
        ctrl.fwmm_flag, ctrl.twmm_flag, ctrl.pwmm_flag,
        ctrl.ftmm_flag, ctrl.ttmm_flag, ctrl.ptmm_flag,
        ctrl.fkmm_flag, ctrl.tkmm_flag, ctrl.pkmm_flag,
        ctrl.frmm_flag, ctrl.trmm_flag, ctrl.prmm_flag
    ]

    for m in 1:length(intr.methods)
        if !flags[m]
            continue
        end

        method = intr.methods[m]
        intr.method = method
        is_matrix_method = occursin("mm", method)

        if ctrl.verbose
            println("---------------------------------------------")
            println("$method - $(intr.docs[m])")
        end

        # Allocate full block-diagonal matrix if needed
        op_wmats_full = nothing
        if is_matrix_method && intr.numpert_total > 0
            op_wmats_full = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 6)
        end

        total_torque = ComplexF64(0.0)
        npert = max(intr.npert, 1)

        for n_idx in 1:npert
            n = intr.nlow + n_idx - 1
            if n == 0
                n = ctrl.nn  # fallback to control parameter for single-n
            end

            # Dynamic ODE integration over ψ
            result = integrate_psi_ode(
                n, ctrl.nl, ctrl.zi, ctrl.mi,
                ctrl.wdfac, ctrl.divxfac, ctrl.electron,
                method, equil, intr, ctrl, kinetic_profiles;
                psi_min=ctrl.psilims[1], psi_max=ctrl.psilims[2])

            total_torque += result.total

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
            total_energy=total_energy
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


