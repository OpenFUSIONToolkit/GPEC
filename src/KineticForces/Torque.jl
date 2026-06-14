"""
    tpsi!(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method, equil, intr,
          kinetic_profiles; op_wmats=nothing)

Toroidal torque resulting from nonambipolar transport in perturbed equilibrium.
Imaginary component is proportional to the kinetic energy Im(T) = 2*n*dW_k.

# Arguments
- `tpsi_var`: Output complex torque value
- `psi::Float64`: Normalized poloidal flux
- `n::Int`: Toroidal mode number
- `l::Int`: Bounce harmonic number
- `zi::Int`: Ion charge in fundamental units (e)
- `mi::Int`: Ion mass (units of proton mass)
- `wdfac::Float64`: Drift factor
- `divxfac::Float64`: Divergence factor
- `electron::Bool`: Calculate quantities for electrons (zi,mi ignored)
- `method::String`: Integration method (RLAR, CLAR, *GAR, *TMM, *WMM, *KMM)
    where * = F,T,P for full,trapped,passing
- `equil`: PlasmaEquilibrium with 2D interpolants and named profile/geometry splines
- `intr::KineticForcesInternal`: Internal state with mode indexing and perturbation splines
- `kinetic_profiles::Equilibrium.KineticProfileSplines`: Named kinetic-profile splines
    (n_i, n_e, T_i, T_e, ω_E, ν_i, ν_e) loaded from `kinetic.dat`

# Optional Arguments
- `op_wmats::Array{ComplexF64,3}`: Store ForceFreeStates matrix elements

# Returns
- `ComplexF64`: Toroidal torque due to nonambipolar transport
"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int,
              zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
              electron::Bool, method::String, equil, intr::KineticForcesInternal,
              kinetic_profiles::Equilibrium.KineticProfileSplines;
              op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing,
              rex_override::Union{Nothing,Float64}=nothing,
              imx_override::Union{Nothing,Float64}=nothing,
              atol_xlmda::Float64=1e-9, rtol_xlmda::Float64=1e-6)

    if intr.verbose
        println("torque - tpsi function, psi = ", psi)
        println("  electron ", electron)
        println("  ell ", l)
    end

    # Enforce bounds
    if psi > 1
        tpsi_var[] = 0.0
        return
    end

    # Set species
    if electron
        chrg = -1 * e
        mass = me
        s = 2
    else
        chrg = zi * e
        mass = mi * mp
        s = 1
    end

    # Get perturbations (CubicSeriesInterpolant callable syntax).
    # Matrix-only calls (op_wmats provided for an "mm" method) build a linear
    # operator from the smats/tmats/xmats/ymats/zmats geometric coefficients;
    # dbob_m / divx_m only feed the scalar dJdJ normalisation in calculate_gar
    # which is irrelevant when rex_override / imx_override are set. Gate the
    # dereference so a caller can pass `dbob_m === nothing` (the default) and
    # still drive the matrix path.
    is_matrix_only = occursin("mm", method) && !isnothing(op_wmats)
    if is_matrix_only && isnothing(intr.dbob_m)
        dbob_m_f = zeros(ComplexF64, intr.mpert)
        divx_m_f = zeros(ComplexF64, intr.mpert)
    else
        dbob_m_f = intr.dbob_m(psi; hint=intr.dbob_m_hint)
        divx_m_f = intr.divx_m(psi; hint=intr.divx_m_hint)
    end

    # Sample poloidal quantities on theta grid (buffers pre-allocated on intr).
    mthsurf_local = intr.mthsurf
    xs            = intr.tpsi_xs
    B_vals        = intr.tpsi_B
    dBdpsi_vals   = intr.tpsi_dBdpsi
    dBdtheta_vals = intr.tpsi_dBdtheta
    jac_vals      = intr.tpsi_jac
    djdpsi_vals   = intr.tpsi_djdpsi

    hB = intr.hint2d_eqfun_B
    hJ = intr.hint2d_rzphi_jac
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        pt = (psi, theta)

        B_vals[i+1] = equil.eqfun_B(pt; hint=hB)
        dBdpsi_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hB) / intr.chi1
        dBdtheta_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hB)
        jac_vals[i+1] = equil.rzphi_jac(pt; hint=hJ) / intr.chi1
        djdpsi_vals[i+1] = equil.rzphi_jac(pt; deriv=DerivOp(1, 0), hint=hJ) / intr.chi1^2
    end

    # Create periodic interpolant for poloidal quantities
    tspl = cubic_interp(xs, Series(hcat(B_vals, dBdpsi_vals, dBdtheta_vals, jac_vals, djdpsi_vals)); bc=PeriodicBC())

    bmax = maximum(B_vals)
    ibmax = argmax(B_vals)
    if bmax != B_vals[ibmax]
        error("ERROR: tpsi! - Equilibrium field maximum not consistent with index")
    end

    # "4th smallest" heuristic for initial bmin like Fortran
    bmin = minimum(B_vals)
    for _ in 2:4
        mask = B_vals .> bmin
        if !any(mask)
            break
        end
        bmin = minimum(B_vals[mask])
    end
    # Use >= like Fortran mask for ibmin
    inds = findall(B_vals .>= bmin)
    isempty(inds) && error("ERROR: tpsi! - could not find ibmin")
    ibmin = inds[argmin(B_vals[inds])]
    if !isapprox(bmin, B_vals[ibmin]; rtol=0, atol=0)
        error("ERROR: tpsi! - Equilibrium field minimum not consistent with index")
    end

    theta_bmin = xs[ibmin]
    theta_bmax = xs[ibmax]

    # Find roots of dB/dθ = 0 (extrema) using Roots.jl
    dBdtheta_interp = cubic_interp(xs, dBdtheta_vals; bc=PeriodicBC())
    extrema_roots = Float64[]
    for i in 1:length(xs)-1
        if dBdtheta_vals[i] * dBdtheta_vals[i+1] < 0
            push!(extrema_roots, find_zero(dBdtheta_interp, (xs[i], xs[i+1]), Roots.Brent()))
        end
    end
    for θ in extrema_roots
        θn = mod(θ, 1.0)
        Bθ = tspl(θn)[1]
        if Bθ < bmin
            bmin = Bθ
            theta_bmin = θn
        end
        if Bθ > bmax
            bmax = Bθ
            theta_bmax = θn
        end
    end

    # Flux-function quantities — read directly from named splines on the
    # PlasmaEquilibrium and the externally-loaded KineticProfileSplines.
    q       = equil.profiles.q_spline(psi)
    dVdpsi  = equil.profiles.dVdpsi_spline(psi)
    if electron
        n_s       = kinetic_profiles.ne_spline(psi)
        T_s       = kinetic_profiles.Te_spline(psi)
        dn_s_dpsi = kinetic_profiles.ne_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Te_deriv(psi)
        nu_s      = kinetic_profiles.nue_spline(psi)
    else
        n_s       = kinetic_profiles.ni_spline(psi)
        T_s       = kinetic_profiles.Ti_spline(psi)
        dn_s_dpsi = kinetic_profiles.ni_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Ti_deriv(psi)
        nu_s      = kinetic_profiles.nui_spline(psi)
    end
    welec = kinetic_profiles.omegaE_spline(psi)

    # Diamagnetic frequencies (Logan & Park 2013, Eq. 7) computed at evaluation
    # time from the kinetic-profile derivatives — no longer baked into the loader.
    wdian = -twopi * T_s * dn_s_dpsi / (chrg * intr.chi1 * n_s)
    wdiat = -twopi * dT_s_dpsi / (chrg * intr.chi1)
    wphi  = welec + wdian + wdiat
    wtran = sqrt(2 * T_s / mass) / (q * intr.ro)
    wgyro = chrg * intr.bo / mass
    nuk   = nu_s

    rsquared_bmin = equil.rzphi_rsquared((psi, theta_bmin))
    if rsquared_bmin <= 0
        println("  psi = ", psi, " -> r^2 at min(B) = ", rsquared_bmin)
        println("  -- theta at min(B) = ", theta_bmin)
        for i in 0:10
            θ = i / 10.0
            Bθ = tspl(θ)[1]
            rz = equil.rzphi_rsquared((psi, θ))
            println("  -- theta,B(theta),r^2(theta) = ", θ, ", ", Bθ, ", ", rz)
        end
        error("ERROR: torque - minor radius is negative")
    end

    avg_r = equil.geometry.avg_r_spline(psi)
    avg_R = equil.geometry.avg_R_spline(psi)
    epsr  = avg_r / avg_R
    wbhat = (π / 4) * sqrt(epsr / 2) * wtran
    wdhat = q^3 * wtran^2 / (4 * epsr * wgyro) * wdfac
    nueff = nu_s / (2 * epsr)

    if intr.verbose
        @printf("   eq values = %.1e %.1e %.1e %.1e %.1e %.1e %.1f %d\n",
               wdian, wdiat, welec, wdhat, wbhat, nueff, q, 0)
    end

    if intr.verbose
        println("  method = ", method)
    end

    # Method selection
    if method == "fcgl"
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac,
                                    n_s, T_s, equil, intr)

    elseif method == "rlar"
        tpsi_var[] = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    wdhat, wbhat, nueff, dVdpsi, n_s, T_s,
                                    dbob_m_f, intr.bo, bmin)

    elseif method == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, intr.bo, bmax, bmin, n_s, T_s, mass, chrg,
                                    tspl, dbob_m_f, divx_m_f, divxfac, wdfac)

    elseif method in ["fgar", "tgar", "pgar", "fwmm", "twmm", "pwmm",
                      "ftmm", "ttmm", "ptmm", "fkmm", "tkmm", "pkmm",
                      "frmm", "trmm", "prmm"]
        # Evaluate geometric matrices at current ψ if matrix path
        smat_f = nothing
        tmat_f = nothing
        xmat_f = nothing
        ymat_f = nothing
        zmat_f = nothing
        if !isnothing(op_wmats) && !isnothing(intr.smats)
            smat_f = reshape(intr.smats(psi), intr.mpert, intr.mpert)
            tmat_f = reshape(intr.tmats(psi), intr.mpert, intr.mpert)
            xmat_f = reshape(intr.xmats(psi), intr.mpert, intr.mpert)
            ymat_f = reshape(intr.ymats(psi), intr.mpert, intr.mpert)
            zmat_f = reshape(intr.zmats(psi), intr.mpert, intr.mpert)
        end
        tpsi_var[] = calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                   nuk, intr.bo, bmax, bmin, n_s, T_s, mass, chrg,
                                   tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
                                   method, op_wmats;
                                   chi1=intr.chi1, ro=intr.ro, mfac=intr.mfac,
                                   mpert=intr.mpert, ibmax=ibmax, theta_bmax=theta_bmax,
                                   smat=smat_f, tmat=tmat_f, xmat=xmat_f,
                                   ymat=ymat_f, zmat=zmat_f,
                                   energy_atol=atol_xlmda, energy_rtol=rtol_xlmda,
                                   pitch_atol=atol_xlmda, pitch_rtol=rtol_xlmda,
                                   rex_override=rex_override, imx_override=imx_override)
    else
        error("ERROR: torque - unknown method")
    end

    if intr.verbose
        println("torque - end function, psi = ", psi)
    end

    return nothing
end


# ============================================================================
# Method-specific torque calculation functions
# ============================================================================

"""
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac, n_s, T_s,
                   equil, intr)::ComplexF64

Calculate FCGL (Full Circular Gyrokinetic Landau) torque.
Implements simplified energy balance equation.
Only valid for bounce harmonic l=0.

Based on: [Logan et al., Phys. Plasmas 2013]
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac::Float64,
                        n_s::Float64, T_s::Float64, equil,
                        intr::KineticForcesInternal)::ComplexF64

    # Only implemented for l=0
    if l != 0
        return ComplexF64(0.0, 0.0)
    end

    mthsurf_local = intr.mthsurf

    # Create spline for poloidal integrands
    cglspl = zeros(2, mthsurf_local + 1)
    theta_vals = range(0, 1, length=mthsurf_local + 1)

    # Evaluate poloidal functions and field perturbations
    hB = intr.hint2d_eqfun_B
    hJ = intr.hint2d_rzphi_jac
    for i in eachindex(theta_vals)
        theta = theta_vals[i]

        B_val = equil.eqfun_B((psi, theta); hint=hB)
        jac_val = equil.rzphi_jac((psi, theta); hint=hJ)

        # Fourier component of field perturbations
        expm = exp(im * twopi * intr.mfac * theta)
        dbob = sum(dbob_m_f .* expm)  # dB/B
        divx = sum(divx_m_f .* expm) * divxfac  # ∇·ξ⊥

        kapx = -0.5 * (dbob + divx)  # Kappa coupling

        # Poloidal integrands (include 0.5 factor for quadratic form)
        cglspl[1, i] = jac_val * 0.5 * abs(divx)^2
        cglspl[2, i] = jac_val * 0.5 * abs(divx + 3.0 * kapx)^2
    end

    # Trapezoidal integration
    integral_1 = 0.0
    integral_2 = 0.0
    for i in 2:length(theta_vals)
        dtheta = theta_vals[i] - theta_vals[i-1]
        integral_1 += (cglspl[1, i-1] + cglspl[1, i]) / 2 * dtheta
        integral_2 += (cglspl[2, i-1] + cglspl[2, i]) / 2 * dtheta
    end

    # Calculate torque: T = 2*n*i*n_s*T_s * [weighted integral]
    result = 2.0 * n * im * n_s * T_s *
        (0.5 * (5.0/3.0) * integral_1 +
         0.5 * (1.0/3.0) * integral_2)

    return result
end


"""
    calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, wdhat, wbhat,
                   nueff, dVdpsi, n_s, T_s, dbob_m_f, bo, bmin)::ComplexF64

Calculate RLAR (Reduced Large Aspect Ratio) torque.
Uses energy space integration with pitch angle averaging.
Valid for low aspect ratio tokamaks (ε << 1).

Reference: [Logan et al., Phys. Plasmas, 2013]
"""
function calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
                        wdhat, wbhat, nueff, dVdpsi::Float64,
                        n_s::Float64, T_s::Float64,
                        dbob_m_f, bo, bmin=0.5)::ComplexF64

    # Setup parameters for energy integration
    lnq = Float64(l)  # Resonant mode for trapped particles

    # Maximum pitch angle parameter (use safe estimate)
    lmdamax = min(1.0/(1-epsr), bo/bmin)

    # Energy-space quadrature
    # This computes ∫ K(x) dx where K is the resonance operator
    xint = integrate_energy(wdian, wdiat, welec, wdhat, wbhat, nueff,
                         l, lnq, n, psi, lmdamax, "rlar")

    # Kappa/bounce averaging (effect of field perturbations)
    # Placeholder: simplified estimate
    kappaint_val = sqrt(mean(abs.(dbob_m_f).^2))

    # dV/dpsi gradient term from Clebsch coordinate Jacobian
    psi_factor = dVdpsi

    # Normalization: √(ε/(2π³)) * n² * n_s * T_s
    norm = sqrt(epsr / (2.0 * π^3)) * Float64(n)^2 * n_s * T_s

    # Result: dT/dpsi = -(dψ/dpsi) * κ_int * x_int * norm
    result = psi_factor * kappaint_val * 0.5 * (-xint) * norm

    return result
end


"""
    calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                   bmax, bmin, n_s, T_s, mass, chrg, tspl, dbob_m_f, divx_m_f,
                   divxfac, wdfac)::ComplexF64

Calculate CLAR (Circular Large Aspect Ratio) torque.
Uses pitch-angle resolved calculations for trapped and passing particles.
Includes bounce-averaged integrals over lambda (pitch angle).

Status: Partially implemented (stub for full calculation)
"""
function calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                        bmax, bmin, n_s::Float64, T_s::Float64, mass, chrg, tspl,
                        dbob_m_f, divx_m_f, divxfac, wdfac)::ComplexF64

    @warn "CLAR method not yet fully implemented, returning zero" maxlog=1
    return ComplexF64(0.0, 0.0)
end


"""
    calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo, bmax,
                  bmin, n_s, T_s, mass, chrg, tspl, dbob_m_f, divx_m_f,
                  divxfac, wdfac, method, op_wmats; kwargs...)::ComplexF64

Calculate GAR (General Aspect Ratio) torque.
Fully general method without aspect ratio expansion.
Handles variants: FGAR (full), TGAR (trapped), PGAR (passing).
Can compute torque (*TMM), energy (*WMM), or matrix elements (*KMM/*RMM).

Ports Fortran torque.F90 GAR branch (lines 529-932).

# Steps
1. Compute bounce-averaged quantities via `compute_bounce_data()`
2. Build fbnce interpolant over λ, normalize for numerical stability
3. Integrate over pitch angle via `integrate_pitch_gar_quadgk()`
4. Apply torque normalization (Eq. 19, Logan et al. 2013)
5. If matrix path: assemble and normalize kinetic matrices

# Keyword Arguments (rex/imx override)
- `rex_override::Union{Nothing,Float64}`: Override real-part multiplier for resonance
  operator. When both overrides are provided, bypasses method-string derivation.
- `imx_override::Union{Nothing,Float64}`: Override imaginary-part multiplier.
  Use `rex_override=1.0, imx_override=1.0` to get full complex result for
  simultaneous kwmat/ktmat extraction via `compute_kinetic_matrices_at_psi!`.

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                       bmax, bmin, n_s::Float64, T_s::Float64, mass, chrg, tspl,
                       dbob_m_f, divx_m_f, divxfac, wdfac, method, op_wmats;
                       chi1::Float64, ro::Float64, mfac::Vector{Int}, mpert::Int,
                       ibmax::Int, theta_bmax::Float64,
                       smat=nothing, tmat=nothing, xmat=nothing,
                       ymat=nothing, zmat=nothing,
                       nlmda::Int=64, ntheta::Int=128,
                       nutype::String="harmonic", f0type::String="maxwellian",
                       nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                       energy_atol::Float64=1e-7, energy_rtol::Float64=1e-5,
                       pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6,
                       rex_override::Union{Nothing,Float64}=nothing,
                       imx_override::Union{Nothing,Float64}=nothing)::ComplexF64

    # 1. Compute bounce-averaged quantities
    bounce = compute_bounce_data(
        psi, n, l, q, bo, bmax, bmin, ibmax, theta_bmax,
        tspl, mfac, chi1, ro, dbob_m_f, divx_m_f, divxfac, wdfac,
        mass, chrg, T_s, method;
        nlmda, ntheta, smat, tmat, xmat, ymat, zmat)

    if bounce.nlmda == 0
        return ComplexF64(0.0, 0.0)
    end

    # 2. Build fbnce interpolant: [wb, wd, f₁, f₂, ...]
    # Number of flux quantities: 1 (scalar dJdJ) + optional packed matrix storage
    # (Hermitian-triangle for A/D/H + full blocks for B/C/E; see `nqty_matrix`).
    do_matrices = !isnothing(op_wmats) && !isnothing(bounce.wmats_vs_lambda)
    nqty_mat = do_matrices ? nqty_matrix(mpert) : 0
    nqty = 1 + nqty_mat

    # Pack all quantities into a matrix for interpolation.
    fbnce_data = zeros(Float64, bounce.nlmda, 2 + nqty)
    fbnce_data[:, 1] .= bounce.wb
    fbnce_data[:, 2] .= bounce.wd
    fbnce_data[:, 3] .= bounce.dJdJ

    if do_matrices
        # Packed matrix block lives at columns 4:(3+nqty_mat). The real() wrap
        # keeps the historical OLD-path behavior (torque pipeline uses only the
        # real part of the outer products).
        @inbounds for q in 1:nqty_mat, ilmda in 1:bounce.nlmda
            fbnce_data[ilmda, 3 + q] = real(bounce.wmats_vs_lambda[ilmda, q])
        end
    end

    # Normalize flux quantities by 1/median for numerical stability (Fortran lines 819-823).
    # Compute medians from the unscaled data and rescale in place BEFORE building
    # the interpolant — the pre-normalization build was dead work (allocates a
    # CubicSeriesInterpolant that was immediately overwritten).
    fbnce_norm = ones(Float64, nqty)
    for i in 1:nqty
        col = view(fbnce_data, :, i + 2)
        med = median(abs.(col))
        if med > 0
            fbnce_norm[i] = 1.0 / med
            col .*= fbnce_norm[i]
        end
    end

    # Build CubicSeriesInterpolant on normalized data (single build)
    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=ZeroCurvBC())

    # 3. Set rex/imx multipliers (Fortran lines 839-847)
    method_suffix = length(method) >= 4 ? method[2:4] : ""
    if !isnothing(rex_override) && !isnothing(imx_override)
        rex = rex_override
        imx = imx_override
    else
        rex = 1.0
        imx = 1.0
        if method_suffix in ["wmm", "kmm"]
            rex = 0.0  # energy only
        elseif method_suffix in ["tmm", "rmm"]
            imx = 0.0  # torque only
        end
    end

    # 4. Pitch angle integration (adaptive Gauss-Kronrod over λ)
    lxint = integrate_pitch_gar_quadgk(
        wdian, wdiat, welec, nuk, bo / bmax, epsr, q,
        fbnce, fbnce_norm, nqty, l, n, rex, imx, psi, method;
        nutype, f0type, nufac, ximag, qt,
        energy_atol, energy_rtol, pitch_atol, pitch_rtol)

    # 5. Compute scalar torque (Fortran lines 852-854)
    # Eq. (19) of Logan et al., Phys. Plasmas 20, 122507 (2013)
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * n_s * T_s *
            (chi1 / twopi)  # unit conversion from ψ to ψ_n, θ_n to θ
    tpsi_val = tnorm * (lxint[1] / fbnce_norm[1])

    # 6. Kinetic matrix assembly (Fortran lines 858-923)
    if do_matrices
        op_wmats .= 0
        energy_factor = tnorm * (1 / (2 * im * n))  # convert torque → energy
        Mu = (mpert * (mpert + 1)) ÷ 2
        base = 1  # lxint offset after scalar-torque slot (index 1)

        # Helper: fetch normalized pitch integral at lxint[base+q] for q ∈ [1..nqty_mat].
        @inline elem(q) = complex(lxint[base + q] / fbnce_norm[base + q]) * energy_factor

        # A (Hermitian, k=1): upper triangle stored at q ∈ [1..Mu]; mirror to lower.
        off = 0
        @inbounds for j in 1:mpert, i in 1:j
            v = elem(off + _tri_idx(i, j))
            op_wmats[i, j, 1] = v
            if i != j
                op_wmats[j, i, 1] = conj(v)
            end
        end
        off += Mu
        # D (Hermitian, k=4)
        @inbounds for j in 1:mpert, i in 1:j
            v = elem(off + _tri_idx(i, j))
            op_wmats[i, j, 4] = v
            if i != j
                op_wmats[j, i, 4] = conj(v)
            end
        end
        off += Mu
        # H (Hermitian, k=6)
        @inbounds for j in 1:mpert, i in 1:j
            v = elem(off + _tri_idx(i, j))
            op_wmats[i, j, 6] = v
            if i != j
                op_wmats[j, i, 6] = conj(v)
            end
        end
        off += Mu
        # B (full, k=2)
        @inbounds for j in 1:mpert, i in 1:mpert
            op_wmats[i, j, 2] = elem(off + _full_idx(i, j, mpert))
        end
        off += mpert^2
        # C (full, k=3)
        @inbounds for j in 1:mpert, i in 1:mpert
            op_wmats[i, j, 3] = elem(off + _full_idx(i, j, mpert))
        end
        off += mpert^2
        # E (full, k=5)
        @inbounds for j in 1:mpert, i in 1:mpert
            op_wmats[i, j, 5] = elem(off + _full_idx(i, j, mpert))
        end

        # DCON normalization (Fortran lines 874-876)
        op_wmats .*= 2 * μ₀
        op_wmats[:, :, 1:3] ./= chi1
        op_wmats[:, :, 1] ./= chi1

        # Method-specific output
        if method_suffix in ["kmm", "rmm"]
            # Matrix norms independent of ξ (Fortran lines 878-897)
            tpsi_val = ComplexF64(0.0)
            for k in 1:6
                # Spectral norm via SVD
                mat = op_wmats[:, :, k]
                sv = svdvals(mat' * mat)
                tpsi_val += maximum(real.(sv))
            end
            tpsi_val = complex(rex + imx * im) * sqrt(abs(tpsi_val))

        elseif method_suffix in ["wmm", "tmm", "pmm"]
            # Mode-coupled dW contraction with displacement vectors (Fortran lines 898-914)
            # TODO: Requires xs_m displacement interpolants from PerturbedEquilibriumState
            # For now, store matrices and return scalar torque
        end
    end

    return tpsi_val
end


"""
    _setup_surface_state(psi, n, l, zi, mi, wdfac, electron,
                          equil, intr, kinetic_profiles) → NamedTuple

Private helper for the calculated-matrix path. Reproduces the per-surface
setup in `tpsi!` (theta-grid sampling, bounce-extremum finding, flux-function
evaluation, diamagnetic/drift frequencies) without any of the perturbation-
dependent bookkeeping or method dispatch.

This keeps `compute_kinetic_matrices_at_psi!` structurally independent of
`tpsi!` so the matrix-only path can be evolved (e.g. QuadGK pitch in Phase C)
without perturbing the perturbative torque pipeline.
"""
function _setup_surface_state(
    psi::Float64, zi::Int, mi::Int,
    electron::Bool, equil, intr::KineticForcesInternal,
    kinetic_profiles::Equilibrium.KineticProfileSplines,
)
    if electron
        chrg = -1 * e
        mass = me
    else
        chrg = zi * e
        mass = mi * mp
    end

    mthsurf_local = intr.mthsurf
    xs            = intr.tpsi_xs
    B_vals        = intr.tpsi_B
    dBdpsi_vals   = intr.tpsi_dBdpsi
    dBdtheta_vals = intr.tpsi_dBdtheta
    jac_vals      = intr.tpsi_jac
    djdpsi_vals   = intr.tpsi_djdpsi

    hB = intr.hint2d_eqfun_B
    hJ = intr.hint2d_rzphi_jac
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        pt = (psi, theta)
        B_vals[i+1] = equil.eqfun_B(pt; hint=hB)
        dBdpsi_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hB) / intr.chi1
        dBdtheta_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hB)
        jac_vals[i+1] = equil.rzphi_jac(pt; hint=hJ) / intr.chi1
        djdpsi_vals[i+1] = equil.rzphi_jac(pt; deriv=DerivOp(1, 0), hint=hJ) / intr.chi1^2
    end

    tspl = cubic_interp(xs, Series(hcat(B_vals, dBdpsi_vals, dBdtheta_vals, jac_vals, djdpsi_vals)); bc=PeriodicBC())

    bmax = maximum(B_vals)
    ibmax = argmax(B_vals)
    bmax == B_vals[ibmax] ||
        error("ERROR: _setup_surface_state - Equilibrium field maximum not consistent with index")

    bmin = minimum(B_vals)
    for _ in 2:4
        mask = B_vals .> bmin
        any(mask) || break
        bmin = minimum(B_vals[mask])
    end
    inds = findall(B_vals .>= bmin)
    isempty(inds) && error("ERROR: _setup_surface_state - could not find ibmin")
    ibmin = inds[argmin(B_vals[inds])]
    isapprox(bmin, B_vals[ibmin]; rtol=0, atol=0) ||
        error("ERROR: _setup_surface_state - Equilibrium field minimum not consistent with index")

    theta_bmin = xs[ibmin]
    theta_bmax = xs[ibmax]

    dBdtheta_interp = cubic_interp(xs, dBdtheta_vals; bc=PeriodicBC())
    for i in 1:length(xs)-1
        if dBdtheta_vals[i] * dBdtheta_vals[i+1] < 0
            θ = find_zero(dBdtheta_interp, (xs[i], xs[i+1]), Roots.Brent())
            θn = mod(θ, 1.0)
            Bθ = tspl(θn)[1]
            if Bθ < bmin
                bmin = Bθ
                theta_bmin = θn
            end
            if Bθ > bmax
                bmax = Bθ
                theta_bmax = θn
            end
        end
    end

    q = equil.profiles.q_spline(psi)
    if electron
        n_s       = kinetic_profiles.ne_spline(psi)
        T_s       = kinetic_profiles.Te_spline(psi)
        dn_s_dpsi = kinetic_profiles.ne_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Te_deriv(psi)
        nu_s      = kinetic_profiles.nue_spline(psi)
    else
        n_s       = kinetic_profiles.ni_spline(psi)
        T_s       = kinetic_profiles.Ti_spline(psi)
        dn_s_dpsi = kinetic_profiles.ni_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Ti_deriv(psi)
        nu_s      = kinetic_profiles.nui_spline(psi)
    end
    welec = kinetic_profiles.omegaE_spline(psi)

    wdian = -twopi * T_s * dn_s_dpsi / (chrg * intr.chi1 * n_s)
    wdiat = -twopi * dT_s_dpsi / (chrg * intr.chi1)
    wtran = sqrt(2 * T_s / mass) / (q * intr.ro)
    wgyro = chrg * intr.bo / mass
    nuk   = nu_s

    rsquared_bmin = equil.rzphi_rsquared((psi, theta_bmin))
    rsquared_bmin > 0 ||
        error("ERROR: _setup_surface_state - minor radius is negative at psi=$psi, theta_bmin=$theta_bmin")

    avg_r = equil.geometry.avg_r_spline(psi)
    avg_R = equil.geometry.avg_R_spline(psi)
    epsr  = avg_r / avg_R

    return (;
        chrg, mass,
        tspl, bmax, bmin, ibmax, theta_bmax,
        q, n_s, T_s, welec,
        wdian, wdiat, wtran, wgyro, nuk,
        epsr,
    )
end


"""
    kinetic_energy_matrices_for_euler_lagrange!(kwmat, ktmat, state, psi, n, l, wdfac, intr;
                                                kwargs...) → nothing

Compute the six kinetic Euler-Lagrange coefficient matrices of Logan 2015
Eqs 7.30–7.35 (A_k, B_k, C_k, D_k, E_k, H_k) at a single (ψ, n, ℓ) and write
them into pre-allocated `kwmat[mpert, mpert, 6]` (fwmm half, Fortran
rex=0, imx=1) and `ktmat[mpert, mpert, 6]` (ftmm half, rex=1, imx=0).

Matrix-only path (no scalar torque slot in the pitch-angle buffer), so
`nqty = mpert²·6` instead of `1 + mpert²·6`.

Uses `integrate_pitch_gar_quadgk_wt` to emit both halves from a single
energy integration per (λ, E), reproducing Fortran's two-pass semantics
(`torque.F90:842-847`). Per-surface matrix dumps confirm element-by-element
match against Fortran `fourfit.F:1080-1082` (`kwmat_l`, `ktmat_l`).

For the Hermitian-outer-product blocks A/D/H stored as upper-triangles,
the mirror rule differs between halves:
  kwmat[j,i] =  conj(kwmat[i,j])   — Hermitian (S_w pure imaginary)
  ktmat[j,i] = -conj(ktmat[i,j])   — anti-Hermitian (S_t pure real)
Derivation: `conj(S_w) = -S_w` vs `conj(S_t) = S_t`, combined with
`conj(factor) = -factor` (factor = -i/(2n)). These mirrors recover
Fortran's independent-slot computation at the mirrored (j,i) positions.
"""
function kinetic_energy_matrices_for_euler_lagrange!(
    kwmat::Array{ComplexF64,3},
    ktmat::Array{ComplexF64,3},
    state::NamedTuple,
    psi::Float64, n::Int, l::Int, wdfac::Float64,
    intr::KineticForcesInternal;
    nlmda::Int=128, ntheta::Int=128,
    nutype::String="harmonic", f0type::String="maxwellian",
    nufac::Float64=1.0, ximag::Float64=0.0,
    energy_atol::Float64=1e-9, energy_rtol::Float64=1e-6,
    pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6
)
    mpert = intr.mpert
    mfac = intr.mfac
    chi1 = intr.chi1
    ro   = intr.ro
    bo   = intr.bo

    # Geometric matrices at this ψ (Fortran torque.F90 block matrices)
    smat_f = reshape(intr.smats(psi), mpert, mpert)
    tmat_f = reshape(intr.tmats(psi), mpert, mpert)
    xmat_f = reshape(intr.xmats(psi), mpert, mpert)
    ymat_f = reshape(intr.ymats(psi), mpert, mpert)
    zmat_f = reshape(intr.zmats(psi), mpert, mpert)

    # Matrix-only path: dbob/divx perturbation coefficients are not used
    # (compute_bounce_data folds them into the scalar dJdJ we discard).
    dbob_m_f = zeros(ComplexF64, mpert)
    divx_m_f = zeros(ComplexF64, mpert)

    bounce = compute_bounce_data(
        psi, n, l, state.q, bo, state.bmax, state.bmin, state.ibmax, state.theta_bmax,
        state.tspl, mfac, chi1, ro, dbob_m_f, divx_m_f, 1.0, wdfac,
        state.mass, state.chrg, state.T_s, "fwmm";
        nlmda, ntheta, smat=smat_f, tmat=tmat_f, xmat=xmat_f, ymat=ymat_f, zmat=zmat_f)

    if bounce.nlmda == 0
        fill!(kwmat, 0)
        fill!(ktmat, 0)
        return nothing
    end

    # Pack wb, wd, and packed matrix block into fbnce — NO scalar slot.
    # fbnce is COMPLEX on this path to carry the full op_wmats phase (Fortran
    # torque.F90:789 writes `wbbar*op_wmats(i,j,k)/ro**2` into a complex cspline;
    # dropping the imag part zeros out off-Hermitian matrix structure — in
    # particular op_B = W_Z†W_X, op_C = W_Z†W_Y, op_E = W_X†W_Y whose
    # diagonals carry genuine phase. wb/wd slots are stored with zero imag.
    # Matrix block is packed as 3 Hermitian upper-triangles + 3 full blocks;
    # see `nqty_matrix` in BounceAveraging.jl.
    nqty = nqty_matrix(mpert)
    fbnce_data = zeros(ComplexF64, bounce.nlmda, 2 + nqty)
    fbnce_data[:, 1] .= bounce.wb
    fbnce_data[:, 2] .= bounce.wd
    # bounce.wmats_vs_lambda is already (nlmda, nqty) in the packed layout,
    # so the matrix block is a direct copy.
    fbnce_data[:, 3:end] .= bounce.wmats_vs_lambda

    # Column-wise median normalization for numerical stability (Fortran lines 819-823).
    fbnce_norm = ones(Float64, nqty)
    for i in 1:nqty
        col = view(fbnce_data, :, i + 2)
        med = median(abs.(col))
        if med > 0
            fbnce_norm[i] = 1.0 / med
            col .*= fbnce_norm[i]
        end
    end

    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=ZeroCurvBC())

    # Dual-output pitch integration: one energy sweep per (λ, E) produces both
    # halves. Packed return is [wmm | tmm], each length nqty.
    lxint = integrate_pitch_gar_quadgk_wt(
        state.wdian, state.wdiat, state.welec, state.nuk,
        bo / state.bmax, state.epsr, state.q,
        fbnce, fbnce_norm, nqty, l, n, psi, "fwmm";
        nutype, f0type, nufac, ximag, qt=false,
        energy_atol, energy_rtol, pitch_atol, pitch_rtol)

    # Torque → energy normalization (Fortran torque.F90 lines 860-876).
    # Eq. (19) of Logan et al., Phys. Plasmas 20, 122507 (2013).
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * state.n_s * state.T_s *
            (chi1 / twopi)
    energy_factor = tnorm * (1 / (2 * im * n))

    Mu = (mpert * (mpert + 1)) ÷ 2
    # Fetch normalized element at pitch-integral slot q, from either half.
    # half_offset = 0 for wmm (→kwmat), = nqty for tmm (→ktmat).
    @inline elem(q, half_offset) = complex(lxint[half_offset + q] / fbnce_norm[q]) * energy_factor

    # For A/D/H Hermitian-outer-product blocks stored as upper-triangles, the
    # lower-triangle reconstruction uses different mirrors per half:
    #   kwmat[j,i] =  conj(kwmat[i,j])  (Hermitian; S_w pure imaginary)
    #   ktmat[j,i] = -conj(ktmat[i,j])  (anti-Hermitian; S_t pure real)
    # This matches Fortran's independently-computed slots at (j,i).
    @inline function _assemble_hermitian!(dest, k, off, half, mirror_sign)
        @inbounds for j in 1:mpert, i in 1:j
            v = elem(off + _tri_idx(i, j), half)
            dest[i, j, k] = v
            if i != j
                dest[j, i, k] = mirror_sign * conj(v)
            end
        end
    end

    @inline function _assemble_full!(dest, k, off, half)
        @inbounds for j in 1:mpert, i in 1:mpert
            dest[i, j, k] = elem(off + _full_idx(i, j, mpert), half)
        end
    end

    for (dest, half, mirror_sign) in ((kwmat, 0, 1.0), (ktmat, nqty, -1.0))
        off = 0
        _assemble_hermitian!(dest, 1, off, half, mirror_sign);  off += Mu        # A (k=1)
        _assemble_hermitian!(dest, 4, off, half, mirror_sign);  off += Mu        # D (k=4)
        _assemble_hermitian!(dest, 6, off, half, mirror_sign);  off += Mu        # H (k=6)
        _assemble_full!(dest, 2, off, half);                    off += mpert^2   # B (k=2)
        _assemble_full!(dest, 3, off, half);                    off += mpert^2   # C (k=3)
        _assemble_full!(dest, 5, off, half)                                      # E (k=5)
    end

    # DCON normalization (Fortran torque.F90 lines 874-876)
    for mat in (kwmat, ktmat)
        mat .*= 2 * μ₀
        mat[:, :, 1:3] ./= chi1
        mat[:, :, 1] ./= chi1
    end

    return nothing
end


"""
    compute_kinetic_matrices_at_psi!(kwmat, ktmat, psi, n, l, zi, mi,
        wdfac, divxfac, electron, equil, intr, kinetic_profiles)

Compute the six kinetic Euler-Lagrange coefficient matrices at a single
flux surface and split them into `kwmat` and `ktmat` in the convention
used by the DCON matrix-assembly path (Logan 2015 Eqs 7.30–7.35).

Rather than running two integrations like the reference Fortran PENTRC
(one with `rex=0, imx=1` for `kwmat` and another with `rex=1, imx=0` for
`ktmat`), this path integrates once with `rex=imx=1` to get the full
complex response, then decomposes by real/imag parts — equivalent math
at half the work. After the `-i/(2n)` normalization inside
`kinetic_energy_matrices_for_euler_lagrange!`, the full complex response
splits cleanly:

  - `kwmat` ← fwmm half (Fortran rex=0, imx=1 pass)
  - `ktmat` ← ftmm half (Fortran rex=1, imx=0 pass)

Each half is complex (not pure real / pure imag), matching Fortran's two
independent integration passes at `torque.F90:842-847`. Per-surface matrix
dumps confirm element-by-element agreement with Fortran `fourfit.F:1080-1082`
(`kwmat_l`, `ktmat_l`). This is the Fortran convention required by the
adjoint combinations `kwmat ± ktmat` in `ForceFreeStates/Kinetic.jl` /
`~/Code/gpec/dcon/sing.f:967-1075` for non-Hermitian B_k, C_k, E_k.

# Arguments
- `kwmat::Array{ComplexF64,3}`: Output (mpert×mpert×6), fwmm half, zeroed on entry
- `ktmat::Array{ComplexF64,3}`: Output (mpert×mpert×6), ftmm half, zeroed on entry
- `psi, n, l, zi, mi, wdfac, divxfac, electron`: Same as `tpsi!` (divxfac unused
  on the matrix path — retained for call-site compatibility)
- `equil`: PlasmaEquilibrium
- `intr::KineticForcesInternal`: Internal state with mode indexing, geometric
  matrices (smats/tmats/xmats/ymats/zmats), and per-surface θ-grid buffers
- `kinetic_profiles::Equilibrium.KineticProfileSplines`: Named kinetic-profile splines

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function compute_kinetic_matrices_at_psi!(
    kwmat::Array{ComplexF64,3},
    ktmat::Array{ComplexF64,3},
    psi::Float64, n::Int, l::Int,
    zi::Int, mi::Int, wdfac::Float64, _divxfac::Float64,
    electron::Bool, equil, intr::KineticForcesInternal,
    kinetic_profiles::Equilibrium.KineticProfileSplines;
    nutype::String="harmonic", f0type::String="maxwellian", nufac::Float64=1.0,
    atol_xlmda::Float64=1e-9, rtol_xlmda::Float64=1e-6)

    # Bypass ψ > 1 (no kinetic contribution outside plasma)
    if psi > 1
        fill!(kwmat, 0)
        fill!(ktmat, 0)
        return nothing
    end

    state = _setup_surface_state(psi, zi, mi, electron,
                                  equil, intr, kinetic_profiles)

    kinetic_energy_matrices_for_euler_lagrange!(
        kwmat, ktmat, state, psi, n, l, wdfac, intr;
        nutype, f0type, nufac,
        energy_atol=atol_xlmda, energy_rtol=rtol_xlmda,
        pitch_atol=atol_xlmda, pitch_rtol=rtol_xlmda)

    return nothing
end


