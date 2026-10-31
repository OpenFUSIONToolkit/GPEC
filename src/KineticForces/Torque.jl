"""
    tpsi!(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method, equil, intr,
          kinetic_profiles; op_wmats=nothing, surface=nothing, kwargs...) -> nothing

Toroidal torque from nonambipolar transport at one ψ and bounce harmonic `l`, written to
`tpsi_var[]`. The imaginary part carries the kinetic energy, Im(T) = 2n·δW_k. Zero for ψ > 1.

# Arguments

  - `tpsi_var::Ref{ComplexF64}`: Output torque
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
  - `surface`: Per-surface bundle from `_torque_surface`, shared across the bounce harmonics of
    one ψ; built here when not given

"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int,
    zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
    electron::Bool, method::String, equil::Equilibrium.PlasmaEquilibrium, intr::KineticForcesInternal,
    kinetic_profiles::Equilibrium.KineticProfileSplines;
    op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing,
    surface::Union{Nothing,NamedTuple}=nothing,
    rex_override::Union{Nothing,Float64}=nothing,
    imx_override::Union{Nothing,Float64}=nothing,
    atol_xlmda::Float64=1e-9, rtol_xlmda::Float64=1e-6,
    atol_x::Float64=NaN, rtol_x::Float64=NaN,
    nested_tolerance_margin::Float64=1e-2)

    # The pitch integrand is itself the energy integral, so the energy level is resolved
    # tighter than the pitch level (explicit atol_x/rtol_x override the derived values).
    atol_energy = isnan(atol_x) ? nested_tolerance_margin * atol_xlmda : atol_x
    rtol_energy = isnan(rtol_x) ? nested_tolerance_margin * rtol_xlmda : rtol_x

    # Enforce bounds
    if psi > 1
        tpsi_var[] = 0.0
        return
    end

    # Everything up to the method dispatch depends on ψ alone, so the caller may share one
    # bundle across the bounce harmonics of a surface.
    surf = surface === nothing ?
           _torque_surface(psi, zi, mi, electron, method, equil, intr, kinetic_profiles;
        need_matrices=(!isnothing(op_wmats)), n, wdfac, divxfac) :
           surface
    (; chrg, mass, tspl, bmax, bmin, q, n_s, T_s, welec, wdian, wdiat, wtran, wgyro, nuk, epsr, dVdpsi, dbob_m_f, divx_m_f) = surf
    wbhat = (π / 4) * sqrt(epsr / 2) * wtran
    wdhat = q^3 * wtran^2 / (4 * epsr * wgyro) * wdfac
    nueff = nuk / (2 * epsr)

    # Method selection — route on the registry dispatch tag (errors on unknown method)
    kind = method_kind(method)
    if kind == :fcgl
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac,
            n_s, T_s, equil, intr)

    elseif kind == :rlar
        tpsi_var[] = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
            wdhat, wbhat, nueff, dVdpsi, n_s, T_s,
            dbob_m_f, intr.bo, bmin)

    elseif kind == :clar
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
            nuk, intr.bo, bmax, bmin, n_s, T_s, mass, chrg,
            tspl, dbob_m_f, divx_m_f, divxfac, wdfac)

    else  # :gar — fgar/tgar/pgar + all *mm matrix methods
        tpsi_var[] = calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec,
            nuk, intr.bo, bmax, bmin, n_s, T_s, mass, chrg,
            tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
            method, op_wmats;
            chi1=intr.chi1, ro=intr.ro, mfac=intr.mfac,
            mpert=intr.mpert, theta_bmax=surf.theta_bmax,
            B_vpar=surf.B_vpar, surface_field=surf.bf,
            smat=surf.smat_f, tmat=surf.tmat_f, xmat=surf.xmat_f,
            ymat=surf.ymat_f, zmat=surf.zmat_f,
            energy_atol=atol_energy, energy_rtol=rtol_energy,
            pitch_atol=atol_xlmda, pitch_rtol=rtol_xlmda,
            rex_override=rex_override, imx_override=imx_override,
            bounce_geometry=get(surf, :geometry, nothing))
    end

    return nothing
end


# ============================================================================
# Method-specific torque calculation functions
# ============================================================================

"""
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac, n_s, T_s,
                   equil, intr)::ComplexF64

Fluid Chew–Goldberger–Low limit of the kinetic energy: the double-adiabatic δW built from
`∇·ξ⊥` and `κ·ξ⊥ = -(δB/B + ∇·ξ⊥)` on the surface, returned as `2n·i·δW` so `Im` carries the
energy like the kinetic methods. Only the l = 0 harmonic is nonzero.

Based on: [Logan et al., Phys. Plasmas 2013]
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, divxfac::Float64,
    n_s::Float64, T_s::Float64, equil,
    intr::KineticForcesInternal)::ComplexF64

    # Only implemented for l=0.
    if l != 0
        return ComplexF64(0.0, 0.0)
    end

    mthsurf_local = intr.mthsurf

    # Poloidal integrands sampled on the θ grid
    cglspl = zeros(2, mthsurf_local + 1)
    theta_vals = range(0, 1; length=mthsurf_local + 1)

    # Local search hints: ψ nodes and harmonics run concurrently, so `intr` is read-only here.
    hB = (Ref(1), Ref(1))
    hJ = (Ref(1), Ref(1))
    for i in eachindex(theta_vals)
        theta = theta_vals[i]

        B_val = equil.eqfun_B((psi, theta); hint=hB)
        jac_val = equil.rzphi_jac((psi, theta); hint=hJ)

        # Fourier component of field perturbations
        expm = exp.(im * twopi * intr.mfac * theta)
        dbob = sum(dbob_m_f .* expm)  # dB/B
        divx = sum(divx_m_f .* expm) * divxfac  # ∇·ξ⊥

        kapx = -(dbob + divx)  # κ·ξ⊥ from δB/B = -(∇·ξ⊥ + κ·ξ⊥), the convention dbob_m is built with

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
             (0.5 * (5.0 / 3.0) * integral_1 +
              0.5 * (1.0 / 3.0) * integral_2)

    return result
end


"""
    calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, wdhat, wbhat,
                   nueff, dVdpsi, n_s, T_s, dbob_m_f, bo, bmin)::ComplexF64

RLAR (Reduced Large Aspect Ratio) torque: energy-space integration with a pitch-averaged
bounce response. Valid for large aspect ratio (ε ≪ 1); the κ bounce average is a placeholder estimate.

Reference: [Logan et al., Phys. Plasmas, 2013]
"""
function calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
    wdhat, wbhat, nueff, dVdpsi::Float64,
    n_s::Float64, T_s::Float64,
    dbob_m_f, bo, bmin=0.5)::ComplexF64

    # Setup parameters for energy integration
    lnq = Float64(l)  # Resonant mode for trapped particles

    # Maximum pitch angle parameter (use safe estimate)
    lmdamax = min(1.0 / (1 - epsr), bo / bmin)

    # Energy-space quadrature ∫ K(x) dx of the resonance operator K
    xint = integrate_energy(wdian, wdiat, welec, wdhat, wbhat, nueff,
        l, lnq, n, psi, lmdamax, "rlar")

    # Kappa/bounce averaging (effect of field perturbations)
    # Placeholder: simplified estimate
    kappaint_val = sqrt(mean(abs.(dbob_m_f) .^ 2))

    # dV/dpsi gradient term from Clebsch coordinate Jacobian
    psi_factor = dVdpsi

    # Normalization: √(ε/(2π³)) * n² * n_s * T_s
    norm = sqrt(epsr / (2.0 * π^3)) * Float64(n)^2 * n_s * T_s

    # dT/dψ = -0.5 · dV/dψ · κ_int · x_int · norm
    result = psi_factor * kappaint_val * 0.5 * (-xint) * norm

    return result
end


"""
    calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                   bmax, bmin, n_s, T_s, mass, chrg, tspl, dbob_m_f, divx_m_f,
                   divxfac, wdfac)::ComplexF64

CLAR (Circular Large Aspect Ratio) torque. Not implemented: warns once and returns zero.
"""
function calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
    bmax, bmin, n_s::Float64, T_s::Float64, mass, chrg, tspl,
    dbob_m_f, divx_m_f, divxfac, wdfac)::ComplexF64

    @warn "CLAR method not yet fully implemented, returning zero" maxlog = 1
    return ComplexF64(0.0, 0.0)
end


"""
    calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo, bmax,
                  bmin, n_s, T_s, mass, chrg, tspl, dbob_m_f, divx_m_f,
                  divxfac, wdfac, method, op_wmats; kwargs...)::ComplexF64

GAR (General Aspect Ratio) torque, with no aspect-ratio expansion: bounce averages over λ, then
an adaptive pitch/energy integration and the Eq. 19 normalization of Logan et al. 2013. Variants
FGAR/TGAR/PGAR (full/trapped/passing); the `*tmm`/`*wmm`/`*kmm`/`*rmm` methods also fill
`op_wmats` with the six kinetic matrices (A, B, C, D, E, H).

# Keyword Arguments

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
    theta_bmax::Float64, B_vpar,
    smat=nothing, tmat=nothing, xmat=nothing,
    ymat=nothing, zmat=nothing,
    nlmda::Int=128, ntheta::Int=128,
    nutype::String="harmonic", f0type::String="maxwellian",
    nufac::Float64=1.0, qt::Bool=false,
    energy_atol::Float64=1e-7, energy_rtol::Float64=1e-5,
    pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6,
    rex_override::Union{Nothing,Float64}=nothing,
    imx_override::Union{Nothing,Float64}=nothing,
    surface_field::Union{Nothing,SurfaceBField}=nothing,
    bounce_geometry::Union{Nothing,BounceGeometry}=nothing)::ComplexF64

    # Bounce-averaged quantities per pitch angle; the harmonic-free geometry is shared across ℓ when given.
    bounce =
        bounce_geometry === nothing ?
        compute_bounce_data(
            psi, n, l, q, bo, bmax, bmin, theta_bmax,
            tspl, B_vpar, mfac, chi1, ro, dbob_m_f, divx_m_f, divxfac, wdfac,
            mass, chrg, T_s, method;
            nlmda, ntheta, smat, tmat, xmat, ymat, zmat, surface_field) :
        bounce_harmonic(bounce_geometry, l, n, q, ro, mpert; smat, tmat, xmat, ymat, zmat)

    if bounce.nlmda == 0
        return ComplexF64(0.0, 0.0)
    end

    # Build fbnce interpolant: [wb, wd, f₁, f₂, ...]
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
        # Packed matrix block at columns 4:(3+nqty_mat); this torque path keeps only its real part.
        @inbounds for q in 1:nqty_mat, ilmda in 1:bounce.nlmda
            fbnce_data[ilmda, 3+q] = real(bounce.wmats_vs_lambda[ilmda, q])
        end
    end

    # Normalize flux quantities by 1/median for numerical stability; rescale in place
    # before building the interpolant.
    fbnce_norm = ones(Float64, nqty)
    for i in 1:nqty
        col = view(fbnce_data, :, i + 2)
        med = median(abs.(col))
        if med > 0
            fbnce_norm[i] = 1.0 / med
            col .*= fbnce_norm[i]
        end
    end

    # CubicFit endpoint BC matches the extrap fit.
    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=CubicFit())

    # rex/imx multipliers select torque-only, energy-only, or full complex output.
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

    # Pitch angle integration (adaptive Gauss-Kronrod over λ)
    lxint = integrate_pitch_gar_quadgk(
        wdian, wdiat, welec, nuk, bo / bmax, epsr, q,
        fbnce, fbnce_norm, nqty, l, n, rex, imx, psi, method;
        nutype, f0type, nufac, qt,
        energy_atol, energy_rtol, pitch_atol, pitch_rtol)

    # Scalar torque, Eq. (19) of Logan et al., Phys. Plasmas 20, 122507 (2013).
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * n_s * T_s *
            (chi1 / twopi)  # unit conversion from ψ to ψ_n, θ_n to θ
    tpsi_val = tnorm * (lxint[1] / fbnce_norm[1])

    # Kinetic matrix assembly (A,B,C,D,E,H)
    if do_matrices
        op_wmats .= 0
        energy_factor = tnorm * (1 / (2 * im * n))  # convert torque → energy
        Mu = (mpert * (mpert + 1)) ÷ 2
        base = 1  # lxint offset after scalar-torque slot (index 1)

        # Helper: fetch normalized pitch integral at lxint[base+q] for q ∈ [1..nqty_mat].
        @inline elem(q) = complex(lxint[base+q] / fbnce_norm[base+q]) * energy_factor

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

        # DCON normalization
        op_wmats .*= 2 * μ₀
        op_wmats[:, :, 1:3] ./= chi1
        op_wmats[:, :, 1] ./= chi1

        # Method-specific output
        if method_suffix in ["kmm", "rmm"]
            # Matrix norms independent of ξ
            tpsi_val = ComplexF64(0.0)
            for k in 1:6
                # Spectral norm via SVD
                mat = op_wmats[:, :, k]
                sv = svdvals(mat' * mat)
                tpsi_val += maximum(real.(sv))
            end
            tpsi_val = complex(rex + imx * im) * sqrt(abs(tpsi_val))

        elseif method_suffix in ["wmm", "tmm", "pmm"]
            # No ξ contraction of the matrices: they are stored and the scalar torque is returned.
        end
    end

    return tpsi_val
end


"""
    _setup_surface_state(psi, zi, mi, electron, equil, intr, kinetic_profiles) → NamedTuple

Per-surface setup shared by the torque and calculated-matrix paths: θ-grid sampling of B and
the Jacobian, the periodic fits (`tspl`, `B_vpar`) and the decomposed field `bf`, bounce
extrema, flux-function values and the diamagnetic/drift frequencies. Nothing here depends on
the toroidal mode or the bounce harmonic, so one bundle serves every (n, ℓ) at a ψ. Reads `intr`
only (own θ buffers and search hints), so surfaces of different ψ may be built concurrently.
"""
function _setup_surface_state(
    psi::Float64, zi::Int, mi::Int,
    electron::Bool, equil, intr::KineticForcesInternal,
    kinetic_profiles::Equilibrium.KineticProfileSplines
)
    if electron
        chrg = -1 * e
        mass = me
    else
        chrg = zi * e
        mass = mi * mp
    end

    mthsurf_local = intr.mthsurf
    xs = intr.tpsi_xs
    B_vals = Vector{Float64}(undef, mthsurf_local + 1)
    dBdpsi_vals = similar(B_vals)
    dBdtheta_vals = similar(B_vals)
    jac_vals = similar(B_vals)
    djdpsi_vals = similar(B_vals)

    hB = (Ref(1), Ref(1))
    hJ = (Ref(1), Ref(1))
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        pt = (psi, theta)
        B_vals[i+1] = equil.eqfun_B(pt; hint=hB)
        dBdpsi_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(1, 0), hint=hB) / intr.chi1
        dBdtheta_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(0, 1), hint=hB)
        jac_vals[i+1] = equil.rzphi_jac(pt; hint=hJ) / intr.chi1
        djdpsi_vals[i+1] = equil.rzphi_jac(pt; deriv=DerivOp(1, 0), hint=hJ) / intr.chi1^2
    end

    # The repeated periodic endpoint must match exactly, including sampled derivatives.
    for values in (B_vals, dBdpsi_vals, dBdtheta_vals, jac_vals, djdpsi_vals)
        values[end] = values[1]
    end

    tspl = cubic_interp(xs, Series(hcat(B_vals, dBdpsi_vals, dBdtheta_vals, jac_vals, djdpsi_vals)); bc=PeriodicBC())
    # Periodic cubic of B for v_par and bounce points (Fortran vspl equivalent); 1−(λ/bo)B
    # is periodic on the closed θ interval, so a non-periodic fit would break at the seam.
    B_vpar = cubic_interp(xs, B_vals; bc=PeriodicBC())

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
    for θ in find_sign_change_roots(dBdtheta_interp, xs)
        θn = mod(θ, 1.0)
        Bθ = tspl(θn)[1]
        if Bθ < bmin
            bmin = Bθ
            theta_bmin = θn
        end
        if Bθ > bmax
            bmax = Bθ
            # theta_bmax stays at the nodal knot xs[ibmax]: the transit starts at
            # the knot (as in Fortran), not at the refined extremum.
        end
    end

    q = equil.profiles.q_spline(psi)
    if electron
        n_s = kinetic_profiles.ne_spline(psi)
        T_s = kinetic_profiles.Te_spline(psi)
        dn_s_dpsi = kinetic_profiles.ne_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Te_deriv(psi)
        nu_s = kinetic_profiles.nue_spline(psi)
    else
        n_s = kinetic_profiles.ni_spline(psi)
        T_s = kinetic_profiles.Ti_spline(psi)
        dn_s_dpsi = kinetic_profiles.ni_deriv(psi)
        dT_s_dpsi = kinetic_profiles.Ti_deriv(psi)
        nu_s = kinetic_profiles.nui_spline(psi)
    end
    welec = kinetic_profiles.omegaE_spline(psi)

    wdian = -twopi * T_s * dn_s_dpsi / (chrg * intr.chi1 * n_s)
    wdiat = -twopi * dT_s_dpsi / (chrg * intr.chi1)
    wtran = sqrt(2 * T_s / mass) / (q * intr.ro)
    wgyro = chrg * intr.bo / mass
    nuk = nu_s

    rsquared_bmin = equil.rzphi_rsquared((psi, theta_bmin))
    rsquared_bmin > 0 ||
        error("ERROR: _setup_surface_state - minor radius is negative at psi=$psi, theta_bmin=$theta_bmin")

    avg_r = equil.geometry.avg_r_spline(psi)
    avg_R = equil.geometry.avg_R_spline(psi)
    epsr = avg_r / avg_R
    dVdpsi = equil.profiles.dVdpsi_spline(psi)

    # Cells and stationary points of B(θ), reused by every λ of every harmonic.
    bf = _surface_b_field(B_vpar)

    return (;
        chrg, mass,
        tspl, B_vpar, bf, bmax, bmin, theta_bmax,
        q, n_s, T_s, welec,
        wdian, wdiat, wtran, wgyro, nuk,
        epsr, dVdpsi
    )
end

"""
    _torque_surface(psi, zi, mi, electron, method, equil, intr, kinetic_profiles;
                    need_matrices=false, n=nothing, wdfac=1.0, divxfac=1.0) → NamedTuple

`_setup_surface_state` plus the perturbation modes and, when `need_matrices`, the geometric
matrices at ψ: everything `tpsi!` needs before its method dispatch. With the toroidal mode `n`
given, a GAR-family method also gets its harmonic-free bounce geometry (`geometry`), shared by
every bounce harmonic of the surface; otherwise `geometry` is `nothing`. Reads `intr` only.
"""
function _torque_surface(psi::Float64, zi::Int, mi::Int, electron::Bool, method::String,
    equil, intr::KineticForcesInternal, kinetic_profiles::Equilibrium.KineticProfileSplines;
    need_matrices::Bool=false, n::Union{Nothing,Int}=nothing, wdfac::Float64=1.0, divxfac::Float64=1.0)
    state = _setup_surface_state(psi, zi, mi, electron, equil, intr, kinetic_profiles)

    # Matrix-only methods build their operator from the geometric matrices; dbob_m / divx_m
    # only feed the scalar dJdJ normalisation, so a run without perturbations passes zeros.
    if occursin("mm", method) && need_matrices && isnothing(intr.dbob_m)
        dbob_m_f = zeros(ComplexF64, intr.mpert)
        divx_m_f = zeros(ComplexF64, intr.mpert)
    else
        dbob_m_f = intr.dbob_m(psi; hint=Ref(1))
        divx_m_f = intr.divx_m(psi; hint=Ref(1))
    end

    if need_matrices && !isnothing(intr.smats)
        k = (n === nothing ? intr.nlow : n) - intr.nlow + 1
        smat_f = ForceFreeStates.kinetic_metric_block(intr.smats, psi, intr.mpert, k)
        tmat_f = ForceFreeStates.kinetic_metric_block(intr.tmats, psi, intr.mpert, k)
        xmat_f = ForceFreeStates.kinetic_metric_block(intr.xmats, psi, intr.mpert, k)
        ymat_f = ForceFreeStates.kinetic_metric_block(intr.ymats, psi, intr.mpert, k)
        zmat_f = ForceFreeStates.kinetic_metric_block(intr.zmats, psi, intr.mpert, k)
    else
        smat_f = tmat_f = xmat_f = ymat_f = zmat_f = nothing
    end

    geometry = if n !== nothing && method_kind(method) == :gar
        compute_bounce_geometry(psi, n, state.q, intr.bo, state.bmax, state.bmin, state.theta_bmax,
            state.tspl, state.B_vpar, intr.mfac, intr.chi1, intr.ro, dbob_m_f, divx_m_f, divxfac, wdfac,
            state.mass, state.chrg, state.T_s, method;
            do_matrices=need_matrices && !isnothing(smat_f), surface_field=state.bf)
    else
        nothing
    end

    return (; state..., dbob_m_f, divx_m_f, smat_f, tmat_f, xmat_f, ymat_f, zmat_f, geometry)
end

"""
    _matrix_bounce_geometry(state, psi, n, wdfac, intr; nlmda=128, ntheta=128) → BounceGeometry

Harmonic-free bounce geometry of the calculated-matrix path at (ψ, n): the W_μ/W_E samples of the
full λ range, with no perturbation modes (the matrix path discards the scalar |δJ|²).
"""
function _matrix_bounce_geometry(state::NamedTuple, psi::Float64, n::Int, wdfac::Float64, intr::KineticForcesInternal;
    nlmda::Int=128, ntheta::Int=128)
    no_modes = zeros(ComplexF64, intr.mpert)
    return compute_bounce_geometry(psi, n, state.q, intr.bo, state.bmax, state.bmin, state.theta_bmax,
        state.tspl, state.B_vpar, intr.mfac, intr.chi1, intr.ro, no_modes, no_modes, 1.0, wdfac,
        state.mass, state.chrg, state.T_s, "fwmm";
        nlmda, ntheta, do_matrices=true, surface_field=state.bf)
end


"""
    kinetic_energy_matrices_for_euler_lagrange!(kwmat, ktmat, state, psi, n, l, wdfac, intr;
                                                kwargs...) → nothing

Compute the six kinetic Euler-Lagrange coefficient matrices (A, B, C, D, E, H) of Logan 2015
Eqs 7.30–7.35 at one (ψ, n, ℓ) into pre-allocated `kwmat[mpert, mpert, 6]` (fwmm half, Fortran
rex=0, imx=1) and `ktmat[mpert, mpert, 6]` (ftmm half, rex=1, imx=0). One energy integration per
(λ, E) via `integrate_pitch_gar_quadgk_wt` yields both halves. The upper-triangle-stored A/D/H
blocks mirror as Hermitian in `kwmat` (`kwmat[j,i] = conj(kwmat[i,j])`) and anti-Hermitian in
`ktmat` (`ktmat[j,i] = -conj(ktmat[i,j])`).
"""
function kinetic_energy_matrices_for_euler_lagrange!(
    kwmat::Array{ComplexF64,3},
    ktmat::Array{ComplexF64,3},
    state::NamedTuple,
    psi::Float64, n::Int, l::Int, wdfac::Float64,
    intr::KineticForcesInternal;
    nlmda::Int=128, ntheta::Int=128,
    nutype::String="harmonic", f0type::String="maxwellian",
    nufac::Float64=1.0,
    energy_atol::Float64=1e-9, energy_rtol::Float64=1e-6,
    pitch_atol::Float64=1e-9, pitch_rtol::Float64=1e-6,
    geometry::Union{Nothing,BounceGeometry}=nothing
)
    mpert = intr.mpert
    mfac = intr.mfac
    chi1 = intr.chi1
    ro = intr.ro
    bo = intr.bo

    # Geometric matrices of this n at this ψ
    k = n - intr.nlow + 1
    smat_f = ForceFreeStates.kinetic_metric_block(intr.smats, psi, mpert, k)
    tmat_f = ForceFreeStates.kinetic_metric_block(intr.tmats, psi, mpert, k)
    xmat_f = ForceFreeStates.kinetic_metric_block(intr.xmats, psi, mpert, k)
    ymat_f = ForceFreeStates.kinetic_metric_block(intr.ymats, psi, mpert, k)
    zmat_f = ForceFreeStates.kinetic_metric_block(intr.zmats, psi, mpert, k)

    # Matrix-only path: dbob/divx perturbation coefficients are not used
    # (compute_bounce_data folds them into the scalar dJdJ we discard).
    dbob_m_f = zeros(ComplexF64, mpert)
    divx_m_f = zeros(ComplexF64, mpert)

    bounce =
        geometry === nothing ?
        compute_bounce_data(
            psi, n, l, state.q, bo, state.bmax, state.bmin, state.theta_bmax,
            state.tspl, state.B_vpar, mfac, chi1, ro, dbob_m_f, divx_m_f, 1.0, wdfac,
            state.mass, state.chrg, state.T_s, "fwmm";
            nlmda, ntheta, smat=smat_f, tmat=tmat_f, xmat=xmat_f, ymat=ymat_f, zmat=zmat_f,
            surface_field=state.bf) :
        bounce_harmonic(geometry, l, n, state.q, ro, mpert; smat=smat_f, tmat=tmat_f, xmat=xmat_f, ymat=ymat_f, zmat=zmat_f)

    if bounce.nlmda == 0
        fill!(kwmat, 0)
        fill!(ktmat, 0)
        return nothing
    end

    # Pack wb, wd and the matrix block (no scalar slot); complex because B/C/E carry genuine phase.
    nqty = nqty_matrix(mpert)
    fbnce_data = zeros(ComplexF64, bounce.nlmda, 2 + nqty)
    fbnce_data[:, 1] .= bounce.wb
    fbnce_data[:, 2] .= bounce.wd
    # wmats_vs_lambda is already (nlmda, nqty) in the packed layout.
    fbnce_data[:, 3:end] .= bounce.wmats_vs_lambda

    # Column-wise median normalization for numerical stability.
    fbnce_norm = ones(Float64, nqty)
    for i in 1:nqty
        col = view(fbnce_data, :, i + 2)
        med = median(abs.(col))
        if med > 0
            fbnce_norm[i] = 1.0 / med
            col .*= fbnce_norm[i]
        end
    end

    # CubicFit endpoint BC matches the extrap fit.
    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=CubicFit())

    # One energy sweep per (λ, E) gives both halves, packed as [wmm | tmm], each length nqty.
    lxint = integrate_pitch_gar_quadgk_wt(
        state.wdian, state.wdiat, state.welec, state.nuk,
        bo / state.bmax, state.epsr, state.q,
        fbnce, fbnce_norm, nqty, l, n, psi, "fwmm";
        nutype, f0type, nufac, qt=false,
        energy_atol, energy_rtol, pitch_atol, pitch_rtol)

    # Torque → energy normalization, Eq. (19) of Logan et al., Phys. Plasmas 20, 122507 (2013).
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * state.n_s * state.T_s *
            (chi1 / twopi)
    energy_factor = tnorm * (1 / (2 * im * n))

    Mu = (mpert * (mpert + 1)) ÷ 2
    # Normalized element at slot q; half_offset = 0 for wmm (kwmat), nqty for tmm (ktmat).
    @inline elem(q, half_offset) = complex(lxint[half_offset+q] / fbnce_norm[q]) * energy_factor

    # A/D/H lower triangles: Hermitian mirror for kwmat (S_w imaginary), anti-Hermitian for ktmat (S_t real).
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
        _assemble_hermitian!(dest, 1, off, half, mirror_sign)
        off += Mu        # A (k=1)
        _assemble_hermitian!(dest, 4, off, half, mirror_sign)
        off += Mu        # D (k=4)
        _assemble_hermitian!(dest, 6, off, half, mirror_sign)
        off += Mu        # H (k=6)
        _assemble_full!(dest, 2, off, half)
        off += mpert^2   # B (k=2)
        _assemble_full!(dest, 3, off, half)
        off += mpert^2   # C (k=3)
        _assemble_full!(dest, 5, off, half)                                      # E (k=5)
    end

    # DCON normalization
    for mat in (kwmat, ktmat)
        mat .*= 2 * μ₀
        mat[:, :, 1:3] ./= chi1
        mat[:, :, 1] ./= chi1
    end

    return nothing
end


"""
    compute_kinetic_matrices_at_psi!(kwmat, ktmat, psi, n, l, zi, mi,
        wdfac, electron, equil, intr, kinetic_profiles)

Compute the six kinetic Euler-Lagrange coefficient matrices (Logan 2015 Eqs 7.30–7.35) at one flux
surface, split into `kwmat` (fwmm half, Fortran rex=0, imx=1) and `ktmat` (ftmm half, rex=1, imx=0)
as the ForceFreeStates kinetic assembly expects for its `kwmat ± ktmat` combinations. Both halves
are complex and come from a single integration. Zero for ψ > 1.

# Arguments

  - `kwmat::Array{ComplexF64,3}`: Output (mpert×mpert×6), fwmm half, overwritten
  - `ktmat::Array{ComplexF64,3}`: Output (mpert×mpert×6), ftmm half, overwritten
  - `psi, n, l, zi, mi, wdfac, electron`: Same as `tpsi!`
  - `equil`: PlasmaEquilibrium
  - `intr::KineticForcesInternal`: Internal state with mode indexing, geometric
    matrices (smats/tmats/xmats/ymats/zmats), and per-surface θ-grid buffers
  - `kinetic_profiles::Equilibrium.KineticProfileSplines`: Named kinetic-profile splines
  - `state`, `geometry` (keywords): precomputed `_setup_surface_state` bundle and bounce geometry
    to share across harmonics; built here when `nothing`

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function compute_kinetic_matrices_at_psi!(
    kwmat::Array{ComplexF64,3},
    ktmat::Array{ComplexF64,3},
    psi::Float64, n::Int, l::Int,
    zi::Int, mi::Int, wdfac::Float64,
    electron::Bool, equil::Equilibrium.PlasmaEquilibrium, intr::KineticForcesInternal,
    kinetic_profiles::Equilibrium.KineticProfileSplines;
    nutype::String="harmonic", f0type::String="maxwellian", nufac::Float64=1.0,
    atol_xlmda::Float64=1e-9, rtol_xlmda::Float64=1e-6,
    atol_x::Float64=NaN, rtol_x::Float64=NaN,
    nested_tolerance_margin::Float64=1e-2,
    state::Union{Nothing,NamedTuple}=nothing,
    geometry::Union{Nothing,BounceGeometry}=nothing)

    # Bypass ψ > 1 (no kinetic contribution outside plasma)
    if psi > 1
        fill!(kwmat, 0)
        fill!(ktmat, 0)
        return nothing
    end

    # See tpsi!: the energy integral is the pitch integrand, so it is resolved tighter.
    atol_energy = isnan(atol_x) ? nested_tolerance_margin * atol_xlmda : atol_x
    rtol_energy = isnan(rtol_x) ? nested_tolerance_margin * rtol_xlmda : rtol_x

    # The surface setup is (n, ℓ)-independent; a caller looping harmonics passes it in once.
    surf = state === nothing ? _setup_surface_state(psi, zi, mi, electron, equil, intr, kinetic_profiles) : state

    kinetic_energy_matrices_for_euler_lagrange!(
        kwmat, ktmat, surf, psi, n, l, wdfac, intr;
        nutype, f0type, nufac,
        energy_atol=atol_energy, energy_rtol=rtol_energy,
        pitch_atol=atol_xlmda, pitch_rtol=rtol_xlmda, geometry)

    return nothing
end
