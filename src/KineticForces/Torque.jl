"""
    tpsi!(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method, equil, intr;
         op_wmats=nothing)

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
- `equil`: PlasmaEquilibrium with 2D interpolants
- `intr::KineticForcesInternal`: Internal state with profile interpolants and geometry

# Optional Arguments
- `op_wmats::Array{ComplexF64,3}`: Store ForceFreeStates matrix elements

# Returns
- `ComplexF64`: Toroidal torque due to nonambipolar transport
"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int,
              zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
              electron::Bool, method::String, equil, intr::KineticForcesInternal;
              op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing,
              rex_override::Union{Nothing,Float64}=nothing,
              imx_override::Union{Nothing,Float64}=nothing)

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

    # Get perturbations (CubicSeriesInterpolant callable syntax)
    dbob_m_f = intr.dbob_m(psi)
    divx_m_f = intr.divx_m(psi)

    # Sample poloidal quantities on theta grid
    mthsurf_local = intr.mthsurf
    xs = collect(range(0.0, 1.0, length=mthsurf_local + 1))
    B_vals = Vector{Float64}(undef, mthsurf_local + 1)
    dBdpsi_vals = Vector{Float64}(undef, mthsurf_local + 1)
    dBdtheta_vals = Vector{Float64}(undef, mthsurf_local + 1)
    jac_vals = Vector{Float64}(undef, mthsurf_local + 1)
    djdpsi_vals = Vector{Float64}(undef, mthsurf_local + 1)

    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        pt = (psi, theta)

        B_vals[i+1] = equil.eqfun_B(pt)
        dBdpsi_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(1, 0)) / intr.chi1
        dBdtheta_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(0, 1))
        jac_vals[i+1] = equil.rzphi_jac(pt) / intr.chi1
        djdpsi_vals[i+1] = equil.rzphi_jac(pt; deriv=DerivOp(1, 0)) / intr.chi1^2
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

    # Get flux function variables (CubicSeriesInterpolant callable syntax)
    sq_s_f = intr.sq(psi)
    kin_f = intr.kin(psi)
    kin_f1 = intr.kin(psi; deriv=1)
    geom_f = intr.geom(psi)

    q = sq_s_f[4]
    welec = kin_f[5]
    wdian = -twopi * kin_f[s+2] * kin_f1[s] / (chrg * intr.chi1 * kin_f[s])
    wdiat = -twopi * kin_f1[s+2] / (chrg * intr.chi1)
    wphi = welec + wdian + wdiat
    wtran = sqrt(2 * kin_f[s+2] / mass) / (q * intr.ro)
    wgyro = chrg * intr.bo / mass
    nuk = kin_f[s+6]

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

    epsr = geom_f[2] / geom_f[3]
    wbhat = (π / 4) * sqrt(epsr / 2) * wtran
    wdhat = q^3 * wtran^2 / (4 * epsr * wgyro) * wdfac
    nueff = kin_f[s+6] / (2 * epsr)

    if intr.verbose
        @printf("   eq values = %.1e %.1e %.1e %.1e %.1e %.1e %.1f %d\n",
               wdian, wdiat, welec, wdhat, wbhat, nueff, q, 0)
    end

    if intr.verbose
        println("  method = ", method)
    end

    # Method selection
    if method == "fcgl"
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil, intr)

    elseif method == "rlar"
        tpsi_var[] = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    wdhat, wbhat, nueff, sq_s_f, kin_f, s,
                                    dbob_m_f, intr.bo, bmin)

    elseif method == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, intr.bo, bmax, bmin, kin_f, s, mass, chrg,
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
                                   nuk, intr.bo, bmax, bmin, kin_f, s, mass, chrg,
                                   tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
                                   method, op_wmats;
                                   chi1=intr.chi1, ro=intr.ro, mfac=intr.mfac,
                                   mpert=intr.mpert, ibmax=ibmax, theta_bmax=theta_bmax,
                                   smat=smat_f, tmat=tmat_f, xmat=xmat_f,
                                   ymat=ymat_f, zmat=zmat_f,
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
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil, intr)::ComplexF64

Calculate FCGL (Full Circular Gyrokinetic Landau) torque.
Implements simplified energy balance equation.
Only valid for bounce harmonic l=0.

Based on: [Logan et al., Phys. Plasmas 2013]
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil, intr::KineticForcesInternal)::ComplexF64

    # Only implemented for l=0
    if l != 0
        return ComplexF64(0.0, 0.0)
    end

    mthsurf_local = intr.mthsurf

    # Create spline for poloidal integrands
    cglspl = zeros(2, mthsurf_local + 1)
    theta_vals = range(0, 1, length=mthsurf_local + 1)

    # Evaluate poloidal functions and field perturbations
    for i in eachindex(theta_vals)
        theta = theta_vals[i]

        B_val = equil.eqfun_B((psi, theta))
        jac_val = equil.rzphi_jac((psi, theta))

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

    # Calculate torque: T = 2*n*i*n_i*T_i * [weighted integral]
    result = 2.0 * n * im * kin_f[s] * kin_f[s+2] *
        (0.5 * (5.0/3.0) * integral_1 +
         0.5 * (1.0/3.0) * integral_2)

    return result
end


"""
    calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, wdhat, wbhat,
                   nueff, sq_s_f, kin_f, s, dbob_m_f, bmin)::ComplexF64

Calculate RLAR (Reduced Large Aspect Ratio) torque.
Uses energy space integration with pitch angle averaging.
Valid for low aspect ratio tokamaks (ε << 1).

Reference: [Logan et al., Phys. Plasmas, 2013]
"""
function calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
                        wdhat, wbhat, nueff, sq_s_f, kin_f, s, dbob_m_f, bo, bmin=0.5)::ComplexF64

    # Setup parameters for energy integration
    lnq = Float64(l)  # Resonant mode for trapped particles

    # Maximum pitch angle parameter (use safe estimate)
    lmdamax = min(1.0/(1-epsr), bo/bmin)

    # Energy space ODE integration
    # This computes ∫ K(x) dx where K is the resonance operator
    xint = integrate_energy_ode(wdian, wdiat, welec, wdhat, wbhat, nueff,
                         l, lnq, n, psi, lmdamax, "rlar")

    # Kappa/bounce averaging (effect of field perturbations)
    # Placeholder: simplified estimate
    kappaint_val = sqrt(mean(abs.(dbob_m_f).^2))

    # dψ/dpsi gradient term from magnetic geometry
    psi_factor = sq_s_f[3]  # From Clebsch coordinate Jacobian

    # Normalization: √(ε/(2π³)) * n² * n_i * T_i
    norm = sqrt(epsr / (2.0 * π^3)) * Float64(n)^2 * kin_f[s] * kin_f[s+2]

    # Result: dT/dpsi = -(dψ/dpsi) * κ_int * x_int * norm
    result = psi_factor * kappaint_val * 0.5 * (-xint) * norm

    return result
end


"""
    calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                   bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f, divx_m_f, divxfac, wdfac)::ComplexF64

Calculate CLAR (Circular Large Aspect Ratio) torque.
Uses pitch-angle resolved calculations for trapped and passing particles.
Includes bounce-averaged integrals over lambda (pitch angle).

Status: Partially implemented (stub for full calculation)
"""
function calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                        bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f,
                        divx_m_f, divxfac, wdfac)::ComplexF64

    @warn "CLAR method not yet fully implemented, returning zero" maxlog=1
    return ComplexF64(0.0, 0.0)
end


"""
    calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo, bmax,
                  bmin, kin_f, s, mass, chrg, tspl, dbob_m_f, divx_m_f,
                  divxfac, wdfac, method, op_wmats; kwargs...)::ComplexF64

Calculate GAR (General Aspect Ratio) torque.
Fully general method without aspect ratio expansion.
Handles variants: FGAR (full), TGAR (trapped), PGAR (passing).
Can compute torque (*TMM), energy (*WMM), or matrix elements (*KMM/*RMM).

Ports Fortran torque.F90 GAR branch (lines 529-932).

# Steps
1. Compute bounce-averaged quantities via `compute_bounce_data()`
2. Build fbnce interpolant over λ, normalize for ODE stability
3. Integrate over pitch angle via `integrate_pitch_gar()`
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
                       bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f,
                       divx_m_f, divxfac, wdfac, method, op_wmats;
                       chi1::Float64, ro::Float64, mfac::Vector{Int}, mpert::Int,
                       ibmax::Int, theta_bmax::Float64,
                       smat=nothing, tmat=nothing, xmat=nothing,
                       ymat=nothing, zmat=nothing,
                       nlmda::Int=64, ntheta::Int=128,
                       nutype::String="harmonic", f0type::String="maxwellian",
                       nufac::Float64=1.0, ximag::Float64=0.0, qt::Bool=false,
                       energy_atol::Float64=1e-12, energy_rtol::Float64=1e-9,
                       pitch_atol::Float64=1e-12, pitch_rtol::Float64=1e-9,
                       rex_override::Union{Nothing,Float64}=nothing,
                       imx_override::Union{Nothing,Float64}=nothing)::ComplexF64

    # 1. Compute bounce-averaged quantities
    bounce = compute_bounce_data(
        psi, n, l, q, bo, bmax, bmin, ibmax, theta_bmax,
        tspl, mfac, chi1, ro, dbob_m_f, divx_m_f, divxfac, wdfac,
        mass, chrg, kin_f, s, method;
        nlmda, ntheta, smat, tmat, xmat, ymat, zmat)

    if bounce.nlmda == 0
        return ComplexF64(0.0, 0.0)
    end

    # 2. Build fbnce interpolant: [wb, wd, f₁, f₂, ...]
    # Number of flux quantities: 1 (scalar dJdJ) + optional mpert²×6 (matrix elements)
    do_matrices = !isnothing(op_wmats) && !isnothing(bounce.wmats_vs_lambda)
    if do_matrices
        nqty = 1 + mpert^2 * 6
    else
        nqty = 1
    end

    # Pack all quantities into a matrix for interpolation
    fbnce_data = zeros(Float64, bounce.nlmda, 2 + nqty)
    fbnce_data[:, 1] .= bounce.wb
    fbnce_data[:, 2] .= bounce.wd
    fbnce_data[:, 3] .= bounce.dJdJ

    if do_matrices
        for ilmda in 1:bounce.nlmda
            iqty = 4
            for k in 1:6
                for j in 1:mpert
                    for i in 1:mpert
                        fbnce_data[ilmda, iqty] = real(bounce.wmats_vs_lambda[i, j, k, ilmda])
                        iqty += 1
                    end
                end
            end
        end
    end

    # Build CubicSeriesInterpolant
    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=NaturalBC())

    # Normalize flux quantities by 1/median for ODE stability (Fortran lines 819-823)
    fbnce_norm = ones(Float64, nqty)
    for i in 1:nqty
        col = fbnce_data[:, i + 2]
        med = median(abs.(col))
        if med > 0
            fbnce_norm[i] = 1.0 / med
            # Apply normalization to the interpolant data
            fbnce_data[:, i + 2] .*= fbnce_norm[i]
        end
    end
    # Rebuild interpolant with normalized data
    fbnce = cubic_interp(bounce.lambda, Series(fbnce_data); bc=NaturalBC())

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

    # 4. Pitch angle ODE integration
    lxint = integrate_pitch_gar(
        wdian, wdiat, welec, nuk, bo / bmax, epsr, q,
        fbnce, fbnce_norm, nqty, l, n, rex, imx, psi, method;
        nutype, f0type, nufac, ximag, qt,
        energy_atol, energy_rtol, pitch_atol, pitch_rtol)

    # 5. Compute scalar torque (Fortran lines 852-854)
    # Eq. (19) of Logan et al., Phys. Plasmas 20, 122507 (2013)
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * kin_f[s] * kin_f[s+2] *
            (chi1 / twopi)  # unit conversion from ψ to ψ_n, θ_n to θ
    tpsi_val = tnorm * (lxint[1] / fbnce_norm[1])

    # 6. Kinetic matrix assembly (Fortran lines 858-923)
    if do_matrices
        op_wmats .= 0
        iqty = 2  # lxint index (1 is scalar torque)
        for k in 1:6
            for j in 1:mpert
                for i in 1:mpert
                    op_wmats[i, j, k] = complex(lxint[iqty] / fbnce_norm[iqty - 1]) *
                        tnorm * (1 / (2 * im * n))  # convert torque → energy
                    iqty += 1
                end
            end
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
    compute_kinetic_matrices_at_psi!(kwmat, ktmat, psi, n, l, zi, mi,
        wdfac, divxfac, electron, equil, intr)

Compute both kinetic energy (`kwmat`) and torque (`ktmat`) matrices at a
single flux surface in one pass. Replaces the Fortran pattern of calling
`tpsi` twice (once with "fwmm", once with "ftmm").

The bounce averaging, pitch-angle ODE, and energy-space ODE are identical
for both paths — only the final resonance operator decomposition differs
(Fortran torque.F90 lines 839-847, Julia `PitchIntegration.jl:243`):
  - wmm: keeps imaginary part (`rex=0, imx=1`) → kinetic energy
  - tmm: keeps real part (`rex=1, imx=0`) → torque

By running with `rex=1, imx=1` we get the full complex result and split:
  - `kwmat[i,j,k] = complex(0, imag(full[i,j,k]))` (energy)
  - `ktmat[i,j,k] = complex(real(full[i,j,k]), 0)` (torque)

# Arguments
- `kwmat::Array{ComplexF64,3}`: Output energy matrices (mpert×mpert×6), zeroed on entry
- `ktmat::Array{ComplexF64,3}`: Output torque matrices (mpert×mpert×6), zeroed on entry
- `psi, n, l, zi, mi, wdfac, divxfac, electron`: Same as `tpsi!`
- `equil`: PlasmaEquilibrium
- `intr::KineticForcesInternal`: Internal state with profile and geometry interpolants

Reference: [Logan et al., Phys. Plasmas 20, 122507 (2013)]
"""
function compute_kinetic_matrices_at_psi!(
    kwmat::Array{ComplexF64,3},
    ktmat::Array{ComplexF64,3},
    psi::Float64, n::Int, l::Int,
    zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
    electron::Bool, equil, intr::KineticForcesInternal)

    mpert = intr.mpert

    # Full complex matrices (rex=1, imx=1)
    full_wmats = zeros(ComplexF64, mpert, mpert, 6)
    tpsi_val = Ref{ComplexF64}(0.0 + 0.0im)

    # Call tpsi! with "fwmm" method but override rex/imx to get full complex result
    tpsi!(tpsi_val, psi, n, l, zi, mi, wdfac, divxfac,
          electron, "fwmm", equil, intr;
          op_wmats=full_wmats,
          rex_override=1.0, imx_override=1.0)

    # Split into energy (imaginary) and torque (real)
    for k in 1:6, j in 1:mpert, i in 1:mpert
        val = full_wmats[i, j, k]
        kwmat[i, j, k] = complex(0.0, imag(val))
        ktmat[i, j, k] = complex(real(val), 0.0)
    end
end


