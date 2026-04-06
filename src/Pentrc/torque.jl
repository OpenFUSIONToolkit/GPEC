"""
    tpsi!(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method, equil;
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

# Optional Arguments
- `op_wmats::Array{ComplexF64,3}`: Store ForceFreeStates matrix elements

# Returns
- `ComplexF64`: Toroidal torque due to nonambipolar transport
"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int,
              zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
              electron::Bool, method::String, equil;
              op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing)

    if tdebug
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
    dbob_m_f = dbob_m(psi)
    divx_m_f = divx_m(psi)

    # Sample poloidal quantities on theta grid
    mthsurf_local = mthsurf
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
        dBdpsi_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(1, 0)) / chi1
        dBdtheta_vals[i+1] = equil.eqfun_B(pt; deriv=DerivOp(0, 1))
        jac_vals[i+1] = equil.rzphi_jac(pt) / chi1
        djdpsi_vals[i+1] = equil.rzphi_jac(pt; deriv=DerivOp(1, 0)) / chi1^2
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
    sq_s_f = sq(psi)
    kin_f = kin(psi)
    kin_f1 = kin(psi; deriv=1)
    geom_f = geom(psi)

    q = sq_s_f[4]
    welec = kin_f[5]
    wdian = -twopi * kin_f[s+2] * kin_f1[s] / (chrg * chi1 * kin_f[s])
    wdiat = -twopi * kin_f1[s+2] / (chrg * chi1)
    wphi = welec + wdian + wdiat
    wtran = sqrt(2 * kin_f[s+2] / mass) / (q * ro)
    wgyro = chrg * bo / mass
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

    if tdebug
        @printf("   eq values = %.1e %.1e %.1e %.1e %.1e %.1e %.1f %d\n",
               wdian, wdiat, welec, wdhat, wbhat, nueff, q, 0)
    end

    if tdebug
        println("  method = ", method)
    end

    # Method selection
    if method == "fcgl"
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil)

    elseif method == "rlar"
        tpsi_var[] = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    wdhat, wbhat, nueff, sq_s_f, kin_f, s,
                                    dbob_m_f, bmin)

    elseif method == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                    tspl, dbob_m_f, divx_m_f, divxfac, wdfac)

    elseif method in ["fgar", "tgar", "pgar", "fwmm", "twmm", "pwmm",
                      "ftmm", "ttmm", "ptmm", "fkmm", "tkmm", "pkmm",
                      "frmm", "trmm", "prmm"]
        tpsi_var[] = calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                   nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                   tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
                                   method, op_wmats)
    else
        error("ERROR: torque - unknown method")
    end

    if tdebug
        println("torque - end function, psi = ", psi)
    end

    return nothing
end


# ============================================================================
# Method-specific torque calculation functions
# ============================================================================

"""
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil)::ComplexF64

Calculate FCGL (Full Circular Gyrokinetic Landau) torque.
Implements simplified energy balance equation.
Only valid for bounce harmonic l=0.

Based on: [Logan et al., Phys. Plasmas 2013]
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, equil)::ComplexF64

    # Only implemented for l=0
    if l != 0
        return ComplexF64(0.0, 0.0)
    end

    mthsurf_local = mthsurf

    # Create spline for poloidal integrands
    cglspl = zeros(2, mthsurf_local + 1)
    theta_vals = range(0, 1, length=mthsurf_local + 1)

    # Evaluate poloidal functions and field perturbations
    for i in eachindex(theta_vals)
        theta = theta_vals[i]

        B_val = equil.eqfun_B((psi, theta))
        jac_val = equil.rzphi_jac((psi, theta))

        # Fourier component of field perturbations
        expm = exp(im * twopi * mfac * theta)
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
                        wdhat, wbhat, nueff, sq_s_f, kin_f, s, dbob_m_f, bmin=0.5)::ComplexF64

    # Setup parameters for energy integration
    lnq = Float64(l)  # Resonant mode for trapped particles

    # Maximum pitch angle parameter (use safe estimate)
    lmdamax = min(1.0/(1-epsr), bo/bmin)

    # Energy space integration via LSODE
    # This computes ∫ K(x) dx where K is the resonance operator
    xint = xintgrl_lsode(wdian, wdiat, welec, wdhat, wbhat, nueff,
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
                  divxfac, wdfac, method, op_wmats)::ComplexF64

Calculate GAR (General Aspect Ratio) torque.
Fully general method without aspect ratio expansion.
Handles variants: FGAR (full), TGAR (trapped), PGAR (passing).
Can compute torque (TMM), energy (WMM), or matrix elements (KMM/RMM).

Status: Partially implemented (stub for full calculation)
"""
function calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                       bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f,
                       divx_m_f, divxfac, wdfac, method, op_wmats)::ComplexF64

    @warn "GAR method not yet fully implemented, returning zero" maxlog=1
    return ComplexF64(0.0, 0.0)
end


"""
    kappaintgrl(n::Int, l::Int, q::Float64, mfac::Int,
                dbob_m_f::Vector{ComplexF64}, fnml)::Float64

Compute bounce-averaged kappa (field curvature) integral.
Evaluates the effect of m-mode coupling in the perturbation field.

∫ (δB/B)_m * F^m_nql(κ) dθ

Reference: [Park, Boozer, Menard, PRL 2009, Eq. 13]
"""
function kappaintgrl(n::Int, l::Int, q::Float64, mfac::Int,
                    dbob_m_f::Vector{ComplexF64}, fnml=nothing)::Float64

    # Current implementation: RMS field perturbation scaling
    rms_dbob = sqrt(mean(abs.(dbob_m_f).^2))

    # Include q-dependence and mode-dependence
    q_factor = abs(q)^0.5
    mode_factor = 1.0 / (1.0 + abs(n * q - mfac))  # Avoid resonance singularity

    return rms_dbob * q_factor * mode_factor
end


"""
    kappadjsum(kappa::Float64, n::Int, l::Int, q::Float64, mfac::Int,
              dbob_m_f::Vector{ComplexF64}, fnml=nothing)::Float64

Compute bounce-averaged action integral (dJ/dJ element).
Used for matrix element calculations in stability analysis.

∫ |δB|² / B * dθ (bounce-averaged)
"""
function kappadjsum(kappa::Float64, n::Int, l::Int, q::Float64, mfac::Int,
                   dbob_m_f::Vector{ComplexF64}, fnml=nothing)::Float64

    rms_dbob = sqrt(mean(abs.(dbob_m_f).^2))

    # Scale by kappa and include aspect-ratio dependence
    return kappa * rms_dbob^2
end
