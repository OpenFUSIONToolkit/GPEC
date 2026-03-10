"""
    tpsi(tpsi_var, psi, n, l, zi, mi, wdfac, divxfac, electron, method; 
         op_erecord=false, op_ffuns=nothing, op_orecord=false, op_wmats=nothing)

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

# Optional Arguments
- `op_wmats::Array{ComplexF64,3}`: Store DCON matrix elements

# Returns
- `ComplexF64`: Toroidal torque due to nonambipolar transport
"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int, 
              zi::Int, mi::Int, wdfac::Float64, divxfac::Float64, 
              electron::Bool, method::String; ; op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing
                )
    
    
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
    
    # Get perturbations
    dbob_m_f = Spl.spline_eval!(dbob_m, psi)
    divx_m_f = Spl.spline_eval!(divx_m, psi)

    # Poloidal functions
    mthsurf_local = mthsurf  # Get from global or parameter
    tspl = Spl.SplineType(mthsurf_local, 5)  # Allocate spline
    tspl.xs = range(0, 1, length=mthsurf_local+1) |> collect
    
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        eqfun_f, eqfun_fx, eqfun_fy = Spl.bicube_deriv1!(eqfun, psi, theta)
        rzphi_f, rzphi_fx, rzphi_fy = Spl.bicube_deriv1!(rzphi, psi, theta)
        
        tspl.fs[i+1, 1] = eqfun_f[1]            # b
        tspl.fs[i+1, 2] = eqfun_fx[1] / chi1    # db/dpsi
        tspl.fs[i+1, 3] = eqfun_fy[1]           # db/dtheta
        tspl.fs[i+1, 4] = rzphi_f[4] / chi1     # jac
        tspl.fs[i+1, 5] = rzphi_fx[4] / chi1^2  # dj/dpsi
    end
    
    Spl.spline_fit!(tspl, "periodic")

    bmax = maximum(tspl.fs[:, 1])
    ibmax = argmax(tspl.fs[:, 1])                      # 1-based
    if bmax != tspl.fs[ibmax, 1]
        error("ERROR: tpsi! - Equilibrium field maximum not consistent with index")
    end

    # "4th smallest" heuristic for initial bmin like Fortran
    bmin = minimum(tspl.fs[:, 1])
    for _ in 2:4
        mask = tspl.fs[:, 1] .> bmin
        if !any(mask)
            break
        end
        bmin = minimum(tspl.fs[mask, 1])
    end
    # Use >= like Fortran mask for ibmin
    inds = findall(tspl.fs[:, 1] .>= bmin)
    isempty(inds) && error("ERROR: tpsi! - could not find ibmin")
    ibmin = inds[argmin(tspl.fs[inds, 1])]
    if !isapprox(bmin, tspl.fs[ibmin,1]; rtol=0, atol=0)
        error("ERROR: tpsi! - Equilibrium field minimum not consistent with index")
    end

    theta_bmin = tspl.xs[ibmin]
    theta_bmax = tspl.xs[ibmax]

    # Build derivative spline dbdtspl: store dB/dθ at knots
    # Prefer analytic derivative storage if available; otherwise use tspl.fs[:,3] (= dB/dθ from eqfun) which you already filled.
    dbdtspl = Spl.SplineType(mthsurf_local, 1)
    dbdtspl.xs = copy(tspl.xs)
    dbdtspl.fs[:, 1] = tspl.fs[:, 3]  # dB/dθ from eqfun (already sampled)
    Spl.spline_fit!(dbdtspl, "periodic")

    # Find roots of dB/dθ = 0 (extrema)
    extrema = Spl.spline_roots(dbdtspl, 1)  # assume returns Vector{Float64}
    for θ in extrema
        θn = mod(θ, 1.0)
        f = Spl.spline_eval!(tspl, θn)      # returns qty vector; f[1]=B
        Bθ = f[1]
        if Bθ < bmin
            bmin = Bθ
            theta_bmin = θn
        end
        if Bθ > bmax
            bmax = Bθ
            theta_bmax = θn
        end
    end

    Spl.spline_dealloc!(dbdtspl)
    
    # Get flux function variables
    sq_s_f = Spl.spline_eval!(sq, psi)
    kin_f, kin_f1 = Spl.spline_deriv1!(kin, psi)
    geom_f = Spl.spline_eval!(geom, psi)

    q = sq_s_f[4]
    welec = kin_f[5]
    wdian = -twopi * kin_f[s+2] * kin_f1[s] / (chrg * chi1 * kin_f[s])
    wdiat = -twopi * kin_f1[s+2] / (chrg * chi1)
    wphi = welec + wdian + wdiat
    wtran = sqrt(2 * kin_f[s+2] / mass) / (q * ro)
    wgyro = chrg * bo / mass
    nuk = kin_f[s+6]
    
    rzphi_bmin = Spl.bicube_eval!(rzphi, psi, theta_bmin)
    if rzphi_bmin[1] <= 0
        println("  psi = ", psi, " -> r^2 at min(B) = ", rzphi_bmin[1])
        println("  -- theta at min(B) = ", theta_bmin)
        for i in 0:10
            θ = i / 10.0
            Bθ = Spl.spline_eval!(tspl, θ)[1]
            rz = Spl.bicube_eval!(rzphi, psi, θ)
            println("  -- theta,B(theta),r^2(theta) = ", θ, ", ", Bθ, ", ", rz[1])
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
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)
        
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

    Spl.spline_dealloc!(tspl)
    
    return nothing
end


# ============================================================================
# Method-specific torque calculation functions
# ============================================================================

"""
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)::ComplexF64

Calculate FCGL (Full Circular Gyrokinetic Landau) torque.
Implements simplified energy balance equation.
Only valid for bounce harmonic l=0.

Based on: [Logan et al., Phys. Plasmas 2013]
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)::ComplexF64
    
    # Only implemented for l=0
    if l != 0
        return ComplexF64(0.0, 0.0)
    end
    
    mthsurf_local = length(tspl.xs) - 1
    
    # Create spline for poloidal integrands
    cglspl = zeros(2, mthsurf_local + 1)
    theta_vals = range(0, 1, length=mthsurf_local + 1)
    
    # Evaluate poloidal functions and field perturbations
    for i in eachindex(theta_vals)
        theta = theta_vals[i]
        
        eqfun_f = Spl.bicube_eval!(eqfun, psi, theta)
        rzphi_f = Spl.bicube_eval!(rzphi, psi, theta)
        
        # Fourier component of field perturbations
        expm = exp(im * twopi * mfac * theta)
        dbob = sum(dbob_m_f .* expm)  # dB/B
        divx = sum(divx_m_f .* expm) * divxfac  # ∇·ξ⊥
        
        kapx = -0.5 * (dbob + divx)  # Kappa coupling
        
        # Poloidal integrands (include 0.5 factor for quadratic form)
        cglspl[1, i] = rzphi_f[4] * 0.5 * abs(divx)^2
        cglspl[2, i] = rzphi_f[4] * 0.5 * abs(divx + 3.0 * kapx)^2
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
    
    # CLAR implementation requires:
    # 1. Pitch angle grid generation (powspace_sub)
    # 2. Bounce point calculations for each lambda
    # 3. Bounce-averaged frequency calculations
    # 4. Resonance operator integration
    # 5. Lambda-dependent output
    
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
    
    # GAR implementation requires:
    # 1. Fine pitch-angle grids with appropriate concentration
    # 2. Poloidal orbit calculations (bounce points for each lambda)
    # 3. Bounce-averaged Lambda functions (frequencies, actions, curvatures)
    # 4. Energy space resonance operator evaluation
    # 5. Optional: Eigenvalue/matrix elements for stability analysis
    # 6. Different handling for Full/Trapped/Passing particle species
    
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
    
    # Simplified implementation:
    # For full version, would compute:
    # ∫ (2π ∂B/B)_m * F^1/2_m,nql * dθ weighted by Jacobian
    # where F^1/2_mnql is the special function from Park et al. 2009
    
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
    
    # Simplified implementation
    # Full version would compute magnetic action through Fourier integrals
    # dJ = ∮ (B·∇θ) dθ with perturbations included
    
    rms_dbob = sqrt(mean(abs.(dbob_m_f).^2))
    
    # Scale by kappa and include aspect-ratio dependence
    return kappa * rms_dbob^2
end