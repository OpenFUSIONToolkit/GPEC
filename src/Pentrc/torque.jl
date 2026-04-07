using Printf
using Statistics

global psi_warned = 0.0

function _trace_active(method_key::AbstractString, psi::Real, ell::Integer)
    trace_enabled || return false
    trace_method == "none" || trace_method == lowercase(String(method_key)) || return false
    trace_psi < 0 || abs(Float64(psi) - trace_psi) <= trace_tol || return false
    trace_ell == typemin(Int) || trace_ell == ell || return false
    return true
end

function _trace_complex_vector(label::AbstractString, vals)
    parts = String[]
    for (i, val) in enumerate(vals)
        push!(parts, @sprintf("[%d]=%.12e %+.12ei", i, real(val), imag(val)))
    end
    println("TRACE ", label, " ", join(parts, " "))
    return nothing
end

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
                            electron::Bool, method::String; op_erecord::Bool=false,
                            op_ffuns=nothing, op_orecord::Bool=false,
                            op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing)
    
    
    method_key = lowercase(strip(method))
    trace_this = _trace_active(method_key, psi, l)

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

    if trace_this
        @printf("TRACE tpsi start method=%s psi=%.15e n=%d l=%d electron=%s\n",
                method_key, psi, n, l, electron ? "T" : "F")
        @printf("TRACE eq q=%.12e epsr=%s bmin=%.12e bmax=%.12e\n",
                q, "NA", bmin, bmax)
        @printf("TRACE eq theta_bmin=%.12e theta_bmax=%.12e\n", theta_bmin, theta_bmax)
        @printf("TRACE eq wdian=%.12e wdiat=%.12e welec=%.12e wtran=%.12e wgyro=%.12e nuk=%.12e\n",
                wdian, wdiat, welec, wtran, wgyro, nuk)
        _trace_complex_vector("dbob_m_f=", dbob_m_f)
        _trace_complex_vector("divx_m_f=", divx_m_f)
    end

    if method_key == "fcgl"
        tpsi_var[] = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s, divxfac;
                                    trace_this=trace_this)
        if trace_this
            @printf("TRACE tpsi end method=%s psi=%.15e tpsi=%.12e %+.12ei\n",
                    method_key, psi, real(tpsi_var[]), imag(tpsi_var[]))
        end
        return nothing
    end
    
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

    if trace_this
        @printf("TRACE eq q=%.12e epsr=%.12e bmin=%.12e bmax=%.12e\n",
                q, epsr, bmin, bmax)
    end
    
    if tdebug
        @printf("   eq values = %.1e %.1e %.1e %.1e %.1e %.1e %.1f %d\n",
               wdian, wdiat, welec, wdhat, wbhat, nueff, q, 0)
    end
    
    
    if tdebug
        println("  method = ", method)
    end
    
    # Method selection
    if method_key == "rlar"
        tpsi_var[] = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, 
                                    wdhat, wbhat, nueff, sq_s_f, kin_f, s,
                                    dbob_m_f, bmin)
        
    elseif method_key == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                    tspl, dbob_m_f, divx_m_f, divxfac, wdfac)
        
    elseif method_key in ["fgar", "tgar", "pgar", "fwmm", "twmm", "pwmm",
                          "ftmm", "ttmm", "ptmm", "fkmm", "tkmm", "pkmm",
                          "frmm", "trmm", "prmm"]
        tpsi_var[] = calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                   nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                   tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
                                   method_key, op_wmats)
    else
        error("ERROR: torque - unknown method: $(method)")
    end
    
    if tdebug
        println("torque - end function, psi = ", psi)
    end

    if trace_this
        @printf("TRACE tpsi end method=%s psi=%.15e tpsi=%.12e %+.12ei\n",
                method_key, psi, real(tpsi_var[]), imag(tpsi_var[]))
    end

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
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s,
                        divxfac::Float64=1.0; trace_this::Bool=false)::ComplexF64
    
    # Only implemented for l=0
    if l != 0
        return ComplexF64(0.0, 0.0)
    end
    
    mthsurf_local = length(tspl.xs) - 1
    theta_vals = collect(range(0.0, 1.0; length=mthsurf_local + 1))
    cglspl = zeros(Float64, length(theta_vals), 2)

    for i in eachindex(theta_vals)
        theta = theta_vals[i]
        rzphi_f = Spl.bicube_eval!(rzphi, psi, theta)
        expm = exp.(im * twopi * mfac * theta)
        dbob = sum(dbob_m_f .* expm)
        divx = sum(divx_m_f .* expm) * divxfac
        kapx = -0.5 * (dbob + divx)

        cglspl[i, 1] = rzphi_f[4] * 0.5 * abs(divx)^2
        cglspl[i, 2] = rzphi_f[4] * 0.5 * abs(divx + 3.0 * kapx)^2

        if trace_this
            @printf("TRACE fcgl theta=%.12e dbob=%.12e %+.12ei divx=%.12e %+.12ei kapx=%.12e %+.12ei jac=%.12e\n",
                    theta, real(dbob), imag(dbob), real(divx), imag(divx), real(kapx), imag(kapx), rzphi_f[4])
            @printf("TRACE fcgl integrands=%.12e %.12e\n", cglspl[i, 1], cglspl[i, 2])
        end
    end

    cgl_spline = Spl.CubicSpline(theta_vals, cglspl; bctype="periodic")
    Spl.spline_integrate!(cgl_spline)
    integral_1 = cgl_spline.fsi[end, 1]
    integral_2 = cgl_spline.fsi[end, 2]

    result = 2.0 * n * im * kin_f[s] * kin_f[s+2] * 
        (0.5 * (5.0/3.0) * integral_1 + 
         0.5 * (1.0/3.0) * integral_2)

    if trace_this
        @printf("TRACE fcgl integral_1=%.12e integral_2=%.12e\n", integral_1, integral_2)
        @printf("TRACE fcgl result=%.12e %+.12ei\n", real(result), imag(result))
    end
    
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
                        wdhat, wbhat, nueff, sq_s_f, kin_f, s,
                        dbob_m_f, bmin=0.5)::ComplexF64
    
    # Setup parameters for energy integration
    lnq = Float64(l)  # Resonant mode for trapped particles
    
    # Maximum pitch angle parameter (use safe estimate)
    lmdamax = min(1.0/(1-epsr), bo/bmin)
    
    # Energy space integration via LSODE
    # This computes ∫ K(x) dx where K is the resonance operator
    xint = xintgrl_lsode(wdian, wdiat, welec, wdhat, wbhat, nueff, 
                         l, lnq, n, psi, lmdamax, "rlar")
    
    # Kappa/bounce averaging (effect of field perturbations)
    kappaint_val = kappaintgrl(n, l, q, mfac, dbob_m_f)
    
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
    lmdatpb = bo / bmax
    lmdamin = max(1.0 / (1.0 + epsr), bo / bmax)
    lmdamax = min(1.0 / (1.0 - epsr), bo / bmin)
    ldl = powspace_sub(lmdamin, lmdamax, 1, nlmda, "both")
    flambda_fs = zeros(ComplexF64, size(ldl, 2), 3)
    turns_fs = zeros(Float64, size(ldl, 2), 6)
    bhat = sqrt(2 * kin_f[s + 2] / mass)
    dhat = kin_f[s + 2] / chrg

    for ilmda in 1:size(ldl, 2)
        lmda = ldl[1, ilmda]
        kk = (1 - lmda * (1 - epsr)) / (2 * epsr * lmda)
        κ = kk <= 0 ? 0.0 : kk >= 1 ? 1 - 1e-6 : sqrt(kk)

        wbbar = π * sqrt(2 * epsr * lmda * bo) / (4 * q * ro * ellipk(κ^2))
        wdbar = (2 * q * lmda * (ellipe(κ^2) / ellipk(κ^2) - 0.5) / (ro^2 * epsr)) * wdfac
        djdj = kappadjsum(κ, n, l, q, mfac, dbob_m_f, fnml) * (0.5 * π / epsr) * (mass * bhat * q * ro)^2 / (2 * π)
        djdj /= (mass * bhat * ro)^2

        flambda_fs[ilmda, 1] = wbbar * bhat
        flambda_fs[ilmda, 2] = wdbar * dhat
        flambda_fs[ilmda, 3] = wbbar * djdj
        turns_fs[ilmda, :] .= _turn_points_from_lambda(tspl, psi, lmda, bo, true)
    end

    fbnce_norm = _median_inverse_scale(real.(flambda_fs[:, 3]))
    flambda_fs[:, 3] .*= fbnce_norm

    flambda = Spl.CubicSpline(vec(ldl[1, :]), flambda_fs; bctype="extrap")
    turns = Spl.CubicSpline(vec(ldl[1, :]), turns_fs; bctype="extrap")
    lxint = lambdaintgrl_lsode(wdian, wdiat, welec, nuk, bo / bmax,
                               epsr, q, flambda, l, n, 1.0, 1.0, psi,
                               turns, "clar"; op_record=false)[1]

    tnorm = (-2.0 * n^2 / sqrt(pi)) * (ro / bo) * kin_f[s] * kin_f[s + 2] * (chi1 / twopi)
    return tnorm * lxint / fbnce_norm
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
    lmdatpb = bo / bmax
    lmdamin = 0.0
    lmdamax = bo / bmin
    ibmax = argmax(tspl.fs[:, 1])
    ibprev, ibnext = _periodic_neighbors(ibmax, size(tspl.fs, 1))
    lmdatpe = min(bo / tspl.fs[ibnext, 1], bo / tspl.fs[ibprev, 1])

    method_key = lowercase(strip(method))
    family = first(method_key)
    trace_this = _trace_active(method_key, psi, l)

    if family == 't'
        ldl = powspace_sub(lmdatpb, lmdamax, 1, 2 + nlmda, "both")[:, 2:nlmda+1]
    elseif family == 'p'
        ldl = powspace_sub(lmdamin, lmdatpb, 1, 2 + nlmda, "both")[:, 2:nlmda+1]
    else
        ldl_p = powspace_sub(lmdamin, lmdatpb, 2, 2 + div(nlmda, 2), "upper")
        ldl_t = powspace_sub(lmdatpb, lmdamax, 2, 2 + nlmda - div(nlmda, 2), "lower")
        ldl = zeros(Float64, 2, nlmda)
        ldl[1, :] = vcat(ldl_p[1, 2:1 + div(nlmda, 2)], ldl_t[1, 2:1 + nlmda - div(nlmda, 2)])
        ldl[2, :] = vcat(ldl_p[2, 2:1 + div(nlmda, 2)], ldl_t[2, 2:1 + nlmda - div(nlmda, 2)])
    end

    if op_wmats !== nothing
        op_wmats .= 0
    end
    flambda_fs = zeros(ComplexF64, size(ldl, 2), 3)
    turns_fs = zeros(Float64, size(ldl, 2), 6)

    for ilmda in 1:size(ldl, 2)
        lmda = ldl[1, ilmda]
        sigma = lmda > lmdatpb ? 0 : 1
        lnq = l + sigma * n * q
        vspl = _gar_vspl(tspl, lmda, bo)
        t1, t2 = _gar_turn_interval(tspl, vspl, psi, lmda, bo, sigma, lmdamin, lmdatpb, lmdamax)
        turns_fs[ilmda, :] .= _turn_points(psi, t1, t2)

        tdt = sigma == 0 ? powspace_sub(t1, t2, 4, ntheta, "both") : powspace_sub(t1, t2, 2, ntheta, "both")
        npts = size(tdt, 2)
        bspl_fs = zeros(Float64, npts, 2)
        jvtheta = zeros(ComplexF64, npts)

        for i in 2:(npts - 1)
            θraw = tdt[1, i]
            θ = mod(θraw, 1.0)
            ts = Spl.spline_eval!(tspl, θ)
            b_val = ts[1]
            jac = ts[4]
            dbdpsi = ts[2]
            djdpsi = ts[5]
            vpar = Spl.spline_eval!(vspl, θ)[1]
            if trace_this
                @printf("TRACE gar theta i=%12d theta=% .16e dt=% .16e\n", i, θraw, tdt[2, i])
                @printf("TRACE gar theta b=% .16e jac=% .16e dbdpsi=% .16e djdpsi=% .16e\n",
                        b_val, jac, dbdpsi, djdpsi)
            end
            if vpar <= 0
                if lmda > lmdatpe || lmda < (2 * lmdatpb - lmdatpe)
                    if abs(psi_warned - psi) > 0.1
                        @printf(" !! WARNING: vpar zero crossing at psi=%10.3e, t=%10.3e<=%10.3e<=%10.3e\n",
                                psi, t1, tdt[1, i], t2)
                        global psi_warned = psi
                    end
                end
                if i < div(npts, 2)
                    bspl_fs[1:(i - 1), :] .= 0.0
                    jvtheta[1:i] .= 0.0
                    continue
                else
                    prev = max(i - 2, 1)
                    bspl_fs[(i - 1):end, 1] .= bspl_fs[prev, 1]
                    bspl_fs[(i - 1):end, 2] .= bspl_fs[prev, 1]
                    jvtheta[i:end] .= jvtheta[max(i - 1, 1)]
                    break
                end
            end

            sqrtv = sqrt(vpar)
            bspl_fs[i - 1, 1] = tdt[2, i] * jac * b_val / sqrtv
            bspl_fs[i - 1, 2] = tdt[2, i] * jac * dbdpsi * (1 - 1.5 * lmda * b_val / bo) / sqrtv +
                                tdt[2, i] * djdpsi * b_val * sqrtv

            expm = exp.(im * twopi * mfac * θraw)
            dbob = sum(dbob_m_f .* expm)
            divx = sum(divx_m_f .* expm) * divxfac
            jvtheta[i] = tdt[2, i] * jac * b_val *
                         (divx * sqrtv + dbob * (1 - 1.5 * lmda * b_val / bo) / sqrtv) *
                         exp(-twopi * im * n * q * (θraw - tdt[1, 1]))
            if trace_this
                @printf("TRACE gar theta vpar=% .16e dbob=%.16e %+.16ei divx=%.16e %+.16ei jvtheta=%.16e %+.16ei\n",
                        vpar, real(dbob), imag(dbob), real(divx), imag(divx), real(jvtheta[i]), imag(jvtheta[i]))
            end

            if i >= 3 && bspl_fs[i - 2, 1] == 0.0
                bspl_fs[2:(i - 2), 1] .= bspl_fs[i - 1, 1]
                bspl_fs[2:(i - 2), 2] .= bspl_fs[i - 1, 2]
                jvtheta[2:(i - 1)] .= jvtheta[i]
            end
            if trace_this
                @printf("TRACE gar theta bspl=% .16e % .16e\n", bspl_fs[i - 1, 1], bspl_fs[i - 1, 2])
            end
        end

        sgrid = collect(range(0.0, 1.0, length=npts))
        bspl = Spl.CubicSpline(sgrid, bspl_fs; bctype="extrap")
        Spl.spline_integrate!(bspl)
        int1 = bspl.fsi[:, 1]
        int2 = bspl.fsi[:, 2]
        denom = max((2 - sigma) * int1[end], eps())
        wbbar = ro * twopi / denom
        wdbar = ro^2 * bo * wdfac * wbbar * 2 * (2 - sigma) * int2[end]
        bhat = sqrt(2 * kin_f[s + 2] / mass) / ro
        dhat = (kin_f[s + 2] / chrg) / (bo * ro^2)
        flambda_fs[ilmda, 1] = wbbar * bhat
        flambda_fs[ilmda, 2] = wdbar * dhat

        pl = exp.(-twopi * im * lnq .* int1 ./ denom)
        bjspl_fs = conj.(jvtheta) .* (pl .+ (1 - sigma) ./ pl)
        bjspl_pair = hcat(real.(bjspl_fs), imag.(bjspl_fs))
        bjspl = Spl.CubicSpline(sgrid, bjspl_pair; bctype="extrap")
        Spl.spline_integrate!(bjspl)
        jint = ComplexF64(bjspl.fsi[end, 1], bjspl.fsi[end, 2])
        flambda_fs[ilmda, 3] = wbbar * abs2(jint) / (2 * ro^2)
    end

    fbnce_norm = _median_inverse_scale(real.(flambda_fs[:, 3]))
    flambda_fs[:, 3] .*= fbnce_norm

    flambda = Spl.CubicSpline(vec(ldl[1, :]), flambda_fs; bctype="extrap")
    turns = Spl.CubicSpline(vec(ldl[1, :]), turns_fs; bctype="extrap")

    rex = 1.0
    imx = 1.0
    if endswith(method_key, "wmm") || endswith(method_key, "kmm")
        rex = 0.0
    elseif endswith(method_key, "tmm") || endswith(method_key, "rmm")
        imx = 0.0
    end

    lxint = lambdaintgrl_lsode(wdian, wdiat, welec, nuk, bo / bmax,
                               epsr, q, flambda, l, n, rex, imx, psi,
                               turns, method_key; op_record=false)[1]

    tnorm = (-2.0 * n^2 / sqrt(pi)) * (ro / bo) * kin_f[s] * kin_f[s + 2] * (chi1 / twopi)
    result = tnorm * (lxint / fbnce_norm)

    if trace_this
        @printf("TRACE gar final lxint=%.12e %+.12ei fbnce_norm=%.12e tnorm=%.12e\n",
                real(lxint), imag(lxint), fbnce_norm, tnorm)
        @printf("TRACE gar final tpsi=%.12e %+.12ei\n", real(result), imag(result))
    end

    if endswith(method_key, "kmm") || endswith(method_key, "rmm")
        return complex(abs(result), 0.0)
    end

    return result
end

function _turn_points(psi::Float64, t1::Float64, t2::Float64)
    out = zeros(Float64, 6)
    out[1] = t1
    rz1 = Spl.bicube_eval!(rzphi, psi, mod(t1, 1.0))
    out[2] = ro + sqrt(max(rz1[1], 0.0)) * cos(twopi * (mod(t1, 1.0) + rz1[2]))
    out[3] = zo + sqrt(max(rz1[1], 0.0)) * sin(twopi * (mod(t1, 1.0) + rz1[2]))
    out[4] = t2
    rz2 = Spl.bicube_eval!(rzphi, psi, mod(t2, 1.0))
    out[5] = ro + sqrt(max(rz2[1], 0.0)) * cos(twopi * (mod(t2, 1.0) + rz2[2]))
    out[6] = zo + sqrt(max(rz2[1], 0.0)) * sin(twopi * (mod(t2, 1.0) + rz2[2]))
    return out
end

function _turn_points_from_lambda(tspl, psi::Float64, lmda::Float64, bo::Float64, trapped_only::Bool)
    vspl = _gar_vspl(tspl, lmda, bo)
    roots = Spl.spline_roots(vspl, 1)

    if isempty(roots)
        t1 = trapped_only ? 0.0 : tspl.xs[argmax(tspl.fs[:, 1])]
        t2 = t1 + 1.0
    elseif length(roots) == 1
        t1 = roots[1]
        t2 = roots[1] + 1.0
    else
        t1, t2 = _deepest_well(vspl, roots)
    end
    return _turn_points(psi, t1, t2)
end

function _deepest_well(vspl, roots::AbstractVector{<:Real})
    best_mid = -Inf
    best_t1 = 0.0
    best_t2 = 1.0
    for idx in eachindex(roots)
        left = Float64(roots[idx])
        jdx = idx == lastindex(roots) ? firstindex(roots) : idx + 1
        right = Float64(roots[jdx])
        midpoint =
            if left > right
                mod(0.5 * (left + right + 1.0), 1.0)
            else
                0.5 * (left + right)
            end
        val = Spl.spline_eval!(vspl, midpoint)[1]
        if val > best_mid
            best_t1 = left
            best_t2 = right
            if best_t2 < best_t1
                best_t2 += 1.0
            end
            best_mid = val
        end
    end
    return best_t1, best_t2
end

function _gar_vspl(tspl, lmda::Float64, bo::Float64)
    vspl = Spl.SplineType(tspl.mx, 1)
    vspl.xs = copy(tspl.xs)
    vspl.fs[:, 1] .= 1.0 .- (lmda / bo) .* tspl.fs[:, 1]
    Spl.spline_fit!(vspl, "extrap")
    return vspl
end

function _gar_turn_interval(tspl, vspl, psi::Float64, lmda::Float64, bo::Float64, sigma::Int,
                            lmdamin::Float64, lmdatpb::Float64, lmdamax::Float64)
    ibmax = argmax(tspl.fs[:, 1])
    if sigma == 0
        roots = Spl.spline_roots(vspl, 1)
        if isempty(roots)
            println("!!")
            println("!! WARNING: Found passing particle in bounce particle integrals")
            println("  > psi = ", psi, ", lambda = ", lmda)
            println("  > lambda min, boundary, max = ", lmdamin, ", ", lmdatpb, ", ", lmdamax)
            println("  > Consider refining equilibrium or changing equil spline sizes")
            println("!!")
            t1 = tspl.xs[ibmax]
            t2 = t1 + 1.0
        elseif length(roots) == 1
            t1 = roots[1]
            t2 = roots[1] + 1.0
        else
            t1, t2 = _deepest_well(vspl, roots)
        end
    else
        t1 = tspl.xs[ibmax]
        t2 = t1 + 1.0
    end
    return t1, t2
end

function _periodic_neighbors(i::Int, n::Int)
    prev = i == 1 ? n - 1 : i - 1
    next = i == n ? 2 : i + 1
    return prev, next
end

function _median_inverse_scale(vals::AbstractVector{<:Real})
    nz = sort(abs.(collect(vals)))
    nz = filter(!iszero, nz)
    isempty(nz) && return 1.0
    return 1.0 / median(nz)
end

function _cumtrapz(xs::AbstractVector{<:Real}, ys::AbstractVector{T}) where {T<:Number}
    out = zeros(T, length(xs))
    for i in 2:length(xs)
        dx = Float64(xs[i] - xs[i - 1])
        out[i] = out[i - 1] + 0.5 * dx * (ys[i - 1] + ys[i])
    end
    return out
end


"""
    kappaintgrl(n::Int, l::Int, q::Float64, mfac::Int, 
                dbob_m_f::Vector{ComplexF64}, fnml)::Float64

Compute bounce-averaged kappa (field curvature) integral.
Evaluates the effect of m-mode coupling in the perturbation field.

∫ (δB/B)_m * F^m_nql(κ) dθ

Reference: [Park, Boozer, Menard, PRL 2009, Eq. 13]
"""
function kappaintgrl(n::Int, l::Int, q::Float64, marray::AbstractVector{<:Integer},
                    dbob_m_f::AbstractVector{ComplexF64}, fnml_in=nothing)::Float64
    fnml_spl = isnothing(fnml_in) ? fnml : fnml_in
    isnothing(fnml_spl) && error("RLAR requires `load_fnml!` before calling `kappaintgrl`.")

    nk = 50
    κgrid = [(k + 1.0) / (nk + 2.0) for k in 0:nk]
    integrand = zeros(Float64, nk + 1)
    mvals = Int.(marray)
    mpert_local = length(mvals)

    for (ik, κ) in enumerate(κgrid)
        fm = zeros(Float64, mpert_local, 2)
        for i in eachindex(mvals)
            shift_minus = mvals[i] - n * q - l
            shift_plus = mvals[i] - n * q + l
            abs(shift_minus) <= fnml_spl.ys[end] || error("pitch - RLAR approximation m-nq-l out of Fkmnql range")
            abs(shift_plus) <= fnml_spl.ys[end] || error("pitch - RLAR approximation m-nq+l out of Fkmnql range")
            fm[i, 1] = Spl.bicube_eval!(fnml_spl, κ, shift_minus)[1]
            fm[i, 2] = Spl.bicube_eval!(fnml_spl, κ, shift_plus)[1]
        end

        acc = 0.0
        for i in eachindex(mvals), j in eachindex(mvals)
            coupling = 0.25 * (fm[i, 1] * fm[j, 1] + fm[i, 2] * fm[j, 1] + fm[i, 1] * fm[j, 2] + fm[i, 2] * fm[j, 2])
            acc += 2 * κ * coupling * real(dbob_m_f[i] * conj(dbob_m_f[j])) / (4 * ellipk(κ^2))
        end
        integrand[ik] = acc
    end

    kspl = Spl.CubicSpline(collect(κgrid), reshape(integrand, :, 1); bctype="extrap")
    Spl.spline_integrate!(kspl)
    return kspl.fsi[end, 1]
end


"""
    kappadjsum(kappa::Float64, n::Int, l::Int, q::Float64, mfac::Int,
              dbob_m_f::Vector{ComplexF64}, fnml=nothing)::Float64

Compute bounce-averaged action integral (dJ/dJ element).
Used for matrix element calculations in stability analysis.

∫ |δB|² / B * dθ (bounce-averaged)
"""
function kappadjsum(kappa::Float64, n::Int, l::Int, q::Float64, marray::AbstractVector{<:Integer},
                   dbob_m_f::AbstractVector{ComplexF64}, fnml_in=nothing)::Float64
    fnml_spl = isnothing(fnml_in) ? fnml : fnml_in
    isnothing(fnml_spl) && error("Kappa action sum requires `load_fnml!` before calling `kappadjsum`.")

    mvals = Int.(marray)
    mpert_local = length(mvals)
    fm = zeros(Float64, mpert_local, 2)

    for i in eachindex(mvals)
        shift_minus = mvals[i] - n * q - l
        shift_plus = mvals[i] - n * q + l
        abs(shift_minus) <= fnml_spl.ys[end] || error("pitch - RLAR approximation m-nq-l out of Fkmnql range")
        abs(shift_plus) <= fnml_spl.ys[end] || error("pitch - RLAR approximation m-nq+l out of Fkmnql range")
        fm[i, 1] = Spl.bicube_eval!(fnml_spl, kappa, shift_minus)[1]
        fm[i, 2] = Spl.bicube_eval!(fnml_spl, kappa, shift_plus)[1]
    end

    acc = 0.0
    for i in eachindex(mvals), j in eachindex(mvals)
        coupling = 0.25 * (fm[i, 1] * fm[j, 1] + fm[i, 2] * fm[j, 1] + fm[i, 1] * fm[j, 2] + fm[i, 2] * fm[j, 2])
        acc += coupling * real(dbob_m_f[i] * conj(dbob_m_f[j]))
    end
    return acc
end
