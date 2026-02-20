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
    
    # Get perturbations (using external evaluation)
    dbob_m_f = Spl.cspline_eval_external(dbob_m, psi)
    divx_m_f = Spl.cspline_eval_external(divx_m, psi)

    # Poloidal functions
    mthsurf_local = mthsurf  # Get from global or parameter
    tspl = Spl.SplineType(mthsurf_local, 5)  # Allocate spline
    tspl.xs = range(0, 1, length=mthsurf_local+1) |> collect
    
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        eqfun_f = Spl.bicube_eval_external(eqfun, psi, theta, 1)
        rzphi_f = Spl.bicube_eval_external(rzphi, psi, theta, 1)
        
        tspl.fs[i+1, 1] = eqfun_f[1]            # b
        tspl.fs[i+1, 2] = eqfun_f[2] / chi1     # db/dpsi
        tspl.fs[i+1, 3] = eqfun_f[3]            # db/dtheta
        tspl.fs[i+1, 4] = rzphi_f[4] / chi1     # jac
        tspl.fs[i+1, 5] = rzphi_f[5] / chi1^2   # dj/dpsi
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
        f = Spl.spline_eval(tspl, θn)       # returns qty vector; f[1]=B
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
    sq_s_f = Spl.spline_eval_external(sq, psi)
    kin_f, kin_f1 = Spl.spline_eval_external(kin, psi, deriv=1)
    geom_f = Spl.spline_eval_external(geom, psi)

    q = sq_s_f[4]
    welec = kin_f[5]
    wdian = -twopi * kin_f[s+2] * kin_f1[s] / (chrg * chi1 * kin_f[s])
    wdiat = -twopi * kin_f1[s+2] / (chrg * chi1)
    wphi = welec + wdian + wdiat
    wtran = sqrt(2 * kin_f[s+2] / mass) / (q * ro)
    wgyro = chrg * bo / mass
    nuk = kin_f[s+6]
    
    rzphi_bmin = Spl.bicube_eval_external(rzphi, psi, theta_bmin, 0)
    if rzphi_bmin[1] <= 0
        println("  psi = ", psi, " -> r^2 at min(B) = ", rzphi_bmin[1])
        println("  -- theta at min(B) = ", theta_bmin)
        for i in 0:10
            θ = i / 10.0
            Bθ = Spl.spline_eval(tspl, θ)[1]
            rz = Spl.bicube_eval_external(rzphi, psi, θ, 0)
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
                                    dbob_m_f)
        
    elseif method == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                    tspl, dbob_m_f)
        
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