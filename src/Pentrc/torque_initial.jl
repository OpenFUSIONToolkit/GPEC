# PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE
const Splines = SplinesMod

# Provide a lightweight constructor compatible with previous CSplineType() usage
const CSplineType = SplinesMod.CubicSpline

# Module-level variables
atol_psi = 1e-3
rtol_psi = 1e-6
psi_warned = 0.0
tdebug = false
output_ascii = false
output_netcdf = true
nlmda = 128
ntheta = 128
nrecorded = 0

const nfluxfuns = 19
const nthetafuns = 12

# Record structures
mutable struct Record
    is_recorded::Bool
    psi_record::Matrix{Float64}
    ell_record::Array{Float64,3}
end

Record() = Record(false, Matrix{Float64}(undef, 0, 0), Array{Float64}(undef, 0, 0, 0))

torque_record = [[Record() for _ in 1:ngrids] for _ in 1:nmethods]

mutable struct DiagnosticRecord
    is_recorded::Bool
    psi_index::Int
    lambda_index::Int
    ell_index::Int
    ell::Vector{Int}
    psi::Vector{Float64}
    fpsi::Matrix{Float64}
    lambda::Array{Float64,3}
    fs::Array{Float64,5}
end

function DiagnosticRecord()
    DiagnosticRecord(false, 0, 0, 0, Int[], Float64[], 
                    Matrix{Float64}(undef, 0, 0),
                    Array{Float64}(undef, 0, 0, 0),
                    Array{Float64}(undef, 0, 0, 0, 0, 0))
end

orbit_record = [DiagnosticRecord() for _ in 1:nmethods]

# Global spline objects (would be defined based on your spline module)
elems = Array{ComplexF64,3}(undef, 0, 0, 0)
kelmm = [CSplineType() for _ in 1:6]  # Placeholder - define based on your cspline module
trans = CSplineType()  # Placeholder

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
- `op_erecord::Bool`: Store energy space integrands in memory
- `op_ffuns::Vector{Float64}`: Store flux function quantities
- `op_orecord::Bool`: Store poloidal orbit functions in memory
- `op_wmats::Array{ComplexF64,3}`: Store DCON matrix elements

# Returns
- `ComplexF64`: Toroidal torque due to nonambipolar transport
"""
function tpsi!(tpsi_var::Ref{ComplexF64}, psi::Float64, n::Int, l::Int, 
              zi::Int, mi::Int, wdfac::Float64, divxfac::Float64, 
              electron::Bool, method::String; 
              op_erecord::Bool=false, op_ffuns::Union{Nothing,Vector{Float64}}=nothing,
              op_orecord::Bool=false, op_wmats::Union{Nothing,Array{ComplexF64,3}}=nothing)
    
    erecord = op_erecord
    orecord = op_orecord
    
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
    dbob_m_f = cspline_eval_external(dbob_m, psi)
    divx_m_f = cspline_eval_external(divx_m, psi)
    
    # Poloidal functions
    mthsurf_local = mthsurf  # Get from global or parameter
    tspl = SplineType(mthsurf_local, 5)  # Allocate spline
    tspl.xs = range(0, 1, length=mthsurf_local+1) |> collect
    
    for i in 0:mthsurf_local
        theta = i / mthsurf_local
        eqfun_f = bicube_eval_external(eqfun, psi, theta, 1)
        rzphi_f = bicube_eval_external(rzphi, psi, theta, 1)
        
        tspl.fs[i+1, 1] = eqfun_f[1]            # b
        tspl.fs[i+1, 2] = eqfun_f[2] / chi1     # db/dpsi
        tspl.fs[i+1, 3] = eqfun_f[3]            # db/dtheta
        tspl.fs[i+1, 4] = rzphi_f[4] / chi1     # jac
        tspl.fs[i+1, 5] = rzphi_f[5] / chi1^2   # dj/dpsi
    end
    
    spline_fit!(tspl, "periodic")
    
    # Find bmin and bmax
    bmax = maximum(tspl.fs[:, 1])
    bmin = minimum(tspl.fs[:, 1])
    ibmax = argmax(tspl.fs[:, 1]) - 1
    
    # Refine bmin - find 4th smallest
    for i in 2:4
        bmin = minimum(tspl.fs[tspl.fs[:, 1] .> bmin, 1])
    end
    ibmin = findfirst(tspl.fs[:, 1] .== bmin) - 1
    
    # Get flux function variables
    sq_s_f = spline_eval_external(sq, psi)
    kin_f, kin_f1 = spline_eval_external(kin, psi, deriv=1)
    geom_f = spline_eval_external(geom, psi)
    
    q = sq_s_f[4]
    welec = kin_f[5]
    wdian = -twopi * kin_f[s+2] * kin_f1[s] / (chrg * chi1 * kin_f[s])
    wdiat = -twopi * kin_f1[s+2] / (chrg * chi1)
    wphi = welec + wdian + wdiat
    wtran = sqrt(2 * kin_f[s+2] / mass) / (q * ro)
    wgyro = chrg * bo / mass
    nuk = kin_f[s+6]
    
    epsr = geom_f[2] / geom_f[3]
    wbhat = (π / 4) * sqrt(epsr / 2) * wtran
    wdhat = q^3 * wtran^2 / (4 * epsr * wgyro) * wdfac
    nueff = kin_f[s+6] / (2 * epsr)
    
    if tdebug
        @printf("   eq values = %.1e %.1e %.1e %.1e %.1e %.1e %.1f %d\n",
               wdian, wdiat, welec, wdhat, wbhat, nueff, q, 0)
    end
    
    # Store flux functions if requested
    if op_ffuns !== nothing
        op_ffuns[:] = [epsr, kin_f[1:8]..., q, sq_s_f[2],
                      wdian, wdiat, wtran, wgyro, wbhat, wdhat, 0.0, 0.0]
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
                                    dbob_m_f, erecord)
        
    elseif method == "clar"
        tpsi_var[] = calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                    nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                    tspl, dbob_m_f, erecord)
        
    elseif method in ["fgar", "tgar", "pgar", "fwmm", "twmm", "pwmm",
                      "ftmm", "ttmm", "ptmm", "fkmm", "tkmm", "pkmm",
                      "frmm", "trmm", "prmm"]
        tpsi_var[] = calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec,
                                   nuk, bo, bmax, bmin, kin_f, s, mass, chrg,
                                   tspl, dbob_m_f, divx_m_f, divxfac, wdfac,
                                   method, erecord, orecord, op_wmats)
    else
        error("ERROR: torque - unknown method")
    end
    
    if tdebug
        println("torque - end function, psi = ", psi)
    end
    
    spline_dealloc!(tspl)
    
    return nothing
end

"""
    calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)

Calculate torque using the Chew-Goldberger-Low (CGL) limit.
"""
function calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)
    if l != 0
        return ComplexF64(0, 0)
    end
    
    cglspl = SplineType(mthsurf, 2)
    
    for i in 0:mthsurf
        theta = i / mthsurf
        eqfun_f = bicube_eval_external(eqfun, psi, theta, 0)
        rzphi_f = bicube_eval_external(rzphi, psi, theta, 0)
        
        expm = exp.(im * twopi * mfac * theta)
        jbb = rzphi_f[4] * eqfun_f[1]^2
        dbob = sum(dbob_m_f .* expm)
        divx = sum(divx_m_f .* expm)
        kapx = -0.5 * (dbob + divx)
        
        cglspl.xs[i+1] = theta
        cglspl.fs[i+1, 1] = rzphi_f[4] * 0.5 * abs(divx)^2
        cglspl.fs[i+1, 2] = rzphi_f[4] * 0.5 * abs(divx + 3.0 * kapx)^2
    end
    
    spline_fit!(cglspl, "periodic")
    spline_int!(cglspl)
    
    tpsi_out = 2.0 * n * im * kin_f[s] * kin_f[s+2] *
               (0.5 * (5.0/3.0) * cglspl.fsi[end, 1] +
                0.5 * (1.0/3.0) * cglspl.fsi[end, 2])
    
    spline_dealloc!(cglspl)
    
    return tpsi_out
end

"""
    calculate_rlar(...)

Calculate torque using Reduced Large Aspect Ratio approximation.
"""
function calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, wdhat, wbhat,
                       nueff, sq_s_f, kin_f, s, dbob_m_f, erecord)
    lnq = Float64(l)
    sigma = 0
    
    if tdebug
        println("  rlar integrations. modes = ", n, " ", l)
    end
    
    # Energy space integrations
    lmdamax = min(1.0 / (1 - epsr), bo / bmin)
    xint = xintgrl_lsode(wdian, wdiat, welec, wdhat, wbhat, nueff, l, lnq, n,
                        psi, lmdamax, "rlar"; op_record=erecord)
    
    # Kappa integration
    if tdebug
        println("  <|dB/B|> = ", sum(abs.(dbob_m_f)) / length(dbob_m_f))
    end
    kappaint = kappaintgrl(n, l, q, mfac, dbob_m_f, fnml)
    
    # dT/dpsi
    tpsi_out = sq_s_f[3] * kappaint * 0.5 * (-xint) *
               sqrt(epsr / (2 * π^3)) * n^2 * kin_f[s] * kin_f[s+2]
    
    if tdebug
        println("  ->  xint ", xint, ", kint ", kappaint, ", tpsi ", tpsi_out)
    end
    
    return tpsi_out
end

"""
    calculate_clar(...)

Calculate torque using Circular Large Aspect Ratio approximation.
"""
function calculate_clar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                       bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f, erecord)
    # This is a simplified version - full implementation would follow Fortran logic
    # Set up Lambda functions and bounce integration
    
    lmdatpb = bo / bmax
    lmdamin = max(1.0 / (1 + epsr), bo / bmax)
    lmdamax = min(1.0 / (1 - epsr), bo / bmin)
    
    # Create pitch angle space
    ldl = powspace(lmdamin, lmdamax, 1, nlmda, "both")
    
    # Energy integration (simplified)
    rex = 1.0
    imx = 1.0
    
    # This would call lambdaintgrl_lsode with proper bounce-averaged functions
    # Placeholder for now
    lxint = ComplexF64(0, 0)
    
    # Calculate torque
    tpsi_out = (-2 * n^2 / sqrt(π)) * (ro / bo) * kin_f[s] * kin_f[s+2] *
               lxint * (chi1 / twopi)
    
    return tpsi_out
end

"""
    calculate_gar(...)

Calculate torque using General Aspect Ratio method.
This is the most complex calculation with full bounce averaging.
"""
function calculate_gar(psi, n, l, q, epsr, wdian, wdiat, welec, nuk, bo,
                      bmax, bmin, kin_f, s, mass, chrg, tspl, dbob_m_f,
                      divx_m_f, divxfac, wdfac, method, erecord, orecord, op_wmats)
    
    if tdebug
        println("  General aspect ratio calculation for method: ", method)
    end
    
    # Set up splines for bounce integration
    vspl = spline_alloc(tspl.mx, 1)  # vparallel(theta) -> roots are bounce pts
    bspl = spline_alloc(ntheta-1, 2)  # omegab,d bounce integration
    bjspl = cspline_alloc(ntheta-1, 1)  # action bounce integration
    vspl.xs = copy(tspl.xs)
    linspace_sub!(0.0, 1.0, ntheta, bspl.xs)
    bjspl.xs = copy(bspl.xs)
    turns = spline_alloc(nlmda-1, 6)  # (theta,r,z) of lower and upper turns
    
    # Handle optional wmats calculation
    has_wmats = op_wmats !== nothing
    if has_wmats
        fbnce = cspline_alloc(nlmda-1, 3 + mpert^2 * 6)
        bwspl = [cspline_alloc(ntheta-1, mpert) for _ in 1:2]
        for i in 1:2
            bwspl[i].xs = copy(bjspl.xs)
        end
        
        # Build equilibrium geometric matrices
        smat = reshape(cspline_eval_external(smats, psi), (mpert, mpert))
        tmat = reshape(cspline_eval_external(tmats, psi), (mpert, mpert))
        xmat = reshape(cspline_eval_external(xmats, psi), (mpert, mpert))
        ymat = reshape(cspline_eval_external(ymats, psi), (mpert, mpert))
        zmat = reshape(cspline_eval_external(zmats, psi), (mpert, mpert))
    else
        fbnce = cspline_alloc(nlmda-1, 3)
    end
    
    fbnce.title = fill("elemij", nlmda)
    fbnce.title[1:4] = ["Lambda", "omegab", "omegaD", "wbdJdJ"]
    
    # Set up lambda space based on method type
    lmdamin = 0.0
    lmdatpb = bo / bmax
    ibmax_val = findfirst(==(maximum(tspl.fs[:, 1])), tspl.fs[:, 1]) - 1
    lmdatpe = min(bo / tspl.fs[ibmax_val+2, 1], bo / tspl.fs[ibmax_val, 1])
    lmdamax = bo / bmin
    
    if method[1] == 't'  # trapped
        ldl_inc = powspace_sub(lmdatpb, lmdamax, 1, 2+nlmda, "both")
        ldl = ldl_inc[:, 2:nlmda+1]  # exclude boundaries
    elseif method[1] == 'p'  # passing
        ldl_inc = powspace_sub(lmdamin, lmdatpb, 1, 2+nlmda, "both")
        ldl = ldl_inc[:, 2:nlmda+1]
    else  # full
        if lmdatpb == lmdamax
            @warn "bmax = bmin @ psi $psi"
        end
        ldl_p = powspace_sub(lmdamin, lmdatpb, 2, 2+div(nlmda,2), "upper")
        ldl_t = powspace_sub(lmdatpb, lmdamax, 2, 2+nlmda-div(nlmda,2), "lower")
        ldl = hcat(ldl_p[:, 2:div(nlmda,2)+1], ldl_t[:, 2:nlmda-div(nlmda,2)+1])
    end
    
    if tdebug
        println(" Lambda space ", ldl[1, 1], " ", ldl[1, nlmda], 
                ", t/p boundary = ", lmdatpb)
    end
    
    # Optional orbit recording setup
    ilambda_out = Int[]
    if orecord
        ilambda_out = [round(Int, ilmda * (nlmda-1) / (nlambda_out-1)) + 1 
                      for ilmda in 0:nlambda_out-1]
    end
    
    # Loop over pitch angles
    for ilmda in 1:nlmda
        lmda = ldl[1, ilmda]
        
        # Determine if trapped or passing
        sigma = lmda > (bo / bmax) ? 0 : 1  # 0=trapped, 1=passing
        lnq = l + sigma * n * q
        
        # Determine bounce points
        vspl.fs[:, 1] = 1.0 .- (lmda / bo) .* tspl.fs[:, 1]
        spline_fit!(vspl, "extrap")
        
        if sigma == 0  # trapped particle
            nbpts, bpts = spline_roots(vspl, 1)
            
            if nbpts < 1
                @warn "Found passing particle in bounce particle integrals at psi=$psi, lambda=$lmda"
                t1 = tspl.xs[ibmax_val+1]
                t2 = t1 + 1
            elseif nbpts < 2
                t1 = bpts[1]
                t2 = bpts[1] + 1.0
            else
                # Find deepest potential well
                vpar = 0.0
                t1 = 0.0
                t2 = 1.0
                for i in 1:nbpts
                    j = i < nbpts ? i + 1 : 1
                    if bpts[i] > bpts[j]
                        # Wrapping case
                        theta_mid = mod((bpts[i] + bpts[j] + 1.0) / 2, 1.0)
                    else
                        theta_mid = (bpts[i] + bpts[j]) / 2
                    end
                    vpar_test = spline_eval(vspl, theta_mid, 0)[1]
                    if vpar_test > vpar
                        t1 = bpts[i]
                        t2 = bpts[j]
                        if t2 < t1
                            t2 += 1.0
                        end
                        vpar = vpar_test
                    end
                end
                if vpar == 0
                    error("Could not find potential well with positive vpar")
                end
            end
            tdt = powspace_sub(t1, t2, 4, ntheta, "both")
        else  # passing particle
            t1 = tspl.xs[ibmax_val+1]
            t2 = t1 + 1
            tdt = powspace_sub(t1, t2, 2, ntheta, "both")
        end
        
        if tdebug && mod(ilmda, div(ntheta, 10)) == 0
            println("(t1,t2) = ", t1, " ", t2)
            println("tdt rng = ", tdt[1, 1], " ", tdt[1, ntheta])
        end
        
        # Record bounce locations
        turns.fs[ilmda, 1] = t1
        rzphi_f = bicube_eval_external(rzphi, psi, t1, 0)
        turns.fs[ilmda, 2] = ro + sqrt(rzphi_f[1]) * cos(twopi * (t1 + rzphi_f[2]))
        turns.fs[ilmda, 3] = zo + sqrt(rzphi_f[1]) * sin(twopi * (t1 + rzphi_f[2]))
        turns.fs[ilmda, 4] = t2
        rzphi_f = bicube_eval_external(rzphi, psi, t2, 0)
        turns.fs[ilmda, 5] = ro + sqrt(rzphi_f[1]) * cos(twopi * (t2 + rzphi_f[2]))
        turns.fs[ilmda, 6] = zo + sqrt(rzphi_f[1]) * sin(twopi * (t2 + rzphi_f[2]))
        turns.xs[ilmda] = lmda
        
        # Bounce averages
        wmu_mt = zeros(ComplexF64, mpert, ntheta)
        wen_mt = zeros(ComplexF64, mpert, ntheta)
        jvtheta = zeros(ComplexF64, ntheta)
        bspl.fs .= 0.0
        
        for i in 2:ntheta-1  # edge dts are 0
            tspl_vals = spline_eval(tspl, mod(tdt[1, i], 1.0), 0)
            vspl_vals = spline_eval(vspl, mod(tdt[1, i], 1.0), 0)
            vpar = vspl_vals[1]
            
            if vpar <= 0  # local zero between nodes
                if lmda > lmdatpe || lmda < (2*lmdatpb - lmdatpe)
                    if abs(psi_warned - psi) > 0.1
                        @warn @sprintf("vpar zero crossing at psi=%.3e, t=%.3e<=%.3e<=%.3e",
                                      psi, t1, tdt[1, i], t2)
                        global psi_warned = psi
                    end
                end
                if i < div(ntheta, 2)
                    bspl.fs[1:i-1, :] .= 0.0
                    jvtheta[1:i] .= 0
                    continue
                else
                    bspl.fs[i-1:end, 1] .= bspl.fs[i-2, 1]
                    bspl.fs[i-1:end, 2] .= bspl.fs[i-2, 2]
                    jvtheta[i:end] .= jvtheta[i-1]
                    break
                end
            end
            
            bspl.fs[i-1, 1] = tdt[2, i] * tspl_vals[4] * tspl_vals[1] / sqrt(vpar)
            bspl.fs[i-1, 2] = tdt[2, i] * tspl_vals[4] * tspl_vals[2] *
                             (1 - 1.5 * lmda * tspl_vals[1] / bo) / sqrt(vpar) +
                             tdt[2, i] * tspl_vals[5] * tspl_vals[1] * sqrt(vpar)
            
            expm = exp.(im * twopi * mfac * tdt[1, i])
            jbb = chi1 * tspl_vals[4] * tspl_vals[1]^2
            dbob = sum(dbob_m_f .* expm)
            divx = sum(divx_m_f .* expm) * divxfac
            
            jvtheta[i] = tdt[2, i] * tspl_vals[4] * tspl_vals[1] *
                        (divx * sqrt(vpar) + dbob * (1 - 1.5 * lmda * tspl_vals[1] / bo) / sqrt(vpar)) *
                        exp(-twopi * im * n * q * (tdt[1, i] - tdt[1, 1]))
            
            # Euler-Lagrange matrix vectors
            if has_wmats
                wmu_mt[:, i] = tdt[2, i] * (lmda / bo) .* expm ./ sqrt(vpar) .*
                              exp(-twopi * im * n * q * (tdt[1, i] - tdt[1, 1])) ./
                              (2 * chi1)
                wen_mt[:, i] = tdt[2, i] .* expm ./ (tspl_vals[1] * sqrt(vpar)) .*
                              exp(-twopi * im * n * q * (tdt[1, i] - tdt[1, 1])) ./
                              (2 * chi1)
            end
            
            if i > 2 && bspl.fs[i-3, 1] == 0
                bspl.fs[2:i-2, 1] .= bspl.fs[i-1, 1]
                bspl.fs[2:i-2, 2] .= bspl.fs[i-1, 2]
                jvtheta[2:i-1] .= jvtheta[i]
            end
        end
        
        spline_fit!(bspl, "extrap")
        spline_int!(bspl)
        
        # Bounce averaged Lambda functions
        fbnce.xs[ilmda] = lmda
        wbbar = ro * twopi / ((2 - sigma) * bspl.fsi[end, 1])
        wdbar = ro^2 * bo * wdfac * wbbar * 2 * (2 - sigma) * bspl.fsi[end, 2]
        bhat = sqrt(2 * kin_f[s+2] / mass) / ro
        dhat = (kin_f[s+2] / chrg) / (bo * ro^2)
        
        fbnce.fs[ilmda, 1] = wbbar * bhat
        fbnce.fs[ilmda, 2] = wdbar * dhat
        
        # Phase factor and action
        pl = exp.(-twopi * im * lnq .* bspl.fsi[:, 1] ./ ((2 - sigma) * bspl.fsi[end, 1]))
        bjspl.fs[:, 1] = conj.(jvtheta) .* (pl .+ (1 - sigma) ./ pl)
        
        cspline_fit!(bjspl, "extrap")
        cspline_int!(bjspl)
        
        # Division by 2 corrects quadratic use
        fbnce.fs[ilmda, 3] = wbbar * abs(bjspl.fsi[end, 1])^2 / 2 / ro^2
        
        # Matrix elements if requested
        if has_wmats
            # Bounce integrate vectors
            for i in 1:mpert
                bwspl[1].fs[:, i] = conj.(wmu_mt[i, :]) .* (pl .+ (1 - sigma) ./ pl)
                bwspl[2].fs[:, i] = conj.(wen_mt[i, :]) .* (pl .+ (1 - sigma) ./ pl)
            end
            for i in 1:2
                cspline_fit!(bwspl[i], "extrap")
                cspline_int!(bwspl[i])
            end
            
            # Build complete action matrices
            wmmt = reshape(bwspl[1].fsi[end, :], (1, mpert))
            wemt = reshape(bwspl[2].fsi[end, :], (1, mpert))
            wxmt = wmmt * xmat
            wymt = wmmt * (3 * smat + ymat) - 2.0 * wemt * smat
            wzmt = wmmt * (3 * tmat + zmat) - 2.0 * wemt * tmat
            wxmc = conj(transpose(wxmt))
            wymc = conj(transpose(wymt))
            wzmc = conj(transpose(wzmt))
            
            op_wmats_local = zeros(ComplexF64, mpert, mpert, 6)
            op_wmats_local[:, :, 1] = wzmc * wzmt  # A
            op_wmats_local[:, :, 2] = wzmc * wxmt  # B
            op_wmats_local[:, :, 3] = wzmc * wymt  # C
            op_wmats_local[:, :, 4] = wxmc * wxmt  # D
            op_wmats_local[:, :, 5] = wxmc * wymt  # E
            op_wmats_local[:, :, 6] = wymc * wymt  # H
            
            for k in 1:6
                for j in 1:mpert
                    for i in 1:mpert
                        iqty = ((k-1) * mpert + j - 1) * mpert + i + 3
                        fbnce.fs[ilmda, iqty] = wbbar * op_wmats_local[i, j, k] / ro^2
                    end
                end
            end
        end
        
        # Optional orbit recording
        if orecord && ilmda in ilambda_out
            if tdebug
                println("  recording bounce functions")
            end
            orbitfs = zeros(nthetafuns, ntheta)
            for i in 1:ntheta
                expm = exp.(im * twopi * mfac * tdt[1, i])
                dbob = sum(dbob_m_f .* expm)
                divx = sum(divx_m_f .* expm) * divxfac
                he_t = bspl.fsi[i, 1] / ((2 - sigma) * bspl.fsi[end, 1])
                hd_t = bspl.fsi[i, 2] / ((2 - sigma) * bspl.fsi[end, 2])
                wb_t = bspl.fs[i, 1]
                wd_t = bspl.fs[i, 2]
                
                orbitfs[:, i] = [tdt[1, i], tdt[2, i],
                                real(bjspl.fs[i, 1]), imag(bjspl.fs[i, 1]),
                                real(dbob), imag(dbob), real(divx), imag(divx),
                                wb_t, wd_t, he_t, hd_t]
            end
            record_orbit(method, psi, l, lmda, [q, lmdatpb], orbitfs)
        end
    end  # lambda loop
    
    # Normalize lambda functions for lsode
    fbnce_norm = zeros(fbnce.nqty - 2)
    for i in 1:fbnce.nqty-2
        fbnce_norm[i] = 1.0 / median(abs.(fbnce.fs[:, i+2]))
        fbnce.fs[:, i+2] .*= fbnce_norm[i]
    end
    cspline_fit!(fbnce, "extrap")
    spline_fit!(turns, "extrap")
    
    if tdebug
        println("lambda ~ ", ldl[1, 1:10:end], " ", ldl[1, nlmda])
        println("wb(lmda) ~ ", fbnce.fs[1:10:end, 1], " ", fbnce.fs[nlmda, 1])
        println("wd(lmda) ~ ", fbnce.fs[1:10:end, 2], " ", fbnce.fs[nlmda, 2])
    end
    
    # Energy space integrations
    lxint = zeros(ComplexF64, fbnce.nqty - 2)
    wtwnorm = 1.0
    rex = 1.0
    imx = 1.0
    
    if method[2:4] == "wmm" || method[2:4] == "kmm"
        rex = 0.0
    elseif method[2:4] == "tmm" || method[2:4] == "rmm"
        imx = 0.0
        wtwnorm = -1.0
    end
    
    lxint = lambdaintgrl_lsode(wdian, wdiat, welec, nuk, bo/bmax,
                               epsr, q, fbnce, l, n, rex, imx, psi, turns,
                               method; op_record=erecord)
    
    # dT/dpsi
    tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * kin_f[s] * kin_f[s+2] *
            (chi1 / twopi)
    tpsi_out = tnorm * (lxint[1] / fbnce_norm[1])
    
    if tdebug
        println("  ->  lxint ", lxint[1], ", tpsi ", tpsi_out)
    end
    
    # Euler-Lagrange matrices if requested
    if has_wmats
        op_wmats .= 0
        for k in 1:6
            for j in 1:mpert
                for i in 1:mpert
                    iqty = ((k-1) * mpert + j - 1) * mpert + i + 1
                    op_wmats[i, j, k] = lxint[iqty] / fbnce_norm[iqty] *
                                       tnorm * (1 / (2 * im * n))
                end
            end
        end
        
        for i in 1:2
            cspline_dealloc!(bwspl[i])
        end
        
        # DCON norms
        op_wmats .*= 2 * mu0
        op_wmats[:, :, 1:3] ./= chi1
        op_wmats[:, :, 1] ./= chi1
        
        if method[2:4] == "kmm" || method[2:4] == "rmm"
            # Euler-Lagrange Matrix norms
            xix = ones(ComplexF64, mpert, 1) / sqrt(Float64(mpert))
            tpsi_temp = 0.0
            for i in 1:6
                # Compute spectral norm (largest singular value)
                a = transpose(op_wmats[:, :, i]) * op_wmats[:, :, i]
                svals = svdvals(a)
                if tdebug
                    println(i, " ", maximum(svals))
                end
                tpsi_temp += maximum(svals)
            end
            tpsi_out = (rex + imx * im) * sqrt(tpsi_temp)
        elseif occursin("mm", method)
            # Mode-coupled dW of T_phi
            xs_m1_f = cspline_eval_external(xs_m[1], psi)
            xs_m2_f = cspline_eval_external(xs_m[2], psi)
            xs_m3_f = cspline_eval_external(xs_m[3], psi)
            xix = reshape(xs_m1_f, (mpert, 1))
            xiy = reshape(xs_m2_f, (mpert, 1))
            xiz = reshape(xs_m3_f, (mpert, 1))
            
            # Division by 2 corrects quadratic use
            t_zz = conj(transpose(xiz)) * op_wmats[:, :, 1] * xiz / 2 * chi1^2
            t_zx = conj(transpose(xiz)) * op_wmats[:, :, 2] * xix / 2 * chi1
            t_zy = conj(transpose(xiz)) * op_wmats[:, :, 3] * xiy / 2 * chi1
            t_xx = conj(transpose(xix)) * op_wmats[:, :, 4] * xix / 2
            t_xy = conj(transpose(xix)) * op_wmats[:, :, 5] * xiy / 2
            t_yy = conj(transpose(xiy)) * op_wmats[:, :, 6] * xiy / 2
            
            tpsi_out = (2 * n * im / (2 * mu0)) *
                      (t_zz[1,1] + t_xx[1,1] + t_yy[1,1] +
                       t_zx[1,1] + t_zy[1,1] + t_xy[1,1] +
                       wtwnorm * conj(t_zx[1,1] + t_zy[1,1] + t_xy[1,1]))
            
            if tdebug
                println(" -> WxWx ~ ", op_wmats[20:25, 20, 1])
                println("expected to be all real (wmm) or all imag (tmm):")
                println("  xx = ", t_xx)
                println("  yy = ", t_yy)
                println("  zz = ", t_zz)
            end
        end
    end
    
    # Clean up
    spline_dealloc!(vspl)
    spline_dealloc!(bspl)
    spline_dealloc!(turns)
    cspline_dealloc!(bjspl)
    cspline_dealloc!(fbnce)
    
    return tpsi_out
end

"""
    tintgrl_lsode(psilim, n, nl, zi, mi, wdfac, divxfac, electron, method)

Torque integral over psi using LSODE-style ODE integration.

# Arguments
- `psilim::Vector{Float64}`: Integration bounds [min, max]
- `n::Int`: Mode number
- `nl::Int`: Number of bounce harmonics to include (-nl to nl)
- `zi::Int`: Ion charge
- `mi::Int`: Ion mass
- `wdfac::Float64`: Drift factor
- `divxfac::Float64`: Divergence factor
- `electron::Bool`: Electron flag
- `method::String`: Integration method

# Returns
- `ComplexF64`: Integrated torque
"""
function tintgrl_lsode(psilim::Vector{Float64}, n::Int, nl::Int,
                      zi::Int, mi::Int, wdfac::Float64, divxfac::Float64,
                      electron::Bool, method::String)
    
    # Reset module variables
    global psi_warned = 0.0
    
    # Set common variables (use closure or global)
    wdcom = wdfac
    dxcom = divxfac
    methcom = method
    fcom = zeros(nfluxfuns)
    
    if tdebug
        println("torque - lsode wrapper method ", method, " -> ", methcom)
    end
    
    # Set up ODE parameters
    neq = 2 * (1 + 2 * nl)
    neqarray = [neq, n, nl, zi, mi, btoi(electron)]
    
    # Integration bounds
    x_start = max(psilim[1], sq.xs[1], xs_m[1].xs[1])
    x_end = min(psilim[2], sq.xs[end], xs_m[1].xs[end])
    
    if tdebug
        println("psilim = ", psilim)
        println("sq lim = ", sq.xs[1], " ", sq.xs[end])
        println("xs lim = ", xs_m[1].xs[1], " ", xs_m[1].xs[end])
        println("x, x_end = ", x_start, " ", x_end)
    end
    
    # Species parameters
    if electron
        chrg = -1 * e
        s = 2
    else
        chrg = zi * e
        s = 1
    end
    
    # Storage for profiles
    profiles = Matrix{Float64}(undef, 0, 0)
    ellprofiles = Matrix{Float64}(undef, 0, 0)
    gam = zeros(Float64, neq)
    chi = zeros(Float64, neq)
    
    # Storage for Euler-Lagrange matrices if needed
    # `const` is not allowed for local variables inside a function in Julia,
    # so use a regular local binding here.
    maxsteps = 10000
    kel_flat_mats = zeros(ComplexF64, maxsteps, mpert^2, 6)
    
    # Define ODE function
    function tintgrnd_ode!(du, u, p, x)
        # Unpack parameters
        neq, n, nl, zi, mi, ee = neqarray
        electron_flag = itob(ee)
        
        if tdebug && x < 1e-2
            println("torque - lsode subroutine wdfac ", wdcom)
            println("torque - lsode subroutine method ", methcom)
        end
        
        psi = x
        du .= 0.0
        
        # Allocate/reset elems if first call
        if !isdefined(Main, :elems_allocated) || !Main.elems_allocated
            global elems = zeros(ComplexF64, mpert, mpert, 6)
            Main.elems_allocated = true
        else
            global elems .= 0
        end
        
        # Parallel computation over bounce harmonics
        Threads.@threads for l in -nl:nl
            wtw_l = zeros(ComplexF64, mpert, mpert, 6)
            trq = Ref(ComplexF64(0, 0))
            
            if l == 0
                if occursin("mm", methcom)
                    tpsi!(trq, psi, n, l, zi, mi, wdcom, dxcom, 
                         electron_flag, methcom; op_ffuns=fcom, op_wmats=wtw_l)
                    elems .+= wtw_l
                else
                    tpsi!(trq, psi, n, l, zi, mi, wdcom, dxcom,
                         electron_flag, methcom; op_ffuns=fcom)
                end
            else
                if occursin("mm", methcom)
                    tpsi!(trq, psi, n, l, zi, mi, wdcom, dxcom,
                         electron_flag, methcom; op_wmats=wtw_l)
                    elems .+= wtw_l
                else
                    tpsi!(trq, psi, n, l, zi, mi, wdcom, dxcom,
                         electron_flag, methcom)
                end
            end
            
            # Decouple real and imaginary parts
            du[2*(l+nl)+1] = real(trq[])
            du[2*(l+nl)+2] = imag(trq[])
        end
        
        if tdebug
            println("torque - intgrnd complete at psi ", x)
            println("n,nl,zi,mi = ", n, " ", nl, " ", zi, " ", mi)
            println("x,psi,du = ", x, " ", psi, " ", du)
        end
        
        return nothing
    end
    
    # Set up ODE problem
    u0 = zeros(neq)
    tspan = (x_start, x_end)
    prob = ODEProblem(tintgrnd_ode!, u0, tspan)
    
    # Solve with callback to save profiles at each step
    saved_x = Float64[]
    saved_u = Vector{Float64}[]
    saved_du = Vector{Float64}[]
    step_count = 0
    
    function save_callback(u, t, integrator)
        step_count += 1
        
        # Compute derivative at this point
        du = similar(u)
        tintgrnd_ode!(du, u, nothing, t)
        
        # Compute flux/diffusivity profiles
        sq_vals = spline_eval(sq, t, 0)
        kin_vals, kin_vals1 = spline_eval(kin, t, 1)
        geom_vals = spline_eval(geom, t, 1)
        
        gam_vals = du .* twopi ./ (chrg * chi1 * geom_vals[1])
        
        if qt
            drive = (kin_vals[s] / kin_vals[s+2]) * 
                   (kin_vals1[s+2] / geom_vals1[2])
        else
            drive = kin_vals1[s] / geom_vals1[2] +
                   (chrg * kin_vals[s] / kin_vals[s+2]) *
                   ((kin_vals[5] / twopi) / geom_vals1[2])
        end
        
        chi_vals = -gam_vals ./ drive
        
        # Save tables for output
        push!(saved_x, t)
        push!(saved_u, copy(u))
        push!(saved_du, copy(du))
        
        # Build profile row
        profile_row = [t, sq_vals[3],
                      sum(gam_vals[1:2:neq]), sum(gam_vals[2:2:neq]),
                      sum(chi_vals[1:2:neq]), sum(chi_vals[2:2:neq]),
                      sum(du[1:2:neq]), sum(du[2:2:neq]),
                      sum(u[1:2:neq]), sum(u[2:2:neq]),
                      fcom...]
        
        if isempty(profiles)
            profiles = reshape(profile_row, (:, 1))
        else
            profiles = hcat(profiles, profile_row)
        end
        
        # Ell profiles
        for l in -nl:nl
            ell_row = [t, Float64(l),
                      gam_vals[2*(nl+l)+1], gam_vals[2*(nl+l)+2],
                      chi_vals[2*(nl+l)+1], chi_vals[2*(nl+l)+2],
                      du[2*(nl+l)+1], du[2*(nl+l)+2],
                      u[2*(nl+l)+1], u[2*(nl+l)+2]]
            
            if isempty(ellprofiles)
                ellprofiles = reshape(ell_row, (:, 1))
            else
                ellprofiles = hcat(ellprofiles, ell_row)
            end
        end
        
        # Save matrix coefficients if needed
        if occursin("mm", method) && step_count <= maxsteps
            for j in 1:6
                kel_flat_mats[step_count, :, j] = reshape(elems[:, :, j], (mpert^2,))
            end
        end
        
        # Print progress
        if (mod(t, 0.1) < mod(last_x, 0.1) || t == x_start) && verbose
            @printf(" psi = %.1f -> T_phi = %.2e %.2ej\n",
                   t, sum(u[1:2:neq]), sum(u[2:2:neq]))
        end
        last_x = t
        
        return false  # Don't terminate
    end
    
    last_x = x_start
    cb = FunctionCallingCallback(save_callback; func_everystep=true)
    
    # Solve ODE
    sol = solve(prob, Tsit5();
               abstol=atol_psi, reltol=rtol_psi,
               callback=cb, maxiters=maxsteps,
               save_everystep=false)
    
    if sol.retcode != :Success
        @warn "ODE integration may have issues: $(sol.retcode)"
    end
    
    # Set global transport and torque spline
    if trans.nqty != 0
        cspline_dealloc!(trans)
    end
    
    trans = cspline_alloc(step_count-1, 2)
    trans.xs = saved_x[1:step_count]
    trans.fs[:, 1] = profiles[3, 1:step_count] .+ 
                     im .* profiles[4, 1:step_count]  # particle/heat flux
    trans.fs[:, 2] = profiles[7, 1:step_count] .+ 
                     im .* profiles[8, 1:step_count]  # torque & energy
    cspline_fit!(trans, "extrap")
    
    # Optionally set global Euler-Lagrange coefficient splines
    if occursin("mm", method)
        for j in 1:6
            if kelmm[j].nqty != 0
                cspline_dealloc!(kelmm[j])
            end
            kelmm[j] = cspline_alloc(step_count-1, mpert^2)
            kelmm[j].xs = saved_x[1:step_count]
            kelmm[j].fs = kel_flat_mats[1:step_count, :, j]
            cspline_fit!(kelmm[j], "extrap")
        end
    end
    
    # Record results
    record_method(method, "lsode", profiles, ellprofiles)
    
    if output_ascii
        output_torque_ascii(n, zi, mi, electron, method, profiles, ellprofiles)
    end
    
    # Return complex integral
    final_u = sol.u[end]
    tintgrl_out = sum(final_u[1:2:neq]) + im * sum(final_u[2:2:neq])
    
    return tintgrl_out
end

# Additional helper functions would be defined here...

"""
    output_torque_ascii(n, zi, mi, electron, method, prof, profl)

Write ASCII torque profile files.
"""
function output_torque_ascii(n::Int, zi::Int, mi::Int, electron::Bool,
                            method::String, prof::Matrix{Float64},
                            profl::Matrix{Float64})
    
    # Set species
    if electron
        chrg = -1 * e
        mass = me
        s = 2
    else
        chrg = zi * e
        mass = mi * mp
        s = 2
    end
    
    # Create filenames
    nstring = @sprintf("%d", n)
    file1 = "pentrc_$(method)_n$(nstring).out"
    file2 = "pentrc_$(method)_ell_n$(nstring).out"
    
    if qt
        file1 = file1[1:7] * "heat_" * file1[8:end]
        file2 = file2[1:7] * "heat_" * file2[8:end]
    end
    
    # Write files
    open(file1, "w") do f
        println(f, "PERTURBED EQUILIBRIUM NONAMBIPOLAR TRANSPORT CODE:")
        println(f)
        if qt
            println(f, " Gamma = Q/T = Heat flux in 1/(sm^2)")
            println(f, " chi = Heat diffusivity in m^2/s")
        else
            println(f, " Gamma = Particle flux in 1/(sm^2)")
            println(f, " chi = Particle diffusivity in m^2/s")
        end
        @printf(f, "\nn = %4d\n", n)
        @printf(f, "charge = %17.8e  mass = %17.8e\n", chrg, mass)
        @printf(f, "R0 = %17.8e  B0 = %17.8e  chi1 = %17.8e\n\n", ro, bo, chi1)
        
        # Write data
        for i in 1:size(prof, 2)
            for j in 1:size(prof, 1)
                @printf(f, "%17.8e", prof[j, i])
            end
            println(f)
        end
    end
    
    # Similar for file2...
    
    if verbose
        println("Wrote ASCII output to $file1 and $file2")
    end
end

"""
    output_torque_netcdf(n, nl, zi, mi, electron, wdfac)

Write NetCDF torque output files.
"""
function output_torque_netcdf(n::Int, nl::Int, zi::Int, mi::Int,
                             electron::Bool, wdfac::Float64)
    
    if verbose
        println("Writing output to netcdf")
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
    
    # Create NetCDF file
    ncfile = "pentrc_output_n$(n).nc"
    
    NCDataset(ncfile, "c") do ds
        # Define global attributes
        ds.attrib["title"] = "PENTRC fundamental outputs"
        ds.attrib["version"] = version
        ds.attrib["shot"] = Int(shotnum)
        ds.attrib["time"] = Int(shottime)
        ds.attrib["machine"] = machine
        ds.attrib["n"] = n
        ds.attrib["charge"] = chrg
        ds.attrib["mass"] = mass
        ds.attrib["R0"] = ro
        ds.attrib["B0"] = bo
        ds.attrib["chi1"] = chi1
        
        # Define dimensions
        defDim(ds, "i", 2)
        defDim(ds, "ell", 2*nl+1)
        defDim(ds, "psi_n", sq.mx+1)
        
        # Define and write variables
        defVar(ds, "i", Int, ("i",))
        ds["i"][:] = [0, 1]
        
        defVar(ds, "ell", Int, ("ell",))
        ds["ell"][:] = collect(-nl:nl)
        
        defVar(ds, "psi_n", Float64, ("psi_n",))
        ds["psi_n"][:] = sq.xs
        
        # Additional variables would be defined here...
    end
    
    if verbose
        println("Finished NetCDF output: $ncfile")
    end
end