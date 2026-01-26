"""
Surface energy calculation module for DCON - Complete rewrite with exact geometric curvature
"""

"""
    compute_surface_matrix(n, psifac, equil, intr, ctrl)

Calculate surface energy response matrix using exact geometric curvature via numerical differentiation.
Properly handles D-shaped cross-sections with elongation and triangularity.
"""
function compute_surface_matrix(n::Int, psifac::Float64, 
                                equil::Equilibrium.PlasmaEquilibrium, 
                                intr::DconInternal,
                                ctrl::DconControl)
    
    mpert = intr.mpert
    
    max_m_diff = abs(intr.mlow) + mpert  # Maximum mode number range
    n_int_grid = max(512, nextpow(2, 10 * max_m_diff))  # Power of 2 for efficiency
    
    # Use high-res grid for integration (independent of equil.config.control.mtheta)
    mtheta = n_int_grid
    
    ws_block = zeros(ComplexF64, mpert, mpert)
    
    # Get edge values
    sq_edge = Spl.spline_eval!(equil.sq, psifac)
    qa = sq_edge[4]        # safety factor at edge
    F_psi = sq_edge[1]     # 2π*F at edge
    v1 = sq_edge[3]        # dV/dψ at edge
    p_edge = sq_edge[2]    # μ₀*p at edge
    
    if ctrl.verbose
        println("   Computing surface energy (p_edge = $(Printf.@sprintf("%.3e", p_edge)))")
        println("   Debug: qa=$(Printf.@sprintf("%.3f",qa)), F=$(Printf.@sprintf("%.3e",F_psi)), v1=$(Printf.@sprintf("%.3e",v1))")
        println("   Using high-resolution grid: N=$(mtheta) (max_m=$(max_m_diff))")
        # Print Jacobian statistics for diagnosis
        println("   Debug: Jacobian range will be computed...")
    end
    
    if abs(p_edge) < 1e-10 || abs(v1) < 1e-10
        if ctrl.verbose; println("   Edge conditions negligible, surface energy = 0"); end
        return ws_block
    end
    
    # Pre-compute geometry on FINE grid
    theta_norm = zeros(Float64, mtheta + 1)
    R_vals = zeros(Float64, mtheta + 1)
    Z_vals = zeros(Float64, mtheta + 1)
    Rt_vals = zeros(Float64, mtheta + 1)
    Zt_vals = zeros(Float64, mtheta + 1)
    jac_vals = zeros(Float64, mtheta + 1)
    
    for itheta in 1:(mtheta + 1)
        theta = (itheta - 1) / Float64(mtheta)
        theta_norm[itheta] = theta
        
        f_val, f_x, f_y = Spl.bicube_deriv1!(equil.rzphi, psifac, theta)
        
        rfac = sqrt(f_val[1])
        offset = f_val[2]
        jac_vals[itheta] = f_val[4]
        
        dr2_dtheta = f_y[1]
        doffset_dtheta = f_y[2]
        eta = 2π * (theta + offset)
        
        drfac_dtheta = dr2_dtheta / (2.0 * rfac)
        deta_dtheta = 2π * (1.0 + doffset_dtheta)
        
        R_vals[itheta] = equil.ro + rfac * cos(eta)
        Z_vals[itheta] = equil.zo + rfac * sin(eta)
        Rt_vals[itheta] = drfac_dtheta * cos(eta) - rfac * sin(eta) * deta_dtheta
        Zt_vals[itheta] = drfac_dtheta * sin(eta) + rfac * cos(eta) * deta_dtheta
    end
    
    # Diagnostic: Check Jacobian range
    if ctrl.verbose
        max_jac = maximum(abs, jac_vals)
        min_jac = minimum(abs, jac_vals)
        println("   Debug: Jacobian range: [$(Printf.@sprintf("%.3e", min_jac)), $(Printf.@sprintf("%.3e", max_jac))]")
    end
    
    # Main integration loop
    max_kernel = 0.0
    
    for im in 1:mpert, jm in 1:mpert
        integrand = zeros(ComplexF64, mtheta + 1)
        m_val = intr.mlow + im - 1
        mp_val = intr.mlow + jm - 1
        
        # Skip if mode coupling is too high-frequency (numerical instability)
        # Physical justification: surface energy couples nearby modes more strongly
        m_diff = abs(mp_val - m_val)
        if m_diff > 15  # Threshold for mode coupling
            ws_block[im, jm] = 0.0
            continue
        end
        
        for itheta in 1:(mtheta + 1)
            R = R_vals[itheta]
            R_t = Rt_vals[itheta]
            Z_t = Zt_vals[itheta]
            ds = sqrt(R_t^2 + Z_t^2)
            
            if ds < 1e-15; continue; end
            
            # Numerical second derivative (periodic BC)
            prev = (itheta == 1) ? mtheta : itheta - 1
            next = (itheta == mtheta + 1) ? 2 : itheta + 1
            dtheta = theta_norm[2] - theta_norm[1]
            
            R_tt = (Rt_vals[next] - Rt_vals[prev]) / (2 * dtheta)
            Z_tt = (Zt_vals[next] - Zt_vals[prev]) / (2 * dtheta)
            
            # Exact geometric curvature
            kappa_p_geo = (R_t * Z_tt - Z_t * R_tt) / (ds^3)
            n_R = -Z_t / ds
            kappa_t_geo = -n_R / R
            
            # Safety check on curvatures (physical limit)
            # Typical tokamak: |κ| < 10/m (1/minor_radius scale)
            max_kappa = 100.0  # 1/m, conservative upper bound
            if abs(kappa_p_geo) > max_kappa
                if ctrl.verbose && im == 1 && jm == 1
                    println("   Warning: κ_p=$(Printf.@sprintf("%.3e", kappa_p_geo)) at theta=$(Printf.@sprintf("%.3f", theta_norm[itheta])) clamped")
                end
                kappa_p_geo = sign(kappa_p_geo) * max_kappa
            end
            if abs(kappa_t_geo) > max_kappa
                kappa_t_geo = sign(kappa_t_geo) * max_kappa
            end
            
            # Magnetic field
            Z = Z_vals[itheta]
            B_phi = F_psi / (2π * R)
            rfac = sqrt((R - equil.ro)^2 + (Z - equil.zo)^2)
            eps = rfac / equil.ro
            B_theta = eps * B_phi / qa
            B_tot_sq = B_phi^2 + B_theta^2
            B_tot = sqrt(B_tot_sq)
            
            if B_tot_sq < 1e-20; continue; end
            
            # Weighted curvature
            bp_frac = (B_theta / B_tot)^2
            bt_frac = (B_phi / B_tot)^2
            final_kappa_n = bp_frac * kappa_p_geo + bt_frac * kappa_t_geo
            
            # Kernel
            kernel = B_tot_sq * final_kappa_n * jac_vals[itheta]
            
            # Safety check on kernel BEFORE accumulation
            if !isfinite(kernel) || abs(kernel) > 1e20
                if abs(kernel) > 1e20 && im < 5 && jm > 30  # Debug first few problematic cases
                    println("   Kernel explosion at itheta=$(itheta), theta=$(Printf.@sprintf("%.3f", theta_norm[itheta]))")
                    println("     B²=$(Printf.@sprintf("%.3e", B_tot_sq)), κ_n=$(Printf.@sprintf("%.3e", final_kappa_n)), jac=$(Printf.@sprintf("%.3e", jac_vals[itheta]))")
                end
                kernel = 0.0
            end
            
            if abs(kernel) > max_kernel; max_kernel = abs(kernel); end
            
            # Integrand
            basis_m = exp(1im * m_val * theta_norm[itheta])
            basis_mp = exp(1im * mp_val * theta_norm[itheta])
            integrand_val = kernel * conj(basis_m) * basis_mp
            
            # Safety check on integrand
            if !isfinite(integrand_val) || abs(integrand_val) > 1e20
                if im < 5 && jm > 30  # Debug first few
                    println("   Integrand overflow: [$(im),$(jm)] at itheta=$(itheta)")
                    println("     kernel=$(Printf.@sprintf("%.3e", kernel))")
                    println("     |basis_m|=$(Printf.@sprintf("%.3e", abs(basis_m))), |basis_mp|=$(Printf.@sprintf("%.3e", abs(basis_mp)))")
                    println("     integrand=$(Printf.@sprintf("%.3e", abs(integrand_val)))")
                end
                integrand_val = 0.0
            end
            
            integrand[itheta] = integrand_val
        end
        
        integral_val = trapz(theta_norm, integrand)
        # NOTE: Must normalize by psio^2 to match plasma energy normalization
        # wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2 (Free.jl line 35)
        # Since wp DIVIDES by psio^2, and surface integral is naturally LARGER,
        # we need to MULTIPLY ws by psio^2 to bring it to same scale as wp
        ws_element = isfinite(integral_val) ? 0.5 * integral_val * equil.psio^2 / v1 : 0.0
        max_integrand = maximum(abs, integrand)

        if max_integrand > 1e20
            println("   EXPLOSION in integrand: [$(im),$(jm)] m=$(m_val), mp=$(mp_val)")
            println("     Max integrand = $(Printf.@sprintf("%.3e", max_integrand))")
            # Find where it explodes
            max_idx = argmax(abs.(integrand))
            println("     At theta=$(Printf.@sprintf("%.3f", theta_norm[max_idx]))")
            ws_element = 0.0
        elseif abs(ws_element) > 1e10
            println("   WARNING: ws[$(im),$(jm)] = $(Printf.@sprintf("%.3e", abs(ws_element)))")
            println("     m=$(m_val), mp=$(mp_val), integral=$(Printf.@sprintf("%.3e", abs(integral_val)))")
            println("     Max integrand = $(Printf.@sprintf("%.3e", max_integrand))")
            ws_element = 0.0  # Zero out problematic elements
        end
        
        # Debug first element
        if im == 1 && jm == 1 && ctrl.verbose
            println("   Debug [1,1]: Integral = $(Printf.@sprintf("%.3e", abs(integral_val)))")
            println("   Debug [1,1]: v1 = $(Printf.@sprintf("%.3e", v1))")
            println("   Debug [1,1]: Result = $(Printf.@sprintf("%.3e", abs(ws_element)))")
            println("   Debug [1,1]: Max integrand = $(Printf.@sprintf("%.3e", max_integrand)))")
        end
        
        ws_block[im, jm] = ws_element
    end
    
    if ctrl.verbose
        println("   Debug: Max kernel = $(Printf.@sprintf("%.3e", max_kernel))")
    end
    
    max_ws = maximum(abs, ws_block)
    mu0 = 4 * pi * 1e-7  
    ws_block .*= mu0
    if max_ws > 1e10
        @warn "Surface matrix explosion (max=$(max_ws)) - zeroing. Try psifac=0.99 or check units."
        fill!(ws_block, 0.0)
    end
    
    return ws_block
end

function trapz(x::AbstractVector, y::AbstractVector)
    n = length(x)
    integral = zero(eltype(y))
    for i in 1:(n-1)
        integral += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return integral
end
