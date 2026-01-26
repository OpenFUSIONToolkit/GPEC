"""
Surface energy calculation module for DCON.

Implements calculation of surface energy contribution δW_s at the plasma edge
when finite pressure exists at the boundary.

Formula:
    δW_s = (1/2) ∫ (n·ξ)² × n·[∇(P + B²/2)] dS
         = (1/2) ∫ (n·ξ)² × B²(a) × (n·κ) dS

where:
    - n·ξ = normal displacement at edge
    - B(a) = magnetic field magnitude at plasma edge  
    - n·κ = normal curvature
    - [·] = discontinuity across plasma-vacuum interface
"""

"""
    compute_surface_matrix(n::Int, psifac::Float64, equil, intr) -> ws_block

Calculate surface energy response matrix for toroidal mode number n.

# Arguments
- `n::Int`: Toroidal mode number
- `psifac::Float64`: Flux surface value at plasma boundary (psilim)
- `equil::Equilibrium.PlasmaEquilibrium`: Equilibrium data
- `intr::DconInternal`: Internal DCON parameters

# Returns
- `ws_block::Matrix{ComplexF64}`: Surface response matrix (mpert × mpert) for this n

# Physics
The surface energy arises from the pressure discontinuity at the plasma-vacuum
interface when edge pressure is finite. The formula is:

    δW_s = (1/2) ∫ (n·ξ)² × B²(a) × (n·κ) dS

This is discretized into a matrix form where ws[m,m'] represents the coupling
between poloidal modes m and m' through the surface energy.
"""
function compute_surface_matrix(n::Int, psifac::Float64, 
                                equil::Equilibrium.PlasmaEquilibrium, 
                                intr::DconInternal)
    
    mpert = intr.mpert
    mtheta = equil.config.control.mtheta
    ws_block = zeros(ComplexF64, mpert, mpert)
    
    # Get edge quantities from equilibrium
    sq_vals = Spl.spline_eval!(equil.sq, psifac)
    p_edge = sq_vals[2]     # mu0 * p at edge
    v1 = sq_vals[3]         # dV/dpsi
    qa = sq_vals[4]         # safety factor at edge
    F_edge = sq_vals[1]     # 2π*F at edge
    
    # Check if edge pressure is significant
    # TODO: Surface energy calculation has numerical issues, temporarily disabled
    @warn "Surface energy calculation is experimental and currently disabled"
    return ws_block
    
    if abs(p_edge) < 1e-12
        return ws_block  # Return zeros if no pressure at edge
    end
    
    # Check if we have valid B field data
    if equil.params.b0 === nothing || isnan(equil.params.b0) || equil.params.b0 == 0.0
        @warn "Invalid B0 field, setting surface energy to zero"
        return ws_block
    end
    
    # Prepare arrays for edge geometry and fields
    theta_norm = Vector(equil.rzphi.ys)
    B_edge = zeros(Float64, mtheta + 1)        # |B| at edge
    kappa_n = zeros(Float64, mtheta + 1)       # n·κ (normal curvature)
    jac = zeros(Float64, mtheta + 1)           # Jacobian
    
    # Compute edge quantities at each poloidal angle
    for itheta in 1:(mtheta + 1)
        theta = theta_norm[itheta]
        
        # Get position and derivatives from bicubic spline
        f = Spl.bicube_eval!(equil.rzphi, psifac, theta)
        
        rfac = sqrt(f[1])           # sqrt(R²) coordinate
        offset = f[2]               # poloidal angle offset
        jac[itheta] = f[4]          # Jacobian
        
        # Compute position in (R,Z) coordinates
        eta = 2π * (theta + offset)
        R = equil.ro + rfac * cos(eta)
        Z = equil.zo + rfac * sin(eta)
        
        # Compute magnetic field magnitude at edge
        # B_θ ≈ F/(2πR), B_φ from equilibrium toroidal field
        B_theta = F_edge / (2π * R)
        B_phi = equil.params.b0 * equil.ro / R  # Simple toroidal field approximation
        B_psi = 0.0  # Radial field component (small at edge)
        
        B_sq = B_theta^2 + B_phi^2 + B_psi^2
        B_edge[itheta] = sqrt(B_sq)
        
        # Safety check for division by zero
        if B_sq < 1e-20
            kappa_n[itheta] = 0.0
        else
            # Compute normal curvature n·κ
            # Simplified: dominant contribution from toroidal curvature
            # κ_R ≈ -B_φ² / (R * B²) for major radius curvature
            kappa_n[itheta] = -B_phi^2 / (R * B_sq)
        end
    end
    
    # Build surface matrix ws[m,m']
    # The surface kernel connects modes through the geometry and field at edge
    for im in 1:mpert
        for jm in 1:mpert
            integrand = zeros(ComplexF64, mtheta + 1)
            
            # Mode numbers
            m_val = intr.mlow + im - 1
            mp_val = intr.mlow + jm - 1
            
            for itheta in 1:(mtheta + 1)
                theta = theta_norm[itheta]
                
                # Fourier basis functions for modes m and m'
                # The displacement has form: ξ ~ exp(i*(m*θ - n*φ))
                basis_m = exp(1im * m_val * theta)
                basis_mp = exp(1im * mp_val * theta)
                
                # Surface energy kernel: B²(θ) × (n·κ)(θ) × Jacobian
                kernel = B_edge[itheta]^2 * kappa_n[itheta] * jac[itheta]
                
                # The integrand for surface energy matrix element
                integrand[itheta] = kernel * conj(basis_m) * basis_mp
            end
            
            # Integrate over poloidal angle using trapezoidal rule
            # Factor of 0.5 from the δW_s formula
            integral_val = trapz(theta_norm, integrand)
            
            # Safety check for NaN/Inf
            if !isfinite(integral_val)
                ws_block[im, jm] = 0.0
            else
                ws_block[im, jm] = 0.5 * integral_val / v1
            end
        end
    end
    
    return ws_block
end

"""
    trapz(x::AbstractVector, y::AbstractVector)

Trapezoidal integration of function y over coordinates x.

# Arguments
- `x::AbstractVector`: Integration coordinates
- `y::AbstractVector`: Function values

# Returns
- Integral value (same type as elements of y)
"""
function trapz(x::AbstractVector, y::AbstractVector)
    n = length(x)
    @assert n == length(y) "x and y must have same length"
    
    integral = zero(eltype(y))
    for i in 1:(n-1)
        integral += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    
    return integral
end
