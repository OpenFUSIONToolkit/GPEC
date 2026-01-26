# Surface Energy Implementation Plan

## Goal
Implement surface energy contribution δW_s ≠ 0 for cases with finite pressure at edge and/or skin current discontinuity.

## Physics Formula (from Surface.tex equation 84-86)

```
δW_s = (1/2) ∫ (n·ξ)² × n·[∇(P + B²/2)] dS
     = (1/2) ∫ (n·ξ)² × B²(a) × (n·κ) dS
```

where:
- `n·ξ` = normal displacement at edge
- `B(a)` = magnetic field magnitude at plasma edge
- `n·κ` = normal curvature
- `[·]` = discontinuity across plasma-vacuum interface
- Integration over plasma surface S_p at psilim

## Current Code Structure

### Energy Components (Free.jl)
Currently:
```julia
wp = U₂ * U₁⁻¹  (plasma response matrix)
wv              (vacuum response matrix from Green's function)
wt = wp + wv    (total response matrix)
```

Need to add:
```julia
ws              (surface response matrix)
wt = wp + wv + ws
```

### Matrix Structure
- `wp`: (numpert_total × numpert_total) - from ODE integration
- `wv`: (numpert_total × numpert_total) - from Vacuum.compute_vacuum_response
- `ws`: (numpert_total × numpert_total) - **NEW: from surface contribution**

Block diagonal structure for each n:
- Each n has (mpert × mpert) block
- Total: (npert blocks) along diagonal

## Implementation Plan

---

## Phase 1: Data Structure Extensions

### 1.1 DconStructs.jl - Extend VacuumData

**File**: `src/DCON/DconStructs.jl`

**Location**: ~line 272 in `VacuumData` struct

**Changes**:
```julia
@kwdef mutable struct VacuumData
    mthvac::Int
    mpert::Int
    numpert_total::Int

    wt::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wt0::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    wv::Array{ComplexF64,2} = Array{ComplexF64}(undef, numpert_total, numpert_total)
    
    # ADD: Surface response matrix
    ws::Array{ComplexF64,2} = zeros(ComplexF64, numpert_total, numpert_total)
    
    ep::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    ev::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    et::Vector{ComplexF64} = Vector{ComplexF64}(undef, numpert_total)
    
    # ADD: Surface energy eigenvalues
    es::Vector{ComplexF64} = zeros(ComplexF64, numpert_total)

    grri::Array{Float64,2} = Array{Float64}(undef, 2 * (mthvac + 5), 2 * mpert)
    xzpts::Array{Float64,2} = Array{Float64}(undef, mthvac + 5, 4)
end
```

### 1.2 DconControl - Add surf_flag

**File**: `src/DCON/DconStructs.jl`

**Location**: ~line 140-230 in `DconControl` struct

**Changes**:
```julia
@kwdef mutable struct DconControl
    # ... existing fields ...
    vac_flag::Bool = true
    
    # ADD: Surface energy calculation flag
    surf_flag::Bool = false  # Default false for backward compatibility
    
    # ... rest of fields ...
end
```

---

## Phase 2: Surface Energy Calculation Module

### 2.1 Create Surface.jl

**File**: `src/DCON/Surface.jl` (NEW FILE)

**Purpose**: Calculate surface energy response matrix ws

**Key Functions**:

```julia
"""
    compute_surface_matrix(n::Int, psifac::Float64, equil, intr) -> ws_block

Calculate surface energy response matrix for toroidal mode number n.

Formula:
    δW_s = (1/2) ∫ (n·ξ)² × B²(a) × (n·κ) dS

Returns:
    ws_block: (mpert × mpert) surface response matrix for this n
"""
function compute_surface_matrix(n::Int, psifac::Float64, 
                                equil::Equilibrium.PlasmaEquilibrium, 
                                intr::DconInternal)
    
    mpert = intr.mpert
    mtheta = equil.config.control.mtheta
    ws_block = zeros(ComplexF64, mpert, mpert)
    
    # 1. Get edge quantities
    sq_vals = Spl.spline_eval!(equil.sq, psifac)
    p_edge = sq_vals[2]     # mu0 * p at edge
    v1 = sq_vals[3]         # dV/dpsi
    qa = sq_vals[4]         # safety factor
    
    # Check if edge pressure is significant
    if abs(p_edge) < 1e-12
        return ws_block  # Return zeros if no pressure at edge
    end
    
    # 2. Compute edge geometry and B field
    theta_norm = Vector(equil.rzphi.ys)
    B_edge = zeros(Float64, mtheta + 1)        # |B| at edge
    kappa_n = zeros(Float64, mtheta + 1)       # n·κ (normal curvature)
    jac = zeros(Float64, mtheta + 1)           # Jacobian
    
    for itheta in 1:(mtheta + 1)
        theta = theta_norm[itheta]
        
        # Get position and derivatives from bicubic spline
        f, fx, fy, fxy = Spl.bicube_all_derivs!(equil.rzphi, psifac, theta)
        
        rfac = sqrt(f[1])           # R coordinate
        offset = f[2]               # poloidal angle offset
        nu = f[3]                   # normal displacement factor
        jac[itheta] = f[4]          # Jacobian
        
        # Compute position in (R,Z) coordinates
        eta = 2π * (theta + offset)
        R = equil.ro + rfac * cos(eta)
        Z = equil.zo + rfac * sin(eta)
        
        # Compute magnetic field components and magnitude
        # B_θ = F/R, B_φ = I(ψ)/R (from equilibrium)
        # Need to implement get_magnetic_field_at_edge(...)
        B_theta, B_phi, B_psi = compute_B_field_components(
            equil, psifac, theta, R, Z
        )
        B_edge[itheta] = sqrt(B_theta^2 + B_phi^2 + B_psi^2)
        
        # Compute normal curvature n·κ
        # κ = (B·∇)B / B²
        kappa_n[itheta] = compute_normal_curvature(
            equil, psifac, theta, R, Z, B_theta, B_phi, B_psi
        )
    end
    
    # 3. Build surface matrix ws[m,m']
    # This requires computing (n·ξ) for each m mode
    # The normal displacement relates to delta: delta = -f[3]/qa
    
    for im in 1:mpert
        for jm in 1:mpert
            integrand = zeros(ComplexF64, mtheta + 1)
            
            for itheta in 1:(mtheta + 1)
                # Surface energy kernel:
                # K(m,m',θ) = B²(θ) × (n·κ)(θ) × basis_m(θ) × basis_m'(θ)
                
                # Basis functions for Fourier modes
                # For mode m: exp(i*m*theta_eff)
                m_val = intr.mlow + im - 1
                mp_val = intr.mlow + jm - 1
                
                theta_eff = theta_norm[itheta]  # effective poloidal angle
                basis_m = exp(1im * m_val * theta_eff)
                basis_mp = exp(1im * mp_val * theta_eff)
                
                # Surface kernel (real part gives stabilizing effect)
                kernel = B_edge[itheta]^2 * kappa_n[itheta] * jac[itheta]
                integrand[itheta] = kernel * conj(basis_m) * basis_mp
            end
            
            # Integrate over poloidal angle
            ws_block[im, jm] = 0.5 * trapz(theta_norm, integrand) / v1
        end
    end
    
    return ws_block
end

"""
    compute_B_field_components(equil, psi, theta, R, Z)

Compute magnetic field components at edge location.
"""
function compute_B_field_components(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64, theta::Float64, R::Float64, Z::Float64
)
    # B_θ component (from F and geometry)
    F = Spl.spline_eval!(equil.sq, psi)[1]  # 2π*F
    B_theta = F / (2π * R)
    
    # B_φ component (toroidal field from equilibrium)
    # This requires equilibrium pressure gradient terms
    # For now, approximate as primary toroidal field
    B_phi = equil.params.b0 * equil.ro / R
    
    # B_ψ component (poloidal field from flux surfaces)
    # ∂ψ/∂R, ∂ψ/∂Z terms from spline derivatives
    # Requires implementation of equilibrium field calculation
    B_psi = 0.0  # Placeholder - needs proper implementation
    
    return B_theta, B_phi, B_psi
end

"""
    compute_normal_curvature(equil, psi, theta, R, Z, B_theta, B_phi, B_psi)

Compute normal curvature n·κ where κ = (B·∇)B / B².
"""
function compute_normal_curvature(
    equil::Equilibrium.PlasmaEquilibrium,
    psi::Float64, theta::Float64, 
    R::Float64, Z::Float64,
    B_theta::Float64, B_phi::Float64, B_psi::Float64
)
    # Curvature vector: κ = (B·∇)B / B²
    # Normal curvature: n·κ where n is normal to flux surface
    
    # This requires computing gradients of B components
    # Placeholder - needs detailed equilibrium field derivatives
    
    B_mag = sqrt(B_theta^2 + B_phi^2 + B_psi^2)
    
    # Simplified curvature (needs full implementation)
    # Major radius curvature term (dominant in tokamak)
    kappa_R = -B_phi^2 / (R * B_mag^2)
    
    return kappa_R  # Placeholder
end

"""
    trapz(x, y)

Trapezoidal integration.
"""
function trapz(x::AbstractVector, y::AbstractVector)
    n = length(x)
    integral = zero(eltype(y))
    for i in 1:(n-1)
        integral += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return integral
end
```

---

## Phase 3: Integrate Surface Energy in Free.jl

### 3.1 Modify free_run! function

**File**: `src/DCON/Free.jl`

**Location**: Lines 14-140 (free_run! function)

**Changes**:

```julia
function free_run!(odet::OdeState, ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, 
                   ffit::FourFitVars, intr::DconInternal, wall_settings::Vacuum.WallShapeSettings)

    normalize = true
    complex_flag = true
    wall_flag = false
    ahg_file = "ahg2msc_dcon.out"

    # Initializations and allocations
    vac = VacuumData(ctrl.mthvac, intr.mpert, intr.numpert_total)
    etemp = zeros(ComplexF64, intr.numpert_total)
    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wpt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wvt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    wst = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)  # ADD

    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix
    if ctrl.ode_flag
        @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2
    end

    # Compute vacuum and surface response matrices
    wv_block = zeros(ComplexF64, intr.mpert, intr.mpert)
    ws_block = zeros(ComplexF64, intr.mpert, intr.mpert)  # ADD
    
    for ipert_n in 1:intr.npert
        n = ipert_n - 1 + intr.nlow
        
        # Set VACUUM run parameters and boundary shape
        vac_inputs = set_vacuum_inputs(intr.psilim, n, equil, intr, ctrl)
        fill!(vac.grri, 0.0)
        fill!(vac.xzpts, 0.0)

        farwall_flag = wall_settings.shape == "nowall" ? true : false

        # Output debug data if requested
        if intr.debug_settings.output_benchmark_data
            @info "Outputting top level vacuum debug data for n = $n"
            benchmark_inputs = VacuumBenchmarkInputs(
                    wv_block, intr.mpert, equil.config.control.mtheta, ctrl.mthvac, 
                    complex_flag, vac_inputs.kernelsign, wall_flag,
                    farwall_flag, vac.grri, vac.xzpts, ahg_file, intr.dir_path,
                    vac_inputs, wall_settings,
                    n, ipert_n, intr.psilim
            )
            @save "vacuum_response_inputs.jld2" benchmark_inputs
        end

        # Compute vacuum energy matrix
        wv_block, vac.grri, vac.xzpts = Vacuum.compute_vacuum_response(vac_inputs, wall_settings)

        # ADD: Compute surface energy matrix
        if ctrl.surf_flag
            # Check if edge pressure is significant
            p_edge = Spl.spline_eval!(equil.sq, intr.psilim)[2]
            if abs(p_edge) > 1e-12
                if ctrl.verbose && ipert_n == 1
                    println("   Computing surface energy (p_edge = $(@sprintf("%.3e", p_edge)))")
                end
                ws_block = Surface.compute_surface_matrix(n, intr.psilim, equil, intr)
            else
                fill!(ws_block, 0.0)
            end
        else
            fill!(ws_block, 0.0)
        end

        # Scale vacuum and surface matrices by singfac = (m - n*qlim)
        singfac = collect(intr.mlow:intr.mhigh) .- (n * intr.qlim)
        @inbounds for ipert in 1:intr.mpert
            @views wv_block[ipert, :] .*= singfac[ipert]
            @views wv_block[:, ipert] .*= singfac[ipert]
            
            # ADD: Scale surface matrix same way
            if ctrl.surf_flag
                @views ws_block[ipert, :] .*= singfac[ipert]
                @views ws_block[:, ipert] .*= singfac[ipert]
            end
        end

        # Store blocks in full matrices
        idx_range = (ipert_n-1)*intr.mpert+1 : ipert_n*intr.mpert
        @views vac.wv[idx_range, idx_range] .= wv_block
        @views vac.ws[idx_range, idx_range] .= ws_block  # ADD

        Vacuum.unset_dcon_params()
    end

    # Compute complex energy eigenvalues and vectors
    # MODIFY: Include surface contribution
    vac.wt .= wp .+ vac.wv .+ vac.ws  # MODIFIED (was: wp + wv)
    vac.wt0 .= vac.wt
    Ev = eigen(vac.wt)
    vac.et .= Ev.values
    eindex = sortperm(real.(vac.et); rev=true)

    etemp .= vac.et
    # Rearrange wt columns for descending real eigenvalues
    for ipert in 1:intr.numpert_total
        vac.wt[:, ipert] .= Ev.vectors[:, eindex[intr.numpert_total+1-ipert]]
        vac.et[ipert] = etemp[eindex[intr.numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy
    if normalize
        for isol in 1:intr.numpert_total
            norm = 0.0 + 0.0im
            for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
                ipert = (ipert_n - 1) * intr.mpert + ipert_m
                jpert = (ipert_n - 1) * intr.mpert + jpert_m
                norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * vac.wt[ipert, isol] * conj(vac.wt[jpert, isol])
            end
            norm /= v1
            vac.wt[:, isol] ./= sqrt(norm)
            vac.et[isol] /= norm
        end
    end

    # Normalize phase
    imax = 0
    for isol in 1:intr.numpert_total
        imax = argmax(abs.(vac.wt[:, isol]))
        phase = abs(vac.wt[imax, isol]) / vac.wt[imax, isol]
        vac.wt[:, isol] .*= phase
    end

    # Compute plasma, vacuum, and surface contributions
    # MODIFY: Add surface contribution
    wpt .= adjoint(vac.wt) * (wp * vac.wt)
    wvt .= adjoint(vac.wt) * (vac.wv * vac.wt)
    wst .= adjoint(vac.wt) * (vac.ws * vac.wt)  # ADD
    
    for ipert in 1:intr.numpert_total
        vac.ep[ipert] = wpt[ipert, ipert]
        vac.ev[ipert] = wvt[ipert, ipert]
        vac.es[ipert] = wst[ipert, ipert]  # ADD
    end

    # Normalize eigenvectors based on scaled wt
    coeffs = odet.u[:,:,1,end] \ (vac.wt .* (2π * equil.psio * 1e-3))
    for istep in 1:odet.step
        odet.u_store[:, :, 1, istep] .= odet.u_store[:, :, 1, istep] * coeffs
        odet.u_store[:, :, 2, istep] .= odet.u_store[:, :, 2, istep] * coeffs
        odet.ud_store[:, :, 1, istep] .= odet.ud_store[:, :, 1, istep] * coeffs
        odet.ud_store[:, :, 2, istep] .= odet.ud_store[:, :, 2, istep] * coeffs
    end

    # Write energies to screen
    # MODIFY: Add surface energy output
    if ctrl.verbose
        println("Least Stable Eigenmode Energies:")
        println("  Plasma  = ", (@sprintf "%+.3e %+.3ei" real(vac.ep[1]) imag(vac.ep[1])))
        if ctrl.surf_flag && abs(vac.es[1]) > 1e-15
            println("  Surface = ", (@sprintf "%+.3e %+.3ei" real(vac.es[1]) imag(vac.es[1])))
        end
        println("  Vacuum  = ", (@sprintf "%+.3e %+.3ei" real(vac.ev[1]) imag(vac.ev[1])))
        println("  Total   = ", (@sprintf "%+.3e %+.3ei" real(vac.et[1]) imag(vac.et[1])))
    end

    return vac
end
```

### 3.2 Modify free_compute_total function

**File**: `src/DCON/Free.jl`

**Location**: Lines 288-333 (free_compute_total function)

**Changes**:

```julia
function free_compute_total(equil::Equilibrium.PlasmaEquilibrium, ffit::FourFitVars, 
                           intr::DconInternal, odet::OdeState, ctrl::DconControl)  # ADD ctrl

    normalize = true

    wp = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)
    ws = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)  # ADD
    tot_eigvals = zeros(ComplexF64, intr.numpert_total)
    wt = zeros(ComplexF64, intr.numpert_total, intr.numpert_total)

    v1 = Spl.spline_eval!(equil.sq, intr.psilim)[3]

    # Compute plasma response matrix
    @views wp = (odet.u[:, :, 2] / odet.u[:, :, 1]) ./ equil.psio^2

    # Compute vacuum matrix from spline
    wv = reshape(Spl.spline_eval!(odet.wvmat_spline, odet.psifac), intr.numpert_total, intr.numpert_total)

    # ADD: Compute surface contribution if enabled
    if ctrl.surf_flag
        p_edge = Spl.spline_eval!(equil.sq, odet.psifac)[2]
        if abs(p_edge) > 1e-12
            # Need to compute ws at current psifac
            # For simplicity, could interpolate from stored ws or recompute
            # For now, approximate as zero in edge region (conservative)
            # TODO: Implement surface energy spline similar to wvmat_spline
        end
    end

    # Compute total energy matrix and eigen-decomposition
    wt .= wp .+ wv .+ ws  # MODIFIED (was: wp + wv)
    Ev = eigen(wt)

    # Sort eigenvalues and reorder columns of wt
    eindex = sortperm(real.(Ev.values); rev=true)
    for ipert in 1:intr.numpert_total
        wt[:, ipert] .= Ev.vectors[:, eindex[intr.numpert_total+1-ipert]]
        tot_eigvals[ipert] = Ev.values[eindex[intr.numpert_total+1-ipert]]
    end

    # Normalize eigenfunction and energy (only need the first eigenmode)
    if normalize
        isol = 1
        norm = 0.0 + 0.0im
        for ipert_n in 1:intr.npert, ipert_m in 1:intr.mpert, jpert_m in 1:intr.mpert
            ipert = (ipert_n - 1) * intr.mpert + ipert_m
            jpert = (ipert_n - 1) * intr.mpert + jpert_m
            norm += ffit.jmat[jpert_m-ipert_m+intr.mband+1] * wt[ipert, isol] * conj(wt[jpert, isol])
        end
        norm /= v1
        tot_eigvals[isol] /= norm
    end

    return tot_eigvals[1]
end
```

**Note**: This function also needs `ctrl` parameter passed from calling functions in Ode.jl

---

## Phase 4: Module Integration

### 4.1 Add Surface module to DCON.jl

**File**: `src/DCON/DCON.jl`

**Location**: After other includes (~line 15)

**Changes**:
```julia
module DCON

using TOML
using Printf
using LinearAlgebra
using JLD2
using ..Equilibrium
using ..Spl
using ..Vacuum

# Include all submodules
include("DconStructs.jl")
include("Utils.jl")
include("Mercier.jl")
include("Bal.jl")
include("Fourfit.jl")
include("Sing.jl")
include("Ode.jl")
include("FixedBoundaryStability.jl")
include("Free.jl")
include("Surface.jl")  # ADD
include("Main.jl")

# ... exports ...

end
```

---

## Phase 5: HDF5 Output

### 5.1 Modify write_outputs_to_HDF5

**File**: `src/DCON/Main.jl`

**Location**: Lines 196-338 (write_outputs_to_HDF5 function)

**Changes**:

```julia
function write_outputs_to_HDF5(ctrl::DconControl, equil::Equilibrium.PlasmaEquilibrium, 
                               intr::DconInternal, odet::OdeState, vac::Union{VacuumData, Nothing})

    h5open(joinpath(intr.dir_path, ctrl.HDF5_filename), "w") do out_h5

        # ... existing input/info/equil/splines/locstab/integration/singular writes ...

        # Write vacuum Data
        if ctrl.vac_flag
            out_h5["vacuum/wt"] = vac.wt
            out_h5["vacuum/wt0"] = vac.wt0
            out_h5["vacuum/ep"] = vac.ep
            out_h5["vacuum/ev"] = vac.ev
            out_h5["vacuum/et"] = vac.et
            out_h5["vacuum/x_plasma"] = vac.xzpts[:, 1]
            out_h5["vacuum/z_plasma"] = vac.xzpts[:, 2]
            out_h5["vacuum/x_wall"] = vac.xzpts[:, 3]
            out_h5["vacuum/z_wall"] = vac.xzpts[:, 4]
            
            # ADD: Surface energy data
            if ctrl.surf_flag
                out_h5["vacuum/ws"] = vac.ws
                out_h5["vacuum/es"] = vac.es
                
                # Save edge pressure for reference
                p_edge = Spl.spline_eval!(equil.sq, intr.psilim)[2]
                out_h5["vacuum/p_edge"] = p_edge
            end
        end
    end
end
```

---

## Phase 6: Testing and Validation

### 6.1 Test Cases

**Test 1: Zero Surface Energy (Validation)**
- Use existing Solovev example
- Set `surf_flag = false`
- Verify results unchanged from current implementation

**Test 2: Finite Pressure Edge**
- Use example with p(psilim) ≠ 0
- Set `surf_flag = true`
- Check that:
  - `es` is calculated
  - `et = ep + ev + es`
  - Surface contribution is stabilizing (typically Re(es) > 0)

**Test 3: DIIID-like Case**
- Use DIIID example
- Compare with/without surface energy
- Document magnitude of surface contribution

### 6.2 Validation Checks

1. **Energy conservation**: `et[i] ≈ ep[i] + ev[i] + es[i]` for each eigenmode
2. **Sign check**: Surface energy typically stabilizing (Re(es) > 0)
3. **Edge pressure dependence**: es → 0 as p_edge → 0
4. **Backward compatibility**: Results with `surf_flag=false` match original code

---

## Phase 7: Documentation

### 7.1 Update README.md

Document new `surf_flag` option in DCON_CONTROL section

### 7.2 Update dcon.toml examples

Add `surf_flag` to example input files:
```toml
[DCON_CONTROL]
surf_flag = false  # Set true to include surface energy contribution
```

### 7.3 Add Physics Documentation

Create or update documentation explaining:
- When surface energy matters (finite pressure edge, skin currents)
- Physical interpretation of n·κ term
- Expected magnitude of contribution

---

## Implementation Order

1. ✅ **Phase 1**: Data structures (DconStructs.jl)
2. **Phase 2**: Surface.jl module (start with simplified version)
3. **Phase 3**: Integrate into Free.jl
4. **Phase 4**: Module integration (DCON.jl)
5. **Phase 5**: HDF5 output
6. **Phase 6**: Testing with simple case
7. **Phase 7**: Documentation

---

## Open Questions / TODOs

### Critical Physics Questions:
1. **Magnetic field calculation**: Need proper implementation of B field components at edge
   - Current equilibrium structure has what B field data?
   - Need ∂B/∂R, ∂B/∂Z for curvature?

2. **Curvature calculation**: How to compute n·κ accurately?
   - Use equilibrium derivatives?
   - Analytic approximation for circular cross-section?

3. **Displacement basis functions**: How exactly does (n·ξ) couple to Fourier modes?
   - Is it through the `delta` parameter already computed?
   - Need relationship between u[:,:,1,end] and actual ξ

### Technical Questions:
1. **Edge region integration**: Should surface energy be included in `free_compute_total`?
   - If yes, need spline for ws(psi) similar to wv
   - If no, might underestimate edge stability

2. **Singfac scaling**: Should surface energy have same (m-nq) scaling?
   - Physics says yes (resonance dependence)
   - Need to verify

3. **Normalization**: Does surface energy affect normalization procedure?
   - Currently normalizes with J matrix
   - Surface term might need different treatment?

---

## Expected Results

When `surf_flag = true` and p_edge ≠ 0:
- Surface energy `es` should be computed
- Typically stabilizing: Re(es) > 0
- Magnitude: Usually |es| << |ev| (surface vs volume)
- Edge modes more affected than core modes

---

## Files to Modify

1. `src/DCON/DconStructs.jl` - Data structures
2. `src/DCON/Surface.jl` - NEW FILE
3. `src/DCON/Free.jl` - Integration logic
4. `src/DCON/DCON.jl` - Module include
5. `src/DCON/Main.jl` - HDF5 output
6. `src/DCON/Ode.jl` - Pass ctrl to free_compute_total (minor)
7. Example toml files - Add surf_flag documentation

---

## Estimated Effort

- Phase 1 (structures): 30 min
- Phase 2 (Surface.jl): 4-6 hours (main work)
- Phase 3 (Free.jl): 1-2 hours
- Phase 4 (integration): 30 min
- Phase 5 (output): 30 min
- Phase 6 (testing): 2-3 hours
- Phase 7 (docs): 1 hour

**Total: ~10-14 hours**

Main complexity is in getting the physics right in Surface.jl (B field, curvature, coupling).
