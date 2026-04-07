"""
    Grid

Grid management and manipulation for PENTRC calculations.
Handles equilibrium grids, input grids, and LSODE integration grids.
"""

"""
    powspace_sub(x_min::Float64, x_max::Float64, npower::Int, nx::Int,
                 spacing::String)::Matrix{Float64}

Power-law spaced grid generation with analytic derivative calculation.
Creates a grid with power-law spacing (more points near boundaries).

Based on [Logan, GPEC Documentation] and NASA/TM paper on grid generation.
Uses analytic derivatives for powers 1-9, numeric integration for others.

# Arguments
- `x_min, x_max::Float64`: Grid boundaries
- `npower::Int`: Power law exponent (1=linear, 2=quadratic, ..., 9=high resolution)
- `nx::Int`: Number of grid points
- `spacing::String`: "lower", "upper", or "both" (symmetric around midpoint)

# Returns
- `Matrix{Float64}`: Grid with shape (2, nx) where:
  - Row 1: Grid point positions in [x_min, x_max]
  - Row 2: Derivative dr/d(norm) w.r.t. unit linear space

# Algorithm
Maps normalized auxiliary space y ∈ [-1,1] using x = (x_min + x_max)/2 + (x_max - x_min)/2 * ∫F(y)
where F(y) = |1-y²|^npower and antiderivatives are computed analytically.
"""
function powspace_sub(x_min::Float64, x_max::Float64, npower::Int, nx::Int,
                     spacing::String)::Matrix{Float64}
    
    result = zeros(Float64, 2, nx)
    
    # Step 1: Create auxiliary space based on endpoint concentration
    y = zeros(Float64, nx)
    if spacing == "lower"
        y = range(-1.0, 0.0, length=nx)
    elseif spacing == "upper"
        y = range(0.0, 1.0, length=nx)
    elseif spacing == "both"
        y = range(-1.0, 1.0, length=nx)
    else
        error("Unknown spacing type: $spacing")
    end
    
    # Step 2: Compute weight function and its integral (antiderivative)
    # weight = |y² - 1|^npower, antiderivative computed via case statement
    
    F = zeros(Float64, nx)  # Antiderivative F(y)
    deriv = zeros(Float64, nx)  # Weight function derivative
    
    for i in eachindex(y)
        deriv[i] = abs((y[i] - 1) * (y[i] + 1))^npower
        
        # Compute F(y) = ∫ F(y) dy with analytic expressions for powers 1-9
        yi = y[i]
        if npower == 1
            F[i] = -yi + yi^3 / 3
        elseif npower == 2
            F[i] = yi - 2*yi^3 / 3 + yi^5 / 5
        elseif npower == 3
            F[i] = -yi + yi^3 - 3*yi^5 / 5 + yi^7 / 7
        elseif npower == 4
            F[i] = yi - 4*yi^3 / 3 + 6*yi^5 / 5 - 4*yi^7 / 7 + yi^9 / 9
        elseif npower == 5
            F[i] = -yi + 5*yi^3 / 3 - 2*yi^5 + 10*yi^7 / 7 - 5*yi^9 / 9 + yi^11 / 11
        elseif npower == 6
            F[i] = yi - 2*yi^3 + 3*yi^5 - 20*yi^7 / 7 + 5*yi^9 / 3 - 6*yi^11 / 11 + yi^13 / 13
        elseif npower == 7
            F[i] = -yi + 7*yi^3 / 3 - 21*yi^5 / 5 + 5*yi^7 - 35*yi^9 / 9 + 21*yi^11 / 11 - 7*yi^13 / 13 + yi^15 / 15
        elseif npower == 8
            F[i] = yi - 8*yi^3 / 3 + 28*yi^5 / 5 - 8*yi^7 + 70*yi^9 / 9 - 56*yi^11 / 11 + 28*yi^13 / 13 - 8*yi^15 / 15 + yi^17 / 17
        elseif npower == 9
            F[i] = -yi + 3*yi^3 - 36*yi^5 / 5 + 12*yi^7 - 14*yi^9 + 126*yi^11 / 11 - 84*yi^13 / 13 + 12*yi^15 / 5 - 9*yi^17 / 17 + yi^19 / 19
        else
            # For powers > 9, use numeric integration warning
            @warn "Power $npower not in analytic database, using numeric integration" maxlog=1
            # Use Simpson's rule for integration: ∫ |y²-1|^p dy
            F[i] = numeric_integral(y[i], npower)
        end
    end
    
    # Step 3: Normalize the grid to physical space [x_min, x_max]
    # Stretch and shift F to fill the domain
    delta_y = F[nx] - F[1]  # Range of F values
    delta_x = x_max - x_min  # Physical range
    
    result[2, :] = deriv .* (y[nx] - y[1])  # Spacing: weight * range of y
    result[1, :] = (F .- F[1]) .* (delta_x / delta_y) .+ x_min  # Scaled and shifted grid
    
    return result
end

# Helper function for numeric integration (fallback for powers > 9)
function numeric_integral(y::Float64, npower::Int)::Float64
    # Simple trapezoidal rule from 0 to y
    if y == 0.0
        return 0.0
    end
    
    # Use 100 points for numeric integration
    npts = 100
    y_int = range(0.0, y, length=npts)
    f_int = abs.((y_int .- 1) .* (y_int .+ 1)) .^ npower
    
    # Trapezoidal rule
    integral = (f_int[1] + f_int[npts]) / 2
    for i in 2:(npts-1)
        integral += f_int[i]
    end
    integral *= (y / (npts - 1))
    
    return integral
end


"""
    linspace_sub(x_min::Float64, x_max::Float64, nx::Int,
                 xs::Vector{Float64})

Fill uniform linear grid into provided array.
Convenience wrapper for uniform gridding.

# Arguments
- `x_min, x_max::Float64`: Grid boundaries
- `nx::Int`: Number of grid points  
- `xs::Vector{Float64}`: Output array to fill (must have length ≥ nx)
"""
function linspace_sub(x_min::Float64, x_max::Float64, nx::Int,
                     xs::Vector{Float64})
    
    if length(xs) < nx
        error("Output array too small: need $nx elements, got $(length(xs))")
    end
    
    xs[1:nx] = range(x_min, x_max, length=nx)
    return nothing
end


"""
    setup_psi_grid(grid_type::String, equil, intr::KineticForcesInternal, 
                   ctrl::KineticForcesControl)::Vector{Float64}

Set up poloidal flux grid for calculations based on specified type.

# Arguments
- `grid_type::String`: One of "equil", "input", or "lsode"
- `equil`: Equilibrium structure
- `intr::KineticForcesInternal`: Internal state with equilibrium data
- `ctrl::KineticForcesControl`: Control parameters

# Returns
- `Vector{Float64}`: Poloidal flux grid points
"""
function setup_psi_grid(grid_type::String, equil, intr::KineticForcesInternal,
                       ctrl::KineticForcesControl)::Vector{Float64}
    
    psi_grid = Float64[]
    
    if grid_type == "equil"
        # Use equilibrium grid
        if isdefined(equil, :sq) && isdefined(equil.sq, :xs)
            psi_grid = copy(equil.sq.xs)
        else
            error("Equilibrium grid not available")
        end
        
    elseif grid_type == "input"
        # Use input displacement perturbation grid
        # TODO: Implement based on input file structure
        error("Input grid not yet implemented")
        
    elseif grid_type == "lsode"
        # Use LSODE integration grid
        # This is computed adaptively during integration
        psi_grid = ctrl.psilims  # Start with limits, will be populated during ODE solve
        
    else
        error("Unknown grid type: $grid_type")
    end
    
    # Filter to integration limits if needed
    mask = (psi_grid .>= ctrl.psilims[1]) .& (psi_grid .<= ctrl.psilims[2])
    psi_grid = psi_grid[mask]
    
    intr.grid_type = grid_type
    intr.psi_grid = psi_grid
    
    return psi_grid
end


"""
    get_psi_output_surfaces(ctrl::KineticForcesControl)::Vector{Float64}

Get list of psi surfaces where detailed output is requested.

# Arguments
- `ctrl::KineticForcesControl`: Control parameters containing psi_out

# Returns
- `Vector{Float64}`: Valid psi values for output (0 < psi < 1)
"""
function get_psi_output_surfaces(ctrl::KineticForcesControl)::Vector{Float64}
    
    psi_valid = Float64[]
    
    for psi in ctrl.psi_out
        if 0 < psi < 1
            push!(psi_valid, psi)
        end
    end
    
    return psi_valid
end


"""
    validate_grid(psi_grid::Vector{Float64}, psilims::Vector{Float64})::Bool

Check that grid is valid (monotonic, within limits).

# Arguments
- `psi_grid::Vector{Float64}`: Grid to validate
- `psilims::Vector{Float64}`: Allowed limits [psi_min, psi_max]

# Returns
- `Bool`: True if valid, false otherwise
"""
function validate_grid(psi_grid::Vector{Float64}, psilims::Vector{Float64})::Bool
    
    # Check length
    if length(psi_grid) < 2
        return false
    end
    
    # Check monotonicity
    for i in 2:length(psi_grid)
        if psi_grid[i] <= psi_grid[i-1]
            return false
        end
    end
    
    # Check limits
    if psi_grid[1] < psilims[1] || psi_grid[end] > psilims[2]
        return false
    end
    
    return true
end


"""
    refine_grid(psi_grid::Vector{Float64}, nref::Int)::Vector{Float64}

Refine grid by adding intermediate points (linear interpolation).

# Arguments
- `psi_grid::Vector{Float64}`: Original grid
- `nref::Int`: Number of points to insert between each pair

# Returns
- `Vector{Float64}`: Refined grid
"""
function refine_grid(psi_grid::Vector{Float64}, nref::Int)::Vector{Float64}
    
    if nref <= 0
        return copy(psi_grid)
    end
    
    psi_refined = Float64[]
    
    for i in 1:(length(psi_grid)-1)
        # Add original point
        push!(psi_refined, psi_grid[i])
        
        # Add intermediate points
        for j in 1:nref
            psi_interp = psi_grid[i] + j * (psi_grid[i+1] - psi_grid[i]) / (nref + 1)
            push!(psi_refined, psi_interp)
        end
    end
    
    # Add final point
    push!(psi_refined, psi_grid[end])
    
    return psi_refined
end


"""
    coarsen_grid(psi_grid::Vector{Float64}, factor::Int)::Vector{Float64}

Coarsen grid by keeping every n-th point.

# Arguments
- `psi_grid::Vector{Float64}`: Original grid
- `factor::Int`: Keep every factor-th point

# Returns
- `Vector{Float64}`: Coarsened grid
"""
function coarsen_grid(psi_grid::Vector{Float64}, factor::Int)::Vector{Float64}
    
    if factor <= 1
        return copy(psi_grid)
    end
    
    indices = vcat(1:factor:length(psi_grid), length(psi_grid))
    return psi_grid[unique(indices)]
end
