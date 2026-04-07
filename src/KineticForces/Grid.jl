"""
    Grid

Grid management for KineticForces calculations.
Power-law grid spacing is available via `Utilities.powspace()`.
"""

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
