"""
    extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64) -> (r, z, ν)

Extract plasma surface geometry from equilibrium at specified flux coordinate.

Evaluates equilibrium bicubic spline around the flux surface to get R, Z coordinates
and computes the toroidal angle offset ν for vacuum calculations.

## Arguments

  - `equil`: Equilibrium solution with rzphi bicubic spline
  - `ψ`: Normalized flux coordinate (0 at axis, 1 at edge)

## Returns

Tuple of:

  - `r::Vector{Float64}`: R-coordinates around flux surface [mtheta]
  - `z::Vector{Float64}`: Z-coordinates around flux surface [mtheta]
  - `ν::Vector{Float64}`: Toroidal angle offset ν [mtheta]

## Implementation

The equilibrium bicubic spline `rzphi` stores:

  - rzphi.f[1] = r² (or rfac²)
  - rzphi.f[2] = offset (straight-fieldline poloidal angle offset from geometric poloidal angle)
  - rzphi.f[3] = ν (toroidal angle offset from geometric toroidal angle)
  - rzphi.f[4] = jac (Jacobian)

From these we compute:

  - r_minor = √(rzphi.f[1])
  - θ_cyl = 2π*(θ_sfl + rzphi.f[2])
  - R = R₀ + r_minor * cos(θ_cyl)
  - Z = Z₀ + r_minor * sin(θ_cyl)

## Reference

[Chance Phys. Plasmas 1997 2161]
Matches GPEC's ahg_write and gpvacuum_flxsurf approach (gpvacuum.f line 291-296)
"""
function extract_plasma_surface_at_psi(equil::Equilibrium.PlasmaEquilibrium, ψ::Float64)

    # Allocate output arrays
    mtheta = length(equil.rzphi_ys)
    r_minor = zeros(mtheta)
    θ_cyl = zeros(mtheta)
    ν = zeros(mtheta)

    # Evaluate equilibrium around the flux surface
    hint2d = (Ref(1), Ref(1))
    for (i, θ_sfl) in enumerate(equil.rzphi_ys)
        # Get minor radius, geometric poloidal angle, and toroidal angle offset
        r_minor[i] = sqrt(equil.rzphi_rsquared((ψ, θ_sfl); hint=hint2d))
        θ_cyl[i] = 2π * (θ_sfl + equil.rzphi_offset((ψ, θ_sfl); hint=hint2d))
        ν[i] = equil.rzphi_nu((ψ, θ_sfl); hint=hint2d)
    end

    # Compute R and Z using the computed cylindrical coordinates
    r = equil.ro .+ r_minor .* cos.(θ_cyl)
    z = equil.zo .+ r_minor .* sin.(θ_cyl)
    return r, z, ν
end

"""
    distribute_to_equal_arc_grid(xin, zin)

Given a set of points (xin, zin) that define a closed curve in 2D, redistribute these points to be equally spaced
in terms of arc length along the curve. This is useful for ensuring that points on a wall or plasma surface are
uniformly distributed in space rather than in the parameter θ. This performs the same function as eqarcw in the
Fortran code. We now use FastInterpolations instead of a manual Lagrange interpolation.

# Arguments

  - `xin::Vector{Float64}`: x-coordinates of the original points defining the curve (endpoint not included)
  - `zin::Vector{Float64}`: z-coordinates of the original points defining the curve (endpoint not included)

# Returns

  - `xout::Vector{Float64}`: x-coordinates of the redistributed points, equally spaced in arc length
  - `zout::Vector{Float64}`: z-coordinates of the redistributed points, equally spaced in arc length
"""
function distribute_to_equal_arc_grid(xin::Vector{Float64}, zin::Vector{Float64})

    mtheta = length(xin)
    dθ = 2π / mtheta
    θ_grid = range(; start=0, length=mtheta, step=dθ)

    # Build periodic splines for derivatives on the closed loop
    spline_x = cubic_interp(θ_grid, xin; bc=PeriodicBC(; endpoint=:exclusive))
    spline_z = cubic_interp(θ_grid, zin; bc=PeriodicBC(; endpoint=:exclusive))

    # Calculate cumulative arc length using numerical integration
    arc_length = zeros(mtheta + 1)
    for i in 1:mtheta
        # Use a mid-point derivative approximation
        theta_mid = θ_grid[i] + dθ / 2.0
        d_xin = spline_x(theta_mid; deriv=1)
        d_zin = spline_z(theta_mid; deriv=1)
        # Accumulate length: ds = (ds/dt) * dt
        ds_dθ = sqrt(d_xin^2 + d_zin^2)
        arc_length[i+1] = arc_length[i] + ds_dθ * dθ
    end

    # Re-parameterize based on equal arc-length segments by interpolate the original (x,z) data at the equal arc length points
    arc_length_targets = range(; start=0, length=mtheta, step=arc_length[end]/mtheta)
    # arc_length is an uneven grid so we have to specify the period explicitly
    periodic_bc = PeriodicBC(; endpoint=:exclusive, period=arc_length[end])
    xout = cubic_interp(arc_length[1:(end-1)], xin, arc_length_targets; bc=periodic_bc)
    zout = cubic_interp(arc_length[1:(end-1)], zin, arc_length_targets; bc=periodic_bc)
    return xout, zout
end

"""
    periodic_wrap(x, n) -> Int

Inline function for fast periodic wrapping for indices near the valid range [1, n].
Equivalent to `mod1(x, n)` but avoids division for small offsets.
Only valid when `x` is within one period of the valid range (i.e., `1-n < x < 2n`).
"""
@inline periodic_wrap(x::Int, n::Int) = x < 1 ? x + n : (x > n ? x - n : x)

"""
    cross3!(c, a, b, idx)

Inline function for fast cross product of two 3D vectors at a given index.

# Arguments

  - `c::AbstractMatrix`: Output matrix
  - `a::AbstractMatrix`: First input vector
  - `b::AbstractMatrix`: Second input vector
  - `idx`: Index of the output vector
"""
@inline function cross3!(
    c::AbstractMatrix,
    a::AbstractMatrix,
    b::AbstractMatrix,
    idx::Int
)
    @inbounds begin
        a1 = a[idx, 1];
        a2 = a[idx, 2];
        a3 = a[idx, 3]
        b1 = b[idx, 1];
        b2 = b[idx, 2];
        b3 = b[idx, 3]

        c[idx, 1] = a2*b3 - a3*b2
        c[idx, 2] = a3*b1 - a1*b3
        c[idx, 3] = a1*b2 - a2*b1
    end
end

"""
    write_surface_to_vtk(surface_coords::Matrix{Float64}, mtheta::Int, nzeta::Int, filename::String)

Write a 3D toroidal surface to a VTK unstructured grid file for visualization in ParaView.

The surface is represented as a structured grid of quadrilaterals with mtheta poloidal points
and nzeta toroidal points. The grid is periodic in both directions.

# Arguments

  - `surface_coords::Matrix{Float64}`: Array of shape (mtheta * nzeta, 3) containing (X, Y, Z) coordinates
  - `mtheta::Int`: Number of poloidal grid points
  - `nzeta::Int`: Number of toroidal grid points
  - `filename::String`: Base filename for output (without extension)

# Returns

  - `String`: Full path to the created VTK file
"""
function write_surface_to_vtk(surface_coords::Matrix{Float64}, mtheta::Int, nzeta::Int, filename::String)
    num_points = size(surface_coords, 1)
    @assert num_points == mtheta * nzeta "Surface coordinates size ($num_points) must equal mtheta * nzeta ($(mtheta * nzeta))"

    # WriteVTK expects points in format (dim, num_points), so transpose
    points = transpose(surface_coords)  # (3, num_points)

    # Create connectivity for quadrilateral cells
    # Each cell connects 4 points: (i, j), (i+1, j), (i+1, j+1), (i, j+1)
    # where i is poloidal index (0:mtheta-1) and j is toroidal index (0:nzeta-1)
    # Handle periodicity: wrap around at boundaries
    cells = Vector{MeshCell}()

    for j in 0:(nzeta-1)
        j_next = mod(j + 1, nzeta)
        for i in 0:(mtheta-1)
            i_next = mod(i + 1, mtheta)

            # Indices for the 4 corners of the quadrilateral (1-based indexing)
            idx1 = i + j * mtheta + 1  # (i, j)
            idx2 = i_next + j * mtheta + 1  # (i+1, j)
            idx3 = i_next + j_next * mtheta + 1  # (i+1, j+1)
            idx4 = i + j_next * mtheta + 1  # (i, j+1)

            # VTK_QUAD cell type
            push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, [idx1, idx2, idx3, idx4]))
        end
    end

    # Create VTK grid
    vtk = vtk_grid(filename, points, cells)

    # Write the file
    vtk_save(vtk)

    return "$filename.vtu"
end
