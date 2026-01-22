"""
    BIEST

Module for interfacing with BIEST (Boundary Integral Equation Solver for Toroidal surfaces).
Provides access to boundary integral operator functionality for 3D vacuum calculations.
"""
module BIEST

# Determine the library file based on OS
const lib_base_path = joinpath(@__DIR__, "BIEST", "lib")
const libbiest = if Sys.isapple()
    joinpath(lib_base_path, "libbiest.dylib")
else
    joinpath(lib_base_path, "libbiest.so")
end

"""
    compute_green_matrices(R::Vector{Float64}, Z::Vector{Float64}, Nt::Integer) -> (G::Matrix{Float64}, K::Matrix{Float64})

Compute Green's function matrices for Laplace single-layer and double-layer kernels on an axisymmetric toroidal surface.

Takes a 2D poloidal contour defined by `R` (major radius) and `Z` (height) coordinates, and extends it
toroidally using `Nt` toroidal grid points to create a 3D axisymmetric surface with `Np*Nt` total points
(where `Np = length(R)`).

Computes two matrices `G` and `K` of size `(Np*Nt) × (Np*Nt)` where:

  - `G[i,j]`: Single-layer kernel value between 3D points `x_i` and `x_j` on the surface
  - `K[i,j]`: Double-layer kernel value between 3D points `x_i` and `x_j` on the surface

Uses the MatrixExtractor::test() method from extract_matrix.hpp to build the explicit matrix
representation of the boundary integral operators.

# Arguments

  - `G::Matrix{Float64}`: Single-layer kernel matrix, size `(Np*Nt, Np*Nt)`
  - `K::Matrix{Float64}`: Double-layer kernel matrix, size `(Np*Nt, Np*Nt)`
  - `R::Vector{Float64}`: Poloidal major radius coordinates (length Np)
  - `Z::Vector{Float64}`: Poloidal height coordinates (length Np)
  - `Nt::Integer`: Number of toroidal grid points

# Note

Requires the BIEST library to be compiled. Build it with:

```bash
cd src/BIEST && make
```

For large grids (e.g., `Nt*Np > 200`), this computation can be expensive as it requires
evaluating the boundary integral operator `Nt*Np` times to build each matrix column.
"""
function compute_green_matrices!(G::Matrix{Float64}, K::Matrix{Float64}, R::Vector{Float64}, Z::Vector{Float64}, ν::Vector{Float64}, Nt::Integer)
    isfile(libbiest) || error("BIEST library not found at $libbiest. Build it with: cd src/BIEST && make")

    # Call C++ wrapper which uses MatrixExtractor<double>::test()
    ccall((:biest_compute_green_matrices, libbiest), Cvoid,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
        R, Z, ν, Cint(length(R)), Cint(Nt), G, K)
end

function compute_green_matrices!(G::Matrix{Float64}, K::Matrix{Float64}, surf)
    isfile(libbiest) || error("BIEST library not found at $libbiest. Build it with: cd src/BIEST && make")

    # Call C++ wrapper which builds the surface from supplied coordinates
    ccall((:biest_compute_green_matrices_from3D, libbiest), Cvoid,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
        surf.x, surf.y, surf.z, Cint(surf.ntheta), Cint(surf.nzeta), G, K)
end

export compute_green_matrices!

end # module BIEST
