using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using LinearAlgebra
using Printf

# Access internal modules
using GeneralizedPerturbedEquilibrium: CoilSet, parse_coil_file, compute_biot_savart_boundary!

function main()
    # Paths
    coil_file = joinpath(@__DIR__, "d3d_c.dat")
    output_file = joinpath(@__DIR__, "b_field_output.csv")

    if !isfile(coil_file)
        error("Could not find coil data file: $coil_file")
    end

    # Parse d3d_c.dat into CoilSet objects
    # This uses the parser logic in src/ForcingTerms/CoilGeometry.jl
    println("Parsing coil geometry from $coil_file...")
    coil_sets = [parse_coil_file(coil_file)]

    # Define observation points [m, rad, m]
    # Example: a 2D slice at phi = 0
    Rs = collect(range(1.0, 4.0, length=50))
    Phis = zeros(length(Rs))
    Zs = collect(range(-1.5, 1.5, length=50))

    # Flatten into 1D vectors for the BiotSavart kernel
    n_r = length(Rs)
    n_z = length(Zs)
    nobs = n_r * n_z

    obs_R   = zeros(nobs)
    obs_phi = zeros(nobs)
    obs_Z   = zeros(nobs)

    idx = 1
    for i in 1:n_r
        for j in 1:n_z
            obs_R[idx]   = Rs[i]
            obs_phi[idx] = 0.0
            obs_Z[idx]   = Zs[j]
            idx += 1
        end
    end

    # Output arrays
    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)

    println("Computing Biot-Savart field at $nobs points...")
    # Core kernel from src/ForcingTerms/BiotSavart.jl
    # This function uses internal multithreading via Threads.@threads
    compute_biot_savart_boundary!(
        B_R, B_phi, B_Z,
        obs_R, obs_phi, obs_Z,
        coil_sets
    )

    # Write results to CSV
    println("Writing results to $output_file...")
    open(output_file, "w") do io
        println(io, "R,phi,Z,BR,Bphi,BZ")
        for i in 1:nobs
            @printf(io, "%.6f,%.6f,%.6f,%.8e,%.8e,%.8e\n", 
                    obs_R[i], obs_phi[i], obs_Z[i], B_R[i], B_phi[i], B_Z[i])
        end
    end

    println("Biot-Savart calculation complete.")
end

main()