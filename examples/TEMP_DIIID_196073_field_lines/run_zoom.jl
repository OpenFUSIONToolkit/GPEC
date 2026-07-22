# Zoom driver: re-trace the three vacuum slices with field lines densely seeded in
# ψ_N ∈ [0.58, 0.77] to resolve the m=2 island in detail. Writes to separate
# t<slice>_vacuum_zoom/ directories so the full-range cases are untouched.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GeneralizedPerturbedEquilibrium, HDF5
const HERE = @__DIR__

for slice in ["2840", "2845", "2850"]
    src = joinpath(HERE, "t$(slice)_vacuum", "gpec.toml")
    dst_dir = joinpath(HERE, "t$(slice)_vacuum_zoom")
    mkpath(dst_dir)
    toml = read(src, String)
    # Redirect the equilibrium path (one level deeper is the same '../' since both are
    # single subdirectories directly under HERE) and densify the launch band.
    toml = replace(toml,
        "n_lines = 80" => "n_lines = 200",
        "psi_start = 0.05" => "psi_start = 0.58",
        "psi_end = 0.98" => "psi_end = 0.77",
        "n_transits = 500" => "n_transits = 600",
        "tol = 1e-9" => "tol = 1e-7",
        "compute_connection_length = true" => "compute_connection_length = false",
        "compute_footprints = true" => "compute_footprints = false")
    write(joinpath(dst_dir, "gpec.toml"), toml)
    @info "==== ZOOM RUN: $(slice) ms vacuum, band [0.58,0.77] ===="
    GeneralizedPerturbedEquilibrium.main([dst_dir])
end
println("ZOOM RUNS DONE")
