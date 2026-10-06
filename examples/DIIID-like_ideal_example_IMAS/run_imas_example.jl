"""
IMAS Integration Benchmark — DIII-D-like H-mode equilibrium

Demonstrates that GPEC produces the same stability eigenvalue when reading the
equilibrium from an IMAS data dictionary (IMASdd.dd) as when reading directly
from the g-file. Both runs use identical ForceFreeStates and Wall settings so
the only difference is how the equilibrium is loaded.

EFIT.jl is NOT a GPEC dependency. Install it in your own environment before running:
    using Pkg; Pkg.add("EFIT")
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

import GeneralizedPerturbedEquilibrium as GPEC
using IMASdd
using TOML
using Printf
using EFIT

const GEQDSK = joinpath(@__DIR__, "../DIIID-like_ideal_example/TkMkr_D3Dlike_Hmode.geqdsk")
@assert isfile(GEQDSK) "G-file not found at $GEQDSK"

# Shared settings: both runs use identical ForceFreeStates and Wall config.
# Only the [Equilibrium] section differs between IMAS and g-file paths.
const SHARED_CONFIG = Dict(
    "ForceFreeStates" => Dict(
        "nn_low" => 1,
        "nn_high" => 1,
        "vac_flag" => true,
        "local_stability_flag" => false,
        "write_outputs_to_HDF5" => false,
        "verbose" => false,
        "delta_mhigh" => 16,
        "qlow" => 1.02
    ),
    "Wall" => Dict("shape" => "nowall")
)

const EQ_COMMON = Dict(
    "jac_type" => "boozer",
    "psilow" => 0.01,
    "psihigh" => 0.9995,
    "mpsi" => 128,
    "mtheta" => 256
)

@info "Loading g-file and converting to IMAS (COCOS 11)..."
g = EFIT.readg(GEQDSK)
dd = IMASdd.dd()
EFIT.geqdsk2imas!([g], dd; dd_cocos=11)

@info "Running GPEC from IMAS dd..."
tmpdir_imas = mktempdir(; prefix="gpec_imas_")
try
    config_imas = merge(SHARED_CONFIG, Dict("Equilibrium" =>
        merge(EQ_COMMON, Dict("eq_type" => "imas", "eq_filename" => "from_dd", "imas_cocos" => 11))))
    open(joinpath(tmpdir_imas, "gpec.toml"), "w") do io
        ;
        TOML.print(io, config_imas);
    end
    result_imas = GPEC.main([tmpdir_imas]; dd=dd)
    global et_imas = real(result_imas.ffs.free_boundary.et[1])
    global mpert_imas = result_imas.ffs.mpert
    GPEC.write_imas(dd, result_imas)
    @assert dd.mhd_linear.time_slice[1].toroidal_mode[1].energy_perturbed ≈ et_imas
finally
    rm(tmpdir_imas; recursive=true)
end
@info @sprintf "IMAS path:   δW(n=1) = %.6f  [mpert = %d]" et_imas mpert_imas

@info "Running GPEC from g-file (reference)..."
tmpdir_gfile = mktempdir(; prefix="gpec_gfile_")
try
    config_gfile = merge(SHARED_CONFIG, Dict("Equilibrium" =>
        merge(EQ_COMMON, Dict("eq_type" => "efit", "eq_filename" => GEQDSK))))
    open(joinpath(tmpdir_gfile, "gpec.toml"), "w") do io
        ;
        TOML.print(io, config_gfile);
    end
    result_gfile = GPEC.main([tmpdir_gfile])
    global et_gfile = real(result_gfile.ffs.free_boundary.et[1])
    global mpert_gfile = result_gfile.ffs.mpert
finally
    rm(tmpdir_gfile; recursive=true)
end
@info @sprintf "G-file path: δW(n=1) = %.6f  [mpert = %d]" et_gfile mpert_gfile

rel_diff = abs(et_imas - et_gfile) / (abs(et_gfile) + 1e-12)

println("\n" * "="^60)
println("  IMAS vs g-file comparison (n=1, nowall, boozer)")
println("="^60)
@printf "  G-file δW(n=1) : %+.6f  [mpert = %d]\n" et_gfile mpert_gfile
@printf "  IMAS   δW(n=1) : %+.6f  [mpert = %d]\n" et_imas mpert_imas
@printf "  Relative diff  : %.2f%%\n" rel_diff * 100
println("="^60)

if rel_diff < 0.01
    @info "Agreement within 1% — IMAS and g-file paths are consistent."
elseif rel_diff < 0.05
    @warn "Agreement within 5% — likely due to grid resampling in geqdsk2imas!."
else
    @warn "Relative difference $(round(rel_diff*100; digits=1))% exceeds 5%. Check COCOS convention and grid resolution."
end
