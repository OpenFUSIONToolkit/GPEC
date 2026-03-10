using HDF5, JPEC, TOML, Random
import EFIT, EFIT.IMASdd

Random.seed!(1234)
test_data_dir = joinpath(@__DIR__, "test/test_data/regression_equilibrium_example")
geqdsk_path = joinpath(test_data_dir, "EQDSK_COCOS_02")

dir1 = mktempdir(prefix="qcheck_efit_")
dir2 = mktempdir(prefix="qcheck_imas_")

config = Dict(
    "Equilibrium" => Dict("eq_type" => "efit", "eq_filename" => geqdsk_path, "jac_type" => "boozer", "psilow" => 0.01, "psihigh" => 0.994, "mpsi" => 32, "mtheta" => 256),
    "ForceFreeStates" => Dict("nn_low" => 1, "nn_high" => 1, "vac_flag" => true, "ode_flag" => true, "mat_flag" => true, "mer_flag" => false, "bal_flag" => false, "write_outputs_to_HDF5" => true, "HDF5_filename" => "jpec.h5"),
    "Wall" => Dict("shape" => "conformal", "a" => 0.2)
)

println("Running EFIT path...")
open(joinpath(dir1, "jpec.toml"), "w") do io; TOML.print(io, config); end
JPEC.main([dir1])

println("Running IMAS path...")
g = EFIT.readg(geqdsk_path; set_time=0.0)
dd = IMASdd.dd()
dd.global_time = g.time
dd.equilibrium.time = [g.time]
resize!(dd.equilibrium.time_slice, 1)
dd.equilibrium.time_slice[1].time = g.time
# Use JPEC's converter to avoid COCOS round-trip errors
JPEC.Equilibrium.jpec_geqdsk_to_imas!(g, dd.equilibrium.time_slice[1]; wall=nothing)

imas_config = deepcopy(config)
imas_config["Equilibrium"]["eq_type"] = "imas"
open(joinpath(dir2, "jpec.toml"), "w") do io; TOML.print(io, imas_config); end
JPEC.main([dir2], dd)

# Compare q profiles from HDF5
q_efit = h5read(joinpath(dir1, "jpec.h5"), "splines/profiles/q")
q_imas = h5read(joinpath(dir2, "jpec.h5"), "splines/profiles/q")

println()
println("="^70)
println("Q PROFILE COMPARISON FROM HDF5")
println("="^70)
println()
println("Sample points:")
for i in [1, 5, 10, 15, length(q_efit)]
    match = isapprox(q_efit[i], q_imas[i], rtol=1e-10)
    println("  q[$i]: EFIT=$(q_efit[i]), IMAS=$(q_imas[i]), match=$match")
end
println()
println("Overall match: $(isapprox(q_efit, q_imas, rtol=1e-10))")

if isapprox(q_efit, q_imas, rtol=1e-10)
    println("✅ Q PROFILES MATCH - FIX WORKED!")
else
    println("❌ Q PROFILES STILL DIFFER - FIX DID NOT WORK")
    println("Max difference: $(maximum(abs.(q_efit .- q_imas)))")
end

rm(dir1, recursive=true)
rm(dir2, recursive=true)
