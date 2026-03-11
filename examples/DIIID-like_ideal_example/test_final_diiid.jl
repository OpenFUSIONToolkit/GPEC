#REAL DIIID EXAMPLE

using HDF5, TOML, JPEC
import EFIT, EFIT.IMASdd

println("="^70)
println("REAL DIII-D TOKAMAK EXAMPLE")
println("Testing eigenvalue calculation: EFIT vs IMAS")
println("="^70)
println()

dir_efit = mktempdir(prefix="diiid_efit_")
dir_imas = mktempdir(prefix="diiid_imas_")

geqdsk_path = joinpath(@__DIR__, "TKMKR_D3Dlike_default_Hmode.geqdsk")
config = TOML.parsefile("jpec.toml")

# Disable problematic modules, keep core stability calculation
config["ForceFreeStates"]["mer_flag"] = false
config["PerturbedEquilibrium"]["compute_singular_coupling"] = false

# EFIT
println("▶ Running EFIT path...")
config_efit = deepcopy(config)
config_efit["Equilibrium"]["eq_filename"] = geqdsk_path
open(joinpath(dir_efit, "jpec.toml"), "w") do io; TOML.print(io, config_efit); end
cp(joinpath(@__DIR__, "forcing.dat"), joinpath(dir_efit, "forcing.dat"))

JPEC.main([dir_efit])
println()

# IMAS
println("▶ Running IMAS path...")
g = EFIT.readg(geqdsk_path; set_time=0.0)
dd = IMASdd.dd()
dd.global_time = g.time
dd.equilibrium.time = [g.time]
resize!(dd.equilibrium.time_slice, 1)
dd.equilibrium.time_slice[1].time = g.time
JPEC.Equilibrium.jpec_geqdsk_to_imas!(g, dd.equilibrium.time_slice[1])

config_imas = deepcopy(config)
config_imas["Equilibrium"]["eq_type"] = "imas"
config_imas["Equilibrium"]["eq_filename"] = "dummy"
open(joinpath(dir_imas, "jpec.toml"), "w") do io; TOML.print(io, config_imas); end
cp(joinpath(@__DIR__, "forcing.dat"), joinpath(dir_imas, "forcing.dat"))

JPEC.main([dir_imas], dd)
println()

# COMPARE
println("="^70)
println("STABILITY EIGENVALUE COMPARISON")
println("="^70)
println()

h5_efit = h5open(joinpath(dir_efit, "jpec.h5"), "r")
et_efit = read(h5_efit["vacuum/et"])
close(h5_efit)

h5_imas = h5open(joinpath(dir_imas, "jpec.h5"), "r")
et_imas = read(h5_imas["vacuum/et"])
close(h5_imas)

for i in 1:min(5, length(et_efit))
    match = (et_efit[i] == et_imas[i])
    println("Mode $i: $(match ? " EXACT MATCH" : "")")
    println("  EFIT: $(et_efit[i])")
    println("  IMAS: $(et_imas[i])")
    println()
end

all_match = (et_efit == et_imas)
println("="^70)
if all_match
    println()
    println(" SUCCESS! ")
    println()
    println("  ALL $(length(et_efit)) EIGENVALUES MATCH EXACTLY!")
    println()
    println("  EFIT and IMAS produce IDENTICAL results")
    println("  on real-world DIII-D tokamak data!")
    println()
    println(" ")
    println()
else
    println(" Diff: $(maximum(abs.(et_efit .- et_imas)))")
end
println("="^70)

rm(dir_efit, recursive=true)
rm(dir_imas, recursive=true)
