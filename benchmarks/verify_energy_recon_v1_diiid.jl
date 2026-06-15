# Decisive v1 validation on a NON-marginal case (DIIID-like, ep[1]~O(1)), where ∫δW is
# not a tiny cancellation. If v1 is correct, ∫dW(v1)/dW_boundary(v2) is a constant
# across eigenmodes.
#
# Run:  ~/julia-1.11.2/bin/julia --project=. benchmarks/verify_energy_recon_v1_diiid.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using LinearAlgebra, Printf, Statistics

src_dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
tmp_dir = mktempdir()
for f in readdir(src_dir)
    p = joinpath(src_dir, f)
    isfile(p) && cp(p, joinpath(tmp_dir, f); force=true)
end

res = GPEC.main([tmp_dir])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
metric = FFS.make_metric(equil; mband=intr.mband)
de = FFS.build_tex_de_matrices(equil, intr, metric)

nmodes = min(6, size(vac_data.wt, 2))
println("\n==============  v1 vs v2 on DIIID-like (non-marginal)  ==============")
@printf "%-5s %-15s %-15s %-15s %-12s\n" "mode" "ep[k]" "v1 dW_total" "v2 dW_bndry" "v1/v2"
println("--------------------------------------------------------------------")
ratios = Float64[]
for k in 1:nmodes
    v = vac_data.wt[:, k]
    r1 = PE.reconstruct_energy_realspace(equil, intr, ffit, odet, metric, v)
    v2 = FFS.integrate_energy_v2(equil, intr, ffit, odet, v; de=de)
    rat = real(r1.dW_total / v2.dW_boundary)
    push!(ratios, rat)
    @printf "%-5d %+.5e %+.5e %+.5e %-12.5f\n" k real(vac_data.ep[k]) real(r1.dW_total) real(v2.dW_boundary) rat
end
println("--------------------------------------------------------------------")
@printf "ratio mean = %.5f   std/|mean| = %.3e\n" mean(ratios) (std(ratios)/abs(mean(ratios)))
println("CONSTANT across modes ⇒ v1 validated (factor = unit conversion).")
