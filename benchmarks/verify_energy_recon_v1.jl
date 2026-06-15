# Verify v1 (real-space) energy reconstruction against the verified v2 boundary energy.
#
# For ANY boundary displacement ξ_b, the plasma δW is well-defined and v2 gives it
# exactly: v2 dW_boundary(ξ_b) = psio²·(ξ_b'·wp·ξ_b). v1's ∫dW dψ must equal that up
# to ONE unit-conversion constant. Testing many arbitrary ξ_b (well-conditioned, unlike
# the near-marginal eigenmodes) isolates a clean-constant (v1 correct) from a
# direction-dependent (structural bug) discrepancy.
#
# Run:  ~/julia-1.11.2/bin/julia --project=. benchmarks/verify_energy_recon_v1.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using LinearAlgebra, Printf, Random, Statistics

src_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
tmp_dir = mktempdir()
for f in readdir(src_dir)
    cp(joinpath(src_dir, f), joinpath(tmp_dir, f); force=true)
end

res = GPEC.main([tmp_dir])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
psio = equil.psio
n = intr.numpert_total
metric = FFS.make_metric(equil; mband=intr.mband)
de = PE.build_tex_de_matrices(equil, intr, metric)

# Build a set of well-conditioned test boundary displacements.
Random.seed!(1234)
test_vecs = Vector{Vector{ComplexF64}}()
push!(test_vecs, vac_data.wt[:, 1])                       # least-stable eigenmode
for j in (1, n ÷ 2, n)                                    # unit vectors
    e = zeros(ComplexF64, n); e[j] = 1; push!(test_vecs, e)
end
for _ in 1:4                                              # random complex directions
    push!(test_vecs, randn(ComplexF64, n))
end

println("\n=========================  v1 vs v2 (arbitrary boundary displacements)  =========================")
@printf "%-22s %-16s %-16s %-16s\n" "test vector" "v1 dW_total" "v2 dW_boundary" "v1/v2"
println("-------------------------------------------------------------------------------------------------")
labels = ["eig wt[:,1]", "unit e_1", "unit e_$(n÷2)", "unit e_$n", "rand 1", "rand 2", "rand 3", "rand 4"]
ratios = Float64[]
for (lab, v) in zip(labels, test_vecs)
    r1 = PE.reconstruct_energy_realspace(equil, intr, ffit, odet, metric, v)
    v2 = PE.integrate_energy_v2(equil, intr, ffit, odet, v; de=de)
    rat = real(r1.dW_total / v2.dW_boundary)
    push!(ratios, rat)
    @printf "%-22s %+.6e %+.6e %-16.6f\n" lab real(r1.dW_total) real(v2.dW_boundary) rat
end
println("-------------------------------------------------------------------------------------------------")
@printf "ratio mean = %.6f   std/|mean| = %.3e\n" (sum(ratios)/length(ratios)) (std(ratios)/abs(sum(ratios)/length(ratios)))
println("CONSTANT ratio  ⇒ v1 correct (factor = unit conversion).  VARYING ⇒ structural bug.")

# --- Pointwise density localization (Fortran-free) for the least-stable mode ---
# v2 verified density: d/dψ(Ξ_ψ†·u₂). v1 density: r1.dW = ½(epf−dst). Compare shapes.
mv = vac_data.wt[:, 1]
r1 = PE.reconstruct_energy_realspace(equil, intr, ffit, odet, metric, mv)
alpha = odet.u_store[:, :, 1, odet.step] \ mv
ns = odet.step
psi = r1.psi
Xu2 = [real(dot(odet.u_store[:, :, 1, i] * alpha, odet.u_store[:, :, 2, i] * alpha)) for i in 1:ns]
dens_v2 = similar(Xu2)
for i in 2:ns-1
    dens_v2[i] = (Xu2[i+1] - Xu2[i-1]) / (psi[i+1] - psi[i-1])
end
dens_v2[1] = (Xu2[2] - Xu2[1]) / (psi[2] - psi[1])
dens_v2[ns] = (Xu2[ns] - Xu2[ns-1]) / (psi[ns] - psi[ns-1])

# normalize both density profiles to unit integral (sign-preserving) and sample
iv1 = r1.dW_total
iv2 = sum(0.5 .* (dens_v2[1:end-1] .+ dens_v2[2:end]) .* diff(psi))
println("\n--- pointwise density: v1 ½(epf−dst)  vs  v2 d/dψ(Ξ†u₂)  (least-stable mode) ---")
@printf "∫v1 = %+.4e   ∫v2 = %+.4e   (∫v1/∫v2 = %.4e)\n" iv1 iv2 (iv1/iv2)
@printf "%-10s %-14s %-14s %-12s\n" "ψ" "v1 dens" "v2 dens" "v1/v2"
for i in round.(Int, range(2, ns-1; length=12))
    @printf "%-10.4f %+.5e %+.5e %-12.4e\n" psi[i] real(r1.dW[i]) dens_v2[i] real(r1.dW[i])/dens_v2[i]
end
