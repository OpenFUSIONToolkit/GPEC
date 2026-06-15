# Verify v2 (matrix-form) energy-principle reconstruction against the DCON
# boundary-term plasma energy.
#
# See src/ForceFreeStates/EnergyReconstruction.jl and
# docs/tex/recon/main.tex §"Reconstruction v2".
#
# Success criteria for the least-stable eigenmode:
#   1. ‖AΞ_s + BΞ_ψ' + CΞ_ψ‖ ≈ 0      (Ξ_s is the energy-minimizing surface variable)
#   2. dW_volume ≈ dW_boundary          (∫ of the primitive A–H form = DCON boundary term)
#   3. dW_boundary ≈ psio²·ep[1]        (storage / normalization consistency)
#
# Run:  ~/julia-1.11.2/bin/julia --project=. benchmarks/verify_energy_recon_v2.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
using LinearAlgebra
using Printf

# Copy example inputs to a temp dir so gpec.h5 does not pollute examples/.
src_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
tmp_dir = mktempdir()
for f in readdir(src_dir)
    cp(joinpath(src_dir, f), joinpath(tmp_dir, f); force=true)
end

res = GPEC.main([tmp_dir])
equil = res.equil
intr = res.intr
ffit = res.ffit
odet = res.odet
vac_data = res.vac_data

psio = equil.psio
# Build the tex D,E matrices once and reuse across modes.
metric = FFS.make_metric(equil; mband=intr.mband)
de = PE.build_tex_de_matrices(equil, intr, metric)

nmodes = min(4, size(vac_data.wt, 2))
println("\n=====================  v2 energy reconstruction  =====================")
@printf "psio = %.6e\n" psio
println("----------------------------------------------------------------------")
@printf "%-5s %-16s %-16s %-12s %-12s\n" "mode" "dW_volume" "dW_boundary" "vol/bndry" "bndry/ep·χ²"
println("----------------------------------------------------------------------")
for k in 1:nmodes
    mode_vec = vac_data.wt[:, k]
    out = PE.integrate_energy_v2(equil, intr, ffit, odet, mode_vec; de=de)
    r_vb = real(out.dW_volume / out.dW_boundary)
    r_be = real(out.dW_boundary / (psio^2 * vac_data.ep[k]))
    @printf "%-5d %+.6e %+.6e %-12.6f %-12.6f\n" k real(out.dW_volume) real(out.dW_boundary) r_vb r_be
end
println("======================================================================")
println("vol/bndry → 1 confirms the tex matrix form (residual = trapz quadrature).")
println("bndry/ep·χ² = 1 confirms storage/normalization (Ξ_ψ†u₂ = psio²·ep).")
