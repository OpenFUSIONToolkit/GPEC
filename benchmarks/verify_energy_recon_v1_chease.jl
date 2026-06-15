# Validate v1 against the Fortran recon reference on the SAME equilibrium (CHEASE/EQDSK
# interchange case with gpec_recon_integration_sol1.out reference). Compares amplitude-
# independent term fractions (c2_psi/theta/zeta share of epf; k1/k2/k3 share of dst;
# dst/epf) at matching ψ — a structural-bug localizer.
#
# Run:  ~/julia-1.11.2/bin/julia --project=. benchmarks/verify_energy_recon_v1_chease.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using LinearAlgebra, Printf, DelimitedFiles

# Directory holding the Fortran recon reference (EQDSK_COCOS_02.OUT +
# gpec_recon_integration_sol1.out). Override with GPEC_RECON_REF_DIR.
ref_dir = get(ENV, "GPEC_RECON_REF_DIR",
    "/home/satelite2517/data/chease/test/instabilities/cases/PT/01_interchange")
tmp = mktempdir()
cp(joinpath(ref_dir, "EQDSK_COCOS_02.OUT"), joinpath(tmp, "EQDSK_COCOS_02.OUT"); force=true)

open(joinpath(tmp, "gpec.toml"), "w") do io
    write(io, """
[Equilibrium]
eq_filename = "EQDSK_COCOS_02.OUT"
eq_type = "efit"
jac_type = "hamada"
power_bp = 0
power_b = 0
power_r = 0
grid_type = "ldp"
psilow = 1e-4
psihigh = 0.993
mpsi = 128
mtheta = 256
newq0 = 0
force_termination = false

[Wall]
shape = "nowall"

[ForceFreeStates]
bal_flag = false
mat_flag = true
ode_flag = true
vac_flag = true
mer_flag = true
qlow = 1.02
qhigh = 1e3
sing_start = 0
nn_low = 1
nn_high = 1
delta_mlow = 8
delta_mhigh = 8
delta_mband = 0
mthvac = 512
kinetic_factor = 0.0
use_parallel = false
populate_dense_xi = true
set_psilim_via_dmlim = true
dmlim = 0.2
""")
end

res = GPEC.main([tmp])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
@printf "\nJulia: ep[1]=%+.4e  et[1]=%+.4e  mpert=%d  nsteps=%d\n" real(vac_data.ep[1]) real(vac_data.et[1]) intr.mpert odet.step

metric = FFS.make_metric(equil; mband=intr.mband)
r1 = PE.reconstruct_energy_realspace(equil, intr, ffit, odet, metric, vac_data.wt[:, 1])

# Parse Fortran reference table (skip comment lines starting with non-numeric).
ref = readdlm(joinpath(ref_dir, "gpec_recon_integration_sol1.out"))
rows = [i for i in 1:size(ref, 1) if (ref[i, 1] isa Number) && (ref[i, 15] isa Number)]
P = Float64.(ref[rows, 1]); C2 = Float64.(ref[rows, 2]); KX = Float64.(ref[rows, 3])
C2p = Float64.(ref[rows, 10]); C2t = Float64.(ref[rows, 11]); C2z = Float64.(ref[rows, 12])
K1 = Float64.(ref[rows, 13]); K2 = Float64.(ref[rows, 14]); K3 = Float64.(ref[rows, 15])

nearest(p) = argmin(abs.(P .- p))
nearestj(p) = argmin(abs.(r1.psi .- p))

println("\n========  v1 (Julia)  vs  Fortran recon  — amplitude-independent fractions  ========")
for ptgt in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)
    i = nearest(ptgt); j = nearestj(ptgt)
    fc = (C2[i] == 0 ? NaN : 1 / C2[i]); fk = (KX[i] == 0 ? NaN : 1 / KX[i])
    jc = (r1.epf[j] == 0 ? NaN : 1 / r1.epf[j]); jk = (r1.dst[j] == 0 ? NaN : 1 / r1.dst[j])
    @printf "\nψ≈%.2f   (Fortran ψ=%.4f, Julia ψ=%.4f)\n" ptgt P[i] r1.psi[j]
    @printf "  epf comp (ψ:θ:ζ)  FOR: %+.3f %+.3f %+.3f   JUL: %+.3f %+.3f %+.3f\n" C2p[i]*fc C2t[i]*fc C2z[i]*fc r1.epf_p[j]*jc r1.epf_t[j]*jc r1.epf_z[j]*jc
    @printf "  K comp (k1:k2:k3) FOR: %+.3f %+.3f %+.3f   JUL: %+.3f %+.3f %+.3f\n" K1[i]*fk K2[i]*fk K3[i]*fk r1.dst1[j]*jk r1.dst2[j]*jk r1.dst3[j]*jk
    @printf "  dst/epf           FOR: %+.4e   JUL: %+.4e\n" (KX[i]/C2[i]) (r1.dst[j]/r1.epf[j])
end
println("\nMatching fractions ⇒ v1 structurally correct. Diverging term ⇒ that's the bug.")

# Integral sign + magnitude check vs verified v2 boundary energy (this case is robustly
# unstable, so ∫δW is NOT a near-cancellation — a clean go/no-go).
v2 = PE.integrate_energy_v2(equil, intr, ffit, odet, vac_data.wt[:, 1])
println("\n--- integral check (robustly-unstable case) ---")
@printf "v1 ∫dW dψ        = %+.5e   (sign should be NEGATIVE = unstable)\n" r1.dW_total
@printf "v2 dW_boundary   = %+.5e   (= psio²·ep[1], verified)\n" real(v2.dW_boundary)
@printf "v1 / v2          = %.5e\n" real(r1.dW_total / v2.dW_boundary)
@printf "Fortran dW_dcon(K) reference = -3.99323298e+01 (from gpec_recon_integration_sol1.out)\n"
