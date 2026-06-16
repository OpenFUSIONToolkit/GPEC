# Reverse-direction energy: create perturbations that satisfy the Euler-Lagrange equation,
# then compute δW from each.
#
# Any boundary normal-displacement ξ_b (in the (m) Fourier basis) maps to the unique regular
# EL solution ξ_ψ(ψ) = u_store(ψ)·(u_store(edge)⁻¹ ξ_b) — the fundamental solutions stored by
# the DCON integration span ALL regular EL solutions. The plasma energy of that perturbation is
#     δW(ξ_b) = psio²·(ξ_b† W_p ξ_b)            [W_p = vac_data.wp]
# and the free-boundary eigenmodes vac_data.wt[:,k] are the special perturbations that
# diagonalise W_p (δW = psio²·ep[k]).
#
# Run:  PATH=$HOME/julia-1.11.2/bin:$PATH ~/julia-1.11.2/bin/julia --project=. benchmarks/dW_from_perturbation.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium, LinearAlgebra, Printf, Random
const GPEC = GeneralizedPerturbedEquilibrium
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

src = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
tmp = mktempdir()
for f in readdir(src)
    cp(joinpath(src, f), joinpath(tmp, f); force=true)
end
res = GPEC.main([tmp])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
n = intr.numpert_total
mlow = intr.mlow
psio = equil.psio

# Cheap exact reverse δW for any perturbation (= PE.plasma_energy).
dW(xi) = PE.plasma_energy(vac_data, equil, xi)

println("\n==========  δW from EL-satisfying perturbations  (Solovev n=1)  ==========")
println("\n-- eigenmodes (special perturbations diagonalising W_p) --")
@printf "%-14s %-16s %-16s %-10s\n" "perturbation" "δW=psio²ξ†W_pξ" "psio²·ep[k]" "match"
for k in 1:min(4, n)
    xi = vac_data.wt[:, k]
    a = dW(xi); b = psio^2 * real(vac_data.ep[k])
    @printf "%-14s %+.6e   %+.6e   %.2e\n" "eig wt[:,$k]" a b abs(a - b)
end

println("\n-- single-Fourier-mode perturbations ξ_b = e_m --")
@printf "%-14s %-16s\n" "perturbation" "δW"
for j in (1, n ÷ 4, n ÷ 2, 3n ÷ 4, n)
    e = zeros(ComplexF64, n); e[j] = 1
    @printf "%-14s %+.6e\n" "m=$(mlow+j-1)" dW(e)
end

println("\n-- arbitrary (random) perturbations + full spatial reconstruction --")
Random.seed!(7)
@printf "%-14s %-16s %-16s %-16s\n" "perturbation" "δW (quad form)" "v2 dW_boundary" "v1 ∫dW (spatial)"
for t in 1:3
    xi = randn(ComplexF64, n)
    rec = PE.energy_for_perturbation(equil, intr, ffit, odet, vac_data, xi)
    @printf "%-14s %+.6e   %+.6e   %+.6e\n" "random $t" dW(xi) real(rec.v2.dW_boundary) rec.v1.dW_total
end

println("\n-- a user-built perturbation (cos-like mix of 3 harmonics) --")
xi_c = zeros(ComplexF64, n)
j0 = n ÷ 2
xi_c[j0-1] = 0.5; xi_c[j0] = 1.0; xi_c[j0+1] = 0.5
rec = PE.energy_for_perturbation(equil, intr, ffit, odet, vac_data, xi_c)
@printf "δW(quad form)       = %+.6e\n" dW(xi_c)
@printf "v2 dW_boundary      = %+.6e   (matches quad form: exact by construction)\n" real(rec.v2.dW_boundary)
@printf "v1 ∫dW (real-space) = %+.6e\n" rec.v1.dW_total
@printf "reconstructed EL solution ξ_ψ(ψ): %d radial points, |ξ_ψ(edge)| = %.3e\n" length(rec.v1.psi) abs(maximum(abs.(xi_c)))
println("==========================================================================")
println("Any ξ_b is admissible: it generates a regular EL solution; δW = psio²·ξ_b†W_pξ_b.")
