# Compute δW for a user-specified perturbation SHAPE Ξ_ψ(ψ, m) (e.g. an m=2 kink, or an
# externally-computed eigenfunction such as from ELITE, converted to GPEC (ψ, m)).
#
# Run:  PATH=$HOME/julia-1.11.2/bin:$PATH ~/julia-1.11.2/bin/julia --project=. benchmarks/dW_of_shape_demo.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium, LinearAlgebra, Printf
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

src = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
tmp = mktempdir()
for f in readdir(src); cp(joinpath(src, f), joinpath(tmp, f); force=true); end
res = GPEC.main([tmp])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
n = intr.numpert_total; mlow = intr.mlow; mhigh = intr.mhigh
nstep = odet.step
psi = collect(odet.psi_store[1:nstep])
psin = (psi .- psi[1]) ./ (psi[end] - psi[1])      # 0..1 radial coordinate
metric = FFS.make_metric(equil; mband=intr.mband)
de = PE.build_tex_de_matrices(equil, intr, metric)

@printf "\nSolovev n=1: mlow=%d, mhigh=%d, mpert=%d, %d radial points\n" mlow mhigh n nstep

# ---- consistency: feed the EIGENMODE shape Ξ_ψ(ψ,m) → should recover the v2 eigenvalue ----
alpha = odet.u_store[:, :, 1, nstep] \ vac_data.wt[:, 1]
xi_eig = Matrix{ComplexF64}(undef, nstep, n)
for i in 1:nstep
    xi_eig[i, :] = odet.u_store[:, :, 1, i] * alpha
end
s_eig = PE.dW_of_shape(equil, intr, ffit, psi, xi_eig; de=de)
v2 = PE.integrate_energy_v2(equil, intr, ffit, odet, vac_data.wt[:, 1])
@printf "\nconsistency (eigenmode shape): dW_of_shape=%+.5e   v2 dW_volume=%+.5e   v2 boundary=%+.5e\n" real(s_eig.dW) real(v2.dW_volume) real(v2.dW_boundary)

# ---- user-built m=2 kink shapes (only the m=2 harmonic populated) ----
function shape_m(mtarget, f)        # build [nstep, n] with harmonic mtarget = f(ψ)
    X = zeros(ComplexF64, nstep, n)
    j = mtarget - mlow + 1
    @assert 1 <= j <= n "m=$mtarget not in [$mlow,$mhigh]"
    X[:, j] .= ComplexF64.(f)
    return X
end
shapes = Dict(
    "m=2 rigid (f=1)"          => shape_m(2, ones(nstep)),
    "m=2 internal (sin πψ)"    => shape_m(2, sin.(π .* psin)),
    "m=2 edge-peaked (ψ)"      => shape_m(2, psin),
    "m=3 internal (sin πψ)"    => shape_m(3, sin.(π .* psin)),
)
println("\n-- δW for user-specified mode shapes (DCON-normalised, same units as v2 dW_volume) --")
@printf "%-26s %-16s\n" "shape" "δW"
for (lab, X) in sort(collect(shapes); by=first)
    s = PE.dW_of_shape(equil, intr, ffit, psi, X; de=de)
    @printf "%-26s %+.5e\n" lab real(s.dW)
end
@printf "\n(least-stable eigenvalue v2 dW_volume = %+.5e — variational lower bound; any shape ≥ this)\n" real(v2.dW_volume)
println("\nELITE/外部 모양: ξ_ψ(ψ,m)을 GPEC (ψ_norm, DCON θ, m) 기저로 변환해 [npsi,mpert] 행렬로 넣으면 동일하게 δW가 나옵니다.")
