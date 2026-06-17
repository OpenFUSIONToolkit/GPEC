# Compare the area-normalized b^ψ (perturbed_equilibrium/response/psi_area = b^ψ/⟨J·|∇ψ|⟩_θ) for one
# poloidal harmonic between two GPEC runs that are identical except for which ξ feeds PerturbedEquilibrium:
#   (1) IDEAL galerkin matched ξ   (gal_match_flag=true, gal_ideal_flag=true)
#   (2) SHOOTING ξ                  (gal_match_flag=false)
#
# PE writes no ψ grid, so it's reconstructed: gal-ideal → galerkin/solution/psi minus issing points;
# shooting → integration/psi.
# Usage: julia --project=. benchmarks/compare_jbgradpsi_m2.jl [gal_h5] [shoot_h5] [out.png] [m]

using HDF5, Plots, Printf

gal_h5 = length(ARGS) >= 1 ? ARGS[1] : "/tmp/gal_ideal_test/gpec.h5"
sh_h5 = length(ARGS) >= 2 ? ARGS[2] : "/tmp/shooting_test/gpec.h5"
outpng = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "cmp_jbgradpsi_m2.png")
mtarget = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2

to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

# gal-ideal run: PE grid = gal solution grid with the on-surface (issing) points dropped
pa_g, psi_g, mlow, sing_psi, sing_m = h5open(gal_h5) do f
    pa = to_c(read(f["perturbed_equilibrium/response/psi_area"]))   # [npsi, mpert]
    iss = Bool.(read(f["galerkin/solution/issing"]))
    (pa, read(f["galerkin/solution/psi"])[.!iss], read(f["info/mlow"]),
        read(f["galerkin/sing_psi"]), read(f["galerkin/sing_m"]))
end
# shooting run: PE grid = integration/psi
pa_s, psi_s = h5open(sh_h5) do f
    (to_c(read(f["perturbed_equilibrium/response/psi_area"])), read(f["integration/psi"]))
end

size(pa_g, 1) == length(psi_g) || error("gal grid mismatch: npsi=$(size(pa_g,1)) vs grid=$(length(psi_g))")
size(pa_s, 1) == length(psi_s) || error("shoot grid mismatch: npsi=$(size(pa_s,1)) vs grid=$(length(psi_s))")

col = mtarget - mlow + 1
@printf("m=%d → column %d (mlow=%d, mpert=%d)\n", mtarget, col, mlow, size(pa_g, 2))
g = @view pa_g[:, col]
s = @view pa_s[:, col]

# resonant surface for this m (q = m/n)
psi_res = mtarget in sing_m ? sing_psi[findfirst(==(mtarget), sing_m)] : NaN
# value at the grid point nearest the resonant surface, each run
if !isnan(psi_res)
    ig = argmin(abs.(psi_g .- psi_res))
    is = argmin(abs.(psi_s .- psi_res))
    @printf("m=%d resonant surface ψ=%.4f\n", mtarget, psi_res)
    @printf("  nearest |b^ψ_area|:  gal-ideal=%.4e (ψ=%.4f)   shooting=%.4e (ψ=%.4f)   ratio=%.3f\n",
        abs(g[ig]), psi_g[ig], abs(s[is]), psi_s[is], abs(g[ig]) / abs(s[is]))
end

plt = plot(; size=(1000, 620), xlabel="ψ_N", ylabel="|b^ψ / ⟨J·|∇ψ|⟩|  (area-normalized)",
    title="m=$mtarget perturbed normal field — gal-ideal vs shooting PE", legend=:topleft,
    left_margin=13Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
plot!(plt, psi_g, abs.(g); lw=2.2, color=1, label="gal-ideal ξ → PE")
plot!(plt, psi_s, abs.(s); lw=1.8, color=2, ls=:dash, label="shooting ξ → PE")
for (j, ps) in enumerate(sing_psi)
    ism2 = sing_m[j] == mtarget
    vline!(plt, [ps]; color=(ism2 ? :red : :black), ls=:dot, lw=(ism2 ? 1.8 : 1),
        label=(ism2 ? "q=$mtarget surface" : (j == 1 ? "rational q" : "")))
end
savefig(plt, outpng)
println("saved: ", abspath(outpng))
