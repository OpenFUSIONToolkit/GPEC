# Overlay the IDEAL gal matched ξ(ψ) against the EL total-energy eigenmode ξ(ψ), for the same edge
# eigenvector w. The ideal gal solution should reproduce the EL (DCON) ideal solution.
#
# EL :  ξ_EL(ψ)  = U_EL(ψ) · (U_EL_edge \ w)          (fundamental matrix, integration/xi_psi)
# gal:  ξ_gal(ψ) = U_gal(ψ) · w                       (identity-at-edge ⇒ coefficient is w itself)
# w = eigenvector of the total energy operator W = W_plasma + W_vacuum (FreeBoundaryStability/XiNorm).
#
# Needs ONE gpec.h5 from a run with populate_dense_xi=true (EL dense u_store), gal_match_flag=true,
# gal_ideal_flag=true (gal ideal matched), vac_flag=true (energy operator).
# Usage: julia --project=. benchmarks/compare_gal_vs_el.jl [gpec.h5] [out.png] [mode]
#   mode = "highest" (default, most stable), "lowest" (most unstable), or an integer eigenmode index.

using HDF5, LinearAlgebra, Plots, Printf

h5path = length(ARGS) >= 1 ? ARGS[1] : "/tmp/gal_ideal_test/gpec.h5"
outpng = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "compare_gal_vs_el.png")
ksel = length(ARGS) >= 3 ? ARGS[3] : "highest"

to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

et, wt, u1, psiE, gxi, psiG, issing, mlow, sing_psi = h5open(h5path) do f
    (to_c(read(f["FreeBoundaryStability/XiNorm/eigenmode_energies"])),
        to_c(read(f["FreeBoundaryStability/XiNorm/W_freeboundary_eigenmodes"])),
        to_c(read(f["integration/xi_psi"])), read(f["integration/psi"]),
        to_c(read(f["galerkin/match/xi"])), read(f["galerkin/solution/psi"]),
        Bool.(read(f["galerkin/solution/issing"])), read(f["info/mlow"]),
        read(f["galerkin/sing_psi"]))
end

mpert = size(u1, 1)
k = ksel == "highest" ? argmax(real.(et)) :
    ksel == "lowest" ? argmin(real.(et)) :
    parse(Int, ksel)
w = wt[:, k]
@printf("comparing eigenmode k=%d   δW = %.4e %+.4ei\n", k, real(et[k]), imag(et[k]))

# EL profile
cEL = u1[:, :, end] \ w
xiE = reduce(hcat, (u1[:, :, ip] * cEL for ip in 1:size(u1, 3)))   # (mpert, nE)

# gal ideal profile (identity-at-edge ⇒ coefficient = w); drop on-surface points
keep = .!issing
psiGk = psiG[keep]
xiG = reduce(hcat, (gxi[:, ip, :] * w for ip in findall(keep)))    # (mpert, nGk)

ms = mlow .+ (0:mpert-1)
peak = [maximum(abs, @view xiE[i, :]) for i in 1:mpert]
order = sortperm(peak; rev=true)
ndom = min(5, mpert)

# quantitative bulk agreement on the dominant harmonic (linear-interp EL onto gal grid, off rationals/edge)
lininterp(xq, x, y) = (j = clamp(searchsortedlast(x, xq), 1, length(x) - 1);
t = (xq - x[j]) / (x[j+1] - x[j]); y[j] * (1 - t) + y[j+1] * t)
idom = order[1]
inbulk(p) = 0.1 <= p <= 0.92 && all(abs(p - ps) > 8e-3 for ps in sing_psi)
sel = [ip for ip in eachindex(psiGk) if inbulk(psiGk[ip])]
num = sum(abs2(xiG[idom, ip] - lininterp(psiGk[ip], psiE, xiE[idom, :])) for ip in sel)
den = sum(abs2(xiG[idom, ip]) for ip in sel)
@printf("dominant m=%d:  bulk relative L2 ‖gal−EL‖/‖gal‖ = %.3e  (n=%d, off-layer/off-edge)\n",
    ms[idom], sqrt(num / den), length(sel))

plt = plot(; size=(1000, 650), xlabel="ψ_N", ylabel="|ξ^ψ_m(ψ)|",
    title=@sprintf("Ideal gal vs EL total-energy eigenmode  (δW=%.2e)  — solid: EL, dashed: gal", real(et[k])),
    legend=:topleft, left_margin=12Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
palette = theme_palette(:auto)
for (c, i) in enumerate(order[1:ndom])
    col = palette[mod1(c, length(palette))]
    plot!(plt, psiE, abs.(@view xiE[i, :]); color=col, lw=2.2, label="EL m=$(ms[i])")
    plot!(plt, psiGk, abs.(@view xiG[i, :]); color=col, lw=1.6, ls=:dash, label="gal m=$(ms[i])")
end
for (j, ps) in enumerate(sing_psi)
    vline!(plt, [ps]; color=:black, ls=:dot, lw=1, label=(j == 1 ? "rational q" : ""))
end

display(plt)
savefig(plt, outpng)
println("dominant harmonics (m): ", [ms[i] for i in order[1:ndom]])
println("saved: ", abspath(outpng))
