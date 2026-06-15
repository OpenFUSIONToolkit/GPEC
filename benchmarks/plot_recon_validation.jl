# Plot + value validation of the energy-principle reconstruction (v1 real-space, v2 matrix)
# against the Fortran gpout_recon reference on the same EFIT equilibrium.
#
# Saves figures to benchmarks/recon_plots/ and prints the δW value comparison.
#
# Run:  PATH=$HOME/julia-1.11.2/bin:$PATH \
#       ~/julia-1.11.2/bin/julia --project=. benchmarks/plot_recon_validation.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium, LinearAlgebra, Printf, DelimitedFiles, HDF5
using Plots
const GPEC = GeneralizedPerturbedEquilibrium
const FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
gr()

outdir = joinpath(@__DIR__, "recon_plots")
mkpath(outdir)

ref_dir = get(ENV, "GPEC_RECON_REF_DIR",
    "/home/satelite2517/data/chease/test/instabilities/cases/PT/01_interchange")

# --- Run Julia GPEC on the same equilibrium ---
tmp = mktempdir()
cp(joinpath(ref_dir, "EQDSK_COCOS_02.OUT"), joinpath(tmp, "EQDSK_COCOS_02.OUT"); force=true)
open(joinpath(tmp, "gpec.toml"), "w") do io
    write(io, """
[Equilibrium]
eq_filename = "EQDSK_COCOS_02.OUT"
eq_type = "efit"
jac_type = "hamada"
grid_type = "ldp"
psilow = 1e-4
psihigh = 0.993
mpsi = 128
mtheta = 256

[Wall]
shape = "nowall"

[ForceFreeStates]
bal_flag = false
mat_flag = true
ode_flag = true
vac_flag = true
mer_flag = true
recon_flag = true
qlow = 1.02
qhigh = 1e3
nn_low = 1
nn_high = 1
delta_mlow = 8
delta_mhigh = 8
mthvac = 512
use_parallel = false
populate_dense_xi = true
set_psilim_via_dmlim = true
dmlim = 0.2
""")
end
res = GPEC.main([tmp])
equil = res.equil; intr = res.intr; ffit = res.ffit; odet = res.odet; vac_data = res.vac_data
psio = equil.psio

# Persist the standalone recon.h5 the pipeline produced.
if isfile(joinpath(tmp, "recon.h5"))
    cp(joinpath(tmp, "recon.h5"), joinpath(outdir, "recon.h5"); force=true)
    println("recon.h5 written to ", joinpath(outdir, "recon.h5"))
end

mode_vec = vac_data.wt[:, 1]
metric = FFS.make_metric(equil; mband=intr.mband)
r1 = PE.reconstruct_energy_realspace(equil, intr, ffit, odet, metric, mode_vec)
v2 = PE.integrate_energy_v2(equil, intr, ffit, odet, mode_vec)

# --- Parse Fortran reference table ---
ref = readdlm(joinpath(ref_dir, "gpec_recon_integration_sol1.out"))
rows = [i for i in 1:size(ref, 1) if (ref[i, 1] isa Number) && (ref[i, 15] isa Number)]
fP = Float64.(ref[rows, 1]); fC2 = Float64.(ref[rows, 2]); fKX = Float64.(ref[rows, 3])
fdw = 0.5 .* (fC2 .- fKX)   # Fortran dW density = 0.5(c2_mu0 - k_xin2)

# --- Single amplitude scale so Julia epf integral matches Fortran epf integral ---
jint(y) = sum(0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(r1.psi))
fint(y) = sum(0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(fP))
scale = fint(fC2) / jint(r1.epf)
@printf "\namplitude scale (Fortran/Julia, from epf integral) = %.4e\n" scale

# =====================  δW value table  =====================
println("\n================  δW VALUE CHECK  ================")
@printf "Julia ep[1] (DCON plasma eigenvalue)   = %+.6e\n" real(vac_data.ep[1])
@printf "Julia et[1] (total)                    = %+.6e\n" real(vac_data.et[1])
@printf "v2 dW_boundary  = Ξ†u₂|edge            = %+.6e\n" real(v2.dW_boundary)
@printf "v2 psio²·ep[1]                         = %+.6e   (ratio %.10f)\n" (psio^2*real(vac_data.ep[1])) real(v2.dW_boundary/(psio^2*vac_data.ep[1]))
@printf "v2 dW_volume (matrix ∫)               = %+.6e   (vol/bndry %.6f)\n" real(v2.dW_volume) real(v2.dW_volume/v2.dW_boundary)
@printf "v1 ∫dW dψ (real-space)                = %+.6e\n" r1.dW_total
@printf "v1/v2 (unit const, psio-dependent)    = %.6e\n" real(r1.dW_total/v2.dW_boundary)
@printf "Fortran dW_gpec (∫, raw)              = %+.6e\n" fint(fdw)
@printf "Fortran/Julia dW after epf-scale       = %.6f  (→1 means same shape & balance)\n" (fint(fdw)/(scale*r1.dW_total))
@printf "v2 dW_running[end] (= ∫ over full ψ)  = %+.6e   (vs boundary %.10f)\n" v2.dW_running[end] (v2.dW_running[end]/real(v2.dW_boundary))
println("=================================================")

# --- (1) Julia vs Fortran setup/energies (locate the discrepancy) ---
gq0 = gq95 = gbetan = NaN
isfile(joinpath(tmp, "gpec.h5")) && h5open(joinpath(tmp, "gpec.h5"), "r") do f
    haskey(f, "equil/q0") && (gq0 = read(f["equil/q0"]))
    haskey(f, "equil/q95") && (gq95 = read(f["equil/q95"]))
    haskey(f, "equil/betan") && (gbetan = read(f["equil/betan"]))
end
println("\n=====  Julia vs Fortran (same interchange EQDSK)  =====")
@printf "%-14s %-14s %-14s\n" "quantity" "Julia" "Fortran"
@printf "%-14s %-14.4f %-14.4f\n" "q0" gq0 1.048
@printf "%-14s %-14.4f %-14.4f\n" "q95" gq95 3.844
@printf "%-14s %-14.4f %-14.4f\n" "betan" gbetan 2.301
@printf "%-14s %-14d %-14d\n" "mlow" intr.mlow -12
@printf "%-14s %-14d %-14d\n" "mhigh" intr.mhigh 20
@printf "%-14s %-14d %-14d\n" "mpert" intr.mpert 33
@printf "%-14s %-14.4f %-14.4f\n" "psilim" intr.psilim 0.9781
@printf "%-14s %-14.3e %-14.3e\n" "plasma ep" real(vac_data.ep[1]) -40.37
@printf "%-14s %-14.3e %-14.3e\n" "vacuum ev" real(vac_data.ev[1]) 28.14
@printf "%-14s %-14.3e %-14.3e\n" "total et" real(vac_data.et[1]) -12.23
println("======================================================")

# =====================  Figure 1: densities  =====================
p1 = plot(r1.psi, scale .* r1.epf; lw=2, label="Julia v1 (scaled)", color=:dodgerblue,
    xlabel="ψ", ylabel="|C|²/μ₀ density", title="epf  (field-line bending)", legend=:topleft)
plot!(p1, fP, fC2; lw=2, ls=:dash, label="Fortran recon", color=:black)
p2 = plot(r1.psi, scale .* r1.dst; lw=2, label="Julia v1 (scaled)", color=:crimson,
    xlabel="ψ", ylabel="J K |ξ_n|² density", title="dst  (K destabilizing)", legend=:topleft)
plot!(p2, fP, fKX; lw=2, ls=:dash, label="Fortran recon", color=:black)
p3 = plot(r1.psi, scale .* r1.dW; lw=2, label="Julia v1 (scaled)", color=:seagreen,
    xlabel="ψ", ylabel="½(epf−dst) density", title="dW  (energy-principle density)", legend=:topleft)
plot!(p3, fP, fdw; lw=2, ls=:dash, label="Fortran recon", color=:black)
fig1 = plot(p1, p2, p3; layout=(3, 1), size=(900, 1000),
    left_margin=12Plots.mm, bottom_margin=4Plots.mm, plot_title="recon δW densities — Julia v1 vs Fortran gpout_recon")
f1 = joinpath(outdir, "recon_densities.png"); savefig(fig1, f1)

# =====================  Figure 2: cumulative dW  =====================
cumtrapz(x, y) = [i == 1 ? 0.0 : sum(0.5 .* (y[1:i-1] .+ y[2:i]) .* diff(x[1:i])) for i in 1:length(x)]
jcum = cumtrapz(r1.psi, r1.dW) .* scale
fcum = cumtrapz(fP, fdw)
fig2 = plot(r1.psi, jcum; lw=2.5, label="Julia v1 ∫dW (scaled)", color=:seagreen,
    xlabel="ψ", ylabel="cumulative δW", title="Cumulative energy-principle δW(ψ)  (sign = stability)",
    legend=:bottomleft, left_margin=12Plots.mm, bottom_margin=4Plots.mm, size=(900, 500))
plot!(fig2, fP, fcum; lw=2.5, ls=:dash, label="Fortran recon", color=:black)
hline!(fig2, [0.0]; color=:gray, ls=:dot, label="")
f2 = joinpath(outdir, "recon_cumulative_dW.png"); savefig(fig2, f2)

# =====================  Figure 3: 2D (R,Z) energy-density maps from recon.h5  =====================
f3 = nothing
h5open(joinpath(outdir, "recon.h5"), "r") do f
    if haskey(f, "recon/maps")
        R = read(f["recon/maps/R"]); Z = read(f["recon/maps/Z"])
        C2 = read(f["recon/maps/C2_density"]); KX = read(f["recon/maps/Kxin2_density"])
        pA = scatter(vec(R), vec(Z); marker_z=vec(C2), ms=1.6, msw=0, c=:viridis, label="",
            xlabel="R [m]", ylabel="Z [m]", title="|C|²/μ₀ density", aspect_ratio=:equal, colorbar=true)
        pB = scatter(vec(R), vec(Z); marker_z=vec(KX), ms=1.6, msw=0, c=:plasma, label="",
            xlabel="R [m]", ylabel="Z [m]", title="K|ξ_n|² density", aspect_ratio=:equal, colorbar=true)
        fig3 = plot(pA, pB; layout=(1, 2), size=(1150, 620), left_margin=8Plots.mm, bottom_margin=4Plots.mm,
            plot_title="recon 2D energy-density maps (recon.h5)")
        global f3 = joinpath(outdir, "recon_rz_maps.png")
        savefig(fig3, f3)
    end
end

# =====================  Figure 4: v2 running cumulative δW over full ψ  =====================
# v2 cumulative δW(ψ) = Ξ_ψ(ψ)†·u₂(ψ): the matrix-form energy accumulated over the whole ψ
# domain (regular through rationals; endpoint = exact boundary δW).
fig4 = plot(v2.psi, v2.dW_running; lw=2.5, color=:purple, label="v2 running δW(ψ) = Ξ†u₂",
    xlabel="ψ", ylabel="cumulative δW (DCON units)", title="v2 matrix-form δW over full ψ  (endpoint = exact δW)",
    legend=:topright, left_margin=12Plots.mm, bottom_margin=4Plots.mm, size=(900, 500))
hline!(fig4, [real(v2.dW_boundary)]; color=:black, ls=:dash, label="exact boundary δW = psio²·ep")
hline!(fig4, [0.0]; color=:gray, ls=:dot, label="")
f4 = joinpath(outdir, "recon_v2_running_dW.png"); savefig(fig4, f4)

println("\nFigures saved:")
println("  ", f1)
println("  ", f2)
f3 === nothing || println("  ", f3)
println("  ", f4)
