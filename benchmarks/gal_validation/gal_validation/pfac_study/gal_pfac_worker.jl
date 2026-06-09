# Worker: for ONE pc, run the full pipeline once, then loop galerkin_solve over MANY pfacs.
# Args: pc, out_csv_path. Appends rows "pc,pfac,gal_q2,stride_q2,gal_q3,stride_q3".
using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates
const EQ = GPE.Equilibrium
using TOML, Printf, LinearAlgebra
import FastInterpolations: cubic_interp, Series, ExtendExtrap
LinearAlgebra.BLAS.set_num_threads(1)   # avoid oversubscription across parallel workers

const REPO = "/Users/pharr/Projects/GPEC_dev/docs/GPEC_julia"
const PFACS = exp10.(range(-4, 0; length=30))   # 30 packing ratios, 1e-4 .. 1e0

const PCLIST = parse.(Float64, split(ARGS[1], ','))   # comma-separated pc values
const OUTCSV = ARGS[2]

function run_pc(pc)
inputs = TOML.parsefile(joinpath(REPO, "examples", "LAR_beta_scan", "gpec.toml"))
inputs["TJ_ANALYTIC_INPUT"]["pc"] = pc
ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
ctrl.gal_flag = true; ctrl.gal_solver = "LU"; ctrl.verbose = false
dir = mktempdir(; prefix="galw_"); intr = FFS.ForceFreeStatesInternal(; dir_path=dir)
eq_config = EQ.EquilibriumConfig(inputs["Equilibrium"], dir)
tj = EQ.TJAnalyticConfig(; (Symbol(k) => v for (k, v) in inputs["TJ_ANALYTIC_INPUT"])...)
equil = EQ.setup_equilibrium(eq_config, tj)

# --- replicate main() pipeline (mirrors GeneralizedPerturbedEquilibrium.jl) ---
FFS.sing_lim!(intr, ctrl, equil)
xs = equil.profiles.xs; ls = zeros(length(xs), 5); FFS.mercier_scan!(ls, equil)
intr.locstab = cubic_interp(xs, Series(ls); extrap=ExtendExtrap())
intr.nlow = ctrl.nn_low; intr.nhigh = ctrl.nn_high; intr.npert = 1
FFS.sing_find!(intr, equil)
intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
intr.mpert = intr.mhigh - intr.mlow + 1
intr.mband = min(max(intr.mpert - 1 - ctrl.delta_mband, 0), intr.mpert - 1)
intr.numpert_total = intr.mpert * intr.npert
metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)
ffit = FFS.make_matrix(equil, intr, metric)
odet, fm_propagators, fm_chunks, fm_S_left = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
vac_data = FFS.free_run!(odet, ctrl, equil, ffit, intr)
if intr.msing > 0 && fm_propagators !== nothing
    FFS.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
        wv=vac_data.wv, psio=equil.psio, debug=false,
        S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, ffit=ffit)
end
dpm = intr.delta_prime_matrix
m_s = [s.m[1] for s in intr.sing if s.psifac < intr.psilim]
sidx2 = findfirst(==(2), m_s); sidx3 = findfirst(==(3), m_s)
s2 = sidx2 === nothing ? NaN : real(dpm[sidx2, sidx2])
s3 = sidx3 === nothing ? NaN : real(dpm[sidx3, sidx3])

open(OUTCSV, "a") do io
    for pfac in PFACS
        ctrl.gal_pfac = pfac
        g2 = NaN; g3 = NaN
        try
            res = FFS.galerkin_solve(ctrl, equil, ffit, intr; vac_data=vac_data)
            gm = res.sing_m
            gi2 = findfirst(==(2), gm); gi3 = findfirst(==(3), gm)
            g2 = gi2 === nothing ? NaN : real(res.Deltap[gi2, gi2])
            g3 = gi3 === nothing ? NaN : real(res.Deltap[gi3, gi3])
        catch e
            @warn "gal failed pc=$pc pfac=$pfac" exception=e
        end
        @printf(io, "%.6f,%.6e,%.6f,%.6f,%.6f,%.6f\n", pc, pfac, g2, s2, g3, s3); flush(io)
    end
end
rm(dir; force=true, recursive=true)
return nothing
end

for pc in PCLIST
    try
        run_pc(pc); println("done pc=$pc"); flush(stdout)
    catch e
        @warn "pc=$pc failed" exception=(e, catch_backtrace())
    end
end
println("WORKER DONE")
