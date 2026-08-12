# Worker: for ONE TokaMaker geqdsk, run the full GPEC pipeline once, then loop galerkin_solve
# over many pfacs. Records EVERY rational surface. Args: geqdsk_path, beta, out_csv.
# Appends rows: beta,pfac,m,q,gal,stride
using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates
const EQ = GPE.Equilibrium
using TOML, Printf, LinearAlgebra
import FastInterpolations: cubic_interp, Series, ExtendExtrap
LinearAlgebra.BLAS.set_num_threads(1)

const PFACS = exp10.(range(-4, 0; length=30))
const GEQDSK = ARGS[1]
const BETA = parse(Float64, ARGS[2])
const OUTCSV = ARGS[3]

# DIIID-like stability settings (efit, free-boundary nowall), no PE/forcing.
function base_inputs(dir)
    Dict(
        "Equilibrium" => Dict("eq_filename" => basename(GEQDSK), "eq_type" => "efit",
            "jac_type" => "hamada", "power_bp" => 0, "power_b" => 0, "power_r" => 0,
            "grid_type" => "log_asymptotic", "psilow" => 1e-4, "psihigh" => 0.999,
            "mpsi" => 0, "psi_accuracy" => 0.001, "mtheta" => 256, "newq0" => 0,
            "etol" => 1e-10, "force_termination" => false),
        "Wall" => Dict("shape" => "nowall", "a" => 0.2415),
        "ForceFreeStates" => Dict(
            "local_stability_flag" => true,
            "vac_flag" => true,
            "qlow" => 1.02, "qhigh" => 1e3, "sing_start" => 0,
            "nn_low" => 1, "nn_high" => 1, "delta_mlow" => 8, "delta_mhigh" => 8,
            "mthvac" => 512, "thmax0" => 1, "kinetic_factor" => 0.0,
            "eulerlagrange_tolerance" => 1e-10, "singfac_min" => 1e-4, "ucrit" => 1e4,
            "use_parallel" => true, "parallel_threads" => 1, "populate_dense_xi" => false,
            "truncate_at_dW_peak" => false, "set_psilim_via_dmlim" => true, "dmlim" => 0.2,
            "gal_flag" => true, "gal_solver" => "LU"))
end

# Run setup + the full stability/STRIDE pipeline at a given psihigh. High-β limited TokaMaker
# equilibria need a smaller psihigh (the LCFS sits at the grid edge; tracing too far out fails),
# so the caller retries down a ladder.
function run_pipeline(psihigh)
    dir = mktempdir(; prefix="galtok_")
    cp(GEQDSK, joinpath(dir, basename(GEQDSK)))
    inputs = base_inputs(dir)
    inputs["Equilibrium"]["psihigh"] = psihigh
    ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
    ctrl.verbose = false
    intr = FFS.ForceFreeStatesInternal(; dir_path=dir)
    eq_config = EQ.EquilibriumConfig(inputs["Equilibrium"], dir)
    equil = EQ.setup_equilibrium(eq_config, nothing)
    FFS.sing_lim!(intr, ctrl, equil)
    xs = equil.profiles.xs;
    ls = zeros(length(xs), 5);
    FFS.compute_ballooning_stability!(ctrl, ls, equil)
    intr.locstab = cubic_interp(xs, Series(ls); extrap=ExtendExtrap())
    intr.nlow = ctrl.nn_low;
    intr.nhigh = ctrl.nn_high;
    intr.npert = 1
    FFS.sing_find!(intr, equil)
    # Replicate main()'s post-find surface filter: drop surfaces outside [qlow, qlim].
    # Without this the worker keeps the q=1 internal-kink surface (and mis-aligns the STRIDE
    # matrix indexing), which main() excludes — the source of the spurious β≈1.6 error cliff.
    if intr.msing > 0
        qmin_int = max(ctrl.qlow, equil.params.qmin)
        keep = [j for j in 1:intr.msing if intr.sing[j].q >= qmin_int && intr.sing[j].psifac <= intr.psilim]
        intr.sing = intr.sing[keep]
        intr.msing = length(keep)
    end
    intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + ctrl.delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert
    metric = FFS.make_metric(equil, intr.mpert)
    ffit = FFS.make_matrix(equil, intr, metric)
    odet, fm_propagators, fm_chunks, fm_S_left = FFS.eulerlagrange_integration(ctrl, equil, ffit, intr)
    vac_data = FFS.free_run(odet, ctrl, equil, ffit, intr)
    if intr.msing > 0 && fm_propagators !== nothing
        FFS.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
            wv=vac_data.wv, psio=equil.psio, debug=false,
            S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, ffit=ffit)
    end
    return (ctrl=ctrl, equil=equil, ffit=ffit, intr=intr, vac_data=vac_data, dir=dir, psihigh=psihigh)
end

P = nothing
for ph in (0.999, 0.99, 0.98, 0.97, 0.95)
    try
        global P = run_pipeline(ph);
        break
    catch e
        @warn "pipeline failed at psihigh=$ph for $(basename(GEQDSK)); retrying lower" exception=e
    end
end
P === nothing && error("all psihigh retries failed for $(basename(GEQDSK))")
ctrl = P.ctrl;
equil = P.equil;
ffit = P.ffit;
intr = P.intr;
vac_data = P.vac_data;
dir = P.dir
@printf("  using psihigh=%.3f for %s\n", P.psihigh, basename(GEQDSK))
dpm = intr.delta_prime_matrix
m_s = [s.m[1] for s in intr.sing if s.psifac < intr.psilim]
q_s = [s.q for s in intr.sing if s.psifac < intr.psilim]
# STRIDE delta_prime_matrix may have fewer rows than m_s (it drops surfaces beyond its own
# active-domain cut). Match by position but guard the access so extra surfaces just get NaN.
ndpm = (dpm === nothing || isempty(dpm)) ? 0 : size(dpm, 1)
stride_diag = [i <= ndpm ? real(dpm[i, i]) : NaN for i in 1:length(m_s)]

open(OUTCSV, "a") do io
    for pfac in PFACS
        ctrl.gal_pfac = pfac
        gdiag = fill(NaN, length(m_s));
        gm = Int[]
        try
            res = FFS.galerkin_solve(ctrl, equil, ffit, intr; vac_data=vac_data)
            gm = res.sing_m
            for (i, m) in enumerate(m_s)
                gi = findfirst(==(m), gm)
                gdiag[i] = gi === nothing ? NaN : real(res.Deltap[gi, gi])
            end
        catch e
            @warn "gal failed beta=$BETA pfac=$pfac" exception=e
        end
        for i in 1:length(m_s)
            @printf(io, "%.4f,%.6e,%d,%.4f,%.6f,%.6f\n", BETA, pfac, m_s[i], q_s[i], gdiag[i], stride_diag[i])
        end
        flush(io)
    end
end
rm(dir; force=true, recursive=true)
@printf("WORKER DONE beta=%.4f  msing=%d  surfaces q=%s\n", BETA, length(m_s), string(round.(q_s; digits=2)))
