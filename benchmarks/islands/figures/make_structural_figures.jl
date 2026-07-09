# make_structural_figures.jl — pinned figure script (design docs/07 §2)
#
# Regenerates the structural (A-ladder) figures embedded in
# docs/src/islands/numerics.md from the verification machinery itself — no
# archived data needed yet because everything here is deterministic M1/M2
# structure (manufactured coefficients, no physics). Physics (B-ladder) figures
# will read archived benchmark data once QUESTIONS Q2–Q4 clear.
#
# Usage (repo root):
#   env -u LD_LIBRARY_PATH julia --project=. benchmarks/islands/figures/make_structural_figures.jl
#
# Writes PNGs into docs/src/islands/figures/ (committed as docs assets so the
# Documenter CI build does not need to run this script).

ENV["GKSwstype"] = "100"    # headless GR

using Plots
using LinearAlgebra
using GeneralizedPerturbedEquilibrium
const Isl = GeneralizedPerturbedEquilibrium.Islands
const PS = Isl.PhaseSpace
const Op = Isl.Operators
const So = Isl.Solvers
const V = Isl.Verify
const Fi = Isl.Fields

outdir = normpath(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "islands", "figures"))
mkpath(outdir)
saved = String[]
function save!(p, name)
    path = joinpath(outdir, name)
    savefig(p, path)
    push!(saved, path)
    return path
end

# ---------------------------------------------------------------------------
# F1 — layer-clustered grids: the sinh maps and node placement (04 §1)
# ---------------------------------------------------------------------------
let
    gx = PS.MappedFDGrid(33; halfwidth=6.0, clustering=2.0, order=4)
    gy = PS.MappedFDGrid(25; halfwidth=4.0, clustering=2.0, center=1.0, domain=:half, order=4)
    s = range(-1, 1; length=33)
    p1 = plot(s, gx.nodes; lw=2, xlabel="computational coordinate s", ylabel="x",
        label="x(s), β=2", legend=:topleft, title="radial map (packs x = 0)")
    scatter!(p1, s, gx.nodes; ms=3, label="nodes")
    p2 = plot(; xlabel="node index", ylabel="y", title="pitch grid (packs y_c = 1)", legend=:topleft)
    scatter!(p2, 1:gy.n, gy.nodes; ms=4, label="y nodes, β=2")
    hline!(p2, [1.0]; ls=:dash, lc=:red, label="y_c (trapped–passing)")
    p = plot(p1, p2; layout=(1, 2), size=(950, 380), left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    save!(p, "grids_clustering.png")
end

# ---------------------------------------------------------------------------
# F2 — verification ladder A1: MMS convergence, operators + assembled + solve
# ---------------------------------------------------------------------------
let
    mkg(n) = PS.IslandGrid(; nx=n, nxi=8, ny=n, nE=3, halfwidth_x=6.0, clustering_x=1.0,
        y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)
    ns = [9, 17, 33]
    p = plot(; xscale=:log10, yscale=:log10, xlabel="radial/pitch resolution n",
        ylabel="max error", legend=:bottomleft, title="MMS convergence (ladder A1)",
        size=(700, 480), left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    for (term, lab) in ((:streaming, "streaming"), (:exb, "E×B bracket"),
        (:collisions, "collisions"), (:perp, "⊥ transport"))
        errs = [V.mms_operator_error(mkg(n), term) for n in ns]
        plot!(p, ns, errs; marker=:circle, lw=2, label=lab)
    end
    errs = [V.mms_assembled_error(mkg(n)) for n in ns]
    plot!(p, ns, errs; marker=:square, lw=3, lc=:black, label="assembled residual")
    solve_errs = [V.solve_mms(n).err for n in (9, 17, 33)]
    plot!(p, [9, 17, 33], solve_errs; marker=:diamond, lw=3, ls=:dash, label="converged solve (A1-solve)")
    guide = errs[1] .* (ns[1] ./ ns) .^ 4
    plot!(p, ns, guide; ls=:dot, lc=:gray, label="4th order")
    save!(p, "mms_convergence.png")
end

# ---------------------------------------------------------------------------
# F3 — the flattened-electron geometry functions and the A7 identity (01 §2.4)
# ---------------------------------------------------------------------------
let
    Ωout = 1.02:0.02:5.0
    Ωin = -0.98:0.02:0.98
    Q = [Fi.Q_omega(Ω) for Ω in Ωout]
    Qin = [Fi.Q_omega(Ω) for Ω in Ωin]
    h = [Fi.h_profile(Ω; prefactor=1.0) for Ω in Ωout]
    p1 = plot(Ωin, Qin; lw=2, label="Q(Ω), inside", xlabel="Ω", ylabel="Q",
        title="Q(Ω) = (1/2π)∮√(Ω+cos ξ) dξ", legend=:topleft)
    plot!(p1, Ωout, Q; lw=2, label="Q(Ω), outside")
    vline!(p1, [1.0]; ls=:dash, lc=:red, label="separatrix")
    a7 = [abs(Fi.flat_average_d2h_dx2(Ω, 1.0)) for Ω in (1.2, 2.0, 3.0, 5.0)]
    p2 = plot(Ωout, h; lw=2, label="h(Ω) (unit prefactor)", xlabel="Ω", ylabel="h",
        title="electron profile h(Ω)", legend=:topleft)
    vline!(p2, [1.0]; ls=:dash, lc=:red, label="separatrix (h ≡ 0 inside)")
    annotate!(p2, 1.6, 0.35, text("A7: max |⟨∂²h/∂x²⟩_Ω| = $(round(maximum(a7); sigdigits=2))", 9, :left))
    p = plot(p1, p2; layout=(1, 2), size=(950, 380), left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=6Plots.mm)
    save!(p, "hQ_profiles.png")
end

# ---------------------------------------------------------------------------
# F4 — preconditioner quality (04 §5): GMRES iterations with/without YBlockJacobi
# ---------------------------------------------------------------------------
let
    g = PS.IslandGrid(; nx=9, nxi=8, ny=9, nE=2, halfwidth_x=6.0, clustering_x=1.0,
        y_max=4.0, y_c=1.0, clustering_y=0.8, order=4)
    nx, nξ, ny, nE, nσ = PS.nnodes(g)
    P = @. g.y.nodes * (4.0 - g.y.nodes)
    K, = Op.conservative_pitch_operator(g.y, P, ones(ny))
    cstiff = fill(30.0, nx, nξ, nE, nσ)
    shift = fill(-1.0, nx, nξ, ny, nE, nσ)
    stack = Op.IslandStack((Op.PitchAngleDiffusion(K, cstiff), Op.RadiationSink(shift)),
        Op.Quasineutrality(1.3))
    f0! = So.flat_residual(stack, g)
    N = Op.statelength(g)
    b = sin.((1:N) ./ 7)
    f!(out, u) = (f0!(out, u); out .-= b; out)
    pc = So.YBlockJacobi(g, (ix, iξ, iE, iσ) -> I(ny) + cstiff[ix, iξ, iE, iσ] .* K; phi_scale=-1.3)
    s0 = So.newton_krylov(f!, zeros(N); rtol=1e-10, memory=300)
    s1 = So.newton_krylov(f!, zeros(N); rtol=1e-10, memory=300, precond=pc)
    p = bar(["unpreconditioned", "y-block Jacobi (TSVD)"], [s0.gmres_iters, s1.gmres_iters];
        ylabel="total GMRES iterations", legend=false,
        title="preconditioner: stiff collisional solve",
        ylims=(0, 1.25 * s0.gmres_iters),
        size=(600, 430), left_margin=12Plots.mm, bottom_margin=6Plots.mm, top_margin=4Plots.mm)
    annotate!(p, [(1, s0.gmres_iters + 0.06 * s0.gmres_iters, text("$(s0.gmres_iters)", 10)),
        (2, s1.gmres_iters + 0.06 * s0.gmres_iters, text("$(s1.gmres_iters)", 10))])
    save!(p, "preconditioner_gmres.png")
end

# ---------------------------------------------------------------------------
# F5 — pseudo-arclength continuation around the toy fold (03 §3)
# ---------------------------------------------------------------------------
let
    ftoy!(out, u, p) = (out[1] = u[1]^2 + p; out)
    pa = So.pseudo_arclength(ftoy!, [1.0], -1.0; ds=0.15, nsteps=30, rtol=1e-12, atol=1e-12)
    us = [z[1] for z in pa.us]
    p = plot(pa.ps, us; marker=:circle, lw=2, xlabel="parameter p", ylabel="u",
        label="continuation path", legend=:topleft,
        title="pseudo-arclength steps around the fold (u² + p = 0)",
        size=(650, 430), left_margin=12Plots.mm, bottom_margin=6Plots.mm)
    plot!(p, -1.2:0.01:0.0, sqrt.(-(-1.2:0.01:0.0)); ls=:dash, lc=:gray, label="analytic ±√(−p)")
    plot!(p, -1.2:0.01:0.0, -sqrt.(-(-1.2:0.01:0.0)); ls=:dash, lc=:gray, label="")
    if !isempty(pa.folds)
        k = pa.folds[1] + 1
        scatter!(p, [pa.ps[k]], [us[k]]; ms=8, mc=:red, label="fold detected")
    end
    save!(p, "continuation_fold.png")
end

println("Saved figures:")
foreach(f -> println("  ", f), saved)
