# Measure the spline third-derivative JUMPS of the EL coefficient matrices at their knots --
# the quantity an adaptive integrator actually chokes on when crossing a C2 interpolant --
# and predict the accepted-step count from them via h_k = (tol/J_k)^(1/4).
using HDF5, Printf, Statistics, FastInterpolations

const TOL = 1e-10

# f''' on each interval from Hermite data (cubic spline = Hermite with its nodal derivatives)
function interval_f3(xs, ys, yp)
    n = length(xs); f3 = zeros(n - 1)
    for k in 1:n-1
        h = xs[k+1] - xs[k]
        f3[k] = 12 * (ys[k] - ys[k+1]) / h^3 + 6 * (yp[k] + yp[k+1]) / h^2
    end
    return f3
end

function knot_jumps(xs, ys)
    sp = cubic_interp(xs, ys)
    d = deriv1(sp)
    yp = [d(x) for x in xs]
    f3 = interval_f3(xs, ys, yp)
    return abs.(diff(f3))          # J at interior knots 2..n-1
end

function analyze(rundir; nelem=20)
    h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        psi = read(h5["ForceFreeStates/EulerLagrangeMatrices/psi"])
        n = length(psi)
        Jmax = zeros(n - 2)        # worst relative jump across sampled elements, per interior knot
        for key in ("F", "K", "G")
            M = read(h5["ForceFreeStates/EulerLagrangeMatrices/Ideal"][key])
            amp = dropdims(sum(abs, M; dims=1); dims=1)
            order = sortperm(vec(amp); rev=true)[1:min(nelem, length(amp))]
            for lin in order
                i, j = Tuple(CartesianIndices(amp)[lin])
                col = real.(M[:, i, j]); s = maximum(abs, col); s == 0 && continue
                J = knot_jumps(psi, col) ./ s
                Jmax .= max.(Jmax, J)
            end
        end
        # per-knot forced step and predicted extra steps: local spacing / h_k when h_k < spacing
        dbar = [(psi[k+1] - psi[k-1]) / 2 for k in 2:n-1]
        h = (TOL ./ Jmax) .^ 0.25
        pred = sum(max.(dbar ./ h .- 1.0, 0.0))
        core = findall(k -> psi[k] < 0.05, 2:n-1)
        pred_core = sum(max.(dbar[core] ./ h[core] .- 1.0, 0.0))
        medJcore = isempty(core) ? NaN : median(Jmax[core])
        medJmid = median(Jmax[[k for k in eachindex(dbar) if 0.3 <= psi[k+1] <= 0.7]])
        return (n=n, pred=pred, pred_core=pred_core, medJcore=medJcore, medJmid=medJmid)
    end
end

@printf("%-14s %6s | %9s %9s | %11s %11s\n", "run", "npsi", "predEx", "predCore", "medJ core", "medJ mid")
for rd in ARGS
    isfile(joinpath(rd, "gpec.h5")) || continue
    r = analyze(rd)
    @printf("%-14s %6d | %9.0f %9.0f | %11.3e %11.3e\n", basename(rd), r.n, r.pred, r.pred_core, r.medJcore, r.medJmid)
end
