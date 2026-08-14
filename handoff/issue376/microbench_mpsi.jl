# Issue #376 microbenchmark: does CubicSeriesInterpolant scalar eval cost grow with knot count?
# Mirrors the ForceFreeStates hot path: 676 ComplexF64 series (numpert=26), non-uniform Vector
# psi grid, in-place eval with a persistent hint, monotone / Vern9-stage-like query patterns.
using FastInterpolations
using Printf

const NSERIES = 26^2
const PSILOW, PSIHIGH = 1e-4, 0.995
const NPOINTS_LADDER = [33, 65, 129, 257, 513, 1025, 2049]

# Mildly non-uniform monotone grid (edge-packed like real ldp/auto grids)
function make_grid(n)
    t = range(0.0, 1.0; length=n)
    return PSILOW .+ (PSIHIGH - PSILOW) .* (0.6 .* t .+ 0.4 .* t .^ 3)
end

# Smooth mode-dependent complex data, deterministic
function make_series_data(x, nseries)
    n = length(x)
    Y = Matrix{ComplexF64}(undef, n, nseries)
    for k in 1:nseries
        f = 1.0 + 0.02 * k
        @. Y[:, k] = complex(sin(f * x) + 0.1 * cos(3.1 * f * x), cos(0.7 * f * x))
    end
    return Y
end

# Vern9-like query schedule: n_steps ODE steps across the domain, 16 stage evals per step at
# fixed non-monotone fractional offsets (stresses the hint walk the way an RK stage loop does).
const STAGE_C = [0.0, 0.03462, 0.09703, 0.14553, 0.561, 0.229, 0.5449, 0.6453, 0.4874, 0.9648, 0.8755, 0.9553, 0.7514, 1.0, 0.9847, 1.0]
function stage_queries(n_steps)
    h = (PSIHIGH - PSILOW) / n_steps
    q = Vector{Float64}(undef, n_steps * length(STAGE_C))
    i = 0
    for s in 1:n_steps
        t0 = PSILOW + (s - 1) * h
        for c in STAGE_C
            q[i+=1] = min(t0 + c * h, PSIHIGH)
        end
    end
    return q
end

monotone_queries(neval) = collect(range(PSILOW, PSIHIGH; length=neval))

# One timed pass: reset hint, evaluate all queries in place; returns seconds
function eval_pass!(out, sitp, queries, hint)
    hint isa Base.RefValue && (hint[] = 1)
    t0 = time_ns()
    if hint === nothing
        @inbounds for xq in queries
            sitp(out, xq)
        end
    else
        @inbounds for xq in queries
            sitp(out, xq; hint=hint)
        end
    end
    return (time_ns() - t0) / 1e9
end

function bench(sitp, queries; use_hint=true, reps=5)
    out = Vector{ComplexF64}(undef, length(sitp(first(queries))))
    hint = use_hint ? Ref(1) : nothing
    eval_pass!(out, sitp, queries, hint)  # warmup (also builds lazy transpose)
    best = minimum(eval_pass!(out, sitp, queries, hint) for _ in 1:reps)
    allocs = @allocated eval_pass!(out, sitp, queries, hint)
    return best / length(queries) * 1e9, allocs / length(queries)  # ns/eval, bytes/eval
end

# Single-series (q_spline analogue) control
function bench_scalar(itp, queries; use_hint=true, reps=5)
    hint = use_hint ? Ref(1) : nothing
    function pass()
        hint isa Base.RefValue && (hint[] = 1)
        acc = 0.0
        t0 = time_ns()
        if hint === nothing
            @inbounds for xq in queries
                acc += itp(xq)
            end
        else
            @inbounds for xq in queries
                acc += itp(xq; hint=hint)
            end
        end
        return (time_ns() - t0) / 1e9 + 0.0 * acc
    end
    pass()
    best = minimum(pass() for _ in 1:reps)
    return best / length(queries) * 1e9
end

function main()
    n_ode_steps = 1000                 # fixed step density, independent of knot count (the real situation)
    qs_stage = stage_queries(n_ode_steps)
    qs_mono = monotone_queries(16 * n_ode_steps)

    println("Queries: $(length(qs_stage)) stage-pattern evals, $(length(qs_mono)) monotone evals")
    println("Series count: $NSERIES ComplexF64\n")

    @printf("%7s | %11s %11s | %11s %11s | %11s | %11s | %9s | %10s\n",
            "npts", "mono ns/ev", "stage ns/ev", "nohint mono", "nohint stg", "range grid", "3sep stage", "B/eval", "MB footpr")
    for n in NPOINTS_LADDER
        x = collect(make_grid(n))
        Y = make_series_data(x, NSERIES)
        sitp = cubic_interp(x, Series(Y); extrap=ExtendExtrap())
        precompute_transpose!(sitp)

        t_mono, b_mono = bench(sitp, qs_mono)
        t_stage, _ = bench(sitp, qs_stage)
        t_mono_nh, _ = bench(sitp, qs_mono; use_hint=false)
        t_stage_nh, _ = bench(sitp, qs_stage; use_hint=false)

        # uniform range grid (DirectSearch path)
        xr = range(PSILOW, PSIHIGH; length=n)
        sitp_r = cubic_interp(xr, Series(make_series_data(collect(xr), NSERIES)); extrap=ExtendExtrap())
        precompute_transpose!(sitp_r)
        t_range, _ = bench(sitp_r, qs_stage)

        # 3 separate interpolants at same psi (fmats/kmats/gmats pattern) — report per single-interp eval
        s1 = cubic_interp(x, Series(Y); extrap=ExtendExtrap())
        s2 = cubic_interp(x, Series(make_series_data(x .* 0.999 .+ 1e-5, NSERIES)); extrap=ExtendExtrap())
        s3 = cubic_interp(x, Series(make_series_data(x .* 1.001, NSERIES)); extrap=ExtendExtrap())
        foreach(precompute_transpose!, (s1, s2, s3))
        out3 = Vector{ComplexF64}(undef, NSERIES)
        h3 = Ref(1)
        function pass3()
            h3[] = 1
            t0 = time_ns()
            @inbounds for xq in qs_stage
                s1(out3, xq; hint=h3)
                s2(out3, xq; hint=h3)
                s3(out3, xq; hint=h3)
            end
            return (time_ns() - t0) / 1e9
        end
        pass3()
        t_3sep = minimum(pass3() for _ in 1:3) / (3 * length(qs_stage)) * 1e9

        mb = Base.summarysize(sitp) / 1e6
        @printf("%7d | %11.1f %11.1f | %11.1f %11.1f | %11.1f | %11.1f | %9.1f | %10.2f\n",
                n, t_mono, t_stage, t_mono_nh, t_stage_nh, t_range, t_3sep, b_mono, mb)
    end

    println("\nFirst-touch lazy transpose cost (ms) and single-series q_spline control (ns/eval, stage pattern):")
    @printf("%7s | %12s | %12s %12s\n", "npts", "permute ms", "qspl hint", "qspl nohint")
    for n in NPOINTS_LADDER
        x = collect(make_grid(n))
        sitp = cubic_interp(x, Series(make_series_data(x, NSERIES)); extrap=ExtendExtrap())
        out = Vector{ComplexF64}(undef, NSERIES)
        t_first = @elapsed sitp(out, 0.5)   # includes permutedims of y and z

        y = sin.(3.0 .* x) .+ 0.2 .* x
        itp = cubic_interp(x, y; extrap=ExtendExtrap())
        tq = bench_scalar(itp, qs_stage)
        tq_nh = bench_scalar(itp, qs_stage; use_hint=false)
        @printf("%7d | %12.2f | %12.1f %12.1f\n", n, t_first * 1e3, tq, tq_nh)
    end
end

main()
