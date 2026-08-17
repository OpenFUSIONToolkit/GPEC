# BruteForceScan.jl
#
# Brute-force evaluation of a complex-Q-callable residual (`SurfaceCoupling`,
# `MultiSurfaceCoupling`, or any user-supplied function) on a regular 2D
# Q-plane grid. The output `ScanResult` is then consumed by
# `find_growth_rates` (`GrowthRateExtraction.jl`) to extract growth-rate
# eigenvalues from the Re(Δ)=0 ∩ Im(Δ)=0 contour intersections.
#
# Resolution and box are entirely user-controlled. Threading is enabled by
# default; pass `threaded=false` for deterministic single-threaded
# evaluation (e.g. when the residual is itself non-thread-safe).

"""
    ScanResult

Output of a brute-force or AMR Q-plane scan.

| field     | meaning                                                  |
|:--------- |:-------------------------------------------------------- |
| `Q`       | Complex Q values (`Matrix` for grid, `Vector` for AMR)   |
| `Δ`       | Residual values, same shape as `Q`                       |
| `re_axis` | Real-axis grid (only for regular-grid `ScanResult`)      |
| `im_axis` | Imaginary-axis grid (only for regular-grid `ScanResult`) |
"""
struct ScanResult
    Q::Matrix{ComplexF64}
    Δ::Matrix{ComplexF64}
    re_axis::Vector{Float64}
    im_axis::Vector{Float64}
end

"""
    brute_force_scan(f, Q_re_range, Q_im_range; nre, nim,
                      threaded::Bool=true) -> ScanResult

Evaluate the Q-callable residual `f` on a regular `nre × nim` grid spanning
the rectangle `Q_re_range × Q_im_range` in the complex Q plane. `f` must
accept a single `Complex` argument and return a `Complex` value (typically a
`SurfaceCoupling` or `MultiSurfaceCoupling`, but any callable works).

Use `find_growth_rates(scan, tauk; ...)` to extract growth-rate eigenvalues
from the result.

# Arguments

  - `f`           -- Q-callable residual (e.g. `SurfaceCoupling`, `MultiSurfaceCoupling`)
  - `Q_re_range`  -- `(re_min, re_max)` tuple
  - `Q_im_range`  -- `(im_min, im_max)` tuple

# Keyword arguments

  - `nre`, `nim`  -- grid resolution along each axis
  - `threaded`    -- distribute Q evaluations across `Threads.@threads`
"""
function brute_force_scan(f, Q_re_range::NTuple{2,<:Real},
    Q_im_range::NTuple{2,<:Real};
    nre::Integer, nim::Integer,
    threaded::Bool=true)
    nre >= 2 || throw(ArgumentError("brute_force_scan: nre must be ≥ 2"))
    nim >= 2 || throw(ArgumentError("brute_force_scan: nim must be ≥ 2"))
    re_axis = collect(range(Float64(Q_re_range[1]); stop=Float64(Q_re_range[2]),
        length=nre))
    im_axis = collect(range(Float64(Q_im_range[1]); stop=Float64(Q_im_range[2]),
        length=nim))
    Q = ComplexF64[(qr + qi*im) for qr in re_axis, qi in im_axis]
    Δ = Matrix{ComplexF64}(undef, nre, nim)
    if threaded
        # Pin BLAS to one thread for the parallel region — `f(Q)` calls `det()`,
        # and concurrent multithreaded-BLAS calls give non-reproducible
        # reductions that can flip the extracted root run-to-run. The residual
        # matrices are tiny, so single-threaded BLAS costs nothing here.
        _blas0 = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for j in 1:nim
                for i in 1:nre
                    Δ[i, j] = f(Q[i, j])
                end
            end
        finally
            BLAS.set_num_threads(_blas0)
        end
    else
        for j in 1:nim, i in 1:nre
            Δ[i, j] = f(Q[i, j])
        end
    end
    return ScanResult(Q, Δ, re_axis, im_axis)
end
