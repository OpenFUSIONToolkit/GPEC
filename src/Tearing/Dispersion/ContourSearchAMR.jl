# Cell-based adaptive mesh refinement of the complex Q plane: cells whose corner residuals straddle
# zero in Re or Im are split into quadrants, and every f(Q) evaluation is shared through a hash cache.
# The axis-aligned cells feed the marching-squares extraction in `GrowthRateExtraction.jl`.

# Corner ordering: 1 = BL, 2 = BR, 3 = TL, 4 = TR.

"""
    AMRCell

A single axis-aligned-rectangle cell of an AMR scan. The four corner Q
values (`q_bl`, `q_br`, `q_tl`, `q_tr`) and corresponding residual values
(`d_bl`, `d_br`, `d_tl`, `d_tr`) are sufficient for marching-squares
contour extraction.
"""
struct AMRCell
    q_bl::ComplexF64
    q_br::ComplexF64
    q_tl::ComplexF64
    q_tr::ComplexF64
    d_bl::ComplexF64
    d_br::ComplexF64
    d_tl::ComplexF64
    d_tr::ComplexF64
end

"""
    AMRResult

Output of `amr_scan`.

| field       | meaning                                                  |
|:----------- |:-------------------------------------------------------- |
| `cells`     | Final list of `AMRCell` after all refinement passes      |
| `Q`         | Flat `Vector{ComplexF64}` of every unique residual eval  |
| `Δ`         | Corresponding `Vector{ComplexF64}` of residual values    |
| `truncated` | `true` if `max_cells` was hit under `:warn_truncate`; the scan is then not converged |
"""
struct AMRResult
    cells::Vector{AMRCell}
    Q::Vector{ComplexF64}
    Δ::Vector{ComplexF64}
    truncated::Bool
end

# A scan that ran to completion is not truncated.
AMRResult(cells::Vector{AMRCell}, Q::Vector{ComplexF64}, Δ::Vector{ComplexF64}) =
    AMRResult(cells, Q, Δ, false)

# Return the cached residual at `q`, evaluating and storing it on a miss.
@inline function _cached_eval!(cache::Dict{ComplexF64,ComplexF64},
    f, q::ComplexF64)
    haskey(cache, q) && return cache[q]
    Δ = ComplexF64(f(q))
    cache[q] = Δ
    return Δ
end

# Evaluate `f` at every uncached point of `qs` (threaded when `parallel` and several threads exist),
# then fill `cache` serially to avoid Dict races. `f` must be thread-safe.
function _bulk_eval_into_cache!(cache::Dict{ComplexF64,ComplexF64}, f,
    qs::AbstractVector{ComplexF64};
    parallel::Bool)
    # Collect the unique uncached points.
    seen = Set{ComplexF64}()
    new_qs = Vector{ComplexF64}()
    for q in qs
        if !haskey(cache, q) && !(q in seen)
            push!(new_qs, q)
            push!(seen, q)
        end
    end
    isempty(new_qs) && return
    new_vals = Vector{ComplexF64}(undef, length(new_qs))
    if parallel && Threads.nthreads() > 1
        # Pin BLAS to one thread: concurrent multithreaded `det()` reductions are not reproducible
        # and can flip the extracted root between near-degenerate solutions.
        _blas0 = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            Threads.@threads for k in eachindex(new_qs)
                new_vals[k] = ComplexF64(f(new_qs[k]))
            end
        finally
            BLAS.set_num_threads(_blas0)
        end
    else
        @inbounds for k in eachindex(new_qs)
            new_vals[k] = ComplexF64(f(new_qs[k]))
        end
    end
    @inbounds for k in eachindex(new_qs)
        cache[new_qs[k]] = new_vals[k]
    end
    return
end

# Does `vals` straddle zero? A non-finite corner (the layer's NaN sentinel near a pole) counts as
# a crossing so the cell is refined rather than silently dropped.
@inline function _crosses_zero(vals)
    any(!isfinite, vals) && return true
    return minimum(vals) * maximum(vals) <= 0
end

"""
    amr_scan(f, Q_re_range, Q_im_range;
              nre0, nim0, passes,
              max_cells=10_000_000,
              max_cells_action=:error,
              snapshot_callback=nothing,
              parallel=Threads.nthreads() > 1) -> AMRResult

Adaptively refine a Q-plane scan of the residual `f(Q)`. Starting from an `nre0 × nim0` grid of cells
over `Q_re_range × Q_im_range`, each of `passes` passes splits every cell whose corner residuals
straddle zero in `Re(Δ)` or `Im(Δ)` into 4 quadrants. Evaluations are cached, so shared corners cost
one evaluation. The `AMRResult` holds the final cells and every unique Q/Δ evaluation.

# Keyword arguments

  - `nre0`, `nim0`   -- initial grid cell counts along each axis
  - `passes`         -- number of refinement passes
  - `max_cells`      -- cap on total cells
  - `max_cells_action` -- on hitting the cap, `:error` throws and `:warn_truncate` warns and returns the
    partial result with `truncated = true`
  - `snapshot_callback` -- if set, called as `(pass, cells, cache)` after the initial grid (pass 0) and
    each pass; the arguments are live references, so copy them to keep them
  - `parallel`       -- evaluate `f` with `Threads.@threads`; `f` must be thread-safe. The result does
    not depend on the thread count.
"""
function amr_scan(f, Q_re_range::NTuple{2,<:Real},
    Q_im_range::NTuple{2,<:Real};
    nre0::Integer, nim0::Integer, passes::Integer,
    max_cells::Integer=10_000_000,
    max_cells_action::Symbol=:error,
    snapshot_callback::Union{Nothing,Function}=nothing,
    parallel::Bool=Threads.nthreads() > 1)
    nre0 >= 1 || throw(ArgumentError("amr_scan: nre0 must be ≥ 1"))
    nim0 >= 1 || throw(ArgumentError("amr_scan: nim0 must be ≥ 1"))
    passes >= 0 || throw(ArgumentError("amr_scan: passes must be ≥ 0"))
    max_cells_action in (:error, :warn_truncate) ||
        throw(ArgumentError("amr_scan: max_cells_action must be :error or " *
                            ":warn_truncate, got :$max_cells_action"))

    re_lo, re_hi = Float64.(Q_re_range)
    im_lo, im_hi = Float64.(Q_im_range)
    re_step = (re_hi - re_lo) / nre0
    im_step = (im_hi - im_lo) / nim0

    cache = Dict{ComplexF64,ComplexF64}()

    # Coarse initial grid: evaluate all corners, then build cells from the cache.
    ncorners_x = nre0 + 1
    ncorners_y = nim0 + 1
    corners = Vector{ComplexF64}(undef, ncorners_x * ncorners_y)
    @inbounds for j in 0:nim0, i in 0:nre0
        corners[j*ncorners_x+i+1] =
            ComplexF64(re_lo + i * re_step, im_lo + j * im_step)
    end
    _bulk_eval_into_cache!(cache, f, corners; parallel=parallel)

    cells = Vector{AMRCell}(undef, nre0 * nim0)
    @inbounds for j in 0:nim0-1, i in 0:nre0-1
        # Reuse the cached corner values exactly; recomputing them can miss the cache by one ulp.
        q_bl = corners[j*ncorners_x+i+1]
        q_br = corners[j*ncorners_x+(i+1)+1]
        q_tl = corners[(j+1)*ncorners_x+i+1]
        q_tr = corners[(j+1)*ncorners_x+(i+1)+1]
        cells[j*nre0+i+1] = AMRCell(q_bl, q_br, q_tl, q_tr,
            cache[q_bl], cache[q_br],
            cache[q_tl], cache[q_tr])
    end

    snapshot_callback === nothing || snapshot_callback(0, cells, cache)

    # Refinement passes
    truncated = false   # set true when max_cells is hit and action == :warn_truncate
    for pass_idx in 1:passes
        truncated && break
        # Flag cells with a zero crossing and collect their 5 midpoints (BM, TM, LM, RM, MM).
        flagged_idx = Int[]
        new_qs = Vector{ComplexF64}()
        sizehint!(new_qs, length(cells))
        for (idx, cell) in enumerate(cells)
            re_corners = (real(cell.d_bl), real(cell.d_br),
                real(cell.d_tl), real(cell.d_tr))
            im_corners = (imag(cell.d_bl), imag(cell.d_br),
                imag(cell.d_tl), imag(cell.d_tr))
            if _crosses_zero(re_corners) || _crosses_zero(im_corners)
                push!(flagged_idx, idx)
                push!(new_qs, 0.5 * (cell.q_bl + cell.q_br))
                push!(new_qs, 0.5 * (cell.q_tl + cell.q_tr))
                push!(new_qs, 0.5 * (cell.q_bl + cell.q_tl))
                push!(new_qs, 0.5 * (cell.q_br + cell.q_tr))
                push!(new_qs, 0.25 * (cell.q_bl + cell.q_br +
                                      cell.q_tl + cell.q_tr))
            end
        end

        _bulk_eval_into_cache!(cache, f, new_qs; parallel=parallel)

        # Build the refined cell list from the cache.
        new_cells = Vector{AMRCell}()
        sizehint!(new_cells, length(cells) + 3 * length(flagged_idx))
        flagged_set = Set(flagged_idx)
        skip_remaining = false   # true once max_cells is hit (warn_truncate path)
        for (idx, cell) in enumerate(cells)
            if idx in flagged_set && !skip_remaining
                q_bm = 0.5 * (cell.q_bl + cell.q_br)
                q_tm = 0.5 * (cell.q_tl + cell.q_tr)
                q_lm = 0.5 * (cell.q_bl + cell.q_tl)
                q_rm = 0.5 * (cell.q_br + cell.q_tr)
                q_mm = 0.25 * (cell.q_bl + cell.q_br +
                               cell.q_tl + cell.q_tr)
                d_bm = cache[q_bm]
                d_tm = cache[q_tm]
                d_lm = cache[q_lm]
                d_rm = cache[q_rm]
                d_mm = cache[q_mm]
                push!(new_cells,
                    AMRCell(cell.q_bl, q_bm, q_lm, q_mm,
                        cell.d_bl, d_bm, d_lm, d_mm),
                    AMRCell(q_bm, cell.q_br, q_mm, q_rm,
                        d_bm, cell.d_br, d_mm, d_rm),
                    AMRCell(q_lm, q_mm, cell.q_tl, q_tm,
                        d_lm, d_mm, cell.d_tl, d_tm),
                    AMRCell(q_mm, q_rm, q_tm, cell.q_tr,
                        d_mm, d_rm, d_tm, cell.d_tr))
            else
                push!(new_cells, cell)
            end
            if length(new_cells) > max_cells
                if max_cells_action === :error
                    error(
                        "amr_scan: exceeded max_cells=$max_cells " *
                        "(currently $(length(new_cells))). Reduce " *
                        "`passes` or raise `max_cells`, or pass " *
                        "max_cells_action=:warn_truncate to truncate gracefully."
                    )
                else  # :warn_truncate (validated at function entry)
                    @warn "amr_scan: max_cells=$max_cells reached at pass=$pass_idx cell=$idx/$(length(cells)); truncating refinement here and skipping remaining passes"
                    skip_remaining = true
                    truncated = true
                end
            end
        end
        cells = new_cells
        snapshot_callback === nothing || snapshot_callback(pass_idx, cells, cache)
    end

    # Flatten the cache into output Q/Δ vectors
    n = length(cache)
    Q = Vector{ComplexF64}(undef, n)
    Δ = Vector{ComplexF64}(undef, n)
    for (k, (q, d)) in enumerate(cache)
        Q[k] = q
        Δ[k] = d
    end

    return AMRResult(cells, Q, Δ, truncated)
end

# Multi-box AMR scan with pre-screen: skip refinement on boxes whose coarse grid shows no zero
# crossing and no large |Δ|. The magnitude test catches a pole inside one pre-screen cell whose
# corners all sample the same lobe, which the sign tests would miss.

"""
    BoxActivity

Why a box was retained or skipped by `multi_box_amr_scan`. `NoActivity` means
the pre-screen grid showed no zero-isoline crossings and no large-`|Δ|`
corners; the box is excluded from refinement. The other variants record which
criterion fired first.
"""
@enum BoxActivity NoActivity ReZeroCrossing ImZeroCrossing PoleMagnitude

# Return the first activity criterion met by any pre-screen cell, or `NoActivity`.
function _check_box_activity(cells::AbstractVector{AMRCell},
    pole_magnitude_threshold::Real)
    @inbounds for cell in cells
        re_corners = (real(cell.d_bl), real(cell.d_br),
            real(cell.d_tl), real(cell.d_tr))
        im_corners = (imag(cell.d_bl), imag(cell.d_br),
            imag(cell.d_tl), imag(cell.d_tr))
        _crosses_zero(re_corners) && return ReZeroCrossing
        _crosses_zero(im_corners) && return ImZeroCrossing
        if max(abs(cell.d_bl), abs(cell.d_br),
            abs(cell.d_tl), abs(cell.d_tr)) >= pole_magnitude_threshold
            return PoleMagnitude
        end
    end
    return NoActivity
end

"""
    MultiBoxAMRResult

Output of `multi_box_amr_scan`. Per-box `AMRResult`s plus the aggregated
cells/Q/Δ across all *active* boxes. Pre-screen-inactive boxes have `nothing`
for their `AMRResult` and contribute nothing to the aggregated arrays.

| field             | meaning                                                |
|:----------------- |:------------------------------------------------------ |
| `box_results`     | per-box `AMRResult`, or `nothing` if box was skipped   |
| `box_activity`    | per-box `BoxActivity` enum                             |
| `cells`           | concatenated `AMRCell`s from all active boxes          |
| `Q`               | union of all unique `Q` evaluations (active + skipped) |
| `Δ`               | corresponding `Δ` values                               |
| `prescreen_evals` | total `f(Q)` evaluations spent on pre-screening        |

The aggregated `(cells, Q, Δ)` are suitable for direct consumption by
`find_growth_rates`. Pre-screen evaluations are still included in `Q`/`Δ` even
for skipped boxes, so any downstream pole-magnitude diagnostic that uses the
flat residual list sees the full sample.
"""
struct MultiBoxAMRResult
    box_results::Vector{Union{Nothing,AMRResult}}
    box_activity::Vector{BoxActivity}
    cells::Vector{AMRCell}
    Q::Vector{ComplexF64}
    Δ::Vector{ComplexF64}
    prescreen_evals::Int
end

"""
    multi_box_amr_scan(f, boxes;
                       pole_magnitude_threshold,
                       prescreen_nre=25, prescreen_nim=25,
                       nre0=25, nim0=25, passes=4,
                       max_cells=10_000_000,
                       max_cells_action=:error,
                       parallel=Threads.nthreads() > 1) -> MultiBoxAMRResult

Run `amr_scan` over several Q-plane boxes, skipping boxes that a coarse pre-screen finds inactive.
A box is active if any `prescreen_nre × prescreen_nim` pre-screen cell has a sign change of `Re(Δ)`
or `Im(Δ)` across its corners, or a corner with `|Δ| ≥ pole_magnitude_threshold`. Active boxes get a
full `amr_scan`; inactive ones get `nothing`.

# Arguments

  - `f`: residual `Q::ComplexF64 → Δ::ComplexF64`; must be thread-safe if `parallel=true`.
  - `boxes`: vector of `(Q_re_range, Q_im_range)` tuples; boxes may overlap or share edges.
  - `pole_magnitude_threshold`: `|Δ|` activity threshold, e.g. the adaptive `pole_threshold` used in
    `find_growth_rates`.
  - `nre0, nim0, passes, max_cells, max_cells_action, parallel`: forwarded to each `amr_scan`.

Pass the result through `as_amr_result` for `find_growth_rates`.

# Example

```julia
boxes = [((-75.0, -25.0), (-25.0, 25.0)),
    ((-25.0, 25.0), (-25.0, 25.0)),
    ((25.0, 75.0), (-25.0, 25.0))]
result = multi_box_amr_scan(f_residual, boxes;
    pole_magnitude_threshold=1e-3,
    prescreen_nre=25, prescreen_nim=25,
    nre0=25, nim0=25, passes=4)
roots = find_growth_rates(as_amr_result(result), tauk; pole_threshold=1e-3)
```
"""
function multi_box_amr_scan(f,
    boxes::AbstractVector;
    pole_magnitude_threshold::Real,
    prescreen_nre::Integer=25, prescreen_nim::Integer=25,
    nre0::Integer=25, nim0::Integer=25, passes::Integer=4,
    max_cells::Integer=10_000_000,
    max_cells_action::Symbol=:error,
    parallel::Bool=Threads.nthreads() > 1)
    prescreen_nre >= 1 || throw(ArgumentError("multi_box_amr_scan: prescreen_nre must be ≥ 1"))
    prescreen_nim >= 1 || throw(ArgumentError("multi_box_amr_scan: prescreen_nim must be ≥ 1"))
    pole_magnitude_threshold >= 0 ||
        throw(ArgumentError("multi_box_amr_scan: pole_magnitude_threshold must be ≥ 0"))

    n_boxes = length(boxes)
    box_results = Vector{Union{Nothing,AMRResult}}(undef, n_boxes)
    box_activity = Vector{BoxActivity}(undef, n_boxes)
    prescreen_evals_total = 0

    # Deduplicated Q/Δ over all boxes and pre-screens; cells from active boxes only.
    qd_aggregate = Dict{ComplexF64,ComplexF64}()
    cells_aggregate = AMRCell[]

    for (b_idx, box) in enumerate(boxes)
        Q_re_range, Q_im_range = box
        re_lo, re_hi = Float64.(Q_re_range)
        im_lo, im_hi = Float64.(Q_im_range)
        re_step = (re_hi - re_lo) / prescreen_nre
        im_step = (im_hi - im_lo) / prescreen_nim
        ncorners_x = prescreen_nre + 1
        ncorners_y = prescreen_nim + 1

        # Separate pre-screen cache, merged into the aggregate below.
        box_cache = Dict{ComplexF64,ComplexF64}()
        corners = Vector{ComplexF64}(undef, ncorners_x * ncorners_y)
        @inbounds for j in 0:prescreen_nim, i in 0:prescreen_nre
            corners[j*ncorners_x+i+1] =
                ComplexF64(re_lo + i * re_step, im_lo + j * im_step)
        end
        _bulk_eval_into_cache!(box_cache, f, corners; parallel=parallel)
        prescreen_evals_total += length(box_cache)

        ps_cells = Vector{AMRCell}(undef, prescreen_nre * prescreen_nim)
        @inbounds for j in 0:prescreen_nim-1, i in 0:prescreen_nre-1
            q_bl = corners[j*ncorners_x+i+1]
            q_br = corners[j*ncorners_x+(i+1)+1]
            q_tl = corners[(j+1)*ncorners_x+i+1]
            q_tr = corners[(j+1)*ncorners_x+(i+1)+1]
            ps_cells[j*prescreen_nre+i+1] =
                AMRCell(q_bl, q_br, q_tl, q_tr,
                    box_cache[q_bl], box_cache[q_br],
                    box_cache[q_tl], box_cache[q_tr])
        end

        activity = _check_box_activity(ps_cells, pole_magnitude_threshold)
        box_activity[b_idx] = activity

        # Keep pre-screen samples of skipped boxes too.
        for (q, d) in box_cache
            qd_aggregate[q] = d
        end

        if activity == NoActivity
            box_results[b_idx] = nothing
        else
            res = amr_scan(f, Q_re_range, Q_im_range;
                nre0=nre0, nim0=nim0, passes=passes,
                max_cells=max_cells,
                max_cells_action=max_cells_action,
                parallel=parallel)
            box_results[b_idx] = res
            append!(cells_aggregate, res.cells)
            for k in eachindex(res.Q)
                qd_aggregate[res.Q[k]] = res.Δ[k]
            end
        end
    end

    n = length(qd_aggregate)
    Q_all = Vector{ComplexF64}(undef, n)
    Δ_all = Vector{ComplexF64}(undef, n)
    for (k, (q, d)) in enumerate(qd_aggregate)
        Q_all[k] = q
        Δ_all[k] = d
    end

    return MultiBoxAMRResult(box_results, box_activity, cells_aggregate,
        Q_all, Δ_all, prescreen_evals_total)
end

"""
    as_amr_result(mbres::MultiBoxAMRResult) -> AMRResult

Wrap the aggregated cells/Q/Δ from a multi-box scan as a plain `AMRResult` so
it can be passed directly to `find_growth_rates(::AMRResult, tauk; ...)`.
"""
as_amr_result(mbres::MultiBoxAMRResult) =
    AMRResult(mbres.cells, mbres.Q, mbres.Δ,
        any(r -> r !== nothing && r.truncated, mbres.box_results))
