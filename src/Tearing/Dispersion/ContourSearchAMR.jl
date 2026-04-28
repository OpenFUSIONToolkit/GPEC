# ContourSearchAMR.jl
#
# Cell-based adaptive mesh refinement scanner of the complex Q plane. Port
# of the Fortran `dispersion_AMR_v2` (growthrates.f:367-533) and its helpers
# `get_or_compute_v2`, `check_cell_crossing_sub`, `subdivide_cell_sub`.
#
# Each `AMRCell` is an axis-aligned rectangle holding its 4 corner Q values
# and the corresponding Δ values evaluated by the user-supplied residual
# `f(Q)`. After `passes` refinement steps, every cell that brackets a zero
# in `Re(Δ)` or `Im(Δ)` has been subdivided into 4 quadrant children
# carrying 5 freshly evaluated midpoint Δ values.
#
# All evaluations of `f(Q)` are deduplicated through a `Dict{ComplexF64,
# ComplexF64}` hash cache so that adjacent cells sharing a corner (and
# adjacent refinement levels sharing an edge midpoint) cost only one
# evaluation. Replaces the Fortran's hand-rolled prime-multiplier hash with
# Julia's standard `Dict`, which already uses the right tricks for
# `ComplexF64` keys.
#
# Output: `AMRResult` holds the final list of `AMRCell`s (preserving the
# axis-aligned-rectangle structure that downstream marching-squares contour
# extraction in `GrowthRateExtraction.jl` exploits) plus the flat
# (Q::Vector, Δ::Vector) of all unique evaluations.

# Corner ordering matches the Fortran convention (growthrates.f:431-436):
# 1 = BL, 2 = BR, 3 = TL, 4 = TR.

"""
    AMRCell

A single axis-aligned-rectangle cell of an AMR scan. The four corner Q
values (`q_bl`, `q_br`, `q_tl`, `q_tr`) and corresponding residual values
(`d_bl`, `d_br`, `d_tl`, `d_tr`) are sufficient for marching-squares
contour extraction.
"""
struct AMRCell
    q_bl::ComplexF64; q_br::ComplexF64
    q_tl::ComplexF64; q_tr::ComplexF64
    d_bl::ComplexF64; d_br::ComplexF64
    d_tl::ComplexF64; d_tr::ComplexF64
end

"""
    AMRResult

Output of `amr_scan`.

| field    | meaning                                                       |
|----------|---------------------------------------------------------------|
| `cells`  | Final list of `AMRCell` after all refinement passes           |
| `Q`      | Flat `Vector{ComplexF64}` of every unique residual evaluation |
| `Δ`      | Corresponding `Vector{ComplexF64}` of residual values         |
"""
struct AMRResult
    cells::Vector{AMRCell}
    Q::Vector{ComplexF64}
    Δ::Vector{ComplexF64}
end

# Hash-cached residual evaluator. Returns the cached Δ value if `q` is
# already known, otherwise evaluates `f(q)`, stores it, and returns it.
@inline function _cached_eval!(cache::Dict{ComplexF64,ComplexF64},
                                f, q::ComplexF64)
    haskey(cache, q) && return cache[q]
    Δ = ComplexF64(f(q))
    cache[q] = Δ
    return Δ
end

# Parallel-friendly bulk filler: given a list of Q values, evaluates the
# residual at each one that isn't already in `cache` and stores the result.
# When `parallel=true` AND more than one Julia thread is available, the
# evaluations run via `@threads`; the cache is populated serially afterward
# to avoid Dict data races. Per-call evaluations of `f` are assumed to be
# thread-safe (true for `mc_fort(Q)` which constructs its own local state).
function _bulk_eval_into_cache!(cache::Dict{ComplexF64,ComplexF64}, f,
                                 qs::AbstractVector{ComplexF64};
                                 parallel::Bool)
    # First pass: partition `qs` into already-cached vs new. Keep uniqueness.
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
        Threads.@threads for k in eachindex(new_qs)
            new_vals[k] = ComplexF64(f(new_qs[k]))
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

# Sign-crossing test: does `vals` straddle zero? Used in both Re and Im
# directions on a cell's 4 corners (mirrors check_cell_crossing_sub).
@inline _crosses_zero(vals) = minimum(vals) * maximum(vals) <= 0

# Subdivide a parent cell into 4 quadrants, evaluating Δ at the 5
# midpoints (BM, TM, LM, RM, MM) via the hash cache.
function _subdivide_cell(parent::AMRCell,
                          cache::Dict{ComplexF64,ComplexF64}, f)
    q_bm = 0.5 * (parent.q_bl + parent.q_br)
    q_tm = 0.5 * (parent.q_tl + parent.q_tr)
    q_lm = 0.5 * (parent.q_bl + parent.q_tl)
    q_rm = 0.5 * (parent.q_br + parent.q_tr)
    q_mm = 0.25 * (parent.q_bl + parent.q_br + parent.q_tl + parent.q_tr)

    d_bm = _cached_eval!(cache, f, q_bm)
    d_tm = _cached_eval!(cache, f, q_tm)
    d_lm = _cached_eval!(cache, f, q_lm)
    d_rm = _cached_eval!(cache, f, q_rm)
    d_mm = _cached_eval!(cache, f, q_mm)

    return (
        AMRCell(parent.q_bl, q_bm, q_lm, q_mm,    # bottom-left quadrant
                parent.d_bl, d_bm, d_lm, d_mm),
        AMRCell(q_bm, parent.q_br, q_mm, q_rm,    # bottom-right quadrant
                d_bm, parent.d_br, d_mm, d_rm),
        AMRCell(q_lm, q_mm, parent.q_tl, q_tm,    # top-left quadrant
                d_lm, d_mm, parent.d_tl, d_tm),
        AMRCell(q_mm, q_rm, q_tm, parent.q_tr,    # top-right quadrant
                d_mm, d_rm, d_tm, parent.d_tr),
    )
end

"""
    amr_scan(f, Q_re_range, Q_im_range;
              nre0, nim0, passes,
              max_cells=10_000_000,
              max_cells_action=:error,
              snapshot_callback=nothing,
              parallel=Threads.nthreads() > 1) -> AMRResult

Adaptively refine a Q-plane scan of the residual `f(Q)`. An initial
`nre0 × nim0` axis-aligned grid of cells is built over `Q_re_range ×
Q_im_range` and `passes` rounds of refinement are applied. Each pass:

  1. flags any cell whose 4 corner residuals straddle zero in `Re(Δ)` or
     `Im(Δ)` (mirrors Fortran `check_cell_crossing_sub`);
  2. subdivides each flagged cell into 4 quadrant children, evaluating `f`
     at 5 new midpoints (mirrors Fortran `subdivide_cell_sub`);
  3. unflagged cells are kept unchanged.

All evaluations of `f` are deduplicated through a `Dict{ComplexF64,
ComplexF64}` hash cache so that adjacent cells share a single evaluation
per corner. The returned `AMRResult` carries both the final cell list (for
marching-squares contour extraction) and the flat list of all unique Q/Δ
evaluations.

# Keyword arguments

  - `nre0`, `nim0`   -- initial coarse-grid cell counts along each axis
  - `passes`         -- number of refinement passes
  - `max_cells`      -- safety cap on total cells; behavior on hit is set
    by `max_cells_action`
  - `max_cells_action` -- `:error` (raises) or `:warn_truncate` (logs a
    warning and returns the partial result). The latter is useful for
    convergence-vs-resolution studies where we deliberately push max_cells
    and want graceful degradation. Default `:error` preserves the prior
    safety-rail behaviour.
  - `snapshot_callback` -- if not `nothing`, a function called after each
    pass (and once for the initial grid, pass=0) with arguments
    `(pass::Int, cells::Vector{AMRCell}, cache::Dict{ComplexF64,ComplexF64})`.
    The callback receives live references — copy if you need persistence.
    Used by convergence studies to extract intermediate γ at each pass count.
  - `parallel`       -- evaluate `f` in parallel via `Threads.@threads` within
    each phase (initial grid + each refinement pass). Defaults to `true`
    when more than one Julia thread is available. Per-call evaluations of
    `f` must be thread-safe. Cache updates and cell-list construction stay
    serial, so the result is deterministic regardless of thread count.
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

    # ---- 1. coarse initial grid (nre0 × nim0 cells, (nre0+1)·(nim0+1) corners)
    # Collect every corner Q, evaluate in parallel, then build the cells using
    # cache lookups (no further evaluation happens in the build step).
    ncorners_x = nre0 + 1
    ncorners_y = nim0 + 1
    corners = Vector{ComplexF64}(undef, ncorners_x * ncorners_y)
    @inbounds for j in 0:nim0, i in 0:nre0
        corners[j * ncorners_x + i + 1] =
            ComplexF64(re_lo + i * re_step, im_lo + j * im_step)
    end
    _bulk_eval_into_cache!(cache, f, corners; parallel=parallel)

    cells = Vector{AMRCell}(undef, nre0 * nim0)
    @inbounds for j in 0:nim0-1, i in 0:nre0-1
        # Read corner Q values from the same `corners` array used to populate
        # the cache. Recomputing them with `x + re_step` here would differ in
        # the last floating-point bit from the cache keys, causing spurious
        # KeyErrors on lookup.
        q_bl = corners[j     * ncorners_x + i     + 1]
        q_br = corners[j     * ncorners_x + (i+1) + 1]
        q_tl = corners[(j+1) * ncorners_x + i     + 1]
        q_tr = corners[(j+1) * ncorners_x + (i+1) + 1]
        cells[j * nre0 + i + 1] = AMRCell(q_bl, q_br, q_tl, q_tr,
                                           cache[q_bl], cache[q_br],
                                           cache[q_tl], cache[q_tr])
    end

    # Snapshot the initial grid (pass 0) before any refinement.
    snapshot_callback === nothing || snapshot_callback(0, cells, cache)

    # ---- 2. refinement passes
    truncated = false   # set true when max_cells is hit and action == :warn_truncate
    for pass_idx in 1:passes
        truncated && break
        # Phase A: identify flagged parent cells and collect the midpoints we
        # need to evaluate. The 5 midpoints per parent (BM, TM, LM, RM, MM)
        # mirror _subdivide_cell's coordinates exactly.
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

        # Phase B: evaluate all new midpoints in parallel, fill the cache.
        _bulk_eval_into_cache!(cache, f, new_qs; parallel=parallel)

        # Phase C: build the refined cell list using cache lookups.
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
                d_bm = cache[q_bm]; d_tm = cache[q_tm]
                d_lm = cache[q_lm]; d_rm = cache[q_rm]
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
                    error("amr_scan: exceeded max_cells=$max_cells " *
                          "(currently $(length(new_cells))). Reduce " *
                          "`passes` or raise `max_cells`, or pass " *
                          "max_cells_action=:warn_truncate to truncate gracefully.")
                else  # :warn_truncate (validated at function entry)
                    @warn "amr_scan: max_cells=$max_cells reached at pass=$pass_idx cell=$idx/$(length(cells)); truncating refinement here and skipping remaining passes"
                    skip_remaining = true
                    truncated = true
                end
            end
        end
        cells = new_cells
        # Snapshot after this pass.
        snapshot_callback === nothing || snapshot_callback(pass_idx, cells, cache)
    end

    # ---- 3. flatten the cache into output Q/Δ vectors
    n = length(cache)
    Q = Vector{ComplexF64}(undef, n)
    Δ = Vector{ComplexF64}(undef, n)
    for (k, (q, d)) in enumerate(cache)
        Q[k] = q
        Δ[k] = d
    end

    return AMRResult(cells, Q, Δ)
end

# =============================================================================
# Multi-box AMR scan with pre-screen
# =============================================================================
#
# Motivation. A single wide AMR box (e.g. ω ∈ [-100, +100] kHz, γ ∈ [-25, +25])
# spends most of its evaluations on regions that contain neither roots nor
# poles. Splitting the same area into several smaller boxes and pre-screening
# each on a coarse 25×25 grid lets us skip refinement on inactive boxes
# entirely, while keeping full AMR sensitivity on the active ones.
#
# A box is flagged ACTIVE if any cell of its pre-screen grid satisfies AT LEAST
# ONE of:
#   - sign change in Re(Δ) across the cell's 4 corners (zero-isoline of Re(Δ)
#     crosses the cell — root candidate);
#   - sign change in Im(Δ) across the cell's 4 corners (zero-isoline of Im(Δ)
#     crosses the cell — root candidate);
#   - any corner with |Δ| ≥ `pole_magnitude_threshold` (likely pole inside or
#     near the box; sign-only criteria miss poles unless their fringe sign
#     change happens to land inside the pre-screen resolution).
#
# The pole-magnitude criterion is essential: a tight pole tucked inside one
# pre-screen cell can leave all four corners with the same large-magnitude sign
# (because Re(Δ) and Im(Δ) flip together as you orbit the pole, and at the
# corners we may sample the same lobe), so the sign-change tests would miss it.

"""
    BoxActivity

Why a box was retained or skipped by `multi_box_amr_scan`. `NoActivity` means
the pre-screen grid showed no zero-isoline crossings and no large-`|Δ|`
corners; the box is excluded from refinement. The other variants record which
criterion fired first.
"""
@enum BoxActivity NoActivity ReZeroCrossing ImZeroCrossing PoleMagnitude

# Pre-screen activity check: scan the pre-built cells and return the first
# satisfied criterion (or NoActivity if none fire). Designed for early exit so
# fully-quiet boxes cost just enough cell scans to confirm.
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

| field                | meaning                                                 |
|----------------------|---------------------------------------------------------|
| `box_results`        | per-box `AMRResult`, or `nothing` if box was skipped    |
| `box_activity`       | per-box `BoxActivity` enum                              |
| `cells`              | concatenated `AMRCell`s from all active boxes           |
| `Q`                  | union of all unique `Q` evaluations (active + skipped)  |
| `Δ`                  | corresponding `Δ` values                                |
| `prescreen_evals`    | total `f(Q)` evaluations spent on pre-screening         |

The aggregated `(cells, Q, Δ)` are suitable for direct consumption by
`find_growth_rates`. Pre-screen evaluations are still included in `Q`/`Δ` even
for skipped boxes, so any downstream pole-magnitude diagnostic that uses the
flat residual list sees the full sample.
"""
struct MultiBoxAMRResult
    box_results::Vector{Union{Nothing, AMRResult}}
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

Run `amr_scan` over multiple Q-plane boxes with a coarse pre-screen step that
skips inactive boxes entirely. The typical use case is the three-stripe ω-axis
scan for SLAYER coupled tearing dispersion:

    ω ∈ [-75, -25],  γ ∈ [-25, +25]   (left stripe)
    ω ∈ [-25, +25],  γ ∈ [-25, +25]   (centre stripe)
    ω ∈ [+25, +75],  γ ∈ [-25, +25]   (right stripe)

A single 150×50 box is wasteful when the dispersion is concentrated near a
narrow ω band; splitting into stripes and pre-screening lets the AMR effort
land on the active stripe.

# Pre-screen logic

Each box is sampled on a `prescreen_nre × prescreen_nim` corner grid (default
25×25, matching the typical AMR initial-grid resolution). A box is ACTIVE if
ANY pre-screen cell satisfies at least one criterion:

  1. sign change of `Re(Δ)` across the cell's 4 corners (zero-isoline of
     `Re(Δ)` crosses the cell — root candidate);
  2. sign change of `Im(Δ)` across the cell's 4 corners (zero-isoline of
     `Im(Δ)` crosses the cell — root candidate);
  3. any corner with `|Δ| ≥ pole_magnitude_threshold` (likely pole — the
     sign-only criteria miss poles whose fringe doesn't straddle a corner).

Active boxes get the full `amr_scan` treatment. Inactive boxes are dropped
(their `AMRResult` is `nothing`).

# Arguments

- `f`: residual function `Q::ComplexF64 → Δ::ComplexF64`. Must be thread-safe
  if `parallel=true`.
- `boxes`: vector of `(Q_re_range, Q_im_range)` tuples, one per box. Boxes
  may overlap or share boundaries; the aggregator deduplicates Q values.

# Required keyword

- `pole_magnitude_threshold`: activity threshold for `|Δ|`. A natural choice
  is `≈ |mean(Δ)|` from a baseline (or the same value used for adaptive
  pole_threshold in `find_growth_rates`).

# Optional keywords

- `prescreen_nre`, `prescreen_nim` (default 25 each): pre-screen grid
  resolution. Coarser misses small features; finer wastes evaluations on
  inactive boxes.
- `nre0, nim0, passes, max_cells, max_cells_action, parallel`: forwarded to
  each per-box `amr_scan` call. Defaults match `amr_scan`.

# Returns

A `MultiBoxAMRResult`. The aggregated `(cells, Q, Δ)` can be wrapped in an
`AMRResult` (helper `as_amr_result` below) for direct use with
`find_growth_rates`.

# Notes / TODO

- Each per-box `amr_scan` rebuilds its own cache, so the 25×25 pre-screen
  corners get re-evaluated by the AMR initial pass on active boxes
  (≈ 676 wasted evals per active box). A future refactor could thread a
  shared cache through `amr_scan`. For now the cost is small relative to
  the AMR refinement evals.
- Boxes that share a boundary line (e.g. the three ω-stripe layout above)
  duplicate ≈ `prescreen_nim+1` corner evaluations per shared edge. Also
  small.

# Example

```julia
boxes = [((-75.0, -25.0), (-25.0, 25.0)),
         ((-25.0,  25.0), (-25.0, 25.0)),
         (( 25.0,  75.0), (-25.0, 25.0))]
result = multi_box_amr_scan(f_residual, boxes;
                             pole_magnitude_threshold=1e-3,
                             prescreen_nre=25, prescreen_nim=25,
                             nre0=25, nim0=25, passes=4)
amr = AMRResult(result.cells, result.Q, result.Δ)
roots = find_growth_rates(amr, tauk; pole_threshold=1e-3)
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
    box_results = Vector{Union{Nothing, AMRResult}}(undef, n_boxes)
    box_activity = Vector{BoxActivity}(undef, n_boxes)
    prescreen_evals_total = 0

    # Aggregator: dedupe Q/Δ across all per-box caches and the pre-screen samples.
    # Using a Dict keyed by Q gives O(1) dedup and lets us merge results in any
    # order. We also collect cells (from active boxes only) for downstream
    # marching-squares extraction.
    qd_aggregate = Dict{ComplexF64, ComplexF64}()
    cells_aggregate = AMRCell[]

    for (b_idx, box) in enumerate(boxes)
        Q_re_range, Q_im_range = box
        re_lo, re_hi = Float64.(Q_re_range)
        im_lo, im_hi = Float64.(Q_im_range)
        re_step = (re_hi - re_lo) / prescreen_nre
        im_step = (im_hi - im_lo) / prescreen_nim
        ncorners_x = prescreen_nre + 1
        ncorners_y = prescreen_nim + 1

        # Pre-screen corners for THIS box. Local cache so we can both drive the
        # activity check and feed into the aggregate without polluting an
        # eventual per-box AMR cache.
        box_cache = Dict{ComplexF64, ComplexF64}()
        corners = Vector{ComplexF64}(undef, ncorners_x * ncorners_y)
        @inbounds for j in 0:prescreen_nim, i in 0:prescreen_nre
            corners[j * ncorners_x + i + 1] =
                ComplexF64(re_lo + i * re_step, im_lo + j * im_step)
        end
        _bulk_eval_into_cache!(box_cache, f, corners; parallel=parallel)
        prescreen_evals_total += length(box_cache)

        # Build pre-screen cells
        ps_cells = Vector{AMRCell}(undef, prescreen_nre * prescreen_nim)
        @inbounds for j in 0:prescreen_nim-1, i in 0:prescreen_nre-1
            q_bl = corners[j     * ncorners_x + i     + 1]
            q_br = corners[j     * ncorners_x + (i+1) + 1]
            q_tl = corners[(j+1) * ncorners_x + i     + 1]
            q_tr = corners[(j+1) * ncorners_x + (i+1) + 1]
            ps_cells[j * prescreen_nre + i + 1] =
                AMRCell(q_bl, q_br, q_tl, q_tr,
                        box_cache[q_bl], box_cache[q_br],
                        box_cache[q_tl], box_cache[q_tr])
        end

        # Activity check
        activity = _check_box_activity(ps_cells, pole_magnitude_threshold)
        box_activity[b_idx] = activity

        # Merge pre-screen evals into aggregate (for both active and skipped
        # boxes — diagnostics see all samples).
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

    # Flatten aggregator
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
    AMRResult(mbres.cells, mbres.Q, mbres.Δ)
