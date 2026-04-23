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
  - `max_cells`      -- safety cap on total cells (errors out if exceeded)
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
                  parallel::Bool=Threads.nthreads() > 1)
    nre0 >= 1 || throw(ArgumentError("amr_scan: nre0 must be ≥ 1"))
    nim0 >= 1 || throw(ArgumentError("amr_scan: nim0 must be ≥ 1"))
    passes >= 0 || throw(ArgumentError("amr_scan: passes must be ≥ 0"))

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

    # ---- 2. refinement passes
    for _ in 1:passes
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
        for (idx, cell) in enumerate(cells)
            if idx in flagged_set
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
            length(new_cells) > max_cells &&
                error("amr_scan: exceeded max_cells=$max_cells " *
                      "(currently $(length(new_cells))). Reduce " *
                      "`passes` or raise `max_cells`.")
        end
        cells = new_cells
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
