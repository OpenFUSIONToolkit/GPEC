# ═══════════════════════════════════════════════════════════════════════════════
# PnQuadCache.jl — Pre-computed constants and per-n cache for Gaussian
# quadrature in Pn_minus_half_2007!
#
# The 32-point Gauss quadrature uses two levels of pre-computation:
#
#  1. Node-level constants (_PN_TG02, _PN_WANUMR): depend only on the
#     quadrature rule, computed once at module load time as NTuples.
#
#  2. Per-n sinh/cosh cache (PnQuadEntry): depend on toroidal mode number n
#     but not on the Legendre argument s. Since n is fixed per vacuum run, we
#     cache them on first use and serve millions of subsequent calls from reads.
#
# Thread safety: Dict + SpinLock for storage, atomic last-used entry for
# lock-free fast path. Since n is constant within a vacuum run, the fast
# path (same n as last call) hits ~100% of the time.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Quadrature node constants (integration limits: xl=0, xu=5) ───────────────
# Uses gausslegendre(32) directly; this file is included before Kernel2D.jl
# and needs to be self-contained.
# _PN_TG02[ig]   = tg0² = (agaus + x[ig] * bgaus)²
# _PN_WANUMR[ig] = w[ig] * tg0 * exp(-tg0²)
const _PN_AGAUS = 2.5
const _PN_BGAUS = 2.5
const (_PN_TG02, _PN_WANUMR) = let
    x32, w32 = gausslegendre(32)
    tg0 = [_PN_AGAUS + x32[i] * _PN_BGAUS for i in 1:32]
    tg02   = NTuple{32,Float64}(t * t for t in tg0)
    wanumr = NTuple{32,Float64}(w32[i] * tg0[i] * exp(-tg0[i]^2) for i in 1:32)
    (tg02, wanumr)
end

# ── Per-n sinh/cosh cache ────────────────────────────────────────────────────

"""
    PnQuadEntry

Cached sinh/cosh values for the 32-point Gaussian quadrature at a given
toroidal mode number `n`. Each field is a 32-element vector indexed by
Gauss node `ig`.
"""
struct PnQuadEntry
    sinh::Vector{Float64}   # sinh(tg0²/(2n))
    cosh::Vector{Float64}   # cosh(tg0²/(2n))
    sinhp::Vector{Float64}  # sinh(tg0²/(2n+2))
    coshp::Vector{Float64}  # cosh(tg0²/(2n+2))

    # Pre-computed scale factors for the final Legendre function values.
    # These combine all factors that depend only on n (not on s):
    #   √2, n, √π, and the Gamma function Γ(1/2 - n) or Γ(-1/2 - n).
    # Caching them here avoids an O(n) product-formula computation on
    # every call to Pn_minus_half_2007! (which is called millions of times per run).
    #
    #   gauss_norm_n   = √2 / (n · √π · Γ(1/2 - n))      — scale for P^n_{-1/2}
    #   gauss_norm_np1 = √2 / ((n+1) · √π · Γ(-1/2 - n)) — scale for P^{n+1}_{-1/2}
    gauss_norm_n::Float64
    gauss_norm_np1::Float64
end

# Dict-based storage: works for any positive n, no upper limit.
const _PN_CACHE = Dict{Int,PnQuadEntry}()
const _PN_CACHE_LOCK = Threads.SpinLock()

# Fast-path: last-used n. Since n is constant within a vacuum run,
# this avoids Dict lookup + lock for ~100% of calls.
#
# Thread safety: _PN_LAST_ENTRY (plain Ref) is safe because the slow path
# always writes the entry *before* updating the atomic sentinel _PN_LAST_N.
# A reader that sees _PN_LAST_N == n is therefore guaranteed to see a valid
# entry — the atomic store acts as a release fence for the preceding write.
const _PN_LAST_N = Threads.Atomic{Int}(0)
const _PN_LAST_ENTRY = Ref{PnQuadEntry}(PnQuadEntry(Float64[], Float64[], Float64[], Float64[], 0.0, 0.0))

"""
    get_pn_quad_cache(n::Int) -> PnQuadEntry

Return cached sinh/cosh values for toroidal mode `n`, computing on first access.
Works for any `n ≥ 1` with no upper limit.

The fast path (same `n` as last call) is lock-free: one atomic read + comparison.
"""
@inline function get_pn_quad_cache(n::Int)
    @inbounds _PN_LAST_N[] == n && return @inbounds _PN_LAST_ENTRY[]
    return _get_pn_quad_cache_slow(n)
end

@noinline function _get_pn_quad_cache_slow(n::Int)
    entry = @lock _PN_CACHE_LOCK get!(() -> _make_pn_quad_entry(n), _PN_CACHE, n)
    @inbounds _PN_LAST_ENTRY[] = entry   # plain store (data) — must precede sentinel
    @inbounds _PN_LAST_N[] = n           # seq_cst store-release — makes data visible
    return entry
end

function _make_pn_quad_entry(n::Int)
    @assert n >= 1 "PnQuadEntry is only defined for n ≥ 1 (Γ(1/2 - n) diverges at n = 0)"
    inv_2n   = 1.0 / (2.0 * n)
    inv_2np2 = 1.0 / (2.0 * n + 2.0)
    sh  = Vector{Float64}(undef, 32)
    ch  = Vector{Float64}(undef, 32)
    shp = Vector{Float64}(undef, 32)
    chp = Vector{Float64}(undef, 32)
    @inbounds for ig in 1:32
        x  = _PN_TG02[ig] * inv_2n
        xp = _PN_TG02[ig] * inv_2np2
        sh[ig]  = sinh(x);  ch[ig]  = cosh(x)
        shp[ig] = sinh(xp); chp[ig] = cosh(xp)
    end

    # Compute the Gamma function Γ(1/2 - n) via the product formula.
    # Γ(1/2 - n) = √π / ∏_{i=1}^{n} (1/2 - i)  for n ≥ 1.
    # (n ≥ 1 is guaranteed: this cache is only populated when n·rhohat ≥ 0.1.)
    sqpi = sqrt(π)
    gamn = sqpi / prod(0.5 - i for i in 1:n)  # Γ(1/2 - n)
    gamp = -gamn / (n + 0.5)                   # Γ(-1/2 - n) = Γ(1/2 - (n+1)), via Γ(z-1) = Γ(z)/(z-1)

    # Combine into scale factors used in the final Legendre function assembly.
    # Pre-computing these once per n avoids repeating the Gamma product on every s-call.
    sqtwo = sqrt(2.0)
    gauss_norm_n   = sqtwo / (n * sqpi * gamn)
    gauss_norm_np1 = sqtwo / ((n + 1.0) * sqpi * gamp)

    return PnQuadEntry(sh, ch, shp, chp, gauss_norm_n, gauss_norm_np1)
end
