# ═══════════════════════════════════════════════════════════════════════════════
# PnQuadCache.jl — Pre-computed constants and per-n cache for Gaussian
# quadrature in Pn_minus_half_2007!
#
# The 32-point Gauss quadrature uses two levels of pre-computation:
#
#  1. Node-level constants (_PN_TG02, _PN_WANUMR): depend only on the
#     quadrature rule, computed once at module load time as NTuples.
#
#  2. Per-n sinh/cosh cache: depend on toroidal mode number n but not on
#     the Legendre argument s. Since n is fixed per vacuum run, we cache
#     them on first use and serve ~3.7M subsequent calls from array reads.
#
# Thread safety: per-n Atomic{Bool} flag + SpinLock (double-checked locking).
# Fast path (cache hit) is lock-free: one atomic read + array index.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Quadrature node constants (integration limits: xl=0, xu=5) ───────────────
# Uses gausslegendre(32) directly so this file is self-contained
# (included before Kernel2D.jl where GL32 is defined).
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
const _PN_QUAD_MAX_N = 64

# Pre-allocated storage indexed by toroidal mode number n ∈ 1:_PN_QUAD_MAX_N.
# Each inner Vector{Float64} has 32 elements (one per Gauss node).
# Stores sinh and cosh values for mode n and n+1 quadrature arguments.
const _PN_QUAD_SINH  = [Vector{Float64}(undef, 32) for _ in 1:_PN_QUAD_MAX_N]
const _PN_QUAD_COSH  = [Vector{Float64}(undef, 32) for _ in 1:_PN_QUAD_MAX_N]
const _PN_QUAD_SINHP = [Vector{Float64}(undef, 32) for _ in 1:_PN_QUAD_MAX_N]
const _PN_QUAD_COSHP = [Vector{Float64}(undef, 32) for _ in 1:_PN_QUAD_MAX_N]

# Per-n readiness flag (atomic for lock-free fast path) and shared lock for population
const _PN_QUAD_READY = [Threads.Atomic{Bool}(false) for _ in 1:_PN_QUAD_MAX_N]
const _PN_QUAD_LOCK  = Threads.SpinLock()

"""
    ensure_pn_quad_cache!(n::Int)

Ensure that the Gaussian quadrature sinh/cosh cache is populated for toroidal mode `n`.
After this call, `_PN_QUAD_SINH[n]`, `_PN_QUAD_COSH[n]`, `_PN_QUAD_SINHP[n]`,
and `_PN_QUAD_COSHP[n]` contain valid pre-computed values.

Thread-safe via atomic flags with double-checked locking. The fast path (cache hit)
is lock-free and costs ~1ns (one atomic Bool read).
"""
@inline function ensure_pn_quad_cache!(n::Int)
    @boundscheck 1 <= n <= _PN_QUAD_MAX_N || throw(
        DomainError(n, "toroidal mode number n must be in 1:$_PN_QUAD_MAX_N"))
    @inbounds _PN_QUAD_READY[n][] && return nothing
    _populate_pn_quad_cache!(n)
    return nothing
end

@noinline function _populate_pn_quad_cache!(n::Int)
    @lock _PN_QUAD_LOCK begin
        # Double-check after acquiring lock (another thread may have populated)
        @inbounds _PN_QUAD_READY[n][] && return nothing

        inv_2n   = 1.0 / (2.0 * n)
        inv_2np2 = 1.0 / (2.0 * n + 2.0)

        sinh_v  = @inbounds _PN_QUAD_SINH[n]
        cosh_v  = @inbounds _PN_QUAD_COSH[n]
        sinhp_v = @inbounds _PN_QUAD_SINHP[n]
        coshp_v = @inbounds _PN_QUAD_COSHP[n]

        @inbounds for ig in 1:32
            tg1  = _PN_TG02[ig] * inv_2n
            tg1p = _PN_TG02[ig] * inv_2np2

            sinh_v[ig]  = sinh(tg1)
            cosh_v[ig]  = cosh(tg1)
            sinhp_v[ig] = sinh(tg1p)
            coshp_v[ig] = cosh(tg1p)
        end

        # Atomic store acts as release barrier — all writes above are visible
        # to any thread that subsequently reads true from this flag.
        @inbounds _PN_QUAD_READY[n][] = true
    end
    return nothing
end
