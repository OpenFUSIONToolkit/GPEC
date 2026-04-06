"""
    BiotSavart

Biot-Savart field calculation for 3D coil geometries.

The fundamental formula is (SI units):

  dB⃗ = (μ₀/4π) × I × nw × (dl⃗ × r̂) / |r|²

where μ₀/4π = 1e-7 [T⋅m/A], I is the current [A], nw is the winding multiplier,
dl⃗ is the current element direction (segment vector), and r⃗ is the displacement
from the segment midpoint to the observation point.

Matches the Fortran `field_bs_psi` discretization in `field.F`.
"""

const MU0_OVER_4PI = 1e-7  # T⋅m/A

# Minimum squared distance [m²] to skip in single-point evaluation (r ~ 1 μm); avoids
# singularity when an observation point coincides with a conductor segment midpoint.
const BIOT_SAVART_MIN_DIST_SQ = 1e-12

"""
    PrecomputedSegments

Pre-computed segment vectors and midpoints for one conductor strand, stored
in contiguous arrays for cache-friendly access in the inner Biot-Savart loop.
"""
struct PrecomputedSegments
    dlx::Vector{Float64}
    dly::Vector{Float64}
    dlz::Vector{Float64}
    midx::Vector{Float64}
    midy::Vector{Float64}
    midz::Vector{Float64}
    prefactor::Float64       # MU0_OVER_4PI * current * nw
    nseg::Int
end

"""
    precompute_segments(cs, j, k, current_nw) → PrecomputedSegments

Build contiguous segment-vector and midpoint arrays for strand `(j,k)` of coil set `cs`.
"""
function precompute_segments(cs::CoilSet, j::Int, k::Int, current_nw::Float64)
    nsec = cs.nsec
    nseg = nsec - 1
    dlx  = Vector{Float64}(undef, nseg)
    dly  = Vector{Float64}(undef, nseg)
    dlz  = Vector{Float64}(undef, nseg)
    midx = Vector{Float64}(undef, nseg)
    midy = Vector{Float64}(undef, nseg)
    midz = Vector{Float64}(undef, nseg)

    @inbounds for l in 1:nseg
        x1 = cs.x[j, k, l];   x2 = cs.x[j, k, l+1]
        y1 = cs.y[j, k, l];   y2 = cs.y[j, k, l+1]
        z1 = cs.z[j, k, l];   z2 = cs.z[j, k, l+1]
        dlx[l]  = x2 - x1
        dly[l]  = y2 - y1
        dlz[l]  = z2 - z1
        midx[l] = (x1 + x2) * 0.5
        midy[l] = (y1 + y2) * 0.5
        midz[l] = (z1 + z2) * 0.5
    end

    return PrecomputedSegments(dlx, dly, dlz, midx, midy, midz,
                               MU0_OVER_4PI * current_nw, nseg)
end

"""
    accumulate_strand_field!(Bx, By, Bz, i, obs_x, obs_y, obs_z, xs, ys, zs, current_nw)

Accumulate the Biot-Savart contribution from one conductor strand onto observation point `i`.

Uses midpoint-rule integration: each segment from point `l` to `l+1` contributes:
  dB = μ₀/(4π) × current_nw × (dl⃗ × r⃗) / |r|³
where `r⃗` is the displacement from the segment midpoint to the observer.

## Arguments
- `Bx`, `By`, `Bz`: pre-allocated output arrays (Cartesian field, Tesla); updated in-place
- `i`: observation point index
- `obs_x/y/z`: observer Cartesian coordinates [m]
- `xs`, `ys`, `zs`: strand point coordinates [nsec], Cartesian [m]
- `current_nw`: effective current = I × nw [A⋅turns]
"""
function accumulate_strand_field!(
    Bx::AbstractVector{Float64},
    By::AbstractVector{Float64},
    Bz::AbstractVector{Float64},
    i::Int,
    obs_x::Float64, obs_y::Float64, obs_z::Float64,
    xs::AbstractVector{Float64},
    ys::AbstractVector{Float64},
    zs::AbstractVector{Float64},
    current_nw::Float64
)
    nsec = length(xs)
    prefactor = MU0_OVER_4PI * current_nw

    @inbounds for l in 1:(nsec - 1)
        # Segment vector dl⃗
        dlx = xs[l+1] - xs[l]
        dly = ys[l+1] - ys[l]
        dlz = zs[l+1] - zs[l]

        # Displacement r⃗ from segment midpoint to observer
        rx = obs_x - (xs[l] + xs[l+1]) * 0.5
        ry = obs_y - (ys[l] + ys[l+1]) * 0.5
        rz = obs_z - (zs[l] + zs[l+1]) * 0.5

        r2  = rx*rx + ry*ry + rz*rz
        r2 < BIOT_SAVART_MIN_DIST_SQ && continue  # avoid singularity (observer on conductor)
        inv_r3 = prefactor / (r2 * sqrt(r2))

        # dB = prefactor × (dl⃗ × r⃗) / |r|³  [cross product]
        Bx[i] += (dly*rz - dlz*ry) * inv_r3
        By[i] += (dlz*rx - dlx*rz) * inv_r3
        Bz[i] += (dlx*ry - dly*rx) * inv_r3
    end
end

"""
    _accumulate_precomputed(ox, oy, oz, seg) -> (dbx, dby, dbz)

Inner kernel: accumulate Biot-Savart field from pre-computed segments.
Returns `(dbx, dby, dbz)` increments. Matches Fortran `field_bs_psi` midpoint rule.

Operates on `PrecomputedSegments` with contiguous arrays for maximum
cache efficiency.
"""
@inline function _accumulate_precomputed(
    ox::Float64, oy::Float64, oz::Float64,
    seg::PrecomputedSegments
)
    bx_acc = 0.0
    by_acc = 0.0
    bz_acc = 0.0
    pf = seg.prefactor

    @inbounds @simd for l in 1:seg.nseg
        rx = ox - seg.midx[l]
        ry = oy - seg.midy[l]
        rz = oz - seg.midz[l]

        r2 = rx*rx + ry*ry + rz*rz
        # Guard: zero contribution if observer is on conductor (avoids Inf/NaN)
        safe_r2 = ifelse(r2 < BIOT_SAVART_MIN_DIST_SQ, one(r2), r2)
        scale   = ifelse(r2 < BIOT_SAVART_MIN_DIST_SQ, zero(pf), pf)
        inv_r3  = scale / (safe_r2 * sqrt(safe_r2))

        bx_acc += (seg.dly[l]*rz - seg.dlz[l]*ry) * inv_r3
        by_acc += (seg.dlz[l]*rx - seg.dlx[l]*rz) * inv_r3
        bz_acc += (seg.dlx[l]*ry - seg.dly[l]*rx) * inv_r3
    end

    return bx_acc, by_acc, bz_acc
end

"""
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

Compute the total Biot-Savart field from all coil sets at every observation point.

Observation points are specified in cylindrical coordinates `(R, φ, Z)` and the
returned field components are also cylindrical.

Uses `Threads.@threads` over observation points -- each thread writes to a unique
index so no synchronization is required.

Performance: segment vectors and midpoints are pre-computed once outside the
observation-point loop, then accessed from contiguous arrays in the inner kernel.

## Arguments
- `B_R`, `B_phi`, `B_Z`: output field arrays [Tesla], length `nobs`; overwritten
- `obs_R`, `obs_phi`, `obs_Z`: observation point cylindrical coordinates [m, rad, m]
- `coil_sets`: conductor geometry and current data
"""
function compute_biot_savart_boundary!(
    B_R::AbstractVector{Float64},
    B_phi::AbstractVector{Float64},
    B_Z::AbstractVector{Float64},
    obs_R::AbstractVector{Float64},
    obs_phi::AbstractVector{Float64},
    obs_Z::AbstractVector{Float64},
    coil_sets::Vector{CoilSet}
)
    nobs = length(obs_R)
    @assert length(obs_phi) == nobs && length(obs_Z) == nobs
    @assert length(B_R) == nobs && length(B_phi) == nobs && length(B_Z) == nobs

    # Pre-compute segment geometry for all active strands (done once, O(total_segments))
    all_segments = PrecomputedSegments[]
    for cs in coil_sets
        for j in 1:cs.ncoil
            I_nw = cs.currents[j] * cs.nw
            iszero(I_nw) && continue
            for k in 1:cs.s
                push!(all_segments, precompute_segments(cs, j, k, I_nw))
            end
        end
    end

    # Pre-compute sincos for cylindrical ↔ Cartesian conversion (avoids redundant trig)
    cos_phi = Vector{Float64}(undef, nobs)
    sin_phi = Vector{Float64}(undef, nobs)
    obs_x   = Vector{Float64}(undef, nobs)
    obs_y   = Vector{Float64}(undef, nobs)

    @inbounds for i in 1:nobs
        sp, cp = sincos(obs_phi[i])
        sin_phi[i] = sp
        cos_phi[i] = cp
        obs_x[i] = obs_R[i] * cp
        obs_y[i] = obs_R[i] * sp
    end

    # Each thread writes exclusively to its own index i — no synchronization needed
    Threads.@threads for i in 1:nobs
        @inbounds begin
            ox = obs_x[i]; oy = obs_y[i]; oz = obs_Z[i]
            bx_total = 0.0
            by_total = 0.0
            bz_total = 0.0

            for seg in all_segments
                dbx, dby, dbz = _accumulate_precomputed(ox, oy, oz, seg)
                bx_total += dbx
                by_total += dby
                bz_total += dbz
            end

            # Convert Cartesian → cylindrical in-place
            cp = cos_phi[i]
            sp = sin_phi[i]
            B_R[i]   =  bx_total * cp + by_total * sp
            B_phi[i] = -bx_total * sp + by_total * cp
            B_Z[i]   =  bz_total
        end
    end
end

export accumulate_strand_field!, compute_biot_savart_boundary!
