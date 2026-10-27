"""
    BiotSavart

Biot-Savart field calculation for 3D coil geometries.

Each conductor is a polyline of straight filament segments, and each segment contributes
its exact field (Hanson & Hirshman 2002, Phys. Plasmas 9, 4410, Eq. 8; SI units):

  B⃗ = (μ₀/4π) × I × nw × 2(Rᵢ + R_f) / (Rᵢ R_f ((Rᵢ + R_f)² - L²)) × (dl⃗ × R⃗ᵢ)

where μ₀/4π = 1e-7 [T⋅m/A], I is the current [A], nw is the winding multiplier,
dl⃗ is the segment vector of length L, and R⃗ᵢ, R⃗_f are the displacements from the
segment start and end to the observation point, with lengths Rᵢ and R_f.
"""

const MU0_OVER_4PI = 1e-7  # T⋅m/A

# Minimum squared distance [m²] from a segment (r ~ 1 μm) below which its contribution is
# skipped; avoids the singularity when an observation point lies on a conductor.
const BIOT_SAVART_MIN_DIST_SQ = 1e-12

"""
    _segment_field(rx, ry, rz, dlx, dly, dlz, len) -> (bx, by, bz)

Exact field of a straight unit-current filament segment of vector `dl⃗` and length `len`,
without the μ₀/4π prefactor, at displacement `r⃗` from the segment start
(Hanson & Hirshman 2002, Eq. 8). With ε = Rᵢ + R_f - L, the effective squared distance
ρ² = 2RᵢR_f ε / (Rᵢ + R_f) reduces to the perpendicular distance squared near the
segment, so the contribution is zero for ρ² below `BIOT_SAVART_MIN_DIST_SQ`.
"""
@inline function _segment_field(rx::Float64, ry::Float64, rz::Float64,
    dlx::Float64, dly::Float64, dlz::Float64, len::Float64)
    ri = sqrt(rx * rx + ry * ry + rz * rz)
    fx = rx - dlx
    fy = ry - dly
    fz = rz - dlz
    rf = sqrt(fx * fx + fy * fy + fz * fz)
    sum_r = ri + rf
    rho2_num = 2 * ri * rf * (sum_r - len)
    # Guard: zero contribution if observer is on conductor (avoids Inf/NaN)
    on_wire = rho2_num <= BIOT_SAVART_MIN_DIST_SQ * sum_r
    factor = ifelse(on_wire, 0.0, 4 * sum_r / (ifelse(on_wire, 1.0, rho2_num) * (sum_r + len)))
    return (dly * rz - dlz * ry) * factor, (dlz * rx - dlx * rz) * factor, (dlx * ry - dly * rx) * factor
end

"""
    PrecomputedSegments

Pre-computed segment vectors, lengths and start points for one conductor strand, stored
in contiguous arrays for cache-friendly access in the inner Biot-Savart loop.
"""
struct PrecomputedSegments
    dlx::Vector{Float64}
    dly::Vector{Float64}
    dlz::Vector{Float64}
    len::Vector{Float64}
    x0::Vector{Float64}
    y0::Vector{Float64}
    z0::Vector{Float64}
    prefactor::Float64       # MU0_OVER_4PI * current * nw
    nseg::Int
end

"""
    precompute_segments(cs, j, k, current_nw) → PrecomputedSegments

Build contiguous segment-vector, length and start-point arrays for strand `(j,k)` of coil set `cs`.
"""
function precompute_segments(cs::CoilSet, j::Int, k::Int, current_nw::Float64)
    nsec = cs.nsec
    nseg = nsec - 1
    dlx = Vector{Float64}(undef, nseg)
    dly = Vector{Float64}(undef, nseg)
    dlz = Vector{Float64}(undef, nseg)
    len = Vector{Float64}(undef, nseg)
    x0 = Vector{Float64}(undef, nseg)
    y0 = Vector{Float64}(undef, nseg)
    z0 = Vector{Float64}(undef, nseg)

    @inbounds for l in 1:nseg
        x1 = cs.x[j, k, l]
        x2 = cs.x[j, k, l+1]
        y1 = cs.y[j, k, l]
        y2 = cs.y[j, k, l+1]
        z1 = cs.z[j, k, l]
        z2 = cs.z[j, k, l+1]
        dlx[l] = x2 - x1
        dly[l] = y2 - y1
        dlz[l] = z2 - z1
        len[l] = sqrt(dlx[l]^2 + dly[l]^2 + dlz[l]^2)
        x0[l] = x1
        y0[l] = y1
        z0[l] = z1
    end

    return PrecomputedSegments(dlx, dly, dlz, len, x0, y0, z0,
        MU0_OVER_4PI * current_nw, nseg)
end

"""
    accumulate_strand_field!(Bx, By, Bz, i, obs_x, obs_y, obs_z, xs, ys, zs, current_nw)

Accumulate the Biot-Savart contribution from one conductor strand onto observation point `i`.

Each straight segment from point `l` to `l+1` contributes its exact filament field,
see `_segment_field`.

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

    @inbounds for l in 1:(nsec-1)
        # Segment vector dl⃗
        dlx = xs[l+1] - xs[l]
        dly = ys[l+1] - ys[l]
        dlz = zs[l+1] - zs[l]

        # Displacement r⃗ from segment start to observer
        rx = obs_x - xs[l]
        ry = obs_y - ys[l]
        rz = obs_z - zs[l]

        bx, by, bz = _segment_field(rx, ry, rz, dlx, dly, dlz, sqrt(dlx * dlx + dly * dly + dlz * dlz))
        Bx[i] += bx * prefactor
        By[i] += by * prefactor
        Bz[i] += bz * prefactor
    end
end

"""
    _accumulate_precomputed(ox, oy, oz, seg) -> (dbx, dby, dbz)

Inner kernel: accumulate Biot-Savart field from pre-computed segments.
Returns `(dbx, dby, dbz)` increments using the exact segment field.

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
        bx, by, bz = _segment_field(ox - seg.x0[l], oy - seg.y0[l], oz - seg.z0[l],
            seg.dlx[l], seg.dly[l], seg.dlz[l], seg.len[l])
        bx_acc += bx
        by_acc += by
        bz_acc += bz
    end

    return bx_acc * pf, by_acc * pf, bz_acc * pf
end

"""
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

Compute the total Biot-Savart field from all coil sets at every observation point.

Observation points are specified in cylindrical coordinates `(R, φ, Z)` and the
returned field components are also cylindrical.

Uses `Threads.@threads` over observation points -- each thread writes to a unique
index so no synchronization is required.

Performance: segment vectors, lengths and start points are pre-computed once outside the
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
    obs_x = Vector{Float64}(undef, nobs)
    obs_y = Vector{Float64}(undef, nobs)

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
            ox = obs_x[i]
            oy = obs_y[i]
            oz = obs_Z[i]
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
            B_R[i] = bx_total * cp + by_total * sp
            B_phi[i] = -bx_total * sp + by_total * cp
            B_Z[i] = bz_total
        end
    end
end

export accumulate_strand_field!, compute_biot_savart_boundary!
