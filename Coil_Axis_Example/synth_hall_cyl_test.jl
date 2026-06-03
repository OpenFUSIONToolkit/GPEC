# Tests for synth_hall_cyl.jl using the idealised axisymmetric coil
# sparc_pf1u_axisymmetric.dat. The exact symmetries of a perfect circular
# loop give us crisp physical assertions that are not satisfied by the
# helical sparc_pf1u.dat — which is precisely why we use the axisymmetric
# file for regression testing.

include(joinpath(@__DIR__, "synth_hall_cyl.jl"))

using Test
using LinearAlgebra
using Statistics
using Printf

const AXI_COIL = joinpath(@__DIR__, "..", "examples", "Br_3D_example", "sparc_pf1u_axisymmetric.dat")
const CURRENT = 100.0

# Centre of the axisymmetric loop is at (0, 0, 0); R_coil ≈ 0.926 m.
# The coil is a 32-gon (33 points with closure), so all symmetry-tests
# have a discretisation noise floor at the ~1e-6 T / ~30 ppm level for
# probes inside the loop. Tolerances below reflect this.
const R_COIL = 0.926
const Z0     = 0.0
const Z_OFF  = 0.10    # off-midplane so B_R is non-trivial
const R_TEST = 0.4
const NPHI   = 64
# 360°/32 vertices of the underlying polygon — exactly maps the polygon
# onto itself, so a perfect implementation of Rz produces a field that
# matches tz=0 to floating-point precision.
const TZ_SYM_DEG = 360.0 / 32.0
# A non-symmetric tz used to confirm Rz is actually being applied (vs
# silently ignored). 0.5° produces a small but >> floating-point change.
const TZ_PROBE_DEG = 0.5

function ring(R, phi, Z, n)
    φs = collect(range(0, 2π; length=n + 1)[1:end-1])
    return fill(R, n), φs, fill(Z, n)
end

@testset "synth_hall_cyl (axisymmetric coil)" begin

# 1 — zero-pose axisymmetry baseline (limited by 32-gon discretisation)
@testset "1. zero-pose axisymmetry" begin
    R_p, phi_p, Z_p = ring(R_TEST, 0.0, Z_OFF, NPHI)
    out = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        current_A=CURRENT, R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    Bmag = sqrt(median(out.B_R)^2 + median(out.B_Z)^2)
    # Polygonal discretisation gives ~30 ppm n=32 modulation at this probe R.
    @test maximum(abs.(out.B_R   .- mean(out.B_R)))   < 1e-4 * Bmag
    @test maximum(abs.(out.B_Z   .- mean(out.B_Z)))   < 1e-4 * Bmag
    @test maximum(abs.(out.B_phi))                    < 1e-4 * Bmag
    @printf("    |B|≈%.3e T,  φ-variation < %.1e T (%.0f ppm)\n",
            Bmag, maximum(abs.(out.B_Z .- mean(out.B_Z))),
            1e6 * maximum(abs.(out.B_Z .- mean(out.B_Z))) / Bmag)
end

# 2 — tilt_z is wired through and only produces small (polygon-noise level)
#     changes for an axisymmetric coil. We check both: (a) a non-trivial
#     tz does produce *some* change (not silently dropped), and (b) the
#     change is bounded by discretisation noise (≪ |B|), confirming the
#     coil is acting axisymmetric.
@testset "2. tilt_z invariance (Rz is wired through)" begin
    R_p, phi_p, Z_p = ring(R_TEST, 0.0, Z_OFF, NPHI)
    out_0  = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        tz_deg=0.0,           current_A=CURRENT,
        R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    out_tz = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        tz_deg=TZ_PROBE_DEG,  current_A=CURRENT,
        R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    Bmag = sqrt(median(out_0.B_R)^2 + median(out_0.B_Z)^2)
    dB = maximum(abs.(out_0.B_Z .- out_tz.B_Z))
    # (a) non-zero change (>> float noise) → Rz is actually applied.
    @test dB > 1e-14
    # (b) change << |B| → coil is still acting axisymmetric.
    @test dB < 1e-3 * Bmag
    @printf("    tz=%.2f° produces ΔB=%.2e T (%.1f ppm of |B|=%.2e T)\n",
            TZ_PROBE_DEG, dB, 1e6 * dB / Bmag, Bmag)
end

# 3 — z-shift translational invariance
@testset "3. z-shift = probe-Z shift equivalence" begin
    dz = 0.005
    R_p, phi_p, Z_p = ring(R_TEST, 0.0, Z_OFF, NPHI)
    Z_shift = Z_p .+ dz                        # probes moved up
    out_probes = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        current_A=CURRENT, R_probes=R_p, phi_probes=phi_p, Z_probes=Z_shift)
    out_coil   = synth_hall_cyl(AXI_COIL, 0.0, 0.0, -dz;  # coil moved down
        current_A=CURRENT, R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    @test maximum(abs.(out_probes.B_R   .- out_coil.B_R))   < 1e-12
    @test maximum(abs.(out_probes.B_phi .- out_coil.B_phi)) < 1e-12
    @test maximum(abs.(out_probes.B_Z   .- out_coil.B_Z))   < 1e-12
    @printf("    max ΔB for dz=%.0f mm equivalence: %.3e T\n",
            dz * 1e3, maximum(abs.(out_probes.B_Z .- out_coil.B_Z)))
end

# 4 — x-shift introduces a pure n=1 azimuthal mode
@testset "4. x-shift -> pure n=1 in B_R(φ)" begin
    dx = 0.005
    R_p, phi_p, Z_p = ring(R_TEST, 0.0, Z_OFF, NPHI)  # off-midplane
    out = synth_hall_cyl(AXI_COIL, dx, 0.0, 0.0;
        current_A=CURRENT, R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    M = hcat(ones(NPHI), cos.(phi_p), sin.(phi_p))
    a0, a1, b1 = M \ out.B_R                       # least-squares fit
    residual = out.B_R .- (M * [a0, a1, b1])
    @test abs(a1) > 100 * abs(b1)                  # cos dominates sin
    @test maximum(abs.(residual)) < 0.05 * abs(a1) # n>=2 power < 5%
    @printf("    a1=%.3e, b1=%.3e (b1/a1=%.1e), residual peak %.1e T (%.2f%% of a1)\n",
            a1, b1, b1 / a1, maximum(abs.(residual)),
            100 * maximum(abs.(residual)) / abs(a1))
end

# 5 — tilt_x induces a sin(φ) sinusoid in B_R at off-midplane Z. We don't
#     try to recover the exact tilt angle here (that's a separate algorithm
#     in coil_centering_full.jl::magnetic_z_tilt); we just verify the n=1
#     sinusoidal mode appears, dominates, and scales with the input tilt.
@testset "5. tilt_x introduces n=1 sin(φ) mode in B_R" begin
    tx_true = 1.0  # degrees
    R_p, phi_p, Z_p = ring(R_TEST, 0.0, Z_OFF, NPHI)
    out0  = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        tx_deg=0.0,     current_A=CURRENT,
        R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    out_t = synth_hall_cyl(AXI_COIL, 0.0, 0.0, 0.0;
        tx_deg=tx_true, current_A=CURRENT,
        R_probes=R_p, phi_probes=phi_p, Z_probes=Z_p)
    # n=1 fit to (B_R_tilted − B_R_baseline)
    dBR = out_t.B_R .- out0.B_R
    M = hcat(ones(NPHI), cos.(phi_p), sin.(phi_p))
    a0, a1, b1 = M \ dBR
    residual = dBR .- (M * [a0, a1, b1])
    # tilt about x produces a sin(φ) mode (b1 ≠ 0), not cos (a1).
    @test abs(b1) > 100 * abs(a1)                      # sin dominates cos
    @test maximum(abs.(residual)) < 0.05 * abs(b1)     # n>=2 residual < 5%
    @test abs(b1) > 1e-9                               # actually non-trivial
    @printf("    Δ(B_R): a0=%.2e, a1=%.2e, b1=%.2e (a1/b1=%.1e), residual %.1e T (%.2f%% of b1)\n",
            a0, a1, b1, a1 / b1, maximum(abs.(residual)),
            100 * maximum(abs.(residual)) / abs(b1))
end

# 6 — centroid invariance under pure rotation
@testset "6. place_coil! rotates about centroid (not origin)" begin
    cs = read_coil_dat(AXI_COIL); cs.currents .= CURRENT
    coils = [cs]
    cx0, cy0, cz0 = coil_centroid(coils)
    place_coil!(coils, 0.0, 0.0, 0.0; tx_deg=5.0)  # rotation only
    cx1, cy1, cz1 = coil_centroid(coils)
    @test abs(cx1 - cx0) < 1e-12
    @test abs(cy1 - cy0) < 1e-12
    @test abs(cz1 - cz0) < 1e-12
    @printf("    Δcentroid after tilt_x=5°: (%.1e, %.1e, %.1e) m\n",
            cx1 - cx0, cy1 - cy0, cz1 - cz0)
end

end
