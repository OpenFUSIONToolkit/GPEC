"""
Jacobian-invariance tests for the PowerNorm building blocks, on a single flux
surface of the DIIID-like example. The PowerNorm module's ξ→Φ transformation
composes three pieces: an angle-map convmat, the √weight operator `sqrtamat`,
and a jacobian-dependent diagonal rescaling `T = diag(m − n·q)`. Only the first
two are expected to be jacobian-invariant at the norm level (T is an intrinsically
jac-specific operator; it is the eigenvalue spectrum of M†·W·M that is invariant,
not ‖Φ‖). This script validates each invariant piece against a numerical
precision floor.

  Test 1 — flux-surface area A = ∫(J·|∇ψ|)dθ. Pure geometric invariant; its spread
           across jacobians sets `floor_tol` for every subsequent test.

  Test 2 — ∮|f(η)|² dA for a physical scalar f defined on the shared geometric
           angle η. Validates that each jac's spline evaluation of J·|∇ψ| is
           consistent across jacobians.

  Test 3 — angle-map convmat correctness. Build a scalar field from low-order
           hamada plane waves, compare `convmat·c_hamada` to the direct FT of
           the field on each jac's θ-grid. Tests the mode→mode transformation
           in isolation.

  Test 4 — sqrtamat invariance. For a narrow-band b_fft_hamada (same physical
           scalar field as in Test 3, just expressed as mode coefficients),
           angle-map to each jac via convmat, apply that jac's sqrtamat, and
           compare ‖sqrtamat·b_fft‖² across jacs. This is the direct jac-
           invariance test for the √weight operator.

Usage:
    julia --project=. scripts/test_power_norm_invariance.jl
"""

using TOML
using LinearAlgebra
using Printf
using Roots
using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const EQ = GPE.Equilibrium
const FFS = GPE.ForceFreeStates
const PE = GPE.PerturbedEquilibrium
const FT = GPE.Utilities.FourierTransforms
using FastInterpolations: DerivOp

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
const PSIHIGH = 0.994
const N_TOR = 3

# Basis m-band for convmat and sqrtamat. Deliberately much wider than the narrow
# physics band used for b_fft (see `phys_amps` in main) so that angle-map truncation
# in the mode basis is not the dominant error.
const MLOW  = -128
const MHIGH = 128

# Dense θ-grid. Comfortably above 2·MHIGH to avoid aliasing in mode-basis expansions.
const MTHETA = 1024

const JACS = ("hamada", "pest", "park", "boozer")
const JAC_REF = "hamada"
const JACS_OUT = ("pest", "park", "boozer")

"""Load equilibrium with specified jac_type at PSIHIGH; skip ODE/vacuum stages."""
function build_equil(jac::String)
    inputs = TOML.parsefile(joinpath(EXAMPLE_DIR, "gpec.toml"))
    eq_dict = inputs["Equilibrium"]
    eq_dict["jac_type"] = jac
    eq_dict["psihigh"] = PSIHIGH
    eq_dict["force_termination"] = true
    return EQ.setup_equilibrium(EQ.EquilibriumConfig(eq_dict, EXAMPLE_DIR))
end

"""
    angle_map(equil_src, equil_dst, psi, theta_src) -> Vector

Map a set of source-jacobian poloidal angles θ_src to the corresponding
destination-jacobian angles θ_dst on the same physical flux surface at `psi`.

Both jacobians share the geometric angle η(ψ, θ) = 2π·(θ + rzphi_offset(ψ, θ)),
so θ_dst is found by Newton root-find on
    F(θ_dst) = 2π·(θ_dst + offset_dst(ψ, θ_dst)) − η(ψ, θ_src) = 0.
"""
function angle_map(equil_src, equil_dst, psi::Float64, theta_src::AbstractVector{Float64})
    theta_dst = similar(theta_src)
    hint_src = (Ref(1), Ref(1))
    hint_dst = (Ref(1), Ref(1))
    for (k, ts) in enumerate(theta_src)
        η = 2π * (ts + equil_src.rzphi_offset((psi, ts); hint=hint_src))
        f_and_fp = θ -> begin
            θm = mod(θ, 1.0)
            val = 2π * (θm + equil_dst.rzphi_offset((psi, θm); hint=hint_dst)) - η
            val = mod(val + π, 2π) - π   # unwrap to [-π, π]
            dval = 2π * (1.0 + equil_dst.rzphi_offset((psi, θm); deriv=DerivOp(0, 1), hint=hint_dst))
            return (val, val / dval)
        end
        theta_dst[k] = mod(find_zero(f_and_fp, ts, Roots.Newton()), 1.0)
    end
    return theta_dst
end

"""
    build_convmat(equil_dst, equil_src, psi, mpert, mlow, mtheta) -> Matrix

Pure θ-relabeling matrix: c_dst = convmat · c_src, where c_src are mode coefficients
of a scalar field in the src-jac Fourier basis and c_dst are its coefficients in the
dst-jac Fourier basis. Column i is the dst-grid FT of the src-basis plane wave
exp(−i·2π·m_i·θ_src), evaluated at θ_src = θ_src(θ_dst) from the angle map.
"""
function build_convmat(equil_dst, equil_src, psi::Float64,
    mpert::Int, mlow::Int, mtheta::Int)
    theta_dst = collect((0:(mtheta - 1)) ./ mtheta)
    theta_src = angle_map(equil_dst, equil_src, psi, theta_dst)
    m_vals = collect(mlow:(mlow + mpert - 1))
    ft_dst = FT.FourierTransform(mtheta, mpert, mlow)
    convmat = zeros(ComplexF64, mpert, mpert)
    col = zeros(ComplexF64, mtheta)
    for i in 1:mpert
        @. col = cis(-2π * m_vals[i] * theta_src)
        convmat[:, i] .= ft_dst(col) ./ mtheta
    end
    return convmat
end

"""
    jac_delpsi_at(equil, psi, θ, hint2d) -> Float64

Evaluate J·|∇ψ| at a single (ψ, θ). Mirrors `compute_surface_area_local` in
`src/ForceFreeStates/PowerNorm.jl:58` exactly so Test 2 shares the same numerical
precision floor as Test 1.
"""
function jac_delpsi_at(equil, psi::Float64, θ::Float64, hint2d)
    r2 = equil.rzphi_rsquared((psi, θ); hint=hint2d)
    jac = equil.rzphi_jac((psi, θ); hint=hint2d)
    deta = equil.rzphi_offset((psi, θ); hint=hint2d)
    deta_y = equil.rzphi_offset((psi, θ); deriv=DerivOp(0, 1), hint=hint2d)
    r2_y = equil.rzphi_rsquared((psi, θ); deriv=DerivOp(0, 1), hint=hint2d)

    rfac = sqrt(abs(r2))
    eta = 2π * (θ + deta)
    r = equil.ro + rfac * cos(eta)

    w11 = (1.0 + deta_y) * (2π)^2 * rfac * r / jac
    w12 = -r2_y * π * r / (rfac * jac)
    delpsi = sqrt(w11^2 + w12^2)
    return jac * delpsi
end

"""
    reconstruct_theta(b_fft, mlow, mtheta) -> Vector{ComplexF64}

Reconstruct b(θ) from mode coefficients via the PowerNorm backward FT convention
    b(θ_j) = (1/N) Σ_m b_fft[m] · exp(−i·2π·m·θ_j)
on the equispaced θ-grid θ_j = (j−1)/N for j ∈ 1..N. Used for trapezoidal cross-
checks against the mode-space sqrtamat computation.
"""
function reconstruct_theta(b_fft::AbstractVector{ComplexF64}, mlow::Int, mtheta::Int)
    mpert = length(b_fft)
    m_vals = collect(mlow:(mlow + mpert - 1))
    b_theta = zeros(ComplexF64, mtheta)
    for j in 1:mtheta
        θ = (j - 1) / mtheta
        s = zero(ComplexF64)
        for i in 1:mpert
            s += b_fft[i] * cis(-2π * m_vals[i] * θ)
        end
        b_theta[j] = s / mtheta
    end
    return b_theta
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

"""
    test_area_invariance(equils, psi, mtheta) -> Float64

Test 1: flux-surface area A = ∫(J·|∇ψ|)dθ. Physical invariant; returns the max
relative spread across jac_out ∈ {pest, park, boozer} as the precision floor.

Prints both FFS.compute_surface_area_local and PE.compute_surface_area as a
cross-check that the two implementations of Fortran gpout.f:568 agree.
"""
function test_area_invariance(equils, psi::Float64, mtheta::Int)
    println("=== Test 1: area A = ∫(J·|∇ψ|)dθ invariance (precision floor) ===")
    println("A is GPEC's standard area (Fortran gpout.f:568); physical m² is 2π·A.")
    println("FFS  = FFS.compute_surface_area_local(equil, ψ, mtheta)")
    println("PE   = PerturbedEquilibrium.compute_surface_area(equil, ψ)")
    println("diff = FFS − PE (should be ≈ 0; same formula, different mtheta convention)\n")

    A_ref = FFS.compute_surface_area_local(equils[JAC_REF], psi, mtheta)
    @printf("%-8s | %-18s | %-18s | diff         | 2π·A (m²)    | A/A_ref − 1\n",
            "jac", "FFS A", "PE A")
    println(repeat("-", 100))

    max_rel = 0.0
    for jac in JACS
        A = FFS.compute_surface_area_local(equils[jac], psi, mtheta)
        A_PE = PE.compute_surface_area(equils[jac], psi)
        rel = A / A_ref - 1
        max_rel = max(max_rel, abs(rel))
        @printf("%-8s | %.12e | %.12e | %+.3e    | %.6e | %+.3e\n",
                jac, A, A_PE, A - A_PE, 2π * A, rel)
    end
    @printf("\nFloor tolerance = max |A/A_ref − 1| across jacobians = %.3e\n\n", max_rel)
    return max_rel
end

"""
    test_geometric_scalar_invariant(equils, psi, mtheta, floor_tol) -> Float64

Test 2: ∮|f(η)|² dA = 2π·∫|f|²·J·|∇ψ|·dθ where f is a smooth physical field
defined on the shared geometric angle η. Tests that the J·|∇ψ| weight and
quadrature are applied consistently across jacs; residual should track the floor.
"""
function test_geometric_scalar_invariant(equils, psi::Float64, mtheta::Int, floor_tol::Float64)
    println("=== Test 2: ∮|f(η)|² dA invariance ===")
    println("f(η) defined on the shared geometric angle; residual should match the floor.\n")

    f(η) = 1.0 + 0.5 * cos(η) + 0.3 * cos(2η) + 0.2 * sin(3η)

    function integral_on(equil)
        hint2d = (Ref(1), Ref(1))
        total = 0.0
        for j in 0:(mtheta - 1)
            θ = j / mtheta
            η = 2π * (θ + equil.rzphi_offset((psi, θ); hint=hint2d))
            total += f(η)^2 * jac_delpsi_at(equil, psi, θ, hint2d) / mtheta
        end
        return 2π * total
    end

    I_ref = integral_on(equils[JAC_REF])
    @printf("%-8s | %-18s | I/I_ref − 1 | ratio to floor\n", "jac", "I")
    println(repeat("-", 70))
    @printf("%-8s | %.12e | %+.3e   | --\n", JAC_REF, I_ref, 0.0)

    max_rel = 0.0
    for jac in JACS_OUT
        I = integral_on(equils[jac])
        rel = I / I_ref - 1
        max_rel = max(max_rel, abs(rel))
        ratio = abs(rel) / floor_tol
        @printf("%-8s | %.12e | %+.3e   | %.2f×\n", jac, I, rel, ratio)
    end

    ratio = max_rel / floor_tol
    verdict = ratio < 10 ? "PASS" : "FAIL"
    @printf("\n%s: max |I/I_ref − 1| = %.3e (%.2f× floor)\n\n", verdict, max_rel, ratio)
    return max_rel
end

"""
    test_convmat(equils, psi, mpert, mlow, mtheta)

Test 3: Pure angle-map convmat correctness. Define a scalar field via its hamada
mode spectrum, then for each jac verify
    convmat · c_hamada  ==  FT(field evaluated on jac's θ-grid)
to machine precision. Isolates the mode→mode transformation.
"""
function test_convmat(equils, psi::Float64, mpert::Int, mlow::Int, mtheta::Int)
    println("=== Test 3: angle-map convmat correctness ===")
    println("rel err = ‖convmat·c_ref − c_direct‖ / ‖c_direct‖\n")

    m_support = [-1, 0, 1, 2, 3]
    amps = ComplexF64[0.5+0.1im, 1.0+0.0im, -0.3+0.4im, 0.2-0.15im, 0.1+0.05im]

    equil_ref = equils[JAC_REF]
    theta_grid = collect((0:(mtheta - 1)) ./ mtheta)

    # Field sampled on ref θ-grid, forward-transformed to get c_ref
    field_ref = zeros(ComplexF64, mtheta)
    for (a, m_k) in zip(amps, m_support)
        @. field_ref += a * cis(-2π * m_k * theta_grid)
    end
    c_ref = FT.FourierTransform(mtheta, mpert, mlow)(field_ref)

    @printf("%-8s | %-14s | %-14s | %-11s | status\n",
            "jac", "‖c_pred‖", "‖c_direct‖", "rel err")
    println(repeat("-", 70))

    for jac in JACS
        equil_out = equils[jac]

        # Sample the same physical field on jac's θ-grid via angle map to ref
        theta_ref_at_out = angle_map(equil_out, equil_ref, psi, theta_grid)
        field_out = zeros(ComplexF64, mtheta)
        for (a, m_k) in zip(amps, m_support)
            @. field_out += a * cis(-2π * m_k * theta_ref_at_out)
        end
        c_direct = FT.FourierTransform(mtheta, mpert, mlow)(field_out)
        c_pred = build_convmat(equil_out, equil_ref, psi, mpert, mlow, mtheta) * c_ref

        rel_err = norm(c_pred - c_direct) / max(norm(c_direct), eps())
        status = rel_err < 1e-10 ? "PASS" : rel_err < 1e-6 ? "ok" : "FAIL"
        @printf("%-8s | %.6e | %.6e | %.3e | %s\n",
                jac, norm(c_pred), norm(c_direct), rel_err, status)
    end
    println()
end

"""
    test_sqrtamat_invariance(equils, psi, mpert, mlow, mtheta, b_fft_ref, floor_tol) -> Float64

Test 4: sqrtamat invariance. For a physical scalar field with mode coefficients
`b_fft_ref` in the reference jacobian's Fourier basis:

  1. Angle-map to each jac: b_fft_jac = convmat_{jac←ref} · b_fft_ref.
  2. Compute  w_jac = sqrtamat_jac · b_fft_jac  in that jac's mode basis.
  3. Check ‖w_jac‖² is invariant across jacs.

`sqrtamat` is defined so that ‖sqrtamat · b_fft‖² = N² · ∫|b|²·J·|∇ψ|·dθ when b is
band-limited within [mlow, mhigh] (N = mtheta; the N² arises from the forward-FT
convention carrying no 1/N factor while the backward FT carries 1/N). A cross-check
on the reference jacobian verifies this identity numerically.
"""
function test_sqrtamat_invariance(equils, psi::Float64, mpert::Int, mlow::Int, mtheta::Int,
    b_fft_ref::Vector{ComplexF64}, floor_tol::Float64)
    println("=== Test 4: sqrtamat invariance — ‖sqrtamat · b_fft‖² across jacobians ===\n")

    equil_ref = equils[JAC_REF]
    sqrtamat_ref = FFS.compute_sqrtamat(equil_ref, psi, FT.FourierTransform(mtheta, mpert, mlow))
    I_ref = norm(sqrtamat_ref * b_fft_ref)^2

    # Direct-quadrature cross-check on the reference jac
    b_theta_ref = reconstruct_theta(b_fft_ref, mlow, mtheta)
    sqrt_jdp_ref = FFS.compute_sqrt_jac_delpsi(equil_ref, psi, mtheta)
    I_direct_ref = sum(abs2.(b_theta_ref) .* sqrt_jdp_ref .^ 2) / mtheta

    @printf("%-8s | %-22s | I/I_ref − 1 | ratio to floor\n", "jac", "‖sqrtamat·b_fft‖²")
    println(repeat("-", 78))
    @printf("%-8s | %.14e | %+.3e   | --\n", JAC_REF, I_ref, 0.0)

    max_rel = 0.0
    for jac in JACS_OUT
        equil = equils[jac]
        sqrtamat = FFS.compute_sqrtamat(equil, psi, FT.FourierTransform(mtheta, mpert, mlow))
        convmat = build_convmat(equil, equil_ref, psi, mpert, mlow, mtheta)
        I_jac = norm(sqrtamat * convmat * b_fft_ref)^2
        rel = I_jac / I_ref - 1
        max_rel = max(max_rel, abs(rel))
        @printf("%-8s | %.14e | %+.3e   | %.2f×\n", jac, I_jac, rel, abs(rel) / floor_tol)
    end

    println()
    @printf("Normalization cross-check on %s:\n", JAC_REF)
    @printf("  ‖sqrtamat·b_fft‖²        = %.6e\n", I_ref)
    @printf("  N²·∫|b|²·J|∇ψ|·dθ (trap) = %.6e   [N=%d]\n", mtheta^2 * I_direct_ref, mtheta)
    @printf("  ratio                    = %.6e   (expected ≈ 1)\n",
            I_ref / (mtheta^2 * I_direct_ref))

    ratio = max_rel / floor_tol
    verdict = ratio < 10 ? "PASS" : ratio < 100 ? "WEAK" : "FAIL"
    @printf("\n%s: max |I/I_ref − 1| = %.3e (%.2f× floor)\n\n", verdict, max_rel, ratio)
    return max_rel
end

# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

function main()
    mpert = MHIGH - MLOW + 1
    psi = PSIHIGH

    @info "Loading equilibria for $(JACS)..."
    equils = Dict(jac => build_equil(jac) for jac in JACS)

    q_at_psi = equils[JAC_REF].profiles.q_spline(psi)
    singfac_min = minimum(abs(m - N_TOR * q_at_psi) for m in MLOW:MHIGH)

    println()
    @printf("Surface: ψ=%.4f, q(ψ)=%.4f, n=%d\n", psi, q_at_psi, N_TOR)
    @printf("Fourier basis band:     m ∈ [%d, %d]  (mpert=%d)\n", MLOW, MHIGH, mpert)
    @printf("θ-grid points:          mtheta = %d\n", MTHETA)
    @printf("min|m − n·q| on band:   %.3e\n\n", singfac_min)

    # Narrow-band physical scalar field b (mode coefficients in the reference jac's
    # Fourier basis). Support kept well inside [MLOW, MHIGH] so angle-map leakage to
    # out-of-band modes is negligible.
    phys_amps = Dict(
        -3 =>  0.10 + 0.05im,
        -1 =>  0.50 + 0.10im,
         0 =>  1.00 + 0.00im,
         1 => -0.30 + 0.40im,
         2 =>  0.20 - 0.15im,
         3 =>  0.10 + 0.05im,
         6 =>  0.05 - 0.02im,
    )
    b_fft_ref = zeros(ComplexF64, mpert)
    for (m_k, a) in phys_amps
        b_fft_ref[m_k - MLOW + 1] = a
    end

    floor_tol = test_area_invariance(equils, psi, MTHETA)
    resid_2 = test_geometric_scalar_invariant(equils, psi, MTHETA, floor_tol)
    test_convmat(equils, psi, mpert, MLOW, MTHETA)
    resid_4 = test_sqrtamat_invariance(equils, psi, mpert, MLOW, MTHETA, b_fft_ref, floor_tol)

    println(repeat("=", 78))
    println("SUMMARY")
    println(repeat("=", 78))
    @printf("Floor tolerance (Test 1, area):        %.3e\n\n", floor_tol)
    @printf("  %-44s  residual     × floor\n", "Test")
    println(repeat("-", 78))
    @printf("  %-44s  %.3e    %.2f×\n", "2 — ∮|f(η)|² dA",          resid_2, resid_2 / floor_tol)
    @printf("  %-44s  %.3e    %.2f×\n", "4 — sqrtamat ‖√A·b_fft‖²", resid_4, resid_4 / floor_tol)
    println(repeat("=", 78))
    println("Expected: all residuals within ~10× floor.")
    println(repeat("=", 78))
end

main()
