#!/usr/bin/env julia
#
# Convergence test for the vacuum boundary integral operator using harmonic polynomials.
#
# The vacuum response matrix wv encodes the EXTERIOR BIE (K_ext⁻¹ G), valid for functions
# harmonic outside the plasma that decay at infinity. Harmonic polynomials grow at infinity,
# so they cannot be used with the exterior BIE directly.
#
# However, the code also computes the INTERIOR BIE solution (K_int⁻¹ G), stored in grri.
# Harmonic polynomials ARE valid test functions for the interior BIE (bounded domain, no
# contribution from S_∞). Since interior and exterior BIEs share the same kernel matrices
# K and G — differing only in the diagonal jump term — validating the kernel assembly via
# the interior BIE also validates the exterior BIE.
#
# Test function: P(x,y,z) = xz + (x² - y²)
#
# Both terms are degree-2 solid harmonics (∇²(xz) = 0, ∇²(x²-y²) = 0), so P is harmonic
# everywhere. On a circular cross-section torus, χ = P|_S and dχ/dn = ∇P · n_vec have
# exactly finite Fourier bandwidth, eliminating truncation error entirely and isolating the
# quadrature convergence of the boundary integral operator.

using Printf
using LinearAlgebra
using Pkg
Pkg.instantiate()

using Plots

using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium

using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.Utilities.FourierTransforms: compute_fourier_coefficients

const TWO_PI = 2π
const FOUR_PI2 = 4π^2

function make_axisymmetric_circle_inputs(; mtheta_in::Int, R0::Float64, a::Float64)
    θ_in = range(0.0, TWO_PI; length=mtheta_in)
    x = R0 .+ a .* cos.(θ_in)
    z = a .* sin.(θ_in)
    ν = zeros(Float64, mtheta_in)
    return x, z, ν
end

"""
Evaluate the harmonic polynomial P(x,y,z) = xz + (x² - y²) and its gradient on the surface.

Returns (χ, dχ/dn) where the normal derivative uses the oriented (non-unit) normal n_vec.
"""
function harmonic_polynomial_on_surface(plasma_surf)
    r = plasma_surf.r
    nvec = plasma_surf.normal
    N = size(r, 1)

    chi = zeros(Float64, N)
    dchi_dn = zeros(Float64, N)

    @inbounds for idx in 1:N
        x = r[idx, 1]
        y = r[idx, 2]
        z = r[idx, 3]

        nx = nvec[idx, 1]
        ny = nvec[idx, 2]
        nz = nvec[idx, 3]

        # P(x,y,z) = xz + (x² - y²)
        chi[idx] = x * z + x^2 - y^2

        # ∇P = (z + 2x,  -2y,  x)
        dPdx = z + 2x
        dPdy = -2y
        dPdz = x
        dchi_dn[idx] = dPdx * nx + dPdy * ny + dPdz * nz
    end

    return chi, dchi_dn
end

function project_modes_with_basis(
    f_vals::AbstractVector{<:Real},
    mtheta::Int,
    nzeta::Int,
    mlow::Int,
    mpert::Int,
    nlow::Int,
    npert::Int
)::Vector{ComplexF64}
    exp_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow)
    num_pts = mtheta * nzeta
    return (conj(exp_mn_basis)' * f_vals) ./ num_pts
end

"""
Reconstruct the interior BIE response matrix wv_int from grri.

The vacuum code stores grri = Z * (K_int⁻¹ G) in a real M×2P layout, where Z is the
Fourier basis. Inverting the Fourier transform recovers the mode-space interior response:
wv_int = 4π² K_int⁻¹ G
"""
function reconstruct_wv_int(grri, vac_inputs)
    (; mtheta, nzeta, mpert, mlow, npert, nlow) = vac_inputs
    num_points = mtheta * nzeta
    num_modes = mpert * npert

    exp_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow)

    grri_complex = complex.(
        @view(grri[1:num_points, 1:num_modes]),
        @view(grri[1:num_points, (num_modes+1):(2*num_modes)])
    )

    # Invert the Fourier transform: G_int = Z⁺ · grri_complex = conj(Z)' · grri_complex / N
    # Then wv_int = 4π² · G_int
    G_int = (conj(exp_mn_basis)' * grri_complex) ./ num_points
    return FOUR_PI2 .* G_int
end

function run_harmonic_polynomial_benchmark(;
    R0::Float64=5.0,
    a_minor::Float64=2.0,
    mtheta_values::AbstractVector{<:Integer}=[12, 16, 24, 32, 48, 64]
)
    # Fixed Fourier basis chosen to exactly contain the polynomial's bandwidth.
    # On a circular torus, P = xz + (x²-y²) has χ modes up to |m|=2, |n|=2 and
    # dχ/dn modes up to |m|≈3, |n|≈2. We include margin for safety.
    mlow = -4
    mpert = 9      # m modes: -4 to 4
    nlow = -3
    npert = 7      # n modes: -3 to 3

    aspect_ratio = 2.6
    wall_settings = Vacuum.WallShapeSettings(; shape="nowall")

    results = Vector{NamedTuple{(:mtheta, :nzeta, :relerr, :t_wv),
        Tuple{Int,Int,Float64,Float64}}}()

    nzeta_for(mtheta::Int) = max(2, Int(round(mtheta * aspect_ratio)))

    mtheta_in = 1024
    x, z, ν = make_axisymmetric_circle_inputs(; mtheta_in=mtheta_in, R0=R0, a=a_minor)

    last_spec = nothing

    for mtheta in mtheta_values
        nzeta = nzeta_for(mtheta)

        vac_inputs = VacuumInput(;
            x=x, z=z, ν=ν,
            mtheta_in=mtheta_in, nzeta_in=1,
            mlow=mlow, mpert=mpert,
            nlow=nlow, npert=npert,
            mtheta=mtheta, nzeta=nzeta,
            force_wv_symmetry=false
        )

        t0 = time()
        wv_int, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, wall_settings)
        t_wv = time() - t0

        # wv_int = reconstruct_wv_int(grri, vac_inputs)

        plasma_surf = Vacuum.PlasmaGeometry3D(vac_inputs)
        chi_vals, dchi_dn_vals = harmonic_polynomial_on_surface(plasma_surf)

        chi_modes = project_modes_with_basis(chi_vals, mtheta, nzeta, mlow, mpert, nlow, npert)
        dchi_dn_modes = project_modes_with_basis(dchi_dn_vals, mtheta, nzeta, mlow, mpert, nlow, npert)

        chi_pred = (wv_int ./ FOUR_PI2) * dchi_dn_modes

        relerr = norm(chi_pred .- chi_modes) / norm(chi_modes)
        push!(results, (; mtheta, nzeta, relerr, t_wv))

        m_modes = mlow:(mlow+mpert-1)
        n_modes = nlow:(nlow+npert-1)
        chi_amp = reshape(abs.(chi_modes), mpert, npert)
        dchi_dn_amp = reshape(abs.(dchi_dn_modes), mpert, npert)
        last_spec = (; mtheta, nzeta, m_modes, n_modes, chi_amp, dchi_dn_amp)

        @printf("mtheta=%-3d nzeta=%-3d | wv %.2fs | relerr=%.3e\n",
            mtheta, nzeta, t_wv, relerr)
    end

    # Convergence plot
    xvals = [r.mtheta * r.nzeta for r in results]
    yraw = [r.relerr for r in results]

    plt = plot(xvals, yraw;
        xscale=:log10,
        yscale=:log10,
        lw=2, marker=:circle,
        xlabel="Total grid points (mtheta × nzeta)",
        ylabel="Relative error in χ",
        title="Interior BIE convergence: P(x,y,z) = xz + (x²-y²)",
        label="quadrature error"
    )

    outpath = joinpath(@__DIR__, "vacuum_3d_harmonic_polynomial_convergence.png")
    savefig(plt, outpath)

    # Diagnostic plot: Fourier mode amplitudes at finest resolution
    spec = last_spec
    chi_plot = log10.(spec.chi_amp .+ eps(Float64))
    dchi_plot = log10.(spec.dchi_dn_amp .+ eps(Float64))

    amp_plot = plot(; layout=(1, 2), size=(1100, 400))
    heatmap!(amp_plot[1], collect(spec.n_modes), collect(spec.m_modes), chi_plot;
        xlabel="n mode", ylabel="m mode",
        title="log10|chi| (mtheta=$(spec.mtheta), nzeta=$(spec.nzeta))",
        colorbar=true)
    heatmap!(amp_plot[2], collect(spec.n_modes), collect(spec.m_modes), dchi_plot;
        xlabel="n mode", ylabel="m mode",
        title="log10|dchi/dn| (mtheta=$(spec.mtheta), nzeta=$(spec.nzeta))",
        colorbar=true)
    amp_outpath = joinpath(@__DIR__, "vacuum_3d_harmonic_polynomial_mode_amplitudes.png")
    savefig(amp_plot, amp_outpath)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_harmonic_polynomial_benchmark()
end
