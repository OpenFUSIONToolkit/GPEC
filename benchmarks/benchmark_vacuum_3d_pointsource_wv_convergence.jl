#!/usr/bin/env julia

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
    # VacuumInput axisymmetric convention: x = R(θ), z = Z(θ), ν(θ) = 0
    # with θ sampled on [0, 2π].
    θ_in = range(0.0, TWO_PI; length=mtheta_in)
    x = R0 .+ a .* cos.(θ_in)
    z = a .* sin.(θ_in)
    ν = zeros(Float64, mtheta_in)
    return x, z, ν
end

function analytic_two_point_potential_and_normal_derivative(plasma_surf; R0::Float64, eps::Float64=1e-14)
    # Point sources at x0 = (±R0, 0, 0):
    #   χ(X) = 1/|X-x0| + 1/|X-(-x0)|
    #
    # Neumann data convention:
    #   Vacuum kernels use the oriented (non-unit) normal vector n_vec = ∂r/∂θ × ∂r/∂ζ.
    #   We compute dχ/dñ = ∇χ · n_vec (no unit normalization).
    r = plasma_surf.r
    nvec = plasma_surf.normal
    N = size(r, 1)

    chi = zeros(Float64, N)
    dchi_dn = zeros(Float64, N)

    @inbounds for idx in 1:N
        X = r[idx, 1]
        Y = r[idx, 2]
        Z = r[idx, 3]

        nx = nvec[idx, 1]
        ny = nvec[idx, 2]
        nz = nvec[idx, 3]

        V = 0.0
        dV = 0.0
        for sgn in [1.0] # (-1.0, 1.0)
            x0 = sgn * R0
            dx = X - x0
            dy = Y
            dz = Z

            dist2 = dx*dx + dy*dy + dz*dz + eps
            dist = sqrt(dist2)
            invdist = 1.0 / dist
            invdist3 = invdist / dist2

            # ∇(1/r) = -(x-x0)/r^3
            V += sgn * invdist
            dV += -sgn * (dx*nx + dy*ny + dz*nz) * invdist3
        end

        chi[idx] = V
        dchi_dn[idx] = dV
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
    # Use the same basis construction as Vacuum internals.
    exp_mn_basis = compute_fourier_coefficients(mtheta, mpert, mlow, nzeta, npert, nlow)
    num_pts = mtheta * nzeta
    return (conj(exp_mn_basis)' * f_vals) ./ num_pts
end

function run_wv_pointsource_benchmark(; R0::Float64=4.0, a_minor::Float64=3.0, mtheta_values::AbstractVector{<:Integer}=[16, 24, 32, 48, 64, 72]
)
    # Fourier modes
    mlow = -4
    mpert = 9
    nlow = -10
    npert = 21

    # Observed aspect ratio of the grid
    aspect_ratio_observed = 1.3

    wall_settings = Vacuum.WallShapeSettings(; shape="nowall")

    results = Vector{NamedTuple{(:mtheta, :nzeta, :relerr),Tuple{Int,Int,Float64}}}()
    mode_spectra = Vector{NamedTuple{(:mtheta, :nzeta, :chi_amp, :dchi_dn_amp),Tuple{Int,Int,Matrix{Float64},Matrix{Float64}}}}()

    # Compute the number of toroidal points based on the observed aspect ratio (keep it approximately 1)
    nzeta_for(mtheta::Int) = max(2, Int(round(mtheta * aspect_ratio_observed)))

    mtheta_in = 1024
    x, z, ν = make_axisymmetric_circle_inputs(; mtheta_in=mtheta_in, R0=R0, a=a_minor)

    for (ires, mtheta) in enumerate(mtheta_values)
        nzeta = nzeta_for(mtheta)

        vac_inputs = VacuumInput(;
            x=x, z=z, ν=ν,
            mtheta_in=mtheta_in, nzeta_in=1,
            mlow=mlow, mpert=mpert,
            nlow=nlow, npert=npert,
            mtheta=mtheta, nzeta=nzeta,
            force_wv_symmetry=true
        )

        t0 = time()
        wv, _, _, _, _ = Vacuum.compute_vacuum_response(vac_inputs, wall_settings)
        t_wv = time() - t0

        # Analytic point-source potential and Neumann data on the same surface.
        plasma_surf = Vacuum.PlasmaGeometry3D(vac_inputs)
        chi_vals, dchi_dn_vals = analytic_two_point_potential_and_normal_derivative(plasma_surf; R0=R0)

        # Project onto the same basis as the vacuum response matrix
        chi_modes = project_modes_with_basis(chi_vals, mtheta, nzeta, mlow, mpert, nlow, npert)
        dchi_dn_modes = project_modes_with_basis(dchi_dn_vals, mtheta, nzeta, mlow, mpert, nlow, npert)

        # Save amplitude spectra for diagnostics: shape is (mpert, npert)
        chi_amp = reshape(abs.(chi_modes), mpert, npert)
        dchi_dn_amp = reshape(abs.(dchi_dn_modes), mpert, npert)
        push!(mode_spectra, (; mtheta, nzeta, chi_amp, dchi_dn_amp))

        # Vacuum.jl uses: wv = 4π² * (K⁻¹G) in mode space.
        # With our Neumann convention: chi ≈ (wv / 4π²) * (dchi/dn̂)
        chi_pred = (wv ./ FOUR_PI2) * dchi_dn_modes

        # Compute relative error between analytic and predicted chi
        relerr = norm(chi_pred .- chi_modes) / norm(chi_modes)
        push!(results, (; mtheta=mtheta, nzeta=nzeta, relerr=relerr))

        @printf("mtheta=%d nzeta=%d | wv %.2fs | relerr=%.3e\n",
            mtheta, nzeta, t_wv, relerr)
    end

    # Plot: error vs total grid points (mtheta * nzeta).
    xvals = [r.mtheta * r.nzeta for r in results]
    yraw = [r.relerr for r in results]

    plt = plot(xvals, yraw;
        xscale=:log10,
        yscale=:log10,
        lw=2, marker=:circle,
        xlabel="Total grid points (mtheta * nzeta)",
        ylabel="Relative error in chi",
        title="3D Vacuum wv vs grid resolution",
        label=nothing
    )

    outpath = joinpath(@__DIR__, "vacuum_3d_pointsource_wv_convergence.png")
    savefig(plt, outpath)

    # Diagnostic plot: Fourier mode amplitudes for chi and dchi/dn at finest resolution
    spec = mode_spectra[end]
    m_modes = mlow:(mlow+mpert-1)
    n_modes = nlow:(nlow+npert-1)
    chi_plot = log10.(spec.chi_amp .+ eps(Float64))
    dchi_plot = log10.(spec.dchi_dn_amp .+ eps(Float64))

    amp_plot = plot(; layout=(1, 2), size=(1100, 400))
    heatmap!(amp_plot[1], n_modes, m_modes, chi_plot;
        xlabel="n mode", ylabel="m mode",
        title="log10|chi| (mtheta=$(spec.mtheta), nzeta=$(spec.nzeta))",
        colorbar=true)
    heatmap!(amp_plot[2], n_modes, m_modes, dchi_plot;
        xlabel="n mode", ylabel="m mode",
        title="log10|dchi/dn| (mtheta=$(spec.mtheta), nzeta=$(spec.nzeta))",
        colorbar=true)
    amp_outpath = joinpath(@__DIR__, "vacuum_3d_pointsource_mode_amplitudes.png")
    savefig(amp_plot, amp_outpath)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_wv_pointsource_benchmark()
end
