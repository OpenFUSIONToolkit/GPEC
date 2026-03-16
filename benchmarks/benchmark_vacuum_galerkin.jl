#!/usr/bin/env julia

using Printf
using LinearAlgebra
using TOML
using Plots

using Pkg
Pkg.instantiate()
using BenchmarkTools

using GeneralizedPerturbedEquilibrium
const GPEC = GeneralizedPerturbedEquilibrium

"""
    make_wall_settings(example_dir::AbstractString)

Construct `Vacuum.WallShapeSettings` from the `[Wall]` section in `gpec.toml`
if present; otherwise return default settings.
"""
function make_wall_settings(example_dir::AbstractString)
    inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
    if haskey(inputs, "Wall")
        return GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
    elseif haskey(inputs, "WALL")
        # Some examples use legacy capitalized section name
        return GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["WALL"])...)
    else
        return GPEC.Vacuum.WallShapeSettings()
    end
end

"""
    load_equilibrium(example_dir::AbstractString)

Set up the equilibrium specified by `[Equilibrium]` in `gpec.toml` under `example_dir`.
"""
function load_equilibrium(example_dir::AbstractString)
    inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
    @assert haskey(inputs, "Equilibrium") "[Equilibrium] section missing in gpec.toml for $example_dir"
    eq_cfg = GPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], example_dir)
    return GPEC.Equilibrium.setup_equilibrium(eq_cfg)
end

"""
    make_vacuum_input(
        equil::GPEC.Equilibrium.PlasmaEquilibrium;
        ψ::Real,
        mtheta::Int,
        nzeta::Int,
        mpert::Int,
        mlow::Int,
        npert::Int,
        nlow::Int,
        use_galerkin::Bool,
    ) -> GPEC.Vacuum.VacuumInput

Construct a `VacuumInput` at flux surface `ψ` with the specified resolution and mode set.
The only parameter that differs between the two algorithms we compare is `use_galerkin`.
"""
function make_vacuum_input(
    equil::GPEC.Equilibrium.PlasmaEquilibrium;
    ψ::Real,
    mtheta::Int,
    nzeta::Int,
    mpert::Int,
    mlow::Int,
    npert::Int,
    nlow::Int,
    use_galerkin::Bool
)
    r, z, ν = GPEC.Vacuum.extract_plasma_surface_at_psi(equil, float(ψ))

    return GPEC.Vacuum.VacuumInput(;
        x=reverse(r),
        z=reverse(z),
        ν=reverse(ν),
        mtheta_in=length(r),
        nzeta_in=1,
        mlow=mlow,
        mpert=mpert,
        nlow=nlow,
        npert=npert,
        mtheta=mtheta,
        nzeta=nzeta,
        force_wv_symmetry=true,
        use_galerkin=use_galerkin
    )
end

"""
    benchmark_vacuum_2d(
        example_dir::AbstractString;
        ψ::Real = 0.95,
        mtheta_values::AbstractVector{<:Integer} = 16 .* (2 .^ (0:9)),
        mpert::Int = 32,
        mlow::Int = 0,
        npert::Int = 1,
        nlow::Int = 1,
    )

Benchmark `compute_vacuum_response` for 2D (nzeta = 1) using the Solovev example in
`example_dir`. Scans over `mtheta_values` and compares collocation (`use_galerkin=false`)
against Galerkin (`use_galerkin=true`) for convergence of the `wv` matrix and runtime.
"""
function benchmark_vacuum_2d(
    example_dir::AbstractString;
    ψ::Real=1.0,
    mtheta_values::AbstractVector{<:Integer}=16 .* (2 .^ (0:7)),
    mpert::Int=32,
    mlow::Int=0,
    npert::Int=1,
    nlow::Int=1
)
    println("\n===== 2D Vacuum Benchmark (Solovev, $(basename(example_dir))) =====")
    println("ψ = $(ψ), mtheta ∈ $(collect(mtheta_values)), mpert=$mpert, mlow=$mlow, nlow=$nlow, npert=$npert\n")

    equil = load_equilibrium(example_dir)
    wall_settings = make_wall_settings(example_dir)

    mtheta_values = collect(mtheta_values)
    nm = length(mtheta_values)

    times_colloc = zeros(Float64, nm)
    times_galerkin = zeros(Float64, nm)
    errs_colloc = zeros(Float64, nm)
    errs_galerkin = zeros(Float64, nm)

    # Reference wv for convergence (highest resolution * 2) – always use galerkin
    mtheta_ref = maximum(mtheta_values) * 2
    println("Computing 2D reference matrices at mtheta_ref = $mtheta_ref")

    input_ref_galerkin = make_vacuum_input(
        equil;
        ψ=ψ,
        mtheta=mtheta_ref,
        nzeta=1,
        mpert=mpert,
        mlow=mlow,
        npert=npert,
        nlow=nlow,
        use_galerkin=true
    )
    wv_ref_galerkin, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_ref_galerkin, wall_settings)
    # IMPORTANT: `compute_vacuum_response` uses AdaptiveArrayPools; take a copy so
    # later pooled allocations do not overwrite the reference storage.
    wv_ref_galerkin = copy(wv_ref_galerkin)
    ref_norm_galerkin = norm(wv_ref_galerkin)

    for (i, mtheta) in enumerate(mtheta_values)
        println("  2D: mtheta = $(rpad(string(mtheta), 5))  (nzeta = 1)")

        # Collocation
        input_colloc = make_vacuum_input(
            equil;
            ψ=ψ,
            mtheta=mtheta,
            nzeta=1,
            mpert=mpert,
            mlow=mlow,
            npert=npert,
            nlow=nlow,
            use_galerkin=false
        )
        t_colloc = @belapsed GPEC.Vacuum.compute_vacuum_response($input_colloc, $wall_settings)
        wv, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_colloc, wall_settings)
        errs_colloc[i] = norm(wv .- wv_ref_galerkin) / ref_norm_galerkin
        times_colloc[i] = t_colloc

        # Galerkin
        input_galerkin = make_vacuum_input(
            equil;
            ψ=ψ,
            mtheta=mtheta,
            nzeta=1,
            mpert=mpert,
            mlow=mlow,
            npert=npert,
            nlow=nlow,
            use_galerkin=true
        )
        t_galerkin = @belapsed GPEC.Vacuum.compute_vacuum_response($input_galerkin, $wall_settings)
        wv_g, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_galerkin, wall_settings)
        errs_galerkin[i] = norm(wv_g .- wv_ref_galerkin) / ref_norm_galerkin
        times_galerkin[i] = t_galerkin

        @printf("    Collocation:  t = %.3f s,  rel‖Δwv‖ = %.3e\n", t_colloc, errs_colloc[i])
        @printf("    Galerkin:     t = %.3f s,  rel‖Δwv‖ = %.3e\n", t_galerkin, errs_galerkin[i])
    end

    # Two‑pane plot: left = convergence, right = runtime
    plt = plot(; layout=(1, 2), size=(1100, 420))

    # Convergence
    plot!(plt[1], mtheta_values, errs_colloc;
        lw=2, marker=:circle, xscale=:log10, yscale=:log10,
        label="Collocation", xlabel="mθ", ylabel="rel‖Δwv‖",
        title="2D Vacuum: wv convergence vs mθ")
    plot!(plt[1], mtheta_values, errs_galerkin;
        lw=2, marker=:square, xscale=:log10, yscale=:log10,
        label="Galerkin")

    # Runtime
    plot!(plt[2], mtheta_values, times_colloc;
        lw=2, marker=:circle, xscale=:log10, yscale=:log10,
        label="Collocation", xlabel="mθ", ylabel="runtime [s]",
        title="2D Vacuum: runtime vs mθ")
    plot!(plt[2], mtheta_values, times_galerkin;
        lw=2, marker=:square, xscale=:log10, yscale=:log10,
        label="Galerkin")

    plot!(plt[1]; legend=:bottomleft)
    plot!(plt[2]; legend=:topleft)

    outpath = joinpath(@__DIR__, "vacuum_galerkin_2d.png")
    savefig(plt, outpath)
    println("\n2D results saved to $(outpath)")

    return (; mtheta_values, times_colloc, times_galerkin, errs_colloc, errs_galerkin)
end

"""
    benchmark_vacuum_3d(
        example_dir::AbstractString;
        ψ::Real = 0.95,
        mtheta_values::AbstractVector{<:Integer} = 16 .* (2 .^ (0:3)),
        mpert::Int = 16,
        mlow::Int = 0,
        npert::Int = 1,
        nlow::Int = 1,
    )

Benchmark `compute_vacuum_response` for 3D (nzeta = mtheta) using the Solovev 3D example
in `example_dir`. Scans over `mtheta = nzeta` and compares collocation vs Galerkin.
"""
function benchmark_vacuum_3d(
    example_dir::AbstractString;
    ψ::Real=1.0,
    mtheta_values::AbstractVector{<:Integer}=16 .* (2 .^ (0:2)),
    mpert::Int=16,
    mlow::Int=0,
    npert::Int=1,
    nlow::Int=1
)
    println("\n===== 3D Vacuum Benchmark (Solovev 3D, $(basename(example_dir))) =====")
    println("ψ = $(ψ), mtheta = nzeta ∈ $(collect(mtheta_values)), mpert=$mpert, mlow=$mlow, nlow=$nlow, npert=$npert\n")

    equil = load_equilibrium(example_dir)
    wall_settings = make_wall_settings(example_dir)

    mtheta_values = collect(mtheta_values)
    nm = length(mtheta_values)

    times_colloc = zeros(Float64, nm)
    times_galerkin = zeros(Float64, nm)
    errs_colloc = zeros(Float64, nm)
    errs_galerkin = zeros(Float64, nm)

    # Reference wv for convergence (highest resolution * 2) – always use galerkin
    mtheta_ref = maximum(mtheta_values) * 2
    nzeta_ref = mtheta_ref
    println("Computing 3D reference matrices at mtheta_ref = nzeta_ref = $mtheta_ref")

    input_ref_galerkin = make_vacuum_input(
        equil;
        ψ=ψ,
        mtheta=mtheta_ref,
        nzeta=nzeta_ref,
        mpert=mpert,
        mlow=mlow,
        npert=npert,
        nlow=nlow,
        use_galerkin=true
    )
    wv_ref_galerkin, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_ref_galerkin, wall_settings)
    # Again, protect the reference matrix from being overwritten by pooled allocations.
    wv_ref_galerkin = copy(wv_ref_galerkin)
    ref_norm_galerkin = norm(wv_ref_galerkin)

    for (i, mtheta) in enumerate(mtheta_values)
        nzeta = mtheta
        println("  3D: mtheta = $(rpad(string(mtheta), 5)), nzeta = $(rpad(string(nzeta), 5))")

        # Collocation
        input_colloc = make_vacuum_input(
            equil;
            ψ=ψ,
            mtheta=mtheta,
            nzeta=nzeta,
            mpert=mpert,
            mlow=mlow,
            npert=npert,
            nlow=nlow,
            use_galerkin=false
        )
        t_colloc = @belapsed GPEC.Vacuum.compute_vacuum_response($input_colloc, $wall_settings)
        wv3, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_colloc, wall_settings)
        errs_colloc[i] = norm(wv3 .- wv_ref_galerkin) / ref_norm_galerkin
        times_colloc[i] = t_colloc

        # Galerkin
        input_galerkin = make_vacuum_input(
            equil;
            ψ=ψ,
            mtheta=mtheta,
            nzeta=nzeta,
            mpert=mpert,
            mlow=mlow,
            npert=npert,
            nlow=nlow,
            use_galerkin=true
        )
        t_galerkin = @belapsed GPEC.Vacuum.compute_vacuum_response($input_galerkin, $wall_settings)
        wv3g, _, _, _, _ = GPEC.Vacuum.compute_vacuum_response(input_galerkin, wall_settings)
        errs_galerkin[i] = norm(wv3g .- wv_ref_galerkin) / ref_norm_galerkin
        times_galerkin[i] = t_galerkin

        @printf("    Collocation:  t = %.3f s,  rel‖Δwv‖ = %.3e\n", t_colloc, errs_colloc[i])
        @printf("    Galerkin:     t = %.3f s,  rel‖Δwv‖ = %.3e\n", t_galerkin, errs_galerkin[i])
    end

    # Two‑pane plot: left = convergence, right = runtime (mtheta = nzeta on x-axis)
    plt = plot(; layout=(1, 2), size=(1100, 420))

    # Convergence
    plot!(plt[1], mtheta_values, errs_colloc;
        lw=2, marker=:circle, xscale=:log10, yscale=:log10,
        label="Collocation", xlabel="mθ = nzeta", ylabel="rel‖Δwv‖",
        title="3D Vacuum: wv convergence vs mθ = nzeta")
    plot!(plt[1], mtheta_values, errs_galerkin;
        lw=2, marker=:square, xscale=:log10, yscale=:log10,
        label="Galerkin")

    # Runtime
    plot!(plt[2], mtheta_values, times_colloc;
        lw=2, marker=:circle, xscale=:log10, yscale=:log10,
        label="Collocation", xlabel="mθ = nzeta", ylabel="runtime [s]",
        title="3D Vacuum: runtime vs mθ = nzeta")
    plot!(plt[2], mtheta_values, times_galerkin;
        lw=2, marker=:square, xscale=:log10, yscale=:log10,
        label="Galerkin")

    plot!(plt[1]; legend=:bottomleft)
    plot!(plt[2]; legend=:topleft)

    outpath = joinpath(@__DIR__, "vacuum_galerkin_3d.png")
    savefig(plt, outpath)
    println("\n3D results saved to $(outpath)")

    return (; mtheta_values, times_colloc, times_galerkin, errs_colloc, errs_galerkin)
end

"""
    main()

Entry point when running this file as a script.

Usage (from repository root):

```bash
julia --project=. benchmarks/benchmark_vacuum_galerkin.jl
```

Edit the `mtheta_values` and other keyword arguments in the calls below to
explore different resolution ranges.
"""
function main()
    # 2D Solovev example
    example_2d = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")

    # 3D Solovev example
    example_3d = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example_3D")

    benchmark_vacuum_2d(
        example_2d;
        ψ=1.0,
        mtheta_values=16 .* (2 .^ (0:9)),  # 16 → 8192 (easily editable)
        mpert=31,
        mlow=-15,
        npert=1,
        nlow=1
    )

    benchmark_vacuum_3d(
        example_3d;
        ψ=1.0,
        mtheta_values=16 .* (2 .^ (0:3)), # 16 → 128 (easily editable)
        mpert=31,
        mlow=-15,
        npert=1,
        nlow=1
    )

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
