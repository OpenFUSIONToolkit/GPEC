# Coverage for Vacuum entry points not exercised by runtests_vacuum.jl:
#   - compute_vacuum_response!   (in-place, duck-typed external storage)
#   - extract_plasma_surface_at_psi (equilibrium → surface geometry)
#   - compute_vacuum_field       (currently a LATENT BUG — see below)
# Kept in a separate file to avoid diff collision with the large runtests_vacuum.jl.

@testset "Vacuum coverage" begin
    Vac = GeneralizedPerturbedEquilibrium.Vacuum
    Eq = GeneralizedPerturbedEquilibrium.Equilibrium

    # A circular plasma boundary; mtheta_eq points define the boundary spline.
    _make_inputs(; mtheta=64, mtheta_eq=17, mpert=2, mlow=1, nlow=1, npert=1) = VacuumInput(
        mtheta_in=mtheta_eq,
        nzeta_in=1,
        x=collect(1.7 .+ 0.3 .* cos.(range(0, 2π, length=mtheta_eq))),
        z=collect(0.3 .* sin.(range(0, 2π, length=mtheta_eq))),
        ν=zeros(mtheta_eq),
        mlow=mlow,
        mpert=mpert,
        nlow=nlow,
        npert=npert,
        nzeta=1,
        mtheta=mtheta
    )

    @testset "compute_vacuum_response! matches allocating wrapper" begin
        # compute_vacuum_response is a thin allocating wrapper around the in-place
        # routine, so this verifies the in-place entry correctly populates
        # caller-owned, duck-typed (NamedTuple) storage to the same result.
        for shape in ("nowall", "conformal")
            inputs = _make_inputs()
            wall_settings = shape == "conformal" ? WallShapeSettings(shape=shape, a=0.5) : WallShapeSettings(shape=shape)

            wv_ref, grri_ref, grre_ref, pp_ref, wp_ref = compute_vacuum_response(inputs, wall_settings)

            numpoints = inputs.mtheta * inputs.nzeta
            num_modes = inputs.mpert * inputs.npert
            vac = (
                wv=zeros(ComplexF64, num_modes, num_modes),
                grri=zeros(2 * numpoints, 2 * num_modes),
                grre=zeros(2 * numpoints, 2 * num_modes),
                plasma_pts=zeros(numpoints, 3),
                wall_pts=zeros(numpoints, 3)
            )
            compute_vacuum_response!(vac, inputs, wall_settings)

            @test vac.wv ≈ wv_ref
            @test vac.grri ≈ grri_ref
            @test vac.grre ≈ grre_ref
            @test vac.plasma_pts ≈ pp_ref
            @test vac.wall_pts ≈ wp_ref
        end
    end

    @testset "compute_vacuum_response! green_only suppresses the response matrix" begin
        # Negative control: a full run produces a non-zero wv, so the zeroed wv under
        # green_only is load-bearing (not merely the unconditional wv .= 0 in the 2D path).
        inputs = _make_inputs()
        wall_settings = WallShapeSettings(shape="nowall")
        numpoints = inputs.mtheta * inputs.nzeta
        num_modes = inputs.mpert * inputs.npert
        _vac() = (
            wv=zeros(ComplexF64, num_modes, num_modes),
            grri=zeros(2 * numpoints, 2 * num_modes),
            grre=zeros(2 * numpoints, 2 * num_modes),
            plasma_pts=zeros(numpoints, 3),
            wall_pts=zeros(numpoints, 3)
        )

        full = _vac()
        compute_vacuum_response!(full, inputs, wall_settings; green_only=false)
        @test any(!iszero, full.wv)  # full run drives a real response

        go = _vac()
        compute_vacuum_response!(go, inputs, wall_settings; green_only=true)
        @test all(iszero, go.wv)            # green_only must leave wv unpopulated
        @test go.grri ≈ full.grri           # Green's functions are still computed identically
        @test go.grre ≈ full.grre
    end

    @testset "extract_plasma_surface_at_psi (Solovev)" begin
        # Self-contained analytic Solovev equilibrium (same recipe as runtests_equil.jl).
        eq_config = Eq.EquilibriumConfig(; eq_type="sol", eq_filename="unused",
            jac_type="pest", grid_type="ldp",
            psilow=1e-4, psihigh=0.99999, mpsi=64, mtheta=128)
        sol_config = Eq.SolovevConfig(64, 64, 64, 1.6, 0.33, 1.0, 1.9, 1.0, 1.0, 1.0)
        pe = Eq.equilibrium_solver(Eq.sol_run(eq_config, sol_config))
        @test pe isa Eq.PlasmaEquilibrium

        mtheta = length(pe.rzphi_ys)
        r, z, ν = extract_plasma_surface_at_psi(pe, 0.5)
        @test length(r) == mtheta
        @test length(z) == mtheta
        @test length(ν) == mtheta
        @test all(isfinite, r) && all(isfinite, z) && all(isfinite, ν)
        @test all(r .> 0)                       # R must be physical (positive)
        @test minimum(r) < pe.ro < maximum(r)   # surface brackets the magnetic axis in R

        # Minor-radius extent must grow monotonically outward in ψ.
        extent(rr, zz) = maximum(hypot.(rr .- pe.ro, zz .- pe.zo))
        r_in, z_in, _ = extract_plasma_surface_at_psi(pe, 0.2)
        r_out, z_out, _ = extract_plasma_surface_at_psi(pe, 0.8)
        @test extent(r_out, z_out) > extent(r_in, z_in)
    end

    @testset "compute_vacuum_field — LATENT BUG (unreachable: vaccal! + inputs.n)" begin
        # compute_vacuum_field is currently unreachable, gated by TWO independent
        # defects, and is called nowhere in src/test/benchmarks:
        #   (a) it calls `vaccal!(inputs, plasma_surf, wall)` (Vacuum.jl:317), but
        #       `vaccal!` is defined nowhere → UndefVarError is thrown first; and
        #   (b) further down, _pickup_field reads `inputs.n` (Field.jl:94), but
        #       VacuumInput has no `n` field (only nlow/npert).
        # Both must be resolved before the function can run. Per repo policy these
        # tests document the bug without patching source; flip @test_broken to real
        # shape/finiteness assertions once both blockers are fixed.
        inputs = _make_inputs(mtheta=32, mtheta_eq=17, mpert=3, mlow=-1)
        plasma_surf = Vac.PlasmaGeometry(inputs)
        wall = Vac.WallGeometry(inputs, plasma_surf, WallShapeSettings(shape="conformal", a=0.5))
        Bn = ones(ComplexF64, inputs.mpert)
        R_grid = collect(range(1.5, 1.9, length=3))
        Z_grid = collect(range(-0.2, 0.2, length=3))

        @test !isdefined(Vac, :vaccal!)  # blocker (a): trips when vaccal! is implemented
        @test !hasproperty(inputs, :n)   # blocker (b): trips when the n field is added
        @test_broken (compute_vacuum_field(inputs, plasma_surf, wall, Bn, R_grid, Z_grid); true)
    end
end
