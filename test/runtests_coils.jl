using Test
using TOML
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.Equilibrium

const COIL_DIR = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries")
const D3D_IL = joinpath(COIL_DIR, "d3d_il.dat")

# ---------------------------------------------------------------------------
@testset "CoilGeometry: read_coil_dat" begin
    @test isfile(D3D_IL)

    cs = ForcingTerms.read_coil_dat(D3D_IL)

    @test cs.ncoil == 6
    @test cs.s     == 1
    @test cs.nsec  == 126
    @test cs.nw    ≈ 1.0

    # Coordinates should be in plausible DIII-D range (R ~ 1–2.5 m, |Z| < 2 m)
    R_vals = sqrt.(cs.x .^ 2 .+ cs.y .^ 2)
    @test all(0.5 .< R_vals .< 3.0)
    @test all(-2.5 .< cs.z .< 2.5)

    # Size check
    @test size(cs.x) == (6, 1, 126)
end

# ---------------------------------------------------------------------------
@testset "CoilGeometry: apply_transforms — shift" begin
    cs = ForcingTerms.read_coil_dat(D3D_IL)
    # Assign unit currents
    cs2 = ForcingTerms.CoilSet(cs.name, cs.ncoil, cs.s, cs.nw, cs.nsec,
        cs.x, cs.y, cs.z, ones(cs.ncoil))

    Δx = 0.05
    cfg = ForcingTerms.CoilSetConfig(; shiftx=[Δx for _ in 1:cs.ncoil])
    cs_shifted = ForcingTerms.apply_transforms(cs2, cfg; n_tilt=0)

    # For n_tilt=0 (rigid shift), every x coordinate should increase by Δx
    @test all(cs_shifted.x .- cs.x .≈ Δx)
    @test all(cs_shifted.y .≈ cs.y)
    @test all(cs_shifted.z .≈ cs.z)
end

@testset "CoilGeometry: apply_transforms — zero tilt" begin
    cs = ForcingTerms.read_coil_dat(D3D_IL)
    cs2 = ForcingTerms.CoilSet(cs.name, cs.ncoil, cs.s, cs.nw, cs.nsec,
        cs.x, cs.y, cs.z, ones(cs.ncoil))
    cfg = ForcingTerms.CoilSetConfig()  # all defaults = zero perturbations
    cs_same = ForcingTerms.apply_transforms(cs2, cfg; n_tilt=1)

    @test all(cs_same.x .≈ cs.x)
    @test all(cs_same.y .≈ cs.y)
    @test all(cs_same.z .≈ cs.z)
end

# ---------------------------------------------------------------------------
@testset "BiotSavart: circular loop analytic validation" begin
    # Circular loop of radius a in the xy-plane at z=0, current I in the +θ direction.
    # At observer (0, 0, h): B_z = μ₀ I a² / (2 (a²+h²)^(3/2))   [Jackson 5.38]
    a = 1.0; I_val = 1.0; h = 0.5
    n_seg = 500

    # Build a discrete circular loop as a CoilSet (ncoil=1, s=1, nsec=n_seg+1)
    phis  = range(0; length=n_seg + 1, stop=2π)
    xs    = a .* cos.(phis)
    ys    = a .* sin.(phis)
    zs    = zeros(n_seg + 1)

    cs = ForcingTerms.CoilSet("test_loop", 1, 1, 1.0, n_seg + 1,
        reshape(xs, 1, 1, :),
        reshape(ys, 1, 1, :),
        reshape(zs, 1, 1, :),
        [I_val])

    # Allocate output and evaluate at (0, 0, h)
    Bx = zeros(1); By = zeros(1); Bz = zeros(1)
    ForcingTerms.accumulate_strand_field!(
        Bx, By, Bz, 1,
        0.0, 0.0, h,
        xs, ys, zs,
        I_val * cs.nw
    )

    B_z_analytic = 4π * 1e-7 * I_val * a^2 / (2 * (a^2 + h^2)^(3/2))
    rel_error = abs(Bz[1] - B_z_analytic) / B_z_analytic
    @test rel_error < 0.005  # <0.5% with 500 segments

    # Transverse components should be near zero by symmetry
    @test abs(Bx[1]) < 1e-14
    @test abs(By[1]) < 1e-14
end

@testset "BiotSavart: compute_biot_savart_boundary! threading" begin
    # Same circular loop; verify threaded result matches single-obs result
    a = 0.8; I_val = 500.0
    n_seg = 200
    phis = range(0; length=n_seg + 1, stop=2π)
    xs = a .* cos.(phis); ys = a .* sin.(phis); zs = zeros(n_seg + 1)
    cs = ForcingTerms.CoilSet("loop", 1, 1, 1.0, n_seg + 1,
        reshape(xs, 1, 1, :), reshape(ys, 1, 1, :), reshape(zs, 1, 1, :),
        [I_val])

    # Three observation points
    obs_R   = [0.5, 1.0, 1.5]
    obs_phi = [0.0, π/3, π]
    obs_Z   = [0.3, 0.0, -0.2]
    B_R   = zeros(3); B_phi = zeros(3); B_Z = zeros(3)

    ForcingTerms.compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, [cs])

    # Verify B_Z at (0, 0, 0.3): analytic value
    Bx_s = zeros(1); By_s = zeros(1); Bz_s = zeros(1)
    obs_x = obs_R[1] * cos(obs_phi[1])
    obs_y = obs_R[1] * sin(obs_phi[1])
    ForcingTerms.accumulate_strand_field!(Bx_s, By_s, Bz_s, 1,
        obs_x, obs_y, obs_Z[1], xs, ys, zs, I_val * cs.nw)
    B_R_single   =  Bx_s[1] * cos(obs_phi[1]) + By_s[1] * sin(obs_phi[1])
    @test B_R[1] ≈ B_R_single  rtol=1e-10
    @test B_Z[1] ≈ Bz_s[1]    rtol=1e-10
end

# ---------------------------------------------------------------------------
@testset "CoilFourier: sample_boundary_grid" begin
    # Load Solovev example equilibrium
    equil_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
    @test isdir(equil_dir)

    inputs = TOML.parsefile(joinpath(equil_dir, "gpec.toml"))
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], equil_dir)
    equil = Equilibrium.setup_equilibrium(eq_config)

    mtheta = 48; nzeta = 16
    grid = ForcingTerms.sample_boundary_grid(equil, mtheta, nzeta)

    @test grid.mtheta == mtheta
    @test grid.nzeta  == nzeta
    @test length(grid.R) == mtheta
    @test length(grid.phi_grid) == nzeta

    # R should be positive and near the magnetic axis (Solovev: ro≈1.0, a≈0.33)
    @test all(grid.R .> 0)
    @test all(0.4 .< grid.R .< 2.0)  # physically reasonable for Solovev (r0=1.0, a=0.33)

    # Toroidal grid should span [0, 2π)
    @test grid.phi_grid[1] ≈ 0.0
    @test grid.phi_grid[end] ≈ 2π * (nzeta - 1) / nzeta
end

@testset "CoilFourier: fourier_decompose_bn roundtrip" begin
    # Synthesize bn(θ, ζ) = cos(m₀*θ - n₀*ζ), decompose, verify dominant mode
    mtheta = 128; nzeta = 64
    m0 = 5; n0 = 2

    theta_grid = range(0; length=mtheta, step=2π/mtheta)
    zeta_grid  = range(0; length=nzeta,  step=2π/nzeta)

    bn = [cos(m0 * θ - n0 * ζ) for θ in theta_grid, ζ in zeta_grid]

    # Build a dummy BoundaryGrid (R/Z/derivatives don't matter for this test)
    grid = ForcingTerms.BoundaryGrid(mtheta, nzeta,
        ones(mtheta), zeros(mtheta),
        collect(zeta_grid), ones(mtheta), zeros(mtheta))

    m_low = 1; m_high = 10
    modes = ForcingTerms.fourier_decompose_bn(bn, grid, n0, m_low, m_high)

    @test length(modes) == m_high - m_low + 1

    # Find the (m0, n0) mode
    idx = findfirst(m -> m.m == m0, modes)
    @test !isnothing(idx)
    @test abs(real(modes[idx].amplitude)) ≈ 1.0  rtol=0.02
    @test abs(imag(modes[idx].amplitude)) < 0.05

    # All other modes should be near zero
    for (i, mode) in enumerate(modes)
        i == idx && continue
        @test abs(mode.amplitude) < 0.05
    end
end

# ---------------------------------------------------------------------------
@testset "Integration: D3D I-coil field pipeline" begin
    # Skip if coil geometry files not present (shouldn't happen since bundled)
    @test isdir(COIL_DIR)
    @test isfile(D3D_IL)

    d3d_iu = joinpath(COIL_DIR, "d3d_iu.dat")
    @test isfile(d3d_iu)

    # Load DIII-D-like equilibrium
    equil_dir = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
    @test isdir(equil_dir)

    inputs2 = TOML.parsefile(joinpath(equil_dir, "gpec.toml"))
    eq_config2 = Equilibrium.EquilibriumConfig(inputs2["Equilibrium"], equil_dir)
    equil  = Equilibrium.setup_equilibrium(eq_config2)

    # Build coil sets manually (standard I-coil currents for RMP studies)
    il_cfg = ForcingTerms.CoilSetConfig(;
        dat_file = D3D_IL,
        currents = fill(10e3, 6),
    )
    iu_cfg = ForcingTerms.CoilSetConfig(;
        dat_file = d3d_iu,
        currents = fill(10e3, 6),
    )

    cfg = ForcingTerms.CoilConfig(;
        mtheta_coil = 96,
        nzeta_coil  = 64,
        coil_sets   = [il_cfg, iu_cfg],
    )

    il_set = ForcingTerms.read_coil_dat(D3D_IL)
    il_set = ForcingTerms.CoilSet(il_set.name, il_set.ncoil, il_set.s, il_set.nw, il_set.nsec,
        il_set.x, il_set.y, il_set.z, fill(10e3, 6))
    iu_set = ForcingTerms.read_coil_dat(d3d_iu)
    iu_set = ForcingTerms.CoilSet(iu_set.name, iu_set.ncoil, iu_set.s, iu_set.nw, iu_set.nsec,
        iu_set.x, iu_set.y, iu_set.z, fill(10e3, 6))

    forcing_modes = ForcingMode[]
    n_test = 3  # n=3 is a typical DIII-D RMP configuration
    m_low = 1; m_high = 15

    ForcingTerms.compute_coil_forcing_modes!(
        forcing_modes, [il_set, iu_set], equil, cfg, n_test, m_low, m_high;
        verbose=false
    )

    @test length(forcing_modes) == m_high - m_low + 1
    @test all(m.n == n_test for m in forcing_modes)

    # Field amplitudes should be non-trivial (I-coils with 10 kA give O(1e-3) T)
    amplitudes = abs.(getfield.(forcing_modes, :amplitude))
    @test maximum(amplitudes) > 1e-6

    # Dominant mode should be in the resonant range for DIII-D geometry
    dominant_m = forcing_modes[argmax(amplitudes)].m
    @test m_low <= dominant_m <= m_high
end
