using Test
using TOML
using Statistics
using HDF5
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
    @test cs.s == 1
    @test cs.nsec == 126
    @test cs.nw ≈ 1.0

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
    a = 1.0
    I_val = 1.0
    h = 0.5
    n_seg = 500

    # Build a discrete circular loop as a CoilSet (ncoil=1, s=1, nsec=n_seg+1)
    phis = range(0; length=n_seg + 1, stop=2π)
    xs = a .* cos.(phis)
    ys = a .* sin.(phis)
    zs = zeros(n_seg + 1)

    cs = ForcingTerms.CoilSet("test_loop", 1, 1, 1.0, n_seg + 1,
        reshape(xs, 1, 1, :),
        reshape(ys, 1, 1, :),
        reshape(zs, 1, 1, :),
        [I_val])

    # Allocate output and evaluate at (0, 0, h)
    Bx = zeros(1)
    By = zeros(1)
    Bz = zeros(1)
    ForcingTerms.accumulate_strand_field!(
        Bx, By, Bz, 1,
        0.0, 0.0, h,
        xs, ys, zs,
        I_val * cs.nw
    )

    B_z_analytic = 4π * 1e-7 * I_val * a^2 / (2 * (a^2 + h^2)^(3 / 2))
    rel_error = abs(Bz[1] - B_z_analytic) / B_z_analytic
    @test rel_error < 0.005  # <0.5% with 500 segments

    # Transverse components should be near zero by symmetry
    @test abs(Bx[1]) < 1e-14
    @test abs(By[1]) < 1e-14
end

@testset "BiotSavart: compute_biot_savart_boundary! threading" begin
    # Same circular loop; verify threaded result matches single-obs result
    a = 0.8
    I_val = 500.0
    n_seg = 200
    phis = range(0; length=n_seg + 1, stop=2π)
    xs = a .* cos.(phis)
    ys = a .* sin.(phis)
    zs = zeros(n_seg + 1)
    cs = ForcingTerms.CoilSet("loop", 1, 1, 1.0, n_seg + 1,
        reshape(xs, 1, 1, :), reshape(ys, 1, 1, :), reshape(zs, 1, 1, :),
        [I_val])

    # Three observation points
    obs_R = [0.5, 1.0, 1.5]
    obs_phi = [0.0, π / 3, π]
    obs_Z = [0.3, 0.0, -0.2]
    B_R = zeros(3)
    B_phi = zeros(3)
    B_Z = zeros(3)

    ForcingTerms.compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, [cs])

    # Verify B_Z at (0, 0, 0.3): analytic value
    Bx_s = zeros(1)
    By_s = zeros(1)
    Bz_s = zeros(1)
    obs_x = obs_R[1] * cos(obs_phi[1])
    obs_y = obs_R[1] * sin(obs_phi[1])
    ForcingTerms.accumulate_strand_field!(Bx_s, By_s, Bz_s, 1,
        obs_x, obs_y, obs_Z[1], xs, ys, zs, I_val * cs.nw)
    B_R_single = Bx_s[1] * cos(obs_phi[1]) + By_s[1] * sin(obs_phi[1])
    @test B_R[1] ≈ B_R_single rtol = 1e-10
    @test B_Z[1] ≈ Bz_s[1] rtol = 1e-10
end

# ---------------------------------------------------------------------------
@testset "CoilFourier: sample_boundary_grid" begin
    # Load Solovev example equilibrium
    equil_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
    @test isdir(equil_dir)

    inputs = TOML.parsefile(joinpath(equil_dir, "gpec.toml"))
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], equil_dir)
    equil = Equilibrium.setup_equilibrium(eq_config)

    mtheta = 48
    nzeta = 16
    grid = ForcingTerms.sample_boundary_grid(equil, mtheta, nzeta)

    @test grid.mtheta == mtheta
    @test grid.nzeta == nzeta
    @test length(grid.R) == mtheta
    @test length(grid.phi_grid) == nzeta

    # R should be positive and near the magnetic axis (Solovev: ro≈1.0, a≈0.33)
    @test all(grid.R .> 0)
    @test all(0.4 .< grid.R .< 2.0)  # physically reasonable for Solovev (r0=1.0, a=0.33)

    # Toroidal grid starts at 0 and spans one full toroidal turn.
    # Direction depends on helicity (sign(Bt)*sign(Ip)); test magnitude only.
    @test grid.phi_grid[1] ≈ 0.0
    @test abs(grid.phi_grid[end]) ≈ 2π * (nzeta - 1) / nzeta
end

@testset "CoilFourier: fourier_decompose_bn roundtrip" begin
    # Synthesize bn(θ, ζ) = cos(m₀*θ - n₀*ζ), decompose, verify dominant mode
    mtheta = 128
    nzeta = 64
    m0 = 5
    n0 = 2

    theta_grid = range(0; length=mtheta, step=2π / mtheta)
    zeta_grid = range(0; length=nzeta, step=2π / nzeta)

    bn = [cos(m0 * θ - n0 * ζ) for θ in theta_grid, ζ in zeta_grid]

    # Build a dummy BoundaryGrid (R/Z/derivatives don't matter for this test)
    grid = ForcingTerms.BoundaryGrid(mtheta, nzeta,
        ones(mtheta), zeros(mtheta),
        collect(zeta_grid), zeros(mtheta), ones(mtheta), zeros(mtheta))

    m_low = 1
    m_high = 10
    modes = ForcingTerms.fourier_decompose_bn(bn, grid, n0, m_low, m_high)

    @test length(modes) == m_high - m_low + 1

    # Find the (m0, n0) mode
    idx = findfirst(m -> m.m == m0, modes)
    @test !isnothing(idx)
    @test abs(real(modes[idx].amplitude)) ≈ 1.0 rtol = 0.02
    @test abs(imag(modes[idx].amplitude)) < 0.05

    # All other modes should be near zero
    for (i, mode) in enumerate(modes)
        i == idx && continue
        @test abs(mode.amplitude) < 0.05
    end
end

# ---------------------------------------------------------------------------
@testset "CoilFourier: project_normal_flux! R-factor" begin
    # A uniform vertical field B_Z = B0 on a circular boundary of radius `a`
    # centred at major radius R0.  In unit-norm convention the flux element is:
    #   Phi_x = 2π × R × (B_R × dZ/dθ_norm - B_Z × dR/dθ_norm)
    # where dR/dθ_norm = 2π × dR/dθ_phys = 2π × (-a*sin(θ)).
    # For B_R=0, B_Z=B0:  Phi_x = -2π × R × B0 × dR/dθ_norm = (2π)² × B0 × a × R(θ) × sin(θ)

    mtheta = 64
    nzeta = 1
    B0 = 1.0
    R0 = 2.0
    a = 0.5

    theta_phys = range(0; length=mtheta, step=2π / mtheta)
    R_arr = R0 .+ a .* cos.(theta_phys)
    Z_arr = a .* sin.(theta_phys)

    # Unit-norm derivatives: dR/dθ_norm = 2π × dR/dθ_phys
    dR_dθ_norm = 2π .* (-a .* sin.(theta_phys))
    dZ_dθ_norm = 2π .* (a .* cos.(theta_phys))

    phi_grid = [0.0]
    grid = ForcingTerms.BoundaryGrid(mtheta, nzeta, R_arr, Z_arr, phi_grid, zeros(mtheta), dR_dθ_norm, dZ_dθ_norm)

    # Uniform B_Z = B0, B_R = 0 at all points
    B_R = zeros(mtheta)
    B_Z = fill(B0, mtheta)
    bn = zeros(mtheta, nzeta)
    ForcingTerms.project_normal_flux!(bn, B_R, B_Z, grid)

    # Expected: 2π × R × (0×dZ_norm - B0×dR_norm) = (2π)² × B0 × a × R(θ) × sin(θ)
    expected = (2π)^2 .* B0 .* a .* R_arr .* sin.(theta_phys)
    @test bn[:, 1] ≈ expected rtol = 1e-10

    # Result must vary with R (old code without R factor gave constant amplitude)
    @test std(abs.(bn[:, 1])) > 0.01 * mean(abs.(bn[:, 1]))
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
    equil = Equilibrium.setup_equilibrium(eq_config2)

    # Build coil sets manually (standard I-coil currents for RMP studies)
    il_cfg = ForcingTerms.CoilSetConfig(;
        dat_file=D3D_IL,
        currents=fill(10e3, 6)
    )
    iu_cfg = ForcingTerms.CoilSetConfig(;
        dat_file=d3d_iu,
        currents=fill(10e3, 6)
    )

    cfg = ForcingTerms.CoilConfig(;
        mtheta_coil=96,
        nzeta_coil=64,
        coil_sets=[il_cfg, iu_cfg]
    )

    il_set = ForcingTerms.read_coil_dat(D3D_IL)
    il_set = ForcingTerms.CoilSet(il_set.name, il_set.ncoil, il_set.s, il_set.nw, il_set.nsec,
        il_set.x, il_set.y, il_set.z, fill(10e3, 6))
    iu_set = ForcingTerms.read_coil_dat(d3d_iu)
    iu_set = ForcingTerms.CoilSet(iu_set.name, iu_set.ncoil, iu_set.s, iu_set.nw, iu_set.nsec,
        iu_set.x, iu_set.y, iu_set.z, fill(10e3, 6))

    forcing_modes = ForcingMode[]
    n_test = 3  # n=3 is a typical DIII-D RMP configuration
    m_low = 1
    m_high = 15

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

# ---------------------------------------------------------------------------
@testset "Analytic coils: make_pf_hoop" begin
    a = 1.0
    I_val = 1.0
    h = 0.5
    cs = ForcingTerms.make_pf_hoop(; radius=a, height=0.0, nsec=501)
    @test size(cs.x) == (1, 1, 501)
    @test cs.ncoil == 1 && cs.s == 1
    # Closed loop: last point == first
    @test cs.x[1, 1, 1] ≈ cs.x[1, 1, end] atol = 1e-12
    @test cs.y[1, 1, 1] ≈ cs.y[1, 1, end] atol = 1e-12

    # Reproduce Jackson 5.38 on-axis field B_z = μ₀ I a² / (2 (a²+h²)^{3/2})
    Bx = zeros(1)
    By = zeros(1)
    Bz = zeros(1)
    ForcingTerms.accumulate_strand_field!(Bx, By, Bz, 1, 0.0, 0.0, h,
        cs.x[1, 1, :], cs.y[1, 1, :], cs.z[1, 1, :], I_val * cs.nw)
    B_z_analytic = 4π * 1e-7 * I_val * a^2 / (2 * (a^2 + h^2)^(3 / 2))
    @test abs(Bz[1] - B_z_analytic) / B_z_analytic < 0.005
    @test abs(Bx[1]) < 1e-14
    @test abs(By[1]) < 1e-14

    @test_throws ErrorException ForcingTerms.make_pf_hoop(; radius=-1.0, height=0.0)
    # nsec below the 361 floor (too coarse for a smooth poloidal loop) must error
    @test_throws ErrorException ForcingTerms.make_pf_hoop(; radius=a, height=0.0, nsec=100)
end

@testset "Analytic coils: make_window_pane" begin
    ncoil = 6
    R0 = 2.2
    Zlo = -0.6
    Zhi = 0.6
    cs = ForcingTerms.make_window_pane(; ncoil=ncoil, rz_corners=[[R0, Zlo], [R0, Zhi]],
        gap_fraction=0.15, phi0=0.0)
    @test cs.ncoil == ncoil && cs.s == 1

    # Every coil is a closed loop
    for j in 1:ncoil
        @test cs.x[j, 1, 1] ≈ cs.x[j, 1, end]
        @test cs.y[j, 1, 1] ≈ cs.y[j, 1, end]
        @test cs.z[j, 1, 1] ≈ cs.z[j, 1, end]
    end

    # Vertical frame at constant R: cylindrical radius ≈ R0, Z within [Zlo, Zhi]
    Rcyl = sqrt.(cs.x .^ 2 .+ cs.y .^ 2)
    @test all(isapprox.(Rcyl, R0; atol=1e-9))
    @test minimum(cs.z) ≈ Zlo
    @test maximum(cs.z) ≈ Zhi

    # Toroidal centers spaced 2π/ncoil: mean φ of each coil
    centers = [atan(mean(cs.y[j, 1, :]), mean(cs.x[j, 1, :])) for j in 1:ncoil]
    dphi = mod(centers[2] - centers[1], 2π)
    @test dphi ≈ 2π / ncoil rtol = 1e-6

    @test_throws ErrorException ForcingTerms.make_window_pane(; ncoil=2, rz_corners=[[1.0, 0.0]])
end

@testset "Analytic coils: make_helical" begin
    R0 = 1.7
    a = 0.6
    n_coils = 4
    # Full poloidal turn: single helical strand per coil
    cs = ForcingTerms.make_helical(; R0=R0, a=a, m=4, n=1, n_coils=n_coils)
    @test cs.ncoil == n_coils && cs.s == 1
    # Points lie on the circular cross-section torus (offset=0)
    Rcyl = sqrt.(cs.x .^ 2 .+ cs.y .^ 2)
    rminor = sqrt.((Rcyl .- R0) .^ 2 .+ cs.z .^ 2)
    @test all(isapprox.(rminor, a; atol=1e-9))

    # Partial poloidal range: closed loop with offset (theta in degrees)
    cp = ForcingTerms.make_helical(; R0=R0, a=a, m=3, n=1, n_coils=n_coils,
        theta_lo=-25.0, theta_hi=25.0, offset=0.05)
    for j in 1:n_coils
        @test cp.x[j, 1, 1] ≈ cp.x[j, 1, end] atol = 1e-9
        @test cp.z[j, 1, 1] ≈ cp.z[j, 1, end] atol = 1e-9
    end

    @test_throws ErrorException ForcingTerms.make_helical(; R0=R0, a=a, m=2, n=0, n_coils=2)
end

@testset "Analytic coils: window_pane standoff" begin
    equil_dir = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
    inputs = TOML.parsefile(joinpath(equil_dir, "gpec.toml"))
    equil = Equilibrium.setup_equilibrium(Equilibrium.EquilibriumConfig(inputs["Equilibrium"], equil_dir))

    # surface_point_and_normal: outboard midplane (poloidal_angle = 0)
    R0, Z0, nR0, nZ0 = ForcingTerms.surface_point_and_normal(equil, 0.0)
    @test isapprox(Z0, equil.zo; atol=1e-3)
    @test R0 > equil.ro
    @test isapprox(hypot(nR0, nZ0), 1.0; atol=1e-9)   # unit normal
    @test nR0 > 0 && isapprox(nZ0, 0.0; atol=1e-3)     # points outward in +R

    # Top of the surface (poloidal_angle = 90°): normal points up
    _, Ztop, nRtop, nZtop = ForcingTerms.surface_point_and_normal(equil, deg2rad(90.0))
    @test Ztop > equil.zo
    @test nZtop > 0

    # make_window_pane_standoff: frame center sits ~standoff outside the surface
    standoff = 0.2
    cs = ForcingTerms.make_window_pane_standoff(equil; standoff=standoff, poloidal_angle=0.0,
        poloidal_length=0.6, ncoil=2)
    @test cs.ncoil == 2 && cs.s == 1
    # Centroid of the first frame, in (R, Z)
    Rc = mean(sqrt.(cs.x[1, 1, :] .^ 2 .+ cs.y[1, 1, :] .^ 2))
    Zc = mean(cs.z[1, 1, :])
    @test isapprox(Rc, R0 + standoff; atol=0.05)
    @test isapprox(Zc, Z0; atol=0.05)

    # With poloidal_tilt=0 at the outboard midplane the legs are ~vertical: corners differ in Z
    cs1 = ForcingTerms.make_window_pane_standoff(equil; standoff=standoff, poloidal_angle=0.0,
        poloidal_length=0.6, ncoil=1)
    Zspan = maximum(cs1.z[1, 1, :]) - minimum(cs1.z[1, 1, :])
    Rspan = maximum(sqrt.(cs1.x[1, 1, :] .^ 2 .+ cs1.y[1, 1, :] .^ 2)) -
            minimum(sqrt.(cs1.x[1, 1, :] .^ 2 .+ cs1.y[1, 1, :] .^ 2))
    @test Zspan > Rspan

    # Winding convention: the first poloidal leg (points 1:np_pol, the c1->c2 leg) runs
    # lower-Z -> higher-Z, matching make_window_pane's c1->c2 ordering, so +current -> +b_n_x
    # near the coil (same sense as the corner-built window pane).
    np_pol = 20  # make_window_pane default
    @test cs1.z[1, 1, np_pol] > cs1.z[1, 1, 1]

    # Separatrix-touch guard: a 90° tilt with standoff shorter than half the leg length pushes
    # one corner back through the boundary, which must error. A normal placement still succeeds.
    @test_throws ErrorException ForcingTerms.make_window_pane_standoff(equil; standoff=0.1,
        poloidal_angle=0.0, poloidal_length=1.0, poloidal_tilt=90.0, ncoil=1)
    @test ForcingTerms.make_window_pane_standoff(equil; standoff=0.3, poloidal_angle=0.0,
        poloidal_length=0.4, poloidal_tilt=0.0, ncoil=1) isa ForcingTerms.CoilSet
end

# ---------------------------------------------------------------------------
@testset "Coil HDF5: round-trip vs .dat" begin
    cs0 = ForcingTerms.read_coil_dat(D3D_IL)
    mktempdir() do dir
        h5 = joinpath(dir, "il.h5")
        ForcingTerms.convert_coil_dat_to_h5(D3D_IL, h5)
        sets = ForcingTerms.read_coil_h5(h5)
        @test length(sets) == 1
        cs = sets[1]
        @test cs.ncoil == cs0.ncoil
        @test cs.s == cs0.s
        @test cs.nsec == cs0.nsec
        @test cs.nw ≈ cs0.nw
        @test cs.x ≈ cs0.x
        @test cs.y ≈ cs0.y
        @test cs.z ≈ cs0.z

        # h5 -> dat round-trips back to identical geometry
        dat2 = joinpath(dir, "il_back.dat")
        ForcingTerms.convert_coil_h5_to_dat(h5, dat2)
        csb = ForcingTerms.read_coil_dat(dat2)
        @test csb.x ≈ cs0.x && csb.y ≈ cs0.y && csb.z ≈ cs0.z
    end
end

@testset "Coil HDF5: multi-set + content tolerance" begin
    cs0 = ForcingTerms.read_coil_dat(D3D_IL)
    hoop = ForcingTerms.make_pf_hoop(; radius=2.0, height=0.1, name="hoop")
    mktempdir() do dir
        h5 = joinpath(dir, "multi.h5")
        ForcingTerms.write_coil_h5(h5, [cs0, hoop])

        # Inject unrelated content the reader must ignore
        HDF5.h5open(h5, "r+") do f
            f["coils/junk_dataset"] = [1.0, 2.0, 3.0]
            f["unrelated/other_stuff"] = 42
        end

        sets = ForcingTerms.read_coil_h5(h5)
        @test length(sets) == 2
        @test sort([s.name for s in sets]) == ["d3d_il", "hoop"]

        # Select a single set by name
        one = ForcingTerms.read_coil_h5(h5; set="hoop")
        @test length(one) == 1 && one[1].name == "hoop"
        @test_throws ErrorException ForcingTerms.read_coil_h5(h5; set="nope")
    end
end

# ---------------------------------------------------------------------------
@testset "load_coil_sets: source dispatch" begin
    # Analytic source: currents applied, tilt perturbs geometry
    cfg = ForcingTerms.CoilConfig(; coil_sets=[
        ForcingTerms.CoilSetConfig(; source="pf_hoop", radius=1.5, height=0.2,
            currents=[3.0], tiltx=[10.0])
    ])
    out = ForcingTerms.load_coil_sets(cfg, 1)
    @test length(out) == 1
    @test out[1].currents == [3.0]
    # A 10° X-tilt should move the (originally flat) hoop out of the z=0.2 plane
    @test !all(isapprox.(out[1].z, 0.2; atol=1e-6))

    # File source dispatches on extension: .h5 -> read_coil_h5, .dat -> read_coil_dat
    mktempdir() do dir
        h5 = joinpath(dir, "il.h5")
        ForcingTerms.convert_coil_dat_to_h5(D3D_IL, h5)
        cfg_h5 = ForcingTerms.CoilConfig(; coil_sets=[
            ForcingTerms.CoilSetConfig(; source="file", h5_file=h5, currents=fill(1.0, 6))
        ])
        cfg_dat = ForcingTerms.CoilConfig(; coil_sets=[
            ForcingTerms.CoilSetConfig(; dat_file=D3D_IL, currents=fill(1.0, 6))
        ])
        a = ForcingTerms.load_coil_sets(cfg_h5, 1)
        b = ForcingTerms.load_coil_sets(cfg_dat, 1)
        @test a[1].ncoil == 6 && b[1].ncoil == 6
        @test a[1].x ≈ b[1].x
    end

    @test_throws ErrorException ForcingTerms.load_coil_sets(
        ForcingTerms.CoilConfig(; coil_sets=[ForcingTerms.CoilSetConfig(; source="bogus")]), 1)

    # window_pane standoff mode requires an equilibrium
    cfg_standoff = ForcingTerms.CoilConfig(;
        coil_sets=[
            ForcingTerms.CoilSetConfig(; source="window_pane", ncoil_gen=2,
                standoff=0.2, poloidal_angle=0.0, poloidal_length=0.6, currents=[1.0, -1.0])
        ]
    )
    @test_throws ErrorException ForcingTerms.load_coil_sets(cfg_standoff, 1)

    # Supplying both rz_corners and standoff is ambiguous
    cfg_both = ForcingTerms.CoilConfig(;
        coil_sets=[
            ForcingTerms.CoilSetConfig(; source="window_pane", ncoil_gen=2,
                rz_corners=[[2.2, -0.5], [2.2, 0.5]], standoff=0.2, poloidal_angle=0.0)
        ]
    )
    @test_throws ErrorException ForcingTerms.load_coil_sets(cfg_both, 1)
end

@testset "Coil HDF5 group: save/load round-trip" begin
    sets_in = [ForcingTerms.make_pf_hoop(; radius=1.2, height=0.0, name="hoopA"),
        ForcingTerms.make_window_pane(; ncoil=3, rz_corners=[[2.0, -0.5], [2.0, 0.5]], name="wp")]
    sets_in[1] = ForcingTerms.CoilSet(sets_in[1].name, sets_in[1].ncoil, sets_in[1].s,
        sets_in[1].nw, sets_in[1].nsec, sets_in[1].x, sets_in[1].y, sets_in[1].z, [7.0])
    mktempdir() do dir
        path = joinpath(dir, "snap.h5")
        HDF5.h5open(path, "w") do f
            ForcingTerms.save_coils_to_h5(sets_in, HDF5.create_group(f, "input/raw_inputs/coils"))
        end
        sets_out = ForcingTerms.CoilSet[]
        HDF5.h5open(path, "r") do f
            ForcingTerms.load_coils_from_h5_group!(sets_out, f["input/raw_inputs/coils"])
        end
        @test length(sets_out) == 2
        byname = Dict(s.name => s for s in sets_out)
        @test byname["hoopA"].currents == [7.0]
        @test byname["hoopA"].x ≈ sets_in[1].x
        @test byname["wp"].ncoil == 3
    end
end
