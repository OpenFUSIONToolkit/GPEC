using Test
using HDF5
using LinearAlgebra
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms

const MU0 = 4π * 1e-7

# The 12 outward, counterclockwise triangles of a box with corners 1(−,−,−) 2(+,−,−) 3(+,+,−) 4(−,+,−) 5..8 (same, +z).
const BOX_TRIANGLES = [1 4 3; 1 3 2; 5 6 7; 5 7 8; 1 2 6; 1 6 5; 2 3 7; 2 7 6; 3 4 8; 3 8 7; 4 1 5; 4 5 8]'

# Box corners [3, 8] with half-sizes `h`, rotated by `Q` and centred at `c`.
box_vertices(h; c=zeros(3), Q=I(3)) =
    reduce(hcat, [c .+ Q * (h .* s) for s in ([-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1])])

# One magnetic material from a list of boxes, each `(h, c, Q)`.
function box_material(boxes; saturation_field=1.7, fill_fraction=1.0)
    vertices = reduce(hcat, [box_vertices(h; c, Q) for (h, c, Q) in boxes])
    triangles = reduce(hcat, [BOX_TRIANGLES .+ 8(i - 1) for i in eachindex(boxes)])
    return MagneticMaterial("boxes", vertices, triangles, collect(0:12:12length(boxes)), saturation_field, fill_fraction)
end

# Cylindrical (B_R, B_phi, B_Z) of `material` magnetized by `M` [3, nelement] at Cartesian points `obs` [3, nobs].
function material_field(material, M, obs)
    B = (zeros(size(obs, 2)), zeros(size(obs, 2)), zeros(size(obs, 2)))
    compute_magnetic_material_field!(B..., hypot.(obs[1, :], obs[2, :]), atan.(obs[2, :], obs[1, :]), obs[3, :], material, M)
    return B
end

# The same field as Cartesian vectors, one column per point.
function cartesian_field(material, M, obs)
    BR, Bφ, BZ = material_field(material, M, obs)
    φ = atan.(obs[2, :], obs[1, :])
    return vcat((BR .* cos.(φ) .- Bφ .* sin.(φ))', (BR .* sin.(φ) .+ Bφ .* cos.(φ))', BZ')
end

# Observation points on a sphere of radius d about `c`.
sphere_points(d; c=zeros(3)) = reduce(hcat, [c .+ d .* [sin(t) * cos(p), sin(t) * sin(p), cos(t)] for t in (0.3, 1.1, 1.9, 2.8) for p in (0.2, 1.7, 3.4, 5.0)])

@testset "Magnetic material" begin
    h = [0.29, 0.235, 0.0195]                    # an ITER insert plate, 0.58 × 0.47 × 0.039 m
    plate = box_material([(h, zeros(3), I(3))])
    Ms = 1.7 / MU0

    @testset "element volume and centroid from the closed surface" begin
        shifted = box_material([(h, [9.0, 1.0, 0.5], I(3)), (2h, [-3.0, 0.0, 2.0], I(3))])
        @test shifted.volume ≈ [8prod(h), 64prod(h)] rtol = 1e-12
        @test shifted.centroid ≈ [9.0 -3.0; 1.0 0.0; 0.5 2.0] atol = 1e-12
    end

    @testset "face charges: closed form against quadrature, at close range and far" begin
        # B = μ₀/4π Σ σ ∫(r − r′)/|r − r′|³ dA′ by midpoint quadrature on each face, with an oblique M charging all six.
        M = Ms .* normalize([0.3, -0.5, 0.8])
        V = box_vertices(h)
        function quadrature(r; n=600)
            H = zeros(3)
            for f in 1:2:12
                # Triangles (a, b, c) and (a, c, d) make the face a-b-c-d; b - a and d - a span it, outward-oriented.
                a, b, d = V[:, BOX_TRIANGLES[1, f]], V[:, BOX_TRIANGLES[2, f]], V[:, BOX_TRIANGLES[3, f+1]]
                e1, e2 = b .- a, d .- a
                σ, dA = dot(M, normalize(cross(e1, e2))), norm(cross(e1, e2)) / n^2
                for i in 1:n, j in 1:n
                    s = r .- (a .+ ((i - 0.5) / n) .* e1 .+ ((j - 0.5) / n) .* e2)
                    H .+= σ * dA .* s ./ norm(s)^3
                end
            end
            return MU0 / 4π .* H
        end
        obs = [0.05 0.40 -0.20 1.5; -0.03 0.10 0.30 -0.8; 0.06 -0.05 0.10 0.9]   # 4 cm above the plate, beside it, and far
        Bq = reduce(hcat, [quadrature(obs[:, k]) for k in axes(obs, 2)])
        @test cartesian_field(plate, reshape(M, 3, 1), obs) ≈ Bq rtol = 1e-4
    end

    @testset "uniform M equals its bound surface current: against the coil kernel" begin
        # Magnetized along a box axis, the box is a sheet of M amperes per metre on the faces parallel to M (K = M × n̂).
        obs = sphere_points(0.6)
        for axis in (1, 3)
            i, j = mod1(axis + 1, 3), mod1(axis + 2, 3)
            nloop = 2000
            pts(s, u, v) = (p = zeros(3); p[i] = u * h[i]; p[j] = v * h[j]; p[axis] = -h[axis] + (s - 0.5) * 2h[axis] / nloop; p)
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
            X, Y, Z = (reshape([pts(s, u, v)[k] for s in 1:nloop, (u, v) in corners], nloop, 1, 5) for k in 1:3)
            sheet = CoilSet("sheet", nloop, 1, 1.0, 5, X, Y, Z, fill(Ms * 2h[axis] / nloop, nloop))
            Bc = (zeros(size(obs, 2)), zeros(size(obs, 2)), zeros(size(obs, 2)))
            compute_biot_savart_boundary!(Bc..., hypot.(obs[1, :], obs[2, :]), atan.(obs[2, :], obs[1, :]), obs[3, :], [sheet])
            M = zeros(3, 1)
            M[axis] = Ms
            Bm = material_field(plate, M, obs)
            scale = maximum(maximum(abs, b) for b in Bc)
            @test maximum(maximum(abs, bm .- bc) for (bm, bc) in zip(Bm, Bc)) < 1e-6 * scale
        end
    end

    @testset "far away a magnetized element is the point dipole m = M·V" begin
        M = Ms .* normalize([1.0, 0.4, -0.7])
        m = M .* plate.volume[1]
        function dipole_error(d)
            obs = sphere_points(d)
            Bm = cartesian_field(plate, reshape(M, 3, 1), obs)
            Bd = reduce(hcat, [MU0 / 4π .* (3dot(m, r) .* r ./ norm(r)^5 .- m ./ norm(r)^3) for r in eachcol(obs)])
            return maximum(norm.(eachcol(Bm .- Bd))) / maximum(norm.(eachcol(Bd)))
        end
        e3, e30 = dipole_error(3.0), dipole_error(30.0)
        @test e30 < 1e-3
        @test e30 < e3 / 50      # the finite-size correction falls as (size/d)²
    end

    @testset "a toroidally magnetized ring of plates drives only n = k·N, and vanishes as N grows" begin
        # An axisymmetric body magnetized along φ̂ has no external field; only its discreteness drives one.
        R0, R1, Z1 = 2.0, 1.4, 0.3
        function ring_field(N, nobs)
            φs = 2π .* (0:N-1) ./ N
            rot(φ) = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1]
            ring = box_material([([0.04, 2π * R0 / (3N), 0.03], R0 .* [cos(φ), sin(φ), 0.0], rot(φ)) for φ in φs])
            M = Ms .* vcat((-sin.(φs))', cos.(φs)', zeros(1, N))
            ψ = 2π .* (0:nobs-1) ./ nobs
            B = (zeros(nobs), zeros(nobs), zeros(nobs))
            compute_magnetic_material_field!(B..., fill(R1, nobs), collect(ψ), fill(Z1, nobs), ring, M)
            return ψ, B
        end
        harmonic(ψ, f, n) = abs(sum(f .* cis.(-n .* ψ))) / length(ψ)
        N = 12
        ψ, B = ring_field(N, 8N)
        for f in B
            @test all(harmonic(ψ, f, n) < 1e-10 * maximum(abs, f) for n in 1:N-1)
        end
        @test harmonic(ψ, B[1], N) > 1e-3 * maximum(abs, B[1])
        peak(N) = maximum(maximum(abs, f) for f in ring_field(N, 8N)[2])
        @test peak(24) < peak(12) / 10
    end

    @testset "construction checks, HDF5 round trip and deck blocks" begin
        V = box_vertices(h)
        @test_throws "not a closed" MagneticMaterial("open", V, BOX_TRIANGLES[:, 1:11], [0, 11], 1.7, 1.0)
        @test_throws "must face outward" MagneticMaterial("inside-out", V, BOX_TRIANGLES[[1, 3, 2], :], [0, 12], 1.7, 1.0)
        @test_throws "element_offsets" MagneticMaterial("x", V, BOX_TRIANGLES, [0, 11], 1.7, 1.0)
        @test_throws "fill_fraction" MagneticMaterial("x", V, BOX_TRIANGLES, [0, 12], 1.7, 1.5)
        @test_throws "saturation_field" MagneticMaterial("x", V, BOX_TRIANGLES, [0, 12], 0.0, 1.0)
        @test_logs (:warn, r"M_s in A/m") MagneticMaterial("x", V, BOX_TRIANGLES, [0, 12], 1.4e6, 1.0)

        dir = mktempdir()
        two = box_material([(h, zeros(3), I(3)), (h, [1.0, 0, 0], I(3))]; saturation_field=1.6, fill_fraction=0.9)
        path = write_magnetic_material_h5(joinpath(dir, "material.h5"), [two])
        h5open(path) do f
            @test minimum(read(f, "magnetic_material/boxes/triangles")) == 0      # 0-based on disk
        end
        back = only(read_magnetic_material_h5(path; set="boxes"))
        @test back.vertices == two.vertices && back.triangles == two.triangles && back.element_offsets == two.element_offsets
        @test back.saturation_field == 1.6 && back.fill_fraction == 0.9 && back.volume ≈ two.volume
        @test_throws "not found" read_magnetic_material_h5(path; set="missing")

        # A geometry-only file, as a CAD export writes it, takes its material from the deck block.
        geometry = joinpath(dir, "geometry.h5")
        h5open(geometry, "w") do f
            g = create_group(create_group(f, "magnetic_material"), "plate")
            g["vertices"], g["triangles"], g["element_offsets"] = V, BOX_TRIANGLES .- 1, [0, 12]
        end
        cfg = ForcingTerms._parse_magnetic_material_config(Dict{String,Any}("name" => "plate", "h5_file" => geometry, "saturation_field" => 1.6))
        loaded = load_magnetic_material(cfg)
        @test loaded.saturation_field == 1.6 && loaded.fill_fraction == 1.0 && loaded.volume ≈ [8prod(h)]
        @test_throws "unknown keys" ForcingTerms._parse_magnetic_material_config(Dict{String,Any}("name" => "plate", "mu0Ms" => 1.6))
        @test_throws "needs a saturation_field" load_magnetic_material(MagneticMaterialConfig(; name="plate", h5_file=geometry))
    end

    @testset "a point inside an element is an error" begin
        @test_throws "lies inside element 1" material_field(plate, zeros(3, 1) .+ [0.0, 0.0, Ms], reshape([0.01, -0.02, 0.005], 3, 1))
    end

    @testset "saturated along the equilibrium field, driving only n = k·N from a ring" begin
        EQ = GeneralizedPerturbedEquilibrium.Equilibrium
        FT = GeneralizedPerturbedEquilibrium.ForcingTerms
        gfile = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "TkMkr_D3Dlike_Hmode.geqdsk")
        equil = EQ.PlasmaEquilibrium(
            EQ.EquilibriumConfig(; eq_filename=gfile, eq_type="efit", jac_type="hamada",
                grid_type="core_edge_packed", psilow=0.01, psihigh=0.995, mpsi=64, mtheta=256)
        )
        rot(φ) = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1]
        R0, Z0, hb = 2.38, equil.zo, [0.015, 0.04, 0.05]     # outboard, between the separatrix and the flux grid's edge
        ring(N) = box_material([(hb, rot(φ) * [R0, 0.0, Z0], rot(φ)) for φ in 2π .* (0:N-1) ./ N]; saturation_field=1.6, fill_fraction=0.8)

        one = box_material([(hb, [R0, 0.0, Z0], I(3))]; saturation_field=1.6, fill_fraction=0.8)
        M = saturated_magnetization(one, equil)
        B0 = EQ.equilibrium_field(equil, [R0], [Z0])
        @test norm(M) ≈ 0.8 * 1.6 / MU0
        @test M[:, 1] ./ norm(M) ≈ normalize([B0.B_R[1], B0.B_phi[1], B0.B_Z[1]])     # at φ = 0, (x, y, z) = (R, φ, Z)
        inside = box_material([(hb, [equil.ro + 0.3, 0.0, equil.zo], I(3))])
        @test_throws "inside the separatrix" saturated_magnetization(inside, equil)
        @test_throws "outside the equilibrium's flux grid" saturated_magnetization(box_material([(hb, [3.0, 0.0, 0.0], I(3))]), equil)

        # Axisymmetric B₀ makes every element of a ring identical up to rotation, so only n = k·N survives.
        N = 8
        mats = [ring(N)]
        mags = [saturated_magnetization(mats[1], equil)]
        cfg = FT.CoilConfig(; mtheta_coil=128)
        psi = equil.rzphi_xs[end]
        peak(n) = maximum(abs(mode.amplitude) for mode in material_forcing_modes(mats, mags, equil, cfg, n, -10, 20; psi))
        @test peak(N) > 0
        @test peak(1) < 1e-8 * peak(N) && peak(3) < 1e-8 * peak(N)
        # Breaking the ring by removing one element drives n = 1.
        broken = box_material([(hb, rot(φ) * [R0, 0.0, Z0], rot(φ)) for φ in 2π .* (1:N-1) ./ N]; saturation_field=1.6, fill_fraction=0.8)
        @test maximum(abs(m.amplitude) for m in material_forcing_modes([broken], [saturated_magnetization(broken, equil)], equil, cfg, 1, -10, 20; psi)) >
              1e-3 * peak(N)
    end

    @testset "a ForcingField leaf, from the API and from a deck" begin
        GPEC = GeneralizedPerturbedEquilibrium
        dir = mktempdir()
        geometry = write_magnetic_material_h5(joinpath(dir, "plates.h5"), [box_material([(h, [9.0, 0.0, 0.0], I(3))])])
        leaf = ForcingField(only(read_magnetic_material_h5(geometry)); nzeta_coil=64)
        @test leaf isa GPEC.ForcingTerms.ForcingMagneticMaterial && leaf.nzeta == 64 && leaf.mtheta == 480
        @test (2im * leaf).scale == 2im && repr(leaf) == "ForcingField(magnetic material boxes)"
        write(joinpath(dir, "forcing.dat"), "1 2 1e-4 0.0\n")
        block = Dict{String,Any}("name" => "boxes", "h5_file" => "plates.h5", "saturation_field" => 1.7, "fill_fraction" => 0.5)
        ctrl = GPEC.forcing_terms_control(
            Dict{String,Any}("ForcingTerms" => Dict{String,Any}("forcing_data_file" => "forcing.dat",
                "mtheta_coil" => 96, "magnetic_material" => [block])), dir)
        field = ForcingField(ctrl)
        @test [typeof(t) for t in field.terms] == [GPEC.ForcingTerms.ForcingFile, GPEC.ForcingTerms.ForcingMagneticMaterial]
        material = field.terms[2]
        @test material.mtheta == 96 && material.nzeta == 0
        @test only(material.materials).saturation_field == 1.7 && only(material.materials).fill_fraction == 0.5
        @test GPEC._forcing_table(ctrl)["magnetic_material"][1]["h5_file"] == joinpath(dir, "plates.h5")
        @test only(material.groups) == ""
        grouped = GPEC.forcing_terms_control(
            Dict{String,Any}(
                "ForcingTerms" => Dict{String,Any}(
                    "magnetic_material" => [merge(block, Dict{String,Any}("group" => "tbm")), merge(block, Dict{String,Any}("fill_fraction" => 0.2))])
            ), dir)
        @test ForcingField(grouped).groups == ["tbm", ""]
        @test ForcingField([only(read_magnetic_material_h5(geometry))]; groups=["fi"]).groups == ["fi"]
        @test_throws "1 groups for 2 material sets" ForcingField(read_magnetic_material_h5(geometry)[[1, 1]]; groups=["fi"])
        @test_throws "unknown keys" GPEC.ForcingTerms._parse_magnetic_material_config(Dict{String,Any}("name" => "x", "grup" => "tbm"))
    end

    @testset "forcing sources: grouping, naming and the total" begin
        PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
        FM = GeneralizedPerturbedEquilibrium.ForcingTerms.ForcingMode
        spec(pairs...) = [FM(; n=n, m=m, amplitude=a) for (n, m, a) in pairs]
        a, b, c = spec((1, 2, 1.0), (1, 3, 2.0im)), spec((1, 2, 0.5), (1, 4, 1.0)), spec((1, 3, -1.0))
        grouped = PE._grouped_sources(["fi", "tbm", "fi"], [a, b, c], 2.0 + 0.0im)
        @test first.(grouped) == ["fi", "tbm"]
        @test [(m.m, m.amplitude) for m in grouped[1].second] == [(2, 2.0 + 0.0im), (3, -2.0 + 4.0im)]
        @test [(m.m, m.amplitude) for m in grouped[2].second] == [(2, 1.0 + 0.0im), (4, 2.0 + 0.0im)]
        total = PE._total_forcing(grouped)
        @test [(m.m, m.amplitude) for m in total] == [(2, 3.0 + 0.0im), (3, -2.0 + 4.0im), (4, 2.0 + 0.0im)]
        @test PE._unique_names(["c", "c", "fi", "c"]) == ["c", "c_2", "fi", "c_3"]
    end
end
