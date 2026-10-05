using TOML

# The tolerance TOML schema: the fixture parses into the expected structs with defaults and
# unit conversions applied, every validation rule throws on a bad file, name checks against a
# run's coil sets work, and the metres-to-degrees tilt conversion matches apply_transforms.
@testset "tolerance TOML" begin
    GPEC = GeneralizedPerturbedEquilibrium
    EF = GPEC.ErrorFields
    FT = GPEC.ForcingTerms
    fixture = joinpath(@__DIR__, "test_data", "ErrorFields", "tolerances_two_hoops.toml")

    @testset "fixture parses with defaults and units" begin
        ts = EF.read_tolerance_toml(fixture)
        @test ts isa EF.ToleranceSet
        @test ts.raw == read(fixture, String)
        @test [c.name for c in ts.coils] == ["hoop_tilted", "hoop_axi"]

        c1, c2 = ts.coils
        @test c1.shift_tol_m ≈ 0.5e-3
        @test c1.shift_sigma_m ≈ 0.1e-3          # from [ErrorFields.defaults]
        @test c1.tilt_tol == 0.002 && c1.tilt_units == "m" && c1.tilt_sigma == 0.0005
        @test c1.radial_shape == "flat"          # from [ErrorFields.defaults]
        @test c1.tolerance_model == "additive" && c1.cylinder_half_height_m == 0.0
        @test c2.tolerance_model == "cylinder" && c2.cylinder_half_height_m == 1.5
        @test c2.radial_shape == "hollow" && c2.tilt_units == "deg" && c2.shift_tol_m ≈ 2e-3

        @test length(ts.groups) == 1
        g = ts.groups[1]
        @test g.name == "both_hoops" && g.members == ["hoop_tilted", "hoop_axi"]
        @test g.shift_tol_m ≈ 1e-3 && g.tilt_tol == 0.01 && g.tilt_units == "deg"
        @test g.phase_group == "support" && g.rotation_center_z_m == 0.2
        @test g.radial_shape == "flat"           # defaults apply to groups too
        @test ts.uncorrectable_coils == ["hoop_axi"] && ts.efc_factor == 2.0
        @test ts.other_field.magnitude == 8.2e-6 && ts.other_field.sigma == 1e-6 && ts.other_field.radial_shape == "ring"

        # Defaults when tables are absent, and the phase group defaulting to the group's name.
        minimal = EF.parse_tolerance_toml("""
            [[ErrorFields.coil]]
            name = "a"
            shift_tol_mm = 1.0
            [[ErrorFields.coherent_group]]
            name = "g"
            members = ["a"]
            """)
        @test minimal.coils[1].radial_shape == "hollow" && minimal.coils[1].tilt_units == "deg"
        @test minimal.coils[1].tilt_tol == 0.0 && minimal.coils[1].shift_sigma_m == 0.0
        @test minimal.groups[1].phase_group == "g" && minimal.groups[1].rotation_center_z_m == 0.0
        @test minimal.efc_factor == 2.0 && isempty(minimal.uncorrectable_coils)
        @test minimal.other_field.magnitude == 0.0 && minimal.other_field.radial_shape == "ring"
    end

    @testset "validation errors" begin
        bad(text) = @test_throws ArgumentError EF.parse_tolerance_toml(text)
        bad("[[coil]]\nname = \"a\"\n")                                          # no [ErrorFields] prefix
        bad("[[ErrorFields.coil]]\nshift_tol_mm = 1.0\n")                       # unnamed coil
        bad("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol = 1.0\n")            # misspelled key
        bad("[[ErrorFields.coil]]\nname = \"a\"\nshift_tol_mm = -1.0\n")        # negative tolerance
        bad("[[ErrorFields.coil]]\nname = \"a\"\nradial_shape = \"gaussian\"\n") # unknown shape
        bad("[[ErrorFields.coil]]\nname = \"a\"\ntilt_units = \"rad\"\n")       # unknown units
        bad("[[ErrorFields.coil]]\nname = \"a\"\ntolerance_model = \"cylinder\"\n") # cylinder without height
        bad("[[ErrorFields.coil]]\nname = \"a\"\n[[ErrorFields.coil]]\nname = \"a\"\n") # duplicate
        bad("[[ErrorFields.coherent_group]]\nname = \"g\"\n")                    # no members
        bad("[[ErrorFields.coherent_group]]\nname = \"g\"\nmembers = [\"a\", \"a\"]\n") # duplicate member
        bad("[ErrorFields.correctability]\nefc_factor = 0.5\n")                  # efc_factor < 1
        bad("[ErrorFields.defaults]\nname = \"x\"\n")                            # name is not a default
        bad("[ErrorFields.other_field]\nmagnitude = -1.0\n")
        # Case-insensitive enumerations are normalized.
        @test EF.parse_tolerance_toml("[[ErrorFields.coil]]\nname = \"a\"\nradial_shape = \"Hollow\"\n").coils[1].radial_shape == "hollow"
    end

    @testset "name validation against a run" begin
        ts = EF.read_tolerance_toml(fixture)
        @test EF.validate_tolerances(ts, ["hoop_tilted", "hoop_axi", "c"]) === ts
        err = try
            EF.validate_tolerances(ts, ["hoop_tilted"])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("coil hoop_axi", err.msg) && occursin("member hoop_axi", err.msg) && occursin("uncorrectable coil hoop_axi", err.msg)
    end

    @testset "tilt units through the nominal radius" begin
        hoop = FT.make_pf_hoop(; radius=1.5, height=0.4, name="hoop")
        # A 361-point polygon sits at R·cos(π/360) on its chord midpoints, 4e-5 inside the circle.
        @test FT.nominal_major_radius(hoop) ≈ 1.5 rtol = 1e-4
        @test EF.tilt_tolerance_deg(0.01, "deg", hoop) == 0.01
        @test EF.tilt_tolerance_deg(0.002, "m", hoop) ≈ rad2deg(asin(0.002 / 1.5)) rtol = 1e-4
        @test_throws ArgumentError EF.tilt_tolerance_deg(2.0, "m", hoop)
        ts = EF.read_tolerance_toml(fixture)
        @test EF.tilt_tolerance_deg(ts.coils[1], hoop) ≈ rad2deg(asin(0.002 / 1.5)) rtol = 1e-4
        # The conversion is the one apply_transforms uses for tilt_in_meters: a rim displacement
        # of t metres tilts the hoop by the same angle either way.
        by_meters = FT.apply_transforms(hoop, FT.CoilSetConfig(; tiltx=[0.002], tilt_in_meters=true); n_tilt=1)
        by_degrees = FT.apply_transforms(hoop, FT.CoilSetConfig(; tiltx=[EF.tilt_tolerance_deg(0.002, "m", hoop)]); n_tilt=1)
        @test by_meters.z ≈ by_degrees.z atol = 1e-12
    end
end
