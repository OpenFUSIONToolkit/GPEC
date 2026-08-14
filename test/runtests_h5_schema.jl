using HDF5

# Schema-naming guard for gpec.h5 (anti-drift enforcement for
# docs/development/hdf5-conventions.md): every group-path component must be
# CamelCase or a whitelisted data-driven token. Dataset (leaf) names are not
# constrained here; the metadata contract is asserted separately.

_is_camelcase(name) = occursin(r"^[A-Z][A-Za-z0-9]*$", name)

# Data-driven group names are stored verbatim: anything inside the Input/ raw
# snapshot (coil-set names, ingest layout), KineticForces method tokens, and
# Scan surface indices.
function _group_name_ok(parent_path, name)
    startswith(parent_path, "Input/") && return true
    parent_path == "KineticForces" && return true
    occursin(r"^Surface_\d+$", name) && return true
    return _is_camelcase(name)
end

function _collect_bad_groups(h5)
    bad = String[]
    function walk(node, prefix)
        for k in keys(node)
            child = node[k]
            child isa HDF5.Group || continue
            full = isempty(prefix) ? k : prefix * "/" * k
            _group_name_ok(prefix, k) || push!(bad, full)
            walk(child, full)
        end
    end
    walk(h5, "")
    return bad
end

# Metadata contract (docs/development/hdf5-conventions.md): every dataset carries
# long_name + units, and rank ≥ 2 datasets carry a dims axis-name attribute. Exempt:
# the Input/ raw snapshot and the debug-only GalerkinIntegration Match/ group.
_metadata_exempt(path) = startswith(path, "Input/") || occursin("/Match/", path)

function _collect_metadata_violations(h5)
    bad = String[]
    function walk(node, prefix)
        for k in keys(node)
            child = node[k]
            full = isempty(prefix) ? k : prefix * "/" * k
            if child isa HDF5.Group
                walk(child, full)
            elseif !_metadata_exempt(full)
                a = attrs(child)
                haskey(a, "long_name") || push!(bad, "$full: missing long_name")
                haskey(a, "units") || push!(bad, "$full: missing units")
                ndims(child) >= 2 && !haskey(a, "dims") && push!(bad, "$full: missing dims")
            end
        end
    end
    walk(h5, "")
    return bad
end

# The full-run walk below only exercises an ideal deck, which never produces the
# data-driven group names — pin the whitelist rules directly.
@testset "gpec.h5 schema naming: group-name rule" begin
    @test _group_name_ok("KineticForces", "fgar")            # method tokens verbatim
    @test _group_name_ok("Tearing/Scan", "Surface_1")        # scan indices verbatim
    @test _group_name_ok("Input/RawInputs/Coils", "my_coils") # raw-snapshot names verbatim
    @test _group_name_ok("", "SingularSurfaces")
    @test !_group_name_ok("", "singular")
    @test !_group_name_ok("SingularSurfaces", "kinetic")
end

@testset "gpec.h5 schema naming" begin
    template_dir = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")

    mktempdir() do run_dir
        for name in readdir(template_dir)
            cp(joinpath(template_dir, name), joinpath(run_dir, name))
        end
        toml_path = joinpath(run_dir, "gpec.toml")
        write(toml_path, replace(read(toml_path, String), "write_outputs_to_HDF5 = false" => "write_outputs_to_HDF5 = true"))

        GeneralizedPerturbedEquilibrium.main([run_dir])
        h5_path = joinpath(run_dir, "gpec.h5")
        @test isfile(h5_path)

        h5open(h5_path, "r") do h5
            bad = _collect_bad_groups(h5)
            isempty(bad) || @error "non-CamelCase group paths in gpec.h5" bad
            @test isempty(bad)

            # The CamelCase walk above already forbids every lowercase legacy group; these
            # two moved under ForceFreeStates/ and are CamelCase, so pin them explicitly.
            @test !haskey(h5, "FreeBoundaryStability")
            @test !haskey(h5, "EdgeScan")

            # Inputs live only under Input/; spot-check the rerun-critical paths.
            @test haskey(h5, "Input/gpec_toml_raw")
            @test haskey(h5, "Info/git_version")

            # Metadata contract: long_name/units everywhere, dims on rank ≥ 2 arrays.
            viol = _collect_metadata_violations(h5)
            isempty(viol) || @error "metadata contract violations in gpec.h5" viol
            @test isempty(viol)

            # File-level attributes.
            ra = attrs(h5)
            for k in ("schema_version", "Conventions", "title", "date_created")
                @test haskey(ra, k)
            end
            @test ra["schema_version"] == "2.0"

            # Dimension scales: the ψ_N coordinate of the forward integration is a
            # scale and is attached to its q profile (netCDF-4 pattern).
            fwd = "ForceFreeStates/Solutions/ForwardIntegration"
            @test HDF5.API.h5ds_is_scale(h5["$fwd/psi"])
            @test HDF5.API.h5ds_is_attached(h5["$fwd/q"], h5["$fwd/psi"], 0)
        end
    end
end
