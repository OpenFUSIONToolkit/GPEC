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

            # Retired/renamed legacy top-level groups must not reappear.
            for legacy in ("info", "input", "equil", "splines", "integration", "locstab",
                "singular", "matrices", "kinetic", "galerkin", "slayer", "kinetic_forces",
                "perturbed_equilibrium", "vacuum", "FreeBoundaryStability", "EdgeScan")
                @test !haskey(h5, legacy)
            end

            # Inputs live only under Input/; spot-check the rerun-critical paths.
            @test haskey(h5, "Input/gpec_toml_raw")
            @test haskey(h5, "Info/git_version")
        end
    end
end
