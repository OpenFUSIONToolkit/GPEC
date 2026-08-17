"""
Shared data structures for the regression harness.
"""

"""
Specification for a single quantity to extract from gpec.h5.
"""
struct QuantitySpec
    name::String
    h5path::String          # HDF5 dataset path (e.g. "ForceFreeStates/FreeBoundaryStability/eigenmode_energies"), empty for runtime
    type::String            # "complex_vector", "real_vector", "real_scalar", "int_scalar", "real_matrix", "token", "runtime"
    extract::String         # "value", "real_first", "imag_first", "abs_first", "norm", "all_real", "all_complex",
    # "diagonal_complex", "first_<N>", "first_<N>_complex", "checksum", "toml_key:<dotted.path>"
    label::String           # Human-readable label for reports
    noise_threshold::Float64 # Absolute changes below this are noise
    order::Int              # Display order in reports (lower = earlier)
end

"""
Specification for a test case: what to run and what to extract.

`kind` selects the runner backend:
  - "gpec_run"  (default) — run GPEC end-to-end on `example_dir`, extract from `gpec.h5`
  - "computed"  — run a self-contained Julia computation that writes a small h5
                  (no `example_dir` required); used for analytic/reference cases
                  like the GGJ inner-layer benchmark.
"""
struct CaseSpec
    name::String
    description::String
    example_dir::String     # Relative to repo root; empty for kind="computed"
    quantities::Vector{QuantitySpec}
    kind::String
    overrides::Dict{String,Any}  # gpec.toml keys patched for this run; dotted keys e.g. "KineticForces.nutype" => "zero". Empty = run the deck in place.
end

"""
Result of extracting a single quantity from an H5 file.
"""
struct ExtractedQuantity
    name::String
    label::String
    value_real::Union{Float64,Nothing}
    value_int::Union{Int,Nothing}
    value_text::Union{String,Nothing}  # JSON for arrays, hex for checksums
    value_type::String                  # "real", "integer", "json_array", "checksum", "missing"
    noise_threshold::Float64
end

"""
Parsed CLI options.

`no_pin_manifest` disables copying the working tree's resolved `Manifest.toml` into each
worktree (pinning is on by default, so that two refs differ only by source code).
`allow_env_mismatch` lets a cached result from a different environment be reused instead of
re-run. `fail_on_change` turns any changed quantity into a non-zero exit status, for CI use.

`check` runs the working tree and compares it against the committed golden values, which is the
mode CI gates on. `update_golden` regenerates those values from a fresh run; `reason` records
why, and is mandatory because a golden change is a claim about physics that a reviewer has to be
able to evaluate.
"""
struct CLIOptions
    cases::Vector{String}
    refs::Vector{String}
    ref_range::Union{String,Nothing}
    force::Bool
    list_cases::Bool
    show_qty::Union{String,Nothing}
    show_case::Union{String,Nothing}
    db_path::Union{String,Nothing}
    verbose::Bool
    no_instantiate::Bool
    no_pin_manifest::Bool
    allow_env_mismatch::Bool
    fail_on_change::Bool
    check::Bool
    update_golden::Bool
    reason::Union{String,Nothing}
    help::Bool
end
