"""
Shared data structures for the regression harness.
"""

"""
Specification for a single quantity to extract from gpec.h5.
"""
struct QuantitySpec
    name::String
    h5path::String          # HDF5 dataset path (e.g. "vacuum/et"), empty for runtime
    type::String            # "complex_vector", "real_vector", "real_scalar", "int_scalar", "real_matrix", "runtime"
    extract::String         # "value", "real_first", "imag_first", "abs_first", "norm", "all_real", "all_complex", "checksum"
    label::String           # Human-readable label for reports
    noise_threshold::Float64 # Absolute changes below this are noise
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
end

"""
Result of extracting a single quantity from an H5 file.
"""
struct ExtractedQuantity
    name::String
    label::String
    value_real::Union{Float64, Nothing}
    value_int::Union{Int, Nothing}
    value_text::Union{String, Nothing}  # JSON for arrays, hex for checksums
    value_type::String                  # "real", "integer", "json_array", "checksum", "missing"
    noise_threshold::Float64
end

"""
Parsed CLI options.
"""
struct CLIOptions
    cases::Vector{String}
    refs::Vector{String}
    ref_range::Union{String, Nothing}
    force::Bool
    list_cases::Bool
    show_qty::Union{String, Nothing}
    show_case::Union{String, Nothing}
    db_path::Union{String, Nothing}
    verbose::Bool
    no_instantiate::Bool
    help::Bool
end
