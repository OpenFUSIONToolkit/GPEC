module ForcingTerms
# Module for external field specification and forcing term calculations

using DelimitedFiles
using HDF5
using LinearAlgebra

import ..Equilibrium
import ..Utilities.FourierTransforms: compute_fourier_coefficients

"""
    ForcingTermsControl

User-facing control parameters from TOML [ForcingTerms] section.

## Fields

Forcing Data:

  - `forcing_data_file::String` - Path to forcing data file (n, m, complex amplitude)
  - `forcing_data_format::String` - Format: "ascii", "hdf5", or "coil"

Coil settings (used when `forcing_data_format = "coil"`):

  - `machine::String` - Machine name prefix for .dat files (e.g. "d3d")
  - `dat_dir::String` - Directory containing .dat files; defaults to bundled `coil_geometries/`
  - `mtheta_coil::Int` - Poloidal grid resolution for boundary field evaluation (default: 480)
  - `nzeta_coil::Int` - Toroidal grid resolution; 0 = auto (32 × n)
  - `coil_sets_raw::Vector{Dict{String,Any}}` - Parsed `[[ForcingTerms.coil_set]]` TOML blocks
"""
Base.@kwdef mutable struct ForcingTermsControl
    # Forcing data file settings (ascii/hdf5 formats)
    forcing_data_file::String = "forcing.dat"
    forcing_data_format::String = "ascii"

    # Coil calculation settings (coil format)
    machine::String = ""
    dat_dir::String = ""
    mtheta_coil::Int = 480
    nzeta_coil::Int = 0
    coil_sets_raw::Vector{Dict{String,Any}} = Dict{String,Any}[]
end

"""
    ForcingMode

Data structure for a single forcing mode.

## Fields

  - `n::Int` - Toroidal mode number

  - `m::Int` - Poloidal mode number

  - `amplitude::ComplexF64` - Complex amplitude in unit-norm convention (= Fortran Phi_x,
    T·m² per unit-norm cell) after loading. File inputs are tagged by their input convention:

      + `"normal_field_T"`: B·n̂ in Tesla (2π-angle); converted to unit-norm on load
      + `"sfl_flux_Wb"`: SFL flux in 2π-angle convention; multiplied by (2π)² on load
"""
Base.@kwdef mutable struct ForcingMode
    n::Int = 0
    m::Int = 0
    amplitude::ComplexF64 = 0.0 + 0.0im
end

"""
    load_forcing_data!(forcing_modes, dir_path, forcing_data_file, forcing_data_format, verbose) -> String

Load forcing data from ASCII or HDF5 file. Returns the normalization tag read
from the file (or `"normal_field_T"` by default if no tag is present).

**Normalization tags** (set in the file, not in the TOML):

  - `"normal_field_T"` (default): Fourier modes of B·n̂ [Tesla], 2π-angle convention.
    Most intuitive for users; converted to unit-norm (Phi_x) on load.
  - `"sfl_flux_Wb"`: SFL flux R×(B_R·∂Z/∂θ - B_Z·∂R/∂θ) [T·m²], 2π-angle convention.
    Multiplied by (2π)² on load to reach unit-norm (Phi_x).

**ASCII format** — optional `# normalization: <tag>` header line, then data rows:

```
# normalization: normal_field_T
1  2  0.5  0.1
1  3  0.3  -0.2
```

Columns: n, m, amplitude_real, amplitude_imag

**HDF5 format** — optional root attribute `normalization` (string), plus datasets:

  - `"n"`: Integer array of toroidal mode numbers
  - `"m"`: Integer array of poloidal mode numbers
  - `"amplitude_real"`: Real parts of amplitudes
  - `"amplitude_imag"`: Imaginary parts of amplitudes
"""
function load_forcing_data!(
    forcing_modes::Vector{ForcingMode},
    dir_path::String,
    forcing_data_file::String,
    forcing_data_format::String,
    verbose::Bool=false
)::String
    filepath = joinpath(dir_path, forcing_data_file)

    if verbose
        @info "Loading forcing data from $filepath"
    end

    norm_tag = if forcing_data_format == "ascii"
        load_forcing_ascii!(forcing_modes, filepath, verbose)
    elseif forcing_data_format == "hdf5"
        load_forcing_hdf5!(forcing_modes, filepath, verbose)
    else
        error("Unknown forcing data format: $(forcing_data_format). Use 'ascii' or 'hdf5'.")
    end

    if verbose
        @info "Loaded $(length(forcing_modes)) forcing modes (normalization: $norm_tag)"
    end
    return norm_tag
end

"""
    load_forcing_ascii!(forcing_modes, filepath, verbose) -> String

Load forcing data from ASCII file. Returns normalization tag (`"normal_field_T"` by default).

Optional first `# normalization: <tag>` header line sets the normalization.
Remaining `#`-prefixed lines and blank lines are ignored. Data columns: n, m, amplitude_real, amplitude_imag.
"""
function load_forcing_ascii!(
    forcing_modes::Vector{ForcingMode},
    filepath::String,
    verbose::Bool
)::String
    if !isfile(filepath)
        error("Forcing data file not found: $filepath")
    end

    norm_tag = "normal_field_T"  # default

    # Scan header lines for normalization tag before passing to readdlm
    open(filepath, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            startswith(stripped, '#') || break  # stop at first non-comment line
            m = match(r"#\s*normalization\s*:\s*(\S+)", stripped)
            if !isnothing(m)
                norm_tag = m.captures[1]
            end
        end
    end

    data = readdlm(filepath; comments=true, comment_char=('#'))
    nrows = size(data, 1)
    ncols = size(data, 2)

    if ncols < 4
        error("ASCII forcing file must have 4 columns (n, m, amplitude_real, amplitude_imag)")
    end

    empty!(forcing_modes)

    for i in 1:nrows
        n = Int(data[i, 1])
        m = Int(data[i, 2])
        real_part = Float64(data[i, 3])
        imag_part = Float64(data[i, 4])

        push!(forcing_modes, ForcingMode(;
            n=n,
            m=m,
            amplitude=complex(real_part, imag_part)
        ))
    end
    return norm_tag
end

"""
    load_forcing_hdf5!(forcing_modes, filepath, verbose) -> String

Load forcing data from HDF5 file. Returns normalization tag (`"normal_field_T"` by default).

Optional root attribute `normalization` (string) sets the normalization.
Required datasets: `n`, `m`, `amplitude_real`, `amplitude_imag`.
"""
function load_forcing_hdf5!(
    forcing_modes::Vector{ForcingMode},
    filepath::String,
    verbose::Bool
)::String
    if !isfile(filepath)
        error("Forcing data file not found: $filepath")
    end

    norm_tag = h5open(filepath, "r") do file
        tag = haskey(HDF5.attrs(file), "normalization") ?
              read(HDF5.attrs(file)["normalization"]) : "normal_field_T"

        n_array = read(file, "n")
        m_array = read(file, "m")
        amp_real = read(file, "amplitude_real")
        amp_imag = read(file, "amplitude_imag")

        if length(n_array) != length(m_array) || length(n_array) != length(amp_real)
            error("Inconsistent array lengths in HDF5 forcing file")
        end

        empty!(forcing_modes)

        for i in eachindex(n_array)
            push!(forcing_modes, ForcingMode(;
                n=Int(n_array[i]),
                m=Int(m_array[i]),
                amplitude=complex(amp_real[i], amp_imag[i])
            ))
        end
        return tag
    end
    return norm_tag
end

# Coil-forcing helpers; these depend on ForcingTermsControl and ForcingMode defined above.
include("CoilGeometry.jl")
include("BiotSavart.jl")
include("CoilFourier.jl")

"""
    save_forcing_to_h5(forcing_modes::Vector{ForcingMode}, group)

Write forcing modes to an open HDF5 group using the same layout that
`load_forcing_hdf5!` consumes: datasets `n`, `m`, `amplitude_real`,
`amplitude_imag`. Used by the gpec.h5 snapshot writer so the replay path can
re-read forcing data from the output file with no reference to the original
ASCII/HDF5 ingest file.
"""
function save_forcing_to_h5(forcing_modes::Vector{ForcingMode}, group)
    n_arr = Int[mode.n for mode in forcing_modes]
    m_arr = Int[mode.m for mode in forcing_modes]
    re_arr = Float64[real(mode.amplitude) for mode in forcing_modes]
    im_arr = Float64[imag(mode.amplitude) for mode in forcing_modes]
    group["n"] = n_arr
    group["m"] = m_arr
    group["amplitude_real"] = re_arr
    group["amplitude_imag"] = im_arr
    return nothing
end

"""
    load_forcing_from_h5_group!(forcing_modes::Vector{ForcingMode}, group)

Populate `forcing_modes` from an already-open HDF5 group with datasets `n`,
`m`, `amplitude_real`, `amplitude_imag`. Mirror of `save_forcing_to_h5` for the
rerun path — deliberately accepts an open group rather than a file path so
the snapshot data can live inside `gpec.h5/Input/RawInputs/ForcingTerms/`.
"""
function load_forcing_from_h5_group!(forcing_modes::Vector{ForcingMode}, group)
    n_array = read(group, "n")
    m_array = read(group, "m")
    amp_real = read(group, "amplitude_real")
    amp_imag = read(group, "amplitude_imag")

    if length(n_array) != length(m_array) || length(n_array) != length(amp_real)
        error("Inconsistent array lengths in HDF5 forcing group")
    end

    empty!(forcing_modes)
    for i in eachindex(n_array)
        push!(forcing_modes, ForcingMode(;
            n=Int(n_array[i]),
            m=Int(m_array[i]),
            amplitude=complex(amp_real[i], amp_imag[i])
        ))
    end
    return forcing_modes
end

"""
    RMPField

Abstract supertype of every external-forcing description the scripting API accepts: a
resonant-magnetic-perturbation field, described by where it comes from and how strongly it
drives. Whether the source is a forcing-mode file, a coil set with currents, or (future) a
field given on a control surface, they are all just external fields — one type drives the
perturbed-equilibrium stage.

Nothing is read from disk or computed at construction — the modes are materialized against
an equilibrium when the perturbed-equilibrium stage runs, so one `RMPField` can drive
several solves.

## Construction

    RMPField(path; format=..., scale=1.0, kwargs...)
    RMPField(coil_sets::Vector{Dict{String,Any}}; scale=1.0, kwargs...)
    RMPField(ctrl::ForcingTermsControl; scale=1.0)

The first form points at a forcing-mode file, `format` defaulting to `"hdf5"` for an `.h5`
or `.hdf5` extension and `"ascii"` otherwise. The second form takes TOML-shaped coil blocks
(the `[[ForcingTerms.coil_set]]` layout) and selects the coil format. Remaining `kwargs` are
[`ForcingTermsControl`](@ref) fields. All three return a single-source leaf
([`RMPSource`](@ref)).

## Algebra

`RMPField`s form a vector space: `+`, `-` and multiplication by a scalar (real or complex —
a complex factor phase-rotates the perturbation) build lazy linear combinations without
materializing anything. The perturbed-equilibrium response is linear in the forcing, so
materializing a combination equals combining the materialized sources: each term is
evaluated on the control surface and the mode amplitudes are summed.

```julia
nominal = RMPField("nominal_efc.dat")
weld_field = RMPField("weld_fields.h5"; scale=0.5)
total = 2.0 * nominal + weld_field       # lazy: records terms and weights, computes nothing
pe = perturbed_equilibrium(ffs, total)   # materializes each term, sums, drives PE
```

Amplitude lives with the forcing description itself: per-conductor currents for the coil
format, per-mode `ForcingMode.amplitude` for the file formats. `scale` is NOT a physical
amplitude — it is the source's weight in a linear combination, applied to the materialized
control-surface spectrum uniformly across every toroidal mode (format-independent). Sources
that are not scalar multiples of each other get their own description and the algebra: a
coil set with a failed conductor is not `0.9 * nominal`, it is `nominal - failed_coil` (or
its own source); a magnetic-material field should be computed at the operating point by the
code that owns its physics, with the weight meaningful only for small linear excursions.
A per-n weight dictionary is not supported yet; it needs an n-keyed concept ForcingTerms
does not have.
"""
abstract type RMPField end

"""
    RMPSource

A single-source [`RMPField`](@ref) leaf: one forcing description plus a complex weight.
Built by the `RMPField` constructors; scalar multiplication rescales the weight.

## Fields

  - `ctrl::ForcingTermsControl` - The forcing source: format, file path or machine, and the raw coil-set blocks.
  - `scale::ComplexF64` - The source's weight in a linear combination, applied to the materialized control-surface spectrum (complex = phase rotation). Not a physical amplitude.
"""
struct RMPSource <: RMPField
    ctrl::ForcingTermsControl
    scale::ComplexF64
end

"""
    RMPFieldSum

A lazy linear combination of [`RMPSource`](@ref) leaves, built by `+`/`-` on
[`RMPField`](@ref)s. Holds the flattened term list; scalar multiplication distributes onto
the leaves. Materialization evaluates each term against the equilibrium and sums the mode
amplitudes — valid because the perturbed-equilibrium response is linear in the forcing.

## Fields

  - `terms::Vector{RMPSource}` - The flattened weighted sources.
"""
struct RMPFieldSum <: RMPField
    terms::Vector{RMPSource}
end

"""
    _infer_format(path) -> String

Forcing-data format implied by a file extension: `"hdf5"` for `.h5`/`.hdf5`, else `"ascii"`.
"""
_infer_format(path::AbstractString) = lowercase(splitext(path)[2]) in (".h5", ".hdf5") ? "hdf5" : "ascii"

RMPField(ctrl::ForcingTermsControl; scale::Number=1.0) = RMPSource(ctrl, ComplexF64(scale))

function RMPField(path::AbstractString; format::String=_infer_format(path), scale::Number=1.0, kwargs...)
    ctrl = ForcingTermsControl(; forcing_data_file=abspath(path), forcing_data_format=format, kwargs...)
    return RMPSource(ctrl, ComplexF64(scale))
end

function RMPField(coil_sets::Vector{Dict{String,Any}}; scale::Number=1.0, kwargs...)
    ctrl = ForcingTermsControl(; forcing_data_format="coil", coil_sets_raw=coil_sets, kwargs...)
    return RMPSource(ctrl, ComplexF64(scale))
end

"Flattened weighted-leaf list of any [`RMPField`](@ref)."
_rmp_terms(f::RMPSource) = [f]
_rmp_terms(f::RMPFieldSum) = f.terms

Base.:+(a::RMPField, b::RMPField) = RMPFieldSum(vcat(_rmp_terms(a), _rmp_terms(b)))
Base.:-(a::RMPField) = -1 * a
Base.:-(a::RMPField, b::RMPField) = a + (-1 * b)
Base.:*(c::Number, f::RMPSource) = RMPSource(f.ctrl, ComplexF64(c) * f.scale)
Base.:*(c::Number, f::RMPFieldSum) = RMPFieldSum([c * t for t in f.terms])
Base.:*(f::RMPField, c::Number) = c * f

export ForcingTermsControl, ForcingMode, RMPField, load_forcing_data!, save_forcing_to_h5, load_forcing_from_h5_group!

end # module ForcingTerms
