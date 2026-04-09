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
      - `"normal_field_T"`: B·n̂ in Tesla (2π-angle); converted to unit-norm on load
      - `"sfl_flux_Wb"`: SFL flux in 2π-angle convention; multiplied by (2π)² on load
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
Columns: n, m, real(amplitude), [optional] imag(amplitude)

**HDF5 format** — optional root attribute `normalization` (string), plus datasets:
- `"n"`: Integer array of toroidal mode numbers
- `"m"`: Integer array of poloidal mode numbers
- `"amplitude_real"`: Real parts of amplitudes
- `"amplitude_imag"`: Imaginary parts of amplitudes (optional, default 0)
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
Remaining `#`-prefixed lines and blank lines are ignored. Data columns: n, m, real, [imag].
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

    data = readdlm(filepath; comments=true, comment_char='#')
    nrows = size(data, 1)
    ncols = size(data, 2)

    if ncols < 3
        error("ASCII forcing file must have at least 3 columns (n, m, real_amplitude)")
    end

    empty!(forcing_modes)

    for i in 1:nrows
        n = Int(data[i, 1])
        m = Int(data[i, 2])
        real_part = Float64(data[i, 3])
        imag_part = ncols >= 4 ? Float64(data[i, 4]) : 0.0

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
Required datasets: `n`, `m`, `amplitude_real`. Optional: `amplitude_imag`.
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
        amp_imag = haskey(file, "amplitude_imag") ? read(file, "amplitude_imag") : zeros(length(n_array))

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

include("CoilGeometry.jl")
include("BiotSavart.jl")
include("CoilFourier.jl")

export ForcingTermsControl, ForcingMode, load_forcing_data!

end # module ForcingTerms
