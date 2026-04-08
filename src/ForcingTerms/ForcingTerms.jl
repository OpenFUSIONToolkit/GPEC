module ForcingTerms
# Module for external field specification and forcing term calculations

using DelimitedFiles
using HDF5

"""
    ForcingTermsControl

User-facing control parameters from TOML [ForcingTerms] section.

## Fields

Forcing Data:

  - `forcing_data_file::String` - Path to forcing data file (n, m, complex amplitude)
  - `forcing_data_format::String` - Format of forcing data: "ascii" or "hdf5" (default: "ascii")

Future: Will include Fortran coil.in parameters for external coil configurations
"""
Base.@kwdef mutable struct ForcingTermsControl
    # Forcing data file settings
    forcing_data_file::String = "forcing.dat"
    forcing_data_format::String = "ascii"

    # Future: Fortran coil.in parameters will go here
end

"""
    ForcingMode

Data structure for a single forcing mode.

## Fields

  - `n::Int` - Toroidal mode number
  - `m::Int` - Poloidal mode number
  - `amplitude::ComplexF64` - Complex amplitude of the forcing field
"""
Base.@kwdef mutable struct ForcingMode
    n::Int = 0
    m::Int = 0
    amplitude::ComplexF64 = 0.0 + 0.0im
end

"""
    load_forcing_data!(forcing_modes::Vector{ForcingMode}, dir_path::String, forcing_data_file::String, forcing_data_format::String, verbose::Bool=false)

Load forcing data from ASCII or HDF5 file.

ASCII format: Three or four columns (n, m, real amplitude, [optional] imag amplitude)

  - Column 1: Toroidal mode number (n)
  - Column 2: Poloidal mode number (m)
  - Column 3: Complex amplitude (real part)
  - Column 4: Complex amplitude (imaginary part) [optional, assumes 0 if not present]

Example ASCII file:

```
1  2  0.5  0.1
1  3  0.3  -0.2
2  4  0.8  0.0
```

HDF5 format:

  - Dataset "n": Integer array of toroidal mode numbers
  - Dataset "m": Integer array of poloidal mode numbers
  - Dataset "amplitude_real": Real parts of amplitudes
  - Dataset "amplitude_imag": Imaginary parts of amplitudes
"""
function load_forcing_data!(
    forcing_modes::Vector{ForcingMode},
    dir_path::String,
    forcing_data_file::String,
    forcing_data_format::String,
    verbose::Bool=false
)
    filepath = joinpath(dir_path, forcing_data_file)

    if verbose
        @info "Loading forcing data from $filepath"
    end

    if forcing_data_format == "ascii"
        load_forcing_ascii!(forcing_modes, filepath, verbose)
    elseif forcing_data_format == "hdf5"
        load_forcing_hdf5!(forcing_modes, filepath, verbose)
    else
        error("Unknown forcing data format: $(forcing_data_format). Use 'ascii' or 'hdf5'.")
    end

    if verbose
        @info "Loaded $(length(forcing_modes)) forcing modes"
    end
end

"""
    load_forcing_ascii!(forcing_modes::Vector{ForcingMode}, filepath::String, verbose::Bool)

Load forcing data from ASCII file with columns: n, m, real(amplitude), imag(amplitude)
"""
function load_forcing_ascii!(
    forcing_modes::Vector{ForcingMode},
    filepath::String,
    verbose::Bool
)
    if !isfile(filepath)
        error("Forcing data file not found: $filepath")
    end

    data = readdlm(filepath; comments=true, comment_char='#')
    nrows = size(data, 1)
    ncols = size(data, 2)

    if ncols < 3
        error("ASCII forcing file must have at least 3 columns (n, m, real_amplitude)")
    end

    # Clear existing forcing modes
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
end

"""
    load_forcing_hdf5!(forcing_modes::Vector{ForcingMode}, filepath::String, verbose::Bool)

Load forcing data from HDF5 file with datasets: n, m, amplitude_real, amplitude_imag
"""
function load_forcing_hdf5!(
    forcing_modes::Vector{ForcingMode},
    filepath::String,
    verbose::Bool
)
    if !isfile(filepath)
        error("Forcing data file not found: $filepath")
    end

    h5open(filepath, "r") do file
        n_array = read(file, "n")
        m_array = read(file, "m")
        amp_real = read(file, "amplitude_real")
        amp_imag = haskey(file, "amplitude_imag") ? read(file, "amplitude_imag") : zeros(length(n_array))

        if length(n_array) != length(m_array) || length(n_array) != length(amp_real)
            error("Inconsistent array lengths in HDF5 forcing file")
        end

        # Clear existing forcing modes
        empty!(forcing_modes)

        for i in eachindex(n_array)
            push!(forcing_modes, ForcingMode(;
                n=Int(n_array[i]),
                m=Int(m_array[i]),
                amplitude=complex(amp_real[i], amp_imag[i])
            ))
        end
    end
end

export ForcingTermsControl, ForcingMode, load_forcing_data!

end # module ForcingTerms
