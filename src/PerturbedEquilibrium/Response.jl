"""
    load_forcing_data!(intr::PerturbedEquilibriumInternal, ctrl::PerturbedEquilibriumControl)

Load forcing data from ASCII or HDF5 file.

ASCII format: Three columns (n, m, complex amplitude)
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
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    filepath = joinpath(intr.dir_path, ctrl.forcing_data_file)

    if ctrl.verbose
        println("Loading forcing data from $filepath")
    end

    if ctrl.forcing_data_format == "ascii"
        load_forcing_ascii!(intr, filepath, ctrl.verbose)
    elseif ctrl.forcing_data_format == "hdf5"
        load_forcing_hdf5!(intr, filepath, ctrl.verbose)
    else
        error("Unknown forcing data format: $(ctrl.forcing_data_format). Use 'ascii' or 'hdf5'.")
    end

    if ctrl.verbose
        println("  Loaded $(length(intr.forcing_modes)) forcing modes")
    end
end

"""
    load_forcing_ascii!(intr::PerturbedEquilibriumInternal, filepath::String, verbose::Bool)

Load forcing data from ASCII file with columns: n, m, real(amplitude), imag(amplitude)
"""
function load_forcing_ascii!(
    intr::PerturbedEquilibriumInternal,
    filepath::String,
    verbose::Bool
)
    if !isfile(filepath)
        error("Forcing data file not found: $filepath")
    end

    data = readdlm(filepath, comments=true, comment_char='#')
    nrows = size(data, 1)
    ncols = size(data, 2)

    if ncols < 3
        error("ASCII forcing file must have at least 3 columns (n, m, real_amplitude)")
    end

    # Clear existing forcing modes
    empty!(intr.forcing_modes)

    for i in 1:nrows
        n = Int(data[i, 1])
        m = Int(data[i, 2])
        real_part = Float64(data[i, 3])
        imag_part = ncols >= 4 ? Float64(data[i, 4]) : 0.0

        push!(intr.forcing_modes, ForcingMode(
            n=n,
            m=m,
            amplitude=complex(real_part, imag_part)
        ))
    end
end

"""
    load_forcing_hdf5!(intr::PerturbedEquilibriumInternal, filepath::String, verbose::Bool)

Load forcing data from HDF5 file with datasets: n, m, amplitude_real, amplitude_imag
"""
function load_forcing_hdf5!(
    intr::PerturbedEquilibriumInternal,
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
        empty!(intr.forcing_modes)

        for i in eachindex(n_array)
            push!(intr.forcing_modes, ForcingMode(
                n=Int(n_array[i]),
                m=Int(m_array[i]),
                amplitude=complex(amp_real[i], amp_imag[i])
            ))
        end
    end
end

"""
    compute_plasma_response!(
        state::PerturbedEquilibriumState,
        equil::Equilibrium.PlasmaEquilibrium,
        dcon_results::OdeState,
        vac_data::VacuumData,
        dcon_intr::DconInternal,
        intr::PerturbedEquilibriumInternal,
        ctrl::PerturbedEquilibriumControl
    )

Compute plasma response to external forcing using DCON eigenmode solutions.

Implements resp_index=0 calculation from gpresp.f:
1. Build flux matrix from eigenmodes
2. Calculate plasma inductance (energy-based)
3. Calculate surface inductance from Green's function
4. Compute permeability matrix
5. Apply forcing to get response
"""
function compute_plasma_response!(
    state::PerturbedEquilibriumState,
    equil::Equilibrium.PlasmaEquilibrium,
    dcon_results::OdeState,
    vac_data::VacuumData,
    dcon_intr::DconInternal,
    intr::PerturbedEquilibriumInternal,
    ctrl::PerturbedEquilibriumControl
)
    if ctrl.verbose
        println("Computing plasma response using resp_index=0 (energy-based inductance)")
    end

    # Step 1: Build flux matrix from DCON eigenmodes
    if ctrl.verbose
        println("  Building flux matrix from eigenmodes")
    end
    flux_matrix = build_flux_matrix(dcon_results, vac_data, dcon_intr)

    # Step 2: Calculate plasma inductance matrix
    if ctrl.verbose
        println("  Calculating plasma inductance matrix")
    end
    plasma_inductance = calc_plasma_inductance(flux_matrix, vac_data.et)

    # Step 3: Calculate surface inductance from Green's functions
    if ctrl.verbose
        println("  Calculating surface inductance from Green's functions")
    end
    surface_inductance = calc_surface_inductance(vac_data.grri, vac_data.grre, dcon_intr)

    # Step 4: Calculate permeability matrix
    if ctrl.verbose
        println("  Calculating permeability matrix")
    end
    permeability = calc_permeability(plasma_inductance, surface_inductance)

    # Store permeability in internal state for later use
    intr.plasma_response = permeability

    # Step 5: Map forcing modes to eigenmode basis
    if ctrl.verbose
        println("  Mapping forcing modes to eigenmode basis")
        println("    Number of forcing modes: $(length(intr.forcing_modes))")
    end
    forcing_vector = map_forcing_to_eigenmodes(intr.forcing_modes, dcon_intr)

    # Step 6: Compute plasma response
    if ctrl.verbose
        println("  Computing response = permeability * forcing")
    end
    response_vector = compute_plasma_response_vector(permeability, forcing_vector)

    # Step 7: Store response (for now, just store the vector)
    # TODO: Convert response vector to physical fields (xi, b)
    npsi = size(dcon_results.u_store, 1)  # Number of radial points
    ntheta = 128  # Placeholder: should match equilibrium grid
    nmodes = length(intr.forcing_modes)

    state.xi_perturbed = zeros(ComplexF64, npsi, ntheta, nmodes)
    state.b_perturbed = zeros(ComplexF64, npsi, ntheta, nmodes)

    if ctrl.verbose
        println("  Response calculation complete")
        println("    Response vector size: $(length(response_vector))")
        println("    Max response amplitude: $(maximum(abs.(response_vector)))")
    end
end
