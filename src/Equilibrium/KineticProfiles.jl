"""
    KineticProfiles

Reading and processing kinetic profile data (density, temperature, rotation)
from ASCII tables. These profiles are needed by the KineticForces module
for NTV torque and kinetic energy calculations.
"""

using DelimitedFiles
using Printf

"""
    read_kinetic_profiles(kin_file::String; zi=1, zimp=6, mi=2, mimp=12,
                          nfac=1.0, tfac=1.0, wefac=1.0, wpfac=1.0,
                          write_log=false, verbose=false)

Read kinetic profiles from ASCII file and compute derived quantities
(collision frequencies, Zeff, diamagnetic frequencies).

Expected file format: 6 columns
  psi_n  n_i(m^-3)  n_e(m^-3)  T_i(eV)  T_e(eV)  omega_E(rad/s)

# Arguments
- `kin_file::String`: Path to kinetic profile ASCII file
- `zi::Int`: Main ion charge number (default: 1)
- `zimp::Int`: Impurity charge number (default: 6, carbon)
- `mi::Int`: Main ion mass number (default: 2, deuterium)
- `mimp::Int`: Impurity mass number (default: 12, carbon)
- `nfac::Float64`: Density scaling factor
- `tfac::Float64`: Temperature scaling factor
- `wefac::Float64`: ExB rotation scaling factor
- `wpfac::Float64`: Indirect rotation scaling factor

# Returns
- `Matrix{Float64}`: (nkin+1) × 10 array with columns:
  [psi, n_i, n_e, T_i(J), T_e(J), omega_E, loglam, nu_i, nu_e, zeff]
"""
function read_kinetic_profiles(kin_file::String; zi=1, zimp=6, mi=2, mimp=12,
                               nfac=1.0, tfac=1.0, wefac=1.0, wpfac=1.0,
                               write_log=false, verbose=false)

    if !isfile(kin_file)
        error("Kinetic profile file not found: $kin_file")
    end

    table = DelimitedFiles.readdlm(kin_file)

    # Filter out non-numeric rows (headers)
    table_clean = Float64[]
    for row in eachrow(table)
        if all(x -> isa(x, Number), row)
            push!(table_clean, vec(row)...)
        end
    end
    table = reshape(table_clean, :, 6)

    if write_log
        println("Table shape: ", size(table))
    end

    eV_to_J = 1.602e-19
    mp = 1.672_614e-27
    me = 9.109_1e-31
    e_charge = 1.602_191_7e-19
    twopi = 2π

    nkin = 100

    psi_input = @view table[:, 1]
    n_i_input = @view table[:, 2]
    n_e_input = @view table[:, 3]
    T_i_input = @view table[:, 4]
    T_e_input = @view table[:, 5]
    omega_e_input = @view table[:, 6]

    psi_reg = collect(0:nkin) ./ nkin
    kin_data = zeros(nkin + 1, 10)

    kin_data[:, 1] = psi_reg
    kin_data[:, 2] = _linear_interp_extrap(psi_input, n_i_input, psi_reg)
    kin_data[:, 3] = _linear_interp_extrap(psi_input, n_e_input, psi_reg)
    kin_data[:, 4] = _linear_interp_extrap(psi_input, T_i_input, psi_reg)
    kin_data[:, 5] = _linear_interp_extrap(psi_input, T_e_input, psi_reg)
    kin_data[:, 6] = _linear_interp_extrap(psi_input, omega_e_input, psi_reg)

    # Convert temperatures from eV to Joules
    kin_data[:, 4] .*= eV_to_J
    kin_data[:, 5] .*= eV_to_J

    zeff = similar(psi_reg)
    zpitch = similar(psi_reg)

    for i in 1:(nkin + 1)
        n_i = kin_data[i, 2]
        n_e = kin_data[i, 3]
        T_i = kin_data[i, 4]
        T_e = kin_data[i, 5]

        zeff[i] = n_e > 0 ? zimp - (n_i / n_e) * zi * (zimp - zi) : Float64(zimp)
        zpitch[i] = 1.0 + (1.0 + mimp) / (2.0 * mimp) * zimp * (zeff[i] - 1.0) / (zimp - zeff[i])

        loglam = 17.3 - 0.5 * log10(n_e / 1.0e20) + 1.5 * log10(T_e / 1.602e-16)
        kin_data[i, 7] = loglam

        # Ion collision frequency
        kin_data[i, 8] = T_i > 0 ?
            (zpitch[i] / 3.5e17) * n_i * loglam / (sqrt(1.0 * mi) * (T_i / 1.602e-16)^1.5) : 0.0

        # Electron collision frequency
        kin_data[i, 9] = T_e > 0 ?
            (zpitch[i] / 3.5e17) * n_e * loglam / (sqrt(me / mp) * (T_e / 1.602e-16)^1.5) : 0.0

        kin_data[i, 10] = zeff[i]
    end

    if write_log
        println("Formed kinetic profile on $(nkin + 1) point grid")
    end

    # Apply scaling factors
    kin_data[:, 2] .*= nfac
    kin_data[:, 3] .*= nfac
    kin_data[:, 4] .*= tfac
    kin_data[:, 5] .*= tfac

    welec = kin_data[:, 6] .* wefac

    # Avoid division by zero in rotation
    warning_printed = false
    for i in 1:(nkin + 1)
        if welec[i] == 0
            welec[i] = 1e-9
            if !warning_printed
                println("  ! WARNING ! Modifying omega_ExB = 0 to 1e-9 to avoid nans")
                warning_printed = true
            end
        end
    end

    # Diamagnetic frequencies (numerical derivatives)
    wdian = zeros(nkin + 1)
    wdiat = zeros(nkin + 1)
    for i in 2:nkin
        dn_i_dpsi = (kin_data[i + 1, 2] - kin_data[i - 1, 2]) / (2.0 / nkin)
        dT_i_dpsi = (kin_data[i + 1, 4] - kin_data[i - 1, 4]) / (2.0 / nkin)
        chi1 = 1.0  # TODO: Get from equilibrium
        wdian[i] = -twopi * kin_data[i, 4] * dn_i_dpsi / (e_charge * zi * chi1 * kin_data[i, 2])
        wdiat[i] = -twopi * dT_i_dpsi / (e_charge * zi * chi1)
    end
    wdian[1] = wdian[2]
    wdian[nkin + 1] = wdian[nkin]
    wdiat[1] = wdiat[2]
    wdiat[nkin + 1] = wdiat[nkin]

    # Apply indirect rotation scaling
    for i in 1:(nkin + 1)
        if welec[i] != 0
            wpefac_i = (wpfac * (welec[i] + wdian[i] + wdiat[i]) - (wdian[i] + wdiat[i])) / welec[i]
            welec[i] = wpefac_i * welec[i]
        end
    end
    kin_data[:, 6] = welec

    return kin_data
end


"""
Linear interpolation with extrapolation beyond data range.
"""
function _linear_interp_extrap(x::AbstractVector, y::AbstractVector, x_new::AbstractVector)
    y_new = similar(x_new, Float64)
    for (i, xval) in enumerate(x_new)
        if xval < x[1]
            slope = (y[2] - y[1]) / (x[2] - x[1])
            y_new[i] = y[1] + slope * (xval - x[1])
        elseif xval > x[end]
            slope = (y[end] - y[end - 1]) / (x[end] - x[end - 1])
            y_new[i] = y[end] + slope * (xval - x[end])
        else
            idx = searchsortedlast(x, xval)
            idx = clamp(idx, 1, length(x) - 1)
            t = (xval - x[idx]) / (x[idx + 1] - x[idx])
            y_new[i] = y[idx] * (1 - t) + y[idx + 1] * t
        end
    end
    return y_new
end
