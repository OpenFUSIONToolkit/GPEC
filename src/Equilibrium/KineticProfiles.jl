"""
    KineticProfiles

Reading and processing kinetic profile data (density, temperature, rotation)
from ASCII tables. The output is a `KineticProfileSplines` of independent named
splines that downstream physics modules (KineticForces NTV, etc.) can read
directly without re-implementing data shimming.
"""

using DelimitedFiles

"""
    load_kinetic_profiles(kinetic_file::AbstractString;
                          zi::Int=1, zimp::Int=6, mi::Int=2, mimp::Int=12,
                          density_factor::Float64=1.0, temperature_factor::Float64=1.0,
                          ExB_rotation_factor::Float64=1.0, toroidal_rotation_factor::Float64=1.0,
                          chi1::Union{Nothing,Float64}=nothing)
        → KineticProfileSplines

Parse an ASCII kinetic profile file, interpolate onto a regular 101-point ψ
grid, optionally apply profile scaling knobs, derive collisional / Z_eff
diagnostics, and return a `KineticProfileSplines` with independent named cubic
splines.

# Expected file format

Six whitespace-separated columns (header rows are filtered out):

    psi_n  n_i[m^-3]  n_e[m^-3]  T_i[eV]  T_e[eV]  omega_E[rad/s]

# Arguments

  - `kinetic_file`: Path to the ASCII kinetic profile file
  - `zi`, `zimp`: Main ion and impurity charge numbers
  - `mi`, `mimp`: Main ion and impurity mass numbers (in proton masses)
  - `density_factor`: Density scaling factor (applied to ni, ne)
  - `temperature_factor`: Temperature scaling factor (applied to Ti, Te)
  - `ExB_rotation_factor`: ExB rotation scaling factor (applied to omegaE after rotation reform)
  - `toroidal_rotation_factor`: Toroidal rotation scaling factor (scales total wphi = omegaE + wdian + wdiat)
  - `chi1`: Poloidal flux normalization `2π·ψ₀` — required when `density_factor`, `temperature_factor`, or `toroidal_rotation_factor` differ from 1.0

# Scaling sequence

When any of `density_factor`, `temperature_factor`, `toroidal_rotation_factor` differ from 1.0:
1. Build first-pass cubic splines from unscaled profiles (for derivatives)
2. Compute diamagnetic frequencies `wdian`, `wdiat` and total toroidal rotation `wphi`
3. Scale: `wdian_new = temperature_factor * wdian`, `wdiat_new = temperature_factor * wdiat` (`density_factor` cancels in `T*(dn/dψ)/n`)
4. Reform: `omegaE = toroidal_rotation_factor * wphi - wdian_new - wdiat_new`
5. Scale density/temperature arrays: `ni *= density_factor`, `Ti *= temperature_factor`, etc.

Then `ExB_rotation_factor` is applied independently: `omegaE *= ExB_rotation_factor`.

Collisionality is recomputed from the (possibly scaled) profiles. This differs from
Fortran PENTRC, which computes collisionality from unscaled profiles. Use `nufac`
(in `KineticForcesControl`) for independent collisionality scaling.
"""
function load_kinetic_profiles(kinetic_file::AbstractString;
    zi::Int=1, zimp::Int=6, mi::Int=2, mimp::Int=12,
    density_factor::Float64=1.0, temperature_factor::Float64=1.0,
    ExB_rotation_factor::Float64=1.0, toroidal_rotation_factor::Float64=1.0,
    chi1::Union{Nothing,Float64}=nothing)

    if !isfile(kinetic_file)
        error("Kinetic profile file not found: $kinetic_file")
    end

    psi_input, ni_input, ne_input, Ti_input_eV, Te_input_eV, omegaE_input =
        _read_kinetic_table(kinetic_file)

    eV_to_J = 1.602e-19
    mp = 1.672_614e-27
    me = 9.109_1e-31

    nkin = 100
    psi_reg = collect(0:nkin) ./ nkin

    # Match Fortran pentrc/inputs.f90:215-232 — cubic-spline-then-resample, NOT
    # linear interp. Linear interp of the irregular .kin grid produces large
    # errors wherever the profile has curvature, and is catastrophic where
    # omegaE crosses zero (DIIID: ψ≈0.9 omegaE flips sign in one Δψ≈0.01 cell;
    # linear interp misses sign and magnitude of welec → wrong resonance
    # denominator). See feedback_kf_kin_profile_linear_interp.md.
    ni = _cubic_resample(psi_input, ni_input, psi_reg)
    ne = _cubic_resample(psi_input, ne_input, psi_reg)
    Ti = _cubic_resample(psi_input, Ti_input_eV, psi_reg) .* eV_to_J
    Te = _cubic_resample(psi_input, Te_input_eV, psi_reg) .* eV_to_J
    omegaE = _cubic_resample(psi_input, omegaE_input, psi_reg)

    needs_rotation_reform = density_factor != 1.0 || temperature_factor != 1.0 || toroidal_rotation_factor != 1.0
    any_scaling = needs_rotation_reform || ExB_rotation_factor != 1.0

    if any_scaling
        @info "KineticProfiles: scaling applied — density_factor=$density_factor, temperature_factor=$temperature_factor, ExB_rotation_factor=$ExB_rotation_factor, toroidal_rotation_factor=$toroidal_rotation_factor"
    end

    if needs_rotation_reform
        (chi1 === nothing || chi1 == 0.0) && error("chi1 (= 2π·ψ₀) required when density_factor, temperature_factor, or toroidal_rotation_factor != 1.0")

        # First-pass splines for cubic derivatives (unscaled profiles)
        ni_spl = cubic_interp(collect(Float64, psi_reg), collect(Float64, ni); extrap=ExtendExtrap())
        Ti_spl = cubic_interp(collect(Float64, psi_reg), collect(Float64, Ti); extrap=ExtendExtrap())
        dni_dpsi = deriv1(ni_spl)
        dTi_dpsi = deriv1(Ti_spl)

        # Compute original wdian, wdiat, wphi and reform omegaE at each grid point
        echarge = 1.602e-19
        chrg_ion = zi * echarge
        for i in eachindex(omegaE)
            ψ = psi_reg[i]
            wdian_i = ni[i] > 0 ? -2π * Ti[i] * dni_dpsi(ψ) / (chrg_ion * chi1 * ni[i]) : 0.0
            wdiat_i = -2π * dTi_dpsi(ψ) / (chrg_ion * chi1)
            wphi_i = omegaE[i] + wdian_i + wdiat_i

            # Scaled diamagnetic: density_factor cancels in T*(dn/dψ)/n; temperature_factor enters linearly
            wdian_new = temperature_factor * wdian_i
            wdiat_new = temperature_factor * wdiat_i

            omegaE[i] = toroidal_rotation_factor * wphi_i - wdian_new - wdiat_new
        end

        ni .*= density_factor
        ne .*= density_factor
        Ti .*= temperature_factor
        Te .*= temperature_factor
    end

    if ExB_rotation_factor != 1.0
        omegaE .*= ExB_rotation_factor
    end

    loglam = zeros(Float64, nkin + 1)
    nui = zeros(Float64, nkin + 1)
    nue = zeros(Float64, nkin + 1)
    zeff = zeros(Float64, nkin + 1)

    for i in 1:(nkin + 1)
        n_i = ni[i]
        n_e = ne[i]
        T_i = Ti[i]
        T_e = Te[i]

        z = n_e > 0 ? zimp - (n_i / n_e) * zi * (zimp - zi) : Float64(zimp)
        zpitch = 1.0 + (1.0 + mimp) / (2.0 * mimp) * zimp * (z - 1.0) / (zimp - z)

        ll = 17.3 - 0.5 * log10(n_e / 1.0e20) + 1.5 * log10(T_e / 1.602e-16)
        loglam[i] = ll

        nui[i] = T_i > 0 ?
            (zpitch / 3.5e17) * n_i * ll / (sqrt(1.0 * mi) * (T_i / 1.602e-16)^1.5) : 0.0
        nue[i] = T_e > 0 ?
            (zpitch / 3.5e17) * n_e * ll / (sqrt(me / mp) * (T_e / 1.602e-16)^1.5) : 0.0
        zeff[i] = z
    end

    return KineticProfileSplines(psi_reg, ni, ne, Ti, Te, omegaE, loglam, nui, nue, zeff)
end

"""
Internal helper: parse the raw 6-column kinetic profile table from disk,
filtering out non-numeric header rows. Returns six independent column views.
`#` comment lines (e.g. a provenance header) are stripped so they cannot widen
the parsed matrix and pad the data rows.
"""
function _read_kinetic_table(kinetic_file::AbstractString)
    table = DelimitedFiles.readdlm(kinetic_file; comments=true)

    # Keep numeric rows only (drops text headers). Build row-major by rebuilding
    # the matrix as a stack of rows — reshape(:, 6) is column-major and scrambles
    # columns.
    numeric_rows = [collect(row) for row in eachrow(table) if all(x -> isa(x, Number), row)]
    if isempty(numeric_rows)
        error("No numeric data rows found in kinetic file: $kinetic_file")
    end
    table = reduce(vcat, (reshape(Float64.(r), 1, 6) for r in numeric_rows))

    psi_input = collect(table[:, 1])
    n_i_input = collect(table[:, 2])
    n_e_input = collect(table[:, 3])
    T_i_input = collect(table[:, 4])
    T_e_input = collect(table[:, 5])
    omega_e_input = collect(table[:, 6])

    return psi_input, n_i_input, n_e_input, T_i_input, T_e_input, omega_e_input
end

"""
Cubic-spline resample matching Fortran pentrc/inputs.f90:215-232:
build a cubic spline on the (irregular) input grid, then evaluate at the
regular psi_new grid. Out-of-range points use ExtendExtrap (smooth cubic
extrapolation), matching Fortran's `spline_fit(...,"extrap")`.
"""
function _cubic_resample(x::AbstractVector, y::AbstractVector, x_new::AbstractVector)
    spl = cubic_interp(collect(Float64, x), collect(Float64, y); extrap=ExtendExtrap())
    return [spl(xv) for xv in x_new]
end
