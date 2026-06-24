"""
    KineticProfiles

Reading and processing kinetic profile data (density, temperature, rotation,
and optionally transport diffusivities) from the GPEC HDF5 kinetic format or
from legacy 6-column ASCII tables (`.gpeckf`/`.kin`). The processed output is a
`KineticProfileSplines` of independent named splines that downstream physics
modules (KineticForces NTV, etc.) read directly without re-implementing data
shimming.
"""

using DelimitedFiles
using HDF5

"""
    KineticProfileData

Raw kinetic-profile columns read from a kinetic file (HDF5 or ASCII), before
resampling and spline construction. `psi` (normalized poloidal flux) is always
present; every profile field is `nothing` when the source omits it, so each
consumer validates only the fields it needs. Units: densities m⁻³,
temperatures eV, frequencies rad/s, diffusivities m²/s.

| field       | meaning                              |
|-------------|--------------------------------------|
| `n_i`       | main-ion density                     |
| `n_e`       | electron density                     |
| `T_i`/`T_e` | ion / electron temperature           |
| `omega_E`   | ExB rotation                         |
| `omega_tor` | toroidal rotation (optional)         |
| `chi_e`     | perpendicular heat diffusivity χ⊥    |
| `chi_phi`   | toroidal momentum diffusivity χ_φ    |
"""
struct KineticProfileData
    psi::Vector{Float64}
    n_i::Union{Nothing,Vector{Float64}}
    n_e::Union{Nothing,Vector{Float64}}
    T_i::Union{Nothing,Vector{Float64}}
    T_e::Union{Nothing,Vector{Float64}}
    omega_E::Union{Nothing,Vector{Float64}}
    omega_tor::Union{Nothing,Vector{Float64}}
    chi_e::Union{Nothing,Vector{Float64}}
    chi_phi::Union{Nothing,Vector{Float64}}
    provenance::String
end

function KineticProfileData(; psi, n_i=nothing, n_e=nothing, T_i=nothing, T_e=nothing,
    omega_E=nothing, omega_tor=nothing, chi_e=nothing, chi_phi=nothing, provenance="")
    _v(x) = x === nothing ? nothing : Float64.(collect(x))
    return KineticProfileData(Float64.(collect(psi)), _v(n_i), _v(n_e), _v(T_i), _v(T_e),
        _v(omega_E), _v(omega_tor), _v(chi_e), _v(chi_phi), String(provenance))
end

const _KINETIC_H5_EXTS = (".h5", ".hdf5", ".he5")

# Dataset units written into / expected from the HDF5 kinetic schema.
const _KINETIC_H5_UNITS = Dict("psi" => "normalized poloidal flux", "n_i" => "m^-3",
    "n_e" => "m^-3", "T_i" => "eV", "T_e" => "eV", "omega_E" => "rad/s",
    "omega_tor" => "rad/s", "chi_e" => "m^2/s", "chi_phi" => "m^2/s")

"""
    read_kinetic_file(path; group="/") -> KineticProfileData

Read a kinetic-profile file, dispatching on extension: HDF5 (`.h5`/`.hdf5`) via
the GPEC kinetic schema, otherwise a 6-column ASCII table (`.gpeckf`/`.kin`/`.dat`:
`psi_n n_i n_e T_i[eV] T_e[eV] omega_E`). Returns the raw columns; downstream
consumers validate the fields they require.
"""
function read_kinetic_file(path::AbstractString; group::AbstractString="/")
    isfile(path) || error("Kinetic profile file not found: $path")
    ext = lowercase(splitext(path)[2])
    return ext in _KINETIC_H5_EXTS ? _read_kinetic_h5(path; group=group) : _read_kinetic_ascii(path)
end

"""Read a legacy 6-column ASCII kinetic table into a `KineticProfileData`."""
function _read_kinetic_ascii(path::AbstractString)
    psi, ni, ne, Ti, Te, omegaE = _read_kinetic_table(path)
    return KineticProfileData(; psi=psi, n_i=ni, n_e=ne, T_i=Ti, T_e=Te, omega_E=omegaE,
        provenance="ASCII 6-column: $(basename(path))")
end

"""Read the GPEC HDF5 kinetic schema. Only `psi` is required; other datasets are
optional and surfaced as-is."""
function _read_kinetic_h5(path::AbstractString; group::AbstractString="/")
    h5open(path, "r") do f
        g = group == "/" ? f : f[group]
        haskey(g, "psi") || error("kinetic HDF5 '$path' group '$group': missing required dataset 'psi'")
        rd(name) = haskey(g, name) ? Float64.(vec(read(g[name]))) : nothing
        prov = ""
        ats = attributes(g)
        haskey(ats, "provenance") && (prov = string(read(ats["provenance"])))
        return KineticProfileData(; psi=Float64.(vec(read(g["psi"]))),
            n_i=rd("n_i"), n_e=rd("n_e"), T_i=rd("T_i"), T_e=rd("T_e"),
            omega_E=rd("omega_E"), omega_tor=rd("omega_tor"),
            chi_e=rd("chi_e"), chi_phi=rd("chi_phi"), provenance=prov)
    end
end

"""
    write_kinetic_h5(path, data; group="/", schema_version="1.0", provenance=data.provenance)

Write a `KineticProfileData` to the GPEC HDF5 kinetic schema. Each present field
is written as a dataset with a `units` attribute; absent (`nothing`) fields are
skipped. The group carries `schema_version` and `provenance` attributes.
"""
function write_kinetic_h5(path::AbstractString, data::KineticProfileData;
    group::AbstractString="/", schema_version::AbstractString="1.0",
    provenance::AbstractString=data.provenance)
    h5open(path, "w") do f
        g = group == "/" ? f : create_group(f, group)
        function put(name, v)
            v === nothing && return
            g[name] = collect(Float64, v)
            attributes(g[name])["units"] = _KINETIC_H5_UNITS[name]
        end
        put("psi", data.psi)
        put("n_i", data.n_i)
        put("n_e", data.n_e)
        put("T_i", data.T_i)
        put("T_e", data.T_e)
        put("omega_E", data.omega_E)
        put("omega_tor", data.omega_tor)
        put("chi_e", data.chi_e)
        put("chi_phi", data.chi_phi)
        attributes(g)["schema_version"] = schema_version
        attributes(g)["provenance"] = provenance
    end
    return path
end

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

    data = read_kinetic_file(kinetic_file)

    # NTV requires n_e, T_i, T_e, omega_E; n_i defaults to n_e (quasineutrality)
    # when absent. omega_tor / chi_e / chi_phi (if present) are ignored here.
    _need(field, name) = field === nothing ? error("kinetic file '$kinetic_file' missing required '$name'") : field
    psi_input = data.psi
    ne_input = _need(data.n_e, "n_e")
    Ti_input_eV = _need(data.T_i, "T_i")
    Te_input_eV = _need(data.T_e, "T_e")
    omegaE_input = _need(data.omega_E, "omega_E")
    ni_input = data.n_i === nothing ? copy(ne_input) : data.n_i

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
"""
function _read_kinetic_table(kinetic_file::AbstractString)
    table = DelimitedFiles.readdlm(kinetic_file)

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
