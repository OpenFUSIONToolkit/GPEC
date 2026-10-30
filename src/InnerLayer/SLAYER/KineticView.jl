# KineticView.jl: the inner layers' view of the equilibrium's kinetic data (`equil.kinetic`), shared by the
# tearing and matching problems.

using ...Equilibrium: KineticProfileData

# Layer view of raw kinetic profiles (`equil.kinetic`) on their own ψ grid: the `KineticProfiles`, the χ⊥,e/χ⊥,i/χ_φ(ψ)
# callables the `SLAYERModel` uses (`nothing` when absent, so the caller uses `chi_*_default`), Z_eff(ψ) (the data's
# `zeff`, else from n_i/n_e, else `nothing` for `zeff_default`) with its source, and whether the data carry `omega_E`.
function _slayer_profiles(data::KineticProfileData, control)
    source = isempty(data.provenance) ? "kinetic profiles" : "kinetic profiles ($(data.provenance))"
    for (name, v) in (("n_e", data.n_e), ("T_e", data.T_e), ("T_i", data.T_i))
        v === nothing &&
            error("Inner layer: $source missing required dataset '$name' for the SLAYER inner layer.")
    end
    # `omega_E` carries the E×B rotation Ω_E (per unit n) when the data have it, else zero (no shift);
    # ω_*e/ω_*i are derived per surface from the density and temperature splines by `build_slayer_inputs`.
    omega_E = data.omega_E === nothing ? zeros(length(data.psi)) : data.omega_E
    profiles = KineticProfiles(; psi=data.psi, n_e=data.n_e, T_e=data.T_e,
        T_i=data.T_i, omega_E=omega_E)

    # χ⊥(ψ)/χ_φ(ψ) splines; an absent or all-zero array means "not provided" and defers to the data's `chi_*_default`.
    psi_xs = collect(Float64, data.psi)
    function _chi_spline(name, v)
        (v === nothing || all(iszero, v)) && return nothing
        minimum(v) > 0 || error("Inner layer: $source has $name with minimum $(minimum(v)) ≤ 0; diffusivities must be positive")
        return cubic_interp(psi_xs, collect(Float64, v))
    end
    model = control.P_perp_model
    uses_e = model in (:chi_perp_e, :D_perp) || control.D_c_type !== :none
    chi_perp_e = uses_e ? _chi_spline("chi_e", data.chi_e) : nothing
    chi_perp_i = model in (:chi_perp_i, :D_perp) ? _chi_spline("chi_i", data.chi_i) : nothing
    chi_tor = model !== :tau_E ? _chi_spline("chi_phi", data.chi_phi) : nothing
    # Z_eff(ψ): the file's profile wins; else quasi-neutrality with one impurity of charge `impurity_charge` and a
    # charge-1 main ion, Z_eff = impurity_charge − (n_i/n_e)(impurity_charge − 1).
    _usable(v) = v !== nothing && !all(iszero, v)
    zeff = if _usable(data.zeff)
        minimum(data.zeff) >= 1.0 - 1e-8 ||
            error("Inner layer: $source has zeff = $(minimum(data.zeff)) < 1 — unphysical")
        cubic_interp(psi_xs, collect(Float64, data.zeff))
    elseif !_usable(data.n_i)
        nothing
    else
        zimp = Float64(data.impurity_charge)
        z = [ne > 0 ? zimp - (ni / ne) * (zimp - 1.0) : zimp for (ni, ne) in zip(data.n_i, data.n_e)]
        minimum(z) >= 1.0 - 1e-8 ||
            error("Inner layer: $source has n_i > n_e, giving Z_eff = $(minimum(z)) < 1 — unphysical")
        cubic_interp(psi_xs, z)
    end
    zeff_source = _usable(data.zeff) ? :file : zeff === nothing ? :scalar : :closure
    return (profiles=profiles, chi_perp_e=chi_perp_e, chi_perp_i=chi_perp_i, chi_tor=chi_tor, zeff=zeff, zeff_source=zeff_source,
        has_omega_E=(data.omega_E !== nothing))
end

# Map the TOML resistivity_model symbol to a NeoResistivityModel instance.
function _build_resistivity_model(name::Symbol)
    name === :sauter && return SauterNeoModel()
    name === :redl && return RedlNeoModel()
    name === :spitzer && return SpitzerModel()
    name === :spitzer_harm && return SpitzerHarmModel()
    throw(ArgumentError("_build_resistivity_model: unknown model $name"))
end

# The layer models take one main ion: its mass in proton masses.
function _layer_ion_mass(data::KineticProfileData)
    length(data.ion_species) == 1 ||
        throw(ArgumentError("The tearing layers take one main-ion species; the kinetic data list $(length(data.ion_species))"))
    return Float64(only(data.ion_species).m)
end

# Z_eff(ψ) for the layers, reporting where it came from: the file's profile, the n_i/n_e closure, or the default.
function _layer_zeff(loaded, data::KineticProfileData)
    if loaded.zeff_source === :scalar
        @warn "Inner layer: kinetic file has no usable zeff or n_i profile (dataset absent or all-zero); " *
              "using zeff_default = $(data.zeff_default) at every surface."
        return data.zeff_default
    end
    zeff_origin = loaded.zeff_source === :file ? "zeff profile" : "n_i/n_e with impurity_charge = $(data.impurity_charge)"
    @info "Inner layer: Z_eff(ψ) from the kinetic file's $zeff_origin"
    return loaded.zeff
end
