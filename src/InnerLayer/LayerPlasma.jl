# LayerPlasma.jl: the plasma at the rational surfaces, from the equilibrium's kinetic data. The tearing and matching
# problems both default their layer inputs from these, so every layer model sees one plasma.

function _layer_kinetic_data(equil, what::AbstractString)
    equil.kinetic === nothing &&
        throw(ArgumentError("the layer $what comes from the equilibrium's kinetic data: give it explicitly, or set kinetic_file or call attach_kinetic_profiles!"))
    return equil.kinetic
end

# Throw, naming exactly the layer inputs `who` was not given, when there is no kinetic data to derive them from.
function _require_layer_inputs(who::AbstractString, equil; given...)
    absent = [String(name) for (name, value) in pairs(given) if value === nothing]
    (isempty(absent) || equil.kinetic !== nothing) && return nothing
    throw(ArgumentError("$who: give $(join(absent, ", ", " and ")), or load kinetic profiles to derive them " *
                        "(kinetic_file in [Equilibrium], or attach_kinetic_profiles!)"))
end

"""
    layer_plasma(equil; control=nothing, log=true) -> NamedTuple

The plasma every inner layer sees, from the equilibrium's kinetic data `equil.kinetic`.

## Fields

  - `data` - The kinetic data.
  - `profiles` - `KineticProfiles` splines of n_e, T_e, T_i and Ω_E on the data's ψ grid.
  - `zeff` - Z_eff(ψ) from the data's profile or the n_i/n_e closure, else the scalar `zeff_default`.
  - `mu_i`, `z_i` - Main-ion mass in proton masses (fraction-weighted over the species) and charge.
  - `mu_rho` - Mean ion mass per electron in proton masses at `zeff`, so ρ = μ m_p n_e (a number or a ψ-callable).
  - `resistivity_model`, `lnLambda_form` - The data's resistivity closure and Coulomb-log form.
  - `chi_perp_e`, `chi_perp_i`, `chi_tor` - χ(ψ) splines the `SLAYERModel` `control` reads (`nothing` when absent or
    unread, so the caller uses `chi_*_default`; all `nothing` without `control`).
  - `has_omega_E` - Whether the data carry Ω_E.

`log = false` skips the note on where Z_eff came from.
"""
function layer_plasma(equil; control=nothing, log::Bool=true)
    data = _layer_kinetic_data(equil, "plasma")
    loaded = SLAYER._slayer_profiles(data, control)
    zeff = log ? SLAYER._layer_zeff(loaded, data) : something(loaded.zeff, data.zeff_default)
    return (; data, loaded.profiles, zeff, mu_i=SLAYER._layer_ion_mass(data), z_i=Float64(first(data.ion_species).z),
        mu_rho=SLAYER._layer_mass_per_electron(data, zeff), resistivity_model=SLAYER._build_resistivity_model(data.resistivity_model),
        data.lnLambda_form, loaded.chi_perp_e, loaded.chi_perp_i, loaded.chi_tor, loaded.zeff_source, loaded.has_omega_E)
end

"""
    layer_eta(equil, surfaces) -> Vector{Float64}

Resistivity η in Ω·m at each rational surface of `surfaces` (core to edge), from the equilibrium's kinetic data by
its resistivity closure, the one the SLAYER layer uses.
"""
function layer_eta(equil, surfaces)
    _layer_kinetic_data(equil, "resistivity η")
    plasma = layer_plasma(equil)
    return SLAYER.layer_resistivity(equil, surfaces, plasma.profiles; plasma.zeff, plasma.resistivity_model, plasma.lnLambda_form)
end

"""
    layer_rho(equil, surfaces) -> Vector{Float64}

Ion mass density ρ = μ·m_p·n_e in kg/m³ at each rational surface of `surfaces`, from the equilibrium's kinetic data:
μ is the mean ion mass per electron of the main ions and the impurity at the layer Z_eff, the density the SLAYER
layer uses.
"""
function layer_rho(equil, surfaces)
    _layer_kinetic_data(equil, "mass density ρ")
    plasma = layer_plasma(equil)
    _at(x, ψ) = x isa Real ? x : x(ψ)
    return Float64[_at(plasma.mu_rho, s.psifac) * Utilities.M_P * plasma.profiles.n_e(s.psifac) for s in surfaces]
end

"""
    layer_omega_E(equil, surfaces) -> Vector{Float64}

E×B rotation Ω_E per unit n in rad/s at each rational surface of `surfaces`, from the kinetic data's `omega_E`;
NaN when the equilibrium carries no kinetic data or the data no `omega_E`.
"""
function layer_omega_E(equil, surfaces)
    data = equil.kinetic
    (data === nothing || data.omega_E === nothing) && return fill(NaN, length(surfaces))
    spline = SLAYER._slayer_profiles(data, nothing).profiles.omega_E
    return Float64[spline(s.psifac) for s in surfaces]
end
