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
    layer_eta(equil, surfaces) -> Vector{Float64}

Resistivity η in Ω·m at each rational surface of `surfaces` (core to edge), from the equilibrium's kinetic data by
its resistivity closure, the one the SLAYER layer uses.
"""
function layer_eta(equil, surfaces)
    data = _layer_kinetic_data(equil, "resistivity η")
    loaded = SLAYER._slayer_profiles(data, nothing)
    return SLAYER.layer_resistivity(equil, surfaces, loaded.profiles; zeff=SLAYER._layer_zeff(loaded, data),
        resistivity_model=SLAYER._build_resistivity_model(data.resistivity_model), lnLambda_form=data.lnLambda_form)
end

"""
    layer_rho(equil, surfaces) -> Vector{Float64}

Ion mass density ρ = μ·m_p·n_e in kg/m³ at each rational surface of `surfaces`, from the equilibrium's kinetic data:
μ is the mean ion mass per electron of the main ions and the impurity at the layer Z_eff, the density the SLAYER
layer uses.
"""
function layer_rho(equil, surfaces)
    data = _layer_kinetic_data(equil, "mass density ρ")
    loaded = SLAYER._slayer_profiles(data, nothing)
    mu = SLAYER._layer_mass_per_electron(data, SLAYER._layer_zeff(loaded, data))
    _at(x, ψ) = x isa Real ? x : x(ψ)
    return Float64[_at(mu, s.psifac) * Utilities.M_P * loaded.profiles.n_e(s.psifac) for s in surfaces]
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
