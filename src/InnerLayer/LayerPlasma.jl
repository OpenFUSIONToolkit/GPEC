# LayerPlasma.jl: the plasma at the rational surfaces, from the equilibrium's kinetic data. The tearing and matching
# problems both default their layer inputs from these, so every layer model sees one plasma.

function _layer_kinetic_data(equil, what::AbstractString)
    equil.kinetic === nothing &&
        throw(ArgumentError("the layer $what comes from the equilibrium's kinetic data: give it explicitly, or set kinetic_file or call attach_kinetic_profiles!"))
    return equil.kinetic
end

"""
    layer_eta(equil, surfaces) -> Vector{Float64}

Resistivity η in Ω·m at each rational surface of `surfaces` (core to edge), from the equilibrium's kinetic data by
its resistivity closure, the one the SLAYER layer uses.
"""
function layer_eta(equil, surfaces)
    data = _layer_kinetic_data(equil, "resistivity η")
    loaded = SLAYER._slayer_profiles(data, SLAYERModel())
    return SLAYER.layer_resistivity(equil, surfaces, loaded.profiles; zeff=SLAYER._layer_zeff(loaded, data),
        resistivity_model=SLAYER._build_resistivity_model(data.resistivity_model), lnLambda_form=data.lnLambda_form)
end

"""
    layer_rho(equil, surfaces) -> Vector{Float64}

Mass density ρ = m_i·n_i in kg/m³ of the main ion at each rational surface of `surfaces`, from the equilibrium's
kinetic data.
"""
function layer_rho(equil, surfaces)
    data = _layer_kinetic_data(equil, "mass density ρ")
    data.n_i === nothing && throw(ArgumentError("the layer mass density ρ = m_i·n_i needs an n_i profile in the kinetic data"))
    n_i = cubic_interp(collect(Float64, data.psi), data.n_i)
    return SLAYER._layer_ion_mass(data) * Utilities.M_P .* Float64[n_i(s.psifac) for s in surfaces]
end

"""
    layer_omega_E(equil, surfaces) -> Vector{Float64}

E×B rotation Ω_E per unit n in rad/s at each rational surface of `surfaces`, from the kinetic data's `omega_E`;
NaN when the equilibrium carries no kinetic data or the data no `omega_E`.
"""
function layer_omega_E(equil, surfaces)
    data = equil.kinetic
    (data === nothing || data.omega_E === nothing) && return fill(NaN, length(surfaces))
    spline = SLAYER._slayer_profiles(data, SLAYERModel()).profiles.omega_E
    return Float64[spline(s.psifac) for s in surfaces]
end
