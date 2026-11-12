# CCCParameters.jl
#
# Per-surface parameters of the CCC layer: the GGJ coefficients and local times, which set its normalization
# exactly as for GGJ, plus the electron diamagnetic frequency and the perpendicular-transport weight.

"""
    CCCParameters

Parameters of the drift-resistive layer of Cho, Choi & Choi, IEEE Trans. Plasma Sci. PS-15, 418 (1987), in the form
of Kim, Cho & Choi, Phys. Fluids 31, 2659 (1988), Eqs. (1a)–(1c), at one rational surface. Its Γ is Υ here. Q* = 0 and M⊥/M = 1 give the GGJ equations.

## Fields

  - `ggj::GGJParameters` - GGJ coefficients E, F, G, H, K, M and local times; Kim's coefficients are GGJ's, and
    so are the time and length scales, the Mercier exponents of the far field and the Δ normalization.
  - `Qstar::Float64` - Electron diamagnetic frequency scaled like Q, Q* = ω*ₑ/Q₀, in the sign convention of the
    growth rate s = γ − iω (Kim's Ohm's law carries Q + iQ*).
  - `mperp::Float64` - Perpendicular-transport weight M⊥/M of the QΥ'' term; must be positive.
"""
struct CCCParameters <: ToroidalLayerParameters
    ggj::GGJParameters
    Qstar::Float64
    mperp::Float64
    function CCCParameters(ggj::GGJParameters, Qstar::Real, mperp::Real)
        mperp > 0 || throw(ArgumentError("CCCParameters: mperp = $mperp must be positive"))
        return new(ggj, Float64(Qstar), Float64(mperp))
    end
end

"""
    CCCParameters(; Qstar=0.0, mperp=1.0, ggj_fields...)

Keyword constructor: `ggj_fields` are the `GGJParameters` keywords (`E`, `F`, `G`, `H`, `K`, `M`, `taua`, `taur`,
`v1`, `ising`).
"""
CCCParameters(; Qstar::Real=0.0, mperp::Real=1.0, ggj_fields...) = CCCParameters(GGJParameters(; ggj_fields...), Qstar, mperp)

inner_Q(p::CCCParameters, γ::Number) = inner_Q(p.ggj, γ)
rescale_delta(Δ::AbstractVector, p::CCCParameters) = rescale_delta(Δ, p.ggj)
profile_conversions(p::CCCParameters) = profile_conversions(p.ggj)
power_exponents(p::CCCParameters) = power_exponents(p.ggj)
mercier_di(p::CCCParameters) = mercier_di(p.ggj)
growth_rate_scale(p::CCCParameters) = growth_rate_scale(p.ggj)

function layer_parameter_columns(params::AbstractVector{CCCParameters})
    return [layer_parameter_columns([p.ggj for p in params]);
        "Q_star" => (; values=Float64[p.Qstar for p in params], long_name="scaled electron diamagnetic frequency Q* = −n·ω*ₑ/Q₀ per surface", units="1");
        "M_perp_ratio" => (; values=Float64[p.mperp for p in params], long_name="perpendicular-transport weight M⊥/M per surface", units="1")]
end
