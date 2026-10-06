# LayerParameters.jl
#
# The shared per-surface layer-parameter builder: turn the kinetic profiles attached to the
# equilibrium into the (η, ρ, rotation) the inner-layer models consume, with explicit
# vectors as overrides. Uses the same Coulomb-log / resistivity closures and mass-density
# formula as the SLAYER and GGJ input builders, so every inner-layer consumer sees one
# plasma description per run whichever path derived it.

using ..Utilities.PhysicalConstants: M_P, E_CHG
using ..Utilities.NeoclassicalResistivity: NeoResistivityModel, SpitzerModel,
    coulomb_log_e, eta_spitzer, nu_star_e, eta_neoclassical

"""
    layer_parameters(surfaces, equil; eta=nothing, rho=nothing, rotation=nothing,
                     mu_i=2.0, zeff=1.0, resistivity_model=SpitzerModel(),
                     lnLambda_form=:nrl) -> (; eta, rho, rotation)

Per-surface inner-layer plasma parameters for the rational surfaces in `surfaces`, derived
from the kinetic profiles attached to the equilibrium (`equil.kinetic`) — or taken verbatim
from the explicit override vectors, which always win. Returns one value per surface, core
to edge in the order of `surfaces`:

  - `eta` — resistivity η in Ω·m: Spitzer (Sauter 1999 Eq. 18a) by default, or the
    neoclassical closure selected by `resistivity_model`, which reads the trapped fraction
    and local geometry off each surface's `restype` (populated by `resist_eval_all!`).
  - `rho` — mass density ρ = μᵢ·m_p·n_e(ψ_s) in kg/m³, quasineutral main-ion convention.
  - `rotation` — rotation frequency f in Hz from the E×B frequency, f = ω_E(ψ_s)/2π; the
    forced layer eigenvalue of the driven match is γ_s = 2πi·n·f_s.

A derivation (any override left `nothing`) requires `equil.kinetic` — attach profiles with
`attach_kinetic_profiles!` or at equilibrium construction. Temperatures are converted from
the stored Joules back to eV for the resistivity formulas; derived values agree with the
file-driven SLAYER/GGJ input builders to the kinetic loader's resampling accuracy (the
loader resamples onto a regular ψ grid), not bit-for-bit.

Overrides are artificial-scan and no-kinetic-data paths: each of `eta`, `rho`, `rotation`
may independently be a vector with one entry per surface.
"""
function layer_parameters(
    surfaces::AbstractVector,
    equil;
    eta::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rho::Union{Nothing,AbstractVector{<:Real}}=nothing,
    rotation::Union{Nothing,AbstractVector{<:Real}}=nothing,
    mu_i::Real=2.0,
    zeff::Real=1.0,
    resistivity_model::NeoResistivityModel=SpitzerModel(),
    lnLambda_form::Symbol=:nrl
)
    msing = length(surfaces)
    for (name, v) in (("eta", eta), ("rho", rho), ("rotation", rotation))
        v === nothing || length(v) == msing ||
            error("layer_parameters: $name has length $(length(v)), expected one value per surface (msing=$msing, core to edge)")
    end

    needs_derivation = eta === nothing || rho === nothing || rotation === nothing
    if needs_derivation
        kp = equil.kinetic
        kp === nothing &&
            error("layer_parameters: deriving η/ρ/rotation needs kinetic profiles on the equilibrium — " *
                  "attach them with attach_kinetic_profiles!(equil, file) or pass all three override vectors")

        eta_out = Vector{Float64}(undef, msing)
        rho_out = Vector{Float64}(undef, msing)
        rot_out = Vector{Float64}(undef, msing)
        for (k, sing) in enumerate(surfaces)
            ψ = sing.psifac
            n_e = kp.ne_spline(ψ)
            t_e = kp.Te_spline(ψ) / E_CHG      # stored in J; resistivity formulas take eV
            lnLamb = coulomb_log_e(n_e, t_e; form=lnLambda_form)
            if resistivity_model isa SpitzerModel
                eta_out[k] = eta_spitzer(n_e, t_e, zeff; lnLamb=lnLamb)
            else
                rg = sing.restype
                rg === nothing &&
                    error("layer_parameters: surface $k has restype = nothing — the neoclassical resistivity " *
                          "closure needs the trapped fraction from resist_eval_all!; run it first or use SpitzerModel()")
                nuestar = nu_star_e(n_e, t_e, rg.R_major, rg.eps_local, sing.q, zeff; lnLamb=lnLamb)
                eta_out[k] = eta_neoclassical(resistivity_model, n_e, t_e, zeff, rg.f_trap, nuestar; lnLamb=lnLamb)
            end
            rho_out[k] = mu_i * M_P * n_e
            rot_out[k] = kp.omegaE_spline(ψ) / (2π)
        end
    end

    return (
        eta=eta === nothing ? eta_out : collect(Float64, eta),
        rho=rho === nothing ? rho_out : collect(Float64, rho),
        rotation=rotation === nothing ? rot_out : collect(Float64, rotation)
    )
end
