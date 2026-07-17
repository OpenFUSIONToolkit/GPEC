"""
    Configure

Level-0 **named-configuration assembly** (design `03 §2`): turns a physics
parameter set + a species list into a runnable `Operators.IslandStack`, its
far-field boundary conditions, and the `Δ`-moment prefactors — the object the
solver consumes.

**Every Level-0 operator coefficient is now cleared** (QUESTIONS Q5 fully
cleared): the assembly wires each from `phys` via a cleared `Coefficients.*`
builder — no gated kinetic inputs, no placeholders:

  - the orbit-averaged magnetic drift `c_D` ([`drift_coefficient_table`]); the
    **island-streaming** `a_xi`/`a_x` ([`streaming_coefficients`], `{Ω, ·}`
    advection); the **`E×B` coupling** `c_E` ([`exb_coupling_table`],
    `½⟨1/v̂_∥⟩_θ`, passing-only);
  - the **full orbit-averaged collision operator** (`orbit-averaged-collision.md`):
    the σ-odd mimetic **pitch diffusion** D+E (`P_oa = y⟨√(1−yb)⟩`, flat measure,
    [`pitch_diffusivity_profile`] + [`pitch_collision_coefficient`]); the `∂_x`
    **drag** A ([`collisional_drag_coefficient`]); the `∂²_x` **neoclassical**
    diffusion B ([`neoclassical_diffusion_coefficient`]); the `∂²_{xy}` **cross**
    term C ([`collisional_cross_coefficient`]) — magnitude `ε^{3/2}ν_★` from
    `phys.nu_star`, `1/(m ρ̂_θi)` from `phys.m`;
  - the **quasineutrality field term** — `α = (τ+1)/τ` and the `L̂_{n0}⁻¹(x − ĥ)`
    drive ([`quasineutrality_source`]), closing the Level-0 potential (`01 §3`);
    the **gradient drive** — zero interior source + the neoclassical far field
    ([`gradient_far_field`], I19 Formulation A); and the `Δ`-moment prefactors.

The only collision piece not yet a stack operator is the momentum-restoring
field-particle term (its magnitude `⟨ν̂_ii⟩_u` is cleared in
`collision-magnitude.md`); it is a separate future operator addition.
"""
module Configure

using LinearAlgebra
import ..PhaseSpace: IslandGrid, nnodes, MappedFDGrid
import ..Operators: IslandStack, IslandState, FarFieldConditions,
    ParallelStreaming, MagneticDrift, ExBDrift, PitchAngleDiffusion,
    GradientDrive, Quasineutrality, conservative_pitch_operator,
    CollisionalDrag, NeoclassicalDiffusion, CollisionalCross, MomentumRestoring
import ..Coefficients:
    magnetic_drift_frequency, orbit_average_drift_brackets, orbit_average_exb_bracket,
    orbit_average_pitch_brackets,
    deflection_frequency, momentum_restoring_average, delta_moment_prefactors,
    h_amplitude, quasineutrality_coefficient
import ..Fields: h_profile
import ..Moments: parallel_current!, delta_moments, channel_decomposition
import ..SpeciesLists: Species, Maxwellian, Bulk, validate_species, bulk_species

export Level0Physics, configure_level0, delta_outputs
export drift_coefficient_table, pitch_diffusivity_profile, pitch_collision_coefficient
export collisional_drag_coefficient, neoclassical_diffusion_coefficient, collisional_cross_coefficient
export quasineutrality_source, streaming_coefficients, gradient_far_field
export exb_coupling_table, physical_velocity_weights, parallel_flow_weight
export momentum_restoring_term

# ---------------------------------------------------------------------------
# Physics parameter carrier (the cleared inputs)
# ---------------------------------------------------------------------------
"""
    Level0Physics(; epsilon, inv_Lq, inv_LB, q_s, dq_dpsi, w_psi, mu0_R,
                    tau=1.0, variant=:original, collision_model=:chandrasekhar)

The Level-0 physics parameters (scenario inputs) that feed the **cleared**
coefficient builders (`01 §1`–`§4`). Every Level-0 operator coefficient is now
built from these — there are no gated kinetic inputs (QUESTIONS Q5 cleared).

## Fields

  - `epsilon`         — inverse aspect ratio `ε = r_s/R₀`.
  - `inv_Lq`          — `L̂_q⁻¹ = (ψ_s/q_s) dq/dψ` at `r_s` (drift, `01 §2.1`).
  - `inv_LB`          — `L̂_B⁻¹` (∇B drift term; forced to `0` when
    `variant = :improved`, `01 §2.1`).
  - `q_s`, `dq_dpsi`  — safety factor and its `ψ`-derivative at `r_s`
    (`ψ̃` and `Δ`-prefactor, `01 §1`, `§4`).
  - `w_psi`           — island **half**-width (the `w` in `Ω = 2x²/w² − cosξ`
    and the `ĥ` amplitude `w/2√2`, `01 §1`, `§2.4`).
  - `mu0_R`           — `μ₀R` geometric factor of the `Δ`-moment prefactor
    (`01 §4`).
  - `inv_Ln0`         — inverse density scale length `L̂_{n0}⁻¹` at `r_s`, the
    quasineutrality drive amplitude (`01 §3`).
  - `rho_hat_theta_i` — normalized ion poloidal gyroradius `ρ̂_θi` (`~ ŵ` by the
    O2 ordering); sets the island-streaming scale (`01 §2`, `parallel-streaming.md`).
  - `eta_i`           — `η_i = L_n/L_{T_i} = (T_i'/T_i)/(n'/n)`, the
    temperature-gradient ratio in the gradient-drive far field (`01 §2`,
    `gradient-drive.md`); `0` = flat temperature.
  - `nu_star`         — banana-regime collisionality `ν_★ = ν_{jj}Rq/(ε^{3/2}v_th)`
    (`01 §2.3`, `collision-magnitude.md`); a **scenario scan input** (`ν_★ ≪ 1`,
    Decision D7). Sets the collision magnitude `ν̂_jj = ε^{3/2}ν_★ ν̃_jj(v̂)`.
  - `m`               — resonant **poloidal mode number** (the rational `m/n`); a
    scenario input. Unlike the drift/streaming/E×B channels (where `m` cancels in
    the ÷−m ρ̂_θi normalization), the collision terms carry an explicit `1/(m ρ̂_θi)`
    (`01 §2.3`, `orbit-averaged-collision.md` §3).
  - `tau`             — `T_e/T_i` (quasineutrality closure, `01 §3`).
  - `variant`         — `:original`/`:improved` drift-model toggle (`01 §2.1`).
  - `collision_model` — `:chandrasekhar`/`:vcubed` deflection-frequency energy
    dependence (`01 §2.3`).
"""
Base.@kwdef struct Level0Physics
    epsilon::Float64
    inv_Lq::Float64
    inv_LB::Float64
    q_s::Float64
    dq_dpsi::Float64
    w_psi::Float64
    mu0_R::Float64
    inv_Ln0::Float64
    rho_hat_theta_i::Float64
    eta_i::Float64 = 0.0
    nu_star::Float64 = 0.01
    m::Float64 = 2.0
    tau::Float64 = 1.0
    variant::Symbol = :original
    collision_model::Symbol = :chandrasekhar
end

# ---------------------------------------------------------------------------
# Cleared coefficient wiring — every Level-0 operator coefficient is now built
# from `phys` via cleared `Coefficients.*` builders (QUESTIONS Q5 fully cleared:
# no gated kinetic inputs remain).
# ---------------------------------------------------------------------------
"""
    drift_coefficient_table(grid, phys) -> Array{Float64,5}

Build the orbit-averaged magnetic-drift coefficient `c_D[ix, iξ, iy, iE, iσ]` for
`Operators.MagneticDrift` by evaluating the **cleared**
`Coefficients.magnetic_drift_frequency` on the phase-space grid
(`01 §2.1`). `ω̂_D` depends on `(y, E, σ)` only (through `v̂ = √E` and the orbit
brackets `A(y)`, `G(y)`), so the `(y, E, σ)` table is broadcast over `(x, ξ)`.
The orbit brackets are computed once per `y` (they depend only on `y`, `ε`) and
reused across `E`, `σ` — the drift is linear in `σ v̂`. Uses `phys.variant` for
the `:original`/`:improved` `L̂_B⁻¹` toggle. Grid nodes in the forbidden pitch
region `y ≥ 1/b_min = (1+ε)/(1−ε)` carry no particles, so `c_D ≡ 0` there (a
grid may extend past the physical pitch edge; the deeply-trapped limit is
unambiguous).
"""
# Cleared orbit-average with a graceful miss at the near-separatrix y_c layer:
# returns (A, G), or `nothing` if the bounce-average quadrature cannot converge
# (the gated y_c layer, handled by the caller). Only the singular layer misses;
# every well-defined node returns the exact cleared value.
function _try_drift_brackets(y::Real, ε::Real)
    try
        return orbit_average_drift_brackets(; y=y, epsilon=ε)
    catch err
        err isa Union{ErrorException,DomainError} || rethrow(err)
        return nothing
    end
end

function drift_coefficient_table(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    ε = phys.epsilon
    LB = phys.variant === :improved ? 0.0 : phys.inv_LB
    y_forbidden = (1 + ε) / (1 - ε)                        # 1/b_min: no particle beyond (01 §2.1)
    cD = Array{Float64}(undef, nx, nξ, ny, nE, nσ)
    @inbounds for iy in 1:ny
        y = grid.y.nodes[iy]
        if y >= y_forbidden                               # forbidden region carries no particles ⇒ ω̂_D ≡ 0
            @views cD[:, :, iy, :, :] .= 0.0
            continue
        end
        # The trapped-passing boundary y = y_c = 1 is the near-separatrix pitch
        # layer: θ_b → π and the bounce average develops the integrable
        # 1/√(1−yb) turning-point singularity, so the cleared orbit-average
        # quadrature cannot reach tolerance there. That layer's drift is part of
        # the gated y_c-matching treatment (04 §3, ladder A8, QUESTIONS Q5); the
        # node gets a documented gated placeholder (0), not a guessed value.
        brackets = _try_drift_brackets(y, ε)
        if brackets === nothing
            @views cD[:, :, iy, :, :] .= 0.0
            continue
        end
        A, G = brackets
        bracket = phys.inv_Lq * A - 0.5 * LB * G          # the [·] of ω̂_D (01 §2.1)
        for iσ in 1:nσ
            σ = grid.σ[iσ]
            for iE in 1:nE
                v̂ = sqrt(grid.E.nodes[iE])                 # E = v̂² (Maxwellian energy)
                val = (σ * v̂ / (1 + ε)) * bracket
                for iξ in 1:nξ, ix in 1:nx
                    cD[ix, iξ, iy, iE, iσ] = val
                end
            end
        end
    end
    return cD
end

# Cleared orbit-averaged pitch brackets (S=⟨√(1−yb)⟩, T=⟨1/√(1−yb)⟩) with a
# graceful miss at the near-separatrix y_c layer / forbidden region (returns
# `nothing`), handled by the caller exactly as `_try_drift_brackets`.
function _try_pitch_brackets(y::Real, ε::Real)
    try
        return orbit_average_pitch_brackets(; y=y, epsilon=ε)
    catch err
        err isa Union{ErrorException,DomainError} || rethrow(err)
        return nothing
    end
end

"""
    pitch_diffusivity_profile(grid, phys) -> (P_oa, wmeas)

Build the **cleared orbit-averaged** pitch-diffusion profile for the mimetic
`Operators.conservative_pitch_operator` (`01 §2.3`; derivation
`orbit-averaged-collision.md` §4, terms D+E). The orbit-averaged operator is the
pure `y`-divergence `∂_y(P_oa ∂_y)` with

```math
P_{\\rm oa}(y) = y\\,\\langle\\sqrt{1-yb}\\rangle_\\theta = y\\,S(y),
\\qquad \\texttt{wmeas} = 1 ,
```

`S(y)` from the cleared `Coefficients.orbit_average_pitch_brackets` (passing full
circuit / trapped bounce, forbidden region → `0`). `P_oa` **replaces** the former
local single-`B` diffusivity `λ√(1−λB)` (the `B_profile` placeholder), and the
**flat measure** (`wmeas = 1`) replaces `B/√(1−λB)` — L23 confirms the `⟨√(1−yb)⟩`
term of `∂²_y` vanishes at `y = 0, 1/b` (natural zero-flux BC), so the operator
conserves `∫g dy` and preserves the A4 gate for the new `P_oa ≥ 0`. The
`y_c`-layer node (where the passing/trapped average jumps) carries `P_oa = 0`
(gated placeholder; the physical jump is handled by the `y_c` matching, `04 §3`).
Returns `(P_oa, wmeas)`; introduces **no** new physics parameter (only `ε`).
"""
function pitch_diffusivity_profile(grid::IslandGrid, phys::Level0Physics)
    ny = grid.y.n
    ε = phys.epsilon
    y_forbidden = (1 + ε) / (1 - ε)                       # 1/b_min: no particle beyond
    P_oa = zeros(Float64, ny)                             # forbidden / y_c layer ⇒ 0
    wmeas = ones(Float64, ny)                             # flat measure (divergence form)
    @inbounds for iy in 1:ny
        y = grid.y.nodes[iy]
        y < y_forbidden || continue                       # forbidden region ⇒ 0
        br = _try_pitch_brackets(y, ε)
        br === nothing && continue                        # y_c layer: placeholder 0
        S, _ = br
        P_oa[iy] = y * S                                  # y⟨√(1−yb)⟩_θ
    end
    return P_oa, wmeas
end

# ν̂_ii(v̂) = ε^{3/2}ν_★ ν̃(v̂): the cleared deflection frequency at the code magnitude
@inline _nu_hat_ii(phys::Level0Physics, v̂::Real) =
    deflection_frequency(v̂; nu_tilde=phys.epsilon^1.5 * phys.nu_star, model=phys.collision_model)

"""
    pitch_collision_coefficient(grid, phys) -> Array{Float64,4}

Build the **cleared** σ-odd coefficient `c[ix, iξ, iE, iσ]` of the orbit-averaged
pitch diffusion (terms D+E, `01 §2.3`; `orbit-averaged-collision.md` §3–§4) for
`Operators.PitchAngleDiffusion` (which applies `c ⋅ (K g)` with `K = ∂_y(P_oa ∂_y)`
from [`pitch_diffusivity_profile`]):

```math
c = \\frac{2\\,\\hat\\nu_{ii}(\\hat v)(1+\\varepsilon)}{m\\,\\hat\\rho_{\\theta i}\\,\\sigma\\hat v},
\\qquad \\hat v = \\sqrt E .
```

**σ-odd** (`1/σ = σ`; the `1/v̂_∥` transit weight, as for `ω̂_D`/`c_E`) — this
**corrects** the former σ-even placeholder. `y`-independent (dims `(x,ξ,E,σ)`), as
`PitchAngleDiffusion` requires so the mimetic `K`'s exact conservation is
preserved (the `y`-structure lives in `P_oa`). Carries the collision `1/(m ρ̂_θi)`
normalization (`§3`) and the `(1+ε)`; `ν̂_ii = ε^{3/2}ν_★ ν̃(v̂)`.
"""
function pitch_collision_coefficient(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    # +sign: L23 prints the D+E term with a leading −; ÷(−m ρ̂_θi) flips it to +
    # (the same ÷(−m ρ̂_θi) that makes streaming a_xi positive; §3)
    pref = 2 * (1 + phys.epsilon) / (phys.m * phys.rho_hat_theta_i)
    c = Array{Float64}(undef, nx, nξ, nE, nσ)
    @inbounds for iσ in 1:nσ
        σ = grid.σ[iσ]
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            val = pref * _nu_hat_ii(phys, v̂) / (σ * v̂)   # σ-odd (1/σ = σ)
            for iξ in 1:nξ, ix in 1:nx
                c[ix, iξ, iE, iσ] = val
            end
        end
    end
    return c
end

"""
    collisional_drag_coefficient(grid, phys) -> Array{Float64,5}

Build the **cleared** collision `∂_x` drag coefficient `a_x[ix,iξ,iy,iE,iσ]` for
`Operators.CollisionalDrag` (term A, `01 §2.3`; `orbit-averaged-collision.md`
§2–§3): `a_x = +ν̂_ii(v̂)/m · Θ(y_c − y)` — **σ-even**, passing-only, `ρ̂_θi`
cancels (the `+` is the ÷−m ρ̂_θi flip of L23's leading `−`). `ν̂_ii = ε^{3/2}ν_★ ν̃(√E)`.
"""
function collisional_drag_coefficient(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    a = zeros(Float64, nx, nξ, ny, nE, nσ)
    @inbounds for iy in 1:ny
        grid.y.nodes[iy] < grid.y_c || continue          # passing-only (Θ_y)
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            val = _nu_hat_ii(phys, v̂) / phys.m       # +: ÷(−m ρ̂_θi) of L23's leading −ν̂_ii ρ̂_θ
            for iσ in 1:nσ, iξ in 1:nξ, ix in 1:nx
                a[ix, iξ, iy, iE, iσ] = val
            end
        end
    end
    return a
end

"""
    neoclassical_diffusion_coefficient(grid, phys) -> Array{Float64,5}

Build the **cleared** neoclassical `∂²_x` diffusion coefficient
`c[ix,iξ,iy,iE,iσ]` for `Operators.NeoclassicalDiffusion` (term B, `01 §2.3`;
`orbit-averaged-collision.md` §2–§3):

```math
c = \\frac{\\hat\\nu_{ii}(\\hat v)\\,\\sigma\\hat v\\,\\hat\\rho_{\\theta i}}{2m(1+\\varepsilon)}\\,y\\,T(y),
\\qquad T(y)=\\langle 1/\\sqrt{1-yb}\\rangle_\\theta ,
```

**σ-odd** (`σu`), with `T` from `Coefficients.orbit_average_pitch_brackets`
(passing + trapped; the `1/√` `y_c`-layer node → gated placeholder `0`). The `ρ̂_θi²`
of L23 (§2.6 amendment) reduces to a single `ρ̂_θi` here after the ÷−m ρ̂_θi
normalization. `ν̂_ii = ε^{3/2}ν_★ ν̃(√E)`.
"""
function neoclassical_diffusion_coefficient(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    ε = phys.epsilon
    pref = phys.rho_hat_theta_i / (2 * phys.m * (1 + ε))   # +: ÷(−m ρ̂_θi) of L23's leading −
    y_forbidden = (1 + ε) / (1 - ε)
    c = zeros(Float64, nx, nξ, ny, nE, nσ)
    @inbounds for iy in 1:ny
        y = grid.y.nodes[iy]
        y < y_forbidden || continue                       # forbidden region ⇒ 0
        br = _try_pitch_brackets(y, ε)
        br === nothing && continue                        # y_c layer ⇒ 0
        _, T = br
        for iσ in 1:nσ
            σ = grid.σ[iσ]
            for iE in 1:nE
                v̂ = sqrt(grid.E.nodes[iE])
                val = pref * _nu_hat_ii(phys, v̂) * (σ * v̂) * y * T   # σ-odd
                for iξ in 1:nξ, ix in 1:nx
                    c[ix, iξ, iy, iE, iσ] = val
                end
            end
        end
    end
    return c
end

"""
    collisional_cross_coefficient(grid, phys) -> Array{Float64,5}

Build the **cleared** collision `∂²_{xy}` cross coefficient `c[ix,iξ,iy,iE,iσ]`
for `Operators.CollisionalCross` (term C, `01 §2.3`;
`orbit-averaged-collision.md` §2–§3): `c = +2ν̂_ii(v̂) y/m · Θ(y_c − y)` —
**σ-even**, passing-only, `ρ̂_θi` cancels (the `+` is the ÷−m ρ̂_θi flip of L23's
leading `−`). `ν̂_ii = ε^{3/2}ν_★ ν̃(√E)`.
"""
function collisional_cross_coefficient(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    c = zeros(Float64, nx, nξ, ny, nE, nσ)
    @inbounds for iy in 1:ny
        y = grid.y.nodes[iy]
        y < grid.y_c || continue                          # passing-only (Θ_y)
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            val = 2 * _nu_hat_ii(phys, v̂) * y / phys.m   # +: ÷(−m ρ̂_θi) of L23's leading −
            for iσ in 1:nσ, iξ in 1:nξ, ix in 1:nx
                c[ix, iξ, iy, iE, iσ] = val
            end
        end
    end
    return c
end

"""
    streaming_coefficients(grid, phys) -> (a_xi, a_x)

Build the **cleared** island-streaming coefficients for
`Operators.ParallelStreaming` (`01 §2`; derivation `parallel-streaming.md`,
sign-off 2026-07-11):

```math
a_\\xi = \\frac{\\hat L_q^{-1}}{\\hat\\rho_{\\theta i}}\\,x\\,\\Theta(y_c-y),
\\qquad
a_x = -\\frac{\\hat L_q^{-1}\\hat w^2}{4\\,\\hat\\rho_{\\theta i}}\\,\\sin\\xi\\,\\Theta(y_c-y),
```

with `ŵ = phys.w_psi`, `ρ̂_θi = phys.rho_hat_theta_i`, and `Θ(y_c − y)` the
passing-particle mask (`1` for `y < grid.y_c`, `0` for trapped). Together they
are `(L̂_q⁻¹ ŵ²/4ρ̂_θi)Θ · {Ω, ·}` — advection along the island flux surfaces
(derivation §3). Depends on `(x, ξ, y)` only, broadcast over `(E, σ)`. The
normalization is chosen so the cleared drift `c_D = ω̂_D` is unchanged (§2).
Returns `(a_xi, a_x)`, each shaped like `g`.
"""
function streaming_coefficients(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    w = phys.w_psi
    ρ = phys.rho_hat_theta_i
    ρ != 0 || throw(ArgumentError("rho_hat_theta_i must be nonzero"))
    a_xi = Array{Float64}(undef, nx, nξ, ny, nE, nσ)
    a_x = Array{Float64}(undef, nx, nξ, ny, nE, nσ)
    @inbounds for iy in 1:ny
        Θ = grid.y.nodes[iy] < grid.y_c ? 1.0 : 0.0        # passing-only (01 §2)
        for iξ in 1:nξ
            sξ = sin(grid.ξ.nodes[iξ])
            for ix in 1:nx
                x = grid.x.nodes[ix]
                v_xi = (phys.inv_Lq / ρ) * x * Θ           # (L̂_q⁻¹/ρ̂_θi) x Θ
                v_x = -(phys.inv_Lq * w^2 / (4 * ρ)) * sξ * Θ  # −(L̂_q⁻¹ŵ²/4ρ̂_θi) sinξ Θ
                for iσ in 1:nσ, iE in 1:nE
                    a_xi[ix, iξ, iy, iE, iσ] = v_xi
                    a_x[ix, iξ, iy, iE, iσ] = v_x
                end
            end
        end
    end
    return a_xi, a_x
end

"""
    exb_coupling_table(grid, phys) -> Array{Float64,5}

Build the **cleared** `E×B` coupling `c_E[ix, iξ, iy, iE, iσ]` for
`Operators.ExBDrift` (`01 §2`; derivation `exb-coupling.md`, sign-off 2026-07-12):

```math
c_E = \\tfrac12\\big\\langle 1/\\hat v_\\parallel\\big\\rangle_\\theta
    = \\frac{\\sigma}{2\\sqrt E}\\,B_1(y)\\,\\Theta(y_c-y),
```

**passing-only** (`Θ(y_c−y)`; trapped `c_E ≡ 0` by the σ-odd banana-leg
cancellation, derivation §4), with `B₁(y) = Coefficients.orbit_average_exb_bracket`
and `v̂ = √E`. The coupling is **σ-odd** (equal and opposite for `v_∥ ≷ 0`), the
same parity as the drift shift `x_D ∝ σ` (`01 §2.2`). Depends on `(y, E, σ)` only,
broadcast over `(x, ξ)`. Trapped nodes, the forbidden pitch region, and the
near-separatrix `y_c` layer (where `B₁` diverges and the quadrature cannot reach
tolerance) all carry `c_E = 0` — the same gated-placeholder-at-the-layer
treatment as [`drift_coefficient_table`] (`04 §3`, ladder A8, QUESTIONS Q5). The
`−m ρ̂_θi` normalization cancels `ρ̂_θi` exactly, so `c_E` introduces **no new
physics parameter** beyond the already-cleared `ε` (derivation §3).
"""
# Cleared passing-orbit E×B bracket with a graceful miss at the y_c layer:
# returns B₁(y), or `nothing` if the quadrature cannot converge (the log-singular
# near-separatrix layer, handled by the caller as for the drift brackets).
function _try_exb_bracket(y::Real, ε::Real)
    try
        return orbit_average_exb_bracket(; y=y, epsilon=ε)
    catch err
        err isa Union{ErrorException,DomainError} || rethrow(err)
        return nothing
    end
end

function exb_coupling_table(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    ε = phys.epsilon
    cE = zeros(Float64, nx, nξ, ny, nE, nσ)               # trapped/forbidden ⇒ 0 (§4)
    @inbounds for iy in 1:ny
        y = grid.y.nodes[iy]
        y < grid.y_c || continue                          # passing-only; trapped c_E ≡ 0
        B1 = _try_exb_bracket(y, ε)
        B1 === nothing && continue                        # y_c layer: gated placeholder 0
        for iσ in 1:nσ
            σ = grid.σ[iσ]
            for iE in 1:nE
                v̂ = sqrt(grid.E.nodes[iE])                 # E = v̂² (Maxwellian energy)
                val = (σ / (2 * v̂)) * B1                   # ½⟨1/v̂_∥⟩ = (σ/2v̂) B₁(y)
                for iξ in 1:nξ, ix in 1:nx
                    cE[ix, iξ, iy, iE, iσ] = val
                end
            end
        end
    end
    return cE
end

"""
    physical_velocity_weights(grid, phys) -> (wy_phys, wE_phys)

Build the **cleared physical `∫d³v` measure** weights (`01 §4`; derivation
`velocity-moment-measure.md`, sign-off 2026-07-13) for
`Operators.velocity_moment!`/`weighted_moment!`, replacing the flat quadrature:

  - `wE_phys[iE] = wE[iE]·√E/2` — the `∫dv̂ v̂²` **speed Jacobian** folded into the
    Gauss–Laguerre `e^{−E}` weight (`E = v̂²`);
  - `wy_phys[iy]` — the **pitch Jacobian** `∫dy/√(1−y b_min)` on the `y`-grid, with
    `b_min = (1−ε)/(1+ε)` the flux-surface (outboard) field (§7, flux-surface `b`).
    Built as an exact **singular-weight quadrature** (each node carries the exact
    `∫dy/√(1−y b_min)` over its cell, clipped to `[0, 1/b_min]`), so the integrable
    `y→1/b_min` edge is handled without blow-up (the `IinvB` treatment, L23 §8.4.1)
    and the forbidden region `y>1/b_min` carries zero weight.

The global `πb_min` constant is absorbed into the downstream normalization
(`L̂_{n0}`/the `Δ` prefactor). Setting `W=1` gives the physical density
`δn̄_i = {ĝ}_v`; `W = v̂_∥` ([`parallel_flow_weight`]) gives `J̄_∥`.
"""
function physical_velocity_weights(grid::IslandGrid, phys::Level0Physics)
    ε = phys.epsilon
    β = (1 - ε) / (1 + ε)                                 # b_min (outboard midplane, flux-surface b)
    yedge = 1 / β                                         # = (1+ε)/(1−ε), the deeply-trapped edge
    yn = grid.y.nodes
    ny = grid.y.n
    # exact ∫_a^b dy/√(1−βy), clipped to [0, yedge]
    Jint(a, b) = (a = clamp(a, 0.0, yedge); b = clamp(b, 0.0, yedge);
    b > a ? (2 / β) * (sqrt(max(1 - β * a, 0.0)) - sqrt(max(1 - β * b, 0.0))) : 0.0)
    wy_phys = zeros(Float64, ny)
    @inbounds for iy in 1:ny
        yn[iy] < yedge || continue                        # forbidden (no particles) ⇒ 0
        yl = iy == 1 ? yn[1] : (yn[iy-1] + yn[iy]) / 2    # cell = [midpoint-left, midpoint-right]
        # extend the last valid node's cell to the singular edge (don't split the
        # edge integral onto the zeroed forbidden node)
        yr = (iy < ny && yn[iy+1] >= yedge) ? yedge : (iy == ny ? yn[ny] : (yn[iy] + yn[iy+1]) / 2)
        wy_phys[iy] = Jint(yl, yr)
    end
    wE_phys = [grid.E.weights[iE] * sqrt(grid.E.nodes[iE]) / 2 for iE in 1:grid.E.n]
    return wy_phys, wE_phys
end

"""
    parallel_flow_weight(grid, phys) -> Array{Float64,3}

Build the **cleared** parallel-flow velocity weight `W[iy,iE,iσ] = v̂_∥ =
σ√E√(1−y b_min)` (`01 §4`, Q3; derivation `velocity-moment-measure.md`) for
`Operators.weighted_moment!` → `Moments.parallel_current!` → `J̄_∥`. With the
physical measure ([`physical_velocity_weights`]) the `√(1−y b_min)` cancels the
pitch Jacobian, so `J̄_∥ ∝ Σ_σ σ ∫dy∫dE (E/2) g` is regular. **σ-odd**; `b_min =
(1−ε)/(1+ε)`; `√(max(1−y b_min,0))` zeros the forbidden region.
"""
function parallel_flow_weight(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    β = (1 - phys.epsilon) / (1 + phys.epsilon)           # b_min
    W = Array{Float64}(undef, ny, nE, nσ)
    @inbounds for iσ in 1:nσ
        σ = grid.σ[iσ]
        for iE in 1:nE
            v̂ = sqrt(grid.E.nodes[iE])
            for iy in 1:ny
                W[iy, iE, iσ] = σ * v̂ * sqrt(max(1 - grid.y.nodes[iy] * β, 0.0))
            end
        end
    end
    return W
end

"""
    momentum_restoring_term(grid, phys) -> Operators.MomentumRestoring

Build the **cleared** momentum-restoring field-particle collision term (term F,
`01 §2.3`; `orbit-averaged-collision.md` §6, `velocity-moment-measure.md`), the
one nonlocal Level-0 term. It forms the parallel-flow moment
`Ū(x,ξ) = (1/√π⟨ν̂_ii⟩_u) {ν̂_ii v̂_∥ g}_v` (physical `∫d³v` measure) and adds the
σ-even redistribution `2ν̂_ii(√E)(1+ε)/(m ρ̂_θi) · Ū` (positive — ÷−m ρ̂_θi of the
RHS; the `F̂_M=e^{−E}` cancels in the `g=shape` convention). The moment weight is
`W = ν̂_ii(√E)·v̂_∥ = ν̂_ii·σ√E√(1−y b_min)`, and `⟨ν̂_ii⟩_u` is the cleared
`Coefficients.momentum_restoring_average` (`collision-magnitude.md`).
"""
function momentum_restoring_term(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    ε = phys.epsilon
    β = (1 - ε) / (1 + ε)                                 # b_min
    νu = momentum_restoring_average(; epsilon=ε, nu_star=phys.nu_star)   # ⟨ν̂_ii⟩_u
    νu != 0 || throw(ArgumentError("⟨ν̂_ii⟩_u = 0 (nu_star = 0); momentum restoring needs collisions"))
    inv_norm = 1 / (sqrt(π) * νu)
    W = Array{Float64}(undef, ny, nE, nσ)                 # ν̂_ii · v̂_∥ (σ-odd)
    redistribute = Vector{Float64}(undef, nE)             # 2ν̂_ii(1+ε)/(m ρ̂_θi), E-dependent, σ-even
    @inbounds for iE in 1:nE
        v̂ = sqrt(grid.E.nodes[iE])
        ν̂ii = _nu_hat_ii(phys, v̂)
        redistribute[iE] = 2 * ν̂ii * (1 + ε) / (phys.m * phys.rho_hat_theta_i)
        for iσ in 1:nσ
            σ = grid.σ[iσ]
            for iy in 1:ny
                W[iy, iE, iσ] = ν̂ii * σ * v̂ * sqrt(max(1 - grid.y.nodes[iy] * β, 0.0))  # ν̂_ii v̂_∥
            end
        end
    end
    wy_phys, wE_phys = physical_velocity_weights(grid, phys)
    return MomentumRestoring(W, redistribute, wy_phys, wE_phys, inv_norm)
end

"""
    gradient_far_field(grid, phys) -> FarFieldConditions

Build the **cleared** neoclassical far-field boundary state — the Level-0
gradient drive (`01 §2`; derivation `gradient-drive.md`, sign-off 2026-07-11).
I19's master equation is homogeneous (no interior source); the gradients enter
as the far field `Ḡ₀ → g_drive = p_φ F'_{Mi}`, which in the code normalization is

```math
g_{\\rm far}(x{=}\\pm L_x, \\xi, y, E, \\sigma) = x\\,\\hat L_{n0}^{-1}\\,[\\,1 + (E-\\tfrac32)\\eta_i\\,]
```

\\noindent
— linear in the boundary `x = ±halfwidth`, the temperature correction through
`E = v̂²`, isotropic in `ξ, y, σ` at leading order (the Maxwellian `e^{-E}` is
carried by the energy-grid measure). `L̂_{n0}⁻¹ = phys.inv_Ln0`,
`η_i = phys.eta_i`. At Level 0 `ω_E = 0`, so `Φ̂_far = 0`. Returns the
`Operators.FarFieldConditions` (the companion `Operators.GradientDrive` source is
zero — I19 Formulation A).

`mode` selects the radial far-field form (QUESTIONS Q7):

  - `:dirichlet` (default) — pin the value `g → g_far` (the `∝x` form above);
  - `:neumann` — pin the **slope** `∂g/∂x → ∂g_far/∂x = L̂_{n0}^{-1}[1+(E-\\tfrac32)η_i]`
    (the York/kokuchou localized form: `g` floats by a constant so the response
    reaches its own asymptote instead of forming an edge boundary layer). The slope
    is the `x`-derivative of the same linear far field, so the drive is unchanged.
"""
function gradient_far_field(grid::IslandGrid, phys::Level0Physics; mode::Symbol=:dirichlet)
    mode in (:dirichlet, :neumann) || throw(ArgumentError("gradient_far_field: mode must be :dirichlet or :neumann (got $mode)"))
    nx, nξ, ny, nE, nσ = nnodes(grid)
    x_left = grid.x.nodes[1]
    x_right = grid.x.nodes[nx]
    g_left = Array{Float64}(undef, nξ, ny, nE, nσ)
    g_right = Array{Float64}(undef, nξ, ny, nE, nσ)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        temp = 1 + (grid.E.nodes[iE] - 1.5) * phys.eta_i     # 1 + (E − 3/2)η_i
        slope = phys.inv_Ln0 * temp                          # ∂g_far/∂x = L̂_{n0}⁻¹[1+(E−3/2)η_i]
        if mode === :neumann
            g_left[iξ, iy, iE, iσ] = slope                   # pin the slope (localized/York form)
            g_right[iξ, iy, iE, iσ] = slope
        else
            g_left[iξ, iy, iE, iσ] = x_left * slope          # pin the value g_far ∝ x
            g_right[iξ, iy, iE, iσ] = x_right * slope
        end
    end
    # forbidden pitch region (y ≥ 1/b_min): no particles ⇒ pin g = 0 (else the
    # physically-zeroed collision coefficients leave these nodes unconstrained)
    y_forbidden = (1 + phys.epsilon) / (1 - phys.epsilon)
    forbidden_y = [grid.y.nodes[iy] >= y_forbidden for iy in 1:ny]
    return FarFieldConditions(g_left, g_right, zeros(nξ), zeros(nξ), forbidden_y; mode=mode)  # Φ̂_far = 0 (ω_E = 0)
end

"""
    quasineutrality_source(grid, phys) -> Matrix{Float64}

Build the **cleared** flattened-electron drive `S[ix, iξ] = L̂_{n0}⁻¹(x − ĥ(Ω))`
for `Operators.Quasineutrality` (`01 §3`; derivation `quasineutrality-closure.md`,
sign-off 2026-07-11). `Ω = 2x²/w² − cosξ` with `w = phys.w_psi`, and
`ĥ(Ω) = Coefficients.h_amplitude(w) · ∫₁^Ω dΩ′/Q(Ω′)` (`Fields.h_profile`, the
cleared far-field-matched profile: `ĥ = 0` inside the separatrix, `ĥ → x`
outside). `L̂_{n0}⁻¹ = phys.inv_Ln0`. This is the drive whose absence left the
Level-0 potential trivially zero (QUESTIONS Q5, now closed for the field term).
"""
function quasineutrality_source(grid::IslandGrid, phys::Level0Physics)
    nx, nξ = grid.x.n, grid.ξ.n
    w = phys.w_psi
    C = h_amplitude(w)                                 # cleared ĥ amplitude w/(2√2)
    S = Array{Float64}(undef, nx, nξ)
    @inbounds for iξ in 1:nξ
        cξ = cos(grid.ξ.nodes[iξ])
        for ix in 1:nx
            x = grid.x.nodes[ix]
            Ω = 2 * x^2 / w^2 - cξ                     # island label (Moments.omega_label)
            ĥ = h_profile(Ω; prefactor=C)              # cleared: 0 inside, →x outside
            S[ix, iξ] = phys.inv_Ln0 * (x - ĥ)
        end
    end
    return S
end

# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------
"""
    configure_level0(grid, phys, species; farfield_mode=:dirichlet)

Assemble the Level-0 named configuration (`03 §2`): returns a NamedTuple
`(stack, bc, delta_prefactors, cleared, gated)` where

  - `stack::Operators.IslandStack` — the operator stack (streaming, magnetic
    drift, `E×B`, the full orbit-averaged collision operator [mimetic pitch
    diffusion + `∂_x` drag + `∂²_x` neoclassical + `∂²_{xy}` cross], gradient
    drive) + the quasineutrality field term;
  - `bc` — the cleared far-field `Operators.FarFieldConditions` (the gradient
    drive, from [`gradient_far_field`]);
  - `delta_prefactors` — the cleared `(cos, sin)` `Δ`-moment prefactors;
  - `cleared`, `gated` — provenance tuples; **`gated` is now empty** — every
    Level-0 operator coefficient is built from `phys` via cleared `Coefficients.*`
    builders (QUESTIONS Q5 fully cleared).

Every coefficient is populated from the **cleared** builders: the magnetic drift
`c_D`, island streaming `a_xi`/`a_x`, `E×B` `c_E`, the orbit-averaged collision
coefficients (pitch diffusion `P_oa`/`K`/`c` [σ-odd], drag, neoclassical, cross;
`orbit-averaged-collision.md`), the quasineutrality field term (`α` + the
`L̂_{n0}⁻¹(x − ĥ)` drive), the gradient drive (zero source + far field `bc`), and
the `Δ` prefactors. The momentum-restoring field-particle term is a separate
future operator addition (its magnitude `⟨ν̂_ii⟩_u` is cleared).

`farfield_mode` selects the radial far-field boundary form ([`gradient_far_field`],
QUESTIONS Q7): `:dirichlet` (default, pin `g → g_far ∝ x` — the I19-Formulation-A
drive-in-the-BC form) or `:neumann` (pin `∂g/∂x → slope` — the York/kokuchou
localized form, so the response reaches its own asymptote rather than forming an
edge boundary layer). A toggle for the far-field comparison; the physical choice is
an open Q7 decision.

`species` is validated (a Level-0 config must have a bulk ion).
"""
function configure_level0(grid::IslandGrid, phys::Level0Physics, species::AbstractVector{<:Species}; farfield_mode::Symbol=:dirichlet)
    validate_species(species)
    nx, nξ, ny, nE, nσ = nnodes(grid)

    # cleared magnetic drift + island streaming + E×B (01 §2)
    c_D = drift_coefficient_table(grid, phys)
    a_xi, a_x = streaming_coefficients(grid, phys)
    c_E = exb_coupling_table(grid, phys)

    # cleared full orbit-averaged collision operator (01 §2.3; orbit-averaged-collision.md):
    #   D+E mimetic pitch diffusion (σ-odd, P_oa = y⟨√(1−yb)⟩, flat measure)
    P_oa, wmeas = pitch_diffusivity_profile(grid, phys)
    K, _ = conservative_pitch_operator(grid.y, P_oa, wmeas)
    c_pitch = pitch_collision_coefficient(grid, phys)
    a_drag = collisional_drag_coefficient(grid, phys)          # A: ∂_x drag
    c_neo = neoclassical_diffusion_coefficient(grid, phys)     # B: ∂²_x neoclassical
    c_cross = collisional_cross_coefficient(grid, phys)        # C: ∂²_xy cross
    mom_restore = momentum_restoring_term(grid, phys)          # F: nonlocal momentum restoring

    # cleared quasineutrality field term (α + drive from the signed-off closure),
    # with the cleared physical ∫d³v measure for δn̄_i = M[g] (velocity-moment-measure.md)
    α = 1 / quasineutrality_coefficient(phys.tau)              # (τ+1)/τ, adiabatic shielding
    S_Φ = quasineutrality_source(grid, phys)                   # L̂_{n0}⁻¹(x − ĥ)
    wy_phys, wE_phys = physical_velocity_weights(grid, phys)

    # cleared gradient drive (I19 Formulation A, gradient-drive.md): homogeneous
    # interior — zero source; the drive is the far field.
    drive0 = zeros(Float64, nx, nξ, ny, nE, nσ)
    bc = gradient_far_field(grid, phys; mode=farfield_mode)

    Δpref = delta_moment_prefactors(; mu0_R=phys.mu0_R, w_psi=phys.w_psi, dq_dpsi=phys.dq_dpsi, q_s=phys.q_s)

    kinetic = (
        ParallelStreaming(a_xi, a_x),                         # cleared (01 §2)
        MagneticDrift(c_D; variant=phys.variant),             # cleared
        ExBDrift(c_E),                                         # cleared (exb-coupling.md)
        PitchAngleDiffusion(K, c_pitch),                      # cleared collision D+E (σ-odd)
        CollisionalDrag(a_drag),                              # cleared collision A (∂_x)
        NeoclassicalDiffusion(c_neo),                         # cleared collision B (∂²_x)
        CollisionalCross(c_cross),                            # cleared collision C (∂²_xy)
        mom_restore,                                          # cleared collision F (momentum, nonlocal)
        GradientDrive(drive0)                                 # cleared: zero source (far field)
    )
    stack = IslandStack(kinetic, Quasineutrality(α, S_Φ, wy_phys, wE_phys))  # cleared closure + physical measure (01 §3, §4)

    return (stack=stack, bc=bc, delta_prefactors=Δpref,
        cleared=(:magnetic_drift, :streaming, :exb, :pitch_diffusion, :collisional_drag,
            :neoclassical_diffusion, :collisional_cross, :momentum_restoring, :collision_magnitude,
            :delta_prefactors, :quasineutrality, :gradient_drive, :far_field),
        gated=())
end

# ---------------------------------------------------------------------------
# Output assembly: solved state → the Δ growth/torque moments (01 §4)
# ---------------------------------------------------------------------------
"""
    delta_outputs(grid, phys, species, Usol, cfg)

Assemble the Level-0 output moments from a **converged** solve (`01 §4`), the
first physics deliverable of the solved state. Given the solved `Usol::IslandState`
(the bulk-ion distribution `g`), the physics parameters `phys`, the `species`
list, and the configuration `cfg` (from [`configure_level0`](@ref), carrying the
cleared `Δ` prefactors), it:

  - builds the cleared parallel-flow weight `W = v̂_∥` ([`parallel_flow_weight`])
    and the physical `∫d³v` measure ([`physical_velocity_weights`]);
  - forms the parallel current `J̄_∥(x, ξ) = Σ_j Z_j ∫ W_j g_j`
    (`Moments.parallel_current!`) — at Level 0 the single solved **bulk-ion**
    channel (the electron/species partition of `01 §4` is a later diagnostic);
  - projects `J̄_∥` onto the growth and torque moments with the cleared
    prefactors `∓μ₀R/2ψ̃`: `Δ_neo ≡ Δ_cos` (stationarity `Δ′ + Δ_neo = 0`) and
    `Δ_sin`, so `Δ_cos + iΔ_sin` maps onto the linear-layer `Δ(Q)`;
  - decomposes the growth moment into the bootstrap+curvature and polarization
    channels (`Moments.channel_decomposition`, an approximate diagnostic).

Returns a NamedTuple `(Jpar, Δ_neo, Δcos, Δsin, bootstrap_curvature,
polarization, omega_average_profile)`. Requires exactly one bulk species (the
Level-0 solve is single-bulk-ion; multi-bulk is a later milestone).
"""
function delta_outputs(grid::IslandGrid, phys::Level0Physics, species::AbstractVector{<:Species}, Usol::IslandState, cfg)
    validate_species(species)
    bulk = bulk_species(species)
    length(bulk) == 1 || throw(ArgumentError("delta_outputs: the Level-0 solve is single-bulk-ion (got $(length(bulk)) bulk species)"))

    W = parallel_flow_weight(grid, phys)
    wy, wE = physical_velocity_weights(grid, phys)
    Jpar = zeros(eltype(Usol.g), grid.x.n, grid.ξ.n)
    parallel_current!(Jpar, [Usol.g], bulk, [W], grid; wy=wy, wE=wE)

    pc, ps = cfg.delta_prefactors.cos, cfg.delta_prefactors.sin
    Δ = delta_moments(Jpar, grid; prefactor_cos=pc, prefactor_sin=ps)
    dec = channel_decomposition(Jpar, grid, phys.w_psi; prefactor_cos=pc)

    return (Jpar=Jpar, Δ_neo=Δ.Δcos, Δcos=Δ.Δcos, Δsin=Δ.Δsin,
        bootstrap_curvature=dec.bootstrap_curvature, polarization=dec.polarization,
        omega_average_profile=dec.omega_average_profile)
end

end # module Configure
