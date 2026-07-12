"""
    Configure

Level-0 **named-configuration assembly** (design `03 §2`): turns a physics
parameter set + a species list into a runnable `Operators.IslandStack`, its
far-field boundary conditions, and the `Δ`-moment prefactors — the object the
solver consumes.

**What this module does and does not do (the honest state, M2c).** The M2b
derivation lane cleared six Level-0 coefficient families. Only some of those map
onto operator-stack coefficients *cleanly*; the assembly wires exactly those and
**gates the rest**:

  - **Cleared, wired here** (populated by `Coefficients.*`, never literals):
    the orbit-averaged magnetic drift `c_D` ([`drift_coefficient_table`], from
    [`Coefficients.magnetic_drift_frequency`](@ref)); the pitch-collision
    *shapes* ([`Coefficients.pitch_diffusivity`](@ref) →
    `Operators.conservative_pitch_operator`, and
    [`Coefficients.deflection_frequency`](@ref) → the energy-dependent collision
    coefficient); and the `Δ`-moment prefactors
    ([`Coefficients.delta_moment_prefactors`](@ref)).
  - **Not yet a cleared coefficient family → supplied, gated** (QUESTIONS Q5):
    the parallel-streaming coefficients `a_xi`/`a_x`, the `E×B` coupling `c_E`,
    the gradient-drive source `drive`, the quasineutrality operator coefficient
    `α`, the orbit-averaged pitch measure/field `B_profile`, the collision
    magnitude `nu_tilde` (carries the deferred `⟨ν̂_ii⟩_u`), and the neoclassical
    far-field state. These enter through [`GatedLevel0Inputs`]; nothing here
    assigns a physics value to them.

The gated pieces are the subject of a future derivation lane (QUESTIONS Q5): the
assembly *surfaces* exactly what Level-0 physics is still uncleared rather than
papering over it with guesses. [`level0_placeholders`](@ref) supplies documented
**non-physics** values for the gated inputs so the assembled residual/solve can
be exercised *structurally* (that the stack is well-formed and solvable) — never
for a physics result.
"""
module Configure

using LinearAlgebra
import ..PhaseSpace: IslandGrid, nnodes, MappedFDGrid
import ..Operators: IslandStack, FarFieldConditions,
    ParallelStreaming, MagneticDrift, ExBDrift, PitchAngleDiffusion,
    GradientDrive, Quasineutrality, conservative_pitch_operator
import ..Coefficients: magnetic_drift_frequency, orbit_average_drift_brackets,
    pitch_diffusivity, deflection_frequency, delta_moment_prefactors
import ..SpeciesLists: Species, Maxwellian, Bulk, validate_species

export Level0Physics, GatedLevel0Inputs, configure_level0, level0_placeholders
export drift_coefficient_table, collision_coefficient, pitch_diffusivity_profile

# ---------------------------------------------------------------------------
# Physics parameter carrier (the cleared inputs)
# ---------------------------------------------------------------------------
"""
    Level0Physics(; epsilon, inv_Lq, inv_LB, q_s, dq_dpsi, w_psi, mu0_R,
                    tau=1.0, variant=:original, collision_model=:chandrasekhar)

The Level-0 physics parameters that feed the **cleared** coefficient builders
(`01 §1`–`§4`). These are ordinary scenario inputs, not gated coefficients — the
gated (uncleared) pieces live in [`GatedLevel0Inputs`].

## Fields

  - `epsilon`         — inverse aspect ratio `ε = r_s/R₀`.
  - `inv_Lq`          — `L̂_q⁻¹ = (ψ_s/q_s) dq/dψ` at `r_s` (drift, `01 §2.1`).
  - `inv_LB`          — `L̂_B⁻¹` (∇B drift term; forced to `0` when
    `variant = :improved`, `01 §2.1`).
  - `q_s`, `dq_dpsi`  — safety factor and its `ψ`-derivative at `r_s`
    (`ψ̃` and `Δ`-prefactor, `01 §1`, `§4`).
  - `w_psi`           — island **half**-width in `ψ`-space (`ψ̃`, `01 §1`).
  - `mu0_R`           — `μ₀R` geometric factor of the `Δ`-moment prefactor
    (`01 §4`).
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
    tau::Float64 = 1.0
    variant::Symbol = :original
    collision_model::Symbol = :chandrasekhar
end

# ---------------------------------------------------------------------------
# Gated inputs carrier (the uncleared pieces — supplied, never guessed here)
# ---------------------------------------------------------------------------
"""
    GatedLevel0Inputs(; a_xi, a_x, c_E, drive, alpha, nu_tilde, B_profile, bc)

The Level-0 operator coefficients that are **not yet a cleared coefficient
family** and are therefore *supplied* to [`configure_level0`], not derived from
`Coefficients.*` (QUESTIONS Q5). Nothing in this module assigns them a physics
value; a caller either supplies cleared physics (once available) or the
documented non-physics placeholders of [`level0_placeholders`](@ref).

## Fields

  - `a_xi`, `a_x`  — parallel-streaming coefficients, shaped like `g`
    (`Operators.ParallelStreaming`; uncleared, QUESTIONS Q5).
  - `c_E`          — `E×B` coupling scalar (`Operators.ExBDrift`; the frame/
    normalization is uncleared, QUESTIONS Q3/Q5).
  - `drive`        — gradient-drive source, shaped like `g`
    (`Operators.GradientDrive`; `(v·∇F₀)` with the frame shift NaN-gated in
    `Frames`, QUESTIONS Q3/Q5).
  - `alpha`        — quasineutrality operator coefficient of
    `Operators.Quasineutrality` (`R_Φ = M[g] − α Φ`). **Structural gap**
    (QUESTIONS Q5): the cleared closure `Φ̂ = τ/(τ+1)[δn̄_i/n₀ + L̂_{n0}⁻¹(x−ĥ)]`
    implies `α = (τ+1)/τ` *and* an `L̂_{n0}⁻¹(x−ĥ)` field source the current
    operator does not carry — so `α` is not wired from
    `quasineutrality_coefficient` here.
  - `nu_tilde`     — collision magnitude scaling the cleared `ν_{jj}(v̂)` shape;
    carries the deferred `⟨ν̂_ii⟩_u`/`ν_★` normalization (QUESTIONS Q3).
  - `B_profile`    — orbit-averaged `|B|/B_max` on the `y`-grid, feeding the
    cleared `pitch_diffusivity` shape and the collision measure (the orbit
    average is gated, QUESTIONS Q5).
  - `bc`           — `Operators.FarFieldConditions`: the neoclassical no-island
    far field (gated physics, QUESTIONS Q3).
"""
Base.@kwdef struct GatedLevel0Inputs{A5,S,V,BC}
    a_xi::A5
    a_x::A5
    c_E::S
    drive::A5
    alpha::S
    nu_tilde::S
    B_profile::V
    bc::BC
end

# ---------------------------------------------------------------------------
# Cleared coefficient wiring
# ---------------------------------------------------------------------------
"""
    drift_coefficient_table(grid, phys) -> Array{Float64,5}

Build the orbit-averaged magnetic-drift coefficient `c_D[ix, iξ, iy, iE, iσ]` for
`Operators.MagneticDrift` by evaluating the **cleared**
[`Coefficients.magnetic_drift_frequency`](@ref) on the phase-space grid
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

"""
    pitch_diffusivity_profile(grid, B_profile) -> (P, wmeas)

Evaluate the **cleared** Lorentz pitch diffusivity
[`Coefficients.pitch_diffusivity`](@ref) `P(λ) = λ√(1−λB)` and the collision
measure `w = B/√(1−λB)` on the `y`-grid (`01 §2.3`), with `λ = y` in the
`B_max = 1` normalization and `B = B_profile[iy]` the (gated) orbit-averaged
field. Returns `(P, wmeas)` for `Operators.conservative_pitch_operator`, which
builds the mimetic operator preserving the A4 conservation gate for any `P ≥ 0`.
The caller must supply a `B_profile` keeping `0 ≤ y·B ≤ 1` on the grid (the
turning-point structure is part of the gated orbit average, QUESTIONS Q5).
"""
function pitch_diffusivity_profile(grid::IslandGrid, B_profile::AbstractVector)
    ny = grid.y.n
    length(B_profile) == ny || throw(ArgumentError("B_profile must have length ny = $ny"))
    P = Vector{Float64}(undef, ny)
    wmeas = Vector{Float64}(undef, ny)
    @inbounds for iy in 1:ny
        λ = grid.y.nodes[iy]
        B = B_profile[iy]
        arg = 1 - λ * B
        (0 <= λ * B <= 1) || throw(ArgumentError("λB = $(λ * B) at iy=$iy outside [0,1]; supply a valid orbit-averaged B_profile"))
        P[iy] = pitch_diffusivity(λ, B)                   # cleared: λ√(1−λB)
        # collision measure w = B/√(1−λB); regularize the zero-flux edge (√arg→0)
        wmeas[iy] = B / sqrt(max(arg, 1e-12))
    end
    return P, wmeas
end

"""
    collision_coefficient(grid, phys, nu_tilde) -> Array{Float64,4}

Build the energy-dependent collision coefficient `c[ix, iξ, iE, iσ]` for
`Operators.PitchAngleDiffusion` from the **cleared**
[`Coefficients.deflection_frequency`](@ref) `ν_{jj}(v̂)` (`01 §2.3`), scaled by the
gated magnitude `nu_tilde` (QUESTIONS Q3, carries `⟨ν̂_ii⟩_u`/`ν_★`). It is
**`y`-independent by construction** (dimensions `(x, ξ, E, σ)`), as
`PitchAngleDiffusion` requires so the mimetic `K`'s exact conservation is
preserved; the physical velocity dependence lives entirely in the `E`-axis via
`v̂ = √E`. Uses `phys.collision_model` for the `:chandrasekhar`/`:vcubed` toggle.
"""
function collision_coefficient(grid::IslandGrid, phys::Level0Physics, nu_tilde::Real)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    c = Array{Float64}(undef, nx, nξ, nE, nσ)
    @inbounds for iE in 1:nE
        v̂ = sqrt(grid.E.nodes[iE])
        ν = deflection_frequency(v̂; nu_tilde=nu_tilde, model=phys.collision_model)
        for iσ in 1:nσ, iξ in 1:nξ, ix in 1:nx
            c[ix, iξ, iE, iσ] = ν
        end
    end
    return c
end

# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------
"""
    configure_level0(grid, phys, species; gated)

Assemble the Level-0 named configuration (`03 §2`): returns a NamedTuple
`(stack, bc, delta_prefactors, cleared, gated)` where

  - `stack::Operators.IslandStack` — the operator stack (streaming, magnetic
    drift, `E×B`, mimetic pitch collisions, gradient drive) + the
    quasineutrality field term;
  - `bc` — the far-field `Operators.FarFieldConditions` (from `gated.bc`);
  - `delta_prefactors` — the cleared `(cos, sin)` `Δ`-moment prefactors
    ([`Coefficients.delta_moment_prefactors`](@ref));
  - `cleared`, `gated` — the provenance tuples naming which coefficients came
    from cleared `Coefficients.*` builders vs. supplied gated inputs.

The magnetic drift `c_D`, the pitch-collision `P`/`K` and `c`, and the `Δ`
prefactors are populated from the **cleared** coefficient builders. The
streaming, `E×B`, gradient-drive, quasineutrality-`α`, collision magnitude,
pitch measure, and far field are **supplied** through `gated::GatedLevel0Inputs`
(QUESTIONS Q3/Q5) — this function assigns no physics value to them.

`species` is validated (a Level-0 config must have a bulk ion); its
per-species roles/backgrounds drive the gated builders that are not yet cleared.
"""
function configure_level0(grid::IslandGrid, phys::Level0Physics, species::AbstractVector{<:Species};
    gated::GatedLevel0Inputs)
    validate_species(species)

    # cleared coefficient wiring
    c_D = drift_coefficient_table(grid, phys)
    P, wmeas = pitch_diffusivity_profile(grid, gated.B_profile)
    K, _ = conservative_pitch_operator(grid.y, P, wmeas)
    c_coll = collision_coefficient(grid, phys, gated.nu_tilde)
    Δpref = delta_moment_prefactors(; mu0_R=phys.mu0_R, w_psi=phys.w_psi, dq_dpsi=phys.dq_dpsi, q_s=phys.q_s)

    kinetic = (
        ParallelStreaming(gated.a_xi, gated.a_x),          # gated
        MagneticDrift(c_D; variant=phys.variant),          # cleared
        ExBDrift(gated.c_E),                               # gated
        PitchAngleDiffusion(K, c_coll),                    # cleared shape (magnitude gated)
        GradientDrive(gated.drive)                         # gated
    )
    stack = IslandStack(kinetic, Quasineutrality(gated.alpha))   # α gated (QUESTIONS Q5)

    return (stack=stack, bc=gated.bc, delta_prefactors=Δpref,
        cleared=(:magnetic_drift, :pitch_diffusivity, :deflection_frequency, :delta_prefactors),
        gated=(:streaming, :exb, :gradient_drive, :quasineutrality_alpha, :nu_tilde, :pitch_measure, :far_field))
end

# ---------------------------------------------------------------------------
# Documented non-physics placeholders (structural runs only)
# ---------------------------------------------------------------------------
"""
    level0_placeholders(grid; drive_amp=0.1, c_E=0.0, alpha=1.0, nu_tilde=1.0,
                        B_edge=0.999)

Build a [`GatedLevel0Inputs`] of **documented non-physics placeholder** values so
the assembled stack can be exercised *structurally* — that `configure_level0`
produces a well-formed, solvable `IslandStack` — **never for a physics result**
(the gated coefficients are uncleared, QUESTIONS Q5). Choices:

  - streaming `a_xi = a_x = 1` (order-unity advection);
  - `c_E = 0` (drop the `E×B` nonlinearity so the structural residual is linear);
  - a smooth localized `drive` of amplitude `drive_amp`;
  - `alpha = 1`, `nu_tilde = 1` (order-unity);
  - a `B_profile` that keeps `y·B ≤ 1` on the grid (`B = min(1, B_edge/y)`), so
    the cleared `pitch_diffusivity` stays in-domain;
  - zero far-field states (homogeneous Dirichlet at `|x| = L_x`).

Every value is a structural stand-in; a physics run supplies cleared inputs.
"""
function level0_placeholders(grid::IslandGrid; drive_amp::Real=0.1, c_E::Real=0.0,
    alpha::Real=1.0, nu_tilde::Real=1.0, B_edge::Real=0.999)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    ones5 = ones(Float64, nx, nξ, ny, nE, nσ)
    drive = Array{Float64}(undef, nx, nξ, ny, nE, nσ)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ, ix in 1:nx
        x = grid.x.nodes[ix]
        ξ = grid.ξ.nodes[iξ]
        drive[ix, iξ, iy, iE, iσ] = drive_amp * exp(-x^2) * cos(ξ)
    end
    # B_profile ≤ 1/y keeps the cleared pitch_diffusivity in [0,1]; B ≤ 1 (B_max norm)
    B_profile = [min(1.0, B_edge / max(grid.y.nodes[iy], eps())) for iy in 1:ny]
    bc = FarFieldConditions(zeros(nξ, ny, nE, nσ), zeros(nξ, ny, nE, nσ), zeros(nξ), zeros(nξ))
    return GatedLevel0Inputs(; a_xi=copy(ones5), a_x=copy(ones5), c_E=Float64(c_E),
        drive=drive, alpha=Float64(alpha), nu_tilde=Float64(nu_tilde), B_profile=B_profile, bc=bc)
end

end # module Configure
