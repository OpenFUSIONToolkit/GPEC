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
    `Coefficients.magnetic_drift_frequency`); the **island-streaming**
    coefficients `a_xi`/`a_x` ([`streaming_coefficients`], the `{Ω, ·}`
    flux-surface advection, `01 §2`); the **`E×B` coupling** `c_E`
    ([`exb_coupling_table`], `½⟨1/v̂_∥⟩_θ`, passing-only, `01 §2`); the
    pitch-collision *shapes* (`Coefficients.pitch_diffusivity` →
    `Operators.conservative_pitch_operator`, and
    `Coefficients.deflection_frequency` → the energy-dependent collision
    coefficient); the **quasineutrality field term** — `α = (τ+1)/τ` from
    `Coefficients.quasineutrality_coefficient` and the `L̂_{n0}⁻¹(x − ĥ)` drive
    ([`quasineutrality_source`], from the cleared `ĥ` profile), closing the
    Level-0 potential (`01 §3`); the **gradient drive** — zero interior source
    plus the neoclassical far field `g_far = x L̂_{n0}⁻¹[1+(E−3/2)η_i]`
    ([`gradient_far_field`], I19 Formulation A, `01 §2`); and the `Δ`-moment
    prefactors (`Coefficients.delta_moment_prefactors`).
  - **Not yet a cleared coefficient family → supplied, gated** (QUESTIONS Q5):
    the orbit-averaged pitch measure/field `B_profile` and the collision
    magnitude `nu_tilde` (carries the deferred `⟨ν̂_ii⟩_u`). These enter through
    [`GatedLevel0Inputs`]; nothing here assigns a physics value to them.

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
import ..Coefficients:
    magnetic_drift_frequency, orbit_average_drift_brackets, orbit_average_exb_bracket,
    pitch_diffusivity, deflection_frequency, delta_moment_prefactors,
    h_amplitude, quasineutrality_coefficient
import ..Fields: h_profile
import ..SpeciesLists: Species, Maxwellian, Bulk, validate_species

export Level0Physics, GatedLevel0Inputs, configure_level0, level0_placeholders
export drift_coefficient_table, collision_coefficient, pitch_diffusivity_profile
export quasineutrality_source, streaming_coefficients, gradient_far_field
export exb_coupling_table

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
    tau::Float64 = 1.0
    variant::Symbol = :original
    collision_model::Symbol = :chandrasekhar
end

# ---------------------------------------------------------------------------
# Gated inputs carrier (the uncleared pieces — supplied, never guessed here)
# ---------------------------------------------------------------------------
"""
    GatedLevel0Inputs(; nu_tilde, B_profile)

The Level-0 operator coefficients that are **not yet a cleared coefficient
family** and are therefore *supplied* to [`configure_level0`], not derived from
`Coefficients.*` (QUESTIONS Q5). Nothing in this module assigns them a physics
value; a caller either supplies cleared physics (once available) or the
documented non-physics placeholders of [`level0_placeholders`](@ref).

## Fields

  - `nu_tilde`     — collision magnitude scaling the cleared `ν_{jj}(v̂)` shape;
    carries the deferred `⟨ν̂_ii⟩_u`/`ν_★` normalization (QUESTIONS Q3).
  - `B_profile`    — orbit-averaged `|B|/B_max` on the `y`-grid, feeding the
    cleared `pitch_diffusivity` shape and the collision measure (the orbit
    average is gated, QUESTIONS Q5).
"""
Base.@kwdef struct GatedLevel0Inputs{S,V}
    nu_tilde::S
    B_profile::V
end

# ---------------------------------------------------------------------------
# Cleared coefficient wiring
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

"""
    pitch_diffusivity_profile(grid, B_profile) -> (P, wmeas)

Evaluate the **cleared** Lorentz pitch diffusivity
`Coefficients.pitch_diffusivity` `P(λ) = λ√(1−λB)` and the collision
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
`Coefficients.deflection_frequency` `ν_{jj}(v̂)` (`01 §2.3`), scaled by the
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
"""
function gradient_far_field(grid::IslandGrid, phys::Level0Physics)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    x_left = grid.x.nodes[1]
    x_right = grid.x.nodes[nx]
    g_left = Array{Float64}(undef, nξ, ny, nE, nσ)
    g_right = Array{Float64}(undef, nξ, ny, nE, nσ)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iy in 1:ny, iξ in 1:nξ
        temp = 1 + (grid.E.nodes[iE] - 1.5) * phys.eta_i     # 1 + (E − 3/2)η_i
        g_left[iξ, iy, iE, iσ] = x_left * phys.inv_Ln0 * temp
        g_right[iξ, iy, iE, iσ] = x_right * phys.inv_Ln0 * temp
    end
    return FarFieldConditions(g_left, g_right, zeros(nξ), zeros(nξ))  # Φ̂_far = 0 (ω_E = 0)
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
    configure_level0(grid, phys, species; gated)

Assemble the Level-0 named configuration (`03 §2`): returns a NamedTuple
`(stack, bc, delta_prefactors, cleared, gated)` where

  - `stack::Operators.IslandStack` — the operator stack (streaming, magnetic
    drift, `E×B`, mimetic pitch collisions, gradient drive) + the
    quasineutrality field term;
  - `bc` — the cleared far-field `Operators.FarFieldConditions` (the gradient
    drive, from [`gradient_far_field`]);
  - `delta_prefactors` — the cleared `(cos, sin)` `Δ`-moment prefactors
    (`Coefficients.delta_moment_prefactors`);
  - `cleared`, `gated` — the provenance tuples naming which coefficients came
    from cleared `Coefficients.*` builders vs. supplied gated inputs.

The magnetic drift `c_D`, the island streaming `a_xi`/`a_x`, the `E×B` coupling
`c_E`, the pitch-collision `P`/`K` and `c`, the quasineutrality field term (`α` +
the `L̂_{n0}⁻¹(x − ĥ)` drive), the gradient drive (zero source + the far field
`bc`), and the `Δ` prefactors are populated from the **cleared** coefficient
builders. The collision magnitude and pitch measure are **supplied** through
`gated::GatedLevel0Inputs` (QUESTIONS Q5) — this function assigns no physics
value to them.

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

    # cleared island streaming (advection along Ω; parallel-streaming.md)
    a_xi, a_x = streaming_coefficients(grid, phys)
    # cleared E×B coupling (½⟨1/v̂_∥⟩, passing-only; exb-coupling.md)
    c_E = exb_coupling_table(grid, phys)
    # cleared quasineutrality field term (α + drive from the signed-off closure)
    α = 1 / quasineutrality_coefficient(phys.tau)          # (τ+1)/τ, adiabatic shielding
    S_Φ = quasineutrality_source(grid, phys)               # L̂_{n0}⁻¹(x − ĥ)

    # cleared gradient drive (I19 Formulation A, gradient-drive.md): the master
    # equation is homogeneous — zero interior source; the drive is the far field.
    nx, nξ, ny, nE, nσ = nnodes(grid)
    drive0 = zeros(Float64, nx, nξ, ny, nE, nσ)
    bc = gradient_far_field(grid, phys)

    kinetic = (
        ParallelStreaming(a_xi, a_x),                      # cleared (01 §2)
        MagneticDrift(c_D; variant=phys.variant),          # cleared
        ExBDrift(c_E),                                      # cleared (01 §2, exb-coupling.md)
        PitchAngleDiffusion(K, c_coll),                    # cleared shape (magnitude gated)
        GradientDrive(drive0)                              # cleared: zero source (drive is the far field)
    )
    stack = IslandStack(kinetic, Quasineutrality(α, S_Φ))  # cleared closure (01 §3)

    return (stack=stack, bc=bc, delta_prefactors=Δpref,
        cleared=(:magnetic_drift, :streaming, :exb, :pitch_diffusivity, :deflection_frequency, :delta_prefactors, :quasineutrality, :gradient_drive, :far_field),
        gated=(:nu_tilde, :pitch_measure))
end

# ---------------------------------------------------------------------------
# Documented non-physics placeholders (structural runs only)
# ---------------------------------------------------------------------------
"""
    level0_placeholders(grid; nu_tilde=1.0, B_edge=0.999)

Build a [`GatedLevel0Inputs`] of **documented non-physics placeholder** values so
the assembled stack can be exercised *structurally* — that `configure_level0`
produces a well-formed, solvable `IslandStack` — **never for a physics result**
(the remaining gated coefficients are uncleared, QUESTIONS Q5). Choices:

  - `nu_tilde = 1` (order-unity);
  - a `B_profile` that keeps `y·B ≤ 1` on the grid (`B = min(1, B_edge/y)`), so
    the cleared `pitch_diffusivity` stays in-domain.

Every value is a structural stand-in; a physics run supplies cleared inputs. The
`E×B` coupling, gradient drive, and far field are now *cleared* (built by
`configure_level0` from `phys`), so they are no longer placeholders here.
"""
function level0_placeholders(grid::IslandGrid; nu_tilde::Real=1.0, B_edge::Real=0.999)
    ny = grid.y.n
    # B_profile ≤ 1/y keeps the cleared pitch_diffusivity in [0,1]; B ≤ 1 (B_max norm)
    B_profile = [min(1.0, B_edge / max(grid.y.nodes[iy], eps())) for iy in 1:ny]
    return GatedLevel0Inputs(; nu_tilde=Float64(nu_tilde), B_profile=B_profile)
end

end # module Configure
