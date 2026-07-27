# 02 — Species abstraction, tungsten, and energetic particles

## 1. Species as a first-class dimension (Level 0 requirement, D3)

Every kinetic object in Islands is indexed by species. The solve is per-species DKEs
coupled through (i) quasineutrality (and Ampère at L3), (ii) the collision
operator's field-particle terms (L1+), and (iii) the output moments, which sum
over species. Designing for N species at Level 0 costs ~nothing (the L0 test is a
trace deuterium copy of the bulk); retrofitting costs a rewrite.

### 1.1 Species definition

```julia
abstract type AbstractBackground end
struct Maxwellian    <: AbstractBackground  # n, T, dlnn/dr, dlnT/dr at r_s
struct SlowingDown   <: AbstractBackground  # S0, v_birth, v_crit(T_e, composition),
                                            # dln(source)/dr ; isotropic at L2 entry
struct Species{B<:AbstractBackground}
    name::Symbol
    Z::Float64            # charge number (allow Float for mean-charge-state W)
    m::Float64            # mass ratio to reference ion
    background::B
    role::SpeciesRole
    collisional_coupling::Bool   # participate in field-particle terms? (see §1.3)
end
```

### 1.2 Roles: the trace-species economy

```
@enum SpeciesRole Bulk Trace
```

- **Bulk**: full nonlinear participant. In quasineutrality (+ Ampère at L3);
  its g enters the Newton state vector.
- **Trace** (n_j Z_j ≪ n_e for charge, n_j ≪ n_e for current — check both): the
  trace DKE is *linear in g_j* given the converged bulk fields (Φ̃, island, bulk
  flows for friction). Solved as a post-processing pass — one linear solve, no
  Newton coupling — with an additive contribution to Δ_cos/Δ_sin and to profiles.
  This is the computational backbone of both the W and alpha tracks: parameter
  scans over trace-species properties reuse one bulk solve.
- Promotion rule: any species violating trace criteria (e.g. W at high
  concentration where Z n_W is non-negligible, or when friction back-reaction on
  bulk flows matters — see §2) must be run as Bulk; the code checks the criteria
  and warns, never silently degrades.

### 1.3 Collisional coupling matrix

Friction is directional: a trace species always *feels* the bulk (drag,
pitch-angle scattering off bulk); whether the bulk feels the trace
(field-particle back-reaction) is the `collisional_coupling` flag. W at reactor-
relevant concentrations: back-reaction ON (Z² n_W friction on bulk ions is not
small even when charge-trace holds — this is exactly the mixed case the analytic
MRE cannot do). Dilute alphas: back-reaction OFF is usually safe; verify with the
flag flip (a one-line toggle — the whole point of the architecture).

---

## 2. Tungsten (physics gate: Level 1; radiative channel: Level 4)

### 2.1 Why W breaks the analytic MRE terms

Collisionality scales ~ Z²(?) with low v_th: W sits in Pfirsch–Schlüter or plateau
while bulk D and electrons are banana — a *mixed-regime* multi-species problem.
Analytic bootstrap terms in the MRE assume a per-species regime; the friction
between a PS impurity and banana bulk ions modifies the bootstrap current (and
hence Δ_bs) in ways only a full multi-species collision operator captures.
Additional channels: Z_eff shift of electron collisionality (electron bootstrap
and, at L3, resistive layer physics); impurity contribution to polarization is
small (tiny ρ_θW) — a prediction to *verify*, not assume.

### 2.2 Level-1 deliverables

- Δ_bs(w; n_W, Z_W, ν̂) surfaces with W friction — quantified departure from
  Sauter-based MRE terms.
- **In-island impurity asymmetry**: n_W(Ω, ξ) structure (parallel compression +
  friction with flattened bulk flows inside the island). Directly comparable to
  AUG/DIII-D measurements of W behavior at islands, and the input to the L4
  radiative channel.
- Charge state: single mean-Z at L1 (Z̄(T_e) from coronal tables as a parameter);
  multi-charge-state bundle only if asymmetry results prove sensitive.

### 2.3 Level-4 radiative/thermo-resistive channel

Energy closure with radiation sink Q_rad = n_e n_W L_W(T_e) inside the island,
temperature-dependent resistivity η(T_e) entering the L3 Ohm/Ampère system →
radiation-driven island growth (Gates & Delgado-Aparicio-class mechanism,
density-limit relevance). Gate: reproduce published thermo-resistive island model
trends (docs/05 D3). Depends on: L1 (W transport into island) + L3 (η in field
equation) + L4 energy closure. This is deliberately the *last* W milestone.

---

## 3. Energetic particles (physics gate: Level 2)

### 3.1 Why EPs are the reason Level 2 exists

Alphas violate the orderings W leaves intact:

- **ρ_θα ≳ w** (and possibly ≳ L_x): the drift-island shift is not a perturbative
  O(ρ_θ) displacement — it is the dominant structure. Radially local, constant-
  gradient assumptions fail; the domain must span several ρ_θα with profile
  variation toggles.
- **Orbit-average survives, resonance does not**: ω_bα is still fast (O5 holds for
  alphas), but island rotation can satisfy **ω ~ ω_D,α** (precession) — a
  collisionless, resonant polarization-type contribution to Δ with no fluid
  analog and no regime formula. Flagship EP physics target: ITER/SPARC NTM
  thresholds with self-consistent alpha kinetics.
- **Non-Maxwellian F₀**: slowing-down background changes the drive terms
  (∂F₀/∂r structure, no temperature gradient in the usual sense) and the
  collision physics (drag on electrons + bulk ions rather than self-collisions;
  self-collisions negligible). Mechanically modest once L2 orbits are right;
  pointless before.

### 3.2 Implementation sequence within Level 2

1. Trace Maxwellian "hot ion" with artificially large ρ_θ — isolates finite-orbit
   nonlocality from F₀ shape.
2. SlowingDown F₀, isotropic — adds drive/drag changes.
3. ω-scan through ω_D,α — the precession-resonance study. Requires the L2
   orbit machinery to deliver accurate ω_D(λ, E) (benchmark vs. standalone orbit
   integrator, docs/05 C2). Methodological antecedent: Dudkovskaia et al. JPCS
   1125, 012009 (2018) — *phase-space* island stability for EP-driven modes;
   its bounce/angle-variable and separatrix-layer machinery is the same toolkit
   (docs/08), though its physics (bump-on-tail secondary modes) is not part of
   this study.
4. (Optional/later) anisotropic F₀ for NBI/RF fast ions — same machinery,
   different B; parked unless a collaborator needs it.

### 3.3 Division of labor with the outer region

EP pressure also modifies Δ′ (outer-region kinetic corrections — same physics
class as GPEC/PENT stability integrals). That stays on the perturbed-equilibrium
side and arrives through the Δ′(w) input. Islands owns only resonant/orbit-width EP
physics at the island. State this in every EP paper to preempt double-counting
questions, and define the split precisely: outer kinetic Δ′ evaluated with the
island region excised at the matching radius |x| = L_x [interface spec in
docs/03 §5].

### 3.4 Alpha–W interplay (free deliverable)

Both tracks live in the same species list; an L2 run with {D bulk, e bulk/model,
W trace, α trace} costs one bulk solve + two linear passes. Alpha drag heating
asymmetries vs. W radiative cooling inside islands is an unexplored combination —
cheap to look at once both tracks exist, potentially interesting for burning-
plasma island stability. Not a milestone; an opportunity.
