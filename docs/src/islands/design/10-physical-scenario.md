# 10 — The pinned physical scenario (plan)

**Status:** plan (2026-07-23). Motivated by the parameter audit
(`notes/physical-parameter-audit.md`): the Level-0 code runs on hand-set normalized
numbers with **no physical grounding** — `ρ̂_θi`, `ν_★`, `L̂_n0⁻¹`, `η_i`, `L̂_q⁻¹`
are independent knobs, not derived from any `T_i/B/R/a`, and the `Species` `T`/`n`/
gradient fields are inert. This doc plans a **pinned physical scenario**: one real
equilibrium + profiles → a **derived, self-consistent** normalized input vector, so a
`w`-scan is a scan *at fixed physical plasma* with a *fixed physical matching radius*
(never `Lx∝w`).

## Why this is the fix, not a nicety

Every pathology in the 2026-07 stall traces to ungrounded normalization: islands and
domains larger than the plasma (`ŵ=0.5`, `Lx=20w=1.0`, i.e. past the magnetic axis at
`x=−1`), a "convergence" bought by a plasma-sized box, and a `Δ_neo` that drifts with
an arbitrary domain. With a pinned scenario, `w`, the domain, `ρ̂_θi`, `ν_★` are all in
physical proportion and the questions become well-posed:
- **scan** `w/ρ_θi` from `≪1` (vanishing island) to `~few` (large), at **fixed** ε,
  `ν_★`, profiles;
- **domain** fixed at a physical matching radius `x_match` (a fraction of `r_s`, e.g.
  to the nearest neighbouring rational surface), independent of `w`;
- **target** `Δ_neo(w)` finite as `w→0` (ρ_θi-regularized), recovering `∝1/w` at
  `w≫ρ_θi` — testable because the numbers are physical.

## Reuse the inbuilt large-aspect-ratio example (do NOT invent a new equilibrium)

`examples/a10_kinetic_example` is the ready-made basis: an EFIT g-file
(`fix_a100_k10_q2_bn010_prof1`) — large aspect ratio, **q=2 surface**, circular
(`k10`≈κ=1), β_N=0.10 — plus a kinetic profile `a10_prof1.txt` with columns
`ψ, n_i[m⁻³], n_e, T_i[eV], T_e[eV], ω_E[rad/s]` (n_i≈2×10¹⁸ m⁻³, T_i≈0.48 keV).
`examples/LAR_epsilon_scan` / `LAR_beta_scan` (the analytic `lar`
`AnalyticEquilibrium`) are the alternative if a purely-analytic ε is wanted for a
clean scan. **Use `setup_equilibrium` to ingest it** (the same path the rest of GPEC
uses); read `q(ψ)`, `R(ψ,θ)`, `B` from the `rzphi`/`sq` splines, and `T_i(ψ)`,
`n_i(ψ)` from the kinetic-profile ingest already in `KineticForces`.

## The derivation: physical equilibrium → normalized Level-0 vector

At the rational surface `ψ_s` where `q(ψ_s)=m/n=2` (locate via the `sq` q-spline):

| normalized input | physical formula (SI, at `ψ_s`) | source |
|---|---|---|
| `ε` | `r_s/R_0` | g-file geometry |
| `q_s` | `q(ψ_s) = 2` | q-spline |
| `inv_Lq` `L̂_q⁻¹` | `(ψ_s/q) dq/dψ|_s` | q-spline derivative |
| `v_thi` | `√(2 T_i/m_i)` | profile `T_i(ψ_s)` |
| `ρ_i` | `√(2 T_i m_i)/(Z e B)` | `T_i`, `B` at `ψ_s` |
| `ρ_θi` | `ρ_i · B/B_θ = ρ_i · q/ε` (LAR) | above + `ε`, `q` |
| `rho_hat_theta_i` `ρ̂_θi` | `ρ_θi/r_s` | above |
| `ν_ii` | Braginskii ion self-collision `∝ n_i Z⁴ lnΛ /(√m_i T_i^{3/2})` | `n_i`, `T_i` |
| `nu_star` `ν_★` | `ν_ii R_0 q/(ε^{3/2} v_thi)` | above |
| `inv_Ln0` `L̂_n0⁻¹` | `(ψ_s/n) dn/dψ|_s` | profile `n_i(ψ)` |
| `eta_i` `η_i` | `(dln T_i/dψ)/(dln n/dψ)|_s` | profiles |
| `mu0_R` | `μ_0 R_0` (units consistent with `ψ̃`) | g-file |
| `dq_dpsi` | `dq/dψ|_s` | q-spline |
| `w_psi` `ŵ` | the **scan variable** (island half-width); scenario fixes only the *conversion* `w[m] ↔ w_ψ` via `w_ψ = w·R B_θ` | docs/01 §5 |

**Consistency check built in**: `ρ̂_θi`, `ν_★`, `η_i`, `L̂_n0⁻¹` now all come from the
*same* `T_i(ψ)`, `n_i(ψ)`, `B(ψ)` — a self-consistency the hand-set knobs never had.
Sanity target for the a10 case: `ε≈0.1`, `q=2`, `ν_★~O(10⁻²–10⁻¹)` (low-collisionality
banana — verify from `n_i=2e18`, `T_i≈0.5 keV`), `ρ̂_θi~O(10⁻²)`.

## Deliverables (the build this plan authorizes)

1. `Configure.physical_scenario(equil_path; m, n, w_psi) -> Level0Physics` — ingests
   the a10 (or any) equilibrium+profiles via `setup_equilibrium`, finds `ψ_s`,
   derives the vector above. **Reuse** existing GPEC physical-quantity code
   (`KineticForces` collision/thermal helpers, the `sq`/`rzphi` splines) — do not
   reimplement gyroradius/collisionality.
2. `Configure.physical_domain(scenario) -> (Lx, x_match)` — the fixed physical
   matching radius (a fraction of `r_s`; default e.g. `x_match = 0.1–0.2`, or to the
   nearest neighbouring rational), **independent of `w`**. All physics grids use this;
   `resolved_island_grid`/`drift_island_grid` take `Lx` from here, never `Lx∝w` at
   plasma scale.
3. **A regression case + a physical example manifest** (docs/09): the a10-derived
   Level-0 vector recorded, so it is versioned and reproducible (replaces the ad-hoc
   `_phys`/`_b5_phys` literals for *physics* runs).
4. **Un-gate B5** on this scenario: same ε/ν_★/ρ̂_θi/η_i, the physical domain, the T2
   `:original`/`:improved` differential (~×6) and T3 trends — the **York-replication
   gate** the audit flagged as never demonstrated.

## Discipline

Every derived quantity is a **physics formula from cleared conventions** (docs/01 §5)
applied to ingested profiles — not a guessed coefficient. The one item to sign off is
the `ρ_θi = ρ_i q/ε` LAR relation and the `ν_★` Braginskii prefactor against the
`docs/01 §2.3` `ν_★` definition (`[CHECKED: L23 Eq. 2.3.40]`) — a `[DERIVED]` +
physics-verifier check, then human sign-off. Structural/MMS unit tests keep their
abstract boxes (they verify discretization, not physics); only *physics* runs use the
scenario.
