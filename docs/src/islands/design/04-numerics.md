# 04 — Numerics

The numerics posture is now informed by three generations of prior-art
implementation experience: DK-NTM (I19 Appendix A — shooting method, nested
Picard loops), RDK-NTM (Diss19 Appendix E — S-space reduction, analytic
layers), and kokuchou (L23 Chs. 3–6 — the most complete forensic record of
where the direct 4D approach actually breaks). Citations per docs/01 header.

## 1. Discretization by coordinate

- **ξ (helical angle)**: Fourier pseudo-spectral, periodic. Nonlinear terms
  (v_E·∇g, Φ̃ coupling) via dealiased transforms. At L0 the field has one
  harmonic but g does not — keep the full spectrum from the start. Harmonic
  content of the residual doubles as a resolution diagnostic. (kokuchou used
  2nd-order FD with n_ξ = 30 and flagged accuracy limits; spectral ξ is a
  deliberate upgrade.)
- **x (radial)**: mapped collocation or high-order FD on a stretched grid.
  **Packing must target the drift-island separatrices, not the magnetic
  island**: the layer follows contours shifted by ±ρ̂_θi ω̂_D L̂_q per (y, v̂, σ)
  (docs/01 §2.2), so the packed envelope is max over the σ-shifts of the Ω=1
  curve — L23 §3.1.6 identifies the rectilinear-mesh/shifted-round-island
  mismatch as kokuchou's dominant accuracy limiter, and proposes a mapped
  radial coordinate absorbing the θ-dependent orbit shift (its Eq. 7.1.1,
  p̃ = ψ − I(v_∥(θ)/ω_ci(θ) − v_∥(0)/ω_ci(0))) — adopt as a candidate *grid
  map* x(s; y, v̂, σ-envelope), never as a solve-coordinate change (D1 stands).
  Far-field BCs at |x| = L_x: g → neoclassical (no-island) solution, Φ̃ →
  background E_r potential. **Never bare Neumann ∂g/∂x = 0**: L23 traced its
  spurious "winged" solution branch to Neumann non-uniqueness (docs/01 §3).
  L_x is a convergence parameter reported with every result (and is the Δ′
  excision radius, docs/03 §5).
- **λ (pitch)**: collocation with clustering at the trapped-passing boundary
  y_c = 1. Layer width **in pitch angle: ε_λ ≈ [(2ν̂_j/v̂) a(λ_c, v̂; σ)]^{1/2}
  ~ √ν_★** with a(λ) = ⟨σλ(1−λB)^{1/2}R/B_φ⟩_θ — now a confirmed scaling
  [CHECKED: Diss19 p. 58; D23b §3.1], electron layer wider by (m_i/m_e)^{1/4}.
  Packing parameter set adaptively from input ν̂. Internal boundary
  conditions at y_c (the DK-NTM/kokuchou matching set):
  Σ_σ σg^p = 0, Σ_σ g^p = 2g^t, Σ_σ ∂_y g^p = 2 ∂_y g^t
  [CHECKED: I19 Eqs. A.7–A.9; L23 Eqs. 2.3.52–2.3.54]. σ = ± sheets joined at
  bounce points for trapped particles.
- **E (energy)**: Gauss-type quadrature nodes on a mapped semi-infinite domain
  weighted by F₀ — Maxwellian weights at L0; slowing-down weights (L2) change
  the map, not the machinery. Mind the integrable divergences the sources hit:
  ν̃(v̂) ∝ v̂⁻² at low v̂ (use the analytic ⟨ν̂⟩_v, docs/01 §2.3), (1−yb)^{−1/2}
  at bounce points, and the y → 1/b flow integrals — analytic correction
  terms/asymptotics, not naive quadrature [CHECKED: L23 pp. 69, 87–88, App. 8.4].
  Collision-operator energy diffusion (L1) prefers modest-order collocation
  over pure spectral here; decide with a convergence study on the Sauter
  benchmark.
- **θ (poloidal, L2)**: Fourier. The 4D orbit-averaged mode must be recoverable
  as the θ-average of the 5D solve — a built-in cross-check of both.

## 2. The two internal layers (the numerical cliff — now quantified)

Both layers are collisional with **width ∝ ν^{1/2}** [CHECKED: D23b §3.1,
footnote 11]:

- **Trapped-passing dissipation layer** at y_c: ε_λ ~ [(2ν̂/v̂)a(λ_c)]^{1/2}.
- **Drift-island separatrix layer** at S = S_c: ε_S ~ [(2ν̂/v̂)C_SS(S_c)]^{1/2}
  in the S variable; in kokuchou's (p, ξ) variables the equivalent widths are
  Δ_p,pass ~ √(ν̂L̂_q/ŵ)·ρ̂_θi and Δ_p,trap ~ √(ν̂L̂_q/ŵ)·√(ŵρ̂_θi)
  [CHECKED: L23 §6.1.2].

Two prior-art failure modes to design against:

1. **The layer moves during the solve.** When E×B dominates the layer balance
   (ν̂/(ŵL̂_q) < 1 — satisfied over part of the velocity grid in *every* L23
   production run), the width becomes Δ_p,E×B ~ ν̂ρ̂_θi/ŵ and **depends on Φ̂,
   so it shifts between nonlinear iterations** — a mesh packed for the initial
   iterate can be unresolved at the converged one [CHECKED: L23 Eqs. 6.1.1–6.1.2].
   Mitigation: pack from the layer-width *lower bound* over the expected Φ̂
   range; validate the estimate against measured solution structure post-solve
   (estimates logged); re-mesh-and-continue as a fallback.
2. **The accessible parameter window shrinks with correct physics.** L23's
   corrected ∂²ĝ/∂p̂² coefficient (∝ ρ̂²_θi, per the §2.6 amendments) makes
   separatrix gradients *steeper* than in the original DK-NTM runs — kokuchou
   could not reach DK-NTM's ν_★ = 10⁻³ operating point (floor at 5×10⁻³) nor
   ŵ > 0.75ρ̂_θi at that floor, with memory as the binding constraint
   [CHECKED: L23 §5.3, §6.1.2, p. 116]. Islands' matrix-free posture removes
   kokuchou's specific memory wall (§6 below) but not the resolution demand.

Posture (unchanged in spirit, now with numbers): adaptive packing driven by
the ε_λ, ε_S, Δ_p,E×B estimates from input parameters; a documented ν̂ floor
per resolution tier; below the floor, results flagged `layer_unresolved` and
the RDK cross-check mode (analytic layer treatment per Diss19 §3/D23b §3.1.1 —
Fourier-matched layer solutions) is the reference. Disagreement between DK and
RDK modes above the floor is a release-blocking bug; below it, it's the
measured cost of the RDK ordering — a publishable toggle-impact result.
Grid-convergence studies are first-class benchmark artifacts.

## 3. The trapped-passing boundary is *singular* — regularize deliberately

The hardest-won lesson in the reference set [CHECKED: L23 §4.2, pp. 94–97]:
the linear system coupling the two sides of y_c through the matching
conditions is **intrinsically singular/ill-conditioned** (kokuchou measured
rcond ≈ 10⁻¹⁶–10⁻¹⁹, det = 0 or ±Inf, at every energy grid point; plain LU
gave machine-dependent noise in ĝ(v̂); the same latent defect was reproduced in
DK-NTM once other bugs were fixed). kokuchou's fix: truncated SVD (cutoff
10⁻⁷; exactly one singular value truncated) applied only at the boundary
solve.

Implications for Islands' Newton–Krylov (no y-sweep, one global residual):

- The same near-null-space will reappear as **Jacobian ill-conditioning
  localized at the y_c block**. The physics-block preconditioner must treat
  the y_c matching rows explicitly (small dense block per (x-locality, v̂):
  factor with SVD/complete pivoting, truncate/regularize below a documented
  cutoff), so GMRES never has to resolve the near-singular directions itself.
- Add a CI-level diagnostic: monitor the smallest singular value of the y_c
  matching block and the GMRES convergence stagnation signature; a silent
  regression here produced noise, not crashes, in the prior art — it must be
  *tested for*, not observed.
- Root cause is the asymptotic v_∥⁻¹ structure at the boundary; any basis
  change that removes the 1/v_∥ divergence from the matched unknowns (e.g.
  solving for flux-like variables across y_c) is worth a design spike in M1.

## 4. Manufactured solutions

Before any physics benchmark: MMS per operator and for the assembled L0 system
(source terms chosen so a prescribed smooth g*, Φ̃* solve the equations).
Verifies discretization order and the AD-generated JVPs simultaneously
(JVP checked against finite differences of the residual). MMS configs live in
`src/Islands/verify/` and run in CI at low resolution. Supplement with the sources' cheap
analytic unit targets (docs/01 §6: h(Ω) identities, k = −1.173, ⟨ν̂⟩_v,
f_p = 1 − 1.46√ε) — L23 Ch. 4 demonstrates these catch inherited bugs that
integration tests miss.

## 5. Newton–Krylov details

- Inexact Newton (Eisenstat–Walker forcing), line search + continuation
  globalization (docs/03 §3). Expect and handle fold points (penetration
  bifurcation at L3): pseudo-arclength with tangent monitoring from day one.
- The prior art's nested Picard loops (Φ outer / ū_∥i inner) are the explicit
  anti-pattern: kokuchou's production runs *never met* their Picard convergence
  criterion (Φ̂ array-max residual >100%/iteration at large ŵ) even as Δ_loc
  stabilized, and the array-averaged residual hid locally-divergent regions
  [CHECKED: L23 §3.1.5, §6.1.1]. Islands solves (g_j, Φ̃) as one Newton system;
  convergence is measured by the global residual norm *and* its spatial max,
  both archived.
- **Preconditioning** (the make-or-break): physics-block preconditioner —
  approximate inverse built from the ξ-averaged, drift-free operator
  (streaming + collisions per species: block-tridiagonal-ish in x per (λ, E)),
  a Schur-type block for Φ̃ (and A_∥ at L3), plus the explicit y_c matching
  block of §3. Cheap to factor, captures the stiff parallel/collisional
  physics; Krylov handles drifts and nonlinear coupling. Iteration counts vs.
  p logged; preconditioner quality is a tracked metric, not folklore.
- Krylov: GMRES (restarted) via Krylov.jl or LinearSolve.jl; matrix-free JVP
  through the operator stack (ForwardDiff duals). No global sparse Jacobian is
  ever formed except in tiny-grid debug mode (also useful for eigenvalue
  diagnostics near folds and for the y_c singular-value monitor).

## 6. Cost model and why matrix-free is mandatory (prior-art data point)

kokuchou's y-sweep shooting method stores dense (n_ξn_p)² recursion blocks:
at n_ξ = 30, n_p = 145 that is ≈16.6 GB *per energy grid point* and
O(0.67 N³) ≈ 5.5×10¹⁰ flops per y-point (0.4 hr/energy-point on ARCHER2,
O(100 GB) RAM total) — and memory, not physics, set its ν_★ floor and ŵ
ceiling [CHECKED: L23 pp. 80–84]. Islands' matrix-free Newton–Krylov stores
O(#dof) vectors instead; the trade is preconditioner engineering (§5). Every
solve logs a cost-model entry (dof, iterations, wall time) — the emulator
strategy depends on knowing what a point costs, and the L23 numbers are the
baseline to beat.

## 7. Trace-species linear pass

One GMRES solve per trace species with frozen bulk fields; same preconditioner
with the trace species' collision blocks. Sensitivities of Δ w.r.t. trace
parameters (n_W, Z̄_W, alpha source) are nearly free here (linear problem +
AD) — the W/EP parameter scans (docs/02) should exploit this before any bulk
rescans.

## 8. Surface generation & emulator posture (M12)

- Continuation walks generate curves; a scheduler tiles (ŵ, ω̂_E) planes per
  remaining-parameter grid point, warm-starting from nearest neighbors. (L23's
  practical trick — warm-starting Φ̂ from a stable neighboring run to avoid the
  spurious branch — is the same mechanism; continuation makes it systematic.)
- Store everything (docs/03 §4); train emulator (GP or small NN — decide later)
  on (p → Δ_cos, Δ_sin) with AD sensitivities as extra supervision.
- Publish surfaces with the named configuration and grid tier; emulator
  uncertainty must exceed measured grid-convergence error or the tier is bumped.

## 9. Language/tooling specifics

Julia ≥ LTS current at project start. Key packages: Krylov.jl / LinearSolve.jl,
ForwardDiff.jl (+ Enzyme.jl evaluation), FFTW.jl, Interpolations.jl (equilibrium
ingest; mind boundary conditions), HDF5.jl/JLD2.jl, TOML stdlib. Threading with
per-thread preallocated caches (no allocation in `apply!` hot paths — enforce
with an allocation regression test). Revise.jl workflow assumed; keep
world-age-safe (no runtime `eval` in the stack). All hot kernels `@inbounds`-safe
with explicit bounds-check test coverage first.
