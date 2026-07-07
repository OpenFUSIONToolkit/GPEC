# 01 — Level 0 physics formulation

Scope: the equation set for the benchmark configuration (roadmap O1–O9).

**Tag semantics (updated 2026-07-07).** The full source set now lives in-repo at
`docs/resources/Drift_Kinetic_Island_References/` (see docs/08 for the library
map). Expressions below were transcribed from the PDFs and checked against them
by AI extraction; these carry **[CHECKED: source, Eq./p.]** and still require
one human sign-off before the corresponding [VERIFY] discipline is considered
cleared (CLAUDE.md policy). Anything still uncertain carries **[VERIFY: ...]**
with the specific question stated.

Primary sources: Imada et al. NF 59, 046016 (2019) — **I19** (complete DK-NTM
reference; the 2018 PRL 121, 175001 and JPCS 1125, 012013 are its compact
antecedents); Dudkovskaia PhD dissertation, York 2019 — **Diss19** (full RDK
derivation chain, Appendices C–E); Dudkovskaia et al. PPCF 63, 054001 (2021) —
**D21**; Dudkovskaia et al. NF 63, 016020 (2023) — **D23a** (finite-β, shaped
geometry); NF 63, 126040 (2023) — **D23b** (separatrix layer, polarization,
ω-dependence); Leigh PhD thesis, York Dec 2023 — **L23** (the `kokuchou` code:
amended DK-NTM equations, finite-ν★ thresholds, numerics forensics); Wilson,
Connor, Hastie & Hegna, PoP 3, 248 (1996) — **WCHH96** (analytic electron
closure and large-w limits).

> **Load-bearing warning.** L23 §2.6 (pp. 59–60) documents concrete errors in
> the *published* I19 equation set (Eq. A.1): a missing ρ̂_θi factor on the
> ∂²ĝ/∂p̂² diffusion term (making it ∝ ρ̂²_θi), a missing ν̂_ii ρ̂_θi coefficient
> on ∂ĝ/∂p̂, missing factors on the Maxwellian-gradient drive terms, a corrected
> momentum-conserving term Û_∥i(ĝ + p̂F̂′_Ms), and a sign fix in the Δ_loc
> relation. **Islands must implement from an independently re-derived equation
> set benchmarked against L23's amended form, never from I19 Eq. (A.1) as
> printed.** This is the empirical justification for the whole [VERIFY]
> policy: the literature's O(1) coefficients are demonstrably not to be
> trusted without re-derivation. [CHECKED: L23 §2.6]

---

## 1. Geometry and coordinates

Local region around the rational surface ψ_s (minor radius r_s) of an m/n mode
in a large-aspect-ratio circular tokamak, B₀ = I(ψ)∇φ + ∇φ×∇ψ, I = RB_φ,
B ≈ B₀(1 − ε cos θ), ε = r_s/R₀ ≪ 1 [CHECKED: I19 Eq. (3)].

- Radial coordinate: x ∝ ψ − ψ_s. **Pin one normalization and write the map to
  the others**: I19 uses x = (ψ−ψ_s)/ψ_s with ŵ = w/r_s, ρ̂_θi = ρ_θi/r_s;
  D21/D23b normalize radial quantities to the island width w_ψ
  (ρ̂_θj = I V_Tj/(ω_cj w_ψ)); the 2018 PRL normalizes to ψ_s. These
  inconsistent conventions across the same lineage are a transcription hazard —
  Islands' own normalization (§5) is r_s-based, with conversion factors in one
  place. [CHECKED: I19 p. 6; D21 Eq. 19; PRL Eq. (4) note]
- Helical angle: ξ = m(θ − φ/q_s) (I19 Eq. (6)); Diss19/D21 use ξ = φ − q_s θ
  with the cos nξ harmonic — same island, different angle multiplicity. Pin
  Islands' ξ to the I19 form and document the map. **Island rest frame:** all
  Level-0 solves are steady in this frame.
- Poloidal angle θ is eliminated at leading order by orbit averaging at fixed
  p_φ (O5); it reappears at Level 2.

Perturbation (prescribed at Level 0, O3), single-helicity, constant-ψ:

    A_∥ = −(ψ̃/R) cos ξ,      ψ̃ = (w_ψ²/4)(q_s′/q_s),   q_s′ = dq/dψ|_s
                              [CHECKED: I19 Eq. (5); Diss19 p. 30; L23 Eq. (2.1.4)]

where **w_ψ is the island HALF-width in ψ-space**, w = w_ψ/(RB_θ) the
half-width in minor radius. Note: one extraction of I19 rendered the amplitude
as (w_ψ²/4)(q_s/q_s′); dimensional analysis and Diss19/D21/L23 all give
q_s′/q_s. [VERIFY: check I19 as printed — possible typo in the paper itself.]

Island label and convention (pinned, matches every source in the lineage):

    Ω(x, ξ) = 2(ψ−ψ_s)²/w_ψ² − cos ξ,   Ω = −1 at O-point, Ω = +1 at separatrix
                              [CHECKED: I19 Eq. (7); Diss19 Eq. 2.7; L23 Eq. (2.1.8)]

**All York threshold numbers are HALF-widths** (D23a abstract states
"threshold magnetic island half-width"; L23 footnote p. 130 notes La Haye's
experimental fits quote the *full* width w_marg = 2w_c). This half/full-width
bookkeeping is pinned here and in docs/05.

Even at Level 0, the *stored representation* of the field is A_∥(x, ξ) on the
(x, ξ) grid (decision D1); Ω is computed, never fundamental.

## 2. Kinetic equation

Per species j, phase space (x, ξ, λ, v, σ) with pitch λ = μ/E (grid variable
y = λB_max, trapped–passing boundary y_c = 1), v the speed, σ = sgn(v_∥). Split

    f_j = (1 − e_j Φ/T_j) F_Mj(ψ_s) + g_j,       [CHECKED: I19 Eq. (28) form; Diss19 Eq. 2.15]

with F_M a Maxwellian carrying background gradients at r_s (strictly local,
O2). The steady drift-kinetic equation in the island frame [CHECKED: I19 Eq. (8)]:

    v_∥∇_∥f + v_E·∇f + v_b·∇f − (e_j/m_j v)(v_∥∇_∥Φ + v_b·∇Φ) ∂f/∂v = C_j(f)

with v_E = B×∇Φ/B², v_b = −v_∥ b×∇(v_∥/ω_cj). Orderings: Δ = w/r ≪ 1;
e_jΦ/T_j ~ g_j/F_M ~ Δ; B₁/B₀ ~ εΔ²; collisions O(Δ) below free streaming;
ions retain ρ_θi ~ w (finite orbit width — the key relaxation), electrons have
ρ_θe ≪ w. [CHECKED: I19 §1, §4; Diss19 p. 33]

The radial coordinate is traded for the canonical momentum

    p_φ = (ψ − ψ_s) − I v_∥/ω_cj        [CHECKED: I19 Eq. (2)]

and θ is annihilated by orbit averaging at fixed p_φ (passing: (1/2π)∮dθ;
trapped: (1/2π)Σ_σ σ∫_{−θ_b}^{θ_b}dθ) [CHECKED: I19 Eq. (31); Diss19 Eq. 2.24].
The master 4D equation for the orbit-averaged distribution Ḡ₀(p̂, ξ, y; v̂, σ)
is **I19 Eq. (32)** (structure confirmed; coefficients subject to the L23 §2.6
amendments — implement from re-derivation):

    −m[ (p̂/L̂_q)Θ(y_c−y) + ρ̂_θi ω̂_D − (ρ̂_θi/2)⟨(1/v̂_∥)∂Φ̂/∂x⟩_θ ] ∂Ḡ₀/∂ξ|_p̂
    + m[ (ŵ²/4L̂_q) sin ξ Θ(y_c−y) − (ρ̂_θi/2)⟨(1/v̂_∥)∂Φ̂/∂ξ⟩_θ ] ∂Ḡ₀/∂p̂
    = ⟨(1/v̂_∥)Ĉ_ii(Ḡ₀)⟩_θ

The three transport channels of the design (island-induced streaming, magnetic
drift, E×B) are the three bracketed frequencies above; they map one-to-one onto
the operator stack (docs/03 §2).

### 2.1 Magnetic drift frequency: the original/improved toggle (now precise)

Orbit-averaged precession [CHECKED: I19 Eq. (32) def.; D21 Eqs. 15, B1]:

    ω̂_D = [σv̂/(1+ε)] [ (1/L̂_q)⟨√(1−yb)/b⟩_θ − (1/2)(1/L̂_B)⟨(2−yb)/(b√(1−yb))⟩_θ ]

- **:original** (I19/DK-NTM): finite constant L̂_B⁻¹ = (ψ_s/B)∂B/∂ψ — retains a
  non-vanishing ∇B term after orbit averaging.
- **:improved** (D21/RDK-NTM): Appendix A of D21 shows
  ∂B/∂ψ = −(B_φ/(R₀²B_θ)) cos θ + O(ε²) — the cos θ modulation makes the term
  ε-small after orbit averaging; **L̂_B⁻¹ = 0 is the documented proxy** (D21
  footnote 10, Fig. 8 compares proxy vs full cos θ form directly).
  [CHECKED: D21 Eq. A2, p. 16]

This single toggle is what moved the threshold half-width 8.73 ρ_bi → 1.46 ρ_bi
(D23a abstract). It is the archetype of the toggle-impact studies (docs/05 E1).

### 2.2 The drift-island structure (must emerge from the solve, not be assumed)

The exact drift-surface label [CHECKED: I19 Eq. (33); D21 Eq. 21; Diss19 Eq. 2.37]:

    S = (ŵ²/4L̂_q)[ 2(p̂ − ρ̂_θi ω̂_D L̂_q)²/ŵ² − cos ξ ] Θ(y_c−y)
        − p̂ ρ̂_θi ω̂_D Θ(y−y_c) − (1/2)⟨(ρ̂_θi/v̂_∥) Φ̂⟩_θ

Passing particles: constant-S surfaces are the magnetic island **radially
shifted by x_D = ρ̂_θi ω̂_D(y, v̂; σ) L̂_q** — σ-dependent, equal and opposite
for v_∥ ≷ 0, pitch/energy-dependent through ω̂_D. (There is no separate
"h(λ,E)" shift function in the sources; the shift *is* ρ̂_θi ω̂_D L̂_q. The
symbol h(Ω) is reserved for the electron profile function, §2.4.) Flattening
of f on drift islands rather than the magnetic island sustains pressure
gradients across small islands (w ~ ρ_θi) and weakens the bootstrap drive —
the kinetic threshold mechanism, carried by **passing** particles (D21 §7:
passing-particle physics, not banana-orbit physics; ρ_bi is merely the natural
unit at ε = 0.1). Trapped particles: S ∝ p̂ (no island structure); response
tied to the magnetic island.

In DK (4D direct) mode Islands does **not** impose S-structure; it must *emerge*.
The RDK reduction — solve the 1D collisional constraint ⟨Ĉ/𝒜⟩_ξ^S g^(0,0) = 0
per S-contour [CHECKED: D21 Eqs. 23–24; explicit coefficient forms Diss19
Eqs. D.60–D.62 and D23b Eq. 19 + Appendix A] — is retained as a cross-check
mode valid for δ_j = ν_j/(εω_b) ≪ 1.

### 2.3 Collision operator (Level 0)

Momentum-conserving pitch-angle (Lorentz) model [CHECKED: I19 Eqs. (9)–(12);
Diss19 Eqs. 2.25–2.30; lineage: WCHH96 Eq. (62)]:

    C_jj(f) = 2ν_jj(v)[ (√(1−λB)/B) ∂_λ( λ√(1−λB) ∂_λ f ) + v_∥ ū_∥j f /v²_thj · F_Mj-normalized ]
    ū_∥j(f) = (1/(n⟨ν_jj⟩_v)) ∫d³v ν_jj v_∥ f          (momentum restoring)
    C_ei drags on the ION flow u_∥i (species coupling)

λ-derivatives at **fixed ψ**, not fixed p_φ (a classic transcription trap).
Energy dependence: two variants exist in the lineage and become a documented
sub-toggle — I19/L23 use the full ν_jj(v) = ν̃_jj[φ(v̂) − G(v̂)]/v̂³ (Chandrasekhar
G; needed for neoclassical fidelity), while Diss19/D21 use the simpler
ν(V) ∝ V⁻³. L23 additionally derives the analytic velocity average
⟨ν̂_ii⟩_u = (4ε^{3/2}ν_★/3√π)(√2 − ln(1+√2)) and uses it in place of an
inaccurate numerical u-integral (ν̃ ∝ u⁻² divergence at low u) — adopt this.
[CHECKED: L23 Eq. 4.1.6, p. 88]

Collisionality normalization: ν_★ = ν_jj Rq/(ε^{3/2} v_th) (banana regime
ν_★ ≪ 1); ν̂_jj = ε^{3/2}ν_★ ν̃_jj(u). [CHECKED: L23 Eq. (2.3.40); Diss19
footnote 26]

Replaced wholesale at Level 1 by the multi-species Fokker–Planck operator.

### 2.4 Electrons at Level 0 (O7) — closure now exact

ρ_θe ≪ w ⇒ electron drift islands coincide with the magnetic island. The
analytic closure is WCHH96's, as used by I19/L23 [CHECKED: I19 Eqs. (14)–(22);
L23 §2.4]:

    f_e = (1 − e_eΦ/T_e) F_Mes + h(Ω) F′_Mes − (Iv_∥/ω_ce) F′_Mes ∂h/∂ψ + h̄_e
    h(Ω) = Θ(Ω−1) (w_ψ/2√2) ∫₁^Ω dΩ′/Q(Ω′),   Q(Ω) = (1/2π)∮√(Ω+cos ξ) dξ

h(Ω) is exactly flat inside the separatrix, → x far away, and satisfies
⟨∂²h/∂x²⟩_Ω = 0 (unit-test target; L23 Eq. 4.1.1). Flux-surface-averaged
electron flow [CHECKED: I19 Eq. (22); L23 Eqs. 2.5.5–2.5.8]:

    ⟨⟨Bu_∥e⟩_θ⟩_Ω/(B₀v_the) = −[f_t/(1+f_t)](Iv_the/ω_ce)(n′/n)(1 + η_e + ½ k f_c η_e)⟨∂h/∂ψ⟩_Ω
                              + [f_c/(1+f_t)] ⟨⟨Bu_∥i⟩_θ⟩_Ω/(B₀v_thi)

with k ≃ −1.173 (Hirshman–Sigmar; unit-test: L23 reproduces −1.1730) and
f_p ≃ 1 − 1.46√ε. Note the electron current depends on the *numerically
computed ion flow* (momentum conservation) — the closure is coupled, not
one-way. Toggle `electrons = :flattened | :kinetic`: the `:kinetic` option is
exactly RDK-NTM's defining feature (electrons solved with the same drift-island
machinery as ions, Diss19 Eq. D.61 / D21 §5) and is *required* at Level 3
(shielding); running it at Level 0 against `:flattened` is toggle study E4.

## 3. Field equation (Level 0: quasineutrality only)

    n_i[Φ; g_i] = n_e[Φ; closure]   →   Φ(x, ξ)

Exact Level-0 closed form with flattened electrons [CHECKED: I19 Eq. (A.11);
L23 Eq. (2.4.14)]:

    e_iΦ̂/T_i = [ δn̄_i/n₀ + x − ĥ(Ω) ] / (2 L̂_{n0})

(T_e = T_i assumed in the sources; Islands keeps τ = T_e/T_i general and flags
departures). With kinetic electrons, the Picard form δΦ̂ = (δn̂_i − δn̂_e)/2
[CHECKED: Diss19 Eq. 2.45]. In Islands both reduce to one quasineutrality
residual inside the global Newton system (docs/03) — the sources' nested
Picard loops (Φ outer, ū_∥i inner; I19 fig. A1) are precisely the fragile
iteration structure Newton–Krylov replaces; L23 §6.1.1 reports the Picard
convergence criterion was *never met* in production (Φ̂ array-max residuals
> 100%/iteration at large ŵ) even as Δ stabilized — treat that as the
cautionary tale motivating D2.

Boundary conditions: g → neoclassical (no-island) solution and Φ̂ → background
E_r potential as |x| → L_x; periodic in ξ. **Do not use bare Neumann
∂ĝ/∂p̂ = 0**: L23 §5.3/§7.1 traces its non-physical "winged" solution branch
(flows extending 8–10 island widths, disagreeing with neoclassical theory) to
the Neumann condition admitting multiple numerically-valid solutions, and
recommends matching to the analytic far-field limit — which is exactly Islands'
neoclassical-matching BC. [CHECKED: L23 pp. 113–115, 141]

Ampère is **not** solved at Level 0 (O3). The Ampère residual is evaluated as
a diagnostic from day one; its resonant moments are the Δ outputs:

## 4. Output moments and MRE assembly (normalization now exact)

Parallel current J̄_∥ = θ-average of Σ_j e_j n_j u_∥j. The two projections of
parallel Ampère through the island [CHECKED: Diss19 Eqs. 2.9–2.10; D21
Eqs. 7–8, 32]:

    (1/μ₀R) Δ′ ψ̃ = ∫_ℝ dψ ∮ dξ J̄_∥ cos ξ        (growth: matching to Δ′)
    0            = ∫_ℝ dψ ∮ dξ J̄_∥ sin ξ        (torque balance / rotation)

so the kinetic drive and torque moments are

    Δ_cos ≡ Δ_neo = −(μ₀R/2ψ̃) ∫ dψ ∮ dξ J̄_∥ cos ξ,     stationarity: Δ′ + Δ_neo = 0
    Δ_sin          =  (μ₀R/2ψ̃) ∫ dψ ∮ dξ J̄_∥ sin ξ     [CHECKED: Diss19 Eq. 4.12 for Δ_neo;
                                                          sin-moment normalization chosen symmetric — [DERIVED] pin at implementation]

with ψ̃ = (w_ψ²/4)(q_s′/q_s), and the Rutherford LHS (2τ_R/r_s²) dw/dt (w =
half-width). Decomposition diagnostics [CHECKED: Diss19 Eqs. 4.13–4.15; D21
Eqs. 33–34; D23b §4]:

- **Bootstrap+curvature part**: the Ω-flux-surface-constant part of J̄_∥,
  ⟨J̄_∥⟩_Ω with ⟨·⟩_Ω = ∮·(Ω+cosξ)^{−1/2}dξ / ∮(Ω+cosξ)^{−1/2}dξ.
- **Polarization part**: Δ_pol = Δ_neo − (Δ_bs+Δ_cur) — the piece that
  flux-surface-averages to zero. (L23 Eq. 2.5.3 flags this split as
  approximate bookkeeping — "could comprise similar contributions from other
  sources" — which is the design's position: partition is diagnostic, the
  solve never separates channels.)
- Species partition (ion vs electron) alongside: L23 finds the *electron*
  channel dominates both Δ_bs and (unexpectedly, at ω_E = 0) the stabilizing
  Δ_pol — an open physics question Islands can settle with the ω_E scan.

**Analytic large-w limits to recover** (ladder B2): Δ_bs+Δ_cur ∝ 1/w matching
WCHH96 Eq. (85) — with the caveat that Eq. (85) is derived in the E_r = 0
frame while the island-frame calculation must be mapped before comparison
(Diss19 p. 86) — generic scaling Δ_bs ~ ε^{1/2}(L_q/L_p)(β_θ/w); and
Δ_pol ∝ 1/w³ at large w. [CHECKED: Diss19 pp. 84–86; D21 p. 2]

In the linear limit, (Δ_cos + iΔ_sin) ↔ the complex layer Δ(Q) (SLAYER
convention map still [VERIFY: Park PoP 29 (2022) — paper not yet in the
reference library; acquire]).

## 5. Nondimensionalization and frames (the input parameter vector p)

Normalizations (r_s-based, following I19): x = (ψ−ψ_s)/ψ_s, ŵ = w/r_s,
ρ̂_θj = ρ_θj/r_s, v̂ = v/v_thj, y = λB_max, b = B/B_max, L̂_q⁻¹ = (ψ_s/q)dq/dψ,
L̂_n⁻¹ = (ψ_s/n)dn/dψ, Φ̂ = e_jΦ/T_j, ν̂ = ε^{3/2}ν_★ν̃(v̂). Conversion maps to
the D21 (w_ψ-based) and PRL (ψ_s-based) conventions live in `src/frames/`
alongside the frequency maps. [CHECKED: I19 p. 6; L23 Eqs. 2.3.40–2.3.46]

**Frame identities (now source-confirmed, the frames-module spec):**

- ω_dia,e = m T_e n₀′/(−e q_s n₀); ω_E ≡ m Φ′_eqm/q_s; ω̂_E = ω_E/ω_dia,e.
  [CHECKED: Diss19 p. 46]
- The combination **ω − ω_E is frame-independent**; with ω₀ the island
  propagation frequency in the frame where E_r → 0 far from the island,
  **ω₀ = −ω_E** (island-rest-frame calculation at equilibrium-potential
  gradient ω_E ⇔ island rotating at −ω_E in the zero-E_r frame).
  [CHECKED: Diss19 pp. 47–48]
- The effective density gradient shifts with frame:
  L_n⁻¹ = L_{n0}⁻¹(1 + Z_j ω_E/ω_dia,e). [CHECKED: Diss19 p. 46]
- Level-0 sources' published thresholds are at ω_E = 0 (no equilibrium E_r);
  D23b treats ω_E as an input parameter — exactly Islands' O4. Torque-balance
  roots (Δ_sin = 0) exist at discrete ω̂_E (Diss19 benchmark: ω₀ = −0.93
  ω_dia,e selected among ±0.93, ±1.28); Δ_pol ∝ ω_E² away from zero and
  **reverses sign at ω_E ≈ −0.89 ω_dia,e** (D23b Fig. 8) — the modern, frame-
  pinned statement of the polarization sign controversy. These are ladder-B4
  targets.

Level-0 input vector:

    p = ( ŵ = w/ρ_θi  (half-width),
          ω̂_E = ω_E/ω_dia,e   (≡ −ω₀/ω_dia,e; SLAYER Q-map [VERIFY: Park 2022]),
          ν̂_j = ν_★j  per species,
          ε, ŝ (via L̂_q), q_s, τ = T_e/T_i,
          η_j = L_n/L_Tj,
          species list: {Z_j, m_j/m_i, n_j/n_e, T_j/T_i, gradients, F0 type, role} )

Frequency bookkeeping owns its own unit tests; the polarization-sign disputes
in the literature are largely frame disputes, and the identities above make
the conversions mechanical. One module (`src/frames/`) owns them.

## 6. Symmetries and conserved checks (unit-test targets)

- Parity: Δ_cos even / Δ_sin odd under the appropriate (ξ, σ, ω_E) reflection
  [derive at implementation and record as [DERIVED]; consistency targets:
  Δ_pol(ω_E) parabolic/even to leading order away from the linear-in-ω_E
  region near zero, D23b Fig. 8].
- Zero-gradient, zero-Φ̃ Maxwellian: g = 0 exactly; residual = machine zero.
- No island (ψ̃ → 0), gradients on: recover standard local neoclassics
  (bootstrap J_∥ vs. Sauter/NEO) — the most powerful global check (docs/05 B1).
- Electron-closure identities: ⟨∂²h/∂x²⟩_Ω = 0; k → −1.173; f_p → 1 − 1.46√ε;
  ⟨ν̂_ii⟩_u analytic value (§2.3). [CHECKED: L23 Ch. 4]
- Collision operator: particle conservation (L0); +momentum/energy per pair
  (L1); discrete entropy sign ∫ g C[g]/F_M ≤ 0.

## 7. Explicitly out of Level-0 scope (recorded to prevent creep)

Ampère & multi-harmonic (L3); torque-balance closure of ω_E (L4 — but ω_E is
an input *parameter scan* from day one; publishing single-ω_E Δ values is
forbidden per the risk register); χ_⊥/w_d (L4); general geometry & 5D (L2);
slowing-down F₀ (L2); radiation (L4); gyroaveraging beyond drift order (out of
program scope — documented limitation vs. gyrokinetic island studies; L23
p. 142 draws the same boundary: ŵ approaching ρ_i needs gyrokinetics).
