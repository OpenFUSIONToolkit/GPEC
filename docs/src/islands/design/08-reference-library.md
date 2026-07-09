# 08 — Reference library

The primary sources live in-repo at
`docs/resources/Drift_Kinetic_Island_References/`. This file maps each PDF to
its role in the project, its abbreviation used across docs/00–05, and the
load-bearing content a reader (human or agent) should pull from it. Equation
transcriptions from these sources into docs/01 carry [CHECKED]/[VERIFY] tags
per the docs/01 header semantics.

## The DK-NTM / RDK-NTM / kokuchou lineage (core Level-0/1 sources)

| Abbrev. | File | What it is | Load-bearing content |
|---|---|---|---|
| **PRL18** | `2018-Imada-Nonlinear_Kinetic_Ion_Response_to_Small_Scale_Magnetic_Islands_in_Tokamak_Plasmas.pdf` | Imada et al., PRL 121, 175001 (2018). First announcement of DK-NTM | Compact statement of the drift-island result (w_c ≃ 2.7 ρ_θi); **caution: uses ψ_s-based normalizations, different from I19's r_s-based ones** — see docs/01 §1 |
| **JPCS18** | `2018-Imada-Drift_kinetic_response_of_ions_to_magnetic_island_perturbation_and_effects_on_NTM_threshold.pdf` | Imada et al., Varenna proceedings (2018) | Condensed DK-NTM derivation; explicit electron-flow and h(Ω) formulas; renames Δ′_bs → Δ′_loc; MRE context incl. Δ_pol ∝ 1/w³ discussion |
| **I19** | `2019-Imada-Finite_ion_orbit_width_effect_on_the_neoclassical_tearing_mode_threshold_in_a_tokamak_plasma.pdf` | Imada et al., NF 59, 046016 (2019). The complete DK-NTM reference paper | Full equation hierarchy (Eqs. 23–34); master 4D equation Eq. (32); S-function Eq. (33); collision operator Eqs. (9)–(12); electron closure §3 (Eqs. 14–22); numerics appendix (shooting method, y_c matching Eqs. A.7–A.10, Picard loops, Eq. A.11 quasineutrality); w_c ≃ 2.76 ρ_θi (Fig. 9). **Known errata: see L23 §2.6 amendment list** (docs/01 header warning) |
| **Diss19** | `2019-Dudkovskaia-Modelling_NTMs_in_tokamak_plasmas_PhD_dissertation.pdf` | Dudkovskaia PhD dissertation, York 2019 | The full RDK derivation chain: island geometry & Ω convention (Ch. 2), S-coordinate reduction, trapped-passing layer analytics (Ch. 3, width √(ν/εω) in λ), Δ_neo normalization (Eq. 4.12) and bootstrap/polarization split (Eqs. 4.13–4.15), frame identities ω₀ = −ω_E (pp. 47–48), torque-balance roots (Fig. 4.18), **complete solver coefficient sets in Appendices C–E (Eqs. D.60–D.62)** — the RDK cross-check mode's spec |
| **D21** | `2021-Dudkovskaia-Drift_kinetic_theory_of_neoclassical_tearing_modes_in_a_low_collisionality_tokamak_plasma_magnetic_island_threshold_physics.pdf` | Dudkovskaia et al., PPCF 63, 054001 (2021). RDK-NTM v.1 | The improved magnetic-drift model (App. A, Eq. A2: cos θ structure of ∂B/∂ψ; L̂_B⁻¹ = 0 proxy, footnote 10) → w_c ≈ 0.45 ρ_θi ≡ 1.41–1.47 ρ_bi half-width; DK-NTM benchmark in App. C (agreement window ν_★ ~ 10⁻³–10⁻⁴); threshold-mechanism statement (§7: passing-particle physics) |
| **D23a** | `2023-Dudkovskaia-Drift_kinetic_theory_of_the_NTM_magnetic_islands_in_a_finite_beta_general_geometry_tokamak_plasma.pdf` | NF 63, 016020 (2023). RDK-NTM v.2: finite β, shaped (Miller) geometry | Authoritative "8.73 → 1.46 ρ_bi half-width" statement (abstract); finite-β drift terms (Eqs. 28–31); Miller parametrization (Eq. 33); shaping results: triangularity 2w_c = 1.82 ρ_bi (δ=+0.42) → 2.90 ρ_bi (δ=−0.5); ε ≈ 0.3 crossover of the w_c scaling; β_θ trend vs. EAST 91972 (ladder C4) |
| **D23b** | `2023-Dudkovskaia-Drift_kinetic_theory_of_neoclassical_tearing_modes_in_tokamak_plasmas_polarisation_current_and_its_effect_on_magnetic_island_threshold_physics.pdf` | NF 63, 126040 (2023). RDK-NTM v.3: separatrix layer + polarization + ω dependence | **Table 1** = the code-family comparison (DK-NTM vs RDK v.1/v.2/v.3); both layer widths ∝ ν^{1/2} (§3.1, footnote 11); ω_E as input parameter; Δ_pol(ω_E) sign reversal at ≈ −0.89 ω_dia,e (Fig. 8); layer effect on threshold 0.78 → 0.52 ρ_θi (Fig. 9); the B6 figure set (Figs. 3, 4, 6, 7, 9, 11, 13) |
| **L23** | `2023-Leigh-Drift_kinetic_simulations_of_Neoclassical_Tearing_Mode_instabilities_in_finite_collisionality_tokamak_plasmas.pdf` | Leigh PhD thesis, York Dec 2023. The `kokuchou` code (DK-NTM successor, finite ν_★) | **§2.6 amendment list against I19 Eq. (A.1)** (the [VERIFY] policy's empirical justification); WCHH96 electron closure spelled out (§2.4–2.5, k = −1.173, f_p = 1−1.46√ε); TSVD treatment of the singular y_c matching matrix (§4.2); separatrix-layer width scalings incl. the iteration-dependent E×B regime (§6.1.2); Picard non-convergence forensics (§6.1.1); spurious "winged" Neumann branch (§5.3, §7.1); memory/cost data for the dense shooting method (pp. 80–84); **w_c ≈ 0.440 ρ̂_θi + 0.0178 ν_★ − 7.54×10⁻⁵** (Eq. 6.3.2, ν_★ ∈ [0.005, 0.020]); future-work list §7.1 (mapped p̃ coordinate, analytic far-field BC) |

## Background / adjacent

| Abbrev. | File | What it is | Role |
|---|---|---|---|
| **JOP18** | `2018-Dudkovskaia-Island_Stability_in_Phase_Space.pdf` | Dudkovskaia, Garbet, Lesur, Wilson — JPCS 1125, 012009 (2018) | **Not about magnetic islands.** Bump-on-tail *phase-space* island stability (Vlasov–Fokker-Planck–Poisson secondary modes). Relevant only as (a) the methodological antecedent of the RDK bounce/angle-variable and separatrix-layer machinery, (b) EP-physics background for the Level-2 precession-resonance study (ladder C6). Do not cite it as an NTM threshold source |

## Referenced but not yet in the library (acquire)

- H. R. Wilson, J. W. Connor, R. J. Hastie & C. C. Hegna, "Threshold for
  neoclassical magnetic islands in a low collision frequency tokamak",
  Phys. Plasmas **3**, 248 (1996), doi:10.1063/1.871830 — **WCHH96**. The
  analytic electron closure (its Eq. 55/74 lineage) and the large-w limit
  target (its Eq. 85) are load-bearing for docs/01 §2.4 and ladder B2;
  currently cited via I19/L23/Diss19 transcriptions only.
- ~~Park, Phys. Plasmas 29 (2022) — SLAYER. Needed for the D1 Q-convention
  map.~~ **Found in-repo** (2026-07-09): it lives in the general GPEC library,
  `docs/resources/2022-Park-Parametric dependencies of resonant layer responses
  across linear, two-fluid, drift-MHD regimes.pdf` — outside this island
  subfolder, which is why this map missed it. The D1 Q-convention `[VERIFY]`
  can be worked from that file.
- La Haye et al. (2012 NSTX/DIII-D scaling; 2006) — experimental threshold fits
  behind ladder B9.
- Sauter et al., PoP 6, 2834 (1999); Glasser–Greene–Johnson 1975; Fitzpatrick
  1993/1995/1998; Cole & Fitzpatrick 2006; Rutherford 1973; Waelbroeck &
  Fitzpatrick 1997; Smolyakov — the classical MRE-term and penetration
  literature (ladder B1–B4, D4–D5 targets).

## Known cross-source inconsistencies (pinned so nobody re-trips on them)

1. **Normalization drift within the lineage**: PRL18 normalizes to ψ_s, I19/L23
   to r_s, D21/D23b to w_ψ. Islands pins r_s-based forms with maps in
   `src/frames/` (docs/01 §5).
2. **Helical angle**: I19/L23 use ξ = m(θ − φ/q_s); Diss19/D21 use
   ξ = φ − q_s θ with cos nξ. Same island, different angle multiplicity.
3. **Collision-frequency energy dependence**: I19/L23 use the Chandrasekhar
   form; Diss19/D21 use V⁻³ (docs/05 E3 sub-toggle).
4. **ψ̃ amplitude**: one I19 extraction rendered ψ̃ = (w_ψ²/4)(q_s/q_s′);
   dimensional analysis and Diss19/D21/L23 give (w_ψ²/4)(q_s′/q_s)
   [VERIFY against I19 as printed — possible typo in the paper].
5. **DK-NTM run collisionality**: I19 §4.2 states ν_★ = 0.01; L23 p. 82 quotes
   DK-NTM at ν_★ = 10⁻³; D21 App. C benchmarks at ν_★ ~ 10⁻³–10⁻⁴
   [VERIFY before pinning B5a tolerances].
6. **Half vs. full width**: all York w_c values are half-widths; La Haye
   experimental fits are quoted as full widths (w_marg = 2w_c). Ladder
   reporting rule 5 exists because of this.
