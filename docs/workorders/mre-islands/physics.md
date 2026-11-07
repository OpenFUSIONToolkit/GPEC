# physics.md — configurable modified Rutherford equation and saturated-island flattening

Status: DRAFT for sign-off, 2026-11-06 (rev. 2: Benjamin Eqs. 8–14 and C.1 and Leuthold Eqs. 4.1, 4.2, 4.6 read from the rendered pages). Signed off by Nik C. Logan, 2026-11-06, in conversation (decisions D1–D7 below, answers recorded). Authoritative from the creation of `feature/mre-islands`; changes need a Decisions card. The agent may not change this file without a Decisions card.

Sources are cited by BBT key and equation/page; the reading notes are in `refs/notes/`. Two Fortran sources are code, not papers: `refs/notes/fortran_gpec160_rdcon_mre_terms.md` (PrincetonUniversity/GPEC v1.6.0 `rdcon/mercier.f`, the toroidal MRE surface terms of PrincetonUniversity/GPEC#224/#242) and `refs/notes/fortran_gpec160_callen.md` (v1.6.0 `gpec/gpout.f`, the Callen features of #629).

## 0. Coordinates and conventions

- ψ is normalized poloidal flux (0 axis, 1 boundary). Island width w is the **full** width in ψ, at the resonant surface ψ<sub>s</sub> of mode (m, n). Conversion to a length uses the toroidal-flux minor radius ρ<sub>tor</sub> = (Φ<sub>tor</sub>/(πB<sub>0</sub>))<sup>1/2</sup> as in Fortran `gpout.f` (`rhotor`): w<sub>r</sub> = w · dρ<sub>tor</sub>/dψ. All widths written to `gpec.h5` are in ψ; lengths are derived and labelled.
- Half-widths appear only inside the Callen preset (Fortran convention) and are converted at its boundary.
- Only linearly stable surfaces enter the root finding (gate in §4). Single helicity: each (m, n) surface is treated alone.

## 1. The MRE as a sum of switchable terms

Steady form; the time prefactor does not change the roots and is not implemented in this project.

  F(w) = s<sub>Δ</sub> Δ′(w) + s<sub>nc</sub> Δ<sub>nc</sub>(w) + s<sub>GGJ</sub> Δ<sub>GGJ</sub>(w) + s<sub>pol</sub> Δ<sub>pol</sub>(w; ω) + s<sub>vac</sub> Δ<sub>vac</sub>(w)

s<sub>i</sub> ∈ {0, 1} are deck switches; each term has the coefficients listed with it, all exposed in `[Tearing.ModifiedRutherford]`. Source: benjaminMacroscopicTrendsNeoclassical2026a Eq. 7 (p. 4) and Leuthold JPP 2026 Eq. 4.6, which agree; Δ<sub>pol</sub> from Benjamin App. C; Δ<sub>vac</sub> from Callen 2016 and Sweeney 2020 §2.2.

### 1a. Classical term with finite-pressure correction

  Δ′(w) = Δ′<sub>mn</sub> (w/2)<sup>−2α<sub>−</sub></sup> (−4D<sub>I</sub>)<sup>1/2</sup>,  α<sub>±</sub> = ½ ± (−D<sub>I</sub>)<sup>1/2</sup>

Benjamin Eq. 10 (p. 4), from Schlutt & Hegna. Limit: D<sub>I</sub> → −¼ gives Δ′(w) → Δ′<sub>mn</sub>. Benjamin's α convention is used (Leuthold prints (−4D<sub>I</sub>)<sup>1/2</sup> inside α; that form does not reduce to Δ′<sub>mn</sub> at zero pressure).
- Δ′<sub>mn</sub>: real part of the diagonal entry of `ForceFreeStatesResult.delta_prime.matrix` for the surface. [VERIFY V1: GPEC's Δ′ normalization (ψ<sup>−(−4D<sub>I</sub>)<sup>1/2</sup></sup> units, left+right jump) against Benjamin Eq. 6; check on one surface against the Fortran rdcon value.]
- D<sub>I</sub> = E + F + H − ¼ from `ResistGeometry` (`ForceFreeStates/Surfaces/ResistEval.jl`), equal to Fortran `mercier.f` `di`.
- Option `delta_prime_model = "mercier_power" | "constant"` (constant: Δ′(w) = Δ′<sub>mn</sub>, the Callen and Sweeney form).

### 1b. Bootstrap drive with transport threshold

  Δ<sub>nc</sub>(w) = k<sub>1</sub> D<sub>nc</sub> w / (w² + (k<sub>1</sub>/0.5) w<sub>d</sub>²),  k<sub>1</sub> = 1.7

Benjamin Eq. 11 (p. 4); Leuthold Eq. 4.6. Limits: → k<sub>1</sub>D<sub>nc</sub>/w for w ≫ w<sub>d</sub>; → 0.5 D<sub>nc</sub> w/w<sub>d</sub>² for w ≪ w<sub>d</sub>. The denominator width (k<sub>1</sub>/0.5)<sup>1/2</sup>w<sub>d</sub> = 1.84 w<sub>d</sub> agrees with Fitzpatrick 1995 W<sub>d</sub> = 1.8 W<sub>c</sub> (Eq. 124).

D<sub>nc</sub> follows the Fortran 1.6.0 surface term (`mercier.f`, after Hegna 1999 and Callen 2010):

  D<sub>nc</sub> = −(q/q′) (p′/⟨B²⟩) \overline{R²} ⟨B²/|∇ψ|²⟩ ψ<sub>o</sub><sup>−2</sup> · μ<sub>e</sub>/(ν<sub>e</sub> + μ<sub>e</sub>)

with \overline{R²} = ⟨R² V′/J⟩ (integrand 20 in `mercier.f`), p′ including μ<sub>0</sub>, and the banana-limit viscosity ratio

  μ<sub>e</sub>/(ν<sub>e</sub> + μ<sub>e</sub>) = f<sub>t</sub>(1 + 0.533/Z<sub>eff</sub>) / [(1 − f<sub>t</sub>) + f<sub>t</sub>(1 + 0.533/Z<sub>eff</sub>)].

Checked against the rendered Benjamin Eq. 13 (p. 4): D<sub>nc</sub> = −(q/(dq/dψ)) (μ<sub>0</sub>dp/dψ/⟨B²⟩) \overline{R²} ⟨B²/|∇Ψ|²⟩ μ<sub>e</sub>/(ν<sub>e</sub>+μ<sub>e</sub>), Ψ = ψΨ<sub>LCFS</sub>; identical to the Fortran (the ψ<sub>o</sub><sup>−2</sup> factor is the Ψ→ψ conversion). Benjamin Eq. 9 gives μ<sub>e</sub> = ν<sub>e</sub> f<sub>t</sub>/(1−f<sub>t</sub>)(1 + 0.533/Z<sub>eff</sub>), from which the ratio above follows; f<sub>t</sub> is the Sauter formula (Benjamin ref. 25), which is the Fortran form.

w<sub>d</sub> (transport threshold width; Fitzpatrick 1995 Eq. 27; Leuthold Eq. 4.2):

  w<sub>d</sub>(w) = c<sub>wd</sub> Ψ<sub>LCFS</sub><sup>−1</sup> [ (χ<sub>⊥</sub>/χ<sub>∥</sub>(w)) V′² ⟨B²⟩ ⟨|∇Ψ|²⟩ q²/(m² q′²) ]<sup>1/4</sup>,  c<sub>wd</sub> = 1

Benjamin Eq. 14 (p. 4, rendered), flux-space form after Schlutt & Hegna 2012; `mercier.f` `Wc_prefac`/m² is the bracket without χ<sub>⊥</sub>/χ<sub>∥</sub>. Transport estimates as Benjamin p. 4: χ<sub>⊥</sub> ~ n<sub>e</sub>κa²/(6τ<sub>E</sub>), χ<sub>∥</sub>(w) ~ n<sub>e</sub>v<sub>Te</sub>λ<sub>∥</sub>(w), λ<sub>∥</sub>(w) ~ R<sub>0</sub>q/(w n dq/dψ); with `KineticProfileData.chi_e` used for χ<sub>⊥</sub> when present. Leuthold Eq. 4.2 (rendered) instead uses the large-aspect-ratio form w<sub>d</sub> = 2√2(χ<sub>⊥</sub>/χ<sub>∥</sub>)<sup>1/4</sup>(εsn)<sup>−1/2</sup> (Fitzpatrick 1995 Eq. 27), and Sauter 1997 uses a prefactor 5.1. The prefactor is therefore exposed as c<sub>wd</sub> with nominal 1 (Benjamin, needed for the tearing-physics-suite cross-check) and swept over 1 to 2√2 in the uncertainty analysis (§6, D7). Schlutt & Hegna 2012 Eq. 41 and the text below it (p. 082514-6, rendered) give W<sub>C</sub> ~ ((χ<sub>⊥</sub>/χ<sub>∥</sub>) \overline{𝒥B²} \overline{𝒥g<sup>ψψ</sup>}/(ι′²m²))<sup>1/4</sup> as an order-of-magnitude scaling with no prefactor, so the prefactor is an order-unity uncertainty rather than a sourced constant.

### 1c. Glasser–Greene–Johnson (curvature) term

  Δ<sub>GGJ</sub>(w) = [k<sub>1</sub> D<sub>R</sub>/(α<sub>+</sub> − H)] / [w + k<sub>1</sub> w<sub>d</sub>/(0.3(1 + α<sub>+</sub>))],  D<sub>R</sub> = D<sub>I</sub> + (H − ½)²

Benjamin Eq. 12 (p. 5); Leuthold Eq. 4.6. D<sub>R</sub>, H from `ResistGeometry` (identical to `mercier.f` `locstab`). D<sub>R</sub> > 0 is the resistive-interchange instability flag and excludes the surface (§4).

### 1d. Ion polarization with island rotation as input

  Δ<sub>pol</sub>(w; ω) = −c<sub>pol</sub> 3.28 ε<sup>3/2</sup> I<sub>1</sub> [ (ρ<sub>iθ</sub>/r<sub>s</sub>) / (w/(2ψ<sub>s</sub>)) ]² (β<sub>p</sub>/w) [ (n<sub>i</sub><sup>−1</sup>dn<sub>i</sub>/dψ) / (q<sup>−1</sup>dq/dψ) ]² · ω(ω − ω<sub>\*pi</sub>)/ω<sub>\*i</sub>² · w<sup>4</sup>/(w<sup>4</sup> + w<sub>ib</sub><sup>4</sup>)

Benjamin Eq. C.1 (p. 20, rendered), from Eqs. 20–21 of its ref. 44: I<sub>1</sub> = 0.71; β<sub>p</sub> = 2μ<sub>0</sub>n<sub>i</sub>T<sub>i</sub>/⟨B<sub>pol</sub>²⟩ (Benjamin writes the Gaussian 8πn<sub>i</sub>T<sub>i</sub>); ρ<sub>iθ</sub> = m<sub>i</sub>v<sub>Ti</sub>/(⟨B<sub>pol</sub>⟩e); w<sub>ib</sub> = ε<sup>1/2</sup>ρ<sub>iθ</sub>, converted to ψ with the same r<sub>s</sub>↔2ψ<sub>s</sub> mapping as the first bracket; ω<sub>\*pi</sub> = ω<sub>\*i</sub>(1 + (T<sub>i</sub><sup>−1</sup>dT<sub>i</sub>/dψ)/(n<sub>i</sub><sup>−1</sup>dn<sub>i</sub>/dψ)); ψ<sub>s</sub>, r<sub>s</sub> at the rational surface. The coefficient product equals Wilson 1996 Eq. 85 (3.28 × 0.71 = 1.64 × 1.42). ω is the island frequency in the frame where E<sub>r</sub> = 0, per surface; ω<sub>\*i</sub> from the kinetic profiles. The rotation input follows §7. The mapping (ρ<sub>iθ</sub>/r<sub>s</sub>)/(w/2ψ<sub>s</sub>) is Benjamin's conversion from metres to normalized flux (its ref. 84) and is used as written.

Sign: the formula as written is stabilizing (Δ<sub>pol</sub> < 0) when ω(ω − ω<sub>\*pi</sub>) > 0, i.e. for ω outside (0, ω<sub>\*pi</sub>) with ω<sub>\*i</sub> > 0 in the ion direction. Benjamin's own text (p. 20) says rotation at ω<sub>\*i</sub> is stabilizing and electron-direction rotation may destabilize, which is the opposite reading unless ω<sub>\*i</sub> carries a negative sign in its convention. Wilson 1996 (Eq. 124) finds collisionless islands propagate at ω<sub>\*e</sub>(1 + η<sub>e</sub>/4), electron direction, with a stabilizing term. The literature sign is not settled (Imada 2018, Dudkovskaia 2023 revise it). This is decision D4.

Reduced alternative (`pol_model = "reduced"`): −c<sub>pol</sub> w<sub>pol</sub>²/w³ with w<sub>pol</sub> = C<sub>pol</sub> w<sub>ib</sub>, C<sub>pol</sub> = 0.5 (pedestal top, Callen report p. 23) to 1.9 (La Haye 2006). This form carries no ω and is used by the Callen preset.

Default `s_pol = 0` reproduces Benjamin and Leuthold, who drop it.

### 1e. External (vacuum-island) drive

  Δ<sub>vac</sub>(w) = c<sub>vac</sub> (w<sub>v</sub>/w)²

w<sub>v</sub> from `PerturbedEquilibrium` (`island_half_width`, ×2). c<sub>vac</sub> = Δ′<sub>RMP</sub> (Callen; definition in §2) or 2m cos Δφ in r<sub>s</sub>-normalized units (Sweeney 2020 §2.2). Default `s_vac = 0` (natural NTM). Used by the Callen preset and by RMP-driven-island cases.

### 1f. No separate saturation term

None of the sources has one; saturation comes from the w-dependence of Δ′(w) and the 1/w fall of Δ<sub>nc</sub>. The large-w analytic check is Sauter 1997 Eq. 5, w<sub>sat</sub> = β<sub>p</sub>(a<sub>bs</sub> − a<sub>GGJ</sub>)/(−Δ′) (large aspect ratio).

## 2. Callen preset (Fortran 1.6.0, reproduced as written)

From `gpout.f` v1.6.0 lines 1764–1890, half-widths in metres of ρ<sub>tor</sub>:

- ρ<sub>i</sub> = (2T<sub>i</sub>/m<sub>i</sub>)<sup>1/2</sup> / (Z<sub>i</sub>eB<sub>0</sub>/m<sub>i</sub>); w<sub>pol</sub> = 0.5 q ρ<sub>i</sub> / (r/R<sub>0</sub>)<sup>1/2</sup>
- Δ<sub>RMP</sub> = (|δ|/(2πR<sub>0</sub>q)) B<sub>0</sub>/|Φ<sub>res</sub>|, with δ the singular-current jump and Φ<sub>res</sub> the resonant flux of the applied field (total, not vacuum, as the Fortran comment says)
- Δ<sub>c</sub> = −2m/r
- critical vacuum half-width: h<sub>v,crit</sub> = ½ w<sub>pol</sub><sup>2/3</sup> ((27/4)|Δ<sub>c</sub>|)<sup>1/6</sup> / Δ<sub>RMP</sub><sup>1/2</sup>
- cubic for half-width x: Δ<sub>c</sub> x³ + Δ<sub>RMP</sub>(2h<sub>v</sub>)² x − w<sub>pol</sub>² = 0; the smaller positive root is h<sub>min</sub>, the larger h<sub>sat</sub>
- critical resonant flux: Φ<sub>res,crit</sub> = |Φ<sub>vac</sub>| (h<sub>v,crit</sub>/h<sub>v</sub>)²

This preset is matched to the Fortran outputs on the DIII-D-like regression example (§5 check C1). It differs from Callen's report Eq. (18) by an order-unity factor (the notes record 2.6 cm from the printed formula against 2.3 cm quoted); the Fortran is the reference because #630 requires agreement with 1.6.0.

## 3. Roots

F(w) on a log-spaced grid from w<sub>lo</sub> = max(δ<sub>r</sub>, 10<sup>−5</sup>) to w<sub>hi</sub> = 0.2 (ψ units), then bracketed root refinement.
- w<sub>m</sub>: smallest root with F crossing from negative to positive (seed threshold). w<sub>m</sub>\* = min over surfaces (Benjamin §2.4, p. 5).
- w<sub>sat</sub>: largest root with F crossing from positive to negative.
- F < 0 everywhere: "fully stabilized", recorded as a flag (Benjamin writes w<sub>m</sub>\* = 1; this project writes NaN plus the flag).
- F > 0 at w<sub>lo</sub>: "unstable to any seed", flag.

## 4. Gates and validity

- Interchange gate (Benjamin §2.3): D<sub>R</sub> > 0 → excluded as resistive-interchange unstable.
- Linear tearing gate [M0-D3, revised 2026-11-07]: the existing GPEC tearing solve, `solve(TearingProblem, SLAYERModel(; D_c_type = :toroidal))` in `:uncoupled` mode, on the same vacuum Δ′ the MRE uses. The SLAYER two-fluid layer gives the linear growth rate per surface; its critical Δ is the Connor, Ham, Hastie & Liu 2015 χ∥-matching closure with toroidal geometry (`toroidal_dgeo`, `toroidal_kpar`), including curvature through D<sub>R</sub> = E + F + H². A root with γ > 0 → linearly tearing unstable, flagged, no MRE roots. The critical Δ's χ⊥ (`chi_perp_Dc`) is set to the same χ⊥ that sets w<sub>d</sub> in §1b and takes the same ×0.3–×3 UQ multiplier, so the gate and the MRE share one transport assumption. The gate code accepts any inner-layer model, so the toroidal drift-resistive layer `CCCModel` (open PR #654, on #653) can be swapped in as a cross-check once it is on develop. Leuthold Eq. 4.1 is not used (its printed sign contradicts its text and its source, Fitzpatrick 2025 Eq. 7).
- Validity: δ<sub>r</sub> < w ≪ 1; Δ′w < 1 (Benjamin p. 5); constant-ψ; the polarization ordering needs w ≳ w<sub>ib</sub>. Roots outside these are flagged, not discarded.
- Quality screen on Δ′: |Re Δ′/Im Δ′| > 10³ (Benjamin footnote 2).

## 5. Pressure flattening (static)

Fortran 1.6.0 rule, generalized to any set of surfaces S (core or edge, any n, NTM- or RMP-driven): set dp/dψ = 0 on the union of [ψ<sub>s</sub> − w<sub>sat</sub>/2, ψ<sub>s</sub> + w<sub>sat</sub>/2] over S, anchor p at the boundary, integrate inward (trapezoid on the profile grid). Overlaps are counted once.
Outputs: p<sub>mod</sub>(ψ); ΔW = (3/2)∫(p − p<sub>mod</sub>) dV; ΔW/W; pedestal-top pressure change for edge sets; Δj<sub>bs</sub>: to leading order the bootstrap current is proportional to p′ at fixed collisionality coefficients, so inside a flattened island Δ⟨j<sub>bs</sub>·B⟩ = −⟨j<sub>bs</sub>·B⟩ and outside it is unchanged; ⟨j<sub>bs</sub>·B⟩ from the Sauter 1999 formula on the kinetic profiles (as the ARC basis Fig. 7 uses). Approximation recorded in the output attributes; the Fortran `Hbs_prefac` (Shi 2024 identity) converts ⟨j<sub>bs</sub>·B⟩ to the H<sub>bs</sub> term if needed (D6). No time evolution.
Option `flatten_model = "sharp" | "smooth"`; smooth weights the flattening by w²/(w² + (k<sub>1</sub>/0.5)w<sub>d</sub>²), consistent with §1b. This is a design choice, not sourced; default sharp.
Calibration (check, not a fit): Sauter 2010 Eq. 1, Δτ<sub>E</sub>/τ<sub>E</sub> = ϑ w/a, ϑ = 4(ρ<sub>s</sub>/a)³ [notes: equation partly reconstructed]; Sauter 1997: w/r<sub>s</sub> ≈ 10% costs about 20% of τ<sub>E</sub>.

## 6. Constants and ranges for the uncertainty sweeps

| Symbol | Nominal | Range for UQ | Source |
|---|---|---|---|
| k<sub>1</sub> | 1.7 | 1.0–2.5 [D3] | Benjamin p. 5 (footnote: value not important); Leuthold |
| c<sub>wd</sub> | 1 | 1 to 2√2 [D7] | Benjamin Eq. 14; Leuthold Eq. 4.2; Fitzpatrick Eq. 27 (Sauter 1997 uses 5.1 in its own normalization) |
| χ<sub>⊥</sub> | flat per machine [M1-D2]: SPARC κ<sub>a</sub>a²/(6τ<sub>E</sub>) = 0.123 m²/s (H-mode, τ<sub>E</sub> 0.77 s) and 0.215 m²/s (L-mode, τ<sub>E</sub> 0.44 s), with a = 0.57 m, κ<sub>a</sub> = 1.75; ARC 1.07 m²/s, the TORAX χ<sub>e</sub> at q = 2 of the Columbia TokaMaker+TORAX V3A run; a kinetic χ<sub>e</sub> profile overrides | ×0.3 to ×3 [D3] | Benjamin p. 4; Creely et al. JPP 2020 Table 1 and p. 13; Burgess `v3a_for_daniel.nc` slice 16 (t = 475 s) |
| c<sub>pol</sub> / C<sub>pol</sub> | 1 / 1.9 | C<sub>pol</sub> 0.5–1.9 | Callen report p. 23; La Haye 2006 |
| ω̂ = ω/ω<sub>\*i</sub> | ω<sub>\*e</sub>(1 + η<sub>e</sub>/4)/ω<sub>\*i</sub> [M1-D1] | scan −1 to 1, plus stalled ω = 0 | Wilson 1996 Eq. 124; Benjamin App. C |
| Δ′<sub>mn</sub> | Riccati, vacuum on | none [M0-D2]: Riccati and Galerkin agree to 10<sup>−3</sup> (M0); the conducting-wall case is not swept | M0 report |
| f<sub>t</sub> | Sauter-2002 form (Benjamin, Fortran) [D2] | Lin-Liu & Miller as alternative | `mercier.f`; Benjamin ref. 25; `NeoclassicalResistivity.jl` |

## 7. Rotation model (input to Δ<sub>pol</sub> and to the E×B frame)

Follows the ARC MHD basis (Leuthold JPP 2026 §2, "Rotation estimates", read from the rendered preprint pp. 4–5), applied to both machines:
- Core: toroidal rotation from the Rice et al. 2007 wave-heated intrinsic-rotation scaling at q = 2, M<sub>A</sub> = 0.65 β<sub>T</sub><sup>1.4</sup> q\*<sup>2.3</sup> (M<sub>A</sub> the Alfvén Mach number, q\* the cylindrical safety factor), applied from the axis to ψ<sub>q=2</sub> + 0.05 as Leuthold does; E<sub>r</sub> from the toroidal-rotation term of radial force balance.
- Edge: the E<sub>r</sub> well from the main-ion diamagnetic term (Viezzer et al. 2013, as cited by Leuthold), with the zero crossing at ψ<sub>N</sub> = 0.93 and E<sub>r</sub> = 0 at the separatrix.
- Range: 0.5× and 2× the model (Leuthold Fig. 2e). Cross-check: the turbulent momentum-transport workflow of Zimmermann et al. 2024 with zero edge rotation gives 3–9 krad/s at q = 2 for ARC, at or below the 1× case (Leuthold p. 5). Intrinsic torque for ARC 5–20 N m (Rice et al. 2021 gives 8 N m).
- If the SPARC and ARC kinetic files on feynman already carry `omega_E` built this way (the ARC files from the MHD-basis work should), those are used and the multiplier is applied to them; otherwise the agent builds the profile from the recipe above and records it in a report.
- Island frequency [M1-D1, 2026-11-07]: ω in Δ<sub>pol</sub> is the island frequency relative to the E<sub>r</sub> = 0 frame (§1d), not the bulk E×B rotation. Nominal ω = ω<sub>\*e</sub>(1 + η<sub>e</sub>/4), Wilson 1996's collisionless propagation in the electron direction, entered as ω̂ = ω/ω<sub>\*i</sub>; scanned over ω̂ ∈ [−1, 1] as in Benjamin App. C; the stalled island ω = 0 (Δ<sub>pol</sub> = 0) bounds it. ω = n ω<sub>E</sub> stays available as a lab-frame option only. The ω<sub>E</sub> profiles of this section still set the E×B frame of the SLAYER gate (§4).

## Decisions (signed 2026-11-06)

- D1. Master form: the toroidal Benjamin/Leuthold form (§1) with the Fortran `mercier.f` surface terms as the reference implementation; large-aspect-ratio Sauter/La Haye coefficients are used only in the TJ cylindrical checks.
- D2. Trapped fraction: the Sauter-2002 form of `mercier.f`, which is also what Benjamin used (ref. 25), as the MRE default, with Lin-Liu & Miller (already in Julia for resistivity) as an option. Reason: the tearing-physics-suite cross-check and #630's 1.6.0 comparison both use the Sauter form. Answer: accepted.
- D3. UQ ranges (no source states one): k<sub>1</sub> 1.0–2.5; χ<sub>⊥</sub> ×0.3–×3 about the profile value; stated in the paper as assumed ranges. Answer: accepted.
- D4. Polarization sign. Options: (a) implement Eq. C.1 as written, ω<sub>\*i</sub> > 0 in the ion direction, so ion-direction rotation in (0, ω<sub>\*pi</sub>) is destabilizing and electron-direction rotation stabilizing (consistent with Wilson 1996); (b) flip so that ion-direction rotation is stabilizing (Benjamin's text, La Haye 2022 'gate'). Answer: (a), Eq. C.1 as written is the default, a `pol_sign` switch flips it, and KR2 is reported for both signs.
- D5. Rotation: the ARC-basis model of §7 (Nik, 2026-11-06: follow the Leuthold rotation description, the Rice–Zimmermann lineage), at 0.5×, 1× and 2×, with the stalled island ω = 0 as a separate bounding case.
- D6. Flattening default sharp (Fortran rule); Δj<sub>bs</sub> by the leading-order p′ proportionality above.
- D7. c<sub>wd</sub> nominal 1 (Benjamin toroidal form), swept 1 to 2√2.

## Decisions from milestone reports

- M0-D2 (2026-11-07, Nik: option c). Δ′ is computed with the vacuum only; the Riccati–Galerkin spread (about 10<sup>−3</sup>) is the recorded Δ′ formalism uncertainty and the wall case is dropped from the sweeps.
- M1-D1 (2026-11-07, Nik: option a). Polarization island frequency prescribed relative to the diamagnetic frequency (§7, §6); ω = ω<sub>E</sub> mixed frames (M1 report §5).
- M1-D2 (2026-11-07, Nik: option a, with TORAX χ where available). Nominal χ<sub>⊥</sub> per machine (§6). Expected to be replaced by co-author transport runs after draft 1.
- M0-D3 (2026-11-07, Nik). The linear tearing gate reuses the existing tearing module instead of Leuthold Eq. 4.1. Revised the same day: SLAYER inner layer with the toroidal critical-Δ (`D_c_type = :toroidal`) as nominal; `CCCModel` (PR #654) as a cross-check when merged; §4 amended.

## [VERIFY] items remaining

V1 Δ′ normalization (GPEC `delta_prime.matrix` against Benjamin Eq. 6, units ~ψ<sup>−(−4D<sub>I</sub>)<sup>1/2</sup></sup>). This is a numerical check, made in M0 by comparing the Galerkin (RDCON) value with the Fortran rdcon value on the DIII-D-like example; if they differ by a convention factor, M0 records it as a Decisions card. No other [VERIFY] remains: V2–V4 were read from the rendered Benjamin pages, V5 became D4, V6 became D6.
