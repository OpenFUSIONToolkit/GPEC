# Derivations

Independent re-derivations of the Level-0 physics coefficients (Decision D7,
"re-derivation first"): each is marked `[DERIVED: date]`, carries a cross-check
table against the `[CHECKED]` literature transcriptions (docs/01), and flags
every discrepancy in the open (policy rule 4, docs/05 triage). **A derivation
authorizes a coefficient in `src/` only after human sign-off** (recorded in
docs/01, policy rule 3); until then the corresponding coefficient stays a gated,
supplied argument.

These pages are *not* literature transcriptions — they derive the result from a
stated starting point and assumptions, then compare. That distinction is the
spine of the milestone (policy rule 4: never present a derivation as a
transcription or vice versa).

## Chapters

| Coefficient | Chapter | Status |
|---|---|---|
| Island flux amplitude ``\tilde\psi`` | [ψ̃ amplitude](psi-tilde-amplitude.md) | ✅ signed off 2026-07-11 |
| Magnetic drift frequency ``\hat\omega_D`` + `:original`/`:improved` toggle | [ω̂_D drift frequency](omega-D-drift-frequency.md) | ✅ signed off 2026-07-11 |
| Pitch-angle collision operator + deflection frequency + ``\nu_\star`` | [collision operator](collision-operator.md) | ✅ signed off 2026-07-11 |
| Collision magnitude ``\nu_\star`` + momentum-restoring ``\langle\hat\nu_{ii}\rangle_u=\tfrac{4\varepsilon^{3/2}\nu_\star}{3\sqrt\pi}(\sqrt2-\ln(1+\sqrt2))`` | [collision magnitude](collision-magnitude.md) | ✅ signed off 2026-07-12 |
| Full orbit-averaged collision operator (6 terms; σ-odd pitch diffusion ``\partial_y(y\langle\sqrt{1-yb}\rangle\partial_y)``) | [orbit-averaged collision](orbit-averaged-collision.md) | ✅ signed off 2026-07-12 (5/6 differential terms implemented; momentum F pending) |
| Flattened-electron closure ``h(\Omega)`` + amplitude | [electron closure](electron-closure.md) | ✅ signed off 2026-07-11 (``k``, ``f_p`` deferred) |
| Quasineutrality closure ``1/(2\hat L_{n0})`` (arbitrary ``\tau``) | [quasineutrality closure](quasineutrality-closure.md) | ✅ signed off 2026-07-11 |
| ``\Delta_{\cos}/\Delta_{\sin}`` moment prefactors ``\mp\mu_0 R/2\tilde\psi`` | [Δ-moment prefactors](delta-moment-prefactors.md) | ✅ signed off 2026-07-11 |
| Parallel (island) streaming ``a_\xi``, ``a_x`` (advection along ``\Omega``) | [parallel streaming](parallel-streaming.md) | ✅ signed off 2026-07-11 |
| E×B coupling ``c_E = \tfrac12\langle 1/\hat v_\parallel\rangle_\theta`` (passing σ-odd, trapped ≡ 0) | [E×B coupling](exb-coupling.md) | ✅ signed off 2026-07-12 |
| Gradient drive = the diamagnetic far-field BC (I19 Eq. 29) | [gradient drive](gradient-drive.md) | ✅ signed off 2026-07-11 |
| Passing fraction ``f_p \simeq 1-1.46\sqrt\varepsilon`` (electron closure) | [passing fraction](passing-fraction.md) | ✅ signed off 2026-07-11 |

The six main Q3/Q4 coefficient families are signed off, plus the parallel
streaming, the passing fraction ``f_p`` (`Coefficients.passing_fraction`), the
E×B coupling ``c_E``, and the collision magnitude (``\nu_\star`` normalization +
``\langle\hat\nu_{ii}\rangle_u``). The gradient drive is cleared as the
diamagnetic far-field BC (not a source). Of the **deferred numerical
sub-constants**, ``f_p`` and ``\langle\hat\nu_{ii}\rangle_u`` are now cleared;
only the Hirshman–Sigmar ``k`` (electron closure, `QUESTIONS.md` Q3/Q5 — its own
parallel-viscosity moment problem, L23 Eq. 4.1.7) remains escalated rather than
derived speculatively. The one remaining Level-0 operator gate is the
orbit-averaged pitch measure `B_profile`.
