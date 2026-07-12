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
| Pitch-angle collision operator + deflection frequency + ``\nu_\star`` | [collision operator](collision-operator.md) | ✅ signed off 2026-07-11 (``\langle\hat\nu_{ii}\rangle_u`` deferred) |
| Flattened-electron closure ``h(\Omega)`` + amplitude | [electron closure](electron-closure.md) | ✅ signed off 2026-07-11 (``k``, ``f_p`` deferred) |
| Quasineutrality closure ``1/(2\hat L_{n0})`` (arbitrary ``\tau``) | [quasineutrality closure](quasineutrality-closure.md) | ✅ signed off 2026-07-11 |
| ``\Delta_{\cos}/\Delta_{\sin}`` moment prefactors ``\mp\mu_0 R/2\tilde\psi`` | [Δ-moment prefactors](delta-moment-prefactors.md) | ✅ signed off 2026-07-11 |
| Parallel (island) streaming ``a_\xi``, ``a_x`` (advection along ``\Omega``) | [parallel streaming](parallel-streaming.md) | ✅ signed off 2026-07-11 |
| Gradient drive = the diamagnetic far-field BC (I19 Eq. 29) | [gradient drive](gradient-drive.md) | ✅ signed off 2026-07-11 |
| Passing fraction ``f_p \simeq 1-1.46\sqrt\varepsilon`` (electron closure) | [passing fraction](passing-fraction.md) | ✅ signed off 2026-07-11 |

The six main Q3/Q4 coefficient families are signed off, plus the parallel
streaming and the passing fraction ``f_p`` (`Coefficients.passing_fraction`).
The gradient drive's *structure* is found (it is the far-field BC, not a source);
its normalized amplitude is bundled with the frame convention and awaits
completion. Of the **deferred numerical sub-constants**, ``f_p`` is now cleared;
the remaining two — ``\langle\hat\nu_{ii}\rangle_u`` (collision) and the
Hirshman–Sigmar ``k`` (electron closure) — are escalated in `QUESTIONS.md`
Q3/Q5 (each needs its specific source integrand) rather than derived
speculatively.
