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

The remaining Q3 items (the analytic ``\langle\hat\nu_{ii}\rangle_u`` velocity
average, the flattened-electron closure ``h(\Omega)``/``k``/``f_p``, the
quasineutrality closure, and the ``\Delta_{\cos}/\Delta_{\sin}`` prefactors) are
added here as the derivation lane proceeds (M2b contract).
