# TOML Annotation Conventions

Config-style TOML files (`examples/*/gpec.toml`, `examples/*/sol.toml`, the
`test/test_data/*` fixtures, and `regression-harness/cases/*.toml`) follow one shared
annotation style so they stay consistent and self-documenting for users who copy and
modify them. **Do not invent a new convention** — match what the existing example files do.

1. **File header (2–10 lines).** Begin each config-style TOML with a plain `#` comment block
   stating the example's purpose/context and calling out key settings (wall choice, forcing,
   n-range, ideal vs kinetic). No ASCII-art separators or decorative dividers.
2. **Inline annotation on every variable line.** Every `key = value` gets a trailing
   `# description`. Source the wording from the matching config struct's docstring
   (`EquilibriumConfig`, `WallShapeSettings`, `ForceFreeStatesControl`, `ForcingTermsControl`,
   `PerturbedEquilibriumControl`, `SolovevConfig`, `TJAnalyticConfig`, `KineticForcesControl`)
   and keep it to one terse line. **Use the same description for the same variable across all
   files** — `examples/DIIID-like_ideal_example/gpec.toml` is the canonical reference for the
   common sections. Exception: in `regression-harness/cases/*.toml` the repeated
   `[quantities.*]` schema keys (`h5path`, `type`, `extract`, `label`, `noise_threshold`,
   `order`) are developer metadata and do **not** need inline comments — a case file needs
   the header plus an informative block comment per quantity group instead.
3. **Section comments only when informative.** No comment is required above a section. Keep a
   block comment only when it carries real information (e.g. the Solovev `[Wall]` note on why a
   conformal wall is needed). Never add decorative section dividers.
4. **No Fortran references — annotations must be self-contained and Julia-centric.** The Julia
   code is the production workhorse; a new user must not need to know the legacy Fortran GPEC
   code to read an example. Headers explain the *scenario and its unique aspect*; inline comments
   explain the *variable*. Do **not** cite Fortran namelist files (`dcon.in`/`equil.in`/`pentrc.in`),
   Fortran flag names (`kin_flag`, `sas_flag`, `electron_flag`, …), or legacy code names
   (`DCON`/`STRIDE`/`PENTRC`) as things the reader must know. Non-Fortran external-model citations
   (e.g. the analytic TJ model behind `tj_analytic`) are acceptable provenance. (This applies to
   the user-facing `examples/*` and `test/test_data/*` TOMLs; `regression-harness/cases/*` may
   retain algorithm names as developer metadata.)
5. **Only active, meaningful variables in examples.** Do not set deprecated or "not yet
   implemented" variables just to mirror defaults — they add clutter and imply false relevance
   (e.g. the `truncate_at_dW_peak` edge-peak truncation, `thmax0`). Omit them so examples show
   only knobs that do something on that run.
6. **`Project.toml`-family files are exempt.** `Project.toml`, `docs/Project.toml`,
   `regression-harness/Project.toml`, and `.JuliaFormatter.toml` are machine-managed; do not
   inline-annotate dependency UUIDs or `[compat]` entries.

## Enforcement

Four `pygrep` hooks in `.pre-commit-config.yaml` (no external scripts — a pygrep hook
fails when its pattern matches a violation) lint-check the covered TOMLs on commit:

- `toml-header-block` — the file must open with a `#` header block (all covered files).
- `toml-no-decorative-dividers` — no `# ----`-style divider lines (all covered files).
- `toml-inline-annotations` — every `key = value` line carries a trailing `# description`
  (`examples/*` and `test/test_data/*` only, per the rule-2 exception above).
- `toml-no-deprecated-keys` — no deprecated config keys; its key list mirrors
  `_DEPRECATED_FFS_KEYS`/`_DEPRECATED_EQUIL_KEYS` in `src/GeneralizedPerturbedEquilibrium.jl`,
  so extend the hook's pattern whenever a key is deprecated there.

Run them manually with:

```bash
pre-commit run --all-files
```
