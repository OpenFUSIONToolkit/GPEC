# GPEC documentation standard

This is the authoring standard for the GPEC user documentation under
`docs/src/`. The goal is that each physics module page reads like a compact
*Journal of Computational Physics* methods paper: the governing equations, the
numerical method, and the validation that the method works — with figures whose
provenance is explicit. Those module pages **are** built by Documenter and
published on the public documentation site.

This standard file (`docs/DOC_STANDARD.md`) is itself the one exception: it lives
outside `docs/src/` and is not listed in `make.jl`, so Documenter does not
render it to the website. It is a contributor-facing reference, read here in the
repository — which is why `developer_notes.md` links to it by GitHub URL rather
than as a site-internal page.

The `Ballooning and Mercier Local Stability` and `Inner Layer Module` pages are
the reference exemplars; new or reworked module pages should follow their shape.

## Module-page template

A module reference page (`docs/src/<module>.md`) has these sections, in order.
Not every section is large for every module, but the skeleton is the same.

1. **Overview** — what the module computes and where it sits in the
   equilibrium → stability → perturbed-equilibrium pipeline.
2. **Governing equations** — the physics the module discretizes, written as
   numbered display equations. Each equation or block cites the paper it comes
   from (the PDFs in `docs/resources/`), e.g. *(GWP2016 Eq. 11)*. Citing
   equation numbers is required, not optional — it is what makes the
   implementation auditable against theory.
3. **Numerical method** — the discretization and algorithm: what is expanded in
   what basis, how the linear/eigenvalue/BVP problem is assembled and solved,
   and the *design choices* that matter (why this contour, this ordering, this
   precision). This is the part that distinguishes a methods page from an API
   dump.
4. **Validation & benchmarks** — evidence the method is correct: a table of
   pinned values cross-checked against an independent code or analytic result,
   and figures (see below) showing convergence, invariance, or agreement.
5. **Practical usage** — the configuration keys and the public entry points a
   user calls, with a short worked example.
6. **API Reference** — an `@autodocs` block for the module so every exported
   docstring is rendered and `checkdocs=:exports` stays satisfied.

## Figures

### Where figures live

- **Content figures** (anything shown on a page) live in
  `docs/src/figures/<module>/`. Each committed `<name>.png` sits next to the
  script `make_<name>.jl` that produced it — script and figure are always
  committed together, so it is always clear how a figure was made.
- `docs/src/assets/` is reserved for **Documenter chrome only** (the site logo,
  themes, custom CSS). Content figures never go there.
- The shared helper `docs/figure_tools.jl` (outside `src/`, so it is not
  published) provides `save_doc_figure`, provenance stamping, the manifest, and
  the `step_series` spectrum helper. Every `make_<name>.jl` `include`s it.

### Where a figure's data comes from

Documentation figures get their **own** generator scripts under
`docs/src/figures/`. They are not forced to route through `benchmarks/`. When a
figure genuinely *is* a benchmark or Fortran cross-check result, its
`make_<name>.jl` may call the relevant `benchmarks/` script, and its `depends`
list records that coupling — but the committed PNG and a runnable script still
live together under `docs/src/figures/<module>/`.

`benchmarks/` remains the home of performance comparisons and Fortran
cross-checks (its outputs are git-ignored and may be heavy). `docs/src/figures/`
is the home of the small, purpose-built, committed figures that illustrate a
page.

### Figures do not build with the docs

Figure scripts run **manually**, in the root project:

```bash
julia --project=. docs/src/figures/<module>/make_<name>.jl
```

They are **never** run at Documenter build time. The build only embeds the
committed PNGs, so `docs/Project.toml` stays minimal (no `Plots`) and the build
stays fast. This is also why we do not regenerate every figure on every docs
change — see the regeneration policy below.

### Provenance

`save_doc_figure` stamps a small `GPEC <shorthash> · <date>` mark in the corner
of every figure (so a figure's age is visible when browsing the rendered site)
and writes a machine-readable entry to `docs/src/figures/manifest.toml`:

```toml
[inner_layer.rotated_ray_contour]
script  = "make_rotated_ray_contour.jl"
commit  = "3a0837e3"
date    = "2026-07-08"
depends = ["src/InnerLayer/GGJ/Ray.jl"]
```

`depends` lists the repo-relative source files whose *numbers* the figure
visualizes. Generate figures as the last step before committing a change; the
stamp then reflects the state they were made against (a `-dirty` suffix marks a
figure generated with an uncommitted working tree, which is honest, not an
error).

### When to regenerate a figure

Regenerate a figure (and its manifest entry) only when:

1. a file in its `depends` list changed the numbers it shows — the regression
   harness is the trigger: if a tracked quantity for that module moved, its
   figures are suspect;
2. the physics or method the figure illustrates changed; or
3. the figure script itself changed.

Do **not** regenerate for prose edits, formatting, or changes to unrelated
modules. Because each figure carries its generating commit and a `depends`
list, a reviewer (or a future `docs/check_figures.jl`) can tell when a figure
predates the last change to a file it depends on and is therefore stale.

### Legacy figures

A figure migrated from before this system, whose original generator was not
preserved and which cannot be faithfully reproduced from current code, is
registered with `register_legacy_figure` (`commit = "legacy"`, a `note`
explaining why). This is honest bookkeeping for genuinely irreproducible
figures — it is never a substitute for writing a generator for a new figure.
