# Pinned CI dependency sets

`Manifest.toml` is gitignored, as it should be for a library: developers resolve
their own environments. CI is the one consumer that wants the opposite, because
an unpinned resolve makes every run non-reproducible and quietly wastes time:

- Each run re-resolves against the live registry, so two runs an hour apart can
  test different dependency versions and a failure may not reproduce.
- Any upstream release invalidates the precompiled objects for that package and
  everything downstream of it in the restored depot cache, so the run pays to
  recompile a slice of the tree that had not otherwise changed.

`.github/workflows/test.yaml` therefore copies the pin matching the running Julia
minor version into `Manifest.toml` before building. The copy exists only inside
the CI runner and never reaches a developer's working tree.

## Compilation caches

CI uses `JULIA_CPU_TARGET=generic` for portable package images across runner CPUs of the same
architecture. The production depot and source-cache keys record the optimization and bounds-checking
settings, so incompatible images are not restored. Existing production caches remain reusable.

CI restores the dependency depot first, then overlays GPEC's compiled package directory from a
smaller cache keyed by Julia version, runner architecture, dependency pins, source contents, and
the test compilation settings. Successful runs save a missing GPEC entry on pull requests as well
as pushes. A dependency-key miss also saves a refreshed depot after a prefix restore, so newly
compiled dependency images are retained on pull requests. This lets test-only updates reuse the current package image without storing a new copy
of the full depot for each source change. The dependency cache alone is immutable and cannot
refresh GPEC after a source edit when its dependency key still matches.

For each Julia version, a production precompile job builds and caches the package before releasing
three parallel test groups (`core`, `api`, `errorfields`). Each group restores the exact depot and
source-cache keys published by that job, along with its resolved `Manifest.toml` and Julia version.
A missing cache fails the handoff rather than silently rebuilding dependencies in every group.
The two Julia versions progress independently. Cold compilation is paid once per version; warm
runs reuse the existing production images.

Test jobs use optimization level 1 to reduce first-use compilation. Julia can still reuse cached
package images built at level 2; the production build job keeps that default optimization level.
Only the test orchestration modules disable optimization entirely.

Bounds checking remains enabled. A source change still requires compiling a new GPEC image;
the source cache helps subsequent runs of that source, not the first build. Cache eviction or a
new Julia version likewise requires rebuilding. The test runner logs elapsed and compilation
seconds per file to distinguish this startup cost from work inside the tests.

## Files

| File | Purpose |
| --- | --- |
| `Manifest-v1.11.toml` | Dependency set for the `1.11` matrix job |
| `Manifest-v1.12.toml` | Dependency set for local Julia 1.12 runs |
| `Manifest-v1.13.toml` | Dependency set for the `1.x` matrix job |
| `project.sha256` | `Project.toml` checksum when the pins were generated |
| `update.jl` | Regenerates the above |

## Regenerating

Run once per Julia minor version the test matrix builds, from anywhere in the
repository:

```bash
julia +1.11 ci/manifests/update.jl
julia +1.12 ci/manifests/update.jl
julia +1.13 ci/manifests/update.jl
```

Each run resolves a bare copy of `Project.toml` in a temporary directory, so your
own `Manifest.toml` is left alone.

Regenerate whenever you change `[deps]` or `[compat]`, and periodically to pick up
upstream releases — a pin that never moves means CI stops noticing that a new
dependency version breaks the package.

## When CI tells you a pin is stale

The workflow compares `Project.toml` against `project.sha256` and emits a warning
annotation, not a failure, when they differ; `Pkg` also warns during the build
step that the manifest no longer satisfies the project. In that state CI still
passes, having re-resolved the affected packages, so the only cost is the
determinism and cache reuse the pin was there to provide. Regenerate and commit.

Because the checksum covers the whole file, bumping `version` in `Project.toml`
also trips the warning even though no dependency changed. Regenerating clears it.

## Adding a Julia version to the matrix

Add the matrix entry in `.github/workflows/test.yaml`, then generate the matching
`Manifest-v<major>.<minor>.toml`. A job with no matching pin is not an error — the
workflow warns and lets `Pkg` resolve normally.
