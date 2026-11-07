# Plotting Conventions

GPEC draws figures with [Makie](https://docs.makie.org). Makie is a weak dependency: the drawing code lives in the
package extension `ext/GPECMakieExt/`, which loads when a session loads any Makie backend (`using CairoMakie` for
files, `using GLMakie` for interactive windows). `src/Analysis/` holds the data extraction, the registry and the
public `plot_*` functions, and calls the extension's drawing functions through `Analysis._draw`.

## Adding an Analysis plot

**Sparingly.** Analysis plots are polished figures that users of a module want often in their everyday workflow:
flagship results, diagnostics and applications of the code. Never add one for a one-off benchmark or test, a
diagnostic of a scan, a comparison against Fortran, or anything of that kind.

Before adding one, ask: *will I look at this often when I use this workflow?* Add it only if the answer is yes, and
would be yes for other users too. Otherwise draw it yourself in a script: the results and the `_x_data` extractors
make simple and one-off plots easy to make by hand.

AI agents must not offer to add Analysis plots unprompted, and a human must approve any plot an agent adds as meeting
these criteria before it lands. A new feature does not by itself warrant a new Analysis plot.

Every plot:

- is registered in the plot registry (`src/Analysis/Registry.jl`) with the capabilities it needs;
- follows the theme and the figure conventions below;
- carries little text: a short title that states the run's result, terse legends, and no explanatory notes on the
  figure;
- throws an `ArgumentError` that names what to change in the run when the data are absent, never a text panel or
  `nothing`.

### Mechanics

- **Data.** Read the data through one `_x_data(src)` method per source (the result's fields, and the same datasets
  from a `gpec.h5` path), returning a NamedTuple, so both sources reach one drawing function.
- **Public function.** `plot_x(src; ..., save_path=nothing)` in `src/Analysis/` starts with `_require(plot_x, src)`,
  which throws the registry's reason when the source lacks the data, and ends with
  `_draw(_plot_x, data; save_path, kwargs...)`. Declare the drawing function in the module with `function _plot_x end`.
  Composite summaries skip the panels they cannot draw (`_can`) and throw only when none can.
- **Registry.** Call `register!(plot_x, sources, needs, description)` after the definition. A new capability in
  `needs` gets a `capabilities` predicate for each result type that can hold it, a dataset in `_H5_CAPABILITIES`, and
  a `REASONS` sentence saying what to change in the run.
- **Drawing.** Add the `A.<Module>._plot_x` method to the matching file in `ext/GPECMakieExt/`. Build the figure inside
  `themed() do ... end`, and reuse the shared renderers before writing new ones: `_surface_slice!` (a field on the
  control surface in φ and θ), `_poloidal_slice!` (a field on the (R, Z) flux-surface grid), `_coil_traces!`,
  `_surfaces_on_top!` (rational surfaces along the top of a ψ_N axis), and the label helpers in `theme.jl` (`_num`,
  `_power`, `_unit`, `sub`, `sup`, `PSIN`).
- **Tests.** `test/runtests_analysis_plots.jl` draws every registered plot from both sources and checks the refusals;
  add an extractor to its list when the plot reads new data. Plot tests run in the `plots` test group, which stacks a
  CairoMakie environment onto the load path; the other groups never load Makie.

## Figure conventions

- **One theme.** `GPEC_THEME` in `ext/GPECMakieExt/theme.jl`: 15 pt labels, 13 pt ticks, a 16 pt regular-weight
  title, faint grid, full box, the colour-blind-safe Wong palette, and legends on a light translucent backing placed in
  empty space or outside the data. Figure size follows the panel count, with no empty grid cells.
- **Self-describing.** The title and axis labels say what is plotted, its normalization and its units. A run label
  appears only when several runs are overlaid.
- **Typography.** Real glyphs and rich text, never ASCII `_` or `^`: ψ_N, ξ^ψ_m, |b̄_res|, Δ′. Units in parentheses.
  Numbers to three significant figures, with ×10ⁿ outside [1e-2, 1e3); an axis whose magnitudes sit outside that range
  is scaled by a power of ten in its label, as in "|b̄_res| (10⁻⁴ T)". Tick labels never repeat one value:
  near-constant data is reported as such.
- **Axes.** Per-rational-surface quantities go on an evenly spaced categorical `q = m/n` axis, as stems and dots,
  unless where in ψ they sit is the point. Use a log scale when the data span more than about two decades, with ticks
  at whole decades, and a diverging, symmetric colour range for signed data. Values that are roundoff against the
  run's scale are left out and counted in the title.
- **3D.** Equal data aspect, and solid surfaces occlude what is behind them (CairoMakie has no depth buffer, so draw
  in back-to-front order).

### Spectrum plots

Any plot with discrete mode numbers m or n on the x axis draws centred steps padded with a zero at each end:

```julia
step_series(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))
stairs!(ax, step_series(m, amplitudes)...; step=:center, linewidth=2, label="...")
```

## Saving figures

- `save_path` saves through `Makie.save` with whichever backend is loaded. CairoMakie writes PNG, PDF and SVG;
  GLMakie writes PNG only and throws an `ArgumentError` for vector formats.
- Always print the full absolute path of any figure you save, so it can be opened directly.
