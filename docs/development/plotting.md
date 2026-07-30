# Plotting Conventions

## Spectrum plots

Any plot with discrete mode numbers m or n on the x-axis must use `seriestype=:steppre` with a `step_series` helper that pads zeros on both ends. Pattern from `benchmarks/benchmark_coil_ForcingTerms_against_fortran.jl`:

```julia
function step_series(m_vals, amps)
    m_ext   = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end
# Usage:
m_ext, a_ext = step_series(m_vals, amplitudes)
plot!(p, m_ext, a_ext; seriestype=:steppre, lw=2, label="...")
```

## Figures and plots

- Always print the full absolute path of any figure or plot file you save, so the user can open it directly without searching the filesystem.
- Always check that axis labels are not clipped. In Plots.jl there is no `tight_layout()` equivalent; use explicit margins instead: `left_margin=12Plots.mm`, `bottom_margin=4Plots.mm`, etc. When in doubt, add a generous `left_margin` to prevent y-axis label cutoff.
