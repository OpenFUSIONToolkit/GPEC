"""
    _plot_cond_fbar.jl

Thin shim used by the kinetic-DCON benchmark scripts. Delegates to
`GeneralizedPerturbedEquilibrium.Analysis.ForceFreeStates.plot_cond_fbar`
so the benchmarks share the same visualisation that every user gets from
the Analysis module.
"""

using GeneralizedPerturbedEquilibrium
const _AnalysisFFS = GeneralizedPerturbedEquilibrium.Analysis.ForceFreeStates

"""
    plot_cond_fbar_scan(h5path, png_path; title="") → png_path or nothing

Save a cond(F̄) vs ψ plot to `png_path` by calling
`Analysis.ForceFreeStates.plot_cond_fbar(h5path; save_path=png_path)`. The
`title` argument is ignored (the Analysis function composes its own title
from the stored kmsing count); the shim keeps the old signature so
existing benchmark scripts do not need to change.
"""
function plot_cond_fbar_scan(h5path::String, png_path::String; title::String="")
    _ = title  # kept for backward compatibility; Analysis function composes its own title
    p = _AnalysisFFS.plot_cond_fbar(h5path; save_path=png_path)
    p === nothing && return nothing
    println(stderr, "  cond(F̄) plot: ", abspath(png_path))
    return abspath(png_path)
end
