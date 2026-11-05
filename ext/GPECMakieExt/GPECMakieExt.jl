"""
    GPECMakieExt

Makie drawing for `GeneralizedPerturbedEquilibrium.Analysis`. Loads with any Makie backend: `using CairoMakie` for
files, `using GLMakie` for interactive windows.
"""
module GPECMakieExt

using Makie
import GeneralizedPerturbedEquilibrium.Analysis as A

include("theme.jl")
include("equilibrium.jl")

# Save `fig` to `path` with the loaded backend; GLMakie writes raster images only.
function A._save(fig::Figure, path::AbstractString)
    ext = lowercase(splitext(path)[2])
    backend = nameof(Makie.current_backend())
    backend === :GLMakie && ext in (".pdf", ".svg", ".eps") &&
        throw(ArgumentError("GLMakie writes PNG only; load CairoMakie (`using CairoMakie`) to save $ext"))
    save(path, fig)
    return nothing
end

__init__() = (A._BACKEND[] = true)

end # module GPECMakieExt
