"""
    GPECMakieExt

Makie drawing for `GeneralizedPerturbedEquilibrium.Analysis`. Loads with any Makie backend: `using CairoMakie` for
files, `using GLMakie` for interactive windows.
"""
module GPECMakieExt

using Makie
import GeneralizedPerturbedEquilibrium.Analysis as A

include("theme.jl")
include("fields.jl")
include("equilibrium.jl")
include("forcefreestates.jl")
include("perturbedequilibrium.jl")
include("coilforcing.jl")
include("errorfields.jl")

# Save `fig` to `path` with the loaded backend; GLMakie writes raster images only.
function A._save(fig::Figure, path::AbstractString)
    ext = lowercase(splitext(path)[2])
    backend = nameof(Makie.current_backend())
    backend === :GLMakie && ext in (".pdf", ".svg", ".eps") &&
        throw(ArgumentError("GLMakie writes PNG only; load CairoMakie (`using CairoMakie`) to save $ext"))
    save(path, fig)
    return nothing
end

# Math is set in the theme's sans face, so accented symbols such as F̄ match the text; this applies to the session's
# LaTeX strings once GPEC's figures are loaded.
function __init__()
    Makie.MathTeXEngine.set_texfont_family!(Makie.MathTeXEngine.FontFamily("TeXGyreHeros"))
    A._BACKEND[] = true
    return nothing
end

end # module GPECMakieExt
