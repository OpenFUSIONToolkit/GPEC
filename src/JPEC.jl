# JPEC.jl
module JPEC

include("Splines/Splines.jl")
import .SplinesMod as Spl
export SplinesMod, Spl

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

include("Vacuum/Vacuum.jl")
import .Vacuum as Vacuum
export Vacuum

include("ForceFreeStates/ForceFreeStates.jl")
import .ForceFreeStates as ForceFreeStates
export ForceFreeStates

include("ForcingTerms/ForcingTerms.jl")
import .ForcingTerms as ForcingTerms
export ForcingTerms

include("PerturbedEquilibrium/PerturbedEquilibrium.jl")
import .PerturbedEquilibrium as PerturbedEquilibrium
export PerturbedEquilibrium

include("Main.jl")
export Main

include(joinpath(@__DIR__, "..", "deps", "build_helpers.jl"))
export build_fortran, build_spline_fortran, build_vacuum_fortran

end # module JPEC