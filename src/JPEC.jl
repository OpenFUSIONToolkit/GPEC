# JPEC.jl
module JPEC

include("Splines/Splines.jl")
import .SplinesMod as Spl
export SplinesMod, Spl

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

include("Vacuum/Vacuum.jl")
import .VacuumMod as VacuumMod
export VacuumMod

include("DCON/DCON.jl")
import .DCON as DCON
export DCON

include("Pentrc/pentrc.jl")
import .PENTRC as PENTRC
export PENTRC

include(joinpath(@__DIR__, "..", "deps", "build_helpers.jl"))
export build_fortran, build_spline_fortran, build_vacuum_fortran

end # module JPEC