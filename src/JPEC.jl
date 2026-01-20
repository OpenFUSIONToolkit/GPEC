# JPEC.jl
module JPEC

include("Splines/Splines.jl")
import .SplinesMod as Spl
export SplinesMod, Spl

include("Equilibrium/Equilibrium.jl")
import .Equilibrium as Equilibrium
export Equilibrium

include("BIEST.jl")
import .BIEST as BIEST
export BIEST

include("Vacuum/Vacuum.jl")
import .Vacuum as Vacuum
export Vacuum

include("DCON/DCON.jl")
import .DCON as DCON
export DCON

include(joinpath(@__DIR__, "..", "deps", "build_helpers.jl"))
export build_fortran, build_spline_fortran, build_vacuum_fortran, build_biest

end # module JPEC
