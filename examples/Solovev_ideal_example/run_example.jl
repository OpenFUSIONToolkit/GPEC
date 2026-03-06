using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using JPEC
JPEC.main([dirname(@__FILE__)])
