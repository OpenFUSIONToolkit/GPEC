using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using JPEC
JPEC.main([dirname(@__FILE__)])
JPEC.main([joinpath(@__DIR__, "single_n_1")])
JPEC.main([joinpath(@__DIR__, "single_n_2")])
