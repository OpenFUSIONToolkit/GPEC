using Test

# Smoke tests for Inputs module (minimal, non-invasive)
# This file is intentionally small: it verifies that the Inputs module can
# be included and that `newm` behaves as expected for a simple case.

include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
include(joinpath(@__DIR__, "..", "src", "pentrc", "inputs.jl"))
using .Inputs

@testset "Inputs smoke tests" begin
    v = Inputs.newm([1,2,3], [1+1im, 2+0im, 3-1im], [1,2,3])
    @test eltype(v) == ComplexF64
    @test v == ComplexF64[1+1im, 2+0im, 3-1im]
end
