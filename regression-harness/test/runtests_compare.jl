using HDF5, JSON, Test

include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "extractor.jl"))

# A stored quantity as get_quantities returns it.
stored(value_type; value_real=nothing) =
    (label="q", value_real=value_real, value_int=nothing, value_text=nothing, value_type=value_type, noise_threshold=1e-10)

@testset "A quantity that disappears or changes type is a change" begin
    real1, real2 = stored("real"; value_real=1.0), stored("real"; value_real=2.0)
    @test compare_values(real1, real1)[3] == "OK"
    @test compare_values(real1, real2)[3] == "CHANGED"
    @test compare_values(stored("missing"), stored("missing"))[3] == "N/A"
    @test compare_values(real1, stored("missing"))[3] == "CHANGED (missing)"
    @test compare_values(stored("missing"), real1)[3] == "NEW"
    @test compare_values(real1, stored("integer"))[3] == "CHANGED (type)"
end
