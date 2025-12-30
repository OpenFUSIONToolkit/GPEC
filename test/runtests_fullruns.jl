# Run JPEC.main on the provided example directories and assert it completes without throwing.
@testset "Full DCON runs" begin
    ex1 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    @info "Running Solovev ideal example"
    @test isnothing(JPEC.main(ex1))

    ex2 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example_multi_n")
    @info "Running Solovev ideal multi-n example"
    @test isnothing(JPEC.main(ex2))
end