# Inner-layer model capability gates and option forwarding.
@testset "Matching models" begin
    GPEC = GeneralizedPerturbedEquilibrium
    FFS = GPEC.ForceFreeStates

    @testset "model capability gates and option forwarding" begin
        IL = GPEC.InnerLayer
        @test IL.GGJModel() isa IL.GGJModel{:ray}
        @test FFS.closure_capable(IL.GGJModel())
        @test !FFS.closure_capable(IL.SLAYERModel())
        @test !FFS.closure_capable(IL.GGJModel(; solver=:shooting))    # no layer profiles
        @test IL.GGJModel(; solver=:galerkin, nx=640).options == (nx=640,)
        # Carried options reach the backend exactly as call-site keywords do.
        p = IL.glasser_wang_2020_eq55()
        γ = 1e-3im
        @test IL.solve_inner(IL.GGJModel(; solver=:galerkin, nx=256), p, γ) == IL.solve_inner(IL.GGJModel(; solver=:galerkin), p, γ; nx=256)
        @test IL.solve_inner(IL.GGJModel(; solver=:galerkin, nx=256), p, γ; nx=512) == IL.solve_inner(IL.GGJModel(; solver=:galerkin), p, γ)
    end
end
