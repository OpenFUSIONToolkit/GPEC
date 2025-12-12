@testset "DCON Struct Type Stability" begin
    @info "Testing type-stable DCON struct initialization"
    
    # Test DCON struct type stability
    @testset "DCON Struct Type Stability" begin
        # Test SingType initialization
        st = JPEC.DCON.SingType()
        @test typeof(st.power) == Vector{ComplexF64}
        @test typeof(st.vmat) == Array{ComplexF64, 4}
        @test typeof(st.mmat) == Array{ComplexF64, 4}
        @test length(st.power) == 0
        @test size(st.vmat) == (0, 0, 0, 0)
        @test size(st.mmat) == (0, 0, 0, 0)
        # Verify no Union types
        @test fieldtype(typeof(st), :power) == Vector{ComplexF64}
        @test fieldtype(typeof(st), :vmat) == Array{ComplexF64, 4}
        @test fieldtype(typeof(st), :mmat) == Array{ComplexF64, 4}
        
        # Test DconInternal initialization
        di = JPEC.DCON.DconInternal()
        @test typeof(di.locstab) == JPEC.Spl.CubicSpline{Float64}
        @test di.locstab.handle == C_NULL
        @test typeof(di.sing) == Vector{JPEC.DCON.SingType}
        @test typeof(di.kinsing) == Vector{JPEC.DCON.SingType}
        # Verify no Union types
        @test fieldtype(typeof(di), :locstab) == JPEC.Spl.CubicSpline{Float64}
        @test fieldtype(typeof(di), :sing) == Vector{JPEC.DCON.SingType}
        @test fieldtype(typeof(di), :kinsing) == Vector{JPEC.DCON.SingType}
        
        # Test FourFitVars initialization
        ffv = JPEC.DCON.FourFitVars(mpert=5, mband=4)
        @test typeof(ffv.amats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.bmats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.cmats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.dmats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.emats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.hmats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.fmats_lower) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.kmats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test typeof(ffv.gmats) == JPEC.Spl.CubicSpline{ComplexF64}
        # Verify all have C_NULL handles
        @test ffv.amats.handle == C_NULL
        @test ffv.bmats.handle == C_NULL
        @test ffv.cmats.handle == C_NULL
        @test ffv.dmats.handle == C_NULL
        @test ffv.emats.handle == C_NULL
        @test ffv.hmats.handle == C_NULL
        @test ffv.fmats_lower.handle == C_NULL
        @test ffv.kmats.handle == C_NULL
        @test ffv.gmats.handle == C_NULL
        # Verify no Union types
        @test fieldtype(typeof(ffv), :amats) == JPEC.Spl.CubicSpline{ComplexF64}
        @test fieldtype(typeof(ffv), :bmats) == JPEC.Spl.CubicSpline{ComplexF64}
        
        # Test OdeState initialization
        os = JPEC.DCON.OdeState(numpert_total=10, numsteps_init=100, numunorms_init=20, msing=2)
        @test typeof(os.wvmat_spline) == JPEC.Spl.CubicSpline{ComplexF64}
        @test os.wvmat_spline.handle == C_NULL
        # Verify no Union types
        @test fieldtype(typeof(os), :wvmat_spline) == JPEC.Spl.CubicSpline{ComplexF64}
    end
end
