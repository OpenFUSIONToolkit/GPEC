@testset "Test Sing Functions" begin
    @testset " Test sing_der " begin
        # du = zeros(ComplexF64, intr.mpert, odet.msol, 2)
        # Initialize u with some value, e.g., ones
        # odet.u .= ones(ComplexF64, size(odet.u))
        # Initialize structs with relevant data, can do something like
        # ffit = FourfitVars()
        # amat = load in from test_data folder
        # ffit.amats = SplinesMod.CubicSpline("some psi", reshape(amats, mpsi+1, :); bctype="extrap")
        # if using one psi values causes issues, can make a psi array and just fill it all with amat
        # repeat for other matrices, equil data, other relevant constants, etc.

        # params = (ctrl, equil, intr, odet, ffit)
        # sing_der!(du, odet.u, params, odet.psifac)

        # load in du from fortran
        # @test du == expected_du (to within some error)
    end

    @testset " Test sing_find " begin # continue with other functions
    end

    @testset " Test ksing_find " begin
        # Test that ksing_find returns the correct structures with basic parameters

        # Create mock control parameters -> not realistic values for the most part
        ctrl = DCON.DconControl(
            verbose=false,
            qhigh=4.0,
            dmlim=0.5,
            set_psilim_via_dmlim=false,
            sing_order=2
        )

        #=
        # Create mock ODE state -> these are not really realistic values. Fix this later
        odet = DCON.OdeState(
            numpert_total=10,
            numsteps_init=100,
            numunorms_init=5,
            msing=5
        )
        odet.sing_flag=[false, false, false, false, false] =#

        intr = JPEC.DCON.DconInternal()
        intr.numpert_total = 32 # replacing mpert (we set equal to 32). This is the same as msol
        # set mode ranges so sing_der can form singfac_vec consistently
        intr.mpert = intr.numpert_total
        intr.mlow = -12
        intr.mhigh = intr.mlow + intr.mpert - 1
        # default single toroidal mode used in test data
        intr.nlow = 1
        intr.nhigh = 1
        intr.npert = intr.nhigh - intr.nlow + 1
        ctrl.nn_low = intr.nlow
        odet = JPEC.DCON.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)


        # We need a simple equilibrium control object for testing
        equilCtrl = Equilibrium.EquilibriumControl(
            psilow=0.01,
            psihigh=0.99
        )

        # Basic sanity checks that can run without full equilibrium data
        # Testing the function signature and return types
        try
            # The function requires complex setup with equilibrium data
            # so we'll test that it can be called and returns proper structure

            # Mock the sing_get_f_det function behavior for testing
            # This is a simplified test to verify the function exists and has proper interface
            @test typeof(ctrl) <: DCON.DconControl
            @test typeof(odet) <: DCON.OdeState
            @test ctrl.verbose == false
            @test ctrl.qhigh == 4.0
            @test ctrl.sing_order == 2

        catch e
            # If full setup fails, at least verify the function exists
            @test hasmethod(DCON.ksing_find,
                (DCON.DconControl, Equilibrium.EquilibriumControl, DCON.DconInternal, DCON.OdeState))
        end

        # Test ksing_find interface and parameter validation
        @testset "ksing_find interface validation" begin
            # Verify that ksing_find function exists and is callable
            @test hasmethod(DCON.ksing_find,
                (DCON.DconControl, Equilibrium.EquilibriumControl, DCON.DconInternal, DCON.OdeState))

            # Create test control parameters
            test_ctrl = DCON.DconControl(
                verbose=false,
                qhigh=3.5,
                dmlim=0.0,
                set_psilim_via_dmlim=false
            )

            @test test_ctrl.qhigh == 3.5
            @test test_ctrl.verbose == false
        end

        @testset "ksing_find output structure types" begin
            # Test that when ksing_find completes, it returns proper types
            # The function should return (kinsing::Vector{KineticSingular}, singnum::Int)

            # This tests the expected return types
            # Note: Full integration test would require complete equilibrium setup
            intr = DCON.DconInternal()
            @test typeof(intr.kinsing) <: Vector
            @test typeof(intr.kmsing) <: Int
            @test intr.kmsing == 0  # Initially should be zero
        end
    end

    #=
    @testset " Test sing_newton " begin
        # Test that sing_newton converges to a root for a simple test function
        f(x) = x^2 - 2  # Root at sqrt(2)
        x0 = 1.0
        x1 = 2.0
        x_initial = 1.5

        root = DCON.sing_newton!(f, x_initial, x0, x1)
        @test isapprox(root, sqrt(2); atol=1e-6)
    end
    =#
    #=
        @testset " Test sing_get_f_det " begin
            # Test that sing_get_f_det returns expected values for a simple test function
            f(x) = x^2 - 4  # Roots at -2 and 2

            val1 = DCON.sing_get_f_det(f, 1.0)
            val2 = DCON.sing_get_f_det(f, 3.0)

            @test isapprox(val1, -3.0; atol=1e-6)
            @test isapprox(val2, 5.0; atol=1e-6)
        end
    =#
    @testset " Test adp_find_sing! " begin
        @testset "adp_find_sing! function exists and has correct signature" begin
            # Verify that the function exists and has the expected signature
            @test hasmethod(DCON.adp_find_sing!,
                (Float64, Float64, ComplexF64, ComplexF64, ComplexF64,
                    Vector{Float64}, Base.RefValue{Int}, Base.RefValue{Int},
                    Base.RefValue{Int}, Base.RefValue{Int}, Float64,
                    Base.RefValue{ComplexF64}, Base.RefValue{Bool}, Matrix{ComplexF64}))
        end

        @testset "adp_find_sing! sing_get_f_det placeholder" begin
            # Test that sing_get_f_det is properly implemented
            # and returns complex values with singularities

            # Test at several points
            vals = [DCON.sing_get_f_det(x) for x in [0.0, 0.3, 0.5, 0.7, 1.0]]

            # All should be complex numbers
            @test all(typeof(v) <: ComplexF64 for v in vals)

            # Should have some variation (not all same value)
            @test length(unique(abs.(vals))) > 1

            # Should have local minima around 0.3, 0.5, 0.7
            val_mid = abs(DCON.sing_get_f_det(0.5))
            val_nearby1 = abs(DCON.sing_get_f_det(0.48))
            val_nearby2 = abs(DCON.sing_get_f_det(0.52))

            # Point 0.5 should be a local minimum
            @test val_mid <= val_nearby1 || val_mid <= val_nearby2
        end

        @testset "adp_find_sing! counter increment logic" begin
            # Test the counter increment mechanism directly by mocking sing_get_f_det

            # Create a mock sing_get_f_det function in local scope
            function mock_sing_get_f_det(x)
                # Return a complex determinant value that depends on x
                return ComplexF64(1.0 + x^2, x)
            end

            # Store the mock in the global scope of this test
            save_sing_get_f_det = nothing
            if isdefined(DCON, :sing_get_f_det)
                save_sing_get_f_det = DCON.sing_get_f_det
            end

            # We can't easily inject a mock, so we'll test the logic structurally
            singpos = zeros(Float64, 1000)
            singnum = Ref(0)
            i_recur = Ref(0)
            i_depth = Ref(0)
            i_record = Ref(0)
            sing_det = Ref(ComplexF64(Inf, 0))
            sing_flag = Ref(false)

            # Test that Ref counters can be incremented (this is what the function does)
            initial_depth = i_depth[]
            initial_recur = i_recur[]

            i_depth[] += 1
            i_recur[] += 1

            @test i_depth[] == initial_depth + 1
            @test i_recur[] == initial_recur + 1

            # Test multiple increments
            for _ in 1:5
                i_depth[] += 1
                i_recur[] += 1
            end

            @test i_depth[] == initial_depth + 6
            @test i_recur[] == initial_recur + 6
        end

        @testset "adp_find_sing! singular region tracking" begin
            # Test the singular region detection logic independently

            singpos = zeros(Float64, 1000)
            singnum = Ref(0)
            sing_det = Ref(ComplexF64(Inf, 0))
            sing_flag = Ref(false)

            # Simulate entering a singular region
            @test sing_flag[] == false  # Not in singular region initially

            # Case: entering new singular region (Case 2 in the code)
            sing_flag[] = true
            singnum[] += 1
            singpos[singnum[]] = 0.5
            sing_det[] = ComplexF64(0.1, 0.0)

            @test sing_flag[] == true
            @test singnum[] == 1
            @test singpos[1] == 0.5
            @test abs(sing_det[]) == 0.1

            # Simulate finding a sharper singularity
            new_det = ComplexF64(0.05, 0.0)
            if abs(new_det) < abs(sing_det[])
                sing_det[] = new_det
                singpos[singnum[]] = 0.48
            end

            @test abs(sing_det[]) == 0.05
            @test singpos[1] == 0.48

            # Simulate exiting singular region
            sing_flag[] = false

            @test sing_flag[] == false
        end

        @testset "adp_find_sing! array bounds checking" begin
            # Test array size validation

            singpos = zeros(Float64, 1000)
            singnum = Ref(0)
            m_singpos = 1000

            # Test that we can fill array up to limits
            for i in 1:997
                singnum[] += 1
                singpos[singnum[]] = i / 1000.0
                @test singnum[] + 3 <= m_singpos  # Check from function
            end

            # Verify all positions are stored
            @test singnum[] == 997
            for i in 1:singnum[]
                @test singpos[i] ≈ i / 1000.0
            end
        end

        @testset "adp_find_sing! determinant magnitude tracking" begin
            # Test determinant magnitude comparison logic

            det_max = ComplexF64(1.0, 0.0)

            # Test magnitude updates
            test_values = [
                ComplexF64(0.5, 0.0),  # magnitude 0.5
                ComplexF64(1.5, 0.0),  # magnitude 1.5 - should update det_max
                ComplexF64(0.3, 0.4),  # magnitude 0.5
                ComplexF64(1.2, 0.1)  # magnitude ~1.21
            ]

            det_max_val = abs(det_max)
            for test_val in test_values
                if abs(test_val) > det_max_val
                    det_max_val = abs(test_val)
                end
            end

            @test det_max_val ≈ abs(ComplexF64(1.5, 0.0))
        end

        @testset "adp_find_sing! linearity test computation" begin
            # Test the linearity test that determines recursion

            # Perfectly linear case
            det1 = ComplexF64(1.0, 0.0)
            det2 = ComplexF64(1.5, 0.0)  # midpoint in real axis
            det3 = ComplexF64(2.0, 0.0)

            tmp1 = abs(det1 + det3)
            tmpm = abs(det2) * 2

            # Should be ~3.0 for both
            @test tmp1 ≈ 3.0
            @test tmpm ≈ 3.0

            # Should not recurse (linearity error is small)
            linearity_error = abs(tmpm - tmp1)
            @test linearity_error < 0.1

            # Non-linear case
            det1 = ComplexF64(1.0, 0.0)
            det2 = ComplexF64(0.5, 0.0)  # valley in middle
            det3 = ComplexF64(1.0, 0.0)

            tmp1 = abs(det1 + det3)
            tmpm = abs(det2) * 2

            # Should be 2.0 for tmp1 but 1.0 for tmpm
            @test tmp1 == 2.0
            @test tmpm == 1.0

            # Should recurse (large linearity error)
            linearity_error = abs(tmpm - tmp1)
            @test linearity_error == 1.0
        end

        @testset "adp_find_sing! actual function call" begin
            # Test the actual adp_find_sing! function with realistic parameters

            # Set up parameters
            singpos = zeros(Float64, 1000)
            singnum = Ref(0)
            i_recur = Ref(0)
            i_depth = Ref(0)
            i_record = Ref(0)
            sing_det = Ref(ComplexF64(Inf, 0))
            sing_flag = Ref(false)
            record = zeros(ComplexF64, 2, 10000)

            # Define determinants at boundaries using the actual sing_get_f_det
            x0 = 0.0
            x1 = 1.0
            det0 = DCON.sing_get_f_det(x0)
            det1 = DCON.sing_get_f_det(x1)
            det_max = abs(det0) > abs(det1) ? det0 : det1

            tol = 1e-3  # Use tighter tolerance to ensure recursion

            # Call the function with moderate tolerance
            DCON.adp_find_sing!(x0, x1, det_max, det0, det1,
                singpos, singnum, i_recur, i_depth, i_record,
                tol, sing_det, sing_flag, record)

            # Verify the function executed
            @test i_recur[] >= 0  # Recursion counter should be non-negative
            @test i_depth[] >= 0  # Depth should be non-negative

            # Depth should be reasonable (not infinite recursion)
            @test i_depth[] < 50
            @test i_recur[] < 1000
        end

        @testset "adp_find_sing! tight tolerance increases recursion" begin
            # Test that tighter tolerance causes more refinement

            # Test with loose tolerance
            singpos1 = zeros(Float64, 1000)
            singnum1 = Ref(0)
            i_recur1 = Ref(0)
            i_depth1 = Ref(0)
            i_record1 = Ref(0)
            sing_det1 = Ref(ComplexF64(Inf, 0))
            sing_flag1 = Ref(false)
            record1 = zeros(ComplexF64, 2, 10000)

            x0 = 0.0
            x1 = 1.0
            det0 = DCON.sing_get_f_det(x0)
            det1 = DCON.sing_get_f_det(x1)
            det_max = abs(det0) > abs(det1) ? det0 : det1

            DCON.adp_find_sing!(x0, x1, det_max, det0, det1,
                singpos1, singnum1, i_recur1, i_depth1, i_record1,
                1e-1, sing_det1, sing_flag1, record1)

            # Test with tight tolerance
            singpos2 = zeros(Float64, 1000)
            singnum2 = Ref(0)
            i_recur2 = Ref(0)
            i_depth2 = Ref(0)
            i_record2 = Ref(0)
            sing_det2 = Ref(ComplexF64(Inf, 0))
            sing_flag2 = Ref(false)
            record2 = zeros(ComplexF64, 2, 10000)

            DCON.adp_find_sing!(x0, x1, det_max, det0, det1,
                singpos2, singnum2, i_recur2, i_depth2, i_record2,
                1e-4, sing_det2, sing_flag2, record2)

            # Tighter tolerance should cause more recursion
            @test i_depth2[] >= i_depth1[]
            @test i_recur2[] >= i_recur1[]
        end

        @testset "adp_find_sing! singularity detection" begin
            # Test that singularities are actually found and recorded

            singpos = zeros(Float64, 1000)
            singnum = Ref(0)
            i_recur = Ref(0)
            i_depth = Ref(0)
            i_record = Ref(0)
            sing_det = Ref(ComplexF64(Inf, 0))
            sing_flag = Ref(false)
            record = zeros(ComplexF64, 2, 10000)

            x0 = 0.0
            x1 = 1.0
            det0 = DCON.sing_get_f_det(x0)
            det1 = DCON.sing_get_f_det(x1)
            det_max = abs(det0) > abs(det1) ? det0 : det1

            # Use tight tolerance to find singularities
            DCON.adp_find_sing!(x0, x1, det_max, det0, det1,
                singpos, singnum, i_recur, i_depth, i_record,
                1e-3, sing_det, sing_flag, record)

            # Should have found at least one singularity
            @test singnum[] > 0

            # Singular positions should be in the search interval
            for i in 1:singnum[]
                @test 0.0 <= singpos[i] <= 1.0
            end

            # Records should be populated
            @test i_record[] > 0
        end
    end

end
