using LinearAlgebra

# TODO: perhaps this isn't the best place for this function?
# Should I do include("../dcon/utils.jl") instead? or maybe save these functions in a separate file?
# associated TODO: come up with Gaussian reduction test that doesn't rely on external data

function load_u_matrix(filename)
    lines = readlines(filename)
    data = [parse.(Float64, split(l)) for l in lines[2:end]]
    i_vals = Int.(getindex.(data, 1))
    j_vals = Int.(getindex.(data, 2))
    ncols = (length(data[1]) - 2) ÷ 2
    imax = maximum(i_vals)
    jmax = maximum(j_vals)
    mat = zeros(ComplexF64, imax, jmax, ncols)
    for row in data
        i = Int(row[1])
        j = Int(row[2])
        for k in 1:ncols
            re = Float64(row[2*k+1])
            im = Float64(row[2*k+2])
            mat[i, j, k] = complex(re, im)
        end
    end
    return mat
end

@testset "ODE Tests" begin
    @testset "resize_storage!" begin
        # Test that resize_storage! doubles the size of storage arrays
        mpert = 3
        numsteps_init = 10
        odet = JPEC.DCON.OdeState(mpert, numsteps_init, 10, 5)
        
        # Fill some data
        odet.step = 8
        for i in 1:odet.step
            odet.psi_store[i] = Float64(i)
            odet.q_store[i] = Float64(i * 2)
            odet.u_store[:, :, :, i] .= ComplexF64(i)
            odet.ud_store[:, :, :, i] .= ComplexF64(i + 0.5)
        end
        
        # Resize storage
        JPEC.DCON.resize_storage!(odet)
        
        # Check new size is doubled
        @test length(odet.psi_store) == 2 * numsteps_init
        @test length(odet.q_store) == 2 * numsteps_init
        @test size(odet.u_store, 4) == 2 * numsteps_init
        @test size(odet.ud_store, 4) == 2 * numsteps_init
        
        # Check data is preserved
        @test all(odet.psi_store[1:odet.step] .== Float64.(1:odet.step))
        @test all(odet.q_store[1:odet.step] .== Float64.(2:2:2*odet.step))
        for i in 1:odet.step
            @test all(odet.u_store[:, :, :, i] .== ComplexF64(i))
            @test all(odet.ud_store[:, :, :, i] .== ComplexF64(i + 0.5))
        end

        # Check that you can resize again
        JPEC.DCON.resize_storage!(odet)
        @test length(odet.psi_store) == 4 * numsteps_init
        @test length(odet.q_store) == 4 * numsteps_init
        @test size(odet.u_store, 4) == 4 * numsteps_init
        @test size(odet.ud_store, 4) == 4 * numsteps_init
    end

    @testset "trim_storage!" begin
        # Test that trim_storage! resizes arrays to actual step count
        mpert = 3
        numsteps_init = 20
        odet = JPEC.DCON.OdeState(mpert, numsteps_init, 10, 5)
        
        # Set step to less than initial size
        odet.step = 12
        for i in 1:odet.step
            odet.psi_store[i] = Float64(i)
            odet.q_store[i] = Float64(i * 2)
            odet.u_store[:, :, :, i] .= ComplexF64(i)
            odet.ud_store[:, :, :, i] .= ComplexF64(i + 0.5)
        end
        
        # Trim storage
        JPEC.DCON.trim_storage!(odet)
        
        # Check sizes match step count
        @test length(odet.psi_store) == odet.step
        @test length(odet.q_store) == odet.step
        @test size(odet.u_store, 4) == odet.step
        @test size(odet.ud_store, 4) == odet.step
        
        # Check all data is preserved
        @test all(odet.psi_store .== Float64.(1:odet.step))
        @test all(odet.q_store .== Float64.(2:2:2*odet.step))
    end

    @testset "compute_tols" begin
        # Test tolerance computation
        mpert = 3
        ctrl = JPEC.DCON.DconControl()
        ctrl.tol_r = 1e-6
        ctrl.tol_nr = 1e-4
        ctrl.crossover = 0.01
        
        intr = JPEC.DCON.DconInternal(; mpert=mpert)
        intr.msing = 2
        intr.sing = [JPEC.DCON.SingType(), JPEC.DCON.SingType()]
        intr.sing[1].q = 2.0
        intr.sing[1].n = [1]
        intr.sing[2].q = 3.0
        intr.sing[2].n = [1]
        
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 2)
        
        # Test 1: Far from singular surface (singfac > crossover)
        odet.ising = 1
        odet.q = 1.5  # Far from q=2.0
        rtol, atol = JPEC.DCON.compute_tols(ctrl, intr, odet)
        @test rtol == ctrl.tol_nr  # Should use non-resonant tolerance

        # Test 2: Close to singular surface (singfac < crossover)
        odet.ising = 1
        odet.q = 1.999  # Very close to q=2.0
        rtol, atol = JPEC.DCON.compute_tols(ctrl, intr, odet)
        @test rtol == ctrl.tol_r  # Should use resonant tolerance

        # Test 3: Between two singular surfaces
        odet.ising = 2
        odet.q = 2.5  # Between q=2.0 and q=3.0
        rtol, atol = JPEC.DCON.compute_tols(ctrl, intr, odet)
        @test rtol == ctrl.tol_nr  # Should use min distance to either surface

        # Test 4: Beyond all singular surfaces
        odet.ising = 3
        odet.q = 4.0
        rtol, atol = JPEC.DCON.compute_tols(ctrl, intr, odet)
        @test rtol == ctrl.tol_nr

        # Edge case - no singular surfaces
        mpert = 2
        ctrl = JPEC.DCON.DconControl()
        ctrl.tol_r = 1e-6
        ctrl.tol_nr = 1e-4
        ctrl.crossover = 0.01
        
        intr = JPEC.DCON.DconInternal(; mpert=mpert)
        intr.msing = 0
        intr.sing = []
        
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 0)
        odet.ising = 1
        odet.q = 2.0

        # Should return non-resonant tolerance when no singular surfaces
        rtol, atol = JPEC.DCON.compute_tols(ctrl, intr, odet)
        @test rtol == ctrl.tol_nr
    end

    @testset "transform_u!" begin
        # Test transformation of solution vectors
        mpert = 2
        intr = JPEC.DCON.DconInternal(; mpert=mpert, numpert_total=mpert)
        odet = JPEC.DCON.OdeState(mpert, 10, 5, 2)
        
        # Set up a simple fixup scenario
        odet.ifix = 1
        odet.step = 5
        odet.sing_flag[1] = false
        odet.fixstep[1] = 3
        odet.zeroed_idx[1] = Int[]
        
        # Initialize fixfac with some transformation
        odet.fixfac[1, 1, 1] = 1.0
        odet.fixfac[1, 2, 1] = 0.5
        odet.fixfac[2, 1, 1] = 0.0
        odet.fixfac[2, 2, 1] = 1.0
        
        # Initialize index (sorted by unorm)
        odet.index[:, 1] = [1, 2]
        
        # Set up some u_store and ud_store data
        for i in 1:odet.step
            odet.u_store[:, :, 1, i] .= ComplexF64(i)
            odet.u_store[:, :, 2, i] .= ComplexF64(i + 0.1)
            odet.ud_store[:, :, 1, i] .= ComplexF64(i + 0.2)
            odet.ud_store[:, :, 2, i] .= ComplexF64(i + 0.3)
        end
        
        u_orig = copy(odet.u_store)
        
        # Apply transformation
        JPEC.DCON.transform_u!(odet, intr)
        
        # Check that u_store was modified (transformation applied)
        @test !all(odet.u_store .== u_orig)
        
        # The transformation should preserve the structure but apply the fixfac matrices
        # transform_u! doesn't resize arrays - it only applies transformations in-place
        # The storage arrays retain their original allocated size
        @test size(odet.u_store) == size(u_orig)
    end

    @testset "ode_fixup!" begin
        # Initialize to random u
        mpert = 5
        ifix = 1
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 10)
        odet.u = randn(ComplexF64, mpert, mpert, 2)
        odet.unorm = [norm(odet.u[:, i, 1]) for i in 1:mpert]
        odet.ifix = ifix
        odet.fixfac = zeros(ComplexF64, mpert, mpert, ifix)
        intr = JPEC.DCON.DconInternal(; numpert_total=mpert)

        # Save copy of original u and run
        u_orig = copy(odet.u)
        JPEC.DCON.ode_fixup!(odet.u, odet, intr, false)

        # Very simple tests
        @test !all(odet.u .== u_orig)  # u should have changed
        @test all(abs.(diag(odet.fixfac[:, :, ifix])) .≈ 1)  # diagonal of fixfac = 1
        @test odet.fixstep[1] == odet.step - 1 # fixstep should be set
        @test odet.sing_flag[1] == false # sing_flag should match input

        # --- Real Fortran data check ---
        mpert = 31
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 10)
        # We'll load in Fortran data for u pulled before and after a fixup
        # Note that this was generated by manually setting
        # unorm = [norm(odet.u[:,i,1]) for i in 1:msol] in the Fortran to avoid
        # also having to save unorm0
        odet.u = load_u_matrix(joinpath(@__DIR__, "test_data", "u_prefixup.dat"))
        odet.unorm = [norm(odet.u[:, i, 1]) for i in 1:mpert]
        odet.ifix = ifix
        odet.fixfac = zeros(ComplexF64, mpert, mpert, ifix)
        intr = JPEC.DCON.DconInternal(; numpert_total=mpert)

        JPEC.DCON.ode_fixup!(odet.u, odet, intr, false)

        u_fortran = load_u_matrix(joinpath(@__DIR__, "test_data", "u_postfixup.dat"))
        # test that the outputs are approximately equivalent (1e-3 seems ok to account for loading differences)
        @test all(abs.(odet.u .- u_fortran) .< 1e-3)

        # Test with a simple 2x2 case where we can predict the result
        mpert = 2
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 10)
        
        # Set up a simple u matrix where first column has larger norm
        # u[:, 1, 1] = [3, 4] (norm = 5)
        # u[:, 2, 1] = [1, 0] (norm = 1)
        odet.u[:, 1, 1] .= [3.0 + 0.0im, 4.0 + 0.0im]
        odet.u[:, 2, 1] .= [1.0 + 0.0im, 0.0 + 0.0im]
        odet.u[:, :, 2] .= 0.0  # Set second equation to zero for simplicity
        
        odet.unorm = [norm(odet.u[:, i, 1]) for i in 1:mpert]
        odet.ifix = 1
        odet.fixfac = zeros(ComplexF64, mpert, mpert, 1)
        intr = JPEC.DCON.DconInternal(; numpert_total=mpert)
        
        u_before = copy(odet.u)
        
        JPEC.DCON.ode_fixup!(odet.u, odet, intr, false)
        
        # After fixup:
        # - index should sort by norm: [1, 2] (largest first)
        @test odet.index[:, 1] == [1, 2]
        
        # - The largest element in the first column should be used as pivot
        # - The second column should be modified to eliminate that element
        # - fixfac should capture the elimination factor
        @test odet.fixfac[1, 1, 1] == 1.0  # Diagonal
        
        # The pivot element should not change
        pivot_idx = argmax(abs.(u_before[:, 1, 1]))
        @test abs(odet.u[pivot_idx, 1, 1] - u_before[pivot_idx, 1, 1]) < 1e-10
    end

    @testset "ode_unorm!" begin
        mpert = 2
        odet = JPEC.DCON.OdeState(mpert, 10, 10, 10)
        intr = JPEC.DCON.DconInternal(; mpert=mpert)
        ctrl = JPEC.DCON.DconControl()
        ctrl.ucrit = 10.0

        # Case 1: Basic norm computation
        odet.u = zeros(ComplexF64, 2, 2, 2)
        odet.u[:, 1, 1] .= [3, 4]          # norm = 5
        odet.u[:, 2, 1] .= [0, 2]          # norm = 2

        JPEC.DCON.ode_unorm!(odet.u, odet, ctrl, intr, false)
        # After the first run with new=True (default), unorm0 should be set to unorm
        # and new should be false
        @test odet.unorm[1:intr.mpert] ≈ [5, 2]
        @test odet.unorm0 == odet.unorm
        @test odet.new == false

        # Case 2: Error on zero norm
        odet.u[:, 1, 1] .= 0
        odet.new = true
        @test_throws ErrorException JPEC.DCON.ode_unorm!(odet.u, odet, ctrl, intr, false)

        # Case 3: Normalization on second call
        odet.u[:, 1, 1] .= [3, 4]   # norm = 5
        odet.u[:, 2, 1] .= [0, 2]   # norm = 2
        odet.new = false
        JPEC.DCON.ode_unorm!(odet.u, odet, ctrl, intr, false)
        @test odet.unorm[1:intr.mpert] ≈ [1, 1]

        # Case 4: Trigger fixup via ucrit
        odet.unorm0 = ones(intr.mpert)
        odet.u[:, 1, 1] .= [1000, 0]   # large norm
        odet.u[:, 2, 1] .= [1, 0]      # small norm
        JPEC.DCON.ode_unorm!(odet.u, odet, ctrl, intr, false)
        @test odet.new == true  # implies fixup ran

        # Case 5: Trigger fixup via sing_flag
        odet.new = false
        odet.u[:, 1, 1] .= [1, 0]
        odet.u[:, 2, 1] .= [1, 0]
        JPEC.DCON.ode_unorm!(odet.u, odet, ctrl, intr, true)
        @test odet.new == true  # fixup triggered
    end

    @testset "OdeState construction" begin
        # Test basic OdeState initialization
        numpert_total = 5
        numsteps_init = 100
        numunorms_init = 20
        msing = 10
        
        odet = JPEC.DCON.OdeState(numpert_total, numsteps_init, numunorms_init, msing)
        
        # Check fields are initialized correctly
        @test odet.numpert_total == numpert_total
        @test odet.numsteps_init == numsteps_init
        @test odet.numunorms_init == numunorms_init
        @test odet.msing == msing
        @test odet.step == 1
        @test odet.new == true
        @test odet.ifix == 0
        @test odet.nzero == 0
        
        # Check array dimensions
        @test size(odet.u) == (numpert_total, numpert_total, 2)
        @test size(odet.ud) == (numpert_total, numpert_total, 2)
        @test size(odet.u_store) == (numpert_total, numpert_total, 2, numsteps_init)
        @test size(odet.ud_store) == (numpert_total, numpert_total, 2, numsteps_init)
        @test length(odet.psi_store) == numsteps_init
        @test length(odet.q_store) == numsteps_init
        @test size(odet.ca_r) == (numpert_total, numpert_total, 2, msing)
        @test size(odet.ca_l) == (numpert_total, numpert_total, 2, msing)
        @test size(odet.fixfac) == (numpert_total, numpert_total, numunorms_init)
        @test length(odet.unorm) == numpert_total
        @test length(odet.unorm0) == numpert_total
    end
end