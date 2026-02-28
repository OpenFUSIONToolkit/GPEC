using LinearAlgebra

# TODO: perhaps this isn't the best place for this function?
# Should I do include("../ForceFreeStates/utils.jl") instead? or maybe save these functions in a separate file?
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
    @testset "trim_storage!" begin
        # Test that trim_storage! resizes arrays to actual step count
        mpert = 3
        numsteps_init = 20
        odet = JPEC.ForceFreeStates.OdeState(mpert, numsteps_init, 10, 5)

        # Set step to less than initial size
        odet.step = 12
        for i in 1:odet.step
            odet.psi_store[i] = Float64(i)
            odet.u_store[:, :, i, :] .= ComplexF64(i)
            odet.ud_store[:, :, i, :] .= ComplexF64(i + 0.5)
        end

        # Trim storage
        JPEC.ForceFreeStates.trim_storage!(odet)

        # Check sizes match step count
        @test length(odet.psi_store) == odet.step
        @test size(odet.u_store, 3) == odet.step
        @test size(odet.ud_store, 3) == odet.step

        # Check all data is preserved
        @test all(odet.psi_store .== Float64.(1:odet.step))
    end

    @testset "transform_u!" begin
        # Test transformation of solution vectors
        mpert = 2
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=mpert, numpert_total=mpert)
        odet = JPEC.ForceFreeStates.OdeState(mpert, 10, 5, 2)

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
            odet.u_store[:, :, i, 1] .= ComplexF64(i)
            odet.u_store[:, :, i, 2] .= ComplexF64(i + 0.1)
            odet.ud_store[:, :, i, 1] .= ComplexF64(i + 0.2)
            odet.ud_store[:, :, i, 2] .= ComplexF64(i + 0.3)
        end

        u_orig = copy(odet.u_store)

        # Apply transformation
        JPEC.ForceFreeStates.transform_u!(odet, intr)

        # Check that u_store was modified (transformation applied)
        @test !all(odet.u_store .== u_orig)

        # The transformation should preserve the structure but apply the fixfac matrices
        # transform_u! doesn't resize arrays - it only applies transformations in-place
        # The storage arrays retain their original allocated size
        @test size(odet.u_store) == size(u_orig)
    end

    @testset "apply_gaussian_reduction!" begin
        # Initialize to random u. ifix starts at 0; the function increments it before use.
        mpert = 5
        odet = JPEC.ForceFreeStates.OdeState(mpert, 10, 10, 10)
        odet.u = randn(ComplexF64, mpert, mpert, 2)
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; numpert_total=mpert)

        # Save copy of original u and run
        u_orig = copy(odet.u)
        JPEC.ForceFreeStates.apply_gaussian_reduction!(odet.u, odet, intr, false)

        # Very simple tests
        @test !all(odet.u .== u_orig)  # u should have changed
        @test all(abs.(diag(odet.fixfac[:, :, 1])) .≈ 1)  # diagonal of fixfac = 1
        @test odet.fixstep[1] == odet.step - 1 # fixstep should be set
        @test odet.sing_flag[1] == false # sing_flag should match input

        # --- Real Fortran data check ---
        mpert = 31
        odet = JPEC.ForceFreeStates.OdeState(mpert, 10, 10, 10)
        # Load Fortran data for u pulled before and after a fixup.
        # The Fortran sort was equivalent to sorting by current norms (unorm was set
        # to norm.(eachcol(u[:,i,1])) immediately before apply_gaussian_reduction! was called).
        odet.u = load_u_matrix(joinpath(@__DIR__, "test_data", "u_prefixup.dat"))
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; numpert_total=mpert)

        JPEC.ForceFreeStates.apply_gaussian_reduction!(odet.u, odet, intr, false)

        u_fortran = load_u_matrix(joinpath(@__DIR__, "test_data", "u_postfixup.dat"))
        # test that the outputs are approximately equivalent (1e-3 seems ok to account for loading differences)
        @test all(abs.(odet.u .- u_fortran) .< 1e-3)

        # Test with a simple 2x2 case where we can predict the result
        mpert = 2
        odet = JPEC.ForceFreeStates.OdeState(mpert, 10, 10, 10)

        # Set up a simple u matrix where first column has larger norm
        # u[:, 1, 1] = [3, 4] (norm = 5)
        # u[:, 2, 1] = [1, 0] (norm = 1)
        odet.u[:, 1, 1] .= [3.0 + 0.0im, 4.0 + 0.0im]
        odet.u[:, 2, 1] .= [1.0 + 0.0im, 0.0 + 0.0im]
        odet.u[:, :, 2] .= 0.0  # Set second equation to zero for simplicity

        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; numpert_total=mpert)

        u_before = copy(odet.u)

        JPEC.ForceFreeStates.apply_gaussian_reduction!(odet.u, odet, intr, false)

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

    @testset "OdeState construction" begin
        # Test basic OdeState initialization
        numpert_total = 5
        numsteps_init = 100
        numunorms_init = 20
        msing = 10

        odet = JPEC.ForceFreeStates.OdeState(numpert_total, numsteps_init, numunorms_init, msing)

        # Check fields are initialized correctly
        @test odet.numpert_total == numpert_total
        @test odet.numsteps_init == numsteps_init
        @test odet.numunorms_init == numunorms_init
        @test odet.msing == msing
        @test odet.step == 1
        @test odet.solver_steps == 0
        @test odet.ifix == 0
        @test odet.nzero == 0

        # Check array dimensions
        @test size(odet.u) == (numpert_total, numpert_total, 2)
        @test size(odet.ud) == (numpert_total, numpert_total, 2)
        @test size(odet.u_store) == (numpert_total, numpert_total, numsteps_init, 2)
        @test size(odet.ud_store) == (numpert_total, numpert_total, numsteps_init, 2)
        @test length(odet.psi_store) == numsteps_init
        @test size(odet.ca_r) == (numpert_total, numpert_total, 2, msing)
        @test size(odet.ca_l) == (numpert_total, numpert_total, 2, msing)
        @test size(odet.fixfac) == (numpert_total, numpert_total, numunorms_init)
        @test length(odet.unorm0) == numpert_total
        @test all(odet.unorm0 .== 1.0)
    end

    @testset "chunk_el_integration_bounds tests" begin
        ctrl = JPEC.ForceFreeStates.ForceFreeStatesControl()
        ctrl.numunorms_init = 5
        ctrl.n_subchunks_per_region = 3  # N=3: each region gets edge-weighted sub-chunks [5%, 90%, 5%]

        # Case 1: No singular surfaces -> N chunks to edge (all non-crossing)
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=1, numpert_total=1)
        intr.msing = 0
        intr.psilim = 1.0
        ctrl.singfac_min = 1e-4
        chunks = JPEC.ForceFreeStates.chunk_el_integration_bounds(0.0, 0, ctrl, intr)
        @test length(chunks) == 3         # N sub-chunks for the single final region
        @test all(c.needs_crossing == false for c in chunks)

        # Case 2: One singular surface within limits -> N pre-crossing sub-chunks + N final sub-chunks
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=1, numpert_total=1)
        s = JPEC.ForceFreeStates.SingType()
        s.psifac = 0.5
        s.n = [1]
        s.m = [1]
        s.q1 = 2.0
        intr.sing = [s]
        intr.msing = 1
        intr.psilim = 1.0
        intr.mlow = 1
        intr.mhigh = 1
        ctrl.singfac_min = 1e-4
        chunks = JPEC.ForceFreeStates.chunk_el_integration_bounds(0.0, 0, ctrl, intr)
        @test length(chunks) == 6         # N sub-chunks per region × 2 regions
        @test chunks[3].needs_crossing == true  # last sub-chunk of first region has crossing
        @test all(c.needs_crossing == false for c in [chunks[1], chunks[2], chunks[4], chunks[5], chunks[6]])
        @test chunks[3].psi_end < intr.sing[1].psifac

        # Case 3: Multiple singular surfaces -> N sub-chunks per region × (n_crossings + 1) regions
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=1, numpert_total=1)
        s1 = JPEC.ForceFreeStates.SingType(; psifac=0.3, n=[1], m=[1], q1=1.5)
        s2 = JPEC.ForceFreeStates.SingType(; psifac=0.6, n=[1], m=[1], q1=2.5)
        intr.sing = [s1, s2]
        intr.msing = 2
        intr.psilim = 1.0
        intr.mlow = 1
        intr.mhigh = 1
        ctrl.singfac_min = 1e-6
        chunks = JPEC.ForceFreeStates.chunk_el_integration_bounds(0.0, 0, ctrl, intr)
        @test length(chunks) == 9         # N sub-chunks per region × 3 regions
        @test chunks[3].needs_crossing == true   # last sub-chunk of region 1
        @test chunks[6].needs_crossing == true   # last sub-chunk of region 2
        @test all(c.needs_crossing == false for c in chunks[[1,2,4,5,7,8,9]])

        # Case 4: singfac_min == 0 should disable crossing logic -> N chunks (all non-crossing)
        intr = JPEC.ForceFreeStates.ForceFreeStatesInternal(; mpert=1, numpert_total=1)
        intr.sing = [JPEC.ForceFreeStates.SingType(; psifac=0.4, n=[1], m=[1], q1=2.0)]
        intr.msing = 1
        intr.psilim = 1.0
        ctrl.singfac_min = 0.0
        chunks = JPEC.ForceFreeStates.chunk_el_integration_bounds(0.0, 0, ctrl, intr)
        @test length(chunks) == 3         # N sub-chunks for the single final region (singfac_min=0 bypasses crossing)
        @test all(c.needs_crossing == false for c in chunks)
    end
end
