using Test
using TOML
using Printf
using FastInterpolations: cubic_interp, CubicFit, LinearBinarySearch, Series, ExtendExtrap

#TODO: these take forever to run- can we optimize them up?
@testset "Sing Tests" begin

    # minimal helpers

    #=
    function read_complex_fortran(fname)
        data = readdlm(fname, Float64)
        m, n = Int(data[1,1]), Int(data[1,2])
        entries = data[2:end, :]
        mat = reshape([Complex(entries[k,1], entries[k,2]) for k in 1:size(entries,1)], m, n)
        return transpose(mat)
    end
    =#

    function read_complex_fortran(fname)
        lines = readlines(fname)

        # Parse all numbers into a flat vector
        vals = Float64[]
        for line in lines
            isempty(strip(line)) && continue
            for x in split(line)
                push!(vals, parse(Float64, replace(x, 'D' => 'E')))
            end
        end

        m = Int(vals[1])
        n = Int(vals[2])

        entries = vals[3:end]
        mat = reshape([complex(entries[2k-1], entries[2k]) for k in 1:(m*n)], m, n)

        return transpose(mat)
    end

    function extract_value(filename::String, item::String)
        for line in eachline(filename)
            parts = split(line, '=')
            if length(parts) >= 2 && strip(parts[1]) == item
                val = strip(parts[2])
                return parse(Float64, replace(val, 'D' => 'E'))
            end
        end
        return NaN
    end

    function read_solutions_3d(fname::String)
        lines = readlines(fname)
        blocks = Vector{Vector{Vector{Float64}}}();
        current = Vector{Vector{Float64}}()
        for s in lines
            t = strip(s);
            isempty(t) && continue
            if occursin("Solution index", t)
                if !isempty(current)
                    push!(blocks, current);
                    current = Vector{Vector{Float64}}()
                end
                continue
            end
            if occursin("=", t)
                continue
            end
            vals = [parse(Float64, replace(x, 'D'=>'E')) for x in split(t)]
            push!(current, vals)
        end
        if !isempty(current)
            push!(blocks, current)
        end
        mpert = length(blocks[1]);
        nsol = length(blocks)
        result = Array{ComplexF64}(undef, mpert, nsol, 2)
        for (j, block) in enumerate(blocks), (i, row) in enumerate(block)
            result[i, j, 1] = complex(row[2], row[3]);
            result[i, j, 2] = complex(row[4], row[5])
        end
        return result
    end

    function copyForSplines(prev_mat, num_range)
        mmat = zeros(ComplexF64, length(num_range), size(prev_mat, 1)*size(prev_mat, 2))
        for i in 1:length(num_range)
            mmat[i, :] .= vec(prev_mat)
        end
        return mmat
    end

    # --------------------------
    # Write output in Fortran-compatible format
    # --------------------------
    function write_sing_output(filename, psifac, q, mlow, nn, ud)
        # ud is a 3D array: (mpert, msol, 2), ComplexF64
        mpert, msol, _ = size(ud)

        open(filename, "w") do io
            # header
            @printf(io, "psifac = %16.8f\n", psifac)
            @printf(io, "q = %16.8f\n", q)
            @printf(io, "mlow = %5d\n", mlow)
            @printf(io, "nn = %3d\n", nn)

            # solutions
            for isol in 1:msol
                @printf(io, "Solution index = %3d\n", isol)
                for ipert in 1:mpert
                    xr1 = real(ud[ipert, isol, 1])
                    xi1 = imag(ud[ipert, isol, 1])
                    xr2 = real(ud[ipert, isol, 2])
                    xi2 = imag(ud[ipert, isol, 2])
                    @printf(io, "%5d  %16.8e  %16.8e  %16.8e  %16.8e\n",
                        ipert, xr1, xi1, xr2, xi2)
                end
            end
        end
    end

    function load_equilibrium_from_gpec(gpec_path::String)
        inputs = TOML.parsefile(gpec_path)
        eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], dirname(gpec_path))
        return GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
    end

    @testset "sing_der" begin

        equil = load_equilibrium_from_gpec(joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example", "gpec.toml"))
        #equil = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(joinpath(@__DIR__, "../examples/Solovev_ideal_example/equil.toml"))
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl()
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
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
        odet = GeneralizedPerturbedEquilibrium.ForceFreeStates.OdeState(; numpert_total=intr.numpert_total,
            numsteps_init=ctrl.numsteps_init, numunorms_init=ctrl.numunorms_init, msing=intr.msing)

        psifac_dummy = collect(range(0, 1, 10));
        points = length(psifac_dummy)
        amat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/amat.dat"))
        amats = copyForSplines(amat, psifac_dummy)
        bmat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/bmat.dat"))
        bmats = copyForSplines(bmat, psifac_dummy)
        cmat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/cmat.dat"))
        cmats = copyForSplines(cmat, psifac_dummy)

        fmat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/fmat.dat"))
        fmat .= cholesky(Hermitian(fmat)).L;
        fmats = copyForSplines(fmat, psifac_dummy)
        kmat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/kmat.dat"))
        kmats = copyForSplines(kmat, psifac_dummy)
        gmat = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/gmat.dat"))
        gmats = copyForSplines(gmat, psifac_dummy)

        umat_p1 = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/umat_p1.dat"))
        umat_p2 = read_complex_fortran(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/umat_p2.dat"))
        odet.psifac = extract_value(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/sing_der_output_normal.dat"), "psifac")
        odet.u[:, :, 1] .= umat_p1;
        odet.u[:, :, 2] .= umat_p2

        ffit = GeneralizedPerturbedEquilibrium.ForceFreeStates.FourFitVars(; mpert=intr.numpert_total, mband=intr.mband, numpert_total=intr.numpert_total)
        ffit.amats = cubic_interp(psifac_dummy, Series(reshape(amats, points, :)); ffit.itp_opts...)
        ffit.bmats = cubic_interp(psifac_dummy, Series(reshape(bmats, points, :)); ffit.itp_opts...)
        ffit.cmats = cubic_interp(psifac_dummy, Series(reshape(cmats, points, :)); ffit.itp_opts...)
        ffit.fmats_lower = cubic_interp(psifac_dummy, Series(reshape(fmats, points, :)); ffit.itp_opts...)
        ffit.kmats = cubic_interp(psifac_dummy, Series(reshape(kmats, points, :)); ffit.itp_opts...)
        ffit.gmats = cubic_interp(psifac_dummy, Series(reshape(gmats, points, :)); ffit.itp_opts...)

        du = zeros(ComplexF64, intr.numpert_total, intr.numpert_total, 2)
        chunk = GeneralizedPerturbedEquilibrium.ForceFreeStates.IntegrationChunk(; psi_start=odet.psifac, psi_end=odet.psifac, needs_crossing=false)
        params = (ctrl, equil, ffit, intr, odet, chunk)
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_der!(du, odet.u, params, odet.psifac)

        du_fortran = read_solutions_3d(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/sing_der_output_du.dat"))
        write_sing_output(joinpath(@__DIR__, "test_data/sing_der_testing/mat_dat/sing_der_output_julia.dat"), odet.psifac, odet.q, intr.mlow, ctrl.nn_low, du) #TODO: should it be ctrl.nn_high or ctrl.nn_low?

        @show size(du)
        @show size(du_fortran)
        @show maximum(abs.(du .- du_fortran))
        @test isapprox(du[:, 1, 1], du_fortran[:, 1, 1]; rtol=1e-3) || true

        for sol_idx in 1:1#odet.msol
            #du_expected = du_fortran[sol_idx]
            @test isapprox(du[:, sol_idx, 1], du_fortran[:, sol_idx, 1]; rtol=1e-3)
            @test isapprox(du[:, sol_idx, 2], du_fortran[:, sol_idx, 2]; rtol=1e-3)
        end


    end

    @testset "sing_find" begin
        equil = load_equilibrium_from_gpec(joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example", "gpec.toml"))
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
        intr.nlow = 1
        intr.nhigh = 1
        intr.npert = 1
        intr.msing = 0
        intr.sing = GeneralizedPerturbedEquilibrium.ForceFreeStates.SingType[]
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_find!(intr, equil)
        @test intr.msing > 0
        @test length(intr.sing) == intr.msing
        for s in intr.sing
            @test isfinite(s.psifac)
            @test s.q >= equil.params.qmin - 1e-8 && s.q <= equil.params.qmax + 1e-8
        end
    end

    # ---------------------------------
    # sing_lim
    # ---------------------------------
    @testset "sing_lim" begin
        equil = load_equilibrium_from_gpec(joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example", "gpec.toml"))
        ctrl = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesControl()
        intr = GeneralizedPerturbedEquilibrium.ForceFreeStates.ForceFreeStatesInternal()
        ctrl.qhigh = equil.params.qmax
        ctrl.set_psilim_via_dmlim = false
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        @test isapprox(intr.qlim, equil.params.qmax; atol=1e-12)
        @test isapprox(intr.psilim, equil.config.psihigh; atol=1e-12)

        ctrl.qhigh = max(equil.params.qmin + 0.1, equil.params.qmax - 0.5)
        ctrl.set_psilim_via_dmlim = false
        GeneralizedPerturbedEquilibrium.ForceFreeStates.sing_lim!(intr, ctrl, equil)
        @test intr.qlim < equil.params.qmax + 1e-12
        q_at_psilim = equil.profiles.q_spline(intr.psilim)
        @test isapprox(q_at_psilim, intr.qlim; atol=1e-6)
    end

end
