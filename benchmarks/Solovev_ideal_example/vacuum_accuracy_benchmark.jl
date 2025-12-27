using JLD2
using Pkg
using Plots
using LinearAlgebra

Pkg.activate("$(@__DIR__)/../.."); using JPEC

@load "$(@__DIR__)/../../examples/Solovev_ideal_example/vacuum_response_inputs.jld2" benchmark_inputs

(; wv_block, mpert, mtheta_eq, mthvac, complex_flag, kernelsign,
                wall_flag, farwall_flag, grri, xzpts, ahg_file, dir_path,
                vac_inputs, wall_settings,
                n, ipert_n, psifac) = benchmark_inputs

"""
Compute 0D accuracy metrics between two wv matrices
Returns: (relative_frobenius_norm, max_absolute_error, frobenius_norm)
"""
function compute_accuracy_metrics(wv_test::Matrix, wv_ref::Matrix)
    diff = wv_test - wv_ref
    frobenius_norm = norm(diff)
    relative_frobenius = frobenius_norm / norm(wv_ref)
    max_abs_error = maximum(abs.(diff))
    return relative_frobenius, max_abs_error, frobenius_norm
end

benchmark_n = false
benchmark_a = true

if benchmark_n
    println("="^60)
    println("VACUUM ACCURACY BENCHMARK: Varying n")
    println("="^60)

    # Define range of n values to test
    n_values = range(1,15, step=1)
    n_values = [1,2,3,4,6,10,15]  # For clearer plotting
    # Store results for Julia vs Fortran comparison
    relative_errors_jf = Float64[]
    max_errors_jf = Float64[]
    frobenius_norms_jf = Float64[]

    # Run accuracy benchmarks for each n value
    for n_test in n_values
        println("Testing n = $n_test")

        # Compute Fortran solution
        JPEC.Vacuum.set_dcon_params(mtheta_eq, vac_inputs.mlow, vac_inputs.mhigh,
                                    n_test, vac_inputs.qa, vac_inputs.r,
                                    vac_inputs.z, vac_inputs.delta)

        wv_block_fortran = copy(wv_block)
        JPEC.Vacuum.mscvac(wv_block_fortran, mpert, mtheta_eq, mthvac,
                          complex_flag, kernelsign, wall_flag,
                          farwall_flag, grri, xzpts)

        JPEC.Vacuum.unset_dcon_params()

        # Compute Julia solution
        vac_inputs_test = deepcopy(vac_inputs)
        vac_inputs_test.n = n_test
        wv_block_julia, _, _ = JPEC.Vacuum.compute_vacuum_response(vac_inputs_test, wall_settings)

        # Julia vs Fortran comparison at same n
        rel_err_jf, max_err_jf, frob_jf = compute_accuracy_metrics(wv_block_julia, wv_block_fortran)
        push!(relative_errors_jf, rel_err_jf)
        push!(max_errors_jf, max_err_jf)
        push!(frobenius_norms_jf, frob_jf)

        println("  Julia vs Fortran: rel_err = $(rel_err_jf), max_err = $(max_err_jf)\n")
    end

    # Create plots
    println("Creating plots...")

    # Plot 1: Julia vs Fortran comparison
    p1 = plot(n_values, relative_errors_jf,
             label="Relative Frobenius Error",
             marker=:circle,
             xlabel="n (toroidal mode number)",
             ylabel="Relative Error",
             title="Julia vs Fortran Accuracy",
             legend=:topright,
             linewidth=2,
             yscale=:log10,
             minorgrid=true,
             framestyle=:box)
    plot!(p1, n_values, max_errors_jf,
         label="Max Absolute Error",
         marker=:square,
         linewidth=2)

    figloc1 = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_n_scan.pdf")
    savefig(p1, figloc1)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc1"

end

if benchmark_a
    println("="^60)
    println("VACUUM ACCURACY BENCHMARK: Varying a")
    println("="^60)

    # Define range of n values to test
    a_values = [0.1, 0.2, 0.4, 1.0, 4.0, 4.0, 20]
    # Store results for Julia vs Fortran comparison
    relative_errors_jf = Float64[]
    max_errors_jf = Float64[]
    frobenius_norms_jf = Float64[]

    # Run accuracy benchmarks for each n value
    cp("vac.in", "vac.in.backup", force=true)
    for a_test in a_values
        println("Testing a = $a_test")

        # Compute Fortran solution
        JPEC.Vacuum.set_dcon_params(mtheta_eq, vac_inputs.mlow, vac_inputs.mhigh,
                                    vac_inputs.n, vac_inputs.qa, vac_inputs.r,
                                    vac_inputs.z, vac_inputs.delta)

        lines = readlines("vac.in")
        idx = findfirst(l -> occursin(r"^\s*a\s*=", l), lines)
        lines[idx] = "  a = $a_test"
        write("vac.in", join(lines, "\n") * "\n")

        wv_block_fortran = copy(wv_block)
        JPEC.Vacuum.mscvac(wv_block_fortran, mpert, mtheta_eq, mthvac,
                          complex_flag, kernelsign, wall_flag,
                          farwall_flag, grri, xzpts)

        JPEC.Vacuum.unset_dcon_params()

        # Compute Julia solution
        new_wall = JPEC.Vacuum.WallShapeSettings(
            shape= a_test > 10 ? "nowall" : wall_settings.shape,
            a=a_test,
            aw=wall_settings.aw,
            bw=wall_settings.bw,
            cw=wall_settings.cw,
            dw=wall_settings.dw,
            tw=wall_settings.tw,
            equal_arc_wall=wall_settings.equal_arc_wall,
        )
        wv_block_julia, _, _ = JPEC.Vacuum.compute_vacuum_response(vac_inputs, new_wall)

        # Julia vs Fortran comparison at same n
        rel_err_jf, max_err_jf, frob_jf = compute_accuracy_metrics(wv_block_julia, wv_block_fortran)
        push!(relative_errors_jf, rel_err_jf)
        push!(max_errors_jf, max_err_jf)
        push!(frobenius_norms_jf, frob_jf)

        println("  Julia vs Fortran: rel_err = $(rel_err_jf), max_err = $(max_err_jf)\n")
    end
    # Clean up backup
    cp("vac.in.backup", "vac.in", force=true)
    rm("vac.in.backup")

    # Create plots
    println("Creating plots...")

    # Plot 1: Julia vs Fortran comparison
    p1 = plot(a_values, relative_errors_jf,
             label="Relative Frobenius Error",
             marker=:circle,
             xlabel="a (radial separation)",
             ylabel="Relative Error",
             title="Julia vs Fortran Accuracy",
             legend=:topright,
             linewidth=2,
             yscale=:log10,
             minorgrid=true,
             framestyle=:box)
    plot!(p1, a_values, max_errors_jf,
         label="Max Absolute Error",
         marker=:square,
         linewidth=2)

    figloc1 = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_a_scan.pdf")
    savefig(p1, figloc1)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc1"

end


println("\n" * "="^60)
println("BENCHMARK COMPLETE")
println("="^60)
