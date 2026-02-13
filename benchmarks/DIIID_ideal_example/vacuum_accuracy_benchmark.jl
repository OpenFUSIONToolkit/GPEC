using JLD2
using Pkg
using Plots
using LinearAlgebra

Pkg.activate("$(@__DIR__)/../..");
using JPEC

# this requires you to have run the DIIID-like_ideal_example first to generate the benchmark inputs
@load "$(@__DIR__)/../../examples/DIIID-like_ideal_example/benchmark_inputs.jld2" benchmark_inputs

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

benchmark_n = true
benchmark_m = false

if benchmark_n
    println("="^60)
    println("VACUUM ACCURACY BENCHMARK: Varying n")
    println("="^60)

    # Define range of n values to test
    n_values = range(1, 15; step=1)
    # Store results for Julia vs Fortran comparison
    relative_errors_jf = Float64[]
    max_errors_jf = Float64[]
    frobenius_norms_jf = Float64[]

    # Run accuracy benchmarks for each n value
    for n_test in n_values
        println("Testing n = $n_test")

        # Compute Fortran solution
        JPEC.Vacuum.set_surface_params(mtheta_eq, vac_inputs.mlow, vac_inputs.mhigh,
            n_test, vac_inputs.qa, vac_inputs.r,
            vac_inputs.z, vac_inputs.delta)

        wv_block_fortran = copy(wv_block)
        JPEC.Vacuum.mscvac(wv_block_fortran, mpert, mtheta_eq, mthvac,
            complex_flag, kernelsign, wall_flag,
            farwall_flag, grri, xzpts)

        JPEC.Vacuum.unset_surface_params()

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
    p1 = plot(n_values, relative_errors_jf;
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
    plot!(p1, n_values, max_errors_jf;
        label="Max Absolute Error",
        marker=:square,
        linewidth=2)

    figloc1 = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_n_scan.pdf")
    savefig(p1, figloc1)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc1"

end

if benchmark_m
    println("\n" * "="^60)
    println("VACUUM ACCURACY BENCHMARK: Varying mhigh")
    println("="^60)

    # Define range of mhigh values to test
    mhigh_values = [8, 16, 24, 32] # 48 runs into memory errors on macbooks
    mhigh_reference = 32  # High resolution reference

    # Store results for Julia vs Fortran comparison
    relative_errors_jf_m = Float64[]
    max_errors_jf_m = Float64[]

    # Store results for convergence analysis
    relative_errors_julia_conv_m = Float64[]
    relative_errors_fortran_conv_m = Float64[]

    # Compute reference solutions at high resolution
    println("\nComputing reference solutions at mhigh = $mhigh_reference...")

    # Julia reference
    vac_inputs_ref_julia_m = deepcopy(vac_inputs)
    vac_inputs_ref_julia_m.mhigh = mhigh_reference
    vac_inputs_ref_julia_m.mpert = mhigh_reference - vac_inputs_ref_julia_m.mlow
    wv_ref_julia_m, _, _ = JPEC.Vacuum.compute_vacuum_response(vac_inputs_ref_julia_m, wall_settings)

    # Fortran reference
    mpert_ref = mhigh_reference - vac_inputs.mlow
    JPEC.Vacuum.set_surface_params(mtheta_eq, vac_inputs.mlow, mhigh_reference,
        1, vac_inputs.qa, vac_inputs.r,
        vac_inputs.z, vac_inputs.delta)
    wv_block_ref_fortran_m = zeros(ComplexF64, mpert_ref, mpert_ref)
    grri_ref = zeros(Float64, 2 * (mthvac + 5), 2 * mpert_ref)
    JPEC.Vacuum.mscvac(wv_block_ref_fortran_m, mpert_ref, mtheta_eq, mthvac,
        complex_flag, kernelsign, wall_flag,
        farwall_flag, grri_ref, xzpts, ahg_file, "$(@__DIR__)/../../examples/DIIID-like_ideal_example")
    JPEC.Vacuum.unset_surface_params()

    println("Reference solutions computed.\n")

    # Run accuracy benchmarks for each mhigh value
    for mhigh_test in mhigh_values
        println("Testing mhigh = $mhigh_test")

        mpert_test = mhigh_test - vac_inputs.mlow

        # Compute Fortran solution
        JPEC.Vacuum.set_surface_params(mtheta_eq, vac_inputs.mlow, mhigh_test,
            1, vac_inputs.qa, vac_inputs.r,
            vac_inputs.z, vac_inputs.delta)

        wv_block_fortran_m = zeros(ComplexF64, mpert_test, mpert_test)
        grri_test = zeros(Float64, 2 * (mthvac + 5), 2 * mpert_test)
        JPEC.Vacuum.mscvac(wv_block_fortran_m, mpert_test, mtheta_eq, mthvac,
            complex_flag, kernelsign, wall_flag,
            farwall_flag, grri_test, xzpts, ahg_file, "$(@__DIR__)/../../examples/DIIID-like_ideal_example")

        JPEC.Vacuum.unset_surface_params()

        # Compute Julia solution
        vac_inputs_test_m = deepcopy(vac_inputs)
        vac_inputs_test_m.mhigh = mhigh_test
        vac_inputs_test_m.mpert = mpert_test
        wv_block_julia_m, _, _ = JPEC.Vacuum.compute_vacuum_response(vac_inputs_test_m, wall_settings)

        # Comparison 1: Julia vs Fortran at same mhigh
        rel_err_jf_m, max_err_jf_m, _ = compute_accuracy_metrics(wv_block_julia_m, wv_block_fortran_m)
        push!(relative_errors_jf_m, rel_err_jf_m)
        push!(max_errors_jf_m, max_err_jf_m)

        println("  Julia vs Fortran: rel_err = $(rel_err_jf_m)")

        # Comparison 2: Each vs its own high-resolution reference
        # Need to extract the corresponding block from reference
        idx_range = 1:mpert_test
        wv_ref_julia_block = wv_ref_julia_m[idx_range, idx_range]
        wv_ref_fortran_block = wv_block_ref_fortran_m[idx_range, idx_range]

        rel_err_julia_m, _, _ = compute_accuracy_metrics(wv_block_julia_m, wv_ref_julia_block)
        rel_err_fortran_m, _, _ = compute_accuracy_metrics(wv_block_fortran_m, wv_ref_fortran_block)

        push!(relative_errors_julia_conv_m, rel_err_julia_m)
        push!(relative_errors_fortran_conv_m, rel_err_fortran_m)

        println("  Julia convergence: rel_err = $(rel_err_julia_m)")
        println("  Fortran convergence: rel_err = $(rel_err_fortran_m)\n")
    end

    # Create plots for mhigh scan
    println("Creating plots...")

    # Replace exact zeros with small epsilon for log plotting
    eps_plot = 1e-16
    julia_conv_plot = [x == 0.0 ? eps_plot : x for x in relative_errors_julia_conv_m]
    fortran_conv_plot = [x == 0.0 ? eps_plot : x for x in relative_errors_fortran_conv_m]

    # Plot: Convergence for each implementation (mhigh scan)
    p4 = plot(mhigh_values, julia_conv_plot;
        label="Julia (vs mhigh=$mhigh_reference)",
        marker=:circle,
        xlabel="mhigh (maximum poloidal mode)",
        ylabel="Relative Error vs High-Res Reference",
        title="Convergence Analysis (mhigh scan)",
        legend=:topright,
        linewidth=2,
        yscale=:log10,
        minorgrid=true,
        framestyle=:box)
    plot!(p4, mhigh_values, fortran_conv_plot;
        label="Fortran (vs mhigh=$mhigh_reference)",
        marker=:square,
        linewidth=2)

    figloc4 = joinpath(@__DIR__, "vacuum_accuracy_convergence_mhigh.pdf")
    savefig(p4, figloc4)
    @info "Saved mhigh convergence plot to $figloc4"

    # Combined plot for mhigh
    p5 = plot(mhigh_values, relative_errors_jf_m;
        label="Julia vs Fortran",
        marker=:diamond,
        xlabel="mhigh (maximum poloidal mode)",
        ylabel="Relative Frobenius Error",
        title="Julia vs Fortran Accuracy (mhigh scan)",
        linewidth=2,
        yscale=:log10,
        minorgrid=true,
        framestyle=:box,
        legend=:topright)

    figloc5 = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_mhigh.pdf")
    savefig(p5, figloc5)
    @info "Saved Julia vs Fortran mhigh comparison to $figloc5"

    # Combined plot with both metrics for mhigh scan
    p6 = plot(; layout=(2, 1), size=(800, 800))

    plot!(p6[1], mhigh_values, relative_errors_jf_m;
        label="Julia vs Fortran",
        marker=:diamond,
        ylabel="Relative Frobenius Error",
        title="Julia vs Fortran Agreement (mhigh scan)",
        linewidth=2,
        yscale=:log10,
        minorgrid=true,
        framestyle=:box,
        legend=:topright)

    plot!(p6[2], mhigh_values, julia_conv_plot;
        label="Julia (vs mhigh=$mhigh_reference)",
        marker=:circle,
        xlabel="mhigh (maximum poloidal mode)",
        ylabel="Relative Error vs High-Res",
        title="Convergence to High Resolution",
        linewidth=2,
        yscale=:log10,
        minorgrid=true,
        framestyle=:box,
        legend=:topright)
    plot!(p6[2], mhigh_values, fortran_conv_plot;
        label="Fortran (vs mhigh=$mhigh_reference)",
        marker=:square,
        linewidth=2)

    figloc6 = joinpath(@__DIR__, "vacuum_accuracy_combined_mhigh.pdf")
    savefig(p6, figloc6)
    @info "Saved combined mhigh plot to $figloc6"
end

println("\n" * "="^60)
println("BENCHMARK COMPLETE")
println("="^60)
