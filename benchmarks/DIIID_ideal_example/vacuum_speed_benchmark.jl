using BenchmarkTools
using Profile
using JLD2
using Pkg
using Plots

Pkg.activate("../..");
using JPEC

@load "benchmark_inputs.jld2" benchmark_inputs

(; wv_block, mpert, mtheta_eq, mthvac, complex_flag, kernelsign,
    wall_flag, farwall_flag, grri, xzpts, ahg_file, dir_path,
    vac_inputs, wall_settings,
    n, ipert_n, psifac) = benchmark_inputs

benchmark_n = true
benchmark_m = false

if benchmark_n
    # Define range of n values to test
    n_values = [1, 2, 4, 8, 16, 32, 64]

    # Store results
    fortran_times = Float64[]
    fortran_stdevtimes = Float64[]
    fortran_allocs = Int[]
    fortran_memory = Float64[]

    julia_times = Float64[]
    julia_stdevtimes = Float64[]
    julia_allocs = Int[]
    julia_memory = Float64[]

    # Run benchmarks for each n value
    for n_test in n_values
        println("Benchmarking n = $n_test")

        # Benchmark Fortran version
        JPEC.Vacuum.set_dcon_params(mtheta_eq, vac_inputs.mlow, vac_inputs.mhigh,
            n_test, vac_inputs.qa, vac_inputs.r,
            vac_inputs.z, vac_inputs.delta)

        b_fortran = @benchmark JPEC.Vacuum.mscvac($wv_block, $mpert, $mtheta_eq, $mthvac,
            $complex_flag, $kernelsign, $wall_flag,
            $farwall_flag, $grri, $xzpts, $ahg_file, "../../examples/DIIID_ideal_example") samples=10 seconds=1

        push!(fortran_times, median(b_fortran).time / 1e9)
        push!(fortran_stdevtimes, std(b_fortran).time / 1e9)
        push!(fortran_allocs, median(b_fortran).allocs)
        push!(fortran_memory, median(b_fortran).memory / 1e6)

        JPEC.Vacuum.unset_dcon_params()

        # Benchmark Julia version
        vac_inputs_test = deepcopy(vac_inputs)
        vac_inputs_test.n = n_test

        b_julia = @benchmark JPEC.Vacuum.compute_vacuum_response($vac_inputs_test, $wall_settings) samples=10 seconds=1

        push!(julia_times, median(b_julia).time / 1e9)
        push!(julia_stdevtimes, std(b_julia).time / 1e9)
        push!(julia_allocs, median(b_julia).allocs)
        push!(julia_memory, median(b_julia).memory / 1e6)
    end


    # Time comparison
    p1 = plot(n_values, fortran_times; label="Fortran", marker=:circle,
        xlabel="n", ylabel="Time per execution (s)", title="Execution Time Comparison",
        legend=:topleft, linewidth=2, yscale=:log10)
    plot!(p1, n_values, julia_times; label="Julia", marker=:square, linewidth=2)
    # Add error bars for Fortran
    scatter!(p1, n_values, fortran_times; yerror=fortran_stdevtimes,
        label="", color=:blue, markersize=0)

    # Add error bars for Julia
    scatter!(p1, n_values, julia_times; yerror=julia_stdevtimes,
        label="", color=:red, markersize=0)

    figloc = joinpath(@__DIR__, "vacuum_speed_benchmark_time_comparison.pdf")
    savefig(p1, figloc)
    @info "Saved time comparison plot to $figloc"

end

if benchmark_m
    # Define range of mhigh values to test
    mhigh_values = [8, 16, 32] # 48 runs into memory errors on macbooks

    # Store results
    fortran_mhigh_times = Float64[]
    fortran_mhigh_stdevtimes = Float64[]
    fortran_mhigh_allocs = Int[]
    fortran_mhigh_memory = Float64[]

    julia_mhigh_times = Float64[]
    julia_mhigh_stdevtimes = Float64[]
    julia_mhigh_allocs = Int[]
    julia_mhigh_memory = Float64[]

    # Run benchmarks for each mhigh value
    for mhigh_test in mhigh_values
        println("Benchmarking mhigh = $mhigh_test")

        # Calculate mpert based on mhigh
        mpert_test = mhigh_test - vac_inputs.mlow

        # Benchmark Fortran version
        JPEC.Vacuum.set_dcon_params(mtheta_eq, vac_inputs.mlow, mhigh_test,
            1, vac_inputs.qa, vac_inputs.r,
            vac_inputs.z, vac_inputs.delta)

        b_fortran = @benchmark JPEC.Vacuum.mscvac($wv_block, $mpert_test, $mtheta_eq, $mthvac,
            $complex_flag, $kernelsign, $wall_flag,
            $farwall_flag, $grri, $xzpts, $ahg_file, "../../examples/DIIID_ideal_example")

        push!(fortran_mhigh_times, median(b_fortran).time / 1e9)
        push!(fortran_mhigh_stdevtimes, std(b_fortran).time / 1e9)
        push!(fortran_mhigh_allocs, median(b_fortran).allocs)
        push!(fortran_mhigh_memory, median(b_fortran).memory / 1e6)

        JPEC.Vacuum.unset_dcon_params()

        # Benchmark Julia version
        vac_inputs_test = deepcopy(vac_inputs)
        vac_inputs_test.mhigh = mhigh_test
        vac_inputs_test.mpert = mhigh_test - vac_inputs_test.mlow

        b_julia = @benchmark JPEC.Vacuum.compute_vacuum_response($vac_inputs_test, $wall_settings)

        push!(julia_mhigh_times, median(b_julia).time / 1e9)
        push!(julia_mhigh_stdevtimes, std(b_julia).time / 1e9)
        push!(julia_mhigh_allocs, median(b_julia).allocs)
        push!(julia_mhigh_memory, median(b_julia).memory / 1e6)
    end

    # Time comparison for mhigh scan
    p2 = plot(mhigh_values, fortran_mhigh_times; label="Fortran", marker=:circle,
        xlabel="mhigh", ylabel="Time per execution (s)", title="Execution Time vs mhigh",
        legend=:topleft, linewidth=2, yaxis=:log)
    plot!(p2, mhigh_values, julia_mhigh_times; label="Julia", marker=:square, linewidth=2)

    # Add error bars for Fortran
    scatter!(p2, mhigh_values, fortran_mhigh_times; yerror=fortran_mhigh_stdevtimes,
        label="", color=:blue, markersize=0)

    # Add error bars for Julia
    scatter!(p2, mhigh_values, julia_mhigh_times; yerror=julia_mhigh_stdevtimes,
        label="", color=:red, markersize=0)

    figloc_mhigh = joinpath(@__DIR__, "vacuum_speed_benchmark_mhigh_time_comparison.pdf")
    savefig(p2, figloc_mhigh)
    @info "Saved mhigh time comparison plot to $figloc_mhigh"
end
