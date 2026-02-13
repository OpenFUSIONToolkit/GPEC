using JLD2
using Pkg
using Plots
using LinearAlgebra
using DelimitedFiles

default(; markersize=2)

Pkg.activate("$(@__DIR__)/../..");
using JPEC

# this requires you to have run the Solovev_ideal_example first to generate the benchmark inputs
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
    n_values = range(1, 15; step=1)
    n_values = [1, 2, 3, 4, 6, 10, 15]  # For clearer plotting
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
    # savefig(p1, figloc1)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc1"

end


pdr = plot()
pdl = plot()
prz = plot(; aspect_ratio=:equal, size=(400, 800))

if benchmark_a
    println("="^60)
    println("VACUUM ACCURACY BENCHMARK: Varying a")
    println("="^60)

    # Define range of n values to test
    a_values = [0.01, 0.1, 0.5, 1.0, 4.0, 8.0, 20]
    # Store results for Julia vs Fortran comparison
    relative_errors_jf = Float64[]
    max_errors_jf = Float64[]
    frobenius_norms_jf = Float64[]

    # Run accuracy benchmarks for each n value
    cp("vac.in", "vac.in.backup"; force=true)
    for a_test in a_values
        println("Testing a = $a_test")

        # Compute Fortran solution
        JPEC.Vacuum.set_surface_params(mtheta_eq, vac_inputs.mlow, vac_inputs.mhigh,
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

        JPEC.Vacuum.unset_surface_params()

        # Compute Julia solution
        new_wall = JPEC.Vacuum.WallShapeSettings(;
            shape=a_test > 10 ? "nowall" : wall_settings.shape,
            a=a_test,
            aw=wall_settings.aw,
            bw=wall_settings.bw,
            cw=wall_settings.cw,
            dw=wall_settings.dw,
            tw=wall_settings.tw,
            equal_arc_wall=wall_settings.equal_arc_wall
        )
        wv_block_julia, grri_julia, xzpts_julia = JPEC.Vacuum.compute_vacuum_response(vac_inputs, new_wall)

        # Julia vs Fortran comparison at same n
        rel_err_jf, max_err_jf, frob_jf = compute_accuracy_metrics(wv_block_julia, wv_block_fortran)
        push!(relative_errors_jf, rel_err_jf)
        push!(max_errors_jf, max_err_jf)
        push!(frobenius_norms_jf, frob_jf)

        println("  Julia vs Fortran: rel_err = $(rel_err_jf), max_err = $(max_err_jf)\n")

        # check wall geometry
        # Read the file, skipping the header line
        data = readdlm("wall_geo.out"; skipstart=3)
        theta_fortran = data[:, 1]
        x_fortran = data[:, 2][1:(end-2)]
        z_fortran = data[:, 3][1:(end-2)]
        i_fortran = range(1, length(x_fortran))

        x_julia_plasma = xzpts_julia[:, 1]
        z_julia_plasma = xzpts_julia[:, 2]
        x_julia = xzpts_julia[:, 3]
        z_julia = xzpts_julia[:, 4]
        i_julia = range(1, length(x_julia))
        # distance offset
        dr = ((x_fortran-x_julia) .^ 2 + (z_fortran-z_julia) .^ 2) .^ 0.5 / a_test
        dl_fortran = vcat(0.0, sqrt.(diff(x_fortran) .^ 2 + diff(z_fortran) .^ 2))
        dl_julia = vcat(0.0, sqrt.(diff(x_julia) .^ 2 + diff(z_julia) .^ 2))

        plot!(pdr, i_julia, dr; label="distance error a=$a_test", linewidth=1)

        if a_test == a_values[1]
            plot!(prz, x_julia_plasma, z_julia_plasma; label="Julia Plasma", linewidth=1, marker=:cross, color=:black)
        end
        if a_test < 20
            # this dashed vertical line proves the fortran wall has 3 points at the same x around the midplane to enforce up down symmetry
            plot!(prz, [x_fortran[1], x_fortran[1]], [-maximum(z_julia_plasma), maximum(z_julia_plasma)]; label=nothing, linewidth=1, color=:black, linestyle=:dash)
            plot!(prz, x_fortran, z_fortran; label="Fortran wall, a=$a_test", linewidth=2, marker=:o, markerfillalpha=0)
            last_color = current().series_list[end][:linecolor]
            plot!(prz, x_julia, z_julia; label="Julia wall, a=$a_test", linewidth=1, marker=:cross, markerlinewidth=0,
                color=last_color)
            plot!(prz, x_julia[1:1], z_julia[1:1]; linewidth=0, marker=:dtriangle, color=last_color, label=nothing)
            plot!(prz, x_julia[end:end], z_julia[end:end]; linewidth=0, marker=:diamond, color=last_color, aspect_ratio=:equal, label=nothing)

            plot!(pdl, i_fortran, dl_fortran; label="Fortran dl, a=$a_test", linewidth=1)
            plot!(pdl, i_julia, dl_julia; label="Julia dl, a=$a_test", linewidth=1)
        end

    end
    # Clean up backup
    cp("vac.in.backup", "vac.in"; force=true)
    rm("vac.in.backup")

    # Create plots
    println("Creating plots...")

    xlabel!(pdr, "Wall Point Index")
    ylabel!(pdr, "Distance Error / a")

    xlabel!(prz, "x (m)")
    ylabel!(prz, "z (m)")
    plot!(prz; aspect_ratio=:equal)

    figloc_geo = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_a_scan_geometry_check.pdf")
    # savefig(pdr, figloc_geo)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc_geo"

    display(pdr)
    display(pdl)
    display(prz)


    # Plot 1: Julia vs Fortran comparison
    p1 = plot(a_values, relative_errors_jf;
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
    plot!(p1, a_values, max_errors_jf;
        label="Max Absolute Error",
        marker=:square,
        linewidth=2)

    figloc1 = joinpath(@__DIR__, "vacuum_accuracy_julia_vs_fortran_a_scan.pdf")
    # savefig(p1, figloc1)
    @info "Saved Julia vs Fortran n-scan comparison to $figloc1"

    display(p1)

end


println("\n" * "="^60)
println("BENCHMARK COMPLETE")
println("="^60)
