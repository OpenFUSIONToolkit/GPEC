"""
Speed and allocations benchmark for compute_vacuum_response_3D

This script measures runtime and memory allocations as a function of grid size
for the 3D vacuum response calculation. Uses fixed aspect ratio nzeta/mtheta.

Outputs:
- Runtime scaling vs grid size
- Allocation scaling vs grid size
- Breakdown of time spent in key functions
"""

using Pkg
Pkg.activate("$(@__DIR__)/../..")

using JPEC
using JPEC.Vacuum
using JPEC.Equilibrium
using JPEC: Spl
using LinearAlgebra
using Printf
using Plots
using TOML

default(; markersize=4, linewidth=2)

# ============================================================================
# Helper functions
# ============================================================================

"""
Compute vacuum inputs from equilibrium, following Free.jl logic.
"""
function compute_vacuum_inputs_from_equil(equil::PlasmaEquilibrium, mthvac::Int, mlow::Int, mpert::Int, n::Int)
    psilim = equil.config.control.psihigh
    mtheta_eq = equil.config.control.mtheta + 1
    R = zeros(Float64, mtheta_eq)
    Z = zeros(Float64, mtheta_eq)
    ν = zeros(Float64, mtheta_eq)
    r_minor = zeros(Float64, mtheta_eq)
    θ_SFL = zeros(Float64, mtheta_eq)

    for (i, θ) in enumerate(equil.rzphi.ys)
        f = Spl.bicube_eval!(equil.rzphi, psilim, θ)
        r_minor[i] = sqrt(f[1])
        θ_SFL[i] = 2π * (θ + f[2])
        ν[i] = f[3]
    end

    R .= equil.ro .+ r_minor .* cos.(θ_SFL)
    Z .= equil.zo .+ r_minor .* sin.(θ_SFL)

    return Vacuum.VacuumInput(;
        r=reverse(R),
        z=reverse(Z),
        ν=reverse(ν),
        mlow=mlow,
        mpert=mpert,
        n=n,
        mtheta=mthvac,
        force_wv_symmetry=true
    )
end

"""
Format bytes as human-readable string.
"""
function format_bytes(bytes::Integer)
    if bytes < 1024
        return @sprintf "%d B" bytes
    elseif bytes < 1024^2
        return @sprintf "%.2f KB" bytes / 1024
    elseif bytes < 1024^3
        return @sprintf "%.2f MB" bytes / 1024^2
    else
        return @sprintf "%.2f GB" bytes / 1024^3
    end
end

# ============================================================================
# Setup equilibrium and parameters
# ============================================================================

println("="^70)
println("3D VACUUM RESPONSE SPEED/ALLOCATIONS BENCHMARK")
println("="^70)

# Load equilibrium from Solovev example
example_dir = joinpath(@__DIR__, "../../examples/Solovev_ideal_example")
equil = Equilibrium.setup_equilibrium(joinpath(example_dir, "equil.toml"))

# Load DCON settings to get mode numbers
dcon_inputs = TOML.parsefile(joinpath(example_dir, "dcon.toml"))
wall_inputs = dcon_inputs["WALL"]
wall_settings = Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in wall_inputs)...)

# Determine mode numbers (following Main.jl logic)
nlow = dcon_inputs["DCON_CONTROL"]["nn_low"]
nhigh = dcon_inputs["DCON_CONTROL"]["nn_high"]
npert = nhigh - nlow + 1

# Poloidal mode range
mlow = trunc(Int, min(nlow * equil.params.qmin, 0)) - 4
mhigh = trunc(Int, nhigh * equil.params.qmax)
mpert = mhigh - mlow + 1

println("\nEquilibrium loaded from: $example_dir")
println("  Major radius R0 = $(equil.ro)")
println("  q on axis       = $(equil.params.qmin)")
println("  q at edge       = $(equil.params.qmax)")
println("\nMode numbers:")
println("  n range: $nlow to $nhigh (npert = $npert)")
println("  m range: $mlow to $mhigh (mpert = $mpert)")
println("  Total modes: $(mpert * npert)")

# ============================================================================
# Benchmark parameters
# ============================================================================

# Fixed aspect ratio for 3D: nzeta/mtheta = 1.5
aspect_ratio_grid = 1.5

# Grid sizes to test
mtheta_values = [16, 32, 48, 64, 96]  # Small test set
nzeta_values = [round(Int, aspect_ratio_grid * m) for m in mtheta_values]
total_points = [m * n for (m, n) in zip(mtheta_values, nzeta_values)]

println("\n" * "="^70)
println("BENCHMARK PARAMETERS")
println("="^70)
println("  Grid aspect ratio (nzeta/mtheta): $aspect_ratio_grid")
println("  Mtheta values: $mtheta_values")
println("  Nzeta values:  $nzeta_values")
println("  Total points:  $total_points")

# ============================================================================
# Warmup run (compile everything)
# ============================================================================

println("\n" * "="^70)
println("WARMUP RUN (compiling)")
println("="^70)

mth_warmup = mtheta_values[1]
nz_warmup = nzeta_values[1]
print("  Running warmup with mtheta=$mth_warmup, nzeta=$nz_warmup ... ")

vac_inputs_2D_warmup = compute_vacuum_inputs_from_equil(equil, mth_warmup, mlow, mpert, nlow)
vac_inputs_3D_warmup = Vacuum.VacuumInput3D(vac_inputs_2D_warmup, nz_warmup, nlow, npert)
Vacuum.compute_vacuum_response_3D(vac_inputs_3D_warmup, wall_settings)
println("done")

# ============================================================================
# Speed and allocation benchmark
# ============================================================================

println("\n" * "="^70)
println("3D SPEED/ALLOCATION BENCHMARK (nzeta/mtheta = $aspect_ratio_grid)")
println("="^70)

times_3D = Float64[]
allocs_3D = Int[]
alloc_bytes_3D = Int[]

for (i, mth) in enumerate(mtheta_values)
    nz = nzeta_values[i]
    print("  mtheta=$mth, nzeta=$nz ($(mth*nz) pts) ... ")

    local vac_inputs_2D = compute_vacuum_inputs_from_equil(equil, mth, mlow, mpert, nlow)
    local vac_inputs_3D = Vacuum.VacuumInput3D(vac_inputs_2D, nz, nlow, npert)

    # Run with allocation tracking
    stats = @timed Vacuum.compute_vacuum_response_3D(vac_inputs_3D, wall_settings)

    push!(times_3D, stats.time)
    push!(allocs_3D, Base.gc_alloc_count(stats.gcstats))
    push!(alloc_bytes_3D, stats.bytes)

    println(@sprintf "time=%.3fs, allocs=%d, bytes=%s" stats.time Base.gc_alloc_count(stats.gcstats) format_bytes(stats.bytes))
end

# ============================================================================
# Create plots
# ============================================================================

println("\n" * "="^70)
println("CREATING PLOTS")
println("="^70)

# Plot 1: Runtime scaling
p1 = plot(total_points, times_3D;
    label="Measured",
    marker=:circle,
    xlabel="Total grid points (mtheta × nzeta)",
    ylabel="Runtime (s)",
    title="3D Vacuum Runtime Scaling",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,
    minorgrid=true,
    framestyle=:box)

# Add O(N²) reference line
if length(times_3D) >= 2
    N_ref = total_points
    t_ref_scale = times_3D[end] * (N_ref ./ N_ref[end]) .^ 2
    plot!(p1, N_ref, t_ref_scale; label="O(N²)", linestyle=:dash, color=:gray)
end

# Plot 2: Allocation count scaling
p2 = plot(total_points, allocs_3D;
    label="Allocation count",
    marker=:circle,
    xlabel="Total grid points (mtheta × nzeta)",
    ylabel="Number of allocations",
    title="Allocation Count Scaling",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,
    minorgrid=true,
    framestyle=:box)

# Add O(N) and O(N²) reference lines for allocations
if length(allocs_3D) >= 2
    N_ref = total_points
    alloc_ref_N = allocs_3D[end] * (N_ref ./ N_ref[end])
    alloc_ref_N2 = allocs_3D[end] * (N_ref ./ N_ref[end]) .^ 2
    plot!(p2, N_ref, alloc_ref_N; label="O(N)", linestyle=:dash, color=:blue)
    plot!(p2, N_ref, alloc_ref_N2; label="O(N²)", linestyle=:dash, color=:red)
end

# Plot 3: Memory allocation scaling
p3 = plot(total_points, alloc_bytes_3D ./ 1e6;
    label="Memory allocated",
    marker=:circle,
    xlabel="Total grid points (mtheta × nzeta)",
    ylabel="Memory allocated (MB)",
    title="Memory Allocation Scaling",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,
    minorgrid=true,
    framestyle=:box)

# Add O(N²) reference line for memory
if length(alloc_bytes_3D) >= 2
    N_ref = total_points
    mem_ref_N2 = alloc_bytes_3D[end] * (N_ref ./ N_ref[end]) .^ 2 ./ 1e6
    plot!(p3, N_ref, mem_ref_N2; label="O(N²)", linestyle=:dash, color=:gray)
end

# Plot 4: Allocations per grid point
allocs_per_point = allocs_3D ./ total_points
p4 = plot(total_points, allocs_per_point;
    label="Allocs per grid point",
    marker=:circle,
    xlabel="Total grid points (mtheta × nzeta)",
    ylabel="Allocations / grid point",
    title="Allocations per Grid Point",
    legend=:topright,
    xscale=:log10,
    minorgrid=true,
    framestyle=:box)

# Combined plot
p_combined = plot(p1, p2, p3, p4; layout=(2, 2), size=(1000, 800))

# Save plots
savefig(p1, joinpath(@__DIR__, "vacuum_3D_runtime_scaling.pdf"))
savefig(p2, joinpath(@__DIR__, "vacuum_3D_allocation_count_scaling.pdf"))
savefig(p3, joinpath(@__DIR__, "vacuum_3D_memory_scaling.pdf"))
savefig(p_combined, joinpath(@__DIR__, "vacuum_3D_speed_benchmark_combined.pdf"))

println("Plots saved to $(joinpath(@__DIR__, "vacuum_3D_*.pdf"))")

# Display plots
display(p_combined)

# ============================================================================
# Summary table
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("\n3D Speed/Allocation Benchmark (nzeta/mtheta = $aspect_ratio_grid):")
println("  mtheta | nzeta | Total Pts |  Time (s)  |   Allocs   |   Memory")
println("  " * "-"^70)
for i in eachindex(mtheta_values)
    @printf "  %6d | %5d | %9d |  %8.3f  | %10d | %10s\n" mtheta_values[i] nzeta_values[i] total_points[i] times_3D[i] allocs_3D[i] format_bytes(alloc_bytes_3D[i])
end

# Compute scaling exponents if we have enough data points
if length(mtheta_values) >= 2
    println("\nScaling Analysis:")

    # Runtime scaling: t ∝ N^α
    log_N = log.(total_points)
    log_t = log.(times_3D)
    α_time = (log_t[end] - log_t[1]) / (log_N[end] - log_N[1])
    println(@sprintf "  Runtime scaling exponent: α = %.2f (expect ~2 for O(N²))" α_time)

    # Allocation scaling: allocs ∝ N^β
    log_allocs = log.(allocs_3D)
    β_allocs = (log_allocs[end] - log_allocs[1]) / (log_N[end] - log_N[1])
    println(@sprintf "  Allocation count scaling exponent: β = %.2f" β_allocs)

    # Memory scaling: bytes ∝ N^γ
    log_bytes = log.(alloc_bytes_3D)
    γ_mem = (log_bytes[end] - log_bytes[1]) / (log_N[end] - log_N[1])
    println(@sprintf "  Memory scaling exponent: γ = %.2f (expect ~2 for O(N²) matrices)" γ_mem)
end

println("\n" * "="^70)
println("BENCHMARK COMPLETE")
println("="^70)
