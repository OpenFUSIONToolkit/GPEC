#!/usr/bin/env julia
"""
benchmark_fortran.jl - Compare Julia GPEC results against Fortran GPEC reference outputs

Reads Fortran output NetCDF files from a GPEC Fortran example directory, builds a
matched Julia configuration, optionally runs Julia GPEC, then prints a side-by-side
comparison table.

# Usage

```bash
# Full run (creates bench dir, runs Julia, compares)
julia --project=benchmarks benchmarks/benchmark_fortran.jl /path/to/DIIID_ideal_example --plot

# Skip Julia re-run (use existing gpec.h5 in bench dir)
julia --project=benchmarks benchmarks/benchmark_fortran.jl /path/to/DIIID_ideal_example --plot --skip-run
```

# Arguments

- `fortran_dir`  : Path to Fortran example directory (contains equil.in, dcon.in, *.nc)
- `--plot`       : Generate comparison plots (saved to bench dir)
- `--skip-run`   : Skip running Julia GPEC (assume gpec.h5 already exists)

# Outputs (written to bench dir)

- `gpec.toml`            : Julia configuration matched to Fortran inputs
- `forcing.dat`          : External forcing extracted from Phi_coil
- `<eq_filename>`        : Copy of equilibrium file
- `gpec.h5`              : Julia GPEC output (unless --skip-run)
- `comparison_table.txt` : Side-by-side comparison (also printed to stdout)
- `comparison_plots.png` : Plots (if --plot)
"""

using Dates
using HDF5
using NCDatasets
using Plots
using Printf
using Statistics

# ─── Argument parsing ────────────────────────────────────────────────────────

function parse_args(args)
    fortran_dir = ""
    do_plot     = false
    skip_run    = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--plot"
            do_plot = true
        elseif a == "--skip-run"
            skip_run = true
        elseif !startswith(a, "--")
            fortran_dir = a
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    isempty(fortran_dir) && error("Usage: benchmark_fortran.jl <fortran_dir> [--plot] [--skip-run]")
    return (fortran_dir=abspath(fortran_dir), do_plot=do_plot, skip_run=skip_run)
end

# ─── Fortran namelist parser ─────────────────────────────────────────────────

"""
    parse_namelist(filepath) -> Dict{String, String}

Parse a Fortran namelist file (`&NAME ... /`) and return a flat dict of key→raw_value strings.
Handles quoted strings, booleans (t/f), and numeric values.
"""
function parse_namelist(filepath::String)
    text = read(filepath, String)
    result = Dict{String, String}()
    lines = split(text, '\n')
    cleaned = join([split(l, '!')[1] for l in lines], "\n")
    for m in eachmatch(r"(\w+)\s*=\s*(\"[^\"]*\"|[^=,/\n&]+?)(?=\s*(?:,|\n|\s+\w+\s*=|/))", cleaned)
        key = lowercase(strip(m.captures[1]))
        val = strip(m.captures[2])
        if length(val) >= 2 && val[1] == '"' && val[end] == '"'
            val = val[2:end-1]
        end
        isempty(key) || (result[key] = val)
    end
    return result
end

"""Extract a typed value from the namelist dict, with a default."""
function nl_get(d::Dict, key::String, ::Type{String}, default="")
    haskey(d, key) ? d[key] : default
end
function nl_get(d::Dict, key::String, ::Type{Float64}, default=0.0)
    haskey(d, key) ? parse(Float64, d[key]) : default
end
function nl_get(d::Dict, key::String, ::Type{Int}, default=0)
    haskey(d, key) ? parse(Int, d[key]) : default
end
function nl_get(d::Dict, key::String, ::Type{Bool}, default=false)
    haskey(d, key) ? (lowercase(d[key]) in ("t", "true", ".true.")) : default
end

# ─── Load Fortran NC outputs ─────────────────────────────────────────────────

"""Load Fortran outputs from the three NetCDF files in fortran_dir."""
function load_fortran_outputs(fortran_dir::String, nn::Int)
    dcon_file    = joinpath(fortran_dir, "dcon_output_n$(nn).nc")
    profile_file = joinpath(fortran_dir, "gpec_profile_output_n$(nn).nc")
    control_file = joinpath(fortran_dir, "gpec_control_output_n$(nn).nc")

    isfile(dcon_file)    || error("Not found: $dcon_file")
    isfile(profile_file) || error("Not found: $profile_file")
    isfile(control_file) || error("Not found: $control_file")

    fort = Dict{String,Any}()

    NCDataset(dcon_file, "r") do ds
        fort["psilim"] = Float64(ds.attrib["psilim"])
        fort["qlim"]   = Float64(ds.attrib["qlim"])
        fort["mlow"]   = Int(ds.attrib["mlow"])
        fort["mhigh"]  = Int(ds.attrib["mhigh"])
        fort["mpert"]  = Int(ds.attrib["mpert"])

        fort["psi_n_dcon"] = Float64.(ds["psi_n"][:])
        fort["q_dcon"]     = Float64.(ds["q"][:])
        fort["di"]         = Float64.(ds["di"][:])
        fort["dr"]         = Float64.(ds["dr"][:])
        fort["m_vals"]     = Int.(ds["m"][:])

        # W_t eigenvalues: NC dim (i=2, mode=34) → Julia (mode, i) = (34, 2)
        wt_ev = Float64.(ds["W_t_eigenvalue"][:, :])
        fort["W_t_eigenvalue"] = complex.(wt_ev[:, 1], wt_ev[:, 2])
    end

    NCDataset(profile_file, "r") do ds
        fort["psi_n_rational"] = Float64.(ds["psi_n_rational"][:])
        fort["q_rational"]     = Float64.(ds["q_rational"][:])

        # Phi_res: NC dim (i=2, psi_n_rational=4) → Julia (psi_n_rational, i)
        phi_res = Float64.(ds["Phi_res"][:, :])
        fort["Phi_res"] = complex.(phi_res[:, 1], phi_res[:, 2])

        delta = Float64.(ds["Delta"][:, :])
        fort["Delta"] = complex.(delta[:, 1], delta[:, 2])

        fort["w_isl"] = Float64.(ds["w_isl"][:])
        fort["K_isl"] = Float64.(ds["K_isl"][:])
    end

    NCDataset(control_file, "r") do ds
        # Phi_coil: NC dim (i=2, coil_index=3, m=34) → Julia (m, coil_index, i) = (34, 3, 2)
        phi_coil = Float64.(ds["Phi_coil"][:, :, :])
        amp_real = dropdims(sum(phi_coil[:, :, 1], dims=2), dims=2)
        amp_imag = dropdims(sum(phi_coil[:, :, 2], dims=2), dims=2)
        fort["Phi_coil_amp"] = complex.(amp_real, amp_imag)
    end

    return fort
end

# ─── Load Julia h5 outputs ───────────────────────────────────────────────────

"""Load Julia GPEC outputs from gpec.h5."""
function load_julia_outputs(h5_path::String)
    isfile(h5_path) || error("Julia output not found: $h5_path")

    julia = Dict{String,Any}()
    h5open(h5_path, "r") do f
        julia["psilim"] = read(f, "info/psilim")
        julia["qlim"]   = read(f, "info/qlim")
        julia["et"]     = read(f, "vacuum/et")
        julia["psi_q"]  = read(f, "splines/profiles/xs")
        julia["q"]      = read(f, "splines/profiles/q")
        julia["di"]     = haskey(f, "locstab/di") ? read(f, "locstab/di") : Float64[]
        julia["dr"]     = haskey(f, "locstab/dr") ? read(f, "locstab/dr") : Float64[]

        sc = "perturbed_equilibrium/singular_coupling"
        # New structure: [n_rational] vectors with matching metadata
        julia["rational_psi"]        = haskey(f, "$sc/rational_psi")        ? read(f, "$sc/rational_psi")        : Float64[]
        julia["rational_q"]          = haskey(f, "$sc/rational_q")          ? read(f, "$sc/rational_q")          : Float64[]
        julia["rational_n"]          = haskey(f, "$sc/rational_n")          ? read(f, "$sc/rational_n")          : Int[]
        julia["resonant_flux"]       = haskey(f, "$sc/resonant_flux")       ? read(f, "$sc/resonant_flux")       : ComplexF64[]
        julia["island_half_width"]   = haskey(f, "$sc/island_half_width")   ? read(f, "$sc/island_half_width")   : Float64[]
        julia["chirikov_parameter"]  = haskey(f, "$sc/chirikov_parameter")  ? read(f, "$sc/chirikov_parameter")  : Float64[]
        julia["delta_prime"]         = haskey(f, "$sc/delta_prime")         ? read(f, "$sc/delta_prime")         : ComplexF64[]
    end
    return julia
end

# ─── Write bench_dir/forcing.dat ─────────────────────────────────────────────

"""Write forcing.dat from Phi_coil amplitudes and m values."""
function write_forcing_dat(filepath::String, n::Int, m_vals::Vector{Int}, amp::Vector{ComplexF64})
    open(filepath, "w") do io
        println(io, "# n  m  real(Phi_coil_sum)  imag(Phi_coil_sum)")
        for (m, a) in zip(m_vals, amp)
            @printf(io, "%4d  %4d  %+.8e  %+.8e\n", n, m, real(a), imag(a))
        end
    end
end

# ─── Write bench_dir/gpec.toml ───────────────────────────────────────────────

"""Generate gpec.toml matched to Fortran equil.in and dcon.in parameters."""
function write_gpec_toml(
    filepath::String,
    fortran_dir::String,
    eq::Dict,
    nn::Int,
    dmlim::Float64, qlow::Float64, psiedge::Float64,
    delta_mlow::Int, delta_mhigh::Int
)
    eq_type     = nl_get(eq, "eq_type",     String, "efit")
    eq_filename = nl_get(eq, "eq_filename", String, "")
    jac_type    = nl_get(eq, "jac_type",    String, "hamada")
    grid_type   = nl_get(eq, "grid_type",   String, "ldp")
    psilow      = nl_get(eq, "psilow",      Float64, 1e-4)
    psihigh     = nl_get(eq, "psihigh",     Float64, 0.993)
    mpsi        = nl_get(eq, "mpsi",        Int, 128)
    mtheta      = nl_get(eq, "mtheta",      Int, 256)

    open(filepath, "w") do io
        println(io, "# Auto-generated by benchmark_fortran.jl")
        println(io, "# Matched to Fortran run: $fortran_dir")
        println(io)
        println(io, "[Equilibrium]")
        println(io, "eq_type     = \"$eq_type\"")
        println(io, "eq_filename = \"$eq_filename\"")
        println(io, "jac_type    = \"$jac_type\"")
        println(io, "grid_type   = \"$grid_type\"")
        @printf(io, "psilow      = %.4e\n", psilow)
        @printf(io, "psihigh     = %.6f\n", psihigh)
        println(io, "mpsi        = $mpsi")
        println(io, "mtheta      = $mtheta")
        println(io, "etol        = 1e-7")
        println(io)
        println(io, "[Wall]")
        println(io, "shape = \"nowall\"")
        println(io)
        println(io, "[ForceFreeStates]")
        println(io, "bal_flag = false")
        println(io, "mat_flag = true")
        println(io, "ode_flag = true")
        println(io, "vac_flag = true")
        println(io, "mer_flag = true")
        println(io, "set_psilim_via_dmlim = true")
        @printf(io, "dmlim    = %.4f\n", dmlim)
        @printf(io, "qlow     = %.4f\n", qlow)
        @printf(io, "psiedge  = %.4f\n", psiedge)
        println(io, "nn_low   = $nn")
        println(io, "nn_high  = $nn")
        println(io, "delta_mlow  = $delta_mlow")
        println(io, "delta_mhigh = $delta_mhigh")
        println(io, "mthvac   = 512")
        println(io, "eulerlagrange_tolerance = 1e-7")
        println(io, "singfac_min = 1e-4")
        println(io, "save_interval = 3")
        println(io)
        println(io, "[ForcingTerms]")
        println(io, "forcing_data_file   = \"forcing.dat\"")
        println(io, "forcing_data_format = \"ascii\"")
        println(io)
        println(io, "[PerturbedEquilibrium]")
        println(io, "fixed_boundary          = false")
        println(io, "compute_response        = true")
        println(io, "compute_singular_coupling = true")
        println(io, "verbose                 = true")
        println(io, "write_outputs_to_HDF5   = true")
    end
end

# ─── Comparison table ────────────────────────────────────────────────────────

"""Format a relative percent difference string."""
rel_pct(a, b) = abs(b) > 1e-30 ? @sprintf("%.1f%%", 100 * abs(a - b) / abs(b)) : "N/A"

"""
Find Julia row index for a given Fortran (q_int, nn).

Matches on both n and q-integer from state.rational_n / state.rational_q.
"""
function find_julia_row(julia, q_int::Int, nn::Int)
    rat_q = julia["rational_q"]
    rat_n = julia["rational_n"]
    isempty(rat_q) && return 0
    for row in eachindex(rat_q)
        row > length(rat_n) && break
        if rat_n[row] == nn && round(Int, rat_q[row]) == q_int
            return row
        end
    end
    return 0
end

"""Print and return comparison table lines."""
function build_comparison_table(fort, julia, fortran_dir, bench_dir, nn)
    lines = String[]
    push!(lines, "=" ^ 72)
    push!(lines, "GPEC FORTRAN vs JULIA COMPARISON")
    push!(lines, "=" ^ 72)
    push!(lines, "Fortran : $fortran_dir")
    push!(lines, "Julia   : $(joinpath(bench_dir, "gpec.h5"))")
    push!(lines, "Date    : $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))")
    push!(lines, "")

    # ── Scalar parameters ──
    push!(lines, "--- Scalar Parameters ---")
    push!(lines, @sprintf("%-12s  %10s  %10s  %10s  %6s", "", "Fortran", "Julia", "AbsDiff", "Rel%"))
    for (label, fv, jv) in [
        ("psilim", fort["psilim"], julia["psilim"]),
        ("qlim",   fort["qlim"],   julia["qlim"]),
    ]
        diff = abs(fv - jv)
        push!(lines, @sprintf("%-12s  %10.6f  %10.6f  %10.6f  %6s", label, fv, jv, diff, rel_pct(fv, jv)))
    end
    push!(lines, "")

    # ── W_t eigenvalue ──
    push!(lines, "--- W_t Eigenvalue (least stable mode) ---")
    push!(lines, @sprintf("%-8s  %10s  %10s  %6s", "", "Fortran", "Julia", "Rel%"))
    f_wt = fort["W_t_eigenvalue"]
    j_et = julia["et"]
    fv = isempty(f_wt) ? NaN : f_wt[argmin(real.(f_wt))]
    jv = isempty(j_et) ? NaN : j_et[1]
    push!(lines, @sprintf("|W_t|    %10.4f  %10.4f  %6s", abs(fv), abs(jv), rel_pct(abs(fv), abs(jv))))
    push!(lines, @sprintf("Re(W_t)  %10.4f  %10.4f  %6s", real(fv), real(jv), rel_pct(real(fv), real(jv))))
    push!(lines, "")

    # ── q profile RMS ──
    push!(lines, "--- q Profile (RMS on common grid) ---")
    fq_psi = fort["psi_n_dcon"]
    fq     = fort["q_dcon"]
    jq_psi = julia["psi_q"]
    jq     = julia["q"]
    if !isempty(jq) && !isempty(fq)
        jq_on_fgrid = [_interp1(jq_psi, jq, p) for p in fq_psi]
        rms = sqrt(mean((fq .- jq_on_fgrid).^2))
        rms_rel = rms / mean(abs.(fq)) * 100
        push!(lines, @sprintf("RMS(q_fort - q_julia) = %.5f  (%.2f%% relative)", rms, rms_rel))
    else
        push!(lines, "q profile unavailable")
    end
    push!(lines, "")

    # ── Mercier criterion RMS ──
    push!(lines, "--- Mercier Criterion ---")
    fdi = fort["di"]
    fdr = fort["dr"]
    jdi = julia["di"]
    jdr = julia["dr"]
    if !isempty(jdi) && !isempty(fdi)
        jdi_on_fg = [_interp1(jq_psi, jdi, p) for p in fq_psi]
        jdr_on_fg = !isempty(jdr) ? [_interp1(jq_psi, jdr, p) for p in fq_psi] : fill(NaN, length(fq_psi))
        rms_di = sqrt(mean((fdi .- jdi_on_fg).^2))
        rms_dr = sqrt(mean((fdr .- jdr_on_fg).^2))
        push!(lines, @sprintf("RMS(di_fort - di_julia) = %.5e", rms_di))
        push!(lines, @sprintf("RMS(dr_fort - dr_julia) = %.5e", rms_dr))
    else
        push!(lines, "Mercier data unavailable")
    end
    push!(lines, "")

    # ── Resonant surfaces ──
    push!(lines, "--- Resonant Surface Comparison (n=$nn) ---")
    push!(lines, @sprintf("%-4s  %-8s  %-8s  %-14s  %-14s  %-6s  %-10s  %-10s  %-6s  %-7s  %-7s  %-6s",
        "q", "psi_fort", "psi_julia", "|Phi_res_fort|", "|Phi_res_julia|", "Rel%",
        "w_isl_fort", "w_isl_julia", "Rel%", "K_fort", "K_julia", "Rel%"))

    f_psi_rat = fort["psi_n_rational"]
    f_q_rat   = fort["q_rational"]
    f_phi_res = fort["Phi_res"]     # [n_surf] ComplexF64
    f_w_isl   = fort["w_isl"]       # [n_surf] full island width
    f_K_isl   = fort["K_isl"]       # [n_surf] Chirikov

    j_rat_psi = julia["rational_psi"]    # [n_rational]
    j_phi_res = julia["resonant_flux"]   # [n_rational] ComplexF64
    j_ihw     = julia["island_half_width"]  # [n_rational]
    j_chir    = julia["chirikov_parameter"] # [n_rational]

    for (i, (fq_r, fp_r)) in enumerate(zip(f_q_rat, f_psi_rat))
        q_int = round(Int, fq_r)
        row = find_julia_row(julia, q_int, nn)

        jp_r  = row > 0 && !isempty(j_rat_psi)  ? j_rat_psi[row]  : NaN
        j_phi = row > 0 && !isempty(j_phi_res)  ? abs(j_phi_res[row]) : NaN
        j_wisl= row > 0 && !isempty(j_ihw)      ? 2 * j_ihw[row]  : NaN
        j_kch = row > 0 && !isempty(j_chir)     ? j_chir[row]     : NaN

        f_phi  = abs(f_phi_res[i])
        f_wisl = f_w_isl[i]
        f_kch  = f_K_isl[i]

        push!(lines, @sprintf(
            "%-4d  %8.4f  %8s  %14.4e  %14s  %-6s  %10.4e  %10s  %-6s  %7.4f  %7s  %-6s",
            q_int, fp_r,
            isnan(jp_r)   ? "N/A" : @sprintf("%.4f", jp_r),
            f_phi,
            isnan(j_phi)  ? "N/A" : @sprintf("%.4e", j_phi),
            isnan(j_phi)  ? "N/A" : rel_pct(f_phi, j_phi),
            f_wisl,
            isnan(j_wisl) ? "N/A" : @sprintf("%.4e", j_wisl),
            isnan(j_wisl) ? "N/A" : rel_pct(f_wisl, j_wisl),
            f_kch,
            isnan(j_kch)  ? "N/A" : @sprintf("%.4f", j_kch),
            isnan(j_kch)  ? "N/A" : rel_pct(f_kch, j_kch),
        ))
    end

    push!(lines, "=" ^ 72)
    return lines
end

"""Simple linear interpolation of y(x) at point xi."""
function _interp1(x::AbstractVector, y::AbstractVector, xi::Real)
    n = length(x)
    n < 2 && return y[1]
    xi <= x[1]   && return y[1]
    xi >= x[end] && return y[end]
    i = searchsortedlast(x, xi)
    i = clamp(i, 1, n-1)
    t = (xi - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

# ─── Plot generation ─────────────────────────────────────────────────────────

function generate_plots(fort, julia, bench_dir, nn)
    p1 = plot(fort["psi_n_dcon"], fort["q_dcon"], label="Fortran", xlabel="ψ_n", ylabel="q", title="q profile")
    plot!(p1, julia["psi_q"], julia["q"], label="Julia", linestyle=:dash)

    p2 = plot(fort["psi_n_dcon"], fort["di"], label="Fortran di", xlabel="ψ_n", ylabel="di", title="Mercier di")
    !isempty(julia["di"]) && plot!(p2, julia["psi_q"], julia["di"], label="Julia di", linestyle=:dash)

    p3 = plot(fort["psi_n_dcon"], fort["dr"], label="Fortran dr", xlabel="ψ_n", ylabel="dr", title="Mercier dr")
    !isempty(julia["dr"]) && plot!(p3, julia["psi_q"], julia["dr"], label="Julia dr", linestyle=:dash)

    f_wt = fort["W_t_eigenvalue"]
    j_et = julia["et"]
    fv   = isempty(f_wt) ? NaN : abs(f_wt[argmin(real.(f_wt))])
    jv   = isempty(j_et) ? NaN : abs(j_et[1])
    p4   = bar(["Fortran", "Julia"], [fv, jv], title="|W_t| least stable", ylabel="|W_t|", legend=false)

    f_q_rat  = fort["q_rational"]
    n_surf   = length(f_q_rat)
    q_labels = [@sprintf("q≈%d", round(Int, q)) for q in f_q_rat]

    f_phi_vals  = abs.(fort["Phi_res"])
    j_phi_vals  = fill(NaN, n_surf)
    j_wisl_vals = fill(NaN, n_surf)
    j_kchir_vals= fill(NaN, n_surf)

    for (i, fq_r) in enumerate(f_q_rat)
        q_int = round(Int, fq_r)
        row = find_julia_row(julia, q_int, nn)
        if row > 0
            !isempty(julia["resonant_flux"])      && (j_phi_vals[i]   = abs(julia["resonant_flux"][row]))
            !isempty(julia["island_half_width"])  && (j_wisl_vals[i]  = 2 * julia["island_half_width"][row])
            !isempty(julia["chirikov_parameter"]) && (j_kchir_vals[i] = julia["chirikov_parameter"][row])
        end
    end

    p5 = groupedbar(
        hcat(f_phi_vals, j_phi_vals),
        xticks=(1:n_surf, q_labels),
        label=["Fortran" "Julia"],
        title="|Phi_res| per surface (n=$nn)", ylabel="|Phi_res|"
    )

    layout = @layout [a b c; d e]
    p = plot(p1, p2, p3, p4, p5; layout=layout, size=(1400, 700))
    outfile = joinpath(bench_dir, "comparison_plots.png")
    savefig(p, outfile)
    println("Plots saved to: ", abspath(outfile))
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main(argv=ARGS)
    opts = parse_args(argv)
    fortran_dir = opts.fortran_dir
    do_plot     = opts.do_plot
    skip_run    = opts.skip_run

    isdir(fortran_dir) || error("Fortran directory not found: $fortran_dir")

    equil_file = joinpath(fortran_dir, "equil.in")
    dcon_file  = joinpath(fortran_dir, "dcon.in")
    isfile(equil_file) || error("equil.in not found in $fortran_dir")
    isfile(dcon_file)  || error("dcon.in not found in $fortran_dir")

    eq = parse_namelist(equil_file)
    dc = parse_namelist(dcon_file)

    nn          = nl_get(dc, "nn",          Int,     1)
    delta_mlow  = nl_get(dc, "delta_mlow",  Int,     8)
    delta_mhigh = nl_get(dc, "delta_mhigh", Int,     8)
    dmlim       = nl_get(dc, "dmlim",       Float64, 0.2)
    qlow        = nl_get(dc, "qlow",        Float64, 1.02)
    psiedge     = nl_get(dc, "psiedge",     Float64, 1.0)
    eq_filename = nl_get(eq, "eq_filename", String,  "")

    println("Parsed Fortran inputs: nn=$nn, delta_mlow=$delta_mlow, delta_mhigh=$delta_mhigh")
    println("  dmlim=$dmlim, qlow=$qlow, psiedge=$psiedge")
    println("  eq_filename=$eq_filename")

    println("\nLoading Fortran NetCDF outputs...")
    fort = load_fortran_outputs(fortran_dir, nn)
    println("  psilim=$(fort["psilim"]), qlim=$(fort["qlim"]), $(length(fort["psi_n_rational"])) rational surfaces")

    bench_name = basename(fortran_dir) * "_resonant_response_benchmark"
    bench_dir  = joinpath(homedir(), "Code", "benchmarks_for_gpec_fortran_to_julia", bench_name)
    mkpath(bench_dir)
    println("\nBenchmark directory: $bench_dir")

    if !isempty(eq_filename)
        src = joinpath(fortran_dir, eq_filename)
        dst = joinpath(bench_dir, eq_filename)
        if isfile(src) && !isfile(dst)
            cp(src, dst)
            println("Copied equilibrium file: $eq_filename")
        end
    end

    forcing_path = joinpath(bench_dir, "forcing.dat")
    write_forcing_dat(forcing_path, nn, fort["m_vals"], fort["Phi_coil_amp"])
    println("Wrote forcing.dat ($(length(fort["m_vals"])) modes)")

    toml_path = joinpath(bench_dir, "gpec.toml")
    write_gpec_toml(toml_path, fortran_dir, eq, nn, dmlim, qlow, psiedge, delta_mlow, delta_mhigh)
    println("Wrote gpec.toml")

    h5_path = joinpath(bench_dir, "gpec.h5")
    if skip_run
        println("\n--skip-run: skipping Julia GPEC run")
    else
        println("\nRunning Julia GPEC (subprocess)...")
        # Remove stale output so Julia doesn't append to old data
        isfile(h5_path) && rm(h5_path)
        gpec_root = joinpath(@__DIR__, "..")
        julia_cmd = Base.julia_cmd()
        cmd = `$julia_cmd --project=$gpec_root -e "using GeneralizedPerturbedEquilibrium; GeneralizedPerturbedEquilibrium.main([\"$bench_dir\"])"`
        run(cmd)
    end

    println("\nLoading Julia outputs from $h5_path...")
    julia = load_julia_outputs(h5_path)
    println("  psilim=$(julia["psilim"]), qlim=$(julia["qlim"])")
    n_rat = length(julia["rational_psi"])
    println("  $n_rat resonant (surface, n) pairs found")

    println()
    table_lines = build_comparison_table(fort, julia, fortran_dir, bench_dir, nn)
    for line in table_lines
        println(line)
    end

    table_path = joinpath(bench_dir, "comparison_table.txt")
    open(table_path, "w") do io
        for line in table_lines
            println(io, line)
        end
    end
    println("\nComparison table saved to: ", abspath(table_path))

    if do_plot
        generate_plots(fort, julia, bench_dir, nn)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
