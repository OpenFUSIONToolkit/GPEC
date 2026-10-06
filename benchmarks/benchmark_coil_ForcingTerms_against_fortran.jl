"""
Coil forcing pipeline validation against a Fortran GPEC run.

Reads all parameters (coil currents, grid resolution, toroidal mode number,
equilibrium settings) directly from an existing Fortran GPEC run directory,
runs the Julia Biot-Savart pipeline, and produces a 4-panel diagnostic figure:

  Panel 1 (top-left):  Coil cross-sections (R, Z) + plasma boundary
  Panel 2 (top-right): Normal field bn(θ, ζ) on the plasma boundary
  Panel 3 (bot-left):  Julia vs Fortran |Phi_x| mode spectrum
  Panel 4 (bot-right): Fortran Phi_x / Julia ratio (flat at 1.0 → correct)

Usage:
    julia --project=. benchmarks/check_coil_pipeline.jl [/path/to/fortran/run]

The Fortran run directory (a DIII-D ideal example run) is taken from the command
line, or from the GPEC_FORTRAN_DIR environment variable if no argument is given.
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Analysis.CoilForcing
using NCDatasets
using Plots
using Printf

const OUTPUT_DIR  = joinpath(@__DIR__, "coil_pipeline")
const COIL_DAT_DIR = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries")

# ---------------------------------------------------------------------------
# Fortran namelist parsing helpers
# ---------------------------------------------------------------------------

function _strip_fortran_comments(text::String)
    join([split(line, '!')[1] for line in split(text, '\n')], '\n')
end

function _find_scalar(text::String, key::String; default=nothing)
    m = match(Regex("\\b$(key)\\s*=\\s*([+-]?[\\d.]+(?:[eE][+-]?\\d+)?)", "i"), text)
    isnothing(m) ? default : parse(Float64, m.captures[1])
end

function _find_int(text::String, key::String; default::Int=0)
    v = _find_scalar(text, key; default=Float64(default))
    isnothing(v) ? default : Int(v)
end

function _find_string(text::String, key::String; default::String="")
    m = match(Regex("\\b$(key)\\s*=\\s*\"([^\"]+)\"", "i"), text)
    isnothing(m) ? default : strip(m.captures[1])
end

function _find_indexed_string(text::String, name::String, i::Int; default::String="")
    m = match(Regex("\\b$(name)\\s*\\(\\s*$(i)\\s*\\)\\s*=\\s*\"([^\"]+)\"", "i"), text)
    isnothing(m) ? default : strip(m.captures[1])
end

function _find_indexed2_float(text::String, name::String, i::Int, j::Int; default=nothing)
    m = match(Regex("\\b$(name)\\s*\\(\\s*$(i)\\s*,\\s*$(j)\\s*\\)\\s*=\\s*([+-]?[\\d.]+(?:[eE][+-]?\\d+)?)", "i"), text)
    isnothing(m) ? default : parse(Float64, m.captures[1])
end

# ---------------------------------------------------------------------------
# Parse the Fortran run directory
# ---------------------------------------------------------------------------

struct FortranRunParams
    # Equilibrium
    eq_type::String
    eq_filename::String
    jac_type::String
    psilow::Float64
    psihigh::Float64
    mtheta_equil::Int
    grid_type::String
    # Mode number
    nn::Int
    delta_mlow::Int
    delta_mhigh::Int
    # Coil grid
    mtheta_coil::Int
    nzeta_coil::Int
    machine::String
    # Coil sets: name → current vector
    coil_names::Vector{String}
    coil_currents::Vector{Vector{Float64}}
end

function parse_fortran_run(dir::String)::FortranRunParams
    equil_file = joinpath(dir, "equil.in")
    dcon_file  = joinpath(dir, "dcon.in")
    coil_file  = joinpath(dir, "coil.in")

    isfile(equil_file) || error("equil.in not found in $dir")
    isfile(dcon_file)  || error("dcon.in not found in $dir")
    isfile(coil_file)  || error("coil.in not found in $dir")

    eq_text   = _strip_fortran_comments(read(equil_file, String))
    dcon_text = _strip_fortran_comments(read(dcon_file,  String))
    coil_text = _strip_fortran_comments(read(coil_file,  String))

    # Equilibrium parameters
    eq_type   = _find_string(eq_text, "eq_type";   default="efit")
    eq_file   = _find_string(eq_text, "eq_filename"; default="")
    jac_type  = _find_string(eq_text, "jac_type";  default="hamada")
    psilow    = _find_scalar(eq_text, "psilow";    default=1e-4)
    psihigh   = _find_scalar(eq_text, "psihigh";   default=0.99)
    mtheta_eq = _find_int(eq_text,    "mtheta";    default=256)
    grid_type = _find_string(eq_text, "grid_type"; default="ldp")

    # Toroidal mode number and m-range expansion
    nn          = _find_int(dcon_text, "nn";          default=1)
    delta_mlow  = _find_int(dcon_text, "delta_mlow";  default=8)
    delta_mhigh = _find_int(dcon_text, "delta_mhigh"; default=8)

    # Coil grid parameters
    mtheta_coil = _find_int(coil_text, "cmtheta"; default=480)
    nzeta_coil  = _find_int(coil_text, "cmzeta";  default=40)
    machine     = _find_string(coil_text, "machine"; default="d3d")
    coil_num    = _find_int(coil_text, "coil_num"; default=1)

    coil_names    = String[]
    coil_currents = Vector{Float64}[]
    for ci in 1:coil_num
        name = _find_indexed_string(coil_text, "coil_name", ci; default="coil$ci")
        currents = Float64[]
        for cj in 1:48
            cur = _find_indexed2_float(coil_text, "coil_cur", ci, cj)
            isnothing(cur) && break
            push!(currents, cur)
        end
        push!(coil_names, name)
        push!(coil_currents, currents)
    end

    return FortranRunParams(eq_type, eq_file, jac_type, psilow, psihigh, mtheta_eq,
                            grid_type, nn, delta_mlow, delta_mhigh,
                            mtheta_coil, nzeta_coil, machine,
                            coil_names, coil_currents)
end

function find_output_nc(dir::String, nn::Int)
    path = joinpath(dir, "gpec_control_output_n$(nn).nc")
    isfile(path) || error("Fortran output not found: $path")
    return path
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

run_dir = length(ARGS) > 0 ? ARGS[1] : get(ENV, "GPEC_FORTRAN_DIR", "")
isempty(run_dir) && error("No Fortran run directory given: pass it as the first argument or set GPEC_FORTRAN_DIR")
isdir(run_dir) || error("Run directory not found: $run_dir")
mkpath(OUTPUT_DIR)

println("="^70)
println("Coil Pipeline Validation vs Fortran GPEC")
println("  Run dir: $run_dir")
println("="^70)

# ---------------------------------------------------------------------------
# [1/4] Parse Fortran inputs
# ---------------------------------------------------------------------------
println("\n[1/4] Parsing Fortran inputs ...")
p = parse_fortran_run(run_dir)

@printf "    eq:      %s  %s  (jac=%s, psilow=%.4f, psihigh=%.4f)\n" p.eq_type p.eq_filename p.jac_type p.psilow p.psihigh
@printf "    n:       %d  (delta_mlow=%d, delta_mhigh=%d)\n" p.nn p.delta_mlow p.delta_mhigh
@printf "    grid:    mtheta_coil=%d  nzeta_coil=%d\n" p.mtheta_coil p.nzeta_coil
@printf "    machine: %s  (%d coil sets)\n" p.machine length(p.coil_names)
for (name, curs) in zip(p.coil_names, p.coil_currents)
    @printf "      %-10s  currents: %s A\n" name string(round.(curs; digits=1))
end

nc_path = find_output_nc(run_dir, p.nn)

# Read Fortran output (psilim and Phi_x spectrum)
fortran_m      = Int[]
fortran_Phix   = Float64[]
fortran_psilim = NaN
NCDatasets.Dataset(nc_path, "r") do ds
    global fortran_m      = Int.(ds["m"][:])
    global fortran_psilim = Float64(ds.attrib["psilim"])
    phi_x_raw = ds["Phi_x"][:, :]   # (mpert, 2); col 1=cos, col 2=sin
    global fortran_Phix = sqrt.(phi_x_raw[:, 1].^2 .+ phi_x_raw[:, 2].^2)
end
mlow  = minimum(fortran_m)
mhigh = maximum(fortran_m)
@printf "    Fortran psilim=%.6f,  m=%d:%d\n" fortran_psilim mlow mhigh

# ---------------------------------------------------------------------------
# [2/4] Load equilibrium
# ---------------------------------------------------------------------------
println("\n[2/4] Loading equilibrium ...")
t_equil = @elapsed begin
    eq_dict = Dict{String,Any}(
        "eq_type"      => p.eq_type,
        "eq_filename"  => p.eq_filename,
        "jac_type"     => p.jac_type,
        "psilow"       => p.psilow,
        "psihigh"      => p.psihigh,
        "mtheta"       => p.mtheta_equil,
        "grid_type"    => p.grid_type,
        "psi_accuracy" => 0.001,
        "etol"         => 1e-7,
    )
    eq_config = Equilibrium.EquilibriumConfig(eq_dict, run_dir)
    equil = Equilibrium.setup_equilibrium(eq_config)
end
@printf "    Done in %.1f s  (bt_sign=%d)\n" t_equil equil.params.bt_sign

# ---------------------------------------------------------------------------
# [3/4] Load coil sets and run Biot-Savart at Fortran psilim surface
# ---------------------------------------------------------------------------
println("\n[3/4] Loading coil sets and computing Biot-Savart ...")
println("    grid: $(p.mtheta_coil)×$(p.nzeta_coil),  psi=$(round(fortran_psilim; digits=6))")

coil_sets = CoilSet[]
for (name, currents) in zip(p.coil_names, p.coil_currents)
    dat_path = joinpath(COIL_DAT_DIR, "$(p.machine)_$(name).dat")
    isfile(dat_path) || error("Coil geometry file not found: $dat_path")
    raw = read_coil_dat(dat_path)
    push!(coil_sets, CoilSet(raw.name, raw.ncoil, raw.s, raw.nw, raw.nsec,
                             raw.x, raw.y, raw.z, Float64.(currents[1:raw.ncoil])))
end
for cs in coil_sets
    @printf "    %-10s %d conductors  currents: %s A\n" cs.name cs.ncoil string(round.(cs.currents; digits=1))
end

julia_modes = ForcingMode[]
bnd_grid = nothing
bn       = nothing
t_biot   = @elapsed begin
    bnd_grid = sample_boundary_grid(equil, p.mtheta_coil, p.nzeta_coil; psi=fortran_psilim)
    nobs = p.mtheta_coil * p.nzeta_coil
    obs_R   = zeros(nobs); obs_phi = zeros(nobs); obs_Z = zeros(nobs)
    for j in 1:p.nzeta_coil, i in 1:p.mtheta_coil
        idx = i + (j - 1) * p.mtheta_coil
        obs_R[idx]   = bnd_grid.R[i]
        obs_phi[idx] = bnd_grid.phi_grid[j] + bnd_grid.phi_offset[i]
        obs_Z[idx]   = bnd_grid.Z[i]
    end
    B_R = zeros(nobs); B_phi = zeros(nobs); B_Z = zeros(nobs)
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)
    bn = zeros(p.mtheta_coil, p.nzeta_coil)
    project_normal_flux!(bn, B_R, B_Z, bnd_grid)
    append!(julia_modes, fourier_decompose_bn(bn, bnd_grid, p.nn, mlow, mhigh))
end
@printf "    Done in %.1f s,  max|Phi_x| = %.3e T·m²\n" t_biot maximum(abs, bn)

julia_m    = [md.m for md in sort(julia_modes; by=md -> md.m)]
julia_amps = abs.([md.amplitude for md in sort(julia_modes; by=md -> md.m)])

# Print comparison table
println("\n    m      Julia Phi_x [T·m²]   Fortran Phi_x [Wb]   ratio")
println("    " * "-"^55)
for (jm, ja) in zip(julia_m, julia_amps)
    k = findfirst(==(jm), fortran_m)
    if !isnothing(k)
        fp    = fortran_Phix[k]
        ratio = ja > 1e-40 ? fp / ja : NaN
        @printf "    %3d    %12.4e         %12.4e      %.4f\n" jm ja fp ratio
    end
end
println("    Expected ratio ≈ 1.000 for all modes.")

# ---------------------------------------------------------------------------
# [4/4] Generate 4-panel figure
# ---------------------------------------------------------------------------
println("\n[4/4] Generating figure ...")

function step_series(m_vals, amps)
    m_ext   = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end

# Panel 1: coil cross-sections (R, Z) + plasma boundary
p1 = plot_coil_geometry_rz(coil_sets; equil=equil, psi=fortran_psilim)
title!(p1, "Coil cross-sections  ($(p.machine), n=$(p.nn))")

# Panel 2: bn(θ, ζ) heatmap
p2 = plot_bn_contour(bn, p.mtheta_coil, p.nzeta_coil; n=p.nn)
title!(p2, "Normal flux Φₓ(θ, ζ)  at ψ=$(round(fortran_psilim; digits=4))")

# Panel 3: Julia vs Fortran Phi_x spectrum
jm_ext, ja_ext   = step_series(julia_m, julia_amps)
fm_ext, fa_ext   = step_series(fortran_m, fortran_Phix)
p3 = plot(; xlabel="Poloidal mode m", ylabel="|Phi_x| [T·m²]",
            title="Spectrum: Julia vs Fortran  (n=$(p.nn))",
            legend=:topright)
plot!(p3, jm_ext, ja_ext; seriestype=:steppre, lw=2, color=:blue,
      label="Julia ψ=$(round(fortran_psilim; digits=4))")
plot!(p3, fm_ext, fa_ext; seriestype=:steppre, lw=2, color=:orange,
      linestyle=:dash, label="Fortran Phi_x")
ylims!(p3, (0, Inf)); xlims!(p3, mlow - 2, mhigh + 2)

# Panel 4: Phi_x / Julia ratio
ratio_m    = Int[]
ratio_vals = Float64[]
for (jm, ja) in zip(julia_m, julia_amps)
    k = findfirst(==(jm), fortran_m)
    if !isnothing(k) && ja > 1e-40
        push!(ratio_m, jm)
        push!(ratio_vals, fortran_Phix[k] / ja)
    end
end
p4 = plot(; xlabel="Poloidal mode m", ylabel="Phi_x / Julia",
            title="Ratio (flat at 1.0 → correct)",
            legend=:topright)
if !isempty(ratio_m)
    scatter!(p4, ratio_m, ratio_vals; color=:blue, label="ratio", markersize=5)
    hline!(p4, [1.0]; color=:red, linestyle=:dash, label="expected = 1.0")
end
xlims!(p4, mlow - 2, mhigh + 2)

run_label = basename(run_dir)
fig = plot(p1, p2, p3, p4;
           layout=(2, 2), size=(1400, 900),
           plot_title="$run_label  |  $(p.machine)  n=$(p.nn)  psilim=$(round(fortran_psilim; digits=4))")

out_png = joinpath(OUTPUT_DIR, "coil_pipeline_$(run_label)_n$(p.nn).png")
savefig(fig, out_png)
println("    Saved → $out_png")
println("\n" * "="^70)
println("Done.")
