#!/usr/bin/env julia
"""
benchmark_against_fortran_run.jl - Compare Julia GPEC results against Fortran GPEC reference outputs

Reads Fortran output NetCDF files from a GPEC Fortran example directory, builds a
matched Julia configuration, optionally runs Julia GPEC, then prints a side-by-side
comparison table.

By default uses coil geometry files (harder test): Julia runs its own Biot-Savart
pipeline to produce the forcing, matching the Fortran coil.in configuration.  A
Phi_x spectrum comparison is included in the printout before the PE comparison.

# Usage

```bash
# Full run with coil forcing (default)
julia --project=benchmarks benchmarks/benchmark_fortran.jl /path/to/DIIID_ideal_example --plot

# Use Fortran Phi_coil directly as forcing (easier test)
julia --project=benchmarks benchmarks/benchmark_fortran.jl /path/to/DIIID_ideal_example --use-file-forcing

# Skip Julia re-run (use existing gpec.h5 in bench dir)
julia --project=benchmarks benchmarks/benchmark_fortran.jl /path/to/DIIID_ideal_example --skip-run
```

# Arguments

- `fortran_dir`         : Path to Fortran example directory (contains equil.in, dcon.in, coil.in, *.nc)
- `--plot`              : Generate comparison plots (saved to bench dir)
- `--skip-run`          : Skip running Julia GPEC (assume gpec.h5 already exists)
- `--use-file-forcing`  : Use Fortran Phi_coil directly as forcing.dat (bypasses coil pipeline)

# Outputs (written to bench dir)

- `gpec.toml`            : Julia configuration matched to Fortran inputs
- `forcing.dat`          : External forcing (file-forcing mode only)
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

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.Equilibrium
import GeneralizedPerturbedEquilibrium.Analysis.PerturbedEquilibriumModes

# ─── Argument parsing ────────────────────────────────────────────────────────

function parse_args(args)
    fortran_dir     = ""
    do_plot         = false
    skip_run        = false
    use_file_forcing = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--plot"
            do_plot = true
        elseif a == "--skip-run"
            skip_run = true
        elseif a == "--use-file-forcing"
            use_file_forcing = true
        elseif !startswith(a, "--")
            fortran_dir = a
        else
            error("Unknown argument: $a")
        end
        i += 1
    end
    isempty(fortran_dir) && error(
        "Usage: benchmark_fortran.jl <fortran_dir> [--plot] [--skip-run] [--use-file-forcing]")
    return (fortran_dir=abspath(fortran_dir), do_plot=do_plot,
            skip_run=skip_run, use_file_forcing=use_file_forcing)
end

# ─── Fortran namelist parsing ─────────────────────────────────────────────────

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
    m = match(Regex(
        "\\b$(name)\\s*\\(\\s*$(i)\\s*,\\s*$(j)\\s*\\)\\s*=\\s*([+-]?[\\d.]+(?:[eE][+-]?\\d+)?)", "i"),
        text)
    isnothing(m) ? default : parse(Float64, m.captures[1])
end

# ─── Fortran run parameter struct ─────────────────────────────────────────────

struct FortranRunParams
    eq_type::String
    eq_filename::String
    jac_type::String
    grid_type::String
    psilow::Float64
    psihigh::Float64
    mpsi::Int
    mtheta_equil::Int
    nn::Int
    delta_mlow::Int
    delta_mhigh::Int
    dmlim::Float64
    qlow::Float64
    psiedge::Float64
    mtheta_coil::Int
    nzeta_coil::Int
    machine::String
    coil_names::Vector{String}
    coil_currents::Vector{Vector{Float64}}
    euler_stride::Int
end

function parse_fortran_run(dir::String)
    eq_text   = _strip_fortran_comments(read(joinpath(dir, "equil.in"), String))
    dcon_text = _strip_fortran_comments(read(joinpath(dir, "dcon.in"),  String))
    coil_text = isfile(joinpath(dir, "coil.in")) ?
        _strip_fortran_comments(read(joinpath(dir, "coil.in"), String)) : ""

    eq_type   = _find_string(eq_text, "eq_type";   default="efit")
    eq_file   = _find_string(eq_text, "eq_filename"; default="")
    jac_type  = _find_string(eq_text, "jac_type";  default="hamada")
    grid_type = _find_string(eq_text, "grid_type"; default="ldp")
    psilow    = _find_scalar(eq_text, "psilow";    default=1e-4)
    psihigh   = _find_scalar(eq_text, "psihigh";   default=0.993)
    mpsi      = _find_int(eq_text,    "mpsi";      default=128)
    mtheta_eq = _find_int(eq_text,    "mtheta";    default=256)

    nn          = _find_int(dcon_text, "nn";           default=1)
    delta_mlow  = _find_int(dcon_text, "delta_mlow";  default=8)
    delta_mhigh = _find_int(dcon_text, "delta_mhigh"; default=8)
    dmlim       = _find_scalar(dcon_text, "dmlim";    default=0.2)
    qlow        = _find_scalar(dcon_text, "qlow";     default=1.02)
    psiedge     = _find_scalar(dcon_text, "psiedge";  default=1.0)
    euler_stride = _find_int(dcon_text, "euler_stride"; default=1)

    mtheta_coil = _find_int(coil_text, "cmtheta";  default=480)
    nzeta_coil  = _find_int(coil_text, "cmzeta";   default=40)
    machine     = _find_string(coil_text, "machine"; default="d3d")
    coil_num    = _find_int(coil_text, "coil_num"; default=0)

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

    return FortranRunParams(eq_type, eq_file, jac_type, grid_type, psilow, psihigh, mpsi,
        mtheta_eq, nn, delta_mlow, delta_mhigh, dmlim, qlow, psiedge,
        mtheta_coil, nzeta_coil, machine, coil_names, coil_currents, euler_stride)
end

# ─── Load Fortran NC outputs ─────────────────────────────────────────────────

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

        wt_ev = Float64.(ds["W_t_eigenvalue"][:, :])
        fort["W_t_eigenvalue"] = complex.(wt_ev[:, 1], wt_ev[:, 2])
    end

    NCDataset(profile_file, "r") do ds
        fort["psi_n_rational"] = Float64.(ds["psi_n_rational"][:])
        fort["q_rational"]     = Float64.(ds["q_rational"][:])

        phi_res = Float64.(ds["Phi_res"][:, :])
        fort["Phi_res"] = complex.(phi_res[:, 1], phi_res[:, 2])

        delta = Float64.(ds["Delta"][:, :])
        fort["Delta"] = complex.(delta[:, 1], delta[:, 2])

        fort["w_isl"] = Float64.(ds["w_isl"][:])
        fort["K_isl"] = Float64.(ds["K_isl"][:])
        if haskey(ds, "I_res")
            i_res_raw = Float64.(ds["I_res"][:, :])
            fort["I_res"] = complex.(i_res_raw[:, 1], i_res_raw[:, 2])
        else
            fort["I_res"] = ComplexF64[]
        end

        # radial profiles vs psi for all modes
        fort["psi_n_prof"] = Float64.(ds["psi_n"][:])
        fort["m_out"]      = Int.(ds["m_out"][:])
        b_n_raw = Float64.(ds["b_n"][:, :, :])
        fort["b_n"] = complex.(b_n_raw[:, :, 1], b_n_raw[:, :, 2])
        jbgp_raw = Float64.(ds["Jbgradpsi"][:, :, :])
        fort["Jbgradpsi"] = complex.(jbgp_raw[:, :, 1], jbgp_raw[:, :, 2])
        xigp_raw = Float64.(ds["xigradpsi"][:, :, :])
        fort["xigradpsi"] = complex.(xigp_raw[:, :, 1], xigp_raw[:, :, 2])
        xi_n_raw_f = Float64.(ds["xi_n"][:, :, :])
        fort["xi_n"] = complex.(xi_n_raw_f[:, :, 1], xi_n_raw_f[:, :, 2])

        # Clebsch displacement profiles (for PENTRC benchmark)
        if haskey(ds, "xigradpsi_dpsi")
            xigp1_raw = Float64.(ds["xigradpsi_dpsi"][:, :, :])
            fort["xigradpsi_dpsi"] = complex.(xigp1_raw[:, :, 1], xigp1_raw[:, :, 2])
        else
            fort["xigradpsi_dpsi"] = Matrix{ComplexF64}(undef, 0, 0)
        end
        if haskey(ds, "xigradalpha")
            xiga_raw = Float64.(ds["xigradalpha"][:, :, :])
            fort["xigradalpha"] = complex.(xiga_raw[:, :, 1], xiga_raw[:, :, 2])
        else
            fort["xigradalpha"] = Matrix{ComplexF64}(undef, 0, 0)
        end
    end

    NCDataset(control_file, "r") do ds
        # Phi_coil: NC dim (i=2, coil_index=ncoil, m=mpert)
        phi_coil = Float64.(ds["Phi_coil"][:, :, :])
        amp_real = dropdims(sum(phi_coil[:, :, 1], dims=2), dims=2)
        amp_imag = dropdims(sum(phi_coil[:, :, 2], dims=2), dims=2)
        fort["Phi_coil_amp"] = complex.(amp_real, amp_imag)

        phi_x = Float64.(ds["Phi_x"][:, :])
        fort["Phi_x_amp"] = abs.(complex.(phi_x[:, 1], phi_x[:, 2]))
        fort["Phi_x"]     = complex.(phi_x[:, 1], phi_x[:, 2])

        phi_tot = Float64.(ds["Phi"][:, :])
        fort["Phi_tot"] = complex.(phi_tot[:, 1], phi_tot[:, 2])

        fort["m_vals_ctrl"] = Int.(ds["m"][:])

        # Control surface matrices (resp_index=0): NC shape (mode, mode, 2) in Julia
        for (key, nc_var) in [("P", "P"), ("Lambda", "Lambda"), ("L_surf", "L"), ("rho", "rho")]
            if haskey(ds, nc_var)
                raw = Float64.(ds[nc_var][:, :, :])  # (mode, mode, 2) → last dim real/imag
                fort[key] = complex.(raw[:, :, 1], raw[:, :, 2])
            else
                fort[key] = Matrix{ComplexF64}(undef, 0, 0)
            end
        end
    end

    # R,Z,φ profile data — store as theta-space complex arrays (from gpout_xbrzphifun)
    # Fortran outputs (psi, theta+1, 2) where dim 3 = (real, imag) in machine φ
    NCDataset(profile_file, "r") do ds
        if haskey(ds, "xi_r")
            ntheta = length(ds["theta_dcon"]) - 1  # drop duplicate endpoint
            for (key, nc_var) in [("xi_r_fun", "xi_r"), ("xi_z_fun", "xi_z"), ("xi_phi_fun", "xi_phi"),
                                  ("b_r_fun",  "b_r"),  ("b_z_fun",  "b_z"),  ("b_phi_fun",  "b_phi")]
                raw = Float64.(ds[nc_var][:, :, :])  # (psi, theta+1, 2)
                fort[key] = complex.(raw[:, 1:ntheta, 1], raw[:, 1:ntheta, 2])
            end
        end
    end

    return fort
end

# ─── Julia h5 loader ─────────────────────────────────────────────────────────

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

        julia["psio"] = haskey(f, "equil/psio") ? read(f, "equil/psio") : NaN

        sc = "perturbed_equilibrium/singular_coupling"
        julia["rational_psi"]       = haskey(f, "$sc/rational_psi")       ? read(f, "$sc/rational_psi")       : Float64[]
        julia["rational_q"]         = haskey(f, "$sc/rational_q")         ? read(f, "$sc/rational_q")         : Float64[]
        julia["rational_n"]         = haskey(f, "$sc/rational_n")         ? read(f, "$sc/rational_n")         : Int[]
        julia["rational_m_res"]     = haskey(f, "$sc/rational_m_res")     ? read(f, "$sc/rational_m_res")     : Int[]
        julia["resonant_flux"]      = haskey(f, "$sc/resonant_flux")      ? read(f, "$sc/resonant_flux")      : ComplexF64[]
        julia["resonant_current"]   = haskey(f, "$sc/resonant_current")   ? read(f, "$sc/resonant_current")   : ComplexF64[]
        julia["island_half_width"]  = haskey(f, "$sc/island_half_width")  ? read(f, "$sc/island_half_width")  : Float64[]
        julia["chirikov_parameter"] = haskey(f, "$sc/chirikov_parameter") ? read(f, "$sc/chirikov_parameter") : Float64[]
        julia["delta_prime"]        = haskey(f, "$sc/delta_prime")        ? read(f, "$sc/delta_prime")        : ComplexF64[]

        pe = "perturbed_equilibrium"
        julia["forcing_vec"]  = haskey(f, "$pe/forcing_vec")  ? read(f, "$pe/forcing_vec")  : ComplexF64[]
        julia["response_vec"] = haskey(f, "$pe/response_vec") ? read(f, "$pe/response_vec") : ComplexF64[]
        julia["b_n"]          = haskey(f, "$pe/response/b_n")        ? read(f, "$pe/response/b_n")        : Matrix{ComplexF64}(undef, 0, 0)
        julia["Jbgradpsi"]    = haskey(f, "$pe/response/psi_area")    ? read(f, "$pe/response/psi_area")    : Matrix{ComplexF64}(undef, 0, 0)
        julia["xi_psi"]       = haskey(f, "$pe/response/xi_psi")     ? read(f, "$pe/response/xi_psi")     : Matrix{ComplexF64}(undef, 0, 0)
        julia["xi_n"]         = haskey(f, "$pe/response/xi_n")       ? read(f, "$pe/response/xi_n")       : Matrix{ComplexF64}(undef, 0, 0)
        julia["clebsch_psi1"] = haskey(f, "$pe/response/clebsch_psi1")  ? read(f, "$pe/response/clebsch_psi1")  : Matrix{ComplexF64}(undef, 0, 0)
        julia["clebsch_alpha"]= haskey(f, "$pe/response/clebsch_alpha") ? read(f, "$pe/response/clebsch_alpha") : Matrix{ComplexF64}(undef, 0, 0)
        julia["psi_grid"]     = haskey(f, "integration/psi")    ? read(f, "integration/psi")    : Float64[]

        # R,Z,φ: loaded via modes_to_theta helper below (not raw modes)
        julia["h5_path"] = h5_path  # stash for modes_to_theta
        # mn_index[:, 1] = m values, mn_index[:, 2] = n values for each mode index
        julia["m_modes"]      = haskey(f, "info/mn_index") ? Int.(read(f, "info/mn_index")[:, 1]) : Int[]

        # Control surface matrices
        rm = "$pe/response_matrices"
        julia["permeability"]       = haskey(f, "$rm/permeability")       ? read(f, "$rm/permeability")       : Matrix{ComplexF64}(undef, 0, 0)
        julia["plasma_inductance"]  = haskey(f, "$rm/plasma_inductance")  ? read(f, "$rm/plasma_inductance")  : Matrix{ComplexF64}(undef, 0, 0)
        julia["surface_inductance"] = haskey(f, "$rm/surface_inductance") ? read(f, "$rm/surface_inductance") : Matrix{ComplexF64}(undef, 0, 0)
        julia["reluctance"]         = haskey(f, "$rm/reluctance")         ? read(f, "$rm/reluctance")         : Matrix{ComplexF64}(undef, 0, 0)
        julia["wt0"]                = haskey(f, "vacuum/wt0")             ? read(f, "vacuum/wt0")             : Matrix{ComplexF64}(undef, 0, 0)
    end
    return julia
end

# ─── Write forcing.dat (file-forcing mode only) ───────────────────────────────

"""
Write forcing.dat from Phi_coil amplitudes in unit-norm convention.

Fortran Phi_coil is in unit-norm convention (= Julia's internal Phi_x).  To
round-trip correctly through Julia's file loader (which applies ×(2π)² for
sfl_flux_Wb), we divide by (2π)² before writing and tag the file accordingly.
"""
function write_forcing_dat(filepath::String, n::Int, m_vals::Vector{Int}, amp::Vector{ComplexF64})
    scale = 1.0 / (2π)^2
    open(filepath, "w") do io
        println(io, "# normalization: sfl_flux_Wb")
        println(io, "# n  m  real(Phi_coil_sum / (2π)²)  imag(Phi_coil_sum / (2π)²)")
        for (m, a) in zip(m_vals, amp)
            @printf(io, "%4d  %4d  %+.8e  %+.8e\n", n, m, real(a) * scale, imag(a) * scale)
        end
    end
end

# ─── Write gpec.toml (coil mode) ─────────────────────────────────────────────

function write_gpec_toml_coil(
    filepath::String,
    fortran_dir::String,
    p::FortranRunParams,
    coil_dat_dir::String
)
    open(filepath, "w") do io
        println(io, "# Auto-generated by benchmark_fortran.jl (coil forcing mode)")
        println(io, "# Matched to Fortran run: $fortran_dir")
        println(io)
        println(io, "[Equilibrium]")
        println(io, "eq_type     = \"$(p.eq_type)\"")
        println(io, "eq_filename = \"$(p.eq_filename)\"")
        println(io, "jac_type    = \"$(p.jac_type)\"")
        println(io, "grid_type   = \"$(p.grid_type)\"")
        @printf(io, "psilow      = %.4e\n", p.psilow)
        @printf(io, "psihigh     = %.6f\n", p.psihigh)
        println(io, "mpsi        = $(p.mpsi)")
        println(io, "mtheta      = $(p.mtheta_equil)")
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
        @printf(io, "dmlim    = %.4f\n", p.dmlim)
        @printf(io, "qlow     = %.4f\n", p.qlow)
        @printf(io, "psiedge  = %.4f\n", p.psiedge)
        println(io, "nn_low   = $(p.nn)")
        println(io, "nn_high  = $(p.nn)")
        println(io, "delta_mlow  = $(p.delta_mlow)")
        println(io, "delta_mhigh = $(p.delta_mhigh)")
        println(io, "mthvac   = 512")
        println(io, "eulerlagrange_tolerance = 1e-7")
        println(io, "singfac_min = 1e-4")
        println(io, "save_interval = $(p.euler_stride)")
        println(io)
        println(io, "[ForcingTerms]")
        println(io, "forcing_data_format = \"coil\"")
        println(io, "mtheta_coil         = $(p.mtheta_coil)")
        println(io, "nzeta_coil          = $(p.nzeta_coil)")
        for (name, currents) in zip(p.coil_names, p.coil_currents)
            dat_path = joinpath(coil_dat_dir, "$(p.machine)_$(name).dat")
            println(io)
            println(io, "[[ForcingTerms.coil_set]]")
            println(io, "dat_file = \"$(dat_path)\"")
            println(io, "currents = $(currents)")
        end
        println(io)
        println(io, "[PerturbedEquilibrium]")
        println(io, "fixed_boundary            = false")
        println(io, "compute_response          = true")
        println(io, "compute_singular_coupling = true")
        println(io, "verbose                   = true")
        println(io, "write_outputs_to_HDF5     = true")
    end
end

# ─── Write gpec.toml (file-forcing mode) ─────────────────────────────────────

function write_gpec_toml_file(
    filepath::String,
    fortran_dir::String,
    p::FortranRunParams
)
    open(filepath, "w") do io
        println(io, "# Auto-generated by benchmark_fortran.jl (file forcing mode)")
        println(io, "# Matched to Fortran run: $fortran_dir")
        println(io)
        println(io, "[Equilibrium]")
        println(io, "eq_type     = \"$(p.eq_type)\"")
        println(io, "eq_filename = \"$(p.eq_filename)\"")
        println(io, "jac_type    = \"$(p.jac_type)\"")
        println(io, "grid_type   = \"$(p.grid_type)\"")
        @printf(io, "psilow      = %.4e\n", p.psilow)
        @printf(io, "psihigh     = %.6f\n", p.psihigh)
        println(io, "mpsi        = $(p.mpsi)")
        println(io, "mtheta      = $(p.mtheta_equil)")
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
        @printf(io, "dmlim    = %.4f\n", p.dmlim)
        @printf(io, "qlow     = %.4f\n", p.qlow)
        @printf(io, "psiedge  = %.4f\n", p.psiedge)
        println(io, "nn_low   = $(p.nn)")
        println(io, "nn_high  = $(p.nn)")
        println(io, "delta_mlow  = $(p.delta_mlow)")
        println(io, "delta_mhigh = $(p.delta_mhigh)")
        println(io, "mthvac   = 512")
        println(io, "eulerlagrange_tolerance = 1e-7")
        println(io, "singfac_min = 1e-4")
        println(io, "save_interval = $(p.euler_stride)")
        println(io)
        println(io, "[ForcingTerms]")
        println(io, "forcing_data_file   = \"forcing.dat\"")
        println(io, "forcing_data_format = \"ascii\"")
        println(io)
        println(io, "[PerturbedEquilibrium]")
        println(io, "fixed_boundary            = false")
        println(io, "compute_response          = true")
        println(io, "compute_singular_coupling = true")
        println(io, "verbose                   = true")
        println(io, "write_outputs_to_HDF5     = true")
    end
end

# ─── Equilibrium setup helper ─────────────────────────────────────────────────

function setup_equil(p::FortranRunParams, fortran_dir::String)
    eq_dict = Dict{String,Any}(
        "eq_type"      => p.eq_type,
        "eq_filename"  => p.eq_filename,
        "jac_type"     => p.jac_type,
        "psilow"       => p.psilow,
        "psihigh"      => p.psihigh,
        "mpsi"         => p.mpsi,
        "mtheta"       => p.mtheta_equil,
        "grid_type"    => p.grid_type,
        "etol"         => 1e-7,
    )
    eq_config = Equilibrium.EquilibriumConfig(eq_dict, fortran_dir)
    return Equilibrium.setup_equilibrium(eq_config)
end

# ─── Phi_x comparison (coil mode) ────────────────────────────────────────────

"""
Compute Julia Phi_x via Biot-Savart and compare with Fortran Phi_coil.
Returns (julia_amps, fortran_amps, m_vals, equil).
"""
function compare_phix(
    p::FortranRunParams,
    fort::Dict,
    fortran_dir::String,
    coil_dat_dir::String
)
    println("  Loading equilibrium for Phi_x check...")
    equil = setup_equil(p, fortran_dir)

    psilim = fort["psilim"]
    m_vals = fort["m_vals"]
    mlow   = minimum(m_vals)
    mhigh  = maximum(m_vals)

    coil_sets = ForcingTerms.CoilSet[]
    for (name, currents) in zip(p.coil_names, p.coil_currents)
        dat_path = joinpath(coil_dat_dir, "$(p.machine)_$(name).dat")
        isfile(dat_path) || error("Coil dat file not found: $dat_path")
        raw = ForcingTerms.read_coil_dat(dat_path)
        push!(coil_sets, ForcingTerms.CoilSet(raw.name, raw.ncoil, raw.s, raw.nw, raw.nsec,
            raw.x, raw.y, raw.z, Float64.(currents[1:raw.ncoil])))
    end

    grid = ForcingTerms.sample_boundary_grid(equil, p.mtheta_coil, p.nzeta_coil; psi=psilim)
    nobs = p.mtheta_coil * p.nzeta_coil
    obs_R   = zeros(nobs); obs_phi = zeros(nobs); obs_Z = zeros(nobs)
    for j in 1:p.nzeta_coil, i in 1:p.mtheta_coil
        idx = i + (j - 1) * p.mtheta_coil
        obs_R[idx]   = grid.R[i]
        obs_phi[idx] = grid.phi_grid[j] + grid.phi_offset[i]
        obs_Z[idx]   = grid.Z[i]
    end
    B_R = zeros(nobs); B_phi = zeros(nobs); B_Z = zeros(nobs)
    ForcingTerms.compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)
    bn = zeros(p.mtheta_coil, p.nzeta_coil)
    ForcingTerms.project_normal_flux!(bn, B_R, B_Z, grid)
    julia_modes = ForcingTerms.fourier_decompose_bn(bn, grid, p.nn, mlow, mhigh)

    julia_amps = Dict(md.m => abs(md.amplitude) for md in julia_modes)
    fort_amps  = Dict(m => fort["Phi_x_amp"][k] for (k, m) in enumerate(m_vals))

    return julia_amps, fort_amps, m_vals, equil
end

# ─── Comparison table ────────────────────────────────────────────────────────

rel_pct(a, b) = abs(b) > 1e-30 ? @sprintf("%.1f%%", 100 * abs(a - b) / abs(b)) : "N/A"

function step_series(m_vals, amps)
    m_ext   = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end

function find_julia_row(julia, q_int::Int, nn::Int)
    rat_q = julia["rational_q"]
    rat_n = julia["rational_n"]
    isempty(rat_q) && return 0
    for row in eachindex(rat_q)
        row > length(rat_n) && break
        rat_n[row] == nn && round(Int, rat_q[row]) == q_int && return row
    end
    return 0
end

function build_comparison_table(fort, julia, fortran_dir, bench_dir, nn)
    lines = String[]
    push!(lines, "=" ^ 72)
    push!(lines, "GPEC FORTRAN vs JULIA COMPARISON")
    push!(lines, "=" ^ 72)
    push!(lines, "Fortran : $fortran_dir")
    push!(lines, "Julia   : $(joinpath(bench_dir, "gpec.h5"))")
    push!(lines, "Date    : $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))")
    push!(lines, "")

    push!(lines, "--- Scalar Parameters ---")
    push!(lines, @sprintf("%-12s  %10s  %10s  %10s  %6s", "", "Fortran", "Julia", "AbsDiff", "Rel%"))
    for (label, fv, jv) in [
        ("psilim", fort["psilim"], julia["psilim"]),
        ("qlim",   fort["qlim"],   julia["qlim"]),
    ]
        push!(lines, @sprintf("%-12s  %10.6f  %10.6f  %10.6f  %6s",
            label, fv, jv, abs(fv - jv), rel_pct(fv, jv)))
    end
    push!(lines, "")

    push!(lines, "--- W_t Eigenvalue (least stable mode) ---")
    push!(lines, @sprintf("%-8s  %10s  %10s  %6s", "", "Fortran", "Julia", "Rel%"))
    f_wt = fort["W_t_eigenvalue"]
    j_et = julia["et"]
    fv = isempty(f_wt) ? NaN : f_wt[argmin(real.(f_wt))]
    jv = isempty(j_et) ? NaN : j_et[1]
    push!(lines, @sprintf("|W_t|    %10.4f  %10.4f  %6s", abs(fv), abs(jv), rel_pct(abs(fv), abs(jv))))
    push!(lines, @sprintf("Re(W_t)  %10.4f  %10.4f  %6s", real(fv), real(jv), rel_pct(real(fv), real(jv))))
    push!(lines, "")

    push!(lines, "--- q Profile (RMS on common grid) ---")
    fq_psi = fort["psi_n_dcon"]
    fq     = fort["q_dcon"]
    jq_psi = julia["psi_q"]
    jq     = julia["q"]
    if !isempty(jq) && !isempty(fq)
        jq_on_fgrid = [_interp1(jq_psi, jq, p) for p in fq_psi]
        rms = sqrt(mean((fq .- jq_on_fgrid).^2))
        push!(lines, @sprintf("RMS(q_fort - q_julia) = %.5f  (%.2f%% relative)",
            rms, rms / mean(abs.(fq)) * 100))
    else
        push!(lines, "q profile unavailable")
    end
    push!(lines, "")

    push!(lines, "--- Mercier Criterion ---")
    fdi = fort["di"]; fdr = fort["dr"]
    jdi = julia["di"]; jdr = julia["dr"]
    if !isempty(jdi) && !isempty(fdi)
        jdi_on_fg = [_interp1(jq_psi, jdi, p) for p in fq_psi]
        jdr_on_fg = !isempty(jdr) ?
            [_interp1(jq_psi, jdr, p) for p in fq_psi] : fill(NaN, length(fq_psi))
        push!(lines, @sprintf("RMS(di_fort - di_julia) = %.5e", sqrt(mean((fdi .- jdi_on_fg).^2))))
        push!(lines, @sprintf("RMS(dr_fort - dr_julia) = %.5e", sqrt(mean((fdr .- jdr_on_fg).^2))))
    else
        push!(lines, "Mercier data unavailable")
    end
    push!(lines, "")

    push!(lines, "--- Phi_x and Phi_tot Spectrum (n=$nn) ---")
    m_ctrl   = fort["m_vals_ctrl"]
    f_phi_x  = fort["Phi_x"]
    f_phi_tot = fort["Phi_tot"]
    j_fvec   = julia["forcing_vec"]
    j_rvec   = julia["response_vec"]
    j_mmodes = julia["m_modes"]
    if !isempty(j_fvec) && !isempty(f_phi_x)
        push!(lines, @sprintf("%-5s  %-14s  %-14s  %-6s  %-14s  %-14s  %-6s",
            "m", "|Phi_x_fort|", "|Phi_x_julia|", "Rel%", "|Phi_tot_fort|", "|Phi_tot_julia|", "Rel%"))
        for (k, m) in enumerate(m_ctrl)
            j_idx = findfirst(==(m), j_mmodes)
            j_fx  = isnothing(j_idx) ? NaN : abs(j_fvec[j_idx])
            j_ft  = isnothing(j_idx) ? NaN : abs(j_rvec[j_idx])
            f_fx  = abs(f_phi_x[k])
            f_ft  = abs(f_phi_tot[k])
            push!(lines, @sprintf("%-5d  %14.4e  %14s  %-6s  %14.4e  %14s  %-6s",
                m, f_fx,
                isnan(j_fx) ? "N/A" : @sprintf("%.4e", j_fx),
                isnan(j_fx) ? "N/A" : rel_pct(f_fx, j_fx),
                f_ft,
                isnan(j_ft) ? "N/A" : @sprintf("%.4e", j_ft),
                isnan(j_ft) ? "N/A" : rel_pct(f_ft, j_ft)))
        end
    else
        push!(lines, "Phi_tot data unavailable")
    end
    push!(lines, "")

    push!(lines, "--- P Matrix |diagonal| comparison (n=$nn) ---")
    f_P = fort["P"]
    j_P = julia["permeability"]
    if !isempty(f_P) && !isempty(j_P) && size(f_P) == size(j_P)
        mpert = size(f_P, 1)
        push!(lines, @sprintf("%-5s  %-14s  %-14s  %-6s", "m", "|P_fort[i,i]|", "|P_julia[i,i]|", "Rel%"))
        m_ctrl = fort["m_vals_ctrl"]
        for i in 1:mpert
            m = i <= length(m_ctrl) ? m_ctrl[i] : i
            fp = abs(f_P[i, i])
            jp = abs(j_P[i, i])
            push!(lines, @sprintf("%-5d  %14.4e  %14.4e  %-6s", m, fp, jp, rel_pct(fp, jp)))
        end
    else
        push!(lines, "P matrix unavailable or size mismatch")
    end
    push!(lines, "")

    push!(lines, "--- Lambda (plasma inductance) |diagonal| comparison (n=$nn) ---")
    f_L = fort["Lambda"]
    j_L = julia["plasma_inductance"]
    if !isempty(f_L) && !isempty(j_L) && size(f_L) == size(j_L)
        mpert = size(f_L, 1)
        push!(lines, @sprintf("%-5s  %-14s  %-14s  %-6s  %-10s", "m", "|Λ_fort[i,i]|", "|Λ_julia[i,i]|", "Rel%", "Ratio j/f"))
        m_ctrl = fort["m_vals_ctrl"]
        for i in 1:mpert
            m = i <= length(m_ctrl) ? m_ctrl[i] : i
            fl = abs(f_L[i, i])
            jl = abs(j_L[i, i])
            ratio = fl > 0 ? jl/fl : NaN
            push!(lines, @sprintf("%-5d  %14.4e  %14.4e  %-6s  %10.3f", m, fl, jl, rel_pct(fl, jl), ratio))
        end
    else
        push!(lines, "Lambda matrix unavailable or size mismatch (fort=$(size(f_L)), julia=$(size(j_L)))")
    end
    push!(lines, "")

    push!(lines, "--- L (surface inductance) |diagonal| comparison (n=$nn) ---")
    f_LS = fort["L_surf"]
    j_LS = julia["surface_inductance"]
    if !isempty(f_LS) && !isempty(j_LS) && size(f_LS) == size(j_LS)
        mpert = size(f_LS, 1)
        push!(lines, @sprintf("%-5s  %-14s  %-14s  %-6s  %-10s", "m", "|L_fort[i,i]|", "|L_julia[i,i]|", "Rel%", "Ratio j/f"))
        m_ctrl = fort["m_vals_ctrl"]
        for i in 1:mpert
            m = i <= length(m_ctrl) ? m_ctrl[i] : i
            fl = abs(f_LS[i, i])
            jl = abs(j_LS[i, i])
            ratio = fl > 0 ? jl/fl : NaN
            push!(lines, @sprintf("%-5d  %14.4e  %14.4e  %-6s  %10.3f", m, fl, jl, rel_pct(fl, jl), ratio))
        end
    else
        push!(lines, "L matrix unavailable or size mismatch (fort=$(size(f_LS)), julia=$(size(j_LS)))")
    end
    push!(lines, "")

    push!(lines, "--- Resonant Surface Comparison (n=$nn) ---")
    push!(lines, @sprintf(
        "%-4s  %-8s  %-8s  %-14s  %-14s  %-6s  %-10s  %-10s  %-6s  %-7s  %-7s  %-6s",
        "q", "psi_fort", "psi_julia", "|Phi_res_fort|", "|Phi_res_julia|", "Rel%",
        "w_isl_fort", "w_isl_julia", "Rel%", "K_fort", "K_julia", "Rel%"))

    f_psi_rat = fort["psi_n_rational"]
    f_q_rat   = fort["q_rational"]
    f_phi_res = fort["Phi_res"]
    f_w_isl   = fort["w_isl"]
    f_K_isl   = fort["K_isl"]

    for (i, (fq_r, fp_r)) in enumerate(zip(f_q_rat, f_psi_rat))
        q_int = round(Int, fq_r)
        row   = find_julia_row(julia, q_int, nn)

        jp_r  = (row > 0 && !isempty(julia["rational_psi"]))     ? julia["rational_psi"][row]        : NaN
        j_phi = (row > 0 && !isempty(julia["resonant_flux"]))    ? abs(julia["resonant_flux"][row])   : NaN
        j_wisl= (row > 0 && !isempty(julia["island_half_width"])) ? 2*julia["island_half_width"][row] : NaN
        j_kch = (row > 0 && !isempty(julia["chirikov_parameter"])) ? julia["chirikov_parameter"][row]  : NaN

        push!(lines, @sprintf(
            "%-4d  %8.4f  %8s  %14.4e  %14s  %-6s  %10.4e  %10s  %-6s  %7.4f  %7s  %-6s",
            q_int, fp_r,
            isnan(jp_r)   ? "N/A" : @sprintf("%.4f", jp_r),
            abs(f_phi_res[i]),
            isnan(j_phi)  ? "N/A" : @sprintf("%.4e", j_phi),
            isnan(j_phi)  ? "N/A" : rel_pct(abs(f_phi_res[i]), j_phi),
            f_w_isl[i],
            isnan(j_wisl) ? "N/A" : @sprintf("%.4e", j_wisl),
            isnan(j_wisl) ? "N/A" : rel_pct(f_w_isl[i], j_wisl),
            f_K_isl[i],
            isnan(j_kch)  ? "N/A" : @sprintf("%.4f", j_kch),
            isnan(j_kch)  ? "N/A" : rel_pct(f_K_isl[i], j_kch),
        ))
    end

    push!(lines, "")

    push!(lines, "--- j_c Diagnostic (resonant current coefficient, inferred from I_res/Delta) ---")
    psio = julia["psio"]
    chi1 = 2π * psio
    push!(lines, @sprintf("  chi1 = 2π * psio = %.6e  (psio = %.6e)", chi1, psio))
    push!(lines, @sprintf("  j_c inferred as |I_res| * |m_res| / (|Δ'| * chi1)"))
    push!(lines, @sprintf("%-4s  %-8s  %-14s  %-14s  %-8s",
        "q", "psi_fort", "j_c_fort", "j_c_julia", "ratio j/f"))

    f_delta  = fort["Delta"]
    f_i_res  = fort["I_res"]
    j_delta  = julia["delta_prime"]
    j_i_res  = julia["resonant_current"]
    j_m_res  = julia["rational_m_res"]

    for (i, (fq_r, fp_r)) in enumerate(zip(f_q_rat, f_psi_rat))
        q_int  = round(Int, fq_r)
        m_res_fort = q_int * nn  # m_res = q * n at rational surface
        row = find_julia_row(julia, q_int, nn)

        jc_fort = (i <= length(f_delta) && i <= length(f_i_res) &&
                   abs(f_delta[i]) > 1e-30 && chi1 > 0) ?
            abs(f_i_res[i]) * abs(m_res_fort) / (abs(f_delta[i]) * chi1) : NaN

        jc_julia = (row > 0 && row <= length(j_delta) && row <= length(j_i_res) &&
                    row <= length(j_m_res) && abs(j_delta[row]) > 1e-30 && chi1 > 0) ?
            abs(j_i_res[row]) * abs(j_m_res[row]) / (abs(j_delta[row]) * chi1) : NaN

        ratio = (!isnan(jc_fort) && !isnan(jc_julia) && jc_fort > 0) ? jc_julia / jc_fort : NaN

        push!(lines, @sprintf("%-4d  %8.4f  %14s  %14s  %8s",
            q_int, fp_r,
            isnan(jc_fort)  ? "N/A" : @sprintf("%.4e", jc_fort),
            isnan(jc_julia) ? "N/A" : @sprintf("%.4e", jc_julia),
            isnan(ratio)    ? "N/A" : @sprintf("%.4f", ratio)))
    end

    push!(lines, "")
    push!(lines, "--- Delta' Comparison (isolates jump vs L_mm/area error) ---")
    push!(lines, @sprintf("%-4s  %-8s  %-14s  %-14s  %-8s  %-14s  %-14s  %-8s",
        "q", "psi_fort", "|Delta_fort|", "|Delta_julia|", "Rel%",
        "|I_res_fort|", "|I_res_julia|", "Rel%"))
    for (i, (fq_r, fp_r)) in enumerate(zip(f_q_rat, f_psi_rat))
        q_int = round(Int, fq_r)
        row   = find_julia_row(julia, q_int, nn)
        fd = (i <= length(f_delta)) ? abs(f_delta[i]) : NaN
        jd = (row > 0 && row <= length(j_delta)) ? abs(j_delta[row]) : NaN
        fi = (!isempty(f_i_res) && i <= length(f_i_res)) ? abs(f_i_res[i]) : NaN
        ji = (row > 0 && row <= length(j_i_res)) ? abs(j_i_res[row]) : NaN
        push!(lines, @sprintf("%-4d  %8.4f  %14s  %14s  %8s  %14s  %14s  %8s",
            q_int, fp_r,
            isnan(fd) ? "N/A" : @sprintf("%.4e", fd),
            isnan(jd) ? "N/A" : @sprintf("%.4e", jd),
            (isnan(fd) || isnan(jd)) ? "N/A" : rel_pct(fd, jd),
            isnan(fi) ? "N/A" : @sprintf("%.4e", fi),
            isnan(ji) ? "N/A" : @sprintf("%.4e", ji),
            (isnan(fi) || isnan(ji)) ? "N/A" : rel_pct(fi, ji)))
    end

    push!(lines, "=" ^ 72)
    return lines
end

function _interp1(x::AbstractVector, y::AbstractVector, xi::Real)
    n = length(x); n < 2 && return y[1]
    xi <= x[1]   && return y[1]
    xi >= x[end] && return y[end]
    i = clamp(searchsortedlast(x, xi), 1, n-1)
    t = (xi - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

# ─── Plot generation ─────────────────────────────────────────────────────────

function generate_plots(fort, julia, bench_dir, nn)
    f_q_rat  = fort["q_rational"]
    n_surf   = length(f_q_rat)

    # --- Row 1: q profile, Mercier, W_t scatter, |P| diagonal ---
    p1 = plot(fort["psi_n_dcon"], fort["q_dcon"],
        label="Fortran", xlabel="ψ_n", ylabel="q", title="q profile", lw=2)
    plot!(p1, julia["psi_q"], julia["q"], label="Julia", linestyle=:dash, lw=2)

    p2 = plot(fort["psi_n_dcon"], fort["di"],
        label="Fortran", xlabel="ψ_n", ylabel="D_I", title="Mercier D_I", lw=2)
    !isempty(julia["di"]) && plot!(p2, julia["psi_q"], julia["di"], label="Julia", linestyle=:dash, lw=2)
    hline!(p2, [0.0], color=:black, linestyle=:dot, label="")

    p3 = plot(fort["psi_n_dcon"], fort["dr"],
        label="Fortran", xlabel="ψ_n", ylabel="D_R", title="Mercier D_R", lw=2)
    !isempty(julia["dr"]) && plot!(p3, julia["psi_q"], julia["dr"], label="Julia", linestyle=:dash, lw=2)
    hline!(p3, [0.0], color=:black, linestyle=:dot, label="")

    # W_t eigenvalue scatter (log y), all eigenvalues vs index
    f_wt = fort["W_t_eigenvalue"]
    j_et = julia["et"]
    p4 = plot(; xlabel="mode index", ylabel="|W_t|", title="|W_t| eigenvalues",
               yscale=:log10, legend=:topright)
    if !isempty(f_wt)
        f_sorted = sort(abs.(f_wt))
        scatter!(p4, 1:length(f_sorted), f_sorted; color=:steelblue, markersize=5,
                 markershape=:circle, label="Fortran")
    end
    if !isempty(j_et)
        j_sorted = sort(abs.(j_et))
        scatter!(p4, 1:length(j_sorted), j_sorted; color=:orange, markersize=5,
                 markershape=:circle, label="Julia", markerstrokewidth=0)
    end

    # |P| diagonal comparison
    f_P = fort["P"]
    j_P = julia["permeability"]
    m_ctrl = fort["m_vals_ctrl"]
    p5 = plot(; xlabel="Poloidal mode m", ylabel="|P[i,i]|",
               title="|P| diagonal  (n=$nn)", legend=:topright)
    if !isempty(f_P) && !isempty(j_P)
        mpert_p = min(size(f_P, 1), size(j_P, 1), length(m_ctrl))
        ms = m_ctrl[1:mpert_p]
        fp_diag = [abs(f_P[i,i]) for i in 1:mpert_p]
        jp_diag = [abs(j_P[i,i]) for i in 1:mpert_p]
        me_f, fe = step_series(ms, fp_diag)
        me_j, je = step_series(ms, jp_diag)
        plot!(p5, me_f, fe; seriestype=:steppre, lw=2, color=:steelblue, label="Fortran")
        plot!(p5, me_j, je; seriestype=:steppre, lw=2, color=:orange, linestyle=:dash, label="Julia")
    else
        annotate!(p5, 0.5, 0.5, text("data unavailable", 10, :center))
    end

    # --- Row 2: Phi_x/tot spectrum and singular coupling vs psi ---

    # Phi_x & Phi_tot spectrum
    m_ctrl   = fort["m_vals_ctrl"]
    f_phi_x  = fort["Phi_x"]
    f_phi_tot = fort["Phi_tot"]
    j_fvec   = julia["forcing_vec"]
    j_rvec   = julia["response_vec"]
    j_mmodes = julia["m_modes"]
    p6 = plot(; xlabel="m", ylabel="|Φ| [Wb]", title="Phi_x & Phi_tot spectrum (n=$nn)",
              legend=:topright)
    if !isempty(j_fvec) && !isempty(f_phi_x)
        jfx_amps = [abs(get(Dict(zip(j_mmodes, j_fvec)), m, NaN+0im)) for m in m_ctrl]
        jft_amps = [abs(get(Dict(zip(j_mmodes, j_rvec)), m, NaN+0im)) for m in m_ctrl]
        ffx_amps = abs.(f_phi_x)
        fft_amps = abs.(f_phi_tot)
        me, jfe = step_series(m_ctrl, replace(jfx_amps, NaN=>0.0))
        _,  jte = step_series(m_ctrl, replace(jft_amps, NaN=>0.0))
        _,  ffe = step_series(m_ctrl, ffx_amps)
        _,  fte = step_series(m_ctrl, fft_amps)
        plot!(p6, me, ffe; seriestype=:steppre, lw=2, color=:steelblue, label="Φ_x Fortran")
        plot!(p6, me, jfe; seriestype=:steppre, lw=2, color=:steelblue, linestyle=:dash, label="Φ_x Julia")
        plot!(p6, me, fte; seriestype=:steppre, lw=2, color=:orange,    label="Φ_tot Fortran")
        plot!(p6, me, jte; seriestype=:steppre, lw=2, color=:orange,    linestyle=:dash, label="Φ_tot Julia")
    else
        annotate!(p6, 0.5, 0.5, text("data unavailable", 10, :center))
    end

    # Helper: singular coupling vs psi scatter with q-value labels
    function _singcoup_panel(ylabel_str, title_str,
                              f_psi, f_vals, f_q_labels,
                              j_psi, j_vals)
        p = plot(; xlabel="ψ_n", ylabel=ylabel_str, title=title_str, legend=:topright)
        if !isempty(f_psi) && !isempty(f_vals)
            scatter!(p, f_psi, f_vals; color=:steelblue, markersize=7, markershape=:circle, label="Fortran")
            plot!(p, f_psi, f_vals; color=:steelblue, lw=1, linestyle=:dot, label="")
            for (xi, yi, qi) in zip(f_psi, f_vals, f_q_labels)
                annotate!(p, xi, yi, text(qi, 7, :bottom, :center))
            end
        end
        if !isempty(j_psi) && !isempty(j_vals)
            scatter!(p, j_psi, j_vals; color=:orange, markersize=5, markershape=:circle,
                     label="Julia", markerstrokewidth=0)
            plot!(p, j_psi, j_vals; color=:orange, lw=1, linestyle=:dot, label="")
        end
        return p
    end

    # Collect per-surface values for singular coupling panels
    f_psi_rat = fort["psi_n_rational"]
    f_q_labels = [@sprintf("q=%d", round(Int, q)) for q in f_q_rat]

    # Delta (Δ')
    f_delta_vals = isempty(fort["Delta"]) ? Float64[] : real.(fort["Delta"])
    j_delta_psi  = julia["rational_psi"]
    j_delta_vals = isempty(julia["delta_prime"]) ? Float64[] : real.(julia["delta_prime"])
    p7 = _singcoup_panel("Re(Δ')", "Δ' vs ψ  (n=$nn)",
                          f_psi_rat, f_delta_vals, f_q_labels,
                          j_delta_psi, j_delta_vals)

    # I_res (resonant current)
    f_ires_vals = isempty(fort["I_res"]) ? Float64[] : abs.(fort["I_res"])
    j_ires_psi  = julia["rational_psi"]
    j_ires_vals = isempty(julia["resonant_current"]) ? Float64[] : abs.(julia["resonant_current"])
    p8 = _singcoup_panel("|I_res| [A]", "|I_res| vs ψ  (n=$nn)",
                          f_psi_rat, f_ires_vals, f_q_labels,
                          j_ires_psi, j_ires_vals)

    # Phi_res (resonant flux / area)
    f_phires_vals = isempty(fort["Phi_res"]) ? Float64[] : abs.(fort["Phi_res"])
    j_phires_psi  = julia["rational_psi"]
    j_phires_vals = isempty(julia["resonant_flux"]) ? Float64[] : abs.(julia["resonant_flux"])
    p9 = _singcoup_panel("|Φ_res| [T]", "|Φ_res| vs ψ  (n=$nn)",
                          f_psi_rat, f_phires_vals, f_q_labels,
                          j_phires_psi, j_phires_vals)

    # w_isl (island full width)
    f_wisl_vals = fort["w_isl"]
    j_wisl_psi  = julia["rational_psi"]
    j_wisl_vals = isempty(julia["island_half_width"]) ? Float64[] : 2 .* julia["island_half_width"]
    p10 = _singcoup_panel("w_isl [ψ_n]", "Island width vs ψ  (n=$nn)",
                           f_psi_rat, f_wisl_vals, f_q_labels,
                           j_wisl_psi, j_wisl_vals)

    # --- Row 3: b_psi profiles for resonant m harmonics ---
    # Identify resonant m values from rational surfaces
    res_m_vals = sort(unique(round.(Int, fort["q_rational"])))  # q=m/n → m=round(Int,q) for n=1
    # Clamp to available m range
    res_m_vals = filter(m -> m in m_ctrl, res_m_vals)

    # Helper: build a normalised radial profile panel for one mode
    function _profile_panel(ylabel_str, title_str,
                             f_psi, f_prof_complex, f_label,
                             j_psi, j_prof_complex, j_label)
        pb = plot(; xlabel="ψ_n", ylabel=ylabel_str, title=title_str, legend=:topright)
        f_prof = abs.(f_prof_complex)
        j_prof = abs.(j_prof_complex)
        f_max  = isempty(f_prof) ? 0.0 : maximum(f_prof)
        j_max  = isempty(j_prof) ? 0.0 : maximum(j_prof)
        f_max > 0 && plot!(pb, f_psi, f_prof ./ f_max;
            lw=1.5, color=:steelblue, label=f_label,
            markershape=:x, markersize=2, markerstrokewidth=1, markeralpha=0.6)
        rms_diff = NaN
        if j_max > 0 && f_max > 0 && !isempty(j_psi)
            j_on_fg  = [_interp1(j_psi, j_prof, p) for p in f_psi]
            j_max_on = maximum(j_on_fg)
            f_norm   = f_prof ./ f_max
            if j_max_on > 0
                rms_diff = sqrt(mean((j_on_fg ./ j_max_on .- f_norm).^2))
            end
            plot!(pb, j_psi, j_prof ./ j_max;
                lw=1.5, color=:orange, linestyle=:dash, label=j_label,
                markershape=:x, markersize=4, markerstrokewidth=1.5, markeralpha=0.9)
        end
        !isnan(rms_diff) && annotate!(pb, 0.8, 0.85, text(@sprintf("RMS=%.3f", rms_diff), 9, :center))
        return pb
    end

    j_psi_grid = julia["psi_grid"]
    f_psi_prof = fort["psi_n_prof"]
    f_m_out    = fort["m_out"]

    # Row 3: Jbgradpsi profiles — direct apples-to-apples (both = chi1*singfac*2πi*ξ_ψ in mode space)
    jbgp_panels = []
    j_jbgp = julia["Jbgradpsi"]
    f_jbgp = fort["Jbgradpsi"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_jbgp)) ? f_jbgp[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_jbgp)) ? j_jbgp[:, j_midx] : ComplexF64[]
        push!(jbgp_panels, _profile_panel(
            "|Jbgradpsi| (norm.)", "Jbgradpsi  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(jbgp_panels) < 4
        push!(jbgp_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Row 4: b_n profiles — Julia b_n (via IDFT/divide/DFT) vs Fortran b_n
    bn_panels = []
    j_bn = julia["b_n"]
    f_bn = fort["b_n"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_bn)) ? f_bn[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_bn)) ? j_bn[:, j_midx] : ComplexF64[]
        push!(bn_panels, _profile_panel(
            "|b_n| (norm.)", "b_n  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(bn_panels) < 4
        push!(bn_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Row 5: xigradpsi (xi_psi) profiles
    xigp_panels = []
    j_xigp = julia["xi_psi"]
    f_xigp = fort["xigradpsi"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_xigp)) ? f_xigp[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_xigp)) ? j_xigp[:, j_midx] : ComplexF64[]
        push!(xigp_panels, _profile_panel(
            "|ξ_ψ| (norm.)", "xigradpsi  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(xigp_panels) < 4
        push!(xigp_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Row 6: xi_n profiles
    xin_panels = []
    j_xin = julia["xi_n"]
    f_xin = fort["xi_n"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_xin)) ? f_xin[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_xin)) ? j_xin[:, j_midx] : ComplexF64[]
        push!(xin_panels, _profile_panel(
            "|ξ_n| (norm.)", "xi_n  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(xin_panels) < 4
        push!(xin_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Row 7: Clebsch ∂ξ^ψ/∂ψ (regularized) profiles
    xigp1_panels = []
    j_xigp1 = julia["clebsch_psi1"]
    f_xigp1 = fort["xigradpsi_dpsi"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_xigp1)) ? f_xigp1[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_xigp1)) ? j_xigp1[:, j_midx] : ComplexF64[]
        push!(xigp1_panels, _profile_panel(
            "|∂ξ^ψ/∂ψ| (norm.)", "xigradpsi_dpsi  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(xigp1_panels) < 4
        push!(xigp1_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Row 8: Clebsch ξ^α (regularized) profiles
    xiga_panels = []
    j_xiga = julia["clebsch_alpha"]
    f_xiga = fort["xigradalpha"]
    for m_res in res_m_vals[1:min(end, 4)]
        f_midx = findfirst(==(m_res), f_m_out)
        j_midx = findfirst(==(m_res), j_mmodes)
        fp = (!isnothing(f_midx) && !isempty(f_xiga)) ? f_xiga[:, f_midx] : ComplexF64[]
        jp = (!isnothing(j_midx) && !isempty(j_xiga)) ? j_xiga[:, j_midx] : ComplexF64[]
        push!(xiga_panels, _profile_panel(
            "|ξ^α| (norm.)", "xigradalpha  m=$m_res (n=$nn)",
            isempty(fp) ? Float64[] : f_psi_prof, fp, "Fortran",
            isempty(jp) ? Float64[] : j_psi_grid, jp, "Julia"))
    end
    while length(xiga_panels) < 4
        push!(xiga_panels, plot(; title="(no data)", legend=false, axis=false, border=:none))
    end

    # Pad all profile rows to 5 panels for uniform grid layout
    _empty_panel() = plot(; title="", legend=false, axis=false, grid=false, border=:none)
    for panels in [jbgp_panels, bn_panels, xigp_panels, xin_panels, xigp1_panels, xiga_panels]
        while length(panels) < 5
            push!(panels, _empty_panel())
        end
    end

    p = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
        jbgp_panels..., bn_panels..., xigp_panels..., xin_panels...,
        xigp1_panels..., xiga_panels...;
        layout=(8, 5), size=(2600, 3600),
        left_margin=20Plots.mm, right_margin=5Plots.mm, bottom_margin=8Plots.mm, top_margin=5Plots.mm)
    outfile = joinpath(bench_dir, "comparison_plots.png")
    savefig(p, outfile)
    println("Plots saved to: ", abspath(outfile))

    # ─── R,Z,φ profile comparison figure ──────────────────────────────────────
    # Compare θ=0 radial profiles (real & imag) in machine coordinates.
    # Julia modes are converted to machine theta-space via modes_to_theta with
    # keep_sfl_phi=false (applies exp(i*n*ν) toroidal phase, matching Fortran
    # gpout_xbrzphifun). Fortran data is already in machine theta-space.
    h5_path = julia["h5_path"]
    rzphi_panels = []
    for (comp_label, f_key, h5_var) in [
            ("ξ_R",  "xi_r_fun",   "perturbed_equilibrium/response/xi_R"),
            ("ξ_Z",  "xi_z_fun",   "perturbed_equilibrium/response/xi_Z"),
            ("ξ_φ",  "xi_phi_fun", "perturbed_equilibrium/response/xi_phi"),
            ("b_R",  "b_r_fun",    "perturbed_equilibrium/response/b_R"),
            ("b_Z",  "b_z_fun",    "perturbed_equilibrium/response/b_Z"),
            ("b_φ",  "b_phi_fun",  "perturbed_equilibrium/response/b_phi")]
        f_fun = get(fort, f_key, Matrix{ComplexF64}(undef, 0, 0))

        # Convert Julia SFL modes → machine theta-space
        j_fun = Matrix{ComplexF64}(undef, 0, 0)
        if isfile(h5_path)
            try
                j_theta_data, _, _ = PerturbedEquilibriumModes.modes_to_theta(
                    h5_path, h5_var; keep_sfl_phi=false)
                j_fun = j_theta_data[:, :, 1]  # first (only) n
            catch e
                @warn "modes_to_theta failed for $h5_var" exception=e
            end
        end

        # Extract θ=0 (first column) real and imaginary profiles vs ψ
        f_re = isempty(f_fun) ? Float64[] : real.(f_fun[:, 1])
        f_im = isempty(f_fun) ? Float64[] : imag.(f_fun[:, 1])
        j_re = isempty(j_fun) ? Float64[] : real.(j_fun[:, 1])
        j_im = isempty(j_fun) ? Float64[] : imag.(j_fun[:, 1])

        # Y-axis limits from 2-98 percentiles to avoid spikes
        function _percentile_ylims(arrs...)
            all_vals = vcat([v for v in arrs if !isempty(v)]...)
            isempty(all_vals) && return (-1, 1)
            lo = quantile(all_vals, 0.02)
            hi = quantile(all_vals, 0.98)
            margin = 0.1 * max(hi - lo, eps())
            return (lo - margin, hi + margin)
        end

        # Real part panel
        yl_re = _percentile_ylims(f_re, j_re)
        p_re = plot(; xlabel="ψ_n", ylabel="Re($comp_label)", title="Re($comp_label) θ=0 (n=$nn)",
                      legend=:topright, ylims=yl_re)
        !isempty(f_re) && plot!(p_re, f_psi_prof, f_re; lw=2.5, color=:steelblue, label="Fortran")
        !isempty(j_re) && plot!(p_re, j_psi_grid, j_re; lw=1.0, color=:orange, label="Julia")
        push!(rzphi_panels, p_re)

        # Imaginary part panel
        yl_im = _percentile_ylims(f_im, j_im)
        p_im = plot(; xlabel="ψ_n", ylabel="Im($comp_label)", title="Im($comp_label) θ=0 (n=$nn)",
                      legend=:topright, ylims=yl_im)
        !isempty(f_im) && plot!(p_im, f_psi_prof, f_im; lw=2.5, color=:steelblue, label="Fortran")
        !isempty(j_im) && plot!(p_im, j_psi_grid, j_im; lw=1.0, color=:orange, label="Julia")
        push!(rzphi_panels, p_im)
    end

    p_rzphi = plot(rzphi_panels...;
        layout=(6, 2), size=(1200, 2700),
        left_margin=20Plots.mm, right_margin=5Plots.mm, bottom_margin=8Plots.mm, top_margin=5Plots.mm)
    outfile_rzphi = joinpath(bench_dir, "rzphi_comparison.png")
    savefig(p_rzphi, outfile_rzphi)
    println("R,Z,φ profile plots saved to: ", abspath(outfile_rzphi))
end


# ─── Matrix heatmap figure ───────────────────────────────────────────────────

"""
    _mat_heatmap(M, title_str, m_vals; rel_err=false)

Return a heatmap panel for matrix M.  Values are plotted on a log₁₀ scale so
that orders-of-magnitude variation is visible.  When `rel_err=true` the colour
scale is labelled as log₁₀(rel error).
"""
function _mat_heatmap(M::AbstractMatrix, title_str::String, m_vals;
                      rel_err::Bool=false, clims=nothing)
    vals = log10.(max.(abs.(M), 1e-30))   # floor at 1e-30 to avoid -Inf
    label = rel_err ? "log₁₀(|ΔM/M|)" : "log₁₀|M|"
    kw = isnothing(clims) ? () : (clims=clims,)
    heatmap(m_vals, m_vals, vals;
        title=title_str, xlabel="m′", ylabel="m",
        color=:viridis, colorbar_title=label,
        xrotation=45, yflip=false, kw...)
end

function _mat_heatmap_empty(title_str)
    p = plot(; title=title_str, axis=false, legend=false, border=:none)
    annotate!(p, 0.5, 0.5, text("unavailable", 9, :center))
    return p
end

"""
    generate_matrix_plots(fort, julia, bench_dir, nn)

Save `matrix_comparison_plots.png` showing element-wise |M| heatmaps for the
fundamental response matrices L, Λ, P and the vacuum energy matrix wt0.

Layout (rows × 3 columns):
  • Col 1: |M| Fortran   Col 2: |M| Julia   Col 3: relative error |ΔM/M_fort|
  • Row 4 (wt0): Julia only — no Fortran reference available in NetCDF outputs.
"""
function generate_matrix_plots(fort, julia, bench_dir, _nn)
    m_ctrl = fort["m_vals_ctrl"]

    f_L   = fort["L_surf"]
    f_La  = fort["Lambda"]
    f_P   = fort["P"]
    j_L   = julia["surface_inductance"]
    j_La  = julia["plasma_inductance"]
    j_P   = julia["permeability"]
    j_wt0 = julia["wt0"]

    panels = Any[]

    for (name, f_M, j_M) in [("L  (surface inductance)", f_L, j_L),
                               ("Λ  (plasma inductance)",  f_La, j_La),
                               ("P  (permeability)",        f_P, j_P)]
        ok = !isempty(f_M) && !isempty(j_M) && size(f_M) == size(j_M)
        if ok
            # Use a common colour range for the first two panels
            all_vals = log10.(max.(abs.([f_M; j_M]), 1e-30))
            clims_mat = (minimum(all_vals), maximum(all_vals))
            push!(panels, _mat_heatmap(f_M, "|$name| — Fortran",  m_ctrl; clims=clims_mat))
            push!(panels, _mat_heatmap(j_M, "|$name| — Julia",    m_ctrl; clims=clims_mat))
            rel_err = abs.(j_M .- f_M) ./ (abs.(f_M) .+ 1e-30)
            push!(panels, _mat_heatmap(rel_err, "Rel. error |$name|", m_ctrl; rel_err=true))
        else
            for _ in 1:3
                push!(panels, _mat_heatmap_empty("|$name| (data unavailable)"))
            end
        end
    end

    # wt0 — fundamental vacuum energy matrix, Julia only
    if !isempty(j_wt0)
        push!(panels, _mat_heatmap(j_wt0, "|wt0| — Julia (no Fortran ref.)", m_ctrl))
        # Fill the remaining two slots to keep the 3-column layout
        push!(panels, plot(; title="", axis=false, legend=false, border=:none))
        push!(panels, plot(; title="", axis=false, legend=false, border=:none))
    end

    nrows = length(panels) ÷ 3
    p = plot(panels...;
        layout=(nrows, 3),
        size=(1800, 420 * nrows),
        left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=4Plots.mm)
    outfile = joinpath(bench_dir, "matrix_comparison_plots.png")
    savefig(p, outfile)
    println("Matrix plots saved to: ", abspath(outfile))
end


# ─── Main ────────────────────────────────────────────────────────────────────

function main(argv=ARGS)
    opts = parse_args(argv)
    fortran_dir      = opts.fortran_dir
    do_plot          = opts.do_plot
    skip_run         = opts.skip_run
    use_file_forcing = opts.use_file_forcing

    isdir(fortran_dir) || error("Fortran directory not found: $fortran_dir")
    isfile(joinpath(fortran_dir, "equil.in")) || error("equil.in not found in $fortran_dir")
    isfile(joinpath(fortran_dir, "dcon.in"))  || error("dcon.in not found in $fortran_dir")

    coil_dat_dir = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries")

    forcing_mode = use_file_forcing ? "file (Fortran Phi_coil)" : "coil (Julia Biot-Savart)"
    println("=" ^ 72)
    println("GPEC Fortran vs Julia Benchmark")
    println("  Fortran dir  : $fortran_dir")
    println("  Forcing mode : $forcing_mode")
    println("=" ^ 72)

    println("\n[1/5] Parsing Fortran inputs ...")
    p = parse_fortran_run(fortran_dir)
    @printf "  eq:      %s  %s  (jac=%s, psilow=%.4f, psihigh=%.4f)\n" p.eq_type p.eq_filename p.jac_type p.psilow p.psihigh
    @printf "  n:       %d  (delta_mlow=%d, delta_mhigh=%d)\n" p.nn p.delta_mlow p.delta_mhigh
    if !use_file_forcing
        @printf "  machine: %s  (%d coil sets, grid %d×%d)\n" p.machine length(p.coil_names) p.mtheta_coil p.nzeta_coil
        for (name, curs) in zip(p.coil_names, p.coil_currents)
            @printf "    %-10s  currents: %s A\n" name string(round.(curs; digits=1))
        end
    end

    println("\n[2/5] Loading Fortran NetCDF outputs ...")
    fort = load_fortran_outputs(fortran_dir, p.nn)
    @printf "  psilim=%.6f,  qlim=%.4f,  %d rational surfaces\n" fort["psilim"] fort["qlim"] length(fort["psi_n_rational"])
    @printf "  m range: %d:%d\n" fort["mlow"] fort["mhigh"]

    bench_name = "fortran_vs_julia_" * basename(fortran_dir)
    bench_dir  = joinpath(@__DIR__, bench_name)
    mkpath(bench_dir)
    println("\nBenchmark directory: $bench_dir")

    eq_filename = p.eq_filename
    if !isempty(eq_filename)
        src = joinpath(fortran_dir, eq_filename)
        dst = joinpath(bench_dir, eq_filename)
        if isfile(src) && !isfile(dst)
            cp(src, dst)
            println("Copied equilibrium file: $eq_filename")
        end
    end

    # Step 3: Phi_x comparison (coil mode only)
    equil = nothing
    if use_file_forcing
        println("\n[3/5] Skipped Phi_x comparison (file-forcing mode)")
    else
        println("\n[3/5] Julia Biot-Savart Phi_x vs Fortran Phi_coil ...")
        julia_amps, fort_amps, m_vals, equil = compare_phix(p, fort, fortran_dir, coil_dat_dir)

        println("  m      Julia |Phi_x| [T·m²]   Fortran |Phi_x| [T·m²]   ratio")
        println("  " * "-"^60)
        ratios = Float64[]
        for m in sort(m_vals)
            ja = get(julia_amps, m, NaN)
            fa = get(fort_amps,  m, NaN)
            ratio = (!isnan(ja) && ja > 1e-40) ? fa / ja : NaN
            isnan(ratio) || push!(ratios, ratio)
            @printf "  %4d   %14.4e         %14.4e       %.4f\n" m ja fa (isnan(ratio) ? 0.0 : ratio)
        end
        if !isempty(ratios)
            @printf "  Mean ratio = %.4f ± %.4f  (ideal: 1.0000)\n" mean(ratios) std(ratios)
        end
    end

    # Step 4: Write config and run Julia GPEC
    toml_path = joinpath(bench_dir, "gpec.toml")
    if use_file_forcing
        forcing_path = joinpath(bench_dir, "forcing.dat")
        write_forcing_dat(forcing_path, p.nn, fort["m_vals"], fort["Phi_coil_amp"])
        println("\n[4/5] Wrote forcing.dat ($(length(fort["m_vals"])) modes, sfl_flux_Wb convention)")
        write_gpec_toml_file(toml_path, fortran_dir, p)
    else
        write_gpec_toml_coil(toml_path, fortran_dir, p, abspath(coil_dat_dir))
        println("\n[4/5] Wrote gpec.toml (coil forcing mode)")
    end

    h5_path = joinpath(bench_dir, "gpec.h5")
    if skip_run
        println("  --skip-run: skipping Julia GPEC run")
    else
        println("  Running Julia GPEC (subprocess)...")
        isfile(h5_path) && rm(h5_path)
        gpec_root  = joinpath(@__DIR__, "..")
        julia_cmd  = Base.julia_cmd()
        cmd = `$julia_cmd --project=$gpec_root -e "using GeneralizedPerturbedEquilibrium; GeneralizedPerturbedEquilibrium.main([\"$bench_dir\"])"`
        run(cmd)
    end

    # Step 5: Compare
    println("\n[5/5] Loading Julia outputs and comparing ...")
    julia = load_julia_outputs(h5_path)
    @printf "  psilim=%.6f,  qlim=%.4f,  %d resonant (surface,n) pairs\n" julia["psilim"] julia["qlim"] length(julia["rational_psi"])

    println()
    table_lines = build_comparison_table(fort, julia, fortran_dir, bench_dir, p.nn)
    for line in table_lines; println(line); end

    table_path = joinpath(bench_dir, "comparison_table.txt")
    open(table_path, "w") do io
        for line in table_lines; println(io, line); end
    end
    println("\nComparison table saved to: ", abspath(table_path))

    do_plot && generate_plots(fort, julia, bench_dir, p.nn)
    do_plot && generate_matrix_plots(fort, julia, bench_dir, p.nn)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
