"""
Convergence plot for 3D vacuum (via DCON free-boundary total energy eigenvalue).

Runs the equivalent of `examples/Solovev_ideal_example_3D` while scanning `nzeta` (DCON `nzvac`)
and plotting the difference between the 3D result and a 2D converged reference.

It repeats the scan for three equilibrium coordinate choices (`jac_type` in `equil.toml`):
  - "pest"
  - "boozer"
  - "hamada"

All three curves are plotted on one figure as:

    abs(real(et[1])_3D - real(et[1])_2D)

on a log-scale y-axis. The 2D reference is computed separately for each `jac_type` from
`examples/Solovev_ideal_example` with `nzvac=1`.

Wall handling:
  - Toggle with `--wall=nowall` or `--wall=conformal` (default: nowall)
  - For `conformal`, wall parameters are taken from `examples/Solovev_ideal_example/dcon.toml`'s `[WALL]`
    section (with `shape` forced to `"conformal"`).
"""

using Pkg
Pkg.activate("$(@__DIR__)/../..")

using JPEC
using Plots
using Printf
using TOML
using LaTeXStrings

default(; markersize=4, linewidth=2)

const JAC_TYPES = ["pest", "boozer", "hamada"]

# DCON `nzvac` (VacuumInput3D `nzeta`) values to scan.
# Keep this modest by default: this runs full DCON + 3D vacuum each point.
const MIN_NZETA = 1 # required by 3D kernel patch size (see Vacuum/Kernel3D)
const DEFAULT_NZETA_VALUES = [24, 32, 48, 64, 96]
const DEFAULT_WALL_MODE = "nowall" # "nowall" | "conformal"

function _sanitize_nzeta_values(vals::Vector{Int})
    vals = sort!(unique(vals))
    vals = filter(>=(MIN_NZETA), vals)
    isempty(vals) && error("All requested nzeta values are < $MIN_NZETA; increase nzeta.")
    return vals
end

function _parse_nzeta_values(args::Vector{String})
    for (i, a) in pairs(args)
        if startswith(a, "--nzeta=")
            return _sanitize_nzeta_values(parse.(Int, split(split(a, "="; limit=2)[2], ",")))
        elseif a == "--nzeta" && i < lastindex(args)
            return _sanitize_nzeta_values(parse.(Int, split(args[i+1], ",")))
        end
    end
    return DEFAULT_NZETA_VALUES
end

function _parse_wall_mode(args::Vector{String})
    for (i, a) in pairs(args)
        if startswith(a, "--wall=")
            return split(a, "="; limit=2)[2]
        elseif a == "--wall" && i < lastindex(args)
            return args[i+1]
        end
    end
    return DEFAULT_WALL_MODE
end

function _write_toml(path::String, dict::AbstractDict)
    open(path, "w") do io
        TOML.print(io, dict)
    end
    return nothing
end

function _ensure_abs_eq_filename!(equil_dict::Dict{String,Any}, example_dir::String)
    eq_filename = equil_dict["EQUIL_CONTROL"]["eq_filename"]
    if !isabspath(eq_filename)
        equil_dict["EQUIL_CONTROL"]["eq_filename"] = normpath(joinpath(example_dir, eq_filename))
    end
    return nothing
end

function run_et1_from_example(;
    example_dir::String,
    jac_type::Union{Nothing,String}=nothing,
    nzvac::Union{Nothing,Int}=nothing,
    wall::Union{Nothing,Dict{String,Any}}=nothing,
    verbose::Bool=false,
    write_outputs_to_HDF5::Bool=false
)
    mktempdir() do tmp
        # --- Equilibrium config ---
        equil_dict = TOML.parsefile(joinpath(example_dir, "equil.toml"))
        _ensure_abs_eq_filename!(equil_dict, example_dir)
        if jac_type !== nothing
            equil_dict["EQUIL_CONTROL"]["jac_type"] = jac_type
        end
        _write_toml(joinpath(tmp, "equil.toml"), equil_dict)

        # --- DCON config ---
        dcon_dict = TOML.parsefile(joinpath(example_dir, "dcon.toml"))
        dcon_ctrl = dcon_dict["DCON_CONTROL"]
        dcon_ctrl["verbose"] = verbose
        dcon_ctrl["write_outputs_to_HDF5"] = write_outputs_to_HDF5
        if nzvac !== nothing
            dcon_ctrl["nzvac"] = nzvac
        end
        dcon_dict["DCON_CONTROL"] = dcon_ctrl

        if wall !== nothing
            dcon_dict["WALL"] = wall
        end
        _write_toml(joinpath(tmp, "dcon.toml"), dcon_dict)

        result = JPEC.DCON.Main(tmp)
        return real(result.vac_data.et[1])
    end
end

function main()
    println("="^80)
    println("3D VACUUM CONVERGENCE vs NZETA (|Δ et[1]|) — Solovev ideal example")
    println("="^80)

    nzeta_values = _sanitize_nzeta_values(_parse_nzeta_values(ARGS))
    wall_mode = lowercase(_parse_wall_mode(ARGS))
    wall_mode in ("nowall", "conformal") || error("Unknown --wall=$wall_mode (expected nowall or conformal)")
    println("\nUsing nzeta scan: $nzeta_values (min allowed: $MIN_NZETA)")
    println("Using wall mode: $wall_mode")

    example_3D_dir = normpath(joinpath(@__DIR__, "../../examples/Solovev_ideal_example_3D"))
    example_2D_dir = normpath(joinpath(@__DIR__, "../../examples/Solovev_ideal_example"))

    println("\n3D example: $example_3D_dir")
    println("2D reference example: $example_2D_dir")

    # Wall settings used for ALL runs (2D refs + 3D scan)
    wall_dict = if wall_mode == "nowall"
        Dict{String,Any}("shape" => "nowall")
    else
        template = TOML.parsefile(joinpath(example_2D_dir, "dcon.toml"))
        haskey(template, "WALL") || error("No [WALL] section found in $(joinpath(example_2D_dir, "dcon.toml"))")
        wall = Dict{String,Any}(template["WALL"])
        wall["shape"] = "conformal"
        wall
    end

    # 2D "converged" reference (no wall, nzvac=1), computed per jac_type
    et1_2D_by_jac = Dict{String,Float64}()
    println("\nComputing 2D references (nzvac=1, WALL.shape=\"$(wall_dict["shape"])\")...")
    for jac in JAC_TYPES
        et1_2D = run_et1_from_example(;
            example_dir=example_2D_dir,
            jac_type=jac,
            nzvac=1,
            wall=wall_dict,
            verbose=false,
            write_outputs_to_HDF5=false
        )
        et1_2D_by_jac[jac] = et1_2D
        println(@sprintf "  jac_type=%-6s  2D real(et[1])=%+.8e" jac et1_2D)
    end

    # 3D scan for each jac_type
    et1_by_jac = Dict{String,Vector{Float64}}()
    absdiff_by_jac = Dict{String,Vector{Float64}}()
    for jac in JAC_TYPES
        println("\nScanning 3D et[1] vs nzeta for jac_type=\"$jac\" (WALL.shape=\"$(wall_dict["shape"])\")")
        et1s = Float64[]
        absdiffs = Float64[]
        et1_2D = et1_2D_by_jac[jac]
        for nzeta in nzeta_values
            et1 = run_et1_from_example(;
                example_dir=example_3D_dir,
                jac_type=jac,
                nzvac=nzeta,
                wall=wall_dict,
                verbose=false,
                write_outputs_to_HDF5=false
            )
            push!(et1s, et1)
            absdiff = abs((et1 - et1_2D) / et1_2D)
            push!(absdiffs, absdiff)
            println(@sprintf "  nzeta=%4d  3D real(et[1])=%+.8e  |Δ|=% .3e" nzeta et1 absdiff)
        end
        et1_by_jac[jac] = et1s
        absdiff_by_jac[jac] = absdiffs
    end

    # Plot
    p = plot(;
        xlabel=L"N_\zeta",
        # ylabel=L"|min(e_{t,3D}) - min(e_{t,2D}) / min(e_{t,2D})|",
        ylabel="Relative diff in min eigenvalue of " * L"\bf{W}_T",
        title="Solovev equilibrium, n=2 (WALL=$(wall_dict["shape"]))",
        framestyle=:box,
        xlims=(0, nzeta_values[end]+4),
        legend=:best,
        minorgrid=true,
        yscale=:log10
    )

    colors = Dict("pest" => :dodgerblue3, "boozer" => :darkorange2, "hamada" => :seagreen3)
    for jac in JAC_TYPES
        plot!(p, nzeta_values, absdiff_by_jac[jac];
            label="$jac",
            marker=:circle,
            color=get(colors, jac, :black)
        )
    end

    out_pdf = joinpath(@__DIR__, "vacuum_3D_et1_convergence_vs_nzeta_$(wall_mode)_n2.pdf")
    out_toml = joinpath(@__DIR__, "vacuum_3D_et1_convergence_vs_nzeta_$(wall_mode)_n2_results.toml")
    savefig(p, out_pdf)

    results = Dict{String,Any}(
        "nzeta_values" => nzeta_values,
        "jac_types" => JAC_TYPES,
        "wall_mode" => wall_mode,
        "wall" => wall_dict,
        "et1_2D_by_jac" => et1_2D_by_jac,
        "et1_by_jac" => et1_by_jac,
        "absdiff_by_jac" => absdiff_by_jac
    )
    _write_toml(out_toml, results)

    println("\nSaved plot to: $out_pdf")
    println("Saved results to: $out_toml")

    display(p)
    return nothing
end

main()
