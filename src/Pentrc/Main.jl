"""
    Main(path::String)

PENTRC main entry point for standalone execution.
Reads configuration, sets up equilibrium, and runs calculations.

# Arguments
- `path::String`: Path to directory containing pentrc.toml and equilibrium files

# Usage
```julia
PENTRC.Main("/path/to/config/directory")
```
"""
function _load_equilibrium_for_pentrc(path::String, ctrl::PentrcControl)
    equil_toml = joinpath(path, "equil.toml")
    if isfile(equil_toml)
        return Equilibrium.setup_equilibrium(equil_toml)
    end

    if !isempty(ctrl.idconfile)
        idcon_path = isabspath(ctrl.idconfile) ? ctrl.idconfile : joinpath(path, ctrl.idconfile)
        if isfile(idcon_path)
            return read_equil(idcon_path; verbose=ctrl.verbose)
        end
    end

    error("No equilibrium source found for PENTRC. Expected `equil.toml` in $(path) or a readable `idconfile` from `pentrc.toml`.")
end

function Main(path::String)
    
    # [1] Read configuration
    config_path = joinpath(path, "pentrc.toml")
    ctrl = read_pentrc_config(config_path)
    
    if ctrl.verbose
        println()
        println("PENTRC START => v3.00")
        println("_" ^ 50)
    end
    
    # [2] Setup execution environment
    start_time = time()
    equil = _load_equilibrium_for_pentrc(path, ctrl)
    
    # Clear output directory if requested
    if ctrl.clean && ispath(path)
        try
            run(`bash -c "rm -f pentrc_*.out"`)
        catch
            # Silent fail if directory cleanup isn't possible
        end
    end
    
    # Set threading
    if ctrl.pentrc_threads > 0
        if ctrl.pentrc_threads > Threads.nthreads()
            @warn "Requested $(ctrl.pentrc_threads) threads but only $(Threads.nthreads()) available"
        end
    end
    
    # [3] Setup output
    outp = PentrcOutput(; output_dir=path)
    init_output(outp, path)
    
    # [4] Initialize internal state
    intr = PentrcInternal(; dir_path=path)
    intr.flags = get_method_flags(ctrl)
    initialize_pentrc(ctrl, equil)
    
    # [5] Administrative setup
    if ctrl.fnml_flag
        # TODO: set_fymnl()
    end
    if ctrl.ellip_flag
        # TODO: set_ellip()
    end
    if ctrl.diag_flag
        diagnose_all()
    end
    
    # [6] Print moment type
    if ctrl.verbose
        println("-----" ^ 10)
        if ctrl.qt
            println("Calculating HEAT transport")
        else
            println("Calculating PARTICLE transport and TORQUE")
        end
        println("-----" ^ 10)
    end
    
    # [7] Run computations
    compute_matrix_calculation!(intr, ctrl)
    compute_torque_all_methods!(intr, ctrl)
    
    # [8] Prepare output
    prepare_output(intr, ctrl)
    
    # [9] Print summary
    if ctrl.verbose
        elapsed = time() - start_time
        println()
        println("PENTRC STOP => normal termination")
        @printf("Elapsed time: %.2f seconds\n", elapsed)
        println("=" ^ 50)
    end
    
    return intr
end

# Allow standalone execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        error("Usage: julia Main.jl <path/to/config>")
    end
    config_path = ARGS[1]
    Main(config_path)
end
