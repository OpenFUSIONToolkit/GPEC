"""
    Main(path::String, equil)

PENTRC main entry point for standalone execution.
Reads configuration, sets up equilibrium, and runs calculations.

# Arguments
- `path::String`: Path to directory containing pentrc.toml and equilibrium files
- `equil`: PlasmaEquilibrium from Equilibrium module

# Usage
```julia
PENTRC.Main("/path/to/config/directory", equil)
```
"""
function Main(path::String, equil)

    # Read configuration
    config_path = joinpath(path, "pentrc.toml")
    ctrl = read_pentrc_config(config_path)

    if ctrl.verbose
        println()
        println("PENTRC START => v3.00")
        println("_" ^ 50)
    end

    start_time = time()

    # Setup output
    init_output(ctrl, path)

    # Initialize internal state from equilibrium
    intr = PentrcInternal(; dir_path=path)
    initialize_from_equilibrium!(intr, equil)
    intr.flags = get_method_flags(ctrl)

    # Print moment type
    if ctrl.verbose
        println("-----" ^ 10)
        if ctrl.moment == "heat"
            println("Calculating HEAT transport")
        else
            println("Calculating PARTICLE transport and TORQUE")
        end
        println("-----" ^ 10)
    end

    # Run computations
    compute_matrix_calculation!(intr, ctrl, equil)
    compute_torque_all_methods!(intr, ctrl, equil)

    # Prepare output
    prepare_output(intr, ctrl)

    # Print summary
    if ctrl.verbose
        elapsed = time() - start_time
        println()
        println("PENTRC STOP => normal termination")
        @printf("Elapsed time: %.2f seconds\n", elapsed)
        println("=" ^ 50)
    end

    return intr
end
