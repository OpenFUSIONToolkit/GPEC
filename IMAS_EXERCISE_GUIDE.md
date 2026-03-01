# IMAS Integration Exercise Guide

This guide shows exactly where to add IMAS code to go from Point A (before IMAS) to Point B (after IMAS).

## Files Already Set Up with TODO Markers ✓

1. **src/DCON/WriteImas.jl** - Empty with detailed TODO steps
2. **test/runtests_imas.jl** - Empty with TODO instructions
3. **src/DCON/DCON.jl** - Has 2 TODO markers

---

## src/DCON/Main.jl - IMAS Modifications Needed

The current Main.jl has ALL the IMAS code already written. You need to:

### TODO #1: Add wrapper function (at the very beginning, before current Main)

```julia
"""
    Main(path::String="./")

Run DCON from a directory containing `dcon.toml` and `equil.toml`.
"""
function Main(path::String="./")
    return Main(path, nothing)
end
```

### TODO #2: Change function signature (line ~1)

**Original:**
```julia
function Main(path::String="./")
```

**Change to:**
```julia
"""
    Main(path::String, dd::IMASdd.dd)

Run DCON using IMAS data dictionary `dd` for the equilibrium.
"""
function Main(path::String, dd::Union{IMASdd.dd, Nothing})
```

### TODO #3: Replace equilibrium setup (around line 11-12)

**Original:**
```julia
ctrl = DconControl(; (Symbol(k) => v for (k, v) in inputs["DCON_CONTROL"])...)
equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
```

**Change to:**
```julia
ctrl = DconControl(; (Symbol(k) => v for (k, v) in inputs["DCON_CONTROL"])...)

if dd === nothing
    # Standard file-based equilibrium
    equil = Equilibrium.setup_equilibrium(joinpath(intr.dir_path, "equil.toml"))
else
    # IMAS-based equilibrium
    eq_config = Equilibrium.EquilibriumConfig(;
        control = Equilibrium.EquilibriumControl(;
            eq_type    = "imas",
            eq_filename = joinpath(path, "imas_equilibrium"),
        )
    )
    equil = Equilibrium.setup_equilibrium(eq_config, dd)
end
```

### TODO #4: Add write_imas call before return (around line 210-213)

**Original:**
```julia
    result = (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet, vac_data=ctrl.vac_flag ? vac_data : nothing)

    end_time = time() - start_time
    println("----------------------------------")
    println("Run time: $(@sprintf("%.3e", end_time)) seconds")
    println("Normal termination.")

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet, vac_data=ctrl.vac_flag ? vac_data : nothing)
```

**Change to:**
```julia
    result = (ctrl=ctrl, equil=equil, intr=intr, ffit=ffit, odet=odet, vac_data=ctrl.vac_flag ? vac_data : nothing)

    # Write linear stability results back to the IMAS data dictionary
    if dd !== nothing
        write_imas(dd, result)
        if ctrl.verbose
            println("Stability results written to dd.mhd_linear ✓")
        end
    end

    end_time = time() - start_time
    println("----------------------------------")
    println("Run time: $(@sprintf("%.3e", end_time)) seconds")
    println("Normal termination.")

    # TODO: Do not allow perturbed equilibrium calculations if zero crossings are found

    return result
```

---

## Remaining Files (Need TODO Markers)

### 4. src/Equilibrium/Equilibrium.jl
TODO: Add imas dispatch case to setup_equilibrium function

### 5. src/Equilibrium/ReadEquilibrium.jl
TODO: Add read_imas() function

### 6. Project.toml
TODO: Add IMASdd to dependencies

### 7. test/runtests.jl
TODO: Include runtests_imas.jl

---

## Exercise Instructions

1. Start with the empty files (WriteImas.jl, runtests_imas.jl)
2. Follow the TODO markers in DCON.jl
3. Manually edit Main.jl following the guide above
4. Continue with remaining files once guides are added

This way you'll practice writing all the IMAS code yourself!
