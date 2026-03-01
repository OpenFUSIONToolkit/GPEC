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

---

## DETAILED GUIDES FOR REMAINING FILES

### src/Equilibrium/Equilibrium.jl

**Location:** In the `setup_equilibrium` function

**Find this code:**
```julia
elseif control.eq_type == "efit"
    # EFIT g-file path
    equil_input = read_efit(control.eq_filename)
```

**Add after it:**
```julia
elseif control.eq_type == "imas"
    # IMAS data dictionary (dd provided as second argument)
    equil_input = read_imas(dd)  # dd is the second parameter to setup_equilibrium
```

**Also update function signature** from:
```julia
function setup_equilibrium(config::EquilibriumConfig)
```

**To:**
```julia
function setup_equilibrium(config::EquilibriumConfig, dd::Union{IMASdd.dd, Nothing}=nothing)
```

---

### src/Equilibrium/ReadEquilibrium.jl

**Add this complete function at the end of the file:**

```julia
"""
    read_imas(dd::IMASdd.dd) -> DirectRunInput

Read equilibrium from IMAS data dictionary and return DirectRunInput for JPEC.

Performs COCOS 11 → 2 conversion (IMAS uses COCOS 11, JPEC uses COCOS 2):
- psi_norm: multiply by 1/(2π)
- q: multiply by 2π
"""
function read_imas(dd::IMASdd.dd)
    
    # TODO: Extract equilibrium data from dd
    # Use dd.global_time to find the correct time slice
    # Get psi, q, p, F profiles from dd.equilibrium.time_slice
    # Convert COCOS 11 → 2
    # Return DirectRunInput(...)
    
end
```

---

### Project.toml

**Find the `[deps]` section** and add:
```toml
EFIT = "898e9812-cceb-487a-a3a7-88c7b6b62f2d"
IMASdd = "c5a45a97-b3f9-491c-b9a7-aa88c3bc0067"
```

**Find the `[compat]` section** and add:
```toml
EFIT = "8"
IMASdd = "8"
```

---

### test/runtests.jl

**Find where other test files are included** (look for lines like `include("runtests_equil.jl")`).

**Add:**
```julia
include("runtests_imas.jl")
```

---

## Summary

**Total edits needed across all files:**

1. ✅ WriteImas.jl - Implement complete function (~77 lines)
2. ✅ runtests_imas.jl - Write 2 test sets (~156 lines)
3. ✅ DCON.jl - Add 2 lines (import + include)
4. ⏳ Main.jl - 4 modifications (wrapper, signature, if/else, write_imas call)
5. ⏳ Equilibrium.jl - Add imas dispatch case + update signature
6. ⏳ ReadEquilibrium.jl - Implement read_imas function
7. ⏳ Project.toml - Add 4 lines (2 deps + 2 compat)
8. ⏳ runtests.jl - Add 1 line (include)

**Start with files 1-3, then work through 4-8 using this guide!**
