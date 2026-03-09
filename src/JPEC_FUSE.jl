"""
    run_jpec_fuse(dd::IMASdd.dd; kwargs...) -> IMASdd.dd

FUSE module interface for JPEC linear MHD stability analysis.

This function makes JPEC callable as a FUSE module: it reads equilibrium data from
an IMAS data dictionary `dd`, runs DCON stability analysis, and writes results back
to `dd.mhd_linear`.

## Usage

**Single time-slice run** (analyze stability at one moment):
```julia
dd = run_jpec_fuse(dd; time_indices=1, nn_low=1, nn_high=5)
```

**Time-series run** (analyze stability evolution):
```julia
dd = run_jpec_fuse(dd; time_indices=:all, nn_low=1, nn_high=10)
```

## Parameters

- `dd::IMASdd.dd` — IMAS data dictionary with `dd.equilibrium` already filled by FUSE
  or another equilibrium solver (e.g., EFIT, CHEASE).

## Keyword Arguments

- `time_indices = :all` — Which time slices to analyze:
  - `:all` — run on all available time slices in `dd.equilibrium.time_slice`
  - `1` — run on first time slice only (single-slice mode)
  - `[1, 5, 10]` — run on specific time slices (vector of indices)

- `nn_low = 1` — Lowest toroidal mode number to analyze
- `nn_high = 10` — Highest toroidal mode number to analyze
- `vac_flag = true` — Include vacuum energy calculation (needed for total δW)
- `jac_type = "boozer"` — Coordinate system for equilibrium (`"boozer"` or `"pest"`)
- `psilow = 0.01` — Lower flux surface bound (normalized ψ)
- `psihigh = 0.994` — Upper flux surface bound (normalized ψ)
- `verbose = true` — Print progress messages for each time slice

## What Gets Written to `dd`

For each analyzed time slice, `dd.mhd_linear.time_slice[i]` is populated with:
- One `toroidal_mode` entry per computed eigenmode
- `n_tor` — toroidal mode number
- `energy_perturbed` — δW (negative → unstable)

## Example FUSE Workflow

```julia
using JPEC
import IMASdd

# 1. FUSE runs equilibrium solver and fills dd.equilibrium
dd = FUSE.run_efit(scenario_params)

# 2. JPEC analyzes stability across all time slices
dd = JPEC.run_jpec_fuse(dd; nn_low=1, nn_high=5, verbose=true)

# 3. FUSE reads stability results from dd.mhd_linear and decides next action
stability_margin = dd.mhd_linear.time_slice[1].toroidal_mode[1].energy_perturbed
```

## Returns

The modified `dd` with `dd.mhd_linear` populated for all analyzed time slices.

## Error Handling

- Validates that `dd.equilibrium.time_slice` exists and is non-empty
- Skips time slices with missing or invalid equilibrium data (logs warning)
- Continues to next time slice if DCON fails on one slice (unless only 1 slice requested)
"""
function run_jpec_fuse(dd::IMASdd.dd;
                       time_indices = :all,
                       nn_low::Int = 1,
                       nn_high::Int = 10,
                       vac_flag::Bool = true,
                       jac_type::String = "boozer",
                       psilow::Float64 = 0.01,
                       psihigh::Float64 = 0.994,
                       verbose::Bool = true)

    # ────────────────────────────────────────────────────────────────────────
    # 1. VALIDATE INPUT
    # ────────────────────────────────────────────────────────────────────────

    if isempty(dd.equilibrium.time_slice)
        error("run_jpec_fuse: dd.equilibrium.time_slice is empty. " *
              "FUSE must populate equilibrium data before calling JPEC.")
    end

    n_slices_available = length(dd.equilibrium.time_slice)

    # ────────────────────────────────────────────────────────────────────────
    # 2. DETERMINE WHICH TIME SLICES TO RUN
    # ────────────────────────────────────────────────────────────────────────

    if time_indices === :all
        indices_to_run = 1:n_slices_available
    elseif time_indices isa Integer
        if time_indices < 1 || time_indices > n_slices_available
            error("run_jpec_fuse: time_indices=$time_indices out of range " *
                  "(available: 1:$n_slices_available)")
        end
        indices_to_run = [time_indices]
    elseif time_indices isa AbstractVector{<:Integer}
        if any(i -> i < 1 || i > n_slices_available, time_indices)
            error("run_jpec_fuse: some time_indices out of range " *
                  "(available: 1:$n_slices_available)")
        end
        indices_to_run = time_indices
    else
        error("run_jpec_fuse: time_indices must be :all, an Int, or Vector{Int}")
    end

    if verbose
        println("\n" * "="^70)
        println("JPEC FUSE Module: Linear MHD Stability Analysis")
        println("="^70)
        println("Time slices to analyze: ", indices_to_run)
        println("Toroidal mode range:    n = $nn_low:$nn_high")
        println("Vacuum calculation:     ", vac_flag ? "enabled" : "disabled")
        println("="^70 * "\n")
    end

    # ────────────────────────────────────────────────────────────────────────
    # 3. RUN JPEC ON EACH TIME SLICE
    # ────────────────────────────────────────────────────────────────────────

    for (run_number, idx) in enumerate(indices_to_run)

        ts = dd.equilibrium.time_slice[idx]
        t  = ts.time

        if verbose
            println("[$run_number/$(length(indices_to_run))] Running time slice $idx (t = $t s)...")
        end

        # ── Set global time to this slice (required by read_imas) ────────────
        dd.global_time = t

        # ── Create temporary run directory with TOML configuration ───────────
        # DCON.Main expects to read dcon.toml from a directory.
        # We create a temporary directory and write the minimal required config.
        temp_dir = mktempdir(; prefix="jpec_fuse_")

        # Write dcon.toml with FUSE-provided parameters
        # Note: ode_flag must be true when vac_flag is true (vacuum calc needs ODE results)
        toml_content = """
        [DCON_CONTROL]
        nn_low = $nn_low
        nn_high = $nn_high
        vac_flag = $vac_flag
        ode_flag = $vac_flag
        verbose = false

        [EQUILIBRIUM_CONTROL]
        jac_type = "$jac_type"
        psilow = $psilow
        psihigh = $psihigh
        """
        write(joinpath(temp_dir, "dcon.toml"), toml_content)

        # ── Run DCON stability analysis ───────────────────────────────────────
        try
            # DCON.Main(path, dd) reads dcon.toml from path, loads equilibrium from dd,
            # runs stability analysis, and calls write_imas internally.
            result = DCON.Main(temp_dir, dd)

            if verbose
                # Report most unstable mode (first eigenvalue after sorting)
                if vac_flag && result.vac_data !== nothing
                    δW_min = real(result.vac_data.et[1])
                    n_unstable = count(x -> real(x) < 0, result.vac_data.et)
                    println("  → Most unstable δW = $δW_min  ($n_unstable unstable modes)")
                else
                    println("  → DCON completed (vacuum calculation disabled)")
                end
            end

        catch e
            @warn "Failed to run JPEC for time slice $idx (t=$t s): $e"
            if length(indices_to_run) == 1
                rethrow(e)  # If only one slice requested, fail loudly
            else
                continue    # Otherwise skip this slice and continue
            end
        finally
            # Clean up temporary directory
            rm(temp_dir; recursive=true, force=true)
        end

    end  # end loop over time slices

    if verbose
        println("\n" * "="^70)
        println("JPEC FUSE Module: Analysis Complete ✓")
        println("Results written to dd.mhd_linear.time_slice[$(indices_to_run)]")
        println("="^70 * "\n")
    end

    return dd
end
