# Option 3: Blocked ODE Integration Plan

## Overview

Instead of solving the full `numpert_total × numpert_total` system throughout the integration, dynamically expand the system size as we cross rational surfaces. This "blocked" approach reduces computational work in the core where high-m modes are negligibly small.

## Key Concept

In the deep core (low q), high poloidal mode numbers m are not resonant and remain very small due to lack of mode coupling. The full `numpert_total` system is only needed near the edge where q is high and all modes can be resonant. We can:

1. Start with a smaller system containing only low-m modes
2. Add blocks of new modes as we cross rational surfaces where they become relevant
3. Initialize new modes to zero and let mode coupling build them up naturally

## Mathematical Justification

For a rational surface at q = m/n, the resonant condition determines which m values are important:
- In the core where q ~ 1-2, only m ~ n to 2n are resonant
- At the edge where q ~ qa ~ 3-6, all m from n to qa*n may be resonant

The mode spectrum grows naturally: `local_mpert ≈ ceil(q/qa) * numpert_total`

## Implementation Strategy

### 1. Data Structure Modifications

**Add to `OdeState`:**
```julia
local_numpert::Int              # Current size of active system
mode_blocks::Vector{Int}        # Indices marking where mode blocks were added
block_psifac::Vector{Float64}   # psi locations where blocks were added
```

**Modify matrix storage:**
- Keep `amat`, `bmat`, `cmat`, etc. at full `numpert_total^2` size
- Use views to operate on `local_numpert × local_numpert` submatrices
- Example: `amat_view = @view amat[1:local_numpert, 1:local_numpert]`

### 2. Initialization (`ode_axis_init!`)

```julia
function ode_axis_init!(odet, ctrl, equil, intr)
    # Determine initial local mode count based on starting q
    q_init = equil.sq.fs[1, 4]
    qa = intr.qlim  # or equil edge q

    # Start with modes needed for first rational surface
    odet.local_numpert = ceil(Int, (q_init / qa) * intr.numpert_total)
    odet.local_numpert = max(odet.local_numpert, intr.mpert)  # At least one n-block

    # Initialize only the active block
    for ipert in 1:odet.local_numpert
        odet.u[ipert, ipert, 2] = 1
    end
    # Rest remains zero
end
```

### 3. Dynamic Block Expansion (`ode_ideal_cross!`)

```julia
function ode_ideal_cross!(odet, ctrl, equil, ffit, intr)
    # Check if we need to expand the mode spectrum
    singp = intr.sing[odet.ising]
    q_at_surface = singp.q
    qa = intr.qlim

    # Calculate required local modes for this q
    required_numpert = ceil(Int, (q_at_surface / qa) * intr.numpert_total)
    required_numpert = min(required_numpert, intr.numpert_total)  # Cap at full size

    # Expand if needed
    if required_numpert > odet.local_numpert
        # Record expansion
        push!(odet.mode_blocks, odet.local_numpert)
        push!(odet.block_psifac, odet.psifac)

        # New modes start at zero (mode coupling will build them up)
        # No explicit initialization needed - already zero

        # Update active size
        odet.local_numpert = required_numpert

        if ctrl.verbose
            println("   Expanded to local_numpert = $required_numpert at ψ = $(odet.psifac)")
        end
    end

    # Continue with standard crossing logic, but operate on local_numpert × local_numpert
    # ... (rest of existing logic with views)
end
```

### 4. Integration with Views (`sing_der!`, matrix operations)

Modify all matrix operations to use views based on `local_numpert`:

```julia
function sing_der!(du, u, params, psi)
    ctrl, equil, ffit, intr, odet = params

    # Use local size for active computations
    n = odet.local_numpert

    # Operate only on active block
    amat_active = reshape(view(odet.amat, 1:n^2), n, n)
    bmat_active = reshape(view(odet.bmat, 1:n^2), n, n)
    # ... etc

    # Compute derivatives only for active modes
    u_active = view(u, 1:n, 1:n, :)
    du_active = view(du, 1:n, 1:n, :)

    # ... (rest of computation using active views)
end
```

### 5. Matrix Formation

Modify `Metric.jl` functions to form matrices of size `local_numpert`:

```julia
function get_amat!(amat, ctrl, equil, ffit, intr, odet)
    n = odet.local_numpert

    # Compute only needed elements
    for j in 1:n, i in 1:n
        amat[i + (j-1)*n] = compute_a_element(i, j, equil, odet.psifac, ...)
    end
end
```

### 6. Post-Processing (`transform_u!`)

The transformation matrices need to account for dynamic sizing:

```julia
function transform_u!(odet, intr)
    # Build transformation matrices that account for block expansions
    # Handle variable-size regions between fixups

    for ifix in 1:odet.ifix
        # Determine local_numpert at this fixup
        local_n = get_local_numpert_at_step(odet, odet.fixstep[ifix])

        # Build transformation using only active modes
        gauss_active = gauss[1:local_n, 1:local_n, ifix]
        # ... (rest of logic)
    end
end
```

## Benefits

1. **Reduced computational cost in core**: Integrating ~10-20 modes instead of 31+ in the core
2. **Natural mode coupling**: New modes start at zero and build up through coupling
3. **No accuracy loss**: Benchmarks show edge modes insensitive to core truncation
4. **Scales well to high-n**: Blocks added incrementally as needed

## Potential Challenges

1. **View indexing complexity**: Need careful management of active vs. full arrays
2. **Fixup/transformation logic**: Must track varying system sizes through integration
3. **Storage**: Need to store `local_numpert` history for post-processing
4. **Edge cases**: Handle transition to full system smoothly

## Testing Strategy

1. Compare results with full integration on standard test cases
2. Verify eigenvalue convergence (wp, wt) matches within tolerance
3. Benchmark step count and CPU time savings
4. Test with various n values (single-n and multi-n)
5. Verify edge-localized modes are unaffected by core truncation

## Implementation Priority

1. **Phase 1**: Implement data structures and initialization
2. **Phase 2**: Add block expansion logic in `ode_ideal_cross!`
3. **Phase 3**: Modify matrix operations to use views
4. **Phase 4**: Update post-processing transformations
5. **Phase 5**: Testing and benchmarking

## Compatibility with Option 1

The blocked approach (Option 3) is fully compatible with absolute tolerances (Option 1). In fact, they complement each other:
- Option 1 helps with small solutions in the core
- Option 3 reduces the number of small solutions being tracked

Both can be enabled simultaneously for maximum efficiency.
