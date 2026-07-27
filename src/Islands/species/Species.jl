"""
    SpeciesLists

Species as a first-class dimension of the Islands solve (design `02 §1`,
Decision D3): every kinetic object is indexed by species, and the species list
is an array from day one even though Level-0 *physics* is single-bulk-ion.

Pure data structures and bookkeeping — no physics coefficients live here.
Quantities follow the Level-0 normalization (`01 §5`): densities to `n_e`,
temperatures to `T_i`, masses to the reference bulk ion, gradients as inverse
scale lengths at `r_s`.
"""
module SpeciesLists

export AbstractBackground, Maxwellian, SlowingDown
export SpeciesRole, Bulk, Trace
export Species, is_bulk, is_trace, bulk_species, trace_species
export validate_species, check_trace_criteria

# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------
"""
    AbstractBackground

Supertype of species background distributions (`02 §1.1`). `Maxwellian` is the
Level-0 background; `SlowingDown` enters at Level 2 (energetic particles) and
changes the energy-grid weight map, not the machinery.
"""
abstract type AbstractBackground end

"""
    Maxwellian(; n, T, dlnn_dr, dlnT_dr)

Maxwellian background at the rational surface `r_s` (`02 §1.1`), carrying the
strictly-local constant gradients of ordering O2.

## Fields

  - `n`       — density normalized to `n_e`.
  - `T`       — temperature normalized to `T_i`.
  - `dlnn_dr` — inverse density scale length `L̂_n⁻¹` at `r_s`.
  - `dlnT_dr` — inverse temperature scale length `L̂_T⁻¹` at `r_s`.
"""
Base.@kwdef struct Maxwellian <: AbstractBackground
    n::Float64
    T::Float64
    dlnn_dr::Float64 = 0.0
    dlnT_dr::Float64 = 0.0
end

"""
    SlowingDown(; S0, v_birth, v_crit, dlnS_dr)

Slowing-down background (isotropic at Level-2 entry, `02 §3`). Present now so
the background abstraction is exercised from day one; the Level-2 physics that
consumes it is out of Level-0 scope.

## Fields

  - `S0`      — source rate normalization.
  - `v_birth` — birth speed (normalized to the reference thermal speed).
  - `v_crit`  — critical speed (from `T_e` and composition).
  - `dlnS_dr` — inverse source scale length at `r_s`.
"""
Base.@kwdef struct SlowingDown <: AbstractBackground
    S0::Float64
    v_birth::Float64
    v_crit::Float64
    dlnS_dr::Float64 = 0.0
end

# ---------------------------------------------------------------------------
# Roles and the species struct
# ---------------------------------------------------------------------------
"""
    SpeciesRole

`Bulk` = full nonlinear participant (its `g` enters the Newton state vector).
`Trace` = linear post-processing pass with frozen bulk fields (`02 §1.2`) —
one linear solve, an additive contribution to `Δ_cos`/`Δ_sin`.
"""
@enum SpeciesRole Bulk Trace

"""
    Species(; name, Z, m, background, role, collisional_coupling=false)

One species of the Islands solve (`02 §1.1`).

## Fields

  - `name` — identifying symbol (unique within a list).
  - `Z`    — charge number (`Float64` to allow mean-charge-state impurities).
  - `m`    — mass ratio to the reference bulk ion.
  - `background` — an `AbstractBackground` (Maxwellian at Level 0).
  - `role` — `Bulk` or `Trace` (`02 §1.2`).
  - `collisional_coupling` — whether the bulk feels this species' field-particle
    back-reaction (`02 §1.3`; L1+ physics, plumbing present from day one).
"""
Base.@kwdef struct Species{B<:AbstractBackground}
    name::Symbol
    Z::Float64
    m::Float64
    background::B
    role::SpeciesRole = Bulk
    collisional_coupling::Bool = false
end

is_bulk(sp::Species) = sp.role == Bulk
is_trace(sp::Species) = sp.role == Trace

"""
    bulk_species(list) / trace_species(list)

Filter a species list by role.
"""
bulk_species(list::AbstractVector{<:Species}) = filter(is_bulk, list)
trace_species(list::AbstractVector{<:Species}) = filter(is_trace, list)

# ---------------------------------------------------------------------------
# Validation and the trace-criteria check
# ---------------------------------------------------------------------------
"""
    validate_species(list)

Structural validation of a species list: unique names, positive `Z`-magnitude,
`m`, density and temperature, and at least one `Bulk` species. Throws
`ArgumentError` on violation; returns the list for chaining.
"""
function validate_species(list::AbstractVector{<:Species})
    isempty(list) && throw(ArgumentError("species list is empty"))
    names = [sp.name for sp in list]
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate species names: $names"))
    any(is_bulk, list) || throw(ArgumentError("species list needs at least one Bulk species"))
    for sp in list
        sp.m > 0 || throw(ArgumentError("species $(sp.name): mass ratio must be positive"))
        abs(sp.Z) > 0 || throw(ArgumentError("species $(sp.name): charge must be nonzero"))
        if sp.background isa Maxwellian
            sp.background.n > 0 || throw(ArgumentError("species $(sp.name): density must be positive"))
            sp.background.T > 0 || throw(ArgumentError("species $(sp.name): temperature must be positive"))
        end
    end
    return list
end

"""
    check_trace_criteria(list; threshold=0.05)

Check every `Trace` species against both trace criteria (`02 §1.2`): charge
trace (`n_j Z_j ≪ n_e`) and current trace (`n_j ≪ n_e`), with densities already
normalized to `n_e`. A violating species gets an `@warn` recommending `Bulk`
promotion — **warn, never silently degrade** (`02 §1.2` promotion rule).
`threshold` is a diagnostic heuristic knob for "≪", not a physics coefficient.
Returns the vector of violating species names (empty when clean).
"""
function check_trace_criteria(list::AbstractVector{<:Species}; threshold::Real=0.05)
    violators = Symbol[]
    for sp in trace_species(list)
        sp.background isa Maxwellian || continue
        n = sp.background.n
        charge_frac = n * abs(sp.Z)
        if charge_frac > threshold || n > threshold
            push!(violators, sp.name)
            @warn "species $(sp.name) violates trace criteria (n Z = $(charge_frac), n = $(n) vs threshold $(threshold)); run it as Bulk" maxlog = 1
        end
    end
    return violators
end

end # module SpeciesLists
