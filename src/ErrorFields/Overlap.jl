"""
Minimum toroidal boundary points per field period below which a post-hoc evaluation warns.

Helical windings alias on a coarse toroidal grid. A scan on multi-turn coils against a 128-point
reference put 16 points per period about 1 % off, 20 about 0.4 %, 24 about 0.1 %, and 32 within one
part in 1e5. A few tenths of a percent is the same size as the differences a coil-revision
comparison is trying to resolve, so a grid inherited silently from a stored deck and coarser than
`ForcingTerms.NZETA_POINTS_PER_PERIOD` is worth a warning; the default itself is converged.
"""
const MIN_NZETA_PER_PERIOD = ForcingTerms.NZETA_POINTS_PER_PERIOD

"""
    forcing_grids(rc::ResonantCoupling, equil, cfg::CoilConfig; psi) -> Vector{Tuple{Int,CoilForcingGrid}}

One boundary sampling per toroidal mode of `rc`, on the surface `psi`. The grids depend only on the
equilibrium and the grid dimensions, so building them once and evaluating every coil set against
them is what makes a post-hoc coil sweep cheap.
"""
function forcing_grids(rc::ResonantCoupling, equil::Equilibrium.PlasmaEquilibrium, cfg::CoilConfig; psi::Float64)
    return [(n, CoilForcingGrid(equil, cfg, n; psi)) for n in sort(unique(rc.n_modes))]
end

"""
    applied_spectrum(cs::CoilSet, rc::ResonantCoupling, grids) -> Vector{ComplexF64}

The root-area-weighted control-surface field b̃ that one coil set drives, on `rc`'s (m, n) column
ordering: Biot-Savart on each grid of `grids`, Fourier decomposition, then the conform through
[`rootarea_field`](@ref).

This is the conversion every overlap and every finite-difference tap goes through, kept in one place
so the conform convention has a single implementation.
"""
function applied_spectrum(cs::CoilSet, rc::ResonantCoupling, grids::AbstractVector{<:Tuple{Int,CoilForcingGrid}})
    m_low, m_high = extrema(rc.m_modes)
    modes = ForcingMode[]
    for (n, grid) in grids
        append!(modes, coil_forcing_modes(cs, grid, n, m_low, m_high))
    end
    return rootarea_field(rc, modes)
end

"""
    ResonantDriveContext

The surface a finished run offers for judging coil geometry on, gathered once: the rebuilt
equilibrium, the control surface it integrated to, the coil-grid configuration, the resonant
coupling, the dominant mode over the chosen ψ window, the axis toroidal field, and the boundary
grids. Everything needed to ask how much resonant field a coil drives, and nothing that depends on
which coils they are — so the expensive half is built once and any number of geometries are
evaluated against it.

Build it with the `gpec.h5` constructor and hand it to [`coil_overlaps`](@ref) or
[`compute_coil_sensitivities`](@ref).

## Fields

  - `equil`: the run's equilibrium, rebuilt on its own ψ grid
  - `inputs`: the parsed TOML the run used
  - `psilim`: the control surface the solve integrated to
  - `cfg`: coil configuration, with any grid or directory override already applied
  - `rc`: the resonant coupling matrix and its mode ordering
  - `dom`: the singular decomposition over the requested ψ window
  - `b_t0`: axis toroidal field magnitude, tesla
  - `grids`: one boundary sampling per toroidal mode, shared by every coil set evaluated
"""
struct ResonantDriveContext
    equil::Equilibrium.PlasmaEquilibrium
    inputs::Dict{String,Any}
    psilim::Float64
    cfg::CoilConfig
    rc::ResonantCoupling
    dom::DominantCoupling
    b_t0::Float64
    grids::Vector{Tuple{Int,CoilForcingGrid}}
end

"""
    regrid(cfg::CoilConfig; mtheta_coil=nothing, nzeta_coil=nothing, dat_dir=nothing) -> CoilConfig

`cfg` with the boundary resolution or geometry directory replaced. Each keyword left as `nothing`
keeps the value `cfg` already carries.
"""
function regrid(cfg::CoilConfig; mtheta_coil=nothing, nzeta_coil=nothing, dat_dir=nothing)
    return CoilConfig(;
        machine=cfg.machine,
        dat_dir=dat_dir === nothing ? cfg.dat_dir : String(dat_dir),
        mtheta_coil=mtheta_coil === nothing ? cfg.mtheta_coil : Int(mtheta_coil),
        nzeta_coil=nzeta_coil === nothing ? cfg.nzeta_coil : Int(nzeta_coil),
        coil_sets=cfg.coil_sets
    )
end

"""
    ResonantDriveContext(equil, rc, cfg; psilim, b_t0, psi_low=CORE_PSI_LOW, psi_high=CORE_PSI_HIGH,
                   mtheta_coil=nothing, nzeta_coil=nothing, inputs=Dict{String,Any}())
    ResonantDriveContext(h5path; psi_low=CORE_PSI_LOW, psi_high=CORE_PSI_HIGH, mtheta_coil=nothing,
                   nzeta_coil=nothing, dat_dir=nothing)

Assemble the context. The ψ window selects which rational surfaces the dominant mode is built from.
`mtheta_coil` and `nzeta_coil` override the boundary resolution the stored deck supplies, which is
otherwise inherited silently; the toroidal grid is checked against [`MIN_NZETA_PER_PERIOD`](@ref)
and warns when it is coarser. The `h5path` method lives in `Rerun.jl`, where the equilibrium is
rebuilt from the file.
"""
function ResonantDriveContext(
    equil::Equilibrium.PlasmaEquilibrium,
    rc::ResonantCoupling,
    cfg::CoilConfig;
    psilim::Real,
    b_t0::Real,
    psi_low::Real=PerturbedEquilibrium.CORE_PSI_LOW,
    psi_high::Real=PerturbedEquilibrium.CORE_PSI_HIGH,
    mtheta_coil=nothing,
    nzeta_coil=nothing,
    inputs::Dict{String,Any}=Dict{String,Any}()
)
    b_t0 > 0 || throw(ArgumentError("ResonantDriveContext: b_t0 must be a positive field magnitude (got $b_t0)"))
    cfg = regrid(cfg; mtheta_coil, nzeta_coil)
    _warn_coarse_toroidal_grid(cfg, rc)
    dom = dominant_coupling(rc; psi_low, psi_high)
    grids = forcing_grids(rc, equil, cfg; psi=Float64(psilim))
    return ResonantDriveContext(equil, inputs, Float64(psilim), cfg, rc, dom, Float64(b_t0), grids)
end

function _warn_coarse_toroidal_grid(cfg::CoilConfig, rc::ResonantCoupling)
    for n in sort(unique(rc.n_modes))
        nzeta = cfg.nzeta_coil > 0 ? cfg.nzeta_coil : ForcingTerms.NZETA_POINTS_PER_PERIOD * max(1, abs(n))
        per_period = nzeta / max(1, abs(n))
        per_period < MIN_NZETA_PER_PERIOD && @warn "Toroidal boundary grid is $(round(per_period; digits=1)) points per period for n = $n, " *
              "below the $MIN_NZETA_PER_PERIOD where helical coils are converged. Pass nzeta_coil to override the stored deck's value."
    end
end

"""
    CoilOverlap

Field names follow the ErrorFields result grammar `<quantity>[_instance][_efc][_statistic][_unit]` (see the manual's
"Result names"); every field name is also its HDF5 dataset name.

How strongly one coil set drives the resonant field, carrying every normalization the quantity is
quoted in so a caller never has to know which one a bare number was.

## Fields

  - `coil_name`: name of the coil set
  - `mode`: index of the singular mode projected onto (1 is the dominant mode)
  - `delta`: `Vᴴb̃ / B_T0`, the dimensionless overlap
  - `resonant_field_t`: `Vᴴb̃`, tesla, the same projection unnormalized
  - `resonant_fraction_percent`: `100·|Vᴴb̃| / ‖b̃‖`, how much of this set's own spectrum is resonant;
    `NaN` when the set's field at the run's toroidal mode is round-off (see [`resonant_fraction_percent`](@ref))
  - `spectrum_norm_t`: `‖b̃‖` in tesla
  - `b_t0`: the axis toroidal field the normalization used, tesla
  - `ampere_turns_kat`: the set's current as given, in kilo-ampere-turns, `|winding multiplier| × max |conductor current| / 1000`, so a current factor on this set converts to kAt by one
    multiplication and arrays with different pattern currents compare on one axis; `NaN` for a
    combination of sets
  - `spectrum`: b̃ itself, on the [`ResonantCoupling`](@ref) column ordering

The spectrum travels with the result so a diagnostic can ask *why* a coil couples as it does without
a second pass over the geometry.
"""
struct CoilOverlap
    coil_name::String
    mode::Int
    delta::ComplexF64
    resonant_field_t::ComplexF64
    resonant_fraction_percent::Float64
    spectrum_norm_t::Float64
    b_t0::Float64
    ampere_turns_kat::Float64
    spectrum::Vector{ComplexF64}
end

"""
    coil_overlaps(ctx::ResonantDriveContext, coil_sets; mode=1) -> Vector{CoilOverlap}
    coil_overlaps(h5path, coil_sets; mode=1, kwargs...) -> Vector{CoilOverlap}

Resonant overlap of each coil set against a finished run, evaluated on the run's own control
surface. `coil_sets` is any geometry, not necessarily the run's, so swapping in a new coil revision
costs one Biot-Savart pass per set and no stability or perturbed-equilibrium solve.

The `h5path` method takes [`ResonantDriveContext`](@ref)'s keywords and lives in `Rerun.jl`. For
sensitivities to rigid motion as well, use [`compute_coil_sensitivities`](@ref) instead; this is the
cheaper path when only the overlaps are wanted.
"""
function coil_overlaps(ctx::ResonantDriveContext, coil_sets::AbstractVector{CoilSet}; mode::Int=1)
    1 <= mode <= length(ctx.dom.singular_values) ||
        throw(ArgumentError("mode $mode is outside the $(length(ctx.dom.singular_values)) singular modes of the decomposition"))
    spectra = [applied_spectrum(cs, ctx.rc, ctx.grids) for cs in coil_sets]
    norms = norm.(spectra)
    nrm_ref = isempty(norms) ? 0.0 : maximum(norms)
    return map(zip(coil_sets, spectra, norms)) do (cs, b, nrm)
        raw = coupling_overlap(ctx.dom, b)[mode]
        CoilOverlap(cs.name, mode, raw / ctx.b_t0, raw, resonant_fraction_percent(raw, nrm, nrm_ref), nrm, ctx.b_t0, abs(cs.nw) * maximum(abs, cs.currents) / 1e3, b)
    end
end

const _RESONANT_FRACTION_FLOOR = 1e-8      # of the largest spectrum norm among the coil sets judged together
const _RESONANT_FRACTION_FLOOR_T = 1e-12   # absolute, tesla

"""
    resonant_fraction_percent(raw, nrm, nrm_ref) -> Float64

`100·|raw| / nrm`, the share of a coil set's applied field `b̃` (norm `nrm`) that lies along the
dominant mode (`raw = Vᴴb̃`), or `NaN` when `nrm` is below `1e-8 × nrm_ref` (the largest norm among
the coil sets judged together) or below `1e-12` T. An axisymmetric hoop at n = 1 has a field of
round-off, and the ratio of two round-off vectors lands anywhere between 1 % and 30 %; that is not a
fraction, and `0` would read as a genuine zero overlap of a finite field, which is a different fact.
"""
function resonant_fraction_percent(raw, nrm::Real, nrm_ref::Real)
    nrm > max(_RESONANT_FRACTION_FLOOR * nrm_ref, _RESONANT_FRACTION_FLOOR_T) || return NaN
    return 100 * abs(raw) / nrm
end

coil_overlaps(ctx::ResonantDriveContext, cs::CoilSet; kwargs...) = coil_overlaps(ctx, [cs]; kwargs...)

"""
    combine_overlaps(overlaps, "A" => 20.0, "B" => -15.0; name="combined") -> CoilOverlap

Coherent sum of several coil sets' spectra at relative weights, re-derived into the same
[`CoilOverlap`](@ref) so every normalization stays consistent. The field is linear in the currents,
which is the whole reason a spectrum is worth caching: a design current pattern is applied here
rather than by re-running the coils.

**Weights multiply each spectrum as it was evaluated.** They are not absolute currents, because the
sets may have been swept at any current. The usual idiom is to evaluate every conductor at 1 kA and
then weight by the design current in kA, which makes the weights read as currents without the
function having to guess.

Phase is what a coherent sum is for, and it is the first thing an ad-hoc `abs()` throws away. An
unmatched name is an error rather than a silent zero, so a coil renamed between revisions cannot
quietly drop out of a comparison.
"""
function combine_overlaps(overlaps::AbstractVector{CoilOverlap}, weights::Pair{<:AbstractString,<:Real}...; name::AbstractString="combined")
    isempty(weights) && throw(ArgumentError("combine_overlaps: give at least one coil => weight pair"))
    isempty(overlaps) && throw(ArgumentError("combine_overlaps: no overlaps to combine"))
    by_name = Dict(o.coil_name => o for o in overlaps)
    mode = first(overlaps).mode
    b_t0 = first(overlaps).b_t0
    b = zeros(ComplexF64, length(first(overlaps).spectrum))
    raw = zero(ComplexF64)

    for (nm, w) in weights
        haskey(by_name, nm) ||
            throw(ArgumentError("combine_overlaps: no coil named \"$nm\"; have $(join(sort(collect(keys(by_name))), ", "))"))
        o = by_name[nm]
        o.mode == mode ||
            throw(ArgumentError("combine_overlaps: \"$nm\" is projected onto mode $(o.mode) but the first entry is mode $mode"))
        length(o.spectrum) == length(b) ||
            throw(DimensionMismatch("combine_overlaps: \"$nm\" carries $(length(o.spectrum)) modes but the first entry carries $(length(b))"))
        b .+= w .* o.spectrum
        raw += w * o.resonant_field_t        # projection is linear, so this is exactly Vᴴ of the summed spectrum
    end

    nrm = norm(b)
    nrm_ref = maximum(o.spectrum_norm_t for o in overlaps)
    return CoilOverlap(String(name), mode, raw / b_t0, raw, resonant_fraction_percent(raw, nrm, max(nrm, nrm_ref)), nrm, b_t0, NaN, b)
end
