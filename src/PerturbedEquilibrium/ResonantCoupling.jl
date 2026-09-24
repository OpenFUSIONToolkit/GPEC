"""
    ResonantCoupling

Everything needed to project an applied control-surface field onto the resonant responses of a
perturbed-equilibrium solve, detached from the run that produced it. Build it in memory from a
`PerturbedEquilibriumState` and its `ForceFreeStatesResult`, or post hoc from a `gpec.h5`; then
window and decompose it with [`dominant_coupling`](@ref) and evaluate coil spectra with
[`rootarea_field`](@ref) and [`coupling_overlap`](@ref) as often as needed without re-running.

## Fields

  - `C::Matrix{ComplexF64}` - coupling from the applied root-area-weighted field b̃ to the resonant
    area-weighted field, one row per resonant (surface, n) pair `[n_rational × numpert_total]`
  - `rational_psi`, `rational_q`, `rational_m`, `rational_n` - row labels `[n_rational]`
  - `m_modes`, `n_modes` - column labels: the poloidal and toroidal mode number of each entry of an
    applied spectrum `[numpert_total]`
  - `flux_conform::Matrix{ComplexF64}` - `R = S·A`, mapping b̃ to the unit-norm flux the forcing
    loaders and coil integration produce, `Φ_x = R·b̃` `[numpert_total × numpert_total]`
"""
struct ResonantCoupling
    C::Matrix{ComplexF64}
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
    rational_m::Vector{Int}
    rational_n::Vector{Int}
    m_modes::Vector{Int}
    n_modes::Vector{Int}
    flux_conform::Matrix{ComplexF64}
end

"""
    ResonantCoupling(state::PerturbedEquilibriumState, ffs::ForceFreeStatesResult)

In-memory construction from a solve: the coupling matrix and row labels come from `state`, the
column ordering from `ffs`, and the conform operator from `state` when the response step ran or
from the equilibrium otherwise. Errors when `state` carries no singular-coupling matrix.
"""
function ResonantCoupling(state::PerturbedEquilibriumState, ffs::ForceFreeStatesResult)
    isempty(state.C_resonant_area_weighted_field) &&
        throw(ArgumentError("the perturbed-equilibrium state carries no singular-coupling matrix (compute_singular_coupling off, or no resonant surfaces)"))
    flux_conform = if isempty(state.rootarea_to_area_weight)
        S, A = build_control_surface_rootarea_to_area_weight(ffs.equil, ffs)
        S .* A
    else
        state.rootarea_to_area_weight .* state.surface_area
    end
    N = ffs.numpert_total
    m_modes = [(i - 1) % ffs.mpert + ffs.mlow for i in 1:N]
    n_modes = [(i - 1) ÷ ffs.mpert + ffs.nlow for i in 1:N]
    return ResonantCoupling(state.C_resonant_area_weighted_field, state.rational_psi, state.rational_q,
        state.rational_m_res, state.rational_n, m_modes, n_modes, flux_conform)
end

"""
    ResonantCoupling(h5path::AbstractString)

Post-hoc construction from a `gpec.h5` written with both the response and singular-coupling
steps enabled: reads `PerturbedEquilibrium/SingularCoupling/`, the stored conform operator under
`PerturbedEquilibrium/ResponseMatrices/`, and the column labels from `Info/mn_index`.
"""
function ResonantCoupling(h5path::AbstractString)
    h5open(h5path, "r") do f
        haskey(f, "PerturbedEquilibrium/SingularCoupling/C_resonant_area_weighted_field") ||
            throw(ArgumentError("$h5path has no singular-coupling matrix"))
        haskey(f, "PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator") ||
            throw(ArgumentError("$h5path has no control-surface conform operator (run with compute_response = true)"))
        haskey(f, "Info/mn_index") || throw(ArgumentError("$h5path has no Info/mn_index mode labels"))
        sc = f["PerturbedEquilibrium/SingularCoupling"]
        mn = read(f["Info/mn_index"])
        S = read(f["PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator"])
        A = read(f["PerturbedEquilibrium/ResponseMatrices/surface_area"])
        return ResonantCoupling(read(sc["C_resonant_area_weighted_field"]), read(sc["rational_psi"]), read(sc["rational_q"]),
            read(sc["rational_m"]), read(sc["rational_n"]), mn[:, 1], mn[:, 2], S .* A)
    end
end

"""
    rootarea_field(rc::ResonantCoupling, modes) -> Vector{ComplexF64}

Root-area-weighted control-surface field b̃ of an applied spectrum given in the unit-norm
(Φ_x) convention — what the forcing-file loaders and the coil integration produce. `modes` is a
`Vector{ForcingMode}`, placed on `rc`'s (m, n) column ordering (modes outside the basis are
ignored), or an already-ordered amplitude vector. The result is conformed with R⁻¹
(`Φ_x = R·b̃`) and is the vector the coupling matrix and its singular vectors act on.
"""
function rootarea_field(rc::ResonantCoupling, modes::AbstractVector{ForcingMode})
    flux = zeros(ComplexF64, length(rc.m_modes))
    for mode in modes
        j = findfirst(k -> rc.m_modes[k] == mode.m && rc.n_modes[k] == mode.n, eachindex(rc.m_modes))
        j === nothing || (flux[j] += mode.amplitude)
    end
    return rootarea_field(rc, flux)
end

function rootarea_field(rc::ResonantCoupling, flux::AbstractVector{<:Number})
    length(flux) == length(rc.m_modes) || throw(DimensionMismatch("applied spectrum has $(length(flux)) entries; the basis has $(length(rc.m_modes))"))
    return rc.flux_conform \ Vector{ComplexF64}(flux)
end

"""
    DominantCoupling

Singular-value decomposition `U·diag(σ)·Vᴴ` of a resonant coupling matrix over a set of retained
rational surfaces, produced by [`dominant_coupling`](@ref).

## Fields

  - `singular_values` - σ, descending `[rank]`
  - `right_singular_vectors` - V: applied-b̃ spectra ranked by resonant drive `[numpert_total × rank]`
  - `left_singular_vectors` - U: resonant-field patterns over the retained surfaces `[n_retained × rank]`
  - `rational_index` - rows of the coupling matrix (and its `rational_*` labels) that were retained `[n_retained]`
  - `m_modes`, `n_modes` - the (m, n) basis `V` is expressed on, copied from the coupling it came
    from `[numpert_total]`; empty when built from a bare matrix, which carries no labels
"""
struct DominantCoupling
    singular_values::Vector{Float64}
    right_singular_vectors::Matrix{ComplexF64}
    left_singular_vectors::Matrix{ComplexF64}
    rational_index::Vector{Int}
    m_modes::Vector{Int}
    n_modes::Vector{Int}
end

DominantCoupling(s, V, U, idx) = DominantCoupling(s, V, U, idx, Int[], Int[])

"""
    check_mode_basis(dom::DominantCoupling, m_modes, n_modes, what::AbstractString)

Error when `dom` was decomposed on a different (m, n) ordering than the spectrum being projected.

`V` is only meaningful against the ordering it was decomposed on, and a decomposition from another
equilibrium or another `mlow` can carry the same number of columns while meaning something else —
so a length check passes and the projection returns a plausible number. A no-op when `dom` carries
no basis, which is the bare-matrix construction.
"""
function check_mode_basis(dom::DominantCoupling, m_modes::AbstractVector{Int}, n_modes::AbstractVector{Int}, what::AbstractString)
    isempty(dom.m_modes) && return nothing
    (dom.m_modes == m_modes && dom.n_modes == n_modes) && return nothing
    throw(ArgumentError("$what: the decomposition is on a different (m, n) basis than the spectrum — " *
                        "m $(extrema(dom.m_modes)) vs $(extrema(m_modes)), n $(extrema(dom.n_modes)) vs $(extrema(n_modes)). " *
                        "Decompose the same ResonantCoupling the spectrum was built from."))
end

"""
Upper edge of the core window, ψ_N = 0.9, the default window of the dominant-mode SVD.
Rational surfaces closer to the edge couple strongly to the applied field on any tokamak and
would dominate the decomposition, while the locking physics the overlap feeds concerns the core.
"""
const CORE_PSI_HIGH = 0.9

"""
    dominant_coupling(rc::ResonantCoupling; psi_low=0.0, psi_high=CORE_PSI_HIGH) -> DominantCoupling
    dominant_coupling(C, rational_psi; psi_low=0.0, psi_high=CORE_PSI_HIGH) -> DominantCoupling

Singular-value decomposition of the resonant coupling matrix restricted to the rational
surfaces with `psi_low ≤ ψ_N ≤ psi_high`, by default the core window `ψ_N ≤ 0.9`
([`CORE_PSI_HIGH`](@ref)). With `C_w = U·diag(σ)·Vᴴ` on the retained rows, the right singular
vectors `V[:, k]` are the applied-field spectra ordered by how strongly they drive resonant
field inside the window, and the singular values are coordinate-invariant. The overlap of an
applied spectrum `b̃` with mode `k` is `dot(V[:, k], b̃)` — see [`coupling_overlap`](@ref) — and
the resonant field it drives is `σ[k]` times that coefficient; `k = 1` is the dominant mode
[Park 2007b]. The window is an analysis choice, so re-evaluate it freely on the same `rc`.

Throws `ArgumentError` when the window contains no rational surface.
"""
function dominant_coupling(C::AbstractMatrix{ComplexF64}, rational_psi::AbstractVector{<:Real}; psi_low::Real=0.0, psi_high::Real=CORE_PSI_HIGH)
    size(C, 1) == length(rational_psi) ||
        throw(DimensionMismatch("C has $(size(C, 1)) rows but rational_psi has $(length(rational_psi)) entries"))
    rational_index = findall(ψ -> psi_low <= ψ <= psi_high, rational_psi)
    isempty(rational_index) && throw(ArgumentError("no rational surfaces with ψ_N in [$psi_low, $psi_high]"))
    F = svd(C[rational_index, :])
    return DominantCoupling(F.S, Matrix(F.V), Matrix(F.U), rational_index)
end

function dominant_coupling(rc::ResonantCoupling; kwargs...)
    d = dominant_coupling(rc.C, rc.rational_psi; kwargs...)
    # Carry the basis so a later projection can be checked rather than assumed.
    return DominantCoupling(d.singular_values, d.right_singular_vectors, d.left_singular_vectors,
        d.rational_index, copy(rc.m_modes), copy(rc.n_modes))
end

"""
    coupling_overlap(dom::DominantCoupling, b̃) -> Vector{ComplexF64}
    coupling_overlap(dom, rc::ResonantCoupling, modes) -> Vector{ComplexF64}

Coefficients `Vᴴ·b̃` of an applied root-area-weighted field on the singular modes of `dom`; the
first entry is the overlap with the dominant mode.

**These are unnormalized**, in the units of `b̃` itself. The dimensionless overlap δ that the
error-field literature quotes divides by the axis toroidal field, and the resonant *fraction* of a
coil's own spectrum divides by `‖b̃‖` instead. `ErrorFields.CoilOverlap` carries all three together
so a caller never has to know which normalization a bare number was in. `dom.singular_values .* coupling_overlap(dom, b̃)`
is the resonant field each mode drives, with `dom.left_singular_vectors` giving its pattern over the
retained surfaces. The second form conforms `modes` through [`rootarea_field`](@ref) first.
"""
coupling_overlap(dom::DominantCoupling, b̃::AbstractVector{<:Number}) = dom.right_singular_vectors' * b̃
coupling_overlap(dom::DominantCoupling, rc::ResonantCoupling, modes) = coupling_overlap(dom, rootarea_field(rc, modes))

"""
    compute_dominant_coupling!(state, ctrl)

Store the dominant resonant-coupling decomposition of `state.C_resonant_area_weighted_field`
over the core window (`ψ_N ≤ CORE_PSI_HIGH`) on `state` as a run summary, with the applied
forcing's coefficients on each mode when the response spectrum is available. Other windows are
evaluated post hoc on a [`ResonantCoupling`](@ref). Returns without change when no coupling matrix exists.
"""
function compute_dominant_coupling!(state::PerturbedEquilibriumState, ctrl::PerturbedEquilibriumControl)
    isempty(state.C_resonant_area_weighted_field) && return nothing
    dom = dominant_coupling(state.C_resonant_area_weighted_field, state.rational_psi)
    state.dominant_singular_values = dom.singular_values
    state.dominant_right_singular_vectors = dom.right_singular_vectors
    state.dominant_left_singular_vectors = dom.left_singular_vectors
    state.dominant_rational_index = dom.rational_index
    isempty(state.forcing_b_rootarea) || (state.dominant_forcing_overlap = coupling_overlap(dom, state.forcing_b_rootarea))
    ctrl.verbose &&
        @info "Dominant resonant coupling: $(length(dom.singular_values)) modes over " *
              "$(length(dom.rational_index)) rational surfaces, σ₁ = $(@sprintf("%.3e", dom.singular_values[1]))"
    return nothing
end
