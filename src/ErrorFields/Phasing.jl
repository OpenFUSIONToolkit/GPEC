"""
    Phasing

Coil-array phasing: how the dominant-mode overlap of several independently powered coil
arrays (an error-field correction coil set's upper, middle and lower rows, say) depends on the
relative toroidal phases of their current patterns. The overlap is linear in each array's
field, so the map is the closed form `|Σ_k δ_k e^{iφ_k}|` on a grid of relative phases, per
kilo-ampere-turn of each array, together with the fraction of the applied field that is
resonant. No solve, no field evaluation, no optimizer: the extremes are read off the grid.
"""

"""
    PhasingMap

Dominant-mode overlap of `N` coil arrays against the `N − 1` relative phases of their current
patterns, from [`phasing_map`](@ref). Array 1 sets the reference; array `k > 1` is rotated by
the cumulative phase `Δφ_2 + … + Δφ_k`, so axis `k − 1` of the maps is the phase of array `k`
relative to array `k − 1` (the two-row EFCC convention `Δφ_ML`, `Δφ_UM`).

## Fields

  - `coil_names`: the arrays in order `[N]`
  - `phase_deg`: grid of each relative phase, degrees `[N − 1]` vectors
  - `delta_per_kat`: `|δ|` per kilo-ampere-turn of every array at each grid point `[n_1 × … × n_{N−1}]`
  - `overlap_percent`: `100·|Vᴴb̃| / ‖b̃‖`, the fraction of the applied root-area-weighted field
    that lies along the dominant mode, at each grid point
  - `delta_per_kat_each`: `|δ_k|` per kilo-ampere-turn of each array alone `[N]`
"""
struct PhasingMap
    coil_names::Vector{String}
    phase_deg::Vector{Vector{Float64}}
    delta_per_kat::Array{Float64}
    overlap_percent::Array{Float64}
    delta_per_kat_each::Vector{Float64}
end

"""
    phasing_map(sens::CoilSensitivities, dom::DominantCoupling, coil_names; mode=1, nphase=180) -> PhasingMap
    phasing_map(h5path, coil_names; psi_low=0.0, psi_high=CORE_PSI_HIGH, mode=1, nphase=180) -> PhasingMap

Scan the relative phases of the current patterns of the named coil arrays and evaluate, at
each grid point, the dominant-mode overlap per kilo-ampere-turn and the resonant fraction of
the applied field. Rotating a current pattern by `Δφ` multiplies the spectrum by `e^{iΔφ}` (the
phase of the pattern itself, `n` times the toroidal angle it is rotated through). `nphase`
points per phase axis, spanning `[0, 360)` degrees.

Each array's spectrum is its stored nominal spectrum divided by the magnitude of its
ampere-turns, `|winding_multiplier| × peak_current`. The zero of each phase axis is therefore
the array's current pattern exactly as the run specified it, with the winding sense of its
geometry file included: a negative winding multiplier and a negated current pattern each
remain in the map as the half-turn they physically are. Normalizing by the signed product
would erase how the device defines positive current in that array, which is device-specific
information the map must keep.
"""
function phasing_map(sens::CoilSensitivities, dom::DominantCoupling, coil_names::AbstractVector{<:AbstractString}; mode::Int=1, nphase::Int=180)
    length(coil_names) >= 2 || throw(ArgumentError("phasing_map needs at least two coil arrays"))
    nphase >= 2 || throw(ArgumentError("nphase must be ≥ 2"))
    1 <= mode <= length(dom.singular_values) || throw(ArgumentError("mode $mode is outside the decomposition"))
    idx = [findfirst(==(String(nm)), sens.coil_names) for nm in coil_names]
    any(isnothing, idx) && throw(ArgumentError("coil arrays not in the sensitivities: $(join(coil_names[isnothing.(idx)], ", "))"))
    v = dom.right_singular_vectors[:, mode]
    N = length(idx)
    kat = [abs(sens.winding_multiplier[i]) * sens.peak_current[i] / 1e3 for i in idx]
    all(>(0), kat) || throw(ArgumentError("every array needs a non-zero current to normalize per kilo-ampere-turn"))
    spectra = [sens.nominal_field[:, i] ./ kat[j] for (j, i) in enumerate(idx)]   # b̃ per kAt
    deltas = [dot(v, b) / sens.b_t0 for b in spectra]                            # δ per kAt
    grid = collect(range(0.0, 360.0; length=nphase + 1))[1:end-1]
    dims = ntuple(_ -> nphase, N - 1)
    delta_map = zeros(dims)
    overlap_map = zeros(dims)
    b_sum = similar(spectra[1])
    for I in CartesianIndices(dims)
        fill!(b_sum, 0)
        δ = deltas[1]
        b_sum .+= spectra[1]
        cumulative = 0.0
        for k in 2:N
            cumulative += deg2rad(grid[I[k-1]])
            rot = cis(cumulative)
            δ += deltas[k] * rot
            b_sum .+= spectra[k] .* rot
        end
        delta_map[I] = abs(δ)
        overlap_map[I] = 100 * abs(δ) * sens.b_t0 / norm(b_sum)
    end
    return PhasingMap(String.(collect(coil_names)), [copy(grid) for _ in 1:N-1], delta_map, overlap_map, abs.(deltas))
end

function phasing_map(
    h5path::AbstractString,
    coil_names::AbstractVector{<:AbstractString};
    psi_low::Real=0.0,
    psi_high::Real=PerturbedEquilibrium.CORE_PSI_HIGH,
    mode::Int=1,
    nphase::Int=180
)
    dom = dominant_coupling(ResonantCoupling(h5path); psi_low, psi_high)
    return phasing_map(CoilSensitivities(h5path), dom, coil_names; mode, nphase)
end

"""
    extreme_phasing(map::PhasingMap; quantity=:delta_per_kat, which=:max) -> (value, phases_deg)

The largest (or smallest) value of a map and the relative phases, in degrees, where it occurs.
"""
function extreme_phasing(map::PhasingMap; quantity::Symbol=:delta_per_kat, which::Symbol=:max)
    arr =
        quantity === :delta_per_kat ? map.delta_per_kat :
        quantity === :overlap_percent ? map.overlap_percent :
        throw(ArgumentError("quantity must be :delta_per_kat or :overlap_percent"))
    which in (:max, :min) || throw(ArgumentError("which must be :max or :min"))
    I = which === :max ? argmax(arr) : argmin(arr)
    return arr[I], [map.phase_deg[k][I[k]] for k in 1:length(map.phase_deg)]
end
