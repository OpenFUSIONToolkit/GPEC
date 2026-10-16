"""
    Sampling

Random draws of coil misalignment within a tolerance, the primitives a tolerance Monte Carlo
recombines with the coil sensitivities. A tolerance is one number, the radius of the disk the
coil centre (or axis endpoint) may lie in; the direction is random and the radial density is a
[`RadialDistribution`](@ref). Every sampler works on a caller-supplied `rng` and scalar
arguments so a hot loop can hold the resolved exponents and pass phases shared between
coherently moving groups.

Displacements are complex numbers `Δx + iΔy` in metres and tilts `θx + iθy` in degrees, the
rotation angles about the machine x and y axes that `apply_transforms` and the stored
sensitivities use, so an overlap moves by `S_x·real(Δ) + S_y·imag(Δ) + T_x·real(θ) + T_y·imag(θ)`.
For an axisymmetric coil `S_y = −i·S_x`, which makes that `S_x·conj(Δ)`: a single magnitude
with a free phase, the model the OMFIT tolerance tool used.
"""

"""
    RadialDistribution

Radial density of a point drawn in a disk of radius `R`: `r = R·u^p` with `u` uniform on
`[0, 1]` and `p = randpow(d)`. Subtypes: [`Flat`](@ref) (`p = 1`, uniform in radius),
[`UniformArea`](@ref) (`p = 1/2`, uniform over the disk area), [`Hollow`](@ref) (`p = 1/3`,
peaked toward the edge), [`Ring`](@ref) (`p = 0`, always on the edge), and [`PowerLaw`](@ref)
for any exponent. The radial cumulative distribution is `(r/R)^(1/p)`.
"""
abstract type RadialDistribution end

"""
    Flat <: RadialDistribution

Uniform in radius (`p = 1`): the density of `|Δ|` is constant on `[0, R]`, the OMFIT tool's
`flat` shape. Note this is not uniform over the disk area — see [`UniformArea`](@ref).
"""
struct Flat <: RadialDistribution end

"""
    UniformArea <: RadialDistribution

Uniform over the disk area (`p = 1/2`): every patch of the tolerance disk is equally likely.
"""
struct UniformArea <: RadialDistribution end

"""
    Hollow <: RadialDistribution

Peaked toward the edge (`p = 1/3`, radial CDF `(r/R)³`): the OMFIT tool's default, standing in
for a manufacturing process that uses most of its tolerance.
"""
struct Hollow <: RadialDistribution end

"""
    Ring <: RadialDistribution

Always on the edge (`p = 0`, `r = R`): the worst-case magnitude with a random direction.
"""
struct Ring <: RadialDistribution end

"""
    PowerLaw(p) <: RadialDistribution

`r = R·u^p` for a user-chosen exponent `p ≥ 0`.
"""
struct PowerLaw <: RadialDistribution
    p::Float64
    function PowerLaw(p::Real)
        p >= 0 || throw(ArgumentError("PowerLaw exponent must be ≥ 0 (got $p)"))
        return new(Float64(p))
    end
end

"""
    randpow(d::RadialDistribution) -> Float64

The exponent `p` in `r = R·u^p` of a radial distribution.
"""
randpow(::Flat) = 1.0
randpow(::UniformArea) = 0.5
randpow(::Hollow) = 1 / 3
randpow(::Ring) = 0.0
randpow(d::PowerLaw) = d.p

"""
    radial_distribution(name::AbstractString) -> RadialDistribution

The distribution named by a tolerance file's `radial_shape`: `"flat"`, `"uniform_area"`,
`"hollow"`, or `"ring"` (the `RADIAL_SHAPES` the parser accepts).
"""
function radial_distribution(name::AbstractString)
    s = lowercase(name)
    s == "flat" && return Flat()
    s == "uniform_area" && return UniformArea()
    s == "hollow" && return Hollow()
    s == "ring" && return Ring()
    throw(ArgumentError("unknown radial shape \"$name\" (expected one of $(join(RADIAL_SHAPES, ", ")))"))
end

"""
    disk_radius(rng, R, p) -> Float64

A radius drawn in `[0, R]` with density exponent `p` (see [`RadialDistribution`](@ref)):
`R·rand()^p`. The common exponents (`0`, `1/3`, `1/2`, `1`, from [`Ring`](@ref), [`Hollow`](@ref),
[`UniformArea`](@ref), [`Flat`](@ref)) take a `sqrt`/`cbrt`/identity fast path instead of the
general `^`, which is the hot loop's dominant cost otherwise.
"""
@inline function disk_radius(rng::AbstractRNG, R::Real, p::Real)
    u = rand(rng)
    pf = Float64(p)
    r = pf == 1.0 ? u : pf == 0.5 ? sqrt(u) : pf == (1 / 3) ? cbrt(u) : pf == 0.0 ? 1.0 : u^pf
    return Float64(R) * r
end

"""
    sample_disk(rng, R, p; phase=2π·rand(rng)) -> ComplexF64
    sample_disk(rng, R, d::RadialDistribution; phase) -> ComplexF64

A point in the disk of radius `R`, as `Δx + iΔy`: radius from [`disk_radius`](@ref) and a
uniform direction unless `phase` is given (coherent groups sharing a direction pass one phase
and draw their own radii).
"""
sample_disk(rng::AbstractRNG, R::Real, p::Real; phase::Real=2π * rand(rng)) = disk_radius(rng, R, p) * cis(Float64(phase))
sample_disk(rng::AbstractRNG, R::Real, d::RadialDistribution; kwargs...) = sample_disk(rng, R, randpow(d); kwargs...)

"""
    sample_uncertainty(rng, σ; phase=2π·rand(rng)) -> ComplexF64

The Gaussian uncertainty on where a coil actually sits, added to the tolerance draw: a signed
normal amplitude of standard deviation `σ` times a uniform direction, `σ·randn()·e^{iφ}` — the
OMFIT tool's construction, whose magnitude is a half-normal rather than the Rayleigh a
two-dimensional Gaussian would give. Zero `σ` returns zero without consuming random numbers.
"""
function sample_uncertainty(rng::AbstractRNG, σ::Real; phase::Union{Nothing,Real}=nothing)
    σ == 0 && return zero(ComplexF64)
    φ = phase === nothing ? 2π * rand(rng) : Float64(phase)
    return Float64(σ) * randn(rng) * cis(φ)
end

"""
    sample_additive(rng, R_shift, p_shift, R_tilt, p_tilt; phase_shift, phase_tilt) -> (shift, tilt)

The additive tolerance model: an in-plane shift drawn in the disk of radius `R_shift` metres
and, independently, a tilt drawn in the disk of radius `R_tilt` degrees, each with its own
radial exponent and an optional shared direction. Returns `(Δx + iΔy, θx + iθy)`.
"""
function sample_additive(rng::AbstractRNG, R_shift::Real, p_shift::Real, R_tilt::Real, p_tilt::Real;
    phase_shift::Real=2π * rand(rng), phase_tilt::Real=2π * rand(rng))
    shift = sample_disk(rng, R_shift, p_shift; phase=phase_shift)
    tilt = sample_disk(rng, R_tilt, p_tilt; phase=phase_tilt)
    return shift, tilt
end

"""
    sample_cylinder(rng, R, z_top, p; phase_top, phase_bot) -> (shift, tilt)

The cylinder tolerance model: the coil's axis line must lie inside a cylinder of radius `R`
metres and half-height `z_top` metres centred on the coil. Its two endpoints are drawn
independently in the top and bottom disks (radial exponent `p`, optional fixed directions);
the coil then sits at the axis's midplane crossing and is tilted by the axis's lean:

```
shift = (p_top + p_bot) / 2                      # metres, Δx + iΔy
α     = atan(|p_top − p_bot|, 2 z_top)          # lean angle
tilt  = −α · (sin φ_d + i cos φ_d)  in degrees,  φ_d = arg(p_top − p_bot)
```

Shift and tilt are therefore correlated, and the tilt magnitude never exceeds `atan(R / z_top)`.
The tilt is expressed as the rotation angles `(θx, θy)` about the machine axes that lean the
axis top toward the direction `φ_d`, in the sense `apply_transforms` uses: a positive `θx`
leans the top toward −y and a positive `θy` toward −x.

The OMFIT tool's version of this model measured the endpoint separation in units of the
tolerance radius against a height in metres and scaled the result by the tabulated tilt
tolerance, and reused the top endpoint's direction for the bottom one; this sampler works in
physical units from `(R, z_top)` alone with independent endpoint directions, so benchmarks
against that tool compare the additive model exactly and the cylinder model only in kind.
"""
function sample_cylinder(rng::AbstractRNG, R::Real, z_top::Real, p::Real;
    phase_top::Real=2π * rand(rng), phase_bot::Real=2π * rand(rng))
    z_top > 0 || throw(ArgumentError("the cylinder half-height must be positive (got $z_top)"))
    p_top = sample_disk(rng, R, p; phase=phase_top)
    p_bot = sample_disk(rng, R, p; phase=phase_bot)
    shift = (p_top + p_bot) / 2
    d = p_top - p_bot
    α = rad2deg(atan(abs(d), 2 * Float64(z_top)))
    φ = angle(d)
    tilt = -α * complex(sin(φ), cos(φ))
    return shift, tilt
end
