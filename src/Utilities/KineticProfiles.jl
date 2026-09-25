# KineticProfiles.jl
#
# Radial kinetic-profile container shared across GPEC modules that need
# electron density, electron/ion temperatures, and the three frequencies
# (toroidal rotation + electron/ion diamagnetic) as functions of the
# normalized poloidal flux ψ. SLAYER is the first consumer; PENTRC and
# future resistive-MHD modules will share this object.

using FastInterpolations

"""
    KineticProfiles

Radial kinetic-profile container. All four profiles are 1D cubic splines of
the normalized poloidal flux ψ ∈ [0, 1].

| field     | meaning                      | units |
|:--------- |:---------------------------- |:----- |
| `n_e`     | electron density             | m⁻³   |
| `T_e`     | electron temperature         | eV    |
| `T_i`     | ion temperature              | eV    |
| `omega_E` | E×B rotation Ω_E, per unit n | rad/s |

Construct via the keyword constructor `KineticProfiles(; psi, n_e, T_e, T_i, omega_E)` with
matched-length vectors. The SLAYER runner builds this object from a standardized
kinetic-profile file via `Equilibrium.read_kinetic_file`. Diamagnetic frequencies are not
carried here: `build_slayer_inputs` derives ω_*e and ω_*i per surface from the density and
temperature splines.

Evaluate all profiles at a given ψ via the call operator:

```julia
vals = kp(0.5)    # NamedTuple(n_e=..., T_e=..., T_i=..., omega_E=...)
```
"""
struct KineticProfiles{S}
    n_e::S
    T_e::S
    T_i::S
    omega_E::S
end

function KineticProfiles(; psi::AbstractVector{<:Real},
    n_e::AbstractVector{<:Real},
    T_e::AbstractVector{<:Real},
    T_i::AbstractVector{<:Real},
    omega_E::AbstractVector{<:Real})
    xs = collect(Float64.(psi))
    for (name, v) in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i),
        ("omega_E", omega_E))
        length(v) == length(xs) ||
            throw(ArgumentError("KineticProfiles: length($name) = $(length(v)) " *
                                "≠ length(psi) = $(length(xs))"))
    end
    return KineticProfiles(cubic_interp(xs, Float64.(n_e)),
        cubic_interp(xs, Float64.(T_e)),
        cubic_interp(xs, Float64.(T_i)),
        cubic_interp(xs, Float64.(omega_E)))
end

"""
    (kp::KineticProfiles)(psi::Real) -> NamedTuple

Evaluate all profiles at `psi` and return them as a NamedTuple with fields
`(n_e, T_e, T_i, omega_E)`.
"""
(kp::KineticProfiles)(psi::Real) = (
    n_e=kp.n_e(psi),
    T_e=kp.T_e(psi),
    T_i=kp.T_i(psi),
    omega_E=kp.omega_E(psi)
)
