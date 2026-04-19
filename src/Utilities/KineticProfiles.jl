# KineticProfiles.jl
#
# Radial kinetic-profile container shared across GPEC modules that need
# electron density, electron/ion temperatures, and the three frequencies
# (toroidal rotation + electron/ion diamagnetic) as functions of the
# normalized poloidal flux ψ. SLAYER is the first consumer; PENTRC and
# future resistive-MHD modules will share this object.

using FastInterpolations
using HDF5

"""
    KineticProfiles

Radial kinetic-profile container. All six profiles are 1D cubic splines of
the normalized poloidal flux ψ ∈ [0, 1].

| field     | meaning                                | units   |
|-----------|----------------------------------------|---------|
| `n_e`     | electron density                       | m⁻³     |
| `T_e`     | electron temperature                   | eV      |
| `T_i`     | ion temperature                        | eV      |
| `omega`   | toroidal rotation                      | rad/s   |
| `omega_e` | electron diamagnetic frequency ω\\_\\*e | rad/s   |
| `omega_i` | ion diamagnetic frequency ω\\_\\*i      | rad/s   |

Construct via the keyword constructor `KineticProfiles(; psi, n_e, T_e,
T_i, omega, omega_e, omega_i)` with matched-length vectors, or via
`kinetic_profiles_from_toml` / `kinetic_profiles_from_h5`.

Evaluate all profiles at a given ψ via the call operator:

```julia
vals = kp(0.5)    # NamedTuple(n_e=..., T_e=..., ..., omega_i=...)
```
"""
struct KineticProfiles{S}
    n_e::S
    T_e::S
    T_i::S
    omega::S
    omega_e::S
    omega_i::S
end

function KineticProfiles(; psi::AbstractVector{<:Real},
                           n_e::AbstractVector{<:Real},
                           T_e::AbstractVector{<:Real},
                           T_i::AbstractVector{<:Real},
                           omega::AbstractVector{<:Real},
                           omega_e::AbstractVector{<:Real},
                           omega_i::AbstractVector{<:Real})
    xs = collect(Float64.(psi))
    for (name, v) in (("n_e", n_e), ("T_e", T_e), ("T_i", T_i),
                      ("omega", omega), ("omega_e", omega_e),
                      ("omega_i", omega_i))
        length(v) == length(xs) ||
            throw(ArgumentError("KineticProfiles: length($name) = $(length(v)) " *
                                "≠ length(psi) = $(length(xs))"))
    end
    return KineticProfiles(cubic_interp(xs, Float64.(n_e)),
                           cubic_interp(xs, Float64.(T_e)),
                           cubic_interp(xs, Float64.(T_i)),
                           cubic_interp(xs, Float64.(omega)),
                           cubic_interp(xs, Float64.(omega_e)),
                           cubic_interp(xs, Float64.(omega_i)))
end

"""
    (kp::KineticProfiles)(psi::Real) -> NamedTuple

Evaluate all profiles at `psi` and return them as a NamedTuple with fields
`(n_e, T_e, T_i, omega, omega_e, omega_i)`.
"""
(kp::KineticProfiles)(psi::Real) = (
    n_e     = kp.n_e(psi),
    T_e     = kp.T_e(psi),
    T_i     = kp.T_i(psi),
    omega   = kp.omega(psi),
    omega_e = kp.omega_e(psi),
    omega_i = kp.omega_i(psi),
)

"""
    kinetic_profiles_from_toml(section::AbstractDict) -> KineticProfiles

Build a `KineticProfiles` from an inline TOML table such as:

```toml
[SLAYER.profiles]
psi     = [0.0, 0.1, ...]
n_e     = [...]   # m⁻³
T_e     = [...]   # eV
T_i     = [...]   # eV
omega   = [...]   # rad/s
omega_e = [...]   # rad/s
omega_i = [...]   # rad/s
```

All six profile keys plus `psi` are required; lengths must match.
"""
function kinetic_profiles_from_toml(section::AbstractDict)
    required = ("psi", "n_e", "T_e", "T_i", "omega", "omega_e", "omega_i")
    missing_keys = [k for k in required if !haskey(section, k)]
    isempty(missing_keys) ||
        throw(ArgumentError("kinetic_profiles_from_toml: missing keys " *
                             "$(missing_keys). Required: $(required)."))
    _asvec(x) = Float64.(collect(x))
    return KineticProfiles(
        psi     = _asvec(section["psi"]),
        n_e     = _asvec(section["n_e"]),
        T_e     = _asvec(section["T_e"]),
        T_i     = _asvec(section["T_i"]),
        omega   = _asvec(section["omega"]),
        omega_e = _asvec(section["omega_e"]),
        omega_i = _asvec(section["omega_i"]),
    )
end

"""
    kinetic_profiles_from_h5(path; group="/") -> KineticProfiles

Load a `KineticProfiles` from an HDF5 file. The group specified by `group`
must contain the datasets `psi`, `n_e`, `T_e`, `T_i`, `omega`, `omega_e`,
`omega_i`, all the same length.
"""
function kinetic_profiles_from_h5(path::AbstractString; group::AbstractString="/")
    h5open(path, "r") do f
        g = group == "/" ? f : f[group]
        required = ("psi", "n_e", "T_e", "T_i", "omega", "omega_e", "omega_i")
        for k in required
            haskey(g, k) ||
                throw(ArgumentError("kinetic_profiles_from_h5: group " *
                                     "$(group) is missing dataset $(k). " *
                                     "Required: $(required)."))
        end
        return KineticProfiles(
            psi     = read(g["psi"]),
            n_e     = read(g["n_e"]),
            T_e     = read(g["T_e"]),
            T_i     = read(g["T_i"]),
            omega   = read(g["omega"]),
            omega_e = read(g["omega_e"]),
            omega_i = read(g["omega_i"]),
        )
    end
end
