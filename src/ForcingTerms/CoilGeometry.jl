"""
    CoilGeometry

Structs and I/O for 3D coil geometries used in Biot-Savart field calculations.

Coil geometry files use the Fortran GPEC ASCII format:
  - Header: `ncoil s nsec nw`
  - Data: `ncoil × s × nsec` rows, each with `X Y Z` in meters (Cartesian, lab frame)
"""

# Sentinel value used when nom coordinates are not user-specified (matches Fortran coil.F)
const _NOM_UNSET_SENTINEL = 1e10
# Threshold below which a nom value is treated as user-specified (vs. the unset sentinel)
const _NOM_THRESHOLD = 1e3

"""
    CoilSet

Geometry and current data for a single coil set (one .dat file).

## Fields

  - `name`: coil set identifier (e.g. "il", "iu")
  - `ncoil`: number of independent conductors
  - `s`: sub-systems (strand groups) per conductor
  - `nw`: winding multiplier (turns per conductor element)
  - `nsec`: number of Cartesian points per strand
  - `x`, `y`, `z`: conductor coordinates `[ncoil, s, nsec]` in meters (Cartesian)
  - `currents`: current per conductor `[ncoil]` in Amperes
"""
struct CoilSet
    name::String
    ncoil::Int
    s::Int
    nw::Float64
    nsec::Int
    x::Array{Float64,3}
    y::Array{Float64,3}
    z::Array{Float64,3}
    currents::Vector{Float64}
end

"""
    CoilSetConfig

Per-coil-set configuration from a `[[ForcingTerms.coil_set]]` TOML block.

A coil set comes from one of several sources, selected by `source`:

  - `""` or `"file"`: read a geometry file — `.dat` (legacy ASCII) or `.h5` (modern), chosen
    by extension. The file path comes from `dat_file`/`h5_file` or the `machine`+`name` convention.
  - `"pf_hoop"`: an analytic horizontal circular loop (`radius`, `height`).
  - `"window_pane"`: an analytic rectangular picture-frame array, placed EITHER by explicit
    corners (`rz_corners`) OR by physical standoff from the plasma surface (`standoff`,
    `poloidal_angle`, `poloidal_length`, `poloidal_tilt`); shared toroidal layout via
    `ncoil_gen`, `gap_fraction`, `phi0`.
  - `"helical"`: an analytic helical array on a circular cross-section torus
    (`R0`, `a`, `m_hel`, `n_hel`, `theta_lo`, `theta_hi`, `n_coils`, `gap_fraction`, `phi0`, `offset`).

Shifts/tilts apply to every source (the analytic geometry is built first, then transformed).

## Fields

  - `name`: coil set name; resolves to `{dat_dir}/{machine}_{name}.dat`, or selects a set by name
    from a multi-set `.h5` file
  - `source`: coil source discriminator (see above); empty/`"file"` reads a geometry file
  - `dat_file`: explicit path to .dat file (overrides `machine`+`name` convention)
  - `h5_file`: explicit path to .h5 file (multi-set; `name` selects which set, else the sole set)
  - `currents`: current [A] per conductor `[ncoil]`; shorter arrays pad with zeros
  - `shiftx`, `shifty`, `shiftz`: per-conductor translation [m] `[ncoil]`
  - `tiltx`, `tilty`, `tiltz`: per-conductor tilt in degrees (or meters if `tilt_in_meters`)
  - `xnom`, `ynom`, `znom`: explicit rotation center [m]; defaults to arc-length-weighted center of mass
  - `n_tilt`: toroidal mode number for tilt/shift modulation; -1 means inherit run's n
  - `tilt_in_meters`: interpret tilt as displacement [m] instead of angle [degrees]

Analytic-source parameters (used only for the matching `source`):

  - `radius`, `height`: PF hoop circle radius and Z height [m]
  - `ncoil_gen`: number of generated coils (window-pane / helical arrays)
  - `rz_corners`: window-pane cross-section as two opposite `[R, Z]` corners in m (corner mode)
  - `gap_fraction`: toroidal gap between adjacent coils as a fraction of their spacing
  - `phi0`: toroidal angle offset of the first generated coil [rad]
  - `R0`, `a`: helical torus major radius and minor (circle) radius [m]
  - `m_hel`, `n_hel`: helical poloidal/toroidal mode numbers setting the pitch
  - `theta_lo`, `theta_hi`: helical poloidal extent in degrees; a full turn when `theta_hi-theta_lo == 360`
  - `n_coils`: number of helical coils in the array
  - `offset`: helical coil radial offset from the torus surface [m]
  - `nsec_gen`: optional override of the generated point count per coil

Window-pane standoff placement (alternative to `rz_corners`; needs an equilibrium):

  - `standoff`: outward-normal distance from the control surface to the frame center [m]
  - `poloidal_angle`: cylindrical poloidal angle of the frame center, in degrees
  - `poloidal_length`: poloidal leg span (corner-to-corner) along the surface tangent [m]
  - `poloidal_tilt`: extra tilt of the legs in degrees; 0 means tangential to the surface
"""
Base.@kwdef struct CoilSetConfig
    name::String = ""
    source::String = ""
    dat_file::String = ""
    h5_file::String = ""
    currents::Vector{Float64} = Float64[]
    shiftx::Vector{Float64} = Float64[]
    shifty::Vector{Float64} = Float64[]
    shiftz::Vector{Float64} = Float64[]
    tiltx::Vector{Float64} = Float64[]
    tilty::Vector{Float64} = Float64[]
    tiltz::Vector{Float64} = Float64[]
    xnom::Vector{Float64} = Float64[]
    ynom::Vector{Float64} = Float64[]
    znom::Vector{Float64} = Float64[]
    n_tilt::Int = -1
    tilt_in_meters::Bool = false

    # Analytic-source parameters
    radius::Float64 = 0.0
    height::Float64 = 0.0
    ncoil_gen::Int = 0
    rz_corners::Vector{Vector{Float64}} = Vector{Float64}[]
    gap_fraction::Float64 = 0.0
    phi0::Float64 = 0.0
    R0::Float64 = 0.0
    a::Float64 = 0.0
    m_hel::Int = 0
    n_hel::Int = 0
    theta_lo::Float64 = 0.0
    theta_hi::Float64 = 360.0
    n_coils::Int = 0
    offset::Float64 = 0.0
    nsec_gen::Int = 0

    # Window-pane standoff placement (alternative to rz_corners)
    standoff::Float64 = NaN
    poloidal_angle::Float64 = NaN
    poloidal_length::Float64 = 0.0
    poloidal_tilt::Float64 = 0.0
end

"""
    CoilConfig

Top-level coil configuration from the `[ForcingTerms]` TOML section when
`forcing_data_format = "coil"`.

## Fields

  - `machine`: machine name prefix for .dat files (e.g. "d3d")
  - `dat_dir`: directory containing .dat files; defaults to bundled `coil_geometries/`
  - `mtheta_coil`: poloidal grid resolution for boundary field evaluation (default: 480)
  - `nzeta_coil`: toroidal grid resolution; 0 = auto (32 × n)
  - `coil_sets`: parsed `[[ForcingTerms.coil_set]]` blocks
"""
Base.@kwdef struct CoilConfig
    machine::String = ""
    dat_dir::String = ""
    mtheta_coil::Int = 480
    nzeta_coil::Int = 0
    coil_sets::Vector{CoilSetConfig} = CoilSetConfig[]
end

"""
    CoilConfig(ft_ctrl::ForcingTermsControl)

Construct a `CoilConfig` from a `ForcingTermsControl` populated from TOML.
"""
function CoilConfig(ft_ctrl::ForcingTermsControl)
    dat_dir = isempty(ft_ctrl.dat_dir) ?
              joinpath(@__DIR__, "coil_geometries") :
              ft_ctrl.dat_dir

    coil_sets = CoilSetConfig[_parse_coil_set_config(d) for d in ft_ctrl.coil_sets_raw]

    return CoilConfig(;
        machine=ft_ctrl.machine,
        dat_dir=dat_dir,
        mtheta_coil=ft_ctrl.mtheta_coil,
        nzeta_coil=ft_ctrl.nzeta_coil,
        coil_sets=coil_sets
    )
end

function _parse_coil_set_config(d::Dict{String,Any})
    fvec(key) = Float64.(get(d, key, Float64[]))
    # rz_corners arrives as a Vector of [R, Z] pairs (TOML array of arrays)
    corners = [Float64.(c) for c in get(d, "rz_corners", Vector{Float64}[])]
    return CoilSetConfig(;
        name=get(d, "name", ""),
        source=get(d, "source", ""),
        dat_file=get(d, "dat_file", ""),
        h5_file=get(d, "h5_file", ""),
        currents=fvec("currents"),
        shiftx=fvec("shiftx"),
        shifty=fvec("shifty"),
        shiftz=fvec("shiftz"),
        tiltx=fvec("tiltx"),
        tilty=fvec("tilty"),
        tiltz=fvec("tiltz"),
        xnom=fvec("xnom"),
        ynom=fvec("ynom"),
        znom=fvec("znom"),
        n_tilt=get(d, "n_tilt", -1),
        tilt_in_meters=get(d, "tilt_in_meters", false),
        radius=Float64(get(d, "radius", 0.0)),
        height=Float64(get(d, "height", 0.0)),
        ncoil_gen=get(d, "ncoil_gen", 0),
        rz_corners=corners,
        gap_fraction=Float64(get(d, "gap_fraction", 0.0)),
        phi0=Float64(get(d, "phi0", 0.0)),
        R0=Float64(get(d, "R0", 0.0)),
        a=Float64(get(d, "a", 0.0)),
        m_hel=get(d, "m_hel", 0),
        n_hel=get(d, "n_hel", 0),
        theta_lo=Float64(get(d, "theta_lo", 0.0)),
        theta_hi=Float64(get(d, "theta_hi", 360.0)),
        n_coils=get(d, "n_coils", 0),
        offset=Float64(get(d, "offset", 0.0)),
        nsec_gen=get(d, "nsec_gen", 0),
        standoff=Float64(get(d, "standoff", NaN)),
        poloidal_angle=Float64(get(d, "poloidal_angle", NaN)),
        poloidal_length=Float64(get(d, "poloidal_length", 0.0)),
        poloidal_tilt=Float64(get(d, "poloidal_tilt", 0.0))
    )
end

"""
    read_coil_dat(filepath::String) -> CoilSet

Parse an ASCII coil geometry file.

Format:

```
ncoil s nsec nw
X₁ Y₁ Z₁
X₂ Y₂ Z₂
...
```

Data rows are ordered: for each conductor 1:ncoil, for each sub-system 1:s, read nsec points.
Coordinates are Cartesian in meters (lab frame).
"""
function read_coil_dat(filepath::String)
    isfile(filepath) || error("Coil geometry file not found: $filepath")

    lines = readlines(filepath)
    # Skip comment/blank lines to find header
    header_idx = findfirst(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), lines)
    isnothing(header_idx) && error("Empty coil file: $filepath")

    header = split(strip(lines[header_idx]))
    length(header) >= 4 || error("Invalid header in $filepath: expected 'ncoil s nsec nw'")

    ncoil = parse(Int, header[1])
    s = parse(Int, header[2])
    nsec = parse(Int, header[3])
    nw = parse(Float64, header[4])

    x = zeros(ncoil, s, nsec)
    y = zeros(ncoil, s, nsec)
    z = zeros(ncoil, s, nsec)

    row = header_idx + 1
    for j in 1:ncoil
        for k in 1:s
            for l in 1:nsec
                # Skip blank/comment lines between data rows
                while row <= length(lines) &&
                      (isempty(strip(lines[row])) || startswith(strip(lines[row]), "#"))
                    row += 1
                end
                row > length(lines) && error(
                    "Unexpected end of file in $filepath at conductor $j, sub $k, point $l"
                )
                vals = split(strip(lines[row]))
                length(vals) >= 3 || error(
                    "Expected 3 columns at line $row in $filepath, got $(length(vals))"
                )
                x[j, k, l] = parse(Float64, vals[1])
                y[j, k, l] = parse(Float64, vals[2])
                z[j, k, l] = parse(Float64, vals[3])
                row += 1
            end
        end
    end

    name = splitext(basename(filepath))[1]
    return CoilSet(name, ncoil, s, nw, nsec, x, y, z, zeros(ncoil))
end

"""
    make_pf_hoop(; radius, height, nsec=361, name="pf_hoop") -> CoilSet

Build a single horizontal circular loop (PF hoop) of `radius` [m] centered on the
machine axis at vertical position `height` [m], discretized into `nsec` points.

The loop is closed (point `nsec` coincides with point 1). A bare hoop is axisymmetric
and produces only an n=0 field; tilting or shifting it (via [`apply_transforms`](@ref))
breaks axisymmetry to produce n≠0 forcing. Currents are zeroed; `load_coil_sets` fills them.
"""
function make_pf_hoop(; radius::Real, height::Real, nsec::Int=361, name::String="pf_hoop")
    radius > 0 || error("make_pf_hoop: radius must be > 0 (got $radius)")
    nsec >= 361 || error("make_pf_hoop: nsec must be >= 361 (got $nsec); a full poloidal hoop needs at least ~1° resolution to be smooth")
    phis = range(0; length=nsec, stop=2π)  # closed: phis[end] == phis[1] + 2π
    x = reshape([radius * cos(p) for p in phis], 1, 1, nsec)
    y = reshape([radius * sin(p) for p in phis], 1, 1, nsec)
    z = reshape(fill(Float64(height), nsec), 1, 1, nsec)
    return CoilSet(name, 1, 1, 1.0, nsec, x, y, z, zeros(1))
end

"""
    surface_point_and_normal(equil, theta_cyl; psi=equil.rzphi_xs[end]) -> (R, Z, nR, nZ)

Locate the control-surface point at cylindrical poloidal angle `theta_cyl` (radians) and return
it with the outward unit normal `(nR, nZ)` in the R-Z plane.

The flux-surface splines are parametrized by the SFL angle `theta_sfl`, related to the cylindrical
angle by `theta_cyl = 2π*(theta_sfl + rzphi_offset(psi, theta_sfl))`. Since the offset is a small
smooth perturbation, `theta_sfl` is recovered by a damped fixed-point iteration. The tangent is
taken by central-differencing `R(theta_sfl), Z(theta_sfl)`, and the normal `(dZ/dθ, -dR/dθ)` is
normalized and oriented outward (positive dot with `(R-ro, Z-zo)`), matching the normal-flux
convention of `project_normal_flux!`.
"""
function surface_point_and_normal(equil, theta_cyl::Real; psi::Real=equil.rzphi_xs[end])
    hint = (Ref(1), Ref(1))
    target = mod(theta_cyl / (2π), 1.0)  # normalized cylindrical angle in [0,1)

    # Solve theta_cyl/2π = theta_sfl + offset(theta_sfl) for theta_sfl (offset slope ≪ 1).
    theta_sfl = target
    for _ in 1:50
        off = equil.rzphi_offset((psi, mod(theta_sfl, 1.0)); hint=hint)
        residual = mod(theta_sfl + off - target, 1.0)
        residual -= round(residual)  # wrap to (-0.5, 0.5]
        abs(residual) < 1e-13 && break
        theta_sfl -= residual
    end
    theta_sfl = mod(theta_sfl, 1.0)

    point(tn) = begin
        tw = mod(tn, 1.0)
        r_minor = sqrt(equil.rzphi_rsquared((psi, tw); hint=hint))
        tc = 2π * (tw + equil.rzphi_offset((psi, tw); hint=hint))
        (equil.ro + r_minor * cos(tc), equil.zo + r_minor * sin(tc))
    end

    R, Z = point(theta_sfl)
    h = 1.0e-5
    Rp, Zp = point(theta_sfl + h)
    Rm, Zm = point(theta_sfl - h)
    dR = (Rp - Rm) / (2h)
    dZ = (Zp - Zm) / (2h)

    nR, nZ = dZ, -dR
    nrm = hypot(nR, nZ)
    nR /= nrm
    nZ /= nrm
    if nR * (R - equil.ro) + nZ * (Z - equil.zo) < 0  # force outward orientation
        nR, nZ = -nR, -nZ
    end
    return (R, Z, nR, nZ)
end

"""
    make_window_pane_standoff(equil; standoff, poloidal_angle, poloidal_length,
                              poloidal_tilt=0.0, ncoil, gap_fraction=0.0, phi0=0.0,
                              name="window_pane") -> CoilSet

Build a window-pane array placed by physical standoff instead of explicit corners. The frame
center sits `standoff` [m] along the outward surface normal at cylindrical `poloidal_angle`
(degrees); its poloidal legs span `poloidal_length` [m] along the surface tangent, rotated by
`poloidal_tilt` (degrees, 0 = tangential). The control surface is the boundary
`equil.rzphi_xs[end]`. Toroidal layout (`ncoil`, `gap_fraction`, `phi0`) and the leg/arc
construction are delegated to [`make_window_pane`](@ref).

The poloidal legs wind in the same sense as `make_window_pane`'s `c1 -> c2` ordering (lower-Z to
higher-Z at the outboard midplane), so positive current produces positive `b_n_x` near the coil,
matching the corner-built window pane.
"""
function make_window_pane_standoff(equil; standoff::Real, poloidal_angle::Real,
    poloidal_length::Real, poloidal_tilt::Real=0.0, ncoil::Int,
    gap_fraction::Real=0.0, phi0::Real=0.0, name::String="window_pane")
    poloidal_length > 0 || error("make_window_pane_standoff: poloidal_length must be > 0 (got $poloidal_length)")

    R, Z, nR, nZ = surface_point_and_normal(equil, deg2rad(poloidal_angle))

    # Frame center: standoff along the outward normal.
    Rc = R + standoff * nR
    Zc = Z + standoff * nZ

    # Tangent: +90° rotation of the outward normal, so the poloidal leg runs lower-Z -> higher-Z
    # at the outboard midplane, matching make_window_pane's c1->c2 winding (positive current ->
    # positive b_n_x near the coil). Rotated by poloidal_tilt in the R-Z plane.
    tR, tZ = -nZ, nR
    c, s = cos(deg2rad(poloidal_tilt)), sin(deg2rad(poloidal_tilt))
    uR = c * tR - s * tZ
    uZ = s * tR + c * tZ

    half = poloidal_length / 2

    # Guard against a coil that pokes through the plasma boundary (the standoff is too small for
    # the tilt/length, e.g. a 90° tilt with standoff < half the leg length). The surface point's
    # tangent plane supports the (convex) boundary, so the nearest leg corner's clearance from the
    # boundary along the normal is standoff - half*|sin(tilt)| — the tilt tips the legs toward the
    # surface, while tangential displacement leaves the normal clearance unchanged.
    clearance = standoff - half * abs(sin(deg2rad(poloidal_tilt)))
    clearance > 0 || error(
        "make_window_pane_standoff: coil intersects the plasma boundary " *
        "(standoff=$standoff, poloidal_length=$poloidal_length, poloidal_tilt=$poloidal_tilt); " *
        "an external coil must lie outside the control surface — increase standoff or reduce " *
        "poloidal_length/poloidal_tilt"
    )

    c1 = [Rc - half * uR, Zc - half * uZ]
    c2 = [Rc + half * uR, Zc + half * uZ]

    return make_window_pane(; ncoil=ncoil, rz_corners=[c1, c2],
        gap_fraction=gap_fraction, phi0=phi0, name=name)
end

"""
    make_window_pane(; ncoil, rz_corners, gap_fraction=0.0, phi0=0.0,
                       np_pol=20, np_tor=40, name="window_pane") -> CoilSet

Build an array of `ncoil` rectangular "picture-frame" (window-pane) saddle coils, evenly
spaced toroidally. The two `[R, Z]` points in `rz_corners` are the endpoints of the poloidal
legs; the legs run along the straight segment between them at the two toroidal edges of the
coil, joined by toroidal arcs at each corner. For a vertical frame give the two corners the
same `R` (legs run in Z); for a horizontal frame give them the same `Z`. Adjacent coils are
separated by a toroidal gap of `gap_fraction` times their angular spacing; the first coil is
centered at toroidal angle `phi0` [rad].

Each coil is a closed loop with `np_pol` points per poloidal leg and `np_tor` per toroidal arc.
Currents are zeroed; `load_coil_sets` fills them.
"""
function make_window_pane(; ncoil::Int, rz_corners::AbstractVector,
    gap_fraction::Real=0.0, phi0::Real=0.0,
    np_pol::Int=20, np_tor::Int=40, name::String="window_pane")
    ncoil >= 1 || error("make_window_pane: ncoil must be >= 1 (got $ncoil)")
    length(rz_corners) == 2 || error("make_window_pane: rz_corners must hold exactly two [R, Z] corners")
    0.0 <= gap_fraction < 1.0 || error("make_window_pane: gap_fraction must be in [0, 1) (got $gap_fraction)")
    np_pol >= 2 && np_tor >= 2 || error("make_window_pane: np_pol and np_tor must be >= 2")
    c1, c2 = rz_corners[1], rz_corners[2]
    length(c1) == 2 && length(c2) == 2 || error("make_window_pane: each rz_corner must be [R, Z]")
    R1, Z1 = Float64(c1[1]), Float64(c1[2])
    R2, Z2 = Float64(c2[1]), Float64(c2[2])

    # Closed loop: leg(np_pol) + arc(np_tor-1) + leg(np_pol-1) + arc(np_tor-1), last == first.
    nsec = 2 * np_pol + 2 * np_tor - 3
    x = zeros(ncoil, 1, nsec)
    y = zeros(ncoil, 1, nsec)
    z = zeros(ncoil, 1, nsec)

    dphi_coil = 2π / ncoil
    half_span = (dphi_coil / 2) * (1 - gap_fraction)
    lin(lo, hi, n) = range(lo; stop=hi, length=n)

    Rs = Float64[]
    phis = Float64[]
    Zs = Float64[]
    for j in 0:(ncoil-1)
        empty!(Rs)
        empty!(phis)
        empty!(Zs)
        phi_c = phi0 + j * dphi_coil
        phi_a = phi_c - half_span
        phi_b = phi_c + half_span

        # Poloidal leg corner1 -> corner2 at phi_a (segment in the R-Z plane)
        for s in lin(0.0, 1.0, np_pol)
            push!(Rs, R1 + s * (R2 - R1))
            push!(phis, phi_a)
            push!(Zs, Z1 + s * (Z2 - Z1))
        end
        # Toroidal arc at corner2: phi_a -> phi_b (drop shared junction)
        for p in lin(phi_a, phi_b, np_tor)[2:end]
            push!(Rs, R2)
            push!(phis, p)
            push!(Zs, Z2)
        end
        # Poloidal leg corner2 -> corner1 at phi_b
        for s in lin(0.0, 1.0, np_pol)[2:end]
            push!(Rs, R2 + s * (R1 - R2))
            push!(phis, phi_b)
            push!(Zs, Z2 + s * (Z1 - Z2))
        end
        # Toroidal arc at corner1: phi_b -> phi_a, last point closes onto the first
        for p in lin(phi_b, phi_a, np_tor)[2:end]
            push!(Rs, R1)
            push!(phis, p)
            push!(Zs, Z1)
        end

        for l in 1:nsec
            x[j+1, 1, l] = Rs[l] * cos(phis[l])
            y[j+1, 1, l] = Rs[l] * sin(phis[l])
            z[j+1, 1, l] = Zs[l]
        end
    end
    return CoilSet(name, ncoil, 1, 1.0, nsec, x, y, z, zeros(ncoil))
end

"""
    make_helical(; R0, a, m, n, theta_lo=0.0, theta_hi=360.0, n_coils,
                   gap_fraction=0.0, phi0=0.0, offset=0.0, npts=720,
                   name="helical") -> CoilSet

Build an array of `n_coils` analytic helical coils wound on a circular cross-section torus
of major radius `R0` and minor radius `a` [m], offset radially outward by `offset` [m].
The helical pitch is set by the poloidal/toroidal mode numbers `m`/`n` via `ζ = (m/n)·θ + φ`.

The poloidal extent `theta_lo`/`theta_hi` is in degrees. For a full poloidal turn
(`theta_hi - theta_lo == 360`) each coil is a single helical strand. For a partial poloidal
range the coil is closed into a loop with two helical legs joined by toroidal arcs (matching
OMFIT `form_helical_array.py`). `npts` sets the strand resolution.
Currents are zeroed; `load_coil_sets` fills them.
"""
function make_helical(; R0::Real, a::Real, m::Int, n::Int,
    theta_lo::Real=0.0, theta_hi::Real=360.0, n_coils::Int,
    gap_fraction::Real=0.0, phi0::Real=0.0, offset::Real=0.0,
    npts::Int=720, name::String="helical")
    R0 > 0 || error("make_helical: R0 must be > 0 (got $R0)")
    a > 0 || error("make_helical: a must be > 0 (got $a)")
    n != 0 || error("make_helical: n must be nonzero (sets the helical pitch m/n)")
    n_coils >= 1 || error("make_helical: n_coils must be >= 1 (got $n_coils)")
    0.0 <= gap_fraction < 1.0 || error("make_helical: gap_fraction must be in [0, 1) (got $gap_fraction)")

    # User-facing poloidal extent is in degrees; the internal winding math works in radians.
    theta_lo = deg2rad(theta_lo)
    theta_hi = deg2rad(theta_hi)

    pitch = m / n
    rr = a + offset
    surf_R(θ) = R0 + rr * cos(θ)
    surf_Z(θ) = rr * sin(θ)

    dphi_coil = 2π / n_coils
    coil_width = dphi_coil * (1 - gap_fraction)
    full_turn = isapprox(theta_hi - theta_lo, 2π; atol=1e-9)
    lin(lo, hi, np) = range(lo; stop=hi, length=np)

    Rs = Float64[]
    phis = Float64[]
    Zs = Float64[]
    strands = Vector{NTuple{3,Vector{Float64}}}(undef, n_coils)
    for j in 0:(n_coils-1)
        empty!(Rs)
        empty!(phis)
        empty!(Zs)
        phi_c = phi0 + j * dphi_coil
        if full_turn
            for θ in lin(theta_lo, theta_hi, npts)
                ζ = phi_c + pitch * θ
                push!(Rs, surf_R(θ))
                push!(phis, ζ)
                push!(Zs, surf_Z(θ))
            end
        else
            np_leg = npts ÷ 2
            np_arc = max(2, npts ÷ 8)
            half_w = coil_width / 2
            # Up leg: theta_lo -> theta_hi at toroidal edge -half_w
            for θ in lin(theta_lo, theta_hi, np_leg)
                ζ = phi_c - half_w + pitch * θ
                push!(Rs, surf_R(θ))
                push!(phis, ζ)
                push!(Zs, surf_Z(θ))
            end
            # Top arc at theta_hi, sweep toroidally by +coil_width
            ζ_top = phi_c - half_w + pitch * theta_hi
            for ζ in lin(ζ_top, ζ_top + coil_width, np_arc)[2:end]
                push!(Rs, surf_R(theta_hi))
                push!(phis, ζ)
                push!(Zs, surf_Z(theta_hi))
            end
            # Down leg: theta_hi -> theta_lo at toroidal edge +half_w
            for θ in lin(theta_hi, theta_lo, np_leg)[2:end]
                ζ = phi_c + half_w + pitch * θ
                push!(Rs, surf_R(θ))
                push!(phis, ζ)
                push!(Zs, surf_Z(θ))
            end
            # Bottom arc at theta_lo, sweep back by -coil_width, closing the loop
            ζ_bot = phi_c + half_w + pitch * theta_lo
            for ζ in lin(ζ_bot, ζ_bot - coil_width, np_arc)[2:end]
                push!(Rs, surf_R(theta_lo))
                push!(phis, ζ)
                push!(Zs, surf_Z(theta_lo))
            end
        end
        strands[j+1] = (copy(Rs), copy(phis), copy(Zs))
    end

    nsec = length(strands[1][1])
    x = zeros(n_coils, 1, nsec)
    y = zeros(n_coils, 1, nsec)
    z = zeros(n_coils, 1, nsec)
    for j in 1:n_coils
        Rj, Pj, Zj = strands[j]
        length(Rj) == nsec || error("make_helical: inconsistent strand length across coils")
        for l in 1:nsec
            x[j, 1, l] = Rj[l] * cos(Pj[l])
            y[j, 1, l] = Rj[l] * sin(Pj[l])
            z[j, 1, l] = Zj[l]
        end
    end
    return CoilSet(name, n_coils, 1, 1.0, nsec, x, y, z, zeros(n_coils))
end

# ---------------------------------------------------------------------------
# HDF5 coil geometry I/O (group-based, multi-set, content-tolerant)
#
# A coil set is stored as one subgroup per set under a parent group, named by the
# set's `name`. This mirrors the open-group idiom of `save_forcing_to_h5` so the
# data can live inside `gpec.h5` (e.g. `Input/RawInputs/Coils/`) next to unrelated
# content that the reader silently ignores.
#
#   <group>/<set_name>/x         Float64[ncoil, s, nsec]  (shape gives ncoil, s, nsec)
#   <group>/<set_name>/y         Float64[ncoil, s, nsec]
#   <group>/<set_name>/z         Float64[ncoil, s, nsec]
#   <group>/<set_name>/currents  Float64[ncoil]
#   <group>/<set_name>  attr nw  Float64
# ---------------------------------------------------------------------------

"""
    save_coils_to_h5(coil_sets::Vector{CoilSet}, group)

Write each coil set as a subgroup (named by `CoilSet.name`) under an already-open
HDF5 `group`. Inverse of [`load_coils_from_h5_group!`](@ref). Used by the gpec.h5
snapshot writer so a coil run can be replayed from the output file alone.
"""
function save_coils_to_h5(coil_sets::Vector{CoilSet}, group)
    for cs in coil_sets
        g = create_group(group, cs.name)
        g["x"] = cs.x
        g["y"] = cs.y
        g["z"] = cs.z
        g["currents"] = cs.currents
        HDF5.attrs(g)["nw"] = cs.nw
    end
    return nothing
end

"""
    load_coils_from_h5_group!(coil_sets::Vector{CoilSet}, group) -> Vector{CoilSet}

Populate `coil_sets` from an already-open HDF5 `group`, reading every subgroup that
holds `x`/`y`/`z` datasets as a [`CoilSet`](@ref). Any other datasets or subgroups
under `group` are silently ignored, so the group may live inside a larger file
(e.g. `gpec.h5`). Mirror of [`save_coils_to_h5`](@ref).
"""
function load_coils_from_h5_group!(coil_sets::Vector{CoilSet}, group)
    empty!(coil_sets)
    for key in keys(group)
        sub = group[key]
        sub isa HDF5.Group || continue
        (haskey(sub, "x") && haskey(sub, "y") && haskey(sub, "z")) || continue
        x = Float64.(read(sub, "x"))
        y = Float64.(read(sub, "y"))
        z = Float64.(read(sub, "z"))
        size(x) == size(y) == size(z) || error("x/y/z shape mismatch in coil set '$key'")
        ndims(x) == 3 || error("coil set '$key' x/y/z must be 3D [ncoil, s, nsec]")
        ncoil, s, nsec = size(x)
        nw = haskey(HDF5.attrs(sub), "nw") ? Float64(HDF5.attrs(sub)["nw"]) : 1.0
        currents = haskey(sub, "currents") ? Float64.(read(sub, "currents")) : zeros(ncoil)
        push!(coil_sets, CoilSet(String(key), ncoil, s, nw, nsec, x, y, z, currents))
    end
    return coil_sets
end

"""
    read_coil_h5(filepath; group="coils", set=nothing) -> Vector{CoilSet}

Read coil sets from an HDF5 file. Coil data is expected under the `group` path;
all coil-set subgroups there are returned (or, if `set` names one, just that set).
Unrelated content elsewhere in the file is ignored. See [`save_coils_to_h5`](@ref)
for the layout.
"""
function read_coil_h5(filepath::String; group::String="coils", set::Union{Nothing,String}=nothing)
    isfile(filepath) || error("Coil geometry file not found: $filepath")
    return h5open(filepath, "r") do file
        haskey(file, group) || error("HDF5 file $filepath has no '$group' group")
        coil_sets = CoilSet[]
        load_coils_from_h5_group!(coil_sets, file[group])
        isempty(coil_sets) && error("No coil sets found under '$group' in $filepath")
        if set !== nothing
            idx = findfirst(cs -> cs.name == set, coil_sets)
            isnothing(idx) && error("Coil set '$set' not found under '$group' in $filepath")
            return CoilSet[coil_sets[idx]]
        end
        return coil_sets
    end
end

"""
    write_coil_h5(filepath, coil_sets; group="coils")

Write `coil_sets` to a new HDF5 file under the `group` path, one subgroup per set.
Convenience file-level wrapper around [`save_coils_to_h5`](@ref) for export and the
`.dat`→`.h5` converter.
"""
function write_coil_h5(filepath::String, coil_sets::Vector{CoilSet}; group::String="coils")
    h5open(filepath, "w") do file
        save_coils_to_h5(coil_sets, create_group(file, group))
    end
    return filepath
end

"""
    write_coil_dat(filepath, cs::CoilSet)

Write a single coil set to an ASCII `.dat` file in the legacy Fortran GPEC format
(`ncoil s nsec nw` header, then `ncoil×s×nsec` rows of `X Y Z`).
"""
function write_coil_dat(filepath::String, cs::CoilSet)
    open(filepath, "w") do io
        println(io, "$(cs.ncoil) $(cs.s) $(cs.nsec) $(cs.nw)")
        for j in 1:cs.ncoil, k in 1:cs.s, l in 1:cs.nsec
            println(io, "  $(cs.x[j,k,l])  $(cs.y[j,k,l])  $(cs.z[j,k,l])")
        end
    end
    return filepath
end

"""
    convert_coil_dat_to_h5(dat_path, h5_path; group="coils") -> String

Convert a legacy ASCII `.dat` coil file to the modern HDF5 format (single set).
"""
function convert_coil_dat_to_h5(dat_path::String, h5_path::String; group::String="coils")
    cs = read_coil_dat(dat_path)
    return write_coil_h5(h5_path, CoilSet[cs]; group=group)
end

"""
    convert_coil_h5_to_dat(h5_path, dat_path; group="coils", set=nothing) -> String

Convert one coil set from an HDF5 file to a legacy ASCII `.dat` file. If the file holds
multiple sets, `set` selects which one by name (required when more than one is present).
"""
function convert_coil_h5_to_dat(h5_path::String, dat_path::String;
    group::String="coils", set::Union{Nothing,String}=nothing)
    coil_sets = read_coil_h5(h5_path; group=group, set=set)
    length(coil_sets) == 1 ||
        error("convert_coil_h5_to_dat: $h5_path holds $(length(coil_sets)) sets; pass `set` to pick one")
    return write_coil_dat(dat_path, coil_sets[1])
end

"""
    _arc_length_center(x, y, z) -> (x0, y0, z0)

Compute the arc-length-weighted centroid of a strand (1D array of points).
Matches Fortran center-of-mass convention using midpoints weighted by segment length.
"""
function _arc_length_center(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    nsec = length(x)
    nsec > 1 || return (x[1], y[1], z[1])

    total_length = 0.0
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    for l in 1:(nsec-1)
        xm = (x[l] + x[l+1]) / 2
        ym = (y[l] + y[l+1]) / 2
        zm = (z[l] + z[l+1]) / 2
        dl = sqrt((x[l+1] - x[l])^2 + (y[l+1] - y[l])^2 + (z[l+1] - z[l])^2)
        x0 += xm * dl
        y0 += ym * dl
        z0 += zm * dl
        total_length += dl
    end
    total_length > 0 || return (x[1], y[1], z[1])
    return (x0 / total_length, y0 / total_length, z0 / total_length)
end

"""
    apply_transforms(cs::CoilSet, cfg::CoilSetConfig; n_tilt::Int=1) -> CoilSet

Apply per-conductor shifts and tilts to a coil set, returning a modified copy.

Replicates the Fortran `coil_read` shift/tilt logic (coil.F lines 240–340):

  - Tilts are rotations around the arc-length-weighted center of mass (unless `xnom/ynom/znom` specified)
  - `n_tilt` controls the toroidal periodicity of tilt/shift modulation
  - n_tilt = 0: rigid shift only (no tilts applied)
  - n_tilt ≥ 1: n-fold modulated perturbations
"""
function apply_transforms(cs::CoilSet, cfg::CoilSetConfig; n_tilt::Int=1)
    ncoil = cs.ncoil
    s = cs.s
    nsec = cs.nsec

    # Pad config vectors to ncoil length with zeros
    _pad(v, n) = length(v) >= n ? Float64.(v[1:n]) : vcat(Float64.(v), zeros(n - length(v)))

    shiftx = _pad(cfg.shiftx, ncoil)
    shifty = _pad(cfg.shifty, ncoil)
    shiftz = _pad(cfg.shiftz, ncoil)
    tiltx_cfg = _pad(cfg.tiltx, ncoil)
    tilty_cfg = _pad(cfg.tilty, ncoil)
    tiltz_cfg = _pad(cfg.tiltz, ncoil)

    xnom_cfg = _pad(isempty(cfg.xnom) ? fill(_NOM_UNSET_SENTINEL, ncoil) : cfg.xnom, ncoil)
    ynom_cfg = _pad(isempty(cfg.ynom) ? fill(_NOM_UNSET_SENTINEL, ncoil) : cfg.ynom, ncoil)
    znom_cfg = _pad(isempty(cfg.znom) ? fill(_NOM_UNSET_SENTINEL, ncoil) : cfg.znom, ncoil)

    # Check if n_tilt = 0 suppresses tilts (Fortran: "no n=0 component")
    apply_tilt = n_tilt != 0

    new_x = copy(cs.x)
    new_y = copy(cs.y)
    new_z = copy(cs.z)

    for j in 1:ncoil
        sx = shiftx[j]
        sy = shifty[j]
        sz = shiftz[j]
        tx_deg = tiltx_cfg[j]
        ty_deg = tilty_cfg[j]
        tz_deg = tiltz_cfg[j]

        has_shift = abs(sx) + abs(sy) + abs(sz) > 0
        has_tilt = apply_tilt && (abs(tx_deg) + abs(ty_deg) + abs(tz_deg) > 0)
        (has_shift || has_tilt) || continue

        for k in 1:s
            # Compute rotation center for this strand
            x0, y0, z0 = begin
                cx, cy, cz = _arc_length_center(
                    view(cs.x, j, k, :), view(cs.y, j, k, :), view(cs.z, j, k, :)
                )
                # Use user-specified center if provided (|nom| < _NOM_THRESHOLD, matching Fortran)
                x0 = abs(xnom_cfg[j]) < _NOM_THRESHOLD ? xnom_cfg[j] : cx
                y0 = abs(ynom_cfg[j]) < _NOM_THRESHOLD ? ynom_cfg[j] : cy
                z0 = abs(znom_cfg[j]) < _NOM_THRESHOLD ? znom_cfg[j] : cz
                (x0, y0, z0)
            end

            # Compute nominal radius for tilt_in_meters conversion
            r_nom = if cfg.tilt_in_meters
                total_len = 0.0
                weighted_r = 0.0
                for l in 1:(nsec-1)
                    xm = (cs.x[j, k, l] + cs.x[j, k, l+1]) / 2
                    ym = (cs.y[j, k, l] + cs.y[j, k, l+1]) / 2
                    dl = sqrt((cs.x[j, k, l+1] - cs.x[j, k, l])^2 + (cs.y[j, k, l+1] - cs.y[j, k, l])^2 +
                              (cs.z[j, k, l+1] - cs.z[j, k, l])^2)
                    weighted_r += sqrt(xm^2 + ym^2) * dl
                    total_len += dl
                end
                total_len > 0 ? weighted_r / total_len : 1.0
            else
                1.0  # not used
            end

            # Convert tilt to radians
            dtor = π / 180.0
            tiltx, tilty, tiltz = if cfg.tilt_in_meters
                asin(tx_deg / r_nom), asin(ty_deg / r_nom), asin(tz_deg / r_nom)
            else
                tx_deg * dtor, ty_deg * dtor, tz_deg * dtor
            end

            for l in 1:nsec
                # Work in center frame. Tilts are superposed as independent small-angle
                # perturbations (linear addition), matching Fortran coil.F convention.
                xc = cs.x[j, k, l] - x0
                yc = cs.y[j, k, l] - y0
                zc = cs.z[j, k, l] - z0

                # Perturbations start at center offset (so final = original + perturbations)
                dx = x0
                dy = y0
                dz = z0
                phi = atan(yc, xc)

                # Z-tilt: rotate in xy-plane (toroidal clocking)
                if tiltz != 0
                    r = sqrt(xc^2 + yc^2)
                    ang = atan(yc, xc) + tiltz * cos((n_tilt - 1) * phi)
                    dx += r * cos(ang) - xc
                    dy += r * sin(ang) - yc
                end

                # Y-tilt: rotate in xz-plane
                if tilty != 0
                    r = sqrt(xc^2 + zc^2)
                    ang = atan(zc, xc) + tilty * cos((n_tilt - 1) * phi)
                    dx += r * cos(ang) - xc
                    dz += r * sin(ang) - zc
                end

                # X-tilt: rotate in yz-plane (n=1 is constant, n>1 is sin-modulated)
                if tiltx != 0
                    r = sqrt(zc^2 + yc^2)
                    ang = if n_tilt == 1
                        atan(zc, yc) + tiltx
                    else
                        atan(zc, yc) + tiltx * sin((n_tilt - 1) * phi)
                    end
                    dy += r * cos(ang) - yc
                    dz += r * sin(ang) - zc
                end

                # Shift: n=0 is rigid; n>0 is radial petal pattern
                if n_tilt == 0
                    dx += sx
                    dy += sy
                else
                    dr = sx * cos(n_tilt * phi) + sy * sin(n_tilt * phi)
                    dx += dr * cos(phi)
                    dy += dr * sin(phi)
                end
                dz += sz

                new_x[j, k, l] = xc + dx
                new_y[j, k, l] = yc + dy
                new_z[j, k, l] = zc + dz
            end
        end
    end

    return CoilSet(cs.name, ncoil, s, cs.nw, nsec, new_x, new_y, new_z, cs.currents)
end

"""
    _resolve_coil_path(cfg::CoilConfig, set_cfg::CoilSetConfig) -> String

Resolve the path to a coil geometry file. Prefers `set_cfg.h5_file`, then
`set_cfg.dat_file`, otherwise constructs `{dat_dir}/{machine}_{name}.dat`.
"""
function _resolve_coil_path(cfg::CoilConfig, set_cfg::CoilSetConfig)
    isempty(set_cfg.h5_file) || return set_cfg.h5_file
    isempty(set_cfg.dat_file) || return set_cfg.dat_file
    isempty(set_cfg.name) &&
        error("CoilSetConfig must specify a `source`, or a `name`/`dat_file`/`h5_file`")
    prefix = isempty(cfg.machine) ? "" : "$(cfg.machine)_"
    return joinpath(cfg.dat_dir, "$(prefix)$(set_cfg.name).dat")
end

"""
    _build_raw_coil_set(cfg::CoilConfig, sc::CoilSetConfig) -> CoilSet

Produce the un-transformed coil set for a config block, dispatching on `sc.source`:
analytic generators (`pf_hoop`/`window_pane`/`helical`) or a geometry file
(`.h5` → [`read_coil_h5`](@ref), else `.dat` → [`read_coil_dat`](@ref)). Currents and
shifts/tilts are applied by the caller, so all sources share the same downstream path.
"""
function _build_raw_coil_set(cfg::CoilConfig, sc::CoilSetConfig; equil=nothing)
    if sc.source == "pf_hoop"
        kw = sc.nsec_gen > 0 ? (; nsec=sc.nsec_gen) : (;)
        return make_pf_hoop(; radius=sc.radius, height=sc.height,
            name=isempty(sc.name) ? "pf_hoop" : sc.name, kw...)
    elseif sc.source == "window_pane"
        has_corners = !isempty(sc.rz_corners)
        has_standoff = !isnan(sc.standoff) && !isnan(sc.poloidal_angle)
        if has_corners && has_standoff
            error("window_pane: provide EITHER rz_corners (corner mode) OR " *
                  "standoff+poloidal_angle (standoff mode), not both")
        elseif has_standoff
            equil === nothing &&
                error("window_pane standoff mode (standoff/poloidal_angle) requires an equilibrium; " *
                      "pass `equil` to load_coil_sets, or use rz_corners for corner mode")
            return make_window_pane_standoff(equil; standoff=sc.standoff,
                poloidal_angle=sc.poloidal_angle, poloidal_length=sc.poloidal_length,
                poloidal_tilt=sc.poloidal_tilt, ncoil=sc.ncoil_gen,
                gap_fraction=sc.gap_fraction, phi0=sc.phi0,
                name=isempty(sc.name) ? "window_pane" : sc.name)
        elseif has_corners
            return make_window_pane(; ncoil=sc.ncoil_gen, rz_corners=sc.rz_corners,
                gap_fraction=sc.gap_fraction, phi0=sc.phi0,
                name=isempty(sc.name) ? "window_pane" : sc.name)
        else
            error("window_pane: supply rz_corners (corner mode) or " *
                  "standoff+poloidal_angle (standoff mode)")
        end
    elseif sc.source == "helical"
        kw = sc.nsec_gen > 0 ? (; npts=sc.nsec_gen) : (;)
        return make_helical(; R0=sc.R0, a=sc.a, m=sc.m_hel, n=sc.n_hel,
            theta_lo=sc.theta_lo, theta_hi=sc.theta_hi, n_coils=sc.n_coils,
            gap_fraction=sc.gap_fraction, phi0=sc.phi0, offset=sc.offset,
            name=isempty(sc.name) ? "helical" : sc.name, kw...)
    elseif sc.source == "" || sc.source == "file"
        path = _resolve_coil_path(cfg, sc)
        ext = lowercase(splitext(path)[2])
        if ext in (".h5", ".hdf5")
            sets = read_coil_h5(path; set=isempty(sc.name) ? nothing : sc.name)
            length(sets) == 1 ||
                error("h5 coil file $path holds $(length(sets)) sets; set `name` to pick one")
            return sets[1]
        end
        return read_coil_dat(path)
    else
        error("Unknown coil set source: '$(sc.source)' " *
              "(expected \"\", \"file\", \"pf_hoop\", \"window_pane\", or \"helical\")")
    end
end

"""
    load_coil_sets(cfg::CoilConfig, n_tilt::Int; equil=nothing) -> Vector{CoilSet}

Build all coil sets from config (file or analytic source), apply currents and
shifts/tilts, and return ready-to-use coil data. `n_tilt` is the toroidal mode number
for modulated perturbations (typically the run's n). `equil` is required only when a
`window_pane` set uses standoff placement (it locates the surface point and normal).
"""
function load_coil_sets(cfg::CoilConfig, n_tilt::Int; equil=nothing)
    isempty(cfg.coil_sets) && error("No coil sets specified in [ForcingTerms] configuration")

    coil_sets = CoilSet[]
    for set_cfg in cfg.coil_sets
        cs_raw = _build_raw_coil_set(cfg, set_cfg; equil=equil)

        # Set currents (pad or truncate to ncoil)
        ncoil = cs_raw.ncoil
        currents = if isempty(set_cfg.currents)
            zeros(ncoil)
        else
            len = length(set_cfg.currents)
            len >= ncoil ? Float64.(set_cfg.currents[1:ncoil]) :
            vcat(Float64.(set_cfg.currents), zeros(ncoil - len))
        end
        cs_with_currents = CoilSet(
            cs_raw.name, ncoil, cs_raw.s, cs_raw.nw, cs_raw.nsec,
            cs_raw.x, cs_raw.y, cs_raw.z, currents
        )

        # Resolve n_tilt: -1 means inherit from run
        effective_n = set_cfg.n_tilt == -1 ? n_tilt : set_cfg.n_tilt

        cs = apply_transforms(cs_with_currents, set_cfg; n_tilt=effective_n)
        push!(coil_sets, cs)
    end
    return coil_sets
end

export CoilSet, CoilSetConfig, CoilConfig
export read_coil_dat, apply_transforms, load_coil_sets
export make_pf_hoop, make_window_pane, make_helical
export make_window_pane_standoff, surface_point_and_normal
export read_coil_h5, write_coil_h5, write_coil_dat, save_coils_to_h5, load_coils_from_h5_group!
export convert_coil_dat_to_h5, convert_coil_h5_to_dat
