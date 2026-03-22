"""
    CoilGeometry

Structs and I/O for 3D coil geometries used in Biot-Savart field calculations.

Coil geometry files use the Fortran GPEC ASCII format:
  - Header: `ncoil s nsec nw`
  - Data: `ncoil × s × nsec` rows, each with `X Y Z` in meters (Cartesian, lab frame)
"""

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

## Fields
- `name`: coil set name; resolves to `{dat_dir}/{machine}_{name}.dat`
- `dat_file`: explicit path to .dat file (overrides `machine`+`name` convention)
- `currents`: current [A] per conductor `[ncoil]`; shorter arrays pad with zeros
- `shiftx`, `shifty`, `shiftz`: per-conductor translation [m] `[ncoil]`
- `tiltx`, `tilty`, `tiltz`: per-conductor tilt [degrees] (or [m] if `tilt_in_meters`)
- `xnom`, `ynom`, `znom`: explicit rotation center [m]; defaults to arc-length-weighted center of mass
- `n_tilt`: toroidal mode number for tilt/shift modulation; -1 means inherit run's n
- `tilt_in_meters`: interpret tilt as displacement [m] instead of angle [degrees]
"""
Base.@kwdef struct CoilSetConfig
    name::String = ""
    dat_file::String = ""
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
        machine     = ft_ctrl.machine,
        dat_dir     = dat_dir,
        mtheta_coil = ft_ctrl.mtheta_coil,
        nzeta_coil  = ft_ctrl.nzeta_coil,
        coil_sets   = coil_sets,
    )
end

function _parse_coil_set_config(d::Dict{String,Any})
    fvec(key) = Float64.(get(d, key, Float64[]))
    return CoilSetConfig(;
        name           = get(d, "name", ""),
        dat_file       = get(d, "dat_file", ""),
        currents       = fvec("currents"),
        shiftx         = fvec("shiftx"),
        shifty         = fvec("shifty"),
        shiftz         = fvec("shiftz"),
        tiltx          = fvec("tiltx"),
        tilty          = fvec("tilty"),
        tiltz          = fvec("tiltz"),
        xnom           = fvec("xnom"),
        ynom           = fvec("ynom"),
        znom           = fvec("znom"),
        n_tilt         = get(d, "n_tilt", -1),
        tilt_in_meters = get(d, "tilt_in_meters", false),
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
    s     = parse(Int, header[2])
    nsec  = parse(Int, header[3])
    nw    = parse(Float64, header[4])

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
    _arc_length_center(x, y, z) -> (x0, y0, z0)

Compute the arc-length-weighted centroid of a strand (1D array of points).
Matches Fortran center-of-mass convention using midpoints weighted by segment length.
"""
function _arc_length_center(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    nsec = length(x)
    nsec > 1 || return (x[1], y[1], z[1])

    total_length = 0.0
    x0 = 0.0; y0 = 0.0; z0 = 0.0
    for l in 1:(nsec - 1)
        xm = (x[l] + x[l+1]) / 2
        ym = (y[l] + y[l+1]) / 2
        zm = (z[l] + z[l+1]) / 2
        dl = sqrt((x[l+1]-x[l])^2 + (y[l+1]-y[l])^2 + (z[l+1]-z[l])^2)
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
    s     = cs.s
    nsec  = cs.nsec

    # Pad config vectors to ncoil length with zeros
    _pad(v, n) = length(v) >= n ? Float64.(v[1:n]) : vcat(Float64.(v), zeros(n - length(v)))

    shiftx = _pad(cfg.shiftx, ncoil)
    shifty = _pad(cfg.shifty, ncoil)
    shiftz = _pad(cfg.shiftz, ncoil)
    tiltx_cfg = _pad(cfg.tiltx, ncoil)
    tilty_cfg = _pad(cfg.tilty, ncoil)
    tiltz_cfg = _pad(cfg.tiltz, ncoil)

    xnom_cfg = _pad(isempty(cfg.xnom) ? fill(1e10, ncoil) : cfg.xnom, ncoil)
    ynom_cfg = _pad(isempty(cfg.ynom) ? fill(1e10, ncoil) : cfg.ynom, ncoil)
    znom_cfg = _pad(isempty(cfg.znom) ? fill(1e10, ncoil) : cfg.znom, ncoil)

    # Check if n_tilt = 0 suppresses tilts (Fortran: "no n=0 component")
    apply_tilt = n_tilt != 0

    new_x = copy(cs.x)
    new_y = copy(cs.y)
    new_z = copy(cs.z)

    for j in 1:ncoil
        sx = shiftx[j]; sy = shifty[j]; sz = shiftz[j]
        tx_deg = tiltx_cfg[j]; ty_deg = tilty_cfg[j]; tz_deg = tiltz_cfg[j]

        has_shift = abs(sx) + abs(sy) + abs(sz) > 0
        has_tilt  = apply_tilt && (abs(tx_deg) + abs(ty_deg) + abs(tz_deg) > 0)
        (has_shift || has_tilt) || continue

        for k in 1:s
            # Compute rotation center for this strand
            x0, y0, z0 = begin
                cx, cy, cz = _arc_length_center(
                    view(cs.x, j, k, :), view(cs.y, j, k, :), view(cs.z, j, k, :)
                )
                # Use user-specified center if provided (|nom| < 1e3 check, matching Fortran)
                x0 = abs(xnom_cfg[j]) < 1e3 ? xnom_cfg[j] : cx
                y0 = abs(ynom_cfg[j]) < 1e3 ? ynom_cfg[j] : cy
                z0 = abs(znom_cfg[j]) < 1e3 ? znom_cfg[j] : cz
                (x0, y0, z0)
            end

            # Compute nominal radius for tilt_in_meters conversion
            r_nom = if cfg.tilt_in_meters
                total_len = 0.0; weighted_r = 0.0
                for l in 1:(nsec - 1)
                    xm = (cs.x[j,k,l] + cs.x[j,k,l+1]) / 2
                    ym = (cs.y[j,k,l] + cs.y[j,k,l+1]) / 2
                    dl = sqrt((cs.x[j,k,l+1]-cs.x[j,k,l])^2 + (cs.y[j,k,l+1]-cs.y[j,k,l])^2 +
                              (cs.z[j,k,l+1]-cs.z[j,k,l])^2)
                    weighted_r += sqrt(xm^2 + ym^2) * dl
                    total_len  += dl
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
                # Work in center frame
                xc = cs.x[j, k, l] - x0
                yc = cs.y[j, k, l] - y0
                zc = cs.z[j, k, l] - z0

                # Perturbations start at center offset (so final = original + perturbations)
                dx = x0; dy = y0; dz = z0
                phi = atan(yc, xc)

                # Z-tilt: rotate in xy-plane (toroidal clocking)
                if tiltz != 0
                    r   = sqrt(xc^2 + yc^2)
                    ang = atan(yc, xc) + tiltz * cos((n_tilt - 1) * phi)
                    dx += r * cos(ang) - xc
                    dy += r * sin(ang) - yc
                end

                # Y-tilt: rotate in xz-plane
                if tilty != 0
                    r   = sqrt(xc^2 + zc^2)
                    ang = atan(zc, xc) + tilty * cos((n_tilt - 1) * phi)
                    dx += r * cos(ang) - xc
                    dz += r * sin(ang) - zc
                end

                # X-tilt: rotate in yz-plane (n=1 is constant, n>1 is sin-modulated)
                if tiltx != 0
                    r   = sqrt(zc^2 + yc^2)
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
    _resolve_dat_path(cfg::CoilConfig, set_cfg::CoilSetConfig) -> String

Resolve the path to a coil .dat file. Uses `set_cfg.dat_file` if set,
otherwise constructs `{dat_dir}/{machine}_{name}.dat`.
"""
function _resolve_dat_path(cfg::CoilConfig, set_cfg::CoilSetConfig)
    isempty(set_cfg.dat_file) || return set_cfg.dat_file
    isempty(set_cfg.name) && error("CoilSetConfig must specify either `name` or `dat_file`")
    prefix = isempty(cfg.machine) ? "" : "$(cfg.machine)_"
    return joinpath(cfg.dat_dir, "$(prefix)$(set_cfg.name).dat")
end

"""
    load_coil_sets(cfg::CoilConfig, n_tilt::Int) -> Vector{CoilSet}

Read all coil sets from config, apply shifts/tilts, and return ready-to-use coil data.
`n_tilt` is the toroidal mode number for modulated perturbations (typically the run's n).
"""
function load_coil_sets(cfg::CoilConfig, n_tilt::Int)
    isempty(cfg.coil_sets) && error("No coil sets specified in [ForcingTerms] configuration")

    coil_sets = CoilSet[]
    for set_cfg in cfg.coil_sets
        filepath = _resolve_dat_path(cfg, set_cfg)
        cs_raw = read_coil_dat(filepath)

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
