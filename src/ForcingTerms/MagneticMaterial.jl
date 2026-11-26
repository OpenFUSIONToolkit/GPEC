"""
    MagneticMaterial

A set of ferromagnetic (ferritic) elements, each a closed triangulated polyhedron that is uniformly magnetized. Each
element's volume and centroid are computed from its own surface at construction, which also checks that every
element is closed and outward-oriented.

## Fields

  - `name::String` - Set identifier, e.g. `"iter_inserts"`.
  - `vertices::Matrix{Float64}` - Vertex coordinates `[3, nvertex]` (x, y, z) in metres, in the same machine Cartesian frame as the coil geometry.
  - `triangles::Matrix{Int}` - Vertex indices (1-based) of each triangle `[3, ntriangle]`, ordered counterclockwise seen from outside the element.
  - `element_offsets::Vector{Int}` - Triangle offsets `[nelement + 1]`, from 0: element `e` is triangles `element_offsets[e]+1:element_offsets[e+1]`.
  - `saturation_field::Float64` - Saturation magnetization μ₀M_s of the material, in tesla.
  - `fill_fraction::Float64` - Fraction of each element's volume that is magnetic material, in (0, 1].
  - `volume::Vector{Float64}` - Volume of each element `[nelement]`, in m³.
  - `centroid::Matrix{Float64}` - Centroid of each element `[3, nelement]`, in metres.
"""
struct MagneticMaterial
    name::String
    vertices::Matrix{Float64}
    triangles::Matrix{Int}
    element_offsets::Vector{Int}
    saturation_field::Float64
    fill_fraction::Float64
    volume::Vector{Float64}
    centroid::Matrix{Float64}
    function MagneticMaterial(name, vertices, triangles, element_offsets, saturation_field, fill_fraction)
        label = "magnetic material \"$name\""
        size(vertices, 1) == 3 || throw(ArgumentError("$label: vertices must be 3 × nvertex, got $(size(vertices))"))
        size(triangles, 1) == 3 || throw(ArgumentError("$label: triangles must be 3 × ntriangle, got $(size(triangles))"))
        all(isfinite, vertices) || throw(ArgumentError("$label: vertices must be finite"))
        all(i -> 1 <= i <= size(vertices, 2), triangles) || throw(ArgumentError("$label: a triangle indexes a vertex outside 1:$(size(vertices, 2))"))
        length(element_offsets) >= 2 && first(element_offsets) == 0 && last(element_offsets) == size(triangles, 2) && issorted(element_offsets) ||
            throw(ArgumentError("$label: element_offsets must rise from 0 to the triangle count $(size(triangles, 2))"))
        isfinite(saturation_field) && saturation_field > 0 || throw(ArgumentError("$label: saturation_field must be positive, μ₀M_s in tesla"))
        0 < fill_fraction <= 1 || throw(ArgumentError("$label: fill_fraction must be in (0, 1], got $fill_fraction"))
        saturation_field > MAX_SATURATION_FIELD &&
            @warn "$label: saturation_field = $saturation_field T exceeds any steel's μ₀M_s; is it M_s in A/m rather than μ₀M_s in tesla?"
        nelement = length(element_offsets) - 1
        volume, centroid = zeros(nelement), zeros(3, nelement)
        for e in 1:nelement
            tris = (element_offsets[e]+1):element_offsets[e+1]
            isempty(tris) && throw(ArgumentError("$label: element $e has no triangles"))
            _check_closed(triangles, tris) || throw(ArgumentError("$label: element $e is not a closed, consistently oriented surface"))
            # Divergence theorem: each triangle and the origin span a signed tetrahedron.
            for t in tris
                a, b, c = (view(vertices, :, triangles[k, t]) for k in 1:3)
                v = dot(a, cross(b, c)) / 6
                volume[e] += v
                centroid[:, e] .+= v .* (a .+ b .+ c) ./ 4
            end
            volume[e] > 0 || throw(ArgumentError("$label: element $e has volume $(volume[e]) m³; its triangles must face outward"))
            centroid[:, e] ./= volume[e]
        end
        return new(String(name), vertices, triangles, element_offsets, saturation_field, fill_fraction, volume, centroid)
    end
end

# μ₀M_s of iron is 2.15 T; anything above this is not a ferromagnetic steel.
const MAX_SATURATION_FIELD = 2.5

# A closed, consistently oriented surface uses every directed edge exactly once and its reverse exactly once.
function _check_closed(triangles, tris)
    edges = Dict{Tuple{Int,Int},Int}()
    for t in tris, k in 1:3
        edge = (triangles[k, t], triangles[mod1(k + 1, 3), t])
        edges[edge] = get(edges, edge, 0) + 1
    end
    return all(n == 1 && get(edges, reverse(edge), 0) == 1 for (edge, n) in edges)
end

"""
    MagneticMaterialConfig

One `[[ForcingTerms.magnetic_material]]` TOML block.

## Fields

  - `name::String` - Set to read from the geometry file, and its identifier.
  - `h5_file::String` - Geometry file holding `magnetic_material/<name>/`; see [`read_magnetic_material_h5`](@ref).
  - `saturation_field::Float64` - Saturation magnetization μ₀M_s of the material, in tesla.
  - `fill_fraction::Float64` - Fraction of each element's volume that is magnetic material (default 1).
"""
Base.@kwdef struct MagneticMaterialConfig
    name::String = ""
    h5_file::String = ""
    saturation_field::Float64 = NaN
    fill_fraction::Float64 = 1.0
end

function _parse_magnetic_material_config(d::AbstractDict)
    known = Set(String.(fieldnames(MagneticMaterialConfig)))
    unknown = sort([String(k) for k in keys(d) if !(String(k) in known)])
    isempty(unknown) || throw(ArgumentError("unknown keys $(unknown) in [[ForcingTerms.magnetic_material]]. Known: $(sort(collect(known)))."))
    return MagneticMaterialConfig(; (Symbol(k) => v for (k, v) in d)...)
end

"""
    load_magnetic_material(cfg::MagneticMaterialConfig) -> MagneticMaterial

Read the set `cfg.name` from `cfg.h5_file` and give it the material properties of `cfg`.
"""
function load_magnetic_material(cfg::MagneticMaterialConfig)
    isempty(cfg.h5_file) && throw(ArgumentError("[[ForcingTerms.magnetic_material]] \"$(cfg.name)\" needs an h5_file"))
    isnan(cfg.saturation_field) && throw(ArgumentError("[[ForcingTerms.magnetic_material]] \"$(cfg.name)\" needs a saturation_field (μ₀M_s in tesla)"))
    g = only(read_magnetic_material_h5(cfg.h5_file; set=cfg.name))
    return MagneticMaterial(g.name, g.vertices, g.triangles, g.element_offsets, cfg.saturation_field, cfg.fill_fraction)
end

# HDF5 layout of a magnetic material set (Julia order; h5py sees each array transposed):
#   <group>/<set_name>/vertices         Float64[3, nvertex], metres
#   <group>/<set_name>/triangles        Int[3, ntriangle], 0-based vertex indices, counterclockwise seen from outside
#   <group>/<set_name>/element_offsets  Int[nelement + 1], 0-based triangle offsets
#   <group>/<set_name>  attrs saturation_field (T), fill_fraction; optional in geometry files

"""
    save_magnetic_material_to_h5(materials::Vector{MagneticMaterial}, group)

Write each set as a subgroup (named by its `name`) under an already-open HDF5 `group`, with 0-based indices. Inverse
of [`load_magnetic_material_from_h5_group!`](@ref).
"""
function save_magnetic_material_to_h5(materials::Vector{MagneticMaterial}, group)
    for mat in materials
        g = create_group(group, mat.name)
        g["vertices"] = mat.vertices
        g["triangles"] = mat.triangles .- 1
        g["element_offsets"] = mat.element_offsets
        HDF5.attrs(g)["saturation_field"] = mat.saturation_field
        HDF5.attrs(g)["fill_fraction"] = mat.fill_fraction
    end
    return nothing
end

"""
    load_magnetic_material_from_h5_group!(materials, group) -> Vector{MagneticMaterial}

Populate `materials` from an already-open HDF5 `group`, one set per subgroup holding `vertices`, `triangles` and
`element_offsets`. A set without `saturation_field`/`fill_fraction` attributes (a geometry-only file) gets
μ₀M_s = 1 T and fill fraction 1, to be replaced by its deck block. Mirror of [`save_magnetic_material_to_h5`](@ref).
"""
function load_magnetic_material_from_h5_group!(materials::Vector{MagneticMaterial}, group)
    empty!(materials)
    for key in keys(group)
        sub = group[key]
        (sub isa HDF5.Group && all(haskey(sub, k) for k in ("vertices", "triangles", "element_offsets"))) || continue
        attrs = HDF5.attrs(sub)
        saturation_field = haskey(attrs, "saturation_field") ? Float64(attrs["saturation_field"]) : 1.0
        fill_fraction = haskey(attrs, "fill_fraction") ? Float64(attrs["fill_fraction"]) : 1.0
        push!(
            materials,
            MagneticMaterial(String(key), Float64.(read(sub, "vertices")), Int.(read(sub, "triangles")) .+ 1,
                Int.(read(sub, "element_offsets")), saturation_field, fill_fraction)
        )
    end
    return materials
end

"""
    read_magnetic_material_h5(filepath; group="magnetic_material", set=nothing) -> Vector{MagneticMaterial}

Read the magnetic material sets under `group` of `filepath`, or only the one named `set`. Each set is a subgroup
holding `vertices` (`3 × nvertex` in Julia order, `nvertex × 3` as h5py writes it, in metres), `triangles` (0-based
vertex indices, counterclockwise seen from outside) and `element_offsets` (0-based triangle offsets, `nelement + 1`),
so one machine file can also hold `coils/`.
"""
function read_magnetic_material_h5(filepath::AbstractString; group::String="magnetic_material", set::Union{Nothing,String}=nothing)
    isfile(filepath) || throw(ArgumentError("magnetic material file not found: $filepath"))
    return h5open(filepath, "r") do file
        haskey(file, group) || throw(ArgumentError("$filepath has no '$group' group"))
        materials = load_magnetic_material_from_h5_group!(MagneticMaterial[], file[group])
        isempty(materials) && throw(ArgumentError("no magnetic material sets under '$group' in $filepath"))
        set === nothing && return materials
        idx = findfirst(m -> m.name == set, materials)
        idx === nothing && throw(ArgumentError("magnetic material set '$set' not found under '$group' in $filepath; it has $(join([m.name for m in materials], ", "))"))
        return MagneticMaterial[materials[idx]]
    end
end

"""
    write_magnetic_material_h5(filepath, materials; group="magnetic_material")

Write `materials` to a new HDF5 file under `group`, one subgroup per set.
"""
function write_magnetic_material_h5(filepath::AbstractString, materials::Vector{MagneticMaterial}; group::String="magnetic_material")
    h5open(file -> save_magnetic_material_to_h5(materials, create_group(file, group)), filepath, "w")
    return filepath
end

"""
    compute_magnetic_material_field!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, material, magnetization)

Exact field of `material` with each element uniformly magnetized, `magnetization` `[3, nelement]` (Cartesian, A/m),
at observation points given in cylindrical coordinates (m, rad, m), as cylindrical components in tesla; a point
inside an element is an error. Uniform M is a surface charge σ = M·n̂ on each face, and a flat triangle with constant charge has a
closed-form field: ∫ (r − r′)/|r − r′|³ dA′ = n̂ Ω + Σ_edges m̂ₑ ln[(R₁ + R₂ + ℓ)/(R₁ + R₂ − ℓ)], with Ω the signed
solid angle the triangle subtends (Van Oosterom & Strackee, IEEE Trans. Biomed. Eng. 30, 125 (1983)) and m̂ₑ its
outward in-plane edge normals. Then B = μ₀/4π · Σ_faces σ (n̂ Ω + Σ_edges m̂ₑ Lₑ). Threaded over observation points;
the outputs are overwritten.
"""
function compute_magnetic_material_field!(
    B_R::AbstractVector{Float64},
    B_phi::AbstractVector{Float64},
    B_Z::AbstractVector{Float64},
    obs_R::AbstractVector{Float64},
    obs_phi::AbstractVector{Float64},
    obs_Z::AbstractVector{Float64},
    material::MagneticMaterial,
    magnetization::AbstractMatrix{<:Real}
)
    nobs = length(obs_R)
    @assert length(obs_phi) == nobs && length(obs_Z) == nobs
    @assert length(B_R) == nobs && length(B_phi) == nobs && length(B_Z) == nobs
    nelement = length(material.element_offsets) - 1
    size(magnetization) == (3, nelement) || throw(DimensionMismatch("magnetization must be 3 × $nelement, got $(size(magnetization))"))
    faces = _charged_faces(material, magnetization)
    lo, hi = _element_boxes(material)
    Threads.@threads for i in 1:nobs
        sp, cp = sincos(obs_phi[i])
        r = SVector(obs_R[i] * cp, obs_R[i] * sp, obs_Z[i])
        _check_outside(material, lo, hi, r)
        H = zero(SVector{3,Float64})
        for f in faces
            H += f.σ * _triangle_kernel(r, f)
        end
        B_R[i] = MU0_OVER_4PI * (H[1] * cp + H[2] * sp)
        B_phi[i] = MU0_OVER_4PI * (-H[1] * sp + H[2] * cp)
        B_Z[i] = MU0_OVER_4PI * H[3]
    end
    return nothing
end

# Bounding box of each element, [3, nelement] lower and upper corners.
function _element_boxes(material::MagneticMaterial)
    nelement = length(material.element_offsets) - 1
    lo, hi = fill(Inf, 3, nelement), fill(-Inf, 3, nelement)
    for e in 1:nelement, t in (material.element_offsets[e]+1):material.element_offsets[e+1], k in 1:3
        v = view(material.vertices, :, material.triangles[k, t])
        lo[:, e] .= min.(view(lo, :, e), v)
        hi[:, e] .= max.(view(hi, :, e), v)
    end
    return lo, hi
end

# The field formula holds only outside the material: the solid angles of an element's faces sum to ±4π inside it, 0 outside.
function _check_outside(material::MagneticMaterial, lo, hi, r)
    for e in axes(lo, 2)
        all(k -> lo[k, e] <= r[k] <= hi[k, e], 1:3) || continue
        Ω = 0.0
        for t in (material.element_offsets[e]+1):material.element_offsets[e+1]
            a, b, c = (SVector{3,Float64}(view(material.vertices, :, material.triangles[k, t])) - r for k in 1:3)
            na, nb, nc = norm(a), norm(b), norm(c)
            Ω -= 2 * atan(dot(a, cross(b, c)), na * nb * nc + dot(a, b) * nc + dot(a, c) * nb + dot(b, c) * na)
        end
        abs(Ω) > 2π && throw(ArgumentError(@sprintf("magnetic material \"%s\": the point (x, y, z) = (%.3f, %.3f, %.3f) m lies inside element %d",
            material.name, r[1], r[2], r[3], e)))
    end
    return nothing
end

# A triangle with its unit normal, outward in-plane edge normals, edge lengths and surface charge σ = M·n̂.
struct _ChargedFace
    v::NTuple{3,SVector{3,Float64}}
    n::SVector{3,Float64}
    m::NTuple{3,SVector{3,Float64}}
    len::NTuple{3,Float64}
    σ::Float64
end

function _charged_faces(material::MagneticMaterial, magnetization)
    faces = _ChargedFace[]
    for e in 1:(length(material.element_offsets)-1)
        M = SVector{3,Float64}(magnetization[1, e], magnetization[2, e], magnetization[3, e])
        iszero(M) && continue
        for t in (material.element_offsets[e]+1):material.element_offsets[e+1]
            v = ntuple(k -> SVector{3,Float64}(view(material.vertices, :, material.triangles[k, t])), 3)
            normal = cross(v[2] - v[1], v[3] - v[1])
            n = normal / norm(normal)
            σ = dot(M, n)
            σ == 0 && continue
            tangents = ntuple(k -> v[mod1(k + 1, 3)] - v[k], 3)
            push!(faces, _ChargedFace(v, n, ntuple(k -> cross(tangents[k], n) / norm(tangents[k]), 3), ntuple(k -> norm(tangents[k]), 3), σ))
        end
    end
    return faces
end

# ∫ (r − r′)/|r − r′|³ dA′ over a flat triangle: the signed solid angle along n̂ plus the edge terms.
@inline function _triangle_kernel(r, f::_ChargedFace)
    a, b, c = f.v[1] - r, f.v[2] - r, f.v[3] - r
    na, nb, nc = norm(a), norm(b), norm(c)
    # Van Oosterom & Strackee: positive when r is on the side n̂ points to.
    Ω = -2 * atan(dot(a, cross(b, c)), na * nb * nc + dot(a, b) * nc + dot(a, c) * nb + dot(b, c) * na)
    R = (na, nb, nc)
    edges = sum(f.m[k] * log((R[k] + R[mod1(k + 1, 3)] + f.len[k]) / (R[k] + R[mod1(k + 1, 3)] - f.len[k])) for k in 1:3)
    return Ω * f.n + edges
end

"""
    saturated_magnetization(material, equil) -> Matrix{Float64}

Magnetization `[3, nelement]` in A/m (Cartesian) of each element of `material`, saturated along the equilibrium
field at its centroid: M = f μ₀M_s B̂₀ / μ₀ with f the fill fraction. Every vertex must lie inside the equilibrium's flux
grid and outside the separatrix (ψ_N ≥ 1).
"""
function saturated_magnetization(material::MagneticMaterial, equil::Equilibrium.PlasmaEquilibrium)
    V = material.vertices
    psi_norm = Equilibrium.equilibrium_field(equil, hypot.(V[1, :], V[2, :]), V[3, :]).psi_norm
    k = findfirst(<(1), psi_norm)
    k === nothing || throw(
        ArgumentError(
            @sprintf(
                "magnetic material \"%s\": vertex %d at (R, Z) = (%.3f, %.3f) m lies at ψ_N = %.4f, inside the separatrix",
                material.name, k, hypot(V[1, k], V[2, k]), V[3, k], psi_norm[k])
        )
    )
    c = material.centroid
    φ = atan.(c[2, :], c[1, :])
    B = Equilibrium.equilibrium_field(equil, hypot.(c[1, :], c[2, :]), c[3, :])
    Bx, By = B.B_R .* cos.(φ) .- B.B_phi .* sin.(φ), B.B_R .* sin.(φ) .+ B.B_phi .* cos.(φ)
    Ms = material.fill_fraction * material.saturation_field / (4π * MU0_OVER_4PI)
    Bmag = sqrt.(Bx .^ 2 .+ By .^ 2 .+ B.B_Z .^ 2)
    return Ms .* vcat((Bx ./ Bmag)', (By ./ Bmag)', (B.B_Z ./ Bmag)')
end

"""
    material_forcing_modes(materials, magnetizations, equil, cfg::CoilConfig, n, m_low, m_high; psi) -> Vector{ForcingMode}

Unit-norm forcing modes (`Φ_x`) of toroidal mode `n` and poloidal modes `m_low:m_high` driven by `materials`, each
magnetized by the matching entry of `magnetizations` (see [`saturated_magnetization`](@ref)), on the control surface
`psi` sampled on the boundary grid of `cfg` (`mtheta_coil`, `nzeta_coil`).
"""
function material_forcing_modes(materials::Vector{MagneticMaterial}, magnetizations, equil::Equilibrium.PlasmaEquilibrium,
    cfg::CoilConfig, n::Int, m_low::Int, m_high::Int; psi::Float64)
    forcing_grid = CoilForcingGrid(equil, cfg, n; psi)
    nobs = length(forcing_grid.obs_R)
    B_R, B_Z = zeros(nobs), zeros(nobs)
    B = (zeros(nobs), zeros(nobs), zeros(nobs))
    for (material, M) in zip(materials, magnetizations)
        compute_magnetic_material_field!(B..., forcing_grid.obs_R, forcing_grid.obs_phi, forcing_grid.obs_Z, material, M)
        B_R .+= B[1]
        B_Z .+= B[3]
    end
    bn = zeros(forcing_grid.grid.mtheta, forcing_grid.grid.nzeta)
    project_normal_flux!(bn, B_R, B_Z, forcing_grid.grid)
    return fourier_decompose_bn(bn, forcing_grid.grid, n, m_low, m_high)
end

export MagneticMaterial, MagneticMaterialConfig, load_magnetic_material, saturated_magnetization, material_forcing_modes
export read_magnetic_material_h5, write_magnetic_material_h5, save_magnetic_material_to_h5, load_magnetic_material_from_h5_group!
export compute_magnetic_material_field!
