"""
    HDF5Annotations

Self-describing metadata for `gpec.h5` (the contract in
`docs/development/hdf5-conventions.md`): every dataset carries a `long_name` and
`units` attribute, array datasets carry a `dims` axis-name attribute, and coordinate
datasets are marked as HDF5 Dimension Scales (netCDF-4 coordinate variables) attached
to the arrays that share their axis, so h5py/xarray/HDFView read the file unaided.

Writers stay table-driven: each writer keeps a table of `path => (; long_name, units,
dims)` entries next to it and calls [`annotate!`](@ref) once after its datasets are
written. Paths absent from the file are skipped silently (many writes are conditional).
"""
module HDF5Annotations

using HDF5
using Dates

export annotate!, make_scale!, attach_scale!, write_root_attrs!

"""
    annotate!(parent, table)

Apply a metadata table to datasets under `parent` (an open `HDF5.File` or group).
`table` iterates `path => meta` pairs where `meta` is a NamedTuple with fields
`long_name` (required), `units` (default `"1"` = dimensionless), and optionally
`dims` — a tuple of axis names in Julia (column-major) order, axis 1 first, stored
as the greppable string attribute `dims = "(psi, m)"`. Missing paths are skipped.
"""
function annotate!(parent::Union{HDF5.File,HDF5.Group}, table)
    for (path, meta) in table
        haskey(parent, path) || continue
        a = attrs(parent[path])
        a["long_name"] = String(meta.long_name)
        a["units"] = String(get(meta, :units, "1"))
        d = get(meta, :dims, nothing)
        d === nothing || (a["dims"] = "(" * join(d, ", ") * ")")
    end
    return parent
end

"""
    make_scale!(parent, path, name)

Mark the dataset at `path` as an HDF5 Dimension Scale named `name`. No-op when the
path is absent.
"""
function make_scale!(parent::Union{HDF5.File,HDF5.Group}, path::AbstractString, name::AbstractString)
    haskey(parent, path) || return nothing
    HDF5.API.h5ds_set_scale(parent[path], String(name))
    return nothing
end

"""
    attach_scale!(parent, path, julia_axis, scale_path, label)

Attach the Dimension Scale at `scale_path` to Julia axis `julia_axis` (axis 1 first)
of the dataset at `path`, and label that dimension. The H5DS C API indexes file
(row-major) dimensions, so Julia axis `k` of an `N`-d dataset is C index `N - k`.
No-op when either path is absent or the axis lengths disagree.
"""
function attach_scale!(parent::Union{HDF5.File,HDF5.Group}, path::AbstractString, julia_axis::Int,
    scale_path::AbstractString, label::AbstractString)
    (haskey(parent, path) && haskey(parent, scale_path)) || return nothing
    dset = parent[path]
    sc = parent[scale_path]
    size(dset, julia_axis) == length(sc) || return nothing
    cdim = ndims(dset) - julia_axis
    HDF5.API.h5ds_attach_scale(dset, sc, cdim)
    # h5ds_set_label's wrapper types the C `const char*` as Ref{UInt8}; pass a
    # NUL-terminated byte buffer instead of a String.
    HDF5.API.h5ds_set_label(dset, cdim, Vector{UInt8}(codeunits(String(label) * "\0")))
    return nothing
end

"""
    write_root_attrs!(file; title)

Stamp the file-level contract: `schema_version`, `Conventions`, `references`,
`title` (run description), `date_created` (ISO 8601 UTC). The code version lives in
`Info/git_version`.
"""
function write_root_attrs!(file::HDF5.File; title::AbstractString)
    a = attrs(file)
    a["schema_version"] = "2.0"
    a["Conventions"] = "GPEC-HDF5-2.0"
    a["references"] = "docs/development/hdf5-conventions.md; https://openfusiontoolkit.github.io/GPEC/dev/"
    a["title"] = String(title)
    a["date_created"] = Dates.format(Dates.now(UTC), dateformat"yyyy-mm-dd\THH:MM:SS\Z")
    return file
end

end # module HDF5Annotations
