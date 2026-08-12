# figure_tools.jl
#
# Shared helpers for GPEC documentation figures. Every figure generator
# (`docs/src/figures/<module>/make_<name>.jl`) `include`s this file and calls
# `save_doc_figure` to stamp, save, and register its output. See
# `docs/DOC_STANDARD.md` for the figure-organization and provenance policy.
#
# This file lives OUTSIDE `docs/src/` so Documenter does not publish it. The
# `make_<name>.jl` scripts do live under `docs/src/figures/` (co-located with
# their PNGs) and are published alongside the figures for reproducibility.
#
# Figure scripts run MANUALLY in the root project, never at docs-build time:
#     julia --project=. docs/src/figures/<module>/make_<name>.jl
# so the Documenter build stays fast and Plots-free.

using Plots
using Dates
using TOML

const _DOCS_DIR = @__DIR__                                   # .../docs
const _REPO_ROOT = normpath(joinpath(_DOCS_DIR, ".."))       # repo root
const FIGURES_ROOT = joinpath(_DOCS_DIR, "src", "figures")   # docs/src/figures
const MANIFEST_PATH = joinpath(FIGURES_ROOT, "manifest.toml")

"""
    figure_provenance() -> (; commit, date, dirty)

Short git commit and ISO date stamped onto every figure. `commit` is the
`git rev-parse --short HEAD` hash (reusing the idiom from
`benchmarks/benchmark_git_branches.jl`), with a `-dirty` suffix when the
working tree has uncommitted changes so a figure never claims a cleaner
provenance than it has. Falls back to `"unknown"` outside a git checkout.
"""
function figure_provenance()
    date = string(Dates.today())
    try
        h = strip(read(`git -C $(_REPO_ROOT) rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $(_REPO_ROOT) status --porcelain`, String)))
        return (; commit=(dirty ? "$h-dirty" : h), date=date, dirty=dirty)
    catch
        return (; commit="unknown", date=date, dirty=false)
    end
end

"""
    stamp!(p; commit, date, subplot=length(p.subplots))

Annotate a small `GPEC <commit> · <date>` provenance mark in the bottom-right
corner of subplot `subplot` (default: the last panel), so the figure's age is
visible when browsing the rendered docs. Works on linear or log axes
(positions in data coordinates). Never throws: a plotting hiccup warns and
leaves the figure unstamped rather than losing the plot.
"""
function stamp!(p::Plots.Plot; commit::AbstractString, date::AbstractString,
    subplot::Int=length(p.subplots))
    try
        sp = p[subplot]
        xl = Plots.xlims(sp)
        yl = Plots.ylims(sp)
        xpos = xl[2]                                          # right edge (data coords)
        ypos = yl[1]                                          # bottom edge
        txt = Plots.text("GPEC $(commit) · $(date)", 6, RGBA(0.5, 0.5, 0.5, 0.9),
            :right, :bottom)
        annotate!(sp, xpos, ypos, txt)
    catch err
        @warn "stamp! failed; saving figure without provenance mark" exception = err
    end
    return p
end

"""
    save_doc_figure(p, mod, name; script, depends, npx...) -> String

Stamp `p` with the current git provenance, save it to
`docs/src/figures/<mod>/<name>.png`, and upsert its `manifest.toml` entry
(`script`, `commit`, `date`, `depends`). `depends` lists the repo-relative
source files whose numbers the figure visualizes — the regeneration policy in
`docs/DOC_STANDARD.md` keys off them. Returns the absolute PNG path (printed
so it can be opened directly). Extra keyword args are forwarded to `savefig`.
"""
function save_doc_figure(p::Plots.Plot, mod::AbstractString, name::AbstractString;
    script::AbstractString, depends::AbstractVector{<:AbstractString}=String[])
    prov = figure_provenance()
    stamp!(p; commit=prov.commit, date=prov.date)

    outdir = joinpath(FIGURES_ROOT, mod)
    mkpath(outdir)
    outpath = joinpath(outdir, name * ".png")
    savefig(p, outpath)

    _update_manifest!(mod, name; script=String(script), commit=prov.commit,
        date=prov.date, depends=String.(depends))

    println("Saved doc figure: $(abspath(outpath))")
    return abspath(outpath)
end

# Load / merge-write the figures manifest, keyed [<mod>.<name>]. Kept sorted and
# human-diffable; one entry per committed figure.
function _load_manifest()
    return isfile(MANIFEST_PATH) ? TOML.parsefile(MANIFEST_PATH) : Dict{String,Any}()
end

function _write_manifest(manifest)
    open(MANIFEST_PATH, "w") do io
        println(io, "# GPEC documentation-figure provenance manifest.")
        println(io, "# One [<module>.<figure>] entry per committed PNG under docs/src/figures/.")
        println(io, "# Regenerate a figure (and this entry) only when a file in its `depends`")
        println(io, "# list changes the numbers it shows — see docs/DOC_STANDARD.md.")
        println(io)
        TOML.print(io, manifest; sorted=true)
    end
    return manifest
end

function _update_manifest!(mod::AbstractString, name::AbstractString;
    script::String, commit::String, date::String, depends::Vector{String})
    manifest = _load_manifest()
    modtbl = get!(manifest, mod, Dict{String,Any}())
    modtbl[name] = Dict{String,Any}(
        "script" => script,
        "commit" => commit,
        "date" => date,
        "depends" => depends
    )
    return _write_manifest(manifest)
end

"""
    register_legacy_figure(mod, name; note, depends=String[])

Record a manifest entry for a figure that predates this provenance system and
whose original generator script was not preserved (`commit = "legacy"`). Use
this only for migrated figures that cannot be faithfully reproduced from the
current code — never to skip writing a generator for a new figure. `note`
states why the figure is legacy (e.g. "compared the since-removed Mercier.jl").
"""
function register_legacy_figure(mod::AbstractString, name::AbstractString;
    note::AbstractString, depends::AbstractVector{<:AbstractString}=String[])
    manifest = _load_manifest()
    modtbl = get!(manifest, mod, Dict{String,Any}())
    modtbl[name] = Dict{String,Any}(
        "script" => "(not preserved)",
        "commit" => "legacy",
        "date" => "unknown",
        "depends" => String.(depends),
        "note" => String(note)
    )
    _write_manifest(manifest)
    println("Registered legacy figure: $(mod).$(name)")
    return manifest
end

"""
    step_series(m_vals, amps) -> (m_ext, amp_ext)

Canonical spectrum-plot helper (CLAUDE.md plotting convention): pad a zero on
each end of a discrete-mode series so a `seriestype=:steppre` plot draws clean
boxes that fall to zero at the edges. Centralized here so doc figures and
benchmarks share one definition.
"""
function step_series(m_vals, amps)
    m_ext = [m_vals[1] - 1; m_vals; m_vals[end] + 1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end
