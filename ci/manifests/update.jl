"""
Regenerate the pinned CI dependency sets in `ci/manifests/`.

Run once per Julia minor version that `.github/workflows/test.yaml` builds, from
anywhere:

    julia +1.11 ci/manifests/update.jl
    julia +1.12 ci/manifests/update.jl

Each run resolves `Project.toml` for the running Julia version and writes
`ci/manifests/Manifest-v<major>.<minor>.toml`, then refreshes
`ci/manifests/project.sha256` so CI can tell when a pin has fallen behind
`Project.toml`. Resolution happens in a temporary directory, so the developer's
own `Manifest.toml` is left untouched.
"""

using SHA
using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PIN_DIR = joinpath(REPO_ROOT, "ci", "manifests")
const PROJECT = joinpath(REPO_ROOT, "Project.toml")

pin = joinpath(PIN_DIR, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")

# Resolve a bare copy of Project.toml so the pin records a clean maximal resolve
# rather than whatever the working tree happens to be sitting on.
mktempdir() do dir
    cp(PROJECT, joinpath(dir, "Project.toml"))
    withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
        Pkg.activate(dir)
        Pkg.resolve()
        Pkg.instantiate()
    end
    cp(joinpath(dir, "Manifest.toml"), pin; force=true)
end

write(joinpath(PIN_DIR, "project.sha256"), bytes2hex(open(sha256, PROJECT)) * "\n")

@info "Wrote $(relpath(pin, REPO_ROOT)) for Julia $VERSION"
