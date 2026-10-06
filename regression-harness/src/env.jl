"""
Environment fingerprinting.

A cached regression result is only comparable to a fresh one if both were produced by the same
Julia, on the same machine, against the same package set. Because `Manifest.toml` is untracked,
a worktree checkout resolves whatever is newest at run time, so two runs of the *same source*
can differ by a package set — and machine-epsilon differences in library math are amplified by
the adaptive ODE controller into large apparent regressions. The fingerprint below makes that
confound visible (and, by default, invalidating) instead of silent.
"""

"""
Identity of the environment a run was produced in.

## Fields

  - `julia_version::String` — version of the `julia` that ran GPEC
  - `os_arch::String` — `Sys.MACHINE` of the running host
  - `manifest_sha::String` — SHA-256 of the `Manifest.toml` the run resolved against ("" if absent)
  - `nthreads::Int` — `Threads.nthreads()` in the run (-1 if unknown)
  - `blas_threads::Int` — `BLAS.get_num_threads()` in the run (-1 if unknown)
  - `pinned::Bool` — whether the harness copied its own Manifest into the run's project
"""
struct EnvFingerprint
    julia_version::String
    os_arch::String
    manifest_sha::String
    nthreads::Int
    blas_threads::Int
    pinned::Bool
end

const UNKNOWN_ENV = EnvFingerprint("", "", "", -1, -1, false)

"""
Cache key for an environment.

Deliberately built from only the three fields that are knowable *before* a run: the Julia
version, the host, and the package set the run will be pinned to. Thread counts are recorded and
reported but not keyed, because the harness does not force them (see `--threads` in the CLI) and
so cannot predict them ahead of a run.

When the Manifest is not pinned, the package set is unknowable in advance and the key records
`unpinned`; the resolved Manifest hash is still stored per run for display and mismatch warnings.
"""
function env_key(julia_version::AbstractString, os_arch::AbstractString, manifest_mode::AbstractString)::String
    return bytes2hex(SHA.sha256("$(julia_version)|$(os_arch)|$(manifest_mode)"))[1:16]
end

env_key(fp::EnvFingerprint) = env_key(fp.julia_version, fp.os_arch, fp.pinned ? fp.manifest_sha : "unpinned")

"""SHA-256 of a file, or "" when it does not exist."""
function file_sha256(path::AbstractString)::String
    isfile(path) || return ""
    return bytes2hex(SHA.sha256(read(path)))
end

"""
Version string of the `julia` the harness will launch for subprocess runs.

This is not necessarily the Julia running the harness itself, so it is probed rather than read
from `VERSION`.
"""
function subprocess_julia_version()::String
    try
        out = strip(read(`julia --version`, String))
        m = match(r"(\d+\.\d+\.\d+\S*)", out)
        return m === nothing ? out : m.captures[1]
    catch
        return "unknown"
    end
end

"""
The environment key that runs launched *now* will carry.

`manifest_path` is the Manifest the harness will pin into each worktree; pass `nothing` when
pinning is disabled.
"""
function expected_env_key(manifest_path::Union{String,Nothing})::String
    mode = manifest_path === nothing ? "unpinned" : file_sha256(manifest_path)
    return env_key(subprocess_julia_version(), string(Sys.MACHINE), mode)
end

"""
Parse the `key=value` run-info file written by a run subprocess.

Returns `(runtime_s, fingerprint)`. A missing or malformed file yields `NaN` and `UNKNOWN_ENV`
rather than throwing, so a run that produced output but no metadata still reports its numbers.
"""
function read_runinfo(path::String, pinned::Bool)
    isfile(path) || return (NaN, UNKNOWN_ENV)
    fields = Dict{String,String}()
    for line in eachline(path)
        parts = split(line, '='; limit=2)
        length(parts) == 2 && (fields[strip(parts[1])] = strip(parts[2]))
    end
    getf = (k, d) -> get(fields, k, d)
    runtime_s = tryparse(Float64, getf("runtime_s", ""))
    fp = EnvFingerprint(
        getf("julia_version", ""),
        getf("os_arch", ""),
        getf("manifest_sha", ""),
        something(tryparse(Int, getf("nthreads", "")), -1),
        something(tryparse(Int, getf("blas_threads", "")), -1),
        pinned
    )
    return (something(runtime_s, NaN), fp)
end

"""One-line human-readable summary of an environment, for report headers."""
function describe_env(fp::EnvFingerprint)::String
    isempty(fp.julia_version) && return "environment unknown (cached before fingerprinting)"
    mani = isempty(fp.manifest_sha) ? "no Manifest" : "manifest " * fp.manifest_sha[1:min(8, end)]
    pin = fp.pinned ? "pinned" : "unpinned"
    threads = "$(fp.nthreads) thread$(fp.nthreads == 1 ? "" : "s")/$(fp.blas_threads) BLAS"
    return "julia $(fp.julia_version), $(fp.os_arch), $mani ($pin), $threads"
end
