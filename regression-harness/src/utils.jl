"""
Git utilities and helper functions.
"""

const LOCAL_REF = "local"

"""
Resolved reference: a human-readable name and the full commit SHA.
For local (uncommitted) working tree, commit_hash is "local".
"""
struct ResolvedRef
    name::String        # Original ref name (branch, tag, or short SHA)
    commit_hash::String # Full 40-char SHA, or "local" for working tree
end

is_local_ref(ref::ResolvedRef) = ref.commit_hash == LOCAL_REF

"""
Resolve a git ref (branch, tag, SHA) to a full commit hash.
Returns a ResolvedRef with the original name and full SHA.
"""
function resolve_ref(ref::String, repo_root::String)::ResolvedRef
    if ref == LOCAL_REF
        return ResolvedRef("local", LOCAL_REF)
    end
    hash = try
        strip(read(`git -C $repo_root rev-parse --verify --quiet $ref`, String))
    catch
        # Try with origin/ prefix for remote tracking branches
        try
            strip(read(`git -C $repo_root rev-parse --verify --quiet origin/$ref`, String))
        catch
            error("Could not resolve ref '$ref' (also tried 'origin/$ref')")
        end
    end
    return ResolvedRef(ref, hash)
end

"""
Expand a ref range (e.g. "develop~10..develop") to a list of ResolvedRefs.
Returns oldest-first ordering.
"""
function expand_ref_range(range::String, repo_root::String)::Vector{ResolvedRef}
    commits = try
        lines = readlines(`git -C $repo_root rev-list --reverse $range`)
        filter(!isempty, lines)
    catch e
        error("Could not expand ref range '$range': $e")
    end
    if isempty(commits)
        error("No commits found in range '$range'")
    end
    return [ResolvedRef(hash[1:8], hash) for hash in commits]
end

"""
Get commit metadata: short hash, date, first line of message.
"""
function get_commit_info(commit_hash::String, repo_root::String)
    short = strip(read(`git -C $repo_root rev-parse --short $commit_hash`, String))
    date = strip(read(`git -C $repo_root log -1 --format=%aI $commit_hash`, String))
    msg = strip(read(`git -C $repo_root log -1 --format=%s $commit_hash`, String))
    # Truncate message to 60 chars
    if length(msg) > 60
        msg = msg[1:57] * "..."
    end
    return (short=short, date=date, msg=msg)
end

"""
How far a ref lags the remote branch it tracks.

Returns `(upstream, behind)` when both the ref and an upstream can be resolved, otherwise
`nothing` (detached SHAs, tags, and local-only branches have no meaningful upstream). A ref that
is behind its upstream is a stale baseline: the harness would happily benchmark against
weeks-old code without saying so.
"""
function upstream_lag(ref::String, repo_root::String)
    ref == LOCAL_REF && return nothing
    # `origin/<ref>` is only a meaningful fallback for an actual local branch name. Without this
    # guard, `HEAD` matches the always-present symbolic ref `origin/HEAD` (→ origin/develop) and
    # every feature branch gets reported as a stale copy of develop.
    is_branch = success(`git -C $repo_root rev-parse --verify --quiet refs/heads/$ref`)
    candidates = is_branch ? ("$(ref)@{upstream}", "origin/$(ref)") : ("$(ref)@{upstream}",)
    upstream = nothing
    for candidate in candidates
        try
            resolved = strip(read(`git -C $repo_root rev-parse --abbrev-ref --verify --quiet $candidate`, String))
            if !isempty(resolved)
                upstream = resolved
                break
            end
        catch
            continue
        end
    end
    upstream === nothing && return nothing
    try
        behind = parse(Int, strip(read(`git -C $repo_root rev-list --count $(ref)..$(upstream)`, String)))
        return (upstream=upstream, behind=behind)
    catch
        return nothing
    end
end

"""
Print a banner for every resolved ref that lags its remote tracking branch.

`resolve_ref` takes the *local* branch pointer, so a local `develop` that has not been fetched
in weeks silently becomes the baseline. Nothing else in the report says the baseline is old.
"""
function warn_stale_refs(refs::Vector{ResolvedRef}, repo_root::String)
    for ref in refs
        lag = upstream_lag(ref.name, repo_root)
        (lag === nothing || lag.behind == 0) && continue
        println()
        println("!! STALE BASELINE: '$(ref.name)' is $(lag.behind) commit(s) behind $(lag.upstream).")
        println("   This comparison is against out-of-date code. Update it with:")
        println("     git fetch && git checkout $(ref.name) && git merge --ff-only $(lag.upstream)")
        println()
    end
end

"""
Create a temporary git worktree for a commit. Returns the worktree path.

`pin_manifest_from` copies an already-resolved `Manifest.toml` into the worktree so that the
subprocess `Pkg.instantiate()` reproduces that exact package set instead of resolving whatever is
newest. Without it, two refs are compared across two different package sets and library-level
differences surface as physics regressions.
"""
function create_worktree(commit_hash::String, repo_root::String;
    pin_manifest_from::Union{String,Nothing}=nothing)::String
    short = commit_hash[1:min(8, length(commit_hash))]
    worktree_path = tempname() * "_gpec_$(short)"
    try
        run(`git -C $repo_root worktree add --detach $worktree_path $commit_hash`)
    catch e
        error("Failed to create worktree for $short: $e")
    end
    if pin_manifest_from !== nothing && isfile(pin_manifest_from)
        cp(pin_manifest_from, joinpath(worktree_path, "Manifest.toml"); force=true)
    end
    return worktree_path
end

"""
Remove a git worktree. Tolerates errors (worktree may already be gone).
"""
function remove_worktree(worktree_path::String, repo_root::String)
    try
        run(`git -C $repo_root worktree remove --force $worktree_path`)
    catch
        # Force cleanup if worktree remove fails
        try
            rm(worktree_path; recursive=true, force=true)
            run(`git -C $repo_root worktree prune`)
        catch e
            @warn "Failed to clean up worktree at $worktree_path: $e"
        end
    end
end
