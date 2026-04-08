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
Create a temporary git worktree for a commit.
Returns the worktree path.
"""
function create_worktree(commit_hash::String, repo_root::String)::String
    short = commit_hash[1:min(8, length(commit_hash))]
    worktree_path = joinpath(tempdir(), "gpec_regress_$(short)_$(getpid())")
    try
        run(`git -C $repo_root worktree add --detach $worktree_path $commit_hash`)
    catch e
        error("Failed to create worktree for $short: $e")
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
        catch
        end
    end
end
