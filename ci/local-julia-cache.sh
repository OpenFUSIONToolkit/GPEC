#!/usr/bin/env bash
# Publish immutable depot snapshots; restore into each runner's private depot.
set -euo pipefail

operation=${1:?Expected restore, require, or save}
key=${2:?Expected a cache key}
: "${GPEC_LOCAL_CACHE_DIR:?Set the shared local cache directory}"
[[ "$key" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'Invalid cache key' >&2; exit 1; }
entry="$GPEC_LOCAL_CACHE_DIR/$key"
paths=(artifacts packages registries compiled scratchspaces logs)

case "$operation" in
    restore|require)
        if [[ ! -f "$entry/complete" ]]; then
            echo "Local Julia cache miss: $key"
            if [[ "$operation" == require ]]; then
                echo 'The precompile job did not publish a usable local cache.' >&2
                exit 1
            fi
            exit 0
        fi
        mkdir -p "$HOME/.julia"
        for path in "${paths[@]}"; do
            if [[ -d "$entry/depot/$path" ]]; then
                mkdir -p "$HOME/.julia/$path"
                cp -a --reflink=auto "$entry/depot/$path/." "$HOME/.julia/$path/"
            fi
        done
        cp "$entry/Manifest.toml" Manifest.toml
        echo "Local Julia cache restored: $key"
        ;;
    save)
        mkdir -p "$GPEC_LOCAL_CACHE_DIR"
        # Readers only see complete snapshots; concurrent publishers reuse the winner.
        exec 9>"$GPEC_LOCAL_CACHE_DIR/.publish.lock"
        flock 9
        if [[ -f "$entry/complete" ]]; then
            echo "Local Julia cache already published: $key"
            exit 0
        fi
        staging=$(mktemp -d "$GPEC_LOCAL_CACHE_DIR/.staging.XXXXXXXX")
        trap 'rm -rf -- "$staging"' EXIT
        mkdir "$staging/depot"
        for path in "${paths[@]}"; do
            if [[ -d "$HOME/.julia/$path" ]]; then
                cp -a --reflink=auto "$HOME/.julia/$path" "$staging/depot/"
            fi
        done
        cp Manifest.toml "$staging/Manifest.toml"
        touch "$staging/complete"
        mv -T "$staging" "$entry"
        echo "Local Julia cache published: $key"
        ;;
    *) echo "Unknown operation: $operation" >&2; exit 1 ;;
esac
