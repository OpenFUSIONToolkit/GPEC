#!/usr/bin/env bash
# WSL / Linux: sweep thread counts and write CSV + optional plot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$ROOT/benchmarks/equilibrium_threads_results.csv}"
NMAX="${2:-20}"
julia --project="$ROOT" "$ROOT/benchmarks/equilibrium_threads_sweep.jl" "$OUT" "$NMAX"
julia --project="$ROOT" "$ROOT/benchmarks/plot_equilibrium_threads.jl" "$OUT"
