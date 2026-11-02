# Benchmarking

Use the generic benchmarking tool at `benchmarks/benchmark_git_branches.jl` to compare performance between branches or commits.

**Tool usage:**

```bash
# Compare feature branch against develop
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --branch1 develop \
  --branch2 feature-branch

# Compare specific commits
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --commit1 abc123 \
  --commit2 def456

# Compare current develop vs develop from 1 month ago
julia benchmarks/benchmark_git_branches.jl \
  --example examples/DIIID-like_ideal_example \
  --branch1 develop \
  --commit1 HEAD~10 \
  --branch2 develop
```

**Default benchmark case:** `examples/DIIID-like_ideal_example`

**Reported metrics:**
1. **Eigenmode energy (`et[1]`)** - First eigenvalue; verifies calculation correctness
2. **Integration steps** - Total ODE solver steps
3. **Runtime (warmed)** - Wall-clock time averaged over multiple warm runs (JIT warmup handled automatically)
4. **Commit hash** - Git commit of code tested

**The tool automatically:**
- Handles JIT warmup (runs example 3 times, averages last 2)
- Switches between branches/commits
- Stashes uncommitted changes if necessary
- Restores original branch when done
- Reports comparison with percentage differences

**Important notes:**
- Working directory should be clean or changes will be stashed during branch switching
- Tool requires HDF5.jl for reading `gpec.h5` output
- Each benchmark run takes several minutes per branch (includes compilation + warm runs)

**Benchmark script conventions:**
- Benchmark scripts must reference input data from `examples/` (e.g., `joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")`). Never duplicate example inputs into `benchmarks/`.
- If a benchmark needs modified TOML settings or a parameter scan, copy inputs to a temporary local directory at runtime — do not commit these copies.
- All outputs (figures, CSVs, HDF5 files) must be saved into `benchmarks/` itself (or a self-described subdirectory within it, e.g., `benchmarks/coil_scan_results/`). Output files are not committed.
