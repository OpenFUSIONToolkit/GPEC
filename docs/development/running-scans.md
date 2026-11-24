# Running cases and scans

Every fresh Julia process that runs GPEC pays a fixed cost before it computes anything: about
12 s to load the package, then tens of seconds compiling the first case's code paths. A process
that runs one case and exits pays that cost on every case, and a source edit adds a rebuild of
the package image on top. This page measures those costs and gives the run structures that pay
them once. It is written for people and for AI agents driving GPEC on a workstation or a Slurm
cluster; the regression harness applies the same ideas to its own cases (see
[regression-harness.md](regression-harness.md)).

## Rules

1. **Run many cases in one process.** Load GPEC once and loop over deck variants; never launch
   `julia` once per case.
2. **Parallelize across cases before threads.** For small and medium cases a scan scales better
   as a job array of chunks, one process per task, than as one process with many threads.
3. **Build the package image once, before submitting**, on the CPU type the jobs will run on or
   with a `JULIA_CPU_TARGET` that covers every node type.
4. **Never edit the checkout a job loads from.** Run scans from a worktree pinned to a commit.
5. **For edit-test loops, keep one Julia session alive with Revise**; do not restart Julia after
   every edit.
6. **Give every case its own directory** and its own log, and make the script skip cases that
   already finished.
7. **Hold threads, build mode and commit fixed** across the points of a scan you compare.
8. **Log load and first-case time separately from compute**, so reports show what the physics
   cost.

## Where the time goes

Measured with Julia 1.11 on a two-socket Xeon (Ice Lake) node through Slurm, with the pinned
manifest from `ci/manifests/` and the default build (`precompile_workload` on). Each number is
the range over two runs.

| Cost | Time |
|---|---|
| `julia -e 1` | 0.16 s |
| `using GeneralizedPerturbedEquilibrium`, valid package image | 12.0–13.7 s |
| Rebuild of the GPEC image after editing one file under `src/` | 191–195 s |
| Same, with `precompile_workload = false` | 37 s |
| Build of a fresh depot (all dependencies and GPEC), 8 CPUs | 533 s |
| `examples/Solovev_ideal_example`: first case in a process / every later case (4 threads) | 19.5–21.7 s / 9.2–10.1 s |
| `examples/DIIID-like_ideal_example`: first case / every later case (4 threads) | 70–74 s / 22.8–24.8 s |
| Peak resident memory of a process running the DIII-D-like case | 3.1 GB |

For the DIII-D-like case, 59 s of the 83 s to the first result are load and compilation; the
physics takes 24 s. Touching a file under `src/` without changing its content does not trigger a
rebuild.

A 10-point scan of the Solovev elongation in one 4-CPU allocation, structured four ways:

| Structure | Wall time |
|---|---|
| One `julia` process per case, valid image | 327–332 s |
| One `julia` process per case, after a source edit (the first process rebuilds) | 521–545 s |
| One process, loop over the 10 cases, 4 threads | 117–120 s |
| `Distributed`, 4 workers × 1 thread in one job | 72–73 s |
| Job array of 4 tasks × 1 CPU, one chunk of cases per task | 60–61 s |

A Solovev case takes 9.2–10.1 s at 4 threads and 11.6–12.0 s at 1 thread, so four single-threaded cases
side by side finish far sooner than four-threaded cases in a row. Thread scaling depends on the
case; measure your own before choosing threads over tasks.

## A scan script

Load GPEC once, then loop. Copy the whole example directory for each case, so relative file
references in the deck (`eq_filename`, `kinetic_file`, wall files) still resolve, then edit the
copied deck. [`examples/LAR_beta_scan/run_scan.jl`](../../examples/LAR_beta_scan/run_scan.jl) is
a complete scan that also extracts and tabulates results; the skeleton below adds per-case
directories kept on disk, restart after an interruption, and job-array chunking.

```julia
# Solovev elongation scan in one Julia process: julia --project=<GPEC repo> -t 4 scan_elongation.jl <outdir>
# Under a Slurm job array, task k of N runs cases k+1, k+1+N, ...
const T0 = time()
using GeneralizedPerturbedEquilibrium, TOML, Printf, Logging
const T_LOAD = time() - T0

const BASE = joinpath(pkgdir(GeneralizedPerturbedEquilibrium), "examples", "Solovev_ideal_example")
const OUT = abspath(get(ARGS, 1, "elongation_scan"))
mkpath(OUT)
const VALUES = range(1.4, 1.85; length=10)

function run_case(dir, value)
    isfile(joinpath(dir, "DONE")) && return 0.0  # finished in an earlier attempt
    rm(dir; force=true, recursive=true)
    cp(BASE, dir)  # the whole deck directory, so relative file references still resolve
    deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
    deck["Solovev"]["elongation"] = value
    open(io -> TOML.print(io, deck), joinpath(dir, "gpec.toml"), "w")
    t = @elapsed open(joinpath(dir, "run.log"), "w") do io
        redirect_stdio(; stdout=io, stderr=io) do
            with_logger(ConsoleLogger(io)) do
                GeneralizedPerturbedEquilibrium.main([dir])
            end
        end
    end
    touch(joinpath(dir, "DONE"))
    return t
end

task = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0")) - parse(Int, get(ENV, "SLURM_ARRAY_TASK_MIN", "0"))
ntask = parse(Int, get(ENV, "SLURM_ARRAY_TASK_COUNT", "1"))
@printf("load %.1f s, %d threads, task %d of %d\n", T_LOAD, Threads.nthreads(), task + 1, ntask)
for (i, value) in enumerate(VALUES)
    (i - 1) % ntask == task || continue
    dir = joinpath(OUT, @sprintf("case_%02d", i))
    t = try
        run_case(dir, value)
    catch e
        @error "case $i failed; see $(joinpath(dir, "run.log"))" exception = (e, catch_backtrace())
        NaN
    end
    @printf("case %2d  elongation %.3f  %6.1f s\n", i, value, t)
end
```

The first case a process runs includes compilation, so its time is not the cost of the physics;
the load time and that first case are printed separately for that reason. A case that fails is
logged and the loop moves on; rerunning the script redoes only the cases without a `DONE` marker.

**Reuse upstream stages.** `main` solves every stage of the deck for every case. When the scanned
parameter only affects a later stage, the [scripting API](../src/api.md) can build the
equilibrium or the force-free states once and loop over the later call alone. In the DIII-D-like
example the equilibrium is 5.5 s and the force-free states 7.8 s of the 24 s compute.

## Parallel scans

**Job array, one chunk of cases per task.** Each task is one process looping over its share, as
in the script above (`$GPEC` is the GPEC checkout; size `--mem` from one test case, 3.1 GB for
the DIII-D-like example):

```bash
sbatch --array=0-3 -c 1 --mem=4G \
  --wrap 'export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
          julia --project=$GPEC scan_elongation.jl $SCRATCH/elongation_scan'
```

Choose the number of tasks so each runs several cases; one case per task pays the full load and
compilation for one result. Tasks are independent, so a crash, an out-of-memory kill or a node
failure loses one chunk, and the queue can start tasks as cores free up.

**`Distributed` in one job.** `addprocs(k; exeflags=["--project=$GPEC", "-t", "1"])`, then
`@everywhere` load GPEC and `pmap` over the cases. Every worker loads and compiles GPEC itself,
so it costs at least the overhead of an array of `k` tasks (here 72–73 s against 60–61 s, the
difference being worker start-up), and all workers live and die with one job. Set
`OPENBLAS_NUM_THREADS=1` for the workers, or `k` workers each start a full set of BLAS threads.
Use it when the cases must share one allocation or results must be gathered in memory.

**Threads within one case** pay off for large cases (high `mpsi`, many toroidal modes, kinetic
runs); measure a few thread counts on one representative case before choosing. Results can
differ in the last bits between thread counts, and the parallel Riccati integrator's chunking
follows the thread count, so use one thread count for every point you compare.

## Build the package image once

Julia keeps compiled package images in the depot (`$JULIA_DEPOT_PATH/compiled`). A job that
finds the image missing or stale builds it before it does anything else. Building takes over
three minutes for GPEC alone and several more for a fresh depot, with the job holding its whole
allocation while one core does most of the work. Build it once, in its own step, before
submitting the scan:

```bash
julia --project=$GPEC -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

**Concurrent tasks on a stale image.** Four array tasks started after a source edit did not
build four images: one built it and the others waited on Julia's lock file, so the array took
247–256 s instead of 60 s, with every task idle for over three minutes. A separate build step
followed by the array was not faster in total (255 s, and 400 s in a run whose build step was
unexpectedly slow), but it held one core instead of
four during the build and reports a build failure once, before anything is queued. Only tasks on
one host were tested; tasks on different nodes that share a depot rely on the lock file working
across the shared filesystem.

**CPU targets.** By default Julia compiles for the CPU of the machine that builds the image
(`native`). A node with a different CPU rejects that image and rebuilds it. With `JULIA_DEBUG=loading`
the rejection reads `Unable to find compatible target in cached code image`. In a test with one
depot on a shared filesystem, an image built on an Ice Lake node was rejected on a Cascade Lake
node, which then spent 482 s rebuilding. Building with a target list that covers both:

```bash
export JULIA_CPU_TARGET="generic;cascadelake,clone_all;icelake-server,clone_all"
```

took about as long (523 s against 533 s for the native build) and loaded on the Cascade Lake
node with no rejection, including from a job that did not set the variable. Set it in the environment of every process that might
rebuild the image (shell profile or job script), so a rebuild on any node produces an image every
node can use. List the microarchitectures of your login and compute nodes
(`julia -e 'println(Sys.CPU_NAME)'` on each); `generic` keeps a fallback for anything else. If
the login and compute nodes share a CPU type, building on the login node is enough.

**Depot location.** The first load of a freshly built image from a network filesystem took
about twice as long as later loads (30 s with debug logging on, then 12.5 s). Put the depot where every compute node can read it, on the fastest such filesystem, and
build into it from one place.

## Edit-test loops

**Revise.** In a Julia session that loaded `Revise` before GPEC, an edit under `src/` recompiles
only the changed methods and their callers instead of rebuilding the image (see
[Revise.jl](../src/developer_notes.md#revisejl)). Measured on the Solovev example, after a
one-line edit inside a ForceFreeStates function, `Revise.revise()` took 2.8–2.9 s and the next
case 10.2–10.4 s, against 9.8–10.4 s with no edit. The same edit without Revise costs a
191–195 s rebuild, a 12 s load and a 20 s first case. On a machine whose inotify watch limit is
exhausted, Revise reports `FolderMonitor ... no space left on device (ENOSPC)`; set
`JULIA_REVISE_POLL=1` so it polls files instead.

**A warm worker for scripted loops.** An agent or a shell script cannot type into a REPL, but it
can keep one Julia process alive and feed it scripts. This is the pattern the regression harness
uses: load once, run each request in a fresh module. It needs Revise installed as above.

```julia
# warm_worker.jl: keep GPEC loaded and run each script path written to the named pipe ARGS[1], picking up source edits through Revise.
using Revise, GeneralizedPerturbedEquilibrium
println("ready"); flush(stdout)
while true
    for script in eachline(ARGS[1])  # blocks until something writes to the pipe
        Revise.revise()
        t = @elapsed try
            Base.include(Module(), script)  # fresh namespace for every request
        catch e
            @error "failed: $script" exception = (e, catch_backtrace())
        end
        println("done $script in $(round(t; digits=1)) s"); flush(stdout)
    end
end
```

```bash
mkfifo requests
julia --project=$GPEC -t 4 warm_worker.jl requests > worker.log 2>&1 &
echo $PWD/my_case.jl > requests     # returns at once; watch worker.log for "done"
```

In a test the worker was ready after 15 s. Its first Solovev request took 20 s (compilation);
later ones took about 10 s, including the first after a source edit. Revise cannot apply a change
to a `struct` definition, so restart the worker after one and after switching branches.

**`precompile_workload`.** The package image carries a small precompile workload
(`src/Precompile.jl`), on by default. It makes each image build about 150 s longer and every
fresh process faster to its first result:

| | Workload on | Workload off |
|---|---|---|
| Rebuild after a source edit | 191–195 s | 37 s |
| First Solovev case in a process | 19.5–19.7 s | 100–103 s |
| First DIII-D-like case in a process | 70–72 s | 236–238 s |

Keep it on for production scans and anything that starts fresh processes from an unchanged
checkout. Turn it off ([Precompile workload](../src/developer_notes.md#precompile-workload)) only
for loops that rebuild and run one small case per edit, and prefer Revise to either. The two
builds can round differently in the last bit, which near-marginal quantities such as Δ′
amplify, so do not mix build modes within one scan.

**Inspecting results.** Each `julia -e` one-liner pays the full load. Read `gpec.h5` from the
session that produced it, from a warm worker, or with an HDF5 tool (`h5ls`, `h5dump`, Python
`h5py`); the layout is described in [hdf5-conventions.md](hdf5-conventions.md).

## State isolation between cases

The regression harness runs every case at a commit in one process and produces results
bit-identical to one process per case. What makes that safe, and what a scan script must do too:

- **Output paths.** `main` writes `gpec.h5` (`[Run] output_file`) into the deck directory, so
  every case needs its own directory. Two processes running the same directory overwrite each
  other's output.
- **Logs.** Redirect each case's stdout, stderr and logger to a file in its directory, as above,
  so a failure can be read without searching a combined log.
- **Module-level caches.** Vacuum keeps per-`n` Legendre quadrature entries and the singular
  quadrature data; KineticForces keeps θ-quadrature weights per `ntheta`. Each entry is a pure
  function of its key and is guarded by a lock, so reusing them across cases and threads is
  correct and saves time. `Vacuum.reset_caches!()` and `KineticForces.reset_caches!()` exist to
  keep run state out of the package image, not for use between cases.
- **Script globals.** A case script `include`d repeatedly into `Main` shares its globals and
  `const`s with every other case, and redefining a `const` warns or errors. Include each into a
  fresh `Module()`, as the warm worker does, or put the case in a function.
- **Build preference.** `precompile_workload` is read when the process starts, so cases that need
  the other build mode need another process.

## Snapshots

A running job sees the source it loaded; a job that has not yet started sees whatever is on disk
when it starts. Editing the checkout while a scan is queued or running therefore produces a scan
whose points came from different code, or tasks that rebuild the image mid-scan. Run scans from
a worktree pinned to a commit and leave it alone until the jobs finish:

```bash
git worktree add --detach ../gpec-scan <commit>
cp Manifest.toml ../gpec-scan/     # same package set as the checkout you tested with
julia --project=../gpec-scan -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Keep the worktree path stable across scans: package images are keyed on the source path, so a
new path is a full GPEC rebuild. Record the commit and the `Manifest.toml` hash with the results.
