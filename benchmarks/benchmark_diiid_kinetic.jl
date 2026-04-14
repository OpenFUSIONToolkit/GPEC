#!/usr/bin/env julia
"""
    benchmark_diiid_kinetic.jl

Benchmark Julia KineticForces NTV against Fortran PENTRC reference values.

Loads Fortran `gpec_xclebsch_n1.out` directly to populate PE Clebsch fields,
then runs the KF JBB deweighting + torque integration pipeline. This isolates
the KF code from the PE pipeline for direct comparison.

Fortran reference values (pentrc_output_n1.nc global attributes):
  T_total_fgar  = 1.78040606348608 N·m
  T_total_tgar  = 0.948961891682954 N·m
  dW_total_fgar = 0.136828994301428
  dW_total_tgar = 0.112296587981593
"""

using Printf

# Load GPEC
using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates
const KF  = GPE.KineticForces
const Eq  = GPE.Equilibrium
const PE  = GPE.PerturbedEquilibrium

# Paths
const BENCHMARK_DIR = joinpath(@__DIR__, "DIIID_kinetic_example")
const FORTRAN_DIR   = expanduser("~/Code/gpec/docs/examples/DIIID_ideal_example")

# Fortran reference values
const REF = (
    T_total_fgar  = 1.78040606348608,
    T_total_tgar  = 0.948961891682954,
    dW_total_fgar = 0.136828994301428,
    dW_total_tgar = 0.112296587981593
)

"""
    load_fortran_xclebsch(filepath) → (psi_grid, clebsch_psi, clebsch_psi1, clebsch_alpha, mpert, mlow)

Parse Fortran `gpec_xclebsch_n1.out` into arrays matching PE xi_modes layout.

Header format:
    Line 1: title
    Line 6: mstep, mpert, mthsurf
    Line 8: column headers
    Line 9+: data (psi, m, 6 floats)
"""
function load_fortran_xclebsch(filepath::String)
    lines = readlines(filepath)

    # Parse mpert from header line 6: "      mstep =   1537      mpert =  34   mthsurf = 512"
    header_line = lines[6]
    m_mpert = match(r"mpert\s*=\s*(\d+)", header_line)
    mpert = parse(Int, m_mpert[1])

    # Find first data line (skip headers)
    data_start = 9
    data_lines = lines[data_start:end]
    filter!(l -> !isempty(strip(l)), data_lines)

    # Determine mlow from first data block
    first_m = parse(Int, split(data_lines[1])[2])
    mlow = first_m

    # Parse data
    ndata = length(data_lines)
    npsi_total = ndata ÷ mpert
    @assert ndata == npsi_total * mpert "Data count $ndata not divisible by mpert=$mpert"

    psi_vals = Float64[]
    clebsch_psi1_data = zeros(ComplexF64, npsi_total, mpert)
    clebsch_psi_data  = zeros(ComplexF64, npsi_total, mpert)
    clebsch_alpha_data = zeros(ComplexF64, npsi_total, mpert)

    idx = 0
    for i in 1:npsi_total
        for j in 1:mpert
            idx += 1
            vals = parse.(Float64, split(data_lines[idx]))
            psi = vals[1]
            m   = Int(vals[2])
            if j == 1
                push!(psi_vals, psi)
            end
            ipert = m - mlow + 1
            @assert 1 <= ipert <= mpert "Mode m=$m out of range [mlow=$mlow, mhigh=$(mlow+mpert-1)]"
            clebsch_psi1_data[i, ipert]  = complex(vals[3], vals[4])   # ∂ξ^ψ/∂ψ
            clebsch_psi_data[i, ipert]   = complex(vals[5], vals[6])   # ξ^ψ
            clebsch_alpha_data[i, ipert] = complex(vals[7], vals[8])   # ξ^α
        end
    end

    return psi_vals, clebsch_psi_data, clebsch_psi1_data, clebsch_alpha_data, mpert, mlow
end


_p(args...) = (println(stderr, args...); flush(stderr))
_pf(fmt, args...) = (print(stderr, Printf.format(Printf.Format(fmt), args...)); flush(stderr))

function run_benchmark()
    _p("=" ^ 70)
    _p("  DIIID Kinetic Benchmark: Julia KineticForces vs Fortran PENTRC")
    _p("=" ^ 70)

    # =========================================================================
    # Step 1: Run FFS via main() with force_termination=true
    # =========================================================================
    _p("\n--- Step 1: Equilibrium + ForceFreeStates (via main()) ---")
    t0 = time()
    result = GPE.main([BENCHMARK_DIR])
    equil  = result.equil
    intr   = result.intr
    ctrl   = result.ctrl

    _pf("  FFS completed in %.1f s\n", time() - t0)
    _pf("  mpert=%d, mlow=%d, mhigh=%d\n", intr.mpert, intr.mlow, intr.mhigh)

    # Build metric (needed for JBB deweighting)
    metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)

    # =========================================================================
    # Step 2: Load Fortran xclebsch data
    # =========================================================================
    _p("\n--- Step 2: Load Fortran xclebsch ---")
    xclebsch_path = joinpath(FORTRAN_DIR, "gpec_xclebsch_n1.out")
    if !isfile(xclebsch_path)
        error("Fortran xclebsch file not found: $xclebsch_path")
    end

    psi_grid_f, clebsch_psi, clebsch_psi1, clebsch_alpha, mpert_f, mlow_f =
        load_fortran_xclebsch(xclebsch_path)

    npsi_f = length(psi_grid_f)
    _pf("  Loaded %d ψ surfaces, mpert=%d, mlow=%d from Fortran xclebsch\n",
        npsi_f, mpert_f, mlow_f)
    _pf("  ψ range: [%.6f, %.6f]\n", psi_grid_f[1], psi_grid_f[end])

    if mpert_f != intr.mpert || mlow_f != intr.mlow
        @warn "Mode ranges differ: Fortran mpert=$mpert_f,mlow=$mlow_f vs Julia mpert=$(intr.mpert),mlow=$(intr.mlow)"
    end

    # =========================================================================
    # Step 3: Build PE state from Fortran data and run JBB deweighting
    # =========================================================================
    _p("\n--- Step 3: Build PE state (Fortran Clebsch → dbob_m/divx_m) ---")
    t1 = time()

    # Construct PerturbedEquilibriumState with Fortran Clebsch data
    # Fortran xclebsch stores ξ^α directly; PE convention is ξ^α/χ₁
    chi1 = 2π * equil.psio
    pe_state = PE.PerturbedEquilibriumState(;
        psi_grid = psi_grid_f,
        xi_modes = (
            clebsch_psi   = clebsch_psi,
            clebsch_psi1  = clebsch_psi1,
            clebsch_alpha = clebsch_alpha ./ chi1,  # Store as ξ^α/χ₁ per PE convention
        )
    )

    # KF configuration matching Fortran pentrc.in.
    kf_ctrl = KF.KineticForcesControl(;
        fgar_flag=true, tgar_flag=true, nn=1, nl=4,
        mi=2, zi=1, nutype="harmonic", f0type="maxwellian",
        nufac=1, psilims=[0.0, 1.0], kinetic_file="kinetic.dat",
        verbose=false)

    # Initialize KF internal state
    kf_intr = KF.KineticForcesInternal(equil; verbose=false)

    # Run set_perturbation_data! — builds dbob_m, divx_m, xs_m via JBB deweighting
    KF.set_perturbation_data!(kf_intr, pe_state, intr, equil, metric)

    _pf("  JBB deweighting completed in %.1f s\n", time() - t1)

    # Sanity checks
    if kf_intr.dbob_m === nothing || kf_intr.divx_m === nothing
        error("dbob_m or divx_m is nothing after set_perturbation_data!")
    end

    # Sample dbob_m at midplane
    psi_mid = 0.5
    dbob_sample = kf_intr.dbob_m(psi_mid)
    divx_sample = kf_intr.divx_m(psi_mid)
    _pf("  dbob_m(ψ=0.5) max|mode| = %.4e\n", maximum(abs.(dbob_sample)))
    _pf("  divx_m(ψ=0.5) max|mode| = %.4e\n", maximum(abs.(divx_sample)))

    # =========================================================================
    # Step 4: Load kinetic profiles and run torque integration
    # =========================================================================
    _p("\n--- Step 4: Torque integration ---")
    t2 = time()

    kinetic_profiles = Eq.load_kinetic_profiles(
        joinpath(BENCHMARK_DIR, kf_ctrl.kinetic_file);
        zi=kf_ctrl.zi, zimp=kf_ctrl.zimp, mi=kf_ctrl.mi, mimp=kf_ctrl.mimp)

    kf_state = KF.KineticForcesState()
    KF.compute_torque_all_methods!(kf_state, kf_intr, kf_ctrl, equil, kinetic_profiles)

    _pf("  Torque integration completed in %.1f s\n", time() - t2)

    # =========================================================================
    # Step 5: Compare results
    # =========================================================================
    _p("\n" * "=" ^ 70)
    _p("  Results Comparison")
    _p("=" ^ 70)

    results = Dict{String,NamedTuple}()

    for (method_key, ref_T, ref_dW) in [
        ("fgar", REF.T_total_fgar, REF.dW_total_fgar),
        ("tgar", REF.T_total_tgar, REF.dW_total_tgar)]

        if haskey(kf_state.method_results, method_key)
            mr = kf_state.method_results[method_key]
            T_julia  = real(mr.total_torque)
            dW_julia = real(mr.total_energy)
            T_err    = abs(T_julia - ref_T) / abs(ref_T) * 100
            dW_err   = abs(dW_julia - ref_dW) / abs(ref_dW) * 100

            results[method_key] = (; T_julia, dW_julia, T_err, dW_err)

            _pf("\n  %-5s Torque:  Julia = %12.6f  Fortran = %12.6f  err = %6.2f%%\n",
                uppercase(method_key), T_julia, ref_T, T_err)
            _pf("  %-5s Energy:  Julia = %12.6f  Fortran = %12.6f  err = %6.2f%%\n",
                uppercase(method_key), dW_julia, ref_dW, dW_err)
        else
            _pf("\n  %-5s: NOT RUN\n", uppercase(method_key))
        end
    end

    _p("\n" * "=" ^ 70)

    # Check pass/fail
    all_pass = true
    for (method_key, r) in results
        if r.T_err > 5.0 || r.dW_err > 5.0
            @warn "$(uppercase(method_key)) exceeds 5% threshold" T_err=r.T_err dW_err=r.dW_err
            all_pass = false
        end
    end

    if all_pass && !isempty(results)
        _p("  All methods within 5% of Fortran reference")
    elseif isempty(results)
        _p("  No methods produced results")
    else
        _p("  Some methods exceed 5% threshold")
    end

    return results
end

# Run
results = run_benchmark()
