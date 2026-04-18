#!/usr/bin/env python3
"""
Run Fortran STRIDE on DIII-D equilibria and extract Δ' results.

Creates a temporary working directory for each (shot, n), writes input files
matching the Julia GPEC parameters, runs STRIDE, and parses the NetCDF output.

Usage:
    python3 run_fortran_stride.py              # All shots
    python3 run_fortran_stride.py --shots 147131  # Single shot
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
STRIDE_BIN = Path.home() / 'Desktop' / 'plasma' / 'GPEC' / 'stride' / 'stride'

# ── Shot definitions ────────────────────────────────────────────────────────

SHOTS = {
    '147131': {
        'geqdsk': str(Path.home() / 'Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT'),
        'n_values': [1, 2],
        'psilow': 1e-4,
        'psihigh': 0.993,
        'mpsi': 128,
        'mtheta': 256,
        'description': 'DIII-D KEFIT H-mode',
    },
    '204441': {
        'geqdsk': str(Path.home() / 'Desktop/plasma/IDA_run/MSE_constraint/test_data/204441_4400_withEr/g204441.04409.geqdsk'),
        'n_values': [1, 2],
        'psilow': 1e-4,
        'psihigh': 0.993,
        'mpsi': 128,
        'mtheta': 256,
        'description': 'DIII-D IDA MSE w/ Er',
    },
    '153833': {
        'geqdsk': str(Path.home() / 'Desktop/plasma/IDA_run/IDA_workflow_examples/153833_3450/g153833.3450_results_baseline.geqdsk'),
        'n_values': [1, 2],
        'psilow': 1e-4,
        'psihigh': 0.993,
        'mpsi': 128,
        'mtheta': 256,
        'description': 'DIII-D IDA workflow',
    },
    # 153072: Reversed shear (qmin=1.86) — excluded, multi-resonance not yet supported in Julia
}


def write_equil_in(path, shot_cfg, geqdsk_basename):
    """Write equil.in namelist matching Julia GPEC parameters."""
    with open(path, 'w') as f:
        f.write(f"""&EQUIL_CONTROL
    eq_type="efit"
    eq_filename = "{geqdsk_basename}"
    jac_type="hamada"
    power_bp=0
    power_b=0
    power_r=0
    grid_type="ldp"
    psilow={shot_cfg['psilow']:.1e}
    psihigh={shot_cfg['psihigh']:.3f}
    mpsi={shot_cfg['mpsi']}
    mtheta={shot_cfg['mtheta']}
    newq0=0
    use_classic_splines=f
/
&EQUIL_OUTPUT
    gse_flag=f
    out_eq_1d=f
    bin_eq_1d=f
    out_eq_2d=f
    bin_eq_2d=t
    out_2d=f
    bin_2d=f
    dump_flag=f
/
""")


def write_stride_in(path, n):
    """Write stride.in namelist matching Julia GPEC parameters exactly.

    Uses the full namelist format from the DIIID_ideal_example to ensure
    compatibility with the Fortran namelist parser.
    """
    with open(path, 'w') as f:
        f.write(f"""&stride_control
    bal_flag=f
    mat_flag=t
    ode_flag=t
    vac_flag=t
    mer_flag=t

    sas_flag=t
    dmlim=0.2
    qlow=1.02
    qhigh=1e3
    sing_start=0

    nn={n}
    delta_mlow=8
    delta_mhigh=8
    delta_mband=0
    mthvac=512
    thmax0=1

    tol_nr=1e-8
    tol_r=1e-8
    crossover=1e-2
    singfac_min=1e-4
    ucrit=1e4
    sing_order=6

    use_classic_splines=f
    use_notaknot_splines=f
/

&stride_output
    out_ahg2msc=f
    crit_break=t
    ahb_flag=f
    msol_ahb=1
    mthsurf0=1
    bin_euler=f
    euler_stride=1
    out_bal1=f
    bin_bal1=f
    out_bal2=f
    bin_bal2=f
    netcdf_out=t
/

&stride_params
    nThreads=1
    fourfit_metric_parallel=f
    vac_parallel=f
    nIntervalsTot=33
    grid_packing="singularities"
    axis_mid_pt_skew=12.0
    asymp_at_sing=t
    kill_big_soln_for_ideal_dW=f
    calc_delta_prime=t
    calc_dp_with_vac=t
    big_soln_err_tol=1e-7
    integrate_riccati=f
    riccati_bounce=t
    riccati_match_hamiltonian_evals=f
    verbose_riccati_output=f
    ric_dt=1e-8
    ric_tol=1e-6
    verbose_performance_output=f
/
""")


def write_vac_in(path):
    """Write vac.in matching Julia conformal wall a=20.

    Copies the reference vac.in from the DIIID_ideal_example to avoid
    namelist parsing issues with the Fortran reader.
    """
    ref_vac = Path.home() / 'Desktop' / 'plasma' / 'GPEC' / 'docs' / 'examples' / 'DIIID_ideal_example' / 'vac.in'
    if ref_vac.exists():
        shutil.copy2(str(ref_vac), path)
        return
    # Fallback
    with open(path, 'w') as f:
        f.write("""&MODES
   mth = 480
   xiin(1:9) = 0 0 0 0 0 0 0 1 0
   lsymz = .TRUE.
   leqarcw = 1
   lzio = 0
   lgato = 0
   lrgato = 0
/
&DEBUGS
   checkd = .FALSE.
   check1 = .FALSE.
   check2 = .FALSE.
   checke = .FALSE.
   checks = .FALSE.
   wall = .FALSE.
   lkplt = 0
   verbose_timer_output = f
/
&VACDAT
   ishape = 6
   aw = 0.05
   bw = 1.5
   cw = 0
   dw = 0.5
   tw = 0.05
   nsing = 500
   epsq = 1e-05
   noutv = 37
   idgt = 6
   delg = 15.01
   delfac = 0.001
   cn0 = 1
/
&SHAPE
   ipshp = 0
   xpl = 100
   apl = 1
   a = 20
   b = 170
   bpl = 1
   dpl = 0
   r = 1
/
&DIAGNS
   lkdis = .FALSE.
/
&SPRK
/
""")


def run_single(shot_name, shot_cfg, n, verbose=True):
    """Run Fortran STRIDE for a single (shot, n)."""
    geqdsk = shot_cfg['geqdsk']
    if not os.path.isfile(geqdsk):
        print(f'  SKIP: geqdsk not found: {geqdsk}')
        return None

    # Create temp working directory
    run_dir = tempfile.mkdtemp(prefix=f'stride_{shot_name}_n{n}_')

    try:
        # Copy geqdsk
        geqdsk_basename = os.path.basename(geqdsk)
        shutil.copy2(geqdsk, os.path.join(run_dir, geqdsk_basename))

        # Write input files
        write_equil_in(os.path.join(run_dir, 'equil.in'), shot_cfg, geqdsk_basename)
        write_stride_in(os.path.join(run_dir, 'stride.in'), n)
        write_vac_in(os.path.join(run_dir, 'vac.in'))

        if verbose:
            print(f'  Running STRIDE: shot={shot_name} n={n} dir={run_dir}')

        # Run STRIDE
        result = subprocess.run(
            [str(STRIDE_BIN)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f'  STRIDE failed (rc={result.returncode})')
            # Print last lines of output
            for line in result.stdout.split('\n')[-10:]:
                print(f'    {line}')
            return None

        # Parse NetCDF output
        nc_path = os.path.join(run_dir, f'stride_output_n{n}.nc')
        if not os.path.isfile(nc_path):
            print(f'  No NetCDF output found')
            return None

        return parse_stride_nc(nc_path, shot_name, n)

    except subprocess.TimeoutExpired:
        print(f'  STRIDE timed out after 300s')
        return None
    except Exception as e:
        print(f'  Error: {e}')
        return None
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def parse_stride_nc(nc_path, shot_name, n):
    """Parse STRIDE NetCDF output and extract Δ' matrix diagonal."""
    import netCDF4
    ds = netCDF4.Dataset(nc_path)

    q_rational = np.array(ds.variables['q_rational'][:]).flatten()
    # Remove masked/fill values
    q_rational = q_rational[~np.ma.getmaskarray(q_rational)] if hasattr(q_rational, 'mask') else q_rational
    msing = len(q_rational)
    psi_rational = np.array(ds.variables['psi_n_rational'][:]).flatten()[:msing]
    dp_raw = np.array(ds.variables['Delta_prime'][:])
    # dp_raw shape: (2, msing, msing) = (real/imag, msing, msing)
    dp_complex = dp_raw[0] + 1j * dp_raw[1]

    dW_p = float(ds.getncattr('plasma1'))
    dW_v = float(ds.getncattr('vacuum1'))
    dW_t = float(ds.getncattr('total1'))
    qlim = float(ds.getncattr('qlim')) if 'qlim' in ds.ncattrs() else float('nan')
    q0 = float(ds.getncattr('q0')) if 'q0' in ds.ncattrs() else float('nan')

    surfaces = []
    for i in range(msing):
        m = int(round(q_rational[i] * n))
        surfaces.append({
            'm': m,
            'n': n,
            'q': float(q_rational[i]),
            'psi': float(psi_rational[i]),
            'dp_real': float(dp_complex[i, i].real),
            'dp_imag': float(dp_complex[i, i].imag),
        })

    ds.close()

    return {
        'shot': shot_name,
        'n': n,
        'msing': msing,
        'surfaces': surfaces,
        'dW_plasma': dW_p,
        'dW_vacuum': dW_v,
        'dW_total': dW_t,
        'qlim': qlim,
        'q0': q0,
    }


def save_results(all_results, output_path):
    """Save all Fortran results to JSON for Julia comparison script to load."""
    serializable = []
    for r in all_results:
        if r is not None:
            serializable.append(r)
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\nFortran results saved to {output_path}')


def print_results(all_results):
    """Print formatted results table."""
    print('\n' + '=' * 90)
    print('  Fortran STRIDE Results')
    print('=' * 90)
    for r in all_results:
        if r is None:
            continue
        print(f"\n  Shot {r['shot']} n={r['n']}: msing={r['msing']}, "
              f"dW_p={r['dW_plasma']:.4f}, dW_v={r['dW_vacuum']:.4f}, dW_t={r['dW_total']:.4f}, "
              f"qlim={r['qlim']:.2f}")
        for s in r['surfaces']:
            print(f"    {s['m']}/{s['n']}  q={s['q']:.4f}  psi={s['psi']:.6f}  "
                  f"dp={s['dp_real']:+.4f} {s['dp_imag']:+.4f}i")


def main():
    parser = argparse.ArgumentParser(description='Run Fortran STRIDE on DIII-D shots')
    parser.add_argument('--shots', help='Comma-separated shot list')
    args = parser.parse_args()

    if not STRIDE_BIN.exists():
        print(f'ERROR: STRIDE binary not found at {STRIDE_BIN}')
        print(f'  Build with: cd ~/Desktop/plasma/GPEC/install && make stride')
        sys.exit(1)

    shot_filter = args.shots.split(',') if args.shots else None
    shots_to_run = {k: v for k, v in SHOTS.items()
                    if shot_filter is None or k in shot_filter}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    total = sum(len(v['n_values']) for v in shots_to_run.values())
    idx = 0
    for shot_name, shot_cfg in shots_to_run.items():
        for n in shot_cfg['n_values']:
            idx += 1
            print(f'[{idx}/{total}] {shot_name} n={n}')
            result = run_single(shot_name, shot_cfg, n)
            all_results.append(result)

    print_results([r for r in all_results if r is not None])
    save_results([r for r in all_results if r is not None],
                 OUTPUT_DIR / 'fortran_stride_results.json')


if __name__ == '__main__':
    main()
