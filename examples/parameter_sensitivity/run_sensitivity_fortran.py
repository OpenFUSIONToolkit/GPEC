#!/usr/bin/env python3
"""
Fortran STRIDE parameter sensitivity study.

Matches the Julia run_sensitivity.jl parameter sweeps exactly.
Runs Fortran STRIDE on 147131 (n=1,2) and LAR ε=0.4072 (n=1,2),
sweeping one parameter at a time from a common baseline.

Outputs: sensitivity_results_fortran.h5 (same structure as Julia's HDF5)

Usage:
    python3 run_sensitivity_fortran.py
    python3 run_sensitivity_fortran.py --category ode    # single category
    python3 run_sensitivity_fortran.py --case 147131_n1  # single case
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STRIDE_BIN = Path.home() / 'Desktop' / 'plasma' / 'GPEC' / 'stride' / 'stride'
REF_VAC_IN = Path.home() / 'Desktop' / 'plasma' / 'GPEC' / 'docs' / 'examples' / 'DIIID_ideal_example' / 'vac.in'
OUTPUT_H5 = SCRIPT_DIR / 'sensitivity_results_fortran.h5'

# ── Equilibria ──────────────────────────────────────────────────────────

CASES = {
    '147131_n1': {
        'geqdsk': str(Path.home() / 'Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT'),
        'n': 1, 'psilow': 1e-4, 'psihigh': 0.993,
    },
    '147131_n2': {
        'geqdsk': str(Path.home() / 'Desktop/plasma/GPEC/docs/examples/DIIID_ideal_example/g147131.02300_DIIID_KEFIT'),
        'n': 2, 'psilow': 1e-4, 'psihigh': 0.993,
    },
    'LAR_0.4072_n1': {
        'geqdsk': str(Path(__file__).resolve().parent.parent / 'LAR_epsilon_scan/equilibria/TJ_epsilon_scan_0.4072.geqdsk'),
        'n': 1, 'psilow': 0.01, 'psihigh': 0.995,
    },
    'LAR_0.4072_n2': {
        'geqdsk': str(Path(__file__).resolve().parent.parent / 'LAR_epsilon_scan/equilibria/TJ_epsilon_scan_0.4072.geqdsk'),
        'n': 2, 'psilow': 0.01, 'psihigh': 0.995,
    },
}

# ── Baseline config (Fortran namelist values) ──────────────────────────

def baseline_config(case):
    c = CASES[case]
    return {
        'eq_filename': os.path.basename(c['geqdsk']),
        'geqdsk_path': c['geqdsk'],
        'n': c['n'],
        'psilow': c['psilow'],
        'psihigh': c['psihigh'],
        'mpsi': 128,
        'mtheta': 256,
        'etol': 1e-7,
        'sas_flag': True,   # set_psilim_via_dmlim
        'dmlim': 0.2,
        'qlow': 1.02,
        'qhigh': 1e3,
        'sing_start': 0,
        'delta_mlow': 8,
        'delta_mhigh': 8,
        'delta_mband': 0,
        'mthvac': 512,
        'tol_nr': 1e-12,    # maps to eulerlagrange_tolerance
        'tol_r': 1e-12,
        'singfac_min': 1e-4,
        'ucrit': 1e4,
        'sing_order': 6,
        'wall_a': 20,
        'force_wv_symmetry': True,
    }

# ── Parameter sweeps (matching Julia exactly) ──────────────────────────

SWEEPS = [
    # (category, fortran_param, values, label)
    # ODE
    ('ode', 'tol_nr+tol_r', [1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14], 'el_tol'),
    ('ode', 'singfac_min', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 'singfac_min'),
    ('ode', 'sing_order', [2, 4, 6, 8, 10, 12], 'sing_order'),
    ('ode', 'ucrit', [1e2, 1e3, 1e4, 1e5, 1e6], 'ucrit'),
    # Mode
    ('mode', 'delta_m', [2, 4, 6, 8, 10, 12, 16], 'delta_m'),
    ('mode', 'delta_mband', [0, 2, 4, 8], 'delta_mband'),
    # Grid
    ('grid', 'mpsi', [32, 64, 128, 256, 512], 'mpsi'),
    ('grid', 'mtheta', [64, 128, 256, 512], 'mtheta'),
    ('grid', 'etol', [1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'etol'),
    # Domain
    ('domain', 'psilow', [1e-2, 1e-3, 1e-4, 1e-5], 'psilow'),
    ('domain', 'psihigh', [0.98, 0.99, 0.993, 0.995, 0.999, 1.0], 'psihigh'),
    ('domain', 'qhigh', [3.6, 4.0, 5.0, 10.0, 100.0, 1e3], 'qhigh'),
    ('domain', 'dmlim', [0.1, 0.2, 0.3, 0.5], 'dmlim'),
    # Vacuum
    ('vacuum', 'mthvac', [128, 256, 512, 960, 1920], 'mthvac'),
    ('vacuum', 'wall_a', [1.5, 2.0, 5.0, 10.0, 20.0, 100.0], 'wall_a'),
]


def write_equil_in(path, cfg):
    with open(path, 'w') as f:
        f.write(f"""&EQUIL_CONTROL
    eq_type="efit"
    eq_filename = "{cfg['eq_filename']}"
    jac_type="hamada"
    power_bp=0
    power_b=0
    power_r=0
    grid_type="ldp"
    psilow={cfg['psilow']:.1e}
    psihigh={cfg['psihigh']:.4f}
    mpsi={cfg['mpsi']}
    mtheta={cfg['mtheta']}
    newq0=0
    etol={cfg['etol']:.1e}
    use_classic_splines=f
/
&EQUIL_OUTPUT
    bin_eq_2d=t
/
""")


def write_stride_in(path, cfg):
    sas = 't' if cfg['sas_flag'] else 'f'
    with open(path, 'w') as f:
        f.write(f"""&stride_control
    bal_flag=f
    mat_flag=t
    ode_flag=t
    vac_flag=t
    mer_flag=t
    sas_flag={sas}
    dmlim={cfg['dmlim']}
    qlow={cfg['qlow']}
    qhigh={cfg['qhigh']}
    sing_start={cfg['sing_start']}
    nn={cfg['n']}
    delta_mlow={cfg['delta_mlow']}
    delta_mhigh={cfg['delta_mhigh']}
    delta_mband={cfg['delta_mband']}
    mthvac={cfg['mthvac']}
    thmax0=1
    tol_nr={cfg['tol_nr']:.1e}
    tol_r={cfg['tol_r']:.1e}
    crossover=1e-2
    singfac_min={cfg['singfac_min']:.1e}
    ucrit={cfg['ucrit']:.1e}
    sing_order={cfg['sing_order']}
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


def write_vac_in(path, cfg):
    """Write vac.in with wall_a parameter."""
    # Copy reference and patch the wall distance
    if REF_VAC_IN.exists():
        txt = REF_VAC_IN.read_text()
        # Replace the 'a = 20' line in &SHAPE section
        import re
        txt = re.sub(r'(a\s*=\s*)\d+', f'\\g<1>{int(cfg["wall_a"])}', txt, count=1)
        Path(path).write_text(txt)
    else:
        Path(path).write_text(f"""&MODES
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
   a = {int(cfg['wall_a'])}
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


def run_single(case_name, cfg, timeout=300):
    """Run Fortran STRIDE with given config, return dict of results or None."""
    c = CASES[case_name]
    run_dir = tempfile.mkdtemp(prefix=f'fsens_{case_name}_')
    try:
        shutil.copy2(c['geqdsk'], os.path.join(run_dir, cfg['eq_filename']))
        write_equil_in(os.path.join(run_dir, 'equil.in'), cfg)
        write_stride_in(os.path.join(run_dir, 'stride.in'), cfg)
        write_vac_in(os.path.join(run_dir, 'vac.in'), cfg)

        t0 = time.time()
        result = subprocess.run([str(STRIDE_BIN)], cwd=run_dir,
                                capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0

        if result.returncode != 0:
            return None

        nc_path = os.path.join(run_dir, f'stride_output_n{cfg["n"]}.nc')
        if not os.path.isfile(nc_path):
            return None

        import netCDF4
        ds = netCDF4.Dataset(nc_path)
        q_r = np.array(ds.variables['q_rational'][:]).flatten()
        msing = len(q_r)
        dp_raw = np.array(ds.variables['Delta_prime'][:])
        dp_complex = dp_raw[0] + 1j * dp_raw[1]  # (msing, msing)

        dW_p = float(ds.getncattr('plasma1'))
        dW_v = float(ds.getncattr('vacuum1'))
        dW_t = float(ds.getncattr('total1'))
        ds.close()

        return {
            'dp_matrix': dp_complex,
            'dW_plasma': dW_p,
            'dW_vacuum': dW_v,
            'dW_total': dW_t,
            'msing': msing,
            'runtime': elapsed,
        }
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f'    Error: {e}')
        return None
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def store_result(h5_path, group_path, result):
    with h5py.File(str(h5_path), 'a') as f:
        if group_path in f:
            del f[group_path]
        g = f.create_group(group_path)
        g.create_dataset('dp_matrix', data=result['dp_matrix'])
        g.attrs['dW_plasma'] = result['dW_plasma']
        g.attrs['dW_vacuum'] = result['dW_vacuum']
        g.attrs['dW_total'] = result['dW_total']
        g.attrs['msing'] = result['msing']
        g.attrs['runtime'] = result['runtime']


def value_label(v):
    if isinstance(v, bool):
        return str(v).lower()
    elif isinstance(v, float):
        if v >= 1 and v == int(v):
            return f'{v:.1f}'
        elif abs(v) < 0.01 or abs(v) >= 1000:
            return f'{v:.0e}'
        else:
            return f'{v:.2e}'
    else:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', help='Run only this category')
    parser.add_argument('--case', help='Run only this case')
    args = parser.parse_args()

    if not STRIDE_BIN.exists():
        print(f'ERROR: STRIDE binary not found at {STRIDE_BIN}')
        sys.exit(1)

    cases = list(CASES.keys())
    if args.case:
        cases = [c for c in cases if c == args.case]

    sweeps = SWEEPS
    if args.category:
        sweeps = [s for s in sweeps if s[0] == args.category]

    # Count total runs
    total = 0
    for case in cases:
        total += 1  # baseline
        for cat, param, values, label in sweeps:
            for v in values:
                total += 1
    print(f'Total runs: {total} ({len(cases)} cases × {len(sweeps)} sweeps)')

    run_idx = 0
    for case in cases:
        bl_cfg = baseline_config(case)

        # Baseline
        run_idx += 1
        print(f'[{run_idx}/{total}] {case} baseline')
        result = run_single(case, bl_cfg)
        if result:
            store_result(OUTPUT_H5, f'{case}/baseline', result)
            dp = result['dp_matrix']
            print(f'  dp[0,0]={dp[0,0].real:+.4f}{dp[0,0].imag:+.4f}i, msing={result["msing"]}, t={result["runtime"]:.1f}s')
        else:
            print(f'  FAILED')

        # Sweeps
        for cat, param, values, label in sweeps:
            for v in values:
                cfg = baseline_config(case)

                # Apply parameter
                if param == 'tol_nr+tol_r':
                    cfg['tol_nr'] = v
                    cfg['tol_r'] = v
                elif param == 'delta_m':
                    cfg['delta_mlow'] = v
                    cfg['delta_mhigh'] = v
                elif param == 'wall_a':
                    cfg['wall_a'] = v
                else:
                    if param not in cfg:
                        continue
                    cfg[param] = v

                # Skip if same as baseline
                bl = baseline_config(case)
                if param == 'tol_nr+tol_r':
                    is_baseline = (bl['tol_nr'] == v)
                elif param == 'delta_m':
                    is_baseline = (bl['delta_mlow'] == v)
                elif param == 'wall_a':
                    is_baseline = (bl['wall_a'] == v)
                else:
                    is_baseline = (bl.get(param) == v)

                run_idx += 1
                vl = value_label(v)
                if is_baseline:
                    print(f'[{run_idx}/{total}] {case} {label}={vl} (=baseline, skip)')
                    continue

                print(f'[{run_idx}/{total}] {case} {label}={vl}', end='', flush=True)
                result = run_single(case, cfg)
                if result:
                    store_result(OUTPUT_H5, f'{case}/{label}/{vl}', result)
                    dp = result['dp_matrix']
                    print(f'  dp[0,0]={dp[0,0].real:+.4f}, t={result["runtime"]:.1f}s')
                else:
                    print(f'  FAILED')

    print(f'\nDone. Results in {OUTPUT_H5}')


if __name__ == '__main__':
    main()
