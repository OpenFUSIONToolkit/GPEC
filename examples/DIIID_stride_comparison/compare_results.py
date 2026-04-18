#!/usr/bin/env python3
"""
Compare Julia GPEC vs Fortran STRIDE Δ' results for DIII-D equilibria.

Reads:
  - outputs/fortran_stride_results.json (from run_fortran_stride.py)
  - outputs/<shot>_n<N>/gpec.h5 (from run_comparison.jl)

Usage:
    python3 compare_results.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'

# Try h5py for Julia outputs
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("WARNING: h5py not installed, cannot read Julia HDF5 outputs")


def load_fortran_results():
    """Load Fortran STRIDE results from JSON."""
    path = OUTPUT_DIR / 'fortran_stride_results.json'
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_fortran_stride.py first.")
        return {}
    with open(path) as f:
        results = json.load(f)
    # Index by (shot, n)
    return {(r['shot'], r['n']): r for r in results}


def load_julia_results():
    """Load Julia GPEC results from HDF5 output files."""
    if not HAS_H5PY:
        return {}

    results = {}
    for d in OUTPUT_DIR.iterdir():
        if not d.is_dir():
            continue
        m = None
        parts = d.name.rsplit('_n', 1)
        if len(parts) == 2:
            shot = parts[0]
            try:
                n = int(parts[1])
            except ValueError:
                continue
        else:
            continue

        h5_path = d / 'gpec.h5'
        if not h5_path.exists():
            continue

        try:
            with h5py.File(str(h5_path), 'r') as f:
                msing = int(f['singular/msing'][()])
                m_sing = f['singular/m'][:]  # shape (npert, msing) or (msing, max_modes)
                q_sing = f['singular/q'][:]
                psi_sing = f['singular/psi'][:]

                dp_mat = f['singular/delta_prime_matrix'][:] if 'singular/delta_prime_matrix' in f else None

                ep = f['vacuum/ep'][:]
                ev = f['vacuum/ev'][:]
                et = f['vacuum/et'][:]
                qlim = float(f['info/qlim'][()]) if 'info/qlim' in f else float('nan')
                q0 = float(f['equil/q0'][()])

                surfaces = []
                for s in range(msing):
                    # m array may be (npert, msing) or (msing, max_modes)
                    if m_sing.ndim == 2:
                        if m_sing.shape[0] == msing:
                            m_val = int(m_sing[s, 0])
                        else:
                            m_val = int(m_sing[0, s])
                    else:
                        m_val = int(m_sing[s])
                    if dp_mat is not None and s < dp_mat.shape[0]:
                        dp_val = dp_mat[s, s]
                    else:
                        dp_val = float('nan') + 1j * float('nan')
                    surfaces.append({
                        'm': m_val, 'n': n,
                        'q': float(q_sing[s]),
                        'psi': float(psi_sing[s]),
                        'dp_real': float(dp_val.real),
                        'dp_imag': float(dp_val.imag),
                    })

                results[(shot, n)] = {
                    'shot': shot, 'n': n, 'msing': msing,
                    'surfaces': surfaces,
                    'dW_plasma': float(ep[0].real),
                    'dW_vacuum': float(ev[0].real),
                    'dW_total': float(et[0].real),
                    'qlim': qlim, 'q0': q0,
                }
        except Exception as e:
            print(f"  Warning: could not read {h5_path}: {e}")

    return results


def print_comparison(fortran, julia):
    """Print side-by-side comparison table."""
    all_keys = sorted(set(list(fortran.keys()) + list(julia.keys())))

    print()
    print('=' * 130)
    print('  DIII-D STRIDE Comparison: Fortran vs Julia GPEC')
    print('=' * 130)

    shots = sorted(set(k[0] for k in all_keys))
    for shot in shots:
        print(f'\n{"─" * 130}')
        print(f'  Shot: {shot}')
        print(f'{"─" * 130}')

        ns = sorted(set(k[1] for k in all_keys if k[0] == shot))
        for n in ns:
            key = (shot, n)
            fr = fortran.get(key)
            jr = julia.get(key)

            if fr is None and jr is None:
                continue

            # Header with dW comparison
            print(f'\n  n={n}:')
            if fr:
                print(f'    Fortran: dW_p={fr["dW_plasma"]:+.4f}  dW_v={fr["dW_vacuum"]:+.4f}  '
                      f'dW_t={fr["dW_total"]:+.4f}  qlim={fr["qlim"]:.2f}  msing={fr["msing"]}')
            if jr:
                print(f'    Julia:   dW_p={jr["dW_plasma"]:+.4f}  dW_v={jr["dW_vacuum"]:+.4f}  '
                      f'dW_t={jr["dW_total"]:+.4f}  qlim={jr["qlim"]:.2f}  msing={jr["msing"]}')
            if fr and jr:
                err_dW = abs(jr['dW_total'] - fr['dW_total'])
                pct_dW = 100 * err_dW / max(abs(fr['dW_total']), 1e-30)
                print(f'    Δ dW_t = {jr["dW_total"] - fr["dW_total"]:+.4f} ({pct_dW:.1f}%)')

            # Surface-by-surface comparison
            print(f'    {"m/n":>6s}  {"q":>6s}  {"ψ":>8s}  │ {"Fortran Re(Δ′)":>14s} {"Julia Re(Δ′)":>14s} {"err%":>8s} │ '
                  f'{"Fortran Im(Δ′)":>14s} {"Julia Im(Δ′)":>14s} {"err%":>8s} │ {"│Im│/│Re│ F":>10s} {"│Im│/│Re│ J":>10s}')
            print(f'    {"─" * 120}')

            # Build surface lookup
            f_surfs = {s['m']: s for s in (fr['surfaces'] if fr else [])}
            j_surfs = {s['m']: s for s in (jr['surfaces'] if jr else [])}
            all_ms = sorted(set(list(f_surfs.keys()) + list(j_surfs.keys())))

            for m in all_ms:
                fs = f_surfs.get(m)
                js = j_surfs.get(m)
                m_n = f'{m}/{n}'
                q = fs['q'] if fs else (js['q'] if js else 0)
                psi = fs['psi'] if fs else (js['psi'] if js else 0)

                f_re = fs['dp_real'] if fs else float('nan')
                f_im = fs['dp_imag'] if fs else float('nan')
                j_re = js['dp_real'] if js else float('nan')
                j_im = js['dp_imag'] if js else float('nan')

                # Errors
                if np.isfinite(f_re) and np.isfinite(j_re) and abs(f_re) > 1e-10:
                    err_re = f'{100 * abs(j_re - f_re) / abs(f_re):.1f}%'
                else:
                    err_re = '--'
                if np.isfinite(f_im) and np.isfinite(j_im) and abs(f_im) > 1e-10:
                    err_im = f'{100 * abs(j_im - f_im) / abs(f_im):.1f}%'
                else:
                    err_im = '--'

                # |Im|/|Re| ratio — diagnostic for numerical artifacts
                f_ratio = abs(f_im / f_re) if np.isfinite(f_re) and abs(f_re) > 1e-10 and np.isfinite(f_im) else float('nan')
                j_ratio = abs(j_im / j_re) if np.isfinite(j_re) and abs(j_re) > 1e-10 and np.isfinite(j_im) else float('nan')
                f_ratio_s = f'{f_ratio:.3f}' if np.isfinite(f_ratio) else '--'
                j_ratio_s = f'{j_ratio:.3f}' if np.isfinite(j_ratio) else '--'

                f_re_s = f'{f_re:+14.4f}' if np.isfinite(f_re) else f'{"--":>14s}'
                f_im_s = f'{f_im:+14.4f}' if np.isfinite(f_im) else f'{"--":>14s}'
                j_re_s = f'{j_re:+14.4f}' if np.isfinite(j_re) else f'{"--":>14s}'
                j_im_s = f'{j_im:+14.4f}' if np.isfinite(j_im) else f'{"--":>14s}'

                print(f'    {m_n:>6s}  {q:6.3f}  {psi:8.5f}  │ {f_re_s} {j_re_s} {err_re:>8s} │ '
                      f'{f_im_s} {j_im_s} {err_im:>8s} │ {f_ratio_s:>10s} {j_ratio_s:>10s}')

    print(f'\n{"=" * 130}')

    # Summary: Im/Re ratio analysis
    print('\n  Im(Δ′) / Re(Δ′) Ratio Analysis:')
    print('  (Values > ~0.1 suggest numerical artifacts; < 0.01 = clean)')
    for shot in shots:
        ns = sorted(set(k[1] for k in all_keys if k[0] == shot))
        for n in ns:
            key = (shot, n)
            fr = fortran.get(key)
            jr = julia.get(key)
            if not fr or not jr:
                continue
            f_surfs = {s['m']: s for s in fr['surfaces']}
            j_surfs = {s['m']: s for s in jr['surfaces']}
            common_ms = sorted(set(f_surfs.keys()) & set(j_surfs.keys()))
            if not common_ms:
                continue
            f_ratios = []
            j_ratios = []
            for m in common_ms:
                fs = f_surfs[m]; js = j_surfs[m]
                if abs(fs['dp_real']) > 1e-5 and np.isfinite(fs['dp_imag']):
                    f_ratios.append(abs(fs['dp_imag'] / fs['dp_real']))
                if abs(js['dp_real']) > 1e-5 and np.isfinite(js['dp_imag']):
                    j_ratios.append(abs(js['dp_imag'] / js['dp_real']))
            if f_ratios and j_ratios:
                print(f'    {shot} n={n}: Fortran median |Im/Re| = {np.median(f_ratios):.4f}, '
                      f'Julia median |Im/Re| = {np.median(j_ratios):.4f}  '
                      f'{"✓ Julia cleaner" if np.median(j_ratios) < np.median(f_ratios) else "Fortran cleaner"}')


def save_csv(fortran, julia):
    """Save comparison CSV."""
    path = OUTPUT_DIR / 'diiid_stride_comparison.csv'
    with open(path, 'w') as f:
        f.write('shot,n,m,q,psi,'
                'fortran_dp_real,fortran_dp_imag,julia_dp_real,julia_dp_imag,'
                'err_re_pct,err_im_pct,fortran_im_re_ratio,julia_im_re_ratio,'
                'fortran_dW_t,julia_dW_t\n')

        all_keys = sorted(set(list(fortran.keys()) + list(julia.keys())))
        for key in all_keys:
            fr = fortran.get(key)
            jr = julia.get(key)
            if not fr and not jr:
                continue
            shot, n = key
            f_surfs = {s['m']: s for s in (fr['surfaces'] if fr else [])}
            j_surfs = {s['m']: s for s in (jr['surfaces'] if jr else [])}
            all_ms = sorted(set(list(f_surfs.keys()) + list(j_surfs.keys())))
            for m in all_ms:
                fs = f_surfs.get(m, {'dp_real': float('nan'), 'dp_imag': float('nan'), 'q': 0, 'psi': 0})
                js = j_surfs.get(m, {'dp_real': float('nan'), 'dp_imag': float('nan'), 'q': 0, 'psi': 0})
                q = fs.get('q', 0) or js.get('q', 0)
                psi = fs.get('psi', 0) or js.get('psi', 0)
                err_re = 100 * abs(js['dp_real'] - fs['dp_real']) / max(abs(fs['dp_real']), 1e-30) if np.isfinite(fs['dp_real']) and np.isfinite(js['dp_real']) else float('nan')
                err_im = 100 * abs(js['dp_imag'] - fs['dp_imag']) / max(abs(fs['dp_imag']), 1e-30) if np.isfinite(fs['dp_imag']) and np.isfinite(js['dp_imag']) and abs(fs['dp_imag']) > 1e-10 else float('nan')
                f_ratio = abs(fs['dp_imag'] / fs['dp_real']) if np.isfinite(fs['dp_real']) and abs(fs['dp_real']) > 1e-10 and np.isfinite(fs['dp_imag']) else float('nan')
                j_ratio = abs(js['dp_imag'] / js['dp_real']) if np.isfinite(js['dp_real']) and abs(js['dp_real']) > 1e-10 and np.isfinite(js['dp_imag']) else float('nan')
                f_dW = fr['dW_total'] if fr else float('nan')
                j_dW = jr['dW_total'] if jr else float('nan')
                f.write(f'{shot},{n},{m},{q:.6f},{psi:.8f},'
                        f'{fs["dp_real"]:.8e},{fs["dp_imag"]:.8e},{js["dp_real"]:.8e},{js["dp_imag"]:.8e},'
                        f'{err_re:.4f},{err_im:.4f},{f_ratio:.6f},{j_ratio:.6f},'
                        f'{f_dW:.8e},{j_dW:.8e}\n')
    print(f'\nCSV saved to {path}')


def main():
    fortran = load_fortran_results()
    julia = load_julia_results()

    if not fortran:
        print("No Fortran results. Run: python3 run_fortran_stride.py")
    if not julia:
        print("No Julia results. Run: julia --project=../.. run_comparison.jl")

    if fortran or julia:
        print_comparison(fortran, julia)
        save_csv(fortran, julia)

        # Save table to file too
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_comparison(fortran, julia)
        sys.stdout = old_stdout
        table_path = OUTPUT_DIR / 'comparison_table.txt'
        with open(table_path, 'w') as f:
            f.write(buf.getvalue())
        print(f'Table saved to {table_path}')


if __name__ == '__main__':
    main()
