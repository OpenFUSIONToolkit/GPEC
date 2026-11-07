"""Assemble reports/M4.html from M4.template.html with every figure inlined (self-contained, no external files).

Inputs: fig_m4_*.png and fig_alpha_scan.png (m4_figures.py) in results/m4_arc and results/m4_sparc, m4_family.svg (d2render.sh
on m4_family.d2); the KR1 table is computed from the run files by m4_figures.scan(). TESTS (argv 1) is the test summary, FIXCHK (argv 2) the old-vs-fixed λ = 8 comparison.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

import m4_figures as M

HERE = Path(__file__).resolve().parent
ARC, SPARC = M.DEVICES["ARC"], M.DEVICES["SPARC"]


def png(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    return re.sub(r"<!DOCTYPE[^>]*>", "", s).strip()


def kr1_table():
    """One row per device and equilibrium, columns λ: Re forb / Re fgar of the fixed-ν totals, dense runs only."""
    S = [s for s in M.scan() if s["kind"] == "fixed" and s["src"].startswith(s["eq"] + "D")]
    full = {}
    for s in S:
        full[s["dev"], s["eq"]] = max(full.get((s["dev"], s["eq"]), 0), s["nodes"])
    out = ['<div class="tbl"><table>', "<tr><th>Device, q</th>" + "".join(f"<th>λ = {lam}</th>" for lam in M.LAMS) + "</tr>"]
    for dev in M.DEVICES:
        for eq, name in (("", "published"), ("q110_", "q₀ = 1.1")):
            row = {s["lam"]: s for s in S if s["dev"] == dev and s["eq"] == eq}
            if not row:
                continue
            cells = []
            for lam in M.LAMS:
                s = row.get(lam)
                if s is None:
                    cells.append("<td>–</td>")
                    continue
                r = s["forb"].real / s["fgar"].real
                part = "" if s["nodes"] == full[dev, eq] else f' <span class="part">({s["nodes"]}/{full[dev, eq]})</span>'
                cells.append(f"<td>{r:.3f}<br><small>Δ<sub>b</sub>/a {s['width']:.3f}</small>{part}</td>")
            out.append(f"<tr><td>{dev}, {name}</td>" + "".join(cells) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M4.template.html").read_text()
    html = html.replace("<!-- Built by m4_build.py from M4.template.html; edit the template, not M4.html. -->\n", "")
    order = []
    for key in re.findall(r"\{\{N_(\w+)\}\}", html):
        if key not in order:
            order.append(key)
    numbers = {f"N_{k}": str(1 + i) for i, k in enumerate(order)}
    fields = {
        "GPEC_COMMIT": commit,
        "TESTS": sys.argv[1] if len(sys.argv) > 1 else "not run",
        "FIXCHK": sys.argv[2] if len(sys.argv) > 2 else "pending",
        "KR1_TABLE": kr1_table(),
        "D2_FAMILY": svg(ARC / "m4_family.svg"),
        "FIG_COILS": png(ARC / "fig_m4_coils.png", "plasma boundaries and forcing coils of ARC, ITER and SPARC"),
        "FIG_SCAN": png(ARC / "fig_alpha_scan.png", "orbit to surface torque ratio against orbit width"),
        "FIG_NU": png(ARC / "fig_m4_nu.png", "per-surface difference against lambda, fixed nu and nu over lambda"),
        "FIG_SHIFT": png(ARC / "fig_m4_shift.png", "torque density across a resonance step for four lambda"),
        "FIG_INNER": png(SPARC / "fig_m4_inner.png", "SPARC inner-edge torque with three extensions of the response"),
        "FIG_Q0": png(ARC / "fig_m4_q0.png", "ARC surface alpha torque density for four central q values"),
        "FIG_DEFECT": png(ARC / "fig_m4_defect.png", "clustered energy samples, cubic overshoot and pitch kernel"),
        **numbers,
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M4.html").write_text(html)
    print(f"wrote {HERE / 'M4.html'} ({len(html) / 1024:.0f} KiB); figures {numbers}")


if __name__ == "__main__":
    main()
