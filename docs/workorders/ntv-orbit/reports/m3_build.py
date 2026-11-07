"""Assemble reports/M3.html from M3.template.html with every figure inlined (self-contained, no external files).

Inputs: fig_m3_*.png from results/m3_zow2, m3_z1phi2 and m3_arc (m3_figures.py), m3_weight.svg in results/m3_dev (d2render.sh
on m3_weight.d2); TESTS is the test summary. Figures are numbered in order of first appearance.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[4] / "results"
DEV, ZOW, Z1, ARC = RES / "m3_dev", RES / "m3_zow2", RES / "m3_z1phi2", RES / "m3_arc"


def png(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    return re.sub(r"<!DOCTYPE[^>]*>", "", s).strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M3.template.html").read_text()
    html = html.replace("<!-- Built by m3_build.py from M3.template.html; edit the template, not M3.html. -->\n", "")
    # Figure numbers in order of first appearance of each {{N_*}} key.
    order = []
    for key in re.findall(r"\{\{N_(\w+)\}\}", html):
        if key not in order:
            order.append(key)
    numbers = {f"N_{k}": str(1 + i) for i, k in enumerate(order)}
    fields = {
        "GPEC_COMMIT": commit,
        "TESTS": sys.argv[1] if len(sys.argv) > 1 else "not run",
        "D2_WEIGHT": svg(DEV / "m3_weight.svg"),
        "FIG_ZOW": png(ZOW / "fig_m3_zow.png", "torque difference against orbit width, M2 and M3, both variants"),
        "FIG_PROFILE": png(Z1 / "fig_m3_profile.png", "thermal-ion torque density, surface method and orbit method at M2 and M3"),
        "FIG_MOMENTS": png(ARC / "fig_m3_moments.png", "alpha density and pressure against psi, model and gacode"),
        "FIG_ALPHA": png(ARC / "fig_m3_alpha.png", "alpha torque density against psi, real and imaginary parts"),
        "FIG_LOST": png(ARC / "fig_m3_lost.png", "alpha lost fraction against psi"),
        "FIG_Q1": png(ARC / "fig_m3_alpha_q1.png", "alpha torque density with the q = 1 surface inside the response"),
        **numbers,
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M3.html").write_text(html)
    print(f"wrote {HERE / 'M3.html'} ({len(html) / 1024:.0f} KiB); figures {numbers}")


if __name__ == "__main__":
    main()
