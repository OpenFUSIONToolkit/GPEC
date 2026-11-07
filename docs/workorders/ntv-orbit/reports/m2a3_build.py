"""Assemble reports/M2a3.html from M2a3.template.html with every figure inlined (self-contained, no external files).

Inputs: results/m2a3_dev, m2a3_zow512 and m2a3_z1phi fig_m2a3_*.png (m2a3_figures.py), m2a3_closure.svg (d2render.sh on
m2a3_closure.d2); TESTS is the test summary. Figures are numbered in order of first appearance.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[4] / "results"
DEV, ZOW, Z1 = RES / "m2a3_dev", RES / "m2a3_zow512", RES / "m2a3_z1phi"


def png(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    return re.sub(r"<!DOCTYPE[^>]*>", "", s).strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M2a3.template.html").read_text()
    html = html.replace("<!-- Built by m2a3_build.py from M2a3.template.html; edit the template, not M2a3.html. -->\n", "")
    # Figure numbers in order of first appearance of each {{N_*}} key.
    order = []
    for key in re.findall(r"\{\{N_(\w+)\}\}", html):
        if key not in order:
            order.append(key)
    numbers = {f"N_{k}": str(1 + i) for i, k in enumerate(order)}
    fields = {
        "GPEC_COMMIT": commit,
        "TESTS": sys.argv[1] if len(sys.argv) > 1 else "not run",
        "D2_CLOSURE": svg(DEV / "m2a3_closure.svg"),
        "FIG_PHI": png(DEV / "fig_m2a3_phi_orbits.png", "orbits trapped by E×B, and orbit classes before and after"),
        "FIG_JAC": png(DEV / "fig_m2a3_jac.png", "rebuilt Jacobian derivative over the equilibrium's, along theta"),
        "FIG_ZOW": png(ZOW / "fig_m2a3_zow.png", "torque difference against orbit width, both variants, attempts 2 and 3"),
        "FIG_FLOOR": png(ZOW / "fig_m2a3_floor.png", "each flux surface's share of the difference"),
        "FIG_CONVERGE": png(DEV / "fig_m2a3_converge.png", "torque change on doubling resolution and cost, potential off and on"),
        "FIG_PROFILE": png(Z1 / "fig_m2a3_profile.png", "torque density at physical width, potential off and on"),
        **numbers,
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M2a3.html").write_text(html)
    print(f"wrote {HERE / 'M2a3.html'} ({len(html) / 1024:.0f} KiB); figures {numbers}")


if __name__ == "__main__":
    main()
