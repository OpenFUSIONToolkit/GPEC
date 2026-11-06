"""Assemble reports/M2x.html (the M2 explainer) from M2x.template.html with figures inlined.

Inputs: results/m2_zow/fig_m2x_*.png (m2x_figures.py) and m2_table_{now,fix}.svg (d2 m2_table_{now,fix}.d2).
"""

import base64
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[4] / "results" / "m2_zow"


def png(name, alt):
    data = base64.b64encode((FIGS / f"{name}.png").read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(name):
    s = (FIGS / f"{name}.svg").read_text()
    return re.sub(r"<\?xml[^>]*\?>", "", s).strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M2x.template.html").read_text()
    html = html.replace("<!-- Built by m2x_build.py from M2x.template.html; edit the template, not M2x.html. -->\n", "")
    fields = {
        "GPEC_COMMIT": commit,
        "FIG_SCHEMATIC": png("fig_m2x_schematic", "pitch and orbit schematic"),
        "FIG_CHARGE": png("fig_m2x_charge", "orbit width and surface torque against charge"),
        "FIG_GRIDS": png("fig_m2x_grids", "pitch, energy and psi grids"),
        "FIG_LAYER": png("fig_m2x_layer", "orbit class maps with the layer band"),
        "FIG_DJ": png("fig_m2x_dj", "orbit action near the trapped-passing edge"),
        "FIG_FIX": png("fig_m2x_fix", "pitch nodes now and proposed"),
        "D2_NOW": svg("m2_table_now"),
        "D2_FIX": svg("m2_table_fix"),
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M2x.html").write_text(html)
    print(f"wrote {HERE / 'M2x.html'} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
