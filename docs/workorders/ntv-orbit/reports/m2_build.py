"""Assemble reports/M2.html from M2.template.html with every figure inlined (self-contained, no external files).

Inputs: results/m2_zow/fig_m2_*.png (m2_figures.py), fig_m2x_*.png (m2x_figures.py) and m2_{pipeline,table_now,table_fix}.svg
(d2 --layout elk on the matching .d2); TESTS is the test summary.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[4] / "results" / "m2_zow"


def png(name, alt):
    data = base64.b64encode((FIGS / f"{name}.png").read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(name):
    s = (FIGS / f"{name}.svg").read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    return re.sub(r"<!DOCTYPE[^>]*>", "", s).strip()


def variant_fields():
    """Check-3 text from variants.csv (m2_checks.jl variants); check 4 is written by hand in the template."""
    rows = {r[0]: r[1:] for r in (line.split(",") for line in (FIGS / "variants.csv").read_text().splitlines()[1:])}
    dev = float(rows["replay_max_node_dev"][0])
    return {"VAR_REPLAY": f"{dev:.1e} (largest node difference over the largest node value)", "VAR_REPLAY_SHORT": f"{dev:.0e}"}


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M2.template.html").read_text()
    html = html.replace("<!-- Built by m2_build.py from M2.template.html; edit the template, not M2.html. -->\n", "")
    fields = {
        "GPEC_COMMIT": commit,
        "TESTS": sys.argv[1] if len(sys.argv) > 1 else "not run",
        "FIG_SCHEMATIC": png("fig_m2x_schematic", "pitch and the two averaging paths"),
        "FIG_GRIDS": png("fig_m2x_grids", "pitch, energy and psi grids"),
        "FIG_CHARGE": png("fig_m2x_charge", "orbit width and surface torque against charge"),
        "FIG_ZOW": png("fig_m2_zow", "relative torque difference against orbit width"),
        "FIG_PROFILE": png("fig_m2_profile", "torque density at the edge"),
        "FIG_LAYER": png("fig_m2x_layer", "orbit class maps with the layer band"),
        "FIG_DJ": png("fig_m2x_dj", "orbit action near the trapped-passing edge"),
        "FIG_FIX": png("fig_m2x_fix", "pitch nodes now and proposed"),
        "FIG_KERNEL": png("fig_m2x_kernel", "local matrix against psi-psi' kernel"),
        "D2_PIPELINE": svg("m2_pipeline"),
        "D2_NOW": svg("m2_table_now"),
        "D2_FIX": svg("m2_table_fix"),
        **variant_fields(),
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M2.html").write_text(html)
    print(f"wrote {HERE / 'M2.html'} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
