"""Assemble reports/M2a2.html from M2a2.template.html with every figure inlined (self-contained, no external files).

Inputs: results/m2a2_dev/fig_m2a2_*.png and results/m2a2_zow/fig_m2a2_*.png (m2a2_figures.py), m2a2_pipeline.svg
(d2render.sh on m2a2_pipeline.d2); TESTS is the test summary. Figures are numbered in order of first appearance.
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[4] / "results"
DEV, ZOW = RES / "m2a2_dev", RES / "m2a2_zow"


def png(path, alt):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{data}">'


def svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    return re.sub(r"<!DOCTYPE[^>]*>", "", s).strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M2a2.template.html").read_text()
    html = html.replace("<!-- Built by m2a2_build.py from M2a2.template.html; edit the template, not M2a2.html. -->\n", "")
    # Figure numbers in order of first appearance of each {{N_*}} key; Figures 1-3 are numbered by hand in Section 1.
    order = []
    for key in re.findall(r"\{\{N_(\w+)\}\}", html):
        if key not in order:
            order.append(key)
    numbers = {f"N_{k}": str(4 + i) for i, k in enumerate(order)}
    fields = {
        "GPEC_COMMIT": commit,
        "TESTS": sys.argv[1] if len(sys.argv) > 1 else "not run",
        "D2_PIPELINE": svg(DEV / "m2a2_pipeline.svg"),
        "FIG_URES": png(DEV / "fig_m2a2_ures.png", "orbit quantities along energy at fixed pitch"),
        "FIG_TABLE": png(DEV / "fig_m2a2_table.png", "one per-column orbit table in pitch and energy"),
        "FIG_ZOW": png(ZOW / "fig_m2a2_zow.png", "torque difference against orbit width, both variants"),
        "FIG_FLOOR": png(ZOW / "fig_m2a2_floor.png", "each flux surface's share of the difference"),
        "FIG_WD": png(DEV / "fig_m2a2_wd.png", "orbit precession over surface precession by pitch"),
        "FIG_CONVERGE": png(DEV / "fig_m2a2_converge.png", "torque change on doubling resolution and cost"),
        "FIG_PITCH": png(DEV / "fig_m2a2_pitch.png", "pitch resolution of the replay and fgar"),
        "FIG_PROFILE": png(ZOW / "fig_m2a2_profile.png", "torque density at physical width"),
        **numbers,
    }
    for key, val in fields.items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M2a2.html").write_text(html)
    print(f"wrote {HERE / 'M2a2.html'} ({len(html) / 1024:.0f} KiB); figures {numbers}")


if __name__ == "__main__":
    main()
