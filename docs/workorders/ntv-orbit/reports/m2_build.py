"""Assemble reports/M2.html from M2.template.html with the SVGs inlined (self-contained, no external files).

Inputs: results/m2_zow/fig_m2_zow.svg and fig_m2_profile.svg (m2_figures.py); TESTS is the test summary.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FIGS = ROOT / "results" / "m2_zow"


def inline_svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    return s.strip()


def variant_fields():
    """Check-3 text from variants.csv (m2_checks.jl variants); check 4 is written by hand in the template."""
    rows = {r[0]: r[1:] for r in (line.split(",") for line in (FIGS / "variants.csv").read_text().splitlines()[1:])}
    return {"VAR_REPLAY": f"{float(rows['replay_max_node_dev'][0]):.1e} (largest node difference over the largest node value)"}


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M2.template.html").read_text()
    html = html.replace("<!-- Built by m2_build.py from M2.template.html; edit the template, not M2.html. -->\n", "")
    html = html.replace("{{GPEC_COMMIT}}", commit)
    html = html.replace("{{FIG_ZOW}}", inline_svg(FIGS / "fig_m2_zow.svg"))
    html = html.replace("{{FIG_PROFILE}}", inline_svg(FIGS / "fig_m2_profile.svg"))
    html = html.replace("{{TESTS}}", sys.argv[1] if len(sys.argv) > 1 else "not run")
    for key, val in variant_fields().items():
        html = html.replace("{{" + key + "}}", val)
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M2.html").write_text(html)
    print(f"wrote {HERE / 'M2.html'} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
