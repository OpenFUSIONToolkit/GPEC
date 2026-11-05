"""Assemble reports/M1.html from M1.template.html with the SVGs inlined (self-contained, no external files).

Inputs: results/m1_orbits/fig_m1_orbits.svg and fig_m1_zow.svg (m1_figures.py); TESTS is the test summary.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FIGS = ROOT / "results" / "m1_orbits"


def inline_svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    return s.strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M1.template.html").read_text()
    html = html.replace("<!-- Built by m1_build.py from M1.template.html; edit the template, not M1.html. -->\n", "")
    html = html.replace("{{GPEC_COMMIT}}", commit)
    html = html.replace("{{FIG_ORBITS}}", inline_svg(FIGS / "fig_m1_orbits.svg"))
    html = html.replace("{{FIG_ZOW}}", inline_svg(FIGS / "fig_m1_zow.svg"))
    html = html.replace("{{TESTS}}", sys.argv[1] if len(sys.argv) > 1 else "not run")
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M1.html").write_text(html)
    print(f"wrote {HERE / 'M1.html'} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
