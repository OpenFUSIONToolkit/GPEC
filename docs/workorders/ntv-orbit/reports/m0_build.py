"""Assemble reports/M0.html from M0.template.html with the SVGs inlined (self-contained, no external files).

Inputs: results/m0_report/pipeline.svg (d2 --layout elk m0_pipeline.d2) and fig_m0_torque.svg (m0_figures.py).
"""

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FIGS = ROOT / "results" / "m0_report"


def inline_svg(path):
    s = path.read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s)
    return s.strip()


def main():
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    html = (HERE / "M0.template.html").read_text()
    html = html.replace("<!-- Built by m0_build.py from M0.template.html; edit the template, not M0.html. -->\n", "")
    html = html.replace("{{GPEC_COMMIT}}", commit)
    html = html.replace("{{PIPELINE_SVG}}", inline_svg(FIGS / "pipeline.svg"))
    html = html.replace("{{FIG_SVG}}", inline_svg(FIGS / "fig_m0_torque.svg"))
    assert "{{" not in html, re.findall(r"\{\{\w+\}\}", html)
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    (HERE / "M0.html").write_text(html)
    print(f"wrote {HERE / 'M0.html'} ({len(html) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
