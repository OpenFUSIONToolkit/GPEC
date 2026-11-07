#!/usr/bin/env bash
# Render a D2 diagram to SVG, PDF and PNG with the project look: bold serif text, one node shape, plain strokes.
# Usage: d2render.sh diagram.d2 [outbase]   (outbase defaults to the input name without .d2)
# Conventions for the .d2 source: <= ~12 nodes, all rectangles; color only changed (class: new) or failing (class: fail) nodes.
set -euo pipefail
src=$1; out=${2:-${src%.d2}}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
font_dir=$(dirname "$(fc-match -f '%{file}' 'Liberation Serif:bold')")
reg="$font_dir/LiberationSerif-Bold.ttf"; ital="$font_dir/LiberationSerif-BoldItalic.ttf"
# Shared classes prepended to every diagram.
cat > "$tmp/in.d2" <<'D2'
vars: { d2-config: { layout-engine: dagre; pad: 10 } }
classes: {
  node: { shape: rectangle; style: { fill: "#FFFFFF"; stroke: "#333333"; stroke-width: 2; font-size: 18; bold: true; border-radius: 3 } }
  new: { shape: rectangle; style: { fill: "#FFF1E6"; stroke: "#D55E00"; stroke-width: 3; stroke-dash: 4; font-size: 18; bold: true; border-radius: 3 } }
  fail: { shape: rectangle; style: { fill: "#FDE8E8"; stroke: "#B00020"; stroke-width: 3; font-size: 18; bold: true; border-radius: 3 } }
}
D2
printf '*.style.font-size: 18\n(** -> **)[*].style.stroke: "#333333"\n(** -> **)[*].style.font-color: "#333333"\n' >> "$tmp/in.d2"
cat "$src" >> "$tmp/in.d2"
flags=(--font-regular "$reg" --font-bold "$reg" --font-italic "$ital" --font-semibold "$reg")
d2 "${flags[@]}" "$tmp/in.d2" "$out.svg" </dev/null
# PDF and PNG from the SVG through headless Chrome (d2's own export needs a Playwright browser, absent on the cluster).
read -r W H < <(python3 -c "import re,sys; v=re.search(r'viewBox=\"([-\\d. ]+)\"', open(sys.argv[1]).read()).group(1).split(); print(int(float(v[2]))+2, int(float(v[3]))+2)" "$out.svg")
cat > "$tmp/page.html" <<HTML
<html><head><style>@page{size:${W}px ${H}px;margin:0}html,body{margin:0}img{display:block}</style></head><body><img src="file://$(realpath "$out.svg")"></body></html>
HTML
chrome=(google-chrome --headless=new --disable-gpu --no-sandbox --user-data-dir="$tmp/chrome" --hide-scrollbars)
"${chrome[@]}" --no-pdf-header-footer --print-to-pdf="$(realpath -m "$out.pdf")" "file://$tmp/page.html" </dev/null >/dev/null 2>&1
"${chrome[@]}" --window-size="$W,$H" --force-device-scale-factor=3 --screenshot="$(realpath -m "$out.png")" "file://$tmp/page.html" </dev/null >/dev/null 2>&1
echo "$out.{svg,pdf,png}"
