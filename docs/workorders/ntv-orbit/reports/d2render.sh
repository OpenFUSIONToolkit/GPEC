#!/usr/bin/env bash
# Render a D2 diagram in the project style (bold serif, as the matplotlib figures) to SVG, PDF and 300 dpi PNG.
# Usage: d2render.sh <diagram.d2> <out_dir>   (writes <out_dir>/<name>.{svg,pdf,png})
set -euo pipefail
src=$1; out=$2; name=$(basename "$src" .d2)
font=/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf
mkdir -p "$out"
d2 --layout elk --pad 10 --font-regular "$font" --font-bold "$font" --font-semibold "$font" --font-italic "$font" \
  "$src" "$out/$name.svg" >/dev/null 2>&1
# PDF and PNG through headless Chrome, page sized to the SVG (d2's own PDF export needs a Playwright download).
read -r w h < <(grep -o 'viewBox="[^"]*"' "$out/$name.svg" | head -1 | tr -d '"' | awk '{print $3, $4}')
tmp=$(mktemp -d)
printf '<html><head><style>@page{size:%spx %spx;margin:0}html,body{margin:0;overflow:hidden}</style></head><body>%s</body></html>' \
  "$w" "$h" "$(cat "$out/$name.svg")" > "$tmp/p.html"
chrome() { timeout 60 google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$tmp/c" "$@" 2>/dev/null; }
chrome --no-pdf-header-footer --print-to-pdf="$out/$name.pdf" "file://$tmp/p.html"
chrome --hide-scrollbars --force-device-scale-factor=3.125 --window-size="$w,$h" --screenshot="$out/$name.png" "file://$tmp/p.html"
rm -rf "$tmp"
echo "$out/$name.pdf"
