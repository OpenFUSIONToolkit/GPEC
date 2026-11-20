# The shared look of every Analysis figure and the small text helpers its labels use.

const WONG = Makie.wong_colors()

const GPEC_THEME = Theme(;
    fontsize=15,
    figure_padding=(12, 18, 10, 10),
    Axis=(titlesize=16, titlefont=:regular, titlegap=6, xlabelsize=15, ylabelsize=15, xticklabelsize=13, yticklabelsize=13,
        xgridcolor=(:black, 0.07), ygridcolor=(:black, 0.07), spinewidth=1.0, xtickwidth=1.0, ytickwidth=1.0,
        xminorticksvisible=false, yminorticksvisible=false, topspinevisible=true, rightspinevisible=true),
    Legend=(framevisible=true, framecolor=(:black, 0.15), backgroundcolor=(:white, 0.88), labelsize=13, padding=(6, 6, 4, 4), rowgap=0,
        patchsize=(18, 10)),
    Colorbar=(labelsize=15, ticklabelsize=13, width=12, spinewidth=1.0),
    Label=(fontsize=15,),
    palette=(color=WONG, marker=[:circle, :rect, :utriangle, :diamond, :dtriangle])
)

# Canvas and fixed panel sizes are written at their design size and laid out this much smaller, which makes the text,
# lines and markers larger relative to the figure: about 9 pt text when a single-panel figure is printed 5 inches wide.
const CANVAS_SCALE = 1 / 1.35
_px(length) = round(Int, length * CANVAS_SCALE)
_figure(size; kwargs...) = Figure(; size=_px.(size), kwargs...)

# Build a figure under the GPEC theme without changing the session's global theme.
themed(f) = with_theme(f, GPEC_THEME)

const SUPERSCRIPTS = Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
_sup(n::Integer) = join(SUPERSCRIPTS[c] for c in string(n))

# The power of ten to divide `v` by when its magnitude is outside [1e-2, 1e3), and that factor's axis label.
function _power(v)
    finite = filter(isfinite, v)
    m = isempty(finite) ? 0.0 : maximum(abs, finite)
    (m == 0 || 1e-2 <= m < 1e3) && return 1.0, ""
    k = floor(Int, log10(m))
    return 10.0^k, "10$(_sup(k))"
end

# `v` divided by its power of ten, and that factor's axis label.
function _decade(v)
    factor, label = _power(v)
    return v ./ factor, label
end

# Axis-label unit suffix, e.g. " (10⁻⁴ T)".
_unit(scale, unit) = isempty(scale) ? (isempty(unit) ? "" : " ($unit)") : " ($scale" * (isempty(unit) ? "" : " $unit") * ")"

# Three significant figures, with ×10ⁿ outside [1e-2, 1e3) and no trailing ".0".
function _num(v)
    v == 0 && return "0"
    a = abs(v)
    trim(s) = endswith(s, ".0") ? s[1:(end-2)] : s
    minus(s) = replace(s, "-" => "−")
    1e-2 <= a < 1e3 && return minus(trim(string(round(v; sigdigits=3))))
    k = floor(Int, log10(a))
    return minus(trim(string(round(v / 10.0^k; sigdigits=3)))) * "×10" * _sup(k)
end

sub(a, b) = rich(a, subscript(b))
sup(a, b) = rich(a, superscript(b))
const PSIN = sub("ψ", "N")

# y limits for pseudo-log axes from the data alone, with zero always in view.
function _symlims(v)
    lo, hi = min(minimum(v), 0.0), max(maximum(v), 0.0)
    return (lo < 0 ? 1.5lo : -1.0, hi > 0 ? 1.25hi : 1.0)
end

# Ticks on whole numbers only, for axes that count harmonics or eigenmodes.
struct WholeTicks end
function Makie.get_tickvalues(::WholeTicks, vmin, vmax)
    span = vmax - vmin
    decade = 10^max(0, floor(Int, log10(max(span / 8, 1))))
    step = first(s for s in (1, 2, 5, 10) .* decade if span / s <= 8)
    return Float64.((ceil(Int, vmin / step)*step):step:floor(Int, vmax))
end

# Rational surfaces as a q = m/n axis along the top of the ψ_N axis `ax`, laid over it at grid position `pos`. Every
# surface gets its line; a label is left out where it would run into the one before it at the axis's width on screen.
function _surfaces_on_top!(pos, ax, psi, labels; title="")
    labelsize = 11
    top = Axis(pos; xaxisposition=:top, xticks=(psi, labels), xticklabelsize=labelsize, xticklabelrotation=π / 2, title,
        xgridvisible=false, ygridvisible=false, yticksvisible=false, yticklabelsvisible=false)
    hidespines!(top)
    linkxaxes!(ax, top)
    hideydecorations!(top)
    vlines!(ax, psi; color=(:black, 0.18), linewidth=1)
    onany(top.scene.viewport, top.finallimits) do viewport, limits
        pixels_per_psi = viewport.widths[1] / limits.widths[1]
        shown, last = Any[labels...], -Inf
        for i in sortperm(psi)
            if (psi[i] - last) * pixels_per_psi >= 0.9labelsize
                last = psi[i]
            else
                shown[i] = ""
            end
        end
        shown == top.xticks[][2] || (top.xticks = (psi, shown))
    end
    return top
end
