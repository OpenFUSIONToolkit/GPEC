# Plot gal vs STRIDE Δ′ across the TJ-analytic beta and epsilon scans.
# Reads /tmp/gal_beta_scan_results.csv and /tmp/gal_epsilon_scan_results.csv.
using Pkg
Pkg.activate("/Users/pharr/Projects/GPEC_dev/docs/GPEC_julia")
using Plots, DelimitedFiles, Printf
gr()

_median(v) = (w = sort(v); n = length(w); iseven(n) ? (w[n÷2] + w[n÷2+1]) / 2 : w[(n+1)÷2])

function loadcsv(path)
    isfile(path) || return nothing
    data, hdr = readdlm(path, ','; header=true)
    return Dict(strip(h) => Float64.(data[:, i]) for (i, h) in enumerate(vec(hdr)))
end

# Linear-extrapolate δW_total → 0 from the last k points → marginal-stability estimate.
function marginal_point(cols, xname; k=6)
    x = cols[xname]; dw = cols["dW_total"]
    ord = sortperm(x); x = x[ord]; dw = dw[ord]
    k = min(k, length(x))
    xt = x[end-k+1:end]; yt = dw[end-k+1:end]
    coef = hcat(xt, ones(k)) \ yt
    return coef[1] == 0 ? NaN : -coef[2] / coef[1]
end

# Δ′ vs scan parameter; gal (solid+markers) vs STRIDE (dashed). Log y if all-positive.
function panel(cols, xname, xlabel, title, xmarg)
    x = cols[xname]
    # log-y Δ′; non-positive values (q=3 crosses zero at low ε) are masked to NaN → shown as gaps.
    poslog(v) = [y > 0 ? y : NaN for y in v]
    p = plot(; xlabel=xlabel, ylabel="Δ′ (PEST-3)", title=title, legend=:topleft,
        yscale=:log10,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm)
    for (lbl, col, c) in (("q=2", "m2", :dodgerblue), ("q=3", "m3", :crimson))
        plot!(p, x, poslog(cols["gal_$col"]); lw=2, color=c, marker=:circle, ms=3, label="gal $lbl")
        plot!(p, x, poslog(cols["stride_$col"]); lw=2, ls=:dash, color=c, label="STRIDE $lbl")
    end
    if isfinite(xmarg)
        # extend x-range so the asymptote line (just past the last point) is visible
        plot!(p; xlims=(minimum(x), max(maximum(x), xmarg) * 1.002))
        vline!(p, [xmarg]; color=:black, ls=:dot, lw=2,
            label=@sprintf("marginal (δW=0): %s≈%.4f", xname == "pc" ? "pₐ" : "ε", xmarg))
    end
    return p
end

# δW_total vs scan parameter, with a linear extrapolation of the last points to δW=0
# (the ideal marginal-stability / kink-pole estimate).
function dwpanel(cols, xname, xlabel)
    x = cols[xname]; dw = cols["dW_total"]
    ord = sortperm(x); x = x[ord]; dw = dw[ord]
    p = plot(; xlabel=xlabel, ylabel="δW_total (Re et[1])", legend=:topright,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm)
    plot!(p, x, dw; lw=2, color=:black, marker=:circle, ms=3, label="δW_total")
    hline!(p, [0.0]; color=:gray, ls=:dot, label="marginal (δW=0)")
    x0 = marginal_point(cols, xname)
    if isfinite(x0)
        k = min(6, length(x)); xt = x[end-k+1:end]
        coef = hcat(xt, ones(length(xt))) \ dw[end-k+1:end]
        xext = range(xt[1], x0; length=20)
        plot!(p, xext, coef[1] .* xext .+ coef[2]; lw=1.5, ls=:dash, color=:red,
            label=@sprintf("extrap: δW=0 at %s≈%.4f", xname == "pc" ? "pₐ" : "ε", x0))
        scatter!(p, [x0], [0.0]; color=:red, ms=6, marker=:star5, label="")
        @printf("  marginal-point estimate (%s): δW=0 at %.5f\n", xname, x0)
    end
    return p
end

# Relative |gal-STRIDE|/|STRIDE| in %.
function relpanel(cols, xname, xlabel)
    x = cols[xname]
    p = plot(; xlabel=xlabel, ylabel="|gal−STRIDE|/|STRIDE| (%)", yscale=:log10, legend=:topright,
        left_margin=12Plots.mm, bottom_margin=6Plots.mm, right_margin=4Plots.mm)
    for (lbl, col, c) in (("q=2", "m2", :dodgerblue), ("q=3", "m3", :crimson))
        rel = 100 .* abs.(cols["gal_$col"] .- cols["stride_$col"]) ./ max.(abs.(cols["stride_$col"]), 1e-12)
        rel = max.(rel, 1e-4)
        plot!(p, x, rel; lw=2, color=c, marker=:circle, ms=3, label=lbl)
    end
    return p
end

for (name, path, xn, xl, ttl) in (
        ("beta", "/tmp/gal_beta_scan_results.csv", "pc", "pₐ (on-axis pressure factor)", "TJ-analytic β scan (ε=0.2, qc=1.5, qa=3.6)"),
        ("eps", "/tmp/gal_epsilon_scan_results.csv", "pc", "ε = a/R₀", "TJ-analytic ε scan (pc=0.001, qc=1.5, qa=3.6)"))
    cols = loadcsv(path)
    cols === nothing && continue
    xmarg = marginal_point(cols, xn)
    fig = plot(panel(cols, xn, xl, ttl, xmarg), dwpanel(cols, xn, xl), relpanel(cols, xn, xl); layout=(3, 1), size=(820, 1320))
    out = "/tmp/gal_$(name)_comparison.png"
    savefig(fig, out)
    println("Saved figure: ", out)
    println("=== $name: gal-vs-STRIDE relative diff (%) ===")
    for col in ("m2", "m3")
        rel = 100 .* abs.(cols["gal_$col"] .- cols["stride_$col"]) ./ max.(abs.(cols["stride_$col"]), 1e-12)
        @printf("  %s: median=%.3f%%  max=%.2f%%  (n=%d)\n", col, _median(rel), maximum(rel), length(rel))
    end
end
println("DONE")
