"""
    write_imas(dd::IMASdd.dd, result::NamedTuple) -> IMASdd.dd

Write JPEC/DCON linear stability results into the `mhd_linear` IDS of an IMAS data
dictionary `dd`.

## What gets written

One `time_slice` is created (time taken from `dd.equilibrium.time_slice[1]`).
Inside that slice, one `toroidal_mode` entry is written **per computed eigenmode**
(there are `numpert_total = mpert × npert` eigenmodes in total).

For each eigenmode `i`:
- `n_tor`            — toroidal mode number, assigned by block index `nlow + (i-1) ÷ mpert`.
                       Exact for single-n runs (`npert == 1`); approximate for multi-n
                       runs because the global eigenvalue sort breaks strict block ordering.
- `energy_perturbed` — real part of the `i`-th total (plasma+vacuum) energy eigenvalue
                       `real(et[i])` in DCON's internal normalisation units.
                       Negative value → mode is MHD-unstable (δW < 0 criterion).
                       Only written when `vac_flag = true` and vacuum data is available.

## Top-level metadata written to `dd.mhd_linear`
- `ideal_flag = 1`                (ideal MHD)
- `ids_properties.homogeneous_time = 1`
- `ids_properties.comment`        (code name + run range)
- `code.name = "JPEC"`
"""
function write_imas(dd::IMASdd.dd, result::NamedTuple)

    ctrl     = result.ctrl
    intr     = result.intr
    vac_data = result.vac_data

    # ── Time: take from the equilibrium already stored in dd ─────────────────
    t = dd.equilibrium.time_slice[1].time

    # ── Top-level mhd_linear metadata ────────────────────────────────────────
    mhd = dd.mhd_linear

    mhd.ids_properties.comment         = "JPEC DCON ideal MHD linear stability, n = $(intr.nlow):$(intr.nhigh)"
    mhd.ids_properties.homogeneous_time = 1
    mhd.ideal_flag                      = 1
    mhd.code.name                       = "JPEC"
    mhd.time                            = [t]

    # ── Create one time slice ─────────────────────────────────────────────────
    resize!(mhd.time_slice, 1)
    ts      = mhd.time_slice[1]
    ts.time = t

    # ── One toroidal_mode entry per eigenmode ─────────────────────────────────
    # numpert_total = mpert × npert eigenmodes were computed.  For each one we
    # record the toroidal mode number (from its block position) and, when the
    # vacuum calculation was run, the total perturbed energy eigenvalue.
    resize!(ts.toroidal_mode, intr.numpert_total)

    for i in 1:intr.numpert_total

        tm = ts.toroidal_mode[i]

        # Block-position estimate of the toroidal mode number.
        # For npert == 1 this is exact; for npert > 1 it is approximate because
        # the global eigenvalue sort may interleave modes from different n-blocks.
        ipert_n  = (i - 1) ÷ intr.mpert + 1
        tm.n_tor = intr.nlow + ipert_n - 1

        # Perturbed energy: real part of total (plasma + vacuum) energy eigenvalue.
        # Eigenvalues are sorted ascending — et[1] is the most unstable (most negative).
        # Units are DCON-internal (normalised by kinetic energy / dV/dψ at the edge).
        if ctrl.vac_flag && vac_data !== nothing
            tm.energy_perturbed = real(vac_data.et[i])
        end

    end

    return dd
end
