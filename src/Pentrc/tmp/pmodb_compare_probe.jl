include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using Printf

const P = JPEC.PENTRC

function main(case_dir::String=joinpath(@__DIR__, "runtime_rlar_debug_case"))
    eq = JPEC.Equilibrium.setup_equilibrium(joinpath(case_dir, "equil.toml"))
    P.load_equilibrium!(eq)
    P.load_kinetic_profiles!(joinpath(case_dir, "g147131.02300_DIIID_KEFIT.kin"))
    P.load_pmodb!(joinpath(case_dir, "gpec_pmodb_n1.out"))

    table, _ = P.readtable(joinpath(case_dir, "pentrc_pmodb_n1.out"))
    psi = table[:, 1]
    ms = round.(Int, table[:, 2])
    db_fortran = ComplexF64.(table[:, 3], table[:, 4])
    dx_fortran = ComplexF64.(table[:, 5], table[:, 6])

    mgrid = collect(P.mfac)
    psi_grid = collect(P.dbob_m.xs)
    npsi = length(psi_grid)
    nm = length(mgrid)

    max_db_abs = 0.0
    max_dx_abs = 0.0
    sum_db_abs = 0.0
    sum_dx_abs = 0.0
    count = 0
    worst_db = (0.0, 0, 0.0 + 0.0im, 0.0 + 0.0im)
    worst_dx = (0.0, 0, 0.0 + 0.0im, 0.0 + 0.0im)

    for i in 1:npsi
        for j in 1:nm
            row = (i - 1) * nm + j
            psi_f = psi[row]
            m_f = ms[row]
            db_j = P.dbob_m.fs[i, j]
            dx_j = P.divx_m.fs[i, j]
            db_f = db_fortran[row]
            dx_f = dx_fortran[row]

            if !(isapprox(psi_f, psi_grid[i]; atol=1e-10, rtol=0) && m_f == mgrid[j])
                error("Grid mismatch at row $row: Fortran (psi=$psi_f,m=$m_f) vs Julia (psi=$(psi_grid[i]),m=$(mgrid[j]))")
            end

            edb = abs(db_j - db_f)
            edx = abs(dx_j - dx_f)
            max_db_abs = max(max_db_abs, edb)
            max_dx_abs = max(max_dx_abs, edx)
            sum_db_abs += edb
            sum_dx_abs += edx
            count += 1

            if edb == max_db_abs
                worst_db = (psi_f, m_f, db_f, db_j)
            end
            if edx == max_dx_abs
                worst_dx = (psi_f, m_f, dx_f, dx_j)
            end
        end
    end

    @printf("dbob_max_abs_err=%.12e\n", max_db_abs)
    @printf("divx_max_abs_err=%.12e\n", max_dx_abs)
    @printf("dbob_mean_abs_err=%.12e\n", sum_db_abs / count)
    @printf("divx_mean_abs_err=%.12e\n", sum_dx_abs / count)
    @printf("worst_db psi=%.12e m=%d fortran=%.12e %+.12ei julia=%.12e %+.12ei\n",
            worst_db[1], worst_db[2], real(worst_db[3]), imag(worst_db[3]), real(worst_db[4]), imag(worst_db[4]))
    @printf("worst_dx psi=%.12e m=%d fortran=%.12e %+.12ei julia=%.12e %+.12ei\n",
            worst_dx[1], worst_dx[2], real(worst_dx[3]), imag(worst_dx[3]), real(worst_dx[4]), imag(worst_dx[4]))
end

main()
