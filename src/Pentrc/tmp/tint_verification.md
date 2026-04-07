# Tint Verification

Reference model: direct translation of the `tintgrl_lsode` wrapper behavior in `src/Pentrc/torque.F90`, with a synthetic smooth `tpsi` surrogate and the same clipped psi interval logic used by the Julia port.

Julia solver used for `tintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`

## Effective Interval

- requested `psilims=[0.08, 0.95]`
- clipped interval `(0.18, 0.84)`
- `sq` interval `(0.18, 0.92)`
- `xs_m[1]` interval `(0.12, 0.84)`

## Harmonic Comparison

| l | current | ref | abs err | rel err |
| ---: | ---: | ---: | ---: | ---: |
| -2 | -0.08821931215377467 - 0.0782317209497468im | -0.08821928158615037 - 0.07823172087825915im | 3.056770788584635e-8 | 2.5924539624766356e-7 |
| -1 | 0.37465248179369387 - 0.06391674959999999im | 0.3746534707551254 - 0.06391674960000004im | 9.889614315250483e-7 | 2.602074453228987e-6 |
| 0 | 0.12376889892678643 - 0.051041078126899245im | 0.12376511930012277 - 0.051041078185486574im | 3.779626664114905e-6 | 2.8232126749981635e-5 |
| 1 | 0.3806406725111844 - 0.12144768240005961im | 0.3806490170157334 - 0.12144768949515054im | 8.344507565381933e-6 | 2.0884567193754734e-5 |
| 2 | 0.38338058273738973 - 0.16371317572792424im | 0.3833794862912361 - 0.16371313852674815im | 1.0970770690115948e-6 | 2.631691417614341e-6 |

## Total Comparison

- current total: `1.1742233238152797 - 0.4783504068046298im`
- ref total: `1.1742278117760674 - 0.4783503766856445im`
- component vector abs err: `9.278951526005793e-6`
- component vector rel err: `1.2996935707123797e-5`
- total rel err: `3.5396944712453936e-6`

## Parameters

- `nn=3`, `nl=2`, `zi=1`, `mi=2`
- `wdfac=0.35`, `divxfac=-0.22`, `electron=false`
- `method=clar`
- tolerances: `abstol=0.001`, `reltol=1.0e-6`
