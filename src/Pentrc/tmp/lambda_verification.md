# Lambda Verification

Reference model: direct source translation of `lintgrnd` from `src/Pentrc/pitch.f90`, using the already-verified Julia `xintgrl_lsode` for the energy-space sub-integral.

Julia solver used for `lambdaintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`

| lambda | lint current norm | lint ref norm | lint abs err |
| ---: | ---: | ---: | ---: |
| 0.25 | 4.603911674324343 | 4.603911674324343 | 0.0 |
| 0.3357142857142857 | 4.189294725010249 | 4.189294725010249 | 0.0 |
| 0.42142857142857143 | 3.7912649484919934 | 3.7912649484919934 | 0.0 |
| 0.5071428571428571 | 3.4213714388084338 | 3.4213714388084338 | 0.0 |
| 0.5928571428571429 | 1.5485680138333846 | 1.5485680138333846 | 0.0 |
| 0.6785714285714286 | 1.423802967413129 | 1.423802967413129 | 0.0 |
| 0.7642857142857142 | 1.3533961888169335 | 1.3533961888169335 | 0.0 |
| 0.85 | 1.3541931874605238 | 1.3541931874605238 | 0.0 |

## Integral Comparison

- `lintgrnd!` max sample error: `0.0`
- `lintgrnd!` mean sample error: `0.0`
- `lambdaintgrl_lsode` current: `ComplexF64[0.4242446835920722 - 1.4188072200547126im, -0.22445590623342068 - 0.2802004115122173im, 0.0563560344722568 - 0.11299271475850577im]`
- `lambdaintgrl_lsode` ref: `ComplexF64[0.42401489890133254 - 1.4180362528500345im, -0.224326393993732 - 0.2800391147472692im, 0.05633072034240801 - 0.11294185881294844im]`
- vector norm abs err: `0.0008325915167522714`
- vector norm rel err: `0.0005448308525186521`

## Parameters

- `wn=800.0`, `wt=-150.0`, `we=1300.0`, `nuk=0.9`
- `bobmax=0.53`, `epsr=0.18`, `q=2.4`
- `ell=1`, `n=2`, `rex=1.0`, `imx=1.0`
- `psi=0.42`, `method=clar`
