# 복원된 Mercier.jl 동등성 및 코어 불안정성 분석 노트

## 목적

이 노트는 복원된 `D_R` / Mercier 경로에 대한 리뷰 질문에 답변하는 데 필요한 PR #234 관련 증거를 정리한 것입니다:

1. 기존 `Mercier.jl` 계산을 `analysis/` 아래에 로컬로 복원하여 DIII-D 유사 예제(DIII-D-like example)에 대해 실행했습니다.
2. 복원된 기존 계산을 현재 `Bal.jl`의 `resistive_interchange()` 경로와 비교했습니다.
3. 국소 안정성(local-stability) 값이 급증(spike)하는 코어 영역에서 복원된 기존 Mercier 계산 결과를 확인했습니다.

## 생성된 파일 목록

```text
analysis/pr234_mercier_core/restored_Mercier.jl
analysis/pr234_mercier_core/generate_mercier_comparison.jl
analysis/pr234_mercier_core/plot_mercier_comparison.py
analysis/pr234_mercier_core/mercier_comparison.csv
analysis/pr234_mercier_core/mercier_core_1e-4_to_1e-1.png
analysis/pr234_mercier_core/mercier_outer_1e-1_to_0p99.png
```

## 결과 1: 복원된 Mercier.jl과 Bal.jl의 resistive_interchange()는 동일한 계산임

복원된 기존 계산 코드는 다음 위치에 있습니다:

```text
analysis/pr234_mercier_core/restored_Mercier.jl
```

이 코드는 `origin/develop` 브랜치의 기존 `src/ForceFreeStates/Mercier.jl` 계산을 진단용으로 복사한 것입니다.

현재 브랜치는 다음 파일 내부에서 동일한 면적 평균(surface-average) Mercier 물리량을 계산합니다:

```text
src/ForceFreeStates/Bal.jl
```

해당하는 현재 함수는 다음과 같습니다:

```julia
resistive_interchange(flux_surface_index, plasma_eq)
```

두 경로 모두 동일한 자속면 평균(flux-surface average)을 구성합니다:

```julia
ff_fs[itheta, 1] = bsq / dpsisq
ff_fs[itheta, 2] = 1.0 / dpsisq
ff_fs[itheta, 3] = 1.0 / bsq
ff_fs[itheta, 4] = 1.0 / (bsq * dpsisq)
ff_fs[itheta, 5] = bsq
ff_fs[itheta, :] .*= jac / v1
```

그 후 두 경로 모두 동일한 Mercier 및 저항성 인터체인지(resistive-interchange) 공식을 사용합니다:

```julia
term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
di = -0.25 + term * (1 - term) +
     p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
     (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])
dr = di + (h - 0.5)^2
```

차이점은 단지 구조적인 부분뿐입니다:

- 기존 `Mercier.jl`: 모든 `psi` 면에 대해 루프를 돌며 `locstab_fs`를 직접 작성함
- 현재 `Bal.jl`: 단일 면을 평가하여 `(di, dr, h)`를 반환함

수치적 비교를 통해 이를 확인할 수 있습니다. 복원된 `Mercier.jl`의 `D_I`와 현재 `Bal.jl`의 `resistive_interchange().di` 사이의 최대 상대 오차는 기계 정밀도(machine precision) 수준입니다:

```text
코어 영역 (psi_N = 1e-4 ... 1e-1):
  최대 상대 오차 = 1.18e-16

외곽 영역 (psi_N = 1e-1 ... 0.99):
  최대 상대 오차 = 1.97e-16
```

## 결과 2: 복원된 Mercier.jl 역시 코어 영역에서 불안정함

코어 영역 플롯인:

```text
analysis/pr234_mercier_core/mercier_core_1e-4_to_1e-1.png
```

를 보면 복원된 기존 `Mercier.jl`의 `D_I`가 현재 `Bal.jl`의 `resistive_interchange()` 계산과 동일한 코어 스파이크(값 급증)를 보임을 알 수 있습니다.

대표적인 값들은 다음과 같습니다:

```text
psi_N       q'         restored Mercier D_I   Bal.jl resistive D_I   det(d0bar) D_I
1.00e-4    -3.970     +2.010                +2.010                +4.685
1.50e-4    -2.434     +5.376                +5.376                +6.620
2.26e-4    -0.426     +96.846               +96.846               +66.236
1.73e-3    -0.178     -164.886              -164.886              -151.517
3.91e-3    +0.146     -134.295              -134.295              -136.387
5.88e-3    +0.108     -150.304              -150.304              -149.781
1.33e-2    +0.156     -30.532               -30.532               -30.293
1.00e-1    +0.382     -1.049                -1.049                -1.049
```

이는 기존 Mercier 계산 경로가 코어 영역의 민감성 문제를 피하지 못했음을 의미합니다. 이미 해당 문제를 가지고 있었습니다.

## 결과 3: 조건수가 불량한(ill-conditioned) 코어를 벗어나면 두 곡선이 일치함

외곽 영역 플롯인:

```text
analysis/pr234_mercier_core/mercier_outer_1e-1_to_0p99.png
```

를 보면 복원된 `Mercier.jl`, 현재 `Bal.jl`의 `resistive_interchange()`, 그리고 `det(d0bar)`가 아래 영역 전체에서 함께 움직이는 것을 볼 수 있습니다:

```text
psi_N = 1e-1 ... 0.99
```

이 범위에서:

```text
복원된 Mercier vs 현재 resistive_interchange:
  최대 상대 오차 = 1.97e-16

det(d0bar) vs 복원된 Mercier:
  최대 상대 오차 = 1.53e-3
```

따라서 코어가 아닌 영역에서는 면적 평균 Mercier 경로와 `det(d0bar)` 경로 사이의 의도된 등가성(equivalence)이 성립함을 지지하는 반면, 코어 영역에서는 자기축 근처의 조건화(conditioning) 문제를 드러냅니다.

## 코어 영역의 조건수가 불량한(ill-conditioned) 이유

자기축(magnetic axis) 근처에서는 `q' = dq/dpsi_N` 값이 작고 부호가 바뀝니다. 공식들에는 `1/q'` 및 `1/q'^2`에 대한 의존성이 명시적으로 포함되어 있습니다.

복원된/현재 Mercier 공식에서:

```julia
term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
di = -0.25 + term * (1 - term) +
     p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
     (...)
```

따라서:

```text
term ~ 1/q'
두 번째 기여 항목 ~ 1/q'^2
```

`det(d0bar)` 경로에서:

```julia
nabla_beta_sq_b_sq_peculiar_2nd = term2_factor * (q_derivative^2)
m0_12 = jac_chiprime ./ nabla_beta_sq_b_sq_peculiar_2nd
```

따라서:

```text
m0_12 ~ 1/q'^2
```

이 때문에 두 계산 모두 동일한 코어 영역에서 민감해집니다.

이 경우 자기장 크기(field magnitude)가 원인은 아닙니다. 동일한 코어 표면들에서:

```text
B^2      은 매끄러운 상태를 유지하며 0에서 안전하게 벗어나 있음
jac      은 거의 일정함
dpsisq   는 축 근처에서 예상대로 작지만 매끄럽게 변함
q'       는 정확히 D_I가 급증하는 위치에서 값이 작고 부호가 바뀜
```

## 권장하는 리뷰어 답변 초안

```text
analysis/pr234_mercier_core 아래에 기존 Mercier.jl 계산을 복원하고 현재 Bal.jl의 resistive_interchange() 경로와 직접 비교했습니다. 두 경로는 동일한 면적 평균 Mercier 계산입니다. 복원된 Mercier.jl과 현재의 resistive_interchange()는 기계 정밀도 수준에서 일치합니다.

코어 영역의 스파이크(값 급증)는 Bal.jl 재작성으로 인해 도입된 것이 아닙니다. 복원된 기존 Mercier.jl 계산도 동일한 코어 거동을 보입니다. 원인은 자기축 근처의 q'(자기 셰어, magnetic shear) 조건화 문제입니다. psi_N ~ 1e-4 ... 1e-2 영역에서 q'이 작고 부호가 바뀌는 반면, Mercier 공식과 det(d0bar) 구성 모두 1/q' 또는 1/q'^2 구조를 포함하고 있습니다. 조건수가 불량한 코어 영역을 벗어난 psi_N >= 0.1 에서는 복원된 Mercier, 현재의 resistive_interchange, 그리고 det(d0bar) 곡선이 서로 밀접하게 일치하며 움직집니다.

따라서 복원된 D_R 경로는 기존 Mercier 경로와 일관된다고 생각합니다. 이와 별개로, 자기축 근처의 국소 안정성 진단은 아주 작은 코어 컷오프 값 이하에서는 신뢰할 수 없는 것으로 문서화하거나, 추후 정규화된 축/제로 셰어 한계(regularized axis/zero-shear limit)를 적용하여 처리하는 것이 좋겠습니다.
```
