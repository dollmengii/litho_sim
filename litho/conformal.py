"""DOF 에 대한 분할 적합예측(split conformal prediction) 구간.

왜 이것이 필요한가
------------------
12 주차는 "대리 모형으로 공정창을 **인증**할 수는 없지만 **순위**는 쓸 수
있다" 고 적었고, 그 근거는 측정되지 않았다. 13 주차 되돌림이 순위 상관을
재면서 그 공백을 메웠지만 남은 질문이 있다. "그래서 이 DOF 예측을 얼마나
믿을 수 있는가."

추천기가 이미 내놓는 `[dof, dof_upper]` 구간은 그 질문의 답이 **아니다.**
그 구간은 격자 편향만 담는다. EL 목표를 넘기는 격자 행의 DOF 가 하한이고
상한 행이 상한이며, 모형이 틀린 만큼은 들어 있지 않다. 13 주차 실측에서
포함률이 20~50 % 로 낮았던 것이 그 증거다. 구간이 좁아서가 아니라 애초에
다른 것을 재는 구간이다.

그 자리에 들어갈 것이 분할 적합예측이다. 보정 집합에서 잔차 분위수 하나를
뽑아 `예측 ± q` 로 구간을 만들면 **분포 가정 없이** 표기 피복률이 유한
표본에서 보장된다(Vovk 2005; Lei & Wasserman 2014; Angelopoulos & Bates
2023). 모형 종류와 무관하고 오차가 정규분포일 필요도 없다.

보장의 정확한 형태
------------------
보정 표본 `n` 개의 잔차 `R_1..R_n` 에 대해 순서통계량

    q = R_(k),   k = ceil((n + 1) * (1 - alpha))

를 쓰면 교환가능성 아래에서 `P(R_{n+1} <= q) >= 1 - alpha` 다. 정확한
피복률은 `k / (n + 1)` 이다. 유한 표본 보정 `(n + 1)` 이 핵심이고, 그냥
`np.quantile(R, 1 - alpha)` 를 쓰면 보장이 깨진다(n=20, alpha=0.10 에서
19.0 대 18.1). `k > n` 이면 그 신뢰수준을 이 표본 수로는 낼 수 없다는
뜻이므로 **구간을 만들지 않고 못 만든다고 적는다.** 최댓값으로 때우면
보장을 흉내만 내는 것이다.

왜 데이터셋의 격자에서 보정하지 않는가 (13 주차에 설계를 바꾼 이유)
------------------------------------------------------------------
처음에는 데이터셋이 광학점마다 들고 있는 9 x 9 (초점 x 노광) 격자에서 참
DOF 와 예측 DOF 를 복원해 잔차를 모으려 했다. 두 가지 때문에 쓸 수 없다.

1. 그 격자는 사전 탐색으로 찾은 **적격 영역에 5 % 여유만** 두고 잡힌다
   (`campaign.design_grid`). 그래서 참 공정창이 거의 항상 격자 경계에
   닿고, 참 DOF 가 값이 아니라 하한이 된다. 실측 : 합성 60 그룹 전부가
   경계에 닿아 보정 표본이 0 개였다.
2. 더 근본적으로, 보성 곡선 `CD = base(dose) - c*dz^2` 와 대칭 공차에서는
   초점 경계와 노광 경계가 **동시에** 안쪽일 수 없다. 초점 경계가 전부
   실패하려면 초점 끝의 CD 하락 `D` 가 `A + tol` 보다 커야 하고(A 는
   노광에 의한 CD 변화 폭), 노광 경계가 전부 실패하려면 `D < A - tol`
   이어야 한다. 두 조건이 모순이다. 즉 격자를 어떻게 잡아도 한쪽은 닿는다.
   실측으로 세 조합 전부 확인했다.

그래서 보정을 **추천기가 실제로 쓰는 격자**에서 한다. 참값은 시뮬레이터를
그 격자에서 돌려 얻는다. 그러면 보정하는 양과 적용하는 양이 **같은 양**이
되어 격자 불일치가 사라진다. 비용은 보정 조건 하나당 창 하나 시뮬레이션이고
되돌림 자체와 같은 수준이다.

이 구현이 **보장하지 않는 것** (그대로 적는다)
---------------------------------------------
1. 교환가능성은 보정 조건과 적용 조건이 같은 분포라는 가정이다. 보정은
   데이터셋 검증 분할의 광학 조건에서 뽑고, 추천기는 **최적점 근처**를
   묻는다. 그 이동이 크면 피복률이 표기보다 낮아진다. 그래서 되돌림이
   실측 피복률을 따로 재고 두 숫자를 나란히 적는다. 어긋남의 크기가 분포
   이동의 크기다.
2. 구간은 DOF **한 축**에 대한 것이다. 노광량 폭이나 중심에는 적용되지
   않는다.
3. 참 DOF 는 시뮬레이터가 내는 값이다. 따라서 보장은 "시뮬레이터 대비"
   이고 "웨이퍼 대비" 가 아니다. 이 구분은 VERIFICATION_LIMITS 0 절의
   verification/validation 구분과 같은 것이다.
"""

from __future__ import annotations

import math

import numpy as np

#: 창을 이루는 두 축. 나머지 조건이 모두 같은 행들이 한 광학점이다.
SWEEP_AXES = ("defocus", "dose_rel")


def conformal_quantile(resid, alpha: float) -> dict:
    """분할 적합예측 분위수. 유한 표본 보정 `ceil((n+1)(1-alpha))`.

    돌려주는 것
    -----------
    q        : 분위수. None 이면 못 만든 것이고 `why` 에 이유가 있다.
    k, n     : 쓴 순서통계량의 번호와 보정 표본 수
    guarantee: 정확한 피복률 `k / (n + 1)`. 표기 신뢰수준 이상이다.
    """
    r = np.sort(np.asarray(resid, dtype=np.float64))
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"q": None, "n": 0, "k": None, "why": "보정 표본이 없다"}
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        need = math.ceil((1.0 - alpha) / alpha) if alpha > 0 else None
        return {"q": None, "n": int(n), "k": int(k),
                "why": (f"보정 표본 {n} 개로는 신뢰수준 "
                        f"{100 * (1 - alpha):.0f} % 를 낼 수 없다"
                        + (f" (최소 {need} 개 필요)" if need else ""))}
    return {"q": float(r[k - 1]), "n": int(n), "k": int(k),
            "guarantee": float(k / (n + 1)),
            "r_max": float(r[-1]), "r_median": float(np.median(r))}


def optical_conditions(cond: np.ndarray, ck: list[str],
                       rows: np.ndarray) -> list[dict]:
    """행 색인에서 **서로 다른 광학 조건**을 뽑는다. 초점·노광은 뺀다.

    왜 `merge_dataset.GROUP_KEYS` 를 쓰지 않는가 : 그쪽은 누설 방지용이라
    수차와 공구 상태를 일부러 빼고 묶는다. 창을 복원하려면 그 축들도
    고정해야 하므로, 여기서는 초점·노광을 뺀 **모든** 조건 열로 묶는다.
    섞으면 서로 다른 수차의 점이 한 창에 들어가 창이 뜻을 잃는다.
    """
    if not rows.size:
        return []
    drop = {ck.index(a) for a in SWEEP_AXES if a in ck}
    keep = [j for j in range(cond.shape[1]) if j not in drop]
    seen, out = set(), []
    for r in rows:
        key = tuple(np.round(cond[int(r), keep], 10))
        if key in seen:
            continue
        seen.add(key)
        out.append({ck[j]: float(cond[int(r), j]) for j in keep})
    return out


def near_request(conds: list[dict], pitch: float, duty: float,
                 pitch_window: float, duty_window: float) -> list[dict]:
    """요청 피치·duty 근처의 조건만 남긴다.

    왜 필요한가 (13 주차에 찾은 구멍) : 데이터셋의 피치는 전 범위
    (100~400 nm)이고 추천기는 **한 피치**를 묻는다. DOF 는 피치에 강하게
    의존하므로 전 범위에서 모은 잔차는 한 피치의 잔차와 같은 분포가 아니다.
    교환가능성이 깨지면 적합예측의 보장도 깨진다.

    피치를 **덮어쓰지 않는** 이유 : 덮어쓰면 그 조건이 더 이상 검증 분할의
    그룹이 아니다. 분할은 피치를 포함한 그룹 키로 나뉘므로 피치를 바꾸면
    학습에 쓰인 그룹과 겹칠 수 있다. 표본이 줄어드는 대가를 치르고 누설
    가능성을 0 으로 둔다.

    `pitch_window <= 0` 이면 제한하지 않는다. 표본이 모자랄 때 마지못해
    쓰는 설정이며, 부르는 쪽이 그 사실을 적어야 한다.
    """
    if pitch_window <= 0:
        return list(conds)
    out = []
    for c in conds:
        p_ = float(c.get("pitch", pitch))
        d_ = float(c.get("duty", duty))
        if (abs(p_ / pitch - 1.0) <= pitch_window
                and abs(d_ - duty) <= duty_window):
            out.append(c)
    return out


def residual_quantile(pred: np.ndarray, true: np.ndarray,
                      alpha: float = 0.10) -> dict:
    """예측·참 DOF 짝에서 적합예측 분위수와 편향 방향을 낸다.

    `signed_median` 이 양수면 모형이 DOF 를 **크게** 본다(낙관적). 13 주차
    되돌림이 모든 조합에서 중위 차 <= 0 을 보고했고, 그 방향성과 대조할
    수 있어야 하므로 함께 돌려준다.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    p, t = p[m], t[m]
    out = conformal_quantile(np.abs(p - t), alpha)
    out["alpha"] = float(alpha)
    out["signed_median"] = (float(np.median(p - t)) if p.size else None)
    out["note"] = ("참 DOF 는 시뮬레이터가 **추천기와 같은 격자에서** 낸 "
                   "값이다. 보장은 시뮬레이터 대비이며 웨이퍼 대비가 아니다.")
    return out


def coverage(pred: np.ndarray, true: np.ndarray, q: float) -> dict:
    """주어진 `q` 로 **다른 분할**에서 실측 피복률을 센다.

    보정에 쓴 분할에서 재면 당연히 맞게 나오고 그것은 검증이 아니다.
    보정은 검증 분할, 피복률 확인은 시험 분할로 가른다.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(t)
    if q is None or not m.any():
        return {"n": 0, "hit": 0, "rate": None}
    hit = int(np.count_nonzero(np.abs(p[m] - t[m]) <= float(q) + 1e-9))
    n = int(m.sum())
    return {"n": n, "hit": hit, "rate": hit / n}
