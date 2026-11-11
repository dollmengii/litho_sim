"""13주차 되돌림 검증 : 추천 조건을 시뮬레이터로 되돌려 직접 재기.

왜 이 파일이 있는가
-------------------
11주차가 측정한 대리 모형 오차는 **데이터와 같은 분포에서 뽑은 조건**에 대한
값이다. 그런데 추천기가 내놓는 조건은 데이터에 없는 점일 수 있다. 그 점에서의
오차가 추천기의 실제 오차이고, 그것은 추천 조건을 시뮬레이터에 되돌려 넣어야만
재진다. 11주차 학습 출력이 이 파일을 직접 가리킨다.

12주차는 "보수 공차가 음수라 인증은 못 하지만 **순위는 쓸 수 있다**" 로
끝났다. 근거는 "비교에서는 모형 오차가 부분적으로 상쇄된다" 였고, 그것은
**측정되지 않은 주장**이었다. 이 파일의 중심 판정은 거기에 답하는 것이다.
순위가 살아남지 않으면 12주차 결론을 고쳐야 한다.

무엇을 재는가
-------------
1. DOF 하한      : 대리 모형 예측 대 시뮬레이터 실측
2. 운전점        : 노광 배율과 초점 오프셋
3. **순위 보존**  : 대리 모형 순서와 시뮬레이터 순서의 상관
4. 구간 포함     : 대리 모형이 내놓은 [하한, 상한] 이 실측을 담는가

규약을 맞추는 것이 이 검증의 전제다
-----------------------------------
대리 모형이 배운 것은 `cd_error = cd_mean - target_cd` 이고 `cd_mean` 은
몬테카를로 평균 CD 다. 데이터 생성 경로(`litho/dataset.py`)를 그대로 옮기면

    e_op     = dose_to_size( 공중상(defocus=0),  블러 없음 )
    observed = 공중상(defocus) * e_op * dose_rel
    cd_mean  = monte_carlo(observed, blur_nm=...)   <- 블러 + 잡음
    nils     = extract_cd(observed)                 <- 블러 **없음**

세 가지가 중요하다.

(가) `e_op` 는 **블러 없는** 공중상에서 구한다. 즉 `dose_rel = 1.0` 이
     블러 포함 CD 로 목표를 맞추는 점이 아니다. 노광 축의 원점이 그렇게
     정의되어 있으므로 되돌림도 같아야 한다. 실측 : 블러 유무로 `e_op` 가
     2.0~2.6 % 다르고, 그만큼 노광 중심이 1.5 %p 까지 밀린다.

(나) CD 는 **블러를 넣어야** 한다. `app.py` 의 공정 윈도 패널은 블러를 넣지
     않으므로(`cd_matrix` 가 공중상 CD 를 쓴다) 그 규약으로 비교하면 1.6 %
     의 계통 차이가 모형 오차로 기록된다.

(다) NILS 는 **블러를 넣지 않는다.** 데이터가 공중상 NILS 를 라벨로 썼다.

몬테카를로를 돌리지 않는 이유와 그 정당화
-----------------------------------------
`blur_nm > 0` 이면 확률 경로는 `gaussian_filter(counts, sigma=(b/dy, b/dx),
mode="wrap")` 로 2 차원 블러를 한다. 그런데 기대값은

    E[ blur_2d( Poisson(lambda) ) ] = blur_2d( lambda ) = blur_x( lambda )

이다. `lambda` 가 y 방향으로 **일정**하기 때문이다(공중상은 x 의 함수이고
행마다 같다). 그래서 1 차원 블러를 넣은 결정론 상에서 CD 를 뽑으면 몬테카를로
평균과 같아야 한다. 블러는 선형이므로 세기 묶음을 **한 번만** 흐려 놓고
노광량을 곱하면 되고, 노광 격자 전체가 공짜가 된다.

이 등식은 유도이고, 유도는 틀릴 수 있다. 그래서 `--mc-check` 가 몇 개 점에서
**실제 몬테카를로를 돌려 대조**한다. 몬테카를로 표준오차 안에서 맞지 않으면
그 사실을 보고하고 종료 코드 1 을 돌려준다. 지름길이 조용히 깨지면 되돌림
전체가 무효이므로, 통과를 가정하지 않는다.

과공차 영역을 반드시 포함한다
-----------------------------
11주차 학습 출력이 "13 주차 되돌림 검증은 이 영역을 반드시 포함해야 한다.
쉬운 영역만 넣으면 추천기가 실제보다 좋아 보인다" 고 적었다. `--tool-preset
all` 이 학습 범위의 상단에서 공구 상태를 잡아 그 영역을 돌린다. 값을 여기에
적어 두지 않고 **번들의 학습 범위에서 읽는다** — 적어 두면 데이터를 바꿀
때마다 이 파일만 낡는다.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

import recommend as RC
from litho import dataset as DS
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import resist as RS
from litho import stochastic as ST


# ---------------------------------------------------------------------------
# 시뮬레이터 쪽 측정
# ---------------------------------------------------------------------------
def _setup(spec: DS.SampleSpec):
    """격자, 마스크, 광학 인자를 **데이터 생성 경로와 똑같이** 만든다.

    인자를 손으로 다시 적지 않고 `spec.optics()` 와 `spec.n_periods`,
    `spec.dx_target` 를 쓴다. 손으로 적으면 축이 하나 추가될 때 이쪽만 낡아
    규약이 어긋나고, 그 어긋남이 '모형 오차' 로 기록된다. 되돌림은 규약이
    같다는 전제 위에 서 있으므로 그 전제를 코드로 세운다.
    """
    g = G.build_grid(pitch=spec.pitch, n_periods=spec.n_periods,
                     dx_target=spec.dx_target)
    target = spec.duty * spec.pitch
    m = MK.line_space(g, cd=target, mask_type=spec.mask_type,
                      att_t=spec.att_t)
    return g, m, target, spec.optics()


def simulate_window(spec: DS.SampleSpec, focus: np.ndarray,
                    dose_rel: np.ndarray, el_target: float,
                    tol: float) -> dict:
    """한 조건의 공정 윈도를 시뮬레이터로 직접 재서 추천기와 같은 모양으로.

    규약은 모듈 설명의 (가)(나)(다) 를 따른다. 돌려주는 사전의 키는
    `recommend.window_metrics` 와 같게 맞춘다. 비교할 때 키가 다르면 둘을
    나란히 놓을 수 없다.
    """
    g, m, target, opt = _setup(spec)

    # (가) e_op 는 블러 없는 defocus=0 공중상에서
    i0 = IM.aerial_image(g, m, spec.wavelength, spec.na, defocus=0.0, **opt)
    e_op = PR.dose_to_size(g, i0, target, spec.threshold, max_width=spec.pitch)
    if not np.isfinite(e_op):
        return {"ok": False, "reason": "dose-to-size 를 찾지 못함",
                "e_op": float("nan")}

    stack = PR.focus_sweep(g, m, spec.wavelength, spec.na,
                           defocuses=np.asarray(focus, dtype=np.float64),
                           **opt)
    # (나) CD 용 묶음은 블러를 넣는다. 블러는 선형이라 한 번만 흐려 두면
    #      노광량을 곱해도 같다 : blur(I*d) = d*blur(I).
    if spec.blur_nm > 0:
        stack_b = gaussian_filter1d(stack, spec.blur_nm / g.dx, axis=1,
                                    mode="wrap")
    else:
        stack_b = stack

    nz, nd = len(focus), len(dose_rel)
    cd = np.full((nd, nz), np.nan)
    nils = np.full((nd, nz), np.nan)
    for j in range(nz):
        base_b, base_r = stack_b[j], stack[j]
        for i, rel in enumerate(dose_rel):
            d = e_op * float(rel)
            rb = RS.extract_cd(g.x, base_b * d, spec.threshold,
                               max_width=spec.pitch)
            if rb["formed"] and np.isfinite(rb["cd"]):
                cd[i, j] = rb["cd"]
            # (다) NILS 는 블러 없는 상에서
            rr = RS.extract_cd(g.x, base_r * d, spec.threshold,
                               max_width=spec.pitch)
            if rr["formed"] and np.isfinite(rr["nils"]):
                nils[i, j] = rr["nils"]

    pw = PR.process_window(cd, target, tol_frac=tol)
    curve = PR.el_dof_curve(pw, np.asarray(dose_rel, dtype=np.float64),
                            np.asarray(focus, dtype=np.float64))
    det = PR.dof_bracket_detail(curve, el_target)
    row = det["lo_row"]
    sat = PR.window_saturated(pw)

    if row is not None:
        d_lo, d_hi = row["dose_range"]
        z_lo, z_hi = row["focus_range"]
        dose_c = 0.5 * (float(d_lo) + float(d_hi))
        focus_c = 0.5 * (float(z_lo) + float(z_hi))
        rng_d = (float(d_lo), float(d_hi))
        rng_z = (float(z_lo), float(z_hi))
        _f = np.asarray(focus, dtype=np.float64)
        _d = np.asarray(dose_rel, dtype=np.float64)
        rfe = bool(abs(z_lo - _f.min()) <= 1e-9 or abs(z_hi - _f.max()) <= 1e-9)
        rde = bool(abs(d_lo - _d.min()) <= 1e-9 or abs(d_hi - _d.max()) <= 1e-9)
    else:
        dose_c = focus_c = None
        rng_d = rng_z = None
        rfe = rde = False

    nmin = (float(np.nanmin(np.where(pw, nils, np.nan)))
            if np.any(pw) and np.any(np.isfinite(nils)) else float("nan"))
    return {
        "ok": True, "e_op": float(e_op),
        "dof": float(det["lo"]), "dof_upper": float(det["hi"]),
        "total": float(PR.total_window(curve)),
        "n_pass": int(np.count_nonzero(pw)),
        "focus_edge": bool(sat["focus_edge"]),
        "dose_edge": bool(sat["dose_edge"]),
        "row_focus_edge": rfe, "row_dose_edge": rde,
        "dose_center": dose_c, "focus_center": focus_c,
        "dose_range": rng_d, "focus_range": rng_z,
        "nils_min": nmin,
        "nils_ok": bool(np.isfinite(nmin) and nmin >= spec.min_nils),
        # 실측 DOF 0 에는 두 가지가 섞인다. (가) 정말 공정창이 없다,
        # (나) 하한 격자점에서 EL 목표를 못 넘겨 상한만 살아 있다(격자
        # 미결정). 둘을 구분하지 않으면 (나) 를 '모형이 낙관적이었다' 로
        # 잘못 적는다. `recommend.window_metrics` 와 같은 이름을 쓴다.
        "undetermined": bool(float(det["lo"]) <= 0.0 < float(det["hi"])),
        "tol": float(tol),
        "cd": cd, "pw": pw,
        # CDU 측정이 쓰도록 함께 돌려준다. 다시 계산하면 초점 스윕과
        # NILS 추출을 후보마다 두 번 하게 되고, 그것이 실제 실행에서
        # 후보당 비용을 두 배로 만든다. 크기는 작다(초점 수 x 격자 x 8 B).
        "nils": nils, "stack": stack, "e_op_val": float(e_op),
    }


def measure_cdu(spec: DS.SampleSpec, focus, dose_rel, sim: dict,
                n_points: int = 8, n_rows: int = 256, n_trials: int = 4,
                seed: int = 0) -> dict:
    """실측 CDU 를 몬테카를로로 잰다. 추천기와 **같은 정의**로.

    정의를 맞추는 것이 전부다. 추천기는
    `cdu_max = max over window of (cd_3sigma / cd)` 를 `stoch_tol` 과
    비교한다(`recommend.evaluate`). 라벨 `cd_3sigma` 는 데이터셋에서
    `ST.monte_carlo` 의 `cd_3sigma` 이고 짝이 되는 CD 는 같은 호출의
    `cd_mean` 이다. 그래서 여기서도 `mc["cd_3sigma"] / mc["cd_mean"]` 을
    쓴다. 결정론 CD 로 나누면 분모가 다른 양이 되어 비교가 흐려진다.

    왜 전수가 아니라 표본인가 (비용과 그 대가를 그대로 적는다)
    --------------------------------------------------------
    창 하나에 통과 격자점이 100 개 규모이고, 생성 경로 설정
    (`n_rows=1024`, `n_trials=8`)에서 한 점이 **6.2 s** 다. 후보 15 x 공구
    6 x 모형 3 이면 46 시간이다. `n_rows=256`, `n_trials=4` 로 낮추면 한
    점이 0.68 s 지만 여전히 5 시간이다.

    그래서 **최악일 법한 지점을 겨냥해** 표본을 뽑는다. CDU 는 대체로
    NILS 가 낮은 곳에서 나쁘므로(기울기가 완만하면 같은 광자 잡음이 더 큰
    CD 변동이 된다), 창 안에서 NILS 가 낮은 순으로 뽑고 창의 네 극점을
    더한다.

    **대가** : 이렇게 얻은 값은 창 전체 최댓값의 **하한**이다. 따라서
    시뮬레이터 쪽 `cdu_ok` 는 '인증' 이 아니라 '반증되지 않음' 이다. 그
    구분을 출력에 적는다. 겨냥이 맞는지도 가정으로 두지 않고 **재서**
    확인한다 : 표본 중 CDU 최댓값이 NILS 최소 지점에서 나왔는지를 세고,
    어긋나면 그 사실을 적는다.
    """
    pw, nils = sim.get("pw"), None
    if pw is None or not np.any(pw):
        return {"ok": False, "why": "창이 없다"}
    g, _m, _t, _o = _setup(spec)
    f = np.asarray(focus, dtype=np.float64)
    d = np.asarray(dose_rel, dtype=np.float64)
    # 초점 스윕·NILS·e_op 는 `simulate_window` 가 이미 계산했다. 다시 하면
    # 후보당 비용이 두 배가 된다. 없으면(옛 반환) 그때만 다시 만든다.
    stack = sim.get("stack")
    nils = sim.get("nils")
    e_op = sim.get("e_op_val", sim.get("e_op"))
    if stack is None or nils is None or not np.isfinite(e_op or np.nan):
        return {"ok": False,
                "why": "simulate_window 가 stack/nils/e_op 를 주지 않았다"}

    ij = np.argwhere(pw)
    # 창의 네 극점 : 초점 최소/최대, 노광 최소/최대
    picks = []
    for key in (lambda p: p[1], lambda p: -p[1],
                lambda p: p[0], lambda p: -p[0]):
        picks.append(tuple(min(map(tuple, ij), key=key)))
    # NILS 가 낮은 순
    order = sorted(map(tuple, ij),
                   key=lambda p: (np.inf if not np.isfinite(nils[p])
                                  else nils[p]))
    nils_min_pt = order[0] if order else None
    for p in order:
        if len(picks) >= max(n_points, 4):
            break
        if p not in picks:
            picks.append(p)
    picks = list(dict.fromkeys(picks))

    rng = np.random.default_rng(seed)
    rows, worst, worst_pt = [], -1.0, None
    for (i, j) in picks:
        obs = stack[j] * e_op * float(d[i])
        mc = ST.monte_carlo(g.x, obs, g.dx, spec.dose_mj, spec.wavelength,
                            spec.threshold, n_rows=n_rows,
                            blur_nm=spec.blur_nm, n_trials=n_trials,
                            seed=int(rng.integers(1 << 30)), dy=spec.dy,
                            max_width=spec.pitch)
        if not (mc["formed"] and mc["cd_mean"] > 0):
            continue
        cdu = float(mc["cd_3sigma"] / mc["cd_mean"])
        rows.append({"focus": float(f[j]), "dose_rel": float(d[i]),
                     "cdu": cdu, "nils": float(nils[i, j]),
                     "is_nils_min": (i, j) == nils_min_pt})
        if cdu > worst:
            worst, worst_pt = cdu, (i, j)
    if not rows:
        return {"ok": False, "why": "몬테카를로에서 패턴이 형성되지 않았다"}

    # 표집 잡음의 크기. 같은 지점을 다른 씨앗으로 한 번 더 재서 재현성을
    # 본다. 이것 없이 '최악 CDU' 를 적으면 잡음을 신호로 읽을 수 있다.
    i, j = worst_pt
    mc2 = ST.monte_carlo(g.x, stack[j] * e_op * float(d[i]), g.dx,
                         spec.dose_mj, spec.wavelength, spec.threshold,
                         n_rows=n_rows, blur_nm=spec.blur_nm,
                         n_trials=n_trials, seed=int(rng.integers(1 << 30)),
                         dy=spec.dy, max_width=spec.pitch)
    rep = (float(mc2["cd_3sigma"] / mc2["cd_mean"])
           if mc2["formed"] and mc2["cd_mean"] > 0 else float("nan"))
    return {"ok": True, "rows": rows, "n_sampled": len(rows),
            "n_window": int(np.count_nonzero(pw)),
            "cdu_max": worst, "cdu_repeat": rep,
            "stoch_tol": float(spec.stoch_tol),
            "frac_of_tol": worst / float(spec.stoch_tol),
            "argmax_at_nils_min": bool(worst_pt == nils_min_pt),
            "n_rows": int(n_rows), "n_trials": int(n_trials)}


def mc_crosscheck(spec: DS.SampleSpec, n_points: int,
                  seed: int = 0) -> dict:
    """블러 1 차원 지름길을 **실제 몬테카를로**로 대조한다.

    되돌림 전체가 `E[blur_2d(Poisson)] = blur_x(lambda)` 라는 등식에 기대고
    있다. 유도는 맞아 보이지만 유도가 맞는지는 재 봐야 안다. 여기서 틀리면
    되돌림 숫자가 전부 무효이므로, 통과를 가정하지 않고 종료 코드로 알린다.

    대조 기준을 **공정 공차 대비 상대 크기**로 둔다. 처음에는 몬테카를로
    표준오차의 배수로 잡았는데, 그 표준오차가 행 사이 독립을 가정해 지나치게
    작았다. 2 차원 블러가 인접 행을 섞으므로 독립 표본 수가 `n_rows x
    n_trials` 보다 훨씬 적다. 실측 : 차이가 CD 의 0.01 % 인데 표준오차의
    9.17 배로 나와 불합격 판정이 났다. 공차가 7.07 % 인 판정에서 0.01 %
    차이를 불합격으로 두는 것은 기준이 틀린 것이다.

    그래서 '지름길이 만드는 CD 오차가 **공차의 몇 %인가**' 로 판정한다.
    공차가 이 프로젝트의 모든 판정 단위이므로 기준도 거기에 맞춘다.
    표준오차 배수는 참고로 함께 인쇄한다.
    """
    inf = float("inf")
    g, m, target, opt = _setup(spec)
    i0 = IM.aerial_image(g, m, spec.wavelength, spec.na, defocus=0.0, **opt)
    e_op = PR.dose_to_size(g, i0, target, spec.threshold, max_width=spec.pitch)
    if not np.isfinite(e_op):
        return {"ok": False, "reason": "dose-to-size 를 찾지 못함", "rows": []}

    rng = np.random.default_rng(seed)
    dzs = rng.uniform(-150.0, 150.0, n_points)
    rels = rng.uniform(0.92, 1.08, n_points)
    rows, worst = [], 0.0
    for dz, rel in zip(dzs, rels):
        inten = (i0 if dz == 0.0 else
                 IM.aerial_image(g, m, spec.wavelength, spec.na,
                                 defocus=float(dz), **opt))
        observed = inten * e_op * float(rel)
        # 지름길 : 1 차원 블러 + 결정론 추출
        if spec.blur_nm > 0:
            obs_b = gaussian_filter1d(observed, spec.blur_nm / g.dx,
                                      mode="wrap")
        else:
            obs_b = observed
        short = RS.extract_cd(g.x, obs_b, spec.threshold,
                              max_width=spec.pitch)
        # 참값 : 실제 몬테카를로
        mc = ST.monte_carlo(g.x, observed, g.dx, spec.dose_mj, spec.wavelength,
                            spec.threshold, n_rows=spec.n_rows,
                            blur_nm=spec.blur_nm, n_trials=spec.n_trials,
                            seed=int(rng.integers(1 << 30)), dy=spec.dy,
                            max_width=spec.pitch)
        if not (short["formed"] and mc["formed"]):
            continue
        # 표준오차 : 시행 사이 산포를 반복 수로 나눈 것. cd_3sigma 는 3 시그마
        # 이므로 3 으로 나눠 1 시그마로 되돌린 뒤 사용한다.
        # 표준오차. **낙관적인 값이다.** 행 사이 상관 0 을 가정하는데,
        # 2 차원 블러가 sigma = blur_nm/dy 로 인접 행을 섞으므로 독립 표본
        # 수가 n_rows x n_trials 보다 훨씬 적다. `litho/dataset.py` 의
        # 라벨 잡음 절이 같은 경고를 한다. 그래서 이 배수는 **판정 기준이
        # 아니라 참고**로만 쓴다. 처음에 이것으로 합격선을 잡았더니 차이가
        # CD 의 0.01 % 인데도 9.17 배로 불합격이 났다.
        se = (mc["cd_3sigma"] / 3.0) / np.sqrt(
            max(spec.n_rows * spec.n_trials, 1))
        diff = float(short["cd"] - mc["cd_mean"])
        ratio = abs(diff) / se if se > 0 else float("inf")
        # 판정 기준은 **공정 공차에 대한 상대 크기**다. 지름길이 만드는
        # CD 오차가 공차의 몇 %인지를 본다. 공차가 판정의 단위이기 때문이다.
        rel_tol = abs(diff) / target / spec.pw_tol if target > 0 else inf
        worst = max(worst, rel_tol)
        rows.append({"dz": float(dz), "rel": float(rel),
                     "short": float(short["cd"]),
                     "mc": float(mc["cd_mean"]), "diff": diff,
                     "se": float(se), "ratio": float(ratio),
                     "rel_tol": float(rel_tol)})
    return {"ok": True, "rows": rows, "worst_rel_tol": worst,
            "pw_tol": float(spec.pw_tol)}


# ---------------------------------------------------------------------------
# 순위 상관
# ---------------------------------------------------------------------------
def rank_corr(a: list[float], b: list[float]) -> dict:
    """스피어만과 켄달 타우. 동률이 많으므로 둘 다 본다.

    왜 둘인가 : 포화나 격자 때문에 DOF 가 동률로 묶이는 일이 흔하다.
    스피어만은 동률을 평균 순위로 처리해 값이 부풀 수 있고, 켄달 타우-b 는
    동률을 분모에서 빼므로 더 보수적이다. 한쪽만 보면 과대평가한다.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = x.size
    if n < 3:
        return {"n": int(n), "spearman": float("nan"),
                "kendall": float("nan"), "pairs": 0, "why": "후보가 3 개 미만"}
    # 한쪽이 상수면 상관은 정의되지 않는다. nan 만 찍고 넘어가면 '검사가
    # 깨졌다' 로 읽히므로 이유를 함께 돌려준다. 예측 DOF 가 전부 0 인 경우
    # (윈도가 아예 없는 요청)에 실제로 일어난다.
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        which = "예측" if np.ptp(x) == 0.0 else "실측"
        return {"n": int(n), "spearman": float("nan"),
                "kendall": float("nan"), "pairs": 0,
                "why": f"{which} DOF 가 전부 같은 값이라 순위가 없음"}
    from scipy.stats import kendalltau, spearmanr
    sp = spearmanr(x, y).statistic
    kt = kendalltau(x, y, variant="b").statistic
    return {"n": int(n), "spearman": float(sp), "kendall": float(kt),
            "pairs": int(n * (n - 1) // 2)}


def top_k_overlap(sur: list[int], sim: list[int], k: int) -> float:
    """상위 k 개가 얼마나 겹치는가. 실무에서는 이게 순위 상관보다 중요하다.

    공정 엔지니어는 전체 순서를 쓰지 않고 상위 몇 개만 본다. 전체 상관이
    높아도 상위권이 바뀌면 쓸 수 없고, 반대로 전체 상관이 낮아도 상위권이
    같으면 쓸 수 있다.
    """
    if k <= 0 or not sur or not sim:
        return float("nan")
    k = min(k, len(sur), len(sim))
    return len(set(sur[:k]) & set(sim[:k])) / float(k)


# ---------------------------------------------------------------------------
# 공구 상태 사전 설정
# ---------------------------------------------------------------------------
def common_axes(domains: list[dict]) -> tuple[dict, list[str]]:
    """여러 번들의 학습 범위 **교집합**. 모형 비교의 조건을 하나로 맞춘다.

    왜 교집합인가 (13주차에 찾은 결함)
    ---------------------------------
    공구 상태를 번들마다 그 번들의 범위에서 읽으면 '`z6` 상단 90 %' 가
    모형마다 **다른 물리 조건**이 된다. 보강 전 번들은 `z11` 이 양수뿐이고
    보강 후는 `(-0.05, 0.05)` 라서, 같은 이름의 상태가 0.0429 와 0.0400 처럼
    갈린다. 그 상태에서 잰 상관을 모형 사이에 나란히 적으면 **조건 차이가
    모형 차이로 읽힌다.**

    교집합을 쓰면 모든 모형이 자기 학습 범위 안에서 평가된다. 즉 비교가
    '모형이 얼마나 좋은가' 를 재고, '누가 외삽했는가' 를 재지 않는다.

    대가 : 보강으로 **넓힌 영역**(음수 `z11`)은 여기서 탐침되지 않는다. 그
    영역은 다른 번들의 학습 범위 밖이므로, 넣으면 내삽과 외삽을 비교하게
    된다. 넓힌 영역은 `--models models/s13c` 처럼 **한 모형만** 돌려서 보고,
    그 사실을 한계로 적는다.
    """
    keys = set(domains[0]["axes"])
    for d in domains[1:]:
        keys &= set(d["axes"])
    ax, notes = {}, []
    for k in sorted(keys):
        lo = max(float(d["axes"][k]["min"]) for d in domains)
        hi = min(float(d["axes"][k]["max"]) for d in domains)
        if lo > hi:
            notes.append(f"축 '{k}' 의 학습 범위가 모형 사이에 겹치지 "
                         f"않는다(교집합 공집합). 이 축은 비교에서 뺀다.")
            continue
        ax[k] = {"min": lo, "max": hi}
        for d in domains:
            w = float(d["axes"][k]["max"]) - float(d["axes"][k]["min"])
            if w > (hi - lo) * 1.001:
                notes.append(f"축 '{k}' : 어떤 모형은 더 넓게 학습했으나 "
                             f"[{lo:.4g}, {hi:.4g}] 로 좁혀 비교한다")
                break
    dropped = sorted(set().union(*(set(d["axes"]) for d in domains)) - keys)
    if dropped:
        notes.append("모든 모형에 공통이 아닌 축은 제외 : "
                     + ", ".join(dropped))
    return {"axes": ax}, notes


#: `판정기준_동결.md` 의 기준값. **실제 모형으로 돌리기 전에** 정해졌고
#: 근거는 전부 구조적이다(타우-b 와 확률의 환산, 상위 k 겹침의 의미,
#: 클로퍼-피어슨 하한의 성질). 숫자를 본 뒤에 고치면 그 문서의 변경 기록에
#: 적어야 하고, 그 경우 판정에 "사후선택이 섞였다" 를 달아야 한다.
#:
#: 왜 코드에 두는가 : 문서에만 있으면 보고서를 쓸 때 사람이 기억으로
#: 판정하게 된다. 코드가 판정해서 인쇄하면 기준과 결과가 어긋날 수 없다.
CRITERIA = {
    "tau_min": 0.60,        # (A) 켄달 타우-b. tau=0.60 <-> 두 후보 비교
                            #     정확도 p=0.80 (tau = 2p - 1)
    "n_min": 6,             # (A) 최소 후보 수. 아래 유도 참조
    "null_q": 0.95,         # (A) 귀무분포 분위. 관측이 이보다 커야 한다
    "top3_min": 2.0 / 3.0,  # (B) 상위 3 겹침(DOF 만). 다수가 겹쳐야 목록이다
    "cover_slack": 0.10,    # (C) 실측 피복 하한 >= 보장 - 0.10
}

# `n_min = 6` 과 `null_q` 의 유도 (13 주차 보완 92 번에서 추가)
# ---------------------------------------------------------------------------
# 임계값만 동결하고 **검정력을 동결하지 않은** 것이 구멍이었다. 켄달 타우-b
# 의 귀무 표준편차는 sd(n) = sqrt(2(2n+5) / (9n(n-1))) 이다. 단측 5 % 에
# 필요한 값이 1.645 * sd(n) 이므로
#
#     n = 5 -> 1.645 * 0.408 = 0.671  > 0.60   기준이 우연보다 **약하다**
#     n = 6 -> 1.645 * 0.355 = 0.584 <= 0.60   기준이 비로소 더 엄격하다
#
# 즉 6 은 '동결한 0.60 이 우연선보다 엄격해지는 가장 작은 n' 이다. 관측값을
# 보고 고른 수가 아니라 기준 0.60 **자체에서** 나오는 수다. 그래도 데이터를
# 본 뒤에 추가한 것은 사실이므로 판정기준_동결.md 변경 기록에 그대로 적는다.
#
# n >= 6 만으로는 모자라다. 동률이 있으면 통계량이 가질 수 있는 값의 수가
# 줄어 검정력이 더 떨어진다. 실측 : 후보 3 개에 예측 동률 1 쌍이면 스피어만
# 이 +0.866 / 0 / -0.866 셋만 가능하고, 그 최댓값 +0.866 이 우연히 나올
# 확률이 **33.3 %** 다. 코드가 그 귀무분포를 찍어 놓고도 판정은 고정
# 임계값만 보아 '충족' 을 내놓았다. 그래서 **순열 귀무분포와의 대조**를
# 함께 요구한다. 두 요구는 서로를 대신하지 못한다.


def perm_null(sur, sim, n_perm: int = 2000, seed: int = 0) -> dict | None:
    """순열 귀무분포. 실측 순서를 섞어 '우연히 이만큼 나올 수 있는가' 를 본다.

    왜 고정 임계값만으로는 안 되는가
    --------------------------------
    동결한 기준 A 는 '켄달 타우-b >= 0.60' 이다. 그 값은 두 후보를 올바른
    순서로 놓을 확률 0.80 에 대응하므로 **뜻이 분명한** 기준이다. 그런데
    후보가 적거나 동률이 많으면 통계량이 가질 수 있는 값 자체가 적어져서,
    무작위 순서도 그 임계값을 쉽게 넘는다. 실측된 중심 판정 칸이 그랬다 :
    후보 3 개에 예측 동률 1 쌍 -> 스피어만은 +0.866 / 0 / -0.866 셋만
    가능하고, 관측된 +0.866 은 **최댓값**이면서 우연 확률이 33.3 % 였다.
    임계값은 '얼마나 잘 맞혔나' 를 재고, 귀무분포는 '그게 우연이 아닌가' 를
    잰다. 둘은 다른 질문이므로 둘 다 물어야 한다.

    **작은 n 은 완전 열거한다.** n <= 7 이면 순열이 5040 가지 이하이므로
    전부 세어 정확한 분포를 얻는다. 표집으로 때우면 바로 이 구간에서
    표집 잡음이 판정을 흔든다 — 그런 잡음을 판정에 쓰는 실수를 이 과제에서
    이미 두 번 했다(몬테카를로 표준오차, 적합예측 피복).
    """
    sur = list(sur)
    sim = list(sim)
    n = len(sur)
    if n < 3 or len(sim) != n:
        return None
    if len(set(sim)) < 2 or len(set(sur)) < 2:
        return None          # 한쪽이 상수면 상관 자체가 정의되지 않는다
    if n <= 7:
        perms = list(itertools.permutations(sim))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        perms = [tuple(rng.permutation(sim)) for _ in range(n_perm)]
        exact = False
    taus, sps = [], []
    for p in perms:
        r = rank_corr(sur, list(p))
        if np.isfinite(r["kendall"]):
            taus.append(r["kendall"])
        if np.isfinite(r["spearman"]):
            sps.append(r["spearman"])
    if not taus:
        return None
    obs = rank_corr(sur, sim)
    taus = np.asarray(taus, dtype=np.float64)
    out = {
        "exact": exact,
        "n_perm": len(perms),
        "tau_q": float(np.quantile(taus, CRITERIA["null_q"])),
        "tau_median": float(np.median(taus)),
        # 관측 이상이 우연히 나올 확률. 열거면 정확한 값이다.
        "tau_p": float(np.mean(taus >= obs["kendall"] - 1e-12))
        if np.isfinite(obs["kendall"]) else None,
        "n_distinct_tau": int(len(set(np.round(taus, 9)))),
    }
    if sps:
        sps = np.asarray(sps, dtype=np.float64)
        out["sp_q"] = float(np.quantile(sps, CRITERIA["null_q"]))
        out["sp_median"] = float(np.median(sps))
    return out


def judge(tau, top3, cover_lo, guarantee, n_cand=None, null=None,
          n_dist=None, dof_span=None):
    """세 기준을 각각 판정한다. (이름, 판정, 근거문) 목록.

    **부분 충족을 '성공' 으로 묶지 않는다.** A·B·C 는 다른 질문이고
    (순서가 맞는가 / 상위권을 고를 수 있는가 / 값에 붙인 구간이 맞는가),
    둘이 충족되고 하나가 미달이면 그 하나가 미달이라고 적어야 한다.
    """
    out = []
    c = CRITERIA
    if tau is None or not np.isfinite(tau):
        out.append(("A 순위 보존", "잴 수 없음",
                    "후보가 모자라거나 입력이 상수다"))
    elif n_cand is not None and n_cand < c["n_min"]:
        # 임계값을 넘었는지 **적지 않는다.** 적으면 읽는 사람이 그것을
        # 충족으로 읽는다. 검정력이 없으면 질문 자체에 답할 수 없다.
        out.append(("A 순위 보존", "잴 수 없음",
                    f"후보 {n_cand} 개 < 최소 {c['n_min']} 개. 이 크기에서는 "
                    f"기준 {c['tau_min']:.2f} 가 우연선보다 약하다"))
    elif (n_dist is not None and n_cand is not None
          and n_dist * 2 < n_cand):
        # 동결 문서가 **데이터 전에** 요구한 규칙이다 : "서로 다른 값이 후보
        # 수의 절반 미만이면 타우 자체의 분해능이 낮으므로 ... 격자를
        # 촘촘히 해 다시 재야 한다." 전에는 그 문장을 따로 인쇄만 하고
        # 판정에는 반영하지 않았다. 그래서 분해능이 모자란 칸에 '미달' 이
        # 찍혔다.
        #
        # 왜 미달이 아니라 보류인가 — **재서 확인했다.** DOF 를 눈금에
        # 맞추면 타우가 깎인다. 참 스피어만 0.70, 후보 15 개, 눈금
        # 37.5 nm 에서 관측 켄달의 중위는 +0.558 이고 5~95 백분위가
        # +0.19 ~ +0.79 다. 즉 관측 +0.445 는 **참 타우 0.60 이상과도
        # 양립한다.** 그 상태에서 '미달' 을 적으면 분해능 부족을 모형의
        # 결함으로 잘못 옮긴다.
        #
        # 보류는 느슨한 판정이 아니다. 12 주차 주장을 **확인하지도
        # 반증하지도 못했다**는 뜻이고, 충족보다도 약한 결과다.
        extra = (f", 통과군 실측 DOF 폭 {dof_span:.1f} nm"
                 if dof_span is not None else "")
        out.append(("A 순위 보존", "잴 수 없음",
                    f"서로 다른 DOF 값 {n_dist} 개 < 후보 {n_cand} 개의 절반"
                    f"{extra}. 눈금이 타우를 깎으므로 미달로 읽을 수 없다"))
    else:
        base = (f"켄달 타우-b {tau:+.3f} 대 기준 {c['tau_min']:.2f} "
                f"(두 후보 비교 정확도 {50 * (1 + tau):.0f} % 대 80 %)")
        ok = tau >= c["tau_min"]
        if not ok:
            out.append(("A 순위 보존", "**미달**", base))
        elif null is None:
            out.append(("A 순위 보존", "잴 수 없음",
                        base + " / 귀무분포를 못 구해 우연 여부를 가릴 수 없다"))
        else:
            q, p = null["tau_q"], null.get("tau_p")
            kind = "열거" if null["exact"] else f"표집 {null['n_perm']}"
            if tau <= q + 1e-12:
                out.append(("A 순위 보존", "**미달**",
                            base + f" / 그러나 귀무분포 {100 * c['null_q']:.0f} "
                            f"백분위 {q:+.3f} 를 넘지 못한다"
                            + (f" (우연 확률 {p:.1%}, {kind})" if p is not None
                               else "")))
            else:
                out.append(("A 순위 보존", "충족",
                            base + f" / 귀무분포 {100 * c['null_q']:.0f} "
                            f"백분위 {q:+.3f} 초과"
                            + (f", 우연 확률 {p:.1%} ({kind})"
                               if p is not None else "")))
    if top3 is None or not np.isfinite(top3):
        out.append(("B 상위권 신뢰", "잴 수 없음", "후보 3 개 미만"))
    else:
        ok = top3 >= c["top3_min"] - 1e-9
        out.append(("B 상위권 신뢰", "충족" if ok else "**미달**",
                    f"상위 3 겹침(DOF 만) {top3:.0%} 대 기준 67 %"))
    if cover_lo is None or guarantee is None:
        out.append(("C 구간 타당성", "잴 수 없음",
                    "적합예측 구간이 없거나 후보가 모자라다"))
    else:
        thr = guarantee - c["cover_slack"]
        ok = cover_lo >= thr - 1e-12
        out.append(("C 구간 타당성", "충족" if ok else "**미달**",
                    f"실측 피복 단측 95 % 하한 {100 * cover_lo:.1f} % 대 "
                    f"기준 {100 * thr:.1f} % (보장 {100 * guarantee:.1f} %)"))
    return out


def verdict_text(rows) -> list[str]:
    """`판정기준_동결.md` 3 절의 종합 규칙을 그대로 적용한다."""
    st = {n.split()[0]: v for n, v, _ in rows}
    a, b, cc = st.get("A"), st.get("B"), st.get("C")
    msg = []
    if a == "**미달**":
        msg.append("A 미달 -> **12 주차 주장을 철회한다.** 순위도 쓸 수 "
                   "없으므로 추천기는 '후보 축소 도구' 이지 '추천 도구' 가 "
                   "아니다. 보고서에 그렇게 적어야 한다.")
    elif a == "충족" and b == "**미달**":
        msg.append("A 충족 / B 미달 -> 주장을 **좁힌다.** '순서는 보존되지만 "
                   "상위권을 바로 고를 수는 없다' 이며, 실측 확인을 반드시 "
                   "거치라는 조건을 붙인다.")
    elif a == "충족" and b != "**미달**" and cc == "**미달**":
        msg.append("A·B 충족 / C 미달 -> 순위는 유지하고 **구간만 "
                   "철회한다.** 분포 이동이 보장을 깨뜨렸다는 것 자체가 "
                   "결과이므로 그대로 적는다.")
    elif a == "충족" and b == "충족" and cc == "충족":
        msg.append("A·B·C 모두 충족 -> 12 주차 주장을 **그대로 유지한다.**")
    else:
        msg.append("일부를 잴 수 없다. 잴 수 없는 항목을 '충족' 으로 "
                   "읽지 말 것. 후보 수나 격자를 늘려 다시 재야 한다.")
    return msg


def union_axes(domains: list[dict]) -> dict:
    """여러 번들 학습 범위의 **합집합**. 넓힌 영역을 탐침하기 위한 것.

    왜 합집합도 필요한가 (교집합만 쓰면 보강의 근거를 스스로 버린다)
    ---------------------------------------------------------------
    되돌림이 재는 것은 **모형 대 시뮬레이터**이고 모형 대 모형이 아니다.
    시뮬레이터가 기준이므로, 보강 모형을 그 모형이 학습한 영역에서 재는 데
    다른 모형이 필요하지 않다. 교집합만 쓰면 보강으로 넓힌 영역(음수 `z11`)
    이 아예 측정되지 않고, 그러면 '보강이 필요했다' 는 주장을 뒷받침할
    측정이 없어진다.

    그 영역에서 보강 전 모형은 **외삽**한다. 그것을 숨기지 않고 딱지로 적는
    것이 맞다. 보강 전 모형이 그 영역에서 못 맞히는 것 자체가 보강의 측정
    증거이기 때문이다. 섞어서 한 상관으로 적으면 안 되고, 블록을 갈라
    적는다.
    """
    keys = set()
    for d in domains:
        keys |= set(d["axes"])
    ax = {}
    for k in sorted(keys):
        have = [d["axes"][k] for d in domains if k in d["axes"]]
        ax[k] = {"min": min(float(a["min"]) for a in have),
                 "max": max(float(a["max"]) for a in have)}
    return {"axes": ax}


def outside_axes(tool: dict, domain: dict, tol: float = 1e-9) -> list[str]:
    """이 공구 상태에서 **이 모형의 학습 범위 밖**인 축 이름.

    축이 번들에 없으면 밖이라고 적지 않는다. 그 축을 상수로 생성한 번들이
    그렇고, 그것은 외삽이 아니라 '그 축이 입력이 아니다' 라는 뜻이다.
    """
    out = []
    ax = domain.get("axes", {})
    for k, v in tool.items():
        if k not in ax:
            continue
        lo, hi = float(ax[k]["min"]), float(ax[k]["max"])
        w = max(hi - lo, 1e-12)
        if float(v) < lo - tol * w or float(v) > hi + tol * w:
            out.append(k)
    return sorted(out)


def tool_presets(domain: dict, args) -> list[tuple[str, dict]]:
    """돌릴 공구 상태 목록. 값은 **번들의 학습 범위에서 읽는다.**

    11주차 학습 출력이 z6·z11·보강 영역의 95 백분위가 공차를 넘는다고
    적었고, 되돌림이 그 영역을 반드시 포함해야 한다고 했다. 쉬운 영역만
    넣으면 추천기가 실제보다 좋아 보인다.

    숫자를 이 파일에 적지 않는 이유 : 적어 두면 데이터를 바꿀 때마다 이
    파일만 낡는다. 범위에서 읽으면 어떤 번들에도 맞는다.

    모형이 여럿이면 `domain` 에 `common_axes` 의 교집합을 넘긴다. 그래야
    같은 이름의 공구 상태가 모든 모형에서 같은 물리 조건이 된다.
    """
    ax = domain["axes"]
    base = {"flare": args.flare, "msd_nm": args.msd_nm,
            "blur_nm": args.blur_nm, "z6": args.z6, "z11": args.z11}

    # 축이 없을 수 있다. 모형 교집합에서 빠진 축(모형마다 열 구성이 다른
    # 경우)이나 그 축을 상수로 생성한 번들이 그렇다. 그때는 **요청값을 그대로
    # 쓴다.** 없는 축을 0 으로 채우면 요청하지 않은 조건을 만들어 낸다.
    def hi(k, frac=0.9):
        if k not in ax:
            return float(base[k])
        lo, h = float(ax[k]["min"]), float(ax[k]["max"])
        return lo + frac * (h - lo)

    def mid(k):
        if k not in ax:
            return float(base[k])
        return 0.5 * (float(ax[k]["min"]) + float(ax[k]["max"]))

    missing = [k for k in base if k not in ax]
    out = [("기준 (요청값)", dict(base))]
    if args.tool_preset == "baseline":
        return out
    out += [
        ("z6 상단 90 %", dict(base, z6=hi("z6"))),
        ("z11 상단 90 %", dict(base, z11=hi("z11"))),
        ("flare 상단 90 %", dict(base, flare=hi("flare"))),
        ("msd 상단 90 %", dict(base, msd_nm=hi("msd_nm"))),
        ("수차 둘 다 상단", dict(base, z6=hi("z6"), z11=hi("z11"))),
    ]
    if args.tool_preset == "all":
        out.append(("전부 중앙", {k: mid(k) for k in base}))
    # 같은 이름이 같은 값이 되어 버린 상태는 지운다. 축이 없으면 '상단 90 %'
    # 가 기준과 똑같아지는데, 그대로 두면 같은 조건을 여러 번 재고 그 중복이
    # 상관의 후보 수를 부풀린다.
    seen, uniq = set(), []
    for nm, t in out:
        key = tuple(round(float(t[k]), 12) for k in sorted(base))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((nm, t))
    if missing:
        uniq[0] = (f"기준 (요청값. 학습 범위에 없는 축 "
                   f"{', '.join(missing)} 은 요청값 그대로)", uniq[0][1])
    return uniq


# ---------------------------------------------------------------------------
def pick_candidates(rows: list[dict], n_top: int, n_spread: int) -> list[int]:
    """되돌려 잴 후보의 색인. 상위권 + 전 구간에 퍼진 표본.

    상위권만 재면 순위 상관을 잴 수 없다. 상관은 넓은 구간이 있어야 뜻을
    가지므로, 전체에서 고르게 뽑은 표본을 함께 넣는다. 반대로 퍼진 표본만
    재면 '상위 k 겹침' 을 잴 수 없다.
    """
    n = len(rows)
    idx = list(range(min(n_top, n)))
    if n_spread <= 0 or n <= len(idx):
        return sorted(set(idx))
    # 구간 표본을 **제약 통과군 안에서 먼저** 뽑는다. 중심 판정이 통과군
    # 안에서 계산되므로, 표본이 위반군으로 몰리면 중심 판정의 후보 수가
    # 모자라진다. 실측 : 수차가 큰 영역에서 통과군이 9 개뿐이었다.
    def ok(i):
        return bool(rows[i]["nils_ok"] and rows[i]["cdu_ok"])

    taken = set(idx)
    keep_rest = [i for i in range(n) if i not in taken and ok(i)]
    viol_rest = [i for i in range(n) if i not in taken and not ok(i)]
    n_keep = max(1, (n_spread * 2) // 3)       # 2/3 는 통과군에
    for pool, want in ((keep_rest, n_keep), (viol_rest, n_spread - n_keep)):
        if not pool or want <= 0:
            continue
        step = max(1, len(pool) // want)
        idx += pool[::step][:want]
    return sorted(set(idx))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="추천 조건을 시뮬레이터로 되돌려 재는 13주차 검증")
    ap.add_argument("--models", required=True,
                    help="모형 접두사. 쉼표로 여러 개를 주면 전부 돌린다. "
                         "하나만 골라 보고하면 사후선택이 된다")
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--flare", type=float, default=0.02)
    ap.add_argument("--msd-nm", type=float, default=2.0)
    ap.add_argument("--blur-nm", type=float, default=10.0)
    ap.add_argument("--z6", type=float, default=0.02)
    ap.add_argument("--z11", type=float, default=0.01)
    ap.add_argument("--pol", choices=DS.POLARIZATION_CODES, default=None)
    ap.add_argument("--source", choices=DS.SOURCE_CODES, default=None)
    ap.add_argument("--mask", choices=DS.MASK_CODES, default=None)
    ap.add_argument("--search-arms", action="store_true",
                    help="이산 조합을 고정하지 않고 전부 되돌려 잰다. "
                         "후보가 조합 수만큼 늘어나 느려진다")
    ap.add_argument("--n-na", type=int, default=9)
    ap.add_argument("--n-sigma", type=int, default=9)
    ap.add_argument("--focus-span", type=float, default=300.0)
    ap.add_argument("--n-focus", type=int, default=33)
    ap.add_argument("--dose-span", type=float, default=0.15)
    ap.add_argument("--n-dose", type=int, default=33)
    ap.add_argument("--el", type=float, default=10.0)
    ap.add_argument("--n-top", type=int, default=8,
                    help="상위 몇 개를 되돌려 잴지")
    ap.add_argument("--n-spread", type=int, default=12,
                    help="순위 전 구간에서 몇 개를 더 뽑을지")
    ap.add_argument("--tool-preset", default="baseline",
                    choices=["baseline", "regions", "all"],
                    help="regions/all 은 과공차 영역을 함께 돌린다")
    ap.add_argument("--mc-check", type=int, default=6,
                    help="블러 지름길을 몬테카를로로 대조할 점의 수. "
                         "0 이면 건너뛴다(권하지 않는다)")
    ap.add_argument("--sim-cdu", type=int, default=8,
                    help="실측 CDU 를 재는 창 안 표본 점 수. 0 이면 재지 "
                         "않고 NILS 만으로 판정한다 (기본 8)")
    ap.add_argument("--sim-cdu-rows", type=int, default=256,
                    help="CDU 몬테카를로의 행 수 (기본 256). 생성 경로는 "
                         "1024 를 쓰지만 한 점에 6.2 s 가 걸린다")
    ap.add_argument("--sim-cdu-trials", type=int, default=4,
                    help="CDU 몬테카를로의 시행 수 (기본 4)")
    ap.add_argument("--save-json", default=None,
                    help="후보별 결과를 json 으로 남긴다. figures_w13.py 가 "
                         "읽어 보고서용 그림을 만든다 (예 w13_result.json)")
    ap.add_argument("--conformal", default=None,
                    help="calibrate_w13.py 가 저장한 json. 주면 DOF "
                         "적합예측 구간과 실측 피복률을 함께 적는다")
    ap.add_argument("--mc-tol", type=float, default=0.05,
                    # argparse 는 help 를 `help % params` 로 서식화하므로
                    # 리터럴 백분율 기호는 %% 로 써야 한다. Python 3.15 가
                    # add_argument 시점에 이것을 검증하기 시작했고, 그보다
                    # 낮은 버전에서는 --help 를 부를 때까지 조용하다.
                    help="대조 허용치. 지름길의 CD 오차가 공정 공차의 "
                         "이 비율을 넘으면 실패로 본다 (기본 5 %%)")
    args = ap.parse_args()

    print("=" * 78)
    print(" 13주차 되돌림 검증 : 추천 조건을 시뮬레이터로 되돌려 재기")
    print("=" * 78)

    prefixes = [p.strip() for p in args.models.split(",") if p.strip()]
    results = {}

    # 적합예측 구간. 격자 설정이 보정 때와 다르면 **보정량과 적용량이 다른
    # 양**이 되어 보장이 깨진다. 그때는 구간을 쓰지 않고 경고만 적는다.
    conf = None
    if args.conformal:
        import json
        with open(args.conformal, encoding="utf-8") as fh:
            conf = json.load(fh)
        g = conf.get("grid", {})
        bad = [f"{k}: 보정 {g[k]} 대 지금 {v}"
               for k, v in (("n_focus", args.n_focus),
                            ("n_dose", args.n_dose),
                            ("focus_span", args.focus_span),
                            ("dose_span", args.dose_span),
                            ("el", args.el))
               if k in g and abs(float(g[k]) - float(v)) > 1e-9]
        print(f"\n  적합예측 구간 : q = {conf.get('q_nm')} nm, 보장 "
              f"{100 * float(conf.get('guarantee', 0)):.2f} % "
              f"(보정 표본 {conf.get('n_cal')} 개, {conf.get('models')})")
        if bad:
            print("  [경고] 격자 설정이 보정 때와 다르다 : "
                  + "; ".join(bad))
            print("  보정량과 적용량이 다른 양이 되므로 **피복 보장이")
            print("  깨진다.** 구간을 적지 않는다. 같은 설정으로 다시")
            print("  보정하거나, 보정 때의 격자로 되돌림을 돌릴 것.")
            conf = None
        elif conf.get("q_nm") is None:
            print("  [경고] 보정이 구간을 만들지 못했다. 적지 않는다.")
            conf = None
        if conf is not None:
            import os as _os
            _cm = str(conf.get("models", ""))
            _match = [p for p in prefixes
                      if p == _cm or _os.path.basename(p)
                      == _os.path.basename(_cm)]
            if not _match:
                print(f"  [경고] 보정 파일의 모형 '{_cm}' 이 이번 실행의")
                print(f"  모형 {prefixes} 에 없다. 구간을 적지 않는다.")
                conf = None
            elif len(prefixes) > 1:
                print(f"  구간은 **{_match[0]} 에만** 적는다. 다른 모형에")
                print("  적용하면 보장 없는 숫자에 보장 딱지를 붙이는 것이")
                print("  된다. 모형마다 따로 보정해야 한다.")

    # 번들을 **먼저 전부** 읽는다. 공구 상태를 모형마다 그 모형의 범위에서
    # 읽으면 같은 이름의 상태가 모형마다 다른 물리 조건이 되고, 그러면
    # 조건 차이가 모형 차이로 읽힌다(13주차에 찾은 결함). 교집합에서 한
    # 번만 읽어 모든 모형에 같은 조건을 쓴다.
    loaded = {}
    for pref in prefixes:
        loaded[pref] = RC.load_bundles(pref, RC.LABELS_REQUIRED)
    doms = [loaded[p][RC.LABELS_REQUIRED[0]]["domain"] for p in prefixes]
    dom_of = dict(zip(prefixes, doms))

    # 공구 상태는 두 묶음이다.
    #   교집합 : 모든 모형이 자기 학습 범위 안. **모형 사이 비교**의 조건.
    #   합집합 : 어떤 모형이 더 넓게 학습한 영역. 그 영역에서 좁은 모형은
    #            외삽하므로 비교가 아니라 **모형 대 시뮬레이터** 측정으로
    #            읽고, 외삽 딱지를 붙여 블록을 갈라 적는다.
    # 합집합을 빼면 보강으로 넓힌 영역이 아예 측정되지 않아 '보강이
    # 필요했다' 는 주장의 근거가 사라진다(13주차에 바로잡은 판단).
    states = []
    if len(prefixes) > 1:
        shared, shared_notes = common_axes(doms)
        print(f"\n  공구 상태를 **{len(prefixes)} 모형 학습 범위의 "
              f"교집합**에서 읽는다.")
        print("  같은 이름의 상태가 모형마다 다른 조건이면 비교가 성립하지")
        print("  않는다. 교집합이면 모든 모형이 자기 범위 안에서 평가된다.")
        for nt in shared_notes:
            print(f"  [주의] {nt}")
    else:
        shared = doms[0]
        shared_notes = []
    for nm, t in tool_presets(shared, args):
        states.append((nm, t, "교집합"))

    if len(prefixes) > 1 and args.tool_preset != "baseline":
        uni = union_axes(doms)
        seen = {tuple(round(float(t[k]), 12) for k in sorted(t))
                for _, t, _ in states}
        n_add = 0
        for nm, t in tool_presets(uni, args):
            key = tuple(round(float(t[k]), 12) for k in sorted(t))
            if key in seen:
                continue                      # 교집합과 같은 조건이면 중복
            seen.add(key)
            states.append((nm + " [합집합]", t, "합집합"))
            n_add += 1
        if n_add:
            print(f"\n  여기에 **합집합 영역 {n_add} 상태**를 더해 돌린다.")
            print("  어떤 모형이 더 넓게 학습한 영역이다. 좁은 모형은 거기서")
            print("  외삽하므로 모형 사이 비교에 넣지 않고, 외삽 딱지를 붙여")
            print("  따로 적는다. 되돌림은 모형 대 **시뮬레이터**를 재므로")
            print("  이 영역의 측정 자체는 모형 하나만으로도 뜻을 가진다.")
        else:
            print("\n  모형들의 학습 범위가 같아 합집합 영역이 없다.")

    def conf_for(pref: str) -> bool:
        """이 모형이 보정 파일의 모형인가."""
        if conf is None:
            return False
        import os as _os
        cm = str(conf.get("models", ""))
        return bool(pref == cm
                    or _os.path.basename(pref) == _os.path.basename(cm))

    for pref in prefixes:
        print(f"\n{'=' * 78}")
        print(f" 모형 : {pref}")
        print("=" * 78)
        bundles = loaded[pref]
        ref = bundles[RC.LABELS_REQUIRED[0]]
        feat, dom = list(ref["features"]), ref["domain"]
        rel_p95, rel_src, _m = RC.resolve_rel_p95(bundles, None)
        kind = next((b.get("model_kind") for b in bundles.values()
                     if isinstance(b, dict)), "")
        print(f"  입력 {len(feat)} 열, 학습 행 "
              f"{dom.get('n_train_rows', 0):,}, 모형 종류 {kind or '?'}")
        print(f"  모형 95 백분위 오차 {100 * rel_p95:.2f} %  (출처 : {rel_src})")

        arms, arm_notes = RC.resolve_arms(feat, dom, args)
        for nt in arm_notes:
            print(f"  [주의] {nt}")

        per_model = []
        for tname, tool, scope in states:
            outside = outside_axes(tool, dom_of[pref])
            print(f"\n  {'-' * 74}")
            print(f"  공구 상태 : {tname}"
                  + ("   <- **이 모형의 학습 범위 밖 (외삽) : "
                     + ", ".join(outside) + "**" if outside else ""))
            print(f"    " + ", ".join(f"{k}={v:.4g}" for k, v in tool.items()))
            if outside:
                print("    이 상태의 숫자는 외삽이다. 모형 사이 비교에 넣지")
                print("    않고 외삽 블록에만 적는다.")

            rows_all = []
            focus = dose_rel = None
            tol = None
            refused = None
            # [13차 보완 88 번] **대리 모형의 존재 이유가 측정되지 않았다.**
            # train_surrogate 는 자기를 "시뮬레이터의 빠른 근사" 라고 적고
            # README 는 "그 탐색을 빠르게 하는 것" 이라고 적는데, 얼마나
            # 빠른지는 어디에도 없었다. 되돌림이 이미 두 경로를 다 호출하므로
            # 여기서 재면 추가 비용이 0 이다. 창 하나당 비용으로 비교한다.
            _t_sur = 0.0
            for arm in arms:
                t = dict(tool)
                t.update(arm)
                _t0 = time.perf_counter()
                try:
                    r = RC.rank(bundles, feat, t, args.pitch, args.duty,
                                rel_p95, args.n_na, args.n_sigma,
                                args.focus_span, args.n_focus,
                                args.dose_span, args.n_dose, args.el, False)
                    _t_sur += time.perf_counter() - _t0
                except SystemExit as e:
                    # 거부는 **결과**다. 보강 전 모형이 넓힌 영역에서 상자
                    # 밖이라 추천을 아예 못 하는 것은 '측정할 후보가 없다'
                    # 가 아니라 '그 영역을 덮지 못한다' 는 발견이고, 보강이
                    # 필요했다는 가장 직접적인 증거다. 조용히 건너뛰면 그
                    # 증거가 출력에서 사라진다(13주차에 실제로 사라졌다).
                    # 문구를 짧게 줄인다. 표의 한 칸에 들어가야 하고,
                    # 문장 중간에서 잘리면 무슨 거부인지 읽을 수 없다.
                    _msg = str(e).splitlines()[0].strip()
                    _msg = _msg.replace("[오류] ", "")
                    for _cut in ("니다", "."):
                        if _cut in _msg:
                            _msg = _msg.split(_cut)[0] + (
                                _cut if _cut == "니다" else "")
                            break
                    refused = _msg or "거부"
                    print(f"    [거부] {refused}")
                    rows_all = []
                    break
                lab = RC.arm_label(arm) if len(arms) > 1 else ""
                for row in r["rows"]:
                    row["arm"] = lab
                    row["_arm_codes"] = dict(arm)
                    rows_all.append(row)
                focus, dose_rel = r["focus"], r["dose_rel"]
                if r["rows"]:
                    tol = r["rows"][0]["nominal"]["tol"]
            if not rows_all or tol is None or focus is None:
                if refused is None:
                    refused = "평가할 후보가 없다"
                    print(f"    [거부] {refused}")
                per_model.append((tname, [], scope, outside, refused))
                continue
            rows_all.sort(key=lambda r: (r["nils_ok"] and r["cdu_ok"],
                                         r["nominal"]["dof"],
                                         r["nominal"]["total"]), reverse=True)

            idx = pick_candidates(rows_all, args.n_top, args.n_spread)
            print(f"    후보 {len(rows_all)} 개 중 {len(idx)} 개를 되돌려 잼 "
                  f"(상위 {min(args.n_top, len(rows_all))} + 구간 표본)")

            recs = []
            for rank_i in idx:
                row = rows_all[rank_i]
                spec = DS.SampleSpec(
                    pitch=args.pitch, duty=args.duty,
                    na=row["na"], sigma_out=row["sigma_out"],
                    flare=tool["flare"], msd_nm=tool["msd_nm"],
                    blur_nm=tool["blur_nm"], z6=tool["z6"], z11=tool["z11"],
                    **_arm_kwargs(row.get("_arm_codes", {})))
                _t0 = time.perf_counter()
                sim = simulate_window(spec, focus, dose_rel, args.el,
                                      float(tol))
                sim["_t_win"] = time.perf_counter() - _t0
                if args.sim_cdu > 0 and sim.get("ok"):
                    _t1 = time.perf_counter()
                    sim["cdu"] = measure_cdu(
                        spec, focus, dose_rel, sim,
                        n_points=args.sim_cdu, n_rows=args.sim_cdu_rows,
                        n_trials=args.sim_cdu_trials, seed=rank_i)
                    sim["_t_cdu"] = time.perf_counter() - _t1
                recs.append({"rank": rank_i, "row": row, "sim": sim,
                             "spec": spec})
            # 적합예측 구간은 **그 모형에 대해** 보정한 것이다. 다른
            # 모형에 적용하면 보장이 없는 숫자에 보장 딱지를 붙이는 것이
            # 된다. 3 모형 실행에서 s13c 의 파일을 주면 셋 다에 붙던 버그를
            # 13주차에 잡았다. 접두사가 맞는 모형에만 적는다.
            # 시간은 **보고보다 먼저** 붙여야 한다. 보고가 그 값을 읽기
            # 때문이다. 처음에 순서를 반대로 두어 [속도] 블록이 조용히
            # 빠졌다(13주차에 실제로 그랬다).
            #
            # 대리 모형이 **평가한** 창 수. `rows_all` 은 상자 검사를 통과해
            # 살아남은 것만이므로 그것으로 나누면 창당 비용이 과대평가된다.
            # 격자 전체가 평가 대상이다.
            _n_win = max(args.n_na * args.n_sigma * len(arms), 1)
            for _r in recs:
                _r["_t_sur_win"] = (_t_sur / _n_win if _n_win else None)
            report_tool(tname, recs, args,
                        conf if conf_for(pref) else None)
            per_model.append((tname, recs, scope, outside, None))

        # 블러 지름길 대조. 모형과 무관하지만 공구 상태에 따라 달라지므로
        # 요청한 공구 상태로 한 번 돌린다.
        if args.mc_check > 0:
            spec = DS.SampleSpec(
                pitch=args.pitch, duty=args.duty, na=1.2, sigma_out=0.7,
                flare=args.flare, msd_nm=args.msd_nm, blur_nm=args.blur_nm,
                z6=args.z6, z11=args.z11)
            print(f"\n  {'-' * 74}")
            print("  [자기 시험] 블러 1 차원 지름길 대 실제 몬테카를로")
            cc = mc_crosscheck(spec, args.mc_check)
            if not cc["ok"] or not cc["rows"]:
                print(f"    대조 불가 : {cc.get('reason', '유효 점 없음')}")
            else:
                print(f"    {'dz':>8}{'dose':>7}{'지름길':>11}"
                      f"{'MC 평균':>11}{'차이':>9}{'공차 대비':>10}"
                      f"{'(SE 배수)':>11}")
                for r in cc["rows"]:
                    print(f"    {r['dz']:>8.1f}{r['rel']:>7.3f}"
                          f"{r['short']:>11.4f}{r['mc']:>11.4f}"
                          f"{r['diff']:>+9.4f}"
                          f"{100 * r['rel_tol']:>9.3f}%"
                          f"{r['ratio']:>11.2f}")
                ok = cc["worst_rel_tol"] <= args.mc_tol
                print(f"    최악 : 공차의 {100 * cc['worst_rel_tol']:.3f} % "
                      f"(허용 {100 * args.mc_tol:.1f} %) -> "
                      + ("통과. 지름길이 성립한다"
                         if ok else "**실패. 되돌림 숫자를 쓸 수 없다**"))
                print(f"    (공차 pw_tol = {100 * cc['pw_tol']:.2f} %. "
                      f"SE 배수는 참고용이다. 그 표준오차는 행 사이 독립을")
                print("     가정해 작게 나오고, 2 차원 블러가 행을 섞으므로")
                print("     실제보다 낙관적이다. 판정에 쓰지 않는다.)")
                if not ok:
                    print("    E[blur_2d(Poisson)] = blur_x(lambda) 가 깨졌다.")
                    print("    확률 경로를 직접 돌려야 한다.")
                    return 1
        results[pref] = per_model

    report_cross(results, args)

    # [88 번] 보고서용 그림을 터미널 캡쳐로 만들지 않기 위해 결과를 남긴다.
    # 그림은 `figures_w13.py` 가 이 파일을 읽어 그린다. 숫자를 손으로 옮겨
    # 적으면 그 순간 문서-코드 드리프트가 생긴다(claims.py 가 막으려는 것).
    if args.save_json:
        import json as _js
        _pay = {"models": prefixes, "pitch": args.pitch, "duty": args.duty,
                "el": args.el, "n_focus": args.n_focus,
                "n_dose": args.n_dose, "focus_span": args.focus_span,
                "dose_span": args.dose_span, "n_top": args.n_top,
                "criteria": CRITERIA, "tools": []}
        for _pref, _per in results.items():
            for _tn, _recs, _sc, _out, _rf in _per:
                _rows = []
                for _r in _recs:
                    _sm = _r["sim"]
                    _nm = _r["row"]["nominal"]
                    _cd = _sm.get("cdu") or {}
                    _rows.append({
                        "rank": _r["rank"], "na": _r["row"]["na"],
                        "sigma": _r["row"]["sigma_out"],
                        "sur_dof": _nm["dof"], "sur_upper": _nm["dof_upper"],
                        "sim_ok": bool(_sm.get("ok")),
                        "sim_dof": _sm.get("dof"),
                        "sur_nils_ok": bool(_r["row"]["nils_ok"]),
                        "sur_cdu_ok": bool(_r["row"]["cdu_ok"]),
                        "sim_nils_ok": bool(_sm.get("nils_ok")),
                        "cdu_frac": _cd.get("frac_of_tol"),
                        "cdu_max": _cd.get("cdu_max"),
                        "cdu_tol": _cd.get("stoch_tol"),
                        "t_sur_win": _r.get("_t_sur_win"),
                        "t_sim_win": _sm.get("_t_win"),
                        "t_cdu": _sm.get("_t_cdu"),
                    })
                _pay["tools"].append({
                    "model": _pref, "tool": _tn, "scope": _sc,
                    "outside": _out, "refused": _rf, "rows": _rows})
        with open(args.save_json, "w", encoding="utf-8") as _fh:
            _js.dump(_pay, _fh, ensure_ascii=False, indent=1)
        print(f"\n  결과 저장 : {args.save_json}")
        print("  py figures_w13.py --result " + args.save_json
              + " 로 보고서용 그림을 만들 수 있습니다.")
    print("\n" + "=" * 78)
    print(" 읽는 법")
    print("=" * 78)
    print("  - 순위는 **제약 통과군 안에서** 본다. 추천기의 정렬 키가")
    print("    제약을 먼저 보므로 위반군은 DOF 가 훨씬 크고, 두 군을 섞으면")
    print("    군 사이 격차가 상관을 끌어올린다. 실제 결정은 통과군 안에서")
    print("    일어난다.")
    print("  - 중심 판정은 **순위 보존**이다. 12주차가 '인증은 못 하지만")
    print("    순위는 쓸 수 있다' 고 적었고 그 근거는 측정되지 않았다.")
    print("    상관이 낮으면 12주차 결론을 고쳐야 한다.")
    print("  - '구간 포함' 은 대리 모형의 [하한, 상한] 이 실측을 담는지다.")
    print("    담지 못하면 구간을 넓히는 것이 아니라, 구간이 모형 오차를")
    print("    담지 않는다는 뜻이다(구간은 격자 편향만 담는다).")
    print("  - 모형이 여러 개면 **전부** 적는다. 하나를 골라 보고하면")
    print("    사후선택이다. 주 모형은 분포 내 기준으로 미리 정한다.")
    return 0


def _arm_kwargs(codes: dict) -> dict:
    """범주 코드를 `SampleSpec` 의 이름 인자로 되돌린다."""
    out = {}
    if "pol_code" in codes:
        out["polarization"] = DS.POLARIZATION_CODES[int(codes["pol_code"])]
    if "source_code" in codes:
        out["source_shape"] = DS.SOURCE_CODES[int(codes["source_code"])]
    if "mask_code" in codes:
        out["mask_type"] = DS.MASK_CODES[int(codes["mask_code"])]
    return out


def report_tool(tname: str, recs: list[dict], args,
                conf: dict | None = None) -> None:
    """한 공구 상태의 결과표.

    왜 제약 통과군 안에서만 재는가 (13주차에 실측으로 찾은 결함)
    -----------------------------------------------------------
    `recommend.rank` 의 정렬 키는 `(nils_ok and cdu_ok, dof, total)` 이다.
    즉 **제약 통과군이 먼저** 오고 그 안에서 DOF 내림차순이다. 그래서 순위가
    뒤인 후보가 DOF 는 훨씬 클 수 있다(위반군). 실측 예 : 통과군 DOF
    93.8~206.2, 위반군 168.8~375.0.

    처음에는 두 군을 한 상관에 넣었다. 그러면 **군 사이 격차가 상관을
    끌어올린다.** 실측 : 수차 둘 다 상단에서 혼합 스피어만 +0.799 인데 통과군
    안에서는 **-0.127** 이었다. 즉 '순위가 보존된다' 는 결론이 군 사이 격차
    때문에 나온 것이고, 실제 결정이 일어나는 통과군 안에서는 순위가 보존되지
    않았다. 결론을 뒤집을 만한 차이다.

    또 '예측 순서' 를 예측 DOF 로 재정렬했는데, 그것은 추천기가 내놓는 순서가
    아니다. 추천기는 제약을 먼저 본다. 그래서 상위 k 겹침이 '추천기의 상위
    k 가 살아남는가' 를 재지 못했다(같은 경우에서 1/3 로 보고했지만 추천기
    실제 상위 3 으로 재면 0/3 이었다).

    실측 순서는 **실측 NILS 로 제약을 다시 판정**한 뒤 같은 키로 정렬한다.
    '시뮬레이터로 돌렸다면 순위가 같았겠는가' 가 물어야 할 질문이기 때문이다.
    """
    good = [r for r in recs if r["sim"]["ok"]]
    if not good:
        print("    되돌려 잴 수 있는 후보가 없다")
        return

    def sur_pass(r):
        return bool(r["row"]["nils_ok"] and r["row"]["cdu_ok"])

    def cdu_refuted(r):
        """실측 CDU 가 공차를 **넘은 것이 확인된** 경우만 True.

        표본 한 점이라도 공차를 넘으면 창 전체 최댓값은 반드시 넘는다.
        그러므로 '넘었다' 는 단측으로 엄격하다. 반대로 표본이 전부 공차
        안이어도 창 전체 최댓값이 안이라는 **증명은 아니다**(표본은 최댓값의
        하한이다). 그래서 통과는 '인증' 이 아니라 '반증되지 않음' 이다.
        """
        c = r["sim"].get("cdu")
        if not c or not c.get("ok"):
            return False
        if not (c["cdu_max"] > c["stoch_tol"]):
            return False
        # 표집 잡음으로 모형을 억울하게 반증하지 않는다. 같은 지점을 다른
        # 씨앗으로 다시 재어 **그쪽도** 공차를 넘어야 반증으로 센다.
        # 실측 : `z11 상단 90 %` 에서 재측정 상대차가 최대 40.6 % 였다.
        # 한 번 재고 반증이라고 적으면 그 잡음이 '모형이 틀렸다' 가 된다.
        rep = c.get("cdu_repeat")
        if rep is None or not np.isfinite(rep):
            return True            # 재측정이 없으면 1 차 결과를 따른다
        return bool(rep > c["stoch_tol"])

    def sim_pass(r):
        # 13주차에 실측 CDU 를 몬테카를로로 재도록 고쳤다. 그 전에는 NILS
        # 만으로 판정해 예측(NILS+CDU)과 축이 어긋났고, CDU 로 떨어진 후보가
        # '모형이 틀렸다' 로 세어질 수 있었다. 이제 두 축을 모두 본다.
        return bool(r["sim"]["nils_ok"]) and not cdu_refuted(r)

    print(f"\n    {'순위':>4}{'NA':>7}{'sigma':>7}"
          f"{'예측 DOF':>10}{'실측 DOF':>10}{'차이':>9}{'구간 포함':>10}"
          f"{'노광 차':>9}{'초점 차':>9}  판정(예측/실측)")
    print("    " + "-" * 92)
    for r in good:
        nm, sim = r["row"]["nominal"], r["sim"]
        inside = (sim["dof"] >= nm["dof"] - 1e-9
                  and sim["dof"] <= nm["dof_upper"] + 1e-9)
        dd = (f"{100 * (sim['dose_center'] - nm['dose_center']):+.2f}%p"
              if sim["dose_center"] is not None
              and nm["dose_center"] is not None else "-")
        dz = (f"{sim['focus_center'] - nm['focus_center']:+.1f}"
              if sim["focus_center"] is not None
              and nm["focus_center"] is not None else "-")
        verd = (("통과" if sur_pass(r) else "위반") + " / "
                + ("통과" if sim_pass(r) else "위반"))
        print(f"    {r['rank'] + 1:>4}{r['row']['na']:>7.3f}"
              f"{r['row']['sigma_out']:>7.3f}"
              f"{nm['dof']:>10.1f}{sim['dof']:>10.1f}"
              f"{sim['dof'] - nm['dof']:>+9.1f}"
              f"{('예' if inside else '아니오'):>10}{dd:>9}{dz:>9}"
              f"  {verd}")

    # 뒤집힘은 **같은 축에서만** 세어야 한다. 예측 판정은 NILS 와 CDU 를
    # 둘 다 보는데 실측 판정은 NILS 만 본다(CDU 는 몬테카를로가 필요해 재지
    # 않았다). 그대로 비교하면 **CDU 때문에 예측이 떨어뜨린 후보**가 '실측과
    # 어긋났다' 로 세어진다. 그것은 어긋남이 아니라 **재지 않은 축**이다.
    # 그래서 비교는 NILS 대 NILS 로 하고, CDU 로만 떨어진 후보는 따로 센다.
    def sur_nils(r):
        return bool(r["row"]["nils_ok"])

    measured_cdu = [r for r in good
                    if r["sim"].get("cdu", {}).get("ok")]
    if len(measured_cdu) == len(good) and good:
        # 양쪽이 NILS 와 CDU 를 모두 보므로 **같은 축**이다. 축을 맞추기
        # 위해 CDU 를 빼던 보정이 더 이상 필요하지 않다.
        flip = [r for r in good if sur_pass(r) != sim_pass(r)]
        cdu_only = []
        axis_note = "예측·실측 모두 NILS + CDU"
    else:
        flip = [r for r in good if sur_nils(r) != sim_pass(r)]
        cdu_only = [r for r in good if sur_nils(r) and not r["row"]["cdu_ok"]]
        axis_note = "예측 NILS+CDU 대 실측 NILS (CDU 미측정)"
    keep = [r for r in good if sur_pass(r)]
    viol = [r for r in good if not sur_pass(r)]

    def block(rs: list[dict], label: str, primary: bool) -> None:
        if len(rs) < 3:
            print(f"\n    [{label}] 후보 {len(rs)} 개로 순위를 잴 수 없다")
            return
        sur = [r["row"]["nominal"]["dof"] for r in rs]
        sim = [r["sim"]["dof"] for r in rs]
        rc = rank_corr(sur, sim)
        mark = " <- 중심 판정" if primary else ""
        if rc.get("why"):
            print(f"\n    [{label}] 순위 보존 : 잴 수 없음 ({rc['why']})")
            return
        # 동률 수를 함께 적는다. DOF 는 격자 간격의 배수로만 나오므로 동률이
        # 많고, 동률이 많으면 **상위 1 겹침이 정렬 안정성에 좌우된다.**
        # 실측 : 상위 1 겹침 0 % 인데 상위 3 은 100 % 인 경우가 나왔다.
        # 스피어만은 동률을 평균 순위로 처리해 값이 부풀 수 있으므로 켄달
        # 타우-b 를 함께 본다(동률을 분모에서 뺀다).
        n_u_sur, n_u_sim = len(set(sur)), len(set(sim))
        print(f"\n    [{label}] 순위 보존 : 스피어만 {rc['spearman']:+.3f}, "
              f"켄달 타우-b {rc['kendall']:+.3f}  (후보 {rc['n']} 개){mark}")
        print(f"      서로 다른 DOF 값 : 예측 {n_u_sur} / 실측 {n_u_sim} 개"
              + ("  <- 동률이 많다. 상위 1 겹침은 믿지 말 것"
                 if min(n_u_sur, n_u_sim) * 2 < rc["n"] else ""))
        # 예측 순서는 **추천기의 순위** 를 그대로 쓴다. 실측 순서는 실측으로
        # 제약을 다시 판정한 뒤 같은 키로 정렬한다.
        o_sur = [r["rank"] for r in sorted(
            rs, key=lambda r: (sur_pass(r), r["row"]["nominal"]["dof"],
                               r["row"]["nominal"]["total"]), reverse=True)]
        o_sim = [r["rank"] for r in sorted(
            rs, key=lambda r: (sim_pass(r), r["sim"]["dof"],
                               r["sim"]["total"]), reverse=True)]
        # DOF 만으로 정렬한 순서도 함께 낸다. 둘이 갈리면 원인이 다르다.
        #   제약 포함 겹침이 낮고 DOF 만 겹침이 높다 -> **제약 판정이 뒤집혔다**
        #   둘 다 낮다                              -> DOF 순서 자체가 깨졌다
        # 실측 예 : 스피어만 +0.972 인데 상위 1 겹침 0 % 였다. DOF 순서는
        # 거의 완벽한데 예측 상위 후보가 실측 NILS 로는 제약을 못 넘어서,
        # '추천' 이 뒤집힌 것이었다. 하나만 적으면 이 구분이 사라진다.
        d_sur = [r["rank"] for r in sorted(
            rs, key=lambda r: r["row"]["nominal"]["dof"], reverse=True)]
        d_sim = [r["rank"] for r in sorted(
            rs, key=lambda r: r["sim"]["dof"], reverse=True)]
        for k in (1, 3, 5):
            if len(rs) >= k:
                a = 100 * top_k_overlap(o_sur, o_sim, k)
                b2 = 100 * top_k_overlap(d_sur, d_sim, k)
                print(f"      상위 {k} 겹침 : 제약 포함 {a:.0f} % / "
                      f"DOF 만 {b2:.0f} %")
        if len(rs) >= 3:
            a3 = top_k_overlap(o_sur, o_sim, 3)
            b3 = top_k_overlap(d_sur, d_sim, 3)
            if b3 - a3 > 0.33:
                print("      -> DOF 순서는 남았는데 **제약 판정이 뒤집혀**")
                print("         추천이 바뀐다. 실측 NILS 를 확인할 것.")
            elif b3 < 0.67:
                print("      -> DOF 순서 자체가 깨졌다. 이 영역에서는 순위를")
                print("         쓸 수 없다.")
        n_in = sum(1 for r in rs
                   if r["row"]["nominal"]["dof"] - 1e-9 <= r["sim"]["dof"]
                   <= r["row"]["nominal"]["dof_upper"] + 1e-9)
        if conf and conf.get("q_nm") is not None:
            q = float(conf["q_nm"])
            # 격자 구간과 **다른 것**을 재는 구간이다. 격자 구간은 격자
            # 편향만 담고, 이쪽은 모형 오차의 적합예측 구간이다. 둘을
            # 나란히 적어 차이가 보이게 한다.
            _judge_cf = True
            n_cf = sum(1 for r in rs
                       if abs(r["sim"]["dof"]
                              - r["row"]["nominal"]["dof"]) <= q + 1e-9)
            gu = 100 * float(conf.get("guarantee", 0))
            print(f"      적합예측 구간 포함 : {n_cf}/{len(rs)} "
                  f"({100 * n_cf / len(rs):.0f} %)  "
                  f"[+-{q:.1f} nm, 보장 {gu:.1f} %]")
            if n_cf < len(rs) * float(conf.get("guarantee", 0)) - 1e-9:
                print("        표기 보장보다 낮다. 추천기가 묻는 최적점")
                print("        근처로 **분포가 이동**한 것이다. 보정은")
                print("        데이터셋 검증 분할 조건에서 했고 이쪽은")
                print("        추천 상위 후보이므로 같은 분포가 아니다.")
                print("        (후보 수가 적으면 잡음도 크다. 수를 함께 볼 것)")
        # [13차 보완 87 번] **판정 기준을 코드가 적용해 인쇄한다.**
        # 그전에는 되돌림이 숫자만 내고 '어느 값이면 성공인가' 가 아무 데도
        # 없었다. 그러면 숫자를 본 뒤에 기준을 정하게 되고, 그것은
        # 보강명세_동결.md 가 막으려던 사후선택과 같은 행위다. 기준은
        # 판정기준_동결.md 에 실제 모형으로 돌리기 **전**에 적었다.
        if primary:
            # [88 번] 대리 모형의 **존재 이유**를 숫자로 적는다. 느리면서
            # 부정확하면 쓸 이유가 없고, 빠른 만큼이 오차를 감수할 근거다.
            _ts = [r["_t_sur_win"] for r in rs
                   if r.get("_t_sur_win") is not None]
            _tw = [r["sim"]["_t_win"] for r in rs
                   if r["sim"].get("_t_win") is not None]
            if _ts and _tw:
                _a = float(np.median(_ts)); _b = float(np.median(_tw))
                print(f"\n      [속도] 창 하나당 대리 모형 {1e3 * _a:.1f} ms "
                      f"대 시뮬레이터 {_b:.2f} s"
                      + (f"  -> **{_b / _a:,.0f} 배**" if _a > 0 else ""))
                _tc = [r["sim"]["_t_cdu"] for r in rs
                       if r["sim"].get("_t_cdu") is not None]
                if _tc:
                    print(f"        (CDU 몬테카를로는 창당 추가 "
                          f"{float(np.median(_tc)):.1f} s. 대리 모형은 CDU 를 "
                          f"라벨로 예측하므로 이 비용이 없다)")
                print("        이 배수가 대리 모형을 쓰는 근거다. 느리면서")
                print("        부정확하면 쓸 이유가 없다. 아래 판정은 '그 대가로")
                print("        치르는 오차가 받아들일 만한가' 를 묻는다.")
            _tau = (rc["kendall"] if not rc.get("why") else None)
            _t3 = (top_k_overlap(d_sur, d_sim, 3) if len(rs) >= 3 else None)
            _lo = _gu = None
            if conf and conf.get("q_nm") is not None:
                _q = float(conf["q_nm"])
                _h = sum(1 for r in rs
                         if abs(r["sim"]["dof"]
                                - r["row"]["nominal"]["dof"]) <= _q + 1e-9)
                _gu = float(conf.get("guarantee", 0.0)) or None
                try:
                    from scipy.stats import beta as _bt
                    _n = len(rs)
                    _lo = (0.0 if _h == 0
                           else float(_bt.ppf(0.05, _h, _n - _h + 1)))
                except Exception:
                    _lo = None
            print(f"\n      [판정] 판정기준_동결.md 의 기준을 적용한다")
            _span = (float(max(sim) - min(sim)) if len(sim) > 1 else 0.0)
            _rows = judge(_tau, _t3, _lo, _gu, len(sur), perm_null(sur, sim),
                          min(n_u_sur, n_u_sim), _span)
            for _nm, _vd, _why in _rows:
                print(f"        {_nm:14s} {_vd:10s} {_why}")
            for _m in verdict_text(_rows):
                print(f"        -> {_m}")
            print("        (기준은 실제 모형으로 돌리기 **전**에 정했다.")
            print("         숫자를 보고 기준을 고치면 그 문서의 변경 기록에")
            print("         적어야 하고, 판정에 '사후선택이 섞였다' 를 단다.)")
        print(f"      구간 포함 : {n_in}/{len(rs)} "
              f"({100 * n_in / len(rs):.0f} %)")
        med = float(np.median([b - a for a, b in zip(sur, sim)]))
        print(f"      중위 DOF 차 : {med:+.1f} nm "
              + ("(실측이 더 좁다. 대리 모형이 **낙관적**)"
                 if med < -1e-9 else
                 "(실측이 더 넓다)" if med > 1e-9 else "(같다)"))
        # **선택 편의(승자의 저주)를 분리한다.** 상위권 후보는 대리 모형이
        # DOF 를 가장 크게 본 점이다. 모형 오차가 무작위여도 그 점들은
        # 평균적으로 과대평가되어 있다(최댓값을 고른 효과). 상위권과 구간
        # 표본을 섞어 치우침을 재면 '모형이 낙관적' 과 '내가 낙관적인 점을
        # 골랐다' 가 구분되지 않는다. 구간 표본은 순위와 무관하게 뽑으므로
        # 선택 편의가 작고, 두 수의 차이가 선택 편의의 크기다.
        top = [(a, b, r) for a, b, r in
               zip(sur, sim, [x["rank"] for x in rs]) if r < args.n_top]
        spr = [(a, b, r) for a, b, r in
               zip(sur, sim, [x["rank"] for x in rs]) if r >= args.n_top]
        if top and spr:
            m_t = float(np.median([b - a for a, b, _ in top]))
            m_s = float(np.median([b - a for a, b, _ in spr]))
            print(f"        상위권 {len(top)} 개 {m_t:+.1f} nm  대  "
                  f"구간 표본 {len(spr)} 개 {m_s:+.1f} nm")
            gap = m_t - m_s
            if gap < -1e-9:
                print(f"        상위권이 {abs(gap):.1f} nm 더 낙관적이다. 그만큼은")
                print("        **모형 치우침이 아니라 선택 편의**일 수 있다.")
                print("        모형 자체의 치우침은 구간 표본 쪽으로 읽을 것.")
            elif gap > 1e-9:
                print("        상위권이 오히려 덜 낙관적이다. 선택 편의가")
                print("        치우침을 만든 것은 아니다.")
            else:
                print("        둘이 같다. 선택 편의가 치우침을 설명하지 않는다.")
        elif top or spr:
            print(f"        (한쪽만 있어 선택 편의를 가를 수 없다 : "
                  f"상위권 {len(top)}, 구간 표본 {len(spr)})")
        if primary:
            # [92 번] 귀무분포를 **판정이 쓰는 것과 같은 것**으로 찍는다.
            # 전에는 여기서 표집 200 회로 스피어만만 찍고, 판정은 켄달을
            # 고정 임계값과만 비교했다. 즉 같은 화면에 '우연히 이만큼
            # 나온다' 와 '충족' 이 함께 인쇄됐다.
            _nl = perm_null(sur, sim)
            if _nl:
                _k = "완전 열거" if _nl["exact"] else f"표집 {_nl['n_perm']} 회"
                print(f"      (음성 대조[{_k}] : 실측을 섞으면 켄달 중앙값 "
                      f"{_nl['tau_median']:+.3f}, "
                      f"{100 * CRITERIA['null_q']:.0f} 백분위 "
                      f"{_nl['tau_q']:+.3f})")
                if _nl["n_distinct_tau"] <= 5:
                    print(f"        켄달이 가질 수 있는 값이 "
                          f"{_nl['n_distinct_tau']} 개뿐이다. 이 크기에서는")
                    print("        상관계수로 순위 보존을 가릴 수 없다.")

    block(keep, "제약 통과군", True)
    block(viol, "제약 위반군", False)
    n_und = sum(1 for r in good if r["sim"].get("undetermined"))
    if n_und:
        print(f"\n    실측이 **격자 미결정** 인 후보 : {n_und}/{len(good)}")
        print("      하한 격자점에서 EL 목표를 못 넘겨 DOF 하한이 0 이고")
        print("      상한만 살아 있다. 공정창이 없다는 뜻이 아니므로 이")
        print("      후보의 DOF 차를 '모형이 낙관적' 으로 읽으면 안 된다.")
        print("      --n-focus / --n-dose 를 늘려 다시 재야 한다.")
    print(f"\n    제약 판정이 뒤집힌 후보 : {len(flip)}/{len(good)}  "
          f"(비교 축 : {axis_note})")
    if flip:
        print("      예측과 실측이 제약에서 갈린다. DOF 가 맞아도 이 후보는")
        print("      추천할 수 없거나, 반대로 버린 후보가 쓸 수 있다는 뜻이다.")

    if measured_cdu:
        c0 = measured_cdu[0]["sim"]["cdu"]
        fr = [r["sim"]["cdu"]["frac_of_tol"] for r in measured_cdu]
        ref = [r for r in measured_cdu if cdu_refuted(r)]
        hit = sum(1 for r in measured_cdu
                  if r["sim"]["cdu"]["argmax_at_nils_min"])
        print(f"\n    [실측 CDU] 몬테카를로로 쟀다 "
              f"({len(measured_cdu)}/{len(good)} 후보, 행 {c0['n_rows']}, "
              f"시행 {c0['n_trials']})")
        print(f"      공차 대비 최악 : 중위 {100 * float(np.median(fr)):.1f} %, "
              f"최대 {100 * max(fr):.1f} %  (공차 "
              f"{100 * c0['stoch_tol']:.2f} %)")
        once = [r for r in measured_cdu
                if r["sim"]["cdu"]["cdu_max"]
                > r["sim"]["cdu"]["stoch_tol"]]
        print(f"      공차를 넘은 것이 **확인된** 후보 : "
              f"{len(ref)}/{len(measured_cdu)}"
              + (f"  (1 차에서는 {len(once)} 개였고 재측정에서도 넘은 것만"
                 f" 센다)" if len(once) != len(ref) else ""))
        print(f"      창 안 표본 {c0['n_sampled']} 점 / 통과 격자 "
              f"{c0['n_window']} 점. 표본은 창 전체 최댓값의 **하한**이므로,")
        print("      공차 안이라는 것은 '인증' 이 아니라 '반증되지 않음' 이다.")
        print(f"      표본 최댓값이 NILS 최소 지점에서 나온 후보 : "
              f"{hit}/{len(measured_cdu)}")
        if hit < len(measured_cdu):
            print("        전부가 아니다. 'CDU 최악은 NILS 최악에 있다' 는")
            print("        겨냥이 완전하지 않다는 뜻이고, 표본이 최댓값을")
            print("        놓칠 수 있다. --sim-cdu 를 늘려 확인할 것.")
        rp = [abs(r["sim"]["cdu"]["cdu_repeat"] - r["sim"]["cdu"]["cdu_max"])
              / max(r["sim"]["cdu"]["cdu_max"], 1e-12)
              for r in measured_cdu
              if np.isfinite(r["sim"]["cdu"]["cdu_repeat"])]
        if rp:
            print(f"      같은 지점 재측정 상대차 : 중위 "
                  f"{100 * float(np.median(rp)):.1f} %, 최대 "
                  f"{100 * max(rp):.1f} %  <- 표집 잡음의 크기")
        # 11주차 학습 출력은 CDU 가 공차의 약 2 % 라고 적었다. 그 숫자는
        # **데이터셋의 LHC 조건**에서 나온 것이고, 추천기가 고르는 점은
        # 다르다. 큰 DOF 를 고르면 낮은 NA 로 가고 그쪽은 NILS 가 낮아
        # CDU 가 커진다. 두 숫자를 나란히 적지 않으면 '판정을 가르는 축이
        # 아니다' 라는 11주차 결론을 그대로 물려받게 된다.
        w11 = 0.02
        if max(fr) > 3 * w11:
            print(f"      [주의] 11주차는 CDU 가 공차의 약 {100 * w11:.0f} % "
                  f"라고 적었는데 여기서는 최대 {100 * max(fr):.1f} % 다.")
            print("        그 숫자는 데이터셋의 LHC 조건에서 나온 것이고,")
            print("        추천기는 **큰 DOF 를 고르므로 낮은 NA** 로 가고")
            print("        그쪽은 NILS 가 낮아 CDU 가 커진다. 즉 'CDU 는")
            print("        판정을 가르는 축이 아니다' 는 11주차 결론을")
            print("        추천 조건에 그대로 옮겨 쓸 수 없다. 보고서에는")
            print("        이 숫자를 쓸 것.")
    if cdu_only:
        print(f"\n    CDU 로만 떨어진 후보 (실측에서 재지 않은 축) : "
              f"{len(cdu_only)}/{len(good)}")
        print("      이 후보들은 **비교 대상이 아니다.** 뒤집힘 수에 넣지")
        print("      않았다. 넣으면 '재지 않은 축' 이 '모형이 틀렸다' 로")
        print("      세어진다. --sim-cdu 를 0 보다 크게 주면 재서 비교한다.")


def report_cross(results: dict, args) -> None:
    """모형 사이 비교. 전부 적는다."""
    if len(results) < 2:
        return
    print(f"\n{'=' * 78}")
    print(" 모형 사이 비교 (전부 적는다. 고르지 않는다)")
    print("=" * 78)
    print("\n  **제약 통과군 안에서** 잰 값이다. 실제 결정이 그 안에서")
    print("  일어나기 때문이다. 위반군을 섞으면 군 사이 DOF 격차가 상관을")
    print("  끌어올려 결론이 뒤집힌다(13주차에 실측으로 확인).")
    print("\n  공구 상태는 **모든 모형의 학습 범위 교집합**에서 읽은 같은")
    print("  물리 조건이다. 번들마다 자기 범위에서 읽으면 같은 이름이 다른")
    print("  조건이 되어 조건 차이가 모형 차이로 읽힌다.")
    # 접두사가 길면 열이 겹쳐 표를 읽을 수 없다. 뒤에서 자른다(앞쪽 경로가
    # 아니라 모형 이름이 구분에 쓰이므로 뒤를 남긴다). 전체 이름은 위의
    # 모형별 블록 머리에 이미 한 번 그대로 찍혀 있다.
    def short(name: str, w: int = 20) -> str:
        return name if len(name) <= w else "..." + name[-(w - 3):]

    print(f"\n  {'모형':<20} {'공구 상태':<16}{'스피어만':>10}{'켄달':>8}"
          f"{'구간 포함':>10}{'중위 DOF 차':>12}{'후보':>6}")
    print("  " + "-" * 84)
    for pref, per in results.items():
        for tname, recs, scope, outside, refused in per:
            if scope != "교집합" or outside:
                continue          # 외삽은 비교에 넣지 않는다. 아래 블록으로
            if refused:
                print(f"  {short(pref):<20} {short(tname, 16):<16}"
                      f"{('거부 : ' + refused)[:38]:>38}")
                continue
            # **제약 통과군만** 본다. 두 군을 섞으면 군 사이 격차가 상관을
            # 끌어올려 결론이 뒤집힌다(실측 : +0.799 대 -0.127).
            good = [r for r in recs if r["sim"]["ok"]
                    and r["row"]["nils_ok"] and r["row"]["cdu_ok"]]
            if len(good) < 3:
                print(f"  {short(pref):<20} {short(tname, 16):<16}"
                      f"{'통과군 ' + str(len(good)) + ' 개':>30}")
                continue
            sur = [r["row"]["nominal"]["dof"] for r in good]
            sim = [r["sim"]["dof"] for r in good]
            rc = rank_corr(sur, sim)
            n_in = sum(1 for r in good
                       if r["row"]["nominal"]["dof"] - 1e-9 <= r["sim"]["dof"]
                       <= r["row"]["nominal"]["dof_upper"] + 1e-9)
            med = float(np.median([b - a for a, b in zip(sur, sim)]))
            sp = (f"{rc['spearman']:>+10.3f}" if not rc.get("why")
                  else f"{'잴 수 없음':>10}")
            kd = (f"{rc['kendall']:>+8.3f}" if not rc.get("why")
                  else f"{'-':>8}")
            print(f"  {short(pref):<20} {short(tname, 16):<16}{sp}{kd}"
                  f"{100 * n_in / len(good):>9.0f}%{med:>+12.1f}"
                  f"{len(good):>6}")
    print("\n  주 모형은 **분포 내** 기준으로 미리 정한다(최악 영역 95")
    print("  백분위가 가장 낮은 것). 되돌림은 그 선택을 검증하는 것이고")
    print("  선택의 근거가 아니다. 둘이 어긋나면 그 불일치를 결과로 적는다.")
    report_extrap(results, short)


def report_extrap(results: dict, short) -> None:
    """합집합(넓힌) 영역 블록. 모형마다 **내삽/외삽**을 딱지로 적는다.

    이 블록은 모형 사이 비교가 아니다. 되돌림은 모형 대 **시뮬레이터**를
    재고 시뮬레이터가 기준이므로, 한 모형만으로도 '그 모형이 이 영역에서
    맞히는가' 를 측정할 수 있다. 좁게 학습한 모형이 여기서 못 맞히는 것은
    결함이 아니라 **보강이 필요했다는 측정 증거**다.

    같은 표에 섞어 하나의 상관으로 적으면 안 된다. 내삽과 외삽을 한 숫자로
    묶는 것이기 때문이다. 그래서 딱지를 붙여 행을 갈라 적는다.
    """
    rows = [(pref, tn, recs, out, rf)
            for pref, per in results.items()
            for tn, recs, sc, out, rf in per if sc == "합집합" or out]
    if not rows:
        return
    print(f"\n{'=' * 78}")
    print(" 합집합(넓힌) 영역 — 내삽/외삽 딱지를 붙여 따로 적는다")
    print("=" * 78)
    print("\n  모형 사이 비교가 **아니다.** 되돌림은 모형 대 시뮬레이터를")
    print("  재므로 한 모형만으로도 이 영역의 측정이 뜻을 가진다. 좁게")
    print("  학습한 모형이 여기서 못 맞히는 것은 보강이 필요했다는 증거다.")
    print(f"\n  {'모형':<20} {'공구 상태':<22}{'판정':<6}{'스피어만':>10}"
          f"{'구간 포함':>10}{'중위 DOF 차':>12}{'후보':>6}")
    print("  " + "-" * 88)
    for pref, tname, recs, out, rf in rows:
        tag = ("외삽" if out else "내삽")
        if rf:
            # 상자 밖 거부. 이것이 이 블록의 핵심 증거다.
            print(f"  {short(pref):<20} {short(tname, 22):<22}{tag:<6}"
                  f"{('거부 : ' + rf)[:38]:>38}")
            continue
        good = [r for r in recs if r["sim"]["ok"]
                and r["row"]["nils_ok"] and r["row"]["cdu_ok"]]
        if len(good) < 3:
            print(f"  {short(pref):<20} {short(tname, 22):<22}{tag:<6}"
                  f"{'통과군 ' + str(len(good)) + ' 개':>38}")
            continue
        sur = [r["row"]["nominal"]["dof"] for r in good]
        sim = [r["sim"]["dof"] for r in good]
        rc = rank_corr(sur, sim)
        n_in = sum(1 for r in good
                   if r["row"]["nominal"]["dof"] - 1e-9 <= r["sim"]["dof"]
                   <= r["row"]["nominal"]["dof_upper"] + 1e-9)
        med = float(np.median([b - a for a, b in zip(sur, sim)]))
        sp = (f"{rc['spearman']:>+10.3f}" if not rc.get("why")
              else f"{'잴 수 없음':>10}")
        print(f"  {short(pref):<20} {short(tname, 22):<22}{tag:<6}{sp}"
              f"{100 * n_in / len(good):>9.0f}%{med:>+12.1f}{len(good):>6}")
    print("\n  읽는 법 : '거부' 는 추천기가 그 조건을 **상자 밖**이라고")
    print("  돌려보낸 것이다. 숫자가 나쁜 것보다 센 결과다. 그 모형은 이")
    print("  영역에서 추천을 내놓을 수조차 없다는 뜻이고, 그것이 학습 범위를")
    print("  넓혀야 했던 가장 직접적인 증거다. 거부는 올바른 동작이며,")
    print("  틀린 숫자를 내놓는 것보다 낫다.")
    print("\n  '외삽' 행이 '내삽' 행보다 나쁘면 그 모형의 학습")
    print("  범위를 넓혀야 한다는 뜻이다. 보강 모형의 '내삽' 행이 보강 전")
    print("  모형의 '외삽' 행보다 좋으면, 그 차이가 **보강의 효과**다.")
    print("  (단, 같은 조건에서 잰 것이라도 한쪽은 내삽 한쪽은 외삽이므로")
    print("   '모형이 더 좋다' 가 아니라 '학습 범위가 덮는다' 로 읽어야")
    print("   한다. 모형 자체의 우열은 위의 교집합 표가 재는 것이다.)")


if __name__ == "__main__":
    sys.exit(main())
