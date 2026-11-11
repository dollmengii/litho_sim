"""
splitting.py -- 혼동 집합 생성과 학습/검증 분할

왜 별도 모듈인가
----------------
식별 가능성을 평가하려면 데이터셋이 두 가지를 갖춰야 한다.

1. 혼동 집합 (confusion set)
   관측량은 거의 같은데 공정 조건은 서로 다른 조건들의 묶음. 라틴
   하이퍼큐브로 조건 공간을 고르게 덮으면 이런 묶음이 우연히만 생기므로,
   식별 한계를 재기에는 표본이 너무 적다. 의도적으로 구성해야 한다.
   이것이 전역 식별성(global identifiability)을 재는 재료이다.
   identify.py 의 Cramer-Rao 하한은 한 점에서의 미분에 기반하므로
   국소 식별성만 본다. 멀리 떨어진 두 조건이 같은 관측값을 내는 경우는
   잡아내지 못한다. 두 방법은 상보적이다.

2. 조건 단위 분할
   한 광학 조건에서 Defocus x Dose 격자로 수십 건이 나온다. 이들을
   무작위로 섞어 학습/검증으로 나누면 같은 광학 조건의 형제 표본이 양쪽에
   동시에 들어간다. 모델이 일반화한 것이 아니라 외운 것을 성능으로
   착각하게 되며, 식별 한계를 과소평가한다. 반드시 광학 조건을 통째로
   한쪽에 몰아야 한다.
"""

from dataclasses import replace

import numpy as np

from . import dataset as DS
from . import grid as G
from . import imaging as IM
from . import mask as MK
from . import process as PR
from . import resist as RS

#: 광학 조건을 식별하는 키. 이 값들이 같으면 같은 그룹으로 본다.
OPTICAL_KEYS = ("wavelength", "na", "n_medium", "sigma_out", "sigma_in",
                "polarization", "source_shape", "mask_type", "pitch", "duty")


def optical_group_id(spec: DS.SampleSpec) -> tuple:
    """분할 그룹 식별자. 같은 광학·패턴 조건이면 같은 값."""
    return tuple(
        round(v, 9) if isinstance(v, float) else v
        for v in (getattr(spec, k) for k in OPTICAL_KEYS)
    )


def grouped_split(specs: list[DS.SampleSpec],
                  fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                  seed: int = 0) -> dict:
    """광학 조건 단위로 학습/검증/시험을 나눈다.

    같은 광학 조건에서 나온 모든 표본은 같은 분할에 들어간다. 무작위
    분할과 비교하려면 shuffled_split 을 함께 쓴다.

    Returns
    -------
    {"train": [idx...], "val": [...], "test": [...], "n_groups": int}
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions 의 합은 1 이어야 합니다.")
    groups: dict[tuple, list[int]] = {}
    for i, s in enumerate(specs):
        groups.setdefault(optical_group_id(s), []).append(i)

    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_tr = int(round(n * fractions[0]))
    n_va = int(round(n * fractions[1]))
    parts = {"train": keys[:n_tr],
             "val": keys[n_tr:n_tr + n_va],
             "test": keys[n_tr + n_va:]}
    out = {k: sorted(i for g in v for i in groups[g]) for k, v in parts.items()}
    out["n_groups"] = n
    out["groups_per_split"] = {k: len(v) for k, v in parts.items()}
    # 그룹 수가 적으면 조건 단위 분할이 요청한 비율을 맞출 수 없다. 표본
    # 비율이 크게 어긋나면 알려 준다(그래도 누수 없는 분할이 우선이다).
    total = sum(len(v) for k, v in out.items() if k in ("train", "val", "test"))
    if total:
        got = tuple(len(out[k]) / total for k in ("train", "val", "test"))
        if max(abs(a - b) for a, b in zip(got, fractions)) > 0.15:
            out["warning"] = (
                f"광학 조건이 {n}개뿐이라 표본 비율이 요청({fractions})과 "
                f"다릅니다: {tuple(round(g, 3) for g in got)}. "
                f"조건 수를 늘리면 해소됩니다."
            )
    return out


def shuffled_split(n_items: int,
                   fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                   seed: int = 0) -> dict:
    """무작위 분할. 그룹 누수를 보여주기 위한 비교용이며 실사용 금지."""
    idx = np.arange(n_items)
    np.random.default_rng(seed).shuffle(idx)
    n_tr = int(round(n_items * fractions[0]))
    n_va = int(round(n_items * fractions[1]))
    return {"train": sorted(idx[:n_tr].tolist()),
            "val": sorted(idx[n_tr:n_tr + n_va].tolist()),
            "test": sorted(idx[n_tr + n_va:].tolist())}


def split_leakage(specs: list[DS.SampleSpec], split: dict) -> dict:
    """분할에 그룹 누수가 있는지 센다.

    한 광학 조건이 둘 이상의 분할에 걸쳐 있으면 누수이다.
    """
    where: dict[tuple, set] = {}
    for name in ("train", "val", "test"):
        for i in split.get(name, []):
            where.setdefault(optical_group_id(specs[i]), set()).add(name)
    leaked = {g: s for g, s in where.items() if len(s) > 1}
    n_leaked_items = sum(
        1 for s in specs if optical_group_id(s) in leaked
    )
    return {"n_groups": len(where), "n_leaked_groups": len(leaked),
            "n_leaked_items": n_leaked_items,
            "leak_rate": n_leaked_items / max(len(specs), 1)}


# ───────────────────────── 혼동 집합 ─────────────────────────

def confusion_set(base: DS.SampleSpec,
                  target_cd: float | None = None,
                  n: int = 16,
                  vary: tuple[str, ...] = ("defocus", "flare", "msd_nm", "blur_nm"),
                  ranges: dict | None = None,
                  tol: float = 0.005,
                  seed: int = 0,
                  max_tries: int = 400) -> list[DS.SampleSpec]:
    """관측 선폭이 같아지도록 Dose 를 맞춘 조건 묶음을 만든다.

    절차
    ----
    1. vary 에 준 변수들을 무작위로 뽑는다.
    2. 그 조건에서 목표 CD 를 만드는 Dose 를 역산한다(dose-to-size).
    3. 역산이 성공하고 실제 CD 가 목표에 tol 이내로 맞으면 채택한다.

    결과는 '결정론적 선폭만 보면 구별되지 않는' 조건들의 묶음이다.
    프로파일 전체나 LER 까지 보면 구별되는지가 12주차 평가 대상이 된다.

    Parameters
    ----------
    target_cd : 맞출 선폭. None 이면 base 의 duty*pitch.
    tol : 목표 대비 허용 상대오차.
    """
    rng = np.random.default_rng(seed)
    rng_default = {
        # [11주차 정정] 이전에는 (0, 3*pitch) 로 **양수만** 뽑았다. 초점에
        # 대한 짝함수성(I(+dz) = I(-dz))을 이용해 탐색 공간을 절반으로 줄인
        # 것이고, 수차가 없을 때는 타당하다.
        #
        # 그런데 그 최적화가 이 과제의 주장 2 를 보일 수 없게 만든다.
        #   - 짝함수성 자체의 직접 증거는 "+dz 와 -dz 가 관측으로 완전히
        #     같다" 는 쌍이다. 양수만 뽑으면 그 쌍을 **구성조차 할 수 없다.**
        #     혼동 집합은 '관측은 같은데 조건이 다른' 묶음을 만드는 장치인데,
        #     가장 정확한 예 하나가 구조적으로 빠져 있었다.
        #   - 더 중요한 것은 Z6·Z11 을 조건 공간에 넣은 뒤로 짝함수성이
        #     **깨진다**는 점이다(verify.py C5). 깨지는 것을 보려면 +dz 와
        #     -dz 를 둘 다 가져야 한다. 양수만 있으면 깨짐을 관측할 수 없다.
        # 그래서 대칭으로 바꾼다. 수차가 0 인 조건에서는 두 부호가 같은
        # 관측을 주어 짝함수성의 직접 증거가 되고, Z6·Z11 이 있는 조건에서는
        # 갈려서 '부호를 되짚는 처방' 의 증거가 된다. 양쪽 다 필요하다.
        "defocus": (-3.0 * base.pitch, 3.0 * base.pitch),
        "flare": (0.0, 0.05),
        "msd_nm": (0.0, 6.0),
        "blur_nm": (5.0, 15.0),
        "duty": (max(0.3, base.duty - 0.05), min(0.7, base.duty + 0.05)),
    }
    rng_use = {**rng_default, **(ranges or {})}
    target = target_cd if target_cd is not None else base.duty * base.pitch

    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        tries += 1
        over = {}
        for k in vary:
            lo, hi = rng_use[k]
            over[k] = float(rng.uniform(lo, hi))
        sp = replace(base, **over)

        g = sp_grid(sp)
        m = MK.line_space(g, cd=sp.duty * sp.pitch,
                          mask_type=sp.mask_type, att_t=sp.att_t)
        try:
            i_focus = IM.aerial_image(g, m, sp.wavelength, sp.na,
                                      defocus=sp.defocus, **sp.optics())
        except ValueError:
            continue
        e_op = PR.dose_to_size(g, i_focus, target, sp.threshold,
                               max_width=sp.pitch)
        if not np.isfinite(e_op):
            continue
        r = RS.extract_cd(g.x, i_focus * e_op, sp.threshold, max_width=sp.pitch)
        if not r["formed"] or not np.isfinite(r["cd"]):
            continue
        if abs(r["cd"] - target) / target > tol:
            continue

        # [중요] dataset.make_sample 은 노광량을 '최적 초점' 에서 다시 보정한
        # 뒤 dose_rel 을 곱한다. 따라서 여기서 해당 초점오차에 맞춰 구한
        # e_op 를 그대로 두면 덮어써져 축퇴가 깨진다(실측: 산포 0.16 nm →
        # 7.17 nm). make_sample 의 보정을 상쇄하도록 비율로 환산해 넘긴다.
        #     observed = I(dz) * e_op_focus * dose_rel
        #              = I(dz) * e_op_dz           <- dose_rel = e_op_dz/e_op_focus
        i_best = IM.aerial_image(g, m, sp.wavelength, sp.na,
                                 defocus=0.0, **sp.optics())
        e_focus = PR.dose_to_size(g, i_best, target, sp.threshold,
                                  max_width=sp.pitch)
        if not np.isfinite(e_focus) or e_focus <= 0:
            continue
        out.append(replace(sp, dose_rel=float(e_op / e_focus)))
    return out


def sp_grid(sp: DS.SampleSpec) -> G.SimGrid:
    return G.build_grid(pitch=sp.pitch, n_periods=sp.n_periods,
                        dx_target=sp.dx_target)


def confusion_spread(specs: list[DS.SampleSpec]) -> dict:
    """혼동 집합의 조건 산포와 관측량 산포를 함께 보고한다.

    조건은 넓게 흩어져 있는데 관측량(선폭)은 거의 같다면, 그 관측량만으로는
    조건을 되짚을 수 없다는 직접적인 증거가 된다.
    """
    if not specs:
        return {"n": 0}
    cds, rows = [], {}
    for sp in specs:
        g = sp_grid(sp)
        m = MK.line_space(g, cd=sp.duty * sp.pitch,
                          mask_type=sp.mask_type, att_t=sp.att_t)
        i = IM.aerial_image(g, m, sp.wavelength, sp.na,
                            defocus=sp.defocus, **sp.optics())
        e_op = PR.dose_to_size(g, i, sp.duty * sp.pitch, sp.threshold,
                               max_width=sp.pitch)
        r = RS.extract_cd(g.x, i * e_op, sp.threshold, max_width=sp.pitch)
        cds.append(r["cd"])
        for k in ("defocus", "flare", "msd_nm", "blur_nm"):
            rows.setdefault(k, []).append(getattr(sp, k))
    cds = np.array(cds, dtype=float)
    return {
        "n": len(specs),
        "cd_mean": float(np.nanmean(cds)),
        "cd_spread": float(np.nanmax(cds) - np.nanmin(cds)),
        "condition_ranges": {k: (float(np.min(v)), float(np.max(v)))
                             for k, v in rows.items()},
    }
