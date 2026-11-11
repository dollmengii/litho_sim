"""
campaign.py -- 조건 공간 설계 및 대량 데이터 생성

문제
----
품질 기준(dataset.SampleSpec 의 CD 공차 예산에서 유도되는 NILS 하한과
확률 변동 상한)을 적용하면, 조건 공간을 균등 격자로 쓸었을 때 적격 샘플
비율이 20 % 수준으로 떨어진다. 나머지 80 % 는 연산을 수행한 뒤 버려지므로,
대량 생성 시 자원의 대부분이 낭비된다.

주의 : 기준값을 '업계 표준' 두 개(NILS 2, CD 3σ/CD 10 %)로 못박아 두었던
것은 10주차에 정정했다. 두 값은 독립 기준이 아니라 같은 CD 공차 예산을
나눠 쓰는 항목이므로, 전체 예산과 배분 비율에서 유도한다.

해법
----
1) 사전 탐색 : 결정론 연산만으로 적격 영역의 경계를 먼저 찾는다.
   확률 모델(Monte Carlo)은 결정론 연산보다 수십 배 비싸므로, 값싼
   지표(NILS)로 먼저 거른 뒤 통과한 조건에만 확률 모델을 적용한다.
2) 적응 격자 : 적격 영역 안에서만 조건을 샘플링한다.
3) 병렬 생성 : 조건별 연산이 독립이므로 다중 프로세스로 분산한다.
4) 체크포인트 : 중간 결과를 주기적으로 저장하여 중단 후 재개를 지원한다.
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import dataset as DS
from . import grid as G
from . import imaging as IM
from . import mask as MK
from . import process as PR
from . import resist as RS


def screen_condition(spec: DS.SampleSpec, fast: bool = True) -> dict:
    """확률 모델 없이 결정론 지표만으로 적격 가능성을 선별한다.

    Monte Carlo 를 수행하지 않으므로 전체 생성 대비 수십 배 빠르다.

    [fast] 선별은 '측정' 이 아니라 '거르개' 이므로 스칼라 모델로 수행한다.
    스칼라 모델은 TE 벡터 모델보다 대비를 높게 산출하므로(validate_w10
    Case 3 : 본 조건 공간 전역에서 8% 이내) 판정이 낙관적인 쪽으로만
    치우친다. 즉 통과시키지 말아야 할 조건을 통과시킬 수는 있어도,
    통과해야 할 조건을 탈락시키지는 않는다. 잘못 통과한 조건은 뒤따르는
    전체 생성 단계에서 걸러지므로 최종 데이터 품질에는 영향이 없다.
    벡터 모델로 선별하려면 fast=False 를 준다.
    """
    g = G.build_grid(pitch=spec.pitch, n_periods=spec.n_periods,
                     dx_target=spec.dx_target)
    m = MK.line_space(g, cd=spec.duty * spec.pitch,
                      mask_type=spec.mask_type, att_t=spec.att_t)
    target = spec.duty * spec.pitch
    opt = spec.optics()
    if fast:
        opt["polarization"] = IM.POL_SCALAR

    i_focus = IM.aerial_image(g, m, spec.wavelength, spec.na, defocus=0.0, **opt)
    e_op = PR.dose_to_size(g, i_focus, target, spec.threshold,
                           max_width=spec.pitch)
    if not np.isfinite(e_op):
        return {"pass": False, "nils": np.nan, "cd": np.nan, "reason": "dose-to-size 실패"}

    intensity = (
        i_focus if spec.defocus == 0.0 else
        IM.aerial_image(g, m, spec.wavelength, spec.na,
                        defocus=spec.defocus, **opt)
    )
    observed = intensity * e_op * spec.dose_rel
    det = RS.extract_cd(g.x, observed, spec.threshold, max_width=spec.pitch)
    if not det["formed"]:
        return {"pass": False, "nils": np.nan, "cd": np.nan, "reason": "패턴 미형성"}
    if det["nils"] < spec.min_nils:
        return {"pass": False, "nils": det["nils"], "cd": det["cd"],
                "reason": f"NILS {det['nils']:.2f} < {spec.min_nils}"}
    return {"pass": True, "nils": det["nils"], "cd": det["cd"], "reason": ""}


def feasible_region(
    base: DS.SampleSpec,
    defocus_range: tuple[float, float] = (-300.0, 300.0),
    dose_range: tuple[float, float] = (0.7, 1.3),
    n_coarse: int = 9,
    refine: bool = True,
    n_refine: int = 2,
) -> dict:
    """사전 탐색으로 적격 영역의 경계를 찾는다.

    결정론 선별만 수행하므로 전체 생성 대비 비용이 매우 낮다.

    성긴 격자에서는 적격 영역이 격자 간격보다 좁을 때 경계가 한 점으로
    퇴화하여, 이후 생성 격자가 한 축으로 몰린다. refine 이 True 면
    퇴화한 축에 대해 탐색 범위를 좁혀 재탐색한다.
    """
    dzs = np.linspace(*defocus_range, n_coarse)
    drs = np.linspace(*dose_range, n_coarse)
    grid_pass = np.zeros((n_coarse, n_coarse), dtype=bool)
    nils_map = np.full((n_coarse, n_coarse), np.nan)

    t0 = time.perf_counter()
    for i, dr in enumerate(drs):
        for j, dz in enumerate(dzs):
            sp = DS.SampleSpec(**{**base.__dict__, "defocus": float(dz),
                                  "dose_rel": float(dr)})
            r = screen_condition(sp)
            grid_pass[i, j] = r["pass"]
            nils_map[i, j] = r["nils"]
    elapsed = time.perf_counter() - t0

    if grid_pass.any():
        ii, jj = np.where(grid_pass)
        bounds = {
            "defocus": (float(dzs[jj.min()]), float(dzs[jj.max()])),
            "dose_rel": (float(drs[ii.min()]), float(drs[ii.max()])),
        }
    else:
        bounds = {"defocus": (0.0, 0.0), "dose_rel": (1.0, 1.0)}

    n_probe = n_coarse ** 2

    # 적격 영역이 격자 간격보다 좁아 경계가 퇴화한 경우 범위를 좁혀 재탐색
    if refine and grid_pass.any() and n_refine > 0:
        step_dz = dzs[1] - dzs[0]
        step_dr = drs[1] - drs[0]
        degenerate = (bounds["defocus"][1] - bounds["defocus"][0] < step_dz
                      or bounds["dose_rel"][1] - bounds["dose_rel"][0] < step_dr)
        if degenerate:
            c_dz = 0.5 * (bounds["defocus"][0] + bounds["defocus"][1])
            c_dr = 0.5 * (bounds["dose_rel"][0] + bounds["dose_rel"][1])
            sub = feasible_region(
                base,
                (c_dz - step_dz, c_dz + step_dz),
                (max(0.3, c_dr - step_dr), c_dr + step_dr),
                n_coarse, refine=True, n_refine=n_refine - 1,
            )
            sub["n_probe"] += n_probe
            sub["elapsed_s"] += elapsed
            sub["refined"] = True
            return sub

    return {
        "defocus_axis": dzs, "dose_axis": drs,
        "pass_map": grid_pass, "nils_map": nils_map,
        "bounds": bounds,
        "pass_rate": float(grid_pass.mean()),
        "n_probe": n_probe,
        "elapsed_s": elapsed,
        "refined": False,
    }


def design_grid(
    base: DS.SampleSpec,
    region: dict,
    n_defocus: int = 9,
    n_dose: int = 9,
    margin: float = 0.05,
) -> list[DS.SampleSpec]:
    """적격 영역 안에서만 조건 격자를 구성한다."""
    dz_lo, dz_hi = region["bounds"]["defocus"]
    dr_lo, dr_hi = region["bounds"]["dose_rel"]
    dz_pad = (dz_hi - dz_lo) * margin
    dr_pad = (dr_hi - dr_lo) * margin

    specs = []
    for dr in np.linspace(dr_lo - dr_pad, dr_hi + dr_pad, n_dose):
        for dz in np.linspace(dz_lo - dz_pad, dz_hi + dz_pad, n_defocus):
            specs.append(DS.SampleSpec(**{**base.__dict__,
                                          "defocus": float(dz),
                                          "dose_rel": float(dr)}))
    return specs


# ─────────────────── 다차원 조건 공간 설계 ───────────────────

#: 광학·패턴 조건의 기본 탐색 범위. 각 항목은 (하한, 상한).
#
# [10주차] 결함 요인 축 3개를 추가하였다. 선정 기준은 '현실성' 이 아니라
# '역추정 혼동을 새로 만드는가' 이다.
#   flare  : 대비를 떨어뜨리는 양상이 노광량 과다와 구별되지 않는다.
#   msd_nm : 레지스트 블러와 결정론적 상에서 동일한 연산이다. 다만 잡음
#            평활 여부가 달라 LER 로는 구별된다 (CD 만으로는 불가).
#   blur_nm: 문헌 통용 범위(stochastic.BLUR_RANGE_ARF) 안에서만 뽑는다.
#            9주차까지는 근거 없는 고정 자유 파라미터였다.
# 수차(z5~z11)와 편광·마스크·조명 형상은 이산 조건이므로 이 연속 공간이
# 아니라 캠페인 단위로 나누어 지정한다.
DEFAULT_SPACE = {
    "pitch": (100.0, 400.0),
    "duty": (0.30, 0.70),
    "na": (0.90, 1.35),
    "sigma_out": (0.40, 0.90),
    "flare": (0.0, 0.05),
    "msd_nm": (0.0, 6.0),
    "blur_nm": (5.0, 15.0),
}

#: 이산 조건의 조합. 캠페인을 이 조합별로 나누어 돌린다.
DISCRETE_ARMS = [
    {"polarization": "TE", "mask_type": "binary",  "source_shape": "conventional"},
    {"polarization": "TE", "mask_type": "att-psm", "source_shape": "conventional"},
    {"polarization": "TE", "mask_type": "att-psm", "source_shape": "dipole"},
    {"polarization": "unpolarized", "mask_type": "att-psm", "source_shape": "dipole"},
]


def sample_optical_conditions(
    n: int,
    space: dict | None = None,
    seed: int = 0,
) -> list[dict]:
    """광학·패턴 조건을 라틴 하이퍼큐브로 샘플링한다.

    균등 격자는 차원이 늘어날수록 조합 수가 기하급수로 증가하고
    각 축의 분해능이 떨어진다. 라틴 하이퍼큐브는 같은 표본 수로
    각 축을 고르게 덮으므로 다차원 조건 공간에 적합하다.
    """
    space = space or DEFAULT_SPACE
    rng = np.random.default_rng(seed)
    keys = list(space.keys())
    # 각 축을 n 등분하고 구간마다 하나씩 뽑은 뒤 축별로 섞는다
    cube = np.empty((n, len(keys)))
    for j, k in enumerate(keys):
        lo, hi = space[k]
        edges = np.linspace(lo, hi, n + 1)
        pts = edges[:-1] + rng.random(n) * np.diff(edges)
        cube[:, j] = rng.permutation(pts)
    return [dict(zip(keys, row)) for row in cube]


def _screen_one(args):
    """프로세스 풀용 단위 사전 탐색."""
    base_dict, opt, n_coarse = args
    local = DS.SampleSpec(**{**base_dict, **opt})
    dz_span = min(600.0, 3.0 * local.pitch)
    try:
        reg = feasible_region(local, (-dz_span, dz_span), (0.7, 1.3), n_coarse)
    except Exception:
        return opt, None
    return opt, reg


def plan_campaign(
    base: DS.SampleSpec,
    n_optical: int,
    n_defocus: int = 7,
    n_dose: int = 7,
    space: dict | None = None,
    n_coarse: int = 7,
    seed: int = 0,
    n_workers: int | None = None,
    progress=None,
    conditions: list[dict] | None = None,
) -> dict:
    """다차원 조건 공간 전체에 대한 생성 계획을 수립한다.

    광학 조건마다 적격 Defocus·Dose 영역이 다르므로, 조합별로
    사전 탐색을 수행한 뒤 해당 영역 안에서만 격자를 구성한다.
    사전 탐색은 결정론 연산만 사용하므로 전체 생성 대비 비용이 낮다.

    conditions
    ----------
    광학 조건을 직접 주면 라틴 하이퍼큐브를 돌리지 않고 그것을 쓴다.
    13 주차 보강에서 **그룹 안에서 수차를 흔들기** 위해 필요하다. 호출부가
    조건을 만들어야 하는 이유는, 그 구성이 '어느 축을 그룹 안에서 흔들고
    어느 축을 그룹 사이에서만 흔드는가' 라는 설계 결정이고 그것은 생성
    스크립트의 몫이기 때문이다. 사전 탐색은 조건마다 따로 돌므로 수차가
    달라지면 적격 영역도 각각 다시 잡힌다. 계획 **뒤에** 수차를 바꾸면
    적격 영역이 옛 수차의 것이어서 미형성 표본이 늘어난다.
    """
    optics = (list(conditions) if conditions is not None
              else sample_optical_conditions(n_optical, space, seed))
    specs, regions, skipped = [], [], 0
    t0 = time.perf_counter()

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    args = [(dict(base.__dict__), opt, n_coarse) for opt in optics]

    def _collect(opt, reg):
        nonlocal skipped
        if reg is None or reg["pass_rate"] == 0.0:
            skipped += 1
            return
        local = DS.SampleSpec(**{**base.__dict__, **opt})
        specs.extend(design_grid(local, reg, n_defocus, n_dose))
        regions.append({**opt, **reg["bounds"], "pass_rate": reg["pass_rate"],
                        "refined": reg.get("refined", False)})

    if n_workers == 1:
        for i, a in enumerate(args):
            _collect(*_screen_one(a))
            if progress:
                progress(i + 1, len(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, (opt, reg) in enumerate(ex.map(_screen_one, args, chunksize=4)):
                _collect(opt, reg)
                if progress:
                    progress(i + 1, len(args))

    return {
        "specs": specs,
        "regions": regions,
        "n_optical": n_optical,
        "n_skipped": skipped,
        "n_refined": sum(1 for r in regions if r.get("refined")),
        "screen_s": time.perf_counter() - t0,
    }


def _worker(spec: DS.SampleSpec):
    """프로세스 풀에서 실행되는 단위 작업."""
    try:
        return DS.make_sample(spec)
    except Exception:
        return None


def run_campaign(
    specs: list[DS.SampleSpec],
    n_workers: int | None = None,
    checkpoint: str | None = None,
    checkpoint_every: int = 50,
    progress=None,
) -> dict:
    """다중 프로세스로 데이터셋을 생성한다. 체크포인트로 재개를 지원한다.

    Parameters
    ----------
    n_workers : None 이면 CPU 수에서 1 을 뺀 값을 사용한다.
    checkpoint : 경로를 주면 주기적으로 중간 결과를 저장하고,
                 파일이 이미 있으면 이어서 진행한다.
    progress : callable(done, total) 형태의 진행 보고 함수.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    done_idx, results = set(), {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            saved = pickle.load(f)
        done_idx = set(saved["done_idx"])
        results = saved["results"]

    todo = [(k, sp) for k, sp in enumerate(specs) if k not in done_idx]
    t0 = time.perf_counter()

    def _save():
        if checkpoint:
            with open(checkpoint, "wb") as f:
                pickle.dump({"done_idx": list(done_idx), "results": results}, f)

    if n_workers == 1:
        for k, sp in todo:
            results[k] = _worker(sp)
            done_idx.add(k)
            if progress:
                progress(len(done_idx), len(specs))
            if checkpoint and len(done_idx) % checkpoint_every == 0:
                _save()
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker, sp): k for k, sp in todo}
            for fut in as_completed(futs):
                k = futs[fut]
                results[k] = fut.result()
                done_idx.add(k)
                if progress:
                    progress(len(done_idx), len(specs))
                if checkpoint and len(done_idx) % checkpoint_every == 0:
                    _save()
    _save()

    conds, profs, labs, tgts, kept = [], [], [], [], []
    # [10주차 보정] 품질 플래그가 늘어나도 깨지지 않도록 defaultdict 로 둔다.
    # 고정 딕셔너리로 두었더니 budget_ok 플래그를 추가한 순간 아래 집계가
    # KeyError 로 중단되었다. 품질 기준은 앞으로도 늘어날 수 있다.
    from collections import defaultdict
    reject = defaultdict(int)
    for k in range(len(specs)):
        r = results.get(k)
        if r is None:
            continue
        if r["ok"]:
            conds.append(r["condition"]); profs.append(r["profile"])
            labs.append(r["labels"]); tgts.append(r["target_cd"])
            # 어떤 입력 조건에서 나온 표본인지 기록한다. 이것이 없으면
            # 학습/검증을 광학 조건 단위로 나눌 때 표본과 조건을 맞출 수 없다.
            kept.append(k)
        else:
            for key, v in r["quality"].items():
                if not v:
                    reject[key] += 1
                    break

    return {
        "conditions": np.array(conds) if conds else np.zeros((0, len(DS.CONDITION_KEYS))),
        "profiles": np.array(profs) if profs else np.zeros((0, 1)),
        "labels": np.array(labs) if labs else np.zeros((0, len(DS.LABEL_KEYS))),
        "target_cd": np.array(tgts),
        "kept_index": np.array(kept, dtype=np.int64),
        "ok": np.ones(len(conds), dtype=bool),
        "condition_keys": DS.CONDITION_KEYS,
        "label_keys": DS.LABEL_KEYS,
        "reject": reject,
        "n_requested": len(specs),
        "n_workers": n_workers,
        "elapsed_s": time.perf_counter() - t0,
    }
