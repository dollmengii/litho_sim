"""train_surrogate.py -- 시뮬레이터의 대리 모형(surrogate)을 학습한다.

무엇을 하는 것인가
------------------
시뮬레이터는 한 조건당 약 0.5 초가 걸린다. 12 주차의 추천기는 **조건 공간을
탐색**해야 하므로 한 번 추천할 때 수천~수만 번의 평가가 필요하다. 시뮬레이터로
직접 하면 한 번 추천에 몇 시간이 걸린다. 그래서 시뮬레이터를 근사하는 빠른
함수를 데이터로 학습해 두고, 탐색은 그 위에서 한다. 이것이 대리 모형이다.

  조건 14 개  ->  [대리 모형]  ->  cd_mean, ler_3sigma, nils, contrast, e_op

**추천기가 아니다.** 이것은 순방향 모형이다. 조건을 넣으면 결과가 나온다.
추천은 이 모형을 거꾸로 푸는 것이 아니라, 이 모형을 **목적함수로 삼아
탐색**하는 것이다(12 주차). 역방향으로 회귀를 학습하면 안 되는 이유는
7 주차에서 보인 비식별성 때문이다. 목표 CD 를 만드는 조건이 여러 개 있을 때
역회귀는 그 여러 답의 **평균**을 내놓고, 평균은 어느 답도 아니다.

왜 이 파일이 '정확도 보고' 에 공을 들이는가
-------------------------------------------
대리 모형의 전체 평균 오차는 이 데이터에서 **오해를 부른다.** 병합 데이터는
flare·msd_nm·z6·z11 축에서 약 77 대 23 으로 기울어 있다(원본 9.1 만 건은 이
축들이 거의 0, 보강 2.7 만 건만 변한다). 전체 평균은 표본이 많은 쪽을 따라가므로,
"평균 오차 0.1 nm" 가 "플레어가 큰 영역에서도 0.1 nm" 를 뜻하지 않는다.
그래서 **조건 영역별로 쪼개서** 보고한다.

또 하나. 라벨 자체가 몬테카를로 추정값이므로 **잡음 하한**이 있다. 대리 모형이
그 하한보다 작은 오차를 낸다면 물리가 아니라 잡음을 외운 것이다. 그래서
라벨의 표준오차를 추정해 함께 보고한다.

  cd_mean  : N = n_rows x n_trials = 1024 x 8 = 8192 개 폭의 평균이므로
             SE ~ (cd_3sigma / 3) / sqrt(N)
  ler_3sigma : 시행별 표준편차(표본 1024)를 8 회 평균한 값이므로
             상대 SE ~ 1/sqrt(2(1024-1)) / sqrt(8) ~ 0.78 %

이 하한은 추정이다. 행 사이 상관이 0 이라는 가정에서 나오므로 실제 하한은
이보다 **크다**(상관이 있으면 유효 표본이 줄어든다). 즉 보수적인 방향이 아니다.
그래서 '이 값 근처면 더 조여도 의미 없다' 는 판단에만 쓰고, 합격선으로 쓰지
않는다.

사용법
------
    py train_surrogate.py --data dataset/merged.npz
    py train_surrogate.py --data dataset/merged.npz --model hgb
    py train_surrogate.py --data dataset/merged.npz --quick    # 빠른 확인
    py train_surrogate.py --data dataset/merged.npz --profiles # 프로파일까지

merge_dataset.py 는 출력 경로를 사용자가 지정하므로 정해진 파일 이름이 없다.
병합할 때 준 경로를 그대로 주면 된다. 틀리면 주변 .npz 를 찾아 목록을 보여
준다.

종료 코드
---------
0 : 학습과 보고를 끝냈다. **정확도가 충분하다는 뜻이 아니다.**
1 : 데이터가 학습에 쓸 수 없는 상태다(분할 없음, 라벨 전부 NaN 등).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from litho import dataset as DS

#: 학습 대상 라벨. 전체 9 개 가운데 5 개만 쓴다.
#:
#:   dose_clear_mj       : 상수(9.0). threshold x dose_mj 이고 둘 다 고정이다.
#:   unresolved_row_frac : 한 값에 100 % 몰려 있다. 분산이 없으면 배울 것이 없다.
#:   cd_3sigma           : 뺀다. cd_deterministic 과 달리 이것은 확률 라벨이지만,
#:                         추천기가 쓰는 품질 기준은 ler_3sigma 와 nils 이고
#:                         cd_3sigma 는 그 둘과 강하게 중복된다. 필요하면
#:                         --labels 로 넣을 수 있다.
#:   cd_deterministic    : 뺀다. 블러를 반영하지 않는 라벨이라(알려진 한계,
#:                         VERIFICATION_LIMITS 참조) 추천 기준으로 쓰면 안 된다.
#:   cd_mean             : **직접 쓰지 않는다.** 아래 cd_error 참조.
#:
#: [중요] cd_mean 을 그대로 목표로 두면 안 된다.
#:
#: dataset.py 에서 target_cd = duty x pitch 이고, duty 와 pitch 는 둘 다 입력
#: 조건이다. 게다가 make_sample 이 dose_to_size 로 노광량을 맞춰 CD 를 목표에
#: 붙인다. 그래서
#:
#:     cd_mean ~ duty x pitch + (작은 편차)
#:
#: 가 된다. 목표 CD 는 데이터 안에서 30~280 nm 로 흔들리는데 정작 의미 있는
#: 편차는 1 nm 대다. 즉 **분산의 대부분이 두 입력의 곱이라는 자명한 성분**이고,
#: 어떤 모형이든 R^2 가 0.999 근처로 나온다. 그 숫자는 "리소그래피를 배웠다"
#: 가 아니라 "곱셈을 배웠다" 는 뜻이다.
#:
#: 추천기가 실제로 알아야 하는 것은 '이 조건이 목표를 맞추는가', 즉 **편차**다.
#: 그래서 파생 라벨을 만들어 쓴다.
#:
#:     cd_error = cd_mean - target_cd        [nm]
#:
#: cd_mean 도 --labels 로 넣을 수 있고, 넣으면 위 함정을 분산 분해로 함께
#: 인쇄한다. 보고서에 쓸 숫자는 cd_error 쪽이다.
DEFAULT_LABELS = ("cd_error", "ler_3sigma", "nils", "contrast", "e_op")

#: 저장된 배열에서 계산해 만드는 라벨. (이름, 필요한 것, 설명)
DERIVED = {
    "cd_error": "cd_mean - target_cd. 목표 CD 대비 편차 [nm]",
}

#: 정수 부호로 저장된 범주형 조건. 연속값으로 다루면 '코드 1 과 3 의 중간' 같은
#: 없는 조건을 만들어 낸다.
CATEGORICAL = ("pol_code", "source_code", "mask_code")

#: 몬테카를로 라벨의 표본 수. SampleSpec 의 기본값과 같아야 한다.
N_ROWS_DEFAULT = 1024
N_TRIALS_DEFAULT = 8


# ----------------------------------------------------------------------------
# 데이터 준비
# ----------------------------------------------------------------------------
#: 데이터를 못 찾았을 때 훑어 볼 위치. merge_dataset.py 는 출력 경로를
#: 사용자가 지정하므로 '정해진 이름' 이 없다. 그래서 짐작하지 말고 실제로
#: 찾아서 보여 준다.
SEARCH_DIRS = (".", "dataset", "data", "..")


def find_candidates() -> list[str]:
    """주변의 .npz 를 찾아 크기와 함께 돌려준다."""
    import os
    seen, out = set(), []
    for dirp in SEARCH_DIRS:
        try:
            names = sorted(os.listdir(dirp))
        except OSError:
            continue
        for nm in names:
            if not nm.endswith(".npz"):
                continue
            p = os.path.normpath(os.path.join(dirp, nm))
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            try:
                mb = os.path.getsize(p) / 1024 ** 2
            except OSError:
                continue
            out.append(f"{p}  ({mb:,.1f} MB)")
    return out


def load_data(path: str) -> dict:
    try:
        d = DS.load(path)
    except FileNotFoundError:
        msg = [f"[오류] 파일이 없습니다 : {path}",
               "",
               "       merge_dataset.py 는 출력 경로를 직접 지정하게 되어",
               "       있으므로 정해진 파일 이름이 없습니다. 병합할 때 첫",
               "       인자로 준 경로를 --data 에 그대로 주십시오."]
        cands = find_candidates()
        if cands:
            msg += ["", "       주변에서 찾은 .npz :"]
            msg += [f"         {c}" for c in cands]
            msg += ["",
                    "       가장 큰 것이 병합 결과일 가능성이 높습니다.",
                    "       확실하지 않으면 먼저 확인하십시오 :",
                    "         py inspect_dataset.py <경로>"]
        else:
            msg += ["", "       주변에 .npz 가 없습니다. 아직 생성·병합을",
                    "       하지 않았다면 실행방법.txt 의 [4] 를 보십시오."]
        raise SystemExit("\n".join(msg))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[오류] '{path}' 를 읽을 수 없습니다 : {e}")
    need = ("conditions", "labels")
    for k in need:
        if k not in d:
            raise SystemExit(f"[오류] '{k}' 가 없습니다. 데이터셋이 맞습니까?")
    return d


def resolve_keys(d: dict) -> tuple[list[str], list[str]]:
    """저장된 열 이름을 쓴다. 코드의 상수를 쓰면 스키마가 바뀐 데이터와
    조용히 어긋난다."""
    ck = [str(s) for s in d["condition_keys"]] if "condition_keys" in d \
        else list(DS.CONDITION_KEYS)
    lk = [str(s) for s in d["label_keys"]] if "label_keys" in d \
        else list(DS.LABEL_KEYS)
    return ck, lk


def varying_columns(cond: np.ndarray, ck: list[str]) -> list[int]:
    """값이 둘 이상인 조건 열의 색인. 상수 열은 학습에 정보를 주지 않는다."""
    out = []
    for i in range(len(ck)):
        if np.unique(cond[:, i]).size > 1:
            out.append(i)
    return out


def add_derived(d: dict, lab: np.ndarray,
                lk: list[str]) -> tuple[np.ndarray, list[str]]:
    """파생 라벨을 라벨 배열에 덧붙인다.

    cd_error 를 만드는 것이 핵심이다. 이유는 DEFAULT_LABELS 위의 주석 참조.
    """
    cols, names = [], list(lk)
    if "cd_mean" in lk and "target_cd" in d:
        tgt = np.asarray(d["target_cd"], dtype=np.float64).reshape(-1)
        if tgt.shape[0] == lab.shape[0]:
            cols.append(lab[:, lk.index("cd_mean")] - tgt)
            names.append("cd_error")
    if not cols:
        return lab, names
    return np.column_stack([lab] + cols), names


def variance_split(d: dict, lab: np.ndarray, lk: list[str],
                   rows: np.ndarray) -> dict | None:
    """cd_mean 의 분산을 '목표 CD 성분' 과 '편차 성분' 으로 쪼갠다.

    cd_mean 을 목표로 둘 때 R^2 가 왜 의미 없는지를 숫자로 보이기 위한 것이다.
    """
    if "cd_mean" not in lk or "target_cd" not in d:
        return None
    tgt = np.asarray(d["target_cd"], dtype=np.float64).reshape(-1)
    if tgt.shape[0] != lab.shape[0]:
        return None
    cm = lab[rows, lk.index("cd_mean")]
    tg = tgt[rows]
    good = np.isfinite(cm) & np.isfinite(tg)
    if good.sum() < 10:
        return None
    cm, tg = cm[good], tg[good]
    v_t = float(np.var(tg))
    v_e = float(np.var(cm - tg))
    return {"std_target": float(np.sqrt(v_t)),
            "std_error": float(np.sqrt(v_e)),
            "ratio": float(np.sqrt(v_t / v_e)) if v_e > 0 else float("inf")}


def finite_mask(lab: np.ndarray, lk: list[str], targets: list[str]) -> np.ndarray:
    """학습에 쓸 라벨이 모두 유한한 행."""
    idx = [lk.index(t) for t in targets]
    return np.all(np.isfinite(lab[:, idx]), axis=1)


def get_split(d: dict, n: int, keep: np.ndarray) -> dict[str, np.ndarray]:
    """저장된 그룹 분할을 쓰고, 유효 행만 남긴다.

    분할을 다시 만들지 않는 것이 중요하다. merge_dataset 이 광학 조건 그룹
    단위로 나눈 것이므로, 여기서 무작위로 다시 나누면 같은 광학 조건이
    train 과 test 에 함께 들어가 **누설**이 생긴다.
    """
    if "split_train" not in d:
        raise SystemExit(
            "[오류] 분할 정보가 없습니다. merge_dataset.py 로 병합하거나\n"
            "       splitting.grouped_split 으로 분할을 만들어 저장하십시오.\n"
            "       여기서 무작위 분할을 만들면 광학 조건이 train/test 에\n"
            "       함께 들어가 누설이 생깁니다.")
    keep_set = np.flatnonzero(keep)
    lookup = np.zeros(n, dtype=bool)
    lookup[keep_set] = True
    out = {}
    for part in ("train", "val", "test"):
        raw = np.asarray(d[f"split_{part}"], dtype=np.int64)
        if raw.size and (raw.min() < 0 or raw.max() >= n):
            raise SystemExit(
                f"[오류] split_{part} 에 범위를 벗어난 색인이 있습니다 "
                f"([{raw.min()}, {raw.max()}], 데이터 {n} 건). 분할이 다른"
                f" 데이터셋에서 온 것인지 확인하십시오.")
        out[part] = raw[lookup[raw]]
    return out


def check_disjoint(split: dict[str, np.ndarray]) -> None:
    """분할이 서로 겹치지 않는지. 겹치면 모든 지표가 낙관적으로 나온다."""
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = np.intersect1d(split[a], split[b])
        if inter.size:
            raise SystemExit(f"[오류] {a} 와 {b} 가 {inter.size} 건 겹칩니다.")


#: 광학 조건 그룹을 정의하는 열.
#:
#: [중복을 없앴다] 예전에는 여기에 목록을 그대로 다시 적어 두었다. 같은
#: 목록이 merge_dataset 에도 있어서, 축을 하나 추가하면 한쪽만 고쳐질 수
#: 있다. 그러면 병합은 새 그룹으로 나누는데 여기서는 옛 그룹으로 누설을
#: 검사하게 되어, **누설이 있어도 없다고 보고한다.** 그래서 정의를 한
#: 곳에 두고 들여온다. merge_dataset 은 모듈 수준에 실행 코드가 없으므로
#: 들여와도 안전하다(main 은 __name__ 가드 안에 있다).
from merge_dataset import GROUP_KEYS  # noqa: E402  (의도적으로 여기서)


def check_group_leakage(cond: np.ndarray, ck: list[str],
                        split: dict[str, np.ndarray]) -> dict:
    """같은 광학 조건 그룹이 train 과 val/test 에 함께 들어갔는지 직접 본다.

    색인 중복 검사만으로는 부족하다. 색인이 전부 달라도 **같은 광학 조건**이
    양쪽에 있으면, 모형은 검증 때 사실상 본 조건을 다시 보는 셈이 되어
    오차가 낙관적으로 나온다. merge_dataset 이 그룹 단위로 나누었으므로
    누설이 없어야 하지만, 병합된 데이터는 두 번의 생성을 합친 것이므로
    '없어야 한다' 를 '없다' 로 바꿔 두는 것이 맞다.

    [누설이 지표를 얼마나 부풀리는가 - 실측]
    616 건 시험 데이터에서 그룹 분할과 무작위 분할을 비교하면

        그룹 분할 (올바름) : cd_error R^2 = 0.381
        무작위 분할 (누설) : cd_error R^2 = 0.723

    R^2 가 거의 두 배가 된다. 같은 광학 조건의 이웃을 검증 때 다시 보기
    때문이다. 그룹 분할을 쓰는 근거가 이 숫자다.

    [주의] 같은 비교에서 RMSE 는 6.67 -> 13.84 nm 로 **커진다**. 두 분할의
    검증 집합이 서로 다르고 참값 산포도 다르기 때문이다(무작위 분할의 검증
    집합이 14 개 그룹에 걸쳐 더 넓게 퍼져 있다). 즉 **검증 집합이 다르면
    RMSE 를 그대로 비교할 수 없다.** 분할 방식을 비교할 때는 R^2 처럼
    산포로 정규화된 지표를 쓰거나 검증 집합을 고정해야 한다. 영역별 표에
    '참값 sd' 를 함께 인쇄하는 이유도 같다.
    """
    cols = [ck.index(k) for k in GROUP_KEYS if k in ck]
    if not cols:
        return {"checked": False}

    def gid(idx: np.ndarray) -> set:
        if idx.size == 0:
            return set()
        return {tuple(r) for r in np.round(cond[idx][:, cols], 9)}

    g = {p: gid(split[p]) for p in ("train", "val", "test")}
    leaks = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = g[a] & g[b]
        if shared:
            leaks[f"{a}-{b}"] = len(shared)

    # [음성 대조] '누설 없음' 이 탐지 능력이 없어서 나온 결과가 아님을 보인다.
    # train 의 한 행을 val 에 섞은 가짜 분할을 만들어, 같은 검사가 그것을
    # 잡는지 확인한다. 이 자기 시험이 실패하면 위 결과는 믿을 수 없다.
    selftest = None
    if split["train"].size and split["val"].size:
        planted = dict(split)
        planted["val"] = np.concatenate([split["val"], split["train"][:1]])
        pg = {p: gid(planted[p]) for p in ("train", "val")}
        selftest = bool(pg["train"] & pg["val"])

    return {"checked": True, "n_groups": {p: len(v) for p, v in g.items()},
            "leaks": leaks, "selftest_detects_planted": selftest}


# ----------------------------------------------------------------------------
# 잡음 하한
# ----------------------------------------------------------------------------
def noise_floor(lab: np.ndarray, lk: list[str], rows: np.ndarray,
                n_rows: int, n_trials: int) -> dict[str, float]:
    """라벨의 몬테카를로 표준오차 추정. 가정과 한계는 파일 머리글 참조."""
    out: dict[str, float] = {}
    n_eff = float(n_rows * n_trials)

    if "cd_3sigma" in lk and "cd_mean" in lk:
        s3 = lab[rows, lk.index("cd_3sigma")]
        good = np.isfinite(s3)
        if good.any():
            # cd_mean 은 N 개 폭의 평균. SE = sigma / sqrt(N), sigma = s3/3.
            out["cd_mean"] = float(np.mean(s3[good] / 3.0) / np.sqrt(n_eff))

        # cd_error = cd_mean - target_cd 이고 target_cd 는 결정론적 상수다.
        # 따라서 잡음은 cd_mean 과 같다.
        if "cd_mean" in out:
            out["cd_error"] = out["cd_mean"]

    if "ler_3sigma" in lk:
        ler = lab[rows, lk.index("ler_3sigma")]
        good = np.isfinite(ler)
        if good.any():
            # 표준편차 추정량의 상대 SE = 1/sqrt(2(n-1)). 시행 평균으로
            # sqrt(n_trials) 만큼 줄어든다.
            rel = 1.0 / np.sqrt(2.0 * (n_rows - 1)) / np.sqrt(n_trials)
            out["ler_3sigma"] = float(np.mean(ler[good]) * rel)
    return out


# ----------------------------------------------------------------------------
# 조건 영역
# ----------------------------------------------------------------------------
def regions(cond: np.ndarray, ck: list[str],
            rows: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """검증 오차를 쪼개 볼 영역. 전체 평균이 숨기는 것을 드러내는 것이 목적이다.

    기울어진 축(flare, msd_nm, z6, z11)이 기본값인지로 나눈다. 원본 9.1 만
    건은 이 축들이 거의 0 이고 보강 2.7 만 건만 변하므로, 두 영역의 오차가
    다르면 전체 평균은 많은 쪽(기본값 영역)을 따라간다.
    """
    skewed = [k for k in ("flare", "msd_nm", "z6", "z11") if k in ck]
    if not skewed:
        return [("전체", rows)]

    sub = cond[rows]
    at_base = np.ones(len(rows), dtype=bool)
    for k in skewed:
        at_base &= (sub[:, ck.index(k)] == 0.0)

    out = [("전체", rows),
           ("기본 영역 (결함 요인 전부 0)", rows[at_base]),
           ("보강 영역 (하나 이상 0 아님)", rows[~at_base])]

    # 축별로도 상·하위를 본다. 한 축만 커도 오차가 커지는지 보는 것이다.
    for k in skewed:
        v = sub[:, ck.index(k)]
        nz = np.abs(v) > 0        # 전 축이 0 이상이지만 부호에 무관하게 쓴다
        if nz.sum() < 50:
            continue
        hi = np.zeros(len(rows), dtype=bool)
        thr = np.quantile(np.abs(v[nz]), 0.75)
        hi[nz] = np.abs(v[nz]) >= thr
        if hi.sum() >= 50:
            out.append((f"{k} 상위 25 %", rows[hi]))
    return [(nm, ix) for nm, ix in out if ix.size > 0]


def group_ids(cond: np.ndarray, ck: list[str], rows: np.ndarray) -> int:
    """이 행들이 덮는 광학 조건 그룹의 개수."""
    cols = [ck.index(k) for k in GROUP_KEYS if k in ck]
    if not cols or rows.size == 0:
        return 0
    return len({tuple(r) for r in np.round(cond[rows][:, cols], 9)})


def training_domain(X, feat, cat_mask, rows, const_names, const_vals):
    """학습 조건의 범위를 기록한다. 12 주차 탐색의 **상자 제약**이 된다.

    왜 반드시 저장해야 하는가
    -------------------------
    추천기는 예측 오차를 최소화하는 조건을 찾는다. 그런데 mlp 는 `tanh`
    이므로 학습 범위 밖에서 **포화**한다. 포화 구간은

      - 데이터의 제약을 전혀 받지 않고,
      - 기울기가 0 이어서 평평한 최적점처럼 보인다.

    따라서 탐색은 범위 밖을 우연히 밟는 것이 아니라 **적극적으로 끌려간다.**
    그러면 추천기가 '예측 성능 훌륭함' 을 보고하고, 13 주차 되돌림 검증에서야
    전부 틀렸다는 것이 드러난다. 범위를 모르면 이 실패를 막을 수 없다.

    `campaign.DEFAULT_SPACE` 로 대신할 수 없다. 적격 필터가 범위를 줄이고,
    `defocus`·`dose_rel`·`z6`·`z11` 은 거기에 아예 없다. 실제 학습에 쓴
    행의 범위만이 정답이다.

    min/max 와 함께 1·99 백분위도 남긴다. min/max 는 이상치 한 점이
    정하므로, 그 상자의 구석에는 데이터가 없을 수 있다. 보수적으로 탐색할
    때는 백분위 상자를 쓴다.

    [범위] 상자는 **필요조건이지 충분조건이 아니다.** 축별 범위 안이라도
    축의 조합으로 보면 데이터가 없는 구석이 있다(상자는 볼록 껍질보다
    크다). 상자 제약은 노골적인 외삽만 막는다. 추천 조건이 실제로 맞는지는
    13 주차에 시뮬레이터로 되돌려 재야 한다.
    """
    Xt = np.asarray(X, dtype=np.float64)[rows]
    dom = {"n_train_rows": int(Xt.shape[0]), "axes": {}, "categorical": {},
           "held_constant": {}}
    for j, f in enumerate(feat):
        col = Xt[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        if cat_mask[j]:
            # 범주형은 범위가 아니라 **실제로 나온 수준**이 제약이다.
            dom["categorical"][f] = sorted(
                float(v) for v in np.unique(np.round(col, 9)))
            continue
        dom["axes"][f] = {
            "min": float(col.min()), "max": float(col.max()),
            "p01": float(np.percentile(col, 1)),
            "p99": float(np.percentile(col, 99)),
        }
    # [13차 보완 85 번] 상자 검사가 **얼마나 약한지** 재어 보니 심각했다.
    # 위 docstring 이 "상자는 필요조건이지 충분조건이 아니다" 를 적고 있었고
    # 그 말은 맞다. 그런데 그 약함의 크기를 아무도 재지 않았고, 추천기는
    # 사용자에게 아무 말도 하지 않았다.
    #
    # 10 차원 모사 측정 : 각 축 90 백분위 점은 학습점의 **94.9 %** 보다
    # 데이터에서 멀고, 상자 꼭짓점은 **100 %** 보다 멀다(최근접 2.718 대
    # 학습점 최대 2.390). 그런데 둘 다 상자 검사를 통과한다. 그리고
    # `roundtrip_w13.tool_presets` 가 하는 일이 바로 '각 축 상단 90 %' 다.
    #
    # 그래서 **결합 분포를 요약해 번들이 들고 간다.** 표준화한 부분표본과
    # 학습점 자신의 최근접 거리 분위수를 남기면, 추천기가 질의점의 최근접
    # 거리를 그 분포와 대조해 '학습점의 몇 % 보다 먼가' 를 말할 수 있다.
    # 상자 안/밖의 이진 판정이 **연속적인 외삽 거리**로 바뀐다.
    #
    # 왜 부분표본인가 : 전체를 담으면 번들이 커지고, 분위수만 담으면 질의점
    # 거리를 계산할 수 없다. 3000 행 x 연속축 수 x float32 면 수백 KB 다.
    # **쓸어가는 두 축(`defocus`, `dose_rel`)은 뺀다.** 추천기는 그 둘을
    # 격자로 훑으므로 '질의점' 이 하나로 정해지지 않는다. 그리고 물어야 할
    # 것은 "이 **광학·공구 조건**이 학습 데이터에 있는가" 이지 "이 초점·
    # 노광 격자점이 있는가" 가 아니다. 거울을 적용한 후보의 초점 부호 문제도
    # 이렇게 하면 생기지 않는다(초점이 거리에 안 들어간다).
    SWEPT = ("defocus", "dose_rel")
    con_cols = [j for j, f in enumerate(feat)
                if not cat_mask[j] and feat[j] not in SWEPT]
    if con_cols and Xt.shape[0] >= 50:
        Z = Xt[:, con_cols]
        good = np.all(np.isfinite(Z), axis=1)
        Z = Z[good]
        if Z.shape[0] >= 50:
            mu = Z.mean(axis=0)
            sd = Z.std(axis=0)
            sd[sd <= 0] = 1.0                 # 상수 축은 거리에 안 넣는다
            Zs = (Z - mu) / sd
            # [13차 보완 90 번] **반드시 중복을 지워야 한다.** 위에서 쓸어
            # 가는 두 축을 거리에서 뺐는데, 그 축으로만 다른 행은 남은
            # 부공간에서 **완전히 같은 점**이 된다. 데이터셋은 광학점 하나를
            # 초점·노광 격자로 훑으므로 한 조건이 수십 번 중복된다.
            #
            # 지우지 않으면 참조 분포가 0 으로 가득 찬다. 실측(작은 데이터셋,
            # 223 행) : 서로 다른 행 14 개(6.3 %), 최근접 거리의 **95.1 % 가
            # 정확히 0**, 5·50·95 백분위가 모두 0.0. 외삽 백분위는
            # `100 x mean(nn < d)` 이므로 거리가 0 보다 조금만 크면 즉시
            # 95 % 가 나온다. 그러면 추천기는 후보가 어디 있든 "외삽에
            # 가깝다" 를 찍고 경고를 매번 울린다. **조용히 틀리는 것보다
            # 나쁘다** — 시끄럽게 틀려서 참된 외삽을 가린다.
            #
            # 물어야 할 것은 "이 조건이 학습한 **조건들** 중 어디쯤인가" 다.
            # 중복은 그 부공간의 적용범위에 아무 정보를 더하지 않는다.
            Zr = np.round(Zs, 6)                  # 1e-6 sd. 뜻 있는 차이 아래
            _, uidx = np.unique(Zr, axis=0, return_index=True)
            Zu = Zs[np.sort(uidx)]
            n_rows_in = int(Zs.shape[0])
            n_distinct = int(Zu.shape[0])
            # 상한을 6000 으로 둔다. 실제 데이터셋의 서로 다른 조건 수가
            # 3000 을 조금 넘으므로 3000 이면 솎이고, 솎이면 간격이 늘어
            # 참조 분포가 낙관적으로 치우친다. 6000 x 10 x float32 = 240 KB.
            n_keep = min(6000, n_distinct)
            sel = np.random.default_rng(0).choice(
                n_distinct, size=n_keep, replace=False)
            sub = np.ascontiguousarray(Zu[np.sort(sel)], dtype=np.float32)
            # 최근접 거리 분포는 **저장하는 그 표본에서** 잰다. 질의점의
            # 거리도 같은 표본에 대고 재므로, 솎임이 있어도 두 값이 같은
            # 기하에서 나와 비교가 성립한다.
            try:
                from scipy.spatial import cKDTree
                d, _ = cKDTree(sub.astype(np.float64)).query(
                    sub.astype(np.float64), k=2)
                nn = np.sort(d[:, 1])
            except Exception:
                nn = np.array([], dtype=np.float64)
            dom["coverage"] = {
                "axes": [feat[j] for j in con_cols],
                "mu": [float(v) for v in mu],
                "sd": [float(v) for v in sd],
                "sample": sub,
                "nn_sorted": np.asarray(nn, dtype=np.float32),
                "swept_excluded": list(SWEPT),
                "n_rows_in": n_rows_in,
                "n_distinct": n_distinct,
                "note": ("표준화 좌표에서 **서로 다른 조건**의 부분표본과 "
                         "그 조건들끼리의 최근접 거리 분포. 초점·노광은 "
                         "추천기가 격자로 훑으므로 제외했고, 그 둘로만 다른 "
                         "중복 행은 지웠다(지우지 않으면 참조 분포의 95 % "
                         "가 0 이 되어 백분위가 포화한다). 추천기가 질의점의 "
                         "외삽 거리를 백분위로 보고한다. 상자 검사는 "
                         "필요조건일 뿐이고 10 차원에서 매우 약하다 — 각 축 "
                         "90 백분위 점이 학습점의 94.9 % 보다 멀다(모사 "
                         "측정)."),
            }

    # [12차 보완 74 번] 범주형은 **축별 수준만으로 부족하다.** 축별 수준의
    # 데카르트 곱은 실재하지 않는 조합을 만든다(실측 : 2 x 2 x 2 = 8 가지지만
    # 캠페인이 돈 조합은 4 가지). 그래서 **실제로 나온 조합**을 그대로
    # 기록한다.
    #
    # 왜 번들이 들고 가야 하는가 : 이것이 없으면 추천기가 생성 코드의
    # 상수(`campaign.DISCRETE_ARMS`)를 읽어야 한다. 그러면 추천기가
    # '데이터가 어떻게 생성됐는지' 를 알아야 하는 구조가 되고, 데이터가 그
    # 목록과 다른 조합으로 만들어지면 조용히 틀린다. 번들이 자기 조합을
    # 들고 가면 추천기는 생성 코드와 무관해진다.
    cat_cols = [j for j, f in enumerate(feat) if cat_mask[j]]
    if cat_cols:
        names = [feat[j] for j in cat_cols]
        sub = Xt[:, cat_cols]
        ok = np.all(np.isfinite(sub), axis=1)
        combos = sorted({tuple(int(round(v)) for v in row)
                         for row in np.round(sub[ok], 9)})
        dom["categorical_combos"] = {
            "axes": names,
            "combos": [list(c) for c in combos],
        }

    # 상수로 빠진 열은 '추천하지 말 것' 이 아니라 '이 값으로 고정하고
    # 학습했다' 는 뜻이다. 탐색은 이 값을 그대로 써야 한다.
    for nm, v in zip(const_names, const_vals):
        dom["held_constant"][nm] = float(v)
    return dom


def domain_for_json(dom: dict) -> dict:
    """`domain` 을 **JSON 보고서용으로** 바꾼다. 번들용과 다르다.

    왜 둘을 나누는가
    ----------------
    `coverage` 는 넘파이 배열(`sample` 3000 x 연속축, `nn_sorted`)을 들고
    있다. 같은 `domain` 객체가 두 곳으로 간다.

      - .joblib 번들 : joblib 은 넘파이를 그대로 저장한다. **배열이 필요한
        쪽**이다. 추천기가 질의점의 최근접 거리를 재려면 부분표본 자체가
        있어야 한다.
      - JSON 보고서 : `json` 은 ndarray 를 모른다. 그대로 넣으면
        `TypeError: Object of type ndarray is not JSON serializable` 로
        **학습을 다 끝낸 뒤 마지막 줄에서** 죽는다(13 차 보완 전에는
        `coverage` 가 없었으므로 이 경로가 없었다).

    배열을 리스트로 풀어 넣는 것은 답이 아니다. 3000 x 10 을 `indent=2` 로
    쓰면 보고서가 수십만 줄이 되고, 사람이 읽는 파일로서 쓸모가 없어진다.
    보고서에는 **요약**만 넣는다 : 축 이름, 표준화 계수, 표본 크기, 그리고
    최근접 거리 분포의 분위수. 원본 배열은 번들에만 둔다.

    `n_sample` 과 `nn_quantiles` 를 굳이 남기는 이유 : 보고서만 보고도
    '외삽 거리 판정에 쓸 기준이 실제로 들어갔는지' 와 '학습점끼리의 전형적인
    간격이 얼마인지' 를 확인할 수 있어야 한다. 추천기가 보고하는 외삽
    백분위가 이 분포를 기준으로 나오기 때문이다.
    """
    out = dict(dom)
    cov = dom.get("coverage")
    if not isinstance(cov, dict):
        return out
    nn = np.asarray(cov.get("nn_sorted", []), dtype=np.float64)
    nn = nn[np.isfinite(nn)]
    sample = np.asarray(cov.get("sample", []))
    summary = {
        "axes": list(cov.get("axes", [])),
        "mu": [float(v) for v in cov.get("mu", [])],
        "sd": [float(v) for v in cov.get("sd", [])],
        "swept_excluded": list(cov.get("swept_excluded", [])),
        "n_sample": int(sample.shape[0]) if sample.ndim == 2 else 0,
        # 중복을 얼마나 지웠는지 보고서에 남긴다. 이 둘이 거의 같으면
        # 중복 제거가 안 됐다는 뜻이고, 그러면 외삽 백분위를 믿으면 안 된다
        # (13차 보완 90 번).
        "n_rows_in": int(cov.get("n_rows_in", 0)),
        "n_distinct": int(cov.get("n_distinct", 0)),
        "note": cov.get("note", ""),
        "arrays_in_bundle_only": ("sample, nn_sorted 는 크기 때문에 JSON "
                                  "보고서에 넣지 않는다. .joblib 번들에 "
                                  "들어 있고 추천기는 그쪽을 읽는다."),
    }
    if nn.size:
        summary["nn_quantiles"] = {
            f"p{p:02d}": float(np.percentile(nn, p))
            for p in (5, 25, 50, 75, 95, 99)}
        summary["nn_max"] = float(nn.max())
    out["coverage"] = summary
    return out


def _json_default(o):
    """남은 넘파이 값을 JSON 이 받을 수 있는 꼴로 바꾼다.

    `domain_for_json` 이 알려진 경로를 막는다. 이것은 **그 뒤의 그물**이다.
    보고서는 여러 곳에서 모아 붙이므로 새 항목이 들어올 때 넘파이 스칼라가
    섞일 수 있고, 그러면 또 마지막 줄에서 죽는다. 학습을 수십 분 돌린
    뒤에 보고서를 못 쓰는 것이 가장 나쁘다. 모르는 자료형은 문자열로
    떨어뜨리고, 보고서가 남는 쪽을 택한다.
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else {
            "_dropped_array": list(o.shape), "dtype": str(o.dtype)}
    return str(o)


# ----------------------------------------------------------------------------
# 모형
# ----------------------------------------------------------------------------
def build_model(kind: str, cat_mask: np.ndarray, quick: bool):
    """모형 하나를 만든다. 세 종류를 두는 이유는 비교 기준이 필요하기 때문이다.

      linear : 2 차 다항 + 능형 회귀. **기준선**이다. 비선형 모형이 이것을
               크게 못 이기면 모형을 복잡하게 할 이유가 없다.
      hgb    : 히스토그램 경사부스팅. 표 형식 데이터의 표준 선택이고 범주형을
               직접 받는다. 출력이 구간상수이므로 기울기 기반 탐색은 못 한다.
      mlp    : 다층 퍼셉트론. 출력이 매끄러워 12 주차에서 기울기를 쓸 수 있다.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                       StandardScaler)

    cat_idx = np.flatnonzero(cat_mask).tolist()
    num_idx = np.flatnonzero(~cat_mask).tolist()

    if kind == "hgb":
        # 범주형을 직접 알려 준다. 연속값으로 다루면 코드 사이를 보간한다.
        return HistGradientBoostingRegressor(
            max_iter=200 if quick else 600,
            learning_rate=0.08,
            max_leaf_nodes=31 if quick else 63,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=25, random_state=0,
            categorical_features=cat_mask,
        )

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_idx),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), cat_idx),
    ])

    if kind == "linear":
        return Pipeline([
            ("pre", pre),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if kind == "mlp":
        return Pipeline([
            ("pre", pre),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 128) if quick else (256, 256, 128),
                activation="tanh",       # 매끄러운 미분. relu 는 꺾인다.
                solver="adam", alpha=1e-5,
                learning_rate_init=1e-3, batch_size=256,
                max_iter=60 if quick else 400,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=0)),
        ])

    raise SystemExit(f"[오류] 모르는 모형 '{kind}'")


# ----------------------------------------------------------------------------
# 프로파일 예측 : 과제 제목의 '패턴 형성 예측' 에 해당하는 부분
# ----------------------------------------------------------------------------
#: 레지스트 임계값. SampleSpec.threshold 의 기본값과 같아야 한다.
THRESHOLD = 0.30


def profile_cd(prof: np.ndarray, pitch: float, thr: float) -> float:
    """세기 프로파일에서 선폭을 뽑는다.

    저장된 프로파일은 한 pitch 구간을 64 점으로 균일 샘플링한 것이고
    observed = intensity x e_op x dose_rel 이므로, 경계는 이 값이 임계값을
    지나는 곳이다. 교차점은 선형 보간으로 찾는다(격자 간격 pitch/64 는
    pitch 200 nm 에서 3.1 nm 이므로 보간 없이 쓰면 그만큼 양자화된다).

    [검산] p = 0.5 + 0.5 cos(2 pi x / pitch) 는 임계값 통과 폭의 해석해가
    있다. 격자를 32 -> 4096 으로 늘리며 재면 오차가 5.4e-2 -> 6.4e-6 nm 로
    2 차 수렴한다. 저장 격자인 n = 64 에서의 오차는 **0.022 nm** 이며,
    이것이 '예측 프로파일에서 뽑은 CD' 지표의 하한이다. 선이 주기 경계를
    걸치는 경우도 같은 값을 주고, 미해상(전부 밝음/어두움)은 nan 을 준다.
    """
    n = prof.size
    dx = pitch / n
    # 주기 경계를 넘는 선도 다루기 위해 한 주기를 이어 붙인다.
    p = np.concatenate([prof, prof])
    above = p >= thr
    # 상승(미노광->노광)과 하강 교차를 모두 찾는다.
    cross = np.flatnonzero(above[:-1] != above[1:])
    if cross.size < 2:
        return float("nan")

    def xing(i: int) -> float:
        y0, y1 = p[i], p[i + 1]
        t = 0.0 if y1 == y0 else (thr - y0) / (y1 - y0)
        return (i + t) * dx

    xs = [xing(int(i)) for i in cross]
    # 첫 두 교차 사이의 간격 가운데, 임계값 아래 구간(= 남는 선)의 폭.
    widths = []
    for a, b in zip(xs[:-1], xs[1:]):
        mid = 0.5 * (a + b)
        j = min(int(mid / dx), p.size - 1)
        if p[j] < thr:
            widths.append(b - a)
    if not widths:
        return float("nan")
    return float(widths[0])


def run_profiles(d: dict, X: np.ndarray, cond: np.ndarray, ck: list[str],
                 split: dict, kind: str, cat_mask: np.ndarray,
                 quick: bool, n_comp: int) -> dict:
    """프로파일 자체를 예측한다.

    64 점을 따로 64 개 모형으로 맞추지 않는다. 프로파일은 저차원 다양체에
    놓이므로(결상이 저역 통과이므로 당연하다) **주성분 몇 개**로 거의 전부
    설명된다. 주성분 계수만 맞추면 모형 수가 줄고 매끄러움도 유지된다.

    주성분은 **train 만으로** 구한다. 전체로 구하면 검증 프로파일의 정보가
    기저에 섞여 누설이 된다.

    [분할을 다시 거르는 이유] 넘겨받는 split 은 **선택한 스칼라 라벨**의
    유한성으로 이미 걸러진 것이다. 그것을 그대로 쓰면 `--labels` 를 다르게
    줄 때마다 프로파일 학습 집합이 달라져, 프로파일 숫자가 라벨 선택에
    따라 바뀐다. 프로파일 예측은 스칼라 라벨과 무관한 과제이므로 저장된
    원본 분할에서 **프로파일 유한성만으로** 다시 거른다. 그래야 같은
    데이터에 대해 언제 돌려도 같은 값이 나온다.
    """
    from sklearn.decomposition import PCA

    if "profiles" not in d:
        return {"available": False}
    P = np.asarray(d["profiles"], dtype=np.float64)
    n_all = P.shape[0]
    if "split_train" in d:
        raw = {k: np.asarray(d[f"split_{k}"], dtype=np.int64)
               for k in ("train", "val")}
        raw = {k: v[(v >= 0) & (v < n_all)] for k, v in raw.items()}
        tr, va = raw["train"], raw["val"]
    else:
        tr, va = split["train"], split["val"]
    good_tr = np.all(np.isfinite(P[tr]), axis=1)
    good_va = np.all(np.isfinite(P[va]), axis=1)
    tr, va = tr[good_tr], va[good_va]
    if tr.size < 100 or va.size < 20:
        return {"available": False, "reason": "유효 프로파일이 너무 적다"}

    print("\n" + "=" * 78)
    print(f" 프로파일 예측 (패턴 형성) : {kind}")
    print("=" * 78)

    pca = PCA(n_components=min(n_comp, P.shape[1], tr.size))
    Ztr = pca.fit_transform(P[tr])
    evr = np.cumsum(pca.explained_variance_ratio_)
    print(f"\n  주성분 {pca.n_components_} 개로 분산 {100 * evr[-1]:.3f} % 설명")
    print("  (train 만으로 구한 기저다. 전체로 구하면 검증 정보가 기저에")
    print("   섞여 누설이 된다.)")
    for k in (1, 2, 3, 5, 10):
        if k <= len(evr):
            print(f"    상위 {k:>2} 개 : {100 * evr[k - 1]:.3f} %")

    # 각 주성분 계수를 따로 맞춘다.
    models, t1 = [], time.time()
    for c in range(pca.n_components_):
        m = build_model(kind, cat_mask, quick)
        m.fit(X[tr], Ztr[:, c])
        models.append(m)
    fit_s = time.time() - t1

    Zva = np.column_stack([m.predict(X[va]) for m in models])
    Pva_hat = pca.inverse_transform(Zva)
    Pva = P[va]

    err = Pva_hat - Pva
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # 프로파일의 세기 범위로 정규화해 상대 오차도 본다.
    rng = float(np.mean(Pva.max(axis=1) - Pva.min(axis=1)))
    print(f"\n  [프로파일 형상]  (학습 {fit_s:.1f} s, 모형 "
          f"{pca.n_components_} 개)")
    print(f"    세기 RMSE        {rmse:.5g}")
    print(f"    세기 변동폭 평균  {rng:.5g}  -> 상대 {100 * rmse / rng:.3f} %")
    print("    이 지표가 본체다. 64 점 전체를 비교하므로 자명한 성분이 없다.")

    # --- 예측 프로파일에서 CD 와 NILS 를 뽑아 물리 단위로 본다 ------------
    pitch = cond[va][:, ck.index("pitch")]
    cd_t = np.array([profile_cd(Pva[i], pitch[i], THRESHOLD)
                     for i in range(len(va))])
    cd_p = np.array([profile_cd(Pva_hat[i], pitch[i], THRESHOLD)
                     for i in range(len(va))])
    ok = np.isfinite(cd_t) & np.isfinite(cd_p)
    out: dict = {"available": True, "n_components": int(pca.n_components_),
                 "explained_variance": float(evr[-1]),
                 "profile_rmse": rmse, "profile_range": rng,
                 "fit_seconds": fit_s, "n_val": int(va.size)}
    if ok.sum() >= 20:
        de = cd_p[ok] - cd_t[ok]
        out["cd_from_profile"] = {
            "rmse_nm": float(np.sqrt(np.mean(de ** 2))),
            "mae_nm": float(np.mean(np.abs(de))),
            "p95_nm": float(np.quantile(np.abs(de), 0.95)),
            "n": int(ok.sum())}
        print(f"\n  [예측 프로파일에서 뽑은 CD]  ({ok.sum():,} 건)")
        print(f"    RMSE {out['cd_from_profile']['rmse_nm']:.4g} nm, "
              f"95 % {out['cd_from_profile']['p95_nm']:.4g} nm")
        print("    [주의] 이 지표는 형상 지표보다 약하다. 데이터의 참 CD 가")
        print("    dose_to_size 로 목표값(= duty x pitch)에 붙어 있으므로,")
        print("    '교차점만 맞추는' 모형도 이 값을 작게 만들 수 있다.")
        print("    물리를 배웠는지는 위의 형상 RMSE 로 판단한다.")
    else:
        print("\n  [예측 프로파일에서 뽑은 CD] 교차점을 찾은 건이 너무 적다.")
    return out


# ----------------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------------
def metrics(y_true: np.ndarray, y_pred: np.ndarray,
            scale: np.ndarray | None = None) -> dict[str, float]:
    """오차 지표. scale 을 주면 그것으로 나눈 **상대 오차**도 함께 돌려준다.

    [왜 상대 오차가 필요한가]
    cd_error 의 절대 RMSE 를 그대로 읽으면 안 된다. 공정 공차가 상대값
    (`pw_tol` = 목표 CD 의 7.07 %)인데 데이터의 목표 CD 는 30~280 nm 로
    흔들린다. 절대 4.88 nm 는
        목표 280 nm 에서 1.7 %  (공차의 25 %)
        목표  30 nm 에서  16 %  (공차의 2.3 배, 초과)
    로 전혀 다른 의미다. 즉 **절대 nm 로 뽑은 한 숫자가 작은 패턴에서
    얼마나 나쁜지를 숨긴다.** 그래서 목표 CD 로 정규화한 값을 함께 본다.
    """
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    p95 = float(np.quantile(np.abs(err), 0.95))
    var = float(np.var(y_true))
    # R^2 는 분산이 거의 없는 라벨에서 무의미해지므로 분산을 함께 본다.
    r2 = float(1.0 - np.mean(err ** 2) / var) if var > 0 else float("nan")
    out = {"rmse": rmse, "mae": mae, "p95": p95, "r2": r2,
           "std_true": float(np.sqrt(var))}
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        good = np.isfinite(s) & (np.abs(s) > 0)
        if good.any():
            rel = err[good] / s[good]
            out["rel_rmse"] = float(np.sqrt(np.mean(rel ** 2)))
            out["rel_p95"] = float(np.quantile(np.abs(rel), 0.95))
    return out


def fmt(v: float, unit: str) -> str:
    if not np.isfinite(v):
        return "   --   "
    return f"{v:.4g}{unit}"


UNITS = {"cd_mean": " nm", "cd_error": " nm", "ler_3sigma": " nm",
         "cd_3sigma": " nm", "cd_deterministic": " nm", "nils": "",
         "contrast": "", "e_op": "", "dose_clear_mj": " mJ",
         "unresolved_row_frac": ""}


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="대리 모형 학습")
    ap.add_argument("--data", default="dataset/merged.npz",
                    help="병합 데이터 경로. merge_dataset.py 에 준 "
                         "출력 경로와 같아야 한다")
    ap.add_argument("--model", default="all",
                    choices=("all", "linear", "hgb", "mlp"))
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                    help="쉼표로 구분. 기본값은 분산이 있는 5 개")
    ap.add_argument("--quick", action="store_true",
                    help="반복 수를 줄여 빠르게 확인")
    ap.add_argument("--out", default="surrogate_report.json")
    ap.add_argument("--save-model", default="",
                    help="joblib 로 저장할 경로 접두사(빈 값이면 저장 안 함)")
    ap.add_argument("--n-rows", type=int, default=N_ROWS_DEFAULT)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--profiles", action="store_true",
                    help="프로파일(패턴 형상) 예측도 함께 학습한다. "
                         "과제 제목의 '패턴 형성 예측' 에 해당한다")
    ap.add_argument("--n-components", type=int, default=12,
                    help="프로파일 주성분 개수")
    args = ap.parse_args()

    # --save-model 경로를 **학습 전에** 검사한다.
    #
    # 왜 여기인가 : 저장은 학습이 끝난 뒤에 일어난다. 경로가 못 쓰는
    # 경로이면 수십 분을 돌린 뒤에 터지고 결과가 전부 사라진다. 실제로
    # `--save-model models/smoke` 가 `models/` 폴더가 없어서 학습을 다
    # 끝낸 다음 FileNotFoundError 로 죽었다. 1 초 안에 실패해야 하는
    # 종류의 오류를 몇십 분 뒤에 실패하게 두면 안 된다.
    if args.save_model:
        _dir = os.path.dirname(args.save_model)
        try:
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            _probe = f"{args.save_model}_.writetest"
            with open(_probe, "w", encoding="utf-8") as _f:
                _f.write("ok")
            os.remove(_probe)
        except OSError as e:
            print(f"[오류] --save-model 경로에 쓸 수 없습니다 : "
                  f"{args.save_model!r}")
            print(f"       {type(e).__name__}: {e}")
            print("       학습 전에 멈춥니다. 학습을 끝낸 뒤 저장에서 "
                  "실패하면 결과가 사라집니다.")
            print("       접두사는 '<폴더>/<이름>' 형식입니다 "
                  "(예 : models/s -> models/s_mlp_cd_error.joblib).")
            return 1

    t0 = time.time()
    print("=" * 78)
    print(" 대리 모형 학습 (시뮬레이터의 빠른 근사)")
    print("=" * 78)

    d = load_data(args.data)
    ck, lk_raw = resolve_keys(d)
    cond = np.asarray(d["conditions"], dtype=np.float64)
    lab_raw = np.asarray(d["labels"], dtype=np.float64)
    n = cond.shape[0]
    print(f"\n  데이터 {n:,} 건, 조건 {len(ck)} 열, 라벨 {len(lk_raw)} 열")

    lab, lk = add_derived(d, lab_raw, lk_raw)
    made = [k for k in lk if k not in lk_raw]
    if made:
        print(f"  파생 라벨 {len(made)} 개 추가 : "
              + ", ".join(f"{k} ({DERIVED.get(k, '')})" for k in made))

    targets = [t.strip() for t in args.labels.split(",") if t.strip()]
    unknown = [t for t in targets if t not in lk]
    if unknown:
        raise SystemExit(f"[오류] 모르는 라벨 {unknown}. 가능한 값: {lk}")

    # --- 입력 열 ---------------------------------------------------------
    vary = varying_columns(cond, ck)
    const_idx = [i for i in range(len(ck)) if i not in vary]
    const = [ck[i] for i in const_idx]
    feat = [ck[i] for i in vary]
    print(f"  입력으로 쓰는 조건 {len(feat)} / {len(ck)} 열")
    if const:
        print(f"  상수라 뺀 열 {len(const)} 개 : {', '.join(const)}")
    cat_mask = np.array([f in CATEGORICAL for f in feat], dtype=bool)
    if cat_mask.any():
        print(f"  범주형으로 다루는 열 : "
              f"{', '.join(f for f, c in zip(feat, cat_mask) if c)}")

    # --- 유효 행과 분할 ---------------------------------------------------
    keep = finite_mask(lab, lk, targets)
    if keep.sum() == 0:
        print("\n[오류] 선택한 라벨이 전부 NaN 입니다.")
        return 1
    if keep.sum() < n:
        print(f"  라벨이 유한하지 않아 뺀 행 {n - int(keep.sum()):,} 건 "
              f"({100 * (1 - keep.mean()):.2f} %)")

    split = get_split(d, n, keep)
    check_disjoint(split)
    print(f"  분할 : train {len(split['train']):,} / "
          f"val {len(split['val']):,} / test {len(split['test']):,}")
    if min(len(v) for v in split.values()) == 0:
        # [12차 보완 70 번] 진단과 처방을 함께 적는다. 이 오류는 데이터가
        # 적으면 반드시 나는데 한 줄짜리 '비어 있다' 로는 무엇을 해야 하는지
        # 알 수 없다. 67 번과 같은 종류다.
        #
        # 왜 나는가 : 분할은 **광학 조건 그룹** 단위다(누설 방지). 그룹이
        # 적으면 round(n*0.7), round(n*0.15) 로 나눈 뒤 남는 그룹이 0 개가
        # 될 수 있다. 모자란 것은 표본 수가 아니라 **그룹 수**이므로, 같은
        # 조건에서 표본만 더 뽑아도 해소되지 않는다. 그 구분을 적어 주지
        # 않으면 --target 만 키우며 같은 벽에 다시 부딪힌다.
        empty = [k for k in ("train", "val", "test") if len(split[k]) == 0]
        cols = [ck.index(k) for k in GROUP_KEYS if k in ck]
        n_groups = (len({tuple(np.round(r, 12)) for r in cond[:, cols]})
                    if cols else None)
        print(f"\n[오류] 비어 있는 분할이 있습니다 : {', '.join(empty)}")
        print()
        print("  분할은 표본이 아니라 **광학 조건 그룹** 단위로 나눕니다.")
        print("  같은 조건의 표본이 train 과 test 에 함께 들어가면 오차가")
        print("  낙관적으로 나오기 때문입니다. 그래서 그룹 수가 모자라면")
        print("  비율(0.7/0.15/0.15)을 맞출 수 없고 한쪽이 0 개가 됩니다.")
        if n_groups is not None:
            print(f"  이 데이터셋의 광학 조건 그룹 : {n_groups} 개")
            print("  (0.15 * n >= 1 이려면 n >= 7 입니다. 7 개는 세 분할을")
            print("   채우는 최소치일 뿐이고, 품질을 보려면 더 필요합니다.)")
        print()
        print("  할 수 있는 것 (위에서부터 권합니다)")
        print("    1) 광학 조건을 더 뽑습니다. 같은 조건에서 표본만 늘리면")
        print("       그룹 수가 그대로여서 해소되지 않습니다.")
        print("         py generate_dataset.py --target 600")
        print("    2) 이미 만든 데이터셋이 있으면 병합합니다. merge_dataset 이")
        print("       그룹 단위로 다시 나눕니다.")
        # 출력 경로가 **첫 번째 위치 인자**다. --inputs/--out 같은 옵션은
        # 없다. 틀린 명령줄을 처방으로 적으면 처방이 처방 구실을 못 한다.
        print("         py merge_dataset.py 합친것.npz a.npz b.npz")
        print("    3) 품질이 아니라 학습 경로만 확인하려는 것이라면 작은")
        print("       데이터셋 두 개를 합치는 것으로 충분합니다.")
        return 1

    leak = check_group_leakage(cond, ck, split)
    n_val_groups = max(1, leak.get("n_groups", {}).get("val", 1))
    if leak.get("checked"):
        ng = leak["n_groups"]
        print(f"  광학 조건 그룹 : train {ng['train']:,} / "
              f"val {ng['val']:,} / test {ng['test']:,}")
        if leak["leaks"]:
            for pair, cnt in leak["leaks"].items():
                print(f"  [오류] {pair} 에 같은 광학 조건 그룹 {cnt} 개가 "
                      f"함께 있습니다.")
            print("  색인은 달라도 같은 조건을 다시 보는 것이므로 검증 오차가")
            print("  낙관적으로 나옵니다. merge_dataset 으로 다시 분할하십시오.")
            return 1
        st = leak.get("selftest_detects_planted")
        if st is False:
            print("  [오류] 누설 검사의 자기 시험이 실패했습니다. train 의")
            print("  한 행을 val 에 심어도 검사가 잡지 못합니다. '누설 없음'")
            print("  이라는 위 결과를 믿을 수 없으므로 중단합니다.")
            return 1
        print("  그룹 누설 없음 (색인 중복이 아니라 조건 자체를 대조한 결과,"
              " 자기 시험 통과)")

    X = cond[:, vary]
    tr, va = split["train"], split["val"]

    # 학습 조건의 범위. 12 주차 탐색의 상자 제약이 된다. 학습 분할만으로
    # 잡는다 -- 전체로 잡으면 검증·시험 구간의 정보가 제약으로 새어 든다.
    domain = training_domain(
        X, feat, cat_mask, tr, const,
        [float(np.median(cond[:, i])) for i in const_idx])
    print(f"\n  [학습 범위] 연속 축 {len(domain['axes'])} 개, "
          f"범주 축 {len(domain['categorical'])} 개, "
          f"고정 열 {len(domain['held_constant'])} 개를 기록함")
    _cc = domain.get("categorical_combos")
    if _cc:
        print(f"    범주 조합 {len(_cc['combos'])} 개를 기록함 "
              f"({' x '.join(_cc['axes'])}). 축별 수준의 곱은 "
              f"{np.prod([len(domain['categorical'][a]) for a in _cc['axes']])}"
              f" 가지지만 실제로 나온 조합만 쓴다")
    print("    12 주차 탐색은 이 범위 안에서만 해야 한다. mlp 는 tanh 이므로")
    print("    범위 밖에서 포화하고, 포화 구간은 기울기가 0 이어서 평평한")
    print("    최적점처럼 보인다. 즉 탐색이 범위 밖으로 끌려간다.")

    # 목표 CD. CD 계열 라벨의 상대 오차와 공차 비교에 쓴다.
    tgt_cd = None
    if "target_cd" in d:
        _tc = np.asarray(d["target_cd"], dtype=np.float64).reshape(-1)
        if _tc.shape[0] == cond.shape[0]:
            tgt_cd = _tc

    # 유도된 공정 공차. 이 숫자가 '이 오차가 충분한가' 의 기준이다.
    # dataset.SampleSpec 에서 cd_budget 과 배분 비율로 유도된다(임의값이
    # 아니다). claims.py 가 문서와 대조한다.
    pw_tol = float(DS.SampleSpec(pitch=200.0, duty=0.5).pw_tol)
    cd_budget = float(DS.SampleSpec(pitch=200.0, duty=0.5).cd_budget)

    print(f"\n  [공정 공차] 유도값 pw_tol = {100 * pw_tol:.2f} %, "
          f"전체 예산 cd_budget = {100 * cd_budget:.1f} %")
    print("    CD 오차는 목표 CD 에 대한 **상대값**으로 판정해야 한다. 공차가")
    print("    상대값이고 목표 CD 는 데이터 안에서 수 배 흔들리기 때문이다.")
    print("    그래서 CD 계열 라벨에는 상대 RMSE 열을 함께 인쇄한다.")

    # --- cd_mean 의 분산 분해 --------------------------------------------
    vs = variance_split(d, lab, lk, va)
    if vs:
        print("\n  [cd_mean 을 목표로 두면 안 되는 이유]")
        print(f"    목표 CD(= duty x pitch)의 표준편차 : {vs['std_target']:.3f} nm")
        print(f"    목표 대비 편차의 표준편차          : {vs['std_error']:.3f} nm")
        print(f"    비                                 : {vs['ratio']:.1f} 배")
        print("    cd_mean 의 분산은 거의 전부 '목표 CD' 성분이고, 그것은 두")
        print("    입력(duty, pitch)의 곱이다. cd_mean 을 목표로 두면 어떤")
        print("    모형이든 R^2 가 1 에 가깝게 나오지만 그것은 곱셈을 배운")
        print("    것이다. 그래서 cd_error = cd_mean - target_cd 를 쓴다.")
        report_vs = vs
    else:
        report_vs = None

    # --- 잡음 하한 --------------------------------------------------------
    floor = noise_floor(lab, lk, va, args.n_rows, args.n_trials)
    if floor:
        print("\n  [라벨의 몬테카를로 표준오차 추정]")
        for k, v in floor.items():
            print(f"    {k:<14}{fmt(v, UNITS.get(k, ''))}")
        print("    이 값은 행 사이 상관 0 을 가정한 것이므로 실제 하한은 더")
        print("    크다. 합격선이 아니라 '여기 근처면 더 조여도 의미 없다' 는")
        print("    판단에만 쓴다.")

    kinds = ("linear", "hgb", "mlp") if args.model == "all" else (args.model,)
    report: dict = {"data": args.data, "n": int(n), "features": feat,
                    "constant_columns": const, "labels": targets,
                    "noise_floor": floor, "cd_variance_split": report_vs,
                    "group_leakage": leak,
                    # 12 주차 탐색의 상자 제약. training_domain 참조.
                    # 보고서에는 요약판이 들어간다. 번들에는 배열이 든
                    # 원본(`domain`)이 들어간다. domain_for_json 참조.
                    "training_domain": domain_for_json(domain),
                    "models": {}}

    for kind in kinds:
        print("\n" + "=" * 78)
        print(f" 모형 : {kind}")
        print("=" * 78)
        mrep: dict = {}
        for tgt in targets:
            j = lk.index(tgt)
            model = build_model(kind, cat_mask, args.quick)
            t1 = time.time()
            model.fit(X[tr], lab[tr, j])
            fit_s = time.time() - t1

            unit = UNITS.get(tgt, "")
            # CD 계열 라벨은 공차가 목표 CD 에 대한 상대값이므로 절대 nm
            # 만으로는 판단할 수 없다. metrics 의 주석 참조.
            rel_on = tgt_cd is not None and tgt in ("cd_error", "cd_mean",
                                                    "cd_3sigma")
            print(f"\n  [{tgt}]  (학습 {fit_s:.1f} s)")
            _rh = ("    " + f"{'영역':<30}{'건수':>7}{'그룹':>6}{'RMSE':>12}"
                   f"{'MAE':>12}{'95%':>12}{'참값 sd':>12}{'R2':>9}")
            if rel_on:
                _rh += f"{'상대RMSE':>9}{'상대95%':>10}"
            print(_rh)
            print("    " + "-" * (100 + (19 if rel_on else 0)))

            trep: dict = {"fit_seconds": fit_s, "regions": {}}
            thin = []
            # 루프 밖에서도 쓰므로 여기서 정한다. 루프 안에 두면 '루프가
            # 한 번은 돌았다' 는 가정에 기대게 된다.
            thin_cut = max(3, int(0.10 * n_val_groups))
            for name, idx in regions(cond, ck, va):
                pred = model.predict(X[idx])
                m = metrics(lab[idx, j], pred,
                            scale=tgt_cd[idx] if rel_on else None)
                ng = group_ids(cond, ck, idx)
                m["n_groups"] = ng
                m["n"] = int(idx.size)
                trep["regions"][name] = m
                # 그룹이 1~2 개뿐인 영역은 그 축의 성질이 아니라 그 그룹의
                # 성질을 재고 있다. 표시해 두지 않으면 오독한다.
                # 그룹이 적으면 그 영역의 숫자는 축의 성질이 아니라 그
                # 특정 그룹들의 성질일 수 있다. 1~2 개만 보던 기준을 'val
                # 전체 그룹의 10 % 미만' 으로 넓혔다. 실제 데이터에서 z6·z11
                # 영역이 555 개 중 31·34 개(5.6 %·6.1 %)로 나왔는데 이전
                # 기준으로는 경고가 뜨지 않았다. flare·msd 는 혼동 집합이
                # 그룹 안에서 흔들어 286·251 개로 넓다. 즉 같은 '상위 25 %'
                # 라도 축에 따라 그룹 다양성이 10 배 다르다.
                is_thin = ng < thin_cut and "상위" in name
                flagstr = f" <- 그룹 {ng}개뿐" if is_thin else ""
                if is_thin:
                    thin.append(name)
                relcol = ""
                if "rel_rmse" in m:
                    relcol = (f"{100 * m['rel_rmse']:>8.2f}%"
                              f"{100 * m['rel_p95']:>9.2f}%")
                print(f"    {name:<30}{idx.size:>7,}{ng:>6}"
                      f"{fmt(m['rmse'], unit):>12}{fmt(m['mae'], unit):>12}"
                      f"{fmt(m['p95'], unit):>12}"
                      f"{fmt(m['std_true'], unit):>12}{m['r2']:>9.4f}"
                      f"{relcol}{flagstr}")
            if thin:
                print(f"    [주의] 표시된 영역은 광학 조건 그룹이 "
                      f"{thin_cut} 개 미만이다(val 전체 "
                      f"{n_val_groups} 개의 10 %).")
                print("    그 숫자는 해당 축의 영향이 아니라 그 그룹들의 성질을")
                print("    재고 있을 수 있다. z6·z11 은 조건 묶음마다 한 값만")
                print("    갖는 축이라 '상위 25 %' 가 특정 그룹 집합과 거의")
                print("    같아진다. 반면 flare·msd 는 혼동 집합이 그룹 안에서")
                print("    흔들므로 그룹이 넓다. 따라서 이 두 종류의 영역별")
                print("    숫자를 나란히 놓고 비교하면 안 된다.")

            # 전체 오차와 잡음 하한의 비교를 한 줄로 남긴다.
            allm = trep["regions"].get("전체")
            if allm and tgt in floor and floor[tgt] > 0:
                ratio = allm["rmse"] / floor[tgt]
                trep["rmse_over_noise_floor"] = float(ratio)
                if ratio < 1.0:
                    note = ("하한보다 작다. 잡음을 외운 것인지 의심해야 "
                            "한다(분할 누설도 확인할 것).")
                elif ratio < 3.0:
                    note = "하한에 가깝다. 모형을 더 키워도 얻을 것이 적다."
                else:
                    note = ("하한보다 충분히 크다. 모형·표본을 늘리면 "
                            "줄어들 여지가 있다.")
                print(f"    -> RMSE 가 라벨 잡음 추정의 {ratio:.1f} 배.")
                print(f"       {note}")

            # CD 계열이면 상대 오차를 공차와 직접 견준다. 이것이 '이 오차가
            # 충분한가' 에 답하는 유일한 방법이다.
            if allm and "rel_rmse" in allm:
                frac = allm["rel_rmse"] / pw_tol
                frac95 = allm["rel_p95"] / pw_tol
                trep["rel_rmse_over_pw_tol"] = float(frac)
                trep["rel_p95_over_pw_tol"] = float(frac95)
                print(f"    -> 상대 RMSE 가 공차(pw_tol {100 * pw_tol:.2f} %)의 "
                      f"{100 * frac:.0f} %, 상대 95 % 는 {100 * frac95:.0f} %.")

                # [중요] 전체만 보면 안 된다. 전체가 공차 안이어도 특정
                # 영역은 넘을 수 있고, 추천기는 그 영역의 조건도 내놓는다.
                # 실제 데이터에서 전체 95 % 는 공차의 97 % 로 턱걸이였는데
                # z6 영역은 154 %, z11 은 141 % 였다. 전체 수치만 인쇄하면
                # 이 초과가 보고서에서 사라진다.
                over = []
                for nm_, mm in trep["regions"].items():
                    if nm_ == "전체" or "rel_p95" not in mm:
                        continue
                    r = mm["rel_p95"] / pw_tol
                    if r > 1.0:
                        over.append((nm_, r, mm.get("n_groups", 0)))
                trep["regions_over_tolerance"] = {
                    nm_: float(r) for nm_, r, _ in over}

                if frac95 > 1.0:
                    print("       전체 95 백분위가 공차를 넘는다. 추천 20 건 중")
                    print("       1 건은 대리 모형 오차만으로도 공차를 벗어날")
                    print("       수 있다는 뜻이다.")
                elif frac > 0.5:
                    print("       공차의 절반을 넘게 먹는다. 추천기에 남는")
                    print("       여유가 작다.")
                if over:
                    print("    -> [초과] 전체와 별개로, 아래 영역은 95 백분위가")
                    print("       공차를 넘는다. 전체 수치만 보면 묻힌다.")
                    for nm_, r, ng_ in sorted(over, key=lambda z: -z[1]):
                        print(f"         {nm_:<28} 공차의 {100 * r:>4.0f} %"
                              f"  (그룹 {ng_})")
                    print("       13 주차 되돌림 검증은 이 영역을 반드시")
                    print("       포함해야 한다. 쉬운 영역만 넣으면 추천기가")
                    print("       실제보다 좋아 보인다.")

            # 기울어진 두 영역의 차이를 명시한다. 이것이 이 보고의 핵심이다.
            b = trep["regions"].get("기본 영역 (결함 요인 전부 0)")
            g = trep["regions"].get("보강 영역 (하나 이상 0 아님)")
            if b and g and b["rmse"] > 0:
                r = g["rmse"] / b["rmse"]
                trep["aug_over_base_rmse"] = float(r)
                if r > 1.5:
                    print(f"    -> [주의] 보강 영역 오차가 기본 영역의 "
                          f"{r:.1f} 배다. 전체 평균은")
                    print("       표본이 많은 기본 영역을 따라가므로 이 차이를")
                    print("       숨긴다. 보강 영역 표본을 늘릴 근거가 된다.")
            mrep[tgt] = trep

            # 측정한 상대 95 백분위 오차를 번들에 넣는다.
            #
            # 왜 : 추천기의 보수 공차는 `pw_tol - rel_p95` 이고, 그 뜻은
            # '**이 모형의** 오차를 최악으로 떼어낸다' 다. 그런데 추천기는
            # 기본값으로 11 주차에 측정한 0.0689 를 쓴다. 새로 학습한 모형의
            # 오차가 그와 다르면(실제로 소형 데이터셋에서 공차의 122 % 가
            # 나왔다) 보수 공차가 조용히 잘못 잡힌다. 54 번과 같은 복제
            # 문제이므로 번들이 직접 들고 간다.
            #
            # **최악 영역**도 같이 넣는다. 전체 95 백분위가 공차 안이어도
            # z6·z11 영역은 154 %·141 % 였다. 추천기는 그 영역의 조건도
            # 내놓으므로, 보수 공차에는 전체가 아니라 최악 쪽을 쓰는 것이
            # 맞다. 둘을 모두 넣고 추천기가 고르게 한다.
            _rp = trep.get("regions", {}).get("전체", {}).get("rel_p95")
            _worst_nm, _worst = None, None
            for _nm, _mm in trep.get("regions", {}).items():
                if _nm == "전체" or "rel_p95" not in _mm:
                    continue
                if _worst is None or _mm["rel_p95"] > _worst:
                    _worst_nm, _worst = _nm, float(_mm["rel_p95"])
            measured = {"rel_p95_overall": (float(_rp) if _rp is not None
                                            else None),
                        "rel_p95_worst_region": _worst,
                        "worst_region_name": _worst_nm,
                        "n_train_rows": int(domain.get("n_train_rows", 0))}

            if args.save_model:
                import joblib
                path = f"{args.save_model}_{kind}_{tgt}.joblib"
                # domain 을 같이 넣는 이유 : 12 주차가 joblib 하나만 읽고도
                # 상자 제약을 걸 수 있어야 한다. 보고 json 에만 있으면 둘을
                # 따로 들고 다녀야 하고, 그러면 어긋날 자리가 생긴다
                # (11 주차 54 번과 같은 복제 문제다).
                # 폴더를 만든다. --save-model models/s 처럼 하위 폴더를 주는
                # 것이 기본 사용법인데 폴더가 없으면 여기서 FileNotFoundError
                # 가 난다. 학습이 **끝난 뒤**라서 몇십 분을 버리게 된다.
                # (아래 main 시작부의 사전 검사와 짝이다.)
                _dir = os.path.dirname(path)
                if _dir:
                    os.makedirs(_dir, exist_ok=True)
                joblib.dump({"model": model, "features": feat,
                             "categorical": cat_mask.tolist(),
                             "label": tgt, "domain": domain,
                             "pw_tol": pw_tol,
                             "measured": measured}, path)
                print(f"    저장 : {path}")
        if args.profiles:
            mrep["_profiles"] = run_profiles(
                d, X, cond, ck, split, kind, cat_mask, args.quick,
                args.n_components)
        report["models"][kind] = mrep

    # --- 모형 비교 --------------------------------------------------------
    if len(kinds) > 1:
        print("\n" + "=" * 78)
        print(" 모형 비교 (전체 영역 RMSE)")
        print("=" * 78)
        print(f"\n  {'라벨':<16}" + "".join(f"{k:>14}" for k in kinds))
        print("  " + "-" * (16 + 14 * len(kinds)))
        for tgt in targets:
            row = f"  {tgt:<16}"
            for k in kinds:
                v = report["models"][k][tgt]["regions"]["전체"]["rmse"]
                row += f"{fmt(v, UNITS.get(tgt, '')):>14}"
            print(row)
        print("\n  linear 는 기준선이다. 비선형 모형이 이것을 크게 못 이기면")
        print("  모형을 복잡하게 할 이유가 없다. mlp 는 출력이 매끄러워")
        print("  12 주차에서 기울기 기반 탐색에 쓸 수 있고, hgb 는 구간상수라")
        print("  무작위·격자 탐색에만 쓸 수 있다.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=_json_default)
    print(f"\n  보고서 저장 : {args.out}")
    print(f"  총 소요 {time.time() - t0:.1f} s")

    print("\n" + "=" * 78)
    print(" 이 결과를 어떻게 읽어야 하는가")
    print("=" * 78)
    print("""
  이 파일은 **합격·불합격을 내지 않는다.** 종료 코드 0 은 학습과 보고를
  끝냈다는 뜻이고 정확도가 충분하다는 뜻이 아니다. 어느 오차가 '충분한지'
  는 공정 공차가 정하는 것이고, 그 판단은 보고서가 한다.

  읽을 때 세 가지를 본다.

    1. 영역별 오차의 차이. 전체 평균만 보면 안 된다. 병합 데이터가
       77 대 23 으로 기울어 있으므로 전체 평균은 기본 영역의 값에 가깝다.
    2. 잡음 하한과의 비. 하한의 몇 배인지가 '모형을 더 키워 얻을 것이
       있는가' 를 말해 준다.
    3. 기준선(linear)과의 차이. 비선형 모형이 기준선을 크게 못 이기면
       복잡한 모형을 쓸 근거가 없다.

  그리고 이 오차는 **보간 오차가 아니다.** 여기서 재는 것은 '데이터와 같은
  분포에서 뽑은 조건' 에 대한 오차다. 추천기가 내놓는 조건은 데이터에 없는
  점일 수 있고, 그때의 오차는 13 주차에서 추천 조건을 시뮬레이터에 되돌려
  넣어 직접 재야 한다. 그것이 추천기의 실제 오차다.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
