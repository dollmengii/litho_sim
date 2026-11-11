"""recommend.py -- 공정 조건 추천 (12 주차).

무엇을 최적화하는가 : 두 번째 공허한 목적함수를 피한 결과
---------------------------------------------------------
처음에 세운 목적은 "예측 선폭 오차 `|cd_error|` 를 최소화" 였다. **틀렸다.**

`dataset.make_sample` 은 조건마다 **따로** 최적 초점에서 목표 CD 가 나오도록
노광량을 역산한다(`dose_to_size`). 즉 어떤 (NA, 간섭도) 를 고르더라도 공칭점
(초점 0, 노광 1 배) 의 CD 는 이미 목표에 붙어 있다. 공칭점에 남는 오차는
확률 평균과 결정론 값의 차이뿐이고, 그것은 조건을 고르는 기준이 아니다.
그런 목적함수 위에서 탐색하면 거의 평평한 함수를 헤맨다.

11 주차 `cd_mean` 함정과 **같은 족속**이다. 그때는 `target_cd = duty x pitch`
가 입력이라 R^2 가 허수가 되었고, 여기서는 `dose_to_size` 가 CD 를 고정해
목적함수가 평평해진다. 둘 다 "생성 절차가 이미 정해 놓은 것을 최적화하려
했다" 는 같은 오류다.

조건을 가르는 것은 공칭점의 값이 아니라 **공칭점에서 벗어날 때 CD 가 얼마나
빨리 어긋나는가** 다. 그것이 공정 윈도이고 업계가 실제로 최적화하는 대상이다.
그래서

    목적 : 목표 EL 에서의 DOF 최대화 (동률이면 total window 로 가름)
    제약 : NILS >= min_nils, CD 3sigma / CD <= stoch_tol, 상자 제약

로 둔다. 두 공차는 임의값이 아니라 `SampleSpec.cd_budget` 에서 유도된다
(10 주차 예산 분배). 윈도 판정에는 `pw_tol` 을, 확률 균일도 판정에는
`stoch_tol` 을 쓰는 것이 그 분배의 정의와 맞는 사용법이다.

탐색 축은 왜 둘뿐인가
---------------------
대리 모형의 입력 11 열은 성격이 다르다. 섞어서 탐색하면 "플레어를 0 으로
두시오" 같은 답이 나온다.

    손잡이      na, sigma_out          스캐너에서 설정한다 -> **탐색 축**
    설계 요구   pitch, duty            추천 대상이 아니라 입력이다
    공구 상태   flare, msd_nm, blur_nm, z6, z11
                                       측정하는 값이고 고르는 값이 아니다
    교란        defocus                한 점이 아니라 분포다 -> 격자로 쓸어 윈도를 만든다
    윈도 축     dose_rel               탐색 축이 아니다. 윈도의 중심으로 나온다

따라서 실제 탐색 공간은 (na, sigma_out) **2 차원**이다. 2 차원이면 격자 전수
탐색이 가능하므로 기울기 기반 최적화를 쓰지 않는다. 지역 최솟값에 빠질 일이
없는 쪽이 정직하다. 11 주차에 mlp 를 고른 이유는 그대로 유효하지만, 근거의
무게가 '기울기' 에서 '정확도와 매끄러운 보간' 으로 옮겨간다.

추천기가 **인증기가 아닌** 이유
-------------------------------
대리 모형의 상대 오차는 RMSE 3.47 %, 95 백분위 6.89 % 다(11 주차 실측).
윈도 공차 `pw_tol` 은 7.07 % 다. 즉 **모형 오차가 공차를 거의 다 먹는다.**

그래서 윈도를 두 번 센다.

    dof_nominal  : 공차 = pw_tol            모형 오차를 무시한 낙관값
    dof_guarded  : 공차 = pw_tol - rel_p95  모형 오차를 최악으로 떼어낸 보수값

참값은 둘 사이에 있다. `pw_tol - rel_p95` 가 0 이하로 내려가면 보수 윈도는
비고, 그것은 **대리 모형만으로 공정 윈도를 인증할 수 없다**는 뜻이다. 그때
추천기의 역할은 인증이 아니라 **순위 매기기**로 한정된다. 후보들 사이의
비교에서는 모형 오차가 부분적으로 상쇄되기 때문이다. 상위 후보를 뽑아
시뮬레이터로 되돌려 재는 것이 인증이고, 그것이 13 주차다.

이 구분을 코드가 직접 인쇄한다. 숫자 하나만 보고 "DOF 180 nm 확보" 라고
읽는 것을 막는 것이 목적이다.

사용법
------
    (아래 두 줄 각각을 한 줄로 붙여 쓸 것. 줄 continuation 은 셸마다 다르다)

    py train_surrogate.py --data dataset/merged.npz --model mlp --labels cd_error,nils,cd_3sigma --save-model models/s
    py recommend.py --models models/s_mlp --pitch 200 --duty 0.5

종료 코드
---------
0 : 평가를 끝냈다. **추천이 공차를 만족한다는 뜻이 아니다.**
1 : 평가를 할 수 없었다(모형 없음, 상자 밖, 라벨 부족 등).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from litho import dataset as DS
from litho import process as PR

#: 추천에 반드시 필요한 라벨. 없으면 그 제약을 걸 수 없으므로 거부한다.
#:   cd_error   : 윈도를 만드는 데 쓴다 (CD = target_cd + cd_error)
#:   nils       : 상 품질 제약
#:   cd_3sigma  : 확률 균일도 제약 (stoch_tol 과 비교)
LABELS_REQUIRED = ("cd_error", "nils", "cd_3sigma")

#: 윈도를 만들 때 쓸 격자. 초점은 학습 범위 안으로 자동 축소된다.
N_FOCUS_DEFAULT = 33
N_DOSE_DEFAULT = 33

#: 업계 관례상 공정 윈도는 EL 10 % 에서의 DOF 로 표기한다.
EL_TARGET_DEFAULT = 10.0

#: 대리 모형의 상대 95 백분위 오차. 보수 공차 계산에 쓴다(11 주차 실측값).
#: 모듈 상수로 둔 이유 : argparse 안에 직접 적으면 `claims.py` 가 문서값과
#: 대조할 수 없다. 문서는 이 값에서 유도한 '보수 공차 0.181 %' 를 인용하므로,
#: 한쪽만 바뀌면 문서가 낡는다(11 주차 37 번과 같은 자리).
REL_P95_DEFAULT = 0.0689


# ----------------------------------------------------------------------------
# 모형 번들
# ----------------------------------------------------------------------------
def load_bundles(prefix: str, labels=LABELS_REQUIRED) -> dict:
    """`train_surrogate.py --save-model` 이 저장한 번들을 읽는다.

    번들마다 `features` 와 `domain` 을 들고 있다. 라벨이 달라도 같은 데이터로
    같은 분할에서 학습했으므로 **같아야 한다.** 다르면 서로 다른 학습 실행에서
    나온 파일을 섞은 것이므로 거부한다. 11 주차 54 번(복제 상수가 어긋나면
    누설을 못 잡는다) 과 같은 종류의 위험이다.
    """
    try:
        import joblib
    except ImportError:
        raise SystemExit("[오류] joblib 이 없습니다. py -m pip install joblib")

    # [12차 보완 72 번] 학습은 `<접두사>_<모형>_<라벨>.joblib` 로 저장한다.
    # `--model mlp` 처럼 하나만 지정해도 모형 이름 구간이 들어간다. 추천기는
    # `<접두사>_<라벨>.joblib` 만 찾고 있었으므로, **추천기가 인쇄한 처방을
    # 그대로 따라도 추천기가 돌지 않았다.** 실제로 부딪혔다.
    #   py train_surrogate.py ... --save-model models/sm   -> models/sm_mlp_nils
    #   py recommend.py --models models/sm                 -> models/sm_nils 를 찾음
    # 처방이 `prefix.rsplit('_', 1)[0]` 로 모형 구간을 떼어 내고 있어서
    # 순환까지 했다(떼어 낸 이름으로 저장하면 다시 모형 구간이 붙는다).
    # K6(4) 는 '라벨을 만들 수 있는가' 만 보았고 **파일 이름 규약**은 보지
    # 않았다. 그래서 그 검사를 통과한 채로 이 벽이 남아 있었다.
    #
    # 고치는 방향은 추천기가 규약을 풀어 주는 쪽이다. 학습 쪽 이름을 바꾸면
    # 여러 모형을 한 번에 저장하는 경우가 깨진다.
    #
    # mlp 를 먼저 쓰는 이유 : 출력이 매끄러워 12 주차 탐색에 맞다. hgb 는
    # 구간상수여서 같은 격자에서 계단이 생긴다(학습 보고서의 설명과 같다).
    MODEL_PREF = ("mlp", "hgb", "linear")
    out, missing, used = {}, [], {}
    for lab in labels:
        exact = f"{prefix}_{lab}.joblib"
        if os.path.exists(exact):
            out[lab] = joblib.load(exact)
            used[lab] = ""                      # 모형 구간 없음
            continue
        hit = None
        for m in MODEL_PREF:
            cand = f"{prefix}_{m}_{lab}.joblib"
            if os.path.exists(cand):
                hit = (m, cand)
                break
        if hit is None:
            missing.append(exact)
            continue
        out[lab] = joblib.load(hit[1])
        used[lab] = hit[0]

    if missing:
        near = sorted(glob.glob(f"{prefix}*_*.joblib"))
        msg = ["[오류] 필요한 모형 파일이 없습니다.", ""]
        msg += [f"  없음 : {p}" for p in missing]
        msg += ["", f"  (`{prefix}_<모형>_<라벨>.joblib` 도 함께 찾았습니다. "
                    f"모형은 {', '.join(MODEL_PREF)} 순으로 봅니다.)"]
        if near:
            msg += ["", "  같은 접두사로 찾은 파일 :"]
            msg += [f"    {p}" for p in near]
        # 한 줄로 적는다. 줄 continuation 은 셸마다 다르다(cmd 는 ^,
        # PowerShell 은 백틱, bash 는 \). 어디서든 붙여 쓸 수 있는 쪽이 낫다.
        #
        # 저장 접두사를 **그대로** 적는다. 전에는 `_mlp` 를 떼어 냈는데,
        # 떼어 낸 이름으로 저장하면 다시 붙으므로 처방이 순환했다.
        msg += ["", "  아래로 먼저 학습하십시오. (한 줄로 붙여 쓸 것)", "",
                f"    py train_surrogate.py --data dataset/merged.npz "
                f"--model mlp --labels {','.join(labels)} "
                f"--save-model {prefix}",
                "",
                f"  그 뒤 같은 접두사로 추천기를 돌리면 됩니다 : "
                f"--models {prefix}"]
        raise SystemExit("\n".join(msg))

    # 라벨마다 **다른 모형**이 섞이면 서로 다른 근사를 한 예측을 한 표에
    # 올리는 셈이다. 위에서 '서로 다른 학습 실행을 섞지 않는다' 고 거부하는
    # 것과 같은 종류의 위험이므로 여기서도 거부한다.
    kinds = sorted(set(used.values()))
    if len(kinds) > 1:
        raise SystemExit("\n".join([
            "[오류] 라벨마다 다른 모형 파일이 잡혔습니다 : "
            + ", ".join(f"{lab} -> {m or '(모형 구간 없음)'}"
                        for lab, m in sorted(used.items())),
            "",
            "  모형이 섞이면 서로 다른 근사를 한 예측을 한 표에 올리게 됩니다.",
            "  같은 학습 실행에서 나온 한 벌을 쓰십시오. (한 줄로 붙여 쓸 것)",
            "",
            f"    py train_surrogate.py --data dataset/merged.npz --model mlp "
            f"--labels {','.join(labels)} --save-model {prefix}",
        ]))
    _model_kind = kinds[0] if kinds else ""

    # 옛 번들 거부. `domain` 은 12 주차에 추가했으므로 11 주차에 저장한
    # joblib 에는 없다. 그대로 두면 KeyError 역추적만 뜨는데, 그 파일을 쓰면
    # **상자 제약 없이** 포화 영역까지 탐색하게 되므로 조용히 넘어가서는
    # 안 되는 차이다(61 번). 무엇을 어떻게 다시 하라고 적는다.
    stale = [lab for lab, b in out.items()
             if not isinstance(b, dict) or "domain" not in b
             or "features" not in b
             or not isinstance(b.get("domain"), dict)
             or "axes" not in b["domain"]]
    if stale:
        raise SystemExit("\n".join([
            "[오류] 학습 범위(domain)가 없는 옛 모형 파일입니다 : "
            + ", ".join(sorted(stale)),
            "",
            "  학습 범위는 12 주차에 추가했습니다. 11 주차에 저장한 번들에는",
            "  없습니다. 그 파일로는 상자 제약을 걸 수 없고, 제약이 없으면",
            "  탐색이 모형이 포화한 영역으로 끌려갑니다. 거기서는 예측이",
            "  '완벽' 처럼 나오지만 데이터가 뒷받침하지 않습니다",
            "  (VERIFICATION_LIMITS 61 번).",
            "",
            "  --save-model 로 다시 저장하십시오. (한 줄로 붙여 쓸 것)",
            "",
            f"    py train_surrogate.py --data dataset/merged.npz --model mlp "
            f"--labels {','.join(labels)} "
            f"--save-model {prefix.rsplit('_', 1)[0]}",
        ]))

    # 번들 사이 정합성. 여기서 걸러야 조용히 틀린 추천을 막는다.
    ref_lab = labels[0]
    ref = out[ref_lab]
    for lab, b in out.items():
        if list(b["features"]) != list(ref["features"]):
            raise SystemExit(
                f"[오류] '{lab}' 과 '{ref_lab}' 의 입력 열이 다릅니다. 서로 "
                "다른 학습 실행의 파일을 섞은 것으로 보입니다. 한 번에 다시 "
                "학습하십시오.")
        if b["domain"]["axes"].keys() != ref["domain"]["axes"].keys():
            raise SystemExit(
                f"[오류] '{lab}' 과 '{ref_lab}' 의 학습 범위 축이 다릅니다.")
    # 어느 모형을 읽었는지 적어 둔다. hgb 는 구간상수여서 같은 격자에서
    # mlp 와 다른 계단이 나오므로, 결과를 읽는 사람이 알아야 하는 정보다.
    for b in out.values():
        b["model_kind"] = _model_kind
    return out


def resolve_rel_p95(bundles: dict,
                    override: float | None) -> tuple[float, str, dict | None]:
    """보수 공차에 쓸 상대 95 백분위 오차와 그 **출처**를 정한다.

    우선순위
    --------
    1) `--rel-p95` 로 직접 준 값. 측정값을 손에 들고 있는 경우다.
    2) 번들에 저장된 실측치. **최악 영역**을 쓴다. 전체 95 백분위가 공차 안
       이어도 특정 영역은 넘을 수 있고(실제 데이터에서 z6 154 %, z11 141 %),
       추천기는 그 영역의 조건도 내놓는다. 전체를 쓰면 보수 공차가 낙관적이
       된다.
    3) 둘 다 없으면 11 주차 실측 상수. 12 주차 이전에 저장한 번들이 그렇다.

    왜 번들에서 읽는가
    ------------------
    보수 공차의 뜻은 '**이 모형의** 오차를 최악으로 떼어낸다' 다. 다른 모형의
    숫자를 쓰면 그 뜻이 사라진다. 실제로 소형 데이터셋으로 학습한 모형은
    공차의 122 % 가 나왔는데 기본 상수는 97 % 였다. 숫자가 두 곳에 따로
    적혀 어긋나는 11 주차 54 번과 같은 자리다.

    돌려주는 것
    -----------
    (값, 출처 문구, 번들의 measured 사전 또는 None). 출처를 함께 돌려주는
    이유는 **인쇄해야** 하기 때문이다. 어느 숫자를 왜 썼는지 보이지 않으면
    다른 모형의 값을 쓰고 있어도 알 수 없다.
    """
    cd = bundles.get("cd_error") or next(iter(bundles.values()))
    ms = cd.get("measured") if isinstance(cd, dict) else None
    if not isinstance(ms, dict):
        ms = None

    if override is not None:
        return float(override), "--rel-p95 로 직접 지정", ms
    if ms is not None and ms.get("rel_p95_worst_region") is not None:
        return (float(ms["rel_p95_worst_region"]),
                f"번들 실측 (최악 영역 : "
                f"{ms.get('worst_region_name') or '이름 없음'})", ms)
    if ms is not None and ms.get("rel_p95_overall") is not None:
        return (float(ms["rel_p95_overall"]),
                "번들 실측 (전체. 영역별 값이 없다)", ms)
    return (float(REL_P95_DEFAULT),
            f"11 주차 실측 상수 {REL_P95_DEFAULT} (번들에 실측치가 없다. "
            f"--save-model 로 다시 저장하면 이 모형의 값을 쓴다)", ms)


# ----------------------------------------------------------------------------
# 상자 제약
# ----------------------------------------------------------------------------
def domain_violations(X: np.ndarray, feat: list[str], domain: dict,
                      conservative: bool = False) -> list[dict]:
    """학습 범위를 벗어난 축과 그 정도.

    왜 경고가 아니라 거부인가
    -------------------------
    mlp 는 `tanh` 이므로 범위 밖에서 포화한다. 포화 구간은 데이터의 제약을
    받지 않고 기울기가 0 이어서 **평평한 최적점처럼 보인다.** 즉 탐색은 범위
    밖을 우연히 밟는 것이 아니라 적극적으로 끌려간다. 11 주차 모형으로 실측한
    결과, 상자 경계의 11 배 밖에서 `cd_error` 예측이 0.085 nm(사실상 완벽)
    로 나왔다. 경고로 두면 그 값이 추천으로 올라간다.

    conservative=True 면 min/max 대신 1~99 백분위 상자를 쓴다. min/max 는
    이상치 한 점이 정하므로 그 상자의 구석에는 데이터가 없을 수 있다.
    """
    lo_key, hi_key = ("p01", "p99") if conservative else ("min", "max")
    bad = []
    for j, f in enumerate(feat):
        ax = domain["axes"].get(f)
        if ax is None:                      # 범주형이거나 기록이 없는 축
            continue
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        lo, hi = ax[lo_key], ax[hi_key]
        span = max(hi - lo, 1e-12)
        under, over = lo - col.min(), col.max() - hi
        if under > 1e-9 or over > 1e-9:
            bad.append({"axis": f, "lo": lo, "hi": hi,
                        "got_lo": float(col.min()), "got_hi": float(col.max()),
                        "over_frac": float(max(under, over) / span)})
    return bad


#: 부호 거울로 다룰 수차 축. 아래 SIGN_MIRROR_NOTE 참조.
MIRROR_AXES = ("z6", "z11")

SIGN_MIRROR_NOTE = """부호 거울 : 왜 음수 수차를 학습 범위 없이 평가할 수 있는가

시뮬레이터로 직접 측정한 결과 다음이 **정확히**(차 0.0000 nm) 성립한다.

    CD( +dz, +z6, +z11 ) = CD( -dz, -z6, -z11 )
    CD( +dz, +z6, -z11 ) = CD( -dz, -z6, +z11 )

부호가 무의미한 것이 아니다. 초점을 고정하고 수차 부호만 뒤집으면 CD 가
초점 -150~150 nm 구간에서 A 류 곡선과 B 류 곡선의 차가 최대 **8.24 nm** 로,
공차(100 nm 목표에서 7.07 nm)의 1.17 배다. 즉 부호는 물리적으로 구별되지만
**초점 반전과 축퇴**다. Z6 과 Z11 이 동공 반경의 짝함수 성분을 품어 defocus
와 같은 꼴로 더해지기 때문이다(10 주차 C5 의 전수 분류와 같은 근거).

따라서 (z6, z11) 의 부호 사분면 넷은 초점 거울에 대해 **두 등가류**로 묶인다.

    A 류 : (+,+) 와 (-,-)
    B 류 : (+,-) 와 (-,+)

학습 데이터는 z6, z11 을 모두 (0, 0.05) 에서 뽑으므로 **A 류만** 덮는다.
그래서

  - 둘 다 음수인 공구 : 부호를 뒤집고 초점 축을 반전시켜 평가할 수 있다.
    공정 윈도는 초점에 대해 거울이므로 DOF 와 EL 은 **그대로**이고, 최적
    초점 오프셋의 부호만 뒤집힌다.
  - 부호가 섞인 공구 : 이 변환으로 닿을 수 없다. B 류가 데이터에 없기
    때문이다. 두 류의 차는 공차를 넘으므로 근사로 넘길 수 없다.
    **거부해야 한다.**

    한 초점에서 B 류 대표 하나만 보면 차이를 잘못 잰다. 두 대표는 서로
    거울이라 같은 초점에서 값이 다르므로(한 점에서 6.09 nm 와 21.16 nm 처럼
    갈린다), 초점 구간 전체에서 두 대표 모두와 대조해 최댓값을 써야 한다.
    validate_w12 Case 7 이 그렇게 잰다.

다음 생성에서는 두 축 중 **하나만** 음수로 열면 된다(예 z11 을
(-0.05, 0.05) 로). 그러면 (+,+) 와 (+,-) 가 모두 뽑히고 나머지 둘은 정확한
거울로 따라오므로, **표본 수를 늘리지 않고** 두 등가류를 다 덮는다."""


def mirror_request(base: dict, domain: dict) -> tuple[dict, bool, str]:
    """필요하면 부호 거울을 적용한다. (새 base, 적용 여부, 사유).

    근거는 SIGN_MIRROR_NOTE 에 있다. 측정으로 확인한 정확한 축퇴만 쓰고,
    닿을 수 없는 사분면은 호출부가 거부하도록 사유를 돌려준다.
    """
    vals = {a: float(base[a]) for a in MIRROR_AXES if a in base}
    if not vals:
        return dict(base), False, ""
    signs = [np.sign(v) for v in vals.values() if v != 0.0]
    if not signs:
        return dict(base), False, ""

    # 학습 범위가 양수 쪽만인 축
    pos_only = [a for a in vals
                if a in domain["axes"] and domain["axes"][a]["min"] >= -1e-12]
    if not pos_only:
        return dict(base), False, ""        # 데이터가 이미 양쪽을 덮는다

    bad = [a for a in pos_only if vals.get(a, 0.0) < 0.0]
    if not bad:
        return dict(base), False, ""        # 거울이 필요 없다

    if len(set(signs)) > 1:
        return (dict(base), False,
                "부호가 섞인 수차(한쪽 양수, 한쪽 음수)는 현 데이터로 평가할 "
                "수 없습니다. 초점 거울로 닿지 않는 등가류이고, 그 차이는 "
                "공차를 넘습니다. SIGN_MIRROR_NOTE 참조.")

    out = dict(base)
    for a in vals:
        out[a] = abs(vals[a])
    return out, True, ""


def check_request(base: dict, feat: list[str], domain: dict,
                  conservative: bool = False) -> None:
    """요청의 **고정 부분**(공구 상태, 설계 요구)이 학습 범위 안인지 먼저 본다.

    후보마다 검사하면 "49 개 건너뜀" 만 남고 어느 축이 문제인지 알 수 없다.
    그런데 공구 상태와 피치·duty 는 모든 후보에서 같으므로, 한 번만 보면
    되고 그래야 사유를 정확히 말해 줄 수 있다. (na·sigma_out 은 상자에서
    뽑으므로 위반할 수 없고, 초점·노광 격자는 상자로 자른다.)

    흔한 경우가 '완벽한 장비' 다. 플레어 0, MSD 0 은 라틴 하이퍼큐브가 열린
    구간에서 뽑으므로 학습 데이터에 없다. 모형이 본 적 없는 조건이므로
    거부가 맞는 동작이다. 측정값을 넣어야 한다.
    """
    lo_key, hi_key = ("p01", "p99") if conservative else ("min", "max")
    bad = []
    for f in feat:
        if f in ("defocus", "dose_rel", "na", "sigma_out"):
            continue                        # 격자·탐색 축은 따로 처리한다
        ax = domain["axes"].get(f)
        if ax is None or f not in base:
            continue
        v = float(base[f])
        lo, hi = ax[lo_key], ax[hi_key]
        if v < lo - 1e-9 or v > hi + 1e-9:
            bad.append((f, v, lo, hi))
    if not bad:
        return

    msg = ["[오류] 요청한 조건이 학습 범위 밖입니다. 그 구간의 예측은 믿을 수",
           "없으므로 거부합니다(모형이 포화해 '완벽' 처럼 보이기 때문입니다).",
           "",
           f"  {'축':<10}{'준 값':>12}{'학습 최소':>12}{'학습 최대':>12}"]
    for f, v, lo, hi in bad:
        msg.append(f"  {f:<10}{v:>12.4f}{lo:>12.4f}{hi:>12.4f}")
    msg += ["",
            "  공구 상태(flare, msd_nm, blur_nm, z6, z11)는 고르는 값이 아니라",
            "  측정해서 넣는 값입니다. 0 을 기본값으로 쓰면 '완벽한 장비' 가",
            "  되는데, 라틴 하이퍼큐브가 열린 구간에서 뽑으므로 학습 데이터에",
            "  그런 조건이 없습니다.",
            "",
            "  학습 범위 안의 값으로 다시 주십시오. 범위 전체는"]
    for f in feat:
        ax = domain["axes"].get(f)
        if ax:
            msg.append(f"    {f:<10} [{ax['min']:.4f}, {ax['max']:.4f}]"
                       f"   (1~99 백분위 [{ax['p01']:.4f}, {ax['p99']:.4f}])")
    raise SystemExit("\n".join(msg))


CAT_AXES = ("pol_code", "source_code", "mask_code")

#: 범주 축의 코드를 사람이 읽는 이름으로 되돌리는 표. `litho.dataset` 의
#: 인코딩과 **같은 한 벌**을 쓴다. 여기에 따로 적으면 어긋난다(11 주차 54 번).
CAT_NAMES = {"pol_code": DS.POLARIZATION_CODES,
             "source_code": DS.SOURCE_CODES,
             "mask_code": DS.MASK_CODES}


#: 추천기가 각 조건 열을 **어떻게 공급하는가**. 12 차 보완 74 번에서 이
#: 선언이 없어서 구멍이 났다. 학습 쪽이 어떤 열을 입력으로 쓰기 시작하면
#: 추천기는 그 값을 줄 수 없어 죽는데, 그것을 미리 알 방법이 없었다.
#:
#: `verify.py` K6(4) 는 '추천기가 요구하는 **라벨**을 학습이 만들 수 있는가'
#: 를 본다. 그 **반대 방향**, 즉 '학습이 요구하는 **입력**을 추천기가 줄 수
#: 있는가' 는 아무도 보지 않았다. 이 표가 그 방향을 선언한다.
SUPPLY = {
    # 초점·노광은 격자로 쓸어 윈도를 만든다
    "grid": ("defocus", "dose_rel"),
    # 손잡이. 격자 전수 탐색한다
    "search": ("na", "sigma_out"),
    # 요청으로 받는다 (설계 요구 + 측정한 공구 상태)
    "request": ("pitch", "duty", "flare", "msd_nm", "blur_nm", "z6", "z11"),
    # 이산 조합. 고정하거나 탐색 축으로 쓴다
    "categorical": ("pol_code", "source_code", "mask_code"),
}

#: 추천기가 값을 줄 수 없는 열과 그 사유. 학습 데이터에서 이 열이 변하기
#: 시작하면 추천기는 쓸 수 없으므로, 조용히 죽는 대신 사유를 적어 거부한다.
UNSUPPLIED = {
    "wavelength": "노광 장비가 정하는 상수다. 추천 대상이 아니다",
    "n_medium": "침지 매질이 정하는 상수다",
    "sigma_in": "환형 조명의 내부 시그마. 현재 조명 형상에서 쓰지 않는다",
    "pole_offset": "다극 조명의 극 위치. 조명 형상에 딸린 값이다",
    "pole_radius": "다극 조명의 극 반지름. 조명 형상에 딸린 값이다",
    "att_t": "감쇠 위상 마스크의 투과율. 마스크 종류에 딸린 값이다",
    "dose_mj": "물리 노광량. 상수 임계값 모형에서 threshold 와 축퇴다",
    "z5": "비점수차. 현재 캠페인이 흔들지 않는다",
    "z7": "코마. 현재 캠페인이 흔들지 않는다",
    "z8": "코마. 현재 캠페인이 흔들지 않는다",
}


def unsupplied_inputs(feat: list[str]) -> list[str]:
    """모형이 입력으로 쓰는데 추천기가 공급할 수 없는 열.

    왜 이 함수가 필요한가 (12 차 보완 74 번)
    ----------------------------------------
    실제로 부딪혔다. 데이터를 이산 조합으로 생성하자 편광·조명·마스크가
    상수가 아니게 되어 모형 입력이 14 열로 늘었는데, 추천기에는 그 값을 줄
    옵션이 없었다. 그래서 `[오류] 조건 'pol_code' 의 값이 주어지지
    않았습니다` 로 죽었고, 그 메시지만으로는 무엇을 해야 하는지 알 수 없었다.

    개발용 소형 데이터는 `--arms` 없이 뽑아 그 세 열이 상수로 빠졌으므로
    (입력 11 열) 끝까지 돌았다. 즉 **개발 데이터와 실제 데이터의 열 구성이
    달라서** 구멍이 드러나지 않았다.

    이 함수는 범주형 하나를 고치는 것이 아니라 **방향 전체**를 막는다. 앞으로
    어느 축이 입력이 되더라도 여기서 걸린다.
    """
    known = set()
    for v in SUPPLY.values():
        known.update(v)
    return [f for f in feat if f not in known]


def arm_label(arm: dict) -> str:
    """이산 조합을 사람이 읽는 한 줄로."""
    parts = []
    for f in CAT_AXES:
        if f in arm:
            names = CAT_NAMES[f]
            i = int(round(float(arm[f])))
            parts.append(names[i] if 0 <= i < len(names) else f"{f}={i}")
    return " / ".join(parts) if parts else "(이산 조합 없음)"


def observed_combos(need: list[str],
                    domain: dict) -> tuple[list[dict], str]:
    """데이터에 실제로 나온 범주 조합. (조합 목록, 출처) 를 돌려준다.

    출처는 "bundle" 또는 "fallback" 이다. 어느 쪽인지 호출부가 알아야 한다.
    되돌림 경로는 생성 코드의 상수를 읽는 것이므로 **확정이 아니라 추정**이고,
    그 차이를 사용자에게 보고해야 한다(12 차 보완 74 번).

    조합을 축별 수준으로 한 번 더 걸러 내는 이유 : 번들의 기록은 그 번들의
    학습 행에서 나온 것이므로 이미 일관되지만, 되돌림 목록은 외부 상수라
    데이터와 어긋날 수 있다. 같은 코드로 두 경우를 처리한다.
    """
    levels = {f: set(int(round(v))
                     for v in (domain.get("categorical", {}).get(f) or []))
              for f in need}

    cc = domain.get("categorical_combos")
    if isinstance(cc, dict) and cc.get("axes") and cc.get("combos"):
        axes = list(cc["axes"])
        out = []
        for row in cc["combos"]:
            m = {a: int(round(float(v))) for a, v in zip(axes, row)}
            if all(f in m for f in need):
                out.append({f: m[f] for f in need})
        # 같은 조합이 여러 번 나올 수 있다(필요한 축만 남겼으므로). 중복 제거.
        uniq, seen = [], set()
        for a in out:
            k = tuple(a[f] for f in need)
            if k not in seen:
                seen.add(k)
                uniq.append(a)
        if uniq:
            return uniq, "bundle"

    # 옛 번들. 생성 코드의 상수로 되돌아간다.
    try:
        from litho import campaign as CP
    except Exception:                       # noqa: BLE001
        return [], "fallback"
    out = []
    for a in CP.DISCRETE_ARMS:
        sp = DS.SampleSpec(polarization=a["polarization"],
                           mask_type=a["mask_type"],
                           source_shape=a["source_shape"])
        codes = {"pol_code": sp.pol_code, "source_code": sp.source_code,
                 "mask_code": sp.mask_code}
        arm = {f: codes[f] for f in need}
        if any(levels[f] and arm[f] not in levels[f] for f in need):
            continue
        if arm not in out:
            out.append(arm)
    return out, "fallback"


def resolve_arms(feat: list[str], domain: dict,
                 args) -> tuple[list[dict], list[str]]:
    """모형이 요구하는 범주 축의 값을 정한다.

    왜 필요한가 (12 차 보완 74 번)
    ------------------------------
    데이터를 이산 조합(arm)으로 나누어 생성하면 편광·조명·마스크가 상수가
    아니게 되어 모형의 **입력**이 된다. 그런데 추천기에는 그 값을 줄 방법이
    없었다. 그래서 실제 데이터셋(입력 14 열, 범주 3 개)에서는 추천기가
    `[오류] 조건 'pol_code' 의 값이 주어지지 않았습니다` 로 죽었다. 개발 중
    쓴 소형 데이터는 `--arms` 없이 뽑아 그 세 열이 상수로 빠졌으므로(입력
    11 열) 끝까지 돌았고, 그래서 이 구멍이 드러나지 않았다.

    `verify.py` K6(4) 는 '추천기가 요구하는 **라벨**을 학습이 만들 수 있는가'
    만 보았다. '학습이 요구하는 **입력**을 추천기가 줄 수 있는가' 는 아무도
    보지 않았다. 방향이 반대인 같은 종류의 검사가 비어 있었다.

    무엇을 돌려주는가
    -----------------
    (조합 목록, 주의 문구 목록). 조합 하나는 {축: 코드} 사전이다. 고정하면
    길이 1, `--search-arms` 면 데이터에 있는 조합 전부다.

    왜 조합을 데카르트 곱으로 만들지 않는가
    ---------------------------------------
    축별 관측 수준의 곱은 실재하지 않는 조합을 만든다. 실측 예 : 편광 2 종 x
    조명 2 종 x 마스크 2 종 = 8 가지지만 캠페인이 실제로 돈 조합은 4 가지다.
    없는 조합은 범주형 외삽이므로 연속 축의 상자 제약과 같은 이유로 쓸 수
    없다.

    유효 조합을 어디서 얻는가
    -------------------------
    **번들의 `domain["categorical_combos"]`** 를 쓴다. 학습이 데이터에서 실제로
    나온 조합을 기록해 둔 것이다. 그러면 추천기는 데이터가 어떻게 생성됐는지
    알 필요가 없다.

    그 키가 없는 **옛 번들**은 `campaign.DISCRETE_ARMS` 로 되돌아간다. 처음에
    이것만 썼는데, 그러면 추천기가 생성 코드의 상수를 읽어야 하고 데이터가 그
    목록과 다른 조합으로 만들어지면 조용히 틀린다. 되돌림 경로를 쓸 때는
    **그 사실을 주의로 보고**한다. 조용히 되돌아가면 옛 번들과 새 번들의
    동작 차이를 알 수 없다.
    """
    need = [f for f in CAT_AXES if f in feat]
    notes: list[str] = []
    if not need:
        return [{}], notes

    levels = {f: set(int(round(v))
                     for v in (domain.get("categorical", {}).get(f) or []))
              for f in need}

    given = {"pol_code": args.pol, "source_code": args.source,
             "mask_code": args.mask}

    valid, src = observed_combos(need, domain)
    if src == "fallback":
        notes.append(
            "이 번들에는 범주 조합 기록(`categorical_combos`)이 없어 "
            "`campaign.DISCRETE_ARMS` 로 되돌아갔습니다. 데이터가 그 목록과 "
            "다른 조합으로 생성되었으면 틀릴 수 있습니다. --save-model 로 "
            "다시 저장하면 번들이 자기 조합을 들고 갑니다 (12차 보완 74 번).")

    if args.search_arms:
        if not valid:
            raise SystemExit("\n".join([
                "[오류] 탐색할 이산 조합이 없습니다.",
                "",
                "  학습 데이터의 관측 수준에 들어가는 조합이 하나도 없습니다.",
                "",
                "  관측된 수준 : " + ", ".join(
                    f"{f}={sorted(levels[f])}" for f in need),
                "",
                "  --pol, --source, --mask 로 직접 지정하십시오.",
            ]))
        return list(valid), notes

    missing = [f for f in need if given.get(f) is None]
    if missing:
        lines = ["[오류] 이 모형은 범주형 조건을 입력으로 씁니다. 값을 "
                 "주셔야 합니다.", ""]
        for f in need:
            names = CAT_NAMES[f]
            obs = sorted(levels[f])
            lines.append(f"  {f:<12} 관측된 값 : "
                         + ", ".join(names[i] if 0 <= i < len(names)
                                     else str(i) for i in obs))
        lines += ["",
                  "  예 (한 줄로 붙여 쓸 것)", "",
                  "    py recommend.py --models <접두사> --pitch <값> "
                  "--duty <값> --pol TE --source conventional --mask binary",
                  "",
                  "  또는 조합 자체를 탐색 축으로 쓰려면 --search-arms 를",
                  "  주십시오. 데이터에 있는 조합만 돕니다.",
                  "",
                  "  기본값을 두지 않는 이유는, 편광·조명·마스크가 결과를",
                  "  크게 바꾸는 설계 선택이기 때문입니다. 조용히 하나를",
                  "  고르면 사용자가 고르지 않은 조건의 답을 받습니다."]
        raise SystemExit("\n".join(lines))

    arm = {}
    for f in need:
        names = CAT_NAMES[f]
        code = names.index(given[f])
        if levels[f] and code not in levels[f]:
            raise SystemExit("\n".join([
                f"[오류] '{given[f]}' 는 학습 데이터에 없는 값입니다 "
                f"({f}={code}).",
                "",
                "  범주형은 보간이 없으므로 본 적 없는 수준의 예측은 뜻이",
                "  없습니다. 연속 축의 상자 제약과 같은 이유로 거부합니다.",
                "",
                "  관측된 값 : " + ", ".join(
                    names[i] if 0 <= i < len(names) else str(i)
                    for i in sorted(levels[f])),
            ]))
        arm[f] = code

    # 지정한 **조합**이 데이터에 있는지 본다. 축별 수준 검사는 통과했지만
    # 그 조합 자체는 없을 수 있다. 축별 수준의 곱이 8 가지인데 실재하는
    # 조합이 4 가지인 것이 그 실측 예다.
    if valid and not any(all(k[f] == arm[f] for f in need) for k in valid):
        if src == "bundle":
            raise SystemExit("\n".join([
                f"[오류] 지정한 조합 '{arm_label(arm)}' 은 학습 데이터에 "
                f"없습니다.",
                "",
                "  축별 수준은 전부 관측되었지만 그 **조합**은 나온 적이",
                "  없습니다. 축별 수준의 곱에는 실재하지 않는 조합이 섞이며,",
                "  범주형은 보간이 없으므로 그 예측은 뜻이 없습니다.",
                "",
                "  데이터에 있는 조합 :",
                *[f"    - {arm_label(k)}" for k in valid],
            ]))
        notes.append(
            f"지정한 조합 '{arm_label(arm)}' 은 campaign.DISCRETE_ARMS 에 "
            f"없습니다. 번들에 조합 기록이 없어 확정할 수 없으므로 거부하지 "
            f"않지만, 결과를 그만큼 의심하십시오.")
    return [arm], notes


def clip_grid_to_domain(lo: float, hi: float, n: int, ax: dict,
                        conservative: bool = False) -> np.ndarray:
    """격자를 학습 범위 안으로 줄인다. 줄인 사실은 호출부가 보고한다."""
    lo_key, hi_key = ("p01", "p99") if conservative else ("min", "max")
    a = max(lo, ax[lo_key])
    b = min(hi, ax[hi_key])
    if not (b > a):
        return np.array([])
    return np.linspace(a, b, n)


# ----------------------------------------------------------------------------
# 대리 모형으로 만드는 CD 행렬
# ----------------------------------------------------------------------------
def build_X(base: dict, feat: list[str], focus: np.ndarray,
            dose_rel: np.ndarray) -> np.ndarray:
    """(dose, focus) 격자를 평평하게 펼친 입력 행렬.

    행 순서는 `PR.cd_matrix` 의 출력과 같게 (dose 바깥, focus 안쪽) 둔다.
    순서가 어긋나면 윈도가 전치되어도 숫자가 그럴듯하게 나오므로, 여기서
    맞추고 `validate_w12` 가 대조한다.
    """
    nd, nf = len(dose_rel), len(focus)
    X = np.empty((nd * nf, len(feat)), dtype=np.float64)
    for j, f in enumerate(feat):
        if f == "defocus":
            X[:, j] = np.tile(focus, nd)
        elif f == "dose_rel":
            X[:, j] = np.repeat(dose_rel, nf)
        else:
            if f not in base:
                raise SystemExit(
                    f"[오류] 조건 '{f}' 의 값이 주어지지 않았습니다. 이 열은 "
                    "모형의 입력이므로 요청에 반드시 있어야 합니다.")
            X[:, j] = base[f]
    return X


def surrogate_cd_matrix(bundles: dict, base: dict, feat: list[str],
                        target_cd: float, focus: np.ndarray,
                        dose_rel: np.ndarray,
                        X: np.ndarray | None = None) -> dict:
    """대리 모형으로 CD[dose, focus] 와 제약 라벨 행렬을 만든다.

    `X` 를 주면 다시 만들지 않는다. `evaluate` 가 상자 검사에 이미 만들었기
    때문이다(후보 169 개 x 격자 1089 점을 두 번 쌓던 낭비).
    """
    if X is None:
        X = build_X(base, feat, focus, dose_rel)
    nd, nf = len(dose_rel), len(focus)
    pred = {}
    for lab, b in bundles.items():
        pred[lab] = np.asarray(b["model"].predict(X),
                               dtype=np.float64).reshape(nd, nf)
    cd = target_cd + pred["cd_error"]
    return {"cd": cd, "nils": pred["nils"], "cd_3sigma": pred["cd_3sigma"],
            "X": X}


# ----------------------------------------------------------------------------
# 윈도 지표
# ----------------------------------------------------------------------------
def window_metrics(cd: np.ndarray, target_cd: float, tol: float,
                   dose_rel: np.ndarray, focus: np.ndarray,
                   el_target: float) -> dict:
    """CD 행렬 하나에서 윈도 지표를 뽑는다. tol 은 상대값.

    [12 차 보완] DOF 는 `PR.dof_bracket_at_el` 의 **증명 가능한 구간**으로
    읽는다. 전에는 하한을 `dof_at_el` 로, 상한을 `dof_upper` 의 EL 축 보간으로
    따로 만들었는데, 그러면 노광 격자의 EL 편향이 구간에서 빠진다. 해석적
    Bossung 으로 15 개 격자/영역 조합을 재 보니 옛 방식은 6 개에서 참값을
    놓쳤다(무릎뿐 아니라 여유 영역 65x65 에서도 깨졌다). 자세한 측정은
    `litho/process.dof_bracket_at_el` 의 설명과 `validate_w12` Case 8 에 있다.

    키 이름(`dof`, `dof_upper`)은 유지한다. 뜻은 '하한' 과 '상한' 이다.
    """
    if tol <= 0:
        # 키 집합을 성공 경로와 **같게** 둔다. 모양이 다르면 호출부가
        # KeyError 로 죽거나, 더 나쁘게는 .get 으로 조용히 0 을 읽는다.
        return {"tol": float(tol), "dof": 0.0, "dof_upper": 0.0,
                "total": 0.0, "n_pass": 0,
                "focus_edge": False, "dose_edge": False,
                "row_focus_edge": False, "row_dose_edge": False,
                "undetermined": False, "empty_reason": "공차가 0 이하",
                "dose_center": None, "focus_center": None,
                "dose_range": None, "focus_range": None}
    pw = PR.process_window(cd, target_cd, tol_frac=tol)
    curve = PR.el_dof_curve(pw, np.asarray(dose_rel, dtype=np.float64),
                            np.asarray(focus, dtype=np.float64))
    det = PR.dof_bracket_detail(curve, el_target)
    dof, up = det["lo"], det["hi"]
    sat = PR.window_saturated(pw)
    # 운전점. 하한을 만든 직사각형의 **중심**이다. 공정은 창 가운데에
    # 걸어야 양쪽 여유가 같아지므로, 중심이 추천할 값이다. 끝을 추천하면
    # 한쪽으로 치우쳐 같은 창에서도 수율이 나빠진다.
    #
    # 왜 하한 행인가 : 상한 행은 격자 편향을 낙관적으로 더한 가상의
    # 직사각형이라 그 중심이 실제로 통과한다는 보장이 없다. 하한 행은
    # 격자점이 전부 통과한 실제 직사각형이다.
    row = det["lo_row"]
    # [13주차] 보고된 직사각형이 **실제로** 스윕 경계에 닿았는가.
    #
    # `window_saturated` 는 통과 영역 **전체**의 경계 접촉을 본다. 그런데
    # DOF 하한과 운전점은 그중 **한 직사각형**에서 나온다. 다른 노광 행이
    # 경계에 닿아 있으면, 보고된 직사각형이 안쪽이어도 '포화' 로 찍혔다.
    # 실측으로 그런 경우가 나왔다 : `[초점 포화] [노광 포화]` 인데 범위가
    # 초점 [-343.8, 281.2], 노광 [0.8281, 0.9219] 로 둘 다 스윕 안쪽이었다.
    # 읽는 사람은 '이 운전점이 잘렸다' 로 받는다.
    #
    # 두 판정은 뜻이 다르므로 둘 다 남긴다.
    #   row_* : 보고된 숫자가 잘렸다. DOF 와 허용 범위가 하한이다.
    #   *_edge : 창이 격자보다 크다. total 과 EL 이 과소평가일 수 있다.
    row_focus_edge = row_dose_edge = False
    if row is not None and len(focus) and len(dose_rel):
        _f = np.asarray(focus, dtype=np.float64)
        _d = np.asarray(dose_rel, dtype=np.float64)
        _ez = 1e-9 * max(1.0, float(np.ptp(_f)) if _f.size > 1 else 1.0)
        _ed = 1e-9 * max(1.0, float(np.ptp(_d)) if _d.size > 1 else 1.0)
        _zl, _zh = row["focus_range"]
        _dl, _dh = row["dose_range"]
        row_focus_edge = bool(abs(_zl - _f.min()) <= _ez
                              or abs(_zh - _f.max()) <= _ez)
        row_dose_edge = bool(abs(_dl - _d.min()) <= _ed
                             or abs(_dh - _d.max()) <= _ed)
    if row is not None:
        d_lo, d_hi = row["dose_range"]
        z_lo, z_hi = row["focus_range"]
        dose_c = 0.5 * (float(d_lo) + float(d_hi))
        focus_c = 0.5 * (float(z_lo) + float(z_hi))
        d_rng = (float(d_lo), float(d_hi))
        z_rng = (float(z_lo), float(z_hi))
    else:
        dose_c = focus_c = None
        d_rng = z_rng = None
    return {"tol": float(tol), "dof": float(dof), "dof_upper": float(up),
            "total": float(PR.total_window(curve)),
            "n_pass": int(np.count_nonzero(pw)),
            "focus_edge": bool(sat["focus_edge"]),
            "dose_edge": bool(sat["dose_edge"]),
            "row_focus_edge": row_focus_edge,
            "row_dose_edge": row_dose_edge,
            # 하한 0 인데 상한이 양수면 격자가 판정을 못 한 것이다. 0 으로
            # 적어 순위에서 버리면 참 DOF 가 큰 후보가 조용히 탈락한다.
            "undetermined": bool(dof <= 0.0 < up),
            "dose_center": dose_c, "focus_center": focus_c,
            "dose_range": d_rng, "focus_range": z_rng}


def evaluate(bundles: dict, base: dict, feat: list[str], spec: DS.SampleSpec,
             focus: np.ndarray, dose_rel: np.ndarray, rel_p95: float,
             el_target: float = EL_TARGET_DEFAULT,
             conservative: bool = False) -> dict:
    """후보 하나를 평가한다. 상자 밖이면 평가하지 않고 사유를 돌려준다."""
    target_cd = float(spec.duty * spec.pitch)
    dom = next(iter(bundles.values()))["domain"]

    X = build_X(base, feat, focus, dose_rel)
    viol = domain_violations(X, feat, dom, conservative=conservative)
    if viol:
        return {"ok": False, "reason": "상자 밖", "violations": viol}

    m = surrogate_cd_matrix(bundles, base, feat, target_cd, focus, dose_rel,
                            X=X)

    nom = window_metrics(m["cd"], target_cd, spec.pw_tol,
                         dose_rel, focus, el_target)
    guarded_tol = spec.pw_tol - rel_p95
    grd = window_metrics(m["cd"], target_cd, guarded_tol,
                         dose_rel, focus, el_target)

    # 제약은 윈도 안쪽에서만 따진다. 윈도 밖의 값은 애초에 쓰지 않는다.
    pw = PR.process_window(m["cd"], target_cd, tol_frac=spec.pw_tol)
    if np.count_nonzero(pw):
        nils_min = float(np.nanmin(np.where(pw, m["nils"], np.nan)))
        cdu = np.where(pw, m["cd_3sigma"] / np.maximum(m["cd"], 1e-9), np.nan)
        cdu_max = float(np.nanmax(cdu))
    else:
        nils_min, cdu_max = float("nan"), float("nan")

    return {
        "ok": True,
        "target_cd": target_cd,
        "nominal": nom, "guarded": grd,
        "nils_min": nils_min, "min_nils": float(spec.min_nils),
        "nils_ok": bool(np.isfinite(nils_min) and nils_min >= spec.min_nils),
        "cdu_max": cdu_max, "stoch_tol": float(spec.stoch_tol),
        "cdu_ok": bool(np.isfinite(cdu_max) and cdu_max <= spec.stoch_tol),
        "certifiable": bool(guarded_tol > 0),
    }


# ----------------------------------------------------------------------------
# 탐색
# ----------------------------------------------------------------------------
def rank(bundles: dict, feat: list[str], tool: dict, pitch: float,
         duty: float, rel_p95: float, n_na: int, n_sig: int,
         focus_span: float, n_focus: int, dose_span: float, n_dose: int,
         el_target: float, conservative: bool) -> dict:
    """(na, sigma_out) 격자를 전수 평가하고 순위를 매긴다."""
    dom = next(iter(bundles.values()))["domain"]
    ax = dom["axes"]
    for need in ("na", "sigma_out", "defocus", "dose_rel"):
        if need not in ax:
            raise SystemExit(
                f"[오류] 학습 범위에 '{need}' 축이 없습니다. 이 축이 데이터에서 "
                "상수였다면 탐색도 윈도 구성도 할 수 없습니다.")

    # 부호 거울. 둘 다 음수인 수차는 양수로 바꿔 평가할 수 있다(정확한
    # 축퇴). 부호가 섞이면 닿을 수 없으므로 거부한다. SIGN_MIRROR_NOTE 참조.
    tool, mirrored, why = mirror_request(dict(tool), dom)
    if why:
        raise SystemExit(f"[오류] {why}")

    # 고정 부분을 먼저 본다. 후보마다 보면 "N 개 건너뜀" 만 남아 사유를
    # 알 수 없다. check_request 의 설명 참조.
    probe = dict(tool)
    probe.update({"pitch": float(pitch), "duty": float(duty)})
    check_request(probe, feat, dom, conservative=conservative)

    # 초점 격자는 **반드시 홀수**여야 한다. dz = 0 을 격자점으로 포함해야
    # 하기 때문이다.
    #
    # Bossung 은 dz = 0 에서 극값을 갖는다. 노광량이 높아 최적 초점의 CD 가
    # 규격 상한을 넘는 dose 행에서는 통과 영역이 **고리 모양**이 된다
    # (중심이 규격 초과, 그 바깥 띠가 통과). 금지대 중심이 격자 간격보다
    # 얇으면 짝수 격자는 그것을 **건너뛰고** 양쪽을 하나의 연속 통과 구간으로
    # 읽는다. 그러면 DOF 가 과대평가되고 'DOF 는 항상 작게 나온다' 는
    # 이 프로젝트 전체에 깔린 주장이 깨진다.
    #
    # 실측(해석 Bossung, S=1.36, tol 7 %) : n_focus=32 에서 DOF 500.0 nm 로
    # 보고했고 그 직사각형 안에 규격 밖 점이 1385 개 있었다. 참 윈도는 고리
    # 한쪽 띠뿐이라 약 235.9 nm 이므로 **2.12 배 과대보고**다. 같은 조건에서
    # n_focus=33 은 0 nm 다.
    # 홀수면 **범위가 대칭인 한** dz=0 이 표본에 들어가 금지대를 반드시
    # 밟는다. 그 전제는 아래 `centered_focus_grid` 가 세운다. 홀수 올림만으로
    # 는 부족하다는 것이 12 차 보완 69 번이다.
    #
    # GUI 의 슬라이더가 [17, 21, 33, 41] 로 전부 홀수인 것도 같은 이유다.
    # 거기서는 선택지로 막혀 있고, 여기서는 --n-focus 로 짝수가 들어올 수
    # 있으므로 코드가 막는다. 자세한 것은 VERIFICATION_LIMITS 66 번.
    focus_bumped = (n_focus % 2 == 0)
    if focus_bumped:
        n_focus += 1

    lo_key, hi_key = ("p01", "p99") if conservative else ("min", "max")
    nas = np.linspace(ax["na"][lo_key], ax["na"][hi_key], n_na)
    sigs = np.linspace(ax["sigma_out"][lo_key], ax["sigma_out"][hi_key], n_sig)

    focus = clip_grid_to_domain(-focus_span, focus_span, n_focus,
                                ax["defocus"], conservative)
    dose_rel = clip_grid_to_domain(1.0 - dose_span, 1.0 + dose_span, n_dose,
                                   ax["dose_rel"], conservative)
    if focus.size < 3 or dose_rel.size < 3:
        raise SystemExit(
            "[오류] 학습 범위가 좁아 초점/노광 격자를 만들 수 없습니다 "
            f"(초점 {focus.size} 점, 노광 {dose_rel.size} 점).")

    notes = []
    if focus_bumped:
        notes.append(
            f"요청한 초점 점 수 {n_focus - 1} 개를 {n_focus} 개로 "
            f"**홀수로 올림**. dz=0 이 격자점이어야 Bossung 극값 근처의 규격 "
            f"초과 영역을 건너뛰지 않는다(짝수 격자에서 DOF 를 2 배 "
            f"과대보고한 실측이 있다. VERIFICATION_LIMITS 66 번). dz=0 을 "
            f"실제로 보장하는 것은 아래의 재배치이고, 이 올림은 범위가 "
            f"대칭일 때 재배치가 점을 버리지 않게 해 준다(69 번).")
    if focus.size and (focus[0] > -focus_span + 1e-9
                       or focus[-1] < focus_span - 1e-9):
        notes.append(f"초점 격자를 학습 범위로 줄임 : "
                     f"[{focus[0]:.1f}, {focus[-1]:.1f}] nm")
    if dose_rel.size and (dose_rel[0] > 1 - dose_span + 1e-9
                          or dose_rel[-1] < 1 + dose_span - 1e-9):
        notes.append(f"노광 격자를 학습 범위로 줄임 : "
                     f"[{dose_rel[0]:.3f}, {dose_rel[-1]:.3f}] 배")

    # 거울을 적용하면 **초점 축이 반전된다.** CD(+dz, +z) = CD(-dz, -z) 이므로
    # 양수 수차로 바꿔 평가한 결과를 음수 수차 장비의 결과로 읽으려면 초점
    # 격자를 뒤집어야 한다. 격자가 비대칭이면 뒤집은 격자가 원래 격자와 다른
    # 구간을 덮으므로 DOF 를 그대로 가져올 수 없다. 학습 범위의 defocus
    # min/max 는 표본에서 오므로 정확히 대칭일 이유가 없다. 대칭으로 줄여
    # 논증을 정확하게 만든다. 잃는 것은 한쪽 꼬리뿐이다.
    if mirrored and focus.size > 1:
        half = min(abs(float(focus[0])), abs(float(focus[-1])))
        asym = abs(abs(float(focus[0])) - abs(float(focus[-1])))
        if half > 0.0 and asym > 1e-9:
            focus = np.linspace(-half, half, n_focus)
            notes.append(
                f"거울을 적용하므로 초점 격자를 **대칭**으로 줄임 : "
                f"+-{half:.1f} nm (비대칭 {asym:.2f} nm 제거). 뒤집은 격자가 "
                f"원래와 같은 구간을 덮어야 DOF 를 그대로 가져올 수 있다.")

    # [12차 보완 69 번] 여기까지 온 격자는 점 수가 홀수지만 dz=0 을 담는다는
    # 보장이 없다. 66 번에서 홀수를 강제한 근거는 'dz=0 이 격자점이어야
    # 한다' 였는데, 홀수는 **범위가 대칭일 때만** 그것을 세운다. 위에서
    # 학습 범위로 깎으면(`clip_grid_to_domain`) 범위가 비대칭이 될 수 있고,
    # 대칭화는 거울일 때만 한다. 즉 비거울 + 비대칭 경로에서는 전제가 서지
    # 않았다. 주석이 코드보다 많이 주장하고 있었던 것이다. 구조로 닫는다.
    # 무해함을 실측했지만(process.centered_focus_grid 설명) 그 무해함이
    # 'dose 격자가 촘촘하다' 는 조건에 딸려 있어 주석으로 지킬 수 없다.
    if focus.size >= 2:
        focus, cnote = PR.centered_focus_grid(
            float(focus[0]), float(focus[-1]), int(focus.size))
        if cnote:
            notes.append(cnote)
    if focus.size < 3:
        raise SystemExit(
            "[오류] dz=0 을 격자점으로 포함시키니 초점 격자가 "
            f"{focus.size} 점으로 줄었습니다. --n-focus 를 늘리거나 "
            "--focus-span 을 좁혀 주십시오.")

    rows, skipped = [], 0
    for na in nas:
        for sg in sigs:
            base = dict(tool)
            base.update({"na": float(na), "sigma_out": float(sg),
                         "pitch": float(pitch), "duty": float(duty)})
            spec = DS.SampleSpec(pitch=pitch, duty=duty, na=float(na),
                                 sigma_out=float(sg))
            r = evaluate(bundles, base, feat, spec, focus, dose_rel,
                         rel_p95, el_target, conservative)
            if not r["ok"]:
                skipped += 1
                continue
            r["na"], r["sigma_out"] = float(na), float(sg)
            # 모형에 **실제로 넣은** 질의점을 그대로 들고 간다. 외삽 거리를
            # 재는 데 쓴다. 거울을 적용한 후보는 양수 수차로 바꿔 평가했고
            # 모형이 본 것도 그 점이므로, 외삽 거리도 그 점에서 재는 것이
            # 맞다(부호를 되돌린 점은 모형이 본 점이 아니다).
            r["query"] = dict(base)
            # 거울을 적용한 후보의 **초점 오프셋은 부호를 되돌려야 한다.**
            # CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11) 로 음수 수차 장비를 양수로
            # 바꿔 평가했으므로, 양수 쪽에서 얻은 최적 초점 dz* 는 원래
            # 장비에서 -dz* 다. DOF 와 EL 은 폭이라 그대로지만 **오프셋은
            # 위치**라서 그대로 내놓으면 초점을 반대쪽으로 맞추게 된다.
            # 12 차 보완 65 번에서 초점 격자를 대칭으로 줄인 덕에 뒤집은
            # 구간이 원래 구간과 같아서, 부호만 바꾸면 그 값이 유효하다.
            if mirrored:
                for _w in ("nominal", "guarded"):
                    _m = r.get(_w)
                    if not isinstance(_m, dict):
                        continue
                    if _m.get("focus_center") is not None:
                        _m["focus_center"] = -float(_m["focus_center"])
                    if _m.get("focus_range") is not None:
                        _a, _b = _m["focus_range"]
                        _m["focus_range"] = (-float(_b), -float(_a))
            rows.append(r)

    # 목적 : DOF 최대. 동률이면 total window 로 가름. 제약 위반은 뒤로.
    #
    # 정렬 키는 **하한**이다. 상한으로 정렬하면 격자가 거친 후보가 위로
    # 올라온다(모르면 모를수록 좋아 보인다). 하한은 증명된 값이므로 정렬에
    # 쓸 수 있지만, 하한이 0 인데 상한이 양수인 후보는 **격자가 판정을 못 한
    # 것**이어서 '윈도 없음' 과 뜻이 다르다. 그 둘을 같은 0 으로 묶어 버리면
    # 참 DOF 가 112 nm 인 후보가 조용히 탈락한다(해석 측정 : 33x33 격자에서
    # 구간 [0.0, 281.2], 참값 112.7). 따로 모아 보고한다.
    undet = [r for r in rows if r["nominal"]["undetermined"]]
    undet.sort(key=lambda r: r["nominal"]["dof_upper"], reverse=True)
    rows.sort(key=lambda r: (r["nils_ok"] and r["cdu_ok"],
                             r["nominal"]["dof"], r["nominal"]["total"]),
              reverse=True)
    if mirrored:
        notes.append(
            "수차 부호가 둘 다 음수이므로 **부호 거울**을 적용했다. "
            "CD(+dz,+z) = CD(-dz,-z) 가 정확히 성립하므로 DOF 와 EL 은 "
            "그대로이고, 최적 초점 오프셋의 부호만 뒤집힌다.")
    return {"rows": rows, "skipped": skipped, "notes": notes,
            "mirrored": mirrored, "undetermined": undet,
            "focus": focus, "dose_rel": dose_rel}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def extrap_percentile(domain: dict, feat: list[str],
                      base: dict) -> dict | None:
    """질의점이 **학습점의 몇 % 보다 데이터에서 먼가**.

    거리는 **광학·공구 조건 축에서만** 잰다. `defocus`·`dose_rel` 은
    추천기가 격자로 훑으므로 질의점이 하나로 정해지지 않고, 물어야 할 것도
    "이 조건이 학습 데이터에 있는가" 이지 "이 격자점이 있는가" 가 아니다.
    번들이 그 둘을 빼고 기록하므로 여기서도 자연히 빠진다.

    왜 상자 검사로 부족한가 (13 주차 85 번)
    --------------------------------------
    `training_domain` 의 상자 검사는 축별 범위만 본다. 10 차원에서 그것은
    매우 약하다. 모사 측정에서 **각 축 90 백분위** 점은 학습점의 94.9 %
    보다 데이터에서 멀고, 상자 꼭짓점은 100 % 보다 멀다(최근접 2.718 대
    학습점 최대 2.390). 그런데 둘 다 상자 검사를 통과한다. 그리고
    `roundtrip_w13.tool_presets` 가 하는 일이 바로 '각 축 상단 90 %' 이고,
    추천기 자신은 NA·sigma 를 격자에서 훑어 **최댓값을 고른다.** 즉 질의점이
    상자 검사가 가장 약한 영역에 몰린다.

    그래서 상자 안/밖의 이진 판정 대신 **연속적인 외삽 거리**를 돌려준다.
    번들에 분포 기록이 없으면(옛 번들) None 을 주고, 부르는 쪽이 '상자
    검사만 했다' 고 적는다. 조용히 넘기지 않는다.

    참조 분포가 성립하지 않으면 거절한다 (13 주차 90 번)
    --------------------------------------------------
    백분위는 `100 x mean(nn < d)` 이므로 `nn` 이 **0 으로 가득 차면
    포화한다.** 거리가 0 보다 조금만 커도 즉시 95 % 가 나오고, 후보가
    어디 있든 '외삽에 가깝다' 가 찍힌다. 중복 제거 전 번들이 정확히 그
    상태였다(실측 : 223 행 중 서로 다른 조건 14 개, nn 의 95.1 % 가 0).

    그래서 여기서 **거절 조건**을 둔다. 참조 점이 20 개 미만이거나 최근접
    거리의 절반 이상이 0 이면 None 을 주고, 부르는 쪽이 '재지 못했다' 고
    적는다. 숫자를 내놓고 해석을 부르는 쪽에 맡기면 안 된다 — 백분위가
    `nan` 이면 `nan >= 90` 이 거짓이라 조용히 '데이터 안' 으로 찍힌다.
    """
    cov = (domain or {}).get("coverage")
    if not cov:
        return None
    ax = list(cov.get("axes", []))
    sub = np.asarray(cov.get("sample"), dtype=np.float64)
    nn = np.asarray(cov.get("nn_sorted"), dtype=np.float64)
    if not ax or sub.ndim != 2 or sub.shape[0] < 10:
        return None
    nn = nn[np.isfinite(nn)]
    if nn.size < 20 or float(np.mean(nn <= 0.0)) >= 0.5:
        return None
    mu = np.asarray(cov["mu"], dtype=np.float64)
    sd = np.asarray(cov["sd"], dtype=np.float64)
    try:
        q = np.array([float(base[a]) for a in ax], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(q)):
        return None
    z = (q - mu) / np.where(sd > 0, sd, 1.0)
    d = float(np.sqrt(np.min(np.sum((sub - z) ** 2, axis=1))))
    pct = (100.0 * float(np.mean(nn < d)) if nn.size else float("nan"))
    return {"dist": d, "pct": pct, "n_sample": int(sub.shape[0]),
            "nn_median": (float(np.median(nn)) if nn.size else float("nan")),
            "nn_max": (float(nn.max()) if nn.size else float("nan"))}


def _load_conformal(args) -> dict | None:
    """`--conformal` json 을 읽고 **격자 설정이 같은지** 대조한다.

    왜 대조하는가 : 적합예측의 보장은 보정한 양과 적용하는 양이 같은
    양이라는 전제 위에 선다. DOF 는 초점·노광 격자에서 읽는 값이므로
    격자가 다르면 다른 양이고, 그때 구간을 적으면 **보장 없는 숫자에
    보장 딱지를 붙이는 것**이 된다. 11주차 54 번의 복제 문제와 같은
    종류이므로 코드가 막는다.
    """
    path = getattr(args, "conformal", None)
    if not path:
        return None
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            cf = json.load(fh)
    except OSError as e:
        print(f"\n  [경고] 적합예측 파일을 읽지 못했다 : {e}")
        return None
    if cf.get("q_nm") is None:
        print("\n  [경고] 보정이 구간을 만들지 못한 파일이다. 적지 않는다.")
        return None
    g = cf.get("grid", {})
    mism = [f"{k}: 보정 {g[k]} 대 지금 {v}"
            for k, v in (("n_focus", args.n_focus), ("n_dose", args.n_dose),
                         ("focus_span", args.focus_span),
                         ("dose_span", args.dose_span), ("el", args.el),
                         ("pitch", args.pitch), ("duty", args.duty))
            if k in g and abs(float(g[k]) - float(v)) > 1e-9]
    if mism:
        print("\n  [경고] 적합예측 보정 때와 설정이 다르다 : "
              + "; ".join(mism))
        print("  DOF 는 격자에서 읽는 값이라 격자가 다르면 다른 양이다.")
        print("  피복 보장이 깨지므로 구간을 적지 않는다. 같은 설정으로")
        print("  calibrate_w13.py 를 다시 돌릴 것.")
        return None
    return cf


def main() -> int:
    ap = argparse.ArgumentParser(description="공정 조건 추천 (12주차)")
    ap.add_argument("--models", default="models/s_mlp",
                    help="번들 경로 접두사. '<접두사>_<라벨>.joblib' 를 읽는다")
    ap.add_argument("--pitch", type=float, required=True, help="목표 피치 [nm]")
    ap.add_argument("--duty", type=float, required=True, help="목표 duty")
    # 공구 상태. 고르는 값이 아니라 측정해서 넣는 값이다.
    ap.add_argument("--flare", type=float, default=0.0)
    ap.add_argument("--msd-nm", type=float, default=0.0)
    ap.add_argument("--blur-nm", type=float, default=10.0)
    ap.add_argument("--z6", type=float, default=0.0)
    ap.add_argument("--z11", type=float, default=0.0)

    # [12차 보완 74 번] 범주형 조건. 학습 데이터가 이산 조합(arm)으로 나뉘어
    # 있으면 모형의 입력에 pol_code, source_code, mask_code 가 들어간다.
    # 그 값을 줄 방법이 없어서 **실제 데이터셋에서는 추천기가 아예 돌지
    # 않았다.** 이름으로 받아 코드로 바꾼다. 코드 숫자를 직접 받지 않는
    # 이유는 '코드 1 과 3 의 중간' 같은 없는 조건을 막기 위함이다.
    ap.add_argument("--pol", choices=DS.POLARIZATION_CODES, default=None,
                    help="입사 편광. 학습에 쓰인 값만 받는다")
    ap.add_argument("--source", choices=DS.SOURCE_CODES, default=None,
                    help="조명 형상")
    ap.add_argument("--mask", choices=DS.MASK_CODES, default=None,
                    help="마스크 종류")
    ap.add_argument("--search-arms", action="store_true",
                    help="편광·조명·마스크 조합을 고정하지 않고 **탐색 축**으로 "
                         "쓴다. 데이터에 있는 조합만 돈다")
    # 격자
    ap.add_argument("--n-na", type=int, default=13)
    ap.add_argument("--n-sigma", type=int, default=13)
    ap.add_argument("--focus-span", type=float, default=200.0,
                    help="초점 격자 반폭 [nm]. 학습 범위로 자동 축소된다")
    ap.add_argument("--n-focus", type=int, default=N_FOCUS_DEFAULT)
    ap.add_argument("--dose-span", type=float, default=0.15,
                    help="노광 격자 반폭 [1 배 기준 상대]")
    ap.add_argument("--n-dose", type=int, default=N_DOSE_DEFAULT)
    ap.add_argument("--el", type=float, default=EL_TARGET_DEFAULT,
                    help="DOF 를 읽을 목표 EL [%%]")
    ap.add_argument("--rel-p95", type=float, default=None,
                    help="대리 모형의 상대 95 백분위 오차. 보수 윈도에 쓴다. "
                         "주지 않으면 **번들에 저장된 실측치**(최악 영역)를 "
                         f"쓰고, 그것도 없으면 11 주차 실측 "
                         f"{REL_P95_DEFAULT} 를 쓴다")
    ap.add_argument("--conservative", action="store_true",
                    help="상자를 min/max 대신 1~99 백분위로 쓴다")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--conformal", default=None,
                    help="calibrate_w13.py 가 저장한 json. 주면 DOF 의 "
                         "적합예측 구간을 함께 적는다. 격자 설정이 보정 "
                         "때와 다르면 적지 않고 경고한다")
    args = ap.parse_args()

    print("=" * 78)
    print(" 공정 조건 추천")
    print("=" * 78)

    bundles = load_bundles(args.models)

    rel_p95, rel_src, _ms = resolve_rel_p95(bundles, args.rel_p95)

    feat = list(next(iter(bundles.values()))["features"])
    dom = next(iter(bundles.values()))["domain"]
    print(f"\n  모형 {len(bundles)} 개 : {', '.join(sorted(bundles))}")
    print(f"  입력 {len(feat)} 열, 학습 행 {dom.get('n_train_rows', 0):,}")
    print(f"  상자 : {'1~99 백분위 (보수)' if args.conservative else 'min/max'}")

    spec0 = DS.SampleSpec(pitch=args.pitch, duty=args.duty)
    print(f"\n  요구 : 피치 {args.pitch:.1f} nm, duty {args.duty:.3f}, "
          f"목표 CD {args.duty * args.pitch:.2f} nm")
    print(f"  공차 : pw_tol {100 * spec0.pw_tol:.2f} %, "
          f"stoch_tol {100 * spec0.stoch_tol:.2f} %, "
          f"min_nils {spec0.min_nils:.3f} (전부 cd_budget 에서 유도)")
    print(f"  공구 : flare {args.flare:.4f}, MSD {args.msd_nm:.2f} nm, "
          f"블러 {args.blur_nm:.2f} nm, z6 {args.z6:.4f}, z11 {args.z11:.4f}")

    guarded = spec0.pw_tol - rel_p95
    print(f"\n  [인증 가능성] 보수 공차 = pw_tol - 모형 95 백분위 오차 = "
          f"{100 * spec0.pw_tol:.2f} % - {100 * rel_p95:.2f} % = "
          f"{100 * guarded:.3f} %")
    # 어느 숫자를 왜 썼는지 적는다. 출처가 보이지 않으면 다른 모형의 값을
    # 쓰고 있어도 알 수 없다.
    print(f"    모형 오차 출처 : {rel_src}")
    _kind = next((b.get("model_kind") for b in bundles.values()
                  if isinstance(b, dict)), "")
    if _kind:
        print(f"    읽은 모형 종류 : {_kind}"
              + ("  (구간상수다. 같은 격자에서 계단이 생긴다)"
                 if _kind == "hgb" else ""))
    if isinstance(_ms, dict) and _ms.get("rel_p95_overall") is not None \
            and _ms.get("rel_p95_worst_region") is not None \
            and _ms["rel_p95_worst_region"] > _ms["rel_p95_overall"]:
        print(f"      (전체는 {100 * _ms['rel_p95_overall']:.2f} % 지만 최악 "
              f"영역이 {100 * _ms['rel_p95_worst_region']:.2f} % 다. 추천기는 "
              f"그 영역의")
        print(f"       조건도 내놓으므로 최악 쪽을 쓴다.)")
    if guarded <= 0:
        print("    -> 0 이하다. **대리 모형만으로는 공정 윈도를 인증할 수 없다.**")
        print("       모형 오차가 공차를 다 먹기 때문이다. 아래 순위는 후보를")
        print("       **고르는** 데만 쓰고, 공정 윈도 값은 13 주차에 시뮬레이터로")
        print("       되돌려 재야 한다. 비교에서는 모형 오차가 부분적으로")
        print("       상쇄되므로 순위 자체는 쓸 수 있다.")
    else:
        print("    -> 0 보다 크다. 보수 윈도가 비어 있지 않으면 그 값은 모형")
        print("       오차를 최악으로 떼어낸 뒤에도 남는 윈도다.")

    tool = {"flare": args.flare, "msd_nm": args.msd_nm,
            "blur_nm": args.blur_nm, "z6": args.z6, "z11": args.z11}

    # [12차 보완 74 번] 공급할 수 없는 입력이 있으면 **격자를 돌기 전에**
    # 멈춘다. 돌다가 죽으면 몇 분을 버리고, 메시지만으로는 무엇을 해야
    # 하는지 알 수 없다. 67 번과 같은 종류다.
    _miss = unsupplied_inputs(feat)
    if _miss:
        lines = ["[오류] 이 모형이 입력으로 쓰는 열을 추천기가 공급할 수 "
                 "없습니다.", ""]
        for f in _miss:
            why = UNSUPPLIED.get(f)
            lines.append(f"  {f:<14}" + (why if why else
                                         "추천기가 모르는 열이다"))
        lines += ["",
                  "  추천기는 아래 네 가지 방법으로만 입력을 공급합니다.",
                  f"    격자로 쓸어  : {', '.join(SUPPLY['grid'])}",
                  f"    탐색         : {', '.join(SUPPLY['search'])}",
                  f"    요청으로 받음 : {', '.join(SUPPLY['request'])}",
                  f"    이산 조합    : {', '.join(SUPPLY['categorical'])}",
                  "",
                  "  위 열이 학습 데이터에서 변했다면, 그 축을 고정해 다시",
                  "  학습하거나 추천기에 공급 경로를 추가해야 합니다. 값을",
                  "  임의로 채우면 사용자가 고르지 않은 조건의 답이 나옵니다."]
        raise SystemExit("\n".join(lines))

    # 범주형 조합을 정한다. 고정하거나 탐색 축으로 쓴다.
    arms, arm_note = resolve_arms(feat, dom, args)
    for nt in arm_note:
        print(f"\n  [주의] {nt}")
    if len(arms) > 1:
        print(f"\n  이산 조합 {len(arms)} 개를 **탐색 축**으로 함 :")
        for a in arms:
            print(f"    - {arm_label(a)}")

    res = None
    rows, undet_all, notes_all = [], [], []
    for arm in arms:
        t = dict(tool); t.update(arm)
        r = rank(bundles, feat, t, args.pitch, args.duty, rel_p95,
                 args.n_na, args.n_sigma, args.focus_span, args.n_focus,
                 args.dose_span, args.n_dose, args.el, args.conservative)
        lab = arm_label(arm) if len(arms) > 1 else ""
        for row in r["rows"]:
            row["arm"] = lab
            rows.append(row)
        for row in r["undetermined"]:
            row["arm"] = lab
            undet_all.append(row)
        for nt in r["notes"]:
            if nt not in notes_all:
                notes_all.append(nt)
        res = r
    # 조합을 섞어 다시 정렬한다. 정렬 키는 rank 와 같아야 한다.
    rows.sort(key=lambda r: (r["nils_ok"] and r["cdu_ok"],
                             r["nominal"]["dof"], r["nominal"]["total"]),
              reverse=True)
    undet_all.sort(key=lambda r: r["nominal"]["dof_upper"], reverse=True)
    res = dict(res)
    res["rows"], res["undetermined"], res["notes"] = rows, undet_all, notes_all

    for nt in res["notes"]:
        print(f"\n  [주의] {nt}")
    print(f"\n  격자 : na {args.n_na} x sigma {args.n_sigma} "
          f"= {args.n_na * args.n_sigma} 후보" 
          + (f" x 이산 조합 {len(arms)} = {args.n_na * args.n_sigma * len(arms)} 후보"
             if len(arms) > 1 else "")
          + f", 각 후보마다 초점 "
          f"{len(res['focus'])} x 노광 {len(res['dose_rel'])} 점")
    print(f"  대리 모형 호출 {args.n_na * args.n_sigma * len(arms) * len(res['focus']) * len(res['dose_rel']) * len(bundles):,} 회")

    rows = res["rows"]
    if not rows:
        print("\n[오류] 평가된 후보가 없습니다.")
        return 1

    print("\n" + "=" * 78)
    print(f" 상위 {min(args.top, len(rows))} 후보  (EL {args.el:.0f} % 기준)")
    print("=" * 78)
    _multi = len(arms) > 1
    # 적합예측 구간. 격자 설정이 보정 때와 다르면 보정량과 적용량이 다른
    # 양이 되어 피복 보장이 깨지므로, 그때는 적지 않고 경고한다.
    _cf = _load_conformal(args)
    print(f"\n  {'NA':>6}{'sigma':>8}{'DOF 하한':>11}{'DOF 상한':>10}"
          f"{'보수 하한':>10}{'NILS':>8}{'CDU %':>8}  판정"
          + ("   이산 조합" if _multi else ""))
    print("  " + "-" * (84 if _multi else 72))
    for r in rows[:args.top]:
        ok = "통과" if (r["nils_ok"] and r["cdu_ok"]) else "제약 위반"
        # 보고된 숫자가 잘렸는지를 1차로 적는다. 창 전체가 격자보다 큰 것은
        # 뜻이 달라 표 아래에 따로 적는다(13주차).
        flag = ""
        if r["nominal"]["row_focus_edge"]:
            flag += " [초점 잘림]"
        if r["nominal"]["row_dose_edge"]:
            flag += " [노광 잘림]"
        if r["nominal"]["undetermined"]:
            # 이 표만 읽는 사람이 하한 0 을 '윈도 없음' 으로 읽으면 안 된다.
            flag += " [격자 미결정]"
        print(f"  {r['na']:>6.3f}{r['sigma_out']:>8.3f}"
              f"{r['nominal']['dof']:>11.1f}{r['nominal']['dof_upper']:>10.1f}"
              f"{r['guarded']['dof']:>10.1f}{r['nils_min']:>8.3f}"
              f"{100 * r['cdu_max']:>8.2f}  {ok}{flag}"
              + (f"   {r.get('arm', '')}" if _multi else ""))
        if _cf:
            q = float(_cf["q_nm"])
            lo = max(0.0, r["nominal"]["dof"] - q)
            print(f"        -> DOF {r['nominal']['dof']:.1f} nm, "
                  f"적합예측 {100 * float(_cf['guarantee']):.0f} % 구간 "
                  f"[{lo:.1f}, {r['nominal']['dof'] + q:.1f}] nm "
                  f"(+-{q:.1f})")

    # 외삽 거리. 상자 검사는 필요조건일 뿐이고 10 차원에서 매우 약하다.
    # 추천기는 NA·sigma 의 **최댓값**을 고르므로 질의점이 바로 그 약한
    # 영역에 몰린다. 상자 안이어도 데이터에서 먼지를 말해 준다.
    _cov = []
    for r in rows[:args.top]:
        _q = r.get("query")
        _e = extrap_percentile(dom, feat, _q) if _q else None
        if _e:
            _cov.append((r, _e))
    if _cov:
        print("\n  [외삽 거리] 질의점이 학습 데이터에서 얼마나 먼가")
        print("  상자 검사(축별 범위)는 **필요조건일 뿐**이다. 10 차원에서는")
        print("  각 축 90 백분위 점이 학습점의 94.9 % 보다 멀면서도 상자")
        print("  검사를 통과한다(모사 측정). 추천기는 NA·sigma 의 최댓값을")
        print("  고르므로 질의점이 바로 그 영역에 몰린다.")
        _w = max(e["pct"] for _, e in _cov)
        for r, e in _cov:
            _tag = ("**학습 범위를 벗어난다**" if e["pct"] >= 99.5
                    else "외삽에 가깝다" if e["pct"] >= 90.0 else "데이터 안")
            print(f"    NA {r['na']:.3f} sigma "
                  f"{r['sigma_out']:.3f} : "
                  f"최근접 {e['dist']:.2f} (학습 조건 중 "
                  f"{e['pct']:.1f} % 보다 멀다)  {_tag}")
        # 세는 단위는 **서로 다른 학습 조건**이다. 행이 아니다. 초점·노광
        # 격자로 같은 조건이 수십 번 들어오므로, 행으로 세면 참조 분포가
        # 0 으로 포화한다(13차 보완 90 번). 단위를 분명히 적어 둔다.
        print(f"    (학습 조건끼리의 최근접 거리 중위 "
              f"{_cov[0][1]['nn_median']:.2f}, 최대 "
              f"{_cov[0][1]['nn_max']:.2f}, 서로 다른 조건 "
              f"{_cov[0][1]['n_sample']} 개. 행 수가 아니라 조건 수다)")
        if _w >= 99.5:
            print("    -> 어떤 후보가 학습 데이터 밖이다. 상자 검사를")
            print("       통과했어도 그 예측은 **외삽**이며, 모형 오차가")
            print("       분포 내 측정치(95 백분위)보다 클 수 있다.")
            print("       되돌림(roundtrip_w13.py)으로 반드시 확인할 것.")
    else:
        # 두 가지 경우를 **구분해서** 적는다. 옛 번들(기록 없음)과, 기록은
        # 있으나 참조 분포가 성립하지 않는 경우(13차 보완 90 번)는 사용자가
        # 할 일이 다르다. 한 문장으로 뭉치면 엉뚱한 조치를 하게 된다.
        _cv = (dom or {}).get("coverage") or {}
        _nn = np.asarray(_cv.get("nn_sorted", []), dtype=np.float64)
        _nn = _nn[np.isfinite(_nn)]
        if not _cv:
            _degen = False                      # 기록 자체가 없다 (옛 번들)
        elif _nn.size < 20:
            _degen = True
        else:
            _degen = float(np.mean(_nn <= 0.0)) >= 0.5
        print("\n  [외삽 거리] 재지 못했다. **상자 검사만** 했다. 상자는")
        print("  필요조건일 뿐이고 10 차원에서 매우 약하다.")
        if _degen:
            print("  이유 : 번들의 참조 거리 분포가 성립하지 않는다(서로 다른")
            print("  조건이 20 개 미만이거나 최근접 거리의 절반 이상이 0).")
            print("  백분위가 포화하므로 숫자를 적지 않는다. 데이터셋의 광학")
            print("  조건 가짓수를 늘리거나, 중복 제거를 담은 판으로")
            print("  train_surrogate.py 를 다시 돌려야 한다.")
        else:
            print("  이유 : 번들에 결합 분포 기록이 없다(옛 번들).")
            print("  train_surrogate.py 를 다시 돌려 번들에 분포를 담으면 이")
            print("  칸에 '학습 조건의 몇 % 보다 먼가' 가 나온다.")

    if _cf:
        print(f"\n  [적합예측 구간] 분할 적합예측으로 보정한 구간이다. 위의")
        print(f"  'DOF 하한/상한' 과 **다른 것**을 잰다. 그쪽은 격자 편향만")
        print(f"  담고, 이쪽은 **모형 오차**를 담는다. 보정 표본 "
              f"{_cf.get('n_cal')} 개, 순서통계량 {_cf.get('k')} 번째,")
        print(f"  정확한 보장 피복률 {100 * float(_cf['guarantee']):.2f} % "
              f"(표기 {100 * (1 - float(_cf['alpha'])):.0f} % 이상).")
        print("  보장은 **시뮬레이터 대비**이고 웨이퍼 대비가 아니다. 또한")
        print("  보정 조건은 데이터셋 검증 분할이고 여기는 최적점 근처이므로,")
        print("  분포가 이동한 만큼 실제 피복률은 표기보다 낮을 수 있다.")
        print("  그 이동의 크기는 roundtrip_w13.py 가 따로 잰다.")
        # [13차 보완 86 번] **다중성.** 구간을 후보마다 적으면 '각각' 90 %
        # 이고 '전부 동시에' 90 % 가 아니다. 독립이라 해도 k 개가 모두
        # 맞을 확률은 0.9^k 이며(k=5 면 59 %), 후보들은 같은 모형에서 나와
        # 상관되어 있으므로 실제 값은 그 사이다. 그리고 사용자는 보통
        # **가장 좋은 후보 하나**를 고르는데, 그 선택이 예측값에 의존하므로
        # 선택된 후보의 구간은 표기보다 덜 맞는다(승자의 저주).
        _k = len(rows[:args.top])
        print(f"  [다중성] 위 구간은 후보 **각각** 에 대해 "
              f"{100 * float(_cf['guarantee']):.0f} % 다. {_k} 개가 모두")
        print(f"  맞을 확률은 그보다 낮다(독립이면 "
              f"{100 * float(_cf['guarantee']) ** _k:.0f} %, 후보들이 같은")
        print("  모형에서 나와 상관되므로 실제 값은 그 사이다). 그리고 가장")
        print("  좋은 후보를 고르는 행위 자체가 예측값에 의존하므로, 고른")
        print("  후보의 구간은 표기보다 덜 맞는다. 한 후보만 보장이 필요하면")
        print("  --alpha 를 후보 수로 나눠(본페로니) 다시 보정할 것.")
        _st = _cf.get("q_over_step")
        if _st is not None and float(_st) <= 2.5:
            print(f"  [주의] q 가 보정 때 초점 격자 간격의 {float(_st):.2f} 배다.")
            print("  구간이 모형 오차보다 격자 분해능을 재고 있을 수 있다.")

    # ---- 운전점 ------------------------------------------------------------
    # 왜 따로 인쇄하는가 : 위 표는 창의 **폭**만 준다. '공정 조건 추천' 은
    # 장비에 넣을 값, 즉 노광량과 초점 오프셋을 줘야 한다. 폭만 주면 받는
    # 사람이 공정을 걸 수 없다.
    _best = [r for r in rows[:args.top]
             if r["nominal"].get("dose_center") is not None]
    if _best:
        # 공칭이라고 **명시**한다. 바로 위 표에 '보수 하한 0' 이 찍혀 있어,
        # 적어 두지 않으면 이 운전점이 보수 창의 것으로 읽힌다.
        print("\n  [추천 운전점]  **공칭** 창의 하한을 만든 직사각형의 중심")
        print("  공정은 창 가운데에 걸어야 양쪽 여유가 같아진다. 끝을 쓰면")
        print("  같은 창에서도 한쪽으로 치우쳐 수율이 나빠진다.")
        print("  공칭이므로 모형 오차를 떼지 않은 값이다. 보수 창이 있는")
        print("  경우 그 창은 더 좁고 중심도 달라질 수 있다. 위 '보수 하한'")
        print("  이 0 이면 보수 창이 아예 없다는 뜻이다.")
        if res["mirrored"]:
            print("  부호 거울을 적용했으므로 **초점 오프셋의 부호는 되돌려**")
            print("  이 장비 기준으로 적었다. 노광량은 거울과 무관하다.")
        # 판정을 **이 표에도** 싣는다. 위 표에만 적으면, 운전점만 보고
        # 가져가는 사람이 NILS 미달 조건을 그대로 장비에 넣는다. 걸러
        # 버리지 않고 함께 적는 쪽을 고른 이유는, 제약 위반 후보도 '왜
        # 안 되는지' 를 보여 주는 것이 정보이기 때문이다.
        # 머리행과 본문의 열이 **같아야** 한다. 이산 조합을 탐색할 때 본문에만
        # 조합을 붙이고 머리행에 안 적어 두었더니, 표가 한 열 어긋나 보였다.
        print(f"\n  {'NA':>6}{'sigma':>8}{'노광량 (배율)':>15}"
              f"{'초점 오프셋 nm':>16}{'창 폭 nm':>11}  판정"
              + ("   이산 조합" if _multi else ""))
        print("  " + "-" * (84 if _multi else 72))
        for r in _best:
            nm_ = r["nominal"]
            d_lo, d_hi = nm_["dose_range"]
            z_lo, z_hi = nm_["focus_range"]
            ok_ = "통과" if (r["nils_ok"] and r["cdu_ok"]) else "제약 위반"
            fl_ = ""
            if nm_["row_focus_edge"]:
                fl_ += " [초점 잘림]"
            if nm_["row_dose_edge"]:
                fl_ += " [노광 잘림]"
            print(f"  {r['na']:>6.3f}{r['sigma_out']:>8.3f}"
                  f"{nm_['dose_center']:>15.4f}"
                  f"{nm_['focus_center']:>16.1f}{nm_['dof']:>11.1f}"
                  f"  {ok_}{fl_}"
                  + (f"   {r.get('arm', '')}" if _multi else ""))
            print(f"  {'':>14}허용 범위 : 노광 배율 [{d_lo:.4f}, "
                  f"{d_hi:.4f}], 초점 [{z_lo:.1f}, {z_hi:.1f}] nm")
        # [13차 보완 86 번] **초점 오프셋의 부호는 CD 데이터로 정해지지
        # 않는다.** 거울 대칭 CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11) 가 기계
        # 정밀도로 성립한다(회귀로 고정). 따라서 CD 계측만으로 역추정한
        # 수차는 **부호가 함께 뒤집힌 해와 구분되지 않고**, 그 두 해는 초점
        # 오프셋의 부호가 반대다. 주장 2 가 말하는 비식별이 추천의 가장
        # 실행 가능한 출력에 그대로 전파된다.
        #
        # 실측 비용 : z6 0.03 / z11 0.02 에서 중심이 ±25.0 nm 이므로 부호를
        # 틀리면 최적에서 50 nm 벗어나고, 이는 창 반폭 105 nm 의 **0.476
        # 배**다. z6 0.05 면 중심 40 nm 로 0.73 배가 된다. 창 밖으로 나가진
        # 않지만 한쪽 초점 여유의 절반 가까이를 잃는다. 운전점을 창 가운데에
        # 거는 목적 자체가 무너진다.
        _off = [r for r in _best
                if abs(float(r["nominal"]["focus_center"])) > 1e-9]
        if _off:
            _w = max(abs(float(r["nominal"]["focus_center"])) for r in _off)
            _hw = [((float(r["nominal"]["focus_range"][1])
                     - float(r["nominal"]["focus_range"][0])) / 2.0)
                   for r in _off]
            _hw = max([h for h in _hw if h > 0] or [float("nan")])
            print("\n  [초점 오프셋 부호] 위 오프셋의 **부호는 CD 데이터로")
            print("  정해지지 않는다.** CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11)")
            print("  가 기계 정밀도로 성립하므로(거울 대칭), CD 계측만으로")
            print("  역추정한 수차는 부호가 함께 뒤집힌 해와 구분되지 않고")
            print("  그 두 해는 초점 오프셋의 부호가 반대다.")
            print(f"  부호를 틀리면 최적에서 최대 {2 * _w:.1f} nm 벗어난다"
                  + (f" (창 반폭 {_hw:.1f} nm 의 {2 * _w / _hw:.2f} 배)"
                     if _hw == _hw and _hw > 0 else ""))
            print("  -> 수차 부호는 **장비의 수차 계측**에서 받아야 한다.")
            print("     CD 역추정으로 얻은 값이면 오프셋을 ± 로 읽고, 부호는")
            print("     초점 스윕 실험으로 따로 정할 것. 창 밖으로 나가지는")
            print("     않지만 한쪽 여유의 절반 가까이를 잃는다.")
        if any(not (r["nils_ok"] and r["cdu_ok"]) for r in _best):
            print("\n  '제약 위반' 행의 운전점은 **쓰면 안 된다.** 창은 있지만")
            print("  NILS 또는 CDU 가 공차를 넘는다. 창 폭만 보고 고르면")
            print("  놓치는 조건이므로 함께 적는다.")
        if any(r["nominal"]["row_focus_edge"] or r["nominal"]["row_dose_edge"]
               for r in _best):
            print("\n  '잘림' 행의 허용 범위와 창 폭은 **스윕 경계에서 잘린**")
            print("  값이다. 즉 그 숫자는 하한이고 실제 범위는 더 넓다.")
            print("  --focus-span 또는 --dose-span 을 넓혀 다시 확인할 것.")
        # 창 전체가 격자보다 큰 경우. 보고된 직사각형이 안쪽이어도 EL 과
        # total 은 과소평가일 수 있다. '잘림' 과 뜻이 다르므로 따로 적는다.
        _wide = [r for r in _best
                 if (r["nominal"]["focus_edge"] or r["nominal"]["dose_edge"])
                 and not (r["nominal"]["row_focus_edge"]
                          or r["nominal"]["row_dose_edge"])]
        if _wide:
            print(f"\n  위 {len(_wide)} 개 후보는 보고된 운전점은 스윕 안쪽이지만")
            print("  **통과 영역 전체**가 격자 경계에 닿아 있다. 이 운전점의")
            print("  숫자는 잘리지 않았으나, EL 과 전체 창 면적은 과소평가일")
            print("  수 있다. 순위가 그 두 값으로 갈릴 때만 문제가 된다.")
        # 단위를 정확히 적는다. 이 프로젝트의 `e_op` 는 업계의 E_op
        # (mJ/cm^2)와 이름만 같고 **무차원 광강도 배율**이다. 상수 임계값
        # 모형이라 레지스트 감도를 두지 않으므로 물리 노광량으로 환산할 수
        # 없다(litho/dataset.py 의 LABEL_KEYS 주석이 같은 경고를 한다).
        # 여기서 'x E_op' 를 설명 없이 적으면 읽는 사람이 mJ/cm^2 로 받는다.
        print("\n  노광량은 **무차원 배율**이다. 최적 초점에서 목표 CD 를")
        print("  만드는 광강도를 1.0 으로 둔 값이며, 업계에서 쓰는")
        print("  E_op [mJ/cm^2] 와 이름만 같고 단위가 다르다. 본 모형은")
        print("  상수 임계값 모형이라 레지스트 감도를 두지 않으므로 물리")
        print("  노광량으로 환산할 수 없다. 절대값이 필요하면 데이터셋의")
        print("  dose_clear_mj (= threshold x dose_mj) 를 쓸 것.")
        print("  초점 오프셋은 최적 초점(dz=0) 기준의 nm 단위다.")

    und = res["undetermined"]
    if und:
        # 전체 개수를 함께 적는다. '5 개' 만 보이면 그것이 전부인지 표의
        # 상한인지 읽는 사람이 알 수 없다.
        print(f"\n  [격자 미결정] 후보 {len(rows)} 개 가운데 {len(und)} 개. "
              f"하한 0, 상한 양수다.")
        print("  윈도가 없는 것이 아니라 **격자가 판정하지 못한** 후보다. 위")
        print(f"  표의 0 과 뜻이 다르므로 따로 적는다. 상한 큰 순서로 "
              f"{min(5, len(und))} 개만 보인다.")
        print(f"\n  {'NA':>6}{'sigma':>8}{'DOF 하한':>11}{'DOF 상한':>10}"
              f"{'EL 분해능 %':>13}")
        print("  " + "-" * 52)
        dmid = float(np.mean(res["dose_rel"])) if len(res["dose_rel"]) else 1.0
        el_res = (200.0 * float(np.max(np.diff(res["dose_rel"]))) / dmid
                  if len(res["dose_rel"]) > 1 and dmid > 0 else float("nan"))
        for r in und[:5]:
            print(f"  {r['na']:>6.3f}{r['sigma_out']:>8.3f}"
                  f"{r['nominal']['dof']:>11.1f}"
                  f"{r['nominal']['dof_upper']:>10.1f}{el_res:>13.3f}")
        print(f"\n  처방 : --n-dose 를 올려 EL 분해능을 좁힌다. 지금 "
              f"{el_res:.3f} % 이고")
        print(f"  목표 EL 은 {args.el:.0f} % 다. 분해능이 목표의 몇 %를 먹는지가")
        print("  그대로 구간 폭이 된다. 격자를 늘리면 구간이 좁아지며 하한이")
        print("  올라오거나 상한이 0 으로 내려앉아 판정이 끝난다.")

    print("\n  [읽는 법]")
    print("  - DOF 는 **구간**으로 읽는다. 참값은 하한과 상한 사이에 있다.")
    print("    구간은 초점 격자와 노광 격자 **양쪽**의 편향을 담는다. 노광")
    print("    쪽을 빼면 구간이 참값을 놓친다(12 차 보완 65 번).")
    print("  - 순위는 **하한**으로 매긴다. 상한으로 매기면 격자가 거친 후보가")
    print("    위로 올라온다. 모르면 모를수록 좋아 보이는 정렬은 쓸 수 없다.")
    print("  - '잘림' 은 **보고된 직사각형**이 스윕 경계에 닿았다는 뜻이다.")
    print("    그 DOF 와 허용 범위는 하한이며, 스윕을 넓혀 다시 재야 한다.")
    print("    창 전체가 경계에 닿은 것과는 뜻이 다르다. 후자는 운전점")
    print("    숫자를 잘리게 하지 않고 EL 과 전체 면적만 과소평가한다.")
    print("  - 보수 하한이 0 이면 모형 오차를 떼면 윈도가 남지 않는다는 뜻이다.")
    print("    순위는 유효하지만 값은 인증이 아니다.")
    print()
    print("  [범위] 이 표는 **대리 모형의 예측**이다. 추천 조건이 실제로 이")
    print("  윈도를 갖는지는 시뮬레이터로 되돌려 재야 확인된다(13 주차).")
    print("  모형 오차는 데이터와 같은 분포에서 뽑은 조건에 대한 값이고,")
    print("  추천 조건은 데이터에 없는 점일 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
