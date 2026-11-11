"""claims.py -- README 의 대표 주장에 적힌 숫자를 실행 시점에 재계산해 대조한다.

왜 필요한가
-----------
이 과제에서 반복적으로 난 사고는 코드 버그가 아니라 **문서와 코드의 드리프트**
였다. 코드를 고쳐 숫자가 바뀌었는데 README·VERIFICATION_LIMITS 에 적어 둔
값을 함께 고치지 않는 것이다. 실제로 있었던 것만 적으면

  - 영공간 판정을 보정해 5 변수 defocus 하한이 '무한' 이 되었는데, 본문에는
    이전 값 0.194 nm 가 그대로 남아 있었다.
  - CRB 몬테카를로의 sd/CRB 범위를 0.93~1.07 로 적어 두었는데 실측은
    0.84~1.18 이었다. 범위를 좁게 적으면 '운 좋게 맞은 것' 처럼 보인다.
  - 미지수 사다리의 하한을 적으면서 **초점값(80 nm)을 빼고** 적었다. 그런데
    바로 그 절에서 "계측 조건 없이 인용하면 안 된다" 고 경고하고 있었다.

문서는 어떤 시험도 읽지 않으므로, 회귀 시험 전부와 verify A~K 가 통과해도
이런 드리프트는 잡히지 않는다. 그래서 대표 주장의 숫자만이라도 기계가
재계산해 대조하게 둔다. 사람이 눈으로 맞추는 일을 반복하지 않는 것이 목적이다.

무엇을 대조하는가
-----------------
두 묶음이다.

  1. README '주장 1' 의 숫자 전부. 이 과제의 결론이 걸린 숫자들이고, 모두
     ID.crb / ID.collinearity 로 몇 초 안에 재계산된다.
  2. README '검증 요약표' 의 숫자. 이쪽은 회귀 시험이 계산하기는 하지만
     **README 에 적힌 값과 대조하지는 않는다**. 시험은 자기가 구한 값을
     인쇄하고 허용대 안에 있는지만 보므로, 코드가 바뀌어 값이 허용대
     안에서 움직이면 시험은 계속 통과하고 README 만 낡는다. 실제로 이
     확장을 넣자마자 k1 조명 최솟값이 문서 0.279 / 실측 0.2779 로
     어긋난 것이 드러났다. 허용대가 넓은 시험은 드리프트를 못 잡는다.

LER 처럼 몬테카를로가 필요한 양은 여기 넣지 않는다(verify.py H 가 담당).

의도적으로 묶지 않는 숫자
-------------------------
README 의 모든 수치를 넣는 것이 목적이 아니다. 아래는 넣지 않으며, 이는
누락이 아니라 판단이다.

  - DOI·권호·연도 : 코드가 계산하는 값이 아니다.
  - 허용대 상수(2 시그마 0.90~1.10, 허용 1e-3 등) : 코드에 적힌 상수
    자체이므로 재계산해 대조하는 것이 동어반복이다.
  - 몬테카를로 실측치(sd/CRB 1.02, LER 절대값) : 난수 종자에 따라
    움직이므로 '불일치' 판정을 붙이면 거짓 경보가 된다. verify H 가
    허용대로 본다.
  - 회귀 케이스 총수(현재 66 종) : 세려면 회귀 전부를 돌려야 하므로
    여기서 묶을 수 없다(이 파일은 4 초 안에 끝나야 한다). 대신
    `run_all_checks.py` 가 각 스위트의 'n/n' 을 **합산해 인쇄**하므로
    사람이 한 줄로 대조할 수 있다. 12 주차에 50 에서 늘어나면서 열두
    곳을 함께 고쳤고, 그 드리프트를 막는 장치가 그 합계 줄이다.
  - 시뮬레이터를 직접 돌려 얻는 물리 측정치(부호 거울의 등가류 차
    8.236 nm 등) : 연산이 길어 이 파일의 역할과 맞지 않다.
    `validate_w12.py` Case 7 이 매 회차 재측정한다.

조건을 코드에 함께 적어 둔다. 하한은 조건 없이는 숫자가 아니다.

사용법
------
    py claims.py          # 대조표 출력, 불일치가 있으면 종료 코드 1
"""

from __future__ import annotations

import io
import sys
from dataclasses import replace

import numpy as np

from litho import identify as ID

#: README 주장 1 이 서 있는 조건. IdentifySpec 의 기본값과 같아야 한다.
CONDITION = dict(wavelength=193.0, na=1.35, sigma_out=0.7, pitch=200.0,
                 duty=0.5, defocus=80.0, dose_mj=30.0, y_length=2048.0,
                 dy=2.0)


def _sd(spec, params) -> float:
    """params 를 동시 미지수로 둘 때의 sigma(defocus)."""
    c = ID.crb(spec, params=tuple(params))
    return float(np.asarray(c["sigma_min"], dtype=float)[0])


def _rel(got: float, want: float) -> float:
    """상대 오차. 둘 다 무한이면 0, 한쪽만 무한이면 무한."""
    if not np.isfinite(want) and not np.isfinite(got):
        return 0.0
    if not np.isfinite(want) or not np.isfinite(got):
        return np.inf
    if want == 0.0:
        return abs(got)
    return abs(got - want) / abs(want)


def build_cases(spec):
    """(설명, 문서에 적힌 값, 실측, 허용 상대오차) 목록."""
    full = ID.DEFAULT_PARAMS
    lad = [("defocus",),
           ("defocus", "dose_rel"),
           ("defocus", "dose_rel", "blur_nm"),
           ("defocus", "dose_rel", "blur_nm", "flare"),
           full]
    doc_lad = [0.0096, 0.0096, 0.0256, 1.891, np.inf]

    cases = []
    for ps, want in zip(lad, doc_lad):
        label = f"사다리 {len(ps)} 변수 sigma(dz)"
        cases.append((label, want, _sd(spec, ps), 5e-3))

    # 플레어가 더해질 때의 악화 배수. README 는 '196 배' 로 적는다.
    base = _sd(spec, ("defocus",))
    four = _sd(spec, ("defocus", "dose_rel", "blur_nm", "flare"))
    cases.append(("플레어 추가 악화 배수", 196.0, four / base, 5e-3))

    # 민감도 벡터 사이각.
    col = ID.collinearity(spec, params=full)
    ang = {frozenset((p["a"], p["b"])): p["angle_deg"] for p in col["pairs"]}
    cases.append(("초점-플레어 사이각 [도]", 2.52,
                  ang[frozenset(("defocus", "flare"))], 5e-3))
    # 블러-MSD 는 해석적으로 완전 축퇴다. 남는 것은 수치 잔차이므로
    # 상대오차가 아니라 절대 상한으로 본다.
    bm = ang[frozenset(("blur_nm", "msd_nm"))]
    cases.append(("블러-MSD 사이각 [도] (해석상 0)", 0.0, bm, 1e-2))

    # 계측 처방. 그 변수를 따로 재서 미지수에서 뺐을 때 남는 하한.
    for drop, want in (("flare", 0.0257), ("msd_nm", 1.891),
                       ("blur_nm", np.inf), ("dose_rel", np.inf)):
        ps = tuple(p for p in full if p != drop)
        cases.append((f"{drop} 제외 후 sigma(dz)", want, _sd(spec, ps), 5e-3))

    # 광자를 100 배 모아도 변하지 않는다는 것이 주장의 핵심이다.
    for dose, want in ((10.0, np.inf), (1000.0, np.inf)):
        got = _sd(replace(spec, dose_mj=dose), full)
        cases.append((f"노광량 {dose:.0f} mJ 에서 sigma(dz)", want, got, 0.0))

    # 5 변수에서의 rank 와 영공간 성분.
    c5 = ID.crb(spec, params=full)
    cases.append(("5 변수 rank", 4.0, float(c5["rank"]), 0.0))
    leak = float(np.asarray(c5["nullspace_leak"], dtype=float)[0])
    cases.append(("영공간의 초점 성분", 0.9127, leak, 5e-3))

    # 초점 60 nm 조건. README 는 '초점에 따라 성분은 바뀌지만 결론(무한)은
    # 같다' 는 근거로 이 값을 인용한다. 조건이 바뀌면 숫자가 바뀌므로
    # 조건과 함께 묶어 둔다.
    c60 = ID.crb(replace(spec, defocus=60.0), params=full)
    leak60 = float(np.asarray(c60["nullspace_leak"], dtype=float)[0])
    cases.append(("영공간의 초점 성분 (초점 60 nm)", 0.971, leak60, 5e-3))
    return cases


def build_table_cases(spec):
    """README 검증 요약표의 숫자. 회귀 시험이 계산은 하지만 문서와
    대조하지는 않는 값들이다."""
    from litho import optics as OP
    from litho import process as PR
    from litho import stochastic as ST

    cases = []

    # --- 결맞음 해상 한계 -------------------------------------------------
    # validate_w6 Case 3 과 같은 호출이어야 한다. 인자가 하나라도 다르면
    # 다른 조건의 값을 비교하는 셈이 된다.
    r = PR.find_resolution_limit(
        193.0, 1.35, n_medium=1.44, sigma_out=0.0, contrast_min=0.10,
        pitch_lo=40.0, pitch_hi=500.0, n_iter=14, n_periods=6, dx_target=0.5)
    cases.append(("결맞음 half-pitch [nm]", 71.49, float(r["half_pitch"]), 1e-3))
    cases.append(("결맞음 해석해 lam/2NA [nm]", 71.48, 193.0 / (2 * 1.35), 1e-3))
    cases.append(("결맞음 k1", 0.500, float(r["k1"]), 2e-3))

    # --- k1 의 조명 의존성 ------------------------------------------------
    k1s = []
    for so, si in ((0.0, 0.0), (0.3, 0.0), (0.7, 0.0), (0.9, 0.0),
                   (0.8, 0.6), (0.9, 0.7)):
        rr = PR.find_resolution_limit(
            193.0, 1.35, n_medium=1.44, sigma_out=so, sigma_in=si,
            contrast_min=0.10, pitch_lo=40.0, pitch_hi=500.0,
            n_iter=14, n_periods=6, dx_target=0.5)
        k1s.append(float(rr["k1"]))
    cases.append(("k1 조명 최댓값", 0.500, max(k1s), 2e-3))
    cases.append(("k1 조명 최솟값", 0.278, min(k1s), 2e-3))
    # 문헌값 포함과 이론 하한은 숫자가 아니라 부등식이므로 0/1 로 본다.
    cases.append(("문헌 k1 0.288 이 구간 안", 1.0,
                  float(min(k1s) < 0.288 < max(k1s)), 0.0))
    cases.append(("이론 하한 0.25 하회하지 않음", 1.0,
                  float(min(k1s) >= 0.25), 0.0))

    # --- 광자 통계 --------------------------------------------------------
    st_arf = ST.photon_statistics(30.0, 193.0, area_nm2=4.0)
    st_euv = ST.photon_statistics(30.0, 13.5, area_nm2=4.0)
    ratio_e = st_euv["photon_energy_ev"] / st_arf["photon_energy_ev"]
    ratio_s = st_euv["relative_noise"] / st_arf["relative_noise"]
    cases.append(("광자 에너지비 EUV/ArF", 14.30, float(ratio_e), 2e-3))
    cases.append(("상대 잡음비 EUV/ArF", 3.78, float(ratio_s), 2e-3))

    # --- 이론 상수 --------------------------------------------------------
    # optics.py 의 기준 상수가 문서와 같은지. 상수 대조이므로 동어반복에
    # 가깝지만, 이 둘은 README 가 '문헌 기준' 으로 인용하므로 묶어 둔다.
    cases.append(("k1=0.25 한계 half-pitch [nm]", 35.74,
                  OP.half_pitch_limit(0.25, 193.0, 1.35), 1e-3))

    # --- 공차 예산에서 유도되는 상수 --------------------------------------
    # VERIFICATION_LIMITS 가 '2.0 을 임의로 고른 것이 아니라 예산에서
    # 유도된다' 는 근거로 이 세 값을 인용한다. 유도식을 고치면 값이
    # 바뀌므로 문서와 묶어 둔다.
    from litho import dataset as DS
    sp = DS.SampleSpec(pitch=200.0, duty=0.5)
    cases.append(("pw_tol (유도)", 0.0707, float(sp.pw_tol), 1e-3))
    cases.append(("stoch_tol (유도)", 0.0707, float(sp.stoch_tol), 1e-3))
    cases.append(("min_nils (유도)", 2.121, float(sp.min_nils), 1e-3))
    # 제곱합 검산. 두 몫의 제곱합이 전체 예산과 같아야 한다.
    cases.append(("예산 제곱합 = cd_budget^2", sp.cd_budget ** 2,
                  float(sp.pw_tol ** 2 + sp.stoch_tol ** 2), 1e-9))

    # --- 12주차 추천기 ----------------------------------------------------
    # README 와 VERIFICATION_LIMITS 가 '보수 공차 0.181 %' 를 인용한다. 그
    # 값은 유도 공차에서 11 주차 모형의 95 백분위 오차를 뺀 것이므로, 어느
    # 한쪽이 바뀌면 문서만 낡는다. 11 주차 37 번과 같은 자리이므로 산술을
    # 묶어 둔다.
    # (6.89 % 자체는 학습 실측이라 묶지 않는다. 위 '묶지 않는 숫자' 참조.)
    # 실행방법.txt 는 12 차 보완 68 번 이후 이 숫자를 인용하지 않는다.
    # 추천기가 실제로 쓴 값을 '보수 공차' 줄에 직접 인쇄하므로, 사용 설명서에
    # 박아 두면 모형을 다시 학습할 때마다 그 문서만 낡기 때문이다.
    REL_P95_DOC = 0.0689
    cases.append(("보수 공차 = pw_tol - 모형 95% 오차", 0.00181,
                  float(sp.pw_tol - REL_P95_DOC), 1e-3))
    # 목적함수를 읽는 EL 기준이 문서와 코드에서 같아야 한다.
    import recommend as RC
    cases.append(("추천기 EL 기준 [%]", 10.0,
                  float(RC.EL_TARGET_DEFAULT), 1e-9))
    cases.append(("추천기 기본 rel_p95", REL_P95_DOC,
                  float(RC.REL_P95_DEFAULT), 1e-9))

    # DOF 구간의 해석 참값. README·VERIFICATION_LIMITS 65 번·process.py
    # 주석 세 곳이 112.69 nm 를 인용한다. 시뮬레이터 측정치가 아니라
    # **닫힌 식**이므로 재계산 비용이 없고, 그래서 위 '묶지 않는 숫자' 의
    # 예외다(8.236 nm 는 시뮬레이터가 필요해 계속 제외).
    #
    #   CD/T = 1 + s(d-1) - K dz^2 에서 EL 고정 최대 DOF :
    #     u = tol/s,  l = (u(1-e/2) - e)/(1+e/2),  e = EL/100
    #     D = 2 sqrt((tol + s*l)/K)
    #
    # 왜 Case 8 로 충분하지 않은가 : Case 8 은 참값을 같은 식으로 **다시
    # 만들어** 쓴다. 그래서 식이 바뀌어도 케이스는 계속 통과하고 문서만
    # 낡는다. 문서에 적힌 숫자를 대조하는 것은 여기뿐이다.
    _s, _tol, _K, _e = 1.36, 0.07, 1.2e-6, 0.10
    _u = _tol / _s
    _l = (_u * (1.0 - _e / 2.0) - _e) / (1.0 + _e / 2.0)
    _dof_truth = 2.0 * float(np.sqrt((_tol + _s * _l) / _K))
    cases.append(("DOF 구간 해석 참값 [nm] (s=1.36, tol 7%, EL 10%)",
                  112.69, _dof_truth, 1e-2))

    # --- 광자 변환 --------------------------------------------------------
    # VERIFICATION_LIMITS 가 '손 계산과 일치(291.476 vs 291.473)' 로 인용하는
    # 값. 조건은 30 mJ/cm^2, 193 nm, 1 nm^2 이다. 면적을 바꾸면 값이 비례해
    # 달라지므로(4 nm^2 면 1165.9) 조건을 함께 적어 둔다.
    #   hc = 1239.84 eV nm, 1 mJ/cm^2 = 1e-3 J / 1e14 nm^2
    # 문서가 인용하는 값은 코드 출력 291.476 이다. 같은 줄의 291.473 은
    # 독립적인 손 계산이며, 상수의 유효숫자를 조금 달리 잡은 결과다. 그것을
    # 문서값으로 '대조' 하면 남의 반올림을 재현하려는 셈이 되어 허용오차가
    # 경계에 걸린다. 그래서 둘을 나눈다.
    #   - 문서 대조는 코드 출력에만 건다.
    #   - 코드와 손 계산의 일치는 별도 항목으로, 1e-4 허용에서 본다.
    hand = 30.0e-3 * 1.0e-14 * 1.0 / (1.602176634e-19 * 1239.84 / 193.0)
    got = float(ST.photon_statistics(30.0, 193.0, area_nm2=1.0)["n_in_area"])
    cases.append(("광자수 코드 [30 mJ, 1 nm^2]", 291.476, got, 1e-5))
    cases.append(("광자수 코드-손계산 상대차", 0.0,
                  abs(got - hand) / hand, 1e-4))
    return cases


def main() -> int:
    spec = ID.IdentifySpec()

    # 조건이 문서에 적힌 것과 같은지 먼저 본다. 이것이 어긋나면 아래 숫자는
    # 전부 다른 조건의 값이므로 비교 자체가 무의미하다.
    print("=" * 78)
    print(" README 주장 1 의 숫자 대조 (실행 시점 재계산)")
    print("=" * 78)
    print("\n  [조건] 문서에 적힌 조건이 실제 기본값과 같은지 확인한다.")
    cond_ok = True
    for k, v in CONDITION.items():
        got = getattr(spec, k, None)
        same = got is not None and abs(float(got) - v) <= 1e-12
        cond_ok &= same
        if not same:
            print(f"    {k:<12} 문서 {v} != 실제 {got}   <-- 불일치")
    print(f"    {'일치' if cond_ok else '불일치 있음'}"
          f" ({len(CONDITION)} 항목)")

    bad = 0
    n_tot = 0
    for title, cases in (("주장 1 (식별 한계)", build_cases(spec)),
                         ("검증 요약표", build_table_cases(spec))):
        print(f"\n  [{title}]")
        print(f"  {'항목':<34}{'문서':>12}{'실측':>14}{'판정':>8}")
        print("  " + "-" * 70)
        for label, want, got, tol in cases:
            err = _rel(got, want)
            # 절대 상한으로 보는 항목(해석상 0)은 tol 을 절대값으로 쓴다.
            if want == 0.0:
                ok = abs(got) <= tol
            else:
                ok = err <= tol
            bad += (not ok)
            n_tot += 1
            fw = "무한" if not np.isfinite(want) else f"{want:.6g}"
            fg = "무한" if not np.isfinite(got) else f"{got:.6g}"
            print(f"  {label:<34}{fw:>12}{fg:>14}{'ok' if ok else '불일치':>8}")

    # [13차 보완 87 번] **13 주차 숫자가 이 검사를 우회하고 있었다.**
    # claims.py 는 문서-코드 드리프트를 막기 위해 만든 것인데, 가장 최신이고
    # 가장 눈에 띄는 숫자들(계측 규격, 판정 기준, 회귀 총계)을 한 건도
    # 덮지 않았다(문자열 검색 0 건). 문서에 손으로 적은 숫자가 코드와
    # 어긋나도 회귀가 전부 통과한다.
    #
    # 전부를 재계산할 수는 없다(계측 규격은 몬테카를로가 필요하다). 그래서
    # **코드에 상수로 있는 것**과 **세어서 알 수 있는 것**을 덮는다. 재계산이
    # 필요한 값은 그 도구 자신의 회귀(validate_w13)가 담당한다.
    print("\n" + "=" * 78)
    print(" 13 주차 숫자 대조 (상수·개수 기준)")
    print("=" * 78)
    import importlib
    import roundtrip_w13 as _RT
    _docs = {}
    for _f in ("README.md", "실행방법.txt", "VERIFICATION_LIMITS.md",
               "판정기준_동결.md"):
        try:
            _docs[_f] = io.open(_f, encoding="utf-8").read()
        except OSError:
            _docs[_f] = ""
    _w13 = []
    # (가) 판정 기준 : 코드 상수와 동결 문서가 같아야 한다.
    _w13.append(("판정 기준 타우-b", 0.60, _RT.CRITERIA["tau_min"]))
    _w13.append(("판정 기준 상위3 겹침", 2.0 / 3.0,
                 _RT.CRITERIA["top3_min"]))
    _w13.append(("판정 기준 피복 여유", 0.10, _RT.CRITERIA["cover_slack"]))
    # [92 번] 검정력 요구도 동결 대상이다. 임계값만 맞춰 보고 최소 후보
    # 수를 놓치면 실제로 그랬던 것처럼 '우연선을 못 넘은 값' 이 충족으로
    # 찍힌다. n_min 은 임계값에서 유도되므로 **유도식으로** 대조한다.
    _w13.append(("판정 기준 최소 후보 수", 6, _RT.CRITERIA["n_min"]))
    _w13.append(("판정 기준 귀무 분위", 0.95, _RT.CRITERIA["null_q"]))
    _sd6 = (2.0 * (2 * 6 + 5) / (9.0 * 6 * 5)) ** 0.5
    _w13.append(("n_min 유도 : 1.645*sd(6) <= 타우 기준",
                 True, bool(1.645 * _sd6 <= _RT.CRITERIA["tau_min"] + 1e-9)))
    _sd5 = (2.0 * (2 * 5 + 5) / (9.0 * 5 * 4)) ** 0.5
    _w13.append(("n_min 유도 : 1.645*sd(5) > 타우 기준",
                 True, bool(1.645 * _sd5 > _RT.CRITERIA["tau_min"])))
    # (나) 회귀 총계 : 문서 세 곳이 같은 수를 말해야 한다. 실제 수는
    #      run_all_checks 가 세므로 여기서는 **문서 사이 일치**만 본다.
    # **기계가 읽을 표식**으로만 센다. 산문에서 숫자를 긁으면 과거 기록을
    # 현재 값으로 오인한다(실측 : VERIFICATION_LIMITS 의 '회귀 시험 50 종'
    # 은 그 주차의 이력이고 드리프트가 아닌데 불일치로 잡혔다). 이력은
    # 지우면 안 되므로, 현재 값에만 표식을 단다.
    import re as _re
    _nums = {}
    for _f in ("README.md", "실행방법.txt", "VERIFICATION_LIMITS.md"):
        _nums[_f] = sorted({int(x) for x in _re.findall(
            r"REGRESSION_TOTAL=(\d+)", _docs[_f])})
    _missing = [f for f, v in _nums.items() if not v]
    if _missing:
        print(f"    -> 표식이 없는 문서 : {_missing}")
        print("       'REGRESSION_TOTAL=<수>' 를 현재 총계 옆에 달 것.")
    _all = sorted({n for v in _nums.values() for n in v})
    _w13.append(("문서 세 곳의 회귀 총계가 한 값", 1.0, float(len(_all))))
    print(f"\n  {'항목':<34}{'문서':>12}{'코드':>14}{'판정':>8}")
    print("  " + "-" * 68)
    for _lab, _want, _got in _w13:
        _ok = abs(float(_got) - float(_want)) <= 1e-9
        bad += (not _ok)
        n_tot += 1
        print(f"  {_lab:<34}{_want:>12.6g}{_got:>14.6g}"
              f"{'ok' if _ok else '불일치':>8}")
    if len(_all) != 1:
        print(f"    -> 문서에 적힌 회귀 총계가 여러 값이다 : {_all}")
        print("       README·실행방법·VERIFICATION_LIMITS 를 한 값으로 맞출 것.")
    # (다) 동결 문서가 코드 상수와 같은 값을 **글자로** 담고 있는지.
    for _tok, _lab in (("0.60", "타우 기준"), ("2/3", "상위3 기준"),
                       ("0.10", "피복 여유"),
                       ("최소 후보 수 — **6 개**", "최소 후보 수"),
                       ("순열 귀무분포 대조", "귀무분포 대조"),
                       ("2026-11-11", "92 번 변경 기록")):
        _ok = _tok in _docs["판정기준_동결.md"]
        bad += (not _ok)
        n_tot += 1
        print(f"  {'판정기준_동결.md 에 ' + _lab:<34}{_tok:>12}"
              f"{'있음' if _ok else '없음':>14}{'ok' if _ok else '불일치':>8}")

    print()
    if bad or not cond_ok:
        print(f"  -> 불일치 {bad} 건"
              f"{', 조건 불일치' if not cond_ok else ''}.")
        print("     README 와 VERIFICATION_LIMITS 의 해당 숫자를 고쳐야 한다.")
        print("     문서는 어떤 시험도 읽지 않으므로 회귀가 전부 통과해도")
        print("     이 드리프트는 저절로 드러나지 않는다.")
        return 1
    print(f"  -> {n_tot} 항목 전부 일치. 문서의 대표 주장이 현재 코드와")
    print("     같은 숫자를 말한다.")
    print()
    print("  [범위] 이것은 문서-코드 정합성 검사다. 숫자가 물리적으로 옳다는")
    print("  것을 보이는 것이 아니다. 그것은 verify.py 와 validate*.py 가")
    print("  담당한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
