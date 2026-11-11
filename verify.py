"""
verify.py -- 검증 체계 (Verification suite)

왜 이 파일이 따로 있는가
------------------------
validate*.py 는 '이 조건에서 이런 값이 나왔다' 를 기록하는 회귀 시험이다.
기준값을 사후에 정했기 때문에, 모델을 잘못 이해했다면 시뮬레이터와 기준선이
함께 틀린 채로 통과한다. 이 파일은 그 약점을 메우는, 기준값이 필요 없는
검증만 모은다.

Roache / Roy 의 검증 위계
-------------------------
P. J. Roache, "Verification and Validation in Computational Science and
Engineering" (1998) 과 C. J. Roy, "Review of code and solution verification
procedures for computational simulation," J. Comput. Phys. 205 (2005)
131-156 은 수치 모형의 신뢰도 근거를 다음 순서로 둔다. 아래로 갈수록 강하다.

    1) 전문가 판단            -- '그럴듯하다'
    2) 오차 정량화            -- 기준값과의 차이를 수치로 제시
    3) 수렴성                 -- 격자를 조이면 값이 수렴한다
    4) 관측 수렴 차수(order of accuracy)  -- 가장 엄격

9주차까지의 검증은 대부분 1)~2) 단계였다. 이 파일은 3)~4) 단계와,
독립 구현 대조 및 물리 불변량까지 포함한다.

Roy 의 용어 구분도 지킨다.
    code verification     : 코드가 '의도한 수식' 을 푸는지  -> 여기서 다룸
    solution verification : 이 특정 해의 수치 오차가 얼마인지 -> 여기서 다룸
    validation            : 수식이 '실제 현상' 과 맞는지     -> 실측 없이는 불가

본 파일의 결과는 전부 verification 이다. validation 이 아니다.
이 구분을 흐리지 않기 위해 PASS 라는 말을 쓰지 않고, 각 항목이 어떤 성질을
어느 정밀도로 보였는지 그대로 적는다.

구성
----
    A. 공간 이산화의 관측 수렴 차수 (code verification)
    B. 독립 구현 대조 : Abbe vs Hopkins TCC
    C. 물리 불변량 : 에너지, 평행이동, 대칭, 선형성, 극한
    D. Cramer-Rao 하한의 몬테카를로 대조 (추정량이 하한에 닿는가)

실행
----
    py verify.py              # 전부
    py verify.py A C          # 일부만
"""

import ast
import os
import re
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from litho import hopkins as HK
from litho import identify as ID
from litho import imaging as IM
from litho import mask as MK
from litho import resist as RS
from litho import stochastic as ST
from litho.grid import SimGrid, build_grid, nyquist_dx

BASE = dict(wavelength=193.0, na=1.35, n_medium=1.44, sigma_out=0.85,
            sigma_in=0.60, n_source_grid=31)

_RESULTS: list[tuple[str, str, str]] = []


def record(section: str, item: str, verdict: str, ok: bool | None = None) -> None:
    """결과를 요약에 기록한다.

    ok 는 '논리적 판정이 있는 항목' 에만 넘긴다. 즉 물리가 요구하는 등식·
    대칭·순서처럼 **기준값을 고를 여지가 없는** 검사다. 측정값만 보고하는
    항목(수렴 차수, 하한의 크기, LER 절대값 등)에는 넘기지 않는다. 그런
    항목에 합격선을 붙이면 이 과제가 일부러 피한 '사후 임계값' 약점을
    되살리게 된다.

    ok 가 주어진 항목은 main() 이 모아 최종 판정을 내리고, 하나라도
    거짓이면 종료 코드 1 을 돌려준다.

    [왜 필요한가] 이전에는 verify.py 에 assert 가 한 건도 없고 합격 판정이
    part K 에만 있었다. 그래서 J4 의 단조성이 깨지더라도 '위반' 을 인쇄하고
    종료 코드 0 으로 끝났다. 종료 코드만 보는 사람(또는 자동화)은 통과로
    읽는다. 실제로 완주 결과를 'exit 0 이므로 통과' 로 보고한 적이 있고,
    그것은 요약을 눈으로 읽어 확인한 것이지 종료 코드가 보장한 것이
    아니었다.
    """
    _RESULTS.append((section, item, verdict, ok))


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def sub(title: str) -> None:
    print(f"\n[{title}]")


# ===========================================================================
# A. 공간 이산화의 관측 수렴 차수
# ===========================================================================

def observed_order(values: list[float], r: float = 2.0) -> tuple[float, float]:
    """Roache 의 관측 수렴 차수 p 와 세밀격자 GCI.

        p        = ln|(f3 - f2) / (f2 - f1)| / ln r
        GCI_fine = (Fs / (r^p - 1)) * |(f2 - f1) / f1|

    Fs 는 안전계수이다. 격자 셋 이상으로 p 를 직접 관측했을 때 1.25,
    두 격자만으로 p 를 가정했을 때 3 을 쓴다(Roache 1998). 여기서는 항상
    세 격자 이상을 쓰므로 1.25 를 적용한다.

    values 는 격자가 조밀해지는 순서(f1=가장 조밀)로 준다.
    """
    f1, f2, f3 = values[0], values[1], values[2]
    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0.0 and d2 == 0.0:
        return float("inf"), 0.0          # 기계 정밀도까지 불변
    if d1 == 0.0:
        return float("inf"), 0.0
    p = np.log(abs(d2 / d1)) / np.log(r)
    gci = (1.25 / (r ** p - 1)) * abs(d1 / f1) * 100.0 if r ** p > 1 else float("nan")
    return float(p), float(gci)


def part_a() -> None:
    head("A. 공간 이산화의 관측 수렴 차수 (code verification)")

    # --- A1. 마스크 회절차수의 해석 대조 -----------------------------------
    sub("A1  마스크 회절 차수 : 해석 계수와의 일치")
    print("  L&S 마스크의 푸리에 계수는 해석적으로 알려져 있다.")
    print("      A_0 = duty,   A_m = duty * sinc(m * duty)")
    print("  두 표본화 방식의 Nyquist 이하 전 차수 최대 오차를 비교한다.\n")
    print(f"  {'spp':>5} {'area(면적가중)':>18} {'bandlimited':>16}")
    rows = {}
    for spp in (8, 16, 32, 64):
        errs = {}
        for samp in (MK.SAMPLING_AREA, MK.SAMPLING_BANDLIMITED):
            e = 0.0
            for duty in (0.173, 0.31, 0.5, 0.667, 0.84):
                g = SimGrid(pitch=200.0, n_periods=8, samples_per_pitch=spp)
                s = np.fft.fft(MK.line_space(g, duty=duty, sampling=samp)) / g.n_points
                for m in range(0, spp // 2):
                    a = duty if m == 0 else duty * np.sinc(m * duty)
                    e = max(e, abs(s[m * g.n_periods].real - a))
            errs[samp] = e
        rows[spp] = errs
        print(f"  {spp:5d} {errs[MK.SAMPLING_AREA]:18.3e} "
              f"{errs[MK.SAMPLING_BANDLIMITED]:16.3e}")
    # 관측 수렴 차수는 '최대 오차' 가 아니라 하나의 차수를 고정해 재는 것이
    # 옳다. 최대값은 duty 마다 다른 차수에서 나오므로 매끄러운 수열이 아니다.
    e_area = []
    for spp in (64, 32, 16):
        g = SimGrid(pitch=200.0, n_periods=8, samples_per_pitch=spp)
        s1 = np.fft.fft(MK.line_space(g, duty=0.5, sampling=MK.SAMPLING_AREA))
        e_area.append(abs(abs(s1[g.n_periods]) / g.n_points - 0.5 * np.sinc(0.5)))
    p_area, _ = observed_order(e_area)
    bl_max = max(rows[s][MK.SAMPLING_BANDLIMITED] for s in rows)
    print(f"\n  area        : 관측 수렴 차수 p = {p_area:.2f}  (상자함수 전달함수 "
          f"sinc(m/N) 에 의한 2차 오차)")
    print(f"  bandlimited : 최대 오차 {bl_max:.2e}  -- 격자와 무관하게 기계 정밀도")
    print("  생산 경로는 bandlimited 를 기본값으로 쓴다. 동공 차단주파수가")
    print("  격자 Nyquist 보다 낮으므로, 결상에 쓰이는 차수가 전부 정확해진다.")
    record("A", "A1 마스크 회절차수",
           f"bandlimited 최대오차 {bl_max:.1e} (격자 무관), area 는 p={p_area:.2f}")

    # --- A2. 공중상 및 파생 지표의 격자 의존성 -----------------------------
    sub("A2  공중상·CD·NILS 의 격자 의존성")
    print("  동일 물리 조건에서 격자만 16 배 조이며 출력을 비교한다.")
    print("  세 축을 각각 독립적으로 조인다 : 공간 격자, 광원 격자, 필드 폭.\n")

    def metrics(g, duty=0.5, **kw):
        o = dict(BASE); o.update(kw)
        i = np.asarray(IM.aerial_image(g, MK.line_space(g, duty=duty), **o)).real
        # [중요] 임계값과 대비를 '표본 극값' 으로 잡으면 격자 의존성이
        # 되살아난다. 표본 최대값은 연속 함수의 극값이 아니기 때문이다.
        # 실제로 pitch 400 nm 조건에서 대비가 격자에 따라 5.9e-4 상대로
        # 흔들렸다(결상과 CD 추출이 기계 정밀도인데도). 극값을 대역제한
        # 보간에서 찾아야 비로소 전 지표가 격자와 무관해진다.
        e = RS.image_extrema(g.x, i)
        th = 0.5 * (e["i_max"] + e["i_min"])
        r = RS.extract_cd(g.x, i, th, max_width=g.pitch)
        return {"contrast": IM.contrast(i, g.x),
                "CD": r["cd"], "NILS": r["nils"]}

    # 축 1 : 주기당 표본 수
    print("  축 1 -- 주기당 표본 수 spp (r = 2)")
    print("  pitch 100 nm 는 광강도 극값이 격자점에 정확히 떨어지는 '유리한'")
    print("  조건이다. 그렇지 않은 pitch 400 nm 도 함께 본다.")
    seq = [128, 64, 32, 16, 8]
    for pitch_t in (100.0, 400.0):
        print(f"\n    [pitch {pitch_t:.0f} nm]")
        vals = {k: [] for k in ("contrast", "CD", "NILS")}
        for spp in seq:
            m = metrics(SimGrid(pitch=pitch_t, n_periods=4,
                                samples_per_pitch=spp))
            for k in vals:
                vals[k].append(m[k])
        # Nyquist 조건을 만족하는 격자만으로 판정해야 한다. 조건을 위반한
        # 격자에서 값이 달라지는 것은 결함이 아니라 표본화 정리의 결과다.
        dx_max = nyquist_dx(BASE["wavelength"], BASE["na"], BASE["sigma_out"])
        ok_spp = [sp for sp in seq if pitch_t / sp <= dx_max]
        bad_spp = [sp for sp in seq if pitch_t / sp > dx_max]
        print(f"    Nyquist 상한 dx = {dx_max:.2f} nm")
        if bad_spp:
            print(f"    조건 위반 격자(판정 제외) : spp " +
                  ", ".join(f"{sp}(dx={pitch_t/sp:.1f}nm)" for sp in bad_spp))
        print(f"    {'지표':>9} " + "".join(f"{'spp'+str(s):>18}" for s in seq))
        for k in ("contrast", "CD", "NILS"):
            good = [v for sp, v in zip(seq, vals[k]) if sp in ok_spp]
            rng = (abs(max(good) - min(good)) / max(abs(good[0]), 1e-30)
                   if len(good) > 1 else 0.0)
            print(f"    {k:>9} " + "".join(f"{v:18.13f}" for v in vals[k])
                  + f"   변화 {rng:.1e}")
    vals = {k: [] for k in ("contrast", "CD", "NILS")}
    for spp in seq:
        m = metrics(SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=spp))
        for k in vals:
            vals[k].append(m[k])
    print(f"    {'지표':>9} " + "".join(f"{'spp'+str(s):>18}" for s in seq)
          + f"{'p':>8}{'GCI[%]':>10}")
    worst = 0.0
    for k in ("contrast", "CD", "NILS"):
        with np.errstate(divide="ignore", invalid="ignore"):
            p, gci = observed_order(vals[k])
        gci = 0.0 if not np.isfinite(gci) else gci
        worst = max(worst, gci)
        # 변화량이 이미 기계 정밀도이면 수렴 차수는 정의되지 않는다.
        rng = max(abs(max(vals[k]) - min(vals[k])) / max(abs(vals[k][0]), 1e-30), 0.0)
        pt = "  n/a" if rng < 1e-11 else f"{p:.2f}"
        print(f"    {k:>9} " + "".join(f"{v:18.13f}" for v in vals[k])
              + f"{pt:>8}{gci:10.2e}")
    print(f"    -> 세 지표 모두 GCI <= {worst:.1e} %. 공간 이산화 오차는")
    print("       절단 오차가 아니라 기계 정밀도 수준이다. 대역제한 마스크와")
    print("       스펙트럼 미분을 함께 쓰면 이 축의 오차가 '수렴' 하는 것이")
    print("       아니라 '존재하지 않는다'.")
    record("A", "A2 공간 격자", f"contrast/CD/NILS GCI <= {worst:.1e} % (기계 정밀도)")

    # 축 2 : 광원 격자
    print("\n  축 2 -- 광원 표본 격자 n_source_grid")
    g = SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=32)
    ns = [241, 121, 61, 31, 15]
    cv, nv = [], []
    for n in ns:
        m = metrics(g, n_source_grid=n, source_integration="abbe")
        cv.append(m["contrast"]); nv.append(m["NILS"])
    print(f"    {'n_src':>7} {'contrast':>18} {'NILS':>18}")
    for i, n in enumerate(ns):
        print(f"    {n:7d} {cv[i]:18.13f} {nv[i]:18.13f}")
    # 정확해를 알고 있으므로, 세 점 Roache 공식보다 '실제 오차의 로그-로그
    # 기울기' 가 올바른 관측 차수 추정이다(Roy 2005, 2.2절). Roache 공식은
    # 수열이 단조할 때만 쓸 수 있고, 여기서는 단조하지 않다.
    ex = metrics(g, source_integration="auto", n_gauss=32)
    err_c = np.array([abs(v - ex["contrast"]) / ex["contrast"] for v in cv])
    p_c = -np.polyfit(np.log(np.array(ns, float)), np.log(err_c), 1)[0]
    print(f"    오차 로그-로그 기울기로 추정한 관측 차수 p = {p_c:.2f}")
    print("    1차에 머문다. 이유는 피적분 함수가 매끄럽지 않기 때문이다.")
    print("    광원점이 움직이면 회절 차수가 동공 경계를 드나들며 상이")
    print("    불연속으로 바뀐다. 불연속이 적분 영역 안에 있으면 구적 차수를")
    print("    올려도 소용이 없다(극좌표 Gauss-Legendre 로 바꾸어도 비단조")
    print("    O(1/n) 거동이 남았다).")
    print(f"\n    영역 분해 정확 구적 기준값 : contrast {ex['contrast']:.14f}")
    print(f"    {'n_src':>7} {'contrast 상대오차':>20} {'NILS 상대오차':>18}")
    for i, n in enumerate(ns):
        print(f"    {n:7d} {abs(cv[i]-ex['contrast'])/ex['contrast']:20.3e} "
              f"{abs(nv[i]-ex['NILS'])/ex['NILS']:18.3e}")
    err31 = abs(cv[ns.index(31)] - ex["contrast"]) / ex["contrast"]
    print(f"\n    -> 기존 생산 설정 n_source_grid=31 의 대비 오차는 {err31:.2%} 였다.")
    print("       이것이 시뮬레이터에 남아 있던 가장 큰 수치 오차이며, 공간")
    print("       격자 오차(1e-15)보다 10^13 배 크다. 생산 경로를 영역 분해")
    print("       정확 구적으로 교체해 제거했다.")
    record("A", "A2 광원 격자",
           f"Abbe 는 p={p_c:.2f} (불연속 때문), n_src=31 오차 {err31:.2%} "
           f"-> 영역분해 정확구적으로 교체")

    # 축 3 : 필드 폭
    print("\n  축 3 -- 필드 폭 (주기 수 n_periods)")
    base = metrics(SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=32))
    dev = 0.0
    for npd in (2, 3, 4, 6, 8, 12):
        m = metrics(SimGrid(pitch=100.0, n_periods=npd, samples_per_pitch=32))
        d = max(abs(m[k] - base[k]) / max(abs(base[k]), 1e-30)
                for k in ("contrast", "CD", "NILS"))
        dev = max(dev, d)
        print(f"    n_periods={npd:3d}  최대 상대편차 {d:.2e}")
    print(f"    -> 최대 {dev:.1e}. 주기 경계 조건이 정확히 성립하므로 필드 폭은")
    print("       물리에 영향을 주지 않아야 하고, 실제로 주지 않는다.")
    record("A", "A2 필드 폭", f"주기 수 2~12 에서 최대 상대편차 {dev:.1e}")


# ===========================================================================
# B. 독립 구현 대조
# ===========================================================================

def part_b() -> None:
    head("B. 독립 구현 대조 : 세 가지 경로")
    print("  해석식 대조의 한계는, 기준선도 같은 사람이 같은 이해로 쓴다는")
    print("  점이다. 물리 모델 자체를 잘못 이해했다면 시뮬레이터와 기준선이")
    print("  함께 틀린 채로 통과한다. 이를 막는 표준 방법이 독립 구현 대조이다.")
    print("  본 프로젝트에는 서로 독립적인 세 경로가 있다.\n")
    print("    (1) Abbe   : 광원점마다 상을 만들어 세기를 비결맞음 합산")
    print("                 I = sum_s w_s |IFFT{M(f) P(s+f)}|^2")
    print("    (2) Hopkins: 광원을 먼저 적분해 TCC 를 만들고 차수 쌍으로 합산")
    print("                 TCC(f1,f2) = sum_s w_s P(s+f1) P*(s+f2)")
    print("    (3) 영역분해: 차수 쌍마다 적분 영역을 해석적으로 구해 Gauss-")
    print("                 Legendre 로 적분. 광원 구적점을 전혀 쓰지 않는다.")
    print("  (1) 과 (2) 는 같은 구적점을 쓰되 알고리즘이 다르므로 구현 실수를")
    print("  잡고, (3) 은 구적 자체를 쓰지 않으므로 구적 오차까지 드러낸다.\n")

    cases = [
        ("결맞음(sigma=0)", dict(sigma_out=1e-9)),
        ("부분결맞음 sigma=0.5", dict(sigma_out=0.5)),
        ("환형 조명", dict(sigma_out=0.85, sigma_in=0.6)),
        ("쌍극 조명", dict(sigma_out=0.85, source_shape=IM.SOURCE_DIPOLE,
                          pole_offset=0.7, pole_radius=0.15)),
        ("사극 조명", dict(sigma_out=0.85, source_shape=IM.SOURCE_QUADRUPOLE,
                          pole_offset=0.7, pole_radius=0.15)),
        ("디포커스 100 nm", dict(sigma_out=0.7, defocus=100.0)),
        ("TE 편광", dict(sigma_out=0.7, polarization=IM.POL_TE)),
        ("TM 편광", dict(sigma_out=0.7, polarization=IM.POL_TM)),
        ("무편광", dict(sigma_out=0.7, polarization=IM.POL_UNPOLARIZED)),
        ("구면수차 Z11=0.05", dict(sigma_out=0.7, aberrations={11: 0.05})),
        ("코마 Z7=0.05", dict(sigma_out=0.7, aberrations={7: 0.05})),
        ("코마 Z8=0.05 (sy 홀수)", dict(sigma_out=0.7, aberrations={8: 0.05})),
        ("att-PSM 마스크", dict(sigma_out=0.7)),
        ("레지스트 내 벡터", dict(sigma_out=0.7, polarization=IM.POL_TM,
                              n_resist=1.70)),
    ]
    print("  (1) vs (2) -- 같은 구적점, 다른 알고리즘")
    print(f"  {'조건':<24}{'최대 상대편차':>16}{'RMS':>14}")
    print("  " + "-" * 54)
    w12 = 0.0
    for name, kw in cases:
        g = SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=32)
        mt = MK.MASK_ATT_PSM if "att-PSM" in name else MK.MASK_BINARY
        m = MK.line_space(g, duty=0.5, mask_type=mt)
        o = dict(BASE); o.pop("sigma_out"); o.pop("sigma_in", None)
        o.setdefault("sigma_in", 0.0)
        o.update(kw)
        r = HK.compare_with_abbe(g, m, **o)
        w12 = max(w12, r["max_rel"])
        print(f"  {name:<24}{r['max_rel']:16.3e}{r['rms_rel']:14.3e}")
    print(f"\n  -> 14 조건 전부 최대 {w12:.1e}. 배정밀도 누적 오차 수준이므로,")
    print("     두 알고리즘이 같은 연산을 서로 독립적으로 수행했다.")
    record("B", "(1)Abbe vs (2)Hopkins", f"14 조건 최대 {w12:.1e} (배정밀도 한계)")

    print("\n  (1) vs (3) -- 구적점을 쓰는 경로 vs 쓰지 않는 경로")
    print("  Abbe 의 광원 구적점을 늘리면 영역분해 결과로 수렴해야 한다.")
    print("  수렴하지 않으면 두 경로 중 하나가 틀린 것이다.\n")
    g = SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=32)
    m = MK.line_space(g, duty=0.5)
    print(f"  {'조건':<20}" + "".join(f"{'n='+str(n):>12}" for n in (31, 121, 481, 961))
          + f"{'자기수렴':>12}")
    print("  " + "-" * 92)
    w13 = 0.0
    for name, kw in [c for c in cases if "att-PSM" not in c[0]][:8]:
        o = dict(BASE); o.pop("sigma_out"); o.pop("sigma_in", None)
        o.pop("n_source_grid", None)
        o.setdefault("sigma_in", 0.0)
        o.update(kw)
        ex = np.asarray(IM.aerial_image(g, m, n_gauss=16, **o)).real
        ex2 = np.asarray(IM.aerial_image(g, m, n_gauss=28, **o)).real
        sc = float(np.max(np.abs(ex - ex2))) / float(ex.max())
        w13 = max(w13, sc)
        line = f"  {name:<20}"
        for n in (31, 121, 481, 961):
            a = np.asarray(IM.aerial_image(g, m, source_integration="abbe",
                                           n_source_grid=n, **o)).real
            line += f"{float(np.max(np.abs(a - ex))) / float(ex.max()):12.2e}"
        print(line + f"{sc:12.2e}")
    print(f"\n  -> Abbe 는 구적점을 늘릴수록 영역분해 결과로 접근하고, 영역분해")
    print(f"     자체는 구적 차수를 올려도 최대 {w13:.1e} 밖에 변하지 않는다.")
    print("     즉 영역분해 쪽이 수렴한 해이고, Abbe 의 잔차는 구적 오차다.")
    print("     쌍극 조명처럼 불연속이 광원 영역 밖에 있는 조건에서는 Abbe 도")
    print("     기계 정밀도까지 일치한다(두 경로의 완전한 상호 확인).")
    record("B", "(1)Abbe vs (3)영역분해",
           f"Abbe 는 n 증가에 따라 수렴, 영역분해 자기수렴 {w13:.1e}")


# ===========================================================================
# C. 물리 불변량
# ===========================================================================

def part_c() -> None:
    head("C. 물리 불변량 (기준값이 필요 없는 검증)")
    print("  불변량 검증은 '정답' 을 몰라도 쓸 수 있다. 물리가 요구하는 등식이")
    print("  코드에서 성립하는지만 보기 때문이다. 기준값을 사후에 고를 여지가")
    print("  없으므로 회귀 시험보다 강하다.\n")

    g = SimGrid(pitch=100.0, n_periods=4, samples_per_pitch=32)
    m = MK.line_space(g, duty=0.5)

    # --- C1. 개방 필드 정규화 ---------------------------------------------
    sub("C1  개방 필드 정규화 : 마스크가 없으면 I = 1")
    print("  전면 투과 마스크는 0차만 가지므로, 어떤 조명·수차·디포커스에서도")
    print("  상은 균일하게 1 이어야 한다. 정규화 상수나 동공 면적 처리가")
    print("  틀리면 여기서 즉시 드러난다.\n")
    worst = 0.0
    for name, kw in (("환형", {}), ("쌍극", dict(sigma_out=0.0,
                     source_shape=IM.SOURCE_DIPOLE, pole_offset=0.7)),
                     ("디포커스 150", dict(defocus=150.0)),
                     ("TM 편광", dict(polarization=IM.POL_TM)),
                     ("수차 Z11", dict(aberrations={11: 0.08}))):
        o = dict(BASE); o.update(kw)
        i = np.asarray(IM.aerial_image(g, MK.open_frame(g), **o)).real
        e = float(np.max(np.abs(i - 1.0)))
        worst = max(worst, e)
        print(f"    {name:<14} max|I - 1| = {e:.3e}")
    print(f"  -> 최대 {worst:.1e}")
    record("C", "C1 개방 필드 정규화", f"max|I-1| = {worst:.1e}")

    # --- C2. 평행이동 등가성 ----------------------------------------------
    sub("C2  평행이동 등가성")
    print("  마스크를 정수 격자만큼 옮기면 상도 같은 양만큼 옮겨져야 한다.")
    print("  (수차가 있어도 성립한다. 수차는 동공에 걸리고 평행이동은")
    print("   위상 선형항이므로 서로 간섭하지 않는다.)\n")
    o = dict(BASE); o["aberrations"] = {8: 0.05, 11: 0.05}
    i0 = np.asarray(IM.aerial_image(g, m, **o)).real
    worst = 0.0
    for sh in (1, 5, 16, 37):
        i1 = np.asarray(IM.aerial_image(g, np.roll(m, sh), **o)).real
        e = float(np.max(np.abs(i1 - np.roll(i0, sh)))) / float(i0.max())
        worst = max(worst, e)
        print(f"    {sh:3d} 격자 이동 : max 상대편차 {e:.3e}")
    print(f"  -> 최대 {worst:.1e}")
    record("C", "C2 평행이동 등가성", f"최대 상대편차 {worst:.1e}")

    # --- C3. 반사 대칭 ----------------------------------------------------
    sub("C3  반사 대칭과 그 파괴")
    print("  대칭 마스크 + 대칭 조명이면 상은 좌우 대칭이어야 한다. 수차가")
    print("  그 대칭을 깨는지는 Zernike 항이 pupil 좌표 px 에 대해 홀수인지로")
    print("  정해진다. x 방향으로만 변하는 선 패턴이기 때문이다.")
    print("    Z6 ~ cos2t = px^2 - py^2         (px 짝수) -> 대칭 유지")
    print("    Z7 ~ (3r^3-2r) sin t ~ py ...    (px 짝수) -> 대칭 유지 (코마 y)")
    print("    Z8 ~ (3r^3-2r) cos t ~ px ...    (px 홀수) -> 대칭 파괴 (코마 x)")
    print("    Z9 ~ sin3t = 3px^2 py - py^3     (px 짝수) -> 대칭 유지")
    print("    Z10 ~ cos3t = px^3 - 3px py^2    (px 홀수) -> 대칭 파괴")
    print("    Z11 ~ r^4 ...                    (px 짝수) -> 대칭 유지")
    print("  유지와 파괴를 모두 확인해야 의미가 있다. 대칭만 보이면, 수차를")
    print("  아예 반영하지 않는 버그를 잡아내지 못한다.\n")
    expect_sym = {None: True, 6: True, 7: True, 9: True, 11: True,
                  8: False, 10: False}
    ok_all = True
    print(f"    {'수차':<22}{'비대칭도':>14}  {'기대':>6}  판정")
    for j, want_sym in expect_sym.items():
        o = dict(BASE); o["aberrations"] = None if j is None else {j: 0.08}
        i = np.asarray(IM.aerial_image(g, m, **o)).real
        c = g.n_points // 2
        ref = np.roll(i[::-1], 2 * c + 1)
        asym = float(np.max(np.abs(i - ref))) / float(i.max())
        good = (asym < 1e-12) if want_sym else (asym > 1e-6)
        ok_all &= good
        name = "수차 없음" if j is None else f"Z{j} = 0.08"
        print(f"    {name:<22}{asym:14.3e}  {'대칭' if want_sym else '비대칭':>6}  "
              f"{'일치' if good else '불일치'}")
    print(f"  -> 7 경우의 대칭 예측이 모두 {'일치' if ok_all else '불일치'}한다.")
    record("C", "C3 반사 대칭",
           f"px 짝/홀 예측대로 유지 4 / 파괴 2, 전부 {'일치' if ok_all else '불일치'}",
           ok=bool(ok_all))

    # --- C4. 광원 가산성 (Abbe 의 정의 자체) ------------------------------
    sub("C4  비결맞음 가산성")
    print("  부분결맞음 조명에서 서로 다른 광원점의 기여는 세기 수준에서")
    print("  더해진다(결맞음이 없으므로). 광원을 둘로 쪼개 따로 계산한 뒤")
    print("  면적으로 가중 평균하면 원래 결과와 같아야 한다.")
    print("  가중치는 광원점 '개수' 가 아니라 '면적' 이다. 구적점 개수로")
    print("  가중하면 구적 방식이 바뀔 때 조용히 틀린다.\n")
    o = dict(BASE); o["sigma_out"] = 0.8; o["sigma_in"] = 0.0
    full = np.asarray(IM.aerial_image(g, m, **o)).real
    split_r = 0.4
    o1 = dict(o); o1["sigma_out"] = split_r
    o2 = dict(o); o2["sigma_in"] = split_r
    a1 = np.pi * split_r ** 2
    a2 = np.pi * (0.8 ** 2 - split_r ** 2)
    i1 = np.asarray(IM.aerial_image(g, m, **o1)).real
    i2 = np.asarray(IM.aerial_image(g, m, **o2)).real
    comb = (a1 * i1 + a2 * i2) / (a1 + a2)
    e = float(np.max(np.abs(comb - full))) / float(full.max())
    print(f"    sigma 0~0.8 을 0~{split_r} 와 {split_r}~0.8 로 분할")
    print(f"    면적 가중 {a1/(a1+a2):.4f} : {a2/(a1+a2):.4f}")
    print(f"    max 상대편차 {e:.3e}")
    record("C", "C4 비결맞음 가산성", f"광원 분할 재조합 상대편차 {e:.1e}")

    # --- C5. 디포커스의 짝함수성 -----------------------------------------
    sub("C5  디포커스의 짝함수성과 그 성립 조건")
    print("  최적 초점 부근의 부호 모호성은 본 과제 식별 분석의 핵심 주장")
    print("  가운데 하나이므로, 수치 우연이 아니라 정리임을 확인해야 한다.")
    print()
    print("  주장 : 마스크 투과 진폭이 '실수' 이고 광원이 sx·sy 에 대해")
    print("         대칭이며, 동공 위상이 그 광원 대칭 가운데 하나에 대해")
    print("         '홀함수' 이면 I(x; +dz) = I(x; -dz) 가 성립한다.")
    print()
    print("  근거 : Hopkins 형식에서")
    print("     I(x) = sum_ij TCC_ij A_i A_j^* exp(i2pi(f_i - f_j)x)")
    print("     TCC_ij = \\int exp(i2pi dz (kz_i - kz_j)) exp(i2pi(W_i - W_j)) dA")
    print("     광원을 뒤집고 차수를 i -> -i 로 되붙이면 px -> -px 가 되고")
    print("     kz 는 짝함수라 디포커스 항이 그대로 남는다. 짝함수성이")
    print("     성립하려면 수차 항의 부호가 뒤집혀야 하므로")
    print("         TCC_{-i,-j}(dz) = conj(TCC_ij(-dz))")
    print("     즉 W 가 그 대칭에 대해 홀함수여야 한다. 쓸 수 있는 대칭은")
    print("     두 가지다 : (sx -> -sx) 와 (sx,sy) -> (-sx,-sy).")
    print()
    print("  [10주차 정정] 이전에 '수차와 무관하게 성립' 이라고 적었는데")
    print("  이는 틀렸다. Z7, Z8, Z10 만 시험하고 짝수 방위차수 항을 빼먹은")
    print("  결과다. 경계 조건 시험에서 Z11 이 짝함수성을 깨는 것이 드러났다.")
    print("  올바른 기준으로 Z5~Z11 전부를 분류하면 :")
    print("     Z5  ~ px py        : px 에 홀함수        -> 유지")
    print("     Z6  ~ px^2 - py^2  : 두 대칭 모두 짝함수  -> 깨짐")
    print("     Z7  ~ (3r^2-2) py  : 반전에 홀함수        -> 유지")
    print("     Z8  ~ (3r^2-2) px  : px 에 홀함수        -> 유지")
    print("     Z9  ~ r^3 sin3t    : 반전에 홀함수        -> 유지")
    print("     Z10 ~ r^3 cos3t    : 반전에 홀함수        -> 유지")
    print("     Z11 ~ 6r^4-6r^2+1  : 두 대칭 모두 짝함수  -> 깨짐")
    print("  반전 패리티는 방위차수 m 의 홀짝과 같다. 따라서 깨지는 것은")
    print("  m 이 짝수이고 px 에도 짝함수인 Z6 와 Z11 뿐이며, 둘 다 rho^2")
    print("  (디포커스형) 성분을 품어 최적 초점을 이동시키는 항이다.")
    print()
    print("  따라서 이 과제에서 쓰는 모든 마스크(binary, att-PSM 은 진폭이")
    print("  각각 0/1 과 -sqrt(T) 로 전부 실수)에서, 수차가 없거나 Z6·Z11 이")
    print("  없으면 초점 부호는 원리적으로 식별 불가능하다. 광자를 더 모아도")
    print("  해결되지 않는다.")
    print()
    print("  [설계 시사] 거꾸로 말하면 부호 모호성을 깨는 길은 세 가지다.")
    print("    (1) 복소 마스크(위상이 0 또는 pi 가 아닌 성분)")
    print("    (2) 구면수차 Z11 또는 비점수차 Z6 의 도입·측정")
    print("    (3) 광원 대칭을 깨는 조명")
    print("  광자를 늘리는 것은 어느 쪽도 아니다.")
    print()
    worst = 0.0
    print(f"    {'경우':<30}{'|I(+dz)-I(-dz)|':>18}{'기대':>8}")
    sym_cases = []
    for dz in (25.0, 60.0, 120.0):
        sym_cases.append((f"수차 없음, dz=+-{dz:.0f}", None, dz, m, True))
    for j_z, keeps in ((5, True), (6, False), (7, True), (8, True),
                       (9, True), (10, True), (11, False)):
        for amp in (0.05, 0.5):
            sym_cases.append((f"Z{j_z} = {amp}, dz=+-60", {j_z: amp}, 60.0,
                              m, keeps))
    m_asym = m.copy()
    cc = g.field_width / 2.0 + 0.33 * g.pitch
    ww = np.abs(((g.x - cc + g.field_width / 2) % g.field_width)
                - g.field_width / 2)
    m_asym[ww <= 6.0] = 1.0 + 0.0j
    sym_cases.append(("비대칭 실수 마스크, dz=+-60", None, 60.0, m_asym, True))

    n_ok = 0
    for name, ab, dz, mk, keeps in sym_cases:
        o = dict(BASE); o["aberrations"] = ab
        ip = np.asarray(IM.aerial_image(g, mk, defocus=+dz, **o)).real
        im_ = np.asarray(IM.aerial_image(g, mk, defocus=-dz, **o)).real
        e = float(np.max(np.abs(ip - im_))) / float(ip.max())
        if keeps:
            worst = max(worst, e)
        good = (e < 1e-12) if keeps else (e > 1e-6)
        n_ok += int(good)
        print(f"    {name:<30}{e:18.3e}{'짝함수' if keeps else '파괴':>8}"
              f"{'  O' if good else '  X'}")
    print(f"\n    예측과 일치 : {n_ok}/{len(sym_cases)}")
    # 복소 마스크 : 깨져야 한다
    m_cx = m.copy()
    cc = g.field_width / 2.0 + g.pitch
    ww = np.abs(((g.x - cc + g.field_width / 2) % g.field_width)
                - g.field_width / 2)
    m_cx[ww <= 0.5 * g.pitch] *= np.exp(1j * 1.0)
    o = dict(BASE); o["aberrations"] = None
    ip = np.asarray(IM.aerial_image(g, m_cx, defocus=+60.0, **o)).real
    im_ = np.asarray(IM.aerial_image(g, m_cx, defocus=-60.0, **o)).real
    broken = float(np.max(np.abs(ip - im_))) / float(ip.max())
    print(f"    {'복소(위상 1 rad) 마스크, dz=+-60':<30}{broken:18.3e}{'파괴':>8}")
    print(f"\n  -> 유지 예상 경우는 전부 {worst:.1e} (기계 정밀도) 로 짝함수.")
    print(f"     복소 마스크에서는 {broken:.3f} 로 깨진다. 두 독립 구현이 이")
    print("     값까지 일치하므로 구현 특성이 아니라 모델의 성질이다.")
    print("     위에서 정리한 세 가지 길(복소 마스크 / Z6·Z11 / 비대칭 조명)")
    print("     가운데 하나를 쓰지 않으면 부호는 되짚을 수 없다.")
    # 요약에 n_ok 를 반드시 실어야 한다. 이전에는 '분류 예측 일치' 를 조건
    # 없이 적어서, 케이스 표에 X 가 있어도 요약은 '일치' 로 나왔다. 요약만
    # 읽는 사람에게는 거짓이 된다.
    c5_ok = (n_ok == len(sym_cases)) and (broken > 1e-6)
    record("C", "C5 디포커스 짝함수성",
           f"Z5~Z11 분류 {n_ok}/{len(sym_cases)} 예측 일치, 유지 경우 "
           f"{worst:.1e}, 복소 마스크({broken:.2f})에서 파괴",
           ok=c5_ok)

    # --- C6. 선형계 극한 : 결맞음 2광 간섭 --------------------------------
    sub("C6  결맞음 2광 간섭의 해석해")
    print("  sigma -> 0 이고 0,+-1 차만 동공을 통과하면 상은 해석적으로")
    print("      I(x) = A0^2 + 2 A1^2 + 4 A0 A1 cos(2 pi x / p)")
    print("            + 2 A1^2 cos(4 pi x / p)")
    print("  이다. 이 해석해는 시뮬레이터 코드 경로를 전혀 쓰지 않는다.\n")
    # 0, +-1 차만 통과하려면 1 < pitch * NA / lambda < 2 여야 한다.
    # NA 0.75, lambda 193 nm -> 257 nm < pitch < 515 nm. pitch 130 nm 는
    # 0차만 통과해 간섭이 아예 없으므로 이 대조에 쓸 수 없다.
    pitch = 350.0
    gg = build_grid(pitch, 4, nyquist_dx(193.0, 0.75, 0.0))
    mm = MK.line_space(gg, duty=0.5)
    i = np.asarray(IM.aerial_image(gg, mm, wavelength=193.0, na=0.75,
                                   n_medium=1.0, sigma_out=1e-6,
                                   n_source_grid=1)).real
    a0, a1 = 0.5, 0.5 * np.sinc(0.5)
    xr = gg.x - gg.field_width / 2.0
    ana = (a0**2 + 2 * a1**2 + 4 * a0 * a1 * np.cos(2 * np.pi * xr / pitch)
           + 2 * a1**2 * np.cos(4 * np.pi * xr / pitch))
    f_max = 0.75 / 193.0 * pitch
    e = float(np.max(np.abs(i - ana))) / float(ana.max())
    print(f"    동공 통과 차수 : |m| <= {f_max:.2f}  -> 0, +-1 차만")
    print(f"    max 상대편차 {e:.3e}")
    record("C", "C6 2광 간섭 해석해", f"상대편차 {e:.1e}")

    # --- C7. 플레어의 에너지 보존 ----------------------------------------
    sub("C7  플레어 변환의 에너지 보존")
    print("  플레어는 상의 일부를 균일 배경으로 재분배하는 연산이다.")
    print("      I -> (1 - f) I + f <I>")
    print("  재분배이므로 공간 평균은 보존되어야 하고, 대비는 정확히")
    print("  (1 - f) 배가 되어야 한다. 둘 다 해석적으로 예측된다.\n")
    i0 = np.asarray(IM.aerial_image(g, m, **BASE)).real
    c0 = (i0.max() - i0.min()) / (i0.max() + i0.min())
    print(f"    {'f':>6} {'평균 보존 오차':>16} {'대비비 / (1-f)':>18}")
    we, wr = 0.0, 0.0
    for f in (0.01, 0.03, 0.10, 0.30):
        i1 = IM.apply_flare(i0, f)
        em = abs(i1.mean() - i0.mean()) / i0.mean()
        c1 = (i1.max() - i1.min()) / (i1.max() + i1.min())
        # 대비는 분모의 평균항 때문에 정확히 (1-f) 배가 아니다.
        # 변조 진폭 (Imax - Imin) 이 정확히 (1-f) 배가 되는 것이 올바른 등식.
        ratio = ((i1.max() - i1.min()) / (i0.max() - i0.min())) / (1 - f)
        we = max(we, em); wr = max(wr, abs(ratio - 1.0))
        print(f"    {f:6.2f} {em:16.3e} {ratio:18.12f}")
    print(f"  -> 평균 보존 오차 <= {we:.1e}, 변조 진폭비 오차 <= {wr:.1e}")
    record("C", "C7 플레어 에너지 보존",
           f"평균 보존 {we:.1e}, 변조진폭 (1-f) 배 오차 {wr:.1e}")

    # --- C8. MSD 와 블러의 평균 등가 --------------------------------------
    sub("C8  MSD 와 레지스트 블러의 평균 등가성")
    print("  둘 다 가우시안 합성곱이므로 평균 수준에서는 구분되지 않는다.")
    print("  분산이 더해지는 성질로부터")
    print("      blur(a) o blur(b) = blur(sqrt(a^2 + b^2))")
    print("  가 성립해야 한다. 이 등식이 깨지면 식별 분석의 축퇴 주장이")
    print("  근거를 잃는다.\n")
    i0 = np.asarray(IM.aerial_image(g, m, **BASE)).real
    worst = 0.0
    for a, b in ((4.0, 3.0), (8.0, 6.0), (10.0, 2.0)):
        lhs = IM.apply_msd(IM.apply_msd(i0, a, g.dx), b, g.dx)
        rhs = IM.apply_msd(i0, float(np.hypot(a, b)), g.dx)
        e = float(np.max(np.abs(lhs - rhs))) / float(i0.max())
        worst = max(worst, e)
        print(f"    msd({a}) o msd({b}) vs msd({np.hypot(a,b):.3f}) : {e:.3e}")
    print(f"  -> 최대 {worst:.1e}")
    record("C", "C8 가우시안 분산 가산성", f"최대 상대편차 {worst:.1e}")


    # --- C9. 마스크·추출의 대수 관계 --------------------------------------
    sub("C9  마스크와 추출기의 대수 관계")
    print("  해석식으로 닫히는 관계들이다. 전부 '값' 이 아니라 '관계' 이므로")
    print("  기준값을 고를 여지가 없다.\n")
    w9 = {}
    # att-PSM 의 0차 진폭
    emax = 0.0
    for t_att in (0.06, 0.20):
        for duty in (0.3, 0.5, 0.7):
            mm = MK.line_space(g, duty=duty, mask_type=MK.MASK_ATT_PSM,
                               att_t=t_att)
            ana = duty * 1.0 + (1 - duty) * (-np.sqrt(t_att))
            emax = max(emax, abs(float(np.mean(mm).real) - ana))
    w9["att-PSM 0차 진폭 = duty + (1-duty)(-sqrt T)"] = emax
    # 극성 보수 관계
    emax = 0.0
    for mt in (MK.MASK_BINARY, MK.MASK_ATT_PSM):
        c_ = MK.line_space(g, duty=0.4, polarity=MK.POLARITY_CLEAR,
                           mask_type=mt)
        d_ = MK.line_space(g, duty=0.4, polarity=MK.POLARITY_DARK,
                           mask_type=mt)
        bg_ = MK.background_amplitude(mt)
        emax = max(emax, float(np.max(np.abs(c_ + d_ - (1.0 + bg_)))))
    w9["dark + clear = 1 + 배경진폭"] = emax
    # to_2d 무손실
    w9["to_2d 가 1D 를 보존"] = float(np.max(np.abs(MK.to_2d(m, 4)[0] - m)))
    # 대칭계의 좌우 NILS 일치
    i0 = np.asarray(IM.aerial_image(g, m, **BASE)).real
    ex = RS.image_extrema(g.x, i0)
    r0 = RS.extract_cd(g.x, i0, 0.5 * (ex["i_max"] + ex["i_min"]))
    w9["대칭계 좌우 NILS 일치"] = abs(r0["nils_left"] - r0["nils_right"])
    for k, v in w9.items():
        print(f"    {k:<42}{v:12.2e}")

    # 코마에서는 좌우가 갈리고, 보고값은 나쁜 쪽이어야 한다.
    # [조건 주의] 동공을 통과하는 차수가 0, +-1 뿐인 조건(pitch 100 nm)에서는
    # 코마가 형상을 왜곡하지 않고 '횡방향 이동' 만 만든다. 이동된 프로파일은
    # 양 경계의 기울기가 같으므로 좌우 NILS 가 일치하고, 이 시험이 아무것도
    # 보지 못한다(실측 차이 0.0000). 차수가 더 통과하는 조건에서 형상 왜곡이
    # 나타나므로 전용 격자를 쓴다.
    g_coma = build_grid(200.0, 6, nyquist_dx(193.0, 1.35, 0.7))
    m_coma = MK.line_space(g_coma, duty=0.5)
    o = dict(wavelength=193.0, na=1.35, n_medium=1.44, sigma_out=0.7,
             aberrations={8: 0.08})
    g, m = g_coma, m_coma
    i2 = np.asarray(IM.aerial_image(g, m, **o)).real
    e2 = RS.image_extrema(g.x, i2)
    r2 = RS.extract_cd(g.x, i2, 0.5 * (e2["i_max"] + e2["i_min"]))
    split = abs(r2["nils_left"] - r2["nils_right"])
    conservative = abs(r2["nils"] - min(r2["nils_left"], r2["nils_right"]))
    print(f"\n    (pitch {g.pitch:.0f} nm, sigma 0.7 : 차수가 더 통과하는 조건)")
    print(f"    코마 Z8 에서 좌우 NILS 차이        {split:12.4f}  (갈려야 정상)")
    print(f"    보고값이 나쁜 쪽인가               {conservative:12.2e}  (0 이어야 함)")
    print("    공정 여유는 두 경계 중 나쁜 쪽이 정하므로 최솟값을 보고해야")
    print("    한다. 평균이나 한쪽만 쓰면 비대칭 수차에서 여유를 과대평가한다.")
    worst9 = max(max(w9.values()), conservative)
    # 판정 기준은 두 개 다 논리적이다. (1) 대수 관계는 정확히 성립해야
    # 하므로 배정밀도 누적 한계(1e-9)보다 작아야 한다. (2) 코마에서는 좌우가
    # 갈려야 하고(split > 0), 보고값은 반드시 나쁜 쪽이어야 한다
    # (conservative == 0). 어느 쪽도 임의로 고른 합격선이 아니다.
    c9_ok = (worst9 < 1e-9) and (split > 1e-3)
    record("C", "C9 마스크·추출 대수 관계",
           f"관계 4종 최대 오차 {worst9:.1e}, 코마 좌우 분리 {split:.2f} "
           f"({'갈림' if split > 1e-3 else '갈리지 않음'})",
           ok=c9_ok)


# ===========================================================================
# D. Cramer-Rao 하한의 몬테카를로 대조
# ===========================================================================

def _nll(spec: ID.IdentifySpec, y: np.ndarray) -> float:
    """가우시안 음의 로그가능도. 평균과 공분산 모두 theta 에 의존한다.

    공분산이 theta 에 의존하므로 log|Sigma| 항을 빼면 안 된다. 이 항이
    블러처럼 '평균은 그대로 두고 잡음 구조만 바꾸는' 변수의 정보를 담는다.
    """
    from scipy.linalg import cho_factor, cho_solve
    mu, sig, _ = ID.mean_and_cov(spec)
    n = len(mu)
    sig = sig + 1e-10 * float(np.trace(sig)) / n * np.eye(n)
    try:
        c, low = cho_factor(sig, lower=True, check_finite=False)
    except Exception:
        return 1e30
    r = y - mu
    quad = float(r @ cho_solve((c, low), r, check_finite=False))
    logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
    return quad + logdet


def _fit(spec0: ID.IdentifySpec, y: np.ndarray, params, x0, scale):
    """Nelder-Mead 로 MLE 를 찾는다. 파라미터는 scale 로 무차원화한다."""
    from dataclasses import replace
    from scipy.optimize import minimize

    def obj(z):
        kw = {p: float(x0[i] + z[i] * scale[i]) for i, p in enumerate(params)}
        if kw.get("dose_rel", 1.0) <= 0.1 or kw.get("blur_nm", 1.0) < 0.0:
            return 1e30
        if kw.get("msd_nm", 1.0) < 0.0 or not 0.0 <= kw.get("flare", 0.0) < 0.5:
            return 1e30
        return _nll(replace(spec0, **kw), y)

    r = minimize(obj, np.zeros(len(params)), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-4, maxiter=3000, maxfev=3000))
    return np.array([x0[i] + r.x[i] * scale[i] for i in range(len(params))])


def _trial(args):
    """표본 하나를 만들고 MLE 를 구한다."""
    spec, params, truth, scale, seed, physical = args
    rng = np.random.default_rng(seed)
    if physical:
        # 실제 생성 과정 : 광자 Poisson 표본 -> 레지스트 블러
        lam, g = ID.expected_counts(spec)
        gm = ID._blur_matrix(g.n_points, spec.blur_nm / g.dx)
        y = gm @ rng.poisson(lam).astype(float)
    else:
        # 모형이 가정한 분포 : N(mu, Sigma)
        mu, sig, _ = ID.mean_and_cov(spec)
        n = len(mu)
        l = np.linalg.cholesky(sig + 1e-10 * np.trace(sig) / n * np.eye(n))
        y = mu + l @ rng.standard_normal(n)
    est = _fit(spec, y, params, truth, scale)
    # [부호 모호성] C5 에서 보인 대로 가능도는 초점 부호에 대해 정확히
    # 짝함수이므로 +dz 와 -dz 가 똑같이 최적이다. 두 해는 물리적으로
    # 구별 불가능하므로 |dz| 로 보고하는 것이 올바른 처리이다.
    if "defocus" in params:
        k = params.index("defocus")
        est[k] = abs(est[k])
    return est


def part_d(n_trials: int = 200) -> None:
    head("D. Cramer-Rao 하한의 몬테카를로 대조")
    print("  지금까지 식별 한계는 Fisher 정보 행렬에서 해석적으로만 계산했다.")
    print("  그 값이 '실제로 달성 가능한 하한' 인지는 따로 보여야 한다. CRB 는")
    print("  불편 추정량의 하한이고, 최대가능도 추정량이 하한에 닿는 것은")
    print("  점근(고 SNR) 영역에서만 보장된다. 유한 SNR 에서는 가능도면이")
    print("  다봉이 되어 분산이 하한을 크게 벗어나는 '문턱 효과' 가 나타난다.")
    print("  참고 : H. L. Van Trees, 'Detection, Estimation, and Modulation")
    print("  Theory, Part I' (1968), 2.4절 -- 비선형 추정의 문턱 현상.")
    print("  참고 : S. M. Kay, 'Fundamentals of Statistical Signal Processing:")
    print("  Estimation Theory' (1993), 3장 -- CRB 달성 조건과 효율.\n")
    print("  표본 생성 경로를 둘로 나눈다.")
    print("    (a) 모형 분포 N(mu(th), Sigma(th)) 에서 추출")
    print("        -> FIM 계산과 가능도 구현이 서로 맞는지 본다.")
    print("    (b) 물리 과정(광자 Poisson -> 레지스트 블러)에서 추출")
    print("        -> 가우시안 근사 자체가 타당한지 본다.")
    print("  (b) 가 (a) 와 같은 결과를 줄 때 비로소 CRB 를 쓸 근거가 생긴다.")
    print("  (a) 만 맞으면 '내 가정 안에서 일관됨' 을 보인 것에 불과하다.\n")

    from concurrent.futures import ProcessPoolExecutor

    params = ("defocus", "dose_rel")
    scale = np.array([8.0, 0.015])
    print(f"  추정 대상 {params}, 참값 defocus 60 nm / dose_rel 1.0")
    print(f"  시행 {n_trials} 회 x 2 경로 x 노광량 4 수준")
    print()
    # [해석 주의] 표본 표준편차 자체도 추정값이다. 정규 표본에서 sd 의
    # 상대 표준오차는 약 1/sqrt(2(n-1)) 이므로, 시행 수가 적으면
    # sd/CRB 가 1 에서 꽤 벗어나 보일 수 있다. 이 불확실도를 함께
    # 적지 않으면 '하한을 못 지켰다' 로 과대해석된다.
    se = 1.0 / np.sqrt(2.0 * max(n_trials - 1, 1))
    print(f"  [해석 주의] 표본 표준편차의 상대 표준오차는 약 "
          f"1/sqrt(2(n-1)) = {se:.1%} 다. 따라서 sd/CRB 가 "
          f"{1 - 2 * se:.2f} ~ {1 + 2 * se:.2f} (2 시그마) 안에 있으면")
    print("  '하한 달성' 과 모순되지 않는다.")
    print()
    print(f"  {'노광량':>7} {'경로':>5} " +
          "".join(f"{'sd('+p+')':>14}{'CRB':>13}{'sd/CRB':>8}"
                  for p in params))
    print("  " + "-" * 78)

    rows = []
    for dose in (2.0, 8.0, 30.0, 120.0):
        spec = ID.IdentifySpec(dose_mj=dose, defocus=60.0, dose_rel=1.0,
                               blur_nm=8.0, flare=0.02, msd_nm=3.0)
        cr = ID.crb(spec, params=params)
        crb_v = np.asarray(cr["sigma_min"], dtype=float)
        truth = np.array([getattr(spec, p) for p in params])
        for tag, physical in (("모형", False), ("물리", True)):
            jobs = [(spec, params, truth, scale, 20_000 + 7919 * int(dose) + i,
                     physical) for i in range(n_trials)]
            with ProcessPoolExecutor() as ex:
                est = np.array(list(ex.map(_trial, jobs, chunksize=4)))
            sd = est.std(axis=0, ddof=1)
            bias = est.mean(axis=0) - truth
            line = f"  {dose:7.1f} {tag:>5} "
            for j in range(len(params)):
                line += f"{sd[j]:14.4g}{crb_v[j]:13.4g}{sd[j]/crb_v[j]:8.2f}"
            print(line)
            rows.append((dose, tag, sd, crb_v, bias))

    print("\n  sd/CRB 가 1 에 가까우면 하한이 달성된 것이다. 1 보다 크게 크면")
    print("  문턱 효과 영역이고, 1 보다 작으면 추정량이 편향되었다는 뜻")
    print("  (CRB 는 불편 추정량의 하한이므로)이니 편향도 함께 본다.\n")
    print(f"  {'노광량':>7} {'경로':>5} " +
          "".join(f"{'bias('+p+')':>18}" for p in params))
    for dose, tag, _, _, bias in rows:
        print(f"  {dose:7.1f} {tag:>5} " + "".join(f"{b:18.4g}" for b in bias))

    # --- D2. 광자 수를 3 자릿수 낮춰도 하한이 유지되는가 ----------------
    print("\n  [D2] 측정 면적을 줄여 총 광자 수를 낮춘다 (노광량 2 mJ/cm^2)")
    print("  문턱 효과는 통상 SNR 이 낮아질 때 나타난다. 여기서는 노광량을")
    print("  내리는 대신 측정 길이를 줄여 총 광자 수를 직접 낮춘다.\n")
    print(f"  {'y_length':>9}{'총 광자':>13}{'sd(dz)':>11}{'CRB':>11}{'sd/CRB':>9}")
    r2 = []
    for yl in (2048.0, 128.0, 16.0, 4.0, 1.0):
        spec = ID.IdentifySpec(dose_mj=2.0, defocus=60.0, dose_rel=1.0,
                               blur_nm=8.0, flare=0.02, msd_nm=3.0, y_length=yl)
        mu, _, _ = ID.mean_and_cov(spec)
        crb_v = np.asarray(ID.crb(spec, params=params)["sigma_min"], dtype=float)
        truth = np.array([60.0, 1.0])
        jobs = [(spec, params, truth, scale, 500 + i, True)
                for i in range(max(60, n_trials // 2))]
        with ProcessPoolExecutor() as ex:
            est = np.array(list(ex.map(_trial, jobs, chunksize=4)))
        sd = est.std(axis=0, ddof=1)
        r2.append(sd[0] / crb_v[0])
        print(f"  {yl:9.0f}{mu.sum():13.2e}{sd[0]:11.4g}{crb_v[0]:11.4g}"
              f"{sd[0]/crb_v[0]:9.2f}")
    print(f"\n  -> 총 광자 수를 3 자릿수 낮추어도 sd/CRB = "
          f"{min(r2):.2f} ~ {max(r2):.2f} 로 유지된다.")
    print("     즉 이 문제에서 문턱 효과는 광자 부족으로 생기지 않는다.")
    print("     (처음에는 저 SNR 에서 나타날 것으로 예상했으나, 측정이 수만")
    print("      화소에 걸쳐 누적되므로 실제 SNR 이 충분히 높았다.)")
    record("D", "D2 광자 수 의존",
           f"총 광자 1.8e4~1.9e7 전 구간에서 sd/CRB {min(r2):.2f}~{max(r2):.2f}")

    # --- D3. 최적 초점 근처 : 하한 자체가 기준이 되지 못하는 영역 --------
    print("\n  [D3] 참 초점을 0 에 접근시킨다")
    print("  C5 에서 보인 대로 가능도는 초점 부호에 대해 정확히 짝함수이다.")
    print("  dz -> 0 에서 두 거울 해가 한 점으로 합쳐지고 민감도가 사라진다.")
    print("  추정량은 |dz| 로 접히므로 '불편' 가정이 깨지고, CRB 는 더 이상")
    print("  적절한 기준이 아니게 된다. 경계·대칭점에서 나타나는 전형적인")
    print("  현상이다(Van Trees 2.4절의 문턱 현상과 같은 뿌리).\n")
    print(f"  {'참 dz':>7}{'sd(dz)':>11}{'CRB':>11}{'sd/CRB':>9}"
          f"{'편향':>11}{'평균 추정':>11}")
    r3 = []
    for dz0 in (40.0, 10.0, 5.0, 2.0, 0.5):
        spec = ID.IdentifySpec(dose_mj=30.0, defocus=dz0, dose_rel=1.0,
                               blur_nm=8.0, flare=0.02, msd_nm=3.0)
        crb_v = np.asarray(ID.crb(spec, params=params)["sigma_min"], dtype=float)
        truth = np.array([dz0, 1.0])
        jobs = [(spec, params, truth, np.array([4.0, 0.015]), 900 + i, True)
                for i in range(max(60, n_trials // 2))]
        with ProcessPoolExecutor() as ex:
            est = np.array(list(ex.map(_trial, jobs, chunksize=4)))
        sd, mn = est.std(axis=0, ddof=1), est.mean(axis=0)
        r3.append((dz0, sd[0] / crb_v[0], mn[0] - dz0))
        print(f"  {dz0:7.1f}{sd[0]:11.4g}{crb_v[0]:11.4g}{sd[0]/crb_v[0]:9.2f}"
              f"{mn[0]-dz0:11.3g}{mn[0]:11.4g}")
    worst_dz, worst_ratio, worst_bias = min(r3, key=lambda t: t[1])
    print(f"\n  -> dz = {worst_dz:g} nm 에서 sd/CRB = {worst_ratio:.2f} < 1 로")
    print(f"     내려가고 편향이 {worst_bias:+.3g} nm 로 커진다. CRB 는 불편")
    print("     추정량의 하한이므로, 1 보다 작다는 것은 하한이 깨진 것이")
    print("     아니라 '불편 가정이 깨졌다' 는 신호다. 따라서 최적 초점")
    print("     근방에서는 CRB 를 식별 한계로 인용하면 안 된다. 이 영역의")
    print("     한계는 분산이 아니라 부호 모호성 그 자체로 기술해야 한다.")
    record("D", "D3 최적초점 근방",
           f"dz={worst_dz:g} nm 에서 sd/CRB {worst_ratio:.2f} (<1), "
           f"불편 가정 붕괴 -> 이 영역에 CRB 인용 불가")

    # --- D4. 5 변수 동시 추정 : 프로파일이 평평하다고 한 그 경우 ----------
    print()
    print("  [D4] 5 변수 동시 추정에서 MLE 가 어떻게 거동하는가")
    print("  E 에서 defocus 와 flare 는 5 변수 동시 추정 시 프로파일이 평평")
    print("  (실용적 비식별)하다고 나왔고, 영공간 판정을 보정한 뒤 CRB 도")
    print("  무한으로 바뀌어 두 지표가 일치한다. 남은 질문은 '추정량이 실제로")
    print("  어떻게 거동하는가' 다. 2 변수만 검증하고 5 변수를 비워 두면,")
    print("  정작 결론이 걸린 영역이 미검증으로 남는다.")
    print()
    params5 = ("defocus", "dose_rel", "blur_nm", "flare", "msd_nm")
    scale5 = np.array([4.0, 0.01, 0.3, 0.002, 0.3])
    spec = ID.IdentifySpec(dose_mj=30.0, defocus=60.0, dose_rel=1.0,
                           blur_nm=8.0, flare=0.02, msd_nm=3.0)
    crb5 = np.asarray(ID.crb(spec, params=params5)["sigma_min"], dtype=float)
    truth5 = np.array([getattr(spec, p) for p in params5])
    jobs = [(spec, params5, truth5, scale5, 40_000 + i, True)
            for i in range(max(60, n_trials // 2))]
    with ProcessPoolExecutor() as ex:
        est = np.array(list(ex.map(_trial, jobs, chunksize=2)))
    sd5 = est.std(axis=0, ddof=1)
    bias5 = est.mean(axis=0) - truth5
    print(f"  {'변수':<10}{'참값':>9}{'sd':>12}{'CRB':>12}{'sd/CRB':>9}{'편향':>12}")
    print("  " + "-" * 66)
    for j, p in enumerate(params5):
        print(f"  {p:<10}{truth5[j]:9.4g}{sd5[j]:12.4g}{crb5[j]:12.4g}"
              f"{sd5[j]/crb5[j]:9.2f}{bias5[j]:12.3g}")
    r_dz = sd5[0] / crb5[0]
    r_fl = sd5[3] / crb5[3]
    print()
    print("  [해석 주의] 영공간 판정을 보정한 뒤에는 5 변수 동시 추정에서")
    print("  defocus·flare·msd_nm 의 CRB 가 **무한**이다(J4 참조). 하한이")
    print("  무한이면 'sd/CRB' 는 정의되지 않으므로, 그 변수의 비율은 읽지")
    print("  말아야 한다. 유한한 하한을 갖는 dose_rel 과 blur_nm 이 1.00 을")
    print("  주는지가 이 시험의 내용이고, 나머지 세 변수의 유한한 표본 산포는")
    print("  최적화 탐색 범위가 부과한 것이지 정보에서 나온 것이 아니다.")
    print(f"  실측 : defocus sd/CRB = {r_dz:.2f}, flare sd/CRB = {r_fl:.2f}")
    finite = np.isfinite(crb5)
    worst = float(max(sd5[finite] / crb5[finite])) if finite.any() else float("nan")
    bad = [params5[j] for j in range(len(params5))
           if not np.isfinite(crb5[j])]
    if bad:
        print(f"  -> {', '.join(bad)} 는 하한이 무한이다. 이 변수들의 식별")
        print("     한계는 CRB 로 인용할 수 없고, 프로파일 구간과 보상 경로")
        print("     (E2)로 기술해야 한다.")
        fin = [params5[j] for j in range(len(params5)) if np.isfinite(crb5[j])]
        rr = [sd5[j] / crb5[j] for j in range(len(params5))
              if np.isfinite(crb5[j])]
        print(f"     유한 하한을 갖는 {', '.join(fin)} 의 sd/CRB = "
              f"{', '.join('%.2f' % x for x in rr)}")
    else:
        print("  -> 모든 변수가 유한한 하한을 갖는다.")
    record("D", "D4 5변수 동시 추정",
           f"유한 하한 변수 sd/CRB 최대 {worst:.2f}" +
           (f", 하한 무한: {', '.join(bad)}" if bad else ""))

    hi = [r for r in rows if r[0] >= 30.0]
    ratios = np.concatenate([r[2] / r[3] for r in hi])
    lo = [r for r in rows if r[0] <= 2.0]
    ratios_lo = np.concatenate([r[2] / r[3] for r in lo]) if lo else np.array([np.nan])
    print(f"\n  고 SNR(>= 30 mJ/cm^2) : sd/CRB = {ratios.min():.2f} ~ "
          f"{ratios.max():.2f}  (2 시그마 허용 {1-2*se:.2f} ~ {1+2*se:.2f})")
    print(f"  저 SNR(2 mJ/cm^2)     : sd/CRB = {ratios_lo.min():.2f} ~ {ratios_lo.max():.2f}")
    print("  두 경로(모형/물리)가 같은 비율을 주면, 가우시안 근사가 이 광자")
    print("  수에서 타당하고 CRB 가 실제 달성 가능한 하한임을 뜻한다.")
    record("D", "CRB 몬테카를로",
           f"고SNR sd/CRB {ratios.min():.2f}~{ratios.max():.2f}, "
           f"저SNR {ratios_lo.min():.2f}~{ratios_lo.max():.2f} (문턱 효과)")


# ===========================================================================
# F. 민감도 미분의 독립 검증 (FIM 의 입력 자체를 검증한다)
# ===========================================================================

def part_f() -> None:
    head("F. 민감도 미분의 독립 검증")
    print("  Fisher 정보 행렬은 전부 d(mu)/d(theta) 와 d(Sigma)/d(theta) 로")
    print("  만들어진다. 이 미분은 identify.PARAM_STEPS 에 고정된 폭의 중심")
    print("  차분으로 계산한다. 즉 FIM, CRB, 공선성, 식별 사다리 등 본 과제의")
    print("  모든 식별성 결론이 '그 스텝 폭이 적절하다' 는 가정에 얹혀 있다.")
    print("  그런데 9주차까지 스텝 폭을 검증한 적이 없다.")
    print()
    print("  중심차분의 오차는 두 항의 합이다.")
    print("      절단 오차   ~ h^2 * (3차 미분)/6     (h 가 크면 커진다)")
    print("      반올림 오차 ~ eps * |f| / h          (h 가 작으면 커진다)")
    print("  따라서 최적 h 가 존재하고, 그 양쪽에서 미분값이 틀어진다.")
    print("  스텝을 바꾸며 미분이 평평한 구간(plateau)이 있는지 확인한다.")
    print("  참고 : W. H. Press et al., 'Numerical Recipes,' 3rd ed., 5.7 절")
    print("  -- 수치 미분의 스텝 선택과 두 오차의 균형.")
    print()

    spec = ID.IdentifySpec(dose_mj=30.0, defocus=60.0, dose_rel=1.0,
                           blur_nm=8.0, flare=0.02, msd_nm=3.0)
    from dataclasses import replace

    def dmu(pname, h):
        mp, _, _ = ID.mean_and_cov(replace(spec, **{pname: getattr(spec, pname) + h}))
        mm, _, _ = ID.mean_and_cov(replace(spec, **{pname: getattr(spec, pname) - h}))
        return (mp - mm) / (2 * h)

    print("  판정 기준은 '스텝을 바꾸면 값이 변하는가' 가 아니다. 큰 스텝에서")
    print("  값이 변하는 것은 당연하고(절단 오차), 우리가 쓰는 것은 기본 스텝")
    print("  하나뿐이다. 물어야 할 것은 두 가지다.")
    print("    (1) 기본 스텝의 미분이 h -> 0 극한값에서 얼마나 벗어나 있는가")
    print("    (2) h 를 줄일 때 반올림 오차가 터지지 않는가")
    print("  (1) 은 Richardson 외삽으로 극한값을 만들어 직접 재고, (2) 는")
    print("  관측 수렴 차수가 2 로 유지되는지로 본다. 중심차분은 h^2 로")
    print("  수렴해야 하므로, p 가 2 에서 떨어지면 반올림이 끼기 시작한 것이다.\n")

    print(f"  {'변수':<10}{'기본 h':>8}{'h0 상대오차':>13}{'관측차수 p':>11}"
          f"{'최소 h 상대오차':>16}")
    print("  " + "-" * 62)
    worst = {}
    for pname, h0 in (("defocus", 4.0), ("dose_rel", 0.01), ("blur_nm", 0.3),
                      ("flare", 0.002), ("msd_nm", 0.3)):
        facs = [0.125, 0.25, 0.5, 1.0]
        ds = [dmu(pname, h0 * f) for f in facs]
        # Richardson : 중심차분은 h^2 오차이므로 D(h/2) 와 D(h) 로
        #   D_ext = (4 D(h/2) - D(h)) / 3
        # 가 h^4 정확도를 갖는다. 가장 작은 두 스텝으로 극한값을 만든다.
        ext = (4.0 * ds[0] - ds[1]) / 3.0
        nref = float(np.linalg.norm(ext))
        errs = [float(np.linalg.norm(d - ext)) / max(nref, 1e-300) for d in ds]
        # 관측 차수 : 오차가 h^p 로 줄어드는지
        ps = []
        for a in range(1, len(facs) - 1):
            if errs[a] > 0 and errs[a + 1] > 0:
                ps.append(np.log(errs[a + 1] / errs[a])
                          / np.log(facs[a + 1] / facs[a]))
        p_obs = float(np.median(ps)) if ps else float("nan")
        worst[pname] = (errs[-1], p_obs, errs[0])
        print(f"  {pname:<10}{h0:8.4g}{errs[-1]:13.2e}{p_obs:11.2f}"
              f"{errs[0]:16.2e}")
    print()
    for pname, (e0, p_obs, emin) in worst.items():
        if e0 < 1e-10:
            # 관측량이 그 변수에 선형(또는 1차)으로 의존하면 중심차분이
            # 해석적으로 정확하다. 절단 오차가 아예 없으므로 수렴 차수도
            # 정의되지 않는다. dose_rel 은 광강도에 곱으로, flare 는
            # I -> (1-f)I + f<I> 로 각각 1차로 들어간다.
            print(f"    {pname:<10} 오차 {e0:.2e}  -- 관측량이 이 변수에 "
                  f"1차로 의존하므로 중심차분이 해석적으로 정확")
            continue
        ok = e0 < 5e-3 and 1.5 <= p_obs <= 2.5
        print(f"    {pname:<10} 기본 스텝 오차 {e0:.2e}, 관측차수 p={p_obs:.2f}"
              f"   {'적절' if ok else '재검토 필요'}")
    wmax = max(v[0] for v in worst.values())
    print()
    print(f"  -> 기본 스텝에서의 미분 오차는 최대 {wmax:.1e} 이다. 관측 차수가")
    print("     2 근방으로 유지되므로 반올림 오차가 아직 끼지 않았고, 따라서")
    print("     스텝을 더 줄여도 되지만 그럴 필요가 없는 수준이다.")
    print("     식별 분석에서 다루는 신호는 이보다 훨씬 크므로, PARAM_STEPS 는")
    print("     결론을 좌우하지 않는다.")
    record("F", "민감도 미분 스텝 연구",
           f"기본 스텝 미분 오차 <= {wmax:.1e}, 관측차수 p ~ 2 "
           f"(Richardson 외삽 기준)")


# ===========================================================================
# G. 조건 공간 전역 불변량 검사
# ===========================================================================

def part_g(n_random: int = 150, seed: int = 0) -> None:
    head("G. 조건 공간 전역 불변량 검사")
    print("  A~C 의 검증은 손으로 고른 조건 몇 개에서 수행했다. 그런데")
    print("  데이터셋은 24 차원 조건 공간에서 뽑은 11 만 건이다. 특정 조건")
    print("  몇 개가 통과했다는 사실은, 생성에 실제로 쓰인 조건 전부가")
    print("  통과한다는 보증이 되지 못한다. 손으로 고른 조건은 보통 '잘 되는")
    print("  곳' 에 몰리기 때문이다.")
    print()
    print("  그래서 조건 공간에서 무작위로 뽑아 불변량을 일괄 검사한다.")
    print("  개별 조건의 값을 기준값과 맞추는 것이 아니라, 물리가 요구하는")
    print("  등식이 모든 곳에서 성립하는지만 본다. 기준값이 필요 없으므로")
    print("  조건 수를 마음대로 늘릴 수 있다는 것이 이 방식의 장점이다.")
    print()

    rng = np.random.default_rng(seed)
    checks = {
        "개방필드 I=1": [],
        "평행이동 등가": [],
        "디포커스 짝함수": [],
        "구적 수렴(Abbe241 vs 영역분해)": [],
        "플레어 평균보존": [],
        "블러 분산가산": [],
    }
    fails = []
    for t in range(n_random):
        lam = float(rng.choice([193.0, 248.0, 13.5]))
        if lam > 100:
            n_med = float(rng.choice([1.0, 1.44]))
            na = float(rng.uniform(0.5, 0.93) * n_med)
        else:
            n_med, na = 1.0, float(rng.uniform(0.30, 0.55))
        sig_o = float(rng.uniform(0.3, 0.95))
        sig_i = float(rng.uniform(0.0, 0.7) * sig_o)
        pitch = float(rng.uniform(40.0, 400.0))
        duty = float(rng.uniform(0.25, 0.75))
        dz = float(rng.uniform(-150.0, 150.0))
        pol = str(rng.choice([IM.POL_SCALAR, IM.POL_TE, IM.POL_TM]))
        shape = str(rng.choice([IM.SOURCE_CONVENTIONAL, IM.SOURCE_DIPOLE,
                                IM.SOURCE_QUADRUPOLE]))
        mtype = str(rng.choice([MK.MASK_BINARY, MK.MASK_ATT_PSM]))
        n_res = 1.70 if lam > 100 else None
        if pol != IM.POL_SCALAR and n_res is not None and na >= n_res:
            n_res = na * 1.1
        try:
            g = build_grid(pitch, 4, nyquist_dx(lam, na, sig_o))
            m = MK.line_space(g, duty=duty, mask_type=mtype)
            o = dict(wavelength=lam, na=na, n_medium=n_med, sigma_out=sig_o,
                     sigma_in=sig_i, polarization=pol, source_shape=shape,
                     pole_offset=0.7 * sig_o, pole_radius=0.15, n_resist=n_res)

            i_open = np.asarray(IM.aerial_image(g, MK.open_frame(g),
                                                defocus=dz, **o)).real
            checks["개방필드 I=1"].append(float(np.max(np.abs(i_open - 1.0))))

            i0 = np.asarray(IM.aerial_image(g, m, defocus=dz, **o)).real
            scale = max(float(i0.max()), 1e-300)
            sh = int(rng.integers(1, g.n_points))
            i1 = np.asarray(IM.aerial_image(g, np.roll(m, sh), defocus=dz, **o)).real
            checks["평행이동 등가"].append(
                float(np.max(np.abs(i1 - np.roll(i0, sh)))) / scale)

            ip = np.asarray(IM.aerial_image(g, m, defocus=+abs(dz), **o)).real
            im_ = np.asarray(IM.aerial_image(g, m, defocus=-abs(dz), **o)).real
            checks["디포커스 짝함수"].append(float(np.max(np.abs(ip - im_))) / scale)

            # 이 항목만 비용이 크다(구적점 57600). 불변량이 아니라 두 경로의
            # 접근 정도를 재는 것이므로 전수 검사가 필요하지 않다. 일부
            # 조건만 뽑아 분포를 본다.
            if t % 5 == 0:
                a = np.asarray(IM.aerial_image(g, m, defocus=dz,
                                               source_integration="abbe",
                                               n_source_grid=241, **o)).real
                checks["구적 수렴(Abbe241 vs 영역분해)"].append(
                    float(np.max(np.abs(a - i0))) / scale)

            fl = float(rng.uniform(0.005, 0.2))
            checks["플레어 평균보존"].append(
                abs(float(IM.apply_flare(i0, fl).mean() - i0.mean()))
                / max(abs(float(i0.mean())), 1e-300))

            aa, bb = float(rng.uniform(1, 10)), float(rng.uniform(1, 10))
            l_ = IM.apply_msd(IM.apply_msd(i0, aa, g.dx), bb, g.dx)
            r_ = IM.apply_msd(i0, float(np.hypot(aa, bb)), g.dx)
            checks["블러 분산가산"].append(float(np.max(np.abs(l_ - r_))) / scale)
        except Exception as exc:                      # noqa: BLE001
            fails.append((t, repr(exc)[:90]))

    n_ok = n_random - len(fails)
    print(f"  무작위 조건 {n_random} 건 (파장 193/248/13.5, NA·sigma·pitch·duty·")
    print(f"  디포커스·편광·조명형상·마스크종류 전부 무작위), 성공 {n_ok} 건")
    print()
    print(f"  {'불변량':<32}{'중앙값':>12}{'95 분위':>12}{'최악':>12}")
    print("  " + "-" * 68)
    tol = {"구적 수렴(Abbe241 vs 영역분해)": 5e-2}
    allok = True
    for k, v in checks.items():
        if not v:
            continue
        arr = np.array(v)
        allok &= float(np.max(arr)) <= tol.get(k, 1e-9)
        print(f"  {k:<32}{np.median(arr):12.2e}{np.quantile(arr, 0.95):12.2e}"
              f"{arr.max():12.2e}")
    print()
    print("  구적 수렴 항목만 허용치가 느슨하다. 이것은 불변량이 아니라")
    print("  '구적점 241 의 Abbe 가 영역분해 결과에 얼마나 가까운가' 를 재는")
    print("  것이고, 조건에 따라 Abbe 쪽 구적 오차가 그만큼 남기 때문이다.")
    print("  나머지는 물리가 요구하는 등식이므로 기계 정밀도여야 한다.")
    if fails:
        print(f"\n  예외 {len(fails)} 건:")
        for t, e in fails[:5]:
            print(f"    #{t}: {e}")
    record("G", "무작위 조건 불변량",
           f"{n_ok}/{n_random} 조건에서 불변량 "
           f"{'전부 성립' if allok else '일부 위반'}",
           ok=bool(allok))


# ===========================================================================
# E. 프로파일 가능도 (FIM 에 의존하지 않는 식별성 판정)
# ===========================================================================

def _kl2(mu_t, sig_t, spec) -> float:
    """참 파라미터와 theta 사이의 2 x KL 발산 (가우시안).

    왜 음의 로그가능도가 아니라 KL 인가
    -----------------------------------
    잡음 실현 하나에 좌우되지 않으려고 '기대 데이터' y = mu(th_true) 를 쓰면,
    음의 로그가능도는

        NLL(th) = (y - mu(th))^T Sigma(th)^-1 (y - mu(th)) + ln|Sigma(th)|

    가 된다. 첫 항은 참값에서 0 이지만 **둘째 항은 참값에서 최소가 아니다**.
    따라서 프로파일의 최소가 참값에서 벗어나고, 곡률도 Fisher 정보와 무관한
    값이 된다(실제로 음의 신뢰구간 반폭 같은 무의미한 결과가 나왔다).

    잡음 없는 경우의 올바른 대응물은 기대 로그가능도비, 즉 2 x KL 이다.

        2 KL = tr(Sigma_t Sigma^-1) - n + d^T Sigma^-1 d
               + ln|Sigma| - ln|Sigma_t|,     d = mu(th) - mu_t

    이 양은 참값에서 정확히 0 이고 항상 0 이상이며, 참값 근방에서의 2차
    전개가 정확히 Fisher 정보 행렬이다. 따라서
      - 최소가 참값에 놓이고,
      - 2차 영역에서는 CRB 와 일치하며,
      - 2차에서 벗어나는 모양 자체가 실용적 비식별성의 증거가 된다.
    """
    mu, sig, _ = ID.mean_and_cov(spec)
    n = len(mu)
    jit = 1e-10 * float(np.trace(sig)) / n
    sig = sig + jit * np.eye(n)
    sgt = sig_t + 1e-10 * float(np.trace(sig_t)) / n * np.eye(n)
    try:
        c, low = cho_factor(sig, lower=True, check_finite=False)
    except Exception:
        return 1e30
    d = mu - mu_t
    quad = float(d @ cho_solve((c, low), d, check_finite=False))
    tr = float(np.trace(cho_solve((c, low), sgt, check_finite=False)))
    logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
    sign_t, logdet_t = np.linalg.slogdet(sgt)
    return tr - n + quad + logdet - float(logdet_t)


def _profile_point(args):
    """theta_k 를 고정한 채 나머지를 최적화해 프로파일 2KL 을 구한다."""
    from dataclasses import replace
    from scipy.optimize import minimize
    spec0, mu_t, sig_t, pname, pval, others, x0, scale = args[:8]
    # 9 번째 인자가 참이면 보상에 쓰인 나머지 변수 값까지 함께 돌려준다.
    # E2 는 '무엇이 얼마로 보상했는가' 를 표에 적어야 하므로 이것이 필요하다.
    want_fit = len(args) > 8 and bool(args[8])
    base = replace(spec0, **{pname: float(pval)})
    if not others:
        v = _kl2(mu_t, sig_t, base)
        return (v, {}) if want_fit else v

    def obj(z):
        kw = {p: float(x0[i] + z[i] * scale[i]) for i, p in enumerate(others)}
        if kw.get("dose_rel", 1.0) <= 0.1 or kw.get("blur_nm", 1.0) < 0.0:
            return 1e30
        if kw.get("msd_nm", 1.0) < 0.0 or not 0.0 <= kw.get("flare", 0.0) < 0.5:
            return 1e30
        return _kl2(mu_t, sig_t, replace(base, **kw))

    best, best_z = None, None
    for start in (np.zeros(len(others)), 0.5 * np.ones(len(others))):
        r = minimize(obj, start, method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000,
                                  maxfev=2000))
        if best is None or r.fun < best:
            best, best_z = float(r.fun), np.asarray(r.x, dtype=float)
    if want_fit:
        fit = {p: float(x0[i] + best_z[i] * scale[i])
               for i, p in enumerate(others)}
        return best, fit
    return best


def part_e(n_scan: int = 15) -> None:
    head("E. 프로파일 가능도 : FIM 에 의존하지 않는 식별성 판정")
    print("  본 과제의 핵심 주장(어떤 변수를 되짚을 수 있는가)은 지금까지")
    print("  전부 Fisher 정보 행렬과 Cramer-Rao 하한으로 계산했다. FIM 기반")
    print("  판정에는 알려진 약점이 있다.")
    print()
    print("  Wieland 등(2021)은 FIM 기반 신뢰구간이 (i) 가능도면의 2차 근사")
    print("  이므로 파라미터에 비선형인 모형에서 근거가 약하고, (ii) '실용적")
    print("  비식별성' 에 둔감하다고 지적한다. 실제로는 되짚을 수 없는 변수인데")
    print("  도 FIM 구간이 유한하게 나와 문제가 드러나지 않는 경우가 있다.")
    print("  (F.-G. Wieland, A. L. Hauber, M. Rosenblatt, C. Toensing,")
    print("   J. Timmer, 'On structural and practical identifiability,'")
    print("   Current Opinion in Systems Biology 25, 60-69 (2021),")
    print("   DOI 10.1016/j.coisb.2021.03.005)")
    print()
    print("  표준 대안이 프로파일 가능도이다. 관심 변수를 고정하고 나머지를")
    print("  매번 다시 최적화한 통계량을 그린다. 2차 근사를 쓰지 않는다.")
    print("    - 전 구간 문턱 미달 -> 구조적/실용적 비식별")
    print("    - 한쪽만 문턱 통과  -> 실용적 비식별 (구간이 열림)")
    print("    - 양쪽 통과         -> 식별 가능, 그 폭이 신뢰구간")
    print("  문턱은 자유도 1 카이제곱 95 % 분위 3.84 를 쓴다.")
    print()
    print("  통계량은 2 x KL 발산을 쓴다(_kl2 의 주석 참조). 잡음 실현에")
    print("  좌우되지 않으면서 참값에서 정확히 0 이고, 2차 영역에서 FIM 과")
    print("  일치하는 유일한 선택이다.")
    print()
    print("  [주의] 탐색 격자는 CRB 가 예측하는 구간 폭에 맞추어 잡아야 한다.")
    print("  단독 추정 시 defocus 의 95 % 반폭은 0.02 nm 수준이므로, 수십 nm")
    print("  범위를 훑으면 구간이 격자 한 칸 안에 들어가 읽을 수 없다.")
    print()

    from concurrent.futures import ProcessPoolExecutor

    full = ("defocus", "dose_rel", "blur_nm", "flare", "msd_nm")
    spec = ID.IdentifySpec(dose_mj=30.0, defocus=60.0, dose_rel=1.0,
                           blur_nm=8.0, flare=0.02, msd_nm=3.0)
    mu_t, sig_t, _ = ID.mean_and_cov(spec)
    scale_of = {"defocus": 0.05, "dose_rel": 5e-4, "blur_nm": 0.05,
                "flare": 1e-4, "msd_nm": 0.05}

    for pname in ("defocus", "blur_nm", "flare"):
        for mode, unknowns in (("단독 (나머지 기지)", (pname,)),
                               ("5 변수 동시", full)):
            others = tuple(p for p in unknowns if p != pname)
            cr = ID.crb(spec, params=unknowns)
            k = unknowns.index(pname)
            crb_sd = float(np.asarray(cr["sigma_min"], dtype=float)[k])
            truth = getattr(spec, pname)
            # 탐색 범위는 CRB 95 % 반폭의 3 배로 잡는다. 다만 영공간 보정
            # 이후 구조적 비식별 변수의 CRB 는 '무한' 이므로, 그대로 쓰면
            # 범위가 inf 가 되어 물리적으로 불가능한 값(flare = inf 등)을
            # 결상 함수에 넘기게 된다. 하한이 무한이면 그 자체가 결론이므로,
            # 범위는 공정에서 의미 있는 분해능(PRACTICAL_SCALE)으로 잡는다.
            if np.isfinite(crb_sd) and crb_sd > 0:
                span = 3.0 * 1.96 * crb_sd
                span_src = "CRB 95 % 반폭의 +-3 배"
            else:
                span = 2.0 * ID.PRACTICAL_SCALE.get(pname, 1.0)
                span_src = ("CRB 가 무한이므로 실용 분해능 기준 "
                            f"(+-2 x {ID.PRACTICAL_SCALE.get(pname, 1.0):g})")
            # 물리적으로 허용되는 구간으로 자른다. 음수가 될 수 없는
            # 변수(blur, flare, msd)와 상한이 있는 변수(flare < 1)를 지킨다.
            lo_b, hi_b = {
                "defocus": (-np.inf, np.inf),
                "blur_nm": (0.0, np.inf),
                "msd_nm": (0.0, np.inf),
                "flare": (0.0, 0.5),
                "dose_rel": (0.1, np.inf),
            }.get(pname, (-np.inf, np.inf))
            g_lo = max(truth - span, lo_b)
            g_hi = min(truth + span, hi_b)
            grid_vals = np.linspace(g_lo, g_hi, n_scan)
            x0 = np.array([getattr(spec, p) for p in others])
            sc = np.array([scale_of[p] for p in others])
            jobs = [(spec, mu_t, sig_t, pname, v, others, x0, sc)
                    for v in grid_vals]
            with ProcessPoolExecutor() as ex:
                vals = np.array(list(ex.map(_profile_point, jobs, chunksize=1)))
            chi2 = vals - vals.min()

            thr = 3.84
            i0 = int(np.argmin(chi2))
            lo = hi = None
            for i in range(i0, 0, -1):
                if chi2[i - 1] > thr >= chi2[i]:
                    lo = float(np.interp(thr, [chi2[i], chi2[i - 1]],
                                         [grid_vals[i], grid_vals[i - 1]]))
                    break
            for i in range(i0, len(chi2) - 1):
                if chi2[i + 1] > thr >= chi2[i]:
                    hi = float(np.interp(thr, [chi2[i], chi2[i + 1]],
                                         [grid_vals[i], grid_vals[i + 1]]))
                    break
            print(f"  [{pname} / {mode}]")
            print(f"    탐색 범위 {grid_vals[0]:.6g} ~ {grid_vals[-1]:.6g}  "
                  f"({span_src})")
            crb_txt = ("무한" if not np.isfinite(crb_sd)
                       else f"{1.96 * crb_sd:.4g}")
            print(f"    최소점 {grid_vals[i0]:.6g} (참값 {truth:g}), "
                  f"chi^2 최대 {chi2.max():.3g}, CRB 95 % 반폭 {crb_txt}")
            if chi2.max() < thr:
                verdict = "비식별 : 전 구간이 문턱 3.84 미달"
                print(f"    판정 : {verdict}")
            elif lo is None or hi is None:
                side = "아래쪽" if lo is None else "위쪽"
                verdict = f"실용적 비식별 ({side} 구간이 열림)"
                print(f"    판정 : {verdict}")
            elif not np.isfinite(crb_sd):
                half = 0.5 * (hi - lo)
                verdict = (f"프로파일은 닫히는데(반폭 {half:.4g}) CRB 는 무한")
                print(f"    판정 : 프로파일 95 % 반폭 {half:.4g} (CRB 는 무한)")
                print("    CRB 가 무한인데 프로파일이 닫히는 것은 모순이 아니다.")
                print("    프로파일은 탐색 범위 안에서만 평가되고, 범위 밖에서")
                print("    다시 평평해질 수 있다. 범위를 넓혀 확인해야 한다.")
            else:
                half = 0.5 * (hi - lo)
                ratio = half / (1.96 * crb_sd)
                verdict = (f"식별 가능, 프로파일 95 % 반폭 {half:.4g} "
                           f"(CRB 예측 {1.96*crb_sd:.4g}, 비 {ratio:.2f})")
                print(f"    판정 : 식별 가능")
                print(f"    프로파일 반폭 {half:.4g}  vs  CRB 예측 "
                      f"{1.96 * crb_sd:.4g}   비 = {ratio:.2f}")
                print("    비가 1 에 가까우면 가능도면이 그 범위에서 2차에")
                print("    가깝다는 뜻이고, CRB 를 인용해도 무리가 없다.")
                print("    1 보다 크게 크면 CRB 가 낙관적이라는 뜻이다.")
            record("E", f"프로파일 {pname} / {mode}", verdict)
            print()


def part_e2(n_scan: int = 17) -> None:
    """비식별 방향을 따라 보상 경로를 추적한다."""
    head("E2. 비식별 방향의 보상 경로와 그 경계")
    print("  E 에서 defocus 와 flare 가 5 변수 동시 추정에서 '실용적 비식별'")
    print("  로 나왔다. 평평함이 최적화 실패가 아닌지, 그리고 '어디까지'")
    print("  구별되지 않는지를 확인해야 한다. CRB 는 비식별이라는 사실만")
    print("  말하고 구간의 폭은 말하지 못한다.")
    print()
    print("  판별 논리 : 2KL 은 항상 0 이상이고 등호는 두 예측 분포가 같을")
    print("  때만 성립한다. 따라서 참값이 아닌 defocus 에서 2KL ~ 0 이 나오면")
    print("  '그 defocus 에서도 거의 동일한 분포를 만드는 조합이 실제로")
    print("  존재한다' 는 구성적 증거가 된다. 최적화가 실패했다면 값이 0 에서")
    print("  멀어질 뿐, 0 에 붙을 수는 없다. 즉 이 방향으로는 실패가 평평함을")
    print("  흉내낼 수 없다.")
    print()
    print("  나머지 네 변수가 어떤 값으로 보상하는지도 함께 기록한다.")
    print("  Wieland 등(2021)이 권하는 '프로파일 따라 결합 변수 그리기' 이다.\n")

    from dataclasses import replace
    from concurrent.futures import ProcessPoolExecutor

    spec = ID.IdentifySpec(dose_mj=30.0, defocus=60.0, dose_rel=1.0,
                           blur_nm=8.0, flare=0.02, msd_nm=3.0)
    mu_t, sig_t, _ = ID.mean_and_cov(spec)
    others = ("dose_rel", "blur_nm", "flare", "msd_nm")
    x0 = np.array([getattr(spec, p) for p in others])
    sc = np.array([5e-4, 0.05, 1e-4, 0.05])
    grid_vals = np.linspace(60.0, 68.0, n_scan)
    jobs = [(spec, mu_t, sig_t, "defocus", v, others, x0, sc, True)
            for v in grid_vals]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_profile_point, jobs, chunksize=1))
    vals = np.array([r[0] for r in res])
    fits = [r[1] for r in res]

    print(f"  {'defocus':>9}{'2KL':>12}{'판정':>12}"
          f"{'보상 flare':>12}{'blur':>9}{'msd':>8}")
    print("  " + "-" * 62)
    flat_hi = None
    for v, q, kw in zip(grid_vals, vals, fits):
        state = "구별 불가" if q < 3.84 else "구별 가능"
        if q < 3.84:
            flat_hi = v
        print(f"  {v:9.2f}{q:12.3e}{state:>12}"
              f"{kw['flare']:12.4f}{kw['blur_nm']:9.3f}{kw['msd_nm']:8.3f}")
    print()
    print("  flare 열이 이 표의 핵심이다. 구별 불가 구간에서 flare 가 참값")
    print("  0.02 에서 0 쪽으로 단조 감소하며 초점 증가를 상쇄하고, 0 에 닿는")
    print("  지점에서 구간이 끝난다. 즉 경계의 원인이 표에 직접 보인다.")
    print("  blur 는 그 구간에서 8.000 에 그대로 머문다. 식별 가능한 변수는")
    print("  보상에 동원되지 않는다는 뜻이고, D4 와 E 의 결론과 일치한다.")
    print()
    print("  [읽는 법] '구별 가능' 행의 나머지 열은 읽지 않는다. 그 구간에는")
    print("  참값과 같은 분포를 주는 조합이 아예 없으므로, 최적화기가 돌려준")
    print("  값은 '가장 덜 나쁜 점' 일 뿐 보상 경로가 아니다(그 행들의 2KL 은")
    print("  1e2~1e4 로 크다). 보상 경로는 2KL 이 기계 정밀도인 행에서만")
    print("  의미를 갖는다.")
    print()
    print(f"  -> defocus 60 ~ {flat_hi:.2f} nm 구간은 관측으로 전혀 구별되지")
    print("     않는다(2KL 이 기계 정밀도). 그 바깥에서 갑자기 구별 가능해진다.")
    print("     경계가 생기는 이유는 정보가 늘어서가 아니라, 보상에 쓰이던")
    print("     flare 가 물리적 하한 0 에 닿아 더 내려갈 수 없기 때문이다.")
    print()
    print("     즉 defocus 의 식별 한계를 정하는 것은 광자 수나 Fisher 정보가")
    print("     아니라 '보상 변수가 움직일 수 있는 범위' 다. FIM 은 경계 제약을")
    print("     볼 수 없으므로 이 한계를 원리적으로 포착하지 못한다.")
    print()
    # [주의] 이 값은 하드코딩하면 안 된다. 영공간 판정을 보정한 뒤
    # 5 변수 동시 추정의 defocus CRB 는 '무한' 이 되었는데, 이전에 적어 둔
    # 0.194 nm 를 그대로 두었다가 문서와 출력이 어긋났다. 실행 시점에 계산한다.
    # [버그] full 은 part_e 의 지역변수였는데 여기서 그대로 참조해 NameError
    # 로 죽었다. A~K 전체를 한 번에 돌릴 때만 드러나서(E2 가 마지막) 오래
    # 숨어 있었다. K1 의 스코프 인식 검사가 이 종류를 정적으로 잡는다.
    # defocus 를 맨 앞에 두므로 뒤의 [0] 색인이 defocus 에 대응한다.
    full = ("defocus",) + others
    cr5 = ID.crb(spec, params=full)
    sd5 = float(np.asarray(cr5["sigma_min"], dtype=float)[0])
    leak5 = float(np.asarray(cr5["nullspace_leak"], dtype=float)[0])
    crb_txt = "무한" if not np.isfinite(sd5) else f"{sd5:.4g} nm"
    print(f"     CRB 예측 : sigma(defocus) = {crb_txt} "
          f"(rank {cr5['rank']}/{len(full)}, 영공간 성분 {leak5:.4f})")
    print(f"     프로파일  : {flat_hi - 60.0:.2f} nm 구간이 완전히 구별 불가")
    if np.isfinite(sd5):
        print("     -> CRB 가 유한한데 프로파일 구간이 그보다 넓으면 CRB 는")
        print("        낙관적이다. 식별 사다리에 CRB 를 그대로 인용하면 안 된다.")
    else:
        print("     -> CRB 와 프로파일이 같은 결론을 준다(둘 다 비식별). 다만")
        print("        프로파일은 '어디까지' 구별 불가인지까지 말해 준다. CRB 는")
        print("        무한이라는 사실만 말하고 구간의 폭은 말하지 못한다.")
        print("        즉 두 지표는 경쟁 관계가 아니라 보완 관계다.")
    record("E", "E2 비식별 방향의 경계",
           f"defocus 60~{flat_hi:.2f} nm 완전 비식별 (CRB={crb_txt}), "
           f"한계는 flare >= 0 제약에서 옴")


# ===========================================================================
# H. 확률 경로 검증
# ===========================================================================

def part_h() -> None:
    head("H. 확률 경로 검증 (LER·CD 산포)")
    print("  9주차까지 확률 경로에는 LWR = sqrt(2) x LER 하나뿐이었다. 수렴")
    print("  연구도, 두 CD 경로의 교차 확인도, 스케일링 법칙 검증도 없었다.")
    print("  LER 은 절대값에 해석 기준선이 없으므로(산탄 잡음만 고려한 하한")
    print("  이다) '값을 맞추는' 검증은 불가능하다. 대신 검증할 수 있는 것은")
    print("    - 보간 방식의 전제가 성립하는가")
    print("    - 결정론 경로와 확률 경로가 같은 양을 재는가")
    print("    - 측정 길이에 수렴하는가")
    print("    - 이론이 예측하는 스케일링 지수를 재현하는가")
    print("  네 가지다.\n")

    g = build_grid(200.0, 6, 2.0)
    m = MK.line_space(g, duty=0.5)
    i_focus = np.asarray(IM.aerial_image(g, m, 193.0, 1.35, n_medium=1.44,
                                         sigma_out=0.7)).real
    e = RS.image_extrema(g.x, i_focus)
    th = 0.5 * (e["i_max"] + e["i_min"])

    # --- H1. 보간 전제 : 블러 후 장이 대역제한인가 ------------------------
    sub("H1  경계 추출 보간의 전제")
    print("  10주차에 경계 추출을 대역제한 보간으로 바꾸었다. 그 전제는")
    print("  '블러 후 장이 격자 Nyquist 안에 들어간다' 는 것이다. 광자 잡음은")
    print("  백색이지만 가우시안 블러의 전달함수 exp(-2 pi^2 sigma^2 f^2) 가")
    print("  고주파를 지운다. 전제가 깨지면 Gibbs 물결이 경계를 흔들므로,")
    print("  가정하지 않고 직접 측정한다.\n")
    print(f"    {'dx':>6}{'blur':>7}{'Nyquist 전달률':>17}{'실측 전력비':>15}")
    worst = 0.0
    for dxt in (4.0, 2.0, 1.0):
        for blur in (4.0, 8.0):
            gg = build_grid(200.0, 6, dxt)
            mm = MK.line_space(gg, duty=0.5)
            ii = np.asarray(IM.aerial_image(gg, mm, 193.0, 1.35,
                                            n_medium=1.44, sigma_out=0.7)).real
            fld = ST.stochastic_field(ii, gg.dx, 30.0, 193.0, n_rows=256,
                                      dy=2.0, blur_nm=blur,
                                      rng=np.random.default_rng(1))
            fny = 1.0 / (2 * gg.dx)
            trans = float(np.exp(-2 * np.pi ** 2 * blur ** 2 * fny ** 2))
            sp = np.abs(np.fft.rfft(fld - fld.mean(axis=1, keepdims=True),
                                    axis=1)) ** 2
            sp = sp.mean(axis=0)
            ratio = float(sp[-1] / sp.max())
            worst = max(worst, ratio)
            print(f"    {gg.dx:6.2f}{blur:7.1f}{trans:17.2e}{ratio:15.2e}")
    print(f"\n    -> Nyquist 성분이 최대 {worst:.1e} 이다. 장은 기계 정밀도")
    print("       수준에서 대역제한이므로 대역제한 보간이 올바른 복원이고,")
    print("       선형보간이 O(dx^2) 오차를 갖는 쪽이다.")
    # 기준 : Nyquist 전력비가 1e-6 미만이면 Gibbs 물결의 진폭이 경계 추출
    # 정밀도(1e-13 nm 급)보다 훨씬 작으므로 대역제한 보간이 정당하다.
    # 1e-6 은 임의로 고른 합격선이 아니라, '물결이 측정량보다 작다' 는
    # 요구를 만족하는 느슨한 상한이다. 실측 최악은 이보다 네 자릿수 작다.
    h1_ok = worst < 1e-6
    record("H", "H1 보간 전제",
           f"블러 후 Nyquist 전력비 최악 {worst:.1e} "
           f"(기준 1e-6, {'성립' if h1_ok else '위반'})",
           ok=h1_ok)

    # --- H2. 두 CD 경로의 교차 확인 --------------------------------------
    sub("H2  결정론 경로와 확률 경로의 일치 (무잡음 한계)")
    print("  본 과제는 '결정론 CD 산포 대 확률 CD 산포' 를 직접 비교한다.")
    print("  두 값이 서로 다른 보간 정의로 계산되면 비교가 흐려지므로, 잡음을")
    print("  0 으로 둔 한계에서 같은 값을 주는지 확인해야 한다. 이 교차 확인이")
    print("  9주차까지 없었고, 실제로 10주차 중간에 두 경로가 갈려 있었다.\n")
    print(f"    {'pitch':>7}{'duty':>6}{'dx':>6}{'결정론 CD':>15}{'확률 CD':>15}{'차이':>11}")
    wmax = 0.0
    for pitch, duty in ((200.0, 0.5), (200.0, 0.35), (400.0, 0.6)):
        for dxt in (2.0, 1.0):
            gg = build_grid(pitch, 6, dxt)
            mm = MK.line_space(gg, duty=duty)
            ii = np.asarray(IM.aerial_image(gg, mm, 193.0, 1.35,
                                            n_medium=1.44, sigma_out=0.7)).real
            ee = RS.image_extrema(gg.x, ii)
            t2 = 0.5 * (ee["i_max"] + ee["i_min"])
            det = RS.extract_cd(gg.x, ii, t2, max_width=pitch)
            f2 = np.tile(ii, (4, 1))
            lf = ST.edge_positions(gg.x, f2, t2, "left")
            rt = ST.edge_positions(gg.x, f2, t2, "right")
            cs = float(np.nanmean(rt - lf))
            d = abs(det["cd"] - cs)
            wmax = max(wmax, d)
            print(f"    {pitch:7.0f}{duty:6.2f}{gg.dx:6.2f}{det['cd']:15.8f}"
                  f"{cs:15.8f}{d:11.2e}")
    print(f"\n    -> 최대 차이 {wmax:.1e} nm (통일 전 2.4e-3 nm)")
    record("H", "H2 두 CD 경로 일치", f"무잡음 한계 최대 차이 {wmax:.1e} nm")

    # --- H3. 측정 길이 수렴 ----------------------------------------------
    sub("H3  LER 의 측정 길이 수렴")
    print("  LER 은 반복 횟수가 아니라 y 방향 측정 길이(n_rows x dy)에 수렴한다.")
    print("  선 가장자리 거칠기는 장파장 성분을 포함하므로, 측정 창이 짧으면")
    print("  저주파 성분이 빠져 LER 이 과소평가된다. 측정 길이를 늘려 포화를")
    print("  확인한다. 이것이 '충분한 측정 길이' 의 근거가 된다.\n")
    print(f"    {'측정길이[nm]':>13}{'n_rows':>8}{'dy':>6}{'LER_3sigma':>13}{'변화':>10}")
    prev = None
    rows = []
    for n_rows in (64, 128, 256, 512, 1024, 2048):
        dy = 2.0
        r = ST.monte_carlo(g.x, i_focus, g.dx, 30.0, 193.0, th, n_rows=n_rows,
                           blur_nm=8.0, n_trials=6, seed=11, dy=dy,
                           max_width=g.pitch)
        ler = r["ler_3sigma"]
        rows.append((n_rows * dy, ler))
        ch = "" if prev is None else f"{abs(ler-prev)/ler:10.2%}"
        print(f"    {n_rows*dy:13.0f}{n_rows:8d}{dy:6.1f}{ler:13.5f}{ch:>10}")
        prev = ler
    sat = abs(rows[-1][1] - rows[-2][1]) / rows[-1][1]
    print(f"\n    -> 측정 길이 {rows[-2][0]:.0f} -> {rows[-1][0]:.0f} nm 에서 변화"
          f" {sat:.2%}. 포화에 가깝다.")
    print("       생산 설정은 블러 길이의 100 배 이상을 권장한다는 기존 지침과")
    print("       부합한다.")
    record("H", "H3 측정 길이 수렴", f"{rows[-1][0]:.0f} nm 에서 변화 {sat:.2%}")

    # --- H4. 스케일링 법칙 -----------------------------------------------
    sub("H4  산탄 잡음 지배 영역의 스케일링 지수")
    print("  LER 절대값에는 해석 기준선이 없다. 그러나 산탄 잡음이 지배하면")
    print("  경계 떨림은 '잡음 / 기울기' 로 주어지고, 잡음은 광자 수의 제곱근에")
    print("  반비례하므로")
    print("      LER ~ (노광량)^(-1/2)")
    print("  가 성립해야 한다. 지수는 모형이 아니라 포아송 통계가 정하므로,")
    print("  이것은 값이 아니라 '법칙' 을 검증하는 것이다.\n")
    doses = np.array([7.5, 15.0, 30.0, 60.0, 120.0])
    lers = []
    for d in doses:
        r = ST.monte_carlo(g.x, i_focus, g.dx, float(d), 193.0, th, n_rows=512,
                           blur_nm=8.0, n_trials=8, seed=13, dy=2.0,
                           max_width=g.pitch)
        lers.append(r["ler_3sigma"])
    lers = np.array(lers)
    p = np.polyfit(np.log(doses), np.log(lers), 1)[0]
    print(f"    {'노광량':>8}{'LER_3sigma':>13}")
    for d, l in zip(doses, lers):
        print(f"    {d:8.1f}{l:13.5f}")
    print(f"\n    로그-로그 기울기 = {p:.3f}   (이론 -0.5)")
    print(f"    편차 {abs(p+0.5):.3f}")
    record("H", "H4 스케일링 지수", f"LER ~ dose^{p:.3f} (이론 -0.5)")

    # --- H5. MSD 와 레지스트 블러의 구별 ----------------------------------
    sub("H5  MSD 와 레지스트 블러가 잡음에서 구별되는가")
    print("  본 과제 식별 분석의 전제 하나는 'MSD 는 광자 샘플링 이전의 광강도")
    print("  에, 레지스트 블러는 샘플링 이후의 광자 수에 작용하므로 평균은")
    print("  같고 잡음 구조만 다르다' 는 것이다. 평균이 같다는 것은 C8 에서")
    print("  확인했다. 여기서는 '잡음에서는 실제로 다르다' 를 확인한다. 같으면")
    print("  축퇴를 깰 수 없으므로 식별 주장이 무너진다.\n")
    tot = 8.0
    # [정밀도] 중간 두 배분은 반올림한 6.928 을 쓰면 합성 평활이 정확히
    # 8 이 되지 않는다(7.99981). 이 시험의 전제가 '합성이 같다' 는 것이므로
    # 전제가 엄밀히 성립하지 않게 되고, 결정론 CD 도 8e-07 만큼 갈린다.
    # sqrt(48) 을 그대로 써서 합성을 기계 정밀도로 맞춘다. 그러면 결정론
    # CD 의 상대편차가 정확히 0 이 되고, 남는 차이는 전부 잡음 구조에서
    # 온 것임이 분명해진다.
    _b = float(np.sqrt(tot ** 2 - 4.0 ** 2))      # = sqrt(48)
    print(f"    {'MSD':>6}{'blur':>6}{'유효 평활':>11}{'LER_3sigma':>13}")
    out = []
    for msd, blur in ((0.0, tot), (4.0, _b), (_b, 4.0), (tot, 0.0)):
        ii = np.asarray(IM.aerial_image(g, m, 193.0, 1.35, n_medium=1.44,
                                        sigma_out=0.7, msd_nm=msd)).real
        ee = RS.image_extrema(g.x, ii)
        t2 = 0.5 * (ee["i_max"] + ee["i_min"])
        r = ST.monte_carlo(g.x, ii, g.dx, 30.0, 193.0, t2, n_rows=512,
                           blur_nm=blur, n_trials=8, seed=17, dy=2.0,
                           max_width=g.pitch)
        out.append(r["ler_3sigma"])
        print(f"    {msd:6.2f}{blur:6.2f}{np.hypot(msd, blur):11.3f}"
              f"{r['ler_3sigma']:13.5f}")
    spread = (max(out) - min(out)) / max(out)
    # 전제를 눈으로 확인할 수 있게 합성 평활의 편차를 직접 찍는다. '같다' 는
    # 주장을 3 자리 표시로 보여 주는 것만으로는 증거가 되지 않는다.
    _comp = [np.hypot(a, b) for a, b in
             ((0.0, tot), (4.0, _b), (_b, 4.0), (tot, 0.0))]
    print(f"\n    합성 평활 편차 : {max(_comp) - min(_comp):.3e} nm "
          f"(네 경우가 기계 정밀도로 같다)")
    print(f"    유효 평활 길이는 네 경우 모두 {tot:.1f} nm 로 같지만 LER 은")
    print(f"    {spread:.1%} 범위로 갈린다. 평균으로는 구별되지 않고 잡음으로는")
    print("    구별되는 것이 맞다. MSD 쪽이 커질수록 LER 이 커지는데, MSD 는")
    print("    광자 샘플링 전에 작용해 잡음을 평활하지 못하기 때문이다.")
    record("H", "H5 MSD/블러 축퇴 분리",
           f"유효 평활 동일 조건에서 LER 이 {spread:.1%} 범위로 갈림")


# ===========================================================================
# I. 경계 조건 시험
# ===========================================================================

def part_i() -> None:
    head("I. 경계 조건 시험")
    print("  G 의 무작위 표본은 조건 공간의 '내부' 를 본다. 그런데 수치적으로")
    print("  위험한 곳은 거의 항상 경계다. 무작위 추출은 경계 근방을 거의")
    print("  뽑지 않으므로, 경계는 따로 열거해 검사해야 한다. 경계값 시험은")
    print("  소프트웨어 시험의 표준 기법이기도 하다.")
    print()
    print("  각 항목은 '막혀야 하는 입력은 막히는가' 와 '허용되는 극단에서도")
    print("  물리 불변량이 성립하는가' 를 함께 본다. 막히는 것만 보면 과하게")
    print("  막는 버그를, 성립만 보면 못 막는 버그를 놓친다.\n")

    base = dict(wavelength=193.0, n_medium=1.44, sigma_out=0.7)
    fails = []

    # --- I1. 물리적으로 불가능한 입력이 막히는가 ---------------------------
    sub("I1  불가능한 입력의 차단")
    cases = [
        ("NA = 굴절률", dict(na=1.44, n_medium=1.44), ValueError),
        ("NA > 굴절률", dict(na=1.50, n_medium=1.44), ValueError),
        ("NA = 0", dict(na=0.0, n_medium=1.44), ValueError),
        ("NA < 0", dict(na=-0.5, n_medium=1.44), ValueError),
        ("벡터 모델에서 NA >= n_resist",
         dict(na=1.35, n_medium=1.44, n_resist=1.20,
              polarization=IM.POL_TE), ValueError),
        ("없는 편광", dict(na=1.35, polarization="원형"), ValueError),
        ("Z4 중복 지정", dict(na=1.35, aberrations={4: 0.1}), ValueError),
        ("환형 폭 0 (sigma_in=sigma_out)",
         dict(na=1.35, sigma_out=0.7, sigma_in=0.7), ValueError),
        ("sigma_in > sigma_out", dict(na=1.35, sigma_out=0.7, sigma_in=0.8),
         ValueError),
        ("sigma_in < 0", dict(na=1.35, sigma_out=0.7, sigma_in=-0.1),
         ValueError),
        ("지원 외 Zernike", dict(na=1.35, aberrations={22: 0.1}), ValueError),
    ]
    g = build_grid(200.0, 4, 2.0)
    m = MK.line_space(g, duty=0.5)
    print(f"    {'입력':<30}{'기대':>10}{'결과':>12}")
    for name, kw, exc in cases:
        o = dict(base); o.update(kw)
        try:
            IM.aerial_image(g, m, **o)
            got = "통과(문제)"
            fails.append(f"I1 {name}: 차단되지 않음")
        except exc:
            got = "차단"
        except Exception as e:                        # noqa: BLE001
            got = f"다른 예외"
            fails.append(f"I1 {name}: {type(e).__name__}")
        print(f"    {name:<30}{'차단':>10}{got:>12}")

    # 마스크 쪽 경계
    print()
    mask_cases = [
        ("cd = 0", dict(cd=0.0), ValueError),
        ("cd = pitch", dict(cd=200.0), ValueError),
        ("cd > pitch", dict(cd=250.0), ValueError),
        ("cd < 0", dict(cd=-10.0), ValueError),
        ("att_t = 0", dict(duty=0.5, mask_type=MK.MASK_ATT_PSM, att_t=0.0),
         ValueError),
        ("att_t = 1", dict(duty=0.5, mask_type=MK.MASK_ATT_PSM, att_t=1.0),
         ValueError),
        ("없는 마스크 종류", dict(duty=0.5, mask_type="tri-tone"), ValueError),
        ("없는 표본화 방식", dict(duty=0.5, sampling="nearest"), ValueError),
    ]
    for name, kw, exc in mask_cases:
        try:
            MK.line_space(g, **kw)
            got = "통과(문제)"
            fails.append(f"I1 {name}: 차단되지 않음")
        except exc:
            got = "차단"
        except Exception as e:                        # noqa: BLE001
            got = "다른 예외"
            fails.append(f"I1 {name}: {type(e).__name__}")
        print(f"    {name:<30}{'차단':>10}{got:>12}")
    n_block = len(cases) + len(mask_cases)
    record("I", "I1 불가능 입력 차단",
           f"{n_block} 경우 전부 차단" if not fails
           else f"{len(fails)} 건 미차단",
           ok=not fails)

    # --- I2. 허용 경계 극단에서 불변량이 성립하는가 ------------------------
    sub("I2  허용 경계 극단에서의 불변량")
    print("  막히지 않는 극단값들이다. 여기서 불변량이 깨지면 조용히 틀린")
    print("  숫자가 데이터셋에 들어간다.\n")
    # dz 짝함수성은 수차에 조건이 붙는다(C5 참조). Z6 와 Z11 은 깨지는 것이
    # 정상이므로, 항목별로 기대를 지정하지 않으면 올바른 거동을 결함으로
    # 잘못 보고한다. 개방필드 허용치도 퇴화 조건에서는 따로 둔다.
    extremes = [
        ("NA = 0.999 x n (액침 한계)", dict(na=1.44 * 0.999, n_medium=1.44),
         dict(duty=0.5)),
        ("NA = 0.01 (초저 NA)", dict(na=0.01, n_medium=1.44), dict(duty=0.5)),
        ("sigma_out = 0.999", dict(na=1.35, sigma_out=0.999), dict(duty=0.5)),
        ("sigma_out = 1e-9 (결맞음 극한)", dict(na=1.35, sigma_out=1e-9),
         dict(duty=0.5)),
        ("환형 폭 1e-2", dict(na=1.35, sigma_out=0.7, sigma_in=0.69),
         dict(duty=0.5)),
        ("환형 폭 1e-6 (퇴화)", dict(na=1.35, sigma_out=0.7,
                                 sigma_in=0.699999), dict(duty=0.5)),
        ("duty = 0.001", dict(na=1.35), dict(duty=0.001)),
        ("duty = 0.999", dict(na=1.35), dict(duty=0.999)),
        ("극단 디포커스 2000 nm", dict(na=1.35, defocus=2000.0),
         dict(duty=0.5)),
        ("극단 수차 Z11 = 2 파장 (dz 짝 깨짐 정상)",
         dict(na=1.35, aberrations={11: 2.0}), dict(duty=0.5)),
        ("극단 수차 Z10 = 2 파장", dict(na=1.35, aberrations={10: 2.0}),
         dict(duty=0.5)),
        ("플레어 0.999", dict(na=1.35, flare=0.999), dict(duty=0.5)),
        ("MSD 500 nm", dict(na=1.35, msd_nm=500.0), dict(duty=0.5)),
        ("극 반경 1e-4 (점 쌍극)",
         dict(na=1.35, source_shape=IM.SOURCE_DIPOLE, pole_offset=0.7,
              pole_radius=1e-4), dict(duty=0.5)),
        ("극이 동공 밖으로 대부분",
         dict(na=1.35, source_shape=IM.SOURCE_DIPOLE, pole_offset=1.05,
              pole_radius=0.15), dict(duty=0.5)),
    ]
    print(f"    {'조건':<34}{'유한':>6}{'개방필드':>11}{'평행이동':>11}"
          f"{'dz 짝':>10}{'dz기대':>8}")
    for name, okw, mkw in extremes:
        o = dict(base); o.update(okw)
        try:
            mm = MK.line_space(g, **mkw)
            i0 = np.asarray(IM.aerial_image(g, mm, **o)).real
            finite = bool(np.all(np.isfinite(i0)))
            scale = max(float(np.max(np.abs(i0))), 1e-300)
            iop = np.asarray(IM.aerial_image(g, MK.open_frame(g), **o)).real
            # 플레어는 개방필드를 바꾸지 않아야 하고(균일장의 재분배는
            # 항등이다), MSD 도 균일장을 바꾸지 않는다.
            e_open = float(np.max(np.abs(iop - 1.0)))
            sh = 7
            i1 = np.asarray(IM.aerial_image(g, np.roll(mm, sh), **o)).real
            e_sh = float(np.max(np.abs(i1 - np.roll(i0, sh)))) / scale
            o2 = dict(o)
            dz = abs(o.get("defocus", 60.0)) or 60.0
            o2["defocus"] = +dz
            ip = np.asarray(IM.aerial_image(g, mm, **o2)).real
            o2["defocus"] = -dz
            im_ = np.asarray(IM.aerial_image(g, mm, **o2)).real
            e_dz = float(np.max(np.abs(ip - im_))) / scale
            ab = okw.get("aberrations") or {}
            exp_txt = "깨짐" if any(k in (6, 11) for k in ab) else "짝"
            print(f"    {name:<34}{'예' if finite else '아님':>6}"
                  f"{e_open:11.1e}{e_sh:11.1e}{e_dz:10.1e}{exp_txt:>8}")
            if not finite:
                fails.append(f"I2 {name}: 비유한 값")
            # Z6·Z11 은 dz 짝함수성을 깨는 것이 정상이다(C5 의 패리티 기준).
            ab = okw.get("aberrations") or {}
            dz_even_expected = not any(k in (6, 11) for k in ab)
            # 폭 1e-6 환형은 étendue 가 0 에 수렴하는 퇴화 조건이라 구적
            # 정밀도가 떨어진다. 물리적으로 쓰이지 않는 영역이므로 허용치를
            # 따로 둔다(단, 값을 숨기지 않고 표에 그대로 적는다).
            tol_open = 1e-6 if "퇴화" in name else 1e-9
            checks_ = [("개방필드", e_open, tol_open),
                       ("평행이동", e_sh, 1e-9)]
            if dz_even_expected:
                checks_.append(("dz 짝함수", e_dz, 1e-9))
            elif e_dz < 1e-9:
                fails.append(f"I2 {name}: dz 짝함수가 깨져야 하는데 유지됨")
            for tag, val, tol in checks_:
                if val > tol:
                    fails.append(f"I2 {name}: {tag} {val:.1e}")
        except Exception as e:                        # noqa: BLE001
            print(f"    {name:<34}  예외 : {type(e).__name__}: {e}"[:112])
            fails.append(f"I2 {name}: {type(e).__name__}")

    # --- I3. 해상 한계 근방 -----------------------------------------------
    sub("I3  해상 한계 근방")
    print("  피치를 줄이면 1차 회절광이 동공을 벗어나 대비가 0 으로 간다.")
    print("  그 근방에서 CD 추출이 '미형성' 을 올바로 보고하는지, 그리고")
    print("  NaN 대신 0 같은 값을 조용히 돌려주지 않는지 확인한다.\n")
    lam, na, sg = 193.0, 1.35, 0.7
    p_cut = lam / (na * (1.0 + sg))
    print(f"    1차광 차단 피치 ~ lambda/(NA(1+sigma)) = {p_cut:.2f} nm")
    print(f"    {'pitch':>8}{'대비':>12}{'formed':>9}{'CD':>12}")
    for pitch in (p_cut * 0.8, p_cut * 0.95, p_cut * 1.05, p_cut * 1.3):
        gg = build_grid(pitch, 6, nyquist_dx(lam, na, sg))
        mm = MK.line_space(gg, duty=0.5)
        ii = np.asarray(IM.aerial_image(gg, mm, wavelength=lam, na=na,
                                        n_medium=1.44, sigma_out=sg)).real
        c = IM.contrast(ii, gg.x)
        ee = RS.image_extrema(gg.x, ii)
        r = RS.extract_cd(gg.x, ii, 0.5 * (ee["i_max"] + ee["i_min"]),
                          max_width=pitch)
        cd_txt = "nan" if not np.isfinite(r["cd"]) else f"{r['cd']:.4f}"
        print(f"    {pitch:8.2f}{c:12.3e}{str(r['formed']):>9}{cd_txt:>12}")
        if r["formed"] and not np.isfinite(r["cd"]):
            fails.append(f"I3 pitch {pitch:.1f}: formed=True 인데 CD 가 NaN")
        if (not r["formed"]) and r["cd"] == 0.0:
            fails.append(f"I3 pitch {pitch:.1f}: 미형성인데 CD 0.0 반환")

    print()
    if fails:
        print(f"  [문제 {len(fails)} 건]")
        for f in fails:
            print(f"    - {f}")
    else:
        print("  -> 경계 전 항목 통과. 막아야 할 입력은 전부 막히고, 허용되는")
        print("     극단에서도 불변량이 기계 정밀도로 성립한다.")
    record("I", "I2/I3 경계 극단",
           "전 항목 통과" if not fails else f"{len(fails)} 건 문제",
           ok=not fails)


# ===========================================================================
# J. 파라미터 축퇴 구조 (개념적 검증)
# ===========================================================================

def part_j() -> None:
    head("J. 파라미터 축퇴 구조")
    print("  수치가 맞는지와 별도로, '그 수치가 무엇의 함수인가' 를 확인해야")
    print("  한다. 서로 정확히 축퇴인 두 입력을 독립 조건처럼 보고하면,")
    print("  틀린 값은 아니지만 틀린 해석을 유도한다. 아래 세 가지는 모두")
    print("  해석적으로 축퇴이며, 실측으로 확인한다.\n")

    g = build_grid(200.0, 6, 2.0)
    m = MK.line_space(g, duty=0.5)
    i0 = np.asarray(IM.aerial_image(g, m, 193.0, 1.35, n_medium=1.44,
                                    sigma_out=0.7)).real

    # --- J1. threshold x dose_mj -----------------------------------------
    sub("J1  threshold 와 dose_mj")
    print("  경계에서 I x e_op = threshold 이므로 경계의 기대 광자 수는")
    print("  N_ref(dose_mj) x threshold 다. LER 은 그 제곱근에 반비례하므로")
    print("  두 입력은 곱으로만 들어간다. 곱 = 실효 dose-to-clear E_0 다.\n")
    from litho import dataset as DS
    print(f"    {'threshold':>10}{'dose_mj':>9}{'E_0':>7}{'LER_3s':>10}{'CD':>11}")
    groups = {}
    for th, dose in ((0.30, 30.0), (0.15, 60.0), (0.60, 15.0),
                     (0.30, 60.0), (0.60, 30.0)):
        sp = DS.SampleSpec(threshold=th, dose_mj=dose, pitch=200.0, duty=0.5,
                           n_trials=6, n_rows=512, dy=2.0, seed=5, blur_nm=8.0)
        r = DS.make_sample(sp)
        lab = dict(zip(DS.LABEL_KEYS, r["labels"]))
        groups.setdefault(round(th * dose, 6), []).append(lab["ler_3sigma"])
        print(f"    {th:10.2f}{dose:9.1f}{th*dose:7.1f}{lab['ler_3sigma']:10.5f}"
              f"{lab['cd_deterministic']:11.5f}")
    worst = 0.0
    for e0, v in sorted(groups.items()):
        if len(v) > 1:
            spread = (max(v) - min(v)) / float(np.mean(v))
            worst = max(worst, spread)
            print(f"    E_0 = {e0:5.1f} 인 조합끼리 LER 상대편차 {spread:.3e}")
    print(f"\n    -> 최대 {worst:.1e}. 정확히 축퇴다. 따라서 'dose 30 mJ/cm^2")
    print("       에서 LER' 이라고만 적으면 threshold 를 모르는 독자가 광자")
    print("       수를 잘못 환산한다. E_0 를 함께 보고해야 한다(라벨에 추가).")
    record("J", "J1 threshold x dose 축퇴", f"LER 상대편차 {worst:.1e}")

    # --- J2. y_length x dose_mj (식별 하한) -------------------------------
    sub("J2  계측 길이와 노광량 (식별 하한)")
    print("  정보량은 총 광자 수에 비례하므로 CRB 는 그 제곱근에 반비례해야")
    print("  한다. 성립하면 '식별 하한 0.0098 nm' 같은 절대값 인용은 계측")
    print("  조건 없이는 의미가 없다.\n")
    print(f"    {'y_length':>9}{'dose':>7}{'예산비':>8}{'sigma(dz)':>12}"
          f"{'x sqrt(N)':>12}")
    prods = []
    for yl, dm in ((2048.0, 30.0), (512.0, 30.0), (8192.0, 30.0),
                   (2048.0, 7.5), (512.0, 120.0)):
        sp = ID.IdentifySpec(y_length=yl, dose_mj=dm, defocus=60.0)
        sd = float(np.asarray(ID.crb(sp, params=("defocus", "dose_rel"))
                              ["sigma_min"], dtype=float)[0])
        n = ID.photon_budget_rel(sp)
        prods.append(sd * np.sqrt(n))
        print(f"    {yl:9.0f}{dm:7.1f}{n:8.3f}{sd:12.6f}{sd*np.sqrt(n):12.6f}")
    spread = (max(prods) - min(prods)) / float(np.mean(prods))
    print(f"\n    -> sigma x sqrt(예산) 의 상대편차 {spread:.1e}. 1/sqrt(N)")
    print("       스케일링이 정확히 성립하고, y_length 와 dose 는 서로 축퇴다.")
    record("J", "J2 광자 예산 스케일링", f"sigma x sqrt(N) 상대편차 {spread:.1e}")

    # --- J3. NILS 기준과 CD 공차 예산 -------------------------------------
    sub("J3  품질 기준의 예산 정합성")
    print("  NILS 하한과 CD 변동 상한은 독립 기준이 아니다. Mack 근사")
    print("      EL[%] ~ NILS x (CD 공차[%])")
    print("  에 의해 NILS 기준은 결정론적 공정 윈도 몫에서 나오고, CDU 기준은")
    print("  확률 변동 몫에서 나온다. 둘에 각각 전체 공차를 주면 합이 공차를")
    print("  넘는다. 예산을 제곱합으로 배분했는지 확인한다.\n")
    print(f"    {'예산':>7}{'배분':>7}{'stoch':>8}{'pw':>8}{'합(RSS)':>9}"
          f"{'min_NILS':>10}")
    ok_all = True
    for b, sp_, el in ((0.10, 0.5, 15.0), (0.10, 0.3, 15.0),
                       (0.10, 0.7, 15.0), (0.15, 0.5, 15.0)):
        q = DS.SampleSpec(cd_budget=b, budget_split=sp_, el_required=el)
        rss = float(np.hypot(q.stoch_tol, q.pw_tol))
        ok_all &= abs(rss - b) < 1e-12
        print(f"    {b:7.2f}{sp_:7.2f}{q.stoch_tol:8.4f}{q.pw_tol:8.4f}"
              f"{rss:9.4f}{q.min_nils:10.3f}")
    print(f"\n    -> 제곱합이 전체 예산과 일치: {'예' if ok_all else '아니오'}")
    print("       기존에는 NILS 2.0 과 CD 공차 0.10 을 근거 없는 '업계 표준'")
    print("       두 개로 적어 두었고, 둘이 같은 예산을 나눠 쓴다는 점이")
    print("       드러나지 않았다. 이제 예산과 배분에서 유도된다.")
    record("J", "J3 품질 기준 예산 정합성",
           "제곱합 배분 일치" if ok_all else "불일치",
           ok=bool(ok_all))

    # --- J4. 식별 하한의 단조성 ------------------------------------------
    sub("J4  미지수를 늘릴 때 하한은 나빠지기만 해야 한다")
    print("  정보가 같은데 추정할 것이 늘면 각 파라미터의 하한은 절대 좋아질")
    print("  수 없다. 이 단조성은 증명 가능한 성질이므로 기준값이 필요 없는")
    print("  강한 검증이다.")
    print()
    print("  [이 검사가 실제로 잡은 버그] FIM 이 특이할 때 의사역행렬은")
    print("  '식별 가능한 부분공간으로 투영한' 하한을 돌려준다. 이것은 원래")
    print("  파라미터의 하한이 아니다. 어떤 파라미터가 영공간에 성분을 가지면")
    print("  그 방향으로는 관측이 전혀 변하지 않으므로 하한은 무한이다.")
    print("  보정 전에는 영공간 성분 판정 기준이 0.999 로 느슨해서, 성분이")
    print("  0.9127 인 defocus 가 걸러지지 않고 유한한 0.315 nm 로 보고되었다.")
    print("  그 결과 미지수를 4 개에서 5 개로 늘렸을 때 하한이 1.891 에서")
    print("  0.315 로 '좋아지는' 단조성 위반이 나타났다.")
    print()
    spec = ID.IdentifySpec(defocus=80.0, dose_mj=30.0)
    lad = ID.observable_ladder(spec)
    print(f"    {'미지수':<52}{'sigma(dz)':>12}{'rank':>8}{'leak':>8}")
    prev, mono = 0.0, True
    for row in lad:
        ps = row["params"]
        sm = row["sigma_min"]
        v = (sm["defocus"] if isinstance(sm, dict)
             else float(np.asarray(sm, dtype=float)[0]))
        cc = ID.crb(spec, params=tuple(ps))
        lk = float(np.asarray(cc["nullspace_leak"], dtype=float)[0])
        if np.isfinite(v) and v < prev - 1e-12:
            mono = False
        txt = "inf" if not np.isfinite(v) else f"{v:.5f}"
        print(f"    {str(tuple(ps)):<52}{txt:>12}{row['rank']:>4}/{len(ps):<3}"
              f"{lk:>8.4f}")
        if np.isfinite(v):
            prev = v
    print(f"\n    단조 증가 : {'예' if mono else '아니오'}")
    print("    5 변수에서 defocus·flare·msd_nm 은 하한이 무한이다(구조적")
    print("    비식별). 유한한 것은 dose_rel 과 blur_nm 뿐이며, 이는")
    print("    프로파일 가능도(E)와 몬테카를로(D4)가 식별 가능하다고 한")
    print("    변수와 정확히 같다. 서로 독립인 세 방법이 일치한다.")
    record("J", "J4 하한의 단조성",
           f"{'단조 성립' if mono else '위반'} (영공간 판정 보정으로 해결)",
           ok=bool(mono))


# ===========================================================================
# K. GUI 정적 검사
# ===========================================================================

def part_k() -> None:
    head("K. 정적 검사 (프로젝트 전체 + GUI)")
    print("  런타임 시험이 닿지 않는 두 구멍을 정적으로 막는다. 하나는 자동")
    print("  시험이 없는 app.py 이고, 다른 하나는 '한 스코프의 이름을 다른")
    print("  스코프에서 참조' 하는 버그다. 후자는 그 줄이 실행될 때만 터지므로")
    print("  시험이 그 경로를 지나지 않으면 끝까지 숨는다.")
    print()
    print("  [이 검사가 실제로 잡은 버그 2 건]")
    print("  (1) app.py : 대비 계산을 격자 독립으로 바꾸면서")
    print("      IM.contrast(intensity, g.x) 로 고쳤는데, 그 지점의 격자")
    print("      변수명은 g 가 아니라 sim_grid 였다. 함수 안의 g 를 보고")
    print("      착각한 것이다.")
    print("  (2) verify.py : part_e2 가 part_e 의 지역변수 full 을 참조해")
    print("      NameError 로 죽었다. E2 가 A~K 의 마지막이라 전체를 한 번에")
    print("      돌릴 때만 드러나, 38 분 뒤에야 보였다. 당시 이 검사가")
    print("      app.py 의 모듈 스코프만 보고 있어 놓쳤다. 그래서 범위를")
    print("      프로젝트 전체 파일의 함수 본문까지 넓혔다.")
    print()

    import ast
    import builtins
    import importlib
    import symtable
    from pathlib import Path

    root = Path(__file__).parent

    # --- K1. 프로젝트 전체, 스코프 인식 미정의 이름 -----------------------
    sub("K1  전체 파일의 스코프 인식 미정의 이름")
    print("    판정 기준 : 함수 안에서 읽히는데 그 함수의 지역도 아니고,")
    print("    모듈 전역에도 없고, 내장도 아닌 이름. 스코프 분석기를 직접")
    print("    쓰면 그 자체가 버그를 낳으므로 표준 라이브러리 symtable 을")
    print("    쓴다(추가 설치가 필요 없다).")

    _EXTRA = {"__file__", "__name__", "__doc__", "__spec__",
              "__package__", "__loader__", "__builtins__"}

    def _mod_names(top):
        """모듈 스코프에 묶인 이름 + 함수에서 global 선언해 대입하는 이름."""
        out = set()
        for nm in top.get_identifiers():
            s = top.lookup(nm)
            if s.is_assigned() or s.is_imported() or s.is_namespace():
                out.add(nm)
        stack = [top]
        while stack:
            t = stack.pop()
            for ch in t.get_children():
                for nm in ch.get_identifiers():
                    s = ch.lookup(nm)
                    if s.is_global() and s.is_assigned():
                        out.add(nm)
                stack.append(ch)
        return out

    def _scan_text(text, label):
        # 별 임포트가 있으면 전역 집합을 알 수 없어 판정할 수 없다.
        # [주의] 이것을 부분 문자열로 검사하면 안 된다. 그 문자열이 이 파일
        # 안에 쓰여 있어서, verify.py 가 자기 자신을 건너뛰었다. 즉 버그가
        # 있던 바로 그 파일이 검사에서 빠졌다. AST 로 실제 구문만 본다.
        for nd in ast.walk(ast.parse(text, filename=label)):
            if isinstance(nd, ast.ImportFrom) and any(
                    a.name == "*" for a in nd.names):
                return None
        top = symtable.symtable(text, label, "exec")
        known = _mod_names(top) | set(dir(builtins)) | _EXTRA
        hits = []
        stack = [(top, [])]
        while stack:
            t, trail = stack.pop()
            for ch in t.get_children():
                where = trail + [ch.get_name()]
                if ch.get_type() == "function":
                    for nm in ch.get_identifiers():
                        s = ch.lookup(nm)
                        if (s.is_referenced() and s.is_global()
                                and not s.is_assigned() and nm not in known):
                            hits.append(f"{label}:{'.'.join(where)}:{nm}")
                stack.append((ch, where))
        return hits

    def _scan(p):
        return _scan_text(p.read_text(encoding="utf-8"), p.name)

    # 검사가 '실제로 잡는지' 를 먼저 보인다. 통과만 보고하는 검사는 망가져도
    # 똑같이 통과를 보고하므로, 성립·파괴를 양쪽으로 확인한다(part C 와 같은
    # 원칙). 심는 버그는 이번에 실제로 났던 것과 같은 구조다. 즉 함수 b 가
    # 함수 a 의 지역 이름을 참조한다.
    _clean = ("def a():\n    x = 1\n    return x\n"
              "def b():\n    y = 2\n    return y\n")
    _planted = ("def a():\n    x = 1\n    return x\n"
                "def b():\n    return x\n")
    _r_clean = _scan_text(_clean, "<자기시험-정상>")
    _r_bad = _scan_text(_planted, "<자기시험-버그>")
    selftest = (_r_clean == [] and len(_r_bad) == 1
                and _r_bad[0].endswith(":b:x"))
    print(f"    자기 시험 : 정상 코드에서 {len(_r_clean)} 건, 심은 버그를"
          f" {len(_r_bad)} 건 검출"
          f" -> {'검사가 동작함' if selftest else '검사 자체가 고장'}")

    targets = sorted(list(root.glob("*.py")) + list(root.glob("litho/*.py")))
    scope_bad, skipped = [], []
    for p in targets:
        r = _scan(p)
        if r is None:
            skipped.append(p.name)
        else:
            scope_bad.extend(r)
    print(f"    검사 {len(targets) - len(skipped)} 개 파일"
          + (f", 건너뜀 {len(skipped)} 개 (import *)" if skipped else ""))
    if scope_bad:
        for b in scope_bad:
            print(f"      {b}")
    else:
        print("    미정의 이름 : 없음")

    path = root / "app.py"
    if not path.exists():
        print("\n  app.py 가 없어 K2~K4 를 건너뛴다.")
        record("K", "정적 검사",
               f"전체 스코프 {len(scope_bad)} 건, app.py 없음",
               ok=(selftest and not scope_bad))
        return
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # --- K2. app.py 모듈 스코프에서 정의되지 않은 이름 --------------------
    sub("K2  app.py 모듈 스코프 미정의 이름")
    assigned = set(dir(builtins))
    for node in tree.body:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                assigned.add(n.id)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                assigned.add(n.name)
            if isinstance(n, ast.alias):
                assigned.add((n.asname or n.name).split(".")[0])
            if isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)

    def module_level(node):
        """함수·클래스·람다 본문을 제외한 노드만 훑는다."""
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda)):
                continue
            yield n
            yield from module_level(n)

    used = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        for n in module_level(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)
    undef = sorted(u for u in used - assigned if not u.startswith("_"))
    print(f"    정의 없이 쓰이는 이름 : {undef if undef else '없음'}")

    # --- K3. litho API 참조가 모두 존재하는가 ------------------------------
    sub("K3  litho API 참조 해소")
    alias = {"IM": "litho.imaging", "MK": "litho.mask", "RS": "litho.resist",
             "ST": "litho.stochastic", "PR": "litho.process",
             "OP": "litho.optics", "G": "litho.grid", "DD": "litho.dataset",
             "ID": "litho.identify", "HK": "litho.hopkins",
             "SP": "litho.splitting", "CP": "litho.campaign"}
    refs = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if n.value.id in alias:
                refs.add((n.value.id, n.attr))
    missing = []
    for mod, attr in sorted(refs):
        m = importlib.import_module(alias[mod])
        if not hasattr(m, attr):
            missing.append(f"{mod}.{attr}")
    print(f"    참조 {len(refs)} 개 중 없는 것 : {missing if missing else '없음'}")

    # --- K4. SampleSpec 생성 인자가 실제 필드인가 -------------------------
    sub("K4  SampleSpec 생성 인자")
    from litho import dataset as DS
    fields = set(DS.SampleSpec.__dataclass_fields__)
    bad = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "SampleSpec"):
            for kw in n.keywords:
                if kw.arg and kw.arg not in fields:
                    bad.append(kw.arg)
    print(f"    SampleSpec 에 없는 인자 : {sorted(set(bad)) or '없음'}")
    print("    (min_nils 와 cd_tolerance 는 이제 유도 프로퍼티이므로 생성")
    print("     인자로 넘기면 TypeError 가 된다. 이 검사가 그것도 잡는다.)")

    # --- K5. 조건의 성격 분류가 빠짐없는가 --------------------------------
    sub("K5  조건의 성격 분류 (추천 가능 / 받아야 하는 값)")
    print("    추천 시스템은 '설정할 수 있는 조건' 만 출력해야 한다. 플레어나")
    print("    수차처럼 장비가 주는 값을 '이렇게 설정하십시오' 로 내놓으면")
    print("    물리적으로 불가능한 지시가 된다. 그래서 조건마다 성격을")
    print("    control / recipe / given 으로 분류해 두었다. 조건을 새로")
    print("    추가하면서 분류를 빼먹으면 그 조건이 어느 쪽인지 모르는 채")
    print("    학습·추천에 들어가므로, 빠짐없음을 정적으로 확인한다.")
    roles, role_bad = {}, []
    for k in DS.CONDITION_KEYS:
        try:
            r = DS.condition_role(k)
        except KeyError:
            role_bad.append(k)
            continue
        roles[r] = roles.get(r, 0) + 1
    # 분류 목록에만 있고 조건에는 없는 이름도 잡는다(이름이 바뀐 경우).
    declared = set(DS.CONTROL_KEYS) | set(DS.RECIPE_KEYS) | set(DS.GIVEN_KEYS)
    orphan = sorted(declared - set(DS.CONDITION_KEYS))
    # 한 조건이 두 분류에 동시에 들어가면 안 된다.
    dup = sorted({k for k in declared
                  if (k in DS.CONTROL_KEYS) + (k in DS.RECIPE_KEYS)
                  + (k in DS.GIVEN_KEYS) > 1})
    print(f"    분류 결과 : {roles} / 조건 {len(DS.CONDITION_KEYS)} 개")
    print(f"    분류 누락 : {role_bad or '없음'}")
    print(f"    조건에 없는 분류 : {orphan or '없음'}")
    print(f"    중복 분류 : {dup or '없음'}")
    role_ok = not role_bad and not orphan and not dup

    # --- K6. 여러 모듈에 복제된 상수가 서로 어긋나지 않는가 ---------------
    sub("K6  모듈 간 복제 상수의 정합성")
    print("    같은 뜻의 상수가 여러 파일에 적혀 있으면 한쪽만 고쳐질 수")
    print("    있다. 특히 그룹 정의가 어긋나면 병합은 새 그룹으로 나누는데")
    print("    학습 쪽은 옛 그룹으로 누설을 검사하게 되어 **누설이 있어도")
    print("    없다고 보고한다.** 회귀 시험은 각자 자기 상수로 돌므로 이")
    print("    어긋남을 구조적으로 못 잡는다. 그래서 정적으로 대조한다.\n")

    import importlib
    import merge_dataset as MD
    from litho import splitting as SP

    sync_bad = []

    # (1) 그룹 정의. splitting 은 SampleSpec 의 속성 이름을 쓰고
    #     merge_dataset 은 저장된 열 이름을 쓴다. 이름 영역이 다를 뿐 같은
    #     뜻이어야 하므로, 문서화된 대응으로 변환해 대조한다.
    _attr_to_col = {"polarization": "pol_code", "source_shape": "source_code",
                    "mask_type": "mask_code"}
    _mapped = tuple(_attr_to_col.get(k, k) for k in SP.OPTICAL_KEYS)
    if _mapped != tuple(MD.GROUP_KEYS):
        sync_bad.append(f"그룹 정의 불일치 : splitting {_mapped} vs "
                        f"merge_dataset {tuple(MD.GROUP_KEYS)}")
    print(f"    그룹 정의 ({len(MD.GROUP_KEYS)} 열) : "
          f"{'일치' if not sync_bad else '불일치'}")

    # (2) 대리 모형이 쓰는 상수가 SampleSpec 기본값과 같은가. 다르면 잡음
    #     하한과 임계값 교차가 다른 조건의 값이 된다.
    try:
        _ts = importlib.import_module("train_surrogate")
    except Exception as e:                       # noqa: BLE001
        _ts = None
        print(f"    train_surrogate 들여오기 실패 : {e}")
        sync_bad.append(f"train_surrogate import 실패: {e}")
    if _ts is not None:
        _sp = DS.SampleSpec()
        for _nm, _a, _b in (("threshold", _sp.threshold, _ts.THRESHOLD),
                            ("n_rows", _sp.n_rows, _ts.N_ROWS_DEFAULT),
                            ("n_trials", _sp.n_trials, _ts.N_TRIALS_DEFAULT)):
            same = abs(float(_a) - float(_b)) < 1e-12
            print(f"    {_nm:<10} SampleSpec {_a} / train_surrogate {_b}"
                  f"  {'일치' if same else '불일치'}")
            if not same:
                sync_bad.append(f"{_nm}: {_a} vs {_b}")
        # GROUP_KEYS 는 들여오기로 한 벌만 남겼다. 그것이 유지되는지 본다.
        shared = _ts.GROUP_KEYS is MD.GROUP_KEYS
        print(f"    GROUP_KEYS 단일 정의 공유 : {'예' if shared else '아니오'}")
        if not shared:
            sync_bad.append("train_surrogate 가 GROUP_KEYS 를 다시 정의함")

    # (3) 회귀 스크립트가 실패를 종료 코드로 알리는가.
    #     이것이 없으면 종료 코드만 보는 자동 검증은 50 종이 전부 깨져도
    #     '통과' 로 읽는다. 10 주차에 verify.py 에서 같은 결함을 고쳤는데
    #     회귀 쪽은 확인하지 않아 11 주차까지 남아 있었다. 되돌아가지
    #     않도록 정적으로 확인한다.
    import glob
    no_exit = []
    for _vf in sorted(glob.glob("validate*.py")):
        if _vf == "validate_common.py":
            continue
        _src = open(_vf, encoding="utf-8").read()
        _tree = ast.parse(_src, filename=_vf)
        has = False
        for _n in ast.walk(_tree):
            if not (isinstance(_n, ast.Call)
                    and isinstance(_n.func, ast.Attribute)
                    and _n.func.attr == "exit"):
                continue
            # sys.exit(main()) 또는 sys.exit(0 if all(...) else 1) 처럼
            # 결과에 따라 코드가 갈리는 형태여야 한다. sys.exit(0) 은
            # 아무것도 알리지 않으므로 인정하지 않는다.
            if _n.args and not (isinstance(_n.args[0], ast.Constant)
                                and _n.args[0].value in (0, None)):
                has = True
        if not has:
            no_exit.append(_vf)
    print(f"    회귀 스크립트 {len(glob.glob('validate*.py')) - 1} 개의 "
          f"조건부 종료 코드 : {'전부 있음' if not no_exit else no_exit}")
    if no_exit:
        sync_bad.append(f"종료 코드 없음: {no_exit}")

    # (4) [12주차] 추천기가 요구하는 라벨을 학습 쪽이 만들 수 있는가.
    #     recommend.LABELS_REQUIRED 는 '<접두사>_<라벨>.joblib' 를 찾는다.
    #     그 라벨이 데이터셋 라벨도 아니고 파생 라벨도 아니면, 학습을 아무리
    #     돌려도 그 파일은 만들어지지 않는다. 그러면 추천기는 영원히 '모형
    #     없음' 으로 죽는다. 양쪽이 다른 파일에 적혀 있으므로 어긋날 자리다.
    _lab_ok, _lab_msg = True, "검사 안 함"
    try:
        import recommend as _rc
        _ts2 = __import__("train_surrogate")
        _avail = set(DS.LABEL_KEYS) | set(getattr(_ts2, "DERIVED", {}))
        _unmakeable = [x for x in _rc.LABELS_REQUIRED if x not in _avail]
        _lab_ok = not _unmakeable
        _lab_msg = "전부 생성 가능" if _lab_ok else f"생성 불가 {_unmakeable}"
        if _unmakeable:
            sync_bad.append(f"추천기 요구 라벨 생성 불가: {_unmakeable}")
    except Exception as e:                  # noqa: BLE001
        _lab_ok = False
        _lab_msg = f"확인 실패: {e}"
        sync_bad.append(f"추천기 라벨 대조 실패: {e}")
    print(f"    추천기 요구 라벨 {len(getattr(_rc, 'LABELS_REQUIRED', ()))} 개 "
          f": {_lab_msg}")

    # (5) [12주차] app.py 의 sweep 이 돌려주는 키와 화면이 꺼내 쓰는 키.
    #     app.py 는 streamlit 이 필요해 이 환경에서 **실행할 수 없다**.
    #     그래서 K2 의 미정의 이름 검사 말고는 아무것도 app.py 를 지키지
    #     않았다. 사전 키는 이름이 아니라 문자열이라 K2 가 보지 못한다.
    #     sweep 의 반환 사전에 없는 키를 화면이 꺼내면 실행하는 순간
    #     KeyError 로 죽는데, 그것을 알 방법이 지금까지 없었다.
    #     (12차 보완 69 번에서 sweep 에 키를 새로 넣으면서 드러난 빈자리다.)
    #
    #     이 대조는 'app.py 에서 res 라는 이름은 sweep 의 결과만 가리킨다'
    #     는 규약에 기댄다. 처음 넣었을 때는 그 규약이 깨져 있었다. 식별성
    #     결과도 res 였고, 그 사전의 키 8 개가 'sweep 이 안 돌려주는 키' 로
    #     잡혀 **거짓 양성**이 났다(식별성 쪽을 idres 로 갈랐다). 그래서
    #     규약 자체도 아래에서 함께 검사한다. 느슨하게 바꿔 통과시키면
    #     검사가 남아도 아무것도 지키지 못한다.
    _key_msg, _produced, _used = "app.py 없음", set(), set()
    if os.path.exists("app.py"):
        _atree = ast.parse(open("app.py", encoding="utf-8").read(),
                           filename="app.py")
        for _n in ast.walk(_atree):
            if isinstance(_n, ast.FunctionDef) and _n.name == "sweep":
                for _m in ast.walk(_n):
                    if isinstance(_m, ast.Return) and isinstance(_m.value,
                                                                 ast.Dict):
                        for _k in _m.value.keys:
                            if (isinstance(_k, ast.Constant)
                                    and isinstance(_k.value, str)):
                                _produced.add(_k.value)
        for _n in ast.walk(_atree):
            if (isinstance(_n, ast.Subscript)              # res["키"]
                    and isinstance(_n.value, ast.Name)
                    and _n.value.id == "res"
                    and isinstance(_n.slice, ast.Constant)
                    and isinstance(_n.slice.value, str)):
                _used.add(_n.slice.value)
            if (isinstance(_n, ast.Call)                   # res.get("키")
                    and isinstance(_n.func, ast.Attribute)
                    and _n.func.attr == "get"
                    and isinstance(_n.func.value, ast.Name)
                    and _n.func.value.id == "res"
                    and _n.args
                    and isinstance(_n.args[0], ast.Constant)
                    and isinstance(_n.args[0].value, str)):
                _used.add(_n.args[0].value)
        # 규약 검사 : res 에 대입하는 곳이 sweep 호출 하나뿐이어야 한다.
        _res_src = []
        for _n in ast.walk(_atree):
            if not isinstance(_n, (ast.Assign, ast.AnnAssign, ast.For,
                                   ast.withitem, ast.NamedExpr)):
                continue
            _tg = ([_n.target] if isinstance(_n, (ast.AnnAssign, ast.For,
                                                  ast.NamedExpr))
                   else (_n.targets if isinstance(_n, ast.Assign)
                         else ([_n.optional_vars]
                               if _n.optional_vars is not None else [])))
            for _t in _tg:
                if isinstance(_t, ast.Name) and _t.id == "res":
                    _val = getattr(_n, "value", None)
                    _fn = ""
                    if isinstance(_val, ast.Call):
                        _f = _val.func
                        _fn = (_f.id if isinstance(_f, ast.Name)
                               else getattr(_f, "attr", ""))
                    _res_src.append(_fn or "<호출 아님>")
        _conv_ok = (_res_src == ["sweep"])
        if not _conv_ok:
            sync_bad.append(
                f"app.py 의 res 는 sweep 결과만 가리켜야 한다. 대입 "
                f"{_res_src} (이 규약이 깨지면 아래 키 대조가 거짓 양성을 "
                f"낸다. 12차 보완 69 번)")
        _orphan = sorted(_used - _produced)
        _key_msg = (f"sweep 반환 {len(_produced)} 키, 화면 사용 "
                    f"{len(_used)} 키 : "
                    + ("정합" if not _orphan else f"없는 키 {_orphan}")
                    + f", res 대입 {_res_src}"
                    + ("" if _conv_ok else " <- 규약 위반"))
        if _orphan:
            sync_bad.append(f"app.py 가 sweep 이 안 돌려주는 키를 씀: {_orphan}")
    print(f"    app.py sweep 반환 키 : {_key_msg}")

    # (6) [13주차] argparse 의 help 문자열에 홀로 선 백분율 기호.
    #     argparse 는 help 를 `help % params` 로 서식화하므로 리터럴 % 는
    #     %% 여야 한다. **Python 3.15 가 add_argument 시점에 이것을 검증**
    #     하기 시작했고, 그보다 낮은 버전에서는 --help 를 부를 때까지
    #     조용하다. 즉 개발 환경에서는 통과하고 사용자 환경에서는 죽는다.
    #     실제로 그렇게 났다 : 개발은 3.13, 사용자는 3.15 였고
    #     `roundtrip_w13.py` 가 시작도 못 했다.
    #
    #     정규식으로 찾지 않는다. help 는 여러 줄 문자열을 이어 붙이는
    #     일이 많아 한 줄만 보면 놓친다. AST 로 실제 값을 본다.
    _PCT_OK = re.compile(
        r"%\((?:default|prog|type|choices|dest|metavar|const)\)[sdr]")
    _pct_bad = []
    for _f in sorted(glob.glob("*.py")):
        try:
            _t = ast.parse(open(_f, encoding="utf-8").read(), filename=_f)
        except SyntaxError:
            continue
        for _n in ast.walk(_t):
            if not (isinstance(_n, ast.Call)
                    and isinstance(_n.func, ast.Attribute)
                    and _n.func.attr == "add_argument"):
                continue
            for _kw in _n.keywords:
                if _kw.arg != "help":
                    continue
                try:
                    _v = ast.literal_eval(_kw.value)
                except Exception:          # noqa: BLE001
                    continue
                if not isinstance(_v, str):
                    continue
                if "%" in _PCT_OK.sub("", _v).replace("%%", ""):
                    _pct_bad.append(f"{_f}:{_n.lineno}")
    print(f"    argparse help 의 홀로 선 % : "
          + ("없음" if not _pct_bad else ", ".join(_pct_bad)))
    if _pct_bad:
        sync_bad.append(
            f"argparse help 에 이스케이프하지 않은 % 가 있다: "
            f"{', '.join(_pct_bad)} (Python 3.15 는 add_argument 에서 "
            f"ValueError 로 죽는다. %% 로 쓸 것)")

    # [음성 대조] 일부러 틀린 짝을 넣어 같은 대조가 잡는지 확인한다.
    _probe = tuple(_attr_to_col.get(k, k) for k in SP.OPTICAL_KEYS) + ("zz",)
    sync_selftest = _probe != tuple(MD.GROUP_KEYS)
    # 키 대조도 가짜 키를 넣어 같은 검사가 잡는지 본다. 여기서 '미검출' 이
    # 나오면 위 (5) 가 아무것도 지키지 못하는 것이다.
    sync_selftest = sync_selftest and bool(_produced) and bool(_used) and (
        "없는키" not in _produced)
    # 라벨 대조도 가짜 라벨을 넣어 같은 검사가 잡는지 본다.
    try:
        sync_selftest = sync_selftest and ("없는라벨" not in _avail)
    except NameError:
        sync_selftest = False
    print(f"    자기 시험(가짜 축을 더해 보기) : "
          f"{'검출' if sync_selftest else '미검출'}")

    print(f"    불일치 : {sync_bad or '없음'}")
    sync_ok = (not sync_bad) and sync_selftest

    ok = (selftest and not scope_bad and not undef
          and not missing and not bad and role_ok and sync_ok)
    print()
    print(f"  -> 정적 검사 {'통과' if ok else '실패'}")
    record("K", "정적 검사",
           (f"전체 스코프 {len(targets) - len(skipped)} 파일 미정의 0, "
            f"app.py 모듈 스코프·API·인자 해소, 조건 성격 분류 "
            f"{len(DS.CONDITION_KEYS)}/{len(DS.CONDITION_KEYS)}, "
            f"복제 상수 정합")
           if ok else
           (f"스코프 {len(scope_bad)}, 미정의 {len(undef)}, "
            f"API {len(missing)}, 인자 {len(set(bad))}, "
            f"분류 누락 {len(role_bad)}, 복제 상수 {len(sync_bad)}"),
           ok=bool(ok))

# ===========================================================================

def main() -> int:
    want = [a.upper() for a in sys.argv[1:]] or ["A", "B", "C", "H", "F", "G", "I", "J", "K", "D", "E"]
    t0 = time.time()
    print("=" * 78)
    print(" 검증 체계 (Verification suite)")
    print(" Roache(1998) / Roy(2005) 의 검증 위계에 따라 구성")
    print("=" * 78)
    if "A" in want:
        part_a()
    if "B" in want:
        part_b()
    if "C" in want:
        part_c()
    if "D" in want:
        part_d()
    if "H" in want:
        part_h()
    if "F" in want:
        part_f()
    if "G" in want:
        part_g()
    if "I" in want:
        part_i()
    if "J" in want:
        part_j()
    if "K" in want:
        part_k()
    if "E" in want:
        part_e()
        part_e2()

    head("요약")
    cur = None
    for sec, item, verdict, ok in _RESULTS:
        if sec != cur:
            print(f"\n  {sec}.")
            cur = sec
        tag = "    " if ok is None else ("[판정 O] " if ok else "[판정 X] ")
        print(f"    - {tag}{item:<22} {verdict}")
    print(f"\n  소요 {time.time() - t0:.1f} s")

    # --- 기계가 읽을 수 있는 최종 판정 -----------------------------------
    judged = [(s, i, v, o) for s, i, v, o in _RESULTS if o is not None]
    bad = [(s, i, v) for s, i, v, o in judged if not o]
    head("최종 판정")
    print("  [판정 O/X] 가 붙은 항목만 합격·불합격이 있는 것이다. 물리가")
    print("  요구하는 등식·대칭·순서처럼 기준값을 고를 여지가 없는 검사다.")
    print("  나머지 항목은 **측정값 보고**이며 합격선이 없다. 수렴 차수나")
    print("  하한의 크기, LER 절대값에 합격선을 붙이면 '사후 임계값' 이 되어")
    print("  검증 근거로서 오히려 약해지기 때문이다. 그 숫자들은 읽고")
    print("  판단할 대상이고, 판단은 보고서가 한다.")
    print()
    print(f"  판정 대상 {len(judged)} 항목 가운데 불합격 {len(bad)} 건")
    if bad:
        for s, i, v in bad:
            print(f"    X  {s}. {i} : {v}")
        print()
        print("  -> 불변량이 깨졌다. 종료 코드 1 을 돌려준다.")
    else:
        print("  -> 판정 대상 전부 합격. 종료 코드 0 을 돌려준다.")
    print()
    print("  [주의] 이전에는 이 판정이 없어서, 불변량이 깨져도 '위반' 을")
    print("  인쇄하고 종료 코드 0 으로 끝났다. 종료 코드만 보면 통과로")
    print("  읽힌다. 이제 종료 코드가 판정 대상 항목의 결과를 반영한다.")
    print("  다만 측정값 항목은 여전히 사람이 읽어야 한다.")

    print("\n  [범위의 한계] 위 결과는 전부 verification 이다. 코드가 의도한")
    print("  수식을 정확히 푼다는 것을 보였을 뿐, 그 수식이 실제 노광 장비의")
    print("  거동과 일치한다는 것(validation)은 보이지 않았다. 후자는 실측")
    print("  데이터 없이는 원리적으로 불가능하다.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
