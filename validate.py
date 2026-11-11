"""
validate.py -- 시뮬레이터 정량 검증 (5주차 검증 케이스 6종)

실행: py validate.py
"""

import sys

import numpy as np

import validate_common as VC

from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import optics as OP
from litho import resist as RS


_CASES = []

def _report(name, ok, detail):
    _CASES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail:
        print(f"        {line}")
    print()
    return ok


def case1_open_frame():
    """정규화 검증: open frame 에서 I = 1.0"""
    g = G.build_grid(pitch=200.0, n_periods=4, dx_target=1.0)
    i = IM.aerial_image(g, MK.open_frame(g), wavelength=193.0, na=0.75, sigma_out=0.7)
    err = float(np.max(np.abs(i - 1.0)))
    return _report(
        "Case 1  open frame 정규화 (I = 1.0)",
        err < 1e-9,
        [f"I_mean = {i.mean():.9f}", f"최대 편차 = {err:.3e}"],
    )


def case2_k1_limit_region():
    """70nm/70nm L&S @ 193nm / NA 0.75 -> k1 = 0.272, 해상 한계 영역"""
    pitch, cd, lam, na = 140.0, 70.0, 193.0, 0.75
    p = OP.printability(pitch, k1=0.288, wavelength=lam, na=na)

    g = G.build_grid(pitch=pitch, n_periods=8, dx_target=0.5)
    m = MK.line_space(g, cd=cd)
    c_conv = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.7), g.x)
    c_ann = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.9, sigma_in=0.7), g.x)

    in_limit_region = 0.25 <= p["k1_required"] < 0.30
    ok = abs(p["k1_required"] - 0.272) < 0.002 and in_limit_region and c_conv < 0.05
    return _report(
        "Case 2  k1 = 0.272 해상 한계 영역 판정 (70/70nm, NA 0.75)",
        ok,
        [
            f"k1_required = {p['k1_required']:.4f}  -> 0.25~0.30 해상 한계 영역: {in_limit_region}",
            f"contrast: conventional s=0.7 = {c_conv:.4f} / annular 0.7-0.9 = {c_ann:.4f}",
            "-> 통상 조명으로는 미해상. 문헌상 k1 < 0.30 은 RET 또는 차세대 장비가 필요한 영역.",
        ],
    )


def case3_k1_theoretical_limit():
    """k1 = 0.25 이론 한계: half-pitch < lambda/(4*NA) 는 해상 불가"""
    lam, na, n_med, sigma = 193.0, 1.35, 1.44, 0.9
    hp_limit = OP.K1_THEORETICAL_LIMIT * lam / na
    rows = [f"k1=0.25 한계 half-pitch = {hp_limit:.1f} nm (lambda=193, NA=1.35, n=1.44)"]
    ok = True
    for factor, expect in ((1.20, True), (0.80, False)):
        hp = hp_limit * factor
        g = G.build_grid(pitch=2 * hp, n_periods=8, dx_target=0.25)
        i = IM.aerial_image(
            g, MK.line_space(g, duty=0.5), lam, na, n_medium=n_med, sigma_out=sigma
        )
        c = IM.contrast(i, g.x)
        resolved = c > 0.02
        ok = ok and (resolved == expect)
        rows.append(
            f"half-pitch {hp:5.1f} nm (k1={0.25 * factor:.3f}) -> contrast {c:.5f} "
            f"{'해상' if resolved else '미해상'}"
        )
    return _report("Case 3  k1 = 0.25 이론 한계", ok, rows)


def case4_coherent_error():
    """coherent 가정의 오류: lambda/NA = 210nm, pitch 180nm"""
    lam = 193.0
    na = lam / 210.0
    pitch = 180.0
    g = G.build_grid(pitch=pitch, n_periods=8, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)

    c_coh = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.0), g.x)
    c_par = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.7), g.x)
    first = next(
        o for o in IM.diffraction_orders(pitch, lam, na, sigma_out=0.7, max_order=1)
        if o["order"] == 1
    )
    ok = c_coh < 0.01 and c_par > 0.1
    return _report(
        "Case 4  coherent 가정의 오류 (lambda/NA=210nm, pitch=180nm)",
        ok,
        [
            f"pupil cutoff = {na / lam:.6f} /nm, 1차 회절광 = {first['f']:.6f} /nm",
            f"on-axis 통과 = {first['on_axis']}, sigma=0.7 포함 = {first['with_sigma']}",
            f"contrast: coherent = {c_coh:.5f} / sigma=0.7 = {c_par:.5f}",
            "-> coherent 가정 시 실제 구현되는 패턴이 미해상으로 오판정됨",
        ],
    )


def case5_illumination():
    """조명 조건 비교: conventional vs annular (pitch 160nm, NA 0.75)"""
    lam, na, pitch = 193.0, 0.75, 160.0
    g = G.build_grid(pitch=pitch, n_periods=8, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    c_coh = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.0), g.x)
    c_conv = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.7), g.x)
    c_ann = IM.contrast(IM.aerial_image(g, m, lam, na, sigma_out=0.8, sigma_in=0.6), g.x)
    gain = c_ann / c_conv if c_conv > 0 else float("inf")
    ok = c_ann > c_conv > c_coh
    return _report(
        "Case 5  조명 조건 비교 (pitch 160nm, half-pitch 80nm)",
        ok,
        [
            f"coherent          contrast = {c_coh:.4f}",
            f"conventional s=0.7 contrast = {c_conv:.4f}",
            f"annular 0.6-0.8    contrast = {c_ann:.4f}  ({gain:.1f}배)",
            "-> 동일 lambda/NA 에서 조명 형상만으로 contrast 가 결정적으로 달라짐",
        ],
    )


def case6_cd_extraction():
    """CD 추출 정확도: 해석해 대비 검증 + dose-to-size 임계값"""
    # (a) 해석해 검증: I(x) = 0.5*(1+cos(2*pi*(x-xc)/P))
    period, dx, threshold = 200.0, 1.0, 0.30
    x = np.arange(0, 4 * period, dx)
    xc = x[len(x) // 2]
    profile = 0.5 * (1.0 + np.cos(2 * np.pi * (x - xc) / period))
    cd_exact = 2.0 * (period / (2 * np.pi)) * np.arccos(2 * threshold - 1.0)
    cd_interp = RS.extract_cd(x, profile, threshold)["cd"]
    central = np.abs(x - xc) <= period / 2.0
    cd_naive = float(np.sum(profile[central] >= threshold) * dx)

    # (b) dose-to-size: 부분결맞음 에지의 임계 광강도
    g = G.build_grid(pitch=1000.0, n_periods=4, dx_target=1.0)
    i = IM.aerial_image(g, MK.line_space(g, cd=500.0), 193.0, 0.75, sigma_out=0.7)
    i_edge = float(np.interp(g.field_width / 2 + 250.0, g.x, i))
    cd_at_edge = RS.extract_cd(g.x, i, i_edge)["cd"]

    ok = abs(cd_interp - cd_exact) < 0.05 and 0.25 < i_edge < 0.5 and abs(cd_at_edge - 500.0) < 1.0
    return _report(
        "Case 6  CD 추출 정확도 및 dose-to-size",
        ok,
        [
            f"해석해 CD = {cd_exact:.4f} nm",
            f"선형보간 CD = {cd_interp:.4f} nm (오차 {abs(cd_interp - cd_exact):.4f} nm)",
            f"격자 단순계수 CD = {cd_naive:.1f} nm (오차 {abs(cd_naive - cd_exact):.3f} nm, dx={dx} nm 양자화)",
            f"dose-to-size 임계 광강도 = {i_edge:.4f} (결맞음 0.25 ~ 비결맞음 0.5 사이)",
            f"해당 임계값에서 CD = {cd_at_edge:.3f} nm (설계 500 nm)",
        ],
    )


def main() -> int:
    print("=" * 72)
    print(" 포토리소그래피 시뮬레이터 정량 검증 (5주차)")
    print("=" * 72 + "\n")
    results = [
        case1_open_frame(),
        case2_k1_limit_region(),
        case3_k1_theoretical_limit(),
        case4_coherent_error(),
        case5_illumination(),
        case6_cd_extraction(),
    ]
    print("=" * 72)
    print(f" 결과: {sum(results)}/{len(results)} 통과")
    print()
    print(VC.summarize(_CASES))
    print("=" * 72)
    return 0 if all(results) else 1


# [11주차 수정] 실패해도 종료 코드 0 을 돌려주고 있었다. 10 주차에
# verify.py 에서 같은 결함을 고쳤는데(불변량이 깨져도 '위반' 만 인쇄하고
# 0 으로 끝남) 회귀 스크립트 일곱 개는 확인조차 하지 않았다. 사람이 출력을
# 읽으면 드러나지만, 종료 코드만 보는 자동 검증은 50 종이 전부 깨져도
# '통과' 로 읽는다. 실제로 패키지 검증에서 그렇게 놓쳤다.
if __name__ == "__main__":
    sys.exit(main())
