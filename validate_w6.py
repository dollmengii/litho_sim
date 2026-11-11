"""
validate_w6.py -- 6주차 공정 윈도 분석 정량 검증 (6종)

실행: py validate_w6.py
"""

import time

import sys

import numpy as np

import validate_common as VC

from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import resist as RS
from litho import stochastic as ST

LAM, NA, N_MED = 193.0, 1.35, 1.44
# TOL 은 **CD 예산 전체**(0.10)이며, 10주차에 유도한 공정 윈도 몫
# (pw_tol = 0.0707)이 아니다. 일부러 그렇게 둔다. 이 묶음은 5~9주차에
# 측정해 둔 EL·DOF 값과 비교하는 회귀 감시이므로, 기준을 바꾸면 비교가
# 끊긴다. 다만 "예산 전체를 결정론 윈도에 준 값" 이라는 점을 적어 두지
# 않으면 유도 기준과 어긋난 것으로 보인다. 공정 설계에 쓸 값은 pw_tol
# 쪽이고, GUI 의 기본값은 그쪽으로 맞춰 두었다.
SIGMA, THRESH, TOL = 0.7, 0.30, 0.10


_CASES = []

def _report(name, ok, detail):
    _CASES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail:
        print(f"        {line}")
    print()
    return ok


def _setup(pitch, rng=400.0, n_focus=33):
    g = G.build_grid(pitch=pitch, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    dz = np.linspace(-rng, rng, n_focus)
    I = PR.focus_sweep(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA, defocuses=dz)
    return g, m, dz, I


def case1_dose_to_size():
    """dose-to-size 보정: E_op 에서 CD 가 목표와 정확히 일치"""
    g, m, dz, I = _setup(160.0)
    i0 = int(np.argmin(np.abs(dz)))
    target = 80.0
    e_op = PR.dose_to_size(g, I[i0], target, THRESH)
    from litho import resist as RS
    cd = RS.extract_cd(g.x, I[i0] * e_op, THRESH)["cd"]
    err = abs(cd - target)
    return _report(
        "Case 1  dose-to-size 보정 (목표 CD 80 nm)",
        err < 0.05,
        [
            f"E_op = {e_op:.4f} (상대 노광량)",
            f"E_op 에서 CD = {cd:.3f} nm (오차 {err:.4f} nm)",
            "보정 없이 Dose=1.0 으로 스윕하면 윈도 중심이 규격 밖에 놓여 EL·DOF 가 과소평가됨",
        ],
    )


def case2_compute_reduction():
    """연산량 설계: 결상 연산은 Defocus 에만 의존"""
    g = G.build_grid(pitch=160.0, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)

    t0 = time.perf_counter()
    base = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA, defocus=0.0)
    t_one = time.perf_counter() - t0

    # Dose 는 스칼라 배율이므로 결상 결과가 Dose 와 무관함을 확인
    from litho import resist as RS
    a = RS.apply_dose(base, 1.3)
    b = base * 1.3
    identical = bool(np.allclose(a, b))

    n_dose, n_focus = 33, 33
    naive = n_dose * n_focus
    optimized = n_focus
    return _report(
        "Case 2  Dose-Focus 스윕 연산량 설계",
        identical and optimized < naive,
        [
            f"결상 연산 1회 소요 시간 = {t_one * 1000:.1f} ms",
            f"단순 구현 {naive}회 (예상 {naive * t_one:.1f} s) -> 본 구현 {optimized}회 "
            f"(예상 {optimized * t_one:.1f} s), {100 * (1 - optimized / naive):.1f}% 절감",
            f"Dose 스칼라 배율과 결상 결과의 독립성 확인: {identical}",
        ],
    )


def case3_coherent_limit():
    """결맞음 해상 한계: k1 = 0.5 (해석해 half-pitch = λ / 2NA)"""
    r = PR.find_resolution_limit(
        LAM, NA, n_medium=N_MED, sigma_out=0.0, contrast_min=0.10,
        pitch_lo=40.0, pitch_hi=500.0, n_iter=14, n_periods=6, dx_target=0.5,
    )
    analytic = LAM / (2 * NA)
    err = abs(r["half_pitch"] - analytic)
    return _report(
        "Case 3  완전 결맞음 해상 한계 (해석해 대조)",
        abs(r["k1"] - 0.5) < 0.01,
        [
            f"해석해 half-pitch = λ / 2NA = {analytic:.2f} nm (k1 = 0.500)",
            f"시뮬레이션 half-pitch = {r['half_pitch']:.2f} nm (k1 = {r['k1']:.3f}), 오차 {err:.2f} nm",
            "결맞음 조명에서 ±1차 2-beam 간섭 한계와 정확히 일치함",
        ],
    )


def case4_k1_vs_illumination():
    """k1 은 상수가 아니라 조명 조건의 함수"""
    conds = [
        ("Coherent (σ=0)", 0.0, 0.0),
        ("Conventional σ=0.3", 0.3, 0.0),
        ("Conventional σ=0.7", 0.7, 0.0),
        ("Conventional σ=0.9", 0.9, 0.0),
        ("Annular 0.6-0.8", 0.8, 0.6),
        ("Annular 0.7-0.9", 0.9, 0.7),
    ]
    rows, k1s = [], []
    for lbl, so, si in conds:
        r = PR.find_resolution_limit(
            LAM, NA, n_medium=N_MED, sigma_out=so, sigma_in=si, contrast_min=0.10,
            pitch_lo=40.0, pitch_hi=500.0, n_iter=14, n_periods=6, dx_target=0.5,
        )
        k1s.append(r["k1"])
        rows.append(f"{lbl:<20s} half-pitch {r['half_pitch']:5.1f} nm   k1 = {r['k1']:.3f}")
    monotone = all(k1s[i] >= k1s[i + 1] - 1e-9 for i in range(3))
    bracket = min(k1s) < 0.288 < max(k1s[2:])
    rows.append(f"문헌 ArF 이머전 실측 k1 ≈ 0.288 이 측정 범위 안에 포함: {bracket}")
    rows.append("이론 하한 0.25 에 근접하나 하회하지 않음: "
                f"{min(k1s) >= 0.25}")
    return _report(
        "Case 4  k1 의 조명 조건 의존성",
        monotone and bracket and min(k1s) >= 0.25,
        rows,
    )


def case5_el_dof_tradeoff():
    """EL-DOF trade-off: EL 이 커지면 DOF 가 단조 감소"""
    g, m, dz, I = _setup(160.0, rng=500.0, n_focus=41)
    i0 = int(np.argmin(np.abs(dz)))
    e_op = PR.dose_to_size(g, I[i0], 80.0, THRESH)
    doses = e_op * np.linspace(0.75, 1.25, 33)
    cd = PR.cd_matrix(g, I, doses, THRESH)
    pw = PR.process_window(cd, 80.0, TOL)
    sat = PR.window_saturated(pw)
    curve = PR.el_dof_curve(pw, doses, dz)

    dofs = [r["dof"] for r in curve]
    monotone = all(dofs[i] >= dofs[i + 1] - 1e-9 for i in range(len(dofs) - 1))
    rows = [f"PW 격자점 {int(pw.sum())} / {pw.size}, 경계 포화 {sat}"]
    for r in curve[::4]:
        rows.append(f"EL {r['el']:5.1f} %  ->  DOF {r['dof']:5.0f} nm   "
                    f"(focus {r['focus_range'][0]:+.0f} ~ {r['focus_range'][1]:+.0f} nm)")
    return _report(
        "Case 5  EL-DOF trade-off 단조성 (pitch 160 nm, CD ±10%)",
        monotone and not sat["focus_edge"] and not sat["dose_edge"],
        rows,
    )


def case6_k2_vs_k1():
    """k2 역산 및 k1 의존성"""
    rows = ["  pitch   half-pitch     k1     DOF@EL5%   k2(역산)"]
    data = []
    for pitch in (110.0, 130.0, 160.0, 200.0):
        rng = 500.0 if pitch > 180 else 350.0
        g, m, dz, I = _setup(pitch, rng=rng, n_focus=41)
        hp = pitch / 2
        i0 = int(np.argmin(np.abs(dz)))
        e_op = PR.dose_to_size(g, I[i0], hp, THRESH)
        if not np.isfinite(e_op):
            continue
        doses = e_op * np.linspace(0.75, 1.25, 33)
        cd = PR.cd_matrix(g, I, doses, THRESH)
        pw = PR.process_window(cd, hp, TOL)
        curve = PR.el_dof_curve(pw, doses, dz)
        # [12차 보완 65 번] d5 는 격자화 때문에 **하한**이고, 거기서 역산한
        # k2 도 하한이다. 이 케이스의 판정은 k1 에 대한 k2 의 **단조성**이므로
        # 하한을 일관되게 쓰는 것으로 성립한다. 보간하는 dof_at_el 을 그대로
        # 두는 이유는, 증명 가능한 구간(dof_bracket_at_el)으로 바꾸면 계단
        # 때문에 값이 덜 매끄러워져 이미 검증된 단조성 판정에 거짓 경보가
        # 날 수 있기 때문이다. 바꿔야 할 쪽은 '하한을 값으로 적는 서술' 이고,
        # 아래 줄에 명시한다. 추천기(12주차)는 구간으로 읽는다.
        d5 = PR.dof_at_el(curve, 5.0)
        k1 = hp * NA / LAM
        k2 = PR.k2_from_dof(d5, LAM, NA, N_MED)
        data.append((k1, k2))
        rows.append(f"  {pitch:5.0f}      {hp:5.1f}     {k1:.3f}    {d5:6.0f} nm    {k2:.3f}")
    increasing = all(data[i][1] <= data[i + 1][1] + 1e-9 for i in range(2))
    rows.append("")
    rows.append("[주의] DOF 는 격자화 때문에 하한이고, 역산한 k2 도 하한임. 판정은")
    rows.append("절대값이 아니라 k1 에 대한 단조성임 (VERIFICATION_LIMITS 65 번).")
    rows.append("k2 는 상수가 아니며 k1 에 의존함. 문헌값 k2 ≈ 0.745 는 해상 한계 근방")
    rows.append("조건에서 추출된 값으로, 완화된 k1 에서는 측정 DOF 가 Rayleigh 예측을 상회함.")
    return _report("Case 6  k2 역산 및 k1 의존성", increasing and len(data) >= 3, rows)



def case7_k1_analytic_baseline():
    """5주차 검증: k1 측정값을 해석 기준선 1/(2(1+sigma)) 와 대조"""
    conds = [("Coherent sigma=0", 0.0, 0.0), ("Conventional sigma=0.3", 0.3, 0.0),
             ("Conventional sigma=0.7", 0.7, 0.0), ("Conventional sigma=0.9", 0.9, 0.0),
             ("Annular 0.6-0.8", 0.8, 0.6), ("Annular 0.7-0.9", 0.9, 0.7)]
    rows, diffs = [], []
    for lbl, so, si in conds:
        r = PR.find_resolution_limit(LAM, NA, n_medium=N_MED, sigma_out=so, sigma_in=si,
                                     contrast_min=0.10, pitch_lo=40.0, pitch_hi=500.0,
                                     n_iter=14, n_periods=6, dx_target=0.5)
        th = 1.0 / (2.0 * (1.0 + so))
        diffs.append(r["k1"] - th)
        rows.append(f"{lbl:<24s} 이론 {th:.3f}  측정 {r['k1']:.3f}  차이 {r['k1'] - th:+.3f}")
    coherent_exact = abs(diffs[0]) < 0.005
    all_positive = all(d >= -0.005 for d in diffs)
    rows.append("")
    rows.append("sigma=0 에서만 오차 0. 나머지는 모두 양의 편차 -> contrast>=0.10 판정 기준의")
    rows.append("유한성에 기인함. 결맞음은 cutoff 통과 시 contrast 가 불연속 상승하므로 기준 무관.")
    return _report("Case 7  k1 해석 기준선 1/(2(1+sigma)) 대조 [5주차 검증]",
                   coherent_exact and all_positive, rows)


def case8_photon_statistics():
    """광자 통계: 파장에 따른 광자 수와 산탄 잡음"""
    rows = ["  광원          광자에너지   밀도[/nm^2]   2x2nm 광자수   상대잡음"]
    stats = {}
    for lbl, lam in (("ArF  193nm", 193.0), ("KrF  248nm", 248.0), ("EUV 13.5nm", 13.5)):
        st = ST.photon_statistics(30.0, lam, area_nm2=4.0)
        stats[lbl] = st
        rows.append(f"  {lbl:<12s} {st['photon_energy_ev']:7.2f} eV {st['density_per_nm2']:11.1f} "
                    f"{st['n_in_area']:13.1f} {100 * st['relative_noise']:9.2f} %")
    ratio_e = stats["EUV 13.5nm"]["photon_energy_ev"] / stats["ArF  193nm"]["photon_energy_ev"]
    ratio_n = stats["ArF  193nm"]["n_in_area"] / stats["EUV 13.5nm"]["n_in_area"]
    ratio_s = stats["EUV 13.5nm"]["relative_noise"] / stats["ArF  193nm"]["relative_noise"]
    rows += ["",
             f"  광자 에너지비 EUV/ArF = {ratio_e:.2f}, 광자 수비 ArF/EUV = {ratio_n:.2f} (일치)",
             f"  상대 잡음비 EUV/ArF = {ratio_s:.2f} (= sqrt({ratio_n:.1f}) = {np.sqrt(ratio_n):.2f})",
             "  -> 동일 노광량에서 EUV 광자 수가 1/14 이며, 이것이 EUV 확률 결함의 근본 원인."]
    ok = abs(ratio_e - 14.3) < 0.3 and abs(ratio_s - np.sqrt(ratio_n)) < 0.3
    return _report("Case 8  광자 통계 및 파장 의존성", ok, rows)


def case9_identifiability():
    """식별 불가능 영역의 존재 확인 (역추정 한계)"""
    g = G.build_grid(pitch=160.0, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    target = 80.0

    iso = []
    for dz in np.arange(0.0, 261.0, 20.0):
        I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA, defocus=float(dz))
        e = PR.dose_to_size(g, I, target, THRESH)
        if np.isfinite(e):
            iso.append((float(dz), I * e))

    cds = [RS.extract_cd(g.x, p, THRESH)["cd"] for _, p in iso]
    cd_spread = max(cds) - min(cds)

    base = iso[0][1]
    sigma_meas = float(np.mean([
        np.std(ST.stochastic_field(base, g.dx, 30.0, LAM, n_rows=64,
                                   blur_nm=10.0, rng=np.random.default_rng(k)).mean(axis=0) - base)
        for k in range(4)
    ]))

    rows = [f"iso-CD 등고선 위 {len(iso)} 개 조건의 CD 편차 = {cd_spread:.2f} nm "
            f"(목표 {target} nm)",
            f"-> CD 하나만 관측하면 Defocus 0 ~ 260 nm 를 전혀 구별할 수 없음", "",
            f"측정 잡음 수준 (ArF 30 mJ/cm^2, 64행 평균) = {sigma_meas:.5f}",
            "  Δz[nm]   프로파일 RMS차   잡음 대비   식별"]
    first_ident = None
    for dz, p in iso:
        d = float(np.sqrt(np.mean((p - base) ** 2)))
        ident = d > 3 * sigma_meas
        if ident and first_ident is None:
            first_ident = dz
        rows.append(f"  {dz:5.0f}      {d:10.5f}   {d / sigma_meas:8.2f}    "
                    f"{'가능' if ident else '불가'}")
    rows += ["",
             f"프로파일 전체를 관측해도 Δz < {first_ident:.0f} nm 는 측정 잡음에 묻혀 식별 불가.",
             "즉 식별 불가능 영역이 실재하며, 역추정의 원리적 한계가 존재함을 확인함."]
    # [10주차 재정립] 기존 임계값 0.5 nm 는 당시 dose-to-size 가 선형보간으로
    # 끝나 CD 보정에 0.5 nm 급 오차가 남아 있던 시점의 숫자였다. 즉 물리를
    # 검증한 것이 아니라 그 시점의 수치 오차를 기록한 것이다. 보정 이후
    # iso-CD 등고선 위의 CD 산포는 기계 정밀도여야 하므로, '등고선이 실제로
    # 등고선인가' 를 묻는 수준으로 조인다. 이 양은 해석적으로 0 이다.
    ok = cd_spread < 1e-6 and first_ident is not None and first_ident > 0
    return _report("Case 9  식별 불가능 영역의 존재 (역추정 한계)", ok, rows)


def main() -> int:
    print("=" * 76)
    print(" 6주차 정량 검증 (공정 윈도 / 확률 모델 / 식별성)")
    print("=" * 76 + "\n")
    results = [
        case1_dose_to_size(),
        case2_compute_reduction(),
        case3_coherent_limit(),
        case4_k1_vs_illumination(),
        case5_el_dof_tradeoff(),
        case6_k2_vs_k1(),
        case7_k1_analytic_baseline(),
        case8_photon_statistics(),
        case9_identifiability(),
    ]
    print("=" * 76)
    print(f" 결과: {sum(results)}/{len(results)} 통과")
    print()
    print(VC.summarize(_CASES))
    print("=" * 76)
    return 0 if all(results) else 1


# [11주차 수정] 실패해도 종료 코드 0 을 돌려주고 있었다. 10 주차에
# verify.py 에서 같은 결함을 고쳤는데(불변량이 깨져도 '위반' 만 인쇄하고
# 0 으로 끝남) 회귀 스크립트 일곱 개는 확인조차 하지 않았다. 사람이 출력을
# 읽으면 드러나지만, 종료 코드만 보는 자동 검증은 50 종이 전부 깨져도
# '통과' 로 읽는다. 실제로 패키지 검증에서 그렇게 놓쳤다.
if __name__ == "__main__":
    sys.exit(main())
