"""
validate_w10.py -- 10주차 검증

이번 주 작업은 두 갈래다.
  (1) 9주차까지의 코드에서 검증이 잡아내지 못한 결함 보정
  (2) 벡터(편광) 결상, 위상반전마스크, 플레어, 스캔 동기화 오차,
      다이폴 조명, Zernike 수차 추가

검증 원칙은 이전 주차와 같다. 해석적으로 답이 정해진 조건, 또는 문헌에
보고된 관계를 기준으로 삼고 시뮬레이션 결과를 대조한다.

실행:  py validate_w10.py
"""

import sys

import numpy as np

import validate_common as VC

from litho import grid as G
from litho import mask as MK
from litho import imaging as IM
from litho import resist as RS
from litho import process as PR
from litho import stochastic as ST
from litho import dataset as DS

LAM, NA, N_MED, N_PR, SIG, TH = 193.0, 1.35, 1.44, 1.70, 0.7, 0.30

_results = []


_CASES = []

def report(name, passed, lines):
    _results.append(passed)
    tag = "PASS" if passed else "FAIL"
    _CASES.append((name, bool(passed)))
    print(f"[{tag}] {name}")
    for ln in lines:
        print(f"        {ln}")
    print()


def grid_for(pitch, sigma=SIG, na=NA):
    return G.build_grid(pitch=pitch, n_periods=8,
                        dx_target=G.nyquist_dx(LAM, na, sigma))


# ---------------------------------------------------------------- Case 1
def case1_na_guard():
    """NA >= n 은 물리적으로 불가능하다. 결상 함수가 이를 차단해야 한다.

    차단하지 않으면 pupil 바깥 띠에서 kz^2 < 0 이 되어 defocus 위상이
    0 으로 눌리고, 초점 응답이 비단조가 된다 (9주차까지의 결함).
    """
    g = grid_for(200.0)
    m = MK.line_space(g, duty=0.5)
    lines, ok = [], True

    try:
        IM.aerial_image(g, m, LAM, 1.35, n_medium=1.0, sigma_out=SIG)
        ok = False
        lines.append("NA=1.35, n=1.00 조합이 차단되지 않음")
    except ValueError as e:
        lines.append(f"NA=1.35, n=1.00 차단됨 : {str(e)[:52]}...")

    # 정상 조합은 통과해야 한다
    I = IM.aerial_image(g, m, LAM, 1.35, n_medium=1.44, sigma_out=SIG, defocus=100.0)
    lines.append(f"NA=1.35, n=1.44 정상 수행 : contrast {IM.contrast(I, g.x):.4f}")

    # 초점 응답이 단조 감소하는지 (비물리 조합에서는 단조성이 깨졌음)
    cs = [IM.contrast(IM.aerial_image(g, m, LAM, 1.35, n_medium=1.44,
                                      sigma_out=SIG, defocus=float(d)), g.x)
          for d in (0, 50, 100, 150, 200)]
    mono = all(cs[i] >= cs[i + 1] - 1e-9 for i in range(len(cs) - 1))
    ok &= mono
    lines.append("초점 응답 contrast : " + "  ".join(f"{c:.4f}" for c in cs))
    lines.append(f"단조 감소 = {mono}")
    report("Case 1  NA >= 굴절률 조합 차단 및 초점 응답 단조성", ok, lines)


# ---------------------------------------------------------------- Case 2
def case2_vector_vs_analytic():
    """2광선 간섭 조건에서 TM/TE 대비가 해석식 cos(2*theta) 와 일치하는가.

    광원점을 h = lambda/(2*NA*pitch) 에 두면 0 차와 -1 차가 pupil 중심에
    대해 대칭으로 놓이고, +1 차는 3h > 1 이면 pupil 밖으로 벗어나 진성
    2광선 조건이 된다. 이때 두 광선 사이각은 2*theta 이며
    sin(theta) = lambda / (2*n*pitch) 이다.
    """
    lines, errs = [], []
    lines.append(f"{'pitch':>7} {'광원점 h':>8} | {'측정 TM/TE':>11} {'해석 |cos2θ|':>12} {'오차':>9}")
    for pitch in (80.0, 100.0, 130.0, 160.0, 190.0, 210.0):
        h = LAM / (2 * NA * pitch)
        if not (0.0 < h <= 1.0 and 3 * h > 1.0):
            continue
        g = G.build_grid(pitch=pitch, n_periods=4,
                         dx_target=G.nyquist_dx(LAM, NA, 1.0))
        m = MK.line_space(g, duty=0.5)
        kw = dict(wavelength=LAM, na=NA, n_medium=N_MED, n_resist=N_MED,
                  sigma_out=1.0, source_shape=IM.SOURCE_DIPOLE,
                  pole_offset=h, pole_radius=0.012, n_source_grid=201)
        cte = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_TE, **kw), g.x)
        ctm = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_TM, **kw), g.x)
        if cte <= 1e-9:
            continue
        th = abs(IM.polarization_factor(NA, pitch, LAM, N_MED, IM.POL_TM))
        meas = ctm / cte
        errs.append(abs(meas - th))
        lines.append(f"{pitch:7.0f} {h:8.3f} | {meas:11.6f} {th:12.6f} {abs(meas-th):9.2e}")
    ok = bool(errs) and max(errs) < 1e-3
    lines.append("")
    lines.append(f"최대 오차 {max(errs):.2e} (허용 1e-3). 잔차는 유한한 극 반경에서 비롯됨.")
    lines.append("출처 : Flagello, Milster, Rosenbluth, J. Opt. Soc. Am. A 13(1), 53 (1996)")
    report("Case 2  벡터 결상의 해석 편광 계수 대조", ok, lines)


# ---------------------------------------------------------------- Case 3
def case3_scalar_is_te():
    """스칼라 모델은 '틀린 모델' 이 아니라 'TE 편광 가정 모델' 이다.

    저NA 에서 TE 벡터 결과가 스칼라에 수렴하는지, 고NA 에서 편차가
    얼마인지를 측정한다. 또 편광 가중을 어느 굴절률에서 계산하는가에
    따른 차이를 구간으로 제시한다.
    """
    lines, ok = [], True
    lines.append("open frame 정규화가 모든 편광에서 유지되는가")
    g0 = G.build_grid(pitch=200.0, n_periods=4, dx_target=G.nyquist_dx(LAM, NA, SIG))
    of = MK.open_frame(g0)
    for pol in IM.POLARIZATIONS:
        I = IM.aerial_image(g0, of, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                            sigma_out=SIG, polarization=pol)
        err = abs(float(np.mean(I)) - 1.0)
        ok &= err < 1e-9
        lines.append(f"  {pol:<13} <I> = {np.mean(I):.12f}  오차 {err:.1e}")

    lines.append("")
    lines.append(f"{'NA':>5} {'pitch':>7} | {'스칼라':>8} {'TE(n=1.44)':>11} {'TE(n=1.70)':>11} {'무편광':>8}")
    devs = []
    for na in (0.60, 0.90, 1.20, 1.35):
        pitch = 1.25 * LAM / na
        g = G.build_grid(pitch=pitch, n_periods=4,
                         dx_target=G.nyquist_dx(LAM, na, 0.9))
        m = MK.line_space(g, duty=0.5)
        kw = dict(wavelength=LAM, na=na, n_medium=N_MED, sigma_out=0.9)
        cs = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_SCALAR, **kw), g.x)
        c1 = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_TE,
                                         n_resist=N_MED, **kw), g.x)
        c2 = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_TE,
                                         n_resist=N_PR, **kw), g.x)
        cu = IM.contrast(IM.aerial_image(g, m, polarization=IM.POL_UNPOLARIZED,
                                         n_resist=N_PR, **kw), g.x)
        devs.append(1.0 - c1 / cs)
        lines.append(f"{na:5.2f} {pitch:7.1f} | {cs:8.4f} {c1:11.4f} {c2:11.4f} {cu:8.4f}")
    lines.append("")
    lines.append(f"스칼라 대비 TE 편차 : 최대 {100*max(devs):.1f} % (침지액 굴절률 기준, 보수적)")
    lines.append("레지스트 굴절률로 계산하면 편차가 더 작다. 박막 적층을 모델링하지")
    lines.append("않으므로 참값은 두 값 사이이며, 보고서에는 구간으로 기재한다.")
    ok &= max(devs) < 0.10
    report("Case 3  스칼라 모델과 TE 벡터 모델의 관계", ok, lines)


# ---------------------------------------------------------------- Case 4
def case4_att_psm():
    """감쇠형 위상반전마스크가 패턴 경사를 개선하는가.

    차광부가 세기 6%, 위상 180도로 투과하면 경계에서 상쇄간섭이 일어나
    NILS 가 올라간다. 현 ArF 공정의 표준 마스크이다.
    """
    pitch = 180.0
    g = grid_for(pitch)
    lines, ok = [], True

    amp = MK.background_amplitude(MK.MASK_ATT_PSM, 0.06)
    t_meas = abs(amp) ** 2
    phase = np.degrees(np.angle(amp))
    ok &= abs(t_meas - 0.06) < 1e-12 and abs(abs(phase) - 180.0) < 1e-9
    lines.append(f"차광부 진폭 {amp.real:+.5f} → 세기 투과율 {t_meas:.4f}, 위상 {phase:.1f}도")

    rows = []
    for mt in (MK.MASK_BINARY, MK.MASK_ATT_PSM):
        m = MK.line_space(g, duty=0.5, mask_type=mt)
        st = MK.mask_stats(m, g, 0.5 * pitch, mask_type=mt)
        I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                            sigma_out=SIG, polarization=IM.POL_TE)
        e_op = PR.dose_to_size(g, I, 0.5 * pitch, TH, max_width=pitch)
        r = RS.extract_cd(g.x, I * e_op, TH, max_width=pitch)
        rows.append((mt, IM.contrast(I, g.x), r["nils"], st["measured_duty"],
                     st["mean_intensity"]))
        # duty 는 격자 양자화로 최대 dx/pitch 만큼 어긋난다. 이는 결함이 아니다.
        ok &= st["error"] <= g.dx / pitch + 1e-9

    lines.append("")
    lines.append(f"{'마스크':<10} {'contrast':>9} {'NILS':>7} {'평균 세기':>10} {'측정 duty':>10}")
    for mt, c, n, d, mi in rows:
        lines.append(f"{mt:<10} {c:9.4f} {n:7.3f} {mi:10.4f} {d:10.4f}")

    # 핵심 검증 : 차광부가 6% 투과하므로 평균 세기는 서로 달라야 하지만,
    # 배경 기여를 제거한 duty 추정은 두 마스크에서 동일해야 한다.
    same_duty = abs(rows[0][3] - rows[1][3]) < 1e-12
    diff_mean = abs(rows[0][4] - rows[1][4]) > 1e-6
    ok &= same_duty and diff_mean
    lines.append("")
    lines.append(f"평균 세기는 서로 다름 = {diff_mean} (차광부 6% 투과 때문)")
    lines.append(f"배경 보정 후 duty 는 동일 = {same_duty}")
    lines.append(f"격자 양자화 오차 {g.dx/pitch:.4f} (= dx/pitch) 이내")

    gain = rows[1][2] / rows[0][2]
    ok &= gain > 1.0
    lines.append("")
    lines.append(f"att-PSM 의 NILS 개선 {gain:.3f}배. Mack 의 EL ~ 10 x NILS 에 따르면")
    lines.append(f"노광 여유도가 약 {100*(gain-1):.1f} % 늘어나는 것에 해당함.")
    report("Case 4  감쇠형 위상반전마스크", ok, lines)


# ---------------------------------------------------------------- Case 5
def case5_flare_analytic():
    """플레어 모델이 해석적으로 예측한 대비 저하와 일치하는가.

    I' = (1-f) I + f <I> 이므로
        contrast' / contrast = (1-f)(Imax+Imin) / ((1-f)(Imax+Imin) + 2 f <I>)
    """
    g = grid_for(180.0)
    m = MK.line_space(g, duty=0.5)
    I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                        sigma_out=SIG, polarization=IM.POL_TE)
    s = float(I.max() + I.min())
    mean = float(np.mean(I))
    c0 = IM.contrast(I, g.x)

    lines, errs = [], []
    lines.append(f"{'flare':>7} | {'측정 비':>10} {'해석 비':>10} {'오차':>10} {'NILS':>7}")
    for f in (0.00, 0.01, 0.02, 0.03, 0.05):
        If = IM.apply_flare(I, f)
        meas = IM.contrast(If, g.x) / c0
        pred = (1 - f) * s / ((1 - f) * s + 2 * f * mean)
        errs.append(abs(meas - pred))
        e_op = PR.dose_to_size(g, If, 90.0, TH, max_width=180.0)
        nils = RS.extract_cd(g.x, If * e_op, TH, max_width=180.0)["nils"]
        lines.append(f"{f:7.2f} | {meas:10.6f} {pred:10.6f} {abs(meas-pred):10.2e} {nils:7.3f}")
    ok = max(errs) < 1e-9
    lines.append("")
    lines.append(f"최대 오차 {max(errs):.1e}")
    lines.append("플레어는 대비와 NILS 를 함께 떨어뜨리며, 그 양상이 노광량 과다와")
    lines.append("구별되지 않는다. 역추정에서 새로운 혼동 축이 된다.")
    report("Case 5  플레어 모델의 해석 대조", ok, lines)


# ---------------------------------------------------------------- Case 6
def case6_msd_blur_degeneracy():
    """스캔 동기화 오차와 레지스트 블러의 축퇴 구조.

    두 효과는 모두 가우시안 평활이므로 평균 상에서는 합성 분산
    sqrt(msd^2 + blur^2) 만 남고 배분은 드러나지 않는다. 그러나 MSD 는
    광자 샘플링 이전, 블러는 이후에 작용하므로 잡음 평활 여부가 달라
    LER 에서는 구별된다. 즉 CD 만 재면 식별 불가, LER 을 함께 재면
    식별 가능한 사례이다.
    """
    pitch = 180.0
    total = 10.0
    lines, cds, lers, mc_cds = [], [], [], []

    # [11주차 재정립 - 이 시험은 공허하게 통과하고 있었다]
    # 이전에는 make_sample 이 돌려주는 cd_deterministic 으로 축퇴를 확인하고,
    # 산포 3.8e-15 를 '기계 정밀도로 동일' 의 근거로 인용했다. 그런데
    # make_sample 은 조건마다 dose_to_size 로 노광량을 보정해 **CD 를 목표값에
    # 맞춘다.** 그래서 cd_deterministic 은 네 조건 전부 정확히 목표값(100.0)이
    # 되고, 산포는 축퇴가 아니라 **보정 솔버의 수렴 정밀도**를 재고 있었다.
    # 축퇴가 성립하든 깨지든 똑같이 통과한다. 항목 19 와 같은 '공허한 통과' 다.
    #
    # 올바른 시험은 노광량 보정을 끼우지 않고 결정론 프로파일을 직접 비교하는
    # 것이다. 평균 수준에서 두 효과가 같은 근거는 기대값의 선형성이다.
    #     E[blur(Poisson(lam))] = blur(E[Poisson(lam)]) = blur(lam)
    # 즉 광자 샘플링 **이후**의 블러도 평균에는 이전과 똑같이 작용한다. 따라서
    # 결정론 등가물은 '세기에 MSD 와 블러를 차례로 거는 것' 이고, 합성 분산이
    # 같으면 완전히 같아야 한다. 여기에 가우시안 분산 가산성(verify.py C8,
    # 2.5e-16)을 더하면 결론이 닫힌다.
    cds_direct = []
    g_d = G.build_grid(pitch, 6, G.nyquist_dx(LAM, NA, SIG))
    m_d = MK.line_space(g_d, duty=0.5)
    i_base = np.asarray(IM.aerial_image(g_d, m_d, LAM, NA, n_medium=N_MED,
                                        n_resist=N_PR, sigma_out=SIG,
                                        polarization=IM.POL_TE)).real
    for msd_d in (0.0, 4.0, 6.0, 8.0):
        blur_d = float(np.sqrt(max(total ** 2 - msd_d ** 2, 0.0)))
        prof = IM.apply_msd(i_base, msd_d, g_d.dx) if msd_d > 0 else i_base
        if blur_d > 0:
            prof = IM.gaussian_blur_periodic(prof, blur_d / g_d.dx)
        ex = RS.image_extrema(g_d.x, prof)
        thr_d = 0.5 * (ex["i_max"] + ex["i_min"])
        cds_direct.append(RS.extract_cd(g_d.x, prof, thr_d,
                                        max_width=g_d.pitch)["cd"])
    direct_spread = ((max(cds_direct) - min(cds_direct))
                     / float(np.mean(cds_direct)))
    lines.append("[직접 시험] 노광량 보정을 끼우지 않고 결정론 프로파일을 비교")
    lines.append(f"  CD 값 : {', '.join('%.9f' % c for c in cds_direct)}")
    lines.append(f"  상대 산포 {direct_spread:.3e}  (해석적으로 0 이어야 함)")
    lines.append("")
    lines.append(f"합성 블러 sqrt(msd^2 + blur^2) = {total:.1f} nm 로 고정하고 배분만 바꿈")
    lines.append("")
    lines.append(f"{'msd':>6} {'blur':>6} | {'결정론 CD':>11} {'MC CD':>9} "
                 f"{'LER 3σ':>9}")
    for msd in (0.0, 4.0, 6.0, 8.0):
        blur = float(np.sqrt(max(total**2 - msd**2, 0.0)))
        sp = DS.SampleSpec(pitch=pitch, duty=0.5, na=NA, n_medium=N_MED,
                           n_resist=N_PR, sigma_out=SIG, polarization=IM.POL_TE,
                           msd_nm=msd, blur_nm=blur, dose_mj=25.0,
                           n_trials=6, n_rows=1024, dy=2.0, seed=1)
        r = DS.make_sample(sp)
        lab = dict(zip(DS.LABEL_KEYS, r["labels"]))
        cds.append(lab["cd_deterministic"]); lers.append(lab["ler_3sigma"])
        mc_cds.append(lab["cd_mean"])
        lines.append(f"{msd:6.1f} {blur:6.1f} | {lab['cd_deterministic']:11.6f} "
                     f"{lab['cd_mean']:9.3f} {lab['ler_3sigma']:9.3f}")

    cd_spread = (max(cds) - min(cds)) / np.mean(cds)
    ler_spread = (max(lers) - min(lers)) / np.mean(lers)
    mc_spread = (max(mc_cds) - min(mc_cds)) / np.mean(mc_cds)
    # [10주차 재정립] 축퇴 주장은 '결정론' CD 에 대한 것이다. 평균 수준에서
    # MSD 와 블러는 분산이 더해지는 같은 가우시안 평활이므로, 합성 분산을
    # 고정하면 결정론 CD 는 해석적으로 완전히 같아야 한다. 기존 기준 0.02 는
    # 당시 수치 오차 수준을 기록한 값이었고, 더구나 몬테카를로 cd_mean 으로
    # 재고 있었다. cd_mean 에는 표본 잡음(약 LER/sqrt(N))이 섞여 있어 0 이 될
    # 수 없으므로, 그 값으로는 축퇴를 '정확히' 확인할 수 없다.
    # 따라서 결정론 CD 로 축퇴를 확인하고, MC 산포는 참고로만 적는다.
    lines.append("")
    lines.append(f"보정 후 CD 산포 {cd_spread:.2e}  <- 축퇴의 근거가 아니다.")
    lines.append("  make_sample 이 dose_to_size 로 CD 를 목표값에 맞추므로 이 값은")
    lines.append("  네 조건 전부 목표값이 된 결과이며, 보정 솔버의 수렴 정밀도다.")
    lines.append(f"MC cd_mean 산포 {mc_spread:.2e} (표본 잡음 포함, 0 이 아님)")
    lines.append(f"LER 산포 {ler_spread:.2e}")
    # 합격 판정은 **직접 시험**에 둔다. 보정이 끼어 있는 cd_spread 로 판정하면
    # 축퇴가 깨져도 통과한다.
    ok = (direct_spread < 1e-12 and ler_spread > 50 * direct_spread
          and ler_spread > 0.05)
    lines.append("")
    lines.append(f"CD 상대 편차(직접) {100*direct_spread:7.4f} %   <- 배분을 바꿔도 불변")
    lines.append(f"LER 상대 편차 {100*ler_spread:5.2f} %   <- 배분에 따라 달라짐")
    lines.append("")
    lines.append("결론 : CD 단독 관측으로는 두 요인을 분리할 수 없고, LER 을 함께")
    lines.append("관측하면 분리 가능하다. 6주차 Defocus-Dose 축퇴와 같은 성격의")
    lines.append("식별 한계이며, 관측량을 늘리면 한계가 이동함을 보이는 사례다.")
    report("Case 6  MSD 와 레지스트 블러의 축퇴 및 분리 조건", ok, lines)


# ---------------------------------------------------------------- Case 7
def case7_lwr_ler_ratio():
    """LWR 과 LER 의 관계 : sqrt(2) 는 '경계 독립' 극한값이다.

    [10주차 재정립] 기존에는 LWR / LER = sqrt(2) 를 '이론값' 으로 두고
    편차 0.15 를 허용했다. 그런데 sqrt(2) 는 이론이 아니라 **두 경계가
    통계적으로 독립일 때의 특수한 값**이다. Mack 의 거칠기 자료에도 이
    환산 계수는 명시되지 않는다. 선폭은 W = x_R - x_L 이므로 표본 통계의
    대수적 항등식

        Var(W) = Var(x_R) + Var(x_L) - 2 Cov(x_R, x_L)

    이 항상 성립하고, 두 경계의 분산이 같다면

        LWR = sqrt(2 - 2 rho) * LER,    rho = corr(x_L, x_R)

    이 된다. rho = 0 일 때만 sqrt(2) 다. 따라서 올바른 검증은
      (1) 항등식이 정확히 성립하는지 (대수적으로 반드시 참 -> 기계 정밀도)
      (2) rho 가 물리적으로 납득되는 방향으로 움직이는지
    이고, sqrt(2) 와의 편차를 허용 오차로 덮는 것이 아니다.

    실측 결과 rho 는 음수이고 CD/blur 가 작아질수록 커진다. feature 내부의
    밝은 잡음 요동이 양 경계를 동시에 '바깥으로' 밀어내므로, 블러 길이가
    CD 에 가까워지면 두 경계가 한 덩어리로 움직여 음의 상관이 강해진다.
    그 결과 비는 sqrt(2) 보다 커진다.
    """
    lines, errs, rhos = [], [], []
    lines.append("W = x_R - x_L 의 분산 항등식 검증 (허용 오차 없음)")
    lines.append("")
    lines.append(f"{'CD':>6} {'blur':>6} {'CD/blur':>8} {'rho':>9} "
                 f"{'LWR/LER':>9} {'sqrt(2-2rho)':>13} {'항등식오차':>11}")
    for pitch, duty, blur in ((400.0, 0.5, 8.0), (200.0, 0.5, 8.0),
                              (120.0, 0.5, 8.0), (100.0, 0.4, 16.0),
                              (100.0, 0.35, 20.0)):
        g = G.build_grid(pitch, 6, 2.0)
        m = MK.line_space(g, duty=duty)
        i = np.asarray(IM.aerial_image(g, m, LAM, NA, n_medium=N_MED,
                                       n_resist=N_PR, sigma_out=SIG,
                                       polarization=IM.POL_TE)).real
        e = RS.image_extrema(g.x, i)
        th = 0.5 * (e["i_max"] + e["i_min"])
        ls, rs = [], []
        for k in range(8):
            f = ST.stochastic_field(i, g.dx, 30.0, LAM, n_rows=1024, dy=2.0,
                                    blur_nm=blur,
                                    rng=np.random.default_rng(300 + k))
            lf = ST.edge_positions(g.x, f, th, "left")
            rt = ST.edge_positions(g.x, f, th, "right")
            ok_ = np.isfinite(lf) & np.isfinite(rt) & ((rt - lf) < pitch)
            ls.append(lf[ok_]); rs.append(rt[ok_])
        lf = np.concatenate(ls); rt = np.concatenate(rs)
        w = rt - lf
        s_l, s_r, s_w = lf.std(ddof=1), rt.std(ddof=1), w.std(ddof=1)
        rho = float(np.corrcoef(lf, rt)[0, 1])
        pred = float(np.sqrt(s_l ** 2 + s_r ** 2 - 2 * rho * s_l * s_r))
        err = abs(s_w - pred) / s_w
        errs.append(err); rhos.append(rho)
        ler = 0.5 * (3 * s_l + 3 * s_r)
        lines.append(f"{duty*pitch:6.0f} {blur:6.1f} {duty*pitch/blur:8.2f} "
                     f"{rho:9.4f} {3*s_w/ler:9.4f} {np.sqrt(2-2*rho):13.4f} "
                     f"{err:11.2e}")
    lines.append("")
    lines.append(f"항등식 최대 오차 {max(errs):.2e} (대수적 항등식이므로 0)")
    lines.append(f"rho 범위 {min(rhos):+.4f} ~ {max(rhos):+.4f}. CD/blur 가 큰")
    lines.append("조건에서는 0 근방(독립)이고, 블러 길이가 CD 에 가까워지면")
    lines.append(f"음으로 커진다. sqrt(2) = {np.sqrt(2):.4f} 는 rho = 0 극한이며,")
    lines.append("상관이 생기면 실제 비가 그보다 커진다.")
    lines.append("")
    lines.append("결론 : 기존의 'sqrt(2) 재현' 주장은 독립 극한에서만 옳다.")
    lines.append("CD 가 블러 길이에 가까운 조건(미세 패턴)에서는 경계 상관을")
    lines.append("무시할 수 없으므로, LWR 을 LER 로 환산할 때 rho 를 함께")
    lines.append("보고해야 한다.")
    # 항등식은 대수적으로 참이므로 기계 정밀도를 요구한다. rho 가 전부
    # 음수이고 CD/blur 와 함께 단조로 움직이는지도 확인한다(물리 방향성).
    monotone = rhos[0] > rhos[-1]
    # rho 가 큰 CD/blur 에서 0 근방이고, 작은 쪽에서 음으로 커지는지를 본다.
    ok = (max(errs) < 1e-12 and abs(rhos[0]) < 0.05
          and rhos[-1] < -0.1 and monotone)
    report("Case 7  LWR 와 LER 의 관계 (경계 상관 포함)", ok, lines)


# ---------------------------------------------------------------- Case 8
def case8_zernike_and_coma():
    """Zernike 항의 정규화와 수차가 식별 한계에 주는 영향.

    Noll 정규화 Zernike 는 단위 원 위에서 RMS = 1 이다. 또 비점수차는
    초점 어긋남과, 코마는 패턴 이동과 유사한 변화를 만들어 역추정에서
    혼동 요인이 된다.
    """
    lines, ok = [], True
    # 정규화 검증
    n = 801
    gx = np.linspace(-1, 1, n)
    PX, PY = np.meshgrid(gx, gx)
    inside = (PX**2 + PY**2) <= 1.0
    lines.append("Noll 정규화 Zernike 의 단위 원 RMS (이론값 1)")
    for j in (5, 6, 7, 8, 11):
        W = IM.zernike_phase(PX, PY, {j: 1.0})
        rms = float(np.sqrt(np.mean(W[inside] ** 2)))
        ok &= abs(rms - 1.0) < 5e-3
        lines.append(f"  Z{j:<3} RMS = {rms:.5f}")

    # Z4 중복 지정 차단
    try:
        IM.zernike_phase(PX, PY, {4: 0.1}); ok = False
        lines.append("  Z4 중복 지정이 차단되지 않음")
    except ValueError:
        lines.append("  Z4(초점)는 defocus 인자와 중복되므로 차단됨")

    # 수차가 NILS 에 주는 영향
    pitch = 180.0
    g = grid_for(pitch)
    m = MK.line_space(g, duty=0.5)
    lines.append("")
    lines.append(f"{'수차':<18} {'contrast':>9} {'NILS':>7} {'좌우 NILS 차':>12}")
    for name, ab in (("없음", {}), ("Z6 비점 0.05", {6: 0.05}),
                     ("Z8 코마 0.05", {8: 0.05}), ("Z11 구면 0.05", {11: 0.05})):
        I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                            sigma_out=SIG, polarization=IM.POL_TE, aberrations=ab)
        e_op = PR.dose_to_size(g, I, 0.5 * pitch, TH, max_width=pitch)
        r = RS.extract_cd(g.x, I * e_op, TH, max_width=pitch)
        asym = abs(r["nils_left"] - r["nils_right"])
        lines.append(f"{name:<18} {IM.contrast(I, g.x):9.4f} {r['nils']:7.3f} {asym:12.4f}")
    lines.append("")
    lines.append("코마는 좌우 경계의 NILS 를 비대칭으로 만든다. 9주차까지는 NILS 를")
    lines.append("오른쪽 경계에서만 계산했으므로 이 비대칭이 품질 판정에 반영되지")
    lines.append("않았다. 10주차부터 두 경계 중 나쁜 쪽을 쓴다.")
    report("Case 8  Zernike 정규화 및 수차의 비대칭 효과", ok, lines)


# ---------------------------------------------------------------- Case 9
def case9_source_shapes():
    """조명 형상별 성능. 조밀 L&S 의 실사용 조명은 다이폴이다."""
    pitch = 150.0
    g = grid_for(pitch, sigma=0.9)
    m = MK.line_space(g, duty=0.5, mask_type=MK.MASK_ATT_PSM)
    lines = []
    lines.append(f"pitch {pitch:.0f} nm, att-PSM, TE 편광")
    lines.append("")
    lines.append(f"{'조명':<24} {'광원점(참고)':>12} {'contrast':>9} {'NILS':>7}")
    best, rows = None, []
    for name, kw in (
        ("conventional σ0.7", dict(sigma_out=0.7)),
        ("annular 0.6-0.9", dict(sigma_out=0.9, sigma_in=0.6)),
        ("dipole", dict(sigma_out=0.9, source_shape=IM.SOURCE_DIPOLE,
                        pole_offset=0.75, pole_radius=0.15)),
        ("quadrupole", dict(sigma_out=0.9, source_shape=IM.SOURCE_QUADRUPOLE,
                            pole_offset=0.75, pole_radius=0.15)),
    ):
        # [10주차 주의] 광원점 개수는 더 이상 생산 경로를 설명하지 않는다.
        # 기본 경로가 영역 분해 정확 구적으로 바뀌어 광원 구적점을 아예
        # 쓰지 않기 때문이다. 여기서는 조명 형상의 '상대적 크기' 를 보여
        # 주는 참고값으로만 남기고, 그렇게 표기한다.
        src = IM.source_points(kw.get("sigma_out", 0.7), kw.get("sigma_in", 0.0), 31,
                               shape=kw.get("source_shape", IM.SOURCE_CONVENTIONAL),
                               pole_offset=kw.get("pole_offset", 0.0),
                               pole_radius=kw.get("pole_radius", 0.15))
        I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                            polarization=IM.POL_TE, **kw)
        e_op = PR.dose_to_size(g, I, 0.5 * pitch, TH, max_width=pitch)
        nils = RS.extract_cd(g.x, I * e_op, TH, max_width=pitch)["nils"] \
            if np.isfinite(e_op) else float("nan")
        rows.append((name, len(src), IM.contrast(I, g.x), nils))
        lines.append(f"{name:<24} {len(src):12d} {IM.contrast(I, g.x):9.4f} "
                     f"{nils:7.3f}")
        if best is None or IM.contrast(I, g.x) > best[1]:
            best = (name, IM.contrast(I, g.x))
    ok = best[0] == "dipole"
    lines.append("")
    lines.append(f"최고 대비 조명 = {best[0]} ({best[1]:.4f})")
    lines.append("조밀 선패턴에서는 0 차와 1 차만 pupil 양단에 넣는 다이폴이 유리하다.")
    lines.append("")
    lines.append("[주의] '광원점' 은 참고값이다. 생산 경로는 영역 분해 정확")
    lines.append("구적이라 광원 구적점을 쓰지 않으므로, 이 수는 계산 비용이나")
    lines.append("정확도와 무관하다.")
    report("Case 9  조명 형상 비교", ok, lines)


# ---------------------------------------------------------------- Case 10
def case10_regression():
    """9주차까지의 결함이 재발하지 않는지 확인한다."""
    lines, ok = [], True

    # (a) 미형성 시 cd 는 항상 NaN
    g = grid_for(200.0)
    x = g.x
    dark = np.zeros(g.n_points)
    r = RS.extract_cd(x, dark, 0.3)
    ok &= (not r["formed"]) and np.isnan(r["cd"])
    lines.append(f"(a) 미형성 반환 cd = {r['cd']}  formed = {r['formed']}")
    lines.append("    0.0 을 돌려주면 '선폭 0 nm' 로 평균에 섞인다. NaN 으로 통일함.")

    # (b) cd_vs_dose 가 max_width 를 받는다
    import inspect
    sig = inspect.signature(RS.cd_vs_dose)
    ok &= "max_width" in sig.parameters
    lines.append(f"(b) cd_vs_dose 인자 : {list(sig.parameters)}")

    # (c) cd_matrix / dose_to_size 가 max_width 를 받는다
    for fn in (PR.cd_matrix, PR.dose_to_size):
        has = "max_width" in inspect.signature(fn).parameters
        ok &= has
        lines.append(f"(c) {fn.__name__:<14} max_width 지원 = {has}")

    # (d) LER 이 병합 행을 거르는가
    src = inspect.getsource(ST.monte_carlo)
    filtered = "np.std(r_ok)" in src and "np.std(l_ok)" in src
    ok &= filtered
    lines.append(f"(d) monte_carlo 의 LER 이 병합 행 필터 후 값을 사용 = {filtered}")
    lines.append("    8주차 LER 400 nm 급 이상값의 구조적 원인이었음.")

    # (e) NILS 가 두 경계의 최솟값인가
    m = MK.line_space(g, duty=0.5)
    I = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, n_resist=N_PR,
                        sigma_out=SIG, polarization=IM.POL_TE, aberrations={8: 0.08})
    e_op = PR.dose_to_size(g, I, 100.0, TH, max_width=200.0)
    rr = RS.extract_cd(g.x, I * e_op, TH, max_width=200.0)
    is_min = abs(rr["nils"] - min(rr["nils_left"], rr["nils_right"])) < 1e-12
    ok &= is_min
    lines.append(f"(e) NILS = min(좌 {rr['nils_left']:.3f}, 우 {rr['nils_right']:.3f}) "
                 f"= {rr['nils']:.3f}  → {is_min}")
    report("Case 10  9주차 결함 재발 방지", ok, lines)


if __name__ == "__main__":
    print("=" * 78)
    print(" 10주차 검증 : 결함 보정 및 벡터·마스크·결함요인 모델 추가")
    print("=" * 78)
    print()
    case1_na_guard()
    case2_vector_vs_analytic()
    case3_scalar_is_te()
    case4_att_psm()
    case5_flare_analytic()
    case6_msd_blur_degeneracy()
    case7_lwr_ler_ratio()
    case8_zernike_and_coma()
    case9_source_shapes()
    case10_regression()
    print("=" * 78)
    print(f" 결과: {sum(_results)}/{len(_results)} 통과")
    print()
    print(VC.summarize(_CASES))
    print("=" * 78)
    # [11주차 수정] 실패해도 종료 코드 0 을 돌려주고 있었다. 10 주차에
    # verify.py 에서 같은 결함을 고쳤는데(불변량이 깨져도 '위반' 만 인쇄하고
    # 0 으로 끝남) 회귀 스크립트 일곱 개는 확인조차 하지 않았다. 사람이 출력을
    # 읽으면 드러나지만, 종료 코드만 보는 자동 검증은 50 종이 전부 깨져도
    # '통과' 로 읽는다. 실제로 패키지 검증에서 그렇게 놓쳤다.
    sys.exit(0 if all(_results) else 1)
