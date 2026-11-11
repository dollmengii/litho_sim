"""13주차 : 주장 3을 **계측 규격**으로 바꾼다.

주장 3은 "MSD 와 레지스트 블러는 평균으로 축퇴이고 잡음 구조로는 분리된다"
까지 말하고 멈춘다. 이 도구가 그 다음을 잰다.

    "그럼 CD 와 LER 을 얼마나 정밀하게 재야 분리되는가. 지점 몇 개를
     평균해야 하는가."

음성 결과가 **요구 규격**으로 바뀐다. 계산의 근거와 한계는
`litho/metrospec.py` 의 모듈 설명에 있다.

실행
----
    py metrospec_w13.py
    py metrospec_w13.py --msd 4.5 --blur 7.0 --target 1.0 --scan
    py metrospec_w13.py --sigma-cd 0.5 --ler-rel 0.10 --n-seeds 6

읽는 법
-------
- `교차곱 D` 가 0 이면 어떤 정밀도로도 분리되지 않는다. 사이각만 보면 안
  된다. **180 도는 0 도만큼 나쁘다**(반평행도 공선이다).
- 필요한 LER 정밀도는 **상대값**으로 읽어야 한다. 이 과제의 LER 은 산탄
  잡음만 고려한 하한이라 절대 크기가 실제보다 훨씬 작다.
- 지점 수는 지점 사이 독립을 전제한다. 같은 웨이퍼의 인접 지점은 계통
  성분을 공유하므로 실제로는 더 필요하다.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import metrospec as MS
from litho import resist as RS
from litho import stochastic as ST

#: 주장 3(verify.py H5)과 **같은 조건**을 쓴다. 다른 조건에서 재면 기존
#: 주장을 잇는 것이 아니라 새 설정에서 다시 재는 것이 된다.
COND = dict(pitch=200.0, n_periods=6, dx_target=2.0, duty=0.5,
            wavelength=193.0, na=1.35, n_medium=1.44, sigma_out=0.7,
            dose_mj=30.0, dy=2.0, n_rows=512, n_trials=8)

#: 생성 경로가 쓰는 범위(`campaign.DEFAULT_SPACE`). 기준점과 스텝을 이 안에
#: 두어야 외삽이 되지 않는다.
MSD_RANGE = (0.0, 6.0)
BLUR_RANGE = (5.0, 15.0)


def _setup():
    g = G.build_grid(pitch=COND["pitch"], n_periods=COND["n_periods"],
                     dx_target=COND["dx_target"])
    m = MK.line_space(g, duty=COND["duty"])
    i0 = np.asarray(IM.aerial_image(
        g, m, COND["wavelength"], COND["na"], n_medium=COND["n_medium"],
        sigma_out=COND["sigma_out"])).real
    e = RS.image_extrema(g.x, i0)
    # **임계값을 평활 없는 상에서 한 번 잡아 고정한다.** MSD 적용 후 상에서
    # 매번 다시 잡으면 그 규약이 정보를 흘려 축퇴가 깨진 것처럼 보인다.
    # 실측으로 결정론 CD 가 1.1594 nm 갈렸고, 고정하니 정확히 0 이 되었다.
    thr = 0.5 * (float(e["i_max"]) + float(e["i_min"]))
    return g, m, thr


def observe(g, m, thr, msd: float, blur: float, seed: int) -> np.ndarray:
    ii = np.asarray(IM.aerial_image(
        g, m, COND["wavelength"], COND["na"], n_medium=COND["n_medium"],
        sigma_out=COND["sigma_out"], msd_nm=float(msd))).real
    r = ST.monte_carlo(g.x, ii, g.dx, COND["dose_mj"], COND["wavelength"],
                       thr, n_rows=COND["n_rows"], blur_nm=float(blur),
                       n_trials=COND["n_trials"], seed=int(seed),
                       dy=COND["dy"], max_width=g.pitch)
    return np.array([r["cd_mean"], r["ler_3sigma"]], dtype=np.float64)


def mean_obs(g, m, thr, msd, blur, n_seeds):
    v = np.array([observe(g, m, thr, msd, blur, 900 + 17 * k)
                  for k in range(n_seeds)])
    return v.mean(axis=0), v.std(axis=0, ddof=1) / np.sqrt(n_seeds)


def degeneracy_check(g, m, thr) -> dict:
    """축퇴가 **참인지** 먼저 확인한다. 깨져 있으면 이 분석의 전제가 없다.

    합성 평활을 고정하고 배분을 바꾸며 결정론 CD 를 본다. 기계 정밀도로
    같아야 한다. 같지 않으면 임계값 규약이 어긋난 것이거나 가우시안
    가산성이 깨진 것이고, 둘 다 먼저 고쳐야 한다.
    """
    tot = 8.0
    b4 = float(np.sqrt(tot ** 2 - 4.0 ** 2))
    det, mcm, ler = [], [], []
    for msd, blur in ((0.0, tot), (4.0, b4), (b4, 4.0), (tot, 0.0)):
        ii = np.asarray(IM.aerial_image(
            g, m, COND["wavelength"], COND["na"],
            n_medium=COND["n_medium"], sigma_out=COND["sigma_out"],
            msd_nm=msd)).real
        ib = IM.apply_msd(ii, blur, g.dx) if blur > 0 else ii
        det.append(float(RS.extract_cd(g.x, ib, thr,
                                       max_width=g.pitch)["cd"]))
        v = observe(g, m, thr, msd, blur, 17)
        mcm.append(float(v[0]))
        ler.append(float(v[1]))
    return {"det_range": max(det) - min(det),
            "mean_range": max(mcm) - min(mcm),
            "ler_range": max(ler) - min(ler),
            "ler_rel_range": (max(ler) - min(ler)) / max(ler)}


def gradients(g, m, thr, msd0, blur0, h_msd, h_blur, n_seeds):
    """중심차분. 스텝을 범위 안으로 잘라내고 **실제 쓴 폭**으로 나눈다.

    기준점이 범위 경계에 가까우면 스텝이 잘린다. 그때 명목 스텝으로 나누면
    기울기가 작게 나온다. 잘린 폭을 그대로 쓴다.
    """
    mp, mm = min(msd0 + h_msd, MSD_RANGE[1]), max(msd0 - h_msd, MSD_RANGE[0])
    bp, bm = (min(blur0 + h_blur, BLUR_RANGE[1]),
              max(blur0 - h_blur, BLUR_RANGE[0]))
    sp_m, sp_b = mp - mm, bp - bm
    if sp_m <= 0 or sp_b <= 0:
        return None, None, None
    (vp, ep), (vq, eq) = mean_obs(g, m, thr, mp, blur0, n_seeds), \
        mean_obs(g, m, thr, mm, blur0, n_seeds)
    (wp, fp), (wq, fq) = mean_obs(g, m, thr, msd0, bp, n_seeds), \
        mean_obs(g, m, thr, msd0, bm, n_seeds)
    g_cd = np.array([(vp[0] - vq[0]) / sp_m, (wp[0] - wq[0]) / sp_b])
    g_ler = np.array([(vp[1] - vq[1]) / sp_m, (wp[1] - wq[1]) / sp_b])
    # 기울기 자신의 표집 오차. 중심차분은 두 끝의 오차가 독립이므로 제곱합.
    se_cd = np.array([np.hypot(ep[0], eq[0]) / sp_m,
                      np.hypot(fp[0], fq[0]) / sp_b])
    se_ler = np.array([np.hypot(ep[1], eq[1]) / sp_m,
                       np.hypot(fp[1], fq[1]) / sp_b])
    return g_cd, g_ler, {"se_cd": se_cd, "se_ler": se_ler,
                         "span_msd": sp_m, "span_blur": sp_b}


def report_point(g, m, thr, msd0, blur0, args, ler_val) -> dict | None:
    print(f"\n  기준점 : msd {msd0:.2f} nm, blur {blur0:.2f} nm")
    rows = []
    for h_m, h_b in ((args.h_msd, args.h_blur),
                     (2 * args.h_msd, 2 * args.h_blur)):
        g_cd, g_ler, info = gradients(g, m, thr, msd0, blur0,
                                      h_m, h_b, args.n_seeds)
        if g_cd is None:
            continue
        D = MS.cross(g_cd, g_ler)
        rows.append((info["span_msd"], info["span_blur"], g_cd, g_ler, D,
                     info))
        print(f"    스텝 msd ±{info['span_msd'] / 2:.2f} / "
              f"blur ±{info['span_blur'] / 2:.2f} nm")
        print(f"      dCD/d(msd,blur)  = ({g_cd[0]:+.5f}, {g_cd[1]:+.5f})"
              f"   SE ({info['se_cd'][0]:.5f}, {info['se_cd'][1]:.5f})")
        print(f"      dLER/d(msd,blur) = ({g_ler[0]:+.5f}, {g_ler[1]:+.5f})"
              f"   SE ({info['se_ler'][0]:.5f}, {info['se_ler'][1]:.5f})")
        print(f"      교차곱 D {D:+.3e}, 사이각 "
              f"{MS.angle_deg(g_cd, g_ler):.1f} 도, |sin| "
              f"{abs(np.sin(np.radians(MS.angle_deg(g_cd, g_ler)))):.3f}")
    if not rows:
        print("    스텝이 범위 안으로 잘려 기울기를 낼 수 없다.")
        return None
    # 큰 스텝 쪽을 쓴다. 표집 잡음에 덜 흔들린다. 두 스텝의 차이가
    # 비선형성 + 잡음의 크기이고, 그것을 아래에서 범위로 보고한다.
    sp_m, sp_b, g_cd, g_ler, D, info = rows[-1]
    if len(rows) > 1:
        d0 = rows[0][4]
        print(f"    두 스텝의 D 차이 {abs(D - d0) / abs(D) * 100:.1f} % "
              f"<- 비선형성 + 표집 잡음의 크기")

    sig_l = args.ler_rel * ler_val
    cr = MS.crb(g_cd, g_ler, args.sigma_cd, sig_l, args.n_cd, args.n_ler)
    print(f"\n    [현재 계측으로 달성 가능한 하한]  CD 1시그마 "
          f"{args.sigma_cd:.3f} nm, LER 1시그마 {sig_l:.4f} nm "
          f"({100 * args.ler_rel:.0f} % 상대)")
    for name, kor in (("blur", "레지스트 블러"), ("msd", "MSD")):
        rng = MSD_RANGE if name == "msd" else BLUR_RANGE
        width = rng[1] - rng[0]
        sd = cr[f"sd_{name}"]
        verdict = ("**결정 가능**" if sd <= 0.2 * width
                   else "결정 불가 (자기 범위보다 넓다)" if sd >= width
                   else "경계")
        print(f"      sd({kor}) = {sd:7.2f} nm  "
              f"(범위 폭 {width:.0f} nm 의 {sd / width:5.1%})  {verdict}")
        print(f"        한계가 되는 관측량 : {cr[f'binds_{name}']}  "
              f"(LER항 {cr[f'term_ler_{name}']:.3g}, "
              f"CD항 {cr[f'term_cd_{name}']:.3g})")

    print(f"\n    [목표 ±{args.target:.1f} nm 를 맞추려면]  분산을 반반으로 "
          f"나눈 예산")
    out = {"msd0": msd0, "blur0": blur0, "D": D, "crb": cr, "budget": {}}
    for name, kor in (("blur", "레지스트 블러"), ("msd", "MSD")):
        bd = MS.budget(g_cd, g_ler, args.target, name,
                       args.n_cd, args.n_ler)
        n_c = MS.sites_needed(args.sigma_cd, bd["sigma_cd"])
        n_l = MS.sites_needed(sig_l, bd["sigma_ler"])
        rel = (bd["sigma_ler"] / ler_val) if bd["sigma_ler"] else None
        print(f"      {kor} : CD {bd['sigma_cd']:.4f} nm (지점 {n_c}), "
              f"LER {bd['sigma_ler']:.5f} nm = {100 * rel:.2f} % 상대 "
              f"(지점 {n_l})")
        out["budget"][name] = {"sigma_cd": bd["sigma_cd"],
                               "sigma_ler": bd["sigma_ler"],
                               "ler_rel": rel, "n_cd": n_c, "n_ler": n_l}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="주장 3을 계측 규격으로 바꾼다 (MSD/블러 분리에 필요한 "
                    "CD·LER 정밀도와 지점 수)")
    ap.add_argument("--msd", type=float, default=3.0, help="기준점 MSD [nm]")
    ap.add_argument("--blur", type=float, default=10.0,
                    help="기준점 레지스트 블러 [nm]")
    ap.add_argument("--h-msd", type=float, default=1.5,
                    help="MSD 중심차분 스텝 [nm]. 두 배 스텝도 함께 돌린다")
    ap.add_argument("--h-blur", type=float, default=2.0,
                    help="블러 중심차분 스텝 [nm]")
    ap.add_argument("--n-seeds", type=int, default=4,
                    help="각 점에서 평균할 씨앗 수 (기울기 표집오차를 줄인다)")
    ap.add_argument("--sigma-cd", type=float, default=0.5,
                    help="CD 계측 1시그마 [nm]. CD-SEM 반복성 수준")
    ap.add_argument("--ler-rel", type=float, default=0.10,
                    help="LER 계측 1시그마를 LER 값의 비율로 (기본 0.10)")
    ap.add_argument("--n-cd", type=int, default=1,
                    help="평균하는 CD 계측 지점 수")
    ap.add_argument("--n-ler", type=int, default=1,
                    help="평균하는 LER 계측 지점 수")
    ap.add_argument("--target", type=float, default=1.0,
                    help="목표 표준편차 [nm]")
    ap.add_argument("--scan", action="store_true",
                    help="범위 안 여러 기준점을 훑어 '어디서 재야 하는가' "
                         "를 본다")
    args = ap.parse_args()

    print("=" * 78)
    print(" 주장 3의 계측 규격 : MSD 와 레지스트 블러를 가르려면")
    print("=" * 78)
    print("\n  조건은 주장 3(verify.py H5)과 같다 : "
          f"pitch {COND['pitch']:.0f} nm, lambda {COND['wavelength']:.0f} nm,")
    print(f"  NA {COND['na']}, n {COND['n_medium']}, "
          f"sigma {COND['sigma_out']}, dose {COND['dose_mj']:.0f} mJ/cm^2, "
          f"행 {COND['n_rows']}, 시행 {COND['n_trials']}")

    g, m, thr = _setup()
    print(f"\n  임계값을 평활 없는 상에서 한 번 잡아 고정 : {thr:.6f}")
    print("  MSD 적용 후 상에서 매번 다시 잡으면 그 규약이 정보를 흘려")
    print("  축퇴가 깨진 것처럼 보인다. 실측으로 결정론 CD 가 1.1594 nm")
    print("  갈렸고, 고정하니 아래처럼 0 이 된다.")

    dg = degeneracy_check(g, m, thr)
    print(f"\n  [축퇴 확인] 합성 평활 8 nm 고정, 배분 4 가지")
    print(f"    결정론 CD 범위  {dg['det_range']:.3e} nm   <- 축퇴 성립")
    print(f"    측정 평균 범위  {dg['mean_range']:.4f} nm   <- **축퇴 아님**")
    print(f"    LER 범위        {dg['ler_range']:.4f} nm "
          f"({dg['ler_rel_range']:.1%})  <- 주장 3의 신호")
    if dg["det_range"] > 1e-9:
        print("\n  [경고] 결정론 CD 가 같지 않다. 축퇴가 성립하지 않으므로")
        print("  이 분석의 전제가 없다. 임계값 규약과 가우시안 가산성을")
        print("  먼저 확인할 것. 아래 숫자를 쓰지 말 것.")
        return 1
    print("\n  측정 평균이 축퇴가 아닌 것은 역치 추정기의 분산 유발")
    print("  치우침이다(치우침 ~ LER^2 적합 R^2 0.9998, 형성률 1.000 이라")
    print("  추출 실패 설명은 배제). 다만 **독립한 두 번째 채널은 아니다** —")
    print("  LER^2 의 함수이므로 grad LER 과 평행한 방향만 보탠다.")
    print("  두 관측량이 함께 식별하는 이유는 더 단순하다 :")
    print("    결정론 CD 는 합성 평활 hypot(msd, blur) **만의** 함수다.")
    print("    -> CD 가 **합성 크기**를, LER 이 **배분**을 준다.")

    ler_val = float(observe(g, m, thr, args.msd, args.blur, 17)[1])
    pts = ([(args.msd, args.blur)] if not args.scan else
           [(1.5, 7.0), (3.0, 10.0), (4.5, 13.0), (1.5, 13.0), (4.5, 7.0)])
    got = []
    for mm, bb in pts:
        lv = float(observe(g, m, thr, mm, bb, 17)[1])
        r = report_point(g, m, thr, mm, bb, args, lv)
        if r:
            r["ler"] = lv
            got.append(r)

    print(f"\n{'=' * 78}")
    print(" 결론")
    print("=" * 78)
    if not got:
        print("\n  기준점에서 기울기를 낼 수 없었다.")
        return 1
    if args.scan and len(got) > 1:
        print(f"\n  {'msd':>5}{'blur':>6}{'LER':>8}{'|D|':>11}"
              f"{'sd(blur)':>10}{'sd(msd)':>10}{'CD지점(msd)':>12}")
        for r in got:
            print(f"  {r['msd0']:5.1f}{r['blur0']:6.1f}{r['ler']:8.4f}"
                  f"{abs(r['D']):11.2e}{r['crb']['sd_blur']:10.2f}"
                  f"{r['crb']['sd_msd']:10.2f}"
                  f"{str(r['budget']['msd']['n_cd']):>12}")
        best = min(got, key=lambda r: r["crb"]["sd_msd"])
        print(f"\n  가장 유리한 기준점 : msd {best['msd0']:.1f}, "
              f"blur {best['blur0']:.1f} (MSD 쪽 sd 가 가장 작다)")
        print("  MSD 가 크고 블러가 작은 쪽이 유리하다. 거기서 LER 이 크고")
        print("  LER 의 블러 민감도가 커서 두 기울기가 덜 공선이다.")
    b = got[0]
    print(f"\n  1. LER 을 더하면 축퇴가 **형식적으로** 깨진다 "
          f"(교차곱 D = {b['D']:+.2e} != 0).")
    print("     주장 3이 맞다. 다만 형식적 식별과 실무적 측정은 다르다.")
    print(f"\n  2. 현재 수준 계측(CD 1시그마 {args.sigma_cd:.1f} nm, "
          f"LER {100 * args.ler_rel:.0f} % 상대, 1 지점)에서")
    print(f"     레지스트 블러 : sd {b['crb']['sd_blur']:.2f} nm "
          f"-> 범위 10 nm 대비 쓸 수 있다")
    print(f"     MSD          : sd {b['crb']['sd_msd']:.2f} nm "
          f"-> 자기 범위(6 nm)보다 넓어 **결정 불가**")
    print(f"\n  3. 한계가 되는 관측량이 다르다. 블러는 "
          f"{b['crb']['binds_blur']}, MSD 는 {b['crb']['binds_msd']} 다.")
    print("     즉 MSD 를 재려면 LER 이 아니라 **CD 정밀도**를 올려야 한다.")
    bm, bb_ = b["budget"]["msd"], b["budget"]["blur"]
    print(f"\n  4. 목표 ±{args.target:.1f} nm 의 계측 예산")
    print(f"     블러 : CD {bb_['sigma_cd']:.3f} nm ({bb_['n_cd']} 지점), "
          f"LER {100 * bb_['ler_rel']:.1f} % 상대 ({bb_['n_ler']} 지점)")
    print(f"     MSD  : CD {bm['sigma_cd']:.4f} nm (**{bm['n_cd']} 지점**), "
          f"LER {100 * bm['ler_rel']:.2f} % 상대 ({bm['n_ler']} 지점)")
    print("\n     블러는 표준 CD-SEM 과 LER 계측으로 몇 지점만 평균해도")
    print("     닿는다. MSD 는 CD 지점 수가 비용을 지배한다. 계측 레시피가")
    print("     수백 지점을 재는 것은 현장에서 흔하므로 **닿을 수 있는")
    print("     규격**이지만, 지점 사이 독립을 전제한 수다. 같은 웨이퍼의")
    print("     인접 지점은 계통 성분을 공유하므로 실제로는 더 필요하다.")
    print("\n  [전제] 반드시 함께 적어야 하는 것 셋")
    print("   (1) 관측량 2 개로는 **미지수 2 개까지만** 정할 수 있다")
    print("       (rank(J) <= 관측량 수). 위 숫자는 초점·노광량·플레어가")
    print("       **정확히 알려져 있다**는 전제 위에 선다. 주장 1 은 바로 그")
    print("       5 변수 문제를 다루므로 **다른 문제**다. 미지수 n 개를")
    print("       가르려면 독립 관측량이 최소 n 개 필요하다.")
    print("   (2) CD 와 LER 의 계측 오차를 독립으로 가정했다. 같은 SEM")
    print("       영상에서 나오므로 독립일 이유가 없다. 상관 rho=±0.9 에서")
    print("       영향이 0.35~1.37 배이고 **일률적으로 보수적이지 않다.**")
    print("       결론은 뒤집히지 않지만 숫자에 ±40 % 를 붙여 읽을 것.")
    print("   (3) CRB 는 하한이라 **부정 결론에만 엄격하다.** 'MSD 결정")
    print("       불가' 는 엄격하고, '블러 결정 가능' 은 그 하한을 달성하는")
    print("       추정기가 있다는 증명이 아니다.")
    print("\n  [한계] 이 과제의 LER 은 산탄 잡음만 고려한 하한이라 절대값이")
    print("  실제보다 작다. 필요한 LER 정밀도는 상대값으로 옮기는 편이")
    print("  낫지만 그것도 중립이 아니다 — 실제 LER 에는 msd·blur 에 반응")
    print("  하지 않는 성분(레지스트 화학, 마스크 거칠기)이 제곱합으로")
    print("  더해져 상대 민감도를 **묽게** 한다. 따라서 실제 요구 정밀도는")
    print("  여기 숫자보다 **더 엄격하다.** 방향이 정해진 편의다.")
    print("  기울기는 국소값이며 기준점에 따라 변한다(--scan 으로 확인).")
    print("  그리고 전부 시뮬레이터 기준이다 — 웨이퍼 기준이 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
