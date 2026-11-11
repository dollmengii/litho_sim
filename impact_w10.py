"""
impact_w10.py -- 10주차 수정이 기존 데이터셋에 주는 영향 정량화

왜 필요한가
-----------
10주차에 다음 네 가지를 고쳤다.

  (1) 마스크 표본화 : 격자점 지시함수 -> 해석 푸리에 계수 합성
      (duty 가 1/N 만큼 계통적으로 커지던 오차 제거)
  (2) 광원 적분     : 정방 격자 균일 가중 -> 영역 분해 정확 구적
      (대비에 2.4 % 오차가 있던 것을 1e-7 수준으로)
  (3) CD/NILS 추출  : 선형보간 + 중심차분 -> 대역제한 보간 + 스펙트럼 미분
  (4) dose-to-size  : 선형보간 종료 -> 이분법으로 기계 정밀도까지 수렴
      (목표 CD 80 nm 대비 0.5 nm 보정 오차 제거)

(1)~(4) 는 모두 '같은 물리 조건에서 나오는 숫자' 를 바꾼다. 따라서 기존
11 만 건 데이터셋은 수정 전 코드의 산출물이며, 그대로 학습에 쓰면 모형이
수정 전 오차까지 함께 학습한다. 재생성이 필요한지 판단하려면 그 차이가
'식별하려는 신호' 에 비해 큰지 작은지를 알아야 한다.

이 스크립트는 데이터셋 생성에 실제로 쓰인 조건 분포에서 표본을 뽑아,
수정 전후의 CD·NILS·대비를 직접 비교한다.

실행:
    py impact_w10.py            # 60 개 조건
    py impact_w10.py 200        # 조건 수 지정
"""

import sys

import numpy as np

from litho import dataset as DS
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import resist as RS
from litho.grid import build_grid, nyquist_dx

#: campaign.DEFAULT_SPACE 와 DISCRETE_ARMS 에 맞춘 조건 분포
ARMS = [
    dict(polarization=IM.POL_TE, mask_type=MK.MASK_BINARY,
         source_shape=IM.SOURCE_CONVENTIONAL),
    dict(polarization=IM.POL_TE, mask_type=MK.MASK_ATT_PSM,
         source_shape=IM.SOURCE_CONVENTIONAL),
    dict(polarization=IM.POL_TE, mask_type=MK.MASK_ATT_PSM,
         source_shape=IM.SOURCE_DIPOLE),
    dict(polarization=IM.POL_UNPOLARIZED, mask_type=MK.MASK_ATT_PSM,
         source_shape=IM.SOURCE_DIPOLE),
]


def sample_condition(rng):
    arm = ARMS[int(rng.integers(len(ARMS)))]
    pitch = float(rng.uniform(80.0, 400.0))
    return dict(
        wavelength=193.0, na=1.35, n_medium=1.44, n_resist=1.70,
        sigma_out=float(rng.uniform(0.40, 0.90)),
        pitch=pitch, duty=float(rng.uniform(0.35, 0.65)),
        defocus=float(rng.uniform(-80.0, 80.0)),
        flare=float(rng.uniform(0.0, 0.05)),
        msd_nm=float(rng.uniform(0.0, 6.0)),
        **arm,
    )


def metrics(c, legacy: bool):
    """legacy=True 면 9주차까지의 경로로 계산한다."""
    samp = MK.SAMPLING_AREA if legacy else MK.SAMPLING_BANDLIMITED
    integ = "abbe" if legacy else "auto"
    interp = "linear" if legacy else "spectral"
    nscan = 400 if legacy else 64

    g = build_grid(c["pitch"], 6, nyquist_dx(c["wavelength"], c["na"],
                                             c["sigma_out"]))
    m = MK.line_space(g, cd=c["duty"] * c["pitch"], mask_type=c["mask_type"],
                      sampling=samp)
    kw = dict(wavelength=c["wavelength"], na=c["na"], n_medium=c["n_medium"],
              n_resist=c["n_resist"], sigma_out=c["sigma_out"], sigma_in=0.0,
              polarization=c["polarization"], source_shape=c["source_shape"],
              pole_offset=0.7 * c["sigma_out"], pole_radius=0.15,
              flare=c["flare"], msd_nm=c["msd_nm"],
              source_integration=integ)
    i_focus = np.asarray(IM.aerial_image(g, m, defocus=0.0, **kw)).real
    target = c["duty"] * c["pitch"]
    e_op = PR.dose_to_size(g, i_focus, target, 0.30, n_scan=nscan,
                           max_width=c["pitch"])
    if not np.isfinite(e_op):
        return None
    # 보정이 제대로 되었는지는 '보정한 그 조건(최적 초점)' 에서 봐야 한다.
    r0 = RS.extract_cd(g.x, i_focus * e_op, 0.30, max_width=c["pitch"],
                       interp=interp)
    i_dz = np.asarray(IM.aerial_image(g, m, defocus=c["defocus"], **kw)).real
    r = RS.extract_cd(g.x, i_dz * e_op, 0.30, max_width=c["pitch"],
                      interp=interp)
    if not r["formed"] or not r0["formed"]:
        return None
    contrast = float((i_dz.max() - i_dz.min()) / (i_dz.max() + i_dz.min()))
    return {"cd": r["cd"], "nils": r["nils"], "contrast": contrast,
            "e_op": e_op, "target": target, "cd_at_focus": r0["cd"]}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(0)
    rows = []
    while len(rows) < n:
        c = sample_condition(rng)
        a = metrics(c, legacy=True)
        b = metrics(c, legacy=False)
        if a is None or b is None:
            continue
        rows.append((c, a, b))

    print("=" * 78)
    print(" 10주차 수정 전후 비교 (데이터셋 생성 조건 분포에서 표본)")
    print("=" * 78)
    print(f" 조건 {len(rows)} 건\n")

    def stats(key, rel=True):
        v = []
        for c, a, b in rows:
            if rel:
                v.append(abs(b[key] - a[key]) / max(abs(b[key]), 1e-30))
            else:
                v.append(abs(b[key] - a[key]))
        return np.array(v)

    print(f" {'지표':<18}{'중앙값':>12}{'95 분위':>12}{'최악':>12}")
    print(" " + "-" * 54)
    for key, label, rel in (("cd", "CD 상대차", True),
                            ("nils", "NILS 상대차", True),
                            ("contrast", "대비 상대차", True),
                            ("e_op", "보정 노광량 상대차", True)):
        s = stats(key, rel)
        print(f" {label:<18}{np.median(s):12.3e}{np.quantile(s,0.95):12.3e}"
              f"{s.max():12.3e}")
    # [분리 집계] NILS 가 낮은 조건은 사실상 해상되지 않는 영역이다. 그런
    # 곳에서는 NILS 자체가 0 에 가까워 상대차가 발산하므로, 해상되는 조건과
    # 섞어 집계하면 분포가 왜곡된다.
    #
    # [기준] 여기서 2.0 을 하드코딩하면 안 된다. 품질 기준은 dataset.py 에서
    # CD 공차 예산과 배분 비율로 **유도**되며(Mack 근사 : EL[%] ~ NILS x
    # CD공차[%]), 기본 설정에서 2.121 이 나온다. 2.0 을 따로 적어 두면
    # 같은 개념에 기준이 두 개가 되어, VERIFICATION_LIMITS 항목 14 가 고친
    # '독립된 업계 표준 두 개' 문제를 이 분석에서 되살리게 된다. 유도값을
    # 그대로 가져온다.
    min_nils = DS.SampleSpec(pitch=200.0, duty=0.5).min_nils
    res = [r for r in rows if min(r[1]["nils"], r[2]["nils"]) >= min_nils]
    deg = [r for r in rows if min(r[1]["nils"], r[2]["nils"]) < min_nils]
    if res:
        print(f"\n 해상 가능한 조건만 (NILS >= {min_nils:.3f} "
              f"-- 유도값, {len(res)} 건)")
        print(f" {'지표':<18}{'중앙값':>12}{'95 분위':>12}{'최악':>12}")
        print(" " + "-" * 54)
        for key, label in (("cd", "CD 상대차"), ("nils", "NILS 상대차"),
                           ("contrast", "대비 상대차")):
            v = np.array([abs(b[key] - a[key]) / max(abs(b[key]), 1e-30)
                          for c, a, b in res])
            print(f" {label:<18}{np.median(v):12.3e}"
                  f"{np.quantile(v,0.95):12.3e}{v.max():12.3e}")
        v = np.array([abs(b["cd"] - a["cd"]) for c, a, b in res])
        print(f" {'CD 절대차 [nm]':<18}{np.median(v):12.4f}"
              f"{np.quantile(v,0.95):12.4f}{v.max():12.4f}")
    if deg:
        print(f"\n 해상 한계 근방 조건 ({len(deg)} 건) : NILS 가 0 에 가까워")
        print(" 상대차가 발산한다. 수정 전후 어느 쪽도 신뢰할 수 없는 영역이며,")
        print(" 데이터셋 품질 필터로 걸러야 하는 조건이다.")

    cd_abs = stats("cd", rel=False)
    print(f" {'CD 절대차 [nm]':<18}{np.median(cd_abs):12.4f}"
          f"{np.quantile(cd_abs,0.95):12.4f}{cd_abs.max():12.4f}")

    print("\n 보정 정확도 : 최적 초점에서 목표 CD 를 맞췄는가 [nm]")
    print(f" {'':<10}{'중앙값':>14}{'95 분위':>14}{'최악':>14}")
    for nm_, k in (("수정 전", 1), ("수정 후", 2)):
        d = np.array([abs(r[k]["cd_at_focus"] - r[k]["target"]) for r in rows])
        print(f" {nm_:<10}{np.median(d):14.3e}{np.quantile(d,0.95):14.3e}"
              f"{d.max():14.3e}")

    print("\n 가장 크게 바뀐 조건 5 건")
    order = np.argsort(-stats("nils"))[:5]
    print(f" {'pitch':>7}{'duty':>6}{'sigma':>7}{'dz':>7}{'조명':>13}"
          f"{'NILS 전':>10}{'NILS 후':>10}{'상대차':>9}")
    for i in order:
        c, a, b = rows[i]
        print(f" {c['pitch']:7.1f}{c['duty']:6.3f}{c['sigma_out']:7.3f}"
              f"{c['defocus']:7.1f}{c['source_shape']:>13}"
              f"{a['nils']:10.4f}{b['nils']:10.4f}"
              f"{abs(b['nils']-a['nils'])/b['nils']:9.3f}")

    nils_rel = stats("nils")
    print("\n" + "=" * 78)
    print(" 판단 기준")
    print("=" * 78)
    print(" 식별하려는 신호의 크기와 비교해야 한다. 식별 분석에서 쓰는")
    print(" 중심차분 스텝(예: defocus 4 nm, flare 0.002)이 만드는 관측량")
    print(" 변화가 '신호' 이고, 위 상대차가 '계통 오차' 다.")
    print(f" NILS 중앙 상대차 {np.median(nils_rel):.2%}, 95 분위"
          f" {np.quantile(nils_rel, 0.95):.2%} 이므로,")
    print(" 이보다 작은 변화를 식별하겠다는 주장은 수정 전 데이터로는")
    print(" 성립하지 않는다. 재생성 여부는 이 수치로 판단한다.")


if __name__ == "__main__":
    main()
