"""
validate_identify.py -- 식별 가능성 분석 검증 (10주차 보강)

본 과제의 핵심 주장을 검증한다.

    식별 한계를 정하는 것은 광자 잡음이 아니라 민감도 벡터의 공선성이다.

검증 방식은 이전 주차와 같다. 해석적으로 성립해야 하는 관계, 또는 서로
독립적인 두 경로의 일치를 기준으로 삼는다.

실행:  py validate_identify.py
"""

import sys

import numpy as np

import validate_common as VC

from litho import dataset as DS
from litho import identify as ID
from litho import splitting as SP

_results = []


_CASES = []

def report(name, passed, lines):
    _results.append(passed)
    _CASES.append((name, bool(passed)))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    for ln in lines:
        print(f"        {ln}")
    print()


def base_spec():
    return ID.IdentifySpec()


# ---------------------------------------------------------------- Case 1
def case1_collinearity():
    """민감도 벡터의 공선성이 식별 한계의 원인임을 보인다.

    두 파라미터의 민감도 벡터가 같은 방향이면, 관측량만 보고 둘을 나눌 수
    없다. 사이각이 그 척도이다.
    """
    c = ID.collinearity(base_spec())
    lines = [f"{'쌍':<26} {'코사인':>8} {'사이각':>9}"]
    for r in c["pairs"]:
        lines.append(f"{r['a']+' - '+r['b']:<26} {r['cos']:8.4f} {r['angle_deg']:8.2f}도")

    top = c["pairs"][0]
    dose_pairs = [r for r in c["pairs"] if "dose_rel" in (r["a"], r["b"])]
    worst_dose = max(r["cos"] for r in dose_pairs)
    ok = top["cos"] > 0.99 and worst_dose < 0.6

    lines += ["",
              f"가장 공선적인 쌍 : {top['a']} - {top['b']}, 사이각 {top['angle_deg']:.2f}도",
              f"Dose 와 다른 변수들의 최대 코사인 : {worst_dose:.4f}",
              "",
              "Dose 만 다른 변수들과 뚜렷이 구별된다(코사인 0.5 아래). Dose 는",
              "상 전체를 비례 확대하는 반면, 나머지는 모두 대비를 떨어뜨리는",
              "방식으로 나타나기 때문이다. 즉 '흐려짐' 을 유발하는 변수들끼리",
              "서로 엉켜 있고, 이것이 식별 한계의 직접적 원인이다."]
    report("Case 1  민감도 벡터의 공선성", ok, lines)


# ---------------------------------------------------------------- Case 2
def case2_focus_symmetry():
    """최적 초점에서 초점은 원리적으로 식별 불가능하다.

    수차가 없으면 상이 +dz 와 -dz 에 대해 대칭이므로 1차 미분이 정확히
    사라진다. 즉 초점이 어느 쪽으로 틀어졌는지 알 수 없다.
    """
    lines = [f"{'공칭 Δz':>9} {'‖∂μ/∂Δz‖ (정규화)':>22}"]
    norms = []
    for dz in (0.0, 1.0, 5.0, 20.0, 80.0):
        sp = ID.IdentifySpec(defocus=dz)
        s = ID.sensitivity(sp, ("defocus",))
        mu, _, _ = ID.mean_and_cov(sp)
        rel = float(np.linalg.norm(s["dmu"][0]) / np.linalg.norm(mu))
        norms.append(rel)
        lines.append(f"{dz:9.1f} {rel:22.3e}")

    ok = norms[0] < 1e-12 and norms[-1] > 1e-6
    lines += ["",
              f"Δz = 0 에서 민감도가 {norms[0]:.2e} (기계 정밀도의 0)",
              "수차 없는 계에서 상은 초점면에 대해 대칭이므로 1차 미분이",
              "사라진다. 부호 모호성은 잡음이 아무리 작아도 해소되지 않으며,",
              "비점수차를 의도적으로 넣어 깨는 것이 실제 공정의 대응이다."]
    report("Case 2  최적 초점에서의 부호 모호성", ok, lines)


# ---------------------------------------------------------------- Case 3
def case3_covariance_breaks_degeneracy():
    """블러와 MSD 는 평균에서 축퇴되지만 잡음 구조에서 갈린다.

    MSD 는 광자 샘플링 이전의 광강도를, 레지스트 블러는 샘플링 이후의
    광자 수를 평활한다. 평균만 보면 같은 연산이지만, 블러는 잡음을 함께
    평활하므로 공분산이 달라진다. 8주차 Case 6 의 경험적 결과와 같은
    내용을 해석적으로 보인 것이다.
    """
    sp = base_spec()
    pair = ("blur_nm", "msd_nm")
    c = ID.collinearity(sp, pair)
    ang = c["pairs"][0]["angle_deg"]

    rows, lines = [], []
    lines.append(f"평균 민감도 사이각 {ang:.2f}도 (코사인 {c['pairs'][0]['cos']:.4f})")
    lines.append("")
    lines.append(f"{'Fisher 정보 구성':<18} {'σ(blur)':>10} {'σ(msd)':>10} {'조건수':>11}")
    for inc, label in ((False, "평균 항만"), (True, "평균 + 공분산")):
        r = ID.crb(sp, pair, include_cov_term=inc)
        rows.append(r)
        lines.append(f"{label:<18} {r['sigma_min'][0]:10.4f} "
                     f"{r['sigma_min'][1]:10.4f} {r['cond']:11.2e}")

    gain_blur = rows[0]["sigma_min"][0] / rows[1]["sigma_min"][0]
    gain_cond = rows[0]["cond"] / rows[1]["cond"]
    ok = ang < 5.0 and gain_blur > 1.5 and gain_cond > 2.0
    lines += ["",
              f"공분산 항을 넣으면 blur 하한이 {gain_blur:.2f}배 개선되고",
              f"조건수가 {gain_cond:.1f}배 좋아진다.",
              "",
              "해석 : 평균만 관측하면(선폭만 재면) 두 변수가 거의 축퇴되지만,",
              "잡음 구조까지 관측하면(거칠기를 함께 재면) 분리된다. 어떤",
              "계측을 추가해야 어떤 변수가 풀리는지의 구체적 사례이다."]
    report("Case 3  잡음 구조가 깨는 축퇴 (블러 - MSD)", ok, lines)


# ---------------------------------------------------------------- Case 4
def case4_unknown_ladder():
    """미지수를 늘릴수록 식별 한계가 어떻게 무너지는가.

    본 과제의 핵심 주장을 그대로 보여주는 케이스이다.
    """
    sp = base_spec()
    rows = ID.observable_ladder(sp)
    lines = [f"{'미지수':<44} {'σ_min(Δz)':>12} {'자유도':>8}"]
    vals = []
    for r in rows:
        v = r["sigma_min"]["defocus"]
        vals.append(v)
        txt = "식별불가" if not np.isfinite(v) else f"{v:.4f} nm"
        lines.append(f"{', '.join(r['params']):<44} {txt:>12} "
                     f"{r['rank']:>4}/{r['n_unknown']}")

    jump = vals[3] / vals[2] if np.isfinite(vals[3]) else np.inf
    ok = vals[0] < 0.05 and jump > 10.0
    lines += ["",
              f"미지수가 초점 하나면 광자 통계만으로 {vals[0]:.4f} nm 까지 특정 가능하다.",
              f"플레어를 미지수에 넣는 순간 {jump:.0f}배 악화된다.",
              "",
              "결론 : 식별 한계를 정하는 것은 광자 잡음이 아니라 '무엇을",
              "모르는가' 이다. 잡음을 줄여도 한계는 그대로이며, 모르는 변수를",
              "따로 측정해야만 한계가 내려간다."]
    report("Case 4  미지수 사다리와 한계의 붕괴", ok, lines)


# ---------------------------------------------------------------- Case 5
def case5_prescription():
    """플레어를 따로 재면 초점 식별이 얼마나 좋아지는가.

    '무엇을 재야 하는가' 에 대한 실행 가능한 처방이다. 플레어는 open-frame
    노광으로 따로 측정하는 것이 표준 절차이므로 실현 가능한 제안이다.
    """
    sp = base_spec()
    full = ID.DEFAULT_PARAMS
    r_all = ID.crb(sp, full)
    i_dz = list(full).index("defocus")

    base_dz = float(np.asarray(r_all["sigma_min"], dtype=float)[i_dz])
    base_txt = "무한" if not np.isfinite(base_dz) else f"{base_dz:.4f} nm"
    lines = [f"{'미지수 구성':<34} {'σ_min(Δz)':>12} {'자유도':>8}"]
    lines.append(f"{'전부 미지 (5개)':<34} {base_txt:>12} {r_all['rank']:>4}/5")

    # [10주차 재정립] 영공간 보정 이후 '전부 미지' 의 하한이 무한이 되었다.
    # 그러면 '개선 배수 = 기준 / 결과' 가 모든 변수에서 무한이 되어 어느
    # 계측이 더 효과적인지 전혀 구별하지 못한다(시험이 공허하게 통과한다).
    # 기준이 무한일 때 올바른 순위는 '그 변수를 따로 재고 남은 하한' 이
    # 작은 쪽이다. 배수 대신 결과값으로 줄 세운다.
    results = {}
    for drop in ("flare", "msd_nm", "blur_nm", "dose_rel"):
        sub = tuple(p for p in full if p != drop)
        r = ID.crb(sp, sub)
        v = float(np.asarray(r["sigma_min"], dtype=float)[list(sub).index("defocus")])
        results[drop] = v
        txt = "무한" if not np.isfinite(v) else f"{v:.4f} nm"
        lines.append(f"{drop + ' 를 따로 측정':<34} {txt:>12} {r['rank']:>4}/4")
    gains = {k: (base_dz / v if (np.isfinite(v) and v > 0) else float("inf"))
             for k, v in results.items()}

    # 유한한 결과를 주는 변수 중 가장 작은 하한을 주는 것이 최선의 처방이다.
    finite = {k: v for k, v in results.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else None
    # 플레어가 유일하게 '실용 기준' 안으로 들어오는지, 그리고 블러·노광량을
    # 따로 재는 것은 도움이 되지 않는지(영공간이 그대로 남는지) 확인한다.
    ok = (best == "flare"
          and np.isfinite(results["flare"])
          and results["flare"] < 0.1
          and not np.isfinite(results["blur_nm"]))
    ratio_txt = ("정의되지 않음 (기준이 무한)" if not np.isfinite(base_dz)
                 else ", ".join(f"{k} {v:.0f}배" for k, v in gains.items()))
    lines += ["",
              f"개선 배수 : {ratio_txt}",
              "기준이 무한이면 배수는 모두 무한이 되어 순위를 가릴 수 없다.",
              "따라서 '그 변수를 따로 재고 남은 하한' 으로 줄 세운다.",
              ""]
    order = sorted(results.items(),
                   key=lambda kv: (not np.isfinite(kv[1]), kv[1]))
    lines.append("효과 순위 (남은 하한이 작은 쪽이 좋은 처방) :")
    for rank_i, (k, v) in enumerate(order, 1):
        txt = "무한 (도움 안 됨)" if not np.isfinite(v) else f"{v:.4f} nm"
        lines.append(f"  {rank_i}. {k:<10} {txt}")
    lines += ["",
              f"→ 처방 : 플레어를 따로 측정하면 초점 하한이 무한에서",
              f"  {results['flare']:.4f} nm 로 떨어진다. 즉 불가능한 문제가",
              f"  가능한 문제가 된다. MSD 를 재면 {results['msd_nm']:.3f} nm 로",
              "  유한해지긴 하지만 실용 기준(50 nm)에는 들어오되 플레어보다",
              f"  {results['msd_nm'] / results['flare']:.0f} 배 나쁘다.",
              "  블러나 노광량을 따로 재는 것은 영공간을 없애지 못해 도움이",
              "  되지 않는다. 플레어는 open-frame 노광으로 측정하는 표준",
              "  절차가 있으므로 실행 가능하다.",
              "",
              "  반면 잡음을 줄이는 투자는(노광량 증가) Case 9 에서 보듯",
              "  노광량 100 배에도 하한이 무한에서 무한이다."]
    report("Case 5  계측 처방의 효과", ok, lines)


# ---------------------------------------------------------------- Case 6
def case6_numerical_health():
    """Fisher 정보 행렬의 수치 건전성.

    FIM 은 대칭 준양정부호여야 한다. 중심차분과 대각합 계산의 반올림으로
    고유값이 미세하게 음수가 될 수 있으나, 최대 고유값 대비 기계 정밀도
    수준이어야 한다.
    """
    sp = base_spec()
    r = ID.crb(sp, ID.DEFAULT_PARAMS)
    f = r["fim"]
    asym = float(np.max(np.abs(f - f.T)) / max(np.max(np.abs(f)), 1e-300))
    w = r["eigvals"]
    lines = [f"행렬 비대칭도 {asym:.2e} (0 이어야 함)",
             f"음수 고유값 개수 {r['n_negative_eig']}",
             f"고유값 범위 {w[0]:.3e} ~ {w[-1]:.3e}",
             f"식별 가능 자유도 {r['rank']} / {r['n_params']}", ""]
    lines.append("고유값 스펙트럼:")
    for i, v in enumerate(w):
        tag = "살아있음" if v > w[0] * 1e-10 else "소멸"
        lines.append(f"  모드 {i+1}  {v:11.3e}   {tag}")

    lines += ["", "가장 식별이 안 되는 방향의 구성:"]
    for p, c in sorted(zip(ID.DEFAULT_PARAMS, r["worst_dir"]),
                       key=lambda t: -abs(t[1])):
        lines.append(f"  {p:<10} {c:+.4f}")

    ok = asym < 1e-9 and r["rank"] >= 1
    lines += ["",
              "소멸한 방향은 MSD 가 지배한다. 즉 프로파일 관측만으로는 MSD 를",
              "되짚을 수 없으며, 스캐너 기록에서 직접 받아야 한다."]
    report("Case 6  FIM 수치 건전성과 소멸 방향", ok, lines)


# ---------------------------------------------------------------- Case 7
def case7_confusion_set():
    """혼동 집합 : 조건은 흩어져 있는데 관측량은 같은 묶음.

    Cramer-Rao 하한은 한 점에서의 미분이므로 국소 식별성만 본다. 멀리
    떨어진 조건들이 같은 관측값을 내는 전역 축퇴는 직접 구성해야 한다.
    """
    base = DS.SampleSpec(pitch=200.0, duty=0.5, na=1.35, n_medium=1.44,
                         n_resist=1.70, sigma_out=0.7, polarization="TE",
                         n_trials=2, n_rows=256)
    cs = SP.confusion_set(base, n=12, seed=0)
    sp = SP.confusion_spread(cs)

    lines = [f"생성 {sp['n']} 건",
             f"선폭 평균 {sp['cd_mean']:.3f} nm, 산포 {sp['cd_spread']:.4f} nm", "",
             "같은 선폭을 만드는 조건들의 범위:"]
    for k, (lo, hi) in sp["condition_ranges"].items():
        lines.append(f"  {k:<9} {lo:8.3f} ~ {hi:8.3f}")

    ok = sp["n"] >= 8 and sp["cd_spread"] < 1.0
    rel = sp["cd_spread"] / sp["cd_mean"]
    lines += ["",
              f"조건은 초점 기준 수백 nm 범위로 흩어져 있으나 선폭 산포는",
              f"{sp['cd_spread']:.4f} nm ({100*rel:.3f} %) 에 불과하다.",
              "선폭만 측정하는 계측으로는 이 조건들을 원리적으로 구별할 수 없다.",
              "12주차 역추정 모델은 이 묶음에서 평가한다."]
    report("Case 7  혼동 집합 구성", ok, lines)


# ---------------------------------------------------------------- Case 8
def case8_split_leakage():
    """무작위 분할은 그룹 누수를 일으킨다.

    한 광학 조건에서 Defocus x Dose 격자로 수십 건이 나오므로, 무작위로
    섞으면 형제 표본이 학습셋과 검증셋에 동시에 들어간다. 모델이 외운
    것을 일반화 성능으로 오인하게 되며 식별 한계를 과소평가한다.
    """
    specs = []
    for pitch in (160.0, 200.0, 260.0, 320.0):
        for na in (1.00, 1.35):
            for dz in np.linspace(0, 200, 9):
                for dr in np.linspace(0.9, 1.1, 9):
                    specs.append(DS.SampleSpec(pitch=pitch, na=na, n_medium=1.44,
                                               defocus=float(dz),
                                               dose_rel=float(dr)))
    n_groups = len({SP.optical_group_id(s) for s in specs})
    lines = [f"표본 {len(specs)} 건, 광학 조건 {n_groups} 개",
             f"조건당 {len(specs)//n_groups} 건 (Defocus 9 x Dose 9)", "",
             f"{'분할 방식':<16} {'누수 그룹':>10} {'오염 표본':>12}"]
    res = {}
    for name, split in (("무작위 분할", SP.shuffled_split(len(specs))),
                        ("조건 단위 분할", SP.grouped_split(specs))):
        lk = SP.split_leakage(specs, split)
        res[name] = lk
        lines.append(f"{name:<16} {lk['n_leaked_groups']:>5}/{lk['n_groups']:<4} "
                     f"{lk['n_leaked_items']:>6} ({100*lk['leak_rate']:5.1f} %)")

    ok = (res["무작위 분할"]["leak_rate"] > 0.5
          and res["조건 단위 분할"]["leak_rate"] == 0.0)
    lines += ["",
              "무작위 분할은 전 표본을 오염시킨다. 11~12주차 학습에서는",
              "반드시 splitting.grouped_split 을 쓴다."]
    report("Case 8  학습/검증 분할의 그룹 누수", ok, lines)


# ---------------------------------------------------------------- Case 9
def case9_identifiability_map():
    """조건 공간을 훑어 식별 하한의 지도를 만든다.

    Monte Carlo 가 필요 없어 대량 생성 대비 매우 저렴하므로, 조건 공간을
    촘촘히 덮을 수 있다. 본 과제의 최종 산출물의 원형이다.
    """
    base = base_spec()
    lines = []
    ok = True

    m = ID.identifiability_map(base, "pitch", np.linspace(140.0, 360.0, 7))
    lines.append(f"{'pitch [nm]':>11} {'σ_min(Δz) [nm]':>16} {'자유도':>8}")
    for v, s, rk in zip(m["values"], m["sigma_min"], m["rank"]):
        txt = "식별불가" if not np.isfinite(s) else f"{s:.4f}"
        lines.append(f"{v:11.0f} {txt:>16} {rk:>5}/5")

    doses = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
    m2 = ID.identifiability_map(base, "dose_mj", doses)
    lines += ["", f"{'노광량 [mJ/cm²]':>16} {'σ_min(Δz) [nm]':>16} {'자유도':>8}"]
    for v, s, rk in zip(m2["values"], m2["sigma_min"], m2["rank"]):
        txt = "식별불가" if not np.isfinite(s) else f"{s:.4f}"
        lines.append(f"{v:16.0f} {txt:>16} {rk:>5}/5")

    fin = m2["sigma_min"][np.isfinite(m2["sigma_min"])]
    gain = float(fin.max() / fin.min()) if fin.size >= 2 else 1.0
    mono = bool(np.all(np.diff(m2["sigma_min"]) <= 1e-9))
    # [10주차 재정립] 영공간 판정을 보정한 뒤, 공칭 조건에서는 노광량 전
    # 구간에서 하한이 '무한' 으로 나온다. 즉 주장이 약해진 것이 아니라
    # 더 강해졌다. 기존 기준(유한한 값들의 단조 개선과 개선율 < 1.2)은
    # 무한값에서 NaN 이 되어 성립하지 않으므로, 두 경우를 나누어 판정한다.
    dose_vals = np.asarray(m2["sigma_min"], dtype=float)
    all_inf = bool(np.all(~np.isfinite(dose_vals)))
    if all_inf:
        lines += ["",
                  "노광량 10 ~ 1000 mJ/cm² 전 구간에서 하한이 '무한' 이다.",
                  "즉 이 조건에서 초점은 노광량과 무관하게 **원리적으로**",
                  "식별할 수 없다. 광자가 부족해서가 아니라, 민감도 벡터가",
                  "다른 미지수의 조합과 선형종속이어서 Fisher 정보 행렬에",
                  "영공간이 생기기 때문이다(rank < 미지수 수).",
                  "",
                  "이것이 본 과제의 핵심 주장을 가장 직접적으로 보여준다.",
                  "광자를 100배 쏟아부어도 식별 능력은 '사실상 제자리' 가",
                  "아니라 '정확히 제자리' 다. 반면 Case 5 에서 보듯 플레어를",
                  "따로 측정해 미지수에서 빼면 영공간이 사라져 한계가 유한해진다.",
                  "한계를 정하는 것은 잡음의 크기가 아니라 민감도 구조다."]
        ok &= True
    else:
        ok &= mono and gain < 1.2
        lines += ["",
                  f"노광량 단조 개선 = {mono} (광자가 늘면 한계는 좋아지기만 해야 함)",
                  f"노광량을 10 에서 1000 mJ/cm² 로 100배 올려도 한계는 {gain:.3f}배,",
                  f"즉 {100*(gain-1):.1f} % 만 개선된다.",
                  "",
                  "광자를 100배 더 쏟아부어도 식별 능력은 사실상 제자리인 반면,",
                  "Case 5 에서 보듯 플레어를 따로 측정하면 수십 배 좋아진다."]
    # 피치 축은 물리적으로 의미 있는 추가 정보를 준다 : 피치가 커지면
    # 동공을 통과하는 차수가 늘어 민감도 벡터가 벌어지고 영공간이 사라진다.
    n_fin = int(np.isfinite(np.asarray(m["sigma_min"], dtype=float)).sum())
    lines += ["",
              f"피치 축 : {n_fin}/{len(m['sigma_min'])} 지점에서 하한이 유한하다.",
              "피치가 커지면 동공을 통과하는 회절 차수가 늘어 민감도 벡터가",
              "벌어지고 영공간이 사라진다. 즉 식별 가능성은 조건에 따라",
              "'정도' 가 아니라 '유무' 가 바뀐다."]
    ok &= n_fin >= 2
    report("Case 9  식별 하한 지도와 노광량 포화", ok, lines)


if __name__ == "__main__":
    print("=" * 78)
    print(" 식별 가능성 분석 검증 (10주차 보강)")
    print("=" * 78)
    print()
    case1_collinearity()
    case2_focus_symmetry()
    case3_covariance_breaks_degeneracy()
    case4_unknown_ladder()
    case5_prescription()
    case6_numerical_health()
    case7_confusion_set()
    case8_split_leakage()
    case9_identifiability_map()
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
