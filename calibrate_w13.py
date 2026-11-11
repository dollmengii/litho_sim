"""13주차 : DOF 예측의 분할 적합예측 구간을 보정한다.

무엇을 하는가
-------------
데이터셋의 **검증 분할** 광학 조건을 뽑아, 각 조건에서

  - 대리 모형이 내놓는 DOF (추천기와 **똑같은 경로**: `recommend.evaluate`)
  - 시뮬레이터가 내놓는 DOF (같은 초점·노광 격자, `roundtrip_w13`)

를 짝지어 잔차를 모으고, 유한 표본 보정을 쓴 적합예측 분위수 `q` 를 낸다.
그리고 **시험 분할**에서 같은 `q` 의 실측 피복률을 센다. 보정에 쓴 분할에서
재면 당연히 맞으므로 그것은 검증이 아니다.

결과를 json 으로 저장하고, 추천기와 되돌림이 그 파일을 읽어 구간을 함께
보고한다.

왜 번들에 넣지 않고 따로 저장하는가
-----------------------------------
보정은 학습이 아니다. 시뮬레이터를 돌려야 하고(모형 학습 경로에 없다),
추천기의 격자 설정(`--n-focus`, `--n-dose`, `--focus-span`, `--el`)에
의존한다. 번들에 박아 넣으면 격자 설정이 바뀌어도 낡은 숫자가 따라다닌다.
파일에 **그 설정을 함께 적어** 두고, 읽는 쪽이 설정이 다르면 경고하게 하는
것이 맞다. 11주차 54 번의 복제 문제를 반대 방향으로 만들지 않기 위한 것이다.

왜 '추천기와 똑같은 경로' 가 중요한가
-------------------------------------
13주차에 되돌림이 손으로 적은 광학 인자를 써서 규약이 어긋난 적이 있다
(VERIFICATION_LIMITS 75 번). 보정도 같은 함정이 있다. 그래서 예측 DOF 는
`recommend.evaluate` 를 그대로 부르고, 참 DOF 는 `roundtrip_w13.simulate_window`
를 그대로 부른다. 둘 다 이미 규약 검사(`validate_w13` Case 6, 7)가 걸려 있다.

실행
----
    py calibrate_w13.py --models models/s13c --dataset dataset/merged13c.npz

    py calibrate_w13.py --models models/s13c --dataset dataset/merged13c.npz \\
        --n-cal 60 --n-test 40 --alpha 0.10 --out models/s13c_conformal.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import recommend as RC
import roundtrip_w13 as RT
from litho import conformal as CF
from litho import dataset as DS
from litho import process as PR


def _spec_from(cond: dict, pitch: float, duty: float) -> DS.SampleSpec:
    """조건 사전을 `SampleSpec` 으로. 이름이 다른 축은 그대로 넘긴다."""
    kw = {"pitch": pitch, "duty": duty}
    for k in ("na", "sigma_out", "sigma_in", "n_medium", "flare", "msd_nm",
              "blur_nm", "z5", "z6", "z7", "z8", "z11", "wavelength",
              "n_resist", "pole_offset", "pole_radius", "att_t"):
        if k in cond:
            kw[k] = float(cond[k])
    for key, table, name in (("pol_code", DS.POLARIZATION_CODES,
                              "polarization"),
                             ("source_code", DS.SOURCE_CODES, "source_shape"),
                             ("mask_code", DS.MASK_CODES, "mask_type")):
        if key in cond:
            kw[name] = table[int(round(cond[key]))]
    return DS.SampleSpec(**kw)


def pair_one(bundles: dict, feat: list[str], cond: dict, args,
             rel_p95: float) -> tuple[tuple[float, float] | None, str | None]:
    """조건 하나에서 ((예측 DOF, 참 DOF), None), 못 재면 (None, 사유)."""
    pitch = float(cond.get("pitch", args.pitch))
    duty = float(cond.get("duty", args.duty))
    spec = _spec_from(cond, pitch, duty)

    # 격자는 추천기와 같은 방식으로 잡는다. dz=0 을 격자에 고정하는
    # `centered_focus_grid` 를 그대로 쓴다(VERIFICATION_LIMITS 69 번).
    n_f = args.n_focus if args.n_focus % 2 == 1 else args.n_focus + 1
    focus, _note = PR.centered_focus_grid(-0.5 * args.focus_span,
                                          0.5 * args.focus_span, n_f)
    dose_rel = np.linspace(1.0 - 0.5 * args.dose_span,
                           1.0 + 0.5 * args.dose_span, args.n_dose)

    base = {k: float(v) for k, v in cond.items()
            if k not in CF.SWEEP_AXES}
    base.setdefault("pitch", pitch)
    base.setdefault("duty", duty)
    try:
        ev = RC.evaluate(bundles, base, feat, spec, focus, dose_rel,
                         rel_p95, args.el, False)
    except SystemExit as e:
        return None, str(e).splitlines()[0].replace("[오류] ", "")[:40]
    if not ev.get("ok"):
        return None, str(ev.get("reason", "평가 불가"))[:40]
    pred = float(ev["nominal"]["dof"])

    tol = float(ev["nominal"]["tol"])
    sim = RT.simulate_window(spec, focus, dose_rel, args.el, tol)
    if not sim.get("ok"):
        return None, str(sim.get("reason")
                         or sim.get("empty_reason") or "창 없음")[:40]
    if sim.get("undetermined"):
        # 참 DOF 가 값이 아니라 하한이다. 잔차가 뜻을 잃으므로 제외하고
        # 그 수를 보고한다. 보장은 그 조건부가 된다.
        return None, "격자 미결정"
    return (pred, float(sim["dof"])), None


def collect(bundles, feat, conds, args, rel_p95, label):
    pred, true, skip = [], [], {}
    for i, c in enumerate(conds, 1):
        got, why = pair_one(bundles, feat, c, args, rel_p95)
        if got is None:
            skip[why] = skip.get(why, 0) + 1
        else:
            pred.append(got[0])
            true.append(got[1])
        if i % 10 == 0 or i == len(conds):
            print(f"    {label} {i}/{len(conds)}  "
                  f"쓸 수 있는 짝 {len(pred)}", flush=True)
    return np.asarray(pred), np.asarray(true), skip


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DOF 예측의 분할 적합예측 구간 보정")
    ap.add_argument("--models", required=True,
                    help="번들 접두사 (예 models/s13c). 하나만 받는다")
    ap.add_argument("--dataset", required=True,
                    help="분할 정보가 있는 npz (merge_dataset.py 산출물)")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="1-alpha 가 표기 신뢰수준 (기본 0.10 -> 90 %%)")
    ap.add_argument("--n-cal", type=int, default=60,
                    help="보정에 쓸 검증 분할 조건 수")
    ap.add_argument("--n-test", type=int, default=40,
                    help="피복률 확인에 쓸 시험 분할 조건 수")
    ap.add_argument("--pitch", type=float, default=250.0)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--el", type=float, default=10.0)
    ap.add_argument("--focus-span", type=float, default=600.0)
    ap.add_argument("--n-focus", type=int, default=33)
    ap.add_argument("--dose-span", type=float, default=0.30)
    ap.add_argument("--n-dose", type=int, default=21)
    ap.add_argument("--pitch-window", type=float, default=0.15,
                    help="보정 조건을 요청 피치의 이 상대 범위 안으로 "
                         "제한한다 (기본 0.15 = +-15 %%). 0 이면 제한하지 "
                         "않고 데이터셋 전 범위를 모은다")
    ap.add_argument("--duty-window", type=float, default=0.06,
                    help="duty 의 절대 허용 범위 (기본 0.06)")
    ap.add_argument("--seed", type=int, default=1313)
    ap.add_argument("--out", default=None,
                    help="저장 경로. 기본 <models>_conformal.json")
    args = ap.parse_args()

    print("=" * 78)
    print(" DOF 예측의 분할 적합예측 구간 보정")
    print("=" * 78)

    bundles = RC.load_bundles(args.models, RC.LABELS_REQUIRED)
    ref = bundles[RC.LABELS_REQUIRED[0]]
    feat = list(ref["features"])
    rel_p95, rel_src, _ = RC.resolve_rel_p95(bundles, None)
    print(f"\n  모형 {args.models}, 입력 {len(feat)} 열")
    print(f"  모형 95 백분위 오차 {100 * rel_p95:.2f} %  (출처 : {rel_src})")

    d = np.load(args.dataset, allow_pickle=False)
    if "split_val" not in d or "split_test" not in d:
        print("\n[오류] 분할 정보가 없습니다. merge_dataset.py 로 병합한")
        print("       npz 를 주십시오. 여기서 무작위로 나누면 학습에 쓴")
        print("       조건이 보정에 섞여 피복 보장이 깨집니다.")
        return 1
    cond = np.asarray(d["conditions"], dtype=np.float64)
    ck = [str(s) for s in d["condition_keys"]]
    rng = np.random.default_rng(args.seed)

    # 피치·duty 로 걸러낸다. 왜 (13주차에 찾은 구멍)
    # ------------------------------------------------
    # 데이터셋의 피치는 100~400 nm 전 범위이고 추천기는 **한 피치**를 묻는다.
    # DOF 는 피치에 강하게 의존하므로, 전 범위에서 모은 잔차는 한 피치의
    # 잔차와 같은 분포가 아니다. 교환가능성이 깨지면 적합예측의 보장도
    # 깨진다. 그래서 요청 피치 근처로 제한한다.
    #
    # 피치를 **덮어쓰지** 않는 이유 : 덮어쓰면 그 조건이 더 이상 검증 분할의
    # 그룹이 아니다. 분할은 피치를 포함한 그룹 키로 나뉘므로, 피치를 바꾸면
    # 학습에 쓰인 그룹과 겹칠 수 있다. 표본이 줄어드는 대가를 치르고
    # 누설 가능성을 0 으로 둔다.
    def pick(part, n):
        rows = np.asarray(d[f"split_{part}"], dtype=np.int64)
        allc = CF.optical_conditions(cond, ck, rows)
        n_all = len(allc)
        allc = CF.near_request(allc, args.pitch, args.duty,
                               args.pitch_window, args.duty_window)
        n_near = len(allc)
        if len(allc) > n:
            sel = rng.choice(len(allc), size=n, replace=False)
            allc = [allc[int(i)] for i in sorted(sel)]
        return allc, n_all, n_near

    cal_c, n_cal_all, n_cal_near = pick("val", args.n_cal)
    tst_c, n_tst_all, n_tst_near = pick("test", args.n_test)
    if args.pitch_window > 0:
        print(f"\n  피치 {args.pitch:.0f} nm +-{100 * args.pitch_window:.0f} %, "
              f"duty {args.duty:.3f} +-{args.duty_window:.3f} 로 제한한다.")
        print("  데이터셋 피치는 전 범위이고 추천기는 한 피치를 묻는다. DOF 가")
        print("  피치에 강하게 의존하므로 전 범위에서 모은 잔차는 한 피치의")
        print("  잔차와 같은 분포가 아니다. 그러면 보장이 깨진다.")
    else:
        print("\n  [주의] --pitch-window 0 : 피치를 제한하지 않았다.")
        print("  데이터셋 전 범위의 잔차를 섞으므로 **교환가능성이 깨지고**")
        print("  구간의 피복 보장이 성립하지 않을 수 있다. 표본이 모자랄 때")
        print("  마지못해 쓰는 설정이며, 그 사실을 보고서에 적어야 한다.")
    print(f"\n  검증 분할 광학 조건 {n_cal_all} 개 -> 피치·duty 조건 "
          f"{n_cal_near} 개 -> 보정에 {len(cal_c)} 개")
    print(f"  시험 분할 광학 조건 {n_tst_all} 개 -> 피치·duty 조건 "
          f"{n_tst_near} 개 -> 피복 확인에 {len(tst_c)} 개")
    print("  두 분할은 **광학 조건 그룹 단위**로 나뉘어 있어 누설이 없다.")
    if n_cal_near < 9:
        print(f"\n  [주의] 피치 조건을 만족하는 검증 그룹이 {n_cal_near} 개뿐이다.")
        print("  alpha 0.10 은 최소 9 개가 필요하다. --pitch-window 를 넓히면")
        print("  표본이 늘지만 교환가능성이 약해진다. 둘의 교환이며, 넓혔으면")
        print("  얼마나 넓혔는지 보고서에 적을 것.")
    print(f"\n  격자 : 초점 {args.n_focus} x 노광 {args.n_dose}, "
          f"초점 폭 {args.focus_span:.0f} nm, EL {args.el:.0f} %")
    print("  추천기가 쓰는 격자와 같아야 한다. 다르면 보정량과 적용량이")
    print("  다른 양이 되어 보장이 깨진다. 저장 파일에 설정을 함께 적는다.")

    print("\n  보정 짝 수집")
    p_cal, t_cal, sk_cal = collect(bundles, feat, cal_c, args, rel_p95, "보정")
    print("\n  시험 짝 수집")
    p_tst, t_tst, sk_tst = collect(bundles, feat, tst_c, args, rel_p95, "시험")

    res = CF.residual_quantile(p_cal, t_cal, args.alpha)
    print(f"\n{'=' * 78}")
    print(" 결과")
    print("=" * 78)
    if sk_cal:
        print(f"\n  보정에서 제외 : "
              + ", ".join(f"{k} {v}" for k, v in sorted(sk_cal.items())))
    if res["q"] is None:
        print(f"\n  [구간 없음] {res['why']}")
        print("  최댓값으로 때우지 않는다. 그러면 보장을 흉내만 내는 것이다.")
        print("  --n-cal 을 늘리거나 --alpha 를 키워 다시 돌릴 것.")
        return 1

    print(f"\n  보정 표본 {res['n']} 개, 순서통계량 {res['k']} 번째")
    print(f"  적합예측 분위수 q = **{res['q']:.1f} nm**")
    print(f"  정확한 보장 피복률 = k/(n+1) = {100 * res['guarantee']:.2f} % "
          f"(표기 {100 * (1 - args.alpha):.0f} % 이상)")
    print(f"  잔차 중위 {res['r_median']:.1f} nm, 최대 {res['r_max']:.1f} nm")
    sm = res["signed_median"]
    if sm is not None:
        print(f"  부호 있는 잔차 중위 {sm:+.1f} nm — "
              + ("모형이 DOF 를 **크게** 본다(낙관적)" if sm > 0
                 else "모형이 DOF 를 작게 본다(보수적)" if sm < 0
                 else "치우침 없음"))
        print("    13주차 되돌림은 중위 DOF 차 <= 0 을 보고했다. 그쪽은")
        print("    실측-예측이므로 부호가 반대다. 두 숫자의 부호가 서로")
        print("    맞는지 확인할 것.")

    cov = CF.coverage(p_tst, t_tst, res["q"])
    if sk_tst:
        print(f"\n  피복 확인에서 제외 : "
              + ", ".join(f"{k} {v}" for k, v in sorted(sk_tst.items())))
    if cov["rate"] is None:
        print("\n  [피복 확인 불가] 시험 분할에서 쓸 수 있는 짝이 없다.")
    else:
        print(f"\n  **다른 분할** 실측 피복 {cov['hit']}/{cov['n']} = "
              f"{100 * cov['rate']:.1f} %")
        # 표본이 적으면 실측 피복률은 잡음이 크다. 보장 미달이라고 적기
        # 전에 그 잡음을 반드시 따져야 한다. 13주차에 몬테카를로 표준오차와
        # 적합예측 시뮬레이션에서 같은 실수를 두 번 했다.
        #
        # 정규근사 표준오차는 쓰지 않는다. 피복률이 0 이나 1 이면 SE 가 0 이
        # 되어 판정이 무의미해진다(실측 8/8 에서 바로 그렇게 나왔다). 이항
        # 분포의 **정확한** 단측 하한(Clopper-Pearson)을 쓴다.
        from scipy.stats import beta as _beta
        n, h = cov["n"], cov["hit"]
        lo = 0.0 if h == 0 else float(_beta.ppf(0.05, h, n - h + 1))
        print(f"    정확 단측 95 % 하한 (Clopper-Pearson) "
              f"{100 * lo:.1f} %  대  보장 {100 * res['guarantee']:.1f} %")
        if lo >= res["guarantee"] - 1e-12:
            print("    -> 보장 이상이다. 구간을 쓸 수 있다.")
        elif cov["rate"] >= res["guarantee"]:
            print("    -> 점추정은 보장 이상이지만 표본이 적어 하한이 못")
            print("       미친다. 어긋났다는 증거는 아니다. --n-test 를")
            print("       늘리면 판정이 선명해진다.")
        else:
            print("    -> **보장보다 낮다.** 교환가능성이 깨진 것이다.")
            print("       보정 조건(데이터셋 검증 분할)과 적용 조건의 분포가")
            print("       다르다는 뜻이고, 구간을 그대로 쓰면 안 된다.")

    # 격자 분해능 대조. 이것을 빼면 '모형 오차가 q 다' 로 읽히는데, 격자가
    # 거칠면 q 는 모형 오차가 아니라 **격자 간격**을 재고 있다. 실측 :
    # 초점 17 점 / 600 nm 에서 q = 75.0 nm 가 나왔고 그것은 격자 간격
    # 37.5 nm 의 정확히 2 배였다. DOF 는 격자 간격의 배수로만 나오므로
    # 당연한 일이고, 그 사실을 적지 않으면 숫자를 잘못 읽는다.
    step = args.focus_span / max(args.n_focus - 1, 1)
    print(f"\n  초점 격자 간격 {step:.1f} nm, q / 간격 = "
          f"{res['q'] / step:.2f} 배")
    if res["q"] <= 2.5 * step:
        print("    [주의] q 가 격자 간격의 2.5 배 이하다. 이 구간은 모형")
        print("    오차보다 **격자 분해능**을 재고 있을 수 있다. DOF 는")
        print("    격자 간격의 배수로만 나오므로 잔차도 그 배수로 양자화된다.")
        print("    치우치는 방향은 정해져 있지 않다. 거친 격자가 q 를 작게")
        print("    만들 수도 있다 — 실측으로 초점 17 점에서 75.0 nm 였던")
        print("    것이 33 점에서 93.8 nm, 61 점에서 90.0 nm 로 **커지며**")
        print("    수렴했다. 따라서 --n-focus 를 늘려 q 가 '줄어드는지' 가")
        print("    아니라 **안정되는지**를 봐야 한다. 안정된 값이 모형")
        print("    오차이고, 거친 격자의 값은 그 추정으로 쓸 수 없다.")
    else:
        print("    간격의 2.5 배를 넘으므로 격자 분해능이 아니라 모형")
        print("    오차가 q 를 지배한다고 볼 수 있다. 확인하려면 --n-focus")
        print("    를 두 배로 올려 q 가 안정되는지 보면 된다.")

    out = args.out or f"{args.models}_conformal.json"
    payload = {
        "models": args.models, "dataset": args.dataset,
        "alpha": args.alpha, "q_nm": res["q"], "n_cal": res["n"],
        "k": res["k"], "guarantee": res["guarantee"],
        "r_median": res["r_median"], "r_max": res["r_max"],
        "signed_median": res["signed_median"],
        "skipped_cal": sk_cal, "skipped_test": sk_tst,
        "coverage": cov,
        "focus_step_nm": args.focus_span / max(args.n_focus - 1, 1),
        "q_over_step": res["q"] / (args.focus_span
                                   / max(args.n_focus - 1, 1)),
        "grid": {"n_focus": args.n_focus, "n_dose": args.n_dose,
                 "focus_span": args.focus_span, "dose_span": args.dose_span,
                 "el": args.el, "pitch": args.pitch, "duty": args.duty},
        "pitch_window": args.pitch_window, "duty_window": args.duty_window,
        "n_near_cal": n_cal_near, "n_near_test": n_tst_near,
        "note": res["note"],
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n  저장 : {out}")
    print("  되돌림에 --conformal 로 넘기면 구간과 실측 피복률을 함께")
    print("  보고한다. 격자 설정이 다르면 그쪽이 경고한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
