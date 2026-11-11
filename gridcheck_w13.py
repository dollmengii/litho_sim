"""제약 관문이 **격자를 촘촘히 하면 수렴하는가.**

왜 이 도구가 필요한가
---------------------
13 주차 되돌림을 `--n-focus 33` 과 `--n-focus 257` 로 두 번 돌렸다. 모형도
조건도 같았는데 **예측 통과군이 15 개에서 2 개로** 줄었다. 바뀐 것은 격자
하나뿐이다.

이유는 제약의 **생김새**에 있다. 추천기는

    nils_min = min  over window of nils
    cdu_max  = max  over window of (cd_3sigma / cd)

를 기준과 비교한다. 격자 위의 최솟값·최댓값은 **격자를 촘촘히 할수록
단조로 나빠진다**(점이 늘면 더 나쁜 점을 찾을 기회만 늘어난다). 그러므로
이 관문이 쓸 만하려면 그 단조 변화가 **수렴**해야 한다. 수렴하지 않으면
'통과' 는 물리가 아니라 격자 선택의 결과가 된다.

시뮬레이터 쪽은 수렴한다 (별도 측정)
------------------------------------
같은 조건에서 시뮬레이터로 직접 재면 `nils_min` 이 n_focus 17 -> 257 에서
4.3181 -> 4.0960 (5 % 변화) 으로 거의 평평하고, DOF 는
75.0 -> 93.8 -> 103.1 -> 107.8 -> 110.2 nm 로 **차이가 절반씩 줄며**
수렴한다(이산화 폭이 한 눈금씩 메워지는 정상 거동).

즉 물리량은 n_focus 129 쯤에서 수렴한다. 그런데 **예측** 통과군은 같은
구간에서 15 -> 2 로 무너졌다. 그러면 발산하는 것은 물리가 아니라 **대리
모형의 격자 규모 예측 잡음**이다. 점을 촘촘히 깔면 모형이 튀는 점을 더
많이 만나고, 최솟값·최댓값만 보는 관문이 그 튐을 그대로 받는다.

이 도구는 그 진단을 **번들로 직접 확인**한다. 대리 모형만 쓰므로 수 초면
끝난다(시뮬레이터를 돌리지 않는다).

읽는 법
-------
- `통과` 가 격자를 촘촘히 해도 안정되면 관문은 쓸 수 있다. 격자 하나를
  규약으로 못 박고 보고서에 적으면 된다.
- `통과` 가 계속 줄어들면 관문은 **재현되지 않는다.** 그때는 통과/미통과를
  모형의 판단으로 보고할 수 없다. 이것은 순위가 나쁜 것보다 **센** 결함이다
  — 순위를 재기 전에 선별이 격자에 흔들린다는 뜻이다.
- `cdu_max 중위` 와 `nils_min 중위` 가 함께 나빠지는지 본다. 물리량이
  수렴하는 구간에서 이 둘이 계속 나빠지면 모형 잡음이 원인이다.
"""

from __future__ import annotations

import argparse

import numpy as np

import recommend as RC
from litho import dataset as DS


def one_grid(bundles, feat, tool, args, n_focus: int) -> dict:
    """한 격자 밀도에서 전수 평가하고 통과 수와 제약값 분포를 돌려준다."""
    try:
        r = RC.rank(bundles, feat, dict(tool), args.pitch, args.duty,
                    args.rel_p95, args.n_na, args.n_sigma,
                    args.focus_span, n_focus, args.dose_span, args.n_dose,
                    args.el, False)
    except SystemExit as e:
        return {"n_focus": n_focus, "refused": str(e)}
    rows = r["rows"]
    ok = [x for x in rows if x["nils_ok"] and x["cdu_ok"]]
    cdu = np.array([x["cdu_max"] for x in rows
                    if np.isfinite(x.get("cdu_max", np.nan))])
    nil = np.array([x["nils_min"] for x in rows
                    if np.isfinite(x.get("nils_min", np.nan))])
    dof = np.array([x["nominal"]["dof"] for x in rows
                    if np.isfinite(x["nominal"]["dof"])])
    # [13차 보완 95 번] **물리 범위를 벗어난 예측을 센다.**
    # NILS 는 `litho/resist._nils` 에서 `CD * |d(ln I)/dx|` 로 정의된다.
    # 절댓값에 CD(>0)를 곱하므로 **0 이상이 구조적으로 보장**되고, 따라서
    # 학습 라벨도 전부 0 이상이다. 그런데 회귀 모형은 그 범위를 지키지
    # 않는다 — `recommend.surrogate_cd_matrix` 는 `model.predict` 를 그대로
    # 돌려주고 범위 검사가 없다.
    #
    # 음수 NILS 가 섞이면 `nils_min = min over window` 가 그 값을 집어서
    # 후보를 떨어뜨린다. 그러면 관문은 물리가 아니라 **모형이 범위를 벗어난
    # 지점**을 재고 있는 것이다. 격자를 촘촘히 할수록 그런 점을 더 많이
    # 만나므로 통과 수가 계속 줄어든다.
    #
    # 여기서 **고치지 않고 센다.** 값을 잘라 올리면(clip) 관문이 멀쩡해
    # 보이면서 모형 실패가 숨는다. 13 주차의 일은 그 실패를 드러내는 것이다.
    bad_nils = [x for x in rows
                if np.isfinite(x.get("nils_min", np.nan))
                and x["nils_min"] < 0.0]
    bad_cdu = [x for x in rows
               if np.isfinite(x.get("cdu_max", np.nan))
               and x["cdu_max"] < 0.0]
    return {
        "n_focus": n_focus,
        "n_rows": len(rows),
        "n_pass": len(ok),
        "n_undet": len(r["undetermined"]),
        "cdu_med": float(np.median(cdu)) if cdu.size else float("nan"),
        "cdu_max": float(np.max(cdu)) if cdu.size else float("nan"),
        "nils_med": float(np.median(nil)) if nil.size else float("nan"),
        "nils_worst": float(np.min(nil)) if nil.size else float("nan"),
        "n_bad_nils": len(bad_nils),
        "n_bad_cdu": len(bad_cdu),
        "dof_top": float(np.max(dof)) if dof.size else float("nan"),
    }


def verdict(good: list[dict]) -> tuple[str, list[str]]:
    """통과 수가 격자에 수렴했는지 판정한다. **인쇄와 분리해 둔다.**

    분리하는 이유 : 이 판정이 13 주차 결론을 가르므로 회귀로 고정해야 하고,
    인쇄문에 섞여 있으면 시험할 수 없다. 92 번에서 같은 교훈을 얻었다 —
    판정과 인쇄가 한 덩어리면 서로 어긋나도 드러나지 않는다.

    수렴 여부는 마지막 두 격자로 보되, **전체 낙폭도 함께** 본다.

    [13차 보완 95 번] 처음 판은 마지막 두 격자만 보았다. 그래서 통과 수가
    27 -> 15 -> 6 -> 2 -> 2 로 **13 배 무너진** 실측 결과에 '안정되었다,
    관문을 쓸 수 있다' 를 찍었다. 마지막 두 격자가 2 -> 2 로 같았기
    때문이다. 틀린 말은 아니었지만(통과 수는 2 로 수렴했다) **읽는 사람을
    오도했다.** 수렴했다는 것과 그 수렴값을 믿을 수 있다는 것은 다른
    얘기다. 그래서 전체 낙폭을 따로 적고, 낙폭이 크면 '거친 격자가
    과대평가했다' 를 분명히 적는다.

    그리고 물리 범위를 벗어난 예측이 있으면 **그것이 다른 모든 판정보다
    앞선다.** 음수 NILS 를 집어서 떨어뜨린 통과/미통과는 수렴하든 안 하든
    물리의 판단이 아니다.
    """
    first, last = good[0], good[-1]
    a, b = good[-2], good[-1]
    drift = abs(b["n_pass"] - a["n_pass"]) / max(a["n_pass"], 1)
    fall = (last["n_pass"] / first["n_pass"]) if first["n_pass"] else 1.0
    out = [f"    통과 수 : {first['n_focus']} 점에서 {first['n_pass']} 개 -> "
           f"{last['n_focus']} 점에서 {last['n_pass']} 개"
           + (f"  ({1 / fall:.1f} 배 감소)" if 0 < fall < 1 else ""),
           f"    마지막 두 격자({a['n_focus']} -> {b['n_focus']}) 변화 "
           f"{100 * drift:.1f} %"]

    bad = max(x.get("n_bad_nils", 0) for x in good)
    if bad:
        bad_last = last.get("n_bad_nils", 0)
        out += [f"    물리 범위를 벗어난 예측 : 음수 NILS 후보 최대 {bad} 개"
                f" (가장 촘촘한 격자에서 {bad_last}/{last['n_rows']},"
                f" 최악 {last.get('nils_worst', float('nan')):+.3f})",
                "    -> **관문이 물리가 아니라 모형 실패를 재고 있다.**",
                "       NILS 는 CD x |d(ln I)/dx| 이므로 0 이상이 구조적으로",
                "       보장되고 학습 라벨도 전부 0 이상이다. 음수는 회귀",
                "       모형이 범위를 벗어난 것이다. min over window 가 그",
                "       값을 집어 후보를 떨어뜨리므로, 통과/미통과를 모형의",
                "       물리 판단으로 보고할 수 없다. 격자를 촘촘히 할수록",
                "       그런 점을 더 많이 만나 통과가 줄어든다.",
                "       고칠 곳은 둘이다 : (1) 예측에 물리 범위를 강제하고",
                "       (2) 범위를 벗어난 후보는 '위반' 이 아니라 '판정 불가'",
                "       로 분류해야 한다. 음수 NILS 는 '물리가 안 된다' 는",
                "       말이 아니다. 둘 다 12 주차 설계 변경이다."]
        return "nonphysical", out

    if last["n_pass"] == 0 and first["n_pass"] > 0:
        key = "collapse"
        out += ["    -> **관문이 수렴하지 않는다.** 격자를 촘촘히 하니 통과가",
                "       0 이 되었다. 통과/미통과를 모형의 판단으로 보고할 수",
                "       없다. 순위가 나쁜 것보다 센 결함이다 - 순위를 재기",
                "       전에 선별 자체가 격자 선택에 흔들린다."]
    elif drift > 0.20:
        key = "drifting"
        out += ["    -> **아직 수렴하지 않았다.** 마지막 두 격자에서 통과 수가",
                "       20 % 넘게 바뀐다. 더 촘촘한 격자까지 보아야 한다."]
    elif fall < 0.5:
        key = "stable_coarse_wrong"
        out += ["    -> 통과 수는 수렴했지만 **거친 격자가 크게 과대평가했다.**",
                "       수렴값은 쓸 수 있으나 거친 격자로 얻은 이전 결과는",
                "       무효다. 보고서에 격자를 반드시 함께 적어야 하고,",
                "       격자가 다른 결과를 나란히 비교하면 안 된다."]
    else:
        key = "stable"
        out += ["    -> 통과 수가 안정되었다. 그 격자를 규약으로 못 박고",
                "       보고서에 격자를 함께 적으면 관문을 쓸 수 있다."]
    return key, out


def main() -> int:
    p = argparse.ArgumentParser(
        description="제약 관문의 격자 수렴성을 본다 (대리 모형만, 수 초)")
    p.add_argument("--models", required=True)
    p.add_argument("--pitch", type=float, default=250.0)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--n-na", type=int, default=17)
    p.add_argument("--n-sigma", type=int, default=17)
    p.add_argument("--focus-span", type=float, default=600.0)
    p.add_argument("--dose-span", type=float, default=0.30)
    p.add_argument("--n-dose", type=int, default=21)
    p.add_argument("--el", type=float, default=10.0)
    p.add_argument("--grids", default="17,33,65,129,257",
                   help="볼 n_focus 들 (홀수여야 한다)")
    p.add_argument("--flare", type=float, default=0.02)
    p.add_argument("--msd", type=float, default=2.0)
    p.add_argument("--blur", type=float, default=10.0)
    p.add_argument("--z6", type=float, default=0.02)
    p.add_argument("--z11", type=float, default=0.01)
    p.add_argument("--pol", default="TE", choices=DS.POLARIZATION_CODES)
    p.add_argument("--source", default="conventional",
                   choices=DS.SOURCE_CODES)
    p.add_argument("--mask", default="binary", choices=DS.MASK_CODES)
    args = p.parse_args()

    bundles = RC.load_bundles(args.models, RC.LABELS_REQUIRED)
    feat = list(next(iter(bundles.values()))["features"])
    args.rel_p95, src, _ = RC.resolve_rel_p95(bundles, None)

    # 범주 열은 번들에서 **부호화된 이름**(`pol_code` 등)으로 쓰인다.
    # 이름이 다르면 "값이 주어지지 않았습니다" 로 거부된다. 되돌림과 같은
    # 표를 써서 맞춘다(규약이 어긋나면 비교 자체가 성립하지 않는다).
    tool = {"flare": args.flare, "msd_nm": args.msd, "blur_nm": args.blur,
            "z6": args.z6, "z11": args.z11,
            "pol_code": float(DS.POLARIZATION_CODES.index(args.pol)),
            "source_code": float(DS.SOURCE_CODES.index(args.source)),
            "mask_code": float(DS.MASK_CODES.index(args.mask))}
    tool = {k: v for k, v in tool.items() if k in feat}
    missing = [c for c in feat
               if c.endswith("_code") and c not in tool]
    if missing:
        print(f"  [주의] 번들이 요구하는 범주 열 {missing} 을 채우지 못했다.")

    print("=" * 78)
    print(" 제약 관문의 격자 수렴성")
    print("=" * 78)
    print(f"  모형 {args.models} / 95 백분위 오차 "
          f"{100 * args.rel_p95:.2f} % (출처 : {src})")
    print(f"  피치 {args.pitch:g} duty {args.duty:g}, 후보 격자 "
          f"{args.n_na} x {args.n_sigma}")
    print(f"  공구 상태 : " + ", ".join(f"{k}={v:g}" for k, v in tool.items()
                                        if isinstance(v, (int, float))))
    print()
    print("  n_focus   후보   통과   미정   cdu_max 중위   cdu_max 최대   "
          "nils_min 중위   nils 최악   음수NILS   최대 DOF")
    print("  " + "-" * 112)

    grids = [int(x) for x in args.grids.split(",") if x.strip()]
    res = []
    for nf in grids:
        if nf % 2 == 0:
            print(f"  {nf:7d}   건너뜀 (초점 격자는 홀수여야 한다)")
            continue
        r = one_grid(bundles, feat, tool, args, nf)
        res.append(r)
        if "refused" in r:
            print(f"  {nf:7d}   거부 : {r['refused'][:60]}")
            continue
        print(f"  {r['n_focus']:7d}  {r['n_rows']:5d}  {r['n_pass']:5d}  "
              f"{r['n_undet']:5d}   {100 * r['cdu_med']:11.2f} %  "
              f"{100 * r['cdu_max']:11.2f} %   {r['nils_med']:12.3f}  "
              f"{r['nils_worst']:10.3f}   {r['n_bad_nils']:7d}   "
              f"{r['dof_top']:7.1f}")

    good = [r for r in res if "refused" not in r]
    if len(good) < 2:
        print("\n  격자가 둘 이상 필요하다.")
        return 1

    print()
    print("  [판정]")
    key, lines = verdict(good)
    for ln in lines:
        print(ln)
    print()
    print("    참고 (시뮬레이터로 따로 잰 값) : 같은 조건에서 물리량은")
    print("    수렴한다 - nils_min 4.3181 -> 4.0960 (n_focus 17 -> 257, 5 %),")
    print("    DOF 75.0 -> 93.8 -> 103.1 -> 107.8 -> 110.2 nm (차이가 절반씩).")
    print("    시뮬레이터 NILS 는 늘 양수다. 물리량이 수렴하는 구간에서")
    print("    예측 통과가 계속 줄면 원인은 물리가 아니라 모형이다. 그 중")
    print("    **음수 NILS** 는 모형이 물리 범위를 벗어난 것이라서, 격자가")
    print("    아니라 예측 자체를 고쳐야 한다(위 판정이 그 경우를 가린다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
