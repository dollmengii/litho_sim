"""12주차 보고서용 그림을 만든다.

12주차의 내용은 **공정 조건 추천기와 그 검증 체계**다. 그림도 거기에
맞춘다. 데이터셋도 학습된 모형도 필요 없다 — 전부 시뮬레이터에서 직접
계산하므로 받은 그대로 돌아간다.

실행
----
    py figures_w12.py
    py figures_w12.py --outdir figures_w12

만들어지는 것
-------------
    w12_01_공정윈도.png        초점-노광 공정창과 DOF 구간의 의미
    w12_02_DOF구간.png         DOF 를 한 값이 아니라 구간으로 읽는 이유
    w12_03_부호거울.png        CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11)
    w12_04_초점격자.png        격자에 dz=0 을 고정해야 하는 이유
    w12_05_검증체계.png        세 층의 검증과 그 성격 구성
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import figlib
from litho import dataset as DS
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import resist as RS


def _window(spec, focus, dose_rel):
    """CD[노광, 초점] 행렬과 목표 CD. 데이터셋과 **같은 규약**을 쓴다."""
    g = G.build_grid(pitch=spec.pitch, n_periods=spec.n_periods,
                     dx_target=spec.dx_target)
    m = MK.line_space(g, cd=spec.duty * spec.pitch,
                      mask_type=spec.mask_type, att_t=spec.att_t)
    opt = spec.optics()
    tgt = spec.duty * spec.pitch
    i0 = IM.aerial_image(g, m, spec.wavelength, spec.na, defocus=0.0, **opt)
    e = PR.dose_to_size(g, i0, tgt, spec.threshold, max_width=spec.pitch)
    st = PR.focus_sweep(g, m, spec.wavelength, spec.na,
                        defocuses=focus, **opt)
    cd = np.empty((dose_rel.size, focus.size))
    for j in range(focus.size):
        for i, dr in enumerate(dose_rel):
            cd[i, j] = RS.extract_cd(g.x, st[j] * e * float(dr),
                                     spec.threshold,
                                     max_width=spec.pitch)["cd"]
    return cd, tgt


def fig_window(outdir):
    spec = DS.SampleSpec(pitch=250.0, duty=0.5, na=1.2, sigma_out=0.7,
                         z6=0.02, z11=0.01)
    focus = np.linspace(-250.0, 250.0, 51)
    dose = np.linspace(0.88, 1.12, 25)
    cd, tgt = _window(spec, focus, dose)
    err = 100.0 * (cd - tgt) / tgt
    pw = PR.process_window(cd, tgt, tol_frac=spec.pw_tol)

    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.4, 3.5))
    cs = a1.contourf(focus, dose, err, levels=np.linspace(-12, 12, 25),
                     cmap="RdBu_r", extend="both")
    a1.contour(focus, dose, np.abs(err), levels=[100 * spec.pw_tol],
               colors="k", linewidths=1.6)
    fig.colorbar(cs, ax=a1, label="CD 오차 [%]")
    a1.set_xlabel("초점 이탈 [nm]")
    figlib.ylab(a1, "노광량 배율")
    a1.set_title(f"CD 오차 지도 (검은 선 = 공차 ±{100 * spec.pw_tol:.0f} %)")

    a2.imshow(pw, origin="lower", aspect="auto", cmap="Greens",
              extent=[focus[0], focus[-1], dose[0], dose[-1]])
    cur = PR.el_dof_curve(pw, dose, focus)
    det = PR.dof_bracket_detail(cur, 10.0)
    row = det["lo_row"]
    if row is not None:
        lo, hi = row["focus_range"]
        dlo, dhi = row["dose_range"]
        a2.add_patch(figlib.plt.Rectangle(
            (lo, dlo), hi - lo, dhi - dlo, fill=False, edgecolor="crimson",
            lw=2.0, label=f"EL 10 % 직사각형 (DOF {det['lo']:.0f} nm)"))
        a2.legend(loc="upper right", framealpha=0.9)
    a2.set_xlabel("초점 이탈 [nm]")
    figlib.ylab(a2, "노광량 배율")
    a2.set_title("공정창(통과 영역)과 EL 기준 DOF")
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w12_01_공정윈도",
        f"CD 오차 지도와 공정창. NA {spec.na}, sigma {spec.sigma_out}, "
        f"피치 {spec.pitch:.0f} nm, 공차 ±{100 * spec.pw_tol:.0f} %. "
        f"붉은 직사각형은 노광 관용도 10 % 를 만족하는 최대 직사각형이고 "
        f"그 가로 폭이 DOF({det['lo']:.0f} nm)다.")


def fig_bracket(outdir):
    """DOF 를 **구간**으로 읽어야 하는 이유. 12주차 보완 65 번의 그림."""
    spec = DS.SampleSpec(pitch=250.0, duty=0.5, na=1.2, sigma_out=0.7)
    focus = np.linspace(-250.0, 250.0, 51)
    rows = []
    for n_d in (5, 9, 13, 21, 41):
        dose = np.linspace(0.88, 1.12, n_d)
        cd, tgt = _window(spec, focus, dose)
        pw = PR.process_window(cd, tgt, tol_frac=spec.pw_tol)
        det = PR.dof_bracket_detail(PR.el_dof_curve(pw, dose, focus), 10.0)
        rows.append((n_d, det["lo"], det["hi"]))
    n = np.array([r[0] for r in rows], float)
    lo = np.array([r[1] for r in rows], float)
    hi = np.array([r[2] for r in rows], float)

    fig, ax = figlib.plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(n, lo, hi, alpha=0.25, color="tab:blue",
                    label="증명 가능한 구간 [하한, 상한]")
    ax.plot(n, lo, "o-", color="tab:blue", label="하한 (순위에 쓰는 값)")
    ax.plot(n, hi, "s--", color="tab:red", label="상한")
    ax.set_xlabel("노광량 격자점 수")
    figlib.ylab(ax, "EL 10 % 기준 DOF [nm]")
    ax.set_title("DOF 는 한 값이 아니라 구간이다")
    ax.legend()
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w12_02_DOF구간",
        "노광량 격자를 촘촘히 하면 구간이 좁아진다. 격자가 거칠 때 DOF 를 "
        "한 값으로 적으면 그 값이 참값인지 격자 때문인지 구분되지 않는다. "
        "순위는 **하한**으로 매긴다. 상한으로 매기면 격자가 거친 후보가 "
        "위로 올라와 '모르면 모를수록 좋아 보이는' 정렬이 된다.")


def fig_mirror(outdir):
    """부호 거울. 음수 수차를 학습 범위 없이 평가할 수 있는 근거."""
    focus = np.linspace(-250.0, 250.0, 51)
    out = {}
    for tag, z6, z11 in (("(+z6, +z11)", 0.03, 0.02),
                         ("(-z6, -z11)", -0.03, -0.02)):
        spec = DS.SampleSpec(pitch=250.0, duty=0.5, na=1.2, sigma_out=0.7,
                             z6=z6, z11=z11)
        cd, tgt = _window(spec, focus, np.array([1.0]))
        out[tag] = cd[0]
    a = out["(+z6, +z11)"]
    b = out["(-z6, -z11)"]
    # 큰 초점 이탈에서는 패턴이 형성되지 않아 CD 가 NaN 이다. 둘 다 유한한
    # 점에서만 비교한다. NaN 을 섞으면 최대 차가 nan 으로 나와 그림의
    # 근거가 사라진다(실제로 그렇게 나왔다).
    d = np.abs(a - b[::-1])
    ok = np.isfinite(d)
    gap = float(np.max(d[ok])) if ok.any() else float("nan")
    n_bad = int((~ok).sum())

    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.4, 3.4))
    a1.plot(focus, a, lw=1.8, label="CD(dz, +z6, +z11)")
    a1.plot(focus, b, lw=1.4, ls="--", label="CD(dz, -z6, -z11)")
    a1.set_xlabel("초점 이탈 dz [nm]")
    a1.set_ylabel("CD [nm]")
    a1.set_title("수차 부호를 바꾸면 곡선이 좌우로 뒤집힌다")
    a1.legend()
    a2.plot(focus, a, lw=1.8, label="CD(dz, +z6, +z11)")
    a2.plot(focus, b[::-1], lw=1.4, ls="--",
            label="CD(-dz, -z6, -z11)  (초점도 뒤집음)")
    a2.set_xlabel("초점 이탈 dz [nm]")
    a2.set_ylabel("CD [nm]")
    a2.set_title(f"둘을 함께 뒤집으면 일치 (최대 차 {gap:.2e} nm)"
                 + (f"\n* 미형성 {n_bad} 점 제외" if n_bad else ""))
    a2.legend()
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w12_03_부호거울",
        f"CD(+dz, +z6, +z11) = CD(-dz, -z6, -z11) 가 최대 {gap:.2e} nm 로 "
        + (f"성립한다(패턴이 형성되지 않은 {n_bad} 점은 제외). " if n_bad
           else "성립한다. ") +
        "이 대칭 덕분에 음수 수차 장비를 **학습 범위를 넓히지 "
        "않고** 평가할 수 있다. 단 DOF 와 EL 은 폭이라 그대로지만 초점 "
        "오프셋은 위치라서 부호를 되돌려야 한다.")


def fig_focus_grid(outdir):
    """격자에 dz=0 을 고정해야 하는 이유 (보완 69 번)."""
    lo, hi = -183.0, 250.0
    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.4, 3.2))
    for ax, n in ((a1, 33), (a2, 41)):
        naive = np.linspace(lo, hi, n)
        cent, _ = PR.centered_focus_grid(lo, hi, n)
        ax.plot(naive, np.full(n, 1.0), "o", ms=4, color="tab:red",
                label=f"linspace  (최소 |dz| = {np.min(np.abs(naive)):.3f} nm)")
        ax.plot(cent, np.full(cent.size, 0.6), "s", ms=4, color="tab:blue",
                label=f"중심 고정  (최소 |dz| = {np.min(np.abs(cent)):.3f} nm)")
        ax.axvline(0.0, color="k", lw=1.2, ls=":")
        ax.set_ylim(0.3, 1.3)
        ax.set_yticks([])
        ax.set_xlabel("초점 격자점 [nm]")
        ax.set_title(f"격자점 {n} 개, 범위 [{lo:.0f}, {hi:.0f}] nm")
        ax.legend(loc="upper center", fontsize=7.5)
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w12_04_초점격자",
        "격자점 수가 홀수여도 범위가 비대칭이면 dz=0 이 격자에 들어가지 "
        "않는다(점선이 붉은 점 사이를 지난다). 최적 초점을 격자점으로 "
        "갖지 못하면 창의 중심이 격자 간격만큼 치우친다. 중심 고정 격자는 "
        "간격을 유지하면서 dz=0 을 격자에 넣는다.")


def fig_verify(outdir):
    """검증 체계의 세 층과 성격 구성. 과장을 막기 위한 그림."""
    rows = [
        ["① 회귀 시험", "validate*.py", "270 종", "바뀌었는지"],
        ["② 기준값 없는 수치 검증", "verify.py A~K", "판정 10 항목",
         "물리가 강제하는 것"],
        ["③ 문서-코드 정합성", "claims.py", "44 항목", "문서가 코드와 같은가"],
    ]
    f1 = figlib.table_fig(
        rows, ["층", "도구", "규모", "무엇을 지키는가"],
        "세 층의 검증 — 각각 다른 것을 지킨다",
        col_widths=[0.26, 0.2, 0.18, 0.3])
    figlib.save(f1, outdir, "w12_05a_검증체계",
                "회귀는 '바뀌었는지' 를 지키고, verify 는 물리가 강제하는 "
                "등식·대칭·순서만 판정하며, claims 는 문서가 코드와 같은 "
                "숫자를 말하는지 본다. 세 층은 겹치지 않는다.")

    labels = ["해석\n(시뮬레이터 밖 기준)", "불변량", "기록\n(사후 임계값)",
              "정합성", "성능"]
    vals = [20, 19, 14, 8, 6]
    cols = ["#2d7d46", "#4a90d9", "#d9822b", "#8e7cc3", "#9aa0a6"]
    fig, ax = figlib.plt.subplots(figsize=(6.6, 3.2))
    b = ax.barh(labels[::-1], vals[::-1], color=cols[::-1])
    ax.bar_label(b, padding=3, fontsize=8)
    ax.set_xlabel("항목 수")
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("회귀의 성격 구성 (분류표 67 항목)")
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w12_05b_성격구성",
        "회귀를 한 숫자로 묶어 '전부 검증' 이라고 쓰면 과장이다. 해석해나 "
        "문헌값처럼 **시뮬레이터 밖의 기준**에 대조한 것만이 '틀렸는지' 를 "
        "지키고, '기록' 은 측정값에 사후 임계값을 붙인 것이라 회귀 감시용 "
        "이다. 그래서 성격별로 나누어 집계한다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="12주차 보고서용 그림 생성")
    ap.add_argument("--outdir", default="figures_w12")
    args = ap.parse_args()
    out = figlib.setup(args.outdir)
    print("=" * 74)
    print(" 12주차 보고서용 그림")
    print("=" * 74)
    print("\n  데이터셋도 학습된 모형도 필요 없습니다. 시뮬레이터에서")
    print("  직접 계산하므로 받은 그대로 돌아갑니다.\n")
    for fn in (fig_window, fig_bracket, fig_mirror, fig_focus_grid,
               fig_verify):
        fn(out)
        print()
    print("=" * 74)
    print(f" 완료. {out}/ 에 PNG 가 있습니다. 300 dpi 이므로 문서에")
    print(" 넣고 폭을 줄여도 글자가 읽힙니다.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
