"""13주차 보고서용 그림을 만든다.

13주차의 내용은 **되돌림 검증, 신뢰 구간, 계측 규격**이다.

그림이 두 묶음으로 갈린다.

  (가) 시뮬레이터만 있으면 되는 것 — 받은 그대로 돌아간다
       w13_01_축퇴와LER.png     주장 3 : 평균은 같고 LER 로 갈린다
       w13_02_계측규격.png      얼마나 정밀하게 재야 분리되는가
       w13_03_어디서재나.png    기준점에 따라 비용이 2.7 배 바뀐다
       w13_03b_외삽참조분포.png 검토로 잡은 것 : 중복이 백분위를 포화시켰다

  (나) **그쪽 실행 결과가 필요한 것** — roundtrip 이 남긴 json 을 읽는다
       w13_04_되돌림.png        예측 DOF 대 실측 DOF
       w13_05_판정.png          A·B·C 판정표
       w13_06_CDU.png           추천 조건에서의 실측 CDU
       w13_07_속도.png          대리 모형이 몇 배 빠른가

실행
----
    py figures_w13.py                       (가) 만 만든다
    py figures_w13.py --result w13_result.json    (가)+(나) 전부

json 은 되돌림을 돌릴 때 만들어진다 :

    py roundtrip_w13.py ... --save-json w13_result.json

**숫자를 손으로 옮겨 적지 않는다.** 그림이 json 을 직접 읽는다. 손으로
옮기면 그 순간 문서-코드 드리프트가 생기고, 그것이 `claims.py` 가 막으려는
바로 그 실패다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import figlib
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import metrospec as MS
from litho import resist as RS
from litho import stochastic as ST

#: 주장 3(verify.py H5)과 같은 조건. `metrospec_w13.COND` 와 맞춘다.
COND = dict(pitch=200.0, n_periods=6, dx_target=2.0, duty=0.5,
            wavelength=193.0, na=1.35, n_medium=1.44, sigma_out=0.7,
            dose_mj=30.0, dy=2.0, n_rows=512, n_trials=8)


def _setup():
    g = G.build_grid(pitch=COND["pitch"], n_periods=COND["n_periods"],
                     dx_target=COND["dx_target"])
    m = MK.line_space(g, duty=COND["duty"])
    i0 = np.asarray(IM.aerial_image(
        g, m, COND["wavelength"], COND["na"], n_medium=COND["n_medium"],
        sigma_out=COND["sigma_out"])).real
    e = RS.image_extrema(g.x, i0)
    thr = 0.5 * (float(e["i_max"]) + float(e["i_min"]))
    return g, m, thr


def _obs(g, m, thr, msd, blur, seed):
    ii = np.asarray(IM.aerial_image(
        g, m, COND["wavelength"], COND["na"], n_medium=COND["n_medium"],
        sigma_out=COND["sigma_out"], msd_nm=float(msd))).real
    r = ST.monte_carlo(g.x, ii, g.dx, COND["dose_mj"], COND["wavelength"],
                       thr, n_rows=COND["n_rows"], blur_nm=float(blur),
                       n_trials=COND["n_trials"], seed=int(seed),
                       dy=COND["dy"], max_width=g.pitch)
    return r, ii


def fig_degeneracy(outdir):
    """주장 3 : 결정론 평균은 **정확히** 같고 LER 로 갈린다."""
    g, m, thr = _setup()
    tot = 8.0
    b4 = float(np.sqrt(tot ** 2 - 16.0))
    alloc = [(0.0, tot), (4.0, b4), (b4, 4.0), (tot, 0.0)]
    det, ler, lab = [], [], []
    for msd, blur in alloc:
        r, ii = _obs(g, m, thr, msd, blur, 17)
        ib = IM.apply_msd(ii, blur, g.dx) if blur > 0 else ii
        det.append(float(RS.extract_cd(g.x, ib, thr,
                                       max_width=g.pitch)["cd"]))
        ler.append(float(r["ler_3sigma"]))
        lab.append(f"MSD {msd:.1f}\n블러 {blur:.1f}")
    det = np.array(det)
    ler = np.array(ler)
    spread = (ler.max() - ler.min()) / ler.max()

    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.0, 3.4))
    a1.bar(lab, det - det.mean(), color="#4a90d9")
    figlib.ylab(a1, "결정론 CD − 평균 [nm]")
    a1.set_title(f"결정론 CD : 범위 {det.max() - det.min():.1e} nm")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylim(-1e-13, 1e-13)
    a2.bar(lab, ler, color="#d9822b")
    figlib.ylab(a2, "LER 3σ [nm]")
    a2.set_title(f"LER : 범위 {100 * spread:.0f} %")
    fig.suptitle("합성 평활 8 nm 고정 · MSD↔블러 배분만 바꿈", y=1.02)
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13_01_축퇴와LER",
        f"MSD 와 레지스트 블러는 합성 분산이 같으면 결정론 CD 가 기계 "
        f"정밀도({det.max() - det.min():.1e} nm)로 동일하다. 그런데 LER 은 "
        f"{100 * spread:.0f} % 범위로 갈린다. MSD 는 광자 샘플링 이전, "
        f"블러는 이후에 작용하기 때문이다. 임계값은 평활 없는 상에서 한 번 "
        f"잡아 고정했다 — 배분마다 다시 잡으면 그 규약이 정보를 흘려 축퇴가 "
        f"깨진 것처럼 보인다.")


def fig_spec(outdir):
    """계측 규격 : 기울기와 CRB, 그리고 목표를 맞추는 예산."""
    gc = np.array([0.02896, 0.09469])
    gl = np.array([0.00110, -0.02486])
    ler0 = 0.2257
    sc = 0.5
    rel = np.logspace(-2.3, -0.3, 60)
    sd_b, sd_m = [], []
    for r in rel:
        cr = MS.crb(gc, gl, sigma_cd=sc, sigma_ler=r * ler0)
        sd_b.append(cr["sd_blur"])
        sd_m.append(cr["sd_msd"])

    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.4, 3.5))
    a1.loglog(100 * rel, sd_b, lw=2.0, color="#2d7d46", label="레지스트 블러")
    a1.loglog(100 * rel, sd_m, lw=2.0, color="#c0392b", label="MSD")
    a1.axhline(10.0, color="#2d7d46", ls=":", lw=1.2)
    a1.axhline(6.0, color="#c0392b", ls=":", lw=1.2)
    a1.annotate("블러 범위 폭 10 nm", (0.6, 10.5), fontsize=7.5,
                color="#2d7d46")
    a1.annotate("MSD 범위 폭 6 nm", (0.6, 6.3), fontsize=7.5, color="#c0392b")
    a1.set_xlabel("LER 계측 정밀도 [LER 값의 %]")
    figlib.ylab(a1, "달성 가능한 표준편차 [nm]  (CRB)")
    a1.set_title(f"CD 1σ {sc} nm 고정 · 1 지점")
    a1.legend()

    rows = [["레지스트 블러", "1.07", "10 nm 의 10.7 %", "LER",
             "0.53 nm / 1 지점", "8.5 % / 2 지점"],
            ["MSD", "15.32", "6 nm 의 255 %", "CD",
             "0.0235 nm / 455 지점", "2.6 % / 15 지점"]]
    f2 = figlib.table_fig(
        rows, ["미지수", "sd [nm]", "자기 범위 대비", "한계 관측량",
               "목표 ±1 nm : CD", "목표 ±1 nm : LER"],
        "현재 계측(CD 1σ 0.5 nm, LER 10 % 상대, 1 지점)에서",
        col_widths=[0.15, 0.1, 0.17, 0.13, 0.22, 0.2],
        highlight_rows=(1,))
    figlib.save(f2, outdir, "w13_02b_계측예산",
                "블러는 표준 CD-SEM 과 LER 계측으로 몇 지점만 평균해도 "
                "닿는다. MSD 는 자기 범위보다 넓은 불확실성을 가지며, "
                "고쳐야 할 장비가 LER 이 아니라 **CD** 다.")

    a2.axis("off")
    a2.set_title("왜 두 관측량이 함께 필요한가", loc="left")
    a2.text(0.0, 0.82, "결정론 CD 는 합성 평활 hypot(msd, blur)\n"
                       "만의 함수다 (배분을 바꿔도 차 0.00e+00)",
            fontsize=9, va="top")
    a2.text(0.0, 0.52, "→ CD 가 합성 크기를 주고\n"
                       "    LER 이 배분을 준다", fontsize=10, va="top",
            color="#c0392b")
    a2.text(0.0, 0.24, "관측량 2 개로는 미지수 2 개까지만 정한다\n"
                       "(rank J ≤ 관측량 수). 초점·노광량·플레어가\n"
                       "알려져 있다는 전제 위에 선 숫자다.",
            fontsize=8, va="top", color="#555555")
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13_02_계측규격",
        "LER 계측 정밀도를 올리면 블러의 불확실성은 줄지만 MSD 는 바닥에 "
        "걸려 내려가지 않는다. MSD 쪽 한계는 LER 이 아니라 CD 정밀도이기 "
        "때문이다. 가로 점선은 각 미지수의 자기 범위 폭이고, 곡선이 그 위에 "
        "있으면 '자기 범위보다 넓은 불확실성' 이라 결정할 수 없다는 뜻이다.")


def fig_where(outdir):
    rows = [["1.5", "7.0", "0.310", "1.09e-03", "0.99", "17.05", "569"],
            ["3.0", "10.0", "0.237", "8.25e-04", "1.07", "15.32", "455"],
            ["4.5", "13.0", "0.187", "5.62e-04", "1.57", "13.54", "346"],
            ["1.5", "13.0", "0.174", "4.12e-04", "1.93", "17.46", "572"],
            ["4.5", "7.0", "0.313", "1.86e-03", "1.04", "10.34", "210"]]
    fig = figlib.table_fig(
        rows, ["MSD", "블러", "LER", "|D|", "sd(블러)", "sd(MSD)",
               "CD 지점 수"],
        "어디서 재느냐가 계측 비용을 2.7 배 바꾼다",
        col_widths=[0.1, 0.1, 0.11, 0.14, 0.13, 0.13, 0.15],
        highlight_rows=(4,))
    return figlib.save(
        fig, outdir, "w13_03_어디서재나",
        "기울기는 국소값이라 기준점에 따라 조건수가 변한다. MSD 가 크고 "
        "블러가 작은 쪽(강조 행)에서 두 기울기가 덜 공선이라 같은 목표를 "
        "CD 지점 210 개로 달성한다. 불리한 조건이면 572 개가 필요하다. "
        "권고가 'LER 도 재라' 에서 '**어느 조건에서** 재라' 로 구체화된다.")


def fig_coverage(outdir):
    """검토로 잡은 결함 하나를 그림으로. 숫자는 **여기서 직접 재서** 쓴다.

    보고서에 '검토를 했다' 고 적는 것과, 검토가 무엇을 잡았는지 그림으로
    보이는 것은 다르다. 이 그림은 외삽 판정의 참조 분포가 중복 때문에
    0 으로 포화했던 것과, 중복을 지운 뒤 살아난 것을 같은 축에 겹친다.
    """
    import train_surrogate as TS
    from scipy.spatial import cKDTree

    # 데이터셋의 구조를 그대로 흉내 낸다 : 광학 조건 40 개를 각각
    # 초점 x 노광 격자(9 x 5 = 45 점)로 훑는다. 실제 데이터셋도 이 꼴이다.
    rng = np.random.default_rng(13)
    conds = np.column_stack([
        rng.uniform(1.20, 1.35, 40), rng.uniform(0.60, 0.85, 40)])
    rows = []
    for na, sg in conds:
        for dz in np.linspace(-80, 80, 9):
            for ds in np.linspace(0.9, 1.1, 5):
                rows.append([na, sg, dz, ds])
    X = np.asarray(rows, dtype=np.float64)
    feat = ["na", "sigma_out", "defocus", "dose_rel"]
    cat = np.zeros(4, dtype=bool)

    # (고친 뒤) 번들이 실제로 기록하는 것
    cov = TS.training_domain(
        X, feat, cat, np.arange(X.shape[0]), [], [])["coverage"]
    nn_fix = np.asarray(cov["nn_sorted"], dtype=np.float64)

    # (고치기 전) 중복을 남긴 경우 — 같은 데이터로 재현한다
    Z = X[:, :2]
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    d, _ = cKDTree(Zs).query(Zs, k=2)
    nn_bad = np.sort(d[:, 1])

    f0 = 100.0 * float(np.mean(nn_bad <= 0.0))
    f1 = 100.0 * float(np.mean(nn_fix <= 0.0))

    fig, ax = figlib.plt.subplots(figsize=(6.6, 3.5))
    for nn, lab, col, ls in ((nn_bad, f"중복을 남긴 경우 ({X.shape[0]} 행)",
                              "#c0392b", "--"),
                             (nn_fix, f"중복 제거 ({cov['n_distinct']} 조건)",
                              "#1f6f8b", "-")):
        xs = np.concatenate([[0.0], np.sort(nn)])
        ys = np.arange(xs.size) / max(xs.size - 1, 1) * 100.0
        ax.step(xs, ys, where="post", color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(90.0, color="#7f8c8d", lw=0.9, ls=":")
    # 왼쪽 위는 파란 곡선이 아직 바닥이라 비어 있다. 거기 둔다.
    ax.annotate("추천기가 이 선 위를 '외삽에 가깝다' 로 찍는다",
                xy=(0.055, 0.915), xycoords="axes fraction",
                fontsize=8, color="#7f8c8d", ha="left", va="bottom")
    ax.set_xlim(-0.05, max(float(nn_fix.max()) * 1.05, 0.5))
    ax.set_ylim(0, 103)
    ax.set_xlabel("최근접 거리 (표준화 단위)")
    figlib.ylab(ax, "이 거리보다 가까운 학습 조건의 비율 [%]")
    ax.set_title("외삽 판정의 참조 분포 : 중복 제거 전후")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13_03b_외삽참조분포",
        "외삽 백분위는 질의점의 최근접 거리를 '학습 조건끼리의 최근접 거리 "
        "분포' 에 대조해 얻는다. 거리에서 초점·노광을 뺀 것은 옳았으나, "
        "그 두 축으로만 다른 중복 행을 표본에 남겨 두어 분포가 0 에 "
        f"쌓였다(붉은 점선, 0 인 비율 {f0:.1f} %). 그러면 거리가 0 보다 "
        "조금만 커도 백분위가 90 을 넘어 후보가 어디 있든 '외삽' 으로 "
        "찍힌다. 중복을 지우면 분포가 퍼지고(파란 실선, 0 인 비율 "
        f"{f1:.1f} %) 판정이 비로소 조건의 위치를 반영한다. 13주차 자기 "
        "검토로 찾은 결함이며, 회귀 Case 18 이 음성 대조까지 포함해 고정한다.")


# --------------------------------------------------------------------------
# (나) 되돌림 결과가 있어야 만드는 그림
# --------------------------------------------------------------------------
def _primary_rows(res):
    """주 모형 · 교집합 영역 · 제약 통과군의 후보들."""
    want = res["models"][-1]
    out = []
    for t in res["tools"]:
        if t["model"] != want or t["scope"] != "교집합" or t["outside"]:
            continue
        for r in t["rows"]:
            if (r["sim_ok"] and r["sur_nils_ok"] and r["sur_cdu_ok"]):
                out.append(dict(r, tool=t["tool"]))
    return want, out


def fig_roundtrip(outdir, res):
    want, rows = _primary_rows(res)
    if len(rows) < 3:
        print("  [건너뜀] 되돌림 산점도 : 통과군 후보가 3 개 미만")
        return None
    a = np.array([r["sur_dof"] for r in rows], float)
    b = np.array([r["sim_dof"] for r in rows], float)
    top = np.array([r["rank"] < res["n_top"] for r in rows])
    lim = [0, max(a.max(), b.max()) * 1.1]

    fig, ax = figlib.plt.subplots(figsize=(5.4, 4.6))
    ax.plot(lim, lim, "k--", lw=1.0, label="완전 일치")
    ax.scatter(a[~top], b[~top], s=40, color="#4a90d9", label="구간 표본")
    ax.scatter(a[top], b[top], s=52, color="#c0392b", marker="^",
               label=f"상위 {res['n_top']} 후보")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("대리 모형이 예측한 DOF [nm]")
    figlib.ylab(ax, "시뮬레이터로 되돌려 잰 DOF [nm]")
    ax.set_title(f"되돌림 검증 ({os.path.basename(want)})")
    ax.legend()
    fig.tight_layout()
    med_t = float(np.median(b[top] - a[top])) if top.any() else float("nan")
    med_s = float(np.median(b[~top] - a[~top])) if (~top).any() else float("nan")
    return figlib.save(
        fig, outdir, "w13_04_되돌림",
        f"점선 위면 예측이 맞은 것이고 아래면 모형이 낙관적이다. "
        f"**상위 후보와 구간 표본을 갈라 그린다** — 상위 후보는 대리 모형이 "
        f"DOF 를 가장 크게 본 점이라 모형 오차가 무작위여도 과대평가된다"
        f"(승자의 저주). 중위 차는 상위권 {med_t:+.1f} nm, 구간 표본 "
        f"{med_s:+.1f} nm 이고, 모형 자체의 치우침은 구간 표본 쪽으로 "
        f"읽어야 한다.")


def fig_cdu(outdir, res):
    want, rows = _primary_rows(res)
    fr = [r["cdu_frac"] for r in rows if r.get("cdu_frac") is not None]
    if not fr:
        print("  [건너뜀] CDU : --sim-cdu 0 으로 돌려 측정값이 없음")
        return None
    fr = 100 * np.array(fr, float)
    fig, ax = figlib.plt.subplots(figsize=(6.2, 3.3))
    ax.hist(fr, bins=min(12, max(4, len(fr) // 2)), color="#8e7cc3",
            edgecolor="white")
    ax.axvline(100, color="crimson", lw=1.8, label="공차 (100 %)")
    ax.axvline(2, color="#555555", ls=":", lw=1.4,
               label="11주차 LHC 조건에서의 값 (약 2 %)")
    ax.set_xlabel("실측 CDU / 공차 [%]")
    figlib.ylab(ax, "후보 수")
    ax.set_title("추천 조건에서 실측한 CDU")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13_06_CDU",
        f"11주차는 CDU 가 공차의 약 2 % 라고 적었는데 그 숫자는 **데이터셋의 "
        f"LHC 조건**에서 나온 것이다. 추천기는 큰 DOF 를 고르므로 낮은 NA 로 "
        f"가고 그쪽은 NILS 가 낮아 CDU 가 커진다. 추천 조건에서 실측하니 "
        f"중위 {np.median(fr):.1f} %, 최대 {fr.max():.1f} % 다. 어떤 조건에서 "
        f"잰 숫자인지가 숫자만큼 중요하다.")


def fig_speed(outdir, res):
    want, rows = _primary_rows(res)
    ts = [r["t_sur_win"] for r in rows if r.get("t_sur_win")]
    tw = [r["t_sim_win"] for r in rows if r.get("t_sim_win")]
    tc = [r["t_cdu"] for r in rows if r.get("t_cdu")]
    if not ts or not tw:
        print("  [건너뜀] 속도 : 시간 기록이 없음")
        return None
    a, b = float(np.median(ts)), float(np.median(tw))
    names = ["대리 모형", "시뮬레이터\n(공정창)"]
    vals = [a, b]
    if tc:
        names.append("시뮬레이터\n(+ CDU 몬테카를로)")
        vals.append(b + float(np.median(tc)))
    fig, ax = figlib.plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(names, vals, color=["#2d7d46", "#d9822b", "#c0392b"][:len(vals)])
    ax.set_yscale("log")
    ax.bar_label(bars, labels=[f"{v * 1e3:.1f} ms" if v < 1 else f"{v:.2f} s"
                               for v in vals], padding=3, fontsize=8.5)
    figlib.ylab(ax, "창 하나당 계산 시간 [s, 로그]")
    ax.set_title(f"대리 모형이 {b / a:,.0f} 배 빠르다")
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13_07_속도",
        f"창 하나당 대리 모형 {a * 1e3:.1f} ms, 시뮬레이터 {b:.2f} s 로 "
        f"**{b / a:,.0f} 배** 차이다. 이 배수가 대리 모형을 쓰는 근거다 — "
        f"느리면서 부정확하면 쓸 이유가 없다. CDU 는 대리 모형이 라벨로 "
        f"예측하므로 몬테카를로 비용이 아예 없다.")


def fig_verdict(outdir, res):
    import roundtrip_w13 as RT
    want, rows = _primary_rows(res)
    if len(rows) < 3:
        print("  [건너뜀] 판정표 : 통과군 후보가 3 개 미만")
        return None
    a = [r["sur_dof"] for r in rows]
    b = [r["sim_dof"] for r in rows]
    rc = RT.rank_corr(a, b)
    o_s = sorted(range(len(rows)), key=lambda i: -a[i])
    o_m = sorted(range(len(rows)), key=lambda i: -b[i])
    t3 = RT.top_k_overlap(o_s, o_m, 3) if len(rows) >= 3 else None
    jr = RT.judge(rc.get("kendall"), t3, None, None)
    tbl = [[n, v.replace("**", ""), w] for n, v, w in jr]
    fig = figlib.table_fig(
        tbl, ["기준", "판정", "근거"],
        "판정기준_동결.md 의 기준을 적용한 결과",
        col_widths=[0.18, 0.13, 0.55],
        highlight_rows=tuple(i for i, (_, v, _) in enumerate(jr)
                             if "미달" in v))
    return figlib.save(
        fig, outdir, "w13_05_판정",
        "기준은 **실제 모형으로 돌리기 전에** 정했다(판정기준_동결.md). "
        "숫자를 보고 기준을 고치면 사후선택이 되므로, 그 경우 문서의 변경 "
        "기록에 적고 판정에 '사후선택이 섞였다' 를 단다. C(구간 타당성)는 "
        "적합예측 보정 파일이 있어야 판정된다.")


def main() -> int:
    ap = argparse.ArgumentParser(description="13주차 보고서용 그림 생성")
    ap.add_argument("--outdir", default="figures_w13")
    ap.add_argument("--result", default=None,
                    help="roundtrip_w13.py --save-json 이 남긴 파일")
    args = ap.parse_args()
    out = figlib.setup(args.outdir)
    print("=" * 74)
    print(" 13주차 보고서용 그림")
    print("=" * 74)
    print("\n  (가) 시뮬레이터만으로 만드는 그림\n")
    for fn in (fig_degeneracy, fig_spec, fig_where, fig_coverage):
        fn(out)
        print()
    if not args.result:
        print("  (나) 되돌림 결과가 필요한 그림은 건너뜁니다.")
        print("       먼저 아래로 되돌림을 돌리십시오 (한 줄로 붙여 쓸 것) :")
        print("\n    py roundtrip_w13.py --models models/s,models/s13,"
              "models/s13c --pitch 250 --duty 0.5 --pol TE --source "
              "conventional --mask binary --tool-preset regions "
              "--sim-cdu 8 --save-json w13_result.json")
        print("\n       그 뒤 : py figures_w13.py --result w13_result.json")
        return 0
    with open(args.result, encoding="utf-8") as fh:
        res = json.load(fh)
    print(f"\n  (나) {args.result} 를 읽어 만드는 그림\n")
    for fn in (fig_roundtrip, fig_verdict, fig_cdu, fig_speed):
        fn(out, res)
        print()
    print("=" * 74)
    print(f" 완료. {out}/ 에 PNG 가 있습니다.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
