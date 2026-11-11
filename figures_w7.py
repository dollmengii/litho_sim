"""
figures_w7.py -- 7주차 보고서 활동사진용 그래프 생성

실행: py figures_w7.py
결과: figures_w7/ 폴더에 PNG 5장 저장
"""

import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from litho import dataset as DS
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import stochastic as ST

LAM, NA, N_MED, SIGMA = 193.0, 1.35, 1.44, 0.7
PITCH, THRESH, BLUR, DOSE = 160.0, 0.30, 10.0, 30.0
OUT = "figures_w7"


def set_korean_font():
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic",
                 "NanumBarunGothic", "Noto Sans CJK KR"):
        if name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def base_profile():
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA)
    e = PR.dose_to_size(g, i, PITCH / 2, THRESH)
    return g, m, i * e


def ler_stats(g, prof, n_rows, dy, n_trials, seeds=5, dose=DOSE, blur=BLUR):
    vals = []
    for sd in range(seeds):
        r = ST.monte_carlo(g.x, prof, g.dx, dose, LAM, THRESH,
                           n_rows=n_rows, blur_nm=blur, n_trials=n_trials,
                           seed=sd, dy=dy)
        vals.append(r["ler_3sigma"])
    v = np.array(vals)
    return v.mean(), v.std()


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ── 1. LER 수렴성 ────────────────────────────────────────────────
def fig1(g, prof):
    trials = [2, 4, 8, 16, 32]
    tm, ts = zip(*[ler_stats(g, prof, 256, 0.5, nt) for nt in trials])

    lengths = [(256, 0.5), (512, 0.5), (512, 2.0), (1024, 2.0), (2048, 2.0)]
    ym = [n * d for n, d in lengths]
    lm, ls = zip(*[ler_stats(g, prof, n, d, 8) for n, d in lengths])

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].errorbar(trials, tm, yerr=ts, fmt="o-", capsize=4, lw=1.6)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(trials)
    ax[0].set_xticklabels([str(t) for t in trials])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("Monte Carlo 반복 횟수")
    ax[0].set_ylabel("LER 3σ [nm]")
    ax[0].set_title("(a) 반복 횟수 변화 (y 길이 128 nm 고정)", fontsize=10)
    ax[0].set_ylim(0, max(lm) * 1.25)
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(ym, lm, yerr=ls, fmt="s-", capsize=4, lw=1.6, color="crimson")
    ax[1].axhline(lm[-1], color="gray", ls="--", lw=1.0, label=f"수렴값 {lm[-1]:.3f} nm")
    ax[1].annotate(f"6주차 설정\n{100 * (lm[-1] - lm[0]) / lm[-1]:.0f} % 과소평가",
                   xy=(ym[0], lm[0]), xytext=(ym[0] * 3, lm[0] * 0.78),
                   arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xticks(ym)
    ax[1].set_xticklabels([f"{v:.0f}" for v in ym])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].tick_params(axis="x", labelsize=8)
    ax[1].set_xlabel("y 방향 측정 길이 [nm]")
    ax[1].set_ylabel("LER 3σ [nm]")
    ax[1].set_title("(b) 측정 길이 변화 (반복 8 고정)", fontsize=10)
    ax[1].set_ylim(0, max(lm) * 1.25)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    save(fig, "01_LER_수렴성.png")


# ── 2. dy 분리 효과 ──────────────────────────────────────────────
def fig2(g, prof):
    cases = [("기존\n(dy = dx)", 1024, 0.5), ("보정\n(dy = blur/3)", 1024, 2.0)]
    labels, ylen, mu, sd, tt = [], [], [], [], []
    for lbl, nr, dy in cases:
        t0 = time.perf_counter()
        m, s = ler_stats(g, prof, nr, dy, 8, seeds=3)
        tt.append((time.perf_counter() - t0) / 3)
        labels.append(lbl); ylen.append(nr * dy); mu.append(m); sd.append(s)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for a, vals, ttl, unit in (
        (ax[0], ylen, "y 측정 길이", "nm"),
        (ax[1], tt, "연산 소요 시간", "s"),
        (ax[2], [100 * s / m for m, s in zip(mu, sd)], "LER 추정 산포", "%"),
    ):
        bars = a.bar(labels, vals, color=["#9aa0a6", "#c0392b"], width=0.55)
        a.set_title(ttl, fontsize=10)
        a.set_ylabel(unit)
        a.bar_label(bars, fmt="%.2f", fontsize=8)
        a.grid(alpha=0.3, axis="y")
    save(fig, "02_dy분리_효과.png")


# ── 3. 블러 민감도 ───────────────────────────────────────────────
def fig3(g, prof):
    blurs = [2.0, 5.0, 10.0, 15.0, 20.0]
    lers, cds = [], []
    for b in blurs:
        r = ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH,
                           n_rows=1024, blur_nm=b, n_trials=8, seed=0, dy=max(0.5, b / 3))
        lers.append(r["ler_3sigma"]); cds.append(r["cd_mean"])

    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.plot(blurs, lers, "o-", lw=1.8, color="#1f77b4", label="LER 3σ")
    for b, l in zip(blurs, lers):
        ax1.annotate(f"{l:.2f}", xy=(b, l), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7,
                     color="#1f77b4")
    ax1.set_xlabel("레지스트 블러 [nm]")
    ax1.set_ylabel("LER 3σ [nm]", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(blurs, cds, "s--", lw=1.6, color="#c0392b", label="CD 평균")
    ax2.axhline(PITCH / 2, color="gray", ls=":", lw=1.0)
    ax2.set_ylabel("CD 평균 [nm]", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    ax1.set_title(f"블러 민감도 : LER {lers[0] / lers[-1]:.1f} 배 감소, "
                  f"CD {cds[-1] - cds[0]:+.2f} nm 편향", fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save(fig, "03_블러_민감도.png")


# ── 4. 다운샘플 식별 정보 보존 ───────────────────────────────────
def fig4():
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    iso = []
    for dz in np.arange(0.0, 261.0, 20.0):
        i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA,
                            defocus=float(dz))
        e = PR.dose_to_size(g, i, PITCH / 2, THRESH)
        if np.isfinite(e):
            iso.append(i * e)
    full = np.array([np.sqrt(np.mean((p - iso[0]) ** 2)) for p in iso])

    ns = [8, 16, 32, 64, 128, 256]
    keeps, sizes = [], []
    for n in ns:
        ds = [DS.downsample_profile(g.x, p, PITCH, n) for p in iso]
        d = np.array([np.sqrt(np.mean((q - ds[0]) ** 2)) for q in ds])
        keeps.append(100 * float(np.mean(d[1:] / full[1:])))
        sizes.append(DS.estimate_size(10000, n)["total_mb"])

    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.plot(ns, keeps, "o-", lw=1.8, color="#2e7d32")
    ax1.axhline(100, color="gray", ls=":", lw=1.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("n_profile [점]")
    ax1.set_ylabel("식별 정보 보존율 [%]", color="#2e7d32")
    ax1.set_ylim(90, 102)
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(ns, sizes, width=[n * 0.45 for n in ns], alpha=0.22, color="#777",
            zorder=0)
    ax2.set_ylabel("1만 건 저장 용량 [MB]")

    nyq = LAM / (4 * (1 + SIGMA) * NA)
    ax1.axvline(PITCH / nyq, color="crimson", ls="--", lw=1.2)
    ax1.annotate(f"Nyquist 하한\n({PITCH / nyq:.1f} 점)",
                 xy=(PITCH / nyq, 94), fontsize=8, color="crimson")
    ax1.set_title("다운샘플 해상도에 따른 식별 정보 보존", fontsize=10)
    save(fig, "04_다운샘플_보존율.png")


# ── 5. 보정 전후 Dose-LER 곡선 ──────────────────────────────────
def fig5(g, prof):
    doses = np.linspace(6.0, 60.0, 6)
    old_m, old_s = zip(*[ler_stats(g, prof, 256, 0.5, 8, seeds=3, dose=float(d))
                         for d in doses])
    new_m, new_s = zip(*[ler_stats(g, prof, 1024, 2.0, 8, seeds=3, dose=float(d))
                         for d in doses])

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.errorbar(doses, old_m, yerr=old_s, fmt="o--", lw=1.5, capsize=3,
                color="#9aa0a6", label="6주차 기준 (y 길이 128 nm)")
    ax.fill_between(doses, old_m, new_m, color="#c0392b", alpha=0.10)
    ax.errorbar(doses, new_m, yerr=new_s, fmt="s-", lw=1.8, capsize=4,
                color="#c0392b", label="7주차 보정 (y 길이 2048 nm)")
    ax.set_xlabel("Dose [mJ/cm²]")
    ax.set_ylabel("LER 3σ [nm]")
    gap = 100 * np.mean((np.array(new_m) - np.array(old_m)) / np.array(new_m))
    ax.set_title(f"보정 전후 Dose–LER 곡선 (ArF 193 nm) : 평균 {gap:.0f} % 과소평가",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "05_보정전후_DoseLER.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    set_korean_font()
    print("7주차 그래프 생성 중...")
    g, m, prof = base_profile()
    fig1(g, prof)
    fig2(g, prof)
    fig3(g, prof)
    fig4()
    fig5(g, prof)
    print(f"\n완료. '{OUT}' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
