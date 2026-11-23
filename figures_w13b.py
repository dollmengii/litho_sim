"""13주차 보고서 **뒷부분** 그림을 만든다.

13주차 전반(되돌림 검증·계측 규격)은 `figures_w13.py` 가 만든다. 이 파일은
그 뒤에 한 일을 담는다.

  w13b_01_띠.png         경계 밖 띠를 넣어야 관문이 양쪽을 본다
  w13b_02_관문.png       2 단계 관문 : 형성 분류 뒤 품질 회귀
  w13b_03_극성2x2.png    마스크 극성 x 레지스트 톤 네 조합
  w13b_04_MEEF.png       k1 이 작아지면 마스크 오차가 증폭된다
  w13b_05_예산.png       CD 예산에서 마스크 몫을 떼면 남는 공차

**숫자를 손으로 옮겨 적지 않는다.** 01·02 는 재측정 결과(json)를 읽고,
03·04·05 는 시뮬레이터로 그 자리에서 계산한다. 손으로 옮기면 그 순간
문서-코드 드리프트가 생기고, 그것이 `claims.py` 가 막으려는 실패다.

실행
----
    py figures_w13b.py                          03·04·05 만 만든다
    py figures_w13b.py --summary dataset/summary.txt \\
                       --result surrogate_report.json   전부 만든다
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re

import numpy as np

import figlib
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import resist as RS

#: 그림을 그리는 기준 조건. 화면 기본값과 같게 둔다.
LAM = 193.0
NA = 1.35
NMED = 1.44
SIG = 0.70
THR = 0.30
PITCH = 160.0
DUTY = 0.5


def _aerial(g, m, defocus=0.0):
    return IM.aerial_image(g, m, LAM, NA, n_medium=NMED, sigma_out=SIG,
                           sigma_in=0.0, defocus=defocus, n_source_grid=21)


# ─────────────────────────── (가) 재측정 결과가 필요한 것 ───────────────────

def read_summary(path: str) -> dict:
    """`dataset/summary.txt` 에서 적격·띠 건수와 탈락 사유를 읽는다.

    숫자를 손으로 옮겨 적지 않기 위해 생성기가 쓴 파일을 그대로 읽는다.
    형식이 바뀌면 조용히 틀린 그림이 나오지 않도록, 못 찾으면 예외를 낸다.
    """
    txt = io.open(path, encoding="utf-8").read()
    m = re.search(r"적격\s+([\d,]+)\s*·\s*경계 밖 띠\s+([\d,]+)", txt)
    if not m:
        raise ValueError(f"{path} 에서 '적격 N · 경계 밖 띠 M' 줄을 못 찾았다")
    out = {"n_eligible": int(m.group(1).replace(",", "")),
           "n_band": int(m.group(2).replace(",", "")), "reject": {}}
    m2 = re.search(r"탈락 사유\s+(.+)", txt)
    if not m2:
        raise ValueError(f"{path} 에서 '탈락 사유' 줄을 못 찾았다")
    # 쉼표로 자르면 안 된다. 쉼표는 항목 구분자이면서 천 단위 구분자이기도
    # 해서 '21,004' 가 '21' 과 '004' 로 쪼개진다(실제로 그렇게 틀렸다).
    # 이름과 숫자의 짝을 직접 집는다.
    pairs = re.findall(r"([A-Za-z_]+)\s+([\d,]+)", m2.group(1))
    if not pairs:
        raise ValueError(f"{path} 의 '탈락 사유' 줄에서 짝을 못 읽었다")
    for k, v in pairs:
        out["reject"][k] = int(v.replace(",", ""))
    return out


def fig_band(outdir, sm):
    """경계 밖 띠. 적격만 모으면 관문이 아래쪽을 한 번도 못 본다."""
    n_ok, n_band = sm["n_eligible"], sm["n_band"]
    rej = sm["reject"]
    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.0, 3.4))

    a1.bar(["적격", "경계 밖 띠"], [n_ok, n_band],
           color=["#5B2BD9", "#B4600E"], width=0.55)
    for i, v in enumerate((n_ok, n_band)):
        a1.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    a1.set_title("재생성한 데이터 구성")
    figlib.ylab(a1, "건수")
    a1.set_ylim(0, max(n_ok, n_band) * 1.18)

    # 오른쪽은 '왜 띠가 필요한가' 를 생성기가 센 **실제 탈락 사유**로 보인다.
    # (예전 판은 여기에 난수 히스토그램을 그렸다. 보고서 그림에 지어낸
    #  분포를 넣으면 읽는 사람이 측정값으로 오해한다.)
    names = [k for k in ("formed", "nils_ok", "cdu_ok", "budget_ok")
             if k in rej]
    vals = [rej[k] for k in names]
    ko = {"formed": "미형성", "nils_ok": "NILS 미달",
          "cdu_ok": "CDU 초과", "budget_ok": "예산 초과"}
    a2.barh([ko.get(k, k) for k in names][::-1], vals[::-1],
            color="#B4600E", height=0.5)
    for i, v in enumerate(vals[::-1]):
        a2.text(v, i, f"  {v:,}", va="center", fontsize=9)
    a2.set_xlim(0, max(vals) * 1.22 if vals else 1)
    a2.set_title("무엇 때문에 떨어졌나 (첫 사유)")
    a2.set_xlabel("건수")

    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13b_01_띠",
        f"재생성한 데이터 {n_ok + n_band:,} 건의 구성. 적격 {n_ok:,} 건과 "
        f"경계 밖 띠 {n_band:,} 건을 함께 담았다. 오른쪽은 생성기가 센 "
        "탈락 첫 사유다. 띠가 없으면 관문이 문턱 아래를 한 번도 보지 "
        "못하므로, 무엇을 걸러야 하는지 배우지 못한다.")


def fig_gate(outdir, res):
    """2 단계 관문. 1 단계는 전 영역, 2 단계는 형성된 영역만."""
    clf = res.get("formed_classifier") or {}
    if not clf.get("available"):
        raise ValueError("보고서 json 에 형성 분류기 결과가 없다 "
                         f"({clf.get('reason', '사유 없음')})")
    auc = float(clf["auc"])
    acc = float(clf["accuracy"])
    rate = float(clf["formed_rate"])
    fig, (a1, a2) = figlib.plt.subplots(1, 2, figsize=(9.2, 3.4))

    a1.bar(["형성", "미형성"], [100 * rate, 100 * (1 - rate)],
           color=["#5B2BD9", "#B4600E"], width=0.55)
    for i, v in enumerate((100 * rate, 100 * (1 - rate))):
        a1.text(i, v, f"{v:.1f} %", ha="center", va="bottom", fontsize=9)
    a1.set_ylim(0, 112)
    a1.set_title(f"1 단계가 가르는 비율 (선별 {clf['n_points']:,} 점)")
    figlib.ylab(a1, "비율 [%]")

    # 98 % 와 99.6 % 를 0 부터 막대로 그리면 둘이 같아 보이고 칸만 버린다.
    # 시험 집합에서 실제로 몇 건을 틀렸는지가 읽는 사람에게 쓸모 있다.
    n_test = int(clf["n_test"])
    ff, fu = int(clf["false_formed"]), int(clf["false_unformed"])
    bars = [("맞춤", n_test - ff - fu, "#2F6F6B"),
            ("미형성을 형성이라 함", ff, "#B4600E"),
            ("형성을 미형성이라 함", fu, "#5B2BD9")]
    a2.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1],
            color=[b[2] for b in bars][::-1], height=0.5)
    for i, b in enumerate(bars[::-1]):
        a2.text(b[1], i, f"  {b[1]:,}", va="center", fontsize=9)
    a2.set_xscale("log")
    a2.set_xlim(1, n_test * 4)
    a2.set_title(f"시험 {n_test:,} 건의 판정 (AUC {auc:.4f})")
    a2.set_xlabel("건수 (로그 눈금)")

    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13b_02_관문",
        f"2 단계 관문. 1 단계 형성 분류기가 선별 {clf['n_points']:,} 점을 "
        f"형성 {100 * rate:.1f} % 와 미형성 {100 * (1 - rate):.1f} % 로 "
        f"가르고(AUC {auc:.4f}, 정확도 {100 * acc:.2f} %), 2 단계 품질 "
        "회귀는 형성된 영역에서만 돈다. 품질 회귀 하나로 전 영역을 덮으면 "
        "미형성 구간의 값이 회귀를 끌어당긴다.")


# ─────────────────────────── (나) 그 자리에서 계산하는 것 ───────────────────

def _combo(mask_clear: bool, resist_neg: bool, duty=DUTY, pitch=PITCH):
    """네 조합 하나를 계산한다. app.py 의 `mask_desc` 와 같은 규칙이다.

    **노광량을 dose-to-size 로 맞춘 뒤** 그린다. 맞추지 않으면 실측 CD 가
    목표에서 몇 nm 벗어난 그림이 되어, 모형이 틀린 것처럼 읽힌다. 공정에서
    선폭은 노광량으로 맞추는 양이므로 맞춘 상태가 기준이다.
    """
    w = duty * pitch if mask_clear == resist_neg else (1.0 - duty) * pitch
    pol = "clear" if resist_neg else "dark"
    tone = RS.TONE_BRIGHT if resist_neg else RS.TONE_DARK
    g = G.build_grid(pitch=pitch, n_periods=4, dx_target=0.5)
    m = MK.line_space(g, cd=w, polarity=pol)
    i = _aerial(g, m)
    dose = PR.dose_to_size(g, i, w, THR, tone, max_width=pitch)
    i = i * float(dose)
    prof = RS.resist_profile(i, THR, tone)
    cd = RS.extract_cd(g.x, i, THR, tone, max_width=pitch)["cd"]
    return g, i, prof, w, cd


def fig_polarity(outdir):
    """마스크 극성 x 레지스트 톤. 남는 선이 feature 인지 공간인지가 갈린다."""
    duty = 0.68
    combos = [(True, True, "clear-field + negative"),
              (False, False, "dark-field + positive"),
              (True, False, "clear-field + positive"),
              (False, True, "dark-field + negative")]
    # 세로축 이름을 축마다 붙이지 않는다. 2 x 2 에서는 위 칸의 가로 y 라벨이
    # 아래 칸 제목과 겹친다(실측으로 확인). 한 줄로 묶어 그림 위에 적는다.
    fig, axs = figlib.plt.subplots(2, 2, figsize=(9.6, 5.2), sharex=True,
                                   sharey=True)
    for ax, (mc, rn, name) in zip(axs.ravel(), combos):
        g, i, prof, w, cd = _combo(mc, rn, duty=duty)
        ax.fill_between(g.x, 0, prof * i.max(), step="mid",
                        color="#5B2BD9", alpha=0.22)
        ax.plot(g.x, i, color="#5B2BD9", lw=1.3)
        ax.axhline(THR, color="#B4600E", ls="--", lw=1.1)
        ax.set_title(f"{name}\n목표 {w:.1f} nm  ·  실측 {cd:.1f} nm",
                     fontsize=9.5)
        ax.set_xlim(0, g.field_width)
    for ax in axs[1]:
        ax.set_xlabel("x [nm]")
    fig.suptitle("세로축 : 광강도   ·   파선 : 임계 0.30   ·   "
                 "채운 영역 : 현상 뒤 남는 레지스트", fontsize=9, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(hspace=0.55)
    return figlib.save(
        fig, outdir, "w13b_03_극성2x2",
        f"마스크 극성과 레지스트 톤의 네 조합 (duty {duty}, pitch "
        f"{PITCH:.0f} nm). 마스크 feature 폭은 언제나 {duty * PITCH:.1f} nm "
        f"이지만, 현상 뒤 남는 것이 그 feature 인지 사이 공간인지가 갈리므로 "
        f"목표 CD 가 {duty * PITCH:.1f} nm 와 {(1 - duty) * PITCH:.1f} nm 로 "
        "나뉜다. 네 경우 모두 노광량을 dose-to-size 로 맞춘 상태이며, 실측 "
        "CD 가 목표와 일치한다.")


def _meef_at(pitch: float, delta: float = 2.0) -> tuple[float, float]:
    """(MEEF, 목표 CD). 노광량은 고정해 잰다."""
    cd0 = 0.5 * pitch
    g = G.build_grid(pitch=pitch, n_periods=6, dx_target=0.25)
    i0 = _aerial(g, MK.line_space(g, cd=cd0, polarity="clear"))
    dose = PR.dose_to_size(g, i0, cd0, THR, RS.TONE_BRIGHT, max_width=pitch)
    ims = [_aerial(g, MK.line_space(g, cd=cd0 + d, polarity="clear"))
           for d in (-delta, +delta)]
    return PR.meef(g, ims[0], ims[1], cd0 - delta, cd0 + delta, THR,
                   RS.TONE_BRIGHT, dose=float(dose), max_width=pitch), cd0


def fig_meef(outdir):
    """MEEF 대 피치. k1 이 작아지면 1 을 크게 넘는다."""
    pitches = [400.0, 320.0, 240.0, 200.0, 160.0, 140.0, 120.0, 110.0, 100.0]
    vals, k1s = [], []
    for p in pitches:
        m, cd0 = _meef_at(p)
        vals.append(m)
        k1s.append(cd0 * NA / LAM)
    fig, ax = figlib.plt.subplots(figsize=(6.6, 3.6))
    ax.plot(k1s, vals, "o-", color="#5B2BD9", lw=1.6, ms=5)
    ax.axhline(1.0, color="#5C5768", ls=":", lw=1.0)
    # 기준선 설명은 가운데에 둔다. 왼쪽 끝에 두면 400 nm 라벨과 겹친다.
    ax.text(0.5, 1.0, " MEEF = 1 (증폭 없음)", transform=ax.get_yaxis_transform(),
            ha="center", va="bottom", fontsize=8, color="#5C5768")
    # 라벨이 축 밖으로 나가지 않도록 위쪽 여백을 먼저 확보한다.
    ax.set_ylim(min(vals) - 0.25, max(vals) + 0.55)
    for k, v, p in zip(k1s, vals, pitches):
        if p in (400.0, 160.0, 120.0, 100.0):
            # 맨 오른쪽 점은 라벨을 왼쪽으로 내보낸다.
            dx, ha = ((-8, "right") if p == 100.0 else (7, "left"))
            ax.annotate(f"{p:.0f} nm", (k, v), textcoords="offset points",
                        xytext=(dx, 6), fontsize=8, ha=ha)
    ax.set_xlabel("k1 = 목표 CD x NA / λ")
    figlib.ylab(ax, "MEEF")
    ax.set_title("마스크 오차 증폭 계수")
    ax.invert_xaxis()
    fig.tight_layout()
    return figlib.save(
        fig, outdir, "w13b_04_MEEF",
        f"k1 이 작아질수록 MEEF 가 가파르게 오른다 (피치 {pitches[0]:.0f} nm "
        f"에서 {vals[0]:.2f}, {pitches[-1]:.0f} nm 에서 {vals[-1]:.2f}). "
        "마스크 CD 가 1 nm 틀리면 웨이퍼 CD 가 그 배만큼 틀어지므로, CD "
        "예산에서 마스크 몫이 그만큼 커진다. 노광량은 고정해 쟀다. k1 이 "
        "0.6 보다 큰 구간의 1 안팎 값은 상수 문턱 모형의 성질이므로 '대략 "
        "1' 로만 읽어야 하며(한계 116), 뜻이 있는 것은 k1 이 작아질 때의 "
        "가파른 상승이다.")


def fig_budget(outdir, cdu_nm: float = 2.0, tol: float = 0.0707):
    """CD 예산에서 마스크 몫을 떼면 남는 공차."""
    pitches = [400.0, 240.0, 160.0, 120.0, 100.0]
    rows = []
    for p in pitches:
        meef, cd0 = _meef_at(p)
        budget = tol * cd0
        mask = PR.mask_budget(meef, cdu_nm)
        avail = float(np.sqrt(max(budget ** 2 - mask ** 2, 0.0)))
        rows.append((p, cd0, meef, budget, mask, avail))

    fig, ax = figlib.plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.21, [r[3] for r in rows], width=0.2, label="CD 예산",
           color="#5C5768")
    ax.bar(x, [r[4] for r in rows], width=0.2, label="마스크 몫",
           color="#B4600E")
    ax.bar(x + 0.21, [r[5] for r in rows], width=0.2, label="남는 공차",
           color="#5B2BD9")
    # 0 인 막대는 그려지지 않아 계열이 빠진 것처럼 보인다. 숫자를 적어 둔다.
    for xi, r in zip(x, rows):
        if r[5] <= 0.0:
            ax.annotate("0.00", (xi + 0.21, 0.0), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8,
                        color="#5B2BD9")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]:.0f}\n({r[2]:.2f})" for r in rows])
    ax.set_xlabel("피치 [nm]  (괄호는 MEEF)")
    figlib.ylab(ax, "[nm]")
    ax.set_title(f"CD 예산 분해 (공차 ±{100 * tol:.2f} %, 마스크 CDU "
                 f"{cdu_nm:.1f} nm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    tight = rows[-1]
    return figlib.save(
        fig, outdir, "w13b_05_예산",
        f"CD 예산에서 마스크 몫을 뗀 결과. 피치 {tight[0]:.0f} nm 에서는 "
        f"MEEF {tight[2]:.2f} 라 마스크 몫 {tight[4]:.2f} nm 가 예산 "
        f"{tight[3]:.2f} nm 를 혼자 넘겨 남는 공차가 {tight[5]:.2f} nm 가 "
        "된다. 이 항을 빼고 판정하면 공정 윈도가 그만큼 낙관적이다.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures_w13b")
    ap.add_argument("--summary", default=None,
                    help="dataset/summary.txt. 주면 01(띠) 도 만든다.")
    ap.add_argument("--result", default=None,
                    help="train_surrogate 의 보고 json. 주면 02(관문) 도 만든다.")
    a = ap.parse_args()
    out = figlib.setup(a.outdir)
    print(f"[그림] {out} 에 저장합니다.\n")

    made = [fig_polarity(out), fig_meef(out), fig_budget(out)]

    if a.summary:
        if not os.path.exists(a.summary):
            print(f"  [건너뜀] {a.summary} 가 없습니다.")
        else:
            made.append(fig_band(out, read_summary(a.summary)))
    if a.result:
        if not os.path.exists(a.result):
            print(f"  [건너뜀] {a.result} 가 없습니다.")
        else:
            with io.open(a.result, encoding="utf-8") as f:
                made.append(fig_gate(out, json.load(f)))
    if not (a.summary and a.result):
        print("  [안내] --summary dataset/summary.txt --result "
              "surrogate_report.json 을 주면 띠·관문 그림도 만듭니다.")

    print(f"\n  모두 {len(made)} 장.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
