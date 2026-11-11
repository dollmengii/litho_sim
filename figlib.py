"""보고서용 그림 공통 설정.

`figures_w12.py` 와 `figures_w13.py` 가 함께 쓴다. 터미널 캡쳐 대신 바로
문서에 붙일 PNG 를 만드는 것이 목적이므로, 글자 크기와 여백을 **문서에
축소해 넣었을 때 읽히는** 쪽으로 맞춘다.

한글 폰트는 `app.py` 와 같은 후보 목록을 쓴다. 윈도우에서는 맑은 고딕이
잡히고, 없으면 경고를 찍고 기본 폰트로 둔다(글자가 네모로 나오면 그
경고를 보고 폰트를 깔면 된다).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                      # 화면 없이 파일로만 저장
import matplotlib.font_manager as _fm      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402

#: 저장 폴더. 보고서에 붙일 PNG 가 여기 쌓인다.
OUTDIR = "figures"

#: 문서에 넣을 때를 기준으로 한 기본 크기. 300 dpi 로 저장한다.
DPI = 300


def setup(outdir: str = OUTDIR) -> str:
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]
    avail = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    else:
        print("  [경고] 한글 폰트를 찾지 못했습니다. 글자가 네모로 나오면")
        print("         맑은 고딕(윈도우) 또는 나눔고딕을 설치하십시오.")
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    os.makedirs(outdir, exist_ok=True)
    return outdir


def ylab(ax, text: str) -> None:
    """y 축 이름을 **가로로** 축 위에 둔다.

    한글은 세로로 돌리면 읽기 나쁘다. 라틴 문자는 90 도 회전해도 읽히지만
    한글은 글자마다 방향이 틀어져 보여 가독성이 크게 떨어진다. 그래서 축
    위 왼쪽에 가로로 적는다. 보고서에 축소해 넣을 때 특히 차이가 난다.
    """
    ax.set_ylabel("")
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color="#333333")
    # 제목이 같은 자리에 오면 겹친다. 여기서 밀어 두어도 `set_title()` 이
    # 뒤에 불리면 되돌아가므로 표시만 해 두고 **저장 시점에** 보정한다
    # (호출 순서에 상관없이 듣는다).
    ax._has_ylab = True


#: 가로 y 라벨을 쓴 축의 제목을 위로 미는 양 [pt]. 라벨이 축 위 8 pt 에
#: 놓이고 글자 높이가 8.5 pt 이므로 17 pt 까지 차지한다. 그 위에 여유를 둔다.
TITLE_PAD = 24


def _lift_titles(fig) -> None:
    """가로 y 라벨을 쓴 축의 제목을 라벨 위로 올린다.

    왜 `title.set_y()` 가 아닌가 — **이것으로는 안 된다.**
    ------------------------------------------------------
    처음에 `ax.title.set_y(1.16)` 으로 짰고 호출 직후 위치를 찍어 1.16 이
    들어간 것을 확인했다. 그런데 **`savefig` 가 되돌린다.** matplotlib 은
    그릴 때마다 `_update_title_position` 으로 제목 y 를 축 위쪽 경계에서
    다시 계산하므로 손으로 넣은 값이 버려진다. 실측 : set_y 직후 (0.5, 1.16)
    -> `savefig` 뒤 (0.5, 1.0). 즉 **겹침 방지가 한 번도 듣지 않았다.**
    저장 직후 위치를 찍어 보지 않고 넘어간 것이 원인이다.

    공식 통로는 `set_title(pad=...)` 이다. 그 값은 `_update_title_position`
    이 계산에 **직접 쓰므로** 다시 그려도 살아남는다. loc 세 자리를 모두
    훑는 이유는 `get_title()` 이 가운데 제목만 돌려주기 때문이다.
    """
    for ax in fig.get_axes():
        if not getattr(ax, "_has_ylab", False):
            continue
        for loc in ("center", "left", "right"):
            txt = ax.get_title(loc=loc)
            if txt:
                ax.set_title(txt, loc=loc, pad=TITLE_PAD)


def save(fig, outdir: str, stem: str, caption: str) -> str:
    """PNG 로 저장하고 경로를 돌려준다. 캡션은 화면에 찍어 준다.

    캡션을 그림에 **구워 넣지 않는다.** 보고서 본문에 그림 설명을 따로 쓰는
    것이 보통이고, 그림 안에 또 적으면 중복이라 읽기 나쁘다.
    """
    # 캡션은 문서에 그대로 붙일 글이다. matplotlib 도 워드도 마크다운을
    # 해석하지 않으므로 ** 를 벗긴다. 강조는 문서에서 직접 거는 것이 맞다.
    caption = caption.replace("**", "")
    # 가로 y 라벨을 쓴 축은 제목을 위로 민다. `set_title()` 호출 순서에
    # 상관없이 듣도록 여기서 한다. `_lift_titles` 의 주석에 왜 set_y 가
    # 아닌지 적어 두었다(savefig 가 되돌린다).
    _lift_titles(fig)
    path = os.path.join(outdir, f"{stem}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  저장 : {path}")
    print(f"    캡션(복사해서 쓰십시오) : {caption}")
    return path


def table_fig(rows, col_labels, title, col_widths=None, figsize=(7.2, None),
              highlight_rows=()):
    """표를 그림으로. 터미널 캡쳐 대신 쓰려고 만든 것이다.

    `highlight_rows` 에 든 행 색인은 배경을 칠해 눈에 띄게 한다. 강조는
    굵게가 아니라 **배경색**으로 한다. 한글 폰트는 굵은 자형이 없는 경우가
    많아 굵게가 적용되지 않을 수 있다.
    """
    n = len(rows)
    h = figsize[1] if figsize[1] else max(0.9, 0.34 * (n + 1.6))
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")
    ax.set_title(title, pad=10, loc="left")
    tb = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                  cellLoc="center", colWidths=col_widths)
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1.0, 1.32)
    for j in range(len(col_labels)):
        c = tb[0, j]
        c.set_facecolor("#2f3640")
        c.set_text_props(color="white")
    for i in range(1, n + 1):
        for j in range(len(col_labels)):
            c = tb[i, j]
            c.set_edgecolor("#c8ccd4")
            if (i - 1) in highlight_rows:
                c.set_facecolor("#fff4d6")
            elif i % 2 == 0:
                c.set_facecolor("#f6f7f9")
    fig.tight_layout()
    return fig
