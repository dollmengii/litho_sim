"""화면용 SVG 차트. `app.py` 가 쓴다.

왜 matplotlib 이 아닌가
-----------------------
matplotlib 그림을 PNG 로 박아 넣으면 세 가지가 따라온다.

1. **모양이 matplotlib 의 것이다.** 글꼴, 눈금, 여백이 전부 그쪽 규약이라
   화면의 나머지와 따로 논다. 색만 맞춰도 이질감이 남는다.
2. **한글이 서버 글꼴에 걸린다.** 배포 서버에 한글 글꼴이 없으면 축 이름이
   네모가 된다. `packages.txt` 로 막아 두었지만 글꼴 하나에 기대는 구조다.
   SVG 의 글자는 **브라우저가** 그리므로 화면에 쓰는 웹 글꼴을 그대로 탄다.
3. **고정 해상도다.** 넓은 화면에서 흐려지고, 좁은 화면에서 글자가 뭉갠다.

SVG 로 그리면 셋 다 사라진다. 대신 **그리는 코드를 직접 들고 있어야 하므로**
여기 있는 것은 이 과제가 실제로 쓰는 네 가지뿐이다. 범용 차트 라이브러리를
만들려는 것이 아니다.

좌표 규약
---------
모든 함수는 viewBox 안에서 그린다. 바깥 크기는 CSS 가 정하므로(`width:100%`)
화면 폭이 바뀌어도 글자 크기가 함께 늘지 않도록 viewBox 를 고정한다.
"""

from __future__ import annotations

import html
import math

import numpy as np

import ui as UI

#: viewBox 기준 폭. 높이는 함수마다 받는다.
W = 1000

#: 안쪽 여백. 왼쪽은 y 눈금 글자, 아래는 x 눈금과 축 이름 자리다.
PAD = {"l": 74, "r": 18, "t": 16, "b": 52}

_FONT = ("'IBM Plex Mono','JetBrains Mono',ui-monospace,"
         "SFMono-Regular,Menlo,monospace")
_SANS = ("'IBM Plex Sans KR','Malgun Gothic','Apple SD Gothic Neo',"
         "system-ui,sans-serif")


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """읽기 좋은 눈금값. 1, 2, 2.5, 5 배수 중에서 고른다.

    균등 분할은 0.333 같은 값이 나와 읽기 나쁘다. 눈금은 **사람이 읽는
    것**이므로 간격을 반올림 가능한 수로 맞춘다.
    """
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo, hi]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out or [lo, hi]


def _fmt(v: float, step_hint: float = 1.0) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if step_hint >= 1 or a >= 100:
        return f"{v:.0f}"
    if step_hint >= 0.1:
        return f"{v:.1f}"
    if step_hint >= 0.01:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _frame(h: int, xlo, xhi, ylo, yhi, xlabel, ylabel):
    """격자·눈금·축 이름을 그리고, 자료 좌표를 화면 좌표로 바꾸는 함수를 준다."""
    l, r = PAD["l"], W - PAD["r"]
    t, b = PAD["t"], h - PAD["b"]

    def X(v):
        return l + (r - l) * (v - xlo) / (xhi - xlo) if xhi > xlo else l

    def Y(v):
        return b - (b - t) * (v - ylo) / (yhi - ylo) if yhi > ylo else b

    xt, yt = _nice_ticks(xlo, xhi), _nice_ticks(ylo, yhi)
    xs = (xt[1] - xt[0]) if len(xt) > 1 else 1.0
    ys = (yt[1] - yt[0]) if len(yt) > 1 else 1.0
    p = []
    # 격자. 축보다 먼저 그려 선 아래로 깔린다.
    for v in xt:
        p.append(f'<line x1="{X(v):.1f}" y1="{t}" x2="{X(v):.1f}" y2="{b}" '
                 f'stroke="{UI.LINE}" stroke-width="1" opacity=".55"/>')
    for v in yt:
        p.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{r}" y2="{Y(v):.1f}" '
                 f'stroke="{UI.LINE}" stroke-width="1" opacity=".55"/>')
    # 축
    p.append(f'<path d="M{l} {t} L{l} {b} L{r} {b}" fill="none" '
             f'stroke="{UI.LINE}" stroke-width="1.4"/>')
    # 눈금 글자
    for v in xt:
        p.append(f'<text x="{X(v):.1f}" y="{b + 20}" fill="{UI.MUTED}" '
                 f'font-family="{_FONT}" font-size="15" text-anchor="middle">'
                 f'{_fmt(v, xs)}</text>')
    for v in yt:
        p.append(f'<text x="{l - 10}" y="{Y(v) + 5:.1f}" fill="{UI.MUTED}" '
                 f'font-family="{_FONT}" font-size="15" text-anchor="end">'
                 f'{_fmt(v, ys)}</text>')
    # 축 이름. y 는 가로로 축 위에 둔다(한글을 90 도 돌리면 읽기 나쁘다).
    if xlabel:
        p.append(f'<text x="{(l + r) / 2:.0f}" y="{h - 12}" fill="{UI.MUTED}" '
                 f'font-family="{_SANS}" font-size="15" text-anchor="middle">'
                 f'{_esc(xlabel)}</text>')
    if ylabel:
        p.append(f'<text x="{l}" y="{t - 4}" fill="{UI.MUTED}" '
                 f'font-family="{_SANS}" font-size="15" text-anchor="start">'
                 f'{_esc(ylabel)}</text>')
    return p, X, Y, (l, r, t, b)


def _open(h: int) -> str:
    return (f'<svg viewBox="0 0 {W} {h}" width="100%" height="auto" '
            f'preserveAspectRatio="xMidYMid meet" role="img" '
            f'style="display:block;background:{UI.BG};'
            f'border:1px solid {UI.LINE};border-radius:7px">')


def line(series, *, h=330, xlabel="", ylabel="", hline=None, hlabel="",
         span=None, spanlabel="", ylo=None, yhi=None, note=""):
    """꺾은선. `series` 는 (x, y, 색, 이름, 채움여부) 의 목록.

    `hline` 은 가로 기준선(임계 광강도 등), `span` 은 x 축 위의 구간 강조
    (CD 폭 등)다. 둘 다 이 과제에서 실제로 쓰는 것만 넣었다.
    """
    xs = np.concatenate([np.asarray(s[0], float) for s in series])
    ys = np.concatenate([np.asarray(s[1], float) for s in series])
    ys = ys[np.isfinite(ys)]
    xlo, xhi = float(np.min(xs)), float(np.max(xs))
    lo = float(np.min(ys)) if ylo is None else ylo
    hi = float(np.max(ys)) if yhi is None else yhi
    if hline is not None:
        lo, hi = min(lo, hline), max(hi, hline)
    pad = (hi - lo) * 0.10 or 0.1
    lo, hi = (lo - pad * 0.3, hi + pad)
    if ylo is not None:
        lo = ylo
    p, X, Y, (l, r, t, b) = _frame(h, xlo, xhi, lo, hi, xlabel, ylabel)

    if span is not None:
        x0, x1 = span
        p.append(f'<rect x="{X(x0):.1f}" y="{t}" width="{X(x1) - X(x0):.1f}" '
                 f'height="{b - t}" fill="{UI.RESIST}" opacity=".10"/>')
        for xv in (x0, x1):
            p.append(f'<line x1="{X(xv):.1f}" y1="{t}" x2="{X(xv):.1f}" '
                     f'y2="{b}" stroke="{UI.RESIST}" stroke-width="1.2" '
                     f'opacity=".8"/>')
        if spanlabel:
            p.append(f'<text x="{(X(x0) + X(x1)) / 2:.1f}" y="{t + 20}" '
                     f'fill="{UI.RESIST}" font-family="{_FONT}" '
                     f'font-size="16" text-anchor="middle">'
                     f'{_esc(spanlabel)}</text>')

    # 비교선(fill=None)을 **먼저** 그린다. 나중에 그리면 주인공 곡선을
    # 덮는다. 실제로 Δz=0 비교선이 Δz=120 실측선을 가리는 것을 보고 고쳤다.
    # 굵기와 불투명도도 낮춘다. 비교선은 배경이지 읽을 선이 아니다.
    for sx, sy, col, name, fill in sorted(series, key=lambda s: s[4] is not None):
        sx = np.asarray(sx, float)
        sy = np.asarray(sy, float)
        pts = [f"{X(a):.1f},{Y(c):.1f}" for a, c in zip(sx, sy)
               if math.isfinite(c)]
        if not pts:
            continue
        if fill:
            p.append(f'<polygon points="{X(sx[0]):.1f},{b:.1f} '
                     f'{" ".join(pts)} {X(sx[-1]):.1f},{b:.1f}" '
                     f'fill="{col}" opacity=".12"/>')
        ref = fill is None
        dash = ' stroke-dasharray="7 5"' if ref else ""
        p.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                 f'stroke="{col}" stroke-width="{1.7 if ref else 2.3}" '
                 f'opacity="{".5" if ref else "1"}" '
                 f'stroke-linejoin="round"{dash}/>')

    if hline is not None:
        p.append(f'<line x1="{l}" y1="{Y(hline):.1f}" x2="{r}" '
                 f'y2="{Y(hline):.1f}" stroke="{UI.RESIST}" stroke-width="1.6" '
                 f'stroke-dasharray="6 4"/>')
        if hlabel:
            # 글자 뒤에 바탕을 깐다. 곡선과 겹치면 둘 다 못 읽는다.
            wpx = 9.2 * len(hlabel) + 12
            p.append(f'<rect x="{l + 5}" y="{Y(hline) - 24:.1f}" '
                     f'width="{wpx:.0f}" height="21" rx="4" fill="{UI.BG}" '
                     f'opacity=".82"/>')
            p.append(f'<text x="{l + 11}" y="{Y(hline) - 9:.1f}" '
                     f'fill="{UI.RESIST}" font-family="{_FONT}" '
                     f'font-size="15">{_esc(hlabel)}</text>')

    names = [s for s in series if s[3]]
    if len(names) > 1:
        ly = t + 16
        # 범례 뒤에 바탕을 깐다. 곡선 위에 글자가 겹치면 둘 다 못 읽는다.
        p.append(f'<rect x="{r - 202}" y="{t + 1}" width="194" '
                 f'height="{21 * len(names) + 8}" rx="5" fill="{UI.BG}" '
                 f'opacity=".82" stroke="{UI.LINE}"/>')
        for sx, sy, col, name, fill in names:
            p.append(f'<line x1="{r - 190}" y1="{ly - 5}" x2="{r - 164}" '
                     f'y2="{ly - 5}" stroke="{col}" stroke-width="2.6"/>')
            p.append(f'<text x="{r - 156}" y="{ly}" fill="{UI.FG}" '
                     f'font-family="{_FONT}" font-size="15">{_esc(name)}</text>')
            ly += 21
    if note:
        p.append(f'<text x="{r}" y="{t + 14}" fill="{UI.MUTED}" '
                 f'font-family="{_FONT}" font-size="14" text-anchor="end">'
                 f'{_esc(note)}</text>')
    return _open(h) + "".join(p) + "</svg>"


def strip(values, width_nm, *, h=150, xlabel="x [nm]", on=UI.FG,
          off="#05070B", label=""):
    """위에서 내려다본 패턴 띠. 0/1 값을 사각형으로 그린다.

    `imshow` 대신 쓰는 이유 : 픽셀 그림은 확대하면 뭉개지고 가장자리가
    흐려 **선폭이 실제보다 무르게 보인다.** 사각형으로 그리면 경계가
    정확히 그 자리에 선다.
    """
    v = np.asarray(values, float)
    # numpy 2 에서 ndarray.ptp() 가 사라졌다. np.ptp 를 쓴다.
    rng = float(np.ptp(v))
    v = (v - v.min()) / (rng if rng else 1.0)
    n = v.size
    l, r = PAD["l"], W - PAD["r"]
    # 아래 여백을 줄이지 않는다. 줄이면 눈금 글자와 축 이름이 겹친다.
    t, b = PAD["t"], h - PAD["b"]
    p = [f'<rect x="{l}" y="{t}" width="{r - l}" height="{b - t}" '
         f'fill="{off}" stroke="{UI.LINE}"/>']
    # [13차 보완 100 번] 먼저 0/1 로 가른 뒤에 묶는다.
    # 예전에는 원값끼리 0.02 안에 드는 구간을 묶었다. 그런데 대역제한
    # 마스크의 |t|^2 는 경계에서 1.2 까지 솟았다가 평평해지므로, 밝은 칸
    # 하나가 값 차이 때문에 여러 사각형으로 쪼개졌다. 같은 색이어도 맞닿은
    # 사각형의 경계마다 안티에일리어싱 이음매가 생겨 칸 안에 세로줄이
    # 보였다(2 배 배율에서 뚜렷함). 이 함수는 docstring 대로 **0/1 패턴을
    # 그리는 것**이므로 임계로 가르고 나면 칸마다 사각형이 정확히 하나다.
    # 덤으로 링잉 과도분이 칸 너비에 끼어들지 않는다.
    onoff = v > 0.5
    i = 0
    while i < n:
        j = i
        while j + 1 < n and onoff[j + 1] == onoff[i]:
            j += 1
        if onoff[i]:
            x0 = l + (r - l) * i / n
            x1 = l + (r - l) * (j + 1) / n
            p.append(f'<rect x="{x0:.1f}" y="{t}" width="{max(x1 - x0, .6):.1f}" '
                     f'height="{b - t}" fill="{on}" opacity=".88"/>')
        i = j + 1
    # [13차 보완 97 번] 폭을 5 등분하면 1280 nm 에서 0, 256, 512, 768,
    # 1024, 1280 이 나온다. 바로 위에 놓이는 선 그래프는 _nice_ticks 를
    # 써서 0, 500, 1,000 이므로 같은 탭 안에서 두 패널의 눈금이 어긋났다.
    # 규칙을 한곳에만 둔다. 선 그래프와 같은 함수를 쓴다.
    _w = float(width_nm)
    xt = _nice_ticks(0.0, _w)
    xs = (xt[1] - xt[0]) if len(xt) > 1 else 1.0
    for xv in xt:
        xp = l + (r - l) * xv / (_w if _w else 1.0)
        p.append(f'<text x="{xp:.1f}" y="{b + 20}" fill="{UI.MUTED}" '
                 f'font-family="{_FONT}" font-size="15" text-anchor="middle">'
                 f'{_fmt(xv, xs)}</text>')
    p.append(f'<text x="{(l + r) / 2:.0f}" y="{h - 10}" fill="{UI.MUTED}" '
             f'font-family="{_SANS}" font-size="15" text-anchor="middle">'
             f'{_esc(xlabel)}</text>')
    if label:
        p.append(f'<text x="{l}" y="{t - 4}" fill="{UI.MUTED}" '
                 f'font-family="{_SANS}" font-size="15">{_esc(label)}</text>')
    return _open(h) + "".join(p) + "</svg>"


def bossung(dz, doses, cd, target, tol, *, h=360, highlight=None):
    """노광량별 CD 대 초점. 규격 띠와 함께 그린다.

    `cd` 는 (노광량, 초점) 모양이다. 선이 많으면 읽을 수 없으므로 다섯 개만
    고르고 가운데(목표 노광량)를 진하게 한다.
    """
    dz = np.asarray(dz, float)
    cd = np.asarray(cd, float)
    nd = cd.shape[0]
    finite = cd[np.isfinite(cd)]
    if finite.size == 0:
        return _open(h) + (f'<text x="{W/2}" y="{h/2}" fill="{UI.MUTED}" '
                           f'font-family="{_SANS}" font-size="17" '
                           f'text-anchor="middle">그릴 자료가 없습니다</text></svg>')
    ylo = min(float(np.min(finite)), target - tol * 2.2)
    yhi = max(float(np.max(finite)), target + tol * 2.2)
    p, X, Y, (l, r, t, b) = _frame(h, float(dz.min()), float(dz.max()),
                                   ylo, yhi, "Defocus [nm]", "CD [nm]")
    # 규격 띠
    p.insert(0, f'<rect x="{l}" y="{Y(target + tol):.1f}" width="{r - l}" '
                f'height="{abs(Y(target - tol) - Y(target + tol)):.1f}" '
                f'fill="{UI.SPEC}" opacity=".14"/>')
    for v in (target - tol, target + tol):
        p.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{r}" y2="{Y(v):.1f}" '
                 f'stroke="{UI.SPEC}" stroke-width="1.2" '
                 f'stroke-dasharray="4 4" opacity=".7"/>')
    idx = sorted({0, nd // 4, nd // 2, 3 * nd // 4, nd - 1})
    for di in idx:
        mid = di == nd // 2
        col = UI.BEAM if mid else UI.FG
        op = "1" if mid else ".30"
        pts, run = [], []
        for k, v in enumerate(cd[di]):
            if math.isfinite(v):
                run.append(f"{X(dz[k]):.1f},{Y(v):.1f}")
            else:
                if len(run) > 1:
                    pts.append(" ".join(run))
                run = []
        if len(run) > 1:
            pts.append(" ".join(run))
        for seg in pts:
            p.append(f'<polyline points="{seg}" fill="none" stroke="{col}" '
                     f'stroke-width="{2.6 if mid else 1.6}" opacity="{op}"/>')
    if highlight is not None and dz.min() <= highlight <= dz.max():
        p.append(f'<line x1="{X(highlight):.1f}" y1="{t}" '
                 f'x2="{X(highlight):.1f}" y2="{b}" stroke="{UI.RESIST}" '
                 f'stroke-width="1.4" stroke-dasharray="3 4"/>')
    p.append(f'<text x="{r}" y="{t + 14}" fill="{UI.MUTED}" '
             f'font-family="{_FONT}" font-size="14" text-anchor="end">'
             f'규격 ±{tol:.0f} nm · 목표 {target:.1f} nm</text>')
    return _open(h) + "".join(p) + "</svg>"


def window_map(pw, dz, doses, *, h=360):
    """공정 윈도 격자. 규격을 만족하는 칸만 칠한다."""
    pw = np.asarray(pw, bool)
    dz = np.asarray(dz, float)
    doses = np.asarray(doses, float)
    p, X, Y, (l, r, t, b) = _frame(h, float(dz.min()), float(dz.max()),
                                   float(doses.min()), float(doses.max()),
                                   "Defocus [nm]", "Dose (상대)")
    if pw.size:
        cw = (r - l) / max(pw.shape[1], 1)
        ch = (b - t) / max(pw.shape[0], 1)
        for i in range(pw.shape[0]):
            for j in range(pw.shape[1]):
                if not pw[i, j]:
                    continue
                p.append(f'<rect x="{l + j * cw:.1f}" '
                         f'y="{b - (i + 1) * ch:.1f}" '
                         f'width="{cw + .6:.1f}" height="{ch + .6:.1f}" '
                         f'fill="{UI.BEAM}" opacity=".40"/>')
    p.append(f'<text x="{r}" y="{t + 14}" fill="{UI.MUTED}" '
             f'font-family="{_FONT}" font-size="14" text-anchor="end">'
             f'통과 {int(pw.sum())} / {pw.size} 칸</text>')
    return _open(h) + "".join(p) + "</svg>"
