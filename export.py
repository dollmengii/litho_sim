"""결과 내보내기. `app.py` 의 '공정 윈도' 화면이 쓴다.

무엇을 내보내는가
-----------------
화면에서 방금 계산한 **조건과 결과**다. 두 형식으로 나눈 이유가 있다.

- **PDF 한 장** : 제출·인쇄·전달용. 받는 사람이 열기만 하면 되고 서식이
  환경에 따라 깨지지 않는다.
- **Excel** : 받은 사람이 **더 계산할 수 있게**. 조건표·결과표·격자 전체가
  시트로 들어간다.

Word 를 넣지 않은 이유 : 이 내용은 숫자표와 그림이다. Word 로 내면 받는
사람 환경에서 서식이 밀리고, 결국 PDF 로 바꿔서 본다. 역할이 겹친다.

한글 글꼴
---------
PDF 는 reportlab 의 **내장 CID 글꼴**(`HYGothic-Medium`)을 쓴다. 글꼴
파일을 들고 다니지 않아도 한글이 나오므로 배포 서버에 글꼴이 없어도
깨지지 않는다. `packages.txt` 의 나눔글꼴에 기대지 않는다.

이 파일을 직접 돌리면(`py export.py`) 표본 자료로 두 파일을 만든다.
**한 번도 실행되지 않은 코드를 남기지 않기 위해서다**(14 차 보완 105·108
번에서 같은 함정을 세 번 밟았다).
"""

from __future__ import annotations

import datetime as _dt
import io
import hashlib

#: 화면과 같은 배색. `ui.py` 가 근거를 적어 두었다.
INK = (0x15 / 255, 0x13 / 255, 0x1A / 255)
MUTED = (0x5C / 255, 0x57 / 255, 0x68 / 255)
ACCENT = (0x5B / 255, 0x2B / 255, 0xD9 / 255)
AMBER = (0xB4 / 255, 0x60 / 255, 0x0E / 255)
LINE = (0xE7 / 255, 0xE2 / 255, 0xD8 / 255)
PAPER = (0xFB / 255, 0xF9 / 255, 0xF4 / 255)

#: 한글용 내장 CID 글꼴과 숫자용 글꼴.
KO = "HYGothic-Medium"
NUM = "Helvetica"
NUM_B = "Helvetica-Bold"

TITLE = "포토리소그래피 공정 윈도 결과"
FOOT = ("이 문서는 시뮬레이터가 계산한 값입니다. 공정 인증이 아니며, "
        "대리 모형의 추천은 되돌려 재야 확인됩니다.")


def stamp(bundle: dict) -> str:
    """결과마다 붙는 짧은 코드. 같은 조건이면 같은 코드가 나온다.

    로그인을 만들지 않고도 "내가 뽑은 그 결과" 를 가리킬 수 있게 하려는
    것이다. 개인정보를 하나도 받지 않는다.
    """
    key = "|".join(f"{n}={v}" for n, v, _ in bundle.get("conditions", []))
    h = hashlib.sha256(key.encode("utf-8")).hexdigest().upper()
    return f"{h[:4]}-{h[4:6]}"


def build(conditions, results, *, verdict: str = "", grid=None,
          note: str = "") -> dict:
    """내보낼 한 덩어리를 만든다.

    `conditions`, `results` 는 (이름, 값, 단위) 의 목록이다. 값은 **이미
    화면에 보이는 문자열**로 받는다. 여기서 다시 반올림하면 화면과 문서의
    숫자가 어긋난다(11 차 보완 37 번과 같은 유형).

    `grid` 는 (dz 배열, dose 배열, 통과 여부 2 차원) 이거나 None 이다.
    """
    b = {
        "title": TITLE,
        "when": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "conditions": list(conditions),
        "results": list(results),
        "verdict": verdict,
        "grid": grid,
        "note": note,
    }
    b["code"] = stamp(b)
    return b


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def to_xlsx(b: dict) -> bytes:
    """조건·결과·격자를 시트 셋으로. 받은 사람이 더 계산할 수 있게 한다."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    thin = Side(style="thin", color="E7E2D8")
    head_font = Font(name="맑은 고딕", size=9, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="5B2BD9")
    body = Font(name="맑은 고딕", size=10)
    mono = Font(name="Consolas", size=10)

    wb = Workbook()

    def _table(ws, rows, headers):
        ws.append(headers)
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(horizontal="left", vertical="center")
        for r in rows:
            ws.append(list(r))
        for row in ws.iter_rows(min_row=2):
            for i, c in enumerate(row):
                c.font = mono if i == 1 else body
                c.border = Border(bottom=thin)
        for col, w in zip("ABC", (34, 18, 12)):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"

    ws = wb.active
    ws.title = "결과"
    _table(ws, b["results"], ["항목", "값", "단위"])
    ws.append([])
    ws.append(["판정", b.get("verdict", "")])
    ws.append(["결과 코드", b.get("code", "")])
    ws.append(["생성 시각", b.get("when", "")])
    ws.append([])
    ws.append([FOOT])

    _table(wb.create_sheet("조건"), b["conditions"], ["조건", "값", "단위"])

    g = b.get("grid")
    if g is not None:
        dz, dose, pw = g
        gs = wb.create_sheet("공정 윈도 격자")
        gs.append(["초점 \\ 노광"] + [f"{d:.4f}" for d in dose])
        for c in gs[1]:
            c.font, c.fill = head_font, head_fill
        for i, z in enumerate(dz):
            gs.append([f"{z:.1f}"] + ["통과" if bool(v) else "" for v in pw[i]])
        for row in gs.iter_rows(min_row=2):
            for c in row:
                c.font = mono
        gs.column_dimensions["A"].width = 14
        gs.freeze_panes = "B2"
        gs["A1"].alignment = Alignment(horizontal="left")

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _runs(txt):
    """글자를 아스키 토막과 한글 토막으로 가른다.

    `CD 3시그마` 처럼 섞인 줄을 한 글꼴로 그리면 `CD` 가 타자기 글씨로
    나온다. 토막마다 맞는 글꼴로 그려야 한 줄 안에서도 결이 같다.
    """
    out, cur, cur_ascii = [], [], None
    for ch in str(txt):
        a = ch.isascii()
        if cur and a != cur_ascii:
            out.append(("".join(cur), cur_ascii))
            cur = []
        cur.append(ch)
        cur_ascii = a
    if cur:
        out.append(("".join(cur), cur_ascii))
    return out


def _draw(c, x, y, txt, size, rgb) -> float:
    """토막마다 글꼴을 바꿔 그리고 끝 x 를 돌려준다."""
    c.setFillColorRGB(*rgb)
    for piece, is_ascii in _runs(txt):
        f = NUM if is_ascii else KO
        c.setFont(f, size)
        c.drawString(x, y, piece)
        x += c.stringWidth(piece, f, size)
    return x


def _ascii(c, x, y, txt, size, rgb, bold: bool = False) -> None:
    """아스키만 들어오는 글(머리표·표 이름). 한글이 섞이면 **그 자리에서**
    멈춘다. 라틴 글꼴은 한글을 못 그려서 글자가 통째로 빠지는데, 조용히
    빠지면 아무도 모른다. `_draw` 를 쓰라고 알려 주는 쪽이 낫다."""
    if not str(txt).isascii():
        raise ValueError(f"아스키가 아닌 글입니다. _draw 를 쓰십시오 : {txt!r}")
    c.setFillColorRGB(*rgb)
    c.setFont(NUM_B if bold else NUM, size)
    c.drawString(x, y, str(txt))


def _width(c, txt, size) -> float:
    """`_draw` 로 그렸을 때의 폭."""
    return sum(c.stringWidth(p, NUM if a else KO, size) for p, a in _runs(txt))


def _register() -> None:
    """내장 CID 한글 글꼴을 등록한다. 이미 있으면 조용히 지나간다."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    if KO not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(KO))


def to_pdf(b: dict) -> bytes:
    """A4 한 장. 큰 숫자 넷과 조건표, 그리고 면책."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    _register()
    W, H = A4
    M = 48.0                      # 바깥 여백
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    c.setTitle(b["title"])

    # 바탕 : 화면과 같은 종이색
    c.setFillColorRGB(*PAPER)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    y = H - M

    # 머리표 + 제목
    _ascii(c, M, y, "PHOTOLITHOGRAPHY SIMULATOR", 8, ACCENT, bold=True)
    y -= 24
    # 제목에도 라틴 글자가 섞인다("CD", "NILS"). CID 글꼴로 그리면 타자기처럼
    # 벌어지므로 여기도 토막을 갈라 그린다.
    _draw(c, M, y, str(b["title"]), 19, INK)
    y -= 16
    _draw(c, M, y, f"{b['when']}   ·   결과 코드 {b['code']}", 8.5, MUTED)

    # 판정 알약
    if b.get("verdict"):
        kind = b["verdict"]
        fg = ACCENT if kind == "통과" else (AMBER if kind == "위반" else MUTED)
        tw = _width(c, kind, 9) + 16
        c.setFillColorRGB(*fg)
        c.roundRect(W - M - tw, y - 3, tw, 18, 9, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont(KO, 9)
        c.drawCentredString(W - M - tw / 2, y + 2, kind)

    y -= 18
    c.setStrokeColorRGB(*LINE)
    c.setLineWidth(0.8)
    c.line(M, y, W - M, y)
    y -= 34

    # 큰 숫자 (앞의 넷)
    cols = min(4, len(b["results"]))
    if cols:
        cw = (W - 2 * M) / cols
        for i, (name, val, unit) in enumerate(b["results"][:cols]):
            x = M + i * cw
            _draw(c, x, y, name, 8.5, MUTED)
            # 값에도 한글이 올 수 있다("판정 불가"). 라틴 글꼴로만 그리면
            # 글자가 통째로 빠진다. 폭도 같은 규칙으로 잰다.
            _draw(c, x, y - 27, str(val), 23, INK)
            if unit:
                w = _width(c, str(val), 23)
                _draw(c, x + w + 5, y - 27, unit, 11, MUTED)
        y -= 48
        c.setStrokeColorRGB(*LINE)
        c.line(M, y, W - M, y)
        y -= 26

    def _table(title, rows, y0):
        _ascii(c, M, y0, title, 7.5, ACCENT, bold=True)
        yy = y0 - 15
        for name, val, unit in rows:
            if yy < M + 70:                      # 한 장을 넘기지 않는다
                _draw(c, M, yy, "... 나머지는 Excel 파일에 있습니다", 9.5, MUTED)
                return yy - 14
            _draw(c, M, yy, name, 9.5, INK)
            # 오른쪽 맞춤도 손으로 한다. drawRightString 은 글꼴 하나만 쓴다.
            _draw(c, W - M - 34 - _width(c, str(val), 9.5), yy,
                  str(val), 9.5, INK)
            if unit:
                _draw(c, W - M - 30, yy, unit, 9, MUTED)
            c.setStrokeColorRGB(*LINE)
            c.setLineWidth(0.4)
            c.line(M, yy - 5, W - M, yy - 5)
            yy -= 17
        return yy - 10

    y = _table("RESULTS", b["results"], y)
    y = _table("CONDITIONS", b["conditions"], y - 8)

    if b.get("note"):
        for ln in str(b["note"]).splitlines()[:4]:
            _draw(c, M, y, ln, 8, MUTED)
            y -= 12

    # 면책 : 숫자를 **보는 자리**에 한 번 더 둔다
    c.setStrokeColorRGB(*LINE)
    c.setLineWidth(0.8)
    c.line(M, M + 30, W - M, M + 30)
    _draw(c, M, M + 16, FOOT, 7.5, MUTED)
    _draw(c, M, M + 4, "노광맛집 · 광운대학교 전자통신공학과 참빛설계학기",
          7.5, MUTED)

    c.showPage()
    c.save()
    return out.getvalue()


# ---------------------------------------------------------------------------

def _demo() -> dict:
    """표본 자료. 숫자는 14 주차 실측에서 가져왔다."""
    import numpy as np
    rng = np.linspace(-100, 100, 9)
    dose = np.linspace(0.85, 0.95, 7)
    pw = (np.abs(rng)[:, None] < 70) & (np.abs(dose - 0.897)[None, :] < 0.04)
    return build(
        conditions=[
            ("파장", "193.0", "nm"), ("개구수 NA", "1.162", ""),
            ("조명 sigma", "0.525", ""), ("피치", "250.0", "nm"),
            ("duty", "0.500", ""), ("목표 CD", "125.00", "nm"),
            ("플레어", "0.0200", ""), ("MSD", "2.00", "nm"),
            ("블러", "10.00", "nm"), ("z6", "0.0200", ""),
            ("z11", "0.0200", ""), ("편광", "TE", ""),
        ],
        results=[
            ("초점 심도", "187.5", "nm"), ("NILS", "2.326", ""),
            ("CD 3시그마", "0.65", "%"), ("노광 여유", "10.0", "%"),
            ("dose-to-size", "0.8969", ""), ("공정 윈도 격자점", "286 / 1089", ""),
            ("공차 pw_tol", "7.07", "%"), ("NILS 문턱", "2.121", ""),
        ],
        verdict="통과",
        grid=(rng, dose, pw),
        note="초점 오프셋의 부호는 CD 계측만으로 정해지지 않습니다.\n"
             "장비의 수차 계측에서 받아야 합니다.",
    )


if __name__ == "__main__":
    import os
    import sys
    b = _demo()
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(d, exist_ok=True)
    for name, data in (("공정윈도_결과.pdf", to_pdf(b)),
                       ("공정윈도_결과.xlsx", to_xlsx(b))):
        p = os.path.join(d, name)
        with open(p, "wb") as fh:
            fh.write(data)
        print(f"  {p}  {len(data):,} 바이트")
    print(f"  결과 코드 {b['code']}")
