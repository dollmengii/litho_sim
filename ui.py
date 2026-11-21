"""화면 서식. `app.py` 가 쓴다.

왜 따로 두는가
--------------
서식 코드가 본문에 섞이면 `app.py` 에서 물리 계산을 읽기 어려워진다. 색과
간격은 한곳에 모아 두고 바꾸는 편이 낫다. 수치 자체는 여기서 건드리지
않는다.

설계 방침 : 주제에서 나온 색, 활자 위주의 구조
----------------------------------------------
14 주차에 어두운 '계측 장비' 풍에서 바꾸었다. 바꾼 이유와 규칙을 적어 둔다.

**색은 공정에서 가져왔다.** 클린룸이 노란 조명인 이유는 레지스트가 청색과
자외선에 반응하기 때문이고, ArF 는 193 nm 심자외선이다. 그래서

- 바탕은 따뜻한 종이색(클린룸 황색등)
- 강조는 **자외선 보라**
- 레지스트·임계값은 **클린룸 앰버**

로 두었다. 임의로 고른 파랑이 아니라 이 과제가 다루는 물리에서 나온
배색이므로, 발표에서 한 문장으로 설명할 수 있다.

**판정 세 값은 색상만이 아니라 밝기로도 가른다.** 통과(보라) · 판정 불가
(회색) · 위반(앰버)이다. 빨강·초록으로 가르면 색각 이상에서 구분되지
않는다. 세 색의 명도가 서로 다르므로 흑백으로 인쇄해도 구별된다.

**카드 그림자 대신 머리카락 굵기 선.** 그림자와 큰 모서리를 쓰면 카드가
떠 보이고 한 화면의 정보 밀도가 떨어진다. 1 px 선과 여백으로만 가른다.

**숫자는 고정폭.** 조건을 바꾸면 값이 바뀌는데 비례폭은 자릿수가 달라질
때마다 글자가 좌우로 밀려 비교가 어렵다. 고정폭이면 자리가 고정되어
변화만 눈에 들어온다. '공대생 같다' 는 인상은 고정폭 때문이 아니라
어두운 크롬과 matplotlib 기본값 때문이었다.

**단위는 흐리게.** `193.0 nm` 에서 읽어야 하는 것은 193.0 이다.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

#: 색. 한곳에서만 고친다.
BG = "#FBF9F4"          # 바탕 (클린룸 황색등을 아주 옅게)
SURF = "#FFFFFF"        # 면 (카드·사이드바)
LINE = "#E7E2D8"        # 머리카락 굵기 선
TXT = "#15131A"         # 본문 (보라가 살짝 도는 먹)
MUTED = "#5C5768"       # 단위·설명 (바탕 대비 5.6:1, 작은 글씨도 읽힌다)
ACCENT = "#5B2BD9"      # 강조 — 심자외선 보라
AMBER = "#B4600E"       # 신호 — 클린룸 황색등

#: 판정 세 값. 색상과 **밝기**가 모두 다르다.
PASS = ACCENT
UNDET = MUTED
VIOL = AMBER
#: 판정 알약의 바탕
PASS_BG = "#EFEAFD"
UNDET_BG = "#EFEDE7"
VIOL_BG = "#FBF0DE"

#: 옛 이름. 바깥에서 쓰지 않지만 지우면 남의 코드가 깨지므로 남긴다.
OK, WARN, BAD = PASS, VIOL, "#6B2545"

# 차트 색. `charts.py` 가 쓴다. 세 색의 **명도가 서로 다르므로** 흑백으로
# 인쇄해도 구별된다. 색상만으로 가르면 색각 이상에서 겹친다.
FG = TXT                # 본문과 같은 밝기
BEAM = ACCENT           # 광강도 — 자외선 보라
RESIST = AMBER          # 임계 · 레지스트 — 앰버 (실제 레지스트 색이다)
SPEC = "#2F6F6B"        # 규격 띠 — 청록. 위 둘과 색상·명도가 모두 다르다

#: 글꼴. 한글이 있는 산세리프와 짝이 되는 고정폭을 함께 쓴다.
SANS = "'IBM Plex Sans KR', system-ui, -apple-system, sans-serif"
MONO = "'IBM Plex Mono', 'Consolas', monospace"

#: 웹 글꼴 주소. `css()` 와 내보내기 쪽이 같이 쓴다.
FONT_URL = ("https://fonts.googleapis.com/css2?"
            "family=IBM+Plex+Mono:wght@400;500;600&"
            "family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&display=swap")


def css() -> str:
    """화면 전체에 입히는 스타일. `st.markdown(..., unsafe_allow_html=True)`.

    선택자는 Streamlit 이 만드는 구조를 가리킨다. 판이 바뀌면 일부가 듣지
    않을 수 있으나, **듣지 않아도 화면은 깨지지 않는다** (색과 간격만
    기본값으로 돌아간다). 그래서 레이아웃을 선택자에 의존시키지 않았다.
    """
    return f"""
<style>
@import url('{FONT_URL}');

:root {{
  --bg:{BG}; --surf:{SURF}; --line:{LINE};
  --txt:{TXT}; --muted:{MUTED}; --accent:{ACCENT}; --amber:{AMBER};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg); color: var(--txt);
  font-family: {SANS};
}}

.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1500px; }}

/* 제목 : 굵고 자간을 좁혀 또렷하게. 설명은 한 단계 낮춘다 */
h1 {{
  font-size: 1.9rem !important; font-weight: 700;
  letter-spacing: -0.03em; margin-bottom: .2rem; color: var(--txt);
}}
h2 {{ font-size: 1.05rem !important; font-weight: 600; letter-spacing: -0.01em; }}
h3 {{ font-size: .94rem !important; font-weight: 600; }}

/* 숫자는 고정폭. 자릿수가 바뀌어도 자리가 밀리지 않는다 */
[data-testid="stMetricValue"] {{
  font-family: {MONO}; font-size: 1.6rem; font-weight: 500;
  letter-spacing: -0.02em; color: var(--txt);
}}
[data-testid="stMetricLabel"] {{
  font-size: .74rem; color: var(--muted); letter-spacing: 0;
  text-transform: none; font-weight: 400;
}}
[data-testid="stMetricDelta"] {{ font-family: {MONO}; font-size: .76rem; }}

/* 지표 카드 : 흰 면에 선 한 겹. 그림자 없음 */
[data-testid="stMetric"] {{
  background: var(--surf); border: 1px solid var(--line);
  border-radius: 10px; padding: .85rem 1rem;
}}

/* 탭 : 알약 대신 밑줄 */
.stTabs [data-baseweb="tab-list"] {{ gap: .1rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{
  height: 2.4rem; padding: 0 .95rem; font-size: .86rem;
  color: var(--muted); border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
  color: var(--txt) !important; font-weight: 600;
  box-shadow: inset 0 -2px 0 0 var(--accent);
}}

/* 사이드바 : 손잡이가 많으므로 글자와 간격을 줄인다 */
section[data-testid="stSidebar"] {{
  background: var(--surf); border-right: 1px solid var(--line);
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {{
  font-size: .78rem !important; color: var(--txt);
}}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  font-size: .78rem; margin-bottom: .1rem; color: var(--muted);
}}
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] [role="slider"] {{
  font-family: {MONO}; font-size: .70rem;
}}
section[data-testid="stSidebar"] .stSlider {{ padding-bottom: .2rem; }}

/* 단추 : 주 동작만 채우고 나머지는 선 */
.stButton button, .stDownloadButton button {{
  font-family: {SANS}; font-size: .84rem; font-weight: 500;
  border-radius: 8px; border: 1px solid var(--line);
  background: var(--surf); color: var(--txt); padding: .45rem 1rem;
}}
.stButton button:hover, .stDownloadButton button:hover {{
  border-color: var(--accent); color: var(--accent);
}}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{
  background: var(--accent); border-color: var(--accent); color: #fff;
}}

/* 접힘 머리말 : 구역 이름이므로 작게 */
[data-testid="stExpander"] summary p {{
  font-size: .80rem; letter-spacing: 0; color: var(--muted); font-weight: 500;
}}
[data-testid="stExpander"] {{
  border-color: var(--line) !important; border-radius: 10px !important;
  background: var(--surf);
}}

/* 설명문 */
[data-testid="stCaptionContainer"] p {{
  color: var(--muted); font-size: .79rem; line-height: 1.65;
}}

/* 표 : 숫자 칸을 고정폭으로 */
[data-testid="stDataFrame"] {{ font-family: {MONO}; font-size: .80rem; }}

/* 알림 상자 : 색 띠를 왼쪽 2 px 로만 남기고 바탕은 흰 면.
   22 개가 알록달록하면 정작 중요한 숫자가 묻힌다 */
[data-testid="stAlert"] {{
  background: var(--surf); border: 1px solid var(--line);
  border-left: 2px solid var(--muted); border-radius: 8px;
  color: var(--txt);
}}
[data-testid="stAlert"] p {{ font-size: .82rem; line-height: 1.65; }}
div[data-baseweb="notification"][kind="info"],
[data-testid="stAlert"]:has(svg[data-testid="stInfoIcon"]) {{
  border-left-color: var(--accent);
}}
[data-testid="stAlert"]:has(svg[data-testid="stWarningIcon"]),
[data-testid="stAlert"]:has(svg[data-testid="stErrorIcon"]) {{
  border-left-color: var(--amber);
}}

hr {{ border-color: var(--line); margin: 1.1rem 0; }}

/* 아래 정의한 보조 요소들 */
.k-unit {{ color: var(--muted); font-size: .78em; margin-left: .2em; }}
.k-bar {{ display: flex; gap: .4rem; flex-wrap: wrap; margin: .2rem 0 1rem 0; }}
.k-chip {{
  font-family: {MONO}; font-size: .73rem; color: var(--muted);
  background: var(--surf); border: 1px solid var(--line);
  border-radius: 999px; padding: .2rem .65rem;
}}
.k-chip b {{ color: var(--txt); font-weight: 500; }}
.k-sec {{
  font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); font-family: {MONO}; font-weight: 500;
  margin: 1.2rem 0 .35rem 0;
}}
.k-pill {{
  display: inline-block; font-size: .76rem; font-weight: 600;
  border-radius: 999px; padding: .18rem .6rem;
}}
.k-motif {{
  display: flex; height: 40px; border-radius: 8px;
  overflow: hidden; border: 1px solid var(--line); margin: .3rem 0;
}}
.k-motif > div {{ flex: 1; }}

/* --- 캔버스 디자인을 옮긴 부분 ------------------------------------- */

/* Streamlit 의 기본 머리띠와 '배포' 단추를 지운다. 화면 맨 위가 우리
   머리표로 시작해야 레이아웃이 디자인과 같아진다 */
/* 머리띠는 **지우지 않는다.** 레일을 접으면 다시 펴는 화살표가 그 안에
   들어 있어서, 높이를 0 으로 누르면 화살표까지 잘려 나간다. 한 번 접으면
   새로 고치는 것 말고는 되돌릴 길이 없었다.
   처음에는 화살표만 골라 꺼내려고 `data-testid` 이름을 적었는데, 판마다
   이름이 달라 그 방법으로는 안 먹었다. **이름을 맞히는 대신 가리지 않는
   쪽으로** 바꾼다. 배경만 투명하게 하고 자리는 남겨 둔다 */
header[data-testid="stHeader"] {{
  background: transparent; overflow: visible;
}}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}

/* 아래 규칙은 덤이다. 이름이 맞는 판에서는 화살표를 더 잘 보이게 한다.
   맞지 않아도 위에서 이미 가리지 않으므로 화살표는 남는다 */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
  display: flex !important; visibility: visible !important; z-index: 1000;
}}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {{
  background: {SURF}; border: 1px solid {LINE}; border-radius: 8px;
  color: {TXT}; box-shadow: 0 1px 3px rgba(21,19,26,.08);
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {{ fill: {TXT}; color: {TXT}; }}

/* 사이드바를 '레일' 로. 브랜드 마크가 맨 위에 앉는다 */
section[data-testid="stSidebar"] > div {{ padding-top: 1.6rem; }}
.k-brand {{ margin: 0 0 1.4rem 0; }}
.k-brand .n {{ font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }}
.k-brand .s {{
  font-family: {MONO}; font-size: .66rem; letter-spacing: .14em;
  color: var(--muted); margin-top: .15rem;
}}

/* 왼쪽 메뉴 : 라디오 단추를 **목록**처럼 보이게 한다.
   동그란 단추를 지우고 한 줄 전체를 누를 수 있는 칸으로 만든다.
   선택된 칸만 연보라 바탕 + 굵은 글씨 */
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  width: 100%; padding: .5rem .7rem; border-radius: 8px; margin: 0;
  font-size: .88rem !important; color: var(--muted); cursor: pointer;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: var(--bg);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
  display: none;                 /* 동그란 단추를 지운다 */
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: {PASS_BG}; color: var(--txt); font-weight: 600;
}}

/* 히어로 : 머리표 + 큰 제목 + 설명 */
.k-hero {{ padding-bottom: 1.1rem; border-bottom: 1px solid var(--line); }}
.k-hero .e {{
  font-family: {MONO}; font-size: .70rem; letter-spacing: .16em;
  color: var(--accent); font-weight: 500;
}}
.k-hero .t {{
  font-size: 2.0rem; font-weight: 700; letter-spacing: -0.035em;
  line-height: 1.15; margin: .35rem 0 0 0;
}}
.k-hero .d {{
  font-size: .88rem; color: var(--muted); margin: .5rem 0 0 0;
  line-height: 1.7; max-width: 62ch;
}}

/* 큰 숫자 줄 : 카드가 아니라 **여백과 선**으로 가른다 */
.k-stats {{
  display: grid; gap: 1.6rem; margin: 1.5rem 0 1.3rem 0;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
}}
.k-stat .l {{ font-size: .76rem; color: var(--muted); margin-bottom: .45rem; }}
.k-stat .v {{
  font-family: {MONO}; font-size: 2.1rem; font-weight: 500;
  letter-spacing: -0.03em; line-height: 1;
}}
.k-stat .u {{ font-family: {SANS}; font-size: .8rem; color: var(--muted); margin-left: .25rem; }}
.k-stat .n {{ font-size: .70rem; color: var(--muted); margin-top: .5rem; }}
.k-rule {{ border-top: 1px solid var(--line); margin: 1.3rem 0; }}
</style>
"""


def chips(items) -> str:
    """조건 요약 띠. (이름, 값, 단위) 를 받아 한 줄로 만든다.

    탭을 옮겨 다녀도 **지금 어떤 조건을 보고 있는지**가 늘 보이게 한다.
    조건이 사이드바에만 있으면 접은 뒤에는 알 수 없다.
    """
    out = ['<div class="k-bar">']
    for name, val, unit in items:
        u = f'<span class="k-unit">{unit}</span>' if unit else ""
        out.append(f'<span class="k-chip">{name} <b>{val}</b>{u}</span>')
    out.append("</div>")
    return "".join(out)


def section(title: str) -> str:
    """구역 머리말. 파란 대문자 머리표 자리에 보라 고정폭으로 쓴다."""
    return f'<div class="k-sec">{title}</div>'


def pill(text: str, kind: str = "undet") -> str:
    """판정 알약. kind 는 pass / undet / viol.

    세 값의 **명도가 서로 다르다.** 흑백으로 인쇄해도 구별되고, 색각 이상
    에서도 구별된다. 빨강·초록 짝을 쓰지 않은 이유가 그것이다.
    """
    fg, bg = {
        "pass": (PASS, PASS_BG),
        "viol": (VIOL, VIOL_BG),
    }.get(kind, (UNDET, UNDET_BG))
    return (f'<span class="k-pill" style="color:{fg};background:{bg}">'
            f'{text}</span>')


def motif(blurs=(3.0, 1.7, 0.8, 0.0), pitch=16, line=7) -> str:
    """초점이 맞아 들어가는 선-공간 패턴.

    돌아가는 고리 대신 쓴다. 흐릿하던 패턴이 또렷해지는 것이 이 과제가
    실제로 재는 일이다. 장식이 아니라 주제의 그림이다.
    """
    # 흐림이 줄 너비보다 크면 선이 아예 지워져 회색 얼룩이 된다. 줄을
    # 넓게 두고 흐림을 그보다 작게 잡는다 (실제로 지워져서 고친 값이다).
    cells = "".join(
        f'<div style="background:repeating-linear-gradient('
        f'90deg,{TXT} 0 {line}px,{BG} {line}px {pitch}px);'
        f'filter:blur({b}px)"></div>'
        for b in blurs)
    return f'<div class="k-motif" aria-hidden="true">{cells}</div>'


def brand(name: str = "노광맛집", sub: str = "LITHO SIM") -> str:
    """사이드바 맨 위의 브랜드 마크. 레일처럼 보이게 하는 첫 조각이다."""
    return (f'<div class="k-brand"><div class="n">{name}</div>'
            f'<div class="s">{sub}</div></div>')


def hero(eyebrow: str, title: str, desc: str = "") -> str:
    """머리표 + 큰 제목 + 설명.

    `st.title` 을 쓰지 않는 이유 : 제목 위의 **작은 대문자 머리표**가
    디자인의 핵심인데 Streamlit 에는 그 자리가 없다. 한 덩어리로 그린다.
    """
    d = f'<p class="d">{desc}</p>' if desc else ""
    return (f'<div class="k-hero"><div class="e">{eyebrow}</div>'
            f'<h1 class="t">{title}</h1>{d}</div>')


def stats(items) -> str:
    """큰 숫자 줄. (이름, 값, 단위, 비고) 를 받는다.

    `st.metric` 을 쓰지 않는 이유 : 지표 카드는 테두리 상자로 그려진다.
    이 디자인은 **카드가 아니라 여백과 선**으로 가르므로, 같은 내용을
    상자 없이 그린다. 값은 고정폭, 단위는 작고 흐리게.
    """
    out = ['<div class="k-stats">']
    for it in items:
        name, val = it[0], it[1]
        unit = it[2] if len(it) > 2 else ""
        note = it[3] if len(it) > 3 else ""
        u = f'<span class="u">{unit}</span>' if unit else ""
        n_ = f'<div class="n">{note}</div>' if note else ""
        out.append(f'<div class="k-stat"><div class="l">{name}</div>'
                   f'<div class="v">{val}{u}</div>{n_}</div>')
    out.append("</div>")
    return "".join(out)


def rule() -> str:
    """머리카락 굵기 구분선. `st.divider` 보다 얇고 여백이 작다."""
    return '<div class="k-rule"></div>'


def mpl_theme() -> None:
    """그래프를 화면과 같은 바탕으로 맞춘다.

    **글꼴 계열은 건드리지 않는다.** `app._set_korean_font()` 가 먼저
    정한 한글 글꼴을 덮으면 축 이름이 네모로 나온다. 여기서는 색과 선
    굵기만 정한다.

    위·오른쪽 테두리를 없애고 격자를 아주 연하게 두는 것이 matplotlib
    기본값과 가장 크게 갈리는 지점이다. 기본값의 사방 테두리가 '논문
    그림' 인상을 만든다.
    """
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TXT,
        "text.color": TXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": LINE,
        "grid.alpha": 0.9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.facecolor": SURF,
        "legend.edgecolor": LINE,
        "legend.labelcolor": TXT,
        "legend.frameon": True,
        # 여섯 색의 **명도가 서로 다르다**. 색상만으로 가르지 않는다.
        "axes.prop_cycle": plt.cycler(
            color=[ACCENT, AMBER, SPEC, "#6B2545", "#8A7CC2", "#4F5B33"]),
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "figure.autolayout": True,
    })


#: 옛 이름. `app.py` 가 이 이름으로 부른다. 이름만 남기고 내용은 위와 같다.
#: (바탕이 더 이상 어둡지 않으므로 이름이 사실과 다르지만, 부르는 쪽을
#:  같이 고치지 않으면 조용히 서식이 빠진다. 이름은 15 주차에 정리한다.)
mpl_dark = mpl_theme
