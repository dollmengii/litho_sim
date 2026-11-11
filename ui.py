"""화면 서식. `app.py` 가 쓴다.

왜 따로 두는가
--------------
서식 코드가 본문에 섞이면 `app.py` 에서 물리 계산을 읽기 어려워진다. 색과
간격은 한곳에 모아 두고 바꾸는 편이 낫다. 수치 자체는 여기서 건드리지
않는다.

설계 방침 : 계측 장비의 화면처럼
--------------------------------
이 화면의 독자는 조건을 바꿔 가며 숫자를 읽는 사람이다. 그래서 꾸밈보다
**판독성**을 앞에 둔다.

- **숫자는 고정폭 글꼴로.** 조건을 바꾸면 값이 바뀌는데, 비례폭 글꼴은
  자릿수가 달라질 때마다 글자가 좌우로 밀려 비교가 어렵다. 고정폭이면
  자리가 고정되어 변화만 눈에 들어온다.
- **단위는 흐리게.** `193.0 nm` 에서 읽어야 하는 것은 193.0 이다. 단위를
  같은 밝기로 두면 눈이 매번 단위까지 훑는다.
- **어두운 바탕.** 그래프가 대부분이고 흰 바탕에서는 그래프 영역만
  밝아 시선이 분산된다. 바탕을 낮추면 선과 값이 떠오른다.
- **테두리는 1 px 한 겹만.** 그림자와 둥근 모서리를 키우면 카드가
  떠 보이고 밀도가 떨어진다. 한 겹 선으로 영역만 가른다.
- **여백을 줄임.** 기본 설정은 한 화면에 들어오는 정보가 적다. 위아래
  여백을 줄여 조건과 결과를 같은 화면에서 본다.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

#: 색. 한곳에서만 고친다.
BG = "#0E1116"          # 바탕
SURF = "#161A21"        # 카드·사이드바
LINE = "#262C36"        # 테두리 (한 겹)
TXT = "#C9D1D9"         # 본문
MUTED = "#8B949E"       # 단위·설명
ACCENT = "#58A6FF"      # 강조 (파랑)
OK = "#3FB950"
WARN = "#D29922"
BAD = "#F85149"

#: 숫자용 고정폭 글꼴. 한글은 여기 없으므로 본문 글꼴이 이어받는다.
MONO = "'JetBrains Mono', 'IBM Plex Mono', 'Consolas', monospace"


def css() -> str:
    """화면 전체에 입히는 스타일. `st.markdown(..., unsafe_allow_html=True)`.

    선택자는 Streamlit 이 만드는 구조를 가리킨다. 판이 바뀌면 일부가 듣지
    않을 수 있으나, **듣지 않아도 화면은 깨지지 않는다** (색과 간격만
    기본값으로 돌아간다). 그래서 레이아웃을 선택자에 의존시키지 않았다.
    """
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --bg:{BG}; --surf:{SURF}; --line:{LINE};
  --txt:{TXT}; --muted:{MUTED}; --accent:{ACCENT};
}}

/* 위아래 여백을 줄여 조건과 결과를 같은 화면에 둔다 */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* 제목 : 크기를 낮추고 자간을 좁힌다. 장비 화면의 표제에 가깝게 */
h1 {{ font-size: 1.45rem !important; letter-spacing: -0.02em; margin-bottom: .1rem; }}
h2 {{ font-size: 1.08rem !important; letter-spacing: -0.01em; }}
h3 {{ font-size: .96rem !important; }}

/* 숫자는 고정폭. 자릿수가 바뀌어도 자리가 밀리지 않는다 */
[data-testid="stMetricValue"] {{
  font-family: {MONO}; font-size: 1.32rem; font-weight: 500;
  letter-spacing: -0.01em; color: var(--txt);
}}
[data-testid="stMetricLabel"] {{
  font-size: .70rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: .07em;
}}
[data-testid="stMetricDelta"] {{ font-family: {MONO}; font-size: .78rem; }}

/* 지표 카드 : 테두리 한 겹. 그림자 없음 */
[data-testid="stMetric"] {{
  background: var(--surf); border: 1px solid var(--line);
  border-radius: 6px; padding: .65rem .85rem;
}}

/* 탭 : 알약 대신 밑줄. 개발 도구의 탭 모양 */
.stTabs [data-baseweb="tab-list"] {{
  gap: .15rem; border-bottom: 1px solid var(--line);
}}
.stTabs [data-baseweb="tab"] {{
  height: 2.2rem; padding: 0 .9rem; font-size: .84rem;
  color: var(--muted); border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
  color: var(--txt) !important;
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
  font-size: .78rem; margin-bottom: .1rem;
}}
/* 슬라이더가 보여 주는 현재값도 고정폭으로 */
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] [role="slider"] {{
  font-family: {MONO}; font-size: .70rem;
}}
section[data-testid="stSidebar"] .stSlider {{ padding-bottom: .15rem; }}

/* 접힘 머리말 : 구역 이름이므로 작게, 대문자 자간으로 */
[data-testid="stExpander"] summary p {{
  font-size: .76rem; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted);
}}
[data-testid="stExpander"] {{ border-color: var(--line) !important; }}

/* 설명문 : 본문보다 한 단계 낮춘다 */
[data-testid="stCaptionContainer"] p {{ color: var(--muted); font-size: .78rem; }}

/* 표 : 숫자 칸을 고정폭으로 */
[data-testid="stDataFrame"] {{ font-family: {MONO}; font-size: .80rem; }}

/* 알림 상자 : 좌측 띠만 남기고 배경을 낮춘다 */
[data-testid="stAlert"] {{
  background: var(--surf); border: 1px solid var(--line);
  border-left-width: 3px; border-radius: 4px;
}}

/* 구분선 */
hr {{ border-color: var(--line); margin: .9rem 0; }}

/* 아래 정의한 보조 요소들 */
.k-unit {{ color: var(--muted); font-size: .78em; margin-left: .18em; }}
.k-bar {{
  display: flex; gap: .4rem; flex-wrap: wrap;
  margin: .1rem 0 .9rem 0;
}}
.k-chip {{
  font-family: {MONO}; font-size: .72rem; color: var(--muted);
  background: var(--surf); border: 1px solid var(--line);
  border-radius: 4px; padding: .14rem .5rem;
}}
.k-chip b {{ color: var(--txt); font-weight: 500; }}
.k-sec {{
  font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); border-bottom: 1px solid var(--line);
  padding-bottom: .3rem; margin: .4rem 0 .7rem 0;
}}
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
    """구역 머리말. `st.subheader` 보다 낮은 단계로 쓴다."""
    return f'<div class="k-sec">{title}</div>'


def mpl_dark() -> None:
    """그래프를 화면과 같은 바탕으로 맞춘다.

    **글꼴 계열은 건드리지 않는다.** `app._set_korean_font()` 가 먼저
    정한 한글 글꼴을 덮으면 축 이름이 네모로 나온다. 여기서는 색과 선
    굵기만 정한다.
    """
    plt.rcParams.update({
        "figure.facecolor": SURF,
        "axes.facecolor": BG,
        "savefig.facecolor": SURF,
        "axes.edgecolor": LINE,
        "axes.labelcolor": TXT,
        "axes.titlecolor": TXT,
        "text.color": TXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": LINE,
        "grid.alpha": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.facecolor": SURF,
        "legend.edgecolor": LINE,
        "legend.labelcolor": TXT,
        "axes.prop_cycle": plt.cycler(
            color=[ACCENT, "#F778BA", OK, WARN, "#A371F7", BAD]),
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "figure.autolayout": True,
    })
