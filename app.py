"""
app.py -- 포토리소그래피 공정 시뮬레이터 GUI (Streamlit)

실행: streamlit run app.py          (이 컴퓨터에서만 열림)
바깥 공개: 배포_안내.md 참조

화면 서식(색, 간격, 글꼴)은 `ui.py` 에 모아 두었다. 여기에는 물리와 화면
구성만 둔다.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import ui as UI

from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import optics as OP
from litho import process as PR
from litho import resist as RS
from litho import stochastic as ST
from litho import dataset as DD
from litho import identify as ID


# ───────────────────── 한글 폰트 설정 ─────────────────────
def _set_korean_font():
    """OS별 한글 폰트를 자동 선택한다. 없으면 기본 폰트로 둔다.

    [배포 대응] 배포 서버는 리눅스라 맑은 고딕도 AppleGothic 도 없다.
    `packages.txt` 가 나눔고딕을 깔지만, matplotlib 이 글꼴 목록을 캐시에
    담아 두므로 **방금 깔린 글꼴을 못 보는** 경우가 있다. 후보를 하나도
    못 찾으면 목록을 한 번 다시 읽고 재시도한다. 그래도 없으면 기본
    글꼴로 두고 넘어간다 (그래프의 한글이 네모로 나오지만 계산은 돈다).
    """
    candidates = [
        "Malgun Gothic",      # Windows
        "AppleGothic",        # macOS
        "NanumGothic",        # Linux (packages.txt 의 fonts-nanum)
        "NanumBarunGothic",
        "NanumGothicCoding",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]

    def _pick():
        avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for name in candidates:
            if name in avail:
                plt.rcParams["font.family"] = name
                return True
        return False

    if not _pick():
        try:
            matplotlib.font_manager._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        _pick()
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지


_set_korean_font()
# 그래프 색을 화면과 맞춘다. **한글 글꼴을 정한 뒤에** 부른다. `mpl_dark`
# 는 글꼴 계열을 건드리지 않지만, 순서를 지켜 두어야 나중에 어느 한쪽을
# 고칠 때 조용히 덮이지 않는다.
UI.mpl_dark()

st.set_page_config(page_title="포토리소그래피 공정 시뮬레이터", layout="wide")
st.markdown(UI.css(), unsafe_allow_html=True)
st.title("반도체 포토리소그래피 공정 시뮬레이터")
st.caption("팀 노광맛집 · 참빛설계 : 조건을 바꾸며 상(像)·현상 결과·공정 윈도를 본다")


# ───────────────────── 광학 인자 묶음 ─────────────────────
def _unpack_optics(optx):
    """사이드바의 10주차 광학 인자 묶음을 풀어 aerial_image 인자로 만든다.

    st.cache_data 가 인자를 해시하므로 묶음은 불변 튜플로 넘긴다.
    """
    (pol, shape, p_off, p_rad, flare, msd, z5, z6, z7, z8, z11,
     n_res, mtype, att) = optx
    # Z7(코마 y)도 넣는다. 1 차원 선 패턴이라 효과가 없을 것 같지만, 광원이
    # 확장돼 있으면 광원점마다 차수가 sy 방향으로도 이동해 방위각이 0 이
    # 아니게 된다. 실측하면 0.10 waves 에서 상대 변화 1.18e-01 로 Z6 과
    # 같은 수준이다. 빠져 있으면 사용자가 코마 y 를 탐색할 수 없다.
    ab = {k: v for k, v in ((5, z5), (6, z6), (7, z7), (8, z8), (11, z11))
          if v != 0.0}
    opt = dict(polarization=pol, source_shape=shape, pole_offset=p_off,
               pole_radius=p_rad, aberrations=ab, flare=flare,
               msd_nm=msd, n_resist=n_res)
    return opt, mtype, att


# ───────────────────── 연산 캐시 ─────────────────────
@st.cache_data(show_spinner=False)
def compute(
    pitch, duty, polarity, n_periods, dx_target,
    wavelength, na, n_medium, sigma_out, sigma_in, n_src, defocus, optx,
):
    """슬라이더 값이 같으면 재연산하지 않도록 캐싱한다."""
    opt, mtype, att = _unpack_optics(optx)
    g = G.build_grid(pitch=pitch, n_periods=n_periods, dx_target=dx_target)
    m = MK.line_space(g, cd=duty * pitch, polarity=polarity,
                      mask_type=mtype, att_t=att)
    i = IM.aerial_image(
        g, m, wavelength, na,
        n_medium=n_medium, sigma_out=sigma_out, sigma_in=sigma_in,
        defocus=defocus, n_source_grid=n_src, **opt,
    )
    return g, m, i


@st.cache_data(show_spinner="Dose-Focus 스윕 연산 중...")
def sweep(
    pitch, duty, polarity, n_periods, dx_target,
    wavelength, na, n_medium, sigma_out, sigma_in, n_src,
    dz_range, n_focus, n_dose, dose_span, threshold, tol, optx,
):
    """Dose-Focus 스윕. 결상 연산은 Defocus 에만 의존하므로 n_focus 회만 수행."""
    opt, mtype, att = _unpack_optics(optx)
    g = G.build_grid(pitch=pitch, n_periods=n_periods, dx_target=dx_target)
    m = MK.line_space(g, cd=duty * pitch, polarity=polarity,
                      mask_type=mtype, att_t=att)
    # [12차 보완 69 번] 슬라이더 선택지가 전부 홀수라는 **주석**에만 기대지
    # 않는다. dz=0 이 격자점이어야 Bossung 극값의 금지대를 밟고(66 번), 바로
    # 아래 i0 (dose-to-size 를 재는 최적 초점 행)도 정확히 dz=0 이 된다.
    # 짝수 격자에서는 i0 가 +-간격/2 로 밀려 e_op 부터 틀어진다. 범위가
    # 대칭이므로 홀수에서는 linspace 와 같은 격자가 나오고, 선택지를 짝수로
    # 바꾸더라도 코드가 막는다.
    dz, dz_note = PR.centered_focus_grid(-dz_range, dz_range, n_focus)
    I = PR.focus_sweep(g, m, wavelength, na, n_medium=n_medium,
                       sigma_out=sigma_out, sigma_in=sigma_in,
                       defocuses=dz, n_source_grid=n_src, **opt)
    target = duty * pitch
    i0 = int(np.argmin(np.abs(dz)))
    e_op = PR.dose_to_size(g, I[i0], target, threshold,
                           RS.TONE_BRIGHT if polarity == "clear" else RS.TONE_DARK,
                           max_width=pitch)
    if not np.isfinite(e_op):
        return None
    doses = e_op * np.linspace(1 - dose_span, 1 + dose_span, n_dose)
    cd = PR.cd_matrix(g, I, doses, threshold,
                      RS.TONE_BRIGHT if polarity == "clear" else RS.TONE_DARK,
                      max_width=pitch)
    pw = PR.process_window(cd, target, tol)
    return {
        "dz": dz, "doses": doses, "e_op": e_op, "cd": cd, "pw": pw,
        "target": target, "sat": PR.window_saturated(pw),
        "curve": PR.el_dof_curve(pw, doses, dz),
        "dz_note": dz_note,
    }


@st.cache_data(show_spinner="확률 모델 Monte Carlo 연산 중...")
def stochastic_run(
    pitch, duty, polarity, n_periods, dx_target,
    wavelength, na, n_medium, sigma_out, sigma_in, n_src,
    dose_mj, blur_nm, threshold, y_length, n_trials, defocus, optx,
):
    """확률 레지스트 모델 실행. dose-to-size 보정 후 Monte Carlo.

    7주차 검증 결과를 반영하여 y 격자 간격 dy 를 x 격자와 분리하고,
    측정 길이를 직접 지정하도록 변경함. LER 은 반복 횟수가 아니라
    y 측정 길이에 수렴하므로 1000 nm 이상을 확보해야 한다.
    """
    opt, mtype, att = _unpack_optics(optx)
    dy = max(0.25, blur_nm / 3.0) if blur_nm > 0 else 0.5
    n_rows = max(64, int(round(y_length / dy)))
    g = G.build_grid(pitch=pitch, n_periods=n_periods, dx_target=dx_target)
    m = MK.line_space(g, cd=duty * pitch, polarity=polarity,
                      mask_type=mtype, att_t=att)
    I = IM.aerial_image(g, m, wavelength, na, n_medium=n_medium,
                        sigma_out=sigma_out, sigma_in=sigma_in,
                        defocus=defocus, n_source_grid=n_src, **opt)
    tone_ = RS.TONE_BRIGHT if polarity == "clear" else RS.TONE_DARK
    e_op = PR.dose_to_size(g, I, duty * pitch, threshold, tone_,
                           max_width=pitch)
    if not np.isfinite(e_op):
        return None
    In = I * e_op
    n_show = min(n_rows, 256)
    field = ST.stochastic_field(In, g.dx, dose_mj, wavelength,
                                n_rows=n_show, dy=dy, blur_nm=blur_nm,
                                rng=np.random.default_rng(0))
    mc = ST.monte_carlo(g.x, In, g.dx, dose_mj, wavelength, threshold,
                        n_rows=n_rows, blur_nm=blur_nm, n_trials=n_trials,
                        seed=1, dy=dy, max_width=pitch)
    return {"g": g, "I_det": In, "field": field, "mc": mc, "e_op": e_op,
            "tone": tone_, "dy": dy, "n_rows": n_rows}


# ───────────────────── 사이드바: 입력 ─────────────────────
sb = st.sidebar

with sb.expander("광학 조건", expanded=True):
    wavelength = st.slider("파장 λ [nm]", 13.5, 436.0, 193.0, 0.5)
    n_medium = st.selectbox(
        "결상 매질 굴절률 n", [1.00, 1.44], index=1,
        format_func=lambda v: f"{v:.2f}  ({'공기 (dry)' if v == 1.0 else '물 (ArF immersion)'})",
    )
    na = st.slider("개구수 NA", 0.10, 1.60, 1.35, 0.01)
    k1 = st.slider("공정 상수 k1", 0.25, 0.80, 0.288, 0.001)
    k2 = st.slider("공정 상수 k2", 0.20, 1.50, 0.745, 0.005)

with sb.expander("조명 조건", expanded=True):
    src_type = st.radio("광원 형상",
                        ["Conventional", "Annular", "Dipole", "Quadrupole"],
                        horizontal=True)
    sigma_out = st.slider("σ_out", 0.0, 1.0, 0.70, 0.01)
    sigma_in = (
        st.slider("σ_in", 0.0, 0.99, 0.60, 0.01) if src_type == "Annular" else 0.0
    )
    if src_type in ("Dipole", "Quadrupole"):
        pole_offset = st.slider("극 중심 위치", 0.30, 0.95, 0.75, 0.01)
        pole_radius = st.slider("극 반경", 0.05, 0.40, 0.15, 0.01)
        st.caption("조밀 선패턴은 0차와 1차만 pupil 양단에 넣는 다이폴이 유리함")
    else:
        pole_offset, pole_radius = 0.0, 0.15
    source_shape = {"Conventional": IM.SOURCE_CONVENTIONAL,
                    "Annular": IM.SOURCE_ANNULAR,
                    "Dipole": IM.SOURCE_DIPOLE,
                    "Quadrupole": IM.SOURCE_QUADRUPOLE}[src_type]
    n_src = st.select_slider("광원 샘플링 격자", [21, 31, 41, 51], value=31)

with sb.expander("편광 (10주차)", expanded=True):
    pol_label = st.radio(
        "입사 편광",
        ["TE (선에 평행)", "TM (선에 수직)", "무편광", "스칼라 (편광 무시)"],
        index=0,
    )
    polarization = {"TE (선에 평행)": IM.POL_TE, "TM (선에 수직)": IM.POL_TM,
                    "무편광": IM.POL_UNPOLARIZED,
                    "스칼라 (편광 무시)": IM.POL_SCALAR}[pol_label]
    n_resist = st.slider("편광 계산 굴절률 (레지스트)", 1.00, 2.00, 1.70, 0.01)
    st.caption("상은 레지스트 안에서 맺히므로 레지스트 굴절률로 계산하는 편이 "
               "물리적임. 침지액 값(1.44)을 쓰면 편광 효과를 과대평가함.")

with sb.expander("수차 · 플레어 · 스캔 (10주차)", expanded=False):
    z6 = st.slider("Z6 비점수차 0° [waves]", 0.0, 0.15, 0.0, 0.005)
    z5 = st.slider("Z5 비점수차 45° [waves]", 0.0, 0.15, 0.0, 0.005)
    z7 = st.slider("Z7 코마 y [waves]", 0.0, 0.15, 0.0, 0.005)
    z8 = st.slider("Z8 코마 x [waves]", 0.0, 0.15, 0.0, 0.005)
    z11 = st.slider("Z11 구면수차 [waves]", 0.0, 0.15, 0.0, 0.005)
    flare = st.slider("플레어 비율", 0.0, 0.10, 0.0, 0.005)
    msd_nm = st.slider("스캔 동기화 오차 MSD [nm]", 0.0, 12.0, 0.0, 0.5)
    if msd_nm > 0:
        st.caption("MSD 는 광자 샘플링 이전의 광강도를 평활하고 레지스트 블러는 "
                   "이후의 광자 수를 평활함. 평균 상에서는 구별되지 않으나 "
                   "LER 에서는 구별됨.")

with sb.expander("마스크 패턴", expanded=True):
    pitch = st.slider("Pitch [nm]", 20.0, 1000.0, 160.0, 1.0)
    duty = st.slider("Duty Cycle (CD / Pitch)", 0.10, 0.90, 0.50, 0.01)
    polarity = st.radio("극성", ["clear", "dark"], horizontal=True)
    mask_label = st.radio("마스크 종류", ["바이너리", "감쇠형 위상반전 (att-PSM)"],
                          horizontal=True)
    mask_type = (MK.MASK_BINARY if mask_label == "바이너리" else MK.MASK_ATT_PSM)
    att_t = (st.slider("att-PSM 투과율", 0.03, 0.20, 0.06, 0.01)
             if mask_type == MK.MASK_ATT_PSM else MK.ATT_PSM_DEFAULT_T)
    if mask_type == MK.MASK_ATT_PSM:
        st.caption("차광부가 위상 180°로 소량 투과하여 경계에서 상쇄간섭을 "
                   "일으킴. 현 ArF 공정의 표준 마스크임.")
    n_periods = st.slider("주기 수", 2, 16, 8, 1)
    dx_target = st.select_slider("목표 격자 간격 [nm]", [0.25, 0.5, 1.0, 2.0], value=0.5)

with sb.expander("노광 / 레지스트", expanded=True):
    defocus = st.slider("Defocus Δz [nm]", -300.0, 300.0, 0.0, 5.0)
    dose = st.slider("Dose (상대값)", 0.3, 3.0, 1.0, 0.01)
    threshold = st.slider("PR 임계 광강도", 0.05, 0.95, 0.30, 0.01)

with sb.expander("확률 레지스트 모델 (6주차)", expanded=False):
    dose_mj = st.slider("Dose [mJ/cm²]", 2.0, 100.0, 30.0, 1.0)
    blur_nm = st.slider("레지스트 블러 [nm]", 0.0, 30.0, 8.0, 0.5)
    _blo, _bhi = ST.blur_range(wavelength)
    if not (_blo <= blur_nm <= _bhi):
        st.caption(f"통용 범위 {_blo:.0f}~{_bhi:.0f} nm 밖입니다 "
                   f"(화학증폭형 레지스트의 PEB 확산 길이 대역).")
    y_length = st.select_slider(
        "y 측정 길이 [nm]", [128, 256, 512, 1024, 2048, 4096], value=2048)
    n_trials = st.select_slider("Monte Carlo 반복", [4, 8, 16, 32], value=8)
    if y_length < 1000:
        st.warning("y 측정 길이가 1000 nm 미만이면 LER 이 과소평가됩니다.")

with sb.expander("공정 윈도 분석 (6주차)", expanded=False):
    dz_range = st.slider("Defocus 스윕 범위 ±[nm]", 100.0, 800.0, 400.0, 50.0)
    # 선택지가 전부 **홀수**인 것은 우연이 아니다. dz=0 이 격자점이어야 한다.
    # Bossung 은 dz=0 에서 극값을 가지므로, 노광량이 높아 최적 초점 CD 가
    # 규격 상한을 넘는 dose 행에서는 통과 영역이 고리 모양(중심 초과)이 된다.
    # 짝수 격자는 얇은 금지대를 건너뛰고 양쪽을 하나로 읽어 DOF 를 과대평가
    # 한다(n_focus=32 실측 : 500.0 nm 보고, 참값 약 235.9 nm). 선택지를 홀수로
    # 묶어 막는다. VERIFICATION_LIMITS 66 번.
    #
    # 다만 선택지를 홀수로 묶는 것은 **주석에 기댄 보호**이고, 선택지를
    # 고치는 순간 조용히 깨진다. 실제 보호는 sweep 안의
    # `PR.centered_focus_grid` 가 한다. 69 번.
    n_focus = st.select_slider("Focus 분할 수", [17, 21, 33, 41], value=33)
    n_dose = st.select_slider("Dose 분할 수", [17, 21, 33, 41], value=33)
    dose_span = st.slider("Dose 스윕 폭 (±비율)", 0.10, 0.50, 0.25, 0.05)
    # [기본값 근거] 전체 CD 예산(0.10)을 그대로 쓰면 결정론 공정 윈도에
    # 예산을 다 주고 확률 변동(CDU) 몫을 0 으로 두는 것이 된다. 아래 데이터
    # 생성 패널에서 같은 GUI 가 "공정 윈도 몫 0.0707" 이라고 설명하는데
    # 여기 기본값이 0.10 이면 설명과 다른 값으로 윈도를 보여 주게 된다.
    # 유도값을 기본값으로 쓰고, 슬라이더 간격(0.01)에 맞춰 반올림한다.
    _pw_default = round(DD.SampleSpec(pitch=200.0, duty=0.5).pw_tol, 2)
    cd_tol = st.slider("CD 허용오차 (±비율)", 0.02, 0.20, _pw_default, 0.01)
    st.caption(
        f"기본값 {_pw_default:.2f} 는 CD 예산 0.10 을 공정 윈도와 확률 변동이 "
        f"제곱합으로 나눈 **공정 윈도 몫**입니다. 0.10 으로 올리면 예산 전체를 "
        f"결정론 윈도에 주는 것이므로 확률 변동 몫이 남지 않습니다."
    )
    el_target = st.slider("DOF 산출 기준 EL [%]", 1.0, 20.0, 10.0, 0.5)

cd_nm = duty * pitch

# 10주차 광학 인자 묶음. st.cache_data 해시를 위해 불변 튜플로 둔다.
optx = (polarization, source_shape, pole_offset, pole_radius,
        flare, msd_nm, z5, z6, z7, z8, z11, n_resist, mask_type, att_t)

# ───────────────────── 연산 ─────────────────────
# NA >= 매질 굴절률은 NA = n·sin(theta) 에 어긋나는 조합이다. 9주차까지는
# 결상 함수가 이를 막지 않아 defocus 위상이 조용히 사라졌다. 이제 예외를
# 던지므로 사용자에게 설명하고 멈춘다.
try:
    sim_grid, mask, intensity = compute(
        pitch, duty, polarity, n_periods, dx_target,
        wavelength, na, n_medium, sigma_out, sigma_in, n_src, defocus, optx,
    )
except ValueError as exc:
    st.error(str(exc))
    st.info(
        "NA 는 매질 굴절률을 넘을 수 없습니다 (NA = n·sinθ ≤ n).\n\n"
        "- 침지 노광을 보려면 결상 매질을 **물 (1.44)** 로 두십시오.\n"
        "- 건식(공기, n=1.00)이라면 NA 를 1.00 미만으로 낮추십시오."
    )
    st.stop()
sampling = G.check_sampling(sim_grid, wavelength, na, sigma_out)

try:
    rayleigh = OP.rayleigh_summary(wavelength, na, n_medium, k1, k2, k2,
                                   polarization=polarization)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

print_check = OP.printability(pitch, k1, wavelength, na)

intensity_dosed = RS.apply_dose(intensity, dose)
tone = RS.TONE_BRIGHT if polarity == "clear" else RS.TONE_DARK
cd_result = RS.extract_cd(sim_grid.x, intensity_dosed, threshold, tone,
                          max_width=pitch)

i_min = float(intensity_dosed.min())
i_max = float(intensity_dosed.max())


def failure_reason() -> str:
    """패턴 미형성 원인을 구분하여 설명한다."""
    if tone == RS.TONE_BRIGHT:
        if i_min >= threshold:
            return (
                "전면 노광 — 광강도 최솟값이 임계값보다 높아 패턴이 분리되지 않았습니다. "
                "Defocus 과다로 contrast 가 붕괴했거나 Dose 가 과다합니다."
            )
        if i_max < threshold:
            return "미노광 — 광강도 최댓값이 임계값에 미치지 못합니다. Dose 를 올리세요."
    else:
        if i_max < threshold:
            return "전면 잔존 — 광강도가 전 구간에서 임계값 미만입니다."
        if i_min >= threshold:
            return "전면 제거 — 광강도가 전 구간에서 임계값 이상입니다."
    if cd_result.get("reason"):
        return (f"{cd_result['reason']} — 인접 패턴이 붙어 CD 가 pitch "
                f"{pitch:.0f} nm 를 초과했습니다. Defocus 또는 Dose 를 줄이세요.")
    return "중앙 feature 의 경계를 찾지 못했습니다. 주기 수를 늘려보세요."


for msg in rayleigh["warnings"]:
    st.warning(msg)
if not sampling["ok"]:
    st.error(sampling["message"])
if not print_check["printable"]:
    st.warning(
        f"half-pitch {print_check['half_pitch']:.1f} nm 가 해상 한계 "
        f"{print_check['half_pitch_limit']:.1f} nm 미만입니다. "
        f"필요 k1 = {print_check['k1_required']:.3f}"
        + (" (이론 한계 0.25 미만 — 구현 불가)" if print_check["below_theoretical_limit"] else "")
    )

# ───────────────────── 출력 ─────────────────────
# 조건 요약 띠. 탭을 옮겨 다녀도 **지금 어떤 조건을 보고 있는지**가 보여야
# 한다. 사이드바는 접히고 스크롤되므로 그것만으로는 알 수 없다.
st.markdown(UI.chips([
    ("λ", f"{wavelength:g}", "nm"),
    ("NA", f"{na:.2f}", ""),
    ("n", f"{n_medium:.2f}", ""),
    ("σ", f"{sigma_out:.2f}", ""),
    ("pitch", f"{pitch:g}", "nm"),
    ("duty", f"{duty:.2f}", ""),
    ("목표 CD", f"{cd_nm:.1f}", "nm"),
    ("defocus", f"{defocus:+g}", "nm"),
    ("dose", f"{dose:.2f}", "×"),
    ("임계", f"{threshold:.2f}", ""),
    ("광원", src_type, ""),
    ("마스크", mask_label, ""),
]), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("해상 한계 half-pitch", f"{rayleigh['half_pitch_limit']:.1f} nm")
c2.metric("DOF (근축)", f"{rayleigh['dof_paraxial']:.1f} nm")
c3.metric(
    "DOF (고NA, Lin)",
    f"{rayleigh['dof_high_na']:.1f} nm" if rayleigh["dof_high_na"] else "—",
    delta=f"{(rayleigh.get('dof_ratio', 1) - 1) * 100:+.1f}%" if rayleigh.get("dof_ratio") else None,
)
c4.metric(
    "시뮬레이션 CD",
    f"{cd_result['cd']:.2f} nm" if cd_result["formed"] else "미형성",
    delta=f"{cd_result['cd'] - cd_nm:+.2f} nm" if cd_result["formed"] else None,
)

tab1, tab2, tab3, tab5, tab6, tab8, tab7, tab4 = st.tabs(
    ["마스크 패턴", "공간 광강도", "현상 후 패턴", "공정 윈도",
     "확률 모델", "식별 가능성", "데이터셋", "진단"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.step(sim_grid.x, np.abs(mask) ** 2, where="mid", lw=1.2)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("투과율 $|t|^2$")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 2.0))
    ax2.imshow(
        MK.to_2d(np.abs(mask) ** 2, 120), cmap="gray", aspect="auto",
        extent=[0, sim_grid.field_width, 0, sim_grid.field_width / 4],
    )
    ax2.set_xlabel("x [nm]")
    ax2.set_ylabel("y [nm]")
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)

    st.info(
        f"CD = {cd_nm:.1f} nm · Space = {pitch - cd_nm:.1f} nm · "
        f"half-pitch = {pitch / 2:.1f} nm · {sim_grid.describe()}"
    )

with tab2:
    fig3, ax3 = plt.subplots(figsize=(9, 3.2))
    ax3.plot(sim_grid.x, intensity_dosed, lw=1.8, label=f"Δz = {defocus:.0f} nm")
    if defocus != 0.0:
        _, _, i_focus = compute(
            pitch, duty, polarity, n_periods, dx_target,
            wavelength, na, n_medium, sigma_out, sigma_in, n_src, 0.0,
        )
        ax3.plot(
            sim_grid.x, RS.apply_dose(i_focus, dose),
            lw=1.0, ls="--", alpha=0.7, label="Δz = 0 nm",
        )
    ax3.axhline(threshold, color="crimson", ls=":", lw=1.2, label=f"threshold {threshold}")
    ax3.set_xlabel("x [nm]")
    ax3.set_ylabel("정규화 광강도 I")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3, clear_figure=True)

    c = IM.contrast(intensity, sim_grid.x)
    st.info(
        f"Contrast = {c:.4f} · NILS = {cd_result['nils']:.2f} · "
        f"I_max = {i_max:.3f} / I_min = {i_min:.3f}"
    )
    if defocus != 0.0:
        c0 = IM.contrast(i_focus, sim_grid.x)
        drop = (1 - c / c0) * 100 if c0 > 0 else 0.0
        st.caption(
            f"Δz = 0 대비 contrast {c0:.4f} → {c:.4f} ({drop:+.1f}% 변화). "
            f"고NA DOF 추정값 {rayleigh['dof_high_na']:.0f} nm 와 비교하세요."
            if rayleigh["dof_high_na"] else ""
        )

with tab3:
    profile = RS.resist_profile(intensity_dosed, threshold, tone)
    fig4, ax4 = plt.subplots(figsize=(9, 2.0))
    ax4.imshow(
        MK.to_2d(profile, 120), cmap="gray", aspect="auto",
        extent=[0, sim_grid.field_width, 0, sim_grid.field_width / 4],
    )
    ax4.set_xlabel("x [nm]")
    ax4.set_ylabel("y [nm]")
    fig4.tight_layout()
    st.pyplot(fig4, clear_figure=True)

    if cd_result["formed"]:
        st.success(
            f"중앙 feature CD = {cd_result['cd']:.2f} nm "
            f"(마스크 설계 {cd_nm:.1f} nm, 오차 {cd_result['cd'] - cd_nm:+.2f} nm) · "
            f"edge = [{cd_result['left_edge']:.1f}, {cd_result['right_edge']:.1f}] nm"
        )
    else:
        st.error(f"패턴 미형성 — {failure_reason()}")
        st.caption(
            f"현재 광강도 범위 {i_min:.3f} ~ {i_max:.3f}, 임계값 {threshold:.2f}. "
            "임계값을 이 범위 안으로 옮기거나 Defocus 를 줄이세요."
        )

with tab5:
    st.caption(
        "Dose 는 정규화 광강도에 곱하는 스칼라이므로 결상 연산은 Defocus 에만 의존합니다. "
        f"{n_dose} x {n_focus} 격자를 {n_focus} 회 연산으로 처리합니다 "
        f"({100 * (1 - n_focus / (n_dose * n_focus)):.1f}% 절감)."
    )
    if st.button("Dose-Focus 스윕 실행", type="primary"):
        st.session_state["sweep_done"] = True

    if st.session_state.get("sweep_done"):
        res = sweep(
            pitch, duty, polarity, n_periods, dx_target,
            wavelength, na, n_medium, sigma_out, sigma_in, n_src,
            dz_range, n_focus, n_dose, dose_span, threshold, cd_tol, optx,
        )
        if res is None:
            st.error("현재 조건에서 목표 CD 를 만드는 Dose 가 없습니다. Pitch 또는 조명 조건을 완화하세요.")
        else:
            # DOF 는 단일 숫자가 아니라 **구간**으로 읽는다. 격자화 편향이
            # 초점과 노광 **양쪽**에서 오므로 하한만 적으면 참값이 0 으로
            # 보일 수 있다(12차 보완 65 번. 해석 측정 : 참 DOF 112.7 nm 인
            # 조건에서 33x33 격자의 하한이 0.0 이었다). k2 도 하한에서
            # 역산했으므로 하한이다.
            d5, d5_up = PR.dof_bracket_at_el(res["curve"], el_target)
            k2m = PR.k2_from_dof(d5, wavelength, na, n_medium)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("dose-to-size E_op", f"{res['e_op']:.4f}")
            m2.metric(f"DOF @ EL {el_target:.0f}% (하한)", f"{d5:.0f} nm",
                      delta=f"상한 {d5_up:.0f} nm", delta_color="off")
            m3.metric("역산 k2 (하한)", f"{k2m:.3f}",
                      delta=f"{k2m - k2:+.3f} vs 입력")
            m4.metric("PW 격자점", f"{int(res['pw'].sum())} / {res['pw'].size}")

            st.caption(
                f"DOF 는 격자화 때문에 항상 작게 나오므로 구간 "
                f"[{d5:.0f}, {d5_up:.0f}] nm 로 읽습니다. 구간은 초점 격자와 "
                f"노광 격자 **양쪽**의 편향을 담습니다. 격자를 촘촘히 하면 "
                f"구간이 좁아집니다. 역산 k2 는 하한에서 나온 값이라 역시 "
                f"하한입니다."
            )
            if d5 <= 0.0 < d5_up:
                st.warning(
                    f"**격자가 판정하지 못했습니다.** 하한 0, 상한 "
                    f"{d5_up:.0f} nm 입니다. 윈도가 없다는 뜻이 아니라, 지금 "
                    f"격자로는 EL {el_target:.0f} % 에 닿는지 말할 수 없다는 "
                    f"뜻입니다. Dose 점 수를 늘려 EL 분해능을 좁히세요 "
                    # dose_span 은 **반폭**(±비율)이므로 격자 간격은
                    # 2*dose_span/(n_dose-1) 이고, EL 분해능은 그 2 배다.
                    f"(현재 약 {400.0 * dose_span / max(n_dose - 1, 1):.2f} %)."
                )

            if res.get("dz_note"):
                st.info(res["dz_note"])

            if res["sat"]["focus_edge"] or res["sat"]["dose_edge"]:
                st.warning(
                    "공정 윈도가 스윕 경계에 닿았습니다. DOF 또는 EL 이 과소평가됩니다. "
                    "스윕 범위를 넓히세요."
                )

            # Bossung curve
            figb, axb = plt.subplots(figsize=(9, 3.4))
            step = max(1, n_dose // 7)
            for i in range(0, n_dose, step):
                axb.plot(res["dz"], res["cd"][i], lw=1.2,
                         label=f"Dose {res['doses'][i] / res['e_op']:.2f} E_op")
            axb.axhline(res["target"], color="k", ls="-", lw=1.0)
            axb.axhline(res["target"] * (1 + cd_tol), color="crimson", ls=":", lw=1.0)
            axb.axhline(res["target"] * (1 - cd_tol), color="crimson", ls=":", lw=1.0)
            axb.set_xlabel("Defocus [nm]")
            axb.set_ylabel("CD [nm]")
            axb.set_title("Bossung curve")
            axb.legend(fontsize=7, ncol=2)
            axb.grid(alpha=0.3)
            figb.tight_layout()
            st.pyplot(figb, clear_figure=True)

            colA, colB = st.columns(2)
            with colA:
                figp, axp = plt.subplots(figsize=(4.6, 3.6))
                axp.imshow(
                    res["pw"], origin="lower", aspect="auto", cmap="Greens",
                    extent=[res["dz"][0], res["dz"][-1],
                            res["doses"][0] / res["e_op"], res["doses"][-1] / res["e_op"]],
                )
                axp.set_xlabel("Defocus [nm]")
                axp.set_ylabel("Dose [E_op 대비]")
                axp.set_title(f"Process Window (CD ±{cd_tol * 100:.0f}%)")
                figp.tight_layout()
                st.pyplot(figp, clear_figure=True)
            with colB:
                fige, axe = plt.subplots(figsize=(4.6, 3.6))
                axe.plot([r["el"] for r in res["curve"]],
                         [r["dof"] for r in res["curve"]], "o-", lw=1.4, ms=3)
                axe.axvline(el_target, color="crimson", ls=":", lw=1.0)
                axe.set_xlabel("Exposure Latitude [%]")
                axe.set_ylabel("DOF [nm]")
                axe.set_title("EL - DOF trade-off")
                axe.grid(alpha=0.3)
                fige.tight_layout()
                st.pyplot(fige, clear_figure=True)

            st.info(
                f"측정 DOF {d5:.0f} nm 로부터 역산한 k2 = {k2m:.3f} 입니다. "
                f"해석식 기준값(입력 k2 = {k2:.3f})과의 차이는 Rayleigh 상수가 "
                "조명 조건 및 k1 에 의존하기 때문입니다."
            )
    else:
        st.info("버튼을 눌러 스윕을 실행하세요. 현재 설정 기준 수 초가 소요됩니다.")

with tab6:
    st.caption(
        "레지스트에 입사하는 광자 수는 포아송 분포를 따릅니다. "
        "이 광자 산탄 잡음(photon shot noise)이 LER 과 확률적 결함의 근본 원인입니다."
    )

    st.subheader("파장별 광자 통계 (현재 Dose 기준, 2×2 nm 영역)")
    rows = []
    for lbl, lam_ in (("ArF 193 nm", 193.0), ("KrF 248 nm", 248.0), ("EUV 13.5 nm", 13.5)):
        stt = ST.photon_statistics(dose_mj, lam_, area_nm2=4.0)
        rows.append((lbl, stt))
    st.dataframe(
        {
            "광원": [r[0] for r in rows],
            "광자 에너지 [eV]": [f"{r[1]['photon_energy_ev']:.2f}" for r in rows],
            "광자 밀도 [/nm²]": [f"{r[1]['density_per_nm2']:.1f}" for r in rows],
            "2×2 nm 광자 수": [f"{r[1]['n_in_area']:.1f}" for r in rows],
            "상대 잡음 [%]": [f"{100 * r[1]['relative_noise']:.2f}" for r in rows],
        },
        width="stretch",
    )
    cur = ST.photon_statistics(dose_mj, wavelength, area_nm2=4.0)
    st.info(
        f"현재 파장 λ = {wavelength:.1f} nm · 광자 에너지 {cur['photon_energy_ev']:.2f} eV · "
        f"2×2 nm 영역 광자 수 {cur['n_in_area']:.1f} · 상대 잡음 {100 * cur['relative_noise']:.2f} %"
    )

    if st.button("확률 모델 실행", type="primary"):
        st.session_state["stoch_done"] = True

    if st.session_state.get("stoch_done"):
        sr = stochastic_run(
            pitch, duty, polarity, n_periods, dx_target,
            wavelength, na, n_medium, sigma_out, sigma_in, n_src,
            dose_mj, blur_nm, threshold, y_length, n_trials, defocus, optx,
        )
        if sr is None:
            st.error("현재 조건에서 목표 CD 를 만드는 Dose 가 없습니다. Pitch 또는 조명 조건을 완화하세요.")
        else:
            mc = sr["mc"]
            gg = sr["g"]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("CD 평균", f"{mc['cd_mean']:.2f} nm" if mc["formed"] else "미형성",
                      delta=f"{mc['cd_mean'] - cd_nm:+.2f} nm" if mc["formed"] else None)
            q2.metric("CD 3σ", f"{mc['cd_3sigma']:.2f} nm" if mc["formed"] else "—")
            q3.metric("LER 3σ", f"{mc['ler_3sigma']:.2f} nm" if mc["formed"] else "—")
            q4.metric("미형성 행 비율", f"{mc['defect_rate']:.4f}",
                      help="중앙 feature 의 경계를 찾지 못한 행의 비율입니다. "
                           "확률적 결함 발생률이 아닙니다. 직접 표본화의 "
                           "분해 한계는 1/(행수×반복) 이므로, 공정에서 문제가 "
                           "되는 1e-9~1e-12 수준의 결함률은 이 방식으로 측정할 "
                           "수 없습니다.")
            st.caption(
                f"y 측정 길이 {mc['y_length_nm']:.0f} nm (행 {sr['n_rows']}, "
                f"dy {sr['dy']:.2f} nm) · 총 경계 표본 {mc['n_samples']:,} 개. "
                "LER 은 반복 횟수가 아니라 y 측정 길이에 수렴합니다."
            )

            st.subheader("결정론 모델 대 확률 모델")
            figs, axs = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
            det2d = MK.to_2d(RS.resist_profile(sr["I_det"], threshold, sr["tone"]), 120)
            axs[0].imshow(det2d, cmap="gray", aspect="auto",
                          extent=[0, gg.field_width, 0, gg.field_width / 6])
            axs[0].set_ylabel("y [nm]")
            axs[0].set_title("결정론 (CTR)", fontsize=9)
            sto2d = (sr["field"] >= threshold).astype(float)
            axs[1].imshow(sto2d, cmap="gray", aspect="auto",
                          extent=[0, gg.field_width, 0, gg.field_width / 6])
            axs[1].set_xlabel("x [nm]")
            axs[1].set_ylabel("y [nm]")
            axs[1].set_title(f"확률 (광자 잡음 + 블러 {blur_nm:.0f} nm)", fontsize=9)
            figs.tight_layout()
            st.pyplot(figs, clear_figure=True)

            colX, colY = st.columns(2)
            with colX:
                figp, axp = plt.subplots(figsize=(4.6, 3.2))
                axp.plot(gg.x, sr["I_det"], lw=1.6, label="결정론")
                axp.plot(gg.x, sr["field"].mean(axis=0), lw=1.0, alpha=0.8, label="확률 (행 평균)")
                axp.plot(gg.x, sr["field"][0], lw=0.7, alpha=0.6, label="확률 (단일 행)")
                axp.axhline(threshold, color="crimson", ls=":", lw=1.0)
                axp.set_xlabel("x [nm]"); axp.set_ylabel("정규화 광강도")
                axp.legend(fontsize=7); axp.grid(alpha=0.3)
                figp.tight_layout()
                st.pyplot(figp, clear_figure=True)
            with colY:
                figd, axd = plt.subplots(figsize=(4.6, 3.2))
                doses_scan = np.linspace(max(2.0, dose_mj * 0.2), dose_mj * 2.0, 6)
                lers = []
                for d_ in doses_scan:
                    r_ = ST.monte_carlo(gg.x, sr["I_det"], gg.dx, float(d_), wavelength,
                                        threshold, n_rows=sr["n_rows"],
                                        blur_nm=blur_nm, n_trials=4, seed=2,
                                        dy=sr["dy"], max_width=pitch)
                    lers.append(r_["ler_3sigma"] if r_["formed"] else np.nan)
                axd.plot(doses_scan, lers, "o-", lw=1.4, ms=4)
                axd.axvline(dose_mj, color="crimson", ls=":", lw=1.0)
                axd.set_xlabel("Dose [mJ/cm²]"); axd.set_ylabel("LER 3σ [nm]")
                axd.set_title("Dose 에 따른 LER", fontsize=9)
                axd.grid(alpha=0.3)
                figd.tight_layout()
                st.pyplot(figd, clear_figure=True)

            st.caption(
                "본 모델은 광자 통계만 반영하며 레지스트 내부 화학 잡음은 포함하지 않으므로, "
                "산출 LER 은 하한에 해당합니다. Dose 를 높이면 LER 이 감소하나 처리량이 저하됩니다."
            )
    else:
        st.info("버튼을 눌러 확률 모델을 실행하세요.")


with tab8:
    st.markdown("""
#### 이 조건에서 공정 조건을 되짚을 수 있는가

찍힌 패턴을 보고 공정 조건을 역으로 알아낼 때의 **원리적 한계**를 계산합니다.
광자 수가 포아송 분포를 따른다는 사실로부터 Fisher 정보를 구하고,
Cramér-Rao 하한을 산출합니다. **어떤 추정 방법을 쓰더라도 이보다 정확할 수
없습니다.**
""")
    id_params = st.multiselect(
        "미지수로 둘 공정 변수", list(ID.PARAM_STEPS),
        default=list(ID.DEFAULT_PARAMS),
        help="따로 측정해서 아는 변수는 빼십시오. 뺄수록 한계가 내려갑니다.",
    )
    if st.button("식별 가능성 분석 실행", type="primary"):
        st.session_state["id_done"] = True

    if st.session_state.get("id_done") and id_params:
        idsp = ID.IdentifySpec(
            wavelength=wavelength, na=na, n_medium=n_medium, n_resist=n_resist,
            sigma_out=sigma_out, sigma_in=sigma_in, polarization=polarization,
            source_shape=source_shape, pole_offset=pole_offset,
            pole_radius=pole_radius, mask_type=mask_type, att_t=att_t,
            pitch=pitch, duty=duty, defocus=defocus, dose_rel=dose,
            blur_nm=blur_nm, flare=flare, msd_nm=msd_nm, dose_mj=dose_mj,
            y_length=float(y_length), dy=max(0.25, blur_nm / 3.0) if blur_nm > 0 else 0.5,
            n_periods=n_periods, n_source_grid=n_src, dx_target=dx_target,
        )
        try:
            with st.spinner("Fisher 정보 계산 중..."):
                idres = ID.crb(idsp, tuple(id_params))
                col = ID.collinearity(idsp, tuple(id_params))
        except Exception as exc:
            st.error(f"계산에 실패했습니다: {exc}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("실용 식별 가능", f"{idres['n_identifiable']} / {idres['n_params']}")
            c2.metric("수치 자유도", f"{idres['rank']} / {idres['n_params']}")
            c3.metric("FIM 조건수", f"{idres['cond']:.1e}")

            st.markdown("##### 변수별 하한")
            rows = []
            for p, sg, okp in zip(id_params, idres["sigma_min"], idres["identifiable"]):
                rows.append({
                    "변수": p,
                    "σ_min (하한)": ("식별 불가" if not np.isfinite(sg)
                                   else f"{sg:.4g}"),
                    "실용 기준": ID.PRACTICAL_SCALE.get(p, float("nan")),
                    "판정": "가능" if okp else "불가",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)

            st.markdown("##### 민감도 벡터 사이각")
            st.caption("사이각이 작을수록 두 변수가 관측량에 같은 방식으로 "
                       "나타나 구별되지 않습니다. 이것이 식별 한계의 원인입니다.")
            st.dataframe(
                [{"변수 A": r["a"], "변수 B": r["b"],
                  "코사인": round(r["cos"], 4),
                  "사이각 [도]": round(r["angle_deg"], 2)} for r in col["pairs"]],
                hide_index=True, use_container_width=True,
            )

            fig, ax = plt.subplots(figsize=(7, 2.6))
            w = idres["eigvals"]
            ax.semilogy(range(1, len(w) + 1), np.maximum(w, 1e-20), "o-", lw=1.2)
            ax.axhline(w[0] * len(w) * np.finfo(float).eps, color="r", ls="--", lw=1,
                       label="수치 절단 기준")
            ax.set_xlabel("모드")
            ax.set_ylabel("Fisher 고유값")
            ax.set_xticks(range(1, len(w) + 1))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("기준선 아래로 내려간 모드는 정보가 없는 방향입니다. "
                       "살아남은 모드의 수가 식별 가능한 자유도입니다.")

            lost = sorted(zip(id_params, idres["worst_dir"]),
                          key=lambda t: -abs(t[1]))
            st.info("가장 식별이 안 되는 방향: "
                    + ", ".join(f"{p} {c:+.3f}" for p, c in lost[:3]))

with tab7:
    st.caption(
        "현재 광학·조명 조건을 고정하고 Defocus 와 Dose 를 격자 스윕하여 "
        "머신러닝 학습용 샘플을 생성합니다. 한 건은 (공정 조건, 관측 프로파일, 라벨)로 "
        "구성되며, 관측은 중앙 한 주기를 n_profile 점으로 재샘플링한 값입니다."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        n_dz = st.select_slider("Defocus 분할", [3, 5, 7, 9], value=5)
        dz_max = st.slider("Defocus 범위 ±[nm]", 50.0, 400.0, 200.0, 25.0)
    with d2:
        n_dr = st.select_slider("Dose 분할", [3, 5, 7, 9], value=5)
        dr_span = st.slider("Dose 범위 (±비율)", 0.05, 0.40, 0.20, 0.05)
    with d3:
        n_profile = st.select_slider("n_profile [점]", [16, 32, 64, 128], value=64)
        ds_trials = st.select_slider("샘플당 반복", [2, 4, 8], value=4)
    with st.expander("학습 데이터 품질 기준", expanded=False):
        qa, qb = st.columns(2)
        cd_budget = qa.slider("전체 CD 공차 (목표 CD 대비)", 0.02, 0.30, 0.10, 0.01)
        el_req = qb.slider("요구 노광 여유도 EL [%]", 5.0, 30.0, 15.0, 0.5)
        split = st.slider("확률 변동에 배분하는 분산 비율", 0.1, 0.9, 0.5, 0.05)
        _tmp = DD.SampleSpec(cd_budget=cd_budget, el_required=el_req,
                             budget_split=split)
        st.caption(
            f"CD 공차는 **공정 윈도(결정론)** 와 **확률 변동(CDU)** 이 나눠 "
            f"쓰는 하나의 예산입니다. 두 항목에 각각 전체 공차를 주면 합이 "
            f"공차를 넘으므로, 독립 기여로 보고 제곱합으로 배분합니다 "
            f"(cd_budget² = pw_tol² + stoch_tol²).\n\n"
            f"- 확률 변동 상한 : CD 3σ / CD ≤ **{_tmp.stoch_tol:.3f}**\n"
            f"- 공정 윈도 몫 : **{_tmp.pw_tol:.3f}**\n"
            f"- 이에 따른 NILS 하한 : **{_tmp.min_nils:.2f}** "
            f"(Mack 근사 EL[%] ≈ NILS × CD공차[%] 를 뒤집어 유도. 본 "
            f"시뮬레이터에서 측정한 기울기는 9.13, 상관계수 0.9995)"
        )

    n_total = n_dz * n_dr
    est = DD.estimate_size(n_total, n_profile)
    st.info(
        f"생성 예정 {n_total} 건 · 예상 용량 {est['total_mb']:.3f} MB · "
        f"1건당 {len(DD.CONDITION_KEYS)} 조건 + {n_profile} 프로파일 + "
        f"{len(DD.LABEL_KEYS)} 라벨"
    )

    if st.button("데이터셋 생성", type="primary"):
        specs = []
        for dz in np.linspace(-dz_max, dz_max, n_dz):
            for dr in np.linspace(1 - dr_span, 1 + dr_span, n_dr):
                specs.append(DD.SampleSpec(
                    wavelength=wavelength, na=na, n_medium=n_medium,
                    sigma_out=sigma_out, sigma_in=sigma_in,
                    pitch=pitch, duty=duty, defocus=float(dz), dose_rel=float(dr),
                    dose_mj=dose_mj, blur_nm=blur_nm,
                    dx_target=dx_target, n_source_grid=n_src, threshold=threshold,
                    n_rows=max(64, int(round(y_length / max(0.25, blur_nm / 3.0)))),
                    dy=max(0.25, blur_nm / 3.0),
                    n_trials=ds_trials, n_profile=n_profile,
                    cd_budget=cd_budget, el_required=el_req,
                    budget_split=split,
                ))
        bar = st.progress(0.0, text="생성 중...")
        conds, profs, labs, oks = [], [], [], []
        reject = {"formed": 0, "nils_ok": 0, "cdu_ok": 0, "budget_ok": 0}
        for i, sp in enumerate(specs):
            r = DD.make_sample(sp)
            if r is not None:
                if r["ok"]:
                    conds.append(r["condition"]); profs.append(r["profile"])
                    labs.append(r["labels"]); oks.append(r["ok"])
                else:
                    for k, v in r["quality"].items():
                        if not v:
                            reject[k] += 1
                            break
            bar.progress((i + 1) / len(specs), text=f"생성 중... {i + 1}/{len(specs)}")
        bar.empty()

        if not conds:
            st.error("유효한 샘플이 생성되지 않았습니다. 조건을 완화하세요.")
        else:
            st.session_state["ds"] = {
                "conditions": np.array(conds), "profiles": np.array(profs),
                "labels": np.array(labs), "ok": np.array(oks),
                "reject": reject, "n_total": len(specs),
            }

    if "ds" in st.session_state:
        ds = st.session_state["ds"]
        lab = ds["labels"]
        e1, e2, e3, e4 = st.columns(4)
        n_req = ds.get("n_total", len(ds["conditions"]))
        e1.metric("적격 샘플", f"{len(ds['conditions'])} / {n_req}")
        e2.metric("적격 비율", f"{100 * len(ds['conditions']) / max(n_req, 1):.0f} %")
        idx_cd = DD.LABEL_KEYS.index("cd_mean")
        idx_ler = DD.LABEL_KEYS.index("ler_3sigma")
        e3.metric("CD 범위", f"{np.nanmin(lab[:, idx_cd]):.1f} ~ "
                             f"{np.nanmax(lab[:, idx_cd]):.1f} nm")
        e4.metric("LER 범위", f"{np.nanmin(lab[:, idx_ler]):.3f} ~ "
                              f"{np.nanmax(lab[:, idx_ler]):.3f} nm")

        rj = ds.get("reject", {})
        if sum(rj.values()):
            st.caption(
                f"제외 {sum(rj.values())} 건 : 패턴 미형성 {rj.get('formed', 0)}, "
                f"NILS 미달 {rj.get('nils_ok', 0)}, "
                f"CD 균일도 초과 {rj.get('cdu_ok', 0)}"
            )

        figd1, axd1 = plt.subplots(1, 2, figsize=(9.2, 3.2))
        idx_dz = DD.CONDITION_KEYS.index("defocus")
        idx_dr = DD.CONDITION_KEYS.index("dose_rel")
        sc = axd1[0].scatter(ds["conditions"][:, idx_dz], ds["conditions"][:, idx_dr],
                             c=lab[:, idx_cd], cmap="viridis", s=45)
        axd1[0].set_xlabel("Defocus [nm]"); axd1[0].set_ylabel("Dose [E_op 대비]")
        axd1[0].set_title("조건 공간과 CD 분포", fontsize=9)
        plt.colorbar(sc, ax=axd1[0], label="CD [nm]")
        for row in ds["profiles"][:: max(1, len(ds["profiles"]) // 12)]:
            axd1[1].plot(np.linspace(0, 1, len(row)), row, lw=0.9, alpha=0.7)
        axd1[1].set_xlabel("한 주기 내 위치"); axd1[1].set_ylabel("정규화 광강도")
        axd1[1].set_title("관측 프로파일 표본", fontsize=9)
        axd1[1].grid(alpha=0.3)
        figd1.tight_layout()
        st.pyplot(figd1, clear_figure=True)

        import io
        buf = io.BytesIO()
        np.savez_compressed(
            buf, conditions=ds["conditions"], profiles=ds["profiles"],
            labels=ds["labels"], ok=ds["ok"],
            condition_keys=np.array(DD.CONDITION_KEYS),
            label_keys=np.array(DD.LABEL_KEYS),
        )
        st.download_button("데이터셋 내려받기 (.npz)", buf.getvalue(),
                           file_name="litho_dataset.npz", mime="application/octet-stream")

with tab4:
    st.subheader("회절 차수 통과 판정")
    orders = IM.diffraction_orders(pitch, wavelength, na, sigma_out, max_order=3)
    st.dataframe(
        {
            "차수": [o["order"] for o in orders],
            "공간주파수 [1/nm]": [f"{o['f']:.6f}" for o in orders],
            "on-axis 통과": [o["on_axis"] for o in orders],
            f"σ={sigma_out} 포함": [o["with_sigma"] for o in orders],
        },
        width="stretch",
    )

    st.subheader("샘플링 조건")
    st.write(sampling["message"])
    st.caption(
        "Nyquist 조건 dx ≤ λ / (4·(1+σ)·NA). 마스크 에지는 대역이 무한대이므로 "
        "추가 과샘플링이 필요합니다."
    )

    st.subheader("현재 폰트")
    st.caption(f"matplotlib font.family = {plt.rcParams['font.family']}")
