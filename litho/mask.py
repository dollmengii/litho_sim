"""
mask.py -- 마스크 패턴(Line & Space) 생성

설계 결정
---------
1. 마스크는 '투과 진폭(amplitude)' 으로 저장한다.
   광강도(intensity)로 저장하면 결상 연산(FFT -> pupil -> IFFT) 이
   물리적으로 성립하지 않는다.
2. dtype 은 complex128 로 고정한다.
   현재는 binary mask(투과율 0/1, 실수)이지만, 위상반전마스크(PSM)로
   확장하면 투과율이 복소수(예: -1)가 된다. 나중에 dtype 을 바꾸면
   결상 파이프라인 전체를 수정해야 하므로 처음부터 복소수로 선언한다.
3. feature 중심을 필드 중심(x = L/2)에 맞춘다.
   CD 추출 시 필드 중앙의 feature 를 기준으로 삼기 위함이다.

변수 관계
---------
    duty  = cd / pitch
    space = pitch - cd
    half_pitch = pitch / 2
"""

import numpy as np

from .grid import SimGrid

POLARITY_CLEAR = "clear"  # feature 가 투과 (t=1), 배경 차광
POLARITY_DARK = "dark"    # feature 가 차광 (t=0), 배경 투과

# 마스크 종류 ---------------------------------------------------------------
# binary  : 차광부 투과율 0. 9주차까지의 모델.
# att-psm : 감쇠형 위상반전마스크. 차광부가 소량(통상 6%) 투과하되 위상이
#           180도 뒤집힌다. 패턴 경계에서 상쇄간섭이 일어나 경사가 가팔라지며,
#           현 ArF 공정의 표준 마스크이다. 진폭 투과율은 -sqrt(T) 이다.
MASK_BINARY = "binary"
MASK_ATT_PSM = "att-psm"
MASK_TYPES = (MASK_BINARY, MASK_ATT_PSM)

# 표본화 방식 -------------------------------------------------------------
# area        : 면적 가중(픽셀 적분). 경계 격자칸에 덮인 면적 비율을 진폭에
#               반영한다. 0차는 정확하고 m차는 O(1/N^2) 로 수렴한다.
# bandlimited : 해석적 푸리에 계수로 합성. 격자 Nyquist 이하의 모든 차수가
#               정확하다. 동공 차단주파수가 격자 Nyquist 보다 낮으면(Nyquist
#               격자를 쓰면 항상 성립) 결상에 쓰이는 차수가 전부 정확해져
#               마스크 이산화 오차가 완전히 사라진다. 기본값.
SAMPLING_AREA = "area"
SAMPLING_BANDLIMITED = "bandlimited"
SAMPLING_MODES = (SAMPLING_AREA, SAMPLING_BANDLIMITED)

ATT_PSM_DEFAULT_T = 0.06   # 6% 감쇠형. 업계 표준값


def background_amplitude(mask_type: str, att_t: float = ATT_PSM_DEFAULT_T) -> complex:
    """차광부(배경)의 투과 진폭을 돌려준다.

    att-psm 은 세기 투과율 att_t, 위상 180도이므로 진폭은 -sqrt(att_t) 이다.
    """
    if mask_type == MASK_BINARY:
        return 0.0 + 0.0j
    if mask_type == MASK_ATT_PSM:
        if not 0.0 < att_t < 1.0:
            raise ValueError(f"att_t({att_t}) 는 0 과 1 사이여야 합니다.")
        return complex(-np.sqrt(att_t), 0.0)
    raise ValueError(f"mask_type 은 {MASK_TYPES} 중 하나여야 합니다.")


def line_space(
    grid: SimGrid,
    cd: float | None = None,
    duty: float = 0.5,
    polarity: str = POLARITY_CLEAR,
    mask_type: str = MASK_BINARY,
    att_t: float = ATT_PSM_DEFAULT_T,
    sampling: str = SAMPLING_BANDLIMITED,
) -> np.ndarray:
    """1D Line & Space 마스크 투과 진폭 배열을 생성한다.

    Parameters
    ----------
    cd : feature 폭 [nm]. None 이면 duty * pitch 로 결정.
    duty : cd / pitch. cd 가 주어지면 무시된다.
    polarity : "clear" 또는 "dark".

    Returns
    -------
    np.ndarray(complex128), shape (grid.n_points,)
    """
    pitch = grid.pitch
    if cd is None:
        cd = duty * pitch
    if not (0.0 < cd < pitch):
        raise ValueError(
            f"cd({cd:.2f} nm) 는 0 보다 크고 pitch({pitch:.2f} nm) 보다 작아야 합니다."
        )
    if polarity not in (POLARITY_CLEAR, POLARITY_DARK):
        raise ValueError(f"polarity 는 '{POLARITY_CLEAR}' 또는 '{POLARITY_DARK}' 입니다.")

    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling 은 {SAMPLING_MODES} 중 하나여야 합니다.")

    bg = background_amplitude(mask_type, att_t)
    a_f, a_b = (1.0 + 0.0j, bg) if polarity == POLARITY_CLEAR else (bg, 1.0 + 0.0j)

    if sampling == SAMPLING_BANDLIMITED:
        t = _bandlimited(grid, cd, a_f, a_b)
    else:
        frac = _coverage(grid, cd)
        t = a_b + (a_f - a_b) * frac

    return t.astype(np.complex128)


def _bandlimited(grid: SimGrid, cd: float,
                 a_f: complex, a_b: complex) -> np.ndarray:
    """해석적 푸리에 계수로 마스크를 합성한다.

    왜 이것이 가장 정확한가
    -----------------------
    부분결맞음 결상은 동공을 통과한 회절차수만 쓴다. 통과하는 차수는
        |m| <= pitch * NA (1 + sigma) / lambda
    로 한정되고, Nyquist 조건을 만족하는 격자에서는 이 상한이 격자 Nyquist
        |m| <= N_points / (2 * n_periods)
    보다 낮다. 따라서 격자 Nyquist 이하의 차수를 해석값으로 정확히 넣어주면
    결상에 실제로 쓰이는 차수가 **모두 정확**해지고, 마스크 이산화에서
    비롯된 오차가 결상 결과에서 완전히 사라진다. 면적 가중 방식은 상자함수
    전달함수 sinc(m/N) 과 되풀이(aliasing) 때문에 O(1/N^2) 의 오차가 남는다.

    해석 계수
    ---------
    투과 진폭은 feature 지시함수 ind(x) 에 대해
        t(x) = a_b + (a_f - a_b) ind(x)
    이고, 중심이 x0 인 폭 cd, 주기 pitch 의 사각 펄스열은
        ind(x) = duty + sum_{m != 0} duty sinc(m duty) exp(i 2 pi m (x - x0)/pitch)
    이다(sinc(z) = sin(pi z)/(pi z)). x 공간에서 Gibbs 물결이 보이는 것은
    정상이다. 이것이 유한 격자가 표현할 수 있는 유일한 사각파이며, 동공이
    같은 대역으로 잘라내므로 공간상의 물결은 결상 결과에 나타나지 않는다.
    """
    pitch = grid.pitch
    duty = cd / pitch
    x0 = grid.field_width / 2.0

    # 격자가 표현할 수 있는 최대 차수 (Nyquist)
    m_max = int(grid.samples_per_pitch // 2)
    phi = 2.0 * np.pi * (grid.x - x0) / pitch

    ind = np.full(grid.n_points, duty, dtype=np.complex128)
    for m in range(1, m_max + 1):
        c = duty * np.sinc(m * duty)          # np.sinc(z) = sin(pi z)/(pi z)
        if c == 0.0:
            continue
        # +m 과 -m 을 합치면 2 c cos(m phi). Nyquist 차수는 한 번만 센다.
        w = 1.0 if (m == m_max and grid.samples_per_pitch % 2 == 0) else 2.0
        ind += w * c * np.cos(m * phi)
    return a_b + (a_f - a_b) * ind


def _coverage(grid: SimGrid, cd: float) -> np.ndarray:
    """각 표본 격자칸이 feature 에 덮인 면적 비율(0~1)을 돌려준다.

    왜 지시함수(indicator)를 쓰지 않는가 -- 10주차 보정
    -----------------------------------------------------
    9주차까지는 `|u - pitch/2| <= cd/2` 로 격자점이 feature 안에 있는지만
    판정했다. 이 방식은 cd/2 가 격자점에 정확히 떨어질 때 양 끝 표본을 모두
    포함시켜, 주기당 표본 수가 N 일 때 duty 가 1/N 만큼 **한쪽으로만**
    커진다. spp=16, duty 0.5 조건에서 실제 duty 가 0.5625 로 산출되어
    0차 회절광 진폭에 12.5 %, 3차에 11.8 % 오차가 생겼고, 대칭성 때문에
    존재할 수 없는 짝수차 성분(0.0625)까지 나타났다. 격자점을 하나 빼는
    방식으로 바꾸면 부호만 뒤집힌 같은 크기의 계통 오차가 남는다.
    (10주차 보고서에서 '격자 양자화이므로 결함이 아니다'라고 판단한 것은
     잘못이었다. 양자화 오차는 격자를 조이면 무작위로 줄어들지만, 이 오차는
     부호가 고정되어 있어 0차 진폭을 항상 같은 방향으로 틀리게 만든다.)

    면적 가중(픽셀 적분) 방식은 연속 마스크를 폭 dx 의 상자함수와 합성곱한
    뒤 표본화하는 것과 같다. 따라서
        - 0차(= duty)는 격자와 무관하게 **정확**하다.
        - m차 성분은 해석값에 상자함수의 전달함수 sinc(m/N) 이 곱해진 꼴이
          되어 O(1/N^2) 로 수렴하며, 부호가 고정된 편향이 없다.
        - 짝수차는 정확히 0 으로 유지된다.
    이는 마스크 묘화 장비와 상용 리소 해석기가 쓰는 표준 래스터화와 같다.
    """
    dx = grid.dx
    pitch = grid.pitch
    # feature 중심이 필드 중심(x = L/2)에 오도록 좌표를 접고, [-p/2, p/2) 로 옮긴다
    s = np.mod(grid.x - grid.field_width / 2.0 + pitch / 2.0, pitch) - pitch / 2.0
    lo, hi = s - dx / 2.0, s + dx / 2.0

    # 격자칸 폭이 feature 또는 space 보다 넓으면 이웃 주기까지 걸칠 수 있으므로
    # 좌우 한 주기씩 함께 더한다.
    frac = np.zeros_like(s)
    for k in (-1, 0, 1):
        a, b = k * pitch - cd / 2.0, k * pitch + cd / 2.0
        frac += np.maximum(0.0, np.minimum(hi, b) - np.maximum(lo, a))
    return np.clip(frac / dx, 0.0, 1.0)


def open_frame(grid: SimGrid) -> np.ndarray:
    """마스크 없음(전면 투과). 정규화 검증용."""
    return np.ones(grid.n_points, dtype=np.complex128)


def to_2d(profile_1d: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    """y 방향으로 무한히 긴 선 패턴을 2D 배열로 확장한다.

    y 불변 패턴이므로 1D 결과를 복제하는 것이 정확하며 연산 비용이 없다.
    """
    if n_rows is None:
        n_rows = len(profile_1d)
    return np.tile(profile_1d, (n_rows, 1))


def add_sraf(
    mask: np.ndarray,
    grid: SimGrid,
    width: float,
    offset: float,
    n_pairs: int = 1,
    mask_type: str = MASK_BINARY,
    att_t: float = ATT_PSM_DEFAULT_T,
) -> np.ndarray:
    """주 패턴 양옆에 해상되지 않는 보조 패턴(SRAF)을 추가한다.

    SRAF 는 그 자체로는 레지스트에 찍히지 않을 만큼 가늘지만, 주 패턴의
    회절 분포를 바꾸어 초점심도를 넓힌다. 폭이 해상 한계를 넘으면 보조
    패턴이 함께 찍히므로(side-lobe printing) width 는 충분히 작아야 한다.

    Parameters
    ----------
    width : 보조 패턴 폭 [nm]
    offset : 주 패턴 중심에서 보조 패턴 중심까지 거리 [nm]
    n_pairs : 좌우 쌍의 개수
    """
    if width <= 0 or offset <= 0:
        return mask
    centers = []
    for k in range(1, n_pairs + 1):
        centers += [grid.field_width / 2.0 + k * offset,
                    grid.field_width / 2.0 - k * offset]

    # [10주차 보정] 주 패턴을 해석 푸리에 계수로 합성하므로 SRAF 도 같은
    # 방식이어야 한다. 한쪽만 면적 가중으로 두면 표본화 방식이 섞여, 같은
    # 마스크 안에서 차수 정확도가 서로 다른 두 성분이 공존한다.
    #
    # [중요] 합성곱이 아니라 '더하기' 로 올려야 한다. 투과 진폭은
    #     t(x) = a_b + (a_f - a_b) * [ind_main(x) + ind_sraf(x)]
    # 이므로 지시함수들이 겹치지 않는 한 선형 중첩이 정확하다. 반면
    # out * (1 - frac) + frac 형태의 혼합은 두 대역제한 함수의 '곱' 이므로
    # 대역이 두 배로 넓어지고 Nyquist 를 넘는 성분이 되돌아 접힌다.
    bg = background_amplitude(mask_type, att_t)
    amp = (1.0 + 0.0j) - bg

    ind = _bandlimited_rects(grid, centers, width)
    # 겹침 검사 : SRAF 는 정의상 주 패턴 사이의 차광부에 놓인다. 겹치면
    # 선형 중첩이 성립하지 않으므로 알려 준다.
    occupied = np.abs(mask - bg) > 0.5 * abs(amp) if amp != 0 else None
    if occupied is not None and np.any(occupied & (ind > 0.5)):
        import warnings
        warnings.warn(
            "SRAF 가 주 패턴과 겹칩니다. offset 을 (cd + width)/2 보다 크게 "
            "지정하십시오. 겹친 영역에서는 투과 진폭이 1 을 넘어 물리적으로 "
            "무의미해집니다.", RuntimeWarning, stacklevel=2)
    return (mask + amp * ind).astype(np.complex128)


def _bandlimited_rects(grid: SimGrid, centers, width: float) -> np.ndarray:
    """필드 주기 L 안의 사각 펄스 여러 개를 해석 계수로 합성한다.

    중심 c, 폭 w 인 사각 펄스의 주기 L 푸리에 계수는
        A_0 = w/L,   A_m = (w/L) sinc(m w/L) exp(-i 2 pi m c / L)
    이다. 격자 Nyquist 이하의 차수를 모두 넣으면, 그 격자가 표현할 수 있는
    가장 정확한 표현이 된다. 폭이 격자 간격보다 좁아도(SRAF 가 흔히 그렇다)
    0차가 정확하므로 유효 폭이 설계값과 어긋나지 않는다.
    """
    L = grid.field_width
    m_max = int(grid.n_points // 2)
    frac_w = width / L
    ind = np.zeros(grid.n_points, dtype=np.float64)
    m = np.arange(1, m_max + 1)
    coef = frac_w * np.sinc(m * frac_w)
    w_last = 0.5 if grid.n_points % 2 == 0 else 1.0
    wts = np.ones_like(coef)
    wts[-1] = w_last
    for c in centers:
        ph = 2.0 * np.pi * np.outer(m, (grid.x - c) / L)   # (m_max, n_points)
        ind += frac_w + 2.0 * (wts * coef) @ np.cos(ph)
    return ind


def mask_stats(mask: np.ndarray, grid: SimGrid, cd: float,
               mask_type: str = MASK_BINARY,
               att_t: float = ATT_PSM_DEFAULT_T) -> dict:
    """생성된 배열의 duty 가 설계값과 일치하는지 검증한다.

    duty 는 **평균 진폭**(= 0차 회절광 계수)에서 환산한다. 평균 세기를
    쓰면 안 된다. 면적 가중 표본화에서 경계 격자칸은 0 과 1 사이의 진폭을
    가지므로, <|t|^2> 에는 경계칸의 2차 항이 섞여 들어가 duty 가 과소
    평가된다(진폭은 선형, 세기는 제곱이기 때문).
        <t>      = duty * 1 + (1 - duty) * t_bg      <- 격자와 무관하게 정확
        <|t|^2>  = duty * 1 + (1 - duty) * |t_bg|^2  - O(dx/pitch) 경계항
    att-psm 은 차광부 진폭이 -sqrt(att_t) 이므로 그만큼 보정해 준다.
    """
    mean_a = complex(np.mean(mask))
    bg = background_amplitude(mask_type, att_t)
    measured_duty = ((mean_a - bg) / (1.0 - bg)).real if bg != 1.0 else float("nan")
    design_duty = cd / grid.pitch
    return {
        "design_duty": design_duty,
        "measured_duty": measured_duty,
        "mean_amplitude": mean_a.real,
        "mean_intensity": float(np.mean(np.abs(mask) ** 2)),
        "background_amplitude": bg.real,
        "background_intensity": abs(bg) ** 2,
        "error": abs(measured_duty - design_duty),
        "n_lines": grid.n_periods,
    }
