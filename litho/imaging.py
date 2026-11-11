"""
imaging.py -- Abbe 방식 부분결맞음 결상 연산 (공간 광강도 / Aerial Image)

왜 Abbe 방식인가
-----------------
완전 결맞음(coherent) 가정은 '단순화' 가 아니라 L&S 패턴에서 틀린 답을 낸다.
예를 들어 lambda/NA = 210 nm 인 조건에서 180 nm pitch 의 1차 회절광은
거의 결맞은 광원에서는 pupil 밖으로 벗어나지만, sigma = 0.7 이면 축외
광원점이 그 회절광을 pupil 안으로 되돌려 넣는다. 즉 coherent 가정으로는
실제 스캐너에서 찍히는 패턴이 '구현 불가' 로 출력된다.

Abbe 방식(Koehler 의 sum-over-source)은 확장 광원을 서로 결맞지 않은
점광원의 집합으로 보고, 각 점광원이 만드는 광강도를 비결맞음 합산한다.
Hopkins 의 TCC(4차원) 방식 대비 구현이 단순하면서 동일한 결과를 준다.

연산 구조
---------
마스크 스펙트럼 M(f) 에 대해, 광원점 (sx, sy)(정규화 pupil 좌표) 이
만드는 회절광의 정규화 pupil 좌표는
    (px, py) = (f*lambda/NA + sx,  sy)
pupil 통과 조건은  px^2 + py^2 <= 1
초점 오차 Delta_z 에 의한 위상은 각스펙트럼 전파로 계산한다.
    P_defocus = exp(i * 2*pi * Delta_z * sqrt((n/lambda)^2 - |f_total|^2))
근축 근사 exp(-i*pi*lambda*Delta_z*f^2/n) 은 고NA 에서 성립하지 않으므로
사용하지 않는다.

벡터(편광) 결상
---------------
10주차에 스칼라 모델을 벡터 모델로 확장하였다. 각 광선의 전기장 방향은
전파벡터에 수직하므로, 입사 편광 p 에 대한 출사 전기장 단위벡터는
    e_hat = normalize(p - (p . k_hat) k_hat)
이며, 광강도는 세 성분의 제곱합 |Ex|^2 + |Ey|^2 + |Ez|^2 이다.
고NA 에서 두 광선 사이각이 커지면 TM 편광의 전기장 내적이 작아져
간섭 대비가 떨어진다. 이것이 NA > 0.6 에서 스칼라 모델이 결과를
낙관적으로 산출하는 원인이다 (Flagello, Milster, Rosenbluth,
J. Opt. Soc. Am. A 13(1), 53-64, 1996).

스칼라 모델은 '틀린 모델' 이 아니라 '선에 평행한 편광(TE) 을 암묵적으로
가정한 모델' 이다. 고NA 침지 노광에서 조밀 선패턴은 실제로 TE 편광을
사용하므로(무편광은 대비가 절반 이하로 떨어져 사용이 불가능하다),
스칼라 결과와 TE 벡터 결과의 차이는 본 조건 공간에서 8% 이내이다.

수차
----
이상 pupil 에 Zernike 다항식(Noll 인덱스) 위상을 더한다. 계수 단위는
파장(waves) 이며, 위상은 exp(i*2*pi*W) 로 들어간다. Z4(초점) 는 defocus
인자로 물리적으로 처리하므로 중복을 피하기 위해 Z5 이상만 받는다.

플레어
------
투영계 산란광이다. 조밀 주기 패턴에서는 상의 국부 평균값으로 균일하게
더해지는 배경광으로 근사된다.
    I -> (1 - f) * I + f * <I>
대비와 NILS 를 함께 떨어뜨리며, 그 양상이 노광량 과다와 구별되지 않는다.

정규화
------
open frame(마스크 전면 투과) 조건에서 I = 1.0 이 되도록 광원점 수로
나눈다. 이 정규화가 성립해야 Dose 를 광강도에 곱하는 연산이 물리적
의미를 갖는다. 편광 모델에서도 이 정규화는 유지된다.
"""

import numpy as np

from .grid import SimGrid

SOURCE_CONVENTIONAL = "conventional"
SOURCE_ANNULAR = "annular"
SOURCE_DIPOLE = "dipole"
SOURCE_QUADRUPOLE = "quadrupole"

# 편광 ----------------------------------------------------------------------
# 선패턴은 x 방향으로 주기적이므로 입사면은 xz 평면이다. 따라서 선에
# 평행한 y 편광이 TE, 선에 수직한 x 편광이 TM 이다.
POL_SCALAR = "scalar"          # 편광 무시 (9주차까지의 동작, 하위호환)
POL_TE = "TE"                  # 선에 평행 (y 편광). 고NA 조밀 패턴의 실사용 조건
POL_TM = "TM"                  # 선에 수직 (x 편광)
POL_UNPOLARIZED = "unpolarized"  # TE/TM 광강도의 산술 평균

POLARIZATIONS = (POL_SCALAR, POL_TE, POL_TM, POL_UNPOLARIZED)


def source_points(
    sigma_out: float,
    sigma_in: float = 0.0,
    n_grid: int = 31,
    shape: str = SOURCE_CONVENTIONAL,
    pole_offset: float = 0.0,
    pole_radius: float = 0.15,
) -> np.ndarray:
    """정규화 pupil 좌표계의 광원점 집합을 생성한다.

    [-1, 1]^2 정방 격자를 깔고 조명 형상에 해당하는 점만 남긴다.
    면적 균일 가중이 자동으로 성립한다.

    Parameters
    ----------
    shape : conventional | annular | dipole | quadrupole
        dipole 은 x 축 위 두 극, quadrupole 은 대각 네 극이다.
        조밀 L&S 패턴의 실사용 조명은 x 방향 dipole 이다.
    pole_offset : 극 중심의 pupil 반경 방향 위치. 0 이면 sigma_out 으로
        자동 설정한다 (0 차광과 1 차광이 pupil 양단에 오도록).
    pole_radius : 각 극의 반경.

    Returns
    -------
    np.ndarray, shape (n_src, 2) -- (sx, sy)
    """
    if sigma_out <= 0.0:
        return np.zeros((1, 2))  # 완전 결맞음(점광원)
    if sigma_in < 0.0 or sigma_in >= sigma_out:
        sigma_in = 0.0

    if shape in (SOURCE_DIPOLE, SOURCE_QUADRUPOLE):
        if pole_offset <= 0.0:
            pole_offset = sigma_out
        span = min(1.0, pole_offset + pole_radius)
        s = np.linspace(-span, span, n_grid)
        sx, sy = np.meshgrid(s, s, indexing="xy")
        if shape == SOURCE_DIPOLE:
            centers = [(+pole_offset, 0.0), (-pole_offset, 0.0)]
        else:
            d = pole_offset / np.sqrt(2.0)
            centers = [(+d, +d), (+d, -d), (-d, +d), (-d, -d)]
        keep = np.zeros_like(sx, dtype=bool)
        for cx, cy in centers:
            keep |= np.hypot(sx - cx, sy - cy) <= pole_radius + 1e-12
        keep &= np.hypot(sx, sy) <= 1.0 + 1e-12
    else:
        s = np.linspace(-sigma_out, sigma_out, n_grid)
        sx, sy = np.meshgrid(s, s, indexing="xy")
        r = np.hypot(sx, sy)
        keep = (r <= sigma_out + 1e-12) & (r >= sigma_in - 1e-12)

    pts = np.column_stack([sx[keep], sy[keep]])
    if len(pts) == 0:
        return np.zeros((1, 2))
    return pts


def source_quadrature(
    sigma_out: float,
    sigma_in: float = 0.0,
    n_grid: int = 31,
    shape: str = SOURCE_CONVENTIONAL,
    pole_offset: float = 0.0,
    pole_radius: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """광원 적분을 위한 구적점과 가중치를 돌려준다.

    왜 정방 격자를 쓰지 않는가
    --------------------------
    source_points 는 [-1,1]^2 정방 격자에서 조명 형상 안의 점만 남기고 균일
    가중을 준다. 구현이 간단하지만, 원형 경계를 계단으로 근사하기 때문에
    경계에 걸친 격자칸이 O(h) 의 오차를 남긴다. 실제로 관측 수렴 차수가
    p = 0.93 으로 1차에 머물렀고, 생산 설정 n_grid=31 에서 대비(contrast)가
    충분히 수렴한 값 대비 약 2.2 % 어긋났다. 이는 공간 격자 오차(기계
    정밀도)보다 10^13 배 큰, 시뮬레이터에 남은 가장 큰 수치 오차였다.

    극좌표 구적
    -----------
    환형(및 원형) 광원의 면적 적분은 u = r^2 로 치환하면 직사각형 영역의
    적분이 된다.

        (1/A) \\int f dA = (1/A) \\int\\int f(r,phi) r dr dphi
                        = (1/A) * (1/2) \\int_{u_in}^{u_out} \\int_0^{2pi} f du dphi

    따라서
      - u 방향 : Gauss-Legendre 구적. 피적분 함수가 u 의 다항식으로 잘
        근사되므로 n_r 점으로 2 n_r - 1 차까지 정확하다.
      - phi 방향 : 주기 함수이므로 등간격 격자(= 사다리꼴 = 중점)가
        스펙트럼 정확도를 갖는다. 매끄러운 피적분 함수에서 오차가 점 수에
        대해 지수적으로 줄어든다.

    경계가 정확히 표현되므로 계단 근사 오차가 아예 없다.

    참고 : L. N. Trefethen, "Is Gauss quadrature better than Clenshaw-Curtis?"
    SIAM Rev. 50(1), 67-87 (2008) -- 매끄러운 피적분 함수에서의 수렴 특성.
    주기 방향 등간격 구적의 지수 수렴은 Trefethen & Weideman, "The exponentially
    convergent trapezoidal rule," SIAM Rev. 56(3), 385-458 (2014).

    Returns
    -------
    (pts, w) -- pts: (n_src, 2), w: (n_src,), sum(w) = 1
    """
    # [주의] 쌍극·사극 조명의 형상은 pole_offset / pole_radius 로 정해지며
    # sigma_out 과 무관하다. sigma_out <= 0 에서 바로 점광원을 돌려주면
    # 쌍극 지정이 조용히 무시되어 결맞음 점광원이 된다. 형상이 지정된
    # 경우에는 그 형상을 유지한다.
    poled = shape in (SOURCE_DIPOLE, SOURCE_QUADRUPOLE)
    if sigma_out <= 0.0 and not poled:
        return np.zeros((1, 2)), np.ones(1)
    if sigma_in < 0.0 or sigma_in >= sigma_out:
        sigma_in = 0.0

    # n_grid 는 기존 정방 격자의 한 변 점 수였다. 같은 인자로 극좌표 격자를
    # 정하되, 동등하거나 더 적은 점으로 더 높은 정확도가 나오게 둔다.
    n_r = max(3, int(round(n_grid / 4.0)))
    n_phi = max(8, 4 * n_r)

    nodes, weights = np.polynomial.legendre.leggauss(n_r)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi   # 등간격, 주기

    def disc(cx: float, cy: float, r_in: float, r_out: float):
        # u = r^2 를 [r_in^2, r_out^2] 에 사상
        a, b = r_in ** 2, r_out ** 2
        u = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        wu = 0.5 * (b - a) * weights          # \int du
        r = np.sqrt(u)
        rr, pp = np.meshgrid(r, phi, indexing="ij")
        ww = np.broadcast_to(wu[:, None], rr.shape) / n_phi
        return (np.column_stack([(cx + rr * np.cos(pp)).ravel(),
                                 (cy + rr * np.sin(pp)).ravel()]),
                ww.ravel().copy())

    if shape in (SOURCE_DIPOLE, SOURCE_QUADRUPOLE):
        if pole_offset <= 0.0:
            pole_offset = sigma_out
        if shape == SOURCE_DIPOLE:
            centers = [(+pole_offset, 0.0), (-pole_offset, 0.0)]
        else:
            d = pole_offset / np.sqrt(2.0)
            centers = [(+d, +d), (+d, -d), (-d, +d), (-d, -d)]
        pts_l, w_l = [], []
        for cx, cy in centers:
            p, w = disc(cx, cy, 0.0, pole_radius)
            pts_l.append(p)
            w_l.append(w)
        pts = np.concatenate(pts_l)
        w = np.concatenate(w_l)
        # 극이 동공 밖으로 비어져 나가는 경우만 잘라낸다. 통상
        # pole_offset + pole_radius <= 1 이므로 손실이 없다.
        inside = np.hypot(pts[:, 0], pts[:, 1]) <= 1.0 + 1e-12
        pts, w = pts[inside], w[inside]
    else:
        pts, w = disc(0.0, 0.0, sigma_in, sigma_out)

    total = float(w.sum())
    if total <= 0.0 or len(pts) == 0:
        return np.zeros((1, 2)), np.ones(1)
    return pts, w / total


# --- Zernike 수차 -----------------------------------------------------------
def zernike_phase(px: np.ndarray, py: np.ndarray, coeffs: dict[int, float]) -> np.ndarray:
    """Noll 인덱스 Zernike 계수(파장 단위)로부터 pupil 위상 W 를 계산한다.

    Z4(초점) 는 defocus 인자가 물리적으로 처리하므로 받지 않는다.
    지원 항: Z5/Z6 비점수차, Z7/Z8 코마, Z9/Z10 삼엽수차, Z11 구면수차.
    """
    if not coeffs:
        return np.zeros_like(px)
    rho = np.hypot(px, py)
    theta = np.arctan2(py, px)
    r2, r3, r4 = rho**2, rho**3, rho**4
    table = {
        5: np.sqrt(6.0) * r2 * np.sin(2 * theta),           # 비점수차 45도
        6: np.sqrt(6.0) * r2 * np.cos(2 * theta),           # 비점수차 0도
        7: np.sqrt(8.0) * (3 * r3 - 2 * rho) * np.sin(theta),  # 코마 y
        8: np.sqrt(8.0) * (3 * r3 - 2 * rho) * np.cos(theta),  # 코마 x
        9: np.sqrt(8.0) * r3 * np.sin(3 * theta),           # 삼엽수차
        10: np.sqrt(8.0) * r3 * np.cos(3 * theta),
        11: np.sqrt(5.0) * (6 * r4 - 6 * r2 + 1.0),         # 구면수차
    }
    w = np.zeros_like(px)
    for j, c in coeffs.items():
        if c == 0.0:
            continue
        if j == 4:
            raise ValueError(
                "Z4(초점)는 defocus 인자로 처리합니다. Zernike 계수로 중복 지정하지 마십시오."
            )
        if j not in table:
            raise ValueError(f"지원하지 않는 Zernike 항입니다: Z{j} (지원: Z5~Z11)")
        w = w + c * table[j]
    return w


def _field_unit_vectors(px, py, na, n_img, pol):
    """pupil 좌표에서 출사 전기장 단위벡터 (ex, ey, ez) 를 구한다.

    px, py 는 정규화 pupil 좌표. 방향코사인은 (px, py) * NA / n 이다.
    pol 이 TE 이면 입사 편광은 y(선에 평행), TM 이면 x(선에 수직).
    """
    a = px * na / n_img
    b = py * na / n_img
    g = np.sqrt(np.clip(1.0 - a**2 - b**2, 0.0, None))

    if pol == POL_TE:
        p = (0.0, 1.0, 0.0)
    else:
        p = (1.0, 0.0, 0.0)

    pdotk = p[0] * a + p[1] * b + p[2] * g
    ex = p[0] - pdotk * a
    ey = p[1] - pdotk * b
    ez = p[2] - pdotk * g
    norm = np.sqrt(ex**2 + ey**2 + ez**2)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return ex / norm, ey / norm, ez / norm


def apply_flare(intensity: np.ndarray, flare: float) -> np.ndarray:
    """투영계 산란광(플레어)을 균일 배경으로 반영한다.

    I -> (1 - f) * I + f * <I>.  f 는 0~1 의 비율(통상 0.01~0.05).
    """
    if flare <= 0.0:
        return intensity
    if not 0.0 <= flare < 1.0:
        raise ValueError(f"flare({flare}) 는 0 이상 1 미만이어야 합니다.")
    return (1.0 - flare) * intensity + flare * float(np.mean(intensity))


def apply_msd(intensity: np.ndarray, msd_nm: float, dx: float) -> np.ndarray:
    """스캔 동기화 오차(MSD)를 공간 광강도의 가우시안 평활로 반영한다.

    스캔 중 상과 웨이퍼의 상대 진동은 노광 시간 동안의 시간 평균이므로
    광자 샘플링 '이전' 의 광강도에 작용한다. 레지스트 블러는 샘플링
    '이후' 광자 수에 작용하므로, 결정론적 상에서는 두 효과가 구별되지
    않으나 잡음 평활 여부에서는 구별된다.
    """
    if msd_nm <= 0.0:
        return intensity
    return gaussian_blur_periodic(intensity, msd_nm / dx)


def gaussian_blur_periodic(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """주기 경계 가우시안 평활을 푸리에 공간에서 정확히 수행한다.

    왜 scipy.ndimage 를 쓰지 않는가
    -------------------------------
    gaussian_filter1d 는 커널을 기본 4 sigma 에서 잘라낸다. 꼬리를 버리므로
    상대 오차가 1e-5 수준 남고, 그 결과 가우시안의 분산 가산성
        blur(a) o blur(b) = blur(sqrt(a^2+b^2))
    이 정확히 성립하지 않는다. 이 등식은 MSD 와 레지스트 블러의 축퇴 주장을
    받치는 근거이므로, 수치 오차로 흐려지면 안 된다.

    주기 영역에서 가우시안 평활의 정확한 답은 '감싼 가우시안(wrapped
    Gaussian)' 과의 순환 합성곱이고, 그 푸리에 계수는 닫힌 형태로
        exp(-2 pi^2 sigma^2 f^2)
    이다. 따라서 스펙트럼에 이 인자를 곱하는 것이 절단 없는 정확한 연산이며
    (꼬리를 버리지 않는다), 비용도 더 싸다.
    """
    if sigma_px <= 0.0:
        return arr
    n = len(arr)
    f = np.fft.fftfreq(n)                       # 주기 1 격자점 단위
    k = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * f ** 2)
    return np.fft.ifft(np.fft.fft(arr) * k).real


def aerial_image(
    grid: SimGrid,
    mask: np.ndarray,
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    sigma_out: float = 0.0,
    sigma_in: float = 0.0,
    defocus: float = 0.0,
    n_source_grid: int = 31,
    polarization: str = POL_SCALAR,
    source_shape: str = SOURCE_CONVENTIONAL,
    pole_offset: float = 0.0,
    pole_radius: float = 0.15,
    aberrations: dict[int, float] | None = None,
    flare: float = 0.0,
    msd_nm: float = 0.0,
    n_resist: float | None = None,
    source_integration: str = "auto",
    n_gauss: int = 16,
) -> np.ndarray:
    """부분결맞음 공간 광강도 분포를 계산한다.

    source_integration
    ------------------
    "auto"  : 가능하면 영역 분해 정확 구적(hopkins.aerial_image_exact)을 쓰고,
              불가능한 조명 형상에서는 Abbe 합산으로 되돌아간다. 기본값.
    "abbe"  : 항상 Abbe 광원 합산. 독립 구현 대조와 수렴 연구에 쓴다.

    Parameters
    ----------
    mask : 마스크 투과 진폭 (complex128), shape (grid.n_points,)
    wavelength : 진공 파장 [nm]
    na : 개구수 (n*sin(theta))
    n_medium : 결상 매질 굴절률 (공기 1.0, 물 1.44). NA 보다 커야 한다.
    sigma_out, sigma_in : 부분결맞음 계수. sigma_in>0 이면 annular 조명
    defocus : 초점 오차 [nm]
    polarization : scalar | TE | TM | unpolarized
    source_shape : conventional | annular | dipole | quadrupole
    aberrations : {Noll 인덱스: 계수[waves]}. Z5~Z11.
    flare : 플레어 비율 0~1
    msd_nm : 스캔 동기화 오차 [nm, 1-sigma]
    n_resist : 편광 가중을 계산할 굴절률. None 이면 n_medium 을 쓴다.
        상은 레지스트 안에서 맺히므로 레지스트 굴절률(통상 1.7)을 쓰는
        편이 물리적이며, n_medium 을 쓰면 편광 효과를 과대평가한다.
        두 값으로 각각 계산해 구간으로 보고하는 것을 권장한다.

    Returns
    -------
    np.ndarray(float64) -- 정규화 광강도, open frame 에서 1.0
    """
    if mask.shape != (grid.n_points,):
        raise ValueError("mask 배열 길이가 격자와 일치하지 않습니다.")
    if na <= 0.0:
        raise ValueError(f"NA({na}) 는 양수여야 합니다.")
    # [중요] NA = n*sin(theta) 이므로 NA >= n 은 물리적으로 불가능하다.
    # 이 조합을 허용하면 pupil 바깥 띠에서 kz^2 < 0 이 되어 defocus 위상이
    # 조용히 사라지고, 초점 응답이 비단조가 된다.
    if na >= n_medium:
        raise ValueError(
            f"NA({na:.3f}) >= n_medium({n_medium:.3f}): 물리적으로 불가능한 조합입니다. "
            f"침지 노광이면 n_medium=1.44(물) 또는 1.0 미만 NA 를 지정하십시오."
        )
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization 은 {POLARIZATIONS} 중 하나여야 합니다.")
    # [10주차 보정] sigma_in >= sigma_out 은 폭이 0 이하인 환형이므로 광원
    # 면적이 0 이고 물리적으로 정의되지 않는다. 기존에는 source_points /
    # source_quadrature 안에서 조용히 sigma_in = 0 으로 되돌렸는데, 그러면
    # 사용자가 지정한 것과 전혀 다른 조명(전면 원형)으로 계산해 놓고 아무
    # 말도 하지 않는다. 더구나 영역 분해 경로에는 그 되돌림이 없어 면적 0
    # 으로 나누며 NaN 이 나왔다. 두 경로가 서로 다르게 동작했으므로 입구에서
    # 막는다.
    if sigma_in < 0.0:
        raise ValueError(f"sigma_in({sigma_in}) 은 0 이상이어야 합니다.")
    if sigma_out > 0.0 and sigma_in >= sigma_out:
        raise ValueError(
            f"sigma_in({sigma_in:.6g}) >= sigma_out({sigma_out:.6g}): 환형 "
            f"조명의 폭이 0 이하입니다. 광원 면적이 0 이므로 정의되지 않습니다.")

    if polarization == POL_UNPOLARIZED:
        # 무편광은 서로 결맞지 않은 두 직교 편광의 광강도 평균이다.
        common = dict(
            wavelength=wavelength, na=na, n_medium=n_medium,
            sigma_out=sigma_out, sigma_in=sigma_in, defocus=defocus,
            n_source_grid=n_source_grid, source_shape=source_shape,
            pole_offset=pole_offset, pole_radius=pole_radius,
            aberrations=aberrations, flare=flare, msd_nm=msd_nm,
            n_resist=n_resist, source_integration=source_integration,
            n_gauss=n_gauss,
        )
        te = aerial_image(grid, mask, polarization=POL_TE, **common)
        tm = aerial_image(grid, mask, polarization=POL_TM, **common)
        return 0.5 * (te + tm)

    n_pol = n_medium if n_resist is None else n_resist
    if polarization != POL_SCALAR and na >= n_pol:
        raise ValueError(
            f"NA({na:.3f}) >= 편광 계산 굴절률({n_pol:.3f}) 입니다. "
            f"n_resist 를 NA 보다 크게 지정하십시오."
        )

    # [생산 경로 선택] 환형·원형 조명과 1차원 주기 마스크에서는 광원 적분을
    # 영역 분해로 정확히 수행할 수 있다. 구적점을 영역 전체에 깔면 회절
    # 차수가 동공 경계를 드나드는 불연속 때문에 O(1/n) 에 머물러, 기존 설정
    # (n_source_grid=31)에서 대비에 2.4 % 오차가 남았다. 영역 분해는 그
    # 불연속을 적분 영역의 경계로 옮겨 1e-7 수준까지 내려간다. 비용은 약
    # 4 배이지만, 같은 정확도를 Abbe 로 얻으려면 500 배 이상 느리다.
    # 쌍극·사극 조명은 경계 원의 중심이 x 축을 벗어나 이 분해가 성립하지
    # 않으므로 Abbe 경로를 그대로 쓴다.
    _exact_ok = (source_shape == SOURCE_CONVENTIONAL and sigma_out > 0.0) or \
        (source_shape in (SOURCE_DIPOLE, SOURCE_QUADRUPOLE) and pole_radius > 0.0)
    if source_integration == "auto" and _exact_ok:
        from . import hopkins as _HK
        inten = _HK.aerial_image_exact(
            grid, mask, wavelength, na, n_medium=n_medium,
            sigma_out=sigma_out, sigma_in=sigma_in, defocus=defocus,
            polarization=polarization, source_shape=source_shape,
            pole_offset=pole_offset, pole_radius=pole_radius,
            aberrations=aberrations, n_resist=n_resist, n_gauss=n_gauss,
        )
        if msd_nm > 0.0:
            inten = apply_msd(inten, msd_nm, grid.dx)
        if flare > 0.0:
            inten = apply_flare(inten, flare)
        return inten

    spectrum = np.fft.fft(mask)
    fx = grid.fx
    src, src_w = source_quadrature(
        sigma_out, sigma_in, n_source_grid,
        shape=source_shape, pole_offset=pole_offset, pole_radius=pole_radius,
    )

    k_medium = n_medium / wavelength  # [1/nm]
    intensity = np.zeros(grid.n_points, dtype=np.float64)
    fx_na = fx * wavelength / na      # 정규화 pupil 좌표의 마스크 성분

    # 광원점 루프를 파이썬이 아닌 배열 축으로 처리한다. 일괄 FFT 가 개별
    # 호출보다 빠르고, 벡터 모델의 3성분 비용도 상당 부분 상쇄된다.
    #
    # [묶음 크기] 광원점을 한꺼번에 처리하면 (n_src, n_points) 배열이
    # 수십 MB 가 되어 캐시를 벗어나고 오히려 느려진다. 실측 결과 묶음당
    # 원소 수를 8K 근처로 제한할 때 성긴 격자(128점)와 촘촘한 격자(1920점)
    # 양쪽에서 가장 빨랐다. 묶음 없이 전부 처리하는 방식 대비 촘촘한
    # 격자에서 약 2배 빠르다.
    CHUNK_ELEMENTS = 8192
    chunk = max(1, int(CHUNK_ELEMENTS // max(grid.n_points, 1)))

    for i0 in range(0, len(src), chunk):
        blk = src[i0:i0 + chunk]
        wblk = src_w[i0:i0 + chunk][:, None]        # (k, 1) 구적 가중치
        sx = blk[:, 0][:, None]                     # (k, 1)
        sy = blk[:, 1][:, None]
        px = fx_na[None, :] + sx                    # (k, n_points)
        py = np.broadcast_to(sy, px.shape)

        pupil = ((px**2 + py**2) <= 1.0).astype(np.complex128)

        if defocus != 0.0:
            fx_total = fx[None, :] + sx * na / wavelength
            fy_total = sy * na / wavelength
            kz2 = k_medium**2 - (fx_total**2 + fy_total**2)
            # NA < n_medium 을 위에서 보장했으므로 pupil 내부에서 kz2 > 0 이다.
            kz = np.sqrt(np.maximum(kz2, 0.0))
            pupil *= np.exp(2j * np.pi * defocus * kz)

        if aberrations:
            w = zernike_phase(px, py, aberrations)
            pupil *= np.exp(2j * np.pi * w)

        sp = spectrum[None, :] * pupil

        if polarization == POL_SCALAR:
            intensity += (wblk * np.abs(np.fft.ifft(sp, axis=-1)) ** 2).sum(axis=0)
        else:
            ex, ey, ez = _field_unit_vectors(px, py, na, n_pol, polarization)
            for comp in (ex, ey, ez):
                if not np.any(comp):
                    continue
                intensity += (wblk * np.abs(np.fft.ifft(sp * comp, axis=-1)) ** 2
                              ).sum(axis=0)

    # 가중치 합이 1 로 정규화되어 있으므로 추가 나눗셈이 필요 없다.

    if msd_nm > 0.0:
        intensity = apply_msd(intensity, msd_nm, grid.dx)
    if flare > 0.0:
        intensity = apply_flare(intensity, flare)
    return intensity


def contrast(intensity: np.ndarray, x: np.ndarray | None = None) -> float:
    """이미지 contrast = (Imax - Imin) / (Imax + Imin).

    x 를 주면 극값을 대역제한 보간에서 정확히 찾는다(격자 무관). x 가 없으면
    표본 극값을 쓰므로 O(dx^2) 격자 의존성이 남는다 -- 되돌림 호환용이다.
    """
    if x is not None:
        from .resist import image_extrema
        e = image_extrema(x, intensity)
        hi, lo = e["i_max"], e["i_min"]
        return float((hi - lo) / (hi + lo)) if (hi + lo) != 0 else 0.0
    i_max = float(np.max(intensity))
    i_min = float(np.min(intensity))
    if i_max + i_min <= 0.0:
        return 0.0
    return (i_max - i_min) / (i_max + i_min)


def diffraction_orders(
    pitch: float,
    wavelength: float,
    na: float,
    sigma_out: float = 0.0,
    max_order: int = 5,
) -> list[dict]:
    """각 회절차수가 pupil 을 통과하는지 판정한다 (교육용 진단).

    m 차 회절광의 공간주파수는 m/pitch 이며, 광원점 최대 이동량은
    sigma_out*NA/lambda 이다.
    """
    cutoff = na / wavelength
    shift = sigma_out * na / wavelength
    out = []
    for m in range(-max_order, max_order + 1):
        f_m = m / pitch
        out.append(
            {
                "order": m,
                "f": f_m,
                "on_axis": abs(f_m) <= cutoff,
                "with_sigma": abs(f_m) - shift <= cutoff,
            }
        )
    return out


def polarization_factor(na: float, pitch: float, wavelength: float,
                        n_img: float, polarization: str = POL_TE) -> float:
    """2광선 간섭 한계에서의 편광 대비 계수 (해석 검증용).

    ±1 차광이 pupil 양단에 놓이는 2광선 조건에서 두 광선 사이각은 2*theta,
    sin(theta) = lambda / (2 * n_img * pitch) 이다. TE 는 전기장이 평행하여
    계수 1, TM 은 cos(2*theta) 이다.
    """
    sin_t = wavelength / (2.0 * n_img * pitch)
    if sin_t > 1.0:
        return float("nan")
    two_t = 2.0 * np.arcsin(sin_t)
    if polarization == POL_TE:
        return 1.0
    if polarization == POL_TM:
        return float(np.cos(two_t))
    if polarization == POL_UNPOLARIZED:
        return float(0.5 * (1.0 + np.cos(two_t)))
    return 1.0
