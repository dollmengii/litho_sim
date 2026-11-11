"""
hopkins.py -- Hopkins TCC 방식 결상 (Abbe 방식의 독립 대조용)

왜 두 번 구현하는가
-------------------
기존 검증은 모두 시뮬레이터가 산출한 값을 해석식이나 문헌값과 비교하는
방식이었다. 그러나 해석식 역시 같은 사람이 같은 이해로 작성한 것이므로,
물리 모델 자체를 잘못 이해했다면 시뮬레이터와 기준선이 함께 틀린 채로
통과한다. 이를 막는 표준적인 방법이 독립 구현 대조이다.

Abbe 방식과 Hopkins 방식은 부분결맞음 결상을 기술하는 수학적으로 동등한
두 표현이지만, 계산 경로가 완전히 다르다.

    Abbe    : 광원점마다 상을 만들고 광강도를 비결맞음 합산
              I(x) = (1/N) * sum_s | IFFT{ M(f) P(s+f) } |^2

    Hopkins : 광원을 먼저 적분하여 전달교차계수(TCC)를 만들고,
              마스크 스펙트럼의 모든 차수 쌍에 대해 합산
              TCC(f1,f2) = (1/N) * sum_s P(s+f1) P*(s+f2)
              I(x)       = sum_{f1,f2} TCC(f1,f2) M(f1) M*(f2)
                           * exp(i 2 pi (f1-f2) x)

Abbe 는 광원 축을 바깥 루프에, Hopkins 는 차수 축을 바깥 루프에 둔다.
중간 산출물(한쪽은 각 광원점의 상, 다른 쪽은 4차원 계수 행렬)이 전혀
다르므로, 두 결과가 일치하면 공통된 구현 실수가 있을 가능성이 크게 낮아진다.

편광 처리
---------
벡터 모델에서는 두 차수의 전기장 방향이 다르므로 교차항에 편광 내적이
들어간다.

    TCC_vec(f1,f2) = (1/N) * sum_s P(s+f1) P*(s+f2) ( e(s+f1) . e*(s+f2) )

이 형태로 구현하면 편광 경로까지 함께 대조된다.

비용
----
차수 수 K 에 대해 TCC 는 K x K 이고, 선패턴은 동공 안의 차수가 많아야
수십 개이므로 비용이 작다. 다만 2차원 패턴이나 조밀한 스펙트럼에서는
K 가 급증하여 Abbe 가 유리하다. 본 모듈은 대조 전용이며 생산 경로가
아니다.

참고 : H. H. Hopkins, "On the diffraction theory of optical images,"
Proc. R. Soc. Lond. A 217, 408-432 (1953).
"""

import numpy as np

from . import imaging as IM
from .grid import SimGrid


def _orders(grid: SimGrid, spectrum: np.ndarray, rel_tol: float = 1e-12,
            reach: float | None = None):
    """마스크 스펙트럼에서 유효한 회절 차수만 골라낸다.

    주기 경계와 정수 주기 조건 덕분에 L&S 스펙트럼은 이산 차수에만 값을
    가지므로, 에너지가 있는 성분만 추려 K 를 작게 유지할 수 있다.

    reach 를 주면 '어떤 광원점에서도 동공을 통과할 수 없는' 차수를 추가로
    버린다. 정규화 동공 좌표에서 차수 위치가 |p| > 1 + sigma_out 이면
    광원점을 어디에 두어도 동공 안으로 들어오지 못하므로 상에 전혀 기여하지
    않는다. 대역제한 마스크는 격자 Nyquist 까지 모든 차수가 0 이 아니어서
    (해석 계수를 그대로 넣기 때문) 이 선별이 없으면 K 가 수십 개로 불어나고,
    TCC 비용이 K^2 이므로 수백 배 낭비가 된다. 물리적으로 버리는 것이 없다.
    """
    power = np.abs(spectrum) ** 2
    keep = power > power.max() * rel_tol
    if reach is not None:
        keep &= np.abs(grid.fx) <= reach + 1e-12
    idx = np.nonzero(keep)[0]
    return idx, grid.fx[idx], spectrum[idx]


def aerial_image_tcc(
    grid: SimGrid,
    mask: np.ndarray,
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    sigma_out: float = 0.0,
    sigma_in: float = 0.0,
    defocus: float = 0.0,
    n_source_grid: int = 31,
    polarization: str = IM.POL_SCALAR,
    source_shape: str = IM.SOURCE_CONVENTIONAL,
    pole_offset: float = 0.0,
    pole_radius: float = 0.15,
    aberrations: dict[int, float] | None = None,
    n_resist: float | None = None,
    rel_tol: float = 1e-12,
) -> np.ndarray:
    """Hopkins TCC 방식 공간 광강도.

    imaging.aerial_image 와 동일한 인자를 받으며 동일한 결과를 내야 한다.
    플레어와 스캔 동기화 오차는 결상 이후의 후처리이므로 여기서는 다루지
    않는다(대조 대상이 결상 연산 자체이기 때문).
    """
    if mask.shape != (grid.n_points,):
        raise ValueError("mask 배열 길이가 격자와 일치하지 않습니다.")
    if na >= n_medium:
        raise ValueError(
            f"NA({na:.3f}) >= n_medium({n_medium:.3f}): 물리적으로 불가능한 조합입니다.")

    if polarization == IM.POL_UNPOLARIZED:
        kw = dict(wavelength=wavelength, na=na, n_medium=n_medium,
                  sigma_out=sigma_out, sigma_in=sigma_in, defocus=defocus,
                  n_source_grid=n_source_grid, source_shape=source_shape,
                  pole_offset=pole_offset, pole_radius=pole_radius,
                  aberrations=aberrations, n_resist=n_resist, rel_tol=rel_tol)
        te = aerial_image_tcc(grid, mask, polarization=IM.POL_TE, **kw)
        tm = aerial_image_tcc(grid, mask, polarization=IM.POL_TM, **kw)
        return 0.5 * (te + tm)

    n_pol = n_medium if n_resist is None else n_resist
    spectrum = np.fft.fft(mask)
    reach = (1.0 + max(sigma_out, 0.0)) * na / wavelength
    idx, f_ord, a_ord = _orders(grid, spectrum, rel_tol, reach=reach)
    K = len(idx)

    src, src_w = IM.source_quadrature(sigma_out, sigma_in, n_source_grid,
                                      shape=source_shape,
                                      pole_offset=pole_offset,
                                      pole_radius=pole_radius)
    k_medium = n_medium / wavelength
    lam_na = wavelength / na

    # 차수별 동공 좌표 (광원점 이동 전)
    px0 = f_ord * lam_na                       # (K,)

    tcc = np.zeros((K, K), dtype=np.complex128)
    for (sx, sy), wq in zip(src, src_w):
        px = px0 + sx                          # (K,)
        py = np.full(K, sy)
        inside = (px**2 + py**2) <= 1.0
        P = inside.astype(np.complex128)

        if defocus != 0.0:
            fx_t = f_ord + sx * na / wavelength
            fy_t = sy * na / wavelength
            kz = np.sqrt(np.maximum(k_medium**2 - (fx_t**2 + fy_t**2), 0.0))
            P = P * np.exp(2j * np.pi * defocus * kz)
        if aberrations:
            P = P * np.exp(2j * np.pi * IM.zernike_phase(px, py, aberrations))

        if polarization == IM.POL_SCALAR:
            tcc += wq * np.outer(P, np.conj(P))
        else:
            ex, ey, ez = IM._field_unit_vectors(px, py, na, n_pol, polarization)
            # 두 차수의 전기장 방향 내적이 교차항 가중이 된다
            dot = (np.outer(ex, ex) + np.outer(ey, ey) + np.outer(ez, ez))
            tcc += wq * np.outer(P, np.conj(P)) * dot
    # 구적 가중치 합이 1 이므로 추가 정규화가 없다.

    # I(x) = sum_{m,n} TCC[m,n] A_m A_n^* exp(i2pi (f_m - f_n) x)
    coef = tcc * np.outer(a_ord, np.conj(a_ord))        # (K,K)
    phase = np.exp(2j * np.pi * np.subtract.outer(f_ord, f_ord)[..., None]
                   * grid.x[None, None, :])             # (K,K,n_points)
    inten = np.einsum("mn,mnx->x", coef, phase)

    # 정규화 : Abbe 구현은 IFFT 의 1/N 규약을 포함하므로 맞춰 준다
    inten = inten.real / grid.n_points ** 2
    return inten


# ===========================================================================
# 영역 분해 정확 구적
# ===========================================================================
#
# 왜 필요한가
# -----------
# Abbe 방식은 광원 영역 전체에 구적점을 깔고 평균한다. 그런데 피적분 함수는
# 광원 위치에 대해 매끄럽지 않다. 광원점이 움직이면 회절 차수가 동공 경계를
# 드나들면서 상이 **불연속적으로** 바뀐다. 실제로 pitch 100 nm, NA 1.35,
# lambda 193 nm 조건에서 -1 차는 sx = 1 - lambda/(NA*pitch) = 0.4296 에서
# 갑자기 들어온다.
#
# 불연속이 적분 영역 안에 있으면 구적 차수를 올려도 소용이 없다. 정방 격자든
# Gauss-Legendre 든 O(1/n) 을 넘지 못한다(실측 관측 차수 p = 0.93, 그리고
# 극좌표 Gauss-Legendre 로 바꾸어도 비단조 O(1/n) 거동이 남았다). 이것이
# 생산 설정 n_source_grid=31 에서 대비에 2.4 % 오차를 남긴 원인이다.
#
# 해결
# ----
# 불연속을 적분 영역 '안' 에서 '경계' 로 옮긴다. Hopkins 형식에서 각 차수 쌍
# (m, n) 의 전달교차계수는
#
#     W_mn = (1/A) \int_{S \cap P_m \cap P_n} g_mn(s) dA
#
# 이고, 적분 영역은 광원 영역 S 와 두 이동 동공 P_m, P_n 의 교집합이다. 이
# 영역 안에서 피적분 함수 g_mn 은 매끄럽다(위상 항과 편광 내적뿐이다).
# 1차원 선 패턴에서는 모든 차수 이동이 x 축 방향이므로, 모든 경계 원의 중심이
# x 축 위에 있다. 따라서 sx 를 고정하면 허용되는 sy 구간이 **해석적으로**
# 정해진다.
#
#     P_m :  |sy| <= sqrt(1 - (sx + f_m)^2)
#     S   :  sqrt(max(0, sigma_in^2 - sx^2)) <= |sy| <= sqrt(sigma_out^2 - sx^2)
#
# sx 축은 구간 구조가 바뀌는 지점(원이 끝나는 점, 두 반경이 교차하는 점)에서
# 쪼갠 뒤 각 구간마다 Gauss-Legendre 를 쓴다. 그러면 불연속이 전부 구간
# 경계에 놓이므로 고차 수렴을 회복한다.

def _lens_area(d: float, r1: float, r2: float) -> float:
    """중심거리 d, 반경 r1, r2 인 두 원의 교집합 면적 (해석식)."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    tri = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2)
                            * (d - r1 + r2) * (d + r1 + r2)))
    return a1 + a2 - tri


def source_regions(sigma_out: float, sigma_in: float, shape: str,
                   pole_offset: float, pole_radius: float):
    """광원 영역을 'x 축 위에 중심을 둔 원(환) 들' 로 기술한다.

    영역 분해 정확 구적이 성립하는 조건은, 모든 경계 원의 중심이 x 축 위에
    있어야 한다는 것뿐이다. 그러면 sx 를 고정했을 때 허용 sy 구간이 해석적
    으로 닫힌 형태로 나온다. 원형·환형 조명은 중심이 원점이고, 쌍극 조명은
    두 극이 x 축 위에 있으므로 모두 해당한다. 사극 조명은 극이 대각선에
    있어 해당하지 않으므로 None 을 돌려준다(Abbe 경로를 써야 한다).

    Returns
    -------
    [(cx, r_out, r_in), ...] 와 전체 면적, 또는 None
    """
    if shape in (IM.SOURCE_DIPOLE, IM.SOURCE_QUADRUPOLE):
        off = pole_offset if pole_offset > 0.0 else sigma_out
        if shape == IM.SOURCE_DIPOLE:
            cs = [(+off, 0.0), (-off, 0.0)]
        else:
            d = off / np.sqrt(2.0)
            cs = [(+d, +d), (+d, -d), (-d, +d), (-d, -d)]
        regions = [(cx, cy, pole_radius, 0.0) for cx, cy in cs]
        area = sum(_lens_area(float(np.hypot(cx, cy)), 1.0, pole_radius)
                   if np.hypot(cx, cy) + pole_radius > 1.0
                   else np.pi * pole_radius ** 2 for cx, cy, _, _ in regions)
        return regions, area
    if sigma_out <= 0.0:
        return None
    so = min(sigma_out, 1.0)
    area = np.pi * (so ** 2 - sigma_in ** 2)
    if area <= 0.0:
        return None          # 폭 0 이하 환형 : 호출부가 Abbe 로 되돌리거나 막는다
    return [(0.0, 0.0, so, sigma_in)], area


def _circle_intersect_x(c1, r1, c2, r2) -> list[float]:
    """두 원의 교점의 x 좌표를 해석적으로 구한다.

    중심 (x1,y1), (x2,y2), 반경 r1, r2. 교점이 없으면 빈 목록.
    근호선(radical line)을 쓰면 닫힌 형태로 풀린다.
    """
    (x1, y1), (x2, y2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    d2 = dx * dx + dy * dy
    d = np.sqrt(d2)
    if d <= 1e-15 or d > r1 + r2 or d < abs(r1 - r2):
        return []
    a = (r1 * r1 - r2 * r2 + d2) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < 0.0:
        return []
    h = np.sqrt(h2)
    xm = x1 + a * dx / d
    return [float(xm + h * dy / d), float(xm - h * dy / d)]


def _breakpoints(px0: np.ndarray, regions) -> np.ndarray:
    """sx 적분 구간을 쪼갤 지점을 해석적으로 모은다.

    쪼개야 하는 곳은 허용 sy 구간의 구조가 바뀌는 모든 sx 이다. 경계가 전부
    원이므로 그 지점은 두 종류뿐이다.
      (1) 어떤 경계 원의 x 방향 끝점        : sx = cx +- r
      (2) 두 경계 원의 교점의 x 좌표        : _circle_intersect_x
    (2) 를 빠뜨리면 사극 조명처럼 극이 x 축을 벗어난 경우에 적분 한계가
    구간 내부에서 꺾여 수렴이 멈춘다. 축상 영역(원형·환형·쌍극)에서는
    두 원이 모두 x 축 중심이어서 (2) 가 (1) 과 같은 점으로 환원된다.
    """
    circles = [((0.0, 0.0), 1.0)]                     # 동공 좌표 단위원
    for cx, cy, ro, ri in regions:
        circles.append(((cx, cy), ro))
        if ri > 0.0:
            circles.append(((cx, cy), ri))
    for a in px0:                                      # 이동 동공
        circles.append(((-float(a), 0.0), 1.0))

    lo = min(c - r for c, _, r, _ in regions)
    hi = max(c + r for c, _, r, _ in regions)
    lo, hi = max(lo, -1.0), min(hi, 1.0)

    cand = {lo, hi}
    for (cx, cy), r in circles:
        cand |= {cx - r, cx + r}
    for i, (c1, r1) in enumerate(circles):
        for c2, r2 in circles[i + 1:]:
            cand |= set(_circle_intersect_x(c1, r1, c2, r2))
    pts = np.array(sorted(c for c in cand if lo - 1e-12 <= c <= hi + 1e-12))
    pts = np.unique(np.round(pts, 12))
    if len(pts) < 2:
        return np.array([lo, hi])
    return pts


def tcc_decomposed(
    grid: SimGrid,
    f_ord: np.ndarray,
    wavelength: float,
    na: float,
    n_medium: float,
    regions,
    area: float,
    defocus: float,
    polarization: str,
    aberrations: dict[int, float] | None,
    n_pol: float,
    n_gauss: int = 24,
    n_gauss_y: int | None = None,
) -> np.ndarray:
    """영역 분해 Gauss-Legendre 로 TCC 를 정확히 적분한다.

    sx 방향과 sy 방향의 구적점 수를 따로 둔다. sx 방향은 적분 한계가
    sqrt 로 꺾이는 구간 끝을 포함하므로 점이 많이 필요하지만, sy 방향은
    구간 내부에서 피적분 함수가 매끄러운 위상 항뿐이므로 훨씬 적은 점으로
    충분하다. 실측상 sy 는 sx 의 절반이면 같은 정확도가 나온다.

    차수 쌍 (i, j) 마다 적분 영역은
        (광원 영역) ∩ (동공) ∩ (i 차 이동 동공) ∩ (j 차 이동 동공)
    이고, 모든 경계 원의 중심이 x 축 위에 있으므로 sx 고정 시 허용 sy 는
        bot(sx) <= |sy| <= top(sx)
    로 닫힌 형태가 된다. sx 축은 _breakpoints 로 쪼개 두었으므로 각 구간
    안에서 피적분 함수와 적분 한계가 모두 매끄럽다.
    """
    K = len(f_ord)
    lam_na = wavelength / na
    px0 = f_ord * lam_na
    k_medium = n_medium / wavelength
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    ny = n_gauss_y or max(6, n_gauss // 2)
    nodes_y, weights_y = np.polynomial.legendre.leggauss(ny)
    tcc = np.zeros((K, K), dtype=np.complex128)

    def phase(px, py, ftx, fty):
        """동공 위상 인자. 영역 안에서는 투과가 보장되므로 진폭은 1 이다."""
        ph = 0.0
        if defocus != 0.0:
            kz = np.sqrt(np.maximum(k_medium ** 2 - (ftx ** 2 + fty ** 2), 0.0))
            ph = ph + defocus * kz
        if aberrations:
            ph = ph + IM.zernike_phase(px, py, aberrations)
        if np.isscalar(ph) and ph == 0.0:
            return np.ones_like(px, dtype=np.complex128)
        return np.exp(2j * np.pi * ph)

    for cx, cy, r_out, r_in in regions:
        bp = _breakpoints(px0, [(cx, cy, r_out, r_in)])
        # 꺾임 지점은 _breakpoints 가 원 교점까지 포함해 모두 올려 두므로
        # 추가 분할이 필요 없다.
        n_panel = 1
        edges = []
        for a in range(len(bp) - 1):
            edges += list(np.linspace(bp[a], bp[a + 1], n_panel + 1)[:-1])
        edges.append(bp[-1])
        for a in range(len(edges) - 1):
            x_lo, x_hi = edges[a], edges[a + 1]
            if x_hi - x_lo < 1e-13:
                continue
            # [사인 치환] 구간 끝에서 sy 구간 폭이 sqrt(x - x0) 처럼 꺾인다.
            # 경계가 원이므로 필연적이고, 그대로 Gauss-Legendre 를 쓰면 이
            # 제곱근 특이점이 수렴을 제한한다(오차 바닥 ~3e-5).
            #     sx = mid + half * sin(theta),  theta in [-pi/2, pi/2]
            # 로 치환하면 sqrt(r^2 - (sx-c)^2) 형태가 theta 의 매끄러운
            # 함수가 되어 고차 수렴이 회복된다. 원형 영역 적분에서 쓰는
            # 표준 변수변환이다.
            mid_x, half_x = 0.5 * (x_hi + x_lo), 0.5 * (x_hi - x_lo)
            th = 0.5 * np.pi * nodes
            sx = mid_x + half_x * np.sin(th)
            wx = 0.5 * np.pi * weights * half_x * np.cos(th)

            rr_src = r_out ** 2 - (sx - cx) ** 2
            rr_pup = 1.0 - sx ** 2
            bot2 = np.maximum(r_in ** 2 - (sx - cx) ** 2, 0.0)
            rr_m = 1.0 - (sx[None, :] + px0[:, None]) ** 2

            for i in range(K):
                for j in range(i, K):
                    # 동공 쪽 제약 : |sy| <= R
                    cap2 = np.minimum(np.minimum(rr_pup, rr_m[i]), rr_m[j])
                    ok_src = rr_src > 0.0
                    ok = ok_src & (cap2 > 0.0)
                    if not np.any(ok):
                        continue
                    R = np.sqrt(np.maximum(cap2, 0.0))
                    a_src = np.sqrt(np.maximum(rr_src, 0.0))
                    if abs(cy) < 1e-12:
                        # 광원 영역도 x 축 중심 -> |sy| in [bot, top], 양쪽 대칭
                        top2_ = np.minimum(np.minimum(rr_src, rr_pup),
                                           np.minimum(rr_m[i], rr_m[j]))
                        top = np.minimum(a_src, R)
                        bot = np.sqrt(bot2)
                        ok = ok & (top > bot + 1e-14)
                        signs = (+1.0, -1.0)
                        # [수치] half = (top - bot)/2 를 그대로 쓰면 두 값이
                        # 가까울 때(폭이 매우 좁은 환형) 상쇄로 유효숫자를
                        # 잃는다. 제곱의 차로 바꾸면 상쇄가 없다.
                        #     (top - bot) = (top^2 - bot^2) / (top + bot)
                        den = top + bot
                        half = np.where(den > 0.0,
                                        (top2_ - bot2) / (2.0 * np.where(
                                            den > 0.0, den, 1.0)),
                                        0.0)
                        mid = bot + half
                    else:
                        # 중심이 (cx, cy) 인 극 : sy in [cy-a, cy+a] 와
                        # [-R, R] 의 교집합. 항상 단일 구간이다.
                        s_lo = np.maximum(cy - a_src, -R)
                        s_hi = np.minimum(cy + a_src, R)
                        ok = ok & (s_hi > s_lo + 1e-14)
                        signs = (+1.0,)
                        mid, half = 0.5 * (s_hi + s_lo), 0.5 * (s_hi - s_lo)
                    if not np.any(ok):
                        continue
                    acc = np.zeros(len(sx), dtype=np.complex128)
                    for sgn in signs:
                        base = mid[:, None] + half[:, None] * nodes_y[None, :]
                        sy = sgn * base if sgn < 0 else base
                        wy = np.abs(half)[:, None] * weights_y[None, :]
                        pxi = (sx[:, None] + px0[i]) * np.ones_like(sy)
                        pxj = (sx[:, None] + px0[j]) * np.ones_like(sy)
                        ftx_i = f_ord[i] + sx[:, None] * na / wavelength
                        ftx_j = f_ord[j] + sx[:, None] * na / wavelength
                        fty = sy * na / wavelength
                        g = phase(pxi, sy, ftx_i, fty) * np.conj(
                            phase(pxj, sy, ftx_j, fty))
                        if polarization != IM.POL_SCALAR:
                            exi, eyi, ezi = IM._field_unit_vectors(
                                pxi, sy, na, n_pol, polarization)
                            exj, eyj, ezj = IM._field_unit_vectors(
                                pxj, sy, na, n_pol, polarization)
                            g = g * (exi * exj + eyi * eyj + ezi * ezj)
                        acc += np.where(ok[:, None], g * wy, 0.0).sum(axis=1)
                    val = np.sum(acc * wx)
                    tcc[i, j] += val
                    if i != j:
                        tcc[j, i] += np.conj(val)
    return tcc / area


def zernike_phase_safe(px, py, coeffs):
    """zernike_phase 를 배열 모양 그대로 적용한다."""
    return IM.zernike_phase(px, py, coeffs)


def aerial_image_exact(
    grid: SimGrid,
    mask: np.ndarray,
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    sigma_out: float = 0.0,
    sigma_in: float = 0.0,
    defocus: float = 0.0,
    polarization: str = IM.POL_SCALAR,
    source_shape: str = IM.SOURCE_CONVENTIONAL,
    pole_offset: float = 0.0,
    pole_radius: float = 0.15,
    aberrations: dict[int, float] | None = None,
    n_resist: float | None = None,
    n_gauss: int = 16,
    n_gauss_y: int | None = None,
    rel_tol: float = 1e-12,
) -> np.ndarray:
    """영역 분해 정확 구적으로 계산한 공간 광강도.

    환형·원형 조명과 1차원 주기 마스크에 대해 광원 적분의 구적 오차를
    제거한다. 쌍극·사극 조명은 경계 원의 중심이 x 축을 벗어나므로 이
    경로를 쓸 수 없고, 기존 Abbe 경로를 써야 한다.
    """
    if polarization == IM.POL_UNPOLARIZED:
        kw = dict(wavelength=wavelength, na=na, n_medium=n_medium,
                  sigma_out=sigma_out, sigma_in=sigma_in, defocus=defocus,
                  source_shape=source_shape, pole_offset=pole_offset,
                  pole_radius=pole_radius, aberrations=aberrations,
                  n_resist=n_resist, n_gauss=n_gauss,
                  n_gauss_y=n_gauss_y, rel_tol=rel_tol)
        return 0.5 * (aerial_image_exact(grid, mask, polarization=IM.POL_TE, **kw)
                      + aerial_image_exact(grid, mask, polarization=IM.POL_TM, **kw))

    n_pol = n_medium if n_resist is None else n_resist
    spectrum = np.fft.fft(mask)
    reach = (1.0 + max(sigma_out, 0.0)) * na / wavelength
    idx, f_ord, a_ord = _orders(grid, spectrum, rel_tol, reach=reach)
    reg = source_regions(sigma_out, sigma_in, source_shape,
                         pole_offset, pole_radius)
    if reg is None:
        raise ValueError(
            "이 조명 형상(사극 등)은 영역 분해 정확 구적이 성립하지 않습니다. "
            "imaging.aerial_image(source_integration='abbe') 를 쓰십시오.")
    regions, area = reg

    # [구적점 자동 조정] 환형이 얇아지면 sx 방향 피적분 함수가
    #     top(sx) - bot(sx) ~ w * sigma / sqrt(sigma^2 - sx^2)
    # 꼴로 |sx| -> sigma_in 에서 발산하는 적분 가능 특이성을 갖는다. 폭이
    # 넓으면(실사용 환형은 폭 0.3 수준) 기본 구적점으로 기계 정밀도가 나오나,
    # 폭이 1e-2 이하가 되면 개방필드 정규화가 1e-6 까지 어긋난다. 상대 폭에
    # 따라 구적점을 올려 둔다. 실측 : 폭 1e-2 에서 n=64 면 1e-15.
    rel_w = 1.0
    for cx, cy, ro, ri in regions:
        if ro > 0.0:
            rel_w = min(rel_w, (ro - ri) / ro)
    if rel_w < 0.2:
        bump = int(min(8, max(1, np.ceil(np.sqrt(0.2 / max(rel_w, 1e-12))))))
        n_gauss = int(min(192, n_gauss * bump))

    tcc = tcc_decomposed(grid, f_ord, wavelength, na, n_medium, regions, area,
                         defocus, polarization, aberrations, n_pol, n_gauss,
                         n_gauss_y)
    coef = tcc * np.outer(a_ord, np.conj(a_ord))
    phase = np.exp(2j * np.pi * np.subtract.outer(f_ord, f_ord)[..., None]
                   * grid.x[None, None, :])
    inten = np.einsum("mn,mnx->x", coef, phase)
    return inten.real / grid.n_points ** 2


def compare_with_abbe(grid, mask, **kw) -> dict:
    """두 구현의 결과를 비교해 최대 절대·상대 편차를 돌려준다.

    [중요] Abbe 쪽은 반드시 source_integration="abbe" 로 강제한다. 기본값
    "auto" 는 가능하면 이 모듈의 정확 구적으로 우회하므로, 그대로 두면
    자기 자신과 비교하게 되어 대조의 의미가 사라진다.
    """
    drop = ("flare", "msd_nm", "source_integration", "n_gauss")
    tcc_kw = {k: v for k, v in kw.items() if k not in drop}
    abbe_kw = {k: v for k, v in kw.items() if k != "source_integration"}
    a = np.asarray(IM.aerial_image(grid, mask, source_integration="abbe",
                                   **abbe_kw)).real
    h = np.asarray(aerial_image_tcc(grid, mask, **tcc_kw)).real
    scale = max(float(np.max(np.abs(a))), 1e-300)
    diff = np.abs(a - h)
    return {
        "abbe": a,
        "hopkins": h,
        "max_abs": float(diff.max()),
        "max_rel": float(diff.max() / scale),
        "rms_rel": float(np.sqrt(np.mean(diff**2)) / scale),
    }
