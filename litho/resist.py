"""
resist.py -- 레지스트 임계값 모델 및 CD 추출

모델
----
CTR (Constant Threshold Resist): 광강도가 임계값 이상인 영역만 반응한 것으로
간주한다. PR 내부 화학 반응(Dill 파라미터, 현상 속도)은 모델링하지 않는다.

Dose 처리
---------
정규화된 aerial image 에 Dose 비를 곱한다. open frame 에서 I=1.0 이 되도록
정규화되어 있으므로, dose/dose_ref 배율이 그대로 노광량 비에 대응한다.

CD 추출
-------
threshold 교차점을 선형보간으로 산출한다.
격자 단위(dx)로 끊어서 뽑으면 CD 값이 양자화되어, 11주차 머신러닝 학습
라벨에 양자화 노이즈가 섞인다. 반드시 보간해야 한다.

NILS
----
Normalized Image Log Slope = CD * d(ln I)/dx 를 임계값 교차점에서 계산한다.
공정 여유도(LER, dose latitude)의 대표 지표로, 값이 클수록 패턴 경사가
가파르고 공정이 안정적이다.
"""

import numpy as np

TONE_BRIGHT = "bright"  # 광강도가 높은 영역이 feature
TONE_DARK = "dark"      # 광강도가 낮은 영역이 feature


def apply_dose(intensity: np.ndarray, dose: float, dose_ref: float = 1.0) -> np.ndarray:
    """노광량 비를 광강도에 반영한다."""
    if dose_ref <= 0:
        raise ValueError("dose_ref 는 양수여야 합니다.")
    return intensity * (dose / dose_ref)


def resist_profile(intensity: np.ndarray, threshold: float, tone: str = TONE_BRIGHT) -> np.ndarray:
    """현상 후 잔존 여부를 0/1 로 반환한다."""
    if tone == TONE_BRIGHT:
        return (intensity >= threshold).astype(float)
    return (intensity < threshold).astype(float)


class _BandLimited:
    """격자 표본에서 대역제한 보간기(Whittaker-Shannon)를 만든다.

    왜 필요한가
    -----------
    공중상의 대역은 동공 차단주파수의 두 배(세기는 진폭의 제곱이므로)이고,
    grid.nyquist_dx 는 정확히 그 대역을 만족하도록 dx 를 정한다. 따라서
    표본화된 광강도는 **정보 손실 없이** 푸리에 급수로 복원되며, 임의의
    x 에서의 값과 미분을 해석적으로 구할 수 있다.

        I(x)   = sum_k C_k exp(i 2 pi f_k x)
        I'(x)  = sum_k (i 2 pi f_k) C_k exp(i 2 pi f_k x)

    기존 구현은 경계 위치를 선형보간으로, 기울기를 np.gradient 중심차분으로
    구했다. 둘 다 O(dx^2) 오차를 남기므로, 마스크와 결상을 정확하게 만들어도
    CD 와 NILS 에 격자 의존성이 남는다(실측 NILS 관측차수 p = 2.01).
    대역제한 평가로 바꾸면 그 항이 사라진다.

    참고 : L. N. Trefethen, "Spectral Methods in MATLAB," SIAM (2000), 3장
    -- 대역제한 신호의 스펙트럼 미분은 절단 오차가 없다.
    """

    def __init__(self, x: np.ndarray, values: np.ndarray):
        self.n = len(values)
        self.dx = float(x[1] - x[0])
        self.x0 = float(x[0])
        self.c = np.fft.fft(values) / self.n
        self.f = np.fft.fftfreq(self.n, d=self.dx)
        # 짝수 길이에서 Nyquist 성분은 켤레쌍이 없어 미분이 정의되지 않는다.
        # 실수 신호의 스펙트럼 미분에서는 이 성분을 0 으로 둔다(표준 처리).
        self.cd_ = 2j * np.pi * self.f * self.c
        if self.n % 2 == 0:
            self.cd_[self.n // 2] = 0.0
        self.cdd_ = 2j * np.pi * self.f * self.cd_

    def _eval(self, coef: np.ndarray, xq: float) -> float:
        ph = np.exp(2j * np.pi * self.f * (xq - self.x0))
        return float(np.real(coef @ ph))

    def value(self, xq: float) -> float:
        return self._eval(self.c, xq)

    def deriv(self, xq: float) -> float:
        return self._eval(self.cd_, xq)

    def deriv2(self, xq: float) -> float:
        return self._eval(self.cdd_, xq)

    def dense(self, factor: int = 8) -> np.ndarray:
        """스펙트럼 영채움으로 factor 배 조밀한 표본을 만든다(보간 오차 없음)."""
        m = self.n * factor
        c = np.zeros(m, dtype=np.complex128)
        h = self.n // 2
        c[:h + 1] = self.c[:h + 1]
        c[m - (self.n - h - 1):] = self.c[h + 1:]
        return np.fft.ifft(c).real * m / self.n * (self.n / m) * m / self.n \
            if False else (np.fft.ifft(c).real * m)


def image_extrema(x: np.ndarray, intensity: np.ndarray,
                  factor: int = 8) -> dict:
    """연속(대역제한) 광강도의 참 최대·최소와 그 위치를 구한다.

    왜 필요한가
    -----------
    `intensity.max()` 는 연속 함수의 극값이 아니라 **격자점 중 최대값**이다.
    극값이 격자점에 떨어지지 않으면 O(dx^2) 만큼 어긋난다. 이 오차는
    대비(contrast)와, 임계값을 0.5*(Imax+Imin) 로 잡는 모든 계산에 그대로
    전파된다. 실측상 pitch 400 nm 조건에서 대비가 주기당 표본 수에 따라
    5.9e-4 상대로 흔들렸다. 결상 자체와 CD 추출은 기계 정밀도인데도
    '표본 극값' 때문에 격자 의존성이 되살아나는 것이다.

    광강도는 대역제한이므로 극값은 I'(x) = 0 의 해로 정확히 구할 수 있다.
    스펙트럼 영채움으로 조밀한 표본을 만들어 후보를 고르고, I'(x)=0 에
    Newton 법을 적용한다.
    """
    bl = _BandLimited(x, np.asarray(intensity, dtype=float))
    dense = bl.dense(factor)
    dx_f = (x[1] - x[0]) / factor
    x0 = float(x[0])

    def refine(i_start: int) -> tuple[float, float]:
        xq = x0 + i_start * dx_f
        for _ in range(20):
            d1, d2 = bl.deriv(xq), bl.deriv2(xq)
            if d2 == 0.0:
                break
            step = d1 / d2
            xq -= step
            if abs(step) < 1e-13 * max(1.0, abs(xq)):
                break
        return xq, bl.value(xq)

    xmax, imax = refine(int(np.argmax(dense)))
    xmin, imin = refine(int(np.argmin(dense)))
    # Newton 이 엉뚱한 곳으로 가면 조밀 표본 값을 쓴다
    if not (imax >= dense.max() - 1e-9 * abs(dense.max() or 1.0)):
        xmax, imax = x0 + int(np.argmax(dense)) * dx_f, float(dense.max())
    if not (imin <= dense.min() + 1e-9 * abs(dense.min() or 1.0)):
        xmin, imin = x0 + int(np.argmin(dense)) * dx_f, float(dense.min())
    return {"i_max": float(imax), "i_min": float(imin),
            "x_max": float(xmax), "x_min": float(xmin)}


def _refine_crossing(bl: _BandLimited, x_lin: float, threshold: float,
                     dx: float) -> float:
    """선형보간 해를 출발점으로 대역제한 해를 Newton 법으로 조인다.

    한 격자칸 밖으로 벗어나면 보간기가 다른 경계를 잡은 것이므로 버린다.
    """
    xq = x_lin
    for _ in range(12):
        f = bl.value(xq) - threshold
        d = bl.deriv(xq)
        if d == 0.0:
            break
        step = f / d
        xq -= step
        if abs(step) < 1e-12 * max(1.0, abs(xq)):
            break
    return xq if abs(xq - x_lin) <= dx else x_lin


def refine_crossings_2d(x: np.ndarray, rows: np.ndarray, x_lin: np.ndarray,
                        threshold: float, max_shift: float | None = None,
                        n_iter: int = 3) -> np.ndarray:
    """여러 행의 경계 위치를 대역제한 보간으로 한꺼번에 조인다.

    행마다 _BandLimited 를 새로 만들면 행 수만큼 FFT 가 돌아 30 배 느려진다
    (512 행에서 7 ms -> 210 ms 로 실측). 행 축을 유지한 채 한 번의 실수 FFT
    로 모든 행의 스펙트럼을 얻고, Newton 반복을 행 방향으로 벡터화한다.
    결과는 행마다 따로 조이는 것과 수치적으로 동일하다(실측 2e-13 nm).

    실수 신호이므로 rfft 를 쓴다. 양의 주파수만 저장되므로
        I(x) = c_0 + 2 Re( sum_{k>=1} c_k e^{i 2 pi f_k x} )
    이고 (짝수 길이의 Nyquist 항은 두 번 세지 않는다), 배열과 연산량이
    절반이 된다.

    n_iter : 선형보간 해에서 출발하면 Newton 이 2 회에 수렴한다(실측
        4.6e-13 nm, 3 회에서 완전 일치). 기본값 3 으로 둔다.

    max_shift 를 주면 그보다 멀리 움직인 해는 버리고 입력값을 유지한다
    (보간기가 다른 경계를 잡은 경우를 걸러내기 위함).
    """
    rows = np.asarray(rows, dtype=float)
    n = rows.shape[1]
    dx = float(x[1] - x[0])
    c = np.fft.rfft(rows, axis=1) / n                  # (R, n//2+1)
    fr = np.fft.rfftfreq(n, d=dx)
    # 켤레쌍 가중 : 0 차와 (짝수 길이의) Nyquist 는 1, 나머지는 2
    wt = np.full(c.shape[1], 2.0)
    wt[0] = 1.0
    if n % 2 == 0:
        wt[-1] = 1.0
    cw = c * wt[None, :]
    cd = 2j * np.pi * fr[None, :] * cw
    if n % 2 == 0:
        cd[:, -1] = 0.0            # Nyquist 성분의 미분은 정의되지 않는다

    xq = np.array(x_lin, dtype=float, copy=True)
    ok = np.isfinite(xq)
    if not np.any(ok):
        return xq
    x0 = float(x[0])
    cw_ok, cd_ok = cw[ok], cd[ok]
    for _ in range(n_iter):
        ph = np.exp(2j * np.pi * np.outer(xq[ok] - x0, fr))
        val = np.einsum("rk,rk->r", cw_ok, ph).real
        der = np.einsum("rk,rk->r", cd_ok, ph).real
        safe = der != 0.0
        step = np.zeros_like(val)
        step[safe] = (val[safe] - threshold) / der[safe]
        xq[ok] = xq[ok] - step
        if np.max(np.abs(step)) < 1e-13 * max(1.0, float(np.max(np.abs(xq[ok])))):
            break
    if max_shift is not None:
        bad = ~np.isfinite(xq) | (np.abs(xq - x_lin) > max_shift)
        xq = np.where(bad, x_lin, xq)
    return xq


def _walk_crossing(x: np.ndarray, g: np.ndarray, i_start: int, step: int) -> float:
    """g 의 부호가 바뀌는 지점을 선형보간으로 찾는다."""
    n = len(g)
    i = i_start
    while 0 <= i + step < n:
        j = i + step
        if g[j] <= 0.0:
            denom = g[i] - g[j]
            if denom == 0.0:
                return float(x[j])
            t = g[i] / denom
            return float(x[i] + t * (x[j] - x[i]))
        i = j
    return float("nan")


def extract_cd(
    x: np.ndarray,
    intensity: np.ndarray,
    threshold: float,
    tone: str = TONE_BRIGHT,
    max_width: float | None = None,
    interp: str = "spectral",
    _bl: "_BandLimited | None" = None,
) -> dict:
    """필드 중앙 feature 의 CD 를 선형보간으로 산출한다.

    max_width 를 주면 그보다 넓은 결과는 패턴 병합으로 간주하여
    미형성으로 처리한다. Defocus 가 과다하면 인접 패턴이 붙어
    경계 탐색이 여러 주기를 가로질러 물리적으로 불가능한 CD 가
    산출되므로, 통상 pitch 를 상한으로 지정한다.

    Returns
    -------
    dict -- cd, left_edge, right_edge, nils, formed
    """
    g = intensity - threshold if tone == TONE_BRIGHT else threshold - intensity
    i_center = len(x) // 2

    if g[i_center] <= 0.0:
        # [주의] 미형성 시 cd 는 항상 NaN 으로 통일한다. 0.0 을 돌려주면
        # 호출부가 formed 를 확인하지 않을 때 '선폭 0 nm' 라는 실제 값으로
        # 평균에 섞여 들어가 조용히 틀린 통계를 만든다.
        return {
            "cd": float("nan"),
            "left_edge": float("nan"),
            "right_edge": float("nan"),
            "nils": 0.0,
            "formed": False,
            "reason": "중앙이 임계값 미만 (패턴 미형성)",
        }

    left = _walk_crossing(x, g, i_center, -1)
    right = _walk_crossing(x, g, i_center, +1)

    if np.isnan(left) or np.isnan(right):
        return {
            "cd": float("nan"),
            "left_edge": left,
            "right_edge": right,
            "nils": 0.0,
            "formed": False,
        }

    bl = _bl
    if interp == "spectral" and len(x) > 4:
        dx = float(x[1] - x[0])
        if bl is None:
            # [성능] 보간기 구성은 FFT 한 번이다. 같은 프로파일을 노광량만
            # 바꾸어 수백 번 호출하는 경로(dose_to_size)에서는 호출부가
            # 한 번 만들어 넘겨주는 편이 훨씬 싸다.
            bl = _BandLimited(x, np.asarray(intensity, dtype=float))
        left = _refine_crossing(bl, left, threshold, dx)
        right = _refine_crossing(bl, right, threshold, dx)

    cd = right - left
    if max_width is not None and cd >= max_width:
        return {
            "cd": float("nan"),
            "left_edge": left,
            "right_edge": right,
            "nils": 0.0,
            "formed": False,
            "reason": "패턴 병합 (CD 가 상한을 초과)",
        }
    nils_l = _nils(x, intensity, left, cd, bl)
    nils_r = _nils(x, intensity, right, cd, bl)
    return {
        "cd": cd,
        "left_edge": left,
        "right_edge": right,
        # [기준] 공정 여유는 두 경계 중 나쁜 쪽이 결정하므로 최솟값을 쓴다.
        # 대칭 조명·대칭 마스크에서는 좌우가 같지만, 코마 수차나 비대칭
        # 조명에서는 한쪽이 먼저 무너진다.
        "nils": min(nils_l, nils_r),
        "nils_left": nils_l,
        "nils_right": nils_r,
        "formed": True,
    }


def _nils(x: np.ndarray, intensity: np.ndarray, edge: float, cd: float,
          bl: "_BandLimited | None" = None) -> float:
    """Normalized Image Log Slope = CD * |d(ln I)/dx| at edge.

    대역제한 보간기가 주어지면 ln I 를 직접 미분하는 대신
        d(ln I)/dx = I'(x) / I(x)
    를 쓴다. ln I 는 대역제한이 아니지만 I 는 대역제한이므로, 이 형태가
    절단 오차 없는 유일한 경로이다. 로그를 먼저 취하고 중심차분하면
    O(dx^2) 오차가 남는다(관측차수 p = 2.01 로 실측됨).
    """
    if bl is not None:
        i_val = bl.value(edge)
        if i_val > 1e-12:
            return abs(bl.deriv(edge) / i_val) * cd
    safe = np.maximum(intensity, 1e-12)
    slope = np.gradient(np.log(safe), x)
    return abs(float(np.interp(edge, x, slope))) * cd


def cd_vs_dose(
    x: np.ndarray,
    intensity: np.ndarray,
    threshold: float,
    doses: np.ndarray,
    tone: str = TONE_BRIGHT,
    max_width: float | None = None,
) -> np.ndarray:
    """Dose 스윕에 대한 CD 곡선. 미형성 구간은 NaN.

    [10주차] max_width 인자가 빠져 있어 8주차의 패턴 병합 판정이 적용되지
    않았다. 호출부가 없는 함수였으나, 그대로 두면 이후 작업에서 병합을
    감지하지 못한 채 사용될 수 있으므로 다른 경로와 동일하게 맞췄다.
    """
    return np.array(
        [
            extract_cd(x, apply_dose(intensity, d), threshold, tone,
                       max_width=max_width)["cd"]
            for d in doses
        ]
    )
