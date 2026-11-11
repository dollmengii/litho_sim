"""
stochastic.py -- 확률 레지스트 모델 (광자 산탄 잡음 + 레지스트 블러)

배경
----
5주차까지의 CTR 모델은 결정론적이다. 같은 조건을 100번 계산하면 CD 가 100번
동일하게 나온다. 실제 공정은 그렇지 않다. 레지스트에 입사하는 광자 수는
포아송 분포를 따르며, 이 광자 산탄 잡음(photon shot noise, PSN)이 선 가장자리
거칠기(LER)와 확률적 결함의 근본 원인이다.

모델 구조
---------
    1) 광자 밀도    N(x) = Dose * I(x) * A_pixel / E_photon
    2) 포아송 샘플링  n(x) ~ Poisson(N(x))
    3) 레지스트 블러  n_blur = Gaussian(n, sigma_blur)      (산 확산)
    4) 임계값 적용    n_blur / N_ref >= threshold 인 영역만 반응

블러는 광자 수를 보존하므로 평균 광강도는 유지되고 잡음만 평활화된다.
블러 길이가 길수록 LER 은 감소하나 패턴 경사(NILS)도 함께 완만해진다.

파장에 따른 광자 수
-------------------
광자 에너지 E = 1239.84 / lambda [eV·nm] 이므로, 동일 노광량에서
    ArF(193 nm)  E = 6.42 eV
    EUV(13.5 nm) E = 91.8 eV
로 약 14.3배 차이가 나며, 같은 Dose 에서 EUV 의 광자 수는 1/14.3 이다.
산탄 잡음의 상대 크기는 1/sqrt(N) 이므로 EUV 가 약 3.8배 크다.
이것이 EUV 확률 결함(stochastic defect) 문제의 근본 원인이다.

본 모듈은 광자 통계만 다루며, 레지스트 내부 화학 잡음(PAG 분포, 2차 전자,
현상 잡음)은 모델링하지 않는다. 따라서 산출되는 LER 은 하한에 해당한다.

레지스트 블러와 스캔 동기화 오차(MSD)의 구분
---------------------------------------------
두 효과는 모두 가우시안 평활이지만 작용 시점이 다르다.
    MSD   : 노광 시간 동안의 상 진동 -> 광자 샘플링 '이전' 의 광강도에 작용
    블러  : 산 확산 -> 샘플링 '이후' 의 광자 수에 작용
따라서 결정론적 상(평균 광강도)에서는 두 효과가 구별되지 않지만, MSD 는
잡음을 평활하지 않고 블러는 평활하므로 LER 에서는 구별된다. 즉 CD 만
측정하면 식별 불가능하고 LER 을 함께 측정하면 식별 가능한, 역추정
관점의 축퇴(degeneracy) 사례이다. MSD 는 imaging.apply_msd 가 담당한다.

LWR 과 LER 의 관계
------------------
좌우 경계의 거칠기가 서로 독립이면 선폭 거칠기 LWR = sqrt(2) * LER 이다.
본 모듈은 CD 통계(LWR)와 경계 통계(LER)를 서로 다른 경로로 계산하므로,
두 값의 비가 sqrt(2) 에 수렴하는지로 모듈 전체를 교차 검증할 수 있다.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from . import resist as RS

HC_EV_NM = 1239.84          # 플랑크 상수 x 광속 [eV·nm]
EV_TO_J = 1.602176634e-19   # [J/eV]
MJ_CM2_TO_J_NM2 = 1e-17     # 1 mJ/cm^2 = 1e-3 J / 1e14 nm^2

# 레지스트 블러(PEB 산 확산 길이)의 표본 범위 ------------------------------
# [10주차 정정] 이 범위의 성격을 분명히 해 둔다.
#
# 아래 두 값은 **문헌 인용이 아니다.** 화학증폭형 레지스트의 PEB 확산 길이가
# 한 자리 nm 수준이라는 통념에 맞춘 범위이며, 특정 레지스트의 실측값도
# 아니다. 공개 자료의 수치는 레지스트 배합·PEB 조건·측정 방법에 따라 크게
# 달라지므로, 하나의 '표준값' 을 인용하는 것 자체가 타당하지 않다.
#
# 다만 이 값이 결론을 좌우하지는 않는다. blur_nm 은 고정 가정으로 넣는
# 입력이 아니라 **되짚어야 할 식별 대상 파라미터** 이기 때문이다. 즉
#   - 데이터셋에서는 이 범위에서 뽑아 '조건' 으로 기록하고,
#   - 식별성 분석에서는 그 값을 관측으로부터 추정할 수 있는지 묻는다.
# 따라서 이 범위는 '가정한 물리값' 이 아니라 '탐색할 영역' 이다. 공정이
# 정해지면 해당 레지스트의 자료로 영역을 좁히면 된다.
#
# 보고서에 쓸 때는 "블러를 8 nm 로 가정했다" 가 아니라 "블러를 5~15 nm
# 영역에서 추정 대상으로 두었다" 로 적어야 한다.
BLUR_RANGE_ARF = (5.0, 15.0)    # ArF 화학증폭형 [nm, 가우시안 1-sigma]
BLUR_RANGE_EUV = (2.0, 6.0)     # EUV 화학증폭형 [nm]
BLUR_DEFAULT_ARF = 8.0


def blur_range(wavelength: float) -> tuple[float, float]:
    """파장대에 맞는 레지스트 블러 범위를 돌려준다."""
    return BLUR_RANGE_EUV if wavelength < 50.0 else BLUR_RANGE_ARF


def photon_energy_ev(wavelength: float) -> float:
    """광자 1개의 에너지 [eV]."""
    return HC_EV_NM / wavelength


def photon_density(dose_mj_cm2: float, wavelength: float) -> float:
    """단위 면적당 입사 광자 수 [photons/nm^2] (광강도 I = 1 기준)."""
    e_j = photon_energy_ev(wavelength) * EV_TO_J
    return dose_mj_cm2 * MJ_CM2_TO_J_NM2 / e_j


def photon_statistics(dose_mj_cm2: float, wavelength: float, area_nm2: float = 4.0) -> dict:
    """주어진 면적에서의 광자 수와 상대 산탄 잡음."""
    density = photon_density(dose_mj_cm2, wavelength)
    n = density * area_nm2
    return {
        "photon_energy_ev": photon_energy_ev(wavelength),
        "density_per_nm2": density,
        "n_in_area": n,
        "relative_noise": 1.0 / np.sqrt(n) if n > 0 else np.inf,
        "area_nm2": area_nm2,
    }


def stochastic_field(
    intensity_1d: np.ndarray,
    dx: float,
    dose_mj_cm2: float,
    wavelength: float,
    n_rows: int = 256,
    dy: float | None = None,
    blur_nm: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """확률적 잠상(latent image)을 2D 로 생성한다.

    마스크가 y 불변이므로 공간 광강도는 1D 를 복제하여 쓰되, 광자 샘플링과
    블러는 2D 로 수행해야 선 길이 방향 변동(LER)이 나온다.

    Returns
    -------
    정규화 유효 광강도 2D 배열. 결정론 모델의 I 와 같은 단위.
    """
    if rng is None:
        rng = np.random.default_rng()
    if dy is None:
        dy = dx

    n_ref = photon_density(dose_mj_cm2, wavelength) * dx * dy  # I=1 일 때 픽셀당 광자 수
    if n_ref <= 0:
        raise ValueError("Dose 또는 파장이 유효하지 않습니다.")

    lam_map = np.tile(intensity_1d, (n_rows, 1)) * n_ref
    counts = rng.poisson(lam_map).astype(np.float64)

    if blur_nm > 0:
        counts = gaussian_filter(
            counts, sigma=(blur_nm / dy, blur_nm / dx), mode="wrap"
        )
    return counts / n_ref


def edge_positions(
    x: np.ndarray,
    field_2d: np.ndarray,
    threshold: float,
    side: str = "right",
    interp: str = "spectral",
) -> np.ndarray:
    """각 y 행마다 필드 중앙 feature 의 좌/우 경계 위치를 구한다.

    interp
    ------
    "spectral" : 선형보간 해를 출발점으로 대역제한 보간에서 조인다.
        결정론 경로(resist.extract_cd)와 같은 방식이다. 기본값.
    "linear"   : 9주차까지의 선형보간. 비교용으로 남겨 둔다.

    왜 통일해야 하는가
    ------------------
    10주차에 extract_cd 를 대역제한 보간으로 바꾸면서, 결정론 CD 는 스펙트럼
    방식이고 확률 CD 는 선형보간인 상태가 되었다. 본 과제는 '결정론 CD 산포
    대 확률 CD 산포' 를 직접 비교하므로, 두 값이 서로 다른 보간 정의로
    계산되면 비교의 의미가 흐려진다. 무잡음 한계에서 두 경로가 1e-3 nm 까지
    어긋남을 실측했다(LER 의 1/50 수준이라 결론은 바뀌지 않지만, 같은 양을
    두 방법으로 재고 있었다는 점 자체가 검증의 구멍이었다).
    """
    n_rows, n_cols = field_2d.shape
    ic = n_cols // 2
    step = 1 if side == "right" else -1
    out = np.full(n_rows, np.nan)
    dx = float(x[1] - x[0])

    for r in range(n_rows):
        g = field_2d[r] - threshold
        if g[ic] <= 0:
            continue
        i = ic
        while 0 <= i + step < n_cols:
            j = i + step
            if g[j] <= 0:
                denom = g[i] - g[j]
                out[r] = x[j] if denom == 0 else x[i] + (g[i] / denom) * (x[j] - x[i])
                break
            i = j

    if interp == "spectral":
        # 행 축을 벡터화해 한 번에 조인다 (행마다 FFT 를 만들면 30 배 느리다)
        out = RS.refine_crossings_2d(x, field_2d, out, threshold, max_shift=dx)
    return out


def monte_carlo(
    x: np.ndarray,
    intensity_1d: np.ndarray,
    dx: float,
    dose_mj_cm2: float,
    wavelength: float,
    threshold: float,
    n_rows: int = 512,
    blur_nm: float = 10.0,
    n_trials: int = 8,
    seed: int | None = 0,
    dy: float | None = None,
    max_width: float | None = None,
    interp: str = "spectral",
) -> dict:
    """확률적 CD, LER, 결함률을 산출한다.

    interp : 경계 추출 보간 방식. 결정론 경로(resist.extract_cd)와 같은
        "spectral" 이 기본값이다. 두 경로의 정의를 일치시키기 위한 것이며,
        비용은 선형보간 대비 약 1.3 배다. "linear" 로 두면 9주차 거동이다.

    LER 은 경계 위치의 3-sigma 로 정의한다 (업계 관례).
    'defect_rate' 는 중앙 feature 의 경계를 찾지 못한 행의 비율이다
    (패턴 끊김 또는 인접 패턴 병합). 이름과 달리 **확률적 결함 발생률이
    아니다.**

    [측정 한계] 직접 표본화로 분해할 수 있는 최소 비율은 1/(n_rows x
    n_trials) 로 약 1e-4 이다. 공정에서 문제가 되는 확률적 결함률은
    1e-9~1e-12 수준이므로 이 방식으로는 측정할 수 없다. 그 영역은 희소
    사건 추정(중요도 표본화, 극단값 이론)이 필요하고 본 과제 범위 밖이다.
    실측 분포가 0 또는 약 0.99 의 이봉인 것도 이 때문이다. 즉 이 값은
    '패턴이 형성되었는가' 의 이진 판정에 가깝다.

    주의 : LER 은 반복 횟수가 아니라 y 방향 측정 길이(n_rows x dy)에 수렴한다.
    선 가장자리 거칠기는 장파장 성분을 포함하므로 측정 창이 짧으면 저주파
    성분이 누락되어 LER 이 과소평가된다. 측정 길이는 블러 길이의 100배
    이상(통상 1000 nm 이상)을 권장하며, dy 는 블러 길이의 1/3 이하로 둔다.
    dy 를 dx 와 분리하면 연산 비용 증가 없이 측정 길이를 확보할 수 있다.
    """
    if dy is None:
        dy = max(dx, blur_nm / 3.0) if blur_nm > 0 else dx
    rng = np.random.default_rng(seed)
    cds, lers, n_fail, n_total = [], [], 0, 0

    for _ in range(n_trials):
        field = stochastic_field(
            intensity_1d, dx, dose_mj_cm2, wavelength,
            n_rows=n_rows, dy=dy, blur_nm=blur_nm, rng=rng,
        )
        left = edge_positions(x, field, threshold, "left", interp=interp)
        right = edge_positions(x, field, threshold, "right", interp=interp)
        ok = np.isfinite(left) & np.isfinite(right)

        n_total += n_rows
        n_fail += int((~ok).sum())
        if ok.sum() < 2:
            continue

        l_ok, r_ok = left[ok], right[ok]
        w = r_ok - l_ok
        if max_width is not None:
            good = w < max_width
            n_fail += int((~good).sum())
            # [10주차 수정] 이전에는 good 마스크를 CD 에만 적용하고 LER 은
            # 거르지 않은 right[ok] 전체로 계산했다. 병합된 행의 '우측 경계'는
            # 다른 주기에서 잡히므로 위치가 크게 어긋나며, 그 행이 섞이면
            # LER 이 비정상적으로 커진다. 8주차에 관측된 LER 400 nm 급
            # 이상값의 구조적 원인이 이것이다.
            l_ok, r_ok, w = l_ok[good], r_ok[good], w[good]
            if w.size < 2:
                continue
        cds.append(w)
        # 양쪽 경계를 모두 쓴다. 한쪽만 쓰면 표본이 절반이라 추정 분산이 커진다.
        #
        # [추정량 규약 - 반드시 알아야 할 비대칭]
        # ler_3sigma 는 **시행별 표준편차를 구해 평균**한다(trial-mean).
        # 반면 cd_3sigma 는 모든 시행을 합쳐 한 번에 표준편차를 구한다
        # (pooled, 아래 allw 참조). 두 라벨의 규약이 다르다.
        #
        # 그래서 LWR/LER 비를 그대로 '물리 검증' 으로 읽으면 안 된다.
        # 직접 측정하면
        #   둘 다 pooled    : 1.4090  (이론 sqrt(2-2rho) = 1.4086, 0.03 %)
        #   둘 다 시행평균   : 1.4011  (표본 잡음 범위)
        #   혼합(현재 코드) : 1.4354  (약 1.9 % 높음)
        # 즉 비의 1~2 % 잔차는 물리가 아니라 이 규약 차이에서 온다. 실측
        # 좌우 경계 상관은 rho = +0.008 수준이고, 규약을 맞추면 항등식이
        # 0.03 % 로 성립한다.
        #
        # 규약을 통일하지 않는 이유는 하나다. ler_3sigma 는 데이터셋 라벨이고
        # 정의를 바꾸면 이미 만든 9 만 건을 다시 뽑아야 한다(13 시간). 2 %
        # 규약을 맞추려고 기존 데이터와의 일관성을 깨는 것은 손해다. 항등식의
        # 엄밀한 증거는 validate_w10 의 분산 항등식 시험(1.7e-16)에 따로 있다.
        #
        # 물리적으로는 trial-mean 쪽이 LER 추정량으로 더 적절하다. pooled 는
        # 시행마다 평균 경계 위치가 흔들린 것(유한 표본 효과)을 물리적 산포에
        # 더해 과대평가한다.
        lers.append(0.5 * (3.0 * np.std(r_ok) + 3.0 * np.std(l_ok)))

    if not cds:
        return {"formed": False, "cd_mean": np.nan, "cd_3sigma": np.nan,
                "ler_3sigma": np.nan, "defect_rate": 1.0}

    allw = np.concatenate(cds)
    return {
        "formed": True,
        "cd_mean": float(np.mean(allw)),
        "cd_3sigma": float(3.0 * np.std(allw)),
        "ler_3sigma": float(np.mean(lers)),
        "defect_rate": n_fail / n_total,
        "n_samples": int(allw.size),
        "y_length_nm": float(n_rows * dy),
        "dy": float(dy),
    }
