"""
optics.py -- Rayleigh 해상도 / 초점심도 계산 (4주차 모듈의 보정판)

4주차 대비 보정 사항
--------------------
[보정 1] R 의 물리적 의미
    R = k1 * lambda / NA 에서 R 은 '최소 선폭(CD)' 이 아니라
    해당 결상계가 해상할 수 있는 가장 조밀한 1D Line & Space 패턴의
    half-pitch 이다. 2-beam imaging 에서 k1 의 이론적 하한은 0.25 이다.
    equal line/space (duty 0.5) 에서만 선폭 = half-pitch 가 되므로,
    duty 가 0.5 가 아닌 패턴을 다루는 5주차부터는 구분이 필요하다.
    -> 함수명을 half_pitch_limit 으로 변경.

[보정 2] 굴절률 n 의 도입
    NA = n * sin(theta) 이므로 NA > 1 은 n > 1 인 매질(이머전)을 전제한다.
    4주차에는 n 입력이 없는 상태로 NA 1.35 를 허용하여 물리적 모순이 있었다.
    본 모듈은 n 을 명시적 입력으로 받는다.
    ArF 이머전: lambda_vac = 193 nm, n(물) = 1.44, NA = 1.35.

[보정 3] 고NA 초점심도
    DOF = k2 * lambda / NA^2 은 근축(paraxial) 조건에서 유도된 식이며
    NA > 1 영역에서는 성립하지 않는다.
    - 근축형(이머전 이득 포함): DOF = k2 * n * lambda_vac / NA^2
    - 고NA형(B. J. Lin): DOF = k3 * (lambda_vac / n) / sin^2(theta/2),
      sin(theta) = NA / n
    두 식을 모두 제공하여 비교할 수 있도록 한다.

[10주차] 벡터(편광) 모델
    imaging.py 가 벡터 결상을 지원한다. 통상 NA > 0.6 을 고NA 로 보며
    (Flagello et al., JOSA A 13(1), 53, 1996), 그 이상에서 TM 편광의
    contrast 가 저하된다. 본 모듈의 Rayleigh 식 자체는 편광과 무관한
    스칼라 관계식이므로 참고값으로만 쓴다.
    스칼라 결상은 '틀린 모델' 이 아니라 'TE 편광 가정 모델' 이며, 본
    조건 공간에서 TE 벡터 결과와의 차이는 8% 이내이다.

기준 상수(ArF 이머전 실측): k1 ~ 0.288, k2 ~ 0.745
"""

import numpy as np

SCALAR_MODEL_NA_LIMIT = 0.6
K1_THEORETICAL_LIMIT = 0.25


def half_pitch_limit(k1: float, wavelength: float, na: float) -> float:
    """해상 가능한 최소 half-pitch [nm].

    R = k1 * lambda_vac / NA,  NA = n*sin(theta) (이머전 포함)
    """
    return k1 * wavelength / na


def k1_required(half_pitch: float, na: float, wavelength: float) -> float:
    """주어진 half-pitch 를 구현하는 데 필요한 k1 (역산)."""
    return half_pitch * na / wavelength


def dof_paraxial(k2: float, wavelength: float, na: float, n_medium: float = 1.0) -> float:
    """근축 초점심도 [nm].  DOF = k2 * n * lambda_vac / NA^2

    분자의 n 이 이머전에 의한 DOF 이득에 해당한다.
    """
    return k2 * n_medium * wavelength / na**2


def dof_high_na(k3: float, wavelength: float, na: float, n_medium: float = 1.0) -> float:
    """고NA 초점심도 [nm] (B. J. Lin 형).

    DOF = k3 * (lambda_vac / n) / sin^2(theta/2),  sin(theta) = NA / n
    """
    sin_theta = na / n_medium
    if sin_theta >= 1.0:
        raise ValueError(
            f"NA({na}) 가 매질 굴절률 n({n_medium}) 이상입니다. "
            "물리적으로 불가능한 조합입니다."
        )
    theta = np.arcsin(sin_theta)
    return k3 * (wavelength / n_medium) / np.sin(theta / 2.0) ** 2


def rayleigh_summary(
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    k1: float = 0.288,
    k2: float = 0.745,
    k3: float = 0.745,
    polarization: str = "scalar",
) -> dict:
    """해상도 / 초점심도 일괄 계산 및 모델 유효성 경고.

    [10주차] 경고 문구를 편광 설정에 따라 구분한다. 9주차까지는 NA > 0.6
    이면 무조건 '벡터 효과 미반영' 경고를 띄웠는데, 벡터 모델을 켠 경우에도
    같은 문구가 떠서 결과가 무효인 것처럼 읽혔다. 또한 스칼라 모델이라
    하더라도 그것은 TE 편광을 가정한 것에 해당하며, 본 조건 공간에서
    TE 벡터 결과와의 차이는 8% 이내이다(validate_w10 Case 3). 따라서
    '유효범위 초과' 가 아니라 '측정된 오차 범위' 로 기술한다.
    """
    warnings = []
    if na >= n_medium:
        warnings.append(
            f"NA({na:.3f}) >= n({n_medium:.3f}): 물리적으로 불가능한 조합입니다."
        )
    if na > SCALAR_MODEL_NA_LIMIT and polarization == "scalar":
        warnings.append(
            f"NA({na:.3f}) > {SCALAR_MODEL_NA_LIMIT}: 스칼라 모델입니다. "
            "이는 선에 평행한 TE 편광을 가정한 것에 해당하며, TE 벡터 모델 "
            "대비 오차는 본 조건 공간에서 8% 이내입니다. 무편광이나 TM "
            "조건을 보려면 편광 설정을 바꾸십시오."
        )

    out = {
        "half_pitch_limit": half_pitch_limit(k1, wavelength, na),
        "dof_paraxial": dof_paraxial(k2, wavelength, na, n_medium),
        "dof_high_na": None,
        "warnings": warnings,
    }
    if na < n_medium:
        out["dof_high_na"] = dof_high_na(k3, wavelength, na, n_medium)
        out["dof_ratio"] = out["dof_high_na"] / out["dof_paraxial"]
    return out


def printability(pitch: float, k1: float, wavelength: float, na: float) -> dict:
    """주어진 pitch 의 해상 가능 여부 판정."""
    hp = pitch / 2.0
    limit = half_pitch_limit(k1, wavelength, na)
    k1_req = k1_required(hp, na, wavelength)
    return {
        "half_pitch": hp,
        "half_pitch_limit": limit,
        "k1_required": k1_req,
        "printable": hp >= limit,
        "below_theoretical_limit": k1_req < K1_THEORETICAL_LIMIT,
    }
