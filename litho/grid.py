"""
grid.py -- 시뮬레이션 격자 정의 및 샘플링 조건 검증

설계 원칙
---------
1. pitch 는 반드시 dx 의 정수배가 되도록 격자를 구성한다.
   FFT 는 주기 경계를 가정하므로, 필드 폭 L 이 pitch 의 정수배가 아니면
   경계 불연속에 의한 ringing 아티팩트가 발생한다.
2. 샘플 간격 dx 는 Nyquist 조건을 만족해야 한다.
   스테퍼 aerial image 의 Nyquist 샘플링 레이트는 4*NA/lambda 이며,
   부분결맞음 조명(sigma)에서는 필드 대역이 (1+sigma)*NA/lambda 로 확장되므로
       dx <= lambda / (4 * (1 + sigma_out) * NA)
   를 적용한다.
3. 마스크의 사각 에지는 대역이 무한대이므로, 위 조건을 만족하더라도
   실제로는 1~2 nm 수준으로 과샘플링하는 것이 안전하다.

본 시뮬레이터는 모든 길이를 웨이퍼면 기준 nm 단위로 다룬다.
축소 투영비(통상 4x)는 1x 로 단순화한다.
"""

from dataclasses import dataclass

import numpy as np

MIN_SAMPLES_PER_PITCH = 16


@dataclass(frozen=True)
class SimGrid:
    """1D 시뮬레이션 격자. 모든 길이 단위는 nm."""

    pitch: float
    n_periods: int
    samples_per_pitch: int

    @property
    def dx(self) -> float:
        return self.pitch / self.samples_per_pitch

    @property
    def n_points(self) -> int:
        return self.samples_per_pitch * self.n_periods

    @property
    def field_width(self) -> float:
        return self.pitch * self.n_periods

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.n_points) * self.dx

    @property
    def fx(self) -> np.ndarray:
        """FFT 주파수 축 [cycles/nm]. np.fft 순서를 그대로 따른다."""
        return np.fft.fftfreq(self.n_points, d=self.dx)

    def describe(self) -> str:
        return (
            f"pitch={self.pitch:.2f} nm, dx={self.dx:.3f} nm, "
            f"N={self.n_points}, L={self.field_width:.1f} nm"
        )


def nyquist_dx(wavelength: float, na: float, sigma_out: float = 0.0) -> float:
    """Nyquist 조건을 만족하는 최대 샘플 간격 [nm].

    aerial image 의 Nyquist 샘플링 레이트 4*NA/lambda 에 부분결맞음에 의한
    대역 확장 (1+sigma_out) 을 반영한 값.
    """
    return wavelength / (4.0 * (1.0 + sigma_out) * na)


def build_grid(
    pitch: float,
    n_periods: int = 8,
    dx_target: float = 1.0,
) -> SimGrid:
    """pitch 가 dx 의 정수배가 되도록 격자를 구성한다.

    dx_target 에 가장 가깝되 pitch 를 정확히 나누는 dx 를 선택한다.
    samples_per_pitch 는 짝수로 맞춰 feature 중심이 격자점에 오도록 한다.
    """
    if pitch <= 0:
        raise ValueError("pitch 는 양수여야 합니다.")
    if n_periods < 1:
        raise ValueError("n_periods 는 1 이상이어야 합니다.")

    spp = int(round(pitch / dx_target))
    spp = max(spp, MIN_SAMPLES_PER_PITCH)
    if spp % 2 == 1:
        spp += 1
    return SimGrid(pitch=pitch, n_periods=n_periods, samples_per_pitch=spp)


def check_sampling(
    grid: SimGrid,
    wavelength: float,
    na: float,
    sigma_out: float = 0.0,
) -> dict:
    """격자가 Nyquist 조건을 만족하는지 검사한다."""
    dx_max = nyquist_dx(wavelength, na, sigma_out)
    return {
        "dx": grid.dx,
        "dx_max": dx_max,
        "oversampling": dx_max / grid.dx,
        "ok": grid.dx <= dx_max,
        "message": (
            f"dx={grid.dx:.3f} nm (Nyquist 한계 {dx_max:.3f} nm, "
            f"{dx_max / grid.dx:.1f}배 과샘플링)"
            if grid.dx <= dx_max
            else f"dx={grid.dx:.3f} nm 가 Nyquist 한계 {dx_max:.3f} nm 를 초과합니다."
        ),
    }
