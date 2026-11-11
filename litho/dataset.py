"""
dataset.py -- 머신러닝 학습 데이터 생성 모듈

설계 원칙
---------
1. 한 건의 샘플은 (공정 조건, 관측, 라벨) 세 부분으로 구성된다.
   - 공정 조건 : 시뮬레이터 입력. 역추정의 정답에 해당
   - 관측      : 공간 광강도 프로파일. 실제 공정의 SEM 측정에 대응
   - 라벨      : CD, LER, contrast 등 스칼라 지표
2. 관측 프로파일은 한 pitch 구간을 n_profile 점으로 다운샘플링하여 저장한다.
   패턴이 주기적이므로 한 주기면 정보가 보존되며, 용량이 크게 줄어든다.
   다운샘플 수는 식별성 손실이 없는 최소값으로 결정한다(7주차 검증 참조).
3. Dose 는 dose-to-size 보정 배율 E_op 기준의 상대값으로 저장한다.
   절대 Dose 는 pitch 와 조명 조건에 따라 크게 달라져 학습이 어렵다.

연산 비용
---------
한 건당 결상 1회 + Monte Carlo n_trials 회. ArF 기준 약 2~4초.
수만 건을 생성하려면 조건 공간 설계와 병렬화가 필요하다(9주차).
"""

from dataclasses import dataclass

import numpy as np

from . import grid as G
from . import imaging as IM
from . import mask as MK
from . import process as PR
from . import resist as RS
from . import stochastic as ST

# 저장 스키마 ------------------------------------------------------------
# [10주차] 조건 벡터를 11개에서 21개로 확장하였다. 설계 원칙은
# '기록은 전부, 샘플링은 선택' 이다. 한 캠페인에서 실제로 흔드는 축은
# campaign.DEFAULT_SPACE 가 정하고, 흔들지 않은 축도 값은 기록한다.
# 그래야 나중에 캠페인을 합칠 때 열이 어긋나지 않는다.
#
# 범주형 변수는 아래 튜플의 인덱스로 부호화한다(_code 접미사).
POLARIZATION_CODES = ("scalar", "TE", "TM", "unpolarized")
SOURCE_CODES = ("conventional", "annular", "dipole", "quadrupole")
MASK_CODES = ("binary", "att-psm")

# ---------------------------------------------------------------------------
# 조건의 성격 분류 : 추천 시스템이 '추천할 수 있는 것' 과 '받아야 하는 것'
# ---------------------------------------------------------------------------
# 왜 필요한가
# -----------
# 이 과제의 목표는 공정 조건 **추천**이다. 그런데 조건 24 개는 성격이 전혀
# 다른 세 종류가 섞여 있고, 그 구분이 코드에도 문서에도 없었다. 구분하지
# 않으면 추천기가 "플레어를 0.003 으로 설정하십시오" 같은 답을 낸다.
# 플레어는 렌즈 고유 성질이어서 **설정할 수 없다.** 측정해서 받는 값이다.
# 물리적으로 불가능한 지시를 내놓는 추천기는 쓸 수 없다.
#
# 이 구분은 주장 1 의 의미도 바꾼다. 5 변수 중 되짚을 수 없는 것은
# defocus·flare·msd 인데, 이 가운데 **defocus 만 제어 변수**이고 flare·msd 는
# 주어진 값이다. 그래서 "초점과 플레어를 서로 구별할 수 없다" 는 결과는
# 추상적인 한계가 아니라 생산 현장의 구체적 문제가 된다. CD 가 규격을
# 벗어났을 때 **초점이 흐른 것인지 렌즈 플레어가 변한 것인지 웨이퍼만 보고는
# 알 수 없다**는 뜻이다. 그리고 처방이 바로 나온다. 플레어는 설정값이 아니라
# 측정 가능한 장비 성질이므로, open-frame 노광으로 따로 재서 고정하면 초점이
# 0.0257 nm 까지 진단 가능해진다.

#: 공정에서 로트·웨이퍼 단위로 조절하는 값. 추천의 **출력**이다.
CONTROL_KEYS = ("dose_rel", "dose_mj", "defocus")

#: 레시피·설비 설정으로 정하는 값. 추천의 출력이지만 로트 단위로 바꾸지
#: 않는다. 조명 형상, 편광, 마스크 설계가 여기 속한다.
RECIPE_KEYS = ("wavelength", "na", "n_medium", "sigma_out", "sigma_in",
               "pol_code", "source_code", "pole_offset", "pole_radius",
               "pitch", "duty", "mask_code", "att_t")

#: 장비·재료가 주는 값. 측정해서 받는 것이며 **설정할 수 없다.**
#: 추천기의 입력(조건)으로 들어가야 하고, 출력에 나오면 안 된다.
GIVEN_KEYS = ("flare", "msd_nm", "blur_nm", "z5", "z6", "z7", "z8", "z11")


def condition_role(key: str) -> str:
    """조건 하나의 성격을 돌려준다. 'control' / 'recipe' / 'given'."""
    if key in CONTROL_KEYS:
        return "control"
    if key in RECIPE_KEYS:
        return "recipe"
    if key in GIVEN_KEYS:
        return "given"
    raise KeyError(f"분류되지 않은 조건 '{key}'. CONTROL/RECIPE/GIVEN 중 "
                   f"어디에 속하는지 정하고 추가해야 한다.")


CONDITION_KEYS = [
    # 광학계
    "wavelength", "na", "n_medium", "sigma_out", "sigma_in",
    "pol_code", "source_code", "pole_offset", "pole_radius",
    # 마스크
    "pitch", "duty", "mask_code", "att_t",
    # 노광
    "defocus", "dose_rel", "dose_mj",
    # 결함 요인
    "flare", "msd_nm", "blur_nm",
    # 수차 (Noll, 단위 waves)
    #
    # [선택 근거] Z5·Z6(비점수차), Z7·Z8(코마), Z11(구면수차)만 넣는다.
    # Z9·Z10(trefoil)을 뺀 이유는 두 가지다.
    #   (1) [정정] 처음에는 "trefoil 은 통상 한 자릿수 작다" 고 적었으나
    #       그 근거를 제시할 수 없다. 같은 진폭 0.10 waves 에서 실측하면
    #       Z10 의 상대 변화가 1.45e-01 로 Z8(3.89e-01) 다음으로 크고
    #       Z6(1.10e-01)보다 오히려 크다. '작아서 뺀다' 는 틀린 말이다.
    #       실제 이유는 스키마 호환이다. CONDITION_KEYS 를 바꾸면
    #       merge_dataset 이 조건 열 구성을 검사해 기존 데이터셋과의 병합을
    #       거부한다. Z9·Z10 은 애초에 스키마에 없고, 지금 넣으면 이미
    #       만든 9 만 건을 못 쓰게 된다.
    #   (2) 본 과제의 결론이 걸린 성질을 바꾸지 않는다. verify.py C5 에서
    #       Z5~Z11 을 전수 분류한 결과, 초점 부호 모호성을 깨는 것은
    #       rho^2 성분을 품은 Z6 과 Z11 뿐이고 Z9·Z10 은 짝함수성을
    #       유지한다. 즉 넣어도 '부호를 되짚을 수 있게' 되지 않는다.
    # Z4(초점)는 defocus 인자가 물리적으로 처리하므로 여기 넣지 않는다.
    "z5", "z6", "z7", "z8", "z11",
]
#: 라벨과 **단위**. 단위를 적어 두지 않으면 학습 결과를 보고할 때 그대로
#: 틀린다. 특히 `e_op` 는 이름이 업계의 E_op(mJ/cm^2)와 같아 보이지만
#: **무차원 배율**이다. 아래를 반드시 함께 읽을 것.
#:
#:   cd_mean             [nm]  확률 경로의 CD 평균
#:   cd_3sigma           [nm]  확률 경로의 CD 3시그마
#:   ler_3sigma          [nm]  선 가장자리 거칠기 3시그마
#:   contrast            [무차원] (Imax-Imin)/(Imax+Imin). 스케일 불변
#:   nils                [무차원] CD x |d(ln I)/dx| at edge
#:   e_op                [무차원] <-- 주의. mJ/cm^2 가 아니다.
#:                       최적 초점에서 목표 CD 를 만드는 **광강도 배율**이다
#:                       (dose_to_size 의 스윕 범위가 0.2~5.0 인 것이 그
#:                       증거다). 물리 노광량으로 환산하려면 레지스트
#:                       감도가 필요하고, 본 모형은 상수 임계값 모형이라
#:                       감도를 따로 두지 않는다. 절대 노광량이 필요하면
#:                       `dose_clear_mj` 를 쓸 것.
#:   unresolved_row_frac [무차원] 경계를 찾지 못한 행의 비율. 결함률이
#:                       아니다(측정 한계는 아래 주석 참조)
#:   cd_deterministic    [nm]  잡음 없는 경로의 CD
#:   dose_clear_mj       [mJ/cm^2] threshold x dose_mj. 상수 임계값 모형에서
#:                       둘은 정확히 축퇴이므로 이 곱만이 의미를 갖는다
LABEL_KEYS = [
    "cd_mean", "cd_3sigma", "ler_3sigma", "contrast", "nils",
    "e_op", "unresolved_row_frac", "cd_deterministic", "dose_clear_mj",
]


@dataclass
class SampleSpec:
    """한 건의 샘플을 만드는 데 필요한 조건."""
    wavelength: float = 193.0
    na: float = 1.35
    n_medium: float = 1.44
    sigma_out: float = 0.7
    sigma_in: float = 0.0
    pitch: float = 160.0
    duty: float = 0.5
    defocus: float = 0.0
    dose_rel: float = 1.0        # E_op 대비 배율
    dose_mj: float = 30.0        # 광자 통계용 절대 노광량
    blur_nm: float = 8.0         # stochastic.BLUR_DEFAULT_ARF
    # --- 10주차 추가 ---
    polarization: str = IM.POL_TE      # 고NA 조밀 패턴의 실사용 조건
    source_shape: str = IM.SOURCE_CONVENTIONAL
    pole_offset: float = 0.0           # 0 이면 sigma_out 으로 자동 설정
    pole_radius: float = 0.15
    mask_type: str = MK.MASK_BINARY
    att_t: float = MK.ATT_PSM_DEFAULT_T
    n_resist: float = 1.70             # 편광 가중 계산용 굴절률
    flare: float = 0.0                 # 투영계 산란광 비율
    msd_nm: float = 0.0                # 스캔 동기화 오차 [nm, 1-sigma]
    z5: float = 0.0                    # 비점수차 45도 [waves]
    z6: float = 0.0                    # 비점수차 0도
    z7: float = 0.0                    # 코마 y
    z8: float = 0.0                    # 코마 x
    z11: float = 0.0                   # 구면수차
    # 수치 설정
    n_periods: int = 6
    dx_target: float = 0.5
    n_source_grid: int = 31
    threshold: float = 0.30
    n_rows: int = 1024
    dy: float | None = None
    n_trials: int = 8
    n_profile: int = 64
    seed: int = 0
    # 품질 기준 (학습 데이터 적격 판정)
    #
    # [10주차 보정] 기존에는 min_nils = 2.0 과 cd_tolerance = 0.10 을 서로
    # 독립인 '업계 표준' 두 개로 적어 두었다. 그런데 이 둘은 **같은 CD 공차
    # 예산을 나눠 쓰는 항목**이다.
    #   - NILS 기준은 Mack 근사 EL[%] ~ NILS x (CD 공차[%]) 에서 나온다.
    #     즉 결정론적 공정 윈도가 CD 공차 안에 들어가기 위한 조건이다.
    #   - cd_tolerance 는 확률적 변동(CD 3sigma / CD)의 상한이다.
    # 전체 CD 공차가 10 % 인데 공정 윈도에 10 %, 확률 변동에 또 10 % 를 주면
    # 합이 공차를 넘는다. 두 기여는 독립이므로 제곱합으로 더해야 한다.
    #     cd_budget^2 = pw_tol^2 + stoch_tol^2
    # 따라서 전체 예산과 배분 비율을 명시하고 두 기준을 거기서 유도한다.
    cd_budget: float = 0.10      # 목표 CD 대비 전체 허용 변동 (상대값)
    budget_split: float = 0.5    # 확률 변동에 배분하는 '분산' 비율 (0~1)
    el_required: float = 15.0    # 요구 노광 여유도 [%] (공정 윈도 기준)
    #: 명시하면 유도값을 덮어쓴다. None 이면 위 예산에서 계산한다.
    min_nils_override: float | None = None
    cd_tolerance_override: float | None = None

    # --- 파생 값 -----------------------------------------------------------
    @property
    def stoch_tol(self) -> float:
        """확률 변동(CD 3sigma / CD)의 상한. 예산의 제곱합 배분에서 나온다."""
        return float(self.cd_budget * np.sqrt(max(self.budget_split, 0.0)))

    @property
    def pw_tol(self) -> float:
        """결정론적 공정 윈도에 허용되는 CD 변동. 나머지 예산이다."""
        return float(self.cd_budget * np.sqrt(max(1.0 - self.budget_split, 0.0)))

    @property
    def cd_tolerance(self) -> float:
        if self.cd_tolerance_override is not None:
            return float(self.cd_tolerance_override)
        return self.stoch_tol

    @property
    def min_nils(self) -> float:
        """요구 노광 여유도를 만족하는 NILS 하한.

        Mack 근사 EL[%] ~ NILS x (CD 공차[%]) 를 뒤집으면
            NILS >= EL_required / (100 * pw_tol)
        이다. 본 시뮬레이터에서 이 관계의 기울기를 직접 측정하면 9.13
        (상관계수 0.9995) 로, 문헌 근사 10 과 어긋남이 작다
        (validate_w9 Case 1). 기본값 예산(전체 10 %, 절반 배분)과
        EL 15 % 에서 NILS 하한은 약 2.1 이 되어, 기존에 근거 없이 쓰던
        2.0 과 사실상 같은 수준이면서 유도 과정이 명시된다.
        """
        if self.min_nils_override is not None:
            return float(self.min_nils_override)
        pw = self.pw_tol
        if pw <= 0.0:
            return float("inf")
        return float(self.el_required / (100.0 * pw))

    @property
    def pol_code(self) -> int:
        return POLARIZATION_CODES.index(self.polarization)

    @property
    def source_code(self) -> int:
        return SOURCE_CODES.index(self.source_shape)

    @property
    def mask_code(self) -> int:
        return MASK_CODES.index(self.mask_type)

    def aberrations(self) -> dict[int, float]:
        """0 이 아닌 Zernike 항만 모아 돌려준다."""
        d = {5: self.z5, 6: self.z6, 7: self.z7, 8: self.z8, 11: self.z11}
        return {k: v for k, v in d.items() if v != 0.0}

    def optics(self) -> dict:
        """aerial_image 에 넘길 광학 인자 묶음."""
        return dict(
            n_medium=self.n_medium, sigma_out=self.sigma_out,
            sigma_in=self.sigma_in, n_source_grid=self.n_source_grid,
            polarization=self.polarization, source_shape=self.source_shape,
            pole_offset=self.pole_offset, pole_radius=self.pole_radius,
            aberrations=self.aberrations(), flare=self.flare,
            msd_nm=self.msd_nm, n_resist=self.n_resist,
        )

    def condition_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in CONDITION_KEYS], dtype=np.float64)


def downsample_profile(x: np.ndarray, profile: np.ndarray, pitch: float,
                       n_profile: int) -> np.ndarray:
    """필드 중앙의 한 pitch 구간을 n_profile 점으로 재샘플링한다."""
    xc = x[len(x) // 2]
    xs = np.linspace(xc - pitch / 2.0, xc + pitch / 2.0, n_profile, endpoint=False)
    return np.interp(xs, x, profile)


def make_sample(spec: SampleSpec) -> dict | None:
    """조건 하나에 대해 (조건, 관측, 라벨) 한 건을 생성한다.

    목표 CD 를 구현하는 Dose 가 없으면 None 을 반환한다.
    """
    g = G.build_grid(pitch=spec.pitch, n_periods=spec.n_periods,
                     dx_target=spec.dx_target)
    m = MK.line_space(g, cd=spec.duty * spec.pitch,
                      mask_type=spec.mask_type, att_t=spec.att_t)
    target = spec.duty * spec.pitch
    opt = spec.optics()

    # 최적 초점에서 dose-to-size 보정 배율을 먼저 구한다
    i_focus = IM.aerial_image(g, m, spec.wavelength, spec.na, defocus=0.0, **opt)
    e_op = PR.dose_to_size(g, i_focus, target, spec.threshold,
                           max_width=spec.pitch)
    if not np.isfinite(e_op):
        return None

    # 해당 Defocus 에서의 결상
    intensity = (
        i_focus if spec.defocus == 0.0 else
        IM.aerial_image(g, m, spec.wavelength, spec.na,
                        defocus=spec.defocus, **opt)
    )
    observed = intensity * e_op * spec.dose_rel

    det = RS.extract_cd(g.x, observed, spec.threshold, max_width=spec.pitch)
    mc = ST.monte_carlo(g.x, observed, g.dx, spec.dose_mj, spec.wavelength,
                        spec.threshold, n_rows=spec.n_rows, blur_nm=spec.blur_nm,
                        n_trials=spec.n_trials, seed=spec.seed, dy=spec.dy,
                        max_width=spec.pitch)

    # 극값을 대역제한 보간에서 찾아 격자 의존성을 없앤다
    contrast = IM.contrast(intensity, g.x)
    cd_var = (mc["cd_3sigma"] / mc["cd_mean"]
              if mc["formed"] and mc["cd_mean"] > 0 else np.inf)

    labels = {
        "cd_mean": mc["cd_mean"],
        "cd_3sigma": mc["cd_3sigma"],
        "ler_3sigma": mc["ler_3sigma"],
        "contrast": contrast,
        "nils": det["nils"],
        "e_op": e_op,
        # [10주차 정정] 기존 이름 'defect_rate' 는 측정 대상을 잘못 말한다.
        # 이 값은 '확률적 결함 발생률' 이 아니라 '중앙 feature 의 경계를
        # 찾지 못한 행의 비율' 이다. 실측 분포도 연속이 아니라 이봉이다
        # (0 이거나 약 0.99). NILS 가 0 에 가까워 패턴이 아예 형성되지
        # 않을 때만 켜지기 때문이다.
        #
        # [측정 한계] 직접 몬테카를로의 분해 한계는
        #     1 / (n_rows x n_trials) ~ 1e-4
        # 이다. 반면 공정에서 문제가 되는 확률적 결함률은 1e-9~1e-12
        # 수준이므로, 이 방식으로는 원리적으로 측정할 수 없다. 희소 사건
        # 추정(중요도 표본화, 극단값 이론)이 필요하며 본 과제 범위 밖이다.
        #
        # 따라서 이 라벨을 학습 목표로 쓰면 안 된다. 거의 전부 0 이므로
        # '전부 0' 이라고 답하는 모형도 99.9 % 정확도가 나와, 정확도 지표
        # 자체가 의미를 잃는다.
        "unresolved_row_frac": mc["defect_rate"],
        "cd_deterministic": det["cd"] if det["formed"] else np.nan,
        # [10주차 추가] 확률 거동을 좌우하는 물리량은 dose_mj 가 아니라
        # 'threshold x dose_mj' 다. 경계에서 I*e_op = threshold 이므로
        # 경계의 기대 광자 수가 N_ref(dose_mj) x threshold 이고, LER 은
        # 그 제곱근에 반비례한다. 실제로 threshold 와 dose_mj 를 곱이
        # 같도록 바꾸면 LER 이 완전히 동일하다(상대편차 0.000 %).
        # 즉 threshold 와 dose_mj 는 확률 경로에서 정확히 축퇴이며, 물리적
        # 으로 의미 있는 양은 그 곱 = 실효 dose-to-clear 다.
        # 'dose 30 mJ/cm^2 에서 LER' 이라고만 적으면 threshold 를 모르는
        # 독자가 문헌값과 비교할 때 광자 수를 잘못 환산한다.
        "dose_clear_mj": float(spec.threshold * spec.dose_mj),
    }
    # 예산 결합 기준 : 확률 변동과 공정 윈도 기여를 제곱합으로 더해 전체
    # CD 공차와 비교한다.
    #
    # [성격] 제곱합으로 배분했으므로 두 기준이 각각 경계일 때 결합값이
    # 정확히 cd_budget 과 같다(실측 차이 0). 따라서 nils_ok 와 cdu_ok 가
    # 모두 통과하면 budget_ok 는 반드시 통과한다. 즉 이것은 '추가 필터' 가
    # 아니라 **배분이 어긋나지 않았는지 보는 정합성 감시 장치** 다.
    # (배분 규칙을 제곱합이 아닌 것으로 바꾸거나 override 로 두 기준을
    #  따로 지정하면 독립적인 의미를 갖는다.)
    nils = det["nils"] if np.isfinite(det["nils"]) else 0.0
    pw_used = (spec.el_required / (100.0 * nils)) if nils > 0 else np.inf
    combined = float(np.hypot(cd_var, pw_used)) if np.isfinite(cd_var) else np.inf
    quality = {
        "formed": bool(det["formed"]) and bool(mc["formed"]),
        "nils_ok": bool(np.isfinite(det["nils"]) and det["nils"] >= spec.min_nils),
        "cdu_ok": bool(cd_var <= spec.cd_tolerance),
        "budget_ok": bool(combined <= spec.cd_budget),
    }
    return {
        "condition": spec.condition_vector(),
        "profile": downsample_profile(g.x, observed, spec.pitch, spec.n_profile),
        "labels": np.array([labels[k] for k in LABEL_KEYS], dtype=np.float64),
        "target_cd": target,
        "ok": all(quality.values()),
        "quality": quality,
        "cd_variation": float(cd_var) if np.isfinite(cd_var) else np.nan,
    }


def generate(specs: list[SampleSpec], verbose: bool = False,
             keep_invalid: bool = False) -> dict:
    """여러 조건에 대해 데이터셋을 생성한다.

    keep_invalid 가 False 면 품질 기준에 미달하는 샘플을 제외한다.

    품질 기준은 업계에서 통용되는 지표를 사용한다.
      - formed   : 패턴 경계가 검출되고 CD 가 pitch 이내일 것
      - nils_ok  : NILS >= min_nils. 이 하한은 근거 없이 고른 값이 아니라
                   요구 노광 여유도에서 유도한다. Mack 근사
                   EL[%] ~ NILS x (CD 공차[%]) 를 뒤집으면
                   NILS >= EL_required / (100 x pw_tol) 이다.
                   기본 예산(0.10, 절반 배분)과 EL 15 % 에서 2.12 가 되어,
                   기존에 '업계 표준' 으로 적어 두었던 2.0 과 사실상 같은
                   수준이면서 유도 과정이 드러난다. 본 시뮬레이터에서 이
                   관계의 기울기를 직접 측정하면 9.13(상관계수 0.9995)으로
                   문헌 근사 10 과 어긋남이 작다(validate_w9 Case 1).
      - cdu_ok   : CD 3σ / CD <= stoch_tol. 전체 CD 공차에서 확률 변동에
                   배분된 몫이다(기본 예산 0.10, 절반 배분 -> 0.0707).
      - budget_ok: 확률 변동과 공정 윈도 기여를 제곱합으로 더한 값이 전체
                   CD 공차 이내인가.
                   [성격] 이것은 추가 게이트가 아니라 **배분 정합성
                   불변량**이다. RSS 배분에서는 두 성분 기준을 각각
                   통과하면 제곱합이 예산 이내임이 수학적으로 보장되므로,
                   budget_ok 가 단독으로 실패할 수 없다(무작위 60 조건에서
                   0 건 확인). 단독 실패가 나오면 배분 논리가 깨진 것이다.
                   따라서 보고할 때 '세 기준을 통과' 가 아니라 '두 성분
                   기준 + 배분 정합성 확인' 으로 적어야 한다.

    NILS 가 낮은 조건에서는 임계값 교차점이 잡음에 민감해져 CD 가
    산출되더라도 변동이 커진다. 이런 샘플을 학습에 포함하면 라벨 잡음이
    되므로 제외한다. 조건 공간의 어느 영역이 구현 불가인지 파악하려면
    True 로 두고 ok 플래그를 활용한다.
    """
    conds, profs, labs, oks, tgts = [], [], [], [], []
    # [주의] 품질 플래그를 추가할 때 이 딕셔너리를 같이 고치지 않으면
    # 그 플래그가 먼저 실패하는 순간 KeyError 로 죽는다. 하드코딩하지 않고
    # 집계할 때 키를 만든다.
    from collections import defaultdict
    reject = defaultdict(int)
    for i, sp in enumerate(specs):
        r = make_sample(sp)
        if r is None:
            continue
        if not r["ok"]:
            for k, v in r["quality"].items():
                if not v:
                    reject[k] += 1
                    break
        if not keep_invalid and not r["ok"]:
            continue
        conds.append(r["condition"])
        profs.append(r["profile"])
        labs.append(r["labels"])
        oks.append(r["ok"])
        tgts.append(r["target_cd"])
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(specs)} 생성")
    return {
        "conditions": np.array(conds),
        "profiles": np.array(profs),
        "labels": np.array(labs),
        "ok": np.array(oks),
        "target_cd": np.array(tgts),
        "condition_keys": CONDITION_KEYS,
        "label_keys": LABEL_KEYS,
        "reject": reject,
    }


def save(path: str, data: dict) -> None:
    np.savez_compressed(
        path,
        conditions=data["conditions"], profiles=data["profiles"],
        labels=data["labels"], ok=data["ok"], target_cd=data["target_cd"],
        condition_keys=np.array(data["condition_keys"]),
        label_keys=np.array(data["label_keys"]),
    )


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def estimate_size(n_samples: int, n_profile: int) -> dict:
    """저장 용량 추산 [MB]. float64 기준."""
    per = (len(CONDITION_KEYS) + len(LABEL_KEYS) + n_profile + 2) * 8
    return {
        "bytes_per_sample": per,
        "total_mb": n_samples * per / 1024 ** 2,
    }
