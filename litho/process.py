"""
process.py -- Dose-Focus 스윕 및 공정 윈도(Process Window) 분석

연산량 설계
-----------
Dose 는 정규화된 공간 광강도에 곱하는 스칼라이므로, 결상 연산(FFT)은
Defocus 에만 의존한다. 따라서 N_dose x N_focus 격자를 스윕할 때
결상 연산은 N_focus 회만 수행하고 Dose 는 후처리로 적용한다.
21 x 21 격자 기준 441회 -> 21회로 약 95% 절감된다.

용어
----
Bossung curve : 고정 Dose 에서 Defocus 에 대한 CD 곡선. 여러 Dose 를
                겹쳐 그리면 공정 민감도가 한눈에 보인다.
Process Window: CD 가 규격(목표 +- 허용오차) 안에 드는 (Dose, Focus) 영역.
EL            : Exposure Latitude. 규격을 지키는 Dose 허용 폭 [%].
DOF           : Depth of Focus. 규격을 지키는 Focus 허용 폭 [nm].
EL-DOF 곡선   : PW 안에 들어가는 축정렬 직사각형 중 각 EL 에 대한 최대 DOF.
                EL 을 키우면 DOF 가 줄어드는 trade-off 를 정량화한다.
"""

import numpy as np

from . import imaging as IM
from . import mask as MK
from . import resist as RS


def focus_sweep(
    grid,
    mask,
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    sigma_out: float = 0.7,
    sigma_in: float = 0.0,
    defocuses: np.ndarray | None = None,
    n_source_grid: int = 31,
    **optics,
) -> np.ndarray:
    """Defocus 별 정규화 공간 광강도. shape (n_focus, n_points).

    optics 로 polarization, source_shape, aberrations, flare, msd_nm,
    n_resist 등 aerial_image 의 광학 인자를 그대로 전달한다.
    """
    if defocuses is None:
        defocuses = np.linspace(-200.0, 200.0, 21)
    return np.array([
        IM.aerial_image(
            grid, mask, wavelength, na,
            n_medium=n_medium, sigma_out=sigma_out, sigma_in=sigma_in,
            defocus=float(dz), n_source_grid=n_source_grid, **optics,
        )
        for dz in defocuses
    ])


def cd_matrix(
    grid,
    intensities: np.ndarray,
    doses: np.ndarray,
    threshold: float,
    tone: str = RS.TONE_BRIGHT,
    max_width: float | None = None,
) -> np.ndarray:
    """CD[dose, focus] 행렬 [nm]. 패턴 미형성은 NaN.

    max_width(통상 pitch)를 주면 인접 패턴 병합을 미형성으로 처리한다.
    """
    out = np.full((len(doses), intensities.shape[0]), np.nan)
    for j in range(intensities.shape[0]):
        base = intensities[j]
        for i, dose in enumerate(doses):
            r = RS.extract_cd(grid.x, base * dose, threshold, tone,
                              max_width=max_width)
            if r["formed"] and np.isfinite(r["cd"]):
                out[i, j] = r["cd"]
    return out


def process_window(cd_mat: np.ndarray, cd_target: float, tol_frac: float = 0.10) -> np.ndarray:
    """CD 규격을 만족하는 (Dose, Focus) 영역을 bool 행렬로 반환."""
    lo, hi = cd_target * (1 - tol_frac), cd_target * (1 + tol_frac)
    with np.errstate(invalid="ignore"):
        return (cd_mat >= lo) & (cd_mat <= hi)


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    """연속 True 구간 중 가장 긴 것의 (시작, 길이)."""
    best_start, best_len = 0, 0
    start, length = 0, 0
    for i, f in enumerate(flags):
        if f:
            if length == 0:
                start = i
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            length = 0
    return best_start, best_len


def el_dof_curve(
    pw: np.ndarray,
    doses: np.ndarray,
    defocuses: np.ndarray,
) -> list[dict]:
    """PW 안에 들어가는 축정렬 직사각형의 EL-DOF trade-off 곡선.

    Dose 방향 창 높이를 1칸씩 키우며, 그 창 전체가 규격을 만족하는
    Focus 연속 구간이 가장 긴 위치를 찾는다.

    [격자화 편향 — 반드시 함께 읽을 것]
    DOF 를 '통과한 격자점 사이의 거리' 로 재므로, 실제 경계가 마지막 통과점과
    그 다음 실패점 사이에 있을 때 **항상 작게** 나온다. 편향의 크기는 0 에서
    양쪽 각각 한 격자 간격까지, 즉 최대 2 간격이다. 합성 PW 로 측정한 결과
    (격자 25 nm):

        참 DOF   400.0  410.0  420.0  440.0  449.8
        보고     400.0  400.0  400.0  400.0  400.0
        과소     0.0    10.0   20.0   40.0   49.8

    n_focus=33 · 범위 +-400 nm 면 간격이 25 nm 이므로 DOF 가 0~50 nm 작게
    나온다. DOF 가 200 nm 수준이면 상대오차 25 % 다. 공정 윈도를 과대
    약속하지 않는 **안전한 방향**이지만, 크기를 모르면 숫자를 잘못 읽는다.

    [이 '항상 작게' 주장이 성립하는 조건 — 12 차 보완 66 번]
    위 주장은 **격자점에서 통과하면 그 사이에서도 통과한다**는 가정에
    의존한다. 통과 영역이 초점 방향으로 하나의 구간일 때만 참이다.

    그런데 노광량이 높아 최적 초점의 CD 가 규격 **상한**을 넘는 dose 행에서는
    통과 영역이 **고리 모양**이 된다. Bossung 이 dz=0 에서 극값이므로
    중심이 규격 초과이고 그 바깥 띠가 통과하기 때문이다. 중심 금지대가 격자
    간격보다 얇으면 격자는 그것을 건너뛰고 양쪽을 하나의 연속 통과 구간으로
    읽는다. 그러면 DOF 가 **과대**평가된다. 방향이 뒤집히는 것이다.

    실측(해석 Bossung, S=1.36, tol 7 %) : 초점 32 점 격자가 DOF 500.0 nm 를
    보고했고 그 직사각형 안에 규격 밖 점이 1385 개 있었다. 참 윈도는 고리
    한쪽 띠뿐이라 약 235.9 nm 이므로 2.12 배 과대보고다. 같은 조건에서
    33 점은 0 nm 였다. 숫자는 validate_w12 Case 11 이 매 회차 재측정한다.

    차이는 홀짝뿐이다. **홀수 격자는 dz=0 을 반드시 표본에 넣으므로** 금지대
    중심을 항상 밟는다. 그래서 초점 격자는 홀수로 쓴다. `recommend.rank` 가
    짝수를 받으면 1 을 올리고 보고하며, GUI 의 분할 수 선택지도 모두 홀수다.
    노광 방향은 CD 가 dose 에 단조여서 내부 극값이 없으므로 이 문제가 없다
    (병합은 미형성으로 처리되어 통과하지 않는다).

    그래서 각 항목에 `dof_upper` / `el_upper` 를 함께 돌려준다. 참값은
        dof <= 참 DOF <= dof_upper
    구간에 있다 (위 홀수 조건 아래에서). 보간으로 경계를 추정하지 않는 이유는, PW 가 이미 bool 로
    축약되어 경계 근방의 CD 값을 잃었기 때문이다. 더 정확한 값이 필요하면
    격자를 촘촘히 하는 것이 정직하다(간격이 줄면 구간도 같이 좁아진다).
    """
    n_dose = len(doses)
    # 격자 간격. 비균일 격자도 있을 수 있으므로 최대 간격을 보수적으로 쓴다.
    step_dz = float(np.max(np.diff(defocuses))) if len(defocuses) > 1 else 0.0
    step_dr = float(np.max(np.diff(doses))) if len(doses) > 1 else 0.0
    curve = []
    for h in range(1, n_dose + 1):
        best = None
        for i0 in range(n_dose - h + 1):
            band = pw[i0:i0 + h, :].all(axis=0)
            start, run = _longest_run(band)
            if run < 1:
                continue
            dof = defocuses[start + run - 1] - defocuses[start] if run > 1 else 0.0
            if best is None or dof > best["dof"]:
                d_lo, d_hi = doses[i0], doses[i0 + h - 1]
                d_mid = 0.5 * (d_lo + d_hi)
                el = 100.0 * (d_hi - d_lo) / d_mid if d_mid > 0 else 0.0
                best = {
                    "el": el,
                    "dof": float(dof),
                    # 격자화 편향의 상한. 참값은 [dof, dof_upper] 안에 있다.
                    "dof_upper": float(dof + 2.0 * step_dz),
                    "el_upper": (100.0 * (d_hi - d_lo + 2.0 * step_dr) / d_mid
                                 if d_mid > 0 else 0.0),
                    "grid_step_dz": step_dz,
                    "grid_step_dose": step_dr,
                    "dose_range": (float(d_lo), float(d_hi)),
                    "focus_range": (float(defocuses[start]), float(defocuses[start + run - 1])),
                }
        if best is not None and best["dof"] > 0:
            curve.append(best)
    # EL 오름차순, 동일 EL 은 DOF 큰 것만 남김
    curve.sort(key=lambda r: r["el"])
    dedup = []
    for r in curve:
        if dedup and abs(dedup[-1]["el"] - r["el"]) < 1e-9:
            if r["dof"] > dedup[-1]["dof"]:
                dedup[-1] = r
        else:
            dedup.append(r)
    return dedup


def dof_at_el(curve: list[dict], el_target: float = 10.0) -> float:
    """목표 EL 에서의 DOF [nm]. 곡선을 선형보간한다.

    업계 관례상 공정 윈도는 EL 10 % 에서의 DOF 로 표기하는 경우가 많으므로
    기본값을 10 % 로 둔다.

    [이 함수를 새로 쓰려면 먼저 `dof_bracket_at_el` 을 볼 것]
    여기서 돌려주는 값은 '보간한 하한' 이며 두 가지 한계가 있다.

      1. 노광 격자의 EL 편향(`el_upper`)을 담지 않는다. 그래서 상한을 따로
         만들어 붙이면 구간이 참값을 놓친다(12 차 보완 65 번).
      2. `el_target` 과의 비교에 허용오차가 없다. 밴드 높이가 목표 EL 을
         **정확히** 주는 격자에서 계산값이 9.999999999999988 처럼 나오면
         경계 분기가 반대로 타므로 값이 갑자기 0 이 된다.

    고치지 않고 남겨 둔 이유는 회귀 66 종과 claims 37 항목이 이 함수의 현
    거동에 물려 있고, 이 함수를 '값' 으로 쓰는 곳이 더 이상 없기 때문이다
    (추천기와 GUI 는 `dof_bracket_at_el` 을 쓴다. `validate_w6` 는 단조성만
    보고, `validate_w12` Case 8 은 옛 거동을 **일부러** 재현한다).
    새 코드는 `dof_bracket_at_el` 을 쓸 것.
    """
    if not curve:
        return 0.0
    els = np.array([r["el"] for r in curve])
    dofs = np.array([r["dof"] for r in curve])
    if el_target <= els[0]:
        return float(dofs[0])
    if el_target >= els[-1]:
        return 0.0
    return float(np.interp(el_target, els, dofs))


def dof_bracket_at_el(curve: list[dict],
                      el_target: float = 10.0) -> tuple[float, float]:
    """목표 EL 에서의 DOF 를 **증명 가능한 구간** [하한, 상한] 으로 돌려준다.

    왜 `dof_at_el` 로 충분하지 않은가
    ---------------------------------
    `el_dof_curve` 의 격자화 편향은 **두 방향**에서 온다.

      - 초점 격자 : DOF 를 통과점 사이 거리로 재므로 최대 2 간격 작게 나온다.
        이것은 각 항목의 `dof_upper` 에 담겨 있다.
      - 노광 격자 : EL 을 통과한 dose 밴드의 높이로 재므로 EL 도 최대 2 간격
        작게 나온다. 이것은 `el_upper` 에 담겨 있다.

    12 주차에 실제로 난 사고가 이것이다. 추천기는 `dof_at_el` 로 하한을 읽고
    상한은 `dof_upper` 를 **EL 축에서 보간**해 썼다. 그러면 노광 격자의 편향이
    구간에서 통째로 빠진다. EL-DOF 곡선의 무릎 근처에서는 dDOF/dEL 이 커서
    EL 편향 1.9 % 가 DOF 100 nm 이상으로 번지는데, 초점항은 25 nm 였다. 즉
    **두 오차원 가운데 큰 쪽을 누락**했다.

    CD = T(1 + s(dose-1) - K dz^2), s = 1.36, K = 1.2e-6, tol 7 % 로 측정한
    결과다. 참값은 **해석해**다. 이 꼴이면 EL 을 고정한 최대 DOF 를 손으로 풀
    수 있다(유도는 `validate_w12` Case 8). EL 10 % 에서 112.69 nm 다.

        격자(dose x focus)     33      65     129     257  513x257
        새 하한               0.0     0.0    35.2    66.4    97.7
        새 상한             281.2   156.2   148.4   125.0   125.0
        담음                  예      예      예      예      예
        옛 하한               0.0     0.0    52.6    86.3   100.4
        옛 상한               0.0     0.0    60.4    90.2   104.3
        담음                아니오  아니오  아니오  아니오  아니오

    참값을 격자로 만들지 않은 것이 중요하다. 처음에 격자 1025 의 값(108.2)을
    참값이라 적었더니 옛 방식이 그 격자에서는 통과하는 것처럼 보였다. 같은
    격자로 만든 값을 그 격자의 기준으로 쓰는 순환 논증이었다. 해석해로 바꾸면
    옛 방식은 이 영역의 **모든** 격자에서 깨진다.

    33 x 33 은 추천기의 기본 격자다. 거기서 옛 구간이 [0.0, 0.0] 으로 나와
    "참값은 구간 안에 있다" 는 문서의 주장이 **거짓**이었다. 그리고 이것은
    보고만의 문제가 아니다. 하한 0 은 순위의 정렬 키이므로, 참 DOF 가 112 nm
    인 후보가 윈도가 아예 없는 후보와 구별되지 않고 탈락한다.

    세 영역 x 다섯 격자 15 조합에서 새 구간은 0 개, 옛 방식은 6 개를 놓쳤다.
    옛 방식은 무릎(EL 10 % 가 곡선 끝에 걸림)뿐 아니라 여유 영역의 65x65
    에서도 깨진다. 내부 보간이라서 안전하다고 본 것이 잘못이었다.

    규칙
    ----
    각 항목 i 에 대해 참 EL 은 [el_i, el_upper_i] 안, 참 DOF 는
    [dof_i, dof_upper_i] 안에 있다. EL-DOF 는 감소 관계이므로

      하한 = max{ dof_i      : el_i       >= el_target }   (없으면 0)
      상한 = max{ dof_upper_i : el_upper_i >= el_target }   (없으면 0)

    하한에는 EL 의 **비관적** 끝(el_i)을, 상한에는 **낙관적** 끝(el_upper_i)을
    쓴다. el_i >= el_target 이면 el_upper_i >= el_target 이고 dof_i <=
    dof_upper_i 이므로 하한 <= 상한 이 항상 성립한다.

    보간하지 않는 이유는 `dof_at_el` 의 보간이 하한으로서 **증명되지 않기**
    때문이다. 참 DOF(el_target) >= 참 DOF(els[i+1]) >= dof_{i+1} 은 말할 수
    있지만, 보간값은 dof_{i+1} 보다 클 수 있다. 계단으로 내려가는 쪽이 느슨해
    보여도 참이다.

    [범위] 이 구간은 **격자화**에서 오는 불확실성만 담는다. 대리 모형의 예측
    오차나 물리 모형의 타당성은 담지 않는다. 그 둘은 각각 추천기의 보수 공차와
    VERIFICATION_LIMITS 0 절이 다룬다.
    """
    if not curve:
        return 0.0, 0.0
    # EL 비교에 허용오차를 둔다. 이것 없이 `>=` 로 비교하면 **정확히 목표
    # EL 을 주는 격자**에서 유효한 행이 탈락한다. 실측한 예 : 노광 점 25·49·
    # 121 개 격자에서 밴드 높이가 EL 10 % 를 정확히 주는데, 계산된 값이
    # 9.999999999999988 (목표 - 1.24e-14) 이어서 `el >= 10.0` 이 거짓이 되고
    # 하한이 62.5 nm 에서 0.0 으로 떨어졌다. 이 함수가 막으려는 결함(유효한
    # 후보가 조용히 탈락)을 부동소수 비교로 되살리는 셈이다.
    #
    # 허용오차를 두면 '목표보다 무한소만큼 낮은' 행을 받아들이게 되지만,
    # 1e-9 상대오차는 물리적으로 무의미하고 유효한 행을 버리는 쪽이 훨씬
    # 나쁘다. 격자 간격은 보통 0.1 % EL 이상이므로 경계가 흐려질 일은 없다.
    d = dof_bracket_detail(curve, el_target)
    return d["lo"], d["hi"]


def dof_bracket_detail(curve: list[dict],
                       el_target: float = 10.0) -> dict:
    """`dof_bracket_at_el` 과 같은 판정에 **어느 행이 이겼는지**를 더한다.

    왜 따로 두는가 : 추천기가 운전점(노광량·초점 오프셋)을 내놓으려면 하한을
    만든 직사각형의 `dose_range` 와 `focus_range` 가 필요하다. 그 행을 호출부
    에서 다시 고르게 하면 **EL 비교 규칙이 두 곳에 생긴다**. 12 차 보완 65 번
    에서 되살아난 결함이 정확히 그것이었다(부동소수 등식). 그래서 판정은 여기
    한 곳에만 두고, `dof_bracket_at_el` 도 이 함수를 쓴다.

    Returns
    -------
    {"lo", "hi", "lo_row", "hi_row", "thr"} : 행은 이긴 항목(없으면 None).
    """
    if not curve:
        return {"lo": 0.0, "hi": 0.0, "lo_row": None, "hi_row": None,
                "thr": float(el_target)}
    # EL 비교에 허용오차를 둔다. 이것 없이 `>=` 로 비교하면 **정확히 목표
    # EL 을 주는 격자**에서 유효한 행이 탈락한다. 실측한 예 : 노광 점 25·49·
    # 121 개 격자에서 밴드 높이가 EL 10 % 를 정확히 주는데, 계산된 값이
    # 9.999999999999988 (목표 - 1.24e-14) 이어서 `el >= 10.0` 이 거짓이 되고
    # 하한이 62.5 nm 에서 0.0 으로 떨어졌다. 이 함수가 막으려는 결함(유효한
    # 후보가 조용히 탈락)을 부동소수 비교로 되살리는 셈이다.
    #
    # 허용오차를 두면 '목표보다 무한소만큼 낮은' 행을 받아들이게 되지만,
    # 1e-9 상대오차는 물리적으로 무의미하고 유효한 행을 버리는 쪽이 훨씬
    # 나쁘다. 격자 간격은 보통 0.1 % EL 이상이므로 경계가 흐려질 일은 없다.
    eps = 1e-9 * max(1.0, abs(float(el_target)))
    thr = float(el_target) - eps
    lo = 0.0
    hi = 0.0
    lo_row = None
    hi_row = None
    for r in curve:
        if r["el"] >= thr and float(r["dof"]) > lo:
            lo = float(r["dof"])
            lo_row = r
        if (r.get("el_upper", r["el"]) >= thr
                and float(r.get("dof_upper", r["dof"])) > hi):
            hi = float(r.get("dof_upper", r["dof"]))
            hi_row = r
    if hi < lo:            # 도달할 수 없다. 도달하면 규칙이 깨진 것이다.
        raise AssertionError(
            f"DOF 구간이 뒤집혔다 : 하한 {lo} > 상한 {hi}. el_target="
            f"{el_target}. dof <= dof_upper 와 el <= el_upper 가 깨졌다는 뜻이다.")
    return {"lo": lo, "hi": hi, "lo_row": lo_row, "hi_row": hi_row,
            "thr": thr}


def centered_focus_grid(lo: float, hi: float,
                        n: int) -> tuple[np.ndarray, str | None]:
    """dz=0 을 **반드시 격자점으로 포함하는** 균일 초점 격자.

    왜 필요한가 (12 차 보완 69 번)
    ------------------------------
    66 번에서 `n_focus` 를 홀수로 강제했다. 그 근거는 "Bossung 이 dz=0 에서
    극값을 가지므로 dz=0 이 격자점이어야 한다" 였다. 그런데 **홀수는 범위가
    대칭일 때만 그 전제를 세운다**. 추천기는 초점 격자를 학습 범위로 깎으므로
    (`clip_grid_to_domain`), 학습된 defocus 범위가 비대칭이면 홀수로도 dz=0 이
    격자점이 아니다. 실측 : `linspace(-183, 250, 33)` 의 min|dz| 는 6.438 nm,
    `n=41` 도 1.025 nm 이고 어느 쪽도 0 을 담지 않는다. 즉 **코드의 보장이
    주석의 주장보다 약했다**.

    그 틈이 실제로 해로웠는가
    -------------------------
    해석 Bossung 으로 조준해 측정한 결과 **해롭지 않았다**(규격이탈 0 점).
    까닭도 해석으로 설명된다. 통과집합이 환형(중심이 규격 밖)인데 격자가 그
    중심을 못 보고 하나로 병합하려면, 실패 중심 반폭
    `z1(d) = sqrt((s(d-1) - tol)/K)` 가 min|dz| 보다 좁아야 한다. 그 조건이
    성립하는 dose 구간의 폭은

        d in ( 1 + tol/s ,  1 + (tol + K*min|dz|^2)/s )   ->   폭 = K*min|dz|^2/s

    로 min|dz|^2 에 비례해 줄어든다. 중심에 거의 붙은 비대칭 격자에서는 이
    폭이 미시적이어서 dose 격자점이 거의 들어오지 않고, 들어와도 단일 행이라
    `el_upper = 200*step_dr/d_mid` 밖에 안 된다(기본 격자 2.4 %). EL 10 % 목표
    에는 못 미치므로 `dof_bracket_at_el` 이 걸러낸다. 짝수 격자가 위험했던
    이유가 여기서 분명해진다. min|dz| = 간격/2 로 **크고**, 중심이 아예 격자에
    없어 환형 행이 전부 병합되기 때문이다.

    그래도 구조로 닫는 이유는, 이 무해함이 **dose 격자가 촘촘하다는 조건에
    딸려 있기** 때문이다. `--n-dose 7` 처럼 거칠게 주면 단일 병합 행의
    `el_upper` 가 12.7 % 로 10 % 를 넘어 상한을 오염시킨다. 주석으로 지킬 수
    있는 성질이 아니다.

    무엇을 양보하는가
    -----------------
    간격을 유지하면서 0 을 격자점으로 만들려면 범위 끝을 최대 한 간격 포기
    해야 하고, 점수가 요청값보다 1 적어질 수 있다. 둘 다 **과소보고 방향**
    이다. 돌려주는 격자는 [lo, hi] 의 부분집합이므로 여기서 찾은 윈도는 전부
    진짜이고, 끝을 조금 버리면 DOF 가 작게 나올 뿐 크게 나오지 않는다.

    범위가 0 을 가로지르지 않으면(한쪽 스윕) 중심화할 것이 없으므로 평범한
    `linspace` 를 돌려준다. 이것은 결함이 아니다. DOF 는 폭이고, 한쪽 스윕은
    이미 폭을 작게 보고하는 쪽이다.

    Returns
    -------
    (격자, 메모) : 메모는 중심화 때문에 범위나 점수를 양보했을 때만 문자열이고
    그 밖에는 None 이다. 호출부가 사용자에게 그대로 보여 주면 된다.
    """
    lo = float(lo)
    hi = float(hi)
    if n < 2 or not (hi > lo):
        return np.linspace(lo, hi, max(int(n), 1)), None
    if not (lo < 0.0 < hi):
        return np.linspace(lo, hi, n), None

    step = (hi - lo) / (n - 1)
    # 1e-12 : 끝점이 간격의 정확한 배수일 때(대칭 범위 + 홀수 n) floor 가
    # 부동소수 때문에 한 점을 깎지 않도록 하는 여유다. 이 경우 결과는
    # linspace 와 정확히 같아야 한다.
    k_neg = int(np.floor(-lo / step + 1e-12))
    k_pos = int(np.floor(hi / step + 1e-12))
    if k_neg + k_pos + 1 < 2:
        # 간격이 범위의 한쪽보다 넓어 0 하나로 붕괴한다. 중심화를 포기하고
        # 원래 격자를 돌려준다. 점이 2 개뿐이면 DOF 자체가 의미 없다.
        return np.linspace(lo, hi, n), (
            f"초점 격자를 dz=0 중심으로 만들 수 없었다. 간격 {step:.3f} nm 가 "
            f"범위 [{lo:.1f}, {hi:.1f}] 에 비해 너무 넓다. n_focus 를 늘릴 것.")

    grid = step * np.arange(-k_neg, k_pos + 1, dtype=np.float64)
    grid[k_neg] = 0.0          # 부동소수 누적을 쓰지 않고 0 을 정확히 박는다
    lost_lo = abs(lo - float(grid[0]))
    lost_hi = abs(hi - float(grid[-1]))
    note = None
    if grid.size != n or lost_lo > 1e-9 or lost_hi > 1e-9:
        note = (f"초점 격자를 dz=0 이 격자점이 되도록 재배치했다 "
                f"(12차 보완 69 번). 점 {n} -> {grid.size} 개, 범위 "
                f"[{lo:.1f}, {hi:.1f}] -> [{grid[0]:.1f}, {grid[-1]:.1f}] nm. "
                f"버린 끝은 최대 한 간격({step:.2f} nm)이고 과소보고 방향이다.")
    return grid, note


def dose_to_size(
    grid,
    intensity_at_focus: np.ndarray,
    cd_target: float,
    threshold: float,
    tone: str = RS.TONE_BRIGHT,
    dose_lo: float = 0.2,
    dose_hi: float = 5.0,
    n_scan: int = 400,
    max_width: float | None = None,
) -> float:
    """최적 초점에서 목표 CD 를 만드는 **광강도 배율**(dose-to-size)을 구한다.

    [단위 주의] 돌려주는 값은 무차원 배율이며 mJ/cm^2 가 아니다. 이름이
    업계의 E_op(최적 노광량, mJ/cm^2)와 비슷해 혼동하기 쉬운데, 스윕 범위가
    dose_lo=0.2 ~ dose_hi=5.0 인 것에서 보이듯 '기준 세기의 몇 배인가' 다.
    물리 노광량으로 환산하려면 레지스트 감도가 필요하고, 본 모형은 상수
    임계값 모형이라 감도를 따로 두지 않는다. 절대 노광량이 필요하면
    threshold x dose_mj(= dose-to-clear)를 쓸 것.

    Dose 를 임의로 1.0 으로 두고 스윕하면 공정 윈도의 중심이 규격 밖에 놓여
    EL 과 DOF 가 과소평가되므로, 스윕 전에 반드시 보정해야 한다.

    CD 는 Dose 에 대해 단조 증가하지만, Dose 가 과도하면 인접 패턴이 병합되어
    CD 가 정의되지 않는다(NaN). 따라서 단순 이분법은 과노광 구간으로 발산할 수
    있으므로, 전 구간을 스캔하여 CD 가 목표를 통과하는 지점을 선형보간한다.
    """
    # [성능·정확도] CD 는 집합 {x : I(x) * d >= th} 로 정해지므로, 세기를
    # d 배 하는 것과 임계값을 th/d 로 바꾸는 것이 완전히 같다. NILS 도
    # d(ln I)/dx 라 배율에 무관하다. 따라서 프로파일을 매번 새로 만들지 않고
    # 대역제한 보간기를 한 번만 구성해 임계값만 바꾸면 결과는 동일하면서
    # 수백 배 싸다.
    bl = RS._BandLimited(grid.x, np.asarray(intensity_at_focus, dtype=float))

    def cd_of(d: float) -> float:
        if d <= 0.0:
            return float("nan")
        r = RS.extract_cd(grid.x, intensity_at_focus, threshold / d, tone,
                          max_width=max_width, _bl=bl)
        return r["cd"] if r["formed"] and np.isfinite(r["cd"]) else float("nan")

    # 탐색 격자는 조밀할 필요가 없다. 아래에서 이분법으로 조이므로, 목표를
    # 사이에 끼우는 구간 하나만 찾으면 된다. 400 점을 그대로 훑으면 선별
    # 단계가 전체 생성과 맞먹을 만큼 비싸진다(실측 740 ms/건).
    n_probe = max(24, min(n_scan, 64))
    doses = np.linspace(dose_lo, dose_hi, n_probe)
    cds = np.array([cd_of(float(d)) for d in doses])
    ok = np.isfinite(cds)
    if not ok.any():
        return float("nan")
    d_ok, c_ok = doses[ok], cds[ok]

    for i in range(len(d_ok) - 1):
        if (c_ok[i] - cd_target) * (c_ok[i + 1] - cd_target) <= 0.0:
            denom = c_ok[i + 1] - c_ok[i]
            if denom == 0.0:
                return float(d_ok[i])
            t = (cd_target - c_ok[i]) / denom
            d_lin = float(d_ok[i] + t * (d_ok[i + 1] - d_ok[i]))

            # [10주차 보정] 선형보간으로 끝내면 CD(dose) 의 곡률 때문에
            # O(delta_d^2) 오차가 남는다. 스캔 간격 0.012 에서 목표 CD 80 nm
            # 대비 0.5 nm (0.6 %) 오차가 났고, 이것이 iso-CD 등고선 위의 CD
            # 편차로 그대로 드러났다. 데이터셋 전체가 이 보정 노광량을 기준
            # 으로 생성되므로, 식별하려는 신호와 같은 크기의 계통 오차가 된다.
            # 괄호가 이미 확보되어 있으니 이분법으로 기계 정밀도까지 조인다.
            lo, hi = float(d_ok[i]), float(d_ok[i + 1])
            f_lo = float(c_ok[i]) - cd_target
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                c = cd_of(mid)
                if not np.isfinite(c):
                    break
                f_mid = c - cd_target
                if f_mid == 0.0:
                    return mid
                if (f_lo < 0.0) == (f_mid < 0.0):
                    lo, f_lo = mid, f_mid
                else:
                    hi = mid
                if hi - lo < 1e-14 * max(1.0, abs(hi)):
                    break
            d_ref = 0.5 * (lo + hi)
            return d_ref if np.isfinite(cd_of(d_ref)) else d_lin
    return float("nan")  # 목표 CD 가 구현 가능 범위 밖


def window_saturated(pw: np.ndarray) -> dict:
    """공정 윈도가 스윕 경계에 닿았는지 검사한다 (DOF/EL 과소·포화 판정)."""
    return {
        "focus_edge": bool(pw[:, 0].any() or pw[:, -1].any()),
        "dose_edge": bool(pw[0, :].any() or pw[-1, :].any()),
    }


def total_window(curve: list[dict]) -> float:
    """EL-DOF 곡선 아래 면적 [nm·%]. 공정 윈도 전체 크기 지표.

    공정 윈도는 단일 EL 에서의 DOF 또는 DOF-EL 곡선 아래 면적(total window)
    으로 표기하며, 후자는 EL 과 DOF 의 trade-off 를 모두 반영한다.
    """
    if len(curve) < 2:
        return 0.0
    el = np.array([r["el"] for r in curve])
    dof = np.array([r["dof"] for r in curve])
    return float(np.trapezoid(dof, el)) if hasattr(np, "trapezoid") else float(np.trapz(dof, el))


def exposure_latitude(
    grid,
    intensity: np.ndarray,
    cd_target: float,
    threshold: float,
    tol_frac: float = 0.10,
    tone: str = RS.TONE_BRIGHT,
    n_scan: int = 400,
    max_width: float | None = None,
) -> float:
    """CD 가 목표 +- tol_frac 을 유지하는 Dose 범위 [%].

    업계 표준은 CD 허용오차 +-10 % 기준이며, Mack 의 근사에 따르면
    EL [%] ~ 10 x NILS 의 선형 관계가 성립한다.
    """
    doses = np.linspace(0.4, 2.0, n_scan)
    cds = np.array([
        (lambda r: r["cd"] if r["formed"] and np.isfinite(r["cd"]) else np.nan)(
            RS.extract_cd(grid.x, intensity * d, threshold, tone, max_width=max_width)
        )
        for d in doses
    ])
    ok = np.isfinite(cds) & (np.abs(cds - cd_target) <= tol_frac * cd_target)
    if ok.sum() < 2:
        return 0.0
    lo, hi = doses[ok].min(), doses[ok].max()
    return float(100.0 * (hi - lo) / (0.5 * (lo + hi)))


def k2_from_dof(dof: float, wavelength: float, na: float, n_medium: float = 1.0) -> float:
    """측정 DOF 로부터 Rayleigh k2 를 역산한다.

    DOF = k2 * n * lambda_vac / NA^2  ->  k2 = DOF * NA^2 / (n * lambda_vac)
    """
    return dof * na**2 / (n_medium * wavelength)


def find_resolution_limit(
    wavelength: float,
    na: float,
    n_medium: float = 1.0,
    sigma_out: float = 0.7,
    sigma_in: float = 0.0,
    contrast_min: float = 0.10,
    pitch_lo: float = 40.0,
    pitch_hi: float = 600.0,
    n_iter: int = 20,
    n_periods: int = 6,
    dx_target: float = 0.5,
    n_source_grid: int = 31,
) -> dict:
    """해상 가능한 최소 pitch 를 이분법으로 탐색하고 k1 을 역산한다.

    해상 판정은 공간 광강도의 contrast >= contrast_min 으로 한다.
    """
    from . import grid as G

    def contrast_of(pitch: float) -> float:
        g = G.build_grid(pitch=pitch, n_periods=n_periods, dx_target=dx_target)
        m = MK.line_space(g, duty=0.5)
        i = IM.aerial_image(
            g, m, wavelength, na,
            n_medium=n_medium, sigma_out=sigma_out, sigma_in=sigma_in,
            n_source_grid=n_source_grid,
        )
        return IM.contrast(i, g.x)

    if contrast_of(pitch_hi) < contrast_min:
        return {"pitch": float("nan"), "half_pitch": float("nan"), "k1": float("nan")}

    lo, hi = pitch_lo, pitch_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if contrast_of(mid) >= contrast_min:
            hi = mid
        else:
            lo = mid
    half_pitch = hi / 2.0
    return {
        "pitch": hi,
        "half_pitch": half_pitch,
        "k1": half_pitch * na / wavelength,
        "contrast_min": contrast_min,
    }
