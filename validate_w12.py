"""
validate_w12.py -- 12주차 검증 (공정 조건 추천)

검증 전략 : 왜 학습된 모형으로 재지 않는가
------------------------------------------
추천기가 틀리는 길은 둘이다.

  (A) 배관이 틀렸다. 격자가 전치되었거나, 윈도 공차를 잘못 넣었거나,
      상자 제약이 걸리지 않았다.
  (B) 대리 모형이 부정확하다.

학습된 모형으로 재면 둘이 섞여 구분이 안 된다. 특히 (A) 는 숫자가 **그럴듯하게**
나오므로 눈으로는 못 잡는다. 격자를 전치해도 DOF 는 여전히 nm 단위의 적당한
값으로 나온다.

그래서 (A) 는 **해석적으로 답이 정해진 가짜 모형**으로 잰다. 예측값이 우리가
정한 함수이므로 윈도의 참값을 손으로 계산할 수 있고, 배관이 틀리면 반드시
어긋난다. (B) 는 11 주차에 이미 쟀고(상대 RMSE 3.47 %, 95 백분위 6.89 %),
그 크기가 공차를 거의 다 먹는다는 사실 자체가 Case 5 의 대상이다.

Case 4 만 시뮬레이터를 직접 돌린다. 목적함수를 공정 윈도로 바꾼 근거가
물리에 있는지 보는 것이므로, 어떤 모형도 끼우지 않아야 한다.

실행:  py validate_w12.py
"""

import io
import os
import sys

import numpy as np

import validate_common as VC

from litho import grid as G
from litho import mask as MK
from litho import imaging as IM
from litho import process as PR
from litho import dataset as DS

import recommend as RC

LAM, N_MED, TH = 193.0, 1.44, 0.30

_results = []
_CASES = []


def report(name, passed, lines):
    _results.append(passed)
    _CASES.append((name, bool(passed)))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    for ln in lines:
        print(f"        {ln}")
    print()


# ----------------------------------------------------------------------------
# 가짜 모형 : 예측값이 우리가 정한 함수다
# ----------------------------------------------------------------------------
class Planted:
    """지정한 함수를 그대로 돌려주는 모형. 배관 검사용."""

    def __init__(self, fn, feat):
        self.fn, self.feat = fn, feat

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = {f: X[:, j] for j, f in enumerate(self.feat)}
        return self.fn(cols)


FEAT = ["na", "sigma_out", "pitch", "duty", "defocus", "dose_rel",
        "flare", "msd_nm", "blur_nm", "z6", "z11"]


def make_domain(feat=FEAT, **override):
    """넉넉한 상자. override 로 축 하나를 좁힐 수 있다."""
    base = {"na": (0.90, 1.35), "sigma_out": (0.40, 0.90),
            "pitch": (100.0, 400.0), "duty": (0.30, 0.70),
            "defocus": (-200.0, 200.0), "dose_rel": (0.80, 1.20),
            "flare": (0.0, 0.05), "msd_nm": (0.0, 6.0),
            "blur_nm": (5.0, 15.0), "z6": (-0.05, 0.05),
            "z11": (-0.05, 0.05)}
    base.update(override)
    axes = {}
    for f in feat:
        lo, hi = base[f]
        axes[f] = {"min": lo, "max": hi,
                   "p01": lo + 0.01 * (hi - lo), "p99": hi - 0.01 * (hi - lo)}
    return {"n_train_rows": 1000, "axes": axes, "categorical": {},
            "held_constant": {"wavelength": LAM}}


def bundles_from(fns, feat=FEAT, domain=None):
    dom = domain or make_domain(feat)
    return {lab: {"model": Planted(fn, feat), "features": feat,
                  "categorical": [False] * len(feat), "label": lab,
                  "domain": dom, "pw_tol": 0.0707}
            for lab, fn in fns.items()}


# ----------------------------------------------------------------------------
# Case 1 : 격자 방향 정합
# ----------------------------------------------------------------------------
def case1_grid_orientation():
    """build_X 의 행 순서가 PR.cd_matrix 의 (dose, focus) 배치와 같은가.

    전치되어도 DOF 는 그럴듯한 nm 값으로 나오므로 눈으로는 못 잡는다. 그래서
    서로 구별되는 두 축 함수를 심고 재구성된 행렬을 손계산과 대조한다.
    """
    focus = np.array([-100.0, 0.0, 100.0])
    dose = np.array([0.9, 1.0, 1.1, 1.2])
    # 심는 함수 : 100*dose_rel + defocus. 두 축의 기여가 자릿수로 분리된다.
    b = bundles_from({
        "cd_error": lambda c: 100.0 * c["dose_rel"] + c["defocus"],
        "nils": lambda c: np.full(c["defocus"].shape, 3.0),
        "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0),
    })
    base = {"na": 1.2, "sigma_out": 0.7, "pitch": 200.0, "duty": 0.5,
            "flare": 0.0, "msd_nm": 0.0, "blur_nm": 10.0, "z6": 0.0, "z11": 0.0}
    m = RC.surrogate_cd_matrix(b, base, FEAT, 0.0, focus, dose)
    got = m["cd"]
    want = 100.0 * dose[:, None] + focus[None, :]
    err = float(np.max(np.abs(got - want)))

    # 음성 대조 : 전치하면 이 대조가 잡아야 한다
    caught = got.shape != want.T.shape or \
        float(np.max(np.abs(got.T[:min(got.shape), :min(got.shape)]
                            - want[:min(got.shape), :min(got.shape)]))) > 1e-9

    ok = err < 1e-9 and caught
    report("격자 방향 정합 (dose 바깥, focus 안쪽)", ok, [
        f"행렬 shape {got.shape} (dose {len(dose)} x focus {len(focus)})",
        f"심은 함수와의 최대 오차 {err:.3e}",
        f"전치 음성 대조 : {'잡음' if caught else '못 잡음'}",
    ])


# ----------------------------------------------------------------------------
# Case 2 : 윈도 지표 경로 일치
# ----------------------------------------------------------------------------
def case2_window_path():
    """window_metrics 가 PR 을 직접 부른 것과 같은가, 그리고 해석 윈도를 맞히는가.

    공차 안쪽이 초점 방향으로 정확히 지정한 구간인 CD 행렬을 만들고, DOF 가
    그 구간 폭과 맞는지 본다. 격자화 때문에 DOF 는 **항상 작게** 나오므로
    (process.el_dof_curve 의 설명) 참값이 [하한, 상한] 안에 있는지로
    판정한다. 보간으로 맞히는 척하지 않는다.

    [12 차 보완] 대조 대상을 `dof_at_el` 에서 `dof_bracket_at_el` 로 바꾸었다.
    window_metrics 가 그쪽을 쓰기 때문이다. 왜 바꾸었는지는 Case 8 에 있다.
    """
    target, tol = 100.0, 0.10
    focus = np.linspace(-200.0, 200.0, 41)      # 간격 10 nm
    dose = np.linspace(0.9, 1.1, 21)
    # 초점 |dz| <= 95 nm 구간만 규격 안. 참 DOF = 190 nm
    cd = np.where(np.abs(focus)[None, :] <= 95.0, target, target * 1.5)
    cd = np.repeat(cd, len(dose), axis=0)[:len(dose)]
    true_dof = 190.0

    got = RC.window_metrics(cd, target, tol, dose, focus, el_target=10.0)

    pw = PR.process_window(cd, target, tol_frac=tol)
    curve = PR.el_dof_curve(pw, dose, focus)
    direct_dof, direct_up = PR.dof_bracket_at_el(curve, 10.0)
    direct_total = PR.total_window(curve)

    same = (abs(got["dof"] - direct_dof) < 1e-9
            and abs(got["dof_upper"] - direct_up) < 1e-9
            and abs(got["total"] - direct_total) < 1e-9)
    bracket = got["dof"] <= true_dof <= got["dof_upper"] + 1e-9
    ok = same and bracket
    report("윈도 지표 경로 일치 및 해석 윈도 대조", ok, [
        f"참 DOF {true_dof:.1f} nm, 보고 {got['dof']:.1f} nm, "
        f"상한 {got['dof_upper']:.1f} nm",
        f"구간 포함 : {'예' if bracket else '아니오'} "
        f"(격자 간격 {focus[1] - focus[0]:.1f} nm)",
        f"PR 직접 호출과 일치 : {'예' if same else '아니오'}",
        f"초점 경계 포화 {got['focus_edge']}, 노광 경계 포화 {got['dose_edge']}",
    ])


# ----------------------------------------------------------------------------
# Case 3 : 상자 제약 거부
# ----------------------------------------------------------------------------
def case3_domain_refusal():
    """학습 범위를 벗어난 후보를 **거부**하는가, 그리고 안쪽은 통과시키는가.

    경고로 두면 안 되는 이유는 11 주차 모형으로 실측했다. 상자 경계의 11 배
    밖에서 cd_error 예측이 0.085 nm(사실상 완벽)로 나왔다. 오차를 최소화하는
    탐색기는 그 지점을 답으로 고른다.
    """
    dom = make_domain(defocus=(-50.0, 50.0))    # 초점 상자를 좁게
    b = bundles_from({
        "cd_error": lambda c: np.zeros(c["defocus"].shape),
        "nils": lambda c: np.full(c["defocus"].shape, 3.0),
        "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0),
    }, domain=dom)
    base = {"na": 1.2, "sigma_out": 0.7, "pitch": 200.0, "duty": 0.5,
            "flare": 0.0, "msd_nm": 0.0, "blur_nm": 10.0, "z6": 0.0, "z11": 0.0}
    spec = DS.SampleSpec(pitch=200.0, duty=0.5)
    dose = np.linspace(0.9, 1.1, 11)

    out_grid = np.linspace(-200.0, 200.0, 21)   # 상자 밖
    r_out = RC.evaluate(b, base, FEAT, spec, out_grid, dose, 0.0)
    in_grid = np.linspace(-40.0, 40.0, 21)      # 상자 안
    r_in = RC.evaluate(b, base, FEAT, spec, in_grid, dose, 0.0)

    refused = (not r_out["ok"]) and r_out["reason"] == "상자 밖"
    axes = [v["axis"] for v in r_out.get("violations", [])]
    accepted = r_in["ok"]

    # 보수 상자(1~99 백분위)가 min/max 상자보다 엄격한가
    edge = np.linspace(-50.0, 50.0, 11)         # min/max 경계에 딱 닿음
    v_loose = RC.domain_violations(
        RC.build_X(base, FEAT, edge, dose), FEAT, dom, conservative=False)
    v_tight = RC.domain_violations(
        RC.build_X(base, FEAT, edge, dose), FEAT, dom, conservative=True)
    stricter = (len(v_loose) == 0) and (len(v_tight) > 0)

    ok = refused and accepted and stricter and ("defocus" in axes)
    report("상자 제약 거부 및 보수 상자", ok, [
        f"상자 밖 후보 : {'거부' if refused else '통과시킴'} "
        f"(위반 축 {axes})",
        f"상자 안 후보 : {'평가' if accepted else '거부됨'}",
        f"보수 상자가 더 엄격 : {'예' if stricter else '아니오'} "
        f"(min/max 위반 {len(v_loose)} 건, 백분위 위반 {len(v_tight)} 건)",
    ])


# ----------------------------------------------------------------------------
# Case 4 : 공칭 목적함수의 공허성 (시뮬레이터 직접)
# ----------------------------------------------------------------------------
def case4_nominal_objective_is_flat():
    """공칭점 CD 오차가 조건을 가르지 못하고 공정 윈도는 가르는가.

    목적함수를 '|cd_error| 최소화' 에서 '공정 윈도 최대화' 로 바꾼 근거다.
    `dose_to_size` 가 조건마다 따로 CD 를 목표에 붙이므로, 공칭점 오차는
    조건 간 차이를 거의 담지 않는다. 어떤 모형도 끼우지 않고 시뮬레이터로
    직접 잰다.
    """
    pitch, duty = 200.0, 0.5
    target = duty * pitch
    focus = np.linspace(-150.0, 150.0, 13)
    dose = np.linspace(0.90, 1.10, 13)
    cands = [(0.95, 0.5), (1.15, 0.7), (1.35, 0.85)]

    rows = []
    for na, sig in cands:
        g = G.build_grid(pitch=pitch, n_periods=6,
                         dx_target=G.nyquist_dx(LAM, na, sig))
        m = MK.line_space(g, cd=target)
        opt = dict(n_medium=N_MED, sigma_out=sig, sigma_in=0.0,
                   polarization="TE", n_source_grid=21)
        i0 = IM.aerial_image(g, m, LAM, na, defocus=0.0, **opt)
        e_op = PR.dose_to_size(g, i0, target, TH, max_width=pitch)
        if not np.isfinite(e_op):
            continue
        ints = PR.focus_sweep(g, m, LAM, na, defocuses=focus, **opt)
        cd_mat = PR.cd_matrix(g, ints * e_op, dose, TH, max_width=pitch)
        # 공칭점 : 초점 0, 노광 1 배
        jf = int(np.argmin(np.abs(focus)))
        jd = int(np.argmin(np.abs(dose - 1.0)))
        nom_err = abs(float(cd_mat[jd, jf]) - target)
        pw = PR.process_window(cd_mat, target,
                               tol_frac=DS.SampleSpec(pitch=pitch,
                                                      duty=duty).pw_tol)
        # 추천기와 **같은 방식**으로 읽는다. 여기서만 dof_at_el 을 쓰면 같은
        # 양을 두 방식으로 보고하는 셈이 되고, 그 숫자가 README 표에 들어간다.
        dof, dof_up = PR.dof_bracket_at_el(
            PR.el_dof_curve(pw, dose, focus), 10.0)
        rows.append((na, sig, nom_err, dof, dof_up))

    errs = np.array([r[2] for r in rows])
    dofs = np.array([r[3] for r in rows])
    # 상대 변동폭으로 비교한다. 임계값이 아니라 두 변동의 대소를 본다.
    sp_err = (errs.max() - errs.min()) / max(np.mean(errs), 1e-9)
    sp_dof = (dofs.max() - dofs.min()) / max(np.mean(dofs), 1e-9)
    ok = len(rows) >= 2 and sp_dof > sp_err

    lines = [f"후보 {len(rows)} 개, 목표 CD {target:.1f} nm"]
    lines.append(f"{'NA':>6}{'sigma':>8}{'공칭 |오차| [nm]':>18}"
                 f"{'DOF 하한':>11}{'DOF 상한':>10}")
    for na, sig, e, dof, dof_up in rows:
        lines.append(f"{na:>6.2f}{sig:>8.2f}{e:>18.3f}{dof:>11.1f}"
                     f"{dof_up:>10.1f}")
    lines.append(f"상대 변동폭 : 공칭 오차 {sp_err:.3f} 대 DOF {sp_dof:.3f}")
    lines.append("")
    lines.append("공칭 오차가 0 인 것은 우연이 아니라 dose_to_size 의 정의다.")
    lines.append("조건마다 최적 초점에서 목표 CD 가 나오도록 노광량을 역산하므로")
    lines.append("결정론 CD 의 공칭 오차는 **구조적으로 0** 이다(10주차 보정 후")
    lines.append("중앙값 1.2e-13 nm). 즉 그 목적함수는 '거의 평평' 이 아니라")
    lines.append("상수이며, 탐색이 어느 후보도 가릴 수 없다.")
    lines.append("데이터셋의 cd_mean 은 확률 평균이라 공칭점에도 잔차가 남지만")
    lines.append("(시험 데이터 중앙값 2.35 nm), 그것은 몬테카를로 잡음이므로")
    lines.append("최소화하면 잡음을 최소화하는 셈이 된다. 양쪽 다 목적함수가")
    lines.append("될 수 없다.")
    report("공칭 목적함수의 공허성 [시뮬레이터 직접]", ok, lines)


# ----------------------------------------------------------------------------
# Case 5 : 보수 윈도와 인증 가능성
# ----------------------------------------------------------------------------
def case5_guarded_window():
    """모형 오차가 공차를 다 먹으면 '인증 불가' 로 떨어지는가.

    11 주차 실측 상대 95 백분위는 6.89 %, pw_tol 은 7.07 % 다. 즉 보수 공차가
    0.181 % 밖에 남지 않는다. 이 사실이 숫자 하나로 덮이면 "DOF 180 nm 확보"
    로 읽히므로, 코드가 인증 가능성을 따로 판정해야 한다.
    """
    b = bundles_from({
        "cd_error": lambda c: 0.0012 * c["defocus"] ** 2 / 100.0,
        "nils": lambda c: np.full(c["defocus"].shape, 3.0),
        "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0),
    })
    base = {"na": 1.2, "sigma_out": 0.7, "pitch": 200.0, "duty": 0.5,
            "flare": 0.0, "msd_nm": 0.0, "blur_nm": 10.0, "z6": 0.0, "z11": 0.0}
    spec = DS.SampleSpec(pitch=200.0, duty=0.5)
    focus = np.linspace(-200.0, 200.0, 33)
    dose = np.linspace(0.9, 1.1, 21)

    r_free = RC.evaluate(b, base, FEAT, spec, focus, dose, rel_p95=0.0)
    r_real = RC.evaluate(b, base, FEAT, spec, focus, dose, rel_p95=0.0689)
    r_over = RC.evaluate(b, base, FEAT, spec, focus, dose, rel_p95=0.20)

    shrinks = (r_free["guarded"]["dof"] >= r_real["guarded"]["dof"]
               >= r_over["guarded"]["dof"])
    nominal_same = abs(r_free["nominal"]["dof"]
                       - r_over["nominal"]["dof"]) < 1e-9
    killed = (not r_over["certifiable"]) and r_over["guarded"]["dof"] == 0.0
    certif = r_free["certifiable"] and r_real["certifiable"]
    ok = shrinks and nominal_same and killed and certif

    report("보수 윈도와 인증 가능성 판정", ok, [
        f"{'모형 95% 오차':>16}{'보수 공차 [%]':>14}{'보수 DOF [nm]':>14}"
        f"{'인증 가능':>10}",
        f"{'0.00 %':>16}{100 * r_free['guarded']['tol']:>14.2f}"
        f"{r_free['guarded']['dof']:>14.1f}{str(r_free['certifiable']):>10}",
        f"{'6.89 % (실측)':>16}{100 * r_real['guarded']['tol']:>14.2f}"
        f"{r_real['guarded']['dof']:>14.1f}{str(r_real['certifiable']):>10}",
        f"{'20.00 %':>16}{100 * r_over['guarded']['tol']:>14.2f}"
        f"{r_over['guarded']['dof']:>14.1f}{str(r_over['certifiable']):>10}",
        f"낙관 DOF 는 모형 오차에 무관 : {'예' if nominal_same else '아니오'}",
        f"보수 DOF 단조 감소 : {'예' if shrinks else '아니오'}",
        "",
        "[이 표의 보수 DOF 를 일반화하지 말 것] 심은 함수를 의도적으로 매우",
        "평평하게(초점 100 nm 에서 CD 0.12 %) 두었다. 논리만 분리해 보려는",
        "것이므로, 공차 0.181 % 에서 DOF 225 nm 가 나온 것은 심은 함수의 성질",
        "이고 실제 공정의 값이 아니다. 실제 값은 13 주차에 시뮬레이터로 잰다.",
    ])


# ----------------------------------------------------------------------------
# Case 6 : 탐색 축 분리
# ----------------------------------------------------------------------------
def case6_axis_separation():
    """추천기가 손잡이만 바꾸고 공구 상태와 설계 요구는 건드리지 않는가.

    11 열을 한꺼번에 탐색하면 "플레어를 0 으로 두시오" 같은 답이 나온다.
    플레어는 측정하는 값이고 고르는 값이 아니다. 구조적으로 막혔는지 본다.
    """
    b = bundles_from({
        "cd_error": lambda c: 50.0 * c["flare"] + 0.001 * c["defocus"] ** 2
        - 2.0 * c["na"],
        "nils": lambda c: 2.0 + 2.0 * c["na"],
        "cd_3sigma": lambda c: np.full(c["na"].shape, 1.0),
    })
    # 수차는 같은 부호로 둔다. 섞인 부호는 Case 7 에서 따로 다룬다
    # (현 데이터로 닿을 수 없는 등가류이므로 rank 가 거부한다).
    tool = {"flare": 0.04, "msd_nm": 3.0, "blur_nm": 12.0,
            "z6": 0.02, "z11": 0.01}
    res = RC.rank(b, FEAT, tool, pitch=200.0, duty=0.5, rel_p95=0.0,
                  n_na=4, n_sig=3, focus_span=150.0, n_focus=15,
                  dose_span=0.10, n_dose=15, el_target=10.0,
                  conservative=False)
    rows = res["rows"]

    # 평가된 모든 후보의 입력을 다시 만들어 공구 상태가 그대로인지 본다
    held_ok, varied = True, set()
    for r in rows:
        base = dict(tool)
        base.update({"na": r["na"], "sigma_out": r["sigma_out"],
                     "pitch": 200.0, "duty": 0.5})
        X = RC.build_X(base, FEAT, res["focus"], res["dose_rel"])
        for j, f in enumerate(FEAT):
            col = X[:, j]
            if f in tool and not np.allclose(col, tool[f]):
                held_ok = False
            if f in ("pitch", "duty") and np.ptp(col) > 0:
                held_ok = False
        varied.add((round(r["na"], 6), round(r["sigma_out"], 6)))

    n_expect = 4 * 3
    nas = {k[0] for k in varied}
    sigs = {k[1] for k in varied}
    moved = len(nas) > 1 and len(sigs) > 1
    ok = held_ok and moved and len(rows) == n_expect

    report("탐색 축 분리 (손잡이만 움직임)", ok, [
        f"평가된 후보 {len(rows)} / {n_expect} 개",
        f"움직인 축 : na {len(nas)} 값, sigma_out {len(sigs)} 값",
        f"공구 상태와 설계 요구 고정 : {'예' if held_ok else '아니오'} "
        f"(flare {tool['flare']}, MSD {tool['msd_nm']}, 피치 200, duty 0.5)",
        f"최상위 : NA {rows[0]['na']:.3f}, sigma {rows[0]['sigma_out']:.3f}, "
        f"DOF {rows[0]['nominal']['dof']:.1f} nm" if rows else "후보 없음",
    ])


# ----------------------------------------------------------------------------
# Case 7 : 수차 부호 거울 [시뮬레이터 직접]
# ----------------------------------------------------------------------------
def case7_sign_mirror():
    """부호 거울이 **시뮬레이터에서 정확히** 성립하는가, 그리고 닿지 않는
    사분면을 거부하는가.

    이 변환은 추천기가 음수 수차 공구를 학습 범위 없이 평가하는 근거다.
    근거가 측정이 아니라 추론이면 안 되므로, 어떤 모형도 끼우지 않고
    시뮬레이터로 직접 대조한다. 자세한 설명은 RC.SIGN_MIRROR_NOTE 에 있다.
    """
    pitch, duty, a = 200.0, 0.5, 0.04
    target = duty * pitch
    na, sig = 1.35, 0.7
    g = G.build_grid(pitch=pitch, n_periods=6,
                     dx_target=G.nyquist_dx(LAM, na, sig))
    msk = MK.line_space(g, cd=target)
    opt = dict(n_medium=N_MED, sigma_out=sig, sigma_in=0.0,
               polarization="TE", n_source_grid=21)

    def cd(dz, z6, z11):
        ab = {k: v for k, v in ((6, z6), (11, z11)) if v}
        i0 = IM.aerial_image(g, msk, LAM, na, defocus=0.0,
                             aberrations=ab, **opt)
        e = PR.dose_to_size(g, i0, target, TH, max_width=pitch)
        i = IM.aerial_image(g, msk, LAM, na, defocus=float(dz),
                            aberrations=ab, **opt)
        from litho import resist as RS
        r = RS.extract_cd(g.x, i * e, TH, max_width=pitch)
        return float(r["cd"]) if r["formed"] else float("nan")

    # 1) 전체 뒤집기는 정확히 같아야 한다 (같은 부호, 섞인 부호 둘 다)
    same = []
    for z6, z11 in ((a, a), (a, -a)):
        same.append(abs(cd(80.0, z6, z11) - cd(-80.0, -z6, -z11)))
    exact = max(same) < 1e-9

    # 2) 두 등가류가 실제로 다른가. 축퇴가 '부호가 무의미하다' 가 아님을
    #    보인다.
    #
    #    [대조 대상을 정확히 쓸 것] A 류 곡선 CD(dz; +a,+a) 과 B 류 곡선을
    #    비교한다. B 류의 대표는 (+a,-a) 와 (-a,+a) 두 개이고, 한 초점에서
    #    둘의 값은 서로 다르다(거울 관계이므로 dz 를 뒤집어야 같아진다).
    #    그래서 한 점에서 한 대표만 보면 차이를 과소·과대 평가한다.
    #    초점 구간 전체에서 두 대표 모두와 비교해 **최댓값**을 쓴다.
    fgrid = np.linspace(-150.0, 150.0, 11)
    gap = 0.0
    for dz in fgrid:
        ref = cd(dz, a, a)
        for z6, z11 in ((a, -a), (-a, a)):
            v = cd(dz, z6, z11)
            if np.isfinite(ref) and np.isfinite(v):
                gap = max(gap, abs(ref - v))
    spec = DS.SampleSpec(pitch=pitch, duty=duty)
    tol_nm = spec.pw_tol * target
    distinct = gap > tol_nm

    # 3) 거울 적용 : 둘 다 음수면 양수로 바뀌는가
    dom = make_domain(z6=(0.0, 0.05), z11=(0.0, 0.05))
    base_neg = {"z6": -0.02, "z11": -0.01, "flare": 0.01,
                "msd_nm": 1.0, "blur_nm": 10.0}
    got, did, why = RC.mirror_request(base_neg, dom)
    flipped = did and not why and got["z6"] > 0 and got["z11"] > 0 \
        and got["flare"] == base_neg["flare"]

    # 4) 섞인 부호는 사유와 함께 거부되는가
    _, did_mix, why_mix = RC.mirror_request(
        {"z6": 0.02, "z11": -0.01}, dom)
    refused = (not did_mix) and bool(why_mix)

    # 5) 양수는 건드리지 않는가 (음성 대조)
    pos = {"z6": 0.02, "z11": 0.01}
    got_p, did_p, _ = RC.mirror_request(pos, dom)
    untouched = (not did_p) and got_p == pos

    ok = exact and distinct and flipped and refused and untouched
    report("수차 부호 거울 [시뮬레이터 직접]", ok, [
        f"전체 뒤집기 최대 차 {max(same):.3e} nm "
        f"(같은 부호와 섞인 부호 각각)",
        f"등가류 간 최대 차 {gap:.3f} nm (초점 {fgrid[0]:.0f}~{fgrid[-1]:.0f} nm, "
        f"B 류 대표 둘 모두와 대조) 대 공차 {tol_nm:.3f} nm "
        f"-> 두 류는 {'구별됨' if distinct else '구별 안 됨'}",
        f"둘 다 음수 -> 양수로 거울 : {'예' if flipped else '아니오'} "
        f"(수차 외 축은 그대로)",
        f"섞인 부호 -> 거부 : {'예' if refused else '아니오'}",
        f"양수는 그대로 (음성 대조) : {'예' if untouched else '아니오'}",
        "",
        "즉 부호 사분면 넷이 두 등가류로 묶이고 데이터는 한 류만 덮는다.",
        "다음 생성에서 두 축 중 하나만 음수로 열면 표본 수를 늘리지 않고",
        "두 류를 다 덮는다. RC.SIGN_MIRROR_NOTE 참조.",
    ])


# ----------------------------------------------------------------------------
# Case 8 : DOF 구간이 참값을 담는가 [해석]
# ----------------------------------------------------------------------------
def case8_dof_bracket():
    """해석해를 참값으로 두고 DOF 구간을 검증한다. 옛 방식은 왜 틀렸는가.

    CD/T = 1 + s(d-1) - K dz^2 (선형 dose + Bossung) 이면 EL 을 고정한 최대
    DOF 를 손으로 풀 수 있다. 축정렬 직사각형 d in [d_lo, d_hi],
    dz in [-D/2, D/2] 에 대해

        최대 CD/T - 1 = s(d_hi - 1)                  (dz = 0)
        최소 CD/T - 1 = s(d_lo - 1) - K (D/2)^2      (dz = +-D/2)

    이므로 규격 |CD/T - 1| <= tol 은 s*u <= tol 과 s*l - K(D/2)^2 >= -tol,
    u = d_hi - 1, l = d_lo - 1 이다. D 를 키우려면 l 을 키워야 하고, EL 고정
    아래에서 l 은 u 에 대해 증가하므로 u = tol/s 가 최적이다. 그러면

        l = (u(1 - e/2) - e) / (1 + e/2),  e = EL/100
        D = 2 sqrt((tol + s*l) / K)        (tol + s*l <= 0 이면 D = 0)

    이 참값이다. 격자 없이 정확하므로 수렴 격자를 돌릴 필요가 없다.

    [왜 이 케이스가 생겼는가]
    추천기는 하한을 `dof_at_el` 로, 상한을 각 항목 `dof_upper` 의 EL 축
    보간으로 따로 만들었다. 그러면 **노광 격자의 EL 편향이 구간에서 빠진다.**
    EL-DOF 무릎에서는 dDOF/dEL 이 커서 EL 1.9 % 가 DOF 100 nm 넘게 번지는데,
    담겨 있던 초점항은 25 nm 였다. 즉 두 오차원 가운데 큰 쪽을 누락했다.
    그 결과 33x33 격자에서 구간이 [0.0, 0.0] 으로 나와 "참값은 구간 안에
    있다" 는 문서의 주장이 거짓이었고, 하한 0 은 정렬 키이므로 참 DOF 112 nm
    인 후보가 윈도 없는 후보와 구별되지 않고 탈락했다.

    판정은 세 가지다.
      - 새 구간이 세 영역 x 다섯 격자 전부에서 참값을 담는가
      - 옛 방식이 적어도 하나에서 놓치는가 (**음성 대조**. 놓치지 않으면
        이 케이스는 아무것도 지키지 않는 것이므로 실패로 본다)
      - 하한 <= 상한 이 항상 성립하는가
    """
    TGT, TOL, K = 40.0, 0.07, 1.2e-6
    EL = 10.0

    def cd(d, z, s):
        return TGT * (1.0 + s * (d - 1.0) - K * z ** 2)

    def exact(s):
        e = EL / 100.0
        u = TOL / s
        l = (u * (1.0 - e / 2.0) - e) / (1.0 + e / 2.0)
        return 0.0 if TOL + s * l <= 0 else 2.0 * np.sqrt((TOL + s * l) / K)

    def measure(s, n_dose, n_focus):
        dr = np.linspace(0.85, 1.15, n_dose)
        dz = np.linspace(-250.0, 250.0, n_focus)
        pw = PR.process_window(cd(dr[:, None], dz[None, :], s), TGT,
                               tol_frac=TOL)
        c = PR.el_dof_curve(pw, dr, dz)
        lo, hi = PR.dof_bracket_at_el(c, EL)
        # 옛 방식을 그대로 재현한다. 고친 코드를 부르면 대조가 되지 않는다.
        old_lo = PR.dof_at_el(c, el_target=EL)
        els = np.array([r["el"] for r in c]) if c else np.array([0.0])
        ups = np.array([r["dof_upper"] for r in c]) if c else np.array([0.0])
        if c and EL <= els[0]:
            old_hi = float(ups[0])
        elif c and EL < els[-1]:
            old_hi = float(np.interp(EL, els, ups))
        else:
            old_hi = 0.0
        return lo, hi, old_lo, old_hi

    REGIMES = ((1.36, "무릎"), (0.90, "여유"), (1.55, "미달"))
    GRIDS = ((33, 33), (65, 65), (129, 129), (257, 257), (513, 257))

    lines, new_miss, old_miss, order_ok, n = [], 0, 0, True, 0
    for s, name in REGIMES:
        t = exact(s)
        lines.append(f"[{name}] s={s}  해석 참 DOF = {t:.2f} nm")
        lines.append(f"  {'격자':>10}{'새 하한':>9}{'새 상한':>9}{'담음':>6}"
                     f"{'옛 하한':>9}{'옛 상한':>9}{'담음':>6}")
        for nd, nf in GRIDS:
            lo, hi, olo, ohi = measure(s, nd, nf)
            n += 1
            a = lo - 1e-9 <= t <= hi + 1e-9
            b = olo - 1e-9 <= t <= ohi + 1e-9
            new_miss += (not a)
            old_miss += (not b)
            order_ok = order_ok and (lo <= hi + 1e-12)
            lines.append(f"  {f'{nd}x{nf}':>10}{lo:>9.1f}{hi:>9.1f}"
                         f"{('예' if a else '아니오'):>6}"
                         f"{olo:>9.1f}{ohi:>9.1f}"
                         f"{('예' if b else '아니오'):>6}")
        lines.append("")

    # 빈 곡선도 터지지 않는가
    empty_lo, empty_hi = PR.dof_bracket_at_el([], EL)
    empty_ok = (empty_lo, empty_hi) == (0.0, 0.0)

    ok = (new_miss == 0) and (old_miss > 0) and order_ok and empty_ok
    report("DOF 구간이 참값을 담음 [해석]", ok, lines + [
        f"조합 {n} 개 : 새 구간 놓침 {new_miss} 개, 옛 방식 놓침 {old_miss} 개",
        f"하한 <= 상한 항상 성립 : {'예' if order_ok else '아니오'}",
        f"빈 곡선 -> (0, 0) : {'예' if empty_ok else '아니오'}",
        "",
        "옛 방식이 하나도 놓치지 않으면 이 케이스는 아무것도 지키지 못하므로",
        "실패로 본다(음성 대조). 옛 방식은 무릎뿐 아니라 여유 영역 65x65 에서도",
        "깨진다. 내부 보간이 안전하다고 본 것이 잘못이었다.",
    ])


# ----------------------------------------------------------------------------
# Case 9 : 거울을 쓰면 초점 격자가 대칭이 되는가 [불변량]
# ----------------------------------------------------------------------------
def case9_mirror_needs_symmetric_focus():
    """거울 논증의 암묵 가정을 검사한다.

    `CD(+dz, +z6, +z11) = CD(-dz, -z6, -z11)` 로 음수 수차 장비를 양수로 바꿔
    평가한다. 그 결과를 원래 장비의 결과로 읽으려면 **초점 축을 뒤집어야**
    한다. 초점 격자가 비대칭이면 뒤집은 격자가 원래와 다른 구간을 덮으므로
    DOF 를 그대로 가져올 수 없다.

    학습 범위의 defocus min/max 는 표본에서 오므로 정확히 대칭일 이유가 없다.
    그래서 거울을 적용할 때 격자를 대칭으로 줄인다. 그 동작을 여기서 고정한다.

    기준값이 필요 없다. 대칭성과 '거울을 안 쓰면 줄이지 않는다' 만 본다.
    """
    fns = {"cd_error": lambda c: 0.3 * c["defocus"] ** 2 / 1000.0,
           "nils": lambda c: np.full(c["defocus"].shape, 3.0),
           "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0)}
    # 비대칭 학습 범위. 실제 데이터에서 그대로 나올 수 있는 모양이다.
    dom = make_domain(defocus=(-183.0, 200.0), z6=(0.0, 0.05),
                      z11=(0.0, 0.05))
    b = bundles_from(fns, domain=dom)

    common = dict(pitch=200.0, duty=0.5, rel_p95=0.0, n_na=2, n_sig=2,
                  focus_span=300.0, n_focus=25, dose_span=0.15, n_dose=25,
                  el_target=10.0, conservative=False)
    tool_neg = {"flare": 0.02, "msd_nm": 2.0, "blur_nm": 10.0,
                "z6": -0.02, "z11": -0.01}          # 둘 다 음수 -> 거울
    tool_pos = dict(tool_neg, z6=0.02, z11=0.01)    # 양수 -> 거울 안 씀

    r_neg = RC.rank(b, FEAT, dict(tool_neg), **common)
    r_pos = RC.rank(b, FEAT, dict(tool_pos), **common)

    fn, fp = r_neg["focus"], r_pos["focus"]
    neg_sym = bool(abs(abs(fn[0]) - abs(fn[-1])) < 1e-9)
    # 거울을 안 쓰면 학습 범위 그대로(비대칭)여야 한다. 그렇지 않으면 이
    # 케이스는 '항상 대칭' 을 보는 셈이어서 아무것도 지키지 못한다.
    pos_asym = bool(abs(abs(fp[0]) - abs(fp[-1])) > 1e-9)
    noted = any("대칭" in t for t in r_neg["notes"])
    # 대칭화해도 학습 범위를 벗어나지 않아야 한다.
    inside = bool(fn[0] >= dom["axes"]["defocus"]["min"] - 1e-9
                  and fn[-1] <= dom["axes"]["defocus"]["max"] + 1e-9)
    ran = bool(r_neg["rows"] and r_pos["rows"])

    ok = neg_sym and pos_asym and noted and inside and ran
    report("거울 적용 시 초점 격자 대칭", ok, [
        f"학습 범위 defocus [{dom['axes']['defocus']['min']:.1f}, "
        # 비대칭 폭을 상수로 적으면 범위를 바꿀 때 문구만 낡는다. 계산한다.
        f"{dom['axes']['defocus']['max']:.1f}] nm (비대칭 "
        f"{abs(abs(dom['axes']['defocus']['min']) - abs(dom['axes']['defocus']['max'])):.1f} nm)",
        f"거울 적용(수차 둘 다 음수) : 격자 [{fn[0]:.1f}, {fn[-1]:.1f}] nm "
        f"-> 대칭 {'예' if neg_sym else '아니오'}",
        f"거울 미적용(수차 양수)     : 격자 [{fp[0]:.1f}, {fp[-1]:.1f}] nm "
        f"-> 비대칭 유지 {'예' if pos_asym else '아니오'} (음성 대조)",
        f"대칭화를 주의로 보고 : {'예' if noted else '아니오'}",
        f"대칭화 후에도 학습 범위 안 : {'예' if inside else '아니오'}",
        f"두 경우 모두 후보를 평가함 : {'예' if ran else '아니오'}",
        "",
        "비대칭 격자에서 거울을 쓰면 뒤집은 격자가 원래와 다른 구간을 덮어",
        "DOF 비교가 성립하지 않는다. 잃는 것은 긴 쪽 꼬리뿐이다.",
    ])


# ----------------------------------------------------------------------------
# Case 10 : 목표 EL 을 정확히 주는 격자에서 하한이 살아남는가 [불변량]
# ----------------------------------------------------------------------------
def case10_exact_el_tolerance():
    """부동소수 동등 비교로 유효한 행이 탈락하지 않는가.

    밴드 높이가 목표 EL 을 **정확히** 주는 격자가 있다. 그때 계산된 EL 이
    9.999999999999988 (목표 - 1.24e-14) 로 나오므로 허용오차 없는 `>=` 는
    거짓이 된다. 그러면 그 행의 DOF 가 버려지고 하한이 0 으로 떨어진다.
    65 번이 막으려는 결함(유효한 후보가 조용히 탈락)을 비교 연산으로
    되살리는 셈이다.

    실제로 그렇게 만들었다가 잡았다. 노광 점 25·49·73·97·121·145·169 개
    격자 전부에서 하한이 62.5 nm 에서 0.0 으로 떨어져 있었다.

    판정은 세 가지다.
      - 그런 격자가 실제로 존재하는가 (없으면 이 케이스는 아무것도 지키지
        못한다. **음성 대조**)
      - 그 격자에서 하한이 0 보다 큰가
      - 허용오차를 없앤 비교로는 0 이 되는가 (즉 허용오차가 일하고 있는가)
    """
    TGT, TOL, K, S, EL = 40.0, 0.07, 1.2e-6, 1.36, 10.0

    def curve_at(n_dose):
        dr = np.linspace(0.85, 1.15, n_dose)
        dz = np.linspace(-250.0, 250.0, 33)
        cd = TGT * (1.0 + S * (dr[:, None] - 1.0) - K * dz[None, :] ** 2)
        return PR.el_dof_curve(PR.process_window(cd, TGT, tol_frac=TOL), dr, dz)

    GRIDS = (25, 49, 73, 97, 121, 145, 169)
    lines, n_exact, n_rescued, n_naive_zero = [], 0, 0, 0
    lines.append(f"{'n_dose':>7}{'정확히 EL 10 인 행':>20}{'el - 10':>13}"
                 f"{'하한':>8}{'허용오차 없이':>14}")
    for nd in GRIDS:
        c = curve_at(nd)
        hit = [r for r in c if abs(r["el"] - EL) < 1e-6 and r["el"] < EL]
        if not hit:
            lines.append(f"{nd:>7}{'없음':>20}")
            continue
        n_exact += 1
        r0 = max(hit, key=lambda r: r["dof"])
        lo, _hi = PR.dof_bracket_at_el(c, EL)
        # 허용오차를 없앤 비교를 손으로 재현한다. 고친 코드를 부르면 대조가
        # 되지 않는다.
        naive = max([float(r["dof"]) for r in c if r["el"] >= EL], default=0.0)
        n_rescued += (lo > 0.0)
        n_naive_zero += (naive <= 0.0)
        lines.append(f"{nd:>7}{'있음':>20}{r0['el'] - EL:>+13.2e}"
                     f"{lo:>8.1f}{naive:>14.1f}")

    ok = (n_exact == len(GRIDS) and n_rescued == len(GRIDS)
          and n_naive_zero == len(GRIDS))
    report("정확한 목표 EL 에서 하한 보존", ok, lines + [
        "",
        f"목표 EL 을 정확히 주는 격자 {n_exact} / {len(GRIDS)} 개",
        f"그 격자에서 하한 > 0 : {n_rescued} / {len(GRIDS)} 개",
        f"허용오차를 없애면 0 이 되는 격자 : {n_naive_zero} / {len(GRIDS)} 개 "
        f"(음성 대조. 0 이면 허용오차가 아무 일도 안 하는 것이다)",
        "",
        "하한을 0 으로 적으면 '윈도 없음' 과 구별되지 않고 순위에서 탈락한다.",
        "62.5 nm 를 1.2e-14 때문에 버리는 셈이다.",
    ])


# ----------------------------------------------------------------------------
# Case 11 : 초점 격자 홀짝이 DOF 의 방향을 바꾼다 [해석]
# ----------------------------------------------------------------------------
def case11_focus_grid_parity():
    """'DOF 는 항상 작게 나온다' 가 성립하는 조건을 고정한다.

    그 주장은 **격자점에서 통과하면 사이에서도 통과한다**는 가정에 의존한다.
    통과 영역이 초점 방향으로 하나의 구간일 때만 참이다.

    노광량이 높아 최적 초점의 CD 가 규격 **상한**을 넘는 dose 행에서는 통과
    영역이 **고리 모양**이 된다. Bossung 이 dz=0 에서 극값이므로 중심이 규격
    초과이고 그 바깥 띠가 통과한다. 중심 금지대가 격자 간격보다 얇으면 짝수
    격자는 그것을 건너뛰고 양쪽을 하나로 읽어 DOF 를 **과대**평가한다.
    방향이 뒤집힌다.

    홀수 격자는 dz=0 을 반드시 표본에 넣으므로 금지대를 항상 밟는다.

    판정은 셋이다.
      - 짝수 격자가 실제로 규격 밖 점을 품은 윈도를 보고하는가 (**음성 대조**.
        아니면 이 케이스가 아무것도 지키지 못한다)
      - 홀수 격자는 그러지 않는가
      - `recommend.rank` 가 짝수를 받으면 홀수로 올리고 보고하는가
    """
    TGT, TOL, K, S = 40.0, 0.07, 1.2e-6, 1.36

    def cd(d, z):
        return TGT * (1.0 + S * (d - 1.0) - K * z ** 2)

    def probe(nf):
        """격자가 금지대를 건너뛰도록 dose 를 맞춘 뒤 윈도 안을 촘촘히 검사."""
        dz = np.linspace(-250.0, 250.0, nf)
        near = float(np.min(np.abs(dz)))
        # 0 에 가장 가까운 격자점보다 얇은 금지대를 겨냥한다. 홀수면 near=0
        # 이므로 겨냥할 수 없고(항상 밟는다) 비교용 기본값을 쓴다.
        half = 0.7 * near if near > 1e-12 else 10.0
        d_star = 1.0 + (TOL + K * half ** 2) / S
        dr = np.array([1.0, d_star])
        curve = PR.el_dof_curve(
            PR.process_window(cd(dr[:, None], dz[None, :]), TGT, tol_frac=TOL),
            dr, dz)
        bad, dof_r = 0, 0.0
        for r in curve:
            f_lo, f_hi = r["focus_range"]
            d_lo, d_hi = r["dose_range"]
            if f_hi <= f_lo:
                continue
            zz = np.linspace(f_lo, f_hi, 20001)
            for d in (d_lo, d_hi):
                v = cd(d, zz)
                b = np.abs(v - TGT) > TOL * TGT + 1e-12
                if b.any():
                    bad += int(b.sum())
                    dof_r = max(dof_r, float(r["dof"]))
        return bool(np.any(np.abs(dz) < 1e-12)), half, dof_r, bad

    lines = [f"{'n_focus':>8}{'홀짝':>6}{'dz=0 포함':>11}{'금지대 반폭':>12}"
             f"{'보고 DOF':>10}{'규격 밖 점':>11}"]
    even_bad, odd_clean = 0, 0
    EVENS, ODDS = (20, 32, 64), (21, 33, 65)
    for nf in (20, 21, 32, 33, 64, 65):
        has0, half, dof_r, bad = probe(nf)
        if nf in EVENS:
            even_bad += (bad > 0)
        else:
            odd_clean += (bad == 0)
        lines.append(f"{nf:>8}{('홀수' if nf % 2 else '짝수'):>6}"
                     f"{('예' if has0 else '아니오'):>11}{half:>12.2f}"
                     f"{dof_r:>10.1f}{bad:>11}")

    # rank 가 짝수를 홀수로 올리는가
    fns = {"cd_error": lambda c: 0.3 * c["defocus"] ** 2 / 1000.0,
           "nils": lambda c: np.full(c["defocus"].shape, 3.0),
           "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0)}
    b = bundles_from(fns)
    common = dict(pitch=200.0, duty=0.5, rel_p95=0.0, n_na=2, n_sig=2,
                  focus_span=150.0, dose_span=0.15, n_dose=21,
                  el_target=10.0, conservative=False)
    r_even = RC.rank(b, FEAT, {"flare": 0.02, "msd_nm": 2.0, "blur_nm": 10.0,
                               "z6": 0.02, "z11": 0.01},
                     n_focus=32, **common)
    r_odd = RC.rank(b, FEAT, {"flare": 0.02, "msd_nm": 2.0, "blur_nm": 10.0,
                              "z6": 0.02, "z11": 0.01},
                    n_focus=33, **common)
    bumped = len(r_even["focus"]) == 33 and len(r_odd["focus"]) == 33
    bump_noted = any("홀수로 올림" in t for t in r_even["notes"])
    no_false_note = not any("홀수로 올림" in t for t in r_odd["notes"])
    has_zero = bool(np.any(np.abs(r_even["focus"]) < 1e-9))

    ok = (even_bad == len(EVENS) and odd_clean == len(ODDS)
          and bumped and bump_noted and no_false_note and has_zero)
    report("초점 격자 홀짝과 DOF 방향 [해석]", ok, lines + [
        "",
        f"짝수 격자가 규격 밖 점을 품은 윈도를 보고 : {even_bad}/{len(EVENS)} "
        f"(음성 대조. 0 이면 이 케이스는 아무것도 지키지 못한다)",
        f"홀수 격자는 깨끗 : {odd_clean}/{len(ODDS)}",
        f"rank 가 32 -> {len(r_even['focus'])} 점으로 올림 : "
        f"{'예' if bumped else '아니오'}",
        f"올린 사실을 주의로 보고 : {'예' if bump_noted else '아니오'}, "
        f"홀수에는 그 주의를 달지 않음 : {'예' if no_false_note else '아니오'}",
        f"올린 격자에 dz=0 포함 : {'예' if has_zero else '아니오'}",
        "",
        "홀수 격자는 dz=0 을 반드시 표본에 넣으므로 Bossung 극값 근처의",
        "규격 초과 영역을 건너뛸 수 없다. 짝수는 건너뛰고 DOF 를 과대보고한다.",
        "이것은 과소평가가 아니라 **방향이 뒤집히는** 오류다.",
    ])


# ----------------------------------------------------------------------------
# Case 12 : 모형 오차 출처의 우선순위 [불변량]
# ----------------------------------------------------------------------------
def case12_rel_p95_source():
    """보수 공차에 **이 모형의** 오차를 쓰는가.

    보수 공차 = `pw_tol - rel_p95` 이고 그 뜻은 '이 모형의 오차를 최악으로
    떼어낸다' 다. 다른 모형의 숫자를 쓰면 뜻이 사라진다. 실제로 소형
    데이터셋으로 학습한 모형은 공차의 122 % 가 나왔는데 기본 상수는 97 %
    였다. 숫자가 두 곳에 따로 적혀 어긋나는 11 주차 54 번과 같은 자리다.

    전체가 아니라 **최악 영역**을 쓰는 것도 함께 본다. 전체 95 백분위가 공차
    안이어도 z6·z11 영역은 154 %·141 % 였고, 추천기는 그 영역의 조건도
    내놓는다. 전체를 쓰면 보수 공차가 낙관적이 된다.

    기준값이 필요 없다. 우선순위와 출처 문구만 본다.
    """
    fns = {"cd_error": lambda c: np.zeros(c["defocus"].shape),
           "nils": lambda c: np.full(c["defocus"].shape, 3.0),
           "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0)}

    def with_measured(ms):
        b = bundles_from(fns)
        for lab in b:
            if ms is None:
                b[lab].pop("measured", None)
            else:
                b[lab]["measured"] = dict(ms)
        return b

    FULL = {"rel_p95_overall": 0.05, "rel_p95_worst_region": 0.09,
            "worst_region_name": "보강 영역", "n_train_rows": 100}
    ONLY_OVERALL = {"rel_p95_overall": 0.05, "rel_p95_worst_region": None,
                    "worst_region_name": None, "n_train_rows": 100}

    v_over, s_over, _ = RC.resolve_rel_p95(with_measured(FULL), 0.03)
    v_worst, s_worst, _ = RC.resolve_rel_p95(with_measured(FULL), None)
    v_all, s_all, _ = RC.resolve_rel_p95(with_measured(ONLY_OVERALL), None)
    v_def, s_def, ms_def = RC.resolve_rel_p95(with_measured(None), None)

    # cd_error 가 없어도(다른 라벨만 있어도) 터지지 않아야 한다.
    b_partial = {k: v for k, v in with_measured(FULL).items() if k != "cd_error"}
    v_part, _s_part, _ = RC.resolve_rel_p95(b_partial, None)

    checks = [
        ("직접 지정이 최우선", v_over == 0.03 and "직접 지정" in s_over),
        ("번들 실측은 최악 영역", v_worst == 0.09 and "최악 영역" in s_worst),
        ("전체만 있으면 전체", v_all == 0.05 and "전체" in s_all),
        ("실측 없으면 11 주차 상수",
         v_def == RC.REL_P95_DEFAULT and "11 주차" in s_def),
        ("실측 없을 때 measured 는 None", ms_def is None),
        ("cd_error 없어도 동작", v_part == 0.09),
        # 음성 대조 : 최악 영역이 전체보다 커야 이 구분이 의미가 있다.
        ("최악 > 전체 (구분이 의미 있음)", v_worst > v_all),
    ]
    ok = all(c[1] for c in checks)
    report("모형 오차 출처의 우선순위", ok, [
        f"{'검사':<34}판정",
        *[f"{nm:<34}{'예' if good else '아니오'}" for nm, good in checks],
        "",
        f"직접 지정    : {v_over:.4f}  <- {s_over}",
        f"번들 실측    : {v_worst:.4f}  <- {s_worst}",
        f"전체만 있음  : {v_all:.4f}  <- {s_all}",
        f"실측 없음    : {v_def:.4f}  <- {s_def}",
        "",
        "보수 공차의 뜻은 '이 모형의 오차를 떼어낸다' 다. 다른 모형의 숫자를",
        "쓰면 조용히 잘못 잡힌다. 그래서 번들이 실측치를 직접 들고 간다.",
    ])


# ----------------------------------------------------------------------------
# Case 13 : dz=0 이 격자점이라는 전제 [불변량]
# ----------------------------------------------------------------------------
def case13_focus_grid_contains_zero():
    """66 번의 보호 장치가 전제를 실제로 세우는가.

    66 번에서 `n_focus` 를 홀수로 강제했고 근거는 'dz=0 이 격자점이어야
    한다' 였다. 그런데 홀수는 **범위가 대칭일 때만** 그 전제를 세운다.
    추천기는 초점 격자를 학습 범위로 깎고(`clip_grid_to_domain`) 대칭화는
    거울일 때만 한다. 비거울 + 비대칭 경로에서는 홀수여도 dz=0 이 격자점이
    아니다. 주석이 코드보다 많이 주장하고 있었다.

    실측해 보니 지금 수치로는 해롭지 않았다. 까닭은 해석으로 설명된다.
    환형 통과집합이 하나로 병합되려면 실패 중심 반폭이 min|dz| 보다 좁아야
    하고, 그 dose 구간의 폭은 `K*min|dz|^2/s` 로 min|dz|^2 에 비례한다.
    중심에 거의 붙은 격자에서는 미시적이고, 걸려도 단일 행이라 el_upper 가
    EL 10 % 에 못 미쳐 `dof_bracket_at_el` 이 걸러낸다. 짝수 격자가 위험했던
    이유도 같은 식으로 설명된다. min|dz| = 간격/2 로 크기 때문이다.

    그래도 구조로 닫는다. 무해함이 'dose 격자가 촘촘하다' 는 조건에 딸려
    있어서다(`--n-dose 7` 이면 단일 병합 행의 el_upper 가 12.7 % 로 10 % 를
    넘는다). 주석으로 지킬 수 있는 성질이 아니다.

    기준값이 필요 없다. 격자의 성질과 '옛 방식은 전제를 못 세운다' 만 본다.
    """
    GRIDS = [(-183.0, 250.0, 33), (-183.0, 250.0, 41), (-150.0, 250.0, 65),
             (-210.0, 250.0, 33), (-5.0, 400.0, 33), (-400.0, 5.0, 33)]
    lines = [f"{'범위':>18}{'n':>5}{'옛 min|dz|':>12}{'새 0포함':>10}"
             f"{'점수':>7}{'범위안':>8}{'균일':>7}{'버린 끝':>10}{'판정':>14}"]
    # 비대칭이어도 끝점이 간격의 정확한 배수면 옛 방식도 0 을 담는다
    # (실측 : [-150, 250] n=65 는 150/6.25 = 24 로 딱 맞는다). 그러니
    # '옛 방식이 전부 놓친다' 로 적으면 이 케이스가 거짓으로 실패한다.
    # 두 갈래로 나눠 각각 다른 것을 지킨다.
    #   놓친 격자 -> 음성 대조. 새 방식이 반드시 0 을 담아야 한다.
    #   이미 담은 격자 -> 양성 대조. 새 방식이 **건드리지 않아야** 한다.
    old_missed, new_fixed, untouched, already = 0, 0, 0, 0
    for lo, hi, n in GRIDS:
        old = np.linspace(lo, hi, n)
        old_has0 = bool(np.abs(old).min() <= 1e-9)
        g, note = PR.centered_focus_grid(lo, hi, n)
        has0 = bool(np.any(np.abs(g) < 1e-12))
        inside = bool(g[0] >= lo - 1e-9 and g[-1] <= hi + 1e-9)
        uniform = bool(g.size < 3 or np.allclose(np.diff(g), g[1] - g[0]))
        step = float(g[1] - g[0]) if g.size > 1 else 0.0
        # 버린 끝은 한 간격을 넘지 않아야 한다. 넘으면 과소보고가 아니라
        # 범위를 통째로 잘라 먹는 것이다.
        lost = max(abs(lo - float(g[0])), abs(hi - float(g[-1])))
        sound = bool(has0 and inside and uniform and lost <= step + 1e-9)
        if old_has0:
            already += 1
            same_as_old = bool(g.size == n
                               and np.allclose(g, old, atol=1e-9)
                               and note is None)
            if sound and same_as_old:
                untouched += 1
            verdict = "양성:그대로" if same_as_old else "양성:바뀜"
        else:
            old_missed += 1
            if sound:
                new_fixed += 1
            verdict = "음성:고침" if sound else "음성:못고침"
        lines.append(f"[{lo:>7.1f},{hi:>6.1f}]{n:>5}"
                     f"{float(np.abs(old).min()):>12.3f}"
                     f"{('예' if has0 else '아니오'):>10}{g.size:>7}"
                     f"{('예' if inside else '아니오'):>8}"
                     f"{('예' if uniform else '아니오'):>7}{lost:>10.2f}"
                     f"{verdict:>14}")

    # 대칭 + 홀수는 **정확히** linspace 와 같아야 한다. 66 번이 보장한 것을
    # 하나도 바꾸지 않았다는 뜻이고, 이게 깨지면 기존 검증이 전부 흔들린다.
    same = 0
    SYMS = [(-250.0, 250.0, 33), (-500.0, 500.0, 41), (-150.0, 150.0, 17),
            (-183.0, 183.0, 65)]
    for lo, hi, n in SYMS:
        g, note = PR.centered_focus_grid(lo, hi, n)
        if (g.size == n and np.allclose(g, np.linspace(lo, hi, n), atol=1e-9)
                and note is None):
            same += 1

    # 0 을 가로지르지 않는 한쪽 스윕은 중심화할 것이 없다. 건드리지 않는다.
    one_sided = 0
    for lo, hi, n in [(10.0, 400.0, 33), (-400.0, -10.0, 33)]:
        g, note = PR.centered_focus_grid(lo, hi, n)
        if g.size == n and np.allclose(g, np.linspace(lo, hi, n), atol=1e-9):
            one_sided += 1

    # 간격이 한쪽보다 넓어 0 하나로 붕괴하는 경우 : 포기하고 **메모를 단다**.
    g_deg, note_deg = PR.centered_focus_grid(-183.0, 250.0, 2)
    degraded = bool(g_deg.size == 2 and note_deg is not None)

    # 추천기까지 이어지는가. 비대칭 학습 범위 + 거울 미적용 경로.
    fns = {"cd_error": lambda c: 0.3 * c["defocus"] ** 2 / 1000.0,
           "nils": lambda c: np.full(c["defocus"].shape, 3.0),
           "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0)}
    dom = make_domain(defocus=(-183.0, 250.0))
    b = bundles_from(fns, domain=dom)
    r = RC.rank(b, FEAT, {"flare": 0.02, "msd_nm": 2.0, "blur_nm": 10.0,
                          "z6": 0.02, "z11": 0.01},          # 양수 -> 거울 안 씀
                pitch=200.0, duty=0.5, rel_p95=0.0, n_na=2, n_sig=2,
                focus_span=300.0, n_focus=33, dose_span=0.15, n_dose=21,
                el_target=10.0, conservative=False)
    f = r["focus"]
    rank_has0 = bool(np.any(np.abs(f) < 1e-12))
    rank_noted = any("dz=0" in t for t in r["notes"])
    rank_inside = bool(f[0] >= dom["axes"]["defocus"]["min"] - 1e-9
                       and f[-1] <= dom["axes"]["defocus"]["max"] + 1e-9)
    # 음성 대조 : 이 범위는 애초에 비대칭이어야 한다. 대칭이면 이 경로가
    # 아무것도 시험하지 않는다.
    rank_asym = bool(abs(abs(float(f[0])) - abs(float(f[-1]))) > 1e-9)
    rank_ran = bool(r["rows"])

    ok = (old_missed > 0 and new_fixed == old_missed
          and already > 0 and untouched == already
          and same == len(SYMS) and one_sided == 2 and degraded
          and rank_has0 and rank_noted and rank_inside and rank_asym
          and rank_ran)
    report("초점 격자가 dz=0 을 담음", ok, lines + [
        "",
        f"음성 대조 : 옛 방식이 dz=0 을 놓친 격자 {old_missed}/{len(GRIDS)} 개, "
        f"그 전부를 새 방식이 고침 {new_fixed}/{old_missed} "
        f"(놓친 격자가 0 개면 이 케이스는 아무것도 지키지 못한다)",
        f"양성 대조 : 옛 방식도 이미 담은 격자 {already}/{len(GRIDS)} 개, "
        f"그 전부를 새 방식이 그대로 둠 {untouched}/{already} "
        f"(바꾸면 멀쩡한 격자를 괜히 거칠게 만든 것이다)",
        f"대칭+홀수에서 linspace 와 완전히 동일 : {same}/{len(SYMS)} "
        f"(66 번이 보장한 것을 바꾸지 않았다는 뜻)",
        f"한쪽 스윕은 그대로 둠 : {one_sided}/2",
        f"간격이 너무 넓으면 포기하고 메모 : {'예' if degraded else '아니오'}",
        "",
        f"추천기 비대칭 경로 : 격자 [{f[0]:.1f}, {f[-1]:.1f}] nm, {f.size} 점",
        f"  dz=0 포함 {'예' if rank_has0 else '아니오'} / "
        f"재배치 보고 {'예' if rank_noted else '아니오'} / "
        f"학습 범위 안 {'예' if rank_inside else '아니오'} / "
        f"비대칭 유지 {'예' if rank_asym else '아니오'} (음성 대조) / "
        f"후보 평가 {'예' if rank_ran else '아니오'}",
        "",
        "홀수 올림은 '범위가 대칭' 이라는 전제에 딸려 있었고 그 전제는 학습",
        "범위로 깎는 경로에서 세워지지 않았다. 버리는 끝은 한 간격 이하이고",
        "격자가 [lo, hi] 의 부분집합이므로 DOF 는 작게 나올 뿐 크게 나오지",
        "않는다. 즉 양보는 과소보고 방향이다.",
    ])


# ----------------------------------------------------------------------------
# Case 14 : 추천 운전점 [해석]
# ----------------------------------------------------------------------------
def case14_operating_point():
    """추천기가 내놓는 운전점이 창의 중심이고, 거울에서 부호가 뒤집히는가.

    12 주차 목표는 '공정 조건 추천' 이다. 창의 **폭**만 내놓으면 받는 사람이
    장비에 넣을 값이 없다. 그래서 노광량과 초점 오프셋을 함께 내놓는다.

    참값을 해석으로 만든다. `CD/T = 1 + s(d-1) - K(dz - dz0)^2` 로 심으면
    최적 초점이 정확히 `dz0` 이고, 통과집합이 `(dz - dz0)` 에 대해 대칭이므로
    **최대 DOF 행의 초점 구간 중심도 dz0** 이다. 격자 분해능 안에서 맞아야
    한다.

    거울이 중요하다. `CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11)` 로 음수 수차 장비를
    양수로 바꿔 평가하므로, 양수 쪽에서 얻은 최적 초점 `dz*` 는 원래 장비에서
    `-dz*` 다. DOF 와 EL 은 폭이라 그대로지만 **오프셋은 위치**라서 그대로
    내놓으면 초점을 반대쪽으로 맞추게 된다. 부호를 되돌리는지 음성 대조와
    함께 본다.
    """
    S, K = 1.20, 1.0e-5
    DZ0 = 60.0                       # 심은 최적 초점 [nm]
    PITCH, DUTY = 200.0, 0.5
    TGT = PITCH * DUTY

    def planted_cd_error(c):
        return TGT * (S * (c["dose_rel"] - 1.0)
                      - K * (c["defocus"] - DZ0) ** 2)

    fns = {"cd_error": planted_cd_error,
           "nils": lambda c: np.full(c["defocus"].shape, 3.0),
           "cd_3sigma": lambda c: np.full(c["defocus"].shape, 1.0)}
    # 초점 범위는 **대칭**으로 둔다. 거울 경로가 초점 격자를 대칭으로
    # 줄이므로(65 번), 비대칭이면 두 경로의 격자가 달라져 부호만 비교할 수
    # 없다.
    #
    # 수차 범위는 **양수뿐**으로 둔다. 거울은 학습 범위가 한쪽 부호만 덮을
    # 때 쓰는 것이므로(`mirror_request` 의 pos_only 조건), 음수까지 열어
    # 두면 거울이 아예 발동하지 않는다. 처음에 (-0.05, 0.05) 로 적었다가
    # '부호 뒤집힘 아니오' 가 나왔는데, 그건 추천기가 틀린 것이 아니라 이
    # 케이스가 거울 경로를 밟지 못한 것이었다. 실제 데이터도 양수뿐이다.
    dom = make_domain(defocus=(-400.0, 400.0), z6=(0.0, 0.05),
                      z11=(0.0, 0.05))
    b = bundles_from(fns, domain=dom)

    common = dict(pitch=PITCH, duty=DUTY, rel_p95=0.0, n_na=1, n_sig=1,
                  focus_span=400.0, n_focus=81, dose_span=0.20, n_dose=81,
                  el_target=5.0, conservative=False)
    tool_pos = {"flare": 0.02, "msd_nm": 2.0, "blur_nm": 10.0,
                "z6": 0.02, "z11": 0.01}             # 양수 -> 거울 안 씀
    tool_neg = dict(tool_pos, z6=-0.02, z11=-0.01)   # 둘 다 음수 -> 거울

    r_pos = RC.rank(b, FEAT, dict(tool_pos), **common)
    r_neg = RC.rank(b, FEAT, dict(tool_neg), **common)
    m_pos = r_pos["rows"][0]["nominal"]
    m_neg = r_neg["rows"][0]["nominal"]

    step_dz = float(abs(r_pos["focus"][1] - r_pos["focus"][0]))
    fc_pos = float(m_pos["focus_center"])
    fc_neg = float(m_neg["focus_center"])
    z_lo, z_hi = m_pos["focus_range"]
    d_lo, d_hi = m_pos["dose_range"]

    # (1) 심은 최적 초점을 격자 분해능 안에서 되찾는가. 구간 끝이 격자점이라
    #     양 끝이 각각 반 간격까지 밀릴 수 있으므로 한 간격까지 허용한다.
    found = abs(fc_pos - DZ0) <= step_dz + 1e-9
    # (2) 중심이 정말 보고된 구간의 중간값인가 (정의 확인).
    is_mid = (abs(fc_pos - 0.5 * (z_lo + z_hi)) < 1e-9
              and abs(float(m_pos["dose_center"])
                      - 0.5 * (d_lo + d_hi)) < 1e-9)
    # (3) 거울에서 부호가 뒤집히는가. 폭은 그대로여야 한다.
    flipped = abs(fc_neg + fc_pos) < 1e-9
    width_same = abs(float(m_neg["dof"]) - float(m_pos["dof"])) < 1e-9
    # 구간도 함께 뒤집혀야 한다. 중심만 뒤집고 구간을 두면 둘이 어긋난다.
    zn_lo, zn_hi = m_neg["focus_range"]
    rng_flipped = (abs(zn_lo + z_hi) < 1e-9 and abs(zn_hi + z_lo) < 1e-9)
    # (4) 음성 대조 : 애초에 0 이 아닌 오프셋이어야 뒤집힘이 뜻을 가진다.
    nonzero = abs(fc_pos) > step_dz
    # (5) 노광량은 거울과 무관해야 한다.
    dose_same = abs(float(m_neg["dose_center"])
                    - float(m_pos["dose_center"])) < 1e-9
    # (6) 창이 없는 후보에는 운전점이 없어야 한다. 없는 창의 중심을 적으면
    #     '쓸 수 있는 조건' 으로 읽힌다.
    empty = RC.window_metrics(np.zeros((3, 3)), TGT, -1.0,
                              np.ones(3), np.zeros(3), 5.0)
    none_when_empty = (empty["dose_center"] is None
                       and empty["focus_center"] is None)
    # (7) '잘림' 과 '창 경계 접촉' 을 구분하는가. [13주차]
    #     `window_saturated` 는 통과 영역 **전체**의 경계 접촉을 본다.
    #     그런데 DOF 와 운전점은 그중 **한 직사각형**에서 나온다. 둘을
    #     같은 표시로 쓰면, 보고된 직사각형이 안쪽인데도 '포화' 로 찍혀
    #     '이 운전점이 잘렸다' 로 오해하게 된다. 실제로 그런 출력이 났다.
    #
    #     구성 : 순수 2차 Bossung 에서 K 를 '최적 노광 행이 초점 격자
    #     양끝까지 통과' 하도록 잡는다. 그 행은 EL 이 0 에 가까우므로
    #     EL 10 % 를 만드는 직사각형은 여러 노광 행에 걸쳐야 하고, 그러면
    #     초점 폭이 좁아져 **안쪽**에 놓인다. 즉 전체는 닿고 직사각형은
    #     안 닿는 상태를 해석적으로 만들 수 있다.
    SPAN, NZ, ND = 30.0, 7, 21
    fz = np.linspace(-SPAN, SPAN, NZ)
    dr = np.linspace(0.90, 1.10, ND)
    _s, _tol2 = 1.20, 0.07
    _K = _tol2 / SPAN ** 2          # d=1 행이 양끝에서 딱 규격 경계
    cd_q = TGT * (1.0 + _s * (dr[:, None] - 1.0) - _K * fz[None, :] ** 2)
    mq = RC.window_metrics(cd_q, TGT, _tol2, dr, fz, 10.0)
    zq_lo, zq_hi = mq["focus_range"]
    whole_touch = bool(mq["focus_edge"])
    row_inside = not mq["row_focus_edge"]
    really_inside = bool(zq_lo > fz.min() + 1e-9 and zq_hi < fz.max() - 1e-9)
    # 음성 대조 : 두 판정이 같은 값이면 이 구분이 아무 일도 하지 않는다.
    edge_split = bool(whole_touch and row_inside and really_inside
                      and whole_touch != bool(mq["row_focus_edge"]))

    # (8) 두 반환 경로의 **키 집합이 같아야 한다.** 모양이 다르면 호출부가
    #     KeyError 로 죽거나, 더 나쁘게는 .get 으로 조용히 0 을 읽는다.
    #     코드에 '같게 둔다' 는 주석이 있지만 주석은 보호 장치가 아니고,
    #     운전점 키 4 개를 양쪽에 손으로 넣었다. 69 번과 같은 구조라 묶는다.
    #     `empty_reason` 은 빈 창에만 있는 것이 의도이므로 제외한다.
    same_keys = (set(empty) - {"empty_reason"}) == set(m_pos)

    ok = (found and is_mid and flipped and width_same and rng_flipped
          and nonzero and dose_same and none_when_empty and same_keys
          and edge_split)
    report("추천 운전점 [해석]", ok, [
        f"심은 최적 초점 dz0 = {DZ0:.1f} nm, 초점 격자 간격 {step_dz:.2f} nm",
        "",
        f"{'검사':<38}판정",
        f"{'심은 dz0 을 격자 분해능 안에서 되찾음':<38}"
        f"{'예' if found else '아니오'}  (얻은 값 {fc_pos:+.2f} nm)",
        f"{'중심이 보고된 구간의 중간값':<38}{'예' if is_mid else '아니오'}",
        f"{'거울에서 초점 오프셋 부호 뒤집힘':<38}"
        f"{'예' if flipped else '아니오'}  ({fc_pos:+.2f} -> {fc_neg:+.2f})",
        f"{'거울에서 초점 구간도 함께 뒤집힘':<38}"
        f"{'예' if rng_flipped else '아니오'}",
        f"{'거울에서 DOF 는 그대로':<38}{'예' if width_same else '아니오'}",
        f"{'거울에서 노광량은 그대로':<38}{'예' if dose_same else '아니오'}",
        f"{'오프셋이 0 이 아님 (음성 대조)':<38}"
        f"{'예' if nonzero else '아니오'}",
        f"{'창이 없으면 운전점도 없음':<38}"
        f"{'예' if none_when_empty else '아니오'}",
        f"{'창 전체 접촉과 직사각형 잘림을 구분':<38}"
        f"{'예' if edge_split else '아니오'}"
        f"  (전체 {'닿음' if whole_touch else '안닿음'} / 직사각형 "
        f"[{zq_lo:.1f}, {zq_hi:.1f}] vs 격자 [{fz.min():.1f}, {fz.max():.1f}])",
        f"{'두 반환 경로의 키 집합이 같음':<38}"
        f"{'예' if same_keys else '아니오'}"
        f"  (빈 창 {len(set(empty)) - 1} / 성공 {len(set(m_pos))} 키)",
        "",
        f"보고된 운전점 : 노광 배율 {m_pos['dose_center']:.4f} "
        f"[{d_lo:.4f}, {d_hi:.4f}], 초점 {fc_pos:+.1f} nm "
        f"[{z_lo:.1f}, {z_hi:.1f}] nm, DOF {m_pos['dof']:.1f} nm",
        "",
        "DOF 와 EL 은 폭이라 거울에서 그대로지만 오프셋은 **위치**다.",
        "부호를 되돌리지 않으면 초점을 반대쪽으로 맞추게 된다. 폭만",
        "맞는지 보는 검사로는 이 종류를 잡을 수 없다.",
    ])


# ----------------------------------------------------------------------------
# Case 15 : 학습이 쓰는 파일 이름을 추천기가 찾는가 [불변량]
# ----------------------------------------------------------------------------
def case15_bundle_path_convention():
    """학습 -> 추천기 사이의 **파일 이름 규약**을 왕복으로 확인한다.

    실제로 부딪힌 벽이다(12 차 보완 72 번). 학습은
    `<접두사>_<모형>_<라벨>.joblib` 로 저장하는데(`--model mlp` 처럼 하나만
    지정해도 모형 구간이 들어간다) 추천기는 `<접두사>_<라벨>.joblib` 만
    찾고 있었다. 그래서 **추천기가 인쇄한 처방을 그대로 따라도 추천기가
    돌지 않았다.** 처방이 모형 구간을 떼어 내고 있어서 순환까지 했다.

    `verify.py` K6(4) 는 '추천기가 요구하는 라벨을 학습이 만들 수 있는가' 만
    보았다. 라벨은 전부 만들 수 있었으므로 그 검사는 통과했고, 이름 규약은
    아무도 보지 않았다. `load_bundles` 에 대한 시험이 **한 건도 없었다**.

    기준값이 필요 없다. 이름을 실제로 만들어 두고 읽히는지만 본다.
    저장 경로 표현식은 `train_surrogate.py` 원문에서 직접 떠 와 대조하므로,
    학습 쪽이 규약을 바꾸면 여기서 바로 깨진다.
    """
    import tempfile
    try:
        import joblib
    except ImportError:
        report("번들 파일 이름 규약", False, ["joblib 이 없어 확인할 수 없다"])
        return

    dom = make_domain()

    def bundle(lab):
        # 모형 자리에 자리표시자를 넣는다. 이 케이스는 **이름 규약**만 보고
        # 예측을 한 번도 하지 않으므로 진짜 모형이 필요 없다. 지역 함수나
        # 람다를 넣으면 pickle 이 되지 않는다(joblib 은 pickle 기반이다).
        # 진짜 모형으로 도는 경로는 Case 14 를 포함한 다른 케이스가 덮는다.
        return {"model": "자리표시자",
                "features": FEAT, "categorical": [False] * len(FEAT),
                "label": lab, "domain": dom, "pw_tol": 0.0707}

    labs = list(RC.LABELS_REQUIRED)
    checks = []
    with tempfile.TemporaryDirectory() as td:
        # (1) 학습이 실제로 쓰는 이름 : <접두사>_<모형>_<라벨>
        pre = os.path.join(td, "sm")
        for lab in labs:
            joblib.dump(bundle(lab), f"{pre}_mlp_{lab}.joblib")
        got = RC.load_bundles(pre, labs)
        checks.append(("학습이 쓰는 <접두사>_<모형>_<라벨> 를 찾음",
                       set(got) == set(labs)))
        checks.append(("읽은 모형 종류를 mlp 로 적음",
                       all(b.get("model_kind") == "mlp"
                           for b in got.values())))

        # (2) 모형 구간이 없는 이름도 계속 읽혀야 한다(옛 방식 호환).
        pre2 = os.path.join(td, "flat")
        for lab in labs:
            joblib.dump(bundle(lab), f"{pre2}_{lab}.joblib")
        got2 = RC.load_bundles(pre2, labs)
        checks.append(("모형 구간 없는 이름도 찾음", set(got2) == set(labs)))

        # (3) 라벨마다 다른 모형이 섞이면 거부해야 한다. 서로 다른 근사를
        #     한 예측을 한 표에 올리는 것이므로 조용히 넘어가면 안 된다.
        pre3 = os.path.join(td, "mix")
        joblib.dump(bundle(labs[0]), f"{pre3}_mlp_{labs[0]}.joblib")
        for lab in labs[1:]:
            joblib.dump(bundle(lab), f"{pre3}_hgb_{lab}.joblib")
        mixed_refused = False
        try:
            RC.load_bundles(pre3, labs)
        except SystemExit as e:
            mixed_refused = "다른 모형" in str(e)
        checks.append(("모형이 섞이면 거부", mixed_refused))

        # (4) 음성 대조 : 아무 파일도 없으면 반드시 거부해야 한다. 거부하지
        #     않으면 위 (1)(2) 가 '무엇이든 통과' 를 보고 있는 셈이다.
        empty_refused = False
        try:
            RC.load_bundles(os.path.join(td, "nothing"), labs)
        except SystemExit as e:
            msg = str(e)
            empty_refused = ("필요한 모형 파일이 없습니다" in msg
                             # 처방이 순환하지 않아야 한다. 접두사를 그대로
                             # 적어야 그 이름으로 저장하면 그 이름으로 읽힌다.
                             and "--save-model " + os.path.join(td, "nothing")
                             in msg)
        checks.append(("파일이 없으면 거부하고 처방이 순환하지 않음",
                       empty_refused))

    # (5) 학습 원문의 저장 경로 표현식이 위 두 형태 가운데 하나인가.
    #     문구를 바꿔 규약이 어긋나면 여기서 깨진다.
    src = io.open("train_surrogate.py", encoding="utf-8").read()
    pat_model = 'f"{args.save_model}_{kind}_{tgt}.joblib"'
    pat_flat = 'f"{args.save_model}_{tgt}.joblib"'
    checks.append(("학습 원문의 저장 경로가 지원하는 형태",
                   (pat_model in src) or (pat_flat in src)))

    ok = all(c[1] for c in checks)
    report("번들 파일 이름 규약", ok, [
        f"{'검사':<44}판정",
        *[f"{nm:<44}{'예' if good else '아니오'}" for nm, good in checks],
        "",
        f"추천기가 요구하는 라벨 : {', '.join(labs)}",
        f"학습 원문의 저장 경로 : {pat_model if pat_model in src else pat_flat}",
        "",
        "K6(4) 는 '라벨을 만들 수 있는가' 만 보았고 그 검사는 통과했다.",
        "만들 수 있는 라벨이 **찾을 수 없는 이름**으로 저장되고 있었다.",
        "load_bundles 에 대한 시험이 한 건도 없던 것이 원인이다.",
    ])


# ----------------------------------------------------------------------------
# Case 16 : 학습이 요구하는 입력을 추천기가 줄 수 있는가 [불변량]
# ----------------------------------------------------------------------------
def case16_input_supply():
    """추천기 -> 학습 방향이 아니라 **학습 -> 추천기** 방향을 본다.

    실제로 부딪힌 벽이다(12 차 보완 74 번). 데이터를 이산 조합으로 생성하자
    편광·조명·마스크가 상수가 아니게 되어 모형 입력이 14 열로 늘었는데,
    추천기에는 그 값을 줄 옵션이 없었다. 그래서 실제 데이터셋에서 추천기가
    `[오류] 조건 'pol_code' 의 값이 주어지지 않았습니다` 로 죽었다.

    개발용 소형 데이터는 `--arms` 없이 뽑아 그 세 열이 상수로 빠졌으므로
    (입력 11 열) 끝까지 돌았다. 즉 **개발 데이터와 실제 데이터의 열 구성이
    달라서** 구멍이 드러나지 않았다.

    `verify.py` K6(4) 는 '추천기가 요구하는 **라벨**을 학습이 만들 수 있는가'
    를 본다. 그 반대 방향은 아무도 보지 않았다. 범주형 하나를 고치는 것이
    아니라 그 방향 전체를 고정한다.

    기준값이 필요 없다. 분류의 완전성과 거부 동작만 본다.
    """
    class A:
        def __init__(self, pol=None, source=None, mask=None, search=False):
            self.pol, self.source, self.mask = pol, source, mask
            self.search_arms = search

    known = set()
    for v in RC.SUPPLY.values():
        known.update(v)
    cond = set(DS.CONDITION_KEYS)
    declared = set(RC.UNSUPPLIED)

    checks = []
    # (1) 조건 열 전체가 '공급 가능' 또는 '공급 불가 선언' 중 하나에 들어감.
    #     새 축이 생기면 여기서 걸린다.
    checks.append(("조건 열 전체가 분류됨", not (cond - known - declared)))
    # (2) 선언에 조건 열이 아닌 것이 섞여 있지 않음 (오타 방지).
    checks.append(("공급 불가 선언이 조건 열뿐", not (declared - cond)))
    checks.append(("공급 가능 선언이 조건 열뿐", not (known - cond)))
    # (3) 두 분류가 겹치지 않음.
    checks.append(("두 분류가 겹치지 않음", not (known & declared)))
    # (4) 음성 대조 : 모르는 열을 섞으면 반드시 잡아야 한다.
    checks.append(("모르는 열을 검출 (음성 대조)",
                   RC.unsupplied_inputs(["na", "없는열"]) == ["없는열"]))
    # (5) 음성 대조 : 공급 불가로 선언한 열도 잡아야 한다.
    checks.append(("공급 불가 열을 검출 (음성 대조)",
                   RC.unsupplied_inputs(["na", "z5"]) == ["z5"]))
    # (6) 정상 조합은 통과해야 한다. 아니면 (4)(5) 가 '무엇이든 잡는다' 다.
    checks.append(("정상 입력은 통과",
                   RC.unsupplied_inputs(sorted(known)) == []))

    # --- 범주형 공급 경로 ---------------------------------------------------
    CF = ["na", "sigma_out", "pitch", "duty", "defocus", "dose_rel",
          "flare", "msd_nm", "blur_nm", "z6", "z11",
          "pol_code", "source_code", "mask_code"]
    dom = make_domain(feat=[f for f in CF if f not in RC.CAT_AXES])
    dom["categorical"] = {"pol_code": [1.0, 3.0], "source_code": [0.0, 2.0],
                          "mask_code": [0.0, 1.0]}

    # 값을 안 주면 거부하고 관측된 값을 알려 주어야 한다.
    refused = ""
    try:
        RC.resolve_arms(CF, dom, A())
    except SystemExit as e:
        refused = str(e)
    checks.append(("범주형 미지정이면 거부",
                   "범주형 조건을 입력으로" in refused))
    checks.append(("거부 시 관측된 값을 보여 줌",
                   "TE" in refused and "unpolarized" in refused))
    checks.append(("거부 시 --search-arms 를 안내", "--search-arms" in refused))

    # 고정 지정은 코드로 바뀌어야 한다.
    arms, _n = RC.resolve_arms(CF, dom, A("TE", "dipole", "att-psm"))
    checks.append(("고정 지정이 코드로 바뀜",
                   arms == [{"pol_code": 1, "source_code": 2,
                             "mask_code": 1}]))
    # 관측되지 않은 수준은 거부해야 한다. 범주형은 보간이 없다.
    bad = ""
    try:
        RC.resolve_arms(CF, dom, A("TM", "dipole", "att-psm"))
    except SystemExit as e:
        bad = str(e)
    checks.append(("관측 안 된 수준은 거부", "없는 값입니다" in bad))

    # --search-arms 는 **실재하는 조합만** 돌아야 한다. 축별 수준의 데카르트
    # 곱(2x2x2 = 8)을 돌면 없는 조합 4 개가 섞인다.
    many, _n2 = RC.resolve_arms(CF, dom, A(search=True))
    from litho import campaign as CP
    checks.append(("탐색 조합 수가 캠페인 목록과 같음",
                   len(many) == len(CP.DISCRETE_ARMS)))
    checks.append(("데카르트 곱이 아님 (음성 대조)", len(many) < 8))
    labs = [RC.arm_label(a) for a in many]
    checks.append(("조합 이름이 사람이 읽는 형태",
                   all("/" in x for x in labs)))

    # 범주 축이 없는 모형은 빈 조합 하나를 돌려주어야 한다(하위호환).
    plain, _n3 = RC.resolve_arms([f for f in CF if f not in RC.CAT_AXES],
                                 dom, A())
    checks.append(("범주 축이 없으면 빈 조합 하나", plain == [{}]))

    # --- 유효 조합의 출처 ---------------------------------------------------
    # 축별 수준만으로는 부족하다. 곱에는 실재하지 않는 조합이 섞인다.
    # 그래서 번들이 **실제로 나온 조합**을 들고 가야 한다. 들고 가지 않는
    # 옛 번들은 생성 코드 상수로 되돌아가되 그 사실을 보고해야 한다.
    REAL = [[1, 0, 0], [1, 0, 1], [1, 2, 1], [3, 2, 1]]
    dom_cc = dict(dom)
    dom_cc["categorical_combos"] = {"axes": list(RC.CAT_AXES),
                                    "combos": [list(c) for c in REAL]}

    cb, src_b = RC.observed_combos(list(RC.CAT_AXES), dom_cc)
    cf, src_f = RC.observed_combos(list(RC.CAT_AXES), dom)
    checks.append(("조합 기록이 있으면 번들에서 읽음", src_b == "bundle"))
    checks.append(("기록이 없으면 되돌림이라고 밝힘", src_f == "fallback"))
    checks.append(("번들 조합이 기록과 정확히 같음",
                   sorted(tuple(a[f] for f in RC.CAT_AXES) for a in cb)
                   == sorted(tuple(c) for c in REAL)))
    # 음성 대조 : 데카르트 곱이면 8 개다. 4 개여야 한다.
    checks.append(("번들 조합이 데카르트 곱이 아님 (음성 대조)",
                   len(cb) == 4))

    # 축별 수준은 통과하지만 **조합**은 없는 경우.
    #   unpolarized(3) / conventional(0) / binary(0) -> 축별로는 전부 관측됨
    GHOST = dict(pol="unpolarized", source="conventional", mask="binary")
    refused_combo = ""
    try:
        RC.resolve_arms(CF, dom_cc, A(**GHOST))
    except SystemExit as e:
        refused_combo = str(e)
    checks.append(("기록이 있으면 없는 조합을 거부",
                   "학습 데이터에 없습니다" in refused_combo))
    checks.append(("거부 시 있는 조합을 보여 줌",
                   "TE / dipole / att-psm" in refused_combo))
    # 음성 대조 : 되돌림 경로에서는 확정할 수 없으므로 거부하지 않고 주의만.
    #   거부해 버리면 옛 번들을 쓸 수 없게 된다.
    ghost_ok, ghost_notes = RC.resolve_arms(CF, dom, A(**GHOST))
    checks.append(("되돌림에서는 거부하지 않음 (음성 대조)",
                   len(ghost_ok) == 1))
    checks.append(("되돌림에서는 주의를 보고",
                   any("되돌아갔" in t for t in ghost_notes)
                   and any("의심" in t for t in ghost_notes)))
    # 기록이 있으면 되돌림 주의가 **없어야** 한다. 있으면 출처 구분이 무의미.
    _ok2, notes_cc = RC.resolve_arms(CF, dom_cc, A(search=True))
    checks.append(("기록이 있으면 되돌림 주의가 없음",
                   not any("되돌아갔" in t for t in notes_cc)))

    ok = all(c[1] for c in checks)
    report("입력 공급 경로의 완전성", ok, [
        f"조건 열 {len(cond)} 개 = 공급 가능 {len(cond & known)} + "
        f"공급 불가 선언 {len(declared)}",
        "",
        f"{'검사':<34}판정",
        *[f"{nm:<34}{'예' if good else '아니오'}" for nm, good in checks],
        "",
        "탐색 조합 : " + " | ".join(labs),
        "",
        "verify.py K6(4) 는 추천기가 요구하는 라벨을 학습이 만들 수 있는지를",
        "본다. 그 반대 방향이 비어 있어서, 학습이 입력으로 쓰기 시작한 열을",
        "추천기가 줄 수 없는 상태가 드러나지 않았다. 개발 데이터와 실제",
        "데이터의 열 구성이 달랐던 것이 원인이다.",
    ])


if __name__ == "__main__":
    print("=" * 78)
    print(" 12주차 검증 : 공정 조건 추천")
    print("=" * 78)
    print()
    case1_grid_orientation()
    case2_window_path()
    case3_domain_refusal()
    case4_nominal_objective_is_flat()
    case5_guarded_window()
    case6_axis_separation()
    case7_sign_mirror()
    case8_dof_bracket()
    case9_mirror_needs_symmetric_focus()
    case10_exact_el_tolerance()
    case11_focus_grid_parity()
    case12_rel_p95_source()
    case13_focus_grid_contains_zero()
    case14_operating_point()
    case15_bundle_path_convention()
    case16_input_supply()
    print("=" * 78)
    print(f" 결과: {sum(_results)}/{len(_results)} 통과")
    print()
    print(VC.summarize(_CASES))
    print("=" * 78)
    # 실패하면 종료 코드 1. 11주차 58 번 참조.
    sys.exit(0 if all(_results) else 1)
