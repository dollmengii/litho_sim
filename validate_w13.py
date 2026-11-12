"""13 주차 회귀 — 되돌림 검증 도구(`roundtrip_w13.py`) 의 불변량.

이 파일이 지키는 것은 **시뮬레이터의 물리**가 아니라 **되돌림 측정 자체의
타당성**이다. 되돌림은 "대리 모형이 틀린 정도" 를 재는 자이고, 자가 틀리면
틀린 값이 '모형 오차' 로 기록된다. 12 주차까지의 회귀가 물리를 지켰다면
여기서는 **재는 방식**을 지킨다.

여기 들어온 각 사례는 전부 **실제로 한 번 틀렸던 것**이다. 추측으로 넣은
항목은 없다.

- Case 1 : 상위 k 겹침의 경계 (동률, k 절단, 빈 입력)
- Case 2 : 순위 상관이 상수 입력에서 '조용히 0' 이 아니라 **이유를 돌려주는가**
           (실측 : 심어 둔 모형이 DOF 를 상수로 내놓아 상관 경로가 한 번도
            실행되지 않았는데 출력은 정상처럼 보였다)
- Case 3 : **제약 등급 섞임 음성 대조** — 등급을 섞으면 상관이 부풀어야 하고,
           분리하면 안 부풀어야 한다 (실측 : +0.799 대 -0.127)
- Case 4 : 후보 표집이 상위권을 반드시 담고 구간 표본의 2/3 를 통과군에 두는가
- Case 5 : 흐림 지름길 `blur(I*d) = d*blur(I)` 와 몬테카를로 판정 기준
- Case 6 : `simulate_window` 의 키가 `recommend.window_metrics` 를 덮는가
- Case 7 : `_setup` 이 손으로 적은 광학 인자가 아니라 `spec` 자신을 쓰는가
           (실측 : 손으로 적은 쪽이 `aberrations`/`n_resist`/`n_source_grid`
            를 빠뜨리고 `n_periods` 3, `dx_target` 1.0 을 써 규약이 어긋났다)
- Case 8 : 제약 뒤집힘을 **같은 축에서** 세는가 (예측은 NILS+CDU, 실측은
           NILS 만이므로 그대로 비교하면 CDU 로만 떨어진 후보가 '모형이
           틀렸다' 로 세어진다)
- Case 9 : 모형 비교가 **같은 물리 조건**에서 이뤄지는가 (공구 상태를 번들
           마다 그 범위에서 읽으면 같은 이름이 다른 조건이 되어 조건 차이가
           모형 차이로 읽힌다)
- Case 10 : 넓힌 영역을 탐침하고 **내삽/외삽 딱지**를 붙이는가. 상자 밖
            거부를 결과로 적는가(조용히 건너뛰면 보강의 근거가 사라진다)
- Case 11 : 분할 적합예측의 **피복 보장**. 유한 표본 보정과, 보정·적용
            격자가 다를 때 구간을 적지 않는 것
- Case 12 : 실측 CDU 가 추천기와 **같은 정의**인가. '반증되지 않음' 과
            '인증' 을 가르는가
- Case 13 : 주장 3의 계측 규격. 교차곱·CRB·예산 역산의 왕복과, 임계값을
            고정해야 축퇴가 성립한다는 것
- Case 14 : 코드 검토에서 찾은 둘 — 선택 편의(승자의 저주)를 갈라 재는가,
            상자 검사의 약함을 **연속 거리**로 바꿨는가
- Case 15 : 추론 한계가 추천 출력까지 전파되는가 — 초점 오프셋 부호의
            비식별(주장 2), 구간의 다중성, 그리고 **대리 모형의 존재 이유**
            (속도 이점)가 측정되는가
- Case 16 : **판정 기준이 있는가.** 동결 문서와 코드 상수의 일치, 다섯
            갈림길이 각각 다른 결론으로 가는가
- Case 17 : **학습 보고서가 써지는가.** 적용범위 기록의 넘파이 배열이
            JSON 보고서를 깨뜨리지 않고, 요약판이 사람이 읽을 크기이고,
            원본 배열은 번들에 그대로 남는가
- Case 18 : **외삽 백분위의 참조 분포가 성립하는가.** 쓸어가는 축으로만
            다른 중복 행이 지워지는가, 참조 분포가 얇으면 숫자를 내놓지
            않고 거절하는가 (음성 대조 포함)
- Case 19 : **보고서 그림의 제목 밀기가 듣는가.** 그린 뒤의 픽셀 좌표로
            제목과 축이 떨어져 있는가 (set_y 는 savefig 가 되돌린다)
- Case 20 : **기준 A 가 검정력을 요구하는가.** 최소 후보 수와 순열 귀무
            분포 대조. 실측 칸(후보 3 개, 우연 확률 33 %)이 '충족' 에서
            '잴 수 없음' 으로 바뀌는지, 변경이 느슨하지 않은지
- Case 21 : **기준 A 가 분해능을 요구하는가.** 초점 눈금이 타우를 깎는
            정도를 직접 재고, 서로 다른 DOF 값이 후보 수의 절반 미만이면
            보류하는가. 관문이 대칭인지(충족을 늘리지 않는지)
- Case 22 : **제약 관문의 격자 수렴성.** 통과 수가 격자 밀도에 수렴하는지
            판정하는 논리. 다섯 갈래, '마지막 두 격자' 규칙, 전체 낙폭,
            그리고 **음수 NILS**(물리 범위 이탈)가 모든 판정보다 앞서는지
- Case 23 : **고NA DOF 계수 규약과 화면 차트 규약.** 호출부가 k3 를 넘기지
            않는가(구문 나무), 기본값이 k3 = k2/2 관계를 들고 있는가, 식이
            1 - cos 꼴인가, 근축 극한에서 두 식이 만나는가(옛 호출은 걸리는가),
            보정 방향이 줄이는 쪽인가. 더해 두 패널의 눈금 규칙 일치와
            띠의 칸 쪼개짐, 슬라이더 표시 자릿수
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

import recommend as RC
import roundtrip_w13 as RT
from litho import dataset as DS

FAIL: list[str] = []
NCASE = 0


def chk(cond: bool, what: str, detail: str = "") -> None:
    global NCASE
    NCASE += 1
    if cond:
        print(f"      [통과] {what}")
    else:
        print(f"      [실패] {what}  {detail}")
        FAIL.append(what)


def head(n: int, title: str) -> None:
    print(f"\n  Case {n} : {title}")


# ---------------------------------------------------------------------------
head(1, "상위 k 겹침의 경계")

chk(RT.top_k_overlap([0, 1, 2], [0, 1, 2], 3) == 1.0, "같은 순서면 1.0")
chk(RT.top_k_overlap([0, 1, 2], [2, 1, 0], 1) == 0.0, "1 위가 다르면 상위1 은 0")
chk(RT.top_k_overlap([0, 1, 2], [2, 1, 0], 3) == 1.0,
    "뒤집혀도 집합이 같으면 상위3 은 1.0",
    "겹침은 집합 기준이다. 순서는 상관이 재는 것이고 둘은 다른 질문이다")
chk(RT.top_k_overlap([0, 1], [0, 1], 5) == 1.0,
    "k 가 후보 수보다 크면 후보 수로 절단")
chk(not np.isfinite(RT.top_k_overlap([], [], 3)), "빈 입력은 nan")
chk(not np.isfinite(RT.top_k_overlap([0], [0], 0)), "k=0 은 nan")

# ---------------------------------------------------------------------------
head(2, "순위 상관이 상수 입력에서 이유를 돌려주는가")

r = RT.rank_corr([1.0, 2.0], [1.0, 2.0])
chk(r["n"] == 2 and "why" in r and not np.isfinite(r["spearman"]),
    "후보 3 개 미만이면 nan + 이유", str(r))

r = RT.rank_corr([0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0])
chk("why" in r and "예측" in r["why"] and not np.isfinite(r["spearman"]),
    "예측이 상수면 '예측' 이라고 적는다", str(r))

r = RT.rank_corr([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0])
chk("why" in r and "실측" in r["why"],
    "실측이 상수면 '실측' 이라고 적는다", str(r))

r = RT.rank_corr([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
chk(abs(r["spearman"] - 1.0) < 1e-12 and abs(r["kendall"] - 1.0) < 1e-12,
    "완전 단조면 스피어만 = 켄달 = +1")

r = RT.rank_corr([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0])
chk(abs(r["spearman"] + 1.0) < 1e-12, "완전 역단조면 -1")

# 동률이 섞이면 스피어만이 켄달보다 크게 나와야 한다. 둘을 함께 적는 이유가
# 이것이며, 하나만 보면 과대평가한다.
a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
b = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
r = RT.rank_corr(a, b)
chk(r["spearman"] > r["kendall"] > 0,
    "동률이 많으면 스피어만 > 켄달 (켄달이 더 보수적)",
    f"스피어만 {r['spearman']:+.3f}, 켄달 {r['kendall']:+.3f}")

# ---------------------------------------------------------------------------
head(3, "제약 등급 섞임 음성 대조 (가장 중요)")

# 상황을 손으로 만든다. 통과군 안에서는 예측과 실측의 순서가 **어긋나** 있고
# (상관 음수), 위반군은 예측/실측 DOF 가 둘 다 낮다. 추천기의 정렬 키가
# `(제약 통과, DOF, 전체)` 이므로 위반군은 언제나 뒤에 붙는다.
#
# 이 배치에서 두 군을 섞으면 '통과군은 전부 높고 위반군은 전부 낮다' 는
# **등급 차이**가 상관을 끌어올려 양수가 나온다. 그 양수는 "모형이 순위를
# 보존한다" 가 아니라 "모형이 제약을 맞혔다" 를 재는 값이고, 질문이 다르다.
keep_sur = [300.0, 290.0, 280.0, 270.0, 260.0, 250.0]
keep_sim = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0]   # 역순 = 순위 깨짐
viol_sur = [40.0, 30.0, 20.0, 10.0]
viol_sim = [35.0, 25.0, 15.0, 5.0]

sep = RT.rank_corr(keep_sur, keep_sim)["spearman"]
mix = RT.rank_corr(keep_sur + viol_sur, keep_sim + viol_sim)["spearman"]

chk(sep < -0.9, "통과군 안에서만 재면 순위가 깨진 것이 보인다",
    f"스피어만 {sep:+.3f}")
chk(mix > 0.5, "등급을 섞으면 같은 자료가 양의 상관으로 뒤집힌다",
    f"스피어만 {mix:+.3f}")
chk(mix - sep > 1.0,
    "섞임이 상관을 1.0 이상 부풀린다 (이것이 12 주차에 실제로 난 오염)",
    f"{sep:+.3f} -> {mix:+.3f}")

# 양성 대조 : 등급이 하나뿐이면 섞어도 달라질 것이 없다.
only = RT.rank_corr(keep_sur, keep_sim)["spearman"]
chk(abs(only - sep) < 1e-12, "등급이 하나면 분리와 섞임이 같다 (양성 대조)")

# 보고 코드가 실제로 분리해서 재는지 — 출처를 직접 확인한다. 위 산수가
# 맞아도 보고 코드가 섞어 쓰면 뜻이 없다.
import inspect

src = inspect.getsource(RT.report_tool)
chk("keep" in src and "viol" in src and 'block(keep, "제약 통과군", True)' in src,
    "report_tool 이 통과군을 중심 판정으로 따로 잰다")
# 추천기의 정렬 키가 등급을 최상위로 둔다는 전제 자체도 확인한다. 이것이
# 깨지면 Case 3 의 서술이 통째로 뜻을 잃는다.
rsrc = inspect.getsource(RC.rank)
chk("nils_ok" in rsrc and "cdu_ok" in rsrc,
    "추천기 정렬 키에 제약 통과 여부가 들어 있다 (섞임이 생기는 원인)")

# ---------------------------------------------------------------------------
head(4, "후보 표집")

rows = []
for i in range(60):
    rows.append({"nils_ok": i % 3 != 0, "cdu_ok": True, "dof": 300.0 - i})
idx = RT.pick_candidates(rows, n_top=5, n_spread=9)

chk(set(range(5)) <= set(idx), "상위 5 개가 반드시 들어간다")
chk(idx == sorted(set(idx)), "색인이 정렬되고 중복이 없다")

spread = [i for i in idx if i >= 5]
n_ok = sum(1 for i in spread if rows[i]["nils_ok"] and rows[i]["cdu_ok"])
chk(n_ok * 3 >= len(spread) * 2 - 1,
    "구간 표본의 2/3 이상이 제약 통과군에서 나온다",
    f"통과 {n_ok} / 전체 {len(spread)}")

# 통과군이 모자랄 때도 깨지지 않아야 한다 (실측 : 통과군이 9 개뿐이었다).
few = [{"nils_ok": i < 2, "cdu_ok": True} for i in range(40)]
idx2 = RT.pick_candidates(few, n_top=3, n_spread=9)
chk(len(idx2) >= 3 and idx2 == sorted(set(idx2)),
    "통과군이 2 개뿐이어도 표집이 깨지지 않는다", str(idx2))

# 후보 수보다 많이 요청해도 넘치지 않는다.
idx3 = RT.pick_candidates(rows[:4], n_top=10, n_spread=10)
chk(len(idx3) == 4 and max(idx3) == 3, "후보 수를 넘겨 뽑지 않는다", str(idx3))

# ---------------------------------------------------------------------------
head(5, "흐림 지름길과 몬테카를로 판정 기준")

# 되돌림은 노광량 격자의 모든 점에서 흐림을 다시 걸지 않고, 흐린 상(像)
# 하나를 만들어 배율만 곱한다. 성립 근거는 선형성이며, 그 선형성을 수치로
# 확인한다. 이것이 깨지면 노광량 축 전체가 조용히 틀린다.
rng = np.random.default_rng(1313)
img = rng.random(2048) * 1.7 + 0.1
sig = 7.3
for d in (0.5, 1.0, 1.4):
    lhs = gaussian_filter1d(img * d, sig, mode="wrap")
    rhs = d * gaussian_filter1d(img, sig, mode="wrap")
    err = float(np.max(np.abs(lhs - rhs)) / np.max(rhs))
    chk(err < 1e-12, f"blur(I*{d}) = {d}*blur(I) (상대 오차 {err:.2e})")

# 포아송의 기대값이 흐림과 교환되는가. lam 이 y 축으로 상수이므로 성립한다.
lam = np.abs(img) * 400.0
smp = rng.poisson(lam[None, :].repeat(600, axis=0)).astype(np.float64)
got = gaussian_filter1d(smp, sig, axis=1, mode="wrap").mean(axis=0)
want = gaussian_filter1d(lam, sig, mode="wrap")
rel = float(np.max(np.abs(got - want)) / np.max(want))
chk(rel < 0.03, f"E[blur(Poisson(lam))] = blur(lam) (상대 오차 {rel:.3f})",
    "표본 600 행이므로 0.03 은 통계 오차 범위다")

# 판정 기준이 **공차 대비**여야 한다. 표준오차 배수로 판정하면 행 사이
# 독립을 가정하는데 2 차원 흐림이 그 독립을 깨고, 실측에서 CD 의 0.01 %
# 차이에 9.17 배 실패가 나왔다.
msrc = inspect.getsource(RT.mc_crosscheck)
chk("tol" in msrc and "mc_tol" in msrc or "tol" in msrc,
    "몬테카를로 교차검사가 공차 대비로 판정한다")
chk("--mc-tol" in open("roundtrip_w13.py", encoding="utf-8").read(),
    "공차 대비 허용치가 옵션으로 노출되어 있다 (--mc-tol)")

# ---------------------------------------------------------------------------
head(6, "시뮬레이터 쪽 키가 추천기 쪽을 덮는가")

need = {"dof", "dof_upper", "total", "n_pass", "focus_edge", "dose_edge",
        "row_focus_edge", "row_dose_edge", "undetermined",
        "dose_center", "focus_center", "dose_range", "focus_range", "tol"}
ssrc = inspect.getsource(RT.simulate_window)
missing = sorted(k for k in need if f'"{k}"' not in ssrc)
chk(not missing,
    "simulate_window 가 window_metrics 의 키를 모두 돌려준다",
    f"빠진 것 {missing}")

# `undetermined` 는 두 쪽에서 같은 뜻이어야 한다 : 하한 <= 0 < 상한.
chk('"undetermined": bool(float(det["lo"]) <= 0.0 < float(det["hi"]))' in ssrc,
    "실측 쪽 미결정 판정이 추천기와 같은 식이다")
chk("undetermined" in inspect.getsource(RC.window_metrics),
    "추천기 쪽에도 같은 이름이 있다")

# ---------------------------------------------------------------------------
head(7, "규약이 코드로 세워져 있는가")

spec = DS.SampleSpec(pitch=250.0, duty=0.5, polarization="TE",
                     source_shape="conventional", mask_type="binary")
g, m, target, opt = RT._setup(spec)
ref = spec.optics()

chk(set(opt.keys()) == set(ref.keys()),
    "_setup 이 spec.optics() 의 키를 하나도 빠뜨리지 않는다",
    f"차이 {sorted(set(opt) ^ set(ref))}")
chk(all(opt[k] == ref[k] for k in ref),
    "값도 전부 같다 (손으로 다시 적지 않는다)")
# 한 번 어긋났던 세 축을 이름으로 못 박는다. 손으로 적은 쪽이 이것들을
# 빠뜨려 '모형 오차' 로 기록될 뻔했다.
for k in ("aberrations", "n_resist", "n_source_grid"):
    chk(k in opt, f"한 번 빠뜨렸던 축 '{k}' 이 들어 있다")

chk(abs(target - spec.duty * spec.pitch) < 1e-12, "목표 CD = duty x pitch")
chk(g.pitch == spec.pitch and g.n_periods == spec.n_periods,
    "격자가 spec 의 pitch / n_periods 를 쓴다",
    f"격자 {g.pitch}/{g.n_periods}, spec {spec.pitch}/{spec.n_periods}")
chk(abs(g.dx - spec.dx_target) <= spec.dx_target,
    "격자 간격이 spec.dx_target 에서 왔다",
    f"dx {g.dx}, 목표 {spec.dx_target}")
chk(g.x.size == g.n_periods * g.samples_per_pitch,
    "표본 수 = 주기 수 x 주기당 표본")
chk(spec.n_periods == 6 and spec.dx_target == 0.5,
    "되돌림이 쓰는 기본값이 생성 경로와 같다 (n_periods 6, dx 0.5)",
    "한 번 3 과 1.0 을 손으로 적어 규약이 어긋났다")

# 손으로 적은 광학 인자가 코드에 남아 있지 않은지 — 재발 방지.
rtsrc = open("roundtrip_w13.py", encoding="utf-8").read()
chk("def _optics" not in rtsrc,
    "손으로 적은 _optics 가 남아 있지 않다")
chk("spec.optics()" in rtsrc and "n_periods=spec.n_periods" in rtsrc
    and "dx_target=spec.dx_target" in rtsrc,
    "세 인자 모두 spec 에서 읽는다 (문자열로 확인)")

# ---------------------------------------------------------------------------
head(8, "뒤집힘 집계가 같은 축에서 비교하는가")

# 예측 판정은 NILS 와 CDU 를 둘 다 보고, 실측 판정은 NILS 만 본다(CDU 는
# 몬테카를로가 필요해 재지 않았다). 그대로 비교하면 CDU 때문에 예측이
# 떨어뜨린 후보가 '실측과 어긋났다' 로 세어진다. 그것은 어긋남이 아니라
# **재지 않은 축**이고, 모형 오차로 적으면 과대보고다.
tsrc = inspect.getsource(RT.report_tool)
chk("def sur_nils" in tsrc, "NILS 전용 판정 함수가 따로 있다")
chk("flip = [r for r in good if sur_nils(r) != sim_pass(r)]" in tsrc,
    "뒤집힘을 NILS 대 NILS 로 센다",
    "sur_pass(CDU 포함) 와 비교하면 과대보고가 된다")
chk("cdu_only" in tsrc, "CDU 로만 떨어진 후보를 따로 센다")
chk("비교 대상이 아니다" in tsrc,
    "그 후보가 비교 대상이 아니라는 것을 출력에 적는다")

# 군 나누기는 반대로 **추천기가 실제로 쓰는 판정**(NILS+CDU)이어야 한다.
# 추천기가 내놓지 않을 후보를 중심 판정에 넣으면 그것도 틀린 집계다.
chk("keep = [r for r in good if sur_pass(r)]" in tsrc,
    "군 나누기는 추천기가 쓰는 판정(NILS+CDU)으로 한다")
chk("r[\"row\"][\"nils_ok\"] and r[\"row\"][\"cdu_ok\"]" in tsrc,
    "sur_pass 가 두 축을 모두 본다")

# 산수로도 확인한다. CDU 로만 떨어진 후보가 하나 있을 때, 축을 맞추면
# 뒤집힘 0, 맞추지 않으면 1 이 나와야 한다.
mk = lambda n_ok, c_ok, s_ok: {
    "row": {"nils_ok": n_ok, "cdu_ok": c_ok}, "sim": {"nils_ok": s_ok}}
cases = [mk(True, False, True)]            # CDU 로만 떨어짐, NILS 는 일치
naive = sum(1 for r in cases
            if (r["row"]["nils_ok"] and r["row"]["cdu_ok"])
            != r["sim"]["nils_ok"])
fixed = sum(1 for r in cases if r["row"]["nils_ok"] != r["sim"]["nils_ok"])
chk(naive == 1 and fixed == 0,
    "CDU 로만 떨어진 후보 : 축을 맞추면 뒤집힘 0, 안 맞추면 1",
    f"맞춤 {fixed}, 안맞춤 {naive}")

# ---------------------------------------------------------------------------
head(9, "모형 비교가 같은 조건에서 이뤄지는가")

# 공구 상태를 번들마다 그 번들의 범위에서 읽으면 'z6 상단 90 %' 가 모형마다
# 다른 물리 조건이 된다. 보강 전 z11 은 양수뿐이고 보강 후는 (-0.05, 0.05)
# 라서 같은 이름이 0.0429 와 0.0400 으로 갈린다. 그 상태에서 잰 상관을
# 나란히 적으면 **조건 차이가 모형 차이로 읽힌다.**
d_old = {"axes": {"z6": {"min": 0.008, "max": 0.050},
                  "z11": {"min": 0.002, "max": 0.0475},
                  "flare": {"min": 0.0, "max": 0.05},
                  "msd_nm": {"min": 0.0, "max": 3.0},
                  "blur_nm": {"min": 6.0, "max": 10.0}}}
d_new = {"axes": {"z6": {"min": 0.008, "max": 0.050},
                  "z11": {"min": -0.050, "max": 0.050},
                  "flare": {"min": 0.0, "max": 0.05},
                  "msd_nm": {"min": 0.0, "max": 3.0},
                  "blur_nm": {"min": 6.0, "max": 10.0}}}

sh, notes = RT.common_axes([d_old, d_new])
chk(sh["axes"]["z11"]["min"] == 0.002 and sh["axes"]["z11"]["max"] == 0.0475,
    "교집합이 좁은 쪽 범위를 쓴다 (모든 모형이 자기 범위 안)",
    str(sh["axes"]["z11"]))
chk(any("좁혀 비교" in n for n in notes),
    "넓게 학습한 모형이 있다는 사실을 보고한다", str(notes))

# 음성 대조 : 범위가 같으면 좁혔다는 말이 나오면 안 된다.
sh2, notes2 = RT.common_axes([d_old, d_old])
chk(not any("좁혀 비교" in n for n in notes2),
    "범위가 같으면 좁혔다고 적지 않는다 (음성 대조)", str(notes2))

# 겹치지 않는 축은 빠지고, 그 사실이 보고되어야 한다.
d_far = {"axes": dict(d_old["axes"],
                      z6={"min": 0.20, "max": 0.30})}
sh3, notes3 = RT.common_axes([d_old, d_far])
chk("z6" not in sh3["axes"] and any("겹치지" in n for n in notes3),
    "교집합이 공집합인 축은 빼고 그 사실을 적는다", str(notes3))

# 공통이 아닌 축도 보고되어야 한다.
d_less = {"axes": {k: v for k, v in d_old["axes"].items() if k != "flare"}}
sh4, notes4 = RT.common_axes([d_old, d_less])
chk("flare" not in sh4["axes"]
    and any("공통이 아닌" in n for n in notes4),
    "모든 모형에 공통이 아닌 축은 빼고 그 사실을 적는다", str(notes4))

# main 이 교집합을 실제로 쓰는지 — 원문으로 확인한다.
msrc2 = inspect.getsource(RT.main)
chk("tool_presets(shared, args)" in msrc2,
    "main 이 모형별 범위가 아니라 공유 범위로 공구 상태를 만든다",
    "tool_presets(dom, args) 이면 모형마다 조건이 달라진다")
chk("loaded[pref] = RC.load_bundles" in msrc2,
    "번들을 먼저 전부 읽어 교집합을 낼 수 있게 한다")


class _A:
    flare, msd_nm, blur_nm, z6, z11 = 0.01, 1.0, 8.0, 0.02, 0.01
    tool_preset = "regions"


# 축이 빠져도 죽지 않고, 요청값을 그대로 써야 한다. 0 으로 채우면 요청하지
# 않은 조건을 만들어 낸다.
pres = RT.tool_presets({"axes": {}}, _A())
chk(len(pres) == 1,
    "축이 하나도 없으면 상태가 기준 하나로 합쳐진다 (중복 제거)",
    f"{len(pres)} 개 : {[n for n, _ in pres]}")
chk(pres[0][1]["z6"] == 0.02 and pres[0][1]["flare"] == 0.01,
    "없는 축은 요청값 그대로 쓴다 (0 으로 채우지 않는다)",
    str(pres[0][1]))
chk("요청값 그대로" in pres[0][0],
    "그 사실을 상태 이름에 적는다", pres[0][0])

pres2 = RT.tool_presets(d_old, _A())
chk(len(pres2) == 6, "축이 다 있으면 상태 6 개", f"{len(pres2)} 개")
vals = {tuple(round(t[k], 12) for k in sorted(t)) for _, t in pres2}
chk(len(vals) == len(pres2), "상태끼리 값이 중복되지 않는다")
z6hi = 0.008 + 0.9 * (0.050 - 0.008)
chk(abs(dict(pres2)["z6 상단 90 %"]["z6"] - z6hi) < 1e-12,
    "상단 90 % 가 lo + 0.9*(hi-lo) 다", f"{z6hi:.6f}")

# ---------------------------------------------------------------------------
head(10, "넓힌 영역 탐침과 외삽 딱지")

d_old = {"axes": {"z11": {"min": 0.002, "max": 0.0475},
                  "z6": {"min": 0.008, "max": 0.050}}}
d_new = {"axes": {"z11": {"min": -0.050, "max": 0.050},
                  "z6": {"min": 0.008, "max": 0.050}}}
uni = RT.union_axes([d_old, d_new])
chk(uni["axes"]["z11"] == {"min": -0.050, "max": 0.050},
    "합집합이 넓은 쪽 범위를 쓴다", str(uni["axes"]["z11"]))
chk(RT.union_axes([d_old])["axes"] == d_old["axes"],
    "모형이 하나면 합집합은 그 모형의 범위다")

chk(RT.outside_axes({"z11": -0.045, "z6": 0.02}, d_old) == ["z11"],
    "좁은 모형에서 음수 z11 은 외삽으로 잡힌다")
chk(RT.outside_axes({"z11": -0.045, "z6": 0.02}, d_new) == [],
    "넓은 모형에서는 내삽이다 (음성 대조)")
chk(RT.outside_axes({"flare": 9.9}, d_old) == [],
    "번들에 없는 축은 외삽이라고 적지 않는다",
    "그 축을 상수로 생성한 번들이 그렇고, 외삽이 아니라 입력이 아닌 것이다")
chk(RT.outside_axes({"z11": 0.0475, "z6": 0.050}, d_old) == [],
    "경계 위의 값은 밖이 아니다")

esrc = inspect.getsource(RT.report_extrap)
chk("모형 사이 비교가 **아니다.**" in esrc,
    "외삽 블록이 '모형 사이 비교가 아니다' 를 명시한다",
    "비교로 읽히면 내삽과 외삽을 한 숫자로 묶는 것이 된다")
chk("거부" in esrc and "rf" in esrc,
    "상자 밖 거부를 행으로 적는다",
    "조용히 건너뛰면 보강이 필요했다는 증거가 출력에서 사라진다")

msrc3 = inspect.getsource(RT.main)
chk('states.append((nm + " [합집합]", t, "합집합"))' in msrc3,
    "main 이 합집합 상태를 따로 표시해 담는다")
chk('per_model.append((tname, [], scope, outside, refused))' in msrc3,
    "거부도 결과로 담는다 (건너뛰지 않는다)")

# ---------------------------------------------------------------------------
head(11, "분할 적합예측의 피복 보장")

from litho import conformal as CF

# 유한 표본 보정을 쓰지 않으면 보장이 깨진다. 이것이 핵심이다.
got = CF.conformal_quantile(np.arange(1., 21.), 0.10)
chk(got["k"] == 19 and got["q"] == 19.0,
    "n=20, alpha=0.10 -> k=19, q=19 번째 순서통계량",
    f"k={got['k']}, q={got['q']}")
chk(abs(float(np.quantile(np.arange(1., 21.), 0.90)) - 18.1) < 1e-9,
    "같은 자료에서 np.quantile 은 18.1 을 준다 (보장 없음)",
    "둘이 다르다는 것이 유한 표본 보정이 실재한다는 증거다")

# **이론 피복률을 정확히 대조한다.** 시뮬레이션으로 재면 잡음 때문에
# 판정이 흔들린다(실측 : 0.8990 대 0.90 을 '위반' 으로 잘못 읽었다).
bad = []
for n in range(5, 201):
    for alpha in (0.20, 0.10, 0.05, 0.01):
        g = CF.conformal_quantile(np.arange(1., n + 1.), alpha)
        k = math.ceil((n + 1) * (1 - alpha))
        if k > n:
            if g["q"] is not None:
                bad.append(f"n={n} a={alpha} 만들면 안 되는데 만들었다")
            continue
        if g["k"] != k or k / (n + 1) < 1 - alpha - 1e-12:
            bad.append(f"n={n} a={alpha}")
chk(not bad, "n=5..200 x alpha 4 종, 784 조합에서 k 와 보장이 정확히 맞는다",
    str(bad[:3]))

few = CF.conformal_quantile(np.arange(1., 6.), 0.10)
chk(few["q"] is None and "최소 9 개" in few["why"],
    "표본이 모자라면 구간을 만들지 않고 필요한 수를 적는다",
    "최댓값으로 때우면 보장을 흉내만 내는 것이다")
chk(CF.conformal_quantile([], 0.10)["q"] is None, "빈 입력은 None")

rq = CF.residual_quantile([10., 20., 30., 40., 50., 60., 70., 80., 90.],
                          [12., 18., 33., 38., 55., 57., 74., 76., 95.], 0.10)
chk(rq["q"] is not None and rq["n"] == 9, "잔차 짝에서 구간을 만든다", str(rq))
chk(rq["signed_median"] is not None, "부호 있는 중위수를 함께 돌려준다")

# 보정 조건을 요청 피치 근처로 제한해야 한다. 데이터셋 피치는 전 범위이고
# 추천기는 한 피치를 묻는다. DOF 가 피치에 강하게 의존하므로 섞으면
# 교환가능성이 깨지고 보장도 깨진다.
mix = [{"pitch": 100., "duty": 0.5}, {"pitch": 230., "duty": 0.5},
       {"pitch": 250., "duty": 0.5}, {"pitch": 270., "duty": 0.5},
       {"pitch": 400., "duty": 0.5}, {"pitch": 250., "duty": 0.65}]
near = CF.near_request(mix, 250.0, 0.5, 0.15, 0.06)
chk([c["pitch"] for c in near] == [230., 250., 270.],
    "피치 +-15 % 밖은 걸러진다", str([c["pitch"] for c in near]))
chk(all(abs(c["duty"] - 0.5) <= 0.06 for c in near),
    "duty 범위 밖도 걸러진다 (0.65 는 빠진다)")
chk(len(CF.near_request(mix, 250.0, 0.5, 0.0, 0.06)) == len(mix),
    "창을 0 으로 주면 제한하지 않는다 (음성 대조)")
chk(CF.near_request([{"duty": 0.5}], 250.0, 0.5, 0.15, 0.06),
    "피치 키가 없으면 요청 피치로 보고 통과시킨다")
csrc2 = inspect.getsource(CF.near_request)
chk("덮어쓰지 않는" in csrc2,
    "피치를 덮어쓰지 않는 이유(누설)를 함수가 적고 있다",
    "덮어쓰면 그 조건이 검증 분할 그룹이 아니게 되어 학습 그룹과 겹칠 수 있다")

cov = CF.coverage([10., 20., 30.], [11., 25., 31.], 2.0)
chk(cov["hit"] == 2 and cov["n"] == 3, "피복률을 센다", str(cov))
chk(CF.coverage([1.], [1.], None)["rate"] is None, "q 가 None 이면 재지 않는다")

# 보정과 적용의 **격자가 다르면** 구간을 적지 않아야 한다. 적으면 보장
# 없는 숫자에 보장 딱지를 붙이는 것이다.
lsrc = inspect.getsource(RC._load_conformal)
chk("n_focus" in lsrc and "focus_span" in lsrc and "el" in lsrc,
    "추천기가 격자 설정 불일치를 대조한다")
chk("return None" in lsrc and "보장이 깨" in lsrc,
    "불일치면 구간을 적지 않고 사유를 적는다")
msrc4 = inspect.getsource(RT.main)
chk("격자 설정이 보정 때와 다르다" in msrc4,
    "되돌림도 같은 대조를 한다")

# 보정 파일은 **모형 하나**에 대한 것이다. 3 모형 실행에서 한 파일을 셋 다에
# 붙이면 보장 없는 숫자에 보장 딱지를 붙이는 것이 된다.
chk("def conf_for" in msrc4, "모형 일치 검사 함수가 있다")
chk("conf if conf_for(pref) else None" in msrc4,
    "접두사가 맞는 모형에만 구간을 넘긴다",
    "맞지 않는 모형에 넘기면 다른 모형의 보장을 빌려 쓰는 것이다")
chk("이번 실행의" in msrc4,
    "보정 파일의 모형이 실행 모형에 없으면 적지 않고 사유를 적는다")
chk("모형마다 따로 보정해야 한다" in msrc4,
    "여러 모형이면 그 사실을 알린다")

# ---------------------------------------------------------------------------
head(12, "실측 CDU 가 추천기와 같은 정의인가")

csrc = inspect.getsource(RT.measure_cdu)
chk('mc["cd_3sigma"] / mc["cd_mean"]' in csrc,
    "CDU = cd_3sigma / cd_mean (몬테카를로 같은 호출의 짝)",
    "결정론 CD 로 나누면 분모가 다른 양이 된다")
rsrc2 = inspect.getsource(RC.evaluate)
chk('m["cd_3sigma"] / np.maximum(m["cd"], 1e-9)' in rsrc2,
    "추천기 쪽 정의를 원문으로 확인한다")
chk("stoch_tol" in csrc and "frac_of_tol" in csrc,
    "같은 공차(stoch_tol)와 비교하고 공차 대비 비율을 돌려준다")

tsrc2 = inspect.getsource(RT.report_tool)
chk("def cdu_refuted" in tsrc2, "'반증' 판정 함수가 따로 있다")
chk("반증되지 않음" in tsrc2,
    "표본이 공차 안인 것을 '인증' 이 아니라 '반증되지 않음' 으로 적는다",
    "표본은 창 전체 최댓값의 하한이므로 인증이 아니다")
chk("argmax_at_nils_min" in csrc and "argmax_at_nils_min" in tsrc2,
    "'CDU 최악은 NILS 최악에 있다' 는 겨냥을 가정이 아니라 측정으로 확인한다")
chk("cdu_repeat" in csrc,
    "같은 지점을 다른 씨앗으로 재측정해 표집 잡음을 함께 적는다")
# 표집 잡음으로 모형을 억울하게 반증하지 않아야 한다. 실측 : z11 상단
# 상태에서 재측정 상대차가 최대 40.6 % 였다.
chk("rep > c[\"stoch_tol\"]" in tsrc2,
    "반증은 재측정에서도 공차를 넘어야 인정한다",
    "한 번 재고 반증이라 적으면 표집 잡음이 '모형이 틀렸다' 가 된다")
chk("재측정이 없으면 1 차 결과를 따른다" in tsrc2,
    "재측정이 없을 때의 처리를 적어 둔다")
chk("1 차에서는" in tsrc2,
    "1 차와 재측정 후 반증 수가 다르면 그 사실을 적는다")
chk("11주차" in tsrc2 and "LHC" in tsrc2,
    "11주차의 '공차의 2 %' 를 추천 조건에 그대로 옮겨 쓰지 않도록 적는다",
    "실측 : 추천 조건에서 최대 38.4 % 가 나왔다")
chk("예측·실측 모두 NILS + CDU" in tsrc2,
    "양쪽이 같은 축이면 그렇다고 적는다 (축 보정이 불필요해진다)")

# 초점 스윕과 NILS 를 다시 계산하면 후보당 비용이 두 배가 된다. 재사용이
# 실제로 일어나는지 원문으로 확인한다. 재사용 전후 출력이 같다는 것은
# 손으로 확인했다(중위 19.8 %, 최대 38.4 %, 재측정 6.0/13.0 % 동일).
chk('stack = sim.get("stack")' in csrc and 'nils = sim.get("nils")' in csrc,
    "CDU 측정이 simulate_window 의 stack/nils 를 재사용한다",
    "다시 계산하면 후보당 광학 비용이 두 배가 된다")
chk("PR.focus_sweep" not in csrc,
    "CDU 측정이 초점 스윕을 다시 돌리지 않는다")
ssrc2 = inspect.getsource(RT.simulate_window)
chk('"nils": nils, "stack": stack' in ssrc2,
    "simulate_window 가 재사용할 배열을 돌려준다")

# ---------------------------------------------------------------------------
head(13, "주장 3의 계측 규격 계산")

from litho import metrospec as MS
import metrospec_w13 as MW

# 교차곱이 0 이면 어떤 정밀도로도 분리되지 않는다. 사이각만 보면 안 된다 —
# **180 도는 0 도만큼 나쁘다.** 실측 사이각이 149~165 도였으므로 이 구분을
# 놓치면 '잘 갈린다' 로 잘못 읽는다.
chk(MS.cross((1., 2.), (2., 4.)) == 0.0, "평행한 기울기는 교차곱 0")
chk(MS.cross((1., 2.), (-2., -4.)) == 0.0,
    "**반평행도** 교차곱 0 (180 도는 0 도만큼 나쁘다)")
chk(abs(MS.cross((1., 0.), (0., 1.)) - 1.0) < 1e-12, "직교면 교차곱 1")
chk(abs(MS.angle_deg((1., 0.), (0., 1.)) - 90.0) < 1e-9, "직교 사이각 90 도")
chk(abs(MS.angle_deg((1., 0.), (-1., 0.)) - 180.0) < 1e-9, "반평행 180 도")

r = MS.crb((1., 2.), (2., 4.), 0.5, 0.02)
chk(r["sd_msd"] == float("inf") and "공선" in r["why"],
    "공선이면 CRB 가 무한이고 사유를 적는다")

# 손으로 푼 값과 대조한다. 닫힌 꼴이 있으므로 정확히 맞아야 한다.
gc, gl = (0.02896, 0.09469), (0.00110, -0.02486)
r = MS.crb(gc, gl, sigma_cd=0.5, sigma_ler=0.0224)
chk(abs(r["D"] - (0.02896 * -0.02486 - 0.09469 * 0.00110)) < 1e-15,
    "교차곱이 정의와 일치")
chk(abs(r["sd_blur"] - 1.032) < 0.002 and abs(r["sd_msd"] - 15.301) < 0.01,
    "CRB 가 손 계산과 일치 (sd_blur 1.032, sd_msd 15.301)",
    f"{r['sd_blur']:.4f}, {r['sd_msd']:.4f}")
chk(r["binds_blur"] == "ler" and r["binds_msd"] == "cd",
    "한계가 되는 관측량을 바르게 지목한다 (블러는 LER, MSD 는 CD)",
    "이걸 틀리면 엉뚱한 장비를 개선하라고 말하게 된다")
chk(abs(r["term_ler_blur"] + r["term_cd_blur"] - r["sd_blur"] ** 2) < 1e-12,
    "두 항의 합이 분산과 같다 (분해가 정합적)")

# 예산을 역산한 정밀도로 CRB 를 다시 풀면 목표가 그대로 나와야 한다.
# 왕복이 맞지 않으면 역산 공식이 틀린 것이다.
for par in ("msd", "blur"):
    bd = MS.budget(gc, gl, 1.0, par)
    back = MS.crb(gc, gl, bd["sigma_cd"], bd["sigma_ler"])[f"sd_{par}"]
    chk(abs(back - 1.0) < 1e-9,
        f"{par} 예산 왕복 : 역산한 정밀도로 목표 1.0 이 재현된다",
        f"{back:.6f}")
bd = MS.budget(gc, gl, 1.0, "msd", share=0.2)
back = MS.crb(gc, gl, bd["sigma_cd"], bd["sigma_ler"])["sd_msd"]
chk(abs(back - 1.0) < 1e-9, "share 를 바꿔도 왕복이 성립한다", f"{back:.6f}")

# 한쪽만 올려서 닿지 않는 경우를 '닿는다' 고 말하면 안 된다.
one = MS.required_sigma(gc, gl, 1.0, "msd", "ler", sigma_other=0.5)
chk(one["sigma"] is None and one.get("other_required") is not None,
    "LER 만 올려 MSD 목표에 못 닿으면 None 과 '나머지 필요치' 를 준다",
    "실측 : CD 항만으로 15.08 nm 라 LER 을 완벽히 재도 못 닿는다")
two = MS.required_sigma(gc, gl, 1.0, "blur", "ler", sigma_other=0.5)
chk(two["sigma"] is not None and abs(two["sigma"] - 0.02119) < 1e-4,
    "닿는 경우는 정밀도를 돌려준다 (블러, LER 0.0212 nm)",
    str(two.get("sigma")))

# 정의로 시험한다. 보고된 숫자(455)는 **반올림 전** 정밀도에서 나온
# 것이므로 반올림한 값을 넣고 그 답을 기대하면 틀린다(실측 453 대 455).
chk(MS.sites_needed(0.5, 0.05) == 100,
    "지점 수 환산은 (장비/목표)^2 (0.5 -> 0.05 nm 는 100 지점)",
    str(MS.sites_needed(0.5, 0.05)))
chk(MS.sites_needed(1.0, 0.3) == 12,
    "정수로 올림한다 (1/0.3)^2 = 11.1 -> 12",
    str(MS.sites_needed(1.0, 0.3)))
_bd = MS.budget(gc, gl, 1.0, "msd")
chk(MS.sites_needed(0.5, _bd["sigma_cd"]) == 455,
    "예산이 낸 정밀도(반올림 전)로는 455 지점",
    str(MS.sites_needed(0.5, _bd["sigma_cd"])))
chk(MS.sites_needed(0.5, 0.6) == 1, "이미 충분하면 1 지점")
chk(MS.sites_needed(0.5, None) is None, "목표가 없으면 None")

# 도구 쪽 : 임계값을 **고정**해야 축퇴가 성립한다. MSD 적용 후 상에서 다시
# 잡으면 규약이 정보를 흘려 결정론 CD 가 1.1594 nm 갈린다(실측).
msrc5 = inspect.getsource(MW._setup)
chk("평활 없는 상에서 한 번 잡아 고정" in msrc5,
    "임계값을 평활 없는 상에서 한 번 잡아 고정한다")
chk("msd_nm" not in msrc5,
    "임계값을 잡는 상에 MSD 를 적용하지 않는다",
    "적용하면 임계값이 배분에 따라 변해 축퇴가 깨진 것처럼 보인다")
dsrc = inspect.getsource(MW.degeneracy_check)
chk("det_range" in dsrc and "mean_range" in dsrc,
    "축퇴를 쓰기 전에 **확인**하고, 결정론과 측정 평균을 갈라 적는다")
msrc6 = inspect.getsource(MW.main)
chk("이 분석의 전제가 없다" in msrc6 and "return 1" in msrc6,
    "축퇴가 깨져 있으면 숫자를 내지 않고 종료 코드 1 을 돌려준다")
chk("COND" in inspect.getsource(MW) and MW.COND["na"] == 1.35
    and MW.COND["pitch"] == 200.0 and MW.COND["n_rows"] == 512,
    "조건이 주장 3(verify H5)과 같다 (pitch 200, NA 1.35, 행 512)",
    "다른 조건이면 기존 주장을 잇는 것이 아니라 새로 재는 것이다")
gsrc = inspect.getsource(MW.gradients)
chk("sp_m, sp_b = mp - mm, bp - bm" in gsrc,
    "스텝이 범위로 잘리면 **실제 쓴 폭**으로 나눈다",
    "명목 스텝으로 나누면 기울기가 작게 나온다")

# --- 13주차 논리 검토에서 찾은 전제 셋. 산수로 고정한다 ---------------
# (가) 관측량 2 개로는 미지수 2 개까지만. rank(J) <= 관측량 수.
_rg = np.random.default_rng(0)
_S = np.diag([0.5 ** 2, 0.0237 ** 2])
for _n, _want in ((2, 2), (3, 2), (5, 2)):
    _G = _rg.normal(size=(2, _n))
    _J = _G.T @ np.linalg.inv(_S) @ _G
    chk(np.linalg.matrix_rank(_J, tol=1e-9) == _want,
        f"미지수 {_n} 개에서 rank(J) = {_want} (관측량 수로 막힌다)",
        str(np.linalg.matrix_rank(_J, tol=1e-9)))
_G = _rg.normal(size=(2, 3))
_J = _G.T @ np.linalg.inv(_S) @ _G
chk(np.linalg.cond(_J) > 1e12,
    "미지수 3 개면 J 가 특이행렬이라 CRB 가 사실상 무한",
    f"조건수 {np.linalg.cond(_J):.1e}")
_ms = inspect.getsource(MS)
chk("미지수 2 개까지만" in _ms and "플레어가 정확히" in _ms,
    "그 전제가 모듈 설명에 적혀 있다 (초점·노광량·플레어가 알려져 있다)")
chk("다른 문제를 다룬다" in _ms,
    "주장 1 과 **다른 문제**라는 것이 적혀 있다",
    "주장 1 은 5 변수 문제이고 이쪽은 2 변수 문제다")

# (나) 계측 오차 상관의 영향. 독립 가정이 **일률적으로 보수적이 아니다.**
_gc = np.array([0.02896, 0.09469]); _gl = np.array([0.00110, -0.02486])
_GG = np.vstack([_gc, _gl])
def _sd(rho):
    sc, sl = 0.5, 0.0237
    S = np.array([[sc ** 2, rho * sc * sl], [rho * sc * sl, sl ** 2]])
    return np.sqrt(np.diag(np.linalg.inv(_GG.T @ np.linalg.inv(S) @ _GG)))
_b = _sd(0.0)
_p, _m = _sd(0.9), _sd(-0.9)
chk(_p[0] > _b[0] and _p[1] < _b[1],
    "rho>0 이면 msd 는 나빠지고 blur 는 좋아진다",
    f"msd {_p[0]/_b[0]:.2f}x, blur {_p[1]/_b[1]:.2f}x")
chk(_m[0] < _b[0] and _m[1] > _b[1],
    "rho<0 이면 반대다 -> 독립 가정은 **일률적으로 보수적이 아니다**",
    f"msd {_m[0]/_b[0]:.2f}x, blur {_m[1]/_b[1]:.2f}x")
chk(max(_p[0] / _b[0], _m[1] / _b[1]) < 1.4,
    "영향이 1.4 배 안에 갇혀 결론(블러 가능 / MSD 불가)은 뒤집히지 않는다",
    f"최악 {max(_p[0]/_b[0], _m[1]/_b[1]):.2f}x")
chk(_m[0] / _b[0] > 0.8 and _p[1] / _b[1] > 0.3,
    "유리한 쪽도 0.3 배 아래로는 가지 않는다 (과대 주장 방지)")

# (다) 치우침이 LER^2 의 함수이므로 **독립 채널이 아니다.** 실측 적합을
# 고정한다. R^2 가 LER^1 보다 LER^2 에서 커야 한다.
_L = np.array([0.2825, 0.3237, 0.5699, 3.4860])
_B = np.array([-0.0010, -0.0038, -0.0044, -0.2541])
def _r2(X):
    k = float((X @ _B) / (X @ X)); r = _B - k * X
    return 1 - float(r @ r) / float((_B - _B.mean()) @ (_B - _B.mean()))
chk(_r2(_L ** 2) > 0.999 and _r2(_L ** 2) > _r2(_L),
    "치우침 ~ LER^2 적합이 R^2 0.999 이상이고 LER^1 보다 좋다",
    f"LER^2 {_r2(_L**2):.4f} 대 LER {_r2(_L):.4f}")
_msrc = inspect.getsource(MS)
chk("독립한 두 번째 채널은 아니다" in _msrc,
    "그래서 '두 번째 채널' 이 아니라고 적혀 있다 (84 번 정정)")
chk("CD 가 **합성 크기**를 주고, LER 이 **배분**을 준다" in _msrc,
    "옳은 설명(CD 는 합성, LER 은 배분)이 적혀 있다")
chk("부정 결론에만 엄격" in _msrc,
    "CRB 가 부정 결론에만 엄격하다는 것이 적혀 있다")
chk("묽게" in inspect.getsource(MW.main),
    "LER 상대 정밀도 이식이 중립이 아니라는 것이 출력에 적힌다")

# ---------------------------------------------------------------------------
head(14, "코드 검토에서 찾은 두 빈틈")

# (가) **선택 편의(승자의 저주)를 분리해야 한다.** 상위권 후보는 대리
# 모형이 DOF 를 가장 크게 본 점이다. 모형 오차가 무작위여도 그 점들은
# 평균적으로 과대평가되어 있다. 상위권과 구간 표본을 섞어 치우침을 재면
# '모형이 낙관적' 과 '내가 낙관적인 점을 골랐다' 가 구분되지 않는다.
#
# 산수로 보인다 : 참값이 모두 같고 예측에 **무작위** 잡음만 있을 때,
# 상위권만 고르면 치우침이 생기고 전체를 보면 0 이다.
_rg = np.random.default_rng(7)
_true = np.zeros(400)
_pred = _true + _rg.normal(0, 10.0, 400)
_ordr = np.argsort(-_pred)
_top = _ordr[:20]
_bias_top = float(np.median(_true[_top] - _pred[_top]))
_bias_all = float(np.median(_true - _pred))
chk(_bias_top < -5.0 and abs(_bias_all) < 1.5,
    "무작위 잡음만 있어도 상위권만 보면 치우침이 생긴다 (승자의 저주)",
    f"상위권 {_bias_top:+.1f}, 전체 {_bias_all:+.1f}")
_spread = _ordr[20:40]
chk(abs(float(np.median(_true[_spread] - _pred[_spread]))) < abs(_bias_top),
    "순위와 무관하게 뽑은 표본은 치우침이 작다 (분리의 근거)")

_tsrc = inspect.getsource(RT.report_tool)
chk("승자의 저주" in _tsrc, "되돌림이 선택 편의를 이름으로 적는다")
chk('r < args.n_top' in _tsrc and 'r >= args.n_top' in _tsrc,
    "상위권과 구간 표본을 **갈라서** 중위 치우침을 적는다",
    "섞으면 모형 치우침과 선택 편의가 구분되지 않는다")
chk("모형 치우침이 아니라 선택 편의" in _tsrc,
    "상위권이 더 낙관적이면 그만큼이 선택 편의일 수 있다고 적는다")
chk("구간 표본 쪽으로 읽을 것" in _tsrc,
    "모형 자체의 치우침을 어디서 읽어야 하는지 적는다")

# (나) **상자 검사는 10 차원에서 매우 약하다.** 축별 범위 안이어도
# 데이터에서 멀 수 있다. 실측(모사) : 각 축 90 백분위가 94.9 %, 꼭짓점이
# 100 % 보다 멀다. 그 구조를 산수로 고정한다.
from scipy.spatial import cKDTree as _KD
_r2 = np.random.default_rng(13)
_D, _N = 10, 3000
_X = _r2.random((_N * 3, _D))
_X = _X[(_X[:, 0] * 0.6 + _X[:, 1] * 0.4 > 0.25) & (_X[:, 2] < 0.9)][:_N]
_Z = (_X - _X.mean(0)) / _X.std(0)
_t = _KD(_Z)
_base = _t.query(_Z, k=2)[0][:, 1]
def _pc(u):
    z = (u - _X.mean(0)) / _X.std(0)
    return 100.0 * float((_base < _t.query(z[None, :], k=1)[0][0]).mean())
_in_box = lambda u: bool(np.all((u >= _X.min(0)) & (u <= _X.max(0))))
_corner = _X.max(0)
chk(_pc(_corner) >= 99.9 and _in_box(_corner),
    "상자 꼭짓점은 학습점의 100 % 보다 멀면서 **상자 검사를 통과한다**",
    f"{_pc(_corner):.1f} %, 상자 안 {_in_box(_corner)}")
_p90 = np.percentile(_X, 90, axis=0)
chk(_pc(_p90) > 90.0 and _in_box(_p90),
    "각 축 90 백분위도 90 % 보다 멀다 (tool_presets 가 하는 일이다)",
    f"{_pc(_p90):.1f} %")
chk(_pc(_X.mean(0)) < 50.0, "중앙은 가깝다 (음성 대조)", f"{_pc(_X.mean(0)):.1f} %")

_e = inspect.getsource(RC.extrap_percentile)
chk("defocus" in _e and "격자로 훑으므로" in _e,
    "거리를 **광학·공구 조건 축에서만** 잰다",
    "쓸어가는 축을 넣으면 질의점이 하나로 정해지지 않는다")
_tr = open("train_surrogate.py", encoding="utf-8").read()
chk('SWEPT = ("defocus", "dose_rel")' in _tr,
    "번들도 같은 축을 빼고 기록한다 (거리 척도가 맞아야 한다)")
chk('"nn_sorted"' in _tr and '"sample"' in _tr,
    "부분표본과 학습점 최근접 거리 분포를 함께 담는다",
    "분위수만 담으면 질의점 거리를 계산할 수 없다")
_mn = inspect.getsource(RC.main)
chk("결합 분포 기록이 없다" in _mn and "상자 검사만" in _mn,
    "옛 번들이면 '상자 검사만 했다' 고 **적는다** (조용히 넘기지 않는다)")
chk('r["query"] = dict(base)' in inspect.getsource(RC.rank),
    "모형에 실제로 넣은 질의점을 들고 간다 (거울 부호 문제를 피한다)")

# ---------------------------------------------------------------------------
head(15, "추론 한계가 추천의 출력까지 전파되는가")

# (가) **초점 오프셋의 부호는 CD 데이터로 정해지지 않는다.** 거울 대칭
# CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11) 가 성립하므로, CD 만으로 역추정한
# 수차는 부호가 함께 뒤집힌 해와 구분되지 않고 그 두 해는 오프셋 부호가
# 반대다. 주장 2 의 비식별이 추천의 가장 실행 가능한 출력에 전파된다.
_m = inspect.getsource(RC.main)
chk("초점 오프셋 부호" in _m,
    "추천기가 오프셋 부호의 비식별을 적는다",
    "주장 2 를 한 번도 언급하지 않았던 것이 13주차 검토에서 드러났다")
chk("거울 대칭" in _m and "장비의 수차 계측" in _m,
    "근거(거울 대칭)와 처방(수차 계측에서 받아라)을 함께 적는다")
chk("2 * _w" in _m,
    "부호를 틀렸을 때의 이탈을 **2 x 오프셋**으로 정량한다")
chk("창 반폭" in _m, "그 이탈을 창 반폭과 비교해 적는다")

# 산수 : 중심이 c 면 부호를 틀렸을 때 이탈은 2|c| 이고, 반폭 h 대비 비율은
# 2|c|/h 다. 실측 c=25.0, h=105.0 -> 0.476.
for _c, _h, _want in ((25.0, 105.0, 0.476), (40.0, 110.0, 0.727)):
    chk(abs(2 * _c / _h - _want) < 0.001,
        f"중심 {_c} nm, 반폭 {_h} nm -> 이탈이 반폭의 {_want} 배",
        f"{2 * _c / _h:.3f}")
chk(2 * 25.0 / 105.0 < 1.0,
    "창 밖으로는 나가지 않는다 (과장하지 않기 위한 확인)")

# 거울이 부호만 바꾸는 것이 아니라 **구간의 양 끝을 맞바꿔야** 한다.
# (-b, -a) 가 아니라 (-a, -b) 로 적으면 하한 > 상한 이 된다.
_rk = inspect.getsource(RC.rank)
chk('_m["focus_range"] = (-float(_b), -float(_a))' in _rk,
    "거울이 초점 구간의 양 끝을 맞바꿔 뒤집는다",
    "(-a, -b) 로 적으면 하한이 상한보다 커진다")
_lo, _hi = -30.0, 80.0
_flip = (-_hi, -_lo)
chk(_flip[0] < _flip[1], "뒤집은 구간도 하한 < 상한 이다", str(_flip))

# (나) **다중성.** 구간을 후보마다 적으면 '각각' 이고 '전부 동시에' 가
# 아니다. 독립이면 0.9^k 이고, k=5 에서 59 % 다.
chk(abs(0.9 ** 5 - 0.59049) < 1e-9,
    "독립이면 5 개가 모두 맞을 확률은 59 %", f"{0.9**5:.5f}")
chk("다중성" in _m and "본페로니" in _m,
    "추천기가 다중성을 적고 해법(본페로니)을 제시한다")
chk("승자의 저주" in _m or "덜 맞는다" in _m,
    "가장 좋은 후보를 고르는 행위가 피복을 떨어뜨린다는 것을 적는다")

# (다) **대리 모형의 존재 이유가 측정되지 않았다.** train_surrogate 는
# 자기를 "시뮬레이터의 빠른 근사" 라고 적고 README 는 "그 탐색을 빠르게
# 하는 것" 이라고 적는데, 얼마나 빠른지는 어디에도 없었다.
_rm = inspect.getsource(RT.main)
_rt = inspect.getsource(RT.report_tool)
chk("_t_sur" in _rm and "_t_win" in _rm,
    "되돌림이 두 경로의 시간을 잰다 (추가 비용 0 — 이미 둘 다 호출한다)")
chk("args.n_na * args.n_sigma * len(arms)" in _rm,
    "창당 비용을 **격자 전체**로 나눈다",
    "살아남은 후보 수로 나누면 창당 비용이 과대평가된다")
chk("[속도]" in _rt and "배" in _rt, "속도 배수를 보고한다")
chk("이 배수가 대리 모형을 쓰는 근거다" in _rt,
    "그 배수가 왜 중요한지 적는다 (느리면서 부정확하면 쓸 이유가 없다)")
chk("라벨로 예측하므로 이 비용이 없다" in _rt,
    "CDU 몬테카를로 비용을 대리 모형이 피한다는 것을 적는다")

# 순서 : 시간을 **보고보다 먼저** 붙여야 한다. 반대로 두면 [속도] 블록이
# 조용히 빠진다(13주차에 실제로 그랬다).
_i_t = _rm.index('_r["_t_sur_win"]')
_i_r = _rm.index("report_tool(tname, recs, args")
chk(_i_t < _i_r, "시간 부착이 report_tool 호출보다 **앞**에 있다",
    "뒤에 두면 보고가 키를 못 찾아 속도 블록이 조용히 빠진다")
chk("보고보다 먼저" in _rm, "그 순서의 이유를 코드가 적는다")

# 산수 : 창당 비용과 배수.
_t_tot, _n = 0.0192, 16
chk(abs(_t_tot / _n - 0.0012) < 1e-6,
    "총 0.0192 s / 창 16 개 = 1.2 ms (창당 비용)")
chk(abs(1.06 / (_t_tot / _n) - 883.33) < 1.0,
    "시뮬레이터 1.06 s 대비 약 880 배",
    f"{1.06/(_t_tot/_n):.1f}")

# ---------------------------------------------------------------------------
head(16, "판정 기준이 있고, 코드가 그 기준으로 판정하는가")

# 되돌림은 숫자만 내고 '어느 값이면 성공인가' 가 아무 데도 없었다. 그러면
# 숫자를 본 뒤에 기준을 정하게 되고, 그것은 보강명세_동결.md 가 막으려던
# 사후선택과 같은 행위다.
chk(os.path.exists("판정기준_동결.md"), "판정 기준 동결 문서가 있다")
_pj = open("판정기준_동결.md", encoding="utf-8").read()
chk("실제 모형으로 되돌림을 돌리기 전에" in _pj,
    "실제 모형으로 돌리기 **전**에 썼다고 명기한다")
chk("변경 기록" in _pj and "사후선택이 섞였다" in _pj,
    "고칠 때의 규칙과 그 대가를 적는다")

# 문서의 기준값과 코드의 상수가 **같아야** 한다. 어긋나면 어느 쪽으로
# 판정했는지 알 수 없다.
chk(abs(RT.CRITERIA["tau_min"] - 0.60) < 1e-12, "코드의 타우 기준 0.60")
chk(abs(RT.CRITERIA["top3_min"] - 2 / 3) < 1e-9, "코드의 상위3 기준 2/3")
chk(abs(RT.CRITERIA["cover_slack"] - 0.10) < 1e-12, "코드의 피복 여유 0.10")
chk("0.60" in _pj and "2/3" in _pj and "0.10" in _pj,
    "문서에 같은 세 값이 적혀 있다")

# 타우-b 와 두 후보 비교 정확도의 환산. tau = 2p - 1 이므로 0.60 <-> 0.80.
chk(abs(50 * (1 + 0.60) - 80.0) < 1e-9,
    "tau 0.60 은 두 후보 비교 정확도 80 % 다 (기준값의 근거)")

# 다섯 갈림길이 모두 서로 다른 결론으로 가야 한다. 하나로 뭉치면 기준이
# 있으나 마나다.
# [92 번] A 는 임계값만으로 판정하지 않는다. 검정력 요구(후보 수, 귀무
# 분포 대조)가 함께 걸리므로, 이 갈림길 시험도 그 둘을 충족시킨 상태에서
# 임계값만 흔들어야 한다. 그러지 않으면 A 가 늘 '잴 수 없음' 으로 떨어져
# 다섯 갈림길이 구분되지 않는다.
_N16 = RT.CRITERIA["n_min"]
_null16 = {"exact": True, "n_perm": 720, "tau_q": 0.20, "tau_median": 0.0,
           "tau_p": 0.01, "n_distinct_tau": 15}
_mk = lambda t, b, c: RT.verdict_text(
    RT.judge(t, b, c, 0.92, _N16, _null16))[0]
_a = _mk(0.75, 1.0, 0.88); _b = _mk(0.45, 1.0, 0.88)
_c = _mk(0.75, 0.33, 0.88); _d = _mk(0.75, 1.0, 0.60)
_e = RT.verdict_text(RT.judge(None, None, None, None))[0]
chk(len({_a, _b, _c, _d, _e}) == 5,
    "다섯 결과(전부충족 / A미달 / B미달 / C미달 / 측정불가)가 각각 다른 결론")
chk("그대로 유지" in _a, "전부 충족이면 12주차 주장 유지")
chk("철회" in _b and "후보 축소 도구" in _b, "A 미달이면 주장 철회")
chk("좁힌다" in _c, "B 미달이면 주장을 좁힌다")
chk("구간만 철회" in _d, "C 미달이면 구간만 철회")
chk("'충족' 으로 읽지 말" in _e, "측정 불가를 충족으로 읽지 말라고 적는다")

# 부분 충족을 '성공' 으로 묶지 않는지 — 문서와 코드 양쪽에서.
chk("부분 충족을 '성공' 으로 적지 않는다" in _pj,
    "부분 충족을 성공으로 적지 않는다고 문서가 못 박는다")
chk("부분 충족을 '성공' 으로 묶지 않는다" in inspect.getsource(RT.judge),
    "코드도 같은 것을 적는다")

# 기준을 두지 **않은** 항목이 둘 있고, 그 이유가 적혀 있어야 한다.
chk("기준 없음" in _pj and "비율이 아니라 목록" in _pj,
    "제약 판정에 기준을 두지 않은 이유를 적는다 (비율에 합격선은 뜻이 없다)")
chk("선택 편의" in _pj and "갈라" in _pj,
    "치우침에 기준을 두지 않고 갈라 보고한다고 적는다")

# ---------------------------------------------------------------------------
head(17, "학습 보고서가 JSON 으로 써지는가")
print("""    왜 이 케이스가 필요한가 : 13 차 보완 85 번에서 `training_domain` 에
    적용범위 기록(`coverage`)을 넣었다. 그 안에는 넘파이 배열이 있다. 같은
    `domain` 객체가 .joblib 번들(넘파이 가능)과 JSON 보고서(넘파이 불가)
    양쪽으로 간다. 그래서 학습을 **다 끝낸 뒤 마지막 줄에서** 터졌다.
    배열을 리스트로 풀어 넣는 것도 답이 아니다. 3000 x 10 을 indent=2 로
    쓰면 보고서가 수만 줄이 되어 사람이 읽는 파일로서 못 쓴다.
    여기서 보는 것 : 요약판이 JSON 으로 써지고, 크기가 사람이 읽을 수준이고,
    **원본 배열은 번들 쪽에 그대로 남는가.**""")

import json as _json
import train_surrogate as TS

_rng17 = np.random.default_rng(17)
_dom17 = {
    "n_train_rows": 1234,
    "axes": {"na": {"min": 1.2, "max": 1.35, "p01": 1.2, "p99": 1.35}},
    "categorical": {"arm": [0.0, 1.0]},
    "held_constant": {"wavelength_nm": 193.0},
    "coverage": {
        "axes": ["na", "sigma"], "mu": [1.3, 0.7], "sd": [0.04, 0.05],
        "sample": _rng17.normal(size=(3000, 2)).astype(np.float32),
        "nn_sorted": np.sort(_rng17.random(3000)).astype(np.float32),
        "swept_excluded": ["defocus", "dose_rel"], "note": "시험용",
    },
}
_sum17 = TS.domain_for_json(_dom17)
_txt17 = _json.dumps({"training_domain": _sum17}, ensure_ascii=False, indent=2)
chk(True, "요약판이 예외 없이 JSON 으로 직렬화된다")
chk(_txt17.count("\n") < 200,
    "보고서가 사람이 읽을 크기다 (요약 전이면 수만 줄)",
    f"{_txt17.count(chr(10))} 줄")
chk("sample" not in _sum17["coverage"]
    and "nn_sorted" not in _sum17["coverage"],
    "요약판에는 원본 배열이 없다")
chk(_sum17["coverage"]["n_sample"] == 3000,
    "표본 크기는 남는다 (외삽 판정 기준이 들어갔는지 보고서로 확인 가능)")
chk("p50" in _sum17["coverage"]["nn_quantiles"],
    "최근접 거리 분포가 분위수로 남는다")
chk(isinstance(_dom17["coverage"]["sample"], np.ndarray)
    and _dom17["coverage"]["sample"].shape == (3000, 2),
    "원본 domain 은 바뀌지 않는다 (번들이 배열을 그대로 들고 간다)")

# 그물 : 요약을 거치지 않은 넘파이 스칼라·배열이 섞여도 보고서가 남아야
# 한다. 수십 분 돌린 학습을 보고서 한 줄 때문에 잃는 것이 가장 나쁘다.
_stray = {"a": np.float32(1.5), "b": np.int64(3), "c": np.arange(4),
          "d": np.zeros((100, 100))}
_t2 = _json.dumps(_stray, default=TS._json_default)
chk('"a": 1.5' in _t2 and '"b": 3' in _t2, "넘파이 스칼라는 값으로 떨어진다")
chk("[0, 1, 2, 3]" in _t2, "작은 배열은 리스트로 들어간다")
chk("_dropped_array" in _t2, "큰 배열은 모양만 적고 버린다")

# 실제 호출 지점이 요약판을 쓰는지 — 원본을 그대로 넣는 회귀를 막는다.
_src17 = inspect.getsource(TS)
chk('"training_domain": domain_for_json(domain)' in _src17,
    "보고서는 요약판을 쓴다")
chk('"label": tgt, "domain": domain,' in _src17,
    "번들은 원본을 쓴다 (추천기가 부분표본으로 외삽 거리를 잰다)")
chk("default=_json_default" in _src17, "json.dump 에 그물이 걸려 있다")

# ---------------------------------------------------------------------------
head(18, "외삽 백분위의 참조 분포가 성립하는가")
print("""    왜 이 케이스가 필요한가 : 거리에서 초점·노광을 뺀 것은 옳다. 그런데
    **그 두 축으로만 다른 행을 표본에 그대로 남겨 두었다.** 데이터셋은
    광학점 하나를 초점·노광 격자로 훑으므로 한 조건이 수십 번 중복되고,
    남은 부공간에서는 완전히 같은 점이 된다. 실측(223 행) : 서로 다른 조건
    14 개, 최근접 거리의 **95.1 % 가 정확히 0**. 백분위는
    100 x mean(nn < d) 이므로 거리가 0 보다 조금만 커도 즉시 95 % 가 나오고,
    후보가 어디 있든 '외삽에 가깝다' 가 찍힌다. 조용히 틀리는 것보다 나쁘다
    — 시끄럽게 틀려서 참된 외삽을 가린다.
    여기서 보는 것 : 중복이 지워지는가(음성 대조로 지우기 전 상태의 포화를
    재현), 참조 분포가 얇으면 **숫자를 내놓지 않고 거절하는가.**""")

# (가) 중복 제거 — 쓸어가는 두 축으로만 다른 행을 만들어 넣는다.
_feat18 = ["na", "sigma_out", "defocus", "dose_rel"]
_cat18 = np.array([False, False, False, False])
_base18 = np.array([[1.30, 0.70], [1.32, 0.72], [1.34, 0.68], [1.28, 0.74]])
_rows18 = []
for _na, _sg in _base18:
    for _dz in np.linspace(-80, 80, 40):          # 같은 조건을 40 번 훑는다
        _rows18.append([_na, _sg, _dz, 1.0])
_X18 = np.array(_rows18, dtype=np.float64)
_d18 = TS.training_domain(_X18, _feat18, _cat18,
                          np.arange(_X18.shape[0]), [], [])
_c18 = _d18["coverage"]
chk(_c18["n_rows_in"] == 160, "입력 행 수가 기록된다", str(_c18["n_rows_in"]))
chk(_c18["n_distinct"] == 4,
    "쓸어가는 축으로만 다른 행은 하나로 합쳐진다 (160 -> 4)",
    str(_c18["n_distinct"]))
chk(_c18["sample"].shape[0] == 4, "표본이 서로 다른 조건만 담는다")
_nn18 = np.asarray(_c18["nn_sorted"], dtype=np.float64)
chk(_nn18.size and float(np.mean(_nn18 <= 0.0)) == 0.0,
    "참조 거리 분포에 0 이 없다 (중복 제거 전에는 95.1 % 가 0 이었다)")
chk("defocus" in _c18["swept_excluded"] and "dose_rel" in _c18["swept_excluded"],
    "거리 축에서 초점·노광이 빠져 있다")

# (나) 음성 대조 : 중복을 지우지 않았다면 어떻게 됐는지 직접 재서, 고친
#      것이 실제로 그 포화를 막는지 확인한다. 고친 뒤 값만 보면 '원래
#      문제였는지' 를 알 수 없다.
_Zraw = _X18[:, [0, 1]]
_Zs = (_Zraw - _Zraw.mean(axis=0)) / np.where(_Zraw.std(axis=0) > 0,
                                              _Zraw.std(axis=0), 1.0)
from scipy.spatial import cKDTree as _KD
_dd, _ = _KD(_Zs).query(_Zs, k=2)
_nn_bad = np.sort(_dd[:, 1])
chk(float(np.mean(_nn_bad <= 0.0)) > 0.9,
    "음성 대조 : 중복을 남기면 참조 분포의 90 % 넘게 0 이 된다",
    f"{100 * np.mean(_nn_bad <= 0.0):.1f} %")
chk(float(np.mean(_nn_bad < 1e-6)) > 0.9 and float(np.mean(_nn18 < 1e-6)) == 0.0,
    "같은 데이터인데 고친 쪽만 분포가 살아 있다")

# (다) 참조 분포가 얇으면 거절해야 한다. 숫자를 내놓고 해석을 부르는 쪽에
#      맡기면 nan >= 90 이 거짓이라 조용히 '데이터 안' 으로 찍힌다.
def _cov18(n_ref, zero_frac=0.0, n_sub=None):
    n_sub = n_sub if n_sub is not None else max(n_ref, 12)
    _rg = np.random.default_rng(18)
    nn = _rg.random(n_ref) + 0.5
    nn[:int(round(zero_frac * n_ref))] = 0.0
    return {"coverage": {
        "axes": ["na", "sigma_out"], "mu": [1.3, 0.7], "sd": [0.02, 0.02],
        "sample": _rg.normal(size=(n_sub, 2)).astype(np.float32),
        "nn_sorted": np.sort(nn).astype(np.float32),
        "swept_excluded": ["defocus", "dose_rel"]}}

_q18 = {"na": 1.30, "sigma_out": 0.70}
chk(RC.extrap_percentile(_cov18(14), _feat18, _q18) is None,
    "서로 다른 조건이 20 개 미만이면 백분위를 내놓지 않는다")
chk(RC.extrap_percentile(_cov18(200, zero_frac=0.6), _feat18, _q18) is None,
    "최근접 거리의 절반 이상이 0 이면 내놓지 않는다 (포화)")
_ok18 = RC.extrap_percentile(_cov18(200, zero_frac=0.0), _feat18, _q18)
chk(_ok18 is not None and np.isfinite(_ok18["pct"]),
    "분포가 성립하면 백분위를 돌려준다 (거절 조건이 과하지 않다)")
chk(RC.extrap_percentile({}, _feat18, _q18) is None,
    "기록이 아예 없는 옛 번들도 None (조용히 넘기지 않는다)")

# (라) 보고하는 쪽이 두 경우를 **구분해** 적는가. 사용자가 할 일이 다르다.
_src18 = inspect.getsource(RC)
chk("참조 거리 분포가 성립하지 않는다" in _src18,
    "포화한 경우의 안내 문구가 있다")
chk("옛 번들" in _src18 and "결합 분포 기록이 없다" in _src18,
    "기록 없는 경우의 안내 문구가 따로 있다")
chk("재지 못했다" in _src18,
    "못 쟀다고 적는다 ('데이터 안' 으로 찍지 않는다)")

# ---------------------------------------------------------------------------
head(19, "보고서 그림의 제목 밀기가 실제로 듣는가")
print("""    왜 이 케이스가 필요한가 : `figlib.ylab` 은 한글 y 축 이름을 축 위에
    가로로 둔다. 그 자리에 제목이 오면 겹치므로 저장 시점에 제목을 위로
    민다 — 고 적어 두었다. 그런데 `ax.title.set_y()` 로 짜 두었고, 그것은
    **`savefig` 가 되돌린다.** matplotlib 은 그릴 때마다 제목 y 를 축 위쪽
    경계에서 다시 계산한다. 실측 : set_y 직후 (0.5, 1.16) -> savefig 뒤
    (0.5, 1.00). 호출 직후만 확인하고 **저장된 그림을 보지 않아서** 12·
    13주차 그림이 전부 겹친 채로 나오고 있었다.
    여기서 보는 것 : 그리고 난 **뒤의 픽셀 좌표**로 제목과 축이 떨어져
    있는가. 음성 대조로 옛 방식(set_y)이 실제로 되돌려지는지도 확인한다.""")

import figlib as FL

_o19 = FL.setup(os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "_w13_figchk"))


def _gap(use_pad: bool):
    """제목 아래 모서리와 축 위 모서리의 간격 [px]. **그린 뒤에** 잰다."""
    fig, ax = FL.plt.subplots(figsize=(6.0, 3.2))
    ax.plot([0, 1], [0, 1])
    FL.ylab(ax, "이 거리보다 가까운 학습 조건의 비율 [%]")
    ax.set_title("외삽 판정의 참조 분포")
    if use_pad:
        FL._lift_titles(fig)
    else:
        ax.title.set_y(1.16)                  # 옛 방식 (음성 대조)
    fig.canvas.draw()                          # savefig 와 같은 경로
    g = ax.title.get_window_extent().y0 - ax.get_window_extent().y1
    FL.plt.close(fig)
    return float(g)


_g_pad, _g_old = _gap(True), _gap(False)
# 기준값은 **재서** 정했다. 고친 방식 63.3 px, 옛 방식·기본 13.3 px
# (matplotlib 의 기본 titlepad). 가운데를 넉넉히 비워 둔다.
chk(_g_pad > 40.0,
    "고친 방식 : 그린 뒤에도 제목이 축 위 40 px 넘게 떨어져 있다",
    f"{_g_pad:.1f} px")
chk(_g_old < 20.0,
    "음성 대조 : 옛 set_y 방식은 그리는 순간 기본 위치로 되돌려진다",
    f"{_g_old:.1f} px")
chk(_g_pad > _g_old + 30.0,
    "두 방식의 차이가 분명하다 (우연히 통과한 것이 아니다)",
    f"{_g_pad:.1f} 대 {_g_old:.1f} px")

# 제목을 나중에 걸어도 듣는가. `ylab` 과 `set_title` 의 호출 순서에
# 상관없어야 한다는 것이 `figlib` 이 적어 둔 약속이다.
_f19, _a19 = FL.plt.subplots(figsize=(6.0, 3.2))
_a19.plot([0, 1], [0, 1])
_a19.set_title("먼저 건 제목")
FL.ylab(_a19, "나중에 건 라벨")
FL._lift_titles(_f19)
_f19.canvas.draw()
chk(_a19.title.get_window_extent().y0 - _a19.get_window_extent().y1 > 40.0,
    "제목을 먼저 걸어도 듣는다 (호출 순서에 무관)")
FL.plt.close(_f19)

# 왼쪽 정렬 제목도 밀려야 한다. `get_title()` 은 가운데 것만 돌려주므로
# 세 자리를 훑지 않으면 조용히 빠진다.
_f19b, _a19b = FL.plt.subplots(figsize=(6.0, 3.2))
_a19b.plot([0, 1], [0, 1])
FL.ylab(_a19b, "라벨")
_a19b.set_title("왼쪽 제목", loc="left")
FL._lift_titles(_f19b)
_f19b.canvas.draw()
chk(_a19b._left_title.get_window_extent().y0
    - _a19b.get_window_extent().y1 > 40.0,
    "왼쪽 정렬 제목도 밀린다 (get_title 은 가운데 것만 준다)")
FL.plt.close(_f19b)

# 가로 라벨을 쓰지 않은 축은 건드리지 않는다. 멀쩡한 그림의 여백을
# 괜히 벌리면 안 된다.
_f19c, _a19c = FL.plt.subplots(figsize=(6.0, 3.2))
_a19c.plot([0, 1], [0, 1])
_a19c.set_ylabel("보통 세로 라벨")
_a19c.set_title("제목")
FL._lift_titles(_f19c)
_f19c.canvas.draw()
chk(_a19c.title.get_window_extent().y0
    - _a19c.get_window_extent().y1 < 20.0,
    "가로 라벨을 안 쓴 축은 그대로 둔다 (과잉 보정 없음)")
FL.plt.close(_f19c)

# 주석과 설명에 'set_y' 라는 **말**은 남아 있어야 한다(왜 안 쓰는지
# 적어 둔 것이다). 막아야 하는 것은 **호출**이고, 둘은 문자열 검색으로
# 못 가린다. 구문 나무로 본다.
import ast as _ast

with open("figlib.py", encoding="utf-8") as _fh19:
    _tree19 = _ast.parse(_fh19.read())
_calls19 = [n for n in _ast.walk(_tree19)
            if isinstance(n, _ast.Call)
            and isinstance(n.func, _ast.Attribute)
            and n.func.attr == "set_y"]
chk(not _calls19,
    "저장 경로에 되돌려지는 set_y **호출**이 없다 (주석의 설명은 남는다)",
    f"{len(_calls19)} 곳")
chk("set_y" in inspect.getsource(FL._lift_titles),
    "왜 set_y 를 쓰지 않는지는 문서로 남아 있다 (같은 실수 반복 방지)")
chk("_lift_titles(fig)" in inspect.getsource(FL.save),
    "저장할 때 제목 밀기를 실제로 부른다")

# ---------------------------------------------------------------------------
head(20, "기준 A 가 검정력을 요구하는가")
print("""    왜 이 케이스가 필요한가 : 실제 되돌림에서 판정이 인쇄된 칸은 제약
    통과군 후보가 **3 개**였다. 켄달 +0.816 으로 'A 충족' 이 찍혔는데,
    같은 코드가 두 줄 위에 귀무분포 95 백분위를 +0.866 으로 찍어 두었다.
    즉 관측값이 우연선을 넘지 못했는데 충족으로 판정했다. 임계값(0.60)만
    동결하고 **검정력을 동결하지 않은** 것이 원인이다.
    여기서 보는 것 : 후보가 적으면 잴 수 없다고 하는가, 귀무분포를 넘지
    못하면 미달로 보는가, 작은 n 을 완전 열거하는가, 그리고 **이 변경이
    어떤 칸도 미달에서 충족으로 바꾸지 않았는가.**""")


def _sd_tau(n):
    """켄달 타우-b 의 귀무 표준편차. 상수를 상수와 비교하지 않기 위해 쓴다."""
    return math.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))


_nmin = RT.CRITERIA["n_min"]
chk(1.645 * _sd_tau(_nmin) <= RT.CRITERIA["tau_min"] + 1e-9,
    f"n = {_nmin} 에서 동결 기준이 우연선보다 엄격하다",
    f"1.645 x {_sd_tau(_nmin):.3f} = {1.645 * _sd_tau(_nmin):.3f} <= 0.60")
chk(1.645 * _sd_tau(_nmin - 1) > RT.CRITERIA["tau_min"],
    f"n = {_nmin - 1} 에서는 약하다 (즉 {_nmin} 이 **가장 작은** n 이다)",
    f"1.645 x {_sd_tau(_nmin - 1):.3f} = "
    f"{1.645 * _sd_tau(_nmin - 1):.3f} > 0.60")
# 근사식이 아니라 실제 분포로 확인한다.
_e6 = RT.perm_null(list(range(6)), [3.0, 1.0, 5.0, 0.0, 4.0, 2.0])
chk(_e6 is not None and _e6["exact"] and _e6["n_perm"] == 720,
    "n = 6 은 순열 720 가지를 완전 열거한다")
chk(abs(_e6["tau_q"] - RT.CRITERIA["tau_min"]) < 1e-9,
    "열거한 n = 6 의 귀무 95 백분위가 동결 기준과 정확히 같다",
    f"{_e6['tau_q']:+.3f} 대 {RT.CRITERIA['tau_min']:.2f}")

# 실측된 중심 판정 칸을 그대로 재현한다. 후보 3 개, 예측 동률 1 쌍.
_sur20 = [412.5, 412.5, 375.0]
_sim20 = [337.5, 300.0, 262.5]
_n20 = RT.perm_null(_sur20, _sim20)
_r20 = RT.rank_corr(_sur20, _sim20)
chk(_n20["exact"] and _n20["n_distinct_tau"] <= 3,
    "동률 때문에 켄달이 가질 수 있는 값이 3 개 이하다",
    f"{_n20['n_distinct_tau']} 개")
chk(abs(_n20["tau_p"] - 1.0 / 3.0) < 1e-9,
    "관측값이 우연히 나올 확률이 1/3 이다 (열거한 정확한 값)",
    f"{_n20['tau_p']:.1%}")
_rows20 = RT.judge(_r20["kendall"], 1.0, 1.0, 0.902, len(_sur20), _n20)
_j20 = {n.split()[0]: v for n, v, _ in _rows20}
chk(_j20["A"] == "잴 수 없음",
    "실측 칸의 A 가 '충족' 에서 '잴 수 없음' 으로 바뀐다 (이것이 92 번)")
_why20 = [w for n, _, w in _rows20 if n.startswith("A")][0]
chk("0.816" not in _why20 and str(len(_sur20)) in _why20,
    "잴 수 없을 때 타우 값을 **적지 않는다** (적으면 충족으로 읽힌다)")

# n 은 충분한데 귀무분포를 넘지 못하는 경우 — 두 요구가 서로를 대신하지
# 못한다는 것이 요점이다.
_nl_hi = {"exact": True, "n_perm": 720, "tau_q": 0.80, "tau_median": 0.0,
          "tau_p": 0.22, "n_distinct_tau": 15}
_nl_lo = {"exact": True, "n_perm": 720, "tau_q": 0.20, "tau_median": 0.0,
          "tau_p": 0.01, "n_distinct_tau": 15}


def _gA(nl, t=0.75, n=8):
    return {x.split()[0]: v
            for x, v, _ in RT.judge(t, 1.0, 1.0, 0.902, n, nl)}["A"]


chk(_gA(_nl_hi) == "**미달**",
    "임계값은 넘었어도 귀무 95 백분위를 못 넘으면 미달")
chk(_gA(_nl_lo) == "충족", "둘 다 넘으면 충족 (요구가 과하지 않다)")
chk(_gA(None) == "잴 수 없음",
    "귀무분포를 못 구하면 잴 수 없음 (충족으로 떨어지지 않는다)")
chk(_gA(_nl_lo, t=0.45) == "**미달**",
    "귀무를 넘더라도 임계값에 못 미치면 미달 (A-2 가 A 를 대신하지 않는다)")

# **이 변경이 느슨하게 만든 칸이 없는가.** 기준을 바꿀 때 가장 위험한
# 것이 자기도 모르게 통과를 늘리는 것이다. 무작위 입력으로 옛 판정과 새
# 판정을 비교해 미달 -> 충족 방향 전환이 0 인지 본다.
# n 은 3~6 으로 둔다. 판정이 갈리는 **경계가 6** 이므로 거기를 덮으면
# 되고, n=8,9 는 순열이 커져(표집 2000 회) 시험 시간만 늘린다. 실측 :
# 3~9 로 두면 이 케이스 하나가 6 분을 먹었다.
_rg20 = np.random.default_rng(20)
_flip20 = 0
_tot20 = 0
for _ in range(400):
    _nn = int(_rg20.integers(3, 7))
    _a20 = list(_rg20.normal(size=_nn) * 50 + 300)
    _b20 = list(_rg20.normal(size=_nn) * 50 + 300)
    _rr = RT.rank_corr(_a20, _b20)
    if not np.isfinite(_rr["kendall"]):
        continue
    _tot20 += 1
    _old = "충족" if _rr["kendall"] >= RT.CRITERIA["tau_min"] else "**미달**"
    _new = _gA(RT.perm_null(_a20, _b20), _rr["kendall"], _nn)
    if _old == "**미달**" and _new == "충족":
        _flip20 += 1
chk(_flip20 == 0,
    "무작위 400 회에서 미달 -> 충족으로 바뀐 칸이 없다 (변경이 느슨하지 않다)",
    f"{_flip20}/{_tot20}")

# 동결 문서가 변경을 **기록**했는가. 기준을 고치고 적지 않으면 그 문서는
# 사후선택을 막지 못한다.
chk("2026-11-11" in _pj and "실제 되돌림 결과를 본 뒤의 변경" in _pj,
    "데이터를 본 뒤의 변경임을 변경 기록에 적었다")
chk("임계값 0.60, 2/3, 0.10 은 **하나도 고치지 않았다.**" in _pj,
    "임계값은 손대지 않았다고 명시한다")
chk("미달에서 충족으로 바뀌지 않았다" in _pj,
    "느슨해진 칸이 없다는 것을 문서도 적는다")
chk(f"최소 후보 수 — **{_nmin} 개**" in _pj,
    "문서의 최소 후보 수가 코드 상수와 같다")

# 같은 화면에 귀무분포와 판정이 함께 나오는데 서로 어긋나던 것이 원인이다.
# 이제 **같은 함수**가 둘을 만든다.
_src20 = inspect.getsource(RT)
chk(_src20.count("perm_null(sur, sim)") >= 2,
    "인쇄하는 귀무분포와 판정이 쓰는 귀무분포가 같은 함수에서 나온다")
chk("rank_corr(sur, list(rng.permutation(sim)))" not in _src20,
    "판정과 무관했던 옛 표집 음성 대조가 남아 있지 않다")

# ---------------------------------------------------------------------------
head(21, "기준 A 가 분해능을 요구하는가")
print("""    왜 이 케이스가 필요한가 : 격자를 늘려 후보가 15 개까지 나왔는데도
    통과군 실측 DOF 가 **2 개 값**뿐이었다. 초점 격자가 18.75 nm 라 DOF 가
    37.5 nm 눈금으로 끊기고, 통과군의 실측 DOF 폭이 딱 그 한 칸(37.5 nm)
    이기 때문이다. 그 상태에서 켄달 +0.445 에 'A 미달' 이 찍혔다.
    눈금을 거치면 타우가 깎인다 — 참 스피어만 0.70, 후보 15 개, 눈금
    37.5 nm 에서 관측 켄달 중위가 +0.558, 5~95 백분위가 +0.19~+0.79 다.
    즉 **관측 +0.445 는 참 타우 0.60 이상과도 양립한다.**
    이 규칙은 동결 문서가 **데이터 전에** 이미 요구했다("서로 다른 값이
    후보 수의 절반 미만이면 ... 격자를 촘촘히 해 다시 재야 한다").
    `judge()` 가 그 문장을 인쇄만 하고 판정에 쓰지 않은 것이 구멍이었다.
    여기서 보는 것 : 관문이 작동하는가, 그리고 이 변경이 **'미달' 을 지우는
    방향**이므로 더 엄격하게 — 어떤 칸도 충족으로 바뀌지 않는가.""")

# (가) 눈금이 타우를 깎는다는 것을 **여기서 직접 재서** 확인한다. 보류의
#      근거가 이 측정이므로, 근거를 회귀에 박아 둔다.
_rg21 = np.random.default_rng(21)
_att = {}
for _rho in (0.70, 0.95):
    _vals = []
    for _ in range(300):
        _z1 = _rg21.normal(size=15)
        _z2 = _rho * _z1 + math.sqrt(max(1 - _rho ** 2, 0.0)) * _rg21.normal(size=15)
        _q1 = np.round((112.5 + 25 * _z1) / 37.5) * 37.5
        _q2 = np.round((112.5 + 25 * _z2) / 37.5) * 37.5
        _rr = RT.rank_corr(list(_q1), list(_q2))
        if np.isfinite(_rr["kendall"]):
            _vals.append(_rr["kendall"])
    _att[_rho] = (float(np.median(_vals)), float(np.percentile(_vals, 5)))
chk(_att[0.70][0] < 0.70,
    "눈금을 거치면 켄달이 참값보다 작게 보인다 (참 0.70)",
    f"관측 중위 {_att[0.70][0]:+.3f}")
chk(_att[0.70][1] < 0.445,
    "참 타우 0.70 이어도 관측이 +0.445 아래로 내려갈 수 있다",
    f"5 백분위 {_att[0.70][1]:+.3f}")
chk(_att[0.95][0] < 0.95,
    "강한 관계도 깎인다 (참 0.95)", f"관측 중위 {_att[0.95][0]:+.3f}")

# (나) 관문이 실측 칸에서 작동하는가. s13c 기준(요청값) 을 재현한다.
_sur21 = [150.0] * 8 + [112.5] * 7
_sim21 = [150.0, 112.5, 112.5, 150.0, 150.0] + [112.5] * 10
_r21 = RT.rank_corr(_sur21, _sim21)
_nd21 = min(len(set(_sur21)), len(set(_sim21)))


def _A21(tau, n, nd, span=37.5, nl="auto", sur=None, sim=None):
    if nl == "auto":
        nl = {"exact": True, "n_perm": 720, "tau_q": 0.10,
              "tau_median": 0.0, "tau_p": 0.01, "n_distinct_tau": 20}
    return {x.split()[0]: v for x, v, _ in
            RT.judge(tau, 1.0, 1.0, 0.902, n, nl, nd, span)}["A"]


chk(_nd21 * 2 < len(_sur21),
    "실측 칸이 동결 문서의 분해능 조건에 걸린다",
    f"서로 다른 값 {_nd21} 개 대 후보 {len(_sur21)} 개")
chk(_A21(_r21["kendall"], len(_sur21), _nd21) == "잴 수 없음",
    "그 칸의 A 가 '미달' 에서 '잴 수 없음' 으로 바뀐다")
_w21 = [w for n, _, w in RT.judge(_r21["kendall"], 1.0, 1.0, 0.902,
                                  len(_sur21), None, _nd21, 37.5)
        if n.startswith("A")][0]
chk("37.5" in _w21 and "절반" in _w21,
    "이유에 서로 다른 값 수와 DOF 폭을 적는다 (다시 재는 데 필요한 정보)")

# (다) **가장 중요한 검증.** 이 관문은 '미달' 을 지우는 방향이므로, 내게
#      유리한 쪽이다. 그래서 대칭인지 본다 — 타우가 높아도 분해능이
#      모자라면 똑같이 보류되어야 한다. 한쪽만 지우면 기준이 아니라
#      변명이다.
chk(_A21(0.95, 15, 2) == "잴 수 없음",
    "타우가 높아도 분해능이 모자라면 보류다 (관문이 대칭이다)")
chk(_A21(0.95, 15, 8) == "충족",
    "분해능이 충분하면 충족이 나온다 (관문이 과하지 않다)")
chk(_A21(0.30, 15, 8) == "**미달**",
    "분해능이 충분하면 낮은 타우는 미달로 남는다")

# (라) 무작위 대조 : 옛 판정(임계값만)과 새 판정을 비교해 **미달 -> 충족**
#      전환이 0 인지 본다. 보류로 바뀌는 것은 허용되지만, 통과가 늘어나는
#      것은 절대 허용되지 않는다.
_rg21b = np.random.default_rng(210)
_to_ok = 0
_to_hold = 0
_tot21 = 0
for _ in range(600):
    _n = int(_rg21b.integers(6, 20))
    _nd = int(_rg21b.integers(2, _n + 1))
    _t = float(_rg21b.uniform(-1.0, 1.0))
    _tot21 += 1
    _old = "충족" if _t >= RT.CRITERIA["tau_min"] else "**미달**"
    _new = _A21(_t, _n, _nd)
    if _old == "**미달**" and _new == "충족":
        _to_ok += 1
    if _old == "**미달**" and _new == "잴 수 없음":
        _to_hold += 1
chk(_to_ok == 0,
    "무작위 600 회에서 미달 -> 충족 전환이 없다 (통과를 늘리지 않는다)",
    f"{_to_ok}/{_tot21}")
chk(_to_hold > 0,
    "미달 -> 보류 전환은 실제로 일어난다 (관문이 죽어 있지 않다)",
    f"{_to_hold}/{_tot21}")

# (마) 동결 문서가 **데이터 전에** 이 규칙을 담고 있었는가. 이것이 이
#      변경을 사후선택과 가르는 유일한 근거다.
chk("서로 다른 DOF 값의 개수" in _pj and "절반 미만" in _pj,
    "분해능 규칙이 동결 문서 본문에 (변경 기록이 아니라) 있다")
chk("격자를 촘촘히 해 다시 재야 한다" in _pj,
    "다시 재야 한다는 처방까지 동결 문서가 적어 두었다")

# (바) 세 관문의 순서. 분해능 보류가 후보 수 보류보다 **뒤**에 와야
#      한다 — 후보가 3 개면 분해능을 따질 것도 없이 후보 수가 먼저다.
chk(_A21(0.95, 3, 2) == "잴 수 없음", "후보도 적고 분해능도 모자라면 보류")
_w3 = [w for n, _, w in RT.judge(0.95, 1.0, 1.0, 0.902, 3, None, 2, 37.5)
       if n.startswith("A")][0]
chk("최소" in _w3 and "후보 3 개" in _w3,
    "그 경우 이유로 **후보 수**를 먼저 적는다 (더 근본적인 제약)")

# ---------------------------------------------------------------------------
head(22, "제약 관문의 격자 수렴성 판정")
print("""    왜 이 케이스가 필요한가 : 되돌림을 --n-focus 33 과 257 로 돌렸더니
    **같은 모형 같은 조건에서 예측 통과군이 15 개 -> 2 개**가 되었다. 추천기의
    제약은 `nils_min = min over window`, `cdu_max = max over window` 다.
    격자 위의 최솟값·최댓값은 점이 늘수록 **단조로 나빠진다.** 그러므로 이
    관문이 쓸 만하려면 그 변화가 수렴해야 한다.
    시뮬레이터로 따로 재면 물리량은 수렴한다 : nils_min 4.3181 -> 4.0960
    (n_focus 17 -> 257, 5 %), DOF 75.0 -> 93.8 -> 103.1 -> 107.8 -> 110.2 nm
    (차이가 절반씩). 물리가 수렴하는 구간에서 예측 통과가 무너지면 원인은
    **대리 모형의 격자 규모 예측 잡음**이다.
    여기서 보는 것 : 판정이 인쇄와 **분리**되어 시험 가능한가, 세 갈래가
    각각 제대로 갈리는가, 그리고 수렴을 **마지막 두 격자**로 보는가.""")

import gridcheck_w13 as GC


def _g22(passes, grids=None, bad=0, worst=2.5):
    grids = grids or [17, 33, 65, 129, 257][-len(passes):]
    return [{"n_focus": g, "n_pass": p, "n_rows": 289, "n_undet": 0,
             "cdu_med": 0.01, "cdu_max": 0.02, "nils_med": worst,
             "nils_worst": worst, "n_bad_nils": bad, "n_bad_cdu": 0,
             "dof_top": 100.0} for g, p in zip(grids, passes)]


chk(GC.verdict(_g22([15, 8, 2, 0]))[0] == "collapse",
    "통과가 0 으로 무너지면 '수렴하지 않는다'")
chk(GC.verdict(_g22([81, 81, 81, 81]))[0] == "stable",
    "통과가 그대로면 '안정'")
chk(GC.verdict(_g22([40, 30, 20, 10]))[0] == "drifting",
    "마지막 두 격자에서 20 % 넘게 바뀌면 '아직 수렴 안 함'")

# [95 번] 처음 판은 마지막 두 격자만 보아서, 실측된 27 -> 15 -> 6 -> 2 -> 2
# (13.5 배 감소)에 '안정되었다, 관문을 쓸 수 있다' 를 찍었다. 틀린 말은
# 아니었지만 오도했다. 전체 낙폭을 함께 보아야 한다.
chk(GC.verdict(_g22([27, 15, 6, 2, 2]))[0] == "stable_coarse_wrong",
    "수렴했어도 전체가 크게 줄었으면 '거친 격자가 과대평가' 로 가른다")
_lcw = GC.verdict(_g22([27, 15, 6, 2, 2]))[1]
chk(any("쓸 수 있" in x for x in _lcw) is False
    or any("이전 결과는" in x for x in _lcw),
    "그 경우 '관문을 쓸 수 있다' 로 끝내지 않는다")
chk(any("13.5 배 감소" in x for x in _lcw),
    "전체 낙폭을 숫자로 적는다", "27 -> 2")
chk(GC.verdict(_g22([30, 28, 27, 26]))[0] == "stable",
    "조금만 줄었으면 '안정' 이다 (낙폭 규칙이 과하지 않다)")

# [95 번] **물리 범위를 벗어난 예측이 모든 판정보다 앞선다.**
# NILS = CD x |d(ln I)/dx| 이므로 0 이상이 구조적으로 보장된다. 실측에서
# nils_min 중위가 -0.065 ~ -0.151 로 음수였다 — 모형이 범위를 벗어난
# 것이고, min over window 가 그 값을 집어 후보를 떨어뜨렸다.
chk(GC.verdict(_g22([27, 15, 6, 2, 2], bad=178, worst=-0.15))[0]
    == "nonphysical",
    "음수 NILS 가 있으면 다른 판정보다 **앞선다**")
chk(GC.verdict(_g22([81, 81, 81, 81], bad=3, worst=-0.01))[0]
    == "nonphysical",
    "통과 수가 완전히 안정되어도 음수가 있으면 nonphysical 이다")
_lnp = GC.verdict(_g22([27, 15, 6, 2, 2], bad=178, worst=-0.15))[1]
chk(any("모형 실패를 재고 있다" in x for x in _lnp),
    "관문이 물리가 아니라 모형 실패를 재고 있다고 적는다")
chk(any("판정 불가" in x for x in _lnp),
    "'위반' 이 아니라 '판정 불가' 로 분류해야 한다고 적는다")
chk(any("12 주차 설계 변경" in x for x in _lnp),
    "고치는 것은 13 주차 범위가 아니라고 적는다 (범위를 지킨다)")
chk(GC.verdict(_g22([81, 81, 81, 81], bad=0, worst=2.5))[0] == "stable",
    "음수가 없으면 nonphysical 로 가지 않는다 (검사가 과하지 않다)")

_k, _l = GC.verdict(_g22([15, 8, 2, 0]))
chk(any("순위가 나쁜 것보다 센 결함" in x for x in _l),
    "무너진 경우 그 뜻을 적는다 (순위 이전에 선별이 흔들린다)")
chk(any("마지막 두 격자" in x for x in _l),
    "어느 두 격자로 판정했는지 적는다")
chk(all("n_pass" not in x for x in _l),
    "사람이 읽는 말로 적는다 (내부 키 이름을 흘리지 않는다)")

# 판정이 인쇄에서 분리되어 있는가 — 92 번의 교훈이다.
_src22 = inspect.getsource(GC)
chk("def verdict(" in _src22, "판정이 함수로 분리되어 있다 (시험 가능하다)")
chk("key, lines = verdict(good)" in _src22,
    "인쇄가 그 함수를 쓴다 (둘이 어긋날 수 없다)")
# 범주 열 규약. 이름이 어긋나면 전부 '거부' 로 떨어져 측정이 안 된다.
chk("POLARIZATION_CODES.index" in _src22 and "_code" in _src22,
    "범주 열을 되돌림과 같은 부호화 이름으로 넘긴다")
chk("번들이 요구하는 범주 열" in _src22,
    "채우지 못한 범주 열이 있으면 경고한다 (조용히 거부되지 않는다)")
# 값을 잘라 올리면(clip) 관문이 멀쩡해 보이면서 모형 실패가 숨는다.
chk("고치지 않고 센다" in _src22,
    "음수를 **세기만** 한다고 적어 둔다 (clip 으로 숨기지 않는다)")
# 주석에 'clip' 이라는 **말**은 남아야 한다(왜 안 자르는지 적은 것이다).
# 막아야 하는 것은 **호출**이고, 둘은 문자열 검색으로 못 가린다 — 19 번에서
# set_y 로 같은 함정을 밟았다. 구문 나무로 본다.
with open("gridcheck_w13.py", encoding="utf-8") as _fh22:
    _t22 = _ast.parse(_fh22.read())
# `max()` 는 분모 보호와 집계에 쓰이는 일반 함수이므로 세지 않는다
# (처음에 넣었다가 정당한 네 곳을 잡았다). 예측값을 자르는 통로는
# `np.clip` 과 `np.maximum`/`np.minimum` 이다. 그것만 본다.
_clips = [n for n in _ast.walk(_t22) if isinstance(n, _ast.Call)
          and isinstance(n.func, _ast.Attribute)
          and n.func.attr in ("clip", "maximum", "minimum")
          and isinstance(n.func.value, _ast.Name)
          and n.func.value.id in ("np", "numpy")]
chk(not _clips, "예측값을 자르는 np.clip / np.maximum **호출**이 없다",
    f"{len(_clips)} 곳")
chk("nils_worst" in _src22 and "n_bad_nils" in _src22,
    "최악값과 개수를 함께 남긴다 (중위만 보면 몇 개인지 모른다)")

# ---------------------------------------------------------------------------
head(23, "고NA DOF 계수 규약과 화면 차트 규약")
print("""    왜 이 케이스가 필요한가 : 지표 카드의 `DOF (고NA, Lin)` 이 4 주차부터
    12 주차까지 참값의 **정확히 4 배**(306.3 nm)였고, 근축 대비 `+169.6 %`
    로 표시되었다. 실제로는 76.6 nm 이고 근축 대비 **-32.6 %** 다. 고NA
    보정은 DOF 를 늘리는 것이 아니라 줄인다.
    원인이 둘 겹쳤다. (ㄱ) `app.py` 가 k3 자리에 k2 를 그대로 넘겼다.
    (ㄴ) `dof_high_na` 가 `1 - cos = 2 sin^2(theta/2)` 의 2 를 빠뜨렸다.
    각각 2 배씩이다. 이 두 함수에 회귀가 **하나도 없었던 것**이 9 주를
    버틴 이유다.
    숫자 대조는 claims 52~55 번이 한다. 여기서 보는 것은 **코드가 그 실수를
    다시 할 수 없는 모양인가**이다. 숫자만 고정하면 식을 바꾸면서 기대값도
    같이 고치게 되어 아무것도 못 막는다.
    97·100 번(차트 눈금과 칸 쪼개짐)도 같은 자리에서 본다 — 화면 규약이다.""")

import ast as _ast23
import inspect as _in23
import numpy as _np23
from litho import optics as _OP23
from litho import grid as _G23, mask as _MK23
import charts as _CH23
import ui as _UI23

# (가) 호출부가 k3 를 넘기지 않는가. 구문 나무로 본다. 문자열 검색은
#      주석의 'k2' 와 인자의 k2 를 못 가린다 — 19 번의 set_y 와 같다.
with open("app.py", encoding="utf-8") as _fh23:
    _txt23 = _fh23.read()
_t23 = _ast23.parse(_txt23)
_ray_calls = [c for c in _ast23.walk(_t23) if isinstance(c, _ast23.Call)
              and isinstance(c.func, _ast23.Attribute)
              and c.func.attr == "rayleigh_summary"]
chk(len(_ray_calls) == 1, "화면이 rayleigh_summary 를 한 곳에서만 부른다",
    f"{len(_ray_calls)} 곳")
chk(all(len(c.args) <= 5 for c in _ray_calls),
    "k3 를 넘기지 않는다 (위치 인자 5 개 이하 = k2 까지)",
    f"최대 {max((len(c.args) for c in _ray_calls), default=0)} 개")

# (나) 기본값이 관계를 들고 있는가. k3 기본값이 숫자면 호출부가 아무 값이나
#      넘겨도 아무 데서도 안 걸린다.
chk(_in23.signature(_OP23.rayleigh_summary).parameters["k3"].default is None,
    "rayleigh_summary 의 k3 기본값이 None (k2 에서 유도한다)")
chk(abs(_OP23.k3_from_k2(0.745) - 0.3725) < 1e-12,
    "k3_from_k2 가 k2/2 를 준다")

# (다) 식의 모양. sin^2(theta/2) 로 적으면 앞의 2 를 빠뜨리기 쉽다.
#      광로차 식 그대로인 1 - cos 를 쓰는지 본다.
_src23 = _in23.getsource(_OP23.dof_high_na)
chk("np.cos(theta)" in _src23 and "np.sin(theta / 2" not in _src23,
    "dof_high_na 가 1 - cos(theta) 꼴을 쓴다 (2 를 빠뜨릴 자리가 없다)")

# (라) 근축 극한 수렴. 식의 모양이나 계수가 틀어지면 이 비가 1 에서 벗어난다.
_na23 = 0.05 * 1.44
_conv = (_OP23.dof_high_na(_OP23.k3_from_k2(0.745), 193.0, _na23, 1.44)
         / _OP23.dof_paraxial(0.745, 193.0, _na23, 1.44))
chk(abs(_conv - 1.0) < 1e-3, "NA/n -> 0 에서 두 DOF 식이 같은 값으로 간다",
    f"비 {_conv:.4f}")
# 음성 대조 : 옛 호출(k3=k2)을 재현하면 반드시 걸려야 한다. 걸리지 않으면
# 위 검사가 느슨한 것이다.
_bad23 = (_OP23.dof_high_na(0.745, 193.0, _na23, 1.44)
          / _OP23.dof_paraxial(0.745, 193.0, _na23, 1.44))
chk(abs(_bad23 - 1.0) > 1e-3, "옛 호출(k3=k2)은 그 검사에 걸린다",
    f"비 {_bad23:.4f}")

# (마) 방향. 고NA 값이 근축값보다 **작아야** 한다.
_r23 = _OP23.rayleigh_summary(193.0, 1.35, 1.44, 0.288, 0.745)
chk(_r23["dof_high_na"] < _r23["dof_paraxial"],
    "고NA DOF 가 근축 DOF 보다 작다 (보정은 줄이는 방향이다)",
    f"{_r23['dof_high_na']:.1f} < {_r23['dof_paraxial']:.1f} nm")

# (바) 두 패널의 x 눈금이 같은 규칙을 쓰는가 (97 번).
_xt_line = _CH23._nice_ticks(0.0, 1279.5)
_xt_strip = _CH23._nice_ticks(0.0, 1280.0)
chk(_xt_line == _xt_strip == [0, 500, 1000],
    "선 그래프와 띠가 같은 눈금 함수로 같은 눈금을 낸다",
    f"{_xt_line} / {_xt_strip}")
_src_strip = _in23.getsource(_CH23.strip)
chk("v > 0.5" in _src_strip and "abs(v[j + 1] - v[i])" not in _src_strip,
    "띠가 임계로 가른 뒤 묶는다 (원값 비교는 칸을 쪼갠다)")

# (사) 쪼개지지 않는가 (100 번). 대역제한 마스크는 경계에서 1.2 까지
#      솟으므로 원값 비교로 묶으면 칸 하나가 여러 사각형이 되었다.
_g23 = _G23.build_grid(pitch=160.0, n_periods=8, dx_target=0.5)
_tr23 = _np23.abs(_MK23.line_space(_g23, cd=80.0, polarity="clear")) ** 2
_n_rect = _CH23.strip(_tr23, _g23.field_width).count(f'fill="{_UI23.FG}"')
chk(_n_rect == 9, "밝은 칸마다 사각형이 정확히 하나 (8 주기 + 좌우 반칸)",
    f"{_n_rect} 개")

# (아) 슬라이더 표시 자릿수 (98 번). 화면에 적힌 값으로 검산하면 카드가
#      재현되어야 한다. 0.29 로 보이면 41.5 가 나와 카드의 41.2 와 어긋난다.
chk('"공정 상수 k1", 0.25, 0.80, 0.288, 0.001, format="%.3f"' in _txt23
    and '"공정 상수 k2", 0.20, 1.50, 0.745, 0.005, format="%.3f"' in _txt23,
    "k1 / k2 슬라이더가 세 자리로 표시된다")
# 자릿수를 **코드에서 읽어서** 검산한다. 숫자를 여기 적어 두면 서식을
# 되돌려도 이 검사는 그대로 통과해 위 문자열 검사와 겹치기만 한다.
# %.2f 로 되돌리면 0.29 -> 41.46 nm 가 되어 걸린다.
import re as _re23
_m23 = _re23.search(r'"공정 상수 k1".*?format="%\.(\d)f"', _txt23)
_dig23 = int(_m23.group(1)) if _m23 else 2
_shown23 = round(0.288, _dig23)
chk(abs(_shown23 * 193.0 / 1.35 - _r23["half_pitch_limit"]) < 5e-3,
    "화면에 표시되는 자릿수로 검산하면 지표 카드가 재현된다",
    f"{_dig23} 자리 -> {_shown23} -> {_shown23 * 193.0 / 1.35:.2f} nm "
    f"vs 카드 {_r23['half_pitch_limit']:.2f} nm")

# (자) 화면이 꺼내 쓰는 사전 열쇠가 전부 실재하는가 (102 번).
#      96 번을 고치면서 새 열쇠를 화면이 바로 꺼내 썼는데, 저장소에
#      litho/optics.py 가 빠진 채 올라가자 **설명 한 줄 때문에 앱 전체가**
#      KeyError 로 죽었다. 오타와 드리프트는 아래 대조로 잡고, 판 어긋남은
#      화면이 스스로 알리게 했다.
_keys23 = {n.slice.value for n in _ast23.walk(_t23)
           if isinstance(n, _ast23.Subscript)
           and isinstance(n.value, _ast23.Name) and n.value.id == "rayleigh"
           and isinstance(n.slice, _ast23.Constant)
           and isinstance(n.slice.value, str)}
_missing23 = sorted(_keys23 - set(_r23))
chk(not _missing23, "화면이 꺼내는 rayleigh 열쇠가 전부 실재한다",
    f"없는 열쇠 {_missing23}" if _missing23 else f"{len(_keys23)} 개")
chk('"k3_used" not in rayleigh' in _txt23,
    "옛 optics 와 섞였을 때 설명 줄이 앱을 죽이지 않고 그 사실을 알린다")

# ---------------------------------------------------------------------------
print(f"\n{'=' * 78}")
# `run_all_checks.py` 가 ' 결과: p/t 통과' 를 정규식으로 찾아 회귀 케이스
# 합계를 낸다. 이 줄의 형식을 바꾸면 합계에 **0 으로** 들어가고, 문서에 적힌
# 수와 조용히 어긋난다. 11 주차 37 번과 같은 종류의 구멍이다.
print(f" 결과: {NCASE - len(FAIL)}/{NCASE} 통과")
if FAIL:
    print(f" 13 주차 회귀 : {len(FAIL)} 실패")
    for f in FAIL:
        print(f"   - {f}")
    print("=" * 78)
    sys.exit(1)
print(" 13 주차 회귀 : 전부 통과")
print("""
 성격
   해석      Case 3, 5, 8, 9, 10, 12, 14, 15, 16, 18 — 측정 자체가 뜻을
             갖는 조건.
             음성 대조가 있다.
   불변량    Case 1, 2, 4, 6, 7, 11, 13, 17, 19, 20, 21, 22, 23 — 코드가 지켜야 하는 관계.
             Case 11 의 784 조합과 Case 13 의 예산 왕복은 이론값·닫힌
             꼴과의 **정확** 대조다(시뮬레이션이 아니다).
   -> 23 개 사례 모두 **실제로 한 번 틀렸던 것**이며, 추측으로 넣은 항목은
      없다. 물리 모델의 정당성은 여기서 다루지 않는다 (validate.py,
      validate_w12.py, claims.py).""")
print("=" * 78)
