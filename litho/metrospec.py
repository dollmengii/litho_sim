"""주장 3을 **계측 규격**으로 바꾸는 계산.

무엇을 답하는가
---------------
주장 3은 "MSD 와 레지스트 블러는 평균으로 축퇴이고 잡음 구조로는 분리된다"
까지 말하고 멈춘다. 그 다음 질문이 실무에서 쓸 수 있는 형태다.

    "그럼 LER 을 얼마나 정밀하게 재야 분리되는가."

92 % 범위로 갈리는 신호도 계측 잡음이 100 % 면 쓸 수 없다. 그래서 신호의
크기가 아니라 **신호 대 계측 잡음**으로 답해야 한다.

모형
----
관측량 둘을 쓴다. 둘 다 실제 장비가 재는 양이다.

    y1 = 측정된 평균 CD   [nm], 계측 1 시그마 `sigma_cd`
    y2 = 측정된 LER 3시그마 [nm], 계측 1 시그마 `sigma_ler`

미지수는 theta = (msd_nm, blur_nm) 둘이다. 각 관측량의 피셔 정보는 rank 1
이므로(스칼라 관측량 하나는 방향 하나만 준다) 하나만으로는 2 차원을 정할 수
없다. 둘을 합치면

    J = (N_cd / sigma_cd^2) g_cd g_cd^T + (N_ler / sigma_ler^2) g_ler g_ler^T

이고 `g` 는 각 관측량의 theta 에 대한 기울기다. 2x2 에서

    det J = (N_cd/sigma_cd^2)(N_ler/sigma_ler^2) D^2,
    D = g_cd[0] g_ler[1] - g_cd[1] g_ler[0]

이므로 **D 가 0 이면 어떤 정밀도로도 분리되지 않는다.** D 는 두 기울기의
교차곱이고, 기울기가 평행(또는 반평행)하면 0 이다. 즉 '관측량을 늘렸다' 가
곧 '식별된다' 는 아니다. 방향이 달라야 한다.

CRB 를 풀면 닫힌 꼴이 나온다.

    Var(theta_k) = [ (sigma_ler^2/N_ler) g_cd[j]^2
                   + (sigma_cd^2 /N_cd ) g_ler[j]^2 ] / D^2     (j = 1-k)

두 항이 각각 **어느 관측량의 정밀도가 한계인지**를 바로 보여 준다. 그래서
'LER 을 더 정밀하게 재라' 와 'CD 를 더 정밀하게 재라' 를 섞지 않고 가를 수
있다. 역으로 목표 표준편차를 주면 필요한 정밀도가 해석적으로 풀린다.

왜 `identify.py` 를 쓰지 않는가
-------------------------------
`identify.py` 의 피셔 정보는 **광자 계수 벡터**에 대한 모형이다. LER 은 2 차원
확률장의 행 사이 변동에서 나오는 양이므로 1 차원 평균 계수 프로파일의 함수가
아니다. 그 모형 안에 LER 이 없다. 그래서 여기서는 관측량을 요약통계
(CD 평균, LER 3시그마) 로 두고 기울기를 수치로 잰다. 대가는 기울기에
몬테카를로 표집 잡음이 섞이는 것이고, 그 크기를 함께 보고한다.

**임계값을 고정해야 한다** (13 주차에 한 번 틀린 것)
---------------------------------------------------
축퇴는 '같은 임계값, 같은 노광량' 에서 성립한다. 임계값을 MSD 적용 후 상에서
매번 다시 잡으면 그 규약이 정보를 흘려 축퇴가 깨진 것처럼 보인다. 실측으로
결정론 CD 가 1.1594 nm 갈렸는데(씨앗 잡음의 133 배), 임계값을 평활 없는 상에서
한 번 잡아 고정하니 범위가 **정확히 0.000e+00** 이 되었다. 즉 축퇴는 참이고
어긋남은 규약 실수였다. dose-to-size 를 쓰면 CD 가 목표에 못 박혀 정보가
아예 0 이 되므로 그것도 쓸 수 없다.

함께 드러난 것 : **측정되는 평균은 축퇴가 아니다.** 결정론 CD 는 기계
정밀도로 같은데 `cd_mean` 은 0.2337 nm (씨앗 잡음의 26 배) 갈린다. 원인은
역치 추정기의 분산 유발 치우침이다. 치우침을 `LER^2` 에 비례로 적합하면
R^2 = 0.9998 이고 `LER` 1 차 적합은 0.9563 이며, 형성률이 네 배분 모두
1.000 이므로 '추출 실패로 평균이 치우쳤다' 는 대안 설명은 배제된다.

**그런데 그것이 독립한 두 번째 채널은 아니다** (13 주차 논리 검토에서
내 서술을 정정한 것). 치우침이 `LER^2` 의 함수이므로 평균 CD 에 더해지는
그 항은 `grad LER` 과 평행한 방향만 보탠다. 새 방향이 아니다. 실제로 두
관측량이 함께 식별하는 이유는 더 단순하다.

    결정론 CD 는 **합성 평활 `hypot(msd, blur)` 만의 함수**다
    (5 개 `hypot` 값에서 배분을 바꿔도 차가 0.00e+00).
    -> CD 가 **합성 크기**를 주고, LER 이 **배분**을 준다.

수치로도 맞는다. `hypot = 10.44` 에서 `f'(hypot) = +0.10374` 이고,
실측 `g_cd` 를 `grad hypot` 과 `grad LER` 로 분해하면 계수가 `+0.10054`,
`+0.06462` 다. 앞쪽이 `f'` 와 3 % 안에서 일치하고 `grad LER` 쪽은 작다.
따라서 '축퇴를 깨는 경로가 둘' 이 아니라 **'CD 는 합성, LER 은 배분'**
이라고 적어야 맞다.

반드시 함께 적어야 하는 전제 셋
-------------------------------
**(1) 관측량 2 개로는 미지수 2 개까지만 정할 수 있다.** 스칼라 관측량
하나의 피셔 정보는 rank 1 이므로 `rank(J) <= 관측량 수` 다. 미지수를 3 개로
늘리면 `J` 가 특이행렬이 되어 CRB 가 사실상 무한이다(수치로 조건수
1.4e+17, sd ~1e+6). 즉 이 모듈의 숫자는 **초점·노광량·플레어가 정확히
알려져 있다**는 전제 위에 선다. 주장 1 은 바로 그 5 변수 문제에서 MSD 가
비식별이라고 적었으므로, **이 결과와 주장 1 은 다른 문제를 다룬다.** 필요
조건으로 적으면 : 미지수 n 개를 가르려면 서로 다른 방향의 독립 관측량이
**최소 n 개** 필요하고, (CD, LER) 은 2 개다.

**(2) CD 와 LER 의 계측 오차가 독립이라고 가정했다.** 둘은 같은 SEM 영상
에서 나오므로 독립일 이유가 없다. 상관계수 `rho` 를 넣어 풀면
(`J = G^T Sigma^-1 G`) 영향은 **일률적이지 않다** : `rho = +0.9` 에서
sd(msd) 가 1.15 배 나빠지고 sd(blur) 는 0.35 배 좋아지며, `rho = -0.9`
에서는 반대로 sd(msd) 0.83 배, sd(blur) 1.37 배다. 즉 독립 가정은
보수적인 쪽도 아니다. 다만 영향이 1.4 배 안에 갇히므로 '블러는 결정
가능, MSD 는 불가' 라는 결론 자체는 뒤집히지 않는다. 숫자에는 이 ±40 %
를 붙여 읽어야 한다.

**(3) CRB 는 하한이므로 부정 결론에만 엄격하다.** "MSD 는 결정 불가" 는
어떤 불편추정기도 그보다 잘할 수 없다는 뜻이므로 **엄격하다.** 반대로
"블러는 1.07 nm 로 결정 가능" 은 하한이 작다는 뜻일 뿐이고, 그 하한을
달성하는 추정기가 있다는 것은 **증명하지 않았다.** 긍정 결론은 조건부로
읽어야 한다.

보고할 때의 주의
----------------
이 과제의 LER 은 **산탄 잡음만** 고려한 하한이라 절대값이 실제보다 훨씬
작다(여기 조건에서 0.18~0.32 nm, 실제 레지스트는 2~4 nm 급). 따라서 필요한
정밀도를 절대값(nm)으로 인용하면 옮겨 쓸 수 없다. **상대 정밀도**(LER 값의
몇 %)로 적어야 다른 조건에 옮길 수 있다.
"""

from __future__ import annotations

import math

import numpy as np


def cross(g_cd, g_ler) -> float:
    """두 기울기의 교차곱 `D`. 0 이면 어떤 정밀도로도 분리되지 않는다."""
    a = np.asarray(g_cd, dtype=np.float64)
    b = np.asarray(g_ler, dtype=np.float64)
    return float(a[0] * b[1] - a[1] * b[0])


def angle_deg(g_cd, g_ler) -> float:
    """두 기울기 사이각 [도]. 0 또는 180 에 가까우면 공선이다.

    **180 도는 0 도만큼 나쁘다.** 반평행도 공선이므로 `D` 가 0 으로 간다.
    그래서 사이각만 보지 말고 `|sin|` 이나 `D` 를 함께 봐야 한다. 실측으로
    이 과제의 조건에서 사이각은 149~165 도였고, 처음에 그것을 '거의 직교에서
    멀다' 가 아니라 '잘 갈린다' 로 잘못 읽을 수 있었다.
    """
    a = np.asarray(g_cd, dtype=np.float64)
    b = np.asarray(g_ler, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def crb(g_cd, g_ler, sigma_cd: float, sigma_ler: float,
        n_cd: int = 1, n_ler: int = 1) -> dict:
    """계측 정밀도가 주어졌을 때 (msd, blur) 의 크라메르-라오 하한.

    돌려주는 것
    -----------
    sd_msd, sd_blur : 달성 가능한 표준편차 [nm]. 하한이므로 **어떤
                      추정기도 이보다 잘할 수 없다.**
    binds_msd, binds_blur : 각 미지수에서 **어느 관측량의 정밀도가 한계인가**
                      ("cd" 또는 "ler"). 두 항의 크기를 비교한 결과다.
    """
    gc = np.asarray(g_cd, dtype=np.float64)
    gl = np.asarray(g_ler, dtype=np.float64)
    D = cross(gc, gl)
    out = {"D": D, "angle_deg": angle_deg(gc, gl),
           "sin_abs": abs(math.sin(math.radians(angle_deg(gc, gl)))),
           "sigma_cd": float(sigma_cd), "sigma_ler": float(sigma_ler),
           "n_cd": int(n_cd), "n_ler": int(n_ler)}
    if D == 0.0 or not np.isfinite(D):
        out.update({"sd_msd": float("inf"), "sd_blur": float("inf"),
                    "why": "두 기울기가 공선이다. 어떤 정밀도로도 분리되지 "
                           "않는다(교차곱 0)."})
        return out
    vc = float(sigma_cd) ** 2 / max(int(n_cd), 1)
    vl = float(sigma_ler) ** 2 / max(int(n_ler), 1)
    for k, j, name in ((0, 1, "msd"), (1, 0, "blur")):
        t_l = vl * gc[j] ** 2 / D ** 2          # LER 정밀도가 내는 항
        t_c = vc * gl[j] ** 2 / D ** 2          # CD  정밀도가 내는 항
        out[f"sd_{name}"] = float(math.sqrt(max(t_l + t_c, 0.0)))
        out[f"term_ler_{name}"] = float(t_l)
        out[f"term_cd_{name}"] = float(t_c)
        out[f"binds_{name}"] = "cd" if t_c > t_l else "ler"
    return out


def required_sigma(g_cd, g_ler, target_sd: float, param: str,
                   improve: str, sigma_other: float,
                   n_cd: int = 1, n_ler: int = 1) -> dict:
    """목표 표준편차를 주면 **필요한 계측 정밀도**를 역산한다.

    `param` 은 "msd" 또는 "blur", `improve` 는 정밀도를 올릴 관측량
    ("cd" 또는 "ler"), `sigma_other` 는 나머지 관측량의 (고정된) 정밀도다.

    도달 불가일 수 있다. 나머지 관측량이 내는 항만으로 이미 목표를 넘으면
    그쪽을 완벽하게 재도 목표에 못 미친다. 그때는 `sigma` 를 None 으로 주고
    **그 나머지 관측량에 필요한 정밀도**를 함께 돌려준다. 그것이 실제로
    고쳐야 하는 쪽이기 때문이다.
    """
    gc = np.asarray(g_cd, dtype=np.float64)
    gl = np.asarray(g_ler, dtype=np.float64)
    D = cross(gc, gl)
    j = 1 if param == "msd" else 0
    if D == 0.0 or not np.isfinite(D):
        return {"sigma": None, "why": "두 기울기가 공선이다. 분리 불가."}
    tgt = float(target_sd) ** 2 * D ** 2
    if improve == "ler":
        n_imp, n_oth = max(int(n_ler), 1), max(int(n_cd), 1)
        coef_imp, coef_oth = gc[j] ** 2, gl[j] ** 2
    elif improve == "cd":
        n_imp, n_oth = max(int(n_cd), 1), max(int(n_ler), 1)
        coef_imp, coef_oth = gl[j] ** 2, gc[j] ** 2
    else:
        raise ValueError("improve 는 'cd' 또는 'ler' 여야 한다")
    floor = float(sigma_other) ** 2 / n_oth * coef_oth
    if coef_imp <= 0:
        return {"sigma": None,
                "why": f"{improve} 의 기울기 성분이 0 이라 정밀도를 올려도 "
                       f"{param} 에 도움이 되지 않는다"}
    if tgt <= floor:
        # 나머지 쪽이 한계다. 그쪽에 필요한 정밀도를 돌려준다.
        need_oth = math.sqrt(max(tgt, 0.0) * n_oth / coef_oth)
        return {"sigma": None, "floor_sd": math.sqrt(floor / D ** 2),
                "other_required": need_oth,
                "why": (f"{improve} 를 완벽하게 재도 목표에 못 미친다. "
                        f"나머지 관측량의 정밀도가 한계이며, 그쪽을 "
                        f"{need_oth:.4g} nm 까지 올려야 한다.")}
    return {"sigma": math.sqrt((tgt - floor) * n_imp / coef_imp),
            "floor_sd": math.sqrt(floor / D ** 2)}


def budget(g_cd, g_ler, target_sd: float, param: str,
           n_cd: int = 1, n_ler: int = 1, share: float = 0.5) -> dict:
    """목표를 맞추려면 **두 관측량을 동시에** 어디까지 올려야 하는가.

    `required_sigma` 는 한쪽만 올리는 경우를 푼다. 그런데 실측에서 MSD 는
    한쪽만 올려서는 목표에 닿지 않았다. CD 를 완벽하게 재도 LER 항만으로
    2.574 nm, LER 을 완벽하게 재도 CD 항만으로 15.08 nm 였다. 그런 경우
    답은 '어느 쪽을 고쳐라' 가 아니라 **예산 분배**다.

    분산 목표를 `share : 1 - share` 로 나눠 각 관측량에 필요한 정밀도를
    돌려준다. 기본은 반반이다. 반반이 최적이라는 뜻은 아니고, 한쪽이 훨씬
    싸면 그쪽에 더 싣는 것이 낫다. `share` 를 바꿔 그 교환을 볼 수 있다.
    """
    gc = np.asarray(g_cd, dtype=np.float64)
    gl = np.asarray(g_ler, dtype=np.float64)
    D = cross(gc, gl)
    j = 1 if param == "msd" else 0
    if D == 0.0 or not np.isfinite(D):
        return {"sigma_cd": None, "sigma_ler": None,
                "why": "두 기울기가 공선이다. 분리 불가."}
    if not 0.0 < share < 1.0:
        raise ValueError("share 는 0 과 1 사이여야 한다")
    v_cd = float(target_sd) ** 2 * share
    v_ler = float(target_sd) ** 2 * (1.0 - share)
    out = {"D": D, "share": float(share)}
    out["sigma_cd"] = (math.sqrt(v_cd * D ** 2 * max(int(n_cd), 1))
                       / abs(gl[j])) if gl[j] != 0 else None
    out["sigma_ler"] = (math.sqrt(v_ler * D ** 2 * max(int(n_ler), 1))
                        / abs(gc[j])) if gc[j] != 0 else None
    if out["sigma_cd"] is None:
        out["note_cd"] = (f"LER 기울기의 {param} 성분이 0 이라 CD 정밀도는 "
                          f"{param} 에 영향이 없다")
    if out["sigma_ler"] is None:
        out["note_ler"] = (f"CD 기울기의 {param} 성분이 0 이라 LER 정밀도는 "
                           f"{param} 에 영향이 없다")
    return out


def sites_needed(sigma_tool: float, sigma_required: float) -> int | None:
    """장비 정밀도에서 목표 정밀도까지 **몇 지점을 평균해야** 하는가.

    독립 지점 N 개를 평균하면 표준오차가 `1/sqrt(N)` 로 줄어든다. 계측
    레시피가 수백 지점을 재는 것은 현장에서 흔하므로, 달성 불가로 보이는
    정밀도가 지점 수로 닿는 경우가 있다. 그 환산이 이 함수다.

    전제 : 지점 사이가 독립이다. 같은 웨이퍼의 인접 지점은 계통 성분을
    공유하므로 실제로는 이보다 많이 필요하다. 그 사실을 함께 적어야 한다.
    """
    if sigma_required is None or sigma_required <= 0:
        return None
    if sigma_tool <= sigma_required:
        return 1
    return int(math.ceil((sigma_tool / sigma_required) ** 2))
