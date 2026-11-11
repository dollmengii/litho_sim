"""
identify.py -- 공정 조건의 식별 가능성 분석 (Fisher 정보 / Cramer-Rao 하한)

무엇을 답하는 모듈인가
----------------------
"찍힌 패턴을 보고 공정 조건을 되짚을 수 있는가" 를 조건별로 정량화한다.
6~9주차에서는 조건을 여러 개 만들어 관측값을 비교하는 경험적 방법을 썼다.
본 모듈은 같은 질문에 추정이론의 하한으로 답한다. 경험적 방법이 "이 조건
들에서는 구별이 안 되더라" 라면, 여기서는 "어떤 추정량을 쓰더라도 이보다
정확할 수 없다" 가 된다.

핵심 결과 (10주차 측정)
-----------------------
식별 한계를 정하는 것은 광자 잡음이 아니라 민감도 벡터의 공선성이다.
미지수가 초점 하나면 광자 통계만으로 0.01 nm 까지 특정할 수 있으나,
플레어를 함께 모르면 초점을 전혀 특정할 수 없다. 두 민감도 벡터의
사이각이 1.4도에 불과하기 때문이다.

관측 모형
---------
레지스트에 입사하는 광자 수는 포아송 분포를 따른다. 레지스트 블러(산 확산)
는 광자가 흡수된 뒤에 작용하므로, 블러 이후의 관측량은 평균과 공분산이
모두 블러의 영향을 받는다.

    lambda(theta) = N_ref * I_msd(theta)        광자 수 기대값 (블러 이전)
    mu(theta)     = G lambda(theta)             블러 이후 평균
    Sigma(theta)  = G diag(lambda(theta)) G^T   블러 이후 공분산

여기서 G 는 블러 연산 행렬이다. 반면 스캔 동기화 오차(MSD)는 광자 샘플링
이전의 광강도에 작용하므로 lambda 자체를 평활하며, 공분산을 상관시키지
않는다. 즉 두 효과는 평균에서는 거의 구별되지 않으나 공분산에서 구별된다.
이 차이를 반영하지 않으면 블러와 MSD 가 완전히 축퇴된 것처럼 보인다.

큰 광자 수에서 포아송은 가우시안으로 근사되므로, Slepian-Bangs 공식으로
Fisher 정보 행렬을 쓴다.

    F_ij = (d mu/d th_i)^T Sigma^-1 (d mu/d th_j)
         + 0.5 * tr( Sigma^-1 dSigma/dth_i Sigma^-1 dSigma/dth_j )

앞 항은 평균의 이동에서, 뒤 항은 잡음 구조의 변화에서 오는 정보이다.
블러처럼 평균을 거의 바꾸지 않으면서 잡음 상관을 바꾸는 변수는 뒤 항에서만
식별된다.

하한
----
    Cov(theta_hat) >= F^-1        (불편추정량에 대해)
F 가 특이에 가까우면 역행렬이 수치적으로 무의미하므로, 고유분해 후
유효 고유값만 남겨 유사역행렬로 계산하고 고유값 스펙트럼을 함께 보고한다.
스펙트럼에서 살아남은 방향의 수가 '식별 가능한 자유도' 이다.

주의 : 본 하한은 산탄 잡음만 고려한다. 실제 계측에는 다른 잡음이 더해지므로
식별 한계는 넓어지기만 한다. 즉 본 결과는 낙관적인 방향으로만 틀린다.
"""

from dataclasses import dataclass, replace

import numpy as np

# [정리] 예전에는 scipy.ndimage.gaussian_filter1d 를 들여왔는데 쓰이지
# 않는 채로 남아 있었다. 이 모듈은 _blur_matrix 에서 **일부러**
# imaging.gaussian_blur_periodic(스펙트럼 평활)을 쓴다. 커널을 잘라내는
# scipy 구현을 쓰면 공분산 전파가 결상 쪽 평활과 어긋나기 때문이다.
# 죽은 import 를 남겨 두면 읽는 사람이 반대로 이해하므로 지운다.

from . import grid as G
from . import imaging as IM
from . import mask as MK
from . import stochastic as ST

#: 추정 대상 파라미터와 중심차분 폭. 단위는 각 변수의 고유 단위.
PARAM_STEPS = {
    "defocus": 4.0,      # nm
    "dose_rel": 0.01,    # 배율
    "blur_nm": 0.3,      # nm
    "flare": 0.002,      # 비율
    "msd_nm": 0.3,       # nm
    "duty": 0.005,       # 비율
    "pitch": 1.0,        # nm
}

#: 기본 추정 대상. 24개 조건 전부를 미지수로 두면 의미 있는 하한이 나오지
#: 않는다. 실제로 계측으로부터 되짚고 싶은 변수만 고른다.
DEFAULT_PARAMS = ("defocus", "dose_rel", "blur_nm", "flare", "msd_nm")


@dataclass
class IdentifySpec:
    """식별성 분석 한 건의 조건."""
    # 광학계 (기지값으로 둔다)
    wavelength: float = 193.0
    na: float = 1.35
    n_medium: float = 1.44
    n_resist: float = 1.70
    sigma_out: float = 0.7
    sigma_in: float = 0.0
    polarization: str = IM.POL_TE
    source_shape: str = IM.SOURCE_CONVENTIONAL
    pole_offset: float = 0.0
    pole_radius: float = 0.15
    mask_type: str = MK.MASK_BINARY
    att_t: float = MK.ATT_PSM_DEFAULT_T

    # 추정 대상 파라미터의 공칭값
    pitch: float = 200.0
    duty: float = 0.5
    defocus: float = 80.0
    dose_rel: float = 1.0
    blur_nm: float = 8.0
    flare: float = 0.02
    msd_nm: float = 3.0

    # 계측 조건
    dose_mj: float = 30.0
    y_length: float = 2048.0
    dy: float = 2.0

    # 수치 설정
    n_periods: int = 6
    n_source_grid: int = 31
    dx_target: float | None = None

    def grid(self) -> G.SimGrid:
        dx = self.dx_target or G.nyquist_dx(self.wavelength, self.na, self.sigma_out)
        return G.build_grid(self.pitch, self.n_periods, dx)


def _blur_matrix(n: int, sigma_px: float) -> np.ndarray:
    """주기 경계 가우시안 블러의 행렬 표현 G.

    공분산 전파 G diag(lambda) G^T 를 계산하려면 행렬이 필요하다.
    순환 행렬이므로 단위 벡터 하나만 흘려 커널을 얻고 굴린다.
    """
    if sigma_px <= 0:
        return np.eye(n)
    e = np.zeros(n)
    e[0] = 1.0
    # imaging.gaussian_blur_periodic 과 같은 연산을 써야 한다. 커널을
    # 잘라내는 scipy 구현을 쓰면 공분산 전파가 결상 쪽 평활과 미세하게
    # 어긋나, 블러와 MSD 의 축퇴 구조 분석이 수치 잡음에 오염된다.
    k = IM.gaussian_blur_periodic(e, sigma_px)
    return np.stack([np.roll(k, i) for i in range(n)])


def expected_counts(spec: IdentifySpec) -> tuple[np.ndarray, "G.SimGrid"]:
    """레지스트 블러 '이전' 의 기대 광자 수 lambda 와 격자를 돌려준다.

    몬테카를로 검증에서 실제 물리 과정(Poisson 표본 -> 블러)을 재현하려면
    블러 이전의 세기 기대값이 필요하다. mean_and_cov 와 같은 식을 두 곳에
    적어 두면 한쪽만 고쳤을 때 조용히 어긋나므로 여기로 모은다.
    """
    g = spec.grid()
    m = MK.line_space(g, cd=spec.duty * spec.pitch,
                      mask_type=spec.mask_type, att_t=spec.att_t)
    inten = IM.aerial_image(
        g, m, spec.wavelength, spec.na,
        n_medium=spec.n_medium, n_resist=spec.n_resist,
        sigma_out=spec.sigma_out, sigma_in=spec.sigma_in,
        defocus=spec.defocus, n_source_grid=spec.n_source_grid,
        polarization=spec.polarization, source_shape=spec.source_shape,
        pole_offset=spec.pole_offset, pole_radius=spec.pole_radius,
        flare=spec.flare, msd_nm=spec.msd_nm,
    )
    inten = np.asarray(inten).real * spec.dose_rel
    n_rows = max(1.0, spec.y_length / spec.dy)
    n_ref = ST.photon_density(spec.dose_mj, spec.wavelength) * g.dx * spec.dy * n_rows
    return np.maximum(n_ref * inten, 1e-9), g


def mean_and_cov(spec: IdentifySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """관측량(블러 이후 광자 수)의 평균과 공분산.

    Returns
    -------
    (mu, Sigma, G_blur)
    """
    lam, g = expected_counts(spec)
    gm = _blur_matrix(g.n_points, spec.blur_nm / g.dx)
    return gm @ lam, (gm * lam[None, :]) @ gm.T, gm


def _perturb(spec: IdentifySpec, name: str, delta: float) -> IdentifySpec:
    return replace(spec, **{name: getattr(spec, name) + delta})


def sensitivity(spec: IdentifySpec,
                params: tuple[str, ...] = DEFAULT_PARAMS) -> dict:
    """각 파라미터에 대한 평균·공분산의 중심차분 미분."""
    dmu, dsig = [], []
    for p in params:
        if p not in PARAM_STEPS:
            raise ValueError(f"미지원 파라미터입니다: {p} (지원: {list(PARAM_STEPS)})")
        h = PARAM_STEPS[p]
        mp, sp, _ = mean_and_cov(_perturb(spec, p, +h))
        mm, sm, _ = mean_and_cov(_perturb(spec, p, -h))
        dmu.append((mp - mm) / (2 * h))
        dsig.append((sp - sm) / (2 * h))
    return {"params": params, "dmu": np.stack(dmu), "dsigma": np.stack(dsig)}


def fisher_information(spec: IdentifySpec,
                       params: tuple[str, ...] = DEFAULT_PARAMS,
                       include_cov_term: bool = True) -> dict:
    """Fisher 정보 행렬 (Slepian-Bangs).

    include_cov_term=False 로 두면 평균 이동 항만 쓴다. 블러처럼 잡음
    구조로만 식별되는 변수가 어떻게 사라지는지 비교할 때 쓴다.
    """
    mu, sigma, _ = mean_and_cov(spec)
    sens = sensitivity(spec, params)
    dmu, dsig = sens["dmu"], sens["dsigma"]

    n = len(mu)
    # Sigma 도 조건수가 나쁠 수 있으므로 대각 지터를 넣어 안정화한다.
    jitter = 1e-10 * float(np.trace(sigma)) / n
    sig_reg = sigma + jitter * np.eye(n)
    sinv = np.linalg.inv(sig_reg)

    k = len(params)
    f = np.zeros((k, k))
    a = dmu @ sinv                       # (k, n)
    f += a @ dmu.T                       # 평균 항
    if include_cov_term:
        b = [sinv @ d for d in dsig]     # Sigma^-1 dSigma/dth_i
        for i in range(k):
            for j in range(i, k):
                v = 0.5 * float(np.einsum("ij,ji->", b[i], b[j]))
                f[i, j] += v
                if i != j:
                    f[j, i] += v
    return {"params": params, "fim": f, "mu": mu, "sigma": sigma,
            "dmu": dmu, "dsigma": dsig}


def photon_budget_rel(spec: IdentifySpec) -> float:
    """기준 조건(y_length 2048 nm, dy 2 nm, 30 mJ/cm^2) 대비 총 광자 예산의 비.

    총 광자 수는 측정 행 수(y_length / dy)와 노광량의 곱에 비례한다.
    식별 하한은 이 값의 제곱근에 반비례하므로(verify.py J2 에서
    sigma x sqrt(N) 상대편차 7.7e-07 로 확인), 조건이 다른 두 결과를
    비교할 때는 반드시 이 값으로 환산해야 한다.

    [정리] 이전에는 기준값과 반환값 양쪽에 임계값 0.30 이 곱해져 있었다.
    서로 상쇄되어 숫자는 맞았지만, 임계값이 광자 예산에 반영된 것처럼
    보였다. 임계값은 경계의 위치를 정하는 것이고 광자 수에는 들어가지
    않는다. `IdentifySpec` 에는 threshold 필드도 없어 대응물 없는 잔재였고,
    나중에 한쪽만 고치면 그대로 버그가 된다. 양쪽에서 제거했다.
    """
    ref = 2048.0 / 2.0 * 30.0
    n_rows = max(1.0, spec.y_length / spec.dy)
    return float(n_rows * spec.dose_mj / ref)


#: 영공간 성분 판정 한계. 고유벡터 성분의 수치 잡음(~sqrt(eps) = 1.5e-8)
#: 보다 크고, 물리적으로 유의미한 성분보다는 작게 둔다. 이 값이 느슨하면
#: (이전 0.999) 영공간에 성분을 가진 파라미터가 걸러지지 않아 의사역행렬이
#: '투영된' 유한 하한을 내놓고, 미지수를 늘릴 때 하한이 좋아지는 단조성
#: 위반이 생긴다(verify.py J4 참조).
NULLSPACE_LEAK_TOL = 1e-6

#: 각 파라미터의 '실용적 식별 가능' 판정 기준. 하한이 이 값을 넘으면
#: 사실상 되짚을 수 없다고 본다. 공정에서 의미 있는 분해능 수준으로 둔다.
PRACTICAL_SCALE = {
    "defocus": 50.0,     # nm. 초점심도의 수분의 일
    "dose_rel": 0.05,    # 5 %
    "blur_nm": 5.0,      # nm
    "flare": 0.02,       # 2 %p
    "msd_nm": 5.0,       # nm
    "duty": 0.05,
    "pitch": 5.0,        # nm
}


def crb(spec: IdentifySpec,
        params: tuple[str, ...] = DEFAULT_PARAMS,
        include_cov_term: bool = True,
        rcond: float | None = None) -> dict:
    """Cramer-Rao 하한과 식별 가능한 자유도.

    [절단 기준] FIM 이 특이에 가까우면 역행렬이 수치적으로 무의미하므로
    고유분해 후 작은 고유값을 버린다. 이때 기준을 '최대 고유값 대비' 로
    잡으면 함정이 있다. 평균 항은 광자 수에 비례해 커지지만 공분산 항은
    광자 수와 무관하므로, 노광량을 올리면 공분산 항이 받치던 모드가
    정보를 잃지 않았는데도 상대 기준 아래로 밀려난다. 실제로 rcond=1e-10
    으로 두었을 때 노광량 200 mJ/cm^2 이상에서 자유도가 4에서 3으로
    떨어지는 현상이 나타났는데, 이는 물리가 아니라 절단 기준의 결함이었다.
    따라서 기준을 기계 정밀도 수준(k * eps)으로 낮춰, 진짜 수치적 0 만
    버린다.

    [식별 가능 판정] 하한이 유한해도 공정에서 쓸 수 없을 만큼 크면 실용적
    으로는 식별 불가이다. PRACTICAL_SCALE 과 비교해 별도 플래그로 보고하며,
    sigma_min 자체는 가공하지 않은 값을 그대로 돌려준다.

    Returns
    -------
    sigma_min    : 각 파라미터의 표준편차 하한 (가공하지 않은 값).
                   계측 조건에 의존하며 photon_budget_rel 의 제곱근에
                   반비례한다. 절대값으로 인용하지 말 것.
    identifiable : 실용 기준 통과 여부
    rank         : 수치적으로 유효한 자유도
    eigvals      : FIM 고유값 (내림차순)
    worst_dir    : 가장 작은 고유값에 대응하는 방향
    """
    out = fisher_information(spec, params, include_cov_term)
    f = out["fim"]
    # FIM 은 이론적으로 대칭 준양정부호이다. 중심차분과 공분산 항의 대각합
    # 계산에서 생기는 반올림 때문에 고유값이 기계 정밀도 수준으로 음수가 될
    # 수 있다(관측 사례: 최대 고유값의 1e-17 배). 대칭화한 뒤 음수 고유값은
    # 0 으로 보고 소멸 방향으로 처리한다. 이는 정보가 없는 방향이라는 뜻이며
    # 결과를 왜곡하지 않는다.
    f = 0.5 * (f + f.T)
    w, v = np.linalg.eigh(f)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    n_neg = int((w < 0).sum())
    w = np.maximum(w, 0.0)

    k = len(params)
    if rcond is None:
        rcond = k * np.finfo(float).eps   # 진짜 수치적 0 만 버린다
    wmax = float(w[0]) if w.size else 0.0
    keep = w > max(wmax * rcond, 0.0)
    rank = int(keep.sum())

    if rank == 0:
        sig_min = np.full(k, np.inf)
    else:
        pinv = (v[:, keep] / w[keep]) @ v[:, keep].T
        sig_min = np.sqrt(np.abs(np.diag(pinv)))

        # [10주차 보정] FIM 이 특이할 때 의사역행렬은 '식별 가능한 부분공간
        # 으로 투영한' 하한을 돌려준다. 이것은 원래 파라미터의 하한이 아니다.
        # 어떤 파라미터의 단위벡터가 영공간에 성분을 가지면, 그 방향으로는
        # 관측이 전혀 변하지 않으므로 CRB 는 **무한**이다.
        #
        # 기존 기준 leak > 0.999 는 너무 느슨했다. 실제 사례에서 5 변수
        # 동시 추정 시 rank 가 4/5 이고 영공간 방향의 defocus 성분이
        # 0.9127 이었는데, 0.999 를 넘지 못해 걸러지지 않았다. 그 결과
        # defocus 의 하한이 무한이어야 하는데 0.315 nm 로 보고되었고,
        # 미지수를 늘렸는데 하한이 좋아지는(1.891 -> 0.315) 단조성 위반이
        # 나타났다. 미지수가 늘면 하한은 절대 좋아질 수 없다.
        #
        # 올바른 기준은 '성분이 수치 잡음보다 큰가' 이다. 고유벡터 성분의
        # 수치 잡음은 sqrt(eps) ~ 1.5e-8 수준이므로 1e-6 을 쓴다.
        leak = np.zeros(k)
        if rank < k:
            lost = v[:, ~keep]
            leak = np.sqrt((lost ** 2).sum(axis=1))
            sig_min = np.where(leak > NULLSPACE_LEAK_TOL, np.inf, sig_min)

    ident = np.array([
        bool(np.isfinite(s) and s <= PRACTICAL_SCALE.get(p, np.inf))
        for p, s in zip(params, sig_min)
    ])

    out.update({
        "sigma_min": sig_min,
        "nullspace_leak": leak,   # 영공간 성분. 0 이 아니면 하한이 무한이다.
        "identifiable": ident,
        "n_identifiable": int(ident.sum()),
        "rank": rank,
        "n_params": len(params),
        "eigvals": w,
        "cond": float(wmax / w[-1]) if w[-1] > 0 else np.inf,
        "worst_dir": v[:, -1],
        "n_negative_eig": n_neg,   # 0 이 아니면 반올림 수준인지 확인할 것
    })
    return out


def collinearity(spec: IdentifySpec,
                 params: tuple[str, ...] = DEFAULT_PARAMS) -> dict:
    """민감도 벡터 사이의 코사인 유사도 행렬.

    1 에 가까울수록 두 파라미터가 관측량에 같은 방식으로 나타나 구별되지
    않는다. 사이각이 식별 한계의 직관적 척도이다. 평균 민감도만 본다.
    """
    sens = sensitivity(spec, params)
    d = sens["dmu"]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.where(norm < 1e-300, 1.0, norm)
    u = d / norm
    c = np.abs(u @ u.T)
    np.fill_diagonal(c, 1.0)

    k = len(params)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append({
                "a": params[i], "b": params[j],
                "cos": float(c[i, j]),
                "angle_deg": float(np.degrees(np.arccos(min(c[i, j], 1.0)))),
            })
    pairs.sort(key=lambda r: -r["cos"])
    return {"params": params, "matrix": c, "pairs": pairs,
            "norms": norm.ravel()}


def observable_ladder(spec: IdentifySpec,
                      params: tuple[str, ...] = DEFAULT_PARAMS) -> list[dict]:
    """미지수를 하나씩 늘려가며 식별 한계가 어떻게 무너지는지 추적한다.

    프로젝트의 핵심 주장을 그대로 보여주는 절차이다. 광자 잡음이 아니라
    '무엇을 모르는가' 가 한계를 정한다.
    """
    rows = []
    for k in range(1, len(params) + 1):
        sub = params[:k]
        r = crb(spec, sub)
        rows.append({
            "params": sub,
            "n_unknown": k,
            "sigma_min": dict(zip(sub, r["sigma_min"])),
            "rank": r["rank"],
            "cond": r["cond"],
        })
    return rows


def identifiability_map(base: IdentifySpec,
                        axis: str,
                        values,
                        params: tuple[str, ...] = DEFAULT_PARAMS,
                        target: str = "defocus") -> dict:
    """조건 공간의 한 축을 훑으며 식별 하한의 지도를 만든다.

    Monte Carlo 가 필요 없어 대량 생성 대비 매우 저렴하다. 조건 공간을
    촘촘히 덮는 데 쓴다.
    """
    xs, ys, ranks = [], [], []
    for v in values:
        sp = replace(base, **{axis: float(v)})
        try:
            r = crb(sp, params)
        except Exception:
            xs.append(float(v)); ys.append(np.nan); ranks.append(0)
            continue
        idx = list(params).index(target)
        xs.append(float(v))
        ys.append(float(r["sigma_min"][idx]))
        ranks.append(r["rank"])
    return {"axis": axis, "values": np.array(xs),
            "sigma_min": np.array(ys), "rank": np.array(ranks),
            "target": target, "params": params}
