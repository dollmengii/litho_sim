"""
validate_w7.py -- 7주차 정량 검증 (확률 모델 수렴성 / 데이터셋 설계)

실행: py validate_w7.py
"""

import os
import tempfile
import time

import sys

import numpy as np

import validate_common as VC

from litho import dataset as DS
from litho import grid as G
from litho import imaging as IM
from litho import mask as MK
from litho import process as PR
from litho import stochastic as ST

LAM, NA, N_MED, SIGMA = 193.0, 1.35, 1.44, 0.7
PITCH, THRESH, BLUR, DOSE = 160.0, 0.30, 10.0, 30.0


_CASES = []

def _report(name, ok, detail):
    _CASES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail:
        print(f"        {line}")
    print()
    return ok


def _base():
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA)
    e = PR.dose_to_size(g, i, PITCH / 2, THRESH)
    return g, m, i * e


def _ler(g, prof, n_rows, dy, n_trials, seeds=5):
    vals = []
    for sd in range(seeds):
        r = ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH,
                           n_rows=n_rows, blur_nm=BLUR, n_trials=n_trials,
                           seed=sd, dy=dy)
        vals.append(r["ler_3sigma"])
    v = np.array(vals)
    return float(v.mean()), float(100 * v.std() / v.mean())


def case1_ler_convergence():
    """LER 은 반복 횟수가 아니라 y 측정 길이에 수렴한다 [6주차 검증]"""
    g, _, prof = _base()
    rows = ["(a) 반복 횟수 변화 (y 길이 128 nm 고정)"]
    trial_means = []
    for nt in (2, 8, 32):
        mu, cv = _ler(g, prof, 256, 0.5, nt)
        trial_means.append(mu)
        rows.append(f"    반복 {nt:3d}  ->  LER {mu:.4f} nm (변동 {cv:.1f} %)")
    rows.append("    반복을 16배 늘려도 LER 값 자체는 변하지 않음")

    rows.append("")
    rows.append("(b) y 측정 길이 변화 (반복 8 고정)")
    len_means = []
    for n_rows, dy in ((256, 0.5), (512, 0.5), (512, 2.0), (1024, 2.0), (2048, 2.0)):
        mu, cv = _ler(g, prof, n_rows, dy, 8)
        len_means.append(mu)
        rows.append(f"    y 길이 {n_rows * dy:6.0f} nm  ->  LER {mu:.4f} nm (변동 {cv:.1f} %)")

    bias = 100 * (len_means[-1] - len_means[0]) / len_means[-1]
    rows += ["",
             f"6주차 설정(y 길이 128 nm)은 수렴값 대비 {bias:.0f} % 과소평가였음.",
             "선 가장자리 거칠기는 장파장 성분을 포함하므로 측정 창이 짧으면",
             "저주파 성분이 누락됨. 측정 길이는 블러 길이의 100배 이상을 권장함."]
    trial_flat = (max(trial_means) - min(trial_means)) / np.mean(trial_means) < 0.15
    length_grows = len_means[-1] > len_means[0] * 1.1
    return _report("Case 1  LER 수렴성 : 반복 횟수 대 y 측정 길이",
                   trial_flat and length_grows, rows)


def case2_dy_decoupling():
    """dy 를 dx 와 분리하면 비용 증가 없이 측정 길이를 확보한다"""
    g, _, prof = _base()
    rows = ["  설정                      y 길이   LER      변동    소요[s]"]
    results = []
    for n_rows, dy in ((1024, 0.5), (1024, 2.0)):
        t0 = time.perf_counter()
        mu, cv = _ler(g, prof, n_rows, dy, 8, seeds=3)
        el = (time.perf_counter() - t0) / 3
        results.append((n_rows * dy, mu, cv, el))
        rows.append(f"  행 {n_rows}, dy {dy:.1f} nm      {n_rows * dy:6.0f}  {mu:.4f}  "
                    f"{cv:5.1f} %  {el:6.2f}")
    ratio_len = results[1][0] / results[0][0]
    ratio_time = results[1][3] / max(results[0][3], 1e-9)
    rows += ["",
             f"동일 연산 비용(비율 {ratio_time:.2f})으로 측정 길이를 {ratio_len:.0f} 배 확보함.",
             "dy 는 블러 길이의 1/3 이하로 두어야 블러 연산이 왜곡되지 않음."]
    return _report("Case 2  dy 분리에 의한 측정 길이 확보",
                   ratio_len >= 3.5 and ratio_time < 1.5, rows)


def case3_blur_sensitivity():
    """블러 파라미터가 LER 과 CD 에 미치는 영향"""
    g, _, prof = _base()
    rows = ["  블러[nm]    LER 3σ     CD 평균    CD 3σ"]
    lers, cds = [], []
    for blur in (2.0, 5.0, 10.0, 20.0):
        r = ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH,
                           n_rows=1024, blur_nm=blur, n_trials=8, seed=0, dy=2.0)
        lers.append(r["ler_3sigma"])
        cds.append(r["cd_mean"])
        rows.append(f"  {blur:7.0f}   {r['ler_3sigma']:8.4f}  {r['cd_mean']:9.2f}  "
                    f"{r['cd_3sigma']:8.4f}")
    span = lers[0] / lers[-1]
    cd_bias = cds[-1] - cds[0]
    rows += ["",
             f"블러 2 nm 대비 20 nm 에서 LER 이 {span:.1f} 배 감소하고 CD 는 "
             f"{cd_bias:+.2f} nm 이동함.",
             "블러는 LER 을 억제하는 동시에 패턴 경사를 완만하게 하여 CD 를 편향시킴.",
             "본 모델은 블러를 자유 파라미터로 두므로, 절대 LER 값이 아니라",
             "조건 간 상대 비교에 사용해야 함. 문헌값 대조는 차후 과제로 둠."]
    return _report("Case 3  레지스트 블러 민감도", span > 3.0 and abs(cd_bias) > 0.5, rows)


def case4_downsample_identifiability():
    """다운샘플이 식별 정보를 손상시키지 않음을 확인한다"""
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    iso = []
    for dz in np.arange(0.0, 261.0, 20.0):
        i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA,
                            defocus=float(dz))
        e = PR.dose_to_size(g, i, PITCH / 2, THRESH)
        if np.isfinite(e):
            iso.append(i * e)

    full = np.array([np.sqrt(np.mean((p - iso[0]) ** 2)) for p in iso])
    rows = ["  n_profile   1만건 용량[MB]   식별 정보 보존율"]
    keeps = []
    for n in (8, 16, 32, 64, 128):
        ds = [DS.downsample_profile(g.x, p, PITCH, n) for p in iso]
        d = np.array([np.sqrt(np.mean((q - ds[0]) ** 2)) for q in ds])
        keep = 100 * float(np.mean(d[1:] / full[1:]))
        keeps.append(keep)
        rows.append(f"  {n:9d}   {DS.estimate_size(10000, n)['total_mb']:13.1f}   "
                    f"{keep:14.1f} %")
    nyq = LAM / (4 * (1 + SIGMA) * NA)
    rows += ["",
             f"공간 광강도는 대역제한 신호이므로(Nyquist 간격 {nyq:.1f} nm) 재샘플링에",
             "의한 정보 손실이 발생하지 않음. pitch 160 nm 기준 8 점(20 nm 간격)에서도",
             "식별 정보가 완전히 보존됨.",
             "안전 여유를 두어 n_profile = 64 를 기본값으로 채택함(1만 건 6.5 MB)."]
    return _report("Case 4  프로파일 다운샘플의 식별 정보 보존",
                   all(k > 99.0 for k in keeps), rows)


def case5_schema_roundtrip():
    """데이터셋 저장 및 복원 무결성"""
    specs = [DS.SampleSpec(defocus=dz, dose_rel=dr, n_trials=4, n_rows=512, dy=2.0)
             for dz in (0.0, 80.0) for dr in (0.95, 1.05)]
    data = DS.generate(specs)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.npz")
        DS.save(path, data)
        back = DS.load(path)
        same = (np.allclose(data["conditions"], back["conditions"])
                and np.allclose(data["profiles"], back["profiles"])
                and np.allclose(data["labels"], back["labels"], equal_nan=True))
        size_kb = os.path.getsize(path) / 1024
    rows = [f"생성 {len(data['conditions'])} 건, 전 건 유효 = {bool(data['ok'].all())}",
            f"조건 {len(DS.CONDITION_KEYS)} 개 : {', '.join(DS.CONDITION_KEYS)}",
            f"라벨 {len(DS.LABEL_KEYS)} 개 : {', '.join(DS.LABEL_KEYS)}",
            f"프로파일 {data['profiles'].shape[1]} 점",
            f"저장 후 복원 일치 = {same}, 파일 크기 {size_kb:.1f} KB"]
    return _report("Case 5  데이터셋 스키마 저장·복원", same and bool(data["ok"].all()), rows)


def case6_generation_cost():
    """생성 비용 추산"""
    rows = ["  설정                  1건[s]   1만건 환산"]
    times = []
    for nt, nr in ((4, 512), (8, 1024)):
        sp = DS.SampleSpec(n_trials=nt, n_rows=nr, dy=2.0)
        t0 = time.perf_counter()
        DS.make_sample(sp)
        el = time.perf_counter() - t0
        times.append(el)
        rows.append(f"  반복 {nt}, 행 {nr:5d}        {el:5.2f}   {el * 10000 / 3600:6.1f} 시간")
    rows += ["",
             "9~10주차 대량 생성 시 단일 프로세스로는 수 시간 ~ 수십 시간이 소요됨.",
             "조건 공간 축소 또는 다중 프로세스 병렬화가 필요함."]
    return _report("Case 6  데이터 생성 비용 추산", all(t < 30 for t in times), rows)


def main() -> int:
    print("=" * 76)
    print(" 7주차 정량 검증 (확률 모델 수렴성 / 데이터셋 설계)")
    print("=" * 76 + "\n")
    results = [
        case1_ler_convergence(),
        case2_dy_decoupling(),
        case3_blur_sensitivity(),
        case4_downsample_identifiability(),
        case5_schema_roundtrip(),
        case6_generation_cost(),
    ]
    print("=" * 76)
    print(f" 결과: {sum(results)}/{len(results)} 통과")
    print()
    print(VC.summarize(_CASES))
    print("=" * 76)
    return 0 if all(results) else 1


# [11주차 수정] 실패해도 종료 코드 0 을 돌려주고 있었다. 10 주차에
# verify.py 에서 같은 결함을 고쳤는데(불변량이 깨져도 '위반' 만 인쇄하고
# 0 으로 끝남) 회귀 스크립트 일곱 개는 확인조차 하지 않았다. 사람이 출력을
# 읽으면 드러나지만, 종료 코드만 보는 자동 검증은 50 종이 전부 깨져도
# '통과' 로 읽는다. 실제로 패키지 검증에서 그렇게 놓쳤다.
if __name__ == "__main__":
    sys.exit(main())
