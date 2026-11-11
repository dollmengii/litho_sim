"""
inspect_dataset.py -- 생성된 데이터셋 건전성 검사

왜 필요한가
-----------
생성이 "완료" 로 끝났다고 데이터가 쓸 만하다는 뜻은 아니다. 라벨에 결측이
섞이거나, 특정 조합에만 표본이 쏠리거나, 조건 공간의 일부가 비어 있거나,
혼동 집합이 의도대로 만들어지지 않았을 수 있다. 이런 문제는 학습을 돌린
뒤에 발견하면 되돌리기 어렵다. 11주차 전에 한 번 걸러낸다.

실행:
    py inspect_dataset.py [npz 경로]
    (기본값 dataset/litho_dataset.npz)

결과는 화면과 dataset/inspection.txt 에 함께 기록한다.
"""

import os
import sys

import numpy as np

from litho import dataset as DS

OUT_TXT = os.path.join("dataset", "inspection.txt")
_lines = []
_warn = []


def say(s=""):
    print(s)
    _lines.append(s)


def warn(s):
    _warn.append(s)


def head(title):
    say()
    say("=" * 74)
    say(f" {title}")
    say("=" * 74)


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------- 1
def sec_structure(d):
    head("1. 기본 구조")
    cond, prof = d["conditions"], d["profiles"]
    lab, tgt = d["labels"], d["target_cd"]
    ck = [str(s) for s in d["condition_keys"]]
    lk = [str(s) for s in d["label_keys"]]

    say(f"  표본 수          {len(cond):,}")
    say(f"  조건 벡터        {cond.shape[1]} 열")
    say(f"  프로파일         {prof.shape[1]} 점")
    say(f"  라벨             {lab.shape[1]} 열")
    say(f"  파일 크기        {os.path.getsize(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)/1024**2:.1f} MB")

    ok = len({len(cond), len(prof), len(lab), len(tgt)}) == 1
    say(f"  길이 일치        {ok}")
    if not ok:
        warn("배열 길이가 서로 다릅니다. 저장 과정에 문제가 있습니다.")

    if ck != DS.CONDITION_KEYS:
        warn(f"조건 열 이름이 현재 코드와 다릅니다 (파일 {len(ck)} vs 코드 "
             f"{len(DS.CONDITION_KEYS)}). 다른 버전으로 생성된 데이터입니다.")
        say(f"  [주의] 조건 열 이름 불일치")
    if lk != DS.LABEL_KEYS:
        warn("라벨 열 이름이 현재 코드와 다릅니다.")
    return ck, lk


# ---------------------------------------------------------------- 2
def sec_missing(d, lk):
    head("2. 결측 및 이상값")
    lab = d["labels"]
    say(f"  {'라벨':<18} {'NaN':>8} {'inf':>8} {'<=0':>8}  {'최소':>12} {'최대':>12}")
    say("  " + "-" * 70)
    for j, name in enumerate(lk):
        col = lab[:, j]
        n_nan = int(np.isnan(col).sum())
        n_inf = int(np.isinf(col).sum())
        fin = col[np.isfinite(col)]
        n_np = int((fin <= 0).sum())
        lo = fin.min() if fin.size else np.nan
        hi = fin.max() if fin.size else np.nan
        say(f"  {name:<18} {n_nan:>8,} {n_inf:>8,} {n_np:>8,}  {lo:>12.4g} {hi:>12.4g}")
        if n_nan or n_inf:
            warn(f"라벨 '{name}' 에 결측/무한값 {n_nan + n_inf:,} 건")
        # 상수 라벨은 학습 목표로 쓸 수 없다. 상관 행렬에서 0 으로 나누어
        # nan 을 만들기도 한다(실제로 dose_clear_mj 가 그랬다). 조용히
        # 지나가면 모형이 '맞춘' 것처럼 보이는 죽은 열이 남는다.
        if fin.size and np.unique(np.round(fin, 9)).size == 1:
            warn(f"라벨 '{name}' 이 상수({lo:.6g})입니다. 정보량이 0 이므로 "
                 f"학습 목표에서 빼야 합니다. 상관 행렬에도 nan 으로 "
                 f"나타납니다.")
        # 거의 전부가 한 값인 라벨도 목표로 부적절하다. '항상 그 값' 이라고
        # 답하는 모형이 높은 정확도를 받는다(unresolved_row_frac 이 그렇다).
        elif fin.size:
            uniq, cnt = np.unique(np.round(fin, 9), return_counts=True)
            k = int(np.argmax(cnt))
            mode_frac = float(cnt[k] / fin.size)
            if mode_frac > 0.95:
                warn(f"라벨 '{name}' 의 {mode_frac:.1%} 가 같은 값"
                     f"({uniq[k]:.6g})입니다. '항상 그 값' 이라고 답하는 "
                     f"모형이 높은 정확도를 받으므로 학습 목표로 "
                     f"부적절합니다.")

    prof = d["profiles"]
    n_bad = int((~np.isfinite(prof)).sum())
    say()
    say(f"  프로파일 결측    {n_bad:,} 점")
    flat = int((prof.max(axis=1) - prof.min(axis=1) < 1e-9).sum())
    say(f"  평탄한 프로파일  {flat:,} 건  (변화가 없는 표본, 많으면 이상)")
    if n_bad:
        warn(f"프로파일에 결측 {n_bad:,} 점")
    if flat > len(prof) * 0.01:
        warn(f"평탄한 프로파일이 {100*flat/len(prof):.1f} % 입니다.")


# ---------------------------------------------------------------- 3
def sec_arms(d, ck):
    head("3. 이산 조합별 분포")
    cond = d["conditions"]
    idx = {k: ck.index(k) for k in ("pol_code", "source_code", "mask_code")
           if k in ck}
    if len(idx) < 3:
        say("  조건 벡터에 이산 부호가 없습니다 (구버전 데이터).")
        return
    pol = cond[:, idx["pol_code"]].astype(int)
    src = cond[:, idx["source_code"]].astype(int)
    msk = cond[:, idx["mask_code"]].astype(int)

    combos = {}
    for p, s, m in zip(pol, src, msk):
        combos[(p, s, m)] = combos.get((p, s, m), 0) + 1
    say(f"  {'편광':<13} {'조명':<14} {'마스크':<9} {'표본':>9} {'비율':>7}")
    say("  " + "-" * 58)
    for (p, s, m), n in sorted(combos.items(), key=lambda t: -t[1]):
        say(f"  {DS.POLARIZATION_CODES[p]:<13} {DS.SOURCE_CODES[s]:<14} "
            f"{DS.MASK_CODES[m]:<9} {n:>9,} {100*n/len(cond):>6.1f} %")

    if len(combos) > 1:
        lo = min(combos.values()); hi = max(combos.values())
        say()
        say(f"  최다/최소 비     {hi/lo:.2f} 배")
        if hi / lo > 4.0:
            warn(f"조합 간 표본 수가 {hi/lo:.1f}배 차이납니다. 비교 시 "
                 f"가중치를 고려하십시오.")


# ---------------------------------------------------------------- 4
def sec_coverage(d, ck):
    """조건 열 **전부**를 본다.

    [10주차 정정] 이전에는 9 개 축만 하드코딩해서 봤다. 조건 벡터는 24 열인데
    z5~z11, dose_mj, att_t, pole_offset 등 15 열이 보고에서 빠져 있었다. 그
    상태로 종합에 "이상 없음, 학습에 사용할 수 있습니다" 를 찍었다. 즉 조건의
    37 % 만 보고 내린 판정이었다. 빠진 열 가운데 상수 열이 있어도 드러나지
    않는다. 상수 열은 학습에 정보를 주지 않으므로 반드시 알아야 한다.
    """
    head("4. 조건 공간 덮기")
    cond = d["conditions"]
    # 보기 좋은 순서로 먼저 세우고, 나머지는 스키마 순서대로 모두 붙인다.
    pref = ["pitch", "duty", "na", "sigma_out", "defocus", "dose_rel",
            "flare", "msd_nm", "blur_nm"]
    axes = [a for a in pref if a in ck] + [a for a in ck if a not in pref]

    varying, const = [], []
    for a in axes:
        v = cond[:, ck.index(a)]
        nu = len(np.unique(np.round(v, 6)))
        (const if nu == 1 else varying).append((a, v, nu))

    say(f"  {'변수':<14} {'최소':>10} {'최대':>10} {'평균':>10} "
        f"{'고유값':>8}  분포")
    say("  " + "-" * 72)
    # [정정] 범주형 열(pol_code / source_code / mask_code)에 연속 축의 기준을
    # 적용하면 거짓 양성이 난다. 실제로 "고유값이 2 개뿐이라 사실상 고정된
    # 축", "하단 5 % 구간에 몰려 있다" 같은 경고가 5 건 떴는데, 이들은 분류
    # 코드이므로 고유값이 2~4 개인 것이 정상이고 '하단 구간' 이라는 개념이
    # 없다. 범주형은 분포만 보고하고, 균형은 3 절에서 따로 본다.
    CATEGORICAL = {"pol_code", "source_code", "mask_code"}
    for a, v, nu in varying:
        h, _ = np.histogram(v, bins=10)
        bar = "".join("▁▂▃▄▅▆▇█"[min(7, int(7 * c / max(h.max(), 1)))] for c in h)
        tag = " (범주형)" if a in CATEGORICAL else ""
        say(f"  {a:<14} {v.min():>10.4g} {v.max():>10.4g} {v.mean():>10.4g} "
            f"{nu:>8,}  {bar}{tag}")
        if a in CATEGORICAL:
            continue
        if nu < 5:
            warn(f"'{a}' 의 고유값이 {nu} 개뿐입니다. 사실상 고정된 축입니다.")
        # 분포가 한쪽 끝에 몰린 축은 그 방향의 학습 재료가 없다는 뜻이다.
        lo, hi = float(v.min()), float(v.max())
        if hi > lo:
            frac_low = float((v <= lo + 0.05 * (hi - lo)).mean())
            if frac_low > 0.5:
                warn(f"'{a}' 표본의 {frac_low:.0%} 가 하단 5 % 구간에 몰려 "
                     f"있습니다. 이 축 방향의 예측은 사실상 외삽입니다.")

    if const:
        say("")
        say(f"  [상수 열 {len(const)} 개] 값이 하나뿐이라 학습에 정보를 주지 "
            f"않습니다.")
        for a, v, _ in const:
            say(f"    {a:<14} = {v[0]:.6g}")
        warn(f"조건 열 {len(const)} 개가 상수입니다: "
             f"{', '.join(a for a, _, _ in const)}. 학습 입력에서 빼거나, "
             f"의도적으로 고정한 것이면 보고서에 '이 조건은 고정' 이라고 "
             f"적어야 합니다. 상수 열을 입력에 두면 모형이 그 축을 "
             f"'탐색했다' 고 오해하게 만듭니다.")
    say("")
    say(f"  변화하는 조건 {len(varying)} / 전체 {len(axes)} 열")


# ---------------------------------------------------------------- 5
def sec_split(d):
    head("5. 분할 건전성")
    if "split_train" not in d:
        say("  분할 정보가 없습니다 (--no-split 으로 생성됨).")
        warn("분할 정보가 없습니다. 학습 전에 splitting.grouped_split 로 "
             "직접 나누십시오.")
        return
    n = len(d["conditions"])
    parts = {k: d[f"split_{k}"] for k in ("train", "val", "test")}
    tot = sum(len(v) for v in parts.values())
    allidx = np.concatenate(list(parts.values()))
    dup = len(allidx) - len(np.unique(allidx))
    missing = n - len(np.unique(allidx))

    say(f"  {'분할':<10} {'표본':>10} {'비율':>8}")
    say("  " + "-" * 32)
    for k, v in parts.items():
        say(f"  {k:<10} {len(v):>10,} {100*len(v)/n:>7.1f} %")
    say()
    say(f"  합계             {tot:,} / {n:,}")
    say(f"  중복 인덱스      {dup}")
    say(f"  누락 표본        {missing}")
    if dup or missing:
        warn(f"분할에 중복 {dup} 또는 누락 {missing} 이 있습니다.")

    # 그룹 누수 : 같은 광학 조건이 두 분할에 걸쳐 있는가
    cond = d["conditions"]
    ck = [str(s) for s in d["condition_keys"]]
    keys = [k for k in ("wavelength", "na", "n_medium", "sigma_out", "sigma_in",
                        "pol_code", "source_code", "mask_code", "pitch", "duty")
            if k in ck]
    cols = [ck.index(k) for k in keys]
    gid = {}
    for name, v in parts.items():
        for i in v:
            g = tuple(np.round(cond[i, cols], 9))
            gid.setdefault(g, set()).add(name)
    leaked = sum(1 for s in gid.values() if len(s) > 1)
    say(f"  광학 조건 그룹   {len(gid):,}")
    say(f"  누수 그룹        {leaked}")
    if leaked:
        warn(f"그룹 누수 {leaked} 건. 성능이 과대평가됩니다.")


# ---------------------------------------------------------------- 6
def sec_confusion(d, ck, lk):
    head("6. 혼동 구조 (식별 한계의 직접 증거)")
    cond, lab = d["conditions"], d["labels"]
    need = ("pitch", "duty", "na", "sigma_out", "defocus")
    if not all(k in ck for k in need) or "cd_mean" not in lk:
        say("  필요한 열이 없습니다.")
        return
    cols = [ck.index(k) for k in ("pitch", "duty", "na", "sigma_out")]
    i_dz = ck.index("defocus")
    # 혼동 집합은 '결정론적' 선폭을 맞춰 만든다. 확률 선폭(cd_mean)은 광자
    # 잡음이 더해져 있어 맞춤 여부 판정에 쓸 수 없다. 두 값을 함께 보면
    # 잡음이 조건 간 차이보다 큰지도 바로 드러난다.
    j_det = lk.index("cd_deterministic")
    j_sto = lk.index("cd_mean") if "cd_mean" in lk else None

    key = np.round(cond[:, cols], 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    say(f"  광학·패턴 조건 그룹 {len(uniq):,} 개")

    rows = []
    for g in range(len(uniq)):
        m = inv == g
        if m.sum() < 4:
            continue
        det = lab[m, j_det]
        dz = cond[m, i_dz]
        fin = np.isfinite(det)
        if fin.sum() < 4:
            continue
        det, dz = det[fin], dz[fin]
        if det.mean() <= 0:
            continue
        rel = (det.max() - det.min()) / det.mean()
        sto = np.nan
        if j_sto is not None:
            s = lab[m, j_sto][fin]
            s = s[np.isfinite(s)]
            if s.size > 1 and s.mean() > 0:
                sto = (s.max() - s.min()) / s.mean()
        rows.append((rel, dz.max() - dz.min(), det.mean(), int(fin.sum()), sto))
    if not rows:
        say("  표본이 충분한 그룹이 없습니다.")
        return

    rows.sort()
    say()
    say("  결정론적 선폭은 같은데 초점은 크게 다른 그룹 (상위 8개)")
    say(f"  {'CD 산포(결정론)':>15} {'CD 산포(확률)':>14} {'Δz 범위[nm]':>13} "
        f"{'CD[nm]':>9} {'표본':>6}")
    say("  " + "-" * 64)
    for r in rows[:8]:
        s = "   -" if not np.isfinite(r[4]) else f"{100*r[4]:>11.2f} %"
        say(f"  {100*r[0]:>14.3f} % {s:>14} {r[1]:>13.1f} {r[2]:>9.2f} {r[3]:>6}")

    tight = [r for r in rows if r[0] < 0.01 and r[1] > 50]
    say()
    say(f"  CD 산포 1 % 미만이면서 Δz 범위 50 nm 초과인 그룹 : {len(tight):,} 개")
    if tight:
        say("  → 선폭만 측정해서는 이 그룹들의 초점을 구별할 수 없습니다.")
        say("     12주차 역추정 평가의 핵심 표본입니다.")
        amp = [r[4] / r[0] for r in tight
               if np.isfinite(r[4]) and r[0] > 1e-12]
        if amp:
            say()
            say(f"  이 그룹들에서 확률 선폭의 산포는 결정론 산포의 "
                f"{np.median(amp):.0f}배입니다.")
            say("  즉 조건 차이보다 측정 잡음이 더 큽니다. 한 번의 선폭 측정으로는")
            say("  원리적으로 구별이 불가능하다는 뜻입니다.")
    else:
        warn("혼동 구조가 뚜렷한 그룹이 없습니다. 혼동 집합이 "
             "splitting.confusion_set 의 보정을 거쳤는지 확인하십시오.")


# ---------------------------------------------------------------- 7
def sec_labels(d, lk):
    head("7. 라벨 상관 구조")
    lab = d["labels"]
    keep = [j for j, n in enumerate(lk)
            if np.isfinite(lab[:, j]).sum() > len(lab) * 0.5]
    m = np.all(np.isfinite(lab[:, keep]), axis=1)
    sub = lab[m][:, keep]
    names = [lk[j] for j in keep]
    if sub.shape[0] < 10:
        say("  유효 표본이 부족합니다.")
        return
    c = np.corrcoef(sub.T)
    say(f"  유효 표본 {sub.shape[0]:,} 건 기준 상관계수")
    say()
    say("  " + " " * 18 + "".join(f"{n[:9]:>10}" for n in names))
    for i, n in enumerate(names):
        say(f"  {n:<18}" + "".join(f"{c[i,j]:>10.3f}" for j in range(len(names))))
    say()
    # [10주차 정정이 이 파일에 전파되지 않았던 부분]
    # 여기에는 "LWR/LER 이 sqrt(2)=1.414 근처여야 한다" 고 적혀 있었다.
    # 그런데 이 과제가 10주차에 바로 그것을 오류로 지적하고 고쳤다
    # (VERIFICATION_LIMITS 항목 11). 정확한 항등식은
    #     Var(W) = Var(x_R) + Var(x_L) - 2 Cov(x_L, x_R)
    # 이므로 좌우 분산이 같다고 보면
    #     LWR = sqrt(2 - 2 rho) x LER
    # 이고 sqrt(2) 는 **rho = 0 인 극한**일 뿐이다. sqrt(2) 를 '이론값' 으로
    # 두면 실제 상관이 있을 때 정상인 결과를 '편차' 로 보고하게 된다.
    #
    # 또 하나. 두 라벨은 **서로 다른 추정량**이다.
    #   cd_3sigma  : 모든 시행의 선폭을 합쳐 표준편차 (pooled)
    #   ler_3sigma : 시행별 표준편차를 구해 평균 (trial-mean)
    # 직접 재 보면 둘 다 pooled 로 맞추면 이론과 0.03 % 로 일치하는데,
    # 혼합해서 쓰면 비가 약 1.9 % 높게 나온다. 따라서 이 비의 1~2 % 수준
    # 잔차는 물리가 아니라 추정량 규약에서 온다. 라벨 정의를 바꾸면 이미
    # 만든 데이터를 다시 뽑아야 하므로 정의는 그대로 두고, 여기서 그 사실을
    # 밝혀 '물리 검증' 으로 잘못 읽지 않게 한다.
    say("  LWR(cd_3sigma)/LER(ler_3sigma) 비 : 정확한 관계는")
    say("  LWR = sqrt(2 - 2 rho) x LER 이고 sqrt(2) 는 rho=0 극한이다.")
    if "cd_3sigma" in names and "ler_3sigma" in names:
        a = sub[:, names.index("cd_3sigma")]
        b = sub[:, names.index("ler_3sigma")]
        ok = (b > 1e-9)
        if ok.sum():
            r = float(np.median(a[ok] / b[ok]))
            rho_implied = 1.0 - 0.5 * r * r
            say(f"  관측 비 중앙값 {r:.4f}  ->  함의하는 rho = {rho_implied:+.4f}")
            say(f"  (sqrt(2) = 1.4142 는 rho = 0 일 때의 값)")
            say("  이 비의 1~2 % 잔차는 두 라벨의 추정량 규약 차이(pooled vs")
            say("  시행평균)에서 오므로, 물리 검증으로 읽으면 안 된다. 항등식의")
            say("  엄밀한 증거는 validate_w10 의 분산 항등식 시험(1.7e-16)이다.")
            # 물리적으로 가능한 상관 범위를 벗어나면 그때만 경고한다.
            if not (-1.0 <= rho_implied <= 1.0):
                warn(f"함의하는 상관 rho = {rho_implied:+.3f} 가 [-1, 1] 밖입니다. "
                     f"라벨 계산에 문제가 있습니다.")
            elif abs(rho_implied) > 0.5:
                warn(f"함의하는 좌우 경계 상관 rho = {rho_implied:+.3f} 가 "
                     f"이례적으로 큽니다. 조건을 확인하십시오.")


DEFAULT = os.path.join("dataset", "litho_dataset.npz")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print(f"파일이 없습니다: {path}")
        sys.exit(1)
    d = load(path)

    say("=" * 74)
    say(" 데이터셋 건전성 검사")
    say("=" * 74)
    say(f" 대상 : {path}")

    ck, lk = sec_structure(d)
    sec_missing(d, lk)
    sec_arms(d, ck)
    sec_coverage(d, ck)
    sec_split(d)
    sec_confusion(d, ck, lk)
    sec_labels(d, lk)

    head("종합")
    if _warn:
        say(f"  주의 {len(_warn)} 건")
        for w in _warn:
            say(f"   - {w}")
    else:
        say("  이상 없음. 11주차 학습에 사용할 수 있습니다.")

    os.makedirs("dataset", exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\n기록: {OUT_TXT}")
