"""
merge_dataset.py -- 데이터셋 병합 및 분할 재계산

쓰임
----
이미 만든 데이터셋에 추가 생성분을 덧붙일 때 쓴다. 단순히 배열을 이어
붙이는 것만으로는 안 되고, 학습/검증/시험 분할을 **합친 전체에 대해 다시**
계산해야 한다. 각 파일의 분할을 그대로 두고 합치면 같은 광학 조건이 한쪽
파일에서는 학습, 다른 파일에서는 검증으로 들어가 그룹 누수가 생긴다.

조건 벡터에서 광학 조건을 복원해 그룹을 만들고, 그룹 단위로 다시 나눈다.

실행:
    py merge_dataset.py 출력.npz 입력1.npz 입력2.npz [...]

예:
    py merge_dataset.py dataset/merged.npz ^
        dataset/litho_dataset.npz dataset/confusion.npz
"""

import os
import sys

import numpy as np

#: 광학 조건 그룹을 정의하는 열. splitting.OPTICAL_KEYS 와 같은 의미이되,
#: 저장된 조건 벡터에서 직접 읽는다.
GROUP_KEYS = ("wavelength", "na", "n_medium", "sigma_out", "sigma_in",
              "pol_code", "source_code", "mask_code", "pitch", "duty")

ARRAY_KEYS = ("conditions", "profiles", "labels", "target_cd")


def load_all(paths):
    out, meta = [], None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        ck = [str(s) for s in d["condition_keys"]]
        lk = [str(s) for s in d["label_keys"]]
        if meta is None:
            meta = (ck, lk, d["profiles"].shape[1])
        else:
            if ck != meta[0]:
                raise SystemExit(f"{p}: 조건 열 구성이 다릅니다. 병합 불가.")
            if lk != meta[1]:
                raise SystemExit(f"{p}: 라벨 열 구성이 다릅니다. 병합 불가.")
            if d["profiles"].shape[1] != meta[2]:
                raise SystemExit(f"{p}: 프로파일 길이가 다릅니다 "
                                 f"({d['profiles'].shape[1]} vs {meta[2]}).")
        out.append({k: d[k] for k in d.files})
        print(f"  {p:<44} {len(d['conditions']):>8,} 건")
    return out, meta


def grouped_split(cond, ck, fractions=(0.70, 0.15, 0.15), seed=0):
    cols = [ck.index(k) for k in GROUP_KEYS if k in ck]
    groups = {}
    for i in range(len(cond)):
        groups.setdefault(tuple(np.round(cond[i, cols], 9)), []).append(i)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    n = len(keys)
    n_tr = int(round(n * fractions[0]))
    n_va = int(round(n * fractions[1]))
    parts = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va],
             "test": keys[n_tr + n_va:]}
    return ({k: np.array(sorted(i for g in v for i in groups[g]), dtype=np.int64)
             for k, v in parts.items()}, n)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    out_path, inputs = sys.argv[1], sys.argv[2:]
    for p in inputs:
        if not os.path.exists(p):
            sys.exit(f"파일이 없습니다: {p}")

    print("=" * 70)
    print(" 데이터셋 병합")
    print("=" * 70)
    sets, (ck, lk, n_prof) = load_all(inputs)

    merged = {k: np.concatenate([s[k] for s in sets], axis=0)
              for k in ARRAY_KEYS}
    n = len(merged["conditions"])
    print(f"  {'합계':<44} {n:>8,} 건")

    # 완전 중복 표본 제거 (같은 조건에서 같은 seed 로 두 번 생성된 경우)
    _, uniq = np.unique(merged["conditions"], axis=0, return_index=True)
    if len(uniq) < n:
        keep = np.sort(uniq)
        merged = {k: v[keep] for k, v in merged.items()}
        print(f"  중복 조건 {n - len(keep):,} 건 제거 → {len(keep):,} 건")
        n = len(keep)

    print("\n분할 재계산 (광학 조건 단위)")
    split, n_groups = grouped_split(merged["conditions"], ck)
    tot = sum(len(v) for v in split.values())
    for k, v in split.items():
        print(f"  {k:<8} {len(v):>8,} ({100*len(v)/n:>5.1f} %)")
    print(f"  광학 조건 그룹 {n_groups:,}")
    assert tot == n and len(np.unique(np.concatenate(list(split.values())))) == n

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        condition_keys=np.array(ck), label_keys=np.array(lk),
        **merged,
        **{f"split_{k}": v for k, v in split.items()},
    )
    print(f"\n저장: {out_path}  ({os.path.getsize(out_path)/1024**2:.1f} MB)")
    print("\n다음: py inspect_dataset.py " + out_path)


if __name__ == "__main__":
    main()
