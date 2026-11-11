"""
validate_w8.py -- 8주차 정량 검증 (GUI 통합 / 배포 준비)

실행: py validate_w8.py
"""

import os
import re

import sys

import numpy as np

import validate_common as VC

from litho import dataset as DD
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


def case1_gui_correction():
    """GUI 가 7주차 보정 기준을 반영했는지 확인한다 [7주차 검증]"""
    src = open("app.py", encoding="utf-8").read()
    checks = {
        "dy 분리 반영": "dy=dy" in src or "dy = max(" in src,
        "y 측정 길이 입력": "y 측정 길이" in src,
        "짧은 측정 길이 경고": "LER 이 과소평가" in src,
        "dataset 모듈 연동": "from litho import dataset" in src,
        "측정 조건 표시": "y_length_nm" in src,
    }
    rows = [f"{k:<20s} {'OK' if v else '누락'}" for k, v in checks.items()]

    # 보정 전후 LER 차이 재확인
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA)
    prof = i * PR.dose_to_size(g, i, PITCH / 2, THRESH)
    old = ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH,
                         n_rows=256, blur_nm=BLUR, n_trials=8, seed=0, dy=0.5)
    new = ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH,
                         n_rows=614, blur_nm=BLUR, n_trials=8, seed=0, dy=BLUR / 3)
    gap = 100 * (new["ler_3sigma"] - old["ler_3sigma"]) / new["ler_3sigma"]
    rows += ["",
             f"보정 전 (y {old['y_length_nm']:.0f} nm) LER = {old['ler_3sigma']:.4f} nm",
             f"보정 후 (y {new['y_length_nm']:.0f} nm) LER = {new['ler_3sigma']:.4f} nm",
             f"-> {gap:.0f} % 차이. GUI 기본값을 y 2048 nm 로 변경함."]
    return _report("Case 1  GUI 의 7주차 보정 반영", all(checks.values()) and gap > 10, rows)


def case2_dataset_integration():
    """GUI 데이터셋 생성 경로가 동작하는지 확인한다"""
    specs = []
    for dz in np.linspace(-200.0, 200.0, 3):
        for dr in np.linspace(0.85, 1.15, 3):
            specs.append(DD.SampleSpec(
                defocus=float(dz), dose_rel=float(dr),
                n_trials=2, n_rows=614, dy=BLUR / 3, n_profile=64))
    data_all = DD.generate(specs, keep_invalid=True)
    data = DD.generate(specs)
    n = len(data["conditions"])
    idx_cd = DD.LABEL_KEYS.index("cd_mean")
    idx_dz = DD.CONDITION_KEYS.index("defocus")
    cds = data["labels"][:, idx_cd]
    spread = float(np.nanmax(cds) - np.nanmin(cds))
    cd_all = data_all["labels"][:, idx_cd]
    rows = [f"조건 {len(specs)} 건 중 유효 {n} 건 "
            f"(유효율 {100 * data_all['ok'].mean():.0f} %)",
            f"CD 상한(pitch {PITCH:.0f} nm) 검사 적용 후 최대 CD = "
            f"{np.nanmax(cd_all):.2f} nm",
            f"조건 배열 {data['conditions'].shape}, 프로파일 {data['profiles'].shape}, "
            f"라벨 {data['labels'].shape}",
            f"Defocus 범위 {data['conditions'][:, idx_dz].min():.0f} ~ "
            f"{data['conditions'][:, idx_dz].max():.0f} nm",
            f"CD 범위 {np.nanmin(cds):.2f} ~ {np.nanmax(cds):.2f} nm (폭 {spread:.2f} nm)"]
    ok = (n >= 2 and data["profiles"].shape[1] == 64 and spread > 1.0
          and np.nanmax(cd_all) < PITCH)
    return _report("Case 2  데이터셋 생성 경로 통합", ok, rows)


def case3_deployment_files():
    """배포에 필요한 파일이 갖추어졌는지 확인한다"""
    required = {
        "requirements.txt": "의존 패키지 목록",
        "README.md": "저장소 설명 및 한계 명시",
        ".gitignore": "가상환경·캐시 제외",
        "LICENSE": "MIT 라이선스",
        ".streamlit/config.toml": "테마 및 서버 설정",
        "app.py": "진입점",
    }
    rows, missing = [], []
    for f, desc in required.items():
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        rows.append(f"  {f:<24s} {'OK' if exists else '누락':<4s} {size:>7,} B   {desc}")
        if not exists:
            missing.append(f)

    # BOM 및 공백 제거 후 파싱 (Windows 저장 시 UTF-8 BOM 이 붙을 수 있음)
    req = open("requirements.txt", encoding="utf-8-sig").read()
    pkgs = []
    for line in req.splitlines():
        name = line.split("#")[0].strip().lstrip("\ufeff")
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            name = name.split(sep)[0]
        name = name.strip()
        if name:
            pkgs.append(name)
    rows += ["", f"선언 패키지: {', '.join(pkgs)}"]

    # 소스에서 실제로 쓰는 외부 패키지가 전부 선언되었는지
    #
    # [11주차 수정 두 가지]
    # (1) 예전에는 import 이름을 requirements 의 이름과 **그대로** 비교했다.
    #     그런데 배포명과 import 명이 다른 패키지가 있다. scikit-learn 을
    #     추가했을 때 실제로 'sklearn 미선언' 으로 오탐이 났다. 설치된
    #     환경의 메타데이터(importlib.metadata.packages_distributions)로
    #     import 명 -> 배포명을 얻고, 설치돼 있지 않으면 문서화된 별칭표로
    #     되돌린다.
    # (2) 예전에는 알려진 외부 패키지 **고정 목록**과의 교집합만 보았다.
    #     그러면 새 의존성을 추가해도 목록에 없어서 검사를 아예 받지 않는다.
    #     즉 '미선언 없음' 이 검사했다는 뜻이 아니었다. 이제 표준 라이브러리
    #     (sys.stdlib_module_names)와 프로젝트 자체 모듈을 빼는 방식으로
    #     바꾸어, 남는 것은 모두 외부 패키지로 본다.
    import sys
    from importlib import metadata as _md

    #: import 명과 배포명이 다른 것들. 메타데이터를 못 읽을 때만 쓴다.
    ALIAS = {"sklearn": "scikit-learn", "cv2": "opencv-python",
             "PIL": "pillow", "yaml": "pyyaml", "dateutil": "python-dateutil",
             "serial": "pyserial", "Crypto": "pycryptodome"}

    local_mods = {"litho"}
    used = set()
    for root, _, files in os.walk("."):
        if any(p in root for p in ("venv", "__pycache__", ".git")):
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            if root in (".", "./"):
                local_mods.add(fn[:-3])
            src = open(os.path.join(root, fn), encoding="utf-8-sig").read()
            for a, b in re.findall(r"^\s*import (\w+)|^\s*from (\w+)", src,
                                   re.M):
                used.add(a or b)

    std = set(getattr(sys, "stdlib_module_names", ()))
    external = {m for m in used
                if m not in std and m not in local_mods
                and not m.startswith("_")}

    def _norm(s: str) -> str:
        return s.lower().replace("_", "-")

    declared = {_norm(x) for x in pkgs}

    try:
        pd_map = _md.packages_distributions()
    except Exception:                                    # noqa: BLE001
        pd_map = {}

    undeclared, resolved = [], []
    for m in sorted(external):
        dists = pd_map.get(m) or ([ALIAS[m]] if m in ALIAS else [m])
        if any(_norm(dd) in declared for dd in dists):
            if _norm(m) not in declared:
                resolved.append(f"{m}->{dists[0]}")
            continue
        undeclared.append(m)

    rows.append(f"소스 사용 외부 패키지: {', '.join(sorted(external))}")
    if resolved:
        rows.append(f"import 명과 배포명이 다른 것: {', '.join(resolved)}")
    rows.append(f"미선언 패키지: {', '.join(undeclared) if undeclared else '없음'}")
    rows.append(f"판정 방식: 표준 라이브러리 {len(std)} 개와 프로젝트 모듈 "
                f"{len(local_mods)} 개를 뺀 나머지 전부를 검사함 "
                f"(고정 목록이 아님)")
    return _report("Case 3  배포 파일 구성", not missing and not undeclared, rows)


def case4_cold_start():
    """배포 환경의 초기 응답 비용을 추산한다"""
    import time
    t0 = time.perf_counter()
    g = G.build_grid(pitch=PITCH, n_periods=6, dx_target=0.5)
    m = MK.line_space(g, duty=0.5)
    i = IM.aerial_image(g, m, LAM, NA, n_medium=N_MED, sigma_out=SIGMA)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    prof = i * PR.dose_to_size(g, i, PITCH / 2, THRESH)
    t_dts = time.perf_counter() - t0

    t0 = time.perf_counter()
    ST.monte_carlo(g.x, prof, g.dx, DOSE, LAM, THRESH, n_rows=614,
                   blur_nm=BLUR, n_trials=8, seed=0, dy=BLUR / 3)
    t_mc = time.perf_counter() - t0

    rows = [f"초기 화면 결상 연산      {t_first * 1000:7.1f} ms",
            f"dose-to-size 보정        {t_dts * 1000:7.1f} ms",
            f"확률 모델 (반복 8)       {t_mc * 1000:7.1f} ms",
            "",
            "초기 화면은 결상 1회만 수행하므로 즉시 응답함.",
            "공정 윈도와 확률 모델은 버튼 실행 방식이며 결과를 캐싱하여,",
            "무료 호스팅 환경의 자원 제약에서도 초기 접속이 지연되지 않도록 함."]
    return _report("Case 4  배포 환경 초기 응답 비용", t_first < 1.0, rows)


def main() -> int:
    print("=" * 76)
    print(" 8주차 정량 검증 (GUI 통합 / 배포 준비)")
    print("=" * 76 + "\n")
    results = [
        case1_gui_correction(),
        case2_dataset_integration(),
        case3_deployment_files(),
        case4_cold_start(),
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
