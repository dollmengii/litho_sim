"""
generate_dataset.py -- 대량 학습 데이터셋 생성 (10주차)

실행:
    py generate_dataset.py                 # 기본 5만 건 목표
    py generate_dataset.py --target 10000  # 규모 지정
    py generate_dataset.py --workers 4     # 워커 수 지정
    py generate_dataset.py --resume        # 중단 지점부터 재개

주의:
    Windows 에서 다중 프로세스를 사용하므로 반드시 스크립트로 실행해야 한다.
    대화형 인터프리터나 Streamlit 안에서는 동작하지 않는다.

출력:
    dataset/litho_dataset.npz   학습 데이터
    dataset/summary.txt         생성 요약
    dataset/checkpoint.pkl      재개용 (완료 후 삭제 가능)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# NumPy 내부 스레드와 프로세스 풀의 코어 경합을 방지한다.
# import numpy 보다 먼저 설정해야 효과가 있다.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from litho import campaign as CP  # noqa: E402
from litho import dataset as DS  # noqa: E402
from litho import splitting as SPL  # noqa: E402

OUT_DIR = "dataset"
NPZ_PATH = os.path.join(OUT_DIR, "litho_dataset.npz")
CKPT_PATH = os.path.join(OUT_DIR, "checkpoint.pkl")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.txt")

#: 조건 공간. 각 축의 (하한, 상한).
#:
#: [11주차 보강] 이전에는 광학 4 축만 있었다. 그래서 하이퍼큐브 표본은
#: flare=0, msd_nm=0, blur_nm=8, z5~z11=0 **기본값 고정**으로 생성되고,
#: 이 변수들은 혼동 집합에서만 흔들렸다. 10 만 건 데이터셋에서 실제로
#: flare 평균이 0.0035(상한 0.05), msd 평균 0.417(상한 6)로 약 86 % 가
#: 0 근방에 몰렸고, Zernike 5 개는 **전 표본 0** 이었다.
#:
#: 문제는 이 과제의 주장이 바로 그 축에 걸려 있다는 점이다.
#:   - 주장 1 : 플레어가 결정적 보상 변수다           -> flare 필요
#:   - 주장 3 : MSD 와 블러가 잡음으로 분리된다       -> msd_nm, blur_nm 필요
#:   - 주장 2 : 초점 부호는 Z6·Z11 로만 깨진다        -> z6, z11 필요
#: 학습 데이터에 그 축이 없으면 "주장을 모형으로 보일 수 있는가" 에 답할 수
#: 없다. 그래서 다섯 축을 추가한다.
#:
#: [왜 z5·z7·z8 은 빼는가] 축을 늘리면 필요한 표본 수가 2^d 로 늘어난다.
#: 9 축에서 축당 간격을 절반으로 줄이려면 512 배(약 277 일)가 필요하다.
#: 그래서 '주장에 필요한 축' 만 넣는다. verify.py C5 의 전수 분류 결과
#: 초점 부호 모호성을 깨는 것은 rho^2 성분을 품은 Z6 과 Z11 뿐이고,
#: Z5·Z7·Z8 은 짝함수성을 유지하므로 **부호 처방과는** 무관하다.
#:
#: [주의 - 작아서 빼는 것이 아니다] Z5·Z7·Z8 이 결상에 주는 영향은 작지
#: 않다. 같은 0.10 waves 에서 상대 변화가 Z8 3.89e-01, Z5 1.81e-01,
#: Z7 1.18e-01 로 Z6(1.10e-01)과 같은 수준이거나 더 크다. 특히 1 차원 선
#: 패턴이라 코마 y(Z7)는 효과가 없을 것 같지만, 광원이 확장돼 있으면
#: 광원점마다 차수가 sy 방향으로 이동해 방위각이 0 이 아니게 되므로 효과가
#: 있다. 즉 이 세 축을 0 으로 고정하는 것은 **표본 수를 아끼기 위한 범위
#: 제한**이며, 보고서에 '고정 조건' 으로 명시해야 한다. 결상이 이 수차에
#: 둔감하다는 뜻으로 읽히면 안 된다.
#:
#: [Zernike 범위 근거] 0~0.05 waves. 실제 스캐너의 잔류 수차 수준이고,
#: C5 에서 0.05 waves 가 이미 짝함수성을 유의하게 깬다(e > 1e-6). 즉
#: '수차 없음' 부터 '부호가 구별되기 시작' 까지를 덮는다.
SPACE = {
    "pitch": (100.0, 400.0),      # nm
    "duty": (0.30, 0.70),
    "na": (0.90, 1.35),
    "sigma_out": (0.40, 0.90),
    "flare": (0.0, 0.05),         # 투영계 플레어 비율
    "msd_nm": (0.0, 6.0),         # 스캔 동기화 오차 [nm]
    "blur_nm": (5.0, 15.0),       # 레지스트 블러(PEB 확산 길이) [nm]
    "z6": (0.0, 0.05),            # 비점수차 0도 [waves]
    # [13주차 보강] z11 만 음수로 연다. 두 축을 다 열면 같은 등가류를 두 번
    # 뽑는다. CD(+dz,+z6,+z11) = CD(-dz,-z6,-z11) 가 기계 정밀도로 성립하므로
    # 수차 부호 공간은 '같은 부호' 와 '섞인 부호' 두 등가류로 갈린다. 양수만
    # 뽑으면 '섞인 부호' 장비에 닿지 못해 추천기가 거부해야 하는데, z11 만
    # 열면 (+,-) 가 데이터에 들어오고 거울로 (-,+) 도 덮인다. 자세한 근거는
    # 보강명세_동결.md 1-A 절.
    "z11": (-0.05, 0.05),         # 구면수차 [waves]
}

#: 광학 조합당 Defocus x Dose 격자 크기
N_DEFOCUS, N_DOSE = 9, 9

#: 예상 적격률 (9주차 측정 기준). 목표 건수 역산에 사용한다.
EXPECTED_YIELD = 0.71


#: 그룹 안에서 흔드는 축. 수차만이다. 다른 축을 함께 흔들면 '광학 조건
#: 그룹' 의 뜻이 흐려져 누설 검사와 영역별 분해가 성립하지 않는다.
JITTER_AXES = ("z6", "z11")


def jitter_aberrations(n_groups: int, k: int, space: dict,
                       seed: int) -> list[dict]:
    """광학점 하나당 수차를 k 개로 흔든 조건 목록을 만든다.

    왜 필요한가 (13주차 보강 1-B)
    -----------------------------
    분할은 광학 조건 그룹 단위다(누설 방지). 수차가 그룹 식별자에 들어가므로,
    수차를 그룹 **안에서** 흔들지 않으면 모형은 '같은 광학 조건에서 수차만
    변할 때의 CD 변화' 를 한 번도 보지 않는다. 추천기는 측정된 수차를 입력으로
    받아 그 점에서 예측하므로 정확히 그 방향의 민감도가 필요하다.

    부수 효과로 영역별 분해가 복구된다. 11 주차 학습 보고에 "z6·z11 은 조건
    묶음마다 한 값만 갖는 축이라 '상위 25 %' 가 특정 그룹 집합과 거의
    같아진다" 는 주의가 붙어 있었고, 실제로 z6 상위 25 % 가 그룹 31 개뿐이라
    그 숫자를 축의 영향으로 읽을 수 없었다. 그룹 안에서 흔들면 상위 25 % 가
    여러 그룹에 걸쳐 그 진단이 뜻을 갖는다.

    어떻게 흔드는가
    ---------------
    수차 축만 **그룹별 독립 라틴 하이퍼큐브**로 뽑는다. 즉 (z6, z11) 평면을
    k 등분해 그룹마다 따로 섞는다. 다른 축은 그룹의 값을 그대로 k 번 쓴다.

    동결 명세(1-B)는 '그 광학점의 좌표를 중심으로 대칭으로' 라고 적었는데
    **바꿨다.** 사유는 구조적이다.
      - 중심 대칭은 경계에 붙은 그룹에서 폭이 0 으로 줄어 흔들리지 않는다.
        z6 범위가 (0, 0.05) 이므로 0 근방 그룹이 실제로 그렇게 된다.
      - 그룹 안의 변화 폭이 좁으면 위에 적은 '영역별 분해 복구' 가 되지
        않는다. 상위 25 % 가 여전히 특정 그룹 집합에 몰린다.
    전 구간 라틴 하이퍼큐브는 두 문제가 다 없고, 축 전체 덮개도 유지한다.
    이 변경은 되돌림 결과를 보기 전에 결정했으며 보강명세_동결.md 4 절에
    기록했다.
    """
    if k < 2:
        raise ValueError("k 는 2 이상이어야 한다 (1 이면 흔들지 않는 것이다)")
    base = CP.sample_optical_conditions(n_groups, space, seed)
    rng = np.random.default_rng(seed + 991)
    axes = [a for a in JITTER_AXES if a in space]
    out = []
    for g, row in enumerate(base):
        # 축마다 k 등분 구간에서 하나씩 뽑고 **그룹마다 따로** 섞는다.
        draws = {}
        for a in axes:
            lo, hi = space[a]
            edges = np.linspace(lo, hi, k + 1)
            pts = edges[:-1] + rng.random(k) * np.diff(edges)
            draws[a] = rng.permutation(pts)
        for i in range(k):
            c = dict(row)
            for a in axes:
                c[a] = float(draws[a][i])
            out.append(c)
    return out


def _fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}시간 {m:02d}분 {s:02d}초" if h else f"{m:d}분 {s:02d}초"


class Progress:
    """진행률과 예상 잔여 시간을 한 줄로 표시한다."""

    def __init__(self, label: str, every: float = 2.0):
        self.label, self.every = label, every
        self.t0 = time.perf_counter()
        self.last = 0.0

    def __call__(self, done: int, total: int):
        now = time.perf_counter()
        if now - self.last < self.every and done < total:
            return
        self.last = now
        el = now - self.t0
        eta = el / done * (total - done) if done else 0.0
        pct = 100.0 * done / total
        sys.stdout.write(
            f"\r  {self.label} {done:,}/{total:,} ({pct:5.1f} %) "
            f"경과 {_fmt(el)} · 잔여 {_fmt(eta)}    ")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")


def _confusion_one(task):
    """혼동 집합 한 묶음. 프로세스 풀에서 호출되므로 최상위에 둔다.

    윈도우는 프로세스를 spawn 으로 띄우므로 피클 가능한 최상위 함수여야 한다.
    """
    base_dict, over, seed = task
    try:
        return SPL.confusion_set(DS.SampleSpec(**{**base_dict, **over}),
                                 n=16, seed=seed)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100000, help="목표 적격 샘플 수")
    ap.add_argument("--confusion", type=float, default=0.25,
                    help="혼동 집합에 배분할 비율. 식별성 평가용으로, 관측 "
                         "선폭이 같아지도록 Dose 를 맞춘 조건 묶음을 만든다")
    ap.add_argument("--no-split", action="store_true",
                    help="학습/검증 분할 인덱스를 저장하지 않는다")
    ap.add_argument("--confusion-only", action="store_true",
                    help="혼동 집합만 생성한다. 기존 데이터셋에 덧붙일 때 쓴다. "
                         "--target 은 혼동 표본 목표치로 해석된다")
    ap.add_argument("--out", default=None,
                    help="저장 경로. 기본 dataset/litho_dataset.npz")
    ap.add_argument("--workers", type=int, default=None, help="워커 수 (기본 CPU-1)")
    ap.add_argument("--trials", type=int, default=4, help="샘플당 Monte Carlo 반복")
    ap.add_argument("--y-length", type=float, default=2048.0, help="y 측정 길이 [nm]")
    ap.add_argument("--blur", type=float, default=8.0,
                    help="레지스트 블러 [nm]. 고정값. --sweep-blur 를 주면 "
                         "문헌 범위 안에서 흔든다")
    ap.add_argument("--pol", default="TE",
                    choices=["TE", "TM", "unpolarized", "scalar"],
                    help="입사 편광. 고NA 조밀 패턴의 실사용 조건은 TE")
    ap.add_argument("--mask", default="binary",
                    choices=["binary", "att-psm"], help="마스크 종류")
    ap.add_argument("--source", default="conventional",
                    choices=["conventional", "annular", "dipole", "quadrupole"],
                    help="조명 형상")
    ap.add_argument("--arms", action="store_true",
                    help="campaign.DISCRETE_ARMS 의 이산 조합을 모두 생성 "
                         "(목표 수를 조합 수로 나눠 배분)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aberr-jitter", type=int, default=1, metavar="K",
                    help="광학점 하나당 수차(z6, z11) 를 K 개로 흔든다. "
                         "표본 수를 유지하도록 광학점 수를 K 로 나눈다. "
                         "1 이면 끄기(13주차 보강 전 동작)")
    ap.add_argument("--resume", action="store_true", help="체크포인트에서 재개")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    n_cpu = os.cpu_count() or 1
    workers = args.workers or max(1, n_cpu - 1)

    dy = max(0.25, args.blur / 3.0)
    base = DS.SampleSpec(
        n_trials=args.trials, blur_nm=args.blur, dy=dy,
        n_rows=max(64, int(round(args.y_length / dy))),
        n_profile=64,
        polarization=args.pol, mask_type=args.mask, source_shape=args.source,
        pole_offset=0.75 if args.source in ("dipole", "quadrupole") else 0.0,
    )
    arms = CP.DISCRETE_ARMS if args.arms else [{}]

    # 목표 적격 수에서 필요한 광학 조합 수를 역산
    per_optical = N_DEFOCUS * N_DOSE
    n_optical = max(1, int(np.ceil(args.target / (per_optical * EXPECTED_YIELD))))

    print("=" * 72)
    print(" 포토리소그래피 학습 데이터셋 생성")
    print("=" * 72)
    print(f"  목표 적격 샘플   {args.target:,} 건")
    print(f"  광학 조건 조합   {n_optical:,} 개 (라틴 하이퍼큐브)")
    print(f"  조합당 격자      {N_DEFOCUS} x {N_DOSE} = {per_optical} 건")
    print(f"  워커             {workers} / 논리 CPU {n_cpu}")
    print(f"  확률 모델        반복 {args.trials}, y 길이 {args.y_length:.0f} nm, "
          f"블러 {args.blur:.0f} nm")
    if args.arms:
        print(f"  이산 조합        {len(arms)} 종 (편광 x 마스크 x 조명)")
        for a in arms:
            print("                   - " + ", ".join(f"{k}={v}" for k, v in a.items()))
    else:
        print(f"  고정 조건        편광 {args.pol}, 마스크 {args.mask}, "
              f"조명 {args.source}")
    print("  조건 공간        " + ", ".join(
        f"{k} {v[0]:g}~{v[1]:g}" for k, v in SPACE.items()))
    print("=" * 72)

    # 1단계 : 조건 공간 사전 탐색 및 계획 수립
    print("\n[1/3] 조건 공간 사전 탐색 (결정론 연산)")
    specs, screen_s, n_skipped, n_refined = [], 0.0, 0, 0
    if args.confusion_only:
        args.confusion = 1.0
        n_optical = 0
    else:
        n_optical = max(1, int(round(n_optical * (1.0 - args.confusion))))
    per_arm = max(1, n_optical // max(len(arms), 1))
    for ai, arm in enumerate([] if args.confusion_only else arms):
        arm_base = DS.SampleSpec(**{**base.__dict__, **arm})
        if arm:
            print(f"  조합 {ai+1}/{len(arms)} : "
                  + ", ".join(f"{k}={v}" for k, v in arm.items()))
        conds = None
        n_this = per_arm
        if args.aberr_jitter > 1:
            n_this = max(1, per_arm // args.aberr_jitter)
            conds = jitter_aberrations(n_this, args.aberr_jitter,
                                       SPACE, args.seed + ai)
        plan = CP.plan_campaign(
            arm_base, n_optical=n_this, n_defocus=N_DEFOCUS, n_dose=N_DOSE,
            space=SPACE, n_coarse=7, seed=args.seed + ai, n_workers=workers,
            progress=Progress("조합"), conditions=conds,
        )
        specs += plan["specs"]
        screen_s += plan["screen_s"]
        n_skipped += plan["n_skipped"]
        n_refined += plan["n_refined"]
    n_lhc = len(specs)
    print(f"  소요 {_fmt(screen_s)} · 생성 대상 {n_lhc:,} 건")
    print(f"  적격 영역 없음 {n_skipped} 조합 · 경계 세밀화 {n_refined} 조합")

    # 1-b단계 : 혼동 집합
    # 라틴 하이퍼큐브는 조건 공간을 고르게 덮지만, '관측량은 같은데 조건이
    # 다른' 묶음은 우연히만 만들어진다. 식별 한계를 재려면 그런 묶음이
    # 충분히 있어야 하므로 의도적으로 구성한다.
    if args.confusion > 0:
        n_conf = int(round(args.target * args.confusion / EXPECTED_YIELD))
        n_sets = max(1, n_conf // 16)
        print(f"\n[1-b] 혼동 집합 구성 ({n_sets:,} 묶음 x 16 건 목표)")
        # 묶음끼리 독립이므로 병렬로 만든다. 직렬로 두면 10만 건 규모에서
        # 이 단계만 9시간 가까이 걸려 전체의 병목이 된다.
        rng = np.random.default_rng(args.seed + 7777)
        tasks = []
        # 기준 조건은 SPACE 에서 일반적으로 뽑는다. 네 축만 하드코딩해 두면
        # SPACE 에 축을 추가해도 혼동 집합은 기본값(0)에 머문다. 실제로
        # z6·z11 을 추가했을 때 혼동 집합만 수차 0 으로 남았다.
        # confusion_set 이 스스로 흔드는 축은 여기서 뽑지 않는다. 뽑아 두면
        # 그 값이 곧바로 덮여 쓰이지 않는다.
        _varied_by_conf = ("defocus", "flare", "msd_nm", "blur_nm")
        _base_axes = [k for k in SPACE if k not in _varied_by_conf]
        for i in range(n_sets):
            over = {k: float(rng.uniform(*SPACE[k])) for k in _base_axes}
            over.update(arms[i % len(arms)])   # 이산 조합을 고르게 섞는다
            tasks.append((base.__dict__, over, args.seed + i))

        conf, bar = [], Progress("묶음")
        if workers > 1 and n_sets > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for j, part in enumerate(ex.map(_confusion_one, tasks,
                                                chunksize=4)):
                    conf += part
                    bar(j + 1, n_sets)
        else:
            for j, t in enumerate(tasks):
                conf += _confusion_one(t)
                bar(j + 1, n_sets)
        print(f"  혼동 집합 {len(conf):,} 건 생성")
        specs += conf
    print(f"\n  총 생성 대상 {len(specs):,} 건 "
          f"(하이퍼큐브 {n_lhc:,} + 혼동 {len(specs)-n_lhc:,})")

    if not specs:
        print("\n생성 대상이 없습니다. 조건 공간을 확인하세요.")
        return

    # 2단계 : 대량 생성
    print(f"\n[2/3] 데이터 생성 (워커 {workers})")
    ckpt = CKPT_PATH if (args.resume or True) else None
    if not args.resume and os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    result = CP.run_campaign(
        specs, n_workers=workers, checkpoint=ckpt, checkpoint_every=200,
        progress=Progress("샘플"),
    )
    n_ok = len(result["conditions"])
    print(f"  소요 {_fmt(result['elapsed_s'])} · 적격 {n_ok:,} / {len(specs):,} "
          f"({100 * n_ok / len(specs):.1f} %)")
    rj = result["reject"]
    print("  탈락 : " + ", ".join(
        f"{k} {rj.get(k, 0):,}" for k in
        ("formed", "nils_ok", "cdu_ok", "budget_ok") if k in rj or True))

    if n_ok == 0:
        print("\n적격 샘플이 없습니다.")
        return

    # 3단계 : 저장 및 요약
    print("\n[3/3] 저장")
    out_path = args.out or NPZ_PATH

    # 학습/검증 분할을 함께 저장한다. 광학 조건 단위로 나누지 않으면 같은
    # 조건의 형제 표본이 양쪽에 들어가 성능이 과대평가된다.
    split_arrays = {}
    if not args.no_split:
        ki = result.get("kept_index")
        if ki is None or len(ki) != n_ok:
            raise RuntimeError("kept_index 가 적격 표본 수와 맞지 않습니다.")
        kept = [specs[int(i)] for i in ki]
        try:
            sp = SPL.grouped_split(kept, seed=args.seed)
            for k in ("train", "val", "test"):
                split_arrays[f"split_{k}"] = np.array(sp[k], dtype=np.int64)
            lk = SPL.split_leakage(kept, sp)
            print(f"  분할 : 학습 {len(sp['train']):,} / 검증 {len(sp['val']):,} "
                  f"/ 시험 {len(sp['test']):,}")
            print(f"  광학 조건 {lk['n_groups']:,}개, 그룹 누수 {lk['n_leaked_groups']}개")
        except Exception as e:
            print(f"  분할 생략 ({e})")
    np.savez_compressed(
        out_path,
        conditions=result["conditions"], profiles=result["profiles"],
        labels=result["labels"], target_cd=result["target_cd"],
        condition_keys=np.array(DS.CONDITION_KEYS),
        label_keys=np.array(DS.LABEL_KEYS),
        **split_arrays,
    )
    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"  {out_path}  ({size_mb:.1f} MB)")

    lines = [
        "=" * 72,
        " 데이터셋 생성 요약",
        "=" * 72,
        f"생성 일시        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"적격 샘플        {n_ok:,} / {len(specs):,} ({100 * n_ok / len(specs):.1f} %)",
        f"광학 조건 조합   {n_optical:,} (적격 없음 {n_skipped}, "
        f"세밀화 {n_refined})",
        f"혼동 집합        {len(specs) - n_lhc:,} 건 (비율 {args.confusion:.0%})",
        f"워커             {workers} / 논리 CPU {n_cpu}",
        f"사전 탐색        {_fmt(screen_s)}",
        f"생성             {_fmt(result['elapsed_s'])}",
        f"적격 1건당       {result['elapsed_s'] / n_ok:.3f} s",
        f"파일 크기        {size_mb:.1f} MB",
        "탈락 사유        " + ", ".join(
            f"{k} {result['reject'].get(k, 0):,}" for k in
            ("formed", "nils_ok", "cdu_ok", "budget_ok")),
        "",
        "조건 분포",
    ]
    for k in DS.CONDITION_KEYS:
        v = result["conditions"][:, DS.CONDITION_KEYS.index(k)]
        lines.append(f"  {k:<12s} {v.min():12.4f} ~ {v.max():12.4f}   "
                     f"평균 {v.mean():12.4f}")
    lines += ["", "라벨 분포"]
    for k in DS.LABEL_KEYS:
        v = result["labels"][:, DS.LABEL_KEYS.index(k)]
        lines.append(f"  {k:<18s} {np.nanmin(v):12.4f} ~ {np.nanmax(v):12.4f}   "
                     f"평균 {np.nanmean(v):12.4f}")
    lines.append("=" * 72)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {SUMMARY_PATH}")
    print("\n" + "\n".join(lines[3:12]))
    print(f"\n완료. 총 {_fmt(screen_s + result['elapsed_s'])} 소요.")


if __name__ == "__main__":
    main()
