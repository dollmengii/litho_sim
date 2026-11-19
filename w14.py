#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14 주차 재측정 실행기.

런북(재측정_런북.md)의 1~6 단계를 그대로 돌리되, **중단해야 하는 신호를
사람이 눈으로 찾지 않아도 되게** 자동으로 검사한다.

    py w14.py 0        환경 점검 (30 초)
    py w14.py 1        회귀 검증 427/427 확인 (1 분)
    py w14.py 2        데이터 생성 : 소규모 확인 후 본 생성 (약 20 시간)
    py w14.py 3        대리 모형 학습 3 회 (수십 분)
    py w14.py 4        추천기 한 번 (수 분)
    py w14.py 5        되돌림 3 회 (수 시간)
    py w14.py 6        전체 검증 + claims (약 1 시간)
    py w14.py all      0~6 을 순서대로

모든 화면 출력은 w14_out/ 아래에 단계별 .txt 로 같이 저장된다.
중단 신호가 걸리면 **그 자리에서 멈추고** 무엇이 왜 잘못됐는지 적는다.

주요 선택지
    --workers N     생성 워커 수 (기본 : 코어 수 - 1, 최대 8)
    --target N      생성 목표 적격 수 (기본 100000)
    --pitch X       4·5 단계 피치 [nm] (기본 250)
    --smoke-only    2 단계에서 소규모 확인만 하고 멈춘다
    --skip-smoke    2 단계에서 소규모 확인을 건너뛴다 (이미 했을 때)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from typing import NamedTuple

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "w14_out")
DATA = os.path.join(HERE, "dataset", "litho_dataset.npz")
SMOKE = os.path.join(HERE, "dataset", "smoke.npz")
CKPT = os.path.join(HERE, "dataset", "checkpoint.pkl")

BAR = "=" * 74


class Stop(Exception):
    """더 진행하면 안 되는 상태. 메시지를 그대로 화면에 낸다."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(title: str) -> None:
    say("\n" + BAR)
    say(" " + title)
    say(BAR)


def hhmm(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f"{sec} 초"
    if sec < 3600:
        return f"{sec // 60} 분 {sec % 60} 초"
    return f"{sec // 3600} 시간 {(sec % 3600) // 60} 분"


class Run(NamedTuple):
    """하위 스크립트 한 번의 결과. [14 차 보완 113 번]

    처음에는 `run()` 이 화면 출력만 돌려주었고, 성공·실패를 그 **글에서
    문자열로 찾아** 판정했다. 6 단계에서 `"검사 11 개 중 실패" in txt` 로
    본 것이 그것인데, 성공했을 때 찍히는 `"검사 11 개 중 실패 0 개"` 에도
    그 토막이 들어 있다. 그래서 484/484 전부 통과한 실행을 실패로 읽었다.
    종료 코드는 **이미 있는 분명한 신호**다. 글에서 다시 찾지 않는다.
    """
    text: str
    rc: int


def run(name: str, args: list[str], *, allow_fail: bool = False) -> Run:
    """하위 스크립트를 돌리고 화면과 파일에 동시에 적는다.

    화면 출력과 **종료 코드**를 함께 돌려준다(113 번).

    한글 출력이 섞이므로 자식 쪽 인코딩을 utf-8 로 못박는다. 이게 없으면
    윈도우에서 cp949 로 나가다가 UnicodeEncodeError 로 죽는다.
    """
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.txt")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 자식의 stdout 이 파이프라 **블록 버퍼**가 된다. 이게 없으면 20 시간짜리
    # 생성이 도는 동안 화면이 내내 비어 있어서, 도는 중인지 멈춘 건지
    # 구분할 수가 없다. -u 와 환경변수를 둘 다 건다.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-u"] + args
    say(f"\n  $ py {' '.join(args)}")
    say(f"  (화면 기록 : w14_out/{name}.txt)\n")

    t0 = time.time()
    buf: list[str] = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"$ py {' '.join(args)}\n\n")
        proc = subprocess.Popen(
            cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            buf.append(line)
        rc = proc.wait()
        fh.write(f"\n[종료 코드 {rc}]  소요 {hhmm(time.time() - t0)}\n")

    text = "".join(buf)
    say(f"\n  -- 소요 {hhmm(time.time() - t0)} · 종료 코드 {rc}")
    if rc != 0 and not allow_fail:
        raise Stop(
            f"`{' '.join(args)}` 가 종료 코드 {rc} 로 끝났습니다.\n"
            f"  w14_out/{name}.txt 의 마지막 30 줄을 그대로 보내 주십시오."
        )
    return Run(text, rc)


def need(text: str, pattern: str, what: str, why: str) -> re.Match:
    m = re.search(pattern, text)
    if not m:
        raise Stop(f"{what}\n  {why}")
    return m


# ----------------------------------------------------------------------
# 0 단계 : 환경
# ----------------------------------------------------------------------

def step0(a) -> None:
    head("0 단계 : 환경 점검")
    ok = True

    say(f"  파이썬      {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        say("  [문제] 3.10 이상이 필요합니다.")
        ok = False

    cores = os.cpu_count() or 1
    say(f"  코어        {cores} 개 (논리)")
    if a.workers > cores:
        say(f"  [주의] 워커 {a.workers} 개가 코어보다 많습니다. 느려집니다.")

    free = shutil.disk_usage(HERE).free / 1e9
    say(f"  남은 디스크 {free:.1f} GB")
    if free < 5:
        say("  [문제] 5 GB 는 있어야 합니다. 생성 중간에 끊기면 처음부터입니다.")
        ok = False

    need_mod = ["numpy", "scipy", "sklearn", "joblib", "matplotlib"]
    missing = []
    for m in need_mod:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    if missing:
        say(f"  [문제] 없는 꾸러미 : {', '.join(missing)}")
        say("         py -m pip install -r requirements.txt")
        ok = False
    else:
        say(f"  꾸러미      {', '.join(need_mod)} 전부 있음")

    if os.path.exists(CKPT):
        age = (time.time() - os.path.getmtime(CKPT)) / 3600
        say(f"\n  [주의] dataset/checkpoint.pkl 이 있습니다 ({age:.1f} 시간 전).")
        say("         이건 전에 돌리다 만 생성의 찌꺼기입니다. --resume 없이")
        say("         새로 돌릴 것이므로 2 단계에서 지웁니다.")

    if not ok:
        raise Stop("환경이 준비되지 않았습니다. 위 [문제] 를 먼저 고치십시오.")
    say("\n  환경 이상 없음.")


# ----------------------------------------------------------------------
# 1 단계 : 회귀 검증
# ----------------------------------------------------------------------

def step1(a) -> None:
    head("1 단계 : 수정이 들어 있는지 확인")
    say("  11 시간 돌리고 나서 옛 코드였다는 걸 알면 처음부터 다시 해야 합니다.")
    txt = run("01_validate_w13", ["validate_w13.py"]).text

    m = need(txt, r"결과:\s*(\d+)/(\d+)\s*통과",
             "validate_w13.py 가 결과 줄을 내지 않았습니다.",
             "코드 판이 섞였을 수 있습니다.")
    got, tot = int(m.group(1)), int(m.group(2))
    if got != tot:
        raise Stop(f"회귀 {got}/{tot}. 실패한 사례가 있습니다. 진행 금지.")
    if tot < 427:
        raise Stop(
            f"회귀 총계가 {tot} 입니다. 427 이어야 합니다.\n"
            "  14 주차 수정이 안 들어간 옛 판입니다. zip 을 다시 푸십시오."
        )
    for case in ("Case 24", "Case 25", "Case 26", "Case 27", "Case 28",
                 "Case 29", "Case 30", "Case 31", "Case 32"):
        if case not in txt:
            raise Stop(f"{case} 가 없습니다. 14 주차 수정이 빠진 판입니다.")
    say(f"\n  회귀 {got}/{tot} 통과. Case 24~32 전부 확인. 14 주차 판이 맞습니다.")


# ----------------------------------------------------------------------
# 2 단계 : 데이터 생성
# ----------------------------------------------------------------------

def _check_band(txt: str, where: str) -> tuple[int, int]:
    m = need(txt, r"적격\s*([\d,]+)\s*·\s*경계 밖 띠\s*([\d,]+)",
             f"{where} 에서 '적격 N · 경계 밖 띠 M' 줄을 못 찾았습니다.",
             "--keep-formed 가 빠졌거나 옛 코드입니다.")
    elig = int(m.group(1).replace(",", ""))
    band = int(m.group(2).replace(",", ""))
    if band == 0:
        raise Stop(
            f"{where} : 경계 밖 띠가 0 건입니다.\n"
            "  이번 작업의 핵심이 빠진 것입니다. 문턱 아래 학습 데이터가\n"
            "  없으면 모형이 경계의 한쪽만 보고 품질을 판정하게 됩니다\n"
            "  (103 번의 뿌리). --keep-formed 가 붙었는지 확인하십시오."
        )
    if elig == 0:
        raise Stop(f"{where} : 적격이 0 건입니다. 조건 공간 설정이 어긋났습니다.")
    return elig, band


def step2(a) -> None:
    head("2 단계 : 데이터 생성")

    if os.path.exists(CKPT):
        os.remove(CKPT)
        say("  옛 체크포인트를 지웠습니다 (찌꺼기를 이어받지 않도록).")

    if not a.skip_smoke:
        say(f"\n  먼저 소규모 {a.smoke_target} 건으로 설정이 맞는지만 봅니다 (수 분).")
        txt = run("02a_generate_smoke", [
            "generate_dataset.py", "--target", str(a.smoke_target),
            "--workers", str(a.workers), "--aberr-jitter", "3",
            "--keep-formed", "--out", "dataset/smoke.npz",
        ]).text
        e, b = _check_band(txt, "소규모 생성")
        if not os.path.exists(SMOKE):
            raise Stop("dataset/smoke.npz 가 생기지 않았습니다.")
        say(f"\n  소규모 확인 통과 : 적격 {e:,} · 띠 {b:,} "
            f"({100 * b / (e + b):.1f} %)")
        os.remove(SMOKE)
        if os.path.exists(CKPT):
            os.remove(CKPT)
        say("  소규모 결과물은 지웠습니다. 본 생성과 섞이면 안 됩니다.")

    if a.smoke_only:
        say("\n  --smoke-only 라 여기서 멈춥니다.")
        return

    say(f"\n  본 생성 : 목표 적격 {a.target:,} 건, 워커 {a.workers} 개.")
    say("  8 코어 기준 약 20 시간입니다. 밤에 걸어 두십시오.")
    say("  PC 가 절전으로 들어가지 않게 전원 설정을 먼저 확인하십시오.")
    txt = run("02b_generate_full", [
        "generate_dataset.py", "--target", str(a.target),
        "--workers", str(a.workers), "--aberr-jitter", "3", "--keep-formed",
    ]).text
    e, b = _check_band(txt, "본 생성")
    if not os.path.exists(DATA):
        raise Stop("dataset/litho_dataset.npz 가 생기지 않았습니다.")
    say(f"\n  생성 완료 : 적격 {e:,} · 경계 밖 띠 {b:,} "
        f"({100 * b / (e + b):.1f} %)")
    say("  ** 이 두 숫자를 적어 두십시오. 보고서에 들어갑니다. **")


# ----------------------------------------------------------------------
# 3 단계 : 학습
# ----------------------------------------------------------------------

def _check_leak(txt: str, where: str) -> None:
    if "그룹 누설 없음" not in txt:
        raise Stop(
            f"{where} : '그룹 누설 없음' 이 안 나왔습니다.\n"
            "  분할이 샜습니다. 이대로 가면 모든 오차가 낙관적으로 나옵니다.\n"
            f"  w14_out/ 의 해당 .txt 를 보내 주십시오."
        )
    if "검사가 잡지 못합니다" in txt:
        raise Stop(
            f"{where} : 누설 검사의 자기시험이 실패했습니다.\n"
            "  '누설 없음' 이 탐지 능력이 없어서 나온 것일 수 있습니다."
        )


def step3(a) -> None:
    head("3 단계 : 대리 모형 학습")
    if not os.path.exists(DATA):
        raise Stop("dataset/litho_dataset.npz 가 없습니다. 2 단계를 먼저.")

    say("\n  (1/3) 세 모형 비교 + 프로파일 음수 세기 (105 번)")
    t1 = run("03a_train_compare", [
        "train_surrogate.py", "--data", "dataset/litho_dataset.npz",
        "--profiles",
    ]).text
    _check_leak(t1, "학습 1/3")
    _check_band(t1, "학습 1/3")

    say("\n  (2/3) 추천기가 쓸 모형 저장 + 형성 분류기 (108 번)")
    t2 = run("03b_train_save", [
        "train_surrogate.py", "--data", "dataset/litho_dataset.npz",
        "--model", "mlp", "--labels", "cd_error,nils,cd_3sigma",
        "--save-model", "models/s",
    ]).text
    _check_leak(t2, "학습 2/3")
    if "[형성 분류기] 학습하지 않음" in t2:
        m = re.search(r"학습하지 않음\s*:\s*(.+)", t2)
        why = m.group(1).strip() if m else "사유 불명"
        if "부족" in why:
            extra = ("  선별 맵의 점 수가 모자랍니다. 이건 데이터셋이 작아서\n"
                     "  생기는 일이고, --target 을 줄여 돌렸다면 정상입니다.\n"
                     "  본 생성(10 만)에서는 안 나와야 합니다. 본 생성인데도\n"
                     "  나왔다면 --aberr-jitter 3 이 빠졌는지 확인하십시오.")
        elif "없" in why:
            extra = ("  데이터셋에 선별 맵이 아예 없습니다. 거의 항상\n"
                     "  --keep-formed 를 빠뜨린 것입니다. 2 단계를 다시.")
        else:
            extra = "  사유를 그대로 보내 주십시오."
        raise Stop(f"형성 분류기가 학습되지 않았습니다 : {why}\n{extra}")
    if "[형성 분류기]" not in t2:
        raise Stop("형성 분류기 출력이 아예 없습니다. 옛 코드입니다.")
    for f in ("models/s_mlp_cd_error.joblib", "models/s_mlp_nils.joblib",
              "models/s_mlp_cd_3sigma.joblib", "models/s_formed.joblib"):
        if not os.path.exists(os.path.join(HERE, f)):
            raise Stop(f"{f} 가 생기지 않았습니다.")
    say("\n  모형 네 개 저장 확인 (회귀 3 + 형성 분류기 1).")

    say("\n  (3/3) 옛 설계 대조군 : 적격 행만으로 한 번 더 (107 번 대조)")
    say("      이게 있어야 '고쳐서 좋아진 것' 과 '데이터가 달라져서")
    say("      좋아진 것' 을 가를 수 있습니다.")
    t3 = run("03c_train_eligible_only", [
        "train_surrogate.py", "--data", "dataset/litho_dataset.npz",
        "--model", "mlp", "--labels", "cd_error,nils,cd_3sigma",
        "--eligible-only", "--save-model", "models/old",
    ]).text
    _check_leak(t3, "학습 3/3")
    say("\n  학습 3 회 완료. surrogate_report.json 과 w14_out/03*.txt 를 보관하십시오.")


# ----------------------------------------------------------------------
# 4 단계 : 추천기
# ----------------------------------------------------------------------

COMMON = ["--duty", "0.5", "--flare", "0.02", "--msd-nm", "2.0",
          "--blur-nm", "10", "--z6", "0.02", "--z11", "0.02"]

#: 추천기가 1 단계 분류기를 찾았을 때와 못 찾았을 때 찍는 문구.
FORMED_OK = "[1 단계 형성 분류기] 있음"
FORMED_NG = "[1 단계 형성 분류기] **없다.**"


def formed_missing(txt: str) -> bool:
    """추천기 출력에서 '분류기를 못 찾았다' 를 읽어 낸다. [14 차 보완 112 번]

    처음에는 `"1 단계 형성 분류기" in txt and "없다" in txt` 로 짰다. 두
    토막을 **따로** 찾은 것이라, 같은 출력 안 다른 문장의 '없다' 에 걸렸다.
    실제로 걸린 문장은 `"보수 창이 아예 없다는 뜻이다"` 였고, 분류기를
    제대로 찾았는데도 실행기가 멈췄다. 105·108·110 번과 같은 유형이다 --
    **상태를 판정하는 조건이 그 상태를 가리키지 않았다.**

    문구 전체를 한 덩어리로 맞춘다. 그리고 '있음' 이 보이면 그것이 답이다.
    """
    if FORMED_OK in txt:
        return False
    return FORMED_NG in txt


def step4(a) -> None:
    head("4 단계 : 추천기 한 번 돌려 보기")
    txt = run("04_recommend", [
        "recommend.py", "--models", "models/s_mlp",
        "--pitch", str(a.pitch)] + COMMON + [
        "--pol", "TE", "--source", "conventional", "--mask", "binary",
    ], allow_fail=True).text

    if "학습 범위(domain)가 없는 옛 모형" in txt:
        raise Stop(
            "모형 파일이 11 주차 판입니다. 3 단계를 다시 돌려 저장하십시오."
        )
    if "학습 범위 밖입니다" in txt:
        raise Stop(
            f"요청한 조건(피치 {a.pitch} nm)이 학습 범위 밖이라 거부됐습니다.\n"
            "  거부는 **맞는 동작**입니다. mlp 는 범위 밖에서 포화해서 가짜\n"
            "  최적점을 만들기 때문에 경고가 아니라 거부로 둔 것입니다.\n"
            "  화면에 찍힌 범위 안의 값으로 바꿔 다시 돌리십시오.\n"
            "    예 :  py w14.py 4 --pitch 240"
        )
    if formed_missing(txt):
        raise Stop(
            "추천기가 형성 분류기를 못 찾았습니다.\n"
            "  화면에 '찾아본 곳' 으로 찍힌 경로에 _formed.joblib 이 있는지\n"
            "  확인하십시오. 3 단계를 --save-model 과 함께 돌려야 생깁니다."
        )
    if "판정 불가" not in txt:
        say("\n  [주의] '판정 불가' 가 한 건도 안 나왔습니다. 틀린 건 아니지만,")
        say("         14 주차 수정이 걸리는 경우를 하나도 못 본 것이므로")
        say("         이 결과만으로는 수정이 작동한다고 말할 수 없습니다.")
        say("         5 단계(되돌림)에서 다시 봅니다.")
    else:
        say("\n  '판정 불가' 가 나왔습니다. 102~104 번 수정이 작동 중입니다.")
    say("  CD <= 0 예측 점 수도 화면에 찍혔으면 같이 적어 두십시오 (0 이어도 결과입니다).")


# ----------------------------------------------------------------------
# 5 단계 : 되돌림
# ----------------------------------------------------------------------

def step5(a) -> None:
    head("5 단계 : 되돌림 재측정 (이번 작업의 본체)")
    say("  초점 격자를 33 → 129 → 257 로 올리며 세 번 돕니다.")
    say("  13 주차에는 통과군이 27 → 2 로 무너졌습니다. 그 숫자가 어떻게")
    say("  달라지는지가 이번에 보고 싶은 것입니다.\n")

    extra = []
    if a.n_top:
        extra += ["--n-top", str(a.n_top)]
        say(f"  후보를 상위 {a.n_top} 개로 늘려 돕니다. 기준 A 는 통과군이")
        say("  6 개 이상이어야 잴 수 있고, 기본값(8)으로는 4 개뿐이었습니다.")
        say("  **기준 자체는 그대로입니다.** 바꾼 것은 재는 표본 수입니다.")
    if a.n_spread:
        extra += ["--n-spread", str(a.n_spread)]

    for n in (33, 129, 257):
        tag = f"{n:03d}"
        run(f"05_roundtrip_{tag}", [
            "roundtrip_w13.py", "--models", "models/s_mlp",
            "--pitch", str(a.pitch), "--duty", "0.5",
            "--n-focus", str(n), "--save-json", f"rt_{tag}.json",
        ] + extra)
        if not os.path.exists(os.path.join(HERE, f"rt_{tag}.json")):
            raise Stop(f"rt_{tag}.json 이 생기지 않았습니다.")
        say(f"  rt_{tag}.json 저장 완료.")

    say("\n  세 번 모두 끝났습니다. rt_*.json 과 w14_out/05_*.txt 가 보고서의")
    say("  숫자 출처입니다. 둘 다 보내 주십시오.")


# ----------------------------------------------------------------------
# 6 단계 : 전체 검증
# ----------------------------------------------------------------------

def _dump_failed_logs(txt: str) -> None:
    """run_all_checks 가 실패했을 때 실패한 로그의 끝을 바로 찍는다.

    [14 차 보완 110 번의 교훈] 요약표는 '392/396' 까지만 말하고 **어느
    사례인지는 말하지 않는다.** 그 네 줄은 check_logs/ 에 이미 있는데,
    사람이 따로 찾아 열어야 했다. 찾게 만들지 말고 여기서 찍는다.
    """
    bad = re.findall(r"^\s+(\S+\.py)\s+[1-9]\d*\s", txt, re.M)
    if not bad:
        return
    say("\n" + "-" * 74)
    say(" 실패한 검사의 로그 끝부분 (check_logs/ 에 전문이 있습니다)")
    say("-" * 74)
    for script in dict.fromkeys(bad):
        p = os.path.join(HERE, "check_logs", f"{script[:-3]}.txt")
        say(f"\n  === {script}  ({p})")
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                tail = fh.read().splitlines()[-40:]
        except OSError as e:
            say(f"    (로그를 열 수 없습니다 : {e})")
            continue
        for line in tail:
            say("    " + line)


def step6(a) -> None:
    head("6 단계 : 전체 검증 완주")
    r6 = run("06a_run_all_checks", ["run_all_checks.py"], allow_fail=True)
    # [113 번] 성공·실패는 **종료 코드**로 본다. 글에서 '실패' 를 찾으면
    # 성공했을 때 찍히는 "검사 11 개 중 실패 0 개" 에 걸린다. 실제로 걸렸다.
    if r6.rc != 0:
        _dump_failed_logs(r6.text)
        raise Stop(
            "전체 검증에 실패한 검사가 있습니다.\n"
            "  위에 찍힌 로그 끝부분을 그대로 보내 주십시오.\n"
            "  (w14_out/06a_run_all_checks.txt 에도 같이 들어 있습니다)"
        )
    say("\n  종료 코드 0. 끝 요약표를 통째로 보내 주십시오.")

    say("\n  이제 문서에 적힌 숫자와 코드가 어긋나는지 봅니다.")
    say("  전부 일치하면 문서가 코드와 맞는 것입니다. 재측정으로 숫자를")
    say("  고친 뒤라면 몇 개가 실패하는 것이 정상이고, 그 실패 목록이")
    say("  곧 '고쳐야 할 문서 줄' 입니다.")
    run("06b_claims", ["claims.py"], allow_fail=True)


# ----------------------------------------------------------------------

def checklist() -> None:
    head("끝나면 보내 줄 것")
    items = [
        ("dataset/summary.txt", "생성 요약"),
        ("surrogate_report.json", "학습 결과"),
        ("w14_out/02b_generate_full.txt", "적격 N · 경계 밖 띠 M"),
        ("w14_out/03b_train_save.txt", "형성 분류기 AUC·정확도"),
        ("w14_out/03c_train_eligible_only.txt", "옛 설계 대조군"),
        ("rt_033.json", "되돌림 33"),
        ("rt_129.json", "되돌림 129"),
        ("rt_257.json", "되돌림 257"),
        ("w14_out/06a_run_all_checks.txt", "전체 검증 요약표"),
        ("w14_out/06b_claims.txt", "claims 실패 목록"),
    ]
    say("")
    for rel, what in items:
        mark = "있음" if os.path.exists(os.path.join(HERE, rel)) else "없음"
        say(f"  [{mark}]  {rel:<42} {what}")
    say("\n  w14_out 폴더를 통째로 압축해서 보내는 것이 가장 간단합니다.")


STEPS = {"0": step0, "1": step1, "2": step2, "3": step3,
         "4": step4, "5": step5, "6": step6}


def main() -> int:
    ap = argparse.ArgumentParser(description="14 주차 재측정 실행기")
    ap.add_argument("step", choices=list(STEPS) + ["all", "check"],
                    help="돌릴 단계. all 이면 0~6 을 순서대로")
    cores = os.cpu_count() or 2
    ap.add_argument("--workers", type=int, default=min(8, max(1, cores - 1)))
    ap.add_argument("--target", type=int, default=100000)
    ap.add_argument("--pitch", type=float, default=250.0)
    ap.add_argument("--smoke-target", type=int, default=300,
                    help="2 단계 소규모 확인의 목표 적격 수")
    ap.add_argument("--n-top", type=int, default=0,
                    help="5 단계 되돌림의 상위 후보 수 (기본 8). 통과군이 "
                         "6 개가 안 되면 기준 A 를 잴 수 없다")
    ap.add_argument("--n-spread", type=int, default=0,
                    help="5 단계 구간 표본 수 (기본 12)")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if a.step == "check":
        checklist()
        return 0

    order = ["0", "1", "2", "3", "4", "5", "6"] if a.step == "all" else [a.step]
    try:
        for s in order:
            STEPS[s](a)
    except Stop as e:
        say("\n" + "!" * 74)
        say(" 중단합니다. 더 진행하지 마십시오.")
        say("!" * 74)
        say(f"\n  {e}\n")
        return 2
    except KeyboardInterrupt:
        say("\n\n  사용자가 멈췄습니다. 2 단계였다면 --resume 로 이어서 돌릴 수")
        say("  있습니다 :  py generate_dataset.py --target 100000 --workers 8 "
            "--aberr-jitter 3 --keep-formed --resume")
        return 130

    say("\n" + BAR)
    say(f" 끝. 총 소요 {hhmm(time.time() - t0)}")
    say(BAR)
    if a.step in ("all", "6"):
        checklist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
