"""run_all_checks.py -- 모든 검사를 한 번에 돌리고 종료 코드로 판정한다.

왜 이 파일이 필요한가
---------------------
11 주차에 실제로 난 사고가 이것이다. 패키지를 보낼 때마다 `claims.py`,
`verify.py K`, `validate.py` 처럼 **손으로 고른 일부만** 돌려 확인했다.
그래서 `validate_w8` 의 Case 3 이 깨진 상태로 패키지를 다섯 번 내보냈다.
(의존성을 바꾸고 의존성 검사를 다시 돌리지 않았다. VERIFICATION_LIMITS
58~59 번 참조.)

고른 목록이 문제의 원인이므로, 고르지 않는 진입점을 둔다.

  - `validate*.py` 는 **glob 으로 찾는다.** 새 회귀 스크립트를 추가하면
    아무것도 고치지 않아도 자동으로 들어온다. 고정 목록을 적어 두면 추가된
    것이 조용히 빠지는데, 그것이 바로 59 번에서 당한 실수의 구조다.
  - 모든 검사의 **종료 코드**를 모아 하나라도 0 이 아니면 1 을 돌려준다.
    11 주차까지 회귀 일곱 개는 실패해도 0 을 돌려주고 있었다(58 번).
    지금은 고쳤으므로 종료 코드가 뜻을 갖는다.
  - 출력은 각 파일로 남기고 화면에는 요약만 찍는다. 완주가 길기 때문에
    (verify 전체가 약 40 분) 화면이 흐르면 읽을 수 없다.

사용법
------
    py run_all_checks.py              # 전부 (약 55 분)
    py run_all_checks.py --fast       # verify 를 K 만 (약 15 분)
    py run_all_checks.py --list       # 무엇을 돌릴지만 보여 준다

종료 코드
---------
0 : 모든 검사가 0 을 돌려주었다.
1 : 하나라도 실패했다. 어느 것인지 요약표와 로그 파일에 있다.

[범위] 이것은 '전부 돌렸다' 를 보장한다. '물리가 옳다' 를 보장하지 않는다.
verify 는 verification 이고 validation 이 아니다(0 절 참조).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time

#: 로그를 남길 폴더.
LOG_DIR = "check_logs"

#: 회귀가 아닌 검사. 회귀(validate*.py)는 glob 으로 찾으므로 여기 적지 않는다.
#: verify 는 인수에 따라 범위가 달라지므로 따로 둔다.
EXTRA = [
    # [13차 보완 101 번] 여기에 적혀 있던 '37 항목' 은 7 주차 수였고 그
    # 뒤로 한 번도 갱신되지 않았다(실제 55). 이 줄은 **실행 전에** 찍히므로
    # 숫자를 알 수 없고, 적어 두면 반드시 낡는다. 아래 결과 줄이 claims 가
    # 실제로 센 수를 그대로 찍으므로 여기서는 수를 말하지 않는다.
    ("claims.py", [], "문서-코드 정합성"),
]


def discover() -> list[tuple[str, list[str], str]]:
    """돌릴 검사 목록. validate*.py 는 찾아서 넣는다."""
    jobs = list(EXTRA)
    for path in sorted(glob.glob("validate*.py")):
        if path == "validate_common.py":        # 공용 모듈이라 단독 실행 안 함
            continue
        jobs.append((path, [], "회귀 시험"))
    return jobs


#: 검사 하나에 허용하는 최대 시간 [초]. 가장 긴 것이 verify 완주(약 40 분)
#: 이므로 넉넉히 둔다. 타임아웃이 없으면 한 스크립트가 멈출 때 묶음 전체가
#: 조용히 매달린다. 조용히 매달리는 것보다 크게 실패하는 쪽이 낫다.
TIMEOUT_S = 5400


def run_one(script: str, args: list[str], log_path: str) -> tuple[int, float]:
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        try:
            p = subprocess.run([sys.executable, script, *args],
                               stdout=f, stderr=subprocess.STDOUT,
                               timeout=TIMEOUT_S)
            code = p.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n\n[run_all_checks] {TIMEOUT_S} 초를 넘겨 중단함.\n")
            code = 124          # timeout(1) 의 관례와 같은 값
    return code, time.time() - t0


def grep_result(log_path: str) -> str:
    """로그에서 한 줄 요약을 뽑는다. 없으면 빈 문자열."""
    try:
        txt = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    for key in ("결과:", "항목 전부 일치", "판정 대상", "불일치"):
        for line in txt.splitlines():
            if key in line:
                return line.strip()[:58]
    return ""


def count_cases(note: str) -> tuple[int, int]:
    """요약 줄에서 '결과: p/t 통과' 를 뽑는다. 없으면 (0, 0).

    왜 합산하는가 : 회귀 케이스의 **총수**가 README·실행방법·
    VERIFICATION_LIMITS 세 곳에 숫자로 적혀 있다. 11 주차 37 번의 교훈이
    그대로 적용되는 자리다. 문서에 적힌 수를 아무도 세어 보지 않으면 회귀
    스위트를 추가해도 문서만 낡는다(실제로 12 주차에 50 에서 여러 번 늘었다).
    여기서 합을 인쇄해 두면 사람이 한 줄로 대조할 수 있다.

    [범위] 이것은 **인쇄**이지 판정이 아니다. 문서에 적힌 수를 코드가 읽어
    대조하지는 않는다(claims.py 가 의도적으로 묶지 않는 범주에 적어 두었다).
    """
    m = re.search(r"결과:\s*(\d+)\s*/\s*(\d+)", note)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description="모든 검사를 한 번에")
    ap.add_argument("--fast", action="store_true",
                    help="verify 를 K 만 돌린다(정적 검사). 전체는 약 40 분")
    ap.add_argument("--list", action="store_true",
                    help="무엇을 돌릴지만 보여 주고 끝낸다")
    args = ap.parse_args()

    jobs = discover()
    # verify 는 마지막에. 가장 오래 걸리므로 앞의 실패를 먼저 보는 것이 낫다.
    jobs.append(("verify.py", ["K"] if args.fast else [],
                 "정적 검사만" if args.fast else "수치 검증 A~K"))

    print("=" * 78)
    print(" 전체 검사")
    print("=" * 78)
    print(f"\n  돌릴 검사 {len(jobs)} 개"
          f"{' (--fast : verify 는 K 만)' if args.fast else ''}")
    for s, a, desc in jobs:
        print(f"    {s + (' ' + ' '.join(a) if a else ''):<28}{desc}")
    print(f"\n  회귀는 glob 으로 찾는다. 새 validate*.py 를 추가하면 이 파일을")
    print("  고치지 않아도 자동으로 들어온다. 고정 목록을 적어 두면 추가된")
    print("  것이 조용히 빠진다(VERIFICATION_LIMITS 59 번).")

    if args.list:
        return 0

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\n  로그 : {LOG_DIR}/")
    print("=" * 78)

    rows, bad = [], 0
    t_all = time.time()
    for script, a, _ in jobs:
        tag = script.replace(".py", "") + ("_" + "_".join(a) if a else "")
        log = os.path.join(LOG_DIR, f"{tag}.txt")
        print(f"\n  {script}{' ' + ' '.join(a) if a else ''} ...", flush=True)
        code, secs = run_one(script, a, log)
        note = grep_result(log)
        rows.append((script + (" " + " ".join(a) if a else ""),
                     code, secs, note))
        bad += (code != 0)
        print(f"    종료 {code}  {secs:7.1f} s  {note}", flush=True)

    print("\n" + "=" * 78)
    print(" 요약")
    print("=" * 78)
    print(f"\n  {'검사':<26}{'종료':>5}{'초':>9}  요약")
    print("  " + "-" * 74)
    for name, code, secs, note in rows:
        mark = "ok" if code == 0 else "실패"
        print(f"  {name:<26}{code:>5}{secs:>9.1f}  {mark} {note}")

    cp = ct = 0
    silent = []
    for name, _, _, note in rows:
        p, t = count_cases(note)
        cp += p
        ct += t
        # 회귀 스위트가 ' 결과: p/t 통과' 를 안 찍으면 합계에 **0 으로**
        # 들어간다. 그러면 스위트를 추가했는데 합계가 그대로이고, 문서의
        # 숫자와 조용히 어긋난다. 11 주차 37 번과 같은 종류의 구멍이므로
        # 주석이 아니라 실패로 잡는다. 실측 : `validate_w13` 이 자기 형식으로
        # 찍어 51 개가 0 으로 들어갔다.
        if t == 0 and name.startswith("validate") and "common" not in name:
            silent.append(name)
    if ct:
        print(f"\n  회귀 케이스 합계 {cp}/{ct} 통과 "
              f"(문서에 적힌 수와 대조할 것 : README·실행방법·"
              f"VERIFICATION_LIMITS)")
    if silent:
        print("\n  [실패] 아래 회귀 스위트가 케이스 수를 내놓지 않았다.")
        for nm in silent:
            print(f"    - {nm}")
        print("    합계에 0 으로 들어가므로 스위트를 추가해도 합계가 늘지")
        print("    않는다. 요약에 ' 결과: p/t 통과' 형식의 줄을 찍을 것.")
        print("    (`grep_result` 가 찾는 형식이다. 다른 문구는 안 잡힌다.)")

    print(f"\n  총 {time.time() - t_all:.1f} s, 검사 {len(rows)} 개 중 "
          f"실패 {bad} 개"
          + (f", 케이스 수를 안 내놓은 스위트 {len(silent)} 개"
             if silent else ""))
    if bad or silent:
        if bad:
            print("\n  -> 검사 실패가 있다. 해당 로그를 읽을 것.")
        if silent:
            print("\n  -> 검사는 통과했지만 케이스 수 집계가 깨졌다. 합계를")
            print("     문서와 대조할 수 없으므로 통과로 처리하지 않는다.")
        print("     종료 코드 1 을 돌려준다.")
        return 1
    print("\n  -> 전부 0 을 돌려주었다.")
    print()
    print("  [범위] 이것은 '전부 돌렸다' 를 보장한다. '물리가 옳다' 를")
    print("  보장하지 않는다. verify 는 verification 이고 validation 이")
    print("  아니다. 그 구분은 VERIFICATION_LIMITS 0 절에 있다.")
    if args.fast:
        print()
        print("  [주의] --fast 는 verify 를 K 만 돌렸다. 패키지를 내보내기")
        print("  전에는 인수 없이 한 번 완주시킬 것. E2 가 앞 부의 지역변수를")
        print("  참조해 죽는 버그가 있었는데, E2 는 마지막이라 전체를 한 번에")
        print("  돌릴 때만 그 줄에 닿았다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
