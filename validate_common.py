"""
validate_common.py -- 회귀 시험 케이스의 성격 분류

왜 필요한가
-----------
validate*.py 는 쉰 건이 넘는 케이스를 돌리고 "n/n 통과" 를 출력한다. 그런데 이
숫자는 보이는 것보다 약한 증거다. 케이스마다 '무엇에 대조했는가' 가 전혀
다르기 때문이다.

  - 해석식이나 문헌값과 대조한 것 : 틀리면 반드시 드러난다.
  - 물리가 요구하는 등식(불변량)을 본 것 : 기준값이 필요 없어 강하다.
  - 측정해 놓고 임계값을 나중에 고른 것 : 모델을 잘못 이해했다면 시뮬레이터와
    기준선이 함께 틀린 채로 통과한다.
  - 수행 시간이나 파일 구성을 본 것 : 물리 검증이 아니다. 하드웨어와 코드
    구조에 따라 바뀐다.

이 네 가지를 한 숫자로 합쳐 보고하면 과장이 된다. 실제로 10주차에 결상
정확도를 올렸을 때, 물리가 좋아졌는데도 '옛 거동에 맞춰진' 임계값 두 개가
깨져 시험이 실패했다. 그 임계값들은 물리를 검증한 것이 아니라 당시의 숫자를
기록한 것이었다.

그래서 각 케이스에 성격을 붙이고, 요약에서 성격별로 나누어 센다. 사후
임계값을 전부 해석 기준으로 바꾸는 것은 불가능하다(LER 절대값처럼 해석
기준선이 없는 양이 있다). 대신 **어느 것이 약한 근거인지 숨기지 않는 것**이
올바른 처리다.

분류
----
해석  : 해석해·문헌값과의 정량 대조. 기준이 시뮬레이터 밖에서 온다.
불변량 : 물리가 요구하는 등식·순서·대칭. 기준값이 필요 없다.
정합성 : 저장·복원, 병렬·직렬 일치, 두 경로 일치 등 내부 일관성.
기록  : 측정값에 사후 임계값을 붙인 것. 회귀 감시용이며 검증 근거로는 약하다.
성능  : 수행 시간·비용·파일 구성. 물리 검증이 아니다.
"""

KIND_ANALYTIC = "해석"
KIND_INVARIANT = "불변량"
KIND_CONSISTENCY = "정합성"
KIND_RECORD = "기록"
KIND_PERF = "성능"

KIND_ORDER = [KIND_ANALYTIC, KIND_INVARIANT, KIND_CONSISTENCY,
              KIND_RECORD, KIND_PERF]

#: 케이스 이름의 일부 -> 성격. 이름이 바뀌어도 걸리도록 짧은 조각을 쓴다.
_TABLE = [
    # --- 해석해·문헌 대조 ------------------------------------------------
    ("open frame 정규화", KIND_INVARIANT),
    ("k1 = 0.272", KIND_ANALYTIC),
    ("k1 = 0.25", KIND_ANALYTIC),
    ("coherent 가정의 오류", KIND_ANALYTIC),
    ("CD 추출 정확도", KIND_ANALYTIC),
    ("dose-to-size 보정", KIND_ANALYTIC),
    ("완전 결맞음 해상 한계", KIND_ANALYTIC),
    ("k1 의 조명 조건 의존성", KIND_ANALYTIC),
    ("k2 역산", KIND_ANALYTIC),
    ("k1 해석 기준선", KIND_ANALYTIC),
    ("광자 통계 및 파장 의존성", KIND_ANALYTIC),
    ("Mack 근사", KIND_ANALYTIC),
    ("벡터 결상의 해석 편광 계수", KIND_ANALYTIC),
    ("플레어 모델의 해석 대조", KIND_ANALYTIC),
    ("LWR 와 LER 의 이론 관계", KIND_ANALYTIC),
    ("Zernike 정규화", KIND_ANALYTIC),
    # 12주차. 심은 함수와 해석 윈도를 손으로 계산해 대조하므로 기준이
    # 시뮬레이터 밖(손계산)에서 온다. 허용오차는 기계 정밀도와
    # el_dof_curve 가 제 문서에서 유도해 둔 격자 편향 구간이며, 측정해 놓고
    # 나중에 고른 값이 아니다.
    ("격자 방향 정합", KIND_ANALYTIC),
    ("윈도 지표 경로 일치", KIND_ANALYTIC),
    # 선형 dose + Bossung 에서 EL 고정 최대 DOF 를 손으로 풀어 참값으로 쓴다.
    # 격자를 수렴시켜 만든 '참값' 이 아니라 해석해이므로 기준이 코드 밖이다.
    ("DOF 구간이 참값을 담음", KIND_ANALYTIC),
    # 해석 Bossung 의 통과 영역을 손으로 알고, 보고된 윈도 안을 촘촘히
    # 재어 규격 밖 점을 센다. 기준이 코드 밖(해석식)에서 온다.
    ("초점 격자 홀짝과 DOF 방향", KIND_ANALYTIC),

    # --- 불변량 ----------------------------------------------------------
    ("조명 조건 비교", KIND_INVARIANT),
    # 12주차. 기준값이 필요 없다. 거부 여부, 상수성, 단조성, 열 고정만 본다.
    ("상자 제약 거부", KIND_INVARIANT),
    ("공칭 목적함수의 공허성", KIND_INVARIANT),
    ("보수 윈도와 인증 가능성", KIND_INVARIANT),
    ("탐색 축 분리", KIND_INVARIANT),
    # 정확한 축퇴 CD(+dz,+z) = CD(-dz,-z) 를 시뮬레이터에서 기계 정밀도까지
    # 확인한다. 기준값이 필요 없는 대칭이다.
    # 아래 두 줄의 순서가 중요하다. '수차 부호 거울' 이 '거울 적용 시 초점
    # 격자 대칭' 의 부분 문자열은 아니지만, 짧은 조각을 쓰는 방식이므로
    # 새 이름을 넣을 때마다 포함 관계를 확인해야 한다.
    ("거울 적용 시 초점 격자 대칭", KIND_INVARIANT),
    ("정확한 목표 EL 에서 하한 보존", KIND_INVARIANT),
    ("모형 오차 출처의 우선순위", KIND_INVARIANT),
    # 격자가 dz=0 을 담는지는 기준값 없이 격자 자체의 성질로 판정하므로
    # 불변량이다. 담지 않으면 무슨 일이 나는지(DOF 방향 역전)는 '초점 격자
    # 홀짝과 DOF 방향' 쪽에서 해석으로 재고, 둘을 합치지 않는다. 포함 관계
    # 확인 : 이 조각은 위아래 어느 이름의 부분 문자열도 아니다.
    ("초점 격자가 dz=0 을 담음", KIND_INVARIANT),
    # 심은 최적 초점 dz0 를 격자 분해능 안에서 되찾는지 보므로 해석이다.
    # 거울 부호 뒤집힘은 같은 케이스 안에서 함께 보지만, 판정의 중심은
    # '참값을 되찾는가' 이므로 불변량으로 분류하지 않는다.
    ("추천 운전점", KIND_ANALYTIC),
    # 이름을 만들어 두고 읽히는지만 보므로 기준값이 필요 없다.
    ("번들 파일 이름 규약", KIND_INVARIANT),
    # 분류의 완전성과 거부 동작만 보므로 기준값이 필요 없다.
    ("입력 공급 경로의 완전성", KIND_INVARIANT),
    ("수차 부호 거울", KIND_INVARIANT),
    ("EL-DOF trade-off 단조성", KIND_INVARIANT),
    ("NA >= 굴절률 조합 차단", KIND_INVARIANT),
    ("스칼라 모델과 TE 벡터 모델의 관계", KIND_INVARIANT),
    ("최적 초점에서의 부호 모호성", KIND_INVARIANT),
    ("9주차 결함 재발 방지", KIND_INVARIANT),
    ("식별 불가능 영역의 존재", KIND_INVARIANT),

    # --- 정합성 ----------------------------------------------------------
    ("데이터셋 스키마 저장", KIND_CONSISTENCY),
    ("데이터셋 생성 경로 통합", KIND_CONSISTENCY),
    ("GUI 의 7주차 보정 반영", KIND_CONSISTENCY),
    ("체크포인트 저장 및 재개", KIND_CONSISTENCY),
    ("병렬 생성의 정합성", KIND_CONSISTENCY),
    ("학습/검증 분할의 그룹 누수", KIND_CONSISTENCY),
    ("혼동 집합 구성", KIND_CONSISTENCY),
    ("FIM 수치 건전성", KIND_CONSISTENCY),

    # --- 기록 (사후 임계값) ----------------------------------------------
    ("LER 수렴성", KIND_RECORD),
    ("dy 분리에 의한 측정 길이", KIND_RECORD),
    ("레지스트 블러 민감도", KIND_RECORD),
    ("프로파일 다운샘플", KIND_RECORD),
    ("업계 표준 품질 기준", KIND_RECORD),
    ("감쇠형 위상반전마스크", KIND_RECORD),
    ("MSD 와 레지스트 블러의 축퇴", KIND_RECORD),
    ("수차의 비대칭 효과", KIND_RECORD),
    ("조명 형상 비교", KIND_RECORD),
    ("민감도 벡터의 공선성", KIND_RECORD),
    ("잡음 구조가 깨는 축퇴", KIND_RECORD),
    ("미지수 사다리", KIND_RECORD),
    ("계측 처방의 효과", KIND_RECORD),
    ("식별 하한 지도", KIND_RECORD),

    # --- 성능 ------------------------------------------------------------
    ("Dose-Focus 스윕 연산량", KIND_PERF),
    ("데이터 생성 비용 추산", KIND_PERF),
    ("배포 파일 구성", KIND_PERF),
    ("배포 환경 초기 응답 비용", KIND_PERF),
    ("사전 탐색의 비용 효율", KIND_PERF),
    ("적응 격자의 생성 효율", KIND_PERF),
]


def classify(name: str) -> str:
    for frag, kind in _TABLE:
        if frag in name:
            return kind
    return KIND_RECORD          # 분류되지 않은 것은 가장 약한 쪽으로 둔다


def summarize(entries) -> str:
    """[(name, passed), ...] 를 성격별로 센다."""
    buckets = {k: [0, 0] for k in KIND_ORDER}
    for name, passed in entries:
        b = buckets[classify(name)]
        b[0] += int(bool(passed))
        b[1] += 1
    lines = [" 성격별 집계 (같은 '통과' 가 아니다)"]
    for k in KIND_ORDER:
        p, t = buckets[k]
        if t:
            lines.append(f"   {k:<6} {p:>2}/{t:<2}")
    strong = sum(buckets[k][1] for k in (KIND_ANALYTIC, KIND_INVARIANT))
    total = sum(buckets[k][1] for k in KIND_ORDER)
    lines.append(f"   -> 시뮬레이터 밖의 기준에 대조한 것은 {strong}/{total} 건이다.")
    lines.append("      나머지는 회귀 감시(기록)와 수행 비용(성능)이며, 물리")
    lines.append("      모델의 정당성 근거로는 쓸 수 없다. 기준값이 필요 없는")
    lines.append("      수치 검증은 verify.py 에 따로 있다.")
    return "\n".join(lines)
