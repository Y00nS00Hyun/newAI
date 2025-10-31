"""
공통 상수 정의

참가자와 서버 양측에서 공통으로 사용하는 상수를 정의합니다.
"""

# 레이블 매핑
# 학습용: 문자열 → 숫자 (CrossEntropyLoss는 정수 레이블 필요)
LABEL_TO_IDX = {
    "real": 0,
    "fake": 1
}

# 추론용: 숫자 → 문자열
IDX_TO_LABEL = {
    0: "real",
    1: "fake"
}

