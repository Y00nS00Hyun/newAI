"""
모델 레지스트리

새로운 모델을 추가하려면:
1. model_definitions 폴더에 모델 클래스 파일 추가 (예: transformer.py)
2. 이 파일에서 import 후 MODEL_REGISTRY에 등록
"""

from model_definitions.bilstm import BiLSTM
from model_definitions.cnn import TextCNN

# 모델 레지스트리: 모델 이름 -> 모델 클래스
MODEL_REGISTRY = {
    "bilstm": BiLSTM,
    "cnn": TextCNN,
    # 새로운 모델 추가 예시:
    # "transformer": TransformerModel,
}

__all__ = ["MODEL_REGISTRY", "BiLSTM"]
