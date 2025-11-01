from .bilstm import BiLSTM
from .textcnn import TextCNN
from .improved_textcnn import ImprovedTextCNN

MODEL_REGISTRY = {
    "bilstm": BiLSTM,
    "textcnn": TextCNN,
    "improved_textcnn": ImprovedTextCNN,   # ← 추가
}

__all__ = ["MODEL_REGISTRY", "BiLSTM", "TextCNN", "ImprovedTextCNN"]
