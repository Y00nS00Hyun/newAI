from .cnn import TextCNN

MODEL_REGISTRY = {
    "cnn": TextCNN,
    "textcnn": TextCNN,
}

__all__ = ["MODEL_REGISTRY", "TextCNN"]
