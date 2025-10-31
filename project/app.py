"""
FastAPI 서버: Fake News Detection API
- POST /infer: 단일 텍스트 추론
- POST /infer_csv: CSV 파일 일괄 추론
- POST /reload_model: 모델(best.pt) 다시 불러오기
- POST /validate: 검증 데이터 평가
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

from model_definitions import MODEL_REGISTRY
from utils.dataset import read_csv_file, compose_record_text
from utils.vocab import Vocab, WhitespaceTokenizer, collate_batch
from utils.constants import IDX_TO_LABEL, LABEL_TO_IDX

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Fake News Detection API",
    description="API for fake news detection model inference and validation",
    version="1.0.0"
)

# API Key 설정 (환경변수 또는 기본값)
API_KEY = os.getenv("API_KEY", "ULmLAYYhKeeP9J1c")  # 기본값으로 제공된 API Key 사용

# 전역 변수: 모델, vocab, tokenizer 등
model = None
vocab = None
tokenizer = None
model_config = None
train_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 파일 경로
MODEL_PATH = Path("models/best.pt")
VALIDATION_CSV_PATH = Path("data/validation.csv")


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """API Key 검증 (X-API-Key 헤더로 받음)"""
    if not x_api_key:
        raise HTTPException(
            status_code=401, detail="API Key required. Please provide X-API-Key header.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True


class InferenceRequest(BaseModel):
    """단일 텍스트 추론 요청"""
    text: str
    title: Optional[str] = ""


class InferenceResponse(BaseModel):
    """단일 텍스트 추론 응답"""
    prediction: str  # "real" or "fake"
    confidence: float


class CSVInferenceResponse(BaseModel):
    """CSV 일괄 추론 응답"""
    predictions: List[Dict[str, Any]]


def load_model(model_path: Path):
    """모델 및 관련 리소스 로드"""
    global model, vocab, tokenizer, model_config, train_config

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Vocab 복원
    vocab_data = checkpoint['vocab']
    vocab = Vocab(
        max_size=vocab_data['max_size'],
        min_freq=vocab_data['min_freq']
    )
    vocab.token_to_idx = vocab_data['token_to_idx']
    vocab.idx_to_token = vocab_data['idx_to_token']
    vocab._frozen = True

    # Tokenizer 생성
    tokenizer = WhitespaceTokenizer()

    # 모델 설정 복원
    model_config = checkpoint['model_config']
    model_name = checkpoint['model_name']
    train_config = checkpoint['train_config']

    # 모델 생성 및 로드
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        vocab_size=len(vocab),
        num_classes=2,
        **model_config
    )

    # state_dict 로드 (모델 객체가 저장된 경우 또는 state_dict가 직접 저장된 경우 모두 처리)
    if 'model' in checkpoint:
        saved_model = checkpoint['model']
        if hasattr(saved_model, 'state_dict'):
            # 모델 객체가 저장된 경우
            model.load_state_dict(saved_model.state_dict())
        else:
            # state_dict가 직접 저장된 경우
            model.load_state_dict(saved_model)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("Cannot find model state_dict in checkpoint")

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {model_name}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Using device: {device}")


def predict_single_text(text: str, title: str = "") -> tuple:
    """단일 텍스트에 대한 예측"""
    if model is None or vocab is None or tokenizer is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please call /reload_model first.")

    # 텍스트 구성
    composed_text = compose_record_text(
        pd.Series({"title": title, "text": text}),
        train_config['text_fields']
    )

    # 토크나이징 및 인코딩
    token_ids = vocab.encode(composed_text, tokenizer)
    max_len = train_config['max_len']

    # 패딩
    length = min(len(token_ids), max_len)
    padded_input = torch.zeros(max_len, dtype=torch.long)
    if length > 0:
        padded_input[:length] = torch.tensor(
            token_ids[:length], dtype=torch.long)

    # 배치 차원 추가
    inputs = padded_input.unsqueeze(0).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    # 추론
    with torch.no_grad():
        logits = model(inputs, lengths)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    prediction = IDX_TO_LABEL[pred_idx]
    return prediction, confidence


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 자동 로드"""
    if MODEL_PATH.exists():
        try:
            load_model(MODEL_PATH)
            logger.info("Model loaded on startup")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
    else:
        logger.warning(
            f"Model file not found: {MODEL_PATH}. Please train a model first.")


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Fake News Detection API",
        "endpoints": {
            "/infer": "POST - 단일 텍스트 추론",
            "/infer_csv": "POST - CSV 파일 일괄 추론",
            "/reload_model": "POST - 모델 다시 불러오기",
            "/validate": "POST - 검증 데이터 평가"
        }
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest, api_key: bool = Depends(verify_api_key)):
    """단일 텍스트 추론"""
    try:
        prediction, confidence = predict_single_text(
            request.text, request.title or "")
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error in inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_csv")
async def infer_csv(
    file: UploadFile = File(...),
    only_prediction: bool = False,
    api_key: bool = Depends(verify_api_key)
):
    """CSV 파일 일괄 추론"""
    try:
        # CSV 파일 읽기
        contents = await file.read()

        # 임시 파일로 저장
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            df = read_csv_file(Path(tmp_path))
        finally:
            os.unlink(tmp_path)

        # 예측 수행
        predictions = []
        for idx, row in df.iterrows():
            text = str(row.get("text", ""))
            title = str(row.get("title", "")) if pd.notna(
                row.get("title", "")) else ""

            prediction, confidence = predict_single_text(text, title)

            if only_prediction:
                predictions.append({
                    "id": idx,
                    "prediction": prediction
                })
            else:
                predictions.append({
                    "id": idx,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": prediction,
                    "confidence": confidence
                })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        logger.error(f"Error in CSV inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model")
async def reload_model(api_key: bool = Depends(verify_api_key)):
    """모델(best.pt) 다시 불러오기"""
    try:
        load_model(MODEL_PATH)
        return {
            "status": "success",
            "message": f"Model reloaded from {MODEL_PATH}",
            "model_path": str(MODEL_PATH),
            "vocab_size": len(vocab) if vocab else 0
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=f"Model file not found: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error reloading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate")
async def validate(api_key: bool = Depends(verify_api_key)):
    """검증 데이터 평가"""
    if model is None or vocab is None or tokenizer is None:
        raise HTTPException(
            status_code=500, detail="Model not loaded. Please call /reload_model first.")

    if not VALIDATION_CSV_PATH.exists():
        raise HTTPException(
            status_code=404, detail=f"Validation CSV not found: {VALIDATION_CSV_PATH}")

    try:
        # 검증 데이터 로드
        val_df = read_csv_file(VALIDATION_CSV_PATH)

        # 예측 수행
        y_true = []
        y_pred = []

        for _, row in val_df.iterrows():
            text = str(row.get("text", ""))
            title = str(row.get("title", "")) if pd.notna(
                row.get("title", "")) else ""
            label = str(row.get("label", "")).lower()

            # 레이블을 숫자로 변환
            true_idx = LABEL_TO_IDX.get(label, 0)
            y_true.append(true_idx)

            # 예측
            prediction, _ = predict_single_text(text, title)
            pred_idx = LABEL_TO_IDX[prediction]
            y_pred.append(pred_idx)

        # 메트릭 계산
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # 클래스별 메트릭
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        return {
            "status": "success",
            "metrics": {
                "accuracy": float(accuracy),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
                "precision_per_class": {
                    "real": float(precision_per_class[0]),
                    "fake": float(precision_per_class[1])
                },
                "recall_per_class": {
                    "real": float(recall_per_class[0]),
                    "fake": float(recall_per_class[1])
                },
                "f1_per_class": {
                    "real": float(f1_per_class[0]),
                    "fake": float(f1_per_class[1])
                }
            },
            "num_samples": len(y_true),
            "label_distribution": {
                "real": int(sum(1 for y in y_true if y == 0)),
                "fake": int(sum(1 for y in y_true if y == 1))
            }
        }

    except Exception as e:
        logger.error(f"Error in validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
