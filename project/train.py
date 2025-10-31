"""
모델 학습 스크립트
"""
import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from tqdm import tqdm

from model_definitions import MODEL_REGISTRY
from utils.dataset import build_dataloaders
from utils.augmentation import apply_augmentation
from sklearn.metrics import precision_score, recall_score, f1_score
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_one_epoch(model, train_loader, optimizer, criterion, device, max_norm=5.0):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    total_examples = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        inputs, lengths, labels = batch
        inputs, lengths, labels = apply_augmentation(inputs, lengths, labels)
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / max(total_examples, 1)


def evaluate(model, val_loader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            inputs, lengths, labels = batch
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    
    avg_loss = total_loss / max(total_examples, 1)
    accuracy = correct / max(total_examples, 1)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return avg_loss, accuracy, precision, recall, f1


def train_model(model_name: str, config_path: str, device: str = None):
    """모델 학습 메인 함수"""
    # 설정 로드
    config = load_config(config_path)
    
    # 디바이스 설정
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # 데이터 로더 생성
    logger.info("Loading data...")
    train_path = Path(config['data']['train_path'])
    val_path = Path(config['data']['val_path'])
    
    train_loader, val_loader, vocab, tokenizer = build_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        num_workers=config['num_workers'],
        max_vocab_size=config['max_vocab_size'],
        text_fields=config['data']['text_fields'],
    )
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # 모델 생성
    logger.info(f"Creating model: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    model_config = config.get('model', config.get('models', {}).get(model_name, {}))
    model = model_cls(
        vocab_size=len(vocab),
        num_classes=2,
        **model_config
    )
    model = model.to(device)
    
    # 옵티마이저 및 스케줄러
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    logger.info(f"Starting training for {config['epochs']} epochs...")
    best_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['epochs']}")
        
        # 학습
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 평가
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step(val_accuracy)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # Best model 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            
            model_dir = Path(config['output']['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = model_dir / f"{model_name}_best.pt"
            torch.save({
                'model': model,
                'vocab': {
                    'token_to_idx': vocab.token_to_idx,
                    'idx_to_token': vocab.idx_to_token,
                    'max_size': vocab.max_size,
                    'min_freq': vocab.min_freq,
                },
                'model_name': model_name,
                'model_config': model_config,
                'train_config': {
                    'max_len': config['max_len'],
                    'text_fields': config['data']['text_fields'],
                },
                'accuracy': best_accuracy,
            }, save_path)
            logger.info(f"✓ Best model saved: {save_path} (accuracy: {best_accuracy:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    logger.info(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")
    
    # 최종 모델을 best.pt로 복사 (제출용)
    model_dir = Path(config['output']['model_dir'])
    best_model_path = model_dir / f"{model_name}_best.pt"
    final_best_path = model_dir / "best.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_best_path)
        logger.info(f"✓ Final model saved as: {final_best_path}")
    else:
        logger.warning(f"Best model file not found: {best_model_path}")
    
    return model, best_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train fake news detection model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: configs/{model_name}.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda)"
    )
    
    args = parser.parse_args()
    if args.config is None:
        config_path = f"configs/{args.model}.yaml"
    else:
        config_path = args.config
    
    # 학습 실행
    try:
        train_model(args.model, config_path, args.device)
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
