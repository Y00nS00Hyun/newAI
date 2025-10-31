"""
데이터셋을 train / validation으로 split하여 저장
"""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from utils.dataset import read_csv_file

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Raw dataset 파일 경로
RAW_DATASETS = {
    "1": DATA_DIR / "dataset_1.csv",
    "2": DATA_DIR / "dataset_2.csv",
    "3": DATA_DIR / "dataset_3.csv",
}


def load_raw_dataset(path: Path) -> pd.DataFrame:
    
    logger.info(f"Loading dataset: {path.name}")
    
    try:
        df = read_csv_file(path)
    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        raise
    
    if 'label' not in df.columns:
        raise ValueError(f"{path.name} must contain 'label' column")
    
    if 'text' not in df.columns:
        raise ValueError(f"{path.name} must contain 'text' column")
    
    if 'title' not in df.columns:
        logger.warning(f"{path.name} does not have 'title' column. Using empty string.")
        df['title'] = ""
    
    # 필요한 컬럼만 추출
    normalized_df = df[['title', 'text', 'label']].copy()
    
    # 결측치 처리
    normalized_df['title'] = normalized_df['title'].fillna("")
    normalized_df['text'] = normalized_df['text'].fillna("")
    
    logger.info(f"  - Loaded {len(normalized_df)} rows")
    
    return normalized_df


def merge_datasets(dataset_ids: List[str]) -> pd.DataFrame:
    
    dataframes = []
    
    for dataset_id in dataset_ids:
        if dataset_id not in RAW_DATASETS:
            raise ValueError(f"Invalid dataset ID: {dataset_id}. Available: {list(RAW_DATASETS.keys())}")
        
        path = RAW_DATASETS[dataset_id]
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        df = load_raw_dataset(path)
        dataframes.append(df)
    
    # 데이터셋 병합
    merged_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"\nMerged {len(dataset_ids)} datasets: total {len(merged_df)} rows")
    
    return merged_df


def split_dataset(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # 시드 설정
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Train/validation split
    split_idx = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:split_idx]
    val_df = df_shuffled[split_idx:]
    
    logger.info(f"\nDataset split (seed={seed}):")
    logger.info(f"  - Train ratio: {train_ratio:.1%}")
    logger.info(f"  - Train samples: {len(train_df)}")
    logger.info(f"  - Validation samples: {len(val_df)}")
    
    # 레이블 분포 확인
    train_label_dist = train_df['label'].value_counts()
    val_label_dist = val_df['label'].value_counts()
    
    logger.info(f"\nTrain label distribution:")
    for label, count in train_label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    logger.info(f"\nValidation label distribution:")
    for label, count in val_label_dist.items():
        logger.info(f"  - {label}: {count} ({count/len(val_df)*100:.1f}%)")
    
    return train_df, val_df


def save_datasets(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame,
    output_dir: Path = DATA_DIR
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    
    # CSV 파일 저장
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"\nDatasets saved:")
    logger.info(f"  - Train: {train_path}")
    logger.info(f"  - Validation: {val_path}")


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="Split raw datasets into train/validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 모든 데이터셋 사용 (기본 70:30 split)
  python utils/split_data.py --datasets all
  
  # 특정 데이터셋만 사용
  python utils/split_data.py --datasets 1,3
  
  # Train 비율 변경 (80:20 split)
  python utils/split_data.py --datasets all --train-ratio 0.8
  
  # 랜덤 시드 변경
  python utils/split_data.py --datasets all --seed 123
        """
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Datasets to use (comma-separated IDs or 'all'). Available: 1,2,3 (default: all)"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train data ratio (default: 0.7)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Output directory (default: {DATA_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        # 데이터셋 ID 파싱
        if args.datasets.lower() == "all":
            dataset_ids = list(RAW_DATASETS.keys())
        else:
            dataset_ids = [id.strip() for id in args.datasets.split(",")]
        
        logger.info(f"Selected datasets: {dataset_ids}")
        
        # 데이터셋 병합
        merged_df = merge_datasets(dataset_ids)
        
        # Train/validation split
        train_df, val_df = split_dataset(
            merged_df, 
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # 파일 저장
        output_dir = Path(args.output_dir)
        save_datasets(train_df, val_df, output_dir)
        
        logger.info("\n✓ Data split completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

