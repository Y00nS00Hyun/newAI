# 데이터셋 사용 가이드

## 모든 데이터셋 병합 사용 (권장)

모든 데이터셋을 병합하여 사용하면 성능이 향상됩니다:

```bash
# 모든 데이터셋 병합 후 train/validation 분할
python utils/split_data.py --datasets all --train-ratio 0.8

# 학습 실행
python train.py --model cnn --config configs/cnn.yaml --device cuda
```

### 병합된 데이터셋의 장점:

1. **더 많은 데이터**: 32,970개 (32,470 + 250 + 250)
2. **다양한 텍스트 길이**:
   - 짧은 텍스트 (dataset_3: 평균 342자) → 전체 텍스트 활용
   - 긴 텍스트 (dataset_1,2: 평균 2483-2584자) → 핵심 패턴 포착
3. **클래스 불균형 완화**: 각 데이터셋의 서로 다른 분포가 균형을 맞춤
4. **더 다양한 패턴**: 짧은 트윗형부터 긴 기사형까지 학습

### 예상 라벨 분포:
- dataset_1만 사용: Fake 54% / Real 46%
- 모든 데이터셋 병합: 약간 더 균형잡힌 분포 (자동으로 가중치 계산됨)

## 개별 데이터셋 사용

특정 데이터셋만 사용하고 싶은 경우:

```bash
# dataset_1만 사용 (기본)
python utils/split_data.py --datasets 1 --train-ratio 0.8

# dataset_1과 dataset_3만 사용
python utils/split_data.py --datasets 1,3 --train-ratio 0.8
```

## 주의사항

1. **클래스 가중치**: `train.py`에서 자동으로 계산되므로 수동 조정 불필요
2. **max_len**: 512로 설정되어 있어 짧은 텍스트는 전체, 긴 텍스트는 일부만 사용
3. **메타정보**: dataset_2의 `subject`, `date`는 현재 텍스트에 포함되지 않음 (필요시 수정 가능)

