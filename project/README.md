# 참가자 가이드

---

## 0. 빠른 시작(5분)

1. **접속**

   * 제공된 URL 접속 → 상단 버튼으로

     * **code-server**(웹 IDE): 비밀번호 입력
     * **Swagger UI**(API 테스트): `Authorize` 클릭 후 **API Key** 입력

2. **데이터 분할**

   ```bash
   # code-server의 터미널에서
   python utils/split_data.py --ratio 0.8
   ```

   * 결과: `data/train.csv`, `data/validation.csv` 생성
   * 원본 제공 파일명: **`dataset_1.csv`, `dataset_2.csv`, `dataset_3.csv`**

3. **학습**

   ```bash
   # 예) BiLSTM
   python train.py --model bilstm --device cuda
   ```

   * 학습 결과물은 `models/` 아래에 저장됩니다.
   * **제출 대상 파일명은 반드시 `models/best.pt`** 여야 합니다. (이름만 정확하면 방식은 자유)

4. **검증/테스트 (Swagger UI)**

   * `POST /reload_model` : 최신 `models/best.pt` 반영
   * `POST /validate` : 내부 `validation.csv`로 점수 확인
   * `POST /infer_csv` : CSV 업로드로 일괄 예측 확인

5. **제출**

   * 랜딩 페이지의 **“모델 제출”** 섹션에서 안내된 폼을 작성 후 제출

---

## 1. 작업 환경 & 디렉터리

* 참가자는 브라우저로 접속한 **code-server**의 작업 루트에서만 작업합니다.
* 주요 경로(요약):

  ```
    configs/              # 학습 설정 (예: bilstm.yaml)
    data/                 # CSV 데이터(분할 결과 저장 위치)
    model_definitions/    # 모델 클래스들 (새 모델 추가 가능)
    models/               # 학습된 가중치 저장 위치 (제출 대상: best.pt)
    public/               # 안내 페이지 정적 리소스 (참고용)
    utils/
      split_data.py       # train/validation 분할 스크립트
      dataset.py          # 로더/전처리
      vocab.py            # 토크나이저/보캡
    baseline.ipynb        # 전체 흐름 예제 노트북
    train.py              # 학습 스크립트
    requirements.txt
  ```

---

## 2. 데이터셋

* 제공 파일명: **`dataset_1.csv`, `dataset_2.csv`, `dataset_3.csv`** (이미 환경에 준비되어 있음)
* 먼저 **분할 스크립트**를 실행해 학습/검증 세트를 만듭니다.

  ```bash
  python utils/split_data.py --ratio 0.8
  # 생성: data/train.csv, data/validation.csv
  ```
* 기본 컬럼은 `title, text, label` 입니다.
  그 외 데이터셋에 포함된 정보를 활용하는 것은 자유입니다.
  단, 이러한 수정으로 인해 발생하는 오류에 대해서는 **어떠한 안내도 제공되지 않습니다.**

> **팁:** 처음에는 **baseline.ipynb** 를 그대로 따라 하며 흐름을 익히는 것을 권장합니다.
> **안내:** 원본 데이터셋 (`dataset_1.csv`, `dataset_2.csv`, `dataset_3.csv`)을 수정할 경우, 원본 파일을 **다시 제공하지 않습니다.** 데이터셋 수정으로 인해 발생하는 모든 책임은 본인에게 있습니다.

---

## 3. 모델 개발

### 3.1 기본 학습

```bash
python train.py --model bilstm --device cuda
```

* `--model` 은 등록된 모델 이름을 사용합니다.
* config를 지정하지 않으면, 기본적으로 모델명과 동일한 이름의 config를 불러옵니다.
  동일한 이름의 config가 없을 시, 에러가 발생합니다.
* 결과 파일은 `models/` 아래에 저장됩니다.
  **제출 대상 파일은 이름이 `models/best.pt` 여야 합니다.**

### 3.2 새 모델 추가

1. 새 파일 작성: `model_definitions/my_model.py`
2. **레지스트리 등록 (필수):** `model_definitions/__init__.py` 편집

   ```python
   from model_definitions.my_model import MyModel
   MODEL_REGISTRY = {
       "bilstm": BiLSTM,
       "my_model": MyModel,
   }
   ```
3. 학습 실행:

   ```bash
   python train.py --model my_model --device cuda
   ```

* **등록을 빼먹으면** `--model` 파라미터 에러 또는 KeyError가 발생합니다.
> **팁:** 처음에는 **baseline.ipynb** 를 그대로 따라 하며 흐름을 익히는 것을 권장합니다.
---

## 4. 검증 & 테스트 (Swagger UI)

> Swagger 우측 상단 **Authorize** 버튼을 눌러 **API Key** 를 먼저 등록하세요.

1. **모델 반영**

   * `POST /reload_model`
     최신 `models/best.pt` 를 메모리에 로드합니다.

2. **내부 검증**

   * `POST /validate`
     내부 `data/validation.csv` 로 **accuracy / precision / recall / macro-F1 / 샘플 수** 등을 반환합니다.
     `data/validation.csv` 이외 임의 경로를 지정할 수 없습니다.

3. **CSV 일괄 예측**

   * `POST /infer_csv`
     `multipart/form-data` 로 CSV 업로드 → 예측 결과를 반환합니다.

     * 옵션: `only_prediction=true` 를 사용하면 `id,prediction` 형식만 반환

> 평가는 제출 후 평가 서버가 `/infer_csv` 를 호출하여 진행됩니다.

---

## 5. 제출

* 랜딩 페이지의 “모델 제출” 섹션에서
  **평가 서버 URL / 참가자 서버 URL / 팀명 / 제출 비밀번호** 를 입력하고 제출합니다.
* **평가 서버는 참가자 서버의 GPU를 사용**하여 추론을 수행합니다.
  평가가 끝날 때까지 **GPU가 점유**되므로, 동시에 무거운 작업을 돌리면 속도가 느려질 수 있습니다.

### 5-1. 평가 지표
* 평가는 **Macro F1 Score**로 집계합니다.
* 탐지 대상(가짜, fake)를 양성으로 둡니다.
$pos(positive) = Fake(라벨 1) / neg(negative) = Real(라벨0)$

* 동점 시 세부 평가를 통해 순위를 확정합니다. 관련 내용은 동점자 발생 시 추후 안내 예정입니다.

$$
{F1}_{pos} = \frac{2TP}{2TP+FP+FN}
$$

$$
{F1}_{neg} = \frac{2TN}{2TN+FP+FN}
$$

$$
\text{MacroF1} = \frac{{F1}_{pos} + {F1}_{neg}}{2}
$$

---

## 6. 제약사항 (반드시 준수)

* **모델 크기 ≤ 10 GB**
* **추론 시간 ≤ 30분** *(초과 시 제출 실패)*
* **사전 학습(Pretrained) 모델 사용 금지**
* **외부 API 사용 금지** (예: OpenAI, Google 등)
* **외부 데이터셋 추가 금지** (제공된 데이터셋 이외 추가로 데이터셋을 다운받는 등의 행위 금지)
* 위 항목 이외에는 **자유**입니다. 규정을 해치지 않는 한,
  전처리/아키텍처/학습 전략 등은 모두 참가자 재량입니다.

---

## 7. 자주 발생하는 이슈 (체크리스트)

* [ ] **API Key 미설정** → Swagger에서 **Authorize** 먼저
* [ ] **데이터 분할 누락** → `data/train.csv`, `data/validation.csv` 없는 상태로 학습 시 **실패**
* [ ] **모델 레지스트리 누락** → `--model my_model` 실행 불가
* [ ] **best.pt 반영 누락** → `POST /reload_model` 호출 전이면 **이전 모델로 검증됨**
* [ ] **시간/자원 초과** → 모델/배치/전처리 등을 조정하여 제한 내에서 동작하도록 튜닝

---

## 8. 참고

* **baseline.ipynb** 에 “분할 → 학습”의 전체 흐름이 정리되어 있습니다.
  처음에는 노트북을 따라 하며 정상 동작을 확인하세요.

---

궁금한 점은 운영진에게 문의주세요.
좋은 성과를 기원합니다!
