# Colab에서 New_Sample 데이터로 학습하기

AI Hub "AI챗봇 자유대화(일반남여)" 데이터(4화자, wav 2,000개, 16kHz mono)로
PerturbationGenerator를 새로 학습하는 절차.

데이터가 KSS와 동일한 16kHz mono이고 `dataset.py`가 폴더를 재귀 탐색하므로,
**코드 본체는 수정하지 않고** `run_colab.py` 드라이버만 사용한다.

---

## 0. 미리 준비할 것

| 준비물 | 설명 |
|--------|------|
| `voice_generator` 코드 폴더 | 이 폴더 전체 (run_colab.py 포함) |
| `New_Sample`의 `원천데이터` 폴더 | wav 2,000개 (`라벨링데이터`는 학습에 불필요) |
| **`campplus.onnx`** | CAM++ 화자인식 ONNX 모델 — **반드시 별도 확보** (아래 주의 참고) |
| Colab 런타임 | 런타임 → 런타임 유형 변경 → **GPU(T4)** 선택 |

> ⚠ **campplus.onnx가 핵심 블로커입니다.** 이 파일이 없으면 학습이 불가능합니다.
> v3 학습 때 쓰던 것과 동일한 파일이며, 원래 CosyVoice의
> `pretrained_models/Fun-CosyVoice3-0.5B-2512/campplus.onnx` 에 들어있습니다.
> 가지고 계신 그 파일을 Drive에 올려두세요.

---

## 1. 데이터를 Google Drive에 올리기

`원천데이터` 폴더와 `campplus.onnx`를 Drive에 배치한다. 권장 구조:

```
MyDrive/
├── New_Sample/
│   └── 원천데이터/...           ← wav 2,000개 (하위 폴더째 업로드)
├── voice_pgd/
│   └── campplus.onnx
└── voice_gen_ckpt/             ← (자동 생성) 체크포인트 저장 위치
```

> 폴더째 업로드가 느리면, 로컬에서 `원천데이터`를 zip으로 묶어 올린 뒤
> Colab에서 `!unzip`으로 푸는 게 빠르다.

---

## 2. Colab 노트북 셀 순서

### 셀 1 — Drive 마운트
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 셀 2 — 코드 폴더 배치
코드(voice_generator)도 Drive에 올려뒀다면 복사만 하면 된다.
```python
!cp -r "/content/drive/MyDrive/voice_generator" /content/voice_generator
!ls /content/voice_generator
```
(또는 GitHub에 있다면 `!git clone ...`)

### 셀 3 — 의존성 설치
```python
!pip install -q onnx onnx2torch soundfile
# torch / torchaudio / scipy / numpy / matplotlib 은 Colab에 기본 설치돼 있음
```

### 셀 4 — 경로 확인 후 학습 실행
`run_colab.py` 상단의 4개 경로(CODE_DIR / DATA_DIR / CAMPPLUS / OUTPUT_DIR)가
1번에서 만든 Drive 구조와 맞는지 확인한 뒤 실행한다.
```python
!python /content/voice_generator/run_colab.py --epochs 30 --batch 16
```

### 셀 5 — (런타임이 끊겼을 때) 이어서 학습
체크포인트가 Drive(OUTPUT_DIR)에 epoch마다 저장되므로 그대로 재개 가능.
```python
!python /content/voice_generator/run_colab.py --resume
```

---

## 3. 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--epochs` | 10 | 총 epoch (result.md 기준 4~5epoch부터 수렴 시작) |
| `--batch` | 16 | T4면 16 권장, OOM 나면 8로 |
| `--lr` | 1e-4 | 초기 학습률 (CosineAnnealing) |
| `--resume` | off | last.pt에서 이어서 학습 |
| `--skip_infer` | off | 학습 후 샘플 추론 생략 |
| `--save_interval` | 1 | 몇 epoch마다 last.pt 저장 |

---

## 4. 학습 결과물 (OUTPUT_DIR에 생성)

```
voice_gen_ckpt/
├── best.pt              최고 dist 모델 (추론에 사용)
├── last.pt              마지막 체크포인트 (resume용)
├── emb_cache.pt         원본 임베딩 캐시 (재실행 시 재사용 → 시간 절약)
├── train_log.json       epoch별 metrics
├── train_curves.png     loss/dist/snr 곡선
└── quick_output/        샘플 보호 wav (original.wav / protected.wav)
```

추론(보호 적용)은 학습 후:
```python
!python /content/voice_generator/infer.py \
    --checkpoint /content/drive/MyDrive/voice_gen_ckpt/best.pt \
    --input  some_voice.wav \
    --output protected.wav
```

---

## 5. 이 데이터에서 알아둘 점

- **다화자(4명)**: KSS는 단일화자였지만, 학습은 파일별 원본 임베딩을 캐싱해
  화자와 무관하게 동작하므로 4화자 혼합도 그대로 학습된다. 오히려 다양한
  목소리에 일반화된 Generator가 될 수 있다.
- **~5초 길이**: `dataset.py`가 4초로 랜덤 trim(초과) / zero-pad(미만) 처리하므로
  추가 작업 불필요.
- **JSON 라벨 불필요**: 이 프로젝트는 STT 텍스트를 학습에 쓰지 않는다. wav만 사용.
  (라벨의 stt는 나중에 CosyVoice 클로닝 평가용 텍스트로만 쓸 수 있음.)
- **첫 실행 시 임베딩 캐싱**에 시간이 걸린다(2,000개 CAM++ forward). 이후
  `emb_cache.pt`가 Drive에 저장되어 재실행 시 건너뛴다.
