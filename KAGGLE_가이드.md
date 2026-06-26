# Voice Generator v3 — Kaggle 사용 가이드

---

## 목차

1. [전체 흐름 한눈에 보기](#1-전체-흐름-한눈에-보기)
2. [준비물](#2-준비물)
3. [Kaggle 계정 및 API 키 발급](#3-kaggle-계정-및-api-키-발급)
4. [Dataset 업로드](#4-dataset-업로드)
5. [Kaggle Notebook 생성 및 설정](#5-kaggle-notebook-생성-및-설정)
6. [설정값 수정 (필수)](#6-설정값-수정-필수)
7. [실행 방법](#7-실행-방법)
8. [저장 위치 및 결과물](#8-저장-위치-및-결과물)
9. [세션 종료 후 이어서 학습하기](#9-세션-종료-후-이어서-학습하기)
10. [평가 및 추론](#10-평가-및-추론)
11. [전체 경로 구조](#11-전체-경로-구조)
12. [자주 발생하는 오류와 해결법](#12-자주-발생하는-오류와-해결법)

---

## 1. 전체 흐름 한눈에 보기

```
[준비]
  1. Kaggle 가입 + API 키 발급
  2. Dataset 2개 생성:
       ① voice-pgd-generator  ← 코드 + campplus.onnx + kss/ 업로드
       ② voice-pgd-checkpoints ← 빈 Dataset (체크포인트 자동 저장용)

[설정]
  3. run_kaggle.py 상단 3줄 수정 (본인 아이디, API 키)
  4. kaggle_notebook.ipynb 경로 수정

[실행]
  5. Notebook 생성 → GPU T4 켜기 → Dataset 연결
  6. kaggle_notebook.ipynb 셀 순서대로 실행

[결과]
  7. best.pt → voice-pgd-checkpoints Dataset에 자동 저장
  8. 세션 종료 후 --resume으로 이어서 학습
```

---

## 2. 준비물

| 파일/폴더 | 설명 | 비고 |
|-----------|------|------|
| `kss/` | 한국어 음성 데이터셋 (12,853개 wav) | 약 4~5GB |
| `campplus.onnx` | 화자 임베딩 모델 | CosyVoice3 pretrained 폴더 안에 있음 |
| `v3/` 코드 파일 전체 | 이 zip 안의 py 파일들 | — |

**campplus.onnx 위치 예시:**
```
pretrained_models/Fun-CosyVoice3-0.5B-2512/campplus.onnx
```

---

## 3. Kaggle 계정 및 API 키 발급

### 3-1. 계정 만들기

1. [https://www.kaggle.com](https://www.kaggle.com) → **Register**
2. 이메일 가입 후 이메일 인증 완료

### 3-2. API 키 발급

1. 로그인 후 오른쪽 위 프로필 아이콘 → **Settings**
2. 스크롤 내려서 **API** 섹션 → **Create New Token**
3. `kaggle.json` 파일이 자동 다운로드됨

`kaggle.json` 내용:
```json
{
  "username": "내아이디",
  "key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}
```

> `username`이 본인 Kaggle 아이디, `key`가 API 토큰이다.  
> 이 두 값을 나중에 설정 파일에 입력해야 한다.

---

## 4. Dataset 업로드

Kaggle에서는 파일을 **Dataset**이라는 저장소에 올려두고 Notebook에서 불러쓴다.  
총 **2개의 Dataset**을 만들어야 한다.

---

### Dataset ① — voice-pgd-generator (코드 + 데이터)

코드 파일, campplus.onnx, kss 음성 데이터를 모두 여기에 올린다.

#### 웹 브라우저로 업로드

1. [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. 이름: `voice-pgd-generator`
3. 아래 파일들을 전부 업로드:
   - zip 안의 `.py` 파일 전체 (14개)
   - `campplus.onnx`
   - `kss/` 폴더 (zip으로 압축 후 업로드 — Kaggle이 자동으로 압축 해제)
4. **Create** 클릭

업로드 완료 후 Kaggle 내부 경로:
```
/kaggle/input/datasets/본인아이디/voice-pgd-generator/
├── dataset.py
├── features.py
├── psycho.py
├── model.py
├── loss.py
├── train_utils.py
├── train_epoch.py
├── train.py
├── evaluate.py
├── infer.py
├── run_generator_train.py
├── run_kaggle.py
├── campplus.onnx
└── kss/
    ├── 1/
    │   ├── 1_0000.wav
    │   └── ...
    └── ...
```

> **주의:** `kss/` 폴더가 크기 때문에 업로드에 시간이 걸린다 (수십 분).  
> 업로드 중 브라우저를 닫지 말 것.

---

### Dataset ② — voice-pgd-checkpoints (체크포인트 자동 저장용)

학습 중 best.pt가 갱신될 때마다 여기에 자동 업로드된다.  
**처음에는 빈 Dataset으로 만들면 된다.**

1. [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. 이름: `voice-pgd-checkpoints`
3. 파일 없이 그냥 **Create**

---

## 5. Kaggle Notebook 생성 및 설정

### 5-1. 새 Notebook 만들기

1. [https://www.kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**

### 5-2. GPU 설정 (필수)

오른쪽 사이드바 → **Session options** 또는 **Settings**:

| 항목 | 설정값 |
|------|--------|
| Accelerator | **GPU T4 x1** (또는 P100) |
| Internet | **On** |

> GPU를 켜지 않으면 CPU로 돌아가 학습이 매우 느리다.  
> 상단에 `device : cuda`가 출력되면 GPU가 잡힌 것.

### 5-3. Dataset 연결

오른쪽 사이드바 **Input** 섹션 → **Add data**:

- `voice-pgd-generator` 검색 → **Add**
- `voice-pgd-checkpoints` 검색 → **Add** (이어서 학습할 경우)

연결 후 Notebook 안에서 접근 가능한 경로:
```
/kaggle/input/datasets/본인아이디/voice-pgd-generator/   ← 코드 + 데이터
/kaggle/input/본인아이디/voice-pgd-checkpoints/          ← 이전 체크포인트
```

### 5-4. 노트북 파일 사용

이 zip 안의 `kaggle_notebook.ipynb`를 Notebook 편집기에 **Import** 하거나,  
셀 내용을 새 Notebook에 복사해서 쓰면 된다.

---

## 6. 설정값 수정 (필수)

**`run_kaggle.py` 상단 4곳을 본인 것으로 교체해야 한다.**

```python
# ── 설정 (본인 환경에 맞게 수정) ──────────────────────────────
PGD_PATH     = '/kaggle/input/datasets/본인아이디/voice-pgd-generator'
KSS_PATH     = '/kaggle/input/datasets/본인아이디/voice-pgd-generator/kss'
KAGGLE_TOKEN = '여기에_본인_kaggle_api_key_입력'   # kaggle.json의 "key" 값
```

그리고 `upload_checkpoint` 함수 안의 Dataset ID도 수정:

```python
"id": "본인아이디/voice-pgd-checkpoints",
```

**`kaggle_notebook.ipynb`** 안의 경로도 동일하게 수정:

```python
pgd_path = '/kaggle/input/datasets/본인아이디/voice-pgd-generator'
```

> `본인아이디`는 Kaggle username (`kaggle.json`의 `"username"` 값).  
> `KAGGLE_TOKEN`은 `kaggle.json`의 `"key"` 값.

---

## 7. 실행 방법

### 방법 A — kaggle_notebook.ipynb 사용 (추천)

Notebook에서 셀을 순서대로 실행한다.

**셀 1 — 경로 확인:**
```python
import os, sys
pgd_path = '/kaggle/input/datasets/본인아이디/voice-pgd-generator'
kss_path = '/kaggle/input/datasets/본인아이디/voice-pgd-generator/kss'
sys.path.append(pgd_path)
print('pgd:', os.listdir(pgd_path))
print('kss exists:', os.path.exists(kss_path))
```
> `campplus.onnx`, `kss` 등이 출력되면 Dataset이 제대로 연결된 것.

**셀 2 — 의존성 설치:**
```python
!pip install onnx onnx2torch soundfile scipy -q
```
> 1~2분 걸린다.

**셀 3 — 학습 실행:**
```python
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/run_kaggle.py
```
> `run_kaggle.py`가 내부적으로 `run_generator_train.py`를 호출하며  
> 학습 + best.pt 자동 업로드까지 처리한다.

---

### 방법 B — run_generator_train.py 직접 실행 (체크포인트 자동 업로드 없음)

체크포인트 자동 업로드 기능 없이 단순 학습만 할 경우:

```python
!pip install onnx onnx2torch soundfile scipy -q
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/run_generator_train.py --epochs 10 --batch 16
```

---

### 실행 중 출력 예시

```
=======================================================
Generator 학습 시작
  device  : cuda          ← GPU가 잡혀야 함 (cpu면 설정 확인)
  KSS     : /kaggle/input/datasets/.../kss
  output  : /kaggle/working/checkpoints
  epochs  : 10
  batch   : 16
=======================================================
[1] CAM++ 로드...
[2] 모델 초기화...  Generator 파라미터: 54,967,473
[3] DataLoader 생성...  파일 수: 12853, 배치 수: 803
[4] 원본 임베딩 캐싱...  (첫 실행 시 10~20분 소요)
[5] 새 학습 시작
[6] 학습 (epoch 0 ~ 9)
[000] loss=-1.2885  dist=1.2885  snr_l=0.1756  lr=9.76e-05  87.3s
  ★ best.pt 저장 (dist=1.2885)
  [ckpt] best.pt uploaded to dataset    ← 자동 업로드 확인
[001] loss=-1.4409  dist=1.4409  ...
```

> `dist` 값이 클수록 방어 효과가 좋다. 0.3 이상이면 클로닝 방어 성공 수준.

---

## 8. 저장 위치 및 결과물

### 학습 중 생성되는 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `best.pt` | `/kaggle/working/checkpoints/best.pt` | dist 기준 가장 좋은 모델 |
| `last.pt` | `/kaggle/working/checkpoints/last.pt` | 마지막 저장 체크포인트 (resume용) |
| `train_log.json` | `/kaggle/working/checkpoints/train_log.json` | epoch별 loss/dist 수치 기록 |
| `emb_cache.pt` | `/kaggle/working/checkpoints/emb_cache.pt` | 임베딩 캐시 (재사용 시 시간 절약) |
| `train_curves.png` | `/kaggle/working/checkpoints/train_curves.png` | loss/dist 학습 곡선 그래프 |
| `quick_output/` | `/kaggle/working/checkpoints/quick_output/` | 학습 후 샘플 음성 보호 결과 |

### 자동 업로드 위치 (run_kaggle.py 사용 시)

best.pt가 갱신될 때마다 자동으로:
```
voice-pgd-checkpoints Dataset
└── best.pt   ← 항상 최신 best.pt로 덮어씀
```

### 결과물 직접 다운로드

Notebook 오른쪽 사이드바 **Output** 탭 → `checkpoints/` 폴더 → 파일 우클릭 → **Download**

---

## 9. 세션 종료 후 이어서 학습하기

Kaggle 무료 GPU는 **세션당 최대 12시간**이다.  
11시간 45분이 지나면 코드가 자동으로 `last.pt`를 저장하고 종료한다.

### 9-1. 이전 체크포인트 가져오기

새 세션을 시작하면 `/kaggle/working/`이 초기화된다.  
`voice-pgd-checkpoints` Dataset에 저장된 체크포인트를 가져와야 한다.

Notebook **Input** 탭 → `voice-pgd-checkpoints` Dataset 추가 후:

```python
import os, shutil
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# best.pt 복사
shutil.copy(
    '/kaggle/input/voice-pgd-checkpoints/best.pt',
    '/kaggle/working/checkpoints/best.pt'
)
# last.pt도 있으면 복사 (resume 기준)
last_src = '/kaggle/input/voice-pgd-checkpoints/last.pt'
if os.path.exists(last_src):
    shutil.copy(last_src, '/kaggle/working/checkpoints/last.pt')

print('체크포인트 복사 완료')
print(os.listdir('/kaggle/working/checkpoints/'))
```

### 9-2. 이어서 학습 실행

`run_kaggle.py`에 `--resume` 인자를 추가해서 실행:

```python
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/run_kaggle.py --resume
```

> `--resume`이 없으면 처음부터 다시 학습하니 반드시 붙일 것.

### 9-3. 임베딩 캐시 재사용 (시간 절약)

처음 학습 시 원본 임베딩 캐싱에 10~20분이 걸린다.  
`emb_cache.pt`를 Dataset에 같이 올려두면 다음 세션에서 즉시 재사용된다.

```python
# 첫 학습 후 emb_cache.pt를 Dataset에 수동 업로드하거나
# 아래처럼 voice-pgd-generator Dataset에 추가 버전으로 올리면 됨
# (용량이 크지 않아 업로드 빠름)
```

---

## 10. 평가 및 추론

### 학습 결과 수치 확인

```python
import json

with open('/kaggle/working/checkpoints/train_log.json') as f:
    log = json.load(f)

for e in log:
    print(f"epoch {e['epoch']:03d} | dist={e['dist']:.4f} | snr_l={e['snr']:.4f} | lr={e['lr']:.2e}")
```

### 방어 효과 평가 (KSS 전체 또는 일부)

```python
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/evaluate.py \
    --checkpoint /kaggle/working/checkpoints/best.pt \
    --wav_dir /kaggle/input/datasets/본인아이디/voice-pgd-generator/kss \
    --max_files 50
```

> `--max_files 50`: 전체 12,853개 중 50개만 평가. 빠른 확인용.  
> 전체 평가는 `--max_files` 생략.

출력 예시:
```
평가 결과 요약
  파일 수         : 50
  CAM++ dist 평균 : 1.0725  (클수록 방어 성공)
  SNR 평균        : 26.86 dB  (기준: >= 28dB)
  L-inf 평균      : 0.009981  (기준: <= 0.01)
  SNR >= 28dB 만족: 38/50
  L-inf <= eps 만족: 50/50
```

### 단일 파일 추론 (보호된 음성 생성)

```python
# hard_project 적용 (기본 — 심리음향 마스킹 후처리)
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/infer.py \
    --checkpoint /kaggle/working/checkpoints/best.pt \
    --input /kaggle/input/datasets/본인아이디/voice-pgd-generator/kss/1/1_0000.wav

# hard_project 없이 (dist 더 높지만 가청 노이즈 존재)
!python /kaggle/input/datasets/본인아이디/voice-pgd-generator/infer.py \
    --checkpoint /kaggle/working/checkpoints/best.pt \
    --input /kaggle/input/datasets/본인아이디/voice-pgd-generator/kss/1/1_0000.wav \
    --no_psycho
```

출력 파일은 `/kaggle/working/1_0000_protected.wav`로 저장된다.

---

## 11. 전체 경로 구조

```
/kaggle/
│
├── input/
│   ├── datasets/본인아이디/voice-pgd-generator/   ← Dataset ①
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── psycho.py
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── train_utils.py
│   │   ├── train_epoch.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── infer.py
│   │   ├── run_generator_train.py
│   │   ├── run_kaggle.py
│   │   ├── campplus.onnx
│   │   └── kss/
│   │       ├── 1/ (wav 파일들)
│   │       ├── 2/
│   │       ├── 3/
│   │       └── 4/
│   │
│   └── voice-pgd-checkpoints/                    ← Dataset ② (자동 저장)
│       └── best.pt
│
└── working/                                       ← 실행 중 생성되는 파일
    └── checkpoints/
        ├── best.pt          ← 가장 좋은 모델
        ├── last.pt          ← 마지막 체크포인트 (resume용)
        ├── train_log.json   ← epoch별 수치 기록
        ├── emb_cache.pt     ← 임베딩 캐시
        ├── train_curves.png ← 학습 곡선 그래프
        └── quick_output/
            ├── original.wav
            └── protected.wav
```

---

## 12. 자주 발생하는 오류와 해결법

### `device : cpu` 출력 (GPU가 안 잡힘)

Notebook 설정에서 Accelerator가 None으로 되어 있는 것.  
Notebook 오른쪽 사이드바 → **Session options** → **Accelerator: GPU T4** 선택 후 재시작.

---

### `ModuleNotFoundError: No module named 'onnx2torch'`

의존성 설치 셀을 먼저 실행하지 않은 것.

```python
!pip install onnx onnx2torch soundfile scipy -q
```

---

### `FileNotFoundError: /kaggle/input/datasets/본인아이디/voice-pgd-generator/campplus.onnx`

Dataset이 연결되지 않았거나 경로가 틀린 것. 실제 경로 확인:

```python
import os
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))
```

출력된 실제 경로를 `run_kaggle.py`의 `PGD_PATH`에 맞게 수정.

---

### `AssertionError: WAV 파일 없음`

`KSS_PATH`가 잘못된 것. `kss/` 폴더 안에 `1/`, `2/`, `3/`, `4/` 서브폴더가 있어야 한다.

```python
import os
print(os.listdir('/kaggle/input/datasets/본인아이디/voice-pgd-generator/kss'))
# 출력 예시: ['1', '2', '3', '4']
```

---

### `[ckpt] upload failed`

`KAGGLE_TOKEN`이 잘못된 것. `kaggle.json`의 `"key"` 값을 다시 확인.  
또는 `voice-pgd-checkpoints` Dataset이 아직 생성되지 않은 것 → Dataset ② 먼저 만들 것.

---

### 세션이 12시간 안에 끊기는 경우

Kaggle 무료 계정은 주 30시간 GPU 제한이 있다.  
제한에 걸리면 다음날 이어서 학습하면 된다. [9. 세션 종료 후 이어서 학습하기](#9-세션-종료-후-이어서-학습하기) 참고.

---

### `⚠ last.pt 존재하지만 --resume 없이 실행됨` 경고

이어서 학습하려면 `--resume` 필요:
```python
!python .../run_kaggle.py --resume
```
처음부터 다시 하려면 무시해도 된다.
