# Voice Generator v3 — 전체 결과 정리

## 프로젝트 개요

**목표:** PGD adversarial perturbation으로 CosyVoice3 음성 클로닝 방어  
**핵심 아이디어:** 실시간 perturbation Generator 신경망 학습 → 추론 시 ~0.01s/파일  
**실행 환경:** Kaggle T4 GPU (16GB VRAM, 12h 세션)  
**데이터셋:** KSS (한국어 단일 화자, 12,853개 발화, 평균 3~5초)  
**체크포인트:** `v3_best.pt` (630MB, epoch 9)

---

## 1. Generator 방식이란

### 왜 PGD가 아닌 Generator인가

기존 PGD(Projected Gradient Descent) 방식은 파일 1개를 보호할 때마다 수십~수백 번의 최적화 스텝을 반복해야 한다. 파일당 수 초~수십 초가 걸리며, 실시간 처리가 불가능하다.

Generator 방식은 **"perturbation을 만드는 신경망"을 한 번 학습해 두면**, 이후 추론 시에는 신경망 forward pass 1회(~0.01초)만으로 보호 perturbation을 즉시 생성한다.

```
학습 시: 신경망 파라미터를 gradient로 업데이트 (느림, 1회만)
추론 시: 신경망 forward 1회 → perturbation 즉시 생성 (빠름, 매번)
```

### 학습 신호 연결 구조

Generator가 perturbation을 잘 만들도록 학습하려면, Generator의 파라미터 → delta → adv wav → CAM++ → 임베딩 → cosine dist까지 미분 가능해야 한다.

CAM++는 onnx2torch로 PyTorch 모델로 변환되어 있어 gradient가 통한다. RL 없이 순수 gradient-based 학습이 가능한 이유다.

```
원본 wav (detach)
    │
    ├─ mel 추출 (no_grad + detach) ──► Generator ──► delta (grad 있음)
    │                                                     │
    └─────────────────────────────────────────────────────┤
                                                          ▼
                                                    adv = orig + delta
                                                          │
                                                     CAM++ forward
                                                          │
                                                    adv 임베딩 (grad 있음)
                                                          │
                                          cosine_dist(adv_emb, orig_emb_cached)
                                                          │
                                                    L_cam = -mean(dist)
                                                          │
                                                    total_loss.backward()
                                                          │
                                               Generator.parameters() 업데이트
```

원본 임베딩(`orig_emb`)은 학습 시작 전 1회 사전 캐싱 → 배치마다 꺼내 씀 (CAM++ forward 중복 제거).

---

## 2. 모델 설계: PerturbationGenerator

**파일:** `model.py`  
**파라미터 수:** 54,967,473 (약 55M)  
**입력:** mel spectrogram `(B, 1, 80, 401)`  
**출력:** perturbation delta `(B, 64000)`, range `[-eps, +eps]`

### 아키텍처: U-Net

mel 도메인에서 U-Net으로 특징 추출 → 시간 도메인 waveform으로 복원.

```
입력: mel (B, 1, 80, 401)
         │
    ┌────▼─────────────────────────────────────────────────┐
    │  Encoder (3단계, 각 단계: Conv×2 + stride-2 다운샘플)  │
    │                                                       │
    │  enc1: (B,  1, 80, 401) → skip(B, 32, 80, 401)       │
    │                         + down(B, 32, 40, 200)        │
    │  enc2: (B, 32, 40, 200) → skip(B, 64, 40, 200)       │
    │                         + down(B, 64, 20, 100)        │
    │  enc3: (B, 64, 20, 100) → skip(B,128, 20, 100)       │
    │                         + down(B,128, 10,  50)        │
    └───────────────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │  Bottleneck: Conv×2                            │
    │  (B,128,10,50) → (B,256,10,50)                │
    └────────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────────────┐
    │  Decoder (3단계, 각 단계: 업샘플 + skip concat + Conv×2) │
    │                                                        │
    │  dec1: (B,256,10,50) + skip3 → (B,128,20,100)         │
    │  dec2: (B,128,20,100)+ skip2 → (B, 64,40,200)         │
    │  dec3: (B, 64,40,200)+ skip1 → (B, 32,80,401)         │
    └────────────────────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────────────────────┐
    │  TimeDomainProjector: mel → waveform 복원                        │
    │                                                                  │
    │  (B, 32, 80, 401)                                                │
    │    → reshape: (B, 32×80, 401) = (B, 2560, 401)                  │
    │    → ConvTranspose1d(stride=160): (B, 64, ~64160)                │
    │    → ConvTranspose1d(kernel=9): (B, 16, ~64160)                  │
    │    → Conv1d(kernel=7): (B, 1, ~64160) → squeeze → (B, ~64160)   │
    │    → [:64000] 슬라이스 or pad → (B, 64000)                       │
    │    → Tanh × eps → [-eps, +eps] 보장                              │
    └──────────────────────────────────────────────────────────────────┘
         │
    출력: delta (B, 64000)
```

**핵심 설계 포인트:**

| 항목 | 내용 |
|------|------|
| skip connection | U-Net 구조로 fine-grained 주파수 정보 보존 |
| H/W 불일치 보정 | DecoderBlock에서 pad/crop으로 자동 처리 |
| Tanh × eps | L-inf <= eps 수학적 보장 (학습/추론 모두) |
| t_out | 훈련: 64000 고정 / 추론: None (가변 길이) |

---

## 3. Loss 설계

**파일:** `loss.py`

```
Total = w_cam × L_cam + w_snr_dynamic × L_snr + w_psycho × L_psycho + w_linf × L_linf
```

| 항 | 수식 | 역할 | v3 가중치 |
|----|------|------|-----------|
| L_cam | -mean(cosine_dist(CAM++(adv), CAM++(orig))) | 화자 임베딩 거리 최대화 (핵심) | 1.0 |
| L_snr | mean(relu((snr_target - SNR)²/snr_target)) | SNR >= 28dB 제약 | 0.1 (동적 최대 1.0) |
| L_psycho | STFT magnitude 초과분 soft penalty | 심리음향 마스킹 제약 | **0.05** |
| L_linf | relu(|delta| - eps×0.8)×0.1 + relu(|delta| - eps)²×100 | L-inf <= eps 보장 | 10.0 |

**동적 w_snr:** SNR이 28dB에 못 미치면 w_snr을 최대 10배까지 증가시켜 SNR 제약을 강화.

**v3의 근본 문제:** `w_psycho=0.05`가 너무 낮아 심리음향 제약이 실질적으로 무시됨 → 추론 후 hard_project 적용 시 dist -65% 폭락.

---

## 4. 학습 파이프라인

### 데이터 흐름

```
KSS wav 파일 (12,853개)
    │
    ▼
AudioDataset (dataset.py)
  - 4초 초과: 랜덤 시작점에서 trim
  - 4초 미만: zero-pad
  - augment=True: time stretch(0.9~1.1x) / gaussian noise(SNR 35~45dB) / amplitude(0.8~1.0x)
    │
    ▼
DataLoader (batch_size=16, drop_last=True, num_workers=0)
  반환: (orig_batch[B,T], paths[B])
    │
    ├──────────────────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
MelExtractor (no_grad)                              emb_cache (사전 계산, 고정)
  (B,T) → (B,1,80,401)                             path → (192,) CAM++ 임베딩
    │                                               front-4s 고정 trim으로 캐싱
    ▼
Generator (학습 대상, grad 있음)
  mel → delta (B,64000)
    │
    ▼
adv = clamp(orig + delta, -1, 1)
    │
    ├─── L_cam: CAM++(adv) vs orig_emb_cached → cosine dist 최대화
    ├─── L_snr: SNR(orig, delta) >= 28dB
    ├─── L_psycho: STFT magnitude <= masking threshold
    └─── L_linf: |delta| <= eps=0.01
    │
    ▼
total_loss.backward() → clip_grad_norm(1.0) → Adam.step()
```

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| optimizer | Adam |
| lr | 1e-4 (초기) |
| scheduler | CosineAnnealingLR, T_max=epochs, eta_min=lr×0.01 |
| batch_size | 16 |
| grad_clip | 1.0 |
| eps | 0.01 |
| SNR target | 28dB |
| 총 epoch | 10 |

### 원본 임베딩 캐싱 전략

학습 중 배치마다 CAM++ forward를 원본 오디오에 2번 돌리는 낭비를 없애기 위해, 학습 시작 전 1회 전체 데이터셋을 순회하며 모든 파일의 CAM++ 임베딩을 캐싱.

- front-4s 고정 trim으로 캐싱 (랜덤 trim은 레퍼런스 불안정)
- 캐시는 `emb_cache.pt`로 저장 → 다음 세션에서 재사용 가능

---

## 5. 학습 결과 (v3, epoch 0~9)

**학습 환경:** Kaggle T4 GPU, batch=16, epochs=10

| epoch | dist | snr_l | psycho | lr |
|-------|------|-------|--------|----|
| 0 | 1.2885 | 0.1756 | 0.02858 | 9.76e-05 |
| 1 | 1.4409 | 0.1351 | 0.02665 | 9.06e-05 |
| 2 | 1.4720 | 0.1326 | 0.02628 | 7.96e-05 |
| 3 | 1.4908 | 0.1318 | 0.02612 | 6.58e-05 |
| 4 | 1.5043 | 0.1307 | 0.02602 | 5.05e-05 |
| 5 | 1.5163 | 0.1312 | 0.02597 | 3.52e-05 |
| 6 | 1.5268 | 0.1296 | 0.02592 | 2.14e-05 |
| 7 | 1.5354 | 0.1287 | 0.02580 | 1.05e-05 |
| 8 | 1.5408 | 0.1277 | 0.02576 | 3.42e-06 |
| **9** | **1.5437** | **0.1267** | **0.02569** | **1.00e-06** |

- `dist` 범위: [0, 2] (코사인 거리). 1.54는 임베딩이 거의 반대 방향 수준으로 이동됨을 의미.
- epoch 4~5부터 수렴 조짐 → eps=0.01에서의 한계점.
- `snr_l`: 꾸준히 감소하나 목표(0.1) 미달.
- `psycho`: w_psycho=0.05로 너무 낮아 실질 제약 없음.

---

## 6. 클로닝 방어 효과 (KSS 5개 평균)

**테스트 방법:** infer.py로 보호 → 원본 sr(44100Hz)로 업샘플 → CosyVoice3 inference_zero_shot으로 클론 생성 → CAM++ 유사도 비교

| 지표 | 값 |
|------|-----|
| 원본 클론 유사도 | **0.7902** |
| 보호 클론 유사도 | **0.1619** |
| 유사도 감소율 | **79.5%** |

클로닝이 사실상 실패 수준으로 떨어짐.

---

## 7. 추론 파이프라인 (infer.py)

```
입력 wav (임의 sr, 임의 길이)
    │
    ▼
load_wav: 16kHz mono 변환, [-1,1] clip
    │
    ▼
MelExtractor: (1,T) → (1,1,80,T') [가변 T']
    │
    ▼
Generator (projector.t_out=None → 가변 길이 출력)
    │
    ▼
delta: [-eps, +eps]
    │
    ├── [옵션] hard_project(delta, threshold): 심리음향 hard clamp
    │
    ▼
protected = clamp(orig + delta, -1, 1)
    │
    ▼
저장: {stem}_protected.wav (16kHz mono)
```

**hard_project ON vs OFF 비교 (KSS 5개, v3_best.pt):**

| 지표 | OFF | ON | 변화 |
|------|-----|-----|------|
| dist | 1.0725 | 0.3701 | **-65%** |
| SNR | 26.86 dB | 28.66 dB | +1.80 dB |
| psycho 초과량 | 0.000037 | 0.000003 | -92% |

→ 노이즈 감소 효과는 있으나 dist 손실이 너무 큼. 근본 해결은 재훈련(v4) 필요.

---

## 8. v3의 문제점 요약

| 항목 | 내용 |
|------|------|
| w_psycho=0.05 | 심리음향 제약 실질 무시 → 가청 노이즈 존재 |
| hard_project ON 시 dist -65% | 모델이 가청 범위까지 perturbation을 사용하도록 학습됨 |
| snr_l 수렴 한계 | 0.1 아래 미달 (eps=0.01 한계) |
| dist 수렴 한계 | 1.54 수준 (eps=0.01 기준 최대치 근접) |
| CosyVoice3 불안정 | 짧은 텍스트/레퍼런스에서 클로닝 실패 잦음 |

---

## 9. v3 → v4 변경 방향

v3의 근본 문제(w_psycho 너무 낮음)를 해결하기 위해 재훈련.

| 항목 | v3 | v4 |
|------|----|----|
| psycho 모듈 | psycho.py | psycho_v2.py (tonal/noise masker + temporal masking, 임계값 1.67배) |
| w_psycho | 0.05 (고정) | 0.3 + 동적 조절 (psycho > 0.020이면 점진 증가, 최대 3.0) |
| 시작점 | 처음부터 | v3_best.pt에서 fine-tuning (optimizer 새로 초기화) |
| 목표 | dist 최대화 우선 | 비가청 범위 안에서 dist 최대화 |

---

## 10. 파일 목록

```
voice_generator/ (v3 기준)
├── dataset.py          STEP 1  KSS 로드 + augmentation
├── features.py         STEP 2  MelExtractor, kaldi fbank
├── psycho.py           STEP 3  심리음향 마스킹 (ISO/IEC 11172-3)
├── model.py            STEP 4  PerturbationGenerator (U-Net, 54.9M params)
├── loss.py             STEP 5  GeneratorLoss (4개 항 가중합)
├── train_utils.py      STEP 6A emb_cache, checkpoint save/load
├── train_epoch.py      STEP 6B 단일 epoch 루프
├── train.py            STEP 6C 학습 진입점
├── evaluate.py         STEP 7  방어 효과 평가
├── infer.py            STEP 8  단일 파일 추론
├── run_generator_train.py STEP 9  Kaggle 실행 스크립트
├── v3_best.pt          학습 결과 체크포인트 (epoch 9, 630MB)
└── tests/
    ├── test_full_integration.py  전체 통합 테스트 (69/69 PASS)
    ├── test_step8_infer.py
    └── test_quick_infer.py
```

---

## 11. 통합 테스트 결과

```
A. dataset.py    : 12/12 PASS
B. features.py   :  6/6  PASS
C. model.py      :  7/7  PASS
D. loss.py       : 10/10 PASS
E. train_utils.py:  6/6  PASS
F. train_epoch.py:  6/6  PASS
G. evaluate.py   :  5/5  PASS
H. infer.py      : 12/12 PASS
I. 경계/연결     :  4/4  PASS

총계: 69/69 PASS ✅
```

---

## 12. 실행 방법

```bash
# 의존성 설치 (Kaggle)
!pip install onnx onnx2torch -q

# 학습 (처음부터)
!python /kaggle/working/voice_generator/run_generator_train.py --epochs 100 --batch 16

# 학습 (이어서)
!python /kaggle/working/voice_generator/run_generator_train.py --resume

# 추론 (단일 파일)
python infer.py --checkpoint v3_best.pt --input voice.wav

# 추론 (hard_project 없이)
python infer.py --checkpoint v3_best.pt --input voice.wav --no_psycho

# 평가 (디렉토리 전체)
python evaluate.py --checkpoint v3_best.pt --wav_dir /path/to/kss
```

---

## 13. 필수 파일 위치 (Kaggle)

```
/kaggle/working/
├── pgd_verify.py          ← 반드시 루트 (psycho.py가 상위 2단계로 import)
└── voice_generator/
    └── (모든 py 파일)

/kaggle/input/voice-pgd/
├── kss/                   ← KSS wav 파일
└── campplus.onnx          ← CAM++ ONNX 모델
```
