# voice-generator-v3

PGD adversarial perturbation 기반 **음성 클로닝 방어** 시스템 (v3).

원본 음성에 사람 귀에 거의 들리지 않는 미세 노이즈(`delta`)를 더해, AI 음성 클로닝
(CosyVoice3 등)이 해당 화자를 복제하려 할 때 화자 임베딩(CAM++)이 전혀 다른 사람처럼
나오도록 만든다. 핵심은 파일마다 최적화를 반복하는 PGD 대신 **perturbation을 생성하는
신경망(Generator)을 한 번 학습**해 두고, 추론 시 forward 1회(~0.01s)로 즉시 보호하는 방식.

## 파이프라인

```
wav(16kHz) → MelExtractor → PerturbationGenerator(U-Net) → delta(|delta|≤eps)
   → adv = clamp(orig + delta) → CAM++ 임베딩 거리 최대화로 학습
```

학습 신호: `Generator 파라미터 → delta → adv → CAM++ → 임베딩 → cosine_dist`
(CAM++를 onnx2torch로 변환해 gradient가 끝까지 통하므로 순수 gradient 학습 가능)

### Loss (4개 항 가중합)
| 항 | 역할 | 가중치 |
|----|------|--------|
| L_cam | CAM++ 임베딩 거리 최대화 (방어 핵심) | 1.0 |
| L_snr | SNR ≥ 28dB (음질 유지) | 0.1 (동적 최대 10×) |
| L_psycho | 심리음향 마스킹 (가청 노이즈 억제) | 0.05 |
| L_linf | \|delta\| ≤ eps 보장 | 10.0 |

## 파일 구성
| 파일 | 역할 |
|------|------|
| `dataset.py` | wav 로드 + augmentation (16kHz, 4초 고정) |
| `features.py` | MelExtractor(Generator 입력), kaldi fbank(CAM++ 입력) |
| `psycho.py` | 심리음향 마스킹 (ISO/IEC 11172-3) |
| `model.py` | PerturbationGenerator (U-Net, ~55M params) |
| `loss.py` | GeneratorLoss (4개 항) |
| `train_utils.py` / `train_epoch.py` / `train.py` | 임베딩 캐싱 / epoch 루프 / 학습 진입점 |
| `run_generator_train.py` | 학습 실행 본체 (경로 전역변수) |
| `run_kaggle.py` | Kaggle 실행 드라이버 |
| `run_colab.py` | **Colab 실행 드라이버** |
| `infer.py` | 단일 wav 추론(보호 적용) |
| `evaluate.py` | 클로닝 방어 효과 평가 |
| `result.md` | v3 전체 결과/분석 |
| `COLAB_가이드.md` / `KAGGLE_가이드.md` | 실행 가이드 |

## 실행 (Colab)
자세한 절차는 [`COLAB_가이드.md`](COLAB_가이드.md) 참고.

```bash
pip install onnx onnx2torch soundfile
python run_colab.py --epochs 30 --batch 16     # run_colab.py 상단 4개 경로만 맞추면 됨
```

## 필요한 외부 자산 (repo 미포함)
- **`campplus.onnx`** — CAM++ 화자임베딩 모델. CosyVoice 모델 repo에서 다운로드
  (예: `https://huggingface.co/model-scope/CosyVoice-300M/resolve/main/campplus.onnx`, 약 28MB).
- **학습 데이터** — 16kHz mono wav. KSS 또는 AI Hub 자유대화 데이터 등.
  데이터/체크포인트/onnx는 `.gitignore`로 제외되어 있다.

## 주의
방어/연구 목적의 코드다. 학습에는 GPU가 필요하며(CPU는 비현실적으로 느림),
Colab T4 또는 Kaggle T4 환경을 권장한다.
