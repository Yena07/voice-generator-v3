"""
STEP 8: 단일 wav 파일에 perturbation 적용 (추론)

학습된 PerturbationGenerator로 입력 wav를 보호된 wav로 변환.
단일 forward pass (~0.01s), 실시간 사용 가능.

실행 방법:
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav --output protected.wav
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav  (output 자동 생성)
  python infer.py --checkpoint checkpoints/best.pt --input voice.wav --no_psycho  (hard_project 비활성화)

변경 이력:
  2026-06-22  hard_project 후처리 추가 (--no_psycho 플래그로 비활성화 가능)
              - delta 생성 후 심리음향 마스킹 threshold 초과 성분을 STFT 도메인에서 제거
              - 이후 L-inf 재확인 (iSTFT 복원 오차 보정)
              - 적용 전후 SNR/L-inf/psycho_exceed 지표 모두 출력
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PGD_DIR  = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from dataset import MAX_SAMPLES, SAMPLE_RATE
from features import MelExtractor
from model import PerturbationGenerator, EPS_DEFAULT
from psycho import hard_project, compute_threshold


# ── 인자 파싱 ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="학습된 generator 체크포인트 경로 (.pt)")
    p.add_argument("--input",  type=str, required=True,
                   help="보호할 입력 wav 경로")
    p.add_argument("--output", type=str, default=None,
                   help="출력 wav 경로 (None → input_protected.wav)")
    p.add_argument("--no_psycho", action="store_true",
                   help="심리음향 hard_project 후처리 비활성화 (2026-06-22 이전 동작)")
    return p.parse_args()


# ── 핵심 추론 함수 ────────────────────────────────────────────

def load_generator(checkpoint_path: str, device: torch.device) -> PerturbationGenerator:
    """
    체크포인트에서 Generator 로드.
    train_utils 형식 (key='generator') / model.save 형식 (key='state_dict') 모두 지원.
    추론 시 가변 길이 처리를 위해 projector.t_out을 None으로 설정한다.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    eps  = ckpt.get("eps", EPS_DEFAULT)
    gen  = PerturbationGenerator(eps=eps).to(device)
    state_key = "generator" if "generator" in ckpt else "state_dict"
    gen.load_state_dict(ckpt[state_key])
    gen.projector.t_out = None  # 추론 시 가변 길이
    gen.eval()
    return gen


def load_wav(path: str) -> tuple:
    """
    wav 파일 로드 → (audio_np, original_sr)
    audio_np: float32 mono, 16kHz, [-1,1], 길이는 원본 그대로
    """
    from math import gcd
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    original_sr = sr

    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio, original_sr


def _psycho_exceed(delta_np: np.ndarray, threshold: np.ndarray) -> float:
    """
    delta의 STFT magnitude가 심리음향 threshold를 초과하는 평균량.
    0에 가까울수록 가청 성분이 적음.
    """
    import scipy.signal
    window = np.hanning(400).astype(np.float32)
    _, _, stft = scipy.signal.stft(
        delta_np, fs=SAMPLE_RATE, window=window,
        nperseg=400, noverlap=400 - 160, nfft=512, padded=True,
    )
    mag = np.abs(stft)
    n_frames = min(mag.shape[1], threshold.shape[1])
    exceed = np.maximum(mag[:, :n_frames] - threshold[:, :n_frames], 0.0)
    return float(exceed.mean())


def protect_wav(
    audio_np: np.ndarray,
    generator: PerturbationGenerator,
    mel_extractor: MelExtractor,
    device: torch.device,
    use_psycho: bool = True,
) -> tuple:
    """
    임의 길이 wav에 perturbation을 적용.

    입력 전체를 한번에 처리 (가변 길이). TimeDomainProjector가 t_out=None일 때
    입력 T'에 맞는 길이의 delta를 반환하므로 청크 분할 없이 경계 불연속이 없음.
    30초 초과 입력은 훈련 분포(4초)와 크게 달라 경고를 출력한다.

    Parameters
    ----------
    audio_np      : (T,) float32, 16kHz mono, [-1,1]
    generator     : eval 모드 Generator (t_out=None 설정 필요)
    mel_extractor : eval 모드 MelExtractor
    device        : torch.device
    use_psycho    : True면 hard_project 후처리 적용 (기본값)

    Returns
    -------
    protected_np : (T,) float32, 원본과 동일한 길이
    metrics      : dict {
        "linf_before"     : hard_project 전 L-inf,
        "linf_after"      : hard_project 후 L-inf,
        "snr_before"      : hard_project 전 SNR (dB),
        "snr_after"       : hard_project 후 SNR (dB),
        "psycho_before"   : hard_project 전 threshold 초과량,
        "psycho_after"    : hard_project 후 threshold 초과량 (use_psycho=False면 None),
        "psycho_applied"  : bool,
    }
    """
    total_len = len(audio_np)

    if total_len > SAMPLE_RATE * 30:
        print(f"[infer] 경고: 입력이 {total_len/SAMPLE_RATE:.1f}s로 30초를 초과합니다. "
              f"모델은 4초 기준으로 훈련되었으며 10초 이내 클립을 권장합니다.")

    with torch.no_grad():
        audio_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # (1, T)
        mel     = mel_extractor(audio_t)                               # (1,1,80,T')
        delta   = generator(mel)                                       # (1, T'')

        # delta 길이와 원본 길이 맞춤 (ConvTranspose1d 출력이 근사값이므로)
        T = total_len
        T_delta = delta.shape[1]
        if T_delta > T:
            delta = delta[:, :T]
        elif T_delta < T:
            delta = F.pad(delta, [0, T - T_delta])

    delta_np = delta.squeeze(0).cpu().numpy()  # (T,)

    # hard_project 전 지표
    sig_pow = float(np.mean(audio_np ** 2)) + 1e-12
    def _snr(d):
        return 10.0 * np.log10(sig_pow / (float(np.mean(d ** 2)) + 1e-12))

    threshold = compute_threshold(audio_np) if use_psycho else None
    linf_before  = float(np.abs(delta_np).max())
    snr_before   = _snr(delta_np)
    psycho_before = _psycho_exceed(delta_np, threshold) if threshold is not None else None

    # 심리음향 hard_project
    if use_psycho:
        delta_np = hard_project(delta_np, threshold)
        delta_np = np.clip(delta_np, -generator.eps, generator.eps)  # iSTFT 오차 보정

    linf_after  = float(np.abs(delta_np).max())
    snr_after   = _snr(delta_np)
    psycho_after = _psycho_exceed(delta_np, threshold) if threshold is not None else None

    adv_np = np.clip(audio_np + delta_np, -1.0, 1.0)

    metrics = {
        "linf_before":    linf_before,
        "linf_after":     linf_after,
        "snr_before":     snr_before,
        "snr_after":      snr_after,
        "psycho_before":  psycho_before,
        "psycho_after":   psycho_after,
        "psycho_applied": use_psycho,
    }
    return adv_np, metrics


# ── 메인 ─────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_psycho = not args.no_psycho

    # 출력 경로 자동 생성
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_protected" + (ext if ext else ".wav")

    print(f"[infer] device        : {device}")
    print(f"[infer] input         : {args.input}")
    print(f"[infer] output        : {args.output}")
    print(f"[infer] checkpoint    : {args.checkpoint}")
    print(f"[infer] hard_project  : {'ON' if use_psycho else 'OFF (--no_psycho)'}")

    # 모델 로드
    generator     = load_generator(args.checkpoint, device)
    mel_extractor = MelExtractor().to(device).eval()
    print(f"[infer] Generator eps={generator.eps}")

    # 입력 오디오 로드
    audio_np, original_sr = load_wav(args.input)
    print(f"[infer] 입력 길이: {len(audio_np)} samples ({len(audio_np)/SAMPLE_RATE:.2f}s)")

    # Perturbation 적용
    t0 = time.time()
    protected_np, metrics = protect_wav(
        audio_np, generator, mel_extractor, device, use_psycho=use_psycho
    )
    elapsed = time.time() - t0
    print(f"[infer] 처리 시간: {elapsed*1000:.1f}ms")

    # 저장 (16kHz)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, protected_np, SAMPLE_RATE)
    print(f"[infer] 저장 완료: {args.output}")

    # 지표 출력
    eps = generator.eps
    if use_psycho:
        print(f"[infer] ── hard_project 전 ──────────────────────")
        print(f"[infer]   L-inf  : {metrics['linf_before']:.6f}  (기준: <= {eps})")
        print(f"[infer]   SNR    : {metrics['snr_before']:.2f} dB  (기준: >= 28dB)")
        print(f"[infer]   psycho : {metrics['psycho_before']:.6f}  (threshold 초과 평균)")
        print(f"[infer] ── hard_project 후 ──────────────────────")
        print(f"[infer]   L-inf  : {metrics['linf_after']:.6f}  (기준: <= {eps})")
        print(f"[infer]   SNR    : {metrics['snr_after']:.2f} dB  (기준: >= 28dB)")
        print(f"[infer]   psycho : {metrics['psycho_after']:.6f}  (0에 가까울수록 좋음)")
        linf_delta = metrics['linf_before'] - metrics['linf_after']
        snr_delta  = metrics['snr_after']   - metrics['snr_before']
        print(f"[infer] ── 변화량 ───────────────────────────────")
        print(f"[infer]   ΔL-inf : {linf_delta:+.6f}  (음수 = delta 감소)")
        print(f"[infer]   ΔSNR   : {snr_delta:+.2f} dB  (양수 = 음질 개선)")
    else:
        print(f"[infer] L-inf : {metrics['linf_after']:.6f}  (기준: <= {eps})")
        print(f"[infer] SNR   : {metrics['snr_after']:.2f} dB  (기준: >= 28dB)")


if __name__ == "__main__":
    main()
