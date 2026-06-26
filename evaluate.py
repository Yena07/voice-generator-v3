"""
STEP 7: Generator 방어 효과 평가

학습된 PerturbationGenerator로 adv wav를 생성하고
CAM++ cosine dist / SNR / L-inf / 심리음향 초과율을 측정.

실행 방법:
  python evaluate.py --checkpoint checkpoints/best.pt --wav_dir /kaggle/input/voice-pgd/kss
  python evaluate.py --checkpoint checkpoints/best.pt --wav_dir C:/voice_pgd/kss --max_files 100
"""

import argparse
import os
import sys
import json

import torch
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PGD_DIR  = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from dataset import AudioDataset, MAX_SAMPLES
from features import MelExtractor
from model import PerturbationGenerator, EPS_DEFAULT
try:
    from loss_v2 import campplus_embed_single, cosine_dist, snr_db
    from psycho_v2 import compute_threshold, soft_psycho_loss
except ImportError:
    from loss import campplus_embed_single, cosine_dist, snr_db
    from psycho import compute_threshold, soft_psycho_loss


# ── 경로 설정 ─────────────────────────────────────────────────

KAGGLE = os.path.exists("/kaggle/input")

if KAGGLE:
    CAMPPLUS_ONNX = "/kaggle/input/voice-pgd/campplus.onnx"
else:
    CAMPPLUS_ONNX = (
        r"C:\Users\hanso\OneDrive\바탕 화면\졸프\CosyVoice"
        r"\pretrained_models\Fun-CosyVoice3-0.5B-2512\campplus.onnx"
    )


# ── 인자 파싱 ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="학습된 generator 체크포인트 경로 (.pt)")
    p.add_argument("--wav_dir",    type=str, required=True,
                   help="평가할 KSS wav 디렉토리")
    p.add_argument("--max_files",  type=int, default=None,
                   help="평가할 최대 파일 수 (None=전체)")
    p.add_argument("--output",     type=str, default=None,
                   help="결과 JSON 저장 경로 (None=저장 안함)")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


# ── 단일 파일 평가 ────────────────────────────────────────────

@torch.no_grad()
def evaluate_file(
    path: str,
    generator,
    mel_extractor,
    campplus_model,
    device: torch.device,
) -> dict:
    """
    단일 wav 파일에 대해 방어 지표 측정.

    Returns
    -------
    dict:
        dist    : CAM++ cosine dist (orig vs adv), 클수록 방어 성공
        snr     : SNR (dB), 클수록 원음 보존
        linf    : delta L-inf, <= eps 보장
        psycho  : 심리음향 soft loss (0에 가까울수록 마스킹 범위 내)
    """
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    # 오디오 로드 + 4초 고정
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != 16000:
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    if len(audio) >= MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    else:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")

    orig_t = torch.from_numpy(audio).unsqueeze(0).to(device)   # (1, T)

    # 원본 임베딩
    orig_emb = campplus_embed_single(campplus_model, orig_t)    # (192,)

    # Generator → delta → adv
    mel   = mel_extractor(orig_t)                               # (1,1,80,401)
    delta = generator(mel)                                      # (1, T)
    adv_t = torch.clamp(orig_t + delta, -1.0, 1.0)             # (1, T)

    # adv 임베딩
    adv_emb = campplus_embed_single(campplus_model, adv_t)      # (192,)

    # 지표 계산
    dist_val  = cosine_dist(adv_emb, orig_emb).item()
    snr_val   = snr_db(orig_t.squeeze(0), delta.squeeze(0)).item()
    linf_val  = delta.abs().max().item()

    # 심리음향 soft loss (gradient 불필요)
    thresh = compute_threshold(audio)
    psycho_val = soft_psycho_loss(delta.squeeze(0), thresh).item()

    return {
        "dist":   dist_val,
        "snr":    snr_val,
        "linf":   linf_val,
        "psycho": psycho_val,
    }


# ── 전체 평가 루프 ────────────────────────────────────────────

def evaluate(
    generator,
    mel_extractor,
    campplus_model,
    wav_dir: str,
    device: torch.device,
    max_files: int = None,
) -> dict:
    """
    wav_dir 내 파일 전체(또는 max_files개) 평가.

    Returns
    -------
    summary : dict {
        "mean_dist"   : float
        "mean_snr"    : float
        "mean_linf"   : float
        "mean_psycho" : float
        "n_files"     : int
        "per_file"    : list of {path, dist, snr, linf, psycho}
    }
    """
    dataset = AudioDataset(wav_dir, augment=False)
    paths   = dataset.paths
    if max_files is not None:
        paths = paths[:max_files]

    generator.eval()
    mel_extractor.eval()
    campplus_model.eval()

    per_file = []
    for i, path in enumerate(paths):
        result = evaluate_file(path, generator, mel_extractor, campplus_model, device)
        result["path"] = os.path.basename(path)
        per_file.append(result)

        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(
                f"  [{i+1}/{len(paths)}] "
                f"dist={result['dist']:.4f}  "
                f"snr={result['snr']:.1f}dB  "
                f"linf={result['linf']:.5f}  "
                f"psycho={result['psycho']:.5f}"
            )

    mean_dist   = np.mean([r["dist"]   for r in per_file])
    mean_snr    = np.mean([r["snr"]    for r in per_file])
    mean_linf   = np.mean([r["linf"]   for r in per_file])
    mean_psycho = np.mean([r["psycho"] for r in per_file])

    return {
        "mean_dist":   float(mean_dist),
        "mean_snr":    float(mean_snr),
        "mean_linf":   float(mean_linf),
        "mean_psycho": float(mean_psycho),
        "n_files":     len(per_file),
        "per_file":    per_file,
    }


# ── 결과 출력 ────────────────────────────────────────────────

def print_summary(summary: dict):
    print("\n" + "=" * 50)
    print("평가 결과 요약")
    print("=" * 50)
    print(f"  파일 수        : {summary['n_files']}")
    print(f"  CAM++ dist 평균: {summary['mean_dist']:.4f}  (클수록 방어 성공)")
    print(f"  SNR 평균       : {summary['mean_snr']:.2f} dB  (기준: >= 28dB)")
    print(f"  L-inf 평균     : {summary['mean_linf']:.6f}  (기준: <= {EPS_DEFAULT})")
    print(f"  Psycho loss 평균: {summary['mean_psycho']:.6f}  (낮을수록 마스킹 범위 내)")

    snr_ok = sum(1 for r in summary["per_file"] if r["snr"] >= 28.0)
    linf_ok = sum(1 for r in summary["per_file"] if r["linf"] <= EPS_DEFAULT + 1e-5)
    print(f"\n  SNR >= 28dB 만족: {snr_ok}/{summary['n_files']}")
    print(f"  L-inf <= eps 만족: {linf_ok}/{summary['n_files']}")
    print("=" * 50)


# ── 메인 ─────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device     : {device}")
    print(f"[eval] checkpoint : {args.checkpoint}")
    print(f"[eval] wav_dir    : {args.wav_dir}")

    # CAM++ 로드
    print("[eval] CAM++ 로드 중...")
    import onnx, onnx2torch
    campplus = onnx2torch.convert(onnx.load(CAMPPLUS_ONNX)).eval().to(device)
    for p in campplus.parameters():
        p.requires_grad = False

    # Generator 로드
    # train_utils.save_checkpoint는 {"generator": state_dict, "eps": ...} 형식으로 저장
    # model.save/load는 {"state_dict": ..., "eps": ...} 형식 — 두 형식 모두 지원
    ckpt = torch.load(args.checkpoint, map_location=device)
    eps  = ckpt.get("eps", EPS_DEFAULT)
    generator = PerturbationGenerator(eps=eps).to(device)
    state_key = "generator" if "generator" in ckpt else "state_dict"
    generator.load_state_dict(ckpt[state_key])
    generator.eval()
    print(f"[eval] Generator eps={generator.eps}")

    mel_extractor = MelExtractor().to(device).eval()

    # 평가 실행
    print(f"[eval] 평가 시작...")
    summary = evaluate(
        generator=generator,
        mel_extractor=mel_extractor,
        campplus_model=campplus,
        wav_dir=args.wav_dir,
        device=device,
        max_files=args.max_files,
    )

    print_summary(summary)

    # JSON 저장
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[eval] 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
