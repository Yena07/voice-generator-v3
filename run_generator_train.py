"""
Kaggle 학습 실행 스크립트 (노트북 대체용)

Kaggle 노트북에서 아래 한 줄로 실행:
  !python /kaggle/working/voice_generator/run_generator_train.py

또는 로컬:
  python run_generator_train.py
  python run_generator_train.py --epochs 50 --batch 16 --resume

학습 완료 후:
  - checkpoints/best.pt  : 최고 dist 모델
  - checkpoints/last.pt  : 마지막 체크포인트
  - checkpoints/train_log.json : epoch별 metrics
  - quick_output/ : 샘플 음성 보호 결과
"""

import os
import sys
import time
import json
import argparse

# ── 경로 설정 ─────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PGD_DIR  = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

KAGGLE = os.path.exists("/kaggle/input")

if KAGGLE:
    KSS_DIR       = "/kaggle/working/kss"
    CAMPPLUS_ONNX = "/kaggle/input/datasets/asdadzxxc/voice-pgd-generator/campplus.onnx"
    OUTPUT_DIR    = "/kaggle/working/checkpoints"
    SAMPLE_WAV    = "/kaggle/working/kss/1/1_0000.wav"
else:
    KSS_DIR       = r"C:\voice_pgd\kss"
    CAMPPLUS_ONNX = (r"C:\Users\hanso\OneDrive\바탕 화면\졸프\CosyVoice"
                     r"\pretrained_models\Fun-CosyVoice3-0.5B-2512\campplus.onnx")
    OUTPUT_DIR    = os.path.join(_THIS_DIR, "checkpoints")
    SAMPLE_WAV    = r"C:\voice_pgd\kss\1\1_0000.wav"

BEST_CKPT = os.path.join(OUTPUT_DIR, "best.pt")
LAST_CKPT = os.path.join(OUTPUT_DIR, "last.pt")
on_best_saved = None
LOG_PATH  = os.path.join(OUTPUT_DIR, "train_log.json")
QUICK_OUT = os.path.join(OUTPUT_DIR, "quick_output")

SESSION_LIMIT_SEC = 12 * 3600 - 900  # 11시간 45분


# ── 인자 파싱 ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--save_interval", type=int,   default=1)
    p.add_argument("--skip_infer",    action="store_true",
                   help="학습 후 샘플 추론 건너뜀")
    return p.parse_args()


# ── 시각화 ────────────────────────────────────────────────────

def plot_history(log_path: str, out_dir: str):
    """train_log.json → loss/dist 곡선 PNG 저장"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(log_path, encoding="utf-8") as f:
            history = json.load(f)

        epochs = [h["epoch"] for h in history]
        losses = [h["loss"]  for h in history]
        dists  = [h["dist"]  for h in history]
        snrs   = [h["snr"]   for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, losses, "b-o", markersize=3)
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        axes[1].plot(epochs, dists, "r-o", markersize=3)
        axes[1].set_title("CAM++ Cosine Dist (방어 효과)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dist")
        axes[1].axhline(y=0.3, color="gray", linestyle="--", label="목표 0.3")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(epochs, snrs, "g-o", markersize=3)
        axes[2].set_title("SNR Loss (낮을수록 SNR 만족)")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("SNR Loss")
        axes[2].axhline(y=0.0, color="gray", linestyle="--")
        axes[2].grid(True)

        plt.tight_layout()
        out_path = os.path.join(out_dir, "train_curves.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  [plot] 곡선 저장: {out_path}")

    except Exception as e:
        print(f"  [plot] 시각화 실패 (무시): {e}")


# ── 학습 후 샘플 추론 ─────────────────────────────────────────

def run_quick_infer(checkpoint: str, sample_wav: str, out_dir: str, device):
    """학습된 모델로 샘플 음성 1개 보호 후 지표 출력"""
    import numpy as np
    import soundfile as sf
    import torch

    from features import MelExtractor
    from infer import load_generator, load_wav, protect_wav
    from loss import campplus_embed_single, cosine_dist
    from dataset import SAMPLE_RATE

    print("\n" + "=" * 55)
    print("학습 후 샘플 추론")
    print("=" * 55)

    if not os.path.exists(sample_wav):
        print(f"  샘플 파일 없음: {sample_wav}")
        return

    os.makedirs(out_dir, exist_ok=True)

    mel_extractor = MelExtractor().to(device).eval()
    generator     = load_generator(checkpoint, device)

    import onnx, onnx2torch
    campplus = onnx2torch.convert(onnx.load(CAMPPLUS_ONNX)).eval().to(device)
    for p in campplus.parameters():
        p.requires_grad = False

    audio_np, orig_sr = load_wav(sample_wav)
    duration = len(audio_np) / SAMPLE_RATE
    print(f"  입력: {os.path.basename(sample_wav)}  ({duration:.2f}s, {orig_sr}Hz)")

    t0 = time.time()
    protected_np, _ = protect_wav(audio_np, generator, mel_extractor, device)
    elapsed = time.time() - t0

    delta_np = protected_np - audio_np
    linf     = float(np.abs(delta_np).max())
    sig_pow  = float(np.mean(audio_np ** 2)) + 1e-12
    noi_pow  = float(np.mean(delta_np ** 2)) + 1e-12
    snr      = 10 * float(np.log10(sig_pow / noi_pow))

    with torch.no_grad():
        orig_t = torch.from_numpy(audio_np).unsqueeze(0).to(device)
        prot_t = torch.from_numpy(protected_np).unsqueeze(0).to(device)
        orig_emb = campplus_embed_single(campplus, orig_t)
        prot_emb = campplus_embed_single(campplus, prot_t)
        dist = cosine_dist(prot_emb, orig_emb).item()

    print(f"\n  CAM++ dist : {dist:.4f}  {'✅' if dist >= 0.3 else '❌'}  (목표 >= 0.3)")
    print(f"  SNR        : {snr:.2f} dB  {'✅' if snr >= 28.0 else '❌'}  (목표 >= 28dB)")
    print(f"  L-inf      : {linf:.6f}  {'✅' if linf <= generator.eps else '❌'}  (목표 <= {generator.eps})")
    print(f"  처리 시간  : {elapsed*1000:.1f}ms  (RTF={elapsed/duration:.3f})")

    sf.write(os.path.join(out_dir, "original.wav"),  audio_np,     SAMPLE_RATE)
    sf.write(os.path.join(out_dir, "protected.wav"), protected_np, SAMPLE_RATE)
    print(f"\n  원본/보호 wav 저장: {out_dir}")


# ── 메인 ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("Generator 학습 시작")
    print(f"  device  : {device}")
    print(f"  KSS     : {KSS_DIR}")
    print(f"  output  : {OUTPUT_DIR}")
    print(f"  epochs  : {args.epochs}")
    print(f"  batch   : {args.batch}")
    print(f"  lr      : {args.lr}")
    print(f"  resume  : {args.resume}")
    print("=" * 55)

    # ── CAM++ 로드 ────────────────────────────────────────────
    print("\n[1] CAM++ 로드...")
    import onnx, onnx2torch
    campplus = onnx2torch.convert(onnx.load(CAMPPLUS_ONNX)).eval().to(device)
    for p in campplus.parameters():
        p.requires_grad = False
    print("  완료")

    # ── 모델 / Optimizer / Scheduler ─────────────────────────
    print("\n[2] 모델 초기화...")
    from features import MelExtractor
    from model import PerturbationGenerator, EPS_DEFAULT
    from loss import GeneratorLoss
    from train_utils import compute_orig_embeddings, save_checkpoint, load_checkpoint
    from train_epoch import train_one_epoch
    from dataset import AudioDataset, build_dataloader

    generator     = PerturbationGenerator(eps=EPS_DEFAULT).to(device)
    mel_extractor = MelExtractor().to(device).eval()
    criterion     = GeneratorLoss(campplus_model=campplus)
    optimizer     = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    print(f"  Generator 파라미터: {sum(p.numel() for p in generator.parameters()):,}")

    # ── DataLoader ────────────────────────────────────────────
    print("\n[3] DataLoader 생성...")
    loader = build_dataloader(
        KSS_DIR, batch_size=args.batch, augment=True, num_workers=args.workers
    )
    dataset_noaug = AudioDataset(KSS_DIR, augment=False)
    n_files = dataset_noaug.num_files
    print(f"  파일 수: {n_files}, 배치 수: {len(loader)}")

    # ── 원본 임베딩 캐싱 ──────────────────────────────────────
    print("\n[4] 원본 임베딩 캐싱...")
    t0 = time.time()
    cache_path = os.path.join(OUTPUT_DIR, "emb_cache.pt")
    emb_cache = compute_orig_embeddings(campplus, dataset_noaug, device, cache_path=cache_path)
    print(f"  완료: {len(emb_cache)}개, {time.time()-t0:.1f}s")

    # ── Resume ────────────────────────────────────────────────
    start_epoch  = 0
    best_dist    = 0.0
    loss_history = []

    if args.resume and os.path.exists(LAST_CKPT):
        print(f"\n[5] 체크포인트 로드: {LAST_CKPT}")
        start_epoch, best_dist, loss_history = load_checkpoint(
            LAST_CKPT, generator, optimizer, scheduler, device
        )
        print(f"  resume: epoch={start_epoch}, best_dist={best_dist:.4f}")
    elif not args.resume and os.path.exists(LAST_CKPT):
        print(f"\n[5] ⚠ {LAST_CKPT} 존재하지만 --resume 없이 실행됨 → 처음부터 재학습")
    else:
        print("\n[5] 새 학습 시작")

    # ── Epoch 루프 ────────────────────────────────────────────
    print(f"\n[6] 학습 (epoch {start_epoch} ~ {args.epochs - 1})")
    print("-" * 55)

    session_start = time.time()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        metrics = train_one_epoch(
            generator=generator,
            mel_extractor=mel_extractor,
            campplus_model=campplus,
            criterion=criterion,
            optimizer=optimizer,
            loader=loader,
            emb_cache=emb_cache,
            device=device,
        )

        scheduler.step()
        elapsed = time.time() - epoch_start
        lr_now  = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch":   epoch,
            "loss":    round(metrics["loss"],   6),
            "cam":     round(metrics["cam"],    6),
            "snr":     round(metrics["snr"],    6),
            "psycho":  round(metrics["psycho"], 6),
            "linf":    round(metrics["linf"],   6),
            "dist":    round(metrics["dist"],   6),
            "lr":      round(lr_now, 8),
            "elapsed": round(elapsed, 1),
        }
        loss_history.append(log_entry)

        print(
            f"[{epoch:03d}] loss={metrics['loss']:+.4f}  "
            f"dist={metrics['dist']:.4f}  "
            f"snr_l={metrics['snr']:.4f}  "
            f"lr={lr_now:.2e}  {elapsed:.1f}s"
        )

        # best 저장
        if metrics["dist"] > best_dist:
            best_dist = metrics["dist"]
            save_checkpoint(BEST_CKPT, generator, optimizer, scheduler,
                            epoch, best_dist, loss_history)
            print(f"  ★ best.pt 저장 (dist={best_dist:.4f})")
            if callable(getattr(on_best_saved, '__call__', None)):
                on_best_saved(BEST_CKPT)

        # 주기적 last 저장
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(LAST_CKPT, generator, optimizer, scheduler,
                            epoch, best_dist, loss_history)

        # JSON 로그 갱신
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)

        # Kaggle 세션 한계 체크
        if KAGGLE and (time.time() - session_start) > SESSION_LIMIT_SEC:
            print(f"\n  ⚠ 세션 한계 근접, 저장 후 종료")
            save_checkpoint(LAST_CKPT, generator, optimizer, scheduler,
                            epoch, best_dist, loss_history)
            break

    # 최종 저장
    save_checkpoint(LAST_CKPT, generator, optimizer, scheduler,
                    epoch, best_dist, loss_history)

    print("\n" + "=" * 55)
    print(f"학습 완료")
    print(f"  best_dist : {best_dist:.4f}")
    print(f"  best.pt   : {BEST_CKPT}")
    print(f"  last.pt   : {LAST_CKPT}")
    print(f"  log       : {LOG_PATH}")
    print("=" * 55)

    # ── 시각화 ───────────────────────────────────────────────
    print("\n[7] 학습 곡선 시각화...")
    plot_history(LOG_PATH, OUTPUT_DIR)

    # ── 샘플 추론 ────────────────────────────────────────────
    if not args.skip_infer and os.path.exists(BEST_CKPT):
        run_quick_infer(BEST_CKPT, SAMPLE_WAV, QUICK_OUT, device)


if __name__ == "__main__":
    main()
