"""
STEP 6-C: 전체 학습 진입점 (Kaggle 실행용)

실행 방법:
  python train.py [--epochs 100] [--batch 8] [--resume] [--lr 1e-4]

Kaggle 경로 자동 감지:
  KSS       : /kaggle/input/voice-pgd/kss/
  campplus  : /kaggle/input/voice-pgd/campplus.onnx
  output    : /kaggle/working/checkpoints/

로컬 경로:
  KSS       : C:/voice_pgd/kss/
  campplus  : (CAMPPLUS_ONNX_LOCAL)
  output    : C:/voice_pgd/voice_generator/checkpoints/
"""

import argparse
import os
import sys
import time
import json

import torch

# 로컬 실행 시 경로 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PGD_DIR  = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from dataset import AudioDataset, build_dataloader
from features import MelExtractor
from model import PerturbationGenerator, EPS_DEFAULT
from loss import GeneratorLoss
from train_utils import compute_orig_embeddings, save_checkpoint, load_checkpoint
from train_epoch import train_one_epoch


# ── 경로 설정 ─────────────────────────────────────────────────

KAGGLE = os.path.exists("/kaggle/input")

if KAGGLE:
    KSS_DIR        = "/kaggle/input/voice-pgd/kss"
    CAMPPLUS_ONNX  = "/kaggle/input/voice-pgd/campplus.onnx"
    OUTPUT_DIR     = "/kaggle/working/checkpoints"
else:
    KSS_DIR        = r"C:\voice_pgd\kss"
    CAMPPLUS_ONNX  = (
        r"C:\Users\hanso\OneDrive\바탕 화면\졸프\CosyVoice"
        r"\pretrained_models\Fun-CosyVoice3-0.5B-2512\campplus.onnx"
    )
    OUTPUT_DIR = os.path.join(_THIS_DIR, "checkpoints")

BEST_CKPT   = os.path.join(OUTPUT_DIR, "best.pt")
LAST_CKPT   = os.path.join(OUTPUT_DIR, "last.pt")
LOG_PATH    = os.path.join(OUTPUT_DIR, "train_log.json")

# Kaggle 12시간 세션 안전 종료 마진 (초)
SESSION_LIMIT_SEC = 12 * 3600 - 900   # 11시간 45분


# ── 인자 파싱 ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--workers", type=int,   default=0)
    p.add_argument("--resume",  action="store_true",
                   help="마지막 체크포인트에서 재개")
    p.add_argument("--save_interval", type=int, default=5,
                   help="몇 epoch마다 last.pt 저장")
    return p.parse_args()


# ── 메인 ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device : {device}")
    print(f"[train] KSS    : {KSS_DIR}")
    print(f"[train] output : {OUTPUT_DIR}")
    print(f"[train] epochs : {args.epochs}, batch : {args.batch}, lr : {args.lr}")

    # ── CAM++ 로드 ─────────────────────────────────────────────
    print("[train] CAM++ 로드 중...")
    import onnx, onnx2torch
    campplus = onnx2torch.convert(onnx.load(CAMPPLUS_ONNX)).eval().to(device)
    for p in campplus.parameters():
        p.requires_grad = False
    print("[train] CAM++ 로드 완료")

    # ── 모델 / Optimizer / Scheduler ──────────────────────────
    generator     = PerturbationGenerator(eps=EPS_DEFAULT).to(device)
    mel_extractor = MelExtractor().to(device).eval()
    criterion     = GeneratorLoss(campplus_model=campplus)
    optimizer     = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── DataLoader ─────────────────────────────────────────────
    dataset = AudioDataset(KSS_DIR, augment=True)
    loader  = build_dataloader(
        KSS_DIR, batch_size=args.batch, augment=True, num_workers=args.workers
    )
    print(f"[train] 파일 수: {dataset.num_files}, 배치 수: {len(loader)}")

    # ── 원본 임베딩 사전 캐싱 ────────────────────────────────────
    print("[train] 원본 임베딩 캐싱...")
    t0 = time.time()
    # augment=False 전용 dataset으로 캐싱 (원본 고정)
    dataset_noaug = AudioDataset(KSS_DIR, augment=False)
    emb_cache = compute_orig_embeddings(campplus, dataset_noaug, device)
    print(f"[train] 캐싱 완료: {len(emb_cache)}개, {time.time()-t0:.1f}s")

    # ── 체크포인트 resume ────────────────────────────────────────
    start_epoch = 0
    best_dist   = 0.0
    loss_history = []

    if args.resume and os.path.exists(LAST_CKPT):
        print(f"[train] 체크포인트 로드: {LAST_CKPT}")
        start_epoch, best_dist, loss_history = load_checkpoint(
            LAST_CKPT, generator, optimizer, scheduler, device
        )
        print(f"[train] resume: epoch={start_epoch}, best_dist={best_dist:.4f}")
    elif not args.resume and os.path.exists(LAST_CKPT):
        # last.pt가 있는데 --resume을 안 붙이면 처음부터 다시 학습하게 됨
        # LR/optimizer 상태가 리셋되어 학습이 꼬일 수 있으므로 명시적 경고
        print(f"[train] ⚠ 경고: {LAST_CKPT} 존재하지만 --resume 없이 실행됨")
        print(f"[train] ⚠ 처음부터 재학습합니다. 이어서 학습하려면 --resume 추가하세요.")

    # ── 세션 시간 추적 ───────────────────────────────────────────
    session_start = time.time()
    epoch = start_epoch - 1  # 루프가 0회 돌았을 때 last.pt 오염 방지

    # ── Epoch 루프 ────────────────────────────────────────────
    print(f"[train] 학습 시작 (epoch {start_epoch} ~ {args.epochs - 1})")
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
            f"[epoch {epoch:03d}] "
            f"loss={metrics['loss']:+.4f}  "
            f"cam={metrics['cam']:+.4f}  "
            f"dist={metrics['dist']:.4f}  "
            f"snr_l={metrics['snr']:.4f}  "
            f"lr={lr_now:.2e}  "
            f"{elapsed:.1f}s"
        )

        # best 모델 저장
        if metrics["dist"] > best_dist:
            best_dist = metrics["dist"]
            save_checkpoint(
                path=BEST_CKPT,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dist=best_dist,
                loss_history=loss_history,
            )
            print(f"  ★ best.pt 저장 (dist={best_dist:.4f})")

        # 주기적 last.pt 저장
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                path=LAST_CKPT,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dist=best_dist,
                loss_history=loss_history,
            )

        # 로그 JSON 갱신
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)

        # Kaggle 세션 한계 체크 (남은 시간 < 900초면 종료)
        session_elapsed = time.time() - session_start
        if KAGGLE and session_elapsed > SESSION_LIMIT_SEC:
            print(f"[train] ⚠ 세션 한계 근접 ({session_elapsed/3600:.2f}h), 저장 후 종료")
            save_checkpoint(
                path=LAST_CKPT,
                generator=generator,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_dist=best_dist,
                loss_history=loss_history,
            )
            break

    # 마지막 체크포인트 저장 (실제 마지막 epoch 기록)
    save_checkpoint(
        path=LAST_CKPT,
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_dist=best_dist,
        loss_history=loss_history,
    )
    print(f"\n[train] 완료. best_dist={best_dist:.4f}")
    print(f"[train] best.pt : {BEST_CKPT}")
    print(f"[train] last.pt : {LAST_CKPT}")
    print(f"[train] log     : {LOG_PATH}")


if __name__ == "__main__":
    main()
