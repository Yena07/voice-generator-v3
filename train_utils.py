"""
STEP 6-A: 학습 유틸리티
- compute_orig_embeddings : 전체 데이터셋 원본 CAM++ 임베딩 사전 캐싱
- save_checkpoint         : 체크포인트 저장
- load_checkpoint         : 체크포인트 로드 (resume)
"""

import os
import torch

try:
    from loss_v2 import campplus_embed_single
except ImportError:
    from loss import campplus_embed_single
from dataset import MAX_SAMPLES


def compute_orig_embeddings(
    campplus_model,
    dataset,
    device: torch.device,
    cache_path: str = None,
) -> dict:
    """
    전체 데이터셋 원본 CAM++ 임베딩 사전 캐싱.

    학습 중 매 배치마다 원본 embed를 재계산하면 CAM++ forward가
    2배로 늘어남. 원본 오디오는 고정이므로 1회만 계산해서 캐싱.

    Parameters
    ----------
    campplus_model : CAM++ 모델 (eval 상태)
    dataset        : AudioDataset (augment=False로 호출할 것)
    device         : torch.device

    Returns
    -------
    emb_cache : dict { path_str -> (192,) Tensor, cpu에 저장 }
                배치 처리 시 path로 꺼내 씀
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  임베딩 캐시 로드: {cache_path}")
        emb_cache = torch.load(cache_path, map_location="cpu")
        print(f"  캐시 로드 완료: {len(emb_cache)}개")
        return emb_cache

    campplus_model.eval()
    emb_cache = {}

    print(f"  원본 임베딩 캐싱 중... (총 {len(dataset.paths)}개)")

    import soundfile as sf
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    with torch.no_grad():
        for i, path in enumerate(dataset.paths):
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

            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, MAX_SAMPLES)
            emb = campplus_embed_single(campplus_model, audio_t)        # (192,)
            emb_cache[path] = emb.cpu()

            if (i + 1) % 500 == 0 or (i + 1) == len(dataset.paths):
                print(f"    {i+1}/{len(dataset.paths)} 완료")

    print(f"  캐싱 완료: {len(emb_cache)}개")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb_cache, cache_path)
        print(f"  임베딩 캐시 저장: {cache_path}")

    return emb_cache


def get_batch_embeddings(
    emb_cache: dict,
    paths: list,
    device: torch.device,
) -> torch.Tensor:
    """
    배치의 path 리스트로 캐시에서 임베딩 꺼내기.

    Parameters
    ----------
    emb_cache : compute_orig_embeddings 반환값
    paths     : 현재 배치의 파일 경로 리스트 (B개)
    device    : torch.device

    Returns
    -------
    embs : (B, 192) Tensor, detach됨
    """
    embs = torch.stack([emb_cache[p] for p in paths]).to(device)
    return embs.detach()


def save_checkpoint(
    path: str,
    generator,
    optimizer,
    scheduler,
    epoch: int,
    best_dist: float,
    loss_history: list,
):
    """
    체크포인트 저장.

    Parameters
    ----------
    path         : 저장 경로 (.pt)
    generator    : PerturbationGenerator
    optimizer    : Adam
    scheduler    : LR scheduler
    epoch        : 현재 epoch (0-based)
    best_dist    : 지금까지 최고 CAM++ dist
    loss_history : epoch별 loss dict 리스트
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "best_dist":    best_dist,
        "loss_history": loss_history,
        "generator":    generator.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict() if scheduler is not None else None,
        "eps":          generator.eps,
    }, path)


def load_checkpoint(
    path: str,
    generator,
    optimizer,
    scheduler,
    device: torch.device,
):
    """
    체크포인트 로드 (resume).

    Parameters
    ----------
    path      : 체크포인트 경로 (.pt)
    generator : PerturbationGenerator (이미 초기화됨)
    optimizer : Adam
    scheduler : LR scheduler
    device    : torch.device

    Returns
    -------
    epoch        : 재개할 epoch 번호
    best_dist    : 저장된 최고 dist
    loss_history : 저장된 loss 기록
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt["best_dist"], ckpt.get("loss_history", [])
