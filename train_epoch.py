"""
STEP 6-B: 단일 epoch 학습 루프
- train_one_epoch: 배치 순회 → forward → loss → backward → step
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from train_utils import get_batch_embeddings


def train_one_epoch(
    generator,
    mel_extractor,
    campplus_model,
    criterion,
    optimizer,
    loader: DataLoader,
    emb_cache: dict,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """
    단일 epoch 학습.

    Parameters
    ----------
    generator      : PerturbationGenerator (train 모드)
    mel_extractor  : MelExtractor (eval 모드 — 학습 파라미터 없음)
    campplus_model : CAM++ (eval 모드 — 파라미터 고정)
    criterion      : GeneratorLoss
    optimizer      : Adam
    loader         : DataLoader (path 반환 포함)
    emb_cache      : compute_orig_embeddings 반환값
    device         : torch.device
    grad_clip      : gradient clipping max norm

    Returns
    -------
    metrics : dict {
        "loss"   : float, 전체 loss 평균
        "cam"    : float, L_cam 평균 (음수)
        "snr"    : float, L_snr 평균
        "psycho" : float, L_psycho 평균
        "linf"   : float, L_linf 평균
        "dist"   : float, CAM++ cosine dist 평균 (방어 효과 지표)
        "n_batch": int,   처리한 배치 수
    }

    Notes
    -----
    - campplus_model은 eval + no_grad로 원본 임베딩 조회에만 사용
    - 학습 중 campplus forward는 adv wav에만 적용 (gradient 추적)
    - mel_extractor는 gradient 필요 없으므로 no_grad로 실행
    """
    generator.train()
    mel_extractor.eval()
    campplus_model.eval()

    sum_loss   = 0.0
    sum_cam    = 0.0
    sum_snr    = 0.0
    sum_psycho = 0.0
    sum_linf   = 0.0
    sum_dist   = 0.0
    n_batch    = 0
    n_total    = len(loader)

    import time
    t_epoch = time.time()

    for orig_batch, paths in loader:
        # orig_batch: (B, T) float32
        # paths: list of str (B개)
        orig_batch = orig_batch.to(device)                     # (B, T)
        orig_batch_np = orig_batch.detach().cpu().numpy()      # numpy (psycho용)

        # 원본 임베딩 캐시에서 꺼내기 (gradient 없음)
        orig_embs = get_batch_embeddings(emb_cache, paths, device)  # (B, 192)

        # mel 추출 (MelExtractor에 학습 파라미터 없음 → no_grad + detach로 명시)
        with torch.no_grad():
            mel = mel_extractor(orig_batch).detach()           # (B, 1, 80, T')

        # Generator forward → delta
        # mel은 detach됐지만 generator.parameters()가 requires_grad=True이므로
        # delta에는 generator 파라미터 기준 gradient가 생성됨
        delta = generator(mel)                                 # (B, T)

        # adv wav 생성
        adv = torch.clamp(orig_batch.detach() + delta, -1.0, 1.0)  # (B, T)

        # Loss 계산
        result = criterion(
            adv_batch=adv,
            orig_batch=orig_batch.detach(),
            orig_embs=orig_embs,
            orig_batch_np=orig_batch_np,
            delta_batch=delta,
        )

        # backward
        optimizer.zero_grad()
        result["total"].backward()
        # DataParallel 래핑 시 원본 파라미터 기준으로 clip
        params = (generator.module if hasattr(generator, "module") else generator).parameters()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        # dist = -l_cam (l_cam = -mean(dist) 이므로 역산, CAM++ forward 중복 방지)
        avg_dist = -result["cam"].item()

        # 누적
        sum_loss   += result["total"].item()
        sum_cam    += result["cam"].item()
        sum_snr    += result["snr"].item()
        sum_psycho += result["psycho"].item()
        sum_linf   += result["linf"].item()
        sum_dist   += avg_dist
        n_batch    += 1

        if n_batch % 10 == 0:
            elapsed = time.time() - t_epoch
            spd = elapsed / n_batch
            eta = spd * (n_total - n_batch)
            w_p = result.get("w_psycho", None)
            w_p_str = f" w_psycho={w_p:.3f}" if w_p is not None else ""
            print(
                f"  step {n_batch:4d}/{n_total} "
                f"loss={sum_loss/n_batch:+.4f} "
                f"dist={sum_dist/n_batch:.4f} "
                f"psycho={sum_psycho/n_batch:.5f}"
                f"{w_p_str} "
                f"spd={spd:.2f}s/batch "
                f"ETA={eta:.0f}s",
                flush=True
            )

    if n_batch == 0:
        return {"loss": 0, "cam": 0, "snr": 0, "psycho": 0,
                "linf": 0, "dist": 0, "n_batch": 0}

    return {
        "loss":    sum_loss   / n_batch,
        "cam":     sum_cam    / n_batch,
        "snr":     sum_snr    / n_batch,
        "psycho":  sum_psycho / n_batch,
        "linf":    sum_linf   / n_batch,
        "dist":    sum_dist   / n_batch,
        "n_batch": n_batch,
    }
