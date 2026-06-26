"""
STEP 5: Generator Loss 함수

Total Loss = w_cam * L_cam + w_snr * L_snr + w_psycho * L_psycho + w_linf * L_linf

L_cam   : -mean(cosine_dist(adv_emb_i, orig_emb_i))  배치 평균, 최소화 = dist 최대화
L_snr   : mean(relu(SNR_target - SNR_i))              SNR 28dB 미달 시 페널티
L_psycho: soft_psycho_loss 배치 평균                  심리음향 초과분 페널티
L_linf  : relu(delta.abs() - eps).mean()              L-inf soft (Tanh 이중 안전장치)
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# psycho 모듈 (STEP 3)
_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)

# pgd_verify.py 경로
_PGD_DIR = os.path.dirname(_GEN_DIR)
if _PGD_DIR not in sys.path:
    sys.path.insert(0, _PGD_DIR)

from psycho import soft_psycho_loss, compute_threshold
from features import extract_kaldi_fbank

SNR_TARGET  = 28.0   # dB, PGD 최고 결과(#3) 기준
EPS_DEFAULT = 0.01


# ── 유틸 ──────────────────────────────────────────────────────

def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    코사인 거리 = 1 - cosine_similarity
    a, b : (D,) Tensor
    반환 : scalar [0, 2], 클수록 임베딩이 멀어짐
    """
    a_n = F.normalize(a.unsqueeze(0), dim=1)
    b_n = F.normalize(b.unsqueeze(0), dim=1)
    return 1.0 - (a_n * b_n).sum()


def campplus_embed_single(campplus_model, audio_t: torch.Tensor) -> torch.Tensor:
    """
    CAM++ 임베딩 추출 (단일 샘플, gradient 유지)

    Parameters
    ----------
    campplus_model : onnx2torch 변환 모델
    audio_t        : (1, T) float32 Tensor

    Returns
    -------
    emb : (192,) Tensor, gradient 연결됨
    """
    feat = extract_kaldi_fbank(audio_t)   # (T', 80)
    emb  = campplus_model(feat.unsqueeze(0))        # (1, 192)
    return emb.squeeze(0)                           # (192,)


def snr_db(original: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    SNR 계산 (단일 샘플)

    Parameters
    ----------
    original : (T,) Tensor
    delta    : (T,) Tensor

    Returns
    -------
    snr : scalar Tensor (dB)
    """
    signal_power = original.pow(2).mean() + 1e-12
    noise_power  = delta.pow(2).mean()   + 1e-12
    return 10.0 * torch.log10(signal_power / noise_power)


# ── 개별 Loss 항 ──────────────────────────────────────────────

def l_cam(campplus_model, adv_batch: torch.Tensor, orig_embs: torch.Tensor) -> torch.Tensor:
    """
    CAM++ cosine dist 최대화 loss (배치 평균)

    Parameters
    ----------
    campplus_model : CAM++ 모델
    adv_batch      : (B, T) adv waveform, gradient 연결됨
    orig_embs      : (B, 192) 원본 임베딩, detach됨

    Returns
    -------
    loss : scalar, 음수 (minimize → dist 최대화)
    """
    dists = []
    for i in range(adv_batch.shape[0]):
        adv_emb = campplus_embed_single(campplus_model, adv_batch[i:i+1])
        d = cosine_dist(adv_emb, orig_embs[i])
        dists.append(d)
    return -torch.stack(dists).mean()


def l_snr(original_batch: torch.Tensor, delta_batch: torch.Tensor,
          snr_target: float = SNR_TARGET) -> torch.Tensor:
    """
    SNR soft constraint loss (배치 평균)
    SNR >= snr_target이면 loss=0, 미달 시 페널티

    Parameters
    ----------
    original_batch : (B, T) 원본 waveform
    delta_batch    : (B, T) perturbation
    snr_target     : 목표 SNR (dB)

    Returns
    -------
    loss : scalar >= 0
    """
    penalties = []
    for i in range(original_batch.shape[0]):
        snr = snr_db(original_batch[i], delta_batch[i])
        deficit = F.relu(snr_target - snr)           # 목표까지 부족한 dB
        ratio = deficit / snr_target                 # 0~1 정규화
        penalty = ratio ** 2 * snr_target            # 제곱 곡선: 목표 근처에서 급감
        penalties.append(penalty)
    return torch.stack(penalties).mean()


def l_psycho(delta_batch: torch.Tensor, orig_batch_np: np.ndarray) -> torch.Tensor:
    """
    심리음향 soft constraint loss (배치 평균)

    Parameters
    ----------
    delta_batch   : (B, T) Tensor, gradient 연결됨
    orig_batch_np : (B, T) numpy, 원본 오디오 (threshold 계산용)

    Returns
    -------
    loss : scalar >= 0
    """
    losses = []
    for i in range(delta_batch.shape[0]):
        threshold = compute_threshold(orig_batch_np[i])
        l = soft_psycho_loss(delta_batch[i], threshold)
        losses.append(l)
    return torch.stack(losses).mean()


def l_linf(delta_batch: torch.Tensor, eps: float = EPS_DEFAULT) -> torch.Tensor:
    """
    L-inf soft constraint
    - eps*0.8 이하: 약한 선형 패널티
    - eps*0.8 초과: 제곱 폭발 패널티
    perturbation이 eps 한계까지 적극적으로 사용되도록 유도
    """
    abs_delta = delta_batch.abs()
    threshold = eps * 0.8
    linear_part = F.relu(abs_delta - threshold) * 0.1
    quad_part   = F.relu(abs_delta - eps) ** 2 * 100.0
    return (linear_part + quad_part).mean()


# ── 통합 Loss ─────────────────────────────────────────────────

class GeneratorLoss:
    """
    Generator 전체 Loss 계산기

    Parameters
    ----------
    campplus_model : CAM++ 모델 (onnx2torch)
    eps            : L-inf 제약
    snr_target     : SNR 목표 (dB)
    w_cam          : CAM++ loss 가중치
    w_snr          : SNR loss 가중치
    w_psycho       : 심리음향 loss 가중치
    w_linf         : L-inf loss 가중치
    """

    def __init__(
        self,
        campplus_model,
        eps:        float = EPS_DEFAULT,
        snr_target: float = SNR_TARGET,
        w_cam:      float = 1.0,
        w_snr:      float = 0.1,
        w_psycho:   float = 0.05,
        w_linf:     float = 10.0,
    ):
        self.campplus_model = campplus_model
        self.eps        = eps
        self.snr_target = snr_target
        self.w_cam      = w_cam
        self.w_snr      = w_snr
        self.w_psycho   = w_psycho
        self.w_linf     = w_linf

    def __call__(
        self,
        adv_batch:      torch.Tensor,   # (B, T) adv wav, gradient 연결됨
        orig_batch:     torch.Tensor,   # (B, T) 원본 wav, detach
        orig_embs:      torch.Tensor,   # (B, 192) 원본 CAM++ 임베딩, detach
        orig_batch_np:  np.ndarray,     # (B, T) 원본 wav numpy (psycho용)
        delta_batch:    torch.Tensor,   # (B, T) perturbation, gradient 연결됨
    ) -> dict:
        """
        Returns
        -------
        dict:
            total  : 전체 loss (scalar)
            cam    : L_cam 값
            snr    : L_snr 값
            psycho : L_psycho 값
            linf   : L_linf 값
        """
        raw_cam    = l_cam(self.campplus_model, adv_batch, orig_embs)
        raw_snr    = l_snr(orig_batch, delta_batch, self.snr_target)
        raw_psycho = l_psycho(delta_batch, orig_batch_np)
        raw_linf   = l_linf(delta_batch, self.eps)

        # 동적 w_snr: 현재 배치 평균 SNR이 목표보다 낮을수록 w_snr 증가
        with torch.no_grad():
            cur_snr = torch.stack([
                snr_db(orig_batch[i], delta_batch[i])
                for i in range(orig_batch.shape[0])
            ]).mean().item()
        deficit_ratio = max(0.0, (self.snr_target - cur_snr) / self.snr_target)
        w_snr_dynamic = self.w_snr * (1.0 + deficit_ratio * 9.0)  # 최대 10x

        total = (self.w_cam    * raw_cam
               + w_snr_dynamic * raw_snr
               + self.w_psycho * raw_psycho
               + self.w_linf   * raw_linf)

        return {
            "total":  total,
            "cam":    raw_cam.detach(),     # 가중치 미적용 raw값 — 모니터링용
            "snr":    raw_snr.detach(),
            "psycho": raw_psycho.detach(),
            "linf":   raw_linf.detach(),
        }
