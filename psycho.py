"""
STEP 3: 심리음향 마스킹 모듈
- compute_masking_threshold: ISO/IEC 11172-3 기반 임계값 계산 (pgd_verify.py 재활용)
- soft_psycho_loss: 학습 중 gradient 유지되는 soft constraint (torch STFT)
- hard_project: 추론 시 hard projection (numpy, pgd_verify 재활용)
"""

import sys
import os
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

SPL_REF    = 96.0
ATH_OFFSET = -60.0


def _ath_db(freqs: np.ndarray) -> np.ndarray:
    f = np.maximum(freqs, 20.0) / 1000.0
    ath = (
        3.64 * f ** (-0.8)
        - 6.5  * np.exp(-0.6 * (f - 3.3) ** 2)
        + 1e-3 * f ** 4
    )
    return np.clip(ath + ATH_OFFSET, -10.0, 96.0)


def _bark(freqs: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)


def compute_masking_threshold(audio: np.ndarray) -> np.ndarray:
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    _, _, stft = scipy.signal.stft(
        audio, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT, padded=True,
    )
    mag = np.abs(stft).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10)) + SPL_REF
    ath_db = _ath_db(freqs)
    bark = _bark(freqs)
    n_freq = len(freqs)
    spreading = np.zeros((n_freq, n_freq), dtype=np.float32)
    for i in range(n_freq):
        dz = bark - bark[i]
        spreading[i] = np.where(dz >= 0,
                                10.0 ** (-27.0 * dz / 10.0),
                                10.0 ** (  6.0 * dz / 10.0))
    power = 10.0 ** (mag_db / 10.0)
    masked_power = spreading.T @ power
    masking_db = 10.0 * np.log10(np.maximum(masked_power, 1e-30))
    threshold_db = np.maximum(ath_db[:, None], masking_db)
    threshold = 10.0 ** ((threshold_db - SPL_REF) / 20.0)
    return threshold.astype(np.float32)


def _project_psychoacoustic(delta_np: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    window = np.hanning(WIN_LENGTH).astype(np.float32)
    _, _, delta_stft = scipy.signal.stft(
        delta_np, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT, padded=True,
    )
    n_frames_stft   = delta_stft.shape[1]
    n_frames_thresh = threshold.shape[1]
    if n_frames_thresh < n_frames_stft:
        pad = n_frames_stft - n_frames_thresh
        threshold = np.concatenate([threshold, np.tile(threshold[:, -1:], (1, pad))], axis=1)
    elif n_frames_thresh > n_frames_stft:
        threshold = threshold[:, :n_frames_stft]
    mag = np.abs(delta_stft)
    phase = np.angle(delta_stft)
    mag_clipped = np.minimum(mag, threshold)
    delta_stft_proj = mag_clipped * np.exp(1j * phase)
    _, delta_proj = scipy.signal.istft(
        delta_stft_proj, fs=SAMPLE_RATE, window=window,
        nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH, nfft=N_FFT,
    )
    n = len(delta_np)
    if len(delta_proj) >= n:
        delta_proj = delta_proj[:n]
    else:
        delta_proj = np.pad(delta_proj, (0, n - len(delta_proj)))
    return delta_proj.astype(np.float32)

SAMPLE_RATE = 16_000
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400


def soft_psycho_loss(delta: torch.Tensor, threshold_np: np.ndarray) -> torch.Tensor:
    """
    학습 중 gradient 유지되는 심리음향 soft constraint.

    STFT magnitude가 threshold를 초과하는 부분에 페널티.
    torch.stft 사용 → gradient가 delta까지 흐름.

    Parameters
    ----------
    delta        : (T,) float32 Tensor, perturbation (단일 샘플)
    threshold_np : (N_FREQ, n_frames) float32 numpy, compute_masking_threshold 출력

    Returns
    -------
    loss : scalar Tensor, gradient 연결됨
           초과분이 없으면 0.0
    """
    device = delta.device

    window = torch.hann_window(WIN_LENGTH, device=device)

    # torch.stft: (T,) → (N_FREQ, n_frames) complex
    stft = torch.stft(
        delta,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=False,   # scipy compute_masking_threshold와 프레임 수 일치
        return_complex=True,
    )  # (N_FREQ, n_frames) complex

    # magnitude (복소수 절댓값, gradient 유지)
    mag = stft.abs() + 1e-12  # (N_FREQ, n_frames)

    # threshold를 tensor로 변환 + 크기 맞춤
    thresh = torch.from_numpy(threshold_np).to(device)  # (N_FREQ, n_frames_thresh)

    n_frames_stft  = mag.shape[1]
    n_frames_thresh = thresh.shape[1]

    if n_frames_thresh < n_frames_stft:
        pad = n_frames_stft - n_frames_thresh
        thresh = torch.cat([thresh, thresh[:, -1:].expand(-1, pad)], dim=1)
    elif n_frames_thresh > n_frames_stft:
        thresh = thresh[:, :n_frames_stft]

    # 초과분에만 페널티 (ReLU)
    exceed = F.relu(mag - thresh)  # (N_FREQ, n_frames)
    return exceed.mean()


def hard_project(delta_np: np.ndarray, threshold_np: np.ndarray) -> np.ndarray:
    """
    추론 시 hard projection (pgd_verify._project_psychoacoustic 재활용).

    Parameters
    ----------
    delta_np     : (T,) float32 numpy
    threshold_np : (N_FREQ, n_frames) float32 numpy

    Returns
    -------
    delta_proj : (T,) float32 numpy, threshold 이하로 clamp됨
    """
    return _project_psychoacoustic(delta_np, threshold_np)


def compute_threshold(audio_np: np.ndarray) -> np.ndarray:
    """
    compute_masking_threshold wrapper.

    Parameters
    ----------
    audio_np : (T,) float32 numpy, 16kHz

    Returns
    -------
    threshold : (N_FREQ, n_frames) float32 numpy
    """
    return compute_masking_threshold(audio_np)
