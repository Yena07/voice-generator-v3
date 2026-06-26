"""
STEP 2: Feature Extractor
- mel_spectrogram: Generator 입력용 (1, 80, T')
- kaldi_fbank: CAM++ 입력용 (T, 80), gradient 유지
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi

SAMPLE_RATE = 16_000
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400
N_MELS      = 80


class MelExtractor(nn.Module):
    """
    waveform → mel spectrogram (Generator 입력용)

    Parameters
    ----------
    audio : (B, T) float32 Tensor

    Returns
    -------
    mel : (B, 1, N_MELS, T') float32
          1채널로 unsqueeze → Conv2d 입력 형태
    """

    def __init__(self):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B, T)
        mel = self.transform(audio)           # (B, N_MELS, T')
        mel = torch.log1p(mel)                # log 압축
        mel = mel.unsqueeze(1)                # (B, 1, N_MELS, T')
        return mel


def extract_kaldi_fbank(audio_t: torch.Tensor) -> torch.Tensor:
    """
    waveform → kaldi fbank (CAM++ 입력용, gradient 유지)

    Parameters
    ----------
    audio_t : (1, T) float32 Tensor, single sample

    Returns
    -------
    feat : (T', 80) float32 Tensor, gradient 연결됨

    Notes
    -----
    torchaudio.compliance.kaldi.fbank는 (1, T) 입력을 받아
    (T', 80) fbank를 반환하며 autograd 완전 지원.
    dither=0: 재현성 보장
    """
    feat = kaldi.fbank(
        audio_t,
        num_mel_bins=N_MELS,
        dither=0,
        sample_frequency=SAMPLE_RATE,
    )  # (T', 80)
    feat = feat - feat.mean(dim=0, keepdim=True)  # mean normalization
    return feat


def get_mel_time_frames(n_samples: int) -> int:
    """
    n_samples 길이 오디오의 mel T' 프레임 수 계산
    torchaudio MelSpectrogram 기본값: pad = n_fft // 2 = 256 (center=True)
    T' = (n_samples + 2*pad - n_fft) // hop_length + 1
       = (n_samples + 512 - 512) // 160 + 1
       = n_samples // 160 + 1
    """
    pad = N_FFT // 2  # center=True 기본값
    return (n_samples + 2 * pad - N_FFT) // HOP_LENGTH + 1
