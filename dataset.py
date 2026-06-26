"""
STEP 1: 오디오 데이터셋 로드 + 증강
KSS wav 파일 로드, augmentation으로 데이터 확장
"""

import os
import random
from math import gcd

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16_000
MAX_SAMPLES = SAMPLE_RATE * 4   # 4초 고정 길이


def _load_wav(path: str) -> np.ndarray:
    """WAV → 16kHz mono float32, [-1, 1] clamp"""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            g = gcd(SAMPLE_RATE, sr)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
        except ImportError:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
            audio = t.squeeze(0).numpy()
    return np.clip(audio, -1.0, 1.0)


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """길이를 length 샘플로 고정 (pad: zero, trim: 앞에서 자름)"""
    if len(audio) >= length:
        start = random.randint(0, len(audio) - length)
        return audio[start:start + length].copy()
    pad = length - len(audio)
    return np.pad(audio, (0, pad), mode="constant")


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    time stretch via resampling trick (scipy 기반)
    rate > 1.0 → 빨라짐(짧아짐), rate < 1.0 → 느려짐(길어짐)
    짧아진 경우 zero pad 없이 원본에서 필요한 만큼만 쓰도록 원본 반환
    """
    try:
        from scipy.signal import resample_poly
        factor_num = int(round(rate * 100))
        factor_den = 100
        g = gcd(factor_num, factor_den)
        stretched = resample_poly(audio, factor_den // g, factor_num // g)
        stretched = stretched.astype(np.float32)
        return stretched  # 길이는 이후 _pad_or_trim이 맞춤
    except Exception:
        return audio


def _add_gaussian_noise(audio: np.ndarray, snr_db: float = 40.0) -> np.ndarray:
    """가우시안 노이즈 추가 (SNR 40dB → 거의 안들림)"""
    signal_power = np.mean(audio ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def _augment(audio: np.ndarray) -> np.ndarray:
    """
    랜덤 augmentation 1~2개 적용
    - time stretch: 0.9~1.1x
    - gaussian noise: SNR 35~45dB
    - amplitude scaling: 0.8~1.0
    """
    ops = random.sample(["stretch", "noise", "amplitude"], k=random.randint(1, 2))
    for op in ops:
        if op == "stretch":
            rate = random.uniform(0.9, 1.1)
            audio = _time_stretch(audio, rate)
        elif op == "noise":
            snr = random.uniform(35.0, 45.0)
            audio = _add_gaussian_noise(audio, snr)
        elif op == "amplitude":
            scale = random.uniform(0.8, 1.0)
            audio = (audio * scale).astype(np.float32)
    return audio


def _collect_wav_paths(root: str) -> list:
    """
    root 아래 모든 wav 파일 수집 (서브폴더 포함).
    KSS 구조: kss/1/*.wav, kss/2/*.wav, kss/3/*.wav, kss/4/*.wav
    clone/adv/pgd 파일명 제외.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if (f.endswith(".wav")
                    and not f.startswith(".")
                    and "clone" not in f
                    and "adv" not in f
                    and "pgd" not in f):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


class AudioDataset(Dataset):
    """
    KSS wav 파일 로드 + augmentation

    Parameters
    ----------
    wav_dir     : KSS 루트 디렉토리 (서브폴더 1~4 자동 탐색)
                  단일 폴더도 동작
    augment     : True면 augmentation 적용
    max_samples : 오디오 고정 길이 (샘플 수), 기본 4초
    """

    def __init__(
        self,
        wav_dir: str,
        augment: bool = True,
        max_samples: int = MAX_SAMPLES,
    ):
        self.wav_dir = wav_dir
        self.augment = augment
        self.max_samples = max_samples

        self.paths = _collect_wav_paths(wav_dir)
        assert len(self.paths) > 0, f"WAV 파일 없음: {wav_dir}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        audio = _load_wav(path)

        if self.augment:
            audio = _augment(audio)

        audio = _pad_or_trim(audio, self.max_samples)
        return torch.from_numpy(audio), path   # (MAX_SAMPLES,) float32, str

    @property
    def num_files(self):
        return len(self.paths)


def build_dataloader(
    wav_dir: str,
    batch_size: int = 8,
    augment: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader 생성 (Kaggle: num_workers=0 기본)"""
    dataset = AudioDataset(wav_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
