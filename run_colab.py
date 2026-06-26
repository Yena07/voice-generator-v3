"""
Colab 학습 실행 드라이버 (run_kaggle.py의 Colab 버전)

run_generator_train.py의 전역 경로를 Colab 환경에 맞게 덮어쓴 뒤 학습을 실행한다.
코드 본체(run_generator_train / dataset / model / loss ...)는 수정하지 않는다.

경로는 ① 커맨드라인 인자 ② 환경변수 ③ 기본값 순으로 결정된다.
파일을 직접 고치지 않고 Colab 셀에서 경로를 넘길 수 있다.

실행 예:
  # 기본 경로 그대로
  !python /content/voice_generator/run_colab.py --epochs 30 --batch 16

  # 경로를 직접 지정 (데이터 폴더 이름이 다를 때)
  !python /content/voice_generator/run_colab.py \
      --data_dir "/content/drive/MyDrive/New_Sample/원천데이터" \
      --campplus "/content/drive/MyDrive/voice_pgd/campplus.onnx" \
      --output_dir "/content/drive/MyDrive/voice_gen_ckpt" \
      --epochs 30 --batch 16

  # 이어서 학습
  !python /content/voice_generator/run_colab.py --resume
"""

import sys
import os
import glob
import argparse

# ── 기본 경로 (환경에 맞게 바꾸거나, 위 예시처럼 인자로 넘겨도 됨) ──
DEFAULT_CODE_DIR   = "/content/voice_generator"
DEFAULT_DATA_DIR   = "/content/drive/MyDrive/New_Sample/원천데이터"
DEFAULT_CAMPPLUS   = "/content/drive/MyDrive/voice_pgd/campplus.onnx"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/voice_gen_ckpt"
# ─────────────────────────────────────────────────────────────

# 1) 경로 인자만 먼저 파싱하고, 나머지(--epochs 등)는 run_generator_train으로 넘김
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--code_dir")
_p.add_argument("--data_dir")
_p.add_argument("--campplus")
_p.add_argument("--output_dir")
_cfg, _rest = _p.parse_known_args()

CODE_DIR   = _cfg.code_dir   or os.environ.get("CODE_DIR",   DEFAULT_CODE_DIR)
DATA_DIR   = _cfg.data_dir   or os.environ.get("DATA_DIR",   DEFAULT_DATA_DIR)
CAMPPLUS   = _cfg.campplus   or os.environ.get("CAMPPLUS",   DEFAULT_CAMPPLUS)
OUTPUT_DIR = _cfg.output_dir or os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

# run_generator_train.parse_args()가 보게 될 argv는 경로 인자를 뺀 나머지만 남김
sys.argv = [sys.argv[0]] + _rest

# 2) 코드 경로 등록
assert os.path.isdir(CODE_DIR), f"코드 폴더 없음: {CODE_DIR}"
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

# 3) 사전 점검 (실수 빨리 잡기)
assert os.path.isdir(DATA_DIR), f"데이터 폴더 없음: {DATA_DIR}"
assert os.path.isfile(CAMPPLUS), (
    f"campplus.onnx 없음: {CAMPPLUS}\n"
    f"  → CAM++ ONNX 모델을 Drive에 올린 뒤 --campplus 경로를 맞춰주세요."
)

wavs = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
assert len(wavs) > 0, f"wav 파일이 없습니다: {DATA_DIR}"

print(f"[run_colab] wav {len(wavs)}개 발견")
print(f"[run_colab] CODE_DIR   = {CODE_DIR}")
print(f"[run_colab] DATA_DIR   = {DATA_DIR}")
print(f"[run_colab] CAMPPLUS   = {CAMPPLUS}")
print(f"[run_colab] OUTPUT_DIR = {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4) run_generator_train 전역 경로 덮어쓰기
import run_generator_train as rgt

rgt.KSS_DIR       = DATA_DIR
rgt.CAMPPLUS_ONNX = CAMPPLUS
rgt.OUTPUT_DIR    = OUTPUT_DIR
rgt.BEST_CKPT     = os.path.join(OUTPUT_DIR, "best.pt")
rgt.LAST_CKPT     = os.path.join(OUTPUT_DIR, "last.pt")
rgt.LOG_PATH      = os.path.join(OUTPUT_DIR, "train_log.json")
rgt.QUICK_OUT     = os.path.join(OUTPUT_DIR, "quick_output")
rgt.SAMPLE_WAV    = wavs[0]   # 학습 후 샘플 추론용 wav 1개 자동 선택

# 5) 학습 실행
rgt.main()
