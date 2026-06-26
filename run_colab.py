"""
Colab 학습 실행 드라이버 (run_kaggle.py의 Colab 버전)

run_generator_train.py의 전역 경로를 Colab 환경에 맞게 덮어쓴 뒤 학습을 실행한다.
코드 본체(run_generator_train / dataset / model / loss ...)는 수정하지 않는다.

실행 예:
  !python /content/voice_generator/run_colab.py --epochs 30 --batch 16
  !python /content/voice_generator/run_colab.py --resume          # 이어서 학습

아래 ── 설정 ── 블록의 4개 경로만 본인 환경에 맞게 바꾸면 된다.
"""

import sys
import os
import glob

# ── 설정 (본인 환경에 맞게 수정) ──────────────────────────────
# 코드 폴더 (이 파일이 들어있는 voice_generator 폴더)
CODE_DIR   = "/content/voice_generator"
# wav 데이터 루트 — 하위 폴더를 재귀 탐색하므로 '원천데이터'를 그대로 가리키면 됨
DATA_DIR   = "/content/drive/MyDrive/New_Sample/원천데이터"
# CAM++ 화자인식 모델 (반드시 별도 준비해서 Drive에 올려둘 것)
CAMPPLUS   = "/content/drive/MyDrive/voice_pgd/campplus.onnx"
# 체크포인트/로그 저장 폴더 — Drive에 두면 런타임 끊겨도 보존됨
OUTPUT_DIR = "/content/drive/MyDrive/voice_gen_ckpt"
# ─────────────────────────────────────────────────────────────

# 1) 코드 경로 등록
assert os.path.isdir(CODE_DIR), f"코드 폴더 없음: {CODE_DIR}"
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

# 2) 사전 점검 (실수 빨리 잡기)
assert os.path.isdir(DATA_DIR), f"데이터 폴더 없음: {DATA_DIR}"
assert os.path.isfile(CAMPPLUS), (
    f"campplus.onnx 없음: {CAMPPLUS}\n"
    f"  → CAM++ ONNX 모델을 Drive에 올린 뒤 위 CAMPPLUS 경로를 맞춰주세요."
)

wavs = glob.glob(os.path.join(DATA_DIR, "**", "*.wav"), recursive=True)
assert len(wavs) > 0, f"wav 파일이 없습니다: {DATA_DIR}"
print(f"[run_colab] wav {len(wavs)}개 발견")
print(f"[run_colab] DATA_DIR   = {DATA_DIR}")
print(f"[run_colab] CAMPPLUS   = {CAMPPLUS}")
print(f"[run_colab] OUTPUT_DIR = {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3) run_generator_train 전역 경로 덮어쓰기
import run_generator_train as rgt

rgt.KSS_DIR       = DATA_DIR
rgt.CAMPPLUS_ONNX = CAMPPLUS
rgt.OUTPUT_DIR    = OUTPUT_DIR
rgt.BEST_CKPT     = os.path.join(OUTPUT_DIR, "best.pt")
rgt.LAST_CKPT     = os.path.join(OUTPUT_DIR, "last.pt")
rgt.LOG_PATH      = os.path.join(OUTPUT_DIR, "train_log.json")
rgt.QUICK_OUT     = os.path.join(OUTPUT_DIR, "quick_output")
rgt.SAMPLE_WAV    = wavs[0]   # 학습 후 샘플 추론용 wav 1개 자동 선택

# 4) 학습 실행 (--epochs 등 인자는 sys.argv로 그대로 전달됨)
rgt.main()
