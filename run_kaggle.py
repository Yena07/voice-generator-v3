import sys, os, shutil

# ── 설정 (본인 환경에 맞게 수정) ──────────────────────────────
# Kaggle Dataset에 올린 코드 경로 (Dataset 이름에 맞게 수정)
PGD_PATH = '/kaggle/input/datasets/본인아이디/voice-pgd-generator'
# 체크포인트 업로드용 임시 폴더
CKPT_DATASET_DIR = '/kaggle/working/ckpt_upload'
# KSS 데이터셋 경로 (Dataset 이름에 맞게 수정)
KSS_PATH = '/kaggle/input/datasets/본인아이디/voice-pgd/kss'
# Kaggle API 토큰 (본인 것으로 교체 — kaggle.json의 key 값)
KAGGLE_TOKEN = '여기에_본인_kaggle_api_key_입력'
# ─────────────────────────────────────────────────────────────

print(f'[run_kaggle] PGD_PATH={PGD_PATH}')
sys.path.insert(0, PGD_PATH)
os.chdir(PGD_PATH)

import run_generator_train as rgt
rgt.KSS_DIR       = KSS_PATH
rgt.CAMPPLUS_ONNX = f'{PGD_PATH}/campplus.onnx'
rgt.SAMPLE_WAV    = f'{KSS_PATH}/1/1_0000.wav'
rgt.OUTPUT_DIR    = '/kaggle/working/checkpoints'
rgt.BEST_CKPT     = '/kaggle/working/checkpoints/best.pt'
rgt.LAST_CKPT     = '/kaggle/working/checkpoints/last.pt'
rgt.LOG_PATH      = '/kaggle/working/checkpoints/train_log.json'
rgt.QUICK_OUT     = '/kaggle/working/checkpoints/quick_output'

# 임베딩 캐시: Dataset에 있으면 working으로 복사해서 재사용
CACHE_DATASET = f'{PGD_PATH}/emb_cache.pt'
CACHE_WORKING = '/kaggle/working/checkpoints/emb_cache.pt'

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
if not os.path.exists(CACHE_WORKING) and os.path.exists(CACHE_DATASET):
    import torch
    cache = torch.load(CACHE_DATASET, map_location='cpu')
    new_cache = {}
    for old_key, val in cache.items():
        parts = old_key.replace('\\', '/').split('/')
        rel = '/'.join(parts[-2:])
        new_cache[f'{KSS_PATH}/{rel}'] = val
    torch.save(new_cache, CACHE_WORKING)
    print(f'[cache] emb_cache.pt remapped and saved ({len(new_cache)} entries)')
elif os.path.exists(CACHE_WORKING):
    import torch
    cache = torch.load(CACHE_WORKING, map_location='cpu')
    first_key = next(iter(cache))
    if first_key.startswith(KSS_PATH):
        print('[cache] emb_cache.pt already in working')
    else:
        print(f'[cache] remapping keys: {first_key[:60]} -> {KSS_PATH}/...')
        new_cache = {}
        for old_key, val in cache.items():
            parts = old_key.replace('\\', '/').split('/')
            rel = '/'.join(parts[-2:])
            new_cache[f'{KSS_PATH}/{rel}'] = val
        torch.save(new_cache, CACHE_WORKING)
        print(f'[cache] remapped {len(new_cache)} entries')
else:
    print('[cache] no cache found, will compute from scratch')


def upload_checkpoint(best_pt_path: str):
    """best.pt를 Kaggle Dataset에 업로드 (best.pt 항상 최신 유지)"""
    import subprocess, json
    os.makedirs(CKPT_DATASET_DIR, exist_ok=True)
    shutil.copy(best_pt_path, f'{CKPT_DATASET_DIR}/best.pt')

    meta = {
        "title": "voice-pgd-checkpoints",
        "id": "본인아이디/voice-pgd-checkpoints",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}]
    }
    with open(f'{CKPT_DATASET_DIR}/dataset-metadata.json', 'w') as f:
        json.dump(meta, f)

    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = KAGGLE_TOKEN
    result = subprocess.run(
        ['kaggle', 'datasets', 'version', '-p', CKPT_DATASET_DIR, '-m', 'auto checkpoint'],
        capture_output=True, text=True, env=env
    )
    if 'successfully' in result.stdout or 'created' in result.stdout.lower():
        print(f'[ckpt] best.pt uploaded to dataset')
    else:
        print(f'[ckpt] upload failed: {result.stdout} {result.stderr}')


rgt.on_best_saved = upload_checkpoint
rgt.main()
