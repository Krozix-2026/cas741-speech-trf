"""Generate high-res gammatone spectrograms"""
from eelbrain import *
from trftools import gammatone_bank

from settings import WAV_DIR, GAMMATONE_DIR


for wav_path in WAV_DIR.glob('*.wav'):
    dst = GAMMATONE_DIR / f'{wav_path.stem}.pickle'
    if dst.exists():
        continue
    wav = load.wav(wav_path)
    gt = gammatone_bank(wav, 20, 5000, 256, location='left', pad=False)
    gt = resample(gt, 1000)
    save.pickle(gt, dst)
