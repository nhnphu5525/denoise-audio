import shutil
from pathlib import Path

src = Path("data/raw/noise/musan/musan/noise")
dst = Path("data/processed/noise_realtime")

dst.mkdir(parents=True, exist_ok=True)

count = 0

for wav in src.rglob("*.wav"):
    shutil.copy(wav, dst / wav.name)
    count += 1

print(f"Copied {count} noise files")