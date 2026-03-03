"""
Download datasets for the denoise-audio project using kagglehub.

Datasets:
  - Clean speech : VIVOS Vietnamese Speech Corpus for ASR
                   (kynthesis/vivos-vietnamese-speech-corpus-for-asr)
  - Noise        : DEMAND Noise Dataset
                   (chrisfilo/demand)
  - Noise (opt.) : MUSAN Dataset
                   (dogrose/musan-dataset)

Usage:
    python scripts/download_dataset.py                  # download all
    python scripts/download_dataset.py --clean          # clean speech only
    python scripts/download_dataset.py --noise          # noise datasets only
    python scripts/download_dataset.py --no-musan       # skip optional MUSAN
"""

import argparse
import shutil
import sys
from pathlib import Path

import kagglehub
import yaml

# Root of the project (one level above this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"


# ---------------------------------------------------------------------------
# Load dataset registry from configs/data_config.yaml
# ---------------------------------------------------------------------------
def load_datasets() -> dict[str, dict]:
    """Build a flat {id -> info} registry from data_config.yaml."""
    with open(DATA_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    registry: dict[str, dict] = {}
    for category in ("clean", "noise"):
        for entry in cfg.get(category, []):
            registry[entry["id"]] = {
                "handle": entry["kaggle_handle"],
                "dest": entry["local_path"],
                "description": entry["description"],
                "required": entry.get("required", True),
                "category": category,
            }
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_dest(relative: str) -> Path:
    return PROJECT_ROOT / relative


def download_dataset(key: str, info: dict) -> bool:
    """Download a single dataset and copy it to the target directory."""
    dest = resolve_dest(info["dest"])
    print(f"\n{'='*60}")
    print(f"  Dataset : {info['description']}")
    print(f"  Handle  : {info['handle']}")
    print(f"  Target  : {dest}")
    print(f"{'='*60}")

    # Skip if already downloaded
    if dest.exists() and any(dest.iterdir()):
        print(f"[SKIP] Directory already exists and is non-empty: {dest}")
        return True

    try:
        print("[INFO] Downloading via kagglehub …")
        cache_path = kagglehub.dataset_download(info["handle"])
        cache_path = Path(cache_path)
        print(f"[INFO] Cached at: {cache_path}")

        # Copy from kagglehub cache to project data directory
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Copying files to {dest} …")
        shutil.copytree(src=cache_path, dst=dest, dirs_exist_ok=True)
        print(f"[OK] {info['description']} saved to {dest}")
        return True

    except Exception as exc:
        level = "[ERROR]" if info["required"] else "[WARN]"
        print(f"{level} Failed to download '{key}': {exc}")
        if info["required"]:
            return False
        print("[INFO] Skipping optional dataset and continuing.")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets for the denoise-audio project."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--clean",
        action="store_true",
        help="Download clean speech dataset only.",
    )
    group.add_argument(
        "--noise",
        action="store_true",
        help="Download noise datasets only (DEMAND + MUSAN).",
    )
    parser.add_argument(
        "--no-musan",
        action="store_true",
        help="Skip the optional MUSAN dataset.",
    )
    return parser.parse_args()


def select_datasets(args: argparse.Namespace, datasets: dict[str, dict]) -> list[str]:
    all_ids = list(datasets.keys())
    clean_ids = [k for k, v in datasets.items() if v["category"] == "clean"]
    noise_ids = [k for k, v in datasets.items() if v["category"] == "noise"]

    if args.clean:
        return clean_ids
    if args.noise:
        selected = noise_ids
    else:
        selected = all_ids  # default: everything

    if args.no_musan:
        selected = [k for k in selected if k != "musan"]
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    datasets = load_datasets()
    selected = select_datasets(args, datasets)

    print(f"\nConfig : {DATA_CONFIG}")
    print(f"Datasets to download: {', '.join(selected)}")

    failed = []
    for key in selected:
        info = datasets[key]
        success = download_dataset(key, info)
        if not success:
            failed.append(key)

    print(f"\n{'='*60}")
    if failed:
        print(f"[FAIL] The following required datasets could not be downloaded: {failed}")
        sys.exit(1)
    else:
        print("[DONE] All selected datasets downloaded successfully.")
        for key in selected:
            dest = resolve_dest(datasets[key]["dest"])
            print(f"  {datasets[key]['description']:40s} -> {dest}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
