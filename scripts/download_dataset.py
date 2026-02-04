import os
import yaml
import subprocess
from pathlib import Path

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_from_kaggle():
    # Load config
    config = load_config()
    ds_config = config.get("dataset", {})
    sources = ds_config.get("sources", {})
    root_dir = Path(ds_config.get("root_dir", "data/raw"))
    
    if not sources:
        print("Error: No dataset sources found in config.")
        return

    for name, info in sources.items():
        kaggle_path = info.get("kaggle_path")
        if not kaggle_path:
            continue

        # Create subfolder for each dataset
        target_dir = root_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Downloading {name.upper()} dataset: {kaggle_path} ---")
        
        try:
            command = [
                "kaggle", "datasets", "download", 
                "-d", kaggle_path, 
                "-p", str(target_dir)
            ]
            
            if info.get("extract"):
                command.append("--unzip")
                
            subprocess.run(command, check=True)
            print(f"Successfully downloaded {name} to: {target_dir}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {name}: {e}")
        except FileNotFoundError:
            print("Error: 'kaggle' CLI not found. Run 'pip install kaggle'.")
            return

if __name__ == "__main__":
    download_from_kaggle()
