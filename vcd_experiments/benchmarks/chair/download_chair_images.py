#!/usr/bin/env python3
"""
Downloads the 500 COCO val2014 images for the CHAIR benchmark (~50MB total).
Useful for fresh environments without a pre-existing COCO val2014 directory.
"""
import os
import sys
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(CURRENT_DIR, "selected_chair_val2014_seed2027.json")
DEFAULT_TARGET_DIR = os.path.join(CURRENT_DIR, "coco_chair_images")
COCO_BASE_URL = "http://images.cocodataset.org/val2014"

def download_image(args):
    img_name, target_dir = args
    target_path = os.path.join(target_dir, img_name)
    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        return True
    url = f"{COCO_BASE_URL}/{img_name}"
    try:
        urllib.request.urlretrieve(url, target_path)
        return True
    except Exception as e:
        print(f"Error downloading {img_name}: {e}")
        return False

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET_DIR
    os.makedirs(target_dir, exist_ok=True)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"[Dataset] Downloading {len(samples)} benchmark sample images to: {target_dir}")

    tasks = [(item["file_name"], target_dir) for item in samples]

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(download_image, tasks), total=len(tasks), desc="Downloading images"))

    success = sum(results)
    print(f"\n[Dataset] Finished: {success}/{len(samples)} images ready at: {target_dir}")
    print(f"Note: run_chair.py will automatically detect this directory, or you can specify it in configs/data_paths.yaml.")

if __name__ == "__main__":
    main()
