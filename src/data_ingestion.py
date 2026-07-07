import os
import glob
import json
import shutil
import zipfile
import logging
import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import StratifiedShuffleSplit
import yaml

# ─────────────────────────────────────────────────────
# Improvement 1: Logging instead of print()
# ─────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/ingestion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger("data_ingestion")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_local_zip(zip_path, raw_dir):
    if not os.path.exists(zip_path):
        logger.error(f"ZIP not found: '{zip_path}'")
        raise FileNotFoundError(f"CRITICAL: '{zip_path}' not found in the root directory.")

    logger.info(f"Located {zip_path}. Extracting to {raw_dir}...")

    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    logger.info("Extraction complete.")


# ─────────────────────────────────────────────────────
# Improvement 2: Corrupted Image Detection
# ─────────────────────────────────────────────────────

def is_valid_image(path):
    """Return True if image can be opened and decoded by PIL."""
    try:
        with Image.open(path) as img:
            img.verify()        # catches corrupt headers / truncated files
        with Image.open(path) as img:
            img.load()          # fully decode pixel data
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────
# Improvement 5: Parallel File Copy
# ─────────────────────────────────────────────────────

def _copy_one(src, dst):
    shutil.copy2(src, dst)


def parallel_copy(file_paths, labels, target_dir, classes, max_workers=8):
    """Copy files to target_dir/<class>/ using a thread pool."""
    for cls in classes:
        os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for src, label in zip(file_paths, labels):
            dst = os.path.join(target_dir, label, os.path.basename(src))
            futures[executor.submit(_copy_one, src, dst)] = src

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed to copy {futures[future]}: {e}")


def process_data(config):
    raw_dir   = config["data"]["raw_dir"]
    train_dir = config["data"]["train_dir"]
    val_dir   = config["data"]["val_dir"]
    classes   = ["Mastitis", "Healthy"]

    # Collect all image paths
    X, y = [], []
    for class_name in classes:
        paths = (
            glob.glob(f"{raw_dir}/**/{class_name}/*.jpg", recursive=True) +
            glob.glob(f"{raw_dir}/**/{class_name}/*.png", recursive=True)
        )
        X.extend(paths)
        y.extend([class_name] * len(paths))

    logger.info(f"Total images discovered: {len(X)}")
    if len(X) == 0:
        raise FileNotFoundError("No images found. Check the folder structure inside the ZIP.")

    # ── Improvement 2: Remove corrupted images ────────────────────────────
    logger.info("Validating images for corruption...")
    valid_X, valid_y, corrupted = [], [], []
    for path, label in zip(X, y):
        if is_valid_image(path):
            valid_X.append(path)
            valid_y.append(label)
        else:
            corrupted.append(path)
            logger.warning(f"Corrupted image removed: {path}")

    logger.info(f"Valid images    : {len(valid_X)}")
    logger.info(f"Corrupted images: {len(corrupted)} (excluded)")

    # ── Improvement 3: Dataset Statistics Report ──────────────────────────
    logger.info("Computing dataset statistics...")
    widths, heights = [], []
    for path in valid_X:
        with Image.open(path) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)

    stats = {
        "total_images"       : len(valid_X),
        "class_distribution" : {cls: valid_y.count(cls) for cls in classes},
        "avg_width"          : int(sum(widths) / len(widths)),
        "avg_height"         : int(sum(heights) / len(heights)),
        "max_resolution"     : f"{max(widths)}x{max(heights)}",
        "min_resolution"     : f"{min(widths)}x{min(heights)}",
        "corrupted_removed"  : len(corrupted),
    }

    logger.info("=== Dataset Statistics ===")
    logger.info(f"  Total images  : {stats['total_images']}")
    for cls in classes:
        cnt = stats["class_distribution"][cls]
        logger.info(f"  ├─ {cls:<10}: {cnt} images  ({100*cnt//stats['total_images']}%)")
    logger.info(f"  Avg resolution: {stats['avg_width']} x {stats['avg_height']} px")
    logger.info(f"  Max resolution: {stats['max_resolution']} px")
    logger.info(f"  Min resolution: {stats['min_resolution']} px")

    # Stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=config["data"]["train_split"], random_state=42
    )
    for train_idx, val_idx in sss.split(valid_X, valid_y):
        X_train = [valid_X[i] for i in train_idx]
        X_val   = [valid_X[i] for i in val_idx]
        y_train = [valid_y[i] for i in train_idx]
        y_val   = [valid_y[i] for i in val_idx]

    # Clear old split dirs
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # ── Improvement 5: Parallel file copy ────────────────────────────────
    logger.info("Copying train set (parallel)...")
    parallel_copy(X_train, y_train, train_dir, classes)

    logger.info("Copying validation set (parallel)...")
    parallel_copy(X_val, y_val, val_dir, classes)

    logger.info("\n=== Data Processing Execution Verification ===")
    logger.info(f"Train Set Total:      {len(X_train)} images")
    logger.info(f"  └─ Mastitis:        {y_train.count('Mastitis')}")
    logger.info(f"  └─ Healthy:         {y_train.count('Healthy')}")
    logger.info(f"Validation Set Total: {len(X_val)} images")
    logger.info(f"  └─ Mastitis:        {y_val.count('Mastitis')}")
    logger.info(f"  └─ Healthy:         {y_val.count('Healthy')}")

    # ── Improvement 4: Metadata JSON ─────────────────────────────────────
    metadata = {
        "generated_at"  : datetime.datetime.now().isoformat(),
        "dataset_stats" : stats,
        "split": {
            "train_total"     : len(X_train),
            "val_total"       : len(X_val),
            "train_per_class" : {cls: y_train.count(cls) for cls in classes},
            "val_per_class"   : {cls: y_val.count(cls)   for cls in classes},
        }
    }

    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    meta_path = os.path.join(config["data"]["processed_dir"], "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Metadata saved → {meta_path}")


if __name__ == "__main__":
    cfg = load_config()
    logger.info("Initializing Offline Data Ingestion Layer...")

    zip_file_path = "dataset.zip"

    extract_local_zip(zip_file_path, cfg["data"]["raw_dir"])
    process_data(cfg)