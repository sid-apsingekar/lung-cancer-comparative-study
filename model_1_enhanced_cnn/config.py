from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

# model_1_enhanced_cnn folder
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset location
DATASET_DIR = PROJECT_ROOT / "chest_ctscan" / "Data_1"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

# Output folders
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)
CHANNELS = 3
NUM_CLASSES = 4

# ==========================================
# TRAINING SETTINGS
# ==========================================

BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20

# ==========================================
# CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]