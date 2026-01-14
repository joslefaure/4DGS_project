"""
Configuration settings for 4DGS pipeline.
"""
import os

# ============================================
# PATHS - Edit these for your setup
# ============================================

# Path to V3 framework
V3_PATH = "/home/phd/12/josmyfaure/dip_makeup/4DGS_ntsec/V3_v2"

# Path to BackgroundMattingV2
BGMATTING_PATH = "/home/phd/12/josmyfaure/dip_makeup/4DGS_ntsec/BackgroundMattingV2"

# Data directories
DATA_DIR = "/home/phd/12/josmyfaure/dip_makeup/DIP"
OUTPUT_DIR = "/home/phd/12/josmyfaure/dip_makeup/DIP/output"

# ============================================
# TRAINING PARAMETERS
# ============================================

ITERATIONS = 16000
GROUP_SIZE = 20  # Frames per group (1 = independent frames)
SH_DEGREE = 0
WHITE_BACKGROUND = True

# ============================================
# DATA PARAMETERS
# ============================================

NUM_FRAMES = 300  # Total frames to process
FPS = 30
IMAGE_EXTENSION = "png"

# Camera IDs (exclude missing cameras)
CAMERA_IDS = [
    "001", "002", "003", "004", "005", "006",
    "008", "009", "010", "012",  # Note: 007, 011 missing
    "101", "102", "103", "104", "105", "106",
    "107", "108", "109", "110", "111", "112"
]

# ============================================
# DERIVED PATHS (don't edit)
# ============================================

def get_paths():
    """Get all derived paths."""
    return {
        "videos": os.path.join(DATA_DIR, "DIP"),
        "backgrounds": os.path.join(DATA_DIR, "cali+bg/bg"),
        "calibration": os.path.join(DATA_DIR, "cali+bg/0"),
        "nobg": os.path.join(DATA_DIR, "video_processed"),
        "frames": os.path.join(DATA_DIR, "frames"),
        "processed": os.path.join(DATA_DIR, "processed/DIP_processed"),
        "checkpoints": os.path.join(OUTPUT_DIR, "checkpoints"),
        "renders": os.path.join(OUTPUT_DIR, "renders"),
    }
