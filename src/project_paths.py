from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "datasets"))
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", PROJECT_ROOT / "artifacts" / "checkpoints"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", PROJECT_ROOT / "artifacts" / "outputs"))

FOUNDATION_SRC = PROJECT_ROOT / "src" / "foundation"
GENERATION_SRC = PROJECT_ROOT / "src" / "generation"

FOUNDATION_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "foundation"
GENERATION_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "generation"
PROBE_ROOT = OUTPUT_ROOT / "foundation" / "probes"
FEATURE_ROOT = OUTPUT_ROOT / "foundation" / "features"
GENERATION_OUTPUT_ROOT = OUTPUT_ROOT / "generation"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

