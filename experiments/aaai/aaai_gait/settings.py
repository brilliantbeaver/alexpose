"""One root .env, reproducible defaults, and no credentials in notebooks."""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv as load_env, find_dotenv as find_env

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
# An explicit path avoids loading a different .env from a notebook's launch folder.
load_env(ROOT / '.env', override=False)

@dataclass(frozen=True)
class Settings:
    demo: bool = os.getenv('AAAI_DEMO', '1') == '1'
    seed: int = int(os.getenv('AAAI_SEED', '42'))
    epochs: int = int(os.getenv('AAAI_EPOCHS', '3'))
    batch_size: int = int(os.getenv('AAAI_BATCH_SIZE', '16'))
    target_fps: float = float(os.getenv('AAAI_TARGET_FPS', '15'))
    frames: int = 32
    stride: int = 16
    frame_number_base: int = int(os.getenv('AAAI_FRAME_NUMBER_BASE', '1'))
    artifact_dir: Path = EXPERIMENT / 'artifacts'
    pose_dir: Path = EXPERIMENT / 'artifacts' / 'poses'

    @property
    def run_dir(self):
        return self.artifact_dir / ('synthetic_demo' if self.demo else 'gavd')
