"""One validated configuration for single, batch and segmented inference."""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import re

MODES = ("full", "tiny", "tiny-long")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OUTPUT_SUFFIXES = VIDEO_SUFFIXES | {".gif"}


@dataclass(frozen=True)
class InferenceConfig:
    mode: str = "tiny"
    scale: float = 2.0
    seed: int = 0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    color_fix: bool = False
    fps: float | None = None
    quality: int = 10
    device: str = "cuda"
    dtype: str = "bf16"
    tile_dit: bool = False
    tile_vae: bool = False
    tile_size: int = 256
    overlap: int = 24
    keep_audio: bool = False
    model_version: str = "v1.1"
    model_dir: Path | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unsupported inference mode: {self.mode}")
        for name in ("scale", "sparse_ratio", "kv_ratio", "fps"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"--{name.replace('_', '-')} must be finite and greater than zero")
        if not re.fullmatch(r"cuda(?::\d+)?", self.device):
            raise ValueError("use cuda or cuda:N; the sparse backend requires CUDA")
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError("--dtype must be fp16 or bf16")
        for name in ("seed", "local_range", "quality", "tile_size", "overlap"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("--seed must be between 0 and 4294967295")
        if self.local_range <= 0 or not 0 <= self.quality <= 10:
            raise ValueError("--local-range must be positive and --quality must be in 0..10")
        if self.tile_size < 128 or self.tile_size % 32:
            raise ValueError("--tile-size must be a multiple of 32 and at least 128")
        if not 0 <= self.overlap < self.tile_size // 2:
            raise ValueError("--overlap must be nonnegative and less than half --tile-size")
        if self.tile_vae and self.mode != "full":
            raise ValueError("--tile-vae is supported only with --mode full")

    @classmethod
    def from_namespace(cls, args):
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})

    def as_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def validate_input(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir()):
            raise ValueError(f"No PNG/JPEG frames found in {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    elif path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return path


def validate_output(path, input_path=None, keep_audio=False):
    path = Path(path).expanduser().resolve()
    if path.is_dir() or not path.suffix:
        return path
    if path.suffix.lower() not in OUTPUT_SUFFIXES:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    if input_path is not None and path == Path(input_path).resolve():
        raise ValueError("Input and output must be different files")
    if keep_audio and path.suffix.lower() == ".gif":
        raise ValueError("GIF cannot preserve audio; select a video output")
    return path


def output_path(source, destination, config, *, directory=False):
    destination = Path(destination).expanduser().resolve() if directory else validate_output(destination, source, config.keep_audio)
    if directory or destination.is_dir() or not destination.suffix:
        # Include the source suffix to keep clip.mp4 and clip.mov distinct in batches.
        name = f"FlashVSR_{config.mode}_x{config.scale:g}_{Path(source).name}.mp4"
        destination = destination / name
    return validate_output(destination, source, config.keep_audio)


def validate_report_path(path, *protected):
    path = Path(path).expanduser().resolve()
    protected = {Path(item).resolve() for item in protected if item is not None}
    if path in protected or any(item.is_dir() and item in path.parents for item in protected):
        raise ValueError("Metrics/logs must use a different path from input and video output")
    if path.exists() and path.is_dir():
        raise ValueError(f"Report path is a directory: {path}")
    return path
