"""
Utility helper functions used across the Image Classification Enhancement System.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy may be missing in minimal environments
    np = None

from config import RESULTS_DIR, WEIGHTS_DIR


def create_directories() -> None:
    """
    Create required directories for weights and results if they do not exist.
    """
    for folder in (WEIGHTS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def print_separator(title: str = "") -> None:
    """
    Print a simple separator line with an optional title.
    """
    if title:
        print(f"\n===== {title} =====")
    else:
        print("\n====================")


def print_section(title: str) -> None:
    """
    Print a highlighted section header.
    """
    print_separator(title)
    print_separator()


def _json_serializer(obj: Any) -> Any:
    """
    Convert non-serializable objects (e.g., numpy types) into JSON-friendly formats.
    """
    if np is not None:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


def save_results_to_json(results_dict: Dict[str, Any], filename: Path | str) -> Path:
    """
    Save results to a JSON file, ensuring parent directories exist.
    Unserializable objects are converted where possible.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False, default=_json_serializer)
    return path


def load_results_from_json(filename: Path | str) -> Optional[Dict[str, Any]]:
    """
    Load JSON results from file. Returns None if the file does not exist or cannot be read.
    """
    path = Path(filename)
    if not path.exists():
        print(f"[Info] File not found: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[Error] Failed to parse JSON: {exc}")
    return None


def validate_path(path: Path | str) -> bool:
    """
    Validate whether a given path exists (file or directory).
    """
    target = Path(path)
    if not target.exists():
        print(f"[Warning] Path does not exist: {target}")
        return False
    return True


def format_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable string (e.g., '2h 30m 45s').
    """
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def collect_image_files(directory: Path | str, extensions: Sequence[str] | None = None) -> List[Path]:
    """
    Collect image files recursively from a directory.

    Args:
        directory: Directory path to scan.
        extensions: Optional sequence of allowed extensions (case-insensitive).

    Returns:
        List of Path objects pointing to image files.
    """
    exts = extensions or [".jpg", ".jpeg", ".png", ".bmp"]
    root = Path(directory)
    if not root.exists():
        print(f"[Warning] Directory not found: {root}")
        return []
    files: List[Path] = []
    for file in root.rglob("*"):
        if file.suffix.lower() in (ext.lower() for ext in exts):
            files.append(file)
    if not files:
        print(f"[Info] No image files found in {root}")
    return files

