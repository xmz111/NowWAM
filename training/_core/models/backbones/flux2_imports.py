from __future__ import annotations
import sys
from pathlib import Path


def ensure_flux2_importable(flux2_src_path: str | None = None) -> None:
    candidates: list[Path] = []
    if flux2_src_path:
        root = Path(flux2_src_path).expanduser().resolve()
        candidates.extend([root / "src", root])
    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend([repo_root.parent / "flux2" / "src", repo_root.parent / "flux2"])
    for path in candidates:
        if (path / "flux2").is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            return
