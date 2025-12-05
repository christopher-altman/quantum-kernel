"""Create a clean ZIP archive of the repository in dist/."""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

EXCLUDE_DIRS = {".git", ".idea", ".vscode", "__pycache__", ".pytest_cache"}
EXCLUDE_FILES = {".DS_Store"}


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    return False


def main():
    DIST.mkdir(exist_ok=True)
    out_path = DIST / "quantum_kernel_expressivity_bundle.zip"
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(ROOT):
            root_path = Path(root)
            if root_path == DIST:
                continue
            for f in files:
                fp = root_path / f
                rel = fp.relative_to(ROOT)
                if should_exclude(rel):
                    continue
                if rel == out_path.relative_to(ROOT):
                    continue
                z.write(fp, arcname=str(rel))
    print(f"[OK] Wrote repository bundle to {out_path}")


if __name__ == "__main__":
    main()
