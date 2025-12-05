"""Export notebooks to HTML and PDF into docs/."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
DOCS = ROOT / "docs"


def export_notebook(nb_path: Path, to: str = "html"):
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        to,
        str(nb_path),
        "--output-dir",
        str(DOCS),
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    DOCS.mkdir(exist_ok=True)
    nbs = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not nbs:
        print("[WARN] No notebooks found.")
        return
    print("=== Exporting notebooks to HTML ===")
    for nb in nbs:
        export_notebook(nb, to="html")
    print("=== Attempting PDF export (if supported) ===")
    for nb in nbs:
        export_notebook(nb, to="pdf")


if __name__ == "__main__":
    main()
