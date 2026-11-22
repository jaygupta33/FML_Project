"""Execute a notebook locally in place (cwd = repo root, so outputs land in results/ and figures/)."""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=None, kernel_name="fml-venv", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, path)
    print("executed", path)
