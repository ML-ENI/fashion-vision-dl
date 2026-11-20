"""Ejecutar y guardar outputs reales de 01/02, sin sobrescribirlos si fallan."""
from pathlib import Path
import argparse
import os
import time

ROOT = Path(__file__).resolve().parents[1]
for variable, folder in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib")]:
    path = ROOT / "tmp" / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(path))

import nbformat
from nbclient import NotebookClient


def execute(name):
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    started = time.monotonic()
    def completed(cell, cell_index, **kwargs):
        print(f"{name}: celda {cell_index + 1}/{len(notebook.cells)} ({time.monotonic()-started:.0f}s)", flush=True)
    client = NotebookClient(notebook, timeout=-1, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}},
                            on_cell_executed=completed)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, ROOT / "tmp" / ("failed_" + name))
        raise
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if cell.execution_count is None or any(o.output_type == "error" for o in cell.outputs):
                raise ValueError(f"Notebook incompleto: {name}")
    temporary = path.with_suffix(".pending.ipynb")
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    print(f"COMPLETADO {name} ({time.monotonic()-started:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", choices=["01_data_understanding.ipynb", "02_data_preparation.ipynb", "03_eda_images.ipynb"],
                        default=["01_data_understanding.ipynb", "02_data_preparation.ipynb"])
    args = parser.parse_args()
    for name in args.notebooks:
        execute(name)
