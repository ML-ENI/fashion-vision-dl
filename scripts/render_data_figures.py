"""Reejecutar solo las celdas de figuras sobre las publicaciones existentes.

Conserva las salidas de los análisis ya ejecutados y evita volver a procesar
imágenes al modificar exclusivamente el estilo visual.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
for variable, folder in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "matplotlib")]:
    os.environ.setdefault(variable, str(ROOT / "tmp" / folder))

import nbformat
from nbclient import NotebookClient

BOOTSTRAP = """
from audit_images import read_audit
from prepare_local import select_candidates
from local_dataset import LocalImageDataset
audit_run, audit, summary = read_audit()
valid = audit.loc[audit.image_valid].copy()
selection, candidates = select_candidates(audit, config)
output = ROOT / 'data/processed'
run = output / json.loads((output / 'CURRENT.json').read_text())['run_dir']
manifest = pd.read_csv(run / 'image_manifest.csv.gz', dtype={'article_id':'string','product_code':'string'}, keep_default_na=False)
usable = manifest.loc[manifest.split.ne('excluded')].copy()
distribution = pd.crosstab(usable.label, usable.split).reindex(columns=['train', 'validation', 'test'])
distribution['Total'] = distribution.sum(axis=1)
train_plain = LocalImageDataset('train', augment=False, normalize=False, seed=SEED)
train_aug = LocalImageDataset('train', augment=True, normalize=False, seed=SEED)
"""


if __name__ == "__main__":
    paths = [ROOT / f"data/{kind}/CURRENT.json" for kind in ["audit", "processed"]]
    before = {str(p): p.read_bytes() for p in paths}
    for name in ["01_data_understanding.ipynb", "02_data_preparation.ipynb"]:
        path = ROOT / "notebooks" / name
        original = nbformat.read(path, as_version=4)
        if any(c.execution_count is None for c in original.cells if c.cell_type == "code"):
            raise ValueError("Ejecuta primero el notebook completo")
        setup = next(c.source for c in original.cells if c.cell_type == "code" and "from notebook_utils import" in c.source)
        indices = [i for i, c in enumerate(original.cells) if c.cell_type == "code" and "show_figure(" in c.source]
        rendering = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup + "\n" + BOOTSTRAP)] +
                                           [nbformat.v4.new_code_cell(original.cells[i].source) for i in indices])
        NotebookClient(rendering, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        for i, rendered in zip(indices, rendering.cells[1:]):
            original.cells[i].outputs = rendered.outputs
            original.cells[i].execution_count = rendered.execution_count
        original.metadata["visual_style"]["rerender"] = {
            "method": "figure cells executed against existing audited/prepared publications",
            "style_sha256": hashlib.sha256((ROOT / "src/visualization/style.py").read_bytes()).hexdigest(),
        }
        nbformat.validate(original)
        temporary = path.with_suffix(".pending.ipynb")
        nbformat.write(original, temporary)
        temporary.replace(path)
        print(f"Figuras y outputs actualizados: {name}", flush=True)
    assert all(p.read_bytes() == before[str(p)] for p in paths), "Cambió una publicación de datos"
    print("Punteros de auditoría y dataset intactos", flush=True)
