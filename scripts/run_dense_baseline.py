"""Ejecutar el diseño registrado de 04 sin necesidad de abrir Jupyter."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.dense_experiment import load_context, prepare_arrays, normalized_tensors, run_or_load

if __name__ == '__main__':
    context = load_context(ROOT)
    arrays, normalization = prepare_arrays(context)
    tensors = normalized_tensors(arrays, normalization)
    summary, directory = run_or_load(context, tensors, normalization)
    print('Experimento completo:', directory)
