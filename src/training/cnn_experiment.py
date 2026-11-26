"""Experimento CNN de 05: la maquinaria común por fases con el modelo CNN.

Fases: ``h1`` (CNN sin BatchNorm frente a la Dense de 04) y, después,
``batchnorm`` (la misma CNN cambiando solo BatchNorm).
"""
from src.models.cnn import CNNClassifier, multiply_adds
from src.training.phased_experiment import PhasedExperiment

CONFIG = 'config/cnn_baseline_v1.json'
RESULTS = 'results/metrics/05_cnn'
WEIGHTS = 'results/models/05_cnn'
SOURCES = [CONFIG, 'config/experimental_protocol_v1.json', 'src/models/cnn.py', 'src/training/cnn_experiment.py',
           'src/training/phased_experiment.py', 'src/training/dense_experiment.py']


def model_from_config(config, variant):
    a = config['architecture']
    return CNNClassifier(a['image_size'], tuple(a['conv_channels']), tuple(a['hidden_dims']), a['classes'],
                         a['kernel_size'], a['padding'], batchnorm=variant['batchnorm'])


EXPERIMENT = PhasedExperiment(CONFIG, RESULTS, WEIGHTS, SOURCES, model_from_config,
                              lambda model, config: multiply_adds(model, config['architecture']['image_size']))

load_cnn_context = EXPERIMENT.load_context
load_saved_phase = EXPERIMENT.load_saved_phase
run_phase = EXPERIMENT.run_phase
