"""Experimento de 06: la maquinaria común por fases con modelos recurrentes.

Fases: ``cells`` (RNN, LSTM y GRU), después ``attention`` (la misma LSTM con
atención) y ``rows`` (la misma LSTM leyendo filas).
"""
from src.models.sequence import SequenceClassifier, multiply_adds
from src.training.phased_experiment import PhasedExperiment

CONFIG = 'config/sequence_v1.json'
RESULTS = 'results/metrics/06_sequence'
WEIGHTS = 'results/models/06_sequence'
SOURCES = [CONFIG, 'config/experimental_protocol_v1.json', 'src/models/sequence.py',
           'src/training/sequence_experiment.py', 'src/training/phased_experiment.py',
           'src/training/dense_experiment.py']


def model_from_config(config, variant):
    a = config['architecture']
    return SequenceClassifier(variant['cell'], a['features_per_step'], a['hidden_size'], a['classes'],
                              variant['orientation'], variant['readout'])


EXPERIMENT = PhasedExperiment(CONFIG, RESULTS, WEIGHTS, SOURCES, model_from_config,
                              lambda model, config: multiply_adds(model, config['architecture']['steps']))

load_sequence_context = EXPERIMENT.load_context
load_saved_phase = EXPERIMENT.load_saved_phase
run_phase = EXPERIMENT.run_phase
