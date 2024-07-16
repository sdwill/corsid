from pathlib import Path
import matplotlib
from .TrainingData import TrainingData
from .adam import AdamOptimizer

REPO_ROOT = Path(__file__).parents[1]
matplotlib.rc_file(REPO_ROOT / 'matplotlibrc')
