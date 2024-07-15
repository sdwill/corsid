from pathlib import Path
import matplotlib
REPO_ROOT = Path(__file__).parents[1]
matplotlib.rc_file(REPO_ROOT / 'matplotlibrc')
