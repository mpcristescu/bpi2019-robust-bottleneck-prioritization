from pathlib import Path
import platform, sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.xes_utils import find_dataset
print("Python:", sys.version)
print("Platform:", platform.platform())
p = find_dataset(REPO)
print("Dataset:", p)
print("Dataset MB:", round(p.stat().st_size/1024**2, 1))
print("Environment check OK")
