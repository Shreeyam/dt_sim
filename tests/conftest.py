"""Put dt_sim/src on sys.path for tests."""
import sys
from pathlib import Path

DT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DT_SIM / "src"))
