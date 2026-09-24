from pathlib import Path
import sys


IPC_ROOT = Path(__file__).resolve().parents[1]
if str(IPC_ROOT) not in sys.path:
    sys.path.insert(0, str(IPC_ROOT))
