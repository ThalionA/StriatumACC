"""Put the src-layout package on sys.path so the test suite imports it
without needing an editable install."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
