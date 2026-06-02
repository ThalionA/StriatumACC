import os
import sys

# Ensure 'src' is on the path for imports during tests (matches sibling
# subprojects in this repo, whose package directories live under src/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
