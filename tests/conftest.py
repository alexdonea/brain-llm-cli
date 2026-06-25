"""Put the engine package (src/) on sys.path so the tests can `import agent`, `import brain`, etc.
pytest loads this before collecting any test module, so the import in test_cli.py (which sets BRAIN_HOME then
imports agent) resolves against src/."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
