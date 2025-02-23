from pathlib import Path
import sys

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
