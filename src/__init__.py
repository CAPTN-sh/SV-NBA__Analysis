import sys
from pathlib import Path
root_dir = Path.cwd().resolve().parent.parent
if root_dir.exists():
    sys.path.append(root_dir)
else:
    raise FileNotFoundError('Root directory not found')