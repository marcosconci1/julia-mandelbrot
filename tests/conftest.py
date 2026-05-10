import os
from pathlib import Path


mpl_config_dir = Path("/tmp/juliams-mplconfig")
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
