"""Import hygiene: the pure core must not import heavy estimation backends."""

import subprocess
import sys


def test_core_import_does_not_pull_heavy_backends():
    code = (
        "import sys\n"
        "import badgers\n"
        "import badgers.core.dgp\n"
        "heavy = [m for m in sys.modules\n"
        "         if m.split('.')[0] in ('pymc', 'pytensor', 'dowhy')]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
