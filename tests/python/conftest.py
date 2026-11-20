import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:                                    # installed package (pip install .)
    import dlrand  # noqa: F401
except Exception:                       # source checkout: use python/ and a CMake build if present
    sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
    import dlrand  # noqa: F401


@pytest.fixture
def rng():
    g = dlrand.RandomGenerator(20240229)
    yield g
    g.close()
