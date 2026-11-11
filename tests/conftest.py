import numpy as np
import pytest


def random_strip(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Per-pixel random colour strip: the hardest case for exact-equality tests."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def strip():
    return random_strip(64, 154, seed=1)
