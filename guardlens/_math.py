import numpy as np

_NORM_TOL = 1e-3


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity via dot product -- callers must pass L2-normalized
    vectors (asserted here) so this reduces to a plain dot product."""
    assert abs(np.linalg.norm(u) - 1.0) < _NORM_TOL, "cosine() requires a normalized vector (u)"
    assert abs(np.linalg.norm(v) - 1.0) < _NORM_TOL, "cosine() requires a normalized vector (v)"
    return float(np.dot(u, v))
