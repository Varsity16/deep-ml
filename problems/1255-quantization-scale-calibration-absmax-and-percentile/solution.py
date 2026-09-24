import numpy as np

def absmax_scale(x: np.ndarray, bits: int = 8) -> float:
    """Symmetric absmax scale: max|x| / qmax, qmax=2^(bits-1)-1."""
    qmax = 2**(bits - 1) - 1
    max_abs = np.max(np.abs(x))

    if max_abs == 0:
        return 1.0

    return float(max_abs / qmax)


def percentile_scale(x: np.ndarray, bits: int = 8, p: float = 99.9) -> float:
    """Symmetric percentile scale on |x|."""
    qmax = 2**(bits - 1) - 1
    percentile = np.percentile(np.abs(x), p)

    if percentile == 0:
        return 1.0

    return float(percentile / qmax)