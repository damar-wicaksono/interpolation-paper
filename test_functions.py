import numpy as np


def runge(xx: np.ndarray, parameter: float = [1.0, 1.0]) -> np.ndarray:
    """Compute the M-dimensional Runge function."""
    return 1 / (parameter[0]**2 +  parameter[1]**2 * np.sum(xx**2, axis=1))


def f2(xx: np.ndarray, parameter: float = 5 / 4) -> np.ndarray:
    """Compute the M-dimensional F2 function."""
    return 1 / ((xx[:, 0] - parameter)**2 + xx[:, 1]**2)


def f3(xx: np.ndarray, parameter: float = 5 / 4) -> np.ndarray:
    """Compute the M-dimensional F3 function."""
    return 1 / np.sum((xx - parameter)**2, axis=1)


def multidim_sin(xx: np.ndarray, parameter: int = 1) -> np.ndarray:
    """Compute the M-dimensional sine function."""
    params = parameter * np.ones(xx.shape[1])

    return np.sin(np.pi * np.sum(params * xx, axis=1))


def multidim_cos(xx: np.ndarray, parameter: int = 1) -> np.ndarray:
    """Compute the M-dimensional cosine function."""
    params = parameter * np.ones(xx.shape[1])

    return np.cos(np.pi * np.sum(params * xx, axis=1))


def multidim_cos_sin(xx: np.ndarray, parameter: list = [1, 1]) -> np.ndarray:
    """Compute the M-dimensional sine-cosine function."""
    params_1 = parameter[0] * np.ones(xx.shape[1])
    params_2 = parameter[1] * np.ones(xx.shape[1])

    xx_sin = params_1 * xx
    xx_cos = params_2 * xx

    return np.cos(np.pi * np.sum(xx_cos, axis=1)) + np.sin(np.pi * np.sum(xx_sin, axis=1))


def f5(xx: np.ndarray) -> np.ndarray:
    """Compute the M-dimensional F5 function."""
    m = xx.shape[1]
    coeffs = np.arange(1, m + 1)
    coeffs = 5 / coeffs**3

    xx = coeffs * xx

    yy = 1 / (1 + (np.sum(xx, axis=1))**2)

    return yy
