import numpy as np

def polynomial_model(x: np.array, ms: np.array) -> np.array:
    """Nth order polynomial fit, order defined by length of ms"""
    y = 0
    for i, m in enumerate(ms):
        y += ms[i]*(x**i)
    return y


def singaussian_model(x:np.array, ms:np.array) -> np.array:
    """sing gaussian noise
    args
    ------
    x: np.array
        array of times to evaluate
    ms: np.array
        model parameters A, t0, ph, f, width = ms
    """
    A, t0, ph, f, width = ms
    y = A*np.sin(2*np.pi*f*x + ph) * np.exp(-(x - t0)**2/width**2)
    return y