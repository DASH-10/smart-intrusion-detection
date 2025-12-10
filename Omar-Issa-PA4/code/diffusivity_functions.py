import numpy as np

# Bu dosyada Perona–Malik ve Charbonnier difüzyon fonksiyonları bulunuyor.
# Öğrenci gözüyle: |∇u| büyükse g küçülüyor, yani kenarları korumak için akımı yavaşlatıyoruz.


def diffusivity_pm1(grad_mag: np.ndarray, lambda_param: float) -> np.ndarray:
    """Perona–Malik Type 1: g(|x|) = exp(-|x|^2 / lambda^2)."""
    grad_mag = np.asarray(grad_mag, dtype=np.float64)
    lambda_param = max(float(lambda_param), 1e-12)
    return np.exp(-np.square(grad_mag) / (lambda_param ** 2))


def diffusivity_pm2(grad_mag: np.ndarray, lambda_param: float) -> np.ndarray:
    """Perona–Malik Type 2: g(|x|) = 1 / (1 + |x|^2 / lambda^2)."""
    grad_mag = np.asarray(grad_mag, dtype=np.float64)
    lambda_param = max(float(lambda_param), 1e-12)
    return 1.0 / (1.0 + np.square(grad_mag) / (lambda_param ** 2))


def diffusivity_charbonnier(grad_mag: np.ndarray, lambda_param: float) -> np.ndarray:
    """Charbonnier: g(|x|) = 1 / sqrt(1 + |x|^2 / lambda^2)."""
    grad_mag = np.asarray(grad_mag, dtype=np.float64)
    lambda_param = max(float(lambda_param), 1e-12)
    return 1.0 / np.sqrt(1.0 + np.square(grad_mag) / (lambda_param ** 2))


def get_diffusivity(
    grad_mag: np.ndarray, lambda_param: float, mode: str = "pm1"
) -> np.ndarray:
    """
    Dispatch function that calls one of: pm1, pm2, charbonnier.
    mode in {"pm1", "pm2", "charbonnier"}.
    """
    mode = mode.lower()
    if mode == "pm1":
        return diffusivity_pm1(grad_mag, lambda_param)
    if mode == "pm2":
        return diffusivity_pm2(grad_mag, lambda_param)
    if mode == "charbonnier":
        return diffusivity_charbonnier(grad_mag, lambda_param)
    raise ValueError(f"Unknown diffusivity mode: {mode}")
