from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Bu dosya: I/O, metrik ve sentetik görüntü üretimi için yardımcı fonksiyonlar.


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Load an image as grayscale float64 in [0, 255]."""
    # Basit okuma: cv2 BGR döner ama IMREAD_GRAYSCALE tek kanal verdiği için direkt alıyoruz.
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img.astype(np.float64)


def load_color_image(path: str | Path) -> np.ndarray:
    """Load an image as RGB float64 in [0, 255]."""
    # OpenCV BGR okuyor, bu yüzden RGB'ye çeviriyoruz ki matplotlib ile uyumlu olsun.
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float64)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save an image (grayscale or RGB) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if out.ndim == 3:
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), out_bgr)
    else:
        cv2.imwrite(str(path), out)


def compute_psnr(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between two images."""
    # PSNR: gürültü ne kadar az -> değer o kadar yüksek. MSE=0 ise sonsuz.
    ref = ref.astype(np.float64)
    target = target.astype(np.float64)
    if ref.shape != target.shape:
        raise ValueError("Images must have the same shape for PSNR.")
    mse = np.mean(np.square(ref - target))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM for one channel (non-windowed)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float(num / den)


def compute_ssim(ref: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM; averages channels for color images."""
    ref = ref.astype(np.float64)
    target = target.astype(np.float64)
    if ref.shape != target.shape:
        raise ValueError("Images must have the same shape for SSIM.")
    if ref.ndim == 2:
        return _ssim_single_channel(ref, target)
    scores = []
    for c in range(ref.shape[2]):
        scores.append(_ssim_single_channel(ref[..., c], target[..., c]))
    return float(np.mean(scores))


def create_synthetic_image(
    width: int = 256, height: int = 256, color: bool = False, noise_std: float = 15.0
) -> np.ndarray:
    """Create a simple synthetic test image (shapes + gradients + noise)."""
    # Ödevde hazır görüntü yoksa deneme yapmak için basit şekiller + gürültü üretir.
    base = np.full((height, width), 180, dtype=np.float64)
    cv2.rectangle(base, (20, 20), (width // 2, height // 2), 60, thickness=-1)
    cv2.circle(base, (3 * width // 4, height // 3), min(width, height) // 8, 220, -1)
    cv2.line(base, (10, height - 30), (width - 10, height - 50), 120, 3)
    gradient = np.tile(np.linspace(0, 50, width), (height, 1))
    base = np.clip(base + gradient, 0, 255)
    noise = np.random.normal(0, noise_std, size=base.shape)
    gray = np.clip(base + noise, 0, 255)
    if not color:
        return gray
    color_img = np.stack(
        [
            np.clip(gray + 20, 0, 255),
            np.clip(gray, 0, 255),
            np.clip(255 - gray, 0, 255),
        ],
        axis=-1,
    )
    return color_img.astype(np.float64)


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
