from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

# Bu dosya: deney çıktılarını karşılaştırmak ve istatistikleri çizmek için yardımcı plot fonksiyonları.


def _show_image(ax, img: np.ndarray, title: str) -> None:
    # Görseli gri ya da renkli olarak eksenleri gizleyip gösteriyoruz.
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(np.clip(img / 255.0, 0, 1))
    ax.set_title(title)
    ax.axis("off")


def plot_comparison(
    original: np.ndarray,
    linear: np.ndarray,
    pm1: np.ndarray,
    pm2: np.ndarray,
    charbonnier: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Show original + linear + three nonlinear results in a grid."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    images = [
        ("Original", original),
        ("Linear", linear),
        ("PM1", pm1),
        ("PM2", pm2),
        ("Charbonnier", charbonnier),
    ]
    for ax, (title, img) in zip(axes, images):
        _show_image(ax, img, title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_statistics(
    history: Dict[str, List[float]], title: str, save_path: str | Path | None = None
) -> None:
    """Plot mean, variance, and total gradient magnitude over iterations."""
    iterations = np.arange(1, len(history["mean"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    axes[0].plot(iterations, history["mean"], label="Mean intensity")
    axes[0].set_ylabel("Mean")
    axes[0].legend()
    axes[1].plot(iterations, history["variance"], label="Variance", color="orange")
    axes[1].set_ylabel("Variance")
    axes[1].legend()
    axes[2].plot(
        iterations, history["gradient_magnitude"], label="Sum |grad|", color="green"
    )
    axes[2].set_ylabel("Sum grad mag")
    axes[2].set_xlabel("Iteration")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_color_statistics(
    color_history: Dict[str, List], title: str, save_path: str | Path | None = None
) -> None:
    """Plot per-channel mean/variance and total gradient for RGB diffusion."""
    iterations = np.arange(1, len(color_history["mean"]) + 1)
    means = np.array(color_history["mean"])
    variances = np.array(color_history["variance"])
    total_grad = np.array(color_history["total_gradient"])
    colors = ["red", "green", "blue"]
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    for c in range(3):
        axes[0].plot(iterations, means[:, c], label=f"Mean ch{c}", color=colors[c])
    axes[0].set_ylabel("Mean")
    axes[0].legend()
    for c in range(3):
        axes[1].plot(
            iterations, variances[:, c], label=f"Var ch{c}", color=colors[c], linestyle="--"
        )
        
    axes[1].set_ylabel("Variance")
    axes[1].legend()
    axes[2].plot(iterations, total_grad, label="Sum |grad| all channels", color="k")
    axes[2].set_ylabel("Sum grad mag")
    axes[2].set_xlabel("Iteration")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_lambda(
    image: np.ndarray,
    diff_mode: str,
    lambda_values: Iterable[float],
    sigma: float = 1.0,
    dt: float = 0.25,
    num_iterations: int = 30,
    save_path: str | Path | None = None,
) -> None:
    """
    Vary lambda and show the resulting images for one diffusivity mode.
    Lazy import to avoid circular dependency.
    """
    from nonlinear_diffusion import NonlinearDiffusion

    lambda_values = list(lambda_values)
    results = []
    for lam in lambda_values:
        nd = NonlinearDiffusion(
            lambda_param=lam,
            sigma=sigma,
            dt=dt,
            num_iterations=num_iterations,
            diffusivity_mode=diff_mode,
        )
        out, _ = nd.run(image)
        results.append(out)
    cols = len(results) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    _show_image(axes[0], image, "Original")
    for ax, lam, img in zip(axes[1:], lambda_values, results):
        _show_image(ax, img, f"{diff_mode.upper()} λ={lam}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_sigma(
    image: np.ndarray,
    diff_mode: str,
    sigma_values: Iterable[float],
    lambda_param: float = 10.0,
    dt: float = 0.25,
    num_iterations: int = 30,
    save_path: str | Path | None = None,
) -> None:
    """Vary sigma (pre-smoothing) and show results for one diffusivity mode."""
    from nonlinear_diffusion import NonlinearDiffusion

    sigma_values = list(sigma_values)
    results = []
    for sigma in sigma_values:
        nd = NonlinearDiffusion(
            lambda_param=lambda_param,
            sigma=sigma,
            dt=dt,
            num_iterations=num_iterations,
            diffusivity_mode=diff_mode,
        )
        out, _ = nd.run(image)
        results.append(out)
    cols = len(results) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    _show_image(axes[0], image, "Original")
    for ax, sigma, img in zip(axes[1:], sigma_values, results):
        _show_image(ax, img, f"{diff_mode.upper()} σ={sigma}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
