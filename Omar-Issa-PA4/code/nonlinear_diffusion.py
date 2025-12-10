from __future__ import annotations

# Bu dosya projenin kalbi: gri ve renkli Perona–Malik difüzyon sınıfları, linear baseline ve CLI demolar.

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

# Make local imports work whether running as a module or script
# Yerel importlar script veya modül olarak çalışırken bozulmasın diye sys.path eklemesi.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from analysis import (
    compare_lambda,
    compare_sigma,
    plot_color_statistics,
    plot_comparison,
    plot_statistics,
)
from diffusivity_functions import get_diffusivity
import utils


class NonlinearDiffusion:
    """Perona–Malik style nonlinear diffusion for grayscale images."""

    def __init__(
        self,
        lambda_param: float = 10.0,
        sigma: float = 1.0,
        dt: float = 0.25,
        num_iterations: int = 50,
        diffusivity_mode: str = "pm1",
    ) -> None:
        self.lambda_param = float(lambda_param)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.num_iterations = int(num_iterations)
        self.diffusivity_mode = diffusivity_mode

    @staticmethod
    def _compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Central differences with reflective padding."""
        # Basit merkez farkı: sınırda yansıtma (reflect) kullanıp x ve y türevlerini alıyoruz.
        padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
        gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
        gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
        return gx, gy

    @staticmethod
    def _gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        return np.sqrt(gx * gx + gy * gy + 1e-12)

    @staticmethod
    def _divergence(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Divergence of a vector field (fx, fy) using central differences."""
        # ∇·F hesaplıyoruz: akıların türevi. Bu, difüzyon denklemindeki Laplasyan benzeri kısım.
        fx_pad = np.pad(fx, ((1, 1), (1, 1)), mode="reflect")
        fy_pad = np.pad(fy, ((1, 1), (1, 1)), mode="reflect")
        dfx_dx = (fx_pad[1:-1, 2:] - fx_pad[1:-1, :-2]) * 0.5
        dfy_dy = (fy_pad[2:, 1:-1] - fy_pad[:-2, 1:-1]) * 0.5
        return dfx_dx + dfy_dy

    def _diffusion_step(self, image: np.ndarray) -> np.ndarray:
        """One explicit Euler diffusion step."""
        # 1) İsteğe bağlı Gaussian ile yumuşat, 2) |∇u_sigma| ile g hesapla, 3) akı = g * ∇u, 4) diverjansla güncelle.
        if self.sigma > 0:
            u_sigma = gaussian_filter(image, sigma=self.sigma, mode="reflect")
        else:
            u_sigma = image
        gx_sigma, gy_sigma = self._compute_gradients(u_sigma)
        grad_mag = self._gradient_magnitude(gx_sigma, gy_sigma)
        g = get_diffusivity(grad_mag, self.lambda_param, self.diffusivity_mode)

        gx, gy = self._compute_gradients(image)
        flux_x = g * gx
        flux_y = g * gy
        div = self._divergence(flux_x, flux_y)
        u_new = image + self.dt * div
        return np.clip(u_new, 0.0, 255.0)

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
        """Run diffusion for the configured number of iterations."""
        u = image.astype(np.float64).copy()
        history = {"mean": [], "variance": [], "gradient_magnitude": []}
        for i in range(self.num_iterations):
            u = self._diffusion_step(u)
            gx, gy = self._compute_gradients(u)
            grad_mag = self._gradient_magnitude(gx, gy)
            history["mean"].append(float(np.mean(u)))
            history["variance"].append(float(np.var(u)))
            history["gradient_magnitude"].append(float(np.sum(grad_mag)))
            if (i + 1) % max(1, self.num_iterations // 5) == 0:
                print(f"Iteration {i + 1}/{self.num_iterations}")
        return u, history


class ColorNonlinearDiffusion(NonlinearDiffusion):
    """Nonlinear diffusion for RGB images using shared diffusivity."""

    def _diffusion_step_color(self, image: np.ndarray) -> np.ndarray:
        if self.sigma > 0:
            u_sigma = np.stack(
                [
                    gaussian_filter(image[..., c], sigma=self.sigma, mode="reflect")
                    for c in range(3)
                ],
                axis=-1,
            )
        else:
            u_sigma = image
        # R,G,B kanallarının gradyanlarının toplam büyüklüğünü alıp tek bir g alanı üretiyoruz.
        total_grad_mag = np.zeros(image.shape[:2], dtype=np.float64)
        for c in range(3):
            gx_c, gy_c = self._compute_gradients(u_sigma[..., c])
            total_grad_mag += self._gradient_magnitude(gx_c, gy_c)

        g = get_diffusivity(total_grad_mag, self.lambda_param, self.diffusivity_mode)

        updated = np.empty_like(image)
        for c in range(3):
            gx, gy = self._compute_gradients(image[..., c])
            flux_x = g * gx
            flux_y = g * gy
            div = self._divergence(flux_x, flux_y)
            updated[..., c] = np.clip(image[..., c] + self.dt * div, 0.0, 255.0)
        return updated

    def run(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, List]]:
        u = image.astype(np.float64).copy()
        history = {"mean": [], "variance": [], "total_gradient": []}
        for i in range(self.num_iterations):
            u = self._diffusion_step_color(u)
            means = [float(np.mean(u[..., c])) for c in range(3)]
            variances = [float(np.var(u[..., c])) for c in range(3)]
            gx, gy = self._compute_gradients(np.sum(u, axis=2) / 3.0)
            total_grad = float(np.sum(self._gradient_magnitude(gx, gy)))
            history["mean"].append(means)
            history["variance"].append(variances)
            history["total_gradient"].append(total_grad)
            if (i + 1) % max(1, self.num_iterations // 5) == 0:
                print(f"Iteration {i + 1}/{self.num_iterations}")
        return u, history


def linear_diffusion(image: np.ndarray, dt: float = 0.25, num_iterations: int = 50) -> np.ndarray:
    """
    Solve du/dt = Laplacian(u) using explicit finite differences.
    Works for grayscale or RGB (applied per channel).
    """

    def laplacian(channel: np.ndarray) -> np.ndarray:
        padded = np.pad(channel, ((1, 1), (1, 1)), mode="reflect")
        center = padded[1:-1, 1:-1]
        return (
            padded[1:-1, 2:] + padded[1:-1, :-2] + padded[2:, 1:-1] + padded[:-2, 1:-1]
            - 4 * center
        )

    u = image.astype(np.float64).copy()
    for i in range(num_iterations):
        if u.ndim == 2:
            lap = laplacian(u)
            u = np.clip(u + dt * lap, 0.0, 255.0)
        else:
            for c in range(u.shape[2]):
                lap = laplacian(u[..., c])
                u[..., c] = np.clip(u[..., c] + dt * lap, 0.0, 255.0)
        if (i + 1) % max(1, num_iterations // 5) == 0:
            print(f"Linear diffusion iteration {i + 1}/{num_iterations}")
    return u


def _load_sample_image(grayscale: bool = True) -> np.ndarray:
    """Try loading a test image; fallback to synthetic if missing."""
    base_dir = CURRENT_DIR.parent
    candidates = [
        base_dir / "test_image.png",
        base_dir / "data" / "test_image.png",
    ]
    for path in candidates:
        if path.exists():
            return (
                utils.load_grayscale_image(path)
                if grayscale
                else utils.load_color_image(path)
            )
    print("No test image found; using synthetic image.")
    return utils.create_synthetic_image(color=not grayscale)


def grayscale_demo() -> None:
    base_dir = CURRENT_DIR.parent
    results_dir = base_dir / "html" / "results"
    plots_dir = base_dir / "html" / "plots"
    utils.ensure_dir(results_dir)
    utils.ensure_dir(plots_dir)

    image = _load_sample_image(grayscale=True)
    print("Running linear diffusion (grayscale)...")
    linear_img = linear_diffusion(image, dt=0.25, num_iterations=30)

    outputs: Dict[str, np.ndarray] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}
    for mode in ["pm1", "pm2", "charbonnier"]:
        print(f"Running nonlinear diffusion ({mode})...")
        nd = NonlinearDiffusion(
            lambda_param=15.0,
            sigma=1.0,
            dt=0.2,
            num_iterations=30,
            diffusivity_mode=mode,
        )
        out, hist = nd.run(image)
        outputs[mode] = out
        histories[mode] = hist

    comparison_path = results_dir / "comparison_grayscale.png"
    plot_comparison(
        original=image,
        linear=linear_img,
        pm1=outputs["pm1"],
        pm2=outputs["pm2"],
        charbonnier=outputs["charbonnier"],
        save_path=comparison_path,
    )
    stats_path = plots_dir / "statistics_pm1.png"
    plot_statistics(histories["pm1"], "Statistics (PM1)", save_path=stats_path)

    utils.save_image(results_dir / "linear.png", linear_img)
    for mode, img in outputs.items():
        utils.save_image(results_dir / f"{mode}.png", img)

    print(f"PSNR (linear vs original): {utils.compute_psnr(image, linear_img):.2f} dB")
    print(f"SSIM (linear vs original): {utils.compute_ssim(image, linear_img):.3f}")
    print(f"Results saved to {comparison_path} and {stats_path}")


def color_demo() -> None:
    base_dir = CURRENT_DIR.parent
    results_dir = base_dir / "html" / "results"
    plots_dir = base_dir / "html" / "plots"
    utils.ensure_dir(results_dir)
    utils.ensure_dir(plots_dir)

    image = _load_sample_image(grayscale=False)
    print("Running color nonlinear diffusion (PM1)...")
    nd = ColorNonlinearDiffusion(
        lambda_param=20.0, sigma=1.0, dt=0.2, num_iterations=25, diffusivity_mode="pm1"
    )
    filtered, history = nd.run(image)

    utils.save_image(results_dir / "color_original.png", image)
    utils.save_image(results_dir / "color_filtered.png", filtered)
    plot_color_statistics(
        history,
        "Color diffusion statistics (PM1)",
        save_path=plots_dir / "color_diffusion_statistics.png",
    )

    print(f"PSNR (color filtered vs original): {utils.compute_psnr(image, filtered):.2f} dB")
    print(f"SSIM (color filtered vs original): {utils.compute_ssim(image, filtered):.3f}")
    print(f"Saved color result to {results_dir / 'color_filtered.png'}")


def parameter_study() -> None:
    base_dir = CURRENT_DIR.parent
    plots_dir = base_dir / "html" / "plots"
    utils.ensure_dir(plots_dir)
    image = _load_sample_image(grayscale=True)
    lambda_values = [5.0, 10.0, 20.0, 30.0]
    sigma_values = [0.5, 1.0, 2.0]

    print("Running lambda comparison (PM1)...")
    compare_lambda(
        image,
        diff_mode="pm1",
        lambda_values=lambda_values,
        sigma=1.0,
        num_iterations=25,
        save_path=plots_dir / "lambda_comparison_pm1.png",
    )

    print("Running sigma comparison (PM1)...")
    compare_sigma(
        image,
        diff_mode="pm1",
        sigma_values=sigma_values,
        lambda_param=15.0,
        num_iterations=25,
        save_path=plots_dir / "sigma_comparison_pm1.png",
    )
    print("Parameter study plots saved.")


def main() -> None:
    print("Nonlinear Diffusion Filtering (Perona–Malik)")
    print("1) Grayscale demo (linear vs PM1/PM2/Charbonnier)")
    print("2) Color demo (PM1)")
    print("3) Parameter study (lambda and sigma sweeps)")
    # Menü: klavyeden 1-3 girip ilgili deneyi çalıştırıyoruz.
    choice = input("Select an option (1/2/3): ").strip()

    if choice == "1":
        grayscale_demo()
    elif choice == "2":
        color_demo()
    elif choice == "3":
        parameter_study()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
