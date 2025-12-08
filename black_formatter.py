import numpy as np
import matplotlib.pyplot as plt


def compute_f1_grid(n: int = 200, eps: float = 1e-3) -> np.ndarray:
    p = np.linspace(eps, 1.0, n)
    r = np.linspace(eps, 1.0, n)
    pre, re = np.meshgrid(p, r)
    f1 = 2 * pre * re / (pre + re)
    return f1


def plot_f1_heatmap(f1_score: np.ndarray) -> None:
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(f1_score, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    plt.xlabel("P")
    plt.ylabel("R")
    plt.title("F1-score heatmap")
    plt.colorbar(label="F1-score")
    plt.show()


def main() -> None:
    f1_score = compute_f1_grid()
    plot_f1_heatmap(f1_score)


if __name__ == "__main__":
    main()
