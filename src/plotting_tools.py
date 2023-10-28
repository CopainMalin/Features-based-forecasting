from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy.random import choice
import mplcyberpunk
import warnings

plt.style.use("cyberpunk")
warnings.filterwarnings("ignore")


def plot_rolling_features(
    rolling_features: DataFrame, features_to_plot: list = None, save_path: str = None
) -> None:
    features_to_plot = (
        choice(rolling_features.columns, replace=False, size=10)
        if features_to_plot is None
        else features_to_plot
    )

    num_plots = len(features_to_plot)
    num_cols = 2
    num_rows = (num_plots + 1) // 2

    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(1.5 * len(features_to_plot), len(features_to_plot))
    )

    if num_rows == 1:
        axs = [axs]

    for i, feature_name in enumerate(features_to_plot):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row][col]

        ax.plot(rolling_features.loc[:, feature_name], color=f"C{i}")
        mplcyberpunk.add_glow_effects(ax=ax, gradient_fill=True)

        ax.set_title(f"Rolling feature : {feature_name}", fontweight="bold")

    fig.suptitle(
        "Rolling features variations",
        fontweight="bold",
        fontsize=len(features_to_plot) + 5,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches=False)
    else:
        plt.show()


def plot_sequential_validation(
    perfs: dict,
    save_path: str = None,
    metric_name: str = "Error metric",
):
    plt.figure(figsize=(15, 5))
    plt.title(
        f"Sequential validation performance of the estimator",
        fontweight="bold",
        fontsize=13,
    )
    plt.plot(perfs.keys(), perfs.values(), marker="o", color="C3")
    plt.ylabel(f"{metric_name}", color="white", fontweight="bold")
    plt.xlabel(
        "Number of history points used to fit the model",
        color="white",
        fontweight="bold",
    )
    mplcyberpunk.add_glow_effects(gradient_fill=True)
    if save_path:
        plt.savefig(save_path, bbox_inches=False)
    else:
        plt.show()
