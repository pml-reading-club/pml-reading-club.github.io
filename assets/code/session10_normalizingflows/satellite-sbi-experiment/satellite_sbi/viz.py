"""Plotting helpers used by ``notebooks/final.ipynb``."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from simulator import BaseDemoConfig, observation_times


def reshape_observation(x: Tensor | np.ndarray, config: BaseDemoConfig) -> np.ndarray:
    """Reshape a flattened observation back to ``(num_observations, 6)``."""

    return np.asarray(x, dtype=float).reshape(config.num_observations, 6)


def plot_observation(x: Tensor | np.ndarray, config: BaseDemoConfig) -> tuple[Any, Any]:
    """Plot coarse residual norms over time."""

    residuals = reshape_observation(x, config)
    times_h = observation_times(config) / 3600.0
    position_error = np.linalg.norm(residuals[:, :3], axis=1)
    velocity_error = np.linalg.norm(residuals[:, 3:], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(times_h, position_error, color="black", lw=2.0)
    axes[0].set_ylabel(r"$||r-r_{ref}||$ [km]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times_h, velocity_error, color="black", lw=2.0)
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel(r"$||v-v_{ref}||$ [km/s]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Coarse observation used by SBI")
    fig.tight_layout()
    return fig, axes


def plot_joint_posterior(
    samples: Tensor,
    prior_bounds: tuple[tuple[float, float], tuple[float, float]],
    theta_true: Tensor | None = None,
) -> tuple[Any, Any]:
    """Plot the joint posterior and both marginal posteriors."""

    values = samples.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].hist(values[:, 0], bins=32, color="#0B3954", alpha=0.8, density=True)
    axes[0].axvspan(prior_bounds[0][0], prior_bounds[0][1], color="#D9D9D9", alpha=0.25, zorder=0)
    if theta_true is not None:
        axes[0].axvline(float(theta_true[0]), color="#C1121F", lw=2.0, linestyle="--")
    axes[0].set_xlabel(r"$\log_{10} K_p$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Marginal posterior for Kp")
    axes[0].grid(True, alpha=0.25)

    hexbin = axes[1].hexbin(values[:, 0], values[:, 1], gridsize=28, cmap="Blues", mincnt=1)
    fig.colorbar(hexbin, ax=axes[1], label="Sample count")
    if theta_true is not None:
        axes[1].scatter(
            float(theta_true[0]),
            float(theta_true[1]),
            color="#C1121F",
            s=70,
            marker="x",
            linewidth=2.0,
            label="True value",
        )
        axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_xlim(*prior_bounds[0])
    axes[1].set_ylim(*prior_bounds[1])
    axes[1].set_xlabel(r"$\log_{10} K_p$")
    axes[1].set_ylabel(r"$\log_{10} K_d$")
    axes[1].set_title("Joint posterior")
    axes[1].grid(True, alpha=0.2)

    axes[2].hist(values[:, 1], bins=32, color="#087E8B", alpha=0.8, density=True)
    axes[2].axvspan(prior_bounds[1][0], prior_bounds[1][1], color="#D9D9D9", alpha=0.25, zorder=0)
    if theta_true is not None:
        axes[2].axvline(float(theta_true[1]), color="#C1121F", lw=2.0, linestyle="--")
    axes[2].set_xlabel(r"$\log_{10} K_d$")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Marginal posterior for Kd")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Joint posterior over the two controller gains")
    fig.tight_layout()
    return fig, axes


def plot_round_joint_posteriors(
    round_posterior_samples: list[np.ndarray],
    prior_bounds: tuple[tuple[float, float], tuple[float, float]],
    theta_true: Tensor | None = None,
) -> tuple[Any, Any]:
    """Visualize how the joint posterior sharpens over sequential rounds."""

    fig, axes = plt.subplots(
        1,
        len(round_posterior_samples),
        figsize=(4.4 * len(round_posterior_samples), 4.0),
        squeeze=False,
    )
    axes_flat = axes[0]

    for idx, round_samples in enumerate(round_posterior_samples, start=1):
        ax = axes_flat[idx - 1]
        ax.hexbin(round_samples[:, 0], round_samples[:, 1], gridsize=24, cmap="Blues", mincnt=1)
        if theta_true is not None:
            ax.scatter(
                float(theta_true[0]),
                float(theta_true[1]),
                color="#C1121F",
                s=60,
                marker="x",
                linewidth=2.0,
            )
        ax.set_xlim(*prior_bounds[0])
        ax.set_ylim(*prior_bounds[1])
        ax.set_xlabel(r"$\log_{10} K_p$")
        ax.set_ylabel(r"$\log_{10} K_d$")
        ax.set_title(f"Round {idx}")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Sequential rounds focus mass near the observed trajectory")
    fig.tight_layout()
    return fig, axes
