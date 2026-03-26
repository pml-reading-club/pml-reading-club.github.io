"""Inference helpers used by ``notebooks/final.ipynb``."""

from __future__ import annotations

import contextlib
import io
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from simulator import BaseDemoConfig, simulator


PriorBounds = tuple[tuple[float, float], tuple[float, float]]


def make_box_prior(bounds: PriorBounds) -> Any:
    """Construct a numerically safe two-dimensional box prior."""

    from sbi.utils import BoxUniform

    bounds_np = np.asarray(bounds, dtype=float)
    low = torch.tensor(bounds_np[:, 0], dtype=torch.float32)
    high = torch.tensor(bounds_np[:, 1], dtype=torch.float32)
    return BoxUniform(low=low, high=high)


class TrajectoryEmbedding(nn.Module):
    """Small 1D CNN that compresses a flattened coarse trajectory."""

    def __init__(self, num_observations: int) -> None:
        super().__init__()
        channels = 32
        embedding_dim = 24
        self.num_observations = num_observations
        self.features = nn.Sequential(
            nn.Conv1d(6, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.shape[0], self.num_observations, 6).transpose(1, 2)
        return self.project(self.features(x))


def sample_posterior_quiet(posterior: Any, num_samples: int, x: Tensor) -> Tensor:
    """Sample from the posterior while silencing progress output."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return posterior.sample((num_samples,), x=x, show_progress_bars=False)


def posterior_summary(samples: Tensor) -> pd.DataFrame:
    """Summarize posterior samples for joint inference on ``Kp`` and ``Kd``."""

    log10_kp = samples[:, 0].detach().cpu().numpy()
    log10_kd = samples[:, 1].detach().cpu().numpy()
    kp = 10.0 ** log10_kp
    kd = 10.0 ** log10_kd
    quantiles = [0.05, 0.5, 0.95]
    return pd.DataFrame(
        {
            "parameter": ["log10(Kp)", "Kp", "log10(Kd)", "Kd"],
            "q05": [
                np.quantile(log10_kp, quantiles[0]),
                np.quantile(kp, quantiles[0]),
                np.quantile(log10_kd, quantiles[0]),
                np.quantile(kd, quantiles[0]),
            ],
            "median": [
                np.quantile(log10_kp, quantiles[1]),
                np.quantile(kp, quantiles[1]),
                np.quantile(log10_kd, quantiles[1]),
                np.quantile(kd, quantiles[1]),
            ],
            "q95": [
                np.quantile(log10_kp, quantiles[2]),
                np.quantile(kp, quantiles[2]),
                np.quantile(log10_kd, quantiles[2]),
                np.quantile(kd, quantiles[2]),
            ],
        }
    )


def _prepare_sbi(config: BaseDemoConfig, prior_bounds: PriorBounds) -> tuple[Any, Any]:
    from sbi.utils.user_input_checks import (
        check_sbi_inputs,
        process_prior,
        process_simulator,
    )

    prior = make_box_prior(prior_bounds)
    prior, _, prior_returns_numpy = process_prior(prior)
    processed_simulator = process_simulator(
        lambda theta: simulator(theta, config=config),
        prior,
        prior_returns_numpy,
    )
    check_sbi_inputs(processed_simulator, prior)
    return prior, processed_simulator


def _build_density_estimator(model: str, config: BaseDemoConfig) -> Any:
    from sbi.neural_nets import posterior_nn

    return posterior_nn(
        model=model,
        embedding_net=TrajectoryEmbedding(config.num_observations),
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=64,
        num_transforms=4,
    )


def _make_density_thresholder_quiet(
    dist: Any,
    quantile: float,
    num_samples_to_estimate_support: int,
) -> Any:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        samples = dist.sample(
            (num_samples_to_estimate_support,),
            show_progress_bars=False,
        )
        log_probs = dist.log_prob(samples)

    sorted_log_probs, _ = torch.sort(log_probs)
    threshold_index = min(
        max(int(quantile * num_samples_to_estimate_support), 0),
        num_samples_to_estimate_support - 1,
    )
    log_prob_threshold = sorted_log_probs[threshold_index]

    def density_thresholder(theta: Tensor) -> Tensor:
        return (dist.log_prob(theta) > log_prob_threshold).bool()

    return density_thresholder


def run_sequential_npe(
    x_o: Tensor,
    config: BaseDemoConfig,
    prior_bounds: PriorBounds,
    num_rounds: int = 3,
    simulations_per_round: int = 192,
    training_batch_size: int = 64,
    round_posterior_sample_size: int = 1000,
    restricted_prior_quantile: float = 5e-3,
    restricted_prior_support_samples: int = 20_000,
    seed: int = 7,
    density_model: str = "zuko_maf",
) -> dict[str, Any]:
    """Run sequential NPE for joint inference on ``Kp`` and ``Kd``."""

    from sbi.inference import NPE, simulate_for_sbi
    from sbi.utils import RestrictedPrior

    torch.manual_seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings(
        "ignore",
        message="IProgress not found. Please update jupyter and ipywidgets.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="The proposal you passed is a `RestrictedPrior`",
    )

    prior, processed_simulator = _prepare_sbi(config=config, prior_bounds=prior_bounds)
    density_estimator = _build_density_estimator(model=density_model, config=config)
    inference = NPE(prior=prior, density_estimator=density_estimator)

    proposal: Any = prior
    round_summaries: list[dict[str, Any]] = []
    round_posterior_samples: list[np.ndarray] = []

    for round_idx in range(num_rounds):
        proposal_label = "prior" if round_idx == 0 else "restricted prior"

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            theta, x = simulate_for_sbi(
                processed_simulator,
                proposal,
                num_simulations=simulations_per_round,
                seed=seed + round_idx,
                show_progress_bar=False,
            )
            density_estimator = inference.append_simulations(
                theta,
                x,
                proposal=proposal,
            ).train(
                training_batch_size=training_batch_size,
                learning_rate=3e-4,
                validation_fraction=0.1,
                stop_after_epochs=15,
                max_num_epochs=120,
                clip_max_norm=5.0,
                use_combined_loss=True,
                show_train_summary=False,
            )
            posterior = inference.build_posterior(density_estimator).set_default_x(x_o)

        posterior_samples = sample_posterior_quiet(
            posterior=posterior,
            num_samples=round_posterior_sample_size,
            x=x_o,
        )
        posterior_values = posterior_samples.detach().cpu().numpy().copy()
        round_posterior_samples.append(posterior_values)

        log10_kp = posterior_values[:, 0]
        log10_kd = posterior_values[:, 1]
        round_summaries.append(
            {
                "round": round_idx + 1,
                "proposal": proposal_label,
                "num_simulations": int(theta.shape[0]),
                "x_dimension": int(x.shape[-1]),
                "log10_kp_q05": float(np.quantile(log10_kp, 0.05)),
                "log10_kp_median": float(np.quantile(log10_kp, 0.5)),
                "log10_kp_q95": float(np.quantile(log10_kp, 0.95)),
                "log10_kd_q05": float(np.quantile(log10_kd, 0.05)),
                "log10_kd_median": float(np.quantile(log10_kd, 0.5)),
                "log10_kd_q95": float(np.quantile(log10_kd, 0.95)),
                "posterior_corr": float(np.corrcoef(log10_kp, log10_kd)[0, 1]),
            }
        )

        if round_idx < num_rounds - 1:
            density_thresholder = _make_density_thresholder_quiet(
                posterior,
                quantile=restricted_prior_quantile,
                num_samples_to_estimate_support=restricted_prior_support_samples,
            )
            proposal = RestrictedPrior(
                prior=make_box_prior(prior_bounds),
                accept_reject_fn=density_thresholder,
                posterior=posterior,
                sample_with="rejection",
            )

    return {
        "prior": prior,
        "posterior": posterior,
        "round_summaries": round_summaries,
        "round_posterior_samples": round_posterior_samples,
        "method": (
            f"Sequential NPE with {density_model}, a learned trajectory embedding, "
            "and restricted-prior proposals"
        ),
    }
