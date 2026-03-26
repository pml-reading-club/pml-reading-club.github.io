"""Simulator used by ``notebooks/final.ipynb``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

MU_EARTH_KM = 3.986004418e5
R_EARTH_KM = 6378.137
GEO_ALTITUDE_KM = 35786.0
GEO_RADIUS_KM = R_EARTH_KM + GEO_ALTITUDE_KM


@dataclass(frozen=True)
class BaseDemoConfig:
    """Simulation settings for the coarse GEO residual model."""

    horizon_s: float = 10.0 * 3600.0
    dt_internal_s: float = 30.0
    dt_observation_s: float = 25.0 * 60.0
    process_noise_std: float = 3.0e-6
    controller_enabled: bool = True
    controller_gain_scale: float = 1.0
    initial_position_offset_km: tuple[float, float, float] = (25.0, -15.0, 5.0)
    initial_velocity_offset_kms: tuple[float, float, float] = (0.0, 0.0015, -0.0007)

    @property
    def num_internal_steps(self) -> int:
        ratio = self.horizon_s / self.dt_internal_s
        rounded = round(ratio)
        if not np.isclose(ratio, rounded):
            raise ValueError("horizon_s must be an integer multiple of dt_internal_s.")
        return int(rounded)

    @property
    def observation_stride(self) -> int:
        ratio = self.dt_observation_s / self.dt_internal_s
        rounded = round(ratio)
        if not np.isclose(ratio, rounded):
            raise ValueError("dt_observation_s must be an integer multiple of dt_internal_s.")
        return int(rounded)

    @property
    def num_observations(self) -> int:
        return self.num_internal_steps // self.observation_stride + 1

    def gains_from_theta(self, theta_np: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class JointDemoConfig(BaseDemoConfig):
    """Infer ``[log10(Kp), log10(Kd)]`` jointly."""

    def gains_from_theta(self, theta_np: np.ndarray) -> tuple[float, float]:
        if theta_np.shape != (2,):
            raise ValueError("theta must have shape (2,) with [log10(Kp), log10(Kd)].")
        return float(10.0 ** theta_np[0]), float(10.0 ** theta_np[1])


def theta_to_numpy(theta: np.ndarray | Tensor) -> np.ndarray:
    """Convert a torch or numpy parameter vector into a numpy array."""

    if isinstance(theta, Tensor):
        return theta.detach().cpu().numpy().astype(float, copy=False)
    return np.asarray(theta, dtype=float)


def decode_theta(theta: np.ndarray | Tensor, config: BaseDemoConfig) -> tuple[float, float]:
    """Map inferred log-gains to positive PD gains."""

    return config.gains_from_theta(theta_to_numpy(theta))


def config_table(config: BaseDemoConfig) -> pd.DataFrame:
    """Human-readable summary of the simulation setup."""

    return pd.DataFrame(
        {
            "quantity": [
                "Horizon [h]",
                "Internal integration step dt [s]",
                "Observation spacing DT [min]",
                "Number of internal steps",
                "Number of coarse observations",
                "Observation dimension 6N",
                "Controller enabled",
                "Controller gain scale",
                "Process noise std",
                "Initial position offset [km]",
                "Initial velocity offset [km/s]",
            ],
            "value": [
                config.horizon_s / 3600.0,
                config.dt_internal_s,
                config.dt_observation_s / 60.0,
                config.num_internal_steps,
                config.num_observations,
                6 * config.num_observations,
                config.controller_enabled,
                config.controller_gain_scale,
                config.process_noise_std,
                tuple(float(value) for value in config.initial_position_offset_km),
                tuple(float(value) for value in config.initial_velocity_offset_kms),
            ],
        }
    )


def true_parameter_table(
    theta_true: Tensor,
    config: BaseDemoConfig | None = None,
) -> pd.DataFrame:
    """Convenience table for the synthetic truth used in the notebook."""

    config = config or JointDemoConfig()
    kp, kd = decode_theta(theta_true, config)
    return pd.DataFrame(
        {
            "parameter": ["log10(Kp)", "Kp", "log10(Kd)", "Kd"],
            "value": [float(theta_true[0]), kp, float(theta_true[1]), kd],
        }
    )


def observation_times(config: BaseDemoConfig) -> np.ndarray:
    """Times at which coarse observations are returned."""

    return np.linspace(0.0, config.horizon_s, config.num_observations, dtype=float)


def mean_motion() -> float:
    """Circular GEO mean motion in rad/s."""

    return float(np.sqrt(MU_EARTH_KM / GEO_RADIUS_KM**3))


def reference_states(times_s: np.ndarray) -> np.ndarray:
    """Circular GEO reference states at the supplied times."""

    n = mean_motion()
    theta = n * np.asarray(times_s, dtype=float)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x = GEO_RADIUS_KM * cos_theta
    y = GEO_RADIUS_KM * sin_theta
    vx = -GEO_RADIUS_KM * n * sin_theta
    vy = GEO_RADIUS_KM * n * cos_theta
    zeros = np.zeros_like(x)
    return np.column_stack((x, y, zeros, vx, vy, zeros))


def reference_state(time_s: float) -> np.ndarray:
    """Circular GEO reference state at one time."""

    return reference_states(np.asarray([time_s], dtype=float))[0]


def central_gravity(r_km: np.ndarray) -> np.ndarray:
    """Two-body Earth gravity."""

    r_norm = np.linalg.norm(r_km)
    return -MU_EARTH_KM / r_norm**3 * r_km


def simulate_trajectory(
    theta: np.ndarray | Tensor,
    config: BaseDemoConfig,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one trajectory and return its coarse residual states."""

    kp, kd = decode_theta(theta, config)
    times_s = np.linspace(0.0, config.horizon_s, config.num_internal_steps + 1, dtype=float)
    reference = reference_states(times_s)

    state = reference[0].copy()
    state[:3] += np.asarray(config.initial_position_offset_km, dtype=float)
    state[3:] += np.asarray(config.initial_velocity_offset_kms, dtype=float)

    states = np.zeros_like(reference, dtype=float)
    states[0] = state

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(config.dt_internal_s)

    def control_from_state(current_state: np.ndarray, time_s: float) -> np.ndarray:
        ref_state = reference_state(time_s)
        delta_r = current_state[:3] - ref_state[:3]
        delta_v = current_state[3:] - ref_state[3:]
        if not config.controller_enabled:
            return np.zeros(3, dtype=float)
        return config.controller_gain_scale * (-kp * delta_r - kd * delta_v)

    def drift(current_state: np.ndarray, time_s: float) -> np.ndarray:
        total_accel = central_gravity(current_state[:3]) + control_from_state(current_state, time_s)
        return np.hstack((current_state[3:], total_accel))

    for step in range(config.num_internal_steps):
        time_s = times_s[step]
        k1 = drift(state, time_s)
        k2 = drift(state + 0.5 * config.dt_internal_s * k1, time_s + 0.5 * config.dt_internal_s)
        k3 = drift(state + 0.5 * config.dt_internal_s * k2, time_s + 0.5 * config.dt_internal_s)
        k4 = drift(state + config.dt_internal_s * k3, time_s + config.dt_internal_s)
        state_drift = state + config.dt_internal_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        velocity = state_drift[3:]
        if config.process_noise_std > 0.0:
            velocity += config.process_noise_std * sqrt_dt * rng.standard_normal(3)
        state = np.hstack((state_drift[:3], velocity))
        states[step + 1] = state

    coarse_times_s = times_s[:: config.observation_stride]
    coarse_reference = reference[:: config.observation_stride]
    coarse_states = states[:: config.observation_stride]

    return {
        "times_s": times_s,
        "reference": reference,
        "states": states,
        "coarse_times_s": coarse_times_s,
        "coarse_reference": coarse_reference,
        "coarse_states": coarse_states,
        "coarse_residuals": coarse_states - coarse_reference,
    }


def simulate_coarse_residuals(
    theta: np.ndarray | Tensor,
    config: BaseDemoConfig,
    seed: int | None = None,
) -> np.ndarray:
    """Return residual coarse states for one simulation."""

    trajectory = simulate_trajectory(theta, config=config, seed=seed)
    return trajectory["coarse_residuals"].astype(np.float32)


def simulator(theta: Tensor, config: BaseDemoConfig) -> Tensor:
    """Torch-compatible simulator returning a flattened coarse trajectory."""

    residuals = simulate_coarse_residuals(theta, config=config)
    return torch.from_numpy(residuals.reshape(-1))


def make_observation(
    theta_true: Tensor,
    config: BaseDemoConfig,
    seed: int | None = 123,
) -> Tensor:
    """Generate one synthetic observation from the simulator."""

    residuals = simulate_coarse_residuals(theta_true, config=config, seed=seed)
    return torch.from_numpy(residuals.reshape(-1)).to(torch.float32)
