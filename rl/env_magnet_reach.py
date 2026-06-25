from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from config.magnet_layouts import W_same_direction
from config.paths_and_params import k_workspace_same
from config.physical_constants import N_default, l_default, m_seg_default
from core.energy import total_energy
from core.kinematics import forward_positions


@dataclass
class MagnetReachEnvConfig:
    """Configuration for the 2D magnetic reaching task."""

    n_segments: int = N_default
    segment_length: float = l_default
    elastic_k: float = k_workspace_same
    segment_moment: float = m_seg_default
    max_steps: int = 500
    goal_tolerance: float = 1.0e-3
    radius_bounds: tuple[float, float] = (0.01, 0.1)
    magnet_angle_bounds: tuple[float, float] = (-1.5 * np.pi, 1.5 * np.pi)
    fixed_reset_radius: float = 0.03
    fixed_reset_magnet_angle: float = 0.0
    fixed_reset_sigma_angle: float = 0.0
    target_radius_delta: float = 1.0e-2
    target_magnet_angle_delta: float = np.deg2rad(30.0)
    target_sigma_angle_delta: float = np.deg2rad(45.0)

    radius_delta: float = 2.0e-3  # 永磁体半径的增量delta r
    magnet_angle_delta: float = np.deg2rad(3.0)
    sigma_angle_delta: float = np.deg2rad(10.0)

    beta_bounds: tuple[float, float] = (-np.pi, np.pi)
    optimizer_maxiter: int = 80
    optimizer_ftol: float = 1.0e-8
    success_bonus: float = 10.0


class MagnetReachEnv:
    """
    Magnetic 2D reaching environment.

    The action is a normalized 3D vector controlling increments of external
    magnet radius, magnet polar angle, and magnetization direction angle.
    The robot transition is computed by minimizing the existing pseudo-rigid
    body magnetic-elastic energy model.
    """

    action_dim = 3
    observation_dim = 9

    def __init__(
        self,
        config: MagnetReachEnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or MagnetReachEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.W = W_same_direction(self.config.n_segments)
        self.length_scale = max(
            self.config.n_segments * self.config.segment_length,
            np.finfo(float).eps,
        )

        self.beta = np.zeros(self.config.n_segments, dtype=np.float64)
        self.radius = 0.5 * sum(self.config.radius_bounds)
        self.magnet_angle = 0.5 * sum(self.config.magnet_angle_bounds)
        self.sigma_angle = 0.0
        self.target = np.zeros(2, dtype=np.float64)
        self.end_position = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.last_opt_success = True
        self.last_opt_fun = 0.0

    def reset(self) -> np.ndarray:
        """Start a new episode and return the first observation."""
        self.steps = 0
        self.radius = float(np.clip(self.config.fixed_reset_radius, *self.config.radius_bounds))
        self.magnet_angle = float(
            np.clip(
                self.config.fixed_reset_magnet_angle,
                *self.config.magnet_angle_bounds,
            )
        )
        self.sigma_angle = self._wrap_angle(self.config.fixed_reset_sigma_angle)
        self.beta = np.zeros(self.config.n_segments, dtype=np.float64)
        self.beta, self.end_position = self._solve_current_shape(self.beta)

        self.target = self._sample_reachable_target()
        for _ in range(10):
            if np.linalg.norm(self.target - self.end_position) > self.config.goal_tolerance:
                break
            self.target = self._sample_reachable_target()

        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Apply one normalized action and advance the environment."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"Expected action with shape ({self.action_dim},), got {action.shape}.")

        action = np.clip(action, -1.0, 1.0)
        self.radius = float(
            np.clip(
                self.radius + action[0] * self.config.radius_delta,
                *self.config.radius_bounds,
            )
        )
        self.magnet_angle = self._wrap_angle(
            self.magnet_angle + action[1] * self.config.magnet_angle_delta
        )
        self.sigma_angle = self._wrap_angle(
            self.sigma_angle + action[2] * self.config.sigma_angle_delta
        )

        self.beta, self.end_position = self._solve_current_shape(self.beta)
        self.steps += 1

        distance = float(np.linalg.norm(self.end_position - self.target))
        success = distance <= self.config.goal_tolerance
        timeout = self.steps >= self.config.max_steps
        reward = -distance
        if success:
            reward += self.config.success_bonus

        info = {
            "distance": distance,
            "success": success,
            "timeout": timeout,
            "end_position": self.end_position.copy(),
            "target": self.target.copy(),
            "beta": self.beta.copy(),
            "magnet_position": self._magnet_position().copy(),
            "sigma_m": self._sigma_m().copy(),
            "radius": self.radius,
            "magnet_angle": self.magnet_angle,
            "sigma_angle": self.sigma_angle,
            "opt_success": self.last_opt_success,
            "opt_fun": self.last_opt_fun,
        }
        return self._observation(), float(reward), bool(success or timeout), info

    def sample_action(self) -> np.ndarray:
        """Sample a random normalized action."""
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim)

    def current_positions(self) -> np.ndarray:
        """Return all pseudo-rigid body node positions for the current beta."""
        return forward_positions(self.beta, self.config.segment_length)

    def _sample_reachable_target(self) -> np.ndarray:
        old_radius = self.radius
        old_magnet_angle = self.magnet_angle
        old_sigma_angle = self.sigma_angle

        self.radius = self._sample_clipped_delta(
            self.config.fixed_reset_radius,
            self.config.target_radius_delta,
            self.config.radius_bounds,
        )
        self.magnet_angle = self._sample_clipped_delta(
            self.config.fixed_reset_magnet_angle,
            self.config.target_magnet_angle_delta,
            self.config.magnet_angle_bounds,
        )
        self.sigma_angle = self._wrap_angle(
            self.config.fixed_reset_sigma_angle
            + self.rng.uniform(
                -self.config.target_sigma_angle_delta,
                self.config.target_sigma_angle_delta,
            )
        )
        _, target = self._solve_current_shape(np.zeros_like(self.beta))

        self.radius = old_radius
        self.magnet_angle = old_magnet_angle
        self.sigma_angle = old_sigma_angle
        return target

    def _sample_clipped_delta(
        self,
        center: float,
        delta: float,
        bounds: tuple[float, float],
    ) -> float:
        value = center + self.rng.uniform(-delta, delta)
        return float(np.clip(value, *bounds))

    def _solve_current_shape(self, beta0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = minimize(
            lambda beta: total_energy(
                beta,
                self._magnet_position(),
                self._sigma_m(),
                self.W,
                self.config.elastic_k,
                self.config.segment_length,
                self.config.segment_moment,
            ),
            np.asarray(beta0, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[self.config.beta_bounds] * self.config.n_segments,
            options={
                "maxiter": self.config.optimizer_maxiter,
                "ftol": self.config.optimizer_ftol,
            },
        )
        self.last_opt_success = bool(res.success)
        self.last_opt_fun = float(res.fun)
        beta = np.asarray(res.x, dtype=np.float64)
        end_position = forward_positions(beta, self.config.segment_length)[-1, :2]
        return beta, end_position

    def _observation(self) -> np.ndarray:
        radius_min, radius_max = self.config.radius_bounds
        radius_norm = 2.0 * (self.radius - radius_min) / (radius_max - radius_min) - 1.0
        relative = (self.target - self.end_position) / self.length_scale

        obs = np.array(
            [
                relative[0],
                relative[1],
                self.end_position[0] / self.length_scale,
                self.end_position[1] / self.length_scale,
                radius_norm,
                np.cos(self.magnet_angle),
                np.sin(self.magnet_angle),
                np.cos(self.sigma_angle),
                np.sin(self.sigma_angle),
            ],
            dtype=np.float32,
        )
        return obs

    def _magnet_position(self) -> np.ndarray:
        return np.array(
            [
                self.radius * np.cos(self.magnet_angle),
                self.radius * np.sin(self.magnet_angle),
                0.0,
            ],
            dtype=np.float64,
        )

    def _sigma_m(self) -> np.ndarray:
        return np.array(
            [np.cos(self.sigma_angle), np.sin(self.sigma_angle), 0.0],
            dtype=np.float64,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
