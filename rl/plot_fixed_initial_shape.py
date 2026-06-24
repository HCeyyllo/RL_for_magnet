from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.magnet_layouts import W_same_direction
from core.energy import total_energy
from core.kinematics import forward_positions
from core.magnetic_field import magnetic_field_dipole
from rl.env_magnet_reach import MagnetReachEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the equilibrium shape for the fixed reset magnet with a small y offset."
    )
    parser.add_argument("--y-offset-mm", type=float, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/fixed_magnet_check/fixed_reset_y_offset_shape.png"),
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MagnetReachEnvConfig()
    W = W_same_direction(config.n_segments)

    y_offset = args.y_offset_mm * 1.0e-3
    p_magnet = np.array(
        [
            config.fixed_reset_radius * np.cos(config.fixed_reset_magnet_angle),
            config.fixed_reset_radius * np.sin(config.fixed_reset_magnet_angle) + y_offset,
            0.0,
        ],
        dtype=np.float64,
    )
    sigma_m = np.array(
        [
            np.cos(config.fixed_reset_sigma_angle),
            np.sin(config.fixed_reset_sigma_angle),
            0.0,
        ],
        dtype=np.float64,
    )

    res = minimize(
        lambda beta: total_energy(
            beta,
            p_magnet,
            sigma_m,
            W,
            config.elastic_k,
            config.segment_length,
            config.segment_moment,
        ),
        np.zeros(config.n_segments, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[config.beta_bounds] * config.n_segments,
        options={
            "maxiter": args.optimizer_maxiter or config.optimizer_maxiter,
            "ftol": config.optimizer_ftol,
        },
    )

    beta = np.asarray(res.x, dtype=np.float64)
    positions = forward_positions(beta, config.segment_length)
    end_position = positions[-1]

    segment_direction = W[0] / max(np.linalg.norm(W[0]), np.finfo(float).eps)
    sigma_direction = sigma_m / max(np.linalg.norm(sigma_m), np.finfo(float).eps)
    direction_dot = float(np.dot(segment_direction, sigma_direction))
    relation = "same" if direction_dot > 1.0e-9 else "opposite" if direction_dot < -1.0e-9 else "orthogonal"

    field_at_tip = magnetic_field_dipole(end_position, p_magnet, sigma_m)
    field_dot = float(np.dot(field_at_tip, segment_direction))
    field_relation = "same" if field_dot > 0.0 else "opposite" if field_dot < 0.0 else "orthogonal"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_shape(args.output, positions, p_magnet, sigma_m, relation, direction_dot)

    print(f"Saved shape plot to {args.output}")
    print(f"optimizer_success={res.success} optimizer_fun={float(res.fun):.6g}")
    print(f"magnet_position_m={p_magnet.tolist()}")
    print(f"magnet_position_mm={(p_magnet * 1000.0).tolist()}")
    print(f"sigma_m={sigma_m.tolist()}")
    print(f"continuum_segment_direction={segment_direction.tolist()}")
    print(f"sigma_vs_segment_dot={direction_dot:.6g} relation={relation}")
    print(f"tip_position_mm={(end_position * 1000.0).tolist()}")
    print(f"field_at_tip_dot_segment={field_dot:.6g} field_relation={field_relation}")


def plot_shape(
    path: Path,
    positions: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    relation: str,
    direction_dot: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        positions[:, 0] * 1000.0,
        positions[:, 1] * 1000.0,
        marker="o",
        markersize=3,
        linewidth=2,
        label="equilibrium continuum",
    )
    ax.scatter([0.0], [0.0], c="black", s=30, label="base")
    ax.scatter(
        [p_magnet[0] * 1000.0],
        [p_magnet[1] * 1000.0],
        c="red",
        s=55,
        marker="x",
        label="external magnet",
    )

    arrow_scale_mm = 4.0
    ax.arrow(
        p_magnet[0] * 1000.0,
        p_magnet[1] * 1000.0,
        sigma_m[0] * arrow_scale_mm,
        sigma_m[1] * arrow_scale_mm,
        width=0.025,
        head_width=0.35,
        length_includes_head=True,
        color="red",
    )
    ax.text(
        p_magnet[0] * 1000.0,
        p_magnet[1] * 1000.0 + 0.8,
        "sigma_m",
        color="red",
        fontsize=9,
        ha="center",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Fixed reset magnet + y offset: {relation} (dot={direction_dot:.3g})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
