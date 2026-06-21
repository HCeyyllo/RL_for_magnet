from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ddpg import DDPGAgent
from rl.env_magnet_reach import MagnetReachEnv, MagnetReachEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG magnetic reaching policy.")
    parser.add_argument("--model-path", type=Path, default=Path("runs/ddpg_magnet/model.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--optimizer-maxiter", type=int, default=40)
    parser.add_argument("--goal-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ddpg_magnet_eval"))
    parser.add_argument("--shape-every", type=int, default=5)
    parser.add_argument("--no-shape-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = MagnetReachEnv(
        MagnetReachEnvConfig(
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            optimizer_maxiter=args.optimizer_maxiter,
        ),
        seed=args.seed,
    )
    agent = DDPGAgent(env.observation_dim, env.action_dim, device=args.device, seed=args.seed)
    agent.load(args.model_path)

    rows: list[dict[str, float | int]] = []
    shape_rows: list[dict[str, float | int]] = []
    for episode in range(1, args.episodes + 1):
        observation = env.reset()
        append_shape_rows(shape_rows, episode, 0, env.current_positions(), env.target)
        for step in range(1, args.max_steps + 1):
            action = agent.select_action(observation, add_noise=False)
            observation, reward, done, info = env.step(action)
            append_shape_rows(shape_rows, episode, step, env.current_positions(), info["target"])
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "x": float(info["end_position"][0]),
                    "y": float(info["end_position"][1]),
                    "target_x": float(info["target"][0]),
                    "target_y": float(info["target"][1]),
                    "distance": float(info["distance"]),
                    "reward": float(reward),
                    "success": int(bool(info["success"])),
                    "radius": float(info["radius"]),
                    "magnet_angle": float(info["magnet_angle"]),
                    "sigma_angle": float(info["sigma_angle"]),
                }
            )
            if done:
                break

    trajectory_path = args.output_dir / "trajectory.csv"
    shape_path = args.output_dir / "shape_trajectory.csv"
    write_csv(trajectory_path, rows)
    write_csv(shape_path, shape_rows)
    plot_trajectories(args.output_dir / "trajectory.png", rows)
    if not args.no_shape_plots:
        plot_shape_processes(args.output_dir, rows, shape_rows, args.shape_every)
    print(f"Saved trajectory data to {trajectory_path}")
    print(f"Saved shape data to {shape_path}")
    print(f"Saved trajectory plot to {args.output_dir / 'trajectory.png'}")


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_shape_rows(
    shape_rows: list[dict[str, float | int]],
    episode: int,
    step: int,
    positions: np.ndarray,
    target: np.ndarray,
) -> None:
    for node_idx, position in enumerate(positions):
        shape_rows.append(
            {
                "episode": episode,
                "step": step,
                "node": node_idx,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
            }
        )


def plot_trajectories(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    episodes = sorted({int(row["episode"]) for row in rows})
    for episode in episodes:
        ep_rows = [row for row in rows if int(row["episode"]) == episode]
        xs = [float(row["x"]) * 1000.0 for row in ep_rows]
        ys = [float(row["y"]) * 1000.0 for row in ep_rows]
        target_x = float(ep_rows[-1]["target_x"]) * 1000.0
        target_y = float(ep_rows[-1]["target_y"]) * 1000.0
        ax.plot(xs, ys, marker="o", markersize=3, label=f"episode {episode}")
        ax.scatter([target_x], [target_y], marker="x", s=60)

    ax.scatter([0.0], [0.0], c="black", s=25, label="base")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("DDPG magnetic reaching trajectories")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shape_processes(
    output_dir: Path,
    trajectory_rows: list[dict[str, float | int]],
    shape_rows: list[dict[str, float | int]],
    shape_every: int,
) -> None:
    if not shape_rows:
        return
    shape_every = max(1, shape_every)
    episodes = sorted({int(row["episode"]) for row in shape_rows})
    for episode in episodes:
        ep_shapes = [row for row in shape_rows if int(row["episode"]) == episode]
        ep_trajectory = [row for row in trajectory_rows if int(row["episode"]) == episode]
        steps = sorted({int(row["step"]) for row in ep_shapes})
        plot_steps = [step for step in steps if step == 0 or step % shape_every == 0 or step == steps[-1]]

        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(plot_steps)))
        for color, step in zip(colors, plot_steps):
            step_rows = sorted(
                [row for row in ep_shapes if int(row["step"]) == step],
                key=lambda row: int(row["node"]),
            )
            xs = [float(row["x"]) * 1000.0 for row in step_rows]
            ys = [float(row["y"]) * 1000.0 for row in step_rows]
            alpha = 0.25 if step != plot_steps[-1] else 1.0
            linewidth = 1.2 if step != plot_steps[-1] else 2.4
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)

        if ep_trajectory:
            traj_x = [float(row["x"]) * 1000.0 for row in ep_trajectory]
            traj_y = [float(row["y"]) * 1000.0 for row in ep_trajectory]
            target_x = float(ep_trajectory[-1]["target_x"]) * 1000.0
            target_y = float(ep_trajectory[-1]["target_y"]) * 1000.0
            ax.plot(traj_x, traj_y, "k--", linewidth=1.0, label="tip trajectory")
            ax.scatter([target_x], [target_y], marker="x", s=80, c="red", label="target")

        ax.scatter([0.0], [0.0], c="black", s=25, label="base")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Continuum tracking process - episode {episode}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        shape_plot_path = output_dir / f"episode_{episode:03d}_shapes.png"
        fig.savefig(shape_plot_path, dpi=220)
        plt.close(fig)
        print(f"Saved shape process plot to {shape_plot_path}")


if __name__ == "__main__":
    main()
