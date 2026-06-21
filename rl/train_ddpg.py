from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ddpg import DDPGAgent, DDPGConfig
from rl.env_magnet_reach import MagnetReachEnv, MagnetReachEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG for magnetic 2D reaching.")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--goal-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--success-bonus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ddpg_magnet"))
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = MagnetReachEnv(
        MagnetReachEnvConfig(
            max_steps=args.max_steps,
            goal_tolerance=args.goal_tolerance,
            optimizer_maxiter=args.optimizer_maxiter,
            success_bonus=args.success_bonus,
        ),
        seed=args.seed,
    )
    agent = DDPGAgent(
        env.observation_dim,
        env.action_dim,
        config=DDPGConfig(batch_size=args.batch_size, replay_size=args.replay_size),
        device=args.device,
        seed=args.seed,
    )

    start_episode = 1
    total_steps = 0
    if args.resume_from is not None:
        checkpoint = agent.load(args.resume_from, load_optimizers=True)
        start_episode = int(checkpoint.get("episode", 0)) + 1
        total_steps = int(checkpoint.get("total_steps", 0))
        print(
            f"Resumed from {args.resume_from} at episode {start_episode}, "
            f"total_steps={total_steps}"
        )

    log_path = args.output_dir / "training_log.csv"
    append_log = bool(args.resume_from is not None and log_path.exists())
    last_episode = start_episode - 1

    try:
        for episode in range(start_episode, args.episodes + 1):
            observation = env.reset()
            agent.noise.reset()
            episode_reward = 0.0
            last_info: dict = {}
            losses: list[dict[str, float]] = []

            for step in range(1, args.max_steps + 1):
                if total_steps < args.warmup_steps:
                    action = env.sample_action().astype(np.float32)
                else:
                    action = agent.select_action(observation, add_noise=True)

                next_observation, reward, done, info = env.step(action)
                agent.replay_buffer.add(observation, action, reward, next_observation, done)
                loss_info = agent.train_step()
                if loss_info is not None:
                    losses.append(loss_info)

                observation = next_observation
                episode_reward += reward
                total_steps += 1
                last_info = info
                if done:
                    break

            mean_actor_loss = float(np.mean([x["actor_loss"] for x in losses])) if losses else np.nan
            mean_critic_loss = float(np.mean([x["critic_loss"] for x in losses])) if losses else np.nan
            row = {
                "episode": episode,
                "steps": step,
                "total_steps": total_steps,
                "reward": episode_reward,
                "distance": float(last_info.get("distance", np.nan)),
                "success": int(bool(last_info.get("success", False))),
                "actor_loss": mean_actor_loss,
                "critic_loss": mean_critic_loss,
                "buffer_size": len(agent.replay_buffer),
            }
            write_csv_row(log_path, row, append=append_log)
            append_log = True
            last_episode = episode
            print(
                "episode={episode} steps={steps} reward={reward:.6g} "
                "distance={distance:.6g} success={success} buffer={buffer_size}".format(**row)
            )

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                save_training_checkpoint(
                    agent,
                    args.output_dir / "checkpoint_latest.pt",
                    episode,
                    total_steps,
                    args,
                )

    except KeyboardInterrupt:
        interrupted_path = args.output_dir / "checkpoint_interrupted.pt"
        save_training_checkpoint(agent, interrupted_path, last_episode, total_steps, args)
        save_training_checkpoint(agent, args.output_dir / "checkpoint_latest.pt", last_episode, total_steps, args)
        print(f"\nTraining interrupted. Saved checkpoint to {interrupted_path}")
        return

    model_path = args.output_dir / "model.pt"
    agent.save(model_path)
    save_training_checkpoint(agent, args.output_dir / "checkpoint_latest.pt", last_episode, total_steps, args)
    print(f"Saved model to {model_path}")
    print(f"Saved log to {log_path}")


def save_training_checkpoint(
    agent: DDPGAgent,
    path: Path,
    episode: int,
    total_steps: int,
    args: argparse.Namespace,
) -> None:
    agent.save(
        path,
        extra={
            "episode": episode,
            "total_steps": total_steps,
            "train_args": serializable_args(args),
        },
    )
    print(f"Saved checkpoint to {path}")


def serializable_args(args: argparse.Namespace) -> dict:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def write_csv_row(path: Path, row: dict[str, float | int], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not append:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
