from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ddpg import DDPGAgent, DDPGConfig, OUNoise
from rl.env_magnet_reach import MagnetReachEnv, MagnetReachEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG for magnetic 2D reaching.")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--parallel-backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    parser.add_argument("--goal-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ddpg_magnet"))
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def env_worker(conn, config: MagnetReachEnvConfig, seed: int) -> None:
    env = MagnetReachEnv(config, seed=seed)
    try:
        while True:
            command, payload = conn.recv()
            try:
                if command == "reset":
                    response = env.reset()
                elif command == "sample_action":
                    response = env.sample_action().astype(np.float32)
                elif command == "step":
                    response = env.step(payload)
                elif command == "close":
                    conn.send(("ok", None))
                    break
                else:
                    raise ValueError(f"Unknown worker command: {command}")
                conn.send(("ok", response))
            except Exception as exc:
                conn.send(("error", repr(exc)))
    except EOFError:
        pass
    finally:
        conn.close()


def recv_worker_response(conn):
    status, payload = conn.recv()
    if status == "error":
        raise RuntimeError(f"Environment worker failed: {payload}")
    return payload


def start_env_workers(
    env_config: MagnetReachEnvConfig,
    num_envs: int,
    seed: int,
) -> tuple[list, list[mp.Process]]:
    ctx = mp.get_context("spawn")
    conns = []
    processes = []
    for env_index in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=env_worker,
            args=(child_conn, env_config, seed + env_index),
            daemon=True,
        )
        process.start()
        child_conn.close()
        conns.append(parent_conn)
        processes.append(process)
    return conns, processes


def close_env_workers(conns: list, processes: list[mp.Process]) -> None:
    for conn in conns:
        try:
            conn.send(("close", None))
            recv_worker_response(conn)
        except (BrokenPipeError, EOFError, RuntimeError):
            pass
        finally:
            conn.close()

    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def reset_envs(
    envs: list[MagnetReachEnv],
    conns: list | None,
    active_count: int,
) -> list[np.ndarray]:
    if conns is None:
        return [envs[i].reset() for i in range(active_count)]

    for conn in conns[:active_count]:
        conn.send(("reset", None))
    return [recv_worker_response(conn) for conn in conns[:active_count]]


def sample_warmup_actions(
    envs: list[MagnetReachEnv],
    conns: list | None,
    active_indices: list[int],
) -> np.ndarray:
    if conns is None:
        return np.stack([envs[i].sample_action().astype(np.float32) for i in active_indices])

    for i in active_indices:
        conns[i].send(("sample_action", None))
    return np.stack([recv_worker_response(conns[i]) for i in active_indices])


def step_envs(
    envs: list[MagnetReachEnv],
    conns: list | None,
    executor: ThreadPoolExecutor | None,
    active_indices: list[int],
    actions: np.ndarray,
) -> list[tuple[np.ndarray, float, bool, dict]]:
    if conns is None:
        if executor is None:
            raise RuntimeError("Thread executor is required for thread backend.")
        futures = [
            executor.submit(envs[i].step, action)
            for i, action in zip(active_indices, actions)
        ]
        return [future.result() for future in futures]

    for i, action in zip(active_indices, actions):
        conns[i].send(("step", action))
    return [recv_worker_response(conns[i]) for i in active_indices]


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    num_workers = args.num_workers or args.num_envs
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")
    if args.parallel_backend == "process" and num_workers != args.num_envs:
        raise ValueError("--num-workers must equal --num-envs when using process backend.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_config = MagnetReachEnvConfig(
        max_steps=args.max_steps,
        goal_tolerance=args.goal_tolerance,
        optimizer_maxiter=args.optimizer_maxiter,
        success_bonus=args.success_bonus,
    )
    envs: list[MagnetReachEnv] = []
    env_conns = None
    env_processes: list[mp.Process] = []
    if args.parallel_backend == "process":
        env_conns, env_processes = start_env_workers(env_config, args.num_envs, args.seed)
    else:
        envs = [MagnetReachEnv(env_config, seed=args.seed + i) for i in range(args.num_envs)]

    agent = DDPGAgent(
        MagnetReachEnv.observation_dim,
        MagnetReachEnv.action_dim,
        config=DDPGConfig(batch_size=args.batch_size, replay_size=args.replay_size),
        device=args.device,
        seed=args.seed,
    )
    exploration_noises = [
        OUNoise(
            MagnetReachEnv.action_dim,
            mu=agent.config.ou_mu,
            theta=agent.config.ou_theta,
            sigma=agent.config.ou_sigma,
            seed=args.seed + i,
        )
        for i in range(args.num_envs)
    ]

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
        next_episode = start_episode
        executor_context = (
            ThreadPoolExecutor(max_workers=num_workers)
            if env_conns is None
            else nullcontext(None)
        )
        with executor_context as executor:
            while next_episode <= args.episodes:
                active_count = min(args.num_envs, args.episodes - next_episode + 1)
                episode_ids = list(range(next_episode, next_episode + active_count))
                next_episode += active_count

                observations = reset_envs(envs, env_conns, active_count)
                for noise in exploration_noises[:active_count]:
                    noise.reset()

                active = [True] * active_count
                episode_rewards = [0.0] * active_count
                episode_steps = [0] * active_count
                last_infos: list[dict] = [{} for _ in range(active_count)]
                episode_losses: list[list[dict[str, float]]] = [[] for _ in range(active_count)]

                for _ in range(args.max_steps):
                    active_indices = [i for i, is_active in enumerate(active) if is_active]
                    if not active_indices:
                        break

                    if total_steps < args.warmup_steps:
                        actions = sample_warmup_actions(envs, env_conns, active_indices)
                    else:
                        actions = agent.select_actions(
                            np.stack([observations[i] for i in active_indices]),
                            add_noise=True,
                            noises=[exploration_noises[i] for i in active_indices],
                        )

                    results = step_envs(envs, env_conns, executor, active_indices, actions)

                    for env_index, action, result in zip(active_indices, actions, results):
                        observation = observations[env_index]
                        next_observation, reward, done, info = result

                        agent.replay_buffer.add(observation, action, reward, next_observation, done)
                        loss_info = agent.train_step()
                        if loss_info is not None:
                            episode_losses[env_index].append(loss_info)

                        observations[env_index] = next_observation
                        episode_rewards[env_index] += reward
                        episode_steps[env_index] += 1
                        total_steps += 1
                        last_infos[env_index] = info

                        if done:
                            active[env_index] = False
                            episode = episode_ids[env_index]
                            last_episode = episode
                            row = make_log_row(
                                episode,
                                episode_steps[env_index],
                                total_steps,
                                episode_rewards[env_index],
                                last_infos[env_index],
                                episode_losses[env_index],
                                len(agent.replay_buffer),
                            )
                            write_csv_row(log_path, row, append=append_log)
                            append_log = True
                            print(
                                "episode={episode} steps={steps} reward={reward:.6g} "
                                "distance={distance:.6g} success={success} buffer={buffer_size}".format(
                                    **row
                                )
                            )

                            if (
                                args.checkpoint_interval > 0
                                and episode % args.checkpoint_interval == 0
                            ):
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
    finally:
        if env_conns is not None:
            close_env_workers(env_conns, env_processes)

    model_path = args.output_dir / "model.pt"
    agent.save(model_path)
    save_training_checkpoint(agent, args.output_dir / "checkpoint_latest.pt", last_episode, total_steps, args)
    print(f"Saved model to {model_path}")
    print(f"Saved log to {log_path}")


def make_log_row(
    episode: int,
    steps: int,
    total_steps: int,
    reward: float,
    last_info: dict,
    losses: list[dict[str, float]],
    buffer_size: int,
) -> dict[str, float | int]:
    mean_actor_loss = float(np.mean([x["actor_loss"] for x in losses])) if losses else np.nan
    mean_critic_loss = float(np.mean([x["critic_loss"] for x in losses])) if losses else np.nan
    return {
        "episode": episode,
        "steps": steps,
        "total_steps": total_steps,
        "reward": reward,
        "distance": float(last_info.get("distance", np.nan)),
        "success": int(bool(last_info.get("success", False))),
        "actor_loss": mean_actor_loss,
        "critic_loss": mean_critic_loss,
        "buffer_size": buffer_size,
    }


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
