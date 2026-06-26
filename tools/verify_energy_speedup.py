"""
验证 core 向量化 + 解析梯度改造的正确性与加速比。

内容：
1. forward_positions 新实现 vs 旧逐段矩阵连乘实现（数值一致性）
2. total_energy 新实现 vs 旧逐段循环实现（数值一致性）
3. total_energy_and_grad 的解析梯度 vs scipy 有限差分（check_grad）
4. 完整环境 step 在“旧式无梯度求解”与“新式解析梯度求解”下：
   - 平衡形状 beta / 末端位置是否一致（求解成功率与精度不受影响）
   - 单步求解耗时对比（能量函数调用次数 nfev / 实测墙钟时间）

运行：python tools/verify_energy_speedup.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad, minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.magnet_layouts import W_same_direction, W_opposite_direction
from config.physical_constants import mu0
from core.energy import total_energy, total_energy_and_grad
from core.kinematics import forward_positions
from core.magnetic_field import magnetic_field_dipole, magnetic_moment_magnet
from rl.env_magnet_reach import MagnetReachEnv, MagnetReachEnvConfig


# ----------------------- 旧实现（参考基准，原样复刻） -----------------------
def forward_positions_old(beta: np.ndarray, l: float) -> np.ndarray:
    beta = np.asarray(beta)
    N = beta.size
    positions = [np.zeros(3)]
    R = np.eye(3)
    for i in range(N):
        b = beta[i]
        cb, sb = np.cos(b), np.sin(b)
        Rz = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
        pi = positions[-1] + R @ np.array([l, 0.0, 0.0])
        positions.append(pi)
    return np.asarray(positions)


def magnetic_field_dipole_old(p, p_magnet, sigma_m, m_m=None):
    if m_m is None:
        m_m = magnetic_moment_magnet()
    d = p - p_magnet
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-10:
        return np.zeros(3)
    dot = np.dot(sigma_m, d)
    term1 = 3.0 * d * dot / (d_norm**5)
    term2 = sigma_m / (d_norm**3)
    return (mu0 / (4.0 * np.pi)) * m_m * (term1 - term2)


def total_energy_old(beta, p_magnet, sigma_m, W, k, l, m_seg):
    beta = np.asarray(beta)
    positions = forward_positions_old(beta, l)
    VE = 0.0
    VB = 0.0
    R = np.eye(3)
    for i, b in enumerate(beta):
        VE += 0.5 * k * b**2
        cb, sb = np.cos(b), np.sin(b)
        Rz = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
        m_global = m_seg * (R @ W[i])
        pi = positions[i + 1]
        B = magnetic_field_dipole_old(pi, p_magnet, sigma_m)
        VB -= float(np.dot(m_global, B))
    return VE + VB


# ----------------------------- 各项检查 -----------------------------
def check_consistency(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("[1] forward_positions / total_energy 新旧实现一致性")
    cfg = MagnetReachEnvConfig()
    N, l, k, m_seg = cfg.n_segments, cfg.segment_length, cfg.elastic_k, cfg.segment_moment

    max_pos_err = 0.0
    max_e_err = 0.0
    for W in (W_same_direction(N), W_opposite_direction(N)):
        for _ in range(200):
            beta = rng.uniform(-np.pi, np.pi, size=N)
            radius = rng.uniform(0.01, 0.1)
            angle = rng.uniform(-np.pi, np.pi)
            sigma_angle = rng.uniform(-np.pi, np.pi)
            p_magnet = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            sigma_m = np.array([np.cos(sigma_angle), np.sin(sigma_angle), 0.0])

            pos_new = forward_positions(beta, l)
            pos_old = forward_positions_old(beta, l)
            max_pos_err = max(max_pos_err, np.max(np.abs(pos_new - pos_old)))

            e_new = total_energy(beta, p_magnet, sigma_m, W, k, l, m_seg)
            e_old = total_energy_old(beta, p_magnet, sigma_m, W, k, l, m_seg)
            max_e_err = max(max_e_err, abs(e_new - e_old))

    print(f"  forward_positions 最大绝对误差 = {max_pos_err:.3e}")
    print(f"  total_energy      最大绝对误差 = {max_e_err:.3e}")
    assert max_pos_err < 1e-12, "forward_positions 不一致"
    assert max_e_err < 1e-9, "total_energy 不一致"
    print("  通过：新旧实现数值一致。")


def check_gradient(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("[2] 解析梯度 vs 有限差分 (scipy.check_grad)")
    cfg = MagnetReachEnvConfig()
    N, l, k, m_seg = cfg.n_segments, cfg.segment_length, cfg.elastic_k, cfg.segment_moment
    W = W_same_direction(N)

    max_rel = 0.0
    for _ in range(50):
        beta = rng.uniform(-0.5, 0.5, size=N)
        radius = rng.uniform(0.02, 0.08)
        angle = rng.uniform(-np.pi, np.pi)
        sigma_angle = rng.uniform(-np.pi, np.pi)
        p_magnet = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        sigma_m = np.array([np.cos(sigma_angle), np.sin(sigma_angle), 0.0])

        fun = lambda b: total_energy(b, p_magnet, sigma_m, W, k, l, m_seg)
        grad = lambda b: total_energy_and_grad(b, p_magnet, sigma_m, W, k, l, m_seg)[1]

        err = check_grad(fun, grad, beta, epsilon=1e-7)
        g_norm = np.linalg.norm(grad(beta)) + 1e-30
        max_rel = max(max_rel, err / g_norm)

    print(f"  解析梯度 vs 数值梯度 最大相对误差 = {max_rel:.3e}")
    assert max_rel < 1e-4, "解析梯度可能有误"
    print("  通过：解析梯度正确。")


def _solve(beta0, p_magnet, sigma_m, W, cfg, use_grad: bool):
    if use_grad:
        fun = lambda b: total_energy_and_grad(
            b, p_magnet, sigma_m, W, cfg.elastic_k, cfg.segment_length, cfg.segment_moment
        )
        res = minimize(
            fun, beta0, method="L-BFGS-B", jac=True,
            bounds=[cfg.beta_bounds] * cfg.n_segments,
            options={"maxiter": cfg.optimizer_maxiter, "ftol": cfg.optimizer_ftol},
        )
    else:
        fun = lambda b: total_energy_old(
            b, p_magnet, sigma_m, W, cfg.elastic_k, cfg.segment_length, cfg.segment_moment
        )
        res = minimize(
            fun, beta0, method="L-BFGS-B",
            bounds=[cfg.beta_bounds] * cfg.n_segments,
            options={"maxiter": cfg.optimizer_maxiter, "ftol": cfg.optimizer_ftol},
        )
    return res


def check_solver(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("[3] 求解器：无梯度(旧) vs 解析梯度(新) —— 精度/成功率/速度")
    cfg = MagnetReachEnvConfig()
    W = W_same_direction(cfg.n_segments)

    n_trials = 300
    beta0 = np.zeros(cfg.n_segments)
    max_x_err = 0.0
    nfev_old = nfev_new = 0
    succ_old = succ_new = 0
    t_old = t_new = 0.0

    configs = []
    for _ in range(n_trials):
        radius = rng.uniform(0.02, 0.08)
        angle = rng.uniform(-np.pi / 2, np.pi / 2)
        sigma_angle = rng.uniform(-np.pi, np.pi)
        p_magnet = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
        sigma_m = np.array([np.cos(sigma_angle), np.sin(sigma_angle), 0.0])
        configs.append((p_magnet, sigma_m))

    for p_magnet, sigma_m in configs:
        t0 = time.perf_counter()
        res_old = _solve(beta0, p_magnet, sigma_m, W, cfg, use_grad=False)
        t_old += time.perf_counter() - t0
        nfev_old += res_old.nfev
        succ_old += int(res_old.success)

        t0 = time.perf_counter()
        res_new = _solve(beta0, p_magnet, sigma_m, W, cfg, use_grad=True)
        t_new += time.perf_counter() - t0
        nfev_new += res_new.nfev
        succ_new += int(res_new.success)

        end_old = forward_positions(res_old.x, cfg.segment_length)[-1, :2]
        end_new = forward_positions(res_new.x, cfg.segment_length)[-1, :2]
        max_x_err = max(max_x_err, np.linalg.norm(end_old - end_new))

    print(f"  试验次数            = {n_trials}")
    print(f"  末端位置最大差异(m) = {max_x_err:.3e}")
    print(f"  成功率  旧={succ_old}/{n_trials}  新={succ_new}/{n_trials}")
    print(f"  能量调用 nfev 总计  旧={nfev_old}  新={nfev_new}  (降为 {nfev_new/max(nfev_old,1)*100:.1f}%)")
    print(f"  墙钟时间(s)         旧={t_old:.3f}  新={t_new:.3f}  加速={t_old/max(t_new,1e-9):.2f}x")


def check_env_step(rng: np.random.Generator) -> None:
    print("=" * 70)
    print("[4] 完整环境 rollout 计时（新实现）")
    env = MagnetReachEnv(MagnetReachEnvConfig(), seed=0)
    env.reset()
    n_steps = 500
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(env.sample_action())
    dt = time.perf_counter() - t0
    print(f"  {n_steps} 步耗时 {dt:.3f}s  => 平均 {dt/n_steps*1000:.3f} ms/步")


def main() -> None:
    rng = np.random.default_rng(0)
    check_consistency(rng)
    check_gradient(rng)
    check_solver(rng)
    check_env_step(rng)
    print("=" * 70)
    print("全部检查完成。")


if __name__ == "__main__":
    main()
