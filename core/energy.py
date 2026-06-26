import numpy as np
from scipy.optimize import minimize

from config.physical_constants import mu0
from core.magnetic_field import magnetic_field_dipole, magnetic_moment_magnet
from core.kinematics import forward_positions


def _energy_terms(
    beta: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> dict:
    """
    向量化计算总能量所需的全部中间量，供 total_energy 与解析梯度共享。
    全部旋转绕 z 轴、且位于 z=0 平面，第 i 段全局朝向角 theta_i = cumsum(beta)_i。
    """
    beta = np.asarray(beta, dtype=np.float64)
    N = beta.size

    p_magnet = np.asarray(p_magnet, dtype=np.float64)
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    theta = np.cumsum(beta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    positions = forward_positions(beta, l)
    P = positions[1:]  # (N,3) 各段末端节点

    # 每段磁矩（全局坐标）：m_i = m_seg * Rz(theta_i) @ W[i]
    Wx, Wy, Wz = W[:, 0], W[:, 1], W[:, 2]
    m = m_seg * np.stack(
        [cos_t * Wx - sin_t * Wy, sin_t * Wx + cos_t * Wy, Wz],
        axis=1,
    )
    # dm_i / dtheta_i
    m_prime = m_seg * np.stack(
        [-sin_t * Wx - cos_t * Wy, cos_t * Wx - sin_t * Wy, np.zeros(N)],
        axis=1,
    )

    K = (mu0 / (4.0 * np.pi)) * magnetic_moment_magnet()

    d = P - p_magnet[None, :]
    r = np.linalg.norm(d, axis=1)
    r3 = r**3
    r5 = r**5
    r7 = r**7
    sd = d @ sigma_m

    B = K * (3.0 * d * (sd / r5)[:, None] - sigma_m[None, :] / r3[:, None])

    VE = 0.5 * k * float(np.dot(beta, beta))
    VB = -float(np.sum(m * B))

    return {
        "beta": beta,
        "theta": theta,
        "cos_t": cos_t,
        "sin_t": sin_t,
        "m": m,
        "m_prime": m_prime,
        "d": d,
        "r5": r5,
        "r7": r7,
        "sd": sd,
        "B": B,
        "sigma_m": sigma_m,
        "K": K,
        "k": k,
        "l": l,
        "VE": VE,
        "VB": VB,
        "energy": VE + VB,
    }


def total_energy(
    beta: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> float:
    """
    总能量：弹性能 + 磁能（向量化实现，与旧的逐段循环数值一致）。
    beta   : (N,) 各段弯曲角
    p_magnet, sigma_m : 单个永磁体位置与方向
    W     : (N,3) 每段磁矩方向（局部坐标系下）
    k     : 弹性系数
    l     : 分段长度
    m_seg : 每段等效磁矩大小
    """
    return _energy_terms(beta, p_magnet, sigma_m, W, k, l, m_seg)["energy"]


def total_energy_and_grad(
    beta: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> tuple[float, np.ndarray]:
    """
    同时返回总能量及其对 beta 的解析梯度 (energy, grad)。

    梯度推导：
      - 弹性能：dVE/dbeta_k = k * beta_k
      - 磁能 VB = -sum_i m_i·B_i，其中 m_i 依赖 theta_i=cumsum(beta)，
        节点 P_i 依赖 theta_{<=i}，B_i 依赖 P_i。
      令 c_i = m'_i·B_i，w_i = J_B(d_i) m_i（即 d(m_i·B_i)/dP_i），
      u'_j = [-sin theta_j, cos theta_j, 0]，则
        dVB/dbeta_k = -suffix_sum(c)_k - l*suffix_sum(S)_k,
        S_j = u'_j·(sum_{i>=j} w_i)。
    可用 scipy.optimize.check_grad 对该梯度做数值校验。
    """
    t = _energy_terms(beta, p_magnet, sigma_m, W, k, l, m_seg)
    beta = t["beta"]
    N = beta.size
    if N == 0:
        return t["energy"], np.zeros(0, dtype=np.float64)

    d = t["d"]
    m = t["m"]
    B = t["B"]
    sigma_m = t["sigma_m"]
    K = t["K"]
    r5 = t["r5"]
    r7 = t["r7"]
    sd = t["sd"]

    # 磁矩朝向对 beta 的直接贡献
    c = np.sum(t["m_prime"] * B, axis=1)  # (N,)

    # w_i = J_B(d_i) m_i，J_B 为偶极子场对位置的雅可比
    sm = m @ sigma_m
    dm = np.sum(d * m, axis=1)
    w = K * (
        3.0 * (sd[:, None] * m + sm[:, None] * d + dm[:, None] * sigma_m[None, :]) / r5[:, None]
        - 15.0 * (sd * dm)[:, None] * d / r7[:, None]
    )

    uprime = np.stack([-t["sin_t"], t["cos_t"], np.zeros(N)], axis=1)
    w_suffix = np.cumsum(w[::-1], axis=0)[::-1]  # sum_{i>=j} w_i
    S = np.sum(uprime * w_suffix, axis=1)  # (N,)

    suffix_c = np.cumsum(c[::-1])[::-1]
    suffix_S = np.cumsum(S[::-1])[::-1]

    grad = t["k"] * beta - suffix_c - t["l"] * suffix_S
    return t["energy"], grad


def optimize_for_radius(
    p_magnets: np.ndarray,
    sigma_m_list: list[np.ndarray],
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> dict:
    """
    针对给定半径的一条永磁体路径，遍历所有永磁体位置和 sigma_m 朝向，
    返回与原脚本结构兼容的结果字典：
        {sigma_idx: {'theta_all': (M,N), 'end_positions': (M,3), 'sigma_m': ...}}
    """
    N = W.shape[0]
    num_sigma = len(sigma_m_list)
    M = len(p_magnets)

    radius_results: dict[int, dict] = {}
    for idx in range(num_sigma):
        radius_results[idx] = {
            "theta_all": np.zeros((M, N)),
            "end_positions": np.zeros((M, 3)),
            "sigma_m": sigma_m_list[idx],
        }

    for i, p_m in enumerate(p_magnets):
        p_magnet = np.array([p_m[0], p_m[1], 0.0])

        for sigma_idx, sigma_m in enumerate(sigma_m_list):
            if i > 0:
                beta0 = radius_results[sigma_idx]["theta_all"][i - 1]
            else:
                beta0 = np.zeros(N)

            res = minimize(
                lambda b: total_energy_and_grad(b, p_magnet, sigma_m, W, k, l, m_seg),
                beta0,
                method="L-BFGS-B",
                jac=True,
                bounds=[(-np.pi, np.pi)] * N,
                options={"maxiter": 3000, "ftol": 1e-9},
            )

            radius_results[sigma_idx]["theta_all"][i] = res.x
            positions = forward_positions(res.x, l)
            radius_results[sigma_idx]["end_positions"][i] = positions[-1]

    return radius_results

