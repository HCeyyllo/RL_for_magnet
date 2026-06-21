import sys
from pathlib import Path

# 将项目根目录加入路径，以便能 import core
try:
    _root = Path(__file__).resolve().parent.parent
except NameError:
    _root = Path.cwd()

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter
from matplotlib.patches import Polygon

from core.kinematics import forward_positions

def plot_workspace(
    results_shape: dict,
    results_all_r: dict[float, dict],
    W: np.ndarray,
    shape_indices_to_draw: list[int],
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    save_path: str | None = None,
):
    """
    绘制工作空间散点与部分构型。
    参数基本与原脚本保持一致，只是 results_* 来自 core.energy.optimize_for_radius。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    all_end_positions = []
    for sigma_idx in range(len(results_shape)):
        all_end_positions.append(results_shape[sigma_idx]["end_positions"])
    for _, res_r in results_all_r.items():
        for sigma_idx in range(len(res_r)):
            all_end_positions.append(res_r[sigma_idx]["end_positions"])
    all_end_positions = np.vstack(all_end_positions)

    all_end_positions_symmetric = all_end_positions.copy()
    all_end_positions_symmetric[:, 1] = -all_end_positions_symmetric[:, 1]
    all_end_positions_combined = np.vstack(
        [all_end_positions, all_end_positions_symmetric]
    )

    ax.plot(
        all_end_positions_combined[:, 0] * 1000,
        all_end_positions_combined[:, 1] * 1000,
        "o",
        color="gray",
        markersize=4,
        alpha=0.6,
        label="末端位置",
    )

    p_magnets_shape_len = next(iter(results_shape.values()))["theta_all"].shape[0]
    indices_to_draw = [
        int(p_magnets_shape_len * 30 / 90) - 1,
        int(p_magnets_shape_len * 60 / 90) - 1,
        int(p_magnets_shape_len * 90 / 90) - 1,
    ]

    N = W.shape[0]
    for sigma_idx in shape_indices_to_draw:
        theta_all = results_shape[sigma_idx]["theta_all"]
        for idx in indices_to_draw:
            theta_opt = theta_all[idx]
            positions = forward_positions(theta_opt, l=theta_opt.size and 1.0)
            for i in range(N):
                p0 = positions[i][:2] * 1000
                p1 = positions[i + 1][:2] * 1000
                if np.all(W[i] == 0):
                    lw_color = "gray"
                elif np.all(W[i] == [1, 0, 0]):
                    lw_color = "red"
                elif np.all(W[i] == [-1, 0, 0]):
                    lw_color = "dodgerblue"
                else:
                    lw_color = "black"
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=lw_color,
                    linewidth=2.5,
                    linestyle="-",
                    alpha=0.7,
                )

    ax.plot(0, 0, "ko", markersize=6, label="基点")

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="NdFeB段"),
        Line2D([0], [0], color="dodgerblue", lw=2, label="ALNiCo段"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.002, 0.002))

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xlabel("X (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (mm)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.show()


def plot_workspace_single_ratio(
    end_positions: np.ndarray,
    ratio: float,
    save_dir: str = "workspace_imgs",
    xlim: tuple[float, float] = (-40, 40),
    ylim: tuple[float, float] = (-50, 50),
) -> None:
    """
    绘制单个正向磁矩占比下的工作空间散点图（y 轴对称），并保存。
    end_positions: (M, 3) 末端位置，单位 m
    ratio: 正向磁矩占比（用于标题与文件名）
    """
    import os

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(save_dir, exist_ok=True)

    pts_sym = end_positions.copy()
    pts_sym[:, 1] *= -1
    pts_all = np.vstack([end_positions, pts_sym])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.plot(
        pts_all[:, 0] * 1000,
        pts_all[:, 1] * 1000,
        "bo",
        markersize=4,
        alpha=0.6,
        label="末端位置",
    )
    ax.plot(0, 0, "ko", markersize=6, label="基点")
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="NdFeB段"),
        Line2D([0], [0], color="dodgerblue", lw=2, label="ALNiCo段"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.002, 0.002))
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xlabel("X (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (mm)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"磁连续体工作空间（正向磁矩占比 {int(ratio * 100)}%）",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    filename = f"workspace_ratio_{int(ratio * 100)}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=600)
    plt.close(fig)


def plot_ratio_vs_area(
    ratios: np.ndarray,
    areas_mm2: np.ndarray,
    save_path: str | None = "ratio_vs_workspace_area.png",
    title: str | None = None,
    figsize: tuple[float, float] = (3.45, 2.55),
    dpi: int = 600,
) -> None:
    """
    绘制正向磁矩占比–工作空间面积关系图，风格参考 Science Robotics。
    
    参数
    ----
    ratios : np.ndarray
        正向磁矩占比数组，范围 0~1
    areas_mm2 : np.ndarray
        对应工作空间面积，单位 mm^2
    save_path : str | None
        输出路径；若不为 None，则保存 PNG/PDF/SVG
    title : str | None
        图标题；Science Robotics 风格通常单图不在图内加标题，默认 None
    figsize : tuple
        图尺寸（inch），默认适合单栏
    dpi : int
        位图导出分辨率
    """
    ratios = np.asarray(ratios, dtype=float).reshape(-1)
    areas_mm2 = np.asarray(areas_mm2, dtype=float).reshape(-1)

    if ratios.size != areas_mm2.size:
        raise ValueError("ratios 与 areas_mm2 长度必须一致")
    if ratios.size == 0:
        raise ValueError("输入数据为空")

    # 按 ratio 升序，避免折线回跳
    order = np.argsort(ratios)
    ratios = ratios[order]
    areas_mm2 = areas_mm2[order]

    # 转百分比显示更符合“占比”表达
    ratios_percent = ratios * 100.0
    imax = int(np.argmax(areas_mm2))

    # Science Robotics / 工程类顶刊常用：克制、白底、细线、矢量字体
    sr_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "SimHei", "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,

        "axes.linewidth": 0.9,
        "axes.edgecolor": "#222222",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.grid": False,
        "axes.axisbelow": True,

        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,

        "legend.frameon": False,
        "lines.antialiased": True,
    }

    with plt.rc_context(sr_rc):
        fig, ax = plt.subplots(figsize=figsize)

        # 主色：深蓝；高亮：暗红
        line_color = "#1f4e79"
        accent_color = "#c1121f"
        marker_edge = "#ffffff"
        grid_color = "#d9d9d9"

        # 淡横网格，SR 风格里可有可无，这里保留很淡的 y-grid
        ax.yaxis.grid(True, color=grid_color, linewidth=0.55, alpha=0.55)
        ax.xaxis.grid(False)

        # 主折线
        ax.plot(
            ratios_percent,
            areas_mm2,
            color=line_color,
            lw=2.1,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )

        # 常规点
        ax.scatter(
            ratios_percent,
            areas_mm2,
            s=24,
            facecolor=line_color,
            edgecolor=marker_edge,
            linewidth=0.5,
            zorder=3,
        )

        # 最大面积点：星形高亮
        ax.scatter(
            ratios_percent[imax],
            areas_mm2[imax],
            s=130,
            marker="*",
            facecolor=accent_color,
            edgecolor="#111111",
            linewidth=0.55,
            zorder=5,
            label="Maximum area",
        )

        # 标注最大值
        ax.annotate(
            f"Max = {areas_mm2[imax]:.2f} mm$^2$\nα = {ratios[imax]:.2f}%",
            xy=(ratios_percent[imax], areas_mm2[imax]),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=accent_color,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.9,
                color=accent_color,
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=6,
        )

        ax.set_xlabel("Forward magnetic moment ratio (%)")
        ax.set_ylabel("Workspace area (mm$^2$)")

        if title:
            ax.set_title(title, pad=4)

        # x 轴刻度更工程化
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

        ax.tick_params(axis="both", which="major", pad=2)
        ax.tick_params(axis="both", which="minor", pad=2)

        ax.margins(x=0.03, y=0.08)

        ax.legend(
            loc="best",
            frameon=False,
            handlelength=1.2,
            borderpad=0.2,
            labelspacing=0.25,
        )

        fig.tight_layout(pad=0.4)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(save_path, dpi=dpi, facecolor="white", edgecolor="none")
            fig.savefig(save_path.with_suffix(".pdf"), facecolor="white", edgecolor="none")
            fig.savefig(save_path.with_suffix(".svg"), facecolor="white", edgecolor="none")

    backend = mpl.get_backend().lower()
    if backend not in ("agg", "cairo", "pdf", "svg", "ps", "template"):
        plt.show()
    plt.close(fig)


def plot_deformation_single_ratio(
    positions: np.ndarray,
    W: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    Lm: float,
    Dm: float,
    ratio: float,
    save_path: str | None = None,
) -> None:
    """
    与 变形表征/1226_永磁体变形.py 中绘图效果完全一致：
    永磁体作用下磁连续体机器人变形模式单图。
    仅通过 ratio 区分标题与文件名，其余绘图形式不变。
    positions: (N+1, 3) 段节点坐标，单位 m
    W: (N, 3) 每段磁矩方向
    p_magnet: (3,) 永磁体中心位置 (m)
    sigma_m: (3,) 永磁体方向
    Lm, Dm: 永磁体长度、直径 (m)
    ratio: 正向磁矩占比 (0~1)，用于标题与文件名
    save_path: 保存路径，若为 None 则仅显示不保存
    """
    N = W.shape[0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # 绘制机器人构型（与 1226 完全一致）
    for i in range(N):
        p0 = positions[i][:2] * 1000  # 转换为 mm
        p1 = positions[i + 1][:2] * 1000

        if np.all(W[i] == 0):
            color = "gray"
            lw = 2.5
        elif np.all(W[i] == [1, 0, 0]):
            color = "red"
            lw = 2.5
        elif np.all(W[i] == [-1, 0, 0]):
            color = "dodgerblue"
            lw = 2.5
        elif np.all(W[i] == [0, -1, 0]):
            color = "green"
            lw = 2.5
        elif np.all(W[i] == [0, 1, 0]):
            color = "orange"
            lw = 2.5
        else:
            color = "black"
            lw = 2.5

        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw)

    # 永磁体中心位置（mm）
    center = p_magnet[:2] * 1000
    sigma = sigma_m[:2] / np.linalg.norm(sigma_m[:2])
    perp = np.array([-sigma[1], sigma[0]])
    magnet_length = Lm * 1000
    magnet_width = Dm * 1000
    half_len = magnet_length / 2
    half_wid = magnet_width / 2
    p1 = center - half_len * sigma + half_wid * perp
    p2 = center + half_len * sigma + half_wid * perp
    p3 = center + half_len * sigma - half_wid * perp
    p4 = center - half_len * sigma - half_wid * perp
    upper = Polygon(
        [p3, p2, center + half_wid * perp, center - half_wid * perp],
        closed=True,
        facecolor="blue",
        edgecolor="black",
        linewidth=1,
    )
    lower = Polygon(
        [center - half_wid * perp, center + half_wid * perp, p1, p4],
        closed=True,
        facecolor="red",
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(upper)
    ax.add_patch(lower)

    ax.plot(
        positions[0, 0] * 1000,
        positions[0, 1] * 1000,
        "ko",
        markersize=6,
        label="基点",
    )
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="NdFeB段"),
        Line2D([0], [0], color="dodgerblue", lw=2, label="ALNiCo段"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.002, 0.002))

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xlabel("X (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Y (mm)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"永磁体作用下磁连续体机器人变形模式（正向磁矩占比 {int(ratio * 100)}%）",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlim(-20, 40)
    ax.set_ylim(-10, 50)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)