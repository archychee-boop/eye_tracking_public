# -*- coding: utf-8 -*-
"""matplotlib 样式与图表保存:中文字体、固定设计配色、统一外观。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

import config


# ---------- 中文字体 ----------
def _setup_font():
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC"]:
        try:
            fm.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


# ---------- 固定设计配色(分类色,固定顺序,不随筛选改变) ----------
DESIGN_COLORS = {
    "A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52",
    "E": "#8172B3", "F": "#937860", "G": "#DA8BC3", "H": "#8C8C8C",
    "Z": "#3B3B3B",
}


def design_color(d: str) -> str:
    return DESIGN_COLORS.get(d, "#999999")


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """统一坐标轴外观:细灰轴线、轻网格、去多余边框。"""
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=9)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, color="#222222", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#444444")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#444444")


def new_fig(figsize=(6.5, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(fig, name: str, dpi: int = 200):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def bar_designs(values: dict, ax, ylabel=None, title=None, anchor=False,
                show_values=True):
    """按设计 A–H(可选 Z)绘制柱状图。values: {design: float}。"""
    ds = list(config.DESIGNS) + ([config.ANCHOR] if anchor else [])
    vals = [values.get(d, np.nan) for d in ds]
    colors = [design_color(d) for d in ds]
    bars = ax.bar(ds, vals, color=colors, width=0.62, zorder=3)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels(ds)
    style_axes(ax, title=title, ylabel=ylabel)
    if show_values:
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color="#333333")
    return ax


_setup_font()
