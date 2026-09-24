# -*- coding: utf-8 -*-
"""3.76 综合指数 K/U/V 与聚类 Bootstrap(表 3-7-1/3-7-2,图 3-5/3-6-1/3-6-2)。

K = z(正确率),U = z(主观总分),V = −(z(ln扫视)+z(ln路径))/2,G = (K+U+V)/3(z 在 A–H 内标准化)。
Z 个体锚定版:先减去该被试的 Z(锚点)记录,再标准化。Bootstrap 按被试整块、组内分层重抽样。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib.plotting import design_color, style_axes, save_fig

B, SEED = 300, 2026  # demo run; raise for a final research analysis
COMP_COLS = ["正确率", "主观_总分", "扫视log", "路径log"]


def anchor_z(df, ah):
    """Z 个体锚定:每条 A–H 记录减去该被试的 Z 记录值。"""
    z = df[df["design"] == config.ANCHOR][["subj"] + COMP_COLS].groupby("subj").mean()
    out = ah.copy()
    for c in COMP_COLS:
        out[c] = out[c] - out["subj"].map(z[c])
    return out


def composite(ah):
    """在 A–H 记录上算 K/U/V/G 与 Ḡ_d。"""
    d = ah.copy()
    for c in COMP_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["K"] = (d["正确率"] - d["正确率"].mean()) / d["正确率"].std(ddof=1)
    d["U"] = (d["主观_总分"] - d["主观_总分"].mean()) / d["主观_总分"].std(ddof=1)
    zs = (d["扫视log"] - d["扫视log"].mean()) / d["扫视log"].std(ddof=1)
    zp = (d["路径log"] - d["路径log"].mean()) / d["路径log"].std(ddof=1)
    d["V"] = -(zs + zp) / 2
    d["G"] = (d["K"] + d["U"] + d["V"]) / 3
    return d


def gbar_by_design(df, anchored=False):
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    if anchored:
        ah = anchor_z(df, ah)
    d = composite(ah)
    return d.groupby("design")["G"].mean()


def bootstrap_gbar(df, anchored=False, B=B, seed=SEED):
    """组内分层、按被试整块重抽样,返回 Ḡ_d 的抽样分布与排名。"""
    rng = np.random.default_rng(seed)
    groups = sorted(df["group"].unique())
    designs = config.DESIGNS
    # 预计算 组→被试 与 被试→行索引,避免循环内 pandas 布尔过滤
    subj_of_group = {}
    idx_of_subj = {}
    for g in groups:
        subjs = df.loc[df["group"] == g, "subj"].unique()
        subj_of_group[g] = subjs
        for s in subjs:
            idx_of_subj[s] = np.where(df["subj"].values == s)[0]
    gbar_draws = {d: np.empty(B) for d in designs}
    rank_draws = np.empty((B, len(designs)))
    for b in range(B):
        idxs = []
        for g in groups:
            subjs = subj_of_group[g]
            pick = rng.integers(0, len(subjs), size=len(subjs))
            for i in pick:
                idxs.append(idx_of_subj[subjs[i]])
        boot = df.iloc[np.concatenate(idxs)]
        gb = gbar_by_design(boot, anchored=anchored)
        for d in designs:
            gbar_draws[d][b] = gb.get(d, np.nan)
        # 排名(1 = G 最大)
        vals = np.array([gbar_draws[d][b] for d in designs])
        rank_draws[b] = len(designs) - np.array(pd.Series(vals).rank(method="min")) + 1
    return gbar_draws, rank_draws


def summarize(gbar_draws, rank_draws, designs, point_estimates):
    rows = []
    for di, d in enumerate(designs):
        draws = gbar_draws[d]
        rows.append(dict(设计=d,
                         G=round(float(point_estimates[d]), 3),
                         ci_low=round(float(np.percentile(draws, 2.5)), 3),
                         ci_high=round(float(np.percentile(draws, 97.5)), 3),
                         P排名1=round(float(np.mean(rank_draws[:, di] == 1)), 3),
                         P前三=round(float(np.mean(rank_draws[:, di] <= 3)), 3)))
    return pd.DataFrame(rows)


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("3.76 综合指数 Ḡ_d(主方案)与 Z 个体锚定")
    print("=" * 72)

    # 主方案点估计
    gb_main = gbar_by_design(df, anchored=False).reindex(config.DESIGNS)
    print("\n主方案 Ḡ_d:")
    for d in config.DESIGNS:
        print(f"  {d}: G={gb_main[d]:.3f}")

    # Bootstrap(主方案 + Z 锚定)
    gb_draws_main, rank_main = bootstrap_gbar(df, anchored=False)
    gb_draws_z, rank_z = bootstrap_gbar(df, anchored=True)

    tab_main = summarize(gb_draws_main, rank_main, config.DESIGNS, gb_main)
    tab_z = summarize(gb_draws_z, rank_z, config.DESIGNS,
                      gbar_by_design(df, anchored=True))
    print("\n主方案排名概率(P排名1 / P前三):")
    print(tab_main[["设计", "G", "P排名1", "P前三"]].to_string(index=False))
    print("\nZ 锚定排名概率:")
    print(tab_z[["设计", "G", "P排名1", "P前三"]].to_string(index=False))

    tab_main.to_csv(config.TABLE_DIR / "表3-7-1_综合指数主方案.csv", index=False, encoding="utf-8-sig")
    tab_z.to_csv(config.TABLE_DIR / "表3-7-2_综合指数Z锚定.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-5:领域构成 + 点估计排序 ----
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    comp = composite(ah)
    order = comp.groupby("design")["G"].mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    kbar = [comp[comp["design"] == d]["K"].mean() for d in order]
    ubar = [comp[comp["design"] == d]["U"].mean() for d in order]
    vbar = [comp[comp["design"] == d]["V"].mean() for d in order]
    gbar = [comp[comp["design"] == d]["G"].mean() for d in order]
    ax.bar(x, kbar, label="K 知识", color="#4C72B0", width=0.62, zorder=3)
    ax.bar(x, ubar, bottom=kbar, label="U 主观", color="#55A868", width=0.62, zorder=3)
    ax.bar(x, vbar, bottom=np.array(kbar) + np.array(ubar), label="V 视觉效率",
           color="#C44E52", width=0.62, zorder=3)
    ax.plot(x, gbar, "ko--", label="G 综合", markersize=5, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.axhline(0, color="#999999", linewidth=0.8)
    style_axes(ax, title="综合指数的领域构成与点估计排序(按 G 降序)",
               xlabel="设计", ylabel="标准化得分")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_fig(fig, "图3-5_综合指数领域构成.png")

    # ---- 图 3-6-1:Z 基线校正后综合得分 + 95% CI ----
    z_gbar = gbar_by_design(df, anchored=True).reindex(config.DESIGNS)
    order_z = z_gbar.sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, d in enumerate(order_z):
        m = tab_z.loc[tab_z["设计"] == d, "G"].iloc[0]
        lo = tab_z.loc[tab_z["设计"] == d, "ci_low"].iloc[0]
        hi = tab_z.loc[tab_z["设计"] == d, "ci_high"].iloc[0]
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=design_color(d),
                    markersize=7, capsize=4, elinewidth=1.4, zorder=3)
    ax.set_xticks(range(len(order_z))); ax.set_xticklabels(order_z)
    ax.axhline(0, color="#999999", linewidth=0.8)
    style_axes(ax, title="Z 基线校正后的综合得分及 95% 聚类 Bootstrap 置信区间",
               xlabel="设计", ylabel="综合得分 G")
    fig.tight_layout()
    save_fig(fig, "图3-6-1_Z校正综合得分.png")

    # ---- 图 3-6-2:Bootstrap 排名概率 ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order_r = tab_z.sort_values("P前三", ascending=False)["设计"].tolist()
    for i, d in enumerate(order_r):
        p3 = tab_z.loc[tab_z["设计"] == d, "P前三"].iloc[0]
        p1 = tab_z.loc[tab_z["设计"] == d, "P排名1"].iloc[0]
        ax.bar(i, p3, color=design_color(d), alpha=0.35, width=0.62, zorder=3, label="P前三" if i == 0 else None)
        ax.bar(i, p1, color=design_color(d), width=0.62, zorder=4, label="P排名1" if i == 0 else None)
    ax.set_xticks(range(len(order_r))); ax.set_xticklabels(order_r)
    style_axes(ax, title="Z 校正后 Bootstrap 排名概率(浅色=P前三,深色=P排名1)",
               xlabel="设计", ylabel="概率")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_fig(fig, "图3-6-2_排名概率.png")

    print("\n已保存 → 表3-7-1/3-7-2 + 图3-5/3-6-1/3-6-2")


if __name__ == "__main__":
    main()
