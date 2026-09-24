# -*- coding: utf-8 -*-
"""3.87 敏感性分析:不同综合指数设定下的排名 + Spearman + 探索性因素趋势(表 3-8,图 3-7/3-8)。

8 种设定:主方案、Block-z、Z 锚定、三种权重扰动(K/U/V=0.5)、仅K、仅U。
因素趋势按主方案 / Block-z / Z 锚定三种校正口径,给出各水平等权平均 G 与极差。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib.stats_helpers import spearman_rank
from lib.plotting import design_color, style_axes, save_fig

COMP_COLS = ["正确率", "主观_总分", "扫视log", "路径log"]


def _zstd(s):
    return (s - s.mean()) / s.std(ddof=1)


def anchor_z(df, ah):
    z = df[df["design"] == config.ANCHOR][["subj"] + COMP_COLS].groupby("subj").mean()
    out = ah.copy()
    for c in COMP_COLS:
        out[c] = out[c] - out["subj"].map(z[c])
    return out


def composite_spec(df, wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="global"):
    """按指定权重与标准化方式计算 G。z_mode ∈ {global, block, z}。"""
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    for c in COMP_COLS:
        ah[c] = pd.to_numeric(ah[c], errors="coerce")
    if z_mode == "z":
        ah = anchor_z(df, ah)
    if z_mode == "block":
        K = ah.groupby("content")["正确率"].transform(_zstd)
        U = ah.groupby("content")["主观_总分"].transform(_zstd)
        zs = ah.groupby("content")["扫视log"].transform(_zstd)
        zp = ah.groupby("content")["路径log"].transform(_zstd)
    else:
        K = _zstd(ah["正确率"]); U = _zstd(ah["主观_总分"])
        zs = _zstd(ah["扫视log"]); zp = _zstd(ah["路径log"])
    V = -(zs + zp) / 2
    ah["G"] = wk * K + wu * U + wv * V
    return ah.groupby("design")["G"].mean()


def factor_level(d, factor):
    p = config.DESIGN_AREA_PCT[d]
    if factor == "标题层":
        return "高" if p["标题层"] > 10 else "低"
    if factor == "要点层":
        return "高" if p["要点层"] > 10 else "低"
    if factor == "详情层":
        return {34: "高", 17: "中", 3: "低"}[int(p["详情层"])]
    if factor == "关键图片":
        return "大" if p["关键图片"] > 30 else "小"


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 论文 2.6.6 定义的 6 种方案:主方案 + Block-z + Z锚定 + 三种权重扰动(K/U/V 升至 0.50)
    specs = [
        ("主方案", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="global")),
        ("Block-z", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="block")),
        ("Z锚定", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="z")),
        ("K×0.5", dict(wk=0.5, wu=0.25, wv=0.25, z_mode="global")),
        ("U×0.5", dict(wk=0.25, wu=0.5, wv=0.25, z_mode="global")),
        ("V×0.5", dict(wk=0.25, wu=0.25, wv=0.5, z_mode="global")),
    ]

    gbars = {}
    for name, kw in specs:
        gbars[name] = composite_spec(df, **kw).reindex(config.DESIGNS)

    # 排名(1 = 最高)
    rank_df = pd.DataFrame({name: gbars[name].rank(ascending=False)
                            for name, _ in specs}).astype(int)

    # Spearman 与主方案
    print("=" * 70)
    print("3.87 敏感性:各设定排名与 Spearman ρ(对照主方案)")
    print("=" * 70)
    main_rank = rank_df["主方案"].values
    spearman_rows = []
    for name, _ in specs:
        if name == "主方案":
            rho = 1.0
        else:
            rho = spearman_rank(rank_df[name].values, main_rank)["rho"]
        spearman_rows.append(dict(设定=name, Spearman_rho=round(float(rho), 3)))
    spearman_df = pd.DataFrame(spearman_rows)
    print(spearman_df.to_string(index=False))

    print("\n各设定排名(1=最优):")
    print(rank_df.to_string())

    # 保存排名与 Spearman(图 3-7 配套)
    rank_df.to_csv(config.TABLE_DIR / "图3-7_各设定排名.csv", encoding="utf-8-sig")
    spearman_df.to_csv(config.TABLE_DIR / "图3-7_Spearman.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-7 排名热图 ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rmat = rank_df.T.values  # (方案 × 设计)
    cmap = plt.cm.RdYlGn.reversed()  # 绿=靠前(小值),红=靠后(大值)
    im = ax.imshow(rmat, cmap=cmap, norm=Normalize(vmin=1, vmax=8), aspect="auto")
    ax.set_xticks(range(len(config.DESIGNS))); ax.set_xticklabels(config.DESIGNS)
    ax.set_yticks(range(len(specs))); ax.set_yticklabels([s[0] for s in specs])
    for i in range(len(specs)):
        for j in range(len(config.DESIGNS)):
            ax.text(j, i, str(rmat[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title("不同综合指数设定下的设计排名(1=最优)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.9, ticks=range(1, 9), label="排名")
    fig.tight_layout()
    save_fig(fig, "图3-7_各设定排名.png")

    # ---- 因素趋势(3 种校正口径) ----
    factors = ["标题层", "要点层", "详情层", "关键图片"]
    schemes = [("主方案", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="global")),
               ("Block-z", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="block")),
               ("Z锚定", dict(wk=1 / 3, wu=1 / 3, wv=1 / 3, z_mode="z"))]

    trend_rows = []
    for name, kw in schemes:
        gb = composite_spec(df, **kw).reindex(config.DESIGNS)
        for f in factors:
            lvl = {d: factor_level(d, f) for d in config.DESIGNS}
            s = pd.Series({d: gb[d] for d in config.DESIGNS})
            means = s.groupby(lvl).mean()
            best = means.idxmax()
            rng = means.max() - means.min()
            trend_rows.append(dict(方案=name, 因素=f, 最优水平=best,
                                   极差=round(float(rng), 3)))
    trend_df = pd.DataFrame(trend_rows)
    print("\n因素趋势(最优水平 / 极差):")
    print(trend_df.to_string(index=False))
    trend_df.to_csv(config.TABLE_DIR / "表3-8_因素趋势.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-8 因素趋势 ----
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, f in zip(axes, factors):
        lvl = {d: factor_level(d, f) for d in config.DESIGNS}
        for name, kw in schemes:
            gb = composite_spec(df, **kw).reindex(config.DESIGNS)
            s = pd.Series({d: gb[d] for d in config.DESIGNS})
            means = s.groupby(lvl).mean()
            # 固定水平顺序
            order = ["低", "中", "高"] if f == "详情层" else ["低", "高"] if f != "关键图片" else ["小", "大"]
            xs = [order.index(v) for v in means.index if v in order]
            ys = [means[v] for v in means.index if v in order]
            ax.plot(xs, ys, "o-", markersize=5, label=name)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
        style_axes(ax, title=f, xlabel="水平", ylabel="平均 G")
        if f == "标题层":
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("三种校正方案下的探索性版面因素趋势", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, "图3-8_因素趋势.png")

    print("\n已保存 → 图3-7_各设定排名 / 图3-7_Spearman / 表3-8_因素趋势 + 图3-7/3-8")


if __name__ == "__main__":
    main()
