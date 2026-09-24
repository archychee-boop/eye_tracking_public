# -*- coding: utf-8 -*-
"""3.2 Cronbach α + 3.3 指标分布 + 3.54 各设计均值(表 3-2/3-3,图 3-1)。"""
import sys
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib.stats_helpers import cronbach_alpha, clustered_bootstrap_stat
from lib.plotting import (design_color, style_axes, save_fig, DESIGN_COLORS)

ITEMS7 = ["获取信息效率_Q1", "获取信息效率_Q2", "可读性_Q1", "可读性_Q2",
          "吸引力_Q1", "吸引力_Q2", "主观_综合评价"]

def rnd(x, nd: int):
    """四舍五入(ROUND_HALF_UP),匹配论文排版口径(避免 Python 银行家舍入)。"""
    if pd.isna(x):
        return x
    q = Decimal("1") if nd <= 0 else Decimal("0." + "0" * (nd - 1) + "1")
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


INDICATORS = {
    "正确率": "正确率",
    "主观总分": "主观_总分",
    "AOI总注视时长": "AOI注视",
    "AOI注视次数": "AOI注视次数",
    "扫视次数": "扫视次数",
    "总扫视路径": "路径",
    "平均瞳孔直径": "平均瞳孔直径_mm",
    "最小瞳孔直径": "最小瞳孔直径_mm",
    "眨眼次数": "眨眼次数",
}


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 3.2 Cronbach α ----
    a7 = cronbach_alpha(df[ITEMS7].astype(float))
    a6 = cronbach_alpha(df[[c for c in ITEMS7 if c != "主观_综合评价"]].astype(float))
    ah = df[df["design"].isin(config.DESIGNS)]
    a7_ah = cronbach_alpha(ah[ITEMS7].astype(float))
    print("=" * 60)
    print("3.2 Cronbach α")
    print(f"  7 题(全部): {a7:.3f}")
    print(f"  6 题(全部): {a6:.3f}")
    print(f"  7 题(A–H): {a7_ah:.3f}")

    pd.DataFrame([
        dict(口径="7题_全部244", alpha=round(a7, 3)),
        dict(口径="6题_全部244", alpha=round(a6, 3)),
        dict(口径="7题_AH183", alpha=round(a7_ah, 3)),
    ]).to_csv(config.TABLE_DIR / "表3-2_Cronbach_alpha.csv", index=False, encoding="utf-8-sig")

    # ---- 3.3 指标分布(A–H 183 条,表 3-2) ----
    print("\n3.3 指标分布(A–H 183 条:n/均值/SD/中位数/偏度)")
    ah = df[df["design"].isin(config.DESIGNS)]
    rows = []
    for label, col in INDICATORS.items():
        v = pd.to_numeric(ah[col], errors="coerce")
        v = v.dropna()
        rows.append(dict(指标=label, n=len(v), 均值=round(v.mean(), 3),
                         SD=round(v.std(ddof=1), 3), 中位数=round(v.median(), 3),
                         偏度=round(float(v.skew()), 3)))
    dist = pd.DataFrame(rows)
    print(dist.to_string(index=False))
    dist.to_csv(config.TABLE_DIR / "表3-2_指标分布.csv", index=False, encoding="utf-8-sig")

    # ---- 3.54 各设计主要指标原始均值(表 3-3,A–H + Z) ----
    print("\n3.54 各设计主要指标原始均值(表 3-3)")
    dm = df.groupby("design").agg(
        正确率=("正确率", "mean"), 主观总分=("主观_总分", "mean"),
        扫视次数=("扫视次数", "mean"), 路径=("路径", "mean"),
        平均瞳孔=("平均瞳孔直径_mm", "mean"),
    ).reindex(config.DESIGNS_ALL)
    t33 = pd.DataFrame({
        "设计": [f"{d}（{config.DESIGN_LABELS[d]}）" for d in config.DESIGNS_ALL],
        "正确率": [f"{rnd(dm.loc[d, '正确率'] * 100, 1)}%" for d in config.DESIGNS_ALL],
        "主观总分": [rnd(dm.loc[d, "主观总分"], 2) for d in config.DESIGNS_ALL],
        "扫视次数": [rnd(dm.loc[d, "扫视次数"], 1) for d in config.DESIGNS_ALL],
        "路径(kpx)": [rnd(dm.loc[d, "路径"] / 1000, 1) for d in config.DESIGNS_ALL],
        "平均瞳孔(mm)": [rnd(dm.loc[d, "平均瞳孔"], 3) for d in config.DESIGNS_ALL],
    })
    print(t33.to_string(index=False))
    t33.to_csv(config.TABLE_DIR / "表3-3_各设计均值.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-1:四项主要结果 × 8 设计,点 + 95% 聚类 Bootstrap CI ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("各设计四项主要结果的均值与 95% Bootstrap 置信区间", fontsize=12)
    main4 = [("正确率", "正确率"), ("主观总分", "主观_总分"),
             ("扫视次数", "扫视次数"), ("总扫视路径", "路径")]
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    for ax, (label, col) in zip(axes.ravel(), main4):
        xs = range(len(config.DESIGNS))
        means, lows, highs = [], [], []
        for d in config.DESIGNS:
            sub = ah[ah["design"] == d]
            stat = clustered_bootstrap_stat(sub, lambda s: s[col].mean(), col, B=2000)
            means.append(stat["mean"]); lows.append(stat["ci_low"]); highs.append(stat["ci_high"])
        for x, m, lo, hi, d in zip(xs, means, lows, highs, config.DESIGNS):
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=design_color(d),
                        markersize=6, capsize=3, elinewidth=1.2, zorder=3)
        ax.set_xticks(list(xs)); ax.set_xticklabels(config.DESIGNS)
        style_axes(ax, title=label, xlabel="设计", ylabel="均值")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "图3-1_各设计四项主要结果.png")

    print("\n已保存 → 表3-2_Cronbach_alpha.csv / 表3-2_指标分布.csv / 表3-3_各设计均值.csv / 图3-1")


if __name__ == "__main__":
    main()
