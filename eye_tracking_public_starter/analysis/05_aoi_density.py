# -*- coding: utf-8 -*-
"""3.65 AOI 单位面积注意密度:密度比 = 注视时长占比 ÷ 版面面积占比(图 3-4)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib.plotting import save_fig


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 各设计×层 平均注视时长占比(%)
    fix_cols = {l: f"注视时长占比_{l}" for l in config.LAYERS}
    rows = []
    dens = np.full((len(config.LAYERS), len(config.DESIGNS_ALL)), np.nan)
    for li, layer in enumerate(config.LAYERS):
        for di, d in enumerate(config.DESIGNS_ALL):
            sub = df[df["design"] == d]
            fix_pct = sub[fix_cols[layer]].astype(float).mean()
            area = config.DESIGN_AREA_PCT[d][layer]
            ratio = fix_pct / area if area > 0 else np.nan
            dens[li, di] = ratio
            rows.append(dict(设计=d, 内容=config.DESIGN_CONTENT[d], 层级=layer,
                             注视时长占比=round(fix_pct, 2), 面积占比=area,
                             密度比=round(ratio, 3)))

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / "图3-4_AOI注意密度.csv", index=False, encoding="utf-8-sig")

    # 打印示例值
    print("=" * 66)
    print("3.65 AOI 单位面积注意密度比(注视时长占比 ÷ 面积占比)")
    print("=" * 66)
    pivot = out.pivot(index="层级", columns="设计", values="密度比")
    pivot = pivot.reindex(columns=config.DESIGNS_ALL)
    print(pivot.round(2).to_string())
    print("\n模拟示例: C 要点层 = %.2f;" % pivot.loc["要点层", "C"])
    print("           G 详情层 = %.2f; H 详情层 = %.2f"
          % (pivot.loc["详情层", "G"], pivot.loc["详情层", "H"]))

    # ---- 图 3-4 热图(发散,中心 1) ----
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=9)
    im = ax.imshow(dens, cmap="RdYlGn_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(config.DESIGNS_ALL)))
    ax.set_xticklabels(config.DESIGNS_ALL)
    ax.set_yticks(range(len(config.LAYERS)))
    ax.set_yticklabels(config.LAYERS)
    for li in range(len(config.LAYERS)):
        for di in range(len(config.DESIGNS_ALL)):
            v = dens[li, di]
            if np.isfinite(v):
                ax.text(di, li, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v - 1) > 4 else "black")
    ax.set_title("AOI 单位面积注意密度比(1 = 注意份额与面积份额相当)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.9, label="密度比")
    fig.tight_layout()
    save_fig(fig, "图3-4_AOI注意密度.png")

    print("\n已保存 → 图3-4_AOI注意密度.csv / 图3-4_AOI注意密度.png")


if __name__ == "__main__":
    main()
