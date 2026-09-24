# -*- coding: utf-8 -*-
"""3.4 重复测量相关:5 变量 10 对 r_rm + BH-FDR 校正(表 3-x,图 3-2)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib.stats_helpers import rm_corr, fdr_bh
from lib.plotting import save_fig

VARS = [
    ("正确率", "正确率"),
    ("主观总分", "主观_总分"),
    ("扫视log", "扫视log"),
    ("路径log", "路径log"),
    ("瞳孔", "平均瞳孔直径_mm"),
]


def main():
    df = load_master()
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    labels = [v[0] for v in VARS]
    n = len(VARS)
    rmat = np.zeros((n, n))
    pmat = np.zeros((n, n))

    rows = []
    print("=" * 66)
    print("3.4 重复测量相关(A–H 183,df=121)")
    print("=" * 66)
    for i in range(n):
        for j in range(i + 1, n):
            res = rm_corr(ah, VARS[i][1], VARS[j][1], subject="subj")
            rmat[i, j] = rmat[j, i] = res["r"]
            pmat[i, j] = pmat[j, i] = res["p"]
            rows.append(dict(变量1=labels[i], 变量2=labels[j],
                             r_rm=round(res["r"], 3), df=res["df"], p=res["p"]))

    # BH-FDR 校正
    pvals = np.array([r["p"] for r in rows])
    q = fdr_bh(pvals)
    for r, qv in zip(rows, q):
        r["FDR_p"] = round(float(qv), 4)
        r["显著"] = "*" if qv < 0.05 else ""
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(config.TABLE_DIR / "图3-2_重复测量相关.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-2 相关矩阵 ----
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.8)
    im = ax.imshow(rmat, cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            if i < j:
                star = "*" if fdr_bh([pmat[i, j]])[0] < 0.05 else ""
                ax.text(j, i, f"{rmat[i, j]:.2f}{star}", ha="center", va="center",
                        fontsize=10, color="black" if abs(rmat[i, j]) < 0.4 else "white")
            elif i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=10, color="#999")
            else:
                ax.text(j, i, "", ha="center", va="center")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("核心指标的重复测量相关矩阵(* 为 BH-FDR p<0.05)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    save_fig(fig, "图3-2_重复测量相关矩阵.png")

    print("\n已保存 → 图3-2_重复测量相关.csv / 图3-2_重复测量相关矩阵.png")


if __name__ == "__main__":
    main()
