# 眼动实验分析流程：示例数据及代码

`data/synthetic_master.csv` 的 244 行是独立生成的虚构数据。`data/material_measurements.xlsx` 是真实科普牌材料测量表。模拟眼动数据的任何排名、置信区间、p 值及图表均不能当作手稿的实证结果。

## 文件与运行

| 路径 | 说明 |
| --- | --- |
| `generate_demo.py`、`validate_demo.py` | 生成并校验 61 人 × 4 次观看、52 列的虚构主表 |
| `analysis/00_...py` | 对真实材料测量表做分组编码 |
| `analysis/01_...py`～`07_...py` | 从模拟主表计算质量、描述、相关、模型、AOI、综合指数及敏感性结果 |
| `tools/measure_selection_area.jsx` | Photoshop 选区面积测量脚本；原照片未包含 |
| `outputs_synthetic/表格/` | 试跑所得示例 CSV，`00` 的输出例外，来自真实材料测量表 |

在此文件夹运行：

```bash
python -m pip install -r analysis/requirements.txt
python run_demo.py
```

依赖尚未安装齐全时，`run_demo.py` 会跳过 `04_mixed_models.py`，其余脚本仍会尝试运行；安装 `statsmodels` 后可重跑完整流程。生成图表需本机有支持中文的字体；缺少字体的系统可能导出缺字的 PNG。所有输出放在 `outputs_synthetic/`，不覆盖原复现包中的真实结果。


## 模拟规则与公开边界

沿用实验的分组设计（21/20/20 人和每组的版面组合），随机打乱每人的观看顺序；虚构编号 `SIM001` 起。每名示例被试获得独立随机偏移；两道问答各计 0 或 5 分，答对题数与正确率由此计算；7 个主观题各计 1～5 分，重新计算总分和均值；眼动区域次数和时长随机生成，注视时长占比为占**全部注视时长**的百分数，因此四个 AOI 相加可以小于 100%。生理及扫视字段也是独立生成的测试数值。
