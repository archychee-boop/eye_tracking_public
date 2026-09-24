"""Generate an independent, fictional eye-tracking example table.

No participant-level values, empirical distributions, or fitted parameters are
read from the study workbook. The shared 61 x 4 schedule is used solely to
exercise the same file schema and analysis workflow.
"""
import csv
import math
import random
from pathlib import Path

SEED = 20260923
LAYERS = ("标题层", "要点层", "详情层", "关键图片")
SCHEDULE = {
    "组1": (21, ("A", "C", "F", "Z")),
    "组2": (20, ("B", "E", "G", "Z")),
    "组3": (20, ("A", "D", "H", "Z")),
}
CONTENT = {"A": "造山运动", "B": "造山运动", "C": "红耳鹎",
           "D": "红耳鹎", "E": "红耳鹎", "F": "红果仔",
           "G": "红果仔", "H": "红果仔", "Z": "白鹈鹕(锚点)"}
HEADERS = (
    ["受试编号", "组别", "观看顺序", "设计代号", "内容类型",
     "问答1得分", "问答2得分", "答对题数", "正确率"]
    + [f"{x}_{q}" for x in ("获取信息效率", "可读性", "吸引力")
       for q in ("Q1", "Q2", "均值")]
    + ["主观_综合评价", "主观_总分", "主观_总分均值"]
    + [f"{metric}_{layer}" for metric in
       ("总注视时长", "注视次数", "注视时长占比", "首次注视时间", "平均注视时长")
       for layer in LAYERS]
    + ["平均瞳孔直径_mm", "最小瞳孔直径_mm", "最大瞳孔直径_mm",
       "眨眼次数", "平均眨眼频率_per_s", "扫视次数", "平均扫视频率_per_s",
       "总扫视时间_s", "平均眼跳水平距离_px", "平均眼跳垂直距离_px",
       "平均眼跳绝对距离_px"]
)


def bounded(x, lo, hi):
    return max(lo, min(hi, x))


def make_rows():
    rng = random.Random(SEED)
    rows = []
    person = 0
    for group, (count, designs) in SCHEDULE.items():
        for _ in range(count):
            person += 1
            person_effect = rng.gauss(0, 0.4)
            visit_order = list(designs)
            rng.shuffle(visit_order)
            for order, design in enumerate(visit_order, 1):
                row = {"受试编号": f"SIM{person:03d}", "组别": group,
                       "观看顺序": order, "设计代号": design,
                       "内容类型": CONTENT[design]}
                for q in (1, 2):
                    chance = bounded(0.48 + person_effect * 0.12 + rng.uniform(-0.12, 0.12), .1, .9)
                    row[f"问答{q}得分"] = 5 if rng.random() < chance else 0
                row["答对题数"] = (row["问答1得分"] + row["问答2得分"]) // 5
                row["正确率"] = row["答对题数"] / 2
                ratings = []
                for category in ("获取信息效率", "可读性", "吸引力"):
                    vals = [int(round(bounded(3.3 + person_effect + rng.gauss(0, .95), 1, 5)))
                            for _ in range(2)]
                    row[f"{category}_Q1"], row[f"{category}_Q2"] = vals
                    row[f"{category}_均值"] = sum(vals) / 2
                    ratings.extend(vals)
                row["主观_综合评价"] = int(round(bounded(3.3 + person_effect + rng.gauss(0, .95), 1, 5)))
                row["主观_总分"] = sum(ratings) + row["主观_综合评价"]
                row["主观_总分均值"] = round(row["主观_总分"] / 7, 2)

                # Durations/counts are built together so per-fixation averages
                # and shares are arithmetically consistent. They are invented.
                counts = [rng.randint(3, 24) for _ in LAYERS]
                durations = [round(c * rng.uniform(.17, .33), 3) for c in counts]
                total_duration = sum(durations)
                # Source column is percent of *all* fixation time, including
                # time outside the four named areas, not a 0..1 AOI fraction.
                all_fixation_time = total_duration / rng.uniform(.48, .94)
                for layer, count_fix, duration in zip(LAYERS, counts, durations):
                    row[f"注视次数_{layer}"] = count_fix
                    row[f"总注视时长_{layer}"] = duration
                    row[f"注视时长占比_{layer}"] = round(100 * duration / all_fixation_time, 5)
                    row[f"首次注视时间_{layer}"] = round(rng.uniform(.08, 13.0), 3)
                    row[f"平均注视时长_{layer}"] = round(duration / count_fix, 4)
                pupil = bounded(2.55 + person_effect * .1 + rng.gauss(0, .25), 1.7, 3.5)
                row["平均瞳孔直径_mm"] = round(pupil, 4)
                row["最小瞳孔直径_mm"] = round(pupil - rng.uniform(.25, .6), 4)
                row["最大瞳孔直径_mm"] = round(pupil + rng.uniform(.3, .85), 4)
                blink = rng.randint(0, 36)
                saccades = rng.randint(95, 310)
                row["眨眼次数"] = blink
                row["平均眨眼频率_per_s"] = round(blink / 60, 4)
                row["扫视次数"] = saccades
                row["平均扫视频率_per_s"] = round(saccades / 60, 4)
                row["总扫视时间_s"] = round(rng.uniform(8, 28), 3)
                horizontal = rng.uniform(90, 245)
                vertical = rng.uniform(65, 190)
                row["平均眼跳水平距离_px"] = round(horizontal, 3)
                row["平均眼跳垂直距离_px"] = round(vertical, 3)
                row["平均眼跳绝对距离_px"] = round(math.hypot(horizontal, vertical), 3)
                rows.append(row)
    return rows


def main():
    rows = make_rows()
    assert len(HEADERS) == 52 and len(rows) == 244
    assert all(set(row) == set(HEADERS) for row in rows)
    dest = Path(__file__).parent / "data" / "synthetic_master.csv"
    dest.parent.mkdir(exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} fictional viewing records: {dest}")


if __name__ == "__main__":
    main()
