"""Check structural and arithmetic invariants in the fictional table."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from generate_demo import HEADERS, LAYERS, SCHEDULE

path = Path(__file__).parent / "data" / "synthetic_master.csv"
with path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 244 and list(rows[0]) == HEADERS
people = defaultdict(list)
for row in rows:
    people[row["受试编号"]].append(row)
    assert int(row["答对题数"]) == (int(row["问答1得分"]) + int(row["问答2得分"])) // 5
    assert float(row["正确率"]) == int(row["答对题数"]) / 2
    scores = []
    for category in ("获取信息效率", "可读性", "吸引力"):
        pair = [int(row[f"{category}_Q{i}"]) for i in (1, 2)]
        scores.extend(pair)
        assert float(row[f"{category}_均值"]) == sum(pair) / 2
    assert int(row["主观_总分"]) == sum(scores) + int(row["主观_综合评价"])
    assert abs(float(row["主观_总分均值"]) - int(row["主观_总分"]) / 7) <= .005
    assert 0 < sum(float(row[f"注视时长占比_{l}"]) for l in LAYERS) < 100
    assert float(row["最小瞳孔直径_mm"]) < float(row["平均瞳孔直径_mm"]) < float(row["最大瞳孔直径_mm"])
assert len(people) == 61
for participant in people.values():
    assert len(participant) == 4
    assert sorted(int(r["观看顺序"]) for r in participant) == [1, 2, 3, 4]
    group = participant[0]["组别"]
    assert set(r["设计代号"] for r in participant) == set(SCHEDULE[group][1])
counts = Counter(v[0]["组别"] for v in people.values())
assert all(counts[g] == n for g, (n, _) in SCHEDULE.items())
print("验证通过：244 行、52 列、61 位虚构被试，计分及分组关系一致。")
