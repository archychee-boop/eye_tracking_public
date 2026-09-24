"""Generate fictional data, then run each analysis with clear status."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = sorted((ROOT / "analysis").glob("[0-7][0-9]_*.py"))
env = dict(os.environ, MPLBACKEND="Agg")


def run(script):
    print(f"\n>>> {script.relative_to(ROOT)}", flush=True)
    return subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    if run(ROOT / "generate_demo.py") or run(ROOT / "validate_demo.py"):
        sys.exit("示例数据生成或验证失败")
    failed = []
    for script in SCRIPTS:
        if script.name.startswith("04_") and importlib.util.find_spec("statsmodels") is None:
            print("\n>>> 04_ 混合模型跳过：请安装 requirements 中的 statsmodels 后重跑", flush=True)
            continue
        if run(script):
            failed.append(script.name)
    if failed:
        sys.exit("运行失败：" + ", ".join(failed))
    print("\n演示结束。outputs_synthetic 中的数字均不代表论文实证结果。")
