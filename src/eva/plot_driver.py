from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKER = HERE / "plot_worker.py"

IN_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\eelbrain\eva\mtrf_compare")
OUT_DIR = Path(r"C:\linux_project\LENS\runs\trf")

print("[DRIVER] python =", sys.executable)
print("[DRIVER] worker =", WORKER)

pkls = sorted(IN_DIR.glob("*.TTestRelated.pickle"))
if not pkls:
    raise RuntimeError(f"No pickles in {IN_DIR}")

for pkl in pkls:
    print("[DRIVER] Rendering", pkl.name)
    subprocess.run(
        [sys.executable, str(WORKER), "--pkl", str(pkl), "--out_dir", str(OUT_DIR)],
        cwd=str(HERE),
        check=True,   # ✅ worker 非0就立刻失败
    )

print("[DRIVER] DONE")