# plot_only.py (WIN: Win32 OpenGL + wx, safe set_size)
from __future__ import annotations

import os, sys, time, gc, traceback
from pathlib import Path
import argparse

# ------------------------------
# 0) BEFORE importing eelbrain/mayavi/vtk
# ------------------------------
if sys.platform.startswith("win"):
    # Force Win32 OpenGL backend (no OSMesa offscreen)
    os.environ.pop("MAYAVI_OFFSCREEN", None)
    os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkWin32OpenGLRenderWindow"

# Single GUI toolkit: wx (do NOT mix Qt)
os.environ["ETS_TOOLKIT"] = "wx"
os.environ["TRAITSUI_TOOLKIT"] = "wx"
for k in ("QT_API", "QT_QPA_PLATFORM"):
    os.environ.pop(k, None)

# SUBJECTS_DIR
SUBJECTS_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\freesurfer")
if not SUBJECTS_DIR.exists():
    raise RuntimeError(f"SUBJECTS_DIR not found: {SUBJECTS_DIR}")
os.environ["SUBJECTS_DIR"] = str(SUBJECTS_DIR)

print("[DBG] VTK_DEFAULT_OPENGL_WINDOW =", os.environ.get("VTK_DEFAULT_OPENGL_WINDOW"))
print("[DBG] ETS_TOOLKIT =", os.environ.get("ETS_TOOLKIT"), "TRAITSUI_TOOLKIT =", os.environ.get("TRAITSUI_TOOLKIT"))

from eelbrain import configure, load, plot

# ✅ 关键：不要 frame=False；强制 frame=True
# autorun=False: script里不自动进入 GUI mainloop
configure(prompt_toolkit=False, frame=True, autorun=False)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def try_set_size(brain, width: int, height: int) -> bool:
    """
    Brain.set_size() needs Eelbrain wx Frame.
    If no frame, skip instead of crashing.
    """
    fr = getattr(brain, "_frame", None)
    if fr is None:
        return False
    brain.set_size(width, height)
    return True

def save_maps(
    res,
    out_dir: Path,
    name: str,
    subjects_dir: Path,
    surf: str = "inflated",
    view: str = "lateral",
    p0: float = 0.05,
    p1: float = 0.01,
    p_mask: float = 0.05,
    size: tuple[int, int] = (1600, 1000),
    sleep_s: float = 0.6,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = size  # use plot-time sizing, not Brain.set_size()

    for hemi in ("lh", "rh"):
        # --- p-map ---
        b = plot.brain.p_map(
            res,
            p0=p0, p1=p1,
            surf=surf,
            views=view,
            hemi=hemi,
            subjects_dir=str(subjects_dir),
            # size control (supported by plot.brain.* wrappers)
            w=w, h=h,
        )
        time.sleep(sleep_s)
        out_p = out_dir / f"{name}_pmap_{hemi}.png"
        b.save_image(str(out_p), mode="rgb", antialiased=True)
        b.close()
        del b
        gc.collect()

        # --- masked effect (Δ) ---
        md = res.masked_difference(p=p_mask)
        b2 = plot.brain.brain(
            md,
            surf=surf,
            views=view,
            hemi=hemi,
            subjects_dir=str(subjects_dir),
            w=w, h=h,
        )
        time.sleep(sleep_s)
        out_d = out_dir / f"{name}_diff_{hemi}.png"
        b2.save_image(str(out_d), mode="rgb", antialiased=True)
        b2.close()
        del b2
        gc.collect()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default=r"C:\Dataset\Appleseed_BIDS_new\derivatives\eelbrain\eva\mtrf_compare")
    ap.add_argument("--out_dir", type=str, default=r"C:\linux_project\LENS\runs\trf")
    ap.add_argument("--p0", type=float, default=0.05)
    ap.add_argument("--p1", type=float, default=0.01)
    ap.add_argument("--p_mask", type=float, default=0.05)
    ap.add_argument("--surf", type=str, default="inflated")
    ap.add_argument("--view", type=str, default="lateral")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("in_dir:", in_dir)
    print("out_dir:", out_dir)
    print("SUBJECTS_DIR:", SUBJECTS_DIR)

    pickles = sorted(in_dir.glob("*.TTestRelated.pickle"))
    if not pickles:
        raise RuntimeError(f"No *.TTestRelated.pickle in: {in_dir}")

    summary_lines = [
        f"in_dir: {in_dir}",
        f"out_dir: {out_dir}",
        f"SUBJECTS_DIR: {SUBJECTS_DIR}",
        "",
    ]

    for pkl in pickles:
        name = pkl.name.replace(".TTestRelated.pickle", "")
        print(f"\n[INFO] Loading: {pkl.name}")

        try:
            res = load.unpickle(pkl)
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            summary_lines += [f"== {name} ==", "LOAD FAILED", tb, ""]
            continue

        summary_lines.append(f"== {name} ==")
        summary_lines.append(str(res))

        diff = res.difference
        summary_lines.append(f"mean Δ: {safe_float(diff.mean())}")
        summary_lines.append(f"max  Δ: {safe_float(diff.max())}")

        try:
            summary_lines.append(f"min p: {safe_float(res.p.min())}")
        except Exception as err:
            summary_lines.append(f"min p: (unavailable) {err}")

        try:
            save_maps(
                res=res,
                out_dir=out_dir,
                name=name,
                subjects_dir=SUBJECTS_DIR,
                surf=args.surf,
                view=args.view,
                p0=args.p0,
                p1=args.p1,
                p_mask=args.p_mask,
                size=(args.width, args.height),
                sleep_s=args.sleep,
            )
            summary_lines.append("render: OK")
            print("[INFO] render: OK")
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            summary_lines.append("render: FAILED")
            summary_lines.append(tb)

        summary_lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n[OK] Wrote: {out_dir/'summary.txt'}")
    print(f"[OK] PNGs in: {out_dir}")

if __name__ == "__main__":
    main()