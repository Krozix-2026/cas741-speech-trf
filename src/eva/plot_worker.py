# plot_worker.py  (Windows: stabilize rendering + verify surfaces)
from __future__ import annotations

import os, sys, time, argparse, ctypes
from pathlib import Path

# -------------------------
# A) GUI backend：只用 Qt（别碰 wx；你之前那个 GetEventHandler 崩溃就是 wx 清理栈在作妖）
# -------------------------
os.environ["ETS_TOOLKIT"] = "qt"
os.environ["TRAITSUI_TOOLKIT"] = "qt"
os.environ["QT_API"] = "pyqt5"

# -------------------------
# B) 选择渲染模式（推荐先用 onscreen Win32 OpenGL 把图画出来）
#    1) 先跑 ONSCREEN：最稳，能避免 OSMesa “画不出来却不报错”的玄学
# -------------------------
os.environ.pop("MAYAVI_OFFSCREEN", None)
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkWin32OpenGLRenderWindow"

# ---- 如果你一定要 OFFSCREEN（OSMesa），把上面三行改成下面三行，并确保 MESA_DIR 那段启用：
# os.environ["MAYAVI_OFFSCREEN"] = "1"
# os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
# os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"

# -------------------------
# C) （仅 OFFSCREEN+OSMesa 才需要）注入 Mesa DLL
# -------------------------
MESA_DIR = Path(r"C:\linux_project\LENS\third_party\mesa\x64")
if sys.platform.startswith("win") and os.environ.get("MAYAVI_OFFSCREEN") == "1":
    os.add_dll_directory(str(MESA_DIR))
    os.environ["PATH"] = str(MESA_DIR) + os.pathsep + os.environ.get("PATH", "")
    ctypes.WinDLL("osmesa.dll")
    print("[DBG] osmesa.dll load OK from", MESA_DIR)
    sys.stdout.flush()

# -------------------------
# D) SUBJECTS_DIR（非常关键：这里必须包含你要画的 subject，比如 fsaverage）
# -------------------------
SUBJECTS_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\freesurfer")
if not SUBJECTS_DIR.exists():
    raise RuntimeError(f"SUBJECTS_DIR not found: {SUBJECTS_DIR}")
os.environ["SUBJECTS_DIR"] = str(SUBJECTS_DIR)

def must_exist_png(p: Path, min_bytes: int = 50_000):
    if not p.exists():
        raise RuntimeError(f"PNG not created: {p}")
    n = p.stat().st_size
    if n < min_bytes:
        raise RuntimeError(f"PNG too small (likely blank): {p} size={n} bytes")

def pump_render(w: int, h: int, n: int = 25, dt: float = 0.05):
    """强制 Mayavi/VTK 把 scene 真正渲染出来，否则 save_image 可能抓到空白帧。"""
    from mayavi import mlab
    from pyface.api import GUI

    gui = GUI()
    fig = mlab.gcf()

    # 强行设定 render window 尺寸（比传 w/h 参数更硬）
    try:
        fig.scene.render_window.SetSize(w, h)
    except Exception:
        pass

    for _ in range(n):
        try:
            fig.scene.render()
        except Exception:
            pass
        try:
            mlab.process_ui_events()
        except Exception:
            pass
        try:
            gui.process_events()
        except Exception:
            pass
        time.sleep(dt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    pkl = Path(args.pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 必须在环境变量设置后再 import
    from mayavi import mlab
    if os.environ.get("MAYAVI_OFFSCREEN") == "1":
        mlab.options.offscreen = True
    else:
        mlab.options.offscreen = False

    from eelbrain import configure, load, plot
    configure(prompt_toolkit=False, frame=False, autorun=False)

    res = load.unpickle(pkl)
    name = pkl.name.replace(".TTestRelated.pickle", "")

    # ---------- 关键：检查你到底在画哪个 subject ----------
    # res.difference 是 NDVar；source 维度通常是 SourceSpace，里面有 subject 字段
    src = res.difference.get_dim("source")
    subj = getattr(src, "subject", None)
    print(f"[DBG] source.subject = {subj!r}")
    if subj:
        surf_lh = SUBJECTS_DIR / subj / "surf" / "lh.inflated"
        surf_rh = SUBJECTS_DIR / subj / "surf" / "rh.inflated"
        if not surf_lh.exists() or not surf_rh.exists():
            raise RuntimeError(
                "FreeSurfer surface missing for subject used by your result.\n"
                f"Need these files:\n  {surf_lh}\n  {surf_rh}\n"
                "If subj is 'fsaverage', copy fsaverage into SUBJECTS_DIR."
            )

    w, h = 1600, 1000
    sleep_s = 0.2

    for hemi in ("lh", "rh"):
        # --- p-map ---
        b = plot.brain.p_map(
            res, p0=0.05, p1=0.01,
            surf="inflated", views="lateral", hemi=hemi,
            subjects_dir=str(SUBJECTS_DIR),
        )
        time.sleep(sleep_s)
        pump_render(w, h)
        out_p = out_dir / f"{name}_pmap_{hemi}.png"
        b.save_image(str(out_p), mode="rgb", antialiased=False)
        must_exist_png(out_p)

        # 关闭当前 Mayavi scene，避免下一张复用旧 scene 导致“截图空白”
        mlab.close(all=True)

        # --- masked effect ---
        md = res.masked_difference(p=0.05)
        b2 = plot.brain.brain(
            md,
            surf="inflated", views="lateral", hemi=hemi,
            subjects_dir=str(SUBJECTS_DIR),
        )
        time.sleep(sleep_s)
        pump_render(w, h)
        out_d = out_dir / f"{name}_diff_{hemi}.png"
        b2.save_image(str(out_d), mode="rgb", antialiased=False)
        must_exist_png(out_d)

        mlab.close(all=True)

    print("[WORKER] OK:", name)
    os._exit(0)  # 直接硬退出，绕过 GUI 清理期的各种坑

if __name__ == "__main__":
    main()