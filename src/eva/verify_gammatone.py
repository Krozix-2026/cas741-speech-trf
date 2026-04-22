import pickle
from pathlib import Path

p = Path("C:/Dataset/Appleseed/stimuli/1-gammatone100.pickle")
obj = pickle.loads(p.read_bytes())
print(type(obj))

# 常见两种：numpy array 或 eelbrain NDVar
if hasattr(obj, "shape"):
    print("shape:", obj.shape)

# eelbrain NDVar 通常有 .dims / .time
if hasattr(obj, "dims"):
    print("dims:", obj.dims)
if hasattr(obj, "time"):
    # NDVar 的 time 轴常能给采样间隔
    try:
        print("tstep:", obj.time.tstep)
    except Exception as e:
        print("time info error:", e)