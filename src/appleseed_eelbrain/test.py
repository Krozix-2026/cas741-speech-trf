import eelbrain as eb
import numpy as np
import matplotlib.pyplot as plt

p = r"C:\Dataset\Appleseed-BIDS\derivatives\eelbrain\cache\trf\R2349\1-40_emptyroom_fixed-6-MNE-0\sub-R2349_meg nobl -100-1000 50Hz superiortemporal model0 boosting h50 l2 con4ptns ss1 cv.pickle"
cv = eb.load.unpickle(p)
print(type(cv))
print(cv)

def get_score(obj):
    for name in ("r", "scores", "cv_scores", "score"):
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(obj, dict):
        for k in ("r", "scores", "cv_scores", "score"):
            if k in obj:
                return obj[k]
    return None

score = get_score(cv)
print("score =", score)

# 1) 常用字段快速浏览
keys = [k for k in dir(cv) if not k.startswith("_")]
print([k for k in keys if k in ("r", "h", "tstart", "tstop", "basis", "partitions", "x", "y", "model")])

# 2) 直接看 r（你现在的 score）
r = cv.r
print("r dims:", r.dims)
print("r shape:", r.x.shape)
print("r finite:", np.isfinite(r.x).all(), "min/max:", np.nanmin(r.x), np.nanmax(r.x))

plt.figure()
plt.hist(r.x[np.isfinite(r.x)], bins=40)
plt.title("CV correlation across sources (ROI)")
plt.xlabel("r")
plt.ylabel("count")
plt.show()

h = cv.h
print(type(h))         # tuple
print(len(h))          # 2

h0 = h[0]              # gammatone_8 的 TRF
h1 = h[1]              # edge30 的 TRF
print(h0.dims, h1.dims)


b = eb.plot.brain.brain(r)


# ROI 平均相关
r_mean = float(r.mean('source'))
# ROI 中位数相关
r_median = float(np.median(r.x))
# 更稳一点：去掉最差 5% 的 trimmed mean
lo, hi = np.quantile(r.x, [0.05, 0.95])
r_trim = float(np.mean(r.x[(r.x >= lo) & (r.x <= hi)]))

print("ROI mean r:", r_mean)
print("ROI median r:", r_median)
print("ROI trimmed mean r:", r_trim)