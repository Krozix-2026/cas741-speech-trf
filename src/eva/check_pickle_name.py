from __future__ import annotations
from pathlib import Path
from eelbrain import load
import re

ROOT = Path("/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/derivatives/eelbrain/cache/trf/01/1-40_emptyroom_fixed-6-MNE-0")

# 更稳：直接找 cv.pickle（而不是靠文件名里有没有 model）
files = sorted(ROOT.glob("*cv.pickle"))

def _is_big_array_like(x) -> bool:
    # 避免递归扫进巨大矩阵/ndvar导致爆内存/爆输出
    t = type(x)
    mod = getattr(t, "__module__", "")
    name = getattr(t, "__name__", "")
    if mod.startswith("numpy") and name == "ndarray":
        return True
    # 很多科学对象都有 shape / nbytes 之类，也简单跳过
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return True
    return False

def _looks_like_formula(s: str) -> bool:
    # 你可以按自己的 predictor 名字扩展关键字
    KEY = ("~", "Code(", "gammatone", "onset", "phoneme", "surprisal", "gpt", "model")
    return any(k in s for k in KEY)

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def extract_formulas(obj, *, max_depth=5, max_items=60):
    """
    从任意对象里尽可能挖出“像模型公式/预测器定义”的字符串。
    - 优先抓常见字段/属性
    - 再做有限深度递归扫描（带 visited 防环）
    """
    found = []
    visited = set()

    def add_candidate(x):
        s = _safe_str(x)
        if _looks_like_formula(s):
            found.append(s)

    # 先走一圈“高收益字段”
    def try_common_fields(o):
        # dict 常见键
        if isinstance(o, dict):
            for k in ("code", "model", "models", "formula", "predictors", "predictor", "design"):
                if k in o:
                    add_candidate(o[k])
        # object 常见属性
        for a in ("code", "model", "models", "formula", "predictors", "predictor", "design", "x", "X"):
            if hasattr(o, a):
                try:
                    v = getattr(o, a)
                except Exception:
                    continue
                # x / X 往往是大矩阵，容易炸；先看看是不是“字符串/小对象”
                if _is_big_array_like(v):
                    continue
                add_candidate(v)

    def walk(o, depth):
        oid = id(o)
        if oid in visited:
            return
        visited.add(oid)

        if o is None:
            return

        if isinstance(o, str):
            if _looks_like_formula(o):
                found.append(o)
            return

        if isinstance(o, (int, float, bool)):
            return

        if _is_big_array_like(o):
            return

        # 先试高收益字段
        try_common_fields(o)

        if depth >= max_depth:
            return

        # 容器递归（限制 max_items 防止刷屏）
        if isinstance(o, dict):
            for i, (k, v) in enumerate(o.items()):
                if i >= max_items:
                    break
                # key 也可能带信息
                if isinstance(k, str) and _looks_like_formula(k):
                    found.append(k)
                walk(v, depth + 1)
            return

        if isinstance(o, (list, tuple, set)):
            for i, v in enumerate(list(o)[:max_items]):
                walk(v, depth + 1)
            return

        # 普通对象：扫 __dict__（同样限制数量）
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            for i, (k, v) in enumerate(d.items()):
                if i >= max_items:
                    break
                if isinstance(k, str) and _looks_like_formula(k):
                    found.append(k)
                walk(v, depth + 1)

    walk(obj, 0)

    # 去重 + 简单清洗（保留顺序）
    uniq = []
    seen = set()
    for s in found:
        ss = s.strip()
        if ss not in seen:
            seen.add(ss)
            uniq.append(ss)
    return uniq

def extract_model_map(obj):
    """
    如果 pickle 里存在 model0/model1/... 这种结构，尽量挖出每个 modelX 对应的公式。
    """
    out = {}
    # 1) dict: 直接找 model0/model1 key
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and re.fullmatch(r"model\d+", k.lower()):
                cand = extract_formulas(v, max_depth=4)
                if cand:
                    out[k] = cand[0]
        # 有些 dict 会把 models 放在某个 key 下
        for kk in ("models", "model", "code", "design"):
            if kk in obj and not out:
                cand = extract_formulas(obj[kk], max_depth=4)
                if cand:
                    out[kk] = cand[0]
        return out

    # 2) object: 属性里可能有 models / results / etc
    for a in ("models", "model", "code", "design", "results"):
        if hasattr(obj, a):
            try:
                v = getattr(obj, a)
            except Exception:
                continue
            if isinstance(v, dict):
                for k, vv in v.items():
                    if isinstance(k, str) and re.fullmatch(r"model\d+", k.lower()):
                        cand = extract_formulas(vv, max_depth=4)
                        if cand:
                            out[k] = cand[0]
            else:
                cand = extract_formulas(v, max_depth=4)
                if cand:
                    out[a] = cand[0]
            if out:
                return out
    return out

print("ROOT =", ROOT)
print("n_cv_pickles =", len(files))
for p in files:
    x = load.unpickle(p)

    formulas = extract_formulas(x, max_depth=5)
    model_map = extract_model_map(x)

    print("\n" + "=" * 90)
    print("FILE:", p.name)
    print("TYPE:", type(x))

    if model_map:
        print("MODEL MAP (best guess):")
        for k, v in sorted(model_map.items()):
            print(f"  {k:>8} -> {v}")
    else:
        # 没有显式 model0/model1 结构：就打印最像公式的前几个候选
        if formulas:
            print("FORMULA CANDIDATES (top):")
            for s in formulas[:5]:
                print("  -", s)
        else:
            print("No obvious formula string found (try raising max_depth/max_items).")