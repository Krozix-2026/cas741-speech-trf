import numpy as np


data = np.load('round1_word_embeds_devclean.npz', allow_pickle=True)
words = data['words']
counts = data['counts']

# 2. 获取按出现次数从大到小排序的索引
# argsort() 返回升序索引，[::-1] 实现翻转得到降序
top_indices = counts.argsort()[::-1]

# 3. 打印前 50 个高频词
print('top50=')
for i in top_indices[:50]:
    word = words[i]
    count = int(counts[i])
    print(f'{word:>12s}  {count}')