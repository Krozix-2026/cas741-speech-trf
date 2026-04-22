import numpy as np

# 加载文件，注意添加 allow_pickle=True
file_path = r'C:\linux_project\LENS\LLM\Appleseed_LLM_alignment\segments\segment 1.gpt2_features.npz'
data = np.load(file_path, allow_pickle=True)

print("data.files:", data.files)

for key in data.files:
    content = data[key]
    # 如果 content 是个对象数组，有时需要调用 .item() 展开
    print(f"key: {key}")
    print(f"content: {type(content)}")
    
    # 尝试打印维度，如果是标量对象可能没有 .shape
    if hasattr(content, 'shape'):
        print(f"shape: {content.shape}")
    
    # 打印部分内容预览
    print(f"内容预览: {content}\n")

data.close()