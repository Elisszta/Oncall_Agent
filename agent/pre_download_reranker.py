import os
from modelscope import snapshot_download

# 使用 ModelScope 下载模型，这在内地通常非常快
# 模型 ID
model_id = 'BAAI/bge-reranker-base'

# 指定本地下载路径（可以根据需要修改这里）
local_dir = './bge-reranker-base'

print(f"正在从 ModelScope 下载 {model_id} 到 {local_dir}...")

try:
    # 自动处理下载，如果已经下载了一部分，会自动续传
    snapshot_download(model_id, cache_dir=local_dir, revision='master')
    print("\n[成功] 模型下载完成！")
    print(f"请在代码中修改模型路径为: {os.path.abspath(local_dir)}")
except Exception as e:
    print(f"\n[错误] 下载失败: {e}")
    print("请确保已安装 modelscope: pip install modelscope")
