import os
from huggingface_hub import HfApi, create_repo, upload_file

def main():
    # 读取GitHub密钥配置
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    # 定义Hugging Face仓库名称
    repo_id = f"{hf_username}/space_mining_simple_model"
    
    # 初始化HF上传API
    api = HfApi(token=hf_token)
    # 创建仓库（不存在则新建，存在则跳过）
    create_repo(repo_id, exist_ok=True, token=hf_token)
    
    # 需上传的文件列表
    upload_files = [
        ("model/simple_model.pkl", "simple_model.pkl"),
        ("eval_result.txt", "eval_result.txt"),
        ("README.md", "README.md")
    ]
    
    # 批量上传文件
    for local_path, repo_path in upload_files:
        if os.path.exists(local_path):
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                token=hf_token
            )
    print(f"🎉 模型上传成功！访问地址：https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()