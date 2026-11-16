from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775',local_dir=ROOT/'models/base',allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md'])
