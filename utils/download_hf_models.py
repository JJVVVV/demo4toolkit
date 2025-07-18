# nohup python download_hf_models.py &> nohup.out &

from pathlib import Path

from huggingface_hub import snapshot_download

local_models_dir = "~/pretrained_models"
local_models_dir = "/data/jjwang/pretrained/"
local_models_dir = "/home/wangjunjie08/pretrained_models"

# model_name="meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "google/mt5-small"
model_name = "google-bert/bert-base-chinese"

local_models_dir = Path(local_models_dir).expanduser().absolute()
print(local_models_dir / model_name)
# exit()

HF_ENDPOINT = "https://hf-mirror.com"
hf_token = ""

snapshot_download(
    repo_id=model_name,
    token=hf_token or None,
    endpoint=HF_ENDPOINT,
    local_dir=local_models_dir / model_name,
    ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.ot", "original/"],
)
