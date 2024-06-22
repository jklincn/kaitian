from pathlib import Path

CUDA_IMAGE = "jklincn/kaitian:0.0.0-cuda"
MLU_IMAGE = "jklincn/kaitian:0.0.0-mlu"
CUDA_BASE_IMAGE = "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel"
MLU_BASE_IMAGE = (
    "yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310"
)

MAX_COMPUTE_CAPABILITY = 10.0

CONFIG_DIR = Path.home() / ".config" / "kaitian"
CONFIG_FILE = CONFIG_DIR / "kaitian.toml"
