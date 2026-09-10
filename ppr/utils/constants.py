import os


huggingface_cache_dir = (
    os.environ.get("PPR_CACHE_DIR")
    or os.environ.get("HF_HUB_CACHE")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or os.environ.get("HF_HOME")
    or os.environ.get("HUGGING_FACE_CACHE_DIR")
)
UNET_CKPT_NAME = "unet"
UNET_LORA_CKPT_NAME = "unet_lora.pt"
