import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


MODEL_NAME = "openai/whisper-tiny"

PROCESSOR = WhisperProcessor.from_pretrained(MODEL_NAME)
MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
MODEL.config.forced_decoder_ids = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
