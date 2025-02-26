import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_processor(model_name: str = "openai/whisper-tiny"):
    return WhisperProcessor.from_pretrained(model_name)


def load_new_model(model_name: str = "openai/whisper-tiny"):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.use_cache = False
    return model


PROCESSOR = load_processor()
MODEL = load_new_model()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
