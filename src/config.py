import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-tiny"

PROCESSOR = WhisperProcessor.from_pretrained(MODEL_NAME)
MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
MODEL.config.forced_decoder_ids = None

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
