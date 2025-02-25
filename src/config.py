import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-large")
MODEL = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
