import torch
import torchaudio
from transformers import WhisperProcessor
from config import PROCESSOR


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train-clean-100",
        processor: WhisperProcessor = PROCESSOR,
    ):
        self.processor = processor
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            "./data", url=split, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[index]
        waveform_np = waveform.squeeze().numpy()

        # Process the audio. Returns a log-mel spectrogram.
        # Applied STFT, padding, mel filter bank, and log compression.
        features = self.processor(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = features.input_features
        attention_mask = features.attention_mask

        return input_features, attention_mask, transcript
