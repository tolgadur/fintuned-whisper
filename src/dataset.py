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
        # self.dataset = torch.utils.data.Subset(self.dataset, range(1))  # for testing

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
        # Squeeze the batch dimension (dim=0) added by the processor
        input_features = features.input_features.squeeze(0)  # Shape: [80, T]
        attention_mask = features.attention_mask.squeeze(0)  # Shape: [T]

        # Tokenize the transcript
        tokenized = self.processor(text=transcript, return_tensors="pt", padding=True)
        # Squeeze the batch dimension (dim=0) added by the processor
        labels = tokenized.input_ids.squeeze(0)  # Shape: [S] where S is sequence length

        return {
            "input_features": input_features,  # Shape: [80, T]
            "attention_mask": attention_mask,  # Shape: [1, T]
            "labels": labels,  # Shape: [S]
            "transcript": transcript,
        }
