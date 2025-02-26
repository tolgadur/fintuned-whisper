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
        # self.dataset = torch.utils.data.Subset(self.dataset, range(2))  # for testing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[index]
        waveform_np = waveform.squeeze().numpy()

        # Normalize transcript for consistency
        # Convert to lowercase and strip whitespace for consistent casing and formatting
        transcript = transcript.strip().lower()

        # Process the audio. Returns a log-mel spectrogram.
        # Applied STFT, padding, mel filter bank, and log compression.
        features = self.processor(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # Remove the batch dimension (first dimension) as DataLoader will add it back
        input_features = features.input_features.squeeze(0)  # Remove batch dimension
        attention_mask = features.attention_mask.squeeze(0)  # Remove batch dimension

        return input_features, attention_mask, transcript


def collate_fn(batch):
    """
    Process a batch of raw audio data for the Whisper model.

    Args:
        batch: A list of tuples (input_features, attention_mask, transcript)

    Returns:
        A dictionary with processed features ready for the model
    """
    input_features, attention_mask, transcripts = zip(*batch)

    # Stack the input features and attention masks into single batch tensors
    # Each item should be a 2D tensor, and we're stacking along a new first dimension
    input_features = torch.stack(input_features, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)

    # Transcripts are already normalized in __getitem__
    tokenized = PROCESSOR(text=transcripts, return_tensors="pt", padding=True)
    labels = tokenized.input_ids

    return {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "labels": labels,
        "transcript": transcripts,
    }
