import torch
import torchaudio


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train-clean-100"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            "./data", url=split, download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
