# Whisper Fine-Tuning

A project for fine-tuning OpenAI's Whisper model on the LibriSpeech dataset for improved speech recognition.

## Overview

This project provides tools to fine-tune the Whisper speech recognition model on the LibriSpeech dataset. It includes functionality for training, evaluation, and inference.

## Features

- Fine-tune Whisper models on LibriSpeech dataset
- Evaluate model performance on LibriSpeech test sets
- Calculate Word Error Rate (WER) metrics
- Support for mixed precision training
- Single datapoint evaluation for quick testing

## Project Structure

```
whisper/
├── data/                # LibriSpeech dataset (downloaded automatically)
├── logs/                # Training logs
├── models/              # Saved model checkpoints (not included in repo)
├── src/
│   ├── config.py        # Configuration settings
│   ├── dataset.py       # LibriSpeech dataset loader
│   ├── main.py          # Main entry point
│   ├── trainer.py       # Training functionality
│   ├── utils.py         # Evaluation utilities
│   ├── example_huggingface.py  # Example using HuggingFace
│   └── example_official.py     # Example using official Whisper
└── requirements.txt     # Project dependencies
```

## Usage

### Training

To fine-tune the Whisper model on LibriSpeech:

```python
from src.trainer import train

# Start training with default parameters
train()

# Or customize training parameters
train(
    batch_size=16,
    num_epochs=10,
    learning_rate=1e-4,
    use_amp=True
)
```

### Evaluation

To evaluate the model on LibriSpeech test sets:

```python
from src.utils import evaluate_librispeech

# Evaluate using a fine-tuned model
evaluate_librispeech(model_path="models/whisper-tiny-librispeech.pth")
```

To evaluate on a single datapoint:

```python
from src.utils import evaluate_single_datapoint

# Test on a single example
evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")
```

## Model

This project uses the "whisper-tiny" model by default, but you can modify `config.py` to use other Whisper model variants:
- whisper-tiny
- whisper-base
- whisper-small
- whisper-medium
- whisper-large

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [LibriSpeech Dataset](https://www.openslr.org/12)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
