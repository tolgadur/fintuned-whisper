# Whisper Fine-Tuning Experiment

This project explores the tradeoff between dataset-specific performance and generalizability when fine-tuning OpenAI's Whisper speech recognition model.

## Experiment Overview

The goal of this experiment was to improve Whisper's performance on the LibriSpeech dataset while observing how different fine-tuning approaches affect generalizability to out-of-sample audio.

## Key Findings

### LibriSpeech Performance Improvement

Fine-tuning successfully reduced Word Error Rate (WER) on the LibriSpeech dataset:

- The base model achieved a WER of ~0.19 on test-clean and ~0.27 on test-other
- Fine-tuning with just one epoch significantly improved performance
- Further epochs continued to reduce WER on the target dataset

### Generalizability Problem

However, as shown in `logs/old-logs/out-of-sample-eval.log`, the fine-tuned models completely lost their ability to transcribe simple English phrases outside the training distribution:

- **Base model** (no fine-tuning):
  - Successfully transcribed "Hello, my name is Izaak" and "Hello, my name is Tolga"
  - Overall WER: 0.4

- **Fine-tuned models** (standard approach):
  - Failed to transcribe simple phrases, producing outputs like "HELLO MY MAIMS ISICK"
  - Overall WER increased dramatically to 1.0-1.1

### Mitigation Strategies

To address catastrophic forgetting, several techniques were implemented:

- **LoRA** (Low-Rank Adaptation): Fine-tuning only a small set of parameters
- **KL Divergence**: Keeping model outputs close to the original model
- **EWC** (Elastic Weight Consolidation): Preserving important parameters for general tasks

These techniques successfully retained knowledge on the small out-of-sample dataset, demonstrating effective approaches to balance domain-specific performance improvements with general language capabilities.

## Visualization

The project includes code to visualize log-mel spectrograms of audio inputs, providing insights into how the model processes speech data. Example visualizations can be found in the `visualize/` directory.

## Project Structure

```
whisper/
├── data/                # LibriSpeech dataset (downloaded automatically)
├── logs/                # Training and evaluation logs
├── models/              # Saved model checkpoints
├── src/                 # Source code
│   ├── config.py        # Configuration settings
│   ├── dataset.py       # LibriSpeech dataset loader
│   ├── main.py          # Main entry point
│   ├── trainer.py       # Training functionality
│   ├── utils.py         # Evaluation utilities
│   └── visualize.py     # Spectogram visualization
└── visualize/           # Generated spectogram images
```
