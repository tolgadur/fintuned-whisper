import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from dataset import LibriSpeechDataset
from config import PROCESSOR


def visualize_log_mel_spectrogram():
    """
    Visualize the log mel spectrogram for the first file in the dataset.

    This function:
    1. Loads the first audio sample from the LibriSpeech dataset
    2. Computes the log mel spectrogram using the Whisper processor
    3. Visualizes it as a heatmap with matplotlib
    """
    # Load the dataset and get the first item
    dataset = LibriSpeechDataset()
    input_features, _, transcript = dataset[1]

    # input_features is already the log mel spectrogram computed by the processor
    # Shape is [80, T] where 80 is the number of mel filterbanks and T is time frames

    # Convert to numpy for plotting
    spectrogram = input_features.numpy()

    # Create figure with a single plot
    plt.figure(figsize=(10, 6))

    # Plot spectrogram with enhanced visibility
    # Use a colormap and adjust vmin/vmax for better contrast
    im = plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="none",
        vmin=np.percentile(spectrogram, 5),  # Clip lowest values
        vmax=np.percentile(spectrogram, 95),  # Clip highest values
    )
    plt.title("Log Mel Spectrogram")
    plt.ylabel("Mel Filterbank Index")
    plt.xlabel("Time Frame")
    plt.colorbar(im, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig("log_mel_spectrogram.png")
    plt.show()

    print(f"Transcript: {transcript}")
    print(f"Spectrogram shape: {spectrogram.shape}")


def visualize_hello_izaak():
    """
    Visualize the log mel spectrogram for the 'hello-izaak.wav' file.

    This function:
    1. Loads the hello-izaak audio file
    2. Computes the log mel spectrogram using the Whisper processor
    3. Visualizes it as a heatmap with matplotlib
    """
    # Load the audio file
    file_path = "data/hello-izaak.wav"
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample to 16000 Hz if needed
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform_np = waveform.squeeze().numpy()

    # Process the audio to get input features
    features = PROCESSOR(
        waveform_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Extract the log mel spectrogram
    spectrogram = features.input_features.squeeze(0).numpy()

    # Create figure with a single plot
    plt.figure(figsize=(10, 6))

    # Plot spectrogram with enhanced visibility
    im = plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="none",
        vmin=np.percentile(spectrogram, 5),  # Clip lowest values
        vmax=np.percentile(spectrogram, 95),  # Clip highest values
    )
    plt.title("Log Mel Spectrogram - 'Hello, my name is Izaak'")
    plt.ylabel("Mel Filterbank Index")
    plt.xlabel("Time Frame")
    plt.colorbar(im, format="%+2.0f dB")

    # Add time markers on x-axis (approximate)
    duration = len(waveform_np) / sample_rate  # in seconds
    plt.xticks(
        np.linspace(0, spectrogram.shape[1], 5),
        [f"{t:.1f}s" for t in np.linspace(0, duration, 5)],
    )

    plt.tight_layout()
    plt.savefig("hello_izaak_spectrogram.png")
    plt.show()

    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Spectrogram shape: {spectrogram.shape}")
