import evaluate
from config import MODEL, PROCESSOR, DEVICE
from dataset import LibriSpeechDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
import os
from peft import PeftModel


def load_model(model_path=None):
    """
    Load a model from a checkpoint, handling both standard and LoRA models.

    Args:
        model_path: Path to the model checkpoint. If None, returns the base model.

    Returns:
        The loaded model on the correct device.
    """
    if model_path:
        # Check if this is a LoRA model
        adapter_path = os.path.join(model_path, "adapter_config.json")
        is_dir = os.path.isdir(model_path)
        has_adapter = os.path.exists(adapter_path)
        is_lora_model = is_dir and has_adapter

        if is_lora_model:
            print(f"Loading LoRA model from {model_path}")
            # Load the LoRA model
            model = MODEL.to(DEVICE)
            model = PeftModel.from_pretrained(model, model_path)
        else:
            print(f"Loading standard model from {model_path}")
            model_state = torch.load(model_path, map_location=torch.device(DEVICE))
            MODEL.load_state_dict(model_state)
            model = MODEL.to(DEVICE)
    else:
        model = MODEL.to(DEVICE)

    model.eval()
    return model


def calculate_wer(predictions, references):
    wer = evaluate.load("wer")
    return wer.compute(predictions=predictions, references=references)


def evaluate_single_datapoint(model_path: str = None):
    # Load the model
    model = load_model(model_path)

    # Load the dataset
    dataset = LibriSpeechDataset()
    input_features, attention_mask, transcript = dataset[0]

    # Extract data from batch
    input_features = input_features.unsqueeze(0).to(DEVICE)
    attention_mask = attention_mask.unsqueeze(0).to(DEVICE)

    # Generate the transcription.
    # shape of input_features: batch_size, mel-spectrogram features, time steps
    # shape of input_features: [B, 80, T_audio]. In this case [1, 80, 3000].
    # time steps in the audio vary based on audio length
    predicted_ids = model.generate(
        input_features=input_features,
        language="en",
        task="transcribe",
        attention_mask=attention_mask,
    )

    # Decode the transcription
    # shape of predicted_ids: batch_size, sequence length of predicted token ids
    # shape of predicted_ids: [B, T_text]. In this case [1, 49].
    predicted_transcript = PROCESSOR.decode(predicted_ids[0], skip_special_tokens=True)

    # Print the transcription
    print("Whisper transcription:", predicted_transcript)
    print("Original transcript:", transcript)

    # WER rate calculation
    wer = calculate_wer([predicted_transcript], [transcript])
    print("WER rate:", wer)


def evaluate_librispeech(
    test_splits=["test-clean", "test-other"],
    batch_size=32,
    model_path: str = None,
):
    # Load the model
    model = load_model(model_path)

    # Evaluate on test splits
    for split in test_splits:
        print(f"\nEvaluating on {split} split:")
        dataset = LibriSpeechDataset(split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        predictions = []
        references = []

        # Use torch.no_grad() to disable gradient calculation during evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {split}"):
                # Extract data from batch
                input_features, attention_mask, transcripts = batch

                input_features.to(DEVICE)
                attention_mask.to(DEVICE)
                transcripts.to(DEVICE)

                # Generate the transcriptions
                predicted_ids = model.generate(
                    input_features=input_features,
                    language="en",
                    task="transcribe",
                    attention_mask=attention_mask,
                )

                # Decode the transcriptions
                predicted_transcripts = PROCESSOR.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )

                predictions.extend(
                    predicted_transcripts
                    if isinstance(predicted_transcripts, list)
                    else [predicted_transcripts]
                )
                references.extend(transcripts)

        # Calculate WER for this split
        wer = calculate_wer(predictions, references)
        print(f"WER rate for {split}: {wer}")


def evaluate_out_of_sample(model_path: str = None):
    """
    Evaluate the model on custom audio samples.

    Args:
        model_path: Optional path to a saved model checkpoint
    """
    # Load the model
    model = load_model(model_path)

    # Define the audio files and their expected transcripts
    samples = [
        {"path": "data/hello-izaak.wav", "transcript": "Hello, my name is Izaak"},
        {"path": "data/hello-tolga.wav", "transcript": "Hello, my name is Tolga"},
    ]

    predictions = []
    references = []

    print("\nEvaluating out-of-sample audio files:")

    # Process each sample
    for sample in samples:
        file_path = sample["path"]
        expected_transcript = sample["transcript"]

        print(f"\nProcessing: {os.path.basename(file_path)}")
        print(f"Expected transcript: {expected_transcript}")

        # Load and process the audio
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

        waveform_np = waveform.squeeze().numpy()

        # Process the audio to get input features
        features = PROCESSOR(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = features.input_features.to(DEVICE)
        attention_mask = features.attention_mask.to(DEVICE)

        # Generate the transcription
        predicted_ids = model.generate(
            input_features=input_features,
            language="en",
            task="transcribe",
            attention_mask=attention_mask,
        )

        # Decode the transcription
        predicted_transcript = PROCESSOR.decode(
            predicted_ids[0], skip_special_tokens=True
        )
        print(f"Predicted transcript: {predicted_transcript}")

        # Calculate WER
        wer_score = calculate_wer([predicted_transcript], [expected_transcript])
        print(f"WER: {wer_score}")

        predictions.append(predicted_transcript)
        references.append(expected_transcript)

    # Calculate overall WER
    overall_wer = calculate_wer(predictions, references)
    print(f"\nOverall WER for out-of-sample audio: {overall_wer}")

    return overall_wer
