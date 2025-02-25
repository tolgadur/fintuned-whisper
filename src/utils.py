import evaluate
from config import MODEL, PROCESSOR, DEVICE
from dataset import LibriSpeechDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_wer(predictions, references):
    wer = evaluate.load("wer")
    return wer.compute(predictions=predictions, references=references)


def evaluate_single_datapoint(model_path: str = None):
    # Load the model
    if model_path:
        MODEL.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    MODEL.eval()

    # Load the dataset
    dataset = LibriSpeechDataset()
    input_features, attention_mask, transcript = dataset[0]

    # Move tensors to the correct device
    input_features = input_features.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    # Generate the transcription.
    # shape of input_features: batch_size, mel-spectrogram features, time steps
    # shape of input_features: [B, 80, T_audio]. In this case [1, 80, 3000].
    # time steps in the audio vary based on audio length
    predicted_ids = MODEL.generate(
        input_features=input_features,
        language="en",
        task="transcribe",
        attention_mask=attention_mask,
    )

    # Decode the transcription
    # shape of predicted_ids: batch_size, sequence length of predicted token ids
    # shape of predicted_ids: [B, T_text]. In this case [1, 49].
    predicted_transcript = PROCESSOR.decode(predicted_ids[0], skip_special_tokens=True)

    # print the transcription
    print("Whisper transcription:", predicted_transcript)
    print("Original transcript:", transcript)

    # WER rate calculation
    MODEL.eval()
    wer = calculate_wer([predicted_transcript], [transcript])
    print("WER rate:", wer)


def evaluate_librispeech(
    test_splits=["test-clean", "test-other"],
    batch_size=32,
    model_path: str = None,
):
    # Load the model
    if model_path:
        MODEL.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    MODEL.eval()

    # evaluate on test splits
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
                input_features, attention_mask, transcripts = batch

                # Squeeze out the channel dimension (converting from [B,1,80,T] to [B,80,T])
                input_features = input_features.squeeze(1)

                # Move tensors to the same device as the model
                input_features = input_features.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)

                # Generate the transcriptions
                predicted_ids = MODEL.generate(
                    input_features=input_features,
                    language="en",
                    task="transcribe",
                    attention_mask=attention_mask,
                )

                # Decode the transcriptions (try direct decoding first)
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
