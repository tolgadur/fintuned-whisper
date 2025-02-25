import evaluate
from config import MODEL, PROCESSOR
from dataset import LibriSpeechDataset


def calculate_wer(predictions, references):
    wer = evaluate.load("wer")
    return wer.compute(predictions=predictions, references=references)


def example():
    dataset = LibriSpeechDataset()
    input_features, attention_mask, transcript = dataset[0]

    # Generate the transcription
    predicted_ids = MODEL.generate(
        input_features=input_features,
        language="en",
        task="transcribe",
        attention_mask=attention_mask,
    )

    # Decode the transcription
    predicted_transcript = PROCESSOR.decode(predicted_ids[0])

    # print the transcription
    print("Whisper transcription:", predicted_transcript)
    print("Original transcript:", transcript)

    # WER rate calculation
    wer = calculate_wer([predicted_transcript], [transcript])
    print("WER rate:", wer)
