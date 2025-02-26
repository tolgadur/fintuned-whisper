from trainer import train, train_lora_ewc_kd
from utils import evaluate_out_of_sample, evaluate_single_datapoint
from visualize import visualize_log_mel_spectrogram, visualize_hello_izaak


def main():
    # Uncomment the training method you want to use
    # print("Training with standard fine-tuning...")
    # train(num_epochs=1)

    # print("Training with LoRA and KL divergence...")
    # train_lora_ewc_kd(num_epochs=1)

    # Visualize log mel spectrogram for the first file in the dataset
    print("Visualizing log mel spectrogram from LibriSpeech dataset...")
    visualize_log_mel_spectrogram()

    # Visualize log mel spectrogram for the hello-izaak file
    print("\nVisualizing log mel spectrogram for 'hello-izaak.wav'...")
    visualize_hello_izaak()

    # Comment out the following line to evaluate the single datapoint
    # evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd-ewc")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd-ewc")

    print("Evaluating...")
    print("Evaluating regular non-fined tuned model")
    evaluate_out_of_sample()
    print("-" * 100)

    print("Evaluating model finetuned with one epoch")
    evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-1.pth")
    print("-" * 100)

    print("Evaluating model finetuned with five epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-4.pth")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA and KL divergence five epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA, KL divergence, and EWC combined")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd-ewc")
    print("-" * 100)


if __name__ == "__main__":
    main()
