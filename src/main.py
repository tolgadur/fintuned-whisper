from trainer import train, train_lora_ewc_kd
from utils import (
    evaluate_out_of_sample,
    evaluate_librispeech,
    evaluate_single_datapoint,
)
from visualize import visualize_log_mel_spectrogram, visualize_hello_izaak


def main():
    # Uncomment the training method you want to use
    # Training with standard fine-tuning - try with fewer epochs
    # print("Training with standard fine-tuning...")
    # train(
    #     batch_size=8,
    #     num_epochs=1,
    #     learning_rate=1e-4,
    #     use_amp=True,
    # )

    # # Training with LoRA, EWC and KL divergence - recommended approach
    # print("Training with LoRA, EWC and KL divergence...")
    # train_lora_ewc_kd(
    #     num_epochs=1,  # Reduced from 5 to avoid overfitting
    #     learning_rate=1e-5,  # Lower learning rate
    #     batch_size=8,  # Reduced batch size
    #     lora_r=8,  # Increased LoRA rank for better adaptation
    #     kd_weight=0.7,  # Higher weight on knowledge distillation
    #     use_ewc=True,
    #     ewc_lambda=1000.0,  # Stronger regularization to prevent forgetting
    # )

    # Visualize log mel spectrogram for the first file in the dataset
    # print("Visualizing log mel spectrogram from LibriSpeech dataset...")
    # visualize_log_mel_spectrogram()

    # Visualize log mel spectrogram for the hello-izaak file
    # print("\nVisualizing log mel spectrogram for 'hello-izaak.wav'...")
    # visualize_hello_izaak()

    # Comment out the following line to evaluate the single datapoint
    # evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd-ewc")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd-ewc")

    # print("Evaluating...")
    # print("Evaluating regular non-fined tuned model")
    # evaluate_out_of_sample()
    # print("-" * 100)

    # print("Evaluating model finetuned with one epoch")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-1-2.pth")
    # print("-" * 100)

    # print("Evaluating model finetuned with five epochs")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-4.pth")
    # print("-" * 100)

    # print("Evaluating model finetuned with LoRA and KL divergence five epochs")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd")
    # print("-" * 100)

    # print("Evaluating model finetuned with LoRA, KL divergence, and EWC combined")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd-ewc-2")
    # print("-" * 100)

    # In sample evaluation
    # print("Evaluating LibriSpeech dataset no finetuning")
    # evaluate_librispeech()
    # print("-" * 100)

    print("Evaluating LibriSpeech dataset finetuned with one epoch")
    evaluate_librispeech(model_path="models/whisper-tiny-librispeech-epoch-1-2.pth")
    print("-" * 100)

    print("Evaluating LibriSpeech dataset finetuned with five epochs")
    evaluate_librispeech(model_path="models/whisper-tiny-librispeech-epoch-4.pth")
    print("-" * 100)

    print(
        "Evaluating LibriSpeech dataset finetuned with LoRA and KL divergence five epochs"
    )
    evaluate_librispeech(model_path="models/whisper-tiny-lora-kd")
    print("-" * 100)

    print(
        "Evaluating LibriSpeech dataset finetuned with LoRA, KL divergence, and EWC combined"
    )
    evaluate_librispeech(model_path="models/whisper-tiny-lora-kd-ewc-2")
    print("-" * 100)


if __name__ == "__main__":
    main()
