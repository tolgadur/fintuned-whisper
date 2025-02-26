from trainer import train, train_lora_kd
from utils import evaluate_out_of_sample, evaluate_single_datapoint


def main():
    # Uncomment the training method you want to use
    # print("Training with standard fine-tuning...")
    # train(num_epochs=10, batch_size=1)

    # print("Training with LoRA and KL divergence...")
    train_lora_kd(num_epochs=10)

    # Comment out the following line to evaluate the single datapoint
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd")

    # print("Evaluating...")
    # print("Evaluating regular non-fined tuned model")
    # evaluate_out_of_sample()
    # print("-" * 100)

    # print("Evaluating model finetuned with one epoch")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-1.pth")
    # print("-" * 100)

    # print("Evaluating model finetuned with five epochs")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-4.pth")
    # print("-" * 100)

    # Uncomment to evaluate LoRA KD model if it exists
    # print("Evaluating model finetuned with LoRA and KL divergence")
    # evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd")
    # print("-" * 100)


if __name__ == "__main__":
    main()
