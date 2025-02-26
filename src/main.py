from trainer import train, train_lora_kd
from utils import evaluate_out_of_sample, evaluate_single_datapoint


def main():
    # Uncomment the training method you want to use
    # print("Training with standard fine-tuning...")
    # train(num_epochs=5)

    # print("Training with LoRA and KL divergence...")
    # train_lora_kd(num_epochs=5)

    # Comment out the following line to evaluate the single datapoint
    # evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-lora-kd")

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

    print("Evaluating model finetuned with LoRA and KL divergence one epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd/checkpoint-1784")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA and KL divergence two epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd/checkpoint-3568")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA and KL divergence three epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd/checkpoint-5352")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA and KL divergence four epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd/checkpoint-7136")
    print("-" * 100)

    print("Evaluating model finetuned with LoRA and KL divergence five epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd/checkpoint-8920")

    print("Evaluating model finetuned with LoRA and KL divergence five epochs again")
    evaluate_out_of_sample(model_path="models/whisper-tiny-lora-kd")
    print("-" * 100)


if __name__ == "__main__":
    main()
