from trainer import train
from utils import evaluate_out_of_sample


def main():
    # print("Training...")
    # train()

    print("Evaluating...")
    print("Evaluating regular non-fined tuned model")
    evaluate_out_of_sample()
    print("-" * 100)

    print("Evaluating model finetuned with one epoch")
    evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech.pth")
    print("-" * 100)

    print("Evaluating model finetuned with five epochs")
    evaluate_out_of_sample(model_path="models/whisper-tiny-librispeech-epoch-4.pth")
    print("-" * 100)


if __name__ == "__main__":
    main()
