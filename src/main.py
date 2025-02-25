from trainer import train
from utils import evaluate_single_datapoint


def main():
    # print("Training...")
    # train()

    print("Evaluating...")
    evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")


if __name__ == "__main__":
    main()
