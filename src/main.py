from trainer import train
from utils import evaluate_single_datapoint, evaluate_librispeech


def main():
    # print("Training...")
    # train()

    print("Evaluating...")
    evaluate_librispeech(model_path="models/whisper-tiny-librispeech.pth")
    # evaluate_single_datapoint(model_path="models/whisper-tiny-librispeech.pth")


if __name__ == "__main__":
    main()
