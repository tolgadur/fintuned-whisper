from dataset import LibriSpeechDataset


def main():
    dataset = LibriSpeechDataset()
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
