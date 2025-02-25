from torch.utils.data import DataLoader
import torch
import tqdm

from config import MODEL, DEVICE, PROCESSOR
from dataset import LibriSpeechDataset


def train(batch_size: int = 128, num_epochs: int = 10, learning_rate: float = 1e-4):
    # Load the dataset
    dataset = LibriSpeechDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} samples")

    # Load the model
    model = MODEL.to(DEVICE)
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        for i, batch in enumerate(
            tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            input_features, attention_mask, transcript = batch

            # Move batch to device
            input_features = input_features.squeeze(1).to(DEVICE)
            attention_mask = attention_mask.squeeze(1).to(DEVICE)

            # Tokenize the transcript
            tokenized = PROCESSOR(text=transcript, return_tensors="pt", padding=True)
            labels = tokenized.input_ids.to(DEVICE)

            # Forward pass
            outputs = model(
                input_features=input_features,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "models/whisper-tiny-librispeech.pth")

    return model
