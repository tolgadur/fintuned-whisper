from torch.utils.data import DataLoader
import torch
import tqdm
from torch.amp import GradScaler, autocast
import os

from config import MODEL, DEVICE, PROCESSOR
from dataset import LibriSpeechDataset


def train(
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
):
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load the dataset
    dataset = LibriSpeechDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} samples")

    # Load the model
    model = MODEL.to(DEVICE)
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

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

            # Forward pass with mixed precision
            with autocast(device_type=DEVICE, enabled=use_amp):
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            # Backward pass with gradient scaling
            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            print(f"Loss: {loss.item()}")

        # Print epoch summary
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {epoch_loss/len(dataloader):.4f}"
        )

        # Save checkpoint after each epoch
        torch.save(
            model.state_dict(), f"models/whisper-tiny-librispeech-epoch-{epoch+1}.pth"
        )

    # Save the final model
    torch.save(model.state_dict(), "models/whisper-tiny-librispeech.pth")

    return model
