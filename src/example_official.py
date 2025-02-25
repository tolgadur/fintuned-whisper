import whisper
import torch

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATAPATH = "./data/hello-izaak.wav"
OUT_OF_SAMPLE = "./data/hello-tolga.wav"

# Load the model
model = whisper.load_model("tiny.en")
model.to(DEVICE)
model.eval()

# Load and process the audio
audio = whisper.load_audio(DATAPATH)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
mel = mel.to(DEVICE)
mel = mel.unsqueeze(0)

# Transcribe the audio
result = model.transcribe(audio)
print(result["text"])
print(model.transcribe(OUT_OF_SAMPLE)["text"])

# Tokenize the ground truth
ground_truth = "Hello, my name is Izaak"
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
target_ids = tokenizer.encode(ground_truth)
target_tensor = torch.tensor(target_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

sot_token = torch.tensor([[tokenizer.sot]], dtype=torch.long, device=DEVICE)
eot_token = torch.tensor([[tokenizer.eot]], dtype=torch.long, device=DEVICE)

input_tks = torch.cat([sot_token, target_tensor], dim=-1)
target_tks = torch.cat([target_tensor, eot_token], dim=-1)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
model.train()
for step in range(10):
    # forward pass
    output = model(tokens=input_tks, mel=mel)

    # Reshape output to match target shape
    # shape: [batch_size, sequence_length, vocab_size] -> [batch_size * sequence_length, vocab_size], i.e [1, 9, 51864] -> [9, 51864]
    output = output.reshape(-1, output.size(-1))
    # shape: [batch_size, sequence_length] -> [batch_size * sequence_length], i.e [1, 9] -> [9]
    target_tks = target_tks.reshape(-1)

    loss = criterion(output, target_tks)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")

# Test with the model
model.eval()
result = model.transcribe(audio)
print(result["text"])

# catostrophic forgetting?
print(model.transcribe(OUT_OF_SAMPLE)["text"])
