import torch
import torchaudio

from config import MODEL, PROCESSOR, DEVICE


# Config
DATAPATH = "./data/hello-izaak.wav"

# Load the model
MODEL.to(DEVICE)
MODEL.eval()

# Load and process the audio
waveform, sample_rate = torchaudio.load(DATAPATH)
# Convert to mono if stereo and resample to 16kHz
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
waveform_np = waveform.squeeze().numpy()

# Process the audio
features = PROCESSOR(
    audio=waveform_np,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)
input_features = features.input_features.to(DEVICE)
attention_mask = features.attention_mask.to(DEVICE)

# Transcribe the audio
predicted_ids = MODEL.generate(
    input_features=input_features,
    attention_mask=attention_mask,
    language="en",
    task="transcribe",
)

predicted_transcript = PROCESSOR.decode(predicted_ids[0], skip_special_tokens=True)
print("Whisper transcription:", predicted_transcript)

# Tokenize the ground truth
ground_truth = "Hello, my name is Izaak"
tokenized = PROCESSOR(text=ground_truth, return_tensors="pt")
target_tensor = tokenized.input_ids.to(DEVICE)

# Create input and target sequences
# For decoder input, we need all tokens except the last one
# For labels, we need all tokens except the first one (which is bos_token)
input_tks = target_tensor[:, :-1]  # Remove last token for decoder input
target_tks = target_tensor[:, 1:]  # Remove first token (bos) for labels

# Define the optimizer and loss function
optimizer = torch.optim.Adam(MODEL.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
MODEL.train()
for step in range(20):
    # forward pass
    output = MODEL(
        input_features=input_features,
        attention_mask=attention_mask,
        decoder_input_ids=input_tks,
        labels=target_tks,  # Use full target sequence as labels
    )

    loss = output.loss  # Use the built-in loss calculation

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")

# Test with the model
MODEL.eval()
predicted_ids = MODEL.generate(
    input_features=input_features,
    attention_mask=attention_mask,
    language="en",
    task="transcribe",
)

predicted_transcript = PROCESSOR.decode(predicted_ids[0], skip_special_tokens=True)
print("New transcription:", predicted_transcript)
