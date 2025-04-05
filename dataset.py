# audio_deepfake_detection.py

import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path setup
DATASET_PATH = "D:/dataset"
REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

# Utils for audio processing
def get_mel_spectrogram(waveform, sr):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=64)(waveform)
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram

# Dataset class
class VoiceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".wav")]
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".wav")]
        self.files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

# CNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAudioClassifier(nn.Module):
    def __init__(self):
        super(CNNAudioClassifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # -> [B, 16, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 16, H/2, W/2]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # -> [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2)   # -> [B, 32, H/4, W/4]
        )

        # 🔍 Dummy input to calculate output shape
        dummy_input = torch.zeros(1, 1, 64, 2206)  # Same as your spectrogram shape
        out = self.cnn(dummy_input)
        self.flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary: real vs fake
        )

    def forward(self, x):
        if x.dim() == 5:  # Shape: [B, 1, 1, H, W]
            x = x.squeeze(1)  # -> [B, 1, H, W]
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



# RawNet Model
class RawNet(nn.Module):
    def __init__(self):
        super(RawNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Dynamically calculate flatten dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 22050)  # Simulate 1-second audio at 22kHz
            dummy_output = self.conv(dummy_input)
            flatten_dim = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary classification: real vs fake
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # If shape is [B, 1, 1, T] -> [B, 1, T]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




# Wav2Vec2 Model Wrapper
class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super(Wav2Vec2Classifier, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x shape: [batch_size, 1, 1, 16000] --> [batch_size, 16000]
        x = x.squeeze(1).squeeze(1)
        features = self.model(x).last_hidden_state.mean(dim=1)  # global average pooling
        return self.classifier(features)



# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluation loop
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            targets.extend(y)
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    return acc, cm

# Main runner
if __name__ == '__main__':
    # Dataset for CNN
    def transform_cnn(wave):
        mel = get_mel_spectrogram(wave, 16000)
        return mel.unsqueeze(0)  # Add channel dim

    trainset = VoiceDataset(REAL_PATH, FAKE_PATH, transform=transform_cnn)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    # CNN
    cnn_model = CNNAudioClassifier().to(device)
    cnn_optim = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        loss = train(cnn_model, trainloader, cnn_optim, criterion)
        print(f"[CNN] Epoch {epoch+1}, Loss: {loss:.4f}")
    acc, cm = evaluate(cnn_model, trainloader)
    print("CNN Accuracy:", acc)
    print("CNN Confusion Matrix:\n", cm)

    # RawNet
    def transform_raw(wave):
        wave = torch.nn.functional.interpolate(wave.unsqueeze(0), size=22050).squeeze(0)
        return wave.unsqueeze(0)  # [1, time]

    rawset = VoiceDataset(REAL_PATH, FAKE_PATH, transform=transform_raw)
    rawloader = DataLoader(rawset, batch_size=8, shuffle=True)

    rawnet = RawNet().to(device)
    raw_optim = torch.optim.Adam(rawnet.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train(rawnet, rawloader, raw_optim, criterion)
        print(f"[RawNet] Epoch {epoch+1}, Loss: {loss:.4f}")
    acc, cm = evaluate(rawnet, rawloader)
    print("RawNet Accuracy:", acc)
    print("RawNet Confusion Matrix:\n", cm)

    # Wav2Vec2
    def transform_wav2vec(wave):
        wave = torch.nn.functional.interpolate(wave.unsqueeze(0), size=16000).squeeze(0)
        return wave.unsqueeze(0)

    wavset = VoiceDataset(REAL_PATH, FAKE_PATH, transform=transform_wav2vec)
    wavloader = DataLoader(wavset, batch_size=4, shuffle=True)

    wav_model = Wav2Vec2Classifier().to(device)
    wav_optim = torch.optim.Adam(wav_model.classifier.parameters(), lr=1e-4)
  # Fine-tune only FC layer

    for epoch in range(3):
        loss = train(wav_model, wavloader, wav_optim, criterion)
        print(f"[Wav2Vec2] Epoch {epoch+1}, Loss: {loss:.4f}")
    acc, cm = evaluate(wav_model, wavloader)
    print("Wav2Vec2 Accuracy:", acc)
    print("Wav2Vec2 Confusion Matrix:\n", cm)
