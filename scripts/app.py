import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel
import numpy as np
from tempfile import NamedTemporaryFile

# ----------------------------
# Setup
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Whisper encoder and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperModel.from_pretrained("openai/whisper-base").to(DEVICE)
whisper_model.eval()
class ImprovedWhisperMLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=8):
        super(ImprovedWhisperMLP, self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),                     # Normalize input
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

classifier = ImprovedWhisperMLP()
classifier.load_state_dict(torch.load(r"C:\Users\Raihan\OneDrive\Desktop\OPEN_PROJECT_AUDIO\scripts\whisper_model.pth", map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

LABELS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# ----------------------------
# Predict Function
# ----------------------------
def predict_emotion(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():
        outputs = whisper_model.encoder(input_features=input_features)
        last_hidden_state = outputs.last_hidden_state
        embedding = last_hidden_state.mean(dim=1)

        pred = classifier(embedding)
        pred_class = torch.argmax(F.softmax(pred, dim=1), dim=1).item()
        return LABELS[pred_class]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎙️ Audio Emotion Recognition")
st.write("Upload a `.wav` file and get the predicted emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("Analyzing emotion..."):
        emotion = predict_emotion(tmp_path)
    
    st.success(f"**Predicted Emotion:** {emotion}")
