import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel
from moviepy import VideoFileClip
import os
import tempfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# ----------------------------
# Load Model and Processor
# ----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ----------------------------
# Emotion labels (edit if needed)
# ----------------------------
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# ----------------------------
# Function to extract audio
# ----------------------------
def extract_audio_from_video(video_path, output_wav_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_wav_path, fps=16000, codec='pcm_s16le', nbytes=2, ffmpeg_params=["-ac", "1"])
    clip.close()

# ----------------------------
# Function to predict emotion
# ----------------------------
def predict_emotion_from_audio(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # mono

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
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # 🔽 Set your video path here
    video_path = r"C:\Users\Raihan\OneDrive\Desktop\01-02-03-01-02-01-20.mp4" # <<< change this line

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio_from_video(video_path, audio_path)
        emotion = predict_emotion_from_audio(audio_path)
        print(f"Predicted Emotion: {emotion}")
    os.remove(audio_path)
