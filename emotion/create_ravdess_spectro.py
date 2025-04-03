import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define mapping for emotion labels
emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Define directories
input_directory = "ravdess_audio"  # Update this path
output_directory = "spectrograms"  # Update this path

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to create and save spectrograms
def save_spectrogram(audio_path, output_path, emotion, sr=48000):
    y, sr = librosa.load(audio_path, sr=sr)
    
    plt.figure(figsize=(5, 5))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    
    # Create subfolder for the emotion
    emotion_dir = os.path.join(output_path, emotion)
    os.makedirs(emotion_dir, exist_ok=True)  # Ensure directory exists
    
    filename = f"{Path(audio_path).stem}__{emotion}.png"
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Loop through each folder and process audio files
for folder in sorted(os.listdir(input_directory)):
    folder_path = os.path.join(input_directory, folder)
    if not os.path.isdir(folder_path):
        continue
    
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            
            try:
                emotion_code = file.split("-")[2]  # Extract emotion label
                emotion = emotion_labels.get(emotion_code, "unknown")
                
                save_spectrogram(file_path, output_directory, emotion)
                print(f"Processed {file_path}, Emotion: {emotion}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

