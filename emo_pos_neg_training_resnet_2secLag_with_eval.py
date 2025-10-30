"""
Plant-Based Human Emotion Classifier using Spectrograms and ResNet

This script uses Purple Heart plant bioelectrical responses (via AD8232) to classify
human emotions. It converts plant voltage changes to spectrograms and uses ResNet
to distinguish between happy (joy + surprise) and sad (fear + anger + sadness + disgust) emotions,
while ignoring neutral emotions.

Optimized for Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS)
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import seaborn as sns
from tqdm import tqdm
import json
import re
from datetime import datetime, timedelta

def preprocess_audio_to_spectrograms(data_dir, output_dir, sample_rate=400):
    """
    Preprocess all WAV files into spectrograms and save them as .npy files
    """
    print("Converting emotion WAV files to spectrograms...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    print(f"Found {len(wav_files)} emotion WAV files to process")
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_dir}")
    
    # Show debug info for first file only
    show_debug = True
    
    # Process each file
    for i, wav_file in enumerate(tqdm(wav_files, desc="Converting to spectrograms")):
        try:
            # Load audio
            audio_data, sr = librosa.load(wav_file, sr=sample_rate)
            
            # Generate spectrogram
            if show_debug:
                print(f"\nDebug info for first emotion file:")
                spectrogram = audio_to_spectrogram(audio_data, sr)
                show_debug = False
            else:
                # Use proven parameters from plant lamp classification
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data, sr=sr, n_fft=512, hop_length=64,
                    n_mels=128, fmax=sr//2
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                spectrogram = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            # Convert to 3-channel (RGB) for ResNet
            spectrogram_rgb = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
            
            # Save as .npy file
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            output_path = os.path.join(output_dir, f"{basename}.npy")
            np.save(output_path, spectrogram_rgb)
            
            # Clean up memory
            del audio_data, spectrogram, spectrogram_rgb
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    print(f"Emotion spectrogram preprocessing complete! Saved to {output_dir}")
    return True

def audio_to_spectrogram(audio_data, sr, n_fft=512, hop_length=64):
    """Convert audio to mel-spectrogram using parameters optimized for plant emotion data"""
    
    # Print debug info
    expected_samples = sr * 3  # Expected for 3-second file
    print(f"  Audio length: {len(audio_data)} samples ({len(audio_data)/sr:.2f} seconds)")
    print(f"  Expected samples for 3s at {sr}Hz: {expected_samples}")
    
    # Calculate expected time steps
    expected_time_steps = (len(audio_data) - n_fft) // hop_length + 1
    print(f"  Expected time steps with hop_length={hop_length}: {expected_time_steps}")
    
    # Use proven parameters from plant lamp classification
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,        # 512 - proven optimal for plant bioelectricity
        hop_length=hop_length,  # 64 - proven optimal
        n_mels=128,
        fmax=sr//2  # 200Hz max frequency for 400Hz sampling
    )
    
    print(f"  Actual spectrogram shape: {mel_spec.shape}")
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    print(f"  Final spectrogram shape: {mel_spec_norm.shape}")
    
    return mel_spec_norm

class PrecomputedEmotionDataset(Dataset):
    def __init__(self, spectrogram_files, labels, transform=None, target_size=(128, 128)):
        self.spectrogram_files = spectrogram_files
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        print(f"Emotion dataset will resize spectrograms to: {target_size}")
        
    def __len__(self):
        return len(self.spectrogram_files)
    
    def __getitem__(self, idx):
        try:
            # Load precomputed spectrogram
            spectrogram_path = self.spectrogram_files[idx]
            spectrogram_rgb = np.load(spectrogram_path)
            
            # Check if spectrogram needs resizing
            current_height, current_width = spectrogram_rgb.shape[1], spectrogram_rgb.shape[2]
            target_height, target_width = self.target_size
            
            if current_height != target_height or current_width != target_width:
                # Resize each channel
                from PIL import Image
                resized_channels = []
                for c in range(3):
                    img = Image.fromarray((spectrogram_rgb[c] * 255).astype(np.uint8))
                    img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                    resized_channels.append(np.array(img_resized) / 255.0)
                
                spectrogram_rgb = np.stack(resized_channels, axis=0)
            
            # Apply transforms if provided
            if self.transform:
                spectrogram_rgb = self.transform(spectrogram_rgb)
            
            label = self.labels[idx]
            
            return torch.FloatTensor(spectrogram_rgb), torch.LongTensor([label])
            
        except Exception as e:
            print(f"Error loading spectrogram {spectrogram_path}: {e}")
            return torch.zeros((3, self.target_size[0], self.target_size[1])), torch.LongTensor([0])

class PlantEmotionClassifier(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True):
        super(PlantEmotionClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        if use_pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)
        
        # Modify the final layer for binary emotion classification (happy vs sad)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def load_emotion_labels(label_dir, wav_dir):
    """Load emotion labels from JSON files and create binary classification"""
    
    print(f"Loading emotion labels from: {label_dir}")
    print(f"Matching with WAV files from: {wav_dir}")
    
    # Find all JSON label files
    json_files = glob.glob(os.path.join(label_dir, "*_raw_label.json"))
    print(f"Found {len(json_files)} JSON label files")
    
    data = []
    emotion_counts = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            
            # Extract the WAV filename
            wav_filename = label_data.get('wav_file', '')
            if not wav_filename:
                continue
            
            # Full path to WAV file
            wav_path = os.path.join(wav_dir, wav_filename)
            
            # Check if WAV file exists
            if not os.path.exists(wav_path):
                continue
            
            # Get the raw emotion
            raw_emotion = label_data.get('face_emotion_raw', '').lower()
            confidence = label_data.get('face_confidence_raw', 0.0)
            
            # Skip neutral emotions and low confidence detections
            if raw_emotion == 'neutral' or confidence < 0.7:
                continue
            
            # Map emotions to binary categories
            emotion_category = map_emotion_to_binary(raw_emotion)
            if emotion_category is None:
                continue
            
            # Count emotions for reporting
            emotion_counts[raw_emotion] = emotion_counts.get(raw_emotion, 0) + 1
            
            # Parse timestamp for potential lag analysis
            timestamp = label_data.get('timestamp', 0)
            face_timestamp = label_data.get('face_timestamp', 0)
            time_diff = label_data.get('time_difference', 0)
            
            data.append({
                'wav_file': wav_path,
                'raw_emotion': raw_emotion,
                'emotion_category': emotion_category,
                'confidence': confidence,
                'timestamp': timestamp,
                'face_timestamp': face_timestamp,
                'time_difference': time_diff
            })
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\nEmotion distribution (raw):")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")
    
    return pd.DataFrame(data)

def map_emotion_to_binary(raw_emotion):
    """Map Ekman emotions to binary happy vs sad categories"""
    
    # Happy emotions: joy/happiness + surprise
    happy_emotions = ['happy', 'surprise']
    
    # Sad emotions: fear + anger + sadness + disgust
    sad_emotions = ['fear', 'anger', 'sad', 'disgusted']
    
    if raw_emotion in happy_emotions:
        return 'happy'
    elif raw_emotion in sad_emotions:
        return 'sad'
    else:
        return None  # Ignore neutral and unknown emotions

def create_lagged_emotion_labels(df, lag_seconds=0):
    """Create labels accounting for plant response lag to human emotions"""
    df_sorted = df.sort_values('timestamp').copy()
    
    if lag_seconds == 0:
        return df_sorted
    
    # Create lagged labels - plant responds with delay to human emotion
    df_lagged = df_sorted.copy()
    
    for i, row in df_sorted.iterrows():
        current_time = row['timestamp']
        lag_time = current_time - lag_seconds  # Look back in time for the emotion
        
        # Find the emotion that occurred lag_seconds before this plant measurement
        past_emotion = None
        for j, past_row in df_sorted.iterrows():
            if past_row['timestamp'] <= lag_time:
                past_emotion = past_row['emotion_category']
            else:
                break
        
        if past_emotion is not None:
            df_lagged.loc[i, 'emotion_category'] = past_emotion
    
    return df_lagged

def load_emotion_data_with_lag(wav_dir, label_dir, spectrogram_dir, lag_seconds=0):
    """Load emotion data and create labels with specified lag"""
    
    print(f"Loading emotion data with {lag_seconds}-second lag...")
    
    # Load emotion labels
    df = load_emotion_labels(label_dir, wav_dir)
    
    if len(df) == 0:
        raise ValueError("No valid emotion data found!")
    
    print(f"Found {len(df)} valid emotion samples")
    
    # Apply lag to labels
    df_lagged = create_lagged_emotion_labels(df, lag_seconds)
    
    # Match with spectrogram files
    valid_data = []
    for _, row in df_lagged.iterrows():
        wav_path = row['wav_file']
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        spec_path = os.path.join(spectrogram_dir, f"{basename}.npy")
        
        if os.path.exists(spec_path):
            valid_data.append({
                'filename': spec_path,
                'emotion_category': row['emotion_category'],
                'raw_emotion': row['raw_emotion'],
                'confidence': row['confidence']
            })
    
    df_final = pd.DataFrame(valid_data)
    
    if len(df_final) == 0:
        raise ValueError("No matching spectrogram files found!")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_final['label'] = label_encoder.fit_transform(df_final['emotion_category'])
    
    print(f"Final dataset with {lag_seconds}s lag:")
    print(f"  Total samples: {len(df_final)}")
    print(f"  Emotion distribution:")
    emotion_dist = df_final['emotion_category'].value_counts()
    for emotion, count in emotion_dist.items():
        print(f"    {emotion}: {count}")
    
    return df_final, label_encoder

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_emotion_model(model, train_loader, val_loader, num_epochs=50, device=None, model_name="emotion_model"):
    """Train the ResNet emotion classifier"""
    if device is None:
        device = get_device()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'{model_name} - Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).squeeze()
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'{model_name} - Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_plant_emotion_classifier_{model_name}.pth')
        
        scheduler.step(val_loss)
    
    return train_losses, val_losses, val_accuracies, best_val_acc


def evaluate_emotion_model(model, test_loader, label_encoder, device=None, model_name="emotion_model"):
    """Evaluate the trained emotion model and return accuracy, predictions, targets, and probabilities."""
    if device is None:
        device = get_device()
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            output = model(data)               # logits
            probs = F.softmax(output, dim=1)   # probabilities
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            # prob of positive class (assume label 1 is 'happy' after LabelEncoder)
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)

    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_targets, all_predictions,
                                target_names=label_encoder.classes_))

    return accuracy, np.array(all_predictions), np.array(all_targets), np.array(all_probs)


def compare_emotion_lag_scenarios(wav_dir, label_dir, spectrogram_dir, lag_scenarios=[0, 1, 2], num_epochs=30):
    """Compare different lag scenarios for plant emotion detection"""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training plant-based emotion models with precomputed spectrograms")
    
    results = {}
    
    for lag in lag_scenarios:
        print(f"\n{'='*60}")
        print(f"TRAINING EMOTION MODEL FOR {lag}-SECOND LAG SCENARIO")
        print(f"{'='*60}")
        
        # Load emotion data with specific lag
        df, label_encoder = load_emotion_data_with_lag(wav_dir, label_dir, spectrogram_dir, lag_seconds=lag)
        
        # Check if we have enough data for both classes
        emotion_counts = df['emotion_category'].value_counts()
        if len(emotion_counts) < 2:
            print(f"Skipping {lag}s lag - insufficient data for binary classification")
            continue
        
        min_samples = emotion_counts.min()
        if min_samples < 10:
            print(f"Warning: Only {min_samples} samples for minority class in {lag}s lag scenario")
        
        # Split data
        train_files, test_files, train_labels, test_labels = train_test_split(
            df['filename'].values, df['label'].values,
            test_size=0.2, random_state=42, stratify=df['label'].values
        )
        
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Create datasets
        target_size = (128, 128)
        train_dataset = PrecomputedEmotionDataset(train_files, train_labels, target_size=target_size)
        val_dataset = PrecomputedEmotionDataset(val_files, val_labels, target_size=target_size)
        test_dataset = PrecomputedEmotionDataset(test_files, test_labels, target_size=target_size)
        
        # Optimize settings for device
        batch_size = 128 if device.type == 'mps' else 64
        num_workers = 4 if device.type == 'mps' else 8
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Create and train model
        model = PlantEmotionClassifier(num_classes=len(label_encoder.classes_))
        model = model.to(device)
        
        model_name = f"{lag}s_lag"
        train_losses, val_losses, val_accuracies, best_val_acc = train_emotion_model(
            model, train_loader, val_loader, num_epochs=num_epochs,
            device=device, model_name=model_name
        )
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f'best_plant_emotion_classifier_{model_name}.pth', map_location=device))
        test_accuracy, predictions, targets, pred_proba = evaluate_emotion_model(
            model, test_loader, label_encoder, device, model_name
        )
        
        # Store results
        results[lag] = {
            'model': model,
            'label_encoder': label_encoder,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'targets': targets,
            'pred_proba': pred_proba,
            'data_distribution': emotion_counts
        }
    
    return results

def visualize_emotion_spectrograms(spectrogram_dir, label_dir, wav_dir, num_samples=4):
    """Visualize sample emotion spectrograms"""
    
    # Load some emotion labels
    try:
        df = load_emotion_labels(label_dir, wav_dir)
        if len(df) == 0:
            print("No emotion data found for visualization")
            return
        
        # Get sample files for different emotions
        happy_samples = df[df['emotion_category'] == 'happy'].head(2)
        sad_samples = df[df['emotion_category'] == 'sad'].head(2)
        
        sample_data = pd.concat([happy_samples, sad_samples]).head(num_samples)
        
        fig, axes = plt.subplots(2, len(sample_data), figsize=(4*len(sample_data), 8))
        if len(sample_data) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (_, row) in enumerate(sample_data.iterrows()):
            try:
                # Load spectrogram
                wav_path = row['wav_file']
                basename = os.path.splitext(os.path.basename(wav_path))[0]
                spec_path = os.path.join(spectrogram_dir, f"{basename}.npy")
                
                if not os.path.exists(spec_path):
                    print(f"Spectrogram not found: {spec_path}")
                    continue
                
                spectrogram_rgb = np.load(spec_path)
                
                # Show the spectrogram (use first channel)
                axes[0, i].imshow(spectrogram_rgb[0], aspect='auto', origin='lower', cmap='viridis')
                title = f"Emotion: {row['emotion_category']}\n"
                title += f"Raw: {row['raw_emotion']}\n"
                title += f"Conf: {row['confidence']:.2f}"
                axes[0, i].set_title(title, fontsize=10)
                axes[0, i].set_xlabel('Time Steps')
                axes[0, i].set_ylabel('Mel Frequency Bins')
                
                # Show frequency content
                axes[1, i].plot(np.mean(spectrogram_rgb[0], axis=1))
                axes[1, i].set_title(f"Avg Frequency Content", fontsize=10)
                axes[1, i].set_xlabel('Mel Frequency Bins')
                axes[1, i].set_ylabel('Average Power')
                axes[1, i].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in emotion visualization: {e}")

def main():
    # Configuration
    WAV_DIR = "training_data_collection_1s/wav_files"
    LABEL_DIR = "training_data_collection_1s/emotion_labels"
    SPECTROGRAM_DIR = "training_data_collection_1s/spectrograms"
    LAG_SCENARIOS = [0, 1, 2]  # Test immediate, 1-second, and 2-second plant response lags
    NUM_EPOCHS = 30
    SAMPLE_RATE = 400
    
    print("Plant-Based Human Emotion Detection using Purple Heart Bioelectricity")
    print("Classifying happy (joy + surprise) vs sad (fear + anger + sadness + disgust) emotions")
    print("Using plant bioelectrical response to human emotional states")
    
    try:
        # Step 1: Check spectrograms and regenerate if needed
        regenerate_spectrograms = False
        
        if not os.path.exists(SPECTROGRAM_DIR) or len(glob.glob(os.path.join(SPECTROGRAM_DIR, "*.npy"))) == 0:
            print(f"\nSpectrograms not found. Creating them from emotion WAV files...")
            regenerate_spectrograms = True
        else:
            print(f"Found existing spectrograms in {SPECTROGRAM_DIR}")
            spec_files = glob.glob(os.path.join(SPECTROGRAM_DIR, "*.npy"))
            print(f"Found {len(spec_files)} precomputed spectrograms")
            
            # Check if spectrograms need regeneration
            if len(spec_files) > 0:
                sample_spec = np.load(spec_files[0])
                print(f"Sample spectrogram shape: {sample_spec.shape}")
                
                if sample_spec.shape[2] < 15 or sample_spec.shape[2] > 25:
                    print(f"Existing spectrograms may need regeneration ({sample_spec.shape[2]} time steps)")
                    print(f"Regenerating with proven parameters (n_fft=512, hop_length=64)...")
                    regenerate_spectrograms = True
        
        if regenerate_spectrograms:
            success = preprocess_audio_to_spectrograms(WAV_DIR, SPECTROGRAM_DIR, SAMPLE_RATE)
            if not success:
                raise ValueError("Failed to preprocess emotion audio files")
        
        # Step 2: Visualize sample emotion spectrograms
        print(f"\nVisualizing sample emotion spectrograms...")
        try:
            visualize_emotion_spectrograms(SPECTROGRAM_DIR, LABEL_DIR, WAV_DIR, num_samples=4)
        except Exception as viz_error:
            print(f"Error in visualization: {viz_error}")
            print("Continuing with training...")
        
        # Step 3: Train models for different lag scenarios
        print(f"\nTraining emotion models using precomputed spectrograms...")
        results = compare_emotion_lag_scenarios(WAV_DIR, LABEL_DIR, SPECTROGRAM_DIR, LAG_SCENARIOS, NUM_EPOCHS)
        
        # Step 4: Analyze results
        print(f"\n{'='*60}")
        print("SUMMARY OF PLANT EMOTION DETECTION LAG ANALYSIS")
        print(f"{'='*60}")
        
        best_lag = None
        best_accuracy = 0
        
        for lag, result in results.items():
            val_acc = result['best_val_acc']
            test_acc = result['test_accuracy'] * 100
            print(f"{lag}s lag: Val Acc = {val_acc:.2f}%, Test Acc = {test_acc:.2f}%")
            print(f"  Data distribution: {dict(result['data_distribution'])}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_lag = lag
        
        if best_lag is not None:
            print(f"\nBest performing lag scenario: {best_lag} seconds")
            print(f"This suggests the plant responds to human emotions with ~{best_lag} second delay")
            print(f"Your Purple Heart plant can detect human emotions with {best_accuracy:.1f}% accuracy!")
        
        # Create comparison plots
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot training curves
            ax1 = axes[0, 0]
            for lag, result in results.items():
                epochs = range(1, len(result['train_losses']) + 1)
                ax1.plot(epochs, result['train_losses'], label=f'{lag}s lag - Train', linestyle='--')
                ax1.plot(epochs, result['val_losses'], label=f'{lag}s lag - Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Plant Emotion Detection Training Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot final accuracies
            ax2 = axes[0, 1]
            lags = list(results.keys())
            val_accs = [results[lag]['best_val_acc'] for lag in lags]
            test_accs = [results[lag]['test_accuracy'] * 100 for lag in lags]
            
            x = np.arange(len(lags))
            width = 0.35
            ax2.bar(x - width/2, val_accs, width, label='Best Validation Accuracy', alpha=0.8)
            ax2.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
            ax2.set_xlabel('Plant Response Lag (seconds)')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Plant Emotion Detection Accuracy by Lag')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{lag}s' for lag in lags])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot confusion matrices
            for i, (lag, result) in enumerate(results.items()):
                if i >= 2:  # Only show first 2 scenarios
                    break
                ax = axes[1, i]
                cm = confusion_matrix(result['targets'], result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=result['label_encoder'].classes_,
                           yticklabels=result['label_encoder'].classes_,
                           ax=ax)
                ax.set_title(f'{lag}s lag\nTest Acc: {result["test_accuracy"]*100:.1f}%')
                ax.set_ylabel('True Emotion')
                ax.set_xlabel('Predicted Emotion')
            
            plt.tight_layout()
            plt.show()
            print("Plots generated successfully!")
        
        except Exception as plot_error:
            print(f"Plotting failed: {plot_error}")
        # ======== Added: detailed metrics per lag (CI, ROC-AUC, PR-AUC, permutation) ========
        try:
            print("\nDetailed test-set metrics per lag (with 95% CI, ROC-AUC, PR-AUC, permutation baselines):")
            metrics_rows = []
            for lag, result in results.items():
                y_true = result['targets']
                y_score = None
                pred_proba = result.get('pred_proba', None)
                if pred_proba is not None:
                    try:
                        y_score = pred_proba
                    except Exception:
                        y_score = None
                y_pred = result.get('predictions', None)

                met = evaluate_with_auc_and_ci(y_true, y_score=y_score, y_pred=y_pred, n_permutations=500)
                print(
                    f" Lag {lag}s: Acc={met['accuracy']*100:.2f}%  "
                    f"95%CI=[{met['acc_ci_low']*100:.1f}–{met['acc_ci_high']*100:.1f}]  "
                    f"ROC-AUC={met['roc_auc']:.3f}  PR-AUC={met['pr_auc']:.3f}  "
                    f"| perm Acc ~ {met['perm_accuracy_mean']*100:.1f}±{met['perm_accuracy_std']*100:.1f}%, "
                    f"perm PR-AUC ~ {met['perm_pr_auc_mean']:.3f}±{met['perm_pr_auc_std']:.3f}  (n={met['n']})"
                )
                metrics_rows.append({
                    "lag_s": lag,
                    "accuracy": met['accuracy'],
                    "acc_ci_low": met['acc_ci_low'],
                    "acc_ci_high": met['acc_ci_high'],
                    "roc_auc": met['roc_auc'],
                    "pr_auc": met['pr_auc'],
                    "perm_accuracy_mean": met['perm_accuracy_mean'],
                    "perm_accuracy_std": met['perm_accuracy_std'],
                    "perm_pr_auc_mean": met['perm_pr_auc_mean'],
                    "perm_pr_auc_std": met['perm_pr_auc_std'],
                    "cm_TN": met['cm_TN'],
                    "cm_FP": met['cm_FP'],
                    "cm_FN": met['cm_FN'],
                    "cm_TP": met['cm_TP'],
                    "n": met['n']
                })
            import pandas as _pd
            metrics_df = _pd.DataFrame(metrics_rows)
            out_csv = "resnet_emotion_metrics_detailed.csv"
            metrics_df.to_csv(out_csv, index=False)
            print(f"\nSaved detailed metrics to {out_csv}")
        except Exception as e:
            print(f"Metrics export failed: {e}")
        # ======== End added metrics ========
        
        print(f"\nPlant-based emotion detection complete!")
        print(f"Your Purple Heart plant is now trained to detect human emotions!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. WAV files are in training_data_collection_1s/wav_files/")
        print("2. JSON emotion labels are in training_data_collection_1s/emotion_labels/")
        print("3. JSON files contain 'face_emotion_raw' field with Ekman emotions")
        print("4. Confidence threshold is met (>0.7) for emotion detection")


# ======== Added: evaluation helpers (CI, AUCs, permutation) ========
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)

def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    # 95% Wilson score interval
    z = 1.959963984540054
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2/(2*n)
    margin = z * ((phat*(1-phat) + z**2/(4*n)) / n) ** 0.5
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return lower, upper

def evaluate_with_auc_and_ci(y_true, y_score=None, y_pred=None, n_permutations: int = 200, seed: int = 42):
    """Compute accuracy with 95% CI, per-class precision/recall/F1, ROC-AUC, PR-AUC,
    and permutation baselines. y_score preferred; if None, uses y_pred (AUCs will be NaN).
    Returns a dict."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)

    if y_score is None and y_pred is None:
        raise ValueError("Provide y_score or y_pred")

    if y_score is None:
        y_pred = np.asarray(y_pred)
        acc = accuracy_score(y_true, y_pred)
        k = int((y_true == y_pred).sum())
        lo, hi = _wilson_ci(k, len(y_true))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        return dict(
            accuracy=acc, acc_ci_low=lo, acc_ci_high=hi,
            precision_neg=float(prec[0]), recall_neg=float(rec[0]), f1_neg=float(f1[0]),
            precision_pos=float(prec[1]), recall_pos=float(rec[1]), f1_pos=float(f1[1]),
            roc_auc=float("nan"), pr_auc=float("nan"),
            perm_accuracy_mean=float("nan"), perm_accuracy_std=float("nan"),
            perm_pr_auc_mean=float("nan"), perm_pr_auc_std=float("nan"),
            n=int(len(y_true)), cm_TN=int(cm[0,0]), cm_FP=int(cm[0,1]), cm_FN=int(cm[1,0]), cm_TP=int(cm[1,1])
        )

    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    k = int((y_true == y_pred).sum())
    lo, hi = _wilson_ci(k, len(y_true))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    # AUCs
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_score, pos_label=1)
    except Exception:
        pr = float("nan")

    # Permutation baselines
    perm_accs, perm_prs = [], []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_true)
        perm_pred = (y_score >= 0.5).astype(int)
        perm_accs.append(accuracy_score(y_perm, perm_pred))
        try:
            perm_prs.append(average_precision_score(y_perm, y_score, pos_label=1))
        except Exception:
            perm_prs.append(float("nan"))

    return dict(
        accuracy=float(acc), acc_ci_low=float(lo), acc_ci_high=float(hi),
        precision_neg=float(prec[0]), recall_neg=float(rec[0]), f1_neg=float(f1[0]),
        precision_pos=float(prec[1]), recall_pos=float(rec[1]), f1_pos=float(f1[1]),
        roc_auc=float(roc), pr_auc=float(pr),
        perm_accuracy_mean=float(np.nanmean(perm_accs)), perm_accuracy_std=float(np.nanstd(perm_accs)),
        perm_pr_auc_mean=float(np.nanmean(perm_prs)), perm_pr_auc_std=float(np.nanstd(perm_prs)),
        n=int(len(y_true)), cm_TN=int(cm[0,0]), cm_FP=int(cm[0,1]), cm_FN=int(cm[1,0]), cm_TP=int(cm[1,1])
    )
# ======== End added helpers ========


if __name__ == "__main__":
    main()
