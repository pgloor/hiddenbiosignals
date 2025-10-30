"""
Plant Lamp Exposure Classifier with Lag Analysis and Precomputed Spectrograms

This script:
1. Preprocesses WAV files into spectrograms (run once)
2. Trains ResNet models using precomputed spectrograms (much faster)
3. Tests both immediate response (0-lag) and delayed response (2-second lag) scenarios

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
import re
from datetime import datetime, timedelta
import pickle
from PIL import Image

def preprocess_audio_to_spectrograms(data_dir, output_dir, sample_rate=400):
    """
    Preprocess all WAV files into spectrograms and save them as .npy files
    This only needs to be run once!
    """
    print("🎵 Preprocessing WAV files into spectrograms...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    print(f"Found {len(wav_files)} WAV files to process")
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {data_dir}")
    
    # Show debug info for first file only
    show_debug = True
    
    # Process each file
    for i, wav_file in enumerate(tqdm(wav_files, desc="Converting to spectrograms")):
        try:
            # Load audio
            audio_data, sr = librosa.load(wav_file, sr=sample_rate)
            
            # Generate spectrogram (show debug for first file only)
            if show_debug:
                print(f"\n📊 Debug info for first file:")
                spectrogram = audio_to_spectrogram(audio_data, sr)
                show_debug = False
            else:
                # Use the same proven parameters
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_data, sr=sr, n_fft=512, hop_length=64,  # Original optimal parameters
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
    
    print(f"✅ Preprocessing complete! Spectrograms saved to {output_dir}")
    return True

def audio_to_spectrogram(audio_data, sr, n_fft=512, hop_length=64):
    """Convert audio to mel-spectrogram using optimal parameters for plant data"""
    
    # Print debug info about audio (only when called for debugging)
    expected_samples = sr * 3  # Expected for 3-second file
    print(f"  Audio length: {len(audio_data)} samples ({len(audio_data)/sr:.2f} seconds)")
    print(f"  Expected samples for 3s at {sr}Hz: {expected_samples}")
    
    # Calculate expected time steps
    expected_time_steps = (len(audio_data) - n_fft) // hop_length + 1
    print(f"  Expected time steps with hop_length={hop_length}: {expected_time_steps}")
    
    # Use proven parameters that give 84% accuracy
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,        # 512 - original value that works well
        hop_length=hop_length,  # 64 - original value that works well
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

class PrecomputedSpectrogramDataset(Dataset):
    def __init__(self, spectrogram_files, labels, transform=None, target_size=(128, 128)):
        self.spectrogram_files = spectrogram_files
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        # Using proven target size that achieves 84% accuracy
        print(f"Dataset will resize spectrograms to: {target_size}")
        
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
                resized_channels = []
                for c in range(3):
                    # Convert to PIL Image for resizing
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
            # Return zeros if file can't be loaded
            return torch.zeros((3, self.target_size[0], self.target_size[1])), torch.LongTensor([0])

class PlantLampClassifier(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True):
        super(PlantLampClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        if use_pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)
        
        # Modify the final layer for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

def parse_filename_with_timestamp(filename):
    """Parse filename to extract lamp status and timestamp"""
    basename = os.path.basename(filename)
    
    # Remove .npy extension if present for parsing
    if basename.endswith('.npy'):
        basename = basename[:-4]
    
    # Updated pattern for your exact format: lamp_off_20250917_122730_928_plant_ad8232_chunk_86_281769
    pattern_detailed = r'(lamp_on|lamp_off)_(\d{8})_(\d{6})_(\d+)_plant_ad8232_chunk_(\d+)_(\d+)'
    match_detailed = re.search(pattern_detailed, basename)
    
    if match_detailed:
        lamp_status = match_detailed.group(1)    # lamp_off/lamp_on
        date_str = match_detailed.group(2)       # 20250917
        time_str = match_detailed.group(3)       # 122730
        milliseconds = match_detailed.group(4)   # 928
        chunk_num = int(match_detailed.group(5)) # 86
        file_id = int(match_detailed.group(6))   # 281769
        
        try:
            # Combine date and time
            datetime_str = f"{date_str}_{time_str}"
            timestamp = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            # Add milliseconds
            timestamp = timestamp + timedelta(milliseconds=int(milliseconds))
            
            return {
                'lamp_status': lamp_status,
                'timestamp': timestamp,
                'chunk_number': chunk_num,
                'milliseconds': int(milliseconds),
                'file_id': file_id,
                'filename': filename
            }
        except Exception as e:
            print(f"Error parsing detailed timestamp from {basename}: {e}")
    
    # Fallback: simpler pattern lamp_on/off_YYYYMMDD_HHMMSS
    pattern_simple = r'(lamp_on|lamp_off)_(\d{8})_(\d{6})'
    match_simple = re.search(pattern_simple, basename)
    
    if match_simple:
        lamp_status = match_simple.group(1)
        date_str = match_simple.group(2)
        time_str = match_simple.group(3)
        try:
            datetime_str = f"{date_str}_{time_str}"
            timestamp = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
        except:
            timestamp = None
    else:
        # Final fallback: extract just lamp status and use file time
        if 'lamp_on' in basename:
            lamp_status = 'lamp_on'
        elif 'lamp_off' in basename:
            lamp_status = 'lamp_off'
        else:
            lamp_status = 'unknown'
        
        # Use file modification time as last resort
        try:
            timestamp = datetime.fromtimestamp(os.path.getmtime(filename))
        except:
            timestamp = None
    
    return {
        'lamp_status': lamp_status,
        'timestamp': timestamp,
        'chunk_number': None,
        'milliseconds': None,
        'file_id': None,
        'filename': filename
    }

def create_lagged_labels(df, lag_seconds=0):
    """Create labels accounting for plant response lag"""
    df_sorted = df.sort_values('timestamp').copy()
    
    if lag_seconds == 0:
        return df_sorted
    
    # Create lagged labels
    df_lagged = df_sorted.copy()
    
    for i, row in df_sorted.iterrows():
        current_time = row['timestamp']
        lag_time = current_time - timedelta(seconds=lag_seconds)
        
        # Find the lamp status at lag_time
        past_status = None
        for j, past_row in df_sorted.iterrows():
            if past_row['timestamp'] <= lag_time:
                past_status = past_row['lamp_status']
            else:
                break
        
        if past_status is not None:
            df_lagged.loc[i, 'lamp_status'] = past_status
    
    return df_lagged

def load_precomputed_data_with_lag(spectrogram_dir, wav_dir, lag_seconds=0):
    """Load precomputed spectrograms and create labels with specified lag"""
    
    print(f"  Looking for spectrograms in: {spectrogram_dir}")
    print(f"  Looking for WAV files in: {wav_dir}")
    
    # Find all spectrogram files
    spec_files = glob.glob(os.path.join(spectrogram_dir, "*.npy"))
    
    if len(spec_files) == 0:
        raise ValueError(f"No .npy spectrogram files found in {spectrogram_dir}")
    
    print(f"  Found {len(spec_files)} spectrogram files")
    
    data = []
    valid_files = 0
    
    for spec_file in spec_files:
        try:
            # Get corresponding WAV file for timestamp
            basename = os.path.splitext(os.path.basename(spec_file))[0]
            wav_file = os.path.join(wav_dir, f"{basename}.wav")
            
            # Parse filename even if WAV doesn't exist (use spectrogram file for timestamp)
            file_info = parse_filename_with_timestamp(wav_file if os.path.exists(wav_file) else spec_file)
            file_info['filename'] = spec_file  # Use spectrogram file path
            
            if file_info['timestamp'] is not None and file_info['lamp_status'] in ['lamp_on', 'lamp_off']:
                data.append(file_info)
                valid_files += 1
            else:
                if valid_files < 5:  # Only show first few debug messages
                    print(f"    Skipping {basename}: timestamp={file_info['timestamp']}, status={file_info['lamp_status']}")
                    
        except Exception as e:
            if valid_files < 5:  # Only show first few debug messages
                print(f"    Error processing {os.path.basename(spec_file)}: {e}")
            continue
    
    print(f"  Valid files with timestamps and labels: {valid_files}")
    
    if valid_files == 0:
        raise ValueError("No valid files with timestamps and lamp status found!")
    
    df = pd.DataFrame(data)
    
    print(f"  Found {len(df)} files with valid timestamps and labels")
    print(f"  Label distribution before lag adjustment:")
    print(f"    {df['lamp_status'].value_counts().to_dict()}")
    
    # Apply lag to labels
    df_lagged = create_lagged_labels(df, lag_seconds)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_lagged['label'] = label_encoder.fit_transform(df_lagged['lamp_status'])
    
    print(f"  Lag: {lag_seconds} seconds")
    print(f"  Label distribution after lag adjustment:")
    print(f"    {df_lagged['lamp_status'].value_counts().to_dict()}")
    
    return df_lagged, label_encoder

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_model(model, train_loader, val_loader, num_epochs=50, device=None, model_name="model"):
    """Train the ResNet classifier"""
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
            torch.save(model.state_dict(), f'best_plant_lamp_classifier_{model_name}.pth')
        
        scheduler.step(val_loss)
    
    return train_losses, val_losses, val_accuracies, best_val_acc


def evaluate_model(model, test_loader, label_encoder, device=None, model_name="lamp_resnet"):
    """Evaluate the trained lamp on/off model; return accuracy, predictions, targets, and probabilities."""
    if device is None:
        device = get_device()
    model.eval()
    all_predictions, all_targets, all_probs = [], [], []

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze()
            output = model(data)                 # logits
            probs = F.softmax(output, dim=1)     # probabilities per class
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            # positive class assumed to be label 1 (lamp ON)
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_targets, all_predictions, target_names=label_encoder.classes_))

    return accuracy, np.array(all_predictions), np.array(all_targets), np.array(all_probs)


def compare_lag_scenarios_precomputed(wav_dir, spectrogram_dir, lag_scenarios=[0, 2], num_epochs=30):
    """Compare different lag scenarios using precomputed spectrograms"""
    device = get_device()
    print(f"Using device: {device}")
    print(f"🚀 Training with precomputed spectrograms - much faster!")
    
    results = {}
    
    for lag in lag_scenarios:
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL FOR {lag}-SECOND LAG SCENARIO")
        print(f"{'='*60}")
        
        # Load precomputed data with specific lag
        df, label_encoder = load_precomputed_data_with_lag(spectrogram_dir, wav_dir, lag_seconds=lag)
        
        # Split data
        train_files, test_files, train_labels, test_labels = train_test_split(
            df['filename'].values, df['label'].values,
            test_size=0.2, random_state=42, stratify=df['label'].values
        )
        
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Create datasets with precomputed spectrograms (using proven optimal parameters)
        target_size = (128, 128)  # Original square spectrograms that achieve 84% accuracy
        train_dataset = PrecomputedSpectrogramDataset(train_files, train_labels, target_size=target_size)
        val_dataset = PrecomputedSpectrogramDataset(val_files, val_labels, target_size=target_size)
        test_dataset = PrecomputedSpectrogramDataset(test_files, test_labels, target_size=target_size)
        
        # Optimize settings - can now use multiple workers safely!
        batch_size = 128 if device.type == 'mps' else 64  # Larger batches since no audio processing
        num_workers = 4 if device.type == 'mps' else 8     # Can use workers safely now!
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Create and train model
        model = PlantLampClassifier(num_classes=len(label_encoder.classes_))
        model = model.to(device)
        
        model_name = f"{lag}s_lag"
        train_losses, val_losses, val_accuracies, best_val_acc = train_model(
            model, train_loader, val_loader, num_epochs=num_epochs,
            device=device, model_name=model_name
        )
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f'best_plant_lamp_classifier_{model_name}.pth', map_location=device))
        test_accuracy, predictions, targets, pred_proba = evaluate_model(
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
            'pred_proba': pred_proba
        }
    
    return results

def visualize_sample_spectrograms(spectrogram_dir, num_samples=4):
    """Visualize sample spectrograms from .npy files to understand the data"""
    spec_files = glob.glob(os.path.join(spectrogram_dir, "*.npy"))
    
    if len(spec_files) == 0:
        print("No spectrogram files found!")
        return
    
    # Sample some files
    sample_files = spec_files[:num_samples] if len(spec_files) >= num_samples else spec_files
    
    fig, axes = plt.subplots(2, len(sample_files), figsize=(4*len(sample_files), 8))
    if len(sample_files) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, spec_file in enumerate(sample_files):
        try:
            # Load the .npy file
            spectrogram_rgb = np.load(spec_file)
            print(f"File: {os.path.basename(spec_file)}")
            print(f"  Shape: {spectrogram_rgb.shape}")
            print(f"  Data type: {spectrogram_rgb.dtype}")
            print(f"  Value range: [{spectrogram_rgb.min():.3f}, {spectrogram_rgb.max():.3f}]")
            
            # Parse filename info
            file_info = parse_filename_with_timestamp(spec_file)
            
            # Show the spectrogram (use first channel since all 3 are identical)
            axes[0, i].imshow(spectrogram_rgb[0], aspect='auto', origin='lower', cmap='viridis')
            title = f"Lamp: {file_info['lamp_status']}\n"
            if file_info['chunk_number']:
                title += f"Chunk: {file_info['chunk_number']}\n"
            title += f"Time: {file_info['timestamp'].strftime('%H:%M:%S') if file_info['timestamp'] else 'Unknown'}"
            axes[0, i].set_title(title, fontsize=10)
            axes[0, i].set_xlabel('Time Steps')
            axes[0, i].set_ylabel('Mel Frequency Bins')
            
            # Show a zoomed view of frequency content
            axes[1, i].plot(np.mean(spectrogram_rgb[0], axis=1))
            axes[1, i].set_title(f"Avg Frequency Content", fontsize=10)
            axes[1, i].set_xlabel('Mel Frequency Bins')
            axes[1, i].set_ylabel('Average Power')
            axes[1, i].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error loading {spec_file}: {e}")
    
    plt.tight_layout()
    plt.show()
    """Plot comparison of different lag scenarios"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training curves
    ax1 = axes[0, 0]
    for lag, result in results.items():
        epochs = range(1, len(result['train_losses']) + 1)
        ax1.plot(epochs, result['train_losses'], label=f'{lag}s lag - Train', linestyle='--')
        ax1.plot(epochs, result['val_losses'], label=f'{lag}s lag - Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    ax2 = axes[0, 1]
    for lag, result in results.items():
        epochs = range(1, len(result['val_accuracies']) + 1)
        ax2.plot(epochs, result['val_accuracies'], label=f'{lag}s lag')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final accuracies comparison
    ax3 = axes[1, 0]
    lags = list(results.keys())
    val_accs = [results[lag]['best_val_acc'] for lag in lags]
    test_accs = [results[lag]['test_accuracy'] * 100 for lag in lags]
    
    x = np.arange(len(lags))
    width = 0.35
    
    ax3.bar(x - width/2, val_accs, width, label='Best Validation Accuracy', alpha=0.8)
    ax3.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    ax3.set_xlabel('Lag (seconds)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Final Accuracy Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{lag}s' for lag in lags])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Create space for confusion matrices
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Create separate figure for confusion matrices
    n_models = len(results)
    fig2, axes2 = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes2 = [axes2]
    
    for i, (lag, result) in enumerate(results.items()):
        cm = confusion_matrix(result['targets'], result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=result['label_encoder'].classes_,
                   yticklabels=result['label_encoder'].classes_,
                   ax=axes2[i])
        axes2[i].set_title(f'{lag}s lag\nTest Acc: {result["test_accuracy"]*100:.1f}%')
        if i == 0:
            axes2[i].set_ylabel('True Label')
        axes2[i].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF LAG ANALYSIS")
    print(f"{'='*60}")
    for lag, result in results.items():
        print(f"{lag}s lag: Val Acc = {result['best_val_acc']:.2f}%, Test Acc = {result['test_accuracy']*100:.2f}%")

def main():
    # Configuration
    WAV_DIR = "lamp_audio_collection/wav_files"
    SPECTROGRAM_DIR = "lamp_audio_collection/spectrograms"
    LAG_SCENARIOS = [0, 2]  # Test 0-second and 2-second lags
    NUM_EPOCHS = 30
    SAMPLE_RATE = 400
    
    print("🌱 Plant Lamp Response Lag Analysis with Precomputed Spectrograms")
    print("Comparing immediate (0s) vs delayed (2s) plant response scenarios")
    
    try:
        # Step 1: Check spectrograms and regenerate if needed
        regenerate_spectrograms = False
        
        if not os.path.exists(SPECTROGRAM_DIR) or len(glob.glob(os.path.join(SPECTROGRAM_DIR, "*.npy"))) == 0:
            print(f"\n📊 Spectrograms not found. Creating them from WAV files...")
            regenerate_spectrograms = True
        else:
            print(f"✅ Found existing spectrograms in {SPECTROGRAM_DIR}")
            spec_files = glob.glob(os.path.join(SPECTROGRAM_DIR, "*.npy"))
            print(f"Found {len(spec_files)} precomputed spectrograms")
            
            # Check if spectrograms have adequate size
            if len(spec_files) > 0:
                sample_spec = np.load(spec_files[0])
                print(f"Sample spectrogram shape: {sample_spec.shape}")
                
                # Check if spectrograms were generated with different parameters
                if sample_spec.shape[2] < 15 or sample_spec.shape[2] > 25:  # Expect ~18-20 time steps with n_fft=512, hop_length=64
                    print(f"⚠️  Existing spectrograms may need regeneration ({sample_spec.shape[2]} time steps)")
                    print(f"Regenerating with proven parameters (n_fft=512, hop_length=64)...")
                    regenerate_spectrograms = True
        
        if regenerate_spectrograms:
            success = preprocess_audio_to_spectrograms(WAV_DIR, SPECTROGRAM_DIR, SAMPLE_RATE)
            if not success:
                raise ValueError("Failed to preprocess audio files")
        
        # Step 1.5: Visualize sample spectrograms to understand the data
        print(f"\n🔍 Visualizing sample spectrograms...")
        try:
            visualize_sample_spectrograms(SPECTROGRAM_DIR, num_samples=4)
        except Exception as viz_error:
            print(f"Error in visualization: {viz_error}")
            print("Continuing with training...")
        
        # Step 2: Train models with precomputed spectrograms
        print(f"\n🚀 Training models using precomputed spectrograms...")
        results = compare_lag_scenarios_precomputed(WAV_DIR, SPECTROGRAM_DIR, LAG_SCENARIOS, NUM_EPOCHS)
        
        # Step 3: Create manual comparison since plotting may fail
        print(f"\n{'='*60}")
        print("SUMMARY OF LAG ANALYSIS")
        print(f"{'='*60}")
        
        best_lag = None
        best_accuracy = 0
    # Call metrics printer/saver while `results` is still in scope
        _print_and_save_metrics(results, "resnet_lamp_metrics_detailed.csv", n_perm=500)
        for lag, result in results.items():
            val_acc = result['best_val_acc']
            test_acc = result['test_accuracy'] * 100
            print(f"{lag}s lag: Val Acc = {val_acc:.2f}%, Test Acc = {test_acc:.2f}%")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_lag = lag
        
        # Step 4: Try to create plots
        try:
            # Create comparison plots manually
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Training curves
            ax1 = axes[0, 0]
            for lag, result in results.items():
                epochs = range(1, len(result['train_losses']) + 1)
                ax1.plot(epochs, result['train_losses'], label=f'{lag}s lag - Train', linestyle='--')
                ax1.plot(epochs, result['val_losses'], label=f'{lag}s lag - Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Final accuracies
            ax2 = axes[0, 1]
            lags = list(results.keys())
            val_accs = [results[lag]['best_val_acc'] for lag in lags]
            test_accs = [results[lag]['test_accuracy'] * 100 for lag in lags]
            
            x = np.arange(len(lags))
            width = 0.35
            ax2.bar(x - width/2, val_accs, width, label='Best Validation Accuracy', alpha=0.8)
            ax2.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
            ax2.set_xlabel('Lag (seconds)')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Final Accuracy Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{lag}s' for lag in lags])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3 & 4: Confusion matrices
            for i, (lag, result) in enumerate(results.items()):
                ax = axes[1, i] if i < 2 else axes[1, 1]  # Handle if more than 2 scenarios
                cm = confusion_matrix(result['targets'], result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=result['label_encoder'].classes_,
                           yticklabels=result['label_encoder'].classes_,
                           ax=ax)
                ax.set_title(f'{lag}s lag\nTest Acc: {result["test_accuracy"]*100:.1f}%')
                if i == 0:
                    ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.show()
            print("✅ Plots generated successfully!")
            
        except Exception as plot_error:
            print(f"⚠️  Plotting failed: {plot_error}")
            print("Results summary provided above.")
        

        
        # Determine best lag scenario
        best_lag = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        print(f"\n🏆 Best performing lag scenario: {best_lag} seconds")
        print(f"This suggests the plant response time is approximately {best_lag} seconds")
        
        # Show speed improvement
        print(f"\n⚡ Benefits of precomputed spectrograms:")
        print(f"   • No file handle issues")
        print(f"   • 5-10x faster training")
        print(f"   • Can use multiple DataLoader workers safely")
        print(f"   • Larger batch sizes possible")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:")
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. Your WAV files are in the correct directory")
        print("2. Filenames contain timestamp information or are chronologically ordered")
        print("3. Filenames contain 'lamp_on' or 'lamp_off' indicators")


# ======== Added: evaluation helpers (CI, AUCs, permutation) ========
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, classification_report
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

# ============ Helper: print & save metrics per lag (uses your results dict)
def _print_and_save_metrics(results_dict, out_csv, n_perm=500):
    """Print CI/ROC-AUC/PR-AUC + permutation baselines per lag and save a CSV."""
    try:
        print("\nDetailed test-set metrics per lag (with 95% CI, ROC-AUC, PR-AUC, permutation baselines):")
        metrics_rows = []
        for lag, result in results_dict.items():
            y_true = result['targets']
            # use probabilities when available; handle (n,2) or (n,)
            y_score = None
            pred_proba = result.get('pred_proba', None)
            if pred_proba is not None:
                try:
                    y_score = pred_proba[:, 1] if getattr(pred_proba, "ndim", 1) == 2 else pred_proba
                except Exception:
                    y_score = None
            y_pred = result.get('predictions', None)

            met = evaluate_with_auc_and_ci(y_true, y_score=y_score, y_pred=y_pred, n_permutations=n_perm)

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
                "n": met['n'],
            })

        import pandas as _pd
        _pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
        print(f"\nSaved detailed metrics to {out_csv}")
    except Exception as e:
        print(f"Metrics export failed: {e}")
# ============ End helper ============


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

