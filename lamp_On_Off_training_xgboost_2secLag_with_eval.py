"""
Plant Lamp Exposure Classifier using Numerical Features and XGBoost

This script extracts numerical features from Purple Heart plant bioelectrical responses
and uses XGBoost to classify lamp exposure (on vs off), comparing performance
with the ResNet spectrogram approach.

Features extracted:
- Time domain: mean, std, skewness, kurtosis, energy, zero-crossing rate
- Frequency domain: spectral centroid, rolloff, flux, flatness
- Statistical: percentiles, range, peak-to-peak
- Signal processing: RMS, autocorrelation features
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from tqdm import tqdm
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq

def extract_numerical_features(audio_data, sr=400):
    """
    Extract comprehensive numerical features from bioelectrical signal
    """
    features = {}
    
    # Ensure we have data
    if len(audio_data) == 0:
        return {f'feature_{i}': 0 for i in range(50)}  # Return zeros if no data
    
    # Basic time domain features
    features['mean'] = np.mean(audio_data)
    features['std'] = np.std(audio_data)
    features['var'] = np.var(audio_data)
    features['skewness'] = stats.skew(audio_data)
    features['kurtosis'] = stats.kurtosis(audio_data)
    features['min'] = np.min(audio_data)
    features['max'] = np.max(audio_data)
    features['range'] = features['max'] - features['min']
    features['peak_to_peak'] = np.ptp(audio_data)
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        features[f'percentile_{p}'] = np.percentile(audio_data, p)
    
    # Energy and power features
    features['energy'] = np.sum(audio_data ** 2)
    features['power'] = features['energy'] / len(audio_data)
    features['rms'] = np.sqrt(np.mean(audio_data ** 2))
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(audio_data)
    
    # Frequency domain features using FFT
    try:
        # Compute FFT
        fft_vals = fft(audio_data)
        fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = fftfreq(len(audio_data), 1/sr)[:len(fft_vals)//2]
        
        # Spectral features
        if np.sum(fft_magnitude) > 0:
            features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
            features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * fft_magnitude) / np.sum(fft_magnitude))
            features['spectral_rolloff'] = freqs[np.where(np.cumsum(fft_magnitude) >= 0.85 * np.sum(fft_magnitude))[0][0]]
        else:
            features['spectral_centroid'] = 0
            features['spectral_bandwidth'] = 0
            features['spectral_rolloff'] = 0
        
        # Spectral statistics
        features['spectral_mean'] = np.mean(fft_magnitude)
        features['spectral_std'] = np.std(fft_magnitude)
        features['spectral_skewness'] = stats.skew(fft_magnitude)
        features['spectral_kurtosis'] = stats.kurtosis(fft_magnitude)
        features['spectral_flatness'] = stats.gmean(fft_magnitude + 1e-10) / np.mean(fft_magnitude + 1e-10)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]
        features['dominant_magnitude'] = fft_magnitude[dominant_freq_idx]
        
    except Exception as e:
        # Fallback values if FFT fails
        for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                   'spectral_mean', 'spectral_std', 'spectral_skewness', 'spectral_kurtosis',
                   'spectral_flatness', 'dominant_frequency', 'dominant_magnitude']:
            features[key] = 0
    
    # Autocorrelation features
    try:
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        features['autocorr_max'] = np.max(autocorr[1:100]) if len(autocorr) > 100 else 0
        features['autocorr_mean'] = np.mean(autocorr[1:100]) if len(autocorr) > 100 else 0
    except:
        features['autocorr_max'] = 0
        features['autocorr_mean'] = 0
    
    # Signal derivative features
    try:
        diff1 = np.diff(audio_data)
        features['diff1_mean'] = np.mean(diff1)
        features['diff1_std'] = np.std(diff1)
        features['diff1_energy'] = np.sum(diff1 ** 2)
        
        diff2 = np.diff(diff1)
        features['diff2_mean'] = np.mean(diff2)
        features['diff2_std'] = np.std(diff2)
        features['diff2_energy'] = np.sum(diff2 ** 2)
    except:
        for key in ['diff1_mean', 'diff1_std', 'diff1_energy', 'diff2_mean', 'diff2_std', 'diff2_energy']:
            features[key] = 0
    
    # Mel-frequency features (simplified)
    try:
        # Use smaller n_fft for short signals (1200 samples = 3s at 400Hz)
        n_fft = min(512, len(audio_data))  # Ensure n_fft doesn't exceed signal length
        hop_length = n_fft // 4  # Standard hop length
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=13,
            n_fft=n_fft, hop_length=hop_length
        )
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=5)
        
        for i in range(5):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])
        
        features['mel_energy'] = np.sum(mel_spec)
        features['mel_energy_low'] = np.sum(mel_spec[:4])  # Low frequency energy
        features['mel_energy_mid'] = np.sum(mel_spec[4:9])  # Mid frequency energy
        features['mel_energy_high'] = np.sum(mel_spec[9:])  # High frequency energy
        
    except:
        for i in range(5):
            features[f'mfcc_{i}'] = 0
        for key in ['mel_energy', 'mel_energy_low', 'mel_energy_mid', 'mel_energy_high']:
            features[key] = 0
    
    return features

def parse_filename_with_timestamp(filename):
    """Parse filename to extract lamp status and timestamp"""
    basename = os.path.basename(filename)
    
    # Remove .wav extension if present for parsing
    if basename.endswith('.wav'):
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

def extract_features_from_wav_files(wav_files, sample_rate=400):
    """Extract numerical features from WAV files"""
    
    print(f"Extracting numerical features from {len(wav_files)} WAV files...")
    
    features_list = []
    valid_files = []
    
    for wav_file in tqdm(wav_files, desc="Extracting features"):
        try:
            # Load audio
            audio_data, sr = librosa.load(wav_file, sr=sample_rate)
            
            # Extract features
            features = extract_numerical_features(audio_data, sr)
            
            features_list.append(features)
            valid_files.append(wav_file)
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    if len(features_list) == 0:
        raise ValueError("No features could be extracted from WAV files")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    print(f"Extracted {len(features_df.columns)} features from {len(features_df)} files")
    print(f"Feature columns: {list(features_df.columns)}")
    
    return features_df, valid_files

def load_lamp_data_with_features(wav_dir, lag_seconds=0):
    """Load lamp data and extract numerical features with specified lag"""
    
    print(f"Looking for WAV files in: {wav_dir}")
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {wav_dir}")
    
    print(f"Found {len(wav_files)} WAV files")
    
    data = []
    valid_files = 0
    
    for wav_file in wav_files:
        try:
            # Parse filename for timestamp and lamp status
            file_info = parse_filename_with_timestamp(wav_file)
            
            if file_info['timestamp'] is not None and file_info['lamp_status'] in ['lamp_on', 'lamp_off']:
                data.append(file_info)
                valid_files += 1
            else:
                if valid_files < 5:  # Only show first few debug messages
                    print(f"    Skipping {os.path.basename(wav_file)}: timestamp={file_info['timestamp']}, status={file_info['lamp_status']}")
                    
        except Exception as e:
            if valid_files < 5:  # Only show first few debug messages
                print(f"    Error processing {os.path.basename(wav_file)}: {e}")
            continue
    
    print(f"Valid files with timestamps and labels: {valid_files}")
    
    if valid_files == 0:
        raise ValueError("No valid files with timestamps and lamp status found!")
    
    df = pd.DataFrame(data)
    
    print(f"Found {len(df)} files with valid timestamps and labels")
    print(f"Label distribution before lag adjustment:")
    print(f"    {df['lamp_status'].value_counts().to_dict()}")
    
    # Apply lag to labels
    df_lagged = create_lagged_labels(df, lag_seconds)
    
    print(f"Lag: {lag_seconds} seconds")
    print(f"Label distribution after lag adjustment:")
    print(f"    {df_lagged['lamp_status'].value_counts().to_dict()}")
    
    # Extract features from WAV files
    wav_files = df_lagged['filename'].tolist()
    features_df, valid_feature_files = extract_features_from_wav_files(wav_files)
    
    # Match features with labels
    valid_data = []
    for i, wav_file in enumerate(valid_feature_files):
        # Find corresponding row in df_lagged
        matching_rows = df_lagged[df_lagged['filename'] == wav_file]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            feature_dict = features_df.iloc[i].to_dict()
            feature_dict.update({
                'lamp_status': row['lamp_status'],
                'timestamp': row['timestamp'],
                'chunk_number': row['chunk_number'],
                'filename': wav_file
            })
            valid_data.append(feature_dict)
    
    df_final = pd.DataFrame(valid_data)
    
    if len(df_final) == 0:
        raise ValueError("No matching features and labels found!")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_final['label'] = label_encoder.fit_transform(df_final['lamp_status'])
    
    print(f"Final dataset with {lag_seconds}s lag:")
    print(f"  Total samples: {len(df_final)}")
    print(f"  Features: {len([col for col in df_final.columns if col not in ['lamp_status', 'timestamp', 'chunk_number', 'filename', 'label']])}")
    print(f"  Label distribution:")
    label_dist = df_final['lamp_status'].value_counts()
    for label, count in label_dist.items():
        print(f"    {label}: {count}")
    
    return df_final, label_encoder

def train_xgboost_model(X_train, y_train, X_val, y_val, model_name="lamp_model"):
    """Train XGBoost classifier"""
    
    print(f"Training XGBoost model: {model_name}")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # XGBoost parameters optimized for binary classification
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_estimators': 100,
        'early_stopping_rounds': 10
    }
    
    # Create XGBoost classifier
    model = xgb.XGBClassifier(**params)
    
    # Train model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Get training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Get validation accuracy
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"{model_name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, train_acc, val_acc

def evaluate_xgboost_model(model, X_test, y_test, label_encoder, model_name="lamp_model"):
    """Evaluate the trained XGBoost model"""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return accuracy, y_pred, y_pred_proba

def compare_lamp_lag_scenarios_xgboost(wav_dir, lag_scenarios=[0, 2]):
    """Compare different lag scenarios using XGBoost with numerical features"""
    
    print(f"Training XGBoost lamp detection models with numerical features")
    
    results = {}
    
    for lag in lag_scenarios:
        print(f"\n{'='*60}")
        print(f"TRAINING XGBOOST MODEL FOR {lag}-SECOND LAG SCENARIO")
        print(f"{'='*60}")
        
        # Load lamp data with features and specific lag
        df, label_encoder = load_lamp_data_with_features(wav_dir, lag_seconds=lag)
        
        # Check if we have enough data for both classes
        lamp_counts = df['lamp_status'].value_counts()
        if len(lamp_counts) < 2:
            print(f"Skipping {lag}s lag - insufficient data for binary classification")
            continue
        
        min_samples = lamp_counts.min()
        if min_samples < 10:
            print(f"Warning: Only {min_samples} samples for minority class in {lag}s lag scenario")
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col not in ['lamp_status', 'timestamp', 'chunk_number', 'filename', 'label']]
        X = df[feature_columns].values
        y = df['label'].values
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model_name = f"{lag}s_lag"
        model, train_acc, val_acc = train_xgboost_model(
            X_train_scaled, y_train, X_val_scaled, y_val, model_name
        )
        
        # Evaluate
        test_accuracy, predictions, pred_proba = evaluate_xgboost_model(
            model, X_test_scaled, y_test, label_encoder, model_name
        )
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results[lag] = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'targets': y_test,
            'pred_proba': pred_proba,
            'feature_importance': feature_importance,
            'feature_columns': feature_columns,
            'data_distribution': lamp_counts
        }
    
    return results

def visualize_feature_importance(results, top_n=15):
    """Visualize feature importance for different lag scenarios"""
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for i, (lag, result) in enumerate(results.items()):
        importance_df = result['feature_importance'].head(top_n)
        
        axes[i].barh(range(len(importance_df)), importance_df['importance'])
        axes[i].set_yticks(range(len(importance_df)))
        axes[i].set_yticklabels(importance_df['feature'])
        axes[i].set_xlabel('Feature Importance')
        axes[i].set_title(f'{lag}s Lag - Top {top_n} Features')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

def main():
    # Configuration
    WAV_DIR = "lamp_audio_collection/wav_files"
    LAG_SCENARIOS = [0, 2]  # Test 0-second and 2-second lags
    SAMPLE_RATE = 400
    
    print("Plant Lamp Response Detection using XGBoost with Numerical Features")
    print("Comparing immediate (0s) vs delayed (2s) plant response scenarios")
    print("Using numerical features extracted from plant bioelectrical signals")
    
    try:
        # Train XGBoost models for different lag scenarios
        print(f"\nTraining XGBoost lamp detection models with numerical features...")
        results = compare_lamp_lag_scenarios_xgboost(WAV_DIR, LAG_SCENARIOS)
        
        # Analyze results
        print(f"\n{'='*60}")
        print("SUMMARY OF XGBOOST LAMP DETECTION LAG ANALYSIS")
        print(f"{'='*60}")
        
        best_lag = None
        best_accuracy = 0
        
        for lag, result in results.items():
            train_acc = result['train_accuracy'] * 100
            val_acc = result['val_accuracy'] * 100
            test_acc = result['test_accuracy'] * 100
            print(f"{lag}s lag: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%, Test Acc = {test_acc:.2f}%")
            print(f"  Data distribution: {dict(result['data_distribution'])}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_lag = lag
        
        if best_lag is not None:
            print(f"\nBest performing lag scenario: {best_lag} seconds")
            print(f"XGBoost achieves {best_accuracy:.1f}% accuracy on lamp detection")
        
        # Visualize feature importance
        try:
            print("\nVisualizing feature importance...")
            visualize_feature_importance(results, top_n=15)
        except Exception as viz_error:
            print(f"Visualization error: {viz_error}")
        
        # Create comparison plots
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Accuracy comparison
            ax1 = axes[0, 0]
            lags = list(results.keys())
            train_accs = [results[lag]['train_accuracy'] * 100 for lag in lags]
            val_accs = [results[lag]['val_accuracy'] * 100 for lag in lags]
            test_accs = [results[lag]['test_accuracy'] * 100 for lag in lags]
            
            x = np.arange(len(lags))
            width = 0.25
            ax1.bar(x - width, train_accs, width, label='Train Accuracy', alpha=0.8)
            ax1.bar(x, val_accs, width, label='Validation Accuracy', alpha=0.8)
            ax1.bar(x + width, test_accs, width, label='Test Accuracy', alpha=0.8)
            ax1.set_xlabel('Plant Response Lag (seconds)')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('XGBoost Lamp Detection Accuracy by Lag')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'{lag}s' for lag in lags])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Feature importance comparison (first two models)
            if len(results) >= 2:
                ax2 = axes[0, 1]
                lag_list = list(results.keys())[:2]
                for i, lag in enumerate(lag_list):
                    importance_df = results[lag]['feature_importance'].head(10)
                    y_pos = np.arange(len(importance_df)) + i * 0.4
                    ax2.barh(y_pos, importance_df['importance'], height=0.35,
                            label=f'{lag}s lag', alpha=0.8)
                
                ax2.set_yticks(np.arange(len(results[lag_list[0]]['feature_importance'].head(10))) + 0.2)
                ax2.set_yticklabels(results[lag_list[0]]['feature_importance'].head(10)['feature'])
                ax2.set_xlabel('Feature Importance')
                ax2.set_title('Top 10 Feature Importance Comparison')
                ax2.legend()
                ax2.invert_yaxis()
            
            # Plot 3 & 4: Confusion matrices
            for i, (lag, result) in enumerate(list(results.items())[:2]):
                ax = axes[1, i]
                cm = confusion_matrix(result['targets'], result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=result['label_encoder'].classes_,
                           yticklabels=result['label_encoder'].classes_,
                           ax=ax)
                ax.set_title(f'{lag}s lag\nTest Acc: {result["test_accuracy"]*100:.1f}%')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.show()
            print("Plots generated successfully!")
            
        except Exception as plot_error:
            print(f"Plotting failed: {plot_error}")
        
        # Print top features for best model
        if best_lag is not None:
            print(f"\nTop 10 most important features for {best_lag}s lag model:")
            top_features = results[best_lag]['feature_importance'].head(10)
            for idx, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nXGBoost lamp detection analysis complete!")
        
        _print_and_save_metrics(results, "xgb_lamp_metrics_detailed.csv")
        
        # Compare with your reported ResNet results
        print(f"\nCOMPARISON WITH RESNET APPROACH:")
        print(f"Your ResNet achieved ~84% accuracy on lamp detection")
        if best_lag is not None:
            resnet_acc = 84  # From your script comments
            xgboost_acc = best_accuracy
            if xgboost_acc > resnet_acc:
                print(f"XGBoost ({xgboost_acc:.1f}%) outperforms ResNet ({resnet_acc}%) by {xgboost_acc - resnet_acc:.1f}%")
            else:
                print(f"ResNet ({resnet_acc}%) outperforms XGBoost ({xgboost_acc:.1f}%) by {resnet_acc - xgboost_acc:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. Your WAV files are in the correct directory")
        print("2. Filenames contain timestamp information or are chronologically ordered")
        print("3. Filenames contain 'lamp_on' or 'lamp_off' indicators")


# ======== Added: evaluation helpers (CI, AUCs, permutation) ========
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)

def _print_and_save_metrics(results_dict, out_csv):
    """Print CI/ROC-AUC/PR-AUC + permutation baselines per lag and save a CSV."""
    try:
        print("\nDetailed test-set metrics per lag (with 95% CI, ROC-AUC, PR-AUC, permutation baselines):")
        metrics_rows = []
        for lag, result in results_dict.items():
            y_true = result['targets']
            y_score = None
            pred_proba = result.get('pred_proba', None)
            if pred_proba is not None:
                try:
                    # if shape (n,2) take prob of class 1; if (n,), use directly
                    y_score = pred_proba[:, 1] if getattr(pred_proba, "ndim", 1) == 2 else pred_proba
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
                "n": met['n'],
            })

        import pandas as _pd
        _pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
        print(f"\nSaved detailed metrics to {out_csv}")
    except Exception as e:
        print(f"Metrics export failed: {e}")


def _wilson_ci(k: int, n: int, alpha: float = 0.05):
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
    try:
        roc = roc_auc_score(y_true, y_score)
    except Exception:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, y_score, pos_label=1)
    except Exception:
        pr = float("nan")
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

