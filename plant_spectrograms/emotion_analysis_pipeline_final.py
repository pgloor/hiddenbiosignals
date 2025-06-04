import json
import csv
import datetime
import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


class EmotionAnalysisPipeline:
    def __init__(self, output_dir="emotion_analysis_output"):
        """
        Initialize the pipeline with output directory.
        
        Args:
            output_dir (str): Directory to save all output files.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        # Set up subdirectories
        self.spectrograms_dir = os.path.join(output_dir, "spectrograms")
        if not os.path.exists(self.spectrograms_dir):
            os.makedirs(self.spectrograms_dir)
            
        # Check for GPU support
        if tf.config.list_physical_devices('GPU'):
            print("Using GPU for training")
            tf.keras.backend.set_floatx('float32')
        else:
            print("Using CPU for training")

    def json_to_csv(self, json_file, csv_file=None):
        """
        Converts a JSON file to a CSV file, handling the specific timestamp format.

        Args:
            json_file (str): Path to the JSON file.
            csv_file (str, optional): Path to the output CSV file. If None, will save to output_dir.
            
        Returns:
            str: Path to the created CSV file.
        """
        if csv_file is None:
            csv_file = os.path.join(self.output_dir, "emotion_data.csv")
            
        with open(json_file, 'r') as f:
            data = json.load(f)

        fieldnames = list(data[0].keys())  # Convert dict_keys to a list for indexing

        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)  # Write the header row

            for row in data:
                # Convert the timestamp to a datetime object
                timestamp = datetime.datetime.fromtimestamp(row['timestamp'] / 1000)
                # Format the timestamp as an Excel-compatible datetime string
                excel_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                # Create a new row with the converted timestamp
                new_row = list(row.values())  # Convert dict_values to a list for indexing
                new_row[fieldnames.index('timestamp')] = excel_timestamp
                writer.writerow(new_row)
                
        print(f"Converted JSON to CSV: {csv_file}")
        return csv_file

    def find_strongest_emotion(self, csv_file):
        """
        Find the strongest emotion in each row of the CSV file and save as Excel.
        
        Args:
            csv_file (str): Path to the CSV file with emotion data.
            
        Returns:
            str: Path to the output Excel file.
        """
        # Load the CSV file
        data = pd.read_csv(csv_file)
        
        # Convert to Excel format
        excel_file = os.path.join(self.output_dir, "emotion_data_with_strongest.xlsx")
        data.to_excel(excel_file, index=False)
        
        # List of emotion columns
        emotion_columns = ['happy', 'surprised', 'neutral', 'sad', 'angry', 'disgusted', 'fearful']

        # Determine the strongest emotion in each row and add a new column
        data['strongest_emotion'] = data[emotion_columns].idxmax(axis=1)

        # Save the updated data to a new Excel file
        output_file_path = os.path.join(self.output_dir, "emotion_data_with_strongest.xlsx")
        data.to_excel(output_file_path, index=False)

        print(f"Added strongest emotion column and saved to: {output_file_path}")
        return output_file_path

    def align_wav_with_emotion_data(self, wav_file, wav_start_time_str, emotion_start_time_str=None):
        """
        Aligns the WAV file with emotion data based on timestamps. Can handle cases where:
        1. Emotion data starts after WAV file (trims beginning of WAV)
        2. Emotion data starts before WAV file (filters emotion data)
        3. Both start at the same time (no adjustment needed)

        Args:
            wav_file (str): Path to the original WAV file.
            wav_start_time_str (str): Start time of the WAV file in '%Y-%m-%d %H:%M:%S' format.
            emotion_start_time_str (str, optional): Start time of the emotion data in '%Y-%m-%d %H:%M:%S' format.
                                                   If None, uses the wav_start_time (no trimming).
                                                   
        Returns:
            tuple: (path to aligned WAV file, emotion data time offset in seconds)
            The time offset represents how many seconds the emotion data starts before the WAV file.
            This will be 0 if WAV starts before or at the same time as emotion data.
        """
        # If no emotion start time is provided, use the WAV start time (no trimming needed)
        if emotion_start_time_str is None:
            emotion_start_time_str = wav_start_time_str
            
        # Convert start times to datetime objects
        wav_start_time = datetime.datetime.strptime(wav_start_time_str, "%Y-%m-%d %H:%M:%S")
        emotion_start_time = datetime.datetime.strptime(emotion_start_time_str, "%Y-%m-%d %H:%M:%S")
        
        # Calculate the offset in seconds (can be positive or negative)
        time_offset = (emotion_start_time - wav_start_time).total_seconds()
        
        # Output WAV file path
        output_wav_file = os.path.join(self.output_dir, "aligned_audio.wav")
        emotion_data_offset = 0  # Will be set if emotion data starts before WAV
        
        # CASE 1: No alignment needed (both start at the same time)
        if time_offset == 0:
            # Load and save the WAV file as is
            y, sr = librosa.load(wav_file, sr=None)
            sf.write(output_wav_file, y, sr)
            print(f"No alignment needed. Audio saved to {output_wav_file}")
        
        # CASE 2: Emotion data starts AFTER WAV file (trim beginning of WAV)
        elif time_offset > 0:
            print(f"WAV file starts {time_offset} seconds before emotion data. Trimming WAV file...")
            
            # Load the WAV file
            y, sr = librosa.load(wav_file, sr=None)
            
            # Calculate the number of samples to trim
            samples_to_trim = int(time_offset * sr)
            
            # Trim the WAV file
            trimmed_y = y[samples_to_trim:]
            
            # Save the trimmed WAV file
            sf.write(output_wav_file, trimmed_y, sr)
            print(f"Trimmed WAV file saved to {output_wav_file}")
        
        # CASE 3: Emotion data starts BEFORE WAV file
        else:  # time_offset < 0
            print(f"Emotion data starts {abs(time_offset)} seconds before WAV file.")
            print("WAV file will be preserved as is. Emotion data will be filtered during spectrogram generation.")
            
            # Load and save the WAV file as is
            y, sr = librosa.load(wav_file, sr=None)
            sf.write(output_wav_file, y, sr)
            
            # Store the absolute value of the negative offset
            # This will be used later to filter the emotion data
            emotion_data_offset = abs(time_offset)
            
            print(f"Original WAV file saved to {output_wav_file}")
            print(f"Emotion data time offset: {emotion_data_offset} seconds")
        
        return output_wav_file, emotion_data_offset

    def create_emotion_spectrograms(self, wav_file, excel_file, wav_start_time_str, emotion_data_offset=0, n_fft=256, interval_seconds=2):
        """
        Creates spectrograms from a WAV file and labels them with the strongest emotion based on aligned timestamps.
        
        Args:
            wav_file: Path to the input WAV file.
            excel_file: Path to Excel file with columns for emotions and 'timestamp' (as datetime).
            wav_start_time_str: Start time of the WAV file in '%Y-%m-%d %H:%M:%S' format.
            emotion_data_offset: Time in seconds that emotion data starts before WAV file (default: 0).
            n_fft: Size of the FFT window (default: 256).
            interval_seconds: Time interval between spectrograms (default: 2 seconds).
            
        Returns:
            str: Path to the directory containing spectrograms.
        """
        # Load the WAV file with its native sample rate
        y, sr = librosa.load(wav_file, sr=None)  # sr=None means use the file's native sample rate
        
        print(f"Loaded WAV file with sample rate: {sr} Hz")
        print(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
        
        # Load the emotion data
        emotion_df = pd.read_excel(excel_file)
        emotion_columns = ['happy', 'surprised', 'neutral', 'sad', 'angry', 'disgusted', 'fearful']
        
        # Calculate the strongest emotion for each row if not already present
        if 'strongest_emotion' not in emotion_df.columns:
            emotion_df['strongest_emotion'] = emotion_df[emotion_columns].idxmax(axis=1)
            
        emotion_df['timestamp'] = pd.to_datetime(emotion_df['timestamp'])
        
        # Convert WAV start time to datetime
        wav_start_time = datetime.datetime.strptime(wav_start_time_str, "%Y-%m-%d %H:%M:%S")
        
        # If emotion data starts before WAV file, filter out emotion data before WAV start
        if emotion_data_offset > 0:
            print(f"Filtering emotion data to remove entries before WAV start time...")
            # The effective emotion start time that aligns with the WAV start
            emotion_effective_start = wav_start_time - datetime.timedelta(seconds=emotion_data_offset)
            print(f"Effective emotion data start time: {emotion_effective_start}")
            
            # Count total entries before filtering
            total_entries_before = len(emotion_df)
            
            # Filter emotion data to only include entries after WAV started
            emotion_df = emotion_df[emotion_df['timestamp'] >= wav_start_time]
            
            # Count entries after filtering
            total_entries_after = len(emotion_df)
            
            print(f"Filtered out {total_entries_before - total_entries_after} emotion entries that occurred before WAV start")
            
            if emotion_df.empty:
                raise ValueError("After filtering, no emotion data remains that aligns with the WAV file. Check your timestamps.")
        
        # Calculate the number of frames
        duration = librosa.get_duration(y=y, sr=sr)
        num_frames = int(duration // interval_seconds)
        
        if num_frames == 0:
            raise ValueError(f"WAV file duration ({duration:.2f}s) is too short for the specified interval ({interval_seconds}s)")
        
        # Calculate frame parameters
        samples_per_frame = sr * interval_seconds
        hop_length = n_fft // 4
        
        # Make sure we have enough emotion data to continue
        if emotion_df.empty:
            raise ValueError("No emotion data available for the specified time range")
        
        # Generate spectrograms for each frame
        for i in range(num_frames):
            # Calculate the current time in the WAV file
            current_time = i * interval_seconds
            current_wav_time = wav_start_time + datetime.timedelta(seconds=current_time)
            
            # Find the strongest emotion for the current WAV time
            closest_emotion_rows = emotion_df[emotion_df['timestamp'] <= current_wav_time]
            if closest_emotion_rows.empty:
                current_emotion = "unknown"
                print(f"Warning: No emotion data found for time {current_wav_time}")
            else:
                emotion_row = closest_emotion_rows.iloc[-1]
                current_emotion = emotion_row['strongest_emotion']
            
            # Extract the audio segment
            start_sample = i * samples_per_frame
            end_sample = min((i + 1) * samples_per_frame, len(y))
            y_frame = y[start_sample:end_sample]
            
            # Skip if the segment is too short
            if len(y_frame) < n_fft:
                print(f"Skipping frame {i} as it's too short for FFT analysis")
                continue
            
            # Compute the melspectrogram
            S = librosa.feature.melspectrogram(
                y=y_frame,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=64
            )
            
            # Convert to log scale
            log_S = librosa.power_to_db(S, ref=np.max)
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Add spectrogram
            plt.subplot(1, 1, 1)
            librosa.display.specshow(
                log_S,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Time: {current_wav_time} - Strongest Emotion: {current_emotion}")
            
            # Save the plot
            output_file = os.path.join(self.spectrograms_dir, f"spectrogram_{i:04d}s_{current_emotion}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Generated spectrogram {i+1}/{num_frames} - Time: {current_wav_time} - Emotion: {current_emotion}")
        
        print(f"Generated {num_frames} spectrograms in {self.spectrograms_dir}")
        
        # Verify we have generated at least some spectrograms
        spectrograms = [f for f in os.listdir(self.spectrograms_dir) if f.endswith('.png')]
        if not spectrograms:
            raise ValueError("No spectrograms were generated. Check your data and parameters.")
            
        return self.spectrograms_dir

    def load_and_preprocess_data(self, spectrogram_dir):
        """
        Load spectrograms and their labels from the directory.
        """
        images = []
        labels = []
        
        for filename in os.listdir(spectrogram_dir):
            if filename.endswith('.png'):
                # Extract emotion type from filename
                emotion_type = filename.split('_')[-1].replace('.png', '')
                
                # Load and preprocess image
                img_path = os.path.join(spectrogram_dir, filename)
                img = Image.open(img_path)
                # Convert grayscale to RGB by duplicating channels
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                
                # Preprocess for ResNet
                img_array = preprocess_input(img_array)
                
                images.append(img_array)
                labels.append(emotion_type)
        
        return np.array(images), np.array(labels)

    def create_resnet_model(self, num_classes):
        """
        Create a ResNet50 model with transfer learning
        """
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create new model with custom top layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model, base_model

    def filter_classes(self, X, y_onehot, min_samples_per_class):
        """
        Filters out classes with fewer than `min_samples_per_class` samples.
        """
        # Calculate class distribution
        class_counts = Counter(np.argmax(y_onehot, axis=1))
        print("Original class distribution:", class_counts)
        
        # Determine valid classes
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
        print(f"Valid classes (at least {min_samples_per_class} samples): {valid_classes}")
        
        # Validate the filtering condition
        if not valid_classes:
            raise ValueError(f"No classes have at least {min_samples_per_class} samples.")
        
        # Filter indices of valid classes
        valid_indices = np.isin(np.argmax(y_onehot, axis=1), valid_classes)
        X_filtered = X[valid_indices]
        y_filtered = y_onehot[valid_indices]
        
        # Print filtered class distribution for verification
        filtered_class_counts = Counter(np.argmax(y_filtered, axis=1))
        print("Filtered class distribution:", filtered_class_counts)
        
        return X_filtered, y_filtered, valid_classes

    def ensure_stratification(self, X, y, min_samples_per_class):
        """
        Adjust dataset by downsampling larger classes and oversampling smaller ones.
        """
        # Calculate class counts
        class_counts = Counter(np.argmax(y, axis=1))
        print("Original class distribution:", class_counts)

        # Determine target size for downsampling
        counts_sorted = sorted(class_counts.values(), reverse=True)
        target_size = counts_sorted[1] if len(counts_sorted) > 1 else counts_sorted[0]
        print(f"Target size for downsampling: {target_size}")

        X_resampled, y_resampled = [], []

        for cls in np.unique(np.argmax(y, axis=1)):
            X_cls = X[np.argmax(y, axis=1) == cls]
            y_cls = y[np.argmax(y, axis=1) == cls]
            
            if len(X_cls) > target_size:  # Downsample if larger than target
                X_cls, y_cls = resample(X_cls, y_cls, n_samples=target_size, random_state=42, replace=False)
            elif len(X_cls) < min_samples_per_class:  # Oversample if smaller than min_samples
                X_cls, y_cls = resample(X_cls, y_cls, n_samples=min_samples_per_class, random_state=42, replace=True)

            X_resampled.append(X_cls)
            y_resampled.append(y_cls)

        # Combine all resampled classes
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.vstack(y_resampled)

        # Print final class distribution
        final_class_counts = Counter(np.argmax(y_resampled, axis=1))
        print("Final class distribution after stratification:", final_class_counts)

        return X_resampled, y_resampled

    def preprocess_data(self, X, y_onehot, min_samples_per_class=50):
        """
        Preprocess and balance the dataset, ensuring stratification is possible.
        """
        # Step 1: Filter out classes with too few samples
        X_filtered, y_filtered, valid_classes = self.filter_classes(X, y_onehot, min_samples_per_class)

        # Step 2: Oversample to ensure sufficient samples for stratification
        X_resampled, y_resampled = self.ensure_stratification(X_filtered, y_filtered, min_samples_per_class)

        # Step 3: Split the dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=np.argmax(y_resampled, axis=1)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        # Verify class distribution after splitting
        print("Training set class distribution:", Counter(np.argmax(y_train, axis=1)))
        print("Validation set class distribution:", Counter(np.argmax(y_val, axis=1)))
        print("Test set class distribution:", Counter(np.argmax(y_test, axis=1)))

        return X_train, X_val, X_test, y_train, y_val, y_test, valid_classes

    def generate_metrics(self, model, X_test, y_test, label_encoder, valid_classes):
        """
        Generate confusion matrix and classification report for the model.
        """
        # Predict on the test set
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Map valid class indices to their original names
        valid_class_names = [label_encoder.classes_[cls] for cls in valid_classes]
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_class_names)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_file = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_file)
        plt.close()
        print(f"Saved confusion matrix to {cm_file}")

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=valid_class_names, labels=valid_classes
        )
        print("\nClassification Report:")
        print(report)
        
        # Save classification report to file
        report_file = os.path.join(self.output_dir, "classification_report.txt")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Saved classification report to {report_file}")

    def train_emotion_classifier(self, spectrogram_dir, epochs=30, min_samples=0):
        """
        Train a ResNet model to classify emotion types from spectrograms
        
        Args:
            spectrogram_dir (str): Directory containing spectrogram images
            epochs (int): Number of training epochs
            min_samples (int): Minimum samples per class (0 = auto-determine)
            
        Returns:
            tuple: (model, label_encoder, history, valid_classes)
        """
        # Output model path
        model_path = os.path.join(self.output_dir, "emotion_classifier_model.keras")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X, y = self.load_and_preprocess_data(spectrogram_dir)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        y_onehot = tf.keras.utils.to_categorical(y_encoded)
        print("Shape of y_onehot:", y_onehot.shape)
        
        # Dynamic adjustment of min_samples based on dataset size
        if min_samples <= 0:
            avg_samples = len(X) // num_classes
            min_samples_per_class = max(20, min(avg_samples // 2, 50))
            print(f"Auto-selected min_samples_per_class = {min_samples_per_class}")
        else:
            min_samples_per_class = min_samples
            print(f"Using user-specified min_samples_per_class = {min_samples_per_class}")
        
        # Preprocess and retrieve valid classes
        X_train, X_val, X_test, y_train, y_val, y_test, valid_classes = self.preprocess_data(
            X, y_onehot, min_samples_per_class=min_samples_per_class
        )

        # Calculate class weights (inversely proportional to frequency)
        class_counts = Counter(np.argmax(y_train, axis=1))
        total_samples = len(y_train)
        n_classes = len(valid_classes)
        
        class_weights = {}
        for cls_idx in valid_classes:
            class_weights[cls_idx] = total_samples / (n_classes * class_counts[cls_idx])
        
        print("Class weights:", class_weights)
        
        # Create and compile model
        print("Creating model...")
        model, base_model = self.create_resnet_model(num_classes)
        
        # First phase: Train only the top layers
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        
        # Filter out underrepresented classes
        X_train, y_train, valid_classes = self.filter_classes(X_train, y_train, min_samples_per_class)
        X_val, y_val, _ = self.filter_classes(X_val, y_val, min_samples_per_class)
        np.save("label_classes.npy", valid_classes)

# First training phase
        print("Training top layers...")
        history1 = model.fit(
            X_train, y_train,
            epochs=epochs // 2,  # Use half the epochs for the first phase
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Second phase: Fine-tune the last few layers of ResNet
        print("\nFine-tuning ResNet layers...")
        # Unfreeze the last 30 layers (or fewer if the model has fewer layers)
        layers_to_unfreeze = min(30, len(base_model.layers))
        for layer in base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Second training phase
        history2 = model.fit(
            X_train, y_train,
            epochs=epochs // 2,  # Use half the epochs for the second phase
            batch_size=16,  # Smaller batch size for fine-tuning
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        # Combine histories
        history = {}
        for key in history1.history:
            history[key] = history1.history[key] + history2.history[key]
        
        # Evaluate model on unseen test data
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nUnseen test accuracy: {test_accuracy:.4f}")
        
        # Save the LabelEncoder classes to a file
        np.save(os.path.join(self.output_dir, 'label_classes.npy'), label_encoder.classes_)
        
        # Generate metrics for the unseen test set
        self.generate_metrics(model, X_test, y_test, label_encoder, valid_classes)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()

        return model, label_encoder, history, valid_classes

    def predict_emotion(self, model, image_path, label_encoder):
        """
        Predict emotion type for a single spectrogram
        """
        # Load and preprocess image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
        confidence = np.max(prediction[0])
        
        return predicted_class, confidence

    def run_pipeline(self, json_file=None, csv_file=None, wav_file=None,
                    wav_start_time=None, emotion_start_time=None, epochs=30, min_samples=0):
        """
        Run the complete pipeline from JSON/CSV to model training.
        
        Args:
            json_file (str, optional): Path to JSON file with emotion data.
            csv_file (str, optional): Path to CSV file with emotion data (alternative to JSON).
            wav_file (str): Path to the WAV file to analyze.
            wav_start_time (str): Start time of the WAV file in '%Y-%m-%d %H:%M:%S' format.
            emotion_start_time (str, optional): Start time of emotion data, if different from WAV.
            epochs (int): Number of training epochs.
            min_samples (int): Minimum samples per class (0 = auto-determine).
            
        Returns:
            dict: Dictionary with paths to all outputs and the trained model.
        """
        results = {}
        
        print("\n" + "="*50)
        print("STARTING EMOTION ANALYSIS PIPELINE")
        print("="*50 + "\n")
        
        # Step 1: Convert JSON to CSV (if JSON provided)
        print("\nSTEP 1: PROCESSING EMOTION DATA")
        print("-"*30)
        if json_file is not None:
            print(f"Converting JSON to CSV: {json_file}")
            csv_file = self.json_to_csv(json_file)
        
        if csv_file is None:
            raise ValueError("Either json_file or csv_file must be provided")
            
        results['csv_file'] = csv_file
        
        # Step 2: Find strongest emotion
        print("\nSTEP 2: FINDING STRONGEST EMOTIONS")
        print("-"*30)
        excel_file = self.find_strongest_emotion(csv_file)
        results['excel_file'] = excel_file
        
        # Step 3: Align WAV with emotion data
        print("\nSTEP 3: ALIGNING WAV AND EMOTION DATA")
        print("-"*30)
        if wav_file is None:
            raise ValueError("wav_file must be provided")
            
        if wav_start_time is None:
            raise ValueError("wav_start_time must be provided")
        
        print(f"WAV file: {wav_file}")
        print(f"WAV start time: {wav_start_time}")
        print(f"Emotion start time: {emotion_start_time if emotion_start_time else 'Same as WAV'}")
            
        # The updated function now returns the aligned WAV file and an offset value
        aligned_wav, emotion_data_offset = self.align_wav_with_emotion_data(
            wav_file, wav_start_time, emotion_start_time
        )
        results['aligned_wav'] = aligned_wav
        
        # Step 4: Create emotion spectrograms
        print("\nSTEP 4: CREATING SPECTROGRAMS")
        print("-"*30)
        try:
            # Pass the emotion data offset to the spectrogram generation function
            spectrograms_dir = self.create_emotion_spectrograms(
                aligned_wav,
                excel_file,
                wav_start_time,
                emotion_data_offset=emotion_data_offset
            )
            results['spectrograms_dir'] = spectrograms_dir
            
            # Step 5: Train emotion classifier
            print("\nSTEP 5: TRAINING CLASSIFIER MODEL")
            print("-"*30)
            
            # Pass the min_samples parameter if specified
            model, label_encoder, history, valid_classes = self.train_emotion_classifier(
                spectrograms_dir,
                epochs=epochs,
                min_samples=min_samples
            )
            results['model'] = model
            results['label_encoder'] = label_encoder
            results['valid_classes'] = valid_classes
            
            # Step 6: Make sample predictions
            print("\nSTEP 6: GENERATING SAMPLE PREDICTIONS")
            print("-"*30)
            
            # Get a few random spectrograms for prediction
            spectrograms = [os.path.join(spectrograms_dir, f) for f in os.listdir(spectrograms_dir)
                            if f.endswith('.png')]
            
            # Take up to 5 random samples
            sample_size = min(5, len(spectrograms))
            if sample_size > 0:
                sample_spectrograms = np.random.choice(spectrograms, sample_size, replace=False)
                
                predictions = []
                for img_path in sample_spectrograms:
                    predicted_class, confidence = self.predict_emotion(model, img_path, label_encoder)
                    actual_class = img_path.split('_')[-1].replace('.png', '')
                    
                    print(f"File: {os.path.basename(img_path)}")
                    print(f"  Actual emotion: {actual_class}")
                    print(f"  Predicted emotion: {predicted_class}")
                    print(f"  Confidence: {confidence:.2%}")
                    
                    predictions.append({
                        'file': os.path.basename(img_path),
                        'actual': actual_class,
                        'predicted': predicted_class,
                        'confidence': confidence
                    })
                
                # Save predictions to CSV
                predictions_file = os.path.join(self.output_dir, "sample_predictions.csv")
                pd.DataFrame(predictions).to_csv(predictions_file, index=False)
                print(f"\nSaved sample predictions to {predictions_file}")
                results['predictions_file'] = predictions_file
        
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            print("Pipeline encountered an error. Partial results will be returned.")
            # Still return whatever results we have so far
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED")
        print("="*50)
        return results


# Example usage function
def main():
    """
    Main function to run the pipeline with command line arguments.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Emotion Analysis Pipeline")
    
    # Required arguments
    parser.add_argument('--wav_file', type=str, required=True,
                      help='Path to WAV file to analyze')
    parser.add_argument('--wav_start_time', type=str, required=True,
                      help='Start time of WAV file in format "YYYY-MM-DD HH:MM:SS"')
    
    # One of these is required
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json_file', type=str, help='Path to JSON file with emotion data')
    group.add_argument('--csv_file', type=str, help='Path to CSV file with emotion data')
    
    # Optional arguments
    parser.add_argument('--emotion_start_time', type=str,
                      help='Start time of emotion data if different from WAV file')
    parser.add_argument('--output_dir', type=str, default='emotion_analysis_output',
                      help='Directory to save output files')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--min_samples', type=int, default=0,
                      help='Minimum samples per emotion class (0=auto)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose flag
    import logging
    logging_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')
    
    # Initialize and run the pipeline
    pipeline = EmotionAnalysisPipeline(output_dir=args.output_dir)
    
    # Run the pipeline with optional min_samples parameter
    results = pipeline.run_pipeline(
        json_file=args.json_file,
        csv_file=args.csv_file,
        wav_file=args.wav_file,
        wav_start_time=args.wav_start_time,
        emotion_start_time=args.emotion_start_time,
        epochs=args.epochs,
        min_samples=args.min_samples
    )
    
    print("\nPipeline complete! Summary of output files:")
    for key, value in results.items():
        if key not in ['model', 'label_encoder', 'valid_classes']:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
