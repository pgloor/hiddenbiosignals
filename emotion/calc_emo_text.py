# computing emotions from text using a prelabeled dataset with a pretrained model from Huggingface
# pip install transformers pandas tqdm openpyxl

import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# Load Hugging Face's emotion classifier
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", return_all_scores=True, device=0 if device == "cuda" else -1)

# Define the six Ekman emotions
ekman_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Function to safely truncate text to fit within the model limit (512 tokens)
def truncate_text(text, max_tokens=512):
    words = text.split()  # Split text into words
    return " ".join(words[:max_tokens])  # Keep only the first 512 tokens
    
    
# Function to classify emotions
def compute_emotions(text):
    if not isinstance(text, str) or text.strip() == "":
        return {emotion: 0.0 for emotion in ekman_emotions}  # Return zero scores for empty text
    
    try:
        truncated_text = truncate_text(text)  # Truncate text to fit within 512 tokens
        results = classifier(truncated_text)[0]  # Get emotion scores from the model
        emotion_scores = {result['label']: result['score'] for result in results}
        
        # Extract only the six Ekman emotions and normalize missing ones
        return {emotion: emotion_scores.get(emotion, 0.0) for emotion in ekman_emotions}
    
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {e}")
        return {emotion: 0.0 for emotion in ekman_emotions}  # Return zero scores on failure

# Function to process Excel file and append emotion scores
def append_emotions_to_excel(file_path, text_column="story"):
    # Load Excel file
    df = pd.read_excel(file_path)

    # Check if the text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the Excel file.")

    print("Computing emotions...")

    # Compute emotions for each row and append new columns
    emotions_data = [compute_emotions(text) for text in tqdm(df[text_column], desc="Processing text")]

    # Convert emotion data to a DataFrame
    emotions_df = pd.DataFrame(emotions_data)

    # Append emotion scores to the original DataFrame
    df = pd.concat([df, emotions_df], axis=1)

    return df

# Usage example
input_file_path = "text_data.xlsx"
output_file_path = "text_data_with_emotions.xlsx"

# Compute emotions and save the updated file
df_with_emotions = append_emotions_to_excel(input_file_path)
print("Saving updated dataset with emotion scores...")
df_with_emotions.to_excel(output_file_path, index=False)
print(f"Updated dataset saved to {output_file_path}")

