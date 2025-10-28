import json
import pandas as pd
import re
import string
import os

# -------------------------------
# Load Dataset
# -------------------------------
def load_dataset(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    data = [json.loads(line) for line in open(file_path, 'r')]
    return pd.DataFrame(data)

# -------------------------------
# Text Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# -------------------------------
# Main Program
# -------------------------------
def main():
    # Load data
    df = load_dataset("Sarcasm_Headlines_Dataset.json")
    
    # Save original headlines
    df[['headline']].head(25).to_csv("original_headlines.csv", index=False)
    
    # Apply preprocessing
    df['cleaned_headline'] = df['headline'].apply(clean_text)
    
    # Save cleaned headlines
    df[['cleaned_headline']].head(25).to_csv("cleaned_headlines.csv", index=False)
    
    print("✅ CSV files created successfully!")
    print("Files saved in the same folder as this script:")
    print("- original_headlines.csv")
    print("- cleaned_headlines.csv")

if __name__ == "__main__":
    main()
