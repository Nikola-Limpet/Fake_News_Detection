# run_data_prep.py
# Script to prepare the fake news dataset without Kaggle dependency

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare_combined_dataset():
    """Combine and prepare the dataset"""
    print("Preparing dataset...")

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Try to load existing dataset files
    fake_path = data_dir / 'Fake.csv'
    real_path = data_dir / 'True.csv'

    if fake_path.exists() and real_path.exists():
        print("Loading existing dataset files...")

        # Load datasets
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)

        print(f"Loaded fake news: {len(fake_df)} samples")
        print(f"Loaded real news: {len(real_df)} samples")

        # Add labels
        fake_df['label'] = 'FAKE'
        real_df['label'] = 'REAL'

        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)

        # Create combined text from title and text if they exist
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'title' in df.columns and 'text' not in df.columns:
            df['text'] = df['title']

        print(f"Combined dataset: {len(df)} samples")

    else:
        print("CSV files not found. Please ensure Fake.csv and True.csv are in the data/ directory")
        return None

    # Clean the dataset
    df = clean_dataset(df)

    # Split the dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Save datasets
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'validation.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    df.to_csv(data_dir / 'fake_news_data.csv', index=False)

    print(f"\nDataset split:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    # Print class distribution
    print(f"\nClass distribution in training set:")
    print(train_df['label'].value_counts())

    print(f"\nClass distribution in validation set:")
    print(val_df['label'].value_counts())

    print(f"\nClass distribution in test set:")
    print(test_df['label'].value_counts())

    return df

def clean_dataset(df):
    """Clean and prepare the dataset"""

    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['text'])
    if len(df) < initial_size:
        print(f"Removed {initial_size - len(df)} duplicate entries")

    # Remove empty texts
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']

    # Remove very short texts (likely not real articles)
    df = df[df['text'].str.len() > 50]  # Reduced threshold for more data

    # Standardize labels
    df['label'] = df['label'].str.upper()
    df['label'] = df['label'].replace({'TRUE': 'REAL', 'FALSE': 'FAKE', '1': 'FAKE', '0': 'REAL'})

    # Keep only REAL and FAKE labels
    df = df[df['label'].isin(['REAL', 'FAKE'])]

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Final dataset size after cleaning: {len(df)} samples")

    return df

def main():
    """Main function to prepare the dataset"""

    print("="*60)
    print("FAKE NEWS DETECTION - DATA PREPARATION")
    print("="*60)

    # Create data directory
    Path('data').mkdir(exist_ok=True)

    # Prepare the dataset
    df = prepare_combined_dataset()

    if df is not None:
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - data/fake_news_data.csv (full dataset)")
        print("  - data/train.csv (training set)")
        print("  - data/validation.csv (validation set)")
        print("  - data/test.csv (test set)")
        print("\nYou can now run the training script:")
        print("  python fake_news_detection.py")
        print("\nOr start the Streamlit app:")
        print("  streamlit run streamlit_app.py")
    else:
        print("Data preparation failed. Please check that Fake.csv and True.csv exist in the data/ directory.")

if __name__ == "__main__":
    main()