#!/usr/bin/env python3
"""
train_transformer.py
Script to fine-tune DistilBERT on the fake news dataset.

Usage:
    python train_transformer.py                    # Full training
    python train_transformer.py --quick            # Quick test with 1000 samples
    python train_transformer.py --epochs 3         # Custom epochs
    python train_transformer.py --batch-size 8     # Smaller batch for low memory

Requirements:
    pip install torch transformers pandas scikit-learn tqdm
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available - training will be slow on CPU")
    except ImportError:
        missing.append('torch')

    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        missing.append('transformers')

    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Install with: pip install torch transformers")
        return False

    return True


def load_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load and prepare the dataset.

    Args:
        data_path: Path to the CSV file
        sample_size: Optional sample size for quick testing

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        # Try to find appropriate columns
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower()]
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]

        if text_cols and label_cols:
            df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
        else:
            raise ValueError(f"Could not find text/label columns. Columns: {df.columns.tolist()}")

    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 50]  # Remove very short texts

    # Standardize labels
    label_map = {
        'fake': 'FAKE', 'FAKE': 'FAKE', 'Fake': 'FAKE', '1': 'FAKE', 1: 'FAKE',
        'real': 'REAL', 'REAL': 'REAL', 'Real': 'REAL', '0': 'REAL', 0: 'REAL',
        'true': 'REAL', 'TRUE': 'REAL', 'True': 'REAL'
    }
    df['label'] = df['label'].map(lambda x: label_map.get(x, x))
    df = df[df['label'].isin(['FAKE', 'REAL'])]

    logger.info(f"After cleaning: {len(df)} samples")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} samples for training")

    return df


def evaluate_model(model, tokenizer, test_texts, test_labels, device='cpu'):
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_texts: Test texts
        test_labels: Test labels
        device: Device to use

    Returns:
        Dictionary with evaluation metrics
    """
    import torch

    model.eval()
    predictions = []

    logger.info("Evaluating model...")

    # Process in batches
    batch_size = 16
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i + batch_size].tolist()

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(['FAKE' if p == 1 else 'REAL' for p in preds])

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)

    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(test_labels, predictions)}")

    return {
        'accuracy': accuracy,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for fake news detection')
    parser.add_argument('--data', type=str, default='data/fake_news_data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/distilbert',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to use (for testing)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training with 1000 samples')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test set split ratio')

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.max_samples = 1000
        args.epochs = 1
        logger.info("Quick mode: Using 1000 samples and 1 epoch")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    import torch
    from transformer_model import TransformerTrainer

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Available data files:")
        for f in Path('data').glob('*.csv'):
            logger.info(f"  - {f}")
        sys.exit(1)

    df = load_data(str(data_path), args.max_samples)

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=args.test_split,
        random_state=42,
        stratify=df['label'].values
    )

    # Further split training into train/validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels
    )

    logger.info(f"\nDataset splits:")
    logger.info(f"  Training: {len(train_texts)} samples")
    logger.info(f"  Validation: {len(val_texts)} samples")
    logger.info(f"  Test: {len(test_texts)} samples")

    # Initialize trainer
    trainer = TransformerTrainer(
        model_name='distilbert-base-uncased',
        output_dir=args.output
    )

    # Train model
    logger.info("\nStarting training...")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model, tokenizer = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Evaluate on test set
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = evaluate_model(model, tokenizer, test_texts, test_labels, device)

    # Save results
    import json
    results_path = Path(args.output) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'test_samples': len(test_texts),
            'classification_report': results['report']
        }, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Model saved to: {args.output}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Test accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
