# transformer_model.py
# DistilBERT-based fake news classifier
# Optimized for local development (CPU/GPU support)

import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TransformerClassifier:
    """Lightweight transformer classifier for fake news detection"""

    def __init__(self, model_path: Optional[str] = None, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize the transformer classifier.

        Args:
            model_path: Path to fine-tuned model (if available)
            model_name: HuggingFace model name for base model
        """
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else Path('models/distilbert')
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 256  # Reduced from 512 for speed
        self.loaded = False
        self.class_names = ['REAL', 'FAKE']

    def load_model(self) -> bool:
        """
        Load pretrained/fine-tuned model.

        Returns:
            True if successful, False otherwise
        """
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                DistilBertTokenizer,
                DistilBertForSequenceClassification
            )

            # Check if fine-tuned model exists
            if self.model_path.exists() and (self.model_path / 'config.json').exists():
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_path)
                )
            else:
                # Load base model (for inference without fine-tuning)
                logger.info(f"Loading base model: {self.model_name}")
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2
                )

            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            logger.info(f"Transformer model loaded successfully on {self.device}")
            return True

        except ImportError as e:
            logger.error(f"Transformers library not installed: {e}")
            logger.error("Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on single text.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self.loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load transformer model")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        prediction = 'FAKE' if probs[1] > probs[0] else 'REAL'
        confidence = float(max(probs))

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': {
                'REAL': float(probs[0]),
                'FAKE': float(probs[1])
            }
        }

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple texts.

        Args:
            texts: List of texts to classify
            batch_size: Number of texts per batch

        Returns:
            List of prediction dictionaries
        """
        if not self.loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load transformer model")

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for prob in probs:
                prediction = 'FAKE' if prob[1] > prob[0] else 'REAL'
                results.append({
                    'prediction': prediction,
                    'confidence': float(max(prob)),
                    'probability': {'REAL': float(prob[0]), 'FAKE': float(prob[1])}
                })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'device': self.device,
            'max_length': self.max_length,
            'loaded': self.loaded,
            'class_names': self.class_names
        }


class TransformerTrainer:
    """Training utilities for fine-tuning DistilBERT on fake news data"""

    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 output_dir: str = 'models/distilbert'):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save trained model
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def prepare_dataset(self, texts: np.ndarray, labels: np.ndarray,
                        tokenizer, max_length: int = 256):
        """
        Prepare PyTorch dataset for training.

        Args:
            texts: Array of text samples
            labels: Array of labels ('FAKE' or 'REAL')
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length

        Returns:
            PyTorch Dataset
        """
        from torch.utils.data import Dataset

        class FakeNewsDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts.tolist(),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                # Convert string labels to integers
                self.labels = torch.tensor([1 if l == 'FAKE' else 0 for l in labels])

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        return FakeNewsDataset(texts, labels, tokenizer, max_length)

    def train(self, train_texts: np.ndarray, train_labels: np.ndarray,
              val_texts: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None,
              epochs: int = 2, batch_size: int = 16, learning_rate: float = 2e-5,
              max_samples: Optional[int] = None):
        """
        Fine-tune the model on fake news data.

        Args:
            train_texts: Training text samples
            train_labels: Training labels
            val_texts: Validation text samples (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_samples: Maximum samples to use (for faster testing)

        Returns:
            Trained model and tokenizer
        """
        from transformers import (
            DistilBertTokenizer,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments
        )

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {len(train_texts)}")

        # Optionally limit samples for faster training
        if max_samples:
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
            if val_texts is not None:
                val_texts = val_texts[:max_samples // 5]
                val_labels = val_labels[:max_samples // 5]
            logger.info(f"Limited to {max_samples} samples")

        # Load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels, tokenizer)
        val_dataset = None
        if val_texts is not None:
            val_dataset = self.prepare_dataset(val_texts, val_labels, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            eval_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False,
            fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            report_to='none',  # Disable wandb/tensorboard
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))

        logger.info(f"Model saved to {self.output_dir}")

        # Evaluate if validation set provided
        if val_dataset:
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")

        return model, tokenizer


def create_classifier(model_path: Optional[str] = None) -> TransformerClassifier:
    """
    Factory function to create a TransformerClassifier instance.

    Args:
        model_path: Optional path to fine-tuned model

    Returns:
        TransformerClassifier instance
    """
    return TransformerClassifier(model_path=model_path)


def is_transformer_available() -> bool:
    """Check if transformer dependencies are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("TransformerClassifier module loaded successfully")
    print(f"PyTorch available: {is_transformer_available()}")
    if is_transformer_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nUsage:")
    print("  from transformer_model import TransformerClassifier")
    print("  classifier = TransformerClassifier()")
    print("  result = classifier.predict('Your news article text')")
    print("  print(result)")
