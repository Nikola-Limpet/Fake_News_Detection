# model_utils.py
# Utility functions for loading and using trained models

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class for loading and using saved models"""

    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.models = {}
        self.model_info = {}

    def load_vectorizer(self):
        """Load the saved vectorizer"""
        vectorizer_path = self.models_dir / 'vectorizer.pkl'

        if not vectorizer_path.exists():
            logger.warning(f"Vectorizer not found at {vectorizer_path}")
            return False

        try:
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Vectorizer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
            return False

    def load_model(self, model_name):
        """Load a specific trained model"""
        model_path = self.models_dir / f'{model_name}_model.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def load_all_models(self):
        """Load all available models"""
        if not self.models_dir.exists():
            logger.error(f"Models directory {self.models_dir} not found")
            return False

        # Load vectorizer first
        if not self.load_vectorizer():
            return False

        # Find all model files
        model_files = list(self.models_dir.glob('*_model.pkl'))

        if not model_files:
            logger.warning("No model files found")
            return False

        # Load each model
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            self.load_model(model_name)

        # Load training results if available
        self.load_training_results()

        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        return True

    def load_training_results(self):
        """Load training results and metrics"""
        results_path = self.models_dir / 'training_results.json'

        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    self.model_info = json.load(f)
                logger.info("Training results loaded")
            except Exception as e:
                logger.error(f"Error loading training results: {e}")

    def predict(self, text, model_name='logistic'):
        """Make prediction using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded")

        # Preprocess and vectorize text
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])

        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]

        return {
            'prediction': prediction,
            'probability': {
                'FAKE': probability[0] if model.classes_[0] == 'FAKE' else probability[1],
                'REAL': probability[1] if model.classes_[1] == 'REAL' else probability[0]
            },
            'confidence': max(probability)
        }

    def predict_batch(self, texts, model_name='logistic'):
        """Make predictions for multiple texts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        if self.vectorizer is None:
            raise ValueError("Vectorizer not loaded")

        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        texts_vec = self.vectorizer.transform(processed_texts)

        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(texts_vec)
        probabilities = model.predict_proba(texts_vec)

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'text': texts[i],
                'prediction': pred,
                'probability': {
                    'FAKE': prob[0] if model.classes_[0] == 'FAKE' else prob[1],
                    'REAL': prob[1] if model.classes_[1] == 'REAL' else prob[0]
                },
                'confidence': max(prob)
            })

        return results

    def get_model_info(self, model_name=None):
        """Get information about loaded models"""
        if model_name:
            return self.model_info.get(model_name, {})
        else:
            return self.model_info

    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())

    def preprocess_text(self, text):
        """Basic text preprocessing (matches training preprocessing)"""
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        if pd.isna(text) or not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        try:
            # Tokenize
            words = word_tokenize(text)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words and len(word) > 2]

            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]

            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}")
            return text

# Global model loader instance
_model_loader = None

def get_model_loader():
    """Get singleton model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_all_models()
    return _model_loader

def predict_text(text, model_name='logistic'):
    """Convenience function for making predictions"""
    loader = get_model_loader()
    return loader.predict(text, model_name)

def get_available_models():
    """Get list of available models"""
    loader = get_model_loader()
    return loader.get_available_models()

def get_model_info(model_name=None):
    """Get model information"""
    loader = get_model_loader()
    return loader.get_model_info(model_name)

# Example usage
if __name__ == "__main__":
    # Test the model loader
    loader = ModelLoader()

    if loader.load_all_models():
        print("Models loaded successfully!")
        print(f"Available models: {loader.get_available_models()}")

        # Test prediction
        test_text = "Breaking news: Scientists discover cure for cancer using simple home remedy!"

        for model_name in loader.get_available_models():
            try:
                result = loader.predict(test_text, model_name)
                print(f"\n{model_name} prediction:")
                print(f"  Text: {test_text[:50]}...")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Probabilities: FAKE={result['probability']['FAKE']:.3f}, REAL={result['probability']['REAL']:.3f}")
            except Exception as e:
                print(f"Error with {model_name}: {e}")
    else:
        print("Failed to load models")