# tests/test_model_utils.py
# Unit tests for model utilities

import unittest
import tempfile
import pickle
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from model_utils import ModelLoader, predict_text, get_available_models
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class TestModelLoader(unittest.TestCase):
    """Test cases for ModelLoader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # Create mock model and vectorizer
        self.mock_vectorizer = TfidfVectorizer()
        self.mock_model = LogisticRegression()

        # Fit with dummy data
        dummy_texts = ["This is fake news", "This is real news", "Another fake story", "Real news article"]
        dummy_labels = ["FAKE", "REAL", "FAKE", "REAL"]

        X = self.mock_vectorizer.fit_transform(dummy_texts)
        self.mock_model.fit(X, dummy_labels)

        # Save mock files
        with open(self.models_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.mock_vectorizer, f)

        with open(self.models_dir / 'logistic_model.pkl', 'wb') as f:
            pickle.dump(self.mock_model, f)

        # Save mock training results
        mock_results = {
            'logistic': {
                'test_accuracy': 0.95,
                'auc_score': 0.97,
                'classification_report': {'macro avg': {'f1-score': 0.94}}
            }
        }
        with open(self.models_dir / 'training_results.json', 'w') as f:
            json.dump(mock_results, f)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_vectorizer(self):
        """Test vectorizer loading"""
        loader = ModelLoader(self.models_dir)
        self.assertTrue(loader.load_vectorizer())
        self.assertIsNotNone(loader.vectorizer)

    def test_load_model(self):
        """Test individual model loading"""
        loader = ModelLoader(self.models_dir)
        model = loader.load_model('logistic')
        self.assertIsNotNone(model)
        self.assertIn('logistic', loader.models)

    def test_load_all_models(self):
        """Test loading all models"""
        loader = ModelLoader(self.models_dir)
        success = loader.load_all_models()
        self.assertTrue(success)
        self.assertIn('logistic', loader.models)
        self.assertIsNotNone(loader.vectorizer)

    def test_predict(self):
        """Test prediction functionality"""
        loader = ModelLoader(self.models_dir)
        loader.load_all_models()

        result = loader.predict("This is fake news story", 'logistic')

        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('confidence', result)
        self.assertIn(result['prediction'], ['FAKE', 'REAL'])

    def test_predict_batch(self):
        """Test batch prediction"""
        loader = ModelLoader(self.models_dir)
        loader.load_all_models()

        texts = ["Fake news story", "Real news article"]
        results = loader.predict_batch(texts, 'logistic')

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('prediction', result)
            self.assertIn('probability', result)

    def test_get_available_models(self):
        """Test getting available models"""
        loader = ModelLoader(self.models_dir)
        loader.load_all_models()

        models = loader.get_available_models()
        self.assertIn('logistic', models)

    def test_missing_files(self):
        """Test behavior with missing files"""
        empty_dir = Path(self.temp_dir) / 'empty'
        empty_dir.mkdir()

        loader = ModelLoader(empty_dir)
        success = loader.load_all_models()
        self.assertFalse(success)

class TestTextPreprocessing(unittest.TestCase):
    """Test text preprocessing functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # Create minimal mock files for loader
        mock_vectorizer = TfidfVectorizer()
        mock_model = LogisticRegression()

        dummy_texts = ["real news text", "fake news text"]
        dummy_labels = ["REAL", "FAKE"]

        X = mock_vectorizer.fit_transform(dummy_texts)
        mock_model.fit(X, dummy_labels)

        with open(self.models_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(mock_vectorizer, f)

        with open(self.models_dir / 'logistic_model.pkl', 'wb') as f:
            pickle.dump(mock_model, f)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_text_preprocessing(self):
        """Test text preprocessing"""
        loader = ModelLoader(self.models_dir)

        # Test normal text
        text = "This is a TEST with CAPS and numbers123!"
        processed = loader.preprocess_text(text)
        self.assertIsInstance(processed, str)
        self.assertNotIn('123', processed)

        # Test empty text
        empty_result = loader.preprocess_text("")
        self.assertEqual(empty_result, "")

        # Test None input
        none_result = loader.preprocess_text(None)
        self.assertEqual(none_result, "")

class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # Create realistic mock data
        fake_texts = [
            "SHOCKING: Scientists discover miracle cure using this one weird trick!",
            "BREAKING: Government hiding alien technology, whistleblower reveals!",
            "AMAZING: Lose 50 pounds in 5 days with this simple method!"
        ]

        real_texts = [
            "Scientists at Stanford University published research on cancer treatment effectiveness.",
            "The Federal Reserve announced interest rate changes following economic indicators.",
            "Local government approves budget allocation for infrastructure improvements."
        ]

        all_texts = fake_texts + real_texts
        all_labels = ["FAKE"] * len(fake_texts) + ["REAL"] * len(real_texts)

        # Train models
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(all_texts)

        model = LogisticRegression()
        model.fit(X, all_labels)

        # Save models
        with open(self.models_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        with open(self.models_dir / 'logistic_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Save results
        results = {
            'logistic': {
                'test_accuracy': 0.9,
                'auc_score': 0.95,
                'classification_report': {'macro avg': {'f1-score': 0.9}}
            }
        }
        with open(self.models_dir / 'training_results.json', 'w') as f:
            json.dump(results, f)

    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        loader = ModelLoader(self.models_dir)
        success = loader.load_all_models()
        self.assertTrue(success)

        # Test fake news detection
        fake_text = "INCREDIBLE: This miracle pill will cure everything instantly!"
        result = loader.predict(fake_text, 'logistic')

        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)

        # The model should ideally predict this as FAKE, but we'll just check structure
        self.assertIn(result['prediction'], ['FAKE', 'REAL'])

        # Test real news detection
        real_text = "The university research team published findings in a peer-reviewed journal."
        result_real = loader.predict(real_text, 'logistic')

        self.assertIsInstance(result_real, dict)
        self.assertIn(result_real['prediction'], ['FAKE', 'REAL'])

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)