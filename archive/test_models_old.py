# test_models.py
# Comprehensive testing and evaluation suite for fake news detection models

import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your modules (assuming they're in src/ directory)
import sys
sys.path.append('src/')

class ModelEvaluator:
    """Comprehensive model evaluation suite"""
    
    def __init__(self, models_path='models/', data_path='data/'):
        self.models_path = Path(models_path)
        self.data_path = Path(data_path)
        self.loaded_models = {}
        self.evaluation_results = {}
        
    def load_all_models(self):
        """Load all saved models"""
        print("Loading models...")
        
        model_files = {
            'naive_bayes': 'naive_bayes_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'tfidf_vectorizer': 'tfidf_vectorizer.pkl',
            'lstm_tokenizer': 'lstm_tokenizer.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = self.models_path / filename
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        self.loaded_models[model_name] = pickle.load(f)
                    print(f"✓ Loaded {model_name}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
            else:
                print(f"✗ {model_name} not found")
        
        # Try to load deep learning models
        try:
            from tensorflow.keras.models import load_model
            
            lstm_path = self.models_path / 'best_lstm_model.h5'
            if lstm_path.exists():
                self.loaded_models['lstm'] = load_model(lstm_path)
                print("✓ Loaded LSTM model")
            
            bert_path = self.models_path / 'best_bert_lstm_model.h5'
            if bert_path.exists():
                self.loaded_models['bert'] = load_model(bert_path)
                print("✓ Loaded BERT model")
        except ImportError:
            print("✗ TensorFlow not available for deep learning models")
    
    def load_test_data(self):
        """Load test dataset"""
        test_path = self.data_path / 'test.csv'
        if not test_path.exists():
            test_path = self.data_path / 'fake_news_data.csv'
        
        if test_path.exists():
            df = pd.read_csv(test_path)
            print(f"Loaded test data: {len(df)} samples")
            return df
        else:
            print("Test data not found. Creating sample data...")
            return self.create_sample_test_data()
    
    def create_sample_test_data(self):
        """Create sample test data"""
        samples = {
            'text': [
                "Breaking scientific discovery announced by research team at major university",
                "SHOCKING: Secret cure hidden by doctors revealed! Click now!",
                "Government announces new policy after extensive consultation",
                "You won't believe what this celebrity did! Illuminati confirmed!",
                "Market analysis shows steady growth in technology sector"
            ] * 4,
            'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL'] * 4
        }
        return pd.DataFrame(samples)
    
    def evaluate_classical_models(self, X_test, y_test):
        """Evaluate classical ML models"""
        results = {}
        
        if 'tfidf_vectorizer' in self.loaded_models:
            X_test_tfidf = self.loaded_models['tfidf_vectorizer'].transform(X_test)
            
            # Naive Bayes
            if 'naive_bayes' in self.loaded_models:
                print("\nEvaluating Naive Bayes...")
                y_pred = self.loaded_models['naive_bayes'].predict(X_test_tfidf)
                results['naive_bayes'] = self.calculate_metrics(y_test, y_pred)
            
            # Random Forest (if using TF-IDF features)
            if 'random_forest' in self.loaded_models:
                print("\nEvaluating Random Forest...")
                y_pred = self.loaded_models['random_forest'].predict(X_test_tfidf.toarray())
                results['random_forest'] = self.calculate_metrics(y_test, y_pred)
        
        return results
    
    def evaluate_deep_learning_models(self, X_test, y_test):
        """Evaluate deep learning models"""
        results = {}
        
        # LSTM evaluation
        if 'lstm' in self.loaded_models and 'lstm_tokenizer' in self.loaded_models:
            print("\nEvaluating LSTM...")
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            tokenizer = self.loaded_models['lstm_tokenizer']
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            X_test_pad = pad_sequences(X_test_seq, maxlen=100)
            
            y_pred_prob = self.loaded_models['lstm'].predict(X_test_pad)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            results['lstm'] = self.calculate_metrics(y_test, y_pred)
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, matthews_corrcoef
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Calculate AUC if possible
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            pass
        
        return metrics
    
    def run_robustness_tests(self):
        """Test model robustness with adversarial examples"""
        print("\n" + "="*50)
        print("ROBUSTNESS TESTING")
        print("="*50)
        
        test_cases = {
            'typos': self.test_typos,
            'caps': self.test_capitalization,
            'punctuation': self.test_punctuation,
            'length': self.test_different_lengths
        }
        
        robustness_results = {}
        
        for test_name, test_func in test_cases.items():
            print(f"\nRunning {test_name} test...")
            robustness_results[test_name] = test_func()
        
        return robustness_results
    
    def test_typos(self):
        """Test model performance with typos"""
        original = "Scientists discover new treatment for disease"
        typo_version = "Scintists discovr new treatmnt for desease"
        
        results = {}
        # Test each model with typo version
        # (Implementation depends on your model structure)
        return results
    
    def test_capitalization(self):
        """Test model sensitivity to capitalization"""
        original = "Breaking news about important discovery"
        caps_version = "BREAKING NEWS ABOUT IMPORTANT DISCOVERY"
        
        results = {}
        # Test each model
        return results
    
    def test_punctuation(self):
        """Test model with excessive punctuation"""
        original = "This is an important announcement"
        punct_version = "This is an important announcement!!!!!!"
        
        results = {}
        # Test each model
        return results
    
    def test_different_lengths(self):
        """Test model with different text lengths"""
        texts = {
            'very_short': "Breaking news",
            'short': "Scientists make important discovery about climate change",
            'medium': "Scientists at leading university make important discovery about climate change that could impact millions of people worldwide according to new research published today",
            'long': "Scientists at leading university make important discovery about climate change " * 10
        }
        
        results = {}
        # Test each model
        return results
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = f"""
# Model Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
Total models evaluated: {len(self.evaluation_results)}

## Detailed Results
"""
        
        for model_name, metrics in self.evaluation_results.items():
            report += f"""
### {model_name.upper()}
- Accuracy: {metrics.get('accuracy', 'N/A'):.4f}
- Precision: {metrics.get('precision', 'N/A'):.4f}
- Recall: {metrics.get('recall', 'N/A'):.4f}
- F1-Score: {metrics.get('f1_score', 'N/A'):.4f}
- MCC: {metrics.get('mcc', 'N/A'):.4f}
"""
            
            if 'auc' in metrics:
                report += f"- AUC: {metrics['auc']:.4f}\n"
        
        return report
    
    def visualize_results(self):
        """Create comprehensive visualization of results"""
        if not self.evaluation_results:
            print("No results to visualize")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison bar chart
        models = list(self.evaluation_results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics_names):
            values = [self.evaluation_results[m].get(metric, 0) for m in models]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.upper() for m in models])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Confusion matrices
        ax = axes[0, 1]
        if 'naive_bayes' in self.evaluation_results:
            cm = self.evaluation_results['naive_bayes'].get('confusion_matrix', [[0,0],[0,0]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Naive Bayes Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # 3. ROC curves (if AUC available)
        ax = axes[1, 0]
        for model_name, metrics in self.evaluation_results.items():
            if 'auc' in metrics:
                # Simplified ROC curve visualization
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                # Would need actual FPR/TPR values for real ROC curve
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Metrics heatmap
        ax = axes[1, 1]
        metrics_matrix = []
        for model in models:
            row = [
                self.evaluation_results[model].get('accuracy', 0),
                self.evaluation_results[model].get('precision', 0),
                self.evaluation_results[model].get('recall', 0),
                self.evaluation_results[model].get('f1_score', 0)
            ]
            metrics_matrix.append(row)
        
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', 
                   xticklabels=metrics_names, 
                   yticklabels=[m.upper() for m in models],
                   cmap='YlOrRd', ax=ax)
        ax.set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as 'evaluation_results.png'")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("="*60)
        print("COMPLETE MODEL EVALUATION")
        print("="*60)
        
        # Load models
        self.load_all_models()
        
        if not self.loaded_models:
            print("\nNo models found for evaluation!")
            return
        
        # Load test data
        test_df = self.load_test_data()
        
        # Prepare data
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        # Clean text (simplified)
        X_test = test_df['text'].str.lower()
        y_test = le.fit_transform(test_df['label'])
        
        # Evaluate classical models
        print("\n" + "="*40)
        print("EVALUATING CLASSICAL MODELS")
        print("="*40)
        classical_results = self.evaluate_classical_models(X_test, y_test)
        self.evaluation_results.update(classical_results)
        
        # Evaluate deep learning models
        print("\n" + "="*40)
        print("EVALUATING DEEP LEARNING MODELS")
        print("="*40)
        dl_results = self.evaluate_deep_learning_models(X_test, y_test)
        self.evaluation_results.update(dl_results)
        
        # Run robustness tests
        robustness_results = self.run_robustness_tests()
        
        # Generate report
        report = self.generate_report()
        print("\n" + "="*40)
        print("EVALUATION REPORT")
        print("="*40)
        print(report)
        
        # Save report
        with open('evaluation_report.md', 'w') as f:
            f.write(report)
        print("\nReport saved to 'evaluation_report.md'")
        
        # Save detailed results
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.evaluation_results.keys()),
            'results': self.evaluation_results,
            'robustness_tests': robustness_results
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        print("Detailed results saved to 'evaluation_results.json'")
        
        # Visualize results
        self.visualize_results()
        
        return self.evaluation_results


class UnitTests(unittest.TestCase):
    """Unit tests for the fake news detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_texts = [
            "This is a test news article",
            "BREAKING: Test news alert!",
            ""  # Empty text
        ]
        
    def test_text_preprocessing(self):
        """Test text preprocessing functions"""
        from preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Test normal text
        result = preprocessor.clean_text("This is a TEST! With punctuation...")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), result)  # Check lowercase
        
        # Test empty text
        result = preprocessor.clean_text("")
        self.assertEqual(result, "")
        
        # Test None
        result = preprocessor.clean_text(None)
        self.assertEqual(result, "")
    
    def test_model_loading(self):
        """Test model loading functionality"""
        evaluator = ModelEvaluator()
        evaluator.load_all_models()
        
        # Check if at least one model loaded
        self.assertGreater(len(evaluator.loaded_models), 0, 
                          "No models could be loaded")
    
    def test_prediction_format(self):
        """Test prediction output format"""
        # This would test your prediction functions
        pass
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check value ranges
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)


def run_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run evaluation
    print("\n2. Running model evaluation...")
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.run_complete_evaluation()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Unit tests passed: {result.wasSuccessful()}")
    print(f"Models evaluated: {len(evaluation_results)}")
    
    if evaluation_results:
        best_model = max(evaluation_results.items(), 
                        key=lambda x: x[1].get('f1_score', 0))
        print(f"Best performing model: {best_model[0].upper()}")
        print(f"Best F1-Score: {best_model[1]['f1_score']:.4f}")


if __name__ == "__main__":
    run_tests()