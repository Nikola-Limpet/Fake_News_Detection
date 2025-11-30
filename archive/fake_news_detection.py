# fake_news_detection.py
# Main training script for fake news detection models
# Usage: python fake_news_detection.py --model logistic --save-model

import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

# Download NLTK data
nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
for item in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{item}' if 'punkt' in item else f'corpora/{item}')
    except LookupError:
        nltk.download(item, quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsDetector:
    """Main class for fake news detection training and evaluation"""

    def __init__(self, config=None):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.results = {}
        self.config = config or self._load_default_config()

        # Create directories
        Path('models').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('outputs').mkdir(exist_ok=True)

    def _load_default_config(self):
        """Load default configuration"""
        return {
            'data_path': 'data/fake_news_data.csv',
            'test_size': 0.2,
            'random_state': 42,
            'vectorizer': {
                'type': 'tfidf',
                'max_features': 5000,
                'stop_words': 'english',
                'ngram_range': (1, 2)
            },
            'models': {
                'logistic': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                },
                'naive_bayes': {
                    'alpha': [0.1, 1, 10]
                },
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None]
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
        }

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Tokenize
        words = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words and len(word) > 2]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    def load_data(self):
        """Load and preprocess the dataset"""
        logger.info(f"Loading data from {self.config['data_path']}")

        try:
            df = pd.read_csv(self.config['data_path'])
            logger.info(f"Loaded {len(df)} samples")

            # Preprocess text
            logger.info("Preprocessing text...")
            df['processed_text'] = df['text'].apply(self.preprocess_text)

            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            logger.info(f"After preprocessing: {len(df)} samples")

            # Print class distribution
            class_dist = df['label'].value_counts()
            logger.info(f"Class distribution:\n{class_dist}")

            return df

        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config['data_path']}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_vectorizer(self):
        """Create text vectorizer"""
        config = self.config['vectorizer']

        if config['type'] == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=config['max_features'],
                stop_words=config['stop_words'],
                ngram_range=config['ngram_range']
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=config['max_features'],
                stop_words=config['stop_words'],
                ngram_range=config['ngram_range']
            )

        logger.info(f"Created {config['type']} vectorizer")

    def train_single_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a single model with hyperparameter tuning"""
        logger.info(f"Training {model_name} model...")

        # Define models
        models_map = {
            'logistic': LogisticRegression(random_state=self.config['random_state']),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(random_state=self.config['random_state']),
            'svm': SVC(random_state=self.config['random_state'], probability=True)
        }

        if model_name not in models_map:
            raise ValueError(f"Unknown model: {model_name}")

        model = models_map[model_name]
        param_grid = self.config['models'][model_name]

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')

        # Predictions for detailed metrics
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test == 'FAKE', y_pred_proba)

        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        self.models[model_name] = results

        logger.info(f"{model_name} - Test Accuracy: {test_score:.4f}, AUC: {auc_score:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        return results

    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train ensemble model combining best individual models"""
        logger.info("Training ensemble model...")

        # Get trained models
        estimators = []
        for name, result in self.models.items():
            estimators.append((name, result['model']))

        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)

        # Evaluate
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)

        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test == 'FAKE', y_pred_proba)

        results = {
            'model': ensemble,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': report,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        self.models['ensemble'] = results

        logger.info(f"Ensemble - Test Accuracy: {test_score:.4f}, AUC: {auc_score:.4f}")

        return results

    def evaluate_models(self, y_test):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")

        # Summary table
        summary = []
        for name, result in self.models.items():
            summary.append({
                'Model': name,
                'Test_Accuracy': result['test_accuracy'],
                'AUC_Score': result['auc_score'],
                'F1_Score': result['classification_report']['macro avg']['f1-score'],
                'Precision': result['classification_report']['macro avg']['precision'],
                'Recall': result['classification_report']['macro avg']['recall']
            })

        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)

        # Find best model
        best_model_name = summary_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]

        logger.info(f"Best model: {best_model_name}")
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(summary_df.to_string(index=False, float_format='%.4f'))

        # Save results
        summary_df.to_csv('outputs/model_comparison.csv', index=False)

        return summary_df

    def plot_results(self, y_test):
        """Generate visualization plots"""
        logger.info("Generating plots...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)

        # 1. Model Comparison
        summary = []
        for name, result in self.models.items():
            summary.append({
                'Model': name,
                'Accuracy': result['test_accuracy'],
                'AUC': result['auc_score']
            })

        summary_df = pd.DataFrame(summary)

        axes[0, 0].bar(summary_df['Model'], summary_df['Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. AUC Comparison
        axes[0, 1].bar(summary_df['Model'], summary_df['AUC'])
        axes[0, 1].set_title('Model AUC Comparison')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. ROC Curves
        for name, result in self.models.items():
            y_binary = (y_test == 'FAKE').astype(int)
            fpr, tpr, _ = roc_curve(y_binary, result['probabilities'])
            axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={result["auc_score"]:.3f})')

        axes[1, 0].plot([0, 1], [0, 1], 'k--')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()

        # 4. Confusion Matrix for best model
        best_name = max(self.models.keys(), key=lambda x: self.models[x]['test_accuracy'])
        cm = confusion_matrix(y_test, self.models[best_name]['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title(f'Confusion Matrix - {best_name}')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('outputs/evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Plots saved to outputs/evaluation_plots.png")

    def save_models(self, save_path='models'):
        """Save trained models and vectorizer"""
        logger.info(f"Saving models to {save_path}")

        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

        # Save vectorizer
        with open(save_path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # Save individual models
        for name, result in self.models.items():
            model_file = save_path / f'{name}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)

        # Save results
        results_file = save_path / 'training_results.json'
        results_to_save = {}
        for name, result in self.models.items():
            results_to_save[name] = {
                'best_params': result.get('best_params', {}),
                'test_accuracy': result['test_accuracy'],
                'auc_score': result['auc_score'],
                'classification_report': result['classification_report']
            }

        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info("Models saved successfully")

    def train_all(self, models_to_train=None):
        """Train all specified models"""
        # Load data
        df = self.load_data()

        # Split data
        X = df['processed_text']
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Create and fit vectorizer
        self.create_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train models
        if models_to_train is None:
            models_to_train = ['logistic', 'naive_bayes', 'random_forest']

        for model_name in models_to_train:
            try:
                self.train_single_model(model_name, X_train_vec, y_train, X_test_vec, y_test)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        # Train ensemble if multiple models
        if len(self.models) > 1:
            self.train_ensemble(X_train_vec, y_train, X_test_vec, y_test)

        # Evaluate
        self.evaluate_models(y_test)
        self.plot_results(y_test)

        return self.models

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--models', nargs='+', default=['logistic', 'naive_bayes', 'random_forest'],
                       choices=['logistic', 'naive_bayes', 'random_forest', 'svm'],
                       help='Models to train')
    parser.add_argument('--data', default='data/fake_news_data.csv',
                       help='Path to dataset')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--config', help='Path to config file')

    args = parser.parse_args()

    # Initialize detector
    detector = FakeNewsDetector()

    # Update config with command line args
    detector.config['data_path'] = args.data

    logger.info("Starting fake news detection training")
    logger.info(f"Models to train: {args.models}")

    try:
        # Train models
        results = detector.train_all(args.models)

        # Save models if requested
        if args.save_models:
            detector.save_models()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()