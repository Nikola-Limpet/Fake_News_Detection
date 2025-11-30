# explainability.py
# LIME-based text explainability for Fake News Detection
# Provides interpretable explanations for model predictions

import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TextExplainer:
    """LIME-based text explainer for fake news classification"""

    def __init__(self, model, vectorizer, class_names: List[str] = None):
        """
        Initialize the text explainer.

        Args:
            model: Trained sklearn classifier with predict_proba method
            vectorizer: Fitted TF-IDF vectorizer
            class_names: List of class names (default: ['REAL', 'FAKE'])
        """
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ['REAL', 'FAKE']
        self._explainer = None

    @property
    def explainer(self):
        """Lazy load LIME explainer to save memory"""
        if self._explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer
                self._explainer = LimeTextExplainer(
                    class_names=self.class_names,
                    bow=False,  # Don't use bag of words internally
                    split_expression=r'\W+',  # Split on non-word characters
                    random_state=42
                )
                logger.info("LIME explainer initialized")
            except ImportError:
                logger.error("LIME not installed. Run: pip install lime")
                raise ImportError("Please install lime: pip install lime")
        return self._explainer

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for LIME.

        Args:
            texts: List of text strings

        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def explain(self, text: str, num_features: int = 10, num_samples: int = 300) -> Dict[str, Any]:
        """
        Generate LIME explanation for a text.

        Args:
            text: Input text to explain
            num_features: Number of top features to include (default: 10)
            num_samples: Number of samples for LIME (default: 300, lower = faster)

        Returns:
            Dictionary containing:
                - words: List of important words
                - weights: List of feature weights (positive = FAKE, negative = REAL)
                - prediction_proba: Prediction probabilities
                - score: Local model fidelity score
        """
        try:
            exp = self.explainer.explain_instance(
                text,
                self.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=(0, 1)  # Both classes
            )

            # Get explanation as list of (word, weight) tuples
            # Positive weight = pushes toward class 1 (FAKE)
            # Negative weight = pushes toward class 0 (REAL)
            exp_list = exp.as_list(label=1)  # Get weights for FAKE class

            return {
                'words': [x[0] for x in exp_list],
                'weights': [x[1] for x in exp_list],
                'prediction_proba': {
                    'REAL': float(exp.predict_proba[0]),
                    'FAKE': float(exp.predict_proba[1])
                },
                'score': exp.score,
                'local_exp': exp_list
            }

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise

    def get_highlighted_html(self, text: str, explanation: Dict[str, Any],
                             max_words: int = 500) -> str:
        """
        Generate HTML with highlighted words based on explanation.

        Args:
            text: Original input text
            explanation: Explanation dict from explain() method
            max_words: Maximum words to process (for long texts)

        Returns:
            HTML string with highlighted words
        """
        words = text.split()[:max_words]
        word_weights = {w.lower(): wt for w, wt in zip(explanation['words'], explanation['weights'])}

        highlighted_parts = []
        for word in words:
            # Clean word for lookup
            clean_word = word.lower().strip('.,!?;:\'"()[]{}')

            if clean_word in word_weights:
                weight = word_weights[clean_word]
                # Scale opacity based on weight magnitude
                opacity = min(abs(weight) * 2, 0.8)

                if weight > 0:
                    # Positive weight = FAKE indicator (red)
                    color = f"rgba(255, 107, 107, {opacity})"
                    title = f"FAKE indicator (weight: {weight:.3f})"
                else:
                    # Negative weight = REAL indicator (green)
                    color = f"rgba(81, 207, 102, {opacity})"
                    title = f"REAL indicator (weight: {weight:.3f})"

                highlighted_parts.append(
                    f'<span style="background-color: {color}; padding: 2px 4px; '
                    f'border-radius: 3px; margin: 1px;" title="{title}">{word}</span>'
                )
            else:
                highlighted_parts.append(word)

        return ' '.join(highlighted_parts)

    def get_word_importance_data(self, explanation: Dict[str, Any]) -> Dict[str, List]:
        """
        Get word importance data formatted for visualization.

        Args:
            explanation: Explanation dict from explain() method

        Returns:
            Dictionary with 'words', 'weights', 'colors', 'directions' lists
        """
        words = explanation['words']
        weights = explanation['weights']

        # Determine colors and directions
        colors = []
        directions = []

        for w in weights:
            if w > 0:
                colors.append('#ff6b6b')  # Red for FAKE
                directions.append('FAKE')
            else:
                colors.append('#51cf66')  # Green for REAL
                directions.append('REAL')

        return {
            'words': words,
            'weights': weights,
            'colors': colors,
            'directions': directions
        }


def create_explainer(model, vectorizer) -> TextExplainer:
    """
    Factory function to create a TextExplainer instance.

    Args:
        model: Trained sklearn classifier
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        TextExplainer instance
    """
    return TextExplainer(model, vectorizer)


# Example usage and testing
if __name__ == "__main__":
    print("TextExplainer module loaded successfully")
    print("Usage:")
    print("  from explainability import TextExplainer")
    print("  explainer = TextExplainer(model, vectorizer)")
    print("  explanation = explainer.explain('Your text here')")
    print("  html = explainer.get_highlighted_html(text, explanation)")
