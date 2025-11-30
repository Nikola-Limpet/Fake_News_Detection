# gemini_utils.py
# Gemini AI integration for enhanced fake news explanations

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default API key (can be overridden by environment variable)
DEFAULT_API_KEY = "AIzaSyDtZw-yWTpY4iOWp0trhsc2xaWnH8hOL1M"


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment or default."""
    # Try environment variable first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    # Try Streamlit secrets
    try:
        import streamlit as st
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            return api_key
    except:
        pass

    # Fall back to default
    return DEFAULT_API_KEY


def is_gemini_available() -> bool:
    """Check if Gemini API is available."""
    try:
        import google.generativeai as genai
        return True
    except ImportError:
        return False


class GeminiExplainer:
    """Generate AI-powered explanations for fake news predictions."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini explainer.

        Args:
            api_key: Gemini API key (uses default if not provided)
        """
        self.api_key = api_key or get_gemini_api_key()
        self.model = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the Gemini model."""
        if self.initialized:
            return True

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.initialized = True
            logger.info("Gemini AI initialized successfully")
            return True

        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False

    def generate_explanation(
        self,
        text: str,
        prediction: str,
        confidence: float,
        key_words: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate AI explanation for a fake news prediction.

        Args:
            text: The article text analyzed
            prediction: 'FAKE' or 'REAL'
            confidence: Confidence score (0-1)
            key_words: Optional list of influential words from LIME

        Returns:
            Dictionary with explanation, indicators, and recommendations
        """
        if not self.initialize():
            return {
                "error": "Gemini AI not available",
                "explanation": None
            }

        # Truncate text if too long
        max_text_length = 2000
        truncated_text = text[:max_text_length] + "..." if len(text) > max_text_length else text

        # Build the prompt
        key_words_str = ", ".join(key_words[:10]) if key_words else "Not available"

        prompt = f"""You are an expert fact-checker. Analyze this news article classified as {prediction} ({confidence:.0%} ML confidence).

ARTICLE:
{truncated_text}

Provide a BRIEF analysis (max 150 words) in this exact format:

**AI Confidence:** [Give your own confidence 0-100% that this is {prediction} news]

**Why {prediction}:** [2-3 sentences explaining the main reasons]

**Key Indicators:**
- [Indicator 1]
- [Indicator 2]
- [Indicator 3]

**Verify:** [1 sentence on how to fact-check this]

Be concise and direct."""

        try:
            response = self.model.generate_content(prompt)

            return {
                "success": True,
                "explanation": response.text,
                "prediction": prediction,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "error": str(e),
                "explanation": None
            }

    def generate_quick_summary(
        self,
        text: str,
        prediction: str,
        confidence: float
    ) -> Optional[str]:
        """
        Generate a quick one-paragraph summary explanation.

        Args:
            text: The article text
            prediction: 'FAKE' or 'REAL'
            confidence: Confidence score

        Returns:
            Short explanation string or None if failed
        """
        if not self.initialize():
            return None

        # Truncate text
        truncated_text = text[:1000] + "..." if len(text) > 1000 else text

        prompt = f"""Analyze this news article classified as {prediction} ({confidence:.1%} confidence).

TEXT: {truncated_text}

In 2-3 sentences, explain the main reason why this appears to be {prediction.lower()} news. Be specific and mention key indicators from the text."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini quick summary error: {e}")
            return None


# Global instance for caching
_gemini_explainer = None


def get_gemini_explainer() -> Optional[GeminiExplainer]:
    """Get cached Gemini explainer instance."""
    global _gemini_explainer

    if _gemini_explainer is None:
        _gemini_explainer = GeminiExplainer()

    return _gemini_explainer


def explain_prediction(
    text: str,
    prediction: str,
    confidence: float,
    key_words: Optional[list] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate explanation.

    Args:
        text: Article text
        prediction: 'FAKE' or 'REAL'
        confidence: Confidence score
        key_words: Optional influential words

    Returns:
        Explanation dictionary
    """
    explainer = get_gemini_explainer()
    if explainer is None:
        return {"error": "Gemini not available", "explanation": None}

    return explainer.generate_explanation(text, prediction, confidence, key_words)


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Gemini AI Explainer Test")
    print("=" * 40)
    print(f"Gemini available: {is_gemini_available()}")

    if is_gemini_available():
        test_text = """BREAKING: Scientists discover that drinking lemon water every morning
        can cure cancer completely! Doctors are shocked and pharmaceutical companies are
        trying to hide this information from the public. Share this before it gets deleted!"""

        result = explain_prediction(
            text=test_text,
            prediction="FAKE",
            confidence=0.92,
            key_words=["cure", "shocked", "hiding", "share", "deleted"]
        )

        if result.get("success"):
            print("\nGenerated Explanation:")
            print(result["explanation"])
        else:
            print(f"\nError: {result.get('error')}")
