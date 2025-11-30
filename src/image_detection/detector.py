# src/image_detection/detector.py
# Deepfake Face Detection using Vision Transformer
# Uses pre-trained model from HuggingFace

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global detector instance for caching
_detector_instance = None


class ImageDetector:
    """
    Deepfake Face Detector using Vision Transformer.

    Uses the pre-trained Deep-Fake-Detector-v2-Model from HuggingFace
    which achieves ~92% accuracy on deepfake detection.
    """

    def __init__(self, model_name: str = "dima806/deepfake_vs_real_image_detection"):
        """
        Initialize the deepfake detector.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.loaded = False

    def load_model(self) -> bool:
        """
        Load the pre-trained ViT model from HuggingFace.

        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            # Determine device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            logger.info(f"Loading deepfake detector: {self.model_name}")
            logger.info(f"Using device: {self.device}")

            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)

            # Move to device and set eval mode
            self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            logger.info("Deepfake detector loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install torch transformers")
            return False
        except Exception as e:
            logger.error(f"Error loading deepfake detector: {e}")
            return False

    def predict(self, image) -> Dict[str, Any]:
        """
        Detect if an image contains a deepfake face.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with:
                - prediction: 'DEEPFAKE' or 'REAL'
                - confidence: float (0-1)
                - probability: dict with 'REAL' and 'DEEPFAKE' probabilities
                - raw_label: original model label
        """
        import torch
        from PIL import Image

        if not self.loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load deepfake detector model")

        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Get prediction
        predicted_idx = torch.argmax(logits, dim=1).item()
        raw_label = self.model.config.id2label[predicted_idx]

        # Normalize labels - the model uses 'Real' (0) and 'Fake' (1)
        is_fake = any(term in raw_label.lower() for term in ['fake', 'deepfake', 'ai', 'synthetic', 'generated'])

        # Model labels: {0: 'Real', 1: 'Fake'}
        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        return {
            'prediction': 'DEEPFAKE' if is_fake else 'REAL',
            'confidence': float(max(probs)),
            'probability': {
                'REAL': prob_real,
                'DEEPFAKE': prob_fake
            },
            'raw_label': raw_label
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.loaded,
            'labels': self.model.config.id2label if self.loaded else None
        }


def get_image_detector() -> Optional[ImageDetector]:
    """
    Get cached ImageDetector instance.

    Returns:
        ImageDetector instance or None if loading fails
    """
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = ImageDetector()
        if not _detector_instance.load_model():
            _detector_instance = None

    return _detector_instance


def is_detector_available() -> bool:
    """
    Check if the deepfake detector dependencies are available.

    Returns:
        True if torch and transformers are installed
    """
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Deepfake Detector Module")
    print("=" * 40)
    print(f"Dependencies available: {is_detector_available()}")

    if is_detector_available():
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        print("\nUsage:")
        print("  from src.image_detection import ImageDetector")
        print("  detector = ImageDetector()")
        print("  result = detector.predict(pil_image)")
        print("  print(result['prediction'], result['confidence'])")
