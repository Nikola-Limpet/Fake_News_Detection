# src/image_detection/__init__.py
# Deepfake Face Detection Module

from .detector import ImageDetector, get_image_detector, is_detector_available

__all__ = ['ImageDetector', 'get_image_detector', 'is_detector_available']
