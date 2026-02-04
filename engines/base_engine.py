"""Base class for style transfer engines."""

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class BaseStyleEngine(ABC):
    """Abstract base class for all style transfer engines."""

    def __init__(self, device: str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_models(self) -> None:
        """Load model weights and initialize pipeline."""
        pass

    @abstractmethod
    def stylize(
        self,
        content_image: Image.Image | bytes,
        **kwargs
    ) -> bytes:
        """
        Apply style transfer to content image.
        Returns JPEG bytes of stylized image.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return engine name."""
        pass

    @property
    @abstractmethod
    def speed_rating(self) -> str:
        """Return speed rating: 'realtime', 'fast', 'medium', 'slow'."""
        pass

    @property
    @abstractmethod
    def quality_rating(self) -> int:
        """Return quality rating: 1-5."""
        pass
