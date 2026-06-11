from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """Configuration for the PaddleOCR text detection stage."""

    # Minimum confidence score to keep a detected box.
    min_score: float = 0.5


@dataclass
class RecognitionConfig:
    """Configuration for the VietOCR text recognition stage."""

    # VietOCR model name. Common choices: 'vgg_transformer', 'vgg_seq2seq'.
    config_name: str = "vgg_transformer"

    # Device to run inference on: 'cpu' or 'cuda:0'.
    device: str = "cpu"

    # Minimum recognition confidence to include a result (None = keep all).
    min_score: float = 0.0


@dataclass
class OCRConfig:
    """Top-level configuration for the full OCR pipeline."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)

    # Pixel padding added around each detected crop before recognition.
    crop_padding: int = 2

    # If True, sort detected boxes from top-to-bottom, left-to-right.
    sort_boxes: bool = True
