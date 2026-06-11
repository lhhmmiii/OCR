import os
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class TextRecognizer:
    """
    Recognise text in a cropped image patch using VietOCR.

    The recogniser is initialised once and reused across calls so that the
    model is not reloaded on every prediction.

    Args:
        config_name (str): VietOCR architecture name.
            Common choices: 'vgg_transformer' (default), 'vgg_seq2seq'.
        device (str): Inference device, e.g. 'cpu' or 'cuda:0'.
        weights_path (str | None): Optional path to custom pretrained weights.
            When None, VietOCR downloads the default weights automatically.
    """

    def __init__(
        self,
        config_name: str = "vgg_transformer",
        device: str = "cpu",
        weights_path: Optional[str] = None,
    ) -> None:
        self.config = Cfg.load_config_from_name(config_name)
        self.config["device"] = device
        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            self.config["weights"] = weights_path
        self.predictor = Predictor(self.config)

    # ── Public API ────────────────────────────────────────────────────────

    def recognize(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_prob: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        """
        Recognise text in a single image (crop).

        Args:
            image: File path, BGR NumPy array, or PIL Image.
            return_prob: If True, also return the prediction confidence.

        Returns:
            text (str) when ``return_prob=False``.
            (text, confidence) tuple when ``return_prob=True``.
        """
        pil_img = self._to_pil(image)
        if return_prob:
            text, prob = self.predictor.predict(pil_img, return_prob=True)
            return text, float(prob)
        else:
            text = self.predictor.predict(pil_img, return_prob=False)
            return text

    def recognize_batch(
        self,
        images: list,
        return_prob: bool = False,
    ) -> list:
        """
        Recognise text in a batch of images.

        Args:
            images: List of file paths, BGR NumPy arrays, or PIL Images.
            return_prob: If True, each element is a (text, confidence) tuple.

        Returns:
            List of text strings (or (text, conf) tuples).
        """
        return [self.recognize(img, return_prob=return_prob) for img in images]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various image formats to a PIL Image (RGB)."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            import cv2  # local import to keep top-level deps minimal
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, numpy.ndarray, or PIL.Image."
            )


if __name__ == "__main__":
    recognizer = TextRecognizer()
    sample = "../../data/text_recognition/mcocr_public_145013aagqw_9.jpg"
    text, conf = recognizer.recognize(sample, return_prob=True)
    print(f"Text : {text}")
    print(f"Confidence: {conf:.4f}")
