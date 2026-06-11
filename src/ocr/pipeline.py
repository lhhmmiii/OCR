"""
OCR Pipeline
============
Chains PaddleOCR text detection with VietOCR text recognition into a
single, easy-to-use class.

Typical usage
-------------
>>> from src.ocr.pipeline import OCRPipeline
>>> pipeline = OCRPipeline()
>>> results = pipeline.run("image.jpg")
>>> for r in results:
...     print(r["text"], r["box"])
"""

import os
import logging
from typing import Union, List, Dict, Any, Optional

import cv2
import numpy as np

from src.ocr.detect_text   import detect_text
from src.ocr.recognize_text import TextRecognizer
from src.schemas.ocr_config  import OCRConfig, DetectionConfig, RecognitionConfig
from src.utils.image_utils   import load_image, crop_region, draw_results

logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    Full OCR pipeline: detect text regions → crop → recognise.

    Args:
        config (OCRConfig | None): Pipeline configuration. When None, sane
            defaults are used (CPU inference, vgg_transformer recogniser).

    Example
    -------
    >>> pipeline = OCRPipeline()
    >>> results  = pipeline.run("receipt.jpg")
    >>> for r in results:
    ...     print(r["text"])
    """

    def __init__(self, config: Optional[OCRConfig] = None) -> None:
        self.config = config or OCRConfig()
        self._recognizer: Optional[TextRecognizer] = None  # lazy init

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def recognizer(self) -> TextRecognizer:
        """Lazily initialise the TextRecognizer on first access."""
        if self._recognizer is None:
            rec_cfg = self.config.recognition
            self._recognizer = TextRecognizer(
                config_name=rec_cfg.config_name,
                device=rec_cfg.device,
            )
        return self._recognizer

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        image: Union[str, np.ndarray],
        save_vis: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the full OCR pipeline on a single image.

        Args:
            image (str | np.ndarray): Path to an image file or a BGR NumPy
                array.
            save_vis (str | None): If given, an annotated copy of the image
                is written to this file path.

        Returns:
            List[Dict[str, Any]]: One dict per detected text region::

                {
                    "text":              str,           # recognised text
                    "box":               [[x,y], ...],  # 4 corner points
                    "detection_score":   float | None,  # detection confidence
                    "recognition_score": float | None,  # recognition confidence
                }

        The list is ordered top-to-bottom, left-to-right when
        ``OCRConfig.sort_boxes`` is True.
        """
        # 1 ── Load image ─────────────────────────────────────────────────
        bgr_image = load_image(image)
        logger.debug("Image loaded: shape=%s", bgr_image.shape)

        # 2 ── Detect text regions ─────────────────────────────────────────
        det_cfg = self.config.detection
        detections = detect_text(bgr_image, min_score=det_cfg.min_score)
        logger.info("Detected %d region(s).", len(detections))

        if not detections:
            logger.warning("No text regions detected.")
            return []

        # 3 ── Optionally sort boxes (top-to-bottom, left-to-right) ────────
        if self.config.sort_boxes:
            detections = _sort_boxes(detections)

        # 4 ── Crop → Recognise ────────────────────────────────────────────
        rec_cfg = self.config.recognition
        results: List[Dict[str, Any]] = []

        for det in detections:
            box = det["box"]
            try:
                crop = crop_region(bgr_image, box, padding=self.config.crop_padding)
                text, prob = self.recognizer.recognize(crop, return_prob=True)
            except Exception as exc:
                logger.error("Failed to process box %s: %s", box, exc)
                text, prob = "", None

            if prob is not None and prob < rec_cfg.min_score:
                continue  # skip low-confidence recognitions

            results.append({
                "text":              text,
                "box":               box,
                "detection_score":   det.get("score"),
                "recognition_score": prob,
            })

        logger.info("Recognised %d region(s).", len(results))

        # 5 ── (Optional) Save annotated visualisation ─────────────────────
        if save_vis is not None:
            vis = draw_results(bgr_image, results)
            os.makedirs(os.path.dirname(os.path.abspath(save_vis)), exist_ok=True)
            cv2.imwrite(save_vis, vis)
            logger.info("Visualisation saved to: %s", save_vis)

        return results

    def run_batch(
        self,
        images: List[Union[str, np.ndarray]],
        save_vis_dir: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run the pipeline on a list of images.

        Args:
            images:       List of image paths or BGR NumPy arrays.
            save_vis_dir: If given, annotated images are saved to this
                          directory with filenames ``result_0.jpg``,
                          ``result_1.jpg``, …

        Returns:
            A list of result lists, one per input image.
        """
        all_results = []
        for idx, img in enumerate(images):
            save_path: Optional[str] = None
            if save_vis_dir is not None:
                save_path = os.path.join(save_vis_dir, f"result_{idx}.jpg")
            all_results.append(self.run(img, save_vis=save_path))
        return all_results

    def get_texts(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract only the recognised text strings from pipeline results.

        Args:
            results: Output of :meth:`run`.

        Returns:
            List of text strings, in reading order.
        """
        return [r["text"] for r in results]

    def get_full_text(
        self,
        results: List[Dict[str, Any]],
        separator: str = "\n",
    ) -> str:
        """
        Join all recognised texts into a single string.

        Args:
            results:   Output of :meth:`run`.
            separator: String inserted between consecutive lines.

        Returns:
            Single concatenated text string.
        """
        return separator.join(self.get_texts(results))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort bounding boxes from top-to-bottom, then left-to-right.

    Each box's y-coordinate is the mean y of its top two corners;
    its x-coordinate is the mean x of its left two corners.
    """
    def _key(det: Dict[str, Any]):
        box = det["box"]
        # top-left and top-right corners
        y_top = (box[0][1] + box[1][1]) / 2.0
        x_left = (box[0][0] + box[3][0]) / 2.0
        return (y_top, x_left)

    return sorted(detections, key=_key)
