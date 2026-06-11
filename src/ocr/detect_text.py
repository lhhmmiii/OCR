import os
import cv2
import numpy as np
from typing import Union, List, Dict, Any
from paddleocr import TextDetection


def detect_text(
    image: Union[str, np.ndarray],
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Detect text regions in an image using PaddleOCR TextDetection.

    Args:
        image (Union[str, np.ndarray]): Path to the input image or a BGR
            NumPy array representing the image.
        min_score (float): Minimum detection confidence score to keep a box.
            Boxes with a score below this threshold are discarded.

    Returns:
        List[Dict[str, Any]]: A list of detected text regions, each containing:
            - 'box'  : List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
            - 'score': Detection confidence (float) or None if not available.

    Raises:
        ValueError: If the image path does not exist or the NumPy array is empty.
        TypeError:  If the input is neither a str nor a NumPy array.
    """
    # ── Input validation ────────────────────────────────────────────────────
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError(f"Image path does not exist: {image}")
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Provided image array is empty.")
    else:
        raise TypeError("Image must be a file path (str) or a NumPy array.")

    # ── Detection ───────────────────────────────────────────────────────────
    text_detector = TextDetection()
    results = text_detector.predict(image)

    formatted_results: List[Dict[str, Any]] = []
    for item in results:
        polys  = item.get("dt_polys", [])
        scores = item.get("dt_scores", [])

        for poly, score in zip(polys, scores):
            conf = float(score) if score is not None else None
            if conf is not None and conf < min_score:
                continue  # filter low-confidence detections
            formatted_results.append({
                "box":   poly.tolist() if hasattr(poly, "tolist") else poly,
                "score": conf,
            })

    return formatted_results


if __name__ == "__main__":
    image_path = "../../data/raw/train_images/mcocr_public_145013aagqw.jpg"
    try:
        detected = detect_text(image_path)
        print(f"Found {len(detected)} text region(s):")
        for i, r in enumerate(detected):
            print(f"  [{i}] score={r['score']:.3f}  box={r['box']}")
    except Exception as e:
        print(f"Error during text detection: {e}")