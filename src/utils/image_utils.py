import cv2
import numpy as np
from PIL import Image
from typing import List, Union


def load_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an image from a file path or return the NumPy array as-is.

    Args:
        image: File path (str) or a BGR NumPy array.

    Returns:
        BGR NumPy array.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image from path: {image}")
        return img
    elif isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Provided image array is empty.")
        return image
    else:
        raise TypeError("Image must be a file path (str) or a NumPy array.")


def crop_region(image: np.ndarray, polygon: List[List[float]], padding: int = 2) -> Image.Image:
    """
    Crop a text region from the image using a 4-point polygon.

    The polygon is warped into a flat, axis-aligned rectangle using a
    perspective transform so that the recogniser receives a clean,
    de-skewed patch.

    Args:
        image:   BGR NumPy array (full-page image).
        polygon: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        padding: Extra pixels to expand the crop on each side.

    Returns:
        PIL.Image (RGB) of the cropped and de-skewed text region.
    """
    pts = np.array(polygon, dtype=np.float32)

    # Compute target width and height from the bounding polygon
    width_top    = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    height_left  = np.linalg.norm(pts[3] - pts[0])
    height_right = np.linalg.norm(pts[2] - pts[1])

    dst_w = int(max(width_top, width_bottom)) + 2 * padding
    dst_h = int(max(height_left, height_right)) + 2 * padding

    dst_pts = np.array(
        [[padding, padding],
         [dst_w - padding - 1, padding],
         [dst_w - padding - 1, dst_h - padding - 1],
         [padding, dst_h - padding - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))

    # Convert BGR → RGB for PIL
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def draw_results(
    image: np.ndarray,
    results: List[dict],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detected bounding boxes and recognised text onto the image.

    Args:
        image:     BGR NumPy array.
        results:   List of OCR result dicts with keys 'box', 'text', 'score'.
        color:     BGR colour for the bounding box lines.
        thickness: Line thickness in pixels.
        font_scale: OpenCV font scale for the overlaid text.

    Returns:
        Annotated BGR NumPy array.
    """
    vis = image.copy()
    for item in results:
        box = np.array(item["box"], dtype=np.int32)
        cv2.polylines(vis, [box], isClosed=True, color=color, thickness=thickness)

        text  = item.get("text", "")
        score = item.get("recognition_score")
        label = f"{text} ({score:.2f})" if score is not None else text

        # Place text just above the top-left corner of the box
        origin = (int(box[0][0]), max(int(box[0][1]) - 5, 10))
        cv2.putText(
            vis, label, origin,
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, thickness, cv2.LINE_AA,
        )
    return vis
