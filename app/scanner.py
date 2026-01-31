import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def _find_document_contour(edged):
    """
    Find the largest 4-point contour in the edged image.
    """
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    raise ValueError("Document contour not found")


def scan_document(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Invalid image")

    # Resize for faster processing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    resized = imutils.resize(image, height=500)

    # Use RESIZED image for edge detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # Find document contour
    screen_cnt = _find_document_contour(edged)

    # Perspective transform (apply ratio to scale back to original)
    warped = four_point_transform(orig, screen_cnt.reshape(4, 2) * ratio)
    return warped

    # Convert to grayscale & apply adaptive threshold
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped_gray, 11, offset=10, method="gaussian")
    scanned = (warped_gray > T).astype("uint8") * 255

    return scanned
