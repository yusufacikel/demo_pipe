import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Optional, Tuple


def clip_depth(depth, lower_bound, upper_bound):
    valid_mask = (depth >= lower_bound) & (depth <= upper_bound)
    depth_clipped = depth * valid_mask.astype(depth.dtype)
    return depth_clipped

def fit_2dcircle(depth: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, float]]:
    depth_norm = cv2.normalize(depth, np.zeros_like(depth, dtype=np.uint8), 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_uint8 = depth_norm.astype(np.uint8)

    _, binary_img = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_1 = np.ones((5, 5), np.uint8)
    kernel_2 = np.ones((13, 13), np.uint8)
    dilated = cv2.dilate(binary_img, kernel_2, iterations=1)
    eroded = cv2.erode(dilated, kernel_1, iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = np.vstack([cnt.squeeze() for cnt in contours if cnt.shape[0] > 4])

    if all_points.shape[0] < 5:
        return None

    X, Y = all_points[:, 0], all_points[:, 1]

    def calc_R(xc: float, yc: float) -> np.ndarray:
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c: np.ndarray) -> np.ndarray:
        Ri = calc_R(c[0], c[1])
        return Ri - Ri.mean()

    x0 = np.array([np.mean(X), np.mean(Y)]) # type: ignore
    try:
        res = least_squares(cost, x0=x0)
        if not res.success:
            return None
    except Exception:
        return None

    center_2d = (float(res.x[0]), float(res.x[1]))
    radius_2d = float(calc_R(*center_2d).mean())

    valid_depth_mask = depth > 0
    if not np.any(valid_depth_mask):
        return None
    
    valid_depths = depth[valid_depth_mask]
    mean_depth = float(np.mean(valid_depths))
    
    return center_2d, radius_2d, mean_depth

def correct_depth(depth: np.ndarray) -> Tuple[np.ndarray, int]:
    reference_depth = np.max(depth)
    lower_bound = reference_depth - 5
    depth_clipped = clip_depth(depth, lower_bound, reference_depth)
    result = fit_2dcircle(depth_clipped)
    if result is None:
        return depth, 0

    center_2d, radius_2d, _ = result
    cx, cy = int(center_2d[0]), int(center_2d[1])
    r = int(radius_2d)

    h, w = depth.shape
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    correction_mask = circle_mask & (depth > 0)
    
    corrected_count = np.count_nonzero(correction_mask)

    corrected_depth = np.where(correction_mask, 2 * reference_depth - depth, depth)   
    return corrected_depth, corrected_count
