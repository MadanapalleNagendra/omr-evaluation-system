import cv2
import numpy as np
import logging
import os
from .utils import setup_logging, save_debug_image

class SheetDetector:
    def __init__(self, config: dict, debug: bool = False):
        self.config = config
        self.debug = debug
        self.logger = setup_logging()
        self.contour_area_threshold = config['image']['contour_area_threshold']
        self.resize_width = config['image']['resize_width']
        self.resize_height = config['image']['resize_height']

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        try:
            rect = self.order_points(pts)
            (tl, tr, br, bl) = rect
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))
            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
            return warped
        except Exception as e:
            self.logger.error(f"Error in perspective transformation: {e}")
            return image

    def find_largest_contour(self, image: np.ndarray, debug_dir: str = None):
        try:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.logger.warning("No contours found")
                return None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.contour_area_threshold:
                    continue
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    return approx.reshape(4, 2)
            if contours:
                largest_contour = contours[0]
                area = cv2.contourArea(largest_contour)
                if area >= self.contour_area_threshold:
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    return box.astype(np.float32)
            self.logger.warning("No suitable sheet contour found")
            return None
        except Exception as e:
            self.logger.error(f"Error finding largest contour: {e}")
            return None

    def detect_sheet_edges(self, image: np.ndarray, debug_dir: str = None) -> np.ndarray:
        try:
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            return edges
        except Exception as e:
            self.logger.error(f"Error in edge detection: {e}")
            return image

    def detect_and_warp(self, image: np.ndarray, debug_dir: str = None):
        metadata = {"transformation_applied": False, "corner_points": None}
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            edges = self.detect_sheet_edges(gray, debug_dir)
            contour_points = self.find_largest_contour(edges, debug_dir)
            if contour_points is not None:
                metadata["corner_points"] = contour_points.tolist()
                warped = self.four_point_transform(image, contour_points)
                warped_resized = cv2.resize(warped, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
                metadata["transformation_applied"] = True
                return warped_resized, True, metadata
            else:
                resized = cv2.resize(image, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
                return resized, False, metadata
        except Exception as e:
            self.logger.error(f"Error in sheet detection and warping: {e}")
            try:
                resized = cv2.resize(image, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
                return resized, False, metadata
            except:
                return image, False, metadata