
import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from .utils import setup_logging, save_debug_image
import os

class ImagePreprocessor:

    def correct_skew(self, image: np.ndarray, debug_dir: str = None) -> Tuple[np.ndarray, float]:
        """
        Dummy skew correction: returns the image unchanged and skew angle 0.0.
        Replace with real skew correction if needed.
        """
        return image, 0.0

    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize the image to the configured width and height, maintaining aspect ratio if specified.
        """
        try:
            if maintain_aspect:
                h, w = image.shape[:2]
                aspect_ratio = w / h
                target_aspect = self.resize_width / self.resize_height
                if aspect_ratio > target_aspect:
                    new_w = self.resize_width
                    new_h = int(self.resize_width / aspect_ratio)
                else:
                    new_h = self.resize_height
                    new_w = int(self.resize_height * aspect_ratio)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Pad to target size if necessary
                if new_w < self.resize_width or new_h < self.resize_height:
                    if len(image.shape) == 3:
                        padded = np.zeros((self.resize_height, self.resize_width, image.shape[2]), dtype=image.dtype)
                    else:
                        padded = np.zeros((self.resize_height, self.resize_width), dtype=image.dtype)
                    y_offset = (self.resize_height - new_h) // 2
                    x_offset = (self.resize_width - new_w) // 2
                    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    return padded
                else:
                    return resized
            else:
                return cv2.resize(image, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.logger.error(f"Error in resize_image: {e}")
            return image
    """
    Handles preprocessing of OMR sheet images
    """
    
    def __init__(self, config: dict, debug: bool = False):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
            debug: Enable debug mode for saving intermediate images
        """
        self.config = config
        self.debug = debug
        self.logger = setup_logging()
        
        # Extract preprocessing parameters
        self.resize_width = config['image']['resize_width']
        self.resize_height = config['image']['resize_height']
        self.blur_kernel = config['image']['gaussian_blur_kernel']
        self.threshold_block_size = config['image']['threshold_block_size']
        self.threshold_c = config['image']['threshold_c']
    
    def enhance_image(self, image: np.ndarray, debug_dir: str = None) -> np.ndarray:
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if self.debug and debug_dir:
                save_debug_image(gray, "01_grayscale.jpg", debug_dir)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            if self.debug and debug_dir:
                save_debug_image(enhanced, "02_clahe_enhanced.jpg", debug_dir)
            
            # Gaussian blur to reduce noise
            if self.blur_kernel > 0:
                enhanced = cv2.GaussianBlur(enhanced, (self.blur_kernel, self.blur_kernel), 0)
                
                if self.debug and debug_dir:
                    save_debug_image(enhanced, "03_gaussian_blur.jpg", debug_dir)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in image enhancement: {e}")
            return image
    
    def adaptive_threshold(self, image: np.ndarray, debug_dir: str = None) -> np.ndarray:
        try:
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.threshold_block_size,
                self.threshold_c
            )
            
            if self.debug and debug_dir:
                save_debug_image(binary, "04_adaptive_threshold.jpg", debug_dir)
            
            return binary
            
        except Exception as e:
            self.logger.error(f"Error in adaptive thresholding: {e}")
            return image
    
    def remove_noise(self, binary_image: np.ndarray, debug_dir: str = None) -> np.ndarray:
        try:
            # Morphological operations to remove noise
            kernel = np.ones((3, 3), np.uint8)
            denoised = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)
            if self.debug and debug_dir:
                save_debug_image(denoised, "05_denoised.jpg", debug_dir)
            return denoised
        except Exception as e:
            self.logger.error(f"Error in noise removal: {e}")
            return binary_image
    
    def preprocess_pipeline(self, image_path: str, debug_dir: str = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        metadata = {"skew_angle": 0.0, "original_size": None, "final_size": None}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            metadata["original_size"] = image.shape[:2]
            
            # Resize image first
            resized = self.resize_image(image)
            metadata["final_size"] = resized.shape[:2]
            
            if self.debug and debug_dir:
                save_debug_image(resized, "00_original_resized.jpg", debug_dir)
            
            # Correct skew
            skew_corrected, skew_angle = self.correct_skew(resized, debug_dir)
            metadata["skew_angle"] = skew_angle
            
            # Enhance image
            enhanced = self.enhance_image(skew_corrected, debug_dir)
            
            # Apply adaptive threshold
            binary = self.adaptive_threshold(enhanced, debug_dir)
            
            # Remove noise
            final_binary = self.remove_noise(binary, debug_dir)
            
            self.logger.info(f"Successfully preprocessed: {image_path}")
            
            return skew_corrected, final_binary, metadata
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline for {image_path}: {e}")
            raise
    
    def batch_preprocess(self, image_paths: list, output_dir: str = None) -> dict:
        results = {
            "successful": [],
            "failed": [],
            "metadata": {}
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                debug_subdir = None
                if output_dir and self.debug:
                    debug_subdir = os.path.join(output_dir, f"debug_{i:03d}")
                    os.makedirs(debug_subdir, exist_ok=True)
                
                original, binary, metadata = self.preprocess_pipeline(image_path, debug_subdir)
                
                results["successful"].append({
                    "path": image_path,
                    "original": original,
                    "binary": binary,
                    "metadata": metadata
                })
                
            except Exception as e:
                self.logger.error(f"Failed to preprocess {image_path}: {e}")
                results["failed"].append({
                    "path": image_path,
                    "error": str(e)
                })
        
        self.logger.info(f"Batch preprocessing completed: {len(results['successful'])} successful, {len(results['failed'])} failed")
        
        return results

def main():
    import os
    from .utils import load_config, get_image_files
    
    # Load configuration
    config = load_config()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(config, debug=True)
    
    # Get test images
    test_dir = "data/test_images"  # Update path as needed
    if os.path.exists(test_dir):
        image_files = get_image_files(test_dir)
        
        if image_files:
            print(f"Found {len(image_files)} images for testing")
            
            # Test single image preprocessing
            test_image = image_files[0]
            debug_dir = "output/debug/preprocess_test"
            os.makedirs(debug_dir, exist_ok=True)
            
            try:
                original, binary, metadata = preprocessor.preprocess_pipeline(test_image, debug_dir)
                print(f"Successfully preprocessed test image: {test_image}")
                print(f"Metadata: {metadata}")
                
            except Exception as e:
                print(f"Error processing test image: {e}")
        else:
            print("No test images found")
    else:
        print(f"Test directory not found: {test_dir}")

if __name__ == "__main__":
    main()