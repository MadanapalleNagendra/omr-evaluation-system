# """
# Bubble mark classification module
# Classifies whether bubbles are marked or unmarked using computer vision and ML techniques
# """

# import cv2
# import numpy as np
# import logging
# from typing import Dict, List, Tuple, Optional
# from sklearn.cluster import KMeans
# from .utils import setup_logging, calculate_confidence, save_debug_image


import os
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import setup_logging

# class BubbleClassifier:
#     def save_bubble_overlay(self, omr_image: np.ndarray, bubble_coords: List[Tuple[int, int, int, int]], bubble_results: List[Dict], output_path: str):
#         """
#         Draw overlay of detected bubbles on the OMR sheet.
#         Green: detected as filled, Red: detected as empty, Yellow: ambiguous/needs review
#         Args:
#             omr_image: The OMR sheet image (BGR)
#             bubble_coords: List of (x, y, w, h) for each bubble
#             bubble_results: List of result dicts from classify_single_bubble
#             output_path: Path to save overlay image
#         """
#         overlay = omr_image.copy()
#         for (x, y, w, h), result in zip(bubble_coords, bubble_results):
#             center = (int(x + w/2), int(y + h/2))
#             radius = int(min(w, h) / 2) - 2
#             if result.get('is_marked'):
#                 color = (0, 255, 0)  # Green
#             elif result.get('needs_review'):
#                 color = (0, 255, 255)  # Yellow
#             else:
#                 color = (0, 0, 255)  # Red
#             cv2.circle(overlay, center, radius, color, 2)
#             # Optionally, put confidence value
#             conf = result.get('confidence', 0)
#             cv2.putText(overlay, f"{conf:.2f}", (x, y+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#         cv2.imwrite(output_path, overlay)

#     """
#     Classifies bubble marks using multiple techniques
#     """
    
    
class BubbleClassifier:
    def save_bubble_overlay(self, omr_image: np.ndarray, bubble_coords: List[Tuple[int, int, int, int]], bubble_results: List[Dict], output_path: str):
        """
        Draw overlay of detected bubbles on the OMR sheet.
        Green: detected as filled, Red: detected as empty, Yellow: ambiguous/needs review
        Args:
            omr_image: The OMR sheet image (BGR)
            bubble_coords: List of (x, y, w, h) for each bubble
            bubble_results: List of result dicts from classify_single_bubble
            output_path: Path to save overlay image
        """
        overlay = omr_image.copy()
        for (x, y, w, h), result in zip(bubble_coords, bubble_results):
            center = (int(x + w/2), int(y + h/2))
            radius = int(min(w, h) / 2) - 2
            if result.get('is_marked'):
                color = (0, 255, 0)  # Green
            elif result.get('needs_review'):
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            cv2.circle(overlay, center, radius, color, 2)
            # Optionally, put confidence value
            conf = result.get('confidence', 0)
            cv2.putText(overlay, f"{conf:.2f}", (x, y+h+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(output_path, overlay)
    """
    Classifies bubble marks using multiple techniques
    """
    def __init__(self, config: dict, debug: bool = False):
        """
        Initialize bubble classifier with configuration
        Args:
            config: Configuration dictionary
            debug: Enable debug mode for saving intermediate images
        """
        self.config = config
        self.debug = debug
        self.logger = setup_logging()
        # Extract classification parameters
        self.min_confidence = config['quality']['min_confidence']
        self.review_threshold = config['quality']['review_threshold']
        self.multiple_answers_allowed = config['scoring'].get('multiple_answers_allowed', True)
        
        # Thresholds for classification (lowered for robust detection)
        self.fill_threshold = 0.15  # Lowered further for better detection
        self.intensity_threshold = 0.20  # Lowered for better detection
        self.contour_area_threshold = 5  # Lowered minimum contour area
    
    def calculate_fill_ratio(self, bubble_region: np.ndarray) -> float:
        """
        Calculate fill ratio for a bubble region
        """
        try:
            # Extract image from dict if it's a dictionary
            if isinstance(bubble_region, dict) and 'image' in bubble_region:
                img = bubble_region['image']
            else:
                img = bubble_region
            
            # Validate image type and content
            if not isinstance(img, np.ndarray) or img.size == 0:
                self.logger.warning(f"Invalid bubble image: {type(img)}, size: {getattr(img, 'size', 'N/A')}")
                return 0.0
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray_bubble = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_bubble = img.copy()
            
            # Calculate fill ratio using adaptive threshold
            _, binary_bubble = cv2.threshold(gray_bubble, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_pixels = np.sum(binary_bubble == 0)
            total_pixels = gray_bubble.shape[0] * gray_bubble.shape[1]
            
            fill_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0.0
            return min(1.0, max(0.0, fill_ratio))
            
        except Exception as e:
            self.logger.error(f"Error calculating fill ratio: {e}, type: {type(bubble_region)}")
            return 0.0
    
    def analyze_intensity_distribution(self, bubble_region: np.ndarray) -> Dict[str, float]:
        """
        Analyze intensity distribution of bubble region
        """
        try:
            # Extract image from dict if it's a dictionary
            if isinstance(bubble_region, dict) and 'image' in bubble_region:
                img = bubble_region['image']
            else:
                img = bubble_region
            
            # Validate image type and content
            if not isinstance(img, np.ndarray) or img.size == 0:
                return {"mean": 255.0, "std": 0.0}
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray_bubble = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_bubble = img.copy()
            
            # Calculate statistics
            stats = {
                "mean": float(np.mean(gray_bubble)),
                "std": float(np.std(gray_bubble)),
                "min": float(np.min(gray_bubble)),
                "max": float(np.max(gray_bubble))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing intensity: {e}")
            return {"mean": 255.0, "std": 0.0}
    
    def detect_contours_in_bubble(self, bubble_region: np.ndarray) -> int:
        """
        Detect significant contours in bubble region
        """
        try:
            # Extract image from dict if it's a dictionary
            if isinstance(bubble_region, dict) and 'image' in bubble_region:
                img = bubble_region['image']
            else:
                img = bubble_region
            
            # Validate image type and content
            if not isinstance(img, np.ndarray) or img.size == 0:
                return 0
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray_bubble = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_bubble = img.copy()
            
            # Apply threshold and find contours
            _, binary_bubble = cv2.threshold(gray_bubble, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary_bubble, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            significant_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.contour_area_threshold:
                    significant_contours += 1
            return significant_contours
            
        except Exception as e:
            self.logger.error(f"Error detecting contours in bubble: {e}")
            return 0
    
    def classify_single_bubble(self, bubble_region: np.ndarray, question_num: int = 0, option_num: int = 0) -> Dict:
        """
        Classify a single bubble as marked or unmarked
        """
        try:
            # Extract image from dict if present, with robust type checking
            if isinstance(bubble_region, dict) and 'image' in bubble_region:
                img = bubble_region['image']
                self.logger.debug(f"Processing dict bubble - type: {type(img)}")
            elif isinstance(bubble_region, np.ndarray):
                img = bubble_region
                self.logger.debug(f"Processing ndarray bubble - shape: {img.shape}")
            else:
                error_msg = f"Bubble region is not a valid image or dict: {type(bubble_region)}"
                self.logger.error(error_msg)
                return {
                    "is_marked": False,
                    "confidence": 0.0,
                    "fill_ratio": 0.0,
                    "method": "invalid_type",
                    "needs_review": True,
                    "error": error_msg
                }
            
            # Final validation of the image
            if not isinstance(img, np.ndarray):
                error_msg = f"Extracted image is not a numpy array: {type(img)}"
                self.logger.error(error_msg)
                return {
                    "is_marked": False,
                    "confidence": 0.0,
                    "fill_ratio": 0.0,
                    "method": "invalid_image",
                    "needs_review": True,
                    "error": error_msg
                }
            
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                return {
                    "is_marked": False,
                    "confidence": 0.0,
                    "fill_ratio": 0.0,
                    "method": "empty_region",
                    "needs_review": True
                }
            
            # Method 1: Fill ratio analysis
            fill_ratio = self.calculate_fill_ratio(bubble_region)
            
            # Method 2: Intensity analysis
            intensity_stats = self.analyze_intensity_distribution(bubble_region)
            
            # Method 3: Contour analysis
            contour_count = self.detect_contours_in_bubble(bubble_region)
            
            # Debug: Save bubble region for inspection
            if self.debug:
                debug_dir = self.config.get('debug_dir', 'logs/bubble_debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                # Ensure we have a valid image to save
                debug_img = img.copy() if isinstance(img, np.ndarray) else img
                if len(debug_img.shape) == 3:
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
                
                cv2.imwrite(
                    os.path.join(debug_dir, f"bubble_Q{question_num:02d}_{chr(65+option_num)}.png"), 
                    debug_img
                )
            
            # Decision logic combining multiple methods
            is_marked = False
            confidence = 0.0
            method_used = "combined"
            
            # Primary method: Fill ratio
            if fill_ratio >= self.fill_threshold:
                is_marked = True
                confidence = min(1.0, (fill_ratio - self.fill_threshold) / (1.0 - self.fill_threshold))
                method_used = "fill_ratio"
                self.logger.debug(f"Q{question_num} Opt{chr(65+option_num)}: Marked by fill_ratio={fill_ratio:.3f}")
            
            # Secondary method: Intensity (if fill ratio didn't trigger)
            if not is_marked:
                mean_intensity = intensity_stats["mean"]
                normalized_intensity = 1.0 - (mean_intensity / 255.0)
                if normalized_intensity >= self.intensity_threshold:
                    is_marked = True
                    confidence = normalized_intensity
                    method_used = "intensity"
                    self.logger.debug(f"Q{question_num} Opt{chr(65+option_num)}: Marked by intensity={normalized_intensity:.3f}")
            
            # Tertiary method: Multiple contours
            if contour_count > 2:  # Multiple significant contours
                is_marked = True
                confidence = max(confidence, min(0.9, contour_count * 0.3))
                method_used = "contour" if confidence == 0 else method_used + "+contour"
                self.logger.debug(f"Q{question_num} Opt{chr(65+option_num)}: Marked by contours={contour_count}")
            
            # Determine if review is needed
            needs_review = confidence < self.review_threshold or fill_ratio < 0.05 or fill_ratio > 0.95
            
            result = {
                "is_marked": is_marked,
                "confidence": float(confidence),
                "fill_ratio": float(fill_ratio),
                "mean_intensity": float(intensity_stats["mean"]),
                "contour_count": contour_count,
                "method": method_used,
                "needs_review": needs_review,
                "debug_info": {
                    "fill_ratio": fill_ratio,
                    "intensity": intensity_stats["mean"],
                    "contours": contour_count
                }
            }
            
            self.logger.debug(f"Classification result for Q{question_num} Opt{chr(65+option_num)}: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Error classifying bubble Q{question_num} Opt{chr(65+option_num)}: {e}"
            self.logger.error(error_msg)
            return {
                "is_marked": False,
                "confidence": 0.0,
                "fill_ratio": 0.0,
                "method": "error",
                "needs_review": True,
                "error": error_msg
            }
    
    def classify_question_bubbles(self, question_bubbles: Dict[int, np.ndarray], question_num: int) -> Dict[int, Dict]:
        """
        Classify all bubbles for a single question
        """
        try:
            results = {}
            bubble_coords = []
            bubble_results = []
            
            for option_num, bubble_region in question_bubbles.items():
                # Log the type of bubble_region for debugging
                self.logger.debug(f"Processing Q{question_num} Opt{option_num}: type={type(bubble_region)}")
                
                if not isinstance(bubble_region, dict) and not isinstance(bubble_region, np.ndarray):
                    self.logger.error(f"Invalid bubble_region type for Q{question_num} Opt{option_num}: {type(bubble_region)}")
                    results[option_num] = {
                        "is_marked": False,
                        "confidence": 0.0,
                        "fill_ratio": 0.0,
                        "method": "invalid_type",
                        "needs_review": True,
                        "error": f"Invalid type: {type(bubble_region)}"
                    }
                    continue
                
                result = self.classify_single_bubble(bubble_region, question_num, option_num)
                results[option_num] = result
                
                # Extract coordinates for overlay
                if isinstance(bubble_region, dict) and 'bbox' in bubble_region:
                    x, y, w, h = bubble_region['bbox']
                    bubble_coords.append((x, y, w, h))
                elif isinstance(bubble_region, dict) and 'image' in bubble_region and 'bbox' in bubble_region:
                    x, y, w, h = bubble_region['bbox']
                    bubble_coords.append((x, y, w, h))
                else:
                    # Default coordinates if bbox not available
                    if 'image' in result:
                        h_img, w_img = result['image'].shape[:2]
                    else:
                        h_img, w_img = (20, 20)
                    bubble_coords.append((option_num * 30 + 20, question_num * 30 + 20, w_img, h_img))
                
                bubble_results.append(result)
            
            # Post-processing: Handle multiple answers
            marked_options = [opt for opt, res in results.items() if res["is_marked"]]
            if len(marked_options) > 1 and not self.multiple_answers_allowed:
                # Keep only the most confident answer
                best_option = max(marked_options, key=lambda opt: results[opt]["confidence"])
                for opt in marked_options:
                    if opt != best_option:
                        results[opt]["is_marked"] = False
                        results[opt]["needs_review"] = True
                        results[opt]["method"] += "_deselected"
                        self.logger.info(f"Deselected multiple answer for Q{question_num} Opt{opt}")
            
            # Save overlay image if debug enabled - FIXED PATH HANDLING
            if self.debug:
                overlay_path = None
                if debug_dir := self.config.get('debug_dir'):
                    # Use the debug directory from config
                    os.makedirs(debug_dir, exist_ok=True)
                    overlay_path = os.path.join(debug_dir, f"overlay_Q{question_num:02d}.png")
                elif 'omr_image' in self.config and os.path.exists(self.config['omr_image']):
                    # Fallback to default location
                    default_dir = os.path.dirname(self.config['omr_image']) if os.path.isfile(self.config['omr_image']) else 'logs/bubble_debug'
                    os.makedirs(default_dir, exist_ok=True)
                    overlay_path = os.path.join(default_dir, f"overlay_Q{question_num:02d}.png")
                
                if overlay_path and 'omr_image' in self.config:
                    try:
                        omr_image_path = self.config['omr_image']
                        if os.path.exists(omr_image_path):
                            omr_img = cv2.imread(omr_image_path)
                            if omr_img is not None:
                                self.save_bubble_overlay(omr_img, bubble_coords, bubble_results, overlay_path)
                                self.logger.debug(f"Saved question overlay: {overlay_path}")
                                # Store the overlay path in results for Streamlit display
                                results['overlay_path'] = overlay_path
                            else:
                                self.logger.warning(f"Could not load OMR image from {omr_image_path}")
                        else:
                            self.logger.warning(f"OMR image path does not exist: {omr_image_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not save overlay for Q{question_num}: {e}")
            
            # Create full sheet overlay if we have enough data
            if self.debug and len(bubble_coords) > 10 and 'omr_image' in self.config:
                full_overlay_path = None
                if debug_dir := self.config.get('debug_dir'):
                    full_overlay_path = os.path.join(debug_dir, f"overlay_full_Q{question_num:02d}.png")
                elif 'omr_image' in self.config:
                    default_dir = os.path.dirname(self.config['omr_image']) if os.path.isfile(self.config['omr_image']) else 'logs/bubble_debug'
                    os.makedirs(default_dir, exist_ok=True)
                    full_overlay_path = os.path.join(default_dir, f"overlay_full_Q{question_num:02d}.png")
                
                if full_overlay_path:
                    try:
                        omr_img = cv2.imread(self.config['omr_image'])
                        if omr_img is not None:
                            self.save_bubble_overlay(omr_img, bubble_coords, bubble_results, full_overlay_path)
                            self.logger.debug(f"Saved full overlay for Q{question_num}: {full_overlay_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not save full overlay for Q{question_num}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error classifying question {question_num}: {e}")
            return {}
    
    def classify_all_bubbles(self, bubble_regions: Dict[int, Dict[int, np.ndarray]], 
                       debug_dir: str = None) -> Dict[int, Dict[int, Dict]]:
        """
        Classify all bubbles across all questions with enhanced overlay generation
        """
        try:
            if not bubble_regions:
                self.logger.warning("No bubble regions provided for classification")
                return {}
            
            # Set debug directory in config if provided
            if debug_dir:
                self.config['debug_dir'] = debug_dir
                os.makedirs(debug_dir, exist_ok=True)
            
            all_results = {}
            review_needed = []
            successful_classifications = 0
            total_bubbles = 0
            
            # Collect all bubble data for full sheet overlay
            all_bubble_coords = []
            all_bubble_results = []
            
            for question_num, question_bubbles in bubble_regions.items():
                total_bubbles += len(question_bubbles)
                
                # Validate that we have actual bubble data
                valid_bubbles = {}
                for option_num, bubble_data in question_bubbles.items():
                    if isinstance(bubble_data, dict) and 'image' in bubble_data:
                        valid_bubbles[option_num] = bubble_data
                    elif isinstance(bubble_data, np.ndarray):
                        # Wrap numpy array in dict for consistency
                        valid_bubbles[option_num] = {
                            'image': bubble_data, 
                            'bbox': (0, 0, bubble_data.shape[1], bubble_data.shape[0])
                        }
                    else:
                        self.logger.warning(f"Skipping invalid bubble Q{question_num} Opt{option_num}: {type(bubble_data)}")
                
                if valid_bubbles:
                    question_results = self.classify_question_bubbles(valid_bubbles, question_num)
                    if isinstance(question_results, dict):
                        all_results[question_num] = question_results
                        successful_classifications += len(question_results)
                        # Collect data for full overlay
                        for opt_num, result in question_results.items():
                            if opt_num != 'overlay_path':  # Skip metadata entries
                                bubble_data = valid_bubbles.get(opt_num, {})
                                if isinstance(bubble_data, dict) and 'bbox' in bubble_data:
                                    x, y, w, h = bubble_data['bbox']
                                    all_bubble_coords.append((x, y, w, h))
                                    all_bubble_results.append(result)
                        # Track questions needing review
                        if any(result.get("needs_review", False) for result in question_results.values() if isinstance(result, dict)):
                            review_needed.append(question_num)
                    else:
                        self.logger.warning(f"Question {question_num} classification returned non-dict: {type(question_results)}")
                else:
                    self.logger.warning(f"No valid bubbles found for question {question_num}")
            
            # Generate summary statistics
            marked_bubbles = 0
            for q_results in all_results.values():
                if isinstance(q_results, dict):
                    for result in q_results.values():
                        if isinstance(result, dict) and result.get("is_marked", False):
                            marked_bubbles += 1
            
            avg_confidence = 0.0
            if total_bubbles > 0:
                total_confidence = 0.0
                for q_results in all_results.values():
                    if isinstance(q_results, dict):
                        for result in q_results.values():
                            if isinstance(result, dict):
                                total_confidence += result.get("confidence", 0)
                avg_confidence = total_confidence / total_bubbles
            
            self.logger.info(f"Classification completed: {successful_classifications}/{total_bubbles} processed, "
                        f"{marked_bubbles} marked, avg confidence: {avg_confidence:.3f}, "
                        f"{len(review_needed)} questions need review")
            
            # Generate FULL SHEET OVERLAY - This is the key fix
            if self.debug and all_bubble_coords and 'omr_image' in self.config:
                full_sheet_overlay_path = self._generate_full_sheet_overlay(
                    all_bubble_coords, all_bubble_results, debug_dir
                )
                # Store the full sheet overlay path for Streamlit
                all_results['full_sheet_overlay'] = full_sheet_overlay_path
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in bubble classification: {e}")
            return {}
        
    def _generate_full_sheet_overlay(self, all_coords: List[Tuple], all_results: List[Dict], 
                                debug_dir: str = None) -> str:
        """
        Generate overlay for the complete OMR sheet
        Returns the path to the saved overlay image
        """
        overlay_path = None
        
        try:
            if 'omr_image' not in self.config:
                self.logger.warning("No OMR image path in config for full sheet overlay")
                return overlay_path
            
            omr_image_path = self.config['omr_image']
            if not os.path.exists(omr_image_path):
                self.logger.warning(f"OMR image not found: {omr_image_path}")
                return overlay_path
            
            omr_img = cv2.imread(omr_image_path)
            if omr_img is None:
                self.logger.warning(f"Could not load OMR image: {omr_image_path}")
                return overlay_path
            
            # Determine output path
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                overlay_path = os.path.join(debug_dir, "overlay_full_sheet.png")
            else:
                # Fallback to a standard location
                fallback_dir = os.path.join(os.getcwd(), "logs", "overlays")
                os.makedirs(fallback_dir, exist_ok=True)
                overlay_path = os.path.join(fallback_dir, f"overlay_full_sheet_{int(time.time())}.png")
            
            # Generate and save overlay
            self.save_bubble_overlay(omr_img, all_coords, all_results, overlay_path)
            self.logger.info(f"Full sheet overlay saved: {overlay_path}")
            
            return overlay_path
            
        except Exception as e:
            self.logger.error(f"Error generating full sheet overlay: {e}")
            return overlay_path
        
    def get_marked_answers(self, classifications: Dict[int, Dict[int, Dict]]) -> Dict[int, List[int]]:
        """
        Extract marked answers from classifications
        """
        try:
            marked_answers = {}
            
            for question_num, question_results in classifications.items():
                marked_options = []
                if not isinstance(question_results, dict):
                    continue
                for option_num, result in question_results.items():
                    if isinstance(result, dict) and result.get("is_marked", False) and not result.get("error"):
                        marked_options.append(option_num)
                marked_answers[question_num] = sorted(marked_options)
            
            return marked_answers
            
        except Exception as e:
            self.logger.error(f"Error extracting marked answers: {e}")
            return {}
    
    def batch_classify(self, batch_bubble_regions: List[Dict[int, Dict[int, np.ndarray]]], 
                      debug_base_dir: str = None) -> List[Dict]:
        """
        Batch classification for multiple sheets
        """
        results = []
        
        for i, bubble_regions in enumerate(batch_bubble_regions):
            try:
                debug_dir = None
                if debug_base_dir and self.debug:
                    debug_dir = os.path.join(debug_base_dir, f"classify_{i:03d}")
                    os.makedirs(debug_dir, exist_ok=True)
                
                classifications = self.classify_all_bubbles(bubble_regions, debug_dir)
                marked_answers = self.get_marked_answers(classifications)
                
                batch_result = {
                    "classifications": classifications,
                    "marked_answers": marked_answers,
                    "successful_bubbles": sum(len(q_results) for q_results in classifications.values()),
                    "total_bubbles": sum(len(q_bubbles) for q_bubbles in bubble_regions.values())
                }
                
                results.append(batch_result)
                
            except Exception as e:
                self.logger.error(f"Error in batch classification for sheet {i}: {e}")
                results.append({
                    "classifications": {},
                    "marked_answers": {},
                    "error": str(e)
                })
        
        return results









