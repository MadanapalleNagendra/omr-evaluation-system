# # """
# # Grid extraction module for OMR sheets
# # Identifies and extracts bubble grid from processed OMR sheets
# # """

# # import cv2
# # import numpy as np
# # import logging
# # import os
# # from typing import Tuple, List, Dict, Optional
# # from .utils import setup_logging, save_debug_image

# # class GridExtractor:
# #     """
# #     Extracts bubble grid from OMR sheet images
# #     """
    
# #     def __init__(self, config: dict, debug: bool = False):
# #         """
# #         Initialize grid extractor with configuration
        
# #         Args:
# #             config: Configuration dictionary
# #             debug: Enable debug mode for saving intermediate images
# #         """
# #         self.config = config
# #         self.debug = debug
# #         self.logger = setup_logging()
        
# #         # Extract grid parameters
# #         self.total_questions = config['grid']['rows']
# #         self.options_per_question = config['grid']['cols']
# #         self.bubble_min_area = config['grid']['bubble_min_area']
# #         self.bubble_max_area = config['grid']['bubble_max_area']
# #         self.aspect_ratio_tolerance = config['grid']['bubble_aspect_ratio_tolerance']
# #         self.grid_tolerance = config['grid']['grid_tolerance']
    
# #     def find_bubble_contours(self, binary_image: np.ndarray, debug_dir: str = None) -> List[Dict]:
# #         """
# #         Find all potential bubble contours in the image
        
# #         Args:
# #             binary_image: Input binary image
# #             debug_dir: Directory for saving debug images
            
# #         Returns:
# #             List of bubble information dictionaries
# #         """
# #         try:
# #             # Find contours
# #             contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
import cv2
import numpy as np
from .utils import setup_logging, save_debug_image

class GridExtractor:
    def __init__(self, config: dict, debug: bool = False):
        self.config = config
        self.debug = debug
        self.logger = setup_logging()
        self.grid_x = config['grid']['x']
        self.grid_y = config['grid']['y']
        self.grid_width = config['grid']['width']
        self.grid_height = config['grid']['height']
        self.cell_rows = config['grid']['cell_rows']
        self.cell_cols = config['grid']['cell_cols']

    def extract_grid(self, image: np.ndarray, debug_dir: str = None):
        try:
            x, y, w, h = self.grid_x, self.grid_y, self.grid_width, self.grid_height
            grid_image = image[y:y+h, x:x+w]
            metadata = {'grid_bbox': (x, y, w, h)}
            # Save grid region debug image always
            if self.debug and debug_dir:
                save_debug_image(grid_image, "grid_extracted.jpg", debug_dir)
            # Add pixel stats for diagnostics
            metadata['pixel_mean'] = float(np.mean(grid_image))
            metadata['pixel_std'] = float(np.std(grid_image))
            metadata['pixel_min'] = int(np.min(grid_image))
            metadata['pixel_max'] = int(np.max(grid_image))
            # Check if grid region is empty or uniform
            if np.count_nonzero(grid_image) == 0 or np.count_nonzero(grid_image) == grid_image.size:
                metadata['error'] = 'Grid region appears empty or uniform. Check image quality and grid coordinates.'
            return grid_image, metadata
        except Exception as e:
            self.logger.error(f"Error extracting grid: {e}")
            if self.debug and debug_dir:
                save_debug_image(image, "grid_extraction_failed.jpg", debug_dir)
            return image, {'error': f'Grid extraction failed: {e}'}

    def extract_cells(self, grid_image: np.ndarray, debug_dir: str = None):
        try:
            cell_height = grid_image.shape[0] // self.cell_rows
            cell_width = grid_image.shape[1] // self.cell_cols
            cells = []
            for row in range(self.cell_rows):
                for col in range(self.cell_cols):
                    y1 = row * cell_height
                    y2 = (row + 1) * cell_height
                    x1 = col * cell_width
                    x2 = (col + 1) * cell_width
                    cell = grid_image[y1:y2, x1:x2]
                    cells.append(cell)
            # Always save cell debug images
            if self.debug and debug_dir:
                for idx, cell in enumerate(cells):
                    save_debug_image(cell, f"cell_{idx:03d}.jpg", debug_dir)
            # If no cells detected, add error info and save grid region
            if len(cells) == 0:
                self.logger.error("No cells detected in grid extraction.")
                if self.debug and debug_dir:
                    save_debug_image(grid_image, "no_cells_detected.jpg", debug_dir)
            return cells
        except Exception as e:
            self.logger.error(f"Error extracting cells: {e}")
            if self.debug and debug_dir:
                save_debug_image(grid_image, "cell_extraction_failed.jpg", debug_dir)
            return []
# #             grid = {}
            
# #             for row_idx, row in enumerate(rows):
# #                 question_number = row_idx + 1
                
# #                 if question_number > self.total_questions:
# #                     break
                
# #                 grid[question_number] = {}
                
# #                 for col_idx, bubble in enumerate(row):
# #                     if col_idx < self.options_per_question:
# #                         option_number = col_idx  # 0=A, 1=B, 2=C, 3=D
# #                         grid[question_number][option_number] = bubble
            
# #             # Debug: Visualize grid organization
# #             if self.debug and debug_dir and grid:
# #                 self._visualize_grid(bubbles_sorted[0] if bubbles_sorted else None, grid, debug_dir)
            
# #             self.logger.info(f"Organized bubbles into grid: {len(grid)} questions")
# #             return grid
            
# #         except Exception as e:
# #             self.logger.error(f"Error organizing bubbles into grid: {e}")
# #             return {}
    
# #     def _visualize_grid(self, sample_bubble: Dict, grid: Dict[int, Dict[int, Dict]], debug_dir: str):
# #         """
# #         Create visualization of the organized grid
        
# #         Args:
# #             sample_bubble: Sample bubble for image dimensions
# #             grid: Organized grid structure
# #             debug_dir: Directory for saving debug images
# #         """
# #         try:
# #             # Create blank image for visualization
# #             img_height = 1200
# #             img_width = 800
# #             debug_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
# #             # Colors for different options
# #             colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # A, B, C, D
            
# #             for question_num, question_data in grid.items():
# #                 for option_num, bubble_info in question_data.items():
# #                     center = bubble_info['center']
# #                     color = colors[option_num % len(colors)]
                    
# #                     # Draw bubble
# #                     cv2.circle(debug_image, center, 8, color, 2)
                    
# #                     # Add question and option labels
# #                     label = f"Q{question_num}{chr(65+option_num)}"  # Q1A, Q1B, etc.
# #                     cv2.putText(debug_image, label, (center[0]-15, center[1]-15), 
# #                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
# #             save_debug_image(debug_image, "15_grid_organization.jpg", debug_dir)
            
# #         except Exception as e:
# #             self.logger.error(f"Error in grid visualization: {e}")
    
# #     def extract_bubble_regions(self, image: np.ndarray, grid: Dict[int, Dict[int, Dict]], 
# #                               debug_dir: str = None) -> Dict[int, Dict[int, np.ndarray]]:
# #         """
# #         Extract individual bubble regions from the image
        
# #         Args:
# #             image: Input image (grayscale or binary)
# #             grid: Organized grid structure
# #             debug_dir: Directory for saving debug images
            
# #         Returns:
# #             Dictionary of bubble regions: {question_number: {option_number: bubble_image}}
# #         """
# #         try:
# #             bubble_regions = {}
            
# #             # Determine padding for bubble extraction
# #             padding = 5  # pixels to add around each bubble
            
# #             for question_num, question_data in grid.items():
# #                 bubble_regions[question_num] = {}
# #                 for option_num, bubble_info in question_data.items():
# #                     try:
# #                         # Get bubble bounding box
# #                         x, y, w, h = bubble_info['bbox']
# #                         # Add padding
# #                         x_start = max(0, x - padding)
# #                         y_start = max(0, y - padding)
# #                         x_end = min(image.shape[1], x + w + padding)
# #                         y_end = min(image.shape[0], y + h + padding)
# #                         # Extract bubble region
# #                         bubble_img = image[y_start:y_end, x_start:x_end]
# #                         if bubble_img.size > 0:
# #                             bubble_regions[question_num][option_num] = {
# #                                 'image': bubble_img.copy(),
# #                                 'bbox': (x, y, w, h)
# #                             }
# #                     except Exception as e:
# #                         self.logger.warning(f"Error extracting bubble Q{question_num}{chr(65+option_num)}: {e}")
            
# #             # Debug: Save sample bubble regions
# #             if self.debug and debug_dir and bubble_regions:
# #                 self._save_sample_bubbles(bubble_regions, debug_dir)
            
# #             extracted_count = sum(len(q_data) for q_data in bubble_regions.values())
# #             self.logger.info(f"Extracted {extracted_count} bubble regions")
            
# #             return bubble_regions
            
# #         except Exception as e:
# #             self.logger.error(f"Error extracting bubble regions: {e}")
# #             return {}
    
# #     def _save_sample_bubbles(self, bubble_regions: Dict[int, Dict[int, dict]], debug_dir: str):
# #         """
# #         Save sample bubble regions for debugging
        
# #         Args:
# #             bubble_regions: Dictionary of bubble regions
# #             debug_dir: Directory for saving debug images
# #         """
# #         try:
# #             sample_dir = os.path.join(debug_dir, "sample_bubbles")
# #             os.makedirs(sample_dir, exist_ok=True)
            
# #             # Save first few questions as samples
# #             sample_questions = list(bubble_regions.keys())[:5]
            
# #             for question_num in sample_questions:
# #                 if question_num in bubble_regions:
# #                     question_data = bubble_regions[question_num]
# #                     for option_num, bubble_dict in question_data.items():
# #                         option_letter = chr(65 + option_num)  # A, B, C, D
# #                         filename = f"Q{question_num:02d}_{option_letter}.jpg"
# #                         filepath = os.path.join(sample_dir, filename)
# #                         # Extract image from dict
# #                         bubble_image = bubble_dict['image'] if isinstance(bubble_dict, dict) and 'image' in bubble_dict else bubble_dict
# #                         if isinstance(bubble_image, np.ndarray) and bubble_image.shape[0] > 0 and bubble_image.shape[1] > 0:
# #                             resized_bubble = cv2.resize(bubble_image, (50, 50), interpolation=cv2.INTER_NEAREST)
# #                             cv2.imwrite(filepath, resized_bubble)
            
# #         except Exception as e:
# #             self.logger.error(f"Error saving sample bubbles: {e}")
    
# #     def validate_grid_structure(self, grid: Dict[int, Dict[int, Dict]]) -> Tuple[bool, List[str]]:
# #         """
# #         Validate the extracted grid structure
        
# #         Args:
# #             grid: Organized grid structure
            
# #         Returns:
# #             Tuple of (is_valid, list_of_issues)
# #         """
# #         issues = []
        
# #         try:
# #             # Check total number of questions
# #             if len(grid) < self.total_questions * 0.8:  # Allow 20% missing questions
# #                 issues.append(f"Too few questions detected: {len(grid)}/{self.total_questions}")
            
# #             # Check each question has expected number of options
# #             incomplete_questions = []
# #             for question_num in range(1, self.total_questions + 1):
# #                 if question_num in grid:
# #                     options_count = len(grid[question_num])
# #                     if options_count != self.options_per_question:
# #                         incomplete_questions.append(f"Q{question_num}: {options_count} options")
# #                 else:
# #                     incomplete_questions.append(f"Q{question_num}: missing")
            
# #             if incomplete_questions:
# #                 # Only report if more than 10% of questions are problematic
# #                 if len(incomplete_questions) > self.total_questions * 0.1:
# #                     issues.append(f"Incomplete questions: {', '.join(incomplete_questions[:10])}")
# #                     if len(incomplete_questions) > 10:
# #                         issues.append(f"... and {len(incomplete_questions) - 10} more")
            
# #             # Check grid alignment and spacing
# #             if len(grid) >= 2:
# #                 question_numbers = sorted(grid.keys())
# #                 y_positions = []
                
# #                 for q_num in question_numbers:
# #                     if grid[q_num]:
# #                         # Get average y-position for this question
# #                         y_coords = [bubble['center'][1] for bubble in grid[q_num].values()]
# #                         avg_y = sum(y_coords) / len(y_coords)
# #                         y_positions.append(avg_y)
                
# #                 if len(y_positions) >= 2:
# #                     # Check if questions are reasonably spaced
# #                     y_diffs = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
# #                     avg_spacing = sum(y_diffs) / len(y_diffs)
                    
# #                     # Check for inconsistent spacing
# #                     inconsistent_spacing = [diff for diff in y_diffs if abs(diff - avg_spacing) > avg_spacing * 0.5]
# #                     if len(inconsistent_spacing) > len(y_diffs) * 0.2:  # More than 20% inconsistent
# #                         issues.append("Inconsistent question spacing detected")
            
# #             is_valid = len(issues) == 0
            
# #             if is_valid:
# #                 self.logger.info("Grid structure validation passed")
# #             else:
# #                 self.logger.warning(f"Grid structure validation found issues: {issues}")
            
# #             return is_valid, issues
            
# #         except Exception as e:
# #             self.logger.error(f"Error in grid validation: {e}")
# #             return False, [f"Validation error: {str(e)}"]
    
# #     def extract_grid_pipeline(self, processed_image: np.ndarray, debug_dir: str = None) -> Tuple[Dict, Dict, List[str]]:
# #         """
# #         Complete pipeline for grid extraction
        
# #         Args:
# #             processed_image: Preprocessed binary image
# #             debug_dir: Directory for saving debug images
            
# #         Returns:
# #             Tuple of (grid_structure, bubble_regions, validation_issues)
# #         """
# #         try:
# #             self.logger.info("Starting grid extraction pipeline")
            
# #             # Find bubble contours
# #             bubble_candidates = self.find_bubble_contours(processed_image, debug_dir)
            
# #             if not bubble_candidates:
# #                 return {}, {}, ["No bubble candidates found"]
            
# #             # Organize bubbles into grid
# #             grid_structure = self.organize_bubbles_into_grid(bubble_candidates, debug_dir)
            
# #             if not grid_structure:
# #                 return {}, {}, ["Failed to organize bubbles into grid"]
            
# #             # Extract bubble regions
# #             bubble_regions = self.extract_bubble_regions(processed_image, grid_structure, debug_dir)
            
# #             # Validate grid structure
# #             is_valid, validation_issues = self.validate_grid_structure(grid_structure)
            
# #             self.logger.info(f"Grid extraction completed. Valid: {is_valid}, Questions: {len(grid_structure)}")
            
# #             return grid_structure, bubble_regions, validation_issues
            
# #         except Exception as e:
# #             error_msg = f"Error in grid extraction pipeline: {e}"
# #             self.logger.error(error_msg)
# #             return {}, {}, [error_msg]
    
# #     def detect_grid_template(self, image: np.ndarray, debug_dir: str = None) -> Optional[Dict]:
# #         """
# #         Detect grid template/pattern for better bubble localization
        
# #         Args:
# #             image: Input binary image
# #             debug_dir: Directory for saving debug images
            
# #         Returns:
# #             Grid template information or None
# #         """
# #         try:
# #             # This is an advanced feature for detecting regular grid patterns
# #             # Can be implemented using template matching or Hough transforms
            
# #             # For now, return None to use the contour-based approach
# #             return None
            
# #         except Exception as e:
# #             self.logger.error(f"Error in grid template detection: {e}")
# #             return None
    
# #     def repair_grid_gaps(self, grid: Dict[int, Dict[int, Dict]], image_shape: Tuple[int, int]) -> Dict[int, Dict[int, Dict]]:
# #         """
# #         Attempt to repair missing bubbles in the grid using interpolation
        
# #         Args:
# #             grid: Current grid structure
# #             image_shape: Shape of the source image
            
# #         Returns:
# #             Repaired grid structure
# #         """
# #         try:
# #             repaired_grid = grid.copy()
            
# #             # Find questions with missing options
# #             for question_num in range(1, self.total_questions + 1):
# #                 if question_num not in repaired_grid:
# #                     repaired_grid[question_num] = {}
                
# #                 current_options = repaired_grid[question_num]
# #                 missing_options = []
                
# #                 for option_num in range(self.options_per_question):
# #                     if option_num not in current_options:
# #                         missing_options.append(option_num)
                
# #                 if missing_options and len(current_options) > 0:
# #                     # Try to interpolate missing bubble positions
# #                     existing_centers = [bubble['center'] for bubble in current_options.values()]
                    
# #                     if len(existing_centers) >= 2:
# #                         # Calculate average spacing between options
# #                         x_coords = sorted([center[0] for center in existing_centers])
# #                         if len(x_coords) >= 2:
# #                             avg_x_spacing = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
                            
# #                             # Estimate y-coordinate for this question
# #                             avg_y = sum([center[1] for center in existing_centers]) / len(existing_centers)
                            
# #                             # Interpolate missing bubbles
# #                             for missing_option in missing_options:
# #                                 estimated_x = x_coords[0] + (missing_option * avg_x_spacing)
                                
# #                                 # Create synthetic bubble info
# #                                 synthetic_bubble = {
# #                                     'center': (int(estimated_x), int(avg_y)),
# #                                     'bbox': (int(estimated_x-10), int(avg_y-10), 20, 20),
# #                                     'area': 400,
# #                                     'aspect_ratio': 1.0,
# #                                     'synthetic': True  # Mark as synthetic
# #                                 }
                                
# #                                 repaired_grid[question_num][missing_option] = synthetic_bubble
            
# #             repair_count = sum(1 for q_data in repaired_grid.values() 
# #                              for bubble in q_data.values() 
# #                              if bubble.get('synthetic', False))
            
# #             if repair_count > 0:
# #                 self.logger.info(f"Repaired {repair_count} missing bubbles using interpolation")
            
# #             return repaired_grid
            
# #         except Exception as e:
# #             self.logger.error(f"Error in grid repair: {e}")
# #             return grid
    
# #     def batch_extract_grids(self, images: List[np.ndarray], debug_base_dir: str = None) -> List[Dict]:
# #         """
# #         Batch processing for multiple images
        
# #         Args:
# #             images: List of processed binary images
# #             debug_base_dir: Base directory for debug outputs
            
# #         Returns:
# #             List of extraction results
# #         """
# #         results = []
        
# #         for i, image in enumerate(images):
# #             try:
# #                 debug_dir = None
# #                 if debug_base_dir and self.debug:
# #                     debug_dir = os.path.join(debug_base_dir, f"grid_{i:03d}")
# #                     os.makedirs(debug_dir, exist_ok=True)
                
# #                 grid_structure, bubble_regions, issues = self.extract_grid_pipeline(image, debug_dir)
                
# #                 results.append({
# #                     "index": i,
# #                     "grid_structure": grid_structure,
# #                     "bubble_regions": bubble_regions,
# #                     "validation_issues": issues,
# #                     "success": len(issues) == 0
# #                 })
                
# #             except Exception as e:
# #                 self.logger.error(f"Error extracting grid from image {i}: {e}")
# #                 results.append({
# #                     "index": i,
# #                     "grid_structure": {},
# #                     "bubble_regions": {},
# #                     "validation_issues": [str(e)],
# #                     "success": False
# #                 })
        
# #         successful = sum(1 for r in results if r["success"])
# #         self.logger.info(f"Batch grid extraction completed: {successful}/{len(results)} successful")
        
# #         return results

# # def main():
# #     """
# #     Main function for testing grid extraction module
# #     """
# #     import os
# #     from .utils import load_config
# #     from .preprocess import ImagePreprocessor
    
# #     # Load configuration
# #     config = load_config()
    
# #     # Initialize components
# #     preprocessor = ImagePreprocessor(config, debug=True)
# #     extractor = GridExtractor(config, debug=True)
    
# #     # Get test images
# #     test_dir = "data/test_images"  # Update path as needed
# #     if os.path.exists(test_dir):
# #         from .utils import get_image_files
# #         image_files = get_image_files(test_dir)

# #         if image_files:
# #             print(f"Found {len(image_files)} images for testing")

# #             # Test single image grid extraction
# #             test_image_path = image_files[0]
# #             debug_dir = "output/debug/grid_test"
# #             os.makedirs(debug_dir, exist_ok=True)

# #             try:
# #                 # Preprocess image first
# #                 original, binary, metadata = preprocessor.preprocess_pipeline(test_image_path, debug_dir)
# #                 print(f"Preprocessing completed for: {test_image_path}")

# #                 # Extract grid
# #                 grid_structure, bubble_regions, issues = extractor.extract_grid_pipeline(binary, debug_dir)

# #                 print(f"Grid extraction completed")
# #                 print(f"Questions detected: {len(grid_structure)}")
# #                 print(f"Total bubbles: {sum(len(q_data) for q_data in bubble_regions.values())}")

# #                 if issues:
# #                     print(f"Validation issues: {issues}")
# #                 else:
# #                     print("Grid validation passed!")

# #                 # Save results summary
# #                 summary = {
# #                     "questions_count": len(grid_structure),
# #                     "bubbles_count": sum(len(q_data) for q_data in bubble_regions.values()),
# #                     "validation_issues": issues
# #                 }

# #                 import json
# #                 with open(os.path.join(debug_dir, "grid_summary.json"), 'w') as f:
# #                     json.dump(summary, f, indent=2)

# #             except Exception as e:
# #                 print(f"Error in grid extraction test: {e}")
# #         else:
# #             print("No test images found")
# #     else:
# #         print(f"Test directory not found: {test_dir}")

# # if __name__ == "__main__":
# #     main()





# """
# Grid extraction module for OMR sheets
# Identifies and extracts bubble grid from processed OMR sheets
# """

# import cv2
# import numpy as np
# import logging
# import os
# from typing import Tuple, List, Dict, Optional
# from .utils import setup_logging, save_debug_image

# class GridExtractor:
#     """
#     Extracts bubble grid from OMR sheet images
#     """
    
#     def __init__(self, config: dict, debug: bool = False):
#         """
#         Initialize grid extractor with configuration
        
#         Args:
#             config: Configuration dictionary
#             debug: Enable debug mode for saving intermediate images
#         """
#         self.config = config
#         self.debug = debug
#         self.logger = setup_logging()
        
#         # Extract grid parameters with more flexible defaults
#         self.total_questions = config['grid']['rows']
#         self.options_per_question = config['grid']['cols']
#         self.bubble_min_area = config['grid']['bubble_min_area']
#         self.bubble_max_area = config['grid']['bubble_max_area']
#         self.aspect_ratio_tolerance = config['grid']['bubble_aspect_ratio_tolerance']
#         self.grid_tolerance = config['grid']['grid_tolerance']
        
#         # Enhanced detection parameters
#         self.min_bubbles_per_row = 2  # Minimum bubbles to consider a valid row
#         self.max_bubbles_per_row = 6  # Maximum bubbles per row (allows for some variation)
    
#     def find_bubble_contours(self, binary_image: np.ndarray, debug_dir: str = None) -> List[Dict]:
#         """
#         Find all potential bubble contours in the image with improved detection
#         """
#         try:
#             # Ensure binary image is properly formatted
#             if len(binary_image.shape) == 3:
#                 binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            
#             # Apply morphological operations to clean up the binary image
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#             cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
#             cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)
            
#             if self.debug and debug_dir:
#                 save_debug_image(cleaned_image, "13_cleaned_binary.jpg", debug_dir)
            
#             # Find contours with better parameters
#             contours, _ = cv2.findContours(
#                 cleaned_image, 
#                 cv2.RETR_EXTERNAL, 
#                 cv2.CHAIN_APPROX_SIMPLE
#             )
            
#             bubble_candidates = []
#             valid_contours = 0
            
#             for i, contour in enumerate(contours):
#                 # Calculate contour properties
#                 area = cv2.contourArea(contour)
                
#                 # More flexible area filtering
#                 if area < self.bubble_min_area * 0.5 or area > self.bubble_max_area * 2:
#                     continue
                
#                 # Get bounding rectangle
#                 x, y, w, h = cv2.boundingRect(contour)
                
#                 # Skip very small or very large rectangles
#                 if w < 8 or h < 8 or w > 60 or h > 60:
#                     continue
                
#                 # Calculate aspect ratio (more tolerant)
#                 aspect_ratio = w / h if h > 0 else 0
#                 if aspect_ratio < 0.5 or aspect_ratio > 2.0:
#                     continue
                
#                 # Calculate center point using moments for better accuracy
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     center_x = int(M["m10"] / M["m00"])
#                     center_y = int(M["m01"] / M["m00"])
#                 else:
#                     center_x = x + w // 2
#                     center_y = y + h // 2
                
#                 # Additional circularity check
#                 perimeter = cv2.arcLength(contour, True)
#                 if perimeter > 0:
#                     circularity = 4 * np.pi * area / (perimeter * perimeter)
#                     if circularity < 0.2:  # Not circular enough
#                         continue
                
#                 # Store bubble information
#                 bubble_info = {
#                     'contour': contour,
#                     'area': area,
#                     'center': (center_x, center_y),
#                     'bbox': (x, y, w, h),
#                     'aspect_ratio': aspect_ratio,
#                     'circularity': circularity
#                 }
                
#                 bubble_candidates.append(bubble_info)
#                 valid_contours += 1
            
#             # Debug: Draw detected bubbles
#             if self.debug and debug_dir and bubble_candidates:
#                 debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
#                 for i, bubble in enumerate(bubble_candidates):
#                     center = bubble['center']
#                     # Color code by confidence
#                     color = (0, 255, 0) if bubble['circularity'] > 0.5 else (0, 165, 255)
#                     cv2.circle(debug_image, center, 5, color, -1)
#                     x, y, w, h = bubble['bbox']
#                     cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 1)
#                     # Add index for tracking
#                     cv2.putText(debug_image, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
#                 save_debug_image(debug_image, "14_detected_bubbles.jpg", debug_dir)
#                 self.logger.info(f"Found {valid_contours} valid bubble candidates (total contours: {len(contours)})")
            
#             return bubble_candidates
            
#         except Exception as e:
#             self.logger.error(f"Error finding bubble contours: {e}")
#             return []
    
#     def organize_bubbles_into_grid(self, bubbles: List[Dict], debug_dir: str = None) -> Dict[int, Dict[int, Dict]]:
#         """
#         Organize detected bubbles into a grid structure with improved row detection
#         """
#         try:
#             if not bubbles:
#                 self.logger.warning("No bubbles to organize into grid")
#                 return {}
            
#             # Sort bubbles by y-coordinate (top to bottom)
#             bubbles_sorted = sorted(bubbles, key=lambda b: b['center'][1])
            
#             # More sophisticated row grouping
#             rows = self._group_bubbles_into_rows(bubbles_sorted, debug_dir)
            
#             if not rows:
#                 self.logger.warning("No valid rows detected")
#                 return {}
            
#             # Sort bubbles within each row by x-coordinate (left to right)
#             for row in rows:
#                 row.sort(key=lambda b: b['center'][0])
            
#             # Create grid structure with flexible column assignment
#             grid = self._create_flexible_grid(rows)
            
#             # Debug visualization
#             if self.debug and debug_dir:
#                 self._visualize_grid(bubbles_sorted, rows, grid, debug_dir)
            
#             questions_count = len(grid)
#             total_bubbles = sum(len(q_data) for q_data in grid.values())
#             self.logger.info(f"Organized {questions_count} questions with {total_bubbles} bubbles "
#                            f"(expected: {self.total_questions} questions, {self.options_per_question} options)")
            
#             return grid
            
#         except Exception as e:
#             self.logger.error(f"Error organizing bubbles into grid: {e}")
#             return {}
    
#     def _group_bubbles_into_rows(self, bubbles_sorted: List[Dict], debug_dir: str = None) -> List[List[Dict]]:
#         """
#         Group bubbles into rows using clustering-based approach
#         """
#         try:
#             if len(bubbles_sorted) < self.min_bubbles_per_row:
#                 return []
            
#             y_positions = np.array([b['center'][1] for b in bubbles_sorted])
            
#             # Use K-means clustering to find row centers
#             from sklearn.cluster import KMeans
#             n_clusters = min(20, len(bubbles_sorted) // self.min_bubbles_per_row)  # Estimate number of rows
            
#             if n_clusters < 2:
#                 # Fallback to simple spacing-based grouping
#                 return self._simple_row_grouping(bubbles_sorted)
            
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#             cluster_labels = kmeans.fit_predict(y_positions.reshape(-1, 1))
#             row_centers = kmeans.cluster_centers_.flatten()
            
#             # Sort row centers
#             sorted_indices = np.argsort(row_centers)
#             row_centers = row_centers[sorted_indices]
            
#             # Assign bubbles to rows based on proximity to row centers
#             rows = []
#             row_assignments = {}
            
#             for i, bubble in enumerate(bubbles_sorted):
#                 bubble_y = bubble['center'][1]
#                 # Find closest row center
#                 distances = np.abs(row_centers - bubble_y)
#                 closest_row_idx = np.argmin(distances)
                
#                 row_idx = closest_row_idx
#                 if row_idx not in row_assignments:
#                     row_assignments[row_idx] = []
                
#                 row_assignments[row_idx].append(bubble)
            
#             # Filter valid rows (must have minimum number of bubbles)
#             for row_idx in sorted(row_assignments.keys()):
#                 row_bubbles = row_assignments[row_idx]
#                 if len(row_bubbles) >= self.min_bubbles_per_row and len(row_bubbles) <= self.max_bubbles_per_row:
#                     rows.append(row_bubbles)
            
#             # Limit to reasonable number of rows
#             if len(rows) > self.total_questions * 1.5:
#                 # Sort rows by their average y position and take top candidates
#                 rows.sort(key=lambda r: np.mean([b['center'][1] for b in r]))
#                 rows = rows[:self.total_questions]
            
#             self.logger.info(f"Grouped into {len(rows)} valid rows using clustering")
#             return rows
            
#         except Exception as e:
#             self.logger.warning(f"Clustering failed, using simple row grouping: {e}")
#             return self._simple_row_grouping(bubbles_sorted)
    
#     def _simple_row_grouping(self, bubbles_sorted: List[Dict]) -> List[List[Dict]]:
#         """
#         Fallback simple row grouping based on y-spacing
#         """
#         rows = []
#         current_row = []
#         current_y = bubbles_sorted[0]['center'][1]
        
#         # Calculate average row spacing
#         y_diffs = []
#         for i in range(1, len(bubbles_sorted)):
#             diff = bubbles_sorted[i]['center'][1] - bubbles_sorted[i-1]['center'][1]
#             if diff > 10:  # Only consider significant gaps
#                 y_diffs.append(diff)
        
#         if y_diffs:
#             avg_row_spacing = np.median(y_diffs)
#             row_tolerance = avg_row_spacing * 0.4  # 40% tolerance
#         else:
#             avg_row_spacing = 40
#             row_tolerance = 15
        
#         for bubble in bubbles_sorted:
#             bubble_y = bubble['center'][1]
            
#             # Check if this bubble belongs to the current row
#             if abs(bubble_y - current_y) <= row_tolerance:
#                 current_row.append(bubble)
#             else:
#                 # End current row if it has enough bubbles
#                 if len(current_row) >= self.min_bubbles_per_row:
#                     rows.append(current_row)
                
#                 # Start new row
#                 current_row = [bubble]
#                 current_y = bubble_y
        
#         # Add last row if valid
#         if len(current_row) >= self.min_bubbles_per_row:
#             rows.append(current_row)
        
#         return rows
    
#     def _create_flexible_grid(self, rows: List[List[Dict]]) -> Dict[int, Dict[int, Dict]]:
#         """
#         Create grid structure with flexible column assignment
#         """
#         grid = {}
#         question_number = 1
        
#         for row_idx, row_bubbles in enumerate(rows):
#             if question_number > self.total_questions:
#                 break
            
#             # Determine column positions dynamically
#             x_positions = [b['center'][0] for b in row_bubbles]
#             if len(x_positions) < 2:
#                 question_number += 1
#                 continue
            
#             # Sort x positions and create evenly spaced columns
#             sorted_x = sorted(x_positions)
#             x_spacing = (sorted_x[-1] - sorted_x[0]) / (len(sorted_x) - 1) if len(sorted_x) > 1 else 30
            
#             # Create column mapping
#             col_positions = {}
#             for i, x_pos in enumerate(sorted_x):
#                 col_pos = round((x_pos - sorted_x[0]) / x_spacing)
#                 col_positions[x_pos] = min(col_pos, self.options_per_question - 1)
            
#             # Assign bubbles to columns
#             question_data = {}
#             for bubble in row_bubbles:
#                 x_pos = bubble['center'][0]
#                 option_num = col_positions.get(x_pos, 0)  # Default to option 0 if unclear
#                 question_data[option_num] = bubble
            
#             # Fill missing columns if we have enough bubbles
#             available_options = sorted(question_data.keys())
#             if len(available_options) >= 2:
#                 # Try to fill gaps in standard positions (0,1,2,3 for A,B,C,D)
#                 for std_option in range(self.options_per_question):
#                     if std_option not in question_data and len(question_data) < self.options_per_question:
#                         # Find closest available option to fill this position
#                         closest_option = min(available_options, key=lambda x: abs(x - std_option))
#                         if abs(closest_option - std_option) <= 1:  # Only fill adjacent gaps
#                             # Create synthetic entry pointing to real bubble
#                             question_data[std_option] = question_data[closest_option]
#                             self.logger.debug(f"Filled gap for Q{question_number} option {std_option}")
            
#             grid[question_number] = question_data
#             question_number += 1
        
#         return grid
    
#     def _visualize_grid(self, all_bubbles: List[Dict], rows: List[List[Dict]], grid: Dict, debug_dir: str):
#         """
#         Create comprehensive grid visualization
#         """
#         try:
#             # Create larger visualization canvas
#             img_height = max(1400, len(all_bubbles) * 2)
#             img_width = 1000
#             debug_image = np.zeros((img_height, img_width, 3), dtype=np.uint8) * 255
            
#             # Colors for different options
#             colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # A, B, C, D
            
#             # Draw row groupings
#             for row_idx, row in enumerate(rows):
#                 avg_y = np.mean([b['center'][1] for b in row])
#                 # Draw row boundary
#                 cv2.line(debug_image, (0, int(avg_y)), (img_width, int(avg_y)), (200, 200, 200), 1)
#                 cv2.putText(debug_image, f"Row {row_idx+1} ({len(row)} bubbles)", 
#                            (10, int(avg_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
#             # Draw grid bubbles
#             for question_num, question_data in grid.items():
#                 for option_num, bubble_info in question_data.items():
#                     # Scale coordinates to fit visualization
#                     center_x = int(bubble_info['center'][0] * img_width / 800)  # Assuming 800px width
#                     center_y = int(bubble_info['center'][1] * img_height / 1200)  # Assuming 1200px height
#                     color = colors[option_num % len(colors)]
                    
#                     # Draw bubble
#                     radius = max(6, min(bubble_info['bbox'][2], bubble_info['bbox'][3]) // 4)
#                     cv2.circle(debug_image, (center_x, center_y), radius, color, 2)
                    
#                     # Add question and option labels
#                     label = f"Q{question_num}{chr(65+option_num)}"
#                     cv2.putText(debug_image, label, (center_x-20, center_y-10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
#             # Add legend
#             legend_y = img_height - 80
#             for i, (color, letter) in enumerate(zip(colors, 'ABCD')):
#                 x_start = 50 + i * 80
#                 cv2.circle(debug_image, (x_start, legend_y), 8, color, -1)
#                 cv2.putText(debug_image, f"Option {letter}", (x_start+15, legend_y+5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
#             cv2.putText(debug_image, f"Grid: {len(grid)} questions, {sum(len(q) for q in grid.values())} bubbles", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
#             save_debug_image(debug_image, "15_grid_organization.jpg", debug_dir)
            
#         except Exception as e:
#             self.logger.error(f"Error in grid visualization: {e}")
    
#     def extract_bubble_regions(self, image: np.ndarray, grid: Dict[int, Dict[int, Dict]], 
#                               debug_dir: str = None) -> Dict[int, Dict[int, dict]]:
#         """
#         Extract individual bubble regions with improved error handling
#         """
#         try:
#             if len(image.shape) == 3:
#                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray_image = image.copy()
            
#             bubble_regions = {}
#             padding = 8  # Increased padding for better bubble capture
#             successful_extractions = 0
#             total_attempts = 0
            
#             for question_num, question_data in grid.items():
#                 bubble_regions[question_num] = {}
                
#                 for option_num, bubble_info in question_data.items():
#                     total_attempts += 1
#                     try:
#                         # Get bubble bounding box
#                         x, y, w, h = bubble_info['bbox']
                        
#                         # Ensure coordinates are within image bounds
#                         x_start = max(0, x - padding)
#                         y_start = max(0, y - padding)
#                         x_end = min(gray_image.shape[1], x + w + padding)
#                         y_end = min(gray_image.shape[0], y + h + padding)
                        
#                         # Extract bubble region
#                         bubble_img = gray_image[y_start:y_end, x_start:x_end].copy()
                        
#                         if bubble_img.size > 0 and bubble_img.shape[0] > 0 and bubble_img.shape[1] > 0:
#                             bubble_region = {
#                                 'image': bubble_img,
#                                 'bbox': (x, y, w, h),
#                                 'center': bubble_info['center'],
#                                 'area': bubble_info['area'],
#                                 'aspect_ratio': bubble_info['aspect_ratio']
#                             }
#                             bubble_regions[question_num][option_num] = bubble_region
#                             successful_extractions += 1
#                         else:
#                             self.logger.warning(f"Empty bubble region for Q{question_num} Opt{option_num}")
                            
#                     except Exception as e:
#                         self.logger.warning(f"Error extracting bubble Q{question_num} Opt{option_num}: {e}")
#                         continue
            
#             self.logger.info(f"Extracted {successful_extractions}/{total_attempts} bubble regions successfully")
            
#             # Debug: Save sample bubble regions
#             if self.debug and debug_dir and bubble_regions:
#                 self._save_sample_bubbles(bubble_regions, debug_dir)
            
#             return bubble_regions
            
#         except Exception as e:
#             self.logger.error(f"Error extracting bubble regions: {e}")
#             return {}
    
#     def _save_sample_bubbles(self, bubble_regions: Dict[int, Dict[int, dict]], debug_dir: str):
#         """
#         Save sample bubble regions for debugging with better error handling
#         """
#         try:
#             sample_dir = os.path.join(debug_dir, "sample_bubbles")
#             os.makedirs(sample_dir, exist_ok=True)
            
#             # Save first 10 questions as samples
#             sample_questions = list(bubble_regions.keys())[:10]
#             saved_count = 0
            
#             for question_num in sample_questions:
#                 if question_num in bubble_regions:
#                     question_data = bubble_regions[question_num]
#                     for option_num, bubble_dict in question_data.items():
#                         if 'image' in bubble_dict and isinstance(bubble_dict['image'], np.ndarray):
#                             option_letter = chr(65 + option_num)
#                             filename = f"Q{question_num:02d}_{option_letter}.png"
#                             filepath = os.path.join(sample_dir, filename)
                            
#                             bubble_image = bubble_dict['image']
#                             if bubble_image.size > 0:
#                                 # Ensure it's grayscale
#                                 if len(bubble_image.shape) == 3:
#                                     bubble_image = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2GRAY)
                                
#                                 # Resize for consistent visualization
#                                 resized_bubble = cv2.resize(bubble_image, (64, 64), interpolation=cv2.INTER_NEAREST)
#                                 cv2.imwrite(filepath, resized_bubble)
#                                 saved_count += 1
            
#             self.logger.debug(f"Saved {saved_count} sample bubble images")
            
#         except Exception as e:
#             self.logger.error(f"Error saving sample bubbles: {e}")
    
#     def validate_grid_structure(self, grid: Dict[int, Dict[int, Dict]]) -> Tuple[bool, List[str]]:
#         """
#         Validate grid structure with more flexible criteria
#         """
#         issues = []
        
#         try:
#             questions_detected = len(grid)
            
#             # More flexible question count validation
#             expected_min = int(self.total_questions * 0.6)  # Allow 60% minimum
#             if questions_detected < expected_min:
#                 issues.append(f"Too few questions detected: {questions_detected}/{self.total_questions} "
#                              f"(minimum expected: {expected_min})")
            
#             # Check option completeness with tolerance
#             incomplete_count = 0
#             for question_num in range(1, questions_detected + 1):
#                 if question_num in grid:
#                     options_count = len(grid[question_num])
#                     expected_options = self.options_per_question
#                     if options_count < expected_options * 0.75:  # Allow 25% missing options
#                         incomplete_count += 1
            
#             if incomplete_count > questions_detected * 0.3:  # More than 30% incomplete
#                 issues.append(f"High incomplete question rate: {incomplete_count}/{questions_detected}")
            
#             # Check grid spacing consistency (only if we have enough questions)
#             if questions_detected >= 5:
#                 question_numbers = sorted(grid.keys())
#                 y_positions = []
                
#                 for q_num in question_numbers:
#                     if grid[q_num]:
#                         y_coords = [bubble['center'][1] for bubble in grid[q_num].values()]
#                         avg_y = sum(y_coords) / len(y_coords)
#                         y_positions.append(avg_y)
                
#                 if len(y_positions) >= 2:
#                     y_diffs = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
#                     avg_spacing = np.mean(y_diffs)
#                     std_spacing = np.std(y_diffs)
                    
#                     # More tolerant spacing check
#                     if std_spacing > avg_spacing * 0.7:
#                         issues.append(f"High spacing variation: std={std_spacing:.1f}, avg={avg_spacing:.1f}")
            
#             # Overall validity
#             is_valid = len(issues) <= 1  # Allow one minor issue
            
#             if is_valid:
#                 self.logger.info("Grid structure validation passed")
#             else:
#                 self.logger.warning(f"Grid validation issues ({len(issues)}): {issues}")
            
#             return is_valid, issues
            
#         except Exception as e:
#             self.logger.error(f"Error in grid validation: {e}")
#             return False, [f"Validation error: {str(e)}"]
    
#     def extract_grid_pipeline(self, processed_image: np.ndarray, debug_dir: str = None) -> Tuple[Dict, Dict, List[str]]:
#         """
#         Complete pipeline for grid extraction with enhanced error handling
#         """
#         try:
#             self.logger.info("Starting enhanced grid extraction pipeline")
            
#             # Step 1: Find bubble contours with improved detection
#             bubble_candidates = self.find_bubble_contours(processed_image, debug_dir)
            
#             if not bubble_candidates:
#                 issues = ["No bubble candidates found - check image preprocessing"]
#                 return {}, {}, issues
            
#             # Step 2: Organize into grid structure
#             grid_structure = self.organize_bubbles_into_grid(bubble_candidates, debug_dir)
            
#             if not grid_structure:
#                 issues = ["Failed to organize bubbles into valid grid structure"]
#                 return {}, {}, issues
            
#             # Step 3: Extract bubble regions
#             bubble_regions = self.extract_bubble_regions(processed_image, grid_structure, debug_dir)
            
#             # Step 4: Validate and report
#             is_valid, validation_issues = self.validate_grid_structure(grid_structure)
            
#             # Log detailed statistics
#             questions_count = len(grid_structure)
#             total_bubbles = sum(len(q_data) for q_data in bubble_regions.values())
#             self.logger.info(f"Grid extraction completed: {questions_count} questions, {total_bubbles} bubbles, "
#                            f"valid: {is_valid}")
            
#             # Return results
#             return grid_structure, bubble_regions, validation_issues
            
#         except Exception as e:
#             error_msg = f"Error in grid extraction pipeline: {e}"
#             self.logger.error(error_msg)
#             return {}, {}, [error_msg]
    
#     def repair_grid_gaps(self, grid: Dict[int, Dict[int, Dict]], image_shape: Tuple[int, int]) -> Dict[int, Dict[int, Dict]]:
#         """
#         Repair missing bubbles using interpolation with improved logic
#         """
#         try:
#             repaired_grid = {k: v.copy() for k, v in grid.items()}
#             repair_count = 0
            
#             for question_num in range(1, self.total_questions + 1):
#                 if question_num not in repaired_grid:
#                     repaired_grid[question_num] = {}
#                     continue
                
#                 current_options = repaired_grid[question_num]
#                 available_options = sorted(current_options.keys())
                
#                 # Only repair if we have some valid options
#                 if len(available_options) >= 2:
#                     # Calculate expected positions
#                     x_positions = [current_options[opt]['center'][0] for opt in available_options]
#                     avg_y = np.mean([current_options[opt]['center'][1] for opt in available_options])
                    
#                     # Estimate spacing
#                     if len(x_positions) >= 2:
#                         spacing = (max(x_positions) - min(x_positions)) / (len(x_positions) - 1)
                        
#                         # Check for missing standard positions (0,1,2,3)
#                         for expected_option in range(self.options_per_question):
#                             if expected_option not in current_options:
#                                 # Estimate position
#                                 estimated_x = min(x_positions) + expected_option * spacing
                                
#                                 # Only add if position is reasonable
#                                 if abs(estimated_x - min(x_positions)) < spacing * 2:
#                                     synthetic_bubble = {
#                                         'center': (int(estimated_x), int(avg_y)),
#                                         'bbox': (int(estimated_x-12), int(avg_y-12), 24, 24),
#                                         'area': 400,
#                                         'aspect_ratio': 1.0,
#                                         'synthetic': True
#                                     }
#                                     current_options[expected_option] = synthetic_bubble
#                                     repair_count += 1
#                                     self.logger.debug(f"Repaired synthetic bubble for Q{question_num} Opt{expected_option}")
            
#             if repair_count > 0:
#                 self.logger.info(f"Repaired {repair_count} missing bubbles using interpolation")
            
#             return repaired_grid
            
#         except Exception as e:
#             self.logger.error(f"Error in grid repair: {e}")
#             return grid
    
#     def batch_extract_grids(self, images: List[np.ndarray], debug_base_dir: str = None) -> List[Dict]:
#         """
#         Batch processing for multiple images with enhanced error handling
#         """
#         results = []
        
#         for i, image in enumerate(images):
#             try:
#                 debug_dir = None
#                 if debug_base_dir and self.debug:
#                     debug_dir = os.path.join(debug_base_dir, f"grid_{i:03d}")
#                     os.makedirs(debug_dir, exist_ok=True)
                
#                 grid_structure, bubble_regions, issues = self.extract_grid_pipeline(image, debug_dir)
                
#                 # Apply gap repair
#                 if grid_structure:
#                     grid_structure = self.repair_grid_gaps(grid_structure, image.shape)
                
#                 result = {
#                     "index": i,
#                     "grid_structure": grid_structure,
#                     "bubble_regions": bubble_regions,
#                     "validation_issues": issues,
#                     "questions_detected": len(grid_structure),
#                     "total_bubbles": sum(len(q_data) for q_data in bubble_regions.values()),
#                     "success": len(issues) <= 1  # Allow minor issues
#                 }
                
#                 results.append(result)
                
#             except Exception as e:
#                 self.logger.error(f"Error extracting grid from image {i}: {e}")
#                 results.append({
#                     "index": i,
#                     "grid_structure": {},
#                     "bubble_regions": {},
#                     "validation_issues": [str(e)],
#                     "success": False
#                 })
        
#         # Log batch summary
#         successful = sum(1 for r in results if r["success"])
#         total_questions = sum(r.get("questions_detected", 0) for r in results)
#         avg_questions = total_questions / len(results) if results else 0
        
#         self.logger.info(f"Batch grid extraction: {successful}/{len(results)} successful, "
#                         f"avg {avg_questions:.1f} questions per sheet")
        
#         return results

# def main():
#     """
#     Main function for testing grid extraction module
#     """
#     import os
#     from .utils import load_config
#     from .preprocess import ImagePreprocessor
    
#     # Load configuration
#     config = load_config()
    
#     # Initialize components
#     preprocessor = ImagePreprocessor(config, debug=True)
#     extractor = GridExtractor(config, debug=True)
    
#     # Get test images
#     test_dir = "data/test_images"
#     if os.path.exists(test_dir):
#         from .utils import get_image_files
#         image_files = get_image_files(test_dir)

#         if image_files:
#             print(f"Found {len(image_files)} images for testing")

#             # Test single image grid extraction
#             test_image_path = image_files[0]
#             debug_dir = "output/debug/grid_test"
#             os.makedirs(debug_dir, exist_ok=True)

#             try:
#                 # Preprocess image first
#                 original, binary, metadata = preprocessor.preprocess_pipeline(test_image_path, debug_dir)
#                 print(f"Preprocessing completed for: {test_image_path}")

#                 # Extract grid with enhanced pipeline
#                 grid_structure, bubble_regions, issues = extractor.extract_grid_pipeline(binary, debug_dir)

#                 print(f"\nEnhanced Grid Extraction Results:")
#                 print(f"Questions detected: {len(grid_structure)}")
#                 print(f"Total bubbles extracted: {sum(len(q_data) for q_data in bubble_regions.values())}")
#                 print(f"Validation issues: {len(issues)}")

#                 if issues:
#                     print(f"Issues: {issues}")
#                 else:
#                     print("✓ Grid validation passed!")

#                 # Summary statistics
#                 if grid_structure:
#                     question_stats = {}
#                     for q_num, q_data in grid_structure.items():
#                         question_stats[q_num] = len(q_data)
                    
#                     print(f"\nQuestion breakdown:")
#                     for q_num in sorted(question_stats.keys())[:10]:  # First 10
#                         print(f"  Q{q_num}: {question_stats[q_num]} options")
#                     if len(question_stats) > 10:
#                         print(f"  ... and {len(question_stats)-10} more")

#                 # Save results
#                 import json
#                 summary = {
#                     "questions_count": len(grid_structure),
#                     "bubbles_count": sum(len(q_data) for q_data in bubble_regions.values()),
#                     "validation_issues": issues,
#                     "question_stats": {str(k): v for k, v in question_stats.items()}
#                 }

#                 with open(os.path.join(debug_dir, "enhanced_grid_summary.json"), 'w') as f:
#                     json.dump(summary, f, indent=2)

#             except Exception as e:
#                 print(f"Error in enhanced grid extraction test: {e}")
#                 import traceback
#                 traceback.print_exc()
#         else:
#             print("No test images found")
#     else:
#         print(f"Test directory not found: {test_dir}")

# if __name__ == "__main__":
#     main()




"""
Grid extraction module for OMR sheets
Enhanced version with multi-pass detection for better coverage
"""

import cv2
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import KMeans
from .utils import setup_logging, save_debug_image

import json

class GridExtractor:
    """
    Enhanced grid extractor with multi-pass detection for better OMR sheet coverage
    """
    
    def __init__(self, config: dict, debug: bool = False):
        """
        Initialize enhanced grid extractor
        """
        self.config = config
        self.debug = debug
        self.logger = setup_logging()
        
        # Base parameters
        self.total_questions = config['grid']['rows']
        self.options_per_question = config['grid']['cols']
        self.bubble_min_area = config['grid']['bubble_min_area']
        self.bubble_max_area = config['grid']['bubble_max_area']
        self.aspect_ratio_tolerance = config['grid']['bubble_aspect_ratio_tolerance']
        self.grid_tolerance = config['grid']['grid_tolerance']
        
        # Enhanced detection parameters
        self.min_bubbles_per_row = 2
        self.max_bubbles_per_row = 8  # Increased to catch more variations
        self.max_detection_passes = 3  # Multiple passes for better coverage
        self.row_spacing_threshold = 25  # Minimum spacing between rows
        self.col_spacing_threshold = 20  # Minimum spacing between columns
        
        # Adaptive thresholds
        self.adaptive_area_min = self.bubble_min_area * 0.5
        self.adaptive_area_max = self.bubble_max_area * 1.5
    
    def multi_pass_bubble_detection(self, binary_image: np.ndarray, debug_dir: str = None) -> List[Dict]:
        """
        Perform multiple passes of bubble detection with different parameters
        """
        all_bubbles = []
        pass_results = []
        
        # Define different detection strategies
        detection_strategies = [
            {
                'name': 'Standard',
                'min_area': self.bubble_min_area,
                'max_area': self.bubble_max_area,
                'morph_kernel': (3, 3),
                'min_circularity': 0.2
            },
            {
                'name': 'Sensitive',
                'min_area': self.adaptive_area_min,
                'max_area': self.bubble_max_area * 1.2,
                'morph_kernel': (2, 2),
                'min_circularity': 0.15
            },
            {
                'name': 'Robust',
                'min_area': self.bubble_min_area * 0.8,
                'max_area': self.adaptive_area_max,
                'morph_kernel': (4, 4),
                'min_circularity': 0.25
            }
        ]
        
        for pass_num, strategy in enumerate(detection_strategies, 1):
            self.logger.debug(f"Running detection pass {pass_num}: {strategy['name']}")
            
            try:
                # Apply strategy-specific preprocessing
                processed_image = self._preprocess_for_detection(binary_image, strategy, debug_dir, pass_num)
                
                # Find contours
                contours, _ = cv2.findContours(
                    processed_image, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                pass_bubbles = []
                valid_count = 0
                
                for contour in contours:
                    bubble_info = self._analyze_contour(contour, strategy)
                    if bubble_info:
                        # Check for overlap with existing bubbles
                        if not self._has_significant_overlap(bubble_info, all_bubbles):
                            pass_bubbles.append(bubble_info)
                            all_bubbles.append(bubble_info)
                            valid_count += 1
                
                pass_results.append({
                    'strategy': strategy['name'],
                    'valid_bubbles': valid_count,
                    'total_contours': len(contours)
                })
                
                self.logger.debug(f"Pass {pass_num} ({strategy['name']}): {valid_count} valid bubbles from {len(contours)} contours")
                
                # Early exit if we have enough bubbles
                if len(all_bubbles) >= self.total_questions * self.options_per_question * 0.8:
                    self.logger.debug(f"Early exit after pass {pass_num}: sufficient bubbles detected")
                    break
                    
            except Exception as e:
                self.logger.warning(f"Detection pass {pass_num} failed: {e}")
                continue
        
        self.logger.info(f"Multi-pass detection completed: {len(all_bubbles)} total bubbles from {len(pass_results)} passes")
        
        # Debug visualization of all detected bubbles
        if self.debug and debug_dir and all_bubbles:
            self._visualize_all_detected_bubbles(binary_image, all_bubbles, pass_results, debug_dir)
        
        return all_bubbles
    
    def _preprocess_for_detection(self, binary_image: np.ndarray, strategy: dict, 
                                debug_dir: str = None, pass_num: int = 1) -> np.ndarray:
        """
        Apply strategy-specific preprocessing
        """
        try:
            # Ensure binary image is properly formatted
            if len(binary_image.shape) == 3:
                working_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            else:
                working_image = binary_image.copy()
            
            # Apply morphological operations based on strategy
            kernel_size = strategy['morph_kernel']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            
            # Close small gaps
            closed = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Optional: Apply slight dilation for faint bubbles
            if pass_num == 2:  # Sensitive pass
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                opened = cv2.dilate(opened, dilate_kernel, iterations=1)
            
            if self.debug and debug_dir:
                save_debug_image(
                    opened, 
                    f"13_pass{pass_num}_{strategy['name'].lower()}_processed.jpg", 
                    debug_dir
                )
            
            return opened
            
        except Exception as e:
            self.logger.error(f"Preprocessing error for pass {pass_num}: {e}")
            return binary_image
    
    def _analyze_contour(self, contour, strategy: dict) -> Optional[Dict]:
        """
        Analyze a single contour and determine if it's a valid bubble
        """
        try:
            # Basic properties
            area = cv2.contourArea(contour)
            
            # Apply stricter area filtering
            if area < strategy['min_area'] * 1.2 or area > strategy['max_area'] * 0.8:
                return None

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Stricter size filtering
            if w < 12 or h < 12 or w > 50 or h > 50:
                return None

            # Stricter aspect ratio check
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.8 or aspect_ratio > 1.25:
                return None

            # Stricter circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < max(strategy['min_circularity'], 0.5):
                    return None
            
            # Center calculation
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x = x + w // 2
                center_y = y + h // 2
            
            return {
                'contour': contour,
                'area': area,
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'detection_pass': strategy['name']
            }
            
        except Exception as e:
            self.logger.debug(f"Contour analysis failed: {e}")
            return None
    
    def _has_significant_overlap(self, new_bubble: Dict, existing_bubbles: List[Dict], 
                               overlap_threshold: float = 0.3) -> bool:
        """
        Check if new bubble significantly overlaps with existing bubbles
        """
        try:
            new_x, new_y, new_w, new_h = new_bubble['bbox']
            new_area = new_w * new_h
            
            for existing in existing_bubbles:
                ex, ey, ew, eh = existing['bbox']
                ex_area = ew * eh
                
                # Calculate overlap
                x_overlap = max(0, min(new_x + new_w, ex + ew) - max(new_x, ex))
                y_overlap = max(0, min(new_y + new_h, ey + eh) - max(new_y, ey))
                overlap_area = x_overlap * y_overlap
                
                # Check if overlap is significant
                if overlap_area > 0:
                    overlap_ratio = overlap_area / min(new_area, ex_area)
                    if overlap_ratio > overlap_threshold:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _visualize_all_detected_bubbles(self, binary_image: np.ndarray, all_bubbles: List[Dict], 
                                      pass_results: List[Dict], debug_dir: str):
        """
        Create comprehensive visualization of all detected bubbles
        """
        try:
            h, w = binary_image.shape[:2]
            debug_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
            
            # Color coding by detection pass
            colors = {
                'Standard': (0, 255, 0),    # Green
                'Sensitive': (0, 165, 255), # Orange
                'Robust': (255, 0, 255)     # Magenta
            }
            
            bubble_counts = {k: 0 for k in colors.keys()}
            
            for bubble in all_bubbles:
                center = bubble['center']
                detection_pass = bubble.get('detection_pass', 'Standard')
                color = colors.get(detection_pass, (128, 128, 128))
                bubble_counts[detection_pass] += 1
                
                # Draw bubble
                radius = max(3, min(bubble['bbox'][2], bubble['bbox'][3]) // 4)
                cv2.circle(debug_image, center, radius, color, -1)
                
                # Draw bounding box
                x, y, bw, bh = bubble['bbox']
                cv2.rectangle(debug_image, (x, y), (x+bw, y+bh), color, 1)
                
                # Add pass label
                label = detection_pass[0]  # First letter
                cv2.putText(debug_image, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Add legend
            legend_y = 30
            for pass_name, color in colors.items():
                count = bubble_counts.get(pass_name, 0)
                x_start = 20
                cv2.circle(debug_image, (x_start, legend_y), 8, color, -1)
                cv2.putText(debug_image, f"{pass_name}: {count}", 
                           (x_start+15, legend_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                legend_y += 25
            
            # Add total count
            total_bubbles = len(all_bubbles)
            cv2.putText(debug_image, f"Total: {total_bubbles} bubbles", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            save_debug_image(debug_image, "14_all_detected_bubbles.jpg", debug_dir)
            
        except Exception as e:
            self.logger.error(f"Error visualizing detected bubbles: {e}")
    
    def enhanced_row_clustering(self, bubbles: List[Dict], debug_dir: str = None) -> List[List[Dict]]:
        """
        Enhanced row clustering with multiple grouping strategies
        """
        if len(bubbles) < self.min_bubbles_per_row * 2:
            self.logger.warning("Insufficient bubbles for clustering")
            return []
        
        y_positions = np.array([b['center'][1] for b in bubbles])
        x_positions = np.array([b['center'][0] for b in bubbles])
        
        # Strategy 1: K-means clustering
        try:
            estimated_rows = min(25, len(bubbles) // self.min_bubbles_per_row)
            if estimated_rows >= 2:
                kmeans = KMeans(n_clusters=estimated_rows, random_state=42, n_init=10)
                y_clusters = kmeans.fit_predict(y_positions.reshape(-1, 1))
                row_centers = sorted(kmeans.cluster_centers_.flatten())
                
                # Create rows from clusters
                cluster_rows = [[] for _ in row_centers]
                for i, bubble in enumerate(bubbles):
                    cluster_idx = y_clusters[i]
                    cluster_rows[cluster_idx].append(bubble)
                
                # Filter valid cluster rows
                valid_cluster_rows = [
                    row for row in cluster_rows 
                    if self.min_bubbles_per_row <= len(row) <= self.max_bubbles_per_row
                ]
                
                self.logger.debug(f"K-means clustering: {len(valid_cluster_rows)} valid rows")
                
                if len(valid_cluster_rows) >= 5:  # Sufficient rows found
                    return valid_cluster_rows
        except Exception as e:
            self.logger.warning(f"K-means clustering failed: {e}")
        
        # Strategy 2: Density-based grouping
        try:
            return self._density_based_row_grouping(bubbles, debug_dir)
        except Exception as e:
            self.logger.warning(f"Density-based grouping failed: {e}")
        
        # Strategy 3: Simple spacing-based grouping (fallback)
        return self._simple_enhanced_row_grouping(bubbles)
    
    def _density_based_row_grouping(self, bubbles: List[Dict], debug_dir: str = None) -> List[List[Dict]]:
        """
        Group bubbles into rows using density-based approach
        """
        bubbles_sorted = sorted(bubbles, key=lambda b: b['center'][1])
        y_positions = [b['center'][1] for b in bubbles_sorted]
        
        # Calculate adaptive row spacing
        y_diffs = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        y_diffs = [d for d in y_diffs if d > 5]  # Filter out very small differences
        
        if not y_diffs:
            return self._simple_enhanced_row_grouping(bubbles_sorted)
        
        # Use percentiles for adaptive thresholds
        low_threshold = np.percentile(y_diffs, 25)
        high_threshold = np.percentile(y_diffs, 75)
        row_spacing = np.median(y_diffs)
        row_tolerance = max(15, row_spacing * 0.5)
        
        rows = []
        current_row = []
        current_y = y_positions[0]
        
        for i, bubble in enumerate(bubbles_sorted):
            bubble_y = bubble['center'][1]
            y_diff = abs(bubble_y - current_y)
            
            # Determine if this starts a new row
            if y_diff > row_tolerance and len(current_row) >= self.min_bubbles_per_row:
                # End current row
                rows.append(current_row)
                current_row = [bubble]
                current_y = bubble_y
            else:
                # Add to current row
                current_row.append(bubble)
                # Update current_y to be the median of current row
                if len(current_row) > 1:
                    current_y = np.median([b['center'][1] for b in current_row])
        
        # Add final row if valid
        if len(current_row) >= self.min_bubbles_per_row:
            rows.append(current_row)
        
        # Merge close rows
        rows = self._merge_close_rows(rows, row_tolerance * 0.8)
        
        self.logger.debug(f"Density-based grouping: {len(rows)} rows from {len(bubbles)} bubbles")
        return rows
    
    def _merge_close_rows(self, rows: List[List[Dict]], max_distance: float) -> List[List[Dict]]:
        """
        Merge rows that are too close together
        """
        if len(rows) < 2:
            return rows
        
        merged_rows = [rows[0]]
        for row in rows[1:]:
            last_row_y = np.median([b['center'][1] for b in merged_rows[-1]])
            current_row_y = np.median([b['center'][1] for b in row])
            distance = abs(current_row_y - last_row_y)
            
            if distance < max_distance:
                # Merge with last row
                merged_rows[-1].extend(row)
                # Sort merged row by x position
                merged_rows[-1].sort(key=lambda b: b['center'][0])
            else:
                merged_rows.append(row)
        
        return merged_rows
    
    def _simple_enhanced_row_grouping(self, bubbles_sorted: List[Dict]) -> List[List[Dict]]:
        """
        Enhanced simple row grouping with better spacing detection
        """
        if not bubbles_sorted:
            return []
        
        rows = []
        current_row = [bubbles_sorted[0]]
        current_y = bubbles_sorted[0]['center'][1]
        
        # Calculate dynamic thresholds
        y_positions = [b['center'][1] for b in bubbles_sorted]
        y_diffs = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        y_diffs = sorted([d for d in y_diffs if d > 5])
        
        if len(y_diffs) > 0:
            # Use 75th percentile as row break threshold
            row_break_threshold = np.percentile(y_diffs, 75)
            within_row_threshold = np.percentile(y_diffs, 25)
        else:
            row_break_threshold = 40
            within_row_threshold = 15
        
        for i in range(1, len(bubbles_sorted)):
            bubble = bubbles_sorted[i]
            y_diff = bubble['center'][1] - current_y
            
            if y_diff > row_break_threshold and len(current_row) >= self.min_bubbles_per_row:
                # End current row and start new one
                rows.append(current_row)
                current_row = [bubble]
                current_y = bubble['center'][1]
            elif y_diff <= within_row_threshold:
                # Definitely same row
                current_row.append(bubble)
                # Update current_y to median
                current_y = np.median([b['center'][1] for b in current_row])
            else:
                # Ambiguous - check x-position clustering
                if self._should_start_new_row(current_row, bubble, row_break_threshold):
                    rows.append(current_row)
                    current_row = [bubble]
                    current_y = bubble['center'][1]
                else:
                    current_row.append(bubble)
                    current_y = np.median([b['center'][1] for b in current_row])
        
        # Add final row if valid
        if len(current_row) >= self.min_bubbles_per_row:
            rows.append(current_row)
        
        return rows
    
    def _should_start_new_row(self, current_row: List[Dict], new_bubble: Dict, 
                            row_break_threshold: float) -> bool:
        """
        Determine if new bubble should start a new row based on spatial analysis
        """
        try:
            current_y_range = max(b['center'][1] for b in current_row) - min(b['center'][1] for b in current_row)
            new_y = new_bubble['center'][1]
            current_y_center = np.median([b['center'][1] for b in current_row])
            
            # If new bubble is significantly separated vertically
            y_separation = abs(new_y - current_y_center)
            if y_separation > row_break_threshold * 0.7:
                return True
            
            # Check horizontal alignment
            current_x_positions = [b['center'][0] for b in current_row]
            new_x = new_bubble['center'][0]
            
            # If new bubble's x-position doesn't align with existing row
            x_deviation = min(abs(new_x - x_pos) for x_pos in current_x_positions)
            if x_deviation > 50:  # Significant horizontal separation
                return True
            
            return False
            
        except Exception:
            return True  # Default to new row on error
    
    def advanced_grid_formation(self, rows: List[List[Dict]]) -> Dict[int, Dict[int, Dict]]:
        """
        Advanced grid formation with column detection and gap filling
        """
        if not rows:
            return {}
        
        grid = {}
        question_number = 1
        
        for row_idx, row_bubbles in enumerate(rows):
            if question_number > self.total_questions:
                break
            
            # Sort row by x-position
            row_bubbles.sort(key=lambda b: b['center'][0])
            
            # Advanced column detection
            question_data = self._detect_columns_in_row(row_bubbles)
            
            if len(question_data) >= 2:  # Valid row with at least 2 options
                grid[question_number] = question_data
                question_number += 1
        
        # Apply grid repair and enhancement
        enhanced_grid = self._enhance_grid_structure(grid)
        
        self.logger.info(f"Grid formation: {len(grid)} initial questions -> {len(enhanced_grid)} enhanced questions")
        return enhanced_grid
    
    def _detect_columns_in_row(self, row_bubbles: List[Dict]) -> Dict[int, Dict]:
        """
        Detect column positions within a row using clustering
        """
        if len(row_bubbles) < 2:
            return {}
        
        x_positions = np.array([b['center'][0] for b in row_bubbles])
        
        # Use K-means for column clustering
        n_columns = min(5, len(row_bubbles))  # Don't create more columns than bubbles
        if n_columns >= 2:
            try:
                kmeans = KMeans(n_clusters=n_columns, random_state=42, n_init=10)
                x_clusters = kmeans.fit_predict(x_positions.reshape(-1, 1))
                col_centers = sorted(kmeans.cluster_centers_.flatten())
                
                # Assign bubbles to columns
                col_assignments = {}
                for col_idx, col_center in enumerate(col_centers):
                    col_bubbles = [b for i, b in enumerate(row_bubbles) if x_clusters[i] == col_idx]
                    if col_bubbles:
                        # Select the bubble closest to column center
                        best_bubble = min(col_bubbles, key=lambda b: abs(b['center'][0] - col_center))
                        col_assignments[col_idx] = best_bubble
                
                return col_assignments
                
            except Exception as e:
                self.logger.debug(f"Column clustering failed: {e}")
        
        # Fallback: simple spacing-based assignment
        return self._simple_column_assignment(row_bubbles)
    
    def _simple_column_assignment(self, row_bubbles: List[Dict]) -> Dict[int, Dict]:
        """
        Simple column assignment based on x-position spacing
        """
        if not row_bubbles:
            return {}
        
        # Calculate spacing
        x_positions = sorted([b['center'][0] for b in row_bubbles])
        if len(x_positions) < 2:
            return {0: row_bubbles[0]}
        
        spacing = (x_positions[-1] - x_positions[0]) / (len(x_positions) - 1)
        
        assignments = {}
        for i, bubble in enumerate(row_bubbles):
            # Normalize x-position to column index
            col_index = round((bubble['center'][0] - x_positions[0]) / spacing)
            col_index = min(col_index, self.options_per_question - 1)
            assignments[col_index] = bubble
        
        return assignments
    
    def _enhance_grid_structure(self, grid: Dict[int, Dict[int, Dict]]) -> Dict[int, Dict[int, Dict]]:
        """
        Enhance grid structure by filling gaps and correcting alignments
        """
        if not grid:
            return grid
        
        enhanced_grid = grid.copy()
        total_enhancements = 0
        
        # Phase 1: Fill missing questions using interpolation (require at least 2 real neighbors)
        for q_num in range(1, self.total_questions + 1):
            if q_num not in enhanced_grid and len(enhanced_grid) < self.total_questions:
                # Find nearest existing questions
                nearby_questions = [qn for qn in enhanced_grid.keys() if abs(qn - q_num) <= 3]
                # Only interpolate if at least 2 real neighbors
                if len(nearby_questions) >= 2:
                    # Interpolate from nearest question
                    nearest_q = min(nearby_questions, key=lambda x: abs(x - q_num))
                    if nearest_q in enhanced_grid:
                        # Create synthetic question by copying and adjusting y-positions
                        synthetic_data = {}
                        y_offset = (q_num - nearest_q) * 35  # Approximate row height
                        for opt_num, bubble in enhanced_grid[nearest_q].items():
                            synthetic_bubble = bubble.copy()
                            synthetic_bubble['center'] = (
                                synthetic_bubble['center'][0],
                                synthetic_bubble['center'][1] + y_offset
                            )
                            synthetic_bubble['bbox'] = (
                                synthetic_bubble['bbox'][0],
                                synthetic_bubble['bbox'][1] + y_offset,
                                synthetic_bubble['bbox'][2],
                                synthetic_bubble['bbox'][3]
                            )
                            synthetic_bubble['synthetic'] = True
                            synthetic_data[opt_num] = synthetic_bubble
                        enhanced_grid[q_num] = synthetic_data
                        total_enhancements += 1
                        self.logger.debug(f"Enhanced: Added synthetic Q{q_num}")
        
        # Phase 2: Fill missing options within questions (require at least 2 real options)
        for q_num, q_data in list(enhanced_grid.items()):
            available_options = sorted(q_data.keys())
            # Only interpolate if at least 2 real options
            real_options = [opt for opt in available_options if not q_data[opt].get('synthetic', False)]
            if len(real_options) >= 2:
                # Calculate expected column positions
                x_positions = [q_data[opt]['center'][0] for opt in real_options]
                avg_y = np.median([q_data[opt]['center'][1] for opt in real_options])
                spacing = (max(x_positions) - min(x_positions)) / (len(x_positions) - 1) if len(x_positions) > 1 else 40

                # Fill standard option positions
                for expected_opt in range(self.options_per_question):
                    if expected_opt not in q_data:
                        # Estimate position
                        estimated_x = min(x_positions) + expected_opt * spacing

                        # Only fill if position is reasonable
                        if abs(estimated_x - min(x_positions)) <= spacing * 1.5:
                            # Find closest existing real option to base the synthetic one on
                            if real_options:
                                closest_opt = min(real_options, key=lambda opt: abs(opt - expected_opt))
                                base_bubble = q_data[closest_opt]

                                synthetic_bubble = base_bubble.copy()
                                synthetic_bubble['center'] = (int(estimated_x), int(avg_y))
                                synthetic_bubble['bbox'] = (
                                    int(estimated_x - 12), int(avg_y - 12), 24, 24
                                )
                                synthetic_bubble['synthetic'] = True
                                synthetic_bubble['source_option'] = closest_opt

                                q_data[expected_opt] = synthetic_bubble
                                total_enhancements += 1
                                self.logger.debug(f"Enhanced: Filled Q{q_num} Opt{expected_opt}")
        
        if total_enhancements > 0:
            self.logger.info(f"Grid enhancement completed: {total_enhancements} improvements applied")
        
        return enhanced_grid
    
    def extract_bubble_regions_enhanced(self, image: np.ndarray, grid: Dict[int, Dict[int, Dict]], 
                                      debug_dir: str = None) -> Dict[int, Dict[int, dict]]:
        """
        Enhanced bubble region extraction with better boundary handling
        """
        try:
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()

            bubble_regions = {}
            padding = 10  # Generous padding
            successful_extractions = 0
            total_attempts = 0

            # Compute main grid bounding box (min/max of all real bubble centers)
            all_centers = [bubble_info['center'] for q in grid.values() for bubble_info in q.values() if not bubble_info.get('synthetic', False)]
            if not all_centers:
                self.logger.error("No real bubbles found for grid bounding box.")
                return {}
            min_x = min(c[0] for c in all_centers) - 20
            max_x = max(c[0] for c in all_centers) + 20
            min_y = min(c[1] for c in all_centers) - 20
            max_y = max(c[1] for c in all_centers) + 20

            for question_num, question_data in grid.items():
                bubble_regions[question_num] = {}
                for option_num, bubble_info in question_data.items():
                    # Only use real (non-synthetic) bubbles for overlays/classification
                    if bubble_info.get('synthetic', False):
                        continue
                    x, y = bubble_info['center']
                    # Filter: Only keep bubbles inside main grid bounding box
                    if not (min_x <= x <= max_x and min_y <= y <= max_y):
                        continue
                    total_attempts += 1
                    try:
                        bx, by, w, h = bubble_info['bbox']
                        x_start = max(0, min(bx - padding, gray_image.shape[1] - 1))
                        y_start = max(0, min(by - padding, gray_image.shape[0] - 1))
                        x_end = min(gray_image.shape[1], bx + w + padding)
                        y_end = min(gray_image.shape[0], by + h + padding)
                        min_size = 10
                        if x_end - x_start < min_size:
                            x_end = min(gray_image.shape[1], x_start + min_size)
                        if y_end - y_start < min_size:
                            y_end = min(gray_image.shape[0], y_start + min_size)
                        bubble_img = gray_image[y_start:y_end, x_start:x_end].copy()
                        # Only keep if bubble is actually filled (dark)
                        if bubble_img.size > 0 and bubble_img.shape[0] > 0 and bubble_img.shape[1] > 0:
                            mean_intensity = np.mean(bubble_img)
                            # Threshold: only keep if dark enough (filled)
                            if mean_intensity < 140:  # You can tune this threshold
                                bubble_region = {
                                    'image': bubble_img,
                                    'bbox': (x_start, y_start, x_end - x_start, y_end - y_start),
                                    'original_bbox': (bx, by, w, h),
                                    'center': bubble_info['center'],
                                    'area': bubble_info.get('area', w * h),
                                    'aspect_ratio': bubble_info.get('aspect_ratio', w / h if h > 0 else 1.0),
                                    'synthetic': False
                                }
                                bubble_regions[question_num][option_num] = bubble_region
                                successful_extractions += 1
                    except Exception as e:
                        self.logger.warning(f"Error extracting Q{question_num} Opt{option_num}: {e}")
                        continue

            self.logger.info(f"Enhanced extraction: {successful_extractions}/{total_attempts} regions extracted (filtered to grid and filled only)")

            # Enhanced debug output
            if self.debug and debug_dir:
                self._save_enhanced_debug_output(bubble_regions, debug_dir)

            return bubble_regions

        except Exception as e:
            self.logger.error(f"Enhanced extraction error: {e}")
            return {}
    
    def _save_enhanced_debug_output(self, bubble_regions: Dict, debug_dir: str):
        """
        Save enhanced debug output including synthetic bubble visualization
        """
        try:
            sample_dir = os.path.join(debug_dir, "enhanced_bubbles")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Categorize bubbles
            real_bubbles = 0
            synthetic_bubbles = 0
            saved_count = 0
            
            for q_num, q_data in list(bubble_regions.items())[:8]:  # First 8 questions
                for opt_num, bubble_data in q_data.items():
                    if isinstance(bubble_data, dict) and 'image' in bubble_data:
                        bubble_img = bubble_data['image']
                        is_synthetic = bubble_data.get('synthetic', False)
                        
                        if is_synthetic:
                            synthetic_bubbles += 1
                            filename = f"synthetic_Q{q_num:02d}_{chr(65+opt_num)}.png"
                        else:
                            real_bubbles += 1
                            filename = f"real_Q{q_num:02d}_{chr(65+opt_num)}.png"
                        
                        filepath = os.path.join(sample_dir, filename)
                        
                        if bubble_img.size > 0:
                            # Add border color coding
                            if len(bubble_img.shape) == 3:
                                bordered = cv2.cvtColor(bubble_img, cv2.COLOR_GRAY2BGR)
                            else:
                                bordered = cv2.cvtColor(bubble_img, cv2.COLOR_GRAY2BGR)
                            
                            border_color = (0, 255, 0) if not is_synthetic else (0, 0, 255)
                            h, w = bordered.shape[:2]
                            cv2.rectangle(bordered, (0, 0), (w-1, h-1), border_color, 2)
                            
                            # Resize for display
                            display_size = (80, 80)
                            resized = cv2.resize(bordered, display_size, interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(filepath, resized)
                            saved_count += 1
            
            # Create summary
            summary = {
                'real_bubbles': real_bubbles,
                'synthetic_bubbles': synthetic_bubbles,
                'total_questions': len(bubble_regions),
                'total_extracted': sum(len(q_data) for q_data in bubble_regions.values())
            }
            
            with open(os.path.join(sample_dir, "extraction_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.debug(f"Enhanced debug output: {saved_count} images, {real_bubbles} real, {synthetic_bubbles} synthetic")
            
        except Exception as e:
            self.logger.error(f"Enhanced debug output error: {e}")
    
    def enhanced_validation(self, grid: Dict[int, Dict[int, Dict]]) -> Tuple[bool, List[str]]:
        """
        Enhanced validation with more detailed analysis and recommendations
        """
        issues = []
        warnings = []
        
        try:
            questions_detected = len(grid)
            total_bubbles = sum(len(q_data) for q_data in grid.values())
            
            # Question count validation with recommendations
            expected_min = int(self.total_questions * 0.7)  # 70% tolerance
            if questions_detected < expected_min:
                issues.append(f"Low question detection: {questions_detected}/{self.total_questions} "
                             f"(expected minimum: {expected_min})")
                if questions_detected < 20:
                    issues.append("CRITICAL: Very few questions detected. Check:")
                    issues.append("  - Image quality and contrast")
                    issues.append("  - Preprocessing parameters")
                    issues.append("  - Sheet alignment and lighting")
                elif questions_detected < 50:
                    warnings.append("Consider adjusting detection sensitivity")
            else:
                warnings.append(f"Good coverage: {questions_detected} questions detected")
            
            # Option completeness analysis
            complete_questions = 0
            partial_questions = 0
            empty_questions = 0
            
            for q_num in sorted(grid.keys()):
                options_count = len(grid[q_num])
                if options_count == self.options_per_question:
                    complete_questions += 1
                elif options_count >= 2:
                    partial_questions += 1
                else:
                    empty_questions += 1
            
            completeness_rate = (complete_questions + partial_questions) / questions_detected if questions_detected > 0 else 0
            if completeness_rate < 0.7:
                issues.append(f"Low option completeness: {completeness_rate:.1%} questions have 2+ options")
            else:
                warnings.append(f"Good option coverage: {completeness_rate:.1%} questions complete")
            
            # Row spacing analysis
            if questions_detected >= 3:
                row_y_positions = []
                for q_num in sorted(grid.keys()):
                    if grid[q_num]:
                        y_coords = [bubble['center'][1] for bubble in grid[q_num].values()]
                        avg_y = np.mean(y_coords)
                        row_y_positions.append(avg_y)
                
                if len(row_y_positions) >= 2:
                    y_diffs = np.diff(row_y_positions)
                    avg_spacing = np.mean(y_diffs)
                    std_spacing = np.std(y_diffs)
                    
                    spacing_cv = std_spacing / avg_spacing if avg_spacing > 0 else 0
                    if spacing_cv > 0.5:
                        issues.append(f"Irregular row spacing: CV={spacing_cv:.2f} (avg={avg_spacing:.1f}px)")
                    else:
                        warnings.append(f"Consistent spacing: CV={spacing_cv:.2f}")
            
            # Synthetic bubble analysis
            synthetic_count = 0
            total_count = 0
            for q_data in grid.values():
                for bubble in q_data.values():
                    total_count += 1
                    if bubble.get('synthetic', False):
                        synthetic_count += 1
            
            synthetic_ratio = synthetic_count / total_count if total_count > 0 else 0
            if synthetic_ratio > 0.3:
                issues.append(f"High synthetic ratio: {synthetic_ratio:.1%} bubbles interpolated")
            elif synthetic_ratio > 0:
                warnings.append(f"Grid enhancement used: {synthetic_ratio:.1%} synthetic bubbles")
            
            # Overall assessment
            critical_issues = len([issue for issue in issues if 'CRITICAL' in issue or issue.startswith('Low question')])
            major_issues = len(issues) - critical_issues
            
            if critical_issues == 0 and major_issues <= 1:
                validation_status = True
                issues.insert(0, f"✓ VALID: {questions_detected} questions, {total_bubbles} bubbles detected")
            else:
                validation_status = False
                issues.insert(0, f"⚠️  PARTIAL: {questions_detected} questions ({questions_detected/self.total_questions:.1%} coverage)")
            
            # Add warnings if any
            all_messages = issues + warnings
            self.logger.info(f"Enhanced validation: {'PASS' if validation_status else 'WARNINGS'} - {len(issues)} issues, {len(warnings)} warnings")
            
            return validation_status, all_messages
            
        except Exception as e:
            self.logger.error(f"Enhanced validation error: {e}")
            return False, [f"Validation failed: {str(e)}"]
    
    def extract_grid_pipeline_enhanced(self, processed_image: np.ndarray, debug_dir: str = None) -> Tuple[Dict, Dict, List[str]]:
        """
        Enhanced complete pipeline with comprehensive error handling and recovery
        """
        try:
            self.logger.info("Starting ENHANCED grid extraction pipeline")
            
            # Phase 1: Multi-pass bubble detection
            self.logger.info("Phase 1: Multi-pass bubble detection")
            bubble_candidates = self.multi_pass_bubble_detection(processed_image, debug_dir)
            
            if len(bubble_candidates) < 10:
                issues = ["CRITICAL: Insufficient bubble candidates detected"]
                self.logger.error("Pipeline failed: too few bubbles detected")
                return {}, {}, issues
            
            # Phase 2: Enhanced row clustering
            self.logger.info(f"Phase 2: Enhanced row clustering ({len(bubble_candidates)} candidates)")
            detected_rows = self.enhanced_row_clustering(bubble_candidates, debug_dir)
            
            if len(detected_rows) < 3:
                issues = ["CRITICAL: Insufficient rows detected after clustering"]
                self.logger.error("Pipeline failed: too few rows detected")
                return {}, {}, issues
            
            # Phase 3: Advanced grid formation
            self.logger.info(f"Phase 3: Advanced grid formation ({len(detected_rows)} rows)")
            grid_structure = self.advanced_grid_formation(detected_rows)
            
            if not grid_structure:
                issues = ["CRITICAL: Failed to form valid grid structure"]
                self.logger.error("Pipeline failed: no valid grid formed")
                return {}, {}, issues
            
            # Phase 4: Enhanced bubble extraction
            self.logger.info(f"Phase 4: Enhanced bubble extraction ({len(grid_structure)} questions)")
            bubble_regions = self.extract_bubble_regions_enhanced(processed_image, grid_structure, debug_dir)
            
            # Phase 5: Enhanced validation and reporting
            self.logger.info("Phase 5: Enhanced validation")
            is_valid, validation_messages = self.enhanced_validation(grid_structure)
            
            # Final statistics
            questions_count = len(grid_structure)
            total_bubbles = sum(len(q_data) for q_data in bubble_regions.values())
            
            self.logger.info(f"ENHANCED PIPELINE COMPLETED:")
            self.logger.info(f"  ✓ {questions_count} questions detected")
            self.logger.info(f"  ✓ {total_bubbles} bubble regions extracted")
            self.logger.info(f"  ✓ {len(bubble_candidates)} total candidates processed")
            self.logger.info(f"  ✓ Validation: {'PASS' if is_valid else 'WARNINGS'}")
            
            if not is_valid:
                for msg in validation_messages:
                    if msg.startswith('⚠️') or msg.startswith('CRITICAL'):
                        self.logger.warning(msg)
            
            return grid_structure, bubble_regions, validation_messages
            
        except Exception as e:
            error_msg = f"Enhanced pipeline error: {e}"
            self.logger.error(error_msg)
            return {}, {}, [error_msg]
    
    # Legacy methods for backward compatibility
    def find_bubble_contours(self, binary_image: np.ndarray, debug_dir: str = None) -> List[Dict]:
        """Legacy method - use multi_pass_bubble_detection instead"""
        self.logger.warning("Using legacy find_bubble_contours - consider updating to enhanced pipeline")
        return self.multi_pass_bubble_detection(binary_image, debug_dir)
    
    def organize_bubbles_into_grid(self, bubbles: List[Dict], debug_dir: str = None) -> Dict[int, Dict[int, Dict]]:
        """Legacy method - use enhanced_row_clustering + advanced_grid_formation instead"""
        self.logger.warning("Using legacy organize_bubbles_into_grid")
        rows = self.enhanced_row_clustering(bubbles, debug_dir)
        return self.advanced_grid_formation(rows)
    
    def extract_bubble_regions(self, image: np.ndarray, grid: Dict[int, Dict[int, Dict]], 
                             debug_dir: str = None) -> Dict[int, Dict[int, dict]]:
        """Legacy method - use extract_bubble_regions_enhanced instead"""
        self.logger.warning("Using legacy extract_bubble_regions")
        return self.extract_bubble_regions_enhanced(image, grid, debug_dir)
    
    def validate_grid_structure(self, grid: Dict[int, Dict[int, Dict]]) -> Tuple[bool, List[str]]:
        """Legacy method - use enhanced_validation instead"""
        self.logger.warning("Using legacy validate_grid_structure")
        return self.enhanced_validation(grid)
    
    def extract_grid_pipeline(self, processed_image: np.ndarray, debug_dir: str = None) -> Tuple[Dict, Dict, List[str]]:
        """Legacy method - use extract_grid_pipeline_enhanced instead"""
        self.logger.info("Using ENHANCED grid extraction pipeline")
        return self.extract_grid_pipeline_enhanced(processed_image, debug_dir)