# """
# Batch processing script for OMR evaluation
# Orchestrates the complete pipeline from image processing to scoring
# """

# import os
# import logging
# import json
# from typing import List, Dict, Any
# from datetime import datetime
# from pathlib import Path
# import cv2

# from .utils import (
#     setup_logging, load_config, create_directories, get_image_files,
#     extract_set_from_filename, save_results, calculate_accuracy_metrics, load_answer_key
# )
# from .preprocess import ImagePreprocessor
# from .detect_and_warp import SheetDetector
# from .extract_grid import GridExtractor
# from .classify_marks import BubbleClassifier
# from .score import OMRScorer

# class OMRBatchProcessor:
#     """
#     Main class for batch processing OMR sheets
#     """
    
#     def __init__(self, config_path: str = "configs/config.yaml", debug: bool = False):
#         """
#         Initialize batch processor
        
#         Args:
#             config_path: Path to configuration file
#             debug: Enable debug mode
#         """
#         self.config = load_config(config_path)
#         self.debug = debug
#         self.logger = setup_logging(
#             self.config['debug']['log_level'],
#             "logs/batch_processing.log" if not debug else None
#         )
        
#         # Initialize processing components
#         self.preprocessor = ImagePreprocessor(self.config, debug)
#         self.detector = SheetDetector(self.config, debug)
#         self.extractor = GridExtractor(self.config, debug)
#         self.classifier = BubbleClassifier(self.config, debug)
        
#         # Load answer keys
#         self.answer_keys = {}
#         self.scorer = None
#         self._load_answer_keys()
        
#         # Create output directories
#         self.output_dir = self.config['paths']['output_folder']
#         self.debug_dir = os.path.join(self.output_dir, 'debug') if debug else None
#         self._setup_directories()
    
#     def _load_answer_keys(self):
#         """Load answer keys for scoring"""
#         try:
#             answer_key_path = self.config['paths']['answer_keys']
#             if os.path.exists(answer_key_path):
#                 self.answer_keys = load_answer_key(answer_key_path)
#                 self.scorer = OMRScorer(self.config, self.answer_keys, self.debug)
#                 self.logger.info(f"Loaded answer keys for sets: {list(self.answer_keys.keys())}")
#             else:
#                 self.logger.warning(f"Answer key file not found: {answer_key_path}")
#         except Exception as e:
#             self.logger.error(f"Error loading answer keys: {e}")
    
#     def _setup_directories(self):
#         """Create necessary directories"""
#         directories = [
#             self.output_dir,
#             os.path.join(self.output_dir, 'processed_images'),
#             os.path.join(self.output_dir, 'results'),
#             os.path.join(self.output_dir, 'reports'),
#             self.config['paths']['logs_folder']
#         ]
        
#         if self.debug_dir:
#             directories.extend([
#                 self.debug_dir,
#                 os.path.join(self.debug_dir, 'preprocessing'),
#                 os.path.join(self.debug_dir, 'detection'),
#                 os.path.join(self.debug_dir, 'grid_extraction'),
#                 os.path.join(self.debug_dir, 'classification')
#             ])
        
#         create_directories(directories)
    
#     def process_single_image(self, image_path: str, image_index: int) -> Dict[str, Any]:
#         """
#         Process a single OMR sheet image through the complete pipeline
        
#         Args:
#             image_path: Path to the image file
#             image_index: Index of the image in batch
            
#         Returns:
#             Processing result dictionary
#         """
#         start_time = datetime.now()
#         image_name = Path(image_path).stem
        
#         # Extract set information from filename
#         answer_key_set = extract_set_from_filename(image_name)
        
#         # Create debug directory for this image
#         image_debug_dir = None
#         if self.debug_dir:
#             image_debug_dir = os.path.join(self.debug_dir, f"image_{image_index:03d}_{image_name}")
#             os.makedirs(image_debug_dir, exist_ok=True)
        
#         # Save a copy of the original image to debug dir for UI display
#         saved_image_path = None
#         if image_debug_dir:
#             try:
#                 import shutil
#                 saved_image_path = os.path.join(image_debug_dir, f"original_{image_name}.jpg")
#                 shutil.copy(image_path, saved_image_path)
#             except Exception as e:
#                 self.logger.warning(f"Could not save original image for UI: {e}")
#         result = {
#             'image_path': saved_image_path if saved_image_path else image_path,
#             'image_name': image_name,
#             'image_index': image_index,
#             'answer_key_set': answer_key_set,
#             'timestamp': start_time.isoformat(),
#             'success': False,
#             'pipeline_stages': {},
#             'errors': [],
#             'debug_dir': image_debug_dir
#         }
        
#         try:
#             # Stage 1: Preprocessing
#             self.logger.info(f"Processing {image_name} - Stage 1: Preprocessing")
#             try:
#                 original_image, binary_image, prep_metadata = self.preprocessor.preprocess_pipeline(
#                     image_path, 
#                     os.path.join(image_debug_dir, 'preprocessing') if image_debug_dir else None
#                 )
#                 # Save the preprocessed (warped) image for overlay/debug
#                 warped_save_path = None
#                 if image_debug_dir is not None:
#                     warped_save_path = os.path.join(image_debug_dir, f"warped_{image_name}.jpg")
#                     cv2.imwrite(warped_save_path, original_image)
#                 result['pipeline_stages']['preprocessing'] = {
#                     'success': True,
#                     'metadata': prep_metadata,
#                     'warped_image_path': warped_save_path
#                 }
#             except Exception as e:
#                 error_msg = f"Preprocessing failed: {str(e)}"
#                 self.logger.error(error_msg)
#                 result['errors'].append(error_msg)
#                 result['pipeline_stages']['preprocessing'] = {'success': False, 'error': error_msg}
#                 return result
            
#             # Stage 2: Sheet Detection and Warping
#             self.logger.info(f"Processing {image_name} - Stage 2: Sheet Detection")
#             try:
#                 warped_image, detection_success, detection_metadata = self.detector.detect_and_warp(
#                     original_image,
#                     os.path.join(image_debug_dir, 'detection') if image_debug_dir else None
#                 )
#                 result['pipeline_stages']['detection'] = {
#                     'success': detection_success,
#                     'metadata': detection_metadata
#                 }
                
#                 # Use warped image for further processing if detection succeeded, otherwise use preprocessed
#                 processing_image = warped_image if detection_success else original_image
                
#             except Exception as e:
#                 error_msg = f"Sheet detection failed: {str(e)}"
#                 self.logger.warning(error_msg)
#                 result['errors'].append(error_msg)
#                 result['pipeline_stages']['detection'] = {'success': False, 'error': error_msg}
#                 processing_image = original_image  # Continue with original
            
#             # Stage 3: Grid Extraction
#             self.logger.info(f"Processing {image_name} - Stage 3: Grid Extraction")
#             try:
#                 # Re-preprocess the processing image for grid extraction
#                 if detection_success:
#                     enhanced_image = self.preprocessor.enhance_image(processing_image, image_debug_dir)
#                     final_binary = self.preprocessor.adaptive_threshold(enhanced_image, image_debug_dir)
#                     final_binary = self.preprocessor.remove_noise(final_binary, image_debug_dir)
#                 else:
#                     final_binary = binary_image
                
#                 grid_structure, bubble_regions, grid_issues = self.extractor.extract_grid_pipeline(
#                     final_binary,
#                     os.path.join(image_debug_dir, 'grid_extraction') if image_debug_dir else None
#                 )
                
#                 result['pipeline_stages']['grid_extraction'] = {
#                     'success': len(grid_structure) > 0,
#                     'questions_detected': len(grid_structure),
#                     'total_bubbles': sum(len(q_data) for q_data in bubble_regions.values()),
#                     'issues': grid_issues
#                 }
                
#                 if not grid_structure:
#                     error_msg = "Grid extraction failed: No bubbles detected"
#                     result['errors'].append(error_msg)
#                     return result
                
#             except Exception as e:
#                 error_msg = f"Grid extraction failed: {str(e)}"
#                 self.logger.error(error_msg)
#                 result['errors'].append(error_msg)
#                 result['pipeline_stages']['grid_extraction'] = {'success': False, 'error': error_msg}
#                 return result
            
#             # Stage 4: Bubble Classification
#             # Stage 4: Bubble Classification
#             self.logger.info(f"Processing {image_name} - Stage 4: Bubble Classification")
#             try:
#                 # Set the OMR image path and debug directory for overlay generation
#                 if image_debug_dir is not None:
#                     warped_image_path = os.path.join(image_debug_dir, f"warped_{image_name}.jpg")
#                     # Ensure the warped image exists
#                     if os.path.exists(warped_image_path):
#                         self.classifier.config['omr_image'] = warped_image_path
#                     self.classifier.config['debug_dir'] = os.path.join(image_debug_dir, 'classification')
#                     os.makedirs(self.classifier.config['debug_dir'], exist_ok=True)
                
#                 classifications = self.classifier.classify_all_bubbles(
#                     bubble_regions,
#                     os.path.join(image_debug_dir, 'classification') if image_debug_dir else None
#                 )
                
#                 # Check if full sheet overlay was generated
#                 full_overlay_path = None
#                 if 'full_sheet_overlay' in classifications:
#                     full_overlay_path = classifications['full_sheet_overlay']
#                     if full_overlay_path and os.path.exists(full_overlay_path):
#                         result['full_overlay_path'] = full_overlay_path
#                         self.logger.debug(f"Full overlay available at: {full_overlay_path}")
                
#                 marked_answers = self.classifier.get_marked_answers(classifications)
                
#                 # Calculate classification statistics
#                 total_bubbles = sum(len(q_results) for q_results in classifications.values() if isinstance(q_results, dict))
#                 marked_bubbles = sum(1 for q_results in classifications.values() 
#                                     for result in q_results.values() 
#                                     if isinstance(result, dict) and result.get("is_marked", False))
#                 avg_confidence = 0.0
#                 questions_with_answers = 0
                
#                 if total_bubbles > 0:
#                     total_confidence = sum(
#                         result.get("confidence", 0) 
#                         for q_results in classifications.values() 
#                         for result in q_results.values()
#                         if isinstance(result, dict)
#                     )
#                     avg_confidence = total_confidence / total_bubbles
#                     questions_with_answers = len([q for q, ans in marked_answers.items() if ans])
                
#                 result['pipeline_stages']['classification'] = {
#                     'success': True,
#                     'total_bubbles': total_bubbles,
#                     'marked_bubbles': marked_bubbles,
#                     'average_confidence': round(avg_confidence, 3),
#                     'questions_with_answers': questions_with_answers,
#                     'full_overlay_path': full_overlay_path
#                 }
                
#                 # Store classifications and marked answers for detailed view
#                 result['classifications'] = classifications
#                 result['marked_answers'] = marked_answers
                
#             except Exception as e:
#                 error_msg = f"Bubble classification failed: {str(e)}"
#                 self.logger.error(error_msg)
#                 result['errors'].append(error_msg)
#                 result['pipeline_stages']['classification'] = {'success': False, 'error': error_msg}
#                 return result
    
#     def process_batch(self, input_directory: str, file_pattern: str = "*.jpg") -> Dict[str, Any]:
#         """
#         Process a batch of OMR sheet images
        
#         Args:
#             input_directory: Directory containing OMR sheet images
#             file_pattern: File pattern to match (default: "*.jpg")
            
#         Returns:
#             Batch processing results
#         """
#         start_time = datetime.now()
        
#         self.logger.info(f"Starting batch processing from: {input_directory}")
        
#         # Find all image files
#         if os.path.isdir(input_directory):
#             image_files = get_image_files(input_directory)
#         else:
#             self.logger.error(f"Input directory does not exist: {input_directory}")
#             return {'error': f"Input directory does not exist: {input_directory}"}
        
#         if not image_files:
#             self.logger.warning(f"No image files found in: {input_directory}")
#             return {'error': f"No image files found in: {input_directory}"}
        
#         self.logger.info(f"Found {len(image_files)} images to process")
        
#         # Initialize batch results
#         batch_results = {
#             'batch_info': {
#                 'input_directory': input_directory,
#                 'total_images': len(image_files),
#                 'start_time': start_time.isoformat(),
#                 'config_used': self.config
#             },
#             'individual_results': [],
#             'batch_summary': {}
#         }
        
#         # Process each image
#         successful_results = []
#         failed_results = []
        
#         for i, image_path in enumerate(image_files):
#             self.logger.info(f"Processing image {i+1}/{len(image_files)}: {Path(image_path).name}")
            
#             try:
#                 result = self.process_single_image(image_path, i)
#                 batch_results['individual_results'].append(result)
                
#                 if result['success']:
#                     successful_results.append(result)
#                 else:
#                     failed_results.append(result)
                    
#             except Exception as e:
#                 error_msg = f"Failed to process {image_path}: {str(e)}"
#                 self.logger.error(error_msg)
#                 failed_result = {
#                     'image_path': image_path,
#                     'image_index': i,
#                     'success': False,
#                     'errors': [error_msg],
#                     'timestamp': datetime.now().isoformat()
#                 }
#                 batch_results['individual_results'].append(failed_result)
#                 failed_results.append(failed_result)
        
#         # Calculate batch summary
#         end_time = datetime.now()
#         total_processing_time = (end_time - start_time).total_seconds()
        
#         batch_results['batch_summary'] = {
#             'total_processed': len(image_files),
#             'successful': len(successful_results),
#             'failed': len(failed_results),
#             'success_rate': round((len(successful_results) / len(image_files)) * 100, 2),
#             'total_processing_time': total_processing_time,
#             'average_time_per_image': round(total_processing_time / len(image_files), 2),
#             'end_time': end_time.isoformat()
#         }
        
#         # Generate scoring report if applicable
#         if self.scorer and successful_results:
#             scoring_results = [r.get('scoring_result') for r in successful_results if r.get('scoring_result')]
#             if scoring_results:
#                 batch_results['scoring_report'] = self.scorer.generate_score_report(scoring_results)
        
#         # Calculate accuracy metrics
#         batch_results['accuracy_metrics'] = calculate_accuracy_metrics({
#             'students': [r.get('scoring_result', {}) for r in successful_results]
#         })
        
#         self.logger.info(f"Batch processing completed: {len(successful_results)}/{len(image_files)} successful "
#                         f"in {total_processing_time:.2f} seconds")
        
#         return batch_results
    
#     def save_batch_results(self, batch_results: Dict[str, Any], output_prefix: str = "batch_results") -> Dict[str, str]:
#         """
#         Save batch processing results to various formats
        
#         Args:
#             batch_results: Batch processing results
#             output_prefix: Prefix for output files
            
#         Returns:
#             Dictionary of saved file paths
#         """
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         results_dir = os.path.join(self.output_dir, 'results')
#         reports_dir = os.path.join(self.output_dir, 'reports')
        
#         saved_files = {}
        
#         try:
#             # Save complete results as JSON
#             json_path = os.path.join(results_dir, f"{output_prefix}_{timestamp}.json")
#             save_results(batch_results, json_path, "json")
#             saved_files['json'] = json_path
            
#             # Save scoring results as CSV if available
#             if 'individual_results' in batch_results:
#                 scoring_results = []
#                 for result in batch_results['individual_results']:
#                     if result.get('success') and 'scoring_result' in result:
#                         scoring_result = result['scoring_result'].copy()
#                         scoring_result['image_name'] = result['image_name']
#                         scoring_result['image_path'] = result['image_path']
#                         scoring_results.append(scoring_result)
                
#                 if scoring_results and self.scorer:
#                     csv_path = os.path.join(results_dir, f"{output_prefix}_scores_{timestamp}.csv")
#                     if self.scorer.export_results_to_csv(scoring_results, csv_path):
#                         saved_files['csv'] = csv_path
            
#             # Save summary report
#             if 'batch_summary' in batch_results:
#                 summary_path = os.path.join(reports_dir, f"{output_prefix}_summary_{timestamp}.json")
#                 summary_data = {
#                     'batch_summary': batch_results['batch_summary'],
#                     'scoring_report': batch_results.get('scoring_report', {}),
#                     'accuracy_metrics': batch_results.get('accuracy_metrics', {})
#                 }
#                 save_results(summary_data, summary_path, "json")
#                 saved_files['summary'] = summary_path
            
#             self.logger.info(f"Batch results saved to: {saved_files}")
            
#         except Exception as e:
#             self.logger.error(f"Error saving batch results: {e}")
        
#         return saved_files

# def main():
#     """
#     Main function for running batch processing
#     """
#     import argparse
    
#     parser = argparse.ArgumentParser(description="OMR Batch Processing")
#     parser.add_argument("input_dir", help="Directory containing OMR sheet images")
#     parser.add_argument("--config", default="configs/config.yaml", help="Configuration file path")
#     parser.add_argument("--debug", action="store_true", help="Enable debug mode")
#     parser.add_argument("--output-prefix", default="batch_results", help="Output file prefix")
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.input_dir):
#         print(f"Error: Input directory does not exist: {args.input_dir}")
#         return
    
#     try:
#         # Initialize processor
#         processor = OMRBatchProcessor(args.config, args.debug)
        
#         # Process batch
#         results = processor.process_batch(args.input_dir)
        
#         if 'error' in results:
#             print(f"Batch processing failed: {results['error']}")
#             return
        
#         # Print summary
#         summary = results['batch_summary']
#         print(f"\n{'='*50}")
#         print(f"BATCH PROCESSING COMPLETED")
#         print(f"{'='*50}")
#         print(f"Total images processed: {summary['total_processed']}")
#         print(f"Successful: {summary['successful']}")
#         print(f"Failed: {summary['failed']}")
#         print(f"Success rate: {summary['success_rate']}%")
#         print(f"Total processing time: {summary['total_processing_time']:.2f} seconds")
#         print(f"Average time per image: {summary['average_time_per_image']:.2f} seconds")
        
#         # Print scoring summary if available
#         if 'scoring_report' in results:
#             scoring = results['scoring_report']['overall_statistics']
#             print(f"\nSCORING SUMMARY:")
#             print(f"Average score: {scoring['average_score']}")
#             print(f"Average percentage: {scoring['average_percentage']:.2f}%")
#             print(f"Highest score: {scoring['highest_score']}")
#             print(f"Lowest score: {scoring['lowest_score']}")
        
#         # Save results
#         saved_files = processor.save_batch_results(results, args.output_prefix)
#         print(f"\nRESULTS SAVED TO:")
#         for file_type, file_path in saved_files.items():
#             print(f"  {file_type.upper()}: {file_path}")
        
#         # Print failed images if any
#         if summary['failed'] > 0:
#             print(f"\nFAILED IMAGES:")
#             for result in results['individual_results']:
#                 if not result['success']:
#                     print(f"  {result['image_name']}: {', '.join(result.get('errors', ['Unknown error']))}")
        
#     except Exception as e:
#         print(f"Error in batch processing: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()








"""
Batch processing script for OMR evaluation
Orchestrates the complete pipeline from image processing to scoring
"""

import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import cv2

from .utils import (
    setup_logging, load_config, create_directories, get_image_files,
    extract_set_from_filename, save_results, calculate_accuracy_metrics, load_answer_key
)
from .preprocess import ImagePreprocessor
from .detect_and_warp import SheetDetector
from .extract_grid import GridExtractor
from .classify_marks import BubbleClassifier
from .score import OMRScorer

class OMRBatchProcessor:
    """
    Main class for batch processing OMR sheets
    """
    
    def __init__(self, config_path: str = "configs/config.yaml", debug: bool = False):
        """
        Initialize batch processor
        
        Args:
            config_path: Path to configuration file
            debug: Enable debug mode
        """
        self.config = load_config(config_path)
        self.debug = debug
        self.logger = setup_logging(
            self.config['debug']['log_level'],
            "logs/batch_processing.log" if not debug else None
        )
        
        # Initialize processing components
        self.preprocessor = ImagePreprocessor(self.config, debug)
        self.detector = SheetDetector(self.config, debug)
        self.extractor = GridExtractor(self.config, debug)
        self.classifier = BubbleClassifier(self.config, debug)
        
        # Load answer keys
        self.answer_keys = {}
        self.scorer = None
        self._load_answer_keys()
        
        # Create output directories
        self.output_dir = self.config['paths']['output_folder']
        self.debug_dir = os.path.join(self.output_dir, 'debug') if debug else None
        self._setup_directories()
    
    def _load_answer_keys(self):
        """Load answer keys for scoring"""
        try:
            answer_key_path = self.config['paths']['answer_keys']
            if os.path.exists(answer_key_path):
                self.answer_keys = load_answer_key(answer_key_path)
                self.scorer = OMRScorer(self.config, self.answer_keys, self.debug)
                self.logger.info(f"Loaded answer keys for sets: {list(self.answer_keys.keys())}")
            else:
                self.logger.warning(f"Answer key file not found: {answer_key_path}")
        except Exception as e:
            self.logger.error(f"Error loading answer keys: {e}")
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'processed_images'),
            os.path.join(self.output_dir, 'results'),
            os.path.join(self.output_dir, 'reports'),
            self.config['paths']['logs_folder']
        ]
        
        if self.debug_dir:
            directories.extend([
                self.debug_dir,
                os.path.join(self.debug_dir, 'preprocessing'),
                os.path.join(self.debug_dir, 'detection'),
                os.path.join(self.debug_dir, 'grid_extraction'),
                os.path.join(self.debug_dir, 'classification')
            ])
        
        create_directories(directories)
    
    def process_single_image(self, image_path: str, image_index: int) -> Dict[str, Any]:
        """
        Process a single OMR sheet image through the complete pipeline
        
        Args:
            image_path: Path to the image file
            image_index: Index of the image in batch
            
        Returns:
            Processing result dictionary
        """
        start_time = datetime.now()
        image_name = Path(image_path).stem
        
        # Extract set information from filename
        answer_key_set = extract_set_from_filename(image_name)
        
        # Create debug directory for this image
        image_debug_dir = None
        if self.debug_dir:
            image_debug_dir = os.path.join(self.debug_dir, f"image_{image_index:03d}_{image_name}")
            os.makedirs(image_debug_dir, exist_ok=True)
        
        # Save a copy of the original image to debug dir for UI display
        saved_image_path = None
        if image_debug_dir:
            try:
                import shutil
                saved_image_path = os.path.join(image_debug_dir, f"original_{image_name}.jpg")
                shutil.copy(image_path, saved_image_path)
            except Exception as e:
                self.logger.warning(f"Could not save original image for UI: {e}")
                
        result = {
            'image_path': saved_image_path if saved_image_path else image_path,
            'image_name': image_name,
            'image_index': image_index,
            'answer_key_set': answer_key_set,
            'timestamp': start_time.isoformat(),
            'success': False,
            'pipeline_stages': {},
            'errors': [],
            'debug_dir': image_debug_dir
        }
        
        try:
            # Stage 1: Preprocessing
            self.logger.info(f"Processing {image_name} - Stage 1: Preprocessing")
            try:
                original_image, binary_image, prep_metadata = self.preprocessor.preprocess_pipeline(
                    image_path, 
                    os.path.join(image_debug_dir, 'preprocessing') if image_debug_dir else None
                )
                # Save the preprocessed (warped) image for overlay/debug
                warped_save_path = None
                if image_debug_dir is not None:
                    warped_save_path = os.path.join(image_debug_dir, f"warped_{image_name}.jpg")
                    cv2.imwrite(warped_save_path, original_image)
                result['pipeline_stages']['preprocessing'] = {
                    'success': True,
                    'metadata': prep_metadata,
                    'warped_image_path': warped_save_path
                }
            except Exception as e:
                error_msg = f"Preprocessing failed: {str(e)}"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
                result['pipeline_stages']['preprocessing'] = {'success': False, 'error': error_msg}
                return result
            
            # Stage 2: Sheet Detection and Warping
            self.logger.info(f"Processing {image_name} - Stage 2: Sheet Detection")
            try:
                warped_image, detection_success, detection_metadata = self.detector.detect_and_warp(
                    original_image,
                    os.path.join(image_debug_dir, 'detection') if image_debug_dir else None
                )
                result['pipeline_stages']['detection'] = {
                    'success': detection_success,
                    'metadata': detection_metadata
                }
                
                # Use warped image for further processing if detection succeeded, otherwise use preprocessed
                processing_image = warped_image if detection_success else original_image
                
            except Exception as e:
                error_msg = f"Sheet detection failed: {str(e)}"
                self.logger.warning(error_msg)
                result['errors'].append(error_msg)
                result['pipeline_stages']['detection'] = {'success': False, 'error': error_msg}
                processing_image = original_image  # Continue with original
            
            # Stage 3: Grid Extraction
           # Stage 3: Enhanced Grid Extraction
            self.logger.info(f"Processing {image_name} - Stage 3: Enhanced Grid Extraction")
            try:
                # Re-preprocess the processing image for enhanced grid extraction
                if detection_success:
                    # Use multiple preprocessing approaches for better detection
                    enhanced_image = self.preprocessor.enhance_image(processing_image, image_debug_dir)
                    final_binary = self.preprocessor.adaptive_threshold(enhanced_image, image_debug_dir)
                    final_binary = self.preprocessor.remove_noise(final_binary, image_debug_dir)
                    
                    # Create additional binary version for comparison
                    binary_enhanced = cv2.adaptiveThreshold(
                        enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                        cv2.THRESH_BINARY_INV, 15, 5
                    )
                else:
                    final_binary = binary_image
                    binary_enhanced = final_binary
                
                # Try enhanced pipeline first
                grid_structure, bubble_regions, grid_issues = self.extractor.extract_grid_pipeline(
                    final_binary,
                    os.path.join(image_debug_dir, 'grid_extraction') if image_debug_dir else None
                )
                
                # If enhanced pipeline finds too few questions, try alternative binary
                if len(grid_structure) < 15:
                    self.logger.info(f"Enhanced pipeline found only {len(grid_structure)} questions, trying alternative preprocessing")
                    alt_grid, alt_bubbles, alt_issues = self.extractor.extract_grid_pipeline(
                        binary_enhanced,
                        os.path.join(image_debug_dir, 'grid_extraction_alt') if image_debug_dir else None
                    )
                    
                    if len(alt_grid) > len(grid_structure):
                        grid_structure = alt_grid
                        bubble_regions = alt_bubbles
                        grid_issues = alt_issues
                        self.logger.info(f"Alternative preprocessing successful: {len(grid_structure)} questions")
                
                result['pipeline_stages']['grid_extraction'] = {
                    'success': len(grid_structure) > 0,
                    'questions_detected': len(grid_structure),
                    'total_bubbles': sum(len(q_data) for q_data in bubble_regions.values()),
                    'issues': grid_issues,
                    'detection_method': 'enhanced' if len(grid_structure) > 10 else 'basic'
                }
                
                if not grid_structure:
                    error_msg = "Enhanced grid extraction failed: No bubbles detected"
                    result['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    return result
                
                self.logger.info(f"Grid extraction successful: {len(grid_structure)} questions detected")
                
            except Exception as e:
                error_msg = f"Enhanced grid extraction failed: {str(e)}"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
                result['pipeline_stages']['grid_extraction'] = {'success': False, 'error': error_msg}
                return result
            
            # Stage 4: Bubble Classification
            self.logger.info(f"Processing {image_name} - Stage 4: Bubble Classification")
            try:
                # Set the OMR image path and debug directory for overlay generation
                if image_debug_dir is not None:
                    warped_image_path = os.path.join(image_debug_dir, f"warped_{image_name}.jpg")
                    # Ensure the warped image exists
                    if os.path.exists(warped_image_path):
                        self.classifier.config['omr_image'] = warped_image_path
                    self.classifier.config['debug_dir'] = os.path.join(image_debug_dir, 'classification')
                    os.makedirs(self.classifier.config['debug_dir'], exist_ok=True)
                
                classifications = self.classifier.classify_all_bubbles(
                    bubble_regions,
                    os.path.join(image_debug_dir, 'classification') if image_debug_dir else None
                )
                
                # Check if full sheet overlay was generated
                full_overlay_path = None
                if isinstance(classifications, dict) and 'full_sheet_overlay' in classifications:
                    full_overlay_path = classifications['full_sheet_overlay']
                    if full_overlay_path and os.path.exists(full_overlay_path):
                        result['full_overlay_path'] = full_overlay_path
                        self.logger.debug(f"Full overlay available at: {full_overlay_path}")
                elif 'classifications' in result:
                    # Alternative check
                    class_results = result['classifications']
                    if isinstance(class_results, dict) and 'full_sheet_overlay' in class_results:
                        full_overlay_path = class_results['full_sheet_overlay']
                        if full_overlay_path and os.path.exists(full_overlay_path):
                            result['full_overlay_path'] = full_overlay_path
                
                marked_answers = self.classifier.get_marked_answers(classifications)
                
                # Calculate classification statistics
                total_bubbles = 0
                marked_bubbles = 0
                avg_confidence = 0.0
                questions_with_answers = 0
                
                # Handle both dict and list classifications
                if isinstance(classifications, dict):
                    total_bubbles = sum(len(q_results) for q_results in classifications.values() 
                                      if isinstance(q_results, dict))
                    marked_bubbles = sum(1 for q_results in classifications.values() 
                                       for result in q_results.values() 
                                       if isinstance(result, dict) and result.get("is_marked", False))
                    
                    if total_bubbles > 0:
                        total_confidence = sum(
                            result.get("confidence", 0) 
                            for q_results in classifications.values() 
                            for result in q_results.values()
                            if isinstance(result, dict)
                        )
                        avg_confidence = total_confidence / total_bubbles
                        questions_with_answers = len([q for q, ans in marked_answers.items() if ans])
                else:
                    # Fallback for unexpected classification format
                    total_bubbles = len(bubble_regions) if isinstance(bubble_regions, dict) else 0
                    marked_bubbles = len(marked_answers)
                
                result['pipeline_stages']['classification'] = {
                    'success': True,
                    'total_bubbles': total_bubbles,
                    'marked_bubbles': marked_bubbles,
                    'average_confidence': round(avg_confidence, 3),
                    'questions_with_answers': questions_with_answers,
                    'full_overlay_path': full_overlay_path
                }
                
                # Store classifications and marked answers for detailed view
                result['classifications'] = classifications
                result['marked_answers'] = marked_answers
                
            except Exception as e:
                error_msg = f"Bubble classification failed: {str(e)}"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
                result['pipeline_stages']['classification'] = {'success': False, 'error': error_msg}
                # Continue processing even if classification fails
                marked_answers = {}
            
            # Stage 5: Scoring (if answer keys available)
            if self.scorer and answer_key_set in self.answer_keys:
                self.logger.info(f"Processing {image_name} - Stage 5: Scoring")
                try:
                    student_info = {
                        'image_name': image_name,
                        'answer_key_set': answer_key_set,
                        'processing_timestamp': start_time.isoformat()
                    }
                    
                    scoring_result = self.scorer.score_student_sheet(
                        marked_answers, answer_key_set, student_info
                    )
                    
                    result['pipeline_stages']['scoring'] = {
                        'success': 'error' not in scoring_result,
                        'total_score': scoring_result.get('total_score', 0),
                        'percentage': scoring_result.get('percentage', 0),
                        'subject_scores': scoring_result.get('subject_scores', {}),
                        'confidence': scoring_result.get('confidence', 0)
                    }
                    
                    result['scoring_result'] = scoring_result
                    
                except Exception as e:
                    error_msg = f"Scoring failed: {str(e)}"
                    self.logger.error(error_msg)
                    result['errors'].append(error_msg)
                    result['pipeline_stages']['scoring'] = {'success': False, 'error': error_msg}
            else:
                result['pipeline_stages']['scoring'] = {
                    'success': False, 
                    'error': f"No answer key available for set: {answer_key_set}"
                }
            
            # Mark overall success (classification and scoring are not critical for pipeline success)
            core_stages_success = [
                result['pipeline_stages'].get(stage, {}).get('success', False)
                for stage in ['preprocessing', 'detection', 'grid_extraction']
            ]
            result['success'] = all(core_stages_success)
            
            # Finalize result
            result['marked_answers'] = marked_answers
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Successfully processed {image_name} in {result['processing_time']:.2f} seconds")
            self.logger.info(f"  - Detected {len(grid_structure)} questions, {marked_bubbles} marked bubbles")
            if self.scorer and 'scoring_result' in result:
                score = result['scoring_result'].get('total_score', 0)
                total = result['scoring_result'].get('max_possible_score', 80)
                self.logger.info(f"  - Scored: {score}/{total} ({result['scoring_result'].get('percentage', 0):.1f}%)")
            
        except Exception as e:
            error_msg = f"Unexpected error processing {image_name}: {str(e)}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            result['success'] = False
        
        return result
    
    def process_batch(self, input_directory: str, file_pattern: str = "*.jpg") -> Dict[str, Any]:
        """
        Process a batch of OMR sheet images
        
        Args:
            input_directory: Directory containing OMR sheet images
            file_pattern: File pattern to match (default: "*.jpg")
            
        Returns:
            Batch processing results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Starting batch processing from: {input_directory}")
        
        # Find all image files
        if os.path.isdir(input_directory):
            image_files = get_image_files(input_directory)
        else:
            self.logger.error(f"Input directory does not exist: {input_directory}")
            return {'error': f"Input directory does not exist: {input_directory}"}
        
        if not image_files:
            self.logger.warning(f"No image files found in: {input_directory}")
            return {'error': f"No image files found in: {input_directory}"}
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Initialize batch results
        batch_results = {
            'batch_info': {
                'input_directory': input_directory,
                'total_images': len(image_files),
                'start_time': start_time.isoformat(),
                'config_used': self.config
            },
            'individual_results': [],
            'batch_summary': {}
        }
        
        # Process each image
        successful_results = []
        failed_results = []
        
        for i, image_path in enumerate(image_files):
            self.logger.info(f"Processing image {i+1}/{len(image_files)}: {Path(image_path).name}")
            
            try:
                result = self.process_single_image(image_path, i)
                batch_results['individual_results'].append(result)
                
                if result['success']:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                    
            except Exception as e:
                error_msg = f"Failed to process {image_path}: {str(e)}"
                self.logger.error(error_msg)
                failed_result = {
                    'image_path': image_path,
                    'image_index': i,
                    'success': False,
                    'errors': [error_msg],
                    'timestamp': datetime.now().isoformat()
                }
                batch_results['individual_results'].append(failed_result)
                failed_results.append(failed_result)
        
        # Calculate batch summary
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        
        batch_results['batch_summary'] = {
            'total_processed': len(image_files),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': round((len(successful_results) / len(image_files)) * 100, 2),
            'total_processing_time': total_processing_time,
            'average_time_per_image': round(total_processing_time / len(image_files), 2),
            'end_time': end_time.isoformat()
        }
        
        # Generate scoring report if applicable
        if self.scorer and successful_results:
            scoring_results = [r.get('scoring_result') for r in successful_results if r.get('scoring_result')]
            if scoring_results:
                batch_results['scoring_report'] = self.scorer.generate_score_report(scoring_results)
        
        # Calculate accuracy metrics
        batch_results['accuracy_metrics'] = calculate_accuracy_metrics({
            'students': [r.get('scoring_result', {}) for r in successful_results]
        })
        
        self.logger.info(f"Batch processing completed: {len(successful_results)}/{len(image_files)} successful "
                        f"in {total_processing_time:.2f} seconds")
        
        return batch_results
    
    def save_batch_results(self, batch_results: Dict[str, Any], output_prefix: str = "batch_results") -> Dict[str, str]:
        """
        Save batch processing results to various formats
        
        Args:
            batch_results: Batch processing results
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.output_dir, 'results')
        reports_dir = os.path.join(self.output_dir, 'reports')
        
        saved_files = {}
        
        try:
            # Save complete results as JSON
            json_path = os.path.join(results_dir, f"{output_prefix}_{timestamp}.json")
            save_results(batch_results, json_path, "json")
            saved_files['json'] = json_path
            
            # Save scoring results as CSV if available
            if 'individual_results' in batch_results:
                scoring_results = []
                for result in batch_results['individual_results']:
                    if result.get('success') and 'scoring_result' in result:
                        scoring_result = result['scoring_result'].copy()
                        scoring_result['image_name'] = result['image_name']
                        scoring_result['image_path'] = result['image_path']
                        scoring_results.append(scoring_result)
                
                if scoring_results and self.scorer:
                    csv_path = os.path.join(results_dir, f"{output_prefix}_scores_{timestamp}.csv")
                    if self.scorer.export_results_to_csv(scoring_results, csv_path):
                        saved_files['csv'] = csv_path
            
            # Save summary report
            if 'batch_summary' in batch_results:
                summary_path = os.path.join(reports_dir, f"{output_prefix}_summary_{timestamp}.json")
                summary_data = {
                    'batch_summary': batch_results['batch_summary'],
                    'scoring_report': batch_results.get('scoring_report', {}),
                    'accuracy_metrics': batch_results.get('accuracy_metrics', {})
                }
                save_results(summary_data, summary_path, "json")
                saved_files['summary'] = summary_path
            
            self.logger.info(f"Batch results saved to: {saved_files}")
            
        except Exception as e:
            self.logger.error(f"Error saving batch results: {e}")
        
        return saved_files

def main():
    """
    Main function for running batch processing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="OMR Batch Processing")
    parser.add_argument("input_dir", help="Directory containing OMR sheet images")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output-prefix", default="batch_results", help="Output file prefix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    try:
        # Initialize processor
        processor = OMRBatchProcessor(args.config, args.debug)
        
        # Process batch
        results = processor.process_batch(args.input_dir)
        
        if 'error' in results:
            print(f"Batch processing failed: {results['error']}")
            return
        
        # Print summary
        summary = results['batch_summary']
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*50}")
        print(f"Total images processed: {summary['total_processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']}%")
        print(f"Total processing time: {summary['total_processing_time']:.2f} seconds")
        print(f"Average time per image: {summary['average_time_per_image']:.2f} seconds")
        
        # Print scoring summary if available
        if 'scoring_report' in results:
            scoring = results['scoring_report']['overall_statistics']
            print(f"\nSCORING SUMMARY:")
            print(f"Average score: {scoring['average_score']}")
            print(f"Average percentage: {scoring['average_percentage']:.2f}%")
            print(f"Highest score: {scoring['highest_score']}")
            print(f"Lowest score: {scoring['lowest_score']}")
        
        # Save results
        saved_files = processor.save_batch_results(results, args.output_prefix)
        print(f"\nRESULTS SAVED TO:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        
        # Print failed images if any
        if summary['failed'] > 0:
            print(f"\nFAILED IMAGES:")
            for result in results['individual_results']:
                if not result['success']:
                    print(f"  {result['image_name']}: {', '.join(result.get('errors', ['Unknown error']))}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()