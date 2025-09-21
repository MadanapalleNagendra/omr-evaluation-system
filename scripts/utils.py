import os
ANSWER_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "answer_keys_omr.xlsx")
"""
Utility functions for OMR evaluation system
Contains common functions used across multiple modules
"""

import os
import json
import yaml
import logging
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def load_answer_key(file_path: str) -> Dict[str, Dict[int, str]]:
    """
    Load answer keys from Excel file
    
    Args:
        file_path: Path to Excel file containing answer keys
    
    Returns:
        Dictionary with answer keys for each set
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
        answer_keys = {}

        for sheet_name, sheet_data in df.items():
            if 'set' in sheet_name.lower():
                set_name = sheet_name.lower().replace(' ', '_')
                answers = {}
                # Parse answer data - assuming format: Question | Answer
                for _, row in sheet_data.iterrows():
                    if len(row) >= 2 and pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                        question_num = int(row.iloc[0]) if isinstance(row.iloc[0], (int, float)) else int(str(row.iloc[0]).split('.')[0])
                        answer = str(row.iloc[1]).strip().lower()
                        answers[question_num] = answer

                if answers:
                    answer_keys[set_name] = answers

        if not answer_keys:
            logging.warning(f"No valid answer key sets found in file: {file_path}")
        else:
            logging.info(f"Loaded answer keys for sets: {list(answer_keys.keys())}")
        return answer_keys

    except Exception as e:
        raise ValueError(f"Error loading answer key from {file_path}: {e}")

def validate_image(image_path: str) -> bool:
    """
    Validate if image file is readable
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        return img is not None and img.size > 0
    except:
        return False

def save_debug_image(image: np.ndarray, filename: str, output_dir: str) -> None:
    """
    Save debug image if debug mode is enabled
    
    Args:
        image: Image array to save
        filename: Output filename
        output_dir: Output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, filename), image)
    except Exception as e:
        logging.warning(f"Failed to save debug image {filename}: {e}")

def calculate_confidence(marked_pixels: int, total_pixels: int, threshold: float = 0.3) -> float:
    """
    Calculate confidence score for bubble marking
    
    Args:
        marked_pixels: Number of dark pixels in bubble
        total_pixels: Total pixels in bubble area
        threshold: Threshold for considering bubble as marked
    
    Returns:
        Confidence score between 0 and 1
    """
    if total_pixels == 0:
        return 0.0
    
    fill_ratio = marked_pixels / total_pixels
    
    # Calculate confidence based on how far from threshold
    if fill_ratio >= threshold:
        # Marked bubble - higher confidence as fill_ratio increases
        confidence = min(1.0, fill_ratio / (threshold * 2))
    else:
        # Unmarked bubble - higher confidence as fill_ratio decreases
        confidence = min(1.0, (threshold - fill_ratio) / threshold)
    
    return confidence

def parse_multiple_answers(answer: str) -> List[str]:
    """
    Parse multiple answers from string (e.g., "a,b" -> ["a", "b"])
    
    Args:
        answer: Answer string
    
    Returns:
        List of individual answers
    """
    if not answer or pd.isna(answer):
        return []
    
    # Handle various formats: "a,b", "a, b", "a,b,c,d"
    answer_str = str(answer).lower().replace(' ', '')
    
    if ',' in answer_str:
        return [ans.strip() for ans in answer_str.split(',') if ans.strip()]
    else:
        return [answer_str.strip()] if answer_str.strip() else []

def extract_set_from_filename(filename: str) -> str:
    """
    Extract set identifier from filename
    
    Args:
        filename: Image filename
    
    Returns:
        Set identifier (e.g., "set_a", "set_b")
    """
    filename_lower = filename.lower()
    
    if 'set_a' in filename_lower or 'seta' in filename_lower:
        return 'set_a'
    elif 'set_b' in filename_lower or 'setb' in filename_lower:
        return 'set_b'
    else:
        # Default to set_a if unable to determine
        return 'set_a'

def save_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None:
    """
    Save results to file in specified format
    
    Args:
        results: Results dictionary
        output_path: Output file path
        format: Output format ("json", "csv", "excel")
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            # Convert results to DataFrame for CSV export
            if isinstance(results, dict) and 'students' in results:
                df = pd.DataFrame(results['students'])
                df.to_csv(output_path, index=False)
            else:
                # Fallback: save as JSON if structure is complex
                save_results(results, output_path.replace('.csv', '.json'), "json")
        
        elif format.lower() == "excel":
            if isinstance(results, dict) and 'students' in results:
                df = pd.DataFrame(results['students'])
                df.to_excel(output_path, index=False)
            else:
                save_results(results, output_path.replace('.xlsx', '.json'), "json")
    
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")

def get_image_files(directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']) -> List[str]:
    """
    Get all image files from directory
    
    Args:
        directory: Directory path
        extensions: List of valid image extensions
    
    Returns:
        List of image file paths
    """
    image_files = []
    
    if not os.path.exists(directory):
        logging.warning(f"Directory does not exist: {directory}")
        return image_files
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, file))
    
    return sorted(image_files)

def resize_image_aspect_ratio(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target - fit by width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target - fit by height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def calculate_accuracy_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from evaluation results
    
    Args:
        results: Evaluation results
    
    Returns:
        Dictionary with accuracy metrics
    """
    if 'students' not in results:
        return {}
    
    students = results['students']
    total_students = len(students)
    
    if total_students == 0:
        return {}
    
    # Calculate averages
    avg_total_score = sum(s.get('total_score', 0) for s in students) / total_students
    avg_confidence = sum(s.get('confidence', 0) for s in students) / total_students
    
    # Subject-wise averages
    subject_averages = {}
    for student in students:
        for subject, score in student.get('subject_scores', {}).items():
            if subject not in subject_averages:
                subject_averages[subject] = []
            subject_averages[subject].append(score)
    
    # Calculate subject averages
    for subject in subject_averages:
        subject_averages[subject] = sum(subject_averages[subject]) / len(subject_averages[subject])
    
    return {
        'total_students': total_students,
        'average_total_score': avg_total_score,
        'average_confidence': avg_confidence,
        'subject_averages': subject_averages,
        'max_possible_score': 100,
        'overall_percentage': (avg_total_score / 100) * 100
    }