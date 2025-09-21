"""
Scoring module for OMR evaluation
Matches student answers with answer keys and calculates scores
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from .utils import setup_logging, parse_multiple_answers, extract_set_from_filename

class OMRScorer:
    """
    Handles scoring of OMR sheets against answer keys
    """
    
    def __init__(self, config: dict, answer_keys: Dict[str, Dict[int, str]], debug: bool = False):
        """
        Initialize scorer with configuration and answer keys
        
        Args:
            config: Configuration dictionary
            answer_keys: Dictionary with answer keys for each set
            debug: Enable debug mode
        """
        self.config = config
        self.answer_keys = answer_keys
        self.debug = debug
        self.logger = setup_logging()
        
        # Extract scoring parameters
        self.subjects = config['scoring']['subjects']
        self.marks_per_question = config['scoring']['marks_per_question']
        self.negative_marking = config['scoring'].get('negative_marking', False)
        self.multiple_answers_allowed = config['scoring'].get('multiple_answers_allowed', True)
        
        # Create subject mapping
        self.subject_ranges = {}
        for subject in self.subjects:
            self.subject_ranges[subject['name']] = {
                'start': subject['start'],
                'end': subject['end']
            }
    
    def convert_answers_to_letters(self, marked_answers: Dict[int, List[int]]) -> Dict[int, List[str]]:
        """
        Convert numeric option indices to letter options (0->A, 1->B, etc.)
        
        Args:
            marked_answers: Dictionary with numeric options
            
        Returns:
            Dictionary with letter options
        """
        try:
            letter_answers = {}
            
            for question_num, options in marked_answers.items():
                letter_options = [chr(65 + opt) for opt in options if 0 <= opt <= 3]  # A, B, C, D
                letter_answers[question_num] = letter_options
            
            return letter_answers
            
        except Exception as e:
            self.logger.error(f"Error converting answers to letters: {e}")
            return {}
    
    def match_single_answer(self, student_answer: List[str], correct_answer: List[str]) -> Tuple[bool, float]:
        """
        Match a single question's answer with the correct answer
        
        Args:
            student_answer: List of student's selected options
            correct_answer: List of correct options
            
        Returns:
            Tuple of (is_correct, partial_score_ratio)
        """
        try:
            if not student_answer:  # No answer given
                return False, 0.0
            
            if not correct_answer:  # No correct answer defined
                return False, 0.0
            
            # Convert to sets for easier comparison
            student_set = set(answer.lower() for answer in student_answer)
            correct_set = set(answer.lower() for answer in correct_answer)
            
            # Exact match
            if student_set == correct_set:
                return True, 1.0
            
            # Partial credit for multiple answer questions
            if len(correct_set) > 1 and self.multiple_answers_allowed:
                correct_selections = len(student_set.intersection(correct_set))
                incorrect_selections = len(student_set - correct_set)
                
                if correct_selections > 0:
                    # Partial credit: (correct - incorrect) / total_correct
                    partial_score = (correct_selections - incorrect_selections) / len(correct_set)
                    partial_score = max(0.0, min(1.0, partial_score))  # Clamp between 0 and 1
                    
                    return partial_score > 0, partial_score
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"Error matching answer: {e}")
            return False, 0.0
    
    def calculate_subject_scores(self, question_scores: Dict[int, float]) -> Dict[str, float]:
        """
        Calculate scores for each subject
        
        Args:
            question_scores: Dictionary of scores for each question
            
        Returns:
            Dictionary of subject scores
        """
        try:
            subject_scores = {}
            
            for subject_name, subject_range in self.subject_ranges.items():
                start_q = subject_range['start']
                end_q = subject_range['end']
                
                total_score = 0.0
                questions_in_subject = 0
                
                for question_num in range(start_q, end_q + 1):
                    if question_num in question_scores:
                        total_score += question_scores[question_num] * self.marks_per_question
                        questions_in_subject += 1
                
                # Round to avoid floating point precision issues
                subject_scores[subject_name] = round(total_score, 2)
            
            return subject_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating subject scores: {e}")
            return {}
    
    def score_student_sheet(self, marked_answers: Dict[int, List[int]], answer_key_set: str, 
                          student_info: Dict = None) -> Dict:
        """
        Score a single student's OMR sheet
        
        Args:
            marked_answers: Student's marked answers (numeric format)
            answer_key_set: Which answer key set to use (e.g., 'set_a', 'set_b')
            student_info: Optional student information
            
        Returns:
            Complete scoring result
        """
        try:
            # Convert numeric answers to letters
            letter_answers = self.convert_answers_to_letters(marked_answers)
            
            # Get the appropriate answer key
            if answer_key_set not in self.answer_keys:
                raise ValueError(f"Answer key set '{answer_key_set}' not found")
            
            answer_key = self.answer_keys[answer_key_set]
            
            # Score each question
            question_scores = {}
            detailed_results = {}
            
            for question_num in range(1, 101):  # Questions 1-100
                student_answer = letter_answers.get(question_num, [])
                
                if question_num in answer_key:
                    correct_answer = parse_multiple_answers(answer_key[question_num])
                    is_correct, score_ratio = self.match_single_answer(student_answer, correct_answer)
                    
                    question_scores[question_num] = score_ratio
                    
                    detailed_results[question_num] = {
                        'student_answer': student_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct,
                        'score_ratio': score_ratio,
                        'marks_awarded': score_ratio * self.marks_per_question
                    }
                else:
                    # Question not in answer key
                    question_scores[question_num] = 0.0
                    detailed_results[question_num] = {
                        'student_answer': student_answer,
                        'correct_answer': [],
                        'is_correct': False,
                        'score_ratio': 0.0,
                        'marks_awarded': 0.0,
                        'note': 'Question not in answer key'
                    }
            
            # Calculate subject scores
            subject_scores = self.calculate_subject_scores(question_scores)
            
            # Calculate total score
            total_score = sum(subject_scores.values())
            
            # Calculate statistics
            total_questions = len([q for q in range(1, 101) if q in answer_key])
            correct_answers = sum(1 for score in question_scores.values() if score == 1.0)
            partial_answers = sum(1 for score in question_scores.values() if 0 < score < 1.0)
            incorrect_answers = sum(1 for score in question_scores.values() if score == 0.0)
            
            # Calculate confidence (based on number of answered questions)
            answered_questions = len([q for q, ans in letter_answers.items() if ans])
            confidence = answered_questions / 100 if answered_questions > 0 else 0.0
            
            # Compile final result
            result = {
                'student_info': student_info or {},
                'answer_key_set': answer_key_set,
                'subject_scores': subject_scores,
                'total_score': round(total_score, 2),
                'max_possible_score': total_questions * self.marks_per_question,
                'percentage': round((total_score / (total_questions * self.marks_per_question)) * 100, 2) if total_questions > 0 else 0.0,
                'statistics': {
                    'total_questions': total_questions,
                    'questions_answered': answered_questions,
                    'correct_answers': correct_answers,
                    'partial_answers': partial_answers,
                    'incorrect_answers': incorrect_answers,
                    'unanswered': total_questions - answered_questions
                },
                'confidence': round(confidence, 3),
                'detailed_results': detailed_results,
                'marked_answers': letter_answers
            }
            
            self.logger.info(f"Scored sheet: {total_score}/{total_questions * self.marks_per_question} "
                           f"({result['percentage']:.1f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error scoring student sheet: {e}")
            return {
                'error': str(e),
                'total_score': 0.0,
                'subject_scores': {},
                'success': False
            }
    
    def batch_score_sheets(self, batch_marked_answers: List[Dict], 
                          batch_info: List[Dict] = None) -> List[Dict]:
        """
        Score multiple OMR sheets in batch
        
        Args:
            batch_marked_answers: List of marked answers for each sheet
            batch_info: Optional list of student information
            
        Returns:
            List of scoring results
        """
        try:
            results = []
            batch_info = batch_info or [{}] * len(batch_marked_answers)
            
            for i, marked_answers in enumerate(batch_marked_answers):
                try:
                    student_info = batch_info[i] if i < len(batch_info) else {}
                    
                    # Determine answer key set (default to set_a)
                    answer_key_set = student_info.get('answer_key_set', 'set_a')
                    
                    # Score the sheet
                    result = self.score_student_sheet(marked_answers, answer_key_set, student_info)
                    result['sheet_index'] = i
                    result['success'] = 'error' not in result
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error scoring sheet {i}: {e}")
                    results.append({
                        'sheet_index': i,
                        'error': str(e),
                        'success': False,
                        'total_score': 0.0,
                        'subject_scores': {}
                    })
            
            successful = sum(1 for r in results if r.get('success', False))
            self.logger.info(f"Batch scoring completed: {successful}/{len(results)} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch scoring: {e}")
            return []
    
    def generate_score_report(self, scoring_results: List[Dict]) -> Dict:
        """
        Generate comprehensive score report from batch results
        
        Args:
            scoring_results: List of individual scoring results
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            if not scoring_results:
                return {'error': 'No scoring results provided'}
            
            successful_results = [r for r in scoring_results if r.get('success', False)]
            
            if not successful_results:
                return {'error': 'No successful scoring results'}
            
            # Overall statistics
            total_students = len(successful_results)
            total_scores = [r['total_score'] for r in successful_results]
            percentages = [r['percentage'] for r in successful_results]
            
            overall_stats = {
                'total_students': total_students,
                'average_score': round(sum(total_scores) / total_students, 2),
                'average_percentage': round(sum(percentages) / total_students, 2),
                'highest_score': max(total_scores),
                'lowest_score': min(total_scores),
                'highest_percentage': max(percentages),
                'lowest_percentage': min(percentages)
            }
            
            # Subject-wise statistics
            subject_stats = {}
            for subject_name in self.subject_ranges.keys():
                subject_scores = [r['subject_scores'].get(subject_name, 0) for r in successful_results]
                max_subject_score = (self.subject_ranges[subject_name]['end'] - 
                                   self.subject_ranges[subject_name]['start'] + 1) * self.marks_per_question
                
                subject_percentages = [(score / max_subject_score) * 100 for score in subject_scores if max_subject_score > 0]
                
                subject_stats[subject_name] = {
                    'average_score': round(sum(subject_scores) / total_students, 2),
                    'average_percentage': round(sum(subject_percentages) / total_students, 2) if subject_percentages else 0,
                    'highest_score': max(subject_scores),
                    'lowest_score': min(subject_scores),
                    'max_possible_score': max_subject_score
                }
            
            # Grade distribution (example grading scale)
            grade_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
            for percentage in percentages:
                if percentage >= 90:
                    grade_distribution['A'] += 1
                elif percentage >= 80:
                    grade_distribution['B'] += 1
                elif percentage >= 70:
                    grade_distribution['C'] += 1
                elif percentage >= 60:
                    grade_distribution['D'] += 1
                else:
                    grade_distribution['F'] += 1
            
            # Top performers
            top_performers = sorted(successful_results, 
                                  key=lambda x: x['percentage'], 
                                  reverse=True)[:5]
            
            report = {
                'overall_statistics': overall_stats,
                'subject_statistics': subject_stats,
                'grade_distribution': grade_distribution,
                'top_performers': [
                    {
                        'student_info': r.get('student_info', {}),
                        'total_score': r['total_score'],
                        'percentage': r['percentage'],
                        'subject_scores': r['subject_scores']
                    } for r in top_performers
                ],
                'analysis_summary': {
                    'total_processed': len(scoring_results),
                    'successful_evaluations': len(successful_results),
                    'failed_evaluations': len(scoring_results) - len(successful_results),
                    'evaluation_success_rate': round((len(successful_results) / len(scoring_results)) * 100, 2)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating score report: {e}")
            return {'error': str(e)}
    
    def export_results_to_csv(self, scoring_results: List[Dict], output_path: str) -> bool:
        """
        Export scoring results to CSV file
        
        Args:
            scoring_results: List of scoring results
            output_path: Output CSV file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for CSV
            csv_data = []
            
            for result in scoring_results:
                if result.get('success', False):
                    row = {
                        'Sheet_Index': result.get('sheet_index', ''),
                        'Total_Score': result.get('total_score', 0),
                        'Percentage': result.get('percentage', 0),
                        'Answer_Key_Set': result.get('answer_key_set', ''),
                        'Confidence': result.get('confidence', 0)
                    }
                    
                    # Add subject scores
                    for subject_name in self.subject_ranges.keys():
                        row[f'{subject_name}_Score'] = result.get('subject_scores', {}).get(subject_name, 0)
                    
                    # Add statistics
                    stats = result.get('statistics', {})
                    row.update({
                        'Questions_Answered': stats.get('questions_answered', 0),
                        'Correct_Answers': stats.get('correct_answers', 0),
                        'Partial_Answers': stats.get('partial_answers', 0),
                        'Incorrect_Answers': stats.get('incorrect_answers', 0)
                    })
                    
                    csv_data.append(row)
                else:
                    # Failed result
                    csv_data.append({
                        'Sheet_Index': result.get('sheet_index', ''),
                        'Error': result.get('error', 'Unknown error'),
                        'Total_Score': 0,
                        'Percentage': 0
                    })
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Results exported to CSV: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False

def main():
    """
    Main function for testing scoring module
    """
    import os
    from .utils import load_config, load_answer_key
    
    # Load configuration
    config = load_config()
    
    # Load answer keys
    answer_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Key (Set A and B).xlsx")
    if os.path.exists(answer_key_path):
        try:
            answer_keys = load_answer_key(answer_key_path)
            print(f"Loaded answer keys for sets: {list(answer_keys.keys())}")
        except Exception as e:
            print(f"Error loading answer keys: {e}")
            # Create dummy answer keys for testing
            answer_keys = {
                'set_a': {i: 'a' for i in range(1, 101)},  # All A's for testing
                'set_b': {i: 'b' for i in range(1, 101)}   # All B's for testing
            }
            print("Using dummy answer keys for testing")
    else:
        print(f"Answer key file not found: {answer_key_path}")
        return
    
    # Initialize scorer
    scorer = OMRScorer(config, answer_keys, debug=True)
    
    # Create sample marked answers for testing
    sample_answers = {
        i: [0] if i <= 50 else [1] for i in range(1, 101)  # Mix of A and B answers
    }
    
    print("Testing scoring with sample data...")
    
    try:
        # Test single sheet scoring
        result = scorer.score_student_sheet(sample_answers, 'set_a', {'student_id': 'TEST001'})
        
        print(f"\nScoring Results:")
        print(f"Total Score: {result['total_score']}/{result['max_possible_score']}")
        print(f"Percentage: {result['percentage']:.2f}%")
        print(f"Confidence: {result['confidence']:.3f}")
        
        print(f"\nSubject Scores:")
        for subject, score in result['subject_scores'].items():
            print(f"  {subject}: {score}")
        
        print(f"\nStatistics:")
        stats = result['statistics']
        print(f"  Answered: {stats['questions_answered']}")
        print(f"  Correct: {stats['correct_answers']}")
        print(f"  Partial: {stats['partial_answers']}")
        print(f"  Incorrect: {stats['incorrect_answers']}")
        
        # Test batch scoring
        batch_results = scorer.batch_score_sheets([sample_answers], [{'student_id': 'TEST001'}])
        
        # Generate report
        report = scorer.generate_score_report(batch_results)
        print(f"\nBatch Report:")
        print(f"  Students processed: {report['analysis_summary']['total_processed']}")
        print(f"  Success rate: {report['analysis_summary']['evaluation_success_rate']:.2f}%")
        print(f"  Average score: {report['overall_statistics']['average_score']}")
        
        # Test CSV export
        output_dir = "output/test_scoring"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "test_results.csv")
        
        if scorer.export_results_to_csv(batch_results, csv_path):
            print(f"\nResults exported to: {csv_path}")
        
    except Exception as e:
        print(f"Error in scoring test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()