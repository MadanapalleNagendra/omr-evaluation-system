

import streamlit as st
import os
import sys
import tempfile
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import numpy as np

# Add scripts directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from scripts.run_batch import OMRBatchProcessor
from scripts.utils import load_config, get_image_files

st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME: Modern light/blue accent ---
st.markdown("""
<style>
    body, .stApp {
        background: #f7fafd !important;
        color: #222 !important;
    }
    .main-header {
        text-align: center;
        color: #1976d2;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #e3f0fc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        color: #222;
    }
    .success-message {
        color: #388e3c;
        font-weight: bold;
    }
    .error-message {
        color: #d32f2f;
        font-weight: bold;
    }
    .warning-message {
        color: #fbc02d;
        font-weight: bold;
    }
    .stButton>button {
        background: #1976d2;
        color: #fff;
        font-weight: bold;
        border-radius: 0.3rem;
    }
    .stTextInput>div>input {
        background: #fff;
        color: #222;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIN PAGE ---
def login_page():
    st.markdown('<h1 class="main-header">🔒 Login to OMR System</h1>', unsafe_allow_html=True)
    st.write(":key: Please login to continue.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    if login_btn:
        # For demo: hardcoded user, replace with secure check or users.json
        if username == "admin" and password == "admin123":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

def logout_button():
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

# --- MAIN APP LOGIC ---
def main():
    if not st.session_state.get("logged_in", False):
        login_page()
        return
    logout_button()
    # ...existing OMR/visualization logic will go here...
    st.markdown('<h1 class="main-header">📋 OMR Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.write("Welcome! Upload your OMR sheets and process results.")
    # TODO: Insert OMR and visualization logic here

if __name__ == "__main__":
    main()

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #1f77b4;
#         font-size: 2.5rem;
#         font-weight: bold;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#     }
#     .success-message {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .error-message {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .warning-message {
#         color: #ffc107;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# def initialize_session_state():
#     """Initialize session state variables"""
#     if 'processor' not in st.session_state:
#         st.session_state.processor = None
#     if 'batch_results' not in st.session_state:
#         st.session_state.batch_results = None
#     if 'config_loaded' not in st.session_state:
#         st.session_state.config_loaded = False

# def load_processor():
#     """Load and initialize the OMR processor"""
#     try:
#         if st.session_state.processor is None:
#             with st.spinner("Initializing OMR system..."):
#                 st.session_state.processor = OMRBatchProcessor(
#                     config_path="configs/config.yaml",
#                     debug=st.session_state.get('debug_mode', False)
#                 )
#                 st.session_state.config_loaded = True
#         return True
#     except Exception as e:
#         st.error(f"Failed to initialize OMR system: {str(e)}")
#         return False
# if __name__ == "__main__":
#     main()
# def display_header():
#     """Display application header"""
#     st.markdown('<h1 class="main-header">📋 OMR Evaluation System</h1>', unsafe_allow_html=True)
#     st.markdown("---")

# def sidebar_configuration():
#     """Display sidebar configuration options"""
#     st.sidebar.header("⚙️ Configuration")
    
#     # Debug mode toggle with better explanation
#     debug_mode = st.sidebar.checkbox(
#         "🔧 Enable Debug Mode", 
#         value=True,  # Default to True for overlay generation
#         help="""**Recommended for troubleshooting:**
#         - Generates overlay images with detected bubbles
#         - Saves intermediate processing images
#         - Creates detailed debug logs
#         - Essential for overlay visualization"""
#     )
#     st.session_state.debug_mode = debug_mode
    
#     if debug_mode:
#         st.sidebar.success("✅ Debug mode enabled - overlays will be generated")
#     else:
#         st.sidebar.warning("⚠️ Debug mode disabled - no overlays will be generated")
    
#     # Processing options
#     st.sidebar.header("📊 Processing Options")
    
#     # Answer key selection
#     answer_key_options = ["Auto-detect from filename", "Set A", "Set B"]
#     selected_key = st.sidebar.selectbox("Answer Key Set", answer_key_options)
    
#     # Quality thresholds
#     confidence_threshold = st.sidebar.slider(
#         "Confidence Threshold", 
#         min_value=0.1, max_value=1.0, value=0.6, step=0.05,
#         help="Minimum confidence for accepting bubble classifications (lower = more sensitive)"
#     )
    
#     # File format options
#     st.sidebar.header("💾 Output Options")
#     export_formats = st.sidebar.multiselect(
#         "Export Formats",
#         ["JSON", "CSV", "Excel"],
#         default=["CSV", "JSON"]
#     )
    
#     # Debug output location
#     st.sidebar.header("📁 Debug Output")
#     debug_location = st.sidebar.radio(
#         "Save debug files to:",
#         ["Temporary folder (clears on restart)", "Persistent logs folder"],
#         index=1 if debug_mode else 0
#     )
    
#     return {
#         'debug_mode': debug_mode,
#         'answer_key_set': selected_key,
#         'confidence_threshold': confidence_threshold,
#         'export_formats': export_formats,
#         'debug_location': debug_location
#     }

# def file_upload_section():
#     """Handle file upload section"""
#     st.header("📁 Upload OMR Sheets")
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Choose OMR sheet images",
#         type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
#         accept_multiple_files=True,
#         help="Upload multiple OMR sheet images for batch processing"
#     )
    
#     if uploaded_files:
#         st.success(f"✅ {len(uploaded_files)} files uploaded successfully")
        
#         # Display uploaded files
#         with st.expander("📋 Uploaded Files", expanded=False):
#             for i, file in enumerate(uploaded_files):
#                 col1, col2, col3 = st.columns([3, 2, 1])
#                 with col1:
#                     st.text(file.name)
#                 with col2:
#                     st.text(f"{file.size / 1024:.1f} KB")
#                 with col3:
#                     if st.button(f"👁️", key=f"preview_{i}", help="Preview image"):
#                         preview_image(file)
        
#         return uploaded_files
    
#     return None

# def preview_image(uploaded_file):
#     """Preview uploaded image"""
#     try:
#         image = Image.open(uploaded_file)
#         st.image(image, caption=uploaded_file.name, width='stretch')
#     except Exception as e:
#         st.error(f"Error previewing image: {str(e)}")

# def process_images(uploaded_files, config_options):
#     """Process uploaded images with proper debug handling"""
#     if not uploaded_files:
#         return None
    
#     # Create persistent debug directory if debug mode is enabled
#     debug_base_dir = None
#     if config_options['debug_mode']:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         debug_base_dir = os.path.join("logs", f"debug_session_{timestamp}")
#         os.makedirs(debug_base_dir, exist_ok=True)
#         st.info(f"🔧 Debug files will be saved to: `{debug_base_dir}`")
    
#     # Create temporary directory for uploaded files
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Save uploaded files to temporary directory
#         temp_paths = []
#         for uploaded_file in uploaded_files:
#             # Reset file pointer for reading
#             uploaded_file.seek(0)
#             temp_path = os.path.join(temp_dir, uploaded_file.name)
#             with open(temp_path, 'wb') as f:
#                 f.write(uploaded_file.getvalue())
#             temp_paths.append(temp_path)
        
#         # Initialize processor with current config and debug settings
#         processor = OMRBatchProcessor(
#             config_path="configs/config.yaml",
#             debug=config_options['debug_mode']
#         )
        
#         # Set debug directory for the processor
#         if debug_base_dir:
#             processor.debug_dir = debug_base_dir
        
#         # Process batch
#         with st.spinner(f"🔄 Processing {len(uploaded_files)} OMR sheet(s)... This may take a few minutes."):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             status_text.text("📁 Preparing images...")
#             progress_bar.progress(10)
            
#             status_text.text("🔍 Detecting sheets and extracting grids...")
#             progress_bar.progress(50)
            
#             # Process images
#             results = processor.process_batch(temp_dir)
#             progress_bar.progress(90)
            
#             status_text.text("🎯 Classifying bubbles and scoring...")
#             progress_bar.progress(100)
#             status_text.text("✅ Processing completed!")
        
#         # Copy debug files to persistent location if needed
#         if debug_base_dir and os.path.exists(processor.debug_dir):
#             # The debug files should already be in the right place
#             st.success(f"📂 Debug files saved to: `{debug_base_dir}`")
#             st.info("Check the `logs/debug_session_*/` folder for overlay images and intermediate results")
        
#         return results
# def display_results_summary(batch_results):
#     """Display batch processing results summary"""
#     if not batch_results or 'batch_summary' not in batch_results:
#         return
    
#     st.header("📊 Processing Results")
    
#     summary = batch_results['batch_summary']
    
#     # Main metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric(
#             label="Total Images",
#             value=summary['total_processed'],
#             help="Total number of images processed"
#         )
    
#     with col2:
#         st.metric(
#             label="Successful",
#             value=summary['successful'],
#             delta=f"{summary['success_rate']:.1f}%",
#             help="Successfully processed images"
#         )
    
#     with col3:
#         st.metric(
#             label="Processing Time",
#             value=f"{summary['total_processing_time']:.1f}s",
#             delta=f"{summary['average_time_per_image']:.1f}s avg",
#             help="Total and average processing time"
#         )
    
#     with col4:
#         if summary['failed'] > 0:
#             st.metric(
#                 label="Failed",
#                 value=summary['failed'],
#                 delta=f"{(summary['failed']/summary['total_processed']*100):.1f}%",
#                 delta_color="inverse",
#                 help="Images that failed processing"
#             )
#         else:
#             st.metric(
#                 label="Success Rate",
#                 value="100%",
#                 help="All images processed successfully"
#             )

# def display_scoring_results(batch_results):
#     """Display scoring results and analytics"""
#     if not batch_results or 'scoring_report' not in batch_results:
#         return
    
#     st.header("🎯 Scoring Analysis")
    
#     scoring_report = batch_results['scoring_report']
    
#     if 'overall_statistics' in scoring_report:
#         stats = scoring_report['overall_statistics']
        
#         # Overall statistics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Average Score", f"{stats['average_score']:.1f}")
#         with col2:
#             st.metric("Average %", f"{stats['average_percentage']:.1f}%")
#         with col3:
#             st.metric("Highest Score", f"{stats['highest_score']:.1f}")
#         with col4:
#             st.metric("Lowest Score", f"{stats['lowest_score']:.1f}")
        
#         # Score distribution
#         if 'grade_distribution' in scoring_report:
#             st.subheader("📈 Grade Distribution")
            
#             grades = list(scoring_report['grade_distribution'].keys())
#             counts = list(scoring_report['grade_distribution'].values())
            
#             fig = px.bar(
#                 x=grades, y=counts,
#                 labels={'x': 'Grade', 'y': 'Number of Students'},
#                 title="Grade Distribution",
#                 color=counts,
#                 color_continuous_scale='viridis'
#             )
#             st.plotly_chart(fig, width='stretch')
        
#         # Subject-wise performance
#         if 'subject_statistics' in scoring_report:
#             st.subheader("📚 Subject-wise Performance")
            
#             subject_data = []
#             for subject, subject_stats in scoring_report['subject_statistics'].items():
#                 subject_data.append({
#                     'Subject': subject,
#                     'Average Score': subject_stats['average_score'],
#                     'Average %': subject_stats['average_percentage'],
#                     'Highest': subject_stats['highest_score'],
#                     'Lowest': subject_stats['lowest_score']
#                 })
            
#             df_subjects = pd.DataFrame(subject_data)
#             st.dataframe(df_subjects, width='stretch')
            
#             # Subject performance chart
#             fig = px.bar(
#                 df_subjects, 
#                 x='Subject', 
#                 y='Average %',
#                 title="Average Performance by Subject",
#                 color='Average %',
#                 color_continuous_scale='RdYlGn'
#             )
#             fig.update_layout(showlegend=False)
#             st.plotly_chart(fig, width='stretch')

# def display_individual_results(batch_results):
#     """Display individual image processing results with improved overlay handling"""
#     if not batch_results or 'individual_results' not in batch_results:
#         return
    
#     st.header("🔍 Individual Results")
    
#     individual_results = batch_results['individual_results']
    
#     # Create tabs for successful and failed results
#     tab1, tab2 = st.tabs(["✅ Successful", "❌ Failed"])
    
#     with tab1:
#         successful_results = [r for r in individual_results if r.get('success', False)]
        
#         if successful_results:
#             # Create summary table
#             summary_data = []
#             for result in successful_results:
#                 scoring = result.get('scoring_result', {})
#                 row = {
#                     'Image Name': result['image_name'],
#                     'Total Score': scoring.get('total_score', 0),
#                     'Percentage': f"{scoring.get('percentage', 0):.1f}%",
#                     'Processing Time': f"{result.get('processing_time', 0):.1f}s",
#                     'Questions': result['pipeline_stages'].get('grid_extraction', {}).get('questions_detected', 0),
#                     'Bubbles': result['pipeline_stages'].get('classification', {}).get('total_bubbles', 0),
#                     'Marked': result['pipeline_stages'].get('classification', {}).get('marked_bubbles', 0)
#                 }
#                 summary_data.append(row)
            
#             df_summary = pd.DataFrame(summary_data)
#             st.dataframe(df_summary, width='stretch')
            
#             # Individual image displays
#             for result_idx, result in enumerate(successful_results):
#                 with st.expander(f"📋 {result['image_name']} - {result['pipeline_stages'].get('grid_extraction', {}).get('questions_detected', 0)} Questions", expanded=(result_idx == 0)):
                    
#                     # Create two-column layout for images
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         st.markdown("### 🖼️ Original OMR Sheet")
#                         orig_img_path = result.get('image_path')
#                         if orig_img_path and os.path.exists(orig_img_path):
#                             try:
#                                 orig_img = Image.open(orig_img_path)
#                                 st.image(orig_img, caption="Original OMR Sheet", width='stretch')
#                             except Exception as e:
#                                 st.warning(f"Could not display original image: {e}")
#                                 st.write(f"Path: {orig_img_path}")
#                         else:
#                             st.warning("Original image not found.")
                    
#                     with col2:
#                         st.markdown("### 🎯 Detected Bubbles Overlay")
                        
#                         # Try multiple locations to find overlay image
#                         overlay_paths_to_check = []
                        
#                         # 1. Check if full sheet overlay is stored in results
#                         if 'marked_answers' in result and 'full_sheet_overlay' in result.get('marked_answers', {}):
#                             overlay_path = result['marked_answers']['full_sheet_overlay']
#                             if overlay_path and os.path.exists(overlay_path):
#                                 overlay_paths_to_check.append(overlay_path)
                        
#                         # 2. Check debug directory
#                         debug_dir = result.get('debug_dir')
#                         if debug_dir and os.path.isdir(debug_dir):
#                             # Look for any overlay files in debug dir
#                             overlay_files = [f for f in os.listdir(debug_dir) if f.startswith('overlay_') and f.endswith(('.png', '.jpg', '.jpeg'))]
#                             if overlay_files:
#                                 overlay_path = os.path.join(debug_dir, overlay_files[0])  # Use first found
#                                 overlay_paths_to_check.append(overlay_path)
                        
#                         # 3. Check classification directory
#                         if debug_dir:
#                             class_dir = os.path.join(debug_dir, 'classification')
#                             if os.path.exists(class_dir):
#                                 overlay_files = [f for f in os.listdir(class_dir) if f.startswith('overlay_') and f.endswith(('.png', '.jpg', '.jpeg'))]
#                                 if overlay_files:
#                                     overlay_path = os.path.join(class_dir, overlay_files[0])
#                                     overlay_paths_to_check.append(overlay_path)
                        
#                         # 4. Check logs/overlays directory
#                         logs_dir = os.path.join(os.getcwd(), "logs", "overlays")
#                         if os.path.exists(logs_dir):
#                             overlay_files = [f for f in os.listdir(logs_dir) if f.startswith('overlay_full_sheet') and f.endswith(('.png', '.jpg', '.jpeg'))]
#                             if overlay_files:
#                                 # Get the most recent one
#                                 overlay_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
#                                 overlay_path = os.path.join(logs_dir, overlay_files[0])
#                                 overlay_paths_to_check.append(overlay_path)
                        
#                         # Try to display the first valid overlay found
#                         overlay_displayed = False
#                         for overlay_path in overlay_paths_to_check:
#                             if os.path.exists(overlay_path):
#                                 try:
#                                     overlay_img = Image.open(overlay_path)
#                                     st.image(overlay_img, caption=f"Detection Overlay ({os.path.basename(overlay_path)})", width='stretch')
#                                     overlay_displayed = True
#                                     st.info(f"✅ Overlay loaded from: {os.path.basename(overlay_path)}")
#                                     break
#                                 except Exception as e:
#                                     st.warning(f"Could not load overlay from {overlay_path}: {e}")
                        
#                         if not overlay_displayed:
#                             st.warning("❌ **Overlay image not found.** Enable debug mode and check:")
#                             st.markdown("""
#                             - Debug mode is enabled in sidebar
#                             - Look in the `logs/overlays/` folder
#                             - Check `{debug_dir}/classification/` folder
#                             - Verify the OMR image path in config
#                             """)
#                             # Show overlay search paths for debugging
#                             if overlay_paths_to_check:
#                                 st.write("**Searched locations:**")
#                                 for path in overlay_paths_to_check:
#                                     status = "✅" if os.path.exists(path) else "❌"
#                                     st.write(f"{status} {path}")
                    
#                     # Processing statistics
#                     st.markdown("### 📊 Processing Statistics")
#                     grid_stage = result['pipeline_stages'].get('grid_extraction', {})
#                     class_stage = result['pipeline_stages'].get('classification', {})
#                     score_stage = result.get('scoring_result', {})
                    
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("Questions", grid_stage.get('questions_detected', 0), "expected: 100")
#                     with col2:
#                         st.metric("Bubbles", class_stage.get('total_bubbles', 0))
#                     with col3:
#                         st.metric("Marked", class_stage.get('marked_bubbles', 0))
#                     with col4:
#                         st.metric("Avg Confidence", f"{class_stage.get('average_confidence', 0):.2f}")
                    
#                     # Grid issues
#                     issues = grid_stage.get('issues', [])
#                     if issues:
#                         with st.expander("⚠️ Detection Issues", expanded=True):
#                             for issue in issues:
#                                 st.warning(issue)
                    
#                     # Scoring results
#                     if score_stage:
#                         st.markdown("### 🎯 Scoring Results")
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.metric("Total Score", f"{score_stage.get('total_score', 0)}/80")
#                             st.metric("Percentage", f"{score_stage.get('percentage', 0):.1f}%")
#                         with col2:
#                             st.metric("Questions Answered", score_stage.get('statistics', {}).get('questions_answered', 0))
#                             st.metric("Correct Answers", score_stage.get('statistics', {}).get('correct_answers', 0))
                    
#                     # Show marked answers
#                     if 'marked_answers' in result and result['marked_answers']:
#                         st.markdown("### ✅ Detected Answers")
#                         marked_answers = result['marked_answers']
#                         answer_summary = []
#                         for q_num, options in sorted(marked_answers.items()):
#                             if options:  # Only show questions with answers
#                                 option_letters = [chr(65 + opt) for opt in options]
#                                 answer_summary.append(f"Q{q_num}: {', '.join(option_letters)}")
                        
#                         if answer_summary:
#                             st.write("\n".join(answer_summary[:20]))  # Show first 20
#                             if len(answer_summary) > 20:
#                                 st.info(f"... and {len(answer_summary) - 20} more answers")
#                         else:
#                             st.info("No clear answers detected (all bubbles may be ambiguous)")
                    
#                     st.markdown("---")
            
#             # Debug info section
#             with st.expander("🔧 Debug Information", expanded=False):
#                 st.markdown("### Debug Files Generated")
#                 debug_dir = successful_results[0].get('debug_dir') if successful_results else None
#                 if debug_dir and os.path.exists(debug_dir):
#                     debug_files = []
#                     for root, dirs, files in os.walk(debug_dir):
#                         for file in files:
#                             if file.endswith(('.png', '.jpg', '.jpeg')):
#                                 file_path = os.path.join(root, file)
#                                 file_size = os.path.getsize(file_path)
#                                 debug_files.append({
#                                     'name': file,
#                                     'path': file_path,
#                                     'size': f"{file_size/1024:.1f} KB",
#                                     'type': 'Image'
#                                 })
                    
#                     if debug_files:
#                         df_debug = pd.DataFrame(debug_files)
#                         st.dataframe(df_debug, width='stretch')
                        
#                         # Allow viewing debug images
#                         selected_debug_file = st.selectbox(
#                             "View debug image:", 
#                             options=[f"{f['name']} ({f['size']})" for f in debug_files],
#                             format_func=lambda x: x
#                         )
                        
#                         if selected_debug_file:
#                             for debug_file in debug_files:
#                                 if selected_debug_file.startswith(debug_file['name']):
#                                     try:
#                                         debug_img = Image.open(debug_file['path'])
#                                         st.image(debug_img, caption=f"Debug: {debug_file['name']}", width='stretch')
#                                         break
#                                     except Exception as e:
#                                         st.error(f"Could not display {debug_file['name']}: {e}")
#                     else:
#                         st.info("No debug images found in this session")
#                 else:
#                     st.warning("No debug directory found. Enable debug mode to generate debug files.")
        
#         else:
#             st.info("No successful results to display")
    
#     with tab2:
#         failed_results = [r for r in individual_results if not r.get('success', False)]
        
#         if failed_results:
#             for result in failed_results:
#                 with st.expander(f"❌ {result['image_name']}", expanded=True):
#                     st.error("**Processing failed**")
                    
#                     if 'errors' in result:
#                         st.markdown("### Errors:")
#                         for error in result['errors']:
#                             st.error(f"• {error}")
                    
#                     if 'pipeline_stages' in result:
#                         st.markdown("### Pipeline Status:")
#                         for stage, status in result['pipeline_stages'].items():
#                             icon = "✅" if status.get('success', False) else "❌"
#                             stage_name = stage.replace('_', ' ').title()
#                             st.write(f"**{icon} {stage_name}**")
#                             if not status.get('success', True) and 'error' in status:
#                                 st.error(f"  Error: {status['error']}")
#         else:
#             st.success("🎉 No failed results - all images processed successfully!")
# def display_detailed_result(result):
#     """Display detailed results for a single image"""
#     if not result:
#         return
    
#     # Pipeline stages status
#     st.subheader("🔧 Pipeline Status")
#     pipeline_stages = result.get('pipeline_stages', {})
    
#     cols = st.columns(len(pipeline_stages))
#     for i, (stage_name, stage_info) in enumerate(pipeline_stages.items()):
#         with cols[i]:
#             success = stage_info.get('success', False)
#             icon = "✅" if success else "❌"
#             st.metric(
#                 label=stage_name.replace('_', ' ').title(),
#                 value=icon,
#                 help=f"Status: {'Success' if success else 'Failed'}"
#             )
    
#     # Scoring details
#     if 'scoring_result' in result:
#         scoring = result['scoring_result']
        
#         st.subheader("🎯 Scoring Details")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write("**Overall Performance:**")
#             st.write(f"Total Score: {scoring.get('total_score', 0)}")
#             st.write(f"Percentage: {scoring.get('percentage', 0):.2f}%")
#             st.write(f"Confidence: {scoring.get('confidence', 0):.3f}")
        
#         with col2:
#             st.write("**Subject Scores:**")
#             for subject, score in scoring.get('subject_scores', {}).items():
#                 st.write(f"{subject}: {score}")
        
#         # Answer details
#         if 'detailed_results' in scoring:
#             st.subheader("📝 Answer Details")
            
#             detailed_results = scoring['detailed_results']
#             answer_data = []
            
#             for q_num, q_result in detailed_results.items():
#                 answer_data.append({
#                     'Question': q_num,
#                     'Student Answer': ', '.join(q_result.get('student_answer', [])) or 'No answer',
#                     'Correct Answer': ', '.join(q_result.get('correct_answer', [])),
#                     'Status': '✅' if q_result.get('is_correct', False) else '❌',
#                     'Score': q_result.get('marks_awarded', 0)
#                 })
            
#             df_answers = pd.DataFrame(answer_data)
            
#             # Filter options
#             col1, col2 = st.columns(2)
#             with col1:
#                 show_only = st.selectbox(
#                     "Show only:",
#                     ["All", "Correct", "Incorrect", "Unanswered"]
#                 )
            
#             if show_only == "Correct":
#                 df_answers = df_answers[df_answers['Status'] == '✅']
#             elif show_only == "Incorrect":
#                 df_answers = df_answers[df_answers['Status'] == '❌']
#             elif show_only == "Unanswered":
#                 df_answers = df_answers[df_answers['Student Answer'] == 'No answer']
            
#             st.dataframe(df_answers, width='stretch', height=400)

# def download_results(batch_results, config_options):
#     """Provide download options for results"""
#     if not batch_results:
#         return
    
#     st.header("💾 Download Results")
    
#     col1, col2, col3 = st.columns(3)
    
#     # JSON download
#     with col1:
#         if st.button("📄 Download JSON", width='stretch'):
#             json_str = json.dumps(batch_results, indent=2, default=str)
#             st.download_button(
#                 label="Download Complete Results (JSON)",
#                 data=json_str,
#                 file_name=f"omr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                 mime="application/json"
#             )
    
#     # CSV download
#     with col2:
#         if st.button("📊 Download CSV", width='stretch'):
#             # Prepare CSV data
#             csv_data = []
#             for result in batch_results.get('individual_results', []):
#                 if result.get('success') and 'scoring_result' in result:
#                     scoring = result['scoring_result']
#                     row = {
#                         'Image_Name': result['image_name'],
#                         'Total_Score': scoring.get('total_score', 0),
#                         'Percentage': scoring.get('percentage', 0),
#                         'Processing_Time': result.get('processing_time', 0),
#                         'Confidence': scoring.get('confidence', 0)
#                     }
#                     # Add subject scores
#                     for subject, score in scoring.get('subject_scores', {}).items():
#                         row[f'{subject}_Score'] = score
#                     csv_data.append(row)
#             if csv_data:
#                 df = pd.DataFrame(csv_data)
#                 csv_str = df.to_csv(index=False)
#                 st.download_button(
#                     label="Download Scores (CSV)",
#                     data=csv_str,
#                     file_name=f"omr_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                     mime="text/csv"
#                 )
    
#     # Summary report download
#     with col3:
#         if st.button("📈 Download Summary", width='stretch'):
#             summary_data = {
#                 'batch_summary': batch_results.get('batch_summary', {}),
#                 'scoring_report': batch_results.get('scoring_report', {}),
#                 'accuracy_metrics': batch_results.get('accuracy_metrics', {})
#             }
#             summary_str = json.dumps(summary_data, indent=2, default=str)
#             st.download_button(
#                 label="Download Summary Report (JSON)",
#                 data=summary_str,
#                 file_name=f"omr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                 mime="application/json"
#             )

# def display_help_section():
#     """Display help and instructions"""
#     with st.expander("❓ Help & Instructions", expanded=False):
#         st.markdown("""
#         ### 📋 How to Use the OMR Evaluation System
        
#         1. **Upload Images**: Use the file uploader to select multiple OMR sheet images
#         2. **Configure Settings**: Adjust processing options in the sidebar
#         3. **Process**: Click the "Process OMR Sheets" button to start evaluation
#         4. **Review Results**: Check the processing summary and individual results
#         5. **Download**: Export results in your preferred format
        
#         ### 📁 Supported File Formats
#         - JPG/JPEG
#         - PNG
#         - TIFF
#         - BMP
        
#         ### ⚙️ Configuration Options
#         - **Debug Mode**: Saves intermediate processing images for troubleshooting
#         - **Answer Key Set**: Choose the correct answer key or use auto-detection
#         - **Confidence Threshold**: Minimum confidence level for bubble classification
        
#         ### 📊 Output Formats
#         - **JSON**: Complete processing results with all metadata
#         - **CSV**: Scoring results in spreadsheet format
#         - **Summary**: High-level statistics and analytics
        
#         ### 🎯 Answer Key Format
#         The system expects answer keys in Excel format with separate sheets for each set (Set A, Set B).
#         Each sheet should have columns for question numbers and correct answers.
        
#         ### 🔧 Troubleshooting
#         - Ensure images are clear and well-lit
#         - Check that OMR sheets are properly aligned
#         - Use debug mode to identify processing issues
#         - Verify answer key format and file path
#         """)

# def main():
#     """Main application function"""
#     initialize_session_state()
    
#     # Display header
#     display_header()
    
#     # Sidebar configuration
#     config_options = sidebar_configuration()
    
#     # Load processor
#     if not load_processor():
#         st.error("Failed to initialize the system. Please check configuration and try again.")
#         return
    
#     # Main content area
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # File upload section
#         uploaded_files = file_upload_section()
        
#         # Process button
#     if uploaded_files and st.button("🚀 Process OMR Sheets", type="primary", width='stretch'):
#             # Clear previous results
#             st.session_state.batch_results = None
            
#             # Process images
#             results = process_images(uploaded_files, config_options)
            
#             if results:
#                 st.session_state.batch_results = results
#                 st.rerun()
    
#     with col2:
#         # Help section
#         display_help_section()
        
#         # System status
#         st.subheader("🔧 System Status")
#         if st.session_state.config_loaded:
#             st.success("✅ System ready")
#         else:
#             st.warning("⚠️ System not initialized")
        
#         # Quick stats
#         if st.session_state.processor and hasattr(st.session_state.processor, 'answer_keys'):
#             answer_keys = st.session_state.processor.answer_keys
#             if answer_keys:
#                 st.info(f"📋 Answer keys loaded: {', '.join(answer_keys.keys()).upper()}")
#             else:
#                 st.warning("⚠️ No answer keys loaded")
    
#     # Display results if available
#     if st.session_state.batch_results:
#         st.markdown("---")
        
#         # Results summary
#         display_results_summary(st.session_state.batch_results)
        
#         # Scoring results
#         display_scoring_results(st.session_state.batch_results)
        
#         # Individual results
#         display_individual_results(st.session_state.batch_results)
        
#         # Download section
#         st.markdown("---")
#         download_results(st.session_state.batch_results, config_options)
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         "<div style='text-align: center; color: #666666; font-size: 0.9em;'>"
#         "OMR Evaluation System | Built with Streamlit and OpenCV"
#         "</div>",
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()















"""
OMR Evaluation System with Login Authentication
Integrated Streamlit application with authentication and OMR processing
"""

import os
import streamlit as st
import sys
import tempfile
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import numpy as np

# Add scripts directory to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from scripts.run_batch import OMRBatchProcessor
from scripts.utils import load_config, get_image_files

# Page configuration
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'logged_in': False,
        'username': None,
        'processor': None,
        'batch_results': None,
        'config_loaded': False,
        'debug_mode': True,
        'debug_session_dir': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Authentication functions
def show_login_page():
    """Display login page"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #1f77b4;'>🔒 OMR Evaluation System</h1>
        <p style='color: #666; font-size: 1.1rem;'>Secure Login Required</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login Credentials")
        
        username = st.text_input('👤 Username', placeholder="Enter username")
        password = st.text_input('🔑 Password', type='password', placeholder="Enter password")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            login_btn = st.button('🚀 Login', type='primary', use_container_width=True)
        
        # Demo credentials info
        with st.expander("ℹ️ Need Help?"):
            st.info("""
            **Demo Credentials:**
            - **Username:** `admin`
            - **Password:** `admin`
            
            **For Production:**
            - Update credentials in the code
            - Consider using environment variables
            - Implement proper user management
            """)
    
    if login_btn:
        if username == 'admin' and password == 'admin':
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['login_time'] = datetime.now()
            st.success(f'✅ Welcome, {username}! Login successful!')
            st.rerun()
        else:
            st.error('❌ Invalid username or password. Please try again.')
            st.info("**Demo credentials: username=`admin`, password=`admin`**")

def show_logout_button():
    """Display logout functionality"""
    if st.session_state.get('logged_in', False):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👋 **Logged in as:** {st.session_state.get('username', 'User')}")
        st.sidebar.markdown(f"⏰ **Since:** {st.session_state.get('login_time', datetime.now()).strftime('%H:%M:%S')}")
        
        if st.sidebar.button('🚪 Logout', type='secondary'):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key not in ['_sentry_event_id']:  # Preserve Streamlit internal state
                    del st.session_state[key]
            st.session_state['logged_in'] = False
            st.rerun()

def show_main_app():
    """Display the main OMR application"""
    # Custom CSS for the main app
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .success-message {
            color: #28a745;
            font-weight: bold;
        }
        .error-message {
            color: #dc3545;
            font-weight: bold;
        }
        .warning-message {
            color: #ffc107;
            font-weight: bold;
        }
        .user-info {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2196f3;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Welcome message with user info
    st.markdown("""
    <div class='user-info'>
        <h3>👋 Welcome back, {}!</h3>
        <p>OMR Evaluation System - Enhanced Detection Pipeline v2.0</p>
        <small>Session started: {}</small>
    </div>
    """.format(
        st.session_state.get('username', 'User'),
        st.session_state.get('login_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
    ), unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📋 OMR Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

# Main application functions
def load_processor():
    """Load and initialize the OMR processor"""
    try:
        if st.session_state.processor is None:
            with st.spinner("🔄 Initializing OMR Processing Engine..."):
                st.session_state.processor = OMRBatchProcessor(
                    config_path="configs/config.yaml",
                    debug=st.session_state.get('debug_mode', True)
                )
                st.session_state.config_loaded = True
                st.success("✅ OMR Engine initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize OMR system: {str(e)}")
        st.info("**Troubleshooting steps:**")
        st.write("• Check if `configs/config.yaml` exists")
        st.write("• Verify `data/answer_keys.xlsx` is present")
        st.write("• Ensure all `scripts/` modules are available")
        return False

def sidebar_configuration():
    """Display sidebar configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox(
        "🔧 Enable Debug Mode", 
        value=st.session_state.get('debug_mode', True),
        help="""**Recommended for analysis:**
        - Generates detection overlays with bubble visualization
        - Saves intermediate processing images
        - Creates comprehensive debug logs
        - Essential for quality assessment"""
    )
    st.session_state.debug_mode = debug_mode
    
    if debug_mode:
        st.sidebar.success("✅ Debug mode: **ACTIVE** - Full analysis enabled")
    else:
        st.sidebar.warning("⚠️ Debug mode: **OFF** - Limited analysis")
    
    # Processing options
    st.sidebar.header("🎯 Processing Options")
    
    # Answer key selection
    answer_key_options = ["Auto-detect from filename", "Set A", "Set B"]
    selected_key = st.sidebar.selectbox("📋 Answer Key Set", answer_key_options)
    
    # Quality thresholds
    st.markdown("### 🎚️ Detection Sensitivity")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        help="**Lower values** = more sensitive detection, **Higher values** = more selective"
    )
    
    # Processing mode
    st.markdown("### ⚡ Processing Mode")
    processing_mode = st.sidebar.radio(
        "Select mode:",
        ["🚀 Full Analysis (Recommended)", "⚡ Quick Scan", "🔍 Debug Only"],
        index=0,
        help="Full Analysis: Complete processing with scoring\nQuick Scan: Basic detection only\nDebug Only: Analysis without scoring"
    )
    
    # File format options
    st.sidebar.header("💾 Export Options")
    export_formats = st.sidebar.multiselect(
        "Export Formats",
        ["📊 CSV (Scores)", "📄 JSON (Complete)", "📈 Summary Report"],
        default=["📊 CSV (Scores)", "📄 JSON (Complete)"]
    )
    
    # Debug output location
    if debug_mode:
        st.sidebar.header("📁 Debug Output")
        debug_location = st.sidebar.radio(
            "Save analysis files to:",
            ["💾 Persistent logs folder", "🗑️ Temporary session folder"],
            index=0
        )
    
    return {
        'debug_mode': debug_mode,
        'answer_key_set': selected_key,
        'confidence_threshold': confidence_threshold,
        'processing_mode': processing_mode,
        'export_formats': export_formats,
        'debug_location': debug_location if debug_mode else 'temp'
    }

def file_upload_section():
    """Enhanced file upload section with validation"""
    st.header("📁 Upload OMR Sheets")
    st.markdown("*Upload multiple OMR sheets for batch processing*")
    
    # File uploader with enhanced options
    uploaded_files = st.file_uploader(
        "Choose OMR sheet images",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        accept_multiple_files=True,
        help="**Supported formats:** JPG, PNG, TIFF, BMP | **Max size:** 15MB per image"
    )
    
    if uploaded_files:
        # Validate files
        valid_files = []
        invalid_files = []
        
        for uploaded_file in uploaded_files:
            try:
                # Check file size
                if uploaded_file.size > 15 * 1024 * 1024:  # 15MB
                    invalid_files.append(f"{uploaded_file.name} (too large)")
                    continue
                
                # Check if it's actually an image
                uploaded_file.seek(0)
                try:
                    Image.open(uploaded_file).verify()
                    uploaded_file.seek(0)  # Reset pointer
                    valid_files.append(uploaded_file)
                except:
                    invalid_files.append(f"{uploaded_file.name} (invalid image)")
                    continue
                    
            except Exception as e:
                invalid_files.append(f"{uploaded_file.name} (error: {str(e)[:30]})")
        
        # Display results
        if valid_files:
            st.success(f"✅ **{len(valid_files)} valid file(s)** ready for processing")
            
            # File list
            with st.expander(f"📋 File Details ({len(valid_files)} files)", expanded=True):
                for i, file in enumerate(valid_files):
                    col1, col2, col3 = st.columns([3, 1.5, 1])
                    with col1:
                        st.markdown(f"**{file.name}**")
                    with col2:
                        st.write(f"{file.size / 1024:.1f} KB")
                    with col3:
                        if st.button(f"👁️", key=f"preview_{i}", help="Preview image", use_container_width=True):
                            preview_image(file)
                
                if invalid_files:
                    st.warning(f"⚠️ **{len(invalid_files)} file(s) skipped:**")
                    for invalid_file in invalid_files[:3]:  # Show first 3
                        st.write(f"• {invalid_file}")
                    if len(invalid_files) > 3:
                        st.write(f"... and {len(invalid_files) - 3} more")
        else:
            st.error("❌ No valid image files uploaded")
            st.info("Please upload JPG, PNG, TIFF, or BMP files under 15MB each")
            return None
        
        return valid_files
    
    return None

def preview_image(uploaded_file):
    """Enhanced image preview with error handling"""
    try:
        # Create a copy to avoid file pointer issues
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset pointer
        
        # Save temporarily for PIL
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            image = Image.open(tmp_path)
            # Get original dimensions
            orig_width, orig_height = image.size
            
            # Calculate display size (max 600px width, maintain aspect ratio)
            display_width = 600
            if orig_width > display_width:
                ratio = display_width / orig_width
                new_height = int(orig_height * ratio)
                image = image.resize((display_width, new_height), Image.Resampling.LANCZOS)
            
            st.image(image, caption=f"Preview: {uploaded_file.name}", use_column_width=False)
            st.caption(f"Original size: {orig_width}×{orig_height} | File: {uploaded_file.size / 1024:.1f} KB")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        st.error(f"❌ Error previewing image: {str(e)}")
        st.info("This might be due to file corruption or unsupported format")

def process_images(uploaded_files, config_options):
    """Enhanced image processing with progress tracking"""
    if not uploaded_files:
        return None
    
    # Create persistent debug directory
    debug_base_dir = None
    if config_options['debug_mode']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_base_dir = os.path.join("logs", f"debug_session_{timestamp}")
        os.makedirs(debug_base_dir, exist_ok=True)
        st.info(f"🔧 **Debug session created:** `{debug_base_dir}`")
        st.session_state.debug_session_dir = debug_base_dir
    
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temporary directory
        temp_paths = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text(f"📁 Preparing {len(uploaded_files)} file(s)...")
        for i, uploaded_file in enumerate(uploaded_files):
            # Reset file pointer for reading
            uploaded_file.seek(0)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            temp_paths.append(temp_path)
            
            # Update progress (keep between 0.0 and 1.0)
            progress = ((i + 1) / len(uploaded_files)) * 0.2  # 20% for file prep
            progress_bar.progress(progress)
        
        progress_text.text("✅ Files prepared successfully!")
        
        # Initialize processor
        progress_text.text("🔧 Initializing OMR Engine...")
        processor = OMRBatchProcessor(
            config_path="configs/config.yaml",
            debug=config_options['debug_mode']
        )
        
        # Set debug directory
        if debug_base_dir:
            processor.debug_dir = debug_base_dir
        
        progress_bar.progress(25)
        progress_text.text("🚀 Starting OMR processing...")
        
        # Process batch
        try:
            results = processor.process_batch(temp_dir)
            progress_bar.progress(100)
            progress_text.text("✅ Processing completed!")
            
            if 'error' in results:
                st.error(f"❌ **Processing failed:** {results['error']}")
                return None
            else:
                summary = results['batch_summary']
                st.success(f"🎉 **Processing complete!** {summary['successful']}/{summary['total_processed']} successful")
                return results
                
        except Exception as e:
            st.error(f"❌ **Critical error during processing:** {str(e)}")
            progress_text.text("❌ Processing failed")
            return None

def display_results_summary(batch_results):
    """Enhanced results summary with better metrics"""
    if not batch_results or 'batch_summary' not in batch_results:
        return
    
    st.header("📊 Processing Summary")
    summary = batch_results['batch_summary']
    
    # Enhanced metrics layout
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📁 Total Images",
            summary['total_processed'],
            help="Number of images submitted for processing"
        )
    
    with col2:
        success_rate = summary['success_rate']
        color = "normal" if success_rate >= 90 else "inverse" if success_rate < 70 else "off"
        st.metric(
            "✅ Successful",
            summary['successful'],
            delta=f"{success_rate:.0f}%",
            delta_color=color,
            help="Images processed without critical errors"
        )
    
    with col3:
        avg_time = summary['average_time_per_image']
        st.metric(
            "⚡ Avg Time/Image",
            f"{avg_time:.1f}s",
            help="Average processing time per image"
        )
    
    with col4:
        failed_pct = (summary['failed'] / summary['total_processed'] * 100) if summary['total_processed'] > 0 else 0
        st.metric(
            "❌ Failed",
            summary['failed'],
            delta=f"{failed_pct:.1f}%" if failed_pct > 0 else "0%",
            delta_color="inverse",
            help="Images with processing errors"
        )
    
    with col5:
        total_time = summary['total_processing_time']
        st.metric(
            "⏱️ Total Time",
            f"{total_time:.0f}s",
            help="Complete batch processing duration"
        )
    
    # Processing efficiency
    if summary['successful'] > 0:
        st.markdown("---")
        efficiency_col1, efficiency_col2, efficiency_col3 = st.columns(3)
        
        with efficiency_col1:
            questions_total = sum(
                r['pipeline_stages'].get('grid_extraction', {}).get('questions_detected', 0)
                for r in batch_results['individual_results']
                if r.get('success')
            )
            avg_questions = questions_total / summary['successful']
            st.metric("📝 Avg Questions", f"{avg_questions:.0f}", "per sheet")
        
        with efficiency_col2:
            bubbles_total = sum(
                r['pipeline_stages'].get('classification', {}).get('total_bubbles', 0)
                for r in batch_results['individual_results']
                if r.get('success')
            )
            avg_bubbles = bubbles_total / summary['successful']
            st.metric("🔵 Avg Bubbles", f"{avg_bubbles:.0f}", "per sheet")
        
        with efficiency_col3:
            avg_confidence = np.mean([
                r['pipeline_stages'].get('classification', {}).get('average_confidence', 0)
                for r in batch_results['individual_results']
                if r.get('success')
            ])
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")

def display_scoring_results(batch_results):
    """Enhanced scoring results display"""
    if not batch_results or 'scoring_report' not in batch_results:
        return
    
    st.header("🎯 Scoring & Analytics")
    
    scoring_report = batch_results['scoring_report']
    
    if 'overall_statistics' in scoring_report:
        stats = scoring_report['overall_statistics']
        
        # Overall performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = stats['average_score']
            st.metric("📊 Average Score", f"{avg_score:.1f}/80")
        
        with col2:
            avg_pct = stats['average_percentage']
            color = "normal" if avg_pct >= 70 else "inverse" if avg_pct < 50 else "off"
            st.metric("📈 Average %", f"{avg_pct:.1f}%", delta_color=color)
        
        with col3:
            st.metric("🏆 Highest Score", f"{stats['highest_score']:.1f}")
        
        with col4:
            st.metric("📉 Lowest Score", f"{stats['lowest_score']:.1f}")
        
        # Performance distribution
        if 'grade_distribution' in scoring_report:
            st.subheader("📊 Performance Distribution")
            
            grades = list(scoring_report['grade_distribution'].keys())
            counts = list(scoring_report['grade_distribution'].values())
            total_students = sum(counts)
            
            # Create enhanced bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=grades,
                    y=counts,
                    marker_color=['#4CAF50', '#8BC34A', '#FFEB3B', '#FF9800', '#F44336'],
                    text=[f"{c/total_students*100:.0f}%" for c in counts],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Students: %{y}<br>Percentage: %{text}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Grade Distribution",
                xaxis_title="Grade",
                yaxis_title="Number of Students",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Subject-wise performance
        if 'subject_statistics' in scoring_report:
            st.subheader("📚 Subject Performance Analysis")
            
            subject_data = []
            for subject, subject_stats in scoring_report['subject_statistics'].items():
                max_possible = subject_stats.get('max_possible_score', 20)
                avg_pct = (subject_stats['average_score'] / max_possible * 100) if max_possible > 0 else 0
                
                subject_data.append({
                    'Subject': subject,
                    'Avg Score': f"{subject_stats['average_score']:.1f}/{max_possible}",
                    'Avg %': f"{avg_pct:.1f}%",
                    'Best': f"{subject_stats['highest_score']:.1f}",
                    'Worst': f"{subject_stats['lowest_score']:.1f}"
                })
            
            df_subjects = pd.DataFrame(subject_data)
            st.dataframe(df_subjects, use_container_width=True)
            
            # Subject performance chart
            fig = px.bar(
                pd.DataFrame(subject_data),
                x='Subject',
                y='Avg %',
                text='Avg %',
                title="Subject Performance Comparison",
                color='Avg %',
                color_continuous_scale='RdYlGn',
                text_auto='.1f'
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

def display_individual_results(batch_results):
    """Enhanced individual results display"""
    if not batch_results or 'individual_results' not in batch_results:
        return
    
    st.header("🔍 Individual Sheet Analysis")
    
    individual_results = batch_results['individual_results']
    
    # Create tabs
    tab1, tab2 = st.tabs(["✅ Successful Sheets", "❌ Failed Sheets"])
    
    with tab1:
        successful_results = [r for r in individual_results if r.get('success', False)]
        
        if successful_results:
            # Summary table
            summary_data = []
            for result in successful_results:
                scoring = result.get('scoring_result', {})
                grid_stage = result['pipeline_stages'].get('grid_extraction', {})
                class_stage = result['pipeline_stages'].get('classification', {})
                
                row = {
                    'Sheet': result['image_name'][:25] + "..." if len(result['image_name']) > 25 else result['image_name'],
                    'Questions': grid_stage.get('questions_detected', 0),
                    'Bubbles': class_stage.get('total_bubbles', 0),
                    'Marked': class_stage.get('marked_bubbles', 0),
                    'Score': f"{scoring.get('total_score', 0):.0f}/{scoring.get('max_possible_score', 80):.0f}",
                    'Grade': f"{scoring.get('percentage', 0):.0f}%",
                    'Time': f"{result.get('processing_time', 0):.1f}s",
                    'Confidence': f"{class_stage.get('average_confidence', 0):.2f}"
                }
                summary_data.append(row)
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, width='stretch', height=200)
            
            # Individual sheet details
            for result_idx, result in enumerate(successful_results):
                with st.expander(
                    f"📋 {result['image_name']} • "
                    f"{result['pipeline_stages'].get('grid_extraction', {}).get('questions_detected', 0)} Q • "
                    f"{result['pipeline_stages'].get('classification', {}).get('marked_bubbles', 0)}/{result['pipeline_stages'].get('classification', {}).get('total_bubbles', 0)} marked",
                    expanded=(result_idx < 2)  # Expand first 2 by default
                ):
                    # Image comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🖼️ Original Sheet")
                        orig_img_path = result.get('image_path')
                        if orig_img_path and os.path.exists(orig_img_path):
                            try:
                                orig_img = Image.open(orig_img_path)
                                if orig_img.size[0] > 500:
                                    orig_img.thumbnail((500, 700), Image.Resampling.LANCZOS)
                                st.image(orig_img, caption="Original OMR Sheet", width='stretch')
                            except Exception as e:
                                st.warning(f"Could not display original: {e}")
                        else:
                            st.warning("Original image not available")
                    
                    with col2:
                        st.markdown("### 🎯 Detection Overlay")
                        overlay_found = False
                        
                        # Multiple overlay search locations
                        overlay_paths = [
                            result.get('full_overlay_path'),
                            # Check debug directories
                        ]
                        
                        # Search in debug directory
                        debug_dir = result.get('debug_dir')
                        if debug_dir and os.path.isdir(debug_dir):
                            for ext in ['.png', '.jpg', '.jpeg']:
                                overlay_candidate = os.path.join(debug_dir, f"overlay_full_sheet{ext}")
                                if os.path.exists(overlay_candidate):
                                    overlay_paths.append(overlay_candidate)
                            
                            # Check classification subdir
                            class_dir = os.path.join(debug_dir, 'classification')
                            if os.path.exists(class_dir):
                                for ext in ['.png', '.jpg', '.jpeg']:
                                    overlay_candidate = os.path.join(class_dir, f"overlay_full_sheet{ext}")
                                    if os.path.exists(overlay_candidate):
                                        overlay_paths.append(overlay_candidate)
                        
                        # Try each path
                        for overlay_path in overlay_paths:
                            if overlay_path and os.path.exists(overlay_path):
                                try:
                                    overlay_img = Image.open(overlay_path)
                                    if overlay_img.size[0] > 500:
                                        overlay_img.thumbnail((500, 700), Image.Resampling.LANCZOS)
                                    st.image(overlay_img, caption=f"Detection Results\n{os.path.basename(overlay_path)}", width='stretch')
                                    overlay_found = True
                                    st.success(f"✅ Overlay loaded: {os.path.basename(overlay_path)}")
                                    break
                                except Exception as e:
                                    st.warning(f"Could not load overlay {os.path.basename(overlay_path)}: {e}")
                        
                        if not overlay_found:
                            st.warning("❌ **No overlay found**")
                            st.info("""
                            **This can happen when:**
                            1. Debug mode was disabled during processing
                            2. Processing completed before overlays were generated
                            3. File system permissions issue
                            
                            **To generate overlays:**
                            1. Enable Debug Mode in sidebar
                            2. Re-process the sheet
                            3. Check `logs/debug_session_*/` folder
                            
                            **If you see this message repeatedly:**
                            - Ensure your OMR pipeline always saves overlays when debug mode is on
                            - Add error handling to log overlay generation failures
                            - Check for file permission issues in the logs directory
                            - If overlays are not generated, review the detection/classification steps for errors
                            """)
                    
                    # Enhanced statistics
                    st.markdown("### 📊 Processing Metrics")
                    grid_stage = result['pipeline_stages'].get('grid_extraction', {})
                    class_stage = result['pipeline_stages'].get('classification', {})
                    score_stage = result.get('scoring_result', {})
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        questions = grid_stage.get('questions_detected', 0)
                        st.metric("📝 Questions", questions, "expected: 100")
                    with col2:
                        total_bubbles = class_stage.get('total_bubbles', 0)
                        st.metric("🔵 Total Bubbles", total_bubbles)
                    with col3:
                        marked = class_stage.get('marked_bubbles', 0)
                        mark_rate = (marked/total_bubbles*100) if total_bubbles > 0 else 0
                        st.metric("🟢 Marked Bubbles", f"{marked}/{total_bubbles}", f"{mark_rate:.0f}%")
                    with col4:
                        avg_conf = class_stage.get('average_confidence', 0)
                        st.metric("🎯 Confidence", f"{avg_conf:.2f}")
                    
                    # Scoring results
                    if score_stage and 'error' not in score_stage:
                        st.markdown("### 🎯 Scoring Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            total_score = score_stage.get('total_score', 0)
                            max_score = score_stage.get('max_possible_score', 80)
                            percentage = score_stage.get('percentage', 0)
                            st.metric("📊 Final Score", f"{total_score}/{max_score}")
                            st.metric("📈 Percentage", f"{percentage:.1f}%")
                        
                        with col2:
                            stats = score_stage.get('statistics', {})
                            st.metric("✅ Correct", stats.get('correct_answers', 0))
                            st.metric("❓ Answered", stats.get('questions_answered', 0))
                            st.metric("📝 Total", stats.get('total_questions', 100))
                        
                        # Subject breakdown
                        subject_scores = score_stage.get('subject_scores', {})
                        if subject_scores:
                            st.markdown("#### 📚 Subject Breakdown")
                            subject_cols = st.columns(min(5, len(subject_scores)))
                            for i, (subject, score) in enumerate(subject_scores.items()):
                                with subject_cols[i % len(subject_cols)]:
                                    max_sub = 20  # Assuming 20 questions per subject
                                    sub_pct = (score/max_sub*100) if max_sub > 0 else 0
                                    st.metric(
                                        subject[:10], 
                                        f"{score:.0f}/{max_sub}",
                                        f"{sub_pct:.0f}%"
                                    )
                    
                    # Detected answers preview
                    if 'marked_answers' in result and result['marked_answers']:
                        st.markdown("### ✅ Detected Student Answers")
                        marked_answers = result['marked_answers']
                        answer_preview = []
                        
                        # Get first 15 answered questions
                        answered_questions = [
                            (q, opts) for q, opts in sorted(marked_answers.items()) 
                            if opts and q <= 50  # Limit to first 50 questions
                        ][:15]
                        
                        for q_num, options in answered_questions:
                            option_letters = [chr(65 + opt) for opt in sorted(options)]
                            answer_preview.append(f"**Q{q_num}:** {', '.join(option_letters)}")
                        
                        if answer_preview:
                            st.markdown("\n".join(answer_preview))
                            remaining = sum(1 for q, opts in marked_answers.items() if opts and q > 50)
                            if remaining > 0:
                                st.info(f"📊 ... and {remaining} more answers detected (Q51+)")
                        else:
                            st.info("❓ No clear answers detected in first 50 questions")
                    else:
                        st.warning("No marked answers detected or available")
                    
                    # Detection quality assessment
                    grid_stage = result['pipeline_stages'].get('grid_extraction', {})
                    issues = grid_stage.get('issues', [])
                    
                    if issues:
                        st.markdown("### ⚠️ Detection Quality Assessment")
                        quality_score = 100
                        
                        for issue in issues:
                            if "CRITICAL" in issue.upper():
                                quality_score -= 40
                                st.error(f"🔴 **Critical:** {issue}")
                            elif "LOW" in issue.upper() or "few" in issue.lower():
                                quality_score -= 25
                                st.warning(f"🟡 **Warning:** {issue}")
                            elif "high" in issue.lower() or "irregular" in issue.lower():
                                quality_score -= 10
                                st.warning(f"🔶 **Note:** {issue}")
                            else:
                                quality_score -= 5
                                st.info(f"ℹ️ {issue}")
                        
                        quality_score = max(0, quality_score)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Quality Score", f"{quality_score}%")
                        
                        if quality_score < 70:
                            st.markdown("""
                            ### 🔧 Improvement Recommendations
                            **For better results:**
                            1. **Improve lighting** - Ensure even illumination
                            2. **Clean sheets** - Remove smudges and marks
                            3. **Better alignment** - Straighten the OMR sheet
                            4. **Higher resolution** - Use 300 DPI scans
                            5. **Adjust sensitivity** - Lower confidence threshold
                            """)
    
    with tab2:
        failed_results = [r for r in individual_results if not r.get('success', False)]
        
        if failed_results:
            st.markdown(f"### ❌ Failed Processing ({len(failed_results)} sheets)")
            
            for result in failed_results:
                with st.expander(f"❌ {result['image_name']}", expanded=True):
                    st.error(f"**Sheet processing failed**")
                    
                    # Error details
                    if 'errors' in result:
                        st.markdown("### 🔴 Error Details")
                        for i, error in enumerate(result['errors'], 1):
                            st.error(f"{i}. {error}")
                    
                    # Show grid/cell debug images if grid extraction failed
                    failed_stage = None
                    if 'pipeline_stages' in result:
                        for stage, status in result['pipeline_stages'].items():
                            if not status.get('success', True):
                                failed_stage = stage
                                break
                    if failed_stage and 'grid' in failed_stage:
                        debug_dir = result.get('debug_dir')
                        if debug_dir:
                            st.markdown("#### 🟠 Grid Extraction Debug Images")
                            grid_img_path = os.path.join(debug_dir, "grid_extracted.jpg")
                            if os.path.exists(grid_img_path):
                                st.image(Image.open(grid_img_path), caption="Extracted Grid Region", width='stretch')
                            # Show cell debug images if present
                            cell_imgs = [f for f in os.listdir(debug_dir) if f.startswith("cell_") and f.endswith(".jpg")]
                            if cell_imgs:
                                st.markdown("**Extracted Cells:**")
                                for f in sorted(cell_imgs)[:10]:
                                    st.image(Image.open(os.path.join(debug_dir, f)), caption=f, width='stretch')
                            st.warning("No bubbles detected. Possible causes: misaligned sheet, wrong grid config, poor image quality, or blank region. Check debug images above for troubleshooting.")
                    
                    # Pipeline status
                    if 'pipeline_stages' in result:
                        st.markdown("### 🔄 Pipeline Breakdown")
                        failed_stage = None
                        for stage, status in result['pipeline_stages'].items():
                            icon = "✅" if status.get('success', False) else "❌"
                            stage_name = stage.replace('_', ' ').title()
                            
                            with st.container():
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.write(f"**{icon}**")
                                with col2:
                                    st.write(f"**{stage_name}**")
                                    if not status.get('success', True):
                                        if 'error' in status:
                                            st.error(f"**Failed:** {status['error']}")
                                            failed_stage = stage
                                        else:
                                            st.warning("**Failed** - No specific error details")
                        
                        if failed_stage:
                            st.error(f"**Primary failure:** {failed_stage.replace('_', ' ').title()}")
                            st.info("💡 **Common fixes:**")
                            if "preprocessing" in failed_stage:
                                st.write("• Check image format and quality")
                                st.write("• Ensure sufficient contrast")
                            elif "detection" in failed_stage:
                                st.write("• Verify sheet boundaries are clear")
                                st.write("• Try different image resolution")
                            elif "grid" in failed_stage:
                                st.write("• Improve lighting uniformity")
                                st.write("• Clean OMR sheet surface")
                            elif "classification" in failed_stage:
                                st.write("• Adjust confidence threshold")
                                st.write("• Check for printing quality issues")
                    
                    # File information
                    st.markdown("### 📄 File Information")
                    st.write(f"**Filename:** {result.get('image_name', 'Unknown')}")
                    st.write(f"**Attempted at:** {result.get('timestamp', 'Unknown')}")
                    
        else:
            st.success("🎉 **Perfect run!** All sheets processed successfully!")

def download_results(batch_results, config_options):
    """Enhanced download functionality"""
    if not batch_results:
        return
    
    st.header("💾 Export Results")
    st.markdown("*Download your processing results and analysis*")
    
    # Create download columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON Download
        json_str = json.dumps(batch_results, indent=2, default=str)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📄 Complete Analysis (JSON)",
            data=json_str,
            file_name=f"omr_complete_analysis_{timestamp}.json",
            mime="application/json",
            help="Full processing results with all metadata and debug info",
            width='stretch'
        )
    
    with col2:
        # CSV Download
        csv_data = []
        for result in batch_results.get('individual_results', []):
            if result.get('success') and 'scoring_result' in result:
                scoring = result['scoring_result']
                grid_stage = result['pipeline_stages'].get('grid_extraction', {})
                class_stage = result['pipeline_stages'].get('classification', {})
                
                row = {
                    'Sheet_Name': result['image_name'],
                    'Questions_Detected': grid_stage.get('questions_detected', 0),
                    'Total_Bubbles': class_stage.get('total_bubbles', 0),
                    'Marked_Bubbles': class_stage.get('marked_bubbles', 0),
                    'Detection_Confidence': class_stage.get('average_confidence', 0),
                    'Total_Score': scoring.get('total_score', 0),
                    'Percentage': scoring.get('percentage', 0),
                    'Processing_Time_s': result.get('processing_time', 0),
                    'Answer_Key_Set': result.get('answer_key_set', 'Unknown'),
                    'Status': 'Success'
                }
                
                # Add subject scores
                for subject, score in scoring.get('subject_scores', {}).items():
                    row[f'{subject}_Score'] = score
                
                csv_data.append(row)
            elif not result.get('success'):
                # Add failed results
                row = {
                    'Sheet_Name': result.get('image_name', 'Unknown'),
                    'Questions_Detected': 0,
                    'Total_Bubbles': 0,
                    'Marked_Bubbles': 0,
                    'Detection_Confidence': 0,
                    'Total_Score': 0,
                    'Percentage': 0,
                    'Processing_Time_s': result.get('processing_time', 0),
                    'Answer_Key_Set': result.get('answer_key_set', 'Unknown'),
                    'Status': 'Failed',
                    'Error': '; '.join(result.get('errors', ['Unknown error']))
                }
                csv_data.append(row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📊 Scores & Analysis (CSV)",
                data=csv_str,
                file_name=f"omr_results_summary_{timestamp}.csv",
                mime="text/csv",
                help="Spreadsheet with scores, detection stats, and processing metrics",
                width='stretch'
            )
        else:
            st.warning("No data available for CSV export")
    
    with col3:
        # Summary Report
        summary_data = {
            'processing_summary': batch_results.get('batch_summary', {}),
            'scoring_analytics': batch_results.get('scoring_report', {}),
            'accuracy_metrics': batch_results.get('accuracy_metrics', {}),
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'debug_mode': config_options['debug_mode'],
                'confidence_threshold': config_options['confidence_threshold'],
                'username': st.session_state.get('username', 'Unknown')
            }
        }
        summary_str = json.dumps(summary_data, indent=2, default=str)
        st.download_button(
            label="📈 Executive Summary (JSON)",
            data=summary_str,
            file_name=f"omr_executive_report_{timestamp}.json",
            mime="application/json",
            help="High-level statistics and executive overview",
            width='stretch'
        )
    
    # Debug archive (if debug mode enabled)
    if config_options['debug_mode'] and 'debug_session_dir' in st.session_state:
        debug_dir = st.session_state.debug_session_dir
        if os.path.exists(debug_dir):
            with st.expander("📦 Download Debug Archive", expanded=False):
                st.info(f"**Debug session directory:** `{debug_dir}`")
                
                # Count debug files
                image_count = 0
                json_count = 0
                for root, dirs, files in os.walk(debug_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_count += 1
                        elif file.lower().endswith('.json'):
                            json_count += 1
                
                st.write(f"**Contents:** {image_count} images, {json_count} reports")
                
                # Create debug zip (simplified - just show download info)
                st.info("""
                **To download debug files:**
                1. Navigate to `{debug_dir}` in your file explorer
                2. Copy the entire `debug_session_*` folder
                3. Or use command line: `zip -r debug_archive.zip logs/debug_session_*`
                """.format(debug_dir=debug_dir))

def display_help_section():
    """Enhanced help and documentation"""
    with st.expander("📖 Help & Documentation", expanded=False):
        st.markdown("""
        # 🎓 OMR Evaluation System Guide
        
        ## 🚀 **Quick Start Guide**
        
        ### Step 1: Upload Your Sheets
        - Drag & drop or click to select OMR images
        - **Supported formats**: JPG, PNG, TIFF, BMP
        - **Recommended**: 300 DPI, good lighting, clean sheets
        
        ### Step 2: Configure Settings
        - **Debug Mode**: 🟢 **ENABLED** (recommended for analysis)
        - **Answer Key**: Auto-detect or select Set A/B
        - **Sensitivity**: Adjust based on sheet quality
        
        ### Step 3: Process & Analyze
        - Click **🚀 Process OMR Sheets**
        - Wait 2-5 minutes per sheet
        - Review **Detection Overlays** (green = marked, red = unmarked)
        - Check **Scoring Results** against answer key
        
        ### Step 4: Export Results
        - **CSV**: Spreadsheet for grading
        - **JSON**: Complete analysis data
        - **Summary**: Executive report
        
        ## 📊 **Understanding Results**
        
        ### Detection Overlay Guide
        | Color | Meaning | Confidence |
        |-------|---------|------------|
        | 🟢 **Green** | **MARKED** - Student selected this option | High |
        | 🔴 **Red** | **UNMARKED** - No selection detected | High |
        | 🟡 **Yellow** | **AMBIGUOUS** - Needs manual review | Low |
        | ⚪ **White** | Not detected as bubble | N/A |
        
        ### Quality Metrics
        - **Questions Detected**: Target 80-100 for full sheets
        - **Bubble Confidence**: >0.7 = reliable, <0.5 = review needed
        - **Detection Rate**: >80% = good quality scan
        
        ## 🎯 **Answer Key Setup**
        
        ### File Location
        ```
        data/
        └── answer_keys.xlsx
        ```
        
        ### Expected Format
        **Sheet names**: "Set A", "Set B"
        **Structure**:
        | Question | Answer |
        |----------|--------|
        | 1        | A      |
        | 2        | C      |
        | 3        | B,D    |  *(Multiple answers)*
        
        ## 🔧 **Troubleshooting Common Issues**
        
        ### Detection Problems
        | Issue | Symptoms | Solutions |
        |-------|----------|-----------|
        | **Few questions** | <30 detected | ✅ Lower confidence threshold<br>✅ Improve lighting<br>✅ Clean sheet |
        | **No overlays** | Empty right panel | ✅ Enable Debug Mode<br>✅ Re-process sheet |
        | **All bubbles marked** | 100% detection | ✅ Increase confidence threshold<br>✅ Check print quality |
        
        ### Scoring Issues
        | Issue | Symptoms | Solutions |
        |-------|----------|-----------|
        | **0% scores** | No matches | ✅ Verify answer key file<br>✅ Check Set A/B selection |
        | **Wrong answers** | Unexpected results | ✅ Review detection overlay<br>✅ Manual validation needed |
        
        ## 📈 **Performance Optimization**
        
        ### For Best Results
        1. **Scan Quality**: 300 DPI, grayscale, no compression
        2. **Lighting**: Even, no shadows or glare
        3. **Sheet Condition**: Clean, no smudges or folds
        4. **File Size**: 1-5MB per sheet optimal
        5. **Batch Size**: 1-10 sheets at a time
        
        ### Debug Files Generated
        ```
        logs/debug_session_YYYYMMDD_HHMMSS/
        ├── overlay_full_sheet.png      (Complete detection)
        ├── classification/
        │   ├── overlay_Q01.png         (Per-question)
        │   └── sample_bubbles/         (Individual analysis)
        ├── grid_extraction/
        │   ├── 14_all_detected_bubbles.jpg (All candidates)
        │   └── 15_grid_organization.jpg    (Layout analysis)
        └── extraction_summary.json     (Statistics)
        ```
        
        ## 🔒 **Security & Access**
        - **Current user**: `admin` (demo mode)
        - **Session timeout**: 30 minutes of inactivity
        - **Data storage**: Processing files in `logs/` and `output/`
        
        ## 📞 **Support**
        - **Debug files**: Share `logs/debug_session_*` folder
        - **Error logs**: Check browser console (F12)
        - **Contact**: [Your support contact here]
        
        ---
        *OMR Evaluation System v2.0 | Enhanced Detection & Analytics*
        *Built with Streamlit, OpenCV, and Machine Learning*
        """)

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Show logout button if logged in
    show_logout_button()
    
    # Authentication check
    if not st.session_state.get('logged_in', False):
        show_login_page()
        return
    
    # Show main application
    show_main_app()
    
    # Load OMR processor
    if not load_processor():
        st.error("❌ Failed to load OMR processing engine")
        st.info("Please check configuration files and try again")
        return
    
    # Sidebar configuration
    config_options = sidebar_configuration()
    
    # Main content area
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        # File upload and processing
        uploaded_files = file_upload_section()
        
        # Process button
        if uploaded_files:
            process_placeholder = st.empty()
            with process_placeholder.container():
                if st.button("🚀 Process OMR Sheets", type="primary", use_container_width=True):
                    # Clear previous results
                    st.session_state.batch_results = None
                    # Show processing status
                    with st.spinner(f"🔄 Processing {len(uploaded_files)} OMR sheet(s)..."):
                        results = process_images(uploaded_files, config_options)
                        if results and 'error' not in results:
                            st.session_state.batch_results = results
                            st.rerun()
                        else:
                            st.error("❌ Processing failed. Check the error details above.")
    
    with col_sidebar:
        # System status and help
        st.subheader("📊 System Status")
        
        if st.session_state.config_loaded and st.session_state.processor:
            st.success("✅ **OMR Engine:** Ready")
            
            # Processor info
            processor = st.session_state.processor
            if hasattr(processor, 'answer_keys') and processor.answer_keys:
                keys = list(processor.answer_keys.keys())
                st.success(f"📋 **Answer Keys:** {', '.join([k.upper() for k in keys])}")
            else:
                st.warning("⚠️ **Answer Keys:** Not loaded")
                st.info("Place `data/answer_keys.xlsx` in data folder")
            
            # Debug status
            debug_status = "🟢 ACTIVE" if config_options['debug_mode'] else "🔴 OFFLINE"
            st.info(f"🔧 **Debug Mode:** {debug_status}")
            
            if config_options['debug_mode'] and 'debug_session_dir' in st.session_state:
                debug_dir = st.session_state.debug_session_dir
                if debug_dir is not None and os.path.exists(debug_dir):
                    try:
                        file_count = len([f for f in os.listdir(debug_dir) if f.endswith(('.png', '.jpg', '.json'))])
                        st.success(f"📁 **Debug Files:** {file_count} generated")
                    except:
                        st.info("📁 Debug directory ready")
        else:
            st.warning("⚠️ **System:** Initializing...")
        
        # Help section
        display_help_section()
    
    # Display results section
    if st.session_state.get('batch_results'):
        st.markdown("---")
        
        # Results summary
        display_results_summary(st.session_state.batch_results)
        
        # Scoring analysis
        display_scoring_results(st.session_state.batch_results)
        
        # Individual results
        display_individual_results(st.session_state.batch_results)
        
        # Download section
        st.markdown("---")
        download_results(st.session_state.batch_results, config_options)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-size: 0.9em; padding: 1rem; border-top: 1px solid #eee;'>
        <strong>OMR Evaluation System v2.0</strong> | 
        Enhanced Detection & Analytics Pipeline<br>
        🔒 Secure • 📊 Comprehensive • 🎯 Accurate | 
        Built with ❤️ using Streamlit, OpenCV & Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()