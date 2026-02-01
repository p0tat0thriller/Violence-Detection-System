"""
Streamlit Web Application for Real-Time Violence Detection

Features:
- Live webcam feed or video upload
- Real-time skeleton visualization
- Violence detection with confidence scores
- Interactive controls and metrics display
"""

import streamlit as st
import cv2
import torch
import numpy as np
from collections import deque
import tempfile
import os
from pathlib import Path

# Import required modules
from skeleton_extractor import SkeletonExtractor
from ultralytics import YOLO

# Import SkeletonViolenceModel
from skeleton_model import SkeletonViolenceModel

# Page configuration
st.set_page_config(
    page_title="Violence Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load YOLOv8 and violence detection models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load skeleton extractor
    skeleton_extractor = SkeletonExtractor('yolov8n-pose.pt')
    
    # Load YOLO for person detection
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    
    # Load violence detection model
    model = None
    model_loaded = False
    project_root = Path(__file__).parent.parent
    model_path = project_root / "weights" / "best_skeleton_model.pth"
    
    if model_path.exists():
        try:
            model = SkeletonViolenceModel(
                num_keypoints=33,
                num_coords=3,
                hidden_size=384,
                num_layers=2,
                num_classes=2,
                dropout=0.3
            )
            checkpoint = torch.load(str(model_path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            model_loaded = True
        except Exception as e:
            st.warning(f"Could not load violence model: {str(e)}")
    else:
        st.warning("Violence detection model not found. Running in visualization-only mode.")
    
    return skeleton_extractor, yolo, model, model_loaded, device


def draw_skeleton(frame, pose_result):
    """Draw skeleton keypoints and connections on frame"""
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    
    color_skeleton = (0, 255, 0)
    color_keypoint = (0, 0, 255)
    
    if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
        for person_idx in range(len(pose_result.keypoints)):
            keypoints = pose_result.keypoints[person_idx].xy.cpu().numpy()[0]
            conf = pose_result.keypoints[person_idx].conf.cpu().numpy()[0] if pose_result.keypoints[person_idx].conf is not None else np.ones(17)
            
            # Draw skeleton connections
            for connection in skeleton_connections:
                start_idx, end_idx = connection
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    if conf[start_idx] > 0.5 and conf[end_idx] > 0.5:
                        start_point = tuple(map(int, keypoints[start_idx]))
                        end_point = tuple(map(int, keypoints[end_idx]))
                        cv2.line(frame, start_point, end_point, color_skeleton, 2)
            
            # Draw keypoints
            for idx, (x, y) in enumerate(keypoints):
                if conf[idx] > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, color_keypoint, -1)
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)


def process_video(video_source, skeleton_extractor, yolo, model, model_loaded, device, settings):
    """Process video and detect violence"""
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error("Could not open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle webcam (total_frames = 0 for live streams)
    is_webcam = total_frames == 0 or total_frames < 0
    
    # Processing variables
    skeleton_queue = deque(maxlen=settings['sequence_length'])
    prediction_history = deque(maxlen=settings['temporal_smoothing'])
    frame_count = 0
    prev_gray = None
    
    # Streamlit placeholders - create containers that stay in place
    video_placeholder = st.empty()
    
    # Detailed metrics in expandable section
    with st.expander("📊 Detailed Metrics", expanded=False):
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metric1_placeholder = metrics_col1.empty()
        metric2_placeholder = metrics_col2.empty()
        metric3_placeholder = metrics_col3.empty()
    
    progress_bar = st.progress(0)
    
    # Stop control using session state
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    
    stop_col1, stop_col2 = st.columns([1, 4])
    with stop_col1:
        if st.button("⏹️ Stop"):
            st.session_state.stop_processing = True
    with stop_col2:
        st.caption("Click Stop to end processing")
    
    # Display variables
    status_text = "Initializing..."
    status_color = (200, 200, 200)  # Gray
    violence_score = 0.0
    person_count = 0
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        output_frame = frame.copy()
        
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_score = 0.0
        if prev_gray is not None:
            frame_delta = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(frame_delta)
        prev_gray = gray
        
        # Person detection (every 3rd frame)
        if frame_count % 3 == 0:
            person_count = 0
            results = yolo(frame, verbose=False, stream=False)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= settings['person_confidence']:
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Skeleton visualization (every 2nd frame)
        if frame_count % 2 == 0:
            pose_results = skeleton_extractor.model(frame, verbose=False)
            if pose_results and settings['show_skeleton']:
                draw_skeleton(output_frame, pose_results[0])
        
        # Skeleton extraction for AI model (based on FRAME_STRIDE)
        if frame_count % 2 == 0:  # FRAME_STRIDE = 2
            skeleton = skeleton_extractor.extract_keypoints(frame)
            skeleton_queue.append(torch.tensor(skeleton, dtype=torch.float32))
        
        # Violence detection (when queue is full and every SKIP_INFERENCE frames)
        if len(skeleton_queue) == settings['sequence_length'] and frame_count % 5 == 0:  # SKIP_INFERENCE = 5
            if motion_score < settings['motion_threshold'] and person_count < 1:
                status_text = "Static Scene"
                status_color = (128, 128, 128)  # Gray
            else:
                # Prepare sequence and run inference
                sequence = torch.stack(list(skeleton_queue)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(sequence)
                    probs = torch.softmax(outputs, dim=1)
                    current_prediction = probs[0][1].item()
                    
                    # Temporal smoothing
                    prediction_history.append(current_prediction)
                    violence_score = np.mean(prediction_history)
                    
                    # Adjust threshold based on person count
                    adjusted_threshold = settings['confidence_threshold'] + 0.10  # CONFIDENCE_MARGIN
                    if person_count > 2:
                        adjusted_threshold *= 0.90
                    
                    # Determine status
                    if violence_score > adjusted_threshold:
                        status_text = f"VIOLENCE DETECTED ({violence_score:.0%})"
                        status_color = (0, 0, 255)  # Red in BGR
                    else:
                        status_text = f"Normal Activity ({violence_score:.0%})"
                        status_color = (0, 255, 0)  # Green in BGR
        
        # Draw overlay panel with text (like local version)
        # Create semi-transparent black panel at top
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
        
        # Main status text
        cv2.putText(output_frame, status_text, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Debug info line 1
        debug_line1 = f"Frame: {frame_count} | People: {person_count} | Motion: {motion_score:.1f}"
        cv2.putText(output_frame, debug_line1, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Debug info line 2
        debug_line2 = f"AI Score: {violence_score:.3f} | Threshold: {settings['confidence_threshold']:.1f}"
        cv2.putText(output_frame, debug_line2, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Convert to RGB for display
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(output_frame, channels="RGB", use_container_width=True)
        
        # Update metrics in place
        if is_webcam:
            metric1_placeholder.metric("Frame", f"{frame_count}")
        else:
            metric1_placeholder.metric("Frame", f"{frame_count}/{total_frames}")
        metric2_placeholder.metric("People Detected", person_count)
        metric3_placeholder.metric("Violence Confidence", f"{violence_score:.1%}", 
                                  help="Model's confidence that violence is occurring (0-100%)")
        
        # Update progress (only for video files, not webcam)
        if not is_webcam:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        # Check stop button
        if st.session_state.stop_processing:
            st.warning("Processing stopped by user")
            break
    
    cap.release()
    st.session_state.stop_processing = False  # Reset for next run
    st.success("Video processing complete!")


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">Violence Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model settings
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 8, 32, 24, help="Number of frames for LSTM")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.05, help="Violence detection threshold")
    temporal_smoothing = st.sidebar.slider("Temporal Smoothing", 5, 20, 10, help="Smoothing window size")
    
    # Detection settings
    st.sidebar.subheader("Detection Settings")
    motion_threshold = st.sidebar.slider("Motion Threshold", 0.0, 10.0, 3.0, 0.5, help="Minimum motion to trigger")
    person_confidence = st.sidebar.slider("Person Confidence", 0.0, 1.0, 0.45, 0.05, help="YOLO person detection confidence")
    
    # Visualization
    st.sidebar.subheader("Visualization")
    show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
    
    settings = {
        'sequence_length': sequence_length,
        'confidence_threshold': confidence_threshold,
        'temporal_smoothing': temporal_smoothing,
        'motion_threshold': motion_threshold,
        'person_confidence': person_confidence,
        'show_skeleton': show_skeleton
    }
    
    # Load models
    with st.spinner("Loading models..."):
        skeleton_extractor, yolo, model, model_loaded, device = load_models()
    
    st.sidebar.success(f"✅ Models loaded on {device}")
    
    # Main content
    tab1, tab2 = st.tabs(["📹 Upload Video", "🎥 Live Webcam"])
    
    with tab1:
        st.header("Upload Video for Analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            if st.button("Start Processing"):
                process_video(tfile.name, skeleton_extractor, yolo, model, model_loaded, device, settings)
                os.unlink(tfile.name)
    
    with tab2:
        st.header("Live Webcam Feed")
        st.info("🚧 Webcam support coming soon!")
        
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0)
        
        if st.button("Start Webcam"):
            process_video(camera_index, skeleton_extractor, yolo, model, model_loaded, device, settings)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About**
    
    Real-time violence detection using:
    - YOLOv8-Pose for skeleton extraction
    - Bidirectional LSTM for classification
    - Temporal smoothing for stability
    """)


if __name__ == "__main__":
    main()
