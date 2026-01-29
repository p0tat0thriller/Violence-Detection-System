"""
Real-Time Skeletal Violence Detection System

Uses YOLOv8-Pose for skeleton extraction and LSTM for violence classification.
Achieves 70.5% accuracy with 87% violence recall.

Architecture:
- YOLOv8-Pose: Extract 17 skeletal keypoints
- Feature Encoder: Compress skeleton features
- Bidirectional LSTM: Temporal modeling
- Classifier: Binary violence detection
"""

import cv2
import torch
import numpy as np
from collections import deque
import sys
import os
import traceback
from ultralytics import YOLO

# Import SkeletonViolenceModel
from skeleton_model import SkeletonViolenceModel

# Import skeleton extractor
from skeleton_extractor import SkeletonExtractor


# --- 1. CONFIGURATION ---
# 🔴 INPUT: Put the path to your video file here (Use forward slashes /)
# Use relative path from script location
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
VIDEO_SOURCE = os.path.join(_project_root, "testing", "f3.avi")

# MODEL CONFIG
SEQUENCE_LENGTH = 16            # Number of frames for LSTM
CONFIDENCE_THRESHOLD = 0.60   # Violence detection threshold (0-1)
                                # 0.60 = Balanced (80% recall, 60% accuracy)
                                # 0.70 = Fewer false alarms (75% precision, low recall)
                                # 0.50 = Catch all violence (100% recall, many false alarms)
SKIP_INFERENCE = 3              # Run AI every N frames for performance
FRAME_STRIDE = 2                # Sample every Nth frame for sequence

# DETECTION THRESHOLDS
MOTION_THRESHOLD = 3.0          # Minimum motion to trigger inference
PERSON_CONFIDENCE = 0.45        # YOLOv8 person detection confidence
TEMPORAL_SMOOTHING = 10         # Number of predictions to smooth (increased from 5)
MIN_PERSON_COUNT = 1            # Minimum people for violence check
CONFIDENCE_MARGIN = 0.10        # Require score > threshold + margin for positive


def calculate_motion_score(current_gray, prev_gray):
    """
    Calculate motion intensity between frames.
    
    Args:
        current_gray: Current grayscale frame
        prev_gray: Previous grayscale frame
        
    Returns:
        float: Motion score (higher = more motion)
    """
    if prev_gray is None:
        return 0.0
    
    frame_delta = cv2.absdiff(prev_gray, current_gray)
    return np.mean(frame_delta)


def draw_skeleton(frame, pose_result):
    """
    Draw skeleton keypoints and connections on frame.
    
    Args:
        frame: Frame to draw on
        pose_result: YOLOv8-Pose result object
    """
    # COCO skeleton connections (17 keypoints)
    skeleton_connections = [
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Eyes to ears
        (0, 5), (0, 6),  # Nose to shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10), # Right arm
        (5, 6),          # Shoulders
        (5, 11), (6, 12),# Shoulders to hips
        (11, 12),        # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16)   # Right leg
    ]
    
    # Colors for different body parts
    color_skeleton = (0, 255, 0)      # Green for bones
    color_keypoint = (0, 0, 255)      # Red for joints
    
    if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
        for person_idx in range(len(pose_result.keypoints)):
            keypoints = pose_result.keypoints[person_idx].xy.cpu().numpy()[0]  # Get keypoints
            conf = pose_result.keypoints[person_idx].conf.cpu().numpy()[0] if pose_result.keypoints[person_idx].conf is not None else np.ones(17)
            
            # Draw skeleton connections
            for connection in skeleton_connections:
                start_idx, end_idx = connection
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    # Check confidence for both points
                    if conf[start_idx] > 0.5 and conf[end_idx] > 0.5:
                        start_point = tuple(map(int, keypoints[start_idx]))
                        end_point = tuple(map(int, keypoints[end_idx]))
                        
                        # Draw line (bone)
                        cv2.line(frame, start_point, end_point, color_skeleton, 2)
            
            # Draw keypoints (joints)
            for idx, (x, y) in enumerate(keypoints):
                if conf[idx] > 0.5:  # Only draw confident keypoints
                    cv2.circle(frame, (int(x), int(y)), 4, color_keypoint, -1)
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)  # White outline


def main():
    """Main function for real-time skeletal violence detection"""
    
    # Device setup - Force GPU usage
    device = torch.device('cuda')
    print("=" * 70)
    print("🥊 SKELETAL VIOLENCE DETECTION SYSTEM")
    print("=" * 70)
    print(f"🚀 Device: {device}")
    
    # --- LOAD SKELETON EXTRACTOR ---
    print("\n📦 Loading YOLOv8-Pose model...")
    try:
        skeleton_extractor = SkeletonExtractor('yolov8n-pose.pt')
        print("✅ YOLOv8-Pose loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLOv8-Pose: {str(e)}")
        sys.exit(1)
    
    # --- LOAD YOLO FOR PERSON DETECTION ---
    print("\n📦 Loading YOLOv8 for person detection...")
    try:
        yolo = YOLO("yolov8n.pt")
        yolo.to(device)
        print("✅ YOLOv8 loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLOv8: {str(e)}")
        sys.exit(1)
    
    # --- LOAD VIOLENCE DETECTION MODEL ---
    print("\n📦 Loading Skeletal Violence Detection Model...")
    model = SkeletonViolenceModel(
        num_keypoints=33,
        num_coords=3,
        hidden_size=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    # Use relative path from script location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "weights", "best_skeleton_model.pth")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Skeletal Violence Model loaded successfully!")
        print(f"   Model from epoch: {checkpoint['epoch'] + 1}")
        print(f"   Validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"   Model parameters: {model.get_num_params():,}")
    except Exception as e:
        print(f"❌ Error loading violence model: {str(e)}")
        print(f"   Looking for: {model_path}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # --- VIDEO SETUP ---
    print(f"\n🎥 Opening video: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file")
        print(f"   Path: {VIDEO_SOURCE}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video opened successfully!")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"\n{'='*70}")
    print("Press 'q' to quit")
    print("=" * 70)
    
    # --- PROCESSING VARIABLES ---
    skeleton_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    prev_gray = None
    motion_score = 0.0
    
    # Temporal smoothing
    prediction_history = deque(maxlen=TEMPORAL_SMOOTHING)
    
    # Display variables
    status_text = "Initializing..."
    status_color = (255, 255, 0)  # Yellow
    violence_score = 0.0
    person_count = 0
    
    # --- MAIN PROCESSING LOOP ---
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video finished")
                break
            
            frame_count += 1
            output_frame = frame.copy()
            
            # --- 1. MOTION DETECTION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is not None:
                motion_score = calculate_motion_score(gray, prev_gray)
            prev_gray = gray
            
            # --- 2. PERSON DETECTION & SKELETON VISUALIZATION (YOLO) ---
            if frame_count % 3 == 0:  # Every 3rd frame for performance
                results = yolo(frame, verbose=False, stream=False)
                person_count = 0
                
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])
                            if confidence >= PERSON_CONFIDENCE:
                                person_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Draw bounding box
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), 
                                            (0, 255, 255), 2)
                                cv2.putText(output_frame, f"Person {confidence:.2f}", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, (0, 255, 255), 2)
            
            # --- 2b. SKELETON VISUALIZATION ---
            # Run YOLOv8-Pose for skeleton visualization
            if frame_count % 2 == 0:  # Every 2nd frame for performance
                pose_results = skeleton_extractor.model(frame, verbose=False)
                if pose_results:
                    draw_skeleton(output_frame, pose_results[0])
            
            # --- 3. SKELETON EXTRACTION FOR AI MODEL ---
            if frame_count % FRAME_STRIDE == 0:
                skeleton = skeleton_extractor.extract_keypoints(frame)
                skeleton_queue.append(torch.tensor(skeleton, dtype=torch.float32))
            
            # --- 4. VIOLENCE DETECTION (LSTM) ---
            if len(skeleton_queue) == SEQUENCE_LENGTH and frame_count % SKIP_INFERENCE == 0:
                
                # Check if we should run inference
                if motion_score < MOTION_THRESHOLD and person_count < MIN_PERSON_COUNT:
                    status_text = "Static Scene"
                    status_color = (100, 100, 100)  # Gray
                else:
                    # Prepare sequence for model
                    sequence = torch.stack(list(skeleton_queue)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        # Run inference
                        outputs = model(sequence)
                        probs = torch.softmax(outputs, dim=1)
                        current_prediction = probs[0][1].item()
                        
                        # Add to prediction history for temporal smoothing
                        prediction_history.append(current_prediction)
                        
                        # Calculate smoothed prediction with confidence margin
                        violence_score = np.mean(prediction_history)
                        
                        # Adjust threshold based on person count and add margin
                        adjusted_threshold = CONFIDENCE_THRESHOLD + CONFIDENCE_MARGIN
                        if person_count > 2:
                            adjusted_threshold *= 0.90  # More sensitive with multiple people
                        
                        # Determine status - require consistent high confidence
                        if violence_score > adjusted_threshold:
                            status_text = f"⚠️ VIOLENCE DETECTED! ({violence_score:.0%})"
                            status_color = (0, 0, 255)  # Red
                        else:
                            status_text = f"✓ Normal Activity ({violence_score:.0%})"
                            status_color = (0, 255, 0)  # Green
            
            # --- 5. DISPLAY ---
            # Draw status panel
            cv2.rectangle(output_frame, (10, 10), (700, 120), (0, 0, 0), -1)
            
            # Main status
            cv2.putText(output_frame, status_text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            
            # Debug info
            debug_line1 = f"Frame: {frame_count} | People: {person_count} | Motion: {motion_score:.1f}"
            cv2.putText(output_frame, debug_line1, (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            debug_line2 = f"AI Score: {violence_score:.3f} | Threshold: {CONFIDENCE_THRESHOLD}"
            cv2.putText(output_frame, debug_line2, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('Skeletal Violence Detection', output_frame)
            
            # Control playback speed (30ms ≈ 30 FPS)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("\n⏹️ Stopped by user")
                break
        
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            traceback.print_exc()
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("🏁 Processing complete!")
    print(f"   Total frames processed: {frame_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
