"""
Real-time Violence Detection with Live Metric Tracking
Enhanced version of main.py that tracks and displays performance metrics in real-time
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import deque
import sys
import os
import traceback
import json
from datetime import datetime
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
VIDEO_SOURCE = r"C:\Users\akshi\Downloads\test_fight_clip.mp4"
GROUND_TRUTH_FILE = None  # Optional: JSON file with ground truth labels {"frame_ranges": [[start, end, label], ...]}

# MODEL CONFIG
SEQUENCE_LENGTH = 16
CONFIDENCE_THRESHOLD = 0.70
SKIP_INFERENCE = 5
FRAME_STRIDE = 2

# THRESHOLDS
MOTION_THRESHOLD = 5.0
PERSON_CONFIDENCE = 0.5
TEMPORAL_SMOOTHING = 3
MIN_FRAME_QUALITY = 30

# METRIC TRACKING CONFIG
ENABLE_METRICS = True  # Set to False to disable metric tracking
METRICS_UPDATE_INTERVAL = 30  # Update metrics display every N frames
SAVE_METRICS_LOG = True  # Save metrics to file
METRICS_OUTPUT_PATH = r"C:\Users\akshi\OneDrive\Desktop\Akshit\Violence-Detection-System\realtime_metrics_log.json"


# --- 2. MODEL DEFINITION ---
class ViolenceModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, num_layers=1):
        super(ViolenceModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_layers)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        r_out = self.resnet(c_in)
        r_out = r_out.view(r_out.size(0), -1)
        r_out = r_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(r_out)
        output = self.fc(lstm_out[:, -1, :])
        return output


class MetricsTracker:
    """Track and calculate real-time metrics"""
    def __init__(self):
        self.predictions = []  # (frame_num, predicted_label, confidence_score)
        self.ground_truth = []  # (frame_num, true_label)
        
        # Confusion matrix components
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        
        # Running counts
        self.total_violence_detected = 0
        self.total_nonviolence_detected = 0
        self.total_frames_processed = 0
        
        # Temporal tracking
        self.violence_segments = []  # List of (start_frame, end_frame)
        self.current_violence_start = None
        
        # Score tracking
        self.violence_scores = []
        self.motion_scores = []
        
    def add_prediction(self, frame_num, predicted_label, confidence_score, 
                      true_label=None, motion_score=None):
        """Add a prediction to the tracker"""
        self.predictions.append((frame_num, predicted_label, confidence_score))
        self.total_frames_processed += 1
        
        if predicted_label == 1:
            self.total_violence_detected += 1
            if self.current_violence_start is None:
                self.current_violence_start = frame_num
        else:
            self.total_nonviolence_detected += 1
            if self.current_violence_start is not None:
                self.violence_segments.append((self.current_violence_start, frame_num - 1))
                self.current_violence_start = None
        
        # Track scores
        if confidence_score is not None:
            self.violence_scores.append(confidence_score)
        if motion_score is not None:
            self.motion_scores.append(motion_score)
        
        # Update confusion matrix if ground truth is provided
        if true_label is not None:
            self.ground_truth.append((frame_num, true_label))
            if predicted_label == 1 and true_label == 1:
                self.tp += 1
            elif predicted_label == 1 and true_label == 0:
                self.fp += 1
            elif predicted_label == 0 and true_label == 0:
                self.tn += 1
            elif predicted_label == 0 and true_label == 1:
                self.fn += 1
    
    def get_current_metrics(self):
        """Calculate current metrics"""
        metrics = {
            'total_frames': self.total_frames_processed,
            'violence_detected': self.total_violence_detected,
            'nonviolence_detected': self.total_nonviolence_detected,
            'violence_segments': len(self.violence_segments),
            'avg_violence_score': np.mean(self.violence_scores) if self.violence_scores else 0.0,
            'max_violence_score': max(self.violence_scores) if self.violence_scores else 0.0,
            'avg_motion_score': np.mean(self.motion_scores) if self.motion_scores else 0.0,
        }
        
        # Calculate classification metrics if ground truth is available
        if len(self.ground_truth) > 0:
            total = self.tp + self.fp + self.tn + self.fn
            if total > 0:
                metrics['accuracy'] = (self.tp + self.tn) / total
                metrics['precision'] = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
                metrics['recall'] = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                    (metrics['precision'] + metrics['recall']) if \
                                    (metrics['precision'] + metrics['recall']) > 0 else 0
                metrics['specificity'] = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
                
                metrics['confusion_matrix'] = {
                    'TP': self.tp,
                    'FP': self.fp,
                    'TN': self.tn,
                    'FN': self.fn
                }
        
        return metrics
    
    def save_to_file(self, filepath):
        """Save metrics to JSON file"""
        metrics = self.get_current_metrics()
        metrics['predictions'] = [
            {'frame': p[0], 'label': p[1], 'score': p[2]} 
            for p in self.predictions
        ]
        metrics['violence_segments'] = self.violence_segments
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Metrics saved to: {filepath}")


def load_ground_truth(filepath):
    """
    Load ground truth labels from JSON file
    Format: {"frame_ranges": [[start_frame, end_frame, label], ...]}
    """
    if filepath is None or not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get('frame_ranges', [])


def get_ground_truth_label(frame_num, ground_truth_ranges):
    """Get ground truth label for a specific frame"""
    if ground_truth_ranges is None:
        return None
    
    for start, end, label in ground_truth_ranges:
        if start <= frame_num <= end:
            return label
    
    return 0  # Default to non-violence


def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame_normalized = (frame_normalized - mean) / std
    frame_chw = np.transpose(frame_normalized, (2, 0, 1))
    return torch.tensor(frame_chw, dtype=torch.float32)


def calculate_frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def fast_motion_detection(current_gray, prev_gray):
    if prev_gray is None:
        return 0.0
    frame_delta = cv2.absdiff(prev_gray, current_gray)
    return np.mean(frame_delta)


def draw_metrics_panel(frame, metrics_tracker, frame_count):
    """Draw a metrics panel on the frame"""
    metrics = metrics_tracker.get_current_metrics()
    
    # Panel dimensions
    panel_height = 200
    panel_width = 400
    panel_x = frame.shape[1] - panel_width - 10
    panel_y = 10
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw metrics text
    y_offset = panel_y + 25
    line_height = 25
    
    cv2.putText(frame, "=== METRICS ===", (panel_x + 10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Frames: {metrics['total_frames']}", 
                (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Violence: {metrics['violence_detected']}", 
                (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Non-Violence: {metrics['nonviolence_detected']}", 
                (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Avg Score: {metrics['avg_violence_score']:.3f}", 
                (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += line_height
    
    # Show classification metrics if available
    if 'accuracy' in metrics:
        y_offset += 5
        cv2.putText(frame, f"Accuracy: {metrics['accuracy']:.3f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Precision: {metrics['precision']:.3f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Recall: {metrics['recall']:.3f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device Selection: {device}")

    # --- LOAD MODELS ---
    print("Loading YOLOv8 model...")
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)
    print("✅ YOLO loaded")

    print("Loading Violence Detection Model...")
    model = ViolenceModel(hidden_size=128)
    model_path = r"C:\Users\akshi\OneDrive\Desktop\Akshit\Violence-Detection-System\weights\best_model.pth"

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Violence Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading violence model: {str(e)}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # --- INITIALIZE METRICS TRACKER ---
    metrics_tracker = MetricsTracker() if ENABLE_METRICS else None
    ground_truth_ranges = load_ground_truth(GROUND_TRUTH_FILE)
    
    if ground_truth_ranges:
        print(f"✅ Ground truth loaded: {len(ground_truth_ranges)} segments")
    else:
        print("⚠️  No ground truth file - running without accuracy metrics")

    # --- VIDEO SETUP ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open file {VIDEO_SOURCE}")
        return

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    prev_gray = None
    motion_score = 0.0
    prediction_history = deque(maxlen=TEMPORAL_SMOOTHING)
    status_text = "Scanning..."
    status_color = (255, 255, 0)
    violence_score = 0.0

    print(f"🎥 Playing Video: {VIDEO_SOURCE}")
    print("   Press 'q' to quit, 's' to save metrics")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video Finished.")
                break

            frame_count += 1
            output_frame = frame.copy()

            # --- MOTION DETECTION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            motion_score = fast_motion_detection(gray, prev_gray)
            prev_gray = gray

            # --- YOLO DETECTION ---
            person_found = False
            person_count = 0
            if frame_count % 3 == 0:
                results = yolo(frame, verbose=False, stream=False)
                for r in results:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:
                            confidence = float(box.conf[0])
                            if confidence >= PERSON_CONFIDENCE:
                                person_found = True
                                person_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # --- QUEUE MANAGEMENT ---
            if frame_count % FRAME_STRIDE == 0:
                frames_queue.append(preprocess_frame(frame))

            # --- INFERENCE ---
            predicted_label = 0
            if len(frames_queue) == SEQUENCE_LENGTH and frame_count % SKIP_INFERENCE == 0:
                if motion_score < MOTION_THRESHOLD:
                    violence_score = 0.0
                    status_text = f"Static (Motion: {motion_score:.1f})"
                    status_color = (0, 255, 0)
                    predicted_label = 0
                elif not person_found:
                    violence_score = 0.0
                    status_text = "No Person Detected"
                    status_color = (100, 100, 100)
                    predicted_label = 0
                else:
                    sequence = torch.stack(list(frames_queue)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(sequence)
                        probs = torch.softmax(out, dim=1)
                        current_prediction = probs[0][1].item()
                        prediction_history.append(current_prediction)
                        violence_score = np.mean(prediction_history)

                        adjusted_threshold = CONFIDENCE_THRESHOLD
                        if person_count > 2:
                            adjusted_threshold *= 0.9
                        elif person_count == 1:
                            adjusted_threshold *= 1.1

                        if violence_score > adjusted_threshold:
                            status_text = f"VIOLENCE! ({violence_score:.0%}) [P:{person_count}]"
                            status_color = (0, 0, 255)
                            predicted_label = 1
                        else:
                            status_text = f"Active ({violence_score:.0%}) [P:{person_count}]"
                            status_color = (0, 255, 0)
                            predicted_label = 0
                
                # Track metrics
                if ENABLE_METRICS and metrics_tracker:
                    true_label = get_ground_truth_label(frame_count, ground_truth_ranges)
                    metrics_tracker.add_prediction(frame_count, predicted_label, 
                                                   violence_score, true_label, motion_score)

            # --- DISPLAY ---
            cv2.rectangle(output_frame, (10, 10), (600, 90), (0, 0, 0), -1)
            cv2.putText(output_frame, status_text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            debug_info = f"Motion: {motion_score:.1f} | AI: {violence_score:.2f}"
            cv2.putText(output_frame, debug_info, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw metrics panel
            if ENABLE_METRICS and metrics_tracker and frame_count % METRICS_UPDATE_INTERVAL == 0:
                draw_metrics_panel(output_frame, metrics_tracker, frame_count)

            cv2.imshow('Violence Detection with Real-time Metrics', output_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and ENABLE_METRICS and metrics_tracker:
                # Save metrics on demand
                metrics_tracker.save_to_file(METRICS_OUTPUT_PATH)

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save final metrics
    if ENABLE_METRICS and metrics_tracker and SAVE_METRICS_LOG:
        metrics_tracker.save_to_file(METRICS_OUTPUT_PATH)
        
        # Print final summary
        final_metrics = metrics_tracker.get_current_metrics()
        print("\n" + "="*60)
        print("FINAL METRICS SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {final_metrics['total_frames']}")
        print(f"Violence Detected: {final_metrics['violence_detected']}")
        print(f"Non-Violence Detected: {final_metrics['nonviolence_detected']}")
        print(f"Average Violence Score: {final_metrics['avg_violence_score']:.3f}")
        print(f"Max Violence Score: {final_metrics['max_violence_score']:.3f}")
        
        if 'accuracy' in final_metrics:
            print(f"\nClassification Metrics:")
            print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
            print(f"  Precision: {final_metrics['precision']:.4f}")
            print(f"  Recall:    {final_metrics['recall']:.4f}")
            print(f"  F1-Score:  {final_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
