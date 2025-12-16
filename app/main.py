import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import deque
import sys
import os
import traceback

# --- MODEL DEFINITION ---
class ViolenceModel(nn.Module):
    # FIXED: num_layers=1 to match your trained model
    def __init__(self, num_classes=2, hidden_size=128, num_layers=1):
        super(ViolenceModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(2048, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        r_out = self.resnet(c_in)
        r_out = r_out.view(r_out.size(0), -1)
        r_out = r_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(r_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 16
# Lowered threshold to detect fights easier
CONFIDENCE_THRESHOLD = 0.50  
SKIP_INFERENCE = 5   # Only run AI every 5 frames (Reduces Lag)
FRAME_STRIDE = 2     # Only record every 2nd frame (Fixes "Slow Motion" issue)

def preprocess_frame(frame):
    """
    Preprocess frame with standard ImageNet normalization.
    """
    # Resize to (224, 224)
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Apply ImageNet Standardization (Crucial for ResNet accuracy)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    frame_normalized = (frame_normalized - mean) / std
    
    # Transpose to (Channels, Height, Width)
    frame_chw = np.transpose(frame_normalized, (2, 0, 1))
    
    return torch.tensor(frame_chw, dtype=torch.float32)

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Visualizer
    try:
        from visualizer import SkeletonVisualizer
        viz = SkeletonVisualizer()
        print("✅ Visualizer loaded")
    except ImportError:
        print("⚠️ Visualizer not found. Running without skeleton overlay.")
        viz = None

    # Load Model
    print("Loading Violence Detection Model...")
    model = ViolenceModel(hidden_size=128)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("🎥 Camera active! Press 'q' to quit.")
    
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    frame_count = 0
    status_text = "Initializing..."
    status_color = (255, 255, 0)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            
            # 1. DRAW VISUALS (Always runs)
            output_frame = frame.copy()
            if viz:
                try:
                    output_frame = viz.draw(frame.copy())
                except Exception:
                    pass
            
            # 2. PREPARE INPUT (Every 'FRAME_STRIDE' frames)
            if frame_count % FRAME_STRIDE == 0:
                try:
                    preprocessed_frame = preprocess_frame(frame)
                    frames_queue.append(preprocessed_frame)
                except Exception as e:
                    print(f"Preprocessing Error: {e}")

            # 3. RUN AI (Every 'SKIP_INFERENCE' frames, if queue is full)
            if len(frames_queue) == SEQUENCE_LENGTH and frame_count % SKIP_INFERENCE == 0:
                try:
                    sequence_tensor = torch.stack(list(frames_queue)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        model_output = model(sequence_tensor)
                        probabilities = torch.softmax(model_output, dim=1)
                        violence_prob = probabilities[0][1].item()
                        
                        # Print score to terminal
                        print(f"Violence Score: {violence_prob:.4f}")
                        
                        if violence_prob > CONFIDENCE_THRESHOLD:
                            status_text = f"VIOLENCE! ({violence_prob:.0%})"
                            status_color = (0, 0, 255) # Red
                        else:
                            status_text = f"Neutral ({violence_prob:.0%})"
                            status_color = (0, 255, 0) # Green
                            
                except Exception as e:
                    print(f"Inference Error: {e}")

            # 4. SHOW RESULT
            cv2.rectangle(output_frame, (10, 10), (600, 80), (0, 0, 0), -1)
            cv2.putText(output_frame, status_text, (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(output_frame, f"Frames: {frame_count}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Violence Detection System', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Loop Error: {e}")
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()
    if viz:
        viz.close()

if __name__ == "__main__":
    main()