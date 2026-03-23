Skeletal Violence Detection System
🎯 Overview
A privacy-preserving violence detection system using skeletal pose estimation and LSTM networks.

Performance
Accuracy: 70.5% (RWF-2000 validation)
Violence Recall: 87% (excellent at catching violent behavior)
Model Size: ~2.4M parameters (384 hidden units, 2 layers)
Training Data: RWF-2000 + AIRTLab (~2,350 videos)
Approach: Motion-based (skeletal keypoints) instead of appearance-based
🏗️ Architecture
Video Frame → YOLOv8-Pose → Skeleton (17 keypoints)
                ↓
          Feature Encoder (99 → 512 → 256)
                ↓
        Bidirectional LSTM (384 hidden, 2 layers)
                ↓
          Classifier (768 → 128 → 2)
🚀 Quick Start
1. Install Dependencies
pip install -r requirements.txt
2. Run Violence Detection
Option A: Local Video Processing (OpenCV)

cd app
python main_skeletal.py
Option B: Web Interface (Streamlit)

cd app
streamlit run streamlit_app.py
Upload video files or use webcam
Real-time detection with on-screen overlays
Adjustable confidence threshold and settings
3. Configure Video Source
Edit the VIDEO_SOURCE path in main_skeletal.py:

VIDEO_SOURCE = r"path/to/your/video.mp4"
📁 Project Structure
Violence-Detection-System/
├── app/
│   ├── main_skeletal.py          # 🌟 Local inference (OpenCV)
│   ├── streamlit_app.py          # 🌟 Web UI (Streamlit)
│   ├── skeleton_model.py         # 🌟 BiLSTM model (384 hidden)
│   ├── skeleton_extractor.py    # 🌟 YOLOv8-Pose extractor
│   ├── yolov8n.pt               # YOLOv8 person detection
│   ├── yolov8n-pose.pt          # 🌟 YOLOv8-Pose model
│   └── ...
│
├── weights/
│   └── best_skeleton_model.pth  # 🌟 Trained model (384 hidden, 24 seq)
│
├── training/
│   └── training_script.txt      # Kaggle training script (RWF-2000 + AIRTLab)
│
├── archive/
│   └── old_resnet_lstm/         # Old ResNet50+LSTM model (50% accuracy)
│
└── README_SKELETAL.md
🔧 Configuration
Model Parameters (in main_skeletal.py):
SEQUENCE_LENGTH = 24          # Number of frames for LSTM (increased from 16)
HIDDEN_SIZE = 384             # LSTM hidden units (increased from 256)
CONFIDENCE_THRESHOLD = 0.60   # Violence detection threshold (0-1)
CONFIDENCE_MARGIN = 0.10      # Additional threshold buffer
SKIP_INFERENCE = 5            # Run AI every N frames
FRAME_STRIDE = 2              # Sample every Nth frame
MOTION_THRESHOLD = 3.0        # Minimum motion to trigger
PERSON_CONFIDENCE = 0.45      # Person detection confidence
TEMPORAL_SMOOTHING = 5        # Predictions to smooth (weighted averaging)
📊 Model Comparison
Feature	Old Model (ResNet50+LSTM)	New Model (Skeletal)
Accuracy	50%	70.5% ✅
Violence Recall	0%	87% ✅
Model Size	~25M params	~1M params ✅
Input	Raw RGB frames	Skeletal keypoints
Privacy	Captures faces/identity	Privacy-preserving ✅
Speed	Slower (large model)	Faster ✅
🎓 How It Works
Skeleton Extraction (YOLOv8-Pose)

Detects humans in frame
Extracts 17 COCO keypoints per person
Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
Feature Encoding

Compresses 99 features (17×3 keypoints + padding)
2-layer MLP: 99 → 512 → 256
Temporal Modeling (Bidirectional LSTM)

Processes 24-frame sequences (improved temporal context)
2 layers, 384 hidden units (increased capacity)
Looks at past AND future frames
Weighted temporal smoothing (70% current, 30% history)
Classification

Binary output: Violence / Non-Violence
Temporal smoothing for stability
🎯 Use Cases
✅ Surveillance Systems - Real-time violence detection in public spaces
✅ School Safety - Monitor playgrounds and hallways
✅ Prison Monitoring - Detect fights and altercations
✅ Sports Analytics - Detect fouls and aggressive behavior
✅ Content Moderation - Filter violent videos on platforms

🛡️ Privacy Benefits
Unlike appearance-based models (ResNet50), this skeletal approach:

✅ Does NOT capture faces or identifying features
✅ Only tracks body movements and poses
✅ Cannot reconstruct original appearance
✅ GDPR/privacy-friendly for surveillance
📈 Training Details
Datasets:
RWF-2000: ~2,000 real-world fight videos (train + val)
AIRTLab: 350 videos (230 violent, 120 non-violent)
Total: ~2,350 training videos
Training: Kaggle GPU (P100/T4)
Validation: RWF-2000 validation set (consistent benchmarking)
Architecture: 384 hidden units, 24 sequence length
Best F1-Score: 74.68%
ROC AUC: 0.68
🔬 Technical Details
Skeletal Keypoints (17 COCO points):

0: Nose          6: Right Shoulder  12: Right Hip
1: Left Eye      7: Left Elbow      13: Left Knee
2: Right Eye     8: Right Elbow     14: Right Knee
3: Left Ear      9: Left Wrist      15: Left Ankle
4: Right Ear     10: Right Wrist    16: Right Ankle
5: Left Shoulder 11: Left Hip
🐛 Troubleshooting
Model not loading?

Check that weights/best_skeleton_model.pth exists
Verify the path in main_skeletal.py
Low performance?

Reduce SKIP_INFERENCE value (more frequent predictions)
Lower CONFIDENCE_THRESHOLD (more sensitive)
Adjust TEMPORAL_SMOOTHING (reduce for faster response)
No person detected?

Lower PERSON_CONFIDENCE threshold
Check lighting conditions in video
Ensure people are visible in frame
📝 Citation
If you use this model in your research, please cite:

Skeletal Violence Detection System
Based on YOLOv8-Pose and Bidirectional LSTM
Trained on RWF-2000 dataset
Accuracy: 70.5%, Violence Recall: 87%
📄 License
This project is for educational and research purposes.

🤝 Contributing
Contributions welcome! Areas for improvement:

Add more datasets (AIRTLAB, Real Life Violence)
Implement attention mechanisms
Try Graph Neural Networks (GNN)
Add multi-person aggregation
Improve temporal modeling
Created: January 2026
Model Version: v1.0
Status: ✅ Production Ready
