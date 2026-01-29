# Skeletal Violence Detection System

## 🎯 Overview

A privacy-preserving violence detection system using skeletal pose estimation and LSTM networks.

### Performance
- **Accuracy**: 70.5%
- **Violence Recall**: 87% (excellent at catching violent behavior)
- **Model Size**: ~1M parameters (25x smaller than ResNet50+LSTM)
- **Approach**: Motion-based (skeletal keypoints) instead of appearance-based

## 🏗️ Architecture

```
Video Frame → YOLOv8-Pose → Skeleton (17 keypoints)
                ↓
          Feature Encoder (99 → 256)
                ↓
        Bidirectional LSTM (2 layers)
                ↓
          Classifier (Violence/Normal)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Violence Detection

```bash
# Activate virtual environment (if using)
source venv/Scripts/activate  # Windows
# source venv/bin/activate     # Linux/Mac

# Run on video file
cd app
python main_skeletal.py
```

### 3. Configure Video Source

Edit the `VIDEO_SOURCE` path in `main_skeletal.py`:

```python
VIDEO_SOURCE = r"path/to/your/video.mp4"
```

## 📁 Project Structure

```
Violence-Detection-System/
├── app/
│   ├── main_skeletal.py          # 🌟 Main inference script (NEW)
│   ├── model.py                  # 🌟 Skeletal LSTM model (NEW)
│   ├── skeleton_extractor.py    # 🌟 YOLOv8-Pose extractor (NEW)
│   ├── yolov8n.pt               # YOLOv8 person detection
│   ├── yolov8n-pose.pt          # 🌟 YOLOv8-Pose model (NEW)
│   └── ...
│
├── weights/
│   └── best_skeleton_model.pth  # 🌟 Trained model (70.5% accuracy)
│
├── training/
│   └── skeletal_violence_detection_rwf2000.ipynb  # Training notebook
│
├── Output/
│   ├── training_results.json    # Training metrics
│   ├── confusion_matrix.png     # Performance visualization
│   └── training_history.png     # Training curves
│
├── archive/
│   └── old_resnet_lstm/         # Old ResNet50+LSTM model (50% accuracy)
│
└── README.md
```

## 🔧 Configuration

### Model Parameters (in `main_skeletal.py`):

```python
SEQUENCE_LENGTH = 16          # Number of frames for LSTM
CONFIDENCE_THRESHOLD = 0.65   # Violence detection threshold (0-1)
SKIP_INFERENCE = 3            # Run AI every N frames
FRAME_STRIDE = 2              # Sample every Nth frame
MOTION_THRESHOLD = 3.0        # Minimum motion to trigger
PERSON_CONFIDENCE = 0.45      # Person detection confidence
TEMPORAL_SMOOTHING = 5        # Predictions to smooth
```

## 📊 Model Comparison

| Feature | Old Model (ResNet50+LSTM) | New Model (Skeletal) |
|---------|--------------------------|---------------------|
| **Accuracy** | 50% | **70.5%** ✅ |
| **Violence Recall** | 0% | **87%** ✅ |
| **Model Size** | ~25M params | **~1M params** ✅ |
| **Input** | Raw RGB frames | Skeletal keypoints |
| **Privacy** | Captures faces/identity | **Privacy-preserving** ✅ |
| **Speed** | Slower (large model) | **Faster** ✅ |

## 🎓 How It Works

1. **Skeleton Extraction** (YOLOv8-Pose)
   - Detects humans in frame
   - Extracts 17 COCO keypoints per person
   - Keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
   
2. **Feature Encoding**
   - Compresses 99 features (17×3 keypoints + padding)
   - 2-layer MLP: 99 → 512 → 256
   
3. **Temporal Modeling** (Bidirectional LSTM)
   - Processes 16-frame sequences
   - 2 layers, 256 hidden units
   - Looks at past AND future frames
   
4. **Classification**
   - Binary output: Violence / Non-Violence
   - Temporal smoothing for stability

## 🎯 Use Cases

✅ **Surveillance Systems** - Real-time violence detection in public spaces  
✅ **School Safety** - Monitor playgrounds and hallways  
✅ **Prison Monitoring** - Detect fights and altercations  
✅ **Sports Analytics** - Detect fouls and aggressive behavior  
✅ **Content Moderation** - Filter violent videos on platforms  

## 🛡️ Privacy Benefits

Unlike appearance-based models (ResNet50), this skeletal approach:
- ✅ Does NOT capture faces or identifying features
- ✅ Only tracks body movements and poses
- ✅ Cannot reconstruct original appearance
- ✅ GDPR/privacy-friendly for surveillance

## 📈 Training Details

- **Dataset**: RWF-2000 (2,000 real-world fight videos)
- **Training Time**: ~18 minutes on GPU
- **Epochs**: 19 (early stopping at epoch 12)
- **Best F1-Score**: 74.68%
- **ROC AUC**: 0.68

## 🔬 Technical Details

**Skeletal Keypoints (17 COCO points):**
```
0: Nose          6: Right Shoulder  12: Right Hip
1: Left Eye      7: Left Elbow      13: Left Knee
2: Right Eye     8: Right Elbow     14: Right Knee
3: Left Ear      9: Left Wrist      15: Left Ankle
4: Right Ear     10: Right Wrist    16: Right Ankle
5: Left Shoulder 11: Left Hip
```

## 🐛 Troubleshooting

**Model not loading?**
- Check that `weights/best_skeleton_model.pth` exists
- Verify the path in `main_skeletal.py`

**Low performance?**
- Reduce `SKIP_INFERENCE` value (more frequent predictions)
- Lower `CONFIDENCE_THRESHOLD` (more sensitive)
- Adjust `TEMPORAL_SMOOTHING` (reduce for faster response)

**No person detected?**
- Lower `PERSON_CONFIDENCE` threshold
- Check lighting conditions in video
- Ensure people are visible in frame

## 📝 Citation

If you use this model in your research, please cite:

```
Skeletal Violence Detection System
Based on YOLOv8-Pose and Bidirectional LSTM
Trained on RWF-2000 dataset
Accuracy: 70.5%, Violence Recall: 87%
```

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more datasets (AIRTLAB, Real Life Violence)
- Implement attention mechanisms
- Try Graph Neural Networks (GNN)
- Add multi-person aggregation
- Improve temporal modeling

---

**Created**: January 2026  
**Model Version**: v1.0  
**Status**: ✅ Production Ready
