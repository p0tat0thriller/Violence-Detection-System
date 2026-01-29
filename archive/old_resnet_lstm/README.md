# Archived ResNet50+LSTM Model

## Reason for Archival
This folder contains the **original ResNet50+LSTM violence detection model** that has been replaced by the new **Skeletal (YOLOv8-Pose + LSTM) model**.

## Performance Comparison

### Old Model (ResNet50+LSTM)
- **Accuracy**: 50%
- **Violence Recall**: 0% (couldn't detect violence!)
- **Model Size**: ~25M parameters
- **Approach**: Appearance-based (RGB frames)
- **Issues**: 
  - Focused on visual features (clothes, background) instead of actions
  - Overfitted to appearance patterns
  - Large model size
  - Privacy concerns (captures faces/identity)

### New Model (YOLOv8-Pose + Skeletal LSTM)
- **Accuracy**: 70.5%
- **Violence Recall**: 87% (excellent violence detection!)
- **Model Size**: ~1M parameters (25x smaller)
- **Approach**: Motion-based (skeletal keypoints)
- **Advantages**:
  - Captures actual body movements (punches, kicks)
  - Privacy-preserving (no appearance data)
  - Much smaller and faster
  - Better generalization

## Archived Files

1. **model.py** - Original ViolenceModel class with ResNet50 feature extractor
2. **training_model.py** - Training version of the model
3. **dataset.py** - RGB frame dataset loader
4. **train.py** - Training script for ResNet50+LSTM
5. **best_model.pth** - Trained model weights (50% accuracy)
6. **weightsbest_model.pth** - Duplicate weights file

## Date Archived
January 5, 2026

## Note
These files are kept for reference only. The new skeletal model should be used for all future development.
