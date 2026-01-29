"""
Skeleton Extractor using YOLOv8-Pose

Extracts skeletal keypoints from video frames for violence detection.
Uses YOLOv8-Pose to detect humans and extract 17 COCO keypoints.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class SkeletonExtractor:
    """Extract skeletal keypoints using YOLOv8-Pose"""
    
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Initialize the skeleton extractor.
        
        Args:
            model_path: Path to YOLOv8-Pose model weights
        """
        # Load YOLOv8 Pose model
        self.model = YOLO(model_path)
        
    def extract_keypoints(self, frame):
        """
        Extract 17 COCO keypoints from a single frame.
        
        Args:
            frame: BGR image frame (OpenCV format)
            
        Returns:
            numpy array of shape (99,) containing padded keypoint data
            Format: [x1, y1, conf1, x2, y2, conf2, ..., padding...]
        """
        results = self.model(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) > 0:
                # Get first person's keypoints (17 keypoints, x,y coords)
                kp = keypoints[0].flatten()  # Shape: (34,) for 17 keypoints
                
                # Add confidence scores as visibility
                conf = results[0].keypoints.conf.cpu().numpy()[0] if results[0].keypoints.conf is not None else np.ones(17)
                
                # Create full feature vector: [x1,y1,conf1, x2,y2,conf2, ...]
                full_kp = np.zeros(17 * 3, dtype=np.float32)
                for i in range(17):
                    if i * 2 + 1 < len(kp):
                        full_kp[i*3] = kp[i*2] / 640.0      # Normalize x
                        full_kp[i*3 + 1] = kp[i*2 + 1] / 640.0  # Normalize y
                        full_kp[i*3 + 2] = conf[i] if i < len(conf) else 0.0
                
                # Pad to 99 to match expected input size (33 keypoints * 3)
                padded = np.zeros(99, dtype=np.float32)
                padded[:51] = full_kp
                return padded
        
        # Return zero array if no person detected
        return np.zeros(99, dtype=np.float32)
    
    def extract_from_video(self, video_path, num_frames=16, stride=2):
        """
        Extract skeleton sequence from a video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (default: 16)
            stride: Frame sampling stride (default: 2)
            
        Returns:
            numpy array of shape (num_frames, 99) containing skeleton sequence
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames < num_frames * stride:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            start_idx = (total_frames - num_frames * stride) // 2
            frame_indices = np.arange(start_idx, start_idx + num_frames * stride, stride)
        
        skeletons = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                skeleton = self.extract_keypoints(frame)
                skeletons.append(skeleton)
            else:
                skeletons.append(np.zeros(99, dtype=np.float32))
        
        cap.release()
        return np.array(skeletons, dtype=np.float32)
    
    def get_keypoint_names(self):
        """
        Get the names of the 17 COCO keypoints.
        
        Returns:
            list: Names of keypoints
        """
        return [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]


if __name__ == "__main__":
    # Test the skeleton extractor
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        print(f"Testing SkeletonExtractor on: {test_image}")
        
        extractor = SkeletonExtractor()
        frame = cv2.imread(test_image)
        
        if frame is not None:
            skeleton = extractor.extract_keypoints(frame)
            print(f"Extracted skeleton shape: {skeleton.shape}")
            print(f"Non-zero elements: {np.count_nonzero(skeleton)}")
            print(f"First 10 values: {skeleton[:10]}")
        else:
            print("Failed to load image")
    else:
        print("Usage: python skeleton_extractor.py <image_path>")
