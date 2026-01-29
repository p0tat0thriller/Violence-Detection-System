import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class ViolenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=16, transform=None):
        """
        Args:
            root_dir (string): Directory with 'Violence' and 'NonViolence' subfolders.
            sequence_length (int): Number of frames to extract from each video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_paths = []
        self.labels = []
        
        # Supported video extensions
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        # Scan for Violence and NonViolence folders
        violence_dir = os.path.join(root_dir, 'Violence')
        nonviolence_dir = os.path.join(root_dir, 'NonViolence')
        
        # Process Violence folder (label = 1)
        if os.path.exists(violence_dir):
            for video_file in os.listdir(violence_dir):
                if video_file.lower().endswith(self.video_extensions):
                    self.video_paths.append(os.path.join(violence_dir, video_file))
                    self.labels.append(1)
        
        # Process NonViolence folder (label = 0)
        if os.path.exists(nonviolence_dir):
            for video_file in os.listdir(nonviolence_dir):
                if video_file.lower().endswith(self.video_extensions):
                    self.video_paths.append(os.path.join(nonviolence_dir, video_file))
                    self.labels.append(0)
        
        print(f"Dataset initialized with {len(self.video_paths)} videos")
        print(f"Violence videos: {sum(self.labels)}")
        print(f"NonViolence videos: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the video to retrieve.
            
        Returns:
            tuple: (video_tensor, label) where video_tensor is of shape 
                   (sequence_length, 3, 224, 224) and label is an int.
        """
        video_path = self.video_paths[index]
        label = self.labels[index]
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            frames = []
            
            # Frame sampling strategy
            if total_frames >= self.sequence_length:
                # Sample frames uniformly across the video duration
                frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            else:
                # If video is shorter, create indices by looping
                frame_indices = []
                for i in range(self.sequence_length):
                    frame_indices.append(i % total_frames)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    # If we can't read this frame, try to get the last successful frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
                    ret, frame = cap.read()
                    
                    if not ret:
                        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
                
                # Process frame
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                # Convert to CHW format (channels, height, width)
                frame = np.transpose(frame, (2, 0, 1))
                
                frames.append(frame)
            
            cap.release()
            
            # Convert list of frames to tensor of shape (sequence_length, 3, 224, 224)
            video_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
            
            # Apply transform if provided
            if self.transform:
                video_tensor = self.transform(video_tensor)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {str(e)}")
            cap.release() if 'cap' in locals() else None
            
            # Return next valid video (recursive call)
            next_index = (index + 1) % len(self.video_paths)
            if next_index == index:
                # If we've cycled through all videos, raise an error
                raise RuntimeError("All videos in dataset failed to load")
            
            return self.__getitem__(next_index)
    
    def get_class_distribution(self):
        """
        Returns the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary with class counts.
        """
        violence_count = sum(self.labels)
        nonviolence_count = len(self.labels) - violence_count
        
        return {
            'Violence': violence_count,
            'NonViolence': nonviolence_count,
            'Total': len(self.labels)
        }
    
    def get_video_info(self, index):
        """
        Get information about a specific video.
        
        Args:
            index (int): Index of the video.
            
        Returns:
            dict: Dictionary with video information.
        """
        video_path = self.video_paths[index]
        label = self.labels[index]
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        return {
            'path': video_path,
            'label': 'Violence' if label == 1 else 'NonViolence',
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'duration_seconds': duration
        }
