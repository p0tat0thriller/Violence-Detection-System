import cv2
import mediapipe as mp


class SkeletonVisualizer:
    def __init__(self):
        """
        Initialize the SkeletonVisualizer with MediaPipe Pose model.
        
        Sets up the pose detection model with optimized parameters for 
        real-time skeleton visualization.
        """
        # Initialize MediaPipe pose solution
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Setup the Pose model with specified parameters
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,        # For video stream processing
            model_complexity=1,             # Balanced speed/accuracy
            smooth_landmarks=True,          # Smooth landmark positions
            min_detection_confidence=0.5    # Minimum confidence for detection
        )
        
        # Custom drawing specifications for joints (landmarks)
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),    # GREEN color for joints (BGR format)
            thickness=2,          # Thickness of landmark points
            circle_radius=4       # Radius of landmark circles
        )
        
        # Custom drawing specifications for connections (skeleton lines)
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),    # RED color for connections (BGR format)
            thickness=2           # Thickness of connection lines
        )
    
    def draw(self, frame):
        """
        Draw skeleton landmarks and connections on the input frame.
        
        Args:
            frame (numpy.ndarray): Input image frame from OpenCV (BGR format)
            
        Returns:
            numpy.ndarray: Annotated frame with skeleton overlay
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find pose landmarks
        results = self.pose.process(rgb_frame)
        
        # Draw skeleton if landmarks are detected
        if results.pose_landmarks:
            # Draw landmarks (joints) and connections (skeleton lines)
            self.mp_drawing.draw_landmarks(
                frame,                              # Image to draw on (original BGR frame)
                results.pose_landmarks,             # Detected landmarks
                self.mp_pose.POSE_CONNECTIONS,      # Pose connections
                landmark_drawing_spec=self.landmark_drawing_spec,    # GREEN joints
                connection_drawing_spec=self.connection_drawing_spec # RED lines
            )
        
        return frame
    
    def get_landmarks(self, frame):
        """
        Extract pose landmarks from the frame without drawing.
        
        Args:
            frame (numpy.ndarray): Input image frame from OpenCV (BGR format)
            
        Returns:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None:
                Pose landmarks if detected, None otherwise
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        return results.pose_landmarks if results.pose_landmarks else None
    
    def is_pose_detected(self, frame):
        """
        Check if a pose is detected in the frame.
        
        Args:
            frame (numpy.ndarray): Input image frame from OpenCV (BGR format)
            
        Returns:
            bool: True if pose is detected, False otherwise
        """
        landmarks = self.get_landmarks(frame)
        return landmarks is not None
    
    def close(self):
        """
        Release MediaPipe resources.
        """
        if hasattr(self, 'pose'):
            self.pose.close()


def test_skeleton_visualizer():
    """
    Test function to demonstrate SkeletonVisualizer usage with webcam.
    Press 'q' to quit the demo.
    """
    # Initialize the visualizer
    visualizer = SkeletonVisualizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Skeleton Visualizer Demo")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Draw skeleton on frame
        annotated_frame = visualizer.draw(frame)
        
        # Display the frame
        cv2.imshow('Skeleton Visualizer Demo', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    visualizer.close()


if __name__ == "__main__":
    test_skeleton_visualizer()
