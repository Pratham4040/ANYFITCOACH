
#Video processing pipeline for exercise reference extraction.
#Converts uploaded video -> pose landmarks -> angle timeline.


import cv2
import json
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .angle_calculator import compute_frame_angles


def process_video_to_angles(video_path: Path, output_json_path: Path) -> dict:
    """
    Process a video file and extract angle timeline.
    
    Args:
        video_path: Path to input video file
        output_json_path: Path to save angles_timeline.json
    
    Returns:
        dict with keys:
            - success: bool
            - frames: int (number of frames processed)
            - angles: list of frame angle data
            - error: str (if failed)
    """
    try:
        # Load video
        vid = cv2.VideoCapture(str(video_path))
        
        if not vid.isOpened():
            return {
                "success": False,
                "error": f"Could not open video file: {video_path}"
            }
        
        fps = vid.get(cv2.CAP_PROP_FPS)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames @ {fps} FPS")
        
        # Setup MediaPipe Pose Landmarker
        model_path = Path(__file__).parent.parent / "models" / "pose_landmarker_heavy (1).task"
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        angles_timeline = []
        frame_num = 0
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = vid.read()
                
                if not success:
                    break
                
                # Calculate timestamp
                timestamp_ms = int((frame_num / fps) * 1000)
                
                # Convert frame to MediaPipe format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                
                # Detect pose
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    # Extract landmarks
                    landmarks = []
                    for landmark in result.pose_landmarks[0]:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    # Compute angles for this frame
                    frame_angles = compute_frame_angles(landmarks)
                    
                    # Add to timeline
                    angles_timeline.append({
                        "frame": frame_num,
                        "timestamp_ms": timestamp_ms,
                        "angles": frame_angles
                    })
                
                frame_num += 1
        
        vid.release()
        
        # Save to JSON
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(angles_timeline, f, indent=2)
        
        print(f"✅ Processed {len(angles_timeline)} frames")
        print(f"✅ Saved to {output_json_path}")
        
        return {
            "success": True,
            "frames": len(angles_timeline),
            "angles": angles_timeline
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
