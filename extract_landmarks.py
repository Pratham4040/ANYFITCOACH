import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import json
import time
path_to_video = "./SavedVideo/Random/testvid.mp4"
json_path = "./savedJSON/Random/timeline.json"
# setup for the mediapipe
model_path = ".\models\pose_landmarker_heavy (1).task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

#Capture the video
with PoseLandmarker.create_from_options(options) as landmarker:
    vid = cv2.VideoCapture(path_to_video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = 0
    timeline = []  # Initialize BEFORE the main loop
    while True:
        s,f = vid.read()
        if not s:
            print("Frame did not extract by CV2")
            break
        timestamp_ms = int((frame/fps)*1000)
        #convert to mediapipe 
        #first rgb
        rgb = cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # detect
        result = landmarker.detect_for_video(mp_image,timestamp_ms)
        if result.pose_landmarks:
            landmarks_data = []
            for i, landmark in enumerate(result.pose_landmarks[0]):
                landmarks_data.append({
                    "joint_id": i,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })

            timeline.append({
                "frame": frame,
                "timestamp_ms": timestamp_ms,
                "pose_landmarks": landmarks_data
            })
        else:
            print(f"Frame {frame}: No pose detected - skipping")
        frame+=1
    with open(json_path, "w") as f:
        json.dump(timeline, f, indent=2)
        print(f"Saved {len(timeline)} frames to timeline.json")
    vid.release()
# make the landmarker


# # Check if video opened successfully
# if not vid.isOpened():
#     print("Error: Could not open video file")
# else:
#     # Extract properties
#     fps = vid.get(cv2.CAP_PROP_FPS)
#     total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Print them
#     print(f"FPS: {fps}")
#     print(f"Total Frames: {total_frames}")
#     print(f"Width: {width}")
#     print(f"Height: {height}")
    
