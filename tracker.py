import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import time
model_path = ".\models\pose_landmarker_heavy (1).task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, image, timestamp_ms: process_result(result, image, timestamp_ms))

latest_pose_landmarks = None
latest_frame = None

def process_result(result, image, timestamp_ms):
    global latest_pose_landmarks, latest_frame
    latest_pose_landmarks = result.pose_landmarks
    latest_frame = image
    return latest_pose_landmarks

cap = cv2.VideoCapture(0)
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        suc,frame = cap.read()
        if not suc:
            print("help cant see")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Draw landmarks if available
        if latest_pose_landmarks:
            annotated_frame = rgb.copy()
            pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
            pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
            # latest_pose_landmarks is a list of poses, draw each one
            for pose_landmarks in latest_pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=pose_landmark_style,
                    connection_drawing_spec=pose_connection_style)
            
            # Convert back to BGR for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame = frame

        cv2.imshow('Modern AI Fitness Coach', display_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()