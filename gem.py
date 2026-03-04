import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2

# 1. Setup the modern MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 2. Configure the Landmarker
# We tell it where our model file is, and that we want to run in Live Stream mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    # In live stream mode, MediaPipe processes in the background. 
    # We must provide a 'callback' function that it will trigger when a pose is found.
    result_callback=lambda result, image, timestamp_ms: process_result(result, image, timestamp_ms)
)

# A global variable to hold the latest pose data
latest_pose_landmarks = None

# This is our callback function. MediaPipe calls this automatically!
def process_result(result, image, timestamp_ms):
    global latest_pose_landmarks
    latest_pose_landmarks = result.pose_landmarks

# 3. Setup Video & Drawing Utilities
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils # We still use solutions JUST for the drawing tools
mp_pose = mp.solutions.pose

print("Starting modern AI Tracker... Press 'q' to quit.")

# 4. Initialize the model and start the loop
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert OpenCV's BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a specific MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 5. Detect the pose asynchronously! 
        # We must pass a timestamp in milliseconds for live stream mode.
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # 6. Draw the skeleton if we have data
        if latest_pose_landmarks:
            # MediaPipe can technically track multiple people. We will grab the first one.
            for pose_landmarks in latest_pose_landmarks:
                
                # The modern API returns raw coordinates. We have to quickly package them 
                # into a format that the older drawing utilities can understand.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                
                # Draw the connections
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

        # Show the frame
        cv2.imshow('Modern AI Fitness Coach', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()