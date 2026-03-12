
#Video processing pipeline for exercise reference extraction.
#Converts uploaded video -> pose landmarks -> angle timeline.


import cv2
import json
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .angle_calculator import compute_frame_angles


def _angle_mae(angles_a: dict, angles_b: dict) -> float:
    """Mean absolute error between two angle dictionaries; ignores null joints."""
    deltas = []
    for joint, value_a in angles_a.items():
        value_b = angles_b.get(joint)
        if value_a is None or value_b is None:
            continue
        deltas.append(abs(value_a - value_b))

    if not deltas:
        return float("inf")

    return sum(deltas) / len(deltas)


def _movement_peak_from_start(timeline: list, end_idx: int) -> float:
    """Largest deviation from starting pose up to end_idx."""
    if end_idx <= 0:
        return 0.0

    start_angles = timeline[0]["angles"]
    peak = 0.0

    for idx in range(1, end_idx + 1):
        dist = _angle_mae(start_angles, timeline[idx]["angles"])
        if dist != float("inf") and dist > peak:
            peak = dist

    return peak


def _detect_cycle_end_index(timeline: list) -> tuple:
    """
    Detect one full movement cycle that returns near starting pose.

    Returns:
        (cycle_end_idx, metrics_dict)
        cycle_end_idx is None if no reliable cycle is found.
    """
    total = len(timeline)
    if total < 60:
        return None, {
            "reason": "timeline_too_short",
            "total_frames": total,
        }

    start_angles = timeline[0]["angles"]

    # Estimate pose jitter near the start for adaptive thresholding.
    start_jitter_samples = []
    jitter_window = min(8, total - 1)
    for idx in range(1, jitter_window + 1):
        jitter = _angle_mae(start_angles, timeline[idx]["angles"])
        if jitter != float("inf"):
            start_jitter_samples.append(jitter)

    start_jitter = (
        sum(start_jitter_samples) / len(start_jitter_samples)
        if start_jitter_samples
        else 3.0
    )

    similarity_threshold = max(8.0, min(18.0, start_jitter * 4.0))
    movement_threshold = max(16.0, similarity_threshold * 1.5)

    min_cycle_len = max(24, int(total * 0.18))
    max_cycle_len = max(min_cycle_len + 1, int(total * 0.90))
    max_cycle_len = min(max_cycle_len, total - 1)

    best_idx = None
    best_dist = float("inf")
    consecutive_hits = 0

    for idx in range(min_cycle_len, max_cycle_len + 1):
        dist = _angle_mae(start_angles, timeline[idx]["angles"])
        if dist == float("inf"):
            continue

        if dist <= similarity_threshold:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        if dist < best_dist:
            best_dist = dist
            best_idx = idx

        # Require repeated near-start similarity to reduce false positives.
        if consecutive_hits >= 3:
            # Use the first index in this confirmed near-start streak.
            candidate_idx = idx - 2
            peak = _movement_peak_from_start(timeline, candidate_idx)
            if peak >= movement_threshold:
                return candidate_idx, {
                    "reason": "confirmed_repetition",
                    "total_frames": total,
                    "best_distance": best_dist,
                    "start_jitter": start_jitter,
                    "similarity_threshold": similarity_threshold,
                    "movement_peak": peak,
                    "movement_threshold": movement_threshold,
                }

    if best_idx is not None:
        peak = _movement_peak_from_start(timeline, best_idx)
        if best_dist <= similarity_threshold and peak >= movement_threshold:
            return best_idx, {
                "reason": "best_candidate",
                "total_frames": total,
                "best_distance": best_dist,
                "start_jitter": start_jitter,
                "similarity_threshold": similarity_threshold,
                "movement_peak": peak,
                "movement_threshold": movement_threshold,
            }

    return None, {
        "reason": "no_reliable_cycle",
        "total_frames": total,
        "best_distance": best_dist if best_idx is not None else None,
        "start_jitter": start_jitter,
        "similarity_threshold": similarity_threshold,
        "movement_threshold": movement_threshold,
    }


def _compress_to_single_cycle(timeline: list) -> tuple:
    """
    Compress timeline to one canonical cycle when possible.

    Returns:
        (compressed_timeline, cycle_info)
    """
    if not timeline:
        return timeline, {
            "cycle_detected": False,
            "reason": "empty_timeline",
            "original_frames": 0,
            "cycle_frames": 0,
        }

    cycle_end_idx, metrics = _detect_cycle_end_index(timeline)
    if cycle_end_idx is None:
        return timeline, {
            "cycle_detected": False,
            "original_frames": len(timeline),
            "cycle_frames": len(timeline),
            **metrics,
        }

    compressed = timeline[: cycle_end_idx + 1]
    return compressed, {
        "cycle_detected": True,
        "original_frames": len(timeline),
        "cycle_frames": len(compressed),
        "cycle_end_index": cycle_end_idx,
        **metrics,
    }


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
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # Convert frame to MediaPipe format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
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
        
        # Detect one canonical cycle and keep only that segment if reliable.
        compressed_timeline, cycle_info = _compress_to_single_cycle(angles_timeline)

        # Save to JSON
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(compressed_timeline, f, indent=2)
        
        print(f"✅ Processed {len(angles_timeline)} frames")
        if cycle_info["cycle_detected"]:
            print(
                f"✅ Cycle detected: kept {cycle_info['cycle_frames']} / "
                f"{cycle_info['original_frames']} frames"
            )
        else:
            print("ℹ️ No reliable cycle detected, kept full timeline")
        print(f"✅ Saved to {output_json_path}")
        
        return {
            "success": True,
            "frames": len(compressed_timeline),
            "angles": compressed_timeline,
            "cycle_info": cycle_info,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
