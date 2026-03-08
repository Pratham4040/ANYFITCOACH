
#Angle calculation utilities for pose landmarks.
#Converts 3D joint positions to biomechanically meaningful angles.

import numpy as np

# Define angle triplets (joint_a, joint_b, joint_c) where angle is at joint_b
ANGLE_TRIPLETS = {
    "right_elbow": (11, 13, 15),      # shoulder, elbow, wrist
    "left_elbow": (12, 14, 16),       # shoulder, elbow, wrist
    "right_shoulder": (13, 11, 23),   # elbow, shoulder, hip
    "left_shoulder": (14, 12, 24),    # elbow, shoulder, hip
    "right_hip": (11, 23, 25),        # shoulder, hip, knee
    "left_hip": (12, 24, 26),         # shoulder, hip, knee
    "right_knee": (23, 25, 27),       # hip, knee, ankle
    "left_knee": (24, 26, 28),        # hip, knee, ankle
}

CONFIDENCE_THRESHOLD = 0.5


def compute_angle(p1, p2, p3):

    u = np.array([p1['x'], p1['y'], p1['z']]) - np.array([p2['x'], p2['y'], p2['z']])
    v = np.array([p3['x'], p3['y'], p3['z']]) - np.array([p2['x'], p2['y'], p2['z']])
    
    # Compute magnitudes
    u_mag = np.linalg.norm(u)
    v_mag = np.linalg.norm(v)
    
    if u_mag == 0 or v_mag == 0:
        return None
    
    # Compute dot product
    dot = np.dot(u, v)
    
    # Compute angle in radians then convert to degrees
    cos_angle = dot / (u_mag * v_mag)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)


def compute_frame_angles(landmarks):
    """
    Compute all standard angles for a single frame of pose landmarks.
    
    Args:
        landmarks: List of 33 landmark dicts with x, y, z, visibility
    
    Returns:
        dict: Angle name -> angle value (or None if low confidence)
    """
    frame_angles = {}
    
    for angle_name, (idx_a, idx_b, idx_c) in ANGLE_TRIPLETS.items():
        # Get landmarks
        p_a = landmarks[idx_a]
        p_b = landmarks[idx_b]
        p_c = landmarks[idx_c]
        
        # Check confidence
        if (p_a['visibility'] < CONFIDENCE_THRESHOLD or 
            p_b['visibility'] < CONFIDENCE_THRESHOLD or 
            p_c['visibility'] < CONFIDENCE_THRESHOLD):
            frame_angles[angle_name] = None
            continue
        
        # Compute angle
        angle = compute_angle(p_a, p_b, p_c)
        frame_angles[angle_name] = angle
    
    return frame_angles
