
#Exercise management utilities.
#Handles listing, fetching, and deleting exercises.


import json
import shutil
from pathlib import Path
from typing import List, Optional


# Base storage paths
STORAGE_ROOT = Path(__file__).parent.parent / "storage"
VIDEOS_DIR = STORAGE_ROOT / "videos"
ANGLES_DIR = STORAGE_ROOT / "angles"


def list_exercises() -> List[str]:
    """
    List all available exercises.
    
    Returns:
        List of exercise names (folder names in storage/angles/)
    """
    if not ANGLES_DIR.exists():
        return []
    
    exercises = [
        folder.name for folder in ANGLES_DIR.iterdir()
        if folder.is_dir() and (folder / "angles_timeline.json").exists()
    ]
    
    return sorted(exercises)


def get_exercise_angles(exercise_name: str) -> Optional[dict]:


    json_path = ANGLES_DIR / exercise_name / "angles_timeline.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            angles = json.load(f)
        
        return {
            "exercise_name": exercise_name,
            "frames": len(angles),
            "angles": angles
        }
    except Exception as e:
        print(f"Error loading exercise '{exercise_name}': {e}")
        return None


def delete_exercise(exercise_name: str) -> bool:
    """
    Delete an exercise and all its associated files.
    
    Args:
        exercise_name: Name of the exercise to delete
    
    Returns:
        True if deleted successfully, False otherwise
    """
    video_dir = VIDEOS_DIR / exercise_name
    angle_dir = ANGLES_DIR / exercise_name
    
    success = True
    
    try:
        if video_dir.exists():
            shutil.rmtree(video_dir)
            print(f"Deleted video folder: {video_dir}")
        
        if angle_dir.exists():
            shutil.rmtree(angle_dir)
            print(f"Deleted angles folder: {angle_dir}")
        
        return True
    
    except Exception as e:
        print(f"Error deleting exercise '{exercise_name}': {e}")
        return False


def exercise_exists(exercise_name: str) -> bool:
    """
    Check if an exercise already exists.
    
    Args:
        exercise_name: Name to check
    
    Returns:
        True if exercise exists, False otherwise
    """
    angle_path = ANGLES_DIR / exercise_name / "angles_timeline.json"
    return angle_path.exists()
