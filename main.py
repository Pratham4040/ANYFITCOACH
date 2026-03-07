
#FastAPI Backend for AI Fitness Coach
#Provides endpoints for exercise video processing and management.


from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

from services import (
    process_video_to_angles,
    list_exercises,
    get_exercise_angles,
    delete_exercise,
    exercise_exists
)
from services.exercise_manager import VIDEOS_DIR, ANGLES_DIR


app = FastAPI(
    title="AI Fitness Coach API",
    description="Backend API for processing exercise videos and providing real-time form feedback",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "AI Fitness Coach API",
        "version": "1.0.0"
    }


@app.get("/exercises")
async def get_exercises():
    """
    List all available exercises.
    
    Returns:
        List of exercise names
    """
    exercises = list_exercises()
    return {
        "exercises": exercises,
        "count": len(exercises)
    }


@app.get("/exercise/{exercise_name}")
async def get_exercise(exercise_name: str):
    """
    Get cached angle timeline for a specific exercise.
    
    Args:
        exercise_name: Name of the exercise
    
    Returns:
        Exercise data with angle timeline
    """
    exercise_data = get_exercise_angles(exercise_name)
    
    if exercise_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Exercise '{exercise_name}' not found"
        )
    
    return exercise_data


@app.post("/process-exercise")
async def process_exercise(
    exercise_name: str = Form(...),
    video_file: UploadFile = File(...)
):
    """
    Process a new exercise video and extract angle timeline.
    
    Args:
        exercise_name: Name for the exercise (used as folder name)
        video_file: Uploaded video file
    
    Returns:
        Processed angle timeline data
    """
    # Validate exercise name (sanitize for filesystem)
    safe_name = "".join(c for c in exercise_name if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    
    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid exercise name. Use only letters, numbers, spaces, hyphens, and underscores."
        )
    
    # Check if exercise already exists
    if exercise_exists(safe_name):
        raise HTTPException(
            status_code=409,
            detail=f"Exercise '{safe_name}' already exists. Delete it first or use a different name."
        )
    
    # Validate file type
    if not video_file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a video"
        )
    
    # Create directories
    video_dir = VIDEOS_DIR / safe_name
    angle_dir = ANGLES_DIR / safe_name
    video_dir.mkdir(parents=True, exist_ok=True)
    angle_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video
    video_path = video_dir / "reference.mp4"
    try:
        with open(video_path, 'wb') as f:
            shutil.copyfileobj(video_file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video file: {str(e)}"
        )
    finally:
        video_file.file.close()
    
    # Process video to extract angles
    output_json_path = angle_dir / "angles_timeline.json"
    result = process_video_to_angles(video_path, output_json_path)
    
    if not result["success"]:
        # Clean up on failure
        shutil.rmtree(video_dir, ignore_errors=True)
        shutil.rmtree(angle_dir, ignore_errors=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {result.get('error', 'Unknown error')}"
        )
    
    return {
        "success": True,
        "exercise_name": safe_name,
        "frames": result["frames"],
        "angles": result["angles"]
    }


@app.delete("/exercise/{exercise_name}")
async def remove_exercise(exercise_name: str):
    """
    Delete an exercise and all its associated files.
    
    Args:
        exercise_name: Name of the exercise to delete
    
    Returns:
        Success confirmation
    """
    if not exercise_exists(exercise_name):
        raise HTTPException(
            status_code=404,
            detail=f"Exercise '{exercise_name}' not found"
        )
    
    success = delete_exercise(exercise_name)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete exercise '{exercise_name}'"
        )
    
    return {
        "success": True,
        "message": f"Exercise '{exercise_name}' deleted successfully"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI Fitness Coach API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
