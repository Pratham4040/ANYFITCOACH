import { useState } from 'react';
import '../styles/UploadExercise.css';

function UploadExercise({ onUploadSuccess, onCancel }) {
  const [exerciseName, setExerciseName] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setError(null);
    } else {
      setError('Please select a valid video file');
      setVideoFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!exerciseName.trim()) {
      setError('Exercise name is required');
      return;
    }
    
    if (!videoFile) {
      setError('Video file is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('exercise_name', exerciseName);
      formData.append('video_file', videoFile);

      const response = await fetch('http://localhost:8000/process-exercise', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      setSuccessMessage(`✅ Exercise "${exerciseName}" uploaded successfully!`);
      setExerciseName('');
      setVideoFile(null);
      
      // Call parent callback after 1.5s to let user see success message
      setTimeout(() => {
        onUploadSuccess(exerciseName);
      }, 1500);
    } catch (err) {
      setError(`❌ Upload error: ${err.message}`);
      console.error('Upload error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-card">
        <h2>📹 Upload Reference Exercise Video</h2>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="exercise-name">Exercise Name:</label>
            <input
              id="exercise-name"
              type="text"
              value={exerciseName}
              onChange={(e) => setExerciseName(e.target.value)}
              placeholder="e.g., Bicep Curl, Push-up, Squat"
              disabled={isLoading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="video-file">Video File:</label>
            <input
              id="video-file"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              disabled={isLoading}
            />
            {videoFile && <p className="file-selected">✓ {videoFile.name}</p>}
          </div>

          {error && <div className="error-message">{error}</div>}
          {successMessage && <div className="success-message">{successMessage}</div>}

          <div className="button-group">
            <button
              type="submit"
              disabled={isLoading || !videoFile || !exerciseName.trim()}
              className="btn-submit"
            >
              {isLoading ? '⏳ Processing...' : '📤 Upload & Process'}
            </button>
            <button
              type="button"
              onClick={onCancel}
              disabled={isLoading}
              className="btn-cancel"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default UploadExercise;
