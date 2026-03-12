import { useEffect, useState } from 'react';
import '../styles/ExerciseList.css';

function ExerciseList({ onSelectExercise, onUploadNew }) {
  const [exercises, setExercises] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedExercise, setSelectedExercise] = useState(null);

  useEffect(() => {
    fetchExercises();
  }, []);

  const fetchExercises = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/exercises');
      if (!response.ok) throw new Error('Failed to fetch exercises');
      
      const data = await response.json();
      setExercises(data.exercises || []);
    } catch (err) {
      setError(`❌ Error loading exercises: ${err.message}`);
      console.error('Fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectExercise = (exerciseName) => {
    setSelectedExercise(exerciseName);
    setTimeout(() => {
      onSelectExercise(exerciseName);
    }, 300);
  };

  const handleDeleteExercise = async (e, exerciseName) => {
    e.stopPropagation();
    
    if (!window.confirm(`Delete exercise "${exerciseName}"?`)) return;

    try {
      const response = await fetch(`http://localhost:8000/exercise/${exerciseName}`, {
        method: 'DELETE',
      });

      if (!response.ok) throw new Error('Delete failed');

      setExercises(exercises.filter(ex => ex !== exerciseName));
      setSelectedExercise(null);
    } catch (err) {
      alert(`❌ Error deleting exercise: ${err.message}`);
    }
  };

  return (
    <div className="exercise-list-container">
      <div className="exercise-list-header">
        <h2>🏋️ Available Exercises</h2>
        <button onClick={onUploadNew} className="btn-upload-new">
          ➕ Upload New Exercise
        </button>
      </div>

      {isLoading && <p className="loading">⏳ Loading exercises...</p>}
      {error && <p className="error-message">{error}</p>}

      {!isLoading && exercises.length === 0 && (
        <div className="no-exercises">
          <p>No exercises yet. Upload your first reference video!</p>
          <button onClick={onUploadNew} className="btn-start">
            Start by Uploading an Exercise
          </button>
        </div>
      )}

      {!isLoading && exercises.length > 0 && (
        <div className="exercise-grid">
          {exercises.map((exerciseName) => (
            <div
              key={exerciseName}
              className={`exercise-card ${selectedExercise === exerciseName ? 'selected' : ''}`}
              onClick={() => handleSelectExercise(exerciseName)}
            >
              <div className="exercise-name">{exerciseName}</div>
              <div className="exercise-actions">
                <button
                  className="btn-select"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSelectExercise(exerciseName);
                  }}
                >
                  Select
                </button>
                <button
                  className="btn-delete"
                  onClick={(e) => handleDeleteExercise(e, exerciseName)}
                >
                  🗑️
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && exercises.length > 0 && (
        <div className="refresh-section">
          <button onClick={fetchExercises} className="btn-refresh">
            🔄 Refresh List
          </button>
        </div>
      )}
    </div>
  );
}

export default ExerciseList;
