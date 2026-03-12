import { useState } from 'react';
import WebcamCapture from './Components/WebcamCapture';
import UploadExercise from './Components/UploadExercise';
import ExerciseList from './Components/ExerciseList';
import './App.css';

function App() {
  const [appState, setAppState] = useState('list'); // 'list', 'upload', 'coach'
  const [selectedExercise, setSelectedExercise] = useState(null);

  const handleSelectExercise = (exerciseName) => {
    setSelectedExercise(exerciseName);
    setAppState('coach');
  };

  const handleUploadNew = () => {
    setAppState('upload');
  };

  const handleUploadSuccess = () => {
    setAppState('list');
  };

  const handleCancelUpload = () => {
    setAppState('list');
  };

  const handleBackToList = () => {
    setAppState('list');
    setSelectedExercise(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🏋️ AI Fitness Coach</h1>
        <p>Real-time form feedback powered by AI</p>
      </header>

      <main className="app-main">
        {appState === 'list' && (
          <ExerciseList
            onSelectExercise={handleSelectExercise}
            onUploadNew={handleUploadNew}
          />
        )}

        {appState === 'upload' && (
          <UploadExercise
            onUploadSuccess={handleUploadSuccess}
            onCancel={handleCancelUpload}
          />
        )}

        {appState === 'coach' && selectedExercise && (
          <div className="coach-view">
            <button onClick={handleBackToList} className="btn-back">
              ← Back to Exercises
            </button>
            <h2>🎯 Coaching: {selectedExercise}</h2>
            <WebcamCapture exerciseName={selectedExercise} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;