import { useRef, useEffect, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { computeFrameAngles } from '../utils/angleCalculator';

function WebcamCapture() {
  const videoRef = useRef(null);
  const landmarkerRef = useRef(null);       // stores the PoseLandmarker instance
  const animationRef = useRef(null);         // stores the requestAnimationFrame ID (for cleanup)
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState(null);  // stores latest detected landmarks
  const [angles, setAngles] = useState(null);        // stores computed joint angles
  const [debugLog, setDebugLog] = useState([]);       // debug messages

  const addDebug = (msg) => {
    console.log(msg);
    setDebugLog(prev => [...prev.slice(-9), msg]); // keep last 10 messages
  };

  // --- Effect 1: Start camera ---
  useEffect(() => {
    const startCamera = async () => {
      try {
        addDebug('📷 Requesting camera access...');
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          addDebug(' Camera stream started');
          setIsStreaming(true);
        }
      } catch (error) {
        addDebug(`❌ Camera error: ${error.message}`);
        console.error('Camera access denied:', error);
      }
    };

    startCamera();

    return () => {
      const video = videoRef.current;
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        addDebug(' Camera stopped');
      }
    };
  }, []);

  // --- Effect 2: Load MediaPipe model ---
  useEffect(() => {
    const loadModel = async () => {
      try {
        addDebug(' Loading MediaPipe model...');
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        addDebug(' Vision runtime loaded');

        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });

        landmarkerRef.current = poseLandmarker;
        addDebug(' PoseLandmarker model ready');
        setModelLoaded(true);
      } catch (error) {
        addDebug(` Model load error: ${error.message}`);
        console.error('Model loading error:', error);
      }
    };

    loadModel();

    return () => {
      if (landmarkerRef.current) {
        try {
          landmarkerRef.current.close();
          addDebug(' Model closed');
        } catch (e) {
          console.error('Error closing model:', e);
        }
      }
    };
  }, []);

  // --- Effect 3: Detection loop (starts when both camera and model are ready) ---
  useEffect(() => {
    if (!isStreaming || !modelLoaded || !landmarkerRef.current) {
      addDebug(`⏳ Waiting... Stream:${isStreaming} Model:${modelLoaded} Landmarker:${!!landmarkerRef.current}`);
      return;
    }

    addDebug('🎬 Starting detection loop...');
    let lastTimestamp = -1;
    let frameCount = 0;

    const detectFrame = () => {
      const video = videoRef.current;

      if (video && video.readyState >= 2) {
        const timestamp = performance.now();

        if (timestamp !== lastTimestamp) {
          try {
            const result = landmarkerRef.current.detectForVideo(video, timestamp);

            if (result.landmarks && result.landmarks.length > 0) {
              const detectedLandmarks = result.landmarks[0];  // first person's 33 landmarks
              setLandmarks(detectedLandmarks);
              
              // Compute joint angles from landmarks
              const frameAngles = computeFrameAngles(detectedLandmarks);
              setAngles(frameAngles);

              frameCount++;
              if (frameCount % 30 === 0) {
                addDebug(` Detected frame #${frameCount}`);
              }
            }
          } catch (error) {
            console.error('Detection error:', error);
            addDebug(`❌ Detection error: ${error.message}`);
          }

          lastTimestamp = timestamp;
        }
      }

      animationRef.current = requestAnimationFrame(detectFrame);
    };

    animationRef.current = requestAnimationFrame(detectFrame);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isStreaming, modelLoaded]);

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h2>Live Camera</h2>
      
      {/* Debug Panel */}
      <div style={{ 
        margin: '10px 0', 
        padding: '10px', 
        backgroundColor: '#1e1e1e', 
        color: '#0f0',
        borderRadius: '4px',
        fontSize: '12px',
        maxHeight: '120px',
        overflow: 'auto'
      }}>
        <div><strong>Status:</strong></div>
        {debugLog.map((msg, idx) => (
          <div key={idx}>{msg}</div>
        ))}
      </div>

      {/* Video */}
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted 
        width="640"
        height="480"
        style={{ border: '2px solid #ccc', borderRadius: '4px', transform: 'scaleX(-1)' }} 
      />
      <p>{isStreaming ? ' Camera active' : ' Starting camera...'}</p>
      {modelLoaded && <p> Model loaded</p>}
      
      {landmarks && (
        <p> Tracking: {landmarks.length} landmarks detected</p>
      )}
      
      {angles && (
        <div style={{ marginTop: '20px', textAlign: 'left', maxWidth: '600px', margin: '20px auto' }}>
          <h3>Live Joint Angles (degrees)</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {Object.entries(angles).map(([jointName, angleValue]) => (
              <div key={jointName} style={{ 
                padding: '8px', 
                backgroundColor: angleValue === null ? '#555' : '#1a3a1a',
                borderRadius: '4px',
                color: angleValue === null ? '#999' : '#0f0'
              }}>
                <strong>{jointName.replace('_', ' ').toUpperCase()}:</strong>{' '}
                {angleValue !== null ? `${angleValue.toFixed(1)}°` : 'N/A'}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default WebcamCapture;