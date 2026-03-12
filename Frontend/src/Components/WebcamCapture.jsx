import { useRef, useEffect, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { computeFrameAngles } from '../utils/angleCalculator';
import { findBestDtwMatch } from '../utils/dtwMatcher';
import '../styles/WebcamCapture.css';

const MATCH_INTERVAL_FRAMES = 5;
const SEARCH_WINDOW = 24;
const LIVE_DTW_WINDOW = 18;
const MIN_DTW_FRAMES = 8;
const DTW_BAND_RADIUS = 10;
const FEEDBACK_COOLDOWN_MS = 1200;
const CYCLE_RESET_THRESHOLD = 12;
const CYCLE_DEBOUNCE_MS = 900;

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
];

const JOINT_HINTS = {
  right_elbow: 'bend your right elbow a bit more',
  left_elbow: 'bend your left elbow a bit more',
  right_shoulder: 'stabilize your right shoulder',
  left_shoulder: 'stabilize your left shoulder',
  right_hip: 'check your right hip alignment',
  left_hip: 'check your left hip alignment',
  right_knee: 'align your right knee angle',
  left_knee: 'align your left knee angle',
};

function buildCoachingCue(stats) {
  if (!stats || stats.validCount === 0) {
    return { level: 'neutral', message: 'Tracking your pose...' };
  }

  const worstJointLabel = (stats.worstJoint || 'joint').replace('_', ' ');
  const worstHint = JOINT_HINTS[stats.worstJoint] || `adjust your ${worstJointLabel}`;

  if (stats.worstValue >= 28 || stats.average >= 20) {
    return {
      level: 'critical',
      message: `Major correction: ${worstHint}.`,
    };
  }

  if (stats.worstValue >= 16 || stats.average >= 12) {
    return {
      level: 'warning',
      message: `Form cue: ${worstHint}.`,
    };
  }

  return {
    level: 'good',
    message: 'Nice control. Keep this form.',
  };
}

function computeDeviation(liveAngles, referenceAngles) {
  let total = 0;
  let validCount = 0;
  let worstJoint = null;
  let worstValue = -1;

  Object.entries(liveAngles).forEach(([joint, liveValue]) => {
    const refValue = referenceAngles[joint];
    if (liveValue === null || refValue === null || refValue === undefined) {
      return;
    }

    const delta = Math.abs(liveValue - refValue);
    total += delta;
    validCount += 1;

    if (delta > worstValue) {
      worstValue = delta;
      worstJoint = joint;
    }
  });

  return {
    average: validCount > 0 ? total / validCount : 0,
    validCount,
    worstJoint,
    worstValue: worstValue > 0 ? worstValue : 0,
  };
}

function findBestMatchFrame(liveAngles, referenceTimeline, centerIndex, windowSize) {
  const maxIndex = referenceTimeline.length - 1;
  const start = Math.max(0, centerIndex - windowSize);
  const end = Math.min(maxIndex, centerIndex + windowSize);

  let bestIndex = centerIndex;
  let bestDeviation = Infinity;
  let bestStats = null;

  for (let idx = start; idx <= end; idx += 1) {
    const candidate = referenceTimeline[idx];
    if (!candidate || !candidate.angles) {
      continue;
    }

    const stats = computeDeviation(liveAngles, candidate.angles);
    if (stats.validCount === 0) {
      continue;
    }

    if (stats.average < bestDeviation) {
      bestDeviation = stats.average;
      bestIndex = idx;
      bestStats = stats;
    }
  }

  return {
    bestIndex,
    stats: bestStats,
  };
}

function drawSkeleton(ctx, landmarks, width, height) {
  ctx.clearRect(0, 0, width, height);

  if (!landmarks || landmarks.length === 0) {
    return;
  }

  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(0, 255, 128, 0.85)';

  POSE_CONNECTIONS.forEach(([i, j]) => {
    const a = landmarks[i];
    const b = landmarks[j];
    if (!a || !b) {
      return;
    }

    const aVisible = a.visibility === undefined || a.visibility > 0.35;
    const bVisible = b.visibility === undefined || b.visibility > 0.35;
    if (!aVisible || !bVisible) {
      return;
    }

    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  });

  landmarks.forEach((lm) => {
    const visible = lm.visibility === undefined || lm.visibility > 0.35;
    if (!visible) {
      return;
    }

    ctx.beginPath();
    ctx.fillStyle = 'rgba(255, 214, 10, 0.95)';
    ctx.arc(lm.x * width, lm.y * height, 3, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function WebcamCapture({ exerciseName }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const fpsCounterRef = useRef(0);
  const fpsTimerRef = useRef(null);
  const lastMatchedIndexRef = useRef(0);
  const matchFrameCounterRef = useRef(0);
  const lastCueRef = useRef({ message: '', at: 0 });
  const lastCycleAtRef = useRef(0);
  const liveAngleBufferRef = useRef([]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState(null);
  const [angles, setAngles] = useState(null);
  const [blueprintTimeline, setBlueprintTimeline] = useState([]);
  const [blueprintLoaded, setBlueprintLoaded] = useState(false);
  const [matchInfo, setMatchInfo] = useState(null);
  const [coachCue, setCoachCue] = useState({
    level: 'neutral',
    message: 'Load an exercise and start moving.',
  });
  const [debugLog, setDebugLog] = useState([]);
  const [showDebug, setShowDebug] = useState(false);
  const [fps, setFps] = useState(0);
  const [cycleCount, setCycleCount] = useState(0);

  const addDebug = (msg) => {
    console.log(msg);
    setDebugLog((prev) => [...prev.slice(-9), msg]);
  };

  // --- Effect 0: Load reference blueprint for selected exercise ---
  useEffect(() => {
    if (!exerciseName) {
      return;
    }

    let isCancelled = false;

    const loadBlueprint = async () => {
      setBlueprintLoaded(false);
      setMatchInfo(null);
      setCoachCue({ level: 'neutral', message: 'Loading reference blueprint...' });
      lastMatchedIndexRef.current = 0;
      matchFrameCounterRef.current = 0;
      liveAngleBufferRef.current = [];
      lastCueRef.current = { message: '', at: 0 };
      lastCycleAtRef.current = 0;
      setCycleCount(0);

      try {
        addDebug(` Loading blueprint: ${exerciseName}`);
        const response = await fetch(`http://localhost:8000/exercise/${exerciseName}`);
        if (!response.ok) {
          throw new Error(`Blueprint fetch failed (${response.status})`);
        }

        const payload = await response.json();
        const timeline = Array.isArray(payload.angles) ? payload.angles : [];

        if (!isCancelled) {
          setBlueprintTimeline(timeline);
          setBlueprintLoaded(timeline.length > 0);
          setCoachCue({
            level: timeline.length > 0 ? 'neutral' : 'critical',
            message: timeline.length > 0
              ? 'Reference loaded. Start moving to get feedback.'
              : 'Reference blueprint is empty.',
          });
          addDebug(` Blueprint ready: ${timeline.length} frames`);
        }
      } catch (error) {
        if (!isCancelled) {
          setBlueprintTimeline([]);
          setBlueprintLoaded(false);
          setCoachCue({ level: 'critical', message: 'Failed to load reference blueprint.' });
          addDebug(`❌ Blueprint error: ${error.message}`);
        }
      }
    };

    loadBlueprint();

    return () => {
      isCancelled = true;
    };
  }, [exerciseName]);

  // --- Effect 1: Start camera ---
  useEffect(() => {
    const startCamera = async () => {
      try {
        addDebug('📷 Requesting camera access...');
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        });
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
        video.srcObject.getTracks().forEach((track) => track.stop());
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
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
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

    // FPS counter
    fpsTimerRef.current = setInterval(() => {
      setFps(fpsCounterRef.current);
      fpsCounterRef.current = 0;
    }, 1000);

    const detectFrame = () => {
      const video = videoRef.current;

      if (video && video.readyState >= 2) {
        const timestamp = performance.now();

        if (timestamp !== lastTimestamp) {
          try {
            const result = landmarkerRef.current.detectForVideo(video, timestamp);

            if (result.landmarks && result.landmarks.length > 0) {
              const detectedLandmarks = result.landmarks[0];
              setLandmarks(detectedLandmarks);

              const canvas = canvasRef.current;
              if (canvas && video.videoWidth > 0 && video.videoHeight > 0) {
                if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                  canvas.width = video.videoWidth;
                  canvas.height = video.videoHeight;
                }
                const ctx = canvas.getContext('2d');
                if (ctx) {
                  drawSkeleton(ctx, detectedLandmarks, canvas.width, canvas.height);
                }
              }

              // Compute joint angles from landmarks
              const frameAngles = computeFrameAngles(detectedLandmarks);
              setAngles(frameAngles);

              liveAngleBufferRef.current.push(frameAngles);
              if (liveAngleBufferRef.current.length > LIVE_DTW_WINDOW) {
                liveAngleBufferRef.current.shift();
              }

              if (blueprintLoaded && blueprintTimeline.length > 0) {
                matchFrameCounterRef.current += 1;

                if (matchFrameCounterRef.current >= MATCH_INTERVAL_FRAMES) {
                  matchFrameCounterRef.current = 0;

                  const previousIndex = lastMatchedIndexRef.current;
                  let wrappedCycle = false;

                  let match = null;
                  const startFrame = blueprintTimeline[0];
                  const nearEnd = previousIndex >= Math.floor(blueprintTimeline.length * 0.85);
                  if (nearEnd && startFrame && startFrame.angles) {
                    const startStats = computeDeviation(frameAngles, startFrame.angles);
                    if (
                      startStats.validCount > 0
                      && startStats.average <= CYCLE_RESET_THRESHOLD
                    ) {
                      match = {
                        bestIndex: 0,
                        stats: startStats,
                      };
                      wrappedCycle = true;
                    }
                  }

                  if (!match) {
                    const liveWindow = liveAngleBufferRef.current;

                    if (liveWindow.length >= MIN_DTW_FRAMES) {
                      const dtwResult = findBestDtwMatch({
                        liveWindow,
                        referenceTimeline: blueprintTimeline,
                        centerIndex: previousIndex,
                        searchWindow: SEARCH_WINDOW,
                        bandRadius: DTW_BAND_RADIUS,
                      });

                      if (dtwResult) {
                        const refFrame = blueprintTimeline[dtwResult.matchedIndex];
                        if (refFrame && refFrame.angles) {
                          match = {
                            bestIndex: dtwResult.matchedIndex,
                            stats: computeDeviation(frameAngles, refFrame.angles),
                            dtwCost: dtwResult.normalizedCost,
                          };
                        }
                      }
                    }

                    if (!match) {
                      match = findBestMatchFrame(
                        frameAngles,
                        blueprintTimeline,
                        previousIndex,
                        SEARCH_WINDOW
                      );
                    }
                  }

                  if (match.stats) {
                    lastMatchedIndexRef.current = match.bestIndex;

                    if (wrappedCycle) {
                      const now = Date.now();
                      if (now - lastCycleAtRef.current > CYCLE_DEBOUNCE_MS) {
                        setCycleCount((prev) => prev + 1);
                        lastCycleAtRef.current = now;
                        addDebug(` Rep completed. Total cycles: ${cycleCount + 1}`);
                      }
                    }

                    setMatchInfo({
                      matchedIndex: match.bestIndex,
                      averageDeviation: match.stats.average,
                      worstJoint: match.stats.worstJoint,
                      worstValue: match.stats.worstValue,
                      validJointCount: match.stats.validCount,
                      dtwCost: match.dtwCost,
                    });

                    const nextCue = buildCoachingCue(match.stats);
                    const now = Date.now();
                    const canUpdateCue =
                      nextCue.message !== lastCueRef.current.message
                      || now - lastCueRef.current.at > FEEDBACK_COOLDOWN_MS;

                    if (canUpdateCue) {
                      setCoachCue(nextCue);
                      lastCueRef.current = { message: nextCue.message, at: now };
                    }
                  }
                }
              }

              frameCount += 1;
              fpsCounterRef.current += 1;
              if (frameCount % 30 === 0) {
                addDebug(` Frame #${frameCount}`);
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
      if (fpsTimerRef.current) {
        clearInterval(fpsTimerRef.current);
      }
    };
  }, [isStreaming, modelLoaded, blueprintLoaded, blueprintTimeline]);

  return (
    <div className="webcam-container">
      <div className="video-section">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="video-stream"
        />
        <canvas ref={canvasRef} className="skeleton-canvas" />

        <div className="status-bar">
          <div className="status-item">
            <span className={isStreaming ? 'status-dot active' : 'status-dot'}></span>
            {isStreaming ? 'Camera: OK' : 'Camera: Starting...'}
          </div>
          <div className="status-item">
            <span className={modelLoaded ? 'status-dot active' : 'status-dot'}></span>
            {modelLoaded ? 'Model: Ready' : 'Model: Loading...'}
          </div>
          <div className="status-item">FPS: {fps}</div>
          <div className="status-item">
            Blueprint: {blueprintLoaded ? `${blueprintTimeline.length} frames` : 'loading...'}
          </div>
          <div className="status-item">Cycles: {cycleCount}</div>
          {landmarks && (
            <div className="status-item">
              ✓ {landmarks.length} Landmarks
            </div>
          )}
        </div>
      </div>

      {angles && (
        <div className="angles-panel">
          <h3>Live Joint Angles</h3>
          <div className="angles-grid">
            {Object.entries(angles).map(([jointName, angleValue]) => (
              <div
                key={jointName}
                className={`angle-box ${angleValue === null ? 'disabled' : 'active'}`}
              >
                <div className="angle-label">{jointName.replace('_', ' ')}</div>
                <div className="angle-value">
                  {angleValue !== null ? `${angleValue.toFixed(0)}°` : 'N/A'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {matchInfo && (
        <div className="match-panel">
          <div className="match-item">
            <span className="match-label">Matched Frame</span>
            <span className="match-value">{matchInfo.matchedIndex + 1}</span>
          </div>
          <div className="match-item">
            <span className="match-label">Avg Deviation</span>
            <span className="match-value">{matchInfo.averageDeviation.toFixed(1)} deg</span>
          </div>
          <div className="match-item">
            <span className="match-label">Worst Joint</span>
            <span className="match-value">{matchInfo.worstJoint ? matchInfo.worstJoint.replace('_', ' ') : 'N/A'}</span>
          </div>
          <div className="match-item">
            <span className="match-label">Worst Delta</span>
            <span className="match-value">{matchInfo.worstValue.toFixed(1)} deg</span>
          </div>
          <div className="match-item">
            <span className="match-label">DTW Cost</span>
            <span className="match-value">
              {Number.isFinite(matchInfo.dtwCost) ? matchInfo.dtwCost.toFixed(2) : 'fallback'}
            </span>
          </div>
        </div>
      )}

      {coachCue && (
        <div className={`coach-cue ${coachCue.level}`}>
          {coachCue.message}
        </div>
      )}

      <button
        className="btn-debug-toggle"
        onClick={() => setShowDebug(!showDebug)}
      >
        {showDebug ? 'Hide Debug' : 'Show Debug'}
      </button>

      {showDebug && (
        <div className="debug-panel">
          <div className="debug-header">Debug Log</div>
          <div className="debug-content">
            {debugLog.map((msg, idx) => (
              <div key={idx} className="debug-line">{msg}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default WebcamCapture;