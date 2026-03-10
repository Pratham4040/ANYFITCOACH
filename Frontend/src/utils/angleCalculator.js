// Angle calculation utilities for pose landmarks
// Mirrors backend angle_calculator.py logic in JavaScript

// Define angle triplets (joint_a, joint_b, joint_c) where angle is at joint_b
// Landmark indices match MediaPipe Pose standard 33-point model
export const ANGLE_TRIPLETS = {
  right_elbow: [11, 13, 15],      // shoulder, elbow, wrist
  left_elbow: [12, 14, 16],       // shoulder, elbow, wrist
  right_shoulder: [13, 11, 23],   // elbow, shoulder, hip
  left_shoulder: [14, 12, 24],    // elbow, shoulder, hip
  right_hip: [11, 23, 25],        // shoulder, hip, knee
  left_hip: [12, 24, 26],         // shoulder, hip, knee
  right_knee: [23, 25, 27],       // hip, knee, ankle
  left_knee: [24, 26, 28],        // hip, knee, ankle
};

const CONFIDENCE_THRESHOLD = 0.5;

/**
 * Compute angle at joint p2 formed by vectors (p1→p2) and (p3→p2)
 * @param {Object} p1 - First point {x, y, z, visibility}
 * @param {Object} p2 - Middle point (angle vertex) {x, y, z, visibility}
 * @param {Object} p3 - Third point {x, y, z, visibility}
 * @returns {number|null} - Angle in degrees, or null if invalid
 */
export function computeAngle(p1, p2, p3) {
  // Create vectors FROM p2 (the joint where angle is measured)
  const u = {
    x: p1.x - p2.x,
    y: p1.y - p2.y,
    z: p1.z - p2.z,
  };

  const v = {
    x: p3.x - p2.x,
    y: p3.y - p2.y,
    z: p3.z - p2.z,
  };

  // Compute magnitudes (Euclidean length)
  const uMag = Math.sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
  const vMag = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);

  // Handle zero-length vectors (invalid landmarks)
  if (uMag === 0 || vMag === 0) {
    return null;
  }

  // Compute dot product: u·v = ux*vx + uy*vy + uz*vz
  const dot = u.x * v.x + u.y * v.y + u.z * v.z;

  // Compute cosine of angle
  let cosAngle = dot / (uMag * vMag);

  // Clamp to [-1, 1] to prevent numerical errors in Math.acos
  cosAngle = Math.max(-1, Math.min(1, cosAngle));

  // Compute angle in radians, then convert to degrees
  const angleRad = Math.acos(cosAngle);
  const angleDeg = angleRad * (180 / Math.PI);

  return angleDeg;
}

/**
 * Compute all standard angles for a single frame of pose landmarks
 * @param {Array} landmarks - Array of 33 landmark objects with x, y, z, visibility
 * @returns {Object} - Map of angle_name → angle_value (or null if low confidence)
 */
export function computeFrameAngles(landmarks) {
  if (!landmarks || landmarks.length !== 33) {
    return {};
  }

  const frameAngles = {};

  for (const [angleName, [idxA, idxB, idxC]] of Object.entries(ANGLE_TRIPLETS)) {
    // Get the three landmarks
    const pA = landmarks[idxA];
    const pB = landmarks[idxB];
    const pC = landmarks[idxC];

    // Check confidence (visibility threshold)
    if (
      pA.visibility < CONFIDENCE_THRESHOLD ||
      pB.visibility < CONFIDENCE_THRESHOLD ||
      pC.visibility < CONFIDENCE_THRESHOLD
    ) {
      frameAngles[angleName] = null;
      continue;
    }

    // Compute angle
    const angle = computeAngle(pA, pB, pC);
    frameAngles[angleName] = angle;
  }

  return frameAngles;
}
