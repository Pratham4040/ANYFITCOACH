// Dynamic Time Warping matcher for live-vs-reference angle timelines.
// Uses a constrained band and short live buffer for real-time browser performance.

function frameDistance(anglesA, anglesB) {
  let total = 0;
  let valid = 0;

  Object.entries(anglesA).forEach(([joint, valueA]) => {
    const valueB = anglesB[joint];
    if (valueA === null || valueB === null || valueB === undefined) {
      return;
    }

    total += Math.abs(valueA - valueB);
    valid += 1;
  });

  if (valid === 0) {
    return Number.POSITIVE_INFINITY;
  }

  return total / valid;
}

export function findBestDtwMatch({
  liveWindow,
  referenceTimeline,
  centerIndex,
  searchWindow,
  bandRadius,
}) {
  if (!Array.isArray(liveWindow) || liveWindow.length < 2) {
    return null;
  }

  if (!Array.isArray(referenceTimeline) || referenceTimeline.length === 0) {
    return null;
  }

  const maxIndex = referenceTimeline.length - 1;
  const start = Math.max(0, centerIndex - searchWindow);
  const end = Math.min(maxIndex, centerIndex + searchWindow);
  const refSlice = referenceTimeline.slice(start, end + 1);

  if (refSlice.length === 0) {
    return null;
  }

  const n = liveWindow.length;
  const m = refSlice.length;

  // dp[i][j] is minimum cumulative cost for first i live frames and first j ref frames.
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(Number.POSITIVE_INFINITY));
  dp[0][0] = 0;

  for (let i = 1; i <= n; i += 1) {
    const expectedJ = Math.max(1, Math.round((i / n) * m));
    const jMin = Math.max(1, expectedJ - bandRadius);
    const jMax = Math.min(m, expectedJ + bandRadius);

    for (let j = jMin; j <= jMax; j += 1) {
      const refFrame = refSlice[j - 1];
      if (!refFrame || !refFrame.angles) {
        continue;
      }

      const localCost = frameDistance(liveWindow[i - 1], refFrame.angles);
      if (!Number.isFinite(localCost)) {
        continue;
      }

      const prev = Math.min(
        dp[i - 1][j],
        dp[i][j - 1],
        dp[i - 1][j - 1]
      );

      if (!Number.isFinite(prev)) {
        continue;
      }

      dp[i][j] = localCost + prev;
    }
  }

  // Subsequence-style endpoint selection: choose cheapest endpoint in later half of candidate slice.
  const endSearchStart = Math.max(1, Math.floor(m * 0.5));
  let bestJ = null;
  let bestNormalizedCost = Number.POSITIVE_INFINITY;

  for (let j = endSearchStart; j <= m; j += 1) {
    const cost = dp[n][j];
    if (!Number.isFinite(cost)) {
      continue;
    }

    const normalized = cost / (n + j);
    if (normalized < bestNormalizedCost) {
      bestNormalizedCost = normalized;
      bestJ = j;
    }
  }

  if (bestJ === null) {
    return null;
  }

  return {
    matchedIndex: start + (bestJ - 1),
    normalizedCost: bestNormalizedCost,
    referenceRange: [start, end],
  };
}
