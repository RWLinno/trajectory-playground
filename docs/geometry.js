export function trajectory(a, c) {
  return Array.from({ length: 65 }, (_, i) => {
    const x = -0.9 + (1.8 * i) / 64;
    return [x, c[2] * x + a * Math.sin((Math.PI * i) / 64)];
  });
}
export function evaluate(paths, c) {
  let validCount = 0,
    lengthSum = 0;
  const modes = new Set();
  const center = [c[1], c[2] * c[1]];
  for (const path of paths) {
    let valid = path.every((p) => p.every((v) => Math.abs(v) <= 1));
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1],
        b = path[i],
        d = [b[0] - a[0], b[1] - a[1]];
      const length = Math.hypot(...d);
      lengthSum += length;
      const t = Math.max(
        0,
        Math.min(
          1,
          ((center[0] - a[0]) * d[0] + (center[1] - a[1]) * d[1]) /
            Math.max(length * length, 1e-15),
        ),
      );
      if (
        Math.hypot(a[0] + t * d[0] - center[0], a[1] + t * d[1] - center[1]) <=
        c[0]
      )
        valid = false;
    }
    if (valid) {
      validCount++;
      modes.add(path[Math.floor(path.length / 2)][1] > 0);
    }
  }
  return {
    valid_rate: validCount / paths.length,
    valid_modes: modes.size,
    mean_length: lengthSum / paths.length,
  };
}
