import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { trajectory, evaluate } from "../docs/geometry.js";
test("segment collision is detected between clear endpoints", () =>
  assert.equal(
    evaluate(
      [
        [
          [-0.9, 0],
          [0.9, 0],
        ],
      ],
      [0.2, 0, 0],
    ).valid_rate,
    0,
  ));
test("displayed metrics agree with Python evaluation for all presets", () => {
  const data = JSON.parse(
    readFileSync(new URL("../docs/assets/samples.json", import.meta.url)),
  );
  for (const scene of data.scenes)
    for (const bundle of Object.values(scene.models)) {
      const actual = evaluate(
        bundle.amplitudes.map((a) => trajectory(a, scene.condition)),
        scene.condition,
      );
      assert.equal(actual.valid_rate, bundle.metrics.valid_rate);
      assert.equal(actual.valid_modes, bundle.metrics.valid_modes);
      assert.ok(
        Math.abs(actual.mean_length - bundle.metrics.mean_length) < 1e-5,
      );
    }
});
