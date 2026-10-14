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

import { bothRoutes, classify } from '../docs/diagnostics.js';
test('finite bank route coverage matches exhaustive enumeration',()=>{
 const labels=[1,1,-1,0];
 for(let k=1;k<=4;k++){
  let all=0,both=0;
  for(let mask=0;mask<16;mask++){const subset=labels.filter((_,i)=>mask&(1<<i));if(subset.length!==k)continue;all++;if(subset.includes(1)&&subset.includes(-1))both++;}
  assert.ok(Math.abs(bothRoutes(2,1,4,k)-both/all)<1e-12);
 }
 assert.equal(bothRoutes(0,0,64,64),0);
 assert.equal(bothRoutes(32,32,64,64),1);
 assert.equal(classify([0,.5,-.5],[.2,0,0]).filter(r=>r.valid).length,2);
});
