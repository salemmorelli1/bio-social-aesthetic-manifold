import assert from "node:assert/strict";
import test from "node:test";
import { singleFaceMesh } from "../src/lib/vision/landmarker.ts";
import { denseMeshToDlib68 } from "../src/lib/math/morphometrics.ts";

test("a sparse 478-slot array is not a complete observed face mesh",()=>{
  const mesh=Array.from({length:478},()=>({x:.2,y:.3,z:0}));
  // This is outside the 68 selected indices, so conversion alone missed it.
  delete mesh[11];
  assert.throws(()=>singleFaceMesh({faceLandmarks:[mesh]}),/478|finite/);
  assert.throws(()=>denseMeshToDlib68(mesh,1000,600),/478|finite/);
});

test("sparse face collections cannot invent a single-face capture",()=>{
  const faces=new Array(1);
  assert.throws(()=>singleFaceMesh({faceLandmarks:faces}),/mesh|478|face/i);
});
