import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import { buildPhotoWarpGeometry, denseMeshToDlib68, detectDlib68FromImage } from "../assets/js/photo-warp.mjs";

function parser() {
  const context = vm.createContext({
    URL, document: { baseURI: "https://example.test/" },
    window: { addEventListener() {} },
  });
  vm.runInContext(readFileSync(new URL("../assets/js/app.js", import.meta.url), "utf8"), context);
  return (value) => {
    const result = context.flattenJsonLandmarks(value);
    context.validateCoordinateArray(result);
    return Array.from(result);
  };
}

const points = Array.from({ length: 68 }, (_, i) => [i * .01, Math.sin(i) * .02]);

test("JSON coordinates preserve the documented flat and paired layouts", () => {
  const parse = parser();
  assert.deepEqual(parse(points), points.flat());
  assert.deepEqual(parse(points.flat()), points.flat());
  assert.deepEqual(parse({ landmarks: points }), points.flat());
});

test("JSON parsing rejects nested, transposed, and wrong-width matrices", () => {
  const parse = parser();
  for (const value of [
    [points], [points.map(p => p[0]), points.map(p => p[1])],
    Array.from({length:34},(_, i)=>points.flat().slice(4*i,4*i+4)),
  ]) assert.throws(() => parse(value), /landmark|coordinate|68|136/i);
});

test("JSON parsing cannot turn missing values, booleans, or strings into measurements", () => {
  const parse = parser();
  for (const value of [null, true, false, "1.25", "", [], [1]]) {
    const payload = points.flat(); payload[0] = value;
    assert.throws(() => parse(payload), /number|coordinate|landmark/i);
  }
});

test("legacy adapter rejects missing mesh vertices and invalid pixel dimensions", () => {
  const mesh = Array.from({length:478},()=>({x:.2,y:.3,z:0}));
  delete mesh[11];
  assert.throws(() => denseMeshToDlib68(mesh,1000,600), /mesh|landmark|finite/i);
  const complete = Array.from({length:478},()=>({x:.2,y:.3,z:0}));
  for (const width of [0,-1,1.5,Infinity,true,"100"])
    assert.throws(() => denseMeshToDlib68(complete,width,600), /width|dimensions|integer/i);
});

test("legacy adapter keeps both complete 468 and 478 point schemas", () => {
  for (const size of [468,478]) {
    const mesh = Array.from({length:size},()=>({x:.2,y:.3,z:0}));
    assert.equal(denseMeshToDlib68(mesh,1000,600).length,68);
  }
});

test("a missing residual cannot silently become a zero-vector image warp", () => {
  const source = Array.from({length:68},(_,i)=>[80+(i%17)*12,70+Math.floor(i/17)*34]);
  for (const bad of [null,false,"0"]) {
    const residuals=source.map(()=>[0,0]); residuals[0][0]=bad;
    assert.throws(()=>buildPhotoWarpGeometry({sourceLandmarks:source,residualVectors:residuals,
      centroidSize:300,scale:4,width:400,height:300}),/finite number/);
  }
});


test("legacy photo detection cannot silently choose a face from a multiple-face image", async()=>{
  const mesh=Array.from({length:478},()=>({x:.2,y:.3,z:0}));
  const task={detect:()=>({faceLandmarks:[mesh,mesh]})};
  await assert.rejects(()=>detectDlib68FromImage(task,{width:1000,height:600}),/exactly one visible face/);
});
