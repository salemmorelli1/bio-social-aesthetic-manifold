import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

class Element {
  constructor() {
    this.children = [];
    this.textContent = "";
    this.className = "";
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
}

function appContext() {
  const source = readFileSync(new URL("../assets/js/app.js", import.meta.url), "utf8");
  const context = vm.createContext({
    URL,
    document: { baseURI: "https://example.test/", createElement: () => new Element() },
    window: { addEventListener: () => {} },
  });
  vm.runInContext(source, context);
  const ui = vm.runInContext("ui", context);
  const state = vm.runInContext("state", context);
  for (const id of [
    "assessmentProvenance", "assessmentStatus", "assessmentIntro",
    "assessmentProportions", "assessmentBalance", "assessmentLimits", "assessmentCaution",
  ]) ui[id] = new Element();
  return { context, ui, state };
}

test("report renders measured, unavailable, provenance, and clears stale values", () => {
  const { context, ui, state } = appContext();
  const assessment = {
    metrics: [
      { key: "mouth_to_nose", label: "Mouth / nose", value: 1.341, unit: "ratio", status: "measured", definition: "Visible ratio." },
      { key: "eyes_paired_discrepancy_pct", label: "Eyes", value: null, unit: "% of jaw span", status: "unavailable", definition: "Paired difference.", reason: "Zero span." },
    ],
    not_assessed: ["Profile depth", "Skin texture"],
    interpretation: "Pose unverified.",
  };
  state.inputKind = "synthetic";
  context.renderAssessment(assessment);
  assert.match(ui.assessmentProvenance.textContent, /Synthetic example/);
  assert.equal(ui.assessmentStatus.textContent, "1 observed");
  assert.equal(ui.assessmentProportions.children[0].children[1].textContent, "1.34");
  assert.equal(ui.assessmentBalance.children[0].children[1].textContent, "Unavailable");
  assert.equal(ui.assessmentLimits.children[1].textContent, "Skin texture");

  state.inputKind = "photo";
  context.renderAssessment(assessment);
  assert.match(ui.assessmentProvenance.textContent, /Local photo/);
  context.resetAssessmentDisplay();
  assert.equal(ui.assessmentStatus.textContent, "Not run");
  assert.equal(ui.assessmentProportions.children[0].textContent, "Measurements will appear after analysis.");
  assert.equal(ui.assessmentBalance.children[0].textContent, "Paired measurements will appear after analysis.");
});

test("every dynamic report element exists in the HTML view", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  for (const id of [
    "tab-assessment", "assessment-view", "assessment-open-lab", "assessment-provenance",
    "assessment-status", "assessment-intro", "assessment-proportions",
    "assessment-balance", "assessment-limits", "assessment-caution",
  ]) assert.ok(html.includes(`id="${id}"`), `Missing #${id}`);
  assert.match(html, /data-view="assessment"/);
  assert.match(html, /data-panel="assessment"/);
});
