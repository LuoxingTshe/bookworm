/* Headless fCoSE network layout (network-layout.js) for the Obsidian Canvas export.
 *
 * Executes the vendored browser bundles in a vm context: no npm, no network.
 *
 * CLI: node bookworm/layout/layout.cjs < {"nodes": [{"id", "width", "height"}], "edges": [{"source", "target"}],
 *                                  "spacing": 1, "seed": 42}
 *      -> {"layout": "fcose", "positions": {"<id>": {"x", "y"}}}   (node centres)
 */
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const STATIC = __dirname;
const BUNDLES = ["vendor/cytoscape.min.js", "vendor/layout-base.js", "vendor/cose-base.js",
  "vendor/cytoscape-fcose.js", "network-layout.js"];

// Seeded Math.random: the layout starts from random positions, and a fixed seed keeps exports reproducible.
function loadRenderer(seed = 42) {
  const math = Object.create(Math);
  math.random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 2 ** 32);
  const context = vm.createContext({ console, Math: math, setTimeout, clearTimeout });
  context.window = context;
  for (const file of BUNDLES) vm.runInContext(fs.readFileSync(path.join(STATIC, file), "utf8"), context);
  return context;
}

function layout({ nodes = [], edges = [], spacing = 1, seed = 42 }) {
  const env = loadRenderer(seed);
  const ids = new Set(nodes.map((n) => n.id));
  const elements = nodes.map((n) => ({ data: { id: n.id, w: n.width, h: n.height } }))
    .concat(edges.filter((e) => ids.has(e.source) && ids.has(e.target) && e.source !== e.target)
      .map((e, i) => ({ data: { id: `e${i}`, source: e.source, target: e.target } })));
  env.graphOptions = JSON.stringify({ headless: true, styleEnabled: true, elements, layout: { name: "preset" },
    style: [{ selector: "node", style: { width: "data(w)", height: "data(h)", shape: "rectangle" } }] });
  const cy = vm.runInContext("cytoscape(JSON.parse(graphOptions))", env);
  try {
    const advanced = !!env.cytoscape("layout", "fcose");
    env.NetworkLayout.run(cy, spacing, advanced);
    const positions = {};
    cy.nodes().forEach((n) => { positions[n.id()] = { x: n.position("x"), y: n.position("y") }; });
    return { layout: advanced ? "fcose" : "cose", positions };
  } finally { cy.destroy(); }
}

if (require.main === module) {
  process.stdout.write(JSON.stringify(layout(JSON.parse(fs.readFileSync(0, "utf8")))));
}

module.exports = { loadRenderer, layout };
