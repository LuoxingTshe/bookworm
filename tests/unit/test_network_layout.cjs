const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const vm = require('node:vm');
// Executes the exact browser distributions, offline and without npm installation.
const { loadRenderer, layout } = require('../../bookworm/layout/layout.cjs');

const renderer = () => loadRenderer(42);

function graph(env, count = 24, year = (i) => 2000 + i % 6) {
  const elements = Array.from({ length: count }, (_, i) => ({ data: {
    id: `n${i}`, label: `Author ${i}`, year: year(i),
  } }));
  for (let i = 1; i < count; i++) {
    elements.push({ data: { id: `e${i}`, source: `n${i}`, target: `n${Math.floor((i - 1) / 2)}` } });
  }
  env.graphOptions = JSON.stringify({ headless: true, styleEnabled: true, elements,
    style: [{ selector: 'node', style: { width: 40, height: 40, label: 'data(label)', 'font-size': 12 } }],
    layout: { name: 'preset' },
  });
  return vm.runInContext('cytoscape(JSON.parse(graphOptions))', env);
}

function assertSeparated(cy) {
  const nodes = cy.nodes().not('.hidden');
  nodes.forEach((a, i) => {
    assert.ok(Number.isFinite(a.position('x')) && Number.isFinite(a.position('y')));
    nodes.slice(i + 1).forEach((b) => {
      assert.ok(Math.hypot(a.position('x') - b.position('x'), a.position('y') - b.position('y')) >= 40,
        `${a.id()} overlaps ${b.id()}`);
    });
  });
}

test('browser bundles register fCoSE and spread connected nodes without overlap', () => {
  const env = renderer(), cy = graph(env);
  try {
    assert.equal(typeof env.cytoscape('layout', 'fcose'), 'function');
    env.NetworkLayout.run(cy);
    assertSeparated(cy);
    assert.equal(cy.edges().length, 23);
  } finally { cy.destroy(); }
});

test('dense networks with differently sized nodes keep bounding boxes apart', () => {
  const env = renderer(), cy = graph(env, 35);
  try {
    cy.nodes().forEach((n, i) => n.style('width', 25 + i * 2).style('height', 25 + i * 2));
    env.extraEdges = JSON.stringify(Array.from({ length: 30 }, (_, i) => ({
      data: { id: `extra${i}`, source: `n${i + 4}`, target: `n${i % 4}` },
    })));
    env.cy = cy;
    vm.runInContext('cy.add(JSON.parse(extraEdges))', env);
    env.NetworkLayout.run(cy);
    cy.nodes().forEach((a, i) => cy.nodes().slice(i + 1).forEach((b) => {
      const x = a.boundingBox(), y = b.boundingBox();
      assert.ok(x.x2 <= y.x1 || y.x2 <= x.x1 || x.y2 <= y.y1 || y.y2 <= x.y1,
        `${a.id()} and ${b.id()} have overlapping bounds`);
    }));
  } finally { cy.destroy(); }
});

test('year metadata does not constrain network positions', () => {
  const env1 = renderer(), env2 = renderer();
  const cy1 = graph(env1), cy2 = graph(env2, 24, () => null);
  try {
    env1.NetworkLayout.run(cy1);
    env2.NetworkLayout.run(cy2);
    for (let i = 0; i < 24; i++) {
      assert.equal(cy1.$id(`n${i}`).position('x'), cy2.$id(`n${i}`).position('x'));
      assert.equal(cy1.$id(`n${i}`).position('y'), cy2.$id(`n${i}`).position('y'));
    }
  } finally { cy1.destroy(); cy2.destroy(); }
});

test('filtering excludes incident edges and leaves hidden positions untouched', () => {
  const env = renderer(), cy = graph(env);
  try {
    cy.$id('n0').addClass('hidden').position('x', 12345).position('y', 54321);
    const visible = env.NetworkLayout.run(cy);
    assert.equal(visible.nodes().length, 23);
    assert.equal(visible.edges().length, 21);
    assert.equal(cy.$id('n0').position('x'), 12345);
    cy.nodes().addClass('hidden');
    assert.equal(env.NetworkLayout.run(cy).length, 0);
    cy.nodes().removeClass('hidden');
    assert.equal(env.NetworkLayout.run(cy).nodes().length, 24);
    assertSeparated(cy);
  } finally { cy.destroy(); }
});

test('empty, singleton, isolated and disconnected graphs have finite separated positions', () => {
  const env = renderer();
  for (const count of [0, 1, 2, 12]) {
    const cy = graph(env, count);
    try {
      cy.edges().remove();
      env.NetworkLayout.run(cy, 1.8);
      assertSeparated(cy);
    } finally { cy.destroy(); }
  }
});

test('CoSE fallback works and wider spacing increases edge lengths', () => {
  const env1 = renderer(), env2 = renderer();
  const cy1 = graph(env1), cy2 = graph(env2);
  const meanEdgeLength = (cy) => cy.edges().reduce((sum, e) => sum + Math.hypot(
    e.source().position('x') - e.target().position('x'),
    e.source().position('y') - e.target().position('y')), 0) / cy.edges().length;
  try {
    env1.NetworkLayout.run(cy1, 1);
    env2.NetworkLayout.run(cy2, 1.8);
    assert.ok(meanEdgeLength(cy2) > meanEdgeLength(cy1) * 1.2);
    env1.NetworkLayout.run(cy1, 1, false);
    assertSeparated(cy1);
  } finally { cy1.destroy(); cy2.destroy(); }
});

// Canvas cards are wide rectangles; the headless entry point used by the Obsidian export must keep them apart too.
function cards(count = 30) {
  const nodes = Array.from({ length: count }, (_, i) => ({ id: `p${i}`, width: 320, height: 150 }));
  const edges = Array.from({ length: count - 1 }, (_, i) => ({ source: `p${i + 1}`, target: `p${Math.floor(i / 3)}` }));
  edges.push({ source: 'p5', target: 'p5' }, { source: 'p1', target: 'missing' }); // ignored
  return { nodes, edges };
}

function assertCardsApart(positions, w = 320, h = 150) {
  const pts = Object.values(positions);
  pts.forEach((a, i) => pts.slice(i + 1).forEach((b) => {
    assert.ok(Math.abs(a.x - b.x) >= w || Math.abs(a.y - b.y) >= h, 'cards overlap');
  }));
}

test('headless layout() keeps Canvas-sized cards apart and is reproducible', () => {
  const first = layout({ ...cards(), seed: 7 });
  assert.equal(first.layout, 'fcose');
  assert.equal(Object.keys(first.positions).length, 30);
  assertCardsApart(first.positions);
  assert.deepEqual(layout({ ...cards(), seed: 7 }).positions, first.positions);
  assert.deepEqual(layout({ nodes: [], edges: [] }).positions, {});
});

test('layout.cjs CLI reads a graph on stdin and prints positions', () => {
  const r = spawnSync(process.execPath, [path.join(__dirname, '../../bookworm/layout/layout.cjs')],
    { input: JSON.stringify(cards(8)), encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  const out = JSON.parse(r.stdout);
  assert.equal(out.layout, 'fcose');
  assertCardsApart(out.positions);
});
