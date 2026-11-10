/* Shared by the browser and the offline layout regression tests. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.NetworkLayout = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function visibleGraph(cy) {
    const nodes = cy.nodes().not(".hidden");
    const ids = new Set(nodes.map((n) => n.id()));
    return nodes.union(cy.edges().filter((e) =>
      ids.has(e.source().id()) && ids.has(e.target().id())));
  }

  function run(cy, spacing = 1, advanced = true) {
    const elements = visibleGraph(cy);
    if (!elements.nodes().length) return elements;
    const gap = Math.max(1, Math.min(1.8, Number(spacing) || 1));
    // fCoSE's proof pass accounts for label bounds as well as node bodies.
    // Years remain metadata; no positional or alignment constraints are used.
    const options = advanced ? {
      name: "fcose", quality: "proof", randomize: true,
      nodeDimensionsIncludeLabels: true,
      nodeSeparation: 100 * gap,
      idealEdgeLength: () => 120 * gap,
      nodeRepulsion: () => 9000 * gap * gap,
      edgeElasticity: () => 0.25,
      gravity: 0.12, gravityRange: 4.5,
      numIter: 3500,
      tile: true, packComponents: false,
      tilingPaddingVertical: 65 * gap, tilingPaddingHorizontal: 65 * gap,
    } : {
      name: "cose", randomize: true,
      nodeDimensionsIncludeLabels: true,
      nodeRepulsion: () => 12000 * gap * gap,
      idealEdgeLength: () => 120 * gap,
      componentSpacing: 100 * gap, nodeOverlap: 20 * gap,
    };
    elements.layout({ ...options, animate: false, fit: false }).run();
    separateBounds(elements.nodes(), 12 * gap);
    return elements;
  }

  // Leave a small gutter around labels after the spring simulation. Uniform
  // expansion preserves the network's angles and edge-crossing count.
  function separateBounds(nodes, gutter) {
    const occupied = new Set();
    nodes.forEach((n) => {
      let { x, y } = n.position();
      while (occupied.has(`${x},${y}`)) x += n.outerWidth() + gutter;
      occupied.add(`${x},${y}`);
      n.position({ x, y });
    });
    const boxes = nodes.map((n) => {
      const { x, y } = n.position();
      const b = n.boundingBox({ includeLabels: true, includeOverlays: false });
      return { n, x, y, left: x - b.x1, right: b.x2 - x, top: y - b.y1, bottom: b.y2 - y };
    });
    let scale = 1;
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j], dx = b.x - a.x, dy = b.y - a.y;
        const sx = dx ? ((dx > 0 ? a.right + b.left : b.right + a.left) + gutter) / Math.abs(dx) : Infinity;
        const sy = dy ? ((dy > 0 ? a.bottom + b.top : b.bottom + a.top) + gutter) / Math.abs(dy) : Infinity;
        scale = Math.max(scale, Math.min(sx, sy));
      }
    }
    if (scale > 1) {
      nodes.cy().batch(() => boxes.forEach(({ n, x, y }) => n.position({ x: x * scale, y: y * scale })));
    }
  }

  return { run, visibleGraph };
});
