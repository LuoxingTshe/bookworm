# Browser dependencies

These unmodified npm distributions are served locally so rendering works without
an Internet connection. Their license files are included alongside the scripts.

| Package | Version | Distribution | Upstream |
| --- | --- | --- | --- |
| cytoscape | 3.30.4 | dist/cytoscape.min.js | https://github.com/cytoscape/cytoscape.js |
| layout-base | 2.0.1 | layout-base.js | https://github.com/iVis-at-Bilkent/layout-base |
| cose-base | 2.2.0 | cose-base.js | https://github.com/iVis-at-Bilkent/cose-base |
| cytoscape-fcose | 2.2.0 | cytoscape-fcose.js | https://github.com/iVis-at-Bilkent/cytoscape.js-fcose |

Load in the order above. To update, obtain pinned packages from npm with install
scripts disabled, copy the listed distributions and LICENSE files, update this
table, and run `node --test tests/unit/test_network_layout.cjs` plus the Python suite.

Layout reference: H. Balci and U. Dogrusoz, “fCoSE: A Fast Compound Graph Layout
Algorithm with Constraint Support,” IEEE TVCG 28(12), 4582–4593, 2022.
