/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// Scoped mitigation of GHSA-gcqx-w48m-w2h4 while deepmerge has no patched release.
const manifest = require.resolve('deepmerge/package.json');
assert.equal(require(manifest).version, '4.3.1');
const root = path.dirname(manifest);
const marker = '// balance: guard both merge input loops against prototype keys';
const loops = ['getKeys(target).forEach(function(key) {', 'getKeys(source).forEach(function(key) {'];
const guard = "\n\t\t\tif (key === '__proto__' || key === 'constructor' || key === 'prototype') return";
const updates = ['index.js', 'dist/cjs.js', 'dist/umd.js'].map(file => {
  const target = path.join(root, file);
  let source = fs.readFileSync(target, 'utf8');
  if (source.includes(marker)) {
    for (const loop of loops) assert(source.includes(loop + guard), 'Incomplete deepmerge mitigation');
    return {target, output: null};
  }
  for (const loop of loops) {
    assert.equal(source.split(loop).length, 2, 'Unexpected deepmerge loop; review the mitigation');
    source = source.replace(loop, loop + guard);
  }
  return {target, output: marker + '\n' + source};
});
for (const {target, output} of updates) if (output !== null) fs.writeFileSync(target, output);
