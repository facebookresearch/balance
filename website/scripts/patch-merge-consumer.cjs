/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const replacement = require('deepmerge/package.json');
assert.equal(replacement.name, '@fastify/deepmerge', 'Review the merge replacement before applying this pinned consumer patch');
assert.equal(replacement.version, '3.2.4');
assert.equal(require('@svgr/plugin-svgo/package.json').version, '6.5.1', 'Review the audited merge dependency graph before changing this consumer');
const manifest = require.resolve('fork-ts-checker-webpack-plugin/package.json');
assert.equal(require(manifest).version, '6.5.3', 'Review the merge consumer before applying this pinned patch');
const target = path.join(path.dirname(manifest), 'lib/ForkTsCheckerWebpackPlugin.js');
const before = 'const deepmerge_1 = __importDefault(require("deepmerge"));';
const after = 'const deepmerge_1 = __importDefault(require("deepmerge")({symbols: true}));';
const source = fs.readFileSync(target, 'utf8');
if (source.includes(after)) {
  assert.equal(source.split(after).length, 2, 'Unexpected repeated merge adapter');
  assert.equal(source.includes(before), false, 'Incomplete merge consumer patch');
} else {
  assert.equal(source.split(before).length, 2, 'Unexpected merge consumer source; review compatibility');
  fs.writeFileSync(target, source.replace(before, after));
}
