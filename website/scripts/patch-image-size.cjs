/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const {createRequire} = require('node:module');
const path = require('node:path');

// Docusaurus 2.4.3 uses the removed image-size file/callback API. Keep its
// image metadata path on the patched parser's public Promise-based API.
const manifestPath = require.resolve('@docusaurus/mdx-loader/package.json');
const mdx = createRequire(manifestPath);
assert.equal(mdx('./package.json').version, '2.4.3', 'Docusaurus changed; review its image-size compatibility before applying this pinned patch');
assert.equal(typeof mdx('image-size/fromFile').imageSizeFromFile, 'function');
const target = path.join(path.dirname(manifestPath), 'lib/remark/transformImage/index.js');
const oldImport = 'const image_size_1 = tslib_1.__importDefault(require("image-size"));';
const newImport = 'const {imageSizeFromFile} = require("image-size/fromFile");';
const oldCall = 'const size = (await (0, util_1.promisify)(image_size_1.default)(imagePath));';
const newCall = 'const size = await imageSizeFromFile(imagePath);';
const source = fs.readFileSync(target, 'utf8');
if (source.includes(newImport) && source.includes(newCall) && !source.includes(oldImport) && !source.includes(oldCall)) {
  process.exit(0);
}
assert.equal(source.split(oldImport).length, 2, 'Unexpected Docusaurus image import; review the compatibility patch');
assert.equal(source.split(oldCall).length, 2, 'Unexpected Docusaurus image call; review the compatibility patch');
fs.writeFileSync(target, source.replace(oldImport, newImport).replace(oldCall, newCall));
