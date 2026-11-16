/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const {mkdtempSync, rmSync, writeFileSync, readFileSync} = require('node:fs');
const {createRequire} = require('node:module');
const {tmpdir} = require('node:os');
const path = require('node:path');
const {test} = require('node:test');
const {imageSize} = require('image-size');
const {imageSizeFromFile} = require('image-size/fromFile');
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=', 'base64');
const size = {width: 1, height: 1, type: 'png'};

function temporaryFile(t, contents) {
  const directory = mkdtempSync(path.join(tmpdir(), 'balance-image-size-'));
  t.after(() => rmSync(directory, {recursive: true, force: true}));
  const file = path.join(directory, 'pixel.png');
  writeFileSync(file, contents);
  return file;
}

test('patched parser reads a complete PNG from bytes and a file', async t => {
  assert.deepEqual(imageSize(png), size);
  assert.deepEqual(await imageSizeFromFile(temporaryFile(t, png)), size);
});
test('invalid files and missing paths reject through the public Promise API', async t => {
  const file = temporaryFile(t, Buffer.from('not an image'));
  await assert.rejects(imageSizeFromFile(file), TypeError);
  await assert.rejects(imageSizeFromFile(`${file}.missing`), {code: 'ENOENT'});
});

function box(name, contents) {
  const header = Buffer.alloc(8);
  header.writeUInt32BE(8 + contents.length);
  header.write(name, 4, 'ascii');
  return Buffer.concat([header, contents]);
}
// Authored zero-length fixtures for GHSA-w3rx-r6r6-pgpr and GHSA-5p2g-fcmc-qvqq.
const malformed = {
  ICNS: Buffer.from('69636e73000000106963703400000000', 'hex'),
  JXL: Buffer.concat([box('JXL ', Buffer.from('0d0a870a', 'hex')), box('ftyp', Buffer.from('jxl \0\0\0\0jxl ')), Buffer.from('000000006a786c70', 'hex')]),
  HEIF: Buffer.concat([box('ftyp', Buffer.from('mif1\0\0\0\0mif1')), box('meta', Buffer.concat([Buffer.alloc(4), box('iprp', box('ipco', Buffer.from('0000000069737065000000000000000100000001', 'hex')))]))]),
};
for (const [format, contents] of Object.entries(malformed)) {
  test(`zero-length ${format} data rejects within a bounded subprocess`, t => {
    const file = temporaryFile(t, contents);
    const script = `
      const assert = require('node:assert/strict');
      const {imageSize} = require(process.argv[1]);
      const {imageSizeFromFile} = require(process.argv[2]);
      const expected = new RegExp('Invalid ' + process.argv[3]);
      assert.throws(() => imageSize(Buffer.from(process.argv[4], 'hex')), expected);
      assert.rejects(imageSizeFromFile(process.argv[5]), expected).catch(error => {console.error(error); process.exitCode = 1;});
    `;
    const result = spawnSync(process.execPath, ['-e', script, require.resolve('image-size'), require.resolve('image-size/fromFile'), format, contents.toString('hex'), file], {encoding: 'utf8', timeout: 3000});
    assert.ifError(result.error);
    assert.equal(result.signal, null, result.stderr);
    assert.equal(result.status, 0, result.stderr);
  });
}

test('Docusaurus uses the patched file API and retains image dimensions', async t => {
  const file = temporaryFile(t, png);
  const mdx = createRequire(require.resolve('@docusaurus/mdx-loader/package.json'));
  const source = readFileSync(mdx.resolve('./lib/remark/transformImage'), 'utf8');
  assert.match(source, /const size = await imageSizeFromFile\(imagePath\)/);
  const transform = mdx('./lib/remark/transformImage').default;
  const tree = {type: 'root', children: [{type: 'image', url: './pixel.png', alt: 'Pixel', title: null}]};
  await transform({siteDir: path.dirname(file), staticDirs: []})(tree, {path: path.join(path.dirname(file), 'document.md')});
  assert.equal(tree.children[0].type, 'jsx');
  assert.match(tree.children[0].value, /width="1"/);
  assert.match(tree.children[0].value, /height="1"/);
  assert.match(tree.children[0].value, /pixel\.png/);
});
