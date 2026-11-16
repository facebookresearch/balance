/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const createMerge = require('deepmerge');
const merge = createMerge({symbols: true});

test('the alias resolves to the real released Fastify package', () => {
  const manifest = require('deepmerge/package.json');
  assert.equal(manifest.name, '@fastify/deepmerge');
  assert.equal(manifest.version, '3.2.4');
});

test('the audited dependency graph has no additional legacy merge consumers', () => {
  const lock = fs.readFileSync(path.join(__dirname, '../yarn.lock'), 'utf8');
  const consumers = lock.split('\n\n')
    .filter(block => /^    deepmerge /m.test(block))
    .map(block => block.split('\n')[0]);
  assert.deepEqual(consumers, [
    '"@svgr/plugin-svgo@^6.5.1":',
    'fork-ts-checker-webpack-plugin@^6.5.0:',
  ]);
});

test('the actual webpack checker receives merged options, not the merge factory', () => {
  const Checker = require('fork-ts-checker-webpack-plugin');
  const options = {
    async: false,
    typescript: {configOverwrite: {compilerOptions: {strict: true}}},
    issue: {include: [{file: '**/*.ts'}]},
  };
  const checker = new Checker(options);
  assert.deepEqual(checker.options, options);
  assert.notEqual(checker.options, options);
});

// Independently chosen fixtures for the target shapes described in
// https://github.com/TehShrike/deepmerge/issues/273 (GHSA-gcqx-w48m-w2h4).
for (const [name, target] of [
  ['null-prototype target', Object.create(null)],
  ['own prototype-key target', JSON.parse('{"__proto__":{"oldFlag":true}}')],
  ['nested null placeholder', {nested: null}],
]) {
  test(`prototype keys cannot poison a ${name}`, () => {
    const source = name === 'nested null placeholder'
      ? JSON.parse('{"nested":{"__proto__":{"unexpectedFlag":true}}}')
      : JSON.parse('{"__proto__":{"unexpectedFlag":true}}');
    const result = merge(target, source);
    const object = name === 'nested null placeholder' ? result.nested : result;
    assert.equal(Object.getPrototypeOf(object), Object.prototype);
    assert.equal(object.unexpectedFlag, undefined);
    assert.equal(object.oldFlag, undefined);
    assert.equal(Object.hasOwn(object, '__proto__'), false);
    assert.equal({}.unexpectedFlag, undefined);
  });
}
test('forbidden keys are rejected in both input loops without dropping ordinary values', () => {
  const target = JSON.parse('{"constructor":{"targetFlag":true},"prototype":{"targetFlag":true},"kept":1}');
  const source = JSON.parse('{"constructor":{"sourceFlag":true},"prototype":{"sourceFlag":true},"other":2}');
  const result = merge(target, source);
  assert.deepEqual(result, {kept: 1, other: 2});
  assert.equal(result.constructor, Object);
  assert.equal(Object.hasOwn(result, 'prototype'), false);
});
test('nested objects, arrays, symbols and all retain their ordinary merge behavior', () => {
  const key = Symbol('test');
  const target = {nested: {a: 1}, values: [1], [key]: 'kept'};
  const source = {nested: {b: 2}, values: [2]};
  const result = merge(target, source);
  assert.deepEqual(result, {nested: {a: 1, b: 2}, values: [1, 2], [key]: 'kept'});
  assert.deepEqual(target, {nested: {a: 1}, values: [1], [key]: 'kept'});
  const mergeAll = createMerge({all: true, symbols: true});
  assert.deepEqual(mergeAll({a: 1}, {b: 2}), {a: 1, b: 2});
});
test('custom array merge and special-object reference options remain supported', () => {
  const source = {values: [2], date: new Date(0)};
  const configuredMerge = createMerge({
    mergeArray: () => (_target, values) => values,
    cloneProtoObject: value => value,
  });
  const result = configuredMerge({values: [1]}, source);
  assert.equal(result.values, source.values);
  assert.equal(result.date, source.date);
  assert.deepEqual(result.values, [2]);
});
