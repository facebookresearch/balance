/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const assert = require('node:assert/strict');
const {test} = require('node:test');
const merge = require('deepmerge');

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
  assert.deepEqual(merge.all([{a: 1}, {b: 2}]), {a: 1, b: 2});
});
test('custom array merge and clone options remain supported', () => {
  const source = {values: [2], date: new Date(0)};
  const result = merge({values: [1]}, source, {arrayMerge: (_target, values) => values, clone: false});
  assert.equal(result.values, source.values);
  assert.equal(result.date, source.date);
  assert.deepEqual(result.values, [2]);
});
