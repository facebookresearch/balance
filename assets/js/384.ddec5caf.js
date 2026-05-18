"use strict";
exports.id = 384;
exports.ids = [384];
exports.modules = {

/***/ 23126
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  T: () => (/* binding */ Graph)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/constant.js
var constant = __webpack_require__(39142);
// EXTERNAL MODULE: ./node_modules/lodash-es/isFunction.js
var isFunction = __webpack_require__(89610);
// EXTERNAL MODULE: ./node_modules/lodash-es/keys.js
var keys = __webpack_require__(27422);
// EXTERNAL MODULE: ./node_modules/lodash-es/filter.js + 1 modules
var filter = __webpack_require__(11662);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseKeys.js + 1 modules
var _baseKeys = __webpack_require__(69471);
// EXTERNAL MODULE: ./node_modules/lodash-es/_getTag.js + 3 modules
var _getTag = __webpack_require__(9779);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArguments.js + 1 modules
var isArguments = __webpack_require__(29893);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArrayLike.js
var isArrayLike = __webpack_require__(38446);
// EXTERNAL MODULE: ./node_modules/lodash-es/isBuffer.js + 1 modules
var isBuffer = __webpack_require__(99912);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isPrototype.js
var _isPrototype = __webpack_require__(97271);
// EXTERNAL MODULE: ./node_modules/lodash-es/isTypedArray.js + 1 modules
var isTypedArray = __webpack_require__(33858);
;// ./node_modules/lodash-es/isEmpty.js









/** `Object#toString` result references. */
var mapTag = '[object Map]',
    setTag = '[object Set]';

/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var isEmpty_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Checks if `value` is an empty object, collection, map, or set.
 *
 * Objects are considered empty if they have no own enumerable string keyed
 * properties.
 *
 * Array-like values such as `arguments` objects, arrays, buffers, strings, or
 * jQuery-like collections are considered empty if they have a `length` of `0`.
 * Similarly, maps and sets are considered empty if they have a `size` of `0`.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is empty, else `false`.
 * @example
 *
 * _.isEmpty(null);
 * // => true
 *
 * _.isEmpty(true);
 * // => true
 *
 * _.isEmpty(1);
 * // => true
 *
 * _.isEmpty([1, 2, 3]);
 * // => false
 *
 * _.isEmpty({ 'a': 1 });
 * // => false
 */
function isEmpty(value) {
  if (value == null) {
    return true;
  }
  if ((0,isArrayLike/* default */.A)(value) &&
      ((0,isArray/* default */.A)(value) || typeof value == 'string' || typeof value.splice == 'function' ||
        (0,isBuffer/* default */.A)(value) || (0,isTypedArray/* default */.A)(value) || (0,isArguments/* default */.A)(value))) {
    return !value.length;
  }
  var tag = (0,_getTag/* default */.A)(value);
  if (tag == mapTag || tag == setTag) {
    return !value.size;
  }
  if ((0,_isPrototype/* default */.A)(value)) {
    return !(0,_baseKeys/* default */.A)(value).length;
  }
  for (var key in value) {
    if (isEmpty_hasOwnProperty.call(value, key)) {
      return false;
    }
  }
  return true;
}

/* harmony default export */ const lodash_es_isEmpty = (isEmpty);

// EXTERNAL MODULE: ./node_modules/lodash-es/forEach.js
var forEach = __webpack_require__(8058);
// EXTERNAL MODULE: ./node_modules/lodash-es/isUndefined.js
var isUndefined = __webpack_require__(69592);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseFlatten.js + 1 modules
var _baseFlatten = __webpack_require__(13588);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseRest.js
var _baseRest = __webpack_require__(24326);
// EXTERNAL MODULE: ./node_modules/lodash-es/_SetCache.js + 2 modules
var _SetCache = __webpack_require__(62062);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseFindIndex.js
var _baseFindIndex = __webpack_require__(25707);
;// ./node_modules/lodash-es/_baseIsNaN.js
/**
 * The base implementation of `_.isNaN` without support for number objects.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is `NaN`, else `false`.
 */
function baseIsNaN(value) {
  return value !== value;
}

/* harmony default export */ const _baseIsNaN = (baseIsNaN);

;// ./node_modules/lodash-es/_strictIndexOf.js
/**
 * A specialized version of `_.indexOf` which performs strict equality
 * comparisons of values, i.e. `===`.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} value The value to search for.
 * @param {number} fromIndex The index to search from.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function strictIndexOf(array, value, fromIndex) {
  var index = fromIndex - 1,
      length = array.length;

  while (++index < length) {
    if (array[index] === value) {
      return index;
    }
  }
  return -1;
}

/* harmony default export */ const _strictIndexOf = (strictIndexOf);

;// ./node_modules/lodash-es/_baseIndexOf.js




/**
 * The base implementation of `_.indexOf` without `fromIndex` bounds checks.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} value The value to search for.
 * @param {number} fromIndex The index to search from.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function baseIndexOf(array, value, fromIndex) {
  return value === value
    ? _strictIndexOf(array, value, fromIndex)
    : (0,_baseFindIndex/* default */.A)(array, _baseIsNaN, fromIndex);
}

/* harmony default export */ const _baseIndexOf = (baseIndexOf);

;// ./node_modules/lodash-es/_arrayIncludes.js


/**
 * A specialized version of `_.includes` for arrays without support for
 * specifying an index to search from.
 *
 * @private
 * @param {Array} [array] The array to inspect.
 * @param {*} target The value to search for.
 * @returns {boolean} Returns `true` if `target` is found, else `false`.
 */
function arrayIncludes(array, value) {
  var length = array == null ? 0 : array.length;
  return !!length && _baseIndexOf(array, value, 0) > -1;
}

/* harmony default export */ const _arrayIncludes = (arrayIncludes);

;// ./node_modules/lodash-es/_arrayIncludesWith.js
/**
 * This function is like `arrayIncludes` except that it accepts a comparator.
 *
 * @private
 * @param {Array} [array] The array to inspect.
 * @param {*} target The value to search for.
 * @param {Function} comparator The comparator invoked per element.
 * @returns {boolean} Returns `true` if `target` is found, else `false`.
 */
function arrayIncludesWith(array, value, comparator) {
  var index = -1,
      length = array == null ? 0 : array.length;

  while (++index < length) {
    if (comparator(value, array[index])) {
      return true;
    }
  }
  return false;
}

/* harmony default export */ const _arrayIncludesWith = (arrayIncludesWith);

// EXTERNAL MODULE: ./node_modules/lodash-es/_cacheHas.js
var _cacheHas = __webpack_require__(64099);
// EXTERNAL MODULE: ./node_modules/lodash-es/_Set.js
var _Set = __webpack_require__(39857);
;// ./node_modules/lodash-es/noop.js
/**
 * This method returns `undefined`.
 *
 * @static
 * @memberOf _
 * @since 2.3.0
 * @category Util
 * @example
 *
 * _.times(2, _.noop);
 * // => [undefined, undefined]
 */
function noop() {
  // No operation performed.
}

/* harmony default export */ const lodash_es_noop = (noop);

// EXTERNAL MODULE: ./node_modules/lodash-es/_setToArray.js
var _setToArray = __webpack_require__(29959);
;// ./node_modules/lodash-es/_createSet.js




/** Used as references for various `Number` constants. */
var INFINITY = 1 / 0;

/**
 * Creates a set object of `values`.
 *
 * @private
 * @param {Array} values The values to add to the set.
 * @returns {Object} Returns the new set.
 */
var createSet = !(_Set/* default */.A && (1 / (0,_setToArray/* default */.A)(new _Set/* default */.A([,-0]))[1]) == INFINITY) ? lodash_es_noop : function(values) {
  return new _Set/* default */.A(values);
};

/* harmony default export */ const _createSet = (createSet);

;// ./node_modules/lodash-es/_baseUniq.js







/** Used as the size to enable large array optimizations. */
var LARGE_ARRAY_SIZE = 200;

/**
 * The base implementation of `_.uniqBy` without support for iteratee shorthands.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {Function} [iteratee] The iteratee invoked per element.
 * @param {Function} [comparator] The comparator invoked per element.
 * @returns {Array} Returns the new duplicate free array.
 */
function baseUniq(array, iteratee, comparator) {
  var index = -1,
      includes = _arrayIncludes,
      length = array.length,
      isCommon = true,
      result = [],
      seen = result;

  if (comparator) {
    isCommon = false;
    includes = _arrayIncludesWith;
  }
  else if (length >= LARGE_ARRAY_SIZE) {
    var set = iteratee ? null : _createSet(array);
    if (set) {
      return (0,_setToArray/* default */.A)(set);
    }
    isCommon = false;
    includes = _cacheHas/* default */.A;
    seen = new _SetCache/* default */.A;
  }
  else {
    seen = iteratee ? [] : result;
  }
  outer:
  while (++index < length) {
    var value = array[index],
        computed = iteratee ? iteratee(value) : value;

    value = (comparator || value !== 0) ? value : 0;
    if (isCommon && computed === computed) {
      var seenIndex = seen.length;
      while (seenIndex--) {
        if (seen[seenIndex] === computed) {
          continue outer;
        }
      }
      if (iteratee) {
        seen.push(computed);
      }
      result.push(value);
    }
    else if (!includes(seen, computed, comparator)) {
      if (seen !== result) {
        seen.push(computed);
      }
      result.push(value);
    }
  }
  return result;
}

/* harmony default export */ const _baseUniq = (baseUniq);

// EXTERNAL MODULE: ./node_modules/lodash-es/isArrayLikeObject.js
var isArrayLikeObject = __webpack_require__(53533);
;// ./node_modules/lodash-es/union.js





/**
 * Creates an array of unique values, in order, from all given arrays using
 * [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * for equality comparisons.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Array
 * @param {...Array} [arrays] The arrays to inspect.
 * @returns {Array} Returns the new array of combined values.
 * @example
 *
 * _.union([2], [1, 2]);
 * // => [2, 1]
 */
var union = (0,_baseRest/* default */.A)(function(arrays) {
  return _baseUniq((0,_baseFlatten/* default */.A)(arrays, 1, isArrayLikeObject/* default */.A, true));
});

/* harmony default export */ const lodash_es_union = (union);

// EXTERNAL MODULE: ./node_modules/lodash-es/values.js + 1 modules
var values = __webpack_require__(38207);
// EXTERNAL MODULE: ./node_modules/lodash-es/reduce.js + 2 modules
var reduce = __webpack_require__(89463);
;// ./node_modules/dagre-d3-es/src/graphlib/graph.js


var DEFAULT_EDGE_NAME = '\x00';
var GRAPH_NODE = '\x00';
var EDGE_KEY_DELIM = '\x01';

/**
 * @typedef {string} NodeID ID of a node.
 */

/**
 * @typedef {`${string}${typeof EDGE_KEY_DELIM}${string}${typeof EDGE_KEY_DELIM}${string}`} EdgeID ID of an edge.
 * @internal - All public APIs use {@link EdgeObj} instead to refer to edges.
 */

/**
 * @typedef {object} EdgeObj
 * @property {NodeID} v the id of the source or tail node of an edge
 * @property {NodeID} w the id of the target or head node of an edge
 * @property {string | number} [name] Name of the edge. Needed to uniquely identify
 * multiple edges between the same pair of nodes in a multigraph.
 */

/**
 * @template {unknown} T
 * @typedef {T[] | Record<any, T>} Collection
 * Lodash object that can be iterated over with `_.each`.
 *
 * Beware, objects with `.length` are treated as arrays, see
 * https://lodash.com/docs/4.17.15#forEach
 */

// Implementation notes:
//
//  * Node id query functions should return string ids for the nodes
//  * Edge id query functions should return an "edgeObj", edge object, that is
//    composed of enough information to uniquely identify an edge: {v, w, name}.
//  * Internally we use an "edgeId", a stringified form of the edgeObj, to
//    reference edges. This is because we need a performant way to look these
//    edges up and, object properties, which have string keys, are the closest
//    we're going to get to a performant hashtable in JavaScript.

// Implementation notes:
//
//  * Node id query functions should return string ids for the nodes
//  * Edge id query functions should return an "edgeObj", edge object, that is
//    composed of enough information to uniquely identify an edge: {v, w, name}.
//  * Internally we use an "edgeId", a stringified form of the edgeObj, to
//    reference edges. This is because we need a performant way to look these
//    edges up and, object properties, which have string keys, are the closest
//    we're going to get to a performant hashtable in JavaScript.

/**
 * @typedef {object} GraphOptions
 * @property {boolean | undefined} [directed] - set to `true` to get a
 * directed graph and `false` to get an undirected graph.
 * An undirected graph does not treat the order of nodes in an edge as
 * significant.
 * In other words, `g.edge("a", "b") === g.edge("b", "a")` for
 * an undirected graph.
 * Default: `true`
 * @property {boolean | undefined} [multigraph] - set to `true` to allow a
 * graph to have multiple edges between the same pair of nodes.
 * Default: `false`.
 * @property {boolean | undefined} [compound] - set to `true` to allow a
 * graph to have compound nodes - nodes which can be the parent of other
 * nodes.
 * Default: `false`.
 */

/**
 * Graphlib has a single graph type: {@link Graph}. To create a new instance:
 *
 * ```js
 * var g = new Graph();
 * ```
 *
 * By default this will create a directed graph that does not allow multi-edges
 * or compound nodes.
 * The following options can be used when constructing a new graph:
 *
 * * {@link GraphOptions#directed}: set to `true` to get a directed graph and `false` to get an
 *   undirected graph.
 *   An undirected graph does not treat the order of nodes in an edge as
 *   significant. In other words,
 *   `g.edge("a", "b") === g.edge("b", "a")` for an undirected graph.
 *   Default: `true`.
 * * {@link GraphOptions#multigraph}: set to `true` to allow a graph to have multiple edges
 *   between the same pair of nodes. Default: `false`.
 * * {@link GraphOptions#compound}: set to `true` to allow a graph to have compound nodes -
 *   nodes which can be the parent of other nodes. Default: `false`.
 *
 * To set the options, pass in an options object to the `Graph` constructor.
 * For example, to create a directed compound multigraph:
 *
 * ```js
 * var g = new Graph({ directed: true, compound: true, multigraph: true });
 * ```
 *
 * ### Node and Edge Representation
 *
 * In graphlib, a node is represented by a user-supplied String id.
 * All node related functions use this String id as a way to uniquely identify
 * the node. Here is an example of interacting with nodes:
 *
 * ```js
 * var g = new Graph();
 * g.setNode("my-id", "my-label");
 * g.node("my-id"); // returns "my-label"
 * ```
 *
 * Edges in graphlib are identified by the nodes they connect. For example:
 *
 * ```js
 * var g = new Graph();
 * g.setEdge("source", "target", "my-label");
 * g.edge("source", "target"); // returns "my-label"
 * ```
 *
 * However, we need a way to uniquely identify an edge in a single object for
 * various edge queries (e.g. {@link Graph#outEdges}).
 * We use {@link EdgeObj}s for this purpose.
 * They consist of the following properties:
 *
 * * {@link EdgeObj#v}: the id of the source or tail node of an edge
 * * {@link EdgeObj#w}: the id of the target or head node of an edge
 * * {@link EdgeObj#name} (optional): the name that uniquely identifies a multiedge.
 *
 * Any edge function that takes an edge id will also work with an {@link EdgeObj}. For example:
 *
 * ```js
 * var g = new Graph();
 * g.setEdge("source", "target", "my-label");
 * g.edge({ v: "source", w: "target" }); // returns "my-label"
 * ```
 *
 * ### Multigraphs
 *
 * A [multigraph](https://en.wikipedia.org/wiki/Multigraph) is a graph that can
 * have more than one edge between the same pair of nodes.
 * By default graphlib graphs are not multigraphs, but a multigraph can be
 * constructed by setting the {@link GraphOptions#multigraph} property to true:
 *
 * ```js
 * var g = new Graph({ multigraph: true });
 * ```
 *
 * With multiple edges between two nodes we need some way to uniquely identify
 * each edge. We call this the {@link EdgeObj#name} property.
 * Here's an example of creating a couple of edges between the same nodes:
 *
 * ```js
 * var g = new Graph({ multigraph: true });
 * g.setEdge("a", "b", "edge1-label", "edge1");
 * g.setEdge("a", "b", "edge2-label", "edge2");
 * g.edge("a", "b", "edge1"); // returns "edge1-label"
 * g.edge("a", "b", "edge2"); // returns "edge2-label"
 * g.edges(); // returns [{ v: "a", w: "b", name: "edge1" },
 *            //          { v: "a", w: "b", name: "edge2" }]
 * ```
 *
 * A multigraph still allows an edge with no name to be created:
 *
 * ```js
 * var g = new Graph({ multigraph: true });
 * g.setEdge("a", "b", "my-label");
 * g.edge({ v: "a", w: "b" }); // returns "my-label"
 * ```
 *
 * ### Compound Graphs
 *
 * A compound graph is one where a node can be the parent of other nodes.
 * The child nodes form a "subgraph".
 * Here's an example of constructing and interacting with a compound graph:
 *
 * ```js
 * var g = new Graph({ compound: true });
 * g.setParent("a", "parent");
 * g.setParent("b", "parent");
 * g.parent("a");      // returns "parent"
 * g.parent("b");      // returns "parent"
 * g.parent("parent"); // returns undefined
 * ```
 *
 * ### Default Labels
 *
 * When a node or edge is created without a label, a default label can be assigned.
 * See {@link setDefaultNodeLabel} and {@link setDefaultEdgeLabel}.
 *
 * @template [GraphLabel=any] - Label of the graph.
 * @template [NodeLabel=any] - Label of a node.
 * Even though this is a "label", this could be any type that the user requires
 * (and may need to be an object for some layout/ranking algorithms in dagre).
 * @template [EdgeLabel=any] - Label of an edge.
 * Even though this is a "label", this could be any type that the user requires,
 * (and may need to be a object for ranking in dagre).
 */
class Graph {
  /**
   * @param {GraphOptions} [opts] - Graph options.
   */
  constructor(opts = {}) {
    /**
     * @type {boolean}
     * @private
     */
    this._isDirected = Object.prototype.hasOwnProperty.call(opts, 'directed')
      ? opts.directed
      : true;
    /**
     * @type {boolean}
     * @private
     */
    this._isMultigraph = Object.prototype.hasOwnProperty.call(opts, 'multigraph')
      ? opts.multigraph
      : false;
    /**
     * @type {boolean}
     * @private
     */
    this._isCompound = Object.prototype.hasOwnProperty.call(opts, 'compound')
      ? opts.compound
      : false;

    /**
     * @type {GraphLabel | undefined}
     * Label for the graph itself
     */
    this._label = undefined;

    /**
     * Default label to be set when creating a new node.
     *
     * @private
     * @type {(v: NodeID | number) => NodeLabel}
     */
    this._defaultNodeLabelFn = constant/* default */.A(undefined);

    /**
     * Default label to be set when creating a new edge
     *
     * @private
     * @type {(v: NodeID, w: NodeID, name: string | undefined) => EdgeLabel}
     */
    this._defaultEdgeLabelFn = constant/* default */.A(undefined);

    /**
     * @type {Record<NodeID, NodeLabel>}
     * @private
     *
     * v -> label
     */
    this._nodes = {};

    if (this._isCompound) {
      /**
       * @type {Record<NodeID, NodeID>}
       * @private
       * v -> parent
       */
      this._parent = {};

      /**
       * @type {Record<NodeID, Record<NodeID, true>>}
       * @private
       * v -> children
       */
      this._children = {};
      this._children[GRAPH_NODE] = {};
    }

    /**
     * @type {Record<NodeID, Record<EdgeID, EdgeObj>>}
     * @private
     * v -> edgeObj
     */
    this._in = {};

    /**
     * @type {Record<NodeID, Record<NodeID, number>>}
     * @private
     * u -> v -> Number
     */
    this._preds = {};

    /**
     * @type {Record<NodeID, Record<EdgeID, EdgeObj>>}
     * @private
     * v -> edgeObj
     */
    this._out = {};

    /**
     * @type {Record<NodeID, Record<NodeID, number>>}
     * @private
     * v -> w -> Number
     */
    this._sucs = {};

    /**
     * @type {Record<EdgeID, EdgeObj>}
     * @private
     * e -> edgeObj
     */
    this._edgeObjs = {};

    /**
     * @type {Record<EdgeID, EdgeLabel>}
     * @private
     * e -> label
     */
    this._edgeLabels = {};
  }

  /* === Graph functions ========= */

  /**
   *
   * @returns {boolean} `true` if the graph is [directed](https://en.wikipedia.org/wiki/Directed_graph).
   * A directed graph treats the order of nodes in an edge as significant whereas an
   * [undirected](https://en.wikipedia.org/wiki/Graph_(mathematics)#Undirected_graph)
   * graph does not.
   * This example demonstrates the difference:
   *
   * @example
   *
   * ```js
   * var directed = new Graph({ directed: true });
   * directed.setEdge("a", "b", "my-label");
   * directed.edge("a", "b"); // returns "my-label"
   * directed.edge("b", "a"); // returns undefined
   *
   * var undirected = new Graph({ directed: false });
   * undirected.setEdge("a", "b", "my-label");
   * undirected.edge("a", "b"); // returns "my-label"
   * undirected.edge("b", "a"); // returns "my-label"
   * ```
   */
  isDirected() {
    return this._isDirected;
  }
  /**
   * @returns {boolean} `true` if the graph is a multigraph.
   */
  isMultigraph() {
    return this._isMultigraph;
  }
  /**
   * @returns {boolean} `true` if the graph is compound.
   */
  isCompound() {
    return this._isCompound;
  }

  /**
   * Sets the label for the graph to `label`.
   *
   * @param {GraphLabel} label - Label for the graph.
   * @returns {this}
   */
  setGraph(label) {
    this._label = label;
    return this;
  }

  /**
   * @returns {GraphLabel | undefined} the currently assigned label for the graph.
   * If no label has been assigned, returns `undefined`.
   *
   * @example
   *
   * ```js
   * var g = new Graph();
   * g.graph(); // returns undefined
   * g.setGraph("graph-label");
   *  g.graph(); // returns "graph-label"
   * ```
   */
  graph() {
    return this._label;
  }
  /* === Node functions ========== */

  /**
   * Sets a new default value that is assigned to nodes that are created without
   * a label.
   *
   * @param {typeof this._defaultNodeLabelFn | NodeLabel} newDefault - If a function,
   * it is called with the id of the node being created.
   * Otherwise, it is assigned as the label directly.
   * @returns {this}
   */
  setDefaultNodeLabel(newDefault) {
    if (!isFunction/* default */.A(newDefault)) {
      newDefault = constant/* default */.A(newDefault);
    }
    this._defaultNodeLabelFn = newDefault;
    return this;
  }

  /**
   * @returns {number} the number of nodes in the graph.
   */
  nodeCount() {
    return this._nodeCount;
  }

  /**
   * @returns {NodeID[]} the ids of the nodes in the graph.
   *
   * @remarks
   * Use {@link node()} to get the label for each node.
   * Takes `O(|V|)` time.
   */
  nodes() {
    return keys/* default */.A(this._nodes);
  }
  /**
   * @returns {NodeID[]} those nodes in the graph that have no in-edges.
   * @remarks Takes `O(|V|)` time.
   */
  sources() {
    var self = this;
    return filter/* default */.A(this.nodes(), function (v) {
      return lodash_es_isEmpty(self._in[v]);
    });
  }
  /**
   * @returns {NodeID[]} those nodes in the graph that have no out-edges.
   * @remarks Takes `O(|V|)` time.
   */
  sinks() {
    var self = this;
    return filter/* default */.A(this.nodes(), function (v) {
      return lodash_es_isEmpty(self._out[v]);
    });
  }

  /**
   * Invokes setNode method for each node in `vs` list.
   *
   * @param {Collection<NodeID | number>} vs - List of node IDs to create/set.
   * @param {NodeLabel} [value] - If set, update all nodes with this value.
   * @returns {this}
   * @remarks Complexity: O(|names|).
   */
  setNodes(vs, value) {
    var args = arguments;
    var self = this;
    forEach/* default */.A(vs, function (v) {
      if (args.length > 1) {
        self.setNode(v, value);
      } else {
        self.setNode(v);
      }
    });
    return this;
  }

  /**
   * Creates or updates the value for the node `v` in the graph.
   *
   * @param {NodeID | number} v - ID of the node to create/set.
   * @param {NodeLabel} [value] - If supplied, it is set as the value for the node.
   * If not supplied and the node was created by this call then
   * {@link setDefaultNodeLabel} will be used to set the node's value.
   * @returns {this} the graph, allowing this to be chained with other functions.
   * @remarks Takes `O(1)` time.
   */
  setNode(v, value) {
    if (Object.prototype.hasOwnProperty.call(this._nodes, v)) {
      if (arguments.length > 1) {
        this._nodes[v] = value;
      }
      return this;
    }

    this._nodes[v] = arguments.length > 1 ? value : this._defaultNodeLabelFn(v);
    if (this._isCompound) {
      this._parent[v] = GRAPH_NODE;
      this._children[v] = {};
      this._children[GRAPH_NODE][v] = true;
    }
    this._in[v] = {};
    this._preds[v] = {};
    this._out[v] = {};
    this._sucs[v] = {};
    ++this._nodeCount;
    return this;
  }

  /**
   * Gets the label of node with specified name.
   *
   * @param {NodeID | number} v - Node ID.
   * @returns {NodeLabel | undefined} the label assigned to the node with the id `v`
   * if it is in the graph.
   * Otherwise returns `undefined`.
   * @remarks Takes `O(1)` time.
   */
  node(v) {
    return this._nodes[v];
  }

  /**
   * Detects whether graph has a node with specified name or not.
   *
   * @param {NodeID | number} v - Node ID.
   * @returns {boolean} Returns `true` the graph has a node with the id.
   * @remarks Takes `O(1)` time.
   */
  hasNode(v) {
    return Object.prototype.hasOwnProperty.call(this._nodes, v);
  }

  /**
   * Remove the node with the id `v` in the graph or do nothing if the node is
   * not in the graph.
   *
   * If the node was removed this function also removes any incident edges.
   *
   * @param {NodeID | number} v - Node ID to remove.
   * @returns {this} the graph, allowing this to be chained with other functions.
   * @remarks Takes `O(|E|)` time.
   */
  removeNode(v) {
    if (Object.prototype.hasOwnProperty.call(this._nodes, v)) {
      var removeEdge = (e) => this.removeEdge(this._edgeObjs[e]);
      delete this._nodes[v];
      if (this._isCompound) {
        this._removeFromParentsChildList(v);
        delete this._parent[v];
        forEach/* default */.A(this.children(v), (child) => {
          this.setParent(child);
        });
        delete this._children[v];
      }
      forEach/* default */.A(keys/* default */.A(this._in[v]), removeEdge);
      delete this._in[v];
      delete this._preds[v];
      forEach/* default */.A(keys/* default */.A(this._out[v]), removeEdge);
      delete this._out[v];
      delete this._sucs[v];
      --this._nodeCount;
    }
    return this;
  }

  /**
   * Sets the parent for `v` to `parent` if it is defined or removes the parent
   * for `v` if `parent` is undefined.
   *
   * @param {NodeID | number} v - Node ID to set the parent for.
   * @param {NodeID | number} [parent] - Parent node ID. If not defined, removes the parent.
   * @returns {this} the graph, allowing this to be chained with other functions.
   * @throws if the graph is not compound.
   * @throws if setting the parent would create a cycle.
   * @remarks Takes `O(1)` time.
   */
  setParent(v, parent) {
    if (!this._isCompound) {
      throw new Error('Cannot set parent in a non-compound graph');
    }

    if (isUndefined/* default */.A(parent)) {
      parent = GRAPH_NODE;
    } else {
      // Coerce parent to string
      parent += '';
      for (var ancestor = parent; !isUndefined/* default */.A(ancestor); ancestor = this.parent(ancestor)) {
        if (ancestor === v) {
          throw new Error('Setting ' + parent + ' as parent of ' + v + ' would create a cycle');
        }
      }

      this.setNode(parent);
    }

    this.setNode(v);
    this._removeFromParentsChildList(v);
    // @ts-expect-error -- We coerced parent to a string above
    this._parent[v] = parent;
    this._children[parent][v] = true;
    return this;
  }

  /**
   * @private
   * @param {NodeID | number} v - Node ID.
   */
  _removeFromParentsChildList(v) {
    delete this._children[this._parent[v]][v];
  }

  /**
   * Get parent node for node `v`.
   *
   * @param {NodeID | number} v - Node ID.
   * @returns {NodeID | undefined} the node that is a parent of node `v`
   * or `undefined` if node `v` does not have a parent or is not a member of
   * the graph.
   * Always returns `undefined` for graphs that are not compound.
   * @remarks Takes `O(1)` time.
   */
  parent(v) {
    if (this._isCompound) {
      var parent = this._parent[v];
      if (parent !== GRAPH_NODE) {
        return parent;
      }
    }
  }

  /**
   * Gets list of direct children of node v.
   *
   * @param {NodeID | number} [v] - Node ID. If not specified, gets nodes
   * with no parent (top-level nodes).
   * @returns {NodeID[] | undefined} all nodes that are children of node `v` or
   * `undefined` if node `v` is not in the graph.
   * Always returns `[]` for graphs that are not compound.
   * @remarks Takes `O(|V|)` time.
   */
  children(v) {
    if (isUndefined/* default */.A(v)) {
      v = GRAPH_NODE;
    }

    if (this._isCompound) {
      var children = this._children[v];
      if (children) {
        return keys/* default */.A(children);
      }
    } else if (v === GRAPH_NODE) {
      return this.nodes();
    } else if (this.hasNode(v)) {
      return [];
    }
  }

  /**
   * @param {NodeID | number} v - Node ID.
   * @returns {NodeID[] | undefined} all nodes that are predecessors of the
   * specified node or `undefined` if node `v` is not in the graph.
   * @remarks
   * Behavior is undefined for undirected graphs - use {@link neighbors} instead.
   * Takes `O(|V|)` time.
   */
  predecessors(v) {
    var predsV = this._preds[v];
    if (predsV) {
      return keys/* default */.A(predsV);
    }
  }

  /**
   * @param {NodeID | number} v - Node ID.
   * @returns {NodeID[] | undefined} all nodes that are successors of the
   * specified node or `undefined` if node `v` is not in the graph.
   * @remarks
   * Behavior is undefined for undirected graphs - use {@link neighbors} instead.
   * Takes `O(|V|)` time.
   */
  successors(v) {
    var sucsV = this._sucs[v];
    if (sucsV) {
      return keys/* default */.A(sucsV);
    }
  }

  /**
   * @param {NodeID | number} v - Node ID.
   * @returns {NodeID[] | undefined} all nodes that are predecessors or
   * successors of the specified node
   * or `undefined` if node `v` is not in the graph.
   * @remarks Takes `O(|V|)` time.
   */
  neighbors(v) {
    var preds = this.predecessors(v);
    if (preds) {
      return lodash_es_union(preds, this.successors(v));
    }
  }

  /**
   * @param {NodeID | number} v - Node ID.
   * @returns {boolean} True if the node is a leaf (has no successors), false otherwise.
   */
  isLeaf(v) {
    var neighbors;
    if (this.isDirected()) {
      neighbors = this.successors(v);
    } else {
      neighbors = this.neighbors(v);
    }
    return neighbors.length === 0;
  }

  /**
   * Creates new graph with nodes filtered via `filter`.
   * Edges incident to rejected node
   * are also removed.
   * 
   * In case of compound graph, if parent is rejected by `filter`,
   * than all its children are rejected too.

   * @param {(v: NodeID) => boolean} filter - Function that returns `true` for nodes to keep.
   * @returns {Graph<GraphLabel, NodeLabel, EdgeLabel>} A new graph containing only the nodes for which `filter` returns `true`.
   * @remarks Average-case complexity: O(|E|+|V|).
   */
  filterNodes(filter) {
    /**
     * @type {Graph<GraphLabel, NodeLabel, EdgeLabel>}
     */
    // @ts-expect-error
    var copy = new this.constructor({
      directed: this._isDirected,
      multigraph: this._isMultigraph,
      compound: this._isCompound,
    });

    copy.setGraph(this.graph());

    var self = this;
    forEach/* default */.A(this._nodes, function (value, v) {
      if (filter(v)) {
        copy.setNode(v, value);
      }
    });

    forEach/* default */.A(this._edgeObjs, function (e) {
      if (copy.hasNode(e.v) && copy.hasNode(e.w)) {
        copy.setEdge(e, self.edge(e));
      }
    });

    var parents = {};
    function findParent(v) {
      var parent = self.parent(v);
      if (parent === undefined || copy.hasNode(parent)) {
        parents[v] = parent;
        return parent;
      } else if (parent in parents) {
        return parents[parent];
      } else {
        return findParent(parent);
      }
    }

    if (this._isCompound) {
      forEach/* default */.A(copy.nodes(), function (v) {
        copy.setParent(v, findParent(v));
      });
    }

    return copy;
  }

  /* === Edge functions ========== */

  /**
   * Sets a new default value that is assigned to edges that are created without
   * a label.
   *
   * @param {typeof this._defaultEdgeLabelFn | EdgeLabel} newDefault - If a function,
   * it is called with the parameters `(v, w, name)`.
   * Otherwise, it is assigned as the label directly.
   * @returns {this}
   */
  setDefaultEdgeLabel(newDefault) {
    if (!isFunction/* default */.A(newDefault)) {
      newDefault = constant/* default */.A(newDefault);
    }
    this._defaultEdgeLabelFn = newDefault;
    return this;
  }

  /**
   * @returns {number} the number of edges in the graph.
   * @remarks Complexity: O(1).
   */
  edgeCount() {
    return this._edgeCount;
  }

  /**
   * Gets edges of the graph.
   *
   * @returns {EdgeObj[]} the {@link EdgeObj} for each edge in the graph.
   *
   * @remarks
   * In case of compound graph subgraphs are not considered.
   * Use {@link edge()} to get the label for each edge.
   * Takes `O(|E|)` time.
   */
  edges() {
    return values/* default */.A(this._edgeObjs);
  }

  /**
   * Establish an edges path over the nodes in nodes list.
   *
   * If some edge is already exists, it will update its label, otherwise it will
   * create an edge between pair of nodes with label provided or default label
   * if no label provided.
   *
   * @param {Collection<NodeID>} vs - List of node IDs to create edges between.
   * @param {EdgeLabel} [value] - If set, update all edges with this value.
   * @returns {this}
   * @remarks Complexity: O(|nodes|).
   */
  setPath(vs, value) {
    var self = this;
    var args = arguments;
    reduce/* default */.A(vs, function (v, w) {
      if (args.length > 1) {
        self.setEdge(v, w, value);
      } else {
        self.setEdge(v, w);
      }
      return w;
    });
    return this;
  }

  /**
   * Creates or updates the label for the edge (`v`, `w`) with the optionally
   * supplied `name`.
   *
   * @overload
   * @param {EdgeObj} arg0 - Edge object.
   * @param {EdgeLabel} [value] - If supplied, it is set as the label for the edge.
   * If not supplied and the edge was created by this call then
   * {@link setDefaultEdgeLabel} will be used to assign the edge's label.
   * @returns {this} the graph, allowing this to be chained with other functions.
   * @remarks Takes `O(1)` time.
   */
  /**
   * Creates or updates the label for the edge (`v`, `w`) with the optionally
   * supplied `name`.
   *
   * @overload
   * @param {NodeID | number} v - Source node ID. Number values will be coerced to strings.
   * @param {NodeID | number} w - Target node ID. Number values will be coerced to strings.
   * @param {EdgeLabel} [value] - If supplied, it is set as the label for the edge.
   * If not supplied and the edge was created by this call then
   * {@link setDefaultEdgeLabel} will be used to assign the edge's label.
   * @param {string | number} [name] - Edge name. Only useful with multigraphs.
   * @returns {this} the graph, allowing this to be chained with other functions.
   * @remarks Takes `O(1)` time.
   */
  setEdge() {
    var v, w, name, value;
    var valueSpecified = false;
    var arg0 = arguments[0];

    if (typeof arg0 === 'object' && arg0 !== null && 'v' in arg0) {
      v = arg0.v;
      w = arg0.w;
      name = arg0.name;
      if (arguments.length === 2) {
        value = arguments[1];
        valueSpecified = true;
      }
    } else {
      v = arg0;
      w = arguments[1];
      name = arguments[3];
      if (arguments.length > 2) {
        value = arguments[2];
        valueSpecified = true;
      }
    }

    v = '' + v;
    w = '' + w;
    if (!isUndefined/* default */.A(name)) {
      name = '' + name;
    }

    var e = edgeArgsToId(this._isDirected, v, w, name);
    if (Object.prototype.hasOwnProperty.call(this._edgeLabels, e)) {
      if (valueSpecified) {
        this._edgeLabels[e] = value;
      }
      return this;
    }

    if (!isUndefined/* default */.A(name) && !this._isMultigraph) {
      throw new Error('Cannot set a named edge when isMultigraph = false');
    }

    // It didn't exist, so we need to create it.
    // First ensure the nodes exist.
    this.setNode(v);
    this.setNode(w);

    this._edgeLabels[e] = valueSpecified ? value : this._defaultEdgeLabelFn(v, w, name);

    var edgeObj = edgeArgsToObj(this._isDirected, v, w, name);
    // Ensure we add undirected edges in a consistent way.
    v = edgeObj.v;
    w = edgeObj.w;

    Object.freeze(edgeObj);
    this._edgeObjs[e] = edgeObj;
    incrementOrInitEntry(this._preds[w], v);
    incrementOrInitEntry(this._sucs[v], w);
    this._in[w][e] = edgeObj;
    this._out[v][e] = edgeObj;
    this._edgeCount++;
    return this;
  }

  /**
   * Gets the label for the specified edge.
   *
   * @overload
   * @param {EdgeObj} v - Edge object.
   * @returns {EdgeLabel | undefined} the label for the edge (`v`, `w`) if the
   * graph has an edge between `v` and `w` with the optional `name`.
   * Returned `undefined` if there is no such edge in the graph.
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * Takes `O(1)` time.
   */
  /**
   * Gets the label for the specified edge.
   *
   * @overload
   * @param {NodeID | number} v - Source node ID.
   * @param {NodeID | number} w - Target node ID.
   * @param {string | number} [name] - Edge name. Only useful with multigraphs.
   * @returns {EdgeLabel | undefined} the label for the edge (`v`, `w`) if the
   * graph has an edge between `v` and `w` with the optional `name`.
   * Returned `undefined` if there is no such edge in the graph.
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * Takes `O(1)` time.
   */
  edge(v, w, name) {
    var e =
      arguments.length === 1
        ? edgeObjToId(this._isDirected, arguments[0])
        : edgeArgsToId(this._isDirected, v, w, name);
    return this._edgeLabels[e];
  }

  /**
   * Detects whether the graph contains specified edge or not.
   *
   * @overload
   * @param {EdgeObj} v - Edge object.
   * @returns {boolean} `true` if the graph has an edge between `v` and `w`
   * with the optional `name`.
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * No subgraphs are considered.
   * Takes `O(1)` time.
   */
  /**
   * Detects whether the graph contains specified edge or not.
   *
   * @overload
   * @param {NodeID | number} v - Source node ID.
   * @param {NodeID | number} w - Target node ID.
   * @param {string | number} [name] - Edge name. Only useful with multigraphs.
   * @returns {boolean} `true` if the graph has an edge between `v` and `w`
   * with the optional `name`.
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * No subgraphs are considered.
   * Takes `O(1)` time.
   */
  hasEdge(v, w, name) {
    var e =
      arguments.length === 1
        ? edgeObjToId(this._isDirected, arguments[0])
        : edgeArgsToId(this._isDirected, v, w, name);
    return Object.prototype.hasOwnProperty.call(this._edgeLabels, e);
  }

  /**
   * Removes the edge (`v`, `w`) if the graph has an edge between `v` and `w`
   * with the optional `name`. If not this function does nothing.
   *
   * @overload
   * @param {EdgeObj} v - Edge object.
   * @returns {this}
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * No subgraphs are considered.
   * Takes `O(1)` time.
   */
  /**
   * Removes the edge (`v`, `w`) if the graph has an edge between `v` and `w`
   * with the optional `name`. If not this function does nothing.
   *
   * @overload
   * @param {NodeID | number} v - Source node ID.
   * @param {NodeID | number} w - Target node ID.
   * @param {string | number} [name] - Edge name. Only useful with multigraphs.
   * @returns {this}
   * @remarks
   * `v` and `w` can be interchanged for undirected graphs.
   * Takes `O(1)` time.
   */
  removeEdge(v, w, name) {
    var e =
      arguments.length === 1
        ? edgeObjToId(this._isDirected, arguments[0])
        : edgeArgsToId(this._isDirected, v, w, name);
    var edge = this._edgeObjs[e];
    if (edge) {
      v = edge.v;
      w = edge.w;
      delete this._edgeLabels[e];
      delete this._edgeObjs[e];
      decrementOrRemoveEntry(this._preds[w], v);
      decrementOrRemoveEntry(this._sucs[v], w);
      delete this._in[w][e];
      delete this._out[v][e];
      this._edgeCount--;
    }
    return this;
  }

  /**
   * @param {NodeID | number} v - Target node ID.
   * @param {NodeID | number} [u] - Optionally filters edges down to just those
   * coming from node `u`.
   * @returns {EdgeObj[] | undefined} all edges that point to the node `v`.
   * Returns `undefined` if node `v` is not in the graph.
   * @remarks
   * Behavior is undefined for undirected graphs - use {@link nodeEdges} instead.
   * Takes `O(|E|)` time.
   */
  inEdges(v, u) {
    var inV = this._in[v];
    if (inV) {
      var edges = values/* default */.A(inV);
      if (!u) {
        return edges;
      }
      return filter/* default */.A(edges, function (edge) {
        return edge.v === u;
      });
    }
  }

  /**
   * @param {NodeID | number} v - Target node ID.
   * @param {NodeID | number} [w] - Optionally filters edges down to just those
   * that point to `w`.
   * @returns {EdgeObj[] | undefined} all edges that point to the node `v`.
   * Returns `undefined` if node `v` is not in the graph.
   * @remarks
   * Behavior is undefined for undirected graphs - use {@link nodeEdges} instead.
   * Takes `O(|E|)` time.
   */
  outEdges(v, w) {
    var outV = this._out[v];
    if (outV) {
      var edges = values/* default */.A(outV);
      if (!w) {
        return edges;
      }
      return filter/* default */.A(edges, function (edge) {
        return edge.w === w;
      });
    }
  }

  /**
   * @param {NodeID | number} v - Target Node ID.
   * @param {NodeID | number} [w] - If set, filters those edges down to just
   * those between nodes `v` and `w` regardless of direction
   * @returns {EdgeObj[] | undefined} all edges to or from node `v` regardless
   * of direction. Returns `undefined` if node `v` is not in the graph.
   * @remarks Takes `O(|E|)` time.
   */
  nodeEdges(v, w) {
    var inEdges = this.inEdges(v, w);
    if (inEdges) {
      return inEdges.concat(this.outEdges(v, w));
    }
  }
}

/* Number of nodes in the graph. Should only be changed by the implementation. */
Graph.prototype._nodeCount = 0;

/* Number of edges in the graph. Should only be changed by the implementation. */
Graph.prototype._edgeCount = 0;

/**
 * @param {Record<NodeID, number>} map - Object mapping node IDs to counts.
 * @param {NodeID | number} k - Node ID.
 */
function incrementOrInitEntry(map, k) {
  if (map[k]) {
    map[k]++;
  } else {
    map[k] = 1;
  }
}

/**
 * @param {Record<NodeID, number>} map - Object mapping node IDs to counts.
 * @param {NodeID | number} k - Node ID.
 */
function decrementOrRemoveEntry(map, k) {
  if (!--map[k]) {
    delete map[k];
  }
}

/**
 * @param {boolean} isDirected - If `false`, sorts v and w to ensure a consistent ID.
 * @param {EdgeObj['v'] | number} v_ - Source node ID.
 * @param {EdgeObj['w'] | number} w_ - Target node ID.
 * @param {EdgeObj['name']} [name] - Edge name (for multiple edges between the same nodes).
 * @returns {EdgeID} Unique ID for the edge.
 */
function edgeArgsToId(isDirected, v_, w_, name) {
  var v = '' + v_;
  var w = '' + w_;
  if (!isDirected && v > w) {
    var tmp = v;
    v = w;
    w = tmp;
  }
  return v + EDGE_KEY_DELIM + w + EDGE_KEY_DELIM + (isUndefined/* default */.A(name) ? DEFAULT_EDGE_NAME : name);
}

/**
 * @param {boolean} isDirected - If `false`, sorts v and w to ensure a consistent ID.
 * @param {EdgeObj['v'] | number} v_ - Source node ID.
 * @param {EdgeObj['w'] | number} w_ - Target node ID.
 * @param {EdgeObj['name']} [name] - Edge name (for multiple edges between the same nodes).
 * @returns {EdgeObj}
 */
function edgeArgsToObj(isDirected, v_, w_, name) {
  var v = '' + v_;
  var w = '' + w_;
  if (!isDirected && v > w) {
    var tmp = v;
    v = w;
    w = tmp;
  }
  var edgeObj = { v: v, w: w };
  if (name) {
    edgeObj.name = name;
  }
  return edgeObj;
}

/**
 * @param {boolean} isDirected - If `false`, sorts v and w to ensure a consistent ID.
 * @param {EdgeObj} edgeObj - Edge object.
 * @returns {EdgeID} Unique ID for the edge.
 */
function edgeObjToId(isDirected, edgeObj) {
  return edgeArgsToId(isDirected, edgeObj.v, edgeObj.w, edgeObj.name);
}


/***/ },

/***/ 697
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   T: () => (/* reexport safe */ _graph_js__WEBPACK_IMPORTED_MODULE_0__.T)
/* harmony export */ });
/* unused harmony export version */
/* harmony import */ var _graph_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(23126);
// Includes only the "core" of graphlib



const version = '2.1.9-pre';




/***/ },

/***/ 75937
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(72453);
/* harmony import */ var _color_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(74886);
/* IMPORT */


/* MAIN */
const channel = (color, channel) => {
    return _utils_index_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A.lang.round(_color_index_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A.parse(color)[channel]);
};
/* EXPORT */
/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (channel);


/***/ },

/***/ 80127
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _ListCache)
});

;// ./node_modules/lodash-es/_listCacheClear.js
/**
 * Removes all key-value entries from the list cache.
 *
 * @private
 * @name clear
 * @memberOf ListCache
 */
function listCacheClear() {
  this.__data__ = [];
  this.size = 0;
}

/* harmony default export */ const _listCacheClear = (listCacheClear);

// EXTERNAL MODULE: ./node_modules/lodash-es/eq.js
var eq = __webpack_require__(66984);
;// ./node_modules/lodash-es/_assocIndexOf.js


/**
 * Gets the index at which the `key` is found in `array` of key-value pairs.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {*} key The key to search for.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function assocIndexOf(array, key) {
  var length = array.length;
  while (length--) {
    if ((0,eq/* default */.A)(array[length][0], key)) {
      return length;
    }
  }
  return -1;
}

/* harmony default export */ const _assocIndexOf = (assocIndexOf);

;// ./node_modules/lodash-es/_listCacheDelete.js


/** Used for built-in method references. */
var arrayProto = Array.prototype;

/** Built-in value references. */
var splice = arrayProto.splice;

/**
 * Removes `key` and its value from the list cache.
 *
 * @private
 * @name delete
 * @memberOf ListCache
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function listCacheDelete(key) {
  var data = this.__data__,
      index = _assocIndexOf(data, key);

  if (index < 0) {
    return false;
  }
  var lastIndex = data.length - 1;
  if (index == lastIndex) {
    data.pop();
  } else {
    splice.call(data, index, 1);
  }
  --this.size;
  return true;
}

/* harmony default export */ const _listCacheDelete = (listCacheDelete);

;// ./node_modules/lodash-es/_listCacheGet.js


/**
 * Gets the list cache value for `key`.
 *
 * @private
 * @name get
 * @memberOf ListCache
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */
function listCacheGet(key) {
  var data = this.__data__,
      index = _assocIndexOf(data, key);

  return index < 0 ? undefined : data[index][1];
}

/* harmony default export */ const _listCacheGet = (listCacheGet);

;// ./node_modules/lodash-es/_listCacheHas.js


/**
 * Checks if a list cache value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf ListCache
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function listCacheHas(key) {
  return _assocIndexOf(this.__data__, key) > -1;
}

/* harmony default export */ const _listCacheHas = (listCacheHas);

;// ./node_modules/lodash-es/_listCacheSet.js


/**
 * Sets the list cache `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf ListCache
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the list cache instance.
 */
function listCacheSet(key, value) {
  var data = this.__data__,
      index = _assocIndexOf(data, key);

  if (index < 0) {
    ++this.size;
    data.push([key, value]);
  } else {
    data[index][1] = value;
  }
  return this;
}

/* harmony default export */ const _listCacheSet = (listCacheSet);

;// ./node_modules/lodash-es/_ListCache.js






/**
 * Creates an list cache object.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */
function ListCache(entries) {
  var index = -1,
      length = entries == null ? 0 : entries.length;

  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}

// Add methods to `ListCache`.
ListCache.prototype.clear = _listCacheClear;
ListCache.prototype['delete'] = _listCacheDelete;
ListCache.prototype.get = _listCacheGet;
ListCache.prototype.has = _listCacheHas;
ListCache.prototype.set = _listCacheSet;

/* harmony default export */ const _ListCache = (ListCache);


/***/ },

/***/ 68335
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _getNative_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(18744);
/* harmony import */ var _root_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(41917);



/* Built-in method references that are verified to be native. */
var Map = (0,_getNative_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(_root_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A, 'Map');

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (Map);


/***/ },

/***/ 29471
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _MapCache)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_getNative.js + 4 modules
var _getNative = __webpack_require__(18744);
;// ./node_modules/lodash-es/_nativeCreate.js


/* Built-in method references that are verified to be native. */
var nativeCreate = (0,_getNative/* default */.A)(Object, 'create');

/* harmony default export */ const _nativeCreate = (nativeCreate);

;// ./node_modules/lodash-es/_hashClear.js


/**
 * Removes all key-value entries from the hash.
 *
 * @private
 * @name clear
 * @memberOf Hash
 */
function hashClear() {
  this.__data__ = _nativeCreate ? _nativeCreate(null) : {};
  this.size = 0;
}

/* harmony default export */ const _hashClear = (hashClear);

;// ./node_modules/lodash-es/_hashDelete.js
/**
 * Removes `key` and its value from the hash.
 *
 * @private
 * @name delete
 * @memberOf Hash
 * @param {Object} hash The hash to modify.
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function hashDelete(key) {
  var result = this.has(key) && delete this.__data__[key];
  this.size -= result ? 1 : 0;
  return result;
}

/* harmony default export */ const _hashDelete = (hashDelete);

;// ./node_modules/lodash-es/_hashGet.js


/** Used to stand-in for `undefined` hash values. */
var HASH_UNDEFINED = '__lodash_hash_undefined__';

/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _hashGet_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Gets the hash value for `key`.
 *
 * @private
 * @name get
 * @memberOf Hash
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */
function hashGet(key) {
  var data = this.__data__;
  if (_nativeCreate) {
    var result = data[key];
    return result === HASH_UNDEFINED ? undefined : result;
  }
  return _hashGet_hasOwnProperty.call(data, key) ? data[key] : undefined;
}

/* harmony default export */ const _hashGet = (hashGet);

;// ./node_modules/lodash-es/_hashHas.js


/** Used for built-in method references. */
var _hashHas_objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _hashHas_hasOwnProperty = _hashHas_objectProto.hasOwnProperty;

/**
 * Checks if a hash value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf Hash
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function hashHas(key) {
  var data = this.__data__;
  return _nativeCreate ? (data[key] !== undefined) : _hashHas_hasOwnProperty.call(data, key);
}

/* harmony default export */ const _hashHas = (hashHas);

;// ./node_modules/lodash-es/_hashSet.js


/** Used to stand-in for `undefined` hash values. */
var _hashSet_HASH_UNDEFINED = '__lodash_hash_undefined__';

/**
 * Sets the hash `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf Hash
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the hash instance.
 */
function hashSet(key, value) {
  var data = this.__data__;
  this.size += this.has(key) ? 0 : 1;
  data[key] = (_nativeCreate && value === undefined) ? _hashSet_HASH_UNDEFINED : value;
  return this;
}

/* harmony default export */ const _hashSet = (hashSet);

;// ./node_modules/lodash-es/_Hash.js






/**
 * Creates a hash object.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */
function Hash(entries) {
  var index = -1,
      length = entries == null ? 0 : entries.length;

  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}

// Add methods to `Hash`.
Hash.prototype.clear = _hashClear;
Hash.prototype['delete'] = _hashDelete;
Hash.prototype.get = _hashGet;
Hash.prototype.has = _hashHas;
Hash.prototype.set = _hashSet;

/* harmony default export */ const _Hash = (Hash);

// EXTERNAL MODULE: ./node_modules/lodash-es/_ListCache.js + 6 modules
var _ListCache = __webpack_require__(80127);
// EXTERNAL MODULE: ./node_modules/lodash-es/_Map.js
var _Map = __webpack_require__(68335);
;// ./node_modules/lodash-es/_mapCacheClear.js




/**
 * Removes all key-value entries from the map.
 *
 * @private
 * @name clear
 * @memberOf MapCache
 */
function mapCacheClear() {
  this.size = 0;
  this.__data__ = {
    'hash': new _Hash,
    'map': new (_Map/* default */.A || _ListCache/* default */.A),
    'string': new _Hash
  };
}

/* harmony default export */ const _mapCacheClear = (mapCacheClear);

;// ./node_modules/lodash-es/_isKeyable.js
/**
 * Checks if `value` is suitable for use as unique object key.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is suitable, else `false`.
 */
function isKeyable(value) {
  var type = typeof value;
  return (type == 'string' || type == 'number' || type == 'symbol' || type == 'boolean')
    ? (value !== '__proto__')
    : (value === null);
}

/* harmony default export */ const _isKeyable = (isKeyable);

;// ./node_modules/lodash-es/_getMapData.js


/**
 * Gets the data for `map`.
 *
 * @private
 * @param {Object} map The map to query.
 * @param {string} key The reference key.
 * @returns {*} Returns the map data.
 */
function getMapData(map, key) {
  var data = map.__data__;
  return _isKeyable(key)
    ? data[typeof key == 'string' ? 'string' : 'hash']
    : data.map;
}

/* harmony default export */ const _getMapData = (getMapData);

;// ./node_modules/lodash-es/_mapCacheDelete.js


/**
 * Removes `key` and its value from the map.
 *
 * @private
 * @name delete
 * @memberOf MapCache
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function mapCacheDelete(key) {
  var result = _getMapData(this, key)['delete'](key);
  this.size -= result ? 1 : 0;
  return result;
}

/* harmony default export */ const _mapCacheDelete = (mapCacheDelete);

;// ./node_modules/lodash-es/_mapCacheGet.js


/**
 * Gets the map value for `key`.
 *
 * @private
 * @name get
 * @memberOf MapCache
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */
function mapCacheGet(key) {
  return _getMapData(this, key).get(key);
}

/* harmony default export */ const _mapCacheGet = (mapCacheGet);

;// ./node_modules/lodash-es/_mapCacheHas.js


/**
 * Checks if a map value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf MapCache
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function mapCacheHas(key) {
  return _getMapData(this, key).has(key);
}

/* harmony default export */ const _mapCacheHas = (mapCacheHas);

;// ./node_modules/lodash-es/_mapCacheSet.js


/**
 * Sets the map `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf MapCache
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the map cache instance.
 */
function mapCacheSet(key, value) {
  var data = _getMapData(this, key),
      size = data.size;

  data.set(key, value);
  this.size += data.size == size ? 0 : 1;
  return this;
}

/* harmony default export */ const _mapCacheSet = (mapCacheSet);

;// ./node_modules/lodash-es/_MapCache.js






/**
 * Creates a map cache object to store key-value pairs.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */
function MapCache(entries) {
  var index = -1,
      length = entries == null ? 0 : entries.length;

  this.clear();
  while (++index < length) {
    var entry = entries[index];
    this.set(entry[0], entry[1]);
  }
}

// Add methods to `MapCache`.
MapCache.prototype.clear = _mapCacheClear;
MapCache.prototype['delete'] = _mapCacheDelete;
MapCache.prototype.get = _mapCacheGet;
MapCache.prototype.has = _mapCacheHas;
MapCache.prototype.set = _mapCacheSet;

/* harmony default export */ const _MapCache = (MapCache);


/***/ },

/***/ 39857
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _getNative_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(18744);
/* harmony import */ var _root_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(41917);



/* Built-in method references that are verified to be native. */
var Set = (0,_getNative_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(_root_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A, 'Set');

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (Set);


/***/ },

/***/ 62062
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _SetCache)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_MapCache.js + 14 modules
var _MapCache = __webpack_require__(29471);
;// ./node_modules/lodash-es/_setCacheAdd.js
/** Used to stand-in for `undefined` hash values. */
var HASH_UNDEFINED = '__lodash_hash_undefined__';

/**
 * Adds `value` to the array cache.
 *
 * @private
 * @name add
 * @memberOf SetCache
 * @alias push
 * @param {*} value The value to cache.
 * @returns {Object} Returns the cache instance.
 */
function setCacheAdd(value) {
  this.__data__.set(value, HASH_UNDEFINED);
  return this;
}

/* harmony default export */ const _setCacheAdd = (setCacheAdd);

;// ./node_modules/lodash-es/_setCacheHas.js
/**
 * Checks if `value` is in the array cache.
 *
 * @private
 * @name has
 * @memberOf SetCache
 * @param {*} value The value to search for.
 * @returns {boolean} Returns `true` if `value` is found, else `false`.
 */
function setCacheHas(value) {
  return this.__data__.has(value);
}

/* harmony default export */ const _setCacheHas = (setCacheHas);

;// ./node_modules/lodash-es/_SetCache.js




/**
 *
 * Creates an array cache object to store unique values.
 *
 * @private
 * @constructor
 * @param {Array} [values] The values to cache.
 */
function SetCache(values) {
  var index = -1,
      length = values == null ? 0 : values.length;

  this.__data__ = new _MapCache/* default */.A;
  while (++index < length) {
    this.add(values[index]);
  }
}

// Add methods to `SetCache`.
SetCache.prototype.add = SetCache.prototype.push = _setCacheAdd;
SetCache.prototype.has = _setCacheHas;

/* harmony default export */ const _SetCache = (SetCache);


/***/ },

/***/ 11754
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _Stack)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_ListCache.js + 6 modules
var _ListCache = __webpack_require__(80127);
;// ./node_modules/lodash-es/_stackClear.js


/**
 * Removes all key-value entries from the stack.
 *
 * @private
 * @name clear
 * @memberOf Stack
 */
function stackClear() {
  this.__data__ = new _ListCache/* default */.A;
  this.size = 0;
}

/* harmony default export */ const _stackClear = (stackClear);

;// ./node_modules/lodash-es/_stackDelete.js
/**
 * Removes `key` and its value from the stack.
 *
 * @private
 * @name delete
 * @memberOf Stack
 * @param {string} key The key of the value to remove.
 * @returns {boolean} Returns `true` if the entry was removed, else `false`.
 */
function stackDelete(key) {
  var data = this.__data__,
      result = data['delete'](key);

  this.size = data.size;
  return result;
}

/* harmony default export */ const _stackDelete = (stackDelete);

;// ./node_modules/lodash-es/_stackGet.js
/**
 * Gets the stack value for `key`.
 *
 * @private
 * @name get
 * @memberOf Stack
 * @param {string} key The key of the value to get.
 * @returns {*} Returns the entry value.
 */
function stackGet(key) {
  return this.__data__.get(key);
}

/* harmony default export */ const _stackGet = (stackGet);

;// ./node_modules/lodash-es/_stackHas.js
/**
 * Checks if a stack value for `key` exists.
 *
 * @private
 * @name has
 * @memberOf Stack
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function stackHas(key) {
  return this.__data__.has(key);
}

/* harmony default export */ const _stackHas = (stackHas);

// EXTERNAL MODULE: ./node_modules/lodash-es/_Map.js
var _Map = __webpack_require__(68335);
// EXTERNAL MODULE: ./node_modules/lodash-es/_MapCache.js + 14 modules
var _MapCache = __webpack_require__(29471);
;// ./node_modules/lodash-es/_stackSet.js




/** Used as the size to enable large array optimizations. */
var LARGE_ARRAY_SIZE = 200;

/**
 * Sets the stack `key` to `value`.
 *
 * @private
 * @name set
 * @memberOf Stack
 * @param {string} key The key of the value to set.
 * @param {*} value The value to set.
 * @returns {Object} Returns the stack cache instance.
 */
function stackSet(key, value) {
  var data = this.__data__;
  if (data instanceof _ListCache/* default */.A) {
    var pairs = data.__data__;
    if (!_Map/* default */.A || (pairs.length < LARGE_ARRAY_SIZE - 1)) {
      pairs.push([key, value]);
      this.size = ++data.size;
      return this;
    }
    data = this.__data__ = new _MapCache/* default */.A(pairs);
  }
  data.set(key, value);
  this.size = data.size;
  return this;
}

/* harmony default export */ const _stackSet = (stackSet);

;// ./node_modules/lodash-es/_Stack.js







/**
 * Creates a stack cache object to store key-value pairs.
 *
 * @private
 * @constructor
 * @param {Array} [entries] The key-value pairs to cache.
 */
function Stack(entries) {
  var data = this.__data__ = new _ListCache/* default */.A(entries);
  this.size = data.size;
}

// Add methods to `Stack`.
Stack.prototype.clear = _stackClear;
Stack.prototype['delete'] = _stackDelete;
Stack.prototype.get = _stackGet;
Stack.prototype.has = _stackHas;
Stack.prototype.set = _stackSet;

/* harmony default export */ const _Stack = (Stack);


/***/ },

/***/ 241
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _root_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(41917);


/** Built-in value references. */
var Symbol = _root_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A.Symbol;

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (Symbol);


/***/ },

/***/ 43988
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _root_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(41917);


/** Built-in value references. */
var Uint8Array = _root_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A.Uint8Array;

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (Uint8Array);


/***/ },

/***/ 72641
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * A specialized version of `_.forEach` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns `array`.
 */
function arrayEach(array, iteratee) {
  var index = -1,
      length = array == null ? 0 : array.length;

  while (++index < length) {
    if (iteratee(array[index], index, array) === false) {
      break;
    }
  }
  return array;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (arrayEach);


/***/ },

/***/ 2634
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * A specialized version of `_.filter` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 */
function arrayFilter(array, predicate) {
  var index = -1,
      length = array == null ? 0 : array.length,
      resIndex = 0,
      result = [];

  while (++index < length) {
    var value = array[index];
    if (predicate(value, index, array)) {
      result[resIndex++] = value;
    }
  }
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (arrayFilter);


/***/ },

/***/ 83607
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _arrayLikeKeys)
});

;// ./node_modules/lodash-es/_baseTimes.js
/**
 * The base implementation of `_.times` without support for iteratee shorthands
 * or max array length checks.
 *
 * @private
 * @param {number} n The number of times to invoke `iteratee`.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the array of results.
 */
function baseTimes(n, iteratee) {
  var index = -1,
      result = Array(n);

  while (++index < n) {
    result[index] = iteratee(index);
  }
  return result;
}

/* harmony default export */ const _baseTimes = (baseTimes);

// EXTERNAL MODULE: ./node_modules/lodash-es/isArguments.js + 1 modules
var isArguments = __webpack_require__(29893);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/isBuffer.js + 1 modules
var isBuffer = __webpack_require__(99912);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isIndex.js
var _isIndex = __webpack_require__(25353);
// EXTERNAL MODULE: ./node_modules/lodash-es/isTypedArray.js + 1 modules
var isTypedArray = __webpack_require__(33858);
;// ./node_modules/lodash-es/_arrayLikeKeys.js







/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _arrayLikeKeys_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Creates an array of the enumerable property names of the array-like `value`.
 *
 * @private
 * @param {*} value The value to query.
 * @param {boolean} inherited Specify returning inherited property names.
 * @returns {Array} Returns the array of property names.
 */
function arrayLikeKeys(value, inherited) {
  var isArr = (0,isArray/* default */.A)(value),
      isArg = !isArr && (0,isArguments/* default */.A)(value),
      isBuff = !isArr && !isArg && (0,isBuffer/* default */.A)(value),
      isType = !isArr && !isArg && !isBuff && (0,isTypedArray/* default */.A)(value),
      skipIndexes = isArr || isArg || isBuff || isType,
      result = skipIndexes ? _baseTimes(value.length, String) : [],
      length = result.length;

  for (var key in value) {
    if ((inherited || _arrayLikeKeys_hasOwnProperty.call(value, key)) &&
        !(skipIndexes && (
           // Safari 9 has enumerable `arguments.length` in strict mode.
           key == 'length' ||
           // Node.js 0.10 has enumerable non-index properties on buffers.
           (isBuff && (key == 'offset' || key == 'parent')) ||
           // PhantomJS 2 has enumerable non-index properties on typed arrays.
           (isType && (key == 'buffer' || key == 'byteLength' || key == 'byteOffset')) ||
           // Skip index properties.
           (0,_isIndex/* default */.A)(key, length)
        ))) {
      result.push(key);
    }
  }
  return result;
}

/* harmony default export */ const _arrayLikeKeys = (arrayLikeKeys);


/***/ },

/***/ 45572
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * A specialized version of `_.map` for arrays without support for iteratee
 * shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 */
function arrayMap(array, iteratee) {
  var index = -1,
      length = array == null ? 0 : array.length,
      result = Array(length);

  while (++index < length) {
    result[index] = iteratee(array[index], index, array);
  }
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (arrayMap);


/***/ },

/***/ 76912
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Appends the elements of `values` to `array`.
 *
 * @private
 * @param {Array} array The array to modify.
 * @param {Array} values The values to append.
 * @returns {Array} Returns `array`.
 */
function arrayPush(array, values) {
  var index = -1,
      length = values.length,
      offset = array.length;

  while (++index < length) {
    array[offset + index] = values[index];
  }
  return array;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (arrayPush);


/***/ },

/***/ 6240
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseEach)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseForOwn.js
var _baseForOwn = __webpack_require__(79841);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArrayLike.js
var isArrayLike = __webpack_require__(38446);
;// ./node_modules/lodash-es/_createBaseEach.js


/**
 * Creates a `baseEach` or `baseEachRight` function.
 *
 * @private
 * @param {Function} eachFunc The function to iterate over a collection.
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new base function.
 */
function createBaseEach(eachFunc, fromRight) {
  return function(collection, iteratee) {
    if (collection == null) {
      return collection;
    }
    if (!(0,isArrayLike/* default */.A)(collection)) {
      return eachFunc(collection, iteratee);
    }
    var length = collection.length,
        index = fromRight ? length : -1,
        iterable = Object(collection);

    while ((fromRight ? index-- : ++index < length)) {
      if (iteratee(iterable[index], index, iterable) === false) {
        break;
      }
    }
    return collection;
  };
}

/* harmony default export */ const _createBaseEach = (createBaseEach);

;// ./node_modules/lodash-es/_baseEach.js



/**
 * The base implementation of `_.forEach` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array|Object} Returns `collection`.
 */
var baseEach = _createBaseEach(_baseForOwn/* default */.A);

/* harmony default export */ const _baseEach = (baseEach);


/***/ },

/***/ 25707
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * The base implementation of `_.findIndex` and `_.findLastIndex` without
 * support for iteratee shorthands.
 *
 * @private
 * @param {Array} array The array to inspect.
 * @param {Function} predicate The function invoked per iteration.
 * @param {number} fromIndex The index to search from.
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {number} Returns the index of the matched value, else `-1`.
 */
function baseFindIndex(array, predicate, fromIndex, fromRight) {
  var length = array.length,
      index = fromIndex + (fromRight ? 1 : -1);

  while ((fromRight ? index-- : ++index < length)) {
    if (predicate(array[index], index, array)) {
      return index;
    }
  }
  return -1;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseFindIndex);


/***/ },

/***/ 13588
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseFlatten)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayPush.js
var _arrayPush = __webpack_require__(76912);
// EXTERNAL MODULE: ./node_modules/lodash-es/_Symbol.js
var _Symbol = __webpack_require__(241);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArguments.js + 1 modules
var isArguments = __webpack_require__(29893);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
;// ./node_modules/lodash-es/_isFlattenable.js




/** Built-in value references. */
var spreadableSymbol = _Symbol/* default */.A ? _Symbol/* default */.A.isConcatSpreadable : undefined;

/**
 * Checks if `value` is a flattenable `arguments` object or array.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is flattenable, else `false`.
 */
function isFlattenable(value) {
  return (0,isArray/* default */.A)(value) || (0,isArguments/* default */.A)(value) ||
    !!(spreadableSymbol && value && value[spreadableSymbol]);
}

/* harmony default export */ const _isFlattenable = (isFlattenable);

;// ./node_modules/lodash-es/_baseFlatten.js



/**
 * The base implementation of `_.flatten` with support for restricting flattening.
 *
 * @private
 * @param {Array} array The array to flatten.
 * @param {number} depth The maximum recursion depth.
 * @param {boolean} [predicate=isFlattenable] The function invoked per iteration.
 * @param {boolean} [isStrict] Restrict to values that pass `predicate` checks.
 * @param {Array} [result=[]] The initial result value.
 * @returns {Array} Returns the new flattened array.
 */
function baseFlatten(array, depth, predicate, isStrict, result) {
  var index = -1,
      length = array.length;

  predicate || (predicate = _isFlattenable);
  result || (result = []);

  while (++index < length) {
    var value = array[index];
    if (depth > 0 && predicate(value)) {
      if (depth > 1) {
        // Recursively flatten arrays (susceptible to call stack limits).
        baseFlatten(value, depth - 1, predicate, isStrict, result);
      } else {
        (0,_arrayPush/* default */.A)(result, value);
      }
    } else if (!isStrict) {
      result[result.length] = value;
    }
  }
  return result;
}

/* harmony default export */ const _baseFlatten = (baseFlatten);


/***/ },

/***/ 4574
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseFor)
});

;// ./node_modules/lodash-es/_createBaseFor.js
/**
 * Creates a base function for methods like `_.forIn` and `_.forOwn`.
 *
 * @private
 * @param {boolean} [fromRight] Specify iterating from right to left.
 * @returns {Function} Returns the new base function.
 */
function createBaseFor(fromRight) {
  return function(object, iteratee, keysFunc) {
    var index = -1,
        iterable = Object(object),
        props = keysFunc(object),
        length = props.length;

    while (length--) {
      var key = props[fromRight ? length : ++index];
      if (iteratee(iterable[key], key, iterable) === false) {
        break;
      }
    }
    return object;
  };
}

/* harmony default export */ const _createBaseFor = (createBaseFor);

;// ./node_modules/lodash-es/_baseFor.js


/**
 * The base implementation of `baseForOwn` which iterates over `object`
 * properties returned by `keysFunc` and invokes `iteratee` for each property.
 * Iteratee functions may exit iteration early by explicitly returning `false`.
 *
 * @private
 * @param {Object} object The object to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {Function} keysFunc The function to get the keys of `object`.
 * @returns {Object} Returns `object`.
 */
var baseFor = _createBaseFor();

/* harmony default export */ const _baseFor = (baseFor);


/***/ },

/***/ 79841
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseFor_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(4574);
/* harmony import */ var _keys_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(27422);



/**
 * The base implementation of `_.forOwn` without support for iteratee shorthands.
 *
 * @private
 * @param {Object} object The object to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Object} Returns `object`.
 */
function baseForOwn(object, iteratee) {
  return object && (0,_baseFor_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object, iteratee, _keys_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseForOwn);


/***/ },

/***/ 66318
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _castPath_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(63442);
/* harmony import */ var _toKey_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(30901);



/**
 * The base implementation of `_.get` without support for default values.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array|string} path The path of the property to get.
 * @returns {*} Returns the resolved value.
 */
function baseGet(object, path) {
  path = (0,_castPath_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(path, object);

  var index = 0,
      length = path.length;

  while (object != null && index < length) {
    object = object[(0,_toKey_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(path[index++])];
  }
  return (index && index == length) ? object : undefined;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseGet);


/***/ },

/***/ 33831
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _arrayPush_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(76912);
/* harmony import */ var _isArray_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(92049);



/**
 * The base implementation of `getAllKeys` and `getAllKeysIn` which uses
 * `keysFunc` and `symbolsFunc` to get the enumerable property names and
 * symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Function} keysFunc The function to get the keys of `object`.
 * @param {Function} symbolsFunc The function to get the symbols of `object`.
 * @returns {Array} Returns the array of property names and symbols.
 */
function baseGetAllKeys(object, keysFunc, symbolsFunc) {
  var result = keysFunc(object);
  return (0,_isArray_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(object) ? result : (0,_arrayPush_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(result, symbolsFunc(object));
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseGetAllKeys);


/***/ },

/***/ 88496
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseGetTag)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_Symbol.js
var _Symbol = __webpack_require__(241);
;// ./node_modules/lodash-es/_getRawTag.js


/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _getRawTag_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Used to resolve the
 * [`toStringTag`](http://ecma-international.org/ecma-262/7.0/#sec-object.prototype.tostring)
 * of values.
 */
var nativeObjectToString = objectProto.toString;

/** Built-in value references. */
var symToStringTag = _Symbol/* default */.A ? _Symbol/* default */.A.toStringTag : undefined;

/**
 * A specialized version of `baseGetTag` which ignores `Symbol.toStringTag` values.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the raw `toStringTag`.
 */
function getRawTag(value) {
  var isOwn = _getRawTag_hasOwnProperty.call(value, symToStringTag),
      tag = value[symToStringTag];

  try {
    value[symToStringTag] = undefined;
    var unmasked = true;
  } catch (e) {}

  var result = nativeObjectToString.call(value);
  if (unmasked) {
    if (isOwn) {
      value[symToStringTag] = tag;
    } else {
      delete value[symToStringTag];
    }
  }
  return result;
}

/* harmony default export */ const _getRawTag = (getRawTag);

;// ./node_modules/lodash-es/_objectToString.js
/** Used for built-in method references. */
var _objectToString_objectProto = Object.prototype;

/**
 * Used to resolve the
 * [`toStringTag`](http://ecma-international.org/ecma-262/7.0/#sec-object.prototype.tostring)
 * of values.
 */
var _objectToString_nativeObjectToString = _objectToString_objectProto.toString;

/**
 * Converts `value` to a string using `Object.prototype.toString`.
 *
 * @private
 * @param {*} value The value to convert.
 * @returns {string} Returns the converted string.
 */
function objectToString(value) {
  return _objectToString_nativeObjectToString.call(value);
}

/* harmony default export */ const _objectToString = (objectToString);

;// ./node_modules/lodash-es/_baseGetTag.js




/** `Object#toString` result references. */
var nullTag = '[object Null]',
    undefinedTag = '[object Undefined]';

/** Built-in value references. */
var _baseGetTag_symToStringTag = _Symbol/* default */.A ? _Symbol/* default */.A.toStringTag : undefined;

/**
 * The base implementation of `getTag` without fallbacks for buggy environments.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the `toStringTag`.
 */
function baseGetTag(value) {
  if (value == null) {
    return value === undefined ? undefinedTag : nullTag;
  }
  return (_baseGetTag_symToStringTag && _baseGetTag_symToStringTag in Object(value))
    ? _getRawTag(value)
    : _objectToString(value);
}

/* harmony default export */ const _baseGetTag = (baseGetTag);


/***/ },

/***/ 49574
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseIteratee)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_Stack.js + 5 modules
var _Stack = __webpack_require__(11754);
// EXTERNAL MODULE: ./node_modules/lodash-es/_SetCache.js + 2 modules
var _SetCache = __webpack_require__(62062);
;// ./node_modules/lodash-es/_arraySome.js
/**
 * A specialized version of `_.some` for arrays without support for iteratee
 * shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {boolean} Returns `true` if any element passes the predicate check,
 *  else `false`.
 */
function arraySome(array, predicate) {
  var index = -1,
      length = array == null ? 0 : array.length;

  while (++index < length) {
    if (predicate(array[index], index, array)) {
      return true;
    }
  }
  return false;
}

/* harmony default export */ const _arraySome = (arraySome);

// EXTERNAL MODULE: ./node_modules/lodash-es/_cacheHas.js
var _cacheHas = __webpack_require__(64099);
;// ./node_modules/lodash-es/_equalArrays.js




/** Used to compose bitmasks for value comparisons. */
var COMPARE_PARTIAL_FLAG = 1,
    COMPARE_UNORDERED_FLAG = 2;

/**
 * A specialized version of `baseIsEqualDeep` for arrays with support for
 * partial deep comparisons.
 *
 * @private
 * @param {Array} array The array to compare.
 * @param {Array} other The other array to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `array` and `other` objects.
 * @returns {boolean} Returns `true` if the arrays are equivalent, else `false`.
 */
function equalArrays(array, other, bitmask, customizer, equalFunc, stack) {
  var isPartial = bitmask & COMPARE_PARTIAL_FLAG,
      arrLength = array.length,
      othLength = other.length;

  if (arrLength != othLength && !(isPartial && othLength > arrLength)) {
    return false;
  }
  // Check that cyclic values are equal.
  var arrStacked = stack.get(array);
  var othStacked = stack.get(other);
  if (arrStacked && othStacked) {
    return arrStacked == other && othStacked == array;
  }
  var index = -1,
      result = true,
      seen = (bitmask & COMPARE_UNORDERED_FLAG) ? new _SetCache/* default */.A : undefined;

  stack.set(array, other);
  stack.set(other, array);

  // Ignore non-index properties.
  while (++index < arrLength) {
    var arrValue = array[index],
        othValue = other[index];

    if (customizer) {
      var compared = isPartial
        ? customizer(othValue, arrValue, index, other, array, stack)
        : customizer(arrValue, othValue, index, array, other, stack);
    }
    if (compared !== undefined) {
      if (compared) {
        continue;
      }
      result = false;
      break;
    }
    // Recursively compare arrays (susceptible to call stack limits).
    if (seen) {
      if (!_arraySome(other, function(othValue, othIndex) {
            if (!(0,_cacheHas/* default */.A)(seen, othIndex) &&
                (arrValue === othValue || equalFunc(arrValue, othValue, bitmask, customizer, stack))) {
              return seen.push(othIndex);
            }
          })) {
        result = false;
        break;
      }
    } else if (!(
          arrValue === othValue ||
            equalFunc(arrValue, othValue, bitmask, customizer, stack)
        )) {
      result = false;
      break;
    }
  }
  stack['delete'](array);
  stack['delete'](other);
  return result;
}

/* harmony default export */ const _equalArrays = (equalArrays);

// EXTERNAL MODULE: ./node_modules/lodash-es/_Symbol.js
var _Symbol = __webpack_require__(241);
// EXTERNAL MODULE: ./node_modules/lodash-es/_Uint8Array.js
var _Uint8Array = __webpack_require__(43988);
// EXTERNAL MODULE: ./node_modules/lodash-es/eq.js
var eq = __webpack_require__(66984);
;// ./node_modules/lodash-es/_mapToArray.js
/**
 * Converts `map` to its key-value pairs.
 *
 * @private
 * @param {Object} map The map to convert.
 * @returns {Array} Returns the key-value pairs.
 */
function mapToArray(map) {
  var index = -1,
      result = Array(map.size);

  map.forEach(function(value, key) {
    result[++index] = [key, value];
  });
  return result;
}

/* harmony default export */ const _mapToArray = (mapToArray);

// EXTERNAL MODULE: ./node_modules/lodash-es/_setToArray.js
var _setToArray = __webpack_require__(29959);
;// ./node_modules/lodash-es/_equalByTag.js







/** Used to compose bitmasks for value comparisons. */
var _equalByTag_COMPARE_PARTIAL_FLAG = 1,
    _equalByTag_COMPARE_UNORDERED_FLAG = 2;

/** `Object#toString` result references. */
var boolTag = '[object Boolean]',
    dateTag = '[object Date]',
    errorTag = '[object Error]',
    mapTag = '[object Map]',
    numberTag = '[object Number]',
    regexpTag = '[object RegExp]',
    setTag = '[object Set]',
    stringTag = '[object String]',
    symbolTag = '[object Symbol]';

var arrayBufferTag = '[object ArrayBuffer]',
    dataViewTag = '[object DataView]';

/** Used to convert symbols to primitives and strings. */
var symbolProto = _Symbol/* default */.A ? _Symbol/* default */.A.prototype : undefined,
    symbolValueOf = symbolProto ? symbolProto.valueOf : undefined;

/**
 * A specialized version of `baseIsEqualDeep` for comparing objects of
 * the same `toStringTag`.
 *
 * **Note:** This function only supports comparing values with tags of
 * `Boolean`, `Date`, `Error`, `Number`, `RegExp`, or `String`.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {string} tag The `toStringTag` of the objects to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */
function equalByTag(object, other, tag, bitmask, customizer, equalFunc, stack) {
  switch (tag) {
    case dataViewTag:
      if ((object.byteLength != other.byteLength) ||
          (object.byteOffset != other.byteOffset)) {
        return false;
      }
      object = object.buffer;
      other = other.buffer;

    case arrayBufferTag:
      if ((object.byteLength != other.byteLength) ||
          !equalFunc(new _Uint8Array/* default */.A(object), new _Uint8Array/* default */.A(other))) {
        return false;
      }
      return true;

    case boolTag:
    case dateTag:
    case numberTag:
      // Coerce booleans to `1` or `0` and dates to milliseconds.
      // Invalid dates are coerced to `NaN`.
      return (0,eq/* default */.A)(+object, +other);

    case errorTag:
      return object.name == other.name && object.message == other.message;

    case regexpTag:
    case stringTag:
      // Coerce regexes to strings and treat strings, primitives and objects,
      // as equal. See http://www.ecma-international.org/ecma-262/7.0/#sec-regexp.prototype.tostring
      // for more details.
      return object == (other + '');

    case mapTag:
      var convert = _mapToArray;

    case setTag:
      var isPartial = bitmask & _equalByTag_COMPARE_PARTIAL_FLAG;
      convert || (convert = _setToArray/* default */.A);

      if (object.size != other.size && !isPartial) {
        return false;
      }
      // Assume cyclic values are equal.
      var stacked = stack.get(object);
      if (stacked) {
        return stacked == other;
      }
      bitmask |= _equalByTag_COMPARE_UNORDERED_FLAG;

      // Recursively compare objects (susceptible to call stack limits).
      stack.set(object, other);
      var result = _equalArrays(convert(object), convert(other), bitmask, customizer, equalFunc, stack);
      stack['delete'](object);
      return result;

    case symbolTag:
      if (symbolValueOf) {
        return symbolValueOf.call(object) == symbolValueOf.call(other);
      }
  }
  return false;
}

/* harmony default export */ const _equalByTag = (equalByTag);

// EXTERNAL MODULE: ./node_modules/lodash-es/_getAllKeys.js
var _getAllKeys = __webpack_require__(19042);
;// ./node_modules/lodash-es/_equalObjects.js


/** Used to compose bitmasks for value comparisons. */
var _equalObjects_COMPARE_PARTIAL_FLAG = 1;

/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _equalObjects_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * A specialized version of `baseIsEqualDeep` for objects with support for
 * partial deep comparisons.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} stack Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */
function equalObjects(object, other, bitmask, customizer, equalFunc, stack) {
  var isPartial = bitmask & _equalObjects_COMPARE_PARTIAL_FLAG,
      objProps = (0,_getAllKeys/* default */.A)(object),
      objLength = objProps.length,
      othProps = (0,_getAllKeys/* default */.A)(other),
      othLength = othProps.length;

  if (objLength != othLength && !isPartial) {
    return false;
  }
  var index = objLength;
  while (index--) {
    var key = objProps[index];
    if (!(isPartial ? key in other : _equalObjects_hasOwnProperty.call(other, key))) {
      return false;
    }
  }
  // Check that cyclic values are equal.
  var objStacked = stack.get(object);
  var othStacked = stack.get(other);
  if (objStacked && othStacked) {
    return objStacked == other && othStacked == object;
  }
  var result = true;
  stack.set(object, other);
  stack.set(other, object);

  var skipCtor = isPartial;
  while (++index < objLength) {
    key = objProps[index];
    var objValue = object[key],
        othValue = other[key];

    if (customizer) {
      var compared = isPartial
        ? customizer(othValue, objValue, key, other, object, stack)
        : customizer(objValue, othValue, key, object, other, stack);
    }
    // Recursively compare objects (susceptible to call stack limits).
    if (!(compared === undefined
          ? (objValue === othValue || equalFunc(objValue, othValue, bitmask, customizer, stack))
          : compared
        )) {
      result = false;
      break;
    }
    skipCtor || (skipCtor = key == 'constructor');
  }
  if (result && !skipCtor) {
    var objCtor = object.constructor,
        othCtor = other.constructor;

    // Non `Object` object instances with different constructors are not equal.
    if (objCtor != othCtor &&
        ('constructor' in object && 'constructor' in other) &&
        !(typeof objCtor == 'function' && objCtor instanceof objCtor &&
          typeof othCtor == 'function' && othCtor instanceof othCtor)) {
      result = false;
    }
  }
  stack['delete'](object);
  stack['delete'](other);
  return result;
}

/* harmony default export */ const _equalObjects = (equalObjects);

// EXTERNAL MODULE: ./node_modules/lodash-es/_getTag.js + 3 modules
var _getTag = __webpack_require__(9779);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/isBuffer.js + 1 modules
var isBuffer = __webpack_require__(99912);
// EXTERNAL MODULE: ./node_modules/lodash-es/isTypedArray.js + 1 modules
var isTypedArray = __webpack_require__(33858);
;// ./node_modules/lodash-es/_baseIsEqualDeep.js









/** Used to compose bitmasks for value comparisons. */
var _baseIsEqualDeep_COMPARE_PARTIAL_FLAG = 1;

/** `Object#toString` result references. */
var argsTag = '[object Arguments]',
    arrayTag = '[object Array]',
    objectTag = '[object Object]';

/** Used for built-in method references. */
var _baseIsEqualDeep_objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _baseIsEqualDeep_hasOwnProperty = _baseIsEqualDeep_objectProto.hasOwnProperty;

/**
 * A specialized version of `baseIsEqual` for arrays and objects which performs
 * deep comparisons and tracks traversed objects enabling objects with circular
 * references to be compared.
 *
 * @private
 * @param {Object} object The object to compare.
 * @param {Object} other The other object to compare.
 * @param {number} bitmask The bitmask flags. See `baseIsEqual` for more details.
 * @param {Function} customizer The function to customize comparisons.
 * @param {Function} equalFunc The function to determine equivalents of values.
 * @param {Object} [stack] Tracks traversed `object` and `other` objects.
 * @returns {boolean} Returns `true` if the objects are equivalent, else `false`.
 */
function baseIsEqualDeep(object, other, bitmask, customizer, equalFunc, stack) {
  var objIsArr = (0,isArray/* default */.A)(object),
      othIsArr = (0,isArray/* default */.A)(other),
      objTag = objIsArr ? arrayTag : (0,_getTag/* default */.A)(object),
      othTag = othIsArr ? arrayTag : (0,_getTag/* default */.A)(other);

  objTag = objTag == argsTag ? objectTag : objTag;
  othTag = othTag == argsTag ? objectTag : othTag;

  var objIsObj = objTag == objectTag,
      othIsObj = othTag == objectTag,
      isSameTag = objTag == othTag;

  if (isSameTag && (0,isBuffer/* default */.A)(object)) {
    if (!(0,isBuffer/* default */.A)(other)) {
      return false;
    }
    objIsArr = true;
    objIsObj = false;
  }
  if (isSameTag && !objIsObj) {
    stack || (stack = new _Stack/* default */.A);
    return (objIsArr || (0,isTypedArray/* default */.A)(object))
      ? _equalArrays(object, other, bitmask, customizer, equalFunc, stack)
      : _equalByTag(object, other, objTag, bitmask, customizer, equalFunc, stack);
  }
  if (!(bitmask & _baseIsEqualDeep_COMPARE_PARTIAL_FLAG)) {
    var objIsWrapped = objIsObj && _baseIsEqualDeep_hasOwnProperty.call(object, '__wrapped__'),
        othIsWrapped = othIsObj && _baseIsEqualDeep_hasOwnProperty.call(other, '__wrapped__');

    if (objIsWrapped || othIsWrapped) {
      var objUnwrapped = objIsWrapped ? object.value() : object,
          othUnwrapped = othIsWrapped ? other.value() : other;

      stack || (stack = new _Stack/* default */.A);
      return equalFunc(objUnwrapped, othUnwrapped, bitmask, customizer, stack);
    }
  }
  if (!isSameTag) {
    return false;
  }
  stack || (stack = new _Stack/* default */.A);
  return _equalObjects(object, other, bitmask, customizer, equalFunc, stack);
}

/* harmony default export */ const _baseIsEqualDeep = (baseIsEqualDeep);

// EXTERNAL MODULE: ./node_modules/lodash-es/isObjectLike.js
var isObjectLike = __webpack_require__(53098);
;// ./node_modules/lodash-es/_baseIsEqual.js



/**
 * The base implementation of `_.isEqual` which supports partial comparisons
 * and tracks traversed objects.
 *
 * @private
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @param {boolean} bitmask The bitmask flags.
 *  1 - Unordered comparison
 *  2 - Partial comparison
 * @param {Function} [customizer] The function to customize comparisons.
 * @param {Object} [stack] Tracks traversed `value` and `other` objects.
 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
 */
function baseIsEqual(value, other, bitmask, customizer, stack) {
  if (value === other) {
    return true;
  }
  if (value == null || other == null || (!(0,isObjectLike/* default */.A)(value) && !(0,isObjectLike/* default */.A)(other))) {
    return value !== value && other !== other;
  }
  return _baseIsEqualDeep(value, other, bitmask, customizer, baseIsEqual, stack);
}

/* harmony default export */ const _baseIsEqual = (baseIsEqual);

;// ./node_modules/lodash-es/_baseIsMatch.js



/** Used to compose bitmasks for value comparisons. */
var _baseIsMatch_COMPARE_PARTIAL_FLAG = 1,
    _baseIsMatch_COMPARE_UNORDERED_FLAG = 2;

/**
 * The base implementation of `_.isMatch` without support for iteratee shorthands.
 *
 * @private
 * @param {Object} object The object to inspect.
 * @param {Object} source The object of property values to match.
 * @param {Array} matchData The property names, values, and compare flags to match.
 * @param {Function} [customizer] The function to customize comparisons.
 * @returns {boolean} Returns `true` if `object` is a match, else `false`.
 */
function baseIsMatch(object, source, matchData, customizer) {
  var index = matchData.length,
      length = index,
      noCustomizer = !customizer;

  if (object == null) {
    return !length;
  }
  object = Object(object);
  while (index--) {
    var data = matchData[index];
    if ((noCustomizer && data[2])
          ? data[1] !== object[data[0]]
          : !(data[0] in object)
        ) {
      return false;
    }
  }
  while (++index < length) {
    data = matchData[index];
    var key = data[0],
        objValue = object[key],
        srcValue = data[1];

    if (noCustomizer && data[2]) {
      if (objValue === undefined && !(key in object)) {
        return false;
      }
    } else {
      var stack = new _Stack/* default */.A;
      if (customizer) {
        var result = customizer(objValue, srcValue, key, object, source, stack);
      }
      if (!(result === undefined
            ? _baseIsEqual(srcValue, objValue, _baseIsMatch_COMPARE_PARTIAL_FLAG | _baseIsMatch_COMPARE_UNORDERED_FLAG, customizer, stack)
            : result
          )) {
        return false;
      }
    }
  }
  return true;
}

/* harmony default export */ const _baseIsMatch = (baseIsMatch);

// EXTERNAL MODULE: ./node_modules/lodash-es/isObject.js
var isObject = __webpack_require__(23149);
;// ./node_modules/lodash-es/_isStrictComparable.js


/**
 * Checks if `value` is suitable for strict equality comparisons, i.e. `===`.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` if suitable for strict
 *  equality comparisons, else `false`.
 */
function isStrictComparable(value) {
  return value === value && !(0,isObject/* default */.A)(value);
}

/* harmony default export */ const _isStrictComparable = (isStrictComparable);

// EXTERNAL MODULE: ./node_modules/lodash-es/keys.js
var keys = __webpack_require__(27422);
;// ./node_modules/lodash-es/_getMatchData.js



/**
 * Gets the property names, values, and compare flags of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the match data of `object`.
 */
function getMatchData(object) {
  var result = (0,keys/* default */.A)(object),
      length = result.length;

  while (length--) {
    var key = result[length],
        value = object[key];

    result[length] = [key, value, _isStrictComparable(value)];
  }
  return result;
}

/* harmony default export */ const _getMatchData = (getMatchData);

;// ./node_modules/lodash-es/_matchesStrictComparable.js
/**
 * A specialized version of `matchesProperty` for source values suitable
 * for strict equality comparisons, i.e. `===`.
 *
 * @private
 * @param {string} key The key of the property to get.
 * @param {*} srcValue The value to match.
 * @returns {Function} Returns the new spec function.
 */
function matchesStrictComparable(key, srcValue) {
  return function(object) {
    if (object == null) {
      return false;
    }
    return object[key] === srcValue &&
      (srcValue !== undefined || (key in Object(object)));
  };
}

/* harmony default export */ const _matchesStrictComparable = (matchesStrictComparable);

;// ./node_modules/lodash-es/_baseMatches.js




/**
 * The base implementation of `_.matches` which doesn't clone `source`.
 *
 * @private
 * @param {Object} source The object of property values to match.
 * @returns {Function} Returns the new spec function.
 */
function baseMatches(source) {
  var matchData = _getMatchData(source);
  if (matchData.length == 1 && matchData[0][2]) {
    return _matchesStrictComparable(matchData[0][0], matchData[0][1]);
  }
  return function(object) {
    return object === source || _baseIsMatch(object, source, matchData);
  };
}

/* harmony default export */ const _baseMatches = (baseMatches);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseGet.js
var _baseGet = __webpack_require__(66318);
;// ./node_modules/lodash-es/get.js


/**
 * Gets the value at `path` of `object`. If the resolved value is
 * `undefined`, the `defaultValue` is returned in its place.
 *
 * @static
 * @memberOf _
 * @since 3.7.0
 * @category Object
 * @param {Object} object The object to query.
 * @param {Array|string} path The path of the property to get.
 * @param {*} [defaultValue] The value returned for `undefined` resolved values.
 * @returns {*} Returns the resolved value.
 * @example
 *
 * var object = { 'a': [{ 'b': { 'c': 3 } }] };
 *
 * _.get(object, 'a[0].b.c');
 * // => 3
 *
 * _.get(object, ['a', '0', 'b', 'c']);
 * // => 3
 *
 * _.get(object, 'a.b.c', 'default');
 * // => 'default'
 */
function get(object, path, defaultValue) {
  var result = object == null ? undefined : (0,_baseGet/* default */.A)(object, path);
  return result === undefined ? defaultValue : result;
}

/* harmony default export */ const lodash_es_get = (get);

// EXTERNAL MODULE: ./node_modules/lodash-es/hasIn.js + 1 modules
var hasIn = __webpack_require__(39188);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isKey.js
var _isKey = __webpack_require__(86586);
// EXTERNAL MODULE: ./node_modules/lodash-es/_toKey.js
var _toKey = __webpack_require__(30901);
;// ./node_modules/lodash-es/_baseMatchesProperty.js








/** Used to compose bitmasks for value comparisons. */
var _baseMatchesProperty_COMPARE_PARTIAL_FLAG = 1,
    _baseMatchesProperty_COMPARE_UNORDERED_FLAG = 2;

/**
 * The base implementation of `_.matchesProperty` which doesn't clone `srcValue`.
 *
 * @private
 * @param {string} path The path of the property to get.
 * @param {*} srcValue The value to match.
 * @returns {Function} Returns the new spec function.
 */
function baseMatchesProperty(path, srcValue) {
  if ((0,_isKey/* default */.A)(path) && _isStrictComparable(srcValue)) {
    return _matchesStrictComparable((0,_toKey/* default */.A)(path), srcValue);
  }
  return function(object) {
    var objValue = lodash_es_get(object, path);
    return (objValue === undefined && objValue === srcValue)
      ? (0,hasIn/* default */.A)(object, path)
      : _baseIsEqual(srcValue, objValue, _baseMatchesProperty_COMPARE_PARTIAL_FLAG | _baseMatchesProperty_COMPARE_UNORDERED_FLAG);
  };
}

/* harmony default export */ const _baseMatchesProperty = (baseMatchesProperty);

// EXTERNAL MODULE: ./node_modules/lodash-es/identity.js
var identity = __webpack_require__(29008);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseProperty.js
var _baseProperty = __webpack_require__(70805);
;// ./node_modules/lodash-es/_basePropertyDeep.js


/**
 * A specialized version of `baseProperty` which supports deep paths.
 *
 * @private
 * @param {Array|string} path The path of the property to get.
 * @returns {Function} Returns the new accessor function.
 */
function basePropertyDeep(path) {
  return function(object) {
    return (0,_baseGet/* default */.A)(object, path);
  };
}

/* harmony default export */ const _basePropertyDeep = (basePropertyDeep);

;// ./node_modules/lodash-es/property.js





/**
 * Creates a function that returns the value at `path` of a given object.
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Util
 * @param {Array|string} path The path of the property to get.
 * @returns {Function} Returns the new accessor function.
 * @example
 *
 * var objects = [
 *   { 'a': { 'b': 2 } },
 *   { 'a': { 'b': 1 } }
 * ];
 *
 * _.map(objects, _.property('a.b'));
 * // => [2, 1]
 *
 * _.map(_.sortBy(objects, _.property(['a', 'b'])), 'a.b');
 * // => [1, 2]
 */
function property(path) {
  return (0,_isKey/* default */.A)(path) ? (0,_baseProperty/* default */.A)((0,_toKey/* default */.A)(path)) : _basePropertyDeep(path);
}

/* harmony default export */ const lodash_es_property = (property);

;// ./node_modules/lodash-es/_baseIteratee.js






/**
 * The base implementation of `_.iteratee`.
 *
 * @private
 * @param {*} [value=_.identity] The value to convert to an iteratee.
 * @returns {Function} Returns the iteratee.
 */
function baseIteratee(value) {
  // Don't store the `typeof` result in a variable to avoid a JIT bug in Safari 9.
  // See https://bugs.webkit.org/show_bug.cgi?id=156034 for more details.
  if (typeof value == 'function') {
    return value;
  }
  if (value == null) {
    return identity/* default */.A;
  }
  if (typeof value == 'object') {
    return (0,isArray/* default */.A)(value)
      ? _baseMatchesProperty(value[0], value[1])
      : _baseMatches(value);
  }
  return lodash_es_property(value);
}

/* harmony default export */ const _baseIteratee = (baseIteratee);


/***/ },

/***/ 69471
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseKeys)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_isPrototype.js
var _isPrototype = __webpack_require__(97271);
// EXTERNAL MODULE: ./node_modules/lodash-es/_overArg.js
var _overArg = __webpack_require__(40367);
;// ./node_modules/lodash-es/_nativeKeys.js


/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeKeys = (0,_overArg/* default */.A)(Object.keys, Object);

/* harmony default export */ const _nativeKeys = (nativeKeys);

;// ./node_modules/lodash-es/_baseKeys.js



/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _baseKeys_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * The base implementation of `_.keys` which doesn't treat sparse arrays as dense.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */
function baseKeys(object) {
  if (!(0,_isPrototype/* default */.A)(object)) {
    return _nativeKeys(object);
  }
  var result = [];
  for (var key in Object(object)) {
    if (_baseKeys_hasOwnProperty.call(object, key) && key != 'constructor') {
      result.push(key);
    }
  }
  return result;
}

/* harmony default export */ const _baseKeys = (baseKeys);


/***/ },

/***/ 70805
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * The base implementation of `_.property` without support for deep paths.
 *
 * @private
 * @param {string} key The key of the property to get.
 * @returns {Function} Returns the new accessor function.
 */
function baseProperty(key) {
  return function(object) {
    return object == null ? undefined : object[key];
  };
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseProperty);


/***/ },

/***/ 24326
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _identity_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(29008);
/* harmony import */ var _overRest_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(76875);
/* harmony import */ var _setToString_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(67525);




/**
 * The base implementation of `_.rest` which doesn't validate or coerce arguments.
 *
 * @private
 * @param {Function} func The function to apply a rest parameter to.
 * @param {number} [start=func.length-1] The start position of the rest parameter.
 * @returns {Function} Returns the new function.
 */
function baseRest(func, start) {
  return (0,_setToString_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A)((0,_overRest_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(func, start, _identity_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A), func + '');
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseRest);


/***/ },

/***/ 52789
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * The base implementation of `_.unary` without support for storing metadata.
 *
 * @private
 * @param {Function} func The function to cap arguments for.
 * @returns {Function} Returns the new capped function.
 */
function baseUnary(func) {
  return function(value) {
    return func(value);
  };
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseUnary);


/***/ },

/***/ 64099
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Checks if a `cache` value for `key` exists.
 *
 * @private
 * @param {Object} cache The cache to query.
 * @param {string} key The key of the entry to check.
 * @returns {boolean} Returns `true` if an entry for `key` exists, else `false`.
 */
function cacheHas(cache, key) {
  return cache.has(key);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (cacheHas);


/***/ },

/***/ 99922
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _identity_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(29008);


/**
 * Casts `value` to `identity` if it's not a function.
 *
 * @private
 * @param {*} value The value to inspect.
 * @returns {Function} Returns cast function.
 */
function castFunction(value) {
  return typeof value == 'function' ? value : _identity_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (castFunction);


/***/ },

/***/ 63442
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _castPath)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isKey.js
var _isKey = __webpack_require__(86586);
// EXTERNAL MODULE: ./node_modules/lodash-es/_MapCache.js + 14 modules
var _MapCache = __webpack_require__(29471);
;// ./node_modules/lodash-es/memoize.js


/** Error message constants. */
var FUNC_ERROR_TEXT = 'Expected a function';

/**
 * Creates a function that memoizes the result of `func`. If `resolver` is
 * provided, it determines the cache key for storing the result based on the
 * arguments provided to the memoized function. By default, the first argument
 * provided to the memoized function is used as the map cache key. The `func`
 * is invoked with the `this` binding of the memoized function.
 *
 * **Note:** The cache is exposed as the `cache` property on the memoized
 * function. Its creation may be customized by replacing the `_.memoize.Cache`
 * constructor with one whose instances implement the
 * [`Map`](http://ecma-international.org/ecma-262/7.0/#sec-properties-of-the-map-prototype-object)
 * method interface of `clear`, `delete`, `get`, `has`, and `set`.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Function
 * @param {Function} func The function to have its output memoized.
 * @param {Function} [resolver] The function to resolve the cache key.
 * @returns {Function} Returns the new memoized function.
 * @example
 *
 * var object = { 'a': 1, 'b': 2 };
 * var other = { 'c': 3, 'd': 4 };
 *
 * var values = _.memoize(_.values);
 * values(object);
 * // => [1, 2]
 *
 * values(other);
 * // => [3, 4]
 *
 * object.a = 2;
 * values(object);
 * // => [1, 2]
 *
 * // Modify the result cache.
 * values.cache.set(object, ['a', 'b']);
 * values(object);
 * // => ['a', 'b']
 *
 * // Replace `_.memoize.Cache`.
 * _.memoize.Cache = WeakMap;
 */
function memoize(func, resolver) {
  if (typeof func != 'function' || (resolver != null && typeof resolver != 'function')) {
    throw new TypeError(FUNC_ERROR_TEXT);
  }
  var memoized = function() {
    var args = arguments,
        key = resolver ? resolver.apply(this, args) : args[0],
        cache = memoized.cache;

    if (cache.has(key)) {
      return cache.get(key);
    }
    var result = func.apply(this, args);
    memoized.cache = cache.set(key, result) || cache;
    return result;
  };
  memoized.cache = new (memoize.Cache || _MapCache/* default */.A);
  return memoized;
}

// Expose `MapCache`.
memoize.Cache = _MapCache/* default */.A;

/* harmony default export */ const lodash_es_memoize = (memoize);

;// ./node_modules/lodash-es/_memoizeCapped.js


/** Used as the maximum memoize cache size. */
var MAX_MEMOIZE_SIZE = 500;

/**
 * A specialized version of `_.memoize` which clears the memoized function's
 * cache when it exceeds `MAX_MEMOIZE_SIZE`.
 *
 * @private
 * @param {Function} func The function to have its output memoized.
 * @returns {Function} Returns the new memoized function.
 */
function memoizeCapped(func) {
  var result = lodash_es_memoize(func, function(key) {
    if (cache.size === MAX_MEMOIZE_SIZE) {
      cache.clear();
    }
    return key;
  });

  var cache = result.cache;
  return result;
}

/* harmony default export */ const _memoizeCapped = (memoizeCapped);

;// ./node_modules/lodash-es/_stringToPath.js


/** Used to match property names within property paths. */
var rePropName = /[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g;

/** Used to match backslashes in property paths. */
var reEscapeChar = /\\(\\)?/g;

/**
 * Converts `string` to a property path array.
 *
 * @private
 * @param {string} string The string to convert.
 * @returns {Array} Returns the property path array.
 */
var stringToPath = _memoizeCapped(function(string) {
  var result = [];
  if (string.charCodeAt(0) === 46 /* . */) {
    result.push('');
  }
  string.replace(rePropName, function(match, number, quote, subString) {
    result.push(quote ? subString.replace(reEscapeChar, '$1') : (number || match));
  });
  return result;
});

/* harmony default export */ const _stringToPath = (stringToPath);

// EXTERNAL MODULE: ./node_modules/lodash-es/toString.js + 1 modules
var lodash_es_toString = __webpack_require__(28894);
;// ./node_modules/lodash-es/_castPath.js





/**
 * Casts `value` to a path array if it's not one.
 *
 * @private
 * @param {*} value The value to inspect.
 * @param {Object} [object] The object to query keys on.
 * @returns {Array} Returns the cast property path array.
 */
function castPath(value, object) {
  if ((0,isArray/* default */.A)(value)) {
    return value;
  }
  return (0,_isKey/* default */.A)(value, object) ? [value] : _stringToPath((0,lodash_es_toString/* default */.A)(value));
}

/* harmony default export */ const _castPath = (castPath);


/***/ },

/***/ 84171
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _getNative_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(18744);


var defineProperty = (function() {
  try {
    var func = (0,_getNative_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(Object, 'defineProperty');
    func({}, '', {});
    return func;
  } catch (e) {}
}());

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (defineProperty);


/***/ },

/***/ 72136
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/** Detect free variable `global` from Node.js. */
var freeGlobal = typeof global == 'object' && global && global.Object === Object && global;

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (freeGlobal);


/***/ },

/***/ 19042
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseGetAllKeys_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(33831);
/* harmony import */ var _getSymbols_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(14792);
/* harmony import */ var _keys_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(27422);




/**
 * Creates an array of own enumerable property names and symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names and symbols.
 */
function getAllKeys(object) {
  return (0,_baseGetAllKeys_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object, _keys_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A, _getSymbols_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (getAllKeys);


/***/ },

/***/ 18744
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _getNative)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/isFunction.js
var isFunction = __webpack_require__(89610);
// EXTERNAL MODULE: ./node_modules/lodash-es/_root.js
var _root = __webpack_require__(41917);
;// ./node_modules/lodash-es/_coreJsData.js


/** Used to detect overreaching core-js shims. */
var coreJsData = _root/* default */.A['__core-js_shared__'];

/* harmony default export */ const _coreJsData = (coreJsData);

;// ./node_modules/lodash-es/_isMasked.js


/** Used to detect methods masquerading as native. */
var maskSrcKey = (function() {
  var uid = /[^.]+$/.exec(_coreJsData && _coreJsData.keys && _coreJsData.keys.IE_PROTO || '');
  return uid ? ('Symbol(src)_1.' + uid) : '';
}());

/**
 * Checks if `func` has its source masked.
 *
 * @private
 * @param {Function} func The function to check.
 * @returns {boolean} Returns `true` if `func` is masked, else `false`.
 */
function isMasked(func) {
  return !!maskSrcKey && (maskSrcKey in func);
}

/* harmony default export */ const _isMasked = (isMasked);

// EXTERNAL MODULE: ./node_modules/lodash-es/isObject.js
var isObject = __webpack_require__(23149);
// EXTERNAL MODULE: ./node_modules/lodash-es/_toSource.js
var _toSource = __webpack_require__(81121);
;// ./node_modules/lodash-es/_baseIsNative.js





/**
 * Used to match `RegExp`
 * [syntax characters](http://ecma-international.org/ecma-262/7.0/#sec-patterns).
 */
var reRegExpChar = /[\\^$.*+?()[\]{}|]/g;

/** Used to detect host constructors (Safari). */
var reIsHostCtor = /^\[object .+?Constructor\]$/;

/** Used for built-in method references. */
var funcProto = Function.prototype,
    objectProto = Object.prototype;

/** Used to resolve the decompiled source of functions. */
var funcToString = funcProto.toString;

/** Used to check objects for own properties. */
var _baseIsNative_hasOwnProperty = objectProto.hasOwnProperty;

/** Used to detect if a method is native. */
var reIsNative = RegExp('^' +
  funcToString.call(_baseIsNative_hasOwnProperty).replace(reRegExpChar, '\\$&')
  .replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, '$1.*?') + '$'
);

/**
 * The base implementation of `_.isNative` without bad shim checks.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a native function,
 *  else `false`.
 */
function baseIsNative(value) {
  if (!(0,isObject/* default */.A)(value) || _isMasked(value)) {
    return false;
  }
  var pattern = (0,isFunction/* default */.A)(value) ? reIsNative : reIsHostCtor;
  return pattern.test((0,_toSource/* default */.A)(value));
}

/* harmony default export */ const _baseIsNative = (baseIsNative);

;// ./node_modules/lodash-es/_getValue.js
/**
 * Gets the value at `key` of `object`.
 *
 * @private
 * @param {Object} [object] The object to query.
 * @param {string} key The key of the property to get.
 * @returns {*} Returns the property value.
 */
function getValue(object, key) {
  return object == null ? undefined : object[key];
}

/* harmony default export */ const _getValue = (getValue);

;// ./node_modules/lodash-es/_getNative.js



/**
 * Gets the native function at `key` of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {string} key The key of the method to get.
 * @returns {*} Returns the function if it's native, else `undefined`.
 */
function getNative(object, key) {
  var value = _getValue(object, key);
  return _baseIsNative(value) ? value : undefined;
}

/* harmony default export */ const _getNative = (getNative);


/***/ },

/***/ 14792
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _arrayFilter_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(2634);
/* harmony import */ var _stubArray_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(13153);



/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Built-in value references. */
var propertyIsEnumerable = objectProto.propertyIsEnumerable;

/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeGetSymbols = Object.getOwnPropertySymbols;

/**
 * Creates an array of the own enumerable symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of symbols.
 */
var getSymbols = !nativeGetSymbols ? _stubArray_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A : function(object) {
  if (object == null) {
    return [];
  }
  object = Object(object);
  return (0,_arrayFilter_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(nativeGetSymbols(object), function(symbol) {
    return propertyIsEnumerable.call(object, symbol);
  });
};

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (getSymbols);


/***/ },

/***/ 9779
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _getTag)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_getNative.js + 4 modules
var _getNative = __webpack_require__(18744);
// EXTERNAL MODULE: ./node_modules/lodash-es/_root.js
var _root = __webpack_require__(41917);
;// ./node_modules/lodash-es/_DataView.js



/* Built-in method references that are verified to be native. */
var DataView = (0,_getNative/* default */.A)(_root/* default */.A, 'DataView');

/* harmony default export */ const _DataView = (DataView);

// EXTERNAL MODULE: ./node_modules/lodash-es/_Map.js
var _Map = __webpack_require__(68335);
;// ./node_modules/lodash-es/_Promise.js



/* Built-in method references that are verified to be native. */
var Promise = (0,_getNative/* default */.A)(_root/* default */.A, 'Promise');

/* harmony default export */ const _Promise = (Promise);

// EXTERNAL MODULE: ./node_modules/lodash-es/_Set.js
var _Set = __webpack_require__(39857);
;// ./node_modules/lodash-es/_WeakMap.js



/* Built-in method references that are verified to be native. */
var WeakMap = (0,_getNative/* default */.A)(_root/* default */.A, 'WeakMap');

/* harmony default export */ const _WeakMap = (WeakMap);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseGetTag.js + 2 modules
var _baseGetTag = __webpack_require__(88496);
// EXTERNAL MODULE: ./node_modules/lodash-es/_toSource.js
var _toSource = __webpack_require__(81121);
;// ./node_modules/lodash-es/_getTag.js








/** `Object#toString` result references. */
var mapTag = '[object Map]',
    objectTag = '[object Object]',
    promiseTag = '[object Promise]',
    setTag = '[object Set]',
    weakMapTag = '[object WeakMap]';

var dataViewTag = '[object DataView]';

/** Used to detect maps, sets, and weakmaps. */
var dataViewCtorString = (0,_toSource/* default */.A)(_DataView),
    mapCtorString = (0,_toSource/* default */.A)(_Map/* default */.A),
    promiseCtorString = (0,_toSource/* default */.A)(_Promise),
    setCtorString = (0,_toSource/* default */.A)(_Set/* default */.A),
    weakMapCtorString = (0,_toSource/* default */.A)(_WeakMap);

/**
 * Gets the `toStringTag` of `value`.
 *
 * @private
 * @param {*} value The value to query.
 * @returns {string} Returns the `toStringTag`.
 */
var getTag = _baseGetTag/* default */.A;

// Fallback for data views, maps, sets, and weak maps in IE 11 and promises in Node.js < 6.
if ((_DataView && getTag(new _DataView(new ArrayBuffer(1))) != dataViewTag) ||
    (_Map/* default */.A && getTag(new _Map/* default */.A) != mapTag) ||
    (_Promise && getTag(_Promise.resolve()) != promiseTag) ||
    (_Set/* default */.A && getTag(new _Set/* default */.A) != setTag) ||
    (_WeakMap && getTag(new _WeakMap) != weakMapTag)) {
  getTag = function(value) {
    var result = (0,_baseGetTag/* default */.A)(value),
        Ctor = result == objectTag ? value.constructor : undefined,
        ctorString = Ctor ? (0,_toSource/* default */.A)(Ctor) : '';

    if (ctorString) {
      switch (ctorString) {
        case dataViewCtorString: return dataViewTag;
        case mapCtorString: return mapTag;
        case promiseCtorString: return promiseTag;
        case setCtorString: return setTag;
        case weakMapCtorString: return weakMapTag;
      }
    }
    return result;
  };
}

/* harmony default export */ const _getTag = (getTag);


/***/ },

/***/ 85054
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _castPath_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(63442);
/* harmony import */ var _isArguments_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(29893);
/* harmony import */ var _isArray_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(92049);
/* harmony import */ var _isIndex_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(25353);
/* harmony import */ var _isLength_js__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(5254);
/* harmony import */ var _toKey_js__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(30901);







/**
 * Checks if `path` exists on `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array|string} path The path to check.
 * @param {Function} hasFunc The function to check properties.
 * @returns {boolean} Returns `true` if `path` exists, else `false`.
 */
function hasPath(object, path, hasFunc) {
  path = (0,_castPath_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(path, object);

  var index = -1,
      length = path.length,
      result = false;

  while (++index < length) {
    var key = (0,_toKey_js__WEBPACK_IMPORTED_MODULE_5__/* ["default"] */ .A)(path[index]);
    if (!(result = object != null && hasFunc(object, key))) {
      break;
    }
    object = object[key];
  }
  if (result || ++index != length) {
    return result;
  }
  length = object == null ? 0 : object.length;
  return !!length && (0,_isLength_js__WEBPACK_IMPORTED_MODULE_4__/* ["default"] */ .A)(length) && (0,_isIndex_js__WEBPACK_IMPORTED_MODULE_3__/* ["default"] */ .A)(key, length) &&
    ((0,_isArray_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A)(object) || (0,_isArguments_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(object));
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (hasPath);


/***/ },

/***/ 25353
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/** Used as references for various `Number` constants. */
var MAX_SAFE_INTEGER = 9007199254740991;

/** Used to detect unsigned integer values. */
var reIsUint = /^(?:0|[1-9]\d*)$/;

/**
 * Checks if `value` is a valid array-like index.
 *
 * @private
 * @param {*} value The value to check.
 * @param {number} [length=MAX_SAFE_INTEGER] The upper bounds of a valid index.
 * @returns {boolean} Returns `true` if `value` is a valid index, else `false`.
 */
function isIndex(value, length) {
  var type = typeof value;
  length = length == null ? MAX_SAFE_INTEGER : length;

  return !!length &&
    (type == 'number' ||
      (type != 'symbol' && reIsUint.test(value))) &&
        (value > -1 && value % 1 == 0 && value < length);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isIndex);


/***/ },

/***/ 86586
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _isArray_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(92049);
/* harmony import */ var _isSymbol_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(61882);



/** Used to match property names within property paths. */
var reIsDeepProp = /\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/,
    reIsPlainProp = /^\w*$/;

/**
 * Checks if `value` is a property name and not a property path.
 *
 * @private
 * @param {*} value The value to check.
 * @param {Object} [object] The object to query keys on.
 * @returns {boolean} Returns `true` if `value` is a property name, else `false`.
 */
function isKey(value, object) {
  if ((0,_isArray_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value)) {
    return false;
  }
  var type = typeof value;
  if (type == 'number' || type == 'symbol' || type == 'boolean' ||
      value == null || (0,_isSymbol_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(value)) {
    return true;
  }
  return reIsPlainProp.test(value) || !reIsDeepProp.test(value) ||
    (object != null && value in Object(object));
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isKey);


/***/ },

/***/ 97271
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/** Used for built-in method references. */
var objectProto = Object.prototype;

/**
 * Checks if `value` is likely a prototype object.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a prototype, else `false`.
 */
function isPrototype(value) {
  var Ctor = value && value.constructor,
      proto = (typeof Ctor == 'function' && Ctor.prototype) || objectProto;

  return value === proto;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isPrototype);


/***/ },

/***/ 64841
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _freeGlobal_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(72136);


/** Detect free variable `exports`. */
var freeExports = typeof exports == 'object' && exports && !exports.nodeType && exports;

/** Detect free variable `module`. */
var freeModule = freeExports && typeof module == 'object' && module && !module.nodeType && module;

/** Detect the popular CommonJS extension `module.exports`. */
var moduleExports = freeModule && freeModule.exports === freeExports;

/** Detect free variable `process` from Node.js. */
var freeProcess = moduleExports && _freeGlobal_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A.process;

/** Used to access faster Node.js helpers. */
var nodeUtil = (function() {
  try {
    // Use `util.types` for Node.js 10+.
    var types = freeModule && freeModule.require && freeModule.require('util').types;

    if (types) {
      return types;
    }

    // Legacy `process.binding('util')` for Node.js < 10.
    return freeProcess && freeProcess.binding && freeProcess.binding('util');
  } catch (e) {}
}());

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (nodeUtil);


/***/ },

/***/ 40367
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Creates a unary function that invokes `func` with its argument transformed.
 *
 * @private
 * @param {Function} func The function to wrap.
 * @param {Function} transform The argument transform.
 * @returns {Function} Returns the new function.
 */
function overArg(func, transform) {
  return function(arg) {
    return func(transform(arg));
  };
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (overArg);


/***/ },

/***/ 76875
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _overRest)
});

;// ./node_modules/lodash-es/_apply.js
/**
 * A faster alternative to `Function#apply`, this function invokes `func`
 * with the `this` binding of `thisArg` and the arguments of `args`.
 *
 * @private
 * @param {Function} func The function to invoke.
 * @param {*} thisArg The `this` binding of `func`.
 * @param {Array} args The arguments to invoke `func` with.
 * @returns {*} Returns the result of `func`.
 */
function apply(func, thisArg, args) {
  switch (args.length) {
    case 0: return func.call(thisArg);
    case 1: return func.call(thisArg, args[0]);
    case 2: return func.call(thisArg, args[0], args[1]);
    case 3: return func.call(thisArg, args[0], args[1], args[2]);
  }
  return func.apply(thisArg, args);
}

/* harmony default export */ const _apply = (apply);

;// ./node_modules/lodash-es/_overRest.js


/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeMax = Math.max;

/**
 * A specialized version of `baseRest` which transforms the rest array.
 *
 * @private
 * @param {Function} func The function to apply a rest parameter to.
 * @param {number} [start=func.length-1] The start position of the rest parameter.
 * @param {Function} transform The rest array transform.
 * @returns {Function} Returns the new function.
 */
function overRest(func, start, transform) {
  start = nativeMax(start === undefined ? (func.length - 1) : start, 0);
  return function() {
    var args = arguments,
        index = -1,
        length = nativeMax(args.length - start, 0),
        array = Array(length);

    while (++index < length) {
      array[index] = args[start + index];
    }
    index = -1;
    var otherArgs = Array(start + 1);
    while (++index < start) {
      otherArgs[index] = args[index];
    }
    otherArgs[start] = transform(array);
    return _apply(func, this, otherArgs);
  };
}

/* harmony default export */ const _overRest = (overRest);


/***/ },

/***/ 41917
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _freeGlobal_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(72136);


/** Detect free variable `self`. */
var freeSelf = typeof self == 'object' && self && self.Object === Object && self;

/** Used as a reference to the global object. */
var root = _freeGlobal_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A || freeSelf || Function('return this')();

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (root);


/***/ },

/***/ 29959
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Converts `set` to an array of its values.
 *
 * @private
 * @param {Object} set The set to convert.
 * @returns {Array} Returns the values.
 */
function setToArray(set) {
  var index = -1,
      result = Array(set.size);

  set.forEach(function(value) {
    result[++index] = value;
  });
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (setToArray);


/***/ },

/***/ 67525
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _setToString)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/constant.js
var constant = __webpack_require__(39142);
// EXTERNAL MODULE: ./node_modules/lodash-es/_defineProperty.js
var _defineProperty = __webpack_require__(84171);
// EXTERNAL MODULE: ./node_modules/lodash-es/identity.js
var identity = __webpack_require__(29008);
;// ./node_modules/lodash-es/_baseSetToString.js




/**
 * The base implementation of `setToString` without support for hot loop shorting.
 *
 * @private
 * @param {Function} func The function to modify.
 * @param {Function} string The `toString` result.
 * @returns {Function} Returns `func`.
 */
var baseSetToString = !_defineProperty/* default */.A ? identity/* default */.A : function(func, string) {
  return (0,_defineProperty/* default */.A)(func, 'toString', {
    'configurable': true,
    'enumerable': false,
    'value': (0,constant/* default */.A)(string),
    'writable': true
  });
};

/* harmony default export */ const _baseSetToString = (baseSetToString);

;// ./node_modules/lodash-es/_shortOut.js
/** Used to detect hot functions by number of calls within a span of milliseconds. */
var HOT_COUNT = 800,
    HOT_SPAN = 16;

/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeNow = Date.now;

/**
 * Creates a function that'll short out and invoke `identity` instead
 * of `func` when it's called `HOT_COUNT` or more times in `HOT_SPAN`
 * milliseconds.
 *
 * @private
 * @param {Function} func The function to restrict.
 * @returns {Function} Returns the new shortable function.
 */
function shortOut(func) {
  var count = 0,
      lastCalled = 0;

  return function() {
    var stamp = nativeNow(),
        remaining = HOT_SPAN - (stamp - lastCalled);

    lastCalled = stamp;
    if (remaining > 0) {
      if (++count >= HOT_COUNT) {
        return arguments[0];
      }
    } else {
      count = 0;
    }
    return func.apply(undefined, arguments);
  };
}

/* harmony default export */ const _shortOut = (shortOut);

;// ./node_modules/lodash-es/_setToString.js



/**
 * Sets the `toString` method of `func` to return `string`.
 *
 * @private
 * @param {Function} func The function to modify.
 * @param {Function} string The `toString` result.
 * @returns {Function} Returns `func`.
 */
var setToString = _shortOut(_baseSetToString);

/* harmony default export */ const _setToString = (setToString);


/***/ },

/***/ 30901
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _isSymbol_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(61882);


/** Used as references for various `Number` constants. */
var INFINITY = 1 / 0;

/**
 * Converts `value` to a string key if it's not a string or symbol.
 *
 * @private
 * @param {*} value The value to inspect.
 * @returns {string|symbol} Returns the key.
 */
function toKey(value) {
  if (typeof value == 'string' || (0,_isSymbol_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value)) {
    return value;
  }
  var result = (value + '');
  return (result == '0' && (1 / value) == -INFINITY) ? '-0' : result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (toKey);


/***/ },

/***/ 81121
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/** Used for built-in method references. */
var funcProto = Function.prototype;

/** Used to resolve the decompiled source of functions. */
var funcToString = funcProto.toString;

/**
 * Converts `func` to its source code.
 *
 * @private
 * @param {Function} func The function to convert.
 * @returns {string} Returns the source code.
 */
function toSource(func) {
  if (func != null) {
    try {
      return funcToString.call(func);
    } catch (e) {}
    try {
      return (func + '');
    } catch (e) {}
  }
  return '';
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (toSource);


/***/ },

/***/ 39142
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Creates a function that returns `value`.
 *
 * @static
 * @memberOf _
 * @since 2.4.0
 * @category Util
 * @param {*} value The value to return from the new function.
 * @returns {Function} Returns the new constant function.
 * @example
 *
 * var objects = _.times(2, _.constant({ 'a': 1 }));
 *
 * console.log(objects);
 * // => [{ 'a': 1 }, { 'a': 1 }]
 *
 * console.log(objects[0] === objects[1]);
 * // => true
 */
function constant(value) {
  return function() {
    return value;
  };
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (constant);


/***/ },

/***/ 66984
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Performs a
 * [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * comparison between two values to determine if they are equivalent.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to compare.
 * @param {*} other The other value to compare.
 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
 * @example
 *
 * var object = { 'a': 1 };
 * var other = { 'a': 1 };
 *
 * _.eq(object, object);
 * // => true
 *
 * _.eq(object, other);
 * // => false
 *
 * _.eq('a', 'a');
 * // => true
 *
 * _.eq('a', Object('a'));
 * // => false
 *
 * _.eq(NaN, NaN);
 * // => true
 */
function eq(value, other) {
  return value === other || (value !== value && other !== other);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (eq);


/***/ },

/***/ 11662
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_filter)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayFilter.js
var _arrayFilter = __webpack_require__(2634);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseEach.js + 1 modules
var _baseEach = __webpack_require__(6240);
;// ./node_modules/lodash-es/_baseFilter.js


/**
 * The base implementation of `_.filter` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} predicate The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 */
function baseFilter(collection, predicate) {
  var result = [];
  (0,_baseEach/* default */.A)(collection, function(value, index, collection) {
    if (predicate(value, index, collection)) {
      result.push(value);
    }
  });
  return result;
}

/* harmony default export */ const _baseFilter = (baseFilter);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseIteratee.js + 16 modules
var _baseIteratee = __webpack_require__(49574);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
;// ./node_modules/lodash-es/filter.js





/**
 * Iterates over elements of `collection`, returning an array of all elements
 * `predicate` returns truthy for. The predicate is invoked with three
 * arguments: (value, index|key, collection).
 *
 * **Note:** Unlike `_.remove`, this method returns a new array.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [predicate=_.identity] The function invoked per iteration.
 * @returns {Array} Returns the new filtered array.
 * @see _.reject
 * @example
 *
 * var users = [
 *   { 'user': 'barney', 'age': 36, 'active': true },
 *   { 'user': 'fred',   'age': 40, 'active': false }
 * ];
 *
 * _.filter(users, function(o) { return !o.active; });
 * // => objects for ['fred']
 *
 * // The `_.matches` iteratee shorthand.
 * _.filter(users, { 'age': 36, 'active': true });
 * // => objects for ['barney']
 *
 * // The `_.matchesProperty` iteratee shorthand.
 * _.filter(users, ['active', false]);
 * // => objects for ['fred']
 *
 * // The `_.property` iteratee shorthand.
 * _.filter(users, 'active');
 * // => objects for ['barney']
 *
 * // Combining several predicates using `_.overEvery` or `_.overSome`.
 * _.filter(users, _.overSome([{ 'age': 36 }, ['age', 40]]));
 * // => objects for ['fred', 'barney']
 */
function filter(collection, predicate) {
  var func = (0,isArray/* default */.A)(collection) ? _arrayFilter/* default */.A : _baseFilter;
  return func(collection, (0,_baseIteratee/* default */.A)(predicate, 3));
}

/* harmony default export */ const lodash_es_filter = (filter);


/***/ },

/***/ 8058
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _arrayEach_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(72641);
/* harmony import */ var _baseEach_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(6240);
/* harmony import */ var _castFunction_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(99922);
/* harmony import */ var _isArray_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(92049);





/**
 * Iterates over elements of `collection` and invokes `iteratee` for each element.
 * The iteratee is invoked with three arguments: (value, index|key, collection).
 * Iteratee functions may exit iteration early by explicitly returning `false`.
 *
 * **Note:** As with other "Collections" methods, objects with a "length"
 * property are iterated like arrays. To avoid this behavior use `_.forIn`
 * or `_.forOwn` for object iteration.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @alias each
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Array|Object} Returns `collection`.
 * @see _.forEachRight
 * @example
 *
 * _.forEach([1, 2], function(value) {
 *   console.log(value);
 * });
 * // => Logs `1` then `2`.
 *
 * _.forEach({ 'a': 1, 'b': 2 }, function(value, key) {
 *   console.log(key);
 * });
 * // => Logs 'a' then 'b' (iteration order is not guaranteed).
 */
function forEach(collection, iteratee) {
  var func = (0,_isArray_js__WEBPACK_IMPORTED_MODULE_3__/* ["default"] */ .A)(collection) ? _arrayEach_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A : _baseEach_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A;
  return func(collection, (0,_castFunction_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A)(iteratee));
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (forEach);


/***/ },

/***/ 39188
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_hasIn)
});

;// ./node_modules/lodash-es/_baseHasIn.js
/**
 * The base implementation of `_.hasIn` without support for deep paths.
 *
 * @private
 * @param {Object} [object] The object to query.
 * @param {Array|string} key The key to check.
 * @returns {boolean} Returns `true` if `key` exists, else `false`.
 */
function baseHasIn(object, key) {
  return object != null && key in Object(object);
}

/* harmony default export */ const _baseHasIn = (baseHasIn);

// EXTERNAL MODULE: ./node_modules/lodash-es/_hasPath.js
var _hasPath = __webpack_require__(85054);
;// ./node_modules/lodash-es/hasIn.js



/**
 * Checks if `path` is a direct or inherited property of `object`.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Object
 * @param {Object} object The object to query.
 * @param {Array|string} path The path to check.
 * @returns {boolean} Returns `true` if `path` exists, else `false`.
 * @example
 *
 * var object = _.create({ 'a': _.create({ 'b': 2 }) });
 *
 * _.hasIn(object, 'a');
 * // => true
 *
 * _.hasIn(object, 'a.b');
 * // => true
 *
 * _.hasIn(object, ['a', 'b']);
 * // => true
 *
 * _.hasIn(object, 'b');
 * // => false
 */
function hasIn(object, path) {
  return object != null && (0,_hasPath/* default */.A)(object, path, _baseHasIn);
}

/* harmony default export */ const lodash_es_hasIn = (hasIn);


/***/ },

/***/ 29008
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * This method returns the first argument it receives.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Util
 * @param {*} value Any value.
 * @returns {*} Returns `value`.
 * @example
 *
 * var object = { 'a': 1 };
 *
 * console.log(_.identity(object) === object);
 * // => true
 */
function identity(value) {
  return value;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (identity);


/***/ },

/***/ 29893
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_isArguments)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseGetTag.js + 2 modules
var _baseGetTag = __webpack_require__(88496);
// EXTERNAL MODULE: ./node_modules/lodash-es/isObjectLike.js
var isObjectLike = __webpack_require__(53098);
;// ./node_modules/lodash-es/_baseIsArguments.js



/** `Object#toString` result references. */
var argsTag = '[object Arguments]';

/**
 * The base implementation of `_.isArguments`.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an `arguments` object,
 */
function baseIsArguments(value) {
  return (0,isObjectLike/* default */.A)(value) && (0,_baseGetTag/* default */.A)(value) == argsTag;
}

/* harmony default export */ const _baseIsArguments = (baseIsArguments);

;// ./node_modules/lodash-es/isArguments.js



/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var isArguments_hasOwnProperty = objectProto.hasOwnProperty;

/** Built-in value references. */
var propertyIsEnumerable = objectProto.propertyIsEnumerable;

/**
 * Checks if `value` is likely an `arguments` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an `arguments` object,
 *  else `false`.
 * @example
 *
 * _.isArguments(function() { return arguments; }());
 * // => true
 *
 * _.isArguments([1, 2, 3]);
 * // => false
 */
var isArguments = _baseIsArguments(function() { return arguments; }()) ? _baseIsArguments : function(value) {
  return (0,isObjectLike/* default */.A)(value) && isArguments_hasOwnProperty.call(value, 'callee') &&
    !propertyIsEnumerable.call(value, 'callee');
};

/* harmony default export */ const lodash_es_isArguments = (isArguments);


/***/ },

/***/ 92049
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Checks if `value` is classified as an `Array` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an array, else `false`.
 * @example
 *
 * _.isArray([1, 2, 3]);
 * // => true
 *
 * _.isArray(document.body.children);
 * // => false
 *
 * _.isArray('abc');
 * // => false
 *
 * _.isArray(_.noop);
 * // => false
 */
var isArray = Array.isArray;

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isArray);


/***/ },

/***/ 38446
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _isFunction_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(89610);
/* harmony import */ var _isLength_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(5254);



/**
 * Checks if `value` is array-like. A value is considered array-like if it's
 * not a function and has a `value.length` that's an integer greater than or
 * equal to `0` and less than or equal to `Number.MAX_SAFE_INTEGER`.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is array-like, else `false`.
 * @example
 *
 * _.isArrayLike([1, 2, 3]);
 * // => true
 *
 * _.isArrayLike(document.body.children);
 * // => true
 *
 * _.isArrayLike('abc');
 * // => true
 *
 * _.isArrayLike(_.noop);
 * // => false
 */
function isArrayLike(value) {
  return value != null && (0,_isLength_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(value.length) && !(0,_isFunction_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isArrayLike);


/***/ },

/***/ 53533
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _isArrayLike_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(38446);
/* harmony import */ var _isObjectLike_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(53098);



/**
 * This method is like `_.isArrayLike` except that it also checks if `value`
 * is an object.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an array-like object,
 *  else `false`.
 * @example
 *
 * _.isArrayLikeObject([1, 2, 3]);
 * // => true
 *
 * _.isArrayLikeObject(document.body.children);
 * // => true
 *
 * _.isArrayLikeObject('abc');
 * // => false
 *
 * _.isArrayLikeObject(_.noop);
 * // => false
 */
function isArrayLikeObject(value) {
  return (0,_isObjectLike_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(value) && (0,_isArrayLike_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isArrayLikeObject);


/***/ },

/***/ 99912
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_isBuffer)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_root.js
var _root = __webpack_require__(41917);
;// ./node_modules/lodash-es/stubFalse.js
/**
 * This method returns `false`.
 *
 * @static
 * @memberOf _
 * @since 4.13.0
 * @category Util
 * @returns {boolean} Returns `false`.
 * @example
 *
 * _.times(2, _.stubFalse);
 * // => [false, false]
 */
function stubFalse() {
  return false;
}

/* harmony default export */ const lodash_es_stubFalse = (stubFalse);

;// ./node_modules/lodash-es/isBuffer.js



/** Detect free variable `exports`. */
var freeExports = typeof exports == 'object' && exports && !exports.nodeType && exports;

/** Detect free variable `module`. */
var freeModule = freeExports && typeof module == 'object' && module && !module.nodeType && module;

/** Detect the popular CommonJS extension `module.exports`. */
var moduleExports = freeModule && freeModule.exports === freeExports;

/** Built-in value references. */
var Buffer = moduleExports ? _root/* default */.A.Buffer : undefined;

/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeIsBuffer = Buffer ? Buffer.isBuffer : undefined;

/**
 * Checks if `value` is a buffer.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a buffer, else `false`.
 * @example
 *
 * _.isBuffer(new Buffer(2));
 * // => true
 *
 * _.isBuffer(new Uint8Array(2));
 * // => false
 */
var isBuffer = nativeIsBuffer || lodash_es_stubFalse;

/* harmony default export */ const lodash_es_isBuffer = (isBuffer);


/***/ },

/***/ 89610
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseGetTag_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(88496);
/* harmony import */ var _isObject_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(23149);



/** `Object#toString` result references. */
var asyncTag = '[object AsyncFunction]',
    funcTag = '[object Function]',
    genTag = '[object GeneratorFunction]',
    proxyTag = '[object Proxy]';

/**
 * Checks if `value` is classified as a `Function` object.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a function, else `false`.
 * @example
 *
 * _.isFunction(_);
 * // => true
 *
 * _.isFunction(/abc/);
 * // => false
 */
function isFunction(value) {
  if (!(0,_isObject_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(value)) {
    return false;
  }
  // The use of `Object#toString` avoids issues with the `typeof` operator
  // in Safari 9 which returns 'object' for typed arrays and other constructors.
  var tag = (0,_baseGetTag_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value);
  return tag == funcTag || tag == genTag || tag == asyncTag || tag == proxyTag;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isFunction);


/***/ },

/***/ 5254
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/** Used as references for various `Number` constants. */
var MAX_SAFE_INTEGER = 9007199254740991;

/**
 * Checks if `value` is a valid array-like length.
 *
 * **Note:** This method is loosely based on
 * [`ToLength`](http://ecma-international.org/ecma-262/7.0/#sec-tolength).
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a valid length, else `false`.
 * @example
 *
 * _.isLength(3);
 * // => true
 *
 * _.isLength(Number.MIN_VALUE);
 * // => false
 *
 * _.isLength(Infinity);
 * // => false
 *
 * _.isLength('3');
 * // => false
 */
function isLength(value) {
  return typeof value == 'number' &&
    value > -1 && value % 1 == 0 && value <= MAX_SAFE_INTEGER;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isLength);


/***/ },

/***/ 23149
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Checks if `value` is the
 * [language type](http://www.ecma-international.org/ecma-262/7.0/#sec-ecmascript-language-types)
 * of `Object`. (e.g. arrays, functions, objects, regexes, `new Number(0)`, and `new String('')`)
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is an object, else `false`.
 * @example
 *
 * _.isObject({});
 * // => true
 *
 * _.isObject([1, 2, 3]);
 * // => true
 *
 * _.isObject(_.noop);
 * // => true
 *
 * _.isObject(null);
 * // => false
 */
function isObject(value) {
  var type = typeof value;
  return value != null && (type == 'object' || type == 'function');
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isObject);


/***/ },

/***/ 53098
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Checks if `value` is object-like. A value is object-like if it's not `null`
 * and has a `typeof` result of "object".
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is object-like, else `false`.
 * @example
 *
 * _.isObjectLike({});
 * // => true
 *
 * _.isObjectLike([1, 2, 3]);
 * // => true
 *
 * _.isObjectLike(_.noop);
 * // => false
 *
 * _.isObjectLike(null);
 * // => false
 */
function isObjectLike(value) {
  return value != null && typeof value == 'object';
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isObjectLike);


/***/ },

/***/ 61882
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseGetTag_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(88496);
/* harmony import */ var _isObjectLike_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(53098);



/** `Object#toString` result references. */
var symbolTag = '[object Symbol]';

/**
 * Checks if `value` is classified as a `Symbol` primitive or object.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a symbol, else `false`.
 * @example
 *
 * _.isSymbol(Symbol.iterator);
 * // => true
 *
 * _.isSymbol('abc');
 * // => false
 */
function isSymbol(value) {
  return typeof value == 'symbol' ||
    ((0,_isObjectLike_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(value) && (0,_baseGetTag_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(value) == symbolTag);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isSymbol);


/***/ },

/***/ 33858
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_isTypedArray)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseGetTag.js + 2 modules
var _baseGetTag = __webpack_require__(88496);
// EXTERNAL MODULE: ./node_modules/lodash-es/isLength.js
var isLength = __webpack_require__(5254);
// EXTERNAL MODULE: ./node_modules/lodash-es/isObjectLike.js
var isObjectLike = __webpack_require__(53098);
;// ./node_modules/lodash-es/_baseIsTypedArray.js




/** `Object#toString` result references. */
var argsTag = '[object Arguments]',
    arrayTag = '[object Array]',
    boolTag = '[object Boolean]',
    dateTag = '[object Date]',
    errorTag = '[object Error]',
    funcTag = '[object Function]',
    mapTag = '[object Map]',
    numberTag = '[object Number]',
    objectTag = '[object Object]',
    regexpTag = '[object RegExp]',
    setTag = '[object Set]',
    stringTag = '[object String]',
    weakMapTag = '[object WeakMap]';

var arrayBufferTag = '[object ArrayBuffer]',
    dataViewTag = '[object DataView]',
    float32Tag = '[object Float32Array]',
    float64Tag = '[object Float64Array]',
    int8Tag = '[object Int8Array]',
    int16Tag = '[object Int16Array]',
    int32Tag = '[object Int32Array]',
    uint8Tag = '[object Uint8Array]',
    uint8ClampedTag = '[object Uint8ClampedArray]',
    uint16Tag = '[object Uint16Array]',
    uint32Tag = '[object Uint32Array]';

/** Used to identify `toStringTag` values of typed arrays. */
var typedArrayTags = {};
typedArrayTags[float32Tag] = typedArrayTags[float64Tag] =
typedArrayTags[int8Tag] = typedArrayTags[int16Tag] =
typedArrayTags[int32Tag] = typedArrayTags[uint8Tag] =
typedArrayTags[uint8ClampedTag] = typedArrayTags[uint16Tag] =
typedArrayTags[uint32Tag] = true;
typedArrayTags[argsTag] = typedArrayTags[arrayTag] =
typedArrayTags[arrayBufferTag] = typedArrayTags[boolTag] =
typedArrayTags[dataViewTag] = typedArrayTags[dateTag] =
typedArrayTags[errorTag] = typedArrayTags[funcTag] =
typedArrayTags[mapTag] = typedArrayTags[numberTag] =
typedArrayTags[objectTag] = typedArrayTags[regexpTag] =
typedArrayTags[setTag] = typedArrayTags[stringTag] =
typedArrayTags[weakMapTag] = false;

/**
 * The base implementation of `_.isTypedArray` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a typed array, else `false`.
 */
function baseIsTypedArray(value) {
  return (0,isObjectLike/* default */.A)(value) &&
    (0,isLength/* default */.A)(value.length) && !!typedArrayTags[(0,_baseGetTag/* default */.A)(value)];
}

/* harmony default export */ const _baseIsTypedArray = (baseIsTypedArray);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseUnary.js
var _baseUnary = __webpack_require__(52789);
// EXTERNAL MODULE: ./node_modules/lodash-es/_nodeUtil.js
var _nodeUtil = __webpack_require__(64841);
;// ./node_modules/lodash-es/isTypedArray.js




/* Node.js helper references. */
var nodeIsTypedArray = _nodeUtil/* default */.A && _nodeUtil/* default */.A.isTypedArray;

/**
 * Checks if `value` is classified as a typed array.
 *
 * @static
 * @memberOf _
 * @since 3.0.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a typed array, else `false`.
 * @example
 *
 * _.isTypedArray(new Uint8Array);
 * // => true
 *
 * _.isTypedArray([]);
 * // => false
 */
var isTypedArray = nodeIsTypedArray ? (0,_baseUnary/* default */.A)(nodeIsTypedArray) : _baseIsTypedArray;

/* harmony default export */ const lodash_es_isTypedArray = (isTypedArray);


/***/ },

/***/ 69592
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Checks if `value` is `undefined`.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is `undefined`, else `false`.
 * @example
 *
 * _.isUndefined(void 0);
 * // => true
 *
 * _.isUndefined(null);
 * // => false
 */
function isUndefined(value) {
  return value === undefined;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (isUndefined);


/***/ },

/***/ 27422
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _arrayLikeKeys_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(83607);
/* harmony import */ var _baseKeys_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(69471);
/* harmony import */ var _isArrayLike_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(38446);




/**
 * Creates an array of the own enumerable property names of `object`.
 *
 * **Note:** Non-object values are coerced to objects. See the
 * [ES spec](http://ecma-international.org/ecma-262/7.0/#sec-object.keys)
 * for more details.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.keys(new Foo);
 * // => ['a', 'b'] (iteration order is not guaranteed)
 *
 * _.keys('hi');
 * // => ['0', '1']
 */
function keys(object) {
  return (0,_isArrayLike_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A)(object) ? (0,_arrayLikeKeys_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object) : (0,_baseKeys_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(object);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (keys);


/***/ },

/***/ 89463
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_reduce)
});

;// ./node_modules/lodash-es/_arrayReduce.js
/**
 * A specialized version of `_.reduce` for arrays without support for
 * iteratee shorthands.
 *
 * @private
 * @param {Array} [array] The array to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {*} [accumulator] The initial value.
 * @param {boolean} [initAccum] Specify using the first element of `array` as
 *  the initial value.
 * @returns {*} Returns the accumulated value.
 */
function arrayReduce(array, iteratee, accumulator, initAccum) {
  var index = -1,
      length = array == null ? 0 : array.length;

  if (initAccum && length) {
    accumulator = array[++index];
  }
  while (++index < length) {
    accumulator = iteratee(accumulator, array[index], index, array);
  }
  return accumulator;
}

/* harmony default export */ const _arrayReduce = (arrayReduce);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseEach.js + 1 modules
var _baseEach = __webpack_require__(6240);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseIteratee.js + 16 modules
var _baseIteratee = __webpack_require__(49574);
;// ./node_modules/lodash-es/_baseReduce.js
/**
 * The base implementation of `_.reduce` and `_.reduceRight`, without support
 * for iteratee shorthands, which iterates over `collection` using `eachFunc`.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @param {*} accumulator The initial value.
 * @param {boolean} initAccum Specify using the first or last element of
 *  `collection` as the initial value.
 * @param {Function} eachFunc The function to iterate over `collection`.
 * @returns {*} Returns the accumulated value.
 */
function baseReduce(collection, iteratee, accumulator, initAccum, eachFunc) {
  eachFunc(collection, function(value, index, collection) {
    accumulator = initAccum
      ? (initAccum = false, value)
      : iteratee(accumulator, value, index, collection);
  });
  return accumulator;
}

/* harmony default export */ const _baseReduce = (baseReduce);

// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
;// ./node_modules/lodash-es/reduce.js






/**
 * Reduces `collection` to a value which is the accumulated result of running
 * each element in `collection` thru `iteratee`, where each successive
 * invocation is supplied the return value of the previous. If `accumulator`
 * is not given, the first element of `collection` is used as the initial
 * value. The iteratee is invoked with four arguments:
 * (accumulator, value, index|key, collection).
 *
 * Many lodash methods are guarded to work as iteratees for methods like
 * `_.reduce`, `_.reduceRight`, and `_.transform`.
 *
 * The guarded methods are:
 * `assign`, `defaults`, `defaultsDeep`, `includes`, `merge`, `orderBy`,
 * and `sortBy`
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @param {*} [accumulator] The initial value.
 * @returns {*} Returns the accumulated value.
 * @see _.reduceRight
 * @example
 *
 * _.reduce([1, 2], function(sum, n) {
 *   return sum + n;
 * }, 0);
 * // => 3
 *
 * _.reduce({ 'a': 1, 'b': 2, 'c': 1 }, function(result, value, key) {
 *   (result[value] || (result[value] = [])).push(key);
 *   return result;
 * }, {});
 * // => { '1': ['a', 'c'], '2': ['b'] } (iteration order is not guaranteed)
 */
function reduce(collection, iteratee, accumulator) {
  var func = (0,isArray/* default */.A)(collection) ? _arrayReduce : _baseReduce,
      initAccum = arguments.length < 3;

  return func(collection, (0,_baseIteratee/* default */.A)(iteratee, 4), accumulator, initAccum, _baseEach/* default */.A);
}

/* harmony default export */ const lodash_es_reduce = (reduce);


/***/ },

/***/ 13153
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * This method returns a new empty array.
 *
 * @static
 * @memberOf _
 * @since 4.13.0
 * @category Util
 * @returns {Array} Returns the new empty array.
 * @example
 *
 * var arrays = _.times(2, _.stubArray);
 *
 * console.log(arrays);
 * // => [[], []]
 *
 * console.log(arrays[0] === arrays[1]);
 * // => false
 */
function stubArray() {
  return [];
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (stubArray);


/***/ },

/***/ 28894
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_toString)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_Symbol.js
var _Symbol = __webpack_require__(241);
// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayMap.js
var _arrayMap = __webpack_require__(45572);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/isSymbol.js
var isSymbol = __webpack_require__(61882);
;// ./node_modules/lodash-es/_baseToString.js





/** Used as references for various `Number` constants. */
var INFINITY = 1 / 0;

/** Used to convert symbols to primitives and strings. */
var symbolProto = _Symbol/* default */.A ? _Symbol/* default */.A.prototype : undefined,
    symbolToString = symbolProto ? symbolProto.toString : undefined;

/**
 * The base implementation of `_.toString` which doesn't convert nullish
 * values to empty strings.
 *
 * @private
 * @param {*} value The value to process.
 * @returns {string} Returns the string.
 */
function baseToString(value) {
  // Exit early for strings to avoid a performance hit in some environments.
  if (typeof value == 'string') {
    return value;
  }
  if ((0,isArray/* default */.A)(value)) {
    // Recursively convert values (susceptible to call stack limits).
    return (0,_arrayMap/* default */.A)(value, baseToString) + '';
  }
  if ((0,isSymbol/* default */.A)(value)) {
    return symbolToString ? symbolToString.call(value) : '';
  }
  var result = (value + '');
  return (result == '0' && (1 / value) == -INFINITY) ? '-0' : result;
}

/* harmony default export */ const _baseToString = (baseToString);

;// ./node_modules/lodash-es/toString.js


/**
 * Converts `value` to a string. An empty string is returned for `null`
 * and `undefined` values. The sign of `-0` is preserved.
 *
 * @static
 * @memberOf _
 * @since 4.0.0
 * @category Lang
 * @param {*} value The value to convert.
 * @returns {string} Returns the converted string.
 * @example
 *
 * _.toString(null);
 * // => ''
 *
 * _.toString(-0);
 * // => '-0'
 *
 * _.toString([1, 2, 3]);
 * // => '1,2,3'
 */
function toString_toString(value) {
  return value == null ? '' : _baseToString(value);
}

/* harmony default export */ const lodash_es_toString = (toString_toString);


/***/ },

/***/ 38207
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_values)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayMap.js
var _arrayMap = __webpack_require__(45572);
;// ./node_modules/lodash-es/_baseValues.js


/**
 * The base implementation of `_.values` and `_.valuesIn` which creates an
 * array of `object` property values corresponding to the property names
 * of `props`.
 *
 * @private
 * @param {Object} object The object to query.
 * @param {Array} props The property names to get values for.
 * @returns {Object} Returns the array of property values.
 */
function baseValues(object, props) {
  return (0,_arrayMap/* default */.A)(props, function(key) {
    return object[key];
  });
}

/* harmony default export */ const _baseValues = (baseValues);

// EXTERNAL MODULE: ./node_modules/lodash-es/keys.js
var keys = __webpack_require__(27422);
;// ./node_modules/lodash-es/values.js



/**
 * Creates an array of the own enumerable string keyed property values of `object`.
 *
 * **Note:** Non-object values are coerced to objects.
 *
 * @static
 * @since 0.1.0
 * @memberOf _
 * @category Object
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property values.
 * @example
 *
 * function Foo() {
 *   this.a = 1;
 *   this.b = 2;
 * }
 *
 * Foo.prototype.c = 3;
 *
 * _.values(new Foo);
 * // => [1, 2] (iteration order is not guaranteed)
 *
 * _.values('hi');
 * // => ['h', 'i']
 */
function values(object) {
  return object == null ? [] : _baseValues(object, (0,keys/* default */.A)(object));
}

/* harmony default export */ const lodash_es_values = (values);


/***/ },

/***/ 20384
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  diagram: () => (/* binding */ diagram)
});

// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-FMBD7UC4.mjs
var chunk_FMBD7UC4 = __webpack_require__(52501);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-BSJP7CBP.mjs
var chunk_BSJP7CBP = __webpack_require__(11334);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-L5ZTLDWV.mjs
var chunk_L5ZTLDWV = __webpack_require__(25105);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-O5CBEL6O.mjs + 13 modules
var chunk_O5CBEL6O = __webpack_require__(35739);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-5ZQYHXKU.mjs + 13 modules
var chunk_5ZQYHXKU = __webpack_require__(48515);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-CSCIHK7Q.mjs + 3 modules
var chunk_CSCIHK7Q = __webpack_require__(73706);
// EXTERNAL MODULE: ./node_modules/mermaid/dist/chunks/mermaid.core/chunk-AGHRB4JF.mjs
var chunk_AGHRB4JF = __webpack_require__(40797);
// EXTERNAL MODULE: ./node_modules/es-toolkit/dist/predicate/isPrimitive.mjs
var isPrimitive = __webpack_require__(37110);
// EXTERNAL MODULE: ./node_modules/es-toolkit/dist/compat/_internal/getTag.mjs
var getTag = __webpack_require__(99125);
// EXTERNAL MODULE: ./node_modules/es-toolkit/dist/compat/_internal/tags.mjs
var tags = __webpack_require__(89826);
;// ./node_modules/es-toolkit/dist/compat/predicate/isArray.mjs
function isArray(value) {
    return Array.isArray(value);
}



// EXTERNAL MODULE: ./node_modules/es-toolkit/dist/compat/predicate/isTypedArray.mjs
var isTypedArray = __webpack_require__(19663);
;// ./node_modules/es-toolkit/dist/compat/object/clone.mjs






function clone(obj) {
    if ((0,isPrimitive/* isPrimitive */.s)(obj)) {
        return obj;
    }
    const tag = (0,getTag/* getTag */.b)(obj);
    if (!isCloneableObject(obj)) {
        return {};
    }
    if (isArray(obj)) {
        const result = Array.from(obj);
        if (obj.length > 0 && typeof obj[0] === 'string' && Object.hasOwn(obj, 'index')) {
            result.index = obj.index;
            result.input = obj.input;
        }
        return result;
    }
    if ((0,isTypedArray/* isTypedArray */.i)(obj)) {
        const typedArray = obj;
        const Ctor = typedArray.constructor;
        return new Ctor(typedArray.buffer, typedArray.byteOffset, typedArray.length);
    }
    if (tag === tags/* arrayBufferTag */.cT) {
        return new ArrayBuffer(obj.byteLength);
    }
    if (tag === tags/* dataViewTag */.iq) {
        const dataView = obj;
        const buffer = dataView.buffer;
        const byteOffset = dataView.byteOffset;
        const byteLength = dataView.byteLength;
        const clonedBuffer = new ArrayBuffer(byteLength);
        const srcView = new Uint8Array(buffer, byteOffset, byteLength);
        const destView = new Uint8Array(clonedBuffer);
        destView.set(srcView);
        return new DataView(clonedBuffer);
    }
    if (tag === tags/* booleanTag */.$V || tag === tags/* numberTag */.kj || tag === tags/* stringTag */.OG) {
        const Ctor = obj.constructor;
        const clone = new Ctor(obj.valueOf());
        if (tag === tags/* stringTag */.OG) {
            cloneStringObjectProperties(clone, obj);
        }
        else {
            copyOwnProperties(clone, obj);
        }
        return clone;
    }
    if (tag === tags/* dateTag */.vC) {
        return new Date(Number(obj));
    }
    if (tag === tags/* regexpTag */.Av) {
        const regExp = obj;
        const clone = new RegExp(regExp.source, regExp.flags);
        clone.lastIndex = regExp.lastIndex;
        return clone;
    }
    if (tag === tags/* symbolTag */.VP) {
        return Object(Symbol.prototype.valueOf.call(obj));
    }
    if (tag === tags/* mapTag */.pj) {
        const map = obj;
        const result = new Map();
        map.forEach((obj, key) => {
            result.set(key, obj);
        });
        return result;
    }
    if (tag === tags/* setTag */.NA) {
        const set = obj;
        const result = new Set();
        set.forEach(obj => {
            result.add(obj);
        });
        return result;
    }
    if (tag === tags/* argumentsTag */.R_) {
        const args = obj;
        const result = {};
        copyOwnProperties(result, args);
        result.length = args.length;
        result[Symbol.iterator] = args[Symbol.iterator];
        return result;
    }
    const result = {};
    copyPrototype(result, obj);
    copyOwnProperties(result, obj);
    copySymbolProperties(result, obj);
    return result;
}
function isCloneableObject(object) {
    switch ((0,getTag/* getTag */.b)(object)) {
        case tags/* argumentsTag */.R_:
        case tags/* arrayTag */.Uw:
        case tags/* arrayBufferTag */.cT:
        case tags/* dataViewTag */.iq:
        case tags/* booleanTag */.$V:
        case tags/* dateTag */.vC:
        case tags/* float32ArrayTag */.ri:
        case tags/* float64ArrayTag */.ML:
        case tags/* int8ArrayTag */.XZ:
        case tags/* int16ArrayTag */.i1:
        case tags/* int32ArrayTag */._u:
        case tags/* mapTag */.pj:
        case tags/* numberTag */.kj:
        case tags/* objectTag */.GX:
        case tags/* regexpTag */.Av:
        case tags/* setTag */.NA:
        case tags/* stringTag */.OG:
        case tags/* symbolTag */.VP:
        case tags/* uint8ArrayTag */.Qb:
        case tags/* uint8ClampedArrayTag */.q:
        case tags/* uint16ArrayTag */.x6:
        case tags/* uint32ArrayTag */.ZR: {
            return true;
        }
        default: {
            return false;
        }
    }
}
function copyOwnProperties(target, source) {
    for (const key in source) {
        if (Object.hasOwn(source, key)) {
            target[key] = source[key];
        }
    }
}
function copySymbolProperties(target, source) {
    const symbols = Object.getOwnPropertySymbols(source);
    for (let i = 0; i < symbols.length; i++) {
        const symbol = symbols[i];
        if (Object.prototype.propertyIsEnumerable.call(source, symbol)) {
            target[symbol] = source[symbol];
        }
    }
}
function cloneStringObjectProperties(target, source) {
    const stringLength = source.valueOf().length;
    for (const key in source) {
        if (Object.hasOwn(source, key) && (Number.isNaN(Number(key)) || Number(key) >= stringLength)) {
            target[key] = source[key];
        }
    }
}
function copyPrototype(target, source) {
    const proto = Object.getPrototypeOf(source);
    if (proto !== null) {
        const Ctor = source.constructor;
        if (typeof Ctor === 'function') {
            Object.setPrototypeOf(target, proto);
        }
    }
}



// EXTERNAL MODULE: ./node_modules/khroma/dist/methods/channel.js
var channel = __webpack_require__(75937);
// EXTERNAL MODULE: ./node_modules/khroma/dist/methods/rgba.js
var rgba = __webpack_require__(25582);
// EXTERNAL MODULE: ./node_modules/d3/src/index.js + 216 modules
var src = __webpack_require__(70451);
// EXTERNAL MODULE: ./node_modules/dagre-d3-es/src/graphlib/index.js
var graphlib = __webpack_require__(697);
;// ./node_modules/mermaid/dist/chunks/mermaid.core/blockDiagram-GPEHLZMM.mjs








// src/diagrams/block/parser/block.jison
var parser = (function() {
  var o = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(k, v, o2, l) {
    for (o2 = o2 || {}, l = k.length; l--; o2[k[l]] = v) ;
    return o2;
  }, "o"), $V0 = [1, 15], $V1 = [1, 7], $V2 = [1, 13], $V3 = [1, 14], $V4 = [1, 19], $V5 = [1, 16], $V6 = [1, 17], $V7 = [1, 18], $V8 = [8, 30], $V9 = [8, 10, 21, 28, 29, 30, 31, 39, 43, 46], $Va = [1, 23], $Vb = [1, 24], $Vc = [8, 10, 15, 16, 21, 28, 29, 30, 31, 39, 43, 46], $Vd = [8, 10, 15, 16, 21, 27, 28, 29, 30, 31, 39, 43, 46], $Ve = [1, 49];
  var parser2 = {
    trace: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function trace() {
    }, "trace"),
    yy: {},
    symbols_: { "error": 2, "spaceLines": 3, "SPACELINE": 4, "NL": 5, "separator": 6, "SPACE": 7, "EOF": 8, "start": 9, "BLOCK_DIAGRAM_KEY": 10, "document": 11, "stop": 12, "statement": 13, "link": 14, "LINK": 15, "START_LINK": 16, "LINK_LABEL": 17, "STR": 18, "nodeStatement": 19, "columnsStatement": 20, "SPACE_BLOCK": 21, "blockStatement": 22, "classDefStatement": 23, "cssClassStatement": 24, "styleStatement": 25, "node": 26, "SIZE": 27, "COLUMNS": 28, "id-block": 29, "end": 30, "NODE_ID": 31, "nodeShapeNLabel": 32, "dirList": 33, "DIR": 34, "NODE_DSTART": 35, "NODE_DEND": 36, "BLOCK_ARROW_START": 37, "BLOCK_ARROW_END": 38, "classDef": 39, "CLASSDEF_ID": 40, "CLASSDEF_STYLEOPTS": 41, "DEFAULT": 42, "class": 43, "CLASSENTITY_IDS": 44, "STYLECLASS": 45, "style": 46, "STYLE_ENTITY_IDS": 47, "STYLE_DEFINITION_DATA": 48, "$accept": 0, "$end": 1 },
    terminals_: { 2: "error", 4: "SPACELINE", 5: "NL", 7: "SPACE", 8: "EOF", 10: "BLOCK_DIAGRAM_KEY", 15: "LINK", 16: "START_LINK", 17: "LINK_LABEL", 18: "STR", 21: "SPACE_BLOCK", 27: "SIZE", 28: "COLUMNS", 29: "id-block", 30: "end", 31: "NODE_ID", 34: "DIR", 35: "NODE_DSTART", 36: "NODE_DEND", 37: "BLOCK_ARROW_START", 38: "BLOCK_ARROW_END", 39: "classDef", 40: "CLASSDEF_ID", 41: "CLASSDEF_STYLEOPTS", 42: "DEFAULT", 43: "class", 44: "CLASSENTITY_IDS", 45: "STYLECLASS", 46: "style", 47: "STYLE_ENTITY_IDS", 48: "STYLE_DEFINITION_DATA" },
    productions_: [0, [3, 1], [3, 2], [3, 2], [6, 1], [6, 1], [6, 1], [9, 3], [12, 1], [12, 1], [12, 2], [12, 2], [11, 1], [11, 2], [14, 1], [14, 4], [13, 1], [13, 1], [13, 1], [13, 1], [13, 1], [13, 1], [13, 1], [19, 3], [19, 2], [19, 1], [20, 1], [22, 4], [22, 3], [26, 1], [26, 2], [33, 1], [33, 2], [32, 3], [32, 4], [23, 3], [23, 3], [24, 3], [25, 3]],
    performAction: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function anonymous(yytext, yyleng, yylineno, yy, yystate, $$, _$) {
      var $0 = $$.length - 1;
      switch (yystate) {
        case 4:
          yy.getLogger().debug("Rule: separator (NL) ");
          break;
        case 5:
          yy.getLogger().debug("Rule: separator (Space) ");
          break;
        case 6:
          yy.getLogger().debug("Rule: separator (EOF) ");
          break;
        case 7:
          yy.getLogger().debug("Rule: hierarchy: ", $$[$0 - 1]);
          yy.setHierarchy($$[$0 - 1]);
          break;
        case 8:
          yy.getLogger().debug("Stop NL ");
          break;
        case 9:
          yy.getLogger().debug("Stop EOF ");
          break;
        case 10:
          yy.getLogger().debug("Stop NL2 ");
          break;
        case 11:
          yy.getLogger().debug("Stop EOF2 ");
          break;
        case 12:
          yy.getLogger().debug("Rule: statement: ", $$[$0]);
          typeof $$[$0].length === "number" ? this.$ = $$[$0] : this.$ = [$$[$0]];
          break;
        case 13:
          yy.getLogger().debug("Rule: statement #2: ", $$[$0 - 1]);
          this.$ = [$$[$0 - 1]].concat($$[$0]);
          break;
        case 14:
          yy.getLogger().debug("Rule: link: ", $$[$0], yytext);
          this.$ = { edgeTypeStr: $$[$0], label: "" };
          break;
        case 15:
          yy.getLogger().debug("Rule: LABEL link: ", $$[$0 - 3], $$[$0 - 1], $$[$0]);
          this.$ = { edgeTypeStr: $$[$0], label: $$[$0 - 1] };
          break;
        case 18:
          const num = parseInt($$[$0]);
          const spaceId = yy.generateId();
          this.$ = { id: spaceId, type: "space", label: "", width: num, children: [] };
          break;
        case 23:
          yy.getLogger().debug("Rule: (nodeStatement link node) ", $$[$0 - 2], $$[$0 - 1], $$[$0], " typestr: ", $$[$0 - 1].edgeTypeStr);
          const edgeData = yy.edgeStrToEdgeData($$[$0 - 1].edgeTypeStr);
          const startEdgeData = yy.edgeStrToEdgeStartData($$[$0 - 1].edgeTypeStr);
          const lineThickness = yy.edgeStrToThickness($$[$0 - 1].edgeTypeStr);
          const linePattern = yy.edgeStrToPattern($$[$0 - 1].edgeTypeStr);
          this.$ = [
            { id: $$[$0 - 2].id, label: $$[$0 - 2].label, type: $$[$0 - 2].type, directions: $$[$0 - 2].directions },
            { id: $$[$0 - 2].id + "-" + $$[$0].id, start: $$[$0 - 2].id, end: $$[$0].id, label: $$[$0 - 1].label, type: "edge", thickness: lineThickness, pattern: linePattern, directions: $$[$0].directions, arrowTypeEnd: edgeData, arrowTypeStart: startEdgeData },
            { id: $$[$0].id, label: $$[$0].label, type: yy.typeStr2Type($$[$0].typeStr), directions: $$[$0].directions }
          ];
          break;
        case 24:
          yy.getLogger().debug("Rule: nodeStatement (abc88 node size) ", $$[$0 - 1], $$[$0]);
          this.$ = { id: $$[$0 - 1].id, label: $$[$0 - 1].label, type: yy.typeStr2Type($$[$0 - 1].typeStr), directions: $$[$0 - 1].directions, widthInColumns: parseInt($$[$0], 10) };
          break;
        case 25:
          yy.getLogger().debug("Rule: nodeStatement (node) ", $$[$0]);
          this.$ = { id: $$[$0].id, label: $$[$0].label, type: yy.typeStr2Type($$[$0].typeStr), directions: $$[$0].directions, widthInColumns: 1 };
          break;
        case 26:
          yy.getLogger().debug("APA123", this ? this : "na");
          yy.getLogger().debug("COLUMNS: ", $$[$0]);
          this.$ = { type: "column-setting", columns: $$[$0] === "auto" ? -1 : parseInt($$[$0]) };
          break;
        case 27:
          yy.getLogger().debug("Rule: id-block statement : ", $$[$0 - 2], $$[$0 - 1]);
          const id2 = yy.generateId();
          this.$ = { ...$$[$0 - 2], type: "composite", children: $$[$0 - 1] };
          break;
        case 28:
          yy.getLogger().debug("Rule: blockStatement : ", $$[$0 - 2], $$[$0 - 1], $$[$0]);
          const id = yy.generateId();
          this.$ = { id, type: "composite", label: "", children: $$[$0 - 1] };
          break;
        case 29:
          yy.getLogger().debug("Rule: node (NODE_ID separator): ", $$[$0]);
          this.$ = { id: $$[$0] };
          break;
        case 30:
          yy.getLogger().debug("Rule: node (NODE_ID nodeShapeNLabel separator): ", $$[$0 - 1], $$[$0]);
          this.$ = { id: $$[$0 - 1], label: $$[$0].label, typeStr: $$[$0].typeStr, directions: $$[$0].directions };
          break;
        case 31:
          yy.getLogger().debug("Rule: dirList: ", $$[$0]);
          this.$ = [$$[$0]];
          break;
        case 32:
          yy.getLogger().debug("Rule: dirList: ", $$[$0 - 1], $$[$0]);
          this.$ = [$$[$0 - 1]].concat($$[$0]);
          break;
        case 33:
          yy.getLogger().debug("Rule: nodeShapeNLabel: ", $$[$0 - 2], $$[$0 - 1], $$[$0]);
          this.$ = { typeStr: $$[$0 - 2] + $$[$0], label: $$[$0 - 1] };
          break;
        case 34:
          yy.getLogger().debug("Rule: BLOCK_ARROW nodeShapeNLabel: ", $$[$0 - 3], $$[$0 - 2], " #3:", $$[$0 - 1], $$[$0]);
          this.$ = { typeStr: $$[$0 - 3] + $$[$0], label: $$[$0 - 2], directions: $$[$0 - 1] };
          break;
        case 35:
        case 36:
          this.$ = { type: "classDef", id: $$[$0 - 1].trim(), css: $$[$0].trim() };
          break;
        case 37:
          this.$ = { type: "applyClass", id: $$[$0 - 1].trim(), styleClass: $$[$0].trim() };
          break;
        case 38:
          this.$ = { type: "applyStyles", id: $$[$0 - 1].trim(), stylesStr: $$[$0].trim() };
          break;
      }
    }, "anonymous"),
    table: [{ 9: 1, 10: [1, 2] }, { 1: [3] }, { 10: $V0, 11: 3, 13: 4, 19: 5, 20: 6, 21: $V1, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 28: $V2, 29: $V3, 31: $V4, 39: $V5, 43: $V6, 46: $V7 }, { 8: [1, 20] }, o($V8, [2, 12], { 13: 4, 19: 5, 20: 6, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 11: 21, 10: $V0, 21: $V1, 28: $V2, 29: $V3, 31: $V4, 39: $V5, 43: $V6, 46: $V7 }), o($V9, [2, 16], { 14: 22, 15: $Va, 16: $Vb }), o($V9, [2, 17]), o($V9, [2, 18]), o($V9, [2, 19]), o($V9, [2, 20]), o($V9, [2, 21]), o($V9, [2, 22]), o($Vc, [2, 25], { 27: [1, 25] }), o($V9, [2, 26]), { 19: 26, 26: 12, 31: $V4 }, { 10: $V0, 11: 27, 13: 4, 19: 5, 20: 6, 21: $V1, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 28: $V2, 29: $V3, 31: $V4, 39: $V5, 43: $V6, 46: $V7 }, { 40: [1, 28], 42: [1, 29] }, { 44: [1, 30] }, { 47: [1, 31] }, o($Vd, [2, 29], { 32: 32, 35: [1, 33], 37: [1, 34] }), { 1: [2, 7] }, o($V8, [2, 13]), { 26: 35, 31: $V4 }, { 31: [2, 14] }, { 17: [1, 36] }, o($Vc, [2, 24]), { 10: $V0, 11: 37, 13: 4, 14: 22, 15: $Va, 16: $Vb, 19: 5, 20: 6, 21: $V1, 22: 8, 23: 9, 24: 10, 25: 11, 26: 12, 28: $V2, 29: $V3, 31: $V4, 39: $V5, 43: $V6, 46: $V7 }, { 30: [1, 38] }, { 41: [1, 39] }, { 41: [1, 40] }, { 45: [1, 41] }, { 48: [1, 42] }, o($Vd, [2, 30]), { 18: [1, 43] }, { 18: [1, 44] }, o($Vc, [2, 23]), { 18: [1, 45] }, { 30: [1, 46] }, o($V9, [2, 28]), o($V9, [2, 35]), o($V9, [2, 36]), o($V9, [2, 37]), o($V9, [2, 38]), { 36: [1, 47] }, { 33: 48, 34: $Ve }, { 15: [1, 50] }, o($V9, [2, 27]), o($Vd, [2, 33]), { 38: [1, 51] }, { 33: 52, 34: $Ve, 38: [2, 31] }, { 31: [2, 15] }, o($Vd, [2, 34]), { 38: [2, 32] }],
    defaultActions: { 20: [2, 7], 23: [2, 14], 50: [2, 15], 52: [2, 32] },
    parseError: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function parseError(str, hash) {
      if (hash.recoverable) {
        this.trace(str);
      } else {
        var error = new Error(str);
        error.hash = hash;
        throw error;
      }
    }, "parseError"),
    parse: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function parse(input) {
      var self = this, stack = [0], tstack = [], vstack = [null], lstack = [], table = this.table, yytext = "", yylineno = 0, yyleng = 0, recovering = 0, TERROR = 2, EOF = 1;
      var args = lstack.slice.call(arguments, 1);
      var lexer2 = Object.create(this.lexer);
      var sharedState = { yy: {} };
      for (var k in this.yy) {
        if (Object.prototype.hasOwnProperty.call(this.yy, k)) {
          sharedState.yy[k] = this.yy[k];
        }
      }
      lexer2.setInput(input, sharedState.yy);
      sharedState.yy.lexer = lexer2;
      sharedState.yy.parser = this;
      if (typeof lexer2.yylloc == "undefined") {
        lexer2.yylloc = {};
      }
      var yyloc = lexer2.yylloc;
      lstack.push(yyloc);
      var ranges = lexer2.options && lexer2.options.ranges;
      if (typeof sharedState.yy.parseError === "function") {
        this.parseError = sharedState.yy.parseError;
      } else {
        this.parseError = Object.getPrototypeOf(this).parseError;
      }
      function popStack(n) {
        stack.length = stack.length - 2 * n;
        vstack.length = vstack.length - n;
        lstack.length = lstack.length - n;
      }
      (0,chunk_AGHRB4JF/* __name */.K2)(popStack, "popStack");
      function lex() {
        var token;
        token = tstack.pop() || lexer2.lex() || EOF;
        if (typeof token !== "number") {
          if (token instanceof Array) {
            tstack = token;
            token = tstack.pop();
          }
          token = self.symbols_[token] || token;
        }
        return token;
      }
      (0,chunk_AGHRB4JF/* __name */.K2)(lex, "lex");
      var symbol, preErrorSymbol, state, action, a, r, yyval = {}, p, len, newState, expected;
      while (true) {
        state = stack[stack.length - 1];
        if (this.defaultActions[state]) {
          action = this.defaultActions[state];
        } else {
          if (symbol === null || typeof symbol == "undefined") {
            symbol = lex();
          }
          action = table[state] && table[state][symbol];
        }
        if (typeof action === "undefined" || !action.length || !action[0]) {
          var errStr = "";
          expected = [];
          for (p in table[state]) {
            if (this.terminals_[p] && p > TERROR) {
              expected.push("'" + this.terminals_[p] + "'");
            }
          }
          if (lexer2.showPosition) {
            errStr = "Parse error on line " + (yylineno + 1) + ":\n" + lexer2.showPosition() + "\nExpecting " + expected.join(", ") + ", got '" + (this.terminals_[symbol] || symbol) + "'";
          } else {
            errStr = "Parse error on line " + (yylineno + 1) + ": Unexpected " + (symbol == EOF ? "end of input" : "'" + (this.terminals_[symbol] || symbol) + "'");
          }
          this.parseError(errStr, {
            text: lexer2.match,
            token: this.terminals_[symbol] || symbol,
            line: lexer2.yylineno,
            loc: yyloc,
            expected
          });
        }
        if (action[0] instanceof Array && action.length > 1) {
          throw new Error("Parse Error: multiple actions possible at state: " + state + ", token: " + symbol);
        }
        switch (action[0]) {
          case 1:
            stack.push(symbol);
            vstack.push(lexer2.yytext);
            lstack.push(lexer2.yylloc);
            stack.push(action[1]);
            symbol = null;
            if (!preErrorSymbol) {
              yyleng = lexer2.yyleng;
              yytext = lexer2.yytext;
              yylineno = lexer2.yylineno;
              yyloc = lexer2.yylloc;
              if (recovering > 0) {
                recovering--;
              }
            } else {
              symbol = preErrorSymbol;
              preErrorSymbol = null;
            }
            break;
          case 2:
            len = this.productions_[action[1]][1];
            yyval.$ = vstack[vstack.length - len];
            yyval._$ = {
              first_line: lstack[lstack.length - (len || 1)].first_line,
              last_line: lstack[lstack.length - 1].last_line,
              first_column: lstack[lstack.length - (len || 1)].first_column,
              last_column: lstack[lstack.length - 1].last_column
            };
            if (ranges) {
              yyval._$.range = [
                lstack[lstack.length - (len || 1)].range[0],
                lstack[lstack.length - 1].range[1]
              ];
            }
            r = this.performAction.apply(yyval, [
              yytext,
              yyleng,
              yylineno,
              sharedState.yy,
              action[1],
              vstack,
              lstack
            ].concat(args));
            if (typeof r !== "undefined") {
              return r;
            }
            if (len) {
              stack = stack.slice(0, -1 * len * 2);
              vstack = vstack.slice(0, -1 * len);
              lstack = lstack.slice(0, -1 * len);
            }
            stack.push(this.productions_[action[1]][0]);
            vstack.push(yyval.$);
            lstack.push(yyval._$);
            newState = table[stack[stack.length - 2]][stack[stack.length - 1]];
            stack.push(newState);
            break;
          case 3:
            return true;
        }
      }
      return true;
    }, "parse")
  };
  var lexer = /* @__PURE__ */ (function() {
    var lexer2 = {
      EOF: 1,
      parseError: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function parseError(str, hash) {
        if (this.yy.parser) {
          this.yy.parser.parseError(str, hash);
        } else {
          throw new Error(str);
        }
      }, "parseError"),
      // resets the lexer, sets new input
      setInput: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(input, yy) {
        this.yy = yy || this.yy || {};
        this._input = input;
        this._more = this._backtrack = this.done = false;
        this.yylineno = this.yyleng = 0;
        this.yytext = this.matched = this.match = "";
        this.conditionStack = ["INITIAL"];
        this.yylloc = {
          first_line: 1,
          first_column: 0,
          last_line: 1,
          last_column: 0
        };
        if (this.options.ranges) {
          this.yylloc.range = [0, 0];
        }
        this.offset = 0;
        return this;
      }, "setInput"),
      // consumes and returns one char from the input
      input: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        var ch = this._input[0];
        this.yytext += ch;
        this.yyleng++;
        this.offset++;
        this.match += ch;
        this.matched += ch;
        var lines = ch.match(/(?:\r\n?|\n).*/g);
        if (lines) {
          this.yylineno++;
          this.yylloc.last_line++;
        } else {
          this.yylloc.last_column++;
        }
        if (this.options.ranges) {
          this.yylloc.range[1]++;
        }
        this._input = this._input.slice(1);
        return ch;
      }, "input"),
      // unshifts one char (or a string) into the input
      unput: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(ch) {
        var len = ch.length;
        var lines = ch.split(/(?:\r\n?|\n)/g);
        this._input = ch + this._input;
        this.yytext = this.yytext.substr(0, this.yytext.length - len);
        this.offset -= len;
        var oldLines = this.match.split(/(?:\r\n?|\n)/g);
        this.match = this.match.substr(0, this.match.length - 1);
        this.matched = this.matched.substr(0, this.matched.length - 1);
        if (lines.length - 1) {
          this.yylineno -= lines.length - 1;
        }
        var r = this.yylloc.range;
        this.yylloc = {
          first_line: this.yylloc.first_line,
          last_line: this.yylineno + 1,
          first_column: this.yylloc.first_column,
          last_column: lines ? (lines.length === oldLines.length ? this.yylloc.first_column : 0) + oldLines[oldLines.length - lines.length].length - lines[0].length : this.yylloc.first_column - len
        };
        if (this.options.ranges) {
          this.yylloc.range = [r[0], r[0] + this.yyleng - len];
        }
        this.yyleng = this.yytext.length;
        return this;
      }, "unput"),
      // When called from action, caches matched text and appends it on next action
      more: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        this._more = true;
        return this;
      }, "more"),
      // When called from action, signals the lexer that this rule fails to match the input, so the next matching rule (regex) should be tested instead.
      reject: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        if (this.options.backtrack_lexer) {
          this._backtrack = true;
        } else {
          return this.parseError("Lexical error on line " + (this.yylineno + 1) + ". You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).\n" + this.showPosition(), {
            text: "",
            token: null,
            line: this.yylineno
          });
        }
        return this;
      }, "reject"),
      // retain first n characters of the match
      less: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(n) {
        this.unput(this.match.slice(n));
      }, "less"),
      // displays already matched input, i.e. for error messages
      pastInput: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        var past = this.matched.substr(0, this.matched.length - this.match.length);
        return (past.length > 20 ? "..." : "") + past.substr(-20).replace(/\n/g, "");
      }, "pastInput"),
      // displays upcoming input, i.e. for error messages
      upcomingInput: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        var next = this.match;
        if (next.length < 20) {
          next += this._input.substr(0, 20 - next.length);
        }
        return (next.substr(0, 20) + (next.length > 20 ? "..." : "")).replace(/\n/g, "");
      }, "upcomingInput"),
      // displays the character position where the lexing error occurred, i.e. for error messages
      showPosition: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        var pre = this.pastInput();
        var c = new Array(pre.length + 1).join("-");
        return pre + this.upcomingInput() + "\n" + c + "^";
      }, "showPosition"),
      // test the lexed token: return FALSE when not a match, otherwise return token
      test_match: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(match, indexed_rule) {
        var token, lines, backup;
        if (this.options.backtrack_lexer) {
          backup = {
            yylineno: this.yylineno,
            yylloc: {
              first_line: this.yylloc.first_line,
              last_line: this.last_line,
              first_column: this.yylloc.first_column,
              last_column: this.yylloc.last_column
            },
            yytext: this.yytext,
            match: this.match,
            matches: this.matches,
            matched: this.matched,
            yyleng: this.yyleng,
            offset: this.offset,
            _more: this._more,
            _input: this._input,
            yy: this.yy,
            conditionStack: this.conditionStack.slice(0),
            done: this.done
          };
          if (this.options.ranges) {
            backup.yylloc.range = this.yylloc.range.slice(0);
          }
        }
        lines = match[0].match(/(?:\r\n?|\n).*/g);
        if (lines) {
          this.yylineno += lines.length;
        }
        this.yylloc = {
          first_line: this.yylloc.last_line,
          last_line: this.yylineno + 1,
          first_column: this.yylloc.last_column,
          last_column: lines ? lines[lines.length - 1].length - lines[lines.length - 1].match(/\r?\n?/)[0].length : this.yylloc.last_column + match[0].length
        };
        this.yytext += match[0];
        this.match += match[0];
        this.matches = match;
        this.yyleng = this.yytext.length;
        if (this.options.ranges) {
          this.yylloc.range = [this.offset, this.offset += this.yyleng];
        }
        this._more = false;
        this._backtrack = false;
        this._input = this._input.slice(match[0].length);
        this.matched += match[0];
        token = this.performAction.call(this, this.yy, this, indexed_rule, this.conditionStack[this.conditionStack.length - 1]);
        if (this.done && this._input) {
          this.done = false;
        }
        if (token) {
          return token;
        } else if (this._backtrack) {
          for (var k in backup) {
            this[k] = backup[k];
          }
          return false;
        }
        return false;
      }, "test_match"),
      // return next match in input
      next: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
        if (this.done) {
          return this.EOF;
        }
        if (!this._input) {
          this.done = true;
        }
        var token, match, tempMatch, index;
        if (!this._more) {
          this.yytext = "";
          this.match = "";
        }
        var rules = this._currentRules();
        for (var i = 0; i < rules.length; i++) {
          tempMatch = this._input.match(this.rules[rules[i]]);
          if (tempMatch && (!match || tempMatch[0].length > match[0].length)) {
            match = tempMatch;
            index = i;
            if (this.options.backtrack_lexer) {
              token = this.test_match(tempMatch, rules[i]);
              if (token !== false) {
                return token;
              } else if (this._backtrack) {
                match = false;
                continue;
              } else {
                return false;
              }
            } else if (!this.options.flex) {
              break;
            }
          }
        }
        if (match) {
          token = this.test_match(match, rules[index]);
          if (token !== false) {
            return token;
          }
          return false;
        }
        if (this._input === "") {
          return this.EOF;
        } else {
          return this.parseError("Lexical error on line " + (this.yylineno + 1) + ". Unrecognized text.\n" + this.showPosition(), {
            text: "",
            token: null,
            line: this.yylineno
          });
        }
      }, "next"),
      // return next match that has a token
      lex: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function lex() {
        var r = this.next();
        if (r) {
          return r;
        } else {
          return this.lex();
        }
      }, "lex"),
      // activates a new lexer condition state (pushes the new lexer condition state onto the condition stack)
      begin: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function begin(condition) {
        this.conditionStack.push(condition);
      }, "begin"),
      // pop the previously active lexer condition state off the condition stack
      popState: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function popState() {
        var n = this.conditionStack.length - 1;
        if (n > 0) {
          return this.conditionStack.pop();
        } else {
          return this.conditionStack[0];
        }
      }, "popState"),
      // produce the lexer rule set which is active for the currently active lexer condition state
      _currentRules: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function _currentRules() {
        if (this.conditionStack.length && this.conditionStack[this.conditionStack.length - 1]) {
          return this.conditions[this.conditionStack[this.conditionStack.length - 1]].rules;
        } else {
          return this.conditions["INITIAL"].rules;
        }
      }, "_currentRules"),
      // return the currently active lexer condition state; when an index argument is provided it produces the N-th previous condition state, if available
      topState: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function topState(n) {
        n = this.conditionStack.length - 1 - Math.abs(n || 0);
        if (n >= 0) {
          return this.conditionStack[n];
        } else {
          return "INITIAL";
        }
      }, "topState"),
      // alias for begin(condition)
      pushState: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function pushState(condition) {
        this.begin(condition);
      }, "pushState"),
      // return the number of states currently on the stack
      stateStackSize: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function stateStackSize() {
        return this.conditionStack.length;
      }, "stateStackSize"),
      options: {},
      performAction: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function anonymous(yy, yy_, $avoiding_name_collisions, YY_START) {
        var YYSTATE = YY_START;
        switch ($avoiding_name_collisions) {
          case 0:
            yy.getLogger().debug("Found block-beta");
            return 10;
            // removed by dead control flow

          case 1:
            yy.getLogger().debug("Found id-block");
            return 29;
            // removed by dead control flow

          case 2:
            yy.getLogger().debug("Found block");
            return 10;
            // removed by dead control flow

          case 3:
            yy.getLogger().debug(".", yy_.yytext);
            break;
          case 4:
            yy.getLogger().debug("_", yy_.yytext);
            break;
          case 5:
            return 5;
            // removed by dead control flow

          case 6:
            yy_.yytext = -1;
            return 28;
            // removed by dead control flow

          case 7:
            yy_.yytext = yy_.yytext.replace(/columns\s+/, "");
            yy.getLogger().debug("COLUMNS (LEX)", yy_.yytext);
            return 28;
            // removed by dead control flow

          case 8:
            this.pushState("md_string");
            break;
          case 9:
            return "MD_STR";
            // removed by dead control flow

          case 10:
            this.popState();
            break;
          case 11:
            this.pushState("string");
            break;
          case 12:
            yy.getLogger().debug("LEX: POPPING STR:", yy_.yytext);
            this.popState();
            break;
          case 13:
            yy.getLogger().debug("LEX: STR end:", yy_.yytext);
            return "STR";
            // removed by dead control flow

          case 14:
            yy_.yytext = yy_.yytext.replace(/space\:/, "");
            yy.getLogger().debug("SPACE NUM (LEX)", yy_.yytext);
            return 21;
            // removed by dead control flow

          case 15:
            yy_.yytext = "1";
            yy.getLogger().debug("COLUMNS (LEX)", yy_.yytext);
            return 21;
            // removed by dead control flow

          case 16:
            return 42;
            // removed by dead control flow

          case 17:
            return "LINKSTYLE";
            // removed by dead control flow

          case 18:
            return "INTERPOLATE";
            // removed by dead control flow

          case 19:
            this.pushState("CLASSDEF");
            return 39;
            // removed by dead control flow

          case 20:
            this.popState();
            this.pushState("CLASSDEFID");
            return "DEFAULT_CLASSDEF_ID";
            // removed by dead control flow

          case 21:
            this.popState();
            this.pushState("CLASSDEFID");
            return 40;
            // removed by dead control flow

          case 22:
            this.popState();
            return 41;
            // removed by dead control flow

          case 23:
            this.pushState("CLASS");
            return 43;
            // removed by dead control flow

          case 24:
            this.popState();
            this.pushState("CLASS_STYLE");
            return 44;
            // removed by dead control flow

          case 25:
            this.popState();
            return 45;
            // removed by dead control flow

          case 26:
            this.pushState("STYLE_STMNT");
            return 46;
            // removed by dead control flow

          case 27:
            this.popState();
            this.pushState("STYLE_DEFINITION");
            return 47;
            // removed by dead control flow

          case 28:
            this.popState();
            return 48;
            // removed by dead control flow

          case 29:
            this.pushState("acc_title");
            return "acc_title";
            // removed by dead control flow

          case 30:
            this.popState();
            return "acc_title_value";
            // removed by dead control flow

          case 31:
            this.pushState("acc_descr");
            return "acc_descr";
            // removed by dead control flow

          case 32:
            this.popState();
            return "acc_descr_value";
            // removed by dead control flow

          case 33:
            this.pushState("acc_descr_multiline");
            break;
          case 34:
            this.popState();
            break;
          case 35:
            return "acc_descr_multiline_value";
            // removed by dead control flow

          case 36:
            return 30;
            // removed by dead control flow

          case 37:
            this.popState();
            yy.getLogger().debug("Lex: ((");
            return "NODE_DEND";
            // removed by dead control flow

          case 38:
            this.popState();
            yy.getLogger().debug("Lex: ((");
            return "NODE_DEND";
            // removed by dead control flow

          case 39:
            this.popState();
            yy.getLogger().debug("Lex: ))");
            return "NODE_DEND";
            // removed by dead control flow

          case 40:
            this.popState();
            yy.getLogger().debug("Lex: ((");
            return "NODE_DEND";
            // removed by dead control flow

          case 41:
            this.popState();
            yy.getLogger().debug("Lex: ((");
            return "NODE_DEND";
            // removed by dead control flow

          case 42:
            this.popState();
            yy.getLogger().debug("Lex: (-");
            return "NODE_DEND";
            // removed by dead control flow

          case 43:
            this.popState();
            yy.getLogger().debug("Lex: -)");
            return "NODE_DEND";
            // removed by dead control flow

          case 44:
            this.popState();
            yy.getLogger().debug("Lex: ((");
            return "NODE_DEND";
            // removed by dead control flow

          case 45:
            this.popState();
            yy.getLogger().debug("Lex: ]]");
            return "NODE_DEND";
            // removed by dead control flow

          case 46:
            this.popState();
            yy.getLogger().debug("Lex: (");
            return "NODE_DEND";
            // removed by dead control flow

          case 47:
            this.popState();
            yy.getLogger().debug("Lex: ])");
            return "NODE_DEND";
            // removed by dead control flow

          case 48:
            this.popState();
            yy.getLogger().debug("Lex: /]");
            return "NODE_DEND";
            // removed by dead control flow

          case 49:
            this.popState();
            yy.getLogger().debug("Lex: /]");
            return "NODE_DEND";
            // removed by dead control flow

          case 50:
            this.popState();
            yy.getLogger().debug("Lex: )]");
            return "NODE_DEND";
            // removed by dead control flow

          case 51:
            this.popState();
            yy.getLogger().debug("Lex: )");
            return "NODE_DEND";
            // removed by dead control flow

          case 52:
            this.popState();
            yy.getLogger().debug("Lex: ]>");
            return "NODE_DEND";
            // removed by dead control flow

          case 53:
            this.popState();
            yy.getLogger().debug("Lex: ]");
            return "NODE_DEND";
            // removed by dead control flow

          case 54:
            yy.getLogger().debug("Lexa: -)");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 55:
            yy.getLogger().debug("Lexa: (-");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 56:
            yy.getLogger().debug("Lexa: ))");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 57:
            yy.getLogger().debug("Lexa: )");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 58:
            yy.getLogger().debug("Lex: (((");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 59:
            yy.getLogger().debug("Lexa: )");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 60:
            yy.getLogger().debug("Lexa: )");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 61:
            yy.getLogger().debug("Lexa: )");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 62:
            yy.getLogger().debug("Lexc: >");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 63:
            yy.getLogger().debug("Lexa: ([");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 64:
            yy.getLogger().debug("Lexa: )");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 65:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 66:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 67:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 68:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 69:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 70:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 71:
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 72:
            yy.getLogger().debug("Lexa: [");
            this.pushState("NODE");
            return 35;
            // removed by dead control flow

          case 73:
            this.pushState("BLOCK_ARROW");
            yy.getLogger().debug("LEX ARR START");
            return 37;
            // removed by dead control flow

          case 74:
            yy.getLogger().debug("Lex: NODE_ID", yy_.yytext);
            return 31;
            // removed by dead control flow

          case 75:
            yy.getLogger().debug("Lex: EOF", yy_.yytext);
            return 8;
            // removed by dead control flow

          case 76:
            this.pushState("md_string");
            break;
          case 77:
            this.pushState("md_string");
            break;
          case 78:
            return "NODE_DESCR";
            // removed by dead control flow

          case 79:
            this.popState();
            break;
          case 80:
            yy.getLogger().debug("Lex: Starting string");
            this.pushState("string");
            break;
          case 81:
            yy.getLogger().debug("LEX ARR: Starting string");
            this.pushState("string");
            break;
          case 82:
            yy.getLogger().debug("LEX: NODE_DESCR:", yy_.yytext);
            return "NODE_DESCR";
            // removed by dead control flow

          case 83:
            yy.getLogger().debug("LEX POPPING");
            this.popState();
            break;
          case 84:
            yy.getLogger().debug("Lex: =>BAE");
            this.pushState("ARROW_DIR");
            break;
          case 85:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (right): dir:", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 86:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (left):", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 87:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (x):", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 88:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (y):", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 89:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (up):", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 90:
            yy_.yytext = yy_.yytext.replace(/^,\s*/, "");
            yy.getLogger().debug("Lex (down):", yy_.yytext);
            return "DIR";
            // removed by dead control flow

          case 91:
            yy_.yytext = "]>";
            yy.getLogger().debug("Lex (ARROW_DIR end):", yy_.yytext);
            this.popState();
            this.popState();
            return "BLOCK_ARROW_END";
            // removed by dead control flow

          case 92:
            yy.getLogger().debug("Lex: LINK", "#" + yy_.yytext + "#");
            return 15;
            // removed by dead control flow

          case 93:
            yy.getLogger().debug("Lex: LINK", yy_.yytext);
            return 15;
            // removed by dead control flow

          case 94:
            yy.getLogger().debug("Lex: LINK", yy_.yytext);
            return 15;
            // removed by dead control flow

          case 95:
            yy.getLogger().debug("Lex: LINK", yy_.yytext);
            return 15;
            // removed by dead control flow

          case 96:
            yy.getLogger().debug("Lex: START_LINK", yy_.yytext);
            this.pushState("LLABEL");
            return 16;
            // removed by dead control flow

          case 97:
            yy.getLogger().debug("Lex: START_LINK", yy_.yytext);
            this.pushState("LLABEL");
            return 16;
            // removed by dead control flow

          case 98:
            yy.getLogger().debug("Lex: START_LINK", yy_.yytext);
            this.pushState("LLABEL");
            return 16;
            // removed by dead control flow

          case 99:
            this.pushState("md_string");
            break;
          case 100:
            yy.getLogger().debug("Lex: Starting string");
            this.pushState("string");
            return "LINK_LABEL";
            // removed by dead control flow

          case 101:
            this.popState();
            yy.getLogger().debug("Lex: LINK", "#" + yy_.yytext + "#");
            return 15;
            // removed by dead control flow

          case 102:
            this.popState();
            yy.getLogger().debug("Lex: LINK", yy_.yytext);
            return 15;
            // removed by dead control flow

          case 103:
            this.popState();
            yy.getLogger().debug("Lex: LINK", yy_.yytext);
            return 15;
            // removed by dead control flow

          case 104:
            yy.getLogger().debug("Lex: COLON", yy_.yytext);
            yy_.yytext = yy_.yytext.slice(1);
            return 27;
            // removed by dead control flow

        }
      }, "anonymous"),
      rules: [/^(?:block-beta\b)/, /^(?:block:)/, /^(?:block\b)/, /^(?:[\s]+)/, /^(?:[\n]+)/, /^(?:((\u000D\u000A)|(\u000A)))/, /^(?:columns\s+auto\b)/, /^(?:columns\s+[\d]+)/, /^(?:["][`])/, /^(?:[^`"]+)/, /^(?:[`]["])/, /^(?:["])/, /^(?:["])/, /^(?:[^"]*)/, /^(?:space[:]\d+)/, /^(?:space\b)/, /^(?:default\b)/, /^(?:linkStyle\b)/, /^(?:interpolate\b)/, /^(?:classDef\s+)/, /^(?:DEFAULT\s+)/, /^(?:\w+\s+)/, /^(?:[^\n]*)/, /^(?:class\s+)/, /^(?:(\w+)+((,\s*\w+)*))/, /^(?:[^\n]*)/, /^(?:style\s+)/, /^(?:(\w+)+((,\s*\w+)*))/, /^(?:[^\n]*)/, /^(?:accTitle\s*:\s*)/, /^(?:(?!\n||)*[^\n]*)/, /^(?:accDescr\s*:\s*)/, /^(?:(?!\n||)*[^\n]*)/, /^(?:accDescr\s*\{\s*)/, /^(?:[\}])/, /^(?:[^\}]*)/, /^(?:end\b\s*)/, /^(?:\(\(\()/, /^(?:\)\)\))/, /^(?:[\)]\))/, /^(?:\}\})/, /^(?:\})/, /^(?:\(-)/, /^(?:-\))/, /^(?:\(\()/, /^(?:\]\])/, /^(?:\()/, /^(?:\]\))/, /^(?:\\\])/, /^(?:\/\])/, /^(?:\)\])/, /^(?:[\)])/, /^(?:\]>)/, /^(?:[\]])/, /^(?:-\))/, /^(?:\(-)/, /^(?:\)\))/, /^(?:\))/, /^(?:\(\(\()/, /^(?:\(\()/, /^(?:\{\{)/, /^(?:\{)/, /^(?:>)/, /^(?:\(\[)/, /^(?:\()/, /^(?:\[\[)/, /^(?:\[\|)/, /^(?:\[\()/, /^(?:\)\)\))/, /^(?:\[\\)/, /^(?:\[\/)/, /^(?:\[\\)/, /^(?:\[)/, /^(?:<\[)/, /^(?:[^\(\[\n\-\)\{\}\s\<\>:=]+)/, /^(?:$)/, /^(?:["][`])/, /^(?:["][`])/, /^(?:[^`"]+)/, /^(?:[`]["])/, /^(?:["])/, /^(?:["])/, /^(?:[^"]+)/, /^(?:["])/, /^(?:\]>\s*\()/, /^(?:,?\s*right\s*)/, /^(?:,?\s*left\s*)/, /^(?:,?\s*x\s*)/, /^(?:,?\s*y\s*)/, /^(?:,?\s*up\s*)/, /^(?:,?\s*down\s*)/, /^(?:\)\s*)/, /^(?:\s*[xo<]?--+[-xo>]\s*)/, /^(?:\s*[xo<]?==+[=xo>]\s*)/, /^(?:\s*[xo<]?-?\.+-[xo>]?\s*)/, /^(?:\s*~~[\~]+\s*)/, /^(?:\s*[xo<]?--\s*)/, /^(?:\s*[xo<]?==\s*)/, /^(?:\s*[xo<]?-\.\s*)/, /^(?:["][`])/, /^(?:["])/, /^(?:\s*[xo<]?--+[-xo>]\s*)/, /^(?:\s*[xo<]?==+[=xo>]\s*)/, /^(?:\s*[xo<]?-?\.+-[xo>]?\s*)/, /^(?::\d+)/],
      conditions: { "STYLE_DEFINITION": { "rules": [28], "inclusive": false }, "STYLE_STMNT": { "rules": [27], "inclusive": false }, "CLASSDEFID": { "rules": [22], "inclusive": false }, "CLASSDEF": { "rules": [20, 21], "inclusive": false }, "CLASS_STYLE": { "rules": [25], "inclusive": false }, "CLASS": { "rules": [24], "inclusive": false }, "LLABEL": { "rules": [99, 100, 101, 102, 103], "inclusive": false }, "ARROW_DIR": { "rules": [85, 86, 87, 88, 89, 90, 91], "inclusive": false }, "BLOCK_ARROW": { "rules": [76, 81, 84], "inclusive": false }, "NODE": { "rules": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 77, 80], "inclusive": false }, "md_string": { "rules": [9, 10, 78, 79], "inclusive": false }, "space": { "rules": [], "inclusive": false }, "string": { "rules": [12, 13, 82, 83], "inclusive": false }, "acc_descr_multiline": { "rules": [34, 35], "inclusive": false }, "acc_descr": { "rules": [32], "inclusive": false }, "acc_title": { "rules": [30], "inclusive": false }, "INITIAL": { "rules": [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17, 18, 19, 23, 26, 29, 31, 33, 36, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 92, 93, 94, 95, 96, 97, 98, 104], "inclusive": true } }
    };
    return lexer2;
  })();
  parser2.lexer = lexer;
  function Parser() {
    this.yy = {};
  }
  (0,chunk_AGHRB4JF/* __name */.K2)(Parser, "Parser");
  Parser.prototype = parser2;
  parser2.Parser = Parser;
  return new Parser();
})();
parser.parser = parser;
var block_default = parser;

// src/diagrams/block/blockDB.ts

var blockDatabase = /* @__PURE__ */ new Map();
var edgeList = [];
var edgeCount = /* @__PURE__ */ new Map();
var COLOR_KEYWORD = "color";
var FILL_KEYWORD = "fill";
var BG_FILL = "bgFill";
var STYLECLASS_SEP = ",";
var config = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)();
var classes = /* @__PURE__ */ new Map();
var diagramId = "";
var sanitizeText2 = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((txt) => chunk_CSCIHK7Q/* common_default */.Y2.sanitizeText(txt, config), "sanitizeText");
var addStyleClass = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(id, styleAttributes = "") {
  let foundClass = classes.get(id);
  if (!foundClass) {
    foundClass = { id, styles: [], textStyles: [] };
    classes.set(id, foundClass);
  }
  if (styleAttributes !== void 0 && styleAttributes !== null) {
    styleAttributes.split(STYLECLASS_SEP).forEach((attrib) => {
      const fixedAttrib = attrib.replace(/([^;]*);/, "$1").trim();
      if (RegExp(COLOR_KEYWORD).exec(attrib)) {
        const newStyle1 = fixedAttrib.replace(FILL_KEYWORD, BG_FILL);
        const newStyle2 = newStyle1.replace(COLOR_KEYWORD, FILL_KEYWORD);
        foundClass.textStyles.push(newStyle2);
      }
      foundClass.styles.push(fixedAttrib);
    });
  }
}, "addStyleClass");
var addStyle2Node = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(id, styles = "") {
  const foundBlock = blockDatabase.get(id);
  if (styles !== void 0 && styles !== null) {
    foundBlock.styles = styles.split(STYLECLASS_SEP);
  }
}, "addStyle2Node");
var setCssClass = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(itemIds, cssClassName) {
  itemIds.split(",").forEach(function(id) {
    let foundBlock = blockDatabase.get(id);
    if (foundBlock === void 0) {
      const trimmedId = id.trim();
      foundBlock = { id: trimmedId, type: "na", children: [] };
      blockDatabase.set(trimmedId, foundBlock);
    }
    if (!foundBlock.classes) {
      foundBlock.classes = [];
    }
    foundBlock.classes.push(cssClassName);
  });
}, "setCssClass");
var populateBlockDatabase = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((_blockList, parent) => {
  const blockList = _blockList.flat();
  const children = [];
  const columnSettingBlock = blockList.find((b) => b?.type === "column-setting");
  const column = columnSettingBlock?.columns ?? -1;
  for (const block of blockList) {
    if (typeof column === "number" && column > 0 && block.type !== "column-setting" && typeof block.widthInColumns === "number" && block.widthInColumns > column) {
      chunk_AGHRB4JF/* log */.Rm.warn(
        `Block ${block.id} width ${block.widthInColumns} exceeds configured column width ${column}`
      );
    }
    if (block.label) {
      block.label = sanitizeText2(block.label);
    }
    if (block.type === "classDef") {
      addStyleClass(block.id, block.css);
      continue;
    }
    if (block.type === "applyClass") {
      setCssClass(block.id, block?.styleClass ?? "");
      continue;
    }
    if (block.type === "applyStyles") {
      if (block?.stylesStr) {
        addStyle2Node(block.id, block?.stylesStr);
      }
      continue;
    }
    if (block.type === "column-setting") {
      parent.columns = block.columns ?? -1;
    } else if (block.type === "edge") {
      const count = (edgeCount.get(block.id) ?? 0) + 1;
      edgeCount.set(block.id, count);
      block.id = count + "-" + block.id;
      edgeList.push(block);
    } else {
      if (!block.label) {
        if (block.type === "composite") {
          block.label = "";
        } else {
          block.label = block.id;
        }
      }
      const existingBlock = blockDatabase.get(block.id);
      if (existingBlock === void 0) {
        blockDatabase.set(block.id, block);
      } else {
        if (block.type !== "na") {
          existingBlock.type = block.type;
        }
        if (block.label !== block.id) {
          existingBlock.label = block.label;
        }
      }
      if (block.children) {
        populateBlockDatabase(block.children, block);
      }
      if (block.type === "space") {
        const w = block.width ?? 1;
        for (let j = 0; j < w; j++) {
          const newBlock = clone(block);
          newBlock.id = newBlock.id + "-" + j;
          blockDatabase.set(newBlock.id, newBlock);
          children.push(newBlock);
        }
      } else if (existingBlock === void 0) {
        children.push(block);
      }
    }
  }
  parent.children = children;
}, "populateBlockDatabase");
var blocks = [];
var rootBlock = { id: "root", type: "composite", children: [], columns: -1 };
var clear2 = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => {
  chunk_AGHRB4JF/* log */.Rm.debug("Clear called");
  (0,chunk_CSCIHK7Q/* clear */.IU)();
  rootBlock = { id: "root", type: "composite", children: [], columns: -1 };
  blockDatabase = /* @__PURE__ */ new Map([["root", rootBlock]]);
  blocks = [];
  classes = /* @__PURE__ */ new Map();
  edgeList = [];
  edgeCount = /* @__PURE__ */ new Map();
  diagramId = "";
}, "clear");
function typeStr2Type(typeStr) {
  chunk_AGHRB4JF/* log */.Rm.debug("typeStr2Type", typeStr);
  switch (typeStr) {
    case "[]":
      return "square";
    case "()":
      chunk_AGHRB4JF/* log */.Rm.debug("we have a round");
      return "round";
    case "(())":
      return "circle";
    case ">]":
      return "rect_left_inv_arrow";
    case "{}":
      return "diamond";
    case "{{}}":
      return "hexagon";
    case "([])":
      return "stadium";
    case "[[]]":
      return "subroutine";
    case "[()]":
      return "cylinder";
    case "((()))":
      return "doublecircle";
    case "[//]":
      return "lean_right";
    case "[\\\\]":
      return "lean_left";
    case "[/\\]":
      return "trapezoid";
    case "[\\/]":
      return "inv_trapezoid";
    case "<[]>":
      return "block_arrow";
    default:
      return "na";
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(typeStr2Type, "typeStr2Type");
function edgeTypeStr2Type(typeStr) {
  chunk_AGHRB4JF/* log */.Rm.debug("typeStr2Type", typeStr);
  switch (typeStr) {
    case "==":
      return "thick";
    default:
      return "normal";
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(edgeTypeStr2Type, "edgeTypeStr2Type");
function edgeStrToEdgeData(typeStr) {
  const lastChar = typeStr.trim().slice(-1);
  switch (lastChar) {
    case "x":
      return "arrow_cross";
    case "o":
      return "arrow_circle";
    case ">":
      return "arrow_point";
    default:
      return "";
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(edgeStrToEdgeData, "edgeStrToEdgeData");
function edgeStrToEdgeStartData(typeStr) {
  const firstChar = typeStr.trim().charAt(0);
  switch (firstChar) {
    case "x":
      return "arrow_cross";
    case "o":
      return "arrow_circle";
    case "<":
      return "arrow_point";
    default:
      return "arrow_open";
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(edgeStrToEdgeStartData, "edgeStrToEdgeStartData");
function edgeStrToThickness(typeStr) {
  return typeStr.includes("==") ? "thick" : "normal";
}
(0,chunk_AGHRB4JF/* __name */.K2)(edgeStrToThickness, "edgeStrToThickness");
function edgeStrToPattern(typeStr) {
  if (typeStr.includes(".-")) {
    return "dotted";
  }
  return "solid";
}
(0,chunk_AGHRB4JF/* __name */.K2)(edgeStrToPattern, "edgeStrToPattern");
var cnt = 0;
var generateId = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => {
  cnt++;
  return "id-" + Math.random().toString(36).substr(2, 12) + "-" + cnt;
}, "generateId");
var setHierarchy = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((block) => {
  rootBlock.children = block;
  populateBlockDatabase(block, rootBlock);
  blocks = rootBlock.children;
}, "setHierarchy");
var getColumns = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((blockId) => {
  const block = blockDatabase.get(blockId);
  if (!block) {
    return -1;
  }
  if (block.columns) {
    return block.columns;
  }
  if (!block.children) {
    return -1;
  }
  return block.children.length;
}, "getColumns");
var getBlocksFlat = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => {
  return [...blockDatabase.values()];
}, "getBlocksFlat");
var getBlocks = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => {
  return blocks || [];
}, "getBlocks");
var getEdges = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => {
  return edgeList;
}, "getEdges");
var getBlock = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((id) => {
  return blockDatabase.get(id);
}, "getBlock");
var setBlock = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((block) => {
  blockDatabase.set(block.id, block);
}, "setBlock");
var setDiagramId = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((id) => {
  diagramId = id;
}, "setDiagramId");
var getDiagramId = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => diagramId, "getDiagramId");
var getLogger = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => chunk_AGHRB4JF/* log */.Rm, "getLogger");
var getClasses = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function() {
  return classes;
}, "getClasses");
var db = {
  getConfig: /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(() => (0,chunk_CSCIHK7Q/* getConfig */.zj)().block, "getConfig"),
  typeStr2Type,
  edgeTypeStr2Type,
  edgeStrToEdgeData,
  edgeStrToEdgeStartData,
  edgeStrToThickness,
  edgeStrToPattern,
  getLogger,
  getBlocksFlat,
  getBlocks,
  getEdges,
  setHierarchy,
  getBlock,
  setBlock,
  getColumns,
  getClasses,
  clear: clear2,
  generateId,
  setDiagramId,
  getDiagramId
};
var blockDB_default = db;

// src/diagrams/block/styles.ts

var fade = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((color, opacity) => {
  const channel2 = channel/* default */.A;
  const r = channel2(color, "r");
  const g = channel2(color, "g");
  const b = channel2(color, "b");
  return rgba/* default */.A(r, g, b, opacity);
}, "fade");
var getStyles = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((options) => `.label {
    font-family: ${options.fontFamily};
    color: ${options.nodeTextColor || options.textColor};
  }
  .cluster-label text {
    fill: ${options.titleColor};
  }
  .cluster-label span,p {
    color: ${options.titleColor};
  }



  .label text,span,p {
    fill: ${options.nodeTextColor || options.textColor};
    color: ${options.nodeTextColor || options.textColor};
  }

  .node rect,
  .node circle,
  .node ellipse,
  .node polygon,
  .node path {
    fill: ${options.mainBkg};
    stroke: ${options.nodeBorder};
    stroke-width: 1px;
  }
  .flowchart-label text {
    text-anchor: middle;
  }
  // .flowchart-label .text-outer-tspan {
  //   text-anchor: middle;
  // }
  // .flowchart-label .text-inner-tspan {
  //   text-anchor: start;
  // }

  .node .label {
    text-align: center;
  }
  .node.clickable {
    cursor: pointer;
  }

  .arrowheadPath {
    fill: ${options.arrowheadColor};
  }

  .edgePath .path {
    stroke: ${options.lineColor};
    stroke-width: 2.0px;
  }

  .flowchart-link {
    stroke: ${options.lineColor};
    fill: none;
  }

  .edgeLabel {
    background-color: ${options.edgeLabelBackground};
    /*
     * This is for backward compatibility with existing code that didn't
     * add a \`<p>\` around edge labels.
     *
     * TODO: We should probably remove this in a future release.
     */
    p {
      margin: 0;
      padding: 0;
      display: inline;
    }
    rect {
      opacity: 0.5;
      background-color: ${options.edgeLabelBackground};
      fill: ${options.edgeLabelBackground};
    }
    text-align: center;
  }

  /* For html labels only */
  .labelBkg {
    background-color: ${options.edgeLabelBackground};
  }

  .node .cluster {
    // fill: ${fade(options.mainBkg, 0.5)};
    fill: ${fade(options.clusterBkg, 0.5)};
    stroke: ${fade(options.clusterBorder, 0.2)};
    box-shadow: rgba(50, 50, 93, 0.25) 0px 13px 27px -5px, rgba(0, 0, 0, 0.3) 0px 8px 16px -8px;
    stroke-width: 1px;
  }

  .cluster text {
    fill: ${options.titleColor};
  }

  .cluster span,p {
    color: ${options.titleColor};
  }
  /* .cluster div {
    color: ${options.titleColor};
  } */

  div.mermaidTooltip {
    position: absolute;
    text-align: center;
    max-width: 200px;
    padding: 2px;
    font-family: ${options.fontFamily};
    font-size: 12px;
    background: ${options.tertiaryColor};
    border: 1px solid ${options.border2};
    border-radius: 2px;
    pointer-events: none;
    z-index: 100;
  }

  .flowchartTitleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${options.textColor};
  }
  ${(0,chunk_FMBD7UC4/* getIconStyles */.o)()}
`, "getStyles");
var styles_default = getStyles;

// src/diagrams/block/blockRenderer.ts


// src/dagre-wrapper/markers.js
var insertMarkers = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, markerArray, type, id) => {
  markerArray.forEach((markerName) => {
    markers[markerName](elem, type, id);
  });
}, "insertMarkers");
var extension = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  chunk_AGHRB4JF/* log */.Rm.trace("Making markers for ", id);
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-extensionStart").attr("class", "marker extension " + type).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 1,7 L18,13 V 1 Z");
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-extensionEnd").attr("class", "marker extension " + type).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 1,1 V 13 L18,7 Z");
}, "extension");
var composition = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-compositionStart").attr("class", "marker composition " + type).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-compositionEnd").attr("class", "marker composition " + type).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
}, "composition");
var aggregation = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-aggregationStart").attr("class", "marker aggregation " + type).attr("refX", 18).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-aggregationEnd").attr("class", "marker aggregation " + type).attr("refX", 1).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L1,7 L9,1 Z");
}, "aggregation");
var dependency = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-dependencyStart").attr("class", "marker dependency " + type).attr("refX", 6).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("path").attr("d", "M 5,7 L9,13 L1,7 L9,1 Z");
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-dependencyEnd").attr("class", "marker dependency " + type).attr("refX", 13).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 28).attr("orient", "auto").append("path").attr("d", "M 18,7 L9,13 L14,7 L9,1 Z");
}, "dependency");
var lollipop = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-lollipopStart").attr("class", "marker lollipop " + type).attr("refX", 13).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("circle").attr("stroke", "black").attr("fill", "transparent").attr("cx", 7).attr("cy", 7).attr("r", 6);
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-lollipopEnd").attr("class", "marker lollipop " + type).attr("refX", 1).attr("refY", 7).attr("markerWidth", 190).attr("markerHeight", 240).attr("orient", "auto").append("circle").attr("stroke", "black").attr("fill", "transparent").attr("cx", 7).attr("cy", 7).attr("r", 6);
}, "lollipop");
var point = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("marker").attr("id", id + "_" + type + "-pointEnd").attr("class", "marker " + type).attr("viewBox", "0 0 10 10").attr("refX", 6).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 12).attr("markerHeight", 12).attr("orient", "auto").append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
  elem.append("marker").attr("id", id + "_" + type + "-pointStart").attr("class", "marker " + type).attr("viewBox", "0 0 10 10").attr("refX", 4.5).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 12).attr("markerHeight", 12).attr("orient", "auto").append("path").attr("d", "M 0 5 L 10 10 L 10 0 z").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
}, "point");
var circle = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("marker").attr("id", id + "_" + type + "-circleEnd").attr("class", "marker " + type).attr("viewBox", "0 0 10 10").attr("refX", 11).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("circle").attr("cx", "5").attr("cy", "5").attr("r", "5").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
  elem.append("marker").attr("id", id + "_" + type + "-circleStart").attr("class", "marker " + type).attr("viewBox", "0 0 10 10").attr("refX", -1).attr("refY", 5).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("circle").attr("cx", "5").attr("cy", "5").attr("r", "5").attr("class", "arrowMarkerPath").style("stroke-width", 1).style("stroke-dasharray", "1,0");
}, "circle");
var cross = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("marker").attr("id", id + "_" + type + "-crossEnd").attr("class", "marker cross " + type).attr("viewBox", "0 0 11 11").attr("refX", 12).attr("refY", 5.2).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("path").attr("d", "M 1,1 l 9,9 M 10,1 l -9,9").attr("class", "arrowMarkerPath").style("stroke-width", 2).style("stroke-dasharray", "1,0");
  elem.append("marker").attr("id", id + "_" + type + "-crossStart").attr("class", "marker cross " + type).attr("viewBox", "0 0 11 11").attr("refX", -1).attr("refY", 5.2).attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto").append("path").attr("d", "M 1,1 l 9,9 M 10,1 l -9,9").attr("class", "arrowMarkerPath").style("stroke-width", 2).style("stroke-dasharray", "1,0");
}, "cross");
var barb = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((elem, type, id) => {
  elem.append("defs").append("marker").attr("id", id + "_" + type + "-barbEnd").attr("refX", 19).attr("refY", 7).attr("markerWidth", 20).attr("markerHeight", 14).attr("markerUnits", "strokeWidth").attr("orient", "auto").append("path").attr("d", "M 19,7 L9,13 L14,7 L9,1 Z");
}, "barb");
var markers = {
  extension,
  composition,
  aggregation,
  dependency,
  lollipop,
  point,
  circle,
  cross,
  barb
};
var markers_default = insertMarkers;

// src/diagrams/block/layout.ts
var padding = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)()?.block?.padding ?? 8;
function calculateBlockPosition(columns, position) {
  if (columns === 0 || !Number.isInteger(columns)) {
    throw new Error("Columns must be an integer !== 0.");
  }
  if (position < 0 || !Number.isInteger(position)) {
    throw new Error("Position must be a non-negative integer." + position);
  }
  if (columns < 0) {
    return { px: position, py: 0 };
  }
  if (columns === 1) {
    return { px: 0, py: position };
  }
  const px = position % columns;
  const py = Math.floor(position / columns);
  return { px, py };
}
(0,chunk_AGHRB4JF/* __name */.K2)(calculateBlockPosition, "calculateBlockPosition");
var getMaxChildSize = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((block) => {
  let maxWidth = 0;
  let maxHeight = 0;
  for (const child of block.children) {
    const { width, height, x, y } = child.size ?? { width: 0, height: 0, x: 0, y: 0 };
    chunk_AGHRB4JF/* log */.Rm.debug(
      "getMaxChildSize abc95 child:",
      child.id,
      "width:",
      width,
      "height:",
      height,
      "x:",
      x,
      "y:",
      y,
      child.type
    );
    if (child.type === "space") {
      continue;
    }
    const normalizedWidth = width / (child.widthInColumns ?? 1);
    if (normalizedWidth > maxWidth) {
      maxWidth = normalizedWidth;
    }
    if (height > maxHeight) {
      maxHeight = height;
    }
  }
  return { width: maxWidth, height: maxHeight };
}, "getMaxChildSize");
function setBlockSizes(block, db2, siblingWidth = 0, siblingHeight = 0) {
  chunk_AGHRB4JF/* log */.Rm.debug(
    "setBlockSizes abc95 (start)",
    block.id,
    block?.size?.x,
    "block width =",
    block?.size,
    "siblingWidth",
    siblingWidth
  );
  if (!block?.size?.width) {
    block.size = {
      width: siblingWidth,
      height: siblingHeight,
      x: 0,
      y: 0
    };
  }
  let maxWidth = 0;
  let maxHeight = 0;
  if (block.children?.length > 0) {
    for (const child of block.children) {
      setBlockSizes(child, db2);
    }
    const childSize = getMaxChildSize(block);
    maxWidth = childSize.width;
    maxHeight = childSize.height;
    chunk_AGHRB4JF/* log */.Rm.debug("setBlockSizes abc95 maxWidth of", block.id, ":s children is ", maxWidth, maxHeight);
    for (const child of block.children) {
      if (child.size) {
        chunk_AGHRB4JF/* log */.Rm.debug(
          `abc95 Setting size of children of ${block.id} id=${child.id} ${maxWidth} ${maxHeight} ${JSON.stringify(child.size)}`
        );
        child.size.width = maxWidth * (child.widthInColumns ?? 1) + padding * ((child.widthInColumns ?? 1) - 1);
        child.size.height = maxHeight;
        child.size.x = 0;
        child.size.y = 0;
        chunk_AGHRB4JF/* log */.Rm.debug(
          `abc95 updating size of ${block.id} children child:${child.id} maxWidth:${maxWidth} maxHeight:${maxHeight}`
        );
      }
    }
    for (const child of block.children) {
      setBlockSizes(child, db2, maxWidth, maxHeight);
    }
    const columns = block.columns ?? -1;
    let numItems = 0;
    for (const child of block.children) {
      numItems += child.widthInColumns ?? 1;
    }
    let xSize = block.children.length;
    if (columns > 0 && columns < numItems) {
      xSize = columns;
    }
    const ySize = Math.ceil(numItems / xSize);
    let width = xSize * (maxWidth + padding) + padding;
    let height = ySize * (maxHeight + padding) + padding;
    if (width < siblingWidth) {
      chunk_AGHRB4JF/* log */.Rm.debug(
        `Detected to small sibling: abc95 ${block.id} siblingWidth ${siblingWidth} siblingHeight ${siblingHeight} width ${width}`
      );
      width = siblingWidth;
      height = siblingHeight;
      const childWidth = (siblingWidth - xSize * padding - padding) / xSize;
      const childHeight = (siblingHeight - ySize * padding - padding) / ySize;
      chunk_AGHRB4JF/* log */.Rm.debug("Size indata abc88", block.id, "childWidth", childWidth, "maxWidth", maxWidth);
      chunk_AGHRB4JF/* log */.Rm.debug("Size indata abc88", block.id, "childHeight", childHeight, "maxHeight", maxHeight);
      chunk_AGHRB4JF/* log */.Rm.debug("Size indata abc88 xSize", xSize, "padding", padding);
      for (const child of block.children) {
        if (child.size) {
          child.size.width = childWidth;
          child.size.height = childHeight;
          child.size.x = 0;
          child.size.y = 0;
        }
      }
    }
    chunk_AGHRB4JF/* log */.Rm.debug(
      `abc95 (finale calc) ${block.id} xSize ${xSize} ySize ${ySize} columns ${columns}${block.children.length} width=${Math.max(width, block.size?.width || 0)}`
    );
    if (width < (block?.size?.width || 0)) {
      width = block?.size?.width || 0;
      const num = columns > 0 ? Math.min(block.children.length, columns) : block.children.length;
      if (num > 0) {
        const childWidth = (width - num * padding - padding) / num;
        chunk_AGHRB4JF/* log */.Rm.debug("abc95 (growing to fit) width", block.id, width, block.size?.width, childWidth);
        for (const child of block.children) {
          if (child.size) {
            child.size.width = childWidth;
          }
        }
      }
    }
    block.size = {
      width,
      height,
      x: 0,
      y: 0
    };
  }
  chunk_AGHRB4JF/* log */.Rm.debug(
    "setBlockSizes abc94 (done)",
    block.id,
    block?.size?.x,
    block?.size?.width,
    block?.size?.y,
    block?.size?.height
  );
}
(0,chunk_AGHRB4JF/* __name */.K2)(setBlockSizes, "setBlockSizes");
function layoutBlocks(block, db2) {
  chunk_AGHRB4JF/* log */.Rm.debug(
    `abc85 layout blocks (=>layoutBlocks) ${block.id} x: ${block?.size?.x} y: ${block?.size?.y} width: ${block?.size?.width}`
  );
  const columns = block.columns ?? -1;
  chunk_AGHRB4JF/* log */.Rm.debug("layoutBlocks columns abc95", block.id, "=>", columns, block);
  if (block.children && // find max width of children
  block.children.length > 0) {
    const width = block?.children[0]?.size?.width ?? 0;
    const widthOfChildren = block.children.length * width + (block.children.length - 1) * padding;
    chunk_AGHRB4JF/* log */.Rm.debug("widthOfChildren 88", widthOfChildren, "posX");
    const rowHeights = /* @__PURE__ */ new Map();
    {
      let colPos = 0;
      for (const child of block.children) {
        if (!child.size) {
          continue;
        }
        const { py } = calculateBlockPosition(columns, colPos);
        const currentMax = rowHeights.get(py) ?? 0;
        if (child.size.height > currentMax) {
          rowHeights.set(py, child.size.height);
        }
        let filled = child?.widthInColumns ?? 1;
        if (columns > 0) {
          filled = Math.min(filled, columns - colPos % columns);
        }
        colPos += filled;
      }
    }
    const rowYOffsets = /* @__PURE__ */ new Map();
    {
      let offset = 0;
      const rows = [...rowHeights.keys()].sort((a, b) => a - b);
      for (const row of rows) {
        rowYOffsets.set(row, offset);
        offset += (rowHeights.get(row) ?? 0) + padding;
      }
    }
    let columnPos = 0;
    chunk_AGHRB4JF/* log */.Rm.debug("abc91 block?.size?.x", block.id, block?.size?.x);
    let startingPosX = block?.size?.x ? block?.size?.x + (-block?.size?.width / 2 || 0) : -padding;
    let rowPos = 0;
    for (const child of block.children) {
      const parent = block;
      if (!child.size) {
        continue;
      }
      const { width: width2, height } = child.size;
      const { px, py } = calculateBlockPosition(columns, columnPos);
      if (py != rowPos) {
        rowPos = py;
        startingPosX = block?.size?.x ? block?.size?.x + (-block?.size?.width / 2 || 0) : -padding;
        chunk_AGHRB4JF/* log */.Rm.debug("New row in layout for block", block.id, " and child ", child.id, rowPos);
      }
      chunk_AGHRB4JF/* log */.Rm.debug(
        `abc89 layout blocks (child) id: ${child.id} Pos: ${columnPos} (px, py) ${px},${py} (${parent?.size?.x},${parent?.size?.y}) parent: ${parent.id} width: ${width2}${padding}`
      );
      if (parent.size) {
        const halfWidth = width2 / 2;
        child.size.x = startingPosX + padding + halfWidth;
        chunk_AGHRB4JF/* log */.Rm.debug(
          `abc91 layout blocks (calc) px, pyid:${child.id} startingPos=X${startingPosX} new startingPosX${child.size.x} ${halfWidth} padding=${padding} width=${width2} halfWidth=${halfWidth} => x:${child.size.x} y:${child.size.y} ${child.widthInColumns} (width * (child?.w || 1)) / 2 ${width2 * (child?.widthInColumns ?? 1) / 2}`
        );
        startingPosX = child.size.x + halfWidth;
        const rowYOffset = rowYOffsets.get(py) ?? 0;
        const rowHeight = rowHeights.get(py) ?? height;
        child.size.y = parent.size.y - parent.size.height / 2 + rowYOffset + rowHeight / 2 + padding;
        chunk_AGHRB4JF/* log */.Rm.debug(
          `abc88 layout blocks (calc) px, pyid:${child.id}startingPosX${startingPosX}${padding}${halfWidth}=>x:${child.size.x}y:${child.size.y}${child.widthInColumns}(width * (child?.w || 1)) / 2${width2 * (child?.widthInColumns ?? 1) / 2}`
        );
      }
      if (child.children) {
        layoutBlocks(child, db2);
      }
      let columnsFilled = child?.widthInColumns ?? 1;
      if (columns > 0) {
        columnsFilled = Math.min(columnsFilled, columns - columnPos % columns);
      }
      columnPos += columnsFilled;
      chunk_AGHRB4JF/* log */.Rm.debug("abc88 columnsPos", child, columnPos);
    }
  }
  chunk_AGHRB4JF/* log */.Rm.debug(
    `layout blocks (<==layoutBlocks) ${block.id} x: ${block?.size?.x} y: ${block?.size?.y} width: ${block?.size?.width}`
  );
}
(0,chunk_AGHRB4JF/* __name */.K2)(layoutBlocks, "layoutBlocks");
function findBounds(block, { minX, minY, maxX, maxY } = { minX: 0, minY: 0, maxX: 0, maxY: 0 }) {
  if (block.size && block.id !== "root") {
    const { x, y, width, height } = block.size;
    if (x - width / 2 < minX) {
      minX = x - width / 2;
    }
    if (y - height / 2 < minY) {
      minY = y - height / 2;
    }
    if (x + width / 2 > maxX) {
      maxX = x + width / 2;
    }
    if (y + height / 2 > maxY) {
      maxY = y + height / 2;
    }
  }
  if (block.children) {
    for (const child of block.children) {
      ({ minX, minY, maxX, maxY } = findBounds(child, { minX, minY, maxX, maxY }));
    }
  }
  return { minX, minY, maxX, maxY };
}
(0,chunk_AGHRB4JF/* __name */.K2)(findBounds, "findBounds");
function layout(db2) {
  const root = db2.getBlock("root");
  if (!root) {
    return;
  }
  setBlockSizes(root, db2, 0, 0);
  layoutBlocks(root, db2);
  chunk_AGHRB4JF/* log */.Rm.debug("getBlocks", JSON.stringify(root, null, 2));
  const { minX, minY, maxX, maxY } = findBounds(root);
  const height = maxY - minY;
  const width = maxX - minX;
  return { x: minX, y: minY, width, height };
}
(0,chunk_AGHRB4JF/* __name */.K2)(layout, "layout");

// src/diagrams/block/renderHelpers.ts


// src/dagre-wrapper/createLabel.js
var createLabel = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (element, _vertexText, style, isTitle = false, isNode = false) => {
  let vertexText = _vertexText || "";
  if (typeof vertexText === "object") {
    vertexText = vertexText[0];
  }
  const config2 = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)();
  const useHtmlLabels = (0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)(config2);
  return await (0,chunk_O5CBEL6O/* createText */.GZ)(
    element,
    vertexText,
    {
      style,
      isTitle,
      useHtmlLabels,
      markdown: false,
      isNode,
      width: Number.POSITIVE_INFINITY
    },
    config2
  );
}, "createLabel");
var createLabel_default = createLabel;

// src/dagre-wrapper/edges.js


// src/dagre-wrapper/edgeMarker.ts
var addEdgeMarkers = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((svgPath, edge, url, id, diagramType) => {
  if (edge.arrowTypeStart) {
    addEdgeMarker(svgPath, "start", edge.arrowTypeStart, url, id, diagramType);
  }
  if (edge.arrowTypeEnd) {
    addEdgeMarker(svgPath, "end", edge.arrowTypeEnd, url, id, diagramType);
  }
}, "addEdgeMarkers");
var arrowTypesMap = {
  arrow_cross: "cross",
  arrow_point: "point",
  arrow_barb: "barb",
  arrow_circle: "circle",
  aggregation: "aggregation",
  extension: "extension",
  composition: "composition",
  dependency: "dependency",
  lollipop: "lollipop"
};
var addEdgeMarker = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((svgPath, position, arrowType, url, id, diagramType) => {
  const endMarkerType = arrowTypesMap[arrowType];
  if (!endMarkerType) {
    chunk_AGHRB4JF/* log */.Rm.warn(`Unknown arrow type: ${arrowType}`);
    return;
  }
  const suffix = position === "start" ? "Start" : "End";
  svgPath.attr(`marker-${position}`, `url(${url}#${id}_${diagramType}-${endMarkerType}${suffix})`);
}, "addEdgeMarker");

// src/dagre-wrapper/edges.js
var edgeLabels = {};
var terminalLabels = {};
var insertEdgeLabel = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (elem, edge) => {
  const config2 = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)();
  const useHtmlLabels = (0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)(config2);
  const edgeLabel = elem.insert("g").attr("class", "edgeLabel");
  const label = edgeLabel.insert("g").attr("class", "label");
  const isMarkdown = edge.labelType === "markdown";
  const labelElement = await (0,chunk_O5CBEL6O/* createText */.GZ)(
    elem,
    edge.label,
    {
      style: edge.labelStyle,
      useHtmlLabels,
      // TODO: The old code only set addSvgBackground when using markdown, but
      // this function is only used by block diagrams which never use markdown.
      addSvgBackground: isMarkdown,
      isNode: false,
      markdown: isMarkdown,
      // If using markdown, wrap using default width
      width: isMarkdown ? void 0 : Number.POSITIVE_INFINITY
    },
    config2
  );
  label.node().appendChild(labelElement);
  let bbox = labelElement.getBBox();
  let transformBbox = bbox;
  if (useHtmlLabels) {
    const div = labelElement.children[0];
    const dv = (0,src/* select */.Ltv)(labelElement);
    bbox = div.getBoundingClientRect();
    transformBbox = bbox;
    dv.attr("width", bbox.width);
    dv.attr("height", bbox.height);
  } else {
    const textEl = (0,src/* select */.Ltv)(labelElement).select("text").node();
    if (textEl && typeof textEl.getBBox === "function") {
      transformBbox = textEl.getBBox();
    }
  }
  label.attr("transform", (0,chunk_BSJP7CBP/* computeLabelTransform */.Wg)(transformBbox, useHtmlLabels));
  edgeLabels[edge.id] = edgeLabel;
  edge.width = bbox.width;
  edge.height = bbox.height;
  let fo;
  if (edge.startLabelLeft) {
    const startEdgeLabelLeft = elem.insert("g").attr("class", "edgeTerminals");
    const inner = startEdgeLabelLeft.insert("g").attr("class", "inner");
    const startLabelElement = await createLabel_default(inner, edge.startLabelLeft, edge.labelStyle);
    fo = startLabelElement;
    let slBox = startLabelElement.getBBox();
    if (useHtmlLabels) {
      const div = startLabelElement.children[0];
      const dv = (0,src/* select */.Ltv)(startLabelElement);
      slBox = div.getBoundingClientRect();
      dv.attr("width", slBox.width);
      dv.attr("height", slBox.height);
    }
    inner.attr("transform", (0,chunk_BSJP7CBP/* computeLabelTransform */.Wg)(slBox, useHtmlLabels));
    if (!terminalLabels[edge.id]) {
      terminalLabels[edge.id] = {};
    }
    terminalLabels[edge.id].startLeft = startEdgeLabelLeft;
    setTerminalWidth(fo, edge.startLabelLeft);
  }
  if (edge.startLabelRight) {
    const startEdgeLabelRight = elem.insert("g").attr("class", "edgeTerminals");
    const inner = startEdgeLabelRight.insert("g").attr("class", "inner");
    const startLabelElement = await createLabel_default(inner, edge.startLabelRight, edge.labelStyle);
    fo = startLabelElement;
    let slBox = startLabelElement.getBBox();
    if (useHtmlLabels) {
      const div = startLabelElement.children[0];
      const dv = (0,src/* select */.Ltv)(startLabelElement);
      slBox = div.getBoundingClientRect();
      dv.attr("width", slBox.width);
      dv.attr("height", slBox.height);
    }
    inner.attr("transform", (0,chunk_BSJP7CBP/* computeLabelTransform */.Wg)(slBox, useHtmlLabels));
    if (!terminalLabels[edge.id]) {
      terminalLabels[edge.id] = {};
    }
    terminalLabels[edge.id].startRight = startEdgeLabelRight;
    setTerminalWidth(fo, edge.startLabelRight);
  }
  if (edge.endLabelLeft) {
    const endEdgeLabelLeft = elem.insert("g").attr("class", "edgeTerminals");
    const inner = endEdgeLabelLeft.insert("g").attr("class", "inner");
    const endLabelElement = await createLabel_default(endEdgeLabelLeft, edge.endLabelLeft, edge.labelStyle);
    fo = endLabelElement;
    let slBox = endLabelElement.getBBox();
    if (useHtmlLabels) {
      const div = endLabelElement.children[0];
      const dv = (0,src/* select */.Ltv)(endLabelElement);
      slBox = div.getBoundingClientRect();
      dv.attr("width", slBox.width);
      dv.attr("height", slBox.height);
    }
    inner.attr("transform", (0,chunk_BSJP7CBP/* computeLabelTransform */.Wg)(slBox, useHtmlLabels));
    if (!terminalLabels[edge.id]) {
      terminalLabels[edge.id] = {};
    }
    terminalLabels[edge.id].endLeft = endEdgeLabelLeft;
    setTerminalWidth(fo, edge.endLabelLeft);
  }
  if (edge.endLabelRight) {
    const endEdgeLabelRight = elem.insert("g").attr("class", "edgeTerminals");
    const inner = endEdgeLabelRight.insert("g").attr("class", "inner");
    const endLabelElement = await createLabel_default(
      endEdgeLabelRight,
      edge.endLabelRight,
      edge.labelStyle
    );
    fo = endLabelElement;
    let slBox = endLabelElement.getBBox();
    if (useHtmlLabels) {
      const div = endLabelElement.children[0];
      const dv = (0,src/* select */.Ltv)(endLabelElement);
      slBox = div.getBoundingClientRect();
      dv.attr("width", slBox.width);
      dv.attr("height", slBox.height);
    }
    inner.attr("transform", (0,chunk_BSJP7CBP/* computeLabelTransform */.Wg)(slBox, useHtmlLabels));
    if (!terminalLabels[edge.id]) {
      terminalLabels[edge.id] = {};
    }
    terminalLabels[edge.id].endRight = endEdgeLabelRight;
    setTerminalWidth(fo, edge.endLabelRight);
  }
  return labelElement;
}, "insertEdgeLabel");
function setTerminalWidth(fo, value) {
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)()) && fo) {
    fo.style.width = value.length * 9 + "px";
    fo.style.height = "12px";
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(setTerminalWidth, "setTerminalWidth");
var positionEdgeLabel = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((edge, paths) => {
  chunk_AGHRB4JF/* log */.Rm.debug("Moving label abc88 ", edge.id, edge.label, edgeLabels[edge.id], paths);
  let path = paths.updatedPath ? paths.updatedPath : paths.originalPath;
  const siteConfig = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)();
  const { subGraphTitleTotalMargin } = (0,chunk_L5ZTLDWV/* getSubGraphTitleMargins */.O)(siteConfig);
  if (edge.label) {
    const el = edgeLabels[edge.id];
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = chunk_5ZQYHXKU/* utils_default */._K.calcLabelPosition(path);
      chunk_AGHRB4JF/* log */.Rm.debug(
        "Moving label " + edge.label + " from (",
        x,
        ",",
        y,
        ") to (",
        pos.x,
        ",",
        pos.y,
        ") abc88"
      );
      if (paths.updatedPath) {
        x = pos.x;
        y = pos.y;
      }
    }
    el.attr("transform", `translate(${x}, ${y + subGraphTitleTotalMargin / 2})`);
  }
  if (edge.startLabelLeft) {
    const el = terminalLabels[edge.id].startLeft;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = chunk_5ZQYHXKU/* utils_default */._K.calcTerminalLabelPosition(edge.arrowTypeStart ? 10 : 0, "start_left", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
  if (edge.startLabelRight) {
    const el = terminalLabels[edge.id].startRight;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = chunk_5ZQYHXKU/* utils_default */._K.calcTerminalLabelPosition(
        edge.arrowTypeStart ? 10 : 0,
        "start_right",
        path
      );
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
  if (edge.endLabelLeft) {
    const el = terminalLabels[edge.id].endLeft;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = chunk_5ZQYHXKU/* utils_default */._K.calcTerminalLabelPosition(edge.arrowTypeEnd ? 10 : 0, "end_left", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
  if (edge.endLabelRight) {
    const el = terminalLabels[edge.id].endRight;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = chunk_5ZQYHXKU/* utils_default */._K.calcTerminalLabelPosition(edge.arrowTypeEnd ? 10 : 0, "end_right", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
}, "positionEdgeLabel");
var outsideNode = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node, point2) => {
  const x = node.x;
  const y = node.y;
  const dx = Math.abs(point2.x - x);
  const dy = Math.abs(point2.y - y);
  const w = node.width / 2;
  const h = node.height / 2;
  if (dx >= w || dy >= h) {
    return true;
  }
  return false;
}, "outsideNode");
var intersection = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node, outsidePoint, insidePoint) => {
  chunk_AGHRB4JF/* log */.Rm.debug(`intersection calc abc89:
  outsidePoint: ${JSON.stringify(outsidePoint)}
  insidePoint : ${JSON.stringify(insidePoint)}
  node        : x:${node.x} y:${node.y} w:${node.width} h:${node.height}`);
  const x = node.x;
  const y = node.y;
  const dx = Math.abs(x - insidePoint.x);
  const w = node.width / 2;
  let r = insidePoint.x < outsidePoint.x ? w - dx : w + dx;
  const h = node.height / 2;
  const Q = Math.abs(outsidePoint.y - insidePoint.y);
  const R = Math.abs(outsidePoint.x - insidePoint.x);
  if (Math.abs(y - outsidePoint.y) * w > Math.abs(x - outsidePoint.x) * h) {
    let q = insidePoint.y < outsidePoint.y ? outsidePoint.y - h - y : y - h - outsidePoint.y;
    r = R * q / Q;
    const res = {
      x: insidePoint.x < outsidePoint.x ? insidePoint.x + r : insidePoint.x - R + r,
      y: insidePoint.y < outsidePoint.y ? insidePoint.y + Q - q : insidePoint.y - Q + q
    };
    if (r === 0) {
      res.x = outsidePoint.x;
      res.y = outsidePoint.y;
    }
    if (R === 0) {
      res.x = outsidePoint.x;
    }
    if (Q === 0) {
      res.y = outsidePoint.y;
    }
    chunk_AGHRB4JF/* log */.Rm.debug(`abc89 topp/bott calc, Q ${Q}, q ${q}, R ${R}, r ${r}`, res);
    return res;
  } else {
    if (insidePoint.x < outsidePoint.x) {
      r = outsidePoint.x - w - x;
    } else {
      r = x - w - outsidePoint.x;
    }
    let q = Q * r / R;
    let _x = insidePoint.x < outsidePoint.x ? insidePoint.x + R - r : insidePoint.x - R + r;
    let _y = insidePoint.y < outsidePoint.y ? insidePoint.y + q : insidePoint.y - q;
    chunk_AGHRB4JF/* log */.Rm.debug(`sides calc abc89, Q ${Q}, q ${q}, R ${R}, r ${r}`, { _x, _y });
    if (r === 0) {
      _x = outsidePoint.x;
      _y = outsidePoint.y;
    }
    if (R === 0) {
      _x = outsidePoint.x;
    }
    if (Q === 0) {
      _y = outsidePoint.y;
    }
    return { x: _x, y: _y };
  }
}, "intersection");
var cutPathAtIntersect = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((_points, boundaryNode) => {
  chunk_AGHRB4JF/* log */.Rm.debug("abc88 cutPathAtIntersect", _points, boundaryNode);
  let points = [];
  let lastPointOutside = _points[0];
  let isInside = false;
  _points.forEach((point2) => {
    if (!outsideNode(boundaryNode, point2) && !isInside) {
      const inter = intersection(boundaryNode, lastPointOutside, point2);
      let pointPresent = false;
      points.forEach((p) => {
        pointPresent = pointPresent || p.x === inter.x && p.y === inter.y;
      });
      if (!points.some((e) => e.x === inter.x && e.y === inter.y)) {
        points.push(inter);
      }
      isInside = true;
    } else {
      lastPointOutside = point2;
      if (!isInside) {
        points.push(point2);
      }
    }
  });
  return points;
}, "cutPathAtIntersect");
var insertEdge = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(elem, e, edge, clusterDb, diagramType, graph, id) {
  let points = edge.points;
  chunk_AGHRB4JF/* log */.Rm.debug("abc88 InsertEdge: edge=", edge, "e=", e);
  let pointsHasChanged = false;
  const tail = graph.node(e.v);
  var head = graph.node(e.w);
  if (head?.intersect && tail?.intersect) {
    points = points.slice(1, edge.points.length - 1);
    points.unshift(tail.intersect(points[0]));
    points.push(head.intersect(points[points.length - 1]));
  }
  if (edge.toCluster) {
    chunk_AGHRB4JF/* log */.Rm.debug("to cluster abc88", clusterDb[edge.toCluster]);
    points = cutPathAtIntersect(edge.points, clusterDb[edge.toCluster].node);
    pointsHasChanged = true;
  }
  if (edge.fromCluster) {
    chunk_AGHRB4JF/* log */.Rm.debug("from cluster abc88", clusterDb[edge.fromCluster]);
    points = cutPathAtIntersect(points.reverse(), clusterDb[edge.fromCluster].node).reverse();
    pointsHasChanged = true;
  }
  const lineData = points.filter((p) => !Number.isNaN(p.y));
  let curve = src/* curveBasis */.qrM;
  if (edge.curve && (diagramType === "graph" || diagramType === "flowchart")) {
    curve = edge.curve;
  }
  const { x, y } = (0,chunk_BSJP7CBP/* getLineFunctionsWithOffset */.RI)(edge);
  const lineFunction = (0,src/* line */.n8j)().x(x).y(y).curve(curve);
  let strokeClasses;
  switch (edge.thickness) {
    case "normal":
      strokeClasses = "edge-thickness-normal";
      break;
    case "thick":
      strokeClasses = "edge-thickness-thick";
      break;
    case "invisible":
      strokeClasses = "edge-thickness-thick";
      break;
    default:
      strokeClasses = "";
  }
  switch (edge.pattern) {
    case "solid":
      strokeClasses += " edge-pattern-solid";
      break;
    case "dotted":
      strokeClasses += " edge-pattern-dotted";
      break;
    case "dashed":
      strokeClasses += " edge-pattern-dashed";
      break;
  }
  const svgPath = elem.append("path").attr("d", lineFunction(lineData)).attr("id", edge.id).attr("class", " " + strokeClasses + (edge.classes ? " " + edge.classes : "")).attr("style", edge.style);
  let url = "";
  if ((0,chunk_CSCIHK7Q/* getConfig2 */.D7)().flowchart.arrowMarkerAbsolute || (0,chunk_CSCIHK7Q/* getConfig2 */.D7)().state.arrowMarkerAbsolute) {
    url = (0,chunk_CSCIHK7Q/* getUrl */.ID)(true);
  }
  addEdgeMarkers(svgPath, edge, url, id, diagramType);
  let paths = {};
  if (pointsHasChanged) {
    paths.updatedPath = points;
  }
  paths.originalPath = edge.points;
  return paths;
}, "insertEdge");

// src/dagre-wrapper/nodes.js


// src/dagre-wrapper/blockArrowHelper.ts
var expandAndDeduplicateDirections = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((directions) => {
  const uniqueDirections = /* @__PURE__ */ new Set();
  for (const direction of directions) {
    switch (direction) {
      case "x":
        uniqueDirections.add("right");
        uniqueDirections.add("left");
        break;
      case "y":
        uniqueDirections.add("up");
        uniqueDirections.add("down");
        break;
      default:
        uniqueDirections.add(direction);
        break;
    }
  }
  return uniqueDirections;
}, "expandAndDeduplicateDirections");
var getArrowPoints = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((duplicatedDirections, bbox, node, totalWidth) => {
  const directions = expandAndDeduplicateDirections(duplicatedDirections);
  const f = 2;
  const height = bbox.height + 2 * node.padding;
  const midpoint = height / f;
  const width = totalWidth ?? bbox.width + 2 * midpoint + node.padding;
  const padding2 = node.padding / 2;
  if (directions.has("right") && directions.has("left") && directions.has("up") && directions.has("down")) {
    return [
      // Bottom
      { x: 0, y: 0 },
      { x: midpoint, y: 0 },
      { x: width / 2, y: 2 * padding2 },
      { x: width - midpoint, y: 0 },
      { x: width, y: 0 },
      // Right
      { x: width, y: -height / 3 },
      { x: width + 2 * padding2, y: -height / 2 },
      { x: width, y: -2 * height / 3 },
      { x: width, y: -height },
      // Top
      { x: width - midpoint, y: -height },
      { x: width / 2, y: -height - 2 * padding2 },
      { x: midpoint, y: -height },
      // Left
      { x: 0, y: -height },
      { x: 0, y: -2 * height / 3 },
      { x: -2 * padding2, y: -height / 2 },
      { x: 0, y: -height / 3 }
    ];
  }
  if (directions.has("right") && directions.has("left") && directions.has("up")) {
    return [
      { x: midpoint, y: 0 },
      { x: width - midpoint, y: 0 },
      { x: width, y: -height / 2 },
      { x: width - midpoint, y: -height },
      { x: midpoint, y: -height },
      { x: 0, y: -height / 2 }
    ];
  }
  if (directions.has("right") && directions.has("left") && directions.has("down")) {
    return [
      { x: 0, y: 0 },
      { x: midpoint, y: -height },
      { x: width - midpoint, y: -height },
      { x: width, y: 0 }
    ];
  }
  if (directions.has("right") && directions.has("up") && directions.has("down")) {
    return [
      { x: 0, y: 0 },
      { x: width, y: -midpoint },
      { x: width, y: -height + midpoint },
      { x: 0, y: -height }
    ];
  }
  if (directions.has("left") && directions.has("up") && directions.has("down")) {
    return [
      { x: width, y: 0 },
      { x: 0, y: -midpoint },
      { x: 0, y: -height + midpoint },
      { x: width, y: -height }
    ];
  }
  if (directions.has("right") && directions.has("left")) {
    return [
      { x: midpoint, y: 0 },
      { x: midpoint, y: -padding2 },
      { x: width - midpoint, y: -padding2 },
      { x: width - midpoint, y: 0 },
      { x: width, y: -height / 2 },
      { x: width - midpoint, y: -height },
      { x: width - midpoint, y: -height + padding2 },
      { x: midpoint, y: -height + padding2 },
      { x: midpoint, y: -height },
      { x: 0, y: -height / 2 }
    ];
  }
  if (directions.has("up") && directions.has("down")) {
    return [
      // Bottom center
      { x: width / 2, y: 0 },
      // Left pont of bottom arrow
      { x: 0, y: -padding2 },
      { x: midpoint, y: -padding2 },
      // Left top over vertical section
      { x: midpoint, y: -height + padding2 },
      { x: 0, y: -height + padding2 },
      // Top of arrow
      { x: width / 2, y: -height },
      { x: width, y: -height + padding2 },
      // Top of right vertical bar
      { x: width - midpoint, y: -height + padding2 },
      { x: width - midpoint, y: -padding2 },
      { x: width, y: -padding2 }
    ];
  }
  if (directions.has("right") && directions.has("up")) {
    return [
      { x: 0, y: 0 },
      { x: width, y: -midpoint },
      { x: 0, y: -height }
    ];
  }
  if (directions.has("right") && directions.has("down")) {
    return [
      { x: 0, y: 0 },
      { x: width, y: 0 },
      { x: 0, y: -height }
    ];
  }
  if (directions.has("left") && directions.has("up")) {
    return [
      { x: width, y: 0 },
      { x: 0, y: -midpoint },
      { x: width, y: -height }
    ];
  }
  if (directions.has("left") && directions.has("down")) {
    return [
      { x: width, y: 0 },
      { x: 0, y: 0 },
      { x: width, y: -height }
    ];
  }
  if (directions.has("right")) {
    return [
      { x: midpoint, y: -padding2 },
      { x: midpoint, y: -padding2 },
      { x: width - midpoint, y: -padding2 },
      { x: width - midpoint, y: 0 },
      { x: width, y: -height / 2 },
      { x: width - midpoint, y: -height },
      { x: width - midpoint, y: -height + padding2 },
      // top left corner of arrow
      { x: midpoint, y: -height + padding2 },
      { x: midpoint, y: -height + padding2 }
    ];
  }
  if (directions.has("left")) {
    return [
      { x: midpoint, y: 0 },
      { x: midpoint, y: -padding2 },
      // Two points, the right corners
      { x: width - midpoint, y: -padding2 },
      { x: width - midpoint, y: -height + padding2 },
      { x: midpoint, y: -height + padding2 },
      { x: midpoint, y: -height },
      { x: 0, y: -height / 2 }
    ];
  }
  if (directions.has("up")) {
    return [
      // Bottom center
      { x: midpoint, y: -padding2 },
      // Left top over vertical section
      { x: midpoint, y: -height + padding2 },
      { x: 0, y: -height + padding2 },
      // Top of arrow
      { x: width / 2, y: -height },
      { x: width, y: -height + padding2 },
      // Top of right vertical bar
      { x: width - midpoint, y: -height + padding2 },
      { x: width - midpoint, y: -padding2 }
    ];
  }
  if (directions.has("down")) {
    return [
      // Bottom center
      { x: width / 2, y: 0 },
      // Left pont of bottom arrow
      { x: 0, y: -padding2 },
      { x: midpoint, y: -padding2 },
      // Left top over vertical section
      { x: midpoint, y: -height + padding2 },
      { x: width - midpoint, y: -height + padding2 },
      { x: width - midpoint, y: -padding2 },
      { x: width, y: -padding2 }
    ];
  }
  return [{ x: 0, y: 0 }];
}, "getArrowPoints");

// src/dagre-wrapper/intersect/intersect-node.js
function intersectNode(node, point2) {
  return node.intersect(point2);
}
(0,chunk_AGHRB4JF/* __name */.K2)(intersectNode, "intersectNode");
var intersect_node_default = intersectNode;

// src/dagre-wrapper/intersect/intersect-ellipse.js
function intersectEllipse(node, rx, ry, point2) {
  var cx = node.x;
  var cy = node.y;
  var px = cx - point2.x;
  var py = cy - point2.y;
  var det = Math.sqrt(rx * rx * py * py + ry * ry * px * px);
  var dx = Math.abs(rx * ry * px / det);
  if (point2.x < cx) {
    dx = -dx;
  }
  var dy = Math.abs(rx * ry * py / det);
  if (point2.y < cy) {
    dy = -dy;
  }
  return { x: cx + dx, y: cy + dy };
}
(0,chunk_AGHRB4JF/* __name */.K2)(intersectEllipse, "intersectEllipse");
var intersect_ellipse_default = intersectEllipse;

// src/dagre-wrapper/intersect/intersect-circle.js
function intersectCircle(node, rx, point2) {
  return intersect_ellipse_default(node, rx, rx, point2);
}
(0,chunk_AGHRB4JF/* __name */.K2)(intersectCircle, "intersectCircle");
var intersect_circle_default = intersectCircle;

// src/dagre-wrapper/intersect/intersect-line.js
function intersectLine(p1, p2, q1, q2) {
  var a1, a2, b1, b2, c1, c2;
  var r1, r2, r3, r4;
  var denom, offset, num;
  var x, y;
  a1 = p2.y - p1.y;
  b1 = p1.x - p2.x;
  c1 = p2.x * p1.y - p1.x * p2.y;
  r3 = a1 * q1.x + b1 * q1.y + c1;
  r4 = a1 * q2.x + b1 * q2.y + c1;
  if (r3 !== 0 && r4 !== 0 && sameSign(r3, r4)) {
    return;
  }
  a2 = q2.y - q1.y;
  b2 = q1.x - q2.x;
  c2 = q2.x * q1.y - q1.x * q2.y;
  r1 = a2 * p1.x + b2 * p1.y + c2;
  r2 = a2 * p2.x + b2 * p2.y + c2;
  if (r1 !== 0 && r2 !== 0 && sameSign(r1, r2)) {
    return;
  }
  denom = a1 * b2 - a2 * b1;
  if (denom === 0) {
    return;
  }
  offset = Math.abs(denom / 2);
  num = b1 * c2 - b2 * c1;
  x = num < 0 ? (num - offset) / denom : (num + offset) / denom;
  num = a2 * c1 - a1 * c2;
  y = num < 0 ? (num - offset) / denom : (num + offset) / denom;
  return { x, y };
}
(0,chunk_AGHRB4JF/* __name */.K2)(intersectLine, "intersectLine");
function sameSign(r1, r2) {
  return r1 * r2 > 0;
}
(0,chunk_AGHRB4JF/* __name */.K2)(sameSign, "sameSign");
var intersect_line_default = intersectLine;

// src/dagre-wrapper/intersect/intersect-polygon.js
var intersect_polygon_default = intersectPolygon;
function intersectPolygon(node, polyPoints, point2) {
  var x1 = node.x;
  var y1 = node.y;
  var intersections = [];
  var minX = Number.POSITIVE_INFINITY;
  var minY = Number.POSITIVE_INFINITY;
  if (typeof polyPoints.forEach === "function") {
    polyPoints.forEach(function(entry) {
      minX = Math.min(minX, entry.x);
      minY = Math.min(minY, entry.y);
    });
  } else {
    minX = Math.min(minX, polyPoints.x);
    minY = Math.min(minY, polyPoints.y);
  }
  var left = x1 - node.width / 2 - minX;
  var top = y1 - node.height / 2 - minY;
  for (var i = 0; i < polyPoints.length; i++) {
    var p1 = polyPoints[i];
    var p2 = polyPoints[i < polyPoints.length - 1 ? i + 1 : 0];
    var intersect = intersect_line_default(
      node,
      point2,
      { x: left + p1.x, y: top + p1.y },
      { x: left + p2.x, y: top + p2.y }
    );
    if (intersect) {
      intersections.push(intersect);
    }
  }
  if (!intersections.length) {
    return node;
  }
  if (intersections.length > 1) {
    intersections.sort(function(p, q) {
      var pdx = p.x - point2.x;
      var pdy = p.y - point2.y;
      var distp = Math.sqrt(pdx * pdx + pdy * pdy);
      var qdx = q.x - point2.x;
      var qdy = q.y - point2.y;
      var distq = Math.sqrt(qdx * qdx + qdy * qdy);
      return distp < distq ? -1 : distp === distq ? 0 : 1;
    });
  }
  return intersections[0];
}
(0,chunk_AGHRB4JF/* __name */.K2)(intersectPolygon, "intersectPolygon");

// src/dagre-wrapper/intersect/intersect-rect.js
var intersectRect = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node, point2) => {
  var x = node.x;
  var y = node.y;
  var dx = point2.x - x;
  var dy = point2.y - y;
  var w = node.width / 2;
  var h = node.height / 2;
  var sx, sy;
  if (Math.abs(dy) * w > Math.abs(dx) * h) {
    if (dy < 0) {
      h = -h;
    }
    sx = dy === 0 ? 0 : h * dx / dy;
    sy = h;
  } else {
    if (dx < 0) {
      w = -w;
    }
    sx = w;
    sy = dx === 0 ? 0 : w * dy / dx;
  }
  return { x: x + sx, y: y + sy };
}, "intersectRect");
var intersect_rect_default = intersectRect;

// src/dagre-wrapper/intersect/index.js
var intersect_default = {
  node: intersect_node_default,
  circle: intersect_circle_default,
  ellipse: intersect_ellipse_default,
  polygon: intersect_polygon_default,
  rect: intersect_rect_default
};

// src/dagre-wrapper/shapes/util.js

var labelHelper = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node, _classes, isNode) => {
  const config2 = (0,chunk_CSCIHK7Q/* getConfig2 */.D7)();
  let classes2;
  const useHtmlLabels = node.useHtmlLabels || (0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)(config2);
  if (!_classes) {
    classes2 = "node default";
  } else {
    classes2 = _classes;
  }
  const shapeSvg = parent.insert("g").attr("class", classes2).attr("id", node.domId || node.id);
  const label = shapeSvg.insert("g").attr("class", "label").attr("style", node.labelStyle);
  let labelText;
  if (node.labelText === void 0) {
    labelText = "";
  } else {
    labelText = typeof node.labelText === "string" ? node.labelText : node.labelText[0];
  }
  let text;
  if (node.labelType === "markdown") {
    text = (0,chunk_O5CBEL6O/* createText */.GZ)(
      label,
      (0,chunk_CSCIHK7Q/* sanitizeText */.jZ)((0,chunk_5ZQYHXKU/* decodeEntities */.Sm)(labelText), config2),
      {
        useHtmlLabels,
        width: node.width || config2.flowchart.wrappingWidth,
        classes: "markdown-node-label"
      },
      config2
    );
  } else {
    text = await createLabel_default(
      label,
      (0,chunk_CSCIHK7Q/* sanitizeText */.jZ)((0,chunk_5ZQYHXKU/* decodeEntities */.Sm)(labelText), config2),
      node.labelStyle,
      false,
      isNode
    );
  }
  let bbox = text.getBBox();
  const halfPadding = node.padding / 2;
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)(config2)) {
    const div = text.children[0];
    const dv = (0,src/* select */.Ltv)(text);
    await (0,chunk_L5ZTLDWV/* configureLabelImages */.X)(div, labelText);
    bbox = div.getBoundingClientRect();
    dv.attr("width", bbox.width);
    dv.attr("height", bbox.height);
  }
  if (useHtmlLabels) {
    label.attr("transform", "translate(" + -bbox.width / 2 + ", " + -bbox.height / 2 + ")");
  } else {
    label.attr("transform", "translate(0, " + -bbox.height / 2 + ")");
  }
  if (node.centerLabel) {
    label.attr("transform", "translate(" + -bbox.width / 2 + ", " + -bbox.height / 2 + ")");
  }
  label.insert("rect", ":first-child");
  return { shapeSvg, bbox, halfPadding, label };
}, "labelHelper");
var updateNodeBounds = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node, element) => {
  const bbox = element.node().getBBox();
  node.width = bbox.width;
  node.height = bbox.height;
}, "updateNodeBounds");
function insertPolygonShape(parent, w, h, points) {
  return parent.insert("polygon", ":first-child").attr(
    "points",
    points.map(function(d) {
      return d.x + "," + d.y;
    }).join(" ")
  ).attr("class", "label-container").attr("transform", "translate(" + -w / 2 + "," + h / 2 + ")");
}
(0,chunk_AGHRB4JF/* __name */.K2)(insertPolygonShape, "insertPolygonShape");

// src/dagre-wrapper/shapes/note.js
var note = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const useHtmlLabels = node.useHtmlLabels || (0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)());
  if (!useHtmlLabels) {
    node.centerLabel = true;
  }
  const { shapeSvg, bbox, halfPadding } = await labelHelper(
    parent,
    node,
    "node " + node.classes,
    true
  );
  chunk_AGHRB4JF/* log */.Rm.info("Classes = ", node.classes);
  const rect2 = shapeSvg.insert("rect", ":first-child");
  rect2.attr("rx", node.rx).attr("ry", node.ry).attr("x", -bbox.width / 2 - halfPadding).attr("y", -bbox.height / 2 - halfPadding).attr("width", bbox.width + node.padding).attr("height", bbox.height + node.padding);
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "note");
var note_default = note;

// src/dagre-wrapper/nodes.js
var formatClass = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((str) => {
  if (str) {
    return " " + str;
  }
  return "";
}, "formatClass");
var getClassesFromNode = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node, otherClasses) => {
  return `${otherClasses ? otherClasses : "node default"}${formatClass(node.classes)} ${formatClass(
    node.class
  )}`;
}, "getClassesFromNode");
var question = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const s = w + h;
  const points = [
    { x: s / 2, y: 0 },
    { x: s, y: -s / 2 },
    { x: s / 2, y: -s },
    { x: 0, y: -s / 2 }
  ];
  chunk_AGHRB4JF/* log */.Rm.info("Question main (Circle)");
  const questionElem = insertPolygonShape(shapeSvg, s, s, points);
  questionElem.attr("style", node.style);
  updateNodeBounds(node, questionElem);
  node.intersect = function(point2) {
    chunk_AGHRB4JF/* log */.Rm.warn("Intersect called");
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "question");
var choice = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((parent, node) => {
  const shapeSvg = parent.insert("g").attr("class", "node default").attr("id", node.domId || node.id);
  const s = 28;
  const points = [
    { x: 0, y: s / 2 },
    { x: s / 2, y: 0 },
    { x: 0, y: -s / 2 },
    { x: -s / 2, y: 0 }
  ];
  const choice2 = shapeSvg.insert("polygon", ":first-child").attr(
    "points",
    points.map(function(d) {
      return d.x + "," + d.y;
    }).join(" ")
  );
  choice2.attr("class", "state-start").attr("r", 7).attr("width", 28).attr("height", 28);
  node.width = 28;
  node.height = 28;
  node.intersect = function(point2) {
    return intersect_default.circle(node, 14, point2);
  };
  return shapeSvg;
}, "choice");
var hexagon = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const f = 4;
  const h = node.positioned ? node.height : bbox.height + node.padding;
  const m = h / f;
  const w = node.positioned ? node.width : bbox.width + 2 * m + node.padding;
  const points = [
    { x: m, y: 0 },
    { x: w - m, y: 0 },
    { x: w, y: -h / 2 },
    { x: w - m, y: -h },
    { x: m, y: -h },
    { x: 0, y: -h / 2 }
  ];
  const hex = insertPolygonShape(shapeSvg, w, h, points);
  hex.attr("style", node.style);
  updateNodeBounds(node, hex);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "hexagon");
var block_arrow = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(parent, node, void 0, true);
  const f = 2;
  const h = bbox.height + 2 * node.padding;
  const m = h / f;
  const naturalW = bbox.width + 2 * m + node.padding;
  const isSpanning = node.positioned && (node.widthInColumns ?? 1) > 1 && node.width > naturalW;
  const w = isSpanning ? node.width : naturalW;
  const points = getArrowPoints(node.directions, bbox, node, w);
  const blockArrow = insertPolygonShape(shapeSvg, w, h, points);
  blockArrow.attr("style", node.style);
  updateNodeBounds(node, blockArrow);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "block_arrow");
var rect_left_inv_arrow = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: -h / 2, y: 0 },
    { x: w, y: 0 },
    { x: w, y: -h },
    { x: -h / 2, y: -h },
    { x: 0, y: -h / 2 }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  node.width = w + h;
  node.height = h;
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "rect_left_inv_arrow");
var lean_right = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(parent, node, getClassesFromNode(node), true);
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: -2 * h / 6, y: 0 },
    { x: w - h / 6, y: 0 },
    { x: w + 2 * h / 6, y: -h },
    { x: h / 6, y: -h }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "lean_right");
var lean_left = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: 2 * h / 6, y: 0 },
    { x: w + h / 6, y: 0 },
    { x: w - 2 * h / 6, y: -h },
    { x: -h / 6, y: -h }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "lean_left");
var trapezoid = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: -2 * h / 6, y: 0 },
    { x: w + 2 * h / 6, y: 0 },
    { x: w - h / 6, y: -h },
    { x: h / 6, y: -h }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "trapezoid");
var inv_trapezoid = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: h / 6, y: 0 },
    { x: w - h / 6, y: 0 },
    { x: w + 2 * h / 6, y: -h },
    { x: -2 * h / 6, y: -h }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "inv_trapezoid");
var rect_right_inv_arrow = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: 0, y: 0 },
    { x: w + h / 2, y: 0 },
    { x: w, y: -h / 2 },
    { x: w + h / 2, y: -h },
    { x: 0, y: -h }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "rect_right_inv_arrow");
var cylinder = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const rx = w / 2;
  const ry = rx / (2.5 + w / 50);
  const h = bbox.height + ry + node.padding;
  const shape = "M 0," + ry + " a " + rx + "," + ry + " 0,0,0 " + w + " 0 a " + rx + "," + ry + " 0,0,0 " + -w + " 0 l 0," + h + " a " + rx + "," + ry + " 0,0,0 " + w + " 0 l 0," + -h;
  const el = shapeSvg.attr("label-offset-y", ry).insert("path", ":first-child").attr("style", node.style).attr("d", shape).attr("transform", "translate(" + -w / 2 + "," + -(h / 2 + ry) + ")");
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    const pos = intersect_default.rect(node, point2);
    const x = pos.x - node.x;
    if (rx != 0 && (Math.abs(x) < node.width / 2 || Math.abs(x) == node.width / 2 && Math.abs(pos.y - node.y) > node.height / 2 - ry)) {
      let y = ry * ry * (1 - x * x / (rx * rx));
      if (y != 0) {
        y = Math.sqrt(y);
      }
      y = ry - y;
      if (point2.y - node.y > 0) {
        y = -y;
      }
      pos.y += y;
    }
    return pos;
  };
  return shapeSvg;
}, "cylinder");
var rect = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox, halfPadding } = await labelHelper(
    parent,
    node,
    "node " + node.classes + " " + node.class,
    true
  );
  const rect2 = shapeSvg.insert("rect", ":first-child");
  const totalWidth = node.positioned ? node.width : bbox.width + node.padding;
  const totalHeight = node.positioned ? node.height : bbox.height + node.padding;
  const x = node.positioned ? -totalWidth / 2 : -bbox.width / 2 - halfPadding;
  const y = node.positioned ? -totalHeight / 2 : -bbox.height / 2 - halfPadding;
  rect2.attr("class", "basic label-container").attr("style", node.style).attr("rx", node.rx).attr("ry", node.ry).attr("x", x).attr("y", y).attr("width", totalWidth).attr("height", totalHeight);
  if (node.props) {
    const propKeys = new Set(Object.keys(node.props));
    if (node.props.borders) {
      applyNodePropertyBorders(rect2, node.props.borders, totalWidth, totalHeight);
      propKeys.delete("borders");
    }
    propKeys.forEach((propKey) => {
      chunk_AGHRB4JF/* log */.Rm.warn(`Unknown node property ${propKey}`);
    });
  }
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "rect");
var composite = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox, halfPadding } = await labelHelper(
    parent,
    node,
    "node " + node.classes,
    true
  );
  const rect2 = shapeSvg.insert("rect", ":first-child");
  const totalWidth = node.positioned ? node.width : bbox.width + node.padding;
  const totalHeight = node.positioned ? node.height : bbox.height + node.padding;
  const x = node.positioned ? -totalWidth / 2 : -bbox.width / 2 - halfPadding;
  const y = node.positioned ? -totalHeight / 2 : -bbox.height / 2 - halfPadding;
  rect2.attr("class", "basic cluster composite label-container").attr("style", node.style).attr("rx", node.rx).attr("ry", node.ry).attr("x", x).attr("y", y).attr("width", totalWidth).attr("height", totalHeight);
  if (node.props) {
    const propKeys = new Set(Object.keys(node.props));
    if (node.props.borders) {
      applyNodePropertyBorders(rect2, node.props.borders, totalWidth, totalHeight);
      propKeys.delete("borders");
    }
    propKeys.forEach((propKey) => {
      chunk_AGHRB4JF/* log */.Rm.warn(`Unknown node property ${propKey}`);
    });
  }
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "composite");
var labelRect = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg } = await labelHelper(parent, node, "label", true);
  chunk_AGHRB4JF/* log */.Rm.trace("Classes = ", node.class);
  const rect2 = shapeSvg.insert("rect", ":first-child");
  const totalWidth = 0;
  const totalHeight = 0;
  rect2.attr("width", totalWidth).attr("height", totalHeight);
  shapeSvg.attr("class", "label edgeLabel");
  if (node.props) {
    const propKeys = new Set(Object.keys(node.props));
    if (node.props.borders) {
      applyNodePropertyBorders(rect2, node.props.borders, totalWidth, totalHeight);
      propKeys.delete("borders");
    }
    propKeys.forEach((propKey) => {
      chunk_AGHRB4JF/* log */.Rm.warn(`Unknown node property ${propKey}`);
    });
  }
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "labelRect");
function applyNodePropertyBorders(rect2, borders, totalWidth, totalHeight) {
  const strokeDashArray = [];
  const addBorder = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((length) => {
    strokeDashArray.push(length, 0);
  }, "addBorder");
  const skipBorder = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((length) => {
    strokeDashArray.push(0, length);
  }, "skipBorder");
  if (borders.includes("t")) {
    chunk_AGHRB4JF/* log */.Rm.debug("add top border");
    addBorder(totalWidth);
  } else {
    skipBorder(totalWidth);
  }
  if (borders.includes("r")) {
    chunk_AGHRB4JF/* log */.Rm.debug("add right border");
    addBorder(totalHeight);
  } else {
    skipBorder(totalHeight);
  }
  if (borders.includes("b")) {
    chunk_AGHRB4JF/* log */.Rm.debug("add bottom border");
    addBorder(totalWidth);
  } else {
    skipBorder(totalWidth);
  }
  if (borders.includes("l")) {
    chunk_AGHRB4JF/* log */.Rm.debug("add left border");
    addBorder(totalHeight);
  } else {
    skipBorder(totalHeight);
  }
  rect2.attr("stroke-dasharray", strokeDashArray.join(" "));
}
(0,chunk_AGHRB4JF/* __name */.K2)(applyNodePropertyBorders, "applyNodePropertyBorders");
var rectWithTitle = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  let classes2;
  if (!node.classes) {
    classes2 = "node default";
  } else {
    classes2 = "node " + node.classes;
  }
  const shapeSvg = parent.insert("g").attr("class", classes2).attr("id", node.domId || node.id);
  const rect2 = shapeSvg.insert("rect", ":first-child");
  const innerLine = shapeSvg.insert("line");
  const label = shapeSvg.insert("g").attr("class", "label");
  const text2 = node.labelText.flat ? node.labelText.flat() : node.labelText;
  let title = "";
  if (typeof text2 === "object") {
    title = text2[0];
  } else {
    title = text2;
  }
  chunk_AGHRB4JF/* log */.Rm.info("Label text abc79", title, text2, typeof text2 === "object");
  const text = await createLabel_default(label, title, node.labelStyle, true, true);
  let bbox = { width: 0, height: 0 };
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
    const div = text.children[0];
    const dv = (0,src/* select */.Ltv)(text);
    bbox = div.getBoundingClientRect();
    dv.attr("width", bbox.width);
    dv.attr("height", bbox.height);
  }
  chunk_AGHRB4JF/* log */.Rm.info("Text 2", text2);
  const textRows = text2.slice(1, text2.length);
  let titleBox = text.getBBox();
  const descr = await createLabel_default(
    label,
    textRows.join ? textRows.join("<br/>") : textRows,
    node.labelStyle,
    true,
    true
  );
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
    const div = descr.children[0];
    const dv = (0,src/* select */.Ltv)(descr);
    bbox = div.getBoundingClientRect();
    dv.attr("width", bbox.width);
    dv.attr("height", bbox.height);
  }
  const halfPadding = node.padding / 2;
  (0,src/* select */.Ltv)(descr).attr(
    "transform",
    "translate( " + // (titleBox.width - bbox.width) / 2 +
    (bbox.width > titleBox.width ? 0 : (titleBox.width - bbox.width) / 2) + ", " + (titleBox.height + halfPadding + 5) + ")"
  );
  (0,src/* select */.Ltv)(text).attr(
    "transform",
    "translate( " + // (titleBox.width - bbox.width) / 2 +
    (bbox.width < titleBox.width ? 0 : -(titleBox.width - bbox.width) / 2) + ", 0)"
  );
  bbox = label.node().getBBox();
  label.attr(
    "transform",
    "translate(" + -bbox.width / 2 + ", " + (-bbox.height / 2 - halfPadding + 3) + ")"
  );
  rect2.attr("class", "outer title-state").attr("x", -bbox.width / 2 - halfPadding).attr("y", -bbox.height / 2 - halfPadding).attr("width", bbox.width + node.padding).attr("height", bbox.height + node.padding);
  innerLine.attr("class", "divider").attr("x1", -bbox.width / 2 - halfPadding).attr("x2", bbox.width / 2 + halfPadding).attr("y1", -bbox.height / 2 - halfPadding + titleBox.height + halfPadding).attr("y2", -bbox.height / 2 - halfPadding + titleBox.height + halfPadding);
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "rectWithTitle");
var stadium = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const h = bbox.height + node.padding;
  const w = bbox.width + h / 4 + node.padding;
  const rect2 = shapeSvg.insert("rect", ":first-child").attr("style", node.style).attr("rx", h / 2).attr("ry", h / 2).attr("x", -w / 2).attr("y", -h / 2).attr("width", w).attr("height", h);
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "stadium");
var circle2 = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox, halfPadding } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const circle3 = shapeSvg.insert("circle", ":first-child");
  circle3.attr("style", node.style).attr("rx", node.rx).attr("ry", node.ry).attr("r", bbox.width / 2 + halfPadding).attr("width", bbox.width + node.padding).attr("height", bbox.height + node.padding);
  chunk_AGHRB4JF/* log */.Rm.info("Circle main");
  updateNodeBounds(node, circle3);
  node.intersect = function(point2) {
    chunk_AGHRB4JF/* log */.Rm.info("Circle intersect", node, bbox.width / 2 + halfPadding, point2);
    return intersect_default.circle(node, bbox.width / 2 + halfPadding, point2);
  };
  return shapeSvg;
}, "circle");
var doublecircle = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox, halfPadding } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const gap = 5;
  const circleGroup = shapeSvg.insert("g", ":first-child");
  const outerCircle = circleGroup.insert("circle");
  const innerCircle = circleGroup.insert("circle");
  circleGroup.attr("class", node.class);
  outerCircle.attr("style", node.style).attr("rx", node.rx).attr("ry", node.ry).attr("r", bbox.width / 2 + halfPadding + gap).attr("width", bbox.width + node.padding + gap * 2).attr("height", bbox.height + node.padding + gap * 2);
  innerCircle.attr("style", node.style).attr("rx", node.rx).attr("ry", node.ry).attr("r", bbox.width / 2 + halfPadding).attr("width", bbox.width + node.padding).attr("height", bbox.height + node.padding);
  chunk_AGHRB4JF/* log */.Rm.info("DoubleCircle main");
  updateNodeBounds(node, outerCircle);
  node.intersect = function(point2) {
    chunk_AGHRB4JF/* log */.Rm.info("DoubleCircle intersect", node, bbox.width / 2 + halfPadding + gap, point2);
    return intersect_default.circle(node, bbox.width / 2 + halfPadding + gap, point2);
  };
  return shapeSvg;
}, "doublecircle");
var subroutine = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const { shapeSvg, bbox } = await labelHelper(
    parent,
    node,
    getClassesFromNode(node, void 0),
    true
  );
  const w = bbox.width + node.padding;
  const h = bbox.height + node.padding;
  const points = [
    { x: 0, y: 0 },
    { x: w, y: 0 },
    { x: w, y: -h },
    { x: 0, y: -h },
    { x: 0, y: 0 },
    { x: -8, y: 0 },
    { x: w + 8, y: 0 },
    { x: w + 8, y: -h },
    { x: -8, y: -h },
    { x: -8, y: 0 }
  ];
  const el = insertPolygonShape(shapeSvg, w, h, points);
  el.attr("style", node.style);
  updateNodeBounds(node, el);
  node.intersect = function(point2) {
    return intersect_default.polygon(node, points, point2);
  };
  return shapeSvg;
}, "subroutine");
var start = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((parent, node) => {
  const shapeSvg = parent.insert("g").attr("class", "node default").attr("id", node.domId || node.id);
  const circle3 = shapeSvg.insert("circle", ":first-child");
  circle3.attr("class", "state-start").attr("r", 7).attr("width", 14).attr("height", 14);
  updateNodeBounds(node, circle3);
  node.intersect = function(point2) {
    return intersect_default.circle(node, 7, point2);
  };
  return shapeSvg;
}, "start");
var forkJoin = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((parent, node, dir) => {
  const shapeSvg = parent.insert("g").attr("class", "node default").attr("id", node.domId || node.id);
  let width = 70;
  let height = 10;
  if (dir === "LR") {
    width = 10;
    height = 70;
  }
  const shape = shapeSvg.append("rect").attr("x", -1 * width / 2).attr("y", -1 * height / 2).attr("width", width).attr("height", height).attr("class", "fork-join");
  updateNodeBounds(node, shape);
  node.height = node.height + node.padding / 2;
  node.width = node.width + node.padding / 2;
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "forkJoin");
var end = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((parent, node) => {
  const shapeSvg = parent.insert("g").attr("class", "node default").attr("id", node.domId || node.id);
  const innerCircle = shapeSvg.insert("circle", ":first-child");
  const circle3 = shapeSvg.insert("circle", ":first-child");
  circle3.attr("class", "state-start").attr("r", 7).attr("width", 14).attr("height", 14);
  innerCircle.attr("class", "state-end").attr("r", 5).attr("width", 10).attr("height", 10);
  updateNodeBounds(node, circle3);
  node.intersect = function(point2) {
    return intersect_default.circle(node, 7, point2);
  };
  return shapeSvg;
}, "end");
var class_box = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (parent, node) => {
  const halfPadding = node.padding / 2;
  const rowPadding = 4;
  const lineHeight = 8;
  let classes2;
  if (!node.classes) {
    classes2 = "node default";
  } else {
    classes2 = "node " + node.classes;
  }
  const shapeSvg = parent.insert("g").attr("class", classes2).attr("id", node.domId || node.id);
  const rect2 = shapeSvg.insert("rect", ":first-child");
  const topLine = shapeSvg.insert("line");
  const bottomLine = shapeSvg.insert("line");
  let maxWidth = 0;
  let maxHeight = rowPadding;
  const labelContainer = shapeSvg.insert("g").attr("class", "label");
  let verticalPos = 0;
  const hasInterface = node.classData.annotations?.[0];
  const interfaceLabelText = node.classData.annotations[0] ? "\xAB" + node.classData.annotations[0] + "\xBB" : "";
  const interfaceLabel = await createLabel_default(
    labelContainer,
    interfaceLabelText,
    node.labelStyle,
    true,
    true
  );
  let interfaceBBox = interfaceLabel.getBBox();
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
    const div = interfaceLabel.children[0];
    const dv = (0,src/* select */.Ltv)(interfaceLabel);
    interfaceBBox = div.getBoundingClientRect();
    dv.attr("width", interfaceBBox.width);
    dv.attr("height", interfaceBBox.height);
  }
  if (node.classData.annotations[0]) {
    maxHeight += interfaceBBox.height + rowPadding;
    maxWidth += interfaceBBox.width;
  }
  let classTitleString = node.classData.label;
  if (node.classData.type !== void 0 && node.classData.type !== "") {
    if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
      classTitleString += "&lt;" + node.classData.type + "&gt;";
    } else {
      classTitleString += "<" + node.classData.type + ">";
    }
  }
  const classTitleLabel = await createLabel_default(
    labelContainer,
    classTitleString,
    node.labelStyle,
    true,
    true
  );
  (0,src/* select */.Ltv)(classTitleLabel).attr("class", "classTitle");
  let classTitleBBox = classTitleLabel.getBBox();
  if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
    const div = classTitleLabel.children[0];
    const dv = (0,src/* select */.Ltv)(classTitleLabel);
    classTitleBBox = div.getBoundingClientRect();
    dv.attr("width", classTitleBBox.width);
    dv.attr("height", classTitleBBox.height);
  }
  maxHeight += classTitleBBox.height + rowPadding;
  if (classTitleBBox.width > maxWidth) {
    maxWidth = classTitleBBox.width;
  }
  const classAttributes = [];
  node.classData.members.forEach(async (member) => {
    const parsedInfo = member.getDisplayDetails();
    let parsedText = parsedInfo.displayText;
    if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
      parsedText = parsedText.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    const lbl = await createLabel_default(
      labelContainer,
      parsedText,
      parsedInfo.cssStyle ? parsedInfo.cssStyle : node.labelStyle,
      true,
      true
    );
    let bbox = lbl.getBBox();
    if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
      const div = lbl.children[0];
      const dv = (0,src/* select */.Ltv)(lbl);
      bbox = div.getBoundingClientRect();
      dv.attr("width", bbox.width);
      dv.attr("height", bbox.height);
    }
    if (bbox.width > maxWidth) {
      maxWidth = bbox.width;
    }
    maxHeight += bbox.height + rowPadding;
    classAttributes.push(lbl);
  });
  maxHeight += lineHeight;
  const classMethods = [];
  node.classData.methods.forEach(async (member) => {
    const parsedInfo = member.getDisplayDetails();
    let displayText = parsedInfo.displayText;
    if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
      displayText = displayText.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    const lbl = await createLabel_default(
      labelContainer,
      displayText,
      parsedInfo.cssStyle ? parsedInfo.cssStyle : node.labelStyle,
      true,
      true
    );
    let bbox = lbl.getBBox();
    if ((0,chunk_CSCIHK7Q/* getEffectiveHtmlLabels */.E)((0,chunk_CSCIHK7Q/* getConfig2 */.D7)())) {
      const div = lbl.children[0];
      const dv = (0,src/* select */.Ltv)(lbl);
      bbox = div.getBoundingClientRect();
      dv.attr("width", bbox.width);
      dv.attr("height", bbox.height);
    }
    if (bbox.width > maxWidth) {
      maxWidth = bbox.width;
    }
    maxHeight += bbox.height + rowPadding;
    classMethods.push(lbl);
  });
  maxHeight += lineHeight;
  if (hasInterface) {
    let diffX2 = (maxWidth - interfaceBBox.width) / 2;
    (0,src/* select */.Ltv)(interfaceLabel).attr(
      "transform",
      "translate( " + (-1 * maxWidth / 2 + diffX2) + ", " + -1 * maxHeight / 2 + ")"
    );
    verticalPos = interfaceBBox.height + rowPadding;
  }
  let diffX = (maxWidth - classTitleBBox.width) / 2;
  (0,src/* select */.Ltv)(classTitleLabel).attr(
    "transform",
    "translate( " + (-1 * maxWidth / 2 + diffX) + ", " + (-1 * maxHeight / 2 + verticalPos) + ")"
  );
  verticalPos += classTitleBBox.height + rowPadding;
  topLine.attr("class", "divider").attr("x1", -maxWidth / 2 - halfPadding).attr("x2", maxWidth / 2 + halfPadding).attr("y1", -maxHeight / 2 - halfPadding + lineHeight + verticalPos).attr("y2", -maxHeight / 2 - halfPadding + lineHeight + verticalPos);
  verticalPos += lineHeight;
  classAttributes.forEach((lbl) => {
    (0,src/* select */.Ltv)(lbl).attr(
      "transform",
      "translate( " + -maxWidth / 2 + ", " + (-1 * maxHeight / 2 + verticalPos + lineHeight / 2) + ")"
    );
    const memberBBox = lbl?.getBBox();
    verticalPos += (memberBBox?.height ?? 0) + rowPadding;
  });
  verticalPos += lineHeight;
  bottomLine.attr("class", "divider").attr("x1", -maxWidth / 2 - halfPadding).attr("x2", maxWidth / 2 + halfPadding).attr("y1", -maxHeight / 2 - halfPadding + lineHeight + verticalPos).attr("y2", -maxHeight / 2 - halfPadding + lineHeight + verticalPos);
  verticalPos += lineHeight;
  classMethods.forEach((lbl) => {
    (0,src/* select */.Ltv)(lbl).attr(
      "transform",
      "translate( " + -maxWidth / 2 + ", " + (-1 * maxHeight / 2 + verticalPos) + ")"
    );
    const memberBBox = lbl?.getBBox();
    verticalPos += (memberBBox?.height ?? 0) + rowPadding;
  });
  rect2.attr("style", node.style).attr("class", "outer title-state").attr("x", -maxWidth / 2 - halfPadding).attr("y", -(maxHeight / 2) - halfPadding).attr("width", maxWidth + node.padding).attr("height", maxHeight + node.padding);
  updateNodeBounds(node, rect2);
  node.intersect = function(point2) {
    return intersect_default.rect(node, point2);
  };
  return shapeSvg;
}, "class_box");
var shapes = {
  rhombus: question,
  composite,
  question,
  rect,
  labelRect,
  rectWithTitle,
  choice,
  circle: circle2,
  doublecircle,
  stadium,
  hexagon,
  block_arrow,
  rect_left_inv_arrow,
  lean_right,
  lean_left,
  trapezoid,
  inv_trapezoid,
  rect_right_inv_arrow,
  cylinder,
  start,
  end,
  note: note_default,
  subroutine,
  fork: forkJoin,
  join: forkJoin,
  class_box
};
var nodeElems = {};
var insertNode = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async (elem, node, renderOptions) => {
  let newEl;
  let el;
  if (node.link) {
    let target;
    if ((0,chunk_CSCIHK7Q/* getConfig2 */.D7)().securityLevel === "sandbox") {
      target = "_top";
    } else if (node.linkTarget) {
      target = node.linkTarget || "_blank";
    }
    newEl = elem.insert("svg:a").attr("xlink:href", node.link).attr("target", target);
    el = await shapes[node.shape](newEl, node, renderOptions);
  } else {
    el = await shapes[node.shape](elem, node, renderOptions);
    newEl = el;
  }
  if (node.tooltip) {
    el.attr("title", node.tooltip);
  }
  if (node.class) {
    el.attr("class", "node default " + node.class);
  }
  nodeElems[node.id] = newEl;
  if (node.haveCallback) {
    nodeElems[node.id].attr("class", nodeElems[node.id].attr("class") + " clickable");
  }
  return newEl;
}, "insertNode");
var positionNode = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)((node) => {
  const el = nodeElems[node.id];
  chunk_AGHRB4JF/* log */.Rm.trace(
    "Transforming node",
    node.diff,
    node,
    "translate(" + (node.x - node.width / 2 - 5) + ", " + node.width / 2 + ")"
  );
  const padding2 = 8;
  const diff = node.diff || 0;
  if (node.clusterNode) {
    el.attr(
      "transform",
      "translate(" + (node.x + diff - node.width / 2) + ", " + (node.y - node.height / 2 - padding2) + ")"
    );
  } else {
    el.attr("transform", "translate(" + node.x + ", " + node.y + ")");
  }
  return diff;
}, "positionNode");

// src/diagrams/block/renderHelpers.ts
function getNodeFromBlock(block, db2, positioned = false) {
  const vertex = block;
  let classStr = "default";
  if ((vertex?.classes?.length || 0) > 0) {
    classStr = (vertex?.classes ?? []).join(" ");
  }
  classStr = classStr + " flowchart-label";
  let radius = 0;
  let shape = "";
  let padding2;
  switch (vertex.type) {
    case "round":
      radius = 5;
      shape = "rect";
      break;
    case "composite":
      radius = 0;
      shape = "composite";
      padding2 = 0;
      break;
    case "square":
      shape = "rect";
      break;
    case "diamond":
      shape = "question";
      break;
    case "hexagon":
      shape = "hexagon";
      break;
    case "block_arrow":
      shape = "block_arrow";
      break;
    case "odd":
      shape = "rect_left_inv_arrow";
      break;
    case "lean_right":
      shape = "lean_right";
      break;
    case "lean_left":
      shape = "lean_left";
      break;
    case "trapezoid":
      shape = "trapezoid";
      break;
    case "inv_trapezoid":
      shape = "inv_trapezoid";
      break;
    case "rect_left_inv_arrow":
      shape = "rect_left_inv_arrow";
      break;
    case "circle":
      shape = "circle";
      break;
    case "ellipse":
      shape = "ellipse";
      break;
    case "stadium":
      shape = "stadium";
      break;
    case "subroutine":
      shape = "subroutine";
      break;
    case "cylinder":
      shape = "cylinder";
      break;
    case "group":
      shape = "rect";
      break;
    case "doublecircle":
      shape = "doublecircle";
      break;
    default:
      shape = "rect";
  }
  const styles = (0,chunk_5ZQYHXKU/* getStylesFromArray */.sM)(vertex?.styles ?? []);
  const vertexText = vertex.label;
  const bounds = vertex.size ?? { width: 0, height: 0, x: 0, y: 0 };
  const dbDiagramId = db2.getDiagramId();
  const node = {
    labelStyle: styles.labelStyle,
    shape,
    labelText: vertexText,
    rx: radius,
    ry: radius,
    class: classStr,
    style: styles.style,
    id: vertex.id,
    domId: dbDiagramId ? `${dbDiagramId}-${vertex.id}` : vertex.id,
    directions: vertex.directions,
    width: bounds.width,
    height: bounds.height,
    x: bounds.x,
    y: bounds.y,
    positioned,
    intersect: void 0,
    type: vertex.type,
    padding: padding2 ?? (0,chunk_CSCIHK7Q/* getConfig */.zj)()?.block?.padding ?? 0,
    widthInColumns: vertex.widthInColumns ?? 1
  };
  return node;
}
(0,chunk_AGHRB4JF/* __name */.K2)(getNodeFromBlock, "getNodeFromBlock");
async function calculateBlockSize(elem, block, db2) {
  const node = getNodeFromBlock(block, db2, false);
  if (node.type === "group") {
    return;
  }
  const config2 = (0,chunk_CSCIHK7Q/* getConfig */.zj)();
  const nodeEl = await insertNode(elem, node, { config: config2 });
  const boundingBox = nodeEl.node().getBBox();
  const obj = db2.getBlock(node.id);
  obj.size = { width: boundingBox.width, height: boundingBox.height, x: 0, y: 0, node: nodeEl };
  db2.setBlock(obj);
  nodeEl.remove();
}
(0,chunk_AGHRB4JF/* __name */.K2)(calculateBlockSize, "calculateBlockSize");
async function insertBlockPositioned(elem, block, db2) {
  const node = getNodeFromBlock(block, db2, true);
  const obj = db2.getBlock(node.id);
  if (obj.type !== "space") {
    const config2 = (0,chunk_CSCIHK7Q/* getConfig */.zj)();
    await insertNode(elem, node, { config: config2 });
    block.intersect = node?.intersect;
    positionNode(node);
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(insertBlockPositioned, "insertBlockPositioned");
async function performOperations(elem, blocks2, db2, operation) {
  for (const block of blocks2) {
    await operation(elem, block, db2);
    if (block.children) {
      await performOperations(elem, block.children, db2, operation);
    }
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(performOperations, "performOperations");
async function calculateBlockSizes(elem, blocks2, db2) {
  await performOperations(elem, blocks2, db2, calculateBlockSize);
}
(0,chunk_AGHRB4JF/* __name */.K2)(calculateBlockSizes, "calculateBlockSizes");
async function insertBlocks(elem, blocks2, db2) {
  await performOperations(elem, blocks2, db2, insertBlockPositioned);
}
(0,chunk_AGHRB4JF/* __name */.K2)(insertBlocks, "insertBlocks");
async function insertEdges(elem, edges, blocks2, db2, id) {
  const g = new graphlib/* Graph */.T({
    multigraph: true,
    compound: true
  });
  g.setGraph({
    rankdir: "TB",
    nodesep: 10,
    ranksep: 10,
    marginx: 8,
    marginy: 8
  });
  for (const block of blocks2) {
    if (block.size) {
      g.setNode(block.id, {
        width: block.size.width,
        height: block.size.height,
        intersect: block.intersect
      });
    }
  }
  for (const edge of edges) {
    if (edge.start && edge.end) {
      const startBlock = db2.getBlock(edge.start);
      const endBlock = db2.getBlock(edge.end);
      if (startBlock?.size && endBlock?.size) {
        const start2 = startBlock.size;
        const end2 = endBlock.size;
        const points = [
          { x: start2.x, y: start2.y },
          { x: start2.x + (end2.x - start2.x) / 2, y: start2.y + (end2.y - start2.y) / 2 },
          { x: end2.x, y: end2.y }
        ];
        const prefixedEdgeId = id ? `${id}-${edge.id}` : edge.id;
        const thicknessClass = edge.thickness === "thick" ? "edge-thickness-thick" : "edge-thickness-normal";
        const patternClass = edge.pattern === "dotted" ? "edge-pattern-dotted" : "edge-pattern-solid";
        const dynamicClasses = `${thicknessClass} ${patternClass} flowchart-link LS-a1 LE-b1`;
        insertEdge(
          elem,
          { v: edge.start, w: edge.end, name: prefixedEdgeId },
          {
            ...edge,
            id: prefixedEdgeId,
            arrowTypeEnd: edge.arrowTypeEnd,
            arrowTypeStart: edge.arrowTypeStart,
            points,
            classes: dynamicClasses
          },
          void 0,
          "block",
          g,
          id
        );
        if (edge.label) {
          await insertEdgeLabel(elem, {
            ...edge,
            label: edge.label,
            labelStyle: "stroke: #333; stroke-width: 1.5px;fill:none;",
            arrowTypeEnd: edge.arrowTypeEnd,
            arrowTypeStart: edge.arrowTypeStart,
            points,
            classes: dynamicClasses
          });
          positionEdgeLabel(
            { ...edge, x: points[1].x, y: points[1].y },
            {
              originalPath: points
            }
          );
        }
      }
    }
  }
}
(0,chunk_AGHRB4JF/* __name */.K2)(insertEdges, "insertEdges");

// src/diagrams/block/blockRenderer.ts
var getClasses2 = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(function(text, diagObj) {
  return diagObj.db.getClasses();
}, "getClasses");
var draw = /* @__PURE__ */ (0,chunk_AGHRB4JF/* __name */.K2)(async function(text, id, _version, diagObj) {
  const { securityLevel, block: conf } = (0,chunk_CSCIHK7Q/* getConfig */.zj)();
  const db2 = diagObj.db;
  db2.setDiagramId(id);
  let sandboxElement;
  if (securityLevel === "sandbox") {
    sandboxElement = (0,src/* select */.Ltv)("#i" + id);
  }
  const root = securityLevel === "sandbox" ? (0,src/* select */.Ltv)(sandboxElement.nodes()[0].contentDocument.body) : (0,src/* select */.Ltv)("body");
  const svg = securityLevel === "sandbox" ? root.select(`[id="${id}"]`) : (0,src/* select */.Ltv)(`[id="${id}"]`);
  const markers2 = ["point", "circle", "cross"];
  markers_default(svg, markers2, diagObj.type, id);
  const bl = db2.getBlocks();
  const blArr = db2.getBlocksFlat();
  const edges = db2.getEdges();
  const nodes = svg.insert("g").attr("class", "block");
  await calculateBlockSizes(nodes, bl, db2);
  const bounds = layout(db2);
  await insertBlocks(nodes, bl, db2);
  await insertEdges(nodes, edges, blArr, db2, id);
  if (bounds) {
    const bounds2 = bounds;
    const magicFactor = Math.max(1, Math.round(0.125 * (bounds2.width / bounds2.height)));
    const height = bounds2.height + magicFactor + 10;
    const width = bounds2.width + 10;
    const { useMaxWidth } = conf;
    (0,chunk_CSCIHK7Q/* configureSvgSize */.a$)(svg, height, width, !!useMaxWidth);
    chunk_AGHRB4JF/* log */.Rm.debug("Here Bounds", bounds, bounds2);
    svg.attr(
      "viewBox",
      `${bounds2.x - 5} ${bounds2.y - 5} ${bounds2.width + 10} ${bounds2.height + 10}`
    );
  }
}, "draw");
var blockRenderer_default = {
  draw,
  getClasses: getClasses2
};

// src/diagrams/block/blockDiagram.ts
var diagram = {
  parser: block_default,
  db: blockDB_default,
  renderer: blockRenderer_default,
  styles: styles_default
};



/***/ },

/***/ 52501
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   o: () => (/* binding */ getIconStyles)
/* harmony export */ });
/* harmony import */ var _chunk_AGHRB4JF_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(40797);


// src/diagrams/globalStyles.ts
var getIconStyles = /* @__PURE__ */ (0,_chunk_AGHRB4JF_mjs__WEBPACK_IMPORTED_MODULE_0__/* .__name */ .K2)(() => `
  /* Font Awesome icon styling - consolidated */
  .label-icon {
    display: inline-block;
    height: 1em;
    overflow: visible;
    vertical-align: -0.125em;
  }
  
  .node .label-icon path {
    fill: currentColor;
    stroke: revert;
    stroke-width: revert;
  }
`, "getIconStyles");




/***/ }

};
;