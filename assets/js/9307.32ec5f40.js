"use strict";
exports.id = 9307;
exports.ids = [9307];
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

/***/ 83765
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  M: () => (/* binding */ write)
});

// UNUSED EXPORTS: read

// EXTERNAL MODULE: ./node_modules/lodash-es/isUndefined.js
var isUndefined = __webpack_require__(69592);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseClone.js + 15 modules
var _baseClone = __webpack_require__(91641);
;// ./node_modules/lodash-es/clone.js


/** Used to compose bitmasks for cloning. */
var CLONE_SYMBOLS_FLAG = 4;

/**
 * Creates a shallow clone of `value`.
 *
 * **Note:** This method is loosely based on the
 * [structured clone algorithm](https://mdn.io/Structured_clone_algorithm)
 * and supports cloning arrays, array buffers, booleans, date objects, maps,
 * numbers, `Object` objects, regexes, sets, strings, symbols, and typed
 * arrays. The own enumerable properties of `arguments` objects are cloned
 * as plain objects. An empty object is returned for uncloneable values such
 * as error objects, functions, DOM nodes, and WeakMaps.
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Lang
 * @param {*} value The value to clone.
 * @returns {*} Returns the cloned value.
 * @see _.cloneDeep
 * @example
 *
 * var objects = [{ 'a': 1 }, { 'b': 2 }];
 *
 * var shallow = _.clone(objects);
 * console.log(shallow[0] === objects[0]);
 * // => true
 */
function clone(value) {
  return (0,_baseClone/* default */.A)(value, CLONE_SYMBOLS_FLAG);
}

/* harmony default export */ const lodash_es_clone = (clone);

// EXTERNAL MODULE: ./node_modules/lodash-es/map.js
var map = __webpack_require__(52341);
// EXTERNAL MODULE: ./node_modules/dagre-d3-es/src/graphlib/graph.js + 10 modules
var graph = __webpack_require__(23126);
;// ./node_modules/dagre-d3-es/src/graphlib/json.js
/* unused harmony import specifier */ var _;
/* unused harmony import specifier */ var Graph;



/**
 * @import { NodeID, EdgeObj, GraphOptions } from './graph.js';
 */



/**
 * @template [GraphLabel=any] - Label of the graph.
 * @template [NodeLabel=any] - Label of a node.
 * @template [EdgeLabel=any] - Label of an edge.
 *
 * @typedef {object} GraphJSON
 * @property {Required<GraphOptions>} options - The options used to create the graph.
 * @property {Array<{ v: NodeID; value?: NodeLabel; parent?: NodeID }>} nodes - The nodes in the graph.
 * @property {Array<EdgeObj & { value?: EdgeLabel }>} edges - The edges in the graph.
 * @property {GraphLabel} [value] - The graph's value, if any.
 */

/**
 * Creates a JSON representation of the graph that can be serialized to a
 * string with
 * [JSON.stringify](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify).
 * The graph can later be restored using {@link read}.
 *
 * @example
 *
 * ```js
 * var g = new graphlib.Graph();
 * g.setNode("a", { label: "node a" });
 * g.setNode("b", { label: "node b" });
 * g.setEdge("a", "b", { label: "edge a->b" });
 * graphlib.json.write(g);
 * // Returns the object:
 * //
 * // {
 * //   "options": {
 * //     "directed": true,
 * //     "multigraph": false,
 * //     "compound": false
 * //   },
 * //   "nodes": [
 * //     { "v": "a", "value": { "label": "node a" } },
 * //     { "v": "b", "value": { "label": "node b" } }
 * //   ],
 * //   "edges": [
 * //     { "v": "a", "w": "b", "value": { "label": "edge a->b" } }
 * //   ]
 * // }
 * ```
 *
 * @template [GraphLabel=any] - Label of the graph.
 * @template [NodeLabel=any] - Label of a node.
 * @template [EdgeLabel=any] - Label of an edge.
 * @param {Graph<GraphLabel, NodeLabel, EdgeLabel>} g - The graph to serialize.
 * @returns {GraphJSON<GraphLabel, NodeLabel, EdgeLabel>} The JSON representation of the graph.
 */
function write(g) {
  /** @type {GraphJSON<GraphLabel, NodeLabel, EdgeLabel>} */
  var json = {
    options: {
      directed: g.isDirected(),
      multigraph: g.isMultigraph(),
      compound: g.isCompound(),
    },
    nodes: writeNodes(g),
    edges: writeEdges(g),
  };
  if (!isUndefined/* default */.A(g.graph())) {
    json.value = lodash_es_clone(g.graph());
  }
  return json;
}

/**
 * @template NodeLabel - Label of a node.
 *
 * @param {Graph<unknown, NodeLabel, unknown>} g - The graph to serialize.
 * @returns {Array<{ v: NodeID; value?: NodeLabel; parent?: NodeID }>} The nodes in the graph.
 */
function writeNodes(g) {
  return map/* default */.A(g.nodes(), function (v) {
    var nodeValue = g.node(v);
    var parent = g.parent(v);
    /** @type {{ v: NodeID; value?: NodeLabel; parent?: NodeID }} */
    var node = { v: v };
    if (!isUndefined/* default */.A(nodeValue)) {
      node.value = nodeValue;
    }
    if (!isUndefined/* default */.A(parent)) {
      node.parent = parent;
    }
    return node;
  });
}

/**
 * @template EdgeLabel - Label of a node.
 *
 * @param {Graph<unknown, unknown, EdgeLabel>} g - The graph to serialize.
 * @returns {Array<EdgeObj & { value?: EdgeLabel }>} The edges in the graph.
 */
function writeEdges(g) {
  return map/* default */.A(g.edges(), function (e) {
    var edgeValue = g.edge(e);
    /** @type {EdgeObj & { value?: EdgeLabel }} */
    var edge = { v: e.v, w: e.w };
    if (!isUndefined/* default */.A(e.name)) {
      edge.name = e.name;
    }
    if (!isUndefined/* default */.A(edgeValue)) {
      edge.value = edgeValue;
    }
    return edge;
  });
}

/**
 * Takes JSON as input and returns the graph representation.
 *
 * @example
 *
 * For example, if we have serialized the graph in {@link write}
 * to a string named `str`, we can restore it to a graph as follows:
 *
 * ```js
 * var g2 = graphlib.json.read(JSON.parse(str));
 * // or, in order to copy the graph
 * var g3 = graphlib.json.read(graphlib.json.write(g))
 *
 * g2.nodes();
 * // ['a', 'b']
 * g2.edges()
 * // [ { v: 'a', w: 'b' } ]
 * ```
 *
 * @template [GraphLabel=any] - Label of the graph.
 * @template [NodeLabel=any] - Label of a node.
 * @template [EdgeLabel=any] - Label of an edge.
 * @param {GraphJSON<GraphLabel, NodeLabel, EdgeLabel>} json - The JSON representation of the graph.
 * @returns {Graph<GraphLabel, NodeLabel, EdgeLabel>} The restored graph.
 */
function read(json) {
  var g = new Graph(json.options).setGraph(json.value);
  _.each(json.nodes, function (entry) {
    g.setNode(entry.v, entry.value);
    if (entry.parent) {
      g.setParent(entry.v, entry.parent);
    }
  });
  _.each(json.edges, function (entry) {
    g.setEdge({ v: entry.v, w: entry.w, name: entry.name }, entry.value);
  });
  return g;
}


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

/***/ 52851
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseAssignValue_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(52528);
/* harmony import */ var _eq_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(66984);



/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Assigns `value` to `key` of `object` if the existing value is not equivalent
 * using [`SameValueZero`](http://ecma-international.org/ecma-262/7.0/#sec-samevaluezero)
 * for equality comparisons.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {string} key The key of the property to assign.
 * @param {*} value The value to assign.
 */
function assignValue(object, key, value) {
  var objValue = object[key];
  if (!(hasOwnProperty.call(object, key) && (0,_eq_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(objValue, value)) ||
      (value === undefined && !(key in object))) {
    (0,_baseAssignValue_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object, key, value);
  }
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (assignValue);


/***/ },

/***/ 52528
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _defineProperty_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(84171);


/**
 * The base implementation of `assignValue` and `assignMergeValue` without
 * value checks.
 *
 * @private
 * @param {Object} object The object to modify.
 * @param {string} key The key of the property to assign.
 * @param {*} value The value to assign.
 */
function baseAssignValue(object, key, value) {
  if (key == '__proto__' && _defineProperty_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A) {
    (0,_defineProperty_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object, key, {
      'configurable': true,
      'enumerable': true,
      'value': value,
      'writable': true
    });
  } else {
    object[key] = value;
  }
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseAssignValue);


/***/ },

/***/ 91641
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _baseClone)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_Stack.js + 5 modules
var _Stack = __webpack_require__(11754);
// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayEach.js
var _arrayEach = __webpack_require__(72641);
// EXTERNAL MODULE: ./node_modules/lodash-es/_assignValue.js
var _assignValue = __webpack_require__(52851);
// EXTERNAL MODULE: ./node_modules/lodash-es/_copyObject.js
var _copyObject = __webpack_require__(22031);
// EXTERNAL MODULE: ./node_modules/lodash-es/keys.js
var keys = __webpack_require__(27422);
;// ./node_modules/lodash-es/_baseAssign.js



/**
 * The base implementation of `_.assign` without support for multiple sources
 * or `customizer` functions.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @returns {Object} Returns `object`.
 */
function baseAssign(object, source) {
  return object && (0,_copyObject/* default */.A)(source, (0,keys/* default */.A)(source), object);
}

/* harmony default export */ const _baseAssign = (baseAssign);

// EXTERNAL MODULE: ./node_modules/lodash-es/keysIn.js + 2 modules
var keysIn = __webpack_require__(55615);
;// ./node_modules/lodash-es/_baseAssignIn.js



/**
 * The base implementation of `_.assignIn` without support for multiple sources
 * or `customizer` functions.
 *
 * @private
 * @param {Object} object The destination object.
 * @param {Object} source The source object.
 * @returns {Object} Returns `object`.
 */
function baseAssignIn(object, source) {
  return object && (0,_copyObject/* default */.A)(source, (0,keysIn/* default */.A)(source), object);
}

/* harmony default export */ const _baseAssignIn = (baseAssignIn);

// EXTERNAL MODULE: ./node_modules/lodash-es/_cloneBuffer.js
var _cloneBuffer = __webpack_require__(80154);
// EXTERNAL MODULE: ./node_modules/lodash-es/_copyArray.js
var _copyArray = __webpack_require__(39759);
// EXTERNAL MODULE: ./node_modules/lodash-es/_getSymbols.js
var _getSymbols = __webpack_require__(14792);
;// ./node_modules/lodash-es/_copySymbols.js



/**
 * Copies own symbols of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy symbols from.
 * @param {Object} [object={}] The object to copy symbols to.
 * @returns {Object} Returns `object`.
 */
function copySymbols(source, object) {
  return (0,_copyObject/* default */.A)(source, (0,_getSymbols/* default */.A)(source), object);
}

/* harmony default export */ const _copySymbols = (copySymbols);

// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayPush.js
var _arrayPush = __webpack_require__(76912);
// EXTERNAL MODULE: ./node_modules/lodash-es/_getPrototype.js
var _getPrototype = __webpack_require__(15647);
// EXTERNAL MODULE: ./node_modules/lodash-es/stubArray.js
var stubArray = __webpack_require__(13153);
;// ./node_modules/lodash-es/_getSymbolsIn.js





/* Built-in method references for those with the same name as other `lodash` methods. */
var nativeGetSymbols = Object.getOwnPropertySymbols;

/**
 * Creates an array of the own and inherited enumerable symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of symbols.
 */
var getSymbolsIn = !nativeGetSymbols ? stubArray/* default */.A : function(object) {
  var result = [];
  while (object) {
    (0,_arrayPush/* default */.A)(result, (0,_getSymbols/* default */.A)(object));
    object = (0,_getPrototype/* default */.A)(object);
  }
  return result;
};

/* harmony default export */ const _getSymbolsIn = (getSymbolsIn);

;// ./node_modules/lodash-es/_copySymbolsIn.js



/**
 * Copies own and inherited symbols of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy symbols from.
 * @param {Object} [object={}] The object to copy symbols to.
 * @returns {Object} Returns `object`.
 */
function copySymbolsIn(source, object) {
  return (0,_copyObject/* default */.A)(source, _getSymbolsIn(source), object);
}

/* harmony default export */ const _copySymbolsIn = (copySymbolsIn);

// EXTERNAL MODULE: ./node_modules/lodash-es/_getAllKeys.js
var _getAllKeys = __webpack_require__(19042);
// EXTERNAL MODULE: ./node_modules/lodash-es/_baseGetAllKeys.js
var _baseGetAllKeys = __webpack_require__(33831);
;// ./node_modules/lodash-es/_getAllKeysIn.js




/**
 * Creates an array of own and inherited enumerable property names and
 * symbols of `object`.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names and symbols.
 */
function getAllKeysIn(object) {
  return (0,_baseGetAllKeys/* default */.A)(object, keysIn/* default */.A, _getSymbolsIn);
}

/* harmony default export */ const _getAllKeysIn = (getAllKeysIn);

// EXTERNAL MODULE: ./node_modules/lodash-es/_getTag.js + 3 modules
var _getTag = __webpack_require__(9779);
;// ./node_modules/lodash-es/_initCloneArray.js
/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _initCloneArray_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * Initializes an array clone.
 *
 * @private
 * @param {Array} array The array to clone.
 * @returns {Array} Returns the initialized clone.
 */
function initCloneArray(array) {
  var length = array.length,
      result = new array.constructor(length);

  // Add properties assigned by `RegExp#exec`.
  if (length && typeof array[0] == 'string' && _initCloneArray_hasOwnProperty.call(array, 'index')) {
    result.index = array.index;
    result.input = array.input;
  }
  return result;
}

/* harmony default export */ const _initCloneArray = (initCloneArray);

// EXTERNAL MODULE: ./node_modules/lodash-es/_cloneArrayBuffer.js
var _cloneArrayBuffer = __webpack_require__(90565);
;// ./node_modules/lodash-es/_cloneDataView.js


/**
 * Creates a clone of `dataView`.
 *
 * @private
 * @param {Object} dataView The data view to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the cloned data view.
 */
function cloneDataView(dataView, isDeep) {
  var buffer = isDeep ? (0,_cloneArrayBuffer/* default */.A)(dataView.buffer) : dataView.buffer;
  return new dataView.constructor(buffer, dataView.byteOffset, dataView.byteLength);
}

/* harmony default export */ const _cloneDataView = (cloneDataView);

;// ./node_modules/lodash-es/_cloneRegExp.js
/** Used to match `RegExp` flags from their coerced string values. */
var reFlags = /\w*$/;

/**
 * Creates a clone of `regexp`.
 *
 * @private
 * @param {Object} regexp The regexp to clone.
 * @returns {Object} Returns the cloned regexp.
 */
function cloneRegExp(regexp) {
  var result = new regexp.constructor(regexp.source, reFlags.exec(regexp));
  result.lastIndex = regexp.lastIndex;
  return result;
}

/* harmony default export */ const _cloneRegExp = (cloneRegExp);

// EXTERNAL MODULE: ./node_modules/lodash-es/_Symbol.js
var _Symbol = __webpack_require__(241);
;// ./node_modules/lodash-es/_cloneSymbol.js


/** Used to convert symbols to primitives and strings. */
var symbolProto = _Symbol/* default */.A ? _Symbol/* default */.A.prototype : undefined,
    symbolValueOf = symbolProto ? symbolProto.valueOf : undefined;

/**
 * Creates a clone of the `symbol` object.
 *
 * @private
 * @param {Object} symbol The symbol object to clone.
 * @returns {Object} Returns the cloned symbol object.
 */
function cloneSymbol(symbol) {
  return symbolValueOf ? Object(symbolValueOf.call(symbol)) : {};
}

/* harmony default export */ const _cloneSymbol = (cloneSymbol);

// EXTERNAL MODULE: ./node_modules/lodash-es/_cloneTypedArray.js
var _cloneTypedArray = __webpack_require__(1801);
;// ./node_modules/lodash-es/_initCloneByTag.js






/** `Object#toString` result references. */
var boolTag = '[object Boolean]',
    dateTag = '[object Date]',
    mapTag = '[object Map]',
    numberTag = '[object Number]',
    regexpTag = '[object RegExp]',
    setTag = '[object Set]',
    stringTag = '[object String]',
    symbolTag = '[object Symbol]';

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

/**
 * Initializes an object clone based on its `toStringTag`.
 *
 * **Note:** This function only supports cloning values with tags of
 * `Boolean`, `Date`, `Error`, `Map`, `Number`, `RegExp`, `Set`, or `String`.
 *
 * @private
 * @param {Object} object The object to clone.
 * @param {string} tag The `toStringTag` of the object to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the initialized clone.
 */
function initCloneByTag(object, tag, isDeep) {
  var Ctor = object.constructor;
  switch (tag) {
    case arrayBufferTag:
      return (0,_cloneArrayBuffer/* default */.A)(object);

    case boolTag:
    case dateTag:
      return new Ctor(+object);

    case dataViewTag:
      return _cloneDataView(object, isDeep);

    case float32Tag: case float64Tag:
    case int8Tag: case int16Tag: case int32Tag:
    case uint8Tag: case uint8ClampedTag: case uint16Tag: case uint32Tag:
      return (0,_cloneTypedArray/* default */.A)(object, isDeep);

    case mapTag:
      return new Ctor;

    case numberTag:
    case stringTag:
      return new Ctor(object);

    case regexpTag:
      return _cloneRegExp(object);

    case setTag:
      return new Ctor;

    case symbolTag:
      return _cloneSymbol(object);
  }
}

/* harmony default export */ const _initCloneByTag = (initCloneByTag);

// EXTERNAL MODULE: ./node_modules/lodash-es/_initCloneObject.js + 1 modules
var _initCloneObject = __webpack_require__(18598);
// EXTERNAL MODULE: ./node_modules/lodash-es/isArray.js
var isArray = __webpack_require__(92049);
// EXTERNAL MODULE: ./node_modules/lodash-es/isBuffer.js + 1 modules
var isBuffer = __webpack_require__(99912);
// EXTERNAL MODULE: ./node_modules/lodash-es/isObjectLike.js
var isObjectLike = __webpack_require__(53098);
;// ./node_modules/lodash-es/_baseIsMap.js



/** `Object#toString` result references. */
var _baseIsMap_mapTag = '[object Map]';

/**
 * The base implementation of `_.isMap` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a map, else `false`.
 */
function baseIsMap(value) {
  return (0,isObjectLike/* default */.A)(value) && (0,_getTag/* default */.A)(value) == _baseIsMap_mapTag;
}

/* harmony default export */ const _baseIsMap = (baseIsMap);

// EXTERNAL MODULE: ./node_modules/lodash-es/_baseUnary.js
var _baseUnary = __webpack_require__(52789);
// EXTERNAL MODULE: ./node_modules/lodash-es/_nodeUtil.js
var _nodeUtil = __webpack_require__(64841);
;// ./node_modules/lodash-es/isMap.js




/* Node.js helper references. */
var nodeIsMap = _nodeUtil/* default */.A && _nodeUtil/* default */.A.isMap;

/**
 * Checks if `value` is classified as a `Map` object.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a map, else `false`.
 * @example
 *
 * _.isMap(new Map);
 * // => true
 *
 * _.isMap(new WeakMap);
 * // => false
 */
var isMap = nodeIsMap ? (0,_baseUnary/* default */.A)(nodeIsMap) : _baseIsMap;

/* harmony default export */ const lodash_es_isMap = (isMap);

// EXTERNAL MODULE: ./node_modules/lodash-es/isObject.js
var isObject = __webpack_require__(23149);
;// ./node_modules/lodash-es/_baseIsSet.js



/** `Object#toString` result references. */
var _baseIsSet_setTag = '[object Set]';

/**
 * The base implementation of `_.isSet` without Node.js optimizations.
 *
 * @private
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a set, else `false`.
 */
function baseIsSet(value) {
  return (0,isObjectLike/* default */.A)(value) && (0,_getTag/* default */.A)(value) == _baseIsSet_setTag;
}

/* harmony default export */ const _baseIsSet = (baseIsSet);

;// ./node_modules/lodash-es/isSet.js




/* Node.js helper references. */
var nodeIsSet = _nodeUtil/* default */.A && _nodeUtil/* default */.A.isSet;

/**
 * Checks if `value` is classified as a `Set` object.
 *
 * @static
 * @memberOf _
 * @since 4.3.0
 * @category Lang
 * @param {*} value The value to check.
 * @returns {boolean} Returns `true` if `value` is a set, else `false`.
 * @example
 *
 * _.isSet(new Set);
 * // => true
 *
 * _.isSet(new WeakSet);
 * // => false
 */
var isSet = nodeIsSet ? (0,_baseUnary/* default */.A)(nodeIsSet) : _baseIsSet;

/* harmony default export */ const lodash_es_isSet = (isSet);

;// ./node_modules/lodash-es/_baseClone.js























/** Used to compose bitmasks for cloning. */
var CLONE_DEEP_FLAG = 1,
    CLONE_FLAT_FLAG = 2,
    CLONE_SYMBOLS_FLAG = 4;

/** `Object#toString` result references. */
var argsTag = '[object Arguments]',
    arrayTag = '[object Array]',
    _baseClone_boolTag = '[object Boolean]',
    _baseClone_dateTag = '[object Date]',
    errorTag = '[object Error]',
    funcTag = '[object Function]',
    genTag = '[object GeneratorFunction]',
    _baseClone_mapTag = '[object Map]',
    _baseClone_numberTag = '[object Number]',
    objectTag = '[object Object]',
    _baseClone_regexpTag = '[object RegExp]',
    _baseClone_setTag = '[object Set]',
    _baseClone_stringTag = '[object String]',
    _baseClone_symbolTag = '[object Symbol]',
    weakMapTag = '[object WeakMap]';

var _baseClone_arrayBufferTag = '[object ArrayBuffer]',
    _baseClone_dataViewTag = '[object DataView]',
    _baseClone_float32Tag = '[object Float32Array]',
    _baseClone_float64Tag = '[object Float64Array]',
    _baseClone_int8Tag = '[object Int8Array]',
    _baseClone_int16Tag = '[object Int16Array]',
    _baseClone_int32Tag = '[object Int32Array]',
    _baseClone_uint8Tag = '[object Uint8Array]',
    _baseClone_uint8ClampedTag = '[object Uint8ClampedArray]',
    _baseClone_uint16Tag = '[object Uint16Array]',
    _baseClone_uint32Tag = '[object Uint32Array]';

/** Used to identify `toStringTag` values supported by `_.clone`. */
var cloneableTags = {};
cloneableTags[argsTag] = cloneableTags[arrayTag] =
cloneableTags[_baseClone_arrayBufferTag] = cloneableTags[_baseClone_dataViewTag] =
cloneableTags[_baseClone_boolTag] = cloneableTags[_baseClone_dateTag] =
cloneableTags[_baseClone_float32Tag] = cloneableTags[_baseClone_float64Tag] =
cloneableTags[_baseClone_int8Tag] = cloneableTags[_baseClone_int16Tag] =
cloneableTags[_baseClone_int32Tag] = cloneableTags[_baseClone_mapTag] =
cloneableTags[_baseClone_numberTag] = cloneableTags[objectTag] =
cloneableTags[_baseClone_regexpTag] = cloneableTags[_baseClone_setTag] =
cloneableTags[_baseClone_stringTag] = cloneableTags[_baseClone_symbolTag] =
cloneableTags[_baseClone_uint8Tag] = cloneableTags[_baseClone_uint8ClampedTag] =
cloneableTags[_baseClone_uint16Tag] = cloneableTags[_baseClone_uint32Tag] = true;
cloneableTags[errorTag] = cloneableTags[funcTag] =
cloneableTags[weakMapTag] = false;

/**
 * The base implementation of `_.clone` and `_.cloneDeep` which tracks
 * traversed objects.
 *
 * @private
 * @param {*} value The value to clone.
 * @param {boolean} bitmask The bitmask flags.
 *  1 - Deep clone
 *  2 - Flatten inherited properties
 *  4 - Clone symbols
 * @param {Function} [customizer] The function to customize cloning.
 * @param {string} [key] The key of `value`.
 * @param {Object} [object] The parent object of `value`.
 * @param {Object} [stack] Tracks traversed objects and their clone counterparts.
 * @returns {*} Returns the cloned value.
 */
function baseClone(value, bitmask, customizer, key, object, stack) {
  var result,
      isDeep = bitmask & CLONE_DEEP_FLAG,
      isFlat = bitmask & CLONE_FLAT_FLAG,
      isFull = bitmask & CLONE_SYMBOLS_FLAG;

  if (customizer) {
    result = object ? customizer(value, key, object, stack) : customizer(value);
  }
  if (result !== undefined) {
    return result;
  }
  if (!(0,isObject/* default */.A)(value)) {
    return value;
  }
  var isArr = (0,isArray/* default */.A)(value);
  if (isArr) {
    result = _initCloneArray(value);
    if (!isDeep) {
      return (0,_copyArray/* default */.A)(value, result);
    }
  } else {
    var tag = (0,_getTag/* default */.A)(value),
        isFunc = tag == funcTag || tag == genTag;

    if ((0,isBuffer/* default */.A)(value)) {
      return (0,_cloneBuffer/* default */.A)(value, isDeep);
    }
    if (tag == objectTag || tag == argsTag || (isFunc && !object)) {
      result = (isFlat || isFunc) ? {} : (0,_initCloneObject/* default */.A)(value);
      if (!isDeep) {
        return isFlat
          ? _copySymbolsIn(value, _baseAssignIn(result, value))
          : _copySymbols(value, _baseAssign(result, value));
      }
    } else {
      if (!cloneableTags[tag]) {
        return object ? value : {};
      }
      result = _initCloneByTag(value, tag, isDeep);
    }
  }
  // Check for circular references and return its corresponding clone.
  stack || (stack = new _Stack/* default */.A);
  var stacked = stack.get(value);
  if (stacked) {
    return stacked;
  }
  stack.set(value, result);

  if (lodash_es_isSet(value)) {
    value.forEach(function(subValue) {
      result.add(baseClone(subValue, bitmask, customizer, subValue, value, stack));
    });
  } else if (lodash_es_isMap(value)) {
    value.forEach(function(subValue, key) {
      result.set(key, baseClone(subValue, bitmask, customizer, key, value, stack));
    });
  }

  var keysFunc = isFull
    ? (isFlat ? _getAllKeysIn : _getAllKeys/* default */.A)
    : (isFlat ? keysIn/* default */.A : keys/* default */.A);

  var props = isArr ? undefined : keysFunc(value);
  (0,_arrayEach/* default */.A)(props || value, function(subValue, key) {
    if (props) {
      key = subValue;
      subValue = value[key];
    }
    // Recursively populate clone (susceptible to call stack limits).
    (0,_assignValue/* default */.A)(result, key, baseClone(subValue, bitmask, customizer, key, value, stack));
  });
  return result;
}

/* harmony default export */ const _baseClone = (baseClone);


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

/***/ 52568
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _baseEach_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(6240);
/* harmony import */ var _isArrayLike_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(38446);



/**
 * The base implementation of `_.map` without support for iteratee shorthands.
 *
 * @private
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} iteratee The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 */
function baseMap(collection, iteratee) {
  var index = -1,
      result = (0,_isArrayLike_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(collection) ? Array(collection.length) : [];

  (0,_baseEach_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(collection, function(value, key, collection) {
    result[++index] = iteratee(value, key, collection);
  });
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (baseMap);


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

/***/ 90565
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _Uint8Array_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(43988);


/**
 * Creates a clone of `arrayBuffer`.
 *
 * @private
 * @param {ArrayBuffer} arrayBuffer The array buffer to clone.
 * @returns {ArrayBuffer} Returns the cloned array buffer.
 */
function cloneArrayBuffer(arrayBuffer) {
  var result = new arrayBuffer.constructor(arrayBuffer.byteLength);
  new _Uint8Array_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A(result).set(new _Uint8Array_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A(arrayBuffer));
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (cloneArrayBuffer);


/***/ },

/***/ 80154
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _root_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(41917);


/** Detect free variable `exports`. */
var freeExports = typeof exports == 'object' && exports && !exports.nodeType && exports;

/** Detect free variable `module`. */
var freeModule = freeExports && typeof module == 'object' && module && !module.nodeType && module;

/** Detect the popular CommonJS extension `module.exports`. */
var moduleExports = freeModule && freeModule.exports === freeExports;

/** Built-in value references. */
var Buffer = moduleExports ? _root_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A.Buffer : undefined,
    allocUnsafe = Buffer ? Buffer.allocUnsafe : undefined;

/**
 * Creates a clone of  `buffer`.
 *
 * @private
 * @param {Buffer} buffer The buffer to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Buffer} Returns the cloned buffer.
 */
function cloneBuffer(buffer, isDeep) {
  if (isDeep) {
    return buffer.slice();
  }
  var length = buffer.length,
      result = allocUnsafe ? allocUnsafe(length) : new buffer.constructor(length);

  buffer.copy(result);
  return result;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (cloneBuffer);


/***/ },

/***/ 1801
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _cloneArrayBuffer_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(90565);


/**
 * Creates a clone of `typedArray`.
 *
 * @private
 * @param {Object} typedArray The typed array to clone.
 * @param {boolean} [isDeep] Specify a deep clone.
 * @returns {Object} Returns the cloned typed array.
 */
function cloneTypedArray(typedArray, isDeep) {
  var buffer = isDeep ? (0,_cloneArrayBuffer_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(typedArray.buffer) : typedArray.buffer;
  return new typedArray.constructor(buffer, typedArray.byteOffset, typedArray.length);
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (cloneTypedArray);


/***/ },

/***/ 39759
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/**
 * Copies the values of `source` to `array`.
 *
 * @private
 * @param {Array} source The array to copy values from.
 * @param {Array} [array=[]] The array to copy values to.
 * @returns {Array} Returns `array`.
 */
function copyArray(source, array) {
  var index = -1,
      length = source.length;

  array || (array = Array(length));
  while (++index < length) {
    array[index] = source[index];
  }
  return array;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (copyArray);


/***/ },

/***/ 22031
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _assignValue_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(52851);
/* harmony import */ var _baseAssignValue_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(52528);



/**
 * Copies properties of `source` to `object`.
 *
 * @private
 * @param {Object} source The object to copy properties from.
 * @param {Array} props The property identifiers to copy.
 * @param {Object} [object={}] The object to copy properties to.
 * @param {Function} [customizer] The function to customize copied values.
 * @returns {Object} Returns `object`.
 */
function copyObject(source, props, object, customizer) {
  var isNew = !object;
  object || (object = {});

  var index = -1,
      length = props.length;

  while (++index < length) {
    var key = props[index];

    var newValue = customizer
      ? customizer(object[key], source[key], key, object, source)
      : undefined;

    if (newValue === undefined) {
      newValue = source[key];
    }
    if (isNew) {
      (0,_baseAssignValue_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(object, key, newValue);
    } else {
      (0,_assignValue_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(object, key, newValue);
    }
  }
  return object;
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (copyObject);


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

/***/ 15647
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _overArg_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(40367);


/** Built-in value references. */
var getPrototype = (0,_overArg_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A)(Object.getPrototypeOf, Object);

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (getPrototype);


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

/***/ 18598
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ _initCloneObject)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/isObject.js
var isObject = __webpack_require__(23149);
;// ./node_modules/lodash-es/_baseCreate.js


/** Built-in value references. */
var objectCreate = Object.create;

/**
 * The base implementation of `_.create` without support for assigning
 * properties to the created object.
 *
 * @private
 * @param {Object} proto The object to inherit from.
 * @returns {Object} Returns the new object.
 */
var baseCreate = (function() {
  function object() {}
  return function(proto) {
    if (!(0,isObject/* default */.A)(proto)) {
      return {};
    }
    if (objectCreate) {
      return objectCreate(proto);
    }
    object.prototype = proto;
    var result = new object;
    object.prototype = undefined;
    return result;
  };
}());

/* harmony default export */ const _baseCreate = (baseCreate);

// EXTERNAL MODULE: ./node_modules/lodash-es/_getPrototype.js
var _getPrototype = __webpack_require__(15647);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isPrototype.js
var _isPrototype = __webpack_require__(97271);
;// ./node_modules/lodash-es/_initCloneObject.js




/**
 * Initializes an object clone.
 *
 * @private
 * @param {Object} object The object to clone.
 * @returns {Object} Returns the initialized clone.
 */
function initCloneObject(object) {
  return (typeof object.constructor == 'function' && !(0,_isPrototype/* default */.A)(object))
    ? _baseCreate((0,_getPrototype/* default */.A)(object))
    : {};
}

/* harmony default export */ const _initCloneObject = (initCloneObject);


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

/***/ 55615
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  A: () => (/* binding */ lodash_es_keysIn)
});

// EXTERNAL MODULE: ./node_modules/lodash-es/_arrayLikeKeys.js + 1 modules
var _arrayLikeKeys = __webpack_require__(83607);
// EXTERNAL MODULE: ./node_modules/lodash-es/isObject.js
var isObject = __webpack_require__(23149);
// EXTERNAL MODULE: ./node_modules/lodash-es/_isPrototype.js
var _isPrototype = __webpack_require__(97271);
;// ./node_modules/lodash-es/_nativeKeysIn.js
/**
 * This function is like
 * [`Object.keys`](http://ecma-international.org/ecma-262/7.0/#sec-object.keys)
 * except that it includes inherited enumerable properties.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */
function nativeKeysIn(object) {
  var result = [];
  if (object != null) {
    for (var key in Object(object)) {
      result.push(key);
    }
  }
  return result;
}

/* harmony default export */ const _nativeKeysIn = (nativeKeysIn);

;// ./node_modules/lodash-es/_baseKeysIn.js




/** Used for built-in method references. */
var objectProto = Object.prototype;

/** Used to check objects for own properties. */
var _baseKeysIn_hasOwnProperty = objectProto.hasOwnProperty;

/**
 * The base implementation of `_.keysIn` which doesn't treat sparse arrays as dense.
 *
 * @private
 * @param {Object} object The object to query.
 * @returns {Array} Returns the array of property names.
 */
function baseKeysIn(object) {
  if (!(0,isObject/* default */.A)(object)) {
    return _nativeKeysIn(object);
  }
  var isProto = (0,_isPrototype/* default */.A)(object),
      result = [];

  for (var key in object) {
    if (!(key == 'constructor' && (isProto || !_baseKeysIn_hasOwnProperty.call(object, key)))) {
      result.push(key);
    }
  }
  return result;
}

/* harmony default export */ const _baseKeysIn = (baseKeysIn);

// EXTERNAL MODULE: ./node_modules/lodash-es/isArrayLike.js
var isArrayLike = __webpack_require__(38446);
;// ./node_modules/lodash-es/keysIn.js




/**
 * Creates an array of the own and inherited enumerable property names of `object`.
 *
 * **Note:** Non-object values are coerced to objects.
 *
 * @static
 * @memberOf _
 * @since 3.0.0
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
 * _.keysIn(new Foo);
 * // => ['a', 'b', 'c'] (iteration order is not guaranteed)
 */
function keysIn(object) {
  return (0,isArrayLike/* default */.A)(object) ? (0,_arrayLikeKeys/* default */.A)(object, true) : _baseKeysIn(object);
}

/* harmony default export */ const lodash_es_keysIn = (keysIn);


/***/ },

/***/ 52341
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   A: () => (__WEBPACK_DEFAULT_EXPORT__)
/* harmony export */ });
/* harmony import */ var _arrayMap_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(45572);
/* harmony import */ var _baseIteratee_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(49574);
/* harmony import */ var _baseMap_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(52568);
/* harmony import */ var _isArray_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(92049);





/**
 * Creates an array of values by running each element in `collection` thru
 * `iteratee`. The iteratee is invoked with three arguments:
 * (value, index|key, collection).
 *
 * Many lodash methods are guarded to work as iteratees for methods like
 * `_.every`, `_.filter`, `_.map`, `_.mapValues`, `_.reject`, and `_.some`.
 *
 * The guarded methods are:
 * `ary`, `chunk`, `curry`, `curryRight`, `drop`, `dropRight`, `every`,
 * `fill`, `invert`, `parseInt`, `random`, `range`, `rangeRight`, `repeat`,
 * `sampleSize`, `slice`, `some`, `sortBy`, `split`, `take`, `takeRight`,
 * `template`, `trim`, `trimEnd`, `trimStart`, and `words`
 *
 * @static
 * @memberOf _
 * @since 0.1.0
 * @category Collection
 * @param {Array|Object} collection The collection to iterate over.
 * @param {Function} [iteratee=_.identity] The function invoked per iteration.
 * @returns {Array} Returns the new mapped array.
 * @example
 *
 * function square(n) {
 *   return n * n;
 * }
 *
 * _.map([4, 8], square);
 * // => [16, 64]
 *
 * _.map({ 'a': 4, 'b': 8 }, square);
 * // => [16, 64] (iteration order is not guaranteed)
 *
 * var users = [
 *   { 'user': 'barney' },
 *   { 'user': 'fred' }
 * ];
 *
 * // The `_.property` iteratee shorthand.
 * _.map(users, 'user');
 * // => ['barney', 'fred']
 */
function map(collection, iteratee) {
  var func = (0,_isArray_js__WEBPACK_IMPORTED_MODULE_3__/* ["default"] */ .A)(collection) ? _arrayMap_js__WEBPACK_IMPORTED_MODULE_0__/* ["default"] */ .A : _baseMap_js__WEBPACK_IMPORTED_MODULE_2__/* ["default"] */ .A;
  return func(collection, (0,_baseIteratee_js__WEBPACK_IMPORTED_MODULE_1__/* ["default"] */ .A)(iteratee, 3));
}

/* harmony default export */ const __WEBPACK_DEFAULT_EXPORT__ = (map);


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

/***/ 62204
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   IU: () => (/* binding */ clear),
/* harmony export */   OS: () => (/* binding */ adjustClustersAndEdges),
/* harmony export */   dc: () => (/* binding */ findNonClusterChild),
/* harmony export */   ju: () => (/* binding */ clusterDb),
/* harmony export */   sc: () => (/* binding */ sortNodesByHierarchy)
/* harmony export */ });
/* harmony import */ var _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(31293);
/* harmony import */ var _chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(86827);
/* harmony import */ var dagre_d3_es_src_graphlib_index_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(697);
/* harmony import */ var dagre_d3_es_src_graphlib_json_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(83765);



// src/rendering-util/layout-algorithms/dagre/mermaid-graphlib.js


var clusterDb = /* @__PURE__ */ new Map();
var descendants = /* @__PURE__ */ new Map();
var parents = /* @__PURE__ */ new Map();
var clear = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)(() => {
  descendants.clear();
  parents.clear();
  clusterDb.clear();
}, "clear");
var isDescendant = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((id, ancestorId) => {
  const ancestorDescendants = descendants.get(ancestorId) || [];
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.trace("In isDescendant", ancestorId, " ", id, " = ", ancestorDescendants.includes(id));
  return ancestorDescendants.includes(id);
}, "isDescendant");
var edgeInCluster = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((edge, clusterId) => {
  const clusterDescendants = descendants.get(clusterId) || [];
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Descendants of ", clusterId, " is ", clusterDescendants);
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Edge is ", edge);
  if (edge.v === clusterId || edge.w === clusterId) {
    return false;
  }
  if (!clusterDescendants) {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Tilt, ", clusterId, ",not in descendants");
    return false;
  }
  return clusterDescendants.includes(edge.v) || isDescendant(edge.v, clusterId) || isDescendant(edge.w, clusterId) || clusterDescendants.includes(edge.w);
}, "edgeInCluster");
var copy = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((clusterId, graph, newGraph, rootId) => {
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
    "Copying children of ",
    clusterId,
    "root",
    rootId,
    "data",
    graph.node(clusterId),
    rootId
  );
  const nodes = graph.children(clusterId) || [];
  if (clusterId !== rootId) {
    nodes.push(clusterId);
  }
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Copying (nodes) clusterId", clusterId, "nodes", nodes);
  nodes.forEach((node) => {
    if (graph.children(node).length > 0) {
      copy(node, graph, newGraph, rootId);
    } else {
      const data = graph.node(node);
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("cp ", node, " to ", rootId, " with parent ", clusterId);
      newGraph.setNode(node, data);
      if (rootId !== graph.parent(node)) {
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Setting parent", node, graph.parent(node));
        newGraph.setParent(node, graph.parent(node));
      }
      if (clusterId !== rootId && node !== clusterId) {
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Setting parent", node, clusterId);
        newGraph.setParent(node, clusterId);
      } else {
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("In copy ", clusterId, "root", rootId, "data", graph.node(clusterId), rootId);
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug(
          "Not Setting parent for node=",
          node,
          "cluster!==rootId",
          clusterId !== rootId,
          "node!==clusterId",
          node !== clusterId
        );
      }
      const edges = graph.edges(node);
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Copying Edges", edges);
      edges.forEach((edge) => {
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Edge", edge);
        const data2 = graph.edge(edge.v, edge.w, edge.name);
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Edge data", data2, rootId);
        try {
          if (edgeInCluster(edge, rootId)) {
            const rootDescendants = descendants.get(rootId) || [];
            const vIn = rootDescendants.includes(edge.v) || isDescendant(edge.v, rootId) || edge.v === rootId;
            const wIn = rootDescendants.includes(edge.w) || isDescendant(edge.w, rootId) || edge.w === rootId;
            if (vIn && wIn) {
              _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Copying as ", edge.v, edge.w, data2, edge.name);
              newGraph.setEdge(edge.v, edge.w, data2, edge.name);
              _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("newGraph edges ", newGraph.edges(), newGraph.edge(newGraph.edges()[0]));
            } else {
              const newV = vIn ? rootId : edge.v;
              const newW = wIn ? rootId : edge.w;
              _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info("Rebinding cross-boundary edge as ", newV, newW, data2, edge.name);
              graph.setEdge(newV, newW, data2, edge.name);
            }
          } else {
            _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.info(
              "Skipping copy of edge ",
              edge.v,
              "-->",
              edge.w,
              " rootId: ",
              rootId,
              " clusterId:",
              clusterId
            );
          }
        } catch (e) {
          _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.error(e);
        }
      });
    }
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Removing node", node);
    graph.removeNode(node);
  });
}, "copy");
var extractDescendants = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((id, graph) => {
  const children = graph.children(id);
  let res = [...children];
  for (const child of children) {
    parents.set(child, id);
    res = [...res, ...extractDescendants(child, graph)];
  }
  return res;
}, "extractDescendants");
var findCommonEdges = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, id1, id2) => {
  const edges1 = graph.edges().filter((edge) => edge.v === id1 || edge.w === id1);
  const edges2 = graph.edges().filter((edge) => edge.v === id2 || edge.w === id2);
  const edges1Prim = edges1.map((edge) => {
    return { v: edge.v === id1 ? id2 : edge.v, w: edge.w === id1 ? id1 : edge.w };
  });
  const edges2Prim = edges2.map((edge) => {
    return { v: edge.v, w: edge.w };
  });
  const result = edges1Prim.filter((edgeIn1) => {
    return edges2Prim.some((edge) => edgeIn1.v === edge.v && edgeIn1.w === edge.w);
  });
  return result;
}, "findCommonEdges");
var findNonClusterChild = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((id, graph, clusterId) => {
  const children = graph.children(id);
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.trace("Searching children of id ", id, children);
  if (children.length < 1) {
    return id;
  }
  let reserve;
  for (const child of children) {
    const _id = findNonClusterChild(child, graph, clusterId);
    const commonEdges = findCommonEdges(graph, clusterId, _id);
    if (_id) {
      if (commonEdges.length > 0) {
        reserve = _id;
      } else {
        return _id;
      }
    }
  }
  return reserve;
}, "findNonClusterChild");
var getAnchorId = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((id) => {
  if (!clusterDb.has(id)) {
    return id;
  }
  if (!clusterDb.get(id).externalConnections) {
    return id;
  }
  if (clusterDb.has(id)) {
    return clusterDb.get(id).id;
  }
  return id;
}, "getAnchorId");
var adjustClustersAndEdges = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, depth) => {
  if (!graph || depth > 10) {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Opting out, no graph ");
    return;
  } else {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Opting in, graph ");
  }
  graph.nodes().forEach(function(id) {
    const children = graph.children(id);
    if (children.length > 0) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
        "Cluster identified",
        id,
        " Replacement id in edges: ",
        findNonClusterChild(id, graph, id)
      );
      descendants.set(id, extractDescendants(id, graph));
      clusterDb.set(id, { id: findNonClusterChild(id, graph, id), clusterData: graph.node(id) });
    }
  });
  graph.nodes().forEach(function(id) {
    const children = graph.children(id);
    const edges = graph.edges();
    if (children.length > 0) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Cluster identified", id, descendants);
      edges.forEach((edge) => {
        const d1 = isDescendant(edge.v, id);
        const d2 = isDescendant(edge.w, id);
        if (d1 ^ d2) {
          _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Edge: ", edge, " leaves cluster ", id);
          _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Descendants of XXX ", id, ": ", descendants.get(id));
          clusterDb.get(id).externalConnections = true;
        }
      });
    } else {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Not a cluster ", id, descendants);
    }
  });
  for (let id of clusterDb.keys()) {
    const nonClusterChild = clusterDb.get(id).id;
    const parent = graph.parent(nonClusterChild);
    if (parent !== id && clusterDb.has(parent) && !clusterDb.get(parent).externalConnections) {
      clusterDb.get(id).id = parent;
    }
    const hasDirectOutgoingEdge = graph.edges().some((edge) => edge.v === id);
    if (nonClusterChild && clusterDb.get(id)?.externalConnections && hasDirectOutgoingEdge && isNodeInExtractableCluster(graph, nonClusterChild, id)) {
      const safeAnchor = findSafeAnchorNode(graph, id, graph.parent(nonClusterChild));
      if (safeAnchor) {
        clusterDb.get(id).id = safeAnchor;
      }
    }
  }
  graph.edges().forEach(function(e) {
    const edge = graph.edge(e);
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Edge " + e.v + " -> " + e.w + ": " + JSON.stringify(e));
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Edge " + e.v + " -> " + e.w + ": " + JSON.stringify(graph.edge(e)));
    let v = e.v;
    let w = e.w;
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
      "Fix XXX",
      clusterDb,
      "ids:",
      e.v,
      e.w,
      "Translating: ",
      clusterDb.get(e.v),
      " --- ",
      clusterDb.get(e.w)
    );
    if (clusterDb.get(e.v) || clusterDb.get(e.w)) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Fixing and trying - removing XXX", e.v, e.w, e.name);
      v = getAnchorId(e.v);
      w = getAnchorId(e.w);
      graph.removeEdge(e.v, e.w, e.name);
      if (v !== e.v) {
        const parent = graph.parent(v);
        clusterDb.get(parent).externalConnections = true;
        edge.fromCluster = e.v;
      }
      if (w !== e.w) {
        const parent = graph.parent(w);
        clusterDb.get(parent).externalConnections = true;
        edge.toCluster = e.w;
      }
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Fix Replacing with XXX", v, w, e.name);
      graph.setEdge(v, w, edge, e.name);
    }
  });
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Adjusted Graph", dagre_d3_es_src_graphlib_json_js__WEBPACK_IMPORTED_MODULE_3__/* .write */ .M(graph));
  extractor(graph, 0);
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.trace(clusterDb);
}, "adjustClustersAndEdges");
var extractor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, depth) => {
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("extractor - ", depth, dagre_d3_es_src_graphlib_json_js__WEBPACK_IMPORTED_MODULE_3__/* .write */ .M(graph), graph.children("D"));
  if (depth > 10) {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.error("Bailing out");
    return;
  }
  let nodes = graph.nodes();
  let hasChildren = false;
  for (const node of nodes) {
    const children = graph.children(node);
    hasChildren = hasChildren || children.length > 0;
  }
  if (!hasChildren) {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Done, no node has children", graph.nodes());
    return;
  }
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Nodes = ", nodes, depth);
  for (const node of nodes) {
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug(
      "Extracting node",
      node,
      clusterDb,
      clusterDb.has(node) && !clusterDb.get(node).externalConnections,
      !graph.parent(node),
      graph.node(node),
      graph.children("D"),
      " Depth ",
      depth
    );
    if (!clusterDb.has(node)) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Not a cluster", node, depth);
    } else if (clusterDb.get(node)?.clusterData?.explicitDir && graph.children(node) && graph.children(node).length > 0) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Cluster with explicit dir, creating subgraph for children", node, depth);
      const dir = clusterDb.get(node).clusterData.dir;
      const clusterGraph = new dagre_d3_es_src_graphlib_index_js__WEBPACK_IMPORTED_MODULE_2__/* .Graph */ .T({
        multigraph: true,
        compound: true
      }).setGraph({
        rankdir: dir,
        nodesep: 50,
        ranksep: 50,
        marginx: 8,
        marginy: 8
      }).setDefaultEdgeLabel(function() {
        return {};
      });
      copy(node, graph, clusterGraph, node);
      const clusterNodeData = graph.node(node) || {};
      graph.setNode(node, {
        ...clusterNodeData,
        clusterNode: true,
        id: node,
        clusterData: clusterDb.get(node).clusterData,
        label: clusterDb.get(node).label,
        graph: clusterGraph
      });
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
        "Subgraph for cluster with explicit dir created:",
        node,
        dagre_d3_es_src_graphlib_json_js__WEBPACK_IMPORTED_MODULE_3__/* .write */ .M(clusterGraph)
      );
    } else if (!clusterDb.get(node).externalConnections && graph.children(node) && graph.children(node).length > 0) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
        "Cluster without external connections, without a parent and with children",
        node,
        depth
      );
      const graphSettings = graph.graph();
      let dir = graphSettings.rankdir === "TB" ? "LR" : "TB";
      if (clusterDb.get(node)?.clusterData?.dir) {
        dir = clusterDb.get(node).clusterData.dir;
        _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("Fixing dir", clusterDb.get(node).clusterData.dir, dir);
      }
      const clusterGraph = new dagre_d3_es_src_graphlib_index_js__WEBPACK_IMPORTED_MODULE_2__/* .Graph */ .T({
        multigraph: true,
        compound: true
      }).setGraph({
        rankdir: dir,
        nodesep: 50,
        ranksep: 50,
        marginx: 8,
        marginy: 8
      }).setDefaultEdgeLabel(function() {
        return {};
      });
      copy(node, graph, clusterGraph, node);
      const clusterNodeData = graph.node(node) || {};
      graph.setNode(node, {
        ...clusterNodeData,
        clusterNode: true,
        id: node,
        clusterData: clusterDb.get(node).clusterData,
        label: clusterDb.get(node).label,
        graph: clusterGraph
      });
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug("Old graph after copy", dagre_d3_es_src_graphlib_json_js__WEBPACK_IMPORTED_MODULE_3__/* .write */ .M(graph));
    } else {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(
        "Cluster ** ",
        node,
        " **not meeting the criteria !externalConnections:",
        !clusterDb.get(node).externalConnections,
        " no parent: ",
        !graph.parent(node),
        " children ",
        graph.children(node) && graph.children(node).length > 0,
        graph.children("D"),
        depth
      );
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.debug(clusterDb);
    }
  }
  nodes = graph.nodes();
  _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn("New list of nodes", nodes);
  for (const node of nodes) {
    const data = graph.node(node);
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_0__/* .log */ .R.warn(" Now next level", node, data);
    if (data?.clusterNode) {
      extractor(data.graph, depth + 1);
    }
  }
}, "extractor");
var sorter = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, nodes) => {
  if (nodes.length === 0) {
    return [];
  }
  let result = Object.assign([], nodes);
  nodes.forEach((node) => {
    const children = graph.children(node);
    const sorted = sorter(graph, children);
    result = [...result, ...sorted];
  });
  return result;
}, "sorter");
var sortNodesByHierarchy = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph) => sorter(graph, graph.children()), "sortNodesByHierarchy");
var isNodeInExtractableCluster = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, node, rootId) => {
  let parent = graph.parent(node);
  while (parent && parent !== rootId) {
    const cluster = clusterDb.get(parent);
    if (cluster && !cluster.externalConnections) {
      return true;
    }
    parent = graph.parent(parent);
  }
  return false;
}, "isNodeInExtractableCluster");
var findSafeAnchorNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_1__/* .__name */ .K)((graph, clusterId, excludedCluster) => {
  const children = graph.children(clusterId) ?? [];
  for (const child of children) {
    if (child === excludedCluster || isDescendant(child, excludedCluster)) {
      continue;
    }
    const candidate = findNonClusterChild(child, graph, clusterId);
    if (!candidate) {
      continue;
    }
    if (!isNodeInExtractableCluster(graph, candidate, clusterId)) {
      return candidate;
    }
  }
  return null;
}, "findSafeAnchorNode");




/***/ },

/***/ 99307
(__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   render: () => (/* binding */ render)
/* harmony export */ });
/* harmony import */ var _chunk_RYQCIY6F_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(62204);
/* harmony import */ var _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(9290);
/* harmony import */ var _chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(9636);
/* harmony import */ var _chunk_W5SLKNZC_mjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(14782);
/* harmony import */ var _chunk_7BUUIJ7U_mjs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(91865);
/* harmony import */ var _chunk_UBXNYLIW_mjs__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(6129);
/* harmony import */ var _chunk_WRU74C26_mjs__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(47244);
/* harmony import */ var _chunk_4I5QYGJK_mjs__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(14512);
/* harmony import */ var _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(34502);
/* harmony import */ var _chunk_I66GZJ75_mjs__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(59069);
/* harmony import */ var _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(31293);
/* harmony import */ var _chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(86827);
/* harmony import */ var dagre_d3_es_src_graphlib_index_js__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(697);













// src/rendering-util/createGraph.ts

async function createGraphWithElements(element, data4Layout) {
  const graph = new dagre_d3_es_src_graphlib_index_js__WEBPACK_IMPORTED_MODULE_12__/* .Graph */ .T({
    multigraph: true,
    compound: true
  });
  const edgesToProcess = [...data4Layout.edges];
  const config = (0,_chunk_I66GZJ75_mjs__WEBPACK_IMPORTED_MODULE_9__/* .getConfig2 */ .D7)();
  const rootGroups = element.insert("g").attr("class", "root");
  const clusters = rootGroups.insert("g").attr("class", "clusters");
  const edgePaths = rootGroups.insert("g").attr("class", "edges edgePath");
  const edgeLabels2 = rootGroups.insert("g").attr("class", "edgeLabels");
  const nodesGroup = rootGroups.insert("g").attr("class", "nodes");
  const nodeElements = /* @__PURE__ */ new Map();
  const hasDom = element.node() != null;
  await Promise.all(
    data4Layout.nodes.map(async (node) => {
      if (node.isGroup) {
        graph.setNode(node.id, { ...node });
      } else {
        if (hasDom) {
          const childNodeEl = await (0,_chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__/* .insertNode */ .on)(nodesGroup, node, { config, dir: node.dir });
          const boundingBox = childNodeEl.node()?.getBBox() ?? { width: 0, height: 0 };
          nodeElements.set(node.id, childNodeEl);
          node.width = boundingBox.width;
          node.height = boundingBox.height;
        }
        graph.setNode(node.id, { ...node });
      }
    })
  );
  for (const edge of edgesToProcess) {
    graph.setEdge(edge.start, edge.end, { ...edge }, edge.id);
    const edgeExists = data4Layout.edges.some((existingEdge) => existingEdge.id === edge.id);
    if (!edgeExists) {
      data4Layout.edges.push(edge);
    }
  }
  if (globalThis.mermaidCaptureSizes) {
    const { captureNodeSizes } = await __webpack_require__.e(/* import() */ 2857).then(__webpack_require__.bind(__webpack_require__, 62857));
    captureNodeSizes(element, data4Layout);
  }
  return {
    graph,
    groups: { clusters, edgePaths, edgeLabels: edgeLabels2, nodes: nodesGroup, rootGroups },
    nodeElements
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(createGraphWithElements, "createGraphWithElements");

// src/rendering-util/rendering-elements/lineJump.ts
var ROUNDED_CORNER_RADIUS = 5;
var CORNER_EPSILON = 1e-5;
var ENDPOINT_EPSILON = 1e-6;
function buildSegmentList(points) {
  const segments = [];
  for (let i = 0; i < points.length - 1; i++) {
    segments.push({ a: points[i], b: points[i + 1] });
  }
  return segments;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildSegmentList, "buildSegmentList");
function segmentIntersection(a1, a2, b1, b2) {
  const dxA = a2.x - a1.x;
  const dyA = a2.y - a1.y;
  const dxB = b2.x - b1.x;
  const dyB = b2.y - b1.y;
  const denom = dxA * dyB - dyA * dxB;
  if (denom === 0) {
    return null;
  }
  const dx = b1.x - a1.x;
  const dy = b1.y - a1.y;
  const tA = (dx * dyB - dy * dxB) / denom;
  const tB = (dx * dyA - dy * dxA) / denom;
  if (tA <= ENDPOINT_EPSILON || tA >= 1 - ENDPOINT_EPSILON || tB <= ENDPOINT_EPSILON || tB >= 1 - ENDPOINT_EPSILON) {
    return null;
  }
  return {
    point: { x: a1.x + tA * dxA, y: a1.y + tA * dyA },
    tA,
    tB
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentIntersection, "segmentIntersection");
function isHorizontalSeg(seg) {
  return Math.abs(seg.b.x - seg.a.x) >= Math.abs(seg.b.y - seg.a.y);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isHorizontalSeg, "isHorizontalSeg");
function findEdgeIntersections(edges) {
  const crossings = [];
  for (let i = 0; i < edges.length; i++) {
    const edgeA = edges[i];
    const segmentsA = buildSegmentList(edgeA.points);
    for (let j = i + 1; j < edges.length; j++) {
      const edgeB = edges[j];
      const segmentsB = buildSegmentList(edgeB.points);
      for (const [si, segA] of segmentsA.entries()) {
        for (const [sj, segB] of segmentsB.entries()) {
          const hit = segmentIntersection(segA.a, segA.b, segB.a, segB.b);
          if (!hit) {
            continue;
          }
          const aHoriz = isHorizontalSeg(segA);
          const bHoriz = isHorizontalSeg(segB);
          const orthogonalPair = aHoriz !== bHoriz;
          const jumpOnA = orthogonalPair ? aHoriz : false;
          if (jumpOnA) {
            crossings.push({
              jumpEdgeId: edgeA.id,
              otherEdgeId: edgeB.id,
              segIndex: si,
              t: hit.tA,
              point: hit.point
            });
          } else {
            crossings.push({
              jumpEdgeId: edgeB.id,
              otherEdgeId: edgeA.id,
              segIndex: sj,
              t: hit.tB,
              point: hit.point
            });
          }
        }
      }
    }
  }
  return crossings;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(findEdgeIntersections, "findEdgeIntersections");
function fmt(n) {
  const rounded = Math.round(n * 1e3) / 1e3;
  return Number.isInteger(rounded) ? `${rounded}` : `${rounded}`;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(fmt, "fmt");
function pointToString(p) {
  return `${fmt(p.x)},${fmt(p.y)}`;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(pointToString, "pointToString");
function getArcSweepFlag(seg) {
  const dx = seg.b.x - seg.a.x;
  const dy = seg.b.y - seg.a.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? 1 : 0;
  }
  return dy >= 0 ? 1 : 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(getArcSweepFlag, "getArcSweepFlag");
var MIN_JUMP_RADIUS = 1e-3;
function applyMarkerOffsets(points, edge) {
  if (points.length < 2) {
    return points.map((p) => ({ ...p }));
  }
  const out = points.map((p) => ({ ...p }));
  const startOff = edge.arrowTypeStart && _chunk_7BUUIJ7U_mjs__WEBPACK_IMPORTED_MODULE_4__/* .markerOffsets */ .hq[edge.arrowTypeStart];
  if (startOff) {
    const a = points[0];
    const b = points[1];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    out[0].x = a.x + startOff * Math.cos(ang);
    out[0].y = a.y + startOff * Math.sin(ang);
  }
  const endOff = edge.arrowTypeEnd && _chunk_7BUUIJ7U_mjs__WEBPACK_IMPORTED_MODULE_4__/* .markerOffsets */ .hq[edge.arrowTypeEnd];
  if (endOff) {
    const n = points.length;
    const a = points[n - 2];
    const b = points[n - 1];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    out[n - 1].x = b.x - endOff * Math.cos(ang);
    out[n - 1].y = b.y - endOff * Math.sin(ang);
  }
  return out;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(applyMarkerOffsets, "applyMarkerOffsets");
function emitJump(jump, ux, uy, sweep, style) {
  const cx = jump.point.x;
  const cy = jump.point.y;
  const pre = { x: cx - ux * jump.r, y: cy - uy * jump.r };
  const post = { x: cx + ux * jump.r, y: cy + uy * jump.r };
  const out = [`L${pointToString(pre)}`];
  if (style === "arc") {
    out.push(`A${fmt(jump.r)},${fmt(jump.r)} 0 0 ${sweep} ${pointToString(post)}`);
  } else {
    out.push(`M${pointToString(post)}`);
  }
  return out;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(emitJump, "emitJump");
function computeRoundedCorner(prev, curr, next, radius) {
  const dx1 = curr.x - prev.x;
  const dy1 = curr.y - prev.y;
  const dx2 = next.x - curr.x;
  const dy2 = next.y - curr.y;
  const len1 = Math.hypot(dx1, dy1);
  const len2 = Math.hypot(dx2, dy2);
  if (len1 < CORNER_EPSILON || len2 < CORNER_EPSILON) {
    return null;
  }
  const nx1 = dx1 / len1;
  const ny1 = dy1 / len1;
  const nx2 = dx2 / len2;
  const ny2 = dy2 / len2;
  const dot = nx1 * nx2 + ny1 * ny2;
  const clamped = Math.max(-1, Math.min(1, dot));
  const angle = Math.acos(clamped);
  if (angle < CORNER_EPSILON || Math.abs(Math.PI - angle) < CORNER_EPSILON) {
    return null;
  }
  const cutLen = Math.min(radius / Math.sin(angle / 2), len1 / 2, len2 / 2);
  return {
    startX: curr.x - nx1 * cutLen,
    startY: curr.y - ny1 * cutLen,
    endX: curr.x + nx2 * cutLen,
    endY: curr.y + ny2 * cutLen,
    ctrlX: curr.x,
    ctrlY: curr.y,
    cutLen
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(computeRoundedCorner, "computeRoundedCorner");
function rewriteEdgePath(edge, jumps, config) {
  const rawPoints = edge.points;
  if (rawPoints.length < 2) {
    return "";
  }
  const points = applyMarkerOffsets(rawPoints, edge);
  const rounded = edge.curve === "rounded";
  const segments = buildSegmentList(points);
  const bySeg = /* @__PURE__ */ new Map();
  for (const j of jumps) {
    const seg = segments[j.segIndex];
    if (!seg) {
      continue;
    }
    const segLen = Math.hypot(seg.b.x - seg.a.x, seg.b.y - seg.a.y);
    const list = bySeg.get(j.segIndex) ?? [];
    list.push({
      t: j.t,
      point: j.point,
      d: j.t * segLen,
      r: config.jumpRadius
    });
    bySeg.set(j.segIndex, list);
  }
  const parts = [`M${pointToString(points[0])}`];
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const segLen = Math.hypot(seg.b.x - seg.a.x, seg.b.y - seg.a.y);
    const ux = segLen === 0 ? 0 : (seg.b.x - seg.a.x) / segLen;
    const uy = segLen === 0 ? 0 : (seg.b.y - seg.a.y) / segLen;
    const sweep = getArcSweepFlag(seg);
    let segStartConsumed = 0;
    if (rounded && i > 0) {
      const corner = computeRoundedCorner(
        points[i - 1],
        points[i],
        points[i + 1] ?? points[i],
        ROUNDED_CORNER_RADIUS
      );
      if (corner) {
        segStartConsumed = corner.cutLen;
      }
    }
    let segEndStop = segLen;
    let upcomingCorner = null;
    if (rounded && i < segments.length - 1) {
      upcomingCorner = computeRoundedCorner(
        points[i],
        points[i + 1],
        points[i + 2] ?? points[i + 1],
        ROUNDED_CORNER_RADIUS
      );
      if (upcomingCorner) {
        segEndStop = segLen - upcomingCorner.cutLen;
      }
    }
    const segJumps = [...bySeg.get(i) ?? []].sort((a, b) => a.t - b.t);
    for (const j of segJumps) {
      j.r = Math.min(j.r, j.d - segStartConsumed, segEndStop - j.d);
    }
    for (let k = 0; k < segJumps.length - 1; k++) {
      const gap = segJumps[k + 1].d - segJumps[k].d;
      if (segJumps[k].r + segJumps[k + 1].r > gap) {
        const half = gap / 2;
        segJumps[k].r = Math.min(segJumps[k].r, half);
        segJumps[k + 1].r = Math.min(segJumps[k + 1].r, half);
      }
    }
    for (const j of segJumps) {
      if (j.r < MIN_JUMP_RADIUS) {
        continue;
      }
      parts.push(...emitJump(j, ux, uy, sweep, config.jumpStyle));
    }
    if (rounded && upcomingCorner) {
      parts.push(`L${fmt(upcomingCorner.startX)},${fmt(upcomingCorner.startY)}`);
      parts.push(
        `Q${fmt(upcomingCorner.ctrlX)},${fmt(upcomingCorner.ctrlY)} ${fmt(upcomingCorner.endX)},${fmt(upcomingCorner.endY)}`
      );
    } else {
      parts.push(`L${pointToString(seg.b)}`);
    }
  }
  return parts.join(" ");
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(rewriteEdgePath, "rewriteEdgePath");
function isStraightPath(d) {
  return /^[\d\s+,.LMelm-]*$/.test(d);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isStraightPath, "isStraightPath");
function curveSupportsLineHops(curve) {
  if (!curve) {
    return true;
  }
  return curve === "linear" || curve === "rounded" || curve === "step" || curve === "stepBefore" || curve === "stepAfter";
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(curveSupportsLineHops, "curveSupportsLineHops");
function decodeDataPoints(raw) {
  if (!raw) {
    return null;
  }
  try {
    const json = typeof atob === "function" ? atob(raw) : Buffer.from(raw, "base64").toString();
    const parsed = JSON.parse(json);
    if (!Array.isArray(parsed)) {
      return null;
    }
    const pts = [];
    for (const p of parsed) {
      if (p && typeof p.x === "number" && typeof p.y === "number") {
        pts.push({ x: p.x, y: p.y });
      }
    }
    return pts.length >= 2 ? pts : null;
  } catch {
    return null;
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(decodeDataPoints, "decodeDataPoints");
function applyLineJumpsToSvg(edgePathsGroup, edges, config) {
  if (!config.enabled) {
    return;
  }
  const groupNode = edgePathsGroup.node();
  if (!groupNode) {
    return;
  }
  const edgeMeta = /* @__PURE__ */ new Map();
  for (const e of edges) {
    edgeMeta.set(e.id, e);
  }
  const renderedEdges = [];
  const pathById = /* @__PURE__ */ new Map();
  for (const e of edges) {
    const escapedId = typeof CSS !== "undefined" && CSS.escape ? CSS.escape(e.id) : e.id;
    const pathEl = groupNode.querySelector(`path[data-id="${escapedId}"]`);
    if (!pathEl) {
      continue;
    }
    pathById.set(e.id, pathEl);
    const decoded = decodeDataPoints(pathEl.getAttribute("data-points"));
    const points = decoded ?? e.points;
    renderedEdges.push({ ...e, points });
  }
  const crossings = findEdgeIntersections(renderedEdges);
  if (crossings.length === 0) {
    return;
  }
  const jumpsByEdge = /* @__PURE__ */ new Map();
  for (const c of crossings) {
    const list = jumpsByEdge.get(c.jumpEdgeId) ?? [];
    list.push(c);
    jumpsByEdge.set(c.jumpEdgeId, list);
  }
  for (const renderedEdge of renderedEdges) {
    const jumps = jumpsByEdge.get(renderedEdge.id);
    if (!jumps || jumps.length === 0) {
      continue;
    }
    const meta = edgeMeta.get(renderedEdge.id);
    const curveHint = meta?.curve;
    if (curveHint !== void 0 && !curveSupportsLineHops(curveHint)) {
      continue;
    }
    const pathEl = pathById.get(renderedEdge.id);
    if (!pathEl) {
      continue;
    }
    if (curveHint === void 0) {
      const currentD = pathEl.getAttribute("d") ?? "";
      if (!isStraightPath(currentD)) {
        continue;
      }
    }
    const originalStyle = pathEl.getAttribute("style") ?? "";
    const dasharrayMatch = /stroke-dasharray\s*:\s*0\s+([\d.]+)\s+[\d.]+\s+([\d.]+)/.exec(
      originalStyle
    );
    const preservedOValueS = dasharrayMatch ? Number.parseFloat(dasharrayMatch[1]) : null;
    const preservedOValueE = dasharrayMatch ? Number.parseFloat(dasharrayMatch[2]) : null;
    const newD = rewriteEdgePath(renderedEdge, jumps, config);
    pathEl.setAttribute("d", newD);
    if (preservedOValueS !== null && preservedOValueE !== null && typeof pathEl.getTotalLength === "function") {
      const newLen = pathEl.getTotalLength();
      const onLen = Math.max(0, newLen - preservedOValueS - preservedOValueE);
      const newDasharray = `0 ${preservedOValueS} ${onLen} ${preservedOValueE}`;
      const cleaned = originalStyle.replace(/stroke-dasharray\s*:[^;]*;?/g, `stroke-dasharray: ${newDasharray};`).replace(/;\s*;+/g, ";");
      pathEl.setAttribute("style", cleaned);
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(applyLineJumpsToSvg, "applyLineJumpsToSvg");

// src/rendering-util/layout-algorithms/swimlanes/adjustLayout.ts
async function adjustLayout(data4Layout, groups) {
  for (const node of data4Layout.nodes) {
    if (node.isGroup) {
      await (0,_chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__/* .insertCluster */ .U)(groups.clusters, node);
    } else {
      (0,_chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__/* .positionNode */ .U_)(node);
    }
  }
  const nodeById = /* @__PURE__ */ new Map();
  for (const node of data4Layout.nodes) {
    if (node?.id) {
      nodeById.set(node.id, node);
    }
  }
  for (const edge of data4Layout.edges) {
    const startNode = edge.start ? nodeById.get(edge.start) ?? {} : {};
    const endNode = edge.end ? nodeById.get(edge.end) ?? {} : {};
    const paths = (0,_chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .insertEdge */ .Jo)(
      groups.edgePaths,
      { ...edge },
      {},
      data4Layout.type,
      startNode,
      endNode,
      data4Layout.diagramId
    );
    if (edge.label) {
      await (0,_chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .insertEdgeLabel */ .jP)(groups.rootGroups, edge);
    }
    if (edge.label) {
      positionEdgeLabel(edge, paths);
    }
  }
  const lineHopsConfig = data4Layout.config?.swimlane?.lineHops;
  if (lineHopsConfig !== false) {
    const jumpStyle = lineHopsConfig === "gap" ? "gap" : "arc";
    const edgeGeometries = data4Layout.edges.filter((e) => Array.isArray(e.points) && e.points.length >= 2).map((e) => ({
      id: e.id,
      points: e.points,
      curve: e.curve,
      arrowTypeStart: e.arrowTypeStart,
      arrowTypeEnd: e.arrowTypeEnd
    }));
    applyLineJumpsToSvg(groups.edgePaths, edgeGeometries, {
      enabled: true,
      jumpRadius: 6,
      jumpStyle
    });
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(adjustLayout, "adjustLayout");
function positionEdgeLabel(edge, paths) {
  const path = paths?.updatedPath ?? paths?.originalPath;
  const siteConfig = (0,_chunk_I66GZJ75_mjs__WEBPACK_IMPORTED_MODULE_9__/* .getConfig */ .zj)();
  const { subGraphTitleTotalMargin } = (0,_chunk_UBXNYLIW_mjs__WEBPACK_IMPORTED_MODULE_5__/* .getSubGraphTitleMargins */ .O)({
    flowchart: siteConfig.flowchart ?? {}
  });
  if (edge.label) {
    const el = _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .edgeLabels */ .lP.get(edge.id);
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__/* .utils_default */ ._K.calcLabelPosition(path);
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_10__/* .log */ .R.debug(
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
      if (paths) {
        x = pos.x;
        y = pos.y;
      }
    }
    el.attr("transform", `translate(${x}, ${y + subGraphTitleTotalMargin / 2})`);
  }
  if (edge?.startLabelLeft) {
    const el = _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .terminalLabels */ .UQ.get(edge.id).startLeft;
    let x = edge?.x;
    let y = edge?.y;
    if (path) {
      const pos = _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__/* .utils_default */ ._K.calcTerminalLabelPosition(edge.arrowTypeStart ? 10 : 0, "start_left", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
  if (edge.startLabelRight) {
    const el = _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .terminalLabels */ .UQ.get(edge.id).startRight;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__/* .utils_default */ ._K.calcTerminalLabelPosition(
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
    const el = _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .terminalLabels */ .UQ.get(edge.id).endLeft;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__/* .utils_default */ ._K.calcTerminalLabelPosition(edge.arrowTypeEnd ? 10 : 0, "end_left", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
  if (edge.endLabelRight) {
    const el = _chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .terminalLabels */ .UQ.get(edge.id).endRight;
    let x = edge.x;
    let y = edge.y;
    if (path) {
      const pos = _chunk_NSK5VX7P_mjs__WEBPACK_IMPORTED_MODULE_8__/* .utils_default */ ._K.calcTerminalLabelPosition(edge.arrowTypeEnd ? 10 : 0, "end_right", path);
      x = pos.x;
      y = pos.y;
    }
    el.attr("transform", `translate(${x}, ${y})`);
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(positionEdgeLabel, "positionEdgeLabel");

// src/rendering-util/layout-algorithms/swimlanes/helpers.ts
var DEFAULT_SWIMLANE_ID = "__swimlane_default__";
var TOP_LANE_TITLE_BAND_HEIGHT = 21;
var MIN_TOP_LANE_HORIZONTAL_PADDING = 20;
function topLaneHorizontalPadding(lane) {
  return Math.max(lane.padding ?? MIN_TOP_LANE_HORIZONTAL_PADDING, MIN_TOP_LANE_HORIZONTAL_PADDING);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(topLaneHorizontalPadding, "topLaneHorizontalPadding");
function assignTopLaneTitleRect(lane) {
  const { x, y, width, height } = lane;
  const contentTop = lane.swimlaneContentTop;
  if (typeof x !== "number" || typeof y !== "number" || typeof width !== "number" || typeof height !== "number" || typeof contentTop !== "number" || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || !Number.isFinite(height) || !Number.isFinite(contentTop) || width <= 0 || height <= 0) {
    delete lane.groupTitleRect;
    return;
  }
  const top = y - height / 2;
  const headerBottom = Math.min(contentTop, y + height / 2);
  const titleHeight = Math.min(TOP_LANE_TITLE_BAND_HEIGHT, Math.max(0, headerBottom - top));
  const bottom = top + titleHeight;
  if (bottom <= top) {
    delete lane.groupTitleRect;
    return;
  }
  lane.groupTitleRect = {
    left: x - width / 2,
    right: x + width / 2,
    top,
    bottom
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignTopLaneTitleRect, "assignTopLaneTitleRect");
function prepareLayoutForSwimlanes(layout) {
  const direction = layout.direction;
  const nodes = layout.nodes ??= [];
  for (const node of layout.nodes ?? []) {
    if (node.isGroup && !node.parentId) {
      node.shape = "swimlane";
      if (direction) {
        node.direction = direction;
      }
    }
  }
  const looseNodes = nodes.filter((node) => !node.isGroup && !node.parentId);
  if (looseNodes.length === 0) {
    return;
  }
  let defaultLane = nodes.find((node) => node.id === DEFAULT_SWIMLANE_ID);
  if (!defaultLane) {
    defaultLane = {
      id: DEFAULT_SWIMLANE_ID,
      label: "",
      isGroup: true,
      shape: "swimlane",
      padding: 20,
      ...direction ? { direction } : {}
    };
    nodes.push(defaultLane);
  } else if (defaultLane.isGroup) {
    defaultLane.shape = "swimlane";
    if (direction) {
      defaultLane.direction = direction;
    }
  }
  for (const node of looseNodes) {
    node.parentId = DEFAULT_SWIMLANE_ID;
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(prepareLayoutForSwimlanes, "prepareLayoutForSwimlanes");
function toGraphView(layout) {
  const nodeById = /* @__PURE__ */ new Map();
  for (const n of layout.nodes ?? []) {
    nodeById.set(n.id, n);
  }
  const edges = [];
  for (const e of layout.edges ?? []) {
    const src = typeof e.start === "string" ? e.start : void 0;
    const dst = typeof e.end === "string" ? e.end : void 0;
    if (!src || !dst) {
      continue;
    }
    if (e.labelNodeId) {
      continue;
    }
    edges.push({ id: e.id, src, dst, ref: e });
  }
  const allNodes = layout.nodes ?? [];
  const groupNodes = allNodes.filter((n) => n.isGroup);
  const nonGroupNodes = allNodes.filter((n) => !n.isGroup);
  const nodesInGroupOrder = [...groupNodes].reverse();
  const nodes = [...nodesInGroupOrder, ...nonGroupNodes].map((n) => n.id);
  return { nodes, edges, layout, nodeById };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(toGraphView, "toGraphView");
function writeBackToLayoutData(g, ordered, coords, opts) {
  const { layout } = g;
  const nodeMap = g.nodeById;
  const layerGap = opts?.layerGap ?? 100;
  const nodeGap = opts?.nodeGap ?? 40;
  let layerIndex = 0;
  for (const layer of ordered.layers) {
    let orderIndex = 0;
    for (const id of layer) {
      const node = nodeMap.get(id);
      if (!node) {
        orderIndex++;
        continue;
      }
      node.layer = layerIndex;
      node.order = orderIndex;
      const x = coords.x[id] ?? orderIndex * nodeGap;
      const y = coords.y[id] ?? layerIndex * layerGap;
      node.x = x;
      node.y = y;
      orderIndex++;
    }
    layerIndex++;
  }
  const allNodes = layout.nodes ?? [];
  const groupBounds = /* @__PURE__ */ new Map();
  const topLevelGroups = [];
  for (const group of allNodes) {
    if (!group?.isGroup) {
      continue;
    }
    if (!group.parentId) {
      topLevelGroups.push(group);
    }
    const children = allNodes.filter((n) => n.parentId === group.id);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const child of children) {
      const cx = child.x ?? coords.x[child.id];
      const cy = child.y ?? coords.y[child.id];
      const cw = child.width ?? 0;
      const ch = child.height ?? 0;
      if (cx != null && cy != null) {
        minX = Math.min(minX, cx - cw / 2);
        maxX = Math.max(maxX, cx + cw / 2);
        minY = Math.min(minY, cy - ch / 2);
        maxY = Math.max(maxY, cy + ch / 2);
      }
    }
    if (minX === Infinity || minY === Infinity) {
      group.x = group.x ?? 0;
      group.y = group.y ?? 0;
      group.width = group.width ?? 0;
      group.height = group.height ?? 0;
    } else {
      const pad = group.padding ?? 20;
      const horizontalPad = group.parentId ? pad : 2 * topLaneHorizontalPadding(group);
      const verticalPad = pad;
      const w = Math.max(0, maxX - minX) + horizontalPad;
      const h = Math.max(0, maxY - minY) + verticalPad;
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      group.x = cx;
      group.y = cy;
      group.width = w;
      group.height = h;
      groupBounds.set(group.id, { minX, maxX, minY, maxY });
    }
  }
  if (topLevelGroups.length > 0 && groupBounds.size > 0) {
    let globalMinY = Infinity;
    let globalMaxY = -Infinity;
    let maxPad = 0;
    for (const lane of topLevelGroups) {
      const pad = lane.padding ?? 20;
      if (pad > maxPad) {
        maxPad = pad;
      }
      const b = groupBounds.get(lane.id);
      if (!b) {
        continue;
      }
      globalMinY = Math.min(globalMinY, b.minY);
      globalMaxY = Math.max(globalMaxY, b.maxY);
    }
    if (globalMinY !== Infinity && globalMaxY !== -Infinity) {
      const contentHeight = Math.max(0, globalMaxY - globalMinY);
      const minHeaderMargin = 36;
      const verticalMargin = Math.max(maxPad, minHeaderMargin);
      const laneHeight = contentHeight + 2 * verticalMargin;
      const centerY = (globalMinY + globalMaxY) / 2;
      for (const lane of topLevelGroups) {
        lane.y = centerY;
        lane.height = laneHeight;
        lane.swimlaneContentTop = globalMinY;
      }
      const sortedLanes = [...topLevelGroups].sort((a, b) => {
        const ax = a.x ?? 0;
        const bx = b.x ?? 0;
        return ax - bx;
      });
      const laneIds = [];
      const centers = [];
      const baseWidths = [];
      for (const lane of sortedLanes) {
        const b = groupBounds.get(lane.id);
        if (!b) {
          continue;
        }
        const contentWidth = Math.max(0, b.maxX - b.minX) + 2 * topLaneHorizontalPadding(lane);
        const cx = (b.minX + b.maxX) / 2;
        laneIds.push(lane.id);
        centers.push(cx);
        baseWidths.push(contentWidth);
      }
      const count = laneIds.length;
      if (count > 0) {
        const laneWidths = /* @__PURE__ */ new Map();
        if (count === 1) {
          laneWidths.set(laneIds[0], baseWidths[0]);
        } else {
          const d = [];
          for (let i = 0; i < count - 1; i++) {
            d.push(centers[i + 1] - centers[i]);
          }
          const u = new Array(count);
          u[0] = 0;
          for (let i = 0; i < count - 1; i++) {
            u[i + 1] = 2 * d[i] - u[i];
          }
          let lowerBound = 0;
          let upperBound = Number.POSITIVE_INFINITY;
          for (let i = 0; i < count; i++) {
            const baseW = baseWidths[i];
            if (i % 2 === 0) {
              lowerBound = Math.max(lowerBound, baseW - u[i]);
            } else {
              upperBound = Math.min(upperBound, u[i] - baseW);
            }
          }
          let x = lowerBound;
          if (lowerBound <= upperBound) {
            x = (lowerBound + upperBound) / 2;
          } else {
            x = lowerBound;
          }
          for (let i = 0; i < count; i++) {
            const w = u[i] + (i % 2 === 0 ? x : -x);
            const finalWidth = Math.max(baseWidths[i], w);
            laneWidths.set(laneIds[i], finalWidth);
          }
        }
        for (const lane of topLevelGroups) {
          const w = laneWidths.get(lane.id);
          if (w != null) {
            lane.width = w;
          }
          assignTopLaneTitleRect(lane);
        }
      }
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(writeBackToLayoutData, "writeBackToLayoutData");

// src/rendering-util/layout-algorithms/swimlanes/edgeLabelNodes.ts
var EDGE_LABEL_LOG_PREFIX = "[EdgeLabelNodes]";
function createEdgeLabelNodes(data) {
  const nodesToAdd = [];
  const layoutOnlyEdges = [];
  const nodeById = /* @__PURE__ */ new Map();
  for (const node of data.nodes) {
    nodeById.set(node.id, node);
  }
  for (const edge of data.edges) {
    if (!edge.label || edge.label.length === 0) {
      continue;
    }
    if (edge.isLayoutOnly) {
      continue;
    }
    if (edge.labelNodeId) {
      continue;
    }
    const sourceNode = edge.start ? nodeById.get(edge.start) : void 0;
    const targetNode = edge.end ? nodeById.get(edge.end) : void 0;
    if (!sourceNode || !targetNode) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_10__/* .log */ .R.warn(EDGE_LABEL_LOG_PREFIX, `Edge ${edge.id} has missing source or target node`);
      continue;
    }
    const labelNodeId = `edge-label-${edge.start}-${edge.end}-${edge.id}`;
    const isCrossLane = sourceNode.parentId !== targetNode.parentId;
    const labelLane = isCrossLane ? targetNode.parentId : sourceNode.parentId;
    const labelNode = {
      id: labelNodeId,
      label: edge.label,
      edgeStart: edge.start ?? "",
      edgeEnd: edge.end ?? "",
      shape: "labelRect",
      width: 0,
      // populated when rendered / applied from fixture
      height: 0,
      isEdgeLabel: true,
      isDummy: true,
      parentId: labelLane,
      isGroup: false,
      labelStyle: Array.isArray(edge.labelStyle) ? edge.labelStyle[0] : edge.labelStyle ?? "",
      ...sourceNode.dir ? { dir: sourceNode.dir } : {}
    };
    nodesToAdd.push(labelNode);
    edge.labelNodeId = labelNodeId;
    edge.label = void 0;
    edge.text = void 0;
    const toLabelVirtual = {
      id: `${edge.id}-to-label`,
      start: edge.start,
      end: labelNodeId,
      type: "normal",
      isLayoutOnly: true
    };
    const fromLabelVirtual = {
      id: `${edge.id}-from-label`,
      start: labelNodeId,
      end: edge.end,
      type: "normal",
      isLayoutOnly: true
    };
    layoutOnlyEdges.push(toLabelVirtual, fromLabelVirtual);
  }
  const newNodes = [...data.nodes, ...nodesToAdd];
  const newEdges = [...data.edges, ...layoutOnlyEdges];
  return {
    ...data,
    nodes: newNodes,
    edges: newEdges
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(createEdgeLabelNodes, "createEdgeLabelNodes");

// src/rendering-util/layout-algorithms/swimlanes/direction/geometry.ts
var EPS = 1e-3;
function measuredNodeRect(node) {
  const cx = node.x ?? 0;
  const cy = node.y ?? 0;
  const width = node.width ?? 0;
  const height = node.height ?? 0;
  return width > 0 && height > 0 ? { cx, cy, rect: rectFromCenterSize(cx, cy, width, height) } : void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(measuredNodeRect, "measuredNodeRect");
function nodeBoundsInfoFor(node) {
  if (node.isGroup) {
    return void 0;
  }
  const measured = measuredNodeRect(node);
  if (!measured) {
    return void 0;
  }
  const id = String(node.id ?? "");
  return {
    id,
    cx: measured.cx,
    cy: measured.cy,
    rect: measured.rect
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(nodeBoundsInfoFor, "nodeBoundsInfoFor");
function samePoint(a, b, epsilon = EPS) {
  return Math.abs(a.x - b.x) < epsilon && Math.abs(a.y - b.y) < epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(samePoint, "samePoint");
function sameX(a, b, epsilon = EPS) {
  return Math.abs(a.x - b.x) < epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sameX, "sameX");
function sameY(a, b, epsilon = EPS) {
  return Math.abs(a.y - b.y) < epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sameY, "sameY");
function isHorizontalSegment(a, b, epsilon = EPS) {
  return sameY(a, b, epsilon) && Math.abs(a.x - b.x) > epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isHorizontalSegment, "isHorizontalSegment");
function isVerticalSegment(a, b, epsilon = EPS) {
  return sameX(a, b, epsilon) && Math.abs(a.y - b.y) > epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isVerticalSegment, "isVerticalSegment");
function overlapLength(a1, a2, b1, b2) {
  return Math.max(
    0,
    Math.min(Math.max(a1, a2), Math.max(b1, b2)) - Math.max(Math.min(a1, a2), Math.min(b1, b2))
  );
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(overlapLength, "overlapLength");
function sameAxisSegmentOverlapLength(a, b, epsilon = EPS) {
  if (a.horizontal && b.horizontal && sameY(a.a, b.a, epsilon)) {
    return overlapLength(a.a.x, a.b.x, b.a.x, b.b.x);
  }
  if (a.vertical && b.vertical && sameX(a.a, b.a, epsilon)) {
    return overlapLength(a.a.y, a.b.y, b.a.y, b.b.y);
  }
  return 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sameAxisSegmentOverlapLength, "sameAxisSegmentOverlapLength");
function orthogonalSegmentsForPoints(points, epsilon = EPS) {
  const result = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    const horizontal = isHorizontalSegment(a, b, epsilon);
    const vertical = isVerticalSegment(a, b, epsilon);
    if (horizontal || vertical) {
      result.push({ index: i, a, b, horizontal, vertical });
    }
  }
  return result;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(orthogonalSegmentsForPoints, "orthogonalSegmentsForPoints");
function countOrthogonalBends(points, epsilon = EPS) {
  const segments = orthogonalSegmentsForPoints(points, epsilon);
  let bends = 0;
  for (let i = 1; i < segments.length; i++) {
    if (segments[i - 1].horizontal !== segments[i].horizontal) {
      bends++;
    }
  }
  return bends;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(countOrthogonalBends, "countOrthogonalBends");
function dedupeConsecutivePoints(points, epsilon = EPS) {
  const result = [];
  for (const point of points) {
    const last = result.length > 0 ? result[result.length - 1] : void 0;
    if (!last || !samePoint(last, point, epsilon)) {
      result.push({ x: point.x, y: point.y });
    }
  }
  return result;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(dedupeConsecutivePoints, "dedupeConsecutivePoints");
function classifyThreeSegmentRoute(points, epsilon = EPS) {
  if (!points || points.length !== 4) {
    return void 0;
  }
  const [p0, p1, p2, p3] = points;
  const isHVH = isHorizontalSegment(p0, p1, epsilon) && isVerticalSegment(p1, p2, epsilon) && isHorizontalSegment(p2, p3, epsilon);
  if (isHVH) {
    return { kind: "HVH", p0, p1, p2, p3 };
  }
  const isVHV = isVerticalSegment(p0, p1, epsilon) && isHorizontalSegment(p1, p2, epsilon) && isVerticalSegment(p2, p3, epsilon);
  return isVHV ? { kind: "VHV", p0, p1, p2, p3 } : void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(classifyThreeSegmentRoute, "classifyThreeSegmentRoute");
function segmentBoundsOverlapRect(a, b, rect, buffer = 0) {
  const segMinX = Math.min(a.x, b.x);
  const segMaxX = Math.max(a.x, b.x);
  const segMinY = Math.min(a.y, b.y);
  const segMaxY = Math.max(a.y, b.y);
  return segMaxX > rect.left - buffer && segMinX < rect.right + buffer && segMaxY > rect.top - buffer && segMinY < rect.bottom + buffer;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentBoundsOverlapRect, "segmentBoundsOverlapRect");
function pointInsideRect(point, rect, buffer = 0) {
  return point.x > rect.left + buffer && point.x < rect.right - buffer && point.y > rect.top + buffer && point.y < rect.bottom - buffer;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(pointInsideRect, "pointInsideRect");
function rectContainsRect(outer, inner) {
  return outer.left <= inner.left && outer.right >= inner.right && outer.top <= inner.top && outer.bottom >= inner.bottom;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(rectContainsRect, "rectContainsRect");
function rectsOverlap(a, b) {
  return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(rectsOverlap, "rectsOverlap");
function inflateRect(rect, margin) {
  return {
    left: rect.left - margin,
    right: rect.right + margin,
    top: rect.top - margin,
    bottom: rect.bottom + margin
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(inflateRect, "inflateRect");
function rectFromCenterSize(cx, cy, width, height) {
  return {
    left: cx - width / 2,
    right: cx + width / 2,
    top: cy - height / 2,
    bottom: cy + height / 2
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(rectFromCenterSize, "rectFromCenterSize");
function rectOfNodeBounds(node) {
  return measuredNodeRect(node)?.rect;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(rectOfNodeBounds, "rectOfNodeBounds");
function portForRectSide(node, side) {
  switch (side) {
    case "top":
      return { x: node.cx, y: node.rect.top };
    case "bottom":
      return { x: node.cx, y: node.rect.bottom };
    case "left":
      return { x: node.rect.left, y: node.cy };
    case "right":
      return { x: node.rect.right, y: node.cy };
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(portForRectSide, "portForRectSide");
function buildOrthogonalPortPath(src, srcSide, dst, dstSide, anchor, epsilon = EPS) {
  const srcH = srcSide === "left" || srcSide === "right";
  const dstH = dstSide === "left" || dstSide === "right";
  if (srcH && dstH) {
    const opposingDir = srcSide === "right" && dstSide === "left" && src.x < dst.x || srcSide === "left" && dstSide === "right" && src.x > dst.x;
    if (opposingDir) {
      if (sameY(src, dst, epsilon)) {
        return [src, dst];
      }
      const midX = (src.x + dst.x) / 2;
      return [src, { x: midX, y: src.y }, { x: midX, y: dst.y }, dst];
    }
    if (srcSide === dstSide) {
      if (sameY(src, dst, epsilon)) {
        return void 0;
      }
      const intX = srcSide === "left" ? Math.min(src.x, dst.x) - anchor : Math.max(src.x, dst.x) + anchor;
      return [src, { x: intX, y: src.y }, { x: intX, y: dst.y }, dst];
    }
    return void 0;
  }
  if (!srcH && !dstH) {
    if (srcSide === dstSide) {
      if (sameX(src, dst, epsilon)) {
        return void 0;
      }
      const intY = srcSide === "top" ? Math.min(src.y, dst.y) - anchor : Math.max(src.y, dst.y) + anchor;
      return [src, { x: src.x, y: intY }, { x: dst.x, y: intY }, dst];
    }
    const sameDir = srcSide === "bottom" && dstSide === "top" && src.y < dst.y || srcSide === "top" && dstSide === "bottom" && src.y > dst.y;
    if (!sameDir) {
      return void 0;
    }
    if (sameX(src, dst, epsilon)) {
      return [src, dst];
    }
    const midY = (src.y + dst.y) / 2;
    return [src, { x: src.x, y: midY }, { x: dst.x, y: midY }, dst];
  }
  if (srcH && !dstH) {
    const sameDirSrc2 = srcSide === "right" && dst.x > src.x || srcSide === "left" && dst.x < src.x;
    const sameDirDst2 = dstSide === "top" && src.y < dst.y || dstSide === "bottom" && src.y > dst.y;
    return sameDirSrc2 && sameDirDst2 ? [src, { x: dst.x, y: src.y }, dst] : void 0;
  }
  const sameDirSrc = srcSide === "bottom" && dst.y > src.y || srcSide === "top" && dst.y < src.y;
  const sameDirDst = dstSide === "left" && src.x < dst.x || dstSide === "right" && src.x > dst.x;
  return sameDirSrc && sameDirDst ? [src, { x: src.x, y: dst.y }, dst] : void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildOrthogonalPortPath, "buildOrthogonalPortPath");
function buildSameSideTrackPath(src, side, dst, track) {
  return side === "left" || side === "right" ? [src, { x: track, y: src.y }, { x: track, y: dst.y }, dst] : [src, { x: src.x, y: track }, { x: dst.x, y: track }, dst];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildSameSideTrackPath, "buildSameSideTrackPath");
function collectRealNodeBounds(nodes) {
  const nodeInfoById = /* @__PURE__ */ new Map();
  const realNodeRects = [];
  for (const node of nodes) {
    if (node.isEdgeLabel) {
      continue;
    }
    const info = nodeBoundsInfoFor(node);
    if (!info) {
      continue;
    }
    nodeInfoById.set(info.id, info);
    realNodeRects.push({ id: info.id, rect: info.rect });
  }
  return { nodeInfoById, realNodeRects };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collectRealNodeBounds, "collectRealNodeBounds");
function collectNodeRectEntries(nodes) {
  const realNodeRects = [];
  const labelNodeRects = [];
  for (const node of nodes) {
    const info = nodeBoundsInfoFor(node);
    if (!info) {
      continue;
    }
    const entry = { id: info.id, rect: info.rect };
    if (node.isEdgeLabel) {
      labelNodeRects.push(entry);
    } else {
      realNodeRects.push(entry);
    }
  }
  return { realNodeRects, labelNodeRects };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collectNodeRectEntries, "collectNodeRectEntries");
function collectLayoutNodeRects(nodes, { includeEdgeLabels = true } = {}) {
  const result = [];
  for (const node of nodes) {
    if (node.isGroup || !includeEdgeLabels && node.isEdgeLabel) {
      continue;
    }
    const cx = node.x ?? 0;
    const cy = node.y ?? 0;
    const width = node.width ?? 0;
    const height = node.height ?? 0;
    result.push({
      nodeId: node.id,
      ...rectFromCenterSize(cx, cy, width, height)
    });
  }
  return result;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collectLayoutNodeRects, "collectLayoutNodeRects");
function getNodePairGeometry(edge, nodeInfoById, epsilon = EPS) {
  const srcId = edge.start;
  const dstId = edge.end;
  if (!srcId || !dstId) {
    return void 0;
  }
  const srcInfo = nodeInfoById.get(srcId);
  const dstInfo = nodeInfoById.get(dstId);
  if (!srcInfo || !dstInfo) {
    return void 0;
  }
  return {
    srcId,
    dstId,
    srcInfo,
    dstInfo,
    collinearX: Math.abs(srcInfo.cx - dstInfo.cx) < epsilon,
    collinearY: Math.abs(srcInfo.cy - dstInfo.cy) < epsilon
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(getNodePairGeometry, "getNodePairGeometry");
function segmentHitsAnyRect(a, b, rects, excludeIds = [], shrink = 0) {
  for (const entry of rects) {
    if (excludeIds.includes(entry.id)) {
      continue;
    }
    if (segmentBoundsOverlapRect(a, b, entry.rect, -shrink)) {
      return true;
    }
  }
  return false;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentHitsAnyRect, "segmentHitsAnyRect");
function orthogonalSegmentsCross(a1, b1, a2, b2, epsilon = EPS, endpointTolerance = 1e-6) {
  const s1H = sameY(a1, b1, epsilon);
  const s1V = sameX(a1, b1, epsilon);
  const s2H = sameY(a2, b2, epsilon);
  const s2V = sameX(a2, b2, epsilon);
  if (s1H && s2H || s1V && s2V) {
    return false;
  }
  if (!(s1H || s1V) || !(s2H || s2V)) {
    return false;
  }
  const horiz = s1H ? { a: a1, b: b1 } : { a: a2, b: b2 };
  const vert = s1V ? { a: a1, b: b1 } : { a: a2, b: b2 };
  const hY = horiz.a.y;
  const hX1 = Math.min(horiz.a.x, horiz.b.x);
  const hX2 = Math.max(horiz.a.x, horiz.b.x);
  const vX = vert.a.x;
  const vY1 = Math.min(vert.a.y, vert.b.y);
  const vY2 = Math.max(vert.a.y, vert.b.y);
  if (vX < hX1 || vX > hX2 || hY < vY1 || hY > vY2) {
    return false;
  }
  const matchesHorizEndpoint = Math.abs(vX - horiz.a.x) < endpointTolerance && Math.abs(hY - horiz.a.y) < endpointTolerance || Math.abs(vX - horiz.b.x) < endpointTolerance && Math.abs(hY - horiz.b.y) < endpointTolerance;
  const matchesVertEndpoint = Math.abs(vX - vert.a.x) < endpointTolerance && Math.abs(hY - vert.a.y) < endpointTolerance || Math.abs(vX - vert.b.x) < endpointTolerance && Math.abs(hY - vert.b.y) < endpointTolerance;
  return !(matchesHorizEndpoint && matchesVertEndpoint);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(orthogonalSegmentsCross, "orthogonalSegmentsCross");
function sameAxisSegmentsOverlap(a1, b1, a2, b2, epsilon = EPS) {
  const s1H = sameY(a1, b1, epsilon);
  const s1V = sameX(a1, b1, epsilon);
  const s2H = sameY(a2, b2, epsilon);
  const s2V = sameX(a2, b2, epsilon);
  if (s1V && s2V && sameX(a1, a2, epsilon)) {
    return overlapLength(a1.y, b1.y, a2.y, b2.y) > epsilon;
  }
  if (s1H && s2H && sameY(a1, a2, epsilon)) {
    return overlapLength(a1.x, b1.x, a2.x, b2.x) > epsilon;
  }
  return false;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sameAxisSegmentsOverlap, "sameAxisSegmentsOverlap");
function segmentConflictsWithAnyEdge(a, b, edges, excludeEdge, {
  epsilon = EPS,
  skipDegenerateOther = false
} = {}) {
  for (const other of edges) {
    if (other === excludeEdge || other.isLayoutOnly) {
      continue;
    }
    const points = other.points;
    if (!points || points.length < 2) {
      continue;
    }
    for (let i = 0; i < points.length - 1; i++) {
      const oa = points[i];
      const ob = points[i + 1];
      if (skipDegenerateOther && samePoint(oa, ob, epsilon)) {
        continue;
      }
      if (orthogonalSegmentsCross(a, b, oa, ob, epsilon) || sameAxisSegmentsOverlap(a, b, oa, ob, epsilon)) {
        return true;
      }
    }
  }
  return false;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentConflictsWithAnyEdge, "segmentConflictsWithAnyEdge");
function orthogonalSegmentsStrictlyCross(a1, b1, a2, b2, epsilon = EPS) {
  const aHoriz = sameY(a1, b1, epsilon);
  const aVert = sameX(a1, b1, epsilon);
  const bHoriz = sameY(a2, b2, epsilon);
  const bVert = sameX(a2, b2, epsilon);
  if (!(aHoriz && bVert || aVert && bHoriz)) {
    return false;
  }
  const horiz = aHoriz ? { a: a1, b: b1 } : { a: a2, b: b2 };
  const vert = aHoriz ? { a: a2, b: b2 } : { a: a1, b: b1 };
  const hY = horiz.a.y;
  const hXmin = Math.min(horiz.a.x, horiz.b.x);
  const hXmax = Math.max(horiz.a.x, horiz.b.x);
  const vX = vert.a.x;
  const vYmin = Math.min(vert.a.y, vert.b.y);
  const vYmax = Math.max(vert.a.y, vert.b.y);
  return vX > hXmin + epsilon && vX < hXmax - epsilon && hY > vYmin + epsilon && hY < vYmax - epsilon;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(orthogonalSegmentsStrictlyCross, "orthogonalSegmentsStrictlyCross");
function strictlyBetween(value, a, b) {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  return value > lo + EPS && value < hi - EPS;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(strictlyBetween, "strictlyBetween");
function isCollinearIntermediate(prev, cur, next) {
  if (sameX(prev, cur) && sameX(cur, next)) {
    return strictlyBetween(cur.y, prev.y, next.y);
  }
  if (sameY(prev, cur) && sameY(cur, next)) {
    return strictlyBetween(cur.x, prev.x, next.x);
  }
  return false;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isCollinearIntermediate, "isCollinearIntermediate");
function simplifyPolylineOnce(points) {
  let changed = false;
  const out = [];
  for (let i = 0; i < points.length; i++) {
    const prev = out[out.length - 1];
    const cur = points[i];
    const next = i + 1 < points.length ? points[i + 1] : void 0;
    if (prev && next) {
      if (samePoint(prev, next)) {
        i++;
        changed = true;
        continue;
      }
      if (isCollinearIntermediate(prev, cur, next)) {
        changed = true;
        continue;
      }
    }
    out.push(cur);
  }
  return { points: out, changed };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(simplifyPolylineOnce, "simplifyPolylineOnce");
function orthogonalizePolyline(pts) {
  const cleaned = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const prev = cleaned[cleaned.length - 1];
    const curr = pts[i];
    if (!sameX(prev, curr) && !sameY(prev, curr)) {
      const prevPrev = cleaned.length >= 2 ? cleaned[cleaned.length - 2] : void 0;
      const incomingVertical = prevPrev ? sameX(prevPrev, prev) : false;
      const corner = incomingVertical ? { x: prev.x, y: curr.y } : { x: curr.x, y: prev.y };
      cleaned.push(corner);
    }
    cleaned.push(curr);
  }
  const deduped = [];
  for (const p of cleaned) {
    const last = deduped[deduped.length - 1];
    if (!last || !samePoint(last, p)) {
      deduped.push(p);
    }
  }
  return deduped;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(orthogonalizePolyline, "orthogonalizePolyline");
function simplifyPolyline(pts) {
  if (pts.length < 3) {
    return pts;
  }
  let work = [...pts];
  for (let guard = 0; guard < 32; guard++) {
    const result = simplifyPolylineOnce(work);
    work = result.points;
    if (!result.changed) {
      break;
    }
  }
  return work;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(simplifyPolyline, "simplifyPolyline");

// src/rendering-util/layout-algorithms/swimlanes/direction/endpointClip.ts
var EPS2 = 1e-3;
var INSIDE_EPS = 0.5;
var CORNER_CLEARANCE = 4;
function endpointContextFor(edge, nodeByIdMap, minPoints) {
  const candidate = edge;
  if (candidate.isLayoutOnly || !candidate.points || candidate.points.length < minPoints) {
    return void 0;
  }
  const src = candidate.start ? nodeByIdMap.get(candidate.start) : void 0;
  const dst = candidate.end ? nodeByIdMap.get(candidate.end) : void 0;
  return {
    edge: candidate,
    points: candidate.points,
    srcRect: src ? rectOfNodeBounds(src) : void 0,
    dstRect: dst ? rectOfNodeBounds(dst) : void 0
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(endpointContextFor, "endpointContextFor");
function segmentEnterPoint(outside, inside, r) {
  if (sameY(outside, inside, EPS2)) {
    const x = outside.x < r.left ? r.left : r.right;
    return { x, y: outside.y };
  }
  if (sameX(outside, inside, EPS2)) {
    const y = outside.y < r.top ? r.top : r.bottom;
    return { x: outside.x, y };
  }
  return {
    x: Math.min(r.right, Math.max(r.left, outside.x)),
    y: Math.min(r.bottom, Math.max(r.top, outside.y))
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentEnterPoint, "segmentEnterPoint");
function clipEndpoint(points, rect, atStart) {
  const step = atStart ? 1 : -1;
  let outsideIndex = atStart ? 0 : points.length - 1;
  while (outsideIndex >= 0 && outsideIndex < points.length && pointInsideRect(points[outsideIndex], rect, INSIDE_EPS)) {
    outsideIndex += step;
  }
  if (outsideIndex < 0 || outsideIndex >= points.length) {
    return points;
  }
  const insideIndex = outsideIndex - step;
  if (insideIndex < 0 || insideIndex >= points.length) {
    return points;
  }
  const entry = segmentEnterPoint(points[outsideIndex], points[insideIndex], rect);
  return atStart ? [entry, ...points.slice(outsideIndex)] : [...points.slice(0, outsideIndex + 1), entry];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clipEndpoint, "clipEndpoint");
function clipEdgeEndpointsToNodeBoundaries(edges, nodeByIdMap) {
  for (const edge of edges) {
    const context = endpointContextFor(edge, nodeByIdMap, 2);
    if (!context) {
      continue;
    }
    let next = [...context.points];
    if (context.srcRect) {
      next = clipEndpoint(next, context.srcRect, true);
    }
    if (context.dstRect) {
      next = clipEndpoint(next, context.dstRect, false);
    }
    next = simplifyPolyline(orthogonalizePolyline(next));
    next = clearStraightEndpointCornerConnections(next, context.srcRect, context.dstRect);
    context.edge.points = simplifyPolyline(orthogonalizePolyline(next));
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clipEdgeEndpointsToNodeBoundaries, "clipEdgeEndpointsToNodeBoundaries");
function snapEndpointToBoundary(inner, endpoint, r, useApproachSide = false) {
  if (sameY(inner, endpoint, EPS2)) {
    if (endpoint.y < r.top - EPS2 || endpoint.y > r.bottom + EPS2) {
      return endpoint;
    }
    if (useApproachSide) {
      if (inner.x < r.left - EPS2) {
        return { x: r.left, y: inner.y };
      }
      if (inner.x > r.right + EPS2) {
        return { x: r.right, y: inner.y };
      }
    }
    const toLeft = Math.abs(endpoint.x - r.left) <= Math.abs(endpoint.x - r.right);
    return { x: toLeft ? r.left : r.right, y: inner.y };
  }
  if (sameX(inner, endpoint, EPS2)) {
    if (endpoint.x < r.left - EPS2 || endpoint.x > r.right + EPS2) {
      return endpoint;
    }
    if (useApproachSide) {
      if (inner.y < r.top - EPS2) {
        return { x: inner.x, y: r.top };
      }
      if (inner.y > r.bottom + EPS2) {
        return { x: inner.x, y: r.bottom };
      }
    }
    const toTop = Math.abs(endpoint.y - r.top) <= Math.abs(endpoint.y - r.bottom);
    return { x: inner.x, y: toTop ? r.top : r.bottom };
  }
  return endpoint;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(snapEndpointToBoundary, "snapEndpointToBoundary");
function firstDistinctAdjacent(points, endpointIndex, step) {
  const endpoint = points[endpointIndex];
  for (let index = endpointIndex + step; index >= 0 && index < points.length; index += step) {
    const candidate = points[index];
    if (!samePoint(candidate, endpoint, EPS2)) {
      return candidate;
    }
  }
  return points[endpointIndex + step];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(firstDistinctAdjacent, "firstDistinctAdjacent");
function cornerClearanceRange(min, max) {
  const lo = min + CORNER_CLEARANCE;
  const hi = max - CORNER_CLEARANCE;
  return lo <= hi ? { lo, hi } : { lo: (min + max) / 2, hi: (min + max) / 2 };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(cornerClearanceRange, "cornerClearanceRange");
function clampToCornerClearance(value, min, max) {
  const { lo, hi } = cornerClearanceRange(min, max);
  return Math.min(hi, Math.max(lo, value));
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clampToCornerClearance, "clampToCornerClearance");
function intersectRanges(ranges) {
  const lo = Math.max(...ranges.map((range) => range.lo));
  const hi = Math.min(...ranges.map((range) => range.hi));
  if (lo > hi) {
    return void 0;
  }
  return { lo, hi };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(intersectRanges, "intersectRanges");
function clearanceRangeForSide(r, side) {
  return side === "left" || side === "right" ? cornerClearanceRange(r.top, r.bottom) : cornerClearanceRange(r.left, r.right);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clearanceRangeForSide, "clearanceRangeForSide");
function terminalSideForSegment(endpoint, adjacent, r) {
  const yWithin = endpoint.y >= r.top - EPS2 && endpoint.y <= r.bottom + EPS2;
  const xWithin = endpoint.x >= r.left - EPS2 && endpoint.x <= r.right + EPS2;
  if (sameY(endpoint, adjacent, EPS2) && yWithin) {
    if (Math.abs(endpoint.x - r.left) < EPS2) {
      return "left";
    }
    if (Math.abs(endpoint.x - r.right) < EPS2) {
      return "right";
    }
  }
  if (sameX(endpoint, adjacent, EPS2) && xWithin) {
    if (Math.abs(endpoint.y - r.top) < EPS2) {
      return "top";
    }
    if (Math.abs(endpoint.y - r.bottom) < EPS2) {
      return "bottom";
    }
  }
  return void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(terminalSideForSegment, "terminalSideForSegment");
function isHorizontalSide(side) {
  return side === "left" || side === "right";
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isHorizontalSide, "isHorizontalSide");
function straightClearanceRange(start, end, srcRect, dstRect, horizontal) {
  const ranges = [];
  const srcSide = srcRect ? terminalSideForSegment(start, end, srcRect) : void 0;
  const dstSide = dstRect ? terminalSideForSegment(end, start, dstRect) : void 0;
  if (srcRect && srcSide && isHorizontalSide(srcSide) === horizontal) {
    ranges.push(clearanceRangeForSide(srcRect, srcSide));
  }
  if (dstRect && dstSide && isHorizontalSide(dstSide) === horizontal) {
    ranges.push(clearanceRangeForSide(dstRect, dstSide));
  }
  return ranges.length > 0 ? intersectRanges(ranges) : void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(straightClearanceRange, "straightClearanceRange");
function clearStraightEndpointCornerAxis(start, end, srcRect, dstRect, horizontal) {
  const range = straightClearanceRange(start, end, srcRect, dstRect, horizontal);
  if (!range) {
    return void 0;
  }
  const current = horizontal ? start.y : start.x;
  const next = Math.min(range.hi, Math.max(range.lo, current));
  if (Math.abs(next - current) < EPS2) {
    return void 0;
  }
  return horizontal ? [
    { x: start.x, y: next },
    { x: end.x, y: next }
  ] : [
    { x: next, y: start.y },
    { x: next, y: end.y }
  ];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clearStraightEndpointCornerAxis, "clearStraightEndpointCornerAxis");
function clearStraightEndpointCornerConnections(points, srcRect, dstRect) {
  if (points.length !== 2) {
    return points;
  }
  const [start, end] = points;
  if (sameY(start, end, EPS2)) {
    return clearStraightEndpointCornerAxis(start, end, srcRect, dstRect, true) ?? points;
  }
  if (sameX(start, end, EPS2)) {
    return clearStraightEndpointCornerAxis(start, end, srcRect, dstRect, false) ?? points;
  }
  return points;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clearStraightEndpointCornerConnections, "clearStraightEndpointCornerConnections");
function cornerClearedEndpoint(endpoint, r, side) {
  return isHorizontalSide(side) ? { x: endpoint.x, y: clampToCornerClearance(endpoint.y, r.top, r.bottom) } : { x: clampToCornerClearance(endpoint.x, r.left, r.right), y: endpoint.y };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(cornerClearedEndpoint, "cornerClearedEndpoint");
function moveCollinearEndpointRun(points, endpointIndex, step, endpoint, adjusted, horizontalTerminal) {
  const next = points.map((point) => ({ ...point }));
  for (let index = endpointIndex; index >= 0 && index < points.length; index += step) {
    const point = points[index];
    if (horizontalTerminal && !sameY(point, endpoint, EPS2)) {
      break;
    }
    if (!horizontalTerminal && !sameX(point, endpoint, EPS2)) {
      break;
    }
    if (horizontalTerminal) {
      next[index].y = adjusted.y;
    } else {
      next[index].x = adjusted.x;
    }
  }
  return next;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(moveCollinearEndpointRun, "moveCollinearEndpointRun");
function clearEndpointCornerConnection(points, r, atStart) {
  if (points.length < 2) {
    return points;
  }
  const endpointIndex = atStart ? 0 : points.length - 1;
  const step = atStart ? 1 : -1;
  const endpoint = points[endpointIndex];
  const adjacent = firstDistinctAdjacent(points, endpointIndex, step);
  if (!adjacent) {
    return points;
  }
  const side = terminalSideForSegment(endpoint, adjacent, r);
  if (!side) {
    return points;
  }
  const horizontalTerminal = isHorizontalSide(side);
  const adjusted = cornerClearedEndpoint(endpoint, r, side);
  if (samePoint(endpoint, adjusted, EPS2)) {
    return points;
  }
  return moveCollinearEndpointRun(
    points,
    endpointIndex,
    step,
    endpoint,
    adjusted,
    horizontalTerminal
  );
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(clearEndpointCornerConnection, "clearEndpointCornerConnection");
function borderSideForSegment(a, b, r) {
  const xWithin = Math.min(a.x, b.x) >= r.left - EPS2 && Math.max(a.x, b.x) <= r.right + EPS2;
  const yWithin = Math.min(a.y, b.y) >= r.top - EPS2 && Math.max(a.y, b.y) <= r.bottom + EPS2;
  if (Math.abs(a.y - r.top) < EPS2 && Math.abs(b.y - r.top) < EPS2 && xWithin) {
    return "top";
  }
  if (Math.abs(a.y - r.bottom) < EPS2 && Math.abs(b.y - r.bottom) < EPS2 && xWithin) {
    return "bottom";
  }
  if (Math.abs(a.x - r.left) < EPS2 && Math.abs(b.x - r.left) < EPS2 && yWithin) {
    return "left";
  }
  if (Math.abs(a.x - r.right) < EPS2 && Math.abs(b.x - r.right) < EPS2 && yWithin) {
    return "right";
  }
  return void 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(borderSideForSegment, "borderSideForSegment");
function leavesOutward(side, from, to, r) {
  switch (side) {
    case "top":
      return sameX(from, to, EPS2) && to.y < r.top - EPS2;
    case "bottom":
      return sameX(from, to, EPS2) && to.y > r.bottom + EPS2;
    case "left":
      return sameY(from, to, EPS2) && to.x < r.left - EPS2;
    case "right":
      return sameY(from, to, EPS2) && to.x > r.right + EPS2;
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(leavesOutward, "leavesOutward");
function collapseOwnBorderStub(points, r, atStart) {
  if (points.length < 3) {
    return points;
  }
  if (atStart) {
    const side2 = borderSideForSegment(points[0], points[1], r);
    if (side2 && leavesOutward(side2, points[1], points[2], r)) {
      return points.slice(1);
    }
    return points;
  }
  const last = points.length - 1;
  const side = borderSideForSegment(points[last - 1], points[last], r);
  if (side && leavesOutward(side, points[last - 1], points[last - 2], r)) {
    return points.slice(0, last);
  }
  return points;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collapseOwnBorderStub, "collapseOwnBorderStub");
function snapAndCollapseEndpoints(points, srcRect, dstRect) {
  let next = points;
  if (srcRect) {
    const adjacent = firstDistinctAdjacent(next, 0, 1);
    if (adjacent) {
      const snapped = snapEndpointToBoundary(adjacent, next[0], srcRect);
      if (snapped !== next[0]) {
        next = [snapped, ...next.slice(1)];
      }
    }
    next = collapseOwnBorderStub(next, srcRect, true);
  }
  if (dstRect) {
    const last = next.length - 1;
    const adjacent = firstDistinctAdjacent(next, last, -1);
    if (adjacent) {
      const snapped = snapEndpointToBoundary(adjacent, next[last], dstRect, true);
      if (snapped !== next[last]) {
        next = [...next.slice(0, last), snapped];
      }
    }
    next = collapseOwnBorderStub(next, dstRect, false);
  }
  const straightCleared = clearStraightEndpointCornerConnections(next, srcRect, dstRect);
  if (straightCleared !== next || next.length === 2) {
    return straightCleared;
  }
  if (srcRect) {
    next = clearEndpointCornerConnection(next, srcRect, true);
  }
  if (dstRect) {
    next = clearEndpointCornerConnection(next, dstRect, false);
  }
  return next;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(snapAndCollapseEndpoints, "snapAndCollapseEndpoints");
function prepareEdgeEndpointsForRenderer(edges, nodeByIdMap) {
  for (const edge of edges) {
    const context = endpointContextFor(edge, nodeByIdMap, 2);
    if (!context) {
      continue;
    }
    const input = dedupeConsecutivePoints(context.points, EPS2);
    const newPts = snapAndCollapseEndpoints(input, context.srcRect, context.dstRect);
    if (newPts.length < 3) {
      context.edge.points = newPts;
      continue;
    }
    const duplicated = [
      newPts[0],
      { ...newPts[0] },
      ...newPts.slice(1, -1),
      newPts[newPts.length - 1],
      { ...newPts[newPts.length - 1] }
    ];
    context.edge.points = duplicated;
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(prepareEdgeEndpointsForRenderer, "prepareEdgeEndpointsForRenderer");

// src/rendering-util/layout-algorithms/swimlanes/direction/lrTransform.ts
function buildNodeMap(nodes) {
  return new Map(nodes.map((node) => [node.id, node]));
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildNodeMap, "buildNodeMap");
function resolveTopLevelGroupId(node, nodeById) {
  let parentId = node.parentId;
  let topLevelGroupId = null;
  while (parentId) {
    const parent = nodeById.get(parentId);
    if (!parent?.isGroup) {
      break;
    }
    topLevelGroupId = parent.id;
    parentId = parent.parentId;
  }
  return topLevelGroupId;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(resolveTopLevelGroupId, "resolveTopLevelGroupId");
function groupDepth(group, nodeById) {
  let depth = 0;
  let parentId = group.parentId;
  while (parentId) {
    const parent = nodeById.get(parentId);
    if (!parent?.isGroup) {
      break;
    }
    depth++;
    parentId = parent.parentId;
  }
  return depth;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(groupDepth, "groupDepth");
function boundsForChildren(children) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const child of children) {
    const cx = child.x;
    const cy = child.y;
    if (typeof cx !== "number" || typeof cy !== "number") {
      continue;
    }
    const w = child.width ?? 0;
    const h = child.height ?? 0;
    minX = Math.min(minX, cx - w / 2);
    maxX = Math.max(maxX, cx + w / 2);
    minY = Math.min(minY, cy - h / 2);
    maxY = Math.max(maxY, cy + h / 2);
  }
  if (minX === Infinity || minY === Infinity) {
    return null;
  }
  return { minX, maxX, minY, maxY };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(boundsForChildren, "boundsForChildren");
function applyGroupBounds(group, bounds) {
  const pad = group.padding ?? 20;
  group.x = (bounds.minX + bounds.maxX) / 2;
  group.y = (bounds.minY + bounds.maxY) / 2;
  group.width = Math.max(0, bounds.maxX - bounds.minX) + pad;
  group.height = Math.max(0, bounds.maxY - bounds.minY) + pad;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(applyGroupBounds, "applyGroupBounds");
function recomputeNestedGroupBounds(nodes) {
  const nodeById = buildNodeMap(nodes);
  const groupsByDepth = nodes.filter((node) => node.isGroup && node.parentId).sort((a, b) => groupDepth(b, nodeById) - groupDepth(a, nodeById));
  for (const group of groupsByDepth) {
    const children = nodes.filter((node) => node.parentId === group.id);
    const bounds = boundsForChildren(children);
    if (bounds) {
      applyGroupBounds(group, bounds);
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(recomputeNestedGroupBounds, "recomputeNestedGroupBounds");
function mirrorAxis(layout, axis) {
  const nodes = layout.nodes ?? [];
  const edges = layout.edges ?? [];
  const contentNodes = nodes.filter((node) => !node.isGroup);
  let min = Infinity;
  let max = -Infinity;
  for (const node of contentNodes) {
    const value = node[axis];
    if (typeof value !== "number") {
      continue;
    }
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return false;
  }
  const mirror = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((value) => min + max - value, "mirror");
  for (const node of nodes) {
    const value = node[axis];
    if (typeof value === "number") {
      node[axis] = mirror(value);
    }
    const titleRect = node.groupTitleRect;
    if (titleRect) {
      node.groupTitleRect = axis === "x" ? {
        ...titleRect,
        left: mirror(titleRect.right),
        right: mirror(titleRect.left)
      } : {
        ...titleRect,
        top: mirror(titleRect.bottom),
        bottom: mirror(titleRect.top)
      };
    }
  }
  for (const edge of edges) {
    for (const point of edge.points ?? []) {
      point[axis] = mirror(point[axis]);
    }
  }
  return true;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(mirrorAxis, "mirrorAxis");
function applyBtDirectionTransform(layout) {
  const nodes = layout.nodes ?? [];
  if (!nodes.some((node) => !node.isGroup)) {
    return true;
  }
  return mirrorAxis(layout, "y");
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(applyBtDirectionTransform, "applyBtDirectionTransform");
function applyLrDirectionTransform(layout, direction = "LR") {
  const nodes = layout.nodes ?? [];
  const edges = layout.edges ?? [];
  const contentNodes = nodes.filter((n) => !n.isGroup);
  let minX = Infinity;
  let minY = Infinity;
  for (const n of contentNodes) {
    const x0 = n.x ?? 0;
    const y0 = n.y ?? 0;
    if (x0 < minX) {
      minX = x0;
    }
    if (y0 < minY) {
      minY = y0;
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return false;
  }
  const titleBandSize = 36;
  let totalWidth = 0;
  let totalHeight = 0;
  for (const n of contentNodes) {
    totalWidth += n.width ?? 0;
    totalHeight += n.height ?? 0;
  }
  const avgWidth = totalWidth / contentNodes.length;
  const avgHeight = totalHeight / contentNodes.length;
  const horizontalScaleFactor = avgHeight > 0 ? Math.max(1, avgWidth / avgHeight) : 1;
  for (const n of contentNodes) {
    const x0 = n.x ?? 0;
    const y0 = n.y ?? 0;
    const newX = (y0 - minY) * horizontalScaleFactor + titleBandSize;
    const newY = x0 - minX;
    n.x = newX;
    n.y = newY;
  }
  for (const e of edges) {
    if (!e.points) {
      continue;
    }
    for (const p of e.points) {
      const x0 = p.x;
      const y0 = p.y;
      const newX = (y0 - minY) * horizontalScaleFactor + titleBandSize;
      const newY = x0 - minX;
      p.x = newX;
      p.y = newY;
    }
  }
  recomputeNestedGroupBounds(nodes);
  const laneNodes = nodes.filter((n) => n.isGroup && !n.parentId);
  if (laneNodes.length === 0) {
    if (direction === "RL") {
      mirrorAxis(layout, "x");
    }
    return true;
  }
  const nodeById = buildNodeMap(nodes);
  const childrenByLane = /* @__PURE__ */ new Map();
  for (const n of nodes) {
    if (n.isGroup) {
      continue;
    }
    const laneId = resolveTopLevelGroupId(n, nodeById);
    if (!laneId) {
      continue;
    }
    const bucket = childrenByLane.get(laneId) ?? [];
    bucket.push(n);
    childrenByLane.set(laneId, bucket);
  }
  let maxPad = 0;
  for (const lane of laneNodes) {
    const pad = lane.padding ?? 0;
    if (pad > maxPad) {
      maxPad = pad;
    }
  }
  const laneBounds = [];
  let globalMinXChild = Infinity;
  let globalMaxXChild = -Infinity;
  for (const lane of laneNodes) {
    const children = childrenByLane.get(lane.id) ?? [];
    const bounds = boundsForChildren(children);
    if (!bounds) {
      continue;
    }
    globalMinXChild = Math.min(globalMinXChild, bounds.minX);
    globalMaxXChild = Math.max(globalMaxXChild, bounds.maxX);
    laneBounds.push({
      lane,
      contentTop: bounds.minY,
      contentBottom: bounds.maxY,
      centerY: (bounds.minY + bounds.maxY) / 2
    });
  }
  if (globalMinXChild === Infinity || globalMaxXChild === -Infinity) {
    return true;
  }
  const fullContentWidth = Math.max(0, globalMaxXChild - globalMinXChild);
  const horizontalMargin = Math.max(maxPad, 10);
  const bodyWidth = fullContentWidth + 2 * horizontalMargin;
  const laneWidth = titleBandSize + bodyWidth;
  const bodyCenter = (globalMinXChild + globalMaxXChild) / 2;
  const bodyLeft = bodyCenter - bodyWidth / 2;
  const laneLeft = bodyLeft - titleBandSize;
  const centerX = laneLeft + laneWidth / 2;
  const verticalMargin = Math.max(maxPad, titleBandSize);
  laneBounds.sort((a, b) => a.centerY - b.centerY);
  for (let i = 0; i < laneBounds.length; i++) {
    const curr = laneBounds[i];
    let laneTop;
    let laneBottom;
    if (i === 0) {
      laneTop = curr.contentTop - verticalMargin;
    } else {
      const prev = laneBounds[i - 1];
      laneTop = (prev.contentBottom + curr.contentTop) / 2;
    }
    if (i === laneBounds.length - 1) {
      laneBottom = curr.contentBottom + verticalMargin;
    } else {
      const next = laneBounds[i + 1];
      laneBottom = (curr.contentBottom + next.contentTop) / 2;
    }
    const laneHeight = Math.max(0, laneBottom - laneTop);
    const centerY = (laneTop + laneBottom) / 2;
    curr.lane.x = centerX;
    curr.lane.y = centerY;
    curr.lane.width = laneWidth;
    curr.lane.height = laneHeight;
    curr.lane.swimlaneContentTop = curr.contentTop;
    curr.lane.groupTitleRect = {
      left: laneLeft,
      right: laneLeft + titleBandSize,
      top: laneTop,
      bottom: laneBottom
    };
  }
  if (direction === "RL") {
    mirrorAxis(layout, "x");
  }
  return true;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(applyLrDirectionTransform, "applyLrDirectionTransform");

// src/rendering-util/layout-algorithms/swimlanes/direction/portSwap.ts
var EPS3 = 1e-6;
var MIN_PORT_SPACING = 8;
var PORT_SHIFT = MIN_PORT_SPACING;
var TRY_DELTAS = [0, PORT_SHIFT, -PORT_SHIFT, 2 * PORT_SHIFT, -2 * PORT_SHIFT];
function portSwapToLShape(edges, nodes) {
  const { nodeInfoById, realNodeRects } = collectRealNodeBounds(nodes);
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const pts = edge.points;
    if (!pts || pts.length < 4) {
      continue;
    }
    const route = classifyThreeSegmentRoute(dedupeConsecutivePoints(pts, EPS3), EPS3);
    if (!route) {
      continue;
    }
    const { p3 } = route;
    const isHVH = route.kind === "HVH";
    const nodePair = getNodePairGeometry(edge, nodeInfoById, EPS3);
    if (!nodePair) {
      continue;
    }
    const { srcId, dstId, srcInfo, dstInfo, collinearX, collinearY } = nodePair;
    if (collinearX || collinearY) {
      continue;
    }
    let newPts;
    const srcRect = srcInfo.rect;
    for (const delta of TRY_DELTAS) {
      let np0;
      let np1;
      let np2;
      if (isHVH) {
        const dstBelow = dstInfo.cy > srcInfo.cy;
        const newSrcY = dstBelow ? srcRect.bottom : srcRect.top;
        const newSrcX = srcInfo.cx + delta;
        if (newSrcX <= srcRect.left + EPS3 || newSrcX >= srcRect.right - EPS3) {
          continue;
        }
        np0 = { x: newSrcX, y: newSrcY };
        np1 = { x: newSrcX, y: p3.y };
        np2 = { x: p3.x, y: p3.y };
      } else {
        const dstEast = dstInfo.cx > srcInfo.cx;
        const newSrcX = dstEast ? srcRect.right : srcRect.left;
        const newSrcY = srcInfo.cy + delta;
        if (newSrcY <= srcRect.top + EPS3 || newSrcY >= srcRect.bottom - EPS3) {
          continue;
        }
        np0 = { x: newSrcX, y: newSrcY };
        np1 = { x: p3.x, y: newSrcY };
        np2 = { x: p3.x, y: p3.y };
      }
      const firstSegDegenerate = samePoint(np0, np1, EPS3);
      const secondSegDegenerate = samePoint(np1, np2, EPS3);
      if (firstSegDegenerate && secondSegDegenerate) {
        continue;
      }
      if (!firstSegDegenerate && segmentHitsAnyRect(np0, np1, realNodeRects, [srcId], 1)) {
        continue;
      }
      if (!secondSegDegenerate && segmentHitsAnyRect(np1, np2, realNodeRects, [dstId], 1)) {
        continue;
      }
      const firstSegConflicts = !firstSegDegenerate && segmentConflictsWithAnyEdge(np0, np1, edges, edge, {
        epsilon: EPS3,
        skipDegenerateOther: true
      });
      const secondSegConflicts = !secondSegDegenerate && segmentConflictsWithAnyEdge(np1, np2, edges, edge, {
        epsilon: EPS3,
        skipDegenerateOther: true
      });
      if (firstSegConflicts || secondSegConflicts) {
        continue;
      }
      if (firstSegDegenerate) {
        newPts = [np1, np2];
      } else if (secondSegDegenerate) {
        newPts = [np0, np1];
      } else {
        newPts = [np0, np1, np2];
      }
      break;
    }
    if (newPts) {
      edge.points = newPts;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(portSwapToLShape, "portSwapToLShape");

// src/rendering-util/layout-algorithms/swimlanes/direction/terminalStub.ts
function collapseShortTerminalStub(edges, nodeByIdMap) {
  const MIN_STUB = 10;
  const EPS_LOCAL2 = 1e-3;
  const BUFFER = 2;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const rawPts = edge.points;
    if (!rawPts || rawPts.length < 4) {
      continue;
    }
    const pts = dedupeConsecutivePoints(rawPts, EPS_LOCAL2);
    if (pts.length < 4) {
      continue;
    }
    const nLast = pts.length - 1;
    const endPt = pts[nLast];
    const penultPt = pts[nLast - 1];
    const prevPt = pts[nLast - 2];
    const lastDx = endPt.x - penultPt.x;
    const lastDy = endPt.y - penultPt.y;
    const lastLen = Math.hypot(lastDx, lastDy);
    if (lastLen >= MIN_STUB || lastLen < EPS_LOCAL2) {
      continue;
    }
    const penultDx = penultPt.x - prevPt.x;
    const penultDy = penultPt.y - prevPt.y;
    const penultLen = Math.hypot(penultDx, penultDy);
    if (penultLen < EPS_LOCAL2) {
      continue;
    }
    const lastIsHoriz = isHorizontalSegment(penultPt, endPt, EPS_LOCAL2);
    const lastIsVert = isVerticalSegment(penultPt, endPt, EPS_LOCAL2);
    const penultIsHoriz = isHorizontalSegment(prevPt, penultPt, EPS_LOCAL2);
    const penultIsVert = isVerticalSegment(prevPt, penultPt, EPS_LOCAL2);
    if (!(lastIsHoriz && penultIsVert || lastIsVert && penultIsHoriz)) {
      continue;
    }
    const dstId = edge.end;
    const srcId = edge.start;
    const dst = dstId ? nodeByIdMap.get(dstId) : void 0;
    if (!dst) {
      continue;
    }
    const dstCx = dst.x ?? 0;
    const dstCy = dst.y ?? 0;
    const dstRect = rectOfNodeBounds(dst);
    if (!dstRect) {
      continue;
    }
    let newPrev;
    let newEnd;
    if (penultIsVert) {
      const approachFromBelow = penultDy < 0;
      newPrev = { x: dstCx, y: prevPt.y };
      newEnd = { x: dstCx, y: approachFromBelow ? dstRect.bottom : dstRect.top };
    } else {
      const approachFromLeft = penultDx > 0;
      newPrev = { x: prevPt.x, y: dstCy };
      newEnd = { x: approachFromLeft ? dstRect.right : dstRect.left, y: dstCy };
    }
    if (segmentHitsAnyRect(newPrev, newEnd, realNodeRects, dstId ? [dstId] : [], -BUFFER)) {
      continue;
    }
    if (segmentHitsAnyRect(newPrev, newEnd, labelRects, [], -BUFFER)) {
      continue;
    }
    if (srcId) {
      const src = nodeByIdMap.get(srcId);
      const srcRect = src ? rectOfNodeBounds(src) : void 0;
      if (srcRect && pointInsideRect(newPrev, srcRect, BUFFER)) {
        continue;
      }
    }
    const ownSegmentKey = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => `${a.x.toFixed(3)},${a.y.toFixed(3)}|${b.x.toFixed(3)},${b.y.toFixed(3)}`, "ownSegmentKey");
    const selfSegments = /* @__PURE__ */ new Set();
    for (let i = 0; i < pts.length - 1; i++) {
      selfSegments.add(ownSegmentKey(pts[i], pts[i + 1]));
    }
    const segmentCrossesOtherEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((from, to) => {
      for (const other of edges) {
        if (other === edge) {
          continue;
        }
        if (other.isLayoutOnly) {
          continue;
        }
        const oPts = other.points;
        if (!oPts || oPts.length < 2) {
          continue;
        }
        for (let i = 0; i < oPts.length - 1; i++) {
          const a = oPts[i];
          const b = oPts[i + 1];
          if (selfSegments.has(ownSegmentKey(a, b))) {
            continue;
          }
          if (orthogonalSegmentsStrictlyCross(from, to, a, b, EPS_LOCAL2)) {
            return true;
          }
        }
      }
      return false;
    }, "segmentCrossesOtherEdge");
    if (segmentCrossesOtherEdge(newPrev, newEnd)) {
      continue;
    }
    if (nLast - 3 >= 0) {
      const beforePrev = pts[nLast - 3];
      const endpointIds = [srcId, dstId].filter((id) => Boolean(id));
      if (segmentHitsAnyRect(beforePrev, newPrev, realNodeRects, endpointIds, -BUFFER)) {
        continue;
      }
      if (segmentCrossesOtherEdge(beforePrev, newPrev)) {
        continue;
      }
    }
    const head = pts.slice(0, nLast - 2);
    const newPts = [...head, newPrev, newEnd];
    edge.points = newPts;
    const labelId = edge.labelNodeId;
    if (labelId) {
      const labelNode = nodeByIdMap.get(labelId);
      if (labelNode) {
        const lw = labelNode.width ?? 0;
        const lh = labelNode.height ?? 0;
        if (lw > 0 && lh > 0) {
          let bestMidX;
          let bestMidY;
          let bestLen = -1;
          for (let i = 0; i < newPts.length - 1; i++) {
            const a = newPts[i];
            const b = newPts[i + 1];
            const segLen = Math.hypot(b.x - a.x, b.y - a.y);
            const isHoriz = sameY(a, b, EPS_LOCAL2);
            const isVert = sameX(a, b, EPS_LOCAL2);
            const fits = isHoriz && segLen >= lw + 2 || isVert && segLen >= lh + 2;
            if (!fits) {
              continue;
            }
            if (segLen > bestLen) {
              bestLen = segLen;
              bestMidX = (a.x + b.x) / 2;
              bestMidY = (a.y + b.y) / 2;
            }
          }
          if (bestMidX !== void 0 && bestMidY !== void 0) {
            labelNode.x = bestMidX;
            labelNode.y = bestMidY;
          }
        }
      }
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collapseShortTerminalStub, "collapseShortTerminalStub");

// src/rendering-util/layout-algorithms/swimlanes/direction/materializedGeometry.ts
var EPS_LOCAL = 1e-3;
var MIN_SHARED = 8;
var segmentsFor = orthogonalSegmentsForPoints;
var orthogonallyAligned = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => sameX(a, b, EPS_LOCAL) || sameY(a, b, EPS_LOCAL), "orthogonallyAligned");
function separateSharedRenderedTerminalLanes(edges, nodeByIdMap) {
  const MIN_FACE_CLEARANCE = 16;
  const TRACK_SHIFT = 7;
  const rectIntersect = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, point) => {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    const dx = point.x - x;
    const dy = point.y - y;
    let w = (node.width ?? 0) / 2;
    let h = (node.height ?? 0) / 2;
    if (Math.abs(dy) * w > Math.abs(dx) * h) {
      if (dy < 0) {
        h = -h;
      }
      return { x: x + (dy === 0 ? 0 : h * dx / dy), y: y + h };
    }
    if (dx < 0) {
      w = -w;
    }
    return { x: x + w, y: y + (dx === 0 ? 0 : w * dy / dx) };
  }, "rectIntersect");
  const terminalLaneFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, atStart) => {
    const points = dedupeConsecutivePoints(edge.points ?? []);
    if (points.length < 2) {
      return void 0;
    }
    const nodeId = atStart ? edge.start : edge.end;
    const node = nodeId ? nodeByIdMap.get(nodeId) : void 0;
    const rect = node ? rectOfNodeBounds(node) : void 0;
    if (!node || !nodeId || !rect) {
      return void 0;
    }
    const endpoint = atStart ? points[0] : points[points.length - 1];
    const adjacent = atStart ? points[1] : points[points.length - 2];
    const boundary = rectIntersect(node, endpoint);
    let railEnd = endpoint;
    if (orthogonallyAligned(adjacent, boundary)) {
      railEnd = adjacent;
    }
    if (sameX(boundary, railEnd, EPS_LOCAL)) {
      return {
        edge,
        edgeId: String(edge.id ?? ""),
        nodeId,
        atStart,
        orientation: "V",
        coord: boundary.x,
        min: Math.min(boundary.y, railEnd.y),
        max: Math.max(boundary.y, railEnd.y),
        boundary,
        railEnd,
        rect
      };
    }
    if (sameY(boundary, railEnd, EPS_LOCAL)) {
      return {
        edge,
        edgeId: String(edge.id ?? ""),
        nodeId,
        atStart,
        orientation: "H",
        coord: boundary.y,
        min: Math.min(boundary.x, railEnd.x),
        max: Math.max(boundary.x, railEnd.x),
        boundary,
        railEnd,
        rect
      };
    }
    return void 0;
  }, "terminalLaneFor");
  const projectedOverlapLength = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => Math.max(0, Math.min(a.max, b.max) - Math.max(a.min, b.min)), "projectedOverlapLength");
  const sameTerminalFace = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => {
    if (a.nodeId !== b.nodeId || a.orientation !== b.orientation) {
      return false;
    }
    if (a.orientation === "H") {
      const aOnHorizontalFace = Math.abs(a.boundary.x - a.rect.left) < 1 || Math.abs(a.boundary.x - a.rect.right) < 1;
      return aOnHorizontalFace && sameX(a.boundary, b.boundary, 1);
    }
    const aOnVerticalFace = Math.abs(a.boundary.y - a.rect.top) < 1 || Math.abs(a.boundary.y - a.rect.bottom) < 1;
    return aOnVerticalFace && sameY(a.boundary, b.boundary, 1);
  }, "sameTerminalFace");
  const exactTerminalLaneConflict = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => {
    if (a.nodeId !== b.nodeId || a.orientation !== b.orientation) {
      return false;
    }
    const shared = projectedOverlapLength(a, b);
    return shared >= MIN_SHARED && Math.abs(a.coord - b.coord) < 0.5;
  }, "exactTerminalLaneConflict");
  const nearTerminalLaneConflict = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => {
    if (a.nodeId !== b.nodeId || a.orientation !== b.orientation || a.orientation !== "H" || a.atStart === b.atStart) {
      return false;
    }
    const shared = projectedOverlapLength(a, b);
    if (shared < MIN_SHARED) {
      return false;
    }
    const faceSpan = a.rect.bottom - a.rect.top;
    if (shared < faceSpan || shared > 2 * faceSpan) {
      return false;
    }
    return sameTerminalFace(a, b) && Math.abs(a.coord - b.coord) < MIN_FACE_CLEARANCE;
  }, "nearTerminalLaneConflict");
  const shiftedCandidate = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((lane, shift) => {
    const points = dedupeConsecutivePoints(lane.edge.points ?? []);
    if (points.length < 2) {
      return void 0;
    }
    const shiftedBoundary = lane.orientation === "V" ? { x: lane.boundary.x + shift, y: lane.boundary.y } : { x: lane.boundary.x, y: lane.boundary.y + shift };
    const shiftedRailEnd = lane.orientation === "V" ? { x: lane.railEnd.x + shift, y: lane.railEnd.y } : { x: lane.railEnd.x, y: lane.railEnd.y + shift };
    const boundaryStaysOnSameFace = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
      if (Math.abs(lane.boundary.y - lane.rect.top) < 1 || Math.abs(lane.boundary.y - lane.rect.bottom) < 1) {
        return sameY(shiftedBoundary, lane.boundary, EPS_LOCAL) && shiftedBoundary.x >= lane.rect.left + 1 && shiftedBoundary.x <= lane.rect.right - 1;
      }
      if (Math.abs(lane.boundary.x - lane.rect.left) < 1 || Math.abs(lane.boundary.x - lane.rect.right) < 1) {
        return sameX(shiftedBoundary, lane.boundary, EPS_LOCAL) && shiftedBoundary.y >= lane.rect.top + 1 && shiftedBoundary.y <= lane.rect.bottom - 1;
      }
      return false;
    }, "boundaryStaysOnSameFace");
    if (!boundaryStaysOnSameFace()) {
      return void 0;
    }
    if (lane.atStart) {
      const railEndIsAdjacent2 = points.length > 1 && samePoint(points[1], lane.railEnd, EPS_LOCAL);
      const rest = points.slice(railEndIsAdjacent2 ? 2 : 1);
      const next = rest[0];
      if (next && !orthogonallyAligned(next, shiftedRailEnd)) {
        return void 0;
      }
      return [shiftedBoundary, shiftedRailEnd, ...rest];
    }
    const railEndIsAdjacent = points.length > 1 && samePoint(points[points.length - 2], lane.railEnd, EPS_LOCAL);
    const before = points.slice(0, railEndIsAdjacent ? -2 : -1);
    const previous = before[before.length - 1];
    if (previous && !orthogonallyAligned(previous, shiftedRailEnd)) {
      return void 0;
    }
    return [...before, shiftedRailEnd, shiftedBoundary];
  }, "shiftedCandidate");
  const laneIsStraightCollinearConnector = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((lane) => {
    const edge = lane.edge;
    const points = dedupeConsecutivePoints(edge.points ?? []);
    if (points.length !== 2) {
      return false;
    }
    const startId = edge.start;
    const endId = edge.end;
    const start = startId ? nodeByIdMap.get(startId) : void 0;
    const end = endId ? nodeByIdMap.get(endId) : void 0;
    if (!start || !end) {
      return false;
    }
    const startX = start.x ?? 0;
    const startY = start.y ?? 0;
    const endX = end.x ?? 0;
    const endY = end.y ?? 0;
    const [a, b] = points;
    return sameY(a, b, EPS_LOCAL) && Math.abs(startY - endY) < 1 && Math.abs(startX - endX) > 1 || sameX(a, b, EPS_LOCAL) && Math.abs(startX - endX) < 1 && Math.abs(startY - endY) > 1;
  }, "laneIsStraightCollinearConnector");
  const shifts = [
    -TRACK_SHIFT,
    TRACK_SHIFT,
    -2 * TRACK_SHIFT,
    2 * TRACK_SHIFT,
    -3 * TRACK_SHIFT,
    3 * TRACK_SHIFT
  ];
  for (let iteration = 0; iteration < 8; iteration++) {
    const lanes = edges.filter((edge) => !edge.isLayoutOnly).flatMap((edge) => [terminalLaneFor(edge, true), terminalLaneFor(edge, false)]).filter((lane) => Boolean(lane));
    let fixed = false;
    for (let i = 0; i < lanes.length && !fixed; i++) {
      for (let j = i + 1; j < lanes.length && !fixed; j++) {
        const first = lanes[i];
        const second = lanes[j];
        if (first.edge === second.edge || !(exactTerminalLaneConflict(first, second) || nearTerminalLaneConflict(first, second))) {
          continue;
        }
        const fixingNearConflict = !exactTerminalLaneConflict(first, second);
        const candidates = [first, second].sort((a, b) => {
          const aPreservesStraight = laneIsStraightCollinearConnector(a);
          const bPreservesStraight = laneIsStraightCollinearConnector(b);
          if (aPreservesStraight !== bPreservesStraight) {
            return Number(aPreservesStraight) - Number(bPreservesStraight);
          }
          return Number(!b.atStart) - Number(!a.atStart);
        });
        for (const lane of candidates) {
          for (const shift of shifts) {
            const candidate = shiftedCandidate(lane, shift);
            if (!candidate) {
              continue;
            }
            const nextLane = terminalLaneFor({ ...lane.edge, points: candidate }, lane.atStart);
            if (!nextLane || lanes.some(
              (other) => other.edge !== lane.edge && (exactTerminalLaneConflict(nextLane, other) || fixingNearConflict && nearTerminalLaneConflict(nextLane, other))
            )) {
              continue;
            }
            lane.edge.points = candidate;
            fixed = true;
            break;
          }
          if (fixed) {
            break;
          }
        }
      }
    }
    if (!fixed) {
      return;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(separateSharedRenderedTerminalLanes, "separateSharedRenderedTerminalLanes");
function collapseRedundantRectangularDoglegs(edges, nodeByIdMap) {
  const BUFFER = 2;
  const MAX_ITERATIONS = 8;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, candidate) => {
    const sourceId = edge.start;
    const targetId = edge.end;
    const candidateSegments = segmentsFor(candidate);
    if (candidateSegments.length !== candidate.length - 1) {
      return false;
    }
    const endpointIds = [sourceId, targetId].filter((id) => Boolean(id));
    for (const segment of candidateSegments) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
        return false;
      }
      if (segmentHitsAnyRect(segment.a, segment.b, labelRects, [], -BUFFER)) {
        return false;
      }
    }
    for (const other of edges) {
      if (other === edge || other.isLayoutOnly) {
        continue;
      }
      const otherPoints = other.points;
      if (!otherPoints || otherPoints.length < 2) {
        continue;
      }
      for (const candidateSegment of candidateSegments) {
        for (const otherSegment of segmentsFor(dedupeConsecutivePoints(otherPoints))) {
          if (sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED) {
            return false;
          }
          if (orthogonalSegmentsStrictlyCross(
            candidateSegment.a,
            candidateSegment.b,
            otherSegment.a,
            otherSegment.b,
            EPS_LOCAL
          )) {
            return false;
          }
        }
      }
    }
    return true;
  }, "candidateIsSafe");
  const withoutDogleg = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points, i) => {
    if (i + 4 >= points.length) {
      return void 0;
    }
    const p0 = points[i];
    const p1 = points[i + 1];
    const p2 = points[i + 2];
    const p3 = points[i + 3];
    const p4 = points[i + 4];
    const terminalVerticalDogleg = isHorizontalSegment(p0, p1) && isVerticalSegment(p1, p2) && isHorizontalSegment(p2, p3) && isVerticalSegment(p3, p4) && sameX(p0, p3, EPS_LOCAL) && sameX(p0, p4, EPS_LOCAL) && sameX(p1, p2, EPS_LOCAL) && (p1.x - p0.x) * (p3.x - p2.x) < 0;
    const terminalHorizontalDogleg = isVerticalSegment(p0, p1) && isHorizontalSegment(p1, p2) && isVerticalSegment(p2, p3) && isHorizontalSegment(p3, p4) && sameY(p0, p3, EPS_LOCAL) && sameY(p0, p4, EPS_LOCAL) && sameY(p1, p2, EPS_LOCAL) && (p1.y - p0.y) * (p3.y - p2.y) < 0;
    if (terminalVerticalDogleg || terminalHorizontalDogleg) {
      return dedupeConsecutivePoints([...points.slice(0, i + 1), p4, ...points.slice(i + 5)]);
    }
    if (i + 5 >= points.length) {
      return void 0;
    }
    const p5 = points[i + 5];
    const verticalDogleg = isVerticalSegment(p0, p1) && isHorizontalSegment(p1, p2) && isVerticalSegment(p2, p3) && isHorizontalSegment(p3, p4) && isVerticalSegment(p4, p5) && sameX(p0, p4, EPS_LOCAL) && sameX(p0, p5, EPS_LOCAL) && sameX(p2, p3, EPS_LOCAL) && (p2.x - p1.x) * (p4.x - p3.x) < 0;
    const horizontalDogleg = isHorizontalSegment(p0, p1) && isVerticalSegment(p1, p2) && isHorizontalSegment(p2, p3) && isVerticalSegment(p3, p4) && isHorizontalSegment(p4, p5) && sameY(p0, p4, EPS_LOCAL) && sameY(p0, p5, EPS_LOCAL) && sameY(p2, p3, EPS_LOCAL) && (p2.y - p1.y) * (p4.y - p3.y) < 0;
    if (!verticalDogleg && !horizontalDogleg) {
      return void 0;
    }
    return dedupeConsecutivePoints([...points.slice(0, i + 1), p5, ...points.slice(i + 6)]);
  }, "withoutDogleg");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    let fixed = false;
    for (const edge of edges) {
      if (edge.isLayoutOnly) {
        continue;
      }
      const points = dedupeConsecutivePoints(edge.points ?? []);
      for (let i = 0; i <= points.length - 5; i++) {
        const candidate = withoutDogleg(points, i);
        if (!candidate || !candidateIsSafe(edge, candidate)) {
          continue;
        }
        edge.points = candidate;
        fixed = true;
        break;
      }
      if (fixed) {
        break;
      }
    }
    if (!fixed) {
      return;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(collapseRedundantRectangularDoglegs, "collapseRedundantRectangularDoglegs");
function liftObstacleHuggingSameSideRails(edges, nodeByIdMap) {
  const BUFFER = 2;
  const CLEARANCE = 20;
  const MAX_ITERATIONS = 8;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  const visibleEdges = edges.filter((edge) => !edge.isLayoutOnly);
  const pointsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, replacementEdge, replacement) => dedupeConsecutivePoints(
    edge === replacementEdge ? replacement ?? [] : edge.points ?? []
  ), "pointsFor");
  const strictCrossingCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacementEdge, replacement) => {
    let count = 0;
    for (let i = 0; i < visibleEdges.length; i++) {
      const firstSegments = segmentsFor(pointsFor(visibleEdges[i], replacementEdge, replacement));
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const secondSegments = segmentsFor(
          pointsFor(visibleEdges[j], replacementEdge, replacement)
        );
        for (const firstSegment of firstSegments) {
          for (const secondSegment of secondSegments) {
            if (orthogonalSegmentsStrictlyCross(
              firstSegment.a,
              firstSegment.b,
              secondSegment.a,
              secondSegment.b,
              EPS_LOCAL
            )) {
              count++;
            }
          }
        }
      }
    }
    return count;
  }, "strictCrossingCount");
  const middleRail = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points) => {
    const segments = segmentsFor(points);
    if (segments.length !== 3) {
      return void 0;
    }
    const middle = segments[1];
    if (segments[0].horizontal === middle.horizontal || segments[2].horizontal === middle.horizontal) {
      return void 0;
    }
    return {
      index: middle.index,
      horizontal: middle.horizontal,
      vertical: middle.vertical,
      segment: middle
    };
  }, "middleRail");
  const blockingRectsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, rail) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    return realNodeRects.filter((entry) => {
      if (endpointIds.includes(entry.id)) {
        return false;
      }
      const rect = entry.rect;
      if (rail.horizontal) {
        const xOverlap = overlapLength(rail.a.x, rail.b.x, rect.left, rect.right);
        return xOverlap >= MIN_SHARED && rail.a.y >= rect.top - BUFFER && rail.a.y <= rect.bottom + BUFFER;
      }
      const yOverlap = overlapLength(rail.a.y, rail.b.y, rect.top, rect.bottom);
      return yOverlap >= MIN_SHARED && rail.a.x >= rect.left - BUFFER && rail.a.x <= rect.right + BUFFER;
    });
  }, "blockingRectsFor");
  const candidateByMovingRail = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points, rail, coord) => {
    const candidate = points.map((point) => ({ ...point }));
    if (rail.horizontal) {
      candidate[rail.index].y = coord;
      candidate[rail.index + 1].y = coord;
    } else if (rail.vertical) {
      candidate[rail.index].x = coord;
      candidate[rail.index + 1].x = coord;
    } else {
      return void 0;
    }
    const simplified = simplifyPolyline(dedupeConsecutivePoints(candidate));
    return segmentsFor(simplified).length === simplified.length - 1 ? simplified : void 0;
  }, "candidateByMovingRail");
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, candidate, currentCrossings) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    const candidateSegments = segmentsFor(candidate);
    if (candidateSegments.length !== candidate.length - 1) {
      return false;
    }
    for (const segment of candidateSegments) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
        return false;
      }
      if (segmentHitsAnyRect(segment.a, segment.b, labelRects, [], -BUFFER)) {
        return false;
      }
    }
    for (const other of visibleEdges) {
      if (other === edge) {
        continue;
      }
      for (const candidateSegment of candidateSegments) {
        for (const otherSegment of segmentsFor(pointsFor(other))) {
          if (sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED) {
            return false;
          }
        }
      }
    }
    return strictCrossingCount(edge, candidate) <= currentCrossings;
  }, "candidateIsSafe");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const currentCrossings = strictCrossingCount();
    let fixed = false;
    for (const edge of visibleEdges) {
      const points = pointsFor(edge);
      const rail = middleRail(points);
      if (!rail) {
        continue;
      }
      const blockers = blockingRectsFor(edge, rail.segment);
      if (blockers.length === 0) {
        continue;
      }
      const coords = rail.horizontal ? [
        Math.min(...blockers.map((entry) => entry.rect.top)) - CLEARANCE,
        Math.max(...blockers.map((entry) => entry.rect.bottom)) + CLEARANCE
      ] : [
        Math.min(...blockers.map((entry) => entry.rect.left)) - CLEARANCE,
        Math.max(...blockers.map((entry) => entry.rect.right)) + CLEARANCE
      ];
      for (const coord of coords) {
        const candidate = candidateByMovingRail(points, rail.segment, coord);
        if (!candidate || !candidateIsSafe(edge, candidate, currentCrossings)) {
          continue;
        }
        edge.points = candidate;
        fixed = true;
        break;
      }
      if (fixed) {
        break;
      }
    }
    if (!fixed) {
      return;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(liftObstacleHuggingSameSideRails, "liftObstacleHuggingSameSideRails");
function liftTopLaneTitleBandsAboveRails(edges, nodeByIdMap) {
  const CLEARANCE = 4;
  const validTitleRect = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    const rect = node.groupTitleRect;
    if (!rect || typeof rect.left !== "number" || typeof rect.right !== "number" || typeof rect.top !== "number" || typeof rect.bottom !== "number" || !Number.isFinite(rect.left) || !Number.isFinite(rect.right) || !Number.isFinite(rect.top) || !Number.isFinite(rect.bottom) || rect.right <= rect.left || rect.bottom <= rect.top) {
      return void 0;
    }
    return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
  }, "validTitleRect");
  const topLaneTitleFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    if (!node.isGroup || node.parentId) {
      return void 0;
    }
    const rawDirection = node.direction;
    const direction = typeof rawDirection === "string" ? rawDirection.toUpperCase() : "";
    if (direction === "LR" || direction === "RL" || direction === "BT") {
      return void 0;
    }
    const rect = validTitleRect(node);
    const y = node.y;
    const height = node.height;
    if (!rect || typeof y !== "number" || typeof height !== "number" || !Number.isFinite(y) || !Number.isFinite(height) || height <= 0) {
      return void 0;
    }
    const titleWidth = rect.right - rect.left;
    const titleHeight = rect.bottom - rect.top;
    if (titleHeight <= 0 || titleWidth < titleHeight) {
      return void 0;
    }
    return { node, rect };
  }, "topLaneTitleFor");
  const horizontalSegmentIntersectsTitle = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, rect) => {
    if (!segment.horizontal) {
      return false;
    }
    const y = segment.a.y;
    if (y <= rect.top + EPS_LOCAL || y >= rect.bottom - EPS_LOCAL) {
      return false;
    }
    return overlapLength(segment.a.x, segment.b.x, rect.left, rect.right) >= MIN_SHARED;
  }, "horizontalSegmentIntersectsTitle");
  const lanes = [...nodeByIdMap.values()].map(topLaneTitleFor).filter((lane) => Boolean(lane));
  if (lanes.length === 0) {
    return;
  }
  let topDelta = 0;
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const points = dedupeConsecutivePoints(edge.points ?? []);
    for (const segment of segmentsFor(points)) {
      for (const lane of lanes) {
        if (!horizontalSegmentIntersectsTitle(segment, lane.rect)) {
          continue;
        }
        topDelta = Math.max(topDelta, lane.rect.bottom - segment.a.y + CLEARANCE);
      }
    }
  }
  if (topDelta <= EPS_LOCAL) {
    return;
  }
  for (const lane of lanes) {
    const y = lane.node.y;
    const height = lane.node.height;
    if (typeof y !== "number" || typeof height !== "number" || !Number.isFinite(y) || !Number.isFinite(height) || height <= 0) {
      continue;
    }
    lane.node.y = y - topDelta / 2;
    lane.node.height = height + topDelta;
    lane.node.groupTitleRect = {
      ...lane.rect,
      top: lane.rect.top - topDelta,
      bottom: lane.rect.bottom - topDelta
    };
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(liftTopLaneTitleBandsAboveRails, "liftTopLaneTitleBandsAboveRails");
function shiftLeftLaneTitleBandsLeftOfRails(edges, nodeByIdMap) {
  const CLEARANCE = 4;
  const validTitleRect = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    const rect = node.groupTitleRect;
    if (!rect || typeof rect.left !== "number" || typeof rect.right !== "number" || typeof rect.top !== "number" || typeof rect.bottom !== "number" || !Number.isFinite(rect.left) || !Number.isFinite(rect.right) || !Number.isFinite(rect.top) || !Number.isFinite(rect.bottom) || rect.right <= rect.left || rect.bottom <= rect.top) {
      return void 0;
    }
    return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
  }, "validTitleRect");
  const leftLaneTitleFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    if (!node.isGroup || node.parentId) {
      return void 0;
    }
    const rawDirection = node.direction;
    if (rawDirection !== "LR") {
      return void 0;
    }
    const rect = validTitleRect(node);
    const x = node.x;
    const width = node.width;
    if (!rect || typeof x !== "number" || typeof width !== "number" || !Number.isFinite(x) || !Number.isFinite(width) || width <= 0) {
      return void 0;
    }
    const titleWidth = rect.right - rect.left;
    const titleHeight = rect.bottom - rect.top;
    if (titleWidth <= 0 || titleHeight < titleWidth) {
      return void 0;
    }
    return { node, rect };
  }, "leftLaneTitleFor");
  const verticalSegmentIntersectsTitle = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, rect) => {
    if (!segment.vertical) {
      return false;
    }
    const x = segment.a.x;
    if (x <= rect.left + EPS_LOCAL || x >= rect.right - EPS_LOCAL) {
      return false;
    }
    return overlapLength(segment.a.y, segment.b.y, rect.top, rect.bottom) >= MIN_SHARED;
  }, "verticalSegmentIntersectsTitle");
  const horizontalSegmentIntersectsTitle = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, rect) => {
    if (!segment.horizontal) {
      return false;
    }
    const y = segment.a.y;
    if (y <= rect.top + EPS_LOCAL || y >= rect.bottom - EPS_LOCAL) {
      return false;
    }
    return overlapLength(segment.a.x, segment.b.x, rect.left, rect.right) >= MIN_SHARED;
  }, "horizontalSegmentIntersectsTitle");
  const lanes = [...nodeByIdMap.values()].map(leftLaneTitleFor).filter((lane) => Boolean(lane));
  if (lanes.length === 0) {
    return;
  }
  let leftDelta = 0;
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const points = dedupeConsecutivePoints(edge.points ?? []);
    for (const segment of segmentsFor(points)) {
      for (const lane of lanes) {
        if (verticalSegmentIntersectsTitle(segment, lane.rect)) {
          leftDelta = Math.max(leftDelta, lane.rect.right - segment.a.x + CLEARANCE);
        } else if (horizontalSegmentIntersectsTitle(segment, lane.rect)) {
          const segmentLeft = Math.min(segment.a.x, segment.b.x);
          leftDelta = Math.max(leftDelta, lane.rect.right - segmentLeft + CLEARANCE);
        }
      }
    }
  }
  if (leftDelta <= EPS_LOCAL) {
    return;
  }
  for (const lane of lanes) {
    const x = lane.node.x;
    const width = lane.node.width;
    if (typeof x !== "number" || typeof width !== "number" || !Number.isFinite(x) || !Number.isFinite(width) || width <= 0) {
      continue;
    }
    lane.node.x = x - leftDelta / 2;
    lane.node.width = width + leftDelta;
    lane.node.groupTitleRect = {
      ...lane.rect,
      left: lane.rect.left - leftDelta,
      right: lane.rect.right - leftDelta
    };
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(shiftLeftLaneTitleBandsLeftOfRails, "shiftLeftLaneTitleBandsLeftOfRails");
function swapDestinationTerminalTailsToReduceCrossings(edges, nodeByIdMap) {
  const BUFFER = 2;
  const MAX_ITERATIONS = 4;
  const { realNodeRects } = collectNodeRectEntries(nodeByIdMap.values());
  const visibleEdges = edges.filter((edge) => !edge.isLayoutOnly);
  const replacementPointsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, replacements = /* @__PURE__ */ new Map()) => dedupeConsecutivePoints(
    replacements.get(edge) ?? edge.points ?? []
  ), "replacementPointsFor");
  const crossingCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => {
    let count = 0;
    for (let i = 0; i < visibleEdges.length; i++) {
      const firstSegments = segmentsFor(replacementPointsFor(visibleEdges[i], replacements));
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const secondSegments = segmentsFor(replacementPointsFor(visibleEdges[j], replacements));
        for (const firstSegment of firstSegments) {
          for (const secondSegment of secondSegments) {
            if (orthogonalSegmentsStrictlyCross(
              firstSegment.a,
              firstSegment.b,
              secondSegment.a,
              secondSegment.b,
              EPS_LOCAL
            )) {
              count++;
            }
          }
        }
      }
    }
    return count;
  }, "crossingCount");
  const totalBends = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => visibleEdges.reduce(
    (sum, edge) => sum + countOrthogonalBends(replacementPointsFor(edge, replacements)),
    0
  ), "totalBends");
  const terminalTailFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
    const points = replacementPointsFor(edge);
    if (points.length < 4) {
      return void 0;
    }
    const tailStart = points[points.length - 2];
    const terminal = points[points.length - 1];
    if (!isHorizontalSegment(tailStart, terminal, EPS_LOCAL) && !isVerticalSegment(tailStart, terminal, EPS_LOCAL)) {
      return void 0;
    }
    return { tailStart, terminal };
  }, "terminalTailFor");
  const candidateWithDestinationTail = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, tail) => {
    const points = replacementPointsFor(edge);
    if (points.length < 3) {
      return void 0;
    }
    const start = points[0];
    const firstTurn = points[1];
    let connector;
    if (isHorizontalSegment(start, firstTurn, EPS_LOCAL)) {
      connector = { x: firstTurn.x, y: tail.tailStart.y };
    } else if (isVerticalSegment(start, firstTurn, EPS_LOCAL)) {
      connector = { x: tail.tailStart.x, y: firstTurn.y };
    } else {
      return void 0;
    }
    const candidate = simplifyPolyline(
      dedupeConsecutivePoints([start, firstTurn, connector, tail.tailStart, tail.terminal])
    );
    return segmentsFor(candidate).length === candidate.length - 1 ? candidate : void 0;
  }, "candidateWithDestinationTail");
  const pathHasNodeHit = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, path) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    for (const segment of segmentsFor(path)) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
        return true;
      }
    }
    return false;
  }, "pathHasNodeHit");
  const pathHasSharedTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, path, replacements) => {
    for (const other of visibleEdges) {
      if (other === edge) {
        continue;
      }
      for (const candidateSegment of segmentsFor(path)) {
        for (const otherSegment of segmentsFor(replacementPointsFor(other, replacements))) {
          if (sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED) {
            return true;
          }
        }
      }
    }
    return false;
  }, "pathHasSharedTrack");
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, path, replacements) => !pathHasNodeHit(edge, path) && !pathHasSharedTrack(edge, path, replacements), "candidateIsSafe");
  const edgesByDestination = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    const result = /* @__PURE__ */ new Map();
    for (const edge of visibleEdges) {
      const dstId = edge.end;
      if (!dstId || !nodeByIdMap.has(dstId)) {
        continue;
      }
      const points = replacementPointsFor(edge);
      if (points.length < 4) {
        continue;
      }
      const bucket = result.get(dstId) ?? [];
      bucket.push(edge);
      result.set(dstId, bucket);
    }
    return result;
  }, "edgesByDestination");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const currentCrossings = crossingCount();
    if (currentCrossings === 0) {
      return;
    }
    const currentBends = totalBends();
    let bestReplacements;
    let bestCrossings = currentCrossings;
    let bestBends = currentBends;
    for (const destinationEdges of edgesByDestination().values()) {
      for (let i = 0; i < destinationEdges.length; i++) {
        for (let j = i + 1; j < destinationEdges.length; j++) {
          const first = destinationEdges[i];
          const second = destinationEdges[j];
          const firstTail = terminalTailFor(first);
          const secondTail = terminalTailFor(second);
          if (!firstTail || !secondTail) {
            continue;
          }
          const firstCandidate = candidateWithDestinationTail(first, secondTail);
          const secondCandidate = candidateWithDestinationTail(second, firstTail);
          if (!firstCandidate || !secondCandidate) {
            continue;
          }
          const replacements = /* @__PURE__ */ new Map([
            [first, firstCandidate],
            [second, secondCandidate]
          ]);
          if (!candidateIsSafe(first, firstCandidate, replacements) || !candidateIsSafe(second, secondCandidate, replacements)) {
            continue;
          }
          const candidateCrossings = crossingCount(replacements);
          const candidateBends = totalBends(replacements);
          if (candidateCrossings >= currentCrossings) {
            continue;
          }
          if (candidateCrossings > bestCrossings || candidateCrossings === bestCrossings && candidateBends >= bestBends) {
            continue;
          }
          bestReplacements = replacements;
          bestCrossings = candidateCrossings;
          bestBends = candidateBends;
        }
      }
    }
    if (!bestReplacements) {
      return;
    }
    for (const [edge, points] of bestReplacements) {
      edge.points = points;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(swapDestinationTerminalTailsToReduceCrossings, "swapDestinationTerminalTailsToReduceCrossings");
function reassignCrossingExternalRailChannels(edges, nodeByIdMap) {
  const BUFFER = 2;
  const RAIL_CHANNEL_GAP = 12;
  const MAX_ITERATIONS = 4;
  const MAX_EXHAUSTIVE_COMPONENT = 6;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  const visibleEdges = edges.filter((edge) => !edge.isLayoutOnly);
  const replacementPointsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, replacements = /* @__PURE__ */ new Map()) => dedupeConsecutivePoints(
    replacements.get(edge) ?? edge.points ?? []
  ), "replacementPointsFor");
  const strictCrossingCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => {
    let count = 0;
    for (let i = 0; i < visibleEdges.length; i++) {
      const firstSegments = segmentsFor(replacementPointsFor(visibleEdges[i], replacements));
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const secondSegments = segmentsFor(replacementPointsFor(visibleEdges[j], replacements));
        for (const firstSegment of firstSegments) {
          for (const secondSegment of secondSegments) {
            if (orthogonalSegmentsStrictlyCross(
              firstSegment.a,
              firstSegment.b,
              secondSegment.a,
              secondSegment.b,
              EPS_LOCAL
            )) {
              count++;
            }
          }
        }
      }
    }
    return count;
  }, "strictCrossingCount");
  const totalBends = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => visibleEdges.reduce(
    (sum, edge) => sum + countOrthogonalBends(replacementPointsFor(edge, replacements)),
    0
  ), "totalBends");
  const endpointRectsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
    const srcId = edge.start;
    const dstId = edge.end;
    const srcNode = srcId ? nodeByIdMap.get(srcId) : void 0;
    const dstNode = dstId ? nodeByIdMap.get(dstId) : void 0;
    const src = srcNode ? rectOfNodeBounds(srcNode) : void 0;
    const dst = dstNode ? rectOfNodeBounds(dstNode) : void 0;
    return src && dst ? { src, dst } : void 0;
  }, "endpointRectsFor");
  const externalRailForSegment = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, points, segment) => {
    if (segment.index <= 0 || segment.index + 1 >= points.length - 1) {
      return void 0;
    }
    const endpointRects = endpointRectsFor(edge);
    if (!endpointRects) {
      return void 0;
    }
    if (segment.vertical) {
      const coord = segment.a.x;
      const leftBound = Math.min(endpointRects.src.left, endpointRects.dst.left);
      const rightBound = Math.max(endpointRects.src.right, endpointRects.dst.right);
      const side = coord < leftBound - EPS_LOCAL ? "left" : coord > rightBound + EPS_LOCAL ? "right" : void 0;
      if (!side) {
        return void 0;
      }
      return {
        edge,
        points,
        segmentIndex: segment.index,
        axis: "vertical",
        side,
        coord,
        min: Math.min(segment.a.y, segment.b.y),
        max: Math.max(segment.a.y, segment.b.y)
      };
    }
    if (segment.horizontal) {
      const coord = segment.a.y;
      const topBound = Math.min(endpointRects.src.top, endpointRects.dst.top);
      const bottomBound = Math.max(endpointRects.src.bottom, endpointRects.dst.bottom);
      const side = coord < topBound - EPS_LOCAL ? "top" : coord > bottomBound + EPS_LOCAL ? "bottom" : void 0;
      if (!side) {
        return void 0;
      }
      return {
        edge,
        points,
        segmentIndex: segment.index,
        axis: "horizontal",
        side,
        coord,
        min: Math.min(segment.a.x, segment.b.x),
        max: Math.max(segment.a.x, segment.b.x)
      };
    }
    return void 0;
  }, "externalRailForSegment");
  const collectExternalRails = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    const rails = [];
    for (const edge of visibleEdges) {
      const points = replacementPointsFor(edge);
      for (const segment of segmentsFor(points)) {
        const rail = externalRailForSegment(edge, points, segment);
        if (rail) {
          rails.push(rail);
        }
      }
    }
    return rails;
  }, "collectExternalRails");
  const railsInteract = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => a.edge !== b.edge && a.axis === b.axis && a.side === b.side && overlapLength(a.min, a.max, b.min, b.max) >= MIN_SHARED, "railsInteract");
  const connectedComponents = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((rails) => {
    const result = [];
    const seen = /* @__PURE__ */ new Set();
    for (const rail of rails) {
      if (seen.has(rail)) {
        continue;
      }
      const queue = [rail];
      const component = [];
      seen.add(rail);
      while (queue.length > 0) {
        const current = queue.pop();
        component.push(current);
        for (const next of rails) {
          if (!seen.has(next) && railsInteract(current, next)) {
            seen.add(next);
            queue.push(next);
          }
        }
      }
      if (component.length > 1) {
        result.push(component);
      }
    }
    return result;
  }, "connectedComponents");
  const uniqueCoordsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((component) => {
    const coords = [];
    for (const rail of component) {
      if (!coords.some((coord) => Math.abs(coord - rail.coord) < EPS_LOCAL)) {
        coords.push(rail.coord);
      }
    }
    while (coords.length < component.length) {
      const min = Math.min(...coords);
      const max = Math.max(...coords);
      const side = component[0].side;
      coords.push(
        side === "left" || side === "top" ? min - RAIL_CHANNEL_GAP * (component.length - coords.length) : max + RAIL_CHANNEL_GAP * (component.length - coords.length)
      );
    }
    return coords;
  }, "uniqueCoordsFor");
  const coordinateAssignmentsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((component) => {
    const current = component.map((rail) => rail.coord);
    const coords = uniqueCoordsFor(component);
    const assignments = [];
    if (component.length <= MAX_EXHAUSTIVE_COMPONENT) {
      const used = new Array(coords.length).fill(false);
      const next = [];
      const visit = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
        if (next.length === component.length) {
          if (next.some((coord, index) => Math.abs(coord - current[index]) >= EPS_LOCAL)) {
            assignments.push([...next]);
          }
          return;
        }
        for (const [i, coord] of coords.entries()) {
          if (used[i]) {
            continue;
          }
          used[i] = true;
          next.push(coord);
          visit();
          next.pop();
          used[i] = false;
        }
      }, "visit");
      visit();
      return assignments;
    }
    for (let i = 0; i < current.length; i++) {
      for (let j = i + 1; j < current.length; j++) {
        const assignment = [...current];
        [assignment[i], assignment[j]] = [assignment[j], assignment[i]];
        assignments.push(assignment);
      }
    }
    return assignments;
  }, "coordinateAssignmentsFor");
  const replacementsForAssignment = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((component, assignment) => {
    const draftByEdge = /* @__PURE__ */ new Map();
    for (const [i, rail] of component.entries()) {
      const coord = assignment[i];
      const points = draftByEdge.get(rail.edge) ?? rail.points.map((point) => ({ x: point.x, y: point.y }));
      if (rail.axis === "vertical") {
        points[rail.segmentIndex].x = coord;
        points[rail.segmentIndex + 1].x = coord;
      } else {
        points[rail.segmentIndex].y = coord;
        points[rail.segmentIndex + 1].y = coord;
      }
      draftByEdge.set(rail.edge, points);
    }
    const replacements = /* @__PURE__ */ new Map();
    for (const [edge, points] of draftByEdge) {
      const simplified = simplifyPolyline(dedupeConsecutivePoints(points));
      if (segmentsFor(simplified).length !== simplified.length - 1) {
        return void 0;
      }
      replacements.set(edge, simplified);
    }
    return replacements;
  }, "replacementsForAssignment");
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements) => {
    for (const [edge, points] of replacements) {
      const endpointIds = [
        edge.start,
        edge.end
      ].filter((id) => Boolean(id));
      for (const segment of segmentsFor(points)) {
        if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
          return false;
        }
        if (segmentHitsAnyRect(segment.a, segment.b, labelRects, [], -BUFFER)) {
          return false;
        }
      }
    }
    for (let i = 0; i < visibleEdges.length; i++) {
      const first = visibleEdges[i];
      const firstChanged = replacements.has(first);
      const firstSegments = segmentsFor(replacementPointsFor(first, replacements));
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const second = visibleEdges[j];
        if (!firstChanged && !replacements.has(second)) {
          continue;
        }
        const secondSegments = segmentsFor(replacementPointsFor(second, replacements));
        for (const firstSegment of firstSegments) {
          for (const secondSegment of secondSegments) {
            if (sameAxisSegmentOverlapLength(firstSegment, secondSegment, 0.5) >= MIN_SHARED) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }, "candidateIsSafe");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const currentCrossings = strictCrossingCount();
    if (currentCrossings === 0) {
      return;
    }
    let bestReplacements;
    let bestCrossings = currentCrossings;
    let bestBends = totalBends();
    let bestDisplacement = Number.POSITIVE_INFINITY;
    for (const component of connectedComponents(collectExternalRails())) {
      for (const assignment of coordinateAssignmentsFor(component)) {
        const replacements = replacementsForAssignment(component, assignment);
        if (!replacements || !candidateIsSafe(replacements)) {
          continue;
        }
        const candidateCrossings = strictCrossingCount(replacements);
        if (candidateCrossings >= currentCrossings) {
          continue;
        }
        const candidateBends = totalBends(replacements);
        const candidateDisplacement = component.reduce(
          (sum, rail, index) => sum + Math.abs(assignment[index] - rail.coord),
          0
        );
        if (candidateCrossings > bestCrossings || candidateCrossings === bestCrossings && (candidateBends > bestBends || candidateBends === bestBends && candidateDisplacement >= bestDisplacement)) {
          continue;
        }
        bestReplacements = replacements;
        bestCrossings = candidateCrossings;
        bestBends = candidateBends;
        bestDisplacement = candidateDisplacement;
      }
    }
    if (!bestReplacements) {
      return;
    }
    for (const [edge, points] of bestReplacements) {
      edge.points = points;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(reassignCrossingExternalRailChannels, "reassignCrossingExternalRailChannels");
function shortcutRedundantOrthogonalJogs(edges, nodeByIdMap) {
  const BUFFER = 2;
  const MAX_ITERATIONS = 8;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  const visibleEdges = edges.filter((edge) => !edge.isLayoutOnly);
  const pointsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, replacementEdge, replacement) => dedupeConsecutivePoints(
    edge === replacementEdge ? replacement ?? [] : edge.points ?? []
  ), "pointsFor");
  const pathLength = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points) => segmentsFor(points).reduce((sum, segment) => {
    const dx = segment.a.x - segment.b.x;
    const dy = segment.a.y - segment.b.y;
    return sum + Math.hypot(dx, dy);
  }, 0), "pathLength");
  const strictCrossingCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacementEdge, replacement) => {
    let count = 0;
    for (let i = 0; i < visibleEdges.length; i++) {
      const firstSegments = segmentsFor(pointsFor(visibleEdges[i], replacementEdge, replacement));
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const secondSegments = segmentsFor(
          pointsFor(visibleEdges[j], replacementEdge, replacement)
        );
        for (const firstSegment of firstSegments) {
          for (const secondSegment of secondSegments) {
            if (orthogonalSegmentsStrictlyCross(
              firstSegment.a,
              firstSegment.b,
              secondSegment.a,
              secondSegment.b,
              EPS_LOCAL
            )) {
              count++;
            }
          }
        }
      }
    }
    return count;
  }, "strictCrossingCount");
  const segmentRunsAlongRectBorder = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, rect) => {
    if (segment.horizontal) {
      const y = segment.a.y;
      const onBorder = Math.abs(y - rect.top) < 1 || Math.abs(y - rect.bottom) < 1;
      return onBorder && overlapLength(segment.a.x, segment.b.x, rect.left, rect.right) >= MIN_SHARED;
    }
    if (segment.vertical) {
      const x = segment.a.x;
      const onBorder = Math.abs(x - rect.left) < 1 || Math.abs(x - rect.right) < 1;
      return onBorder && overlapLength(segment.a.y, segment.b.y, rect.top, rect.bottom) >= MIN_SHARED;
    }
    return false;
  }, "segmentRunsAlongRectBorder");
  const endpointRectsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    const rects = [];
    for (const id of endpointIds) {
      const node = nodeByIdMap.get(id);
      const rect = node ? rectOfNodeBounds(node) : void 0;
      if (rect) {
        rects.push(rect);
      }
    }
    return rects;
  }, "endpointRectsFor");
  const shortcutCandidatesAt = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points, index) => {
    if (index + 3 >= points.length) {
      return [];
    }
    const p0 = points[index];
    const p1 = points[index + 1];
    const p2 = points[index + 2];
    const p3 = points[index + 3];
    const isHVH = isHorizontalSegment(p0, p1, EPS_LOCAL) && isVerticalSegment(p1, p2, EPS_LOCAL) && isHorizontalSegment(p2, p3, EPS_LOCAL);
    const isVHV = isVerticalSegment(p0, p1, EPS_LOCAL) && isHorizontalSegment(p1, p2, EPS_LOCAL) && isVerticalSegment(p2, p3, EPS_LOCAL);
    if (!isHVH && !isVHV) {
      return [];
    }
    const outerSegmentsOppose = isHVH ? Math.sign(p1.x - p0.x) !== Math.sign(p3.x - p2.x) : Math.sign(p1.y - p0.y) !== Math.sign(p3.y - p2.y);
    if (!outerSegmentsOppose) {
      return [];
    }
    const corners = sameX(p0, p3, EPS_LOCAL) || sameY(p0, p3, EPS_LOCAL) ? [] : [
      { x: p0.x, y: p3.y },
      { x: p3.x, y: p0.y }
    ];
    const rawCandidates = corners.length === 0 ? [[...points.slice(0, index + 1), ...points.slice(index + 3)]] : corners.map((corner) => [
      ...points.slice(0, index + 1),
      corner,
      ...points.slice(index + 3)
    ]);
    const seen = /* @__PURE__ */ new Set();
    return rawCandidates.map((candidate) => simplifyPolyline(dedupeConsecutivePoints(candidate))).filter((candidate) => {
      if (segmentsFor(candidate).length !== candidate.length - 1) {
        return false;
      }
      if (!candidate.some((point) => samePoint(point, p3, EPS_LOCAL))) {
        return false;
      }
      const key = candidate.map((point) => `${point.x.toFixed(3)},${point.y.toFixed(3)}`).join("|");
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, "shortcutCandidatesAt");
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, candidate, currentCrossings) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    const endpointRects = endpointRectsFor(edge);
    for (const segment of segmentsFor(candidate)) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
        return false;
      }
      if (segmentHitsAnyRect(segment.a, segment.b, labelRects, [], -BUFFER)) {
        return false;
      }
      if (endpointRects.some((rect) => segmentRunsAlongRectBorder(segment, rect))) {
        return false;
      }
    }
    for (const other of visibleEdges) {
      if (other === edge) {
        continue;
      }
      for (const candidateSegment of segmentsFor(candidate)) {
        for (const otherSegment of segmentsFor(pointsFor(other))) {
          if (sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED) {
            return false;
          }
        }
      }
    }
    return strictCrossingCount(edge, candidate) <= currentCrossings;
  }, "candidateIsSafe");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const currentCrossings = strictCrossingCount();
    let bestEdge;
    let bestPath;
    let bestCrossings = currentCrossings;
    let bestBends = Number.POSITIVE_INFINITY;
    let bestLength = Number.POSITIVE_INFINITY;
    for (const edge of visibleEdges) {
      const currentPoints = pointsFor(edge);
      const currentBends = countOrthogonalBends(currentPoints, EPS_LOCAL);
      const currentLength = pathLength(currentPoints);
      for (let index = 0; index <= currentPoints.length - 4; index++) {
        for (const candidate of shortcutCandidatesAt(currentPoints, index)) {
          const candidateBends = countOrthogonalBends(candidate, EPS_LOCAL);
          const candidateLength = pathLength(candidate);
          const improvesShape = candidateBends < currentBends || candidateBends === currentBends && candidateLength < currentLength - EPS_LOCAL;
          if (!improvesShape || !candidateIsSafe(edge, candidate, currentCrossings)) {
            continue;
          }
          const candidateCrossings = strictCrossingCount(edge, candidate);
          if (candidateCrossings > bestCrossings || candidateCrossings === bestCrossings && (candidateBends > bestBends || candidateBends === bestBends && candidateLength >= bestLength)) {
            continue;
          }
          bestEdge = edge;
          bestPath = candidate;
          bestCrossings = candidateCrossings;
          bestBends = candidateBends;
          bestLength = candidateLength;
        }
      }
    }
    if (!bestEdge || !bestPath) {
      return;
    }
    bestEdge.points = bestPath;
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(shortcutRedundantOrthogonalJogs, "shortcutRedundantOrthogonalJogs");
function resolveRenderedOrthogonalCrossings(edges, nodeByIdMap) {
  const ANCHOR = 20;
  const EXTRA_CHANNEL_COUNT = 2;
  const MAX_ITERATIONS = 4;
  const MAX_PAIR_CANDIDATES_PER_EDGE = 48;
  const realNodes = [];
  for (const node of nodeByIdMap.values()) {
    if (node.isGroup || node.isEdgeLabel) {
      continue;
    }
    const cx = node.x ?? 0;
    const cy = node.y ?? 0;
    const rect = rectOfNodeBounds(node);
    if (!rect) {
      continue;
    }
    realNodes.push({
      id: String(node.id ?? ""),
      cx,
      cy,
      rect
    });
  }
  if (realNodes.length === 0) {
    return;
  }
  const nodeInfoById = new Map(realNodes.map((node) => [node.id, node]));
  const realNodeRects = realNodes.map((node) => ({ id: node.id, rect: node.rect }));
  const sides = ["top", "bottom", "left", "right"];
  const outsideTracks = {
    top: Math.min(...realNodes.map((node) => node.rect.top)) - ANCHOR,
    bottom: Math.max(...realNodes.map((node) => node.rect.bottom)) + ANCHOR,
    left: Math.min(...realNodes.map((node) => node.rect.left)) - ANCHOR,
    right: Math.max(...realNodes.map((node) => node.rect.right)) + ANCHOR
  };
  const visibleEdges = edges.filter((edge) => !edge.isLayoutOnly);
  const edgeIndex = new Map(visibleEdges.map((edge, index) => [edge, index]));
  const outwardTracksForSide = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((side) => {
    const outward = side === "left" || side === "top" ? -1 : 1;
    const tracks = [];
    for (let channel = 0; channel <= EXTRA_CHANNEL_COUNT; channel++) {
      tracks.push(outsideTracks[side] + outward * ANCHOR * channel);
    }
    return tracks;
  }, "outwardTracksForSide");
  const replacementPointsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, replacements = /* @__PURE__ */ new Map()) => dedupeConsecutivePoints(
    replacements.get(edge) ?? edge.points ?? []
  ), "replacementPointsFor");
  const crossingCountBetweenSegments = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((firstSegments, secondSegments) => {
    let count = 0;
    for (const firstSegment of firstSegments) {
      for (const secondSegment of secondSegments) {
        if (orthogonalSegmentsStrictlyCross(
          firstSegment.a,
          firstSegment.b,
          secondSegment.a,
          secondSegment.b,
          EPS_LOCAL
        )) {
          count++;
        }
      }
    }
    return count;
  }, "crossingCountBetweenSegments");
  const crossingCountBetweenPaths = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((first, second) => crossingCountBetweenSegments(segmentsFor(first), segmentsFor(second)), "crossingCountBetweenPaths");
  const crossingSnapshot = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => {
    let count = 0;
    const pairs = [];
    const edgeSet = /* @__PURE__ */ new Set();
    const edgeOrder = [];
    const addEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
      if (!edgeSet.has(edge)) {
        edgeSet.add(edge);
        edgeOrder.push(edge);
      }
    }, "addEdge");
    for (let i = 0; i < visibleEdges.length; i++) {
      const first = visibleEdges[i];
      const firstPoints = replacementPointsFor(first, replacements);
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const second = visibleEdges[j];
        const pairCount = crossingCountBetweenPaths(
          firstPoints,
          replacementPointsFor(second, replacements)
        );
        if (pairCount > 0) {
          count += pairCount;
          pairs.push({ first, second, count: pairCount });
          addEdge(first);
          addEdge(second);
        }
      }
    }
    edgeOrder.sort((a, b) => (edgeIndex.get(a) ?? 0) - (edgeIndex.get(b) ?? 0));
    return {
      count,
      pairs,
      edgeSet,
      edges: edgeOrder
    };
  }, "crossingSnapshot");
  const crossingCountWithReplacements = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((current, replacements) => {
    const changed = new Set(replacements.keys());
    if (changed.size === 0) {
      return current.count;
    }
    let currentAffected = 0;
    for (const pair of current.pairs) {
      if (changed.has(pair.first) || changed.has(pair.second)) {
        currentAffected += pair.count;
      }
    }
    let replacementAffected = 0;
    for (let i = 0; i < visibleEdges.length; i++) {
      const first = visibleEdges[i];
      const firstChanged = changed.has(first);
      const firstPoints = replacementPointsFor(first, replacements);
      for (let j = i + 1; j < visibleEdges.length; j++) {
        const second = visibleEdges[j];
        if (!firstChanged && !changed.has(second)) {
          continue;
        }
        replacementAffected += crossingCountBetweenPaths(
          firstPoints,
          replacementPointsFor(second, replacements)
        );
      }
    }
    return current.count - currentAffected + replacementAffected;
  }, "crossingCountWithReplacements");
  const crossingComponents = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((snapshot) => {
    const neighbors = /* @__PURE__ */ new Map();
    for (const pair of snapshot.pairs) {
      const firstNeighbors = neighbors.get(pair.first) ?? /* @__PURE__ */ new Set();
      firstNeighbors.add(pair.second);
      neighbors.set(pair.first, firstNeighbors);
      const secondNeighbors = neighbors.get(pair.second) ?? /* @__PURE__ */ new Set();
      secondNeighbors.add(pair.first);
      neighbors.set(pair.second, secondNeighbors);
    }
    const components = [];
    const seen = /* @__PURE__ */ new Set();
    for (const edge of snapshot.edges) {
      if (seen.has(edge)) {
        continue;
      }
      const queue = [edge];
      const component = [];
      seen.add(edge);
      while (queue.length > 0) {
        const current = queue.pop();
        component.push(current);
        for (const next of neighbors.get(current) ?? []) {
          if (!seen.has(next)) {
            seen.add(next);
            queue.push(next);
          }
        }
      }
      component.sort((a, b) => (edgeIndex.get(a) ?? 0) - (edgeIndex.get(b) ?? 0));
      if (component.length > 1) {
        components.push(component);
      }
    }
    return components;
  }, "crossingComponents");
  const endpointIdsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => [edge.start, edge.end].filter(
    (id) => Boolean(id)
  ), "endpointIdsFor");
  const pairSearchGroups = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((snapshot) => {
    const groups = [];
    for (const component of crossingComponents(snapshot)) {
      const componentSet = new Set(component);
      const componentEndpointIds = new Set(component.flatMap((edge) => endpointIdsFor(edge)));
      const group = [...component];
      for (const edge of visibleEdges) {
        if (componentSet.has(edge)) {
          continue;
        }
        if (endpointIdsFor(edge).some((id) => componentEndpointIds.has(id))) {
          group.push(edge);
        }
      }
      group.sort((a, b) => (edgeIndex.get(a) ?? 0) - (edgeIndex.get(b) ?? 0));
      groups.push(group);
    }
    return groups;
  }, "pairSearchGroups");
  const crossingCountWithSingleReplacement = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((current, edge, replacement) => crossingCountWithReplacements(
    current,
    /* @__PURE__ */ new Map([[edge, replacement]])
  ), "crossingCountWithSingleReplacement");
  const currentCrossingsByEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((current) => {
    const result = /* @__PURE__ */ new Map();
    for (const pair of current.pairs) {
      result.set(pair.first, (result.get(pair.first) ?? 0) + pair.count);
      result.set(pair.second, (result.get(pair.second) ?? 0) + pair.count);
    }
    return result;
  }, "currentCrossingsByEdge");
  const pathLength = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((points) => points.slice(1).reduce((sum, point, index) => {
    const previous = points[index];
    return sum + Math.abs(point.x - previous.x) + Math.abs(point.y - previous.y);
  }, 0), "pathLength");
  const totalBends = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => visibleEdges.reduce(
    (sum, edge) => sum + countOrthogonalBends(replacementPointsFor(edge, replacements)),
    0
  ), "totalBends");
  const totalLength = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((replacements = /* @__PURE__ */ new Map()) => visibleEdges.reduce(
    (sum, edge) => sum + pathLength(replacementPointsFor(edge, replacements)),
    0
  ), "totalLength");
  const pathHasSegmentConflict = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, path, replacements = /* @__PURE__ */ new Map()) => {
    const pathSegments = segmentsFor(path);
    for (const other of visibleEdges) {
      if (other === edge) {
        continue;
      }
      for (const candidateSegment of pathSegments) {
        for (const otherSegment of segmentsFor(replacementPointsFor(other, replacements))) {
          if (sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED) {
            return true;
          }
        }
      }
    }
    return false;
  }, "pathHasSegmentConflict");
  const pathHitsNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, path) => {
    const endpointIds = [edge.start, edge.end].filter(
      (id) => Boolean(id)
    );
    for (const segment of segmentsFor(path)) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -2)) {
        return true;
      }
    }
    return false;
  }, "pathHitsNode");
  const pushOrthogonalCandidate = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, points) => {
    const candidate = simplifyPolyline(dedupeConsecutivePoints(points));
    if (segmentsFor(candidate).length === candidate.length - 1) {
      candidates.push(candidate);
    }
  }, "pushOrthogonalCandidate");
  const sideIsHorizontal = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((side) => side === "left" || side === "right", "sideIsHorizontal");
  const localTrackForSameSide = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((src, side, dst) => {
    switch (side) {
      case "left":
        return Math.min(src.x, dst.x) - ANCHOR;
      case "right":
        return Math.max(src.x, dst.x) + ANCHOR;
      case "top":
        return Math.min(src.y, dst.y) - ANCHOR;
      case "bottom":
        return Math.max(src.y, dst.y) + ANCHOR;
    }
  }, "localTrackForSameSide");
  const addSameSideCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, src, srcSide, dst) => {
    const outward = srcSide === "left" || srcSide === "top" ? -1 : 1;
    const trackSeeds = [localTrackForSameSide(src, srcSide, dst), outsideTracks[srcSide]];
    for (const seed of trackSeeds) {
      for (let channel = 0; channel <= EXTRA_CHANNEL_COUNT; channel++) {
        pushOrthogonalCandidate(
          candidates,
          buildSameSideTrackPath(src, srcSide, dst, seed + outward * ANCHOR * channel)
        );
      }
    }
  }, "addSameSideCandidates");
  const addHorizontalToVerticalCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, src, srcSide, dst, dstSide) => {
    for (const xTrack of outwardTracksForSide(srcSide)) {
      for (const yTrack of outwardTracksForSide(dstSide)) {
        pushOrthogonalCandidate(candidates, [
          src,
          { x: xTrack, y: src.y },
          { x: xTrack, y: yTrack },
          { x: dst.x, y: yTrack },
          dst
        ]);
      }
    }
  }, "addHorizontalToVerticalCandidates");
  const addVerticalToHorizontalCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, src, srcSide, dst, dstSide) => {
    for (const yTrack of outwardTracksForSide(srcSide)) {
      for (const xTrack of outwardTracksForSide(dstSide)) {
        pushOrthogonalCandidate(candidates, [
          src,
          { x: src.x, y: yTrack },
          { x: xTrack, y: yTrack },
          { x: xTrack, y: dst.y },
          dst
        ]);
      }
    }
  }, "addVerticalToHorizontalCandidates");
  const addHorizontalPairCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, src, srcSide, dst, dstSide) => {
    const yTracks = [...outwardTracksForSide("top"), ...outwardTracksForSide("bottom")];
    for (const srcTrack of outwardTracksForSide(srcSide)) {
      for (const dstTrack of outwardTracksForSide(dstSide)) {
        for (const yTrack of yTracks) {
          pushOrthogonalCandidate(candidates, [
            src,
            { x: srcTrack, y: src.y },
            { x: srcTrack, y: yTrack },
            { x: dstTrack, y: yTrack },
            { x: dstTrack, y: dst.y },
            dst
          ]);
        }
      }
    }
  }, "addHorizontalPairCandidates");
  const addVerticalPairCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, src, srcSide, dst, dstSide) => {
    const xTracks = [...outwardTracksForSide("left"), ...outwardTracksForSide("right")];
    for (const srcTrack of outwardTracksForSide(srcSide)) {
      for (const dstTrack of outwardTracksForSide(dstSide)) {
        for (const xTrack of xTracks) {
          pushOrthogonalCandidate(candidates, [
            src,
            { x: src.x, y: srcTrack },
            { x: xTrack, y: srcTrack },
            { x: xTrack, y: dstTrack },
            { x: dst.x, y: dstTrack },
            dst
          ]);
        }
      }
    }
  }, "addVerticalPairCandidates");
  const dedupeCandidatePaths = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates) => {
    const seen = /* @__PURE__ */ new Set();
    return candidates.map((candidate) => dedupeConsecutivePoints(candidate)).filter((candidate) => {
      const key = candidate.map((point) => `${point.x.toFixed(3)},${point.y.toFixed(3)}`).join("|");
      if (seen.has(key) || candidate.length < 2) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, "dedupeCandidatePaths");
  const buildCandidatesForSides = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((src, srcSide, dst, dstSide) => {
    const candidates = [];
    const base = buildOrthogonalPortPath(src, srcSide, dst, dstSide, ANCHOR, EPS_LOCAL);
    if (base) {
      pushOrthogonalCandidate(candidates, base);
    }
    if (srcSide === dstSide) {
      addSameSideCandidates(candidates, src, srcSide, dst);
    }
    const srcHorizontal = sideIsHorizontal(srcSide);
    const dstHorizontal = sideIsHorizontal(dstSide);
    if (srcHorizontal && !dstHorizontal) {
      addHorizontalToVerticalCandidates(candidates, src, srcSide, dst, dstSide);
    } else if (!srcHorizontal && dstHorizontal) {
      addVerticalToHorizontalCandidates(candidates, src, srcSide, dst, dstSide);
    } else if (srcHorizontal) {
      addHorizontalPairCandidates(candidates, src, srcSide, dst, dstSide);
    } else {
      addVerticalPairCandidates(candidates, src, srcSide, dst, dstSide);
    }
    return dedupeCandidatePaths(candidates);
  }, "buildCandidatesForSides");
  const addVerticalDepartureOuterTrackCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, first, departure, dstNode) => {
    const externalXTracks = [...outwardTracksForSide("left"), ...outwardTracksForSide("right")];
    const externalYTracks = [...outwardTracksForSide("top"), ...outwardTracksForSide("bottom")];
    for (const side of sides) {
      const dst = portForRectSide(dstNode, side);
      const targetYTracks = side === "top" || side === "bottom" ? outwardTracksForSide(side) : externalYTracks;
      for (const track of externalXTracks) {
        pushOrthogonalCandidate(candidates, [
          first,
          departure,
          { x: track, y: departure.y },
          { x: track, y: dst.y },
          dst
        ]);
        for (const targetTrack of targetYTracks) {
          pushOrthogonalCandidate(candidates, [
            first,
            departure,
            { x: track, y: departure.y },
            { x: track, y: targetTrack },
            { x: dst.x, y: targetTrack },
            dst
          ]);
        }
      }
    }
  }, "addVerticalDepartureOuterTrackCandidates");
  const addHorizontalDepartureOuterTrackCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidates, first, departure, dstNode) => {
    const externalXTracks = [...outwardTracksForSide("left"), ...outwardTracksForSide("right")];
    const externalYTracks = [...outwardTracksForSide("top"), ...outwardTracksForSide("bottom")];
    for (const side of sides) {
      const dst = portForRectSide(dstNode, side);
      const targetXTracks = side === "left" || side === "right" ? outwardTracksForSide(side) : externalXTracks;
      for (const track of externalYTracks) {
        pushOrthogonalCandidate(candidates, [
          first,
          departure,
          { x: departure.x, y: track },
          { x: dst.x, y: track },
          dst
        ]);
        for (const targetTrack of targetXTracks) {
          pushOrthogonalCandidate(candidates, [
            first,
            departure,
            { x: departure.x, y: track },
            { x: targetTrack, y: track },
            { x: targetTrack, y: dst.y },
            dst
          ]);
        }
      }
    }
  }, "addHorizontalDepartureOuterTrackCandidates");
  const terminalPreservingOuterTrackCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
    const srcId = edge.start;
    const dstId = edge.end;
    const dstNode = dstId ? nodeInfoById.get(dstId) : void 0;
    if (!srcId || !dstNode) {
      return [];
    }
    const points = dedupeConsecutivePoints(edge.points ?? []);
    if (points.length < 4) {
      return [];
    }
    const first = points[0];
    const departure = points[1];
    const candidates = [];
    if (isVerticalSegment(first, departure, EPS_LOCAL)) {
      addVerticalDepartureOuterTrackCandidates(candidates, first, departure, dstNode);
    } else if (isHorizontalSegment(first, departure, EPS_LOCAL)) {
      addHorizontalDepartureOuterTrackCandidates(candidates, first, departure, dstNode);
    }
    return candidates;
  }, "terminalPreservingOuterTrackCandidates");
  const candidatePathsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge) => {
    const srcId = edge.start;
    const dstId = edge.end;
    const srcNode = srcId ? nodeInfoById.get(srcId) : void 0;
    const dstNode = dstId ? nodeInfoById.get(dstId) : void 0;
    if (!srcNode || !dstNode) {
      return [];
    }
    const candidates = [];
    for (const srcSide of sides) {
      const srcPort = portForRectSide(srcNode, srcSide);
      for (const dstSide of sides) {
        candidates.push(
          ...buildCandidatesForSides(srcPort, srcSide, portForRectSide(dstNode, dstSide), dstSide)
        );
      }
    }
    candidates.push(...terminalPreservingOuterTrackCandidates(edge));
    return candidates;
  }, "candidatePathsFor");
  const currentSegmentsByEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => new Map(visibleEdges.map((edge) => [edge, segmentsFor(replacementPointsFor(edge))])), "currentSegmentsByEdge");
  const sharedTrackConflictsFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, candidateSegments, baseSegments) => {
    const conflicts = /* @__PURE__ */ new Set();
    for (const other of visibleEdges) {
      if (other === edge) {
        continue;
      }
      const otherSegments = baseSegments.get(other) ?? segmentsFor(replacementPointsFor(other));
      if (candidateSegments.some(
        (candidateSegment) => otherSegments.some(
          (otherSegment) => sameAxisSegmentOverlapLength(candidateSegment, otherSegment, 0.5) >= MIN_SHARED
        )
      )) {
        conflicts.add(other);
      }
    }
    return conflicts;
  }, "sharedTrackConflictsFor");
  const pairCandidatesFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, current, baseSegments, crossingCountByEdge) => {
    const seen = /* @__PURE__ */ new Set();
    const candidates = candidatePathsFor(edge).map((candidate) => simplifyPolyline(dedupeConsecutivePoints(candidate))).filter((candidate) => {
      if (pathHitsNode(edge, candidate)) {
        return false;
      }
      const key = candidate.map((point) => `${point.x.toFixed(3)},${point.y.toFixed(3)}`).join("|");
      if (seen.has(key) || candidate.length < 2) {
        return false;
      }
      seen.add(key);
      return true;
    }).map((candidate) => {
      const candidateSegments = segmentsFor(candidate);
      let replacementAffected = 0;
      for (const other of visibleEdges) {
        if (other === edge) {
          continue;
        }
        replacementAffected += crossingCountBetweenSegments(
          candidateSegments,
          baseSegments.get(other) ?? segmentsFor(replacementPointsFor(other))
        );
      }
      return {
        candidate,
        candidateSegments,
        crossings: current.count - (crossingCountByEdge.get(edge) ?? 0) + replacementAffected,
        bends: countOrthogonalBends(candidate, EPS_LOCAL),
        totalBends: countOrthogonalBends(candidate),
        length: pathLength(candidate)
      };
    }).filter(({ crossings }) => crossings <= current.count).sort((a, b) => a.crossings - b.crossings || a.bends - b.bends || a.length - b.length);
    return candidates.slice(0, MAX_PAIR_CANDIDATES_PER_EDGE).map((candidate) => {
      return {
        path: candidate.candidate,
        segments: candidate.candidateSegments,
        sharedTrackConflicts: sharedTrackConflictsFor(
          edge,
          candidate.candidateSegments,
          baseSegments
        ),
        totalBends: candidate.totalBends,
        length: candidate.length
      };
    });
  }, "pairCandidatesFor");
  const pairCrossingCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((current, firstEdge, firstCandidate, secondEdge, secondCandidate, baseSegments) => {
    let currentAffected = 0;
    for (const pair of current.pairs) {
      if (pair.first === firstEdge || pair.second === firstEdge || pair.first === secondEdge || pair.second === secondEdge) {
        currentAffected += pair.count;
      }
    }
    let replacementAffected = crossingCountBetweenSegments(
      firstCandidate.segments,
      secondCandidate.segments
    );
    for (const other of visibleEdges) {
      if (other === firstEdge || other === secondEdge) {
        continue;
      }
      const otherSegments = baseSegments.get(other) ?? segmentsFor(replacementPointsFor(other));
      replacementAffected += crossingCountBetweenSegments(firstCandidate.segments, otherSegments) + crossingCountBetweenSegments(secondCandidate.segments, otherSegments);
    }
    return current.count - currentAffected + replacementAffected;
  }, "pairCrossingCount");
  const conflictsOnlyWith = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidate, edge) => {
    for (const conflict of candidate.sharedTrackConflicts) {
      if (conflict !== edge) {
        return false;
      }
    }
    return true;
  }, "conflictsOnlyWith");
  const candidatesShareTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((firstCandidate, secondCandidate) => firstCandidate.segments.some(
    (firstSegment) => secondCandidate.segments.some(
      (secondSegment) => sameAxisSegmentOverlapLength(firstSegment, secondSegment, 0.5) >= MIN_SHARED
    )
  ), "candidatesShareTrack");
  const pairCandidatesAreCompatible = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((first, firstCandidate, second, secondCandidate) => conflictsOnlyWith(firstCandidate, second.edge) && conflictsOnlyWith(secondCandidate, first.edge) && !candidatesShareTrack(firstCandidate, secondCandidate), "pairCandidatesAreCompatible");
  const scorePairReplacement = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((context, first, firstCandidate, second, secondCandidate) => {
    const crossings = pairCrossingCount(
      context.current,
      first.edge,
      firstCandidate,
      second.edge,
      secondCandidate,
      context.baseSegments
    );
    if (crossings >= context.current.count) {
      return void 0;
    }
    return {
      replacements: /* @__PURE__ */ new Map([
        [first.edge, firstCandidate.path],
        [second.edge, secondCandidate.path]
      ]),
      crossings,
      bends: context.currentBends - (context.baseBendsByEdge.get(first.edge) ?? 0) - (context.baseBendsByEdge.get(second.edge) ?? 0) + firstCandidate.totalBends + secondCandidate.totalBends,
      length: context.currentLength - (context.baseLengthByEdge.get(first.edge) ?? 0) - (context.baseLengthByEdge.get(second.edge) ?? 0) + firstCandidate.length + secondCandidate.length
    };
  }, "scorePairReplacement");
  const pairScoreIsBetter = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((candidate, best) => candidate.crossings < best.crossings || candidate.crossings === best.crossings && (candidate.bends < best.bends || candidate.bends === best.bends && candidate.length < best.length), "pairScoreIsBetter");
  const bestScoreForOptionPair = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((context, first, second, best) => {
    let pairBest = best;
    for (const firstCandidate of first.candidates) {
      for (const secondCandidate of second.candidates) {
        if (!pairCandidatesAreCompatible(first, firstCandidate, second, secondCandidate)) {
          continue;
        }
        const score = scorePairReplacement(context, first, firstCandidate, second, secondCandidate);
        if (score && pairScoreIsBetter(score, pairBest)) {
          pairBest = score;
        }
      }
    }
    return pairBest;
  }, "bestScoreForOptionPair");
  const bestPairedReplacement = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((current) => {
    const currentBends = totalBends();
    const currentLength = totalLength();
    const baseSegments = currentSegmentsByEdge();
    const crossingCountByEdge = currentCrossingsByEdge(current);
    const baseBendsByEdge = new Map(
      visibleEdges.map((edge) => [edge, countOrthogonalBends(replacementPointsFor(edge))])
    );
    const baseLengthByEdge = new Map(
      visibleEdges.map((edge) => [edge, pathLength(replacementPointsFor(edge))])
    );
    const optionsByEdge = /* @__PURE__ */ new Map();
    const groups = pairSearchGroups(current);
    for (const group of groups) {
      for (const edge of group) {
        if (optionsByEdge.has(edge)) {
          continue;
        }
        const candidates = pairCandidatesFor(edge, current, baseSegments, crossingCountByEdge);
        if (candidates.length > 0) {
          optionsByEdge.set(edge, { edge, candidates });
        }
      }
    }
    let best = {
      replacements: /* @__PURE__ */ new Map(),
      crossings: current.count,
      bends: currentBends,
      length: currentLength
    };
    const scoringContext = {
      current,
      currentBends,
      currentLength,
      baseBendsByEdge,
      baseLengthByEdge,
      baseSegments
    };
    for (const group of groups) {
      const crossingEdgeSet = new Set(group.filter((edge) => current.edgeSet.has(edge)));
      const options = group.map((edge) => optionsByEdge.get(edge)).filter((option) => Boolean(option));
      for (let i = 0; i < options.length; i++) {
        const first = options[i];
        for (let j = i + 1; j < options.length; j++) {
          const second = options[j];
          if (!crossingEdgeSet.has(first.edge) && !crossingEdgeSet.has(second.edge)) {
            continue;
          }
          best = bestScoreForOptionPair(scoringContext, first, second, best);
        }
      }
    }
    return best.replacements.size > 0 ? best.replacements : void 0;
  }, "bestPairedReplacement");
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const current = crossingSnapshot();
    const currentCrossings = current.count;
    if (currentCrossings === 0) {
      return;
    }
    let bestEdge;
    let bestPath;
    let bestCrossings = currentCrossings;
    let bestBends = Number.POSITIVE_INFINITY;
    for (const edge of current.edges) {
      const currentEdgeBends = countOrthogonalBends(replacementPointsFor(edge), EPS_LOCAL);
      for (const candidate of candidatePathsFor(edge)) {
        const candidateHitsNode = pathHitsNode(edge, candidate);
        const candidateHasSegmentConflict = !candidateHitsNode && pathHasSegmentConflict(edge, candidate);
        const candidateCrossings = crossingCountWithSingleReplacement(current, edge, candidate);
        const candidateBends = countOrthogonalBends(candidate, EPS_LOCAL);
        if (candidateHitsNode || candidateHasSegmentConflict) {
          continue;
        }
        const improvesCurrentEdge = candidateCrossings < currentCrossings || candidateCrossings === currentCrossings && candidateBends < currentEdgeBends;
        if (!improvesCurrentEdge) {
          continue;
        }
        if (candidateCrossings > bestCrossings || candidateCrossings === bestCrossings && candidateBends >= bestBends) {
          continue;
        }
        bestEdge = edge;
        bestPath = candidate;
        bestCrossings = candidateCrossings;
        bestBends = candidateBends;
      }
    }
    if (bestEdge && bestPath) {
      bestEdge.points = bestPath;
      continue;
    }
    const pairedReplacement = bestPairedReplacement(current);
    if (!pairedReplacement) {
      return;
    }
    for (const [edge, points] of pairedReplacement) {
      edge.points = points;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(resolveRenderedOrthogonalCrossings, "resolveRenderedOrthogonalCrossings");

// src/rendering-util/layout-algorithms/swimlanes/direction/detourSimplification.ts
var EPS4 = 1e-3;
var MIN_SHARED2 = 8;
function simplifyDetouredEdges(edges, nodes) {
  const { nodeInfoById, realNodeRects } = collectRealNodeBounds(nodes);
  const sides = ["top", "bottom", "left", "right"];
  const ANCHOR = 20;
  const outsideTracks = {
    top: Math.min(...realNodeRects.map((node) => node.rect.top)) - ANCHOR,
    bottom: Math.max(...realNodeRects.map((node) => node.rect.bottom)) + ANCHOR,
    left: Math.min(...realNodeRects.map((node) => node.rect.left)) - ANCHOR,
    right: Math.max(...realNodeRects.map((node) => node.rect.right)) + ANCHOR
  };
  const buildOrthogonalPathCandidates = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((src, srcSide, dst, dstSide) => {
    const paths = [];
    const base = buildOrthogonalPortPath(src, srcSide, dst, dstSide, ANCHOR, EPS4);
    if (base) {
      paths.push(base);
    }
    if (srcSide === dstSide) {
      paths.push(buildSameSideTrackPath(src, srcSide, dst, outsideTracks[srcSide]));
    }
    return paths;
  }, "buildOrthogonalPathCandidates");
  const pathHitsNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pts, excludeIds) => {
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      if (segmentHitsAnyRect(a, b, realNodeRects, excludeIds, 1)) {
        return true;
      }
    }
    return false;
  }, "pathHitsNode");
  const pathConflictCount = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((path, currentEdge, includeIncidentEdges = false) => {
    let conflicts = 0;
    const pathSegments = orthogonalSegmentsForPoints(path, EPS4);
    const currentStart = currentEdge.start;
    const currentEnd = currentEdge.end;
    for (const other of edges) {
      if (other === currentEdge || other.isLayoutOnly) {
        continue;
      }
      const otherStart = other.start;
      const otherEnd = other.end;
      if (!includeIncidentEdges && currentStart && currentEnd && (otherStart === currentStart || otherStart === currentEnd || otherEnd === currentStart || otherEnd === currentEnd)) {
        continue;
      }
      const otherPts = other.points;
      if (!otherPts || otherPts.length < 2) {
        continue;
      }
      for (const pathSegment of pathSegments) {
        for (const otherSegment of orthogonalSegmentsForPoints(otherPts, EPS4)) {
          if (orthogonalSegmentsCross(
            pathSegment.a,
            pathSegment.b,
            otherSegment.a,
            otherSegment.b,
            EPS4,
            EPS4
          )) {
            conflicts++;
            continue;
          }
          if (sameAxisSegmentOverlapLength(pathSegment, otherSegment, EPS4) >= MIN_SHARED2) {
            conflicts++;
          }
        }
      }
    }
    return conflicts;
  }, "pathConflictCount");
  const BEND_THRESHOLD = 4;
  const nearestSideOfRect = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pt, info) => {
    const dTop = Math.abs(pt.y - info.rect.top);
    const dBottom = Math.abs(pt.y - info.rect.bottom);
    const dLeft = Math.abs(pt.x - info.rect.left);
    const dRight = Math.abs(pt.x - info.rect.right);
    let best = "top";
    let bestDist = dTop;
    if (dBottom < bestDist) {
      best = "bottom";
      bestDist = dBottom;
    }
    if (dLeft < bestDist) {
      best = "left";
      bestDist = dLeft;
    }
    if (dRight < bestDist) {
      best = "right";
      bestDist = dRight;
    }
    return best;
  }, "nearestSideOfRect");
  const faceClaims = /* @__PURE__ */ new Map();
  const addFaceClaim = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((nodeId, side, edgeId) => {
    const claims = faceClaims.get(nodeId) ?? [];
    claims.push({ side, edgeId });
    faceClaims.set(nodeId, claims);
  }, "addFaceClaim");
  for (const e of edges) {
    if (e.isLayoutOnly) {
      continue;
    }
    const pts = e.points ?? [];
    if (pts.length < 1) {
      continue;
    }
    const eId = e.id ?? "";
    const startId = e.start;
    const endId = e.end;
    if (startId) {
      const info = nodeInfoById.get(startId);
      if (info) {
        addFaceClaim(startId, nearestSideOfRect(pts[0], info), eId);
      }
    }
    if (endId) {
      const info = nodeInfoById.get(endId);
      if (info) {
        addFaceClaim(endId, nearestSideOfRect(pts[pts.length - 1], info), eId);
      }
    }
  }
  const faceIsClaimed = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((nodeId, side, ignoreEdgeId) => {
    return faceClaims.get(nodeId)?.some((c) => c.edgeId !== ignoreEdgeId && c.side === side) ?? false;
  }, "faceIsClaimed");
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const pts = edge.points;
    if (!pts || pts.length < 2) {
      continue;
    }
    const currentBends = countOrthogonalBends(pts, EPS4);
    if (currentBends < BEND_THRESHOLD) {
      continue;
    }
    const srcId = edge.start;
    const dstId = edge.end;
    if (!srcId || !dstId) {
      continue;
    }
    const srcInfo = nodeInfoById.get(srcId);
    const dstInfo = nodeInfoById.get(dstId);
    if (!srcInfo || !dstInfo) {
      continue;
    }
    const edgeId = edge.id ?? "";
    const currentCrossingConflicts = pathConflictCount(pts, edge, true);
    const currentNonIncidentConflicts = pathConflictCount(pts, edge);
    let bestPath;
    let bestCrossingConflicts = currentCrossingConflicts;
    let bestBends = currentBends;
    for (const srcSide of sides) {
      if (faceIsClaimed(srcId, srcSide, edgeId)) {
        continue;
      }
      const srcPort = portForRectSide(srcInfo, srcSide);
      for (const dstSide of sides) {
        if (faceIsClaimed(dstId, dstSide, edgeId)) {
          continue;
        }
        const dstPort = portForRectSide(dstInfo, dstSide);
        for (const path of buildOrthogonalPathCandidates(srcPort, srcSide, dstPort, dstSide)) {
          if (pathHitsNode(path, [srcId, dstId])) {
            continue;
          }
          const pathBends = countOrthogonalBends(path, EPS4);
          if (currentCrossingConflicts > 0) {
            const pathCrossingConflicts = pathConflictCount(path, edge, true);
            if (pathCrossingConflicts > bestCrossingConflicts || pathCrossingConflicts === bestCrossingConflicts && pathBends >= bestBends) {
              continue;
            }
            bestCrossingConflicts = pathCrossingConflicts;
            bestBends = pathBends;
            bestPath = path;
            continue;
          }
          if (pathConflictCount(path, edge) > currentNonIncidentConflicts) {
            continue;
          }
          if (pathBends < bestBends) {
            bestBends = pathBends;
            bestPath = path;
          }
        }
      }
    }
    if (bestPath) {
      edge.points = bestPath;
      const refreshSrc = faceClaims.get(srcId);
      if (refreshSrc) {
        faceClaims.set(
          srcId,
          refreshSrc.filter((c) => c.edgeId !== edgeId)
        );
      }
      const refreshDst = faceClaims.get(dstId);
      if (refreshDst) {
        faceClaims.set(
          dstId,
          refreshDst.filter((c) => c.edgeId !== edgeId)
        );
      }
      addFaceClaim(srcId, nearestSideOfRect(bestPath[0], srcInfo), edgeId);
      addFaceClaim(dstId, nearestSideOfRect(bestPath[bestPath.length - 1], dstInfo), edgeId);
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(simplifyDetouredEdges, "simplifyDetouredEdges");

// src/rendering-util/layout-algorithms/swimlanes/direction/labelAnchoring.ts
var EPS5 = 1e-3;
var MARKER_CLEARANCE_LENGTH = 10;
var MARKER_CLEARANCE_HALF_WIDTH = 7;
function markerClearanceRectFor(pts, atStart) {
  const terminalIndex = atStart ? 0 : pts.length - 1;
  const step = atStart ? 1 : -1;
  const tip = pts[terminalIndex];
  const inner = pts[terminalIndex + step];
  if (!tip || !inner) {
    return void 0;
  }
  const dx = inner.x - tip.x;
  const dy = inner.y - tip.y;
  const len = Math.abs(dx) + Math.abs(dy);
  if (len < EPS5) {
    return void 0;
  }
  if (Math.abs(dy) <= EPS5) {
    const x2 = tip.x + Math.sign(dx) * MARKER_CLEARANCE_LENGTH;
    return {
      left: Math.min(tip.x, x2),
      right: Math.max(tip.x, x2),
      top: tip.y - MARKER_CLEARANCE_HALF_WIDTH,
      bottom: tip.y + MARKER_CLEARANCE_HALF_WIDTH
    };
  }
  if (Math.abs(dx) <= EPS5) {
    const y2 = tip.y + Math.sign(dy) * MARKER_CLEARANCE_LENGTH;
    return {
      left: tip.x - MARKER_CLEARANCE_HALF_WIDTH,
      right: tip.x + MARKER_CLEARANCE_HALF_WIDTH,
      top: Math.min(tip.y, y2),
      bottom: Math.max(tip.y, y2)
    };
  }
  return {
    left: Math.min(tip.x, inner.x),
    right: Math.max(tip.x, inner.x),
    top: Math.min(tip.y, inner.y),
    bottom: Math.max(tip.y, inner.y)
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(markerClearanceRectFor, "markerClearanceRectFor");
function normalizeRect(rect) {
  return {
    left: Math.min(rect.left, rect.right),
    right: Math.max(rect.left, rect.right),
    top: Math.min(rect.top, rect.bottom),
    bottom: Math.max(rect.top, rect.bottom)
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(normalizeRect, "normalizeRect");
function labelOverlapsOwnMarker(rect, pts) {
  const visiblePts = dedupeConsecutivePoints(pts);
  const startMarker = markerClearanceRectFor(visiblePts, true);
  const endMarker = markerClearanceRectFor(visiblePts, false);
  return [startMarker, endMarker].some(
    (marker) => marker && rectsOverlap(rect, normalizeRect(marker))
  );
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(labelOverlapsOwnMarker, "labelOverlapsOwnMarker");
function anchorLabelsToPolyline(edges, nodeByIdMap) {
  const allEdgeSegments = [];
  for (const other of edges) {
    if (other.isLayoutOnly) {
      continue;
    }
    const pts = other.points;
    if (!pts || pts.length < 2) {
      continue;
    }
    for (let i = 0; i < pts.length - 1; i++) {
      allEdgeSegments.push({ edgeId: other.id, p1: pts[i], p2: pts[i + 1] });
    }
  }
  const foreignNodeRects = [];
  const laneGroups = [];
  for (const n of nodeByIdMap.values()) {
    const isGroup = n.isGroup;
    const parentId = n.parentId;
    if (isGroup && !parentId) {
      const rect2 = rectOfNodeBounds(n);
      if (rect2) {
        laneGroups.push({
          id: n.id,
          rect: rect2
        });
      }
      continue;
    }
    if (isGroup) {
      continue;
    }
    if (n.isEdgeLabel) {
      continue;
    }
    const rect = rectOfNodeBounds(n);
    if (!rect) {
      continue;
    }
    foreignNodeRects.push({
      nodeId: n.id,
      rect
    });
  }
  const LABEL_PLACEMENT_BUFFER = 3;
  const LABEL_LANE_MARGIN = 1;
  const LABEL_ENDPOINT_CLEARANCE = 12;
  const labelOverlapsForeignNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((labelId, rect) => {
    const buffered = inflateRect(rect, LABEL_PLACEMENT_BUFFER);
    for (const { nodeId, rect: nr } of foreignNodeRects) {
      if (nodeId === labelId) {
        continue;
      }
      if (rectsOverlap(buffered, nr)) {
        return true;
      }
    }
    return false;
  }, "labelOverlapsForeignNode");
  const labelOverlapsForeignEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeId, rect) => {
    const buffered = inflateRect(rect, LABEL_PLACEMENT_BUFFER);
    for (const s of allEdgeSegments) {
      if (s.edgeId === edgeId) {
        continue;
      }
      if (segmentBoundsOverlapRect(s.p1, s.p2, buffered)) {
        return true;
      }
    }
    return false;
  }, "labelOverlapsForeignEdge");
  const labelOverlapsAnything = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((labelId, edgeId, rect) => labelOverlapsForeignNode(labelId, rect) || labelOverlapsForeignEdge(edgeId, rect), "labelOverlapsAnything");
  const placedLabelRects = [];
  const findContainingLane = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((rect) => {
    for (const { id, rect: laneRect } of laneGroups) {
      if (rectContainsRect(laneRect, rect)) {
        return id;
      }
    }
    return void 0;
  }, "findContainingLane");
  const overlapsPlacedLabel = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((labelId, rect) => placedLabelRects.some(
    (placed) => placed.labelId !== labelId && rectsOverlap(rect, placed.rect)
  ), "overlapsPlacedLabel");
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const labelId = edge.labelNodeId;
    if (!labelId) {
      continue;
    }
    const labelNode = nodeByIdMap.get(labelId);
    if (!labelNode) {
      continue;
    }
    const pts = edge.points;
    if (!pts || pts.length < 2) {
      continue;
    }
    const lw = labelNode.width ?? 0;
    const lh = labelNode.height ?? 0;
    if (lw <= 0 || lh <= 0) {
      continue;
    }
    const segments = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      const dx = Math.abs(a.x - b.x);
      const dy = Math.abs(a.y - b.y);
      if (dx < EPS5 && dy < EPS5) {
        continue;
      }
      if (dx >= EPS5 && dy >= EPS5) {
        continue;
      }
      segments.push({
        idx: i,
        length: dx + dy,
        orientation: dx >= EPS5 ? "horizontal" : "vertical",
        midX: (a.x + b.x) / 2,
        midY: (a.y + b.y) / 2
      });
    }
    if (segments.length === 0) {
      continue;
    }
    const middleSegments = segments.length >= 3 ? segments.filter((s) => s.idx > 0 && s.idx < segments.length - 1) : segments;
    const poolBase = middleSegments.length > 0 ? middleSegments : segments;
    const labelLongAxis = lw >= lh ? "horizontal" : "vertical";
    const rankSegments = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pool) => {
      return [...pool].sort((a, b) => {
        const aLongAxis = a.orientation === labelLongAxis;
        const bLongAxis = b.orientation === labelLongAxis;
        if (aLongAxis !== bLongAxis) {
          return aLongAxis ? -1 : 1;
        }
        const aFits = a.length >= (a.orientation === "horizontal" ? lw : lh) + 2;
        const bFits = b.length >= (b.orientation === "horizontal" ? lw : lh) + 2;
        if (aFits !== bFits) {
          return aFits ? -1 : 1;
        }
        return b.length - a.length;
      });
    }, "rankSegments");
    const firstVisibleSegment = segments[0];
    const lastVisibleSegment = segments[segments.length - 1];
    const ALONG_SEGMENT_TS = [0.5, 0.25, 0.75, 0.05, 0.95, 0.15, 0.85, 0.1, 0.9];
    const anchorAtT = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg, t) => {
      const a = pts[seg.idx];
      const b = pts[seg.idx + 1];
      return {
        midX: a.x + (b.x - a.x) * t,
        midY: a.y + (b.y - a.y) * t
      };
    }, "anchorAtT");
    const clamp = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((value, min, max) => Math.min(max, Math.max(min, value)), "clamp");
    const pointInsideRectInclusive = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((point, rect) => point.midX >= rect.left - EPS5 && point.midX <= rect.right + EPS5 && point.midY >= rect.top - EPS5 && point.midY <= rect.bottom + EPS5, "pointInsideRectInclusive");
    const placementForAnchor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((anchor) => {
      const centeredRect = rectFromCenterSize(anchor.midX, anchor.midY, lw, lh);
      const centeredLane = findContainingLane(centeredRect);
      if (centeredLane) {
        return { laneId: centeredLane, anchor, rect: centeredRect };
      }
      const containingLane = laneGroups.find(({ rect }) => pointInsideRectInclusive(anchor, rect));
      if (!containingLane) {
        return void 0;
      }
      const minX = containingLane.rect.left + lw / 2 + LABEL_LANE_MARGIN;
      const maxX = containingLane.rect.right - lw / 2 - LABEL_LANE_MARGIN;
      const minY = containingLane.rect.top + lh / 2 + LABEL_LANE_MARGIN;
      const maxY = containingLane.rect.bottom - lh / 2 - LABEL_LANE_MARGIN;
      if (minX > maxX || minY > maxY) {
        return void 0;
      }
      const clampedAnchor = {
        midX: clamp(anchor.midX, minX, maxX),
        midY: clamp(anchor.midY, minY, maxY)
      };
      const clampedRect = rectFromCenterSize(clampedAnchor.midX, clampedAnchor.midY, lw, lh);
      return pointInsideRectInclusive(anchor, clampedRect) ? { laneId: containingLane.id, anchor: clampedAnchor, rect: clampedRect } : void 0;
    }, "placementForAnchor");
    const distanceAlongSegment = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg, anchor, endpoint) => seg.orientation === "horizontal" ? Math.abs(anchor.midX - endpoint.x) : Math.abs(anchor.midY - endpoint.y), "distanceAlongSegment");
    const labelClearsTerminalEndpoints = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg, anchor) => {
      const labelHalfExtent = seg.orientation === "horizontal" ? lw / 2 : lh / 2;
      const requiredDistance = labelHalfExtent + LABEL_ENDPOINT_CLEARANCE;
      if (seg === firstVisibleSegment) {
        const start = pts[seg.idx];
        if (distanceAlongSegment(seg, anchor, start) + EPS5 < requiredDistance) {
          return false;
        }
      }
      if (seg === lastVisibleSegment) {
        const end = pts[seg.idx + 1];
        if (distanceAlongSegment(seg, anchor, end) + EPS5 < requiredDistance) {
          return false;
        }
      }
      return true;
    }, "labelClearsTerminalEndpoints");
    const tryPool = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pool) => {
      const rankedPool = rankSegments(pool);
      for (const seg of rankedPool) {
        for (const t of ALONG_SEGMENT_TS) {
          const anchor = anchorAtT(seg, t);
          if (!labelClearsTerminalEndpoints(seg, anchor)) {
            continue;
          }
          const placement = placementForAnchor(anchor);
          if (!placement) {
            continue;
          }
          if (labelOverlapsOwnMarker(placement.rect, pts)) {
            continue;
          }
          if (overlapsPlacedLabel(labelId, placement.rect)) {
            continue;
          }
          if (!labelOverlapsAnything(labelId, edge.id, placement.rect)) {
            return { laneId: placement.laneId, anchor: placement.anchor };
          }
        }
      }
      return void 0;
    }, "tryPool");
    const findLaneContainingFallback = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pool, requireEndpointClearance, allowForeignEdgeOverlap = false) => {
      const rankedPool = rankSegments(pool);
      for (const seg of rankedPool) {
        const anchor = { midX: seg.midX, midY: seg.midY };
        if (requireEndpointClearance && !labelClearsTerminalEndpoints(seg, anchor)) {
          continue;
        }
        const placement = placementForAnchor(anchor);
        if (placement && !labelOverlapsOwnMarker(placement.rect, pts) && !overlapsPlacedLabel(labelId, placement.rect) && !labelOverlapsForeignNode(labelId, placement.rect) && (allowForeignEdgeOverlap || !labelOverlapsForeignEdge(edge.id, placement.rect))) {
          return { laneId: placement.laneId, anchor: placement.anchor };
        }
      }
      return void 0;
    }, "findLaneContainingFallback");
    const chosen = tryPool(poolBase) ?? (poolBase.length < segments.length ? tryPool(segments) : void 0) ?? findLaneContainingFallback(segments, true) ?? findLaneContainingFallback(segments, false) ?? findLaneContainingFallback(segments, false, true);
    if (chosen) {
      labelNode.x = chosen.anchor.midX;
      labelNode.y = chosen.anchor.midY;
      labelNode.parentId = chosen.laneId;
      const chosenRect = rectFromCenterSize(chosen.anchor.midX, chosen.anchor.midY, lw, lh);
      const priorIdx = placedLabelRects.findIndex((placed) => placed.labelId === labelId);
      if (priorIdx >= 0) {
        placedLabelRects[priorIdx] = { labelId, rect: chosenRect };
      } else {
        placedLabelRects.push({ labelId, rect: chosenRect });
      }
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(anchorLabelsToPolyline, "anchorLabelsToPolyline");

// src/rendering-util/layout-algorithms/swimlanes/direction/siblingSharedFaceRouting.ts
var EPS6 = 1e-6;
var MIN_PORT_SPACING2 = 8;
var PORT_SHIFT2 = MIN_PORT_SPACING2 / 2;
var LABEL_CLEARANCE_BUFFER = 3;
function pairKey(a, b) {
  return a < b ? `${a}::${b}` : `${b}::${a}`;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(pairKey, "pairKey");
function straightenCollinearSiblingDetours(edges, nodes) {
  const { nodeInfoById, realNodeRects } = collectRealNodeBounds(nodes);
  const labelDimById = /* @__PURE__ */ new Map();
  for (const n of nodes) {
    const id = n.id;
    if (n.isGroup) {
      continue;
    }
    if (n.isEdgeLabel) {
      labelDimById.set(id, {
        w: n.width ?? 0,
        h: n.height ?? 0
      });
      continue;
    }
  }
  const labelClearanceFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((thisEdge, thisSrcId, thisDstId, axis) => {
    const targetPair = pairKey(thisSrcId, thisDstId);
    let maxHalf = 0;
    const consider = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((labelId) => {
      if (!labelId) {
        return;
      }
      const dim = labelDimById.get(labelId);
      if (!dim) {
        return;
      }
      const half = axis === "x" ? dim.w / 2 : dim.h / 2;
      if (half > maxHalf) {
        maxHalf = half;
      }
    }, "consider");
    consider(thisEdge.labelNodeId);
    for (const other of edges) {
      if (other === thisEdge) {
        continue;
      }
      if (other.isLayoutOnly) {
        continue;
      }
      const oSrc = other.start;
      const oDst = other.end;
      if (!oSrc || !oDst) {
        continue;
      }
      if (pairKey(oSrc, oDst) !== targetPair) {
        continue;
      }
      consider(other.labelNodeId);
    }
    return maxHalf > 0 ? maxHalf + LABEL_CLEARANCE_BUFFER : 0;
  }, "labelClearanceFor");
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const pts = edge.points;
    if (!classifyThreeSegmentRoute(pts, EPS6)) {
      continue;
    }
    const nodePair = getNodePairGeometry(edge, nodeInfoById, EPS6);
    if (!nodePair) {
      continue;
    }
    const { srcId, dstId, srcInfo, dstInfo, collinearX, collinearY } = nodePair;
    if (collinearX === collinearY) {
      continue;
    }
    let targetSrc;
    let targetDst;
    if (collinearX) {
      const dstBelow = dstInfo.cy > srcInfo.cy;
      targetSrc = { x: srcInfo.cx, y: dstBelow ? srcInfo.rect.bottom : srcInfo.rect.top };
      targetDst = { x: dstInfo.cx, y: dstBelow ? dstInfo.rect.top : dstInfo.rect.bottom };
    } else {
      const dstEast = dstInfo.cx > srcInfo.cx;
      targetSrc = { x: dstEast ? srcInfo.rect.right : srcInfo.rect.left, y: srcInfo.cy };
      targetDst = { x: dstEast ? dstInfo.rect.left : dstInfo.rect.right, y: dstInfo.cy };
    }
    if (segmentHitsAnyRect(targetSrc, targetDst, realNodeRects, [srcId, dstId], 1)) {
      continue;
    }
    const shiftAxis = collinearX ? "x" : "y";
    const labelShift = labelClearanceFor(edge, srcId, dstId, shiftAxis);
    const effectiveShift = labelShift > PORT_SHIFT2 ? labelShift : PORT_SHIFT2;
    const deltas = [0, effectiveShift, -effectiveShift];
    for (const delta of deltas) {
      const shiftedSrc = { ...targetSrc };
      const shiftedDst = { ...targetDst };
      if (collinearX) {
        shiftedSrc.x += delta;
        shiftedDst.x += delta;
        if (shiftedSrc.x <= srcInfo.rect.left || shiftedSrc.x >= srcInfo.rect.right) {
          continue;
        }
        if (shiftedDst.x <= dstInfo.rect.left || shiftedDst.x >= dstInfo.rect.right) {
          continue;
        }
      } else {
        shiftedSrc.y += delta;
        shiftedDst.y += delta;
        if (shiftedSrc.y <= srcInfo.rect.top || shiftedSrc.y >= srcInfo.rect.bottom) {
          continue;
        }
        if (shiftedDst.y <= dstInfo.rect.top || shiftedDst.y >= dstInfo.rect.bottom) {
          continue;
        }
      }
      if (segmentHitsAnyRect(shiftedSrc, shiftedDst, realNodeRects, [srcId, dstId], 1)) {
        continue;
      }
      if (segmentConflictsWithAnyEdge(shiftedSrc, shiftedDst, edges, edge, { epsilon: EPS6 })) {
        continue;
      }
      edge.points = [shiftedSrc, shiftedDst];
      break;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(straightenCollinearSiblingDetours, "straightenCollinearSiblingDetours");

// src/rendering-util/layout-algorithms/swimlanes/direction/sharedTrackNudging.ts
function nudgeSharedInteriorSubpaths(edges, nodeByIdMap) {
  const EPS_LOCAL2 = 1e-3;
  const MIN_SHARED3 = 8;
  const TRACK_SHIFT = 7;
  const MIN_TRACK_GAP = TRACK_SHIFT;
  const SOURCE_DETOUR_STUB = 20;
  const BUFFER = 2;
  const MAX_ITERATIONS = 12;
  const { realNodeRects, labelNodeRects: labelRects } = collectNodeRectEntries(
    nodeByIdMap.values()
  );
  const segmentsFor2 = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, points) => {
    return orthogonalSegmentsForPoints(points, EPS_LOCAL2).map((segment) => ({
      ...segment,
      edge,
      interior: segment.index >= 1 && segment.index <= points.length - 3
    }));
  }, "segmentsFor");
  const allSegments = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    const result = [];
    for (const edge of edges) {
      if (edge.isLayoutOnly) {
        continue;
      }
      const points = edge.points;
      if (!points || points.length < 2) {
        continue;
      }
      result.push(...segmentsFor2(edge, dedupeConsecutivePoints(points)));
    }
    return result;
  }, "allSegments");
  const hasCrowdedParallelTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((a, b) => {
    if (a.horizontal && b.horizontal) {
      return overlapLength(a.a.x, a.b.x, b.a.x, b.b.x) >= MIN_SHARED3 && Math.abs(a.a.y - b.a.y) < MIN_TRACK_GAP;
    }
    if (a.vertical && b.vertical) {
      return overlapLength(a.a.y, a.b.y, b.a.y, b.b.y) >= MIN_SHARED3 && Math.abs(a.a.x - b.a.x) < MIN_TRACK_GAP;
    }
    return false;
  }, "hasCrowdedParallelTrack");
  const candidateIsSafe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edge, candidate) => {
    const sourceId = edge.start;
    const targetId = edge.end;
    const candidateSegments = segmentsFor2(edge, candidate);
    if (candidateSegments.length !== candidate.length - 1) {
      return false;
    }
    const endpointIds = [sourceId, targetId].filter((id) => Boolean(id));
    const ownLabelIds = edge.labelNodeId ? [edge.labelNodeId] : [];
    for (const segment of candidateSegments) {
      if (segmentHitsAnyRect(segment.a, segment.b, realNodeRects, endpointIds, -BUFFER)) {
        return false;
      }
      if (segmentHitsAnyRect(segment.a, segment.b, labelRects, ownLabelIds, -BUFFER)) {
        return false;
      }
    }
    for (const other of edges) {
      if (other === edge || other.isLayoutOnly) {
        continue;
      }
      const otherPoints = other.points;
      if (!otherPoints || otherPoints.length < 2) {
        continue;
      }
      for (const candidateSegment of candidateSegments) {
        for (const otherSegment of segmentsFor2(other, dedupeConsecutivePoints(otherPoints))) {
          if (hasCrowdedParallelTrack(candidateSegment, otherSegment)) {
            return false;
          }
          if (orthogonalSegmentsStrictlyCross(
            candidateSegment.a,
            candidateSegment.b,
            otherSegment.a,
            otherSegment.b,
            EPS_LOCAL2
          )) {
            return false;
          }
        }
      }
    }
    return true;
  }, "candidateIsSafe");
  const shiftedCandidate = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, shift) => {
    const points = dedupeConsecutivePoints(segment.edge.points ?? []);
    if (points.length < 4 || segment.index >= points.length - 1) {
      return void 0;
    }
    const candidate = points.map((p) => ({ ...p }));
    if (segment.horizontal) {
      candidate[segment.index].y += shift;
      candidate[segment.index + 1].y += shift;
    } else if (segment.vertical) {
      candidate[segment.index].x += shift;
      candidate[segment.index + 1].x += shift;
    } else {
      return void 0;
    }
    return segmentsFor2(segment.edge, candidate).length === candidate.length - 1 ? candidate : void 0;
  }, "shiftedCandidate");
  const nodeCenter = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, rect) => ({
    x: node.x ?? (rect.left + rect.right) / 2,
    y: node.y ?? (rect.top + rect.bottom) / 2
  }), "nodeCenter");
  const sourceDetourContextFor = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment) => {
    const edge = segment.edge;
    const points = dedupeConsecutivePoints(edge.points ?? []);
    if (points.length !== 4 || segment.index !== 1) {
      return void 0;
    }
    const sourceNode = edge.start ? nodeByIdMap.get(edge.start) : void 0;
    const targetNode = edge.end ? nodeByIdMap.get(edge.end) : void 0;
    const sourceRect = sourceNode ? rectOfNodeBounds(sourceNode) : void 0;
    const targetRect = targetNode ? rectOfNodeBounds(targetNode) : void 0;
    const tail = points.slice(segment.index + 2);
    if (!sourceNode || !targetNode || !sourceRect || !targetRect || tail.length === 0) {
      return void 0;
    }
    return {
      sourceCenter: nodeCenter(sourceNode, sourceRect),
      targetCenter: nodeCenter(targetNode, targetRect),
      sourceRect,
      tail
    };
  }, "sourceDetourContextFor");
  const verticalSourceDetour = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, shift, sourceCenter, targetCenter, sourceRect, tail) => {
    const targetBelow = targetCenter.y >= sourceCenter.y;
    const sourcePortY = targetBelow ? sourceRect.bottom : sourceRect.top;
    const stubY = sourcePortY + (targetBelow ? SOURCE_DETOUR_STUB : -SOURCE_DETOUR_STUB);
    if (targetBelow && segment.b.y <= stubY + EPS_LOCAL2 || !targetBelow && segment.b.y >= stubY - EPS_LOCAL2) {
      return void 0;
    }
    const railX = segment.a.x + shift;
    return dedupeConsecutivePoints(
      [
        { x: sourceCenter.x, y: sourcePortY },
        { x: sourceCenter.x, y: stubY },
        { x: railX, y: stubY },
        { x: railX, y: segment.b.y },
        ...tail
      ],
      EPS_LOCAL2
    );
  }, "verticalSourceDetour");
  const horizontalSourceDetour = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, shift, sourceCenter, targetCenter, sourceRect, tail) => {
    const targetRight = targetCenter.x >= sourceCenter.x;
    const sourcePortX = targetRight ? sourceRect.right : sourceRect.left;
    const stubX = sourcePortX + (targetRight ? SOURCE_DETOUR_STUB : -SOURCE_DETOUR_STUB);
    if (targetRight && segment.b.x <= stubX + EPS_LOCAL2 || !targetRight && segment.b.x >= stubX - EPS_LOCAL2) {
      return void 0;
    }
    const railY = segment.a.y + shift;
    return dedupeConsecutivePoints(
      [
        { x: sourcePortX, y: sourceCenter.y },
        { x: stubX, y: sourceCenter.y },
        { x: stubX, y: railY },
        { x: segment.b.x, y: railY },
        ...tail
      ],
      EPS_LOCAL2
    );
  }, "horizontalSourceDetour");
  const sourceDetourCandidate = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segment, shift) => {
    const context = sourceDetourContextFor(segment);
    if (!context) {
      return void 0;
    }
    if (segment.vertical) {
      return verticalSourceDetour(
        segment,
        shift,
        context.sourceCenter,
        context.targetCenter,
        context.sourceRect,
        context.tail
      );
    }
    if (segment.horizontal) {
      return horizontalSourceDetour(
        segment,
        shift,
        context.sourceCenter,
        context.targetCenter,
        context.sourceRect,
        context.tail
      );
    }
    return void 0;
  }, "sourceDetourCandidate");
  const shifts = [
    -TRACK_SHIFT,
    TRACK_SHIFT,
    -2 * TRACK_SHIFT,
    2 * TRACK_SHIFT,
    -3 * TRACK_SHIFT,
    3 * TRACK_SHIFT
  ];
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const segments = allSegments();
    let fixed = false;
    for (let i = 0; i < segments.length && !fixed; i++) {
      for (let j = i + 1; j < segments.length && !fixed; j++) {
        const first = segments[i];
        const second = segments[j];
        if (first.edge === second.edge || !hasCrowdedParallelTrack(first, second)) {
          continue;
        }
        const candidates = [first, second].filter((segment) => segment.interior);
        for (const segment of candidates) {
          for (const shift of shifts) {
            const direct = shiftedCandidate(segment, shift);
            if (direct && candidateIsSafe(segment.edge, direct)) {
              segment.edge.points = direct;
              fixed = true;
              break;
            }
            const detoured = sourceDetourCandidate(segment, shift);
            if (detoured && candidateIsSafe(segment.edge, detoured)) {
              segment.edge.points = detoured;
              fixed = true;
              break;
            }
          }
          if (fixed) {
            break;
          }
        }
      }
    }
    if (!fixed) {
      return;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(nudgeSharedInteriorSubpaths, "nudgeSharedInteriorSubpaths");

// src/rendering-util/layout-algorithms/swimlanes/direction/validation.ts
function segmentsIntersect(p1, p2, p3, p4) {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;
  const cross = d1x * d2y - d1y * d2x;
  if (Math.abs(cross) < 1e-10) {
    return false;
  }
  const dx = p3.x - p1.x;
  const dy = p3.y - p1.y;
  const t = (dx * d2y - dy * d2x) / cross;
  const u = (dx * d1y - dy * d1x) / cross;
  const eps = 0.01;
  return t > eps && t < 1 - eps && u > eps && u < 1 - eps;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(segmentsIntersect, "segmentsIntersect");
function validateSwimlanesLayout(layout) {
  const nodes = layout.nodes ?? [];
  const edges = layout.edges ?? [];
  const issues = [];
  if (!edges.length || !nodes.length) {
    return issues;
  }
  const nodeRects = collectLayoutNodeRects(nodes);
  const epsilon = 1;
  const edgeSegments = [];
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const points = edge.points;
    if (!points || points.length < 2) {
      continue;
    }
    const edgeStart = edge.start;
    const edgeEnd = edge.end;
    const ownLabelId = edge.labelNodeId;
    const edgeId = edge.id ?? `${edgeStart}->${edgeEnd}`;
    for (const rect of nodeRects) {
      if (rect.nodeId === edgeStart || rect.nodeId === edgeEnd) {
        continue;
      }
      if (ownLabelId && rect.nodeId === ownLabelId) {
        continue;
      }
      for (let i = 0; i < points.length - 1; i++) {
        if (segmentBoundsOverlapRect(points[i], points[i + 1], rect, -epsilon)) {
          issues.push({
            type: "edge-node-overlap",
            edgeId,
            targetId: rect.nodeId,
            detail: `segment ${i} passes through node "${rect.nodeId}"`
          });
          break;
        }
      }
    }
    for (let i = 0; i < points.length - 1; i++) {
      edgeSegments.push({
        edgeId,
        start: edgeStart,
        end: edgeEnd,
        p1: points[i],
        p2: points[i + 1]
      });
    }
  }
  const crossingPairs = /* @__PURE__ */ new Set();
  for (let i = 0; i < edgeSegments.length; i++) {
    for (let j = i + 1; j < edgeSegments.length; j++) {
      const a = edgeSegments[i];
      const b = edgeSegments[j];
      if (a.edgeId === b.edgeId) {
        continue;
      }
      if (a.start === b.start || a.start === b.end || a.end === b.start || a.end === b.end) {
        continue;
      }
      if (segmentsIntersect(a.p1, a.p2, b.p1, b.p2)) {
        const pairKey2 = a.edgeId < b.edgeId ? `${a.edgeId}|${b.edgeId}` : `${b.edgeId}|${a.edgeId}`;
        if (!crossingPairs.has(pairKey2)) {
          crossingPairs.add(pairKey2);
          issues.push({
            type: "edge-edge-crossing",
            edgeId: a.edgeId,
            targetId: b.edgeId,
            detail: `edges "${a.edgeId}" and "${b.edgeId}" cross`
          });
        }
      }
    }
  }
  if (issues.length > 0) {
    const overlaps = issues.filter((i) => i.type === "edge-node-overlap").length;
    const crossings = issues.filter((i) => i.type === "edge-edge-crossing").length;
    _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_10__/* .log */ .R.warn(
      `[SWIMLANE_VALIDATE] ${issues.length} issue(s) detected: ${overlaps} edge-node overlap(s), ${crossings} edge crossing(s)`
    );
    for (const issue of issues) {
      _chunk_X3CZISLH_mjs__WEBPACK_IMPORTED_MODULE_10__/* .log */ .R.warn(`[SWIMLANE_VALIDATE]   ${issue.type}: ${issue.detail}`);
    }
  }
  return issues;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(validateSwimlanesLayout, "validateSwimlanesLayout");

// src/rendering-util/layout-algorithms/swimlanes/postProcessing.ts
function postProcessSwimlaneLayout(layout, direction) {
  const nodes = layout.nodes ?? [];
  const edges = layout.edges ?? [];
  const contentNodes = nodes.filter((n) => !n.isGroup);
  if ((direction === "LR" || direction === "RL") && contentNodes.length > 0 && !applyLrDirectionTransform(layout, direction)) {
    return;
  }
  if (direction === "BT" && contentNodes.length > 0 && !applyBtDirectionTransform(layout)) {
    return;
  }
  for (const edge of edges) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const pts = edge.points;
    if (!pts || pts.length < 2) {
      continue;
    }
    edge.points = simplifyPolyline(
      orthogonalizePolyline(pts)
    );
  }
  simplifyDetouredEdges(edges, nodes);
  straightenCollinearSiblingDetours(edges, nodes);
  portSwapToLShape(edges, nodes);
  const nodeByIdMap = /* @__PURE__ */ new Map();
  for (const n of nodes) {
    nodeByIdMap.set(String(n.id), n);
  }
  anchorLabelsToPolyline(edges, nodeByIdMap);
  clipEdgeEndpointsToNodeBoundaries(edges, nodeByIdMap);
  collapseShortTerminalStub(edges, nodeByIdMap);
  nudgeSharedInteriorSubpaths(edges, nodeByIdMap);
  separateSharedRenderedTerminalLanes(edges, nodeByIdMap);
  collapseRedundantRectangularDoglegs(edges, nodeByIdMap);
  liftObstacleHuggingSameSideRails(edges, nodeByIdMap);
  swapDestinationTerminalTailsToReduceCrossings(edges, nodeByIdMap);
  const finalizeRenderedEdges = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    resolveRenderedOrthogonalCrossings(edges, nodeByIdMap);
    reassignCrossingExternalRailChannels(edges, nodeByIdMap);
    shortcutRedundantOrthogonalJogs(edges, nodeByIdMap);
    anchorLabelsToPolyline(edges, nodeByIdMap);
    prepareEdgeEndpointsForRenderer(edges, nodeByIdMap);
    liftObstacleHuggingSameSideRails(edges, nodeByIdMap);
    anchorLabelsToPolyline(edges, nodeByIdMap);
    prepareEdgeEndpointsForRenderer(edges, nodeByIdMap);
  }, "finalizeRenderedEdges");
  finalizeRenderedEdges();
  nudgeSharedInteriorSubpaths(edges, nodeByIdMap);
  finalizeRenderedEdges();
  liftTopLaneTitleBandsAboveRails(edges, nodeByIdMap);
  shiftLeftLaneTitleBandsLeftOfRails(edges, nodeByIdMap);
  liftTopLaneTitleBandsAboveRails(edges, nodeByIdMap);
  shiftLeftLaneTitleBandsLeftOfRails(edges, nodeByIdMap);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(postProcessSwimlaneLayout, "postProcessSwimlaneLayout");

// src/rendering-util/layout-algorithms/swimlanes/phase0.helpers.ts
function normalizeGraph(g) {
  const nodeById = new Map(g.nodeById);
  const seen = /* @__PURE__ */ new Set();
  const edges = [];
  for (const e of g.edges) {
    if (!nodeById.has(e.src) || !nodeById.has(e.dst)) {
      continue;
    }
    const key = `${e.id}:${e.src}->${e.dst}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    edges.push(e);
  }
  const nodes = [...nodeById.keys()];
  return { nodes, edges, layout: g.layout, nodeById };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(normalizeGraph, "normalizeGraph");
function incoming(g, v) {
  return g.edges.filter((e) => e.dst === v);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(incoming, "incoming");
function buildSuccessorMap(g) {
  const succs = /* @__PURE__ */ new Map();
  for (const v of g.nodes) {
    succs.set(v, []);
  }
  for (const e of g.edges) {
    succs.get(e.src).push(e.dst);
  }
  return succs;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildSuccessorMap, "buildSuccessorMap");
function buildSortedSuccessorMap(g) {
  const succs = buildSuccessorMap(g);
  for (const successors of succs.values()) {
    successors.sort((a, b) => a.localeCompare(b));
  }
  return succs;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildSortedSuccessorMap, "buildSortedSuccessorMap");
function buildInDegreeMap(g) {
  const indeg = /* @__PURE__ */ new Map();
  for (const v of g.nodes) {
    indeg.set(v, 0);
  }
  for (const e of g.edges) {
    indeg.set(e.dst, (indeg.get(e.dst) ?? 0) + 1);
  }
  return indeg;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildInDegreeMap, "buildInDegreeMap");
function sortedZeroInDegreeNodes(indeg) {
  return [...indeg.entries()].filter(([, degree]) => degree === 0).map(([id]) => id).sort((a, b) => a.localeCompare(b));
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sortedZeroInDegreeNodes, "sortedZeroInDegreeNodes");
function buildPredecessorSuccessorMaps(g, includeEdge = () => true) {
  const preds = /* @__PURE__ */ new Map();
  const succs = /* @__PURE__ */ new Map();
  for (const v of g.nodes) {
    preds.set(v, []);
    succs.set(v, []);
  }
  for (const e of g.edges) {
    if (!includeEdge(e)) {
      continue;
    }
    succs.get(e.src).push(e.dst);
    preds.get(e.dst).push(e.src);
  }
  return { preds, succs };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildPredecessorSuccessorMaps, "buildPredecessorSuccessorMaps");
function buildLayersFromRanks(g, order, rankOf, opts) {
  let maxRank = 0;
  for (const v of g.nodes) {
    if (opts?.skipGroups && g.nodeById.get(v)?.isGroup) {
      continue;
    }
    maxRank = Math.max(maxRank, rankOf[v] ?? 0);
  }
  const layers = Array.from({ length: maxRank + 1 }, () => []);
  for (const v of order) {
    if (opts?.skipGroups && g.nodeById.get(v)?.isGroup) {
      continue;
    }
    layers[Math.max(0, rankOf[v] ?? 0)].push(v);
  }
  return layers;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildLayersFromRanks, "buildLayersFromRanks");
function topoSortIfAcyclic(g) {
  const indeg = buildInDegreeMap(g);
  const queue = sortedZeroInDegreeNodes(indeg);
  const order = [];
  const adj = buildSortedSuccessorMap(g);
  while (queue.length) {
    const u = queue.shift();
    order.push(u);
    for (const v of adj.get(u) ?? []) {
      indeg.set(v, (indeg.get(v) ?? 0) - 1);
      if ((indeg.get(v) ?? 0) === 0) {
        let i = 0;
        while (i < queue.length && queue[i] < v) {
          i++;
        }
        queue.splice(i, 0, v);
      }
    }
  }
  return order.length === g.nodes.length ? order : null;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(topoSortIfAcyclic, "topoSortIfAcyclic");
function buildLayerIndex(layer) {
  const m = /* @__PURE__ */ new Map();
  let index = 0;
  for (const id of layer) {
    m.set(id, index);
    index++;
  }
  return m;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildLayerIndex, "buildLayerIndex");
function countInversions(values) {
  const tmp = new Array(values.length);
  const count = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((left, right) => {
    if (right - left <= 1) {
      return 0;
    }
    const mid = left + right >> 1;
    let inversions = count(left, mid) + count(mid, right);
    let i = left;
    let j = mid;
    let k = left;
    while (i < mid || j < right) {
      if (j >= right || i < mid && values[i] <= values[j]) {
        tmp[k++] = values[i++];
      } else {
        tmp[k++] = values[j++];
        inversions += mid - i;
      }
    }
    for (let t = left; t < right; t++) {
      values[t] = tmp[t];
    }
    return inversions;
  }, "count");
  return count(0, values.length);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(countInversions, "countInversions");

// src/rendering-util/layout-algorithms/swimlanes/phase1.cycles.ts
function removeCycles_DFS(g) {
  const gn = normalizeGraph(g);
  const adj = /* @__PURE__ */ new Map();
  for (const v of gn.nodes) {
    adj.set(v, []);
  }
  for (const e of gn.edges) {
    adj.get(e.src).push(e);
  }
  for (const arr of adj.values()) {
    arr.sort((a, b) => a.dst === b.dst ? a.id.localeCompare(b.id) : a.dst.localeCompare(b.dst));
  }
  const color = /* @__PURE__ */ Object.create(null);
  for (const v of gn.nodes) {
    color[v] = 0;
  }
  const reversed = [];
  const dfs = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((u) => {
    color[u] = 1;
    for (const e of adj.get(u) ?? []) {
      const v = e.dst;
      if (color[v] === 0) {
        dfs(v);
      } else if (color[v] === 1) {
        reversed.push(e);
      }
    }
    color[u] = 2;
  }, "dfs");
  const nodesSorted = [...gn.nodes].sort((a, b) => a.localeCompare(b));
  for (const v of nodesSorted) {
    if (color[v] === 0) {
      dfs(v);
    }
  }
  const toReverse = new Set(reversed.map((e) => `${e.id}:${e.src}->${e.dst}`));
  const acycEdges = gn.edges.map(
    (e) => toReverse.has(`${e.id}:${e.src}->${e.dst}`) ? { id: e.id, src: e.dst, dst: e.src, weight: e.weight, ref: e.ref } : e
  );
  const acyclic = {
    nodes: [...gn.nodes],
    edges: acycEdges,
    layout: gn.layout,
    nodeById: new Map(gn.nodeById)
  };
  return { acyclic, reversed };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(removeCycles_DFS, "removeCycles_DFS");

// src/rendering-util/layout-algorithms/swimlanes/phase2.options.ts
function buildTopLaneMap(g) {
  const cache = /* @__PURE__ */ new Map();
  const resolve = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((id) => {
    if (cache.has(id)) {
      return cache.get(id);
    }
    const node = g.nodeById.get(id);
    if (!node) {
      cache.set(id, null);
      return null;
    }
    const parentId = node.parentId;
    if (!parentId) {
      cache.set(id, null);
      return null;
    }
    const parentLane = resolve(parentId);
    const lane = parentLane ?? parentId;
    cache.set(id, lane);
    return lane;
  }, "resolve");
  for (const id of g.nodes) {
    resolve(id);
  }
  return cache;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildTopLaneMap, "buildTopLaneMap");
function createTopLaneResolver(g) {
  const topLaneMap = buildTopLaneMap(g);
  return (id) => topLaneMap.get(id) ?? null;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(createTopLaneResolver, "createTopLaneResolver");
function buildTopLaneOrder(g) {
  const lanes = [];
  for (const node of g.layout.nodes ?? []) {
    if (node.isGroup && !node.parentId) {
      lanes.push(node.id);
    }
  }
  return [...new Set(lanes)].reverse();
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildTopLaneOrder, "buildTopLaneOrder");
function resolveTopLaneOrder(g, preferredOrder) {
  const sourceOrder = buildTopLaneOrder(g);
  if (!preferredOrder || preferredOrder.length === 0) {
    return sourceOrder;
  }
  const sourceLaneIds = new Set(sourceOrder);
  const seen = /* @__PURE__ */ new Set();
  const resolved = [];
  for (const laneId of preferredOrder) {
    if (!sourceLaneIds.has(laneId) || seen.has(laneId)) {
      continue;
    }
    seen.add(laneId);
    resolved.push(laneId);
  }
  for (const laneId of sourceOrder) {
    if (seen.has(laneId)) {
      continue;
    }
    resolved.push(laneId);
  }
  return resolved;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(resolveTopLaneOrder, "resolveTopLaneOrder");

// src/rendering-util/layout-algorithms/swimlanes/config.ts
var PRECISION = {
  /** Epsilon for floating-point comparisons */
  EPSILON: 1e-6
};
var LAYERING = {
  /** Default number of iterations for gravity-based layering */
  GRAVITY_ITERATIONS: 8,
  /** Maximum number of passes for crossing-based rank optimization */
  MAX_CROSSING_OPTIMIZATION_PASSES: 4,
  /** Whether to compact single-input nodes by default */
  DEFAULT_COMPACT_SINGLE_INPUT: true
};
var COORDINATES = {
  /** Default vertical gap between layers (px) */
  DEFAULT_LAYER_GAP: 100,
  /** Default horizontal gap between nodes (px) */
  DEFAULT_NODE_GAP: 40
};

// src/rendering-util/layout-algorithms/swimlanes/driving-tree.ts
function buildDrivingTree(graph, opts) {
  const g = normalizeGraph(graph);
  const laneOf = opts?.laneOf ?? (() => null);
  const rankHint = opts?.rankHint;
  const { preds } = buildPredecessorSuccessorMaps(g);
  for (const arr of preds.values()) {
    arr.sort((a, b) => a.localeCompare(b));
  }
  const topoOrder = topoSortIfAcyclic(g) ?? [...g.nodes].sort((a, b) => a.localeCompare(b));
  const topoIndex = /* @__PURE__ */ new Map();
  for (const [idx, id] of topoOrder.entries()) {
    topoIndex.set(id, idx);
  }
  const parent = /* @__PURE__ */ new Map();
  const children = /* @__PURE__ */ new Map();
  for (const node of g.nodes) {
    children.set(node, []);
  }
  for (const node of topoOrder) {
    const candidates = (preds.get(node) ?? []).filter((p) => parent.has(p));
    if (candidates.length > 0) {
      const chosen = chooseParent(node, candidates, {
        laneOf,
        rankHint,
        topoIndex
      });
      parent.set(node, chosen);
      children.get(chosen).push(node);
    } else if (!parent.has(node)) {
      parent.set(node, null);
    }
  }
  for (const node of g.nodes) {
    if (!parent.has(node)) {
      parent.set(node, null);
    }
  }
  const rootSet = /* @__PURE__ */ new Set();
  for (const node of g.nodes) {
    if ((parent.get(node) ?? null) === null) {
      rootSet.add(node);
    }
  }
  const roots = [...rootSet].sort((a, b) => {
    const ta = topoIndex.get(a) ?? 0;
    const tb = topoIndex.get(b) ?? 0;
    if (ta === tb) {
      return a.localeCompare(b);
    }
    return ta - tb;
  });
  const adjacency = buildAdjacency(g);
  const adjacencyList = /* @__PURE__ */ new Map();
  for (const [node, set] of adjacency.entries()) {
    adjacencyList.set(
      node,
      [...set].sort((a, b) => a.localeCompare(b))
    );
  }
  const componentOf = assignComponents(adjacencyList);
  const blocks = computeBlocks(adjacencyList);
  const nodeBlocks = /* @__PURE__ */ new Map();
  for (const node of g.nodes) {
    nodeBlocks.set(node, []);
  }
  for (const block of blocks) {
    for (const node of block.nodes) {
      const list = nodeBlocks.get(node);
      if (list) {
        list.push(block.id);
      } else {
        nodeBlocks.set(node, [block.id]);
      }
    }
  }
  const preorder = [];
  const postorder = [];
  const seen = /* @__PURE__ */ new Set();
  const walk = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    if (seen.has(node)) {
      return;
    }
    seen.add(node);
    preorder.push(node);
    for (const child of children.get(node) ?? []) {
      walk(child);
    }
    postorder.push(node);
  }, "walk");
  for (const root of roots) {
    walk(root);
  }
  for (const node of topoOrder) {
    walk(node);
  }
  return {
    parent,
    children,
    roots,
    componentOf,
    blocks,
    nodeBlocks,
    adjacency: adjacencyList,
    preorder,
    postorder,
    topologicalOrder: topoOrder
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildDrivingTree, "buildDrivingTree");
function chooseParent(node, candidates, ctx) {
  const laneNode = ctx.laneOf(node);
  const sorted = [...candidates].sort((a, b) => {
    const laneA = ctx.laneOf(a);
    const laneB = ctx.laneOf(b);
    const sameLaneA = laneA != null && laneA === laneNode;
    const sameLaneB = laneB != null && laneB === laneNode;
    if (sameLaneA !== sameLaneB) {
      return sameLaneA ? -1 : 1;
    }
    const rankA = ctx.rankHint?.[a];
    const rankB = ctx.rankHint?.[b];
    if (rankA != null && rankB != null && rankA !== rankB) {
      return rankB - rankA;
    }
    const idxA = ctx.topoIndex.get(a) ?? 0;
    const idxB = ctx.topoIndex.get(b) ?? 0;
    if (idxA !== idxB) {
      return idxA - idxB;
    }
    return a.localeCompare(b);
  });
  return sorted[0];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(chooseParent, "chooseParent");
function buildAdjacency(g) {
  const adjacency = /* @__PURE__ */ new Map();
  for (const node of g.nodes) {
    adjacency.set(node, /* @__PURE__ */ new Set());
  }
  for (const e of g.edges) {
    adjacency.get(e.src).add(e.dst);
    adjacency.get(e.dst).add(e.src);
  }
  return adjacency;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildAdjacency, "buildAdjacency");
function assignComponents(adjacency) {
  const componentOf = /* @__PURE__ */ new Map();
  let componentId = 0;
  for (const node of adjacency.keys()) {
    if (componentOf.has(node)) {
      continue;
    }
    const stack = [node];
    while (stack.length > 0) {
      const cur = stack.pop();
      if (componentOf.has(cur)) {
        continue;
      }
      componentOf.set(cur, componentId);
      for (const next of adjacency.get(cur) ?? []) {
        if (!componentOf.has(next)) {
          stack.push(next);
        }
      }
    }
    componentId++;
  }
  return componentOf;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignComponents, "assignComponents");
function computeBlocks(adjacency) {
  const discovery = /* @__PURE__ */ new Map();
  const low = /* @__PURE__ */ new Map();
  const edgeStack = [];
  const blocks = [];
  let time = 0;
  const visit = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, parent) => {
    discovery.set(node, ++time);
    low.set(node, time);
    for (const next of adjacency.get(node) ?? []) {
      if (next === parent) {
        continue;
      }
      if (!discovery.has(next)) {
        edgeStack.push([node, next]);
        visit(next, node);
        low.set(node, Math.min(low.get(node) ?? time, low.get(next) ?? time));
        if ((low.get(next) ?? 0) >= (discovery.get(node) ?? 0)) {
          blocks.push(popBlock(node, next, edgeStack, blocks.length));
        }
      } else if ((discovery.get(next) ?? 0) < (discovery.get(node) ?? 0)) {
        edgeStack.push([node, next]);
        low.set(node, Math.min(low.get(node) ?? time, discovery.get(next) ?? time));
      }
    }
  }, "visit");
  for (const node of adjacency.keys()) {
    if (!discovery.has(node)) {
      visit(node, null);
    }
  }
  return blocks;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(computeBlocks, "computeBlocks");
function popBlock(u, v, stack, id) {
  const edges = [];
  const nodes = /* @__PURE__ */ new Set();
  while (stack.length > 0) {
    const edge = stack.pop();
    edges.push(edge);
    nodes.add(edge[0]);
    nodes.add(edge[1]);
    if (edge[0] === u && edge[1] === v || edge[0] === v && edge[1] === u) {
      break;
    }
  }
  return { id, edges, nodes: [...nodes] };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(popBlock, "popBlock");

// src/rendering-util/layout-algorithms/swimlanes/phase2.crossCounts.ts
function computeSubtreeCrossCounts(g, rankOf, tree) {
  const nodes = [...g.nodes];
  const indexOf = /* @__PURE__ */ new Map();
  for (const [i, node] of nodes.entries()) {
    indexOf.set(node, i);
  }
  const n = nodes.length;
  const parentIdx = new Array(n).fill(-1);
  const depth = new Array(n).fill(0);
  const queue = [];
  const seen = /* @__PURE__ */ new Set();
  for (const node of nodes) {
    const parentId = tree.parent.get(node) ?? null;
    const idx = indexOf.get(node);
    if (idx == null) {
      continue;
    }
    if (parentId == null) {
      parentIdx[idx] = -1;
      depth[idx] = 0;
      if (!seen.has(node)) {
        seen.add(node);
        queue.push(node);
      }
    }
  }
  while (queue.length > 0) {
    const current = queue.shift();
    const currentIdx = indexOf.get(current);
    if (currentIdx == null) {
      continue;
    }
    const childList = tree.children.get(current) ?? [];
    for (const child of childList) {
      if (seen.has(child)) {
        continue;
      }
      const childIdx = indexOf.get(child);
      if (childIdx == null) {
        continue;
      }
      parentIdx[childIdx] = currentIdx;
      depth[childIdx] = depth[currentIdx] + 1;
      seen.add(child);
      queue.push(child);
    }
  }
  for (const node of nodes) {
    if (seen.has(node)) {
      continue;
    }
    const idx = indexOf.get(node);
    if (idx == null) {
      continue;
    }
    parentIdx[idx] = -1;
    depth[idx] = 0;
    seen.add(node);
  }
  const maxLog = Math.max(1, Math.ceil(Math.log2(Math.max(1, n))) + 1);
  const up = Array.from({ length: maxLog }, () => new Array(n).fill(-1));
  for (let i = 0; i < n; i++) {
    up[0][i] = parentIdx[i];
  }
  for (let k = 1; k < maxLog; k++) {
    for (let i = 0; i < n; i++) {
      const prev = up[k - 1][i];
      up[k][i] = prev === -1 ? -1 : up[k - 1][prev];
    }
  }
  const lcaIndex = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((aIdx, bIdx) => {
    if (aIdx === -1 || bIdx === -1) {
      return -1;
    }
    if (depth[aIdx] < depth[bIdx]) {
      [aIdx, bIdx] = [bIdx, aIdx];
    }
    const diff = depth[aIdx] - depth[bIdx];
    for (let k = 0; k < maxLog; k++) {
      if (diff >> k & 1) {
        aIdx = up[k][aIdx];
        if (aIdx === -1) {
          return -1;
        }
      }
    }
    if (aIdx === bIdx) {
      return aIdx;
    }
    for (let k = maxLog - 1; k >= 0; k--) {
      const upA = up[k][aIdx];
      const upB = up[k][bIdx];
      if (upA === -1 || upB === -1) {
        continue;
      }
      if (upA !== upB) {
        aIdx = upA;
        bIdx = upB;
      }
    }
    return up[0][aIdx];
  }, "lcaIndex");
  const ownCounts = Array.from({ length: n }, () => /* @__PURE__ */ new Map());
  for (const edge of g.edges) {
    let src = edge.src;
    let dst = edge.dst;
    let ru = rankOf[src];
    let rv = rankOf[dst];
    if (ru == null || rv == null) {
      continue;
    }
    if (ru > rv) {
      [src, dst] = [dst, src];
      [ru, rv] = [rv, ru];
    }
    if (ru == null || rv == null || ru === rv) {
      continue;
    }
    const upperIdx = indexOf.get(src);
    const lowerIdx = indexOf.get(dst);
    if (upperIdx == null || lowerIdx == null) {
      continue;
    }
    const lca = lcaIndex(upperIdx, lowerIdx);
    if (lca === -1) {
      continue;
    }
    const bucket = ownCounts[lca];
    for (let layer = ru; layer < rv; layer++) {
      bucket.set(layer, (bucket.get(layer) ?? 0) + 1);
    }
  }
  const crossCounts = /* @__PURE__ */ new Map();
  const mergeInto = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((target, source) => {
    if (source.size === 0) {
      return;
    }
    for (const [layer, value] of source) {
      target.set(layer, (target.get(layer) ?? 0) + value);
    }
  }, "mergeInto");
  const visited = /* @__PURE__ */ new Set();
  const dfs = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    const idx = indexOf.get(node);
    visited.add(node);
    const base = idx == null ? void 0 : ownCounts[idx];
    const accumulator = base ? new Map(base) : /* @__PURE__ */ new Map();
    const childList = tree.children.get(node) ?? [];
    for (const child of childList) {
      const childMap = dfs(child);
      const parentLayer = rankOf[node];
      if (parentLayer != null) {
        let map = crossCounts.get(node);
        if (!map) {
          map = /* @__PURE__ */ new Map();
          crossCounts.set(node, map);
        }
        let value = childMap.get(parentLayer) ?? 0;
        const childLayer = rankOf[child];
        if (childLayer != null && childLayer > parentLayer) {
          value += 1;
        }
        map.set(child, value);
      }
      mergeInto(accumulator, childMap);
    }
    return accumulator;
  }, "dfs");
  for (const root of tree.roots) {
    if (!visited.has(root)) {
      dfs(root);
    }
  }
  for (const node of nodes) {
    if (!visited.has(node)) {
      dfs(node);
    }
  }
  return crossCounts;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(computeSubtreeCrossCounts, "computeSubtreeCrossCounts");

// src/rendering-util/layout-algorithms/swimlanes/phase2.multitree.core.ts
function annotateMinimumLayers(nodes, children, rankOf) {
  const minLayer = /* @__PURE__ */ new Map();
  const annotate = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    let minL = rankOf[node] ?? 0;
    const childList = [...children.get(node) ?? []];
    childList.sort(compareByRankThenId(rankOf));
    for (const child of childList) {
      annotate(child);
      const childMin = minLayer.get(child);
      if (childMin != null) {
        minL = Math.min(minL, childMin);
      }
    }
    minLayer.set(node, minL);
  }, "annotate");
  for (const node of nodes) {
    annotate(node);
  }
  return minLayer;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(annotateMinimumLayers, "annotateMinimumLayers");
function compareByRankThenId(rankOf) {
  return (a, b) => {
    const ra = rankOf[a] ?? 0;
    const rb = rankOf[b] ?? 0;
    return ra === rb ? a.localeCompare(b) : ra - rb;
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(compareByRankThenId, "compareByRankThenId");
function emitNodesInTreeOrder(roots, allNodes, rankOf, orderChildren) {
  let maxRank = 0;
  for (const node of allNodes) {
    const r = rankOf[node] ?? 0;
    if (r > maxRank) {
      maxRank = r;
    }
  }
  const layers = Array.from({ length: maxRank + 1 }, () => []);
  const emitted = /* @__PURE__ */ new Set();
  const emit = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    if (emitted.has(node)) {
      return;
    }
    emitted.add(node);
    const layer = rankOf[node] ?? 0;
    if (!layers[layer]) {
      layers[layer] = [];
    }
    layers[layer].push(node);
    for (const child of orderChildren(node)) {
      emit(child);
    }
  }, "emit");
  for (const root of roots) {
    emit(root);
  }
  for (const node of allNodes) {
    if (!emitted.has(node)) {
      const layer = rankOf[node] ?? 0;
      if (!layers[layer]) {
        layers[layer] = [];
      }
      layers[layer].push(node);
      emitted.add(node);
    }
  }
  return layers;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(emitNodesInTreeOrder, "emitNodesInTreeOrder");
function deduplicateLayers(layers) {
  const result = [];
  for (const layer of layers) {
    const seen = /* @__PURE__ */ new Set();
    const deduped = [];
    for (const id of layer) {
      if (seen.has(id)) {
        continue;
      }
      seen.add(id);
      deduped.push(id);
    }
    result.push(deduped);
  }
  return result;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(deduplicateLayers, "deduplicateLayers");

// src/rendering-util/layout-algorithms/swimlanes/phase2.multitree.order.ts
function createChildOrderer(children, rankOf, crossCounts, minLayer) {
  return (node) => {
    const raw = children.get(node) ?? [];
    if (raw.length === 0) {
      return [];
    }
    const layer = rankOf[node] ?? 0;
    const future = [];
    const present = [];
    const crossMap = crossCounts.get(node);
    for (const child of raw) {
      const minL = minLayer.get(child) ?? layer;
      if (minL > layer) {
        future.push({ child, min: minL });
      } else {
        present.push(child);
      }
    }
    future.sort((a, b) => {
      if (a.min === b.min) {
        return a.child.localeCompare(b.child);
      }
      return a.min - b.min;
    });
    present.sort((a, b) => {
      const ca = crossMap?.get(a) ?? 0;
      const cb = crossMap?.get(b) ?? 0;
      if (ca !== cb) {
        return ca - cb;
      }
      const ma = minLayer.get(a) ?? layer;
      const mb = minLayer.get(b) ?? layer;
      if (ma !== mb) {
        return ma - mb;
      }
      return a.localeCompare(b);
    });
    return [...future.map((item) => item.child), ...present];
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(createChildOrderer, "createChildOrderer");
function buildMultitreeLayerOrder(g, rankOf, laneOf) {
  const tree = buildDrivingTree(g, {
    rankHint: rankOf,
    laneOf
  });
  const { children, roots } = tree;
  for (const node of g.nodes) {
    if (!children.has(node)) {
      children.set(node, []);
    }
  }
  const crossCounts = computeSubtreeCrossCounts(g, rankOf, tree);
  const rootsSorted = [...roots].sort(compareByRankThenId(rankOf));
  const minLayer = annotateMinimumLayers(rootsSorted, children, rankOf);
  const orderChildren = createChildOrderer(children, rankOf, crossCounts, minLayer);
  let layers = emitNodesInTreeOrder(rootsSorted, g.nodes, rankOf, orderChildren);
  layers = deduplicateLayers(layers);
  return layers;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildMultitreeLayerOrder, "buildMultitreeLayerOrder");

// src/rendering-util/layout-algorithms/swimlanes/phase2.crossOptimization.ts
function countCrossingsBetweenAdjacent(upper, lower, edges) {
  const upperSet = new Set(upper);
  const lowerSet = new Set(lower);
  const li = buildLayerIndex(lower);
  const vs = [];
  for (const e of edges) {
    if (upperSet.has(e.src) && lowerSet.has(e.dst)) {
      vs.push(li.get(e.dst));
    }
  }
  return countInversions(vs);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(countCrossingsBetweenAdjacent, "countCrossingsBetweenAdjacent");
function totalCrossings(layers, edges, rankOf) {
  const expanded = [];
  for (const e of edges) {
    const ru = rankOf[e.src];
    const rv = rankOf[e.dst];
    if (ru == null || rv == null || ru === rv) {
      continue;
    }
    let upper = e.src;
    let lower = e.dst;
    let rUpper = ru;
    let rLower = rv;
    if (ru > rv) {
      upper = e.dst;
      lower = e.src;
      rUpper = rv;
      rLower = ru;
    }
    for (let L = rUpper; L < rLower; L++) {
      expanded.push({ id: `${e.id}@${L}`, src: upper, dst: lower, ref: e.ref });
    }
  }
  let sum = 0;
  for (let i = 0; i + 1 < layers.length; i++) {
    sum += countCrossingsBetweenAdjacent(layers[i], layers[i + 1], expanded);
  }
  return sum;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(totalCrossings, "totalCrossings");
function optimizeRanksByCrossings(g, initialRank) {
  const rankOf = { ...initialRank };
  const { preds } = buildPredecessorSuccessorMaps(g);
  const laneOf = createTopLaneResolver(g);
  const layers = buildMultitreeLayerOrder(g, rankOf, laneOf);
  let best = totalCrossings(layers, g.edges, rankOf);
  const maxPasses = LAYERING.MAX_CROSSING_OPTIMIZATION_PASSES;
  for (let pass = 0; pass < maxPasses; pass++) {
    let changed = false;
    const nodesByRank = [...g.nodes].sort((a, b) => (rankOf[b] ?? 0) - (rankOf[a] ?? 0));
    for (const v of nodesByRank) {
      const r = rankOf[v] ?? 0;
      if (r === 0) {
        continue;
      }
      let lb = 0;
      for (const u of preds.get(v) ?? []) {
        lb = Math.max(lb, (rankOf[u] ?? 0) + 1);
      }
      if (lb >= r) {
        continue;
      }
      const old = r;
      rankOf[v] = lb;
      const trialLayers = buildMultitreeLayerOrder(g, rankOf, laneOf);
      const score = totalCrossings(trialLayers, g.edges, rankOf);
      if (score < best) {
        best = score;
        changed = true;
      } else {
        rankOf[v] = old;
      }
    }
    if (!changed) {
      break;
    }
  }
  return rankOf;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(optimizeRanksByCrossings, "optimizeRanksByCrossings");

// src/rendering-util/layout-algorithms/swimlanes/phase2.crossLaneAdjust.ts
function adjustCrossLaneSources(g, rankOf) {
  const topLaneOf = createTopLaneResolver(g);
  const nodesByRank = [...g.nodes].sort(
    (a, b) => (rankOf[a] ?? 0) - (rankOf[b] ?? 0) || a.localeCompare(b)
  );
  for (const v of nodesByRank) {
    const laneV = topLaneOf(v);
    if (!laneV) {
      continue;
    }
    const outEdges = g.edges.filter((e) => e.src === v);
    if (outEdges.length === 0) {
      continue;
    }
    let hasSameLaneSucc = false;
    let crossLaneCount = 0;
    for (const e of outEdges) {
      const laneDst = topLaneOf(e.dst);
      if (laneDst == null || laneDst === laneV) {
        hasSameLaneSucc = true;
      } else {
        crossLaneCount++;
      }
    }
    if (crossLaneCount === 0 || hasSameLaneSucc) {
      continue;
    }
    let crossLaneIncoming = 0;
    let hasSameLanePred = false;
    for (const e of g.edges) {
      if (e.dst !== v) {
        continue;
      }
      const laneSrc = topLaneOf(e.src);
      if (!laneSrc) {
        continue;
      }
      if (laneSrc === laneV) {
        hasSameLanePred = true;
      } else {
        crossLaneIncoming++;
      }
    }
    if (crossLaneIncoming > 0 || !hasSameLanePred) {
      continue;
    }
    const current = rankOf[v] ?? 0;
    const target = current + crossLaneCount;
    let lb = 0;
    for (const e of g.edges) {
      if (e.dst === v) {
        lb = Math.max(lb, (rankOf[e.src] ?? 0) + 1);
      }
    }
    const newRank = Math.max(current, lb, target);
    if (newRank !== current) {
      rankOf[v] = newRank;
    }
  }
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(adjustCrossLaneSources, "adjustCrossLaneSources");

// src/rendering-util/layout-algorithms/swimlanes/phase2.longestPath.ts
function assignLayers_LongestPath(gAcyclic, opts) {
  const g = normalizeGraph(gAcyclic);
  const order = topoSortIfAcyclic(g) ?? [...g.nodes].sort();
  const compact = opts?.compactSingleInput ?? false;
  const topLaneOf = createTopLaneResolver(g);
  let rankOf = /* @__PURE__ */ Object.create(null);
  for (const v of order) {
    const incAll = incoming(g, v);
    const inc = opts?.ignoreCrossLaneEdges ? incAll.filter((e) => {
      const laneSrc = topLaneOf(e.src);
      const laneDst = topLaneOf(v);
      if (!laneSrc || !laneDst) {
        return true;
      }
      return laneSrc === laneDst;
    }) : incAll;
    if (inc.length === 0) {
      rankOf[v] = 0;
    } else if (compact && inc.length === 1) {
      const u = inc[0].src;
      const laneU = topLaneOf(u);
      const laneV = topLaneOf(v);
      if (laneU !== laneV) {
        rankOf[v] = rankOf[u] ?? 0;
      } else {
        rankOf[v] = (rankOf[u] ?? 0) + 1;
      }
    } else {
      let mx = -Infinity;
      for (const e of inc) {
        mx = Math.max(mx, (rankOf[e.src] ?? 0) + 1);
      }
      rankOf[v] = mx === -Infinity ? 0 : mx;
    }
  }
  if (opts?.optimizeRanksByCrossings ?? false) {
    rankOf = optimizeRanksByCrossings(g, rankOf);
  }
  if (opts?.ignoreCrossLaneEdges) {
    adjustCrossLaneSources(g, rankOf);
  }
  const layers = buildMultitreeLayerOrder(g, rankOf, topLaneOf);
  return { layers, rankOf, dummy: /* @__PURE__ */ new Set() };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignLayers_LongestPath, "assignLayers_LongestPath");

// src/rendering-util/layout-algorithms/swimlanes/phase2.gravity.ts
function assignLayers_Gravity(gAcyclic, opts) {
  const g = normalizeGraph(gAcyclic);
  const base = assignLayers_LongestPath(g, {
    compactSingleInput: opts?.compactSingleInput,
    ignoreCrossLaneEdges: opts?.ignoreCrossLaneEdges,
    optimizeRanksByCrossings: opts?.optimizeRanksByCrossings
  });
  const rankOf = { ...base.rankOf };
  const topLaneOf = createTopLaneResolver(g);
  const { preds, succs } = buildPredecessorSuccessorMaps(g, (e) => {
    if (opts?.ignoreCrossLaneEdges) {
      const laneSrc = topLaneOf(e.src);
      const laneDst = topLaneOf(e.dst);
      if (laneSrc && laneDst && laneSrc !== laneDst) {
        return false;
      }
    }
    return true;
  });
  const order = topoSortIfAcyclic(g) ?? [...g.nodes];
  const revOrder = [...order].reverse();
  const clampFeasible = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((v, desired) => {
    let lb = 0;
    for (const u of preds.get(v) ?? []) {
      lb = Math.max(lb, (rankOf[u] ?? 0) + 1);
    }
    let ub = Number.POSITIVE_INFINITY;
    const s = succs.get(v) ?? [];
    if (s.length > 0) {
      ub = Math.min(...s.map((w) => (rankOf[w] ?? 0) - 1));
    }
    if (!Number.isFinite(ub)) {
      ub = Math.max(lb, desired);
    }
    return Math.min(Math.max(desired, lb), ub);
  }, "clampFeasible");
  const iters = LAYERING.GRAVITY_ITERATIONS;
  const relaxOrder = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((nodeOrder) => {
    let changed = false;
    for (const v of nodeOrder) {
      const ps = preds.get(v) ?? [];
      const ss = succs.get(v) ?? [];
      if (ps.length === 0 && ss.length === 0) {
        continue;
      }
      const predAvg = ps.length > 0 ? ps.reduce((a, u) => a + (rankOf[u] ?? 0) + 1, 0) / ps.length : rankOf[v] ?? 0;
      const succAvg = ss.length > 0 ? ss.reduce((a, w) => a + (rankOf[w] ?? 0) - 1, 0) / ss.length : rankOf[v] ?? 0;
      const desired = Math.round((predAvg + succAvg) / 2);
      const clamped = clampFeasible(v, desired);
      if (clamped !== rankOf[v]) {
        rankOf[v] = clamped;
        changed = true;
      }
    }
    return changed;
  }, "relaxOrder");
  for (let it = 0; it < iters; it++) {
    const forwardChanged = relaxOrder(order);
    const backwardChanged = relaxOrder(revOrder);
    if (!forwardChanged && !backwardChanged) {
      break;
    }
  }
  for (const v of order) {
    let lb = 0;
    for (const u of preds.get(v) ?? []) {
      lb = Math.max(lb, (rankOf[u] ?? 0) + 1);
    }
    if ((rankOf[v] ?? 0) < lb) {
      rankOf[v] = lb;
    }
  }
  for (const v of revOrder) {
    const s = succs.get(v) ?? [];
    if (s.length > 0) {
      const ub = Math.min(...s.map((w) => (rankOf[w] ?? 0) - 1));
      if ((rankOf[v] ?? 0) > ub) {
        rankOf[v] = ub;
      }
    }
  }
  const layers = buildLayersFromRanks(g, order, rankOf);
  return { layers, rankOf, dummy: /* @__PURE__ */ new Set() };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignLayers_Gravity, "assignLayers_Gravity");

// src/rendering-util/layout-algorithms/swimlanes/phase2.laneAwareCompact.ts
function topoSortByGenerationIfAcyclic(g) {
  const indeg = buildInDegreeMap(g);
  const adj = buildSortedSuccessorMap(g);
  let frontier = sortedZeroInDegreeNodes(indeg);
  const order = [];
  while (frontier.length > 0) {
    const nextFrontier = [];
    for (const u of frontier) {
      order.push(u);
      for (const v of adj.get(u) ?? []) {
        indeg.set(v, (indeg.get(v) ?? 0) - 1);
        if ((indeg.get(v) ?? 0) === 0) {
          nextFrontier.push(v);
        }
      }
    }
    frontier = nextFrontier.sort((a, b) => a.localeCompare(b));
  }
  return order.length === g.nodes.length ? order : null;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(topoSortByGenerationIfAcyclic, "topoSortByGenerationIfAcyclic");
function assignLayers_LaneAwareCompact(gAcyclic, opts) {
  const g = normalizeGraph(gAcyclic);
  const order = opts?.direction === "LR" ? topoSortByGenerationIfAcyclic(g) ?? [...g.nodes].sort() : topoSortIfAcyclic(g) ?? [...g.nodes].sort();
  const topLaneOf = createTopLaneResolver(g);
  const laneOf = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((id) => topLaneOf(id) ?? id, "laneOf");
  const rankOf = /* @__PURE__ */ Object.create(null);
  const nextFree = /* @__PURE__ */ new Map();
  const edgeWeight = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((u, v) => {
    const ignoreCrossLane = opts?.ignoreCrossLaneEdges ?? true;
    if (ignoreCrossLane) {
      return laneOf(u) === laneOf(v) ? 1 : 0;
    }
    return 1;
  }, "edgeWeight");
  for (const v of order) {
    const node = g.nodeById.get(v);
    if (node?.isGroup) {
      continue;
    }
    const preds = incoming(g, v);
    let base = 0;
    if (preds.length > 0) {
      for (const e of preds) {
        const u = e.src;
        const ru = rankOf[u] ?? 0;
        base = Math.max(base, ru + edgeWeight(u, v));
      }
    }
    const lane = laneOf(v);
    const nf = nextFree.get(lane) ?? 0;
    const L = Math.max(base, nf);
    rankOf[v] = L;
    nextFree.set(lane, L + 1);
  }
  const layers = buildLayersFromRanks(g, order, rankOf, { skipGroups: true });
  return { layers, rankOf, dummy: /* @__PURE__ */ new Set() };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignLayers_LaneAwareCompact, "assignLayers_LaneAwareCompact");

// src/rendering-util/layout-algorithms/swimlanes/phase2.dummies.ts
function makeProperLayering(layering, gAcyclic) {
  const g = normalizeGraph(gAcyclic);
  const { rankOf } = layering;
  const layers = layering.layers.map((l) => [...l]);
  const dummy = new Set(layering.dummy ? [...layering.dummy] : []);
  let dummySeq = 0;
  const nodeById = new Map(g.nodeById);
  const addDummyAt = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((L) => {
    const id = `placeholder-${dummySeq++}`;
    const dn = { id, isGroup: false, isDummy: true, width: 0, height: 0 };
    nodeById.set(id, dn);
    dummy.add(id);
    while (layers.length <= L) {
      layers.push([]);
    }
    layers[L].push(id);
    rankOf[id] = L;
    return id;
  }, "addDummyAt");
  const edgesSorted = [...g.edges].sort(
    (a, b) => a.id === b.id ? a.src === b.src ? a.dst.localeCompare(b.dst) : a.src.localeCompare(b.src) : a.id.localeCompare(b.id)
  );
  const newEdges = [];
  for (const e of edgesSorted) {
    const rU = rankOf[e.src] ?? 0;
    const rV = rankOf[e.dst] ?? 0;
    if (rV - rU <= 1) {
      newEdges.push(e);
      continue;
    }
    let prev = e.src;
    for (let L = rU + 1, k = 0; L < rV; L++, k++) {
      const d = addDummyAt(L);
      newEdges.push({ id: `${e.id}#${k}`, src: prev, dst: d, weight: e.weight, ref: e.ref });
      prev = d;
    }
    const lastIndex = rV - rU - 2;
    newEdges.push({
      id: `${e.id}#${Math.max(lastIndex + 1, 0)}`,
      src: prev,
      dst: e.dst,
      weight: e.weight,
      ref: e.ref
    });
  }
  const nodes = [...g.nodes, ...[...dummy].filter((id) => !g.nodes.includes(id))];
  const graphWithDummies = { nodes, edges: newEdges, layout: g.layout, nodeById };
  return { layering: { layers, rankOf, dummy }, graphWithDummies };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(makeProperLayering, "makeProperLayering");

// src/rendering-util/layout-algorithms/swimlanes/phase3.ordering.ts
function median(values) {
  const n = values.length;
  if (n === 0) {
    return Number.POSITIVE_INFINITY;
  }
  const a = [...values].sort((x, y) => x - y);
  if (n % 2 === 1) {
    return a[(n - 1) / 2];
  }
  return 0.5 * (a[n / 2 - 1] + a[n / 2]);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(median, "median");
function barycenter(values) {
  if (values.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  const s = values.reduce((acc, v) => acc + v, 0);
  return s / values.length;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(barycenter, "barycenter");
function neighborPositionsFor(targetNodes, fixedIndex, edges, direction) {
  const neighborPositions = /* @__PURE__ */ new Map();
  for (const v of targetNodes) {
    neighborPositions.set(v, []);
  }
  for (const e of edges) {
    if (direction === "down") {
      if (fixedIndex.has(e.src) && neighborPositions.has(e.dst)) {
        neighborPositions.get(e.dst).push(fixedIndex.get(e.src));
      }
    } else if (fixedIndex.has(e.dst) && neighborPositions.has(e.src)) {
      neighborPositions.get(e.src).push(fixedIndex.get(e.dst));
    }
  }
  return neighborPositions;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(neighborPositionsFor, "neighborPositionsFor");
function currentOrderTieBreak(a, b, currentLayerIndex) {
  const ia = currentLayerIndex.get(a) ?? 0;
  const ib = currentLayerIndex.get(b) ?? 0;
  return ia !== ib ? ia - ib : a.localeCompare(b);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(currentOrderTieBreak, "currentOrderTieBreak");
function countCrossingsBetweenAdjacent2(upper, lower, edges) {
  const upperSet = new Set(upper);
  const lowerSet = new Set(lower);
  const upperIndex = buildLayerIndex(upper);
  const lowerIndex = buildLayerIndex(lower);
  const pairs = [];
  for (const e of edges) {
    if (upperSet.has(e.src) && lowerSet.has(e.dst)) {
      pairs.push({ u: upperIndex.get(e.src), v: lowerIndex.get(e.dst) });
    }
  }
  pairs.sort((a, b) => a.u === b.u ? a.v - b.v : a.u - b.u);
  const vs = pairs.map((p) => p.v);
  return countInversions(vs);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(countCrossingsBetweenAdjacent2, "countCrossingsBetweenAdjacent");
function sortByHeuristic(nodes, neighborPositions, currentLayerIndex) {
  return [...nodes].sort((a, b) => {
    const sa = median(neighborPositions.get(a) ?? []);
    const sb = median(neighborPositions.get(b) ?? []);
    if (sa === sb) {
      return currentOrderTieBreak(a, b, currentLayerIndex);
    }
    if (!isFinite(sa)) {
      return 1;
    }
    if (!isFinite(sb)) {
      return -1;
    }
    return sa - sb;
  });
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sortByHeuristic, "sortByHeuristic");
function reorderLayer(fixedLayer, targetLayer, edges, direction, topLaneOf, laneOrder) {
  const fixedIndex = buildLayerIndex(fixedLayer);
  const currIndex = buildLayerIndex(targetLayer);
  const neighborPositions = neighborPositionsFor(targetLayer, fixedIndex, edges, direction);
  if (!topLaneOf || !laneOrder || laneOrder.length === 0) {
    return sortByHeuristic(targetLayer, neighborPositions, currIndex);
  }
  const byLane = /* @__PURE__ */ new Map();
  for (const id of targetLayer) {
    const lane = topLaneOf(id);
    const arr = byLane.get(lane) ?? [];
    arr.push(id);
    byLane.set(lane, arr);
  }
  const result = [];
  for (const lane of laneOrder) {
    const nodesInLane = byLane.get(lane);
    if (!nodesInLane || nodesInLane.length === 0) {
      continue;
    }
    const sorted = sortByHeuristic(nodesInLane, neighborPositions, currIndex);
    result.push(...sorted);
  }
  const nullNodes = byLane.get(null);
  if (nullNodes && nullNodes.length > 0) {
    const sorted = sortByHeuristic(nullNodes, neighborPositions, currIndex);
    for (const nid of sorted) {
      const bc = barycenter(neighborPositions.get(nid) ?? []);
      let bestIdx = result.length;
      if (isFinite(bc)) {
        for (const [i, rid] of result.entries()) {
          const rBc = barycenter(neighborPositions.get(rid) ?? []);
          if (bc < rBc) {
            bestIdx = i;
            break;
          }
        }
      }
      result.splice(bestIdx, 0, nid);
    }
  }
  return result;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(reorderLayer, "reorderLayer");
function transposeImprove(upper, current, edges, next, topLaneOf) {
  const best = [...current];
  const upperSet = new Set(upper);
  const layerSet = new Set(current);
  const nextSet = next ? new Set(next) : null;
  const edgesIn = edges.filter((e) => upperSet.has(e.src) && layerSet.has(e.dst));
  const edgesOut = nextSet ? edges.filter((e) => layerSet.has(e.src) && nextSet.has(e.dst)) : void 0;
  const crossingScore = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((order) => {
    let score = countCrossingsBetweenAdjacent2(upper, order, edgesIn);
    if (edgesOut && next) {
      score += countCrossingsBetweenAdjacent2(order, next, edgesOut);
    }
    return score;
  }, "crossingScore");
  const laneOf = topLaneOf ? /* @__PURE__ */ new Map() : null;
  if (topLaneOf && laneOf) {
    for (const id of current) {
      laneOf.set(id, topLaneOf(id));
    }
  }
  let improved = true;
  let bestScore = crossingScore(best);
  while (improved) {
    improved = false;
    for (let i = 0; i + 1 < best.length; i++) {
      if (laneOf) {
        const laneA = laneOf.get(best[i]);
        const laneB = laneOf.get(best[i + 1]);
        if (laneA !== laneB) {
          continue;
        }
      }
      const prev = bestScore;
      [best[i], best[i + 1]] = [best[i + 1], best[i]];
      const nextScore = crossingScore(best);
      if (nextScore < prev) {
        bestScore = nextScore;
        improved = true;
      } else {
        [best[i], best[i + 1]] = [best[i + 1], best[i]];
      }
    }
  }
  return best;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(transposeImprove, "transposeImprove");
function orderLayers(layering, gWithDummies, opts) {
  const layers = layering.layers.map((l) => [...l]);
  const edges = gWithDummies.edges;
  const topLaneOf = createTopLaneResolver(gWithDummies);
  const laneOrder = resolveTopLaneOrder(gWithDummies, opts?.laneOrder);
  for (let s = 0; s < 3; s++) {
    for (let i = 1; i < layers.length; i++) {
      layers[i] = reorderLayer(layers[i - 1], layers[i], edges, "down", topLaneOf, laneOrder);
      layers[i] = transposeImprove(layers[i - 1], layers[i], edges, layers[i + 1], topLaneOf);
    }
    for (let i = layers.length - 2; i >= 0; i--) {
      layers[i] = reorderLayer(layers[i + 1], layers[i], edges, "up", topLaneOf, laneOrder);
      layers[i] = transposeImprove(layers[i + 1], layers[i], edges, layers[i - 1], topLaneOf);
    }
  }
  return { layers };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(orderLayers, "orderLayers");

// src/rendering-util/layout-algorithms/swimlanes/phase4.coordinates.ts
function assignCoordinates(ordered, gWithDummies, opts) {
  const layerGap = opts?.layerGap ?? COORDINATES.DEFAULT_LAYER_GAP;
  const nodeGap = opts?.nodeGap ?? COORDINATES.DEFAULT_NODE_GAP;
  const laneGap = opts?.laneGap ?? nodeGap * 2;
  const direction = opts?.direction ?? "TB";
  const isHorizontal = direction === "LR" || direction === "RL";
  const layers = ordered.layers;
  const x = /* @__PURE__ */ Object.create(null);
  const y = /* @__PURE__ */ Object.create(null);
  const getNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((id) => gWithDummies.nodeById.get(id), "getNode");
  const getWidth = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((id) => getNode(id)?.width ?? 0, "getWidth");
  const getHeight = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((id) => getNode(id)?.height ?? 0, "getHeight");
  const topLaneOf = createTopLaneResolver(gWithDummies);
  const laneOrderGlobal = resolveTopLaneOrder(gWithDummies, opts?.laneOrder);
  const layerHeights = layers.map(
    (layer) => layer.reduce((m, v) => Math.max(m, getHeight(v)), 0)
  );
  const extraLayerGaps = [];
  if (isHorizontal) {
    for (let i = 0; i + 1 < layers.length; i++) {
      const thisLayerMaxWidth = layers[i].reduce((m, v) => Math.max(m, getWidth(v)), 0);
      const nextLayerMaxWidth = layers[i + 1].reduce((m, v) => Math.max(m, getWidth(v)), 0);
      const thisLayerMaxHeight = layerHeights[i];
      const nextLayerMaxHeight = layerHeights[i + 1];
      const normalSpacing = thisLayerMaxHeight / 2 + nextLayerMaxHeight / 2;
      const requiredSpacing = (thisLayerMaxWidth + nextLayerMaxWidth) / 2;
      const extraNeeded = Math.max(0, requiredSpacing - normalSpacing - layerGap);
      extraLayerGaps.push(extraNeeded);
    }
  }
  const lanesUsedSet = /* @__PURE__ */ new Set();
  for (const layer of layers) {
    for (const id of layer) {
      lanesUsedSet.add(topLaneOf(id));
    }
  }
  const hasNullLane = lanesUsedSet.has(null);
  const lanesUsed = laneOrderGlobal.filter((L) => lanesUsedSet.has(L));
  const laneOrderColumns = [...hasNullLane ? [null] : [], ...lanesUsed];
  const laneWidth = /* @__PURE__ */ Object.create(null);
  for (const L of lanesUsed) {
    laneWidth[L] = 0;
  }
  if (hasNullLane) {
    laneWidth.null = 0;
  }
  for (const layer of layers) {
    const perLane = /* @__PURE__ */ Object.create(null);
    const nullIds = [];
    for (const id of layer) {
      const L = topLaneOf(id);
      if (L === null) {
        nullIds.push(id);
      } else {
        (perLane[L] ||= []).push(id);
      }
    }
    for (const [L, ids] of Object.entries(perLane)) {
      const total = ids.reduce((s, id) => s + getWidth(id), 0) + nodeGap * Math.max(0, ids.length - 1);
      laneWidth[L] = Math.max(laneWidth[L] ?? 0, total);
    }
    if (hasNullLane && nullIds.length) {
      const totalNull = nullIds.reduce((s, id) => s + getWidth(id), 0) + nodeGap * Math.max(0, nullIds.length - 1);
      laneWidth.null = Math.max(laneWidth.null ?? 0, totalNull);
    }
  }
  const centerX = /* @__PURE__ */ new Map();
  {
    const widths = laneOrderColumns.map(
      (L) => (L === null ? laneWidth.null : laneWidth[L]) ?? 0
    );
    const totalW = widths.reduce((a, b) => a + b, 0) + laneGap * Math.max(0, laneOrderColumns.length - 1);
    let cursor = -totalW / 2;
    for (let i = 0; i < laneOrderColumns.length; i++) {
      const L = laneOrderColumns[i];
      const w = widths[i] ?? 0;
      const cx = cursor + w / 2;
      centerX.set(L, cx);
      cursor += w;
      if (i < laneOrderColumns.length - 1) {
        cursor += laneGap;
      }
    }
  }
  let yOffset = 0;
  for (const [li, layer] of layers.entries()) {
    const layerH = layerHeights[li] ?? 0;
    const byLane = /* @__PURE__ */ new Map();
    for (const id of layer) {
      const laneId = topLaneOf(id);
      const arr = byLane.get(laneId) ?? [];
      arr.push(id);
      byLane.set(laneId, arr);
    }
    for (const L of laneOrderColumns) {
      const nodesInLane = byLane.get(L) ?? [];
      if (nodesInLane.length === 0) {
        continue;
      }
      const cx = centerX.get(L);
      if (nodesInLane.length === 1) {
        const id = nodesInLane[0];
        x[id] = cx;
        y[id] = yOffset + layerH / 2;
      } else {
        const widths = nodesInLane.map((id) => getWidth(id));
        const total = widths.reduce((a, b) => a + b, 0) + nodeGap * (nodesInLane.length - 1);
        let start = cx - total / 2;
        for (const [i, id] of nodesInLane.entries()) {
          const w = widths[i];
          x[id] = start + w / 2;
          y[id] = yOffset + layerH / 2;
          start += w + nodeGap;
        }
      }
    }
    const extraGap = extraLayerGaps[li] ?? 0;
    yOffset += layerH + layerGap + extraGap;
  }
  const byRef = /* @__PURE__ */ new Map();
  for (const e of gWithDummies.edges) {
    const rid = e.ref.id;
    if (!byRef.has(rid)) {
      byRef.set(rid, []);
    }
    byRef.get(rid).push(e);
  }
  for (const [, chainEdges] of byRef) {
    if (chainEdges.length === 0) {
      continue;
    }
    const ref = chainEdges[0].ref;
    const src = ref.start;
    const dst = ref.end;
    if (src == null || dst == null) {
      continue;
    }
    const midX = Math.round(((x[src] ?? 0) + (x[dst] ?? 0)) / 2);
    const involved = /* @__PURE__ */ new Set();
    for (const e of chainEdges) {
      involved.add(e.src);
      involved.add(e.dst);
    }
    for (const vid of involved) {
      if (vid === src || vid === dst) {
        continue;
      }
      const node = gWithDummies.nodeById.get(vid);
      if (node?.isDummy) {
        x[vid] = midX;
      }
    }
  }
  return { x, y };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(assignCoordinates, "assignCoordinates");

// src/rendering-util/layout-algorithms/swimlanes/laneOrdering.ts
var AUTOMATIC_LANE_ORDERING_RESTARTS = 8;
function hashString(input) {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(hashString, "hashString");
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 1831565813;
    let t = state;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(mulberry32, "mulberry32");
function deterministicShuffle(order, seed) {
  const shuffled = [...order];
  const random = mulberry32(seed);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(deterministicShuffle, "deterministicShuffle");
function sourceDistance(order, sourceIndex) {
  let distance = 0;
  for (const [index, laneId] of order.entries()) {
    distance += Math.abs(index - (sourceIndex.get(laneId) ?? index));
  }
  return distance;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sourceDistance, "sourceDistance");
function laneArrangementCost(order, weights) {
  const position = /* @__PURE__ */ new Map();
  for (const [index, laneId] of order.entries()) {
    position.set(laneId, index);
  }
  let cost = 0;
  for (const { a, b, weight } of weights) {
    const ai = position.get(a);
    const bi = position.get(b);
    if (ai == null || bi == null) {
      continue;
    }
    cost += weight * Math.abs(ai - bi);
  }
  return cost;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(laneArrangementCost, "laneArrangementCost");
function buildWeightedLaneEdges(g) {
  const sourceOrder = buildTopLaneOrder(g);
  if (sourceOrder.length < 2) {
    return [];
  }
  const sourceIndex = new Map(sourceOrder.map((laneId, index) => [laneId, index]));
  const topLaneOf = createTopLaneResolver(g);
  const weights = /* @__PURE__ */ new Map();
  for (const edge of g.layout.edges ?? []) {
    if (edge.isLayoutOnly) {
      continue;
    }
    const src = typeof edge.start === "string" ? edge.start : void 0;
    const dst = typeof edge.end === "string" ? edge.end : void 0;
    if (!src || !dst || !g.nodeById.has(src) || !g.nodeById.has(dst)) {
      continue;
    }
    const laneA = topLaneOf(src);
    const laneB = topLaneOf(dst);
    if (!laneA || !laneB || laneA === laneB) {
      continue;
    }
    const ia = sourceIndex.get(laneA);
    const ib = sourceIndex.get(laneB);
    if (ia == null || ib == null) {
      continue;
    }
    const [a, b] = ia <= ib ? [laneA, laneB] : [laneB, laneA];
    const key = `${a}\0${b}`;
    const existing = weights.get(key);
    if (existing) {
      existing.weight++;
    } else {
      weights.set(key, { a, b, weight: 1 });
    }
  }
  return [...weights.values()];
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(buildWeightedLaneEdges, "buildWeightedLaneEdges");
function greedySwitch(startOrder, weights, sourceIndex) {
  const order = [...startOrder];
  let cost = laneArrangementCost(order, weights);
  let changed = true;
  let sweeps = 0;
  const maxSweeps = Math.max(1, order.length);
  while (changed && sweeps < maxSweeps) {
    changed = false;
    sweeps++;
    for (let i = 0; i + 1 < order.length; i++) {
      [order[i], order[i + 1]] = [order[i + 1], order[i]];
      const nextCost = laneArrangementCost(order, weights);
      if (nextCost < cost) {
        cost = nextCost;
        changed = true;
      } else {
        [order[i], order[i + 1]] = [order[i + 1], order[i]];
      }
    }
  }
  return {
    order,
    cost,
    sourceDistance: sourceDistance(order, sourceIndex)
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(greedySwitch, "greedySwitch");
function isBetterCandidate(candidate, best) {
  if (candidate.cost !== best.cost) {
    return candidate.cost < best.cost;
  }
  return candidate.sourceDistance < best.sourceDistance;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(isBetterCandidate, "isBetterCandidate");
function seedForRestart(sourceOrder, weights, restartIndex) {
  const weightSignature = [...weights].sort((a, b) => a.a === b.a ? a.b.localeCompare(b.b) : a.a.localeCompare(b.a)).map(({ a, b, weight }) => `${a}:${b}:${weight}`).join("|");
  return hashString(`${sourceOrder.join("|")}#${weightSignature}#${restartIndex}`);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(seedForRestart, "seedForRestart");
function optimizeTopLaneOrder(g, opts = {}) {
  const sourceOrder = buildTopLaneOrder(g);
  if (sourceOrder.length < 2) {
    return sourceOrder;
  }
  const weights = buildWeightedLaneEdges(g);
  if (weights.length === 0) {
    return sourceOrder;
  }
  const sourceIndex = new Map(sourceOrder.map((laneId, index) => [laneId, index]));
  let best = greedySwitch(sourceOrder, weights, sourceIndex);
  const restarts = Math.max(0, opts.restarts ?? AUTOMATIC_LANE_ORDERING_RESTARTS);
  for (let i = 0; i < restarts; i++) {
    const seed = seedForRestart(sourceOrder, weights, i);
    const start = deterministicShuffle(sourceOrder, seed);
    const candidate = greedySwitch(start, weights, sourceIndex);
    if (isBetterCandidate(candidate, best)) {
      best = candidate;
    }
  }
  return best.order;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(optimizeTopLaneOrder, "optimizeTopLaneOrder");

// src/rendering-util/layout-algorithms/swimlanes/pipeline.ts
function sugiyamaLayout(g, opts) {
  const ignoreCrossLaneEdges = opts?.ignoreCrossLaneEdges ?? true;
  const optimizeRanksByCrossings2 = opts?.optimizeRanksByCrossings ?? true;
  const g0 = normalizeGraph(g);
  const laneOrder = opts?.automaticLaneOrdering ? optimizeTopLaneOrder(g0, { restarts: AUTOMATIC_LANE_ORDERING_RESTARTS }) : void 0;
  const cycleRes = removeCycles_DFS(g0);
  const gAcyclic = cycleRes.acyclic;
  const layering = ignoreCrossLaneEdges ? assignLayers_LaneAwareCompact(gAcyclic, {
    compactSingleInput: opts?.compactSingleInput ?? LAYERING.DEFAULT_COMPACT_SINGLE_INPUT,
    ignoreCrossLaneEdges: true,
    direction: opts?.direction
  }) : assignLayers_Gravity(gAcyclic, {
    compactSingleInput: opts?.compactSingleInput ?? LAYERING.DEFAULT_COMPACT_SINGLE_INPUT,
    ignoreCrossLaneEdges: false,
    optimizeRanksByCrossings: optimizeRanksByCrossings2
  });
  const { layering: properLayering, graphWithDummies } = makeProperLayering(layering, gAcyclic);
  const ordered = orderLayers(properLayering, graphWithDummies, { laneOrder });
  const coordinates = assignCoordinates(ordered, graphWithDummies, {
    layerGap: opts?.layerGap,
    nodeGap: opts?.nodeGap,
    direction: opts?.direction,
    laneOrder
  });
  return {
    acyclic: gAcyclic,
    reversed: cycleRes.reversed,
    layering: properLayering,
    ordered,
    coordinates
  };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sugiyamaLayout, "sugiyamaLayout");

// src/rendering-util/layout-algorithms/swimlanes/orthogonalRouter/router.ts
var EPS7 = PRECISION.EPSILON;
var NODE_PADDING = 8;
var HORIZONTAL_PIPE_MARGIN = 15;
var VERTICAL_PIPE_MARGIN = 15;
var ROUTING_MARGIN = 25;
var ANCHOR_OFFSET = 20;
var TRACK_SPACING = 10;
function chooseOrthogonalSide(node, target, fallback) {
  const cx = node.x ?? 0;
  const cy = node.y ?? 0;
  const dx = target.x - cx;
  const dy = target.y - cy;
  const absDx = Math.abs(dx);
  const absDy = Math.abs(dy);
  if (absDx < EPS7 && absDy < EPS7) {
    return fallback;
  }
  const verticalBias = 3;
  if (absDy > EPS7 && absDy * verticalBias >= absDx) {
    return dy > 0 ? "bottom" : "top";
  }
  if (absDx > EPS7) {
    return dx > 0 ? "right" : "left";
  }
  return fallback;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(chooseOrthogonalSide, "chooseOrthogonalSide");
function sharedLineEndpointCoord(line, nextLine) {
  return Math.abs(line.to - nextLine.from) < EPS7 || Math.abs(line.to - nextLine.to) < EPS7 ? line.to : line.from;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(sharedLineEndpointCoord, "sharedLineEndpointCoord");
function pointOnLine(line, along) {
  return line.orient === "vertical" ? { x: line.coord, y: along } : { x: along, y: line.coord };
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(pointOnLine, "pointOnLine");
function routeEdgesOrthogonal(data, direction) {
  const nodes = data.nodes ?? [];
  const originalEdges = data.edges ?? [];
  const edges = [];
  for (const oe of originalEdges) {
    if (oe.isLayoutOnly) {
      continue;
    }
    edges.push({
      ...oe,
      __originalEdge: oe
    });
  }
  const nodeById = /* @__PURE__ */ new Map();
  const laneByNodeId = /* @__PURE__ */ new Map();
  const pipes = [];
  const isLR = direction === "LR";
  for (const n of nodes) {
    nodeById.set(n.id, n);
  }
  const topLevelGroups = nodes.filter((n) => n.isGroup && !n.parentId);
  for (const group of topLevelGroups) {
    const lane = { id: group.id };
    const assignLane = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((n) => {
      laneByNodeId.set(n.id, lane);
      nodes.filter((child) => child.parentId === n.id).forEach(assignLane);
    }, "assignLane");
    assignLane(group);
  }
  const obstacles = nodes.filter((n) => !n.isGroup && !n.isEdgeLabel).map((n) => {
    const w = n.width ?? 10;
    const h = n.height ?? 10;
    const x = n.x ?? 0;
    const y = n.y ?? 0;
    const padding = NODE_PADDING;
    return {
      nodeId: n.id,
      minX: x - w / 2 - padding,
      maxX: x + w / 2 + padding,
      minY: y - h / 2 - padding,
      maxY: y + h / 2 + padding,
      // For LR: TB x becomes LR y, so visual Y extent should be based on height
      visualXHalfExtent: isLR ? h / 2 + padding : w / 2 + padding
    };
  });
  const getOrAddPipe = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((orientation, coord, spanMin, spanMax) => {
    let pipe = pipes.find((p) => p.orientation === orientation && Math.abs(p.coord - coord) < 1);
    if (!pipe) {
      pipe = {
        id: `pipe-${orientation}-${coord.toFixed(0)}`,
        orientation,
        coord,
        spanMin,
        spanMax,
        tracks: []
      };
      pipes.push(pipe);
    }
    pipe.spanMin = Math.min(pipe.spanMin, spanMin);
    pipe.spanMax = Math.max(pipe.spanMax, spanMax);
    return pipe;
  }, "getOrAddPipe");
  const portForSide = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, side) => {
    const w = node.width ?? 10;
    const h = node.height ?? 10;
    const cx = node.x ?? 0;
    const cy = node.y ?? 0;
    switch (side) {
      case "top":
        return { x: cx, y: cy - h / 2 };
      case "bottom":
        return { x: cx, y: cy + h / 2 };
      case "left":
        return { x: cx - w / 2, y: cy };
      case "right":
        return { x: cx + w / 2, y: cy };
    }
  }, "portForSide");
  const getOrthogonalPort = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, target, isSource) => portForSide(node, chooseOrthogonalSide(node, target, isSource ? "bottom" : "top")), "getOrthogonalPort");
  const allRoutedSegments = [];
  const edgeSegmentIndices = [];
  const straightIntraLaneEdges = /* @__PURE__ */ new Set();
  const CROSSING_PENALTY = 1e3;
  const crossingPenalty = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIdx, from, to) => {
    if (allRoutedSegments.length === 0) {
      return 0;
    }
    const isHorizontal = Math.abs(from.y - to.y) < EPS7;
    const isVertical = Math.abs(from.x - to.x) < EPS7;
    if (!isHorizontal && !isVertical) {
      return 0;
    }
    let penalties = 0;
    if (isHorizontal) {
      const y = from.y;
      const minX = Math.min(from.x, to.x) - EPS7;
      const maxX = Math.max(from.x, to.x) + EPS7;
      if (maxX <= minX) {
        return 0;
      }
      for (const seg of allRoutedSegments) {
        if (seg.edgeIndex === edgeIdx || seg.orientation !== "vertical") {
          continue;
        }
        if (seg.pipe.coord < minX || seg.pipe.coord > maxX) {
          continue;
        }
        if (seg.from - EPS7 <= y && seg.to + EPS7 >= y) {
          penalties += CROSSING_PENALTY;
        }
      }
    } else if (isVertical) {
      const x = from.x;
      const minY = Math.min(from.y, to.y) - EPS7;
      const maxY = Math.max(from.y, to.y) + EPS7;
      if (maxY <= minY) {
        return 0;
      }
      for (const seg of allRoutedSegments) {
        if (seg.edgeIndex === edgeIdx || seg.orientation !== "horizontal") {
          continue;
        }
        if (seg.pipe.coord < minY || seg.pipe.coord > maxY) {
          continue;
        }
        if (seg.from - EPS7 <= x && seg.to + EPS7 >= x) {
          penalties += CROSSING_PENALTY;
        }
      }
    }
    return penalties;
  }, "crossingPenalty");
  const routingOrder = edges.map((edge, idx) => {
    if (!edge.start || !edge.end) {
      return { idx, crossLane: 0, dx: 0, dy: 0 };
    }
    const srcNode = nodeById.get(edge.start);
    const dstNode = nodeById.get(edge.end);
    const srcLane = laneByNodeId.get(edge.start);
    const dstLane = laneByNodeId.get(edge.end);
    const crossLane = srcLane && dstLane && srcLane.id !== dstLane.id ? 1 : 0;
    const dx = srcNode && dstNode ? Math.abs((dstNode.x ?? 0) - (srcNode.x ?? 0)) : 0;
    const dy = srcNode && dstNode ? Math.abs((dstNode.y ?? 0) - (srcNode.y ?? 0)) : 0;
    return { idx, crossLane, dx, dy };
  }).sort((a, b) => {
    if (a.crossLane !== b.crossLane) {
      return b.crossLane - a.crossLane;
    }
    const aDist = a.dx + a.dy;
    const bDist = b.dx + b.dy;
    if (Math.abs(aDist - bDist) > 1) {
      return aDist - bDist;
    }
    return a.idx - b.idx;
  }).map((entry) => entry.idx);
  const isSegmentBlocked = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((p1, p2, excludeStart, excludeEnd) => {
    const segMinX = Math.min(p1.x, p2.x);
    const segMaxX = Math.max(p1.x, p2.x);
    const segMinY = Math.min(p1.y, p2.y);
    const segMaxY = Math.max(p1.y, p2.y);
    const blockingObs = obstacles.find((obs) => {
      if (excludeStart && obs.nodeId === excludeStart) {
        return false;
      }
      if (excludeEnd && obs.nodeId === excludeEnd) {
        return false;
      }
      if (Math.abs(p1.x - p2.x) > EPS7) {
        return obs.minY < p1.y && obs.maxY > p1.y && obs.maxX > segMinX && obs.minX < segMaxX;
      } else {
        return obs.minX < p1.x && obs.maxX > p1.x && obs.maxY > segMinY && obs.minY < segMaxY;
      }
    });
    return !!blockingObs;
  }, "isSegmentBlocked");
  const portGroups = /* @__PURE__ */ new Map();
  const incidentEdgeTotals = /* @__PURE__ */ new Map();
  for (const edge of edges) {
    if (!edge.start || !edge.end || edge.start === edge.end) {
      continue;
    }
    incidentEdgeTotals.set(edge.start, (incidentEdgeTotals.get(edge.start) ?? 0) + 1);
    incidentEdgeTotals.set(edge.end, (incidentEdgeTotals.get(edge.end) ?? 0) + 1);
  }
  const determineSide = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node, target) => chooseOrthogonalSide(node, target, "bottom"), "determineSide");
  const sideInfoByIdx = /* @__PURE__ */ new Map();
  for (const [i, e] of edges.entries()) {
    if (!e.start || !e.end || e.start === e.end) {
      continue;
    }
    if (e.points && e.points.length > 0) {
      continue;
    }
    const src = nodeById.get(e.start);
    const dst = nodeById.get(e.end);
    if (!src || !dst) {
      continue;
    }
    const dx = (dst.x ?? 0) - (src.x ?? 0);
    const dy = (dst.y ?? 0) - (src.y ?? 0);
    sideInfoByIdx.set(i, {
      edgeIdx: i,
      srcId: e.start,
      dstId: e.end,
      srcSide: determineSide(src, { x: dst.x ?? 0, y: dst.y ?? 0 }),
      dstSide: determineSide(dst, { x: src.x ?? 0, y: src.y ?? 0 }),
      absDx: Math.abs(dx),
      absDy: Math.abs(dy),
      dxSign: Math.sign(dx),
      dySign: Math.sign(dy)
    });
  }
  const preferenceStrength = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((info) => {
    if (info.srcSide === "top" || info.srcSide === "bottom") {
      return info.absDx === 0 ? Infinity : info.absDy / info.absDx;
    }
    return info.absDy === 0 ? Infinity : info.absDx / info.absDy;
  }, "preferenceStrength");
  const secondarySide = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((info) => {
    if (info.srcSide === "top" || info.srcSide === "bottom") {
      return info.dxSign >= 0 ? "right" : "left";
    }
    return info.dySign >= 0 ? "bottom" : "top";
  }, "secondarySide");
  const sourceSideGroups = /* @__PURE__ */ new Map();
  for (const info of sideInfoByIdx.values()) {
    const key = `${info.srcId}:${info.srcSide}`;
    if (!sourceSideGroups.has(key)) {
      sourceSideGroups.set(key, []);
    }
    sourceSideGroups.get(key).push(info);
  }
  const sideLoad = /* @__PURE__ */ new Map();
  const loadKey = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((nodeId, side) => `${nodeId}:${side}`, "loadKey");
  for (const info of sideInfoByIdx.values()) {
    sideLoad.set(
      loadKey(info.srcId, info.srcSide),
      (sideLoad.get(loadKey(info.srcId, info.srcSide)) ?? 0) + 1
    );
    sideLoad.set(
      loadKey(info.dstId, info.dstSide),
      (sideLoad.get(loadKey(info.dstId, info.dstSide)) ?? 0) + 1
    );
  }
  for (const group of sourceSideGroups.values()) {
    if (group.length < 2) {
      continue;
    }
    group.sort((a, b) => {
      const sa = preferenceStrength(a);
      const sb = preferenceStrength(b);
      if (Math.abs(sa - sb) > 1e-9) {
        return sb - sa;
      }
      return a.edgeIdx - b.edgeIdx;
    });
    for (let g = 1; g < group.length; g++) {
      const info = group[g];
      const secondary = secondarySide(info);
      const primaryLoad = sideLoad.get(loadKey(info.srcId, info.srcSide)) ?? 0;
      const secondaryLoad = sideLoad.get(loadKey(info.srcId, secondary)) ?? 0;
      if (secondaryLoad >= primaryLoad) {
        continue;
      }
      sideLoad.set(loadKey(info.srcId, info.srcSide), primaryLoad - 1);
      sideLoad.set(loadKey(info.srcId, secondary), secondaryLoad + 1);
      info.srcSide = secondary;
    }
  }
  const isDiamondNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((node) => {
    const shape = node?.shape;
    return shape === "question" || shape === "diamond";
  }, "isDiamondNode");
  const inSidesByNode = /* @__PURE__ */ new Map();
  for (const info of sideInfoByIdx.values()) {
    if (!inSidesByNode.has(info.dstId)) {
      inSidesByNode.set(info.dstId, /* @__PURE__ */ new Set());
    }
    inSidesByNode.get(info.dstId).add(info.dstSide);
  }
  for (const info of sideInfoByIdx.values()) {
    if (!isDiamondNode(nodeById.get(info.srcId))) {
      continue;
    }
    const inSides = inSidesByNode.get(info.srcId);
    if (!inSides?.has(info.srcSide)) {
      continue;
    }
    const secondary = secondarySide(info);
    if (inSides.has(secondary) || (sideLoad.get(loadKey(info.srcId, secondary)) ?? 0) > 0) {
      continue;
    }
    const primaryLoad = sideLoad.get(loadKey(info.srcId, info.srcSide)) ?? 0;
    sideLoad.set(loadKey(info.srcId, info.srcSide), Math.max(0, primaryLoad - 1));
    sideLoad.set(loadKey(info.srcId, secondary), 1);
    info.srcSide = secondary;
  }
  for (const info of sideInfoByIdx.values()) {
    const { edgeIdx: i, srcId, dstId, srcSide, dstSide } = info;
    const src = nodeById.get(srcId);
    const dst = nodeById.get(dstId);
    const srcKey = `${srcId}:${srcSide}:src`;
    const dstCoord = srcSide === "top" || srcSide === "bottom" ? dst.x ?? 0 : dst.y ?? 0;
    if (!portGroups.has(srcKey)) {
      portGroups.set(srcKey, []);
    }
    portGroups.get(srcKey).push({ edgeIdx: i, oppositeCoord: dstCoord });
    const dstKey = `${dstId}:${dstSide}:dst`;
    const srcCoord = dstSide === "top" || dstSide === "bottom" ? src.x ?? 0 : src.y ?? 0;
    if (!portGroups.has(dstKey)) {
      portGroups.set(dstKey, []);
    }
    portGroups.get(dstKey).push({ edgeIdx: i, oppositeCoord: srcCoord });
  }
  const portOffsets = /* @__PURE__ */ new Map();
  const MIN_PORT_SPACING3 = 8;
  for (const [key, group] of portGroups) {
    if (group.length < 2) {
      continue;
    }
    group.sort((a, b) => a.oppositeCoord - b.oppositeCoord);
    const parts = key.split(":");
    const nodeId = parts.slice(0, -2).join(":");
    const side = parts[parts.length - 2];
    const role = parts[parts.length - 1];
    const node = nodeById.get(nodeId);
    if (!node) {
      continue;
    }
    const isVerticalSide = side === "left" || side === "right";
    const sideLength = isVerticalSide ? node.height ?? 10 : node.width ?? 10;
    const shape = node.shape;
    const isDiamond = shape === "question" || shape === "diamond";
    const effectiveLength = isDiamond ? sideLength * 0.3 : sideLength;
    const MAX_PORT_SPACING = 20;
    const spacing = Math.min(
      MAX_PORT_SPACING,
      Math.max(MIN_PORT_SPACING3, effectiveLength / (group.length + 1))
    );
    const totalSpan = spacing * (group.length - 1);
    const startOffset = -totalSpan / 2;
    for (const [j, element] of group.entries()) {
      const offset = startOffset + j * spacing;
      const offsetKey = `${element.edgeIdx}:${role}`;
      portOffsets.set(offsetKey, offset);
    }
  }
  const edgeHasLabelNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIdx) => Boolean(edges[edgeIdx]?.labelNodeId), "edgeHasLabelNode");
  const faceHasLabelNode = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((nodeId, side) => {
    if (!nodeId) {
      return false;
    }
    return (portGroups.get(`${nodeId}:${side}:src`) ?? []).some(
      ({ edgeIdx }) => edgeHasLabelNode(edgeIdx)
    ) || (portGroups.get(`${nodeId}:${side}:dst`) ?? []).some(
      ({ edgeIdx }) => edgeHasLabelNode(edgeIdx)
    );
  }, "faceHasLabelNode");
  const applyPortOffset = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((basePort, side, offset) => {
    if (side === "top" || side === "bottom") {
      return { x: basePort.x + offset, y: basePort.y };
    } else {
      return { x: basePort.x, y: basePort.y + offset };
    }
  }, "applyPortOffset");
  const portsForEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIndex, src, dst) => {
    const sideInfo = sideInfoByIdx.get(edgeIndex);
    const srcTarget = { x: dst.x ?? 0, y: dst.y ?? 0 };
    const dstTarget = { x: src.x ?? 0, y: src.y ?? 0 };
    const srcSide = sideInfo?.srcSide ?? determineSide(src, srcTarget);
    const dstSide = sideInfo?.dstSide ?? determineSide(dst, dstTarget);
    let pSrcPort = sideInfo ? portForSide(src, sideInfo.srcSide) : getOrthogonalPort(src, srcTarget, true);
    let pDstPort = sideInfo ? portForSide(dst, sideInfo.dstSide) : getOrthogonalPort(dst, dstTarget, false);
    const srcOffset = portOffsets.get(`${edgeIndex}:src`);
    const dstOffset = portOffsets.get(`${edgeIndex}:dst`);
    if (srcOffset !== void 0) {
      pSrcPort = applyPortOffset(pSrcPort, srcSide, srcOffset);
    }
    if (dstOffset !== void 0) {
      pDstPort = applyPortOffset(pDstPort, dstSide, dstOffset);
    }
    return { pSrcPort, pDstPort, srcSide, dstSide };
  }, "portsForEdge");
  for (const i of routingOrder) {
    const e = edges[i];
    edgeSegmentIndices[i] = [];
    if (!e.start || !e.end) {
      continue;
    }
    if (e.points && e.points.length > 0) {
      continue;
    }
    if (e.start === e.end) {
      continue;
    }
    const src = nodeById.get(e.start);
    const dst = nodeById.get(e.end);
    if (!src || !dst) {
      continue;
    }
    const {
      pSrcPort,
      pDstPort,
      srcSide: srcPortSide,
      dstSide: dstPortSide
    } = portsForEdge(i, src, dst);
    const pSrcAnchor = { ...pSrcPort };
    const pDstAnchor = { ...pDstPort };
    const srcPortIsVertical = srcPortSide === "top" || srcPortSide === "bottom";
    const dstPortIsVertical = dstPortSide === "top" || dstPortSide === "bottom";
    if (srcPortIsVertical) {
      const isBottom = pSrcPort.y > (src.y ?? 0);
      pSrcAnchor.y = isBottom ? pSrcPort.y + ANCHOR_OFFSET : pSrcPort.y - ANCHOR_OFFSET;
    } else {
      const isRight = pSrcPort.x > (src.x ?? 0);
      pSrcAnchor.x = isRight ? pSrcPort.x + ANCHOR_OFFSET : pSrcPort.x - ANCHOR_OFFSET;
    }
    if (dstPortIsVertical) {
      const isBottom = pDstPort.y > (dst.y ?? 0);
      pDstAnchor.y = isBottom ? pDstPort.y + ANCHOR_OFFSET : pDstPort.y - ANCHOR_OFFSET;
    } else {
      const isRight = pDstPort.x > (dst.x ?? 0);
      pDstAnchor.x = isRight ? pDstPort.x + ANCHOR_OFFSET : pDstPort.x - ANCHOR_OFFSET;
    }
    const isPointInObstacle = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pt, excludeNodeIds) => {
      for (const obs of obstacles) {
        if (excludeNodeIds.includes(obs.nodeId)) {
          continue;
        }
        if (pt.x > obs.minX && pt.x < obs.maxX && pt.y > obs.minY && pt.y < obs.maxY) {
          return { inside: true, obstacle: obs };
        }
      }
      return { inside: false };
    }, "isPointInObstacle");
    const obstacleDetour = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((port, node, opposite, obs, portIsVertical) => {
      if (portIsVertical) {
        const leavesPositiveSide2 = port.y > (node.y ?? 0);
        const goRight = (opposite.x ?? 0) >= port.x;
        return {
          x: goRight ? obs.maxX + HORIZONTAL_PIPE_MARGIN : obs.minX - HORIZONTAL_PIPE_MARGIN,
          y: leavesPositiveSide2 ? obs.maxY + VERTICAL_PIPE_MARGIN : obs.minY - VERTICAL_PIPE_MARGIN,
          leavesPositiveSide: leavesPositiveSide2
        };
      }
      const leavesPositiveSide = port.x > (node.x ?? 0);
      const goDown = (opposite.y ?? 0) >= port.y;
      return {
        x: leavesPositiveSide ? obs.maxX + HORIZONTAL_PIPE_MARGIN : obs.minX - HORIZONTAL_PIPE_MARGIN,
        y: goDown ? obs.maxY + VERTICAL_PIPE_MARGIN : obs.minY - VERTICAL_PIPE_MARGIN,
        leavesPositiveSide
      };
    }, "obstacleDetour");
    let srcHandleWaypoints = [];
    const endpointIds = [e.start, e.end];
    const srcCheck = isPointInObstacle(pSrcAnchor, endpointIds);
    if (srcCheck.inside && srcCheck.obstacle) {
      const obs = srcCheck.obstacle;
      if (srcPortIsVertical) {
        const detour = obstacleDetour(pSrcPort, src, dst, obs, true);
        pSrcAnchor.x = detour.x;
        pSrcAnchor.y = detour.y;
        const gapY = detour.leavesPositiveSide ? Math.min(obs.minY - 2, pSrcPort.y + ANCHOR_OFFSET) : Math.max(obs.maxY + 2, pSrcPort.y - ANCHOR_OFFSET);
        srcHandleWaypoints = [
          { x: pSrcPort.x, y: gapY },
          // Orthogonal step in the gap
          { x: detour.x, y: gapY },
          // Horizontal detour
          { x: detour.x, y: detour.y }
          // Down past obstacle
        ];
      } else {
        const detour = obstacleDetour(pSrcPort, src, dst, obs, false);
        const gapX = detour.leavesPositiveSide ? Math.min(obs.minX - 2, pSrcPort.x + ANCHOR_OFFSET) : Math.max(obs.maxX + 2, pSrcPort.x - ANCHOR_OFFSET);
        pSrcAnchor.x = detour.x;
        pSrcAnchor.y = detour.y;
        srcHandleWaypoints = [
          { x: gapX, y: pSrcPort.y },
          // Orthogonal step in the gap
          { x: gapX, y: detour.y },
          // Vertical detour
          { x: detour.x, y: detour.y }
          // Horizontal past obstacle
        ];
      }
    }
    let dstHandleWaypoints = [];
    const dstCheck = isPointInObstacle(pDstAnchor, endpointIds);
    if (dstCheck.inside && dstCheck.obstacle) {
      const obs = dstCheck.obstacle;
      if (dstPortIsVertical) {
        const detour = obstacleDetour(pDstPort, dst, src, obs, true);
        pDstAnchor.x = detour.x;
        pDstAnchor.y = detour.y;
        dstHandleWaypoints = [
          { x: detour.x, y: detour.y },
          // From anchor position
          { x: pDstPort.x, y: detour.y }
          // Go sideways to port's X
          // Then orthogonally to port
        ];
      } else {
        const detour = obstacleDetour(pDstPort, dst, src, obs, false);
        pDstAnchor.x = detour.x;
        pDstAnchor.y = detour.y;
        dstHandleWaypoints = [
          { x: detour.x, y: detour.y },
          // From anchor position
          { x: detour.x, y: pDstPort.y }
          // Go vertically to port's Y
        ];
      }
    }
    if (srcHandleWaypoints.length === 0 && dstHandleWaypoints.length === 0) {
      const hpMargin = HORIZONTAL_PIPE_MARGIN;
      const anchorsSameX = Math.abs(pSrcAnchor.x - pDstAnchor.x) < hpMargin;
      const anchorsSameY = Math.abs(pSrcAnchor.y - pDstAnchor.y) < hpMargin;
      const hasPortOffset = portOffsets.get(`${i}:src`) !== void 0 || portOffsets.get(`${i}:dst`) !== void 0;
      const srcFaceTotal = (portGroups.get(`${e.start ?? ""}:${srcPortSide}:src`)?.length ?? 0) + (portGroups.get(`${e.start ?? ""}:${srcPortSide}:dst`)?.length ?? 0);
      const dstFaceTotal = (portGroups.get(`${e.end ?? ""}:${dstPortSide}:src`)?.length ?? 0) + (portGroups.get(`${e.end ?? ""}:${dstPortSide}:dst`)?.length ?? 0);
      const faceContested = srcFaceTotal > 1 || dstFaceTotal > 1;
      const srcIncidentTotal = incidentEdgeTotals.get(e.start ?? "") ?? 0;
      const dstIncidentTotal = incidentEdgeTotals.get(e.end ?? "") ?? 0;
      const contestedFaceHasLabel = srcFaceTotal > 1 && faceHasLabelNode(e.start, srcPortSide) || dstFaceTotal > 1 && faceHasLabelNode(e.end, dstPortSide);
      const srcContestAllowsCenteredStraight = srcFaceTotal <= 1 || srcIncidentTotal <= 2;
      const dstContestAllowsCenteredStraight = dstFaceTotal <= 1 || dstIncidentTotal <= 2;
      const canPreserveSimpleContestedStraight = faceContested && !contestedFaceHasLabel && srcContestAllowsCenteredStraight && dstContestAllowsCenteredStraight;
      if ((anchorsSameX || anchorsSameY) && !hasPortOffset && (!faceContested || canPreserveSimpleContestedStraight)) {
        const directBlocked = isSegmentBlocked(pSrcPort, pDstPort, e.start, e.end);
        if (!directBlocked) {
          e.points = [{ ...pSrcPort }, { ...pSrcAnchor }, { ...pDstAnchor }, { ...pDstPort }];
          straightIntraLaneEdges.add(i);
          const fastPathOrientation = anchorsSameY ? "horizontal" : "vertical";
          const fastPathCoord = anchorsSameY ? pSrcPort.y : pSrcPort.x;
          const fastPathFrom = anchorsSameY ? Math.min(pSrcPort.x, pDstPort.x) : Math.min(pSrcPort.y, pDstPort.y);
          const fastPathTo = anchorsSameY ? Math.max(pSrcPort.x, pDstPort.x) : Math.max(pSrcPort.y, pDstPort.y);
          const fastPathPipe = {
            id: `fast-path-${fastPathOrientation}-${fastPathCoord.toFixed(0)}-${i}`,
            orientation: fastPathOrientation,
            coord: fastPathCoord,
            spanMin: fastPathFrom,
            spanMax: fastPathTo,
            tracks: []
          };
          allRoutedSegments.push({
            edgeIndex: i,
            segmentIndex: 0,
            orientation: fastPathOrientation,
            pipe: fastPathPipe,
            trackIndex: 0,
            from: fastPathFrom,
            to: fastPathTo
          });
          continue;
        }
      }
    }
    const srcPipe = getOrAddPipe("vertical", pSrcAnchor.x, pSrcAnchor.y, pSrcAnchor.y);
    pSrcAnchor.x = srcPipe.coord;
    const dstPipe = getOrAddPipe("vertical", pDstAnchor.x, pDstAnchor.y, pDstAnchor.y);
    pDstAnchor.x = dstPipe.coord;
    let bbMinX = Math.min(pSrcAnchor.x, pDstAnchor.x) - 50;
    let bbMaxX = Math.max(pSrcAnchor.x, pDstAnchor.x) + 50;
    let bbMinY = Math.min(pSrcAnchor.y, pDstAnchor.y) - 50;
    let bbMaxY = Math.max(pSrcAnchor.y, pDstAnchor.y) + 50;
    for (const obs of obstacles) {
      const pathMinX = Math.min(pSrcAnchor.x, pDstAnchor.x);
      const pathMaxX = Math.max(pSrcAnchor.x, pDstAnchor.x);
      const pathMinY = Math.min(pSrcAnchor.y, pDstAnchor.y);
      const pathMaxY = Math.max(pSrcAnchor.y, pDstAnchor.y);
      const obsBlocksPath = obs.minX < pathMaxX && obs.maxX > pathMinX && obs.minY < pathMaxY && obs.maxY > pathMinY;
      if (obsBlocksPath) {
        bbMinX = Math.min(bbMinX, obs.minX - ROUTING_MARGIN);
        bbMaxX = Math.max(bbMaxX, obs.maxX + ROUTING_MARGIN);
        bbMinY = Math.min(bbMinY, obs.minY - ROUTING_MARGIN);
        bbMaxY = Math.max(bbMaxY, obs.maxY + ROUTING_MARGIN);
      }
    }
    for (const obs of obstacles) {
      if (obs.maxX < bbMinX || obs.minX > bbMaxX || obs.maxY < bbMinY || obs.minY > bbMaxY) {
        continue;
      }
      const hMargin = HORIZONTAL_PIPE_MARGIN;
      getOrAddPipe("horizontal", obs.minY - hMargin, bbMinX, bbMaxX);
      getOrAddPipe("horizontal", obs.maxY + hMargin, bbMinX, bbMaxX);
      const vMargin = VERTICAL_PIPE_MARGIN;
      getOrAddPipe("vertical", obs.minX - vMargin, bbMinY, bbMaxY);
      getOrAddPipe("vertical", obs.maxX + vMargin, bbMinY, bbMaxY);
    }
    getOrAddPipe("horizontal", pSrcAnchor.y, bbMinX, bbMaxX);
    getOrAddPipe("horizontal", pDstAnchor.y, bbMinX, bbMaxX);
    const hPipes = pipes.filter(
      (p) => p.orientation === "horizontal" && p.coord >= bbMinY && p.coord <= bbMaxY
    );
    const vPipes = pipes.filter(
      (p) => p.orientation === "vertical" && p.coord >= bbMinX && p.coord <= bbMaxX
    );
    const getKey = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((x, y) => `${x.toFixed(1)},${y.toFixed(1)}`, "getKey");
    const startKey = getKey(pSrcAnchor.x, pSrcAnchor.y);
    const endKey = getKey(pDstAnchor.x, pDstAnchor.y);
    const gScore = /* @__PURE__ */ new Map();
    const cameFrom = /* @__PURE__ */ new Map();
    const arrivalDir = /* @__PURE__ */ new Map();
    const openSet = /* @__PURE__ */ new Set();
    const openList = [];
    gScore.set(startKey, 0);
    arrivalDir.set(startKey, "n");
    openList.push({
      key: startKey,
      f: Math.hypot(pDstAnchor.x - pSrcAnchor.x, pDstAnchor.y - pSrcAnchor.y),
      pt: pSrcAnchor
    });
    openSet.add(startKey);
    let foundPath = [];
    const checkSegmentBlocked = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((p1, p2) => {
      return isSegmentBlocked(p1, p2, e.start, e.end);
    }, "checkSegmentBlocked");
    const cornerHV = { x: pDstAnchor.x, y: pSrcAnchor.y };
    const seg1HV_blocked = checkSegmentBlocked(pSrcAnchor, cornerHV);
    const seg2HV_blocked = checkSegmentBlocked(cornerHV, pDstAnchor);
    const pathHV_blocked = seg1HV_blocked || seg2HV_blocked;
    const cornerVH = { x: pSrcAnchor.x, y: pDstAnchor.y };
    const seg1VH_blocked = checkSegmentBlocked(pSrcAnchor, cornerVH);
    const seg2VH_blocked = checkSegmentBlocked(cornerVH, pDstAnchor);
    const pathVH_blocked = seg1VH_blocked || seg2VH_blocked;
    if (!pathHV_blocked) {
      if (Math.abs(pSrcAnchor.y - pDstAnchor.y) < EPS7 || Math.abs(pSrcAnchor.x - pDstAnchor.x) < EPS7) {
        foundPath = [pSrcAnchor, pDstAnchor];
      } else {
        foundPath = [pSrcAnchor, cornerHV, pDstAnchor];
      }
    } else if (!pathVH_blocked) {
      if (Math.abs(pSrcAnchor.x - pDstAnchor.x) < EPS7) {
        foundPath = [pSrcAnchor, pDstAnchor];
      } else {
        foundPath = [pSrcAnchor, cornerVH, pDstAnchor];
      }
    }
    if (foundPath.length === 0) {
      while (openList.length > 0) {
        openList.sort((a, b) => a.f - b.f);
        const current = openList.shift();
        openSet.delete(current.key);
        if (current.key === endKey) {
          let currKey = endKey;
          let currPt = pDstAnchor;
          foundPath = [currPt];
          while (cameFrom.has(currKey)) {
            const prev = cameFrom.get(currKey);
            foundPath.unshift(prev);
            currPt = prev;
            currKey = getKey(prev.x, prev.y);
          }
          break;
        }
        const cx = current.pt.x;
        const cy = current.pt.y;
        const sortedVPipes = vPipes.sort((a, b) => a.coord - b.coord);
        const vIdx = sortedVPipes.findIndex((p) => Math.abs(p.coord - cx) < 1);
        const hPipesSorted = hPipes.sort((a, b) => a.coord - b.coord);
        const hIdx = hPipesSorted.findIndex((p) => Math.abs(p.coord - cy) < 1);
        const neighbors = [];
        if (vIdx > 0) {
          neighbors.push({ x: sortedVPipes[vIdx - 1].coord, y: cy });
        }
        if (vIdx >= 0 && vIdx < sortedVPipes.length - 1) {
          neighbors.push({ x: sortedVPipes[vIdx + 1].coord, y: cy });
        }
        if (hIdx > 0) {
          neighbors.push({ x: cx, y: hPipesSorted[hIdx - 1].coord });
        }
        if (hIdx >= 0 && hIdx < hPipesSorted.length - 1) {
          neighbors.push({ x: cx, y: hPipesSorted[hIdx + 1].coord });
        }
        for (const neighbor of neighbors) {
          const minX = Math.min(cx, neighbor.x);
          const maxX = Math.max(cx, neighbor.x);
          const minY = Math.min(cy, neighbor.y);
          const maxY = Math.max(cy, neighbor.y);
          const blocked = obstacles.some((obs) => {
            if (obs.nodeId === e.start || obs.nodeId === e.end) {
              return false;
            }
            if (minX !== maxX) {
              return obs.minY < cy && obs.maxY > cy && obs.maxX > minX && obs.minX < maxX;
            } else {
              return obs.minX < cx && obs.maxX > cx && obs.maxY > minY && obs.minY < maxY;
            }
          });
          if (blocked) {
            continue;
          }
          const nKey = getKey(neighbor.x, neighbor.y);
          const dist = Math.abs(neighbor.x - cx) + Math.abs(neighbor.y - cy);
          const penalty = crossingPenalty(i, current.pt, neighbor);
          let dirPenalty = 0;
          const destDx = pDstAnchor.x - pSrcAnchor.x;
          const destDy = pDstAnchor.y - pSrcAnchor.y;
          const moveDx = neighbor.x - cx;
          const moveDy = neighbor.y - cy;
          if (destDy > 10 && moveDy < -5 || destDy < -10 && moveDy > 5) {
            dirPenalty = Math.abs(moveDy) * 100;
          }
          if (destDx > 10 && moveDx < -5 || destDx < -10 && moveDx > 5) {
            dirPenalty += Math.abs(moveDx) * 50;
          }
          let bendPenalty = 0;
          const currentDir = arrivalDir.get(current.key) ?? "n";
          const moveDir = Math.abs(moveDx) > EPS7 ? "h" : "v";
          if (currentDir !== "n" && currentDir !== moveDir) {
            bendPenalty = 50;
          }
          const stepCost = dist + penalty + dirPenalty + bendPenalty;
          const tentativeG = (gScore.get(current.key) ?? Infinity) + stepCost;
          const h = Math.abs(pDstAnchor.x - neighbor.x) + Math.abs(pDstAnchor.y - neighbor.y);
          if (tentativeG < (gScore.get(nKey) ?? Infinity)) {
            cameFrom.set(nKey, current.pt);
            gScore.set(nKey, tentativeG);
            arrivalDir.set(nKey, moveDir);
            if (!openSet.has(nKey)) {
              openList.push({ key: nKey, f: tentativeG + h, pt: neighbor });
              openSet.add(nKey);
            } else {
              const idx = openList.findIndex((x) => x.key === nKey);
              if (idx !== -1) {
                openList[idx].f = tentativeG + h;
              }
            }
          }
        }
      }
    }
    if (foundPath.length === 0) {
      foundPath = [pSrcAnchor, { x: pSrcAnchor.x, y: pDstAnchor.y }, pDstAnchor];
    }
    if (foundPath.length > 4) {
      const start = foundPath[0];
      const end = foundPath[foundPath.length - 1];
      let minX = Math.min(start.x, end.x);
      let maxX = Math.max(start.x, end.x);
      let minY = Math.min(start.y, end.y);
      let maxY = Math.max(start.y, end.y);
      for (const pt of foundPath) {
        minX = Math.min(minX, pt.x);
        maxX = Math.max(maxX, pt.x);
        minY = Math.min(minY, pt.y);
        maxY = Math.max(maxY, pt.y);
      }
      const wentRight = maxX > Math.max(start.x, end.x);
      const wentLeft = minX < Math.min(start.x, end.x);
      if (isLR) {
        const margin = VERTICAL_PIPE_MARGIN;
        if (wentRight) {
          const pathX = Math.max(start.x, end.x);
          const pathMinY = Math.min(start.y, end.y);
          const pathMaxY = Math.max(start.y, end.y);
          const detourObstacles = obstacles.filter(
            (obs) => obs.minX < pathX && obs.maxX > pathX && // obstacle's x-range contains the path x
            obs.minY < pathMaxY && obs.maxY > pathMinY
            // obstacle's y-range overlaps with path y-range
          );
          if (detourObstacles.length > 0) {
            let visualMaxX = Math.max(start.x, end.x);
            for (const obs of detourObstacles) {
              const obsCenterX = (obs.minX + obs.maxX) / 2;
              if (obs.visualXHalfExtent === void 0 || isNaN(obs.visualXHalfExtent)) {
                continue;
              }
              const visualRight = obsCenterX + obs.visualXHalfExtent + margin;
              visualMaxX = Math.max(visualMaxX, visualRight);
            }
            if (!isNaN(visualMaxX)) {
              maxX = visualMaxX;
            }
          }
        }
        if (wentLeft) {
          const detourObstacles = obstacles.filter(
            (obs) => obs.minX < Math.min(start.x, end.x) + margin && // obstacle extends past the direct path
            obs.minY < Math.max(start.y, end.y) && obs.maxY > Math.min(start.y, end.y)
            // obstacle is in Y range
          );
          if (detourObstacles.length > 0) {
            let visualMinX = Math.min(start.x, end.x);
            for (const obs of detourObstacles) {
              const obsCenterX = (obs.minX + obs.maxX) / 2;
              const visualLeft = obsCenterX - obs.visualXHalfExtent - margin;
              visualMinX = Math.min(visualMinX, visualLeft);
            }
            minX = visualMinX;
          }
        }
      }
      const findBestReturnY = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((detourX) => {
        const goingDown = end.y > start.y;
        const relevantObs = obstacles.filter((obs) => {
          const obsInXRange = Math.min(start.x, end.x) < obs.maxX && Math.max(start.x, end.x) > obs.minX;
          const obsInYRange = Math.min(start.y, end.y) < obs.maxY && Math.max(start.y, end.y) > obs.minY;
          return obsInXRange && obsInYRange;
        });
        let filteredObs = relevantObs;
        if (isLR && relevantObs.length > 0) {
          const obsAtDetourX = relevantObs.filter(
            (obs) => obs.minX < detourX && obs.maxX > detourX
          );
          if (obsAtDetourX.length > 0) {
            filteredObs = obsAtDetourX;
          }
        }
        if (filteredObs.length === 0) {
          return end.y;
        }
        const margin = HORIZONTAL_PIPE_MARGIN;
        if (goingDown) {
          const lowestObsBottom = Math.max(...filteredObs.map((obs) => obs.maxY));
          const bestY = lowestObsBottom + margin;
          if (bestY < end.y - EPS7) {
            return bestY;
          }
        } else {
          const highestObsTop = Math.min(...filteredObs.map((obs) => obs.minY));
          const bestY = highestObsTop - margin;
          if (bestY > end.y + EPS7) {
            return bestY;
          }
        }
        return end.y;
      }, "findBestReturnY");
      const trySimplifyWithDetourX = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((detourX) => {
        const bestY = findBestReturnY(detourX);
        const corner1 = { x: detourX, y: start.y };
        const corner2 = { x: detourX, y: bestY };
        const corner3 = { x: end.x, y: bestY };
        const seg1Blocked = checkSegmentBlocked(start, corner1);
        const seg2Blocked = checkSegmentBlocked(corner1, corner2);
        const seg3Blocked = checkSegmentBlocked(corner2, corner3);
        const seg4Blocked = bestY !== end.y ? checkSegmentBlocked(corner3, end) : false;
        if (!seg1Blocked && !seg2Blocked && !seg3Blocked && !seg4Blocked) {
          if (Math.abs(bestY - end.y) < EPS7) {
            return [start, corner1, corner2, end];
          }
          return [start, corner1, corner2, corner3, end];
        }
        return null;
      }, "trySimplifyWithDetourX");
      const simplified2 = wentRight && !wentLeft ? trySimplifyWithDetourX(maxX) : wentLeft && !wentRight ? trySimplifyWithDetourX(minX) : null;
      if (simplified2) {
        foundPath = simplified2;
      }
    }
    const fullPoints = [
      pSrcPort,
      ...srcHandleWaypoints,
      ...foundPath,
      ...dstHandleWaypoints.reverse(),
      // Reverse because they're stored anchor->waypoint->port
      pDstPort
    ];
    if (fullPoints.length >= 3) {
      const C = fullPoints[fullPoints.length - 1];
      const B = fullPoints[fullPoints.length - 2];
      const A = fullPoints[fullPoints.length - 3];
      const isHoriz = Math.abs(A.y - B.y) < EPS7 && Math.abs(B.y - C.y) < EPS7;
      const isVert = Math.abs(A.x - B.x) < EPS7 && Math.abs(B.x - C.x) < EPS7;
      if (isHoriz) {
        const signAB = Math.sign(B.x - A.x);
        const signAC = Math.sign(C.x - A.x);
        if (signAB !== 0 && signAB === signAC && Math.abs(B.x - A.x) > Math.abs(C.x - A.x)) {
          fullPoints.splice(-2, 1);
        }
      } else if (isVert) {
        const signAB = Math.sign(B.y - A.y);
        const signAC = Math.sign(C.y - A.y);
        if (signAB !== 0 && signAB === signAC && Math.abs(B.y - A.y) > Math.abs(C.y - A.y)) {
          fullPoints.splice(-2, 1);
        }
      }
    }
    const simplified = [fullPoints[0]];
    for (let k = 1; k < fullPoints.length - 1; k++) {
      if (k === 1) {
        simplified.push(fullPoints[k]);
        continue;
      }
      const prev = simplified[simplified.length - 1];
      const curr = fullPoints[k];
      const next = fullPoints[k + 1];
      if (Math.abs(prev.y - curr.y) < EPS7 && Math.abs(curr.y - next.y) < EPS7) {
        const dir1 = curr.x > prev.x;
        const dir2 = next.x > curr.x;
        if (dir1 !== dir2) {
          simplified.push(curr);
          continue;
        }
        continue;
      }
      if (Math.abs(prev.x - curr.x) < EPS7 && Math.abs(curr.x - next.x) < EPS7) {
        const dir1 = curr.y > prev.y;
        const dir2 = next.y > curr.y;
        if (dir1 !== dir2) {
          simplified.push(curr);
          continue;
        }
        continue;
      }
      simplified.push(curr);
    }
    simplified.push(fullPoints[fullPoints.length - 1]);
    for (let k = 0; k < simplified.length - 1; k++) {
      const p1 = simplified[k];
      const p2 = simplified[k + 1];
      const orientation = Math.abs(p1.x - p2.x) < EPS7 ? "vertical" : "horizontal";
      const coord = orientation === "vertical" ? p1.x : p1.y;
      const from = orientation === "vertical" ? Math.min(p1.y, p2.y) : Math.min(p1.x, p2.x);
      const to = orientation === "vertical" ? Math.max(p1.y, p2.y) : Math.max(p1.x, p2.x);
      const pipe = getOrAddPipe(orientation, coord, from, to);
      const rSeg = {
        edgeIndex: i,
        segmentIndex: k,
        orientation,
        pipe,
        trackIndex: 0,
        // Initial track
        from,
        to
      };
      allRoutedSegments.push(rSeg);
      edgeSegmentIndices[i].push(allRoutedSegments.length - 1);
      if (!pipe.tracks[0]) {
        pipe.tracks[0] = { index: 0, coord: pipe.coord, segments: [] };
      }
      pipe.tracks[0].segments.push({
        edgeIndex: i,
        segmentIndex: k,
        from,
        to
      });
    }
  }
  const segmentsOverlap = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((s1, s2) => {
    return s1.from < s2.to && s2.from < s1.to;
  }, "segmentsOverlap");
  const trySwapSegmentsAcrossTracks = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((s1, s2, t1, t2) => {
    const canS1GoT2 = !t2.segments.some(
      (r) => (r.edgeIndex !== s2.edgeIndex || r.segmentIndex !== s2.segmentIndex) && segmentsOverlap(r, s1)
    );
    const canS2GoT1 = !t1.segments.some(
      (r) => (r.edgeIndex !== s1.edgeIndex || r.segmentIndex !== s1.segmentIndex) && segmentsOverlap(r, s2)
    );
    if (canS1GoT2 && canS2GoT1) {
      s1.trackIndex = t2.index;
      s2.trackIndex = t1.index;
      t1.segments = [
        ...t1.segments.filter(
          (r) => r.edgeIndex !== s1.edgeIndex || r.segmentIndex !== s1.segmentIndex
        ),
        {
          edgeIndex: s2.edgeIndex,
          segmentIndex: s2.segmentIndex,
          from: s2.from,
          to: s2.to
        }
      ];
      t2.segments = [
        ...t2.segments.filter(
          (r) => r.edgeIndex !== s2.edgeIndex || r.segmentIndex !== s2.segmentIndex
        ),
        {
          edgeIndex: s1.edgeIndex,
          segmentIndex: s1.segmentIndex,
          from: s1.from,
          to: s1.to
        }
      ];
      return true;
    }
    return false;
  }, "trySwapSegmentsAcrossTracks");
  const createNewTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pipe) => {
    const idx = pipe.tracks.length;
    pipe.tracks[idx] = { index: idx, coord: pipe.coord, segments: [] };
    return idx;
  }, "createNewTrack");
  const moveSegmentToTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg, trackIdx) => {
    const oldTrack = seg.pipe.tracks[seg.trackIndex];
    oldTrack.segments = oldTrack.segments.filter(
      (r) => r.edgeIndex !== seg.edgeIndex || r.segmentIndex !== seg.segmentIndex
    );
    seg.trackIndex = trackIdx;
    const newTrack = seg.pipe.tracks[trackIdx];
    newTrack.segments.push({
      edgeIndex: seg.edgeIndex,
      segmentIndex: seg.segmentIndex,
      from: seg.from,
      to: seg.to
    });
  }, "moveSegmentToTrack");
  const moveSegmentChainToTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg, trackIdx) => {
    const indices = edgeSegmentIndices[seg.edgeIndex];
    for (const idx of indices) {
      const s = allRoutedSegments[idx];
      if (s.pipe === seg.pipe) {
        moveSegmentToTrack(s, trackIdx);
      }
    }
  }, "moveSegmentChainToTrack");
  const getAdjacentSegmentsAlongEdge = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((seg) => {
    const indices = edgeSegmentIndices[seg.edgeIndex];
    const idxInList = indices.indexOf(allRoutedSegments.indexOf(seg));
    const adj = [];
    if (idxInList > 0) {
      adj.push(allRoutedSegments[indices[idxInList - 1]]);
    }
    if (idxInList < indices.length - 1) {
      adj.push(allRoutedSegments[indices[idxInList + 1]]);
    }
    return adj;
  }, "getAdjacentSegmentsAlongEdge");
  const haveAnyCrossing = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((segA, segB) => {
    if (segA.orientation === segB.orientation) {
      return false;
    }
    const h = segA.orientation === "horizontal" ? segA : segB;
    const v = segA.orientation === "horizontal" ? segB : segA;
    return v.pipe.coord > h.from && v.pipe.coord < h.to && h.pipe.coord > v.from && h.pipe.coord < v.to;
  }, "haveAnyCrossing");
  const findAvailableTrack = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((pipe, seg) => {
    for (const track of pipe.tracks) {
      const overlap = track.segments.some(
        (r) => (r.edgeIndex !== seg.edgeIndex || r.segmentIndex !== seg.segmentIndex) && segmentsOverlap(r, seg)
      );
      if (!overlap) {
        return track.index;
      }
    }
    return -1;
  }, "findAvailableTrack");
  const segmentsConflict = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((s1, s2) => {
    if (s1.trackIndex === s2.trackIndex) {
      return segmentsOverlap(s1, s2);
    }
    const adj1 = getAdjacentSegmentsAlongEdge(s1);
    const adj2 = getAdjacentSegmentsAlongEdge(s2);
    return adj1.some((a1) => adj2.some((a2) => haveAnyCrossing(a1, a2)));
  }, "segmentsConflict");
  const resolveTrackConflict = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((s1, s2, move) => {
    if (trySwapSegmentsAcrossTracks(
      s1,
      s2,
      s1.pipe.tracks[s1.trackIndex],
      s2.pipe.tracks[s2.trackIndex]
    )) {
      return;
    }
    const avail = findAvailableTrack(s1.pipe, s2);
    move(s2, avail !== -1 ? avail : createNewTrack(s1.pipe));
  }, "resolveTrackConflict");
  const resolveHandleConflicts = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((handles) => {
    let crossings = 0;
    for (let i = 0; i < handles.length; i++) {
      for (let j = i + 1; j < handles.length; j++) {
        const h1 = handles[i];
        const h2 = handles[j];
        if (h1.pipe !== h2.pipe) {
          continue;
        }
        if (segmentsConflict(h1, h2)) {
          crossings++;
          resolveTrackConflict(h1, h2, moveSegmentChainToTrack);
        }
      }
    }
    return crossings;
  }, "resolveHandleConflicts");
  const destInfoCache = /* @__PURE__ */ new Map();
  const getDestInfo = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIdx) => {
    if (destInfoCache.has(edgeIdx)) {
      return destInfoCache.get(edgeIdx);
    }
    const indices = edgeSegmentIndices[edgeIdx];
    if (indices.length === 0) {
      const info2 = { dest: 0, deviation: 0, base: 0, delta: 0 };
      destInfoCache.set(edgeIdx, info2);
      return info2;
    }
    const firstSeg = allRoutedSegments[indices[0]];
    const base = firstSeg.pipe.coord;
    let dest = base;
    for (let idx = 1; idx < indices.length; idx++) {
      const seg = allRoutedSegments[indices[idx]];
      if (seg.orientation === "horizontal") {
        const candidateA = seg.from;
        const candidateB = seg.to;
        dest = Math.abs(candidateA - base) > Math.abs(candidateB - base) ? candidateA : candidateB;
        break;
      }
    }
    const deviation = Math.abs(dest - base);
    const info = { dest, deviation, base, delta: dest - base };
    destInfoCache.set(edgeIdx, info);
    return info;
  }, "getDestInfo");
  const fixSourceHandleCrossings = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    let crossings = 0;
    const edgesBySource = /* @__PURE__ */ new Map();
    for (const [i, e] of edges.entries()) {
      if (edgeSegmentIndices[i].length === 0) {
        continue;
      }
      if (!e.start) {
        continue;
      }
      if (!edgesBySource.has(e.start)) {
        edgesBySource.set(e.start, []);
      }
      edgesBySource.get(e.start).push(i);
    }
    const getEdgeDistance = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIdx) => {
      const edge = edges[edgeIdx];
      if (!edge.start || !edge.end) {
        return 0;
      }
      const srcNode = nodeById.get(edge.start);
      const dstNode = nodeById.get(edge.end);
      if (!srcNode || !dstNode) {
        return 0;
      }
      const dx = (dstNode.x ?? 0) - (srcNode.x ?? 0);
      const dy = (dstNode.y ?? 0) - (srcNode.y ?? 0);
      return Math.abs(dx) + Math.abs(dy);
    }, "getEdgeDistance");
    for (const grp of edgesBySource.values()) {
      grp.sort((a, b) => {
        const infoA = getDestInfo(a);
        const infoB = getDestInfo(b);
        if (Math.abs(infoA.deviation - infoB.deviation) > 1) {
          return infoA.deviation - infoB.deviation;
        }
        if (Math.abs(infoA.dest - infoB.dest) > 1) {
          return infoA.dest - infoB.dest;
        }
        const distA = getEdgeDistance(a);
        const distB = getEdgeDistance(b);
        if (Math.abs(distA - distB) > 1) {
          return distB - distA;
        }
        const lenA = edgeSegmentIndices[a].length;
        const lenB = edgeSegmentIndices[b].length;
        if (lenA !== lenB) {
          return lenA - lenB;
        }
        if (lenA === 1) {
          const idxA = edgeSegmentIndices[a][0];
          const idxB = edgeSegmentIndices[b][0];
          if (allRoutedSegments[idxA] && allRoutedSegments[idxB]) {
            const segA = allRoutedSegments[idxA];
            const segB = allRoutedSegments[idxB];
            const distA2 = Math.abs(segA.to - segA.from);
            const distB2 = Math.abs(segB.to - segB.from);
            if (Math.abs(distA2 - distB2) > 1) {
              return distA2 - distB2;
            }
          }
        }
        return 0;
      });
      const handles = grp.map((ei) => allRoutedSegments[edgeSegmentIndices[ei][0]]);
      crossings += resolveHandleConflicts(handles);
    }
    return crossings;
  }, "fixSourceHandleCrossings");
  const fixTargetHandleCrossings = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    let crossings = 0;
    const edgesByTarget = /* @__PURE__ */ new Map();
    for (const [i, e] of edges.entries()) {
      const indices = edgeSegmentIndices[i];
      if (indices.length === 0) {
        continue;
      }
      if (!e.end) {
        continue;
      }
      if (!edgesByTarget.has(e.end)) {
        edgesByTarget.set(e.end, []);
      }
      edgesByTarget.get(e.end).push(i);
    }
    for (const grp of edgesByTarget.values()) {
      grp.sort((a, b) => {
        const getDist = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((edgeIdx) => {
          const indices = edgeSegmentIndices[edgeIdx];
          if (indices.length < 2) {
            return 0;
          }
          const prev = allRoutedSegments[indices[indices.length - 2]];
          return Math.abs(prev.to - prev.from);
        }, "getDist");
        const scoreA = getDist(a);
        const scoreB = getDist(b);
        if (Math.abs(scoreA - scoreB) > 0.1) {
          return scoreA - scoreB;
        }
        return a - b;
      });
      const handles = grp.map(
        (ei) => allRoutedSegments[edgeSegmentIndices[ei][edgeSegmentIndices[ei].length - 1]]
      );
      crossings += resolveHandleConflicts(handles);
    }
    return crossings;
  }, "fixTargetHandleCrossings");
  const fixPipeCrossings = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(() => {
    let crossings = 0;
    for (const pipe of pipes) {
      const pipeSegments = [];
      for (const t of pipe.tracks) {
        for (const ref of t.segments) {
          const idx = edgeSegmentIndices[ref.edgeIndex].find(
            (ix) => allRoutedSegments[ix].segmentIndex === ref.segmentIndex
          );
          if (idx !== void 0) {
            pipeSegments.push(allRoutedSegments[idx]);
          }
        }
      }
      pipeSegments.sort((a, b) => a.edgeIndex - b.edgeIndex || a.segmentIndex - b.segmentIndex);
      for (let i = 0; i < pipeSegments.length; i++) {
        for (let j = i + 1; j < pipeSegments.length; j++) {
          const s1 = pipeSegments[i];
          const s2 = pipeSegments[j];
          if (segmentsConflict(s1, s2)) {
            crossings++;
            resolveTrackConflict(s1, s2, moveSegmentToTrack);
          }
        }
      }
    }
    return crossings;
  }, "fixPipeCrossings");
  let iterations = 0;
  const MAX_ITER = 10;
  while (iterations < MAX_ITER) {
    let changed = 0;
    changed += fixSourceHandleCrossings();
    changed += fixTargetHandleCrossings();
    changed += fixPipeCrossings();
    if (changed === 0) {
      break;
    }
    iterations++;
  }
  const segmentCoords = /* @__PURE__ */ new Map();
  for (const pipe of pipes) {
    const segments = [];
    pipe.tracks.forEach((t) => {
      t.segments.forEach((s) => {
        segments.push({
          edgeIndex: s.edgeIndex,
          segmentIndex: s.segmentIndex,
          trackIndex: t.index,
          from: s.from,
          to: s.to
        });
      });
    });
    segments.sort((a, b) => a.from - b.from);
    const clusters = [];
    if (segments.length > 0) {
      let currentCluster = [segments[0]];
      let clusterEnd = segments[0].to;
      for (let k = 1; k < segments.length; k++) {
        const s = segments[k];
        if (s.from < clusterEnd) {
          currentCluster.push(s);
          clusterEnd = Math.max(clusterEnd, s.to);
        } else {
          clusters.push(currentCluster);
          currentCluster = [s];
          clusterEnd = s.to;
        }
      }
      clusters.push(currentCluster);
    }
    for (const cluster of clusters) {
      const usedTracks = /* @__PURE__ */ new Set();
      cluster.forEach((s) => usedTracks.add(s.trackIndex));
      const trackScores = /* @__PURE__ */ new Map();
      cluster.forEach((s) => {
        const info = getDestInfo(s.edgeIndex);
        trackScores.set(s.trackIndex, (trackScores.get(s.trackIndex) ?? 0) + info.delta);
      });
      const leftTracks = [...usedTracks].filter((t) => (trackScores.get(t) ?? 0) < -1);
      const rightTracks = [...usedTracks].filter((t) => (trackScores.get(t) ?? 0) > 1);
      const neutralTracks = [...usedTracks].filter((t) => Math.abs(trackScores.get(t) ?? 0) <= 1);
      leftTracks.sort((a, b) => (trackScores.get(b) ?? 0) - (trackScores.get(a) ?? 0));
      rightTracks.sort((a, b) => (trackScores.get(a) ?? 0) - (trackScores.get(b) ?? 0));
      const assignCoord = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((trackIndex, coord) => {
        cluster.filter((s) => s.trackIndex === trackIndex).forEach((s) => {
          const effectiveCoord = straightIntraLaneEdges.has(s.edgeIndex) ? pipe.coord : coord;
          segmentCoords.set(`${s.edgeIndex}-${s.segmentIndex}`, effectiveCoord);
        });
      }, "assignCoord");
      let leftCount = 0;
      for (const trackIndex of leftTracks) {
        leftCount++;
        assignCoord(trackIndex, pipe.coord - leftCount * TRACK_SPACING);
      }
      if (neutralTracks.length === 0 && usedTracks.size > 0) {
        const bestTrack = [...usedTracks].sort(
          (a, b) => Math.abs(trackScores.get(a) ?? 0) - Math.abs(trackScores.get(b) ?? 0)
        )[0];
        const leftIdx = leftTracks.indexOf(bestTrack);
        if (leftIdx !== -1) {
          leftTracks.splice(leftIdx, 1);
        }
        const rightIdx = rightTracks.indexOf(bestTrack);
        if (rightIdx !== -1) {
          rightTracks.splice(rightIdx, 1);
        }
        neutralTracks.push(bestTrack);
      }
      let neutralAssigned = 0;
      for (const trackIndex of neutralTracks) {
        if (neutralAssigned === 0) {
          assignCoord(trackIndex, pipe.coord);
        } else {
          const dir = neutralAssigned % 2 === 1 ? 1 : -1;
          const magnitude = Math.ceil(neutralAssigned / 2);
          assignCoord(trackIndex, pipe.coord + dir * magnitude * TRACK_SPACING * 0.5);
        }
        neutralAssigned++;
      }
      let rightCount = 0;
      for (const trackIndex of rightTracks) {
        rightCount++;
        assignCoord(trackIndex, pipe.coord + rightCount * TRACK_SPACING);
      }
    }
  }
  for (const [i, e] of edges.entries()) {
    const indices = edgeSegmentIndices[i] ?? [];
    if (indices.length === 0) {
      continue;
    }
    const newPoints = [];
    const src = nodeById.get(e.start);
    const dst = nodeById.get(e.end);
    const { pSrcPort, pDstPort } = portsForEdge(i, src, dst);
    const lines = indices.map((idx) => {
      const s = allRoutedSegments[idx];
      const coord = segmentCoords.get(`${s.edgeIndex}-${s.segmentIndex}`) ?? s.pipe.coord;
      return {
        orient: s.orientation,
        coord,
        from: s.from,
        to: s.to
      };
    });
    newPoints.push(pSrcPort);
    for (let k = 0; k < lines.length; k++) {
      const line = lines[k];
      const prevPt = newPoints[newPoints.length - 1];
      const prevAlong = line.orient === "vertical" ? prevPt.y : prevPt.x;
      const prevTrackCoord = line.orient === "vertical" ? prevPt.x : prevPt.y;
      const nextLine = lines[k + 1];
      const hasNextLine = k < lines.length - 1;
      if (Math.abs(prevTrackCoord - line.coord) > EPS7) {
        newPoints.push(pointOnLine(line, prevAlong));
      }
      if (hasNextLine && nextLine.orient === line.orient) {
        if (Math.abs(line.coord - nextLine.coord) > EPS7) {
          const junction = line.orient === "vertical" ? (prevAlong + nextLine.from) / 2 : sharedLineEndpointCoord(line, nextLine);
          newPoints.push(pointOnLine(line, junction), pointOnLine(nextLine, junction));
        } else if (k === 0 || k === lines.length - 2) {
          newPoints.push(pointOnLine(line, sharedLineEndpointCoord(line, nextLine)));
        }
      } else if (hasNextLine) {
        newPoints.push(pointOnLine(line, nextLine.coord));
      } else {
        const endAlong = Math.abs(line.from - prevAlong) < Math.abs(line.to - prevAlong) ? line.to : line.from;
        newPoints.push(pointOnLine(line, endAlong));
      }
    }
    const last = newPoints[newPoints.length - 1];
    if (Math.abs(last.x - pDstPort.x) > EPS7 || Math.abs(last.y - pDstPort.y) > EPS7) {
      newPoints.push(pDstPort);
    }
    const filtered = [];
    if (newPoints.length > 0) {
      filtered.push(newPoints[0]);
    }
    for (let k = 1; k < newPoints.length; k++) {
      const p = newPoints[k];
      const prev = filtered[filtered.length - 1];
      if (Math.abs(p.x - prev.x) > EPS7 || Math.abs(p.y - prev.y) > EPS7) {
        filtered.push(p);
      }
    }
    e.points = filtered;
  }
  for (const re of edges) {
    const orig = re.__originalEdge;
    if (orig && re.points) {
      orig.points = re.points;
    }
  }
  data.edges = (data.edges ?? []).filter((e) => !e.isLayoutOnly);
  const nodeBoundaryClamp = /* @__PURE__ */ (0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)((p, node) => {
    const cx = node.x ?? 0;
    const cy = node.y ?? 0;
    const w = node.width ?? 0;
    const h = node.height ?? 0;
    if (w <= 0 || h <= 0) {
      return p;
    }
    const left = cx - w / 2;
    const right = cx + w / 2;
    const top = cy - h / 2;
    const bottom = cy + h / 2;
    if (p.x < left || p.x > right || p.y < top || p.y > bottom) {
      return p;
    }
    const dLeft = p.x - left;
    const dRight = right - p.x;
    const dTop = p.y - top;
    const dBottom = bottom - p.y;
    const minD = Math.min(dLeft, dRight, dTop, dBottom);
    if (minD === dLeft) {
      return { x: left, y: p.y };
    }
    if (minD === dRight) {
      return { x: right, y: p.y };
    }
    if (minD === dTop) {
      return { x: p.x, y: top };
    }
    return { x: p.x, y: bottom };
  }, "nodeBoundaryClamp");
  for (const edge of data.edges) {
    const pts = edge.points;
    if (!pts || pts.length < 2) {
      continue;
    }
    const srcId = edge.start;
    const dstId = edge.end;
    const src = srcId ? nodeById.get(srcId) : void 0;
    const dst = dstId ? nodeById.get(dstId) : void 0;
    if (src) {
      pts[0] = nodeBoundaryClamp(pts[0], src);
    }
    if (dst) {
      pts[pts.length - 1] = nodeBoundaryClamp(pts[pts.length - 1], dst);
    }
  }
  return data;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(routeEdgesOrthogonal, "routeEdgesOrthogonal");

// src/rendering-util/layout-algorithms/swimlanes/layoutCore.ts
function getSwimlaneDirection(data4Layout) {
  return data4Layout.direction ?? "TB";
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(getSwimlaneDirection, "getSwimlaneDirection");
function runSwimlaneLayoutCore(data4Layout) {
  const g = toGraphView(data4Layout);
  const nodeGap = data4Layout.config.flowchart?.nodeSpacing ?? 40;
  const layerGap = data4Layout.config.flowchart?.rankSpacing ?? 100;
  const ignoreCrossLaneEdges = data4Layout.config.swimlane?.ignoreCrossLaneEdges ?? true;
  const optimizeRanksByCrossings2 = data4Layout.config.swimlane?.optimizeRanksByCrossings ?? true;
  const automaticLaneOrdering = data4Layout.config.swimlane?.automaticLaneOrdering ?? false;
  const direction = getSwimlaneDirection(data4Layout);
  const { ordered, coordinates } = sugiyamaLayout(g, {
    nodeGap,
    layerGap,
    ignoreCrossLaneEdges,
    optimizeRanksByCrossings: optimizeRanksByCrossings2,
    automaticLaneOrdering,
    direction
  });
  writeBackToLayoutData(g, ordered, coordinates, { nodeGap, layerGap });
  for (const edge of data4Layout.edges ?? []) {
    delete edge.points;
  }
  routeEdgesOrthogonal(data4Layout, direction);
  for (const edge of data4Layout.edges ?? []) {
    if (!edge.curve || edge.curve === "basis") {
      edge.curve = "rounded";
    }
  }
  postProcessSwimlaneLayout(data4Layout, direction);
  validateSwimlanesLayout(data4Layout);
  return direction;
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(runSwimlaneLayoutCore, "runSwimlaneLayoutCore");

// src/rendering-util/layout-algorithms/swimlanes/index.ts
async function render(data4Layout, svg) {
  const element = svg.select("g");
  (0,_chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .markers_default */ .g0)(element, data4Layout.markers, data4Layout.type, data4Layout.diagramId);
  (0,_chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__/* .clear2 */ .gh)();
  (0,_chunk_7Z6QIM7H_mjs__WEBPACK_IMPORTED_MODULE_1__/* .clear */ .IU)();
  (0,_chunk_QR6OTTB3_mjs__WEBPACK_IMPORTED_MODULE_2__/* .clear */ .IU)();
  (0,_chunk_RYQCIY6F_mjs__WEBPACK_IMPORTED_MODULE_0__/* .clear */ .IU)();
  prepareLayoutForSwimlanes(data4Layout);
  const transformedData = createEdgeLabelNodes(data4Layout);
  data4Layout.nodes = transformedData.nodes;
  data4Layout.edges = transformedData.edges;
  const { groups } = await createGraphWithElements(element, data4Layout);
  runSwimlaneLayoutCore(data4Layout);
  await adjustLayout(data4Layout, groups);
}
(0,_chunk_Y2CYZVJY_mjs__WEBPACK_IMPORTED_MODULE_11__/* .__name */ .K)(render, "render");



/***/ }

};
;