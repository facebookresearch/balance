# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

API Reference documentation was built using [Sphinx](https://www.sphinx-doc.org/en/master/index.html), a python documentation generator

FontAwesome icons were used under the
[Creative Commons Attribution 4.0 International](https://fontawesome.com/license/free).

### Installation

### Node.js requirement

This website now requires **Node.js 22 or newer** (`engines.node: >=22`). The repository also enforces engines via `website/.npmrc` (`engine-strict = true`), so installs on Node 20/21 will fail fast with an engine error.

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

balance's website uses Sphinx & Docusaurus for website generation. We suggest running our custom build script for generating all
artifacts:

```
// Run from repo root folder (balance)
$ ./scripts/make_docs.sh
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.
