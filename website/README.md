# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

API Reference documentation was built using [Sphinx](https://www.sphinx-doc.org/en/master/index.html), a python documentation generator

FontAwesome icons were used under the
[Creative Commons Attribution 4.0 International](https://fontawesome.com/license/free).

### Installation

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
