# Dependency compatibility patch

The website overrides `image-size` to the patched upstream 2.0.3 release. Docusaurus 2.4.3 still calls its removed file/callback API, so `patch-image-size.cjs` switches that one pinned consumer to `imageSizeFromFile`, the public Promise-based API. It runs during website postinstall; scripts-disabled installs must run it explicitly before building.

The patch checks the consumer version and exact import/call before changing the installed file, and is idempotent. Unexpected source or versions fail rather than silently dropping width/height metadata. It modifies only local `node_modules`, not upstream sources or installed package versions. `yarn test:image-size` covers real Docusaurus image metadata and bounds malformed ICNS/JXL/HEIF probes to three seconds.

Source advisories: [ICNS](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr) and [JXL/HEIF](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq).

Remove this compatibility patch when Docusaurus natively supports the fixed parser API, after keeping the image-dimension regression green. A successful site build alone is insufficient because the old transformer catches image-size errors.

## Released merge replacement

The website resolves transitive `deepmerge` requests to the real published [`@fastify/deepmerge@3.2.4`](https://github.com/fastify/deepmerge), using a Yarn npm alias. Its package identity, version, archive and integrity are Fastify's; no version is fabricated and no copy of the vulnerable upstream `deepmerge` implementation remains. The scoped replacement rejects reserved prototype keys when cloning and merging objects.

The pinned direct consumers are `fork-ts-checker-webpack-plugin@6.5.3` and `@svgr/plugin-svgo@6.5.1`. Only the former calls deepmerge in its shipped JavaScript. Fastify exports a factory rather than the previous merge function, so `patch-merge-consumer.cjs` initializes that factory with enumerable-symbol support in the checker's one import. This postinstall compatibility patch checks both package identities/versions and the exact consumer source, is idempotent, and fails on unexpected source instead of silently supplying a function as configuration. Scripts-disabled installs must run it explicitly before building.

Fastify uses its own factory options and variadic `all` API, not the old library's per-call `arrayMerge`, `clone`, `customMerge` or array-argument `all` API. The audited consumer uses only the plain two-object call. Tests retain the prototype-poisoning fixtures and ordinary object/array/symbol controls, exercise Fastify's corresponding array/reference options, verify the real package identity and instantiate the actual webpack checker. Arbitrary callbacks or already-poisoned custom objects are not claimed to be universally sanitized.

[GHSA-gcqx-w48m-w2h4](https://github.com/advisories/GHSA-gcqx-w48m-w2h4) still has no patched release of the original package. This is a dependency replacement, not an upstream release fix or an alert dismissal. After landing, GitHub must refresh its dependency graph before its hosted alert status can be confirmed. The website CI job runs `yarn test:deepmerge` after its build.
